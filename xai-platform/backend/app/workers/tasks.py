import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from app.workers.celery_app import celery_app
from app.services.model_loader_service import ModelLoaderService
from app.services.lime_service import LIMEService
from app.db.mongo import get_db, storage
from app.db.repositories.prediction_repository import PredictionRepository
from datetime import datetime
from bson import ObjectId
import pickle
import joblib

@celery_app.task(bind=True, name="compute_shap_values")
def compute_shap_values(self, prediction_id: str, model_id: str) -> Dict[str, Any]:
    """
    Async task to compute SHAP values for a prediction.
    This can be expensive for large datasets, so it's run in background.
    """
    try:
        self.update_state(state="PROGRESS", meta={"status": "loading model", "progress": 10})

        # Get database connection (Celery task needs to establish its own connection)
        import asyncio
        async def async_task():
            # Reconnect database in worker
            from app.db.mongo import connect_db
            await connect_db()

            db = await get_db()
            # Get prediction record
            prediction = await db.predictions.find_one({"_id": ObjectId(prediction_id)})
            if not prediction:
                raise ValueError(f"Prediction {prediction_id} not found")

            # Get model record
            model = await db.models.find_one({"_id": ObjectId(model_id)})
            if not model:
                raise ValueError(f"Model {model_id} not found")

            self.update_state(state="PROGRESS", meta={"status": "loading model file", "progress": 30})

            # Load model from storage
            model_obj, framework = await ModelLoaderService.load_model(model["file_path"])

            # Prepare input data
            input_data = pd.DataFrame([prediction["input_data"]])

            # Map framework to SHAP explainer type
            self.update_state(state="PROGRESS", meta={"status": "computing SHAP values", "progress": 50})

            # Load background data if needed for SHAP (e.g. KernelExplainer)
            background_data = None
            if model.get("background_data_path"):
                bg_bytes = await storage.download_file(model["background_data_path"])
                background_data = pd.read_csv(pd.io.common.BytesIO(bg_bytes))
            else:
                # Fallback: build background from recent predictions for the same model
                # to avoid degenerate all-zero SHAP values with single-row background.
                recent_inputs = []
                model_id_filters = [{"model_id": model_id}]
                if ObjectId.is_valid(model_id):
                    model_id_filters.append({"model_id": ObjectId(model_id)})

                cursor = db.predictions.find(
                    {"$or": model_id_filters},
                    {"input_data": 1, "_id": 0}
                ).sort("created_at", -1).limit(200)

                async for row in cursor:
                    inp = row.get("input_data")
                    if isinstance(inp, dict):
                        recent_inputs.append(inp)

                if len(recent_inputs) >= 2:
                    background_data = pd.DataFrame(recent_inputs)
                    # Keep column order aligned with current input.
                    background_data = background_data.reindex(columns=input_data.columns, fill_value=np.nan)

            shap_values, expected_value = _compute_shap(model_obj, framework, input_data, background_data)

            # Normalize SHAP values for frontend (ensure 2D array with one row)
            shap_values_norm, expected_value_norm = _normalize_shap_local(
                shap_values, expected_value, prediction.get("prediction"), input_data
            )

            self.update_state(state="PROGRESS", meta={"status": "finalizing", "progress": 90})

            # Save explanation to database
            explanation_doc = {
                "prediction_id": prediction_id,
                "model_id": model_id,
                "method": "shap",
                "explanation_type": "local",
                "shap_values": shap_values_norm,
                "expected_value": expected_value_norm,
                "feature_names": list(input_data.columns) if hasattr(input_data, 'columns') else [],
                "nl_explanation": None,  # TODO: Add NLG service
                "task_id": self.request.id,
                "task_status": "complete",
                "created_at": datetime.utcnow()
            }

            result = await db.explanations.insert_one(explanation_doc)
            explanation_id = str(result.inserted_id)

            self.update_state(state="SUCCESS", meta={"status": "complete", "explanation_id": explanation_id, "progress": 100})

            return {"explanation_id": explanation_id, "status": "complete"}

        return asyncio.run(async_task())

    except Exception:
        raise

@celery_app.task(bind=True, name="compute_global_shap")
def compute_global_shap(self, model_id: str, background_data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Async task to compute global SHAP values across a dataset.
    """
    try:
        self.update_state(state="PROGRESS", meta={"status": "loading model", "progress": 10})

        import asyncio
        async def async_task():
            from app.db.mongo import connect_db
            await connect_db()

            db = await get_db()

            # Get model record
            model = await db.models.find_one({"_id": ObjectId(model_id)})
            if not model:
                raise ValueError(f"Model {model_id} not found")

            self.update_state(state="PROGRESS", meta={"status": "loading model file", "progress": 30})

            # Load model
            model_obj, framework = await ModelLoaderService.load_model(model["file_path"])

            # Load background data (provided by user)
            background_data = None
            if background_data_path:
                bg_bytes = await storage.download_file(background_data_path)
                background_data = pd.read_csv(pd.io.common.BytesIO(bg_bytes))

            self.update_state(state="PROGRESS", meta={"status": "computing global SHAP", "progress": 50})

            # Compute SHAP values for all background data
            if background_data is not None:
                input_data = background_data
            else:
                # If no background data, sample from model's training data if available
                raise ValueError("Background data is required for global SHAP computation")

            shap_values, expected_value = _compute_shap(model_obj, framework, input_data, background_data)

            self.update_state(state="PROGRESS", meta={"status": "calculating feature importance", "progress": 80})

            # Calculate global importance
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = [
                {"feature": name, "importance": float(value)}
                for name, value in zip(input_data.columns, mean_abs_shap)
            ]

            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

            self.update_state(state="PROGRESS", meta={"status": "saving results", "progress": 100})

            # Save global explanation
            explanation_doc = {
                "model_id": model_id,
                "method": "shap",
                "explanation_type": "global",
                "shap_values": shap_values.tolist(),
                "expected_value": expected_value.tolist() if isinstance(expected_value, np.ndarray) else float(expected_value),
                "feature_names": list(input_data.columns),
                "global_importance": feature_importance,
                "task_id": self.request.id,
                "task_status": "complete",
                "created_at": datetime.utcnow()
            }

            result = await db.explanations.insert_one(explanation_doc)
            explanation_id = str(result.inserted_id)

            return {"explanation_id": explanation_id, "status": "complete", "feature_importance": feature_importance}

        return asyncio.run(async_task())

    except Exception:
        raise

@celery_app.task(bind=True, name="compute_lime_values")
def compute_lime_values(self, prediction_id: str, model_id: str, num_features: int = 10) -> Dict[str, Any]:
    """
    Async task to compute LIME explanation for a prediction.
    """
    try:
        self.update_state(state="PROGRESS", meta={"status": "loading model", "progress": 10})

        import asyncio
        async def async_task():
            from app.db.mongo import connect_db
            await connect_db()

            db = await get_db()

            # Get prediction record
            prediction = await db.predictions.find_one({"_id": ObjectId(prediction_id)})
            if not prediction:
                raise ValueError(f"Prediction {prediction_id} not found")

            # Get model record
            model = await db.models.find_one({"_id": ObjectId(model_id)})
            if not model:
                raise ValueError(f"Model {model_id} not found")

            self.update_state(state="PROGRESS", meta={"status": "loading model file", "progress": 30})

            # Load model from storage
            model_obj, framework = await ModelLoaderService.load_model(model["file_path"])

            # Get training data for LIME background
            # Use sample from prediction data or stored background data
            if model.get("background_data_path"):
                bg_bytes = await storage.download_file(model["background_data_path"])
                training_df = pd.read_csv(pd.io.common.BytesIO(bg_bytes))
            else:
                # Use prediction data as minimal background (not ideal but fallback)
                training_df = pd.DataFrame([prediction["input_data"]])
                # Add some noise perturbations for LIME background
                training_df = pd.concat([training_df] * 100, ignore_index=True)

            # Determine mode
            mode = "classification" if model.get("task_type") in ["classification", "binary_classification", "multiclass_classification"] else "regression"

            self.update_state(state="PROGRESS", meta={"status": "creating LIME explainer", "progress": 50})

            # Create explainer
            explainer = LIMEService.create_explainer(
                model_obj,
                framework,
                training_df,
                list(training_df.columns),
                mode=mode
            )

            # Prepare input instance
            input_df = pd.DataFrame([prediction["input_data"]])
            input_values = input_df.values[0]

            self.update_state(state="PROGRESS", meta={"status": "computing LIME values", "progress": 70})

            # Compute LIME explanation
            lime_data = LIMEService.explain_instance(
                explainer,
                model_obj,
                input_values,
                num_features=num_features
            )

            self.update_state(state="PROGRESS", meta={"status": "saving results", "progress": 90})

            # Save explanation to database
            explanation_doc = {
                "prediction_id": prediction_id,
                "model_id": model_id,
                "method": "lime",
                "explanation_type": "local",
                "lime_weights": lime_data.get("list_of_contributions", []),
                "lime_intercept": lime_data.get("intercept"),
                "lime_local_pred": lime_data.get("local_pred"),
                "feature_names": list(input_df.columns),
                "nl_explanation": None,
                "task_id": self.request.id,
                "task_status": "complete",
                "created_at": datetime.utcnow()
            }

            result = await db.explanations.insert_one(explanation_doc)
            explanation_id = str(result.inserted_id)

            self.update_state(state="SUCCESS", meta={"status": "complete", "explanation_id": explanation_id, "progress": 100})

            return {"explanation_id": explanation_id, "status": "complete"}

        return asyncio.run(async_task())

    except Exception:
        raise

@celery_app.task(bind=True, name="compute_global_lime")
def compute_global_lime(self, model_id: str, background_data_path: Optional[str] = None, num_features: int = 10) -> Dict[str, Any]:
    """
    Async task to compute global LIME explanation (aggregated across samples).
    """
    try:
        self.update_state(state="PROGRESS", meta={"status": "loading model", "progress": 10})

        import asyncio
        async def async_task():
            from app.db.mongo import connect_db
            await connect_db()

            db = await get_db()

            # Get model record
            model = await db.models.find_one({"_id": ObjectId(model_id)})
            if not model:
                raise ValueError(f"Model {model_id} not found")

            self.update_state(state="PROGRESS", meta={"status": "loading model file", "progress": 30})

            # Load model
            model_obj, framework = await ModelLoaderService.load_model(model["file_path"])

            # Load background data
            if not background_data_path:
                if model.get("background_data_path"):
                    background_data_path = model["background_data_path"]
                else:
                    raise ValueError("Background data required for global LIME computation")

            bg_bytes = await storage.download_file(background_data_path)
            background_df = pd.read_csv(pd.io.common.BytesIO(bg_bytes))

            self.update_state(state="PROGRESS", meta={"status": "creating LIME explainer", "progress": 50})

            # Determine mode
            mode = "classification" if model.get("task_type") in ["classification", "binary_classification", "multiclass_classification"] else "regression"

            # Create explainer
            explainer = LIMEService.create_explainer(
                model_obj,
                framework,
                background_df,
                list(background_df.columns),
                mode=mode
            )

            self.update_state(state="PROGRESS", meta={"status": "computing global LIME", "progress": 60})

            # Compute aggregated LIME importance
            lime_global = LIMEService.explain_global(
                explainer,
                model_obj,
                background_df,
                num_features=num_features
            )

            self.update_state(state="PROGRESS", meta={"status": "saving results", "progress": 100})

            # Save global explanation
            explanation_doc = {
                "model_id": model_id,
                "method": "lime",
                "explanation_type": "global",
                "lime_weights": None,  # Individual weights not stored for global
                "lime_global_importance": lime_global["feature_importance"],
                "feature_names": list(background_df.columns),
                # Keep SHAP fields as None to avoid mixing
                "shap_values": None,
                "expected_value": None,
                "global_importance": None,
                "task_id": self.request.id,
                "task_status": "complete",
                "created_at": datetime.utcnow()
            }

            result = await db.explanations.insert_one(explanation_doc)
            explanation_id = str(result.inserted_id)

            return {"explanation_id": explanation_id, "status": "complete", "feature_importance": lime_global["feature_importance"]}

        return asyncio.run(async_task())

    except Exception:
        raise

def _compute_shap(model_obj, framework: str, input_data: pd.DataFrame, background_data: Optional[pd.DataFrame] = None) -> tuple:
    """
    Compute SHAP values. Handles pipelines, tree models, linear models, etc.
    Returns (shap_values, expected_value).
    """
    import shap
    import numpy as np
    from sklearn.pipeline import Pipeline

    try:
        # Check if model is a sklearn Pipeline
        if isinstance(model_obj, Pipeline):
            # For pipelines, use KernelExplainer with the full pipeline
            # This respects all preprocessing steps
            if background_data is not None and len(background_data) > 0:
                bg = background_data if len(background_data) <= 100 else background_data.iloc[:100]
            else:
                bg = input_data

            # Use the pipeline's predict method
            predict_fn = model_obj.predict

            def _predict_with_columns(values):
                if isinstance(values, np.ndarray):
                    df = pd.DataFrame(values, columns=input_data.columns)
                else:
                    df = values
                preds = predict_fn(df)
                # Ensure return is numpy array
                if isinstance(preds, np.ndarray):
                    return preds
                elif hasattr(preds, 'toarray'):
                    return preds.toarray()
                else:
                    return np.array(preds)

            explainer = shap.KernelExplainer(_predict_with_columns, bg.values)
            shap_values = explainer.shap_values(input_data.values)
            expected_value = explainer.expected_value

        else:
            # For non-pipeline models, try TreeExplainer first for tree-based models
            try:
                explainer = shap.TreeExplainer(model_obj)
                shap_values = explainer.shap_values(input_data)
                expected_value = explainer.expected_value
            except Exception:
                # Fallback to KernelExplainer
                if background_data is not None and len(background_data) > 0:
                    bg = background_data if len(background_data) <= 100 else background_data.iloc[:100]
                else:
                    # Generate synthetic background if none provided
                    bg = input_data

                predict_fn = model_obj.predict_proba if hasattr(model_obj, "predict_proba") else model_obj.predict

                def _predict_with_columns(values):
                    if isinstance(values, np.ndarray):
                        df = pd.DataFrame(values, columns=input_data.columns)
                    else:
                        df = values
                    return predict_fn(df)

                explainer = shap.KernelExplainer(_predict_with_columns, bg.values)
                shap_values = explainer.shap_values(input_data.values)
                expected_value = explainer.expected_value

        return shap_values, expected_value

    except Exception as e:
        raise ValueError(f"SHAP computation failed: {str(e)}")


def _normalize_shap_local(
    shap_values: Any,
    expected_value: Any,
    prediction_value: Any,
    input_data: pd.DataFrame,
) -> tuple[list[list[float]], float]:
    """
    Normalize SHAP output for local explanations.

    Returns:
    - shap values as a single-row 2D list: [[f1, f2, ...]]
    - expected value as a scalar float
    """
    feature_count = len(input_data.columns) if hasattr(input_data, "columns") else int(np.asarray(input_data).shape[-1])

    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _get_class_index(n_classes: int) -> int:
        # For binary classification, index 1 is typically the positive class.
        if n_classes == 2:
            return 1
        try:
            idx = int(np.asarray(prediction_value).item())
            if 0 <= idx < n_classes:
                return idx
        except Exception:
            pass
        return 0

    def _fit_feature_size(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=float).reshape(-1)
        if vec.size < feature_count:
            vec = np.pad(vec, (0, feature_count - vec.size), mode="constant")
        elif vec.size > feature_count:
            vec = vec[:feature_count]
        return vec

    # Normalize SHAP values to one feature vector for the current instance.
    if isinstance(shap_values, list):
        if len(shap_values) == 0:
            local_vec = np.zeros(feature_count, dtype=float)
        else:
            class_idx = _get_class_index(len(shap_values))
            arr = np.asarray(shap_values[class_idx], dtype=float)
            local_vec = arr[0] if arr.ndim > 1 else arr
    else:
        arr = np.asarray(shap_values, dtype=float)
        if arr.ndim == 1:
            local_vec = arr
        elif arr.ndim == 2:
            local_vec = arr[0]
        elif arr.ndim >= 3:
            # Common shape for multiclass can be (classes, samples, features).
            class_idx = _get_class_index(arr.shape[0])
            class_arr = np.asarray(arr[class_idx], dtype=float)
            local_vec = class_arr[0] if class_arr.ndim > 1 else class_arr
        else:
            local_vec = np.zeros(feature_count, dtype=float)

    local_vec = _fit_feature_size(local_vec)

    # Normalize expected value to scalar.
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        ev_arr = np.asarray(expected_value, dtype=float).reshape(-1)
        if ev_arr.size == 0:
            expected_scalar = 0.0
        elif ev_arr.size == 1:
            expected_scalar = _safe_float(ev_arr[0])
        else:
            expected_scalar = _safe_float(ev_arr[_get_class_index(ev_arr.size)], _safe_float(ev_arr[0]))
    else:
        expected_scalar = _safe_float(expected_value)

    return [local_vec.tolist()], expected_scalar