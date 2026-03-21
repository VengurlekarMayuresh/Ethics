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
from sklearn.pipeline import Pipeline
import re

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

            shap_values, expected_value, feature_names = _compute_shap(model_obj, framework, input_data, background_data)

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
                "feature_names": feature_names,
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

            shap_values, expected_value, feature_names = _compute_shap(model_obj, framework, input_data, background_data)

            self.update_state(state="PROGRESS", meta={"status": "calculating feature importance", "progress": 80})

            # Handle classification case where shap_values may be a list (one array per class)
            if isinstance(shap_values, list):
                # For binary classification, use the positive class (index 1)
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                else:
                    # For multi-class, we could take mean across classes or use first class.
                    # Here we'll take the mean absolute value across classes.
                    shap_values = np.mean([np.abs(arr) for arr in shap_values], axis=0)

            # Calculate global importance
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = [
                {"feature": name, "importance": float(value)}
                for name, value in zip(feature_names, mean_abs_shap)
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
                "feature_names": feature_names,
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

            # Get feature names from explainer (may be preprocessed if pipeline)
            explainer_feature_names = getattr(explainer, 'feature_names', list(training_df.columns))

            # Prepare input instance
            input_df = pd.DataFrame([prediction["input_data"]])
            input_values = input_df.values[0]

            # For pipelines, preprocess the input instance to match the explainer's feature space
            instance_to_explain = input_values
            if isinstance(model_obj, Pipeline):
                # Find preprocessor
                preprocessor = None
                for step_name, step_obj in model_obj.steps:
                    if hasattr(step_obj, 'transform'):
                        preprocessor = step_obj
                        break
                if preprocessor is not None:
                    # Preprocess the input
                    processed_input = preprocessor.transform(input_df)
                    if hasattr(processed_input, 'toarray'):
                        processed_input = processed_input.toarray()
                    instance_to_explain = np.asarray(processed_input, dtype=float)[0]

            self.update_state(state="PROGRESS", meta={"status": "computing LIME values", "progress": 70})

            # Compute LIME explanation
            lime_data = LIMEService.explain_instance(
                explainer,
                model_obj,
                instance_to_explain,
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
                "feature_names": explainer_feature_names,
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

            # Create explainer (will preprocess background data if pipeline)
            explainer = LIMEService.create_explainer(
                model_obj,
                framework,
                background_df,
                list(background_df.columns),
                mode=mode
            )

            # Get feature names from explainer (may be preprocessed if pipeline)
            explainer_feature_names = getattr(explainer, 'feature_names', list(background_df.columns))

            self.update_state(state="PROGRESS", meta={"status": "computing global LIME", "progress": 60})

            # For global LIME, the samples passed to explain_global must be in the same space as the explainer.
            # If pipeline, preprocess background_df; otherwise use as-is.
            samples_for_explanation = background_df
            if isinstance(model_obj, Pipeline):
                # Preprocess to numeric space
                preprocessor = None
                for step_name, step_obj in model_obj.steps:
                    if hasattr(step_obj, 'transform'):
                        preprocessor = step_obj
                        break
                if preprocessor is not None:
                    processed_bg = preprocessor.transform(background_df)
                    if hasattr(processed_bg, 'toarray'):
                        processed_bg = processed_bg.toarray()
                    # Convert to DataFrame with feature names from explainer
                    samples_for_explanation = pd.DataFrame(processed_bg, columns=explainer_feature_names)

            # Compute aggregated LIME importance
            lime_global = LIMEService.explain_global(
                explainer,
                model_obj,
                samples_for_explanation,
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
                "feature_names": explainer_feature_names,
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
        expected_columns = list(input_data.columns)

        def _prepare_background(df: Optional[pd.DataFrame]) -> pd.DataFrame:
            """Ensure SHAP background data matches model input columns, tolerating name-style mismatches."""
            if df is None or len(df) == 0:
                return input_data[expected_columns]

            bg_df = df.copy()

            def _norm(name: Any) -> str:
                # Normalize aggressively to handle case, spaces, dashes, punctuation, BOM-like noise.
                text = str(name).strip().lower()
                return re.sub(r"[^a-z0-9]+", "", text)

            # Build normalized lookup for background columns.
            bg_norm_to_actual = {}
            for col in bg_df.columns:
                norm_col = _norm(col)
                if norm_col not in bg_norm_to_actual:
                    bg_norm_to_actual[norm_col] = col

            # Resolve expected columns to actual background columns.
            resolved = {}
            for col in expected_columns:
                if col in bg_df.columns:
                    resolved[col] = col
                    continue
                candidate = bg_norm_to_actual.get(_norm(col))
                if candidate is not None:
                    resolved[col] = candidate

            missing = [c for c in expected_columns if c not in resolved]
            if missing:
                # Safe fallback: synthesize missing columns from the current input row
                # rather than mapping by position (which can mis-map IDs/text fields).
                input_row = input_data.iloc[0].to_dict()
                for col in missing:
                    if col in input_row:
                        bg_df[col] = input_row[col]
                        resolved[col] = col

            missing_after_fill = [c for c in expected_columns if c not in resolved]
            if missing_after_fill:
                raise ValueError(
                    f"Background data missing required columns: {missing_after_fill}. "
                    f"Expected columns: {expected_columns}"
                )

            # Rename resolved columns to expected names, drop extras, and keep input order.
            selected_actual_cols = [resolved[c] for c in expected_columns]
            bg_df = bg_df[selected_actual_cols].copy()
            bg_df.columns = expected_columns
            return bg_df

        # Initialize with input data columns as default feature names
        final_feature_names = expected_columns

        # Check if model is a sklearn Pipeline
        if isinstance(model_obj, Pipeline):
            # For pipelines, we need to work in the preprocessed feature space
            # because the model expects numeric inputs after preprocessing

            # Prepare background data (raw)
            if background_data is not None and len(background_data) > 0:
                bg_raw = background_data if len(background_data) <= 100 else background_data.iloc[:100]
                bg = _prepare_background(bg_raw)
            else:
                bg = _prepare_background(input_data)

            # IMPORTANT: Preprocess the background data to numeric space
            # Find the preprocessing step in the pipeline
            preprocessor = None
            for step_name, step_obj in model_obj.steps:
                if hasattr(step_obj, 'transform'):
                    preprocessor = step_obj
                    break

            if preprocessor is not None:
                # Transform raw background to preprocessed numeric features
                bg_processed = preprocessor.transform(bg)
                if hasattr(bg_processed, 'toarray'):  # sparse matrix
                    bg_processed = bg_processed.toarray()
                bg_numeric = np.asarray(bg_processed, dtype=float)

                # Try to get feature names from the preprocessor
                if hasattr(preprocessor, 'get_feature_names_out'):
                    try:
                        raw_names = preprocessor.get_feature_names_out()
                        cleaned_names = []
                        for name in raw_names:
                            if isinstance(name, bytes):
                                name = name.decode('utf-8')
                            if '__' in name:
                                name = name.split('__', 1)[1]
                            cleaned_names.append(name)
                        final_feature_names = cleaned_names
                    except Exception:
                        # Fallback to generic feature names if extraction fails
                        final_feature_names = [f"feature_{i}" for i in range(bg_numeric.shape[1])]
                else:
                    # No get_feature_names_out, use generic names
                    final_feature_names = [f"feature_{i}" for i in range(bg_numeric.shape[1])]
            else:
                # No clear preprocessor, fall back to raw numeric (may fail if strings present)
                bg_numeric = bg.values.astype(float)
                # Keep default final_feature_names = expected_columns

            # Get the final estimator for prediction on preprocessed data
            final_estimator = model_obj.steps[-1][1]

            # Predict function works directly on preprocessed numeric data
            if hasattr(final_estimator, 'predict_proba'):
                predict_fn = final_estimator.predict_proba
            else:
                predict_fn = final_estimator.predict

            def _predict_preprocessed(values):
                # values is already in preprocessed space (numeric array)
                return predict_fn(values)

            # Create SHAP explainer in preprocessed feature space
            explainer = shap.KernelExplainer(_predict_preprocessed, bg_numeric)

            # Also preprocess the input data
            if preprocessor is not None:
                input_processed = preprocessor.transform(input_data)
                if hasattr(input_processed, 'toarray'):
                    input_processed = input_processed.toarray()
                input_numeric = np.asarray(input_processed, dtype=float)
            else:
                input_numeric = input_data.values.astype(float)

            shap_values = explainer.shap_values(input_numeric)
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
                    bg_raw = background_data if len(background_data) <= 100 else background_data.iloc[:100]
                    bg = _prepare_background(bg_raw)
                else:
                    # Generate synthetic background if none provided
                    bg = _prepare_background(input_data)

                predict_fn = model_obj.predict_proba if hasattr(model_obj, "predict_proba") else model_obj.predict

                def _predict_with_columns(values):
                    if isinstance(values, np.ndarray):
                        df = pd.DataFrame(values, columns=expected_columns)
                    else:
                        df = values[expected_columns] if isinstance(values, pd.DataFrame) else values
                    return predict_fn(df)

                explainer = shap.KernelExplainer(_predict_with_columns, bg.values)
                shap_values = explainer.shap_values(input_data.values)
                expected_value = explainer.expected_value

        return shap_values, expected_value, final_feature_names

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

    # Extract SHAP values for the local instance as a 1D array
    if isinstance(shap_values, list):
        if len(shap_values) == 0:
            # Empty SHAP values; return empty array
            local_vec = np.array([], dtype=float)
        else:
            class_idx = _get_class_index(len(shap_values))
            arr = np.asarray(shap_values[class_idx], dtype=float)
            # arr could be shape (1, n_features) or (n_features,)
            if arr.ndim > 1:
                local_vec = arr[0]
            else:
                local_vec = arr
    else:
        arr = np.asarray(shap_values, dtype=float)
        if arr.ndim == 0:
            local_vec = np.array([float(arr)])
        elif arr.ndim == 1:
            local_vec = arr
        elif arr.ndim == 2:
            # Shape (n_instances, n_features); we want first instance
            local_vec = arr[0]
        elif arr.ndim >= 3:
            # For 3D arrays, we need to handle two common SHAP output shapes:
            # (classes, instances, features) from TreeExplainer or
            # (instances, features, classes) from KernelExplainer
            # Since we're explaining a single instance, the instance dimension should be 1.

            # Check if shape looks like (instances, features, classes)
            if arr.shape[0] == 1 and arr.ndim == 3 and arr.shape[2] > 1:
                # Shape: (1, features, classes) - KernelExplainer output
                class_idx = _get_class_index(arr.shape[2])
                local_vec = arr[0, :, class_idx]
            else:
                # Assume shape: (classes, instances, features) - TreeExplainer output
                class_idx = _get_class_index(arr.shape[0])
                class_arr = np.asarray(arr[class_idx], dtype=float)
                if class_arr.ndim > 1:
                    local_vec = class_arr[0]
                else:
                    local_vec = class_arr
        else:
            local_vec = np.array([], dtype=float)

    # Ensure 1D array
    local_vec = np.asarray(local_vec, dtype=float).reshape(-1)

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