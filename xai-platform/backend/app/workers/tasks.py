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

            shap_values, expected_value = _compute_shap(model_obj, framework, input_data, background_data)

            self.update_state(state="PROGRESS", meta={"status": "finalizing", "progress": 90})

            # Save explanation to database
            explanation_doc = {
                "prediction_id": prediction_id,
                "model_id": model_id,
                "method": "shap",
                "explanation_type": "local",
                "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                "expected_value": expected_value if isinstance(expected_value, float) else expected_value.tolist(),
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

    except Exception as e:
        self.update_state(state="FAILURE", meta={"status": "failed", "error": str(e)})
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

    except Exception as e:
        self.update_state(state="FAILURE", meta={"status": "failed", "error": str(e)})
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

    except Exception as e:
        self.update_state(state="FAILURE", meta={"status": "failed", "error": str(e)})
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

    except Exception as e:
        self.update_state(state="FAILURE", meta={"status": "failed", "error": str(e)})
        raise

def _compute_shap(model_obj, framework: str, input_data: pd.DataFrame, background_data: Optional[pd.DataFrame] = None) -> tuple:
    """
    Compute SHAP values based on framework and model type.
    Returns (shap_values, expected_value).
    """
    import shap
    import numpy as np

    try:
        if framework in ["sklearn", "xgboost", "lightgbm"]:
            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(input_data)
            expected_value = explainer.expected_value

        elif framework == "linear":
            explainer = shap.LinearExplainer(model_obj, input_data)
            shap_values = explainer.shap_values(input_data)
            expected_value = explainer.expected_value

        else:
            # KernelExplainer for neural nets and other models
            if background_data is None:
                raise ValueError("Background data required for KernelExplainer")

            # Use first 100 samples as background
            background_sample = background_data if len(background_data) <= 100 else background_data.iloc[:100]

            explainer = shap.KernelExplainer(model_obj.predict_proba if hasattr(model_obj, 'predict_proba') else model_obj.predict, background_sample)
            shap_values = explainer.shap_values(input_data)
            expected_value = explainer.expected_value

        return shap_values, expected_value

    except Exception as e:
        raise ValueError(f"SHAP computation failed: {str(e)}")