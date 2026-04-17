import shap
import numpy as np
import pandas as pd
import os
import logging
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
import traceback
import importlib
import io

# Maximum number of background samples to use for global SHAP/LIME explanations
# Prevents extremely long computation times with large datasets
MAX_GLOBAL_SHAP_SAMPLES = int(os.getenv("MAX_GLOBAL_SHAP_SAMPLES", "200"))
import re

# Configure logger
logger = logging.getLogger(__name__)
# Ensure logger outputs to console if no handlers configured
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Default to INFO; can be overridden by env var LOGLEVEL

def _align_background_data(background_df: pd.DataFrame, model_doc: dict) -> pd.DataFrame:
    """
    Align background dataset to match the model's expected feature schema.
    - Renames columns to match expected feature names (case-insensitive, normalized)
    - Drops extra columns not in schema
    - Ensures all expected features are present
    - Reorders columns to match feature schema order
    - Validates no missing values
    Returns aligned DataFrame.
    """
    feature_schema = model_doc.get('feature_schema', [])
    if not feature_schema:
        raise ValueError("Model does not have a feature schema defined. Cannot align background data.")

    expected_features = [fs['name'] for fs in feature_schema]

    # Normalization function for fuzzy matching
    def normalize(name):
        return re.sub(r'[^a-z0-9]', '', str(name).strip().lower())

    # Build mapping from normalized background column names to actual names
    bg_norm_to_actual = {}
    for col in background_df.columns:
        norm = normalize(col)
        if norm not in bg_norm_to_actual:
            bg_norm_to_actual[norm] = col

    # Map each expected feature to a background column
    column_mapping = {}
    for exp_feat in expected_features:
        # Exact match
        if exp_feat in background_df.columns:
            column_mapping[exp_feat] = exp_feat
            continue
        # Normalized match
        norm_exp = normalize(exp_feat)
        if norm_exp in bg_norm_to_actual:
            column_mapping[exp_feat] = bg_norm_to_actual[norm_exp]
            continue
        # Not found
        raise ValueError(
            f"Background data missing required feature: '{exp_feat}'. "
            f"Available columns: {list(background_df.columns)}"
        )

    # Build aligned DataFrame with only expected features, in expected order
    aligned = pd.DataFrame()
    for exp_feat in expected_features:
        actual_col = column_mapping[exp_feat]
        aligned[exp_feat] = background_df[actual_col].copy()

    # Check for missing values
    if aligned.isna().any().any():
        nan_cols = aligned.columns[aligned.isna().any()].tolist()
        raise ValueError(
            f"Background data contains missing values in columns: {nan_cols}. "
            "Please impute or remove rows with missing values before uploading."
        )

    return aligned

@celery_app.task(bind=True, name="compute_shap_values")
def compute_shap_values(self, prediction_id: str, model_id: str) -> Dict[str, Any]:
    """
    Async task to compute SHAP values for a prediction.
    This can be expensive for large datasets, so it's run in background.
    """
    try:
        logger.info(f"[compute_shap_values] START: prediction_id={prediction_id}, model_id={model_id}")
        self.update_state(state="PROGRESS", meta={"status": "loading model", "progress": 10})

        # Get database connection (Celery task needs to establish its own connection)
        import asyncio
        async def async_task():
            try:
                # Reconnect database in worker
                from app.db.mongo import connect_db
                await connect_db()

                db = await get_db()
                # Get prediction record
                logger.info(f"[compute_shap_values] Fetching prediction {prediction_id}")
                prediction = await db.predictions.find_one({"_id": ObjectId(prediction_id)})
                if not prediction:
                    logger.error(f"[compute_shap_values] Prediction {prediction_id} not found")
                    raise ValueError(f"Prediction {prediction_id} not found")

                # Get model record
                logger.info(f"[compute_shap_values] Fetching model {model_id}")
                model = await db.models.find_one({"_id": ObjectId(model_id)})
                if not model:
                    logger.error(f"[compute_shap_values] Model {model_id} not found")
                    raise ValueError(f"Model {model_id} not found")

                # Validate model feature_schema
                feature_schema = model.get('feature_schema', [])
                if not feature_schema:
                    logger.error(f"[compute_shap_values] Model {model_id} has empty feature_schema")
                    raise ValueError("Model does not have a feature schema. Cannot compute SHAP.")

                logger.info(f"[compute_shap_values] Model feature_schema has {len(feature_schema)} features")
                expected_features = [fs['name'] for fs in feature_schema]
                logger.debug(f"[compute_shap_values] Expected features (first 10): {expected_features[:10]}")

                self.update_state(state="PROGRESS", meta={"status": "loading model file", "progress": 30})

                # Load model from storage
                logger.info(f"[compute_shap_values] Loading model from {model['file_path']}")
                model_obj, framework = await ModelLoaderService.load_model(model["file_path"])
                logger.info(f"[compute_shap_values] Model loaded: framework={framework}, model_type={type(model_obj).__name__}")

                # Prepare input data
                input_data = pd.DataFrame([prediction["input_data"]])
                logger.info(f"[compute_shap_values] Input data shape: {input_data.shape}, columns={list(input_data.columns)}")

                # Validate input against feature schema
                expected_features = [fs['name'] for fs in model.get('feature_schema', [])]
                missing = [f for f in expected_features if f not in input_data.columns]
                if missing:
                    logger.error(f"[compute_shap_values] Prediction input missing features: {missing[:5]}")
                    raise ValueError(f"Prediction input missing required features: {missing[:5]}. Expected: {expected_features[:10]}")
                # Check for NaN in input
                if input_data.isna().any().any():
                    nan_cols = input_data.columns[input_data.isna().any()].tolist()
                    logger.error(f"[compute_shap_values] Input contains NaN in columns: {nan_cols}")
                    raise ValueError(f"Prediction input contains NaN values in columns: {nan_cols}. Please provide valid values.")
                logger.info(f"[compute_shap_values] Input validated against feature schema")

                # Map framework to SHAP explainer type
                self.update_state(state="PROGRESS", meta={"status": "computing SHAP values", "progress": 50})

                # Load background data if needed for SHAP (e.g. KernelExplainer)
                background_data = None
                if model.get("background_data_path"):
                    logger.info(f"[compute_shap_values] Downloading background data from {model['background_data_path']}")
                    bg_bytes = await storage.download_file(model["background_data_path"])
                    background_data = pd.read_csv(pd.io.common.BytesIO(bg_bytes))
                    logger.info(f"[compute_shap_values] Background data loaded: shape={background_data.shape}")
                else:
                    # Fallback: build background from recent predictions for the same model
                    # to avoid degenerate all-zero SHAP values with single-row background.
                    logger.info(f"[compute_shap_values] No background_data_path, building background from recent predictions")
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

                    if len(recent_inputs) >= 5:
                        background_data = pd.DataFrame(recent_inputs)
                        # Keep column order aligned with current input.
                        background_data = background_data.reindex(columns=input_data.columns, fill_value=np.nan)
                        logger.info(f"[compute_shap_values] Built background from recent predictions: shape={background_data.shape}")
                    else:
                        # Fallback: create synthetic baseline if not enough history
                        logger.warning(f"[compute_shap_values] Not enough recent predictions ({len(recent_inputs)}), building synthetic baseline from schema")
                        
                        synthetic_rows = []
                        feature_schema = model.get("feature_schema", [])
                        
                        def get_baseline_val(fs):
                            # Prioritize schema-defined mean/median, else use zero for numeric
                            if fs.get("type") == "numeric":
                                if fs.get("mean") is not None: return float(fs["mean"])
                                if fs.get("mean_") is not None: return float(fs["mean_"]) # common typo in older dumps
                                return 0.0
                            else:
                                # Categorical: use mode if exists, else first option, else empty string
                                options = fs.get("options", [])
                                return options[0] if options else ""
                        
                        # Row 1: The "Average" person
                        medoid = {fs["name"]: get_baseline_val(fs) for fs in feature_schema if fs["name"] in input_data.columns}
                        synthetic_rows.append(medoid)
                        
                        # Row 2: Perturbed version of input data to ensure variety in every column
                        # This helps ensure KernelExplainer sees at least some distribution.
                        baseline_2 = medoid.copy()
                        for fs in feature_schema:
                            name = fs["name"]
                            if name not in input_data.columns: continue
                            val = input_data.iloc[0][name]
                            if fs.get("type") == "numeric":
                                # If input is same as baseline, add slight noise
                                if abs(val - baseline_2[name]) < 1e-6:
                                    baseline_2[name] = val + 1.0 # arbitrary shift
                            else:
                                options = fs.get("options", [])
                                if len(options) > 1 and val == baseline_2[name]:
                                    baseline_2[name] = options[1] # swap to second option
                        
                        synthetic_rows.append(baseline_2)
                        
                        # If we had some recent predictions but < 5, add them too
                        if recent_inputs:
                            synthetic_rows.extend(recent_inputs[:3])

                        background_data = pd.DataFrame(synthetic_rows)
                        background_data = background_data.reindex(columns=input_data.columns, fill_value=np.nan)
                        logger.info(f"[compute_shap_values] Generated synthetic background: shape={background_data.shape}")

                logger.info(f"[compute_shap_values] Calling _compute_shap: input_data.shape={input_data.shape}, background_data shape={background_data.shape if background_data is not None else 'None'}")
                shap_values, expected_value, feature_names = _compute_shap(model_obj, framework, input_data, background_data)
                logger.info(f"[compute_shap_values] _compute_shap returned: shap_values type={type(shap_values)}, expected_value type={type(expected_value)}, feature_names count={len(feature_names)}")

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

            except Exception as e:
                logger.error(f"[compute_shap_values] async_task failed: {e}", exc_info=True)
                raise

        return asyncio.run(async_task())

    except Exception:
        raise

@celery_app.task(bind=True, name="compute_global_shap")
def compute_global_shap(self, model_id: str, background_data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Async task to compute global SHAP values across a dataset.
    """
    try:
        logger.info(f"[compute_global_shap] START: model_id={model_id}, background_data_path={background_data_path}")
        self.update_state(state="PROGRESS", meta={"status": "loading model", "progress": 10})

        import asyncio
        async def async_task():
            from app.db.mongo import connect_db
            await connect_db()

            db = await get_db()

            # Get model record
            logger.debug(f"[compute_global_shap] Fetching model {model_id}")
            model = await db.models.find_one({"_id": ObjectId(model_id)})
            if not model:
                raise ValueError(f"Model {model_id} not found")

            self.update_state(state="PROGRESS", meta={"status": "loading model file", "progress": 30})

            # Load model
            logger.info(f"[compute_global_shap] Loading model from {model['file_path']}")
            model_obj, framework = await ModelLoaderService.load_model(model["file_path"])
            logger.info(f"[compute_global_shap] Model loaded: framework={framework}, model_type={type(model_obj).__name__}")

            # Load background data (provided by user)
            background_data = None
            if background_data_path:
                logger.info(f"[compute_global_shap] Downloading background data from {background_data_path}")
                bg_bytes = await storage.download_file(background_data_path)
                background_data = pd.read_csv(pd.io.common.BytesIO(bg_bytes))
                logger.info(f"[compute_global_shap] Background data loaded: shape={background_data.shape}, columns={list(background_data.columns)[:10]}")
                # Validate background data
                expected_features = [fs['name'] for fs in model.get('feature_schema', [])]
                missing_features = [f for f in expected_features if f not in background_data.columns]
                if missing_features:
                    logger.error(f"[compute_global_shap] Background data missing required features: {missing_features[:5]}")
                    raise ValueError(f"Background data missing required features: {missing_features[:5]}. Expected: {expected_features[:10]}")
                # Check for NaN values
                nan_cols = background_data.columns[background_data.isna().any()].tolist()
                if nan_cols:
                    logger.error(f"[compute_global_shap] Background data contains NaN values in columns: {nan_cols[:5]}")
                    raise ValueError(f"Background data contains NaN values in columns: {nan_cols[:5]}. Please impute or remove missing values.")
                logger.info(f"[compute_global_shap] Background data validated: all features present, no NaN values")
            else:
                logger.warning(f"[compute_global_shap] No background_data_path provided")

            self.update_state(state="PROGRESS", meta={"status": "computing global SHAP", "progress": 50})

            # Compute SHAP values for all background data
            if background_data is not None:
                input_data = background_data
            else:
                # If no background data, sample from model's training data if available
                raise ValueError("Background data is required for global SHAP computation")

            logger.info(f"[compute_global_shap] Calling _compute_shap: input_data.shape={input_data.shape}")
            shap_values, expected_value, feature_names = _compute_shap(model_obj, framework, input_data, background_data)
            logger.info(f"[compute_global_shap] _compute_shap returned: shap_values type={type(shap_values)}, feature_names count={len(feature_names)}")

            self.update_state(state="PROGRESS", meta={"status": "calculating feature importance", "progress": 80})

            # Normalize shap_values to 2D array (n_instances, n_features) for global importance
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # For classification, shap_values may have shape (instances, features, classes) or (classes, instances, features)
                # Use expected_value to detect class axis if possible
                class_axis = None
                if isinstance(expected_value, (np.ndarray, list, tuple)) and not isinstance(expected_value, str):
                    ev_arr = np.asarray(expected_value)
                    if ev_arr.ndim == 1 and ev_arr.size > 1:
                        for axis, size in enumerate(shap_values.shape):
                            if size == ev_arr.size:
                                class_axis = axis
                                break
                if class_axis is None:
                    # Heuristic: if last dimension is small (2 or 3), likely class axis
                    if shap_values.shape[2] <= 5 and shap_values.shape[1] > shap_values.shape[2]:
                        class_axis = 2
                    else:
                        class_axis = 0
                # Move class axis to front
                if class_axis != 0:
                    shap_values = np.moveaxis(shap_values, class_axis, 0)
                # Now shap_values.shape[0] is number of classes
                # For global importance, we can either:
                # - Use positive class (index 1) for binary classification
                # - Take mean absolute across classes for multi-class
                if shap_values.shape[0] >= 2:
                    # Binary classification: use positive class (index 1)
                    shap_values = shap_values[1]  # shape (instances, features) or still >2D?
                else:
                    shap_values = shap_values[0]
                # Ensure shap_values is 2D (instances, features)
                if shap_values.ndim > 2:
                    # If there are extra dimensions, average them out (unlikely)
                    # For safety, collapse all but the last dimension (features)
                    # Assuming last dim is features
                    shap_values = np.mean(shap_values, axis=tuple(range(shap_values.ndim - 1)))
                # Now shap_values should be 2D

            # Handle classification case where shap_values may be a list (one array per class)
            if isinstance(shap_values, list):
                # For binary classification, use the positive class (index 1)
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                else:
                    # For multi-class, we could take mean across classes or use first class.
                    # Here we'll take the mean absolute value across classes.
                    shap_values = np.mean([np.abs(arr) for arr in shap_values], axis=0)

            # Validate shap_values and feature_names
            shap_array = np.asarray(shap_values)
            if shap_array.ndim == 1:
                # Ensure 2D: single sample -> (1, n_features)
                shap_array = shap_array.reshape(1, -1)
            elif shap_array.ndim > 2:
                # Reduce to 2D: average across extra dimensions (should not happen)
                logger.warning(f"[compute_global_shap] shap_values has {shap_array.ndim} dimensions, reducing to 2D by averaging extra axes")
                # Average across all axes except the last one (features)
                axes_to_average = tuple(range(shap_array.ndim - 1))
                shap_array = np.mean(shap_array, axis=axes_to_average)
                # Ensure still 2D
                if shap_array.ndim == 1:
                    shap_array = shap_array.reshape(1, -1)

            n_samples, n_features = shap_array.shape[0], shap_array.shape[1] if shap_array.ndim >= 2 else 0

            if n_features == 0:
                raise ValueError("SHAP values have zero features - check model and background data compatibility")

            if len(feature_names) != n_features:
                logger.error(f"[compute_global_shap] Mismatch: shap_array features={n_features}, feature_names count={len(feature_names)}")
                # Attempt to reconcile: if feature_names > n_features, truncate; else pad with generic names
                if len(feature_names) > n_features:
                    feature_names = feature_names[:n_features]
                else:
                    feature_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), n_features)]
                logger.info(f"[compute_global_shap] Adjusted feature_names to {len(feature_names)}")

            # Compute global importance
            mean_abs_shap = np.abs(shap_array).mean(axis=0)
            # Validate: replace any NaN/Inf with 0 to prevent frontend issues
            if np.any(~np.isfinite(mean_abs_shap)):
                logger.warning(f"[compute_global_shap] mean_abs_shap contains non-finite values, replacing with 0")
                mean_abs_shap = np.nan_to_num(mean_abs_shap, nan=0.0, posinf=0.0, neginf=0.0)
            feature_importance = [
                {"feature": name, "importance": float(value)}
                for name, value in zip(feature_names, mean_abs_shap)
            ]

            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

            # Sanitize shap_array to remove NaN/Inf
            shap_array = np.nan_to_num(shap_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Convert shap_array back to list for storage
            shap_values = shap_array.tolist()

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

    except Exception as e:
        logger.error(f"[compute_global_shap] async_task failed: {e}", exc_info=True)
        raise

@celery_app.task(bind=True, name="compute_lime_values")
def compute_lime_values(self, prediction_id: str, model_id: str, num_features: int = 10) -> Dict[str, Any]:
    """
    Async task to compute LIME explanation for a prediction.
    """
    try:
        logger.info(f"[compute_lime_values] START: prediction_id={prediction_id}, model_id={model_id}, num_features={num_features}")
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

            logger.info(f"[compute_lime_values] Prediction and model fetched. task_type={model.get('task_type')}, background_data_path={model.get('background_data_path')}")
            self.update_state(state="PROGRESS", meta={"status": "loading model file", "progress": 30})

            # Load model from storage
            model_obj, framework = await ModelLoaderService.load_model(model["file_path"])
            logger.info(f"[compute_lime_values] Model loaded: framework={framework}, type={type(model_obj).__name__}")

            # Prepare input instance FIRST so we know which columns to use
            input_df = pd.DataFrame([prediction["input_data"]])
            input_cols = list(input_df.columns)
            logger.info(f"[compute_lime_values] Input instance columns ({len(input_cols)}): {input_cols}")

            # Get training data for LIME background
            if model.get("background_data_path"):
                logger.info(f"[compute_lime_values] Downloading background data from {model['background_data_path']}")
                bg_bytes = await storage.download_file(model["background_data_path"])
                training_df = pd.read_csv(pd.io.common.BytesIO(bg_bytes))
                logger.info(f"[compute_lime_values] Background data loaded: shape={training_df.shape}, columns={list(training_df.columns)}")
                # Align training_df to ONLY include columns present in the input instance.
                # Background CSV often includes target column and other extra columns that
                # inflate training_array shape and cause LIME scaler shape mismatch.
                common_cols = [c for c in input_cols if c in training_df.columns]
                missing_cols = [c for c in input_cols if c not in training_df.columns]
                if missing_cols:
                    logger.warning(f"[compute_lime_values] Input columns not in background CSV: {missing_cols}. Will use synthetic data for those.")
                if common_cols:
                    training_df = training_df[common_cols]
                else:
                    logger.warning(f"[compute_lime_values] No common columns between input and background CSV! Using input-aligned fallback.")
                    training_df = input_df  # will be expanded below
                logger.info(f"[compute_lime_values] Background data after column alignment: shape={training_df.shape}")
            else:
                # Fallback: build synthetic background from the single input row with Gaussian noise.
                logger.warning(f"[compute_lime_values] No background_data_path. Building synthetic background from input_data with Gaussian noise.")
                training_df = input_df.copy()

            # If training_df is too small, expand with Intelligent Noise
            if len(training_df) < 100:
                logger.info(f"[compute_lime_values] training_df only has {len(training_df)} rows — expanding with intelligent noise to 500 rows")
                base_df = training_df.copy()
                rows = [base_df]
                
                # Fetch feature schema for categorical options
                feature_schema = {fs['name']: fs for fs in model.get('feature_schema', [])}
                
                for _ in range(499):
                    noisy = base_df.sample(1, replace=True).copy()
                    for col in noisy.columns:
                        schema = feature_schema.get(col)
                        
                        if pd.api.types.is_numeric_dtype(noisy[col]):
                            # Gaussian noise for numerics
                            std = max(float(noisy[col].iloc[0]) * 0.2, 1e-3)
                            noisy[col] = noisy[col] + np.random.normal(0, std, size=len(noisy))
                        elif schema and schema.get('type') == 'categorical' and schema.get('options'):
                            # Random sampling from options for categorical to ensure variance
                            noisy[col] = np.random.choice(schema['options'], size=len(noisy))
                            
                    rows.append(noisy)
                training_df = pd.concat(rows, ignore_index=True)
                logger.info(f"[compute_lime_values] Expanded background: shape={training_df.shape}")

            # Determine mode
            mode = "classification" if model.get("task_type") in ["classification", "binary_classification", "multiclass_classification"] else "regression"
            logger.info(f"[compute_lime_values] LIME mode: {mode}")

            self.update_state(state="PROGRESS", meta={"status": "creating LIME explainer", "progress": 50})

            # Create explainer — training_df now has same columns as input_df
            raw_feature_names = [fs['name'] for fs in model.get('feature_schema', [])]
            explainer = LIMEService.create_explainer(
                model_obj,
                training_df,
                mode=mode,
                raw_feature_names=raw_feature_names
            )
            logger.info(f"[compute_lime_values] Explainer created. feature_names count={len(getattr(explainer, 'feature_names', []))}")

            # Get feature names from explainer
            explainer_feature_names = getattr(explainer, 'feature_names', list(training_df.columns))

            # Input instance (same columns as training_df) as 1-row DataFrame
            instance_to_explain = input_df[list(training_df.columns)].iloc[0:1]
            logger.info(f"[compute_lime_values] Input instance shape: {instance_to_explain.shape}")

            # For pipelines: do NOT preprocess the instance.
            # The LIME explainer operates in the ORIGINAL feature space.
            # We wrap the full pipeline's predict_proba as the prediction function,
            # so LIME perturbs in original feature space → pipeline handles encoding internally.
            if isinstance(model_obj, Pipeline):
                logger.info(f"[compute_lime_values] Pipeline detected: LIME will use original {instance_to_explain.shape[1]}-feature space with full pipeline predict_proba")

            self.update_state(state="PROGRESS", meta={"status": "computing LIME values", "progress": 70})

            # Compute LIME explanation
            # Use high num_features to ensure all features are shown as requested by user
            max_feat_count = len(explainer_feature_names)
            requested_num = max(num_features or 10, max_feat_count)
            
            try:
                lime_data = LIMEService.explain_instance(
                    explainer,
                    model_obj,
                    instance_to_explain,
                    num_features=requested_num
                )
                logger.info(f"[compute_lime_values] explain_instance returned: explanation count={len(lime_data.get('explanation', []))}")
            except Exception as lime_err:
                logger.error(f"[compute_lime_values] LIMEService.explain_instance FAILED: {type(lime_err).__name__}: {lime_err}")
                logger.error(f"[compute_lime_values] Full traceback:\n{traceback.format_exc()}")
                raise

            # ── Sanitize LIME output for MongoDB (BSON requires string keys) ──
            # Extract raw explanation: list of (feature_str, float) tuples
            contributions = lime_data.get("explanation", [])
            
            # contributions: list of (feature_str, float) tuples → serialize as list of dicts
            contributions_safe = [
                {"feature": str(feat), "weight": float(np.nan_to_num(w, nan=0.0))}
                for feat, w in contributions
            ] if contributions else []

            # intercept: LIME returns {class_int: float} or float
            raw_intercept = lime_data.get("intercept", {})
            if isinstance(raw_intercept, dict):
                # Ensure keys are strings for MongoDB
                intercept_safe = {str(k): float(np.nan_to_num(v, nan=0.0)) for k, v in raw_intercept.items()}
            else:
                intercept_safe = float(np.nan_to_num(raw_intercept, nan=0.0)) if raw_intercept is not None else 0.0

            # actual_prediction: true pipeline probabilities (list of floats per class)
            raw_actual_pred = lime_data.get("actual_prediction")
            if hasattr(raw_actual_pred, "tolist"):
                local_pred_safe = [float(np.nan_to_num(v, nan=0.0)) for v in raw_actual_pred.tolist()]
            elif isinstance(raw_actual_pred, (list, tuple)):
                local_pred_safe = [float(np.nan_to_num(v, nan=0.0)) for v in raw_actual_pred]
            else:
                local_pred_safe = [float(np.nan_to_num(raw_actual_pred, nan=0.0))] if raw_actual_pred is not None else []
            
            # Extract confidence (max proba) for easier frontend display
            prediction_confidence = 0.0
            if local_pred_safe:
                prediction_confidence = max(local_pred_safe)

            self.update_state(state="PROGRESS", meta={"status": "saving results", "progress": 90})

            # Save explanation to database
            explanation_doc = {
                "prediction_id": prediction_id,
                "model_id": model_id,
                "method": "lime",
                "explanation_type": "local",
                "lime_weights": contributions_safe,
                "lime_intercept": intercept_safe,
                "lime_local_pred": local_pred_safe,
                "prediction_confidence": prediction_confidence,
                "explained_class": lime_data.get("explained_class"),
                "explained_class_index": lime_data.get("explained_class_index"),
                "feature_names": [str(f) for f in explainer_feature_names],
                "nl_explanation": None,
                "task_id": self.request.id,
                "task_status": "complete",
                "created_at": datetime.utcnow()
            }

            result = await db.explanations.insert_one(explanation_doc)
            explanation_id = str(result.inserted_id)
            logger.info(f"[compute_lime_values] Explanation saved: explanation_id={explanation_id}, lime_weights count={len(contributions)}")

            self.update_state(state="SUCCESS", meta={"status": "complete", "explanation_id": explanation_id, "progress": 100})

            return {"explanation_id": explanation_id, "status": "complete"}

        return asyncio.run(async_task())

    except Exception:
        logger.error(f"[compute_lime_values] TASK FAILED:\n{traceback.format_exc()}")
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

            # Sample background data to avoid extremely long computation
            if len(background_df) > MAX_GLOBAL_SHAP_SAMPLES:
                background_df = background_df.sample(n=MAX_GLOBAL_SHAP_SAMPLES, random_state=42)

            self.update_state(state="PROGRESS", meta={"status": "creating LIME explainer", "progress": 50})

            # Determine mode
            mode = "classification" if model.get("task_type") in ["classification", "binary_classification", "multiclass_classification"] else "regression"

            # Create explainer (will preprocess background data if pipeline)
            explainer = LIMEService.create_explainer(
                model_obj,
                background_df,
                mode=mode
            )

            # Get feature names from explainer (may be preprocessed if pipeline)
            explainer_feature_names = getattr(explainer, 'feature_names', list(background_df.columns))

            self.update_state(state="PROGRESS", meta={"status": "computing global LIME", "progress": 60})

            # For global LIME, we pass raw data directly; explain_global calls explain_instance which handles the Pipeline.
            samples_for_explanation = background_df

            # Compute aggregated LIME importance
            lime_global = LIMEService.explain_global(
                explainer,
                model_obj,
                samples_for_explanation,
                num_features=num_features
            )

            # Validate: ensure feature_importance is not empty
            lime_feature_importance = lime_global.get("feature_importance", [])
            if not lime_feature_importance:
                logger.error("[compute_global_lime] LIME returned empty feature_importance - background data may be invalid or LIME failed")
                raise ValueError("Global LIME failed to produce feature importance")

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
        logger.info(f"[_compute_shap] START: framework={framework}, input_data.shape={input_data.shape}")
        logger.debug(f"[_compute_shap] Expected columns (first 10): {expected_columns[:10]}")
        logger.debug(f"[_compute_shap] Expected columns types: {[type(c).__name__ for c in expected_columns[:10]]}")

        def _prepare_background(df: Optional[pd.DataFrame]) -> pd.DataFrame:
            """Ensure SHAP background data matches model input columns, tolerating name-style mismatches."""
            if df is None or len(df) == 0:
                logger.info(f"[_compute_shap] No background data provided, using input_data as background")
                return input_data[expected_columns]

            bg_df = df.copy()
            logger.debug(f"[_compute_shap] _prepare_background: input bg_df.shape={bg_df.shape}, columns={list(bg_df.columns)[:10]}")

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
                logger.debug(f"[_compute_shap] Missing columns: {missing[:5]}, synthesizing from input row")
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
            logger.debug(f"[_compute_shap] selected_actual_cols (first 5): {selected_actual_cols[:5]}")
            logger.debug(f"[_compute_shap] selected_actual_cols types: {[type(c).__name__ for c in selected_actual_cols[:5]]}")
            bg_df = bg_df[selected_actual_cols].copy()
            bg_df.columns = expected_columns
            logger.debug(f"[_compute_shap] _prepare_background: output bg_df.shape={bg_df.shape}")
            return bg_df

        # Initialize with input data columns as default feature names
        final_feature_names = expected_columns
        logger.info(f"[_compute_shap] Expected columns count: {len(expected_columns)}")

        # Check if model is a sklearn Pipeline
        if isinstance(model_obj, Pipeline):
            logger.info("[_compute_shap] Model is a Pipeline - explaining ORIGINAL feature space for perfect mapping")
            
            # For Pipelines, we prefer explaining the ORIGINAL features (the inputs to the whole pipeline)
            # This ensures we get exactly 11 SHAP values matching the 11 input features,
            # avoiding internal complexity like derived features and one-hot encoding in the graph.
            
            predict_fn_full = model_obj.predict_proba if hasattr(model_obj, 'predict_proba') else model_obj.predict

            def _predict_full_pipeline(values):
                """Wraps full pipeline predict for SHAP sampling in original feature space."""
                # SHAP passes numpy arrays; we convert back to DataFrame for the pipeline
                if isinstance(values, np.ndarray):
                    df = pd.DataFrame(values, columns=expected_columns)
                else:
                    df = pd.DataFrame(values, columns=expected_columns)
                return predict_fn_full(df)

            # Prepare background data (raw)
            if background_data is not None and len(background_data) > 0:
                bg_raw = _prepare_background(background_data)
            else:
                bg_raw = _prepare_background(input_data)
            
            # Cap background for performance (KernelExplainer is slow on large backgrounds)
            bg_capped = bg_raw.iloc[:30] if len(bg_raw) > 30 else bg_raw
            
            np.random.seed(42)
            logger.info(f"[_compute_shap] Initializing KernelExplainer on {len(expected_columns)} original features. Background shape: {bg_capped.shape}")
            explainer = shap.KernelExplainer(_predict_full_pipeline, bg_capped.values)
            
            # Predict the instance to be explained
            shap_values = explainer.shap_values(input_data.values)
            expected_value = explainer.expected_value
            final_feature_names = expected_columns
            
            # Detailed Logging
            # Detailed Logging of raw SHAP output
            logger.info(f"[_compute_shap] Raw shap_values type: {type(shap_values)}")
            if isinstance(shap_values, list):
                logger.info(f"[_compute_shap]   List length: {len(shap_values)}")
                if len(shap_values) > 0:
                    logger.info(f"[_compute_shap]   First element shape: {getattr(shap_values[0], 'shape', 'N/A')}")
            elif hasattr(shap_values, 'shape'):
                logger.info(f"[_compute_shap]   Array shape: {shap_values.shape}")

            # ========================================================
            # SYSTEMATIC EXTRACTION V3 (EXTREME DIAGNOSTICS)
            # ========================================================
            try:
                # 1. LOG PIPELINE INPUT INFO
                logger.info(f"[_compute_shap] V3 DIAG: expected_columns count={len(expected_columns)}")
                logger.info(f"[_compute_shap] V3 DIAG: input_data.shape={input_data.shape}")
                logger.info(f"[_compute_shap] V3 DIAG: bg_capped.shape={bg_capped.shape}")
                
                # 2. LOG RAW SHAP INFO
                logger.info(f"[_compute_shap] V3 DIAG: RAW TYPE={type(shap_values)}")
                if isinstance(shap_values, list):
                    logger.info(f"[_compute_shap] V3 DIAG: LIST LEN={len(shap_values)}")
                    for i, v in enumerate(shap_values):
                        logger.info(f"[_compute_shap] V3 DIAG:   Item {i} shape={getattr(v, 'shape', 'N/A')}")
                else:
                    logger.info(f"[_compute_shap] V3 DIAG: ARRAY SHAPE={getattr(shap_values, 'shape', 'N/A')}")

                # 3. ROBUST STACKING
                if isinstance(shap_values, list):
                    # Ensure all are arrays
                    s_arr = np.array([np.asarray(v) for v in shap_values])
                else:
                    s_arr = np.asarray(shap_values)
                
                logger.info(f"[_compute_shap] V3 DIAG: STACKED_ARR.SHAPE={s_arr.shape}")

                # 4. FEATURE DIMENSION IDENTIFICATION
                n_feat = len(expected_columns)
                # Find which axis has size n_feat
                feat_axis = -1
                for ax, size in enumerate(s_arr.shape):
                    if size == n_feat:
                        feat_axis = ax
                        break
                
                logger.info(f"[_compute_shap] V3 DIAG: Feature axis identified as {feat_axis}")

                # 5. EXTRACTION
                # We want a 1D vector of shape (n_feat,) for a specific class.
                if s_arr.ndim == 3:
                    # Case A: (classes, samples, features) -> Common in TreeExplainer
                    if feat_axis == 2:
                        c_idx = 1 if s_arr.shape[0] == 2 else 0
                        instance_vals = s_arr[c_idx, 0, :]
                        logger.info(f"[_compute_shap] V3 DIAG: Extracted from (classes, samples, features) using class={c_idx}")
                    
                    # Case B: (samples, features, classes) -> Common in KernelExplainer
                    elif feat_axis == 1:
                        # Third dimension is classes
                        c_dim_size = s_arr.shape[2]
                        c_idx = 1 if c_dim_size == 2 else 0
                        instance_vals = s_arr[0, :, c_idx]
                        logger.info(f"[_compute_shap] V3 DIAG: Extracted from (samples, features, classes) using class={c_idx}")
                    
                    else:
                        # Fallback for other 3D shapes
                        instance_vals = s_arr[0, 0, :] if s_arr.shape[2] == n_feat else s_arr[0, :, 0]
                
                elif s_arr.ndim == 2:
                    # (samples, features) or (classes, features)
                    if s_arr.shape[1] == n_feat:
                        instance_vals = s_arr[0, :]
                    else:
                        instance_vals = s_arr[:, 0]
                else:
                    instance_vals = s_arr.flatten()

                logger.info(f"[_compute_shap] V3 DIAG: FINAL instance_vals.shape={instance_vals.shape}")
                if len(instance_vals) != n_feat:
                    logger.warning(f"[_compute_shap] V3 DIAG: SHAPE MISMATCH! Expected {n_feat}, got {len(instance_vals)}")

            except Exception as e_extract:
                logger.error(f"[_compute_shap] EXTRACTION V3 FAILED: {e_extract}")
                instance_vals = np.zeros(len(expected_columns))
                raise ValueError(f"SHAP extraction V3 failed. Error: {e_extract}")

            # 6. SYSTEMATIC REPORT V2 (REFASHIONED)
            try:
                from tabulate import tabulate
                
                report_data = []
                for i, name in enumerate(final_feature_names):
                    if i < len(instance_vals):
                        try:
                            v = instance_vals[i]
                            f_val = float(np.asarray(v).flatten()[0])
                            report_data.append([
                                i, name, f"{f_val:+.6f}", abs(f_val),
                                "POS" if f_val > 0 else ("NEG" if f_val < 0 else "ZERO")
                            ])
                        except:
                            report_data.append([i, name, "ERR", 0, "FORMAT"])
                    else:
                        report_data.append([i, name, "MISSING", 0, "ERROR"])
                
                sorted_report = sorted(report_data, key=lambda x: x[3], reverse=True)
                
                logger.info("\n" + "@"*95 + "\n" +
                            "  SYSTEMATIC SHAP EXPLANATION REPORT (GLOBAL COMPATIBLE)\n" +
                            "@"*95 + f"\nExpected Base: {expected_value}\n" +
                            f"Total Features: {len(expected_columns)}\n" +
                            "-"*95 + "\n" +
                            tabulate(sorted_report, headers=["Idx", "Feature Name", "Value", "Abs Impact", "Dir"], tablefmt="grid") +
                            "\n" + "@"*95)
            except Exception as e_log:
                logger.warning(f"[_compute_shap] Logging V3 failed: {e_log}")
            # ========================================================
            
            return shap_values, expected_value, final_feature_names

        else:
            # For non-pipeline models, use shap.Explainer for auto-detection
            logger.info("[_compute_shap] Model is NOT a Pipeline - using model directly")
            try:
                logger.info("[_compute_shap] Attempting shap.Explainer on raw model with check_additivity=False")
                explainer = shap.Explainer(model_obj, input_data, check_additivity=False)
                shap_values = explainer(input_data)
                expected_value = explainer.expected_value
                logger.info(f"[_compute_shap] shap.Explainer succeeded: shap_values type={type(shap_values)}")
                # Convert shap.Explanation to raw array if needed
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values
                    logger.debug(f"[_compute_shap] Converted shap.Explanation to values array")
            except Exception as e:
                logger.warning(f"[_compute_shap] shap.Explainer failed: {e}, falling back to TreeExplainer")
                # Auto-detect failed, fallback to manual TreeExplainer
                try:
                    logger.info("[_compute_shap] Trying shap.TreeExplainer on raw model with check_additivity=False")
                    explainer = shap.TreeExplainer(model_obj, input_data, check_additivity=False)
                    shap_values = explainer.shap_values(input_data)
                    expected_value = explainer.expected_value
                    logger.info(f"[_compute_shap] TreeExplainer succeeded")
                except Exception as e2:
                    logger.warning(f"[_compute_shap] TreeExplainer failed: {e2}, falling back to KernelExplainer")
                    # TreeExplainer failed, fallback to KernelExplainer (slow)
                    if background_data is not None and len(background_data) > 0:
                        bg_raw = background_data if len(background_data) <= MAX_GLOBAL_SHAP_SAMPLES else background_data.iloc[:MAX_GLOBAL_SHAP_SAMPLES]
                        logger.info(f"[_compute_shap] Using provided background data: shape={bg_raw.shape}")
                        bg = _prepare_background(bg_raw)
                    else:
                        # Generate synthetic background if none provided
                        logger.info("[_compute_shap] No background data, using input_data as background")
                        bg = _prepare_background(input_data)
                    logger.info(f"[_compute_shap] Background prepared: shape={bg.shape}")

                    predict_fn = model_obj.predict_proba if hasattr(model_obj, "predict_proba") else model_obj.predict

                    def _predict_with_columns(values):
                        if isinstance(values, np.ndarray):
                            df = pd.DataFrame(values, columns=expected_columns)
                        else:
                            df = values[expected_columns] if isinstance(values, pd.DataFrame) else values
                        return predict_fn(df)

                    np.random.seed(42)  # Ensure deterministic sampling
                    logger.info("[_compute_shap] Creating KernelExplainer with bg.values")
                    explainer = shap.KernelExplainer(_predict_with_columns, bg.values)
                    shap_values = explainer.shap_values(input_data.values)
                    expected_value = explainer.expected_value
                    logger.info(f"[_compute_shap] KernelExplainer completed")

        logger.info(f"[_compute_shap] RETURN: shap_values type={type(shap_values)}, expected_value type={type(expected_value)}, final_feature_names count={len(final_feature_names)}")
        return shap_values, expected_value, final_feature_names

    except Exception as e:
        logger.error(f"[_compute_shap] FAILED with exception: {type(e).__name__}: {e}")
        logger.error(f"[_compute_shap] Full traceback:\n{traceback.format_exc()}")
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

# ==========================================
# New XAI Framework Tasks (InterpretML, Alibi, AIX360)
# ==========================================

def _run_xai_framework_local(self, prediction_id: str, model_id: str, method: str, num_features: int = 10):
    try:
        logger.info(f"[{method} local] START: prediction_id={prediction_id}, model_id={model_id}")
        self.update_state(state="PROGRESS", meta={"status": "loading model", "progress": 10})

        import asyncio
        async def async_task():
            from bson import ObjectId
            from datetime import datetime
            import traceback
            from app.db.mongo import connect_db, get_db

            await connect_db()
            db = await get_db()
            try:
                if method == "interpretml":
                    from app.services.interpretml_service import interpretml_service as service
                elif method == "alibi":
                    from app.services.alibi_service import alibi_service as service
                elif method == "aix360":
                    from app.services.aix360_service import aix360_service as service
                else:
                    raise ValueError(f"Unknown method {method}")
            except ImportError as ie:
                logger.error(f"Framework import error for {method}: {ie}")
                error_msg = f"{method.upper()} library not installed or import error. Detailed error: {ie}"
                
                # Save failure to database so UI can show it
                explanation_doc = {
                    "prediction_id": prediction_id,
                    "model_id": model_id,
                    "method": method,
                    "explanation_type": "local",
                    "task_id": self.request.id,
                    "task_status": "failed",
                    "error": error_msg,
                    "created_at": datetime.utcnow()
                }
                await db.explanations.insert_one(explanation_doc)
                return {"status": "failed", "error": error_msg}

            from app.services.model_loader_service import ModelLoaderService
            import pandas as pd
            import numpy as np

            await connect_db()
            db = await get_db()

            prediction = await db.predictions.find_one({"_id": ObjectId(prediction_id)})
            model = await db.models.find_one({"_id": ObjectId(model_id)})
            if not prediction or not model:
                raise ValueError("Prediction or Model not found")

            model_obj, framework = await ModelLoaderService.load_model(model["file_path"])
            input_df = pd.DataFrame([prediction["input_data"]])
            
            # Align features with model schema if available to ensure correct order
            if "feature_schema" in model and model["feature_schema"]:
                try:
                    input_df = _align_background_data(input_df, model)
                    logger.info("Aligned input data with model schema")
                except Exception as e:
                    logger.warning(f"Failed to align input data: {e}")

            # Improve background data generation (more samples for robust fitting)
            # We also need to vary categorical columns so the model produces multiple
            # prediction classes — required for BRCG to learn meaningful rules.
            training_df = input_df.copy()
            n_samples = 100 if method in ['alibi', 'aix360'] else 50

            # Collect feature schema for categorical value ranges
            feature_schema = model.get("feature_schema", [])
            cat_value_map = {}
            for fs in feature_schema:
                vals = fs.get("categories") or fs.get("values") or fs.get("allowed_values")
                if vals:
                    cat_value_map[fs["name"]] = vals

            for _ in range(n_samples):
                noisy = input_df.copy()
                for col in noisy.columns:
                    col_series = noisy[col]
                    if pd.api.types.is_numeric_dtype(col_series):
                        noise_std = max(float(abs(col_series.iloc[0])) * 0.1, 1e-2)
                        noisy[col] = col_series + np.random.normal(0, noise_std)
                    else:
                        # For categorical columns: randomly sample from known values
                        if col in cat_value_map and cat_value_map[col]:
                            noisy[col] = np.random.choice(cat_value_map[col])
                        # else leave as-is (no known categories)
                training_df = pd.concat([training_df, noisy], ignore_index=True)

            task_type = "classification" if model.get("task_type") in ["classification", "binary_classification"] else "regression"
            
            # Explicitly pass arguments to avoid signature mismatches
            explainer = service.create_explainer(
                model=model_obj, 
                framework=framework, 
                training_data=training_df, 
                feature_names=list(training_df.columns), 
                mode=task_type
            )
            
            local_exp_data = service.explain_instance(
                explainer_obj=explainer,
                instance=input_df.values[0],
                feature_names=list(training_df.columns)
            )

            explanation_doc = {
                "prediction_id": prediction_id,
                "model_id": model_id,
                "method": method,
                "explanation_type": "local",
                "explanation_data": local_exp_data, # Use a generic payload
                "feature_names": list(training_df.columns),
                "task_id": self.request.id,
                "task_status": "complete",
                "created_at": datetime.utcnow()
            }

            result = await db.explanations.insert_one(explanation_doc)
            self.update_state(state="SUCCESS", meta={"status": "complete", "explanation_id": str(result.inserted_id), "progress": 100})
            return {"explanation_id": str(result.inserted_id), "status": "complete"}

        return asyncio.run(async_task())
    except Exception as e:
        import traceback
        error_detail = str(e)
        logger.error(f"[{method} task failed] {traceback.format_exc()}")
        
        # Try to save general failure as well if possible
        try:
            import asyncio
            from app.db.mongo import connect_db, get_db
            from bson import ObjectId
            from datetime import datetime

            async def save_failure():
                await connect_db()
                db = await get_db()
                await db.explanations.insert_one({
                    "prediction_id": prediction_id,
                    "model_id": model_id,
                    "method": method,
                    "explanation_type": "local",
                    "task_id": self.request.id,
                    "task_status": "failed",
                    "error": error_detail,
                    "created_at": datetime.utcnow()
                })
            asyncio.run(save_failure())
        except:
            pass
            
        return {"status": "failed", "error": error_detail}

def _run_xai_framework_global(self, model_id: str, background_data_path: str, method: str, num_features: int = 10):
    try:
        logger.info(f"[{method} global] START: model_id={model_id}")
        self.update_state(state="PROGRESS", meta={"status": "loading model", "progress": 10})

        import asyncio
        async def async_task():
            from app.db.mongo import connect_db, get_db
            from app.services.model_loader_service import ModelLoaderService
            from app.utils.file_handler import storage
            import pandas as pd
            from bson import ObjectId
            from datetime import datetime

            # Dynamic import with fallback
            try:
                if method == "interpretml":
                    from app.services.interpretml_service import interpretml_service as service
                elif method == "alibi":
                    from app.services.alibi_service import alibi_service as service
                elif method == "aix360":
                    from app.services.aix360_service import aix360_service as service
                else:
                    raise ValueError(f"Unknown method {method}")
            except ImportError as ie:
                logger.error(f"Framework import error for {method}: {ie}")
                return {"status": "failed", "error": f"{method.upper()} library not installed or import error. Detailed error: {ie}"}

            await connect_db()
            db = await get_db()

            model = await db.models.find_one({"_id": ObjectId(model_id)})
            if not model:
                raise ValueError("Model not found")

            model_obj, framework = await ModelLoaderService.load_model(model["file_path"])
            
            # Load Background
            bg_bytes = await storage.download_file(background_data_path)
            training_df = pd.read_csv(io.BytesIO(bg_bytes))

            task_type = "classification" if model.get("task_type") in ["classification", "binary_classification"] else "regression"
            
            explainer = service.create_explainer(model_obj, framework, training_df, list(training_df.columns), mode=task_type)
            
            global_exp_data = service.explain_global(
                explainer,
                model_obj,
                training_df,
                num_features=num_features
            )

            explanation_doc = {
                "model_id": model_id,
                "method": method,
                "explanation_type": "global",
                "explanation_data": global_exp_data, # Generic payload
                "feature_names": list(training_df.columns),
                "task_id": self.request.id,
                "task_status": "complete",
                "created_at": datetime.utcnow()
            }

            result = await db.explanations.insert_one(explanation_doc)
            self.update_state(state="SUCCESS", meta={"status": "complete", "explanation_id": str(result.inserted_id), "progress": 100})
            return {"explanation_id": str(result.inserted_id), "status": "complete"}

        return asyncio.run(async_task())
    except Exception as e:
        import traceback
        logger.error(f"[{method} global task failed] {traceback.format_exc()}")
        return {"status": "failed", "error": str(e)}

@celery_app.task(bind=True, name="compute_interpretml_values")
def compute_interpretml_values(self, prediction_id: str, model_id: str, num_features: int = 10):
    return _run_xai_framework_local(self, prediction_id, model_id, "interpretml", num_features)

@celery_app.task(bind=True, name="compute_global_interpretml")
def compute_global_interpretml(self, model_id: str, background_data_path: str, num_features: int = 10):
    import importlib
    return _run_xai_framework_global(self, model_id, background_data_path, "interpretml", num_features)

@celery_app.task(bind=True, name="compute_alibi_values")
def compute_alibi_values(self, prediction_id: str, model_id: str, num_features: int = 10):
    return _run_xai_framework_local(self, prediction_id, model_id, "alibi", num_features)

@celery_app.task(bind=True, name="compute_global_alibi")
def compute_global_alibi(self, model_id: str, background_data_path: str, num_features: int = 10):
    import importlib
    return _run_xai_framework_global(self, model_id, background_data_path, "alibi", num_features)

@celery_app.task(bind=True, name="compute_aix360_values")
def compute_aix360_values(self, prediction_id: str, model_id: str, num_features: int = 10):
    return _run_xai_framework_local(self, prediction_id, model_id, "aix360", num_features)

@celery_app.task(bind=True, name="compute_global_aix360")
def compute_global_aix360(self, model_id: str, background_data_path: str, num_features: int = 10):
    import importlib
    return _run_xai_framework_global(self, model_id, background_data_path, "aix360", num_features)
