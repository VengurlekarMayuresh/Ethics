import shap
import numpy as np
import pandas as pd
import os
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

# Maximum number of background samples to use for global SHAP/LIME explanations
# Prevents extremely long computation times with large datasets
MAX_GLOBAL_SHAP_SAMPLES = int(os.getenv("MAX_GLOBAL_SHAP_SAMPLES", "200"))
import re

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

            # Sample background data to avoid extremely long computation
            if len(background_df) > MAX_GLOBAL_SHAP_SAMPLES:
                background_df = background_df.sample(n=MAX_GLOBAL_SHAP_SAMPLES, random_state=42)

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
            # because the model expects numeric inputs after preprocessing.

            # Find the preprocessing step in the pipeline
            preprocessor = None
            for step_name, step_obj in model_obj.steps:
                if hasattr(step_obj, 'transform'):
                    preprocessor = step_obj
                    break

            # Get the final estimator for prediction
            final_estimator = model_obj.steps[-1][1]

            # Determine final_feature_names from preprocessor (if available)
            # This is needed for aggregating one-hot encoded features later.
            final_feature_names = expected_columns  # default fallback
            if preprocessor is not None and hasattr(preprocessor, 'get_feature_names_out'):
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
                    pass  # keep default

            # Prepare background data (raw) - align columns, use FULL background if available
            if background_data is not None and len(background_data) > 0:
                bg = _prepare_background(background_data)  # use full dataset
            else:
                bg = _prepare_background(input_data)

            # --------------------------------------------------------
            # Try TreeExplainer first - FAST for tree-based models
            # --------------------------------------------------------
            try:
                # TreeExplainer can work directly with the pipeline and raw background
                explainer = shap.TreeExplainer(model_obj, bg)
                shap_values = explainer.shap_values(input_data)
                expected_value = explainer.expected_value
            except Exception:
                # TreeExplainer failed or not supported; fallback to KernelExplainer
                # For KernelExplainer we must limit background size to avoid freeze
                if len(bg) > MAX_GLOBAL_SHAP_SAMPLES:
                    bg_capped = bg.sample(n=MAX_GLOBAL_SHAP_SAMPLES, random_state=42)
                else:
                    bg_capped = bg

                # Preprocess the capped background to numeric space
                if preprocessor is not None:
                    bg_processed = preprocessor.transform(bg_capped)
                    if hasattr(bg_processed, 'toarray'):  # sparse matrix
                        bg_processed = bg_processed.toarray()
                    bg_numeric = np.asarray(bg_processed, dtype=float)
                else:
                    bg_numeric = bg_capped.values.astype(float)

                # Preprocess input data to numeric space for KernelExplainer
                if preprocessor is not None:
                    input_processed = preprocessor.transform(input_data)
                    if hasattr(input_processed, 'toarray'):
                        input_processed = input_processed.toarray()
                    input_numeric = np.asarray(input_processed, dtype=float)
                else:
                    input_numeric = input_data.values.astype(float)

                # Predict function for preprocessed data
                if hasattr(final_estimator, 'predict_proba'):
                    predict_fn = final_estimator.predict_proba
                else:
                    predict_fn = final_estimator.predict

                def _predict_preprocessed(values):
                    return predict_fn(values)

                # Create KernelExplainer
                np.random.seed(42)
                explainer = shap.KernelExplainer(_predict_preprocessed, bg_numeric)
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
                    bg_raw = background_data if len(background_data) <= MAX_GLOBAL_SHAP_SAMPLES else background_data.iloc[:MAX_GLOBAL_SHAP_SAMPLES]
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

                np.random.seed(42)  # Ensure deterministic sampling
                explainer = shap.KernelExplainer(_predict_with_columns, bg.values)
                shap_values = explainer.shap_values(input_data.values)
                expected_value = explainer.expected_value

        # ------------------------------------------------------------
        # AGGREGATION: For pipelines with OneHotEncoder, combine one-hot
        # encoded features back to original categorical features.
        # The frontend expects a manageable number of features (original inputs),
        # not hundreds of one-hot encoded columns.
        # ------------------------------------------------------------
        if isinstance(model_obj, Pipeline):
            preprocessor = None
            for step_name, step_obj in model_obj.steps:
                if hasattr(step_obj, 'transform'):
                    preprocessor = step_obj
                    break

            if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                # We have a preprocessor that expanded features (e.g., OneHotEncoder)
                # Build mapping from original feature to encoded column indices
                from collections import defaultdict
                original_to_encoded = defaultdict(list)
                original_feature_names_set = set()

                if hasattr(preprocessor, 'transformers_'):
                    for transformer_name, transformer_obj, cols in preprocessor.transformers_:
                        transformer_class = transformer_obj.__class__.__name__

                        for col in cols:
                            original_feature_names_set.add(col)
                            # Find all encoded columns that belong to this original column
                            for idx, fname in enumerate(final_feature_names):
                                if isinstance(fname, bytes):
                                    fname = fname.decode('utf-8')
                                fname_str = str(fname)
                                # Normalize: remove transformer prefix if present (e.g., "cat__name_X" -> "name_X")
                                if '__' in fname_str:
                                    parts = fname_str.split('__', 1)
                                    norm_name = parts[1] if len(parts) == 2 else fname_str
                                else:
                                    norm_name = fname_str
                                # Match: exact match for numeric, or starts with "col_" for one-hot
                                if norm_name == col or norm_name.startswith(col + '_'):
                                    original_to_encoded[col].append(idx)

                # If we found any grouping, aggregate
                if original_to_encoded:
                    n_samples = shap_values.shape[0] if hasattr(shap_values, 'shape') else len(shap_values)
                    # Ensure shap_values is 2D
                    if isinstance(shap_values, list):
                        # For classification, we take the class 1 (positive) for binary, or first class
                        # Convert to 2D array
                        if len(shap_values) > 0:
                            # shap_values[i] is for class i
                            # We'll aggregate for class 1 if binary and 2 classes, else class 0
                            if len(shap_values) == 2:
                                class_idx = 1
                            else:
                                class_idx = 0
                            shap_arr = np.asarray(shap_values[class_idx])
                            if shap_arr.ndim == 1:
                                shap_arr = shap_arr.reshape(1, -1)
                        else:
                            shap_arr = np.array([])
                    else:
                        shap_arr = np.asarray(shap_values)
                        if shap_arr.ndim == 1:
                            shap_arr = shap_arr.reshape(1, -1)

                    if shap_arr.ndim == 2 and shap_arr.shape[0] > 0:
                        # Create aggregated array
                        orig_features_list = sorted(original_feature_names_set)
                        aggregated_shap = np.zeros((shap_arr.shape[0], len(orig_features_list)), dtype=float)

                        for agg_idx, orig_feat in enumerate(orig_features_list):
                            encoded_indices = original_to_encoded.get(orig_feat, [])
                            if len(encoded_indices) == 1:
                                aggregated_shap[:, agg_idx] = shap_arr[:, encoded_indices[0]]
                            elif len(encoded_indices) > 1:
                                # Sum contributions from all encoded columns
                                aggregated_shap[:, agg_idx] = shap_arr[:, encoded_indices].sum(axis=1)
                            # else: feature not found (shouldn't happen) - leave as 0

                        shap_values = aggregated_shap
                        final_feature_names = orig_features_list
                    # else: shap_values is empty or malformed, leave as-is

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