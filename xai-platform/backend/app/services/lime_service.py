import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from app.services.model_loader_service import ModelLoaderService
import joblib
import warnings
import logging
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class LIMEService:
    """LIME explainer service for local explanations."""

    @staticmethod
    def _sanitize_training_array(training_array: np.ndarray) -> np.ndarray:
        """
        Ensure LIME background array has finite values and positive variance per column.
        This prevents scipy.stats.truncnorm domain errors during perturbation sampling.
        """
        arr = np.asarray(training_array, dtype=float)

        # Replace NaN/Inf with finite values.
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

        if arr.ndim != 2 or arr.shape[0] == 0:
            return arr

        stds = arr.std(axis=0)
        bad_cols = np.where((~np.isfinite(stds)) | (stds <= 1e-12))[0]
        if bad_cols.size > 0 and arr.shape[0] > 1:
            # Add tiny deterministic jitter only to problematic columns.
            jitter = np.linspace(-1e-6, 1e-6, arr.shape[0])
            for col in bad_cols:
                arr[:, col] = arr[:, col] + jitter

        return arr

    @staticmethod
    def create_explainer(
        model,
        framework: str,
        training_data: pd.DataFrame,
        feature_names: List[str],
        mode: str = "regression"
    ) -> lime.lime_tabular.LimeTabularExplainer:
        """
        Create a LIME tabular explainer for the given model.

        For pipelines, the training data is encoded to numeric codes for LIME (since LIME requires numeric),
        but the pipeline itself expects raw categorical values. We store category lists to decode later.
        For non-pipeline models, training data is also encoded to numeric codes; the model is expected
        to have been trained on numeric data (so no decoding needed).

        Args:
            model: The trained model object
            framework: Model framework (sklearn, xgboost, etc.)
            training_data: Background/training data (raw, may contain strings)
            feature_names: Names of features (raw feature names)
            mode: "regression" or "classification"

        Returns:
            LimeTabularExplainer instance
        """
        from sklearn.pipeline import Pipeline

        # Encode categorical features to numeric codes for LIME (which requires numeric input)
        training_encoded = training_data.copy()
        cat_code_maps = {}  # col -> {str_val: code}
        cat_categories = {}  # col -> list of categories (order = code)
        categorical_features = []

        for i, col in enumerate(training_encoded.columns):
            if training_encoded[col].dtype == 'object' or training_encoded[col].dtype.name == 'category':
                # Convert to pandas Categorical
                training_encoded[col] = training_encoded[col].astype('category')
                categories = list(training_encoded[col].cat.categories)
                cat_categories[col] = categories
                # Build mapping: string -> code
                cat_code_maps[col] = {cat: idx for idx, cat in enumerate(categories)}
                # Replace with numeric codes
                training_encoded[col] = training_encoded[col].cat.codes.astype(float)
                categorical_features.append(i)
            elif len(training_encoded[col].unique()) <= 10 and training_encoded[col].dtype in ['int64', 'int32', 'float64']:
                categorical_features.append(i)

        training_array = training_encoded.values.astype(float)
        if not categorical_features:
            categorical_features = None

        # Ensure stable/valid background stats for LIME perturbation.
        training_array = LIMEService._sanitize_training_array(training_array)

        # Determine number of classes for classification mode
        if mode == "classification" and hasattr(model, 'predict_proba'):
            try:
                if isinstance(model, Pipeline):
                    # Pipeline expects raw data (with strings)
                    sample_data = training_data.iloc[:min(10, len(training_data))]
                else:
                    # Non-pipeline expects numeric-coded data
                    sample_data = training_encoded.iloc[:min(10, len(training_encoded))]
                preds = model.predict_proba(sample_data)
                # Determine num_classes (not strictly used but could be useful)
                if isinstance(preds, list) and len(preds) > 1:
                    num_classes = len(preds)
                elif isinstance(preds, np.ndarray) and preds.ndim == 2:
                    num_classes = preds.shape[1]
            except Exception as e:
                logger.debug(f"Could not determine num_classes: {e}")

        # Create explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_array,
            feature_names=feature_names,
            mode=mode,
            class_names=None,
            kernel_width=3,
            discretize_continuous=True,
            discretizer='quartile',
            categorical_features=categorical_features,
            verbose=False
        )

        # Store metadata on explainer for later use
        explainer._cat_code_maps = cat_code_maps
        explainer._cat_categories = cat_categories
        explainer._is_pipeline = isinstance(model, Pipeline)

        return explainer

    @staticmethod
    def explain_instance(
        explainer: lime.lime_tabular.LimeTabularExplainer,
        model,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
        raw_instance: Optional[np.ndarray] = None  # Optional raw input for reference
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single instance.

        Args:
            explainer: LIME explainer instance (operates on preprocessed features if model is pipeline)
            model: Model to explain (can be sklearn Pipeline)
            instance: Single data point (1D array) in the explainer's feature space (preprocessed for pipelines)
            num_features: Number of top features to return
            num_samples: Number of samples LIME generates
            raw_instance: Optional raw input instance (for extracting original feature values)

        Returns:
            Dictionary with explanation data
        """
        # Get feature names from explainer
        feature_names = getattr(explainer, 'feature_names', None)

        # ENCODE INSTANCE if we have categorical mappings stored
        # The explainer expects numeric codes for categorical features (same as training_array)
        cat_code_maps = getattr(explainer, '_cat_code_maps', None)
        if cat_code_maps and feature_names is not None:
            # instance might be a raw array with strings; convert to encoded numeric
            instance_encoded = instance.copy()
            if isinstance(instance, np.ndarray):
                # For array input, map using feature names
                for i, col_name in enumerate(feature_names):
                    if col_name in cat_code_maps and i < len(instance_encoded):
                        val = instance_encoded[i]
                        # Convert to string to match mapping keys (handles bytes or other types)
                        val_str = str(val) if not isinstance(val, str) else val
                        if val_str in cat_code_maps[col_name]:
                            instance_encoded[i] = float(cat_code_maps[col_name][val_str])
                        else:
                            # Unknown category - use 0 as default
                            logger.warning(f"Unknown categorical value '{val}' for feature '{col_name}'. Using 0 as default code.")
                            instance_encoded[i] = 0.0
            else:
                # Should be array at this point, but fallback
                instance_encoded = instance
            instance_to_use = instance_encoded
        else:
            instance_to_use = instance

        # Define prediction function based on model type
        from sklearn.pipeline import Pipeline

        is_pipeline = getattr(explainer, '_is_pipeline', False)
        cat_categories = getattr(explainer, '_cat_categories', {}) if is_pipeline else {}

        if is_pipeline:
            # For pipeline models: LIME provides numeric codes; we must decode to raw strings
            # before passing to the pipeline, which includes its own encoder.
            if hasattr(model, 'predict_proba'):
                _pipeline = model
                def predict_fn(data):
                    if isinstance(data, np.ndarray):
                        df = pd.DataFrame(data, columns=feature_names)
                    else:
                        df = data.copy()
                    # Decode categorical columns
                    if cat_categories:
                        for col, categories in cat_categories.items():
                            if col in df.columns:
                                try:
                                    # Convert to numeric, round, and clip
                                    codes = pd.to_numeric(df[col], errors='coerce').round().astype(int)
                                    # Map codes to category strings
                                    decoded = []
                                    for code in codes:
                                        if pd.isna(code):
                                            # Use first category if code is missing
                                            decoded.append(categories[0] if categories else code)
                                        elif 0 <= code < len(categories):
                                            decoded.append(categories[code])
                                        else:
                                            # Unknown code -> use first category as fallback
                                            decoded.append(categories[0] if categories else code)
                                    df[col] = decoded
                                except Exception as e:
                                    logger.warning(f"Error decoding column '{col}': {e}")
                                    # Leave column as-is
                    return _pipeline.predict_proba(df)
            else:
                _pipeline = model
                def predict_fn(data):
                    if isinstance(data, np.ndarray):
                        df = pd.DataFrame(data, columns=feature_names)
                    else:
                        df = data.copy()
                    if cat_categories:
                        for col, categories in cat_categories.items():
                            if col in df.columns:
                                try:
                                    codes = pd.to_numeric(df[col], errors='coerce').round().astype(int)
                                    decoded = []
                                    for code in codes:
                                        if pd.isna(code):
                                            decoded.append(categories[0] if categories else code)
                                        elif 0 <= code < len(categories):
                                            decoded.append(categories[code])
                                        else:
                                            decoded.append(categories[0] if categories else code)
                                    df[col] = decoded
                                except Exception as e:
                                    logger.warning(f"Error decoding column '{col}': {e}")
                    return _pipeline.predict(df)
        else:
            # Non-pipeline: model expects numeric input (already encoded)
            if feature_names is not None:
                def predict_fn(data):
                    if isinstance(data, np.ndarray):
                        df = pd.DataFrame(data, columns=feature_names)
                    else:
                        df = data
                    if hasattr(model, 'predict_proba'):
                        return model.predict_proba(df)
                    elif hasattr(model, 'predict'):
                        return model.predict(df)
                    else:
                        raise TypeError("Model has no prediction method")
            else:
                def predict_fn(data):
                    if hasattr(model, 'predict_proba'):
                        return model.predict_proba(data)
                    elif hasattr(model, 'predict'):
                        return model.predict(data)
                    else:
                        raise TypeError("Model has no prediction method")

        # Generate explanation
        explain_kwargs = {
            "num_features": min(num_features, len(explainer.feature_names)),
            "num_samples": num_samples,
        }
        # Only pass top_labels for classification mode
        if getattr(explainer, 'mode', None) == 'classification':
            explain_kwargs["top_labels"] = 1
        exp = explainer.explain_instance(instance_to_use, predict_fn, **explain_kwargs)

        # Extract explanation data
        # exp.intercept can be: float, np.ndarray, or dict (for multi-class)
        if isinstance(exp.intercept, dict):
            # For classification with multiple classes, use intercept for the top predicted class
            # or the first class if we can't determine
            intercept_val = next(iter(exp.intercept.values())) if exp.intercept else 0.0
        elif isinstance(exp.intercept, np.ndarray):
            intercept_val = float(exp.intercept[0]) if exp.intercept.ndim > 0 else float(exp.intercept)
        else:
            intercept_val = float(exp.intercept)

        # exp.local_pred can also be array or scalar
        if isinstance(exp.local_pred, np.ndarray):
            local_pred_val = float(exp.local_pred[0]) if exp.local_pred.ndim > 0 else float(exp.local_pred)
        elif isinstance(exp.local_pred, dict):
            # Use first value from dict if multi-class
            local_pred_val = float(next(iter(exp.local_pred.values()))) if exp.local_pred else 0.0
        else:
            local_pred_val = float(exp.local_pred)

        explanation_data = {
            "intercept": intercept_val,
            "local_exp": {},
            "local_pred": local_pred_val,
            "list_of_contributions": []
        }

        # Get top features and their weights
        if exp.local_exp:
            # For each label (usually just 1 for regression or top class)
            for label, contributions in exp.local_exp.items():
                # contributions can be either a dict {feature_index: weight} or a list [(feature_index, weight), ...]
                if isinstance(contributions, dict):
                    # Convert to list immediately so it can be iterated multiple times
                    contrib_list = list(contributions.items())
                elif isinstance(contributions, list):
                    contrib_list = list(contributions)  # make a copy to be safe
                else:
                    continue

                sorted_contrib = sorted(contrib_list, key=lambda x: abs(x[1]), reverse=True)

                feature_weights = []
                for feature_idx, weight in sorted_contrib[:num_features]:
                    try:
                        feature_name = explainer.feature_names[feature_idx]
                    except (IndexError, KeyError):
                        feature_name = f"feature_{feature_idx}"
                    # For display: safely extract the raw value (may be string or numeric)
                    feature_value = instance[feature_idx] if feature_idx < len(instance) else None
                    # Safely convert to display value - don't crash on strings (e.g. 'No', 'Female')
                    if feature_value is None:
                        display_value = None
                    else:
                        try:
                            display_value = float(feature_value)
                        except (ValueError, TypeError):
                            display_value = str(feature_value)  # keep as string for categorical
                    feature_weights.append({
                        "feature": feature_name,
                        "weight": float(weight),
                        "value": display_value
                    })

                # Store ALL contributions in local_exp (not just top-N)
                explanation_data["local_exp"][str(label)] = [
                    {"feature": (explainer.feature_names[idx] if idx < len(explainer.feature_names) else f"feature_{idx}"),
                     "weight": float(weight)}
                    for idx, weight in contrib_list
                ]
                explanation_data["list_of_contributions"] = feature_weights

        # NOTE: The old aggregation block that mapped 1317 OHE preprocessed features back to
        # original features has been removed. LIME now operates entirely in the original
        # feature space (e.g. 7 features) using the full pipeline as predict_fn, so there
        # are no preprocessed one-hot features to aggregate.

        return explanation_data

    @staticmethod
    def explain_global(
        explainer: lime.lime_tabular.LimeTabularExplainer,
        model,
        samples: pd.DataFrame,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict[str, Any]:
        """
        Generate LIME explanations for multiple samples and aggregate to global importance.

        Args:
            explainer: LIME explainer instance
            model: Model to explain
            samples: Dataset to explain (pd.DataFrame)
            num_features: Number of features to return
            num_samples: LIME samples per instance

        Returns:
            Dictionary with aggregated feature importance
        """
        all_weights = []

        # Sample subset if too many samples (LIME is expensive)
        max_samples = min(50, len(samples))  # Limit to 50 for performance
        sample_indices = np.random.choice(len(samples), max_samples, replace=False)

        for idx in sample_indices:
            instance = samples.iloc[idx].values
            try:
                exp_data = LIMEService.explain_instance(
                    explainer, model, instance,
                    num_features=len(samples.columns),
                    num_samples=num_samples
                )

                # Collect absolute weights for all features
                for feat, weight in exp_data["local_exp"].get("0", exp_data["local_exp"].get("1", []) if len(exp_data["local_exp"]) > 0 else []):
                    # This is simplified - in practice we'd aggregate across all samples
                    all_weights.append({"feature": feat["feature"], "weight": abs(feat["weight"])})
            except Exception as e:
                continue

        # Aggregate by feature
        feature_sums = {}
        feature_counts = {}

        for item in all_weights:
            feat = item["feature"]
            w = item["weight"]
            feature_sums[feat] = feature_sums.get(feat, 0) + w
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

        # Calculate mean absolute weight
        feature_importance = []
        for feat in feature_sums:
            if feature_counts[feat] > 0:
                feature_importance.append({
                    "feature": feat,
                    "importance": feature_sums[feat] / feature_counts[feat]
                })

        # Sort by importance
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "feature_importance": feature_importance[:num_features],
            "num_samples_explained": len(sample_indices)
        }

lime_service = LIMEService()
