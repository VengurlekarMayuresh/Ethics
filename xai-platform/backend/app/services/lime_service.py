import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from app.services.model_loader_service import ModelLoaderService
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline

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

        For pipelines, the training data is preprocessed to numeric space before creating the explainer.
        For non-pipeline models, raw training data is used (must be numeric or have categorical features properly handled).

        Args:
            model: The trained model object
            framework: Model framework (sklearn, xgboost, etc.)
            training_data: Background/training data for LIME (raw or preprocessed)
            feature_names: Names of features (raw or preprocessed, depending on model type)
            mode: "regression" or "classification"

        Returns:
            LimeTabularExplainer instance
        """
        from sklearn.pipeline import Pipeline

        # Check if model is a sklearn Pipeline
        if isinstance(model, Pipeline):
            from sklearn.pipeline import Pipeline
            # Build composite preprocessor from all steps except the final estimator.
            if len(model.steps) > 1:
                preprocessor_pipeline = Pipeline(model.steps[:-1])
            else:
                preprocessor_pipeline = None

            if preprocessor_pipeline is not None:
                # Transform raw training data through full preprocessing chain
                training_processed = preprocessor_pipeline.transform(training_data)
                if hasattr(training_processed, 'toarray'):  # sparse matrix
                    training_processed = training_processed.toarray()
                training_array = np.asarray(training_processed, dtype=float)

                # Determine processed feature names from the last step with get_feature_names_out
                processed_feature_names = None
                for step_name, step_obj in reversed(preprocessor_pipeline.steps):
                    if hasattr(step_obj, 'get_feature_names_out'):
                        try:
                            if hasattr(step_obj, 'feature_names_in_'):
                                input_features = step_obj.feature_names_in_
                                raw_names = step_obj.get_feature_names_out(input_features=input_features)
                            else:
                                raw_names = step_obj.get_feature_names_out()
                            cleaned_names = []
                            for name in raw_names:
                                if isinstance(name, bytes):
                                    name = name.decode('utf-8')
                                if '__' in name:
                                    name = name.split('__', 1)[1]
                                cleaned_names.append(name)
                            if len(cleaned_names) == training_array.shape[1]:
                                processed_feature_names = cleaned_names
                                break
                        except Exception:
                            processed_feature_names = None

                if processed_feature_names is None or len(processed_feature_names) != training_array.shape[1]:
                    processed_feature_names = [f"feature_{i}" for i in range(training_array.shape[1])]

                # No categorical features in preprocessed space (all numeric)
                categorical_features = None
            else:
                # No preprocessing steps, use raw data as-is
                training_array = training_data.values
                processed_feature_names = feature_names
                # Auto-detect categorical features
                categorical_features = []
                for i, col in enumerate(training_data.columns):
                    if training_data[col].dtype == 'object' or training_data[col].dtype.name == 'category':
                        categorical_features.append(i)
                    elif len(training_data[col].unique()) <= 10 and training_data[col].dtype in ['int64', 'int32']:
                        categorical_features.append(i)
                if not categorical_features:
                    categorical_features = None
        else:
            # Non-pipeline: use raw training data
            training_array = training_data.values
            processed_feature_names = feature_names
            # Auto-detect categorical features
            categorical_features = []
            for i, col in enumerate(training_data.columns):
                if training_data[col].dtype == 'object' or training_data[col].dtype.name == 'category':
                    categorical_features.append(i)
                elif len(training_data[col].unique()) <= 10 and training_data[col].dtype in ['int64', 'int32']:
                    categorical_features.append(i)
            if not categorical_features:
                categorical_features = None

        # Ensure stable/valid background stats for LIME perturbation sampling.
        training_array = LIMEService._sanitize_training_array(training_array)

        # Determine if classification and number of classes
        num_classes = None
        if mode == "classification":
            try:
                if hasattr(model, 'predict_proba'):
                    # For sklearn models
                    # Need to predict on preprocessed data if pipeline
                    if isinstance(model, Pipeline):
                        # Predict on preprocessed features
                        preds = model.predict_proba(training_array)
                    else:
                        preds = model.predict_proba(training_data[:min(10, len(training_data))])
                    if isinstance(preds, list) and len(preds) > 1:
                        # Multi-class returns list of arrays
                        num_classes = len(preds)
                    elif isinstance(preds, np.ndarray) and preds.ndim == 2:
                        num_classes = preds.shape[1]
            except:
                pass

        # Create explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_array,
            feature_names=processed_feature_names,
            mode=mode,
            class_names=None,
            kernel_width=3,
            discretize_continuous=True,
            discretizer='quartile',
            categorical_features=categorical_features,
            verbose=False
        )

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

        # Define prediction function based on model type
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):
            # For pipelines, the model expects raw input, but the explainer works on preprocessed data.
            # We need a prediction function that maps preprocessed space back to raw? That's impossible.
            # Actually, LIME will perturb in the explainer's space (preprocessed). We need to predict on that space.
            # So we use the final estimator directly.
            final_estimator = model.steps[-1][1]

            if hasattr(final_estimator, 'predict_proba'):
                predict_fn = final_estimator.predict_proba
            else:
                predict_fn = final_estimator.predict
        else:
            # For non-pipeline models, we may need to handle DataFrames if feature names available
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
        exp = explainer.explain_instance(instance, predict_fn, **explain_kwargs)

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
                    contrib_iterable = contributions.items()
                elif isinstance(contributions, list):
                    # Convert list of tuples to (index, weight) pairs
                    contrib_iterable = contributions
                else:
                    continue

                sorted_contrib = sorted(contrib_iterable, key=lambda x: abs(x[1]), reverse=True)

                feature_weights = []
                for feature_idx, weight in sorted_contrib[:num_features]:
                    feature_name = explainer.feature_names[feature_idx]
                    # For the value, we need the instance value in the explainer's space.
                    feature_value = instance[feature_idx] if feature_idx < len(instance) else None
                    feature_weights.append({
                        "feature": feature_name,
                        "weight": float(weight),
                        "value": float(feature_value) if feature_value is not None else None
                    })

                explanation_data["local_exp"][str(label)] = [
                    {"feature": explainer.feature_names[idx], "weight": float(weight)}
                    for idx, weight in contrib_iterable
                ]
                explanation_data["list_of_contributions"] = feature_weights

        # ------------------------------------------------------------
        # AGGREGATION: For pipelines with OneHotEncoder, combine
        # one-hot encoded features back to original categorical features.
        # ------------------------------------------------------------
        if 'local_exp' in explanation_data and explanation_data['local_exp'] and isinstance(model, Pipeline):
            from collections import defaultdict
            from sklearn.pipeline import Pipeline

            # Find the ColumnTransformer (feature expander) for mapping encoded features to original categories
            feature_expander = None
            for step_name, step_obj in model.steps:
                if hasattr(step_obj, 'transformers_'):
                    feature_expander = step_obj
                    break

            if feature_expander is not None and hasattr(explainer, 'feature_names'):
                # Build mapping: preprocessed feature index -> original categorical feature name
                encoded_to_original = {}
                original_feature_names_set = set()

                # feature_expander has transformers_ attribute
                for transformer_name, transformer_obj, cols in feature_expander.transformers_:
                    transformer_class = transformer_obj.__class__.__name__

                    for col in cols:
                        original_feature_names_set.add(col)
                        # Match encoded features to original column using normalized names
                        for idx, fname in enumerate(explainer.feature_names):
                            if isinstance(fname, bytes):
                                fname = fname.decode('utf-8')
                            fname_str = str(fname)
                            # Normalize: remove transformer prefix if present (e.g., "cat__name_X" -> "name_X")
                            if '__' in fname_str:
                                parts = fname_str.split('__', 1)
                                norm_name = parts[1] if len(parts) == 2 else fname_str
                            else:
                                norm_name = fname_str
                            # Match: exact match for numeric features, or starts with "col_" for one-hot
                            if norm_name == col or norm_name.startswith(col + '_'):
                                encoded_to_original[idx] = col

                # Perform aggregation if we have a mapping
                if encoded_to_original:
                    new_local_exp = {}
                    for label_str, contrib_list in explanation_data['local_exp'].items():
                        aggregated_contrib = defaultdict(float)
                        # contrib_list: list of {"feature": name, "weight": val}
                        for item in contrib_list:
                            feat_name = item['feature']
                            # Find index of this feature in explainer.feature_names
                            try:
                                idx = list(explainer.feature_names).index(feat_name)
                            except ValueError:
                                # Can't find - keep as-is
                                aggregated_contrib[feat_name] += item['weight']
                                continue

                            original = encoded_to_original.get(idx)
                            if original:
                                aggregated_contrib[original] += item['weight']
                            else:
                                # Unmapped feature - keep as-is
                                aggregated_contrib[feat_name] += item['weight']

                        # Sort by absolute weight
                        new_local_exp[label_str] = [
                            {"feature": feat, "weight": w}
                            for feat, w in sorted(aggregated_contrib.items(), key=lambda x: abs(x[1]), reverse=True)
                        ]

                    explanation_data['local_exp'] = new_local_exp

                    # Update list_of_contributions similarly
                    if 'list_of_contributions' in explanation_data and explanation_data['list_of_contributions']:
                        aggregated_contrib2 = defaultdict(float)
                        for item in explanation_data['list_of_contributions']:
                            feat_name = item['feature']
                            try:
                                idx = list(explainer.feature_names).index(feat_name)
                            except ValueError:
                                aggregated_contrib2[feat_name] += abs(item.get('weight', 0))
                                continue
                            original = encoded_to_original.get(idx)
                            if original:
                                aggregated_contrib2[original] += abs(item.get('weight', 0))
                            else:
                                aggregated_contrib2[feat_name] += abs(item.get('weight', 0))

                        explanation_data['list_of_contributions'] = [
                            {"feature": feat, "weight": w, "value": None}
                            for feat, w in sorted(aggregated_contrib2.items(), key=lambda x: x[1], reverse=True)
                        ]

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
