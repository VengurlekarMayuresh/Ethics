import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from app.services.model_loader_service import ModelLoaderService
import joblib
import warnings
warnings.filterwarnings('ignore')

class LIMEService:
    """LIME explainer service for local explanations."""

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

        Args:
            model: The trained model object
            framework: Model framework (sklearn, xgboost, etc.)
            training_data: Background/training data for LIME
            feature_names: Names of features
            mode: "regression" or "classification"

        Returns:
            LimeTabularExplainer instance
        """
        # Convert training data to numpy
        training_array = training_data.values

        # Determine if classification and number of classes
        num_classes = None
        if mode == "classification":
            try:
                if hasattr(model, 'predict_proba'):
                    # For sklearn models
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
            feature_names=feature_names,
            mode=mode,
            class_names=None,
            kernel_width=3,
            num_features=min(10, len(feature_names)),
            discretize_continuous=True,
            discretizer='quartile',
            verbose=False
        )

        return explainer

    @staticmethod
    def explain_instance(
        explainer: lime.lime_tabular.LimeTabularExplainer,
        model,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single instance.

        Args:
            explainer: LIME explainer instance
            model: Model to explain
            instance: Single data point (1D array)
            num_features: Number of top features to return
            num_samples: Number of samples LIME generates

        Returns:
            Dictionary with explanation data
        """
        # Define prediction function
        def predict_fn(data):
            # Convert to DataFrame for models that expect it
            if hasattr(model, 'predict_proba'):
                try:
                    preds = model.predict_proba(data)
                except:
                    preds = model.predict(data)
            else:
                preds = model.predict(data)

            # Handle different output formats
            if isinstance(preds, list):
                # Multi-output or multi-class
                return np.column_stack(preds)
            elif preds.ndim == 1:
                # Regression or binary classification with 1D output
                return preds.reshape(-1, 1) if len(preds.shape) == 1 else preds
            else:
                return preds

        # Generate explanation
        exp = explainer.explain_instance(
            instance,
            predict_fn,
            num_features=min(num_features, len(explainer.feature_names)),
            num_samples=num_samples,
            top_labels=1  # Get explanation for top predicted class
        )

        # Extract explanation data
        explanation_data = {
            "intercept": float(exp.intercept[0]) if isinstance(exp.intercept, np.ndarray) else float(exp.intercept),
            "local_exp": {},
            "local_pred": float(exp.local_pred[0]) if isinstance(exp.local_pred, np.ndarray) else float(exp.local_pred),
            "list_of_contributions": []
        }

        # Get top features and their weights
        if exp.local_exp:
            # For each label (usually just 1 for regression or top class)
            for label, contributions in exp.local_exp.items():
                # contributions is a dict: {feature_index: weight}
                sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

                feature_weights = []
                for feature_idx, weight in sorted_contrib[:num_features]:
                    feature_name = explainer.feature_names[feature_idx]
                    feature_value = instance[feature_idx] if feature_idx < len(instance) else None
                    feature_weights.append({
                        "feature": feature_name,
                        "weight": float(weight),
                        "value": float(feature_value) if feature_value is not None else None
                    })

                explanation_data["local_exp"][str(label)] = [
                    {"feature": explainer.feature_names[idx], "weight": float(weight)}
                    for idx, weight in contributions.items()
                ]
                explanation_data["list_of_contributions"] = feature_weights

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
