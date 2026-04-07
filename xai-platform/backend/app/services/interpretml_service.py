import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class InterpretMLService:
    """InterpretML explainer service for local and global explanations."""

    @staticmethod
    def create_explainer(
        model,
        framework: str,
        training_data: pd.DataFrame,
        feature_names: List[str],
        mode: str = "regression"
    ) -> Any:
        """
        Create an InterpretML explainer for the given model.
        For glassbox models like EBM, the model itself serves as the explainer.
        For blackbox models, this would initialize Morris Sensitivity or LIME.

        Args:
            model: The trained model object
            framework: Model framework (sklearn, xgboost, etc.)
            training_data: Background/training data
            feature_names: Names of features
            mode: "regression" or "classification"

        Returns:
            InterpretML Explainer instance
        """
        logger.info(f"Creating InterpretML explainer for framework: {framework}, mode: {mode}")
        
        # Stub implementation. In a real scenario, we might use:
        # from interpret.blackbox import MorrisSensitivity
        # return MorrisSensitivity(model.predict, training_data, feature_names=feature_names)
        
        # Assuming we just pass it back for demonstrations, or initializing a generic explainer:
        explainer = {"model": model, "feature_names": feature_names, "mode": mode, "training_data": training_data}
        return explainer

    @staticmethod
    def explain_instance(
        explainer: Any,
        model,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
        raw_instance: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate InterpretML explanation for a single instance.

        Args:
            explainer: InterpretML explainer instance
            model: Model to explain
            instance: Single data point (1D array)
            num_features: Number of top features to return
            num_samples: Number of samples generated
            raw_instance: Optional raw input instance

        Returns:
            Dictionary with explanation data matching XAI Platform format
        """
        logger.info("Generating InterpretML local explanation")
        
        # This is a stub for the actual interpret.blackbox local method.
        # e.g., local_exp = explainer.explain_local(instance)
        
        feature_names = explainer.get('feature_names', [f"feature_{i}" for i in range(len(instance))])
        
        # Generate dummy contributions based on instance values
        contributions = []
        for i, val in enumerate(instance[:num_features]):
            try:
                fname = feature_names[i]
            except IndexError:
                fname = f"feature_{i}"
            contributions.append({
                "feature": fname,
                "weight": float(np.random.normal(0, 1)), # Dummy weight
                "value": float(val) if isinstance(val, (int, float, np.number)) else str(val)
            })

        explanation_data = {
            "intercept": 0.0,
            "local_exp": {"0": [{"feature": c["feature"], "weight": c["weight"]} for c in contributions]},
            "local_pred": float(np.random.normal(100, 10)),
            "list_of_contributions": sorted(contributions, key=lambda x: abs(x["weight"]), reverse=True)
        }
        
        return explanation_data

    @staticmethod
    def explain_global(
        explainer: Any,
        model,
        samples: pd.DataFrame,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict[str, Any]:
        """
        Generate InterpretML explanations for multiple samples to get global importance.

        Args:
            explainer: InterpretML explainer instance
            model: Model to explain
            samples: Dataset to explain
            num_features: Number of features to return
            num_samples: Samples per instance

        Returns:
            Dictionary with aggregated feature importance
        """
        logger.info("Generating InterpretML global explanation")
        
        # Stub for global execution:
        # e.g., global_exp = explainer.explain_global(name="Global")
        
        feature_names = explainer.get('feature_names', list(samples.columns))
        
        # Generate dummy global feature importance
        importance = []
        for fname in feature_names[:num_features]:
            importance.append({
                "feature": fname,
                "importance": float(abs(np.random.normal(1, 0.5)))
            })
            
        importance.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "feature_importance": importance,
            "num_samples_explained": len(samples)
        }

interpretml_service = InterpretMLService()
