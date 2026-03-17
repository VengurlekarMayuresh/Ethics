import joblib
import pickle
import numpy as np
import pandas as pd
import onnxruntime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from app.utils.file_handler import storage
from app.models.model_meta import FeatureSchema

class ModelLoaderService:
    """Service for loading and validating ML models."""

    SUPPORTED_FORMATS = {
        ".pkl": "sklearn",
        ".joblib": "sklearn",
        ".json": "xgboost",
        ".onnx": "onnx",
        ".h5": "keras",
        ".pt": "pytorch"
    }

    @staticmethod
    def detect_framework(model_file: str) -> str:
        """Detect ML framework from file extension."""
        ext = Path(model_file).suffix.lower()
        return ModelLoaderService.SUPPORTED_FORMATS.get(ext, "unknown")

    @staticmethod
    async def load_model(model_path: str) -> tuple[Any, str]:
        """
        Load model from storage and return model object with framework.
        Handles various ML frameworks and returns appropriate model object.
        """
        try:
            model_file = await storage.download_file(model_path)
            framework = ModelLoaderService.detect_framework(model_path)

            if framework == "sklearn":
                return joblib.loads(model_file), framework
            elif framework == "xgboost":
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(model_file)
                return model, framework
            elif framework == "onnx":
                return onnxruntime.InferenceSession(model_file), framework
            elif framework == "keras":
                from tensorflow import keras
                return keras.models.model_from_json(model_file.decode('utf-8')), framework
            else:
                raise ValueError(f"Unsupported model format: {framework}")

        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

    @staticmethod
    async def validate_model(model_obj: Any, framework: str) -> bool:
        """Validate that a model object can make predictions."""
        try:
            if framework == "sklearn":
                # Test with dummy data
                test_data = pd.DataFrame([[0.5, 0.5]])  # Adjust based on model
                model_obj.predict(test_data)
            elif framework == "xgboost":
                import xgboost as xgb
                test_data = xgb.DMatrix([[0.5, 0.5]])
                model_obj.predict(test_data)
            elif framework == "onnx":
                test_data = np.array([[0.5, 0.5]], dtype=np.float32)
                model_obj.run(None, {model_obj.get_inputs()[0].name: test_data})
            elif framework == "keras":
                test_data = np.array([[0.5, 0.5]])
                model_obj.predict(test_data)
            else:
                return False
            return True
        except Exception:
            return False

    @staticmethod
    async def get_model_info(model_obj: Any, framework: str) -> Dict[str, Any]:
        """
        Get basic information about the model for metadata.
        Returns task type (classification/regression) and feature info.
        """
        info = {"task_type": "unknown", "feature_info": {}}

        try:
            if framework == "sklearn":
                from sklearn.base import is_classifier, is_regressor
                if is_classifier(model_obj):
                    info["task_type"] = "classification"
                    info["classes"] = model_obj.classes_.tolist() if hasattr(model_obj, 'classes_') else None
                elif is_regressor(model_obj):
                    info["task_type"] = "regression"

                # Get feature names if available
                if hasattr(model_obj, 'feature_names_in'):
                    info["feature_info"] = {
                        "names": model_obj.feature_names_in_.tolist(),
                        "type": "numeric"
                    }

            elif framework == "xgboost":
                # XGBoost models can be used for classification or regression
                import xgboost as xgb
                # Check if it's a classification model by looking at output shape
                try:
                    test_data = xgb.DMatrix([[0.5, 0.5]])
                    prediction = model_obj.predict(test_data)
                    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                        info["task_type"] = "classification"
                        info["classes"] = list(range(prediction.shape[1]))
                    else:
                        info["task_type"] = "regression"
                except:
                    pass

            elif framework == "onnx":
                # Check output shape to determine task type
                test_data = np.array([[0.5, 0.5]], dtype=np.float32)
                outputs = model_obj.run(None, {model_obj.get_inputs()[0].name: test_data})
                if outputs[0].shape[1] > 1:
                    info["task_type"] = "classification"
                else:
                    info["task_type"] = "regression"

            elif framework == "keras":
                # Check output shape
                test_data = np.array([[0.5, 0.5]])
                outputs = model_obj.predict(test_data)
                if outputs.shape[1] > 1:
                    info["task_type"] = "classification"
                else:
                    info["task_type"] = "regression"

        except Exception:
            pass  # If we can't determine, leave as unknown

        return info

    @staticmethod
    async def generate_feature_schema(model_obj: Any, framework: str) -> List[FeatureSchema]:
        """
        Generate feature schema from model object.
        Returns list of FeatureSchema objects.
        """
        features = []

        try:
            if framework == "sklearn":
                if hasattr(model_obj, 'feature_names_in'):
                    feature_names = model_obj.feature_names_in_.tolist()
                    features = [FeatureSchema(name=fn, type="numeric") for fn in feature_names]

            elif framework == "xgboost":
                # XGBoost stores feature names in the Booster object
                import xgboost as xgb
                if hasattr(model_obj, 'feature_names'):
                    feature_names = model_obj.feature_names
                    features = [FeatureSchema(name=fn, type="numeric") for fn in feature_names]

            elif framework == "onnx":
                # Get input names from ONNX model
                input_names = [inp.name for inp in model_obj.get_inputs()]
                features = [FeatureSchema(name=inp, type="numeric") for inp in input_names]

            elif framework == "keras":
                # Get input shape from Keras model
                input_shape = model_obj.input_shape
                if input_shape and len(input_shape) > 1:
                    num_features = input_shape[1] if len(input_shape) > 1 else input_shape[0]
                    features = [FeatureSchema(name=f"feature_{i}", type="numeric") for i in range(num_features)]

        except Exception:
            pass  # If we can't generate, return empty list

        return features

    @staticmethod
    async def create_dummy_prediction(model_obj: Any, framework: str) -> Dict[str, Any]:
        """
        Create a dummy prediction to test model functionality.
        Returns a sample prediction result.
        """
        try:
            if framework == "sklearn":
                # Create dummy data based on model's expected input shape
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(model_obj)
                n_features = model_obj.n_features_in_ if hasattr(model_obj, 'n_features_in_') else 2
                test_data = pd.DataFrame([[0.5] * n_features])
                prediction = model_obj.predict(test_data)
                probability = model_obj.predict_proba(test_data) if hasattr(model_obj, 'predict_proba') else None

            elif framework == "xgboost":
                import xgboost as xgb
                n_features = model_obj.feature_names.__len__() if hasattr(model_obj, 'feature_names') else 2
                test_data = xgb.DMatrix([[0.5] * n_features])
                prediction = model_obj.predict(test_data)
                probability = model_obj.predict_proba(test_data) if hasattr(model_obj, 'predict_proba') else None

            elif framework == "onnx":
                input_name = model_obj.get_inputs()[0].name
                test_data = np.array([[0.5, 0.5]], dtype=np.float32)
                outputs = model_obj.run(None, {input_name: test_data})
                prediction = outputs[0]
                probability = None

            elif framework == "keras":
                test_data = np.array([[0.5, 0.5]])
                prediction = model_obj.predict(test_data)
                probability = None

            else:
                raise ValueError("Unknown framework")

            return {
                "prediction": prediction.tolist() if isinstance(prediction, (np.ndarray, list)) else prediction,
                "probability": probability.tolist() if probability is not None else None,
                "task_type": "classification" if probability is not None else "regression"
            }

        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    async def get_model_summary(model_obj: Any, framework: str) -> Dict[str, Any]:
        """
        Get summary information about the model.
        Returns basic statistics and capabilities.
        """
        summary = {
            "framework": framework,
            "task_type": "unknown",
            "feature_count": 0,
            "is_valid": False,
            "error": None
        }

        try:
            # Validate model
            valid = await ModelLoaderService.validate_model(model_obj, framework)
            summary["is_valid"] = valid

            if valid:
                # Get model info
                info = await ModelLoaderService.get_model_info(model_obj, framework)
                summary.update(info)

                # Count features
                if framework == "sklearn" and hasattr(model_obj, 'n_features_in_'):
                    summary["feature_count"] = model_obj.n_features_in_
                elif framework == "xgboost" and hasattr(model_obj, 'feature_names'):
                    summary["feature_count"] = len(model_obj.feature_names)
                elif framework == "onnx":
                    summary["feature_count"] = len(model_obj.get_inputs())
                elif framework == "keras":
                    summary["feature_count"] = model_obj.input_shape[1] if len(model_obj.input_shape) > 1 else model_obj.input_shape[0]

        except Exception as e:
            summary["error"] = str(e)

        return summary