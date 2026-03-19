import joblib
import io
import json
import numpy as np
import pandas as pd
import onnxruntime
from typing import Any, Dict, List, Union
from pathlib import Path
from app.utils.file_handler import storage
from app.models.model_meta import FeatureSchema

class ModelLoader:
    """Framework-agnostic model loader for various ML frameworks."""

    SUPPORTED_FORMATS = {
        ".pkl": "sklearn",
        ".joblib": "sklearn",
        ".json": "xgboost",
        ".onnx": "onnx",
        ".h5": "keras",
        ".pt": "pytorch"
    }

    @classmethod
    def detect_framework(cls, model_file: str) -> str:
        """Detect ML framework from file extension."""
        ext = Path(model_file).suffix.lower()
        return cls.SUPPORTED_FORMATS.get(ext, "unknown")

    @classmethod
    async def load_model(cls, model_path: str) -> tuple[Any, str]:
        """Load model from storage and return model object with framework."""
        try:
            model_file = await storage.download_file(model_path)

            framework = cls.detect_framework(model_path)

            if framework == "sklearn":
                return joblib.load(io.BytesIO(model_file)), framework
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

class PredictionService:
    """Service for making predictions with loaded models."""

    @staticmethod
    async def validate_input(input_data: Dict[str, Any], feature_schema: List[FeatureSchema]) -> pd.DataFrame:
        """Validate input data against feature schema and convert to DataFrame."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([input_data])

            # Validate each feature
            for schema in feature_schema:
                feature_name = schema.name
                feature_type = schema.type

                if feature_name not in df.columns:
                    raise ValueError(f"Missing required feature: {feature_name}")

                if feature_type == "numeric":
                    try:
                        df[feature_name] = pd.to_numeric(df[feature_name])
                    except (ValueError, TypeError):
                        raise ValueError(f"Feature '{feature_name}' must be numeric")
                elif feature_type == "categorical":
                    if schema.options and df[feature_name][0] not in schema.options:
                        raise ValueError(f"Value for '{feature_name}' must be one of: {schema.options}")

            return df

        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")

    @staticmethod
    async def make_prediction(model_obj: Any, framework: str, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction using the loaded model."""
        try:
            if framework == "sklearn":
                prediction = model_obj.predict(input_data)
                prediction_proba = model_obj.predict_proba(input_data) if hasattr(model_obj, 'predict_proba') else None

            elif framework == "xgboost":
                dmatrix = xgb.DMatrix(input_data)
                prediction = model_obj.predict(dmatrix)
                prediction_proba = model_obj.predict_proba(dmatrix) if hasattr(model_obj, 'predict_proba') else None

            elif framework == "onnx":
                input_name = model_obj.get_inputs()[0].name
                input_array = input_data.values.astype(np.float32)
                outputs = model_obj.run(None, {input_name: input_array})
                prediction = outputs[0]

                # Check if output is probability (softmax)
                if prediction.shape[1] > 1 and np.all(prediction >= 0) and np.all(prediction <= 1):
                    prediction_proba = prediction
                    prediction = np.argmax(prediction, axis=1)
                else:
                    prediction_proba = None

            else:
                raise ValueError(f"Prediction not implemented for framework: {framework}")

            # Prepare result
            result = {
                "prediction": prediction.tolist() if isinstance(prediction, (np.ndarray, list)) else prediction,
                "probability": prediction_proba.tolist() if prediction_proba is not None else None
            }

            return result

        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

    @staticmethod
    def format_prediction_result(result: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format prediction result for API response."""
        formatted = {
            "input_data": input_data,
            "prediction": result["prediction"],
            "prediction_label": result["prediction"] if isinstance(result["prediction"], (str, int, float)) else None,
            "probabilities": result["probability"] if result["probability"] else None,
            "prediction_confidence": None
        }

        # Calculate confidence if probabilities exist
        if formatted["probabilities"]:
            if isinstance(formatted["probabilities"][0], list):
                # Multi-class probabilities
                formatted["prediction_confidence"] = max(formatted["probabilities"])
            elif isinstance(formatted["probabilities"], list):
                # Binary class probabilities - use positive class probability
                formatted["prediction_confidence"] = formatted["probabilities"][-1]

        return formatted