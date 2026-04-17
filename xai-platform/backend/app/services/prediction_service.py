import joblib
import io
import json
import numpy as np
import pandas as pd
import onnxruntime
import xgboost as xgb
import importlib
import inspect
from typing import Any, Dict, List, Union, Optional
from pathlib import Path
from app.utils.file_handler import storage
from app.models.model_meta import FeatureSchema
from app.config import settings

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
    def _inject_custom_pickle_classes(cls) -> None:
        """
        Load user-configured modules and inject class symbols into __main__.
        This helps unpickle sklearn/joblib models saved from notebook/script scopes.
        """
        modules_raw = (settings.PICKLE_CLASS_MODULES or "").strip()
        if not modules_raw:
            return

        module_names = [m.strip() for m in modules_raw.split(",") if m.strip()]
        if not module_names:
            return

        import __main__ as main_module

        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
                for symbol_name, symbol in vars(module).items():
                    if symbol_name.startswith("_"):
                        continue
                    if inspect.isclass(symbol) and not hasattr(main_module, symbol_name):
                        setattr(main_module, symbol_name, symbol)
            except Exception as e:
                print(f"Warning: Failed to import module {module_name} for pickle injection: {e}")

    @classmethod
    async def load_model(cls, model_path: str) -> tuple[Any, str]:
        """Load model from storage and return model object with framework."""
        try:
            model_file = await storage.download_file(model_path)

            framework = cls.detect_framework(model_path)

            if framework == "sklearn":
                try:
                    cls._inject_custom_pickle_classes()
                    return joblib.load(io.BytesIO(model_file)), framework
                except Exception as e:
                    msg = str(e)
                    if "Can't get attribute" in msg:
                        raise ValueError(
                            "Missing custom class during unpickle. "
                            "Check PICKLE_CLASS_MODULES in .env. "
                            f"Original error: {msg}"
                        )
                    raise
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
                    # Validate min/max constraints if defined
                    value = float(df[feature_name].iloc[0])
                    if schema.min is not None and value < schema.min:
                        raise ValueError(f"Feature '{feature_name}' must be >= {schema.min}")
                    if schema.max is not None and value > schema.max:
                        raise ValueError(f"Feature '{feature_name}' must be <= {schema.max}")
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

            # Prepare result and sanitize for JSON/BSON serialization
            # This prevents the 'NaN' issue caused by high-precision numpy floats
            final_prediction = prediction.tolist() if isinstance(prediction, (np.ndarray, list)) else prediction
            
            sanitized_proba = None
            if prediction_proba is not None:
                # Handle both 1D and 2D probability arrays
                safe_proba = np.nan_to_num(prediction_proba, nan=0.0)
                if hasattr(safe_proba, "tolist"):
                    sanitized_proba = safe_proba.tolist()
                else:
                    sanitized_proba = list(safe_proba)

            return {
                "prediction": final_prediction,
                "probability": sanitized_proba
            }

        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

    @staticmethod
    def format_prediction_result(result: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format prediction result for API response."""
        probs = result.get("probability")
        
        # Flatten probabilities if they come in as [[p1, p2]]
        if probs and isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], list):
            probs = probs[0]

        # Ensure all probabilities are clean floats for JSON/BSON
        if probs:
            probs = [float(np.nan_to_num(p, nan=0.0)) for p in probs]

        final_prediction = result["prediction"]
        if isinstance(final_prediction, (list, np.ndarray)) and len(final_prediction) > 0:
            final_prediction = final_prediction[0]

        formatted = {
            "input_data": input_data,
            "prediction": final_prediction,
            "prediction_label": str(final_prediction),
            "probabilities": probs,
            "prediction_confidence": 0.0
        }

        # Calculate confidence if probabilities exist
        if formatted["probabilities"]:
            try:
                # Use max probability as the confidence score
                formatted["prediction_confidence"] = float(np.nan_to_num(max(formatted["probabilities"]), nan=0.0))
            except Exception as e:
                print(f"Error calculating confidence: {e}")
                formatted["prediction_confidence"] = 0.0

        return formatted