import joblib
import io
import importlib
import inspect
import numpy as np
import pandas as pd
import onnxruntime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from app.utils.file_handler import storage
from app.models.model_meta import FeatureSchema
from app.config import settings

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
    def _inject_custom_pickle_classes() -> None:
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
            module = importlib.import_module(module_name)
            for symbol_name, symbol in vars(module).items():
                if symbol_name.startswith("_"):
                    continue
                if inspect.isclass(symbol) and not hasattr(main_module, symbol_name):
                    setattr(main_module, symbol_name, symbol)

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
                try:
                    ModelLoaderService._inject_custom_pickle_classes()
                    return joblib.load(io.BytesIO(model_file)), framework
                except Exception as e:
                    msg = str(e)
                    if "Can't get attribute" in msg:
                        raise ValueError(
                            "Missing custom class during unpickle. "
                            "The model was likely saved from notebook/script scope (__main__). "
                            "Move that class into an importable module and set PICKLE_CLASS_MODULES "
                            "to include that module path (comma-separated), then retry. "
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

    @staticmethod
    async def load_model_from_bytes(model_bytes: bytes, model_filename: str) -> tuple[Any, str]:
        """
        Load model from bytes and return model object with framework.
        Useful for analyzing model before storing.
        """
        try:
            framework = ModelLoaderService.detect_framework(model_filename)

            if framework == "sklearn":
                try:
                    ModelLoaderService._inject_custom_pickle_classes()
                    return joblib.load(io.BytesIO(model_bytes)), framework
                except Exception as e:
                    msg = str(e)
                    if "Can't get attribute" in msg:
                        raise ValueError(
                            "Missing custom class during unpickle. "
                            "The model was likely saved from notebook/script scope (__main__). "
                            "Move that class into an importable module and set PICKLE_CLASS_MODULES "
                            "to include that module path (comma-separated), then retry. "
                            f"Original error: {msg}"
                        )
                    raise
            elif framework == "xgboost":
                import xgboost as xgb
                # XGBoost load_model requires a file path; write bytes to a temporary file.
                import tempfile
                # Use delete=True so the temp file is automatically cleaned up.
                with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
                    tmp.write(model_bytes)
                    tmp.flush()
                    model = xgb.Booster()
                    model.load_model(tmp.name)
                return model, framework
            elif framework == "onnx":
                return onnxruntime.InferenceSession(model_bytes), framework
            elif framework == "keras":
                from tensorflow import keras
                return keras.models.model_from_json(model_bytes.decode('utf-8')), framework
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
    async def generate_feature_schema(
        model_obj: Any,
        framework: str,
        dataset_analysis: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[FeatureSchema]:
        """
        Generate feature schema from model object and optionally enhance with dataset analysis.
        Returns list of FeatureSchema objects with name, type, and constraints (min, max, mean, options).

        CRITICAL: For pipelines, we need RAW input feature names (what users will provide),
        NOT the preprocessed features that the final model expects.

        Args:
            model_obj: Loaded model object
            framework: Detected framework string
            dataset_analysis: Optional dict from analyze_dataset() with feature statistics
        """
        features = []
        feature_names = []

        def _infer_feature_count(obj: Any, fw: str) -> int:
            """Best-effort inference of input feature count when names are unavailable."""
            try:
                if fw == "sklearn":
                    if hasattr(obj, "n_features_in_"):
                        return int(obj.n_features_in_)
                    from sklearn.pipeline import Pipeline
                    if isinstance(obj, Pipeline) and obj.steps:
                        final_model = obj.steps[-1][1]
                        if hasattr(final_model, "n_features_in_"):
                            return int(final_model.n_features_in_)

                elif fw == "xgboost":
                    # xgboost.Booster commonly exposes num_features().
                    if hasattr(obj, "num_features") and callable(obj.num_features):
                        return int(obj.num_features())
                    if hasattr(obj, "feature_names") and obj.feature_names:
                        return int(len(obj.feature_names))

                elif fw == "onnx":
                    if hasattr(obj, "get_inputs"):
                        inputs = obj.get_inputs()
                        if inputs:
                            shape = getattr(inputs[0], "shape", None)
                            if shape and len(shape) > 1 and isinstance(shape[1], int):
                                return int(shape[1])

                elif fw == "keras":
                    shape = getattr(obj, "input_shape", None)
                    if shape and len(shape) > 1 and isinstance(shape[1], int):
                        return int(shape[1])
            except Exception:
                return 0

            return 0

        # Step 1: Extract RAW input feature names (before preprocessing) and their types
        feature_types = {}  # Map feature_name -> "categorical" or "numeric"
        if framework == "sklearn":
            from sklearn.pipeline import Pipeline

            # Check if model is a Pipeline
            if isinstance(model_obj, Pipeline):
                # For pipelines, get feature names from the preprocessing step
                preprocessing_step = None
                for step_name, step in model_obj.steps:
                    if hasattr(step, 'transformers_') or hasattr(step, 'feature_names_in_'):
                        preprocessing_step = step
                        break

                if preprocessing_step and hasattr(preprocessing_step, 'feature_names_in_'):
                    # ColumnTransformer or similar: get the original feature names
                    feature_names = preprocessing_step.feature_names_in_.tolist()

                    # Infer feature types by examining the ColumnTransformer's transformers
                    # Categorical features are those handled by OneHotEncoder or similar
                    if hasattr(preprocessing_step, 'transformers_'):
                        for transformer_name, transformer_obj, cols in preprocessing_step.transformers_:
                            # transformer_obj could be OneHotEncoder, StandardScaler, etc.
                            if isinstance(transformer_obj, type) or hasattr(transformer_obj, '__module__'):
                                # Check if it's a categorical transformer (OneHotEncoder)
                                if 'OneHotEncoder' in transformer_obj.__class__.__name__:
                                    for col in cols:
                                        feature_types[col] = "categorical"
                                elif any(num_type in transformer_obj.__class__.__name__ for num_type in ['StandardScaler', 'MinMaxScaler', 'Normalizer', 'MaxAbsScaler']):
                                    for col in cols:
                                        feature_types[col] = "numeric"
                                else:
                                    # Unknown transformer, mark as numeric by default
                                    for col in cols:
                                        feature_types[col] = "numeric"
                            else:
                                # Assume numeric for safety
                                for col in cols:
                                    feature_types[col] = "numeric"
                    else:
                        # Can't determine from transformers, use dataset analysis or defaults
                        pass

                elif hasattr(model_obj, 'feature_names_in_'):
                    # Pipeline itself might have feature names
                    feature_names = model_obj.feature_names_in_.tolist()
                else:
                    # Fallback
                    if dataset_analysis:
                        feature_names = list(dataset_analysis.keys())
                    else:
                        final_model = model_obj.steps[-1][1] if model_obj.steps else model_obj
                        n_features = getattr(final_model, 'n_features_in_', None)
                        if n_features:
                            feature_names = [f"feature_{i}" for i in range(n_features)]
            else:
                # Regular sklearn model (not a pipeline)
                if hasattr(model_obj, 'feature_names_in_'):
                    feature_names = model_obj.feature_names_in_.tolist()
                elif hasattr(model_obj, 'n_features_in_'):
                    feature_names = [f"feature_{i}" for i in range(model_obj.n_features_in_)]

        elif framework == "xgboost":
            if hasattr(model_obj, 'feature_names'):
                feature_names = model_obj.feature_names
        elif framework == "onnx":
            # ONNX: Use input tensor names
            feature_names = [inp.name for inp in model_obj.get_inputs()]
        elif framework == "keras":
            # Keras: Use input shape to determine number of features
            input_shape = model_obj.input_shape
            if input_shape and len(input_shape) > 1:
                num_features = input_shape[1] if len(input_shape) > 1 else input_shape[0]
                feature_names = [f"feature_{i}" for i in range(num_features)]

        # Step 1.5: Final fallback when explicit names are unavailable.
        if not feature_names:
            if dataset_analysis:
                feature_names = list(dataset_analysis.keys())
            else:
                fallback_count = _infer_feature_count(model_obj, framework)
                if fallback_count > 0:
                    feature_names = [f"feature_{i}" for i in range(fallback_count)]

        # Step 2: Build FeatureSchema objects, using dataset_analysis if available
        for name in feature_names:
            # Determine feature type: prioritize dataset_analysis, then inferred types, then default
            feature_type = "numeric"  # default
            options = []

            if dataset_analysis and name in dataset_analysis:
                analysis = dataset_analysis[name]
                feature_type = analysis.get("type", "numeric")
                options = analysis.get("options", [])
            elif name in feature_types:
                feature_type = feature_types[name]
                # For categorical, we need to extract options from the transformer if possible
                if feature_type == "categorical":
                    # Try to get categories from the preprocessor
                    if preprocessing_step and hasattr(preprocessing_step, 'transformers_'):
                        for transformer_name, transformer_obj, cols in preprocessing_step.transformers_:
                            if name in cols and hasattr(transformer_obj, 'categories_'):
                                # Find index of this column in cols
                                idx = list(cols).index(name)
                                if idx < len(transformer_obj.categories_):
                                    # Get unique values for this column, convert to strings
                                    raw_cats = transformer_obj.categories_[idx]
                                    options = [str(cat) for cat in raw_cats]
                                break
            else:
                # No type info, default numeric
                feature_type = "numeric"

            feature = FeatureSchema(
                name=name,
                type=feature_type,
                options=options,
                min=None,
                max=None,
                mean=None
            )
            features.append(feature)

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
    async def analyze_dataset(dataset_bytes: bytes) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a CSV dataset to extract feature information.
        Returns dict mapping feature names to their properties:
        - type: "numeric" or "categorical"
        - min: float (for numeric)
        - max: float (for numeric)
        - mean: float (for numeric)
        - options: List[str] (for categorical)
        """
        try:
            df = pd.read_csv(io.BytesIO(dataset_bytes))
            analysis = {}

            for column in df.columns:
                col_data = df[column]
                # Determine if numeric or categorical
                if pd.api.types.is_numeric_dtype(col_data):
                    # Numeric feature
                    analysis[column] = {
                        "type": "numeric",
                        "min": float(col_data.min()) if not pd.isna(col_data.min()) else None,
                        "max": float(col_data.max()) if not pd.isna(col_data.max()) else None,
                        "mean": float(col_data.mean()) if not pd.isna(col_data.mean()) else None,
                        "options": []
                    }
                else:
                    # Categorical feature
                    unique_vals = col_data.dropna().unique().tolist()
                    # Convert all to strings
                    unique_vals_str = [str(val) for val in unique_vals]
                    analysis[column] = {
                        "type": "categorical",
                        "min": None,
                        "max": None,
                        "mean": None,
                        "options": unique_vals_str
                    }

            return analysis

        except Exception as e:
            raise ValueError(f"Failed to analyze dataset: {str(e)}")

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

    @staticmethod
    async def detect_model_category(model_obj: Any, framework: str) -> str:
        """
        Detect the category of the model for explainability selection.
        Returns: "linear", "tree", "neural_network", or "unknown"
        """
        try:
            if framework == "sklearn":
                # Check for linear models
                from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
                linear_models = [
                    LinearRegression, LogisticRegression, Ridge,
                    Lasso, ElasticNet
                ]
                model_class = type(model_obj).__name__
                if any(linear in model_class for linear in ["LinearRegression", "LogisticRegression", "Ridge", "Lasso", "ElasticNet"]):
                    return "linear"

                # Check for tree-based models
                from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
                tree_models = [
                    DecisionTreeClassifier, DecisionTreeRegressor,
                    RandomForestClassifier, RandomForestRegressor,
                    GradientBoostingClassifier, GradientBoostingRegressor
                ]
                if any(tree in model_class for tree in ["DecisionTree", "RandomForest", "GradientBoosting", "ExtraTrees"]):
                    return "tree"

                # XGBoost, LightGBM, CatBoost (if available) are tree-based
                if "XGB" in model_class or "LGBM" in model_class or "CatBoost" in model_class:
                    return "tree"

            elif framework == "xgboost":
                # XGBoost models are tree-based by default (unless linear booster)
                if hasattr(model_obj, 'config'):
                    # Check booster type
                    config = model_obj.save_config()
                    if 'tree_method' in config or 'booster' in config.get('learner', {}).get('gradient_booster', {}).get('name', ''):
                        return "tree"
                return "tree"  # Assume tree if can't determine

            elif framework == "keras" or framework == "pytorch":
                # Neural networks
                return "neural_network"

            # Check if model has a coef_ attribute (typically linear models)
            if hasattr(model_obj, 'coef_'):
                return "linear"

            # Check if model has feature_importances_ (tree-based)
            if hasattr(model_obj, 'feature_importances_'):
                return "tree"

        except Exception:
            pass

        return "unknown"
