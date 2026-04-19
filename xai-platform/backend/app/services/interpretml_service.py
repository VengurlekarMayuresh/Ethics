import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import warnings
from interpret.blackbox import LimeTabular, MorrisSensitivity
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CategoricalEncoder:
    """Helper class to encode/decode categorical features for InterpretML."""
    def __init__(self, df: pd.DataFrame, explicit_categorical_cols: List[str] = None, categorical_labels: Dict[str, List[str]] = None):
        self.mappings = {}
        self.reverse_mappings = {}
        self.categorical_cols = explicit_categorical_cols if explicit_categorical_cols is not None else []
        self.categorical_labels = categorical_labels or {}
        
        # If no explicit cols provided, auto-detect
        if explicit_categorical_cols is None:
            for col in df.columns:
                is_cat = (
                    df[col].dtype == object or 
                    isinstance(df[col].dtype, pd.CategoricalDtype) or
                    (not df[col].empty and isinstance(df[col].dropna().iloc[0], str) if not df[col].dropna().empty else False)
                )
                if is_cat:
                    self.categorical_cols.append(col)
        
        for col in self.categorical_cols:
            if col in df.columns:
                if col in self.categorical_labels and self.categorical_labels[col]:
                    labels = self.categorical_labels[col]
                    # Map from label (string) to index
                    self.mappings[col] = {str(val): i for i, val in enumerate(labels)}
                    self.reverse_mappings[col] = {i: str(val) for i, val in enumerate(labels)}
                else:
                    unique_vals = sorted(df[col].dropna().unique().astype(str).tolist())
                    self.mappings[col] = {val: i for i, val in enumerate(unique_vals)}
                    self.reverse_mappings[col] = {i: val for i, val in enumerate(unique_vals)}
                
                # Add stable fallback for unknown/out-of-bound values
                self.reverse_mappings[col][len(self.reverse_mappings[col])] = "Unknown"
                
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns to numeric."""
        df_encoded = df.copy()
        for col in self.categorical_cols:
            if col not in df_encoded.columns:
                continue
                
            unknown_idx = len(self.reverse_mappings[col]) - 1
            
            def get_idx(val):
                s_val = str(val)
                # 1. Try mapping from string label to index
                if s_val in self.mappings[col]:
                    return self.mappings[col][s_val]
                
                # 2. Try treating val as a numeric index directly
                try:
                    i_val = int(float(val))
                    if i_val in self.reverse_mappings[col]:
                        return i_val
                except (ValueError, TypeError):
                    pass
                
                return unknown_idx

            df_encoded[col] = df_encoded[col].apply(get_idx)
        
        return df_encoded.fillna(0).astype(float)
        
    def decode_row(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Decode a single row's categorical values back to strings."""
        decoded = row_dict.copy()
        for col in self.categorical_cols:
            val = row_dict.get(col)
            if val is not None:
                try:
                    # InterpretML/LIME perturbs categorical indices.
                    # We round to the nearest integer index.
                    idx = int(round(float(val)))
                    # Fallback to string if index is out of bounds (which shouldn't happen with proper LIME config)
                    decoded[col] = self.reverse_mappings[col].get(idx, str(val))
                except (ValueError, TypeError):
                    pass
        return decoded

class WrappedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, predict_proba_fn, classes=None, n_features_in=None):
        self.predict_proba_fn = predict_proba_fn
        self._estimator_type = "classifier"
        if classes is not None:
            self.classes_ = classes
        if n_features_in is not None:
            self.n_features_in_ = n_features_in

    def __call__(self, X):
        return self.predict_proba_fn(X)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return self.predict_proba_fn(X)

class WrappedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, predict_fn, n_features_in=None):
        self.predict_fn = predict_fn
        self._estimator_type = "regressor"
        if n_features_in is not None:
            self.n_features_in_ = n_features_in

    def __call__(self, X):
        return self.predict_fn(X)

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.predict_fn(X)

class InterpretMLService:
    @staticmethod
    def _is_glassbox(model) -> bool:
        return isinstance(model, (ExplainableBoostingClassifier, ExplainableBoostingRegressor))

    @staticmethod
    def create_explainer(
        model,
        framework: str,
        training_data: pd.DataFrame,
        feature_names: List[str],
        mode: str = "regression",
        categorical_columns: List[str] = None,
        categorical_labels: Dict[str, List[str]] = None
    ) -> Any:
        logger.info(f"Creating InterpretML explainer for framework: {framework}, mode: {mode}")
        
        if InterpretMLService._is_glassbox(model):
            return {
                "type": "glassbox",
                "model": model,
                "feature_names": feature_names,
                "mode": mode
            }
        
        encoder = CategoricalEncoder(training_data, explicit_categorical_cols=categorical_columns, categorical_labels=categorical_labels)
        numeric_training_data = encoder.encode(training_data)
        
        # Safety: Fill any leftover NaNs in numeric fields (e.g. from the original numeric cols)
        # to satisfy interpret's 'unify_data' requirements.
        numeric_training_data = numeric_training_data.fillna(0)
        
        classes = getattr(model, "classes_", None)
        n_features_in = getattr(model, "n_features_in_", len(feature_names))

        if mode == "classification":
            def predict_proba_fn(x):
                x_df = pd.DataFrame(x, columns=feature_names) if isinstance(x, np.ndarray) else x
                decoded_rows = [encoder.decode_row(row.to_dict()) for _, row in x_df.iterrows()]
                return model.predict_proba(pd.DataFrame(decoded_rows))
            wrapped_model = WrappedClassifier(predict_proba_fn, classes=classes, n_features_in=n_features_in)
        else:
            def predict_fn(x):
                x_df = pd.DataFrame(x, columns=feature_names) if isinstance(x, np.ndarray) else x
                decoded_rows = [encoder.decode_row(row.to_dict()) for _, row in x_df.iterrows()]
                return model.predict(pd.DataFrame(decoded_rows))
            wrapped_model = WrappedRegressor(predict_fn, n_features_in=n_features_in)

        feature_types = []
        categorical_features = []
        categorical_names = {}
        for i, col in enumerate(feature_names):
            if col in encoder.categorical_cols:
                feature_types.append('nominal')
                categorical_features.append(i)
                categorical_names[i] = [
                    encoder.reverse_mappings[col].get(j, str(j))
                    for j in range(len(encoder.reverse_mappings[col]))
                ]
            else:
                feature_types.append('continuous')

        local_explainer = None
        global_explainer = None

        try:
            local_explainer = LimeTabular(
                wrapped_model, 
                numeric_training_data, 
                feature_names=feature_names, 
                feature_types=feature_types,
                categorical_features=categorical_features,
                categorical_names=categorical_names
            )
        except Exception as e:
            logger.error(f"Failed to create InterpretML LimeTabular explainer: {e}")

        try:
            # MorrisSensitivity requires a minimum number of samples. 
            # We wrap it in try-except to avoid failing the whole process if background data is too small.
            global_explainer = MorrisSensitivity(
                wrapped_model, 
                numeric_training_data, 
                feature_names=feature_names, 
                feature_types=feature_types
            )
        except Exception as e:
            logger.error(f"Failed to create InterpretML MorrisSensitivity explainer: {e}")
        
        return {
            "type": "blackbox",
            "local_explainer": local_explainer,
            "global_explainer": global_explainer,
            "feature_names": feature_names,
            "mode": mode,
            "model": model,
            "encoder": encoder,
            "categorical_features": categorical_features
        }

    @staticmethod
    def explain_instance(
        explainer_obj: Any,
        instance: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        instance_df = pd.DataFrame([instance], columns=feature_names)
        
        if explainer_obj["type"] == "glassbox":
            try:
                local_exp = explainer_obj["model"].explain_local(instance_df)
            except Exception as e:
                logger.error(f"Glassbox explain_local failed: {e}")
                raise
        else:
            if explainer_obj.get("local_explainer") is None:
                logger.error("InterpretML Local Explainer (LIME) is missing - initialization must have failed.")
                # We can't proceed without a local explainer
                raise ValueError("InterpretML Local Explainer (LIME) was not initialized correctly.")
            
            instance_encoded = explainer_obj["encoder"].encode(instance_df)
            local_exp = explainer_obj["local_explainer"].explain_local(instance_encoded)

        exp_data = local_exp.data(0)
        names = exp_data.get('names', [])
        scores = exp_data.get('scores', [])
        values = exp_data.get('values', instance)
        extra = exp_data.get('extra', {})

        contributions = []
        for name, score, val in zip(names, scores, values):
            display_val = val
            if explainer_obj["type"] != "glassbox":
                encoder = explainer_obj["encoder"]
                if name in encoder.categorical_cols:
                    try:
                        idx = int(round(float(val)))
                        display_val = encoder.reverse_mappings[name].get(idx, str(val))
                    except (ValueError, TypeError):
                        pass
            
            if display_val is None or display_val == '':
                display_val = 'N/A'

            contributions.append({
                "feature": name,
                "weight": float(score),
                "value": str(display_val)
            })

        contributions.sort(key=lambda x: abs(x["weight"]), reverse=True)

        return {
            "intercept": float(extra.get('intercept', 0.0)) if isinstance(extra.get('intercept'), (int, float, np.number)) else 0.0,
            "local_exp": {"0": [{"feature": c["feature"], "weight": c["weight"]} for c in contributions]},
            "local_pred": float(extra.get('prediction', 0.0)) if isinstance(extra.get('prediction'), (int, float, np.number)) else 0.0,
            "list_of_contributions": contributions,
            "feature_importance": [
                {"feature": c["feature"], "importance": abs(c["weight"])}
                for c in contributions
            ]
        }

    @staticmethod
    def explain_global(explainer_obj, model, samples, num_features=10, num_samples=5000):
        explainer = explainer_obj.get("model") if explainer_obj["type"] == "glassbox" else explainer_obj.get("global_explainer")
        
        if explainer is None:
            logger.warning("Global explainer not available for this InterpretML session.")
            return {"feature_importance": [], "num_samples_explained": 0, "error": "Global explainer not available"}

        try:
            global_exp = explainer.explain_global()
            exp_data = global_exp.data()
            names, scores = exp_data.get('names', []), exp_data.get('scores', [])
            importance = sorted([{"feature": n, "importance": float(abs(s))} for n, s in zip(names, scores)],
                                key=lambda x: x["importance"], reverse=True)
            return {"feature_importance": importance[:num_features], "num_samples_explained": len(samples)}
        except Exception as e:
            logger.error(f"InterpretML explain_global failed: {e}")
            return {"feature_importance": [], "num_samples_explained": 0, "error": str(e)}

interpretml_service = InterpretMLService()
