import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import traceback

try:
    from alibi.explainers import AnchorTabular, KernelShap
    HAS_ALIBI = True
except ImportError:
    HAS_ALIBI = False

logger = logging.getLogger(__name__)

class AlibiService:
    """Production-ready Alibi explainability service."""

    @staticmethod
    def create_explainer(
        model,
        framework: str,
        training_data: pd.DataFrame,
        feature_names: List[str],
        mode: str = "classification",
        categorical_columns: List[str] = None,
        categorical_labels: Dict[str, List[str]] = None
    ) -> Any:
        """
        Create an Alibi explainer (AnchorTabular for classification, KernelShap for regression).
        """
        if not HAS_ALIBI:
            raise ImportError("Alibi library not installed. Please install with 'pip install alibi'.")

        logger.info(f"Creating Alibi explainer: mode={mode}, framework={framework}")

        # ── Identify which columns are numeric so we can coerce the DataFrame correctly ──
        # AnchorTabular works on a FULLY NUMERIC training array.
        # Non-numeric columns must be integer-encoded before fitting.
        numeric_cols = [c for c in feature_names if pd.api.types.is_numeric_dtype(training_data[c])]
        categorical_col_idx = {}  # index -> list of string categories
        ordinal_maps = {}          # col_name -> {str_val: int_code}

        for i, col in enumerate(feature_names):
            if col not in numeric_cols:
                cats = sorted(training_data[col].astype(str).unique().tolist())
                categorical_col_idx[i] = cats
                ordinal_maps[col] = {cat: j for j, cat in enumerate(cats)}

        # Build a fully numeric training array
        train_numeric = training_data.copy()
        for col, mapping in ordinal_maps.items():
            train_numeric[col] = train_numeric[col].astype(str).map(mapping).fillna(0).astype(int)
        train_array = train_numeric[feature_names].values.astype(float)

        def predict_fn(x: np.ndarray) -> np.ndarray:
            """Wraps model prediction. Alibi always passes float numpy arrays."""
            try:
                if x.ndim == 1:
                    x = x.reshape(1, -1)

                # Re-build DataFrame with correct dtypes
                row_data = {}
                for i, col in enumerate(feature_names):
                    col_vals = x[:, i]
                    if i in categorical_col_idx:
                        # Map integer codes back to original string categories
                        cats = categorical_col_idx[i]
                        str_vals = []
                        for v in col_vals:
                            try:
                                idx = int(round(float(v)))
                                idx = max(0, min(idx, len(cats) - 1))
                                str_vals.append(cats[idx])
                            except:
                                str_vals.append(cats[0])
                        row_data[col] = str_vals
                    else:
                        try:
                            row_data[col] = col_vals.astype(float)
                        except:
                            row_data[col] = col_vals

                x_df = pd.DataFrame(row_data, columns=feature_names)

                if mode == "classification":
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(x_df)
                        return np.argmax(np.array(probs), axis=1)
                    else:
                        return np.array(model.predict(x_df)).astype(int)
                else:
                    return model.predict(x_df)

            except Exception as e:
                logger.error(f"Alibi predict_fn failed: {e}")
                logger.error(f"Input shape: {x.shape}, dtype: {x.dtype}")
                logger.error(traceback.format_exc())
                raise

        if mode == "classification":
            logger.info("Initializing Alibi AnchorTabular")
            logger.info(f"Categorical column indices: {list(categorical_col_idx.keys())}")

            explainer = AnchorTabular(
                predict_fn,
                feature_names=feature_names,
                categorical_names=categorical_col_idx
            )
            explainer.fit(train_array, disc_strategy='quintile')
        else:
            logger.info("Initializing Alibi KernelShap")
            explainer = KernelShap(predict_fn, link='identity' if mode == 'regression' else 'logit')
            explainer.fit(train_array)

        # Attach metadata for later use in explain_instance
        setattr(explainer, '_xai_mode', mode)
        setattr(explainer, '_xai_feature_names', feature_names)
        setattr(explainer, '_xai_ordinal_maps', ordinal_maps)
        setattr(explainer, '_xai_categorical_col_idx', categorical_col_idx)

        # Pipeline detection for logging only
        try:
            from sklearn.pipeline import Pipeline
            setattr(explainer, '_is_pipeline', isinstance(model, Pipeline))
        except ImportError:
            setattr(explainer, '_is_pipeline', False)

        return explainer

    @staticmethod
    def explain_instance(
        explainer_obj: Any,
        instance: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate a local explanation for a single instance."""
        if not HAS_ALIBI:
            raise ImportError("Alibi library not installed.")

        mode = getattr(explainer_obj, '_xai_mode', 'classification')
        feature_names = getattr(explainer_obj, '_xai_feature_names', feature_names)
        ordinal_maps = getattr(explainer_obj, '_xai_ordinal_maps', {})
        categorical_col_idx = getattr(explainer_obj, '_xai_categorical_col_idx', {})

        logger.info(f"Generating Alibi local explanation (mode={mode})")

        # Convert instance to numeric array for Alibi
        if isinstance(instance, list):
            instance = np.array(instance, dtype=object)

        # Encode categoricals in the instance to integers
        instance_numeric = []
        for i, col in enumerate(feature_names):
            val = instance[i] if i < len(instance) else 0
            if i in categorical_col_idx:
                cats = categorical_col_idx[i]
                mapping = {cat: j for j, cat in enumerate(cats)}
                str_val = str(val)
                instance_numeric.append(float(mapping.get(str_val, 0)))
            else:
                try:
                    instance_numeric.append(float(val))
                except:
                    instance_numeric.append(0.0)

        instance_2d = np.array(instance_numeric, dtype=float).reshape(1, -1)

        try:
            explanation = explainer_obj.explain(instance_2d)

            if mode == "classification":
                contributions = []
                for cond in explanation.data.get('names', []):
                    contributions.append({
                        "feature": cond.split(' ')[0] if ' ' in cond else cond,
                        "weight": 1.0,
                        "value": cond
                    })

                def _safe_float(val):
                    if isinstance(val, (list, np.ndarray)):
                        return float(val[0])
                    return float(val)

                return {
                    "intercept": 0.0,
                    "local_exp": {"0": [{"feature": c["feature"], "weight": c["weight"]} for c in contributions]},
                    "local_pred": _safe_float(explanation.data.get('prediction', 0)),
                    "list_of_contributions": contributions,
                    "anchor": {
                        "rule": " AND ".join(explanation.data.get('names', [])) or "No anchor found",
                        "conditions": explanation.data.get('names', []),
                        "precision": _safe_float(explanation.data.get('precision', 0.0)),
                        "coverage": _safe_float(explanation.data.get('coverage', 0.0)),
                        "prediction": str(explanation.data.get('prediction', ['Unknown'])[0]
                                         if isinstance(explanation.data.get('prediction'), list)
                                         else explanation.data.get('prediction', 'Unknown'))
                    },
                    "feature_importance": [
                        {"feature": f, "importance": 1.0 if f in [c["feature"] for c in contributions] else 0.0}
                        for f in feature_names
                    ]
                }
            else:
                # KernelShap
                try:
                    shap_vals_raw = explanation.data.get('shap_values', [[]])
                    if isinstance(shap_vals_raw, list) and len(shap_vals_raw) > 0:
                        shap_values = np.array(shap_vals_raw[0])
                    else:
                        shap_values = np.array(shap_vals_raw)
                        
                    # Flatten the shap_values to a 1D array to handle any nesting
                    shap_values = shap_values.flatten()
                    
                    expected_value = explanation.data.get('expected_value', [0])
                    if isinstance(expected_value, list) and len(expected_value) > 0:
                        expected_value = expected_value[0]
                except Exception as ex:
                    logger.error(f"Error extracting shap_values: {ex}")
                    shap_values = np.zeros(len(feature_names))
                    expected_value = 0.0

                def _safe_float(val):
                    try:
                        if isinstance(val, (list, np.ndarray)):
                            return float(np.asarray(val).flatten()[0])
                        return float(val)
                    except:
                        return 0.0

                importance = sorted([
                    {"feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                     "importance": _safe_float(abs(v))}
                    for i, v in enumerate(shap_values)
                ], key=lambda x: x["importance"], reverse=True)

                return {
                    "intercept": _safe_float(expected_value),
                    "feature_importance": importance,
                    "shap_values": [shap_values.tolist()],
                    "expected_value": _safe_float(expected_value),
                    "list_of_contributions": [
                        {"feature": feature_names[i], "weight": _safe_float(v),
                         "value": _safe_float(instance_numeric[i]) if i < len(instance_numeric) else 0.0}
                        for i, v in enumerate(shap_values)
                    ]
                }

        except Exception as e:
            logger.error(f"Alibi explain_instance failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    def explain_global(
        explainer: Any,
        model: Any,
        samples: pd.DataFrame,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """Generate global explanation by aggregating SHAP values."""
        if not HAS_ALIBI:
            raise ImportError("Alibi library not installed.")

        mode = getattr(explainer, '_xai_mode', 'regression')
        feature_names = getattr(explainer, '_xai_feature_names', list(samples.columns))
        categorical_col_idx = getattr(explainer, '_xai_categorical_col_idx', {})
        ordinal_maps = getattr(explainer, '_xai_ordinal_maps', {})

        logger.info(f"Generating Alibi global explanation (mode={mode})")

        try:
            # Encode samples numerically
            samples_numeric = samples.copy()
            for col, mapping in ordinal_maps.items():
                samples_numeric[col] = samples_numeric[col].astype(str).map(mapping).fillna(0).astype(float)

            explanation = explainer.explain(samples_numeric[feature_names].values)
            shap_values = np.array(explanation.data.get('shap_values', [[]])[0])
            mean_importance = np.mean(np.abs(shap_values), axis=0)

            importance = sorted([
                {"feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                 "importance": float(v)}
                for i, v in enumerate(mean_importance)
            ], key=lambda x: x["importance"], reverse=True)

            return {
                "feature_importance": importance,
                "num_samples_explained": len(samples)
            }
        except Exception as e:
            logger.error(f"Alibi explain_global failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# Singleton instance
alibi_service = AlibiService()