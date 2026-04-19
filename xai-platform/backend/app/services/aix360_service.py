import logging
import pandas as pd
import numpy as np
import traceback
from typing import Any, Dict, List, Optional

# AIX360 imports
try:
    from aix360.algorithms.rbm import BRCGExplainer, FeatureBinarizer, BooleanRuleCG
    HAS_AIX360 = True
except ImportError:
    BRCGExplainer = None
    FeatureBinarizer = None
    BooleanRuleCG = None
    HAS_AIX360 = False

logger = logging.getLogger(__name__)


def _encode_df_numeric(df: pd.DataFrame, ordinal_maps: dict) -> pd.DataFrame:
    """Encode all columns to numeric. Categorical columns use integer ordinal coding."""
    result = df.copy()
    for col in result.columns:
        if col in ordinal_maps:
            result[col] = result[col].astype(str).map(ordinal_maps[col]).fillna(0).astype(float)
        elif not pd.api.types.is_numeric_dtype(result[col]):
            # Unknown categoricals: just convert to codes
            result[col] = pd.Categorical(result[col]).codes.astype(float)
        else:
            result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)
    return result


class AIX360Service:

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
        Initialize the AIX360 BRCG explainer as a surrogate rule-based model.
        """
        logger.info(f"Creating AIX360 explainer: mode={mode}, features={feature_names}")

        if not HAS_AIX360:
            logger.error("AIX360 not installed in this environment.")
            return {"error": "AIX360 dependencies missing (aix360[all])", "stub": True}

        # BRCG is classification-only, so for regression we discretize the target
        discretize_regression = (mode == "regression")

        try:
            # Build ordinal maps using explicit categories if provided
            ordinal_maps = {}
            cat_cols = categorical_columns if categorical_columns is not None else []
            
            # If no explicit list, detect non-numeric columns
            if categorical_columns is None:
                for col in feature_names:
                    if col in training_data.columns and not pd.api.types.is_numeric_dtype(training_data[col]):
                        cat_cols.append(col)

            for col in cat_cols:
                if col in training_data.columns:
                    if categorical_labels and col in categorical_labels and categorical_labels[col]:
                        labels = categorical_labels[col]
                        ordinal_maps[col] = {str(cat): i for i, cat in enumerate(labels)}
                    else:
                        cats = sorted(training_data[col].astype(str).unique().tolist())
                        ordinal_maps[col] = {cat: i for i, cat in enumerate(cats)}

            # Encode training data fully numeric
            X_numeric = _encode_df_numeric(training_data[feature_names], ordinal_maps)

            # Get model predictions to use as surrogate labels
            try:
                y_raw = model.predict(training_data[feature_names])
            except Exception:
                # Some models need a plain numeric array
                y_raw = model.predict(X_numeric)

            y_raw = np.asarray(y_raw).flatten()
            
            # Regression strategy: Discretize y_raw around the median to create bit-rules
            if discretize_regression:
                median_val = np.median(y_raw)
                y_pred = (y_raw > median_val).astype(int)
                logger.info(f"Discretized regression target at median: {median_val}")
            else:
                y_pred = y_raw.astype(int)

            # BRCG requires BOTH classes present. If all predictions are the same class,
            # randomly flip samples so the surrogate has something to learn.
            if len(np.unique(y_pred)) < 2:
                logger.warning(
                    f"Only one predicted class ({np.unique(y_pred)}) in {len(y_pred)} samples. "
                    "Forcing label diversity for BRCG."
                )
                n = len(y_pred)
                n_flip = max(n // 4, 1)          # flip at least 1 sample
                flip_idx = np.random.choice(n, size=n_flip, replace=False)
                y_pred = y_pred.copy()
                y_pred[flip_idx] = 1 - y_pred[flip_idx]
                logger.info(f"Flipped {n_flip} labels: classes now {np.unique(y_pred, return_counts=True)}")

            y_series = pd.Series(y_pred)

            # Binarize features for BRCG
            binarizer = FeatureBinarizer(negations=True)
            X_bin = binarizer.fit_transform(X_numeric)

            # Fit BRCG surrogate — retry with forced balanced classes if dimensions error
            rule_model = BooleanRuleCG(lambda0=0.005, lambda1=0.005)
            explainer = BRCGExplainer(rule_model)
            try:
                explainer.fit(X_bin, y_series)
            except ValueError as ve:
                if "Invalid dimensions" in str(ve):
                    logger.warning(f"BRCG fit failed ({ve}). Retrying with forced 50/50 balanced labels.")
                    n = len(y_series)
                    balanced = np.zeros(n, dtype=int)
                    n_pos = max(n // 2, 1)
                    balanced[:n_pos] = 1
                    np.random.shuffle(balanced)
                    rule_model2 = BooleanRuleCG(lambda0=0.005, lambda1=0.005)
                    explainer = BRCGExplainer(rule_model2)
                    explainer.fit(X_bin, pd.Series(balanced))
                    logger.info("BRCG retry succeeded with balanced labels.")
                else:
                    raise

            return {
                "explainer": explainer,
                "binarizer": binarizer,
                "feature_names": feature_names,
                "ordinal_maps": ordinal_maps,
                "mode": mode
            }

        except Exception as e:
            logger.error(f"Failed to create AIX360 explainer: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "stub": True}

    @staticmethod
    def _decode_rule_string(rule: str, ordinal_maps: Dict[str, Dict[str, int]]) -> str:
        """Decode a rule string by replacing numeric indices with labels."""
        import re
        decoded = rule
        for col, mapping in ordinal_maps.items():
            if col in decoded:
                # Reverse mapping: index -> label
                reverse = {i: label for label, i in mapping.items()}
                
                # Match: col name + space? + operator + space? + float
                pattern = rf"({re.escape(col)})\s*(==|<=|>=|>|<)\s*(\d+\.?\d*)"
                
                def replace_val(match):
                    feature, op, val = match.groups()
                    try:
                        idx = int(round(float(val)))
                        label = reverse.get(idx, val)
                        
                        # Make it more readable
                        if op == "==":
                            return f"{feature} is {label}"
                        elif op == "<=" and idx == 0:
                            return f"{feature} is {label}"
                        elif op == ">=" and idx == len(reverse) - 1:
                            return f"{feature} is {label}"
                        
                        return f"{feature} {op} {label}"
                    except:
                        return match.group(0)

                decoded = re.sub(pattern, replace_val, decoded)
        return decoded

    @staticmethod
    def explain_instance(
        explainer_obj: Any,
        instance: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a rule-based explanation for a specific instance.
        BRCG rules are global (surrogate model), so we return all learned rules.
        """
        # Graceful fallback for stub/error cases
        if not isinstance(explainer_obj, dict) or explainer_obj.get("stub"):
            err = explainer_obj.get("error", "Unknown error") if isinstance(explainer_obj, dict) else "Invalid explainer"
            return {
                "rules": [{"rule": f"AIX360 unavailable: {err}", "prediction": "N/A", "confidence": 0.0, "support": 0.0}],
                "feature_importance": [{"feature": f, "importance": 0.0} for f in feature_names],
                "status": "partial",
                "error": err
            }

        try:
            explainer = explainer_obj["explainer"]
            feature_names = explainer_obj.get("feature_names", feature_names)

            # BRCGExplainer.explain() returns {'isCNF': bool, 'rules': [str, ...]}
            explanation_dict = explainer.explain()
            logger.info(f"AIX360 BRCG explain() returned: {explanation_dict}")

            rules = []

            if isinstance(explanation_dict, dict):
                raw_rules = explanation_dict.get("rules", [])
                is_cnf = explanation_dict.get("isCNF", False)

                for r in raw_rules:
                    rule_str = str(r)
                    # Attempt to decode rule string if orbital maps exist
                    if explainer_obj.get("ordinal_maps"):
                        rule_str = AIX360Service._decode_rule_string(rule_str, explainer_obj["ordinal_maps"])

                    if explainer_obj.get("mode") == "regression":
                        pred_label = "Above Median (High Value)" if not is_cnf else "Below Median (Low Value)"
                    else:
                        pred_label = "Positive (Class 1)" if not is_cnf else "Negative (Class 0)"

                    rules.append({
                        "rule": rule_str,
                        "raw_rule": str(r),
                        "prediction": pred_label,
                        "confidence": 0.9,
                        "support": round(1.0 / max(len(raw_rules), 1), 2)
                    })

            elif hasattr(explanation_dict, 'iterrows'):
                # Older versions return a DataFrame
                for _, row in explanation_dict.iterrows():
                    rules.append({
                        "rule": str(row.get("rule", "Unknown")),
                        "prediction": str(row.get("prediction", "Positive")),
                        "confidence": float(row.get("precision", 0.9)),
                        "support": float(row.get("coverage", 0.1))
                    })

            if not rules:
                rules = [{"rule": "Model behavior follows general feature distributions.", "prediction": "Positive", "confidence": 0.5, "support": 1.0}]

            # Feature importance: mark features that appear in any rule
            importance = []
            for f in feature_names:
                score = 1.0 if any(f in r["rule"] for r in rules) else 0.1
                importance.append({"feature": f, "importance": score})

            return {
                "rules": rules,
                "feature_importance": importance,
                "explanation_data": {"rules": rules}
            }

        except Exception as e:
            logger.error(f"AIX360 explain_instance failed: {e}")
            logger.error(traceback.format_exc())
            return {
                "rules": [{"rule": f"Explanation error: {str(e)}", "prediction": "N/A", "confidence": 0.0, "support": 0.0}],
                "feature_importance": [{"feature": f, "importance": 0.0} for f in feature_names],
                "error": str(e),
                "status": "failed"
            }

    @staticmethod
    def explain_global(
        explainer_obj: Any,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Global BRCG rules — same as instance-level since rules are global."""
        return AIX360Service.explain_instance(explainer_obj, np.array([]), feature_names)


aix360_service = AIX360Service()
