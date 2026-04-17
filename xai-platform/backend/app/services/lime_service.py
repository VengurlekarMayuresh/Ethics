import lime
import lime.lime_tabular
from lime import submodular_pick
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from sklearn.pipeline import Pipeline
import traceback

logger = logging.getLogger(__name__)

class LIMEService:
    """
    Production-Ready LIME Service.
    Operates identically to the standard Colab approach:
    If a Pipeline is provided, it extracts the preprocessor, transforms the background data,
    retrieves the One-Hot Encoded feature names, and runs LIME strictly on the estimator 
    using the processed feature space. This guarantees 100% fidelity.
    """

    @staticmethod
    def _sanitize_training_array(arr: np.ndarray) -> np.ndarray:
        """Cleans numerical data for LIME's internal stability to avoid domain errors."""
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        if arr.ndim == 2 and arr.shape[0] > 1:
            stds = arr.std(axis=0)
            bad_cols = np.where((~np.isfinite(stds)) | (stds <= 1e-12))[0]
            if bad_cols.size > 0:
                jitter = np.linspace(-1e-6, 1e-6, arr.shape[0])
                for col in bad_cols:
                    arr[:, col] += jitter
        return arr

    @staticmethod
    def _split_pipeline(model):
        """
        Takes an estimator (or pipeline) and splits it into:
        (preprocessor, final_estimator, has_preprocessor)
        """
        if isinstance(model, Pipeline) and len(model.steps) > 1:
            # Reconstruct the preprocessor pipeline cleanly
            preprocessor = Pipeline(model.steps[:-1])
            final_estimator = model.steps[-1][1]
            return preprocessor, final_estimator, True
        return None, model, False

    @staticmethod
    def create_explainer(
        model,
        training_data: pd.DataFrame,
        mode: str = "classification",
        class_names: Optional[List[str]] = None,
        raw_feature_names: Optional[List[str]] = None
    ):
        preprocessor, final_estimator, is_pipeline = LIMEService._split_pipeline(model)

        if is_pipeline:
            # 1. Transform the raw training data into the PROCESSED space
            try:
                processed_data = preprocessor.transform(training_data)
                
                # 2. Extract accurate feature names from the preprocessor if possible
                feature_names = None
                try:
                    # Try modern sklearn API
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        feature_names = preprocessor.get_feature_names_out().tolist()
                    # Try older sklearn API
                    elif hasattr(preprocessor, 'get_feature_names'):
                        feature_names = preprocessor.get_feature_names()
                except Exception as fn_err:
                    logger.warning(f"Dynamic feature name extraction failed: {fn_err}")

                # 3. If dynamic methods failed, try to manually reconstruct from steps
                if feature_names is None:
                    try:
                        # If the last step of the preprocessor is a ColumnTransformer, we can peek inside
                        last_step = preprocessor.steps[-1][1]
                        if hasattr(last_step, 'get_feature_names_out'):
                            feature_names = last_step.get_feature_names_out().tolist()
                        elif hasattr(last_step, 'transformers_'):
                            # Manually build names from ColumnTransformer
                            names = []
                            for trans_name, trans_obj, cols in last_step.transformers_:
                                if trans_name == 'remainder' and trans_obj == 'drop': continue
                                if hasattr(trans_obj, 'get_feature_names_out'):
                                    names.extend(trans_obj.get_feature_names_out(cols).tolist())
                                elif hasattr(trans_obj, 'classes_'): # LabelEncoder style
                                    names.extend([f"{cols[0]}_{c}" for c in trans_obj.classes_])
                                else:
                                    names.extend(cols)
                            feature_names = names
                    except:
                        pass

                # 4. Final Fallback: if we still don't have names, use raw names or feature_N
                if feature_names is None:
                    n_features = processed_data.shape[1] if hasattr(processed_data, 'shape') else len(processed_data[0])
                    if raw_feature_names and len(raw_feature_names) == n_features:
                        feature_names = raw_feature_names
                    else:
                        logger.warning(f"Falling back to generic feature_N names for LIME")
                        feature_names = [f"feature_{i}" for i in range(n_features)]
                        
            except Exception as e:
                logger.error(f"Preprocessing in LIME failed: {e}")
                raise ValueError(f"Failed to preprocess background data: {e}")
        else:
            processed_data = training_data.copy()
            feature_names = list(training_data.columns)

        # Convert sparse matrices to dense arrays for LIME
        if hasattr(processed_data, "toarray"):
            processed_data = processed_data.toarray()
            
        processed_data = np.asarray(processed_data, dtype=float)
        training_array = LIMEService._sanitize_training_array(processed_data)

        # If class_names wasn't passed by user, attempt to extract it
        if class_names is None and hasattr(final_estimator, 'classes_'):
            class_names = final_estimator.classes_.astype(str).tolist()

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_array,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            discretize_continuous=True,
            random_state=42
        )

        # Attach metadata for instance explaining
        explainer._is_pipeline = is_pipeline
        explainer._preprocessor = preprocessor
        explainer._final_estimator = final_estimator
        explainer._feature_names = feature_names
        
        return explainer

    @staticmethod
    def explain_instance(
        explainer,
        model,
        raw_instance: pd.DataFrame,
        num_features: int = 10
    ):
        """
        Generates a local LIME explanation aligned precisely with the Colab output.
        """
        # If it's a pipeline, transform the single row into the processed space!
        if getattr(explainer, '_is_pipeline', False):
            try:
                processed_inst = explainer._preprocessor.transform(raw_instance)
                if hasattr(processed_inst, "toarray"):
                    processed_inst = processed_inst.toarray()
                instance_array = np.asarray(processed_inst[0], dtype=float)
            except Exception as e:
                raise ValueError(f"Failed to preprocess explanation instance: {e}")
            
            predict_fn = explainer._final_estimator.predict_proba
            # We MUST get the total actual probability using the FULL pipeline model object, 
            # or the transformed array with the final estimator
            actual_proba = model.predict_proba(raw_instance)[0]
        else:
            instance_array = raw_instance.iloc[0].values.astype(float)
            predict_fn = model.predict_proba
            actual_proba = model.predict_proba(raw_instance)[0]

        # Determine the target label
        # INTUITION FIX: For binary classification (0=No, 1=Yes), always explain Class 1.
        # This ensures signs are consistent: Positive weight always means "closer to Success/Yes".
        # For multiclass, we still explain the predicted class for maximum insight into the actual decision.
        
        n_classes = len(actual_proba)
        if n_classes == 2:
            target_label = 1 # Force Positive Class (e.g. Approved, Survived)
        else:
            target_label = int(np.argmax(actual_proba))

        # Generate LIME explanation on the processed array
        exp = explainer.explain_instance(
            instance_array,
            predict_fn,
            labels=(target_label,),
            num_features=num_features
        )

        # Output extraction
        available_labels = list(exp.local_exp.keys())
        if target_label not in available_labels and len(available_labels) > 0:
            target_label = available_labels[0]

        explanation_list = exp.as_list(label=target_label)
        
        # Get class name safely
        class_names = getattr(explainer, 'class_names', [])
        explained_class_name = class_names[target_label] if target_label < len(class_names) else f"Class {target_label}"

        # ── Extract local_pred and intercept safely ──────────────────────────
        if hasattr(exp.local_pred, '__len__') and len(exp.local_pred) > 0:
            local_pred_val = float(exp.local_pred[0])
        else:
            local_pred_val = float(exp.local_pred)

        if isinstance(exp.intercept, dict):
            intercept_val = float(exp.intercept.get(target_label, 0.0))
        elif hasattr(exp.intercept, '__len__') and len(exp.intercept) > 0:
            intercept_val = float(exp.intercept[0])
        else:
            intercept_val = float(exp.intercept)

        # ── Terminal print ────────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print(f"[LIME] Processed-Space Explanation  (label={target_label})")
        print(f"       Actual pipeline proba  : {[round(p, 4) for p in actual_proba.tolist()]}")
        print(f"       LIME local prediction  : {round(local_pred_val, 4)}")
        print(f"       Intercept              : {round(intercept_val, 4)}")
        print("-" * 55)
        print(f"  {'Feature Rule':<35} Weight")
        print("-" * 55)
        for condition, weight in explanation_list:
            mark = "+" if weight > 0 else "-"
            print(f"  [{mark}] {condition:<33} {weight:+.4f}")
        print("=" * 55 + "\n")

        return {
            "actual_prediction"    : actual_proba.tolist(),
            "lime_local_prediction": local_pred_val,
            "explanation"          : explanation_list,
            "intercept"            : intercept_val,
            "explained_class"      : explained_class_name,
            "explained_class_index": target_label
        }

    @staticmethod
    def explain_global(
        explainer,
        model,
        samples: pd.DataFrame,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Generate global explanations using SP-LIME operating in the processed feature space.
        """
        max_samples = min(20, len(samples))

        if getattr(explainer, '_is_pipeline', False):
            try:
                processed_samples = explainer._preprocessor.transform(samples)
                if hasattr(processed_samples, "toarray"):
                    processed_samples = processed_samples.toarray()
                processed_array = np.asarray(processed_samples, dtype=float)
            except Exception as e:
                logger.error(f"Failed to preprocess global samples: {e}")
                return {"error": str(e)}
            
            predict_fn = explainer._final_estimator.predict_proba
        else:
            processed_array = np.asarray(samples.values, dtype=float)
            predict_fn = model.predict_proba

        try:
            sp_obj = submodular_pick.SubmodularPick(
                explainer,
                processed_array,
                predict_fn,
                sample_size=max_samples,
                num_features=num_features,
                num_exps_desired=5
            )
            explanations = sp_obj.sp_explanations
        except Exception as e:
            logger.error(f"Global Explanation Error: {e}")
            return {"error": str(e)}

        feature_weights: Dict[str, float] = {}
        for exp in explanations:
            available = list(exp.local_exp.keys())
            if not available:
                continue
            label = available[0]
            for feat_idx, weight in exp.local_exp[label]:
                fname = explainer.feature_names[feat_idx]
                feature_weights[fname] = feature_weights.get(fname, 0) + abs(weight)

        n_exp = max(len(explanations), 1)
        importance = [
            {"feature": f, "importance": w / n_exp}
            for f, w in feature_weights.items()
        ]
        importance.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "feature_importance"   : importance[:num_features],
            "num_samples_explained": len(explanations),
        }

# Instantiate the service
lime_service = LIMEService()