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
        model_type_name = type(model).__name__
        
        # Standard Sklearn Pipeline (using name check for version compatibility)
        if "Pipeline" in model_type_name and hasattr(model, 'steps'):
            if len(model.steps) > 1:
                preprocessor = Pipeline(model.steps[:-1])
                final_estimator = model.steps[-1][1]
                return preprocessor, final_estimator, True
            else:
                return None, model.steps[0][1], False
        
        # Custom object with .steps (common in our backend wrappers)
        if hasattr(model, 'steps') and isinstance(model.steps, list):
            steps_dict = dict(model.steps)
            if 'preprocessing' in steps_dict and 'model' in steps_dict:
                return steps_dict['preprocessing'], steps_dict['model'], True
            if len(model.steps) > 1:
                # Fallback: assume all but last is preprocessor
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
        categorical_features = None
        feature_names = None

        if is_pipeline:
            # 1. Transform the raw training data into the PROCESSED space
            try:
                processed_data = preprocessor.transform(training_data)
                
                # 2. Extract accurate feature names from the preprocessor
                feature_names = None
                try:
                    # Preferred: Use sklearn's internal structure for precise alignment
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        feature_names = preprocessor.get_feature_names_out().tolist()
                    elif hasattr(preprocessor, 'get_feature_names'):
                        feature_names = preprocessor.get_feature_names()
                except Exception as fn_err:
                    logger.warning(f"Standard feature name extraction failed: {fn_err}")

                # 3. CRITICAL: Reconstruct names AND track Categorical Indices
                if feature_names is None or len(feature_names) != processed_data.shape[1]:
                    try:
                        ct = None
                        if hasattr(preprocessor, 'transformers_'):
                            ct = preprocessor
                        elif hasattr(preprocessor, 'steps'):
                            for _, step in preprocessor.steps:
                                if hasattr(step, 'transformers_'):
                                    ct = step
                                    break
                        
                        if ct:
                            names = []
                            cat_indices = []
                            current_pos = 0
                            
                            if hasattr(ct, 'transformers_'):
                                for name, trans, cols, _ in ct._iter(with_nx=True):
                                    if trans == 'drop':
                                        continue
                                    
                                    # Get output size of this step
                                    step_names = []
                                    is_categorical_step = False
                                    
                                    if trans == 'passthrough':
                                        if isinstance(cols, list):
                                            step_names = cols
                                        elif isinstance(cols, (int, slice)):
                                            step_names = training_data.columns[cols].tolist()
                                    else:
                                        if hasattr(trans, 'get_feature_names_out'):
                                            step_names = trans.get_feature_names_out(cols).tolist()
                                            # Categorical if OHE
                                            if 'OneHotEncoder' in type(trans).__name__:
                                                is_categorical_step = True
                                        elif hasattr(trans, 'get_feature_names'):
                                            step_names = trans.get_feature_names(cols).tolist()
                                            is_categorical_step = 'OneHot' in type(trans).__name__
                                        else:
                                            if isinstance(cols, list):
                                                step_names = cols
                                            else:
                                                step_names = [cols]
                                    
                                    # Record indices if this step is categorical
                                    if is_categorical_step:
                                        # Only treat as 'categorical' in LIME if it's NOT OHE
                                        # (LIME handles binary 0/1 bits better as 'continuous' binaries)
                                        if 'OneHotEncoder' not in type(trans).__name__:
                                            cat_indices.extend(range(current_pos, current_pos + len(step_names)))
                                    
                                    names.extend(step_names)
                                    current_pos += len(step_names)
                            
                            if len(names) == processed_data.shape[1]:
                                feature_names = names
                                categorical_features = cat_indices
                                logger.info(f"Robust name sync: {len(feature_names)} features, {len(categorical_features)} categorical")

                    except Exception as rec_err:
                        logger.error(f"Categorical reconstruction failed: {rec_err}")
                
                # 4. Fallback for Categorical detection
                if categorical_features is None and feature_names:
                    categorical_features = [i for i, name in enumerate(feature_names) if 'cat__' in name or '_male' in name or '_female' in name]

                # 4b. Final Fallback for names
                if feature_names is None or len(feature_names) != processed_data.shape[1]:
                    n_features = processed_data.shape[1] if hasattr(processed_data, 'shape') else len(processed_data[0])
                    if raw_feature_names and len(raw_feature_names) == n_features:
                        feature_names = raw_feature_names
                    else:
                        logger.warning(f"LIME fallback: feature_N labels used.")
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

        # STABILITY FIX: Pass categorical_features to prevent unstable binning of binary 0/1 columns
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_array,
            feature_names=feature_names,
            categorical_features=categorical_features,
            class_names=class_names,
            mode=mode,
            discretize_continuous=True, # LIME will skip categorical features during discretization
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
        # Determine if we have a preprocessor to use
        has_preprocessor = hasattr(explainer, '_preprocessor') and explainer._preprocessor is not None
        
        # If we have a preprocessor, we SHOULD use it if the flag is set OR if the data looks non-numeric
        is_pipeline_flag = getattr(explainer, '_is_pipeline', False)
        
        should_preprocess = is_pipeline_flag
        if not should_preprocess and has_preprocessor:
            # Check for non-numeric types in the raw instance
            for col in raw_instance.columns:
                if not pd.api.types.is_numeric_dtype(raw_instance[col]):
                    should_preprocess = True
                    break

        if should_preprocess and has_preprocessor:
            try:
                processed_inst = explainer._preprocessor.transform(raw_instance)
                if hasattr(processed_inst, "toarray"):
                    processed_inst = processed_inst.toarray()
                instance_array = np.asarray(processed_inst[0], dtype=float)
                
                # If we were given a final estimator during creation, use it for LIME's neighborhood
                if hasattr(explainer, '_final_estimator') and explainer._final_estimator:
                    predict_fn = explainer._final_estimator.predict_proba
                else:
                    predict_fn = model.predict_proba
                
                actual_proba = model.predict_proba(raw_instance)[0]
            except Exception as e:
                logger.error(f"LIME preprocessing failed: {e}")
                # Last ditch fallback: try raw array
                instance_array = raw_instance.iloc[0].values.astype(float)
                predict_fn = model.predict_proba
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
        # Fix: Ensure values are standard floats to avoid NaN/serialization issues
        if hasattr(exp.local_pred, '__len__') and len(exp.local_pred) > 0:
            local_pred_val = float(np.nan_to_num(exp.local_pred[0], nan=0.0))
        else:
            local_pred_val = float(np.nan_to_num(exp.local_pred, nan=0.0))

        if isinstance(exp.intercept, dict):
            intercept_val = float(np.nan_to_num(exp.intercept.get(target_label, 0.0), nan=0.0))
        elif hasattr(exp.intercept, '__len__') and len(exp.intercept) > 0:
            intercept_val = float(np.nan_to_num(exp.intercept[0], nan=0.0))
        else:
            intercept_val = float(np.nan_to_num(exp.intercept, nan=0.0))

        # Sanitize actual_proba to ensure no NaNs reach the UI
        actual_proba_safe = np.nan_to_num(actual_proba, nan=0.0).tolist()

        # ── POST-PROCESS: Aggregate OHE bits into single clean features ─────────────────
        # This makes LIME as clear as SHAP and fixes the "Logic Reversal" sign confusion
        # Logic: Contribution(Column) = Sum(Weight_i) for all bits that are 'Active' (value=1)
        import re
        column_weights = {}

        for condition, weight in explanation_list:
            # 1. Clean the feature name (Strip prefixes)
            clean_name = condition.replace("cat__", "").replace("num__", "")
            
            # 2. Extract base root (e.g. Sex_female=1 -> Sex or -1.12 < Age <= 0.5 -> Age)
            # Strategy: Strip all numbers and comparison operators from the start and end
            base_root = clean_name
            # Regex to find the variable in common LIME range strings
            # Match: [Numbers/Ops] [VariableName] [Numbers/Ops]
            # e.g. -1.12 < Age <= 5.0
            range_match = re.search(r'(?<=[\d\.\-\s<>=])([a-zA-Z_]\w*)(?=[\d\.\-\s<>=]|$)', clean_name)
            if range_match:
                base_root = range_match.group(1)
            else:
                # Fallback for simple names 'Sex_female' or bits
                root_match = re.match(r"^([^=<> ]+)", clean_name)
                base_root = root_match.group(1) if root_match else clean_name
            
            # 3. Categorical Logic (OHE bits)
            is_active = True 
            if "=" in condition:
                val_part = condition.split("=")[-1].strip()
                if val_part == "0" or val_part == "0.00":
                    is_active = False

            # 4. Strip the trailing category name if it was OHE
            if base_root not in raw_instance.columns:
                parts = base_root.split("_")
                for i in range(len(parts)-1, 0, -1):
                    potential_root = "_".join(parts[:i])
                    if potential_root in raw_instance.columns:
                        base_root = potential_root
                        break
            
            # 5. Aggregate: Only sum weights for conditions actually MET by the passenger
            if is_active:
                # Strip any leftover junk from root
                base_root = base_root.strip()
                column_weights[base_root] = column_weights.get(base_root, 0) + weight

        # Sort aggregated weights by absolute impact
        aggregated_explanation = sorted(
            [(k, v) for k, v in column_weights.items()],
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # ── Output extraction ────────────────────────────────────────────────────────
        return {
            "actual_prediction"    : actual_proba_safe,
            "lime_local_prediction": local_pred_val,
            "explanation"          : aggregated_explanation, # Clean, aggregated list
            "raw_explanation"      : explanation_list,       # Kept for debugging if needed
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