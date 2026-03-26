"""
Patch for backend/app/workers/tasks.py

This patch adds aggregation of SHAP values for one-hot encoded categorical features.
It should be applied to the _compute_shap function.

The issue: When using OneHotEncoder, we get 100s of features (one per category).
The frontend can't display this many and shows broken graphs.

Solution: Aggregate SHAP values back to the original categorical feature by summing
their absolute contributions, preserving the sign based on which category was activated.
"""

def aggregate_shap_values(
    shap_values: np.ndarray,
    feature_names: List[str],
    preprocessor
) -> tuple[np.ndarray, List[str]]:
    """
    Aggregate SHAP values for one-hot encoded features back to original categorical features.

    Args:
        shap_values: SHAP values array, shape (n_samples, n_preprocessed_features)
        feature_names: List of preprocessed feature names (from get_feature_names_out)
        preprocessor: The ColumnTransformer from the sklearn pipeline

    Returns:
        aggregated_shap: shape (n_samples, n_original_features)
        original_feature_names: List of original feature names
    """
    import numpy as np
    from collections import defaultdict

    # Build mapping from original categorical feature to its encoded column indices
    # and keep numeric features as-is
    original_to_encoded = defaultdict(list)
    encoded_to_original = {}

    if preprocessor and hasattr(preprocessor, 'transformers_'):
        for transformer_name, transformer_obj, cols in preprocessor.transformers_:
            transformer_class = transformer_obj.__class__.__name__
            if 'OneHotEncoder' in transformer_class:
                # These columns are one-hot encoded
                for col in cols:
                    # Find all encoded columns that belong to this original column
                    for idx, fname in enumerate(feature_names):
                        # Feature names from get_feature_names_out typically have format: "cat__original_categoryvalue"
                        # Or: "original__categoryvalue" depending on the ColumnTransformer
                        # They also might have the transformer prefix like "cat__" or "onehot__"
                        if fname.lower().startswith(transformer_name.lower() + '__'):
                            # Extract the part after the first '__'
                            parts = fname.split('__', 1)
                            if len(parts) == 2:
                                original_name_part = parts[1]
                                # The original feature name is before the first '_' with the category?
                                # Actually the format can be: transformer__feature_catvalue
                                # For ColumnTransformer with OneHotEncoder, the feature names are:
                                # "cat__featurename_categoryvalue" where categoryvalue is the actual category
                                # We need to map back to the original column 'featurename'
                                # But careful: if we created the encoder with ColumnTransformer
                                # the encoded features are like: "cat__name_Maruti Alto" -> original is 'name'

                                # Try to match: if the feature name starts with transformer_name__ and contains the original col
                                if col in fname:
                                    original_to_encoded[col].append(idx)
                                    encoded_to_original[idx] = col
            else:
                # For StandardScaler etc., columns are passed through as-is (may have prefix)
                for col in cols:
                    # Find matching encoded column
                    for idx, fname in enumerate(feature_names):
                        if fname.lower().startswith(transformer_name.lower() + '__'):
                            parts = fname.split('__', 1)
                            if len(parts) == 2 and col in parts[1]:
                                # This is the numeric feature directly (no one-hot expansion)
                                original_to_encoded[col].append(idx)
                                encoded_to_original[idx] = col
                        elif fname == col:
                            # No prefix
                            original_to_encoded[col].append(idx)
                            encoded_to_original[idx] = col

    # If we couldn't build mapping from preprocessor (e.g., no transformers_), fallback
    if not original_to_encoded:
        # Then feature_names may already be original names (unlikely for pipelines with OneHotEncoder)
        # but we'll just return as-is
        return shap_values, feature_names

    # Now aggregate SHAP values for each original feature
    n_samples = shap_values.shape[0]
    original_features = list(original_to_encoded.keys())
    aggregated_shap = np.zeros((n_samples, len(original_features)), dtype=float)

    for orig_idx, orig_feat in enumerate(original_features):
        encoded_indices = original_to_encoded[orig_feat]
        if len(encoded_indices) == 1:
            # Single feature, no aggregation needed
            aggregated_shap[:, orig_idx] = shap_values[:, encoded_indices[0]]
        else:
            # Sum contributions from all encoded columns
            # For SHAP values, we can sum them. The sign (positive/negative) will naturally reflect
            # which category the instance belongs to (since only one one-hot column is active).
            aggregated_shap[:, orig_idx] = shap_values[:, encoded_indices].sum(axis=1)

    return aggregated_shap, original_features


# This function would replace/update parts of _compute_shap in tasks.py
def compute_shap_with_aggregation(model_obj, framework, input_data, background_data):
    """Wrapper that computes SHAP and aggregates if needed."""
    import shap
    import numpy as np
    from sklearn.pipeline import Pipeline
    from app.workers.tasks import _prepare_background  # import the existing helper

    # ... existing _compute_shap logic up to shap_values computation ...

    # After computing shap_values and final_feature_names:
    # shap_values, expected_value, final_feature_names = ... (from existing code)

    # Then, if it's a pipeline with a preprocessor, aggregate one-hot features:
    if isinstance(model_obj, Pipeline):
        preprocessor = None
        for step_name, step_obj in model_obj.steps:
            if hasattr(step_obj, 'transform'):
                preprocessor = step_obj
                break

        if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
            # We have preprocessed features; aggregate one-hot encoded features
            aggregated_shap, original_names = aggregate_shap_values(
                shap_values, final_feature_names, preprocessor
            )
            shap_values = aggregated_shap
            final_feature_names = original_names
            # Note: expected_value remains the same (base value for the model)

    return shap_values, expected_value, final_feature_names
