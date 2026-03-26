#!/usr/bin/env python3
"""
Check SHAP feature names and demonstrate aggregation problem.
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np

# Add backend to path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_dir)

from app.services.model_loader_service import ModelLoaderService
from sklearn.pipeline import Pipeline
import shap

async def test_shap_aggregation():
    print("=" * 80)
    print("INVESTIGATING SHAP FEATURE NAMES AND AGGREGATION")
    print("=" * 80)

    # Load model
    model_obj = joblib.load('car_price_pipeline.pkl')

    # Get preprocessor
    preprocessor = None
    for step_name, step_obj in model_obj.steps:
        if hasattr(step_obj, 'transform'):
            preprocessor = step_obj
            break

    if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
        print(f"\nTotal preprocessed features: {len(feature_names)}")
        print("\nFirst 20 feature names from OneHotEncoder/StandardScaler:")
        for i, name in enumerate(feature_names[:20]):
            print(f"  {i}: {name}")

        # Count how many features come from 'name' (car model)
        name_features = [f for f in feature_names if 'name__' in str(f)]
        print(f"\nFeatures derived from 'name': {len(name_features)}")
        if len(name_features) > 0:
            print(f"  Sample: {name_features[:5]}")
            print(f"  This is the problem - 1306+ one-hot encoded features from car model name!")

        # Show grouping by original feature
        from collections import defaultdict
        grouped = defaultdict(list)
        for full_name in feature_names:
            if '__' in str(full_name):
                transformer, original = str(full_name).split('__', 1)
                grouped[original].append(full_name)
            else:
                grouped[str(full_name)].append(full_name)

        print(f"\nGrouped by original feature ({len(grouped)} original features):")
        for orig, feats in grouped.items():
            print(f"  {orig}: {len(feats)} encoded features")

        print("\n" + "=" * 80)
        print("THE FIX NEEDED:")
        print("=" * 80)
        print("For SHAP/LIME explanations, we should:")
        print("1. Compute SHAP values in preprocessed space (1317 features)")
        print("2. Aggregate SHAP values for one-hot encoded features back to the original categorical feature")
        print("   e.g., sum of SHAP values for all 'name__*' features → single 'name' contribution")
        print("3. Frontend then shows only the original 7 features instead of 1317")
        print("\nThis aggregation should happen in the worker task before returning results.")
    else:
        print("No preprocessor or get_feature_names_out available")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_shap_aggregation())
