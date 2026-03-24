#!/usr/bin/env python3
"""
Test script to verify that ModelLoaderService.generate_feature_schema correctly
identifies categorical vs numeric features for the heart disease pipeline.
"""

import sys
import os
import joblib
import pandas as pd
import asyncio

# Add backend directory to path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_dir)

from app.services.model_loader_service import ModelLoaderService
from app.models.model_meta import FeatureSchema

async def test_schema_generation():
    print("=" * 80)
    print("TESTING FEATURE SCHEMA GENERATION")
    print("=" * 80)

    # Load the pipeline
    model_path = os.path.join(os.path.dirname(__file__), 'heart_disease_pipeline.pkl')
    print(f"\nLoading model from: {model_path}")
    model_obj = joblib.load(model_path)

    print(f"Model type: {type(model_obj)}")
    print(f"Model steps: {[name for name, _ in model_obj.steps]}")

    # Check the preprocessor
    preprocessor = model_obj.named_steps.get('preprocessing')
    if preprocessor:
        print(f"\nPreprocessor type: {type(preprocessor)}")
        if hasattr(preprocessor, 'feature_names_in_'):
            print(f"feature_names_in_: {preprocessor.feature_names_in_.tolist()}")
        if hasattr(preprocessor, 'transformers_'):
            print(f"Transformers: {[(name, type(tr).__name__, cols) for name, tr, cols in preprocessor.transformers_]}")

    # Generate feature schema
    print("\n" + "=" * 80)
    print("Generating feature schema...")
    schema_list = await ModelLoaderService.generate_feature_schema(model_obj, "sklearn", dataset_analysis=None)

    print(f"\nGenerated {len(schema_list)} features:")
    print("-" * 80)

    for fs in schema_list:
        print(f"Feature: {fs.name:15} | Type: {fs.type:12} | Options: {fs.options if fs.options else 'None'}")

    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    categorical_features = [fs.name for fs in schema_list if fs.type == "categorical"]
    numeric_features = [fs.name for fs in schema_list if fs.type == "numeric"]

    print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")

    expected_categorical = ['sex', 'cp', 'restecg', 'exang', 'slope']
    expected_numeric = ['age', 'trestbps', 'chol', 'fbs', 'thalach', 'oldpeak']

    print("\nChecking against expected...")
    if set(categorical_features) == set(expected_categorical):
        print("[PASS] Categorical features match expected")
    else:
        print("[FAIL] Categorical features MISMATCH")
        print(f"  Expected: {expected_categorical}")
        print(f"  Got: {categorical_features}")

    if set(numeric_features) == set(expected_numeric):
        print("[PASS] Numeric features match expected")
    else:
        print("[FAIL] Numeric features MISMATCH")
        print(f"  Expected: {expected_numeric}")
        print(f"  Got: {numeric_features}")

    # Check that categorical features have options
    features_without_options = [fs.name for fs in schema_list if fs.type == "categorical" and not fs.options]
    if features_without_options:
        print(f"[FAIL] Categorical features missing options: {features_without_options}")
    else:
        print("[PASS] All categorical features have options")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_schema_generation())
