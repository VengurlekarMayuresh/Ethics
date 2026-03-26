#!/usr/bin/env python3
"""
Test schema generation for car price model
"""

import sys
import os
import joblib

# Add backend to path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_dir)

from app.services.model_loader_service import ModelLoaderService
import asyncio

async def test_schema():
    print("=" * 80)
    print("CAR PRICE MODEL - SCHEMA TEST")
    print("=" * 80)

    model_path = 'car_price_pipeline.pkl'
    print(f"\nLoading model: {model_path}")
    model_obj = joblib.load(model_path)

    # Check preprocessor
    if hasattr(model_obj, 'named_steps'):
        preprocessor = model_obj.named_steps.get('preprocessing')
        if preprocessor:
            print(f"\nPreprocessor: {type(preprocessor).__name__}")
            if hasattr(preprocessor, 'feature_names_in_'):
                print(f"Input features: {preprocessor.feature_names_in_.tolist()}")
            if hasattr(preprocessor, 'transformers_'):
                for name, tr, cols in preprocessor.transformers_:
                    print(f"  - {name}: {type(tr).__name__} on {cols}")

    # Generate schema
    print("\n" + "=" * 80)
    print("Generating feature schema...")
    schema_list = await ModelLoaderService.generate_feature_schema(model_obj, "sklearn", dataset_analysis=None)

    print(f"\nGenerated {len(schema_list)} features:")
    print("-" * 80)
    for fs in schema_list:
        opts = fs.options if fs.options else []
        print(f"{fs.name:20} | {fs.type:12} | options: {opts[:3] if len(opts) > 3 else opts}")

    # Verify
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    cat_features = [fs.name for fs in schema_list if fs.type == "categorical"]
    num_features = [fs.name for fs in schema_list if fs.type == "numeric"]

    expected_cats = {'name', 'fuel', 'seller_type', 'transmission', 'owner'}
    expected_nums = {'year', 'km_driven'}

    got_cats = set(cat_features)
    got_nums = set(num_features)

    if got_cats == expected_cats:
        print("[PASS] Categorical features correct")
    else:
        print("[FAIL] Categorical mismatch")
        print(f"  Expected: {expected_cats}")
        print(f"  Got: {got_cats}")

    if got_nums == expected_nums:
        print("[PASS] Numeric features correct")
    else:
        print("[FAIL] Numeric mismatch")
        print(f"  Expected: {expected_nums}")
        print(f"  Got: {got_nums}")

    missing_opts = [fs.name for fs in schema_list if fs.type == "categorical" and not fs.options]
    if missing_opts:
        print(f"[FAIL] Missing options: {missing_opts}")
    else:
        print("[PASS] All categorical features have options")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_schema())
