#!/usr/bin/env python3
"""
Test LIME explanation for car price model.
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
from app.services.lime_service import LIMEService
import asyncio

async def test_lime():
    print("=" * 80)
    print("CAR PRICE MODEL - LIME EXPLANATION TEST")
    print("=" * 80)

    # Load model
    model_path = 'car_price_pipeline.pkl'
    print(f"\nLoading model: {model_path}")
    model_obj = joblib.load(model_path)

    # Create test input
    test_input = pd.DataFrame([{
        'name': 'Maruti Alto',
        'year': 2015,
        'km_driven': 50000,
        'fuel': 'Petrol',
        'seller_type': 'Individual',
        'transmission': 'Manual',
        'owner': 'First Owner'
    }])
    print("\nTest input:")
    print(test_input.to_string(index=False))

    # Create LIME explainer
    print("\n" + "=" * 80)
    print("Creating LIME explainer...")
    try:
        explainer = LIMEService.create_explainer(
            model=model_obj,
            framework='sklearn',
            training_data=test_input,  # Using test input as background (small sample)
            feature_names=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner'],
            mode='regression'
        )
        print(f"[OK] LIME explainer created")
        print(f"  Explainer feature_names: {explainer.feature_names[:10]}... (total {len(explainer.feature_names)})")
        print(f"  Categorical features: {getattr(explainer, 'categorical_features', 'N/A')}")
    except Exception as e:
        print(f"[FAIL] Failed to create explainer: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generate explanation
    print("\n" + "=" * 80)
    print("Generating LIME explanation...")
    try:
        # For pipeline, we need to preprocess the instance
        preprocessor = None
        for step_name, step_obj in model_obj.steps:
            if hasattr(step_obj, 'transform'):
                preprocessor = step_obj
                break

        if preprocessor:
            # Preprocess the input to numeric space
            input_processed = preprocessor.transform(test_input)
            if hasattr(input_processed, 'toarray'):
                input_processed = input_processed.toarray()
            instance = np.asarray(input_processed, dtype=float)[0]
        else:
            instance = test_input.values[0]

        exp_data = LIMEService.explain_instance(
            explainer=explainer,
            model=model_obj,
            instance=instance,
            num_features=10,
            num_samples=5000,
            raw_instance=test_input.iloc[0]  # Pass raw for reference
        )

        print(f"[OK] LIME explanation generated")
        print(f"\nLocal prediction: {exp_data.get('local_pred', 'N/A')}")
        print(f"\nTop features (from list_of_contributions):")
        for item in exp_data.get('list_of_contributions', [])[:10]:
            print(f"  {item['feature']:20} | weight: {item['weight']:.4f} | value: {item.get('value')}")

        # Check if we got aggregated original feature names
        feats = [f['feature'] for f in exp_data.get('list_of_contributions', [])]
        original_expected = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
        matching = set(feats) & set(original_expected)
        print(f"\nMatching original features: {len(matching)}/{len(original_expected)}")
        if len(matching) == len(original_expected):
            print("[PASS] All original features present!")
        else:
            print(f"[INFO] Original features: {original_expected}")
            print(f"[INFO] Got features: {feats}")

    except Exception as e:
        print(f"[FAIL] LIME explanation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_lime())
