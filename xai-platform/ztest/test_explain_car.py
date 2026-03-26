#!/usr/bin/env python3
"""
Test SHAP and LIME explanation for car price model.
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
from app.workers.tasks import _compute_shap
import asyncio

async def test_shap():
    print("=" * 80)
    print("CAR PRICE MODEL - SHAP EXPLANATION TEST")
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
    print(test_input)

    # Background data (use training data or a sample)
    # For SHAP, we need a background dataset for the model
    # The model was trained on the car dataset - let's create a simple background
    print("\nGenerating background data...")
    background = pd.DataFrame([
        {'name': 'Maruti Alto', 'year': 2014, 'km_driven': 40000, 'fuel': 'Petrol', 'seller_type': 'Individual', 'transmission': 'Manual', 'owner': 'First Owner'},
        {'name': 'Maruti Alto', 'year': 2016, 'km_driven': 60000, 'fuel': 'Diesel', 'seller_type': 'Dealer', 'transmission': 'Manual', 'owner': 'Second Owner'},
        {'name': 'Hyundai i20', 'year': 2017, 'km_driven': 35000, 'fuel': 'Petrol', 'seller_type': 'Individual', 'transmission': 'Manual', 'owner': 'First Owner'},
        {'name': 'Honda City', 'year': 2018, 'km_driven': 25000, 'fuel': 'Petrol', 'seller_type': 'Dealer', 'transmission': 'Automatic', 'owner': 'First Owner'},
        {'name': 'Toyota Innova', 'year': 2019, 'km_driven': 45000, 'fuel': 'Diesel', 'seller_type': 'Trustmark Dealer', 'transmission': 'Manual', 'owner': 'First Owner'},
    ])
    print("Background data:")
    print(background)

    # Compute SHAP values
    print("\n" + "=" * 80)
    print("Computing SHAP values...")
    try:
        shap_values, expected_value, feature_names = _compute_shap(
            model_obj, 'sklearn', test_input, background
        )
        print(f"[OK] SHAP computed successfully")
        print(f"  shap_values type: {type(shap_values)}")
        print(f"  shap_values shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'N/A'}")
        print(f"  expected_value: {expected_value}")
        print(f"  feature_names: {feature_names}")

        # Try to format for frontend
        print("\nFormatted for frontend:")
        if isinstance(shap_values, np.ndarray):
            # For regression, shap_values is shape (1, n_features) or (n_features,)
            if shap_values.ndim == 2:
                values = shap_values[0].tolist()
            else:
                values = shap_values.tolist()

            formatted = {
                "baseValue": float(expected_value) if isinstance(expected_value, (int, float, np.number)) else float(expected_value[0]),
                "features": [
                    {"feature": name, "contribution": float(val), "value": test_input[name].iloc[0] if name in test_input.columns else None}
                    for name, val in zip(feature_names, values)
                ]
            }
            print(f"{formatted}")
        else:
            print("  Unexpected shap_values format")

    except Exception as e:
        print(f"[FAIL] SHAP computation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_shap())
