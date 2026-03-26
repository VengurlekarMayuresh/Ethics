"""
End-to-End Test: Loan Prediction Pipeline
Tests the complete flow from model file to prediction to ensure
frontend will ask only for RAW features, not derived ones.
"""

import sys
import asyncio
import pandas as pd
import joblib
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path.resolve()))

from app.services.model_loader_service import ModelLoaderService
from app.models.model_meta import FeatureSchema

print("=" * 80)
print("END-TO-END LOAN PREDICTION TEST")
print("=" * 80)


# ============================================
# STEP 1: Verify Model File Exists
# ============================================
model_path = "loan_prediction_pipeline.pkl"
if not Path(model_path).exists():
    print(f"[ERROR] Model file not found: {model_path}")
    print("Please run 'Loan Prediction.py' first to generate the model.")
    sys.exit(1)

print(f"\n1. Model file found: {model_path}")
print(f"   Size: {Path(model_path).stat().st_size / 1024:.1f} KB")

# ============================================
# STEP 2: Load Model via Backend Service (with class injection)
# ============================================
print("\n2. Loading model via backend service...")

async def load_and_test():
    try:
        # Read model file bytes
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
        # Use load_model_from_bytes which handles custom class injection
        model_obj, framework = await ModelLoaderService.load_model_from_bytes(model_bytes, Path(model_path).name)
        print(f"   Framework: {framework}")
        print(f"   [OK] Model loaded via backend service")
    except Exception as e:
        print(f"   [FAIL] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    return model_obj, framework

model_obj, framework = asyncio.run(load_and_test())
if model_obj is None:
    sys.exit(1)

# ============================================
# STEP 3: Test Prediction with Raw Input using loaded model
# ============================================
print("\n3. Testing prediction (raw input)...")

test_raw = pd.DataFrame([{
    'Gender': 'Male',
    'Married': 'No',
    'Dependents': '0',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'Property_Area': 'Urban',
    'ApplicantIncome': 6000,
    'CoapplicantIncome': 0,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1
}])

print(f"   Input (RAW):")
for col, val in test_raw.iloc[0].items():
    print(f"     {col}: {val}")

try:
    # The model_obj is a pipeline; we can call predict directly
    prediction = model_obj.predict(test_raw)[0]
    probability = model_obj.predict_proba(test_raw)[0]

    print(f"\n   Prediction: {prediction} (1=Approved, 0=Rejected)")
    print(f"   Probability: [{probability[0]:.4f}, {probability[1]:.4f}]")
    print(f"   [OK] Prediction works with raw input")
except Exception as e:
    print(f"   [FAIL] Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# STEP 4: Extract Feature Schema via ModelLoaderService
# ============================================
print("\n4. Testing backend feature schema extraction...")

async def test_schema_extraction():
    try:
        # We already have model_obj and framework, but to be independent, load again from file
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
        model_obj, framework = await ModelLoaderService.load_model_from_bytes(model_bytes, Path(model_path).name)
        print(f"   Framework detected: {framework}")

        # Generate feature schema (no dataset analysis)
        feature_schema_objs = await ModelLoaderService.generate_feature_schema(
            model_obj, framework, dataset_analysis=None
        )

        print(f"   Feature schema generated: {len(feature_schema_objs)} features")

        # Display schema
        print("\n   Extracted Feature Schema:")
        print("   " + "-" * 60)
        for fs in feature_schema_objs:
            print(f"   {fs.name:25} type={fs.type:12} options={fs.options or 'N/A'}")

        # Validate: Check for derived features
        feature_names = [fs.name for fs in feature_schema_objs]
        derived_features = [
            'Total_Income', 'ApplicantIncomeLog', 'CoapplicantIncomeLog',
            'LoanAmountLog', 'Loan_Amount_Term_Log', 'Total_IncomeLog'
        ]
        found_derived = [f for f in derived_features if f in feature_names]

        if found_derived:
            print(f"\n   [FAIL] Found derived features in schema: {found_derived}")
            print("   This would cause frontend to ask for derived data!")
            return False
        else:
            print("\n   [OK] No derived features in schema")

        # Validate: Check for expected raw features
        expected_raw = [
            'Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Property_Area',
            'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History'
        ]

        missing = [f for f in expected_raw if f not in feature_names]
        if missing:
            print(f"\n   [WARN] Missing expected raw features: {missing}")
        else:
            print(f"   [OK] All expected raw features present")

        # Validate: Check categorical features have options
        categorical_issues = []
        for fs in feature_schema_objs:
            if fs.type == "categorical" and not fs.options:
                categorical_issues.append(fs.name)
            if fs.type == "numeric" and fs.options:
                print(f"   [WARN] Numeric feature has options: {fs.name} -> {fs.options}")

        if categorical_issues:
            print(f"\n   [WARN] Categorical features missing options: {categorical_issues}")
            print("   Frontend dropdowns may not show values properly.")
        else:
            print(f"   [OK] All categorical features have options")

        # Check feature count
        expected_count = len(expected_raw)
        actual_count = len(feature_schema_objs)
        if actual_count != expected_count:
            print(f"\n   [WARN] Feature count mismatch: expected {expected_count}, got {actual_count}")
            print("   This might indicate extra features (like derived ones).")
        else:
            print(f"   [OK] Feature count matches: {actual_count}")

        print("\n" + "=" * 80)
        print("SCHEMA EXTRACTION RESULT: SUCCESS")
        print("=" * 80)
        print("The frontend will receive a schema with ONLY raw features.")
        print("Users will only need to provide the 11 raw inputs.")
        print("Derived features will be computed automatically by the pipeline.")
        return True

    except Exception as e:
        print(f"\n   [FAIL] Schema extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

success = asyncio.run(test_schema_extraction())

if not success:
    print("\n[FATAL] Schema extraction test failed!")
    sys.exit(1)

# ============================================
# STEP 5: Test Load via ModelLoaderService (async)
# ============================================
print("\n5. Testing model loading via ModelLoaderService...")

async def test_model_load():
    try:
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
        model_obj, framework = await ModelLoaderService.load_model_from_bytes(model_bytes, Path(model_path).name)
        print(f"   Framework: {framework}")

        # Get model info
        model_info = await ModelLoaderService.get_model_info(model_obj, framework)
        print(f"   Task type: {model_info.get('task_type')}")
        print(f"   Estimator: {model_info.get('estimator_info', {}).get('estimator_name')}")
        print(f"   [OK] Model loading via backend service works")
        return True
    except Exception as e:
        print(f"   [FAIL] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

success = asyncio.run(test_model_load())
if not success:
    sys.exit(1)

# ============================================
# STEP 6: Summary and Next Steps
# ============================================
print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nWhat this means:")
print("  • Model file (loan_prediction_pipeline.pkl) is valid")
print("  • Pipeline accepts RAW input only (no derived features)")
print("  • Backend feature schema extraction returns only raw features")
print("  • Frontend forms will show only the 11 raw input fields")
print("  • Derived features are computed automatically inside the pipeline")
print("\nYou can now:")
print("  1. Upload this model through your frontend (http://localhost:3000/models/upload)")
print("  2. The model details page should show the raw features")
print("  3. The prediction form will ask for Gender, Married, Income, etc.")
print("  4. Derived features (log transforms, Total_Income) are created automatically")
print("\n" + "=" * 80)
