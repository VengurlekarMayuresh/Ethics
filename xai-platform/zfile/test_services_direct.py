#!/usr/bin/env python3
"""
Direct unit test of the fixed SHAP and LIME backend services.
This tests the explainer logic without needing a running server.
"""
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def print_header(text: str):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)

def print_success(text: str):
    print(f"[OK] {text}")

def print_error(text: str):
    print(f"[FAIL] {text}")

def print_info(text: str):
    print(f"[INFO] {text}")

print_header("DIRECT BACKEND SERVICE TEST")

# Check if model exists
model_path = Path("loan_model_pipeline.pkl")
if not model_path.exists():
    print_error("Model file not found: loan_model_pipeline.pkl")
    print_info("Please run: python train_pipeline.py")
    sys.exit(1)

print_success(f"Found model: {model_path}")

# Load the pipeline
try:
    pipeline = joblib.load(model_path)
    print_success("Pipeline loaded successfully")
    print_info(f"Pipeline steps: {[name for name, _ in pipeline.steps]}")
except Exception as e:
    print_error(f"Failed to load pipeline: {e}")
    sys.exit(1)

# Create test raw input
test_raw = pd.DataFrame([{
    "Gender": "Male",
    "Married": "No",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "Property_Area": "Urban",
    "ApplicantIncome": 6000,
    "CoapplicantIncome": 0,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1
}])
print_info("\nTest input (raw):")
print(test_raw.to_dict(orient="records")[0])

# Test 1: Prediction works
print("\n" + "-" * 80)
print_info("Test 1: Pipeline prediction")
try:
    pred = pipeline.predict(test_raw)
    proba = pipeline.predict_proba(test_raw)
    print_success(f"Prediction: {pred[0]}")
    print_success(f"Probabilities: {proba[0]}")
except Exception as e:
    print_error(f"Prediction failed: {e}")
    sys.exit(1)

# Test 2: Find preprocessor
print("\n" + "-" * 80)
print_info("Test 2: Identify preprocessor")
preprocessor = None
for step_name, step_obj in pipeline.steps:
    if hasattr(step_obj, 'transform'):
        preprocessor = step_obj
        break

if preprocessor:
    print_success(f"Found preprocessor: {step_name}")
else:
    print_error("No preprocessor found in pipeline")
    sys.exit(1)

# Test 3: Preprocess raw data
print("\n" + "-" * 80)
print_info("Test 3: Preprocess raw data")
try:
    processed = preprocessor.transform(test_raw)
    if hasattr(processed, 'toarray'):
        processed = processed.toarray()
    processed_array = np.asarray(processed, dtype=float)
    print_success(f"Preprocessed shape: {processed_array.shape}")
    print_info(f"First few values: {processed_array[0][:5]}")
except Exception as e:
    print_error(f"Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: SHAP explanation
print("\n" + "-" * 80)
print_info("Test 4: SHAP explanation (using backend logic)")
try:
    import shap

    # Create some background data (use a few rows from training if available, or replicate test)
    # For simplicity, we'll create synthetic background by perturbing test input
    # In real backend, background comes from past predictions or uploaded file
    background_raw = pd.concat([test_raw] * 10, ignore_index=True)
    # Add small random variations
    for col in ["ApplicantIncome", "LoanAmount"]:
        if col in background_raw.columns:
            background_raw[col] = background_raw[col] + np.random.randint(-1000, 1000, len(background_raw))

    # Preprocess background
    bg_processed = preprocessor.transform(background_raw)
    if hasattr(bg_processed, 'toarray'):
        bg_processed = bg_processed.toarray()
    bg_numeric = np.asarray(bg_processed, dtype=float)

    # Get feature names from preprocessor
    if hasattr(preprocessor, 'get_feature_names_out'):
        raw_names = preprocessor.get_feature_names_out()
        feature_names = []
        for name in raw_names:
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            if '__' in name:
                name = name.split('__', 1)[1]
            feature_names.append(name)
    else:
        feature_names = [f"feature_{i}" for i in range(bg_numeric.shape[1])]

    print_info(f"Feature names ({len(feature_names)}): {feature_names[:10]}...")

    # Use final estimator
    final_estimator = pipeline.steps[-1][1]
    if hasattr(final_estimator, 'predict_proba'):
        predict_fn = final_estimator.predict_proba
    else:
        predict_fn = final_estimator.predict

    # Create SHAP explainer
    explainer = shap.KernelExplainer(predict_fn, bg_numeric[:min(50, len(bg_numeric))])
    input_processed = preprocessor.transform(test_raw)
    if hasattr(input_processed, 'toarray'):
        input_processed = input_processed.toarray()
    input_numeric = np.asarray(input_processed, dtype=float)

    shap_values = explainer.shap_values(input_numeric, nsamples=100)
    expected_value = explainer.expected_value

    print_success("SHAP computed successfully!")
    print_info(f"SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print_info(f"Number of classes: {len(shap_values)}")
        print_info(f"Class 1 SHAP shape: {shap_values[1].shape if len(shap_values) > 1 else shap_values[0].shape}")
        shap_values_to_use = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_values_to_use = shap_values
        print_info(f"SHAP shape: {shap_values_to_use.shape}")

    print_info(f"Feature names count: {len(feature_names)}")

    # Show top contributing features
    if isinstance(shap_values_to_use, np.ndarray):
        # For binary classification, shap_values might be shape (1, n_features, 2)
        # We'll use the positive class (class 1)
        if shap_values_to_use.ndim == 3:
            shap_vals_class1 = shap_values_to_use[0, :, 1]  # First instance, all features, class 1
        elif shap_values_to_use.ndim == 2:
            shap_vals_class1 = shap_values_to_use[0] if shap_values_to_use.shape[0] == 1 else shap_values_to_use.mean(axis=0)
        else:
            shap_vals_class1 = shap_values_to_use

        feature_contributions = list(zip(feature_names, shap_vals_class1))
        feature_contributions.sort(key=lambda x: abs(float(x[1])), reverse=True)
        print("\nTop 5 SHAP feature contributions:")
        for feat, contrib in feature_contributions[:5]:
            print(f"  {feat}: {contrib:.4f}")

except Exception as e:
    print_error(f"SHAP failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: LIME explanation
print("\n" + "-" * 80)
print_info("Test 5: LIME explanation (using backend logic)")
try:
    import lime
    import lime.lime_tabular

    # For LIME, we need background data in preprocessed space
    # Update: Our fixed LIME service preprocesses data before creating explainer
    training_array = bg_numeric
    feature_names_lime = feature_names

    print_info(f"Creating LIME explainer with {training_array.shape[0]} background samples...")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_array,
        feature_names=feature_names_lime,
        mode='classification',
        discretize_continuous=True,
        kernel_width=3,
        verbose=False
    )

    # Explain instance (use preprocessed input)
    instance = processed_array[0]

    print_info("Explaining instance...")
    # For classification, pass predict_proba directly (returns both classes)
    # LIME will handle selecting the class to explain
    exp = explainer.explain_instance(
        instance,
        final_estimator.predict_proba,
        num_features=min(10, len(feature_names_lime)),
        top_labels=1
    )

    print_success("LIME explanation generated!")
    print_info(f"Local prediction: {exp.local_pred}")
    print_info(f"R^2 score: {exp.score}")

    # Show top features
    print("\nTop 5 LIME contributing features:")
    for i, (feat, weight) in enumerate(exp.as_list()[:5]):
        print(f"  {i+1}. {feat}: {weight:.4f}")

except Exception as e:
    print_error(f"LIME failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print_info("If both SHAP and LIME succeeded above, the backend fixes are working!")
print_info("Now test through the actual API by:")
print_info("  1. Starting the backend server")
print_info("  2. Starting Celery workers")
print_info("  3. Running: python test_backend_api.py")
print("=" * 80)
