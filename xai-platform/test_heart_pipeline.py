"""
Simple test to demonstrate correct usage of the heart disease pipeline.
This shows that the pipeline expects CATEGORICAL STRING values, not numeric codes.
"""

import pandas as pd
import joblib

# Load the saved pipeline
pipeline = joblib.load('heart_disease_pipeline.pkl')

print("=" * 80)
print("HEART DISEASE PIPELINE - Usage Example")
print("=" * 80)

# CORRECT: Using categorical strings
print("\n[OK] CORRECT: Using categorical string values")
test_input_correct = pd.DataFrame([{
    'age': 63,
    'sex': 'Male',              # String, not 1
    'cp': 'asymptomatic',       # String, not 3
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 'normal',        # String, not 0
    'thalach': 150,
    'exang': 'No',              # String, not 0
    'oldpeak': 2.3,
    'slope': 'upsloping'        # String, not 0
}])

print("Input:")
print(test_input_correct.to_string(index=False))

try:
    prediction = pipeline.predict(test_input_correct)
    probability = pipeline.predict_proba(test_input_correct)
    print(f"\nPrediction: {prediction[0]}")
    print(f"Probabilities: [No Disease: {probability[0][0]:.3f}, Disease: {probability[0][1]:.3f}]")
    print("[OK] SUCCESS - Categorical strings work!")
except Exception as e:
    print(f"[FAIL] ERROR: {e}")

# INCORRECT: Using numeric codes (this will fail)
print("\n" + "=" * 80)
print("[FAIL] INCORRECT: Using numeric codes instead of categorical strings")
test_input_wrong = pd.DataFrame([{
    'age': 63,
    'sex': 1,           # Numeric instead of 'Male'
    'cp': 3,            # Numeric instead of 'asymptomatic'
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 0,       # Numeric instead of 'normal'
    'thalach': 150,
    'exang': 0,         # Numeric instead of 'No'
    'oldpeak': 2.3,
    'slope': 0          # Numeric instead of 'upsloping'
}])

print("Input:")
print(test_input_wrong.to_string(index=False))

try:
    prediction = pipeline.predict(test_input_wrong)
    print(f"\nPrediction: {prediction[0]}")
    print("[FAIL] This shouldn't work - but it did?")
except Exception as e:
    print(f"\n[OK] Expected error occurred:")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Message: {str(e)[:100]}...")
    print("\nThis error is EXPECTED because the pipeline was trained with categorical strings.")
    print("The frontend MUST send categorical string values, not numeric codes.")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("The heart_disease_pipeline.pkl expects:")
print("  Categorical features as STRING values:")
print("    - sex: 'Male' or 'Female'")
print("    - cp: 'typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'")
print("    - restecg: 'normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'")
print("    - exang: 'No' or 'Yes'")
print("    - slope: 'upsloping', 'flat', 'downsloping'")
print("  Numeric features as numbers (int/float)")
print("=" * 80)
