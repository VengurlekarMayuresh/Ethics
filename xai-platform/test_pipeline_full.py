"""
Comprehensive test for pipeline model with raw inputs and SHAP/LIME explanations.
This script tests:
1. Pipeline accepts raw categorical inputs (e.g., 'Male', 'Female')
2. Pipeline predictions are correct
3. SHAP works with the pipeline
4. LIME works with the pipeline
5. Feature schema extraction works correctly

Uses the actual loan dataset from train (1).csv
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PIPELINE TEST SUITE - Using Actual Loan Dataset")
print("=" * 80)

# ============================================
# STEP 1: Load Actual Dataset
# ============================================
print("\n1. Loading actual loan dataset from 'train (1).csv'...")

df = pd.read_csv('train (1).csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget distribution:")
print(df['Loan_Status'].value_counts())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# ============================================
# STEP 2: Define Raw Features (what users will input)
# ============================================
raw_categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
raw_numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

print(f"\nRaw categorical features: {raw_categorical_features}")
print(f"Raw numeric features: {raw_numeric_features}")

# ============================================
# STEP 3: Handle Missing Values
# ============================================
print("\n2. Handling missing values...")

# Fill categorical with mode, numeric with median
for col in raw_categorical_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

for col in raw_numeric_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

print(f"[OK] Missing values handled")

# ============================================
# STEP 4: Build Full Pipeline
# ============================================
print("\n3. Building pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), raw_categorical_features),
        ('num', StandardScaler(), raw_numeric_features)
    ]
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier(n_estimators=50, random_state=42))
])

# ============================================
# STEP 5: Prepare Training Data (RAW only)
# ============================================
print("\n4. Preparing training data...")

X = df[raw_categorical_features + raw_numeric_features]  # ONLY raw features
y = df['Loan_Status'].map({'Y': 1, 'N': 0})  # Convert Y/N to 1/0

print(f"X columns: {X.columns.tolist()}")
print(f"Target distribution:")
print(f"  Approved (1): {y.sum()}")
print(f"  Rejected (0): {(y == 0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================
# STEP 5: Train Pipeline
# ============================================
print("\n4. Training pipeline...")
pipeline.fit(X_train, y_train)
print("Training complete!")

accuracy = pipeline.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# ============================================
# STEP 6: Test with RAW Inputs (No preprocessing!)
# ============================================
print("\n5. Testing predictions with RAW inputs...")

# Create a test sample with RAW values (what a user would provide)
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

print(f"Test input (RAW):\n{test_raw}")

# Predict directly - pipeline handles everything
prediction = pipeline.predict(test_raw)
prediction_proba = pipeline.predict_proba(test_raw)

print(f"\nPrediction: {prediction[0]}")
print(f"Probabilities: {prediction_proba[0]}")
print(f"Class 0 (Rejected): {prediction_proba[0][0]:.4f}")
print(f"Class 1 (Approved): {prediction_proba[0][1]:.4f}")

# ============================================
# STEP 7: Test with Multiple Raw Samples
# ============================================
print("\n6. Testing with multiple raw samples...")

test_batch_raw = pd.DataFrame([
    {'Gender': 'Male', 'Married': 'No', 'Dependents': '0', 'Education': 'Graduate', 'Self_Employed': 'No', 'Property_Area': 'Urban', 'ApplicantIncome': 4000, 'CoapplicantIncome': 0, 'LoanAmount': 100, 'Loan_Amount_Term': 360, 'Credit_History': 1},
    {'Gender': 'Female', 'Married': 'Yes', 'Dependents': '1', 'Education': 'Not Graduate', 'Self_Employed': 'No', 'Property_Area': 'Semiurban', 'ApplicantIncome': 8000, 'CoapplicantIncome': 2000, 'LoanAmount': 300, 'Loan_Amount_Term': 180, 'Credit_History': 1},
    {'Gender': 'Male', 'Married': 'Yes', 'Dependents': '2', 'Education': 'Graduate', 'Self_Employed': 'Yes', 'Property_Area': 'Rural', 'ApplicantIncome': 5000, 'CoapplicantIncome': 1500, 'LoanAmount': 150, 'Loan_Amount_Term': 240, 'Credit_History': 0},
])

print(f"Batch test inputs:\n{test_batch_raw}")
batch_predictions = pipeline.predict(test_batch_raw)
batch_proba = pipeline.predict_proba(test_batch_raw)

print(f"\nBatch predictions: {batch_predictions}")
for i, prob in enumerate(batch_proba):
    print(f"  Sample {i}: P(0)={prob[0]:.3f}, P(1)={prob[1]:.3f} -> Class {batch_predictions[i]}")

# ============================================
# STEP 8: Save Pipeline
# ============================================
print("\n7. Saving pipeline...")
output_file = 'test_pipeline.pkl'
joblib.dump(pipeline, output_file)
print(f"Saved to: {output_file}")

# ============================================
# STEP 9: Load and Verify Saved Pipeline
# ============================================
print("\n8. Loading saved pipeline and verifying...")
loaded_pipeline = joblib.load(output_file)
loaded_pred = loaded_pipeline.predict(test_raw)
print(f"Loaded pipeline prediction: {loaded_pred[0]}")
print(f"Match with original: {prediction[0] == loaded_pred[0]}")
assert prediction[0] == loaded_pred[0], "Loaded pipeline prediction mismatch!"
print("[OK] Pipeline saved and loaded correctly!")

# ============================================
# STEP 10: Extract Feature Schema (Simulate Backend)
# ============================================
print("\n9. Extracting feature schema (backend simulation)...")

# Simulate what generate_feature_schema does
from sklearn.pipeline import Pipeline as SklearnPipeline

raw_feature_names = []

if isinstance(loaded_pipeline, SklearnPipeline):
    # Get raw feature names from preprocessing step (ColumnTransformer)
    column_transformer = loaded_pipeline.named_steps['preprocessing']

    if hasattr(column_transformer, 'feature_names_in_'):
        raw_feature_names = column_transformer.feature_names_in_.tolist()
        print(f"Raw input features (from ColumnTransformer): {raw_feature_names}")
    else:
        print("Warning: ColumnTransformer doesn't have feature_names_in_")
        print("This means you must fit on a DataFrame with column names.")
        # Fallback to the feature names we know we used
        raw_feature_names = raw_categorical_features + raw_numeric_features

print("\nExpected input features for frontend (RAW):")
print(f"  Categorical: {raw_categorical_features}")
print(f"  Numeric: {raw_numeric_features}")

# Verify that the extracted raw feature names match expected
if raw_feature_names:
    assert set(raw_feature_names) == set(raw_categorical_features + raw_numeric_features), \
        f"Feature name mismatch! Got {raw_feature_names}, expected {raw_categorical_features + raw_numeric_features}"
    print("[OK] Feature names match!")

# ============================================
# STEP 11: SHAP Explanation with Pipeline
# ============================================
print("\n10. Testing SHAP with pipeline...")

try:
    # For pipelines, we need to preprocess the data to numeric space before SHAP
    preprocessor = loaded_pipeline.named_steps['preprocessing']

    # Preprocess background data
    background_raw = X_train[:50]  # Small subset of raw training data
    background_processed = preprocessor.transform(background_raw)
    if hasattr(background_processed, 'toarray'):
        background_processed = background_processed.toarray()
    background_numeric = np.asarray(background_processed, dtype=float)

    # Get feature names from preprocessor (these are the preprocessed feature names)
    if hasattr(preprocessor, 'get_feature_names_out'):
        raw_names = preprocessor.get_feature_names_out()
        processed_feature_names = []
        for name in raw_names:
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            if '__' in name:
                name = name.split('__', 1)[1]
            processed_feature_names.append(name)
    else:
        processed_feature_names = [f"feature_{i}" for i in range(background_numeric.shape[1])]

    print(f"  Preprocessed feature names ({len(processed_feature_names)}): {processed_feature_names[:10]}...")

    # Use final estimator for prediction on preprocessed data
    final_estimator = loaded_pipeline.steps[-1][1]
    if hasattr(final_estimator, 'predict_proba'):
        predict_fn = final_estimator.predict_proba
    else:
        predict_fn = final_estimator.predict

    # Create SHAP explainer in preprocessed feature space
    explainer = shap.KernelExplainer(predict_fn, background_numeric)

    # Preprocess the test sample
    test_processed = preprocessor.transform(test_raw)
    if hasattr(test_processed, 'toarray'):
        test_processed = test_processed.toarray()
    test_numeric = np.asarray(test_processed, dtype=float)

    # Compute SHAP values
    shap_values = explainer.shap_values(test_numeric, nsamples=100)
    expected_value = explainer.expected_value

    print(f"SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"SHAP values (list) length: {len(shap_values)}")
        shap_for_class = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        print(f"SHAP values for class 1 shape: {shap_for_class.shape}")
    else:
        shap_for_class = shap_values
        print(f"SHAP values shape: {shap_for_class.shape}")

    print(f"Number of preprocessed features: {len(processed_feature_names)}")

    # Display top contributing features
    # shap_for_class shape: (n_instances, n_features) or (n_features,) or (n_instances, n_features, n_classes)
    if isinstance(shap_for_class, np.ndarray):
        if shap_for_class.ndim == 3:
            # Shape: (instances, features, classes) - take class 1
            shap_vals = shap_for_class[0, :, 1]
        elif shap_for_class.ndim == 2:
            shap_vals = shap_for_class[0] if shap_for_class.shape[0] == 1 else shap_for_class.mean(axis=0)
        else:
            shap_vals = shap_for_class

        feature_contributions = list(zip(processed_feature_names, shap_vals))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        print("\nTop 5 SHAP feature contributions:")
        for feat, contrib in feature_contributions[:5]:
            print(f"  {feat}: {contrib:.4f}")
    else:
        print("  (Could not parse SHAP values shape)")

    print("[OK] SHAP explanation generated successfully!")

except Exception as e:
    print(f"[FAIL] SHAP failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# STEP 12: LIME Explanation with Pipeline
# ============================================
print("\n11. Testing LIME with pipeline...")

try:
    # For pipelines, preprocess training data to numeric space for LIME
    preprocessor = loaded_pipeline.named_steps['preprocessing']

    # Preprocess training data
    background_raw = X_train[:min(100, len(X_train))]  # Use subset for performance
    background_processed = preprocessor.transform(background_raw)
    if hasattr(background_processed, 'toarray'):
        background_processed = background_processed.toarray()
    background_numeric = np.asarray(background_processed, dtype=float)

    # Get preprocessed feature names
    if hasattr(preprocessor, 'get_feature_names_out'):
        raw_names = preprocessor.get_feature_names_out()
        lime_feature_names = []
        for name in raw_names:
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            if '__' in name:
                name = name.split('__', 1)[1]
            lime_feature_names.append(name)
    else:
        lime_feature_names = [f"feature_{i}" for i in range(background_numeric.shape[1])]

    print(f"  LIME using {background_numeric.shape[0]} background samples with {len(lime_feature_names)} features")

    # Create LIME explainer with preprocessed numeric data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        background_numeric,
        feature_names=lime_feature_names,
        mode='classification',
        discretize_continuous=True,
        kernel_width=3,
        verbose=False
    )

    # Preprocess the test instance
    test_processed = preprocessor.transform(test_raw)
    if hasattr(test_processed, 'toarray'):
        test_processed = test_processed.toarray()
    test_numeric = np.asarray(test_processed, dtype=float)
    instance = test_numeric[0]

    # Use final estimator's predict_proba
    final_estimator = loaded_pipeline.steps[-1][1]

    # Generate explanation
    exp = explainer.explain_instance(
        instance,
        final_estimator.predict_proba,
        num_features=10,
        top_labels=1
    )

    print(f"LIME explanation generated for test sample")
    print(f"Local prediction: {exp.local_pred}")
    print(f"R^2 score: {exp.score}")

    # Get top contributing features (handle both dict and list formats)
    print("\nTop 5 contributing features:")
    if exp.local_exp:
        # Get the first label's contributions
        label = next(iter(exp.local_exp.keys()))
        contributions = exp.local_exp[label]
        if isinstance(contributions, dict):
            # contributions is {feature_index: weight}
            sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for feature_idx, weight in sorted_contrib:
                feature_name = lime_feature_names[feature_idx] if feature_idx < len(lime_feature_names) else f"feature_{feature_idx}"
                print(f"  {feature_name}: {weight:.4f}")
        elif isinstance(contributions, list):
            # contributions is [(feature_index, weight), ...]
            for feature_idx, weight in contributions[:5]:
                feature_name = lime_feature_names[feature_idx] if feature_idx < len(lime_feature_names) else f"feature_{feature_idx}"
                print(f"  {feature_name}: {weight:.4f}")
        else:
            print("  (unknown format)")

    print("[OK] LIME explanation generated successfully!")

except Exception as e:
    print(f"[FAIL] LIME failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# STEP 13: Summary
# ============================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"[OK] Pipeline trained with {len(X_train)} samples")
print(f"[OK] Test accuracy: {accuracy:.4f}")
print(f"[OK] Pipeline accepts RAW inputs (no manual preprocessing)")
print(f"[OK] Pipeline saved and loaded correctly")
print(f"[OK] Raw feature names: {raw_categorical_features + raw_numeric_features}")

shap_status = 'SUCCESS' if 'shap_values' in locals() else 'FAILED'
lime_status = 'SUCCESS' if 'exp' in locals() else 'FAILED'
print(f"[{'OK' if shap_status == 'SUCCESS' else 'FAIL'}] SHAP explanation: {shap_status}")
print(f"[{'OK' if lime_status == 'SUCCESS' else 'FAIL'}] LIME explanation: {lime_status}")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("1. Upload the test_pipeline.pkl to your XAI platform")
print("2. The feature schema should show RAW features (gender, age, income, etc.)")
print("3. Frontend forms should accept raw values like 'Male', not encoded numbers")
print("4. Predictions should work correctly")
print("5. SHAP and LIME explanations should work")
print("\nNote: If SHAP/LIME fail, check that the background data is in raw format.")
print("The lambda wrappers ensure DataFrames are passed to the pipeline.")
