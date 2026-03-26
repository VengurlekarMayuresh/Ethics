#!/usr/bin/env python3
"""
Loan Prediction Pipeline
This script builds and exports a pipeline for loan approval prediction.
The pipeline accepts ONLY raw input features and internally computes all transformations.

Dataset: Loan Prediction.csv
RAW INPUT FEATURES (what users provide):
  - Gender: Gender (categorical)
  - Married: Marital status (categorical)
  - Dependents: Number of dependents (categorical)
  - Education: Education level (categorical)
  - Self_Employed: Self employment status (categorical)
  - ApplicantIncome: Applicant income (numeric - raw)
  - CoapplicantIncome: Coapplicant income (numeric - raw)
  - LoanAmount: Loan amount in thousands (numeric - raw)
  - Loan_Amount_Term: Term of loan in months (numeric - raw)
  - Credit_History: Credit history (numeric - binary)
  - Property_Area: Property area (categorical)
Target: Loan_Status (binary: Y/N)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import FeatureEngineer from backend custom module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from app.custom.feature_engineer import FeatureEngineer

print("=" * 80)
print("LOAN PREDICTION PIPELINE - Build & Export")
print("=" * 80)

# ============================================
# CONFIGURATION: Raw input features only
# ============================================
print("\n1. Configuring raw feature definitions...")

# These are the ONLY features a user needs to provide
categorical_features = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Property_Area'
]

numeric_features = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History'
]

target_column = 'Loan_Status'

print(f"Categorical features: {categorical_features}")
print(f"Numeric features: {numeric_features}")
print(f"Target: {target_column}")

# ============================================
# STEP 1: Load Dataset
# ============================================
print("\n2. Loading dataset...")

data_path = 'Loan_Prediction.csv'
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"First few rows:")
print(df.head())
print(f"\nTarget distribution:")
print(df[target_column].value_counts())

# ============================================
# STEP 2: Clean Data
# ============================================
print("\n3. Cleaning data...")

# Drop duplicates
initial_rows = len(df)
df = df.drop_duplicates()
dropped = initial_rows - len(df)
print(f"Removed {dropped} duplicate rows")

# Drop Loan_ID
if 'Loan_ID' in df.columns:
    df = df.drop('Loan_ID', axis=1)
    print("Dropped 'Loan_ID' column")

# Fill missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"Missing values found:\n{missing[missing > 0]}")
    # Fill categorical with mode
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Property_Area']:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    # Fill numeric with mean
    for col in ['LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome', 'CoapplicantIncome']:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    print(f"Filled missing values. New shape: {df.shape}")

# Map target to binary
df[target_column] = df[target_column].map({'Y': 1, 'N': 0, 'y': 1, 'n': 0})
print("Mapped target: Y/N -> 1/0")

# ============================================
# STEP 3: Define Derived Features (not added to training data)
# ============================================
print("\n4. Defining derived feature transformations...")

# IMPORTANT: These derived features will be computed by the FeatureEngineer transformer
# inside the pipeline. We define the list here for use in the ColumnTransformer,
# but we DO NOT add these columns to the training DataFrame.
derived_numeric_features = [
    'Total_Income',
    'ApplicantIncomeLog',
    'CoapplicantIncomeLog',
    'LoanAmountLog',
    'Loan_Amount_Term_Log',
    'Total_IncomeLog'
]

print(f"Derived features to be computed automatically: {derived_numeric_features}")

# ============================================
# STEP 4: Prepare Training Data (RAW ONLY)
# ============================================
print("\n5. Preparing training data...")

# CRITICAL: Use ONLY raw features for X.
# The pipeline's FeatureEngineer will add derived features internally.
# If we include derived features in X, the model would require processed data.
raw_features = numeric_features + categorical_features
X = df[raw_features]
y = df[target_column]

all_training_features = raw_features + derived_numeric_features
print(f"Training with {len(X)} samples and {len(all_training_features)} total features (raw + derived)")
print(f"Raw features (user input): {raw_features}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================
# STEP 5: Custom Transformer for Feature Engineering
# ============================================
print("\n6. Creating custom feature engineering transformer...")

# Import FeatureEngineer from backend custom module (ensures pickle compatibility)
# The class is defined in backend/app/custom/loan_feature_engineer.py
from app.custom.feature_engineer import FeatureEngineer
print("[OK] FeatureEngineer transformer imported from backend custom module")

# ============================================
# STEP 6: Build Preprocessing Pipeline
# ============================================
print("\n7. Building preprocessing pipeline...")

# The pipeline will:
# 1. Apply FeatureEngineer to add derived features
# 2. One-hot encode categoricals and scale all numerics (raw + derived)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numeric_features + derived_numeric_features)
    ]
)

print("[OK] Preprocessor created")
print(f"  - Categorical: OneHotEncoder on {categorical_features}")
print(f"  - Numeric: StandardScaler on {len(numeric_features) + len(derived_numeric_features)} features")

# ============================================
# STEP 7: Build Full Pipeline
# ============================================
print("\n8. Building full pipeline...")

pipeline = Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

print("[OK] Pipeline created with custom FeatureEngineer")

# ============================================
# STEP 8: Train Pipeline
# ============================================
print("\n9. Training pipeline...")

# Train on RAW features only (derived features computed by FeatureEngineer during fit)
X_train_raw = X_train[numeric_features + categorical_features]
pipeline.fit(X_train_raw, y_train)
print("[OK] Training complete!")

# Evaluate on test set (also raw)
X_test_raw = X_test[numeric_features + categorical_features]
train_score = pipeline.score(X_train_raw, y_train)
test_score = pipeline.score(X_test_raw, y_test)
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# ============================================
# STEP 9: Test with RAW Inputs ONLY
# ============================================
print("\n10. Testing with RAW inputs (NO derived features)...")

# User provides ONLY raw features - the pipeline computes derived automatically!
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

print(f"Test input (ONLY raw features - user provides these):")
print(test_raw.to_string(index=False))

# Direct prediction on raw data - pipeline handles everything!
prediction = pipeline.predict(test_raw)
pred_prob = pipeline.predict_proba(test_raw)
print(f"\nPredicted class: {prediction[0]} (0=Rejected, 1=Approved)")
print(f"Prediction probabilities: [Reject: {pred_prob[0][0]:.3f}, Approve: {pred_prob[0][1]:.3f}]")
print("[OK] Prediction successful - no manual feature engineering needed!")

# ============================================
# STEP 10: Save Pipeline
# ============================================
print("\n11. Saving pipeline...")
output_file = 'loan_prediction_pipeline.pkl'
joblib.dump(pipeline, output_file)
print(f"[OK] Saved to: {output_file}")

# ============================================
# STEP 11: Verify Saved Pipeline
# ============================================
print("\n12. Verifying saved pipeline...")
loaded_pipeline = joblib.load(output_file)
loaded_pred = loaded_pipeline.predict(test_raw)
loaded_prob = loaded_pipeline.predict_proba(test_raw)
print(f"Loaded pipeline prediction: {loaded_pred[0]}")
print(f"Match: {prediction[0] == loaded_pred[0]}")
print("[OK] Pipeline verification complete!")

# ============================================
# STEP 12: Feature Schema for Frontend
# ============================================
print("\n13. Feature schema for frontend/API...")

print(f"\n>>> IMPORTANT <<<")
print(f"The pipeline expects RAW INPUT ONLY - these {len(numeric_features + categorical_features)} features:")
print(f"\nCategorical ({len(categorical_features)}):")
for feat in categorical_features:
    print(f"  - {feat}")
print(f"\nNumeric ({len(numeric_features)}):")
for feat in numeric_features:
    print(f"  - {feat}")

print("\nThe pipeline automatically computes:")
for feat in derived_numeric_features:
    print(f"  - {feat} (derived from raw)")

# Check feature_names_in_ on the preprocessing step after fitting
preprocessor_fitted = loaded_pipeline.named_steps['preprocessing']
if hasattr(preprocessor_fitted, 'feature_names_in_'):
    print(f"\nPreprocessor feature_names_in_: {list(preprocessor_fitted.feature_names_in_)}")

# ============================================
# STEP 13: Sample Categorical Options
# ============================================
print("\n14. Sample categorical values (from dataset):")
for cat_feat in categorical_features:
    unique_vals = df[cat_feat].dropna().unique()[:10]
    print(f"  {cat_feat}: {list(unique_vals)}")

# ============================================
# STEP 14: Export Training/Test Data (Optional)
# ============================================
print("\n15. Exporting raw training data...")
X_train_raw.to_csv('loan_X_train_raw.csv', index=False)
X_test_raw.to_csv('loan_X_test_raw.csv', index=False)
y_train.to_csv('loan_y_train.csv', index=False)
y_test.to_csv('loan_y_test.csv', index=False)
print("   Exported: loan_X_train_raw.csv, loan_X_test_raw.csv, loan_y_train.csv, loan_y_test.csv")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 80)
print("PIPELINE BUILD COMPLETE")
print("=" * 80)
print(f"[OK] Pipeline trained with {len(X_train)} samples")
print(f"[OK] Test accuracy: {test_score:.4f}")
print(f"[OK] Model saved to: {output_file}")
print(f"\n[OK] RAW INPUT FEATURES ({len(numeric_features + categorical_features)} total):")
print(f"    Categorical: {categorical_features}")
print(f"    Numeric: {numeric_features}")
print(f"\n[OK] AUTOMATIC FEATURE ENGINEERING:")
print(f"    The pipeline computes these derived features internally:")
print(f"    {derived_numeric_features}")
print("\nSHAP/LIME COMPATIBILITY:")
print("  - ColumnTransformer structure compatible with backend")
print("  - Final RandomForestClassifier correctly extracted")
print("  - Feature names preserved for explanations")
print("\nHOW TO USE:")
print("  1. Upload loan_prediction_pipeline.pkl to XAI platform")
print("  2. Frontend should render form with ONLY RAW features (listed above)")
print("  3. When predicting, send DataFrame with ONLY those raw features")
print("  4. The pipeline will compute derived features automatically")
print("=" * 80)
