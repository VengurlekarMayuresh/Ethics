#!/usr/bin/env python3
"""
Medical Insurance Cost Prediction Pipeline
This script builds and exports a pipeline for medical insurance cost prediction.
It accepts raw inputs (categorical and numeric features) and outputs a trained model saved as .pkl

Dataset: Medical Insurance Cost Prediction (insurance.csv)
Features:
  - age: Age of the person (numeric)
  - sex: Gender (categorical: 'male', 'female')
  - bmi: Body Mass Index (numeric)
  - children: Number of children/dependents (numeric)
  - smoker: Smoking status (categorical: 'yes', 'no')
  - region: Geographic region (categorical: 'southeast', 'southwest', 'northeast', 'northwest')
  - charges: Medical insurance cost (target, numeric - regression)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MEDICAL INSURANCE COST PIPELINE - Build & Export")
print("=" * 80)

# ============================================
# CONFIGURATION: Define features
# ============================================
print("\n1. Configuring feature definitions...")

categorical_features = [
    'sex',       # Gender: 'male' or 'female'
    'smoker',    # Smoking status: 'yes' or 'no'
    'region'     # Region: 'southeast', 'southwest', 'northeast', 'northwest'
]

numeric_features = [
    'age',       # Age
    'bmi',       # Body Mass Index
    'children'   # Number of children
]

target_column = 'charges'

print(f"Categorical features: {categorical_features}")
print(f"Numeric features: {numeric_features}")
print(f"Target: {target_column}")

# ============================================
# STEP 1: Load Dataset
# ============================================
print("\n2. Loading dataset...")

data_path = 'insurance.csv'  # Ensure you run this script from the directory containing insurance.csv
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget statistics:")
print(df[target_column].describe())

# ============================================
# STEP 2: Clean Data
# ============================================
print("\n3. Cleaning data...")

# Drop duplicates
initial_rows = len(df)
df = df.drop_duplicates()
dropped = initial_rows - len(df)
print(f"Removed {dropped} duplicate rows")

# Check for missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"Missing values found:\n{missing[missing > 0]}")
    df = df.dropna()
    print(f"Dropped rows with missing values. New shape: {df.shape}")
else:
    print("No missing values found")

# ============================================
# STEP 3: Build Pipeline
# ============================================
print("\n4. Building pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=1000,
        max_depth=50,
        min_samples_split=7,
        min_samples_leaf=12,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ))
])

print("[OK] Pipeline created")
print(f"  Categorical: {categorical_features} -> OneHotEncoder (drop='first')")
print(f"  Numeric: {numeric_features} -> StandardScaler")
print(f"  Model: RandomForestRegressor with 1000 estimators")

# ============================================
# STEP 4: Prepare Training Data
# ============================================
print("\n5. Preparing training data...")

# Select features and target
all_features = categorical_features + numeric_features
X = df[all_features]
y = df[target_column]

print(f"Training with {len(X)} samples and {len(all_features)} raw features")
print(f"Feature columns: {all_features}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================
# STEP 5: Train Pipeline
# ============================================
print("\n6. Training pipeline...")
pipeline.fit(X_train, y_train)
print("[OK] Training complete!")

# ============================================
# STEP 6: Evaluate Model
# ============================================
print("\n7. Evaluating model...")

# Predictions
y_pred = pipeline.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score (Test): {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"\nSample predictions (first 10):")
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
    print(f"  Predicted: ${y_pred[i]:,.2f}, Actual: ${actual:,.2f}")

# ============================================
# STEP 7: Test with Raw Inputs
# ============================================
print("\n8. Testing with raw inputs...")

# Example raw input with categorical STRING values
test_raw = pd.DataFrame([{
    'age': 30,                    # Numeric
    'sex': 'male',                # Categorical string
    'bmi': 25.0,                  # Numeric
    'children': 2,                # Numeric
    'smoker': 'no',               # Categorical string
    'region': 'southeast'         # Categorical string
}])

print(f"Test input (RAW with categorical strings):")
print(test_raw.to_string(index=False))

prediction = pipeline.predict(test_raw)
print(f"\nPredicted insurance charge: ${prediction[0]:,.2f}")
print("[OK] Prediction successful with categorical string inputs!")

# ============================================
# STEP 8: Save Pipeline
# ============================================
print("\n9. Saving pipeline...")
output_file = 'medical_insurance_pipeline.pkl'
joblib.dump(pipeline, output_file)
print(f"[OK] Saved to: {output_file}")

# ============================================
# STEP 9: Verify Saved Pipeline
# ============================================
print("\n10. Verifying saved pipeline...")
loaded_pipeline = joblib.load(output_file)
loaded_pred = loaded_pipeline.predict(test_raw)
print(f"Loaded pipeline prediction: ${loaded_pred[0]:,.2f}")
print(f"Match: {abs(prediction[0] - loaded_pred[0]) < 0.01}")
print("[OK] Pipeline saved and loaded correctly!")

# ============================================
# STEP 10: Extract Feature Schema for Frontend
# ============================================
print("\n11. Extracting feature schema for frontend integration...")

raw_feature_names = []

# Get raw feature names from preprocessor
if hasattr(loaded_pipeline, 'named_steps'):
    preprocessor = loaded_pipeline.named_steps.get('preprocessing')
    if preprocessor and hasattr(preprocessor, 'feature_names_in_'):
        raw_feature_names = preprocessor.feature_names_in_.tolist()
        print(f"Raw input features: {raw_feature_names}")
    else:
        raw_feature_names = all_features
        print(f"Using defined features: {raw_feature_names}")

print("\nExpected input features for frontend (RAW):")
print(f"  Categorical: {categorical_features}")
print(f"  Numeric: {numeric_features}")

# Generate sample categorical options from dataset
print("\nSample values for categorical features (from dataset):")
for cat_feat in categorical_features:
    unique_vals = df[cat_feat].unique()[:10]
    print(f"  {cat_feat}: {list(unique_vals)}")

# ============================================
# STEP 11: Summary
# ============================================
print("\n" + "=" * 80)
print("PIPELINE BUILD COMPLETE")
print("=" * 80)
print(f"[OK] Pipeline trained with {len(X_train)} samples")
print(f"[OK] Test R² Score: {r2:.4f}, RMSE: {rmse:.4f}")
print(f"[OK] Model saved to: {output_file}")
print(f"[OK] Accepts RAW categorical inputs (e.g., 'male', 'female', 'yes', 'no', 'southeast')")
print(f"[OK] Accepts RAW numeric inputs (age, bmi, children)")
print("\nNEXT STEPS:")
print("1. Upload the .pkl file to your XAI platform via /api/v1/models/upload")
print("2. The frontend will automatically detect categorical features and render dropdowns")
print("3. Users can input values like: sex='male', smoker='no', region='southwest', etc.")
print("=" * 80)
