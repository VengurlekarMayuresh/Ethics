#!/usr/bin/env python3
"""
Car Price Prediction Pipeline
This script builds and exports a pipeline for car price prediction.
It accepts raw inputs (categorical and numeric features) and outputs a trained model saved as .pkl

Dataset: CAR DETAILS FROM CAR DEKHO.csv
Features:
  - name: Car model name (categorical)
  - year: Manufacturing year (numeric)
  - km_driven: Kilometers driven (numeric)
  - fuel: Fuel type (categorical)
  - seller_type: Seller type (categorical)
  - transmission: Transmission type (categorical)
  - owner: Owner type (categorical)
  - selling_price: Target variable (regression)
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
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CAR PRICE PREDICTION PIPELINE - Build & Export")
print("=" * 80)

# ============================================
# CONFIGURATION: Define features
# ============================================
print("\n1. Configuring feature definitions...")

categorical_features = [
    'name',      # Car model name (e.g., 'Maruti 800 AC', 'Hyundai Verna')
    'fuel',      # Fuel type (e.g., 'Petrol', 'Diesel', 'CNG', 'LPG', 'Electric')
    'seller_type',  # Seller type (e.g., 'Individual', 'Dealer', 'Trustmark Dealer')
    'transmission', # Transmission (e.g., 'Manual', 'Automatic')
    'owner'      # Owner type (e.g., 'First Owner', 'Second Owner', 'Third Owner', etc.)
]

numeric_features = [
    'year',      # Manufacturing year
    'km_driven'  # Kilometers driven
]

# Note: 'selling_price' is the target, NOT an input feature
target_column = 'selling_price'

print(f"Categorical features: {categorical_features}")
print(f"Numeric features: {numeric_features}")
print(f"Target: {target_column}")

# ============================================
# STEP 1: Load Dataset
# ============================================
print("\n2. Loading dataset...")

data_path = 'CAR DETAILS FROM CAR DEKHO.csv'  # Ensure you run this script from the project root directory
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget distribution:")
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
    # Drop rows with missing values for simplicity
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
        n_estimators=500,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ))
])

print("[OK] Pipeline created")
print(f"  Categorical: {categorical_features} -> OneHotEncoder (drop='first')")
print(f"  Numeric: {numeric_features} -> StandardScaler")
print(f"  Model: RandomForestRegressor with tuned hyperparameters")

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

# Evaluate
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"Train R^2 score: {train_score:.4f}")
print(f"Test R^2 score: {test_score:.4f}")

# ============================================
# STEP 6: Test with Raw Inputs
# ============================================
print("\n7. Testing with raw inputs...")

# Example raw input - adjust based on your features
test_raw = pd.DataFrame([{
    'name': 'Maruti 800 AC',          # Categorical string
    'year': 2005,                   # Numeric
    'km_driven': 60000,             # Numeric
    'fuel': 'Petrol',               # Categorical string
    'seller_type': 'Individual',    # Categorical string
    'transmission': 'Manual',       # Categorical string
    'owner': 'First Owner'          # Categorical string
}])

print(f"Test input:")
print(test_raw.to_string(index=False))

prediction = pipeline.predict(test_raw)
print(f"\nPredicted selling price: ${prediction[0]:,.2f}")
print("[OK] Prediction successful with categorical string inputs!")

# ============================================
# STEP 7: Save Pipeline
# ============================================
print("\n8. Saving pipeline...")
output_file = 'car_price_pipeline.pkl'
joblib.dump(pipeline, output_file)
print(f"[OK] Saved to: {output_file}")

# ============================================
# STEP 8: Verify Saved Pipeline
# ============================================
print("\n9. Verifying saved pipeline...")
loaded_pipeline = joblib.load(output_file)
loaded_pred = loaded_pipeline.predict(test_raw)
print(f"Loaded pipeline prediction: ${loaded_pred[0]:,.2f}")
print(f"Match: {abs(prediction[0] - loaded_pred[0]) < 0.01}")
print("[OK] Pipeline saved and loaded correctly!")

# ============================================
# STEP 9: Feature Schema Extraction
# ============================================
print("\n10. Extracting feature schema...")

raw_feature_names = []

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

# ============================================
# STEP 10: Sample Categorical Options
# ============================================
print("\n11. Sample values for categorical features (from dataset):")
for cat_feat in categorical_features:
    unique_vals = df[cat_feat].unique()[:10]  # Show up to 10 unique values
    print(f"  {cat_feat}: {list(unique_vals)}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 80)
print("PIPELINE BUILD COMPLETE")
print("=" * 80)
print(f"[OK] Pipeline trained with {len(X_train)} samples")
print(f"[OK] Test R^2 score: {test_score:.4f}")
print(f"[OK] Model saved to: {output_file}")
print(f"[OK] Accepts RAW categorical inputs (e.g., 'Petrol', 'Manual', 'First Owner')")
print(f"[OK] Accepts RAW numeric inputs (year, km_driven)")
print("\nNEXT STEPS:")
print("1. Upload the .pkl file to your XAI platform via /api/v1/models/upload")
print("2. The frontend will automatically detect categorical features and render dropdowns")
print("3. Users can input values like: name='Maruti Alto', fuel='Petrol', etc.")
print("=" * 80)
