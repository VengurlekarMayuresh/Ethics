#!/usr/bin/env python3
"""
Titanic Survival Prediction Pipeline
This script builds and exports a pipeline for Titanic survival prediction.
It accepts raw inputs (categorical and numeric features) and outputs a trained model saved as .pkl

Dataset: titanic.csv (from Kaggle Titanic competition)
Features:
  - Pclass: Passenger class (1, 2, 3) - numeric/categorical
  - Name: Passenger name (text, will extract title)
  - Sex: Gender (categorical)
  - Age: Age in years (numeric)
  - SibSp: Number of siblings/spouses aboard (numeric)
  - Parch: Number of parents/children aboard (numeric)
  - Ticket: Ticket number (text, drop)
  - Fare: Passenger fare (numeric)
  - Cabin: Cabin number (text, drop due to many missing)
  - Embarked: Port of embarkation (C, Q, S) - categorical
  - Survived: Target variable (binary classification: 0 = died, 1 = survived)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TITANIC SURVIVAL PREDICTION PIPELINE - Build & Export")
print("=" * 80)

# ============================================
# CONFIGURATION: Define features
# ============================================
print("\n1. Configuring feature definitions...")

categorical_features = [
    'Sex',       # Gender: male, female
    'Embarked',  # Port: C, Q, S
]

numeric_features = [
    'Pclass',    # Passenger class (1, 2, 3)
    'Age',       # Age (may have missing values)
    'SibSp',     # Siblings/spouses
    'Parch',     # Parents/children
    'Fare',      # Fare paid
]

target_column = 'Survived'

print(f"Categorical features: {categorical_features}")
print(f"Numeric features: {numeric_features}")
print(f"Target: {target_column}")

# ============================================
# STEP 1: Load Dataset
# ============================================
print("\n2. Loading dataset...")

# Try multiple possible filenames and choose one that includes target labels.
# Some Titanic files (e.g., Kaggle test.csv) do not contain the Survived column.
possible_filenames = ['titanic.csv', 'titanic_dataset.csv', 'train.csv', 'test.csv']
data_path = None
df = None
files_found = []

for fname in possible_filenames:
    if not os.path.exists(fname):
        continue

    files_found.append(fname)
    candidate_df = pd.read_csv(fname)

    # Prefer a dataset that has labels so the model can be trained.
    if target_column in candidate_df.columns:
        data_path = fname
        df = candidate_df
        break

if df is None:
    if not files_found:
        raise FileNotFoundError(
            f"Titanic dataset not found. Please ensure one of these files exists: {possible_filenames}"
        )

    raise ValueError(
        "No training dataset with target column 'Survived' was found. "
        f"Files detected: {files_found}. "
        "You are likely pointing to Kaggle test data (which has no labels). "
        "Please provide a training CSV that includes the 'Survived' column "
        "(usually train.csv)."
    )

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget distribution:")
print(df[target_column].value_counts())

# ============================================
# STEP 2: Clean and Preprocess Data
# ============================================
print("\n3. Cleaning and preprocessing data...")

# Drop unnecessary columns
columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
print(f"Dropped columns: {columns_to_drop}")

# Handle missing values
print(f"\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing Age with median
if 'Age' in df.columns and df['Age'].isnull().any():
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)
    print(f"Filled missing Age with median: {age_median:.1f}")

# Fill missing Embarked with mode
if 'Embarked' in df.columns and df['Embarked'].isnull().any():
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    print(f"Filled missing Embarked with mode: {embarked_mode}")

# Fill missing Fare with median (for test set, might not have in train)
if 'Fare' in df.columns and df['Fare'].isnull().any():
    fare_median = df['Fare'].median()
    df['Fare'] = df['Fare'].fillna(fare_median)
    print(f"Filled missing Fare with median: {fare_median:.1f}")

# Drop any remaining rows with missing values
initial_rows = len(df)
df = df.dropna()
dropped = initial_rows - len(df)
if dropped > 0:
    print(f"Dropped {dropped} rows with remaining missing values")
else:
    print("No rows dropped after filling missing values")

print(f"\nMissing values after cleaning:")
print(df.isnull().sum())

# Convert Pclass to numeric (ensure it's integer)
df['Pclass'] = pd.to_numeric(df['Pclass'], errors='coerce')
if df['Pclass'].isnull().any():
    df['Pclass'] = df['Pclass'].fillna(df['Pclass'].median())

# Ensure target is integer
df[target_column] = df[target_column].astype(int)

print(f"\nFinal dataset shape: {df.shape}")
print(f"Feature columns: {categorical_features + numeric_features}")
print(f"Target distribution:\n{df[target_column].value_counts()}")

# ============================================
# STEP 4: Build Pipeline
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
    ('model', RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ))
])

print("[OK] Pipeline created")
print(f"  Categorical: {categorical_features} -> OneHotEncoder (drop='first')")
print(f"  Numeric: {numeric_features} -> StandardScaler")
print(f"  Model: RandomForestClassifier (binary classification)")

# ============================================
# STEP 5: Prepare Training Data
# ============================================
print("\n5. Preparing training data...")

all_features = categorical_features + numeric_features
X = df[all_features]
y = df[target_column]

print(f"Training with {len(X)} samples and {len(all_features)} raw features")
print(f"Feature columns: {all_features}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
print(f"Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")

# ============================================
# STEP 6: Train Pipeline
# ============================================
print("\n6. Training pipeline...")
pipeline.fit(X_train, y_train)
print("[OK] Training complete!")

# Evaluate
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Cross-validation might be better but we'll keep it simple

# ============================================
# STEP 7: Test with Raw Inputs
# ============================================
print("\n7. Testing with raw inputs...")

# Example raw input - typical Titanic passenger
test_raw = pd.DataFrame([{
    'Pclass': 3,
    'Sex': 'male',
    'Age': 22,
    'SibSp': 1,
    'Parch': 0,
    'Fare': 7.25,
    'Embarked': 'S'
}])

print(f"Test input:")
print(test_raw.to_string(index=False))

prediction = pipeline.predict(test_raw)
probability = pipeline.predict_proba(test_raw)
print(f"\nPredicted survival: {'Survived (1)' if prediction[0] == 1 else 'Did not survive (0)'}")
print(f"Probabilities: [Died: {probability[0][0]:.3f}, Survived: {probability[0][1]:.3f}]")
print("[OK] Prediction successful with categorical string inputs!")

# ============================================
# STEP 8: Save Pipeline
# ============================================
print("\n8. Saving pipeline...")
output_file = 'titanic_survival_pipeline.pkl'
joblib.dump(pipeline, output_file)
print(f"[OK] Saved to: {output_file}")

# ============================================
# STEP 9: Verify Saved Pipeline
# ============================================
print("\n9. Verifying saved pipeline...")
loaded_pipeline = joblib.load(output_file)
loaded_pred = loaded_pipeline.predict(test_raw)
loaded_proba = loaded_pipeline.predict_proba(test_raw)
print(f"Loaded pipeline prediction: {'Survived' if loaded_pred[0] == 1 else 'Did not survive'}")
print(f"Probabilities match: {abs(probability[0][0] - loaded_proba[0][0]) < 0.01 and abs(probability[0][1] - loaded_proba[0][1]) < 0.01}")
print("[OK] Pipeline saved and loaded correctly!")

# ============================================
# STEP 10: Generate Feature Schema for Frontend
# ============================================
print("\n10. Extracting feature schema for frontend...")

feature_schema = []

# For numeric features
for feat in numeric_features:
    feature_schema.append({
        "name": feat,
        "type": "numeric",
        "min": float(df[feat].min()) if not pd.isna(df[feat].min()) else None,
        "max": float(df[feat].max()) if not pd.isna(df[feat].max()) else None,
        "mean": float(df[feat].mean()) if not pd.isna(df[feat].mean()) else None,
        "options": []
    })

# For categorical features - get unique options
for feat in categorical_features:
    unique_vals = df[feat].dropna().unique().tolist()
    # Convert all to string
    unique_vals_str = [str(val) for val in unique_vals]
    feature_schema.append({
        "name": feat,
        "type": "categorical",
        "options": unique_vals_str,
        "min": None,
        "max": None,
        "mean": None
    })

print("Feature Schema (for frontend upload):")
for fs in feature_schema:
    print(f"  - {fs['name']} ({fs['type']})")
    if fs['type'] == 'categorical':
        print(f"      Options: {fs['options'][:5]}{'...' if len(fs['options']) > 5 else ''}")

# Save schema to JSON for reference
import json
schema_file = 'titanic_feature_schema.json'
with open(schema_file, 'w') as f:
    json.dump(feature_schema, f, indent=2)
print(f"\n[OK] Feature schema saved to: {schema_file}")

# ============================================
# STEP 11: Sample Inputs for Testing
# ============================================
print("\n11. Sample input combinations for testing:")

print("\n  a) Young male in 3rd class (high risk):")
sample1 = {
    'Pclass': 3,
    'Sex': 'male',
    'Age': 20,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 7.5,
    'Embarked': 'S'
}
print(f"      {sample1}")
pred1 = pipeline.predict(pd.DataFrame([sample1]))[0]
print(f"      Prediction: {'Survived' if pred1 == 1 else 'Did not survive'}")

print("\n  b) Young female in 1st class (low risk):")
sample2 = {
    'Pclass': 1,
    'Sex': 'female',
    'Age': 25,
    'SibSp': 1,
    'Parch': 0,
    'Fare': 100,
    'Embarked': 'C'
}
print(f"      {sample2}")
pred2 = pipeline.predict(pd.DataFrame([sample2]))[0]
print(f"      Prediction: {'Survived' if pred2 == 1 else 'Did not survive'}")

print("\n  c) Elderly male in 2nd class:")
sample3 = {
    'Pclass': 2,
    'Sex': 'male',
    'Age': 60,
    'SibSp': 0,
    'Parch': 1,
    'Fare': 20,
    'Embarked': 'Q'
}
print(f"      {sample3}")
pred3 = pipeline.predict(pd.DataFrame([sample3]))[0]
print(f"      Prediction: {'Survived' if pred3 == 1 else 'Did not survive'}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 80)
print("TITANIC PIPELINE BUILD COMPLETE")
print("=" * 80)
print(f"[OK] Pipeline trained with {len(X_train)} samples")
print(f"[OK] Test accuracy: {test_score:.4f}")
print(f"[OK] Model saved to: {output_file}")
print(f"[OK] Feature schema saved to: {schema_file}")
print(f"[OK] Accepts RAW categorical inputs (e.g., 'male', 'S', 3 for Pclass)")
print(f"[OK] Accepts RAW numeric inputs (Age, Fare, SibSp, Parch)")
print(f"\nCATEGORICAL FEATURES and their values:")
for feat in categorical_features:
    vals = sorted([str(v) for v in df[feat].dropna().unique()])
    print(f"  {feat}: {vals}")
print("\nNUMERIC FEATURES ranges:")
for feat in numeric_features:
    print(f"  {feat}: [{df[feat].min():.1f}, {df[feat].max():.1f}] (mean: {df[feat].mean():.1f})")
print("\nNEXT STEPS:")
print("1. Upload the titanic_survival_pipeline.pkl file to your XAI platform via /api/v1/models/upload")
print("2. The platform will auto-detect it's a classification model (binary)")
print("3. The frontend will automatically detect categorical features and render dropdowns")
print("4. Users can input values like: Sex='female', Pclass=1, Age=25, etc.")
print("=" * 80)
