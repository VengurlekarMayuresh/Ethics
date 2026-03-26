#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("=" * 80)
print("FINAL CLEAN LOAN PIPELINE (RAW INPUT ONLY)")
print("=" * 80)

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv('Loan_Prediction.csv')

# Drop unnecessary column
if 'Loan_ID' in df.columns:
    df = df.drop('Loan_ID', axis=1)

# ============================================
# DEFINE FEATURES
# ============================================
categorical_features = [
    'Gender', 'Married', 'Dependents',
    'Education', 'Self_Employed', 'Property_Area'
]

numeric_features = [
    'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History'
]

target = 'Loan_Status'

# ============================================
# HANDLE MISSING VALUES
# ============================================
for col in categorical_features:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numeric_features:
    df[col] = df[col].fillna(df[col].mean())

# Encode target
df[target] = df[target].map({'Y': 1, 'N': 0})

# ============================================
# TRAIN DATA (RAW ONLY)
# ============================================
X = df[categorical_features + numeric_features]   # IMPORTANT
y = df[target]

# Ensure DataFrame (not numpy)
X = pd.DataFrame(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# PREPROCESSOR
# ============================================
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(
            drop='first',
            handle_unknown='ignore',
            sparse_output=False
        ), categorical_features),

        ('num', StandardScaler(), numeric_features)
    ]
)

# ============================================
# PIPELINE
# ============================================
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# ============================================
# TRAIN
# ============================================
pipeline.fit(X_train, y_train)

print(f"Train Accuracy: {pipeline.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {pipeline.score(X_test, y_test):.4f}")

# ============================================
# 🔥 CRITICAL FIX FOR YOUR ISSUE
# ============================================
# Force platform to see RAW features only
# pipeline.feature_names_in_ = np.array(categorical_features + numeric_features)

# ============================================
# TEST RAW INPUT
# ============================================
test_input = pd.DataFrame([{
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

pred = pipeline.predict(test_input)
prob = pipeline.predict_proba(test_input)

print("\nPrediction Test:")
print(f"Prediction: {pred[0]}")
print(f"Probability: {prob[0]}")

# ============================================
# SAVE MODEL
# ============================================
joblib.dump(pipeline, "loan_pipeline_final.pkl")

print("\n[OK] Model saved as loan_pipeline_final.pkl")
print("\n✔ Only RAW input required")
print("✔ No encoding needed (Male/Female works)")
print("✔ No derived features needed")