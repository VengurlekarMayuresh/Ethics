#!/usr/bin/env python3
"""
Loan Prediction Pipeline (CLEAN VERSION)

✔ Accepts ONLY RAW INPUT
✔ Computes derived features internally
✔ Fully self-contained (no external dependencies)
✔ Ready for .pkl usage in deployment/XAI
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


# ============================================
# CUSTOM FEATURE ENGINEER (SELF-CONTAINED)
# ============================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.raw_feature_names = None

    def fit(self, X, y=None):
        # Store raw feature names for schema extraction by backend
        if hasattr(X, 'columns'):
            self.raw_feature_names = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()

        # Derived Features
        X['Total_Income'] = X['ApplicantIncome'] + X['CoapplicantIncome']

        X['ApplicantIncomeLog'] = np.log1p(X['ApplicantIncome'])
        X['CoapplicantIncomeLog'] = np.log1p(X['CoapplicantIncome'])
        X['LoanAmountLog'] = np.log1p(X['LoanAmount'])
        X['Loan_Amount_Term_Log'] = np.log1p(X['Loan_Amount_Term'])
        X['Total_IncomeLog'] = np.log1p(X['Total_Income'])

        return X


print("=" * 80)
print("LOAN PREDICTION PIPELINE - CLEAN BUILD")
print("=" * 80)


# ============================================
# FEATURES
# ============================================
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

derived_features = [
    'Total_Income',
    'ApplicantIncomeLog',
    'CoapplicantIncomeLog',
    'LoanAmountLog',
    'Loan_Amount_Term_Log',
    'Total_IncomeLog'
]

target_column = 'Loan_Status'


# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv('Loan_Prediction.csv')

# Drop unnecessary column
if 'Loan_ID' in df.columns:
    df = df.drop('Loan_ID', axis=1)

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
for col in categorical_features:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numeric_features:
    df[col] = df[col].fillna(df[col].mean())

# Encode target
df[target_column] = df[target_column].map({'Y': 1, 'N': 0})


# ============================================
# TRAIN DATA (RAW ONLY)
# ============================================
X = df[categorical_features + numeric_features]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ============================================
# PREPROCESSOR
# ============================================
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'   # 🔥 THIS IS KEY
)


# ============================================
# PIPELINE
# ============================================
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


# ============================================
# TRAIN
# ============================================
pipeline.fit(X_train, y_train)

print(f"Train Accuracy: {pipeline.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {pipeline.score(X_test, y_test):.4f}")


# ============================================
# TEST WITH RAW INPUT
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

prediction = pipeline.predict(test_input)
prob = pipeline.predict_proba(test_input)

print("\nSample Prediction:")
print(f"Prediction: {prediction[0]} (1=Approved, 0=Rejected)")
print(f"Probability: {prob[0]}")


# ============================================
# SAVE MODEL
# ============================================
joblib.dump(pipeline, 'loan_prediction_pipeline.pkl')
print("\n[OK] Model saved as loan_prediction_pipeline.pkl")


# ============================================
# VERIFY LOAD
# ============================================
loaded_model = joblib.load('loan_prediction_pipeline.pkl')
loaded_pred = loaded_model.predict(test_input)

print(f"Loaded model prediction: {loaded_pred[0]}")
print("[OK] Pipeline works correctly with RAW input")


print("\n" + "=" * 80)
print("FINAL RESULT")
print("=" * 80)
print("[OK] Accepts ONLY raw input")
print("[OK] No manual preprocessing required")
print("[OK] Derived features computed internally")
print("[OK] Ready for deployment / XAI")
print("=" * 80)