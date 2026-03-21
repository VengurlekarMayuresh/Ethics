"""
Pipeline training script WITHOUT custom classes.
Trains a model with preprocessing pipeline that accepts raw categorical inputs.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = "M:/Neha/Desktop/Ethics/xai-platform/train (1).csv"  # Update this to your CSV path
OUTPUT_FILE = "loan_model_pipeline.pkl"
USE_XGBOOST = True  # Set to False to use RandomForest

# ============================================
# LOAD DATA
# ============================================
print("=" * 80)
print("PIPELINE TRAINING - NO CUSTOM CLASSES")
print("=" * 80)

print(f"\n1. Loading dataset from: {DATA_PATH}")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded: {df.shape}")
except FileNotFoundError:
    print(f"   ERROR: File not found: {DATA_PATH}")
    print("   Please update DATA_PATH in this script to point to your CSV file.")
    sys.exit(1)

# ============================================
# HANDLE MISSING VALUES
# ============================================
print("\n2. Handling missing values...")

categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
for col in categorical_cols:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

numeric_cols = ['LoanAmount', 'Loan_Amount_Term']
for col in numeric_cols:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

print(f"   Missing values remaining: {df.isnull().sum().sum()}")

# ============================================
# CREATE ENGINEERED FEATURES (as regular columns)
# ============================================
print("\n3. Creating engineered features...")

# Check required columns exist
required_raw = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
missing_cols = [col for col in required_raw if col not in df.columns]
if missing_cols:
    print(f"   WARNING: Missing columns: {missing_cols}")
    print("   Skipping engineered features.")
else:
    # Log transforms (add 1 to avoid log(0))
    df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome'] + 1)
    df['LoanAmountLog'] = np.log(df['LoanAmount'] + 1)
    df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term'] + 1)

    # Total income
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Total_Income_Log'] = np.log(df['Total_Income'] + 1)

    print("   Created: ApplicantIncomeLog, LoanAmountLog, Loan_Amount_Term_Log, Total_Income_Log")

# ============================================
# DEFINE FEATURES
# ============================================
print("\n4. Defining feature sets...")

raw_categorical_features = [
    'Gender', 'Married', 'Dependents',
    'Education', 'Self_Employed', 'Property_Area'
]

raw_numeric_features = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History'
]

# Filter to only include columns that exist in df
raw_categorical_features = [col for col in raw_categorical_features if col in df.columns]
raw_numeric_features = [col for col in raw_numeric_features if col in df.columns]

print(f"   Raw categorical: {raw_categorical_features}")
print(f"   Raw numeric: {raw_numeric_features}")

# ============================================
# CREATE PREPROCESSOR
# ============================================
print("\n5. Creating preprocessing pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), raw_categorical_features),
        ('num', StandardScaler(), raw_numeric_features)
    ]
)

# ============================================
# PREPARE X and y
# ============================================
print("\n6. Preparing training data...")

if 'Loan_Status' not in df.columns:
    print("   ERROR: 'Loan_Status' column not found in dataset")
    sys.exit(1)

y = df['Loan_Status'].map({'Y': 1, 'N': 0})

# X should have ONLY raw features (categorical + numeric)
X = df[raw_categorical_features + raw_numeric_features]

print(f"   X shape: {X.shape}")
print(f"   Features: {X.columns.tolist()}")
print(f"   Target distribution:\n{y.value_counts()}")

# ============================================
# SPLIT DATA
# ============================================
print("\n7. Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================
# BUILD AND TRAIN PIPELINE
# ============================================
print("\n8. Training pipeline...")

if USE_XGBOOST:
    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model_name = "XGBoost"
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_name = "Random Forest"

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', model)
])

pipeline.fit(X_train, y_train)
print(f"   Training complete with {model_name}!")

# ============================================
# EVALUATE
# ============================================
print("\n9. Evaluating...")

accuracy = pipeline.score(X_test, y_test)
print(f"   Test Accuracy: {accuracy:.4f}")

from sklearn.metrics import classification_report
y_pred = pipeline.predict(X_test)
print("\n   Classification Report:")
print(classification_report(y_test, y_pred))

# ============================================
# SAVE PIPELINE
# ============================================
print(f"\n10. Saving pipeline to: {OUTPUT_FILE}")
joblib.dump(pipeline, OUTPUT_FILE)
print("   Saved successfully!")

# ============================================
# VERIFY WITH RAW INPUT
# ============================================
print("\n11. Verifying with raw input test...")

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

print(f"   Test sample: {test_raw.to_dict(orient='records')[0]}")

pred = pipeline.predict(test_raw)
proba = pipeline.predict_proba(test_raw)

print(f"   Prediction: {pred[0]} (0=Rejected, 1=Approved)")
print(f"   Probabilities: [{proba[0][0]:.4f}, {proba[0][1]:.4f}]")

# ============================================
# CHECK FEATURE SCHEMA
# ============================================
print("\n12. Checking feature schema extraction...")

col_transformer = pipeline.named_steps['preprocessing']
if hasattr(col_transformer, 'feature_names_in_'):
    extracted_features = col_transformer.feature_names_in_.tolist()
    expected_features = raw_categorical_features + raw_numeric_features

    print(f"   Extracted features: {extracted_features}")
    print(f"   Expected features: {expected_features}")

    if set(extracted_features) == set(expected_features):
        print("   ✓ Feature schema matches!")
    else:
        print("   ✗ Feature schema mismatch!")
        print("   This will cause issues. Check that you fit on a DataFrame with column names.")
else:
    print("   ✗ ColumnTransformer has no feature_names_in_ attribute")
    print("   Ensure you fit the pipeline on a DataFrame (not numpy array)")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Model saved to: {OUTPUT_FILE}")
print(f"Accuracy: {accuracy:.4f}")
print("\nNEXT STEPS:")
print("1. Upload", OUTPUT_FILE, "to your XAI platform")
print("2. Verify feature schema shows RAW features (gender, age, income, etc.)")
print("3. Test predictions with raw values like 'Male', 6000, etc.")
print("4. SHAP and LIME should work automatically in the backend")
