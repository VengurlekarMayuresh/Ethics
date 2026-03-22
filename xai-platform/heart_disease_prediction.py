"""
Heart Disease Prediction Pipeline
This script builds and exports a pipeline for heart disease prediction.
It accepts raw inputs (categorical and numeric features) and outputs a trained model saved as .pkl

Features are based on typical heart disease datasets (e.g., UCI Heart Disease dataset).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HEART DISEASE PREDICTION PIPELINE - Build & Export")
print("=" * 80)

# ============================================
# CONFIGURATION: Define your features here
# ============================================
print("\n1. Configuring feature definitions...")

# Actual column names in the UCI Heart Disease dataset
categorical_features = [
    'sex',  # (1 = male; 0 = female)
    'cp',   # chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)
    'restecg',  # resting electrocardiographic results
    'exang',    # exercise induced angina (1 = yes; 0 = no)
    'slope'     # slope of the peak exercise ST segment
]

numeric_features = [
    'age',
    'trestbps',  # resting blood pressure (in mm Hg on admission to the hospital)
    'chol',      # serum cholesterol in mg/dl
    'fbs',       # fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    'thalach',   # maximum heart rate achieved
    'oldpeak'    # ST depression induced by exercise relative to rest
]

# Target column name
target_column = 'target'  # Binary: 0 or 1

print(f"Categorical features: {categorical_features}")
print(f"Numeric features: {numeric_features}")
print(f"Target: {target_column}")

# ============================================
# STEP 1: Load Your Dataset
# ============================================
print("\n2. Loading dataset...")

# Update this path to your actual dataset
data_path = 'M:/Neha/Desktop/Ethics/xai-platform/heart.csv'  # Change this to your file
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget distribution:")
print(df[target_column].value_counts())

# ============================================
# STEP 2: Convert Numeric Codes to Categorical Values
# ============================================
print("\n2a. Converting numeric codes to categorical values...")

# Map numeric codes to categorical strings for proper frontend input
# UCI Heart Disease dataset mappings
df['sex'] = df['sex'].map({0: 'Female', 1: 'Male'})
df['cp'] = df['cp'].map({0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal pain', 3: 'asymptomatic'})
df['restecg'] = df['restecg'].map({0: 'normal', 1: 'ST-T wave abnormality', 2: 'left ventricular hypertrophy'})
df['exang'] = df['exang'].map({0: 'No', 1: 'Yes'})
df['slope'] = df['slope'].map({0: 'upsloping', 1: 'flat', 2: 'downsloping'})

print("[OK] Numeric codes converted to categorical strings")
print("\nSample after conversion:")
print(df[['sex', 'cp', 'restecg', 'exang', 'slope']].head())

# ============================================
# STEP 3: Handle Missing Values
# ============================================
print("\n3. Handling missing values...")

# Fill categorical with mode, numeric with median
for col in categorical_features:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

for col in numeric_features:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

print("[OK] Missing values handled")

# ============================================
# STEP 3: Build Pipeline
# ============================================
print("\n4. Building pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', KNeighborsClassifier(n_neighbors=5))
])

print("[OK] Pipeline created")

# ============================================
# STEP 4: Prepare Training Data
# ============================================
print("\n5. Preparing training data...")

# Select features
all_features = categorical_features + numeric_features
X = df[all_features]
y = df[target_column]

print(f"Training with {len(X)} samples and {len(all_features)} raw features")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================
# STEP 5: Train Pipeline
# ============================================
print("\n6. Training pipeline...")
pipeline.fit(X_train, y_train)
print("[OK] Training complete!")

accuracy = pipeline.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# ============================================
# STEP 6: Test with Raw Inputs
# ============================================
print("\n7. Testing with raw inputs...")

# Example raw input - adjust based on your features
test_raw = pd.DataFrame([{
    'age': 63,
    'sex': 'Male',
    'cp': 'asymptomatic',
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 'normal',
    'thalach': 150,
    'exang': 'No',
    'oldpeak': 2.3,
    'slope': 'upsloping'
}])

print(f"Test input:\n{test_raw}")

prediction = pipeline.predict(test_raw)
prediction_proba = pipeline.predict_proba(test_raw)

print(f"\nPrediction: {prediction[0]}")
print(f"Probabilities: Class 0: {prediction_proba[0][0]:.4f}, Class 1: {prediction_proba[0][1]:.4f}")

# ============================================
# STEP 7: Save Pipeline
# ============================================
print("\n8. Saving pipeline...")
output_file = 'heart_disease_pipeline.pkl'
joblib.dump(pipeline, output_file)
print(f"[OK] Saved to: {output_file}")

# ============================================
# STEP 8: Verify Saved Pipeline
# ============================================
print("\n9. Verifying saved pipeline...")
loaded_pipeline = joblib.load(output_file)
loaded_pred = loaded_pipeline.predict(test_raw)
print(f"Loaded pipeline prediction: {loaded_pred[0]}")
print(f"Match: {prediction[0] == loaded_pred[0]}")
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
# SUMMARY
# ============================================
print("\n" + "=" * 80)
print("PIPELINE BUILD COMPLETE")
print("=" * 80)
print(f"[OK] Pipeline trained with {len(X_train)} samples")
print(f"[OK] Test accuracy: {accuracy:.4f}")
print(f"[OK] Model saved to: {output_file}")
print(f"[OK] Accepts RAW categorical inputs (e.g., 'Male', 'Female')")
print(f"[OK] Accepts RAW numeric inputs")
print("\nNEXT STEPS:")
print("1. Upload the .pkl file to your XAI platform")
print("2. Use the categorical_features and numeric_features lists in your backend")
print("3. Frontend should accept raw values without preprocessing")
print("=" * 80)
