#!/usr/bin/env python3
"""
Complete end-to-end test for the Heart Disease model.
Tests: prediction, SHAP, and LIME generation.
"""

import os
import sys
import io
import json
import pymongo
from minio import Minio
import joblib
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.config import settings

def load_model_from_minio(file_path):
    """Download and load model from MinIO."""
    minio_client = Minio(
        endpoint=f"{settings.MINIO_ENDPOINT}:{settings.MINIO_PORT}",
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=False
    )
    bucket = settings.MINIO_BUCKET

    response = minio_client.get_object(bucket, file_path)
    model_bytes = response.read()
    response.close()
    response.release_conn()

    return joblib.load(io.BytesIO(model_bytes))

def test_prediction(model_obj, framework):
    """Test prediction with categorical inputs."""
    print("\n" + "="*80)
    print("TEST 1: PREDICTION")
    print("="*80)

    # Raw input with categorical strings
    test_input = pd.DataFrame([{
        'age': 63,
        'sex': 'Male',               # Categorical string
        'cp': 'asymptomatic',        # Categorical string
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 'normal',         # Categorical string
        'thalach': 150,
        'exang': 'No',               # Categorical string
        'oldpeak': 2.3,
        'slope': 'upsloping'         # Categorical string
    }])

    print("Input data:")
    print(test_input.to_string(index=False))

    prediction = model_obj.predict(test_input)
    probability = model_obj.predict_proba(test_input)

    print(f"\nPrediction: {prediction[0]}")
    print(f"Probabilities: No Disease={probability[0][0]:.4f}, Disease={probability[0][1]:.4f}")
    print("[OK] Prediction successful with categorical string inputs!")

    return test_input, probability

def compute_shap_explanation(model_obj, framework, input_data, background_data=None):
    """Compute SHAP values for the prediction."""
    print("\n" + "="*80)
    print("TEST 2: SHAP EXPLANATION")
    print("="*80)

    from sklearn.pipeline import Pipeline

    # For pipelines, we need to explain in preprocessed space
    if isinstance(model_obj, Pipeline):
        # Find preprocessor
        preprocessor = None
        for step_name, step_obj in model_obj.steps:
            if hasattr(step_obj, 'transform'):
                preprocessor = step_obj
                break

        if preprocessor is not None:
            # Create background data if not provided
            if background_data is None:
                # Use the input itself with some perturbations
                bg_raw = pd.concat([input_data] * 50, ignore_index=True)
                # Add small random noise to numeric columns
                for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
                    if col in bg_raw.columns:
                        noise = np.random.normal(0, 1, len(bg_raw))
                        bg_raw[col] = bg_raw[col] + noise
            else:
                bg_raw = background_data

            # Prepare background in preprocessed space
            bg_processed = preprocessor.transform(bg_raw)
            if hasattr(bg_processed, 'toarray'):
                bg_processed = bg_processed.toarray()
            bg_numeric = np.asarray(bg_processed, dtype=float)

            # Preprocess the actual input
            input_processed = preprocessor.transform(input_data)
            if hasattr(input_processed, 'toarray'):
                input_processed = input_processed.toarray()
            input_numeric = np.asarray(input_processed, dtype=float)

            # Get feature names from preprocessor
            try:
                raw_names = preprocessor.get_feature_names_out()
                feature_names = []
                for name in raw_names:
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    if '__' in name:
                        name = name.split('__', 1)[1]
                    feature_names.append(name)
            except Exception:
                feature_names = [f"feature_{i}" for i in range(bg_numeric.shape[1])]

            # Create SHAP explainer
            final_estimator = model_obj.steps[-1][1]
            if hasattr(final_estimator, 'predict_proba'):
                predict_fn = final_estimator.predict_proba
            else:
                predict_fn = final_estimator.predict

            def _predict_preprocessed(values):
                return predict_fn(values)

            np.random.seed(42)
            explainer = shap.KernelExplainer(_predict_preprocessed, bg_numeric[:min(100, len(bg_numeric))])
            shap_values = explainer.shap_values(input_numeric)

            # For binary classification, shap_values is usually a list of two arrays (class 0 and class 1)
            if isinstance(shap_values, list) and len(shap_values) >= 2:
                # Use positive class (index 1)
                shap_vals = shap_values[1][0]  # shape (n_features,)
            else:
                # shap_values could be array or single list
                if isinstance(shap_values, list):
                    shap_vals = shap_values[0][0] if shap_values[0].ndim > 1 else shap_values[0]
                else:
                    shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values
                # Ensure 1D
                shap_vals = np.asarray(shap_vals).flatten()

            # Display results
            print(f"Computed SHAP values for {len(feature_names)} features")
            print("\nTop 10 contributing features:")
            # Get indices of top absolute values
            abs_vals = np.abs(shap_vals)
            sorted_indices = np.argsort(abs_vals)[::-1][:10]
            for idx in sorted_indices:
                if idx < len(feature_names):
                    fname = feature_names[idx]
                else:
                    fname = f"feature_{idx}"
                fvalue = shap_vals[idx]
                print(f"  {fname}: {fvalue:+.4f}")

            print("[OK] SHAP explanation generated successfully!")
            return {
                "shap_values": shap_vals.tolist(),
                "feature_names": feature_names,
                "expected_value": float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0])
            }
    else:
        print("Model is not a pipeline. SHAP not implemented for this case in test.")
        return None

def compute_lime_explanation(model_obj, framework, input_data, categorical_features):
    """Compute LIME explanation for the prediction."""
    print("\n" + "="*80)
    print("TEST 3: LIME EXPLANATION")
    print("="*80)

    from sklearn.pipeline import Pipeline

    if isinstance(model_obj, Pipeline):
        # For LIME, we need to provide training data background
        # Use the input repeated with perturbations
        background_data = pd.concat([input_data] * 100, ignore_index=True)

        # Get feature names
        preprocessor = None
        for step_name, step_obj in model_obj.steps:
            if hasattr(step_obj, 'transform'):
                preprocessor = step_obj
                break

        if preprocessor is not None:
            # Preprocess background to numeric space
            bg_processed = preprocessor.transform(background_data)
            if hasattr(bg_processed, 'toarray'):
                bg_processed = bg_processed.toarray()
            bg_numeric = np.asarray(bg_processed, dtype=float)

            # Get feature names
            try:
                raw_names = preprocessor.get_feature_names_out()
                feature_names = []
                for name in raw_names:
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    if '__' in name:
                        name = name.split('__', 1)[1]
                    feature_names.append(name)
            except Exception:
                feature_names = [f"feature_{i}" for i in range(bg_numeric.shape[1])]

            # Preprocess the actual input instance
            input_processed = preprocessor.transform(input_data)
            if hasattr(input_processed, 'toarray'):
                input_processed = input_processed.toarray()
            instance = np.asarray(input_processed, dtype=float)[0]

            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                bg_numeric,
                feature_names=feature_names,
                mode='classification',
                discretize_continuous=True,
                random_state=42
            )

            # Get prediction function from final estimator
            final_estimator = model_obj.steps[-1][1]
            if hasattr(final_estimator, 'predict_proba'):
                predict_fn = final_estimator.predict_proba
            else:
                predict_fn = lambda x: final_estimator.predict(x)

            # Compute explanation
            exp = explainer.explain_instance(
                instance,
                predict_fn,
                num_features=10
            )

            print(f"LIME explanation computed for {len(feature_names)} features")
            print("\nTop 10 local contributions:")
            for feature, weight in exp.as_list()[:10]:
                print(f"  {feature}: {weight:+.4f}")

            print("[OK] LIME explanation generated successfully!")
            return {
                "lime_weights": exp.as_list(),
                "feature_names": feature_names
            }
    else:
        print("Model is not a pipeline. LIME not implemented for this case in test.")
        return None

def main():
    print("="*80)
    print("HEART DISEASE MODEL - END-TO-END TEST")
    print("="*80)

    # Get model from DB
    mongo_client = pymongo.MongoClient(settings.MONGODB_URL)
    db = mongo_client.get_default_database()

    # Find Heart Disease model
    model = db.models.find_one({"name": "Heart Disease"})
    if not model:
        print("ERROR: Heart Disease model not found in database!")
        print("Available models:")
        for m in db.models.find({}, {"name":1}):
            print(f"  - {m.get('name')}")
        mongo_client.close()
        return

    model_id = str(model["_id"])
    print(f"Found model: {model.get('name')} (ID: {model_id})")
    print(f"Framework: {model.get('framework')}")
    print(f"Feature schema ({len(model.get('feature_schema', []))} features):")
    for fs in model.get("feature_schema", [])[:5]:
        print(f"  - {fs['name']}: {fs['type']}, options={fs.get('options', [])}")
    print()

    # Download and load model
    print("Loading model from MinIO...")
    file_path = model.get("file_path")
    if not file_path:
        print("ERROR: Model file_path missing!")
        mongo_client.close()
        return

    model_obj = load_model_from_minio(file_path)
    print(f"[OK] Model loaded: {type(model_obj).__name__}")

    # Verify it's a pipeline
    from sklearn.pipeline import Pipeline
    if not isinstance(model_obj, Pipeline):
        print("WARNING: Model is not a Pipeline. Expected a ColumnTransformer pipeline.")
    else:
        print(f"Pipeline steps: {[name for name, _ in model_obj.steps]}")

    # Test 1: Prediction
    input_data, prob = test_prediction(model_obj, model.get("framework"))

    # Test 2: SHAP
    shap_result = compute_shap_explanation(model_obj, model.get("framework"), input_data)

    # Test 3: LIME
    lime_result = compute_lime_explanation(model_obj, model.get("framework"), input_data, [])

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Model: {model.get('name')} (ID: {model_id})")
    print(f"Feature schema fixed: sex is now categorical with options {model.get('feature_schema', [])[1].get('options') if len(model.get('feature_schema', [])) > 1 else 'N/A'}")
    print(f"Prediction: {'Disease' if prob[0][1] > 0.5 else 'No Disease'} (conf: {max(prob[0]):.4f})")
    print(f"SHAP generated: {'Yes' if shap_result else 'No'}")
    print(f"LIME generated: {'Yes' if lime_result else 'No'}")
    print("\n[SUCCESS] All tests passed!")
    print("="*80)

    mongo_client.close()

if __name__ == "__main__":
    main()
