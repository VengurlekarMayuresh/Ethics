"""
Test the full API flow: Upload pipeline model, predict, SHAP, LIME
"""

import requests
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Configuration
BASE_URL = "http://localhost:8000"  # Change if your backend runs elsewhere
AUTH_URL = f"{BASE_URL}/auth/login"
UPLOAD_URL = f"{BASE_URL}/api/v1/models"  # Adjust based on your actual upload endpoint
# Your API likely has a prediction endpoint; check actual routes
PREDICT_URL = None  # Will be set after getting model_id

# Test user credentials (UPDATE THESE!)
USERNAME = "test@example.com"
PASSWORD = "your_password"

# Paths
PIPELINE_FILE = "test_pipeline.pkl"
TEST_CSV = "test_inputs.csv"  # Optional: CSV with test data

def get_token():
    """Authenticate and get JWT token."""
    print("Authenticating...")
    response = requests.post(AUTH_URL, json={
        "username": USERNAME,
        "password": PASSWORD
    })
    if response.status_code == 200:
        token = response.json()["access_token"]
        print("[OK] Authenticated successfully")
        return token
    else:
        print(f"[FAIL] Auth failed: {response.status_code} - {response.text}")
        raise Exception("Authentication failed")

def upload_model(token):
    """Upload the pipeline model file."""
    print(f"\nUploading model: {PIPELINE_FILE}")

    headers = {"Authorization": f"Bearer {token}"}

    # Prepare multipart form data
    with open(PIPELINE_FILE, 'rb') as f:
        files = {
            'file': (PIPELINE_FILE, f, 'application/octet-stream')
        }
        data = {
            'name': 'Test Pipeline Model',
            'description': 'Pipeline with raw categorical input support',
            'framework': 'sklearn',  # Should be auto-detected
            'task_type': 'classification',
            'feature_schema': '[]'  # Empty = auto-generate
        }

        response = requests.post(UPLOAD_URL, headers=headers, files=files, data=data)

    if response.status_code in [200, 201]:
        model_data = response.json()
        model_id = model_data["_id"]
        print(f"[OK] Model uploaded successfully! ID: {model_id}")
        print(f"Feature schema extracted: {len(model_data.get('feature_schema', []))} features")
        for feat in model_data.get('feature_schema', []):
            print(f"  - {feat['name']}: {feat['type']}")
        return model_id
    else:
        print(f"[FAIL] Upload failed: {response.status_code} - {response.text}")
        raise Exception("Upload failed")

def create_test_csv():
    """Create a test CSV file with raw inputs."""
    test_data = pd.DataFrame([
        {
            'gender': 'Male',
            'education': 'Graduate',
            'age': 35,
            'income': 75000,
            'loan_amount': 25000,
            'credit_score': 720
        },
        {
            'gender': 'Female',
            'education': 'Not Graduate',
            'age': 45,
            'income': 80000,
            'loan_amount': 30000,
            'credit_score': 780
        },
        {
            'gender': 'Male',
            'education': 'Graduate',
            'age': 55,
            'income': 50000,
            'loan_amount': 15000,
            'credit_score': 580
        }
    ])
    test_data.to_csv(TEST_CSV, index=False)
    print(f"[OK] Created test CSV: {TEST_CSV}")
    return TEST_CSV

def predict_with_json(model_id, token):
    """Make prediction via JSON input."""
    global PREDICT_URL
    PREDICT_URL = f"{BASE_URL}/api/v1/predict/{model_id}"

    print(f"\nMaking prediction via JSON...")
    headers = {"Authorization": f"Bearer {token}"}

    input_data = {
        "gender": "Male",
        "education": "Graduate",
        "age": 35,
        "income": 75000,
        "loan_amount": 25000,
        "credit_score": 720
    }

    # Send as form data with input_data JSON string
    form_data = {
        'input_data': json.dumps(input_data)
    }

    response = requests.post(PREDICT_URL, headers=headers, data=form_data)

    if response.status_code == 200:
        result = response.json()
        print("[OK] Prediction successful!")
        print(f"  Prediction: {result.get('prediction')}")
        print(f"  Probabilities: {result.get('probabilities')}")
        prediction_id = result.get('prediction_id')
        print(f"  Prediction ID: {prediction_id}")
        return prediction_id
    else:
        print(f"[FAIL] Prediction failed: {response.status_code} - {response.text}")
        return None

def predict_with_csv(model_id, token):
    """Make prediction via CSV file upload."""
    global PREDICT_URL
    PREDICT_URL = f"{BASE_URL}/api/v1/predict/{model_id}"

    print(f"\nMaking prediction via CSV...")
    headers = {"Authorization": f"Bearer {token}"}

    csv_path = create_test_csv()

    with open(csv_path, 'rb') as f:
        files = {'file': (csv_path, f, 'text/csv')}
        response = requests.post(PREDICT_URL, headers=headers, files=files)

    if response.status_code == 200:
        results = response.json()
        print(f"[OK] Batch predictions successful! ({len(results)} rows)")
        for i, r in enumerate(results):
            print(f"  Row {i}: {r.get('prediction')} (confidence: {r.get('prediction_confidence', 'N/A')})")
        return True
    else:
        print(f"[FAIL] CSV prediction failed: {response.status_code} - {response.text}")
        return False

def trigger_shap(prediction_id, model_id, token):
    """Trigger SHAP computation (async task)."""
    print(f"\nTriggering SHAP computation...")

    # Endpoint: POST /api/v1/explanations/local/{model_id}
    shap_url = f"{BASE_URL}/api/v1/explanations/local/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}

    # Send input_data as form data
    form_data = {
        'input_data': json.dumps({
            'gender': 'Male',
            'education': 'Graduate',
            'age': 35,
            'income': 75000,
            'loan_amount': 25000,
            'credit_score': 720
        })
    }

    response = requests.post(shap_url, headers=headers, data=form_data)

    if response.status_code in [200, 201]:
        result = response.json()
        task_id = result.get('task_id')
        print(f"[OK] SHAP task submitted. Task ID: {task_id}")
        return task_id
    else:
        print(f"[FAIL] SHAP trigger failed: {response.status_code} - {response.text}")
        return None

def trigger_lime(prediction_id, model_id, token):
    """Trigger LIME computation (async task)."""
    print(f"\nTriggering LIME computation...")

    # Endpoint: POST /api/v1/explanations/lime/{model_id}
    lime_url = f"{BASE_URL}/api/v1/explanations/lime/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}

    form_data = {
        'input_data': json.dumps({
            'gender': 'Male',
            'education': 'Graduate',
            'age': 35,
            'income': 75000,
            'loan_amount': 25000,
            'credit_score': 720
        }),
        'num_features': 10
    }

    response = requests.post(lime_url, headers=headers, data=form_data)

    if response.status_code in [200, 201]:
        result = response.json()
        task_id = result.get('task_id')
        print(f"[OK] LIME task submitted. Task ID: {task_id}")
        return task_id
    else:
        print(f"[FAIL] LIME trigger failed: {response.status_code} - {response.text}")
        return None

def main():
    print("=" * 80)
    print("FULL API INTEGRATION TEST")
    print("=" * 80)

    # Check pipeline file exists
    if not Path(PIPELINE_FILE).exists():
        print(f"[FAIL] Pipeline file not found: {PIPELINE_FILE}")
        print("Please run the training script first to generate the pipeline.")
        return

    try:
        # 1. Authenticate
        token = get_token()

        # 2. Upload model
        model_id = upload_model(token)

        # 3. Test JSON prediction
        prediction_id = predict_with_json(model_id, token)

        # 4. Test CSV prediction
        csv_success = predict_with_csv(model_id, token)

        # 5. Trigger SHAP
        if prediction_id:
            shap_task = trigger_shap(prediction_id, model_id, token)

            # Wait and check task status
            if shap_task:
                print("\nWaiting for SHAP to complete...")
                time.sleep(5)
                task_status = requests.get(
                    f"{BASE_URL}/api/v1/tasks/{shap_task}",
                    headers={"Authorization": f"Bearer {token}"}
                )
                if task_status.status_code == 200:
                    print(f"SHAP task status: {task_status.json().get('status')}")

        # 6. Trigger LIME
        if prediction_id:
            lime_task = trigger_lime(prediction_id, model_id, token)

            if lime_task:
                print("\nWaiting for LIME to complete...")
                time.sleep(5)
                task_status = requests.get(
                    f"{BASE_URL}/api/v1/tasks/{lime_task}",
                    headers={"Authorization": f"Bearer {token}"}
                )
                if task_status.status_code == 200:
                    print(f"LIME task status: {task_status.json().get('status')}")

        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        print(f"Model ID: {model_id}")
        print(f"Prediction ID: {prediction_id}")
        print("\nNext manual checks:")
        print("1. Verify feature schema in DB matches raw features")
        print("2. Check SHAP explanation stored in explanations collection")
        print("3. Check LIME explanation stored in explanations collection")
        print("4. Verify frontend form shows raw inputs (gender: Male/Female)")

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
