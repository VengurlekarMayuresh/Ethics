#!/usr/bin/env python3
"""
Test script to upload heart_disease_pipeline.pkl to the running backend
and verify the generated feature schema.
"""

import requests
import json
import os

BASE_URL = "http://localhost:8000"

def register_user():
    """Register a test user."""
    url = f"{BASE_URL}/api/v1/auth/register"
    data = {
        "email": "test@example.com",
        "password": "testpassword123",
        "name": "Test User"
    }
    resp = requests.post(url, json=data)
    if resp.status_code in (200, 201):
        print("[OK] User registered")
        return True
    elif resp.status_code == 400 and "already exists" in resp.text:
        print("[INFO] User already exists")
        return True
    else:
        print(f"[FAIL] Registration failed: {resp.status_code} {resp.text}")
        return False

def login():
    """Login and get JWT token."""
    url = f"{BASE_URL}/api/v1/auth/login"
    data = {
        "username": "test@example.com",
        "password": "testpassword123"
    }
    resp = requests.post(url, data=data)
    if resp.status_code == 200:
        token = resp.json()["access_token"]
        print("[OK] Logged in, got token")
        return token
    else:
        print(f"[FAIL] Login failed: {resp.status_code} {resp.text}")
        return None

def upload_model(token, model_path):
    """Upload the model file."""
    url = f"{BASE_URL}/api/v1/models/upload"
    headers = {"Authorization": f"Bearer {token}"}

    if not os.path.exists(model_path):
        print(f"[FAIL] Model file not found: {model_path}")
        return None

    with open(model_path, 'rb') as f:
        files = {'file': (os.path.basename(model_path), f, 'application/octet-stream')}
        data = {
            'name': 'Heart Disease Model',
            'description': 'Heart disease prediction model with categorical features',
            'framework': 'sklearn',
            'task_type': 'classification',
            'feature_schema': '[]'  # Auto-generate
        }
        print(f"[INFO] Uploading model {model_path}...")
        resp = requests.post(url, headers=headers, files=files, data=data)

    if resp.status_code in (200, 201):
        model_data = resp.json()
        model_id = model_data["_id"]
        print(f"[OK] Model uploaded successfully. ID: {model_id}")
        print(f"[INFO] Feature schema from response:")
        for fs in model_data.get("feature_schema", []):
            print(f"  {fs['name']:15} | {fs['type']:12} | options: {fs.get('options', [])}")
        return model_id
    else:
        print(f"[FAIL] Upload failed: {resp.status_code} {resp.text}")
        return None

def get_model(token, model_id):
    """Fetch model details to verify schema."""
    url = f"{BASE_URL}/api/v1/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"[FAIL] Get model failed: {resp.status_code} {resp.text}")
        return None

def list_models(token):
    """List all models for current user."""
    url = f"{BASE_URL}/api/v1/models/"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"[FAIL] List models failed: {resp.status_code} {resp.text}")
        return []

def main():
    print("=" * 80)
    print("UPLOAD TEST - HEART DISEASE MODEL")
    print("=" * 80)

    # Step 1: Register
    print("\n1. Registering test user...")
    if not register_user():
        return

    # Step 2: Login
    print("\n2. Logging in...")
    token = login()
    if not token:
        return

    # Step 3: Upload model
    print("\n3. Uploading model...")
    model_id = upload_model(token, "heart_disease_pipeline.pkl")
    if not model_id:
        return

    # Step 4: Fetch the model to verify schema
    print("\n4. Verifying uploaded model schema...")
    model = get_model(token, model_id)
    if not model:
        return

    schema = model.get("feature_schema", [])
    print(f"\nFeature schema stored in DB ({len(schema)} features):")
    print("-" * 80)
    for fs in schema:
        print(f"{fs['name']:15} | {fs['type']:12} | options: {fs.get('options', [])}")

    # Step 5: Check for correctness
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    cat_features = [fs for fs in schema if fs['type'] == 'categorical']
    num_features = [fs for fs in schema if fs['type'] == 'numeric']

    expected_cats = {'sex', 'cp', 'restecg', 'exang', 'slope'}
    expected_nums = {'age', 'trestbps', 'chol', 'fbs', 'thalach', 'oldpeak'}

    got_cats = {fs['name'] for fs in cat_features}
    got_nums = {fs['name'] for fs in num_features}

    if got_cats == expected_cats:
        print("[PASS] Categorical features correct")
    else:
        print("[FAIL] Categorical features mismatch")
        print(f"  Expected: {expected_cats}")
        print(f"  Got: {got_cats}")
        print(f"  Missing: {expected_cats - got_cats}")
        print(f"  Extra: {got_cats - expected_cats}")

    if got_nums == expected_nums:
        print("[PASS] Numeric features correct")
    else:
        print("[FAIL] Numeric features mismatch")
        print(f"  Expected: {expected_nums}")
        print(f"  Got: {got_nums}")

    # Check options for each categorical
    missing_options = []
    for fs in cat_features:
        if not fs.get('options'):
            missing_options.append(fs['name'])
        else:
            # Verify options are strings and non-empty
            if len(fs['options']) == 0:
                missing_options.append(fs['name'])
    if missing_options:
        print(f"[FAIL] Categorical features missing options: {missing_options}")
    else:
        print("[PASS] All categorical features have options")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
