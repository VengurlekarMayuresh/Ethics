#!/usr/bin/env python3
"""
Test full prediction + SHAP explanation via API for car price model.
"""

import requests
import json
import os
import time

BASE_URL = "http://localhost:8000"

# Use the uploaded car price model ID: 69c2d802c6ddccf3e625128d
MODEL_ID = "69c2d802c6ddccf3e625128d"

def login():
    url = f"{BASE_URL}/api/v1/auth/login"
    data = {"username": "test@example.com", "password": "testpassword123"}
    resp = requests.post(url, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def predict(token):
    url = f"{BASE_URL}/api/v1/predict/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {token}"}
    # The endpoint expects form data with input_data as a JSON string
    input_json = {
        "name": "Maruti Alto LX",
        "year": 2015,
        "km_driven": 50000,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner"
    }
    form_data = {
        "input_data": json.dumps(input_json)
    }
    resp = requests.post(url, headers=headers, data=form_data)
    if not resp.ok:
        print(f"[FAIL] Request failed: {resp.status_code}")
        print(f"Response: {resp.text}")
        resp.raise_for_status()
    return resp.json()

def get_explanation(token, explanation_id):
    url = f"{BASE_URL}/api/v1/explain/{explanation_id}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"Failed to get explanation: {resp.status_code} {resp.text}")
        return None

def main():
    print("=" * 80)
    print("FULL API TEST: PREDICTION + SHAP EXPLANATION")
    print("=" * 80)

    print("\n1. Logging in...")
    token = login()
    print("[OK] Token obtained")

    print("\n2. Making prediction...")
    result = predict(token)
    prediction_id = result.get("prediction_id")
    print(f"[OK] Prediction made. ID: {prediction_id}")
    print(f"    Prediction: {result.get('prediction')}")

    print("\n3. Requesting SHAP explanation...")
    task_id = request_shap(token, MODEL_ID, prediction_id=prediction_id)
    if not task_id:
        print("[FAIL] Failed to get SHAP task_id")
        return
    print(f"[OK] SHAP task started: {task_id}")

    print("\n4. Polling for explanation result...")
    explanation = poll_explanation(token, task_id, timeout=30)
    if not explanation:
        print("[FAIL] Explanation not ready or failed")
        return

    print(f"[OK] Explanation completed")
    status = explanation.get('status')
    print(f"    Status: {status}")

    if status != 'complete' or 'explanation' not in explanation:
        print(f"[FAIL] Explanation incomplete. Response: {explanation}")
        return

    expl = explanation['explanation']

    # Examine SHAP values
    shap_values = expl.get('shap_values')
    if shap_values:
        print(f"\nSHAP values ({len(shap_values)} features):")
        print("-" * 80)
        for item in shap_values[:10]:
            print(f"  {item['feature']:20} | contribution: {item['contribution']:>12.2f} | value: {item.get('value')}")
        if len(shap_values) > 10:
            print(f"  ... and {len(shap_values) - 10} more features")
    else:
        print("[WARN] No shap_values in explanation")

    # Check if we have only original 7 features
    expected_features = {'name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner'}
    returned_features = {item['feature'] for item in shap_values} if shap_values else set()
    print(f"\nFeature count: {len(returned_features)}")
    print(f"Expected original features: {expected_features}")
    extra = returned_features - expected_features
    missing = expected_features - returned_features
    if extra:
        print(f"[FAIL] Extra features: {extra}")
    if missing:
        print(f"[FAIL] Missing features: {missing}")
    if not extra and not missing:
        print("[PASS] All features are original categorical/numeric fields!")

    print("\n" + "=" * 80)

def request_shap(token, model_id, prediction_id=None):
    url = f"{BASE_URL}/api/v1/explain/local/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    # We can pass prediction_id as query parameter or form data
    params = {"prediction_id": prediction_id} if prediction_id else {}
    resp = requests.post(url, headers=headers, params=params)
    if not resp.ok:
        print(f"[FAIL] SHAP request failed: {resp.status_code} {resp.text}")
        return None
    data = resp.json()
    return data.get("task_id")

def poll_explanation(token, task_id, timeout=30):
    url = f"{BASE_URL}/api/v1/explain/sharp/{task_id}"
    headers = {"Authorization": f"Bearer {token}"}
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(url, headers=headers)
        if not resp.ok:
            print(f"[FAIL] Poll failed: {resp.status_code} {resp.text}")
            return None
        data = resp.json()
        if data.get("status") == "complete":
            return data
        elif data.get("status") == "failed":
            print(f"[FAIL] Task failed: {data}")
            return None
        time.sleep(2)  # Wait before polling again
    print("[WARN] Timeout waiting for explanation")
    return None

if __name__ == "__main__":
    main()
