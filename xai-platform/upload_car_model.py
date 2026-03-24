#!/usr/bin/env python3
"""
Upload car_price_pipeline.pkl to the XAI platform.
"""

import requests
import os

BASE_URL = "http://localhost:8000"

def login():
    """Login with existing user."""
    url = f"{BASE_URL}/api/v1/auth/login"
    data = {
        "username": "test@example.com",
        "password": "testpassword123"
    }
    resp = requests.post(url, data=data)
    if resp.status_code == 200:
        token = resp.json()["access_token"]
        print("[OK] Logged in")
        return token
    else:
        print(f"[FAIL] Login failed: {resp.status_code} {resp.text}")
        return None

def upload_model(token, model_path, name, description):
    """Upload the model file."""
    url = f"{BASE_URL}/api/v1/models/upload"
    headers = {"Authorization": f"Bearer {token}"}

    if not os.path.exists(model_path):
        print(f"[FAIL] Model file not found: {model_path}")
        return None

    with open(model_path, 'rb') as f:
        files = {'file': (os.path.basename(model_path), f, 'application/octet-stream')}
        data = {
            'name': name,
            'description': description,
            'framework': 'sklearn',
            'task_type': 'regression',  # Car price prediction is regression
            'feature_schema': '[]'  # Auto-generate
        }
        print(f"[INFO] Uploading {name}...")
        resp = requests.post(url, headers=headers, files=files, data=data)

    if resp.status_code in (200, 201):
        model_data = resp.json()
        model_id = model_data["_id"]
        print(f"[OK] Model uploaded. ID: {model_id}")
        print(f"[INFO] Feature schema:")
        for fs in model_data.get("feature_schema", []):
            print(f"  {fs['name']:15} | {fs['type']:12} | options: {len(fs.get('options', []))}")
        return model_id
    else:
        print(f"[FAIL] Upload failed: {resp.status_code} {resp.text}")
        return None

def main():
    print("=" * 80)
    print("UPLOAD CAR PRICE PREDICTION MODEL")
    print("=" * 80)

    # Login
    print("\n1. Logging in...")
    token = login()
    if not token:
        return

    # Upload car price model
    print("\n2. Uploading model...")
    model_id = upload_model(
        token,
        "car_price_pipeline.pkl",
        "Car Price Predictor",
        "Random Forest model to predict used car prices based on brand, year, mileage, and other features"
    )

    if model_id:
        print("\n" + "=" * 80)
        print("[SUCCESS] Car price model uploaded!")
        print("=" * 80)
        print("\nYou can now use this model for predictions via the frontend or API:")
        print(f"  POST /api/v1/predict with model_id={model_id}")
        print("\nExpected input fields:")
        print("  Categorical (as strings):")
        print("    - name: Car model name (e.g., 'Maruti Alto')")
        print("    - fuel: 'Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'")
        print("    - seller_type: 'Individual', 'Dealer', 'Trustmark Dealer'")
        print("    - transmission: 'Manual', 'Automatic'")
        print("    - owner: 'First Owner', 'Second Owner', etc.")
        print("  Numeric (as numbers):")
        print("    - year: manufacturing year (e.g., 2015)")
        print("    - km_driven: kilometers driven (e.g., 50000)")
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
