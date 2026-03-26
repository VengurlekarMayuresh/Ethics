"""
Upload corrected loan model to the XAI platform via API.
Requires a logged-in user token.
"""

import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"
LOGIN_EMAIL = "neha@example.com"  # CHANGE THIS
LOGIN_PASSWORD = "password"  # CHANGE THIS
MODEL_FILE = "loan_prediction_pipeline.pkl"

print("=" * 80)
print("UPLOAD LOAN MODEL TO XAI PLATFORM")
print("=" * 80)

# Step 1: Login
print("\n1. Logging in...")
resp = requests.post(
    f"{BASE_URL}{API_PREFIX}/auth/login",
    json={"email": LOGIN_EMAIL, "password": LOGIN_PASSWORD}
)

if resp.status_code != 200:
    print(f"[FAIL] Login failed: {resp.status_code} {resp.text}")
    print("Please check credentials in script.")
    exit(1)

token = resp.json()["access_token"]
print(f"   [OK] Logged in, token: {token[:20]}...")

# Step 2: Upload model
print(f"\n2. Uploading model file: {MODEL_FILE}")
if not Path(MODEL_FILE).exists():
    print(f"[FAIL] File not found: {MODEL_FILE}")
    exit(1)

files = {
    "file": (MODEL_FILE, open(MODEL_FILE, "rb"), "application/octet-stream")
}
data = {
    "name": "Loan Prediction Fixed",
    "description": "Loan approval prediction model with correct raw feature schema",
    "framework": "sklearn",
    # task_type will be auto-detected; no need to provide
}
headers = {
    "Authorization": f"Bearer {token}"
}

resp = requests.post(
    f"{BASE_URL}{API_PREFIX}/models/upload",
    headers=headers,
    files=files,
    data=data
)

if resp.status_code not in (200, 201):
    print(f"[FAIL] Upload failed: {resp.status_code}")
    print(resp.text)
    exit(1)

model_info = resp.json()
print(f"   [OK] Model uploaded successfully!")
print(f"   Model ID: {model_info['id']}")
print(f"   Name: {model_info['name']}")
print(f"   Framework: {model_info['framework']}")
print(f"   Task type: {model_info['task_type']}")
print(f"   Feature count: {len(model_info['feature_schema'])}")
print("\n   Feature schema:")
for fs in model_info['feature_schema']:
    opts = f", options={fs['options']}" if fs.get('options') else ""
    print(f"     - {fs['name']} ({fs['type']}){opts}")

print("\n" + "=" * 80)
print("UPLOAD SUCCESSFUL")
print("=" * 80)
print("""
Next steps:
1. Go to http://localhost:3000/models to see the model
2. Click on the model to view its details
3. The prediction form should show only raw features (no derived ones)
4. You can now test predictions and explanations
""")
