import requests
import time
import sys

BASE_URL = "http://localhost:8000/api/v1"
# We need a valid JWT token. Replace this with a real token or bypass auth for testing.
# For this script to work fully, you must provide a valid access token and model ID.
TOKEN = "YOUR_ACCESS_TOKEN_HERE" 
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Set a valid model_id and prediction_id that exist in your DB.
MODEL_ID = "YOUR_MODEL_ID"
PREDICTION_ID = "YOUR_PREDICTION_ID"

def check_task(method, task_id):
    print(f"Waiting for {method} task {task_id} to complete...")
    for _ in range(15):
        resp = requests.get(f"{BASE_URL}/explanations/{method}/{task_id}", headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "complete":
                print(f"[SUCCESS] {method} explanation completed!")
                print("Result Snapshot:", str(data.get("explanation"))[:200] + "...")
                return True
            else:
                print(f"  Status: {data.get('status')} - retrying in 2s...")
        else:
            print(f"Error checking {method} status: {resp.status_code} {resp.text}")
            return False
        time.sleep(2)
    print(f"[FAILED] {method} task timed out.")
    return False

def test_endpoints():
    if TOKEN == "YOUR_ACCESS_TOKEN_HERE" or MODEL_ID == "YOUR_MODEL_ID":
        print("Please edit this script to insert a valid TOKEN and MODEL_ID before running.")
        sys.exit(1)

    print("--- Testing New XAI Framework Endpoints ---")
    
    frameworks = ["interpretml", "alibi", "aix360"]
    
    for fw in frameworks:
        print(f"\nTriggering {fw} local explanation...")
        # POST /api/v1/explanations/{framework}/{model_id}?prediction_id={prediction_id}
        url = f"{BASE_URL}/explanations/{fw}/{MODEL_ID}?prediction_id={PREDICTION_ID}"
        resp = requests.post(url, headers=HEADERS)
        
        if resp.status_code == 200:
            data = resp.json()
            task_id = data.get("task_id")
            print(f"[{fw}] Launched successfully. Task ID: {task_id}")
            check_task(fw, task_id)
        else:
            print(f"[ERROR] Failed to start {fw}. HTTP {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    test_endpoints()
