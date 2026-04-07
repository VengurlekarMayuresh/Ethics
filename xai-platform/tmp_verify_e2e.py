import requests
import time
import sys

BASE_URL = "http://localhost:8000/api/v1/explain"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJrZXRha2lucDE5QGdtYWlsLmNvbSIsImV4cCI6MTc3NTA5Mzk5NH0.buD_XsqaJXhF0wNDlu9zkVKQ45OIy_n4I5gB2hwynaw"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

MODEL_ID = "69cce04ecddda9c51728dccc"
PREDICTION_ID = "69cdc5ef2a324ca3f444ba89"

def check_task(method, task_id):
    print(f"Waiting for {method} task {task_id} to complete...")
    for _ in range(30):
        resp = requests.get(f"{BASE_URL}/{method}/{task_id}", headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "complete":
                print(f"[SUCCESS] {method} explanation completed!")
                
                # Check generic polling endpoint too
                print(f"Verifying generic polling endpoint for {method}...")
                poll_resp = requests.get(f"http://localhost:8000/api/v1/explain/prediction/{PREDICTION_ID}?method={method}", headers=HEADERS)
                if poll_resp.status_code == 200:
                    poll_data = poll_resp.json()
                    if poll_data.get("method") == method:
                        print(f"[SUCCESS] Generic polling for {method} returned data!")
                    else:
                        print(f"[ERROR] Generic polling returned wrong data: {poll_data.get('method')}")
                else:
                    print(f"[ERROR] Generic polling for {method} failed: {poll_resp.status_code} {poll_resp.text}")
                
                return True
            elif data.get("status") == "failed":
                print(f"[FAILED] {method} task failed: {data.get('error')}")
                return False
            else:
                # Task might be pending or in progress
                task_state = data.get('task_state') or data.get('status')
                print(f"  Status: {task_state} - retrying in 2s...")
        else:
            print(f"Error checking {method} status: {resp.status_code} {resp.text}")
            return False
        time.sleep(2)
    print(f"[FAILED] {method} task timed out.")
    return False

def test_endpoints():
    print("--- Testing New XAI Framework Endpoints (Fixed Routes) ---")
    
    frameworks = ["interpretml", "alibi", "aix360"]
    
    for fw in frameworks:
        print(f"\nTriggering {fw} local explanation...")
        url = f"{BASE_URL}/{fw}/{MODEL_ID}?prediction_id={PREDICTION_ID}"
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
