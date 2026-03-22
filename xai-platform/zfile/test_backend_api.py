#!/usr/bin/env python3
"""
Test SHAP and LIME explanations through the backend API.
This script verifies that the pipeline model works correctly with the XAI platform.
"""
import requests
import json
import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# ============================================================================
# CONFIGURATION - Update these values
# ============================================================================
BASE_URL = "http://localhost:8000"  # Backend server URL
API_PREFIX = "/api/v1"
AUTH_EMAIL = "test@example.com"      # Your login email
AUTH_PASSWORD = "testpassword"       # Your login password
# ============================================================================

# Global session for reuse
session = requests.Session()


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def print_success(text: str):
    print(f"[OK] {text}")


def print_error(text: str):
    print(f"[FAIL] {text}")


def print_info(text: str):
    print(f"[INFO] {text}")


def login() -> Optional[str]:
    """
    Authenticate and get access token.
    Returns: access_token string or None if failed
    """
    print_info("Logging in...")
    try:
        response = session.post(
            f"{BASE_URL}{API_PREFIX}/auth/login",
            json={"email": AUTH_EMAIL, "password": AUTH_PASSWORD},
            timeout=10
        )
        if response.status_code == 200:
            token = response.json().get("access_token")
            print_success("Logged in successfully")
            return token
        else:
            print_error(f"Login failed: {response.status_code}")
            print_error(f"Response: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to {BASE_URL}. Is the backend running?")
        return None
    except Exception as e:
        print_error(f"Login error: {e}")
        return None


def get_headers(token: str) -> Dict[str, str]:
    """Get authorization headers"""
    return {"Authorization": f"Bearer {token}"}


def list_models(token: str) -> Optional[list]:
    """List all models for the current user"""
    try:
        response = session.get(
            f"{BASE_URL}{API_PREFIX}/models",
            headers=get_headers(token),
            timeout=10
        )
        if response.status_code == 200:
            models = response.json()
            print_success(f"Found {len(models)} model(s)")
            return models
        else:
            print_error(f"Failed to list models: {response.status_code}")
            print_error(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error listing models: {e}")
        return None


def ensure_model_uploaded(token: str, model_path: str) -> Optional[str]:
    """
    Ensure a model is uploaded to the platform.
    Returns model_id if successful, None otherwise.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        print_error(f"Model file not found: {model_path}")
        print_info("Please train a model first by running: python train_pipeline.py")
        return None

    print_success(f"Found model file: {model_file}")

    # Check if already uploaded via model list
    models = list_models(token)
    if models:
        # Check if any model has the same filename
        for model in models:
            if model.get("file_path", "").endswith(model_file.name):
                model_id = str(model.get("_id"))
                print_success(f"Model already uploaded with ID: {model_id}")
                return model_id

    print_info("Model not found in your account. Please upload it manually through:")
    print_info("1. Frontend UI: Go to Models section and upload")
    print_info("2. Or use an admin tool/temporary upload endpoint")
    print_info(f"File to upload: {model_file.absolute()}")
    return None


def request_local_explanation(
    token: str,
    model_id: str,
    input_data: Dict[str, Any],
    method: str = "shap"  # "shap" or "lime"
) -> Optional[str]:
    """
    Request a local explanation (SHAP or LIME).
    Returns task_id if successful, None otherwise.
    """
    print_info(f"Requesting {method.upper()} local explanation...")

    try:
        if method.lower() == "shap":
            endpoint = f"{BASE_URL}{API_PREFIX}/explanations/local/{model_id}"
            data = {"input_data": json.dumps(input_data)}
        elif method.lower() == "lime":
            endpoint = f"{BASE_URL}{API_PREFIX}/explanations/lime/{model_id}"
            data = {
                "input_data": json.dumps(input_data),
                "num_features": 10
            }
        else:
            print_error(f"Unknown method: {method}")
            return None

        response = session.post(
            endpoint,
            headers=get_headers(token),
            data=data,
            timeout=30
        )

        if response.status_code in [200, 201]:
            result = response.json()
            task_id = result.get("task_id")
            prediction_id = result.get("prediction_id")
            print_success(f"{method.upper()} request submitted")
            print_info(f"Task ID: {task_id}")
            print_info(f"Prediction ID: {prediction_id}")
            return task_id
        else:
            print_error(f"Request failed: {response.status_code}")
            print_error(f"Response: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print_error(f"Request timeout after 30 seconds")
        return None
    except Exception as e:
        print_error(f"Error requesting explanation: {e}")
        return None


def poll_task_completion(
    token: str,
    task_id: str,
    method: str,  # "shap" or "lime"
    timeout_seconds: int = 120,
    poll_interval: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Poll for task completion and return explanation if successful.
    """
    print_info(f"Polling {method.upper()} task {task_id}...")

    # SHAP and LIME use different endpoints for getting results
    if method.lower() == "shap":
        endpoint_template = f"{BASE_URL}{API_PREFIX}/explanations/sharp/{{task_id}}"
    elif method.lower() == "lime":
        endpoint_template = f"{BASE_URL}{API_PREFIX}/explanations/lime/{{task_id}}"
    else:
        print_error(f"Unknown method: {method}")
        return None

    start_time = time.time()
    attempts = 0

    while time.time() - start_time < timeout_seconds:
        attempts += 1
        try:
            response = session.get(
                endpoint_template.format(task_id=task_id),
                headers=get_headers(token),
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                status = result.get("status")

                if status == "complete":
                    print_success(f"{method.upper()} task completed after {attempts} attempts")
                    explanation = result.get("explanation", {})
                    return explanation
                elif status == "failed":
                    print_error(f"{method.upper()} task failed")
                    print_error(f"Details: {result}")
                    return None
                else:
                    # Still processing
                    task_state = result.get("task_state", "unknown")
                    info_status = result.get("info", {}).get("status", "")
                    print_info(f"  Attempt {attempts}: {task_state} - {info_status}")
                    time.sleep(poll_interval)
            else:
                print_error(f"Error checking task: {response.status_code}")
                print_error(f"Response: {response.text}")
                time.sleep(poll_interval)

        except requests.exceptions.RequestException as e:
            print_error(f"Request error while polling: {e}")
            time.sleep(poll_interval)
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            time.sleep(poll_interval)

    print_error(f"{method.upper()} task did not complete within {timeout_seconds} seconds")
    return None


def print_explanation_summary(explanation: Dict[str, Any], method: str):
    """Print a summary of the explanation"""
    print_header(f"{method.upper()} EXPLANATION SUMMARY")

    print(f"Explanation ID: {explanation.get('_id')}")
    print(f"Model ID: {explanation.get('model_id')}")
    print(f"Method: {explanation.get('method')}")
    print(f"Type: {explanation.get('explanation_type')}")
    print(f"Created: {explanation.get('created_at')}")

    feature_names = explanation.get("feature_names", [])
    print(f"\nFeature names ({len(feature_names)}):")
    for i, name in enumerate(feature_names[:10]):  # Show first 10
        print(f"  {i+1}. {name}")
    if len(feature_names) > 10:
        print(f"  ... and {len(feature_names) - 10} more")

    if method.lower() == "shap":
        shap_values = explanation.get("shap_values", [])
        expected_value = explanation.get("expected_value")
        print(f"\nSHAP values shape: {len(shap_values)} (list length)")
        if isinstance(shap_values, list) and len(shap_values) > 0:
            print(f"First feature SHAP: {shap_values[0]}")
        print(f"Expected value: {expected_value}")
    elif method.lower() == "lime":
        lime_weights = explanation.get("lime_weights", [])
        lime_intercept = explanation.get("lime_intercept")
        lime_local_pred = explanation.get("lime_local_pred")
        print(f"\nLIME intercept: {lime_intercept}")
        print(f"Local prediction: {lime_local_pred}")
        print(f"Number of feature weights: {len(lime_weights)}")
        if lime_weights:
            print("\nTop 5 contributing features:")
            for i, item in enumerate(lime_weights[:5]):
                print(f"  {i+1}. {item.get('feature')}: {item.get('weight'):.4f} (value: {item.get('value')})")


def main():
    print_header("BACKEND API SHAP/LIME VERIFICATION TEST")

    # Sample raw input data (adjust to match your model's expected features)
    sample_input = {
        "Gender": "Male",
        "Married": "No",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Urban",
        "ApplicantIncome": 6000,
        "CoapplicantIncome": 0,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1
    }

    print_info("Sample input data:")
    for key, value in sample_input.items():
        print(f"  {key}: {value}")

    # Step 1: Login
    token = login()
    if not token:
        print_error("Cannot proceed without authentication")
        sys.exit(1)

    # Step 2: Ensure model is uploaded
    print("\n" + "-" * 80)
    model_id = ensure_model_uploaded(token, "loan_model_pipeline.pkl")
    if not model_id:
        print_info("\nTo get a model_id:")
        print_info("1. Check your uploaded models in the frontend")
        print_info("2. Or run: curl -H 'Authorization: Bearer TOKEN' {BASE_URL}/api/v1/models")
        sys.exit(1)

    print_success(f"Using model ID: {model_id}")

    # Step 3: Request local SHAP explanation
    print("\n" + "-" * 80)
    shap_task = request_local_explanation(token, model_id, sample_input, method="shap")

    # Step 4: Request local LIME explanation
    print("\n" + "-" * 80)
    lime_task = request_local_explanation(token, model_id, sample_input, method="lime")

    # Step 5: Poll for results
    if shap_task:
        print("\n" + "-" * 80)
        shap_explanation = poll_task_completion(token, shap_task, "shap", timeout_seconds=120)
        if shap_explanation:
            print_explanation_summary(shap_explanation, "shap")
        else:
            print_error("SHAP explanation failed or timed out")

    if lime_task:
        print("\n" + "-" * 80)
        lime_explanation = poll_task_completion(token, lime_task, "lime", timeout_seconds=120)
        if lime_explanation:
            print_explanation_summary(lime_explanation, "lime")
        else:
            print_error("LIME explanation failed or timed out")

    # Final summary
    print_header("TEST SUMMARY")
    if shap_task and shap_explanation:
        print_success("SHAP: Working OK")
    else:
        print_error("SHAP: Failed")

    if lime_task and lime_explanation:
        print_success("LIME: Working OK")
    else:
        print_error("LIME: Failed")

    print("\n" + "-" * 80)
    print_info("If both tests passed, your backend fixes are working correctly!")
    print_info("If tests failed, check:")
    print_info("  1. Backend server is running")
    print_info("  2. Celery workers are running")
    print_info("  3. Model was uploaded successfully")
    print_info("  4. Background data is available (see tasks.py for fallback logic)")
    print("=" * 80)


if __name__ == "__main__":
    main()
