# XAI Platform Python SDK

Python client library for the XAI Platform API.

Installation
-----------

```bash
pip install xai-platform
```

Or from source:

```bash
cd sdk
pip install -e .
```

Quickstart
----------

```python
import asyncio
from xai_platform import XAIClient

async def main():
    # Initialize client with JWT token
    async with XAIClient.from_jwt(
        jwt_token="your-jwt-token",
        base_url="http://localhost:8000"
    ) as client:
        # List models
        models = await client.list_models()
        print(f"Found {len(models)} models")

        # Upload a model
        with open("model.pkl", "rb") as f:
            model = await client.upload_model(
                name="My Model",
                model_file=f,
                feature_schema={"age": {"type": "float"}, "income": {"type": "float"}},
                task_type="classification",
                framework="sklearn"
            )
        print(f"Uploaded model: {model.id}")

        # Make a prediction
        prediction = await client.predict(
            model_id=model.id,
            input_data={"age": 35, "income": 50000}
        )
        print(f"Prediction: {prediction.prediction}")

        # Request SHAP explanation
        shap = await client.request_shap_explanation(
            model_id=model.id,
            input_data={"age": 35, "income": 50000}
        )
        print(f"SHAP task ID: {shap.task_id}")

        # Poll for explanation result
        import time
        while True:
            status = await client.get_explanation_status(shap.task_id)
            if status["status"] == "complete":
                print(f"SHAP values: {status['explanation']['shap_values']}")
                break
            time.sleep(1)

        # Analyze bias
        with open("evaluation.csv", "rb") as f:
            bias = await client.analyze_bias(
                model_id=model.id,
                protected_attribute="age_group",
                sensitive_attribute="gender",
                evaluation_data=f
            )
        print(f"Bias metrics: {bias['metrics']}")

        # Export explanation as PDF
        if shap.explanation_id:
            pdf_bytes = await client.export_explanation(shap.explanation_id, format="pdf")
            with open("explanation.pdf", "wb") as f:
                f.write(pdf_bytes)

asyncio.run(main())
```

Synchronous usage:

```python
from xai_platform import XAIClient

with XAIClient.from_api_key(
    api_key="your-api-key",
    base_url="http://localhost:8000"
) as client:
    models = client.list_models_sync()
    print(models)
```

API Reference
------------

### Authentication
- `XAIClient.from_jwt(jwt_token, base_url)` - Create client with JWT
- `XAIClient.from_api_key(api_key, base_url)` - Create client with API key

### Models
- `list_models()` - List all models
- `upload_model(name, model_file, feature_schema, task_type, framework, ...)` - Upload model
- `get_model(model_id)` - Get model details
- `delete_model(model_id)` - Delete model

### Predictions
- `predict(model_id, input_data)` - Single prediction
- `batch_predict(model_id, csv_file)` - Batch predictions
- `get_prediction(prediction_id)` - Get prediction result
- `get_prediction_history(limit=100, skip=0)` - Get prediction history

### Explanations (SHAP)
- `request_shap_explanation(model_id, prediction_id=None, input_data=None)` - Request SHAP
- `get_explanation_status(task_id)` - Poll for SHAP result
- `get_global_shap(model_id)` - Get global SHAP explanation
- `request_global_shap(model_id, background_data)` - Request global SHAP
- `get_shap_dependence(model_id, feature, background_data)` - SHAP dependence
- `export_explanation(explanation_id, format="json|csv|pdf")` - Export explanation

### Explanations (LIME)
- `request_lime_explanation(model_id, prediction_id=None, input_data=None, num_features=10)` - Request LIME
- `get_lime_status(task_id)` - Poll for LIME result

### Bias Analysis
- `analyze_bias(model_id, protected_attribute, sensitive_attribute, evaluation_data)` - Run bias analysis
- `get_bias_reports(model_id, limit=50)` - Get bias reports
- `get_bias_metrics(model_id)` - Get aggregated bias metrics
- `generate_bias_report_pdf(report_id)` - Generate PDF compliance report

### Model Comparison
- `compare_models(model_ids, evaluation_data, protected_attribute, sensitive_attribute)` - Compare models

### API Keys
- `list_api_keys()` - List API keys
- `create_api_key(name, description=None)` - Create API key
- `revoke_api_key(key_id)` - Revoke API key

### Audit Logs
- `get_audit_logs(action=None, start_date=None, end_date=None, limit=100)` - Get audit logs

Error Handling
-------------

The SDK raises the following exceptions:

- `XAIAuthError` - Authentication failed
- `XAIRateLimitError` - Rate limit exceeded
- `XAINotFoundError` - Resource not found
- `XAIValidationError` - Invalid input or validation error
- `XAIExplanationError` - Explanation generation failed
- `XAITaskTimeoutError` - Async task timed out
- `XAIClientError` - Base error class

```python
from xai_platform import XAIClient, XAINotFoundError

try:
    async with XAIClient.from_jwt(token, url) as client:
        model = await client.get_model("nonexistent-id")
except XAINotFoundError:
    print("Model not found!")
```

License
-------

MIT License - see LICENSE file for details.
