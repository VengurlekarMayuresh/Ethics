"""
XAI Platform Python Client.

A convenient wrapper around the XAI Platform REST API.
"""

import json
import httpx
from typing import Optional, Dict, Any, List, BinaryIO
from datetime import datetime

from .models import (
    ModelCreate, ModelResponse,
    PredictionRequest, PredictionResponse,
    SHAPRequest, SHAPResponse,
    LIMERequest, LimeResponse,
    BiasAnalysisRequest, BiasReport,
    APIKeyCreate, APIKeyResponse,
)
from .exceptions import (
    XAIClientError,
    XAIAuthError,
    XAIRateLimitError,
    XAINotFoundError,
    XAIValidationError,
    XAIExplanationError,
    XAITaskTimeoutError,
)


class XAIClient:
    """Client for the XAI Platform API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """
        Initialize the XAI Platform client.

        Args:
            base_url: API base URL (e.g., http://localhost:8000)
            api_key: API key for authentication (alternate to JWT)
            jwt_token: JWT token for authentication (alternate to API key)
            timeout: Request timeout in seconds (default 5 minutes for async tasks)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    def _get_headers(self) -> Dict[str, str]:
        """Build authentication headers."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        return headers

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure an async HTTP client exists."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self._get_headers()
            )
        return self._client

    async def aclose(self):
        """Close the async HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # Synchronous context manager
    def __enter__(self):
        import asyncio
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self.__aenter__())
        return self

    def __exit__(self, *args):
        import asyncio
        self._loop.run_until_complete(self.__aexit__())
        self._loop.close()

    # Authentication
    @classmethod
    def from_jwt(cls, jwt_token: str, base_url: str = "http://localhost:8000") -> "XAIClient":
        """Create client using JWT token."""
        return cls(base_url=base_url, jwt_token=jwt_token)

    @classmethod
    def from_api_key(cls, api_key: str, base_url: str = "http://localhost:8000") -> "XAIClient":
        """Create client using API key."""
        return cls(base_url=base_url, api_key=api_key)

    # Model operations
    async def list_models(self) -> List[ModelResponse]:
        """List all models for the current user."""
        client = self._ensure_client()
        response = await client.get(f"{self.base_url}/api/v1/models")
        response.raise_for_status()
        data = response.json()
        return [ModelResponse(**model) for model in data]

    async def upload_model(
        self,
        name: str,
        model_file: BinaryIO,
        feature_schema: Dict[str, Any],
        task_type: str,
        framework: str,
        description: Optional[str] = None,
        target_schema: Optional[Dict[str, Any]] = None
    ) -> ModelResponse:
        """
        Upload a machine learning model.

        Args:
            name: Model name
            model_file: File-like object containing the model (pkl, joblib, onnx, h5)
            feature_schema: Dictionary describing input features
            task_type: ML task type (classification, regression, etc.)
            framework: ML framework (sklearn, xgboost, onnx, keras, lightgbm)
            description: Optional model description
            target_schema: Optional target/output schema
        """
        client = self._ensure_client()
        files = {"file": (model_file.name, model_file, "application/octet-stream")}
        data = {
            "name": name,
            "description": description or "",
            "task_type": task_type,
            "framework": framework,
            "feature_schema": json.dumps(feature_schema),
        }
        if target_schema:
            data["target_schema"] = json.dumps(target_schema)

        response = await client.post(f"{self.base_url}/api/v1/models/upload", files=files, data=data)
        response.raise_for_status()
        return ModelResponse(**response.json())

    async def get_model(self, model_id: str) -> ModelResponse:
        """Get model details by ID."""
        client = self._ensure_client()
        response = await client.get(f"{self.base_url}/api/v1/models/{model_id}")
        response.raise_for_status()
        return ModelResponse(**response.json())

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        client = self._ensure_client()
        response = await client.delete(f"{self.base_url}/api/v1/models/{model_id}")
        response.raise_for_status()
        return True

    # Prediction operations
    async def predict(self, model_id: str, input_data: Dict[str, Any]) -> PredictionResponse:
        """
        Make a single prediction.

        Args:
            model_id: ID of the model to use
            input_data: Dictionary of feature values
        """
        client = self._ensure_client()
        payload = {"input_data": input_data}
        response = await client.post(f"{self.base_url}/api/v1/predict/{model_id}", json=payload)
        response.raise_for_status()
        return PredictionResponse(**response.json())

    async def batch_predict(self, model_id: str, csv_file: BinaryIO) -> List[PredictionResponse]:
        """
        Make batch predictions from a CSV file.

        Args:
            model_id: ID of the model to use
            csv_file: CSV file with feature columns
        """
        client = self._ensure_client()
        files = {"file": (csv_file.name, csv_file, "text/csv")}
        response = await client.post(f"{self.base_url}/api/v1/predict/{model_id}/batch", files=files)
        response.raise_for_status()
        data = response.json()
        return [PredictionResponse(**pred) for pred in data]

    async def get_prediction(self, prediction_id: str) -> PredictionResponse:
        """Get prediction result by ID."""
        client = self._ensure_client()
        response = await client.get(f"{self.base_url}/api/v1/predict/{prediction_id}")
        response.raise_for_status()
        return PredictionResponse(**response.json())

    async def get_prediction_history(self, limit: int = 100, skip: int = 0) -> List[PredictionResponse]:
        """Get prediction history."""
        client = self._ensure_client()
        response = await client.get(f"{self.base_url}/api/v1/predict/history?limit={limit}&skip={skip}")
        response.raise_for_status()
        data = response.json()
        return [PredictionResponse(**pred) for pred in data]

    # Explanation operations (SHAP)
    async def request_shap_explanation(
        self,
        model_id: str,
        prediction_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None
    ) -> SHAPResponse:
        """
        Request a SHAP explanation for a prediction.

        This is an async operation. Use get_explanation_status() to poll for completion.

        Args:
            model_id: ID of the model
            prediction_id: Optional existing prediction ID to explain
            input_data: Optional new input to predict and explain
        """
        client = self._ensure_client()
        payload = {}
        if prediction_id:
            payload["prediction_id"] = prediction_id
        if input_data:
            payload["input_data"] = input_data

        response = await client.post(f"{self.base_url}/api/v1/explain/local/{model_id}", json=payload)
        response.raise_for_status()
        return SHAPResponse(**response.json())

    async def get_explanation_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of an async explanation task.

        Returns:
            Dictionary with status ("pending", "complete") and explanation if available.
        """
        client = self._ensure_client()
        response = await client.get(f"{self.base_url}/api/v1/explain/local/{task_id}")
        response.raise_for_status()
        return response.json()

    async def get_global_shap(self, model_id: str) -> Dict[str, Any]:
        """Get the latest global SHAP explanation for a model."""
        client = self._ensure_client()
        response = await client.get(f"{self.base_url}/api/v1/explain/global/{model_id}/latest")
        response.raise_for_status()
        return response.json()

    async def request_global_shap(self, model_id: str, background_data: BinaryIO) -> Dict[str, Any]:
        """
        Request global SHAP explanation.

        Args:
            model_id: ID of the model
            background_data: CSV file with background dataset for SHAP reference
        """
        client = self._ensure_client()
        files = {"background_data": (background_data.name, background_data, "text/csv")}
        response = await client.post(f"{self.base_url}/api/v1/explain/global/{model_id}", files=files)
        response.raise_for_status()
        return response.json()

    async def get_shap_dependence(
        self,
        model_id: str,
        feature: str,
        background_data: BinaryIO
    ) -> Dict[str, Any]:
        """
        Get SHAP dependence data for a feature.

        Args:
            model_id: ID of the model
            feature: Feature name to analyze
            background_data: CSV file with background dataset
        """
        client = self._ensure_client()
        files = {"background_data": (background_data.name, background_data, "text/csv")}
        params = {"feature": feature}
        response = await client.post(f"{self.base_url}/api/v1/explain/dependence/{model_id}", files=files, params=params)
        response.raise_for_status()
        return response.json()

    # LIME operations
    async def request_lime_explanation(
        self,
        model_id: str,
        prediction_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        num_features: int = 10
    ) -> LimeResponse:
        """Request a LIME explanation."""
        client = self._ensure_client()
        payload = {"num_features": num_features}
        if prediction_id:
            payload["prediction_id"] = prediction_id
        if input_data:
            payload["input_data"] = input_data

        response = await client.post(f"{self.base_url}/api/v1/explain/lime/{model_id}", json=payload)
        response.raise_for_status()
        return LimeResponse(**response.json())

    async def get_lime_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a LIME explanation task."""
        client = self._ensure_client()
        response = await client.get(f"{self.base_url}/api/v1/explain/lime/{task_id}")
        response.raise_for_status()
        return response.json()

    # Bias analysis
    async def analyze_bias(
        self,
        model_id: str,
        protected_attribute: str,
        sensitive_attribute: str,
        evaluation_data: BinaryIO
    ) -> Dict[str, Any]:
        """
        Run bias analysis on a model.

        Args:
            model_id: ID of the model
            protected_attribute: Column name for protected attribute
            sensitive_attribute: Column name for sensitive attribute
            evaluation_data: CSV file with evaluation dataset
        """
        client = self._ensure_client()
        files = {"file": (evaluation_data.name, evaluation_data, "text/csv")}
        params = {
            "protected_attribute": protected_attribute,
            "sensitive_attribute": sensitive_attribute
        }
        response = await client.post(f"{self.base_url}/api/v1/bias/analyze?model_id={model_id}", files=files, params=params)
        response.raise_for_status()
        return response.json()

    async def get_bias_reports(self, model_id: str, limit: int = 50) -> List[BiasReport]:
        """Get bias reports for a model."""
        client = self._ensure_client()
        response = await client.get(f"{self.base_url}/api/v1/bias/reports/{model_id}?limit={limit}")
        response.raise_for_status()
        return [BiasReport(**report) for report in response.json()]

    async def get_bias_metrics(self, model_id: str) -> Dict[str, Any]:
        """Get aggregated bias metrics for a model."""
        client = self._ensure_client()
        response = await client.get(f"{self.base_url}/api/v1/bias/metrics/{model_id}")
        response.raise_for_status()
        return response.json()

    async def generate_bias_report_pdf(self, report_id: str) -> bytes:
        """
        Generate a PDF compliance report for a bias analysis.

        Returns:
            PDF file bytes
        """
        client = self._ensure_client()
        response = await client.get(f"{self.base_url}/api/v1/bias/generate-report/{report_id}")
        response.raise_for_status()
        return response.content

    # Model comparison
    async def compare_models(
        self,
        model_ids: List[str],
        evaluation_data: BinaryIO,
        protected_attribute: str,
        sensitive_attribute: str
    ) -> Dict[str, Any]:
        """
        Compare bias across multiple models.

        Args:
            model_ids: List of model IDs to compare
            evaluation_data: CSV file with evaluation dataset
            protected_attribute: Column name for protected attribute
            sensitive_attribute: Column name for sensitive attribute
        """
        client = self._ensure_client()
        files = {"file": (evaluation_data.name, evaluation_data, "text/csv")}
        params = {
            "model_ids": model_ids,
            "protected_attribute": protected_attribute,
            "sensitive_attribute": sensitive_attribute
        }
        # Use json for model_ids since it's easier
        response = await client.post(
            f"{self.base_url}/api/v1/compare/?protected_attribute={protected_attribute}&sensitive_attribute={sensitive_attribute}",
            files=files,
            json={"model_ids": model_ids}
        )
        response.raise_for_status()
        return response.json()

    # API Key management
    async def list_api_keys(self) -> List[APIKeyResponse]:
        """List API keys for the current user."""
        client = self._ensure_client()
        response = await client.get(f"{self.base_url}/api/v1/api-keys/")
        response.raise_for_status()
        return [APIKeyResponse(**key) for key in response.json()]

    async def create_api_key(self, name: str, description: Optional[str] = None) -> APIKeyResponse:
        """Create a new API key."""
        client = self._ensure_client()
        payload = {"name": name, "description": description}
        response = await client.post(f"{self.base_url}/api/v1/api-keys/", json=payload)
        response.raise_for_status()
        return APIKeyResponse(**response.json())

    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke (delete) an API key."""
        client = self._ensure_client()
        response = await client.delete(f"{self.base_url}/api/v1/api-keys/{key_id}")
        response.raise_for_status()
        return True

    # Audit logs
    async def get_audit_logs(
        self,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get audit logs for the current user.

        Args:
            action: Optional filter by action type
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of results
        """
        client = self._ensure_client()
        params = {"limit": limit}
        if action:
            params["action"] = action
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        response = await client.get(f"{self.base_url}/api/v1/audit/my", params=params)
        response.raise_for_status()
        return response.json()

    # Explanation export
    async def export_explanation(self, explanation_id: str, format: str = "json") -> bytes:
        """
        Export an explanation in specified format.

        Args:
            explanation_id: ID of the explanation to export
            format: Export format: "json", "csv", or "pdf"

        Returns:
            File content as bytes
        """
        if format not in ["json", "csv", "pdf"]:
            raise XAIValidationError(f"Unsupported format: {format}")

        client = self._ensure_client()
        response = await client.get(f"{self.base_url}/api/v1/explain/export/{explanation_id}?format={format}")
        response.raise_for_status()
        return response.content
