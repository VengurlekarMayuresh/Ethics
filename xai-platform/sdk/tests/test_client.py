"""
Tests for XAI Platform SDK.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from xai_platform import XAIClient, XAIClientError, XAINotFoundError


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient."""
    with patch("httpx.AsyncClient") as mock:
        client_instance = MagicMock()
        client_instance.request = AsyncMock()
        client_instance.aclose = AsyncMock()
        client_instance.is_closed = False
        mock.return_value = client_instance
        yield client_instance


def test_client_initialization():
    """Test client initialization."""
    client = XAIClient(base_url="http://test.com", api_key="test-key")
    assert client.base_url == "http://test.com"
    assert client.api_key == "test-key"
    assert client.jwt_token is None


def test_client_from_api_key():
    """Test creating client with API key."""
    client = XAIClient.from_api_key("api-key-123", "http://test.com")
    assert client.api_key == "api-key-123"


def test_client_from_jwt():
    """Test creating client with JWT."""
    client = XAIClient.from_jwt("jwt-token-456", "http://test.com")
    assert client.jwt_token == "jwt-token-456"


@pytest.mark.asyncio
async def test_list_models(mock_httpx_client):
    """Test listing models."""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "id": "1",
            "name": "Model 1",
            "description": None,
            "task_type": "classification",
            "framework": "sklearn",
            "feature_schema": {},
            "target_schema": None,
            "file_path": "/path/to/model.pkl",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": None
        }
    ]
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.get.return_value = mock_response

    client = XAIClient(base_url="http://test.com")
    models = await client.list_models()
    assert len(models) == 1
    assert models[0].name == "Model 1"


@pytest.mark.asyncio
async def test_error_handling(mock_httpx_client):
    """Test error handling for 404."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not found", request=MagicMock(), response=mock_response
    )
    mock_httpx_client.get.return_value = mock_response

    client = XAIClient(base_url="http://test.com")
    with pytest.raises(XAINotFoundError):
        await client.get_model("nonexistent")


@pytest.mark.asyncio
async def test_context_manager(mock_httpx_client):
    """Test async context manager."""
    async with XAIClient(base_url="http://test.com") as client:
        assert client._client is not None
    mock_httpx_client.aclose.assert_awaited_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
