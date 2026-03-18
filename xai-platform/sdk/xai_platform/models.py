"""
Pydantic models for XAI Platform API request/response schemas.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class ModelCreate(BaseModel):
    """Schema for model upload."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    task_type: str = Field(..., description="Type of ML task: classification, regression, etc.")
    framework: str = Field(..., description="ML framework: sklearn, xgboost, onnx, keras, lightgbm")
    feature_schema: Dict[str, Any] = Field(..., description="Feature schema defining inputs")
    target_schema: Optional[Dict[str, Any]] = Field(None, description="Target/output schema")


class ModelResponse(BaseModel):
    """Schema for model response."""
    id: str
    name: str
    description: Optional[str]
    task_type: str
    framework: str
    feature_schema: Dict[str, Any]
    target_schema: Optional[Dict[str, Any]]
    file_path: str
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    """Schema for single prediction request."""
    input_data: Dict[str, Any] = Field(..., description="Feature values for prediction")


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    id: str
    model_id: str
    input_data: Dict[str, Any]
    prediction: Any
    probability: Optional[float]
    created_at: datetime


class SHAPRequest(BaseModel):
    """Schema for SHAP explanation request."""
    prediction_id: Optional[str] = Field(None, description="Use existing prediction")
    input_data: Optional[Dict[str, Any]] = Field(None, description="New input for prediction + explanation")


class SHAPResponse(BaseModel):
    """Schema for SHAP explanation response."""
    explanation_id: Optional[str]
    task_id: str
    status: str
    explanation_data: Optional[Dict[str, Any]] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None
    shap_values: Optional[List[float]] = None
    feature_names: Optional[List[str]] = None


class LIMERequest(BaseModel):
    """Schema for LIME explanation request."""
    prediction_id: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    num_features: int = Field(10, ge=1, le=100, description="Number of features to include")


class LimeResponse(BaseModel):
    """Schema for LIME explanation response."""
    explanation_id: Optional[str]
    task_id: str
    status: str
    explanation_data: Optional[Dict[str, Any]] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None


class BiasAnalysisRequest(BaseModel):
    """Schema for bias analysis request."""
    model_id: str
    protected_attribute: str = Field(..., description="Column name for protected attribute")
    sensitive_attribute: str = Field(..., description="Column name for sensitive attribute")
    file: bytes = Field(..., description="CSV file bytes for evaluation dataset")


class BiasMetrics(BaseModel):
    """Schema for bias metrics."""
    demographic_parity_diff: float
    equal_opportunity_diff: float
    disparate_impact_ratio: float
    group_metrics: Dict[str, Dict[str, float]]


class BiasReport(BaseModel):
    """Schema for bias report."""
    id: str
    model_id: str
    protected_attribute: str
    sensitive_attribute: str
    metrics: BiasMetrics
    dataset_size: int
    created_at: datetime


class APIKeyCreate(BaseModel):
    """Schema for creating API key."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class APIKeyResponse(BaseModel):
    """Schema for API key response."""
    id: str
    name: str
    key: str  # Only shown on creation
    description: Optional[str]
    is_active: bool
    last_used: Optional[datetime]
    created_at: datetime
    usage_count: int
