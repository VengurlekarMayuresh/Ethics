from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime

class ExplanationBase(BaseModel):
    prediction_id: Optional[str] = None
    model_id: str
    method: str  # "shap" | "lime"
    explanation_type: str  # "local" | "global"
    shap_values: Optional[List[List[float]]] = None
    expected_value: Optional[float] = None
    feature_names: Optional[List[str]] = None
    lime_weights: Optional[List[List[float]]] = None
    nl_explanation: Optional[str] = None
    task_id: Optional[str] = None
    task_status: str = "pending"  # "pending" | "complete" | "failed"

class ExplanationCreate(ExplanationBase):
    pass

class ExplanationResponse(ExplanationBase):
    id: str = Field(alias="_id")
    created_at: datetime

    class Config:
        populate_by_name = True