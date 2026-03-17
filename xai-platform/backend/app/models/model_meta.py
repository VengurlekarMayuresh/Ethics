from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

class FeatureSchema(BaseModel):
    name: str
    type: str # "numeric" | "categorical"
    options: Optional[List[str]] = []

class ModelBase(BaseModel):
    name: str
    description: Optional[str] = ""
    framework: str # "sklearn" | "xgboost" | "keras" | "onnx" | "api"
    task_type: str # "classification" | "regression"
    feature_schema: List[FeatureSchema] = []
    protected_attributes: List[str] = []
    tags: List[str] = []
    version: str = "1.0"

class ModelCreate(ModelBase):
    pass

class ModelResponse(ModelBase):
    id: str = Field(alias="_id")
    user_id: str
    file_path: Optional[str] = None
    background_data_path: Optional[str] = None
    metrics: dict = {}
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True
