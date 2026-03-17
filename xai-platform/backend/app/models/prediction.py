from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime

class PredictionBase(BaseModel):
    input_data: Dict[str, Any]
    prediction: Any
    probability: Optional[List[float]] = None
    latency_ms: Optional[int] = None

class PredictionCreate(PredictionBase):
    model_id: str
    user_id: str

class PredictionResponse(PredictionBase):
    id: str = Field(alias="_id")
    model_id: str
    user_id: str
    created_at: datetime

    class Config:
        populate_by_name = True