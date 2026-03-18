from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class AuditLogCreate(BaseModel):
    """Schema for creating an audit log entry."""
    user_id: str
    action: str = Field(..., description="Action performed (e.g., 'model.upload', 'prediction.create')")
    resource_type: str = Field(..., description="Type of resource (model, prediction, explanation, etc.)")
    resource_id: Optional[str] = Field(None, description="ID of the affected resource")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    ip_address: Optional[str] = Field(None, description="IP address of the requester")
    user_agent: Optional[str] = Field(None, description="User agent of the requester")

class AuditLogResponse(BaseModel):
    """Schema for audit log response."""
    id: str = Field(..., alias="_id")
    user_id: str
    action: str
    resource_type: str
    resource_id: Optional[str]
    details: Optional[Dict[str, Any]] = {}
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: datetime

    class Config:
        populate_by_name = True
        from_attributes = True

class AuditLogFilter(BaseModel):
    """Schema for filtering audit logs."""
    action: Optional[str] = None
    user_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = 100
    skip: int = 0
