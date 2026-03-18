from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import secrets

class APIKeyCreate(BaseModel):
    name: str
    scopes: List[str] = ["read", "predict"]  # e.g., ["read", "predict", "explain", "admin"]
    expires_in_days: Optional[int] = None

class APIKeyResponse(BaseModel):
    id: str = Field(alias="_id")
    name: str
    key: str  # Only shown once upon creation
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None

    class Config:
        populate_by_name = True

class APIKeyInDB(BaseModel):
    id: str
    user_id: str
    name: str
    key_hash: str  # Hashed version of the key
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None

    class Config:
        populate_by_name = True

def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)
