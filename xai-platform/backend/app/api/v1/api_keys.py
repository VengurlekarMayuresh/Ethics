from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import List
from app.models.api_key import APIKeyCreate, APIKeyResponse
from app.api.v1.auth import get_current_user
from app.db.repositories.api_key_repository import APIKeyRepository
from app.utils.audit_logger import log_action, AuditActions
from datetime import datetime

router = APIRouter()

@router.get("/", response_model=List[APIKeyResponse])
async def list_api_keys(current_user: dict = Depends(get_current_user)):
    """List all API keys for the current user."""
    try:
        keys = await APIKeyRepository.get_by_user(current_user["_id"])
        return keys
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=APIKeyResponse)
async def create_api_key(
    request: Request,
    key_data: APIKeyCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new API key."""
    try:
        result = await APIKeyRepository.create(
            user_id=current_user["_id"],
            name=key_data.name,
            scopes=key_data.scopes,
            expires_in_days=key_data.expires_in_days
        )

        # Log audit event
        await log_action(
            user_id=current_user["_id"],
            action=AuditActions.API_KEY_CREATE,
            resource_type="api_key",
            resource_id=result["_id"] if "_id" in result else None,
            details={"name": key_data.name, "scopes": key_data.scopes},
            request=request
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{key_id}")
async def delete_api_key(
    request: Request,
    key_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Revoke (delete) an API key."""
    try:
        success = await APIKeyRepository.delete(key_id, current_user["_id"])
        if not success:
            raise HTTPException(status_code=404, detail="API key not found")

        # Log audit event
        await log_action(
            user_id=current_user["_id"],
            action=AuditActions.API_KEY_DELETE,
            resource_type="api_key",
            resource_id=key_id,
            request=request
        )

        return {"status": "success", "message": "API key revoked"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
