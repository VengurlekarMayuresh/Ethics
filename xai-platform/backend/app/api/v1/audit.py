from fastapi import APIRouter, Depends, HTTPException, Query, Request
from app.models.audit import AuditLogResponse, AuditLogFilter
from app.api.v1.auth import get_current_user, get_current_user_optional
from app.db.mongo import get_db
from app.db.repositories.audit_repository import AuditRepository
from typing import List
from datetime import datetime

router = APIRouter()

@router.get("/", response_model=List[AuditLogResponse])
async def get_audit_logs(
    action: Optional[str] = Query(None, description="Filter by action"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    resource_id: Optional[str] = Query(None, description="Filter by resource ID"),
    start_date: Optional[datetime] = Query(None, description="Filter from date"),
    end_date: Optional[datetime] = Query(None, description="Filter to date"),
    limit: int = Query(100, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """
    Get audit logs with optional filters.
    Admins can see all logs; users can only see their own.
    """
    try:
        db = get_db()

        # For now, allow all logged-in users to see their own logs
        # In production, add role-based access control
        query_user_id = user_id if current_user.get("role") == "admin" else current_user["_id"]

        logs = await AuditRepository.get_all(
            db,
            limit=limit,
            skip=skip,
            action=action,
            user_id=query_user_id,
            start_date=start_date,
            end_date=end_date
        )

        # Convert _id to id for response
        for log in logs:
            log["_id"] = log["_id"] if isinstance(log["_id"], str) else str(log["_id"])

        return logs

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/my", response_model=List[AuditLogResponse])
async def get_my_audit_logs(
    limit: int = Query(100, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """Get audit logs for the current authenticated user."""
    try:
        db = get_db()
        logs = await AuditRepository.get_by_user(db, current_user["_id"], limit, skip)
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/resource/{resource_type}/{resource_id}", response_model=List[AuditLogResponse])
async def get_audit_logs_by_resource(
    resource_type: str,
    resource_id: str,
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(get_current_user)
):
    """Get audit logs for a specific resource."""
    try:
        db = get_db()
        logs = await AuditRepository.get_by_resource(db, resource_type, resource_id, limit)
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/count")
async def get_audit_log_count(
    action: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Get count of audit logs matching filters."""
    try:
        db = get_db()

        query = {}
        if action:
            query["action"] = action
        if user_id and current_user.get("role") == "admin":
            query["user_id"] = user_id
        elif not current_user.get("role") == "admin":
            query["user_id"] = current_user["_id"]
        if start_date or end_date:
            query["created_at"] = {}
            if start_date:
                query["created_at"]["$gte"] = start_date
            if end_date:
                query["created_at"]["$lte"] = end_date

        count = await AuditRepository.count(db, query)
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
