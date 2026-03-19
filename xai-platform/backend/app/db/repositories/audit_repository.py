from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
from app.models.audit import AuditLogCreate

class AuditRepository:
    @staticmethod
    async def create(db, audit_data: Dict[str, Any]) -> str:
        """Create an audit log entry."""
        audit_doc = {
            "user_id": audit_data.get("user_id"),
            "action": audit_data.get("action"),
            "resource_type": audit_data.get("resource_type"),
            "resource_id": audit_data.get("resource_id"),
            "details": audit_data.get("details", {}),
            "ip_address": audit_data.get("ip_address"),
            "user_agent": audit_data.get("user_agent"),
            "created_at": datetime.utcnow()
        }
        result = await db.audit_logs.insert_one(audit_doc)
        return str(result.inserted_id)

    @staticmethod
    async def get_by_user(db, user_id: str, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """Get audit logs for a specific user."""
        cursor = db.audit_logs.find({"user_id": user_id})\
            .sort("created_at", -1)\
            .skip(skip)\
            .limit(limit)
        logs = []
        async for log in cursor:
            log["_id"] = str(log["_id"])
            logs.append(log)
        return logs

    @staticmethod
    async def get_by_resource(
        db,
        resource_type: str,
        resource_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit logs for a specific resource."""
        cursor = db.audit_logs.find({
            "resource_type": resource_type,
            "resource_id": resource_id
        }).sort("created_at", -1).limit(limit)
        logs = []
        async for log in cursor:
            log["_id"] = str(log["_id"])
            logs.append(log)
        return logs

    @staticmethod
    async def get_all(
        db,
        limit: int = 100,
        skip: int = 0,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get all audit logs with optional filters."""
        query = {}
        if action:
            query["action"] = action
        if user_id:
            query["user_id"] = user_id
        if start_date or end_date:
            query["created_at"] = {}
            if start_date:
                query["created_at"]["$gte"] = start_date
            if end_date:
                query["created_at"]["$lte"] = end_date

        cursor = db.audit_logs.find(query)\
            .sort("created_at", -1)\
            .skip(skip)\
            .limit(limit)
        logs = []
        async for log in cursor:
            log["_id"] = str(log["_id"])
            logs.append(log)
        return logs

    @staticmethod
    async def count(db, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count audit logs matching filters."""
        query = filters or {}
        return await db.audit_logs.count_documents(query)
