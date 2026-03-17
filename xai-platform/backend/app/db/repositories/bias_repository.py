from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from app.db.mongo import get_db

class BiasRepository:
    """Repository for bias analysis operations."""

    @staticmethod
    async def create(bias_data: Dict[str, Any]) -> str:
        """Create a new bias report."""
        db = get_db()
        result = await db.bias_reports.insert_one(bias_data)
        return str(result.inserted_id)

    @staticmethod
    async def get_by_id(bias_id: str) -> Dict[str, Any]:
        """Get bias report by ID."""
        db = get_db()
        bias = await db.bias_reports.find_one({"_id": ObjectId(bias_id)})
        if bias:
            bias["_id"] = str(bias["_id"])
        return bias

    @staticmethod
    async def get_by_model(model_id: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """Get bias reports by model."""
        db = get_db()
        cursor = db.bias_reports.find({"model_id": model_id}).sort("created_at", -1).skip(skip).limit(limit)
        bias_reports = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            bias_reports.append(doc)
        return bias_reports

    @staticmethod
    async def get_latest_by_model(model_id: str) -> Dict[str, Any]:
        """Get latest bias report by model."""
        db = get_db()
        bias = await db.bias_reports.find_one({"model_id": model_id}, sort=[("created_at", -1)])
        if bias:
            bias["_id"] = str(bias["_id"])
        return bias

    @staticmethod
    async def get_by_user(user_id: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """Get bias reports by user."""
        db = get_db()
        cursor = db.bias_reports.find({"user_id": user_id}).sort("created_at", -1).skip(skip).limit(limit)
        bias_reports = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            bias_reports.append(doc)
        return bias_reports

    @staticmethod
    async def update_bias(bias_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a bias report."""
        db = get_db()
        result = await db.bias_reports.update_one({"_id": ObjectId(bias_id)}, {"$set": update_data})
        return result.modified_count > 0

    @staticmethod
    async def delete_by_id(bias_id: str) -> bool:
        """Delete bias report by ID."""
        db = get_db()
        result = await db.bias_reports.delete_one({"_id": ObjectId(bias_id)})
        return result.deleted_count > 0

    @staticmethod
    async def delete_by_model(model_id: str) -> int:
        """Delete bias reports by model ID."""
        db = get_db()
        result = await db.bias_reports.delete_many({"model_id": model_id})
        return result.deleted_count

    @staticmethod
    async def delete_by_user(user_id: str) -> int:
        """Delete bias reports by user ID."""
        db = get_db()
        result = await db.bias_reports.delete_many({"user_id": user_id})
        return result.deleted_count

    @staticmethod
    async def count_by_model(model_id: str) -> int:
        """Count bias reports for a specific model."""
        db = get_db()
        return await db.bias_reports.count_documents({"model_id": model_id})

    @staticmethod
    async def count_by_user(user_id: str) -> int:
        """Count bias reports for a specific user."""
        db = get_db()
        return await db.bias_reports.count_documents({"user_id": user_id})

    @staticmethod
    async def get_metrics_by_model(model_id: str) -> List[Dict[str, Any]]:
        """Get all metrics for a specific model."""
        db = get_db()
        cursor = db.bias_reports.find({"model_id": model_id})
        metrics = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            metrics.append(doc)
        return metrics