from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from app.db.mongo import get_db

class ModelRepository:
    """Repository for model operations."""

    @staticmethod
    async def create(model_data: Dict[str, Any]) -> str:
        """Create a new model."""
        db = await get_db()
        result = await db.models.insert_one(model_data)
        return str(result.inserted_id)

    @staticmethod
    async def get_by_id(model_id: str) -> Dict[str, Any]:
        """Get model by ID."""
        db = await get_db()
        model = await db.models.find_one({"_id": ObjectId(model_id)})
        if model:
            model["_id"] = str(model["_id"])
        return model

    @staticmethod
    async def get_by_user(user_id: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """Get models by user."""
        db = await get_db()
        cursor = db.models.find({"user_id": user_id}).sort("created_at", -1).skip(skip).limit(limit)
        models = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            models.append(doc)
        return models

    @staticmethod
    async def get_by_user_with_details(user_id: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """Get models by user with prediction counts."""
        db = await get_db()
        cursor = db.models.find({"user_id": user_id}).sort("created_at", -1).skip(skip).limit(limit)
        models = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            # Add prediction count
            doc["prediction_count"] = await db.predictions.count_documents({"model_id": doc["_id"]})
            models.append(doc)
        return models

    @staticmethod
    async def update_model(model_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a model."""
        db = await get_db()
        result = await db.models.update_one({"_id": ObjectId(model_id)}, {"$set": update_data})
        return result.modified_count > 0

    @staticmethod
    async def delete_by_id(model_id: str) -> bool:
        """Delete model by ID."""
        db = await get_db()
        result = await db.models.delete_one({"_id": ObjectId(model_id)})
        return result.deleted_count > 0

    @staticmethod
    async def delete_by_user(user_id: str) -> int:
        """Delete models by user ID."""
        db = await get_db()
        result = await db.models.delete_many({"user_id": user_id})
        return result.deleted_count

    @staticmethod
    async def count_by_user(user_id: str) -> int:
        """Count models for a specific user."""
        db = await get_db()
        return await db.models.count_documents({"user_id": user_id})

    @staticmethod
    async def find_by_name(user_id: str, name: str) -> Dict[str, Any]:
        """Find model by name for a specific user."""
        db = await get_db()
        model = await db.models.find_one({"user_id": user_id, "name": name})
        if model:
            model["_id"] = str(model["_id"])
        return model

    @staticmethod
    async def update_background_data(model_id: str, background_data_path: str) -> bool:
        """Update background data path for a model."""
        db = await get_db()
        result = await db.models.update_one(
            {"_id": ObjectId(model_id)},
            {"$set": {"background_data_path": background_data_path}}
        )
        return result.modified_count > 0