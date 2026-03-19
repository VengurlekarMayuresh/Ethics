from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from app.db.mongo import get_db

class PredictionRepository:
    """Repository for prediction operations."""

    @staticmethod
    async def create(prediction_data: Dict[str, Any]) -> str:
        """Create a new prediction."""
        db = await get_db()
        result = await db.predictions.insert_one(prediction_data)
        return str(result.inserted_id)

    @staticmethod
    async def get_by_id(prediction_id: str) -> Dict[str, Any]:
        """Get prediction by ID."""
        db = await get_db()
        prediction = await db.predictions.find_one({"_id": ObjectId(prediction_id)})
        if prediction:
            prediction["_id"] = str(prediction["_id"])
        return prediction

    @staticmethod
    async def get_by_user(user_id: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """Get predictions by user."""
        db = await get_db()
        cursor = db.predictions.find({"user_id": user_id}).sort("created_at", -1).skip(skip).limit(limit)
        predictions = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            doc["model_id"] = str(doc["model_id"])
            doc["user_id"] = str(doc["user_id"])
            predictions.append(doc)
        return predictions

    @staticmethod
    async def get_by_model(model_id: str, user_id: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """Get predictions by model and user."""
        db = await get_db()
        cursor = db.predictions.find({"model_id": model_id, "user_id": user_id}).sort("created_at", -1).skip(skip).limit(limit)
        predictions = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            doc["model_id"] = str(doc["model_id"])
            doc["user_id"] = str(doc["user_id"])
            predictions.append(doc)
        return predictions

    @staticmethod
    async def delete_by_id(prediction_id: str) -> bool:
        """Delete prediction by ID."""
        db = await get_db()
        result = await db.predictions.delete_one({"_id": ObjectId(prediction_id)})
        return result.deleted_count > 0

    @staticmethod
    async def delete_by_model(model_id: str) -> int:
        """Delete predictions by model ID."""
        db = await get_db()
        result = await db.predictions.delete_many({"model_id": model_id})
        return result.deleted_count

    @staticmethod
    async def delete_by_user(user_id: str) -> int:
        """Delete predictions by user ID."""
        db = await get_db()
        result = await db.predictions.delete_many({"user_id": user_id})
        return result.deleted_count

    @staticmethod
    async def count_by_model(model_id: str) -> int:
        """Count predictions for a specific model."""
        db = await get_db()
        return await db.predictions.count_documents({"model_id": model_id})

    @staticmethod
    async def count_by_user(user_id: str) -> int:
        """Count predictions for a specific user."""
        db = await get_db()
        return await db.predictions.count_documents({"user_id": user_id})

    @staticmethod
    async def update_prediction(prediction_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a prediction."""
        db = await get_db()
        result = await db.predictions.update_one({"_id": ObjectId(prediction_id)}, {"$set": update_data})
        return result.modified_count > 0