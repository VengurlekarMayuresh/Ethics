from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from app.db.mongo import get_db

class ExplanationRepository:
    """Repository for explanation operations."""

    @staticmethod
    async def create(explanation_data: Dict[str, Any]) -> str:
        """Create a new explanation."""
        db = await get_db()
        result = await db.explanations.insert_one(explanation_data)
        return str(result.inserted_id)

    @staticmethod
    async def get_by_id(explanation_id: str) -> Dict[str, Any]:
        """Get explanation by ID."""
        db = await get_db()
        explanation = await db.explanations.find_one({"_id": ObjectId(explanation_id)})
        if explanation:
            explanation["_id"] = str(explanation["_id"])
        return explanation

    @staticmethod
    async def get_by_prediction(prediction_id: str) -> Dict[str, Any]:
        """Get explanation by prediction ID."""
        db = await get_db()
        explanation = await db.explanations.find_one({"prediction_id": prediction_id})
        if explanation:
            explanation["_id"] = str(explanation["_id"])
        return explanation

    @staticmethod
    async def get_by_model(model_id: str, explanation_type: str = None, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """Get explanations by model."""
        db = await get_db()
        query = {"model_id": model_id}
        if explanation_type:
            query["explanation_type"] = explanation_type

        cursor = db.explanations.find(query).sort("created_at", -1).skip(skip).limit(limit)
        explanations = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            doc["model_id"] = str(doc["model_id"])
            if doc.get("prediction_id"):
                doc["prediction_id"] = str(doc["prediction_id"])
            explanations.append(doc)
        return explanations

    @staticmethod
    async def get_latest_by_model(model_id: str, explanation_type: str = None) -> Dict[str, Any]:
        """Get latest explanation by model."""
        db = await get_db()
        query = {"model_id": model_id}
        if explanation_type:
            query["explanation_type"] = explanation_type

        explanation = await db.explanations.find_one(query, sort=[("created_at", -1)])
        if explanation:
            explanation["_id"] = str(explanation["_id"])
            explanation["model_id"] = str(explanation["model_id"])
            if explanation.get("prediction_id"):
                explanation["prediction_id"] = str(explanation["prediction_id"])
        return explanation

    @staticmethod
    async def update_explanation(explanation_id: str, update_data: Dict[str, Any]) -> bool:
        """Update an explanation."""
        db = await get_db()
        result = await db.explanations.update_one({"_id": ObjectId(explanation_id)}, {"$set": update_data})
        return result.modified_count > 0

    @staticmethod
    async def delete_by_id(explanation_id: str) -> bool:
        """Delete explanation by ID."""
        db = await get_db()
        result = await db.explanations.delete_one({"_id": ObjectId(explanation_id)})
        return result.deleted_count > 0

    @staticmethod
    async def delete_by_model(model_id: str) -> int:
        """Delete explanations by model ID."""
        db = await get_db()
        result = await db.explanations.delete_many({"model_id": model_id})
        return result.deleted_count

    @staticmethod
    async def delete_by_prediction(prediction_id: str) -> int:
        """Delete explanations by prediction ID."""
        db = await get_db()
        result = await db.explanations.delete_many({"prediction_id": prediction_id})
        return result.deleted_count

    @staticmethod
    async def count_by_model(model_id: str) -> int:
        """Count explanations for a specific model."""
        db = await get_db()
        return await db.explanations.count_documents({"model_id": model_id})

    @staticmethod
    async def count_by_prediction(prediction_id: str) -> int:
        """Count explanations for a specific prediction."""
        db = await get_db()
        return await db.explanations.count_documents({"prediction_id": prediction_id})

    @staticmethod
    async def get_pending_tasks(limit: int = 50) -> List[Dict[str, Any]]:
        """Get pending explanations tasks."""
        db = await get_db()
        cursor = db.explanations.find({"task_status": "pending"}).sort("created_at", -1).limit(limit)
        explanations = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            doc["model_id"] = str(doc["model_id"])
            if doc.get("prediction_id"):
                doc["prediction_id"] = str(doc["prediction_id"])
            explanations.append(doc)
        return explanations