from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId
from app.db.mongo import get_db

class UserRepository:
    """Repository for user operations."""

    @staticmethod
    async def create(user_data: Dict[str, Any]) -> str:
        """Create a new user."""
        db = get_db()
        result = await db.users.insert_one(user_data)
        return str(result.inserted_id)

    @staticmethod
    async def get_by_id(user_id: str) -> Dict[str, Any]:
        """Get user by ID."""
        db = get_db()
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if user:
            user["_id"] = str(user["_id"])
        return user

    @staticmethod
    async def get_by_email(email: str) -> Dict[str, Any]:
        """Get user by email."""
        db = get_db()
        user = await db.users.find_one({"email": email})
        if user:
            user["_id"] = str(user["_id"])
        return user

    @staticmethod
    async def exists_by_email(email: str) -> bool:
        """Check if user exists by email."""
        db = get_db()
        count = await db.users.count_documents({"email": email})
        return count > 0

    @staticmethod
    async def update_user(user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user information."""
        db = get_db()
        result = await db.users.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})
        return result.modified_count > 0

    @staticmethod
    async def update_password(user_id: str, hashed_password: str) -> bool:
        """Update user password."""
        db = get_db()
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"hashed_password": hashed_password}}
        )
        return result.modified_count > 0

    @staticmethod
    async def delete_by_id(user_id: str) -> bool:
        """Delete user by ID."""
        db = get_db()
        result = await db.users.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count > 0

    @staticmethod
    async def add_api_key(user_id: str, api_key: str) -> bool:
        """Add API key to user."""
        db = get_db()
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$addToSet": {"api_keys": api_key}}
        )
        return result.modified_count > 0

    @staticmethod
    async def remove_api_key(user_id: str, api_key: str) -> bool:
        """Remove API key from user."""
        db = get_db()
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$pull": {"api_keys": api_key}}
        )
        return result.modified_count > 0

    @staticmethod
    async def list_all(limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """List all users (admin function)."""
        db = get_db()
        cursor = db.users.find({}).sort("created_at", -1).skip(skip).limit(limit)
        users = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            users.append(doc)
        return users