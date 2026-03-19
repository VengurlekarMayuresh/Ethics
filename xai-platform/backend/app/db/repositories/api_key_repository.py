from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bson import ObjectId
from app.db.mongo import get_db
import bcrypt
import secrets

class APIKeyRepository:
    """Repository for API key operations."""

    @staticmethod
    async def create(user_id: str, name: str, scopes: List[str], expires_in_days: Optional[int] = None) -> Dict[str, Any]:
        """Create a new API key."""
        raw_key = secrets.token_urlsafe(32)
        key_hash = bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()
        key_prefix = raw_key[:8]

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        api_key_doc = {
            "user_id": user_id,
            "name": name,
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "scopes": scopes,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "last_used_at": None
        }

        db = await get_db()
        result = await db.api_keys.insert_one(api_key_doc)
        api_key_id = str(result.inserted_id)

        # Return raw key (only time it's shown)
        return {
            "id": api_key_id,
            "name": name,
            "key": raw_key,
            "scopes": scopes,
            "created_at": api_key_doc["created_at"],
            "expires_at": expires_at
        }

    @staticmethod
    async def get_by_id(api_key_id: str) -> Optional[Dict[str, Any]]:
        """Get API key by ID (without sensitive fields)."""
        db = await get_db()
        doc = await db.api_keys.find_one({"_id": ObjectId(api_key_id)})
        if doc:
            return APIKeyRepository._sanitize(doc)
        return None

    @staticmethod
    async def get_by_user(user_id: str) -> List[Dict[str, Any]]:
        """List API keys for a user."""
        db = await get_db()
        cursor = db.api_keys.find({"user_id": user_id}).sort("created_at", -1)
        keys = []
        async for doc in cursor:
            keys.append(APIKeyRepository._sanitize(doc))
        return keys

    @staticmethod
    def _sanitize(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive fields."""
        doc["id"] = str(doc["_id"])
        doc.pop("_id", None)
        doc.pop("key_hash", None)
        doc.pop("key_prefix", None)
        return doc

    @staticmethod
    async def verify(raw_key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key and return its data if valid."""
        if len(raw_key) < 8:
            return None
        key_prefix = raw_key[:8]
        db = await get_db()
        # Find keys with matching prefix
        cursor = db.api_keys.find({"key_prefix": key_prefix})
        async for doc in cursor:
            if bcrypt.checkpw(raw_key.encode(), doc["key_hash"].encode()):
                # Check expiration
                if doc.get("expires_at") and doc["expires_at"] < datetime.utcnow():
                    return None
                # Update last used
                await db.api_keys.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"last_used_at": datetime.utcnow()}}
                )
                return APIKeyRepository._sanitize(doc)
        return None

    @staticmethod
    async def delete(api_key_id: str, user_id: str) -> bool:
        """Revoke an API key."""
        db = await get_db()
        result = await db.api_keys.delete_one({
            "_id": ObjectId(api_key_id),
            "user_id": user_id
        })
        return result.deleted_count > 0
