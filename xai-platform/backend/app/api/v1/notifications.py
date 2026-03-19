"""
WebSocket endpoint for real-time notifications.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException, status
from app.websocket.manager import manager
from app.utils.auth import decode_token
from app.db.repositories.api_key_repository import APIKeyRepository
from app.db.mongo import get_db
from bson import ObjectId
import json
import asyncio

router = APIRouter()

async def get_user_from_token(token: str):
    """Authenticate user from JWT or API key."""
    # Try JWT first
    payload = decode_token(token)
    if payload:
        db = await get_db()
        user = await db.users.find_one({"email": payload.get("sub")})
        if user:
            user["_id"] = str(user["_id"])
            return user

    # Try API key
    api_key_data = await APIKeyRepository.verify(token)
    if api_key_data:
        db = await get_db()
        user = await db.users.find_one({"_id": ObjectId(api_key_data["user_id"])})
        if user:
            user["_id"] = str(user["_id"])
            return user

    return None

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="JWT or API key for authentication")
):
    """
    WebSocket endpoint for real-time notifications.

    Query params:
      - token: JWT or API key for authentication

    The server will automatically send notifications for task completions.
    """
    try:
        # Authenticate
        user = await get_user_from_token(token)
        if not user:
            await websocket.close(code=4001, reason="Unauthorized")
            return

        user_id = str(user["_id"])
        await manager.connect(websocket, user_id)

        # Send welcome message
        await manager.send_personal_message(
            {"type": "connected", "message": "WebSocket connected"},
            websocket
        )

        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": asyncio.get_event_loop().time()}))
            except Exception:
                break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))
