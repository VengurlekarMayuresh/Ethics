"""
WebSocket connection manager for real-time notifications.
"""

from typing import Dict, List, Set
import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState


class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""

    def __init__(self):
        # Map of user_id to set of connections
        self.user_connections: Dict[str, Set[WebSocket]] = {}
        # Map of connection to user_id for cleanup
        self.connection_user: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept a new WebSocket connection and associate with user."""
        await websocket.accept()
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(websocket)
        self.connection_user[websocket] = user_id

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        user_id = self.connection_user.get(websocket)
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        if websocket in self.connection_user:
            del self.connection_user[websocket]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(message)
            except Exception as e:
                # Connection might be closed
                self.disconnect(websocket)

    async def send_to_user(self, message: dict, user_id: str):
        """Send a message to all active connections for a user."""
        if user_id in self.user_connections:
            dead_connections = []
            for websocket in self.user_connections[user_id]:
                try:
                    if websocket.application_state == WebSocketState.CONNECTED:
                        await websocket.send_json(message)
                    else:
                        dead_connections.append(websocket)
                except Exception:
                    dead_connections.append(websocket)

            # Clean up dead connections
            for ws in dead_connections:
                self.disconnect(ws)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        dead_connections = []
        for websocket in self.connection_user:
            try:
                if websocket.application_state == WebSocketState.CONNECTED:
                    await websocket.send_json(message)
                else:
                    dead_connections.append(websocket)
            except Exception:
                dead_connections.append(websocket)

        for ws in dead_connections:
            self.disconnect(ws)

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return sum(len(conns) for conns in self.user_connections.values())


# Global connection manager instance
manager = ConnectionManager()
