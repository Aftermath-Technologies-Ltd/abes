# Author: Bradley R. Kinnard
"""
Chat API routes with WebSocket support for real-time belief streaming.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ...chat import ChatService, BeliefEvent, ChatTurn, get_chat_service
from ...core.deps import get_belief_store
from ...core.models.user import User
from .auth import get_current_user, get_optional_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# === Request/Response Models ===

class ChatMessageRequest(BaseModel):
    """Request to send a chat message."""
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    context: Optional[str] = None
    stream: bool = False


class BeliefEventResponse(BaseModel):
    """A belief event for the API."""
    event_type: str
    belief_id: str
    content: str
    confidence: float
    tension: float
    details: dict = Field(default_factory=dict)
    timestamp: str


class ChatTurnResponse(BaseModel):
    """Response for a chat turn."""
    id: str
    user_message: str
    assistant_message: str
    beliefs_created: list[str]
    beliefs_reinforced: list[str]
    beliefs_mutated: list[str]
    beliefs_deprecated: list[str]
    beliefs_used: list[str]
    events: list[BeliefEventResponse]
    duration_ms: float
    timestamp: str


class SessionResponse(BaseModel):
    """Response for session info."""
    id: str
    turn_count: int
    created_at: str


class SessionListResponse(BaseModel):
    """Response for listing sessions."""
    sessions: list[SessionResponse]
    total: int


# === WebSocket Connection Manager ===

class ConnectionManager:
    """Manages WebSocket connections for belief event streaming."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast_event(self, event: BeliefEvent) -> None:
        """Broadcast a belief event to all connected clients."""
        if not self.active_connections:
            return

        message = {
            "type": "belief_event",
            "data": {
                "event_type": event.event_type,
                "belief_id": str(event.belief_id),
                "content": event.content,
                "confidence": event.confidence,
                "tension": event.tension,
                "details": event.details,
                "timestamp": event.timestamp.isoformat(),
            }
        }

        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                dead_connections.append(connection)

        for conn in dead_connections:
            self.disconnect(conn)

    async def broadcast_chat_chunk(self, chunk: str, done: bool = False) -> None:
        """Broadcast a streaming chat response chunk."""
        if not self.active_connections:
            return

        message = {
            "type": "chat_chunk",
            "data": {
                "content": chunk,
                "done": done,
            }
        }

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


# Global connection manager
manager = ConnectionManager()


def _event_callback(event: BeliefEvent) -> None:
    """Callback to broadcast belief events via WebSocket."""
    asyncio.create_task(manager.broadcast_event(event))


def _get_service() -> ChatService:
    """Get chat service with event callback configured."""
    store = get_belief_store()
    service = get_chat_service(store)
    service.event_callback = _event_callback
    return service


def _turn_to_response(turn: ChatTurn) -> ChatTurnResponse:
    """Convert ChatTurn to API response."""
    return ChatTurnResponse(
        id=str(turn.id),
        user_message=turn.user_message,
        assistant_message=turn.assistant_message,
        beliefs_created=[str(b) for b in turn.beliefs_created],
        beliefs_reinforced=[str(b) for b in turn.beliefs_reinforced],
        beliefs_mutated=[str(b) for b in turn.beliefs_mutated],
        beliefs_deprecated=[str(b) for b in turn.beliefs_deprecated],
        beliefs_used=[str(b) for b in turn.beliefs_used],
        events=[
            BeliefEventResponse(
                event_type=e.event_type,
                belief_id=str(e.belief_id),
                content=e.content,
                confidence=e.confidence,
                tension=e.tension,
                details=e.details,
                timestamp=e.timestamp.isoformat(),
            )
            for e in turn.events
        ],
        duration_ms=turn.duration_ms,
        timestamp=turn.timestamp.isoformat(),
    )


# === REST Endpoints ===

@router.post("/message", response_model=ChatTurnResponse)
async def send_message(req: ChatMessageRequest, user: User = Depends(get_current_user)):
    """
    Send a chat message and get a response.

    The message is processed through the full ABES pipeline:
    1. Extract beliefs from user message
    2. Create/reinforce/mutate beliefs as needed
    3. Rank beliefs by relevance
    4. Generate LLM response with belief context

    Belief events are streamed via WebSocket to /ws/beliefs.
    """
    service = _get_service()

    session_id = UUID(req.session_id) if req.session_id else None

    try:
        turn = await service.process_message(
            message=req.message,
            session_id=session_id,
            context=req.context,
            user_id=user.id,  # Associate with user
        )
        return _turn_to_response(turn)
    except Exception as e:
        logger.exception("Chat message processing failed")
        raise HTTPException(500, f"Chat processing failed: {str(e)}")


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """List all chat sessions."""
    service = _get_service()
    sessions = service.list_sessions()

    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=str(s.id),
                turn_count=len(s.turns),
                created_at=s.created_at.isoformat(),
            )
            for s in sessions
        ],
        total=len(sessions),
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session with its history."""
    service = _get_service()

    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(400, "Invalid session ID")

    session = service.get_session(sid)
    if not session:
        raise HTTPException(404, "Session not found")

    return {
        "id": str(session.id),
        "created_at": session.created_at.isoformat(),
        "turns": [_turn_to_response(t).model_dump() for t in session.turns],
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    service = _get_service()

    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(400, "Invalid session ID")

    if service.clear_session(sid):
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(404, "Session not found")


@router.delete("/sessions")
async def delete_all_sessions():
    """Delete all chat sessions at once."""
    service = _get_service()
    count = service.clear_all_sessions()
    return {"status": "cleared", "deleted": count}


@router.post("/sessions")
async def create_session():
    """Create a new chat session."""
    service = _get_service()
    session = service.get_or_create_session()

    return {
        "id": str(session.id),
        "created_at": session.created_at.isoformat(),
    }


# === WebSocket Endpoint ===

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time belief events and streaming chat.

    Receives:
    - {"type": "chat", "message": "...", "session_id": "..."}
    - {"type": "ping"}

    Sends:
    - {"type": "belief_event", "data": {...}}
    - {"type": "chat_chunk", "data": {"content": "...", "done": bool}}
    - {"type": "chat_complete", "data": {...}}
    - {"type": "pong"}
    """
    await manager.connect(websocket)
    service = _get_service()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "chat":
                message = data.get("message", "")
                session_id = data.get("session_id")

                if not message:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": "Empty message"},
                    })
                    continue

                sid = UUID(session_id) if session_id else None

                # Stream the response
                async for chunk, turn in service.process_message_stream(
                    message=message,
                    session_id=sid,
                ):
                    if chunk:
                        await websocket.send_json({
                            "type": "chat_chunk",
                            "data": {"content": chunk, "done": False},
                        })

                    if turn:
                        await websocket.send_json({
                            "type": "chat_complete",
                            "data": _turn_to_response(turn).model_dump(),
                        })

            else:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"Unknown message type: {msg_type}"},
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.exception("WebSocket error")
        manager.disconnect(websocket)
