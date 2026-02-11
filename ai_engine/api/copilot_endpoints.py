"""
QBITEL - Protocol Intelligence Copilot API Endpoints
FastAPI endpoints for the Protocol Intelligence Copilot feature.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram

from ..copilot.protocol_copilot import (
    create_protocol_copilot,
    CopilotQuery,
    CopilotResponse,
)
from ..core.config import get_config
from ..core.exceptions import QbitelAIException
from .auth import verify_token, get_current_user

# Metrics
COPILOT_API_REQUESTS = Counter(
    "qbitel_copilot_api_requests_total",
    "Total copilot API requests",
    ["endpoint", "status"],
)
COPILOT_API_DURATION = Histogram(
    "qbitel_copilot_api_duration_seconds", "Copilot API request duration", ["endpoint"]
)
WEBSOCKET_CONNECTIONS = Counter(
    "qbitel_copilot_websocket_connections_total", "Total WebSocket connections"
)

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


# Request/Response Models
class CopilotQueryRequest(BaseModel):
    """Request model for copilot queries."""

    query: str = Field(
        ..., min_length=1, max_length=2000, description="Natural language query"
    )
    session_id: Optional[str] = Field(
        None, description="Session identifier for conversation continuity"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context for the query"
    )
    packet_data: Optional[str] = Field(
        None, description="Base64 encoded packet data for analysis"
    )

    @validator("packet_data")
    def validate_packet_data(cls, v):
        if v:
            try:
                import base64

                base64.b64decode(v)
                return v
            except Exception:
                raise ValueError("packet_data must be valid base64 encoded data")
        return v


class CopilotQueryResponse(BaseModel):
    """Response model for copilot queries."""

    response: str = Field(..., description="Copilot's response to the query")
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score of the response"
    )
    query_type: str = Field(..., description="Classified type of the query")
    processing_time: float = Field(
        ..., description="Time taken to process the query in seconds"
    )
    session_id: str = Field(..., description="Session identifier")
    suggestions: List[str] = Field(default=[], description="Follow-up suggestions")
    source_data: List[Dict[str, Any]] = Field(
        default=[], description="Source data used for the response"
    )
    visualizations: Optional[List[Dict[str, Any]]] = Field(
        None, description="Visualization data"
    )
    metadata: Dict[str, Any] = Field(
        default={}, description="Additional response metadata"
    )


class SessionSummaryResponse(BaseModel):
    """Response model for session summaries."""

    session_id: str
    user_id: str
    duration: str
    total_turns: int
    query_types: Dict[str, int]
    average_confidence: float
    top_topics: List[str]
    last_activity: str


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: str = Field(
        ..., description="Message type: 'query', 'response', 'error', 'ping', 'pong'"
    )
    data: Dict[str, Any] = Field(default={}, description="Message data")
    timestamp: Optional[str] = Field(None, description="Message timestamp")
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for request-response matching"
    )


# Router setup
router = APIRouter(prefix="/api/v1/copilot", tags=["Protocol Intelligence Copilot"])

# Global copilot instance (will be initialized on startup)
_copilot_instance = None


async def get_copilot():
    """Dependency to get the copilot instance."""
    global _copilot_instance
    if _copilot_instance is None:
        _copilot_instance = await create_protocol_copilot()
    return _copilot_instance


@router.post("/query", response_model=CopilotQueryResponse)
async def process_copilot_query(
    request: CopilotQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    copilot=Depends(get_copilot),
) -> CopilotQueryResponse:
    """
    Process a natural language query using the Protocol Intelligence Copilot.

    This endpoint allows users to interact with the copilot using natural language
    for protocol analysis, field detection, security assessment, and more.
    """
    start_time = asyncio.get_event_loop().time()

    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Decode packet data if provided
        packet_data = None
        if request.packet_data:
            import base64

            packet_data = base64.b64decode(request.packet_data)

        # Create copilot query
        query = CopilotQuery(
            query=request.query,
            user_id=current_user["user_id"],
            session_id=session_id,
            context=request.context,
            packet_data=packet_data,
            timestamp=datetime.now(),
        )

        # Process query
        response = await copilot.process_query(query)

        # Log successful request
        COPILOT_API_REQUESTS.labels(endpoint="query", status="success").inc()
        COPILOT_API_DURATION.labels(endpoint="query").observe(
            asyncio.get_event_loop().time() - start_time
        )

        # Background task for analytics
        background_tasks.add_task(
            log_query_analytics,
            current_user["user_id"],
            request.query,
            response.query_type,
            response.confidence,
        )

        return CopilotQueryResponse(
            response=response.response,
            confidence=response.confidence,
            query_type=response.query_type,
            processing_time=response.processing_time,
            session_id=session_id,
            suggestions=response.suggestions or [],
            source_data=response.source_data,
            visualizations=response.visualizations,
            metadata=response.metadata or {},
        )

    except QbitelAIException as e:
        COPILOT_API_REQUESTS.labels(endpoint="query", status="error").inc()
        logger.error(f"Copilot processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        COPILOT_API_REQUESTS.labels(endpoint="query", status="error").inc()
        logger.error(f"Unexpected error in copilot query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions", response_model=List[Dict[str, Any]])
async def get_user_sessions(
    limit: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user),
    copilot=Depends(get_copilot),
) -> List[Dict[str, Any]]:
    """Get recent sessions for the current user."""
    try:
        sessions = await copilot.context_manager.get_user_sessions(
            current_user["user_id"], limit
        )

        COPILOT_API_REQUESTS.labels(endpoint="sessions", status="success").inc()
        return sessions

    except Exception as e:
        COPILOT_API_REQUESTS.labels(endpoint="sessions", status="error").inc()
        logger.error(f"Error getting user sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@router.get("/sessions/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    copilot=Depends(get_copilot),
) -> SessionSummaryResponse:
    """Get summary of a specific conversation session."""
    try:
        summary = await copilot.context_manager.get_session_summary(
            current_user["user_id"], session_id
        )

        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")

        COPILOT_API_REQUESTS.labels(endpoint="session_summary", status="success").inc()

        return SessionSummaryResponse(
            session_id=summary["session_id"],
            user_id=summary["user_id"],
            duration=summary["duration"],
            total_turns=summary["total_turns"],
            query_types=summary["query_types"],
            average_confidence=summary["average_confidence"],
            top_topics=summary["top_topics"],
            last_activity=summary["last_activity"],
        )

    except HTTPException:
        raise
    except Exception as e:
        COPILOT_API_REQUESTS.labels(endpoint="session_summary", status="error").inc()
        logger.error(f"Error getting session summary: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve session summary"
        )


@router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    copilot=Depends(get_copilot),
) -> Dict[str, str]:
    """Clear a specific conversation session."""
    try:
        success = await copilot.context_manager.clear_session(
            current_user["user_id"], session_id
        )

        if not success:
            raise HTTPException(
                status_code=404, detail="Session not found or could not be cleared"
            )

        COPILOT_API_REQUESTS.labels(endpoint="clear_session", status="success").inc()
        return {"message": f"Session {session_id} cleared successfully"}

    except HTTPException:
        raise
    except Exception as e:
        COPILOT_API_REQUESTS.labels(endpoint="clear_session", status="error").inc()
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session")


@router.get("/health")
async def get_copilot_health(copilot=Depends(get_copilot)) -> Dict[str, Any]:
    """Get health status of the copilot system."""
    try:
        health_status = copilot.get_health_status()

        return {
            "status": "healthy",
            "components": health_status,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting copilot health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# WebSocket endpoint for real-time interaction
@router.websocket("/ws")
async def websocket_copilot_chat(
    websocket: WebSocket, token: str, copilot=Depends(get_copilot)
):
    """
    WebSocket endpoint for real-time copilot interaction.

    Supports:
    - Real-time query processing
    - Conversation continuity
    - Typing indicators
    - Connection health monitoring
    """
    try:
        # Verify token (simplified - in production use proper WebSocket auth)
        try:
            user_info = verify_token(token)  # You'd implement this
        except Exception as e:
            await websocket.close(code=4001, reason="Invalid token")
            return

        await websocket.accept()
        WEBSOCKET_CONNECTIONS.inc()

        logger.info(
            f"WebSocket connection established for user {user_info.get('user_id')}"
        )

        # Send welcome message
        welcome_msg = WebSocketMessage(
            type="response",
            data={
                "response": "Hello! I'm your Protocol Intelligence Copilot. How can I help you analyze protocols, detect threats, or understand network traffic today?",
                "confidence": 1.0,
                "session_id": str(uuid.uuid4()),
            },
            timestamp=datetime.now().isoformat(),
        )
        await websocket.send_text(welcome_msg.json())

        session_id = None

        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = WebSocketMessage.parse_raw(data)

                if message.type == "ping":
                    # Health check
                    pong_msg = WebSocketMessage(
                        type="pong",
                        data={"timestamp": datetime.now().isoformat()},
                        correlation_id=message.correlation_id,
                    )
                    await websocket.send_text(pong_msg.json())
                    continue

                elif message.type == "query":
                    # Process copilot query
                    query_data = message.data

                    if not query_data.get("query"):
                        error_msg = WebSocketMessage(
                            type="error",
                            data={"error": "Query is required"},
                            correlation_id=message.correlation_id,
                        )
                        await websocket.send_text(error_msg.json())
                        continue

                    # Use existing session ID or create new one
                    session_id = (
                        query_data.get("session_id") or session_id or str(uuid.uuid4())
                    )

                    # Decode packet data if provided
                    packet_data = None
                    if query_data.get("packet_data"):
                        import base64

                        packet_data = base64.b64decode(query_data["packet_data"])

                    # Create and process query
                    copilot_query = CopilotQuery(
                        query=query_data["query"],
                        user_id=user_info["user_id"],
                        session_id=session_id,
                        context=query_data.get("context"),
                        packet_data=packet_data,
                        timestamp=datetime.now(),
                    )

                    # Send typing indicator
                    typing_msg = WebSocketMessage(
                        type="typing",
                        data={"status": "processing"},
                        correlation_id=message.correlation_id,
                    )
                    await websocket.send_text(typing_msg.json())

                    # Process query
                    response = await copilot.process_query(copilot_query)

                    # Send response
                    response_msg = WebSocketMessage(
                        type="response",
                        data={
                            "response": response.response,
                            "confidence": response.confidence,
                            "query_type": response.query_type,
                            "processing_time": response.processing_time,
                            "session_id": session_id,
                            "suggestions": response.suggestions or [],
                            "source_data": response.source_data,
                            "visualizations": response.visualizations,
                            "metadata": response.metadata or {},
                        },
                        timestamp=datetime.now().isoformat(),
                        correlation_id=message.correlation_id,
                    )
                    await websocket.send_text(response_msg.json())

                else:
                    # Unknown message type
                    error_msg = WebSocketMessage(
                        type="error",
                        data={"error": f"Unknown message type: {message.type}"},
                        correlation_id=message.correlation_id,
                    )
                    await websocket.send_text(error_msg.json())

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message processing error: {e}")
                error_msg = WebSocketMessage(
                    type="error",
                    data={"error": "Internal server error"},
                    correlation_id=(
                        getattr(message, "correlation_id", None)
                        if "message" in locals()
                        else None
                    ),
                )
                try:
                    await websocket.send_text(error_msg.json())
                except:
                    break

    except WebSocketDisconnect:
        logger.info(
            f"WebSocket disconnected for user {user_info.get('user_id', 'unknown')}"
        )
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        WEBSOCKET_CONNECTIONS.inc(-1)  # Decrement counter


# Background task functions
async def log_query_analytics(
    user_id: str, query: str, query_type: str, confidence: float
):
    """Log query analytics for improvement and monitoring."""
    try:
        analytics_data = {
            "user_id": user_id,
            "query_length": len(query),
            "query_type": query_type,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }

        # In production, you'd send this to your analytics system
        logger.info(f"Query analytics: {analytics_data}")

    except Exception as e:
        logger.error(f"Error logging query analytics: {e}")
