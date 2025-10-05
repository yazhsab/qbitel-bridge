"""
Comprehensive tests for ai_engine/api/copilot_endpoints.py

Tests cover:
- Copilot query processing
- Session management
- WebSocket connections
- Request validation
- Background tasks
- Health checks
- Metrics tracking
- Error handling
"""

import pytest
import asyncio
import base64
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from fastapi import HTTPException, BackgroundTasks, WebSocket
from fastapi.testclient import TestClient

from ai_engine.api.copilot_endpoints import (
    router,
    CopilotQueryRequest,
    CopilotQueryResponse,
    SessionSummaryResponse,
    WebSocketMessage,
    get_copilot,
    process_copilot_query,
    get_user_sessions,
    get_session_summary,
    clear_session,
    get_copilot_health,
    log_query_analytics,
)
from ai_engine.copilot.protocol_copilot import CopilotQuery, CopilotResponse
from ai_engine.core.exceptions import CronosAIException


# Fixtures

@pytest.fixture
def mock_copilot():
    """Create mock copilot instance."""
    copilot = AsyncMock()
    copilot.process_query = AsyncMock()
    copilot.context_manager = AsyncMock()
    copilot.get_health_status = Mock(return_value={
        "llm": "healthy",
        "knowledge_base": "healthy",
        "context_manager": "healthy"
    })
    return copilot


@pytest.fixture
def mock_current_user():
    """Create mock current user."""
    return {
        "user_id": "user-123",
        "username": "testuser",
        "role": "analyst",
        "permissions": ["copilot_access"]
    }


@pytest.fixture
def mock_background_tasks():
    """Create mock background tasks."""
    return Mock(spec=BackgroundTasks)


@pytest.fixture
def sample_copilot_response():
    """Create sample copilot response."""
    return CopilotResponse(
        response="This is HTTP traffic on port 80.",
        confidence=0.95,
        query_type="protocol_analysis",
        processing_time=0.25,
        suggestions=["Analyze headers", "Check for anomalies"],
        source_data=[{"type": "protocol", "name": "HTTP"}],
        visualizations=[{"type": "chart", "data": []}],
        metadata={"analyzed": True}
    )


# Request Model Tests

def test_copilot_query_request_validation():
    """Test copilot query request validation."""
    request = CopilotQueryRequest(
        query="What is this protocol?",
        session_id="session-123",
        context={"source": "192.168.1.1"}
    )

    assert request.query == "What is this protocol?"
    assert request.session_id == "session-123"


def test_copilot_query_request_with_packet_data():
    """Test request with valid base64 packet data."""
    packet_data = base64.b64encode(b"GET / HTTP/1.1\r\n").decode()

    request = CopilotQueryRequest(
        query="Analyze this packet",
        packet_data=packet_data
    )

    assert request.packet_data == packet_data


def test_copilot_query_request_invalid_packet_data():
    """Test request with invalid base64 packet data."""
    with pytest.raises(ValueError, match="valid base64"):
        CopilotQueryRequest(
            query="Test",
            packet_data="!!!invalid_base64!!!"
        )


def test_copilot_query_request_min_length():
    """Test query minimum length validation."""
    with pytest.raises(ValueError):
        CopilotQueryRequest(query="")


def test_copilot_query_request_max_length():
    """Test query maximum length validation."""
    long_query = "a" * 2001

    with pytest.raises(ValueError):
        CopilotQueryRequest(query=long_query)


# Query Processing Tests

@pytest.mark.asyncio
async def test_process_copilot_query_success(
    mock_copilot,
    mock_current_user,
    mock_background_tasks,
    sample_copilot_response
):
    """Test successful copilot query processing."""
    request = CopilotQueryRequest(query="What protocol is this?")

    mock_copilot.process_query.return_value = sample_copilot_response

    with patch("ai_engine.api.copilot_endpoints.get_copilot", return_value=mock_copilot):
        with patch("ai_engine.api.copilot_endpoints.get_current_user", return_value=mock_current_user):
            response = await process_copilot_query(
                request,
                mock_background_tasks,
                mock_current_user,
                mock_copilot
            )

            assert isinstance(response, CopilotQueryResponse)
            assert response.response == "This is HTTP traffic on port 80."
            assert response.confidence == 0.95
            assert response.query_type == "protocol_analysis"
            assert response.processing_time > 0
            assert len(response.suggestions) == 2
            assert response.session_id is not None


@pytest.mark.asyncio
async def test_process_copilot_query_with_session_id(
    mock_copilot,
    mock_current_user,
    mock_background_tasks,
    sample_copilot_response
):
    """Test query with existing session ID."""
    request = CopilotQueryRequest(
        query="Follow-up question",
        session_id="existing-session-123"
    )

    mock_copilot.process_query.return_value = sample_copilot_response

    response = await process_copilot_query(
        request,
        mock_background_tasks,
        mock_current_user,
        mock_copilot
    )

    assert response.session_id == "existing-session-123"


@pytest.mark.asyncio
async def test_process_copilot_query_with_packet_data(
    mock_copilot,
    mock_current_user,
    mock_background_tasks,
    sample_copilot_response
):
    """Test query with packet data."""
    packet = b"GET /index.html HTTP/1.1\r\n"
    packet_data = base64.b64encode(packet).decode()

    request = CopilotQueryRequest(
        query="Analyze this packet",
        packet_data=packet_data
    )

    mock_copilot.process_query.return_value = sample_copilot_response

    response = await process_copilot_query(
        request,
        mock_background_tasks,
        mock_current_user,
        mock_copilot
    )

    # Verify packet_data was decoded and passed to copilot
    call_args = mock_copilot.process_query.call_args[0][0]
    assert call_args.packet_data == packet


@pytest.mark.asyncio
async def test_process_copilot_query_with_context(
    mock_copilot,
    mock_current_user,
    mock_background_tasks,
    sample_copilot_response
):
    """Test query with additional context."""
    request = CopilotQueryRequest(
        query="Is this suspicious?",
        context={"ip": "10.0.0.1", "port": 443}
    )

    mock_copilot.process_query.return_value = sample_copilot_response

    response = await process_copilot_query(
        request,
        mock_background_tasks,
        mock_current_user,
        mock_copilot
    )

    # Verify context was passed
    call_args = mock_copilot.process_query.call_args[0][0]
    assert call_args.context["ip"] == "10.0.0.1"


@pytest.mark.asyncio
async def test_process_copilot_query_background_task(
    mock_copilot,
    mock_current_user,
    mock_background_tasks,
    sample_copilot_response
):
    """Test background task is scheduled."""
    request = CopilotQueryRequest(query="Test query")

    mock_copilot.process_query.return_value = sample_copilot_response

    await process_copilot_query(
        request,
        mock_background_tasks,
        mock_current_user,
        mock_copilot
    )

    # Verify background task was added
    mock_background_tasks.add_task.assert_called_once()
    assert mock_background_tasks.add_task.call_args[0][0] == log_query_analytics


@pytest.mark.asyncio
async def test_process_copilot_query_cronos_exception(
    mock_copilot,
    mock_current_user,
    mock_background_tasks
):
    """Test handling of CronosAIException."""
    request = CopilotQueryRequest(query="Test")

    mock_copilot.process_query.side_effect = CronosAIException("Processing failed")

    with pytest.raises(HTTPException) as exc_info:
        await process_copilot_query(
            request,
            mock_background_tasks,
            mock_current_user,
            mock_copilot
        )

    assert exc_info.value.status_code == 400
    assert "Processing failed" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_process_copilot_query_unexpected_exception(
    mock_copilot,
    mock_current_user,
    mock_background_tasks
):
    """Test handling of unexpected exceptions."""
    request = CopilotQueryRequest(query="Test")

    mock_copilot.process_query.side_effect = Exception("Unexpected error")

    with pytest.raises(HTTPException) as exc_info:
        await process_copilot_query(
            request,
            mock_background_tasks,
            mock_current_user,
            mock_copilot
        )

    assert exc_info.value.status_code == 500
    assert "Internal server error" in str(exc_info.value.detail)


# Session Management Tests

@pytest.mark.asyncio
async def test_get_user_sessions_success(mock_copilot, mock_current_user):
    """Test getting user sessions."""
    mock_sessions = [
        {"session_id": "session-1", "created_at": "2024-01-01"},
        {"session_id": "session-2", "created_at": "2024-01-02"}
    ]

    mock_copilot.context_manager.get_user_sessions.return_value = mock_sessions

    response = await get_user_sessions(10, mock_current_user, mock_copilot)

    assert len(response) == 2
    assert response[0]["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_get_user_sessions_with_limit(mock_copilot, mock_current_user):
    """Test session retrieval with custom limit."""
    mock_copilot.context_manager.get_user_sessions.return_value = []

    await get_user_sessions(5, mock_current_user, mock_copilot)

    mock_copilot.context_manager.get_user_sessions.assert_called_with(
        mock_current_user["user_id"],
        5
    )


@pytest.mark.asyncio
async def test_get_user_sessions_error(mock_copilot, mock_current_user):
    """Test error handling in get sessions."""
    mock_copilot.context_manager.get_user_sessions.side_effect = Exception("DB error")

    with pytest.raises(HTTPException) as exc_info:
        await get_user_sessions(10, mock_current_user, mock_copilot)

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_get_session_summary_success(mock_copilot, mock_current_user):
    """Test getting session summary."""
    mock_summary = {
        "session_id": "session-123",
        "user_id": "user-123",
        "duration": "15 minutes",
        "total_turns": 10,
        "query_types": {"protocol_analysis": 5, "security": 5},
        "average_confidence": 0.85,
        "top_topics": ["HTTP", "TLS"],
        "last_activity": "2024-01-01T12:00:00"
    }

    mock_copilot.context_manager.get_session_summary.return_value = mock_summary

    response = await get_session_summary(
        "session-123",
        mock_current_user,
        mock_copilot
    )

    assert isinstance(response, SessionSummaryResponse)
    assert response.session_id == "session-123"
    assert response.total_turns == 10
    assert response.average_confidence == 0.85


@pytest.mark.asyncio
async def test_get_session_summary_not_found(mock_copilot, mock_current_user):
    """Test session summary when session not found."""
    mock_copilot.context_manager.get_session_summary.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await get_session_summary("nonexistent", mock_current_user, mock_copilot)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_session_summary_error(mock_copilot, mock_current_user):
    """Test error handling in session summary."""
    mock_copilot.context_manager.get_session_summary.side_effect = Exception("Error")

    with pytest.raises(HTTPException) as exc_info:
        await get_session_summary("session-123", mock_current_user, mock_copilot)

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_clear_session_success(mock_copilot, mock_current_user):
    """Test successful session clearing."""
    mock_copilot.context_manager.clear_session.return_value = True

    response = await clear_session("session-123", mock_current_user, mock_copilot)

    assert "successfully" in response["message"]
    assert "session-123" in response["message"]


@pytest.mark.asyncio
async def test_clear_session_not_found(mock_copilot, mock_current_user):
    """Test clearing non-existent session."""
    mock_copilot.context_manager.clear_session.return_value = False

    with pytest.raises(HTTPException) as exc_info:
        await clear_session("nonexistent", mock_current_user, mock_copilot)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_clear_session_error(mock_copilot, mock_current_user):
    """Test error handling in clear session."""
    mock_copilot.context_manager.clear_session.side_effect = Exception("Error")

    with pytest.raises(HTTPException) as exc_info:
        await clear_session("session-123", mock_current_user, mock_copilot)

    assert exc_info.value.status_code == 500


# Health Check Tests

@pytest.mark.asyncio
async def test_get_copilot_health_success(mock_copilot):
    """Test successful health check."""
    response = await get_copilot_health(mock_copilot)

    assert response["status"] == "healthy"
    assert "components" in response
    assert response["components"]["llm"] == "healthy"
    assert "timestamp" in response


@pytest.mark.asyncio
async def test_get_copilot_health_error():
    """Test health check error handling."""
    mock_copilot = Mock()
    mock_copilot.get_health_status.side_effect = Exception("Health check failed")

    response = await get_copilot_health(mock_copilot)

    assert response["status"] == "unhealthy"
    assert "error" in response


# WebSocket Message Tests

def test_websocket_message_creation():
    """Test WebSocket message creation."""
    message = WebSocketMessage(
        type="query",
        data={"query": "Test"},
        correlation_id="corr-123"
    )

    assert message.type == "query"
    assert message.data["query"] == "Test"
    assert message.correlation_id == "corr-123"


def test_websocket_message_json():
    """Test WebSocket message JSON serialization."""
    message = WebSocketMessage(
        type="response",
        data={"response": "Answer"}
    )

    json_str = message.json()
    assert isinstance(json_str, str)

    # Parse back
    parsed = json.loads(json_str)
    assert parsed["type"] == "response"


# Background Task Tests

@pytest.mark.asyncio
async def test_log_query_analytics_success():
    """Test query analytics logging."""
    # Should not raise exception
    await log_query_analytics(
        "user-123",
        "What is this protocol?",
        "protocol_analysis",
        0.95
    )


@pytest.mark.asyncio
async def test_log_query_analytics_error():
    """Test analytics logging error handling."""
    # Should handle exceptions gracefully
    with patch("ai_engine.api.copilot_endpoints.logger") as mock_logger:
        await log_query_analytics(None, None, None, None)

        # Should have logged an error
        assert mock_logger.error.called


# Dependency Tests

@pytest.mark.asyncio
async def test_get_copilot_singleton():
    """Test copilot singleton pattern."""
    with patch("ai_engine.api.copilot_endpoints.create_protocol_copilot") as mock_create:
        mock_copilot = AsyncMock()
        mock_create.return_value = mock_copilot

        # First call creates instance
        copilot1 = await get_copilot()

        # Second call returns same instance
        copilot2 = await get_copilot()

        assert copilot1 is copilot2
        mock_create.assert_called_once()


# Integration Tests

@pytest.mark.asyncio
async def test_full_query_workflow(
    mock_copilot,
    mock_current_user,
    sample_copilot_response
):
    """Test complete query workflow."""
    # Setup
    mock_copilot.process_query.return_value = sample_copilot_response
    mock_copilot.context_manager.get_user_sessions.return_value = []

    # 1. Process query
    request = CopilotQueryRequest(query="Analyze this traffic")
    background_tasks = Mock(spec=BackgroundTasks)

    response = await process_copilot_query(
        request,
        background_tasks,
        mock_current_user,
        mock_copilot
    )

    session_id = response.session_id

    # 2. Get sessions
    sessions = await get_user_sessions(10, mock_current_user, mock_copilot)

    # 3. Get session summary
    mock_copilot.context_manager.get_session_summary.return_value = {
        "session_id": session_id,
        "user_id": mock_current_user["user_id"],
        "duration": "5 minutes",
        "total_turns": 1,
        "query_types": {"protocol_analysis": 1},
        "average_confidence": 0.95,
        "top_topics": ["HTTP"],
        "last_activity": datetime.now().isoformat()
    }

    summary = await get_session_summary(session_id, mock_current_user, mock_copilot)

    assert summary.session_id == session_id

    # 4. Clear session
    mock_copilot.context_manager.clear_session.return_value = True

    clear_response = await clear_session(session_id, mock_current_user, mock_copilot)

    assert "successfully" in clear_response["message"]


@pytest.mark.asyncio
async def test_multiple_queries_same_session(
    mock_copilot,
    mock_current_user,
    sample_copilot_response
):
    """Test multiple queries in same session."""
    mock_copilot.process_query.return_value = sample_copilot_response

    session_id = "test-session-123"

    # Send multiple queries with same session ID
    for i in range(3):
        request = CopilotQueryRequest(
            query=f"Query {i}",
            session_id=session_id
        )

        response = await process_copilot_query(
            request,
            Mock(spec=BackgroundTasks),
            mock_current_user,
            mock_copilot
        )

        assert response.session_id == session_id


@pytest.mark.asyncio
async def test_concurrent_queries(
    mock_copilot,
    mock_current_user,
    sample_copilot_response
):
    """Test handling concurrent queries."""
    mock_copilot.process_query.return_value = sample_copilot_response

    # Create multiple queries
    queries = [
        CopilotQueryRequest(query=f"Query {i}")
        for i in range(5)
    ]

    # Process concurrently
    results = await asyncio.gather(*[
        process_copilot_query(
            query,
            Mock(spec=BackgroundTasks),
            mock_current_user,
            mock_copilot
        )
        for query in queries
    ])

    assert len(results) == 5
    assert all(isinstance(r, CopilotQueryResponse) for r in results)


# Router Configuration Tests

def test_router_configuration():
    """Test router is properly configured."""
    assert router.prefix == "/api/v1/copilot"
    assert "Protocol Intelligence Copilot" in router.tags
    assert len(router.routes) > 0


# Metrics Tests

def test_metrics_counters_exist():
    """Test that metrics are defined."""
    from ai_engine.api.copilot_endpoints import (
        COPILOT_API_REQUESTS,
        COPILOT_API_DURATION,
        WEBSOCKET_CONNECTIONS
    )

    assert COPILOT_API_REQUESTS is not None
    assert COPILOT_API_DURATION is not None
    assert WEBSOCKET_CONNECTIONS is not None


# Edge Cases

@pytest.mark.asyncio
async def test_process_query_empty_suggestions(
    mock_copilot,
    mock_current_user,
    mock_background_tasks
):
    """Test handling of None suggestions."""
    response_data = CopilotResponse(
        response="Answer",
        confidence=0.9,
        query_type="test",
        processing_time=0.1,
        suggestions=None,  # None instead of list
        source_data=[],
        visualizations=None,
        metadata=None
    )

    mock_copilot.process_query.return_value = response_data

    request = CopilotQueryRequest(query="Test")

    response = await process_copilot_query(
        request,
        mock_background_tasks,
        mock_current_user,
        mock_copilot
    )

    # Should convert None to empty list
    assert response.suggestions == []
    assert response.metadata == {}


@pytest.mark.asyncio
async def test_process_query_large_context(
    mock_copilot,
    mock_current_user,
    mock_background_tasks,
    sample_copilot_response
):
    """Test query with large context dictionary."""
    large_context = {f"key_{i}": f"value_{i}" for i in range(100)}

    request = CopilotQueryRequest(
        query="Test",
        context=large_context
    )

    mock_copilot.process_query.return_value = sample_copilot_response

    # Should handle large context
    response = await process_copilot_query(
        request,
        mock_background_tasks,
        mock_current_user,
        mock_copilot
    )

    assert response is not None


@pytest.mark.asyncio
async def test_session_summary_empty_query_types(mock_copilot, mock_current_user):
    """Test session summary with empty query types."""
    mock_summary = {
        "session_id": "session-123",
        "user_id": "user-123",
        "duration": "0 minutes",
        "total_turns": 0,
        "query_types": {},  # Empty
        "average_confidence": 0.0,
        "top_topics": [],
        "last_activity": datetime.now().isoformat()
    }

    mock_copilot.context_manager.get_session_summary.return_value = mock_summary

    response = await get_session_summary("session-123", mock_current_user, mock_copilot)

    assert response.total_turns == 0
    assert len(response.query_types) == 0
