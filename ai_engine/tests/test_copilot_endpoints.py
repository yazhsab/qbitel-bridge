"""
Tests for QBITEL Protocol Intelligence Copilot API Endpoints
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime
import uuid
import base64
from typing import Dict, Any

from fastapi import HTTPException, WebSocket
from fastapi.testclient import TestClient

from ai_engine.api.copilot_endpoints import (
    router,
    get_copilot,
    process_copilot_query,
    get_user_sessions,
    get_session_summary,
    clear_session,
    get_copilot_health,
    log_query_analytics,
    CopilotQueryRequest,
    CopilotQueryResponse,
    SessionSummaryResponse,
    WebSocketMessage,
)


@pytest.fixture
def mock_copilot():
    """Create a mock copilot instance."""
    copilot = AsyncMock()

    # Mock process_query
    mock_response = Mock()
    mock_response.response = "This is a test response"
    mock_response.confidence = 0.85
    mock_response.query_type = "protocol_analysis"
    mock_response.processing_time = 0.15
    mock_response.suggestions = ["Try analyzing the headers", "Check for anomalies"]
    mock_response.source_data = [{"source": "test"}]
    mock_response.visualizations = [{"type": "chart"}]
    mock_response.metadata = {"test": "data"}

    copilot.process_query = AsyncMock(return_value=mock_response)

    # Mock context_manager
    copilot.context_manager = AsyncMock()
    copilot.context_manager.get_user_sessions = AsyncMock(
        return_value=[
            {
                "session_id": "session1",
                "user_id": "user1",
                "created_at": "2024-01-01T00:00:00",
            }
        ]
    )
    copilot.context_manager.get_session_summary = AsyncMock(
        return_value={
            "session_id": "session1",
            "user_id": "user1",
            "duration": "1h 30m",
            "total_turns": 10,
            "query_types": {"protocol_analysis": 5, "security": 5},
            "average_confidence": 0.85,
            "top_topics": ["HTTP", "Security"],
            "last_activity": "2024-01-01T01:30:00",
        }
    )
    copilot.context_manager.clear_session = AsyncMock(return_value=True)

    # Mock get_health_status
    copilot.get_health_status = Mock(
        return_value={
            "llm": "healthy",
            "context_manager": "healthy",
            "knowledge_base": "healthy",
        }
    )

    return copilot


@pytest.fixture
def mock_current_user():
    """Create a mock current user."""
    return {
        "user_id": "test_user_123",
        "username": "testuser",
        "role": "analyst",
        "permissions": ["protocol_discovery", "copilot_access"],
    }


@pytest.fixture
def mock_background_tasks():
    """Create a mock background tasks."""
    tasks = Mock()
    tasks.add_task = Mock()
    return tasks


class TestCopilotQueryRequest:
    """Tests for CopilotQueryRequest model."""

    def test_valid_request(self):
        """Test valid request creation."""
        request = CopilotQueryRequest(
            query="What protocol is this?",
            session_id="session123",
            context={"source": "test"},
        )

        assert request.query == "What protocol is this?"
        assert request.session_id == "session123"
        assert request.context == {"source": "test"}

    def test_request_with_packet_data(self):
        """Test request with valid packet data."""
        packet_data = base64.b64encode(b"test packet").decode()
        request = CopilotQueryRequest(
            query="Analyze this packet",
            packet_data=packet_data,
        )

        assert request.packet_data == packet_data

    def test_request_with_invalid_packet_data(self):
        """Test request with invalid packet data."""
        with pytest.raises(ValueError, match="packet_data must be valid base64"):
            CopilotQueryRequest(
                query="Analyze this packet",
                packet_data="invalid base64!@#",
            )

    def test_request_min_length_validation(self):
        """Test query minimum length validation."""
        with pytest.raises(ValueError):
            CopilotQueryRequest(query="")

    def test_request_max_length_validation(self):
        """Test query maximum length validation."""
        with pytest.raises(ValueError):
            CopilotQueryRequest(query="x" * 2001)


class TestProcessCopilotQuery:
    """Tests for process_copilot_query endpoint."""

    @pytest.mark.asyncio
    async def test_process_query_success(
        self, mock_copilot, mock_current_user, mock_background_tasks
    ):
        """Test successful query processing."""
        request = CopilotQueryRequest(
            query="What protocol is this?",
            session_id="session123",
        )

        with patch(
            "ai_engine.api.copilot_endpoints.get_copilot", return_value=mock_copilot
        ):
            response = await process_copilot_query(
                request=request,
                background_tasks=mock_background_tasks,
                current_user=mock_current_user,
                copilot=mock_copilot,
            )

        assert response.response == "This is a test response"
        assert response.confidence == 0.85
        assert response.query_type == "protocol_analysis"
        assert response.session_id == "session123"
        assert len(response.suggestions) == 2
        mock_background_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_generates_session_id(
        self, mock_copilot, mock_current_user, mock_background_tasks
    ):
        """Test query processing generates session ID if not provided."""
        request = CopilotQueryRequest(query="What protocol is this?")

        response = await process_copilot_query(
            request=request,
            background_tasks=mock_background_tasks,
            current_user=mock_current_user,
            copilot=mock_copilot,
        )

        assert response.session_id is not None
        assert len(response.session_id) > 0

    @pytest.mark.asyncio
    async def test_process_query_with_packet_data(
        self, mock_copilot, mock_current_user, mock_background_tasks
    ):
        """Test query processing with packet data."""
        packet_data = base64.b64encode(b"test packet").decode()
        request = CopilotQueryRequest(
            query="Analyze this packet",
            packet_data=packet_data,
        )

        response = await process_copilot_query(
            request=request,
            background_tasks=mock_background_tasks,
            current_user=mock_current_user,
            copilot=mock_copilot,
        )

        assert response.response == "This is a test response"
        # Verify packet data was decoded and passed
        call_args = mock_copilot.process_query.call_args
        assert call_args[0][0].packet_data == b"test packet"

    @pytest.mark.asyncio
    async def test_process_query_with_context(
        self, mock_copilot, mock_current_user, mock_background_tasks
    ):
        """Test query processing with context."""
        request = CopilotQueryRequest(
            query="What protocol is this?",
            context={"source": "network_capture", "protocol_hint": "http"},
        )

        response = await process_copilot_query(
            request=request,
            background_tasks=mock_background_tasks,
            current_user=mock_current_user,
            copilot=mock_copilot,
        )

        assert response.response == "This is a test response"
        call_args = mock_copilot.process_query.call_args
        assert call_args[0][0].context == request.context

    @pytest.mark.asyncio
    async def test_process_query_qbitel_exception(
        self, mock_copilot, mock_current_user, mock_background_tasks
    ):
        """Test query processing with QbitelAIException."""
        from ai_engine.core.exceptions import QbitelAIException

        request = CopilotQueryRequest(query="What protocol is this?")
        mock_copilot.process_query.side_effect = QbitelAIException("Processing failed")

        with pytest.raises(HTTPException) as exc_info:
            await process_copilot_query(
                request=request,
                background_tasks=mock_background_tasks,
                current_user=mock_current_user,
                copilot=mock_copilot,
            )

        assert exc_info.value.status_code == 400
        assert "Processing failed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_process_query_unexpected_exception(
        self, mock_copilot, mock_current_user, mock_background_tasks
    ):
        """Test query processing with unexpected exception."""
        request = CopilotQueryRequest(query="What protocol is this?")
        mock_copilot.process_query.side_effect = Exception("Unexpected error")

        with pytest.raises(HTTPException) as exc_info:
            await process_copilot_query(
                request=request,
                background_tasks=mock_background_tasks,
                current_user=mock_current_user,
                copilot=mock_copilot,
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in str(exc_info.value.detail)


class TestGetUserSessions:
    """Tests for get_user_sessions endpoint."""

    @pytest.mark.asyncio
    async def test_get_sessions_success(self, mock_copilot, mock_current_user):
        """Test successful session retrieval."""
        sessions = await get_user_sessions(
            limit=10,
            current_user=mock_current_user,
            copilot=mock_copilot,
        )

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "session1"
        mock_copilot.context_manager.get_user_sessions.assert_called_once_with(
            "test_user_123", 10
        )

    @pytest.mark.asyncio
    async def test_get_sessions_custom_limit(self, mock_copilot, mock_current_user):
        """Test session retrieval with custom limit."""
        sessions = await get_user_sessions(
            limit=5,
            current_user=mock_current_user,
            copilot=mock_copilot,
        )

        mock_copilot.context_manager.get_user_sessions.assert_called_once_with(
            "test_user_123", 5
        )

    @pytest.mark.asyncio
    async def test_get_sessions_exception(self, mock_copilot, mock_current_user):
        """Test session retrieval with exception."""
        mock_copilot.context_manager.get_user_sessions.side_effect = Exception(
            "DB error"
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_user_sessions(
                limit=10,
                current_user=mock_current_user,
                copilot=mock_copilot,
            )

        assert exc_info.value.status_code == 500
        assert "Failed to retrieve sessions" in str(exc_info.value.detail)


class TestGetSessionSummary:
    """Tests for get_session_summary endpoint."""

    @pytest.mark.asyncio
    async def test_get_summary_success(self, mock_copilot, mock_current_user):
        """Test successful session summary retrieval."""
        summary = await get_session_summary(
            session_id="session1",
            current_user=mock_current_user,
            copilot=mock_copilot,
        )

        assert summary.session_id == "session1"
        assert summary.user_id == "user1"
        assert summary.total_turns == 10
        assert summary.average_confidence == 0.85

    @pytest.mark.asyncio
    async def test_get_summary_not_found(self, mock_copilot, mock_current_user):
        """Test session summary not found."""
        mock_copilot.context_manager.get_session_summary.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_session_summary(
                session_id="nonexistent",
                current_user=mock_current_user,
                copilot=mock_copilot,
            )

        assert exc_info.value.status_code == 404
        assert "Session not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_summary_exception(self, mock_copilot, mock_current_user):
        """Test session summary with exception."""
        mock_copilot.context_manager.get_session_summary.side_effect = Exception(
            "DB error"
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_session_summary(
                session_id="session1",
                current_user=mock_current_user,
                copilot=mock_copilot,
            )

        assert exc_info.value.status_code == 500
        assert "Failed to retrieve session summary" in str(exc_info.value.detail)


class TestClearSession:
    """Tests for clear_session endpoint."""

    @pytest.mark.asyncio
    async def test_clear_session_success(self, mock_copilot, mock_current_user):
        """Test successful session clearing."""
        result = await clear_session(
            session_id="session1",
            current_user=mock_current_user,
            copilot=mock_copilot,
        )

        assert result["message"] == "Session session1 cleared successfully"
        mock_copilot.context_manager.clear_session.assert_called_once_with(
            "test_user_123", "session1"
        )

    @pytest.mark.asyncio
    async def test_clear_session_not_found(self, mock_copilot, mock_current_user):
        """Test clearing non-existent session."""
        mock_copilot.context_manager.clear_session.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            await clear_session(
                session_id="nonexistent",
                current_user=mock_current_user,
                copilot=mock_copilot,
            )

        assert exc_info.value.status_code == 404
        assert "Session not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_clear_session_exception(self, mock_copilot, mock_current_user):
        """Test session clearing with exception."""
        mock_copilot.context_manager.clear_session.side_effect = Exception("DB error")

        with pytest.raises(HTTPException) as exc_info:
            await clear_session(
                session_id="session1",
                current_user=mock_current_user,
                copilot=mock_copilot,
            )

        assert exc_info.value.status_code == 500
        assert "Failed to clear session" in str(exc_info.value.detail)


class TestGetCopilotHealth:
    """Tests for get_copilot_health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_copilot):
        """Test successful health check."""
        result = await get_copilot_health(copilot=mock_copilot)

        assert result["status"] == "healthy"
        assert "components" in result
        assert result["components"]["llm"] == "healthy"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_health_check_exception(self, mock_copilot):
        """Test health check with exception."""
        mock_copilot.get_health_status.side_effect = Exception("Health check failed")

        result = await get_copilot_health(copilot=mock_copilot)

        assert result["status"] == "unhealthy"
        assert "error" in result
        assert "Health check failed" in result["error"]


class TestWebSocketMessage:
    """Tests for WebSocketMessage model."""

    def test_valid_message(self):
        """Test valid message creation."""
        message = WebSocketMessage(
            type="query",
            data={"query": "test"},
            timestamp="2024-01-01T00:00:00",
            correlation_id="corr123",
        )

        assert message.type == "query"
        assert message.data == {"query": "test"}
        assert message.correlation_id == "corr123"

    def test_message_types(self):
        """Test different message types."""
        types = ["query", "response", "error", "ping", "pong"]

        for msg_type in types:
            message = WebSocketMessage(type=msg_type, data={})
            assert message.type == msg_type


class TestLogQueryAnalytics:
    """Tests for log_query_analytics background task."""

    @pytest.mark.asyncio
    async def test_log_analytics_success(self):
        """Test successful analytics logging."""
        # Should not raise any exceptions
        await log_query_analytics(
            user_id="user123",
            query="What protocol is this?",
            query_type="protocol_analysis",
            confidence=0.85,
        )

    @pytest.mark.asyncio
    async def test_log_analytics_exception(self):
        """Test analytics logging with exception (should not propagate)."""
        # Should handle exceptions gracefully
        with patch("ai_engine.api.copilot_endpoints.logger") as mock_logger:
            # Force an error by passing invalid data
            await log_query_analytics(
                user_id=None,  # This might cause an error
                query="test",
                query_type="test",
                confidence=0.5,
            )

            # Should not raise exception, just log it
            # The function should complete without error


class TestWebSocketEndpoint:
    """Tests for WebSocket endpoint."""

    @pytest.mark.asyncio
    async def test_websocket_connection_invalid_token(self, mock_copilot):
        """Test WebSocket connection with invalid token."""
        websocket = AsyncMock(spec=WebSocket)
        websocket.close = AsyncMock()

        with patch(
            "ai_engine.api.copilot_endpoints.verify_token",
            side_effect=Exception("Invalid token"),
        ):
            from ai_engine.api.copilot_endpoints import websocket_copilot_chat

            await websocket_copilot_chat(
                websocket=websocket,
                token="invalid_token",
                copilot=mock_copilot,
            )

            websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_welcome_message(self, mock_copilot):
        """Test WebSocket sends welcome message."""
        websocket = AsyncMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=Exception("Stop"))

        with patch(
            "ai_engine.api.copilot_endpoints.verify_token",
            return_value={"user_id": "user123"},
        ):
            from ai_engine.api.copilot_endpoints import websocket_copilot_chat

            try:
                await websocket_copilot_chat(
                    websocket=websocket,
                    token="valid_token",
                    copilot=mock_copilot,
                )
            except:
                pass

            websocket.accept.assert_called_once()
            # Should have sent welcome message
            assert websocket.send_text.call_count >= 1


class TestGetCopilot:
    """Tests for get_copilot dependency."""

    @pytest.mark.asyncio
    async def test_get_copilot_creates_instance(self):
        """Test get_copilot creates instance if not exists."""
        # Reset global instance
        import ai_engine.api.copilot_endpoints as copilot_module

        copilot_module._copilot_instance = None

        with patch(
            "ai_engine.api.copilot_endpoints.create_protocol_copilot"
        ) as mock_create:
            mock_copilot = AsyncMock()
            mock_create.return_value = mock_copilot

            result = await get_copilot()

            assert result == mock_copilot
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_copilot_reuses_instance(self):
        """Test get_copilot reuses existing instance."""
        import ai_engine.api.copilot_endpoints as copilot_module

        mock_copilot = AsyncMock()
        copilot_module._copilot_instance = mock_copilot

        result = await get_copilot()

        assert result == mock_copilot


class TestCopilotQueryResponse:
    """Tests for CopilotQueryResponse model."""

    def test_valid_response(self):
        """Test valid response creation."""
        response = CopilotQueryResponse(
            response="Test response",
            confidence=0.85,
            query_type="protocol_analysis",
            processing_time=0.15,
            session_id="session123",
            suggestions=["suggestion1", "suggestion2"],
            source_data=[{"source": "test"}],
            visualizations=[{"type": "chart"}],
            metadata={"key": "value"},
        )

        assert response.response == "Test response"
        assert response.confidence == 0.85
        assert response.query_type == "protocol_analysis"
        assert len(response.suggestions) == 2

    def test_response_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        response = CopilotQueryResponse(
            response="Test",
            confidence=0.5,
            query_type="test",
            processing_time=0.1,
            session_id="session123",
        )
        assert response.confidence == 0.5

        # Invalid confidence (should be validated by Pydantic)
        with pytest.raises(ValueError):
            CopilotQueryResponse(
                response="Test",
                confidence=1.5,  # > 1.0
                query_type="test",
                processing_time=0.1,
                session_id="session123",
            )


class TestSessionSummaryResponse:
    """Tests for SessionSummaryResponse model."""

    def test_valid_summary(self):
        """Test valid summary creation."""
        summary = SessionSummaryResponse(
            session_id="session123",
            user_id="user123",
            duration="1h 30m",
            total_turns=10,
            query_types={"protocol_analysis": 5, "security": 5},
            average_confidence=0.85,
            top_topics=["HTTP", "Security"],
            last_activity="2024-01-01T01:30:00",
        )

        assert summary.session_id == "session123"
        assert summary.total_turns == 10
        assert summary.average_confidence == 0.85
        assert len(summary.top_topics) == 2
