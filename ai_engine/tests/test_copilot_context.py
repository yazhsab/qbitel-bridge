"""
Tests for Copilot Conversation Context Manager.
Covers ConversationContextManager, session management, and context persistence.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json

from ai_engine.copilot.context_manager import (
    ConversationContextManager,
    ConversationSession,
    ConversationTurn,
)


class TestConversationTurn:
    """Test ConversationTurn dataclass."""

    def test_create_turn(self):
        """Test creating a conversation turn."""
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_query="What is protocol X?",
            assistant_response="Protocol X is...",
            query_type="explanation",
            confidence=0.95,
        )

        assert turn.user_query == "What is protocol X?"
        assert turn.assistant_response == "Protocol X is..."
        assert turn.query_type == "explanation"
        assert turn.confidence == 0.95

    def test_turn_with_context(self):
        """Test turn with additional context."""
        context = {"protocol": "HTTP", "version": "1.1"}
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_query="Test query",
            assistant_response="Test response",
            query_type="general",
            confidence=0.8,
            context_used=context,
        )

        assert turn.context_used == context
        assert turn.context_used["protocol"] == "HTTP"


class TestConversationSession:
    """Test ConversationSession dataclass."""

    def test_create_session(self):
        """Test creating a conversation session."""
        now = datetime.now()
        session = ConversationSession(
            session_id="sess_123",
            user_id="user_456",
            started_at=now,
            last_activity=now,
            turns=[],
            persistent_context={},
        )

        assert session.session_id == "sess_123"
        assert session.user_id == "user_456"
        assert session.started_at == now
        assert len(session.turns) == 0

    def test_session_with_turns(self):
        """Test session with conversation turns."""
        now = datetime.now()
        turn = ConversationTurn(
            timestamp=now,
            user_query="Query 1",
            assistant_response="Response 1",
            query_type="general",
            confidence=0.9,
        )

        session = ConversationSession(
            session_id="sess_123",
            user_id="user_456",
            started_at=now,
            last_activity=now,
            turns=[turn],
            persistent_context={"context_key": "context_value"},
        )

        assert len(session.turns) == 1
        assert session.turns[0].user_query == "Query 1"
        assert session.persistent_context["context_key"] == "context_value"

    def test_session_with_metadata(self):
        """Test session with metadata."""
        session = ConversationSession(
            session_id="sess_123",
            user_id="user_456",
            started_at=datetime.now(),
            last_activity=datetime.now(),
            turns=[],
            persistent_context={},
            metadata={"source": "web", "platform": "desktop"},
        )

        assert session.metadata["source"] == "web"
        assert session.metadata["platform"] == "desktop"


class TestContextManagerInitialization:
    """Test ContextManager initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        manager = ConversationContextManager()

        assert manager.redis_config["host"] == "localhost"
        assert manager.redis_config["port"] == 6379
        assert manager.max_turns_per_session == 50
        assert manager.session_timeout == timedelta(hours=2)

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {"host": "redis.example.com", "port": 6380, "db": 1}
        manager = ConversationContextManager(redis_config=config)

        assert manager.redis_config["host"] == "redis.example.com"
        assert manager.redis_config["port"] == 6380
        assert manager.redis_config["db"] == 1

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        manager = ConversationContextManager()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        with patch("redis.asyncio.Redis", return_value=mock_redis):
            with patch("asyncio.create_task") as mock_task:
                await manager.initialize()

                mock_redis.ping.assert_called_once()
                mock_task.assert_called_once()
                assert manager.redis_client is not None

    @pytest.mark.asyncio
    async def test_initialize_redis_failure(self):
        """Test initialization with Redis connection failure."""
        manager = ConversationContextManager()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("Connection failed"))

        with patch("redis.asyncio.Redis", return_value=mock_redis):
            with pytest.raises(Exception, match="Connection failed"):
                await manager.initialize()


class TestContextRetrieval:
    """Test context retrieval methods."""

    @pytest_asyncio.fixture
    async def manager(self):
        """Create context manager with mocked Redis."""
        mgr = ConversationContextManager()
        mgr.redis_client = AsyncMock()
        mgr.redis_client.ping = AsyncMock()
        return mgr

    @pytest.mark.asyncio
    async def test_get_context_new_session(self, manager):
        """Test getting context for new session."""
        manager.redis_client.hgetall = AsyncMock(return_value={})

        with patch.object(manager, "_save_session_to_redis", new_callable=AsyncMock):
            context = await manager.get_context("user_123", "sess_456")

            assert "session_id" in context
            assert "turns" in context
            assert context["session_id"] == "sess_456"
            assert len(context["turns"]) == 0

    @pytest.mark.asyncio
    async def test_get_context_cached_session(self, manager):
        """Test getting context from cached session."""
        # Create and cache a session
        session = ConversationSession(
            session_id="sess_456",
            user_id="user_123",
            started_at=datetime.now(),
            last_activity=datetime.now(),
            turns=[],
            persistent_context={"cached": True},
        )
        manager.active_sessions["sess_456"] = session

        context = await manager.get_context("user_123", "sess_456")

        assert context["session_id"] == "sess_456"
        assert context["persistent_context"]["cached"] is True

    @pytest.mark.asyncio
    async def test_get_context_from_redis(self, manager):
        """Test loading context from Redis."""
        redis_data = {
            "session_id": "sess_789",
            "user_id": "user_123",
            "started_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "turns": json.dumps([]),
            "persistent_context": json.dumps({"from_redis": True}),
            "metadata": json.dumps({}),
        }

        manager.redis_client.hgetall = AsyncMock(return_value=redis_data)

        with patch.object(manager, "_build_context_from_session") as mock_build:
            mock_build.return_value = {"session_id": "sess_789"}
            context = await manager.get_context("user_123", "sess_789")

            # Session should be cached after load
            assert "sess_789" in manager.active_sessions


class TestContextUpdate:
    """Test context update methods."""

    @pytest_asyncio.fixture
    async def manager(self):
        """Create context manager."""
        mgr = ConversationContextManager()
        mgr.redis_client = AsyncMock()
        return mgr

    @pytest.mark.asyncio
    async def test_update_context_new_turn(self, manager):
        """Test adding a new conversation turn."""
        # Create initial session
        session = ConversationSession(
            session_id="sess_123",
            user_id="user_456",
            started_at=datetime.now(),
            last_activity=datetime.now(),
            turns=[],
            persistent_context={},
        )
        manager.active_sessions["sess_123"] = session

        with patch.object(manager, "_save_session_to_redis", new_callable=AsyncMock):
            with patch.object(manager, "_update_persistent_context", new_callable=AsyncMock):
                await manager.update_context(
                    user_id="user_456",
                    session_id="sess_123",
                    user_query="What is TCP?",
                    assistant_response="TCP is Transmission Control Protocol",
                    query_type="explanation",
                    confidence=0.95,
                )

        # Verify turn was added
        assert len(session.turns) == 1
        assert session.turns[0].user_query == "What is TCP?"
        assert session.turns[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_update_context_max_turns_limit(self, manager):
        """Test maximum turns limit enforcement."""
        manager.max_turns_per_session = 5

        # Create session with max turns
        turns = [
            ConversationTurn(
                timestamp=datetime.now(),
                user_query=f"Query {i}",
                assistant_response=f"Response {i}",
                query_type="general",
                confidence=0.8,
            )
            for i in range(5)
        ]

        session = ConversationSession(
            session_id="sess_123",
            user_id="user_456",
            started_at=datetime.now(),
            last_activity=datetime.now(),
            turns=turns,
            persistent_context={},
        )
        manager.active_sessions["sess_123"] = session

        with patch.object(manager, "_save_session_to_redis", new_callable=AsyncMock):
            with patch.object(manager, "_update_persistent_context", new_callable=AsyncMock):
                await manager.update_context(
                    user_id="user_456",
                    session_id="sess_123",
                    user_query="New query",
                    assistant_response="New response",
                    query_type="general",
                    confidence=0.9,
                )

        # Should still have max_turns_per_session
        assert len(session.turns) == 5
        # Oldest turn should be removed
        assert session.turns[0].user_query != "Query 0"
        assert session.turns[-1].user_query == "New query"

    @pytest.mark.asyncio
    async def test_update_context_with_additional_context(self, manager):
        """Test update with additional context."""
        session = ConversationSession(
            session_id="sess_123",
            user_id="user_456",
            started_at=datetime.now(),
            last_activity=datetime.now(),
            turns=[],
            persistent_context={},
        )
        manager.active_sessions["sess_123"] = session

        additional = {"protocol": "HTTP", "version": "1.1"}

        with patch.object(manager, "_save_session_to_redis", new_callable=AsyncMock):
            with patch.object(manager, "_update_persistent_context", new_callable=AsyncMock):
                await manager.update_context(
                    user_id="user_456",
                    session_id="sess_123",
                    user_query="Test",
                    assistant_response="Response",
                    additional_context=additional,
                )

        assert session.turns[0].context_used == additional


class TestSessionManagement:
    """Test session management methods."""

    @pytest_asyncio.fixture
    async def manager(self):
        """Create context manager."""
        mgr = ConversationContextManager()
        mgr.redis_client = AsyncMock()
        return mgr

    @pytest.mark.asyncio
    async def test_get_user_sessions(self, manager):
        """Test retrieving user sessions."""
        # Mock Redis keys and data
        manager.redis_client.keys = AsyncMock(
            return_value=[
                "context:session:user_123:sess_1",
                "context:session:user_123:sess_2",
            ]
        )

        session_data = {
            "session_id": "sess_1",
            "started_at": "2025-01-01T10:00:00",
            "last_activity": "2025-01-01T11:00:00",
            "turn_count": "5",
            "metadata": json.dumps({"source": "web"}),
        }

        manager.redis_client.hgetall = AsyncMock(return_value=session_data)

        sessions = await manager.get_user_sessions("user_123", limit=10)

        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "sess_1"
        assert sessions[0]["turn_count"] == 5

    @pytest.mark.asyncio
    async def test_clear_session(self, manager):
        """Test clearing a session."""
        # Add session to cache
        session = ConversationSession(
            session_id="sess_123",
            user_id="user_456",
            started_at=datetime.now(),
            last_activity=datetime.now(),
            turns=[],
            persistent_context={},
        )
        manager.active_sessions["sess_123"] = session

        manager.redis_client.delete = AsyncMock(return_value=1)

        result = await manager.clear_session("user_456", "sess_123")

        assert result is True
        assert "sess_123" not in manager.active_sessions
        manager.redis_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_session_not_found(self, manager):
        """Test clearing non-existent session."""
        manager.redis_client.delete = AsyncMock(return_value=0)

        result = await manager.clear_session("user_456", "nonexistent")

        assert result is True  # Still returns True even if not found


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest_asyncio.fixture
    async def manager(self):
        """Create context manager."""
        mgr = ConversationContextManager()
        mgr.redis_client = AsyncMock()
        return mgr

    @pytest.mark.asyncio
    async def test_get_context_redis_error(self, manager):
        """Test get_context handles Redis errors gracefully."""
        manager.redis_client.hgetall = AsyncMock(side_effect=Exception("Redis error"))

        context = await manager.get_context("user_123", "sess_456")

        # Should return empty context on error
        assert context == {}

    @pytest.mark.asyncio
    async def test_update_context_error_handling(self, manager):
        """Test update_context error handling."""
        manager.redis_client.hset = AsyncMock(side_effect=Exception("Redis error"))

        # Should not raise exception
        await manager.update_context(
            user_id="user_456",
            session_id="sess_123",
            user_query="Test",
            assistant_response="Response",
        )

    @pytest.mark.asyncio
    async def test_get_user_sessions_error(self, manager):
        """Test get_user_sessions error handling."""
        manager.redis_client.keys = AsyncMock(side_effect=Exception("Redis error"))

        sessions = await manager.get_user_sessions("user_123")

        # Should return empty list on error
        assert sessions == []

    @pytest.mark.asyncio
    async def test_clear_session_error(self, manager):
        """Test clear_session error handling."""
        manager.redis_client.delete = AsyncMock(side_effect=Exception("Redis error"))

        result = await manager.clear_session("user_456", "sess_123")

        # Should return False on error
        assert result is False


class TestConcurrency:
    """Test concurrent operations."""

    @pytest_asyncio.fixture
    async def manager(self):
        """Create context manager."""
        mgr = ConversationContextManager()
        mgr.redis_client = AsyncMock()
        mgr.redis_client.hgetall = AsyncMock(return_value={})
        mgr.redis_client.hset = AsyncMock()
        mgr.redis_client.expire = AsyncMock()
        return mgr

    @pytest.mark.asyncio
    async def test_concurrent_context_updates(self, manager):
        """Test concurrent context updates."""
        session = ConversationSession(
            session_id="sess_123",
            user_id="user_456",
            started_at=datetime.now(),
            last_activity=datetime.now(),
            turns=[],
            persistent_context={},
        )
        manager.active_sessions["sess_123"] = session

        with patch.object(manager, "_save_session_to_redis", new_callable=AsyncMock):
            with patch.object(manager, "_update_persistent_context", new_callable=AsyncMock):
                # Run multiple updates concurrently
                tasks = [
                    manager.update_context(
                        user_id="user_456",
                        session_id="sess_123",
                        user_query=f"Query {i}",
                        assistant_response=f"Response {i}",
                    )
                    for i in range(10)
                ]

                await asyncio.gather(*tasks)

        # All updates should be recorded
        assert len(session.turns) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
