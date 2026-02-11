"""
QBITEL Engine - Error Storage Tests

Comprehensive test suite for persistent error storage functionality.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from typing import Dict, List, Any

from ai_engine.core.error_storage import (
    PersistentErrorStorage,
    ErrorRecordModel,
    get_error_storage,
)
from ai_engine.core.error_handling import (
    ErrorRecord,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
)


class TestErrorRecordModel:
    """Test ErrorRecordModel SQLAlchemy model."""

    def test_error_record_model_creation(self):
        """Test creating ErrorRecordModel instance."""
        model = ErrorRecordModel(
            error_id="test-error-123",
            timestamp=1234567890.0,
            severity="ERROR",
            category="SYSTEM",
            component="test_component",
            operation="test_operation",
            exception_type="ValueError",
            exception_message="Test error message",
            stack_trace="Traceback...",
            context={"key": "value"},
            recovery_attempted=True,
            recovery_successful=False,
            recovery_strategy="RETRY",
            retry_count=3,
            extra_metadata={"extra": "data"},
        )

        assert model.error_id == "test-error-123"
        assert model.timestamp == 1234567890.0
        assert model.severity == "ERROR"
        assert model.category == "SYSTEM"
        assert model.component == "test_component"
        assert model.operation == "test_operation"
        assert model.exception_type == "ValueError"
        assert model.exception_message == "Test error message"
        assert model.stack_trace == "Traceback..."
        assert model.context == {"key": "value"}
        assert model.recovery_attempted is True
        assert model.recovery_successful is False
        assert model.recovery_strategy == "RETRY"
        assert model.retry_count == 3
        assert model.extra_metadata == {"extra": "data"}

    def test_error_record_model_defaults(self):
        """Test ErrorRecordModel with default values."""
        model = ErrorRecordModel(
            error_id="test-error-456",
            timestamp=1234567890.0,
            severity="WARNING",
            category="NETWORK",
            component="network_component",
            operation="network_operation",
            exception_type="ConnectionError",
            exception_message="Connection failed",
            stack_trace="Traceback...",
            context={},
        )

        assert model.recovery_attempted in (False, None)
        assert model.recovery_successful in (False, None)
        assert model.recovery_strategy is None
        assert model.retry_count == 0
        assert model.extra_metadata is None


class TestPersistentErrorStorage:
    """Test PersistentErrorStorage class."""

    @pytest.fixture
    def error_storage(self):
        """Create PersistentErrorStorage instance."""
        return PersistentErrorStorage(
            redis_url="redis://localhost:6379/0",
            postgres_url="postgresql+asyncpg://user:pass@localhost/qbitel",
        )

    @pytest.fixture
    def sample_error_record(self):
        """Create sample error record."""
        return ErrorRecord(
            error_id="test-error-789",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            component="test_component",
            operation="test_operation",
            exception_type="ValueError",
            exception_message="Test error message",
            stack_trace="Traceback...",
            context={"key": "value"},
            recovery_attempted=True,
            recovery_successful=False,
            recovery_strategy=RecoveryStrategy.RETRY,
            retry_count=3,
            metadata={"extra": "data"},
        )

    def test_persistent_error_storage_initialization(self, error_storage):
        """Test PersistentErrorStorage initialization."""
        assert error_storage is not None
        assert error_storage.redis_url == "redis://localhost:6379/0"
        assert (
            error_storage.postgres_url
            == "postgresql+asyncpg://user:pass@localhost/qbitel"
        )
        assert error_storage.redis_ttl == 86400
        assert error_storage.postgres_retention_days == 90

    def test_persistent_error_storage_custom_config(self):
        """Test PersistentErrorStorage with custom configuration."""
        storage = PersistentErrorStorage(
            redis_url="redis://custom:6379/1",
            postgres_url="postgresql+asyncpg://custom:pass@localhost/custom_db",
            redis_ttl=3600,
            postgres_retention_days=30,
        )

        assert storage.redis_url == "redis://custom:6379/1"
        assert (
            storage.postgres_url
            == "postgresql+asyncpg://custom:pass@localhost/custom_db"
        )
        assert storage.redis_ttl == 3600
        assert storage.postgres_retention_days == 30

    @pytest.mark.asyncio
    async def test_initialize_redis_connection(self, error_storage):
        """Test Redis connection initialization."""
        with patch("ai_engine.core.error_storage.redis") as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.from_url = AsyncMock(return_value=mock_redis_instance)
            mock_redis_instance.ping = AsyncMock()

            with patch(
                "ai_engine.core.error_storage.create_async_engine"
            ) as mock_engine:
                mock_db_engine = AsyncMock()
                mock_engine.return_value = mock_db_engine
                mock_conn = AsyncMock()
                mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
                mock_conn.__aexit__ = AsyncMock()
                mock_conn.run_sync = AsyncMock()
                mock_db_engine.begin.return_value = mock_conn

                await error_storage.initialize()

                mock_redis.from_url.assert_called_once()
                mock_redis_instance.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_error_redis(self, error_storage, sample_error_record):
        """Test storing error in Redis."""
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock()
        mock_redis.zadd = AsyncMock()
        mock_redis.expire = AsyncMock()
        error_storage.redis_client = mock_redis

        result = await error_storage.store_error(sample_error_record)

        assert result is True
        assert mock_redis.setex.called
        assert mock_redis.zadd.call_count == 2  # component and severity sets

    @pytest.mark.asyncio
    async def test_store_error_postgres(self, error_storage, sample_error_record):
        """Test storing error in PostgreSQL."""
        mock_session = AsyncMock()
        mock_session_maker = Mock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()

        error_storage.async_session_maker = mock_session_maker

        result = await error_storage.store_error(sample_error_record)

        assert result is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_error_from_redis(self, error_storage):
        """Test retrieving error from Redis."""
        mock_redis = AsyncMock()
        error_data = {"error_id": "test-123", "severity": "ERROR", "component": "test"}
        mock_redis.get = AsyncMock(return_value=json.dumps(error_data))
        error_storage.redis_client = mock_redis

        result = await error_storage.get_error("test-123")

        assert result is not None
        assert result["error_id"] == "test-123"
        mock_redis.get.assert_called_once_with("error:test-123")

    @pytest.mark.asyncio
    async def test_get_error_from_postgres_fallback(self, error_storage):
        """Test retrieving error from PostgreSQL when not in Redis."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        error_storage.redis_client = mock_redis

        mock_session = AsyncMock()
        mock_session_maker = Mock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_record = Mock()
        mock_record.error_id = "test-123"
        mock_record.severity = "ERROR"
        mock_record.component = "test"
        mock_session.get = AsyncMock(return_value=mock_record)

        error_storage.async_session_maker = mock_session_maker

        result = await error_storage.get_error("test-123")

        assert result is not None
        assert result["error_id"] == "test-123"

    @pytest.mark.asyncio
    async def test_get_errors_by_component(self, error_storage):
        """Test retrieving errors by component."""
        mock_redis = AsyncMock()
        mock_redis.zrangebyscore = AsyncMock(return_value=["error-1", "error-2"])
        mock_redis.get = AsyncMock(
            side_effect=[
                json.dumps({"error_id": "error-1", "component": "test"}),
                json.dumps({"error_id": "error-2", "component": "test"}),
            ]
        )
        error_storage.redis_client = mock_redis

        results = await error_storage.get_errors_by_component("test", limit=10)

        assert len(results) == 2
        assert results[0]["error_id"] == "error-1"
        assert results[1]["error_id"] == "error-2"

    @pytest.mark.asyncio
    async def test_get_error_statistics(self, error_storage):
        """Test getting error statistics."""
        mock_session = AsyncMock()
        mock_session_maker = Mock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        # Mock statistics queries
        mock_session.execute = AsyncMock()
        mock_result = Mock()
        mock_result.scalar = Mock(return_value=100)
        mock_result.first = Mock(return_value=(50, 40))
        mock_session.execute.return_value = mock_result

        error_storage.async_session_maker = mock_session_maker

        stats = await error_storage.get_error_statistics(24)

        assert stats is not None
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_cleanup_old_errors(self, error_storage):
        """Test cleaning up old errors."""
        mock_session = AsyncMock()
        mock_session_maker = Mock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_result = Mock()
        mock_result.rowcount = 42
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        error_storage.async_session_maker = mock_session_maker

        deleted_count = await error_storage.cleanup_old_errors()

        assert deleted_count == 42
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_connections(self, error_storage):
        """Test closing storage connections."""
        mock_redis = AsyncMock()
        mock_redis.close = AsyncMock()
        error_storage.redis_client = mock_redis

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()
        error_storage.db_engine = mock_engine

        await error_storage.close()

        mock_redis.close.assert_called_once()
        mock_engine.dispose.assert_called_once()


class TestGetErrorStorage:
    """Test get_error_storage function."""

    @pytest.mark.asyncio
    async def test_get_error_storage_creates_instance(self):
        """Test that get_error_storage creates a new instance."""
        with patch("ai_engine.core.error_storage._error_storage", None):
            with patch.object(
                PersistentErrorStorage, "initialize", new_callable=AsyncMock
            ):
                storage = await get_error_storage(
                    redis_url="redis://localhost:6379",
                    postgres_url="postgresql+asyncpg://user:pass@localhost/db",
                )

                assert storage is not None
                assert isinstance(storage, PersistentErrorStorage)

    @pytest.mark.asyncio
    async def test_get_error_storage_returns_existing(self):
        """Test that get_error_storage returns existing instance."""
        existing_storage = PersistentErrorStorage()

        with patch("ai_engine.core.error_storage._error_storage", existing_storage):
            storage = await get_error_storage()

            assert storage is existing_storage
