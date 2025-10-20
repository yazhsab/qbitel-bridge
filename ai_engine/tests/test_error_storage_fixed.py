"""
Test suite for error_storage.py with mocked dependencies.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json

from ai_engine.core.error_storage import PersistentErrorStorage, ErrorRecordModel
from ai_engine.core.error_handling import (
    ErrorRecord,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
)


class TestPersistentErrorStorage:
    """Test PersistentErrorStorage with mocked dependencies."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.setex = AsyncMock(return_value=True)
        mock_redis.zadd = AsyncMock(return_value=True)
        mock_redis.expire = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.zrangebyscore = AsyncMock(return_value=[])
        mock_redis.close = AsyncMock()
        return mock_redis

    @pytest.fixture
    def mock_db_engine(self):
        """Mock database engine."""
        mock_engine = AsyncMock()
        mock_engine.begin = AsyncMock()
        mock_engine.dispose = AsyncMock()
        return mock_engine

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        mock_session = AsyncMock()
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock()
        return mock_session

    @pytest.fixture
    def mock_session_maker(self, mock_session):
        """Mock session maker."""
        mock_maker = AsyncMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=None)
        return mock_maker

    @pytest.fixture
    def error_record(self):
        """Create a sample error record."""
        return ErrorRecord(
            error_id="test-error-123",
            timestamp=1234567890.0,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INTERNAL_LOGIC,
            component="test_component",
            operation="test_operation",
            exception_type="TestException",
            exception_message="Test error message",
            stack_trace='Traceback (most recent call last):\n  File "test.py", line 1, in <module>\n    raise TestException()\nTestException: Test error message',
            context=ErrorContext(
                component="test_component",
                operation="test_operation",
                additional_data={"test": "context"},
            ),
            recovery_attempted=False,
            recovery_successful=False,
            recovery_strategy=None,
            retry_count=0,
            metadata={"test": "metadata"},
        )

    @pytest.mark.asyncio
    async def test_initialization_success(
        self, mock_redis, mock_db_engine, mock_session_maker
    ):
        """Test successful initialization."""
        with (
            patch("redis.asyncio.from_url", return_value=mock_redis),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_db_engine,
            ),
            patch(
                "sqlalchemy.ext.asyncio.async_sessionmaker",
                return_value=mock_session_maker,
            ),
        ):

            storage = PersistentErrorStorage()
            await storage.initialize()

            assert storage.redis_client is not None
            assert storage.db_engine is not None
            assert storage.async_session_maker is not None

    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test initialization failure."""
        with patch(
            "redis.asyncio.from_url", side_effect=Exception("Redis connection failed")
        ):
            storage = PersistentErrorStorage()

            with pytest.raises(Exception):
                await storage.initialize()

    @pytest.mark.asyncio
    async def test_store_error_success(
        self, mock_redis, mock_session, mock_session_maker, error_record
    ):
        """Test successful error storage."""
        storage = PersistentErrorStorage()
        storage.redis_client = mock_redis
        storage.async_session_maker = mock_session_maker

        result = await storage.store_error(error_record)

        assert result is True
        mock_redis.setex.assert_called_once()
        mock_redis.zadd.assert_called()
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_error_failure(self, error_record):
        """Test error storage failure."""
        storage = PersistentErrorStorage()
        storage.redis_client = None
        storage.async_session_maker = None

        # This should not raise an exception, just return False
        result = await storage.store_error(error_record)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_error_from_redis(self, mock_redis, error_record):
        """Test getting error from Redis."""
        error_json = json.dumps(error_record.to_dict())
        mock_redis.get.return_value = error_json

        storage = PersistentErrorStorage()
        storage.redis_client = mock_redis
        storage.async_session_maker = None

        result = await storage.get_error("test-error-123")

        assert result is not None
        assert result["error_id"] == "test-error-123"
        mock_redis.get.assert_called_once_with("error:test-error-123")

    @pytest.mark.asyncio
    async def test_get_error_from_database(self, mock_session, mock_session_maker):
        """Test getting error from database."""
        mock_db_record = Mock()
        mock_db_record.error_id = "test-error-123"
        mock_db_record.timestamp = 1234567890.0
        mock_db_record.severity = "ERROR"
        mock_db_record.category = "SYSTEM"
        mock_db_record.component = "test_component"
        mock_db_record.operation = "test_operation"
        mock_db_record.exception_type = "TestException"
        mock_db_record.exception_message = "Test error message"
        mock_db_record.stack_trace = "Test stack trace"
        mock_db_record.context = {"test": "context"}
        mock_db_record.recovery_attempted = False
        mock_db_record.recovery_successful = False
        mock_db_record.recovery_strategy = None
        mock_db_record.retry_count = 0
        mock_db_record.extra_metadata = {"test": "metadata"}

        mock_session.get.return_value = mock_db_record

        storage = PersistentErrorStorage()
        storage.redis_client = None
        storage.async_session_maker = mock_session_maker

        result = await storage.get_error("test-error-123")

        assert result is not None
        assert result["error_id"] == "test-error-123"
        assert result["severity"] == "ERROR"

    @pytest.mark.asyncio
    async def test_get_error_not_found(self, mock_redis):
        """Test getting non-existent error."""
        mock_redis.get.return_value = None

        storage = PersistentErrorStorage()
        storage.redis_client = mock_redis
        storage.async_session_maker = None

        result = await storage.get_error("non-existent-error")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_errors_by_component(
        self, mock_redis, mock_session, mock_session_maker
    ):
        """Test getting errors by component."""
        mock_redis.zrangebyscore.return_value = ["error1", "error2"]

        storage = PersistentErrorStorage()
        storage.redis_client = mock_redis
        storage.async_session_maker = mock_session_maker

        # Mock the get_error method to return sample data
        with patch.object(storage, "get_error", return_value={"error_id": "error1"}):
            result = await storage.get_errors_by_component("test_component")

            assert len(result) == 2
            assert result[0]["error_id"] == "error1"

    @pytest.mark.asyncio
    async def test_get_error_statistics(self, mock_session, mock_session_maker):
        """Test getting error statistics."""
        # Mock the database query results
        mock_result = Mock()
        mock_result.scalar.return_value = 10  # total errors
        mock_result.first.return_value = (5, 3)  # recovery stats

        mock_session.execute.return_value = mock_result

        storage = PersistentErrorStorage()
        storage.redis_client = None
        storage.async_session_maker = mock_session_maker

        result = await storage.get_error_statistics(24)

        assert result["total_errors"] == 10
        assert result["time_window_hours"] == 24

    @pytest.mark.asyncio
    async def test_cleanup_old_errors(self, mock_session, mock_session_maker):
        """Test cleanup of old errors."""
        mock_result = Mock()
        mock_result.rowcount = 5

        mock_session.execute.return_value = mock_result

        storage = PersistentErrorStorage()
        storage.redis_client = None
        storage.async_session_maker = mock_session_maker

        result = await storage.cleanup_old_errors()

        assert result == 5
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_connections(self, mock_redis, mock_db_engine):
        """Test closing connections."""
        storage = PersistentErrorStorage()
        storage.redis_client = mock_redis
        storage.db_engine = mock_db_engine

        await storage.close()

        mock_redis.close.assert_called_once()
        mock_db_engine.dispose.assert_called_once()


class TestErrorRecordModel:
    """Test ErrorRecordModel SQLAlchemy model."""

    def test_model_creation(self):
        """Test that the model can be created."""
        # This test just verifies the model structure is correct
        model = ErrorRecordModel(
            error_id="test-123",
            timestamp=1234567890.0,
            severity="ERROR",
            category="SYSTEM",
            component="test_component",
            operation="test_operation",
            exception_type="TestException",
            exception_message="Test message",
            stack_trace="Test trace",
            context={"test": "context"},
            recovery_attempted=False,
            recovery_successful=False,
            recovery_strategy=None,
            retry_count=0,
            extra_metadata={"test": "metadata"},
        )

        assert model.error_id == "test-123"
        assert model.severity == "ERROR"
        assert model.extra_metadata == {"test": "metadata"}
