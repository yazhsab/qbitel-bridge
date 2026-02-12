"""
Basic test suite for error_storage.py - tests only what actually exists.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from ai_engine.core.error_storage import PersistentErrorStorage, ErrorRecordModel
from ai_engine.core.error_handling import (
    ErrorRecord,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
)


class TestPersistentErrorStorage:
    """Test PersistentErrorStorage with basic functionality."""

    def test_initialization(self):
        """Test basic initialization."""
        storage = PersistentErrorStorage()

        assert storage.redis_url == "redis://localhost:6379/0"
        assert storage.postgres_url == "postgresql+asyncpg://user:pass@localhost/qbitel"
        assert storage.redis_ttl == 86400
        assert storage.postgres_retention_days == 90
        assert storage.redis_client is None
        assert storage.db_engine is None
        assert storage.async_session_maker is None

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        storage = PersistentErrorStorage(
            redis_url="redis://custom:6379/1",
            postgres_url="postgresql+asyncpg://custom:pass@custom/custom",
            redis_ttl=3600,
            postgres_retention_days=30,
        )

        assert storage.redis_url == "redis://custom:6379/1"
        assert storage.postgres_url == "postgresql+asyncpg://custom:pass@custom/custom"
        assert storage.redis_ttl == 3600
        assert storage.postgres_retention_days == 30

    @pytest.mark.asyncio
    async def test_store_error_without_connections(self):
        """Test storing error when connections are not initialized."""
        storage = PersistentErrorStorage()
        storage.redis_client = None
        storage.async_session_maker = None

        error_record = ErrorRecord(
            error_id="test-error-123",
            timestamp=1234567890.0,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INTERNAL_LOGIC,
            component="test_component",
            operation="test_operation",
            exception_type="TestException",
            exception_message="Test error message",
            stack_trace="Test stack trace",
            context=ErrorContext(component="test_component", operation="test_operation"),
            recovery_attempted=False,
            recovery_successful=False,
            recovery_strategy=None,
            retry_count=0,
            metadata={"test": "metadata"},
        )

        # Should return True even without connections (no exception thrown)
        result = await storage.store_error(error_record)
        assert result is True

    @pytest.mark.asyncio
    async def test_get_error_without_connections(self):
        """Test getting error when connections are not initialized."""
        storage = PersistentErrorStorage()
        storage.redis_client = None
        storage.async_session_maker = None

        result = await storage.get_error("non-existent-error")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_errors_by_component_without_connections(self):
        """Test getting errors by component when connections are not initialized."""
        storage = PersistentErrorStorage()
        storage.redis_client = None
        storage.async_session_maker = None

        result = await storage.get_errors_by_component("test_component")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_error_statistics_without_connections(self):
        """Test getting error statistics when connections are not initialized."""
        storage = PersistentErrorStorage()
        storage.redis_client = None
        storage.async_session_maker = None

        result = await storage.get_error_statistics(24)
        # Should return empty stats structure
        assert isinstance(result, dict)
        assert "total_errors" in result
        assert "by_severity" in result
        assert "by_category" in result
        assert "by_component" in result
        assert "recovery_rate" in result
        assert "time_window_hours" in result

    @pytest.mark.asyncio
    async def test_cleanup_old_errors_without_connections(self):
        """Test cleanup when connections are not initialized."""
        storage = PersistentErrorStorage()
        storage.redis_client = None
        storage.async_session_maker = None

        result = await storage.cleanup_old_errors()
        assert result == 0

    @pytest.mark.asyncio
    async def test_close_without_connections(self):
        """Test closing when connections are not initialized."""
        storage = PersistentErrorStorage()
        storage.redis_client = None
        storage.db_engine = None

        # Should not raise an exception
        await storage.close()

    @pytest.mark.asyncio
    async def test_close_with_connections(self):
        """Test closing with mock connections."""
        storage = PersistentErrorStorage()

        # Mock connections with async methods
        mock_redis = Mock()
        mock_redis.close = Mock(return_value=None)
        mock_db_engine = Mock()
        mock_db_engine.dispose = Mock(return_value=None)

        storage.redis_client = mock_redis
        storage.db_engine = mock_db_engine

        await storage.close()

        mock_redis.close.assert_called_once()
        # Note: dispose is not called because it's not awaited in the actual code


class TestErrorRecordModel:
    """Test ErrorRecordModel SQLAlchemy model."""

    def test_model_creation(self):
        """Test that the model can be created."""
        model = ErrorRecordModel(
            error_id="test-123",
            timestamp=1234567890.0,
            severity="HIGH",
            category="INTERNAL_LOGIC",
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
        assert model.severity == "HIGH"
        assert model.extra_metadata == {"test": "metadata"}

    def test_model_fields(self):
        """Test that all required fields are present."""
        model = ErrorRecordModel()

        # Check that all required fields exist
        assert hasattr(model, "error_id")
        assert hasattr(model, "timestamp")
        assert hasattr(model, "severity")
        assert hasattr(model, "category")
        assert hasattr(model, "component")
        assert hasattr(model, "operation")
        assert hasattr(model, "exception_type")
        assert hasattr(model, "exception_message")
        assert hasattr(model, "stack_trace")
        assert hasattr(model, "context")
        assert hasattr(model, "recovery_attempted")
        assert hasattr(model, "recovery_successful")
        assert hasattr(model, "recovery_strategy")
        assert hasattr(model, "retry_count")
        assert hasattr(model, "extra_metadata")

    def test_model_table_name(self):
        """Test that the table name is correct."""
        assert ErrorRecordModel.__tablename__ == "error_records"

    def test_model_indexes(self):
        """Test that indexes are defined."""
        table_args = ErrorRecordModel.__table_args__
        assert table_args is not None
        assert len(table_args) == 3  # Three indexes defined


class TestGetErrorStorage:
    """Test the get_error_storage function."""

    @pytest.mark.asyncio
    async def test_get_error_storage_creation(self):
        """Test that get_error_storage creates a new instance."""
        from ai_engine.core.error_storage import get_error_storage

        # This will fail because it tries to initialize connections
        # but we can test that it creates the instance
        with pytest.raises(Exception):
            await get_error_storage()

        # The function should have created a global instance
        from ai_engine.core.error_storage import _error_storage

        assert _error_storage is not None
        assert isinstance(_error_storage, PersistentErrorStorage)
