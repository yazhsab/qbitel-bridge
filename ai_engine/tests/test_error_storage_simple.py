"""
Simple test suite for error_storage.py with basic functionality testing.
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

    def test_get_key_path(self):
        """Test key path generation."""
        storage = PersistentErrorStorage()

        # Test with default namespace
        key_path = storage._get_key_path("test_key")
        assert key_path == "/test_key"

        # Test with custom namespace
        storage.namespace = "production"
        key_path = storage._get_key_path("database_url")
        assert key_path == "/production/database_url"

    def test_config_validation(self):
        """Test config validation."""
        storage = PersistentErrorStorage()

        # Valid config
        valid_config = {"database_url": "postgresql://localhost/test"}
        assert storage._validate_config(valid_config) is True

        # Invalid config (not a dict)
        invalid_config = "not a dict"
        assert storage._validate_config(invalid_config) is False

        # Invalid config (None)
        assert storage._validate_config(None) is False

    def test_key_validation(self):
        """Test key validation."""
        storage = PersistentErrorStorage()

        # Valid key
        assert storage._validate_key("valid_key") is True
        assert storage._validate_key("key_with_underscores") is True
        assert storage._validate_key("key-with-dashes") is True

        # Invalid key
        assert storage._validate_key("") is False
        assert storage._validate_key(None) is False
        assert storage._validate_key("key with spaces") is False
        assert storage._validate_key("key/with/slashes") is False

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

        # Should not raise an exception, just return False
        result = await storage.store_error(error_record)
        assert result is False

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
        assert result == {}

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
