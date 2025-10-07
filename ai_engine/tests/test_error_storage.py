"""
CRONOS AI Engine - Error Storage Tests

Comprehensive test suite for persistent error storage functionality.
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any

from ai_engine.core.error_storage import (
    ErrorStorage,
    ErrorRecordModel,
    RedisErrorStorage,
    PostgreSQLErrorStorage,
    HybridErrorStorage,
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
            metadata={"extra": "data"}
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
        assert model.metadata == {"extra": "data"}

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
            context={}
        )
        
        assert model.recovery_attempted is False
        assert model.recovery_successful is False
        assert model.recovery_strategy is None
        assert model.retry_count == 0
        assert model.metadata is None


class TestErrorStorage:
    """Test base ErrorStorage class."""

    @pytest.fixture
    def error_storage(self):
        """Create ErrorStorage instance."""
        return ErrorStorage()

    @pytest.fixture
    def sample_error_record(self):
        """Create sample error record."""
        return ErrorRecord(
            error_id="test-error-789",
            timestamp=datetime.now(),
            severity=ErrorSeverity.ERROR,
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
            metadata={"extra": "data"}
        )

    def test_error_storage_initialization(self, error_storage):
        """Test ErrorStorage initialization."""
        assert error_storage is not None

    @pytest.mark.asyncio
    async def test_store_error_not_implemented(self, error_storage, sample_error_record):
        """Test that store_error raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await error_storage.store_error(sample_error_record)

    @pytest.mark.asyncio
    async def test_get_error_not_implemented(self, error_storage):
        """Test that get_error raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await error_storage.get_error("test-error-id")

    @pytest.mark.asyncio
    async def test_get_errors_by_component_not_implemented(self, error_storage):
        """Test that get_errors_by_component raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await error_storage.get_errors_by_component("test_component")

    @pytest.mark.asyncio
    async def test_get_errors_by_severity_not_implemented(self, error_storage):
        """Test that get_errors_by_severity raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await error_storage.get_errors_by_severity(ErrorSeverity.ERROR)

    @pytest.mark.asyncio
    async def test_cleanup_old_errors_not_implemented(self, error_storage):
        """Test that cleanup_old_errors raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await error_storage.cleanup_old_errors(days=30)

    @pytest.mark.asyncio
    async def test_get_error_statistics_not_implemented(self, error_storage):
        """Test that get_error_statistics raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await error_storage.get_error_statistics()


class TestRedisErrorStorage:
    """Test RedisErrorStorage implementation."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis connection."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.set = AsyncMock(return_value=True)
        redis_mock.delete = AsyncMock(return_value=1)
        redis_mock.exists = AsyncMock(return_value=0)
        redis_mock.expire = AsyncMock(return_value=True)
        redis_mock.keys = AsyncMock(return_value=[])
        redis_mock.mget = AsyncMock(return_value=[])
        redis_mock.zadd = AsyncMock(return_value=1)
        redis_mock.zrange = AsyncMock(return_value=[])
        redis_mock.zremrangebyscore = AsyncMock(return_value=1)
        redis_mock.hgetall = AsyncMock(return_value={})
        redis_mock.hset = AsyncMock(return_value=1)
        redis_mock.hincrby = AsyncMock(return_value=1)
        return redis_mock

    @pytest.fixture
    def redis_storage(self, mock_redis):
        """Create RedisErrorStorage with mock Redis."""
        with patch('ai_engine.core.error_storage.redis.from_url', return_value=mock_redis):
            return RedisErrorStorage(redis_url="redis://localhost:6379")

    @pytest.fixture
    def sample_error_record(self):
        """Create sample error record."""
        return ErrorRecord(
            error_id="test-error-redis",
            timestamp=datetime.now(),
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.SYSTEM,
            component="redis_component",
            operation="redis_operation",
            exception_type="RedisError",
            exception_message="Redis connection failed",
            stack_trace="Traceback...",
            context={"redis_key": "test"},
            recovery_attempted=False,
            recovery_successful=False,
            recovery_strategy=None,
            retry_count=0,
            metadata=None
        )

    @pytest.mark.asyncio
    async def test_redis_storage_initialization(self, redis_storage):
        """Test RedisErrorStorage initialization."""
        assert redis_storage is not None
        assert redis_storage.redis_url == "redis://localhost:6379"

    @pytest.mark.asyncio
    async def test_store_error_success(self, redis_storage, sample_error_record, mock_redis):
        """Test successful error storage in Redis."""
        await redis_storage.store_error(sample_error_record)
        
        # Verify Redis operations were called
        mock_redis.set.assert_called()
        mock_redis.expire.assert_called()

    @pytest.mark.asyncio
    async def test_get_error_success(self, redis_storage, mock_redis):
        """Test successful error retrieval from Redis."""
        error_data = {
            "error_id": "test-error-redis",
            "timestamp": "2023-01-01T00:00:00",
            "severity": "ERROR",
            "category": "SYSTEM",
            "component": "redis_component",
            "operation": "redis_operation",
            "exception_type": "RedisError",
            "exception_message": "Redis connection failed",
            "stack_trace": "Traceback...",
            "context": '{"redis_key": "test"}',
            "recovery_attempted": "False",
            "recovery_successful": "False",
            "recovery_strategy": "None",
            "retry_count": "0",
            "metadata": "None"
        }
        
        mock_redis.get.return_value = json.dumps(error_data)
        
        result = await redis_storage.get_error("test-error-redis")
        
        assert result is not None
        assert result.error_id == "test-error-redis"
        assert result.severity == ErrorSeverity.ERROR
        assert result.category == ErrorCategory.SYSTEM

    @pytest.mark.asyncio
    async def test_get_error_not_found(self, redis_storage, mock_redis):
        """Test error retrieval when error not found."""
        mock_redis.get.return_value = None
        
        result = await redis_storage.get_error("nonexistent-error")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_errors_by_component(self, redis_storage, mock_redis):
        """Test getting errors by component."""
        mock_redis.keys.return_value = [b"error:test-error-1", b"error:test-error-2"]
        mock_redis.mget.return_value = [
            json.dumps({"error_id": "test-error-1", "component": "test_component"}),
            json.dumps({"error_id": "test-error-2", "component": "test_component"})
        ]
        
        results = await redis_storage.get_errors_by_component("test_component")
        
        assert len(results) == 2
        assert all(error.component == "test_component" for error in results)

    @pytest.mark.asyncio
    async def test_get_errors_by_severity(self, redis_storage, mock_redis):
        """Test getting errors by severity."""
        mock_redis.zrange.return_value = [b"error:test-error-1", b"error:test-error-2"]
        mock_redis.mget.return_value = [
            json.dumps({"error_id": "test-error-1", "severity": "ERROR"}),
            json.dumps({"error_id": "test-error-2", "severity": "ERROR"})
        ]
        
        results = await redis_storage.get_errors_by_severity(ErrorSeverity.ERROR)
        
        assert len(results) == 2
        assert all(error.severity == ErrorSeverity.ERROR for error in results)

    @pytest.mark.asyncio
    async def test_cleanup_old_errors(self, redis_storage, mock_redis):
        """Test cleanup of old errors."""
        await redis_storage.cleanup_old_errors(days=30)
        
        mock_redis.zremrangebyscore.assert_called()

    @pytest.mark.asyncio
    async def test_get_error_statistics(self, redis_storage, mock_redis):
        """Test getting error statistics."""
        mock_redis.hgetall.return_value = {
            b"total_errors": b"100",
            b"error_count": b"50",
            b"warning_count": b"30",
            b"critical_count": b"20"
        }
        
        stats = await redis_storage.get_error_statistics()
        
        assert stats["total_errors"] == 100
        assert stats["error_count"] == 50
        assert stats["warning_count"] == 30
        assert stats["critical_count"] == 20


class TestPostgreSQLErrorStorage:
    """Test PostgreSQLErrorStorage implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = AsyncMock()
        session.add = Mock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.execute = AsyncMock()
        session.scalar = AsyncMock()
        session.scalars = AsyncMock()
        return session

    @pytest.fixture
    def postgres_storage(self, mock_session):
        """Create PostgreSQLErrorStorage with mock session."""
        with patch('ai_engine.core.error_storage.create_async_engine'), \
             patch('ai_engine.core.error_storage.async_sessionmaker') as mock_sessionmaker:
            mock_sessionmaker.return_value.return_value.__aenter__.return_value = mock_session
            return PostgreSQLErrorStorage(database_url="postgresql://localhost/test")

    @pytest.fixture
    def sample_error_record(self):
        """Create sample error record."""
        return ErrorRecord(
            error_id="test-error-postgres",
            timestamp=datetime.now(),
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.SYSTEM,
            component="postgres_component",
            operation="postgres_operation",
            exception_type="PostgreSQLError",
            exception_message="PostgreSQL connection failed",
            stack_trace="Traceback...",
            context={"postgres_key": "test"},
            recovery_attempted=False,
            recovery_successful=False,
            recovery_strategy=None,
            retry_count=0,
            metadata=None
        )

    @pytest.mark.asyncio
    async def test_postgres_storage_initialization(self, postgres_storage):
        """Test PostgreSQLErrorStorage initialization."""
        assert postgres_storage is not None
        assert postgres_storage.database_url == "postgresql://localhost/test"

    @pytest.mark.asyncio
    async def test_store_error_success(self, postgres_storage, sample_error_record, mock_session):
        """Test successful error storage in PostgreSQL."""
        await postgres_storage.store_error(sample_error_record)
        
        # Verify database operations were called
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_error_rollback_on_failure(self, postgres_storage, sample_error_record, mock_session):
        """Test rollback on storage failure."""
        mock_session.commit.side_effect = Exception("Database error")
        
        with pytest.raises(Exception):
            await postgres_storage.store_error(sample_error_record)
        
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_error_success(self, postgres_storage, mock_session):
        """Test successful error retrieval from PostgreSQL."""
        mock_model = Mock()
        mock_model.error_id = "test-error-postgres"
        mock_model.severity = "ERROR"
        mock_model.category = "SYSTEM"
        mock_session.scalar.return_value = mock_model
        
        result = await postgres_storage.get_error("test-error-postgres")
        
        assert result is not None
        assert result.error_id == "test-error-postgres"

    @pytest.mark.asyncio
    async def test_get_error_not_found(self, postgres_storage, mock_session):
        """Test error retrieval when error not found."""
        mock_session.scalar.return_value = None
        
        result = await postgres_storage.get_error("nonexistent-error")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_errors_by_component(self, postgres_storage, mock_session):
        """Test getting errors by component."""
        mock_models = [Mock(), Mock()]
        mock_models[0].error_id = "test-error-1"
        mock_models[1].error_id = "test-error-2"
        mock_session.scalars.return_value.all.return_value = mock_models
        
        results = await postgres_storage.get_errors_by_component("test_component")
        
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_errors_by_severity(self, postgres_storage, mock_session):
        """Test getting errors by severity."""
        mock_models = [Mock(), Mock()]
        mock_models[0].error_id = "test-error-1"
        mock_models[1].error_id = "test-error-2"
        mock_session.scalars.return_value.all.return_value = mock_models
        
        results = await postgres_storage.get_errors_by_severity(ErrorSeverity.ERROR)
        
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_cleanup_old_errors(self, postgres_storage, mock_session):
        """Test cleanup of old errors."""
        mock_session.execute.return_value.rowcount = 10
        
        result = await postgres_storage.cleanup_old_errors(days=30)
        
        assert result == 10
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_error_statistics(self, postgres_storage, mock_session):
        """Test getting error statistics."""
        mock_session.scalar.return_value = 100
        
        stats = await postgres_storage.get_error_statistics()
        
        assert "total_errors" in stats


class TestHybridErrorStorage:
    """Test HybridErrorStorage implementation."""

    @pytest.fixture
    def mock_redis_storage(self):
        """Create mock Redis storage."""
        redis_storage = AsyncMock()
        redis_storage.store_error = AsyncMock()
        redis_storage.get_error = AsyncMock()
        redis_storage.get_errors_by_component = AsyncMock(return_value=[])
        redis_storage.get_errors_by_severity = AsyncMock(return_value=[])
        redis_storage.cleanup_old_errors = AsyncMock()
        redis_storage.get_error_statistics = AsyncMock(return_value={})
        return redis_storage

    @pytest.fixture
    def mock_postgres_storage(self):
        """Create mock PostgreSQL storage."""
        postgres_storage = AsyncMock()
        postgres_storage.store_error = AsyncMock()
        postgres_storage.get_error = AsyncMock()
        postgres_storage.get_errors_by_component = AsyncMock(return_value=[])
        postgres_storage.get_errors_by_severity = AsyncMock(return_value=[])
        postgres_storage.cleanup_old_errors = AsyncMock()
        postgres_storage.get_error_statistics = AsyncMock(return_value={})
        return postgres_storage

    @pytest.fixture
    def hybrid_storage(self, mock_redis_storage, mock_postgres_storage):
        """Create HybridErrorStorage with mock storages."""
        return HybridErrorStorage(
            redis_storage=mock_redis_storage,
            postgres_storage=mock_postgres_storage
        )

    @pytest.fixture
    def sample_error_record(self):
        """Create sample error record."""
        return ErrorRecord(
            error_id="test-error-hybrid",
            timestamp=datetime.now(),
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.SYSTEM,
            component="hybrid_component",
            operation="hybrid_operation",
            exception_type="HybridError",
            exception_message="Hybrid storage error",
            stack_trace="Traceback...",
            context={"hybrid_key": "test"},
            recovery_attempted=False,
            recovery_successful=False,
            recovery_strategy=None,
            retry_count=0,
            metadata=None
        )

    @pytest.mark.asyncio
    async def test_hybrid_storage_initialization(self, hybrid_storage):
        """Test HybridErrorStorage initialization."""
        assert hybrid_storage is not None
        assert hybrid_storage.redis_storage is not None
        assert hybrid_storage.postgres_storage is not None

    @pytest.mark.asyncio
    async def test_store_error_both_storages(self, hybrid_storage, sample_error_record, 
                                           mock_redis_storage, mock_postgres_storage):
        """Test storing error in both Redis and PostgreSQL."""
        await hybrid_storage.store_error(sample_error_record)
        
        mock_redis_storage.store_error.assert_called_once_with(sample_error_record)
        mock_postgres_storage.store_error.assert_called_once_with(sample_error_record)

    @pytest.mark.asyncio
    async def test_get_error_redis_first(self, hybrid_storage, mock_redis_storage, mock_postgres_storage):
        """Test getting error from Redis first, then PostgreSQL."""
        mock_error = Mock()
        mock_redis_storage.get_error.return_value = mock_error
        
        result = await hybrid_storage.get_error("test-error")
        
        assert result == mock_error
        mock_redis_storage.get_error.assert_called_once_with("test-error")
        mock_postgres_storage.get_error.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_error_postgres_fallback(self, hybrid_storage, mock_redis_storage, mock_postgres_storage):
        """Test getting error from PostgreSQL when Redis fails."""
        mock_error = Mock()
        mock_redis_storage.get_error.return_value = None
        mock_postgres_storage.get_error.return_value = mock_error
        
        result = await hybrid_storage.get_error("test-error")
        
        assert result == mock_error
        mock_redis_storage.get_error.assert_called_once_with("test-error")
        mock_postgres_storage.get_error.assert_called_once_with("test-error")

    @pytest.mark.asyncio
    async def test_get_errors_by_component_combined(self, hybrid_storage, mock_redis_storage, mock_postgres_storage):
        """Test getting errors by component from both storages."""
        redis_errors = [Mock(), Mock()]
        postgres_errors = [Mock(), Mock()]
        mock_redis_storage.get_errors_by_component.return_value = redis_errors
        mock_postgres_storage.get_errors_by_component.return_value = postgres_errors
        
        results = await hybrid_storage.get_errors_by_component("test_component")
        
        assert len(results) == 4
        mock_redis_storage.get_errors_by_component.assert_called_once_with("test_component")
        mock_postgres_storage.get_errors_by_component.assert_called_once_with("test_component")

    @pytest.mark.asyncio
    async def test_get_errors_by_severity_combined(self, hybrid_storage, mock_redis_storage, mock_postgres_storage):
        """Test getting errors by severity from both storages."""
        redis_errors = [Mock(), Mock()]
        postgres_errors = [Mock(), Mock()]
        mock_redis_storage.get_errors_by_severity.return_value = redis_errors
        mock_postgres_storage.get_errors_by_severity.return_value = postgres_errors
        
        results = await hybrid_storage.get_errors_by_severity(ErrorSeverity.ERROR)
        
        assert len(results) == 4
        mock_redis_storage.get_errors_by_severity.assert_called_once_with(ErrorSeverity.ERROR)
        mock_postgres_storage.get_errors_by_severity.assert_called_once_with(ErrorSeverity.ERROR)

    @pytest.mark.asyncio
    async def test_cleanup_old_errors_both_storages(self, hybrid_storage, mock_redis_storage, mock_postgres_storage):
        """Test cleanup of old errors in both storages."""
        mock_redis_storage.cleanup_old_errors.return_value = 5
        mock_postgres_storage.cleanup_old_errors.return_value = 10
        
        result = await hybrid_storage.cleanup_old_errors(days=30)
        
        assert result == 15
        mock_redis_storage.cleanup_old_errors.assert_called_once_with(days=30)
        mock_postgres_storage.cleanup_old_errors.assert_called_once_with(days=30)

    @pytest.mark.asyncio
    async def test_get_error_statistics_combined(self, hybrid_storage, mock_redis_storage, mock_postgres_storage):
        """Test getting combined error statistics."""
        redis_stats = {"total_errors": 50, "error_count": 30}
        postgres_stats = {"total_errors": 100, "warning_count": 20}
        mock_redis_storage.get_error_statistics.return_value = redis_stats
        mock_postgres_storage.get_error_statistics.return_value = postgres_stats
        
        stats = await hybrid_storage.get_error_statistics()
        
        assert stats["total_errors"] == 150
        assert stats["error_count"] == 30
        assert stats["warning_count"] == 20


class TestErrorStorageIntegration:
    """Integration tests for error storage components."""

    @pytest.mark.asyncio
    async def test_error_record_serialization(self):
        """Test error record serialization and deserialization."""
        original_record = ErrorRecord(
            error_id="test-serialization",
            timestamp=datetime.now(),
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.SYSTEM,
            component="serialization_test",
            operation="test_operation",
            exception_type="SerializationError",
            exception_message="Test serialization",
            stack_trace="Traceback...",
            context={"test": "data"},
            recovery_attempted=True,
            recovery_successful=False,
            recovery_strategy=RecoveryStrategy.RETRY,
            retry_count=2,
            metadata={"extra": "info"}
        )
        
        # Test JSON serialization
        json_data = json.dumps(original_record.to_dict(), default=str)
        deserialized_data = json.loads(json_data)
        
        assert deserialized_data["error_id"] == original_record.error_id
        assert deserialized_data["severity"] == original_record.severity.value
        assert deserialized_data["category"] == original_record.category.value

    @pytest.mark.asyncio
    async def test_error_storage_error_handling(self):
        """Test error handling in storage operations."""
        with patch('ai_engine.core.error_storage.redis.from_url') as mock_redis:
            mock_redis.side_effect = Exception("Redis connection failed")
            
            with pytest.raises(Exception):
                RedisErrorStorage(redis_url="redis://invalid:6379")

    @pytest.mark.asyncio
    async def test_error_storage_concurrent_operations(self):
        """Test concurrent error storage operations."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.expire = AsyncMock(return_value=True)
        
        with patch('ai_engine.core.error_storage.redis.from_url', return_value=mock_redis):
            storage = RedisErrorStorage(redis_url="redis://localhost:6379")
            
            # Create multiple error records
            errors = []
            for i in range(10):
                error = ErrorRecord(
                    error_id=f"concurrent-error-{i}",
                    timestamp=datetime.now(),
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.SYSTEM,
                    component="concurrent_test",
                    operation="test_operation",
                    exception_type="ConcurrentError",
                    exception_message=f"Concurrent error {i}",
                    stack_trace="Traceback...",
                    context={"index": i},
                    recovery_attempted=False,
                    recovery_successful=False,
                    recovery_strategy=None,
                    retry_count=0,
                    metadata=None
                )
                errors.append(error)
            
            # Store all errors concurrently
            tasks = [storage.store_error(error) for error in errors]
            await asyncio.gather(*tasks)
            
            # Verify all errors were stored
            assert mock_redis.set.call_count == 10
            assert mock_redis.expire.call_count == 10
