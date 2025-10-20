"""
CRONOS AI Engine - Comprehensive Log Aggregation Tests

Complete test suite for log aggregation functionality including
Elasticsearch, Loki, and LogAggregationManager components.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from typing import Dict, List, Any, Optional
from collections import deque

from ai_engine.monitoring.log_aggregation import (
    StructuredLog,
    LogLevel,
    ElasticsearchLogAggregator,
    LokiLogAggregator,
    LogAggregationManager,
    initialize_log_aggregation,
    get_log_aggregation_manager,
)
from ai_engine.core.config import Config
from ai_engine.core.exceptions import ObservabilityException


class TestStructuredLog:
    """Test StructuredLog dataclass."""

    def test_structured_log_creation(self):
        """Test creating StructuredLog instance."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test log message",
            logger_name="test_logger",
            service="test_service",
            environment="test",
            trace_id="trace123",
            span_id="span123",
            user_id="user123",
            request_id="req123",
            fields={"key": "value", "number": 123},
            tags=["tag1", "tag2"],
        )

        assert log.level == LogLevel.INFO
        assert log.message == "Test log message"
        assert log.logger_name == "test_logger"
        assert log.service == "test_service"
        assert log.environment == "test"
        assert log.trace_id == "trace123"
        assert log.span_id == "span123"
        assert log.user_id == "user123"
        assert log.request_id == "req123"
        assert log.fields == {"key": "value", "number": 123}
        assert log.tags == ["tag1", "tag2"]

    def test_structured_log_defaults(self):
        """Test StructuredLog with default values."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test log message",
            logger_name="test_logger",
            service="test_service",
            environment="test",
        )

        assert log.trace_id is None
        assert log.span_id is None
        assert log.user_id is None
        assert log.request_id is None
        assert log.fields == {}
        assert log.tags == []

    def test_structured_log_to_dict(self):
        """Test StructuredLog serialization."""
        log = StructuredLog(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            level=LogLevel.ERROR,
            message="Error occurred",
            logger_name="error_logger",
            service="error_service",
            environment="production",
            trace_id="trace456",
            user_id="user456",
            fields={"error_code": "E001", "severity": "high"},
            tags=["error", "critical"],
        )

        data = log.to_dict()

        assert data["timestamp"] == "2023-01-01T12:00:00"
        assert data["level"] == "error"
        assert data["message"] == "Error occurred"
        assert data["logger"] == "error_logger"
        assert data["service"] == "error_service"
        assert data["environment"] == "production"
        assert data["trace_id"] == "trace456"
        assert data["user_id"] == "user456"
        assert data["fields"]["error_code"] == "E001"
        assert data["fields"]["severity"] == "high"
        assert data["tags"] == ["error", "critical"]

    def test_structured_log_to_dict_optional_fields(self):
        """Test StructuredLog serialization with optional fields."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Simple message",
            logger_name="simple_logger",
            service="simple_service",
            environment="test",
        )

        data = log.to_dict()

        assert "trace_id" not in data
        assert "span_id" not in data
        assert "user_id" not in data
        assert "request_id" not in data
        assert "fields" not in data
        assert "tags" not in data

    def test_structured_log_levels(self):
        """Test all log levels."""
        levels = [
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
            LogLevel.CRITICAL,
        ]

        for level in levels:
            log = StructuredLog(
                timestamp=datetime.now(),
                level=level,
                message=f"Test {level.value} message",
                logger_name="test_logger",
                service="test_service",
                environment="test",
            )

            assert log.level == level
            assert log.to_dict()["level"] == level.value


class TestElasticsearchLogAggregator:
    """Test ElasticsearchLogAggregator functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.elasticsearch_hosts = ["http://localhost:9200", "http://localhost:9201"]
        config.elasticsearch_index_prefix = "cronos-ai-logs"
        config.elasticsearch_username = "elastic"
        config.elasticsearch_password = "password"
        return config

    @pytest.fixture
    def es_aggregator(self, mock_config):
        """Create ElasticsearchLogAggregator instance."""
        return ElasticsearchLogAggregator(mock_config)

    def test_es_aggregator_initialization(self, es_aggregator, mock_config):
        """Test ElasticsearchLogAggregator initialization."""
        assert es_aggregator.es_hosts == [
            "http://localhost:9200",
            "http://localhost:9201",
        ]
        assert es_aggregator.es_index_prefix == "cronos-ai-logs"
        assert es_aggregator.es_username == "elastic"
        assert es_aggregator.es_password == "password"
        assert es_aggregator._batch_size == 500
        assert es_aggregator._flush_interval == 5
        assert es_aggregator.logs_sent == 0
        assert es_aggregator.logs_failed == 0

    @pytest.mark.asyncio
    async def test_es_aggregator_initialize(self, es_aggregator):
        """Test ElasticsearchLogAggregator initialization."""
        with (
            patch("aiohttp.ClientSession") as mock_session,
            patch.object(es_aggregator, "_create_index_template") as mock_template,
        ):

            await es_aggregator.initialize()

            assert es_aggregator._session is not None
            assert es_aggregator._flush_task is not None
            mock_session.assert_called_once()
            mock_template.assert_called_once()

    @pytest.mark.asyncio
    async def test_es_aggregator_shutdown(self, es_aggregator):
        """Test ElasticsearchLogAggregator shutdown."""
        # Mock session and flush task
        es_aggregator._session = AsyncMock()
        es_aggregator._flush_task = AsyncMock()
        es_aggregator._flush_task.cancel = Mock()

        with patch.object(es_aggregator, "flush") as mock_flush:
            await es_aggregator.shutdown()

            es_aggregator._flush_task.cancel.assert_called_once()
            es_aggregator._session.close.assert_called_once()
            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_es_aggregator_send_log(self, es_aggregator):
        """Test sending log to Elasticsearch."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test log message",
            logger_name="test_logger",
            service="test_service",
            environment="test",
        )

        with patch.object(es_aggregator, "_flush_batch") as mock_flush:
            await es_aggregator.send_log(log)

            assert len(es_aggregator._batch_queue) == 1
            assert es_aggregator._batch_queue[0] == log

    @pytest.mark.asyncio
    async def test_es_aggregator_send_log_batch_flush(self, es_aggregator):
        """Test sending log triggers batch flush."""
        # Fill queue to batch size
        for i in range(es_aggregator._batch_size):
            log = StructuredLog(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message=f"Message {i}",
                logger_name="test_logger",
                service="test_service",
                environment="test",
            )
            es_aggregator._batch_queue.append(log)

        with patch.object(es_aggregator, "_flush_batch") as mock_flush:
            # Add one more log to trigger flush
            extra_log = StructuredLog(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="Extra message",
                logger_name="test_logger",
                service="test_service",
                environment="test",
            )

            await es_aggregator.send_log(extra_log)

            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_es_aggregator_send_log_error(self, es_aggregator):
        """Test sending log with error handling."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test_logger",
            service="test_service",
            environment="test",
        )

        with patch.object(
            es_aggregator, "_flush_batch", side_effect=Exception("Flush error")
        ):
            await es_aggregator.send_log(log)

            # Should increment failed count
            assert es_aggregator.logs_failed == 1

    @pytest.mark.asyncio
    async def test_es_aggregator_flush(self, es_aggregator):
        """Test flushing logs."""
        # Add some logs
        for i in range(3):
            log = StructuredLog(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message=f"Message {i}",
                logger_name="test_logger",
                service="test_service",
                environment="test",
            )
            es_aggregator._batch_queue.append(log)

        with patch.object(es_aggregator, "_flush_batch") as mock_flush:
            await es_aggregator.flush()
            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_es_aggregator_flush_loop(self, es_aggregator):
        """Test background flush loop."""
        es_aggregator._session = AsyncMock()
        es_aggregator._batch_queue.append(Mock())

        with patch.object(es_aggregator, "_flush_batch") as mock_flush:
            # Start flush loop
            task = asyncio.create_task(es_aggregator._flush_loop())

            # Wait a bit for the loop to run
            await asyncio.sleep(0.1)

            # Cancel the task
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should have called flush at least once
            mock_flush.assert_called()

    @pytest.mark.asyncio
    async def test_es_aggregator_flush_batch_success(self, es_aggregator):
        """Test successful batch flush to Elasticsearch."""
        # Create test logs
        logs = []
        for i in range(3):
            log = StructuredLog(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message=f"Message {i}",
                logger_name="test_logger",
                service="test_service",
                environment="test",
            )
            logs.append(log)

        es_aggregator._batch_queue.extend(logs)
        es_aggregator._session = AsyncMock()

        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"errors": False})
        es_aggregator._session.post.return_value.__aenter__.return_value = mock_response

        await es_aggregator._flush_batch()

        assert len(es_aggregator._batch_queue) == 0
        assert es_aggregator.logs_sent == 3
        es_aggregator._session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_es_aggregator_flush_batch_with_errors(self, es_aggregator):
        """Test batch flush with Elasticsearch errors."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test_logger",
            service="test_service",
            environment="test",
        )

        es_aggregator._batch_queue.append(log)
        es_aggregator._session = AsyncMock()

        # Mock response with errors
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"errors": True, "items": [{"error": "mapping error"}]}
        )
        es_aggregator._session.post.return_value.__aenter__.return_value = mock_response

        await es_aggregator._flush_batch()

        # Should not increment sent count due to errors
        assert es_aggregator.logs_sent == 0

    @pytest.mark.asyncio
    async def test_es_aggregator_flush_batch_failure(self, es_aggregator):
        """Test batch flush failure handling."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test_logger",
            service="test_service",
            environment="test",
        )

        es_aggregator._batch_queue.append(log)
        es_aggregator._session = AsyncMock()

        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 500
        es_aggregator._session.post.return_value.__aenter__.return_value = mock_response

        await es_aggregator._flush_batch()

        # Should increment failed count
        assert es_aggregator.logs_failed == 1

    @pytest.mark.asyncio
    async def test_es_aggregator_flush_batch_exception(self, es_aggregator):
        """Test batch flush exception handling."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test_logger",
            service="test_service",
            environment="test",
        )

        es_aggregator._batch_queue.append(log)
        es_aggregator._session = AsyncMock()

        # Mock exception
        es_aggregator._session.post.side_effect = Exception("Network error")

        await es_aggregator._flush_batch()

        # Should increment failed count
        assert es_aggregator.logs_failed == 1

    @pytest.mark.asyncio
    async def test_es_aggregator_create_index_template(self, es_aggregator):
        """Test creating Elasticsearch index template."""
        es_aggregator._session = AsyncMock()

        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        es_aggregator._session.put.return_value.__aenter__.return_value = mock_response

        await es_aggregator._create_index_template()

        es_aggregator._session.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_es_aggregator_create_index_template_failure(self, es_aggregator):
        """Test index template creation failure."""
        es_aggregator._session = AsyncMock()

        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 400
        es_aggregator._session.put.return_value.__aenter__.return_value = mock_response

        # Should not raise exception
        await es_aggregator._create_index_template()

        es_aggregator._session.put.assert_called_once()


class TestLokiLogAggregator:
    """Test LokiLogAggregator functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.loki_url = "http://localhost:3100"
        config.loki_username = "loki_user"
        config.loki_password = "loki_password"
        return config

    @pytest.fixture
    def loki_aggregator(self, mock_config):
        """Create LokiLogAggregator instance."""
        return LokiLogAggregator(mock_config)

    def test_loki_aggregator_initialization(self, loki_aggregator, mock_config):
        """Test LokiLogAggregator initialization."""
        assert loki_aggregator.loki_url == "http://localhost:3100"
        assert loki_aggregator.loki_username == "loki_user"
        assert loki_aggregator.loki_password == "loki_password"
        assert loki_aggregator._batch_size == 100
        assert loki_aggregator._flush_interval == 5
        assert loki_aggregator.logs_sent == 0
        assert loki_aggregator.logs_failed == 0

    @pytest.mark.asyncio
    async def test_loki_aggregator_initialize(self, loki_aggregator):
        """Test LokiLogAggregator initialization."""
        with patch("aiohttp.ClientSession") as mock_session:
            await loki_aggregator.initialize()

            assert loki_aggregator._session is not None
            assert loki_aggregator._flush_task is not None
            mock_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_loki_aggregator_shutdown(self, loki_aggregator):
        """Test LokiLogAggregator shutdown."""
        # Mock session and flush task
        loki_aggregator._session = AsyncMock()
        loki_aggregator._flush_task = AsyncMock()
        loki_aggregator._flush_task.cancel = Mock()

        with patch.object(loki_aggregator, "flush") as mock_flush:
            await loki_aggregator.shutdown()

            loki_aggregator._flush_task.cancel.assert_called_once()
            loki_aggregator._session.close.assert_called_once()
            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_loki_aggregator_send_log(self, loki_aggregator):
        """Test sending log to Loki."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test log message",
            logger_name="test_logger",
            service="test_service",
            environment="test",
        )

        with patch.object(loki_aggregator, "_flush_batch") as mock_flush:
            await loki_aggregator.send_log(log)

            assert len(loki_aggregator._batch_queue) == 1
            assert loki_aggregator._batch_queue[0] == log

    @pytest.mark.asyncio
    async def test_loki_aggregator_send_log_batch_flush(self, loki_aggregator):
        """Test sending log triggers batch flush."""
        # Fill queue to batch size
        for i in range(loki_aggregator._batch_size):
            log = StructuredLog(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message=f"Message {i}",
                logger_name="test_logger",
                service="test_service",
                environment="test",
            )
            loki_aggregator._batch_queue.append(log)

        with patch.object(loki_aggregator, "_flush_batch") as mock_flush:
            # Add one more log to trigger flush
            extra_log = StructuredLog(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="Extra message",
                logger_name="test_logger",
                service="test_service",
                environment="test",
            )

            await loki_aggregator.send_log(extra_log)

            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_loki_aggregator_send_log_error(self, loki_aggregator):
        """Test sending log with error handling."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test_logger",
            service="test_service",
            environment="test",
        )

        with patch.object(
            loki_aggregator, "_flush_batch", side_effect=Exception("Flush error")
        ):
            await loki_aggregator.send_log(log)

            # Should increment failed count
            assert loki_aggregator.logs_failed == 1

    @pytest.mark.asyncio
    async def test_loki_aggregator_flush_batch_success(self, loki_aggregator):
        """Test successful batch flush to Loki."""
        # Create test logs
        logs = []
        for i in range(3):
            log = StructuredLog(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message=f"Message {i}",
                logger_name="test_logger",
                service="test_service",
                environment="test",
            )
            logs.append(log)

        loki_aggregator._batch_queue.extend(logs)
        loki_aggregator._session = AsyncMock()

        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 204
        loki_aggregator._session.post.return_value.__aenter__.return_value = (
            mock_response
        )

        await loki_aggregator._flush_batch()

        assert len(loki_aggregator._batch_queue) == 0
        assert loki_aggregator.logs_sent == 3
        loki_aggregator._session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_loki_aggregator_flush_batch_failure(self, loki_aggregator):
        """Test batch flush failure handling."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test_logger",
            service="test_service",
            environment="test",
        )

        loki_aggregator._batch_queue.append(log)
        loki_aggregator._session = AsyncMock()

        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 500
        loki_aggregator._session.post.return_value.__aenter__.return_value = (
            mock_response
        )

        await loki_aggregator._flush_batch()

        # Should increment failed count
        assert loki_aggregator.logs_failed == 1

    @pytest.mark.asyncio
    async def test_loki_aggregator_flush_batch_exception(self, loki_aggregator):
        """Test batch flush exception handling."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test_logger",
            service="test_service",
            environment="test",
        )

        loki_aggregator._batch_queue.append(log)
        loki_aggregator._session = AsyncMock()

        # Mock exception
        loki_aggregator._session.post.side_effect = Exception("Network error")

        await loki_aggregator._flush_batch()

        # Should increment failed count
        assert loki_aggregator.logs_failed == 1

    def test_loki_aggregator_build_labels(self, loki_aggregator):
        """Test building Loki labels from log entry."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Error message",
            logger_name="error_logger",
            service="error_service",
            environment="production",
            trace_id="trace123",
            user_id="user123",
            tags=["error", "critical"],
        )

        labels = loki_aggregator._build_labels(log)

        assert labels["level"] == "error"
        assert labels["logger"] == "error_logger"
        assert labels["service"] == "error_service"
        assert labels["environment"] == "production"
        assert labels["trace_id"] == "trace123"
        assert labels["user_id"] == "user123"
        assert labels["tag_error"] == "true"
        assert labels["tag_critical"] == "true"

    def test_loki_aggregator_build_labels_minimal(self, loki_aggregator):
        """Test building Loki labels with minimal log entry."""
        log = StructuredLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Simple message",
            logger_name="simple_logger",
            service="simple_service",
            environment="test",
        )

        labels = loki_aggregator._build_labels(log)

        assert labels["level"] == "info"
        assert labels["logger"] == "simple_logger"
        assert labels["service"] == "simple_service"
        assert labels["environment"] == "test"
        assert "trace_id" not in labels
        assert "user_id" not in labels


class TestLogAggregationManager:
    """Test LogAggregationManager functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.enable_elasticsearch_logging = True
        config.enable_loki_logging = True
        config.service_name = "test-service"
        config.environment = Mock()
        config.environment.value = "test"
        return config

    @pytest.fixture
    def log_manager(self, mock_config):
        """Create LogAggregationManager instance."""
        with (
            patch("ai_engine.monitoring.log_aggregation.ElasticsearchLogAggregator"),
            patch("ai_engine.monitoring.log_aggregation.LokiLogAggregator"),
        ):
            return LogAggregationManager(mock_config)

    def test_log_manager_initialization(self, log_manager, mock_config):
        """Test LogAggregationManager initialization."""
        assert log_manager.service_name == "test-service"
        assert log_manager.environment == "test"
        assert len(log_manager.aggregators) == 2

    @pytest.mark.asyncio
    async def test_log_manager_initialize(self, log_manager):
        """Test LogAggregationManager initialization."""
        # Mock aggregators
        for aggregator in log_manager.aggregators:
            aggregator.initialize = AsyncMock()

        await log_manager.initialize()

        # Should initialize all aggregators
        for aggregator in log_manager.aggregators:
            aggregator.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_manager_shutdown(self, log_manager):
        """Test LogAggregationManager shutdown."""
        # Mock aggregators
        for aggregator in log_manager.aggregators:
            aggregator.shutdown = AsyncMock()

        await log_manager.shutdown()

        # Should shutdown all aggregators
        for aggregator in log_manager.aggregators:
            aggregator.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_manager_send_log(self, log_manager):
        """Test sending log through manager."""
        # Mock aggregators
        for aggregator in log_manager.aggregators:
            aggregator.send_log = AsyncMock()

        await log_manager.send_log(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test_logger",
            trace_id="trace123",
            user_id="user123",
            fields={"key": "value"},
            tags=["tag1", "tag2"],
        )

        # Should send to all aggregators
        for aggregator in log_manager.aggregators:
            aggregator.send_log.assert_called_once()

            # Check the log structure
            call_args = aggregator.send_log.call_args[0][0]
            assert call_args.level == LogLevel.INFO
            assert call_args.message == "Test message"
            assert call_args.logger_name == "test_logger"
            assert call_args.service == "test-service"
            assert call_args.environment == "test"
            assert call_args.trace_id == "trace123"
            assert call_args.user_id == "user123"
            assert call_args.fields == {"key": "value"}
            assert call_args.tags == ["tag1", "tag2"]

    @pytest.mark.asyncio
    async def test_log_manager_send_log_minimal(self, log_manager):
        """Test sending log with minimal parameters."""
        # Mock aggregators
        for aggregator in log_manager.aggregators:
            aggregator.send_log = AsyncMock()

        await log_manager.send_log(
            level=LogLevel.WARNING,
            message="Warning message",
            logger_name="warning_logger",
        )

        # Should send to all aggregators
        for aggregator in log_manager.aggregators:
            aggregator.send_log.assert_called_once()

            # Check the log structure
            call_args = aggregator.send_log.call_args[0][0]
            assert call_args.level == LogLevel.WARNING
            assert call_args.message == "Warning message"
            assert call_args.logger_name == "warning_logger"
            assert call_args.trace_id is None
            assert call_args.user_id is None
            assert call_args.fields == {}
            assert call_args.tags == []

    @pytest.mark.asyncio
    async def test_log_manager_send_log_aggregator_error(self, log_manager):
        """Test sending log with aggregator error."""
        # Mock aggregators - one succeeds, one fails
        log_manager.aggregators[0].send_log = AsyncMock()
        log_manager.aggregators[1].send_log = AsyncMock(
            side_effect=Exception("Aggregator error")
        )

        # Should not raise exception
        await log_manager.send_log(
            level=LogLevel.ERROR, message="Error message", logger_name="error_logger"
        )

        # Both aggregators should have been called
        log_manager.aggregators[0].send_log.assert_called_once()
        log_manager.aggregators[1].send_log.assert_called_once()

    def test_log_manager_get_statistics(self, log_manager):
        """Test getting log aggregation statistics."""
        # Mock aggregator statistics
        log_manager.aggregators[0].logs_sent = 100
        log_manager.aggregators[0].logs_failed = 5
        log_manager.aggregators[1].logs_sent = 150
        log_manager.aggregators[1].logs_failed = 3

        stats = log_manager.get_statistics()

        assert stats["total_logs_sent"] == 250
        assert stats["total_logs_failed"] == 8
        assert len(stats["aggregators"]) == 2

        # Check individual aggregator stats
        assert stats["aggregators"][0]["logs_sent"] == 100
        assert stats["aggregators"][0]["logs_failed"] == 5
        assert stats["aggregators"][1]["logs_sent"] == 150
        assert stats["aggregators"][1]["logs_failed"] == 3


class TestLogAggregationIntegration:
    """Integration tests for log aggregation components."""

    @pytest.mark.asyncio
    async def test_initialize_log_aggregation_global(self):
        """Test global log aggregation initialization."""
        mock_config = Mock(spec=Config)
        mock_config.enable_elasticsearch_logging = True
        mock_config.enable_loki_logging = False
        mock_config.service_name = "test-service"
        mock_config.environment = Mock()
        mock_config.environment.value = "test"

        with patch(
            "ai_engine.monitoring.log_aggregation.LogAggregationManager"
        ) as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager

            manager = await initialize_log_aggregation(mock_config)

            assert manager == mock_manager
            mock_manager.initialize.assert_called_once()

    def test_get_log_aggregation_manager(self):
        """Test getting global log aggregation manager."""
        # Test when no manager is set
        from ai_engine.monitoring.log_aggregation import _log_aggregation_manager

        original_manager = _log_aggregation_manager

        try:
            # Clear global manager
            import ai_engine.monitoring.log_aggregation

            ai_engine.monitoring.log_aggregation._log_aggregation_manager = None

            manager = get_log_aggregation_manager()
            assert manager is None

            # Set a mock manager
            mock_manager = Mock()
            ai_engine.monitoring.log_aggregation._log_aggregation_manager = mock_manager

            manager = get_log_aggregation_manager()
            assert manager == mock_manager

        finally:
            # Restore original manager
            ai_engine.monitoring.log_aggregation._log_aggregation_manager = (
                original_manager
            )

    @pytest.mark.asyncio
    async def test_log_aggregation_workflow(self):
        """Test complete log aggregation workflow."""
        mock_config = Mock(spec=Config)
        mock_config.enable_elasticsearch_logging = True
        mock_config.enable_loki_logging = True
        mock_config.service_name = "test-service"
        mock_config.environment = Mock()
        mock_config.environment.value = "test"

        with (
            patch(
                "ai_engine.monitoring.log_aggregation.ElasticsearchLogAggregator"
            ) as mock_es,
            patch(
                "ai_engine.monitoring.log_aggregation.LokiLogAggregator"
            ) as mock_loki,
        ):

            # Mock aggregators
            es_aggregator = AsyncMock()
            loki_aggregator = AsyncMock()
            mock_es.return_value = es_aggregator
            mock_loki.return_value = loki_aggregator

            manager = LogAggregationManager(mock_config)

            # Send various log levels
            await manager.send_log(
                level=LogLevel.INFO, message="Info message", logger_name="info_logger"
            )

            await manager.send_log(
                level=LogLevel.ERROR,
                message="Error message",
                logger_name="error_logger",
                trace_id="trace123",
                fields={"error_code": "E001"},
            )

            await manager.send_log(
                level=LogLevel.WARNING,
                message="Warning message",
                logger_name="warning_logger",
                user_id="user123",
                tags=["warning", "important"],
            )

            # Should send to both aggregators
            assert es_aggregator.send_log.call_count == 3
            assert loki_aggregator.send_log.call_count == 3

    @pytest.mark.asyncio
    async def test_log_aggregation_concurrent_logs(self):
        """Test concurrent log aggregation."""
        mock_config = Mock(spec=Config)
        mock_config.enable_elasticsearch_logging = True
        mock_config.enable_loki_logging = False
        mock_config.service_name = "test-service"
        mock_config.environment = Mock()
        mock_config.environment.value = "test"

        with patch(
            "ai_engine.monitoring.log_aggregation.ElasticsearchLogAggregator"
        ) as mock_es:
            es_aggregator = AsyncMock()
            mock_es.return_value = es_aggregator

            manager = LogAggregationManager(mock_config)

            # Create concurrent log sending tasks
            async def send_logs(log_id):
                for i in range(10):
                    await manager.send_log(
                        level=LogLevel.INFO,
                        message=f"Concurrent message {log_id}-{i}",
                        logger_name=f"concurrent_logger_{log_id}",
                    )

            # Run concurrent tasks
            tasks = [send_logs(i) for i in range(5)]
            await asyncio.gather(*tasks)

            # Should have sent 50 logs total
            assert es_aggregator.send_log.call_count == 50

    def test_structured_log_serialization_roundtrip(self):
        """Test structured log serialization and deserialization."""
        original_log = StructuredLog(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            level=LogLevel.ERROR,
            message="Test error message",
            logger_name="test_logger",
            service="test_service",
            environment="production",
            trace_id="trace123",
            span_id="span123",
            user_id="user123",
            request_id="req123",
            fields={"error_code": "E001", "severity": "high"},
            tags=["error", "critical", "urgent"],
        )

        # Serialize to dict
        data = original_log.to_dict()

        # Verify all fields are present
        assert data["timestamp"] == "2023-01-01T12:00:00"
        assert data["level"] == "error"
        assert data["message"] == "Test error message"
        assert data["logger"] == "test_logger"
        assert data["service"] == "test_service"
        assert data["environment"] == "production"
        assert data["trace_id"] == "trace123"
        assert data["span_id"] == "span123"
        assert data["user_id"] == "user123"
        assert data["request_id"] == "req123"
        assert data["fields"]["error_code"] == "E001"
        assert data["fields"]["severity"] == "high"
        assert data["tags"] == ["error", "critical", "urgent"]

    def test_log_level_comparison(self):
        """Test log level comparison and ordering."""
        levels = [
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
            LogLevel.CRITICAL,
        ]

        # Test ordering
        for i in range(len(levels) - 1):
            assert levels[i].value < levels[i + 1].value

        # Test level values
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"

    @pytest.mark.asyncio
    async def test_log_aggregation_error_recovery(self):
        """Test log aggregation error recovery."""
        mock_config = Mock(spec=Config)
        mock_config.enable_elasticsearch_logging = True
        mock_config.enable_loki_logging = True
        mock_config.service_name = "test-service"
        mock_config.environment = Mock()
        mock_config.environment.value = "test"

        with (
            patch(
                "ai_engine.monitoring.log_aggregation.ElasticsearchLogAggregator"
            ) as mock_es,
            patch(
                "ai_engine.monitoring.log_aggregation.LokiLogAggregator"
            ) as mock_loki,
        ):

            # Mock aggregators - one fails, one succeeds
            es_aggregator = AsyncMock()
            es_aggregator.send_log.side_effect = Exception("ES connection failed")
            loki_aggregator = AsyncMock()

            mock_es.return_value = es_aggregator
            mock_loki.return_value = loki_aggregator

            manager = LogAggregationManager(mock_config)

            # Should not raise exception even if one aggregator fails
            await manager.send_log(
                level=LogLevel.INFO, message="Test message", logger_name="test_logger"
            )

            # Both aggregators should have been called
            es_aggregator.send_log.assert_called_once()
            loki_aggregator.send_log.assert_called_once()
