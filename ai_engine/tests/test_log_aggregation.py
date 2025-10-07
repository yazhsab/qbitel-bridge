"""
CRONOS AI Engine - Log Aggregation Tests

Comprehensive test suite for log aggregation functionality.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional

from ai_engine.monitoring.log_aggregation import (
    LogAggregator,
    LogEntry,
    LogLevel,
    LogSource,
    LogFilter,
    LogProcessor,
    ElasticsearchLogSink,
    KafkaLogSink,
    FileLogSink,
    LogAggregationConfig,
    LogAggregationException,
)


class TestLogEntry:
    """Test LogEntry dataclass."""

    def test_log_entry_creation(self):
        """Test creating LogEntry instance."""
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test log message",
            source=LogSource.APPLICATION,
            component="test_component",
            service="test_service",
            trace_id="trace123",
            span_id="span123",
            user_id="user123",
            session_id="session123",
            tags={"environment": "test", "version": "1.0.0"},
            metadata={"request_id": "req123", "response_time": 150.5}
        )
        
        assert log_entry.level == LogLevel.INFO
        assert log_entry.message == "Test log message"
        assert log_entry.source == LogSource.APPLICATION
        assert log_entry.component == "test_component"
        assert log_entry.service == "test_service"
        assert log_entry.trace_id == "trace123"
        assert log_entry.span_id == "span123"
        assert log_entry.user_id == "user123"
        assert log_entry.session_id == "session123"
        assert log_entry.tags == {"environment": "test", "version": "1.0.0"}
        assert log_entry.metadata == {"request_id": "req123", "response_time": 150.5}

    def test_log_entry_defaults(self):
        """Test LogEntry with default values."""
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test log message"
        )
        
        assert log_entry.source == LogSource.APPLICATION
        assert log_entry.component is None
        assert log_entry.service is None
        assert log_entry.trace_id is None
        assert log_entry.span_id is None
        assert log_entry.user_id is None
        assert log_entry.session_id is None
        assert log_entry.tags == {}
        assert log_entry.metadata == {}

    def test_log_entry_serialization(self):
        """Test LogEntry serialization."""
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Error occurred",
            source=LogSource.SYSTEM,
            component="error_handler",
            tags={"error_type": "validation"},
            metadata={"error_code": "E001"}
        )
        
        serialized = log_entry.to_dict()
        assert serialized["level"] == "ERROR"
        assert serialized["message"] == "Error occurred"
        assert serialized["source"] == "SYSTEM"
        assert serialized["component"] == "error_handler"
        assert serialized["tags"]["error_type"] == "validation"
        assert serialized["metadata"]["error_code"] == "E001"

    def test_log_entry_deserialization(self):
        """Test LogEntry deserialization."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "level": "WARNING",
            "message": "Warning message",
            "source": "SECURITY",
            "component": "auth_service",
            "tags": {"security_level": "medium"},
            "metadata": {"attempt_count": 3}
        }
        
        log_entry = LogEntry.from_dict(data)
        assert log_entry.level == LogLevel.WARNING
        assert log_entry.message == "Warning message"
        assert log_entry.source == LogSource.SECURITY
        assert log_entry.component == "auth_service"
        assert log_entry.tags["security_level"] == "medium"
        assert log_entry.metadata["attempt_count"] == 3

    def test_log_entry_level_comparison(self):
        """Test LogEntry level comparison."""
        info_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Info message"
        )
        
        error_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Error message"
        )
        
        assert error_entry.level > info_entry.level
        assert info_entry.level < error_entry.level
        assert info_entry.level == LogLevel.INFO

    def test_log_entry_has_tag(self):
        """Test LogEntry has_tag method."""
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            tags={"environment": "test", "service": "api"}
        )
        
        assert log_entry.has_tag("environment") is True
        assert log_entry.has_tag("service") is True
        assert log_entry.has_tag("nonexistent") is False

    def test_log_entry_get_tag(self):
        """Test LogEntry get_tag method."""
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            tags={"environment": "test", "service": "api"}
        )
        
        assert log_entry.get_tag("environment") == "test"
        assert log_entry.get_tag("service") == "api"
        assert log_entry.get_tag("nonexistent") is None
        assert log_entry.get_tag("nonexistent", "default") == "default"


class TestLogFilter:
    """Test LogFilter functionality."""

    def test_log_filter_creation(self):
        """Test creating LogFilter instance."""
        log_filter = LogFilter(
            levels=[LogLevel.ERROR, LogLevel.CRITICAL],
            sources=[LogSource.APPLICATION, LogSource.SYSTEM],
            components=["auth_service", "api_gateway"],
            services=["user_service", "payment_service"],
            tags={"environment": "production"},
            exclude_tags={"debug": "true"},
            time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
            message_pattern="error|exception|fail"
        )
        
        assert LogLevel.ERROR in log_filter.levels
        assert LogLevel.CRITICAL in log_filter.levels
        assert LogSource.APPLICATION in log_filter.sources
        assert "auth_service" in log_filter.components
        assert log_filter.tags["environment"] == "production"
        assert log_filter.exclude_tags["debug"] == "true"
        assert log_filter.message_pattern == "error|exception|fail"

    def test_log_filter_matches_level(self):
        """Test LogFilter level matching."""
        log_filter = LogFilter(levels=[LogLevel.ERROR, LogLevel.CRITICAL])
        
        error_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Error message"
        )
        
        info_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Info message"
        )
        
        assert log_filter.matches(error_entry) is True
        assert log_filter.matches(info_entry) is False

    def test_log_filter_matches_source(self):
        """Test LogFilter source matching."""
        log_filter = LogFilter(sources=[LogSource.APPLICATION, LogSource.SYSTEM])
        
        app_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="App message",
            source=LogSource.APPLICATION
        )
        
        security_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Security message",
            source=LogSource.SECURITY
        )
        
        assert log_filter.matches(app_entry) is True
        assert log_filter.matches(security_entry) is False

    def test_log_filter_matches_component(self):
        """Test LogFilter component matching."""
        log_filter = LogFilter(components=["auth_service", "api_gateway"])
        
        auth_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Auth message",
            component="auth_service"
        )
        
        db_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="DB message",
            component="database"
        )
        
        assert log_filter.matches(auth_entry) is True
        assert log_filter.matches(db_entry) is False

    def test_log_filter_matches_tags(self):
        """Test LogFilter tag matching."""
        log_filter = LogFilter(tags={"environment": "production", "service": "api"})
        
        matching_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Production message",
            tags={"environment": "production", "service": "api", "version": "1.0.0"}
        )
        
        non_matching_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            tags={"environment": "test", "service": "api"}
        )
        
        assert log_filter.matches(matching_entry) is True
        assert log_filter.matches(non_matching_entry) is False

    def test_log_filter_excludes_tags(self):
        """Test LogFilter exclude tags."""
        log_filter = LogFilter(exclude_tags={"debug": "true", "test": "true"})
        
        debug_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Debug message",
            tags={"debug": "true", "service": "api"}
        )
        
        normal_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Normal message",
            tags={"service": "api"}
        )
        
        assert log_filter.matches(debug_entry) is False
        assert log_filter.matches(normal_entry) is True

    def test_log_filter_matches_time_range(self):
        """Test LogFilter time range matching."""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        two_hours_ago = now - timedelta(hours=2)
        
        log_filter = LogFilter(time_range=(one_hour_ago, now))
        
        recent_entry = LogEntry(
            timestamp=now - timedelta(minutes=30),
            level=LogLevel.INFO,
            message="Recent message"
        )
        
        old_entry = LogEntry(
            timestamp=two_hours_ago,
            level=LogLevel.INFO,
            message="Old message"
        )
        
        assert log_filter.matches(recent_entry) is True
        assert log_filter.matches(old_entry) is False

    def test_log_filter_matches_message_pattern(self):
        """Test LogFilter message pattern matching."""
        log_filter = LogFilter(message_pattern="error|exception|fail")
        
        error_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Database connection error occurred"
        )
        
        exception_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Null pointer exception in service"
        )
        
        normal_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="User login successful"
        )
        
        assert log_filter.matches(error_entry) is True
        assert log_filter.matches(exception_entry) is True
        assert log_filter.matches(normal_entry) is False

    def test_log_filter_combines_conditions(self):
        """Test LogFilter combining multiple conditions."""
        log_filter = LogFilter(
            levels=[LogLevel.ERROR, LogLevel.CRITICAL],
            sources=[LogSource.APPLICATION],
            components=["auth_service"],
            tags={"environment": "production"}
        )
        
        matching_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Auth error",
            source=LogSource.APPLICATION,
            component="auth_service",
            tags={"environment": "production", "service": "api"}
        )
        
        non_matching_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Auth error",
            source=LogSource.SECURITY,  # Wrong source
            component="auth_service",
            tags={"environment": "production"}
        )
        
        assert log_filter.matches(matching_entry) is True
        assert log_filter.matches(non_matching_entry) is False


class TestLogProcessor:
    """Test LogProcessor functionality."""

    @pytest.fixture
    def log_processor(self):
        """Create LogProcessor instance."""
        return LogProcessor()

    def test_log_processor_initialization(self, log_processor):
        """Test LogProcessor initialization."""
        assert log_processor is not None
        assert log_processor.filters == []
        assert log_processor.transformers == []

    def test_add_filter(self, log_processor):
        """Test adding filter to LogProcessor."""
        log_filter = LogFilter(levels=[LogLevel.ERROR])
        log_processor.add_filter(log_filter)
        
        assert log_filter in log_processor.filters

    def test_remove_filter(self, log_processor):
        """Test removing filter from LogProcessor."""
        log_filter = LogFilter(levels=[LogLevel.ERROR])
        log_processor.add_filter(log_filter)
        log_processor.remove_filter(log_filter)
        
        assert log_filter not in log_processor.filters

    def test_add_transformer(self, log_processor):
        """Test adding transformer to LogProcessor."""
        def transformer(log_entry):
            log_entry.tags["processed"] = "true"
            return log_entry
        
        log_processor.add_transformer(transformer)
        
        assert transformer in log_processor.transformers

    def test_remove_transformer(self, log_processor):
        """Test removing transformer from LogProcessor."""
        def transformer(log_entry):
            log_entry.tags["processed"] = "true"
            return log_entry
        
        log_processor.add_transformer(transformer)
        log_processor.remove_transformer(transformer)
        
        assert transformer not in log_processor.transformers

    def test_process_log_entry_with_filter(self, log_processor):
        """Test processing log entry with filter."""
        log_filter = LogFilter(levels=[LogLevel.ERROR, LogLevel.CRITICAL])
        log_processor.add_filter(log_filter)
        
        error_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Error message"
        )
        
        info_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Info message"
        )
        
        processed_error = log_processor.process(error_entry)
        processed_info = log_processor.process(info_entry)
        
        assert processed_error == error_entry
        assert processed_info is None

    def test_process_log_entry_with_transformer(self, log_processor):
        """Test processing log entry with transformer."""
        def transformer(log_entry):
            log_entry.tags["processed"] = "true"
            log_entry.tags["timestamp"] = log_entry.timestamp.isoformat()
            return log_entry
        
        log_processor.add_transformer(transformer)
        
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message"
        )
        
        processed = log_processor.process(log_entry)
        
        assert processed is not None
        assert processed.tags["processed"] == "true"
        assert "timestamp" in processed.tags

    def test_process_log_entry_with_filter_and_transformer(self, log_processor):
        """Test processing log entry with both filter and transformer."""
        log_filter = LogFilter(levels=[LogLevel.ERROR])
        log_processor.add_filter(log_filter)
        
        def transformer(log_entry):
            log_entry.tags["processed"] = "true"
            return log_entry
        
        log_processor.add_transformer(transformer)
        
        error_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Error message"
        )
        
        processed = log_processor.process(error_entry)
        
        assert processed is not None
        assert processed.tags["processed"] == "true"

    def test_process_multiple_log_entries(self, log_processor):
        """Test processing multiple log entries."""
        log_filter = LogFilter(levels=[LogLevel.ERROR, LogLevel.WARNING])
        log_processor.add_filter(log_filter)
        
        def transformer(log_entry):
            log_entry.tags["batch_processed"] = "true"
            return log_entry
        
        log_processor.add_transformer(transformer)
        
        log_entries = [
            LogEntry(timestamp=datetime.now(), level=LogLevel.ERROR, message="Error 1"),
            LogEntry(timestamp=datetime.now(), level=LogLevel.INFO, message="Info 1"),
            LogEntry(timestamp=datetime.now(), level=LogLevel.WARNING, message="Warning 1"),
            LogEntry(timestamp=datetime.now(), level=LogLevel.ERROR, message="Error 2")
        ]
        
        processed = log_processor.process_batch(log_entries)
        
        assert len(processed) == 3  # ERROR, WARNING, ERROR
        assert all(entry.tags["batch_processed"] == "true" for entry in processed)

    def test_clear_filters_and_transformers(self, log_processor):
        """Test clearing filters and transformers."""
        log_filter = LogFilter(levels=[LogLevel.ERROR])
        log_processor.add_filter(log_filter)
        
        def transformer(log_entry):
            return log_entry
        
        log_processor.add_transformer(transformer)
        
        log_processor.clear_filters()
        log_processor.clear_transformers()
        
        assert len(log_processor.filters) == 0
        assert len(log_processor.transformers) == 0


class TestElasticsearchLogSink:
    """Test ElasticsearchLogSink functionality."""

    @pytest.fixture
    def es_config(self):
        """Create Elasticsearch configuration."""
        return {
            "hosts": ["localhost:9200"],
            "index": "cronos-logs",
            "username": "elastic",
            "password": "password",
            "ssl": False,
            "timeout": 30
        }

    @pytest.fixture
    def es_sink(self, es_config):
        """Create ElasticsearchLogSink instance."""
        with patch('ai_engine.monitoring.log_aggregation.Elasticsearch') as mock_es:
            return ElasticsearchLogSink(es_config)

    def test_es_sink_initialization(self, es_sink, es_config):
        """Test ElasticsearchLogSink initialization."""
        assert es_sink.config == es_config
        assert es_sink.hosts == es_config["hosts"]
        assert es_sink.index == es_config["index"]

    @pytest.mark.asyncio
    async def test_es_sink_send_log_entry(self, es_sink):
        """Test sending log entry to Elasticsearch."""
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test log message",
            source=LogSource.APPLICATION,
            component="test_component"
        )
        
        with patch.object(es_sink, 'client') as mock_client:
            mock_client.index.return_value = {"result": "created"}
            
            await es_sink.send(log_entry)
            
            mock_client.index.assert_called_once()

    @pytest.mark.asyncio
    async def test_es_sink_send_batch(self, es_sink):
        """Test sending batch of log entries to Elasticsearch."""
        log_entries = [
            LogEntry(timestamp=datetime.now(), level=LogLevel.INFO, message="Message 1"),
            LogEntry(timestamp=datetime.now(), level=LogLevel.WARNING, message="Message 2"),
            LogEntry(timestamp=datetime.now(), level=LogLevel.ERROR, message="Message 3")
        ]
        
        with patch.object(es_sink, 'client') as mock_client:
            mock_client.bulk.return_value = {"items": [{"index": {"result": "created"}} for _ in log_entries]}
            
            await es_sink.send_batch(log_entries)
            
            mock_client.bulk.assert_called_once()

    @pytest.mark.asyncio
    async def test_es_sink_connection_error(self, es_sink):
        """Test Elasticsearch connection error handling."""
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message"
        )
        
        with patch.object(es_sink, 'client') as mock_client:
            mock_client.index.side_effect = Exception("Connection failed")
            
            with pytest.raises(LogAggregationException):
                await es_sink.send(log_entry)

    def test_es_sink_create_index_template(self, es_sink):
        """Test creating Elasticsearch index template."""
        with patch.object(es_sink, 'client') as mock_client:
            mock_client.indices.put_template.return_value = {"acknowledged": True}
            
            es_sink.create_index_template()
            
            mock_client.indices.put_template.assert_called_once()


class TestKafkaLogSink:
    """Test KafkaLogSink functionality."""

    @pytest.fixture
    def kafka_config(self):
        """Create Kafka configuration."""
        return {
            "bootstrap_servers": ["localhost:9092"],
            "topic": "cronos-logs",
            "security_protocol": "PLAINTEXT",
            "acks": "all",
            "retries": 3,
            "batch_size": 16384
        }

    @pytest.fixture
    def kafka_sink(self, kafka_config):
        """Create KafkaLogSink instance."""
        with patch('ai_engine.monitoring.log_aggregation.KafkaProducer') as mock_producer:
            return KafkaLogSink(kafka_config)

    def test_kafka_sink_initialization(self, kafka_sink, kafka_config):
        """Test KafkaLogSink initialization."""
        assert kafka_sink.config == kafka_config
        assert kafka_sink.bootstrap_servers == kafka_config["bootstrap_servers"]
        assert kafka_sink.topic == kafka_config["topic"]

    @pytest.mark.asyncio
    async def test_kafka_sink_send_log_entry(self, kafka_sink):
        """Test sending log entry to Kafka."""
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test log message",
            source=LogSource.APPLICATION
        )
        
        with patch.object(kafka_sink, 'producer') as mock_producer:
            mock_producer.send.return_value = Mock()
            mock_producer.flush.return_value = None
            
            await kafka_sink.send(log_entry)
            
            mock_producer.send.assert_called_once()
            mock_producer.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_kafka_sink_send_batch(self, kafka_sink):
        """Test sending batch of log entries to Kafka."""
        log_entries = [
            LogEntry(timestamp=datetime.now(), level=LogLevel.INFO, message="Message 1"),
            LogEntry(timestamp=datetime.now(), level=LogLevel.WARNING, message="Message 2")
        ]
        
        with patch.object(kafka_sink, 'producer') as mock_producer:
            mock_producer.send.return_value = Mock()
            mock_producer.flush.return_value = None
            
            await kafka_sink.send_batch(log_entries)
            
            assert mock_producer.send.call_count == 2
            mock_producer.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_kafka_sink_connection_error(self, kafka_sink):
        """Test Kafka connection error handling."""
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message"
        )
        
        with patch.object(kafka_sink, 'producer') as mock_producer:
            mock_producer.send.side_effect = Exception("Connection failed")
            
            with pytest.raises(LogAggregationException):
                await kafka_sink.send(log_entry)

    def test_kafka_sink_close(self, kafka_sink):
        """Test closing Kafka producer."""
        with patch.object(kafka_sink, 'producer') as mock_producer:
            mock_producer.close.return_value = None
            
            kafka_sink.close()
            
            mock_producer.close.assert_called_once()


class TestFileLogSink:
    """Test FileLogSink functionality."""

    @pytest.fixture
    def file_config(self):
        """Create file configuration."""
        return {
            "file_path": "/tmp/test_logs.jsonl",
            "rotation_size": 10485760,  # 10MB
            "max_files": 5,
            "format": "jsonl"
        }

    @pytest.fixture
    def file_sink(self, file_config):
        """Create FileLogSink instance."""
        with patch('ai_engine.monitoring.log_aggregation.open') as mock_open:
            return FileLogSink(file_config)

    def test_file_sink_initialization(self, file_sink, file_config):
        """Test FileLogSink initialization."""
        assert file_sink.config == file_config
        assert file_sink.file_path == file_config["file_path"]
        assert file_sink.rotation_size == file_config["rotation_size"]

    @pytest.mark.asyncio
    async def test_file_sink_send_log_entry(self, file_sink):
        """Test sending log entry to file."""
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test log message",
            source=LogSource.APPLICATION
        )
        
        with patch('ai_engine.monitoring.log_aggregation.open') as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            await file_sink.send(log_entry)
            
            mock_file.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_file_sink_send_batch(self, file_sink):
        """Test sending batch of log entries to file."""
        log_entries = [
            LogEntry(timestamp=datetime.now(), level=LogLevel.INFO, message="Message 1"),
            LogEntry(timestamp=datetime.now(), level=LogLevel.WARNING, message="Message 2")
        ]
        
        with patch('ai_engine.monitoring.log_aggregation.open') as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            await file_sink.send_batch(log_entries)
            
            assert mock_file.write.call_count == 2

    def test_file_sink_rotation(self, file_sink):
        """Test file rotation."""
        with patch('ai_engine.monitoring.log_aggregation.os.path.getsize') as mock_getsize, \
             patch('ai_engine.monitoring.log_aggregation.shutil.move') as mock_move:
            
            mock_getsize.return_value = 10485761  # Larger than rotation size
            
            file_sink.check_rotation()
            
            mock_move.assert_called_once()

    def test_file_sink_cleanup_old_files(self, file_sink):
        """Test cleanup of old log files."""
        with patch('ai_engine.monitoring.log_aggregation.glob.glob') as mock_glob, \
             patch('ai_engine.monitoring.log_aggregation.os.remove') as mock_remove:
            
            mock_glob.return_value = [
                "/tmp/test_logs.jsonl.1",
                "/tmp/test_logs.jsonl.2",
                "/tmp/test_logs.jsonl.3",
                "/tmp/test_logs.jsonl.4",
                "/tmp/test_logs.jsonl.5",
                "/tmp/test_logs.jsonl.6"  # This should be removed
            ]
            
            file_sink.cleanup_old_files()
            
            mock_remove.assert_called_once_with("/tmp/test_logs.jsonl.6")


class TestLogAggregator:
    """Test LogAggregator main functionality."""

    @pytest.fixture
    def aggregator_config(self):
        """Create log aggregator configuration."""
        return LogAggregationConfig(
            buffer_size=1000,
            flush_interval=30,
            batch_size=100,
            sinks=["elasticsearch", "kafka", "file"],
            filters=[
                {"levels": ["ERROR", "CRITICAL"]},
                {"sources": ["APPLICATION", "SYSTEM"]}
            ],
            processors=[
                {"type": "timestamp_normalizer"},
                {"type": "field_extractor"}
            ]
        )

    @pytest.fixture
    def log_aggregator(self, aggregator_config):
        """Create LogAggregator instance."""
        with patch('ai_engine.monitoring.log_aggregation.ElasticsearchLogSink'), \
             patch('ai_engine.monitoring.log_aggregation.KafkaLogSink'), \
             patch('ai_engine.monitoring.log_aggregation.FileLogSink'):
            return LogAggregator(aggregator_config)

    def test_log_aggregator_initialization(self, log_aggregator, aggregator_config):
        """Test LogAggregator initialization."""
        assert log_aggregator.config == aggregator_config
        assert log_aggregator.buffer_size == aggregator_config.buffer_size
        assert log_aggregator.flush_interval == aggregator_config.flush_interval
        assert len(log_aggregator.sinks) == 3

    def test_add_log_entry(self, log_aggregator):
        """Test adding log entry to aggregator."""
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test log message"
        )
        
        log_aggregator.add_log(log_entry)
        
        assert len(log_aggregator.buffer) == 1
        assert log_aggregator.buffer[0] == log_entry

    def test_add_log_entry_buffer_full(self, log_aggregator):
        """Test adding log entry when buffer is full."""
        # Fill buffer to capacity
        for i in range(log_aggregator.buffer_size):
            log_entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message=f"Message {i}"
            )
            log_aggregator.add_log(log_entry)
        
        # Add one more entry - should trigger flush
        with patch.object(log_aggregator, 'flush') as mock_flush:
            extra_entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="Extra message"
            )
            log_aggregator.add_log(extra_entry)
            
            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_logs(self, log_aggregator):
        """Test flushing logs to sinks."""
        # Add some log entries
        for i in range(5):
            log_entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message=f"Message {i}"
            )
            log_aggregator.add_log(log_entry)
        
        with patch.object(log_aggregator.sinks[0], 'send_batch') as mock_send:
            await log_aggregator.flush()
            
            mock_send.assert_called_once()
            assert len(log_aggregator.buffer) == 0

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self, log_aggregator):
        """Test flushing empty buffer."""
        with patch.object(log_aggregator.sinks[0], 'send_batch') as mock_send:
            await log_aggregator.flush()
            
            mock_send.assert_not_called()

    def test_add_filter(self, log_aggregator):
        """Test adding filter to aggregator."""
        log_filter = LogFilter(levels=[LogLevel.ERROR])
        log_aggregator.add_filter(log_filter)
        
        assert log_filter in log_aggregator.processor.filters

    def test_remove_filter(self, log_aggregator):
        """Test removing filter from aggregator."""
        log_filter = LogFilter(levels=[LogLevel.ERROR])
        log_aggregator.add_filter(log_filter)
        log_aggregator.remove_filter(log_filter)
        
        assert log_filter not in log_aggregator.processor.filters

    def test_add_sink(self, log_aggregator):
        """Test adding sink to aggregator."""
        with patch('ai_engine.monitoring.log_aggregation.FileLogSink') as mock_sink:
            new_sink = mock_sink.return_value
            log_aggregator.add_sink(new_sink)
            
            assert new_sink in log_aggregator.sinks

    def test_remove_sink(self, log_aggregator):
        """Test removing sink from aggregator."""
        sink_to_remove = log_aggregator.sinks[0]
        log_aggregator.remove_sink(sink_to_remove)
        
        assert sink_to_remove not in log_aggregator.sinks

    @pytest.mark.asyncio
    async def test_start_stop_aggregator(self, log_aggregator):
        """Test starting and stopping aggregator."""
        with patch.object(log_aggregator, 'flush') as mock_flush:
            # Start aggregator
            log_aggregator.start()
            assert log_aggregator.is_running is True
            
            # Stop aggregator
            await log_aggregator.stop()
            assert log_aggregator.is_running is False
            
            # Should flush remaining logs on stop
            mock_flush.assert_called()

    def test_log_aggregator_error_handling(self, log_aggregator):
        """Test log aggregator error handling."""
        # Test with invalid log entry
        with pytest.raises(LogAggregationException):
            log_aggregator.add_log(None)

    def test_log_aggregator_concurrent_access(self, log_aggregator):
        """Test log aggregator concurrent access."""
        def add_logs():
            for i in range(100):
                log_entry = LogEntry(
                    timestamp=datetime.now(),
                    level=LogLevel.INFO,
                    message=f"Concurrent message {i}"
                )
                log_aggregator.add_log(log_entry)
        
        # Run concurrent operations
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_logs) for _ in range(5)]
            [future.result() for future in futures]
        
        # All logs should be added successfully
        assert len(log_aggregator.buffer) == 500

    def test_log_aggregator_metrics(self, log_aggregator):
        """Test log aggregator metrics."""
        # Add some logs
        for i in range(10):
            log_entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message=f"Message {i}"
            )
            log_aggregator.add_log(log_entry)
        
        metrics = log_aggregator.get_metrics()
        
        assert metrics["total_logs"] == 10
        assert metrics["buffer_size"] == 10
        assert metrics["sinks_count"] == 3
        assert metrics["filters_count"] == 2
