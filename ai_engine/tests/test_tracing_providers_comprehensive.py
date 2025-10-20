"""
CRONOS AI Engine - Comprehensive Distributed Tracing Tests

Complete test suite for distributed tracing functionality including
JaegerTracingProvider, ZipkinTracingProvider, and tracing provider factory.
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from typing import Dict, List, Any, Optional

from ai_engine.monitoring.tracing_providers import (
    JaegerConfig,
    ZipkinConfig,
    JaegerTracingProvider,
    ZipkinTracingProvider,
    create_tracing_provider,
)
from ai_engine.core.config import Config
from ai_engine.core.exceptions import ObservabilityException


class TestJaegerConfig:
    """Test JaegerConfig dataclass."""

    def test_jaeger_config_creation(self):
        """Test creating JaegerConfig instance."""
        config = JaegerConfig(
            agent_host="jaeger.example.com",
            agent_port=6832,
            collector_endpoint="https://jaeger.example.com:14268",
            service_name="test-service",
            sampling_rate=0.5,
            max_tag_value_length=2048,
            max_packet_size=32000,
        )

        assert config.agent_host == "jaeger.example.com"
        assert config.agent_port == 6832
        assert config.collector_endpoint == "https://jaeger.example.com:14268"
        assert config.service_name == "test-service"
        assert config.sampling_rate == 0.5
        assert config.max_tag_value_length == 2048
        assert config.max_packet_size == 32000

    def test_jaeger_config_defaults(self):
        """Test JaegerConfig with default values."""
        config = JaegerConfig()

        assert config.agent_host == "localhost"
        assert config.agent_port == 6831
        assert config.collector_endpoint is None
        assert config.service_name == "cronos-ai"
        assert config.sampling_rate == 1.0
        assert config.max_tag_value_length == 1024
        assert config.max_packet_size == 65000


class TestZipkinConfig:
    """Test ZipkinConfig dataclass."""

    def test_zipkin_config_creation(self):
        """Test creating ZipkinConfig instance."""
        config = ZipkinConfig(
            endpoint="https://zipkin.example.com:9411",
            service_name="test-service",
            sampling_rate=0.3,
            max_tag_value_length=2048,
        )

        assert config.endpoint == "https://zipkin.example.com:9411"
        assert config.service_name == "test-service"
        assert config.sampling_rate == 0.3
        assert config.max_tag_value_length == 2048

    def test_zipkin_config_defaults(self):
        """Test ZipkinConfig with default values."""
        config = ZipkinConfig()

        assert config.endpoint == "http://localhost:9411"
        assert config.service_name == "cronos-ai"
        assert config.sampling_rate == 1.0
        assert config.max_tag_value_length == 1024


class TestJaegerTracingProvider:
    """Test JaegerTracingProvider functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.monitoring = Mock()
        config.monitoring.jaeger_agent_host = "localhost"
        config.monitoring.jaeger_agent_port = 6831
        config.monitoring.jaeger_collector_endpoint = "http://localhost:14268"
        config.monitoring.service_name = "test-service"
        config.tracing_sampling_rate = 1.0
        return config

    @pytest.fixture
    def jaeger_provider(self, mock_config):
        """Create JaegerTracingProvider instance."""
        return JaegerTracingProvider(mock_config)

    def test_jaeger_provider_initialization(self, jaeger_provider, mock_config):
        """Test JaegerTracingProvider initialization."""
        assert jaeger_provider.jaeger_config.agent_host == "localhost"
        assert jaeger_provider.jaeger_config.agent_port == 6831
        assert (
            jaeger_provider.jaeger_config.collector_endpoint == "http://localhost:14268"
        )
        assert jaeger_provider.jaeger_config.service_name == "test-service"
        assert jaeger_provider.jaeger_config.sampling_rate == 1.0
        assert jaeger_provider._batch_size == 100
        assert jaeger_provider._flush_interval == 10

    @pytest.mark.asyncio
    async def test_jaeger_provider_initialize(self, jaeger_provider):
        """Test JaegerTracingProvider initialization."""
        with patch("aiohttp.ClientSession") as mock_session:
            await jaeger_provider.initialize()

            assert jaeger_provider._session is not None
            assert jaeger_provider._flush_task is not None
            mock_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_jaeger_provider_initialize_no_collector(self, mock_config):
        """Test JaegerTracingProvider initialization without collector endpoint."""
        mock_config.monitoring.jaeger_collector_endpoint = None
        provider = JaegerTracingProvider(mock_config)

        await provider.initialize()

        assert provider._session is None
        assert provider._flush_task is None

    @pytest.mark.asyncio
    async def test_jaeger_provider_shutdown(self, jaeger_provider):
        """Test JaegerTracingProvider shutdown."""
        # Mock session and flush task
        jaeger_provider._session = AsyncMock()
        jaeger_provider._flush_task = AsyncMock()
        jaeger_provider._flush_task.cancel = Mock()

        with patch.object(jaeger_provider, "flush") as mock_flush:
            await jaeger_provider.shutdown()

            jaeger_provider._flush_task.cancel.assert_called_once()
            jaeger_provider._session.close.assert_called_once()
            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_jaeger_provider_export_span_with_collector(self, jaeger_provider):
        """Test exporting span with collector endpoint."""
        # Mock span
        span = Mock()
        span.span_id = "span123"
        span.trace_id = "trace123"
        span.parent_span_id = "parent123"
        span.operation_name = "test_operation"
        span.start_time = time.time()
        span.duration_ms = 100.0
        span.tags = {"service": "test", "operation": "test_op"}
        span.logs = []
        span.links = []

        with patch.object(jaeger_provider, "_flush_batch") as mock_flush:
            await jaeger_provider.export_span(span)

            assert len(jaeger_provider._batch_queue) == 1
            assert jaeger_provider._batch_queue[0] == span

    @pytest.mark.asyncio
    async def test_jaeger_provider_export_span_without_collector(self, mock_config):
        """Test exporting span without collector endpoint."""
        mock_config.monitoring.jaeger_collector_endpoint = None
        provider = JaegerTracingProvider(mock_config)

        # Mock span
        span = Mock()
        span.span_id = "span123"

        with patch.object(provider, "_convert_to_jaeger_format") as mock_convert:
            await provider.export_span(span)

            mock_convert.assert_called_once_with(span)

    @pytest.mark.asyncio
    async def test_jaeger_provider_export_span_batch_flush(self, jaeger_provider):
        """Test exporting span triggers batch flush."""
        # Fill queue to batch size
        for i in range(jaeger_provider._batch_size):
            span = Mock()
            span.span_id = f"span{i}"
            jaeger_provider._batch_queue.append(span)

        with patch.object(jaeger_provider, "_flush_batch") as mock_flush:
            # Add one more span to trigger flush
            extra_span = Mock()
            extra_span.span_id = "span_extra"

            await jaeger_provider.export_span(extra_span)

            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_jaeger_provider_export_span_error(self, jaeger_provider):
        """Test exporting span with error."""
        # Mock span
        span = Mock()
        span.span_id = "span_error"

        with patch.object(
            jaeger_provider,
            "_convert_to_jaeger_format",
            side_effect=Exception("Conversion error"),
        ):
            with pytest.raises(ObservabilityException):
                await jaeger_provider.export_span(span)

    @pytest.mark.asyncio
    async def test_jaeger_provider_export_trace(self, jaeger_provider):
        """Test exporting complete trace."""
        # Mock trace with spans
        span1 = Mock()
        span1.span_id = "span1"
        span2 = Mock()
        span2.span_id = "span2"

        trace = Mock()
        trace.spans = [span1, span2]

        with patch.object(jaeger_provider, "export_span") as mock_export:
            await jaeger_provider.export_trace(trace)

            assert mock_export.call_count == 2
            mock_export.assert_has_calls([call(span1), call(span2)])

    @pytest.mark.asyncio
    async def test_jaeger_provider_export_trace_error(self, jaeger_provider):
        """Test exporting trace with error."""
        # Mock trace
        span = Mock()
        span.span_id = "span_error"

        trace = Mock()
        trace.spans = [span]

        with patch.object(
            jaeger_provider, "export_span", side_effect=Exception("Export error")
        ):
            with pytest.raises(ObservabilityException):
                await jaeger_provider.export_trace(trace)

    @pytest.mark.asyncio
    async def test_jaeger_provider_flush(self, jaeger_provider):
        """Test flushing spans."""
        # Add some spans
        for i in range(3):
            span = Mock()
            span.span_id = f"span{i}"
            jaeger_provider._batch_queue.append(span)

        with patch.object(jaeger_provider, "_flush_batch") as mock_flush:
            await jaeger_provider.flush()
            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_jaeger_provider_flush_loop(self, jaeger_provider):
        """Test background flush loop."""
        jaeger_provider._session = AsyncMock()
        jaeger_provider._batch_queue.append(Mock())

        with patch.object(jaeger_provider, "_flush_batch") as mock_flush:
            # Start flush loop
            task = asyncio.create_task(jaeger_provider._flush_loop())

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
    async def test_jaeger_provider_flush_batch_success(self, jaeger_provider):
        """Test successful batch flush to Jaeger."""
        # Create test spans
        spans = []
        for i in range(3):
            span = Mock()
            span.span_id = f"span{i}"
            span.trace_id = f"trace{i}"
            span.parent_span_id = f"parent{i}"
            span.operation_name = f"operation{i}"
            span.start_time = time.time()
            span.duration_ms = 100.0
            span.tags = {"service": "test"}
            span.logs = []
            span.links = []
            spans.append(span)

        jaeger_provider._batch_queue.extend(spans)
        jaeger_provider._session = AsyncMock()

        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        jaeger_provider._session.post.return_value.__aenter__.return_value = (
            mock_response
        )

        with patch.object(jaeger_provider, "_convert_to_jaeger_format") as mock_convert:
            mock_convert.return_value = {"span": "data"}

            await jaeger_provider._flush_batch()

            assert len(jaeger_provider._batch_queue) == 0
            jaeger_provider._session.post.assert_called_once()
            assert mock_convert.call_count == 3

    @pytest.mark.asyncio
    async def test_jaeger_provider_flush_batch_failure(self, jaeger_provider):
        """Test batch flush failure handling."""
        span = Mock()
        span.span_id = "span_fail"
        jaeger_provider._batch_queue.append(span)
        jaeger_provider._session = AsyncMock()

        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 500
        jaeger_provider._session.post.return_value.__aenter__.return_value = (
            mock_response
        )

        with patch.object(jaeger_provider, "_convert_to_jaeger_format") as mock_convert:
            mock_convert.return_value = {"span": "data"}

            # Should not raise exception
            await jaeger_provider._flush_batch()

            jaeger_provider._session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_jaeger_provider_flush_batch_exception(self, jaeger_provider):
        """Test batch flush exception handling."""
        span = Mock()
        span.span_id = "span_exception"
        jaeger_provider._batch_queue.append(span)
        jaeger_provider._session = AsyncMock()

        # Mock exception
        jaeger_provider._session.post.side_effect = Exception("Network error")

        with patch.object(jaeger_provider, "_convert_to_jaeger_format") as mock_convert:
            mock_convert.return_value = {"span": "data"}

            # Should not raise exception
            await jaeger_provider._flush_batch()

            jaeger_provider._session.post.assert_called_once()

    def test_jaeger_provider_convert_to_jaeger_format(self, jaeger_provider):
        """Test converting span to Jaeger format."""
        # Mock span
        span = Mock()
        span.trace_id = "trace123"
        span.span_id = "span123"
        span.parent_span_id = "parent123"
        span.operation_name = "test_operation"
        span.start_time = 1234567890.0
        span.duration_ms = 100.0
        span.tags = {"service": "test", "operation": "test_op"}
        span.logs = []
        span.links = []

        jaeger_span = jaeger_provider._convert_to_jaeger_format(span)

        assert jaeger_span["traceId"] == "trace123"
        assert jaeger_span["spanId"] == "span123"
        assert jaeger_span["parentSpanId"] == "parent123"
        assert jaeger_span["operationName"] == "test_operation"
        assert jaeger_span["startTime"] == 1234567890000000  # microseconds
        assert jaeger_span["duration"] == 100000  # nanoseconds
        assert len(jaeger_span["tags"]) == 2
        assert jaeger_span["process"]["serviceName"] == "test-service"

    def test_jaeger_provider_convert_to_jaeger_format_with_logs(self, jaeger_provider):
        """Test converting span with logs to Jaeger format."""
        # Mock span with logs
        log = Mock()
        log.timestamp = 1234567890.0
        log.attributes = {"message": "test log", "level": "info"}

        span = Mock()
        span.trace_id = "trace123"
        span.span_id = "span123"
        span.parent_span_id = None
        span.operation_name = "test_operation"
        span.start_time = 1234567890.0
        span.duration_ms = 100.0
        span.tags = {}
        span.logs = [log]
        span.links = []

        jaeger_span = jaeger_provider._convert_to_jaeger_format(span)

        assert len(jaeger_span["logs"]) == 1
        assert jaeger_span["logs"][0]["timestamp"] == 1234567890000000
        assert len(jaeger_span["logs"][0]["fields"]) == 2

    def test_jaeger_provider_convert_to_jaeger_format_with_links(self, jaeger_provider):
        """Test converting span with links to Jaeger format."""
        # Mock span with links
        link = Mock()
        link.trace_id = "linked_trace"
        link.span_id = "linked_span"

        span = Mock()
        span.trace_id = "trace123"
        span.span_id = "span123"
        span.parent_span_id = None
        span.operation_name = "test_operation"
        span.start_time = 1234567890.0
        span.duration_ms = 100.0
        span.tags = {}
        span.logs = []
        span.links = [link]

        jaeger_span = jaeger_provider._convert_to_jaeger_format(span)

        assert len(jaeger_span["references"]) == 1
        assert jaeger_span["references"][0]["refType"] == "CHILD_OF"
        assert jaeger_span["references"][0]["traceId"] == "linked_trace"
        assert jaeger_span["references"][0]["spanId"] == "linked_span"

    def test_jaeger_provider_convert_to_jaeger_format_tag_truncation(
        self, jaeger_provider
    ):
        """Test tag value truncation in Jaeger format."""
        # Mock span with long tag value
        long_value = "x" * 2000  # Longer than max_tag_value_length
        span = Mock()
        span.trace_id = "trace123"
        span.span_id = "span123"
        span.parent_span_id = None
        span.operation_name = "test_operation"
        span.start_time = 1234567890.0
        span.duration_ms = 100.0
        span.tags = {"long_tag": long_value}
        span.logs = []
        span.links = []

        jaeger_span = jaeger_provider._convert_to_jaeger_format(span)

        assert len(jaeger_span["tags"]) == 1
        assert len(jaeger_span["tags"][0]["value"]) == 1024  # max_tag_value_length


class TestZipkinTracingProvider:
    """Test ZipkinTracingProvider functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.monitoring = Mock()
        config.monitoring.zipkin_endpoint = "http://localhost:9411"
        config.monitoring.service_name = "test-service"
        config.tracing_sampling_rate = 1.0
        return config

    @pytest.fixture
    def zipkin_provider(self, mock_config):
        """Create ZipkinTracingProvider instance."""
        return ZipkinTracingProvider(mock_config)

    def test_zipkin_provider_initialization(self, zipkin_provider, mock_config):
        """Test ZipkinTracingProvider initialization."""
        assert zipkin_provider.zipkin_config.endpoint == "http://localhost:9411"
        assert zipkin_provider.zipkin_config.service_name == "test-service"
        assert zipkin_provider.zipkin_config.sampling_rate == 1.0
        assert zipkin_provider._batch_size == 100
        assert zipkin_provider._flush_interval == 10

    @pytest.mark.asyncio
    async def test_zipkin_provider_initialize(self, zipkin_provider):
        """Test ZipkinTracingProvider initialization."""
        with patch("aiohttp.ClientSession") as mock_session:
            await zipkin_provider.initialize()

            assert zipkin_provider._session is not None
            assert zipkin_provider._flush_task is not None
            mock_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_zipkin_provider_shutdown(self, zipkin_provider):
        """Test ZipkinTracingProvider shutdown."""
        # Mock session and flush task
        zipkin_provider._session = AsyncMock()
        zipkin_provider._flush_task = AsyncMock()
        zipkin_provider._flush_task.cancel = Mock()

        with patch.object(zipkin_provider, "flush") as mock_flush:
            await zipkin_provider.shutdown()

            zipkin_provider._flush_task.cancel.assert_called_once()
            zipkin_provider._session.close.assert_called_once()
            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_zipkin_provider_export_span(self, zipkin_provider):
        """Test exporting span to Zipkin."""
        # Mock span
        span = Mock()
        span.span_id = "span123"

        with patch.object(zipkin_provider, "_flush_batch") as mock_flush:
            await zipkin_provider.export_span(span)

            assert len(zipkin_provider._batch_queue) == 1
            assert zipkin_provider._batch_queue[0] == span

    @pytest.mark.asyncio
    async def test_zipkin_provider_export_span_batch_flush(self, zipkin_provider):
        """Test exporting span triggers batch flush."""
        # Fill queue to batch size
        for i in range(zipkin_provider._batch_size):
            span = Mock()
            span.span_id = f"span{i}"
            zipkin_provider._batch_queue.append(span)

        with patch.object(zipkin_provider, "_flush_batch") as mock_flush:
            # Add one more span to trigger flush
            extra_span = Mock()
            extra_span.span_id = "span_extra"

            await zipkin_provider.export_span(extra_span)

            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_zipkin_provider_export_span_error(self, zipkin_provider):
        """Test exporting span with error."""
        # Mock span
        span = Mock()
        span.span_id = "span_error"

        with patch.object(
            zipkin_provider, "_flush_batch", side_effect=Exception("Flush error")
        ):
            with pytest.raises(ObservabilityException):
                await zipkin_provider.export_span(span)

    @pytest.mark.asyncio
    async def test_zipkin_provider_export_trace(self, zipkin_provider):
        """Test exporting complete trace."""
        # Mock trace with spans
        span1 = Mock()
        span1.span_id = "span1"
        span2 = Mock()
        span2.span_id = "span2"

        trace = Mock()
        trace.spans = [span1, span2]

        with patch.object(zipkin_provider, "export_span") as mock_export:
            await zipkin_provider.export_trace(trace)

            assert mock_export.call_count == 2
            mock_export.assert_has_calls([call(span1), call(span2)])

    @pytest.mark.asyncio
    async def test_zipkin_provider_export_trace_error(self, zipkin_provider):
        """Test exporting trace with error."""
        # Mock trace
        span = Mock()
        span.span_id = "span_error"

        trace = Mock()
        trace.spans = [span]

        with patch.object(
            zipkin_provider, "export_span", side_effect=Exception("Export error")
        ):
            with pytest.raises(ObservabilityException):
                await zipkin_provider.export_trace(trace)

    @pytest.mark.asyncio
    async def test_zipkin_provider_flush_batch_success(self, zipkin_provider):
        """Test successful batch flush to Zipkin."""
        # Create test spans
        spans = []
        for i in range(3):
            span = Mock()
            span.span_id = f"span{i}"
            span.trace_id = f"trace{i}"
            span.parent_span_id = f"parent{i}"
            span.operation_name = f"operation{i}"
            span.start_time = time.time()
            span.duration_ms = 100.0
            span.tags = {"service": "test"}
            span.logs = []
            span.kind = Mock()
            span.kind.value = "internal"
            spans.append(span)

        zipkin_provider._batch_queue.extend(spans)
        zipkin_provider._session = AsyncMock()

        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        zipkin_provider._session.post.return_value.__aenter__.return_value = (
            mock_response
        )

        with patch.object(zipkin_provider, "_convert_to_zipkin_format") as mock_convert:
            mock_convert.return_value = {"span": "data"}

            await zipkin_provider._flush_batch()

            assert len(zipkin_provider._batch_queue) == 0
            zipkin_provider._session.post.assert_called_once()
            assert mock_convert.call_count == 3

    @pytest.mark.asyncio
    async def test_zipkin_provider_flush_batch_failure(self, zipkin_provider):
        """Test batch flush failure handling."""
        span = Mock()
        span.span_id = "span_fail"
        zipkin_provider._batch_queue.append(span)
        zipkin_provider._session = AsyncMock()

        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 500
        zipkin_provider._session.post.return_value.__aenter__.return_value = (
            mock_response
        )

        with patch.object(zipkin_provider, "_convert_to_zipkin_format") as mock_convert:
            mock_convert.return_value = {"span": "data"}

            # Should not raise exception
            await zipkin_provider._flush_batch()

            zipkin_provider._session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_zipkin_provider_flush_batch_exception(self, zipkin_provider):
        """Test batch flush exception handling."""
        span = Mock()
        span.span_id = "span_exception"
        zipkin_provider._batch_queue.append(span)
        zipkin_provider._session = AsyncMock()

        # Mock exception
        zipkin_provider._session.post.side_effect = Exception("Network error")

        with patch.object(zipkin_provider, "_convert_to_zipkin_format") as mock_convert:
            mock_convert.return_value = {"span": "data"}

            # Should not raise exception
            await zipkin_provider._flush_batch()

            zipkin_provider._session.post.assert_called_once()

    def test_zipkin_provider_convert_to_zipkin_format(self, zipkin_provider):
        """Test converting span to Zipkin format."""
        # Mock span
        span = Mock()
        span.trace_id = "trace-123"
        span.span_id = "span-123"
        span.parent_span_id = "parent-123"
        span.operation_name = "test_operation"
        span.start_time = 1234567890.0
        span.duration_ms = 100.0
        span.kind = Mock()
        span.kind.value = "internal"
        span.tags = {"service": "test", "operation": "test_op"}
        span.logs = []

        zipkin_span = zipkin_provider._convert_to_zipkin_format(span)

        assert zipkin_span["traceId"] == "trace123"  # Hyphens removed
        assert zipkin_span["id"] == "span123"  # Hyphens removed
        assert zipkin_span["parentId"] == "parent123"  # Hyphens removed
        assert zipkin_span["name"] == "test_operation"
        assert zipkin_span["timestamp"] == 1234567890000000  # microseconds
        assert zipkin_span["duration"] == 100000  # nanoseconds
        assert zipkin_span["kind"] == "CLIENT"
        assert zipkin_span["localEndpoint"]["serviceName"] == "test-service"
        assert zipkin_span["tags"]["service"] == "test"
        assert zipkin_span["tags"]["operation"] == "test_op"

    def test_zipkin_provider_convert_to_zipkin_format_with_logs(self, zipkin_provider):
        """Test converting span with logs to Zipkin format."""
        # Mock span with logs
        log = Mock()
        log.timestamp = 1234567890.0
        log.name = "test_log"

        span = Mock()
        span.trace_id = "trace-123"
        span.span_id = "span-123"
        span.parent_span_id = None
        span.operation_name = "test_operation"
        span.start_time = 1234567890.0
        span.duration_ms = 100.0
        span.kind = Mock()
        span.kind.value = "server"
        span.tags = {}
        span.logs = [log]

        zipkin_span = zipkin_provider._convert_to_zipkin_format(span)

        assert len(zipkin_span["annotations"]) == 1
        assert zipkin_span["annotations"][0]["timestamp"] == 1234567890000000
        assert zipkin_span["annotations"][0]["value"] == "test_log"

    def test_zipkin_provider_convert_to_zipkin_format_no_parent(self, zipkin_provider):
        """Test converting span without parent to Zipkin format."""
        # Mock span without parent
        span = Mock()
        span.trace_id = "trace-123"
        span.span_id = "span-123"
        span.parent_span_id = None
        span.operation_name = "test_operation"
        span.start_time = 1234567890.0
        span.duration_ms = 100.0
        span.kind = Mock()
        span.kind.value = "internal"
        span.tags = {}
        span.logs = []

        zipkin_span = zipkin_provider._convert_to_zipkin_format(span)

        assert "parentId" not in zipkin_span

    def test_zipkin_provider_map_span_kind(self, zipkin_provider):
        """Test mapping span kinds to Zipkin kinds."""
        kind_mappings = {
            "internal": "CLIENT",
            "server": "SERVER",
            "client": "CLIENT",
            "producer": "PRODUCER",
            "consumer": "CONSUMER",
            "unknown": "CLIENT",  # Default
        }

        for internal_kind, zipkin_kind in kind_mappings.items():
            kind = Mock()
            kind.value = internal_kind

            span = Mock()
            span.trace_id = "trace-123"
            span.span_id = "span-123"
            span.parent_span_id = None
            span.operation_name = "test_operation"
            span.start_time = 1234567890.0
            span.duration_ms = 100.0
            span.kind = kind
            span.tags = {}
            span.logs = []

            zipkin_span = zipkin_provider._convert_to_zipkin_format(span)
            assert zipkin_span["kind"] == zipkin_kind

    def test_zipkin_provider_convert_to_zipkin_format_tag_truncation(
        self, zipkin_provider
    ):
        """Test tag value truncation in Zipkin format."""
        # Mock span with long tag value
        long_value = "x" * 2000  # Longer than max_tag_value_length
        span = Mock()
        span.trace_id = "trace-123"
        span.span_id = "span-123"
        span.parent_span_id = None
        span.operation_name = "test_operation"
        span.start_time = 1234567890.0
        span.duration_ms = 100.0
        span.kind = Mock()
        span.kind.value = "internal"
        span.tags = {"long_tag": long_value}
        span.logs = []

        zipkin_span = zipkin_provider._convert_to_zipkin_format(span)

        assert len(zipkin_span["tags"]["long_tag"]) == 1024  # max_tag_value_length


class TestTracingProviderFactory:
    """Test tracing provider factory function."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.monitoring = Mock()
        config.monitoring.jaeger_agent_host = "localhost"
        config.monitoring.jaeger_agent_port = 6831
        config.monitoring.jaeger_collector_endpoint = "http://localhost:14268"
        config.monitoring.zipkin_endpoint = "http://localhost:9411"
        config.monitoring.service_name = "test-service"
        config.tracing_sampling_rate = 1.0
        return config

    def test_create_tracing_provider_jaeger(self, mock_config):
        """Test creating Jaeger tracing provider."""
        provider = create_tracing_provider(mock_config, "jaeger")

        assert isinstance(provider, JaegerTracingProvider)
        assert provider.jaeger_config.service_name == "test-service"

    def test_create_tracing_provider_zipkin(self, mock_config):
        """Test creating Zipkin tracing provider."""
        provider = create_tracing_provider(mock_config, "zipkin")

        assert isinstance(provider, ZipkinTracingProvider)
        assert provider.zipkin_config.service_name == "test-service"

    def test_create_tracing_provider_inmemory(self, mock_config):
        """Test creating in-memory tracing provider."""
        with patch(
            "ai_engine.monitoring.tracing_providers.InMemoryTracingProvider"
        ) as mock_inmemory:
            provider = create_tracing_provider(mock_config, "inmemory")

            mock_inmemory.assert_called_once_with(mock_config)

    def test_create_tracing_provider_default(self, mock_config):
        """Test creating tracing provider with default type."""
        provider = create_tracing_provider(mock_config)

        assert isinstance(provider, JaegerTracingProvider)

    def test_create_tracing_provider_case_insensitive(self, mock_config):
        """Test creating tracing provider with case insensitive type."""
        provider = create_tracing_provider(mock_config, "JAEGER")

        assert isinstance(provider, JaegerTracingProvider)

    def test_create_tracing_provider_unsupported(self, mock_config):
        """Test creating tracing provider with unsupported type."""
        with pytest.raises(ValueError, match="Unknown tracing provider type"):
            create_tracing_provider(mock_config, "unsupported")


class TestTracingIntegration:
    """Integration tests for tracing components."""

    @pytest.mark.asyncio
    async def test_jaeger_tracing_workflow(self):
        """Test complete Jaeger tracing workflow."""
        mock_config = Mock(spec=Config)
        mock_config.monitoring = Mock()
        mock_config.monitoring.jaeger_agent_host = "localhost"
        mock_config.monitoring.jaeger_agent_port = 6831
        mock_config.monitoring.jaeger_collector_endpoint = "http://localhost:14268"
        mock_config.monitoring.service_name = "test-service"
        config.tracing_sampling_rate = 1.0

        provider = JaegerTracingProvider(mock_config)

        # Mock span
        span = Mock()
        span.span_id = "span123"
        span.trace_id = "trace123"
        span.parent_span_id = None
        span.operation_name = "test_operation"
        span.start_time = time.time()
        span.duration_ms = 100.0
        span.tags = {"service": "test"}
        span.logs = []
        span.links = []

        with patch("aiohttp.ClientSession") as mock_session:
            await provider.initialize()

            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            provider._session.post.return_value.__aenter__.return_value = mock_response

            # Export span
            await provider.export_span(span)

            # Should have queued the span
            assert len(provider._batch_queue) == 1

            # Flush
            await provider.flush()

            # Should have sent to Jaeger
            provider._session.post.assert_called_once()

            await provider.shutdown()

    @pytest.mark.asyncio
    async def test_zipkin_tracing_workflow(self):
        """Test complete Zipkin tracing workflow."""
        mock_config = Mock(spec=Config)
        mock_config.monitoring = Mock()
        mock_config.monitoring.zipkin_endpoint = "http://localhost:9411"
        mock_config.monitoring.service_name = "test-service"
        config.tracing_sampling_rate = 1.0

        provider = ZipkinTracingProvider(mock_config)

        # Mock span
        span = Mock()
        span.span_id = "span-123"
        span.trace_id = "trace-123"
        span.parent_span_id = None
        span.operation_name = "test_operation"
        span.start_time = time.time()
        span.duration_ms = 100.0
        span.kind = Mock()
        span.kind.value = "internal"
        span.tags = {"service": "test"}
        span.logs = []

        with patch("aiohttp.ClientSession") as mock_session:
            await provider.initialize()

            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            provider._session.post.return_value.__aenter__.return_value = mock_response

            # Export span
            await provider.export_span(span)

            # Should have queued the span
            assert len(provider._batch_queue) == 1

            # Flush
            await provider.flush()

            # Should have sent to Zipkin
            provider._session.post.assert_called_once()

            await provider.shutdown()

    @pytest.mark.asyncio
    async def test_tracing_provider_concurrent_spans(self):
        """Test concurrent span export."""
        mock_config = Mock(spec=Config)
        mock_config.monitoring = Mock()
        mock_config.monitoring.jaeger_agent_host = "localhost"
        mock_config.monitoring.jaeger_agent_port = 6831
        mock_config.monitoring.jaeger_collector_endpoint = "http://localhost:14268"
        mock_config.monitoring.service_name = "test-service"
        config.tracing_sampling_rate = 1.0

        provider = JaegerTracingProvider(mock_config)

        async def export_span(span_id):
            span = Mock()
            span.span_id = span_id
            span.trace_id = f"trace_{span_id}"
            span.parent_span_id = None
            span.operation_name = f"operation_{span_id}"
            span.start_time = time.time()
            span.duration_ms = 100.0
            span.tags = {"service": "test"}
            span.logs = []
            span.links = []

            await provider.export_span(span)

        with patch("aiohttp.ClientSession") as mock_session:
            await provider.initialize()

            # Export spans concurrently
            tasks = [export_span(f"span_{i}") for i in range(10)]
            await asyncio.gather(*tasks)

            # Should have queued all spans
            assert len(provider._batch_queue) == 10

            await provider.shutdown()

    def test_tracing_provider_error_handling(self):
        """Test tracing provider error handling."""
        mock_config = Mock(spec=Config)
        mock_config.monitoring = Mock()
        mock_config.monitoring.jaeger_agent_host = "localhost"
        mock_config.monitoring.jaeger_agent_port = 6831
        mock_config.monitoring.jaeger_collector_endpoint = "http://localhost:14268"
        mock_config.monitoring.service_name = "test-service"
        config.tracing_sampling_rate = 1.0

        provider = JaegerTracingProvider(mock_config)

        # Test with invalid span
        invalid_span = None

        with pytest.raises(ObservabilityException):
            asyncio.run(provider.export_span(invalid_span))

    def test_tracing_provider_configuration_validation(self):
        """Test tracing provider configuration validation."""
        # Test with minimal config
        mock_config = Mock(spec=Config)
        mock_config.monitoring = Mock()
        mock_config.monitoring.jaeger_agent_host = "localhost"
        mock_config.monitoring.jaeger_agent_port = 6831
        mock_config.monitoring.jaeger_collector_endpoint = None
        mock_config.monitoring.service_name = "test-service"
        config.tracing_sampling_rate = 0.5

        provider = JaegerTracingProvider(mock_config)

        assert provider.jaeger_config.agent_host == "localhost"
        assert provider.jaeger_config.agent_port == 6831
        assert provider.jaeger_config.collector_endpoint is None
        assert provider.jaeger_config.service_name == "test-service"
        assert provider.jaeger_config.sampling_rate == 0.5

    def test_tracing_provider_span_format_compatibility(self):
        """Test span format compatibility between providers."""
        # Test that both providers can handle the same span format
        mock_config = Mock(spec=Config)
        mock_config.monitoring = Mock()
        mock_config.monitoring.jaeger_agent_host = "localhost"
        mock_config.monitoring.jaeger_agent_port = 6831
        mock_config.monitoring.jaeger_collector_endpoint = "http://localhost:14268"
        mock_config.monitoring.zipkin_endpoint = "http://localhost:9411"
        mock_config.monitoring.service_name = "test-service"
        config.tracing_sampling_rate = 1.0

        jaeger_provider = JaegerTracingProvider(mock_config)
        zipkin_provider = ZipkinTracingProvider(mock_config)

        # Mock span
        span = Mock()
        span.span_id = "span123"
        span.trace_id = "trace123"
        span.parent_span_id = "parent123"
        span.operation_name = "test_operation"
        span.start_time = 1234567890.0
        span.duration_ms = 100.0
        span.kind = Mock()
        span.kind.value = "internal"
        span.tags = {"service": "test", "operation": "test_op"}
        span.logs = []
        span.links = []

        # Both providers should be able to convert the span
        jaeger_format = jaeger_provider._convert_to_jaeger_format(span)
        zipkin_format = zipkin_provider._convert_to_zipkin_format(span)

        assert jaeger_format is not None
        assert zipkin_format is not None

        # Both should have the same operation name
        assert jaeger_format["operationName"] == "test_operation"
        assert zipkin_format["name"] == "test_operation"
