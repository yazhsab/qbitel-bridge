"""
CRONOS AI Engine - APM Integration Tests

Comprehensive test suite for Application Performance Monitoring integration.
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional

from ai_engine.monitoring.apm_integration import (
    APMProvider,
    NewRelicAPM,
    DataDogAPM,
    AppDynamicsAPM,
    CustomAPM,
    APMMetrics,
    APMTrace,
    APMSpan,
    APMConfig,
    APMException,
)


class TestAPMMetrics:
    """Test APMMetrics dataclass."""

    def test_apm_metrics_creation(self):
        """Test creating APMMetrics instance."""
        metrics = APMMetrics(
            name="test_metric",
            value=123.45,
            timestamp=datetime.now(),
            tags={"service": "test_service", "environment": "test"},
            unit="ms",
            type="counter"
        )
        
        assert metrics.name == "test_metric"
        assert metrics.value == 123.45
        assert metrics.tags == {"service": "test_service", "environment": "test"}
        assert metrics.unit == "ms"
        assert metrics.type == "counter"

    def test_apm_metrics_serialization(self):
        """Test APMMetrics serialization."""
        metrics = APMMetrics(
            name="test_metric",
            value=123.45,
            timestamp=datetime.now(),
            tags={"service": "test_service"},
            unit="ms",
            type="counter"
        )
        
        serialized = metrics.to_dict()
        assert serialized["name"] == "test_metric"
        assert serialized["value"] == 123.45
        assert serialized["tags"] == {"service": "test_service"}

    def test_apm_metrics_deserialization(self):
        """Test APMMetrics deserialization."""
        data = {
            "name": "test_metric",
            "value": 123.45,
            "timestamp": datetime.now().isoformat(),
            "tags": {"service": "test_service"},
            "unit": "ms",
            "type": "counter"
        }
        
        metrics = APMMetrics.from_dict(data)
        assert metrics.name == "test_metric"
        assert metrics.value == 123.45
        assert metrics.tags == {"service": "test_service"}


class TestAPMSpan:
    """Test APMSpan dataclass."""

    def test_apm_span_creation(self):
        """Test creating APMSpan instance."""
        span = APMSpan(
            span_id="span123",
            trace_id="trace123",
            parent_span_id="parent123",
            operation_name="test_operation",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=100),
            duration_ms=100.0,
            tags={"service": "test_service", "method": "GET"},
            logs=[{"timestamp": datetime.now(), "message": "test log"}],
            error=None
        )
        
        assert span.span_id == "span123"
        assert span.trace_id == "trace123"
        assert span.parent_span_id == "parent123"
        assert span.operation_name == "test_operation"
        assert span.duration_ms == 100.0
        assert span.tags == {"service": "test_service", "method": "GET"}

    def test_apm_span_with_error(self):
        """Test APMSpan with error."""
        error = {
            "type": "ValueError",
            "message": "Test error",
            "stack_trace": "Traceback..."
        }
        
        span = APMSpan(
            span_id="span456",
            trace_id="trace456",
            parent_span_id=None,
            operation_name="error_operation",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=50),
            duration_ms=50.0,
            tags={"service": "test_service"},
            logs=[],
            error=error
        )
        
        assert span.error == error
        assert span.has_error is True

    def test_apm_span_without_error(self):
        """Test APMSpan without error."""
        span = APMSpan(
            span_id="span789",
            trace_id="trace789",
            parent_span_id=None,
            operation_name="success_operation",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=75),
            duration_ms=75.0,
            tags={"service": "test_service"},
            logs=[],
            error=None
        )
        
        assert span.error is None
        assert span.has_error is False

    def test_apm_span_add_log(self):
        """Test adding log to APMSpan."""
        span = APMSpan(
            span_id="span101",
            trace_id="trace101",
            parent_span_id=None,
            operation_name="log_operation",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=25),
            duration_ms=25.0,
            tags={"service": "test_service"},
            logs=[],
            error=None
        )
        
        span.add_log("Test log message", {"level": "info"})
        
        assert len(span.logs) == 1
        assert span.logs[0]["message"] == "Test log message"
        assert span.logs[0]["level"] == "info"

    def test_apm_span_add_tag(self):
        """Test adding tag to APMSpan."""
        span = APMSpan(
            span_id="span102",
            trace_id="trace102",
            parent_span_id=None,
            operation_name="tag_operation",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=30),
            duration_ms=30.0,
            tags={"service": "test_service"},
            logs=[],
            error=None
        )
        
        span.add_tag("custom_tag", "custom_value")
        
        assert span.tags["custom_tag"] == "custom_value"


class TestAPMTrace:
    """Test APMTrace dataclass."""

    def test_apm_trace_creation(self):
        """Test creating APMTrace instance."""
        spans = [
            APMSpan(
                span_id="span1",
                trace_id="trace123",
                parent_span_id=None,
                operation_name="root_operation",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=100),
                duration_ms=100.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            ),
            APMSpan(
                span_id="span2",
                trace_id="trace123",
                parent_span_id="span1",
                operation_name="child_operation",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=50),
                duration_ms=50.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            )
        ]
        
        trace = APMTrace(
            trace_id="trace123",
            spans=spans,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=100),
            duration_ms=100.0,
            service_name="test_service",
            tags={"environment": "test"}
        )
        
        assert trace.trace_id == "trace123"
        assert len(trace.spans) == 2
        assert trace.duration_ms == 100.0
        assert trace.service_name == "test_service"

    def test_apm_trace_root_span(self):
        """Test getting root span from APMTrace."""
        spans = [
            APMSpan(
                span_id="span1",
                trace_id="trace123",
                parent_span_id=None,
                operation_name="root_operation",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=100),
                duration_ms=100.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            ),
            APMSpan(
                span_id="span2",
                trace_id="trace123",
                parent_span_id="span1",
                operation_name="child_operation",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=50),
                duration_ms=50.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            )
        ]
        
        trace = APMTrace(
            trace_id="trace123",
            spans=spans,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=100),
            duration_ms=100.0,
            service_name="test_service",
            tags={}
        )
        
        root_span = trace.get_root_span()
        assert root_span is not None
        assert root_span.span_id == "span1"
        assert root_span.operation_name == "root_operation"

    def test_apm_trace_has_error(self):
        """Test checking if trace has error."""
        spans = [
            APMSpan(
                span_id="span1",
                trace_id="trace123",
                parent_span_id=None,
                operation_name="root_operation",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=100),
                duration_ms=100.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            ),
            APMSpan(
                span_id="span2",
                trace_id="trace123",
                parent_span_id="span1",
                operation_name="error_operation",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=50),
                duration_ms=50.0,
                tags={"service": "test_service"},
                logs=[],
                error={"type": "ValueError", "message": "Test error"}
            )
        ]
        
        trace = APMTrace(
            trace_id="trace123",
            spans=spans,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=100),
            duration_ms=100.0,
            service_name="test_service",
            tags={}
        )
        
        assert trace.has_error is True

    def test_apm_trace_get_spans_by_operation(self):
        """Test getting spans by operation name."""
        spans = [
            APMSpan(
                span_id="span1",
                trace_id="trace123",
                parent_span_id=None,
                operation_name="database_query",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=100),
                duration_ms=100.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            ),
            APMSpan(
                span_id="span2",
                trace_id="trace123",
                parent_span_id="span1",
                operation_name="database_query",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=50),
                duration_ms=50.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            ),
            APMSpan(
                span_id="span3",
                trace_id="trace123",
                parent_span_id="span1",
                operation_name="http_request",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=75),
                duration_ms=75.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            )
        ]
        
        trace = APMTrace(
            trace_id="trace123",
            spans=spans,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=100),
            duration_ms=100.0,
            service_name="test_service",
            tags={}
        )
        
        db_spans = trace.get_spans_by_operation("database_query")
        assert len(db_spans) == 2
        
        http_spans = trace.get_spans_by_operation("http_request")
        assert len(http_spans) == 1


class TestAPMConfig:
    """Test APMConfig dataclass."""

    def test_apm_config_creation(self):
        """Test creating APMConfig instance."""
        config = APMConfig(
            provider="newrelic",
            api_key="test_api_key",
            app_name="test_app",
            environment="test",
            enabled=True,
            sample_rate=1.0,
            buffer_size=1000,
            flush_interval=30,
            tags={"service": "test_service", "version": "1.0.0"}
        )
        
        assert config.provider == "newrelic"
        assert config.api_key == "test_api_key"
        assert config.app_name == "test_app"
        assert config.environment == "test"
        assert config.enabled is True
        assert config.sample_rate == 1.0
        assert config.buffer_size == 1000
        assert config.flush_interval == 30

    def test_apm_config_validation(self):
        """Test APMConfig validation."""
        # Valid config
        config = APMConfig(
            provider="newrelic",
            api_key="test_api_key",
            app_name="test_app",
            environment="test",
            enabled=True,
            sample_rate=0.5,
            buffer_size=500,
            flush_interval=15
        )
        
        assert config.is_valid() is True
        
        # Invalid config - missing required fields
        invalid_config = APMConfig(
            provider="",
            api_key="",
            app_name="",
            environment="",
            enabled=True,
            sample_rate=0.5,
            buffer_size=500,
            flush_interval=15
        )
        
        assert invalid_config.is_valid() is False

    def test_apm_config_sample_rate_validation(self):
        """Test APMConfig sample rate validation."""
        # Valid sample rates
        config1 = APMConfig(
            provider="newrelic",
            api_key="test_api_key",
            app_name="test_app",
            environment="test",
            enabled=True,
            sample_rate=0.0,
            buffer_size=500,
            flush_interval=15
        )
        assert config1.is_valid() is True
        
        config2 = APMConfig(
            provider="newrelic",
            api_key="test_api_key",
            app_name="test_app",
            environment="test",
            enabled=True,
            sample_rate=1.0,
            buffer_size=500,
            flush_interval=15
        )
        assert config2.is_valid() is True
        
        # Invalid sample rates
        config3 = APMConfig(
            provider="newrelic",
            api_key="test_api_key",
            app_name="test_app",
            environment="test",
            enabled=True,
            sample_rate=-0.1,
            buffer_size=500,
            flush_interval=15
        )
        assert config3.is_valid() is False
        
        config4 = APMConfig(
            provider="newrelic",
            api_key="test_api_key",
            app_name="test_app",
            environment="test",
            enabled=True,
            sample_rate=1.1,
            buffer_size=500,
            flush_interval=15
        )
        assert config4.is_valid() is False


class TestAPMProvider:
    """Test base APMProvider class."""

    @pytest.fixture
    def apm_provider(self):
        """Create APMProvider instance."""
        config = APMConfig(
            provider="test",
            api_key="test_key",
            app_name="test_app",
            environment="test",
            enabled=True
        )
        return APMProvider(config)

    def test_apm_provider_initialization(self, apm_provider):
        """Test APMProvider initialization."""
        assert apm_provider is not None
        assert apm_provider.config is not None
        assert apm_provider.enabled is True

    def test_apm_provider_not_implemented_methods(self, apm_provider):
        """Test that APMProvider raises NotImplementedError for abstract methods."""
        with pytest.raises(NotImplementedError):
            apm_provider.send_metrics([])
        
        with pytest.raises(NotImplementedError):
            apm_provider.send_trace(Mock())
        
        with pytest.raises(NotImplementedError):
            apm_provider.send_spans([])
        
        with pytest.raises(NotImplementedError):
            apm_provider.flush()

    def test_apm_provider_enable_disable(self, apm_provider):
        """Test enabling and disabling APM provider."""
        assert apm_provider.enabled is True
        
        apm_provider.disable()
        assert apm_provider.enabled is False
        
        apm_provider.enable()
        assert apm_provider.enabled is True

    def test_apm_provider_add_global_tags(self, apm_provider):
        """Test adding global tags."""
        apm_provider.add_global_tag("service", "test_service")
        apm_provider.add_global_tag("version", "1.0.0")
        
        assert apm_provider.global_tags["service"] == "test_service"
        assert apm_provider.global_tags["version"] == "1.0.0"

    def test_apm_provider_remove_global_tag(self, apm_provider):
        """Test removing global tags."""
        apm_provider.add_global_tag("service", "test_service")
        apm_provider.add_global_tag("version", "1.0.0")
        
        apm_provider.remove_global_tag("version")
        
        assert "service" in apm_provider.global_tags
        assert "version" not in apm_provider.global_tags


class TestNewRelicAPM:
    """Test NewRelicAPM implementation."""

    @pytest.fixture
    def newrelic_config(self):
        """Create NewRelic APM configuration."""
        return APMConfig(
            provider="newrelic",
            api_key="test_newrelic_key",
            app_name="test_app",
            environment="test",
            enabled=True,
            sample_rate=1.0
        )

    @pytest.fixture
    def newrelic_apm(self, newrelic_config):
        """Create NewRelicAPM instance."""
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            return NewRelicAPM(newrelic_config)

    def test_newrelic_apm_initialization(self, newrelic_apm, newrelic_config):
        """Test NewRelicAPM initialization."""
        assert newrelic_apm.config == newrelic_config
        assert newrelic_apm.api_key == "test_newrelic_key"
        assert newrelic_apm.app_name == "test_app"

    def test_newrelic_apm_send_metrics(self, newrelic_apm):
        """Test sending metrics to NewRelic."""
        metrics = [
            APMMetrics(
                name="test.metric",
                value=123.45,
                timestamp=datetime.now(),
                tags={"service": "test_service"},
                unit="ms",
                type="counter"
            )
        ]
        
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 202
            mock_requests.post.return_value = mock_response
            
            newrelic_apm.send_metrics(metrics)
            
            mock_requests.post.assert_called_once()

    def test_newrelic_apm_send_trace(self, newrelic_apm):
        """Test sending trace to NewRelic."""
        spans = [
            APMSpan(
                span_id="span1",
                trace_id="trace1",
                parent_span_id=None,
                operation_name="test_operation",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=100),
                duration_ms=100.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            )
        ]
        
        trace = APMTrace(
            trace_id="trace1",
            spans=spans,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=100),
            duration_ms=100.0,
            service_name="test_service",
            tags={}
        )
        
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 202
            mock_requests.post.return_value = mock_response
            
            newrelic_apm.send_trace(trace)
            
            mock_requests.post.assert_called_once()

    def test_newrelic_apm_send_spans(self, newrelic_apm):
        """Test sending spans to NewRelic."""
        spans = [
            APMSpan(
                span_id="span1",
                trace_id="trace1",
                parent_span_id=None,
                operation_name="test_operation",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=100),
                duration_ms=100.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            )
        ]
        
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 202
            mock_requests.post.return_value = mock_response
            
            newrelic_apm.send_spans(spans)
            
            mock_requests.post.assert_called_once()

    def test_newrelic_apm_flush(self, newrelic_apm):
        """Test flushing data to NewRelic."""
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 202
            mock_requests.post.return_value = mock_response
            
            newrelic_apm.flush()
            
            # Should flush any buffered data
            mock_requests.post.assert_called()

    def test_newrelic_apm_error_handling(self, newrelic_apm):
        """Test NewRelic APM error handling."""
        metrics = [
            APMMetrics(
                name="test.metric",
                value=123.45,
                timestamp=datetime.now(),
                tags={"service": "test_service"},
                unit="ms",
                type="counter"
            )
        ]
        
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            mock_requests.post.side_effect = Exception("Network error")
            
            # Should not raise exception
            newrelic_apm.send_metrics(metrics)


class TestDataDogAPM:
    """Test DataDogAPM implementation."""

    @pytest.fixture
    def datadog_config(self):
        """Create DataDog APM configuration."""
        return APMConfig(
            provider="datadog",
            api_key="test_datadog_key",
            app_name="test_app",
            environment="test",
            enabled=True,
            sample_rate=1.0
        )

    @pytest.fixture
    def datadog_apm(self, datadog_config):
        """Create DataDogAPM instance."""
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            return DataDogAPM(datadog_config)

    def test_datadog_apm_initialization(self, datadog_apm, datadog_config):
        """Test DataDogAPM initialization."""
        assert datadog_apm.config == datadog_config
        assert datadog_apm.api_key == "test_datadog_key"
        assert datadog_apm.app_name == "test_app"

    def test_datadog_apm_send_metrics(self, datadog_apm):
        """Test sending metrics to DataDog."""
        metrics = [
            APMMetrics(
                name="test.metric",
                value=123.45,
                timestamp=datetime.now(),
                tags=["service:test_service"],
                unit="ms",
                type="counter"
            )
        ]
        
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 202
            mock_requests.post.return_value = mock_response
            
            datadog_apm.send_metrics(metrics)
            
            mock_requests.post.assert_called_once()

    def test_datadog_apm_send_trace(self, datadog_apm):
        """Test sending trace to DataDog."""
        spans = [
            APMSpan(
                span_id="span1",
                trace_id="trace1",
                parent_span_id=None,
                operation_name="test_operation",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=100),
                duration_ms=100.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            )
        ]
        
        trace = APMTrace(
            trace_id="trace1",
            spans=spans,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=100),
            duration_ms=100.0,
            service_name="test_service",
            tags={}
        )
        
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 202
            mock_requests.post.return_value = mock_response
            
            datadog_apm.send_trace(trace)
            
            mock_requests.post.assert_called_once()

    def test_datadog_apm_flush(self, datadog_apm):
        """Test flushing data to DataDog."""
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 202
            mock_requests.post.return_value = mock_response
            
            datadog_apm.flush()
            
            mock_requests.post.assert_called()


class TestAppDynamicsAPM:
    """Test AppDynamicsAPM implementation."""

    @pytest.fixture
    def appdynamics_config(self):
        """Create AppDynamics APM configuration."""
        return APMConfig(
            provider="appdynamics",
            api_key="test_appdynamics_key",
            app_name="test_app",
            environment="test",
            enabled=True,
            sample_rate=1.0
        )

    @pytest.fixture
    def appdynamics_apm(self, appdynamics_config):
        """Create AppDynamicsAPM instance."""
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            return AppDynamicsAPM(appdynamics_config)

    def test_appdynamics_apm_initialization(self, appdynamics_apm, appdynamics_config):
        """Test AppDynamicsAPM initialization."""
        assert appdynamics_apm.config == appdynamics_config
        assert appdynamics_apm.api_key == "test_appdynamics_key"
        assert appdynamics_apm.app_name == "test_app"

    def test_appdynamics_apm_send_metrics(self, appdynamics_apm):
        """Test sending metrics to AppDynamics."""
        metrics = [
            APMMetrics(
                name="test.metric",
                value=123.45,
                timestamp=datetime.now(),
                tags={"service": "test_service"},
                unit="ms",
                type="counter"
            )
        ]
        
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_requests.post.return_value = mock_response
            
            appdynamics_apm.send_metrics(metrics)
            
            mock_requests.post.assert_called_once()

    def test_appdynamics_apm_send_trace(self, appdynamics_apm):
        """Test sending trace to AppDynamics."""
        spans = [
            APMSpan(
                span_id="span1",
                trace_id="trace1",
                parent_span_id=None,
                operation_name="test_operation",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=100),
                duration_ms=100.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            )
        ]
        
        trace = APMTrace(
            trace_id="trace1",
            spans=spans,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=100),
            duration_ms=100.0,
            service_name="test_service",
            tags={}
        )
        
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_requests.post.return_value = mock_response
            
            appdynamics_apm.send_trace(trace)
            
            mock_requests.post.assert_called_once()

    def test_appdynamics_apm_flush(self, appdynamics_apm):
        """Test flushing data to AppDynamics."""
        with patch('ai_engine.monitoring.apm_integration.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_requests.post.return_value = mock_response
            
            appdynamics_apm.flush()
            
            mock_requests.post.assert_called()


class TestCustomAPM:
    """Test CustomAPM implementation."""

    @pytest.fixture
    def custom_config(self):
        """Create Custom APM configuration."""
        return APMConfig(
            provider="custom",
            api_key="test_custom_key",
            app_name="test_app",
            environment="test",
            enabled=True,
            sample_rate=1.0
        )

    @pytest.fixture
    def custom_apm(self, custom_config):
        """Create CustomAPM instance."""
        return CustomAPM(custom_config)

    def test_custom_apm_initialization(self, custom_apm, custom_config):
        """Test CustomAPM initialization."""
        assert custom_apm.config == custom_config
        assert custom_apm.api_key == "test_custom_key"
        assert custom_apm.app_name == "test_app"

    def test_custom_apm_send_metrics(self, custom_apm):
        """Test sending metrics to custom APM."""
        metrics = [
            APMMetrics(
                name="test.metric",
                value=123.45,
                timestamp=datetime.now(),
                tags={"service": "test_service"},
                unit="ms",
                type="counter"
            )
        ]
        
        # Custom APM should implement its own logic
        custom_apm.send_metrics(metrics)
        
        # Verify metrics were processed (custom implementation)
        assert len(custom_apm.metrics_buffer) == 1

    def test_custom_apm_send_trace(self, custom_apm):
        """Test sending trace to custom APM."""
        spans = [
            APMSpan(
                span_id="span1",
                trace_id="trace1",
                parent_span_id=None,
                operation_name="test_operation",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=100),
                duration_ms=100.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            )
        ]
        
        trace = APMTrace(
            trace_id="trace1",
            spans=spans,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=100),
            duration_ms=100.0,
            service_name="test_service",
            tags={}
        )
        
        custom_apm.send_trace(trace)
        
        # Verify trace was processed
        assert len(custom_apm.traces_buffer) == 1

    def test_custom_apm_send_spans(self, custom_apm):
        """Test sending spans to custom APM."""
        spans = [
            APMSpan(
                span_id="span1",
                trace_id="trace1",
                parent_span_id=None,
                operation_name="test_operation",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(milliseconds=100),
                duration_ms=100.0,
                tags={"service": "test_service"},
                logs=[],
                error=None
            )
        ]
        
        custom_apm.send_spans(spans)
        
        # Verify spans were processed
        assert len(custom_apm.spans_buffer) == 1

    def test_custom_apm_flush(self, custom_apm):
        """Test flushing data in custom APM."""
        # Add some data to buffers
        custom_apm.metrics_buffer.append(Mock())
        custom_apm.traces_buffer.append(Mock())
        custom_apm.spans_buffer.append(Mock())
        
        custom_apm.flush()
        
        # Buffers should be cleared after flush
        assert len(custom_apm.metrics_buffer) == 0
        assert len(custom_apm.traces_buffer) == 0
        assert len(custom_apm.spans_buffer) == 0


class TestAPMIntegration:
    """Integration tests for APM components."""

    def test_apm_provider_factory(self):
        """Test APM provider factory."""
        from ai_engine.monitoring.apm_integration import create_apm_provider
        
        # Test NewRelic provider
        newrelic_config = APMConfig(
            provider="newrelic",
            api_key="test_key",
            app_name="test_app",
            environment="test",
            enabled=True
        )
        
        with patch('ai_engine.monitoring.apm_integration.NewRelicAPM'):
            provider = create_apm_provider(newrelic_config)
            assert isinstance(provider, NewRelicAPM)
        
        # Test DataDog provider
        datadog_config = APMConfig(
            provider="datadog",
            api_key="test_key",
            app_name="test_app",
            environment="test",
            enabled=True
        )
        
        with patch('ai_engine.monitoring.apm_integration.DataDogAPM'):
            provider = create_apm_provider(datadog_config)
            assert isinstance(provider, DataDogAPM)
        
        # Test AppDynamics provider
        appdynamics_config = APMConfig(
            provider="appdynamics",
            api_key="test_key",
            app_name="test_app",
            environment="test",
            enabled=True
        )
        
        with patch('ai_engine.monitoring.apm_integration.AppDynamicsAPM'):
            provider = create_apm_provider(appdynamics_config)
            assert isinstance(provider, AppDynamicsAPM)
        
        # Test Custom provider
        custom_config = APMConfig(
            provider="custom",
            api_key="test_key",
            app_name="test_app",
            environment="test",
            enabled=True
        )
        
        provider = create_apm_provider(custom_config)
        assert isinstance(provider, CustomAPM)

    def test_apm_provider_unsupported(self):
        """Test APM provider factory with unsupported provider."""
        from ai_engine.monitoring.apm_integration import create_apm_provider
        
        unsupported_config = APMConfig(
            provider="unsupported",
            api_key="test_key",
            app_name="test_app",
            environment="test",
            enabled=True
        )
        
        with pytest.raises(APMException):
            create_apm_provider(unsupported_config)

    def test_apm_metrics_aggregation(self):
        """Test APM metrics aggregation."""
        metrics = [
            APMMetrics(
                name="response_time",
                value=100.0,
                timestamp=datetime.now(),
                tags={"service": "api"},
                unit="ms",
                type="histogram"
            ),
            APMMetrics(
                name="response_time",
                value=150.0,
                timestamp=datetime.now(),
                tags={"service": "api"},
                unit="ms",
                type="histogram"
            ),
            APMMetrics(
                name="response_time",
                value=200.0,
                timestamp=datetime.now(),
                tags={"service": "api"},
                unit="ms",
                type="histogram"
            )
        ]
        
        # Test aggregation logic
        aggregated = {}
        for metric in metrics:
            key = f"{metric.name}:{':'.join(f'{k}={v}' for k, v in metric.tags.items())}"
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(metric.value)
        
        assert len(aggregated) == 1
        assert len(aggregated[list(aggregated.keys())[0]]) == 3

    def test_apm_trace_span_relationship(self):
        """Test APM trace and span relationships."""
        # Create parent span
        parent_span = APMSpan(
            span_id="parent_span",
            trace_id="trace123",
            parent_span_id=None,
            operation_name="parent_operation",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=200),
            duration_ms=200.0,
            tags={"service": "test_service"},
            logs=[],
            error=None
        )
        
        # Create child span
        child_span = APMSpan(
            span_id="child_span",
            trace_id="trace123",
            parent_span_id="parent_span",
            operation_name="child_operation",
            start_time=datetime.now() + timedelta(milliseconds=50),
            end_time=datetime.now() + timedelta(milliseconds=150),
            duration_ms=100.0,
            tags={"service": "test_service"},
            logs=[],
            error=None
        )
        
        # Create trace
        trace = APMTrace(
            trace_id="trace123",
            spans=[parent_span, child_span],
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=200),
            duration_ms=200.0,
            service_name="test_service",
            tags={}
        )
        
        # Verify relationships
        assert trace.get_root_span().span_id == "parent_span"
        assert child_span.parent_span_id == "parent_span"
        assert parent_span.parent_span_id is None

    def test_apm_error_handling(self):
        """Test APM error handling."""
        config = APMConfig(
            provider="newrelic",
            api_key="invalid_key",
            app_name="test_app",
            environment="test",
            enabled=True
        )
        
        with patch('ai_engine.monitoring.apm_integration.NewRelicAPM'):
            apm = NewRelicAPM(config)
            
            # Test with invalid data
            with pytest.raises(APMException):
                apm.send_metrics(None)
            
            with pytest.raises(APMException):
                apm.send_trace(None)
            
            with pytest.raises(APMException):
                apm.send_spans(None)

    def test_apm_performance_impact(self):
        """Test APM performance impact."""
        config = APMConfig(
            provider="custom",
            api_key="test_key",
            app_name="test_app",
            environment="test",
            enabled=True,
            sample_rate=0.1  # 10% sampling
        )
        
        apm = CustomAPM(config)
        
        # Test sampling
        sampled_count = 0
        total_count = 1000
        
        for i in range(total_count):
            if apm.should_sample():
                sampled_count += 1
        
        # Should be approximately 10% (allowing for variance)
        assert 50 <= sampled_count <= 150

    def test_apm_concurrent_operations(self):
        """Test APM concurrent operations."""
        config = APMConfig(
            provider="custom",
            api_key="test_key",
            app_name="test_app",
            environment="test",
            enabled=True
        )
        
        apm = CustomAPM(config)
        
        def send_metrics():
            metrics = [
                APMMetrics(
                    name="test.metric",
                    value=123.45,
                    timestamp=datetime.now(),
                    tags={"service": "test_service"},
                    unit="ms",
                    type="counter"
                )
            ]
            apm.send_metrics(metrics)
        
        # Run concurrent operations
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_metrics) for _ in range(10)]
            [future.result() for future in futures]
        
        # All operations should complete successfully
        assert len(apm.metrics_buffer) == 10
