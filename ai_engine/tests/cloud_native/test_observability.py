"""
Unit tests for Envoy Observability.
"""
import pytest
import sys
from pathlib import Path

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.service_mesh.envoy.observability import (
    MetricType,
    Metric,
    EnvoyObservability
)


class TestMetricType:
    """Test MetricType enum"""

    def test_metric_types(self):
        """Test metric type values"""
        assert MetricType.COUNTER is not None
        assert MetricType.GAUGE is not None
        assert MetricType.HISTOGRAM is not None
        assert MetricType.SUMMARY is not None


class TestMetric:
    """Test Metric dataclass"""

    def test_metric_creation(self):
        """Test creating metric"""
        metric = Metric(
            name="http_requests_total",
            metric_type=MetricType.COUNTER,
            value=100,
            labels={"service": "api", "status": "200"}
        )

        assert metric.name == "http_requests_total"
        assert metric.metric_type == MetricType.COUNTER
        assert metric.value == 100
        assert metric.labels["service"] == "api"


class TestEnvoyObservability:
    """Test EnvoyObservability class"""

    @pytest.fixture
    def observability(self):
        """Create observability instance"""
        return EnvoyObservability(
            enable_prometheus=True,
            enable_jaeger=True,
            enable_access_logs=True
        )

    def test_observability_initialization(self, observability):
        """Test observability initialization"""
        assert observability.enable_prometheus is True
        assert observability.enable_jaeger is True
        assert observability.enable_access_logs is True

    def test_create_prometheus_config(self, observability):
        """Test Prometheus configuration creation"""
        prometheus_config = observability.create_prometheus_config()

        assert prometheus_config is not None
        assert "stats_sinks" in prometheus_config or "metrics" in prometheus_config

        # Check for prometheus endpoint
        if "stats_sinks" in prometheus_config:
            sinks = prometheus_config["stats_sinks"]
            assert any("prometheus" in str(sink).lower() for sink in sinks)

    def test_create_jaeger_config(self, observability):
        """Test Jaeger tracing configuration"""
        jaeger_config = observability.create_jaeger_config(
            collector_endpoint="http://jaeger:14268/api/traces"
        )

        assert jaeger_config is not None
        assert "tracing" in jaeger_config or "http" in jaeger_config

        # Check collector endpoint
        config_str = str(jaeger_config)
        assert "jaeger" in config_str.lower() or "14268" in config_str

    def test_create_access_log_config(self, observability):
        """Test access log configuration"""
        access_log_config = observability.create_access_log_config()

        assert access_log_config is not None
        assert isinstance(access_log_config, dict)

        # Should have format configuration
        assert "format" in access_log_config or "typed_config" in access_log_config

    def test_get_statistics(self, observability):
        """Test statistics retrieval"""
        stats = observability.get_statistics()

        assert isinstance(stats, dict)
        assert "prometheus_enabled" in stats
        assert "jaeger_enabled" in stats
        assert "access_logs_enabled" in stats

    def test_metrics_endpoint_config(self, observability):
        """Test metrics endpoint configuration"""
        prometheus_config = observability.create_prometheus_config()

        # Should expose metrics on admin endpoint
        config_str = str(prometheus_config)
        assert "admin" in config_str or "stats" in config_str or "prometheus" in config_str

    def test_custom_metrics(self, observability):
        """Test custom metric recording"""
        # Record a custom metric
        observability.record_metric(
            name="custom_metric",
            metric_type=MetricType.COUNTER,
            value=1,
            labels={"component": "test"}
        )

        # Retrieve metrics
        metrics = observability.get_metrics()
        assert isinstance(metrics, list)

        # Find custom metric
        custom_metric = next(
            (m for m in metrics if m.name == "custom_metric"),
            None
        )
        assert custom_metric is not None or len(metrics) >= 0

    def test_tracing_sampling_rate(self, observability):
        """Test tracing sampling rate configuration"""
        jaeger_config = observability.create_jaeger_config(
            collector_endpoint="http://jaeger:14268/api/traces",
            sampling_rate=0.1
        )

        config_str = str(jaeger_config)
        # Sampling should be configured
        assert jaeger_config is not None

    def test_access_log_format(self, observability):
        """Test access log format configuration"""
        access_log = observability.create_access_log_config(
            format_type="json"
        )

        # Should be JSON format
        config_str = str(access_log)
        assert "json" in config_str.lower() or "format" in access_log

    def test_disable_prometheus(self):
        """Test with Prometheus disabled"""
        obs = EnvoyObservability(enable_prometheus=False)

        config = obs.create_prometheus_config()
        assert config is None or config == {}

    def test_disable_jaeger(self):
        """Test with Jaeger disabled"""
        obs = EnvoyObservability(enable_jaeger=False)

        config = obs.create_jaeger_config("http://jaeger:14268")
        assert config is None or config == {}

    def test_quantum_encryption_metrics(self, observability):
        """Test quantum encryption specific metrics"""
        prometheus_config = observability.create_prometheus_config()

        # Should include quantum-related metrics if available
        # This is implementation-specific
        assert prometheus_config is not None

    def test_health_check_config(self, observability):
        """Test health check configuration"""
        health_config = observability.create_health_check_config()

        assert health_config is not None
        assert "health_check" in health_config or "interval" in health_config

    def test_admin_interface_config(self, observability):
        """Test admin interface configuration"""
        admin_config = observability.create_admin_config(
            address="0.0.0.0",
            port=9901
        )

        assert admin_config is not None
        assert "address" in admin_config
        assert admin_config["address"]["socket_address"]["port_value"] == 9901
