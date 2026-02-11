"""
Tests for Prometheus metrics collection.
Covers MetricsCollector and metric types.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, patch, MagicMock
from prometheus_client.core import CollectorRegistry, REGISTRY

from ai_engine.monitoring.metrics import (
    MetricsCollector,
    MetricType,
    MetricConfig,
)
from ai_engine.core.config import Config


class TestMetricType:
    """Test MetricType enumeration."""

    def test_metric_type_values(self):
        """Test metric type enum values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.SUMMARY.value == "summary"
        assert MetricType.INFO.value == "info"


class TestMetricConfig:
    """Test MetricConfig dataclass."""

    def test_metric_config_creation(self):
        """Test creating metric configuration."""
        config = MetricConfig(
            name="test_metric",
            help="Test metric description",
            labels=["label1", "label2"],
            unit="seconds",
        )

        assert config.name == "test_metric"
        assert config.help == "Test metric description"
        assert config.labels == ["label1", "label2"]
        assert config.unit == "seconds"
        assert config.namespace == "qbitel"

    def test_metric_config_defaults(self):
        """Test metric config default values."""
        config = MetricConfig(name="default_metric", help="Default metric")

        assert config.labels == []
        assert config.buckets is None
        assert config.unit == ""
        assert config.namespace == "qbitel"

    def test_metric_config_with_buckets(self):
        """Test metric config with histogram buckets."""
        buckets = [0.1, 0.5, 1.0, 5.0, 10.0]
        config = MetricConfig(
            name="histogram_metric", help="Histogram metric", buckets=buckets
        )

        assert config.buckets == buckets


class TestMetricsCollector:
    """Test MetricsCollector class."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear Prometheus registry before each test."""
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except Exception:
                pass
        yield
        # Cleanup after test
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except Exception:
                pass

    @pytest.fixture
    def collector(self):
        """Create metrics collector instance."""
        config = Config()
        return MetricsCollector(config)

    def test_collector_initialization(self, collector):
        """Test metrics collector initialization."""
        assert collector.metrics == {}
        assert collector.custom_metrics == {}
        assert collector.collection_enabled is True
        assert collector.collection_interval == 15

    def test_register_counter_metric(self, collector):
        """Test registering a counter metric."""
        config = MetricConfig(
            name="requests_total",
            help="Total number of requests",
            labels=["method", "endpoint"],
        )

        metric = collector.register_metric(config, MetricType.COUNTER)

        assert metric is not None
        assert "requests_total" in collector.metrics
        assert hasattr(metric, "inc")

    def test_register_histogram_metric(self, collector):
        """Test registering a histogram metric."""
        config = MetricConfig(
            name="request_duration",
            help="Request duration in seconds",
            buckets=[0.1, 0.5, 1.0, 5.0],
        )

        metric = collector.register_metric(config, MetricType.HISTOGRAM)

        assert metric is not None
        assert "request_duration" in collector.metrics
        assert hasattr(metric, "observe")

    def test_register_gauge_metric(self, collector):
        """Test registering a gauge metric."""
        config = MetricConfig(
            name="active_connections", help="Number of active connections"
        )

        metric = collector.register_metric(config, MetricType.GAUGE)

        assert metric is not None
        assert "active_connections" in collector.metrics
        assert hasattr(metric, "set")

    def test_register_summary_metric(self, collector):
        """Test registering a summary metric."""
        config = MetricConfig(name="response_size", help="Response size in bytes")

        metric = collector.register_metric(config, MetricType.SUMMARY)

        assert metric is not None
        assert "response_size" in collector.metrics
        assert hasattr(metric, "observe")

    def test_register_info_metric(self, collector):
        """Test registering an info metric."""
        config = MetricConfig(name="app_info", help="Application information")

        metric = collector.register_metric(config, MetricType.INFO)

        assert metric is not None
        assert "app_info" in collector.metrics
        assert hasattr(metric, "info")

    def test_register_duplicate_metric(self, collector):
        """Test registering duplicate metric returns existing."""
        config = MetricConfig(name="duplicate_metric", help="Duplicate metric")

        metric1 = collector.register_metric(config, MetricType.COUNTER)
        metric2 = collector.register_metric(config, MetricType.COUNTER)

        assert metric1 is metric2

    def test_get_metric(self, collector):
        """Test getting registered metric."""
        config = MetricConfig(name="test_counter", help="Test counter")

        registered = collector.register_metric(config, MetricType.COUNTER)
        retrieved = collector.get_metric("test_counter")

        assert retrieved is registered

    def test_get_nonexistent_metric(self, collector):
        """Test getting non-existent metric returns None."""
        metric = collector.get_metric("nonexistent")
        assert metric is None

    def test_increment_counter(self, collector):
        """Test incrementing counter metric."""
        config = MetricConfig(
            name="test_counter", help="Test counter", labels=["status"]
        )

        collector.register_metric(config, MetricType.COUNTER)
        collector.increment_counter(
            "test_counter", labels={"status": "success"}, value=5.0
        )

        # Metric should have been incremented (can't easily verify value in Prometheus)

    def test_increment_counter_without_labels(self, collector):
        """Test incrementing counter without labels."""
        config = MetricConfig(name="simple_counter", help="Simple counter")

        collector.register_metric(config, MetricType.COUNTER)
        collector.increment_counter("simple_counter", value=1.0)

        # Should not raise error

    def test_set_gauge(self, collector):
        """Test setting gauge value."""
        config = MetricConfig(name="test_gauge", help="Test gauge", labels=["instance"])

        collector.register_metric(config, MetricType.GAUGE)
        collector.set_gauge("test_gauge", 42.0, labels={"instance": "server1"})

        # Gauge should be set (can't easily verify value)

    def test_set_gauge_without_labels(self, collector):
        """Test setting gauge without labels."""
        config = MetricConfig(name="simple_gauge", help="Simple gauge")

        collector.register_metric(config, MetricType.GAUGE)
        collector.set_gauge("simple_gauge", 100.0)

        # Should not raise error

    def test_observe_histogram(self, collector):
        """Test observing histogram value."""
        config = MetricConfig(
            name="test_histogram",
            help="Test histogram",
            labels=["operation"],
            buckets=[0.1, 0.5, 1.0, 5.0],
        )

        collector.register_metric(config, MetricType.HISTOGRAM)
        collector.observe_histogram(
            "test_histogram", 0.75, labels={"operation": "query"}
        )

        # Histogram should have observation

    def test_observe_histogram_without_labels(self, collector):
        """Test observing histogram without labels."""
        config = MetricConfig(name="simple_histogram", help="Simple histogram")

        collector.register_metric(config, MetricType.HISTOGRAM)
        collector.observe_histogram("simple_histogram", 1.5)

        # Should not raise error

    def test_increment_nonexistent_counter(self, collector):
        """Test incrementing non-existent counter."""
        # Should not raise error, just not increment anything
        collector.increment_counter("nonexistent", value=1.0)

    def test_set_nonexistent_gauge(self, collector):
        """Test setting non-existent gauge."""
        # Should not raise error, just not set anything
        collector.set_gauge("nonexistent", 42.0)

    def test_observe_nonexistent_histogram(self, collector):
        """Test observing non-existent histogram."""
        # Should not raise error, just not observe anything
        collector.observe_histogram("nonexistent", 1.0)


class TestMetricsCollectorEdgeCases:
    """Test edge cases for metrics collector."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear Prometheus registry before each test."""
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except Exception:
                pass
        yield
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except Exception:
                pass

    @pytest.fixture
    def collector(self):
        """Create metrics collector instance."""
        config = Config()
        return MetricsCollector(config)

    def test_counter_increment_large_value(self, collector):
        """Test incrementing counter with large value."""
        config = MetricConfig(name="large_counter", help="Large counter")
        collector.register_metric(config, MetricType.COUNTER)

        collector.increment_counter("large_counter", value=1000000.0)
        # Should not raise error

    def test_gauge_negative_value(self, collector):
        """Test setting gauge to negative value."""
        config = MetricConfig(name="negative_gauge", help="Negative gauge")
        collector.register_metric(config, MetricType.GAUGE)

        collector.set_gauge("negative_gauge", -100.0)
        # Gauges can be negative

    def test_histogram_zero_value(self, collector):
        """Test observing zero in histogram."""
        config = MetricConfig(name="zero_histogram", help="Zero histogram")
        collector.register_metric(config, MetricType.HISTOGRAM)

        collector.observe_histogram("zero_histogram", 0.0)
        # Should not raise error

    def test_histogram_extreme_value(self, collector):
        """Test observing extreme value in histogram."""
        config = MetricConfig(
            name="extreme_histogram",
            help="Extreme histogram",
            buckets=[0.1, 1.0, 10.0, 100.0],
        )
        collector.register_metric(config, MetricType.HISTOGRAM)

        collector.observe_histogram("extreme_histogram", 999999.0)
        # Should be in +Inf bucket

    def test_metric_with_many_labels(self, collector):
        """Test metric with many labels."""
        labels = [f"label{i}" for i in range(10)]
        config = MetricConfig(
            name="many_labels", help="Metric with many labels", labels=labels
        )

        metric = collector.register_metric(config, MetricType.COUNTER)
        assert metric is not None

    def test_counter_with_label_values(self, collector):
        """Test counter with specific label values."""
        config = MetricConfig(
            name="labeled_counter",
            help="Labeled counter",
            labels=["method", "status", "endpoint"],
        )

        collector.register_metric(config, MetricType.COUNTER)
        collector.increment_counter(
            "labeled_counter",
            labels={"method": "GET", "status": "200", "endpoint": "/api/users"},
        )

    def test_histogram_default_buckets(self, collector):
        """Test histogram with default buckets."""
        config = MetricConfig(
            name="default_buckets",
            help="Default buckets histogram",
            # No buckets specified, should use defaults
        )

        metric = collector.register_metric(config, MetricType.HISTOGRAM)
        assert metric is not None

    def test_custom_namespace(self, collector):
        """Test metric with custom namespace."""
        config = MetricConfig(
            name="custom_metric",
            help="Custom namespace metric",
            namespace="custom_namespace",
        )

        metric = collector.register_metric(config, MetricType.COUNTER)
        assert metric is not None


class TestMetricsCollectorThreadSafety:
    """Test thread safety of metrics collector."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear Prometheus registry before each test."""
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except Exception:
                pass
        yield
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except Exception:
                pass

    @pytest.fixture
    def collector(self):
        """Create metrics collector instance."""
        config = Config()
        return MetricsCollector(config)

    def test_concurrent_metric_registration(self, collector):
        """Test registering metrics concurrently."""
        import threading

        def register_metric(i):
            config = MetricConfig(
                name=f"concurrent_metric_{i}", help=f"Concurrent metric {i}"
            )
            collector.register_metric(config, MetricType.COUNTER)

        threads = [
            threading.Thread(target=register_metric, args=(i,)) for i in range(10)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(collector.metrics) == 10

    def test_concurrent_counter_increment(self, collector):
        """Test incrementing counter concurrently."""
        import threading

        config = MetricConfig(name="concurrent_counter", help="Concurrent counter")
        collector.register_metric(config, MetricType.COUNTER)

        def increment():
            for _ in range(100):
                collector.increment_counter("concurrent_counter")

        threads = [threading.Thread(target=increment) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
