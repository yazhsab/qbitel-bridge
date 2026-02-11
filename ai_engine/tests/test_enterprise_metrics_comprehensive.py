"""
QBITEL Engine - Comprehensive Enterprise Metrics Tests

Complete test suite for enterprise metrics functionality including
MetricsCollector, PrometheusExporter, AlertManager, HealthChecker,
and EnterpriseMetrics coordinator.
"""

import pytest
import asyncio
import time
import json
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from typing import Dict, List, Any, Optional
from collections import deque

from ai_engine.monitoring.enterprise_metrics import (
    MetricType,
    AlertSeverity,
    MetricPoint,
    Alert,
    HealthStatus,
    MetricsCollector,
    PrometheusExporter,
    AlertManager,
    HealthChecker,
    EnterpriseMetrics,
    get_enterprise_metrics,
    shutdown_enterprise_metrics,
    record_metric,
    increment_counter,
    metrics_context,
)


class TestMetricPoint:
    """Test MetricPoint dataclass."""

    def test_metric_point_creation(self):
        """Test creating MetricPoint instance."""
        point = MetricPoint(
            name="test_metric",
            value=123.45,
            timestamp=time.time(),
            labels={"service": "test_service", "environment": "test"},
            metric_type=MetricType.GAUGE,
        )

        assert point.name == "test_metric"
        assert point.value == 123.45
        assert point.labels == {"service": "test_service", "environment": "test"}
        assert point.metric_type == MetricType.GAUGE

    def test_metric_point_defaults(self):
        """Test MetricPoint with default values."""
        point = MetricPoint(name="default_metric", value=100.0, timestamp=time.time())

        assert point.labels == {}
        assert point.metric_type == MetricType.GAUGE

    def test_metric_point_types(self):
        """Test all metric types."""
        types = [
            MetricType.COUNTER,
            MetricType.GAUGE,
            MetricType.HISTOGRAM,
            MetricType.SUMMARY,
        ]

        for metric_type in types:
            point = MetricPoint(
                name=f"test_{metric_type.value}",
                value=50.0,
                timestamp=time.time(),
                metric_type=metric_type,
            )

            assert point.metric_type == metric_type


class TestAlert:
    """Test Alert dataclass."""

    def test_alert_creation(self):
        """Test creating Alert instance."""
        alert = Alert(
            name="test_alert",
            severity=AlertSeverity.WARNING,
            message="Test alert message",
            timestamp=time.time(),
            labels={"service": "test_service"},
            resolved=False,
        )

        assert alert.name == "test_alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Test alert message"
        assert alert.labels == {"service": "test_service"}
        assert alert.resolved is False
        assert alert.resolved_timestamp is None

    def test_alert_resolved(self):
        """Test resolved alert."""
        alert = Alert(
            name="resolved_alert",
            severity=AlertSeverity.CRITICAL,
            message="Critical alert",
            timestamp=time.time(),
            resolved=True,
            resolved_timestamp=time.time() + 60,
        )

        assert alert.resolved is True
        assert alert.resolved_timestamp is not None

    def test_alert_severities(self):
        """Test all alert severities."""
        severities = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.CRITICAL,
            AlertSeverity.FATAL,
        ]

        for severity in severities:
            alert = Alert(
                name=f"test_{severity.value}",
                severity=severity,
                message=f"Test {severity.value} alert",
                timestamp=time.time(),
            )

            assert alert.severity == severity


class TestHealthStatus:
    """Test HealthStatus dataclass."""

    def test_health_status_creation(self):
        """Test creating HealthStatus instance."""
        status = HealthStatus(
            status="healthy",
            timestamp=time.time(),
            components={"cpu": True, "memory": True, "disk": False},
            metrics={"cpu_usage": 25.5, "memory_usage": 60.0},
            alerts=[],
        )

        assert status.status == "healthy"
        assert status.components == {"cpu": True, "memory": True, "disk": False}
        assert status.metrics == {"cpu_usage": 25.5, "memory_usage": 60.0}
        assert status.alerts == []

    def test_health_status_with_alerts(self):
        """Test HealthStatus with alerts."""
        alert = Alert(
            name="test_alert",
            severity=AlertSeverity.WARNING,
            message="Test alert",
            timestamp=time.time(),
        )

        status = HealthStatus(
            status="degraded",
            timestamp=time.time(),
            components={"cpu": True, "memory": False},
            metrics={"cpu_usage": 90.0},
            alerts=[alert],
        )

        assert status.status == "degraded"
        assert len(status.alerts) == 1
        assert status.alerts[0] == alert


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    @pytest.fixture
    def metrics_collector(self):
        """Create MetricsCollector instance."""
        return MetricsCollector(flush_interval=1.0, max_points=1000)

    def test_metrics_collector_initialization(self, metrics_collector):
        """Test MetricsCollector initialization."""
        assert metrics_collector.flush_interval == 1.0
        assert metrics_collector.max_points == 1000
        assert metrics_collector._running is False
        assert metrics_collector._flush_task is None

    def test_metrics_collector_start_stop(self, metrics_collector):
        """Test starting and stopping metrics collector."""
        assert metrics_collector._running is False

        metrics_collector.start()
        assert metrics_collector._running is True

        # Test stop
        with patch.object(metrics_collector, "_flush_task") as mock_task:
            mock_task.cancel = Mock()
            asyncio.run(metrics_collector.stop())
            assert metrics_collector._running is False

    def test_metrics_collector_register_custom_metric(self, metrics_collector):
        """Test registering custom metric collector."""

        def custom_collector():
            return 42.0

        metrics_collector.register_custom_metric("custom_metric", custom_collector)

        assert "custom_metric" in metrics_collector._custom_metrics
        assert metrics_collector._custom_metrics["custom_metric"] == custom_collector

    def test_metrics_collector_increment_counter(self, metrics_collector):
        """Test incrementing counter metric."""
        metrics_collector.increment_counter("test_counter", 5.0, {"service": "test"})

        assert "test_counter" in metrics_collector._metrics
        assert len(metrics_collector._metrics["test_counter"]) == 1

        point = metrics_collector._metrics["test_counter"][0]
        assert point.name == "test_counter"
        assert point.value == 5.0
        assert point.metric_type == MetricType.COUNTER
        assert point.labels == {"service": "test"}

    def test_metrics_collector_set_gauge(self, metrics_collector):
        """Test setting gauge metric."""
        metrics_collector.set_gauge("test_gauge", 100.0, {"environment": "test"})

        assert "test_gauge" in metrics_collector._metrics
        assert len(metrics_collector._metrics["test_gauge"]) == 1

        point = metrics_collector._metrics["test_gauge"][0]
        assert point.name == "test_gauge"
        assert point.value == 100.0
        assert point.metric_type == MetricType.GAUGE
        assert point.labels == {"environment": "test"}

    def test_metrics_collector_record_histogram(self, metrics_collector):
        """Test recording histogram metric."""
        metrics_collector.record_histogram(
            "test_histogram", 150.0, {"operation": "test"}
        )

        assert "test_histogram" in metrics_collector._metrics
        assert len(metrics_collector._metrics["test_histogram"]) == 1

        point = metrics_collector._metrics["test_histogram"][0]
        assert point.name == "test_histogram"
        assert point.value == 150.0
        assert point.metric_type == MetricType.HISTOGRAM
        assert point.labels == {"operation": "test"}

    def test_metrics_collector_max_points_limit(self, metrics_collector):
        """Test max points limit."""
        # Add more points than max_points
        for i in range(1500):
            metrics_collector.set_gauge("test_metric", float(i))

        # Should only keep max_points
        assert len(metrics_collector._metrics["test_metric"]) == 1000

    def test_metrics_collector_get_metrics_snapshot(self, metrics_collector):
        """Test getting metrics snapshot."""
        metrics_collector.set_gauge("gauge1", 100.0)
        metrics_collector.increment_counter("counter1", 5.0)
        metrics_collector.record_histogram("histogram1", 200.0)

        snapshot = metrics_collector.get_metrics_snapshot()

        assert "gauge1" in snapshot
        assert "counter1" in snapshot
        assert "histogram1" in snapshot
        assert len(snapshot["gauge1"]) == 1
        assert len(snapshot["counter1"]) == 1
        assert len(snapshot["histogram1"]) == 1

    def test_metrics_collector_get_latest_values(self, metrics_collector):
        """Test getting latest values."""
        metrics_collector.set_gauge("test_gauge", 100.0)
        metrics_collector.set_gauge("test_gauge", 200.0)
        metrics_collector.increment_counter("test_counter", 5.0)

        latest = metrics_collector.get_latest_values()

        assert latest["test_gauge"] == 200.0
        assert latest["test_counter"] == 5.0

    @pytest.mark.asyncio
    async def test_metrics_collector_collect_system_metrics(self, metrics_collector):
        """Test system metrics collection."""
        with (
            patch("psutil.cpu_percent") as mock_cpu,
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.disk_usage") as mock_disk,
            patch("psutil.net_io_counters") as mock_net,
            patch("psutil.Process") as mock_process,
        ):

            # Mock system metrics
            mock_cpu.return_value = 25.5
            mock_memory.return_value = Mock(
                percent=60.0, used=1024 * 1024 * 1024, available=1024 * 1024 * 1024
            )
            mock_disk.return_value = Mock(
                used=1024 * 1024 * 1024, total=1024 * 1024 * 1024 * 10
            )
            mock_net.return_value = Mock(
                bytes_sent=1000, bytes_recv=2000, packets_sent=10, packets_recv=20
            )
            mock_process.return_value = Mock(
                cpu_percent=Mock(return_value=15.0),
                memory_info=Mock(
                    return_value=Mock(rss=512 * 1024 * 1024, vms=1024 * 1024 * 1024)
                ),
                num_threads=Mock(return_value=8),
                num_fds=Mock(return_value=50),
            )

            metrics_collector._collect_system_metrics()

            latest = metrics_collector.get_latest_values()
            assert "system_cpu_usage_percent" in latest
            assert "system_memory_usage_bytes" in latest
            assert "system_memory_usage_percent" in latest
            assert "system_disk_usage_bytes" in latest
            assert "system_network_bytes_sent" in latest
            assert "process_cpu_percent" in latest

    @pytest.mark.asyncio
    async def test_metrics_collector_collect_custom_metrics(self, metrics_collector):
        """Test custom metrics collection."""

        def custom_metric1():
            return 42.0

        def custom_metric2():
            return 84.0

        metrics_collector.register_custom_metric("custom1", custom_metric1)
        metrics_collector.register_custom_metric("custom2", custom_metric2)

        metrics_collector._collect_custom_metrics()

        latest = metrics_collector.get_latest_values()
        assert latest["custom_custom1"] == 42.0
        assert latest["custom_custom2"] == 84.0

    @pytest.mark.asyncio
    async def test_metrics_collector_collect_custom_metrics_error(
        self, metrics_collector
    ):
        """Test custom metrics collection with error."""

        def error_metric():
            raise Exception("Custom metric error")

        metrics_collector.register_custom_metric("error_metric", error_metric)

        # Should not raise exception
        metrics_collector._collect_custom_metrics()

        # Should not have the metric
        latest = metrics_collector.get_latest_values()
        assert "custom_error_metric" not in latest

    @pytest.mark.asyncio
    async def test_metrics_collector_flush_loop(self, metrics_collector):
        """Test metrics flush loop."""
        metrics_collector._running = True

        with (
            patch.object(metrics_collector, "_collect_system_metrics") as mock_system,
            patch.object(metrics_collector, "_collect_custom_metrics") as mock_custom,
            patch.object(metrics_collector, "_flush_metrics") as mock_flush,
        ):

            # Start flush loop
            task = asyncio.create_task(metrics_collector._flush_loop())

            # Wait a bit for the loop to run
            await asyncio.sleep(0.1)

            # Cancel the task
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should have called collection methods
            mock_system.assert_called()
            mock_custom.assert_called()
            mock_flush.assert_called()


class TestPrometheusExporter:
    """Test PrometheusExporter functionality."""

    @pytest.fixture
    def prometheus_exporter(self):
        """Create PrometheusExporter instance."""
        return PrometheusExporter(port=8000, host="0.0.0.0")

    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        collector = Mock()
        collector.get_metrics_snapshot.return_value = {
            "test_gauge": [
                MetricPoint(
                    name="test_gauge",
                    value=100.0,
                    timestamp=time.time(),
                    labels={"service": "test"},
                    metric_type=MetricType.GAUGE,
                )
            ],
            "test_counter": [
                MetricPoint(
                    name="test_counter",
                    value=5.0,
                    timestamp=time.time(),
                    labels={"operation": "test"},
                    metric_type=MetricType.COUNTER,
                )
            ],
        }
        return collector

    def test_prometheus_exporter_initialization(self, prometheus_exporter):
        """Test PrometheusExporter initialization."""
        assert prometheus_exporter.port == 8000
        assert prometheus_exporter.host == "0.0.0.0"
        assert prometheus_exporter.metrics_collector is None
        assert prometheus_exporter._server is None
        assert prometheus_exporter._app is None

    def test_prometheus_exporter_set_metrics_collector(
        self, prometheus_exporter, mock_metrics_collector
    ):
        """Test setting metrics collector."""
        prometheus_exporter.set_metrics_collector(mock_metrics_collector)

        assert prometheus_exporter.metrics_collector == mock_metrics_collector

    def test_prometheus_exporter_format_prometheus_metrics(
        self, prometheus_exporter, mock_metrics_collector
    ):
        """Test formatting Prometheus metrics."""
        prometheus_exporter.set_metrics_collector(mock_metrics_collector)

        metrics_text = prometheus_exporter._format_prometheus_metrics()

        assert "# TYPE test_gauge gauge" in metrics_text
        assert "# TYPE test_counter counter" in metrics_text
        assert 'test_gauge{service="test"} 100.0' in metrics_text
        assert 'test_counter{operation="test"} 5.0' in metrics_text

    def test_prometheus_exporter_format_prometheus_metrics_no_collector(
        self, prometheus_exporter
    ):
        """Test formatting metrics without collector."""
        metrics_text = prometheus_exporter._format_prometheus_metrics()

        assert metrics_text == ""

    def test_prometheus_exporter_get_prometheus_type(self, prometheus_exporter):
        """Test getting Prometheus type."""
        assert prometheus_exporter._get_prometheus_type(MetricType.COUNTER) == "counter"
        assert prometheus_exporter._get_prometheus_type(MetricType.GAUGE) == "gauge"
        assert (
            prometheus_exporter._get_prometheus_type(MetricType.HISTOGRAM)
            == "histogram"
        )
        assert prometheus_exporter._get_prometheus_type(MetricType.SUMMARY) == "summary"

    @pytest.mark.asyncio
    async def test_prometheus_exporter_metrics_handler(
        self, prometheus_exporter, mock_metrics_collector
    ):
        """Test metrics HTTP handler."""
        prometheus_exporter.set_metrics_collector(mock_metrics_collector)

        mock_request = Mock()

        with patch("ai_engine.monitoring.enterprise_metrics.web") as mock_web:
            mock_web.Response.return_value = "test_response"

            response = await prometheus_exporter.metrics_handler(mock_request)

            assert response == "test_response"
            mock_web.Response.assert_called_once()

    @pytest.mark.asyncio
    async def test_prometheus_exporter_metrics_handler_error(self, prometheus_exporter):
        """Test metrics handler with error."""
        mock_request = Mock()

        with patch("ai_engine.monitoring.enterprise_metrics.web") as mock_web:
            mock_web.Response.return_value = "error_response"

            response = await prometheus_exporter.metrics_handler(mock_request)

            assert response == "error_response"
            mock_web.Response.assert_called_once_with(
                status=500, text="Internal server error"
            )

    @pytest.mark.asyncio
    async def test_prometheus_exporter_start(self, prometheus_exporter):
        """Test starting Prometheus exporter."""
        with patch("ai_engine.monitoring.enterprise_metrics.web") as mock_web:
            mock_app = Mock()
            mock_runner = Mock()
            mock_site = Mock()

            mock_web.Application.return_value = mock_app
            mock_web.AppRunner.return_value = mock_runner
            mock_web.TCPSite.return_value = mock_site

            mock_runner.setup = AsyncMock()
            mock_site.start = AsyncMock()

            await prometheus_exporter.start()

            mock_web.Application.assert_called_once()
            mock_app.router.add_get.assert_called_once_with(
                "/metrics", prometheus_exporter.metrics_handler
            )
            mock_runner.setup.assert_called_once()
            mock_site.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_prometheus_exporter_start_import_error(self, prometheus_exporter):
        """Test starting exporter with import error."""
        with patch(
            "ai_engine.monitoring.enterprise_metrics.web",
            side_effect=ImportError("aiohttp not available"),
        ):
            # Should not raise exception
            await prometheus_exporter.start()

    @pytest.mark.asyncio
    async def test_prometheus_exporter_stop(self, prometheus_exporter):
        """Test stopping Prometheus exporter."""
        mock_server = AsyncMock()
        prometheus_exporter._server = mock_server

        await prometheus_exporter.stop()

        mock_server.close.assert_called_once()


class TestAlertManager:
    """Test AlertManager functionality."""

    @pytest.fixture
    def alert_manager(self):
        """Create AlertManager instance."""
        return AlertManager(max_alerts=100)

    def test_alert_manager_initialization(self, alert_manager):
        """Test AlertManager initialization."""
        assert alert_manager.max_alerts == 100
        assert len(alert_manager._alerts) == 0
        assert len(alert_manager._active_alerts) == 0
        assert len(alert_manager._alert_rules) == 0
        assert len(alert_manager._alert_handlers) == 0

    def test_alert_manager_register_alert_rule(self, alert_manager):
        """Test registering alert rule."""

        def condition(metrics):
            return metrics.get("cpu_usage", 0) > 90

        alert_manager.register_alert_rule(
            "high_cpu", condition, AlertSeverity.WARNING, "CPU usage is high"
        )

        assert "high_cpu" in alert_manager._alert_rules
        rule = alert_manager._alert_rules["high_cpu"]
        assert rule["severity"] == AlertSeverity.WARNING
        assert rule["message"] == "CPU usage is high"
        assert rule["condition"]({"cpu_usage": 95}) is True
        assert rule["condition"]({"cpu_usage": 50}) is False

    def test_alert_manager_register_alert_rule_default_message(self, alert_manager):
        """Test registering alert rule with default message."""

        def condition(metrics):
            return True

        alert_manager.register_alert_rule("test_rule", condition)

        rule = alert_manager._alert_rules["test_rule"]
        assert rule["message"] == "Alert condition triggered: test_rule"

    def test_alert_manager_add_alert_handler(self, alert_manager):
        """Test adding alert handler."""

        def handler(alert):
            pass

        alert_manager.add_alert_handler(handler)

        assert handler in alert_manager._alert_handlers

    def test_alert_manager_check_alert_conditions_new_alert(self, alert_manager):
        """Test checking alert conditions - new alert."""

        def condition(metrics):
            return metrics.get("cpu_usage", 0) > 90

        alert_manager.register_alert_rule("high_cpu", condition, AlertSeverity.WARNING)

        metrics = {"cpu_usage": 95}
        alert_manager.check_alert_conditions(metrics)

        assert "high_cpu" in alert_manager._active_alerts
        assert len(alert_manager._alerts) == 1

        alert = alert_manager._active_alerts["high_cpu"]
        assert alert.name == "high_cpu"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.resolved is False

    def test_alert_manager_check_alert_conditions_resolve_alert(self, alert_manager):
        """Test checking alert conditions - resolve alert."""

        def condition(metrics):
            return metrics.get("cpu_usage", 0) > 90

        alert_manager.register_alert_rule("high_cpu", condition, AlertSeverity.WARNING)

        # Trigger alert
        metrics = {"cpu_usage": 95}
        alert_manager.check_alert_conditions(metrics)

        assert "high_cpu" in alert_manager._active_alerts

        # Resolve alert
        metrics = {"cpu_usage": 50}
        alert_manager.check_alert_conditions(metrics)

        assert "high_cpu" not in alert_manager._active_alerts
        assert len(alert_manager._alerts) == 1

        alert = alert_manager._alerts[0]
        assert alert.resolved is True
        assert alert.resolved_timestamp is not None

    def test_alert_manager_check_alert_conditions_handler(self, alert_manager):
        """Test alert handler execution."""
        handler_calls = []

        def handler(alert):
            handler_calls.append(alert)

        alert_manager.add_alert_handler(handler)

        def condition(metrics):
            return metrics.get("cpu_usage", 0) > 90

        alert_manager.register_alert_rule("high_cpu", condition, AlertSeverity.WARNING)

        # Trigger alert
        metrics = {"cpu_usage": 95}
        alert_manager.check_alert_conditions(metrics)

        assert len(handler_calls) == 1
        assert handler_calls[0].name == "high_cpu"

    def test_alert_manager_check_alert_conditions_error(self, alert_manager):
        """Test alert condition checking with error."""

        def error_condition(metrics):
            raise Exception("Condition error")

        alert_manager.register_alert_rule("error_rule", error_condition)

        # Should not raise exception
        alert_manager.check_alert_conditions({"test": 1})

        assert "error_rule" not in alert_manager._active_alerts

    def test_alert_manager_get_active_alerts(self, alert_manager):
        """Test getting active alerts."""

        def condition(metrics):
            return metrics.get("cpu_usage", 0) > 90

        alert_manager.register_alert_rule("high_cpu", condition, AlertSeverity.WARNING)

        # Trigger alert
        metrics = {"cpu_usage": 95}
        alert_manager.check_alert_conditions(metrics)

        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].name == "high_cpu"

    def test_alert_manager_get_alert_history(self, alert_manager):
        """Test getting alert history."""

        def condition(metrics):
            return metrics.get("cpu_usage", 0) > 90

        alert_manager.register_alert_rule("high_cpu", condition, AlertSeverity.WARNING)

        # Trigger and resolve alert multiple times
        for _ in range(3):
            alert_manager.check_alert_conditions({"cpu_usage": 95})
            alert_manager.check_alert_conditions({"cpu_usage": 50})

        history = alert_manager.get_alert_history(limit=10)
        assert len(history) == 6  # 3 alerts + 3 resolutions

    def test_alert_manager_max_alerts_limit(self, alert_manager):
        """Test max alerts limit."""
        alert_manager.max_alerts = 2

        def condition(metrics):
            return True

        alert_manager.register_alert_rule("test_rule", condition)

        # Trigger alerts beyond limit
        for i in range(5):
            alert_manager.check_alert_conditions({"test": i})

        assert len(alert_manager._alerts) == 2


class TestHealthChecker:
    """Test HealthChecker functionality."""

    @pytest.fixture
    def health_checker(self):
        """Create HealthChecker instance."""
        return HealthChecker(check_interval=1.0)

    def test_health_checker_initialization(self, health_checker):
        """Test HealthChecker initialization."""
        assert health_checker.check_interval == 1.0
        assert len(health_checker._health_checks) == 0
        assert health_checker._health_status.status == "unknown"
        assert health_checker._running is False
        assert health_checker._check_task is None

    def test_health_checker_register_health_check(self, health_checker):
        """Test registering health check."""

        def cpu_check():
            return True

        health_checker.register_health_check("cpu", cpu_check)

        assert "cpu" in health_checker._health_checks
        assert health_checker._health_checks["cpu"] == cpu_check

    @pytest.mark.asyncio
    async def test_health_checker_start_stop(self, health_checker):
        """Test starting and stopping health checker."""
        assert health_checker._running is False

        health_checker.start()
        assert health_checker._running is True

        # Test stop
        with patch.object(health_checker, "_check_task") as mock_task:
            mock_task.cancel = Mock()
            await health_checker.stop()
            assert health_checker._running is False

    @pytest.mark.asyncio
    async def test_health_checker_perform_health_checks_all_healthy(
        self, health_checker
    ):
        """Test performing health checks - all healthy."""

        def cpu_check():
            return True

        def memory_check():
            return True

        health_checker.register_health_check("cpu", cpu_check)
        health_checker.register_health_check("memory", memory_check)

        await health_checker._perform_health_checks()

        status = health_checker.get_health_status()
        assert status.status == "healthy"
        assert status.components["cpu"] is True
        assert status.components["memory"] is True

    @pytest.mark.asyncio
    async def test_health_checker_perform_health_checks_degraded(self, health_checker):
        """Test performing health checks - degraded."""

        def cpu_check():
            return True

        def memory_check():
            return False

        health_checker.register_health_check("cpu", cpu_check)
        health_checker.register_health_check("memory", memory_check)

        await health_checker._perform_health_checks()

        status = health_checker.get_health_status()
        assert status.status == "degraded"
        assert status.components["cpu"] is True
        assert status.components["memory"] is False

    @pytest.mark.asyncio
    async def test_health_checker_perform_health_checks_unhealthy(self, health_checker):
        """Test performing health checks - unhealthy."""

        def cpu_check():
            return False

        def memory_check():
            return False

        health_checker.register_health_check("cpu", cpu_check)
        health_checker.register_health_check("memory", memory_check)

        await health_checker._perform_health_checks()

        status = health_checker.get_health_status()
        assert status.status == "unhealthy"
        assert status.components["cpu"] is False
        assert status.components["memory"] is False

    @pytest.mark.asyncio
    async def test_health_checker_perform_health_checks_error(self, health_checker):
        """Test performing health checks with error."""

        def error_check():
            raise Exception("Health check error")

        health_checker.register_health_check("error_check", error_check)

        await health_checker._perform_health_checks()

        status = health_checker.get_health_status()
        assert status.status == "unhealthy"
        assert status.components["error_check"] is False

    @pytest.mark.asyncio
    async def test_health_checker_health_check_loop(self, health_checker):
        """Test health check loop."""
        health_checker._running = True

        with patch.object(health_checker, "_perform_health_checks") as mock_perform:
            # Start health check loop
            task = asyncio.create_task(health_checker._health_check_loop())

            # Wait a bit for the loop to run
            await asyncio.sleep(0.1)

            # Cancel the task
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should have called perform health checks
            mock_perform.assert_called()


class TestEnterpriseMetrics:
    """Test EnterpriseMetrics coordinator."""

    @pytest.fixture
    def enterprise_metrics(self):
        """Create EnterpriseMetrics instance."""
        config = {
            "metrics_flush_interval": 1.0,
            "max_metric_points": 1000,
            "prometheus_port": 8000,
            "prometheus_host": "0.0.0.0",
            "max_alerts": 100,
            "health_check_interval": 1.0,
        }
        return EnterpriseMetrics(config)

    def test_enterprise_metrics_initialization(self, enterprise_metrics):
        """Test EnterpriseMetrics initialization."""
        assert enterprise_metrics.metrics_collector is not None
        assert enterprise_metrics.prometheus_exporter is not None
        assert enterprise_metrics.alert_manager is not None
        assert enterprise_metrics.health_checker is not None
        assert enterprise_metrics._running is False
        assert enterprise_metrics._monitor_task is None

    def test_enterprise_metrics_setup_default_alerts(self, enterprise_metrics):
        """Test setup of default alert rules."""
        # Should have default alert rules
        assert "high_cpu_usage" in enterprise_metrics.alert_manager._alert_rules
        assert "high_memory_usage" in enterprise_metrics.alert_manager._alert_rules
        assert "high_disk_usage" in enterprise_metrics.alert_manager._alert_rules
        assert "process_errors" in enterprise_metrics.alert_manager._alert_rules
        assert (
            "compliance_assessment_failures"
            in enterprise_metrics.alert_manager._alert_rules
        )

    def test_enterprise_metrics_setup_default_health_checks(self, enterprise_metrics):
        """Test setup of default health checks."""
        # Should have default health checks
        assert "cpu" in enterprise_metrics.health_checker._health_checks
        assert "memory" in enterprise_metrics.health_checker._health_checks
        assert "disk" in enterprise_metrics.health_checker._health_checks

    @pytest.mark.asyncio
    async def test_enterprise_metrics_start_stop(self, enterprise_metrics):
        """Test starting and stopping enterprise metrics."""
        with (
            patch.object(enterprise_metrics.metrics_collector, "start") as mock_start,
            patch.object(
                enterprise_metrics.prometheus_exporter, "start"
            ) as mock_prom_start,
            patch.object(
                enterprise_metrics.health_checker, "start"
            ) as mock_health_start,
            patch("asyncio.create_task") as mock_create_task,
        ):

            await enterprise_metrics.start()

            assert enterprise_metrics._running is True
            mock_start.assert_called_once()
            mock_prom_start.assert_called_once()
            mock_health_start.assert_called_once()
            mock_create_task.assert_called()

            # Test stop
            with (
                patch.object(enterprise_metrics.metrics_collector, "stop") as mock_stop,
                patch.object(
                    enterprise_metrics.prometheus_exporter, "stop"
                ) as mock_prom_stop,
                patch.object(
                    enterprise_metrics.health_checker, "stop"
                ) as mock_health_stop,
            ):

                await enterprise_metrics.stop()

                assert enterprise_metrics._running is False
                mock_stop.assert_called_once()
                mock_prom_stop.assert_called_once()
                mock_health_stop.assert_called_once()

    def test_enterprise_metrics_record_protocol_discovery_metric(
        self, enterprise_metrics
    ):
        """Test recording protocol discovery metric."""
        enterprise_metrics.record_protocol_discovery_metric(
            "test_metric", 123.45, {"service": "test"}
        )

        latest = enterprise_metrics.metrics_collector.get_latest_values()
        assert latest["protocol_discovery_test_metric"] == 123.45

    def test_enterprise_metrics_increment_protocol_discovery_counter(
        self, enterprise_metrics
    ):
        """Test incrementing protocol discovery counter."""
        enterprise_metrics.increment_protocol_discovery_counter(
            "test_counter", 5.0, {"operation": "test"}
        )

        latest = enterprise_metrics.metrics_collector.get_latest_values()
        assert latest["protocol_discovery_test_counter"] == 5.0

    def test_enterprise_metrics_get_dashboard_metrics(self, enterprise_metrics):
        """Test getting dashboard metrics."""
        # Set some test metrics
        enterprise_metrics.metrics_collector.set_gauge("system_cpu_usage_percent", 25.5)
        enterprise_metrics.metrics_collector.set_gauge(
            "system_memory_usage_percent", 60.0
        )
        enterprise_metrics.metrics_collector.set_gauge(
            "system_disk_usage_percent", 45.0
        )
        enterprise_metrics.metrics_collector.set_gauge(
            "protocol_discovery_operations_total", 100
        )

        dashboard_metrics = enterprise_metrics.get_dashboard_metrics()

        assert "timestamp" in dashboard_metrics
        assert "health" in dashboard_metrics
        assert "system" in dashboard_metrics
        assert "protocol_discovery" in dashboard_metrics
        assert "alerts" in dashboard_metrics

        assert dashboard_metrics["system"]["cpu_percent"] == 25.5
        assert dashboard_metrics["system"]["memory_percent"] == 60.0
        assert dashboard_metrics["system"]["disk_percent"] == 45.0
        assert dashboard_metrics["protocol_discovery"]["operations_total"] == 100

    @pytest.mark.asyncio
    async def test_enterprise_metrics_monitoring_loop(self, enterprise_metrics):
        """Test monitoring loop."""
        enterprise_metrics._running = True

        with (
            patch.object(
                enterprise_metrics.metrics_collector, "get_latest_values"
            ) as mock_get_values,
            patch.object(
                enterprise_metrics.alert_manager, "check_alert_conditions"
            ) as mock_check_alerts,
        ):

            mock_get_values.return_value = {"cpu_usage": 95.0}

            # Start monitoring loop
            task = asyncio.create_task(enterprise_metrics._monitoring_loop())

            # Wait a bit for the loop to run
            await asyncio.sleep(0.1)

            # Cancel the task
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should have called alert checking
            mock_get_values.assert_called()
            mock_check_alerts.assert_called()


class TestEnterpriseMetricsIntegration:
    """Integration tests for enterprise metrics components."""

    def test_get_enterprise_metrics_global(self):
        """Test getting global enterprise metrics instance."""
        # Clear global instance
        import ai_engine.monitoring.enterprise_metrics

        original_metrics = ai_engine.monitoring.enterprise_metrics._enterprise_metrics
        ai_engine.monitoring.enterprise_metrics._enterprise_metrics = None

        try:
            # First call should create new instance
            metrics1 = get_enterprise_metrics()
            assert metrics1 is not None

            # Second call should return same instance
            metrics2 = get_enterprise_metrics()
            assert metrics2 is metrics1

            # Test with config
            config = {"metrics_flush_interval": 2.0}
            metrics3 = get_enterprise_metrics(config)
            assert metrics3 is metrics1  # Should return existing instance

        finally:
            # Restore original instance
            ai_engine.monitoring.enterprise_metrics._enterprise_metrics = (
                original_metrics
            )

    @pytest.mark.asyncio
    async def test_shutdown_enterprise_metrics(self):
        """Test shutting down global enterprise metrics."""
        import ai_engine.monitoring.enterprise_metrics

        # Set a mock instance
        mock_metrics = AsyncMock()
        ai_engine.monitoring.enterprise_metrics._enterprise_metrics = mock_metrics

        await shutdown_enterprise_metrics()

        mock_metrics.stop.assert_called_once()
        assert ai_engine.monitoring.enterprise_metrics._enterprise_metrics is None

    def test_record_metric_convenience_function(self):
        """Test record_metric convenience function."""
        with patch(
            "ai_engine.monitoring.enterprise_metrics.get_enterprise_metrics"
        ) as mock_get_metrics:
            mock_metrics = Mock()
            mock_get_metrics.return_value = mock_metrics

            record_metric("test_metric", 123.45, {"service": "test"})

            mock_metrics.record_protocol_discovery_metric.assert_called_once_with(
                "test_metric", 123.45, {"service": "test"}
            )

    def test_increment_counter_convenience_function(self):
        """Test increment_counter convenience function."""
        with patch(
            "ai_engine.monitoring.enterprise_metrics.get_enterprise_metrics"
        ) as mock_get_metrics:
            mock_metrics = Mock()
            mock_get_metrics.return_value = mock_metrics

            increment_counter("test_counter", 5.0, {"operation": "test"})

            mock_metrics.increment_protocol_discovery_counter.assert_called_once_with(
                "test_counter", 5.0, {"operation": "test"}
            )

    @pytest.mark.asyncio
    async def test_metrics_context_success(self):
        """Test metrics_context with successful operation."""
        with patch(
            "ai_engine.monitoring.enterprise_metrics.get_enterprise_metrics"
        ) as mock_get_metrics:
            mock_metrics = Mock()
            mock_get_metrics.return_value = mock_metrics

            async with metrics_context("test_operation", {"service": "test"}):
                pass

            # Should have recorded success metrics
            mock_metrics.increment_protocol_discovery_counter.assert_called_with(
                "operations_total",
                labels={
                    "operation": "test_operation",
                    "service": "test",
                    "status": "success",
                },
            )
            mock_metrics.record_protocol_discovery_metric.assert_called_with(
                "operation_duration_seconds",
                pytest.approx(0.0, abs=0.1),
                {"operation": "test_operation", "service": "test"},
            )

    @pytest.mark.asyncio
    async def test_metrics_context_error(self):
        """Test metrics_context with error."""
        with patch(
            "ai_engine.monitoring.enterprise_metrics.get_enterprise_metrics"
        ) as mock_get_metrics:
            mock_metrics = Mock()
            mock_get_metrics.return_value = mock_metrics

            with pytest.raises(ValueError):
                async with metrics_context("error_operation", {"service": "test"}):
                    raise ValueError("Test error")

            # Should have recorded error metrics
            mock_metrics.increment_protocol_discovery_counter.assert_called_with(
                "operations_total",
                labels={
                    "operation": "error_operation",
                    "service": "test",
                    "status": "error",
                    "error_type": "ValueError",
                },
            )
            mock_metrics.increment_protocol_discovery_counter.assert_called_with(
                "errors_total",
                labels={"operation": "error_operation", "service": "test"},
            )

    @pytest.mark.asyncio
    async def test_enterprise_metrics_complete_workflow(self):
        """Test complete enterprise metrics workflow."""
        config = {
            "metrics_flush_interval": 0.1,
            "max_metric_points": 100,
            "prometheus_port": 8001,
            "prometheus_host": "127.0.0.1",
            "max_alerts": 50,
            "health_check_interval": 0.1,
        }

        metrics = EnterpriseMetrics(config)

        # Record some metrics
        metrics.record_protocol_discovery_metric("response_time", 150.0)
        metrics.increment_protocol_discovery_counter("requests_total", 1.0)

        # Check metrics were recorded
        latest = metrics.metrics_collector.get_latest_values()
        assert latest["protocol_discovery_response_time"] == 150.0
        assert latest["protocol_discovery_requests_total"] == 1.0

        # Test dashboard metrics
        dashboard = metrics.get_dashboard_metrics()
        assert "timestamp" in dashboard
        assert "health" in dashboard
        assert "system" in dashboard
        assert "protocol_discovery" in dashboard
        assert "alerts" in dashboard

        # Test alert conditions
        metrics.metrics_collector.set_gauge("system_cpu_usage_percent", 95.0)
        latest = metrics.metrics_collector.get_latest_values()
        metrics.alert_manager.check_alert_conditions(latest)

        # Should have triggered high CPU alert
        active_alerts = metrics.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].name == "high_cpu_usage"

    def test_enterprise_metrics_thread_safety(self):
        """Test enterprise metrics thread safety."""
        metrics = EnterpriseMetrics()

        def record_metrics():
            for i in range(100):
                metrics.record_protocol_discovery_metric(f"metric_{i}", float(i))
                metrics.increment_protocol_discovery_counter(f"counter_{i}", 1.0)

        # Run concurrent operations
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(record_metrics) for _ in range(5)]
            [future.result() for future in futures]

        # All operations should complete successfully
        latest = metrics.metrics_collector.get_latest_values()
        assert len(latest) > 0
