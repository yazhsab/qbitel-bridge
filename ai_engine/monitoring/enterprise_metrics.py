"""
Enterprise-grade monitoring and metrics for QBITEL Bridge.
Provides comprehensive metrics collection, Prometheus integration, and alerting.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import logging
import psutil
import socket
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import weakref

try:  # pragma: no cover - optional dependency
    from aiohttp import web  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback stub

    class _AiohttpMissing:
        def __getattr__(self, item):
            raise ModuleNotFoundError("aiohttp is required for the metrics web server functionality")

    web = _AiohttpMissing()  # type: ignore

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics supported."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class MetricPoint:
    """Individual metric data point."""

    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """System alert definition."""

    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_timestamp: Optional[float] = None


@dataclass
class HealthStatus:
    """System health status."""

    status: str  # healthy, degraded, unhealthy
    timestamp: float
    components: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)


class MetricsCollector:
    """High-performance metrics collection system."""

    def __init__(self, flush_interval: float = 30.0, max_points: int = 10000):
        self.flush_interval = flush_interval
        self.max_points = max_points
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self._labels: Dict[str, Dict[str, str]] = {}
        self._lock = threading.RLock()
        self._last_flush = time.time()
        self._running = False
        self._flush_task = None

        # Built-in system metrics
        self._system_metrics_enabled = True
        self._custom_metrics: Dict[str, Callable] = {}

    def start(self):
        """Start metrics collection."""
        if not self._running:
            self._running = True
            self._flush_task = asyncio.create_task(self._flush_loop())
            logger.info("Metrics collector started")

    async def stop(self):
        """Stop metrics collection."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collector stopped")

    def register_custom_metric(self, name: str, collector_func: Callable[[], float]):
        """Register a custom metric collector function."""
        self._custom_metrics[name] = collector_func

    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self._add_metric(name, value, MetricType.COUNTER, labels)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        self._add_metric(name, value, MetricType.GAUGE, labels)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        self._add_metric(name, value, MetricType.HISTOGRAM, labels)

    def _add_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Add a metric point."""
        with self._lock:
            point = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=metric_type,
            )
            self._metrics[name].append(point)
            if labels:
                self._labels[name] = labels

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        if not self._system_metrics_enabled:
            return

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.set_gauge("system_cpu_usage_percent", cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_usage_bytes", memory.used)
            self.set_gauge("system_memory_usage_percent", memory.percent)
            self.set_gauge("system_memory_available_bytes", memory.available)

            # Disk metrics
            disk_usage = psutil.disk_usage("/")
            self.set_gauge("system_disk_usage_bytes", disk_usage.used)
            self.set_gauge("system_disk_usage_percent", (disk_usage.used / disk_usage.total) * 100)

            # Network metrics
            network_io = psutil.net_io_counters()
            if network_io:
                self.increment_counter("system_network_bytes_sent", network_io.bytes_sent)
                self.increment_counter("system_network_bytes_recv", network_io.bytes_recv)
                self.increment_counter("system_network_packets_sent", network_io.packets_sent)
                self.increment_counter("system_network_packets_recv", network_io.packets_recv)

            # Process metrics
            process = psutil.Process()
            self.set_gauge("process_cpu_percent", process.cpu_percent())
            self.set_gauge("process_memory_rss_bytes", process.memory_info().rss)
            self.set_gauge("process_memory_vms_bytes", process.memory_info().vms)
            self.set_gauge("process_num_threads", process.num_threads())
            self.set_gauge("process_num_fds", process.num_fds())

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _collect_custom_metrics(self):
        """Collect custom metrics."""
        for name, collector_func in self._custom_metrics.items():
            try:
                value = collector_func()
                self.set_gauge(f"custom_{name}", value)
            except Exception as e:
                logger.error(f"Error collecting custom metric {name}: {e}")

    async def _flush_loop(self):
        """Main flush loop for metrics collection."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                self._collect_system_metrics()
                self._collect_custom_metrics()
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics flush loop: {e}")

    async def _flush_metrics(self):
        """Flush metrics to storage/monitoring systems."""
        current_time = time.time()
        with self._lock:
            metrics_count = sum(len(points) for points in self._metrics.values())
            if metrics_count > 0:
                logger.debug(f"Flushed {metrics_count} metrics points")
            self._last_flush = current_time

    def get_metrics_snapshot(self) -> Dict[str, List[MetricPoint]]:
        """Get current metrics snapshot."""
        with self._lock:
            return {name: list(points) for name, points in self._metrics.items()}

    def get_latest_values(self) -> Dict[str, float]:
        """Get latest value for each metric."""
        with self._lock:
            latest = {}
            for name, points in self._metrics.items():
                if points:
                    latest[name] = points[-1].value
            return latest


class PrometheusExporter:
    """Prometheus metrics exporter."""

    def __init__(self, port: int = 8000, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.metrics_collector: Optional[MetricsCollector] = None
        self._server = None
        self._app = None

    def set_metrics_collector(self, collector: MetricsCollector):
        """Set the metrics collector to export."""
        self.metrics_collector = collector

    def _format_prometheus_metrics(self) -> str:
        """Format metrics in Prometheus format."""
        if not self.metrics_collector:
            return ""

        lines = []
        metrics_snapshot = self.metrics_collector.get_metrics_snapshot()

        for metric_name, points in metrics_snapshot.items():
            if not points:
                continue

            # Get latest point for each unique label combination
            latest_points = {}
            for point in points:
                labels_key = json.dumps(point.labels, sort_keys=True)
                if labels_key not in latest_points or point.timestamp > latest_points[labels_key].timestamp:
                    latest_points[labels_key] = point

            # Format each unique metric
            for point in latest_points.values():
                labels_str = ""
                if point.labels:
                    labels_pairs = [f'{key}="{value}"' for key, value in point.labels.items()]
                    labels_str = "{" + ",".join(labels_pairs) + "}"

                # Add metric type comment
                if metric_name not in [line.split()[2] for line in lines if line.startswith("# TYPE")]:
                    prometheus_type = self._get_prometheus_type(point.metric_type)
                    lines.append(f"# TYPE {metric_name} {prometheus_type}")

                lines.append(f"{metric_name}{labels_str} {point.value}")

        return "\n".join(lines) + "\n"

    def _get_prometheus_type(self, metric_type: MetricType) -> str:
        """Convert internal metric type to Prometheus type."""
        mapping = {
            MetricType.COUNTER: "counter",
            MetricType.GAUGE: "gauge",
            MetricType.HISTOGRAM: "histogram",
            MetricType.SUMMARY: "summary",
        }
        return mapping.get(metric_type, "gauge")

    async def metrics_handler(self, request):
        """HTTP handler for /metrics endpoint."""
        try:
            metrics_text = self._format_prometheus_metrics()
            return web.Response(text=metrics_text, content_type="text/plain")
        except Exception as e:
            logger.error(f"Error generating Prometheus metrics: {e}")
            return web.Response(status=500, text="Internal server error")

    async def start(self):
        """Start Prometheus exporter server."""
        try:
            from aiohttp import web

            self._app = web.Application()
            self._app.router.add_get("/metrics", self.metrics_handler)

            runner = web.AppRunner(self._app)
            await runner.setup()

            site = web.TCPSite(runner, self.host, self.port)
            await site.start()

            logger.info(f"Prometheus exporter started on {self.host}:{self.port}")

        except ImportError:
            logger.warning("aiohttp not available, Prometheus exporter disabled")
        except Exception as e:
            logger.error(f"Failed to start Prometheus exporter: {e}")

    async def stop(self):
        """Stop Prometheus exporter server."""
        if self._server:
            await self._server.close()
            logger.info("Prometheus exporter stopped")


class AlertManager:
    """Alert management system."""

    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self._alerts: deque = deque(maxlen=max_alerts)
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_rules: Dict[str, Callable[[Dict[str, float]], bool]] = {}
        self._alert_handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()

    def register_alert_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, float]], bool],
        severity: AlertSeverity = AlertSeverity.WARNING,
        message: str = "",
    ):
        """Register an alert rule."""
        self._alert_rules[name] = {
            "condition": condition,
            "severity": severity,
            "message": message or f"Alert condition triggered: {name}",
        }

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function."""
        self._alert_handlers.append(handler)

    def check_alert_conditions(self, metrics: Dict[str, float]):
        """Check all alert conditions against current metrics."""
        with self._lock:
            for rule_name, rule in self._alert_rules.items():
                try:
                    condition_met = rule["condition"](metrics)

                    if condition_met and rule_name not in self._active_alerts:
                        # New alert
                        alert = Alert(
                            name=rule_name,
                            severity=rule["severity"],
                            message=rule["message"],
                            timestamp=time.time(),
                        )
                        self._active_alerts[rule_name] = alert
                        self._alerts.append(alert)
                        self._trigger_alert(alert)

                    elif not condition_met and rule_name in self._active_alerts:
                        # Resolve alert
                        alert = self._active_alerts[rule_name]
                        alert.resolved = True
                        alert.resolved_timestamp = time.time()
                        del self._active_alerts[rule_name]
                        self._trigger_alert(alert)

                except Exception as e:
                    logger.error(f"Error checking alert rule {rule_name}: {e}")

    def _trigger_alert(self, alert: Alert):
        """Trigger alert handlers."""
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self._active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            return list(self._alerts)[-limit:]


class HealthChecker:
    """System health monitoring."""

    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self._health_checks: Dict[str, Callable[[], bool]] = {}
        self._health_status = HealthStatus(status="unknown", timestamp=time.time())
        self._running = False
        self._check_task = None

    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self._health_checks[name] = check_func

    async def start(self):
        """Start health monitoring."""
        if not self._running:
            self._running = True
            self._check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Health checker started")

    async def stop(self):
        """Stop health monitoring."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")

    async def _health_check_loop(self):
        """Main health check loop."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def _perform_health_checks(self):
        """Perform all registered health checks."""
        components = {}
        all_healthy = True

        for name, check_func in self._health_checks.items():
            try:
                is_healthy = check_func()
                components[name] = is_healthy
                if not is_healthy:
                    all_healthy = False
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                components[name] = False
                all_healthy = False

        # Determine overall status
        if all_healthy:
            status = "healthy"
        elif any(components.values()):
            status = "degraded"
        else:
            status = "unhealthy"

        self._health_status = HealthStatus(status=status, timestamp=time.time(), components=components)

    def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        return self._health_status


class EnterpriseMetrics:
    """Main enterprise metrics coordinator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Initialize components
        self.metrics_collector = MetricsCollector(
            flush_interval=config.get("metrics_flush_interval", 30.0),
            max_points=config.get("max_metric_points", 10000),
        )

        self.prometheus_exporter = PrometheusExporter(
            port=config.get("prometheus_port", 8000),
            host=config.get("prometheus_host", "0.0.0.0"),
        )
        self.prometheus_exporter.set_metrics_collector(self.metrics_collector)

        self.alert_manager = AlertManager(max_alerts=config.get("max_alerts", 1000))

        self.health_checker = HealthChecker(check_interval=config.get("health_check_interval", 60.0))

        self._running = False
        self._monitor_task = None

        # Setup default alert rules
        self._setup_default_alerts()

        # Setup default health checks
        self._setup_default_health_checks()

    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # High CPU usage
        self.alert_manager.register_alert_rule(
            "high_cpu_usage",
            lambda m: m.get("system_cpu_usage_percent", 0) > 90,
            AlertSeverity.WARNING,
            "CPU usage is above 90%",
        )

        # High memory usage
        self.alert_manager.register_alert_rule(
            "high_memory_usage",
            lambda m: m.get("system_memory_usage_percent", 0) > 85,
            AlertSeverity.WARNING,
            "Memory usage is above 85%",
        )

        # High disk usage
        self.alert_manager.register_alert_rule(
            "high_disk_usage",
            lambda m: m.get("system_disk_usage_percent", 0) > 90,
            AlertSeverity.CRITICAL,
            "Disk usage is above 90%",
        )

        # Process errors
        self.alert_manager.register_alert_rule(
            "process_errors",
            lambda m: m.get("protocol_discovery_errors_total", 0) > 10,
            AlertSeverity.CRITICAL,
            "High number of protocol discovery errors",
        )

        # Compliance-specific alerts
        self.alert_manager.register_alert_rule(
            "compliance_assessment_failures",
            lambda m: m.get("compliance_assessment_errors_total", 0) > 5,
            AlertSeverity.CRITICAL,
            "High number of compliance assessment failures",
        )

        self.alert_manager.register_alert_rule(
            "low_compliance_score",
            lambda m: m.get("compliance_average_score", 100) < 70,
            AlertSeverity.WARNING,
            "Average compliance score below 70%",
        )

        self.alert_manager.register_alert_rule(
            "compliance_report_failures",
            lambda m: m.get("compliance_report_errors_total", 0) > 3,
            AlertSeverity.WARNING,
            "Multiple compliance report generation failures",
        )

        self.alert_manager.register_alert_rule(
            "audit_trail_integrity",
            lambda m: not m.get("audit_blockchain_integrity", True),
            AlertSeverity.CRITICAL,
            "Blockchain audit trail integrity compromised",
        )

    def _setup_default_health_checks(self):
        """Setup default health checks."""

        def cpu_health():
            return psutil.cpu_percent() < 95

        def memory_health():
            return psutil.virtual_memory().percent < 90

        def disk_health():
            return psutil.disk_usage("/").percent < 95

        self.health_checker.register_health_check("cpu", cpu_health)
        self.health_checker.register_health_check("memory", memory_health)
        self.health_checker.register_health_check("disk", disk_health)

    async def start(self):
        """Start enterprise metrics system."""
        if not self._running:
            self._running = True

            # Start components
            self.metrics_collector.start()
            await self.prometheus_exporter.start()
            await self.health_checker.start()

            # Start monitoring task
            self._monitor_task = asyncio.create_task(self._monitoring_loop())

            logger.info("Enterprise metrics system started")

    async def stop(self):
        """Stop enterprise metrics system."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Stop components
        await self.metrics_collector.stop()
        await self.prometheus_exporter.stop()
        await self.health_checker.stop()

        logger.info("Enterprise metrics system stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Check alert conditions
                metrics = self.metrics_collector.get_latest_values()
                self.alert_manager.check_alert_conditions(metrics)

                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def record_protocol_discovery_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record protocol discovery specific metrics."""
        full_name = f"protocol_discovery_{metric_name}"
        self.metrics_collector.set_gauge(full_name, value, labels)

    def increment_protocol_discovery_counter(
        self,
        metric_name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Increment protocol discovery counters."""
        full_name = f"protocol_discovery_{metric_name}"
        self.metrics_collector.increment_counter(full_name, value, labels)

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display."""
        metrics = self.metrics_collector.get_latest_values()
        health_status = self.health_checker.get_health_status()
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            "timestamp": time.time(),
            "health": {
                "status": health_status.status,
                "components": health_status.components,
            },
            "system": {
                "cpu_percent": metrics.get("system_cpu_usage_percent", 0),
                "memory_percent": metrics.get("system_memory_usage_percent", 0),
                "disk_percent": metrics.get("system_disk_usage_percent", 0),
            },
            "protocol_discovery": {
                key.replace("protocol_discovery_", ""): value
                for key, value in metrics.items()
                if key.startswith("protocol_discovery_")
            },
            "alerts": {
                "active_count": len(active_alerts),
                "critical_count": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "warning_count": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
            },
        }


# Global metrics instance
_enterprise_metrics = None


def get_enterprise_metrics(
    config: Optional[Dict[str, Any]] = None,
) -> EnterpriseMetrics:
    """Get global enterprise metrics instance."""
    global _enterprise_metrics
    if _enterprise_metrics is None:
        _enterprise_metrics = EnterpriseMetrics(config)
    return _enterprise_metrics


async def shutdown_enterprise_metrics():
    """Shutdown global enterprise metrics."""
    global _enterprise_metrics
    if _enterprise_metrics:
        await _enterprise_metrics.stop()
        _enterprise_metrics = None


# Convenience functions
def record_metric(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Record a protocol discovery metric."""
    metrics = get_enterprise_metrics()
    metrics.record_protocol_discovery_metric(name, value, labels)


def increment_counter(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
    """Increment a protocol discovery counter."""
    metrics = get_enterprise_metrics()
    metrics.increment_protocol_discovery_counter(name, value, labels)


@asynccontextmanager
async def metrics_context(operation: str, labels: Optional[Dict[str, str]] = None):
    """Context manager for automatic metrics collection."""
    start_time = time.time()
    labels = labels or {}
    labels["operation"] = operation

    try:
        yield
        # Success metrics
        increment_counter("operations_total", labels={**labels, "status": "success"})
    except Exception as e:
        # Error metrics
        increment_counter(
            "operations_total",
            labels={**labels, "status": "error", "error_type": type(e).__name__},
        )
        increment_counter("errors_total", labels=labels)
        raise
    finally:
        # Duration metrics
        duration = time.time() - start_time
        record_metric("operation_duration_seconds", duration, labels)
