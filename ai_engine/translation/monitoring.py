"""
QBITEL - Translation Studio Monitoring and Observability
Enterprise-grade monitoring, metrics collection, and health checking for translation operations.
"""

import time
import asyncio
import psutil
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
import threading
from collections import defaultdict, deque
import uuid

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi import HTTPException
from fastapi.responses import Response

from ..core.config import Config
from .exceptions import (
    TranslationStudioException,
    ErrorCategory,
    ErrorSeverity,
    TimeoutException,
    ExternalServiceException,
)
from .logging import get_logger, LogComponent, LogOperation, create_context


class HealthStatus(Enum):
    """Health status levels for services and components."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class HealthCheck:
    """Health check result for a component."""

    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Dict[str, Any] = None
    response_time: float = 0.0

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class MetricSnapshot:
    """Snapshot of a metric at a point in time."""

    name: str
    type: MetricType
    value: Union[float, int]
    labels: Dict[str, str]
    timestamp: datetime
    description: str = ""


@dataclass
class SystemResource:
    """System resource usage information."""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_io_counters: Dict[str, int]
    open_file_descriptors: int
    process_count: int
    timestamp: datetime


class PrometheusMetrics:
    """Centralized Prometheus metrics for Translation Studio."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
        self.logger = get_logger(__name__)

    def _setup_metrics(self):
        """Initialize all Prometheus metrics."""

        # === Request Metrics ===
        self.requests_total = Counter(
            "qbitel_translation_requests_total",
            "Total number of translation requests",
            ["component", "operation", "status"],
            registry=self.registry,
        )

        self.request_duration = Histogram(
            "qbitel_translation_request_duration_seconds",
            "Translation request processing time",
            ["component", "operation"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        # === Protocol Discovery Metrics ===
        self.protocol_discoveries_total = Counter(
            "qbitel_translation_protocol_discoveries_total",
            "Total protocol discoveries performed",
            ["protocol_type", "success"],
            registry=self.registry,
        )

        self.protocol_discovery_confidence = Histogram(
            "qbitel_translation_protocol_discovery_confidence",
            "Protocol discovery confidence scores",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        # === API Generation Metrics ===
        self.api_generations_total = Counter(
            "qbitel_translation_api_generations_total",
            "Total API specifications generated",
            ["api_style", "security_level", "success"],
            registry=self.registry,
        )

        self.api_generation_duration = Histogram(
            "qbitel_translation_api_generation_duration_seconds",
            "API generation processing time",
            ["api_style"],
            registry=self.registry,
        )

        # === Code Generation Metrics ===
        self.sdk_generations_total = Counter(
            "qbitel_translation_sdk_generations_total",
            "Total SDKs generated",
            ["language", "success"],
            registry=self.registry,
        )

        self.code_generation_duration = Histogram(
            "qbitel_translation_code_generation_duration_seconds",
            "Code generation processing time",
            ["language", "template_type"],
            registry=self.registry,
        )

        # === Protocol Translation Metrics ===
        self.protocol_translations_total = Counter(
            "qbitel_translation_protocol_translations_total",
            "Total protocol translations performed",
            ["source_protocol", "target_protocol", "success"],
            registry=self.registry,
        )

        self.translation_confidence = Histogram(
            "qbitel_translation_confidence_score",
            "Protocol translation confidence scores",
            ["source_protocol", "target_protocol"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        # === LLM Metrics ===
        self.llm_requests_total = Counter(
            "qbitel_translation_llm_requests_total",
            "Total LLM requests made",
            ["provider", "model", "operation", "status"],
            registry=self.registry,
        )

        self.llm_response_time = Histogram(
            "qbitel_translation_llm_response_time_seconds",
            "LLM response time",
            ["provider", "model"],
            registry=self.registry,
        )

        # === Error Metrics ===
        self.errors_total = Counter(
            "qbitel_translation_errors_total",
            "Total errors by category and severity",
            ["component", "error_category", "error_severity"],
            registry=self.registry,
        )

        # === Resource Metrics ===
        self.cpu_usage_percent = Gauge(
            "qbitel_translation_cpu_usage_percent",
            "CPU usage percentage",
            registry=self.registry,
        )

        self.memory_usage_bytes = Gauge(
            "qbitel_translation_memory_usage_bytes",
            "Memory usage in bytes",
            ["type"],  # 'used', 'available', 'total'
            registry=self.registry,
        )

        self.active_operations = Gauge(
            "qbitel_translation_active_operations",
            "Number of currently active operations",
            ["component", "operation"],
            registry=self.registry,
        )

        # === Service Health Metrics ===
        self.service_health = Gauge(
            "qbitel_translation_service_health",
            "Service health status (1=healthy, 0.5=degraded, 0=unhealthy)",
            ["service", "component"],
            registry=self.registry,
        )

        # === Build Info ===
        self.build_info = Info(
            "qbitel_translation_build_info",
            "Translation Studio build information",
            registry=self.registry,
        )

        # Set build info
        self.build_info.info(
            {
                "version": "1.0.0",
                "build_date": datetime.now(timezone.utc).isoformat(),
                "git_commit": "development",
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            }
        )


class HealthChecker:
    """Health checking service for Translation Studio components."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("health_checker")
        self.checks: Dict[str, callable] = {}
        self.last_results: Dict[str, HealthCheck] = {}
        self.check_intervals: Dict[str, int] = {}
        self._setup_default_checks()

    def _setup_default_checks(self):
        """Setup default health checks."""
        self.register_check("system_resources", self._check_system_resources, interval=30)

        self.register_check("memory_usage", self._check_memory_usage, interval=30)

        self.register_check("disk_space", self._check_disk_space, interval=60)

    def register_check(self, name: str, check_func: callable, interval: int = 60):
        """Register a health check function."""
        self.checks[name] = check_func
        self.check_intervals[name] = interval
        self.logger.info(f"Registered health check: {name} (interval: {interval}s)")

    async def _check_system_resources(self) -> HealthCheck:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            status = HealthStatus.HEALTHY
            message = "System resources normal"

            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High CPU usage: {cpu_percent}%"
            elif cpu_percent > 75:
                status = HealthStatus.DEGRADED
                message = f"Elevated CPU usage: {cpu_percent}%"

            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                message += f", High memory usage: {memory.percent}%"
            elif memory.percent > 75:
                status = max(status, HealthStatus.DEGRADED)
                message += f", Elevated memory usage: {memory.percent}%"

            return HealthCheck(
                component="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_available_gb": memory.available / (1024**3),
                },
            )

        except Exception as e:
            return HealthCheck(
                component="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check system resources: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                details={"error": str(e)},
            )

    async def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage specific to the current process."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            status = HealthStatus.HEALTHY
            message = "Process memory usage normal"

            if memory_percent > 80:
                status = HealthStatus.UNHEALTHY
                message = f"High process memory usage: {memory_percent:.1f}%"
            elif memory_percent > 60:
                status = HealthStatus.DEGRADED
                message = f"Elevated process memory usage: {memory_percent:.1f}%"

            return HealthCheck(
                component="memory_usage",
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                details={
                    "rss_mb": memory_info.rss / (1024**2),
                    "vms_mb": memory_info.vms / (1024**2),
                    "memory_percent": memory_percent,
                    "open_fds": process.num_fds() if hasattr(process, "num_fds") else 0,
                },
            )

        except Exception as e:
            return HealthCheck(
                component="memory_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check memory usage: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                details={"error": str(e)},
            )

    async def _check_disk_space(self) -> HealthCheck:
        """Check disk space usage."""
        try:
            disk_usage = psutil.disk_usage("/")
            usage_percent = (disk_usage.used / disk_usage.total) * 100

            status = HealthStatus.HEALTHY
            message = "Disk space normal"

            if usage_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk space: {usage_percent:.1f}% used"
            elif usage_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {usage_percent:.1f}% used"

            return HealthCheck(
                component="disk_space",
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                details={
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "usage_percent": usage_percent,
                },
            )

        except Exception as e:
            return HealthCheck(
                component="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check disk space: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                details={"error": str(e)},
            )

    async def run_check(self, check_name: str) -> HealthCheck:
        """Run a specific health check."""
        if check_name not in self.checks:
            raise ValueError(f"Unknown health check: {check_name}")

        start_time = time.time()
        try:
            result = await self.checks[check_name]()
            result.response_time = time.time() - start_time
            self.last_results[check_name] = result

            self.logger.debug(
                f"Health check completed: {check_name}",
                extra={
                    "check_name": check_name,
                    "status": result.status.value,
                    "response_time": result.response_time,
                },
            )

            return result

        except Exception as e:
            error_result = HealthCheck(
                component=check_name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                response_time=time.time() - start_time,
                details={"error": str(e)},
            )
            self.last_results[check_name] = error_result

            self.logger.error(
                f"Health check failed: {check_name}",
                exc_info=True,
                extra={"check_name": check_name},
            )

            return error_result

    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}

        for check_name in self.checks:
            results[check_name] = await self.run_check(check_name)

        return results

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health based on all checks."""
        if not self.last_results:
            return HealthStatus.UNKNOWN

        statuses = [check.status for check in self.last_results.values()]

        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif any(status == HealthStatus.UNKNOWN for status in statuses):
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY


class TranslationStudioMonitor:
    """Main monitoring coordinator for Translation Studio."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("translation_monitor")

        # Initialize components
        self.metrics = PrometheusMetrics()
        self.health_checker = HealthChecker(config)

        # Component monitoring
        self.component_monitors: Dict[str, Any] = {}
        self.is_monitoring = False
        self._monitoring_task = None

    async def start_monitoring(self):
        """Start the monitoring system."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return

        self.is_monitoring = True
        self.logger.info("Starting Translation Studio monitoring system")

        # Start background monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        # Register component-specific health checks
        await self._register_component_checks()

    async def stop_monitoring(self):
        """Stop the monitoring system."""
        self.is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Translation Studio monitoring system stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                # Run health checks
                await self._run_periodic_health_checks()

                # Sleep for monitoring interval
                await asyncio.sleep(30)  # Run every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait longer on error

    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            self.metrics.cpu_usage_percent.set(cpu_percent)
            self.metrics.memory_usage_bytes.labels(type="used").set(memory.used)
            self.metrics.memory_usage_bytes.labels(type="available").set(memory.available)
            self.metrics.memory_usage_bytes.labels(type="total").set(memory.total)

        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")

    async def _run_periodic_health_checks(self):
        """Run periodic health checks."""
        try:
            health_results = await self.health_checker.run_all_checks()

            # Update health metrics
            for check_name, result in health_results.items():
                health_value = {
                    HealthStatus.HEALTHY: 1.0,
                    HealthStatus.DEGRADED: 0.5,
                    HealthStatus.UNHEALTHY: 0.0,
                    HealthStatus.UNKNOWN: -1.0,
                }.get(result.status, -1.0)

                self.metrics.service_health.labels(service="translation_studio", component=check_name).set(health_value)

        except Exception as e:
            self.logger.error(f"Failed to run health checks: {e}")

    async def _register_component_checks(self):
        """Register component-specific health checks."""
        # These would be implemented as the system grows
        pass

    def record_operation(self, component: str, operation: str, duration: float, success: bool, **metadata):
        """Record an operation for monitoring."""
        status = "success" if success else "error"

        self.metrics.requests_total.labels(component=component, operation=operation, status=status).inc()

        self.metrics.request_duration.labels(component=component, operation=operation).observe(duration)

        self.logger.debug(
            f"Recorded operation: {component}.{operation}",
            extra={
                "component": component,
                "operation": operation,
                "duration": duration,
                "success": success,
                "metadata": metadata,
            },
        )

    def record_protocol_discovery(self, protocol_type: str, confidence: float, duration: float, success: bool):
        """Record protocol discovery metrics."""
        self.metrics.protocol_discoveries_total.labels(protocol_type=protocol_type, success=str(success).lower()).inc()

        if success and confidence >= 0:
            self.metrics.protocol_discovery_confidence.observe(confidence)

    def record_api_generation(
        self,
        api_style: str,
        security_level: str,
        duration: float,
        endpoints_count: int,
        success: bool,
    ):
        """Record API generation metrics."""
        self.metrics.api_generations_total.labels(
            api_style=api_style,
            security_level=security_level,
            success=str(success).lower(),
        ).inc()

        self.metrics.api_generation_duration.labels(api_style=api_style).observe(duration)

    def record_code_generation(self, language: str, duration: float, lines_generated: int, success: bool):
        """Record code generation metrics."""
        self.metrics.sdk_generations_total.labels(language=language, success=str(success).lower()).inc()

        self.metrics.code_generation_duration.labels(language=language, template_type="sdk").observe(duration)

    def record_protocol_translation(
        self,
        source_protocol: str,
        target_protocol: str,
        confidence: float,
        success: bool,
    ):
        """Record protocol translation metrics."""
        self.metrics.protocol_translations_total.labels(
            source_protocol=source_protocol,
            target_protocol=target_protocol,
            success=str(success).lower(),
        ).inc()

        if success and confidence >= 0:
            self.metrics.translation_confidence.labels(
                source_protocol=source_protocol, target_protocol=target_protocol
            ).observe(confidence)

    def record_llm_request(self, provider: str, model: str, operation: str, duration: float, success: bool):
        """Record LLM request metrics."""
        self.metrics.llm_requests_total.labels(
            provider=provider,
            model=model,
            operation=operation,
            status="success" if success else "error",
        ).inc()

        self.metrics.llm_response_time.labels(provider=provider, model=model).observe(duration)

    def record_error(self, component: str, error_category: str, error_severity: str):
        """Record error metrics."""
        self.metrics.errors_total.labels(
            component=component,
            error_category=error_category,
            error_severity=error_severity,
        ).inc()

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        overall_health = self.health_checker.get_overall_health()
        health_checks = await self.health_checker.run_all_checks()

        return {
            "overall_status": overall_health.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "response_time": check.response_time,
                    "details": check.details,
                }
                for name, check in health_checks.items()
            },
            "system_info": {
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "platform": os.name,
                "pid": os.getpid(),
            },
        }

    def get_metrics_response(self) -> Response:
        """Get Prometheus metrics response."""
        try:
            metrics_data = generate_latest(self.registry)
            return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
        except Exception as e:
            self.logger.error(f"Failed to generate metrics: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate metrics")


# Global monitor instance
_monitor_instance: Optional[TranslationStudioMonitor] = None
_monitor_lock = threading.RLock()


def get_monitor(config: Optional[Config] = None) -> TranslationStudioMonitor:
    """Get or create the global monitor instance."""
    global _monitor_instance

    with _monitor_lock:
        if _monitor_instance is None:
            if config is None:
                raise ValueError("Config required for first monitor initialization")
            _monitor_instance = TranslationStudioMonitor(config)

        return _monitor_instance


async def initialize_monitoring(config: Config) -> TranslationStudioMonitor:
    """Initialize the monitoring system."""
    monitor = get_monitor(config)
    await monitor.start_monitoring()
    return monitor


async def shutdown_monitoring():
    """Shutdown the monitoring system."""
    global _monitor_instance

    with _monitor_lock:
        if _monitor_instance:
            await _monitor_instance.stop_monitoring()
            _monitor_instance = None


# Decorator for automatic operation monitoring
def monitor_operation(
    component: str,
    operation: str,
    record_success: bool = True,
    record_errors: bool = True,
):
    """Decorator to automatically monitor operation performance."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            monitor = get_monitor()
            start_time = time.time()
            success = False

            try:
                # Execute function
                result = await func(*args, **kwargs)
                success = True
                return result

            except Exception as e:
                if record_errors and isinstance(e, TranslationStudioException):
                    monitor.record_error(
                        component=component,
                        error_category=e.category.value,
                        error_severity=e.severity.value,
                    )
                raise

            finally:
                if record_success or not success:
                    duration = time.time() - start_time
                    monitor.record_operation(
                        component=component,
                        operation=operation,
                        duration=duration,
                        success=success,
                    )

        def sync_wrapper(*args, **kwargs):
            monitor = get_monitor()
            start_time = time.time()
            success = False

            try:
                # Execute function
                result = func(*args, **kwargs)
                success = True
                return result

            except Exception as e:
                if record_errors and isinstance(e, TranslationStudioException):
                    monitor.record_error(
                        component=component,
                        error_category=e.category.value,
                        error_severity=e.severity.value,
                    )
                raise

            finally:
                if record_success or not success:
                    duration = time.time() - start_time
                    monitor.record_operation(
                        component=component,
                        operation=operation,
                        duration=duration,
                        success=success,
                    )

        # Return appropriate wrapper
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


__all__ = [
    "HealthStatus",
    "MetricType",
    "HealthCheck",
    "MetricSnapshot",
    "SystemResource",
    "PrometheusMetrics",
    "HealthChecker",
    "TranslationStudioMonitor",
    "get_monitor",
    "initialize_monitoring",
    "shutdown_monitoring",
    "monitor_operation",
]
