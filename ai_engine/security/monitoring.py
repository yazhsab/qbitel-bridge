"""
CRONOS AI Engine - Security Orchestrator Monitoring

Enterprise-grade monitoring with Prometheus metrics, health checks,
and performance tracking for the Zero-Touch Security Orchestrator.
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

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

from .config import get_security_config
from .logging import get_security_logger, SecurityLogType, LogLevel


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types of monitored components."""

    DECISION_ENGINE = "decision_engine"
    RESPONSE_MANAGER = "response_manager"
    THREAT_ANALYZER = "threat_analyzer"
    SECURITY_SERVICE = "security_service"
    LLM_SERVICE = "llm_service"
    DATABASE = "database"
    EXTERNAL_INTEGRATION = "external_integration"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    component: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    active_threads: int
    active_incidents: int
    active_quarantines: int
    events_per_second: float
    decisions_per_minute: float
    average_response_time_ms: float
    error_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class PrometheusMetrics:
    """Prometheus metrics for security orchestrator."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # Service metrics
        self.service_info = Info(
            "cronos_security_service_info",
            "Security orchestrator service information",
            registry=self.registry,
        )

        # Health metrics
        self.component_health = Gauge(
            "cronos_security_component_health",
            "Component health status (1=healthy, 0=unhealthy)",
            ["component"],
            registry=self.registry,
        )

        self.health_check_duration = Histogram(
            "cronos_security_health_check_duration_seconds",
            "Health check duration",
            ["component"],
            registry=self.registry,
        )

        # Performance metrics
        self.cpu_usage = Gauge(
            "cronos_security_cpu_usage_percent",
            "CPU usage percentage",
            registry=self.registry,
        )

        self.memory_usage = Gauge(
            "cronos_security_memory_usage_percent",
            "Memory usage percentage",
            registry=self.registry,
        )

        self.memory_used = Gauge(
            "cronos_security_memory_used_bytes",
            "Memory used in bytes",
            registry=self.registry,
        )

        self.active_threads = Gauge(
            "cronos_security_active_threads",
            "Number of active threads",
            registry=self.registry,
        )

        # Business metrics
        self.security_events_rate = Gauge(
            "cronos_security_events_per_second",
            "Security events processed per second",
            registry=self.registry,
        )

        self.decision_rate = Gauge(
            "cronos_security_decisions_per_minute",
            "Security decisions made per minute",
            registry=self.registry,
        )

        self.response_time = Summary(
            "cronos_security_response_time_seconds",
            "Security response time",
            registry=self.registry,
        )

        self.error_rate = Gauge(
            "cronos_security_error_rate",
            "Error rate (errors per total operations)",
            registry=self.registry,
        )

        # Component-specific metrics
        self.llm_requests_total = Counter(
            "cronos_security_llm_requests_total",
            "Total LLM requests made",
            ["provider", "status"],
            registry=self.registry,
        )

        self.quarantine_operations = Counter(
            "cronos_security_quarantine_operations_total",
            "Total quarantine operations",
            ["operation", "status"],
            registry=self.registry,
        )

        self.threat_analysis_duration = Histogram(
            "cronos_security_threat_analysis_duration_seconds",
            "Threat analysis duration",
            ["analysis_type"],
            registry=self.registry,
        )

        # Alerting metrics
        self.critical_alerts = Counter(
            "cronos_security_critical_alerts_total",
            "Total critical security alerts",
            ["alert_type"],
            registry=self.registry,
        )

        self.sla_violations = Counter(
            "cronos_security_sla_violations_total",
            "Total SLA violations",
            ["sla_type"],
            registry=self.registry,
        )


class HealthCheckManager:
    """Manages health checks for security orchestrator components."""

    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_results: Dict[str, HealthCheckResult] = {}
        self.config = get_security_config()
        self.logger = get_security_logger("cronos.security.monitoring")

        # Health check configuration
        self.check_interval = self.config.monitoring.health_checks.get(
            "interval_seconds", 30
        )
        self.check_timeout = self.config.monitoring.health_checks.get(
            "timeout_seconds", 10
        )

        # Background task management
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None

    def register_health_check(
        self, component: str, check_func: Callable[[], Dict[str, Any]]
    ):
        """Register a health check function for a component."""
        self.health_checks[component] = check_func
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Health check registered for component: {component}",
            level=LogLevel.INFO,
            component=component,
        )

    async def run_health_check(self, component: str) -> HealthCheckResult:
        """Run health check for a specific component."""
        start_time = time.time()

        try:
            if component not in self.health_checks:
                return HealthCheckResult(
                    component=component,
                    status=HealthStatus.UNKNOWN,
                    message="Health check not registered",
                    duration_ms=0,
                    timestamp=datetime.now(),
                )

            check_func = self.health_checks[component]

            # Run health check with timeout
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(check_func), timeout=self.check_timeout
                )
            except asyncio.TimeoutError:
                return HealthCheckResult(
                    component=component,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check timed out after {self.check_timeout}s",
                    duration_ms=(time.time() - start_time) * 1000,
                    timestamp=datetime.now(),
                )

            duration_ms = (time.time() - start_time) * 1000

            # Interpret result
            if isinstance(result, dict):
                status = HealthStatus(result.get("status", HealthStatus.HEALTHY))
                message = result.get("message", "Health check passed")
                details = result.get("details", {})
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "Health check passed" if result else "Health check failed"
                details = {}

            health_result = HealthCheckResult(
                component=component,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                details=details,
            )

            # Store result
            self.health_results[component] = health_result

            # Log health check result
            log_level = (
                LogLevel.ERROR if status == HealthStatus.UNHEALTHY else LogLevel.DEBUG
            )
            self.logger.log_security_event(
                SecurityLogType.PERFORMANCE_METRIC,
                f"Health check for {component}: {status.value} - {message}",
                level=log_level,
                component=component,
                execution_time_ms=duration_ms,
                metadata={"health_status": status.value, "details": details},
            )

            return health_result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_result = HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                details={"error": str(e)},
            )

            self.health_results[component] = error_result

            self.logger.log_security_event(
                SecurityLogType.PERFORMANCE_METRIC,
                f"Health check error for {component}: {str(e)}",
                level=LogLevel.ERROR,
                component=component,
                execution_time_ms=duration_ms,
                error_code="HEALTH_CHECK_ERROR",
            )

            return error_result

    async def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        tasks = []

        for component in self.health_checks:
            task = asyncio.create_task(self.run_health_check(component))
            tasks.append((component, task))

        results = {}
        for component, task in tasks:
            try:
                result = await task
                results[component] = result
            except Exception as e:
                results[component] = HealthCheckResult(
                    component=component,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check exception: {str(e)}",
                    duration_ms=0,
                    timestamp=datetime.now(),
                )

        return results

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.health_results:
            return HealthStatus.UNKNOWN

        statuses = [result.status for result in self.health_results.values()]

        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    async def start_background_checks(self):
        """Start background health checks."""
        if self._running:
            return

        self._running = True
        self._health_check_task = asyncio.create_task(self._background_health_checks())

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Background health checks started",
            level=LogLevel.INFO,
        )

    async def stop_background_checks(self):
        """Stop background health checks."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Background health checks stopped",
            level=LogLevel.INFO,
        )

    async def _background_health_checks(self):
        """Background task for running periodic health checks."""
        while self._running:
            try:
                await self.run_all_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Background health check error: {str(e)}",
                    level=LogLevel.ERROR,
                    error_code="BACKGROUND_HEALTH_CHECK_ERROR",
                )
                await asyncio.sleep(30)  # Wait before retrying


class PerformanceMonitor:
    """Monitors system performance and resource usage."""

    def __init__(self, prometheus_metrics: PrometheusMetrics):
        self.metrics = prometheus_metrics
        self.config = get_security_config()
        self.logger = get_security_logger("cronos.security.monitoring")

        # Performance tracking
        self.start_time = time.time()
        self.event_counts = {"total": 0, "errors": 0}
        self.decision_counts = {"total": 0, "auto_executed": 0}
        self.response_times: List[float] = []

        # Resource monitoring
        self.process = psutil.Process()

        # Background monitoring
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    def record_security_event(self, processing_time_ms: float, success: bool = True):
        """Record a security event for performance tracking."""
        self.event_counts["total"] += 1
        if not success:
            self.event_counts["errors"] += 1

        self.response_times.append(processing_time_ms / 1000.0)  # Convert to seconds

        # Keep only recent response times (last 1000)
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]

        # Update Prometheus metrics
        self.metrics.response_time.observe(processing_time_ms / 1000.0)

    def record_decision(self, auto_executed: bool = False):
        """Record a security decision for tracking."""
        self.decision_counts["total"] += 1
        if auto_executed:
            self.decision_counts["auto_executed"] += 1

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics snapshot."""

        # System resource metrics
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        memory_used_mb = memory_info.rss / 1024 / 1024

        # Thread count
        active_threads = threading.active_count()

        # Calculate rates
        uptime_seconds = time.time() - self.start_time
        events_per_second = self.event_counts["total"] / max(uptime_seconds, 1)
        decisions_per_minute = (
            self.decision_counts["total"] / max(uptime_seconds, 1)
        ) * 60

        # Response time statistics
        avg_response_time = (
            sum(self.response_times) / max(len(self.response_times), 1) * 1000
        )  # ms

        # Error rate
        error_rate = self.event_counts["errors"] / max(self.event_counts["total"], 1)

        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            active_threads=active_threads,
            active_incidents=0,  # To be set by caller
            active_quarantines=0,  # To be set by caller
            events_per_second=events_per_second,
            decisions_per_minute=decisions_per_minute,
            average_response_time_ms=avg_response_time,
            error_rate=error_rate,
        )

    def update_prometheus_metrics(self, metrics: PerformanceMetrics):
        """Update Prometheus metrics with current values."""

        # System metrics
        self.metrics.cpu_usage.set(metrics.cpu_percent)
        self.metrics.memory_usage.set(metrics.memory_percent)
        self.metrics.memory_used.set(
            metrics.memory_used_mb * 1024 * 1024
        )  # Convert to bytes
        self.metrics.active_threads.set(metrics.active_threads)

        # Business metrics
        self.metrics.security_events_rate.set(metrics.events_per_second)
        self.metrics.decision_rate.set(metrics.decisions_per_minute)
        self.metrics.error_rate.set(metrics.error_rate)

    async def start_monitoring(self):
        """Start background performance monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._background_monitoring())

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Performance monitoring started",
            level=LogLevel.INFO,
        )

    async def stop_monitoring(self):
        """Stop background performance monitoring."""
        self._monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Performance monitoring stopped",
            level=LogLevel.INFO,
        )

    async def _background_monitoring(self):
        """Background task for performance monitoring."""
        while self._monitoring:
            try:
                metrics = self.get_current_metrics()
                self.update_prometheus_metrics(metrics)

                # Log performance metrics periodically
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Performance snapshot: CPU {metrics.cpu_percent:.1f}%, "
                    f"Memory {metrics.memory_used_mb:.0f}MB, "
                    f"Events/s {metrics.events_per_second:.2f}",
                    level=LogLevel.DEBUG,
                    execution_time_ms=metrics.average_response_time_ms,
                    metadata=metrics.to_dict(),
                )

                await asyncio.sleep(60)  # Update every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Performance monitoring error: {str(e)}",
                    level=LogLevel.ERROR,
                    error_code="PERFORMANCE_MONITOR_ERROR",
                )
                await asyncio.sleep(30)


class SecurityOrchestrationMonitor:
    """
    Main monitoring service for the Zero-Touch Security Orchestrator.

    Provides comprehensive monitoring, health checks, and performance tracking
    with Prometheus metrics integration.
    """

    def __init__(self):
        self.config = get_security_config()
        self.logger = get_security_logger("cronos.security.monitoring")

        # Core monitoring components
        self.prometheus_registry = CollectorRegistry()
        self.prometheus_metrics = PrometheusMetrics(self.prometheus_registry)
        self.health_check_manager = HealthCheckManager()
        self.performance_monitor = PerformanceMonitor(self.prometheus_metrics)

        # State management
        self._initialized = False
        self._running = False

        # External component references (to be set by security service)
        self.security_service = None
        self.decision_engine = None
        self.response_manager = None
        self.threat_analyzer = None

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Security orchestration monitor initialized",
            level=LogLevel.INFO,
        )

    async def initialize(self):
        """Initialize the monitoring system."""
        if self._initialized:
            return

        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Initializing security orchestration monitor",
                level=LogLevel.INFO,
            )

            # Register default health checks
            await self._register_default_health_checks()

            # Set service information
            self.prometheus_metrics.service_info.info(
                {
                    "version": "1.0.0",
                    "environment": getattr(self.config, "environment", "development"),
                    "start_time": datetime.now().isoformat(),
                }
            )

            self._initialized = True

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Security orchestration monitor initialized successfully",
                level=LogLevel.INFO,
            )

        except Exception as e:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Monitor initialization failed: {str(e)}",
                level=LogLevel.ERROR,
                error_code="MONITOR_INIT_ERROR",
            )
            raise

    async def start(self):
        """Start monitoring services."""
        if not self._initialized:
            await self.initialize()

        if self._running:
            return

        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Starting security orchestration monitor",
                level=LogLevel.INFO,
            )

            # Start background monitoring
            await self.health_check_manager.start_background_checks()
            await self.performance_monitor.start_monitoring()

            self._running = True

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Security orchestration monitor started successfully",
                level=LogLevel.INFO,
            )

        except Exception as e:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Monitor start failed: {str(e)}",
                level=LogLevel.ERROR,
                error_code="MONITOR_START_ERROR",
            )
            raise

    async def stop(self):
        """Stop monitoring services."""
        if not self._running:
            return

        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Stopping security orchestration monitor",
                level=LogLevel.INFO,
            )

            # Stop background monitoring
            await self.health_check_manager.stop_background_checks()
            await self.performance_monitor.stop_monitoring()

            self._running = False

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Security orchestration monitor stopped successfully",
                level=LogLevel.INFO,
            )

        except Exception as e:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Monitor stop failed: {str(e)}",
                level=LogLevel.ERROR,
                error_code="MONITOR_STOP_ERROR",
            )
            raise

    def set_component_references(
        self,
        security_service=None,
        decision_engine=None,
        response_manager=None,
        threat_analyzer=None,
    ):
        """Set references to monitored components."""
        self.security_service = security_service
        self.decision_engine = decision_engine
        self.response_manager = response_manager
        self.threat_analyzer = threat_analyzer

    def record_security_event_processing(
        self, processing_time_ms: float, success: bool = True
    ):
        """Record security event processing metrics."""
        self.performance_monitor.record_security_event(processing_time_ms, success)

    def record_decision_made(self, auto_executed: bool = False):
        """Record security decision metrics."""
        self.performance_monitor.record_decision(auto_executed)

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all components."""
        overall_status = self.health_check_manager.get_overall_health()

        return {
            "overall_status": overall_status.value,
            "components": {
                component: result.to_dict()
                for component, result in self.health_check_manager.health_results.items()
            },
            "last_updated": datetime.now().isoformat(),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.performance_monitor.get_current_metrics()

        # Add component-specific metrics if available
        if self.security_service:
            try:
                service_metrics = asyncio.run(
                    self.security_service.get_service_metrics()
                )
                metrics.active_incidents = service_metrics.get("incidents", {}).get(
                    "active_count", 0
                )
                metrics.active_quarantines = service_metrics.get("systems", {}).get(
                    "active_quarantines", 0
                )
            except Exception as e:
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Failed to get service metrics: {str(e)}",
                    level=LogLevel.WARNING,
                )

        return metrics.to_dict()

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.prometheus_registry).decode("utf-8")

    def get_prometheus_content_type(self) -> str:
        """Get Prometheus metrics content type."""
        return CONTENT_TYPE_LATEST

    async def _register_default_health_checks(self):
        """Register default health checks for core components."""

        # System health check
        def system_health_check() -> Dict[str, Any]:
            try:
                # Check system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")

                # Determine status based on resource usage
                if cpu_percent > 90 or memory.percent > 90 or disk.percent > 95:
                    status = HealthStatus.DEGRADED
                    message = "High resource usage detected"
                elif cpu_percent > 95 or memory.percent > 95 or disk.percent > 98:
                    status = HealthStatus.UNHEALTHY
                    message = "Critical resource usage"
                else:
                    status = HealthStatus.HEALTHY
                    message = "System resources normal"

                return {
                    "status": status,
                    "message": message,
                    "details": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "disk_percent": disk.percent,
                    },
                }

            except Exception as e:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": f"System health check failed: {str(e)}",
                    "details": {"error": str(e)},
                }

        self.health_check_manager.register_health_check("system", system_health_check)

        # Configuration health check
        def config_health_check() -> Dict[str, Any]:
            try:
                # Verify configuration is loaded and valid
                config = get_security_config()
                config.validate()

                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "Configuration valid",
                    "details": {
                        "enabled": config.enabled,
                        "max_incidents": config.max_concurrent_incidents,
                    },
                }

            except Exception as e:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": f"Configuration error: {str(e)}",
                    "details": {"error": str(e)},
                }

        self.health_check_manager.register_health_check(
            "configuration", config_health_check
        )

    async def shutdown(self):
        """Shutdown the monitoring system."""
        await self.stop()
        self._initialized = False


# Global monitor instance
_security_monitor: Optional[SecurityOrchestrationMonitor] = None


def get_security_monitor() -> SecurityOrchestrationMonitor:
    """Get global security monitor instance."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityOrchestrationMonitor()
    return _security_monitor
