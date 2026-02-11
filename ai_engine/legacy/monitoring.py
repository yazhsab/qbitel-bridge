"""
QBITEL Engine - Legacy System Whisperer Monitoring

Comprehensive monitoring and observability for Legacy System Whisperer feature.
Provides metrics collection, health monitoring, alerting, and dashboard integration.
"""

import time
import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
from contextlib import contextmanager

# Prometheus metrics
try:
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

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ..monitoring.metrics import AIEngineMetrics
from ..core.config import Config
from .exceptions import LegacySystemWhispererException, ErrorSeverity
from .models import LegacySystemContext, SystemMetrics
from .logging import LegacySystemLogger, LogCategory


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricData:
    """Generic metric data structure."""

    name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime
    help_text: str = ""

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus exposition format."""
        labels_str = ",".join([f'{k}="{v}"' for k, v in self.labels.items()])
        timestamp_ms = int(self.timestamp.timestamp() * 1000)
        return f"{self.name}{{{labels_str}}} {self.value} {timestamp_ms}"


@dataclass
class HealthCheckResult:
    """Health check result."""

    component: str
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    response_time_ms: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class Alert:
    """Alert definition."""

    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source_component: str
    system_id: Optional[str] = None
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    timestamp: Optional[datetime] = None
    labels: Optional[Dict[str, str]] = None
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.labels is None:
            self.labels = {}


class LegacySystemMetrics:
    """Custom metrics for Legacy System Whisperer."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize custom metrics."""
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()

    def _setup_metrics(self):
        """Setup Prometheus metrics if available."""
        if not PROMETHEUS_AVAILABLE:
            return

        # System registration metrics
        self.registered_systems = Gauge(
            "legacy_whisperer_registered_systems_total",
            "Total number of registered legacy systems",
            ["system_type", "criticality"],
            registry=self.registry,
        )

        # Anomaly detection metrics
        self.anomalies_detected = Counter(
            "legacy_whisperer_anomalies_detected_total",
            "Total anomalies detected",
            ["system_id", "system_type", "severity"],
            registry=self.registry,
        )

        self.anomaly_detection_duration = Histogram(
            "legacy_whisperer_anomaly_detection_duration_seconds",
            "Time spent on anomaly detection",
            ["system_id", "system_type"],
            registry=self.registry,
        )

        # Failure prediction metrics
        self.failure_predictions = Counter(
            "legacy_whisperer_failure_predictions_total",
            "Total failure predictions generated",
            ["system_id", "prediction_horizon", "confidence_level"],
            registry=self.registry,
        )

        self.prediction_accuracy = Gauge(
            "legacy_whisperer_prediction_accuracy_ratio",
            "Prediction accuracy ratio",
            ["system_id", "prediction_type"],
            registry=self.registry,
        )

        # Knowledge capture metrics
        self.knowledge_sessions = Counter(
            "legacy_whisperer_knowledge_sessions_total",
            "Total knowledge capture sessions",
            ["session_type", "status"],
            registry=self.registry,
        )

        self.knowledge_items = Gauge(
            "legacy_whisperer_knowledge_items_total",
            "Total knowledge items captured",
            ["category", "validation_status"],
            registry=self.registry,
        )

        # Decision support metrics
        self.recommendations_generated = Counter(
            "legacy_whisperer_recommendations_total",
            "Total recommendations generated",
            ["decision_category", "confidence_level"],
            registry=self.registry,
        )

        self.decision_response_time = Histogram(
            "legacy_whisperer_decision_response_time_seconds",
            "Decision support response time",
            ["decision_category"],
            registry=self.registry,
        )

        # Maintenance scheduling metrics
        self.maintenance_scheduled = Counter(
            "legacy_whisperer_maintenance_scheduled_total",
            "Total maintenance activities scheduled",
            ["maintenance_type", "system_type"],
            registry=self.registry,
        )

        self.maintenance_optimization_score = Gauge(
            "legacy_whisperer_maintenance_optimization_score",
            "Maintenance optimization score",
            ["system_id"],
            registry=self.registry,
        )

        # LLM integration metrics
        self.llm_requests = Counter(
            "legacy_whisperer_llm_requests_total",
            "Total LLM requests",
            ["provider", "request_type", "status"],
            registry=self.registry,
        )

        self.llm_response_time = Histogram(
            "legacy_whisperer_llm_response_time_seconds",
            "LLM response time",
            ["provider", "request_type"],
            registry=self.registry,
        )

        self.llm_token_usage = Counter(
            "legacy_whisperer_llm_tokens_used_total",
            "Total LLM tokens used",
            ["provider", "request_type"],
            registry=self.registry,
        )

        # System health metrics
        self.system_health_score = Gauge(
            "legacy_whisperer_system_health_score",
            "Overall system health score",
            ["system_id"],
            registry=self.registry,
        )

        self.component_status = Info(
            "legacy_whisperer_component_status",
            "Component status information",
            registry=self.registry,
        )

        # Performance metrics
        self.memory_usage = Gauge(
            "legacy_whisperer_memory_usage_bytes",
            "Memory usage in bytes",
            ["component"],
            registry=self.registry,
        )

        self.cpu_usage = Gauge(
            "legacy_whisperer_cpu_usage_percent",
            "CPU usage percentage",
            ["component"],
            registry=self.registry,
        )

        self.operation_duration = Histogram(
            "legacy_whisperer_operation_duration_seconds",
            "Operation duration",
            ["operation", "component", "status"],
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            "legacy_whisperer_errors_total",
            "Total errors occurred",
            ["error_category", "severity", "component"],
            registry=self.registry,
        )

    def record_system_registration(self, system_type: str, criticality: str):
        """Record system registration."""
        if PROMETHEUS_AVAILABLE:
            self.registered_systems.labels(
                system_type=system_type, criticality=criticality
            ).inc()

    def record_anomaly_detection(
        self, system_id: str, system_type: str, severity: str, duration_seconds: float
    ):
        """Record anomaly detection metrics."""
        if PROMETHEUS_AVAILABLE:
            self.anomalies_detected.labels(
                system_id=system_id, system_type=system_type, severity=severity
            ).inc()

            self.anomaly_detection_duration.labels(
                system_id=system_id, system_type=system_type
            ).observe(duration_seconds)

    def record_failure_prediction(
        self, system_id: str, prediction_horizon: str, confidence_level: str
    ):
        """Record failure prediction metrics."""
        if PROMETHEUS_AVAILABLE:
            self.failure_predictions.labels(
                system_id=system_id,
                prediction_horizon=prediction_horizon,
                confidence_level=confidence_level,
            ).inc()

    def update_prediction_accuracy(
        self, system_id: str, prediction_type: str, accuracy: float
    ):
        """Update prediction accuracy metrics."""
        if PROMETHEUS_AVAILABLE:
            self.prediction_accuracy.labels(
                system_id=system_id, prediction_type=prediction_type
            ).set(accuracy)

    def record_knowledge_session(self, session_type: str, status: str):
        """Record knowledge capture session."""
        if PROMETHEUS_AVAILABLE:
            self.knowledge_sessions.labels(
                session_type=session_type, status=status
            ).inc()

    def update_knowledge_items(self, category: str, validation_status: str, count: int):
        """Update knowledge items count."""
        if PROMETHEUS_AVAILABLE:
            self.knowledge_items.labels(
                category=category, validation_status=validation_status
            ).set(count)

    def record_recommendation(self, decision_category: str, confidence_level: str):
        """Record recommendation generation."""
        if PROMETHEUS_AVAILABLE:
            self.recommendations_generated.labels(
                decision_category=decision_category, confidence_level=confidence_level
            ).inc()

    def record_decision_response_time(
        self, decision_category: str, duration_seconds: float
    ):
        """Record decision support response time."""
        if PROMETHEUS_AVAILABLE:
            self.decision_response_time.labels(
                decision_category=decision_category
            ).observe(duration_seconds)

    def record_maintenance_scheduled(self, maintenance_type: str, system_type: str):
        """Record scheduled maintenance."""
        if PROMETHEUS_AVAILABLE:
            self.maintenance_scheduled.labels(
                maintenance_type=maintenance_type, system_type=system_type
            ).inc()

    def update_maintenance_optimization(self, system_id: str, score: float):
        """Update maintenance optimization score."""
        if PROMETHEUS_AVAILABLE:
            self.maintenance_optimization_score.labels(system_id=system_id).set(score)

    def record_llm_request(
        self,
        provider: str,
        request_type: str,
        status: str,
        duration_seconds: float,
        tokens_used: int = 0,
    ):
        """Record LLM request metrics."""
        if PROMETHEUS_AVAILABLE:
            self.llm_requests.labels(
                provider=provider, request_type=request_type, status=status
            ).inc()

            self.llm_response_time.labels(
                provider=provider, request_type=request_type
            ).observe(duration_seconds)

            if tokens_used > 0:
                self.llm_token_usage.labels(
                    provider=provider, request_type=request_type
                ).inc(tokens_used)

    def update_system_health(self, system_id: str, health_score: float):
        """Update system health score."""
        if PROMETHEUS_AVAILABLE:
            self.system_health_score.labels(system_id=system_id).set(health_score)

    def update_component_status(self, component_info: Dict[str, str]):
        """Update component status information."""
        if PROMETHEUS_AVAILABLE:
            self.component_status.info(component_info)

    def record_resource_usage(
        self, component: str, memory_bytes: float, cpu_percent: float
    ):
        """Record resource usage."""
        if PROMETHEUS_AVAILABLE:
            self.memory_usage.labels(component=component).set(memory_bytes)
            self.cpu_usage.labels(component=component).set(cpu_percent)

    def record_operation(
        self, operation: str, component: str, status: str, duration_seconds: float
    ):
        """Record operation metrics."""
        if PROMETHEUS_AVAILABLE:
            self.operation_duration.labels(
                operation=operation, component=component, status=status
            ).observe(duration_seconds)

    def record_error(self, error_category: str, severity: str, component: str):
        """Record error occurrence."""
        if PROMETHEUS_AVAILABLE:
            self.errors_total.labels(
                error_category=error_category, severity=severity, component=component
            ).inc()


class HealthMonitor:
    """Health monitoring for Legacy System Whisperer components."""

    def __init__(
        self,
        config: Config,
        logger: LegacySystemLogger,
        metrics: Optional[LegacySystemMetrics] = None,
    ):
        """Initialize health monitor."""
        self.config = config
        self.logger = logger
        self.metrics = metrics

        # Health check functions registry
        self.health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}

        # Health history for trend analysis
        self.health_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)  # Keep last 100 health checks per component
        )

        # Component dependencies
        self.component_dependencies: Dict[str, List[str]] = {}

        # Health check intervals and scheduling
        self.check_intervals: Dict[str, int] = {}  # component -> interval in seconds
        self.last_check_times: Dict[str, datetime] = {}

        # Overall health status
        self.overall_status = HealthStatus.UNKNOWN
        self.status_timestamp = datetime.now(timezone.utc)

        # Background health monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None

    def register_health_check(
        self,
        component: str,
        check_function: Callable[[], HealthCheckResult],
        interval_seconds: int = 60,
        dependencies: Optional[List[str]] = None,
    ):
        """Register a health check for a component."""
        self.health_checks[component] = check_function
        self.check_intervals[component] = interval_seconds

        if dependencies:
            self.component_dependencies[component] = dependencies

        self.logger.debug(
            f"Registered health check for component: {component}",
            extra_data={"interval": interval_seconds, "dependencies": dependencies},
        )

    def remove_health_check(self, component: str):
        """Remove health check for a component."""
        self.health_checks.pop(component, None)
        self.check_intervals.pop(component, None)
        self.last_check_times.pop(component, None)
        self.component_dependencies.pop(component, None)
        self.health_history.pop(component, None)

        self.logger.debug(f"Removed health check for component: {component}")

    async def check_component_health(self, component: str) -> HealthCheckResult:
        """Check health of a specific component."""

        if component not in self.health_checks:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNKNOWN,
                message=f"No health check registered for component: {component}",
            )

        start_time = time.time()

        try:
            # Execute health check
            if asyncio.iscoroutinefunction(self.health_checks[component]):
                result = await self.health_checks[component]()
            else:
                result = self.health_checks[component]()

            # Record response time
            result.response_time_ms = (time.time() - start_time) * 1000

            # Update health history
            self.health_history[component].append(result)
            self.last_check_times[component] = datetime.now(timezone.utc)

            # Log health check result
            self.logger.debug(
                f"Health check completed for {component}: {result.status.value}",
                extra_data={
                    "component": component,
                    "status": result.status.value,
                    "response_time_ms": result.response_time_ms,
                    "message": result.message,
                },
            )

            return result

        except Exception as e:
            # Health check failed
            duration_ms = (time.time() - start_time) * 1000

            error_result = HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=duration_ms,
                details={"error": str(e), "error_type": type(e).__name__},
            )

            self.health_history[component].append(error_result)
            self.last_check_times[component] = datetime.now(timezone.utc)

            self.logger.error(
                f"Health check failed for {component}: {e}",
                extra_data={"component": component, "duration_ms": duration_ms},
                exception=e,
            )

            return error_result

    async def check_all_components(self) -> Dict[str, HealthCheckResult]:
        """Check health of all registered components."""

        results = {}

        # Check components in dependency order
        checked_components = set()

        async def check_with_dependencies(component: str):
            if component in checked_components:
                return

            # Check dependencies first
            if component in self.component_dependencies:
                for dependency in self.component_dependencies[component]:
                    if dependency in self.health_checks:
                        await check_with_dependencies(dependency)

            # Check the component
            results[component] = await self.check_component_health(component)
            checked_components.add(component)

        # Check all components
        for component in self.health_checks:
            await check_with_dependencies(component)

        # Update overall status
        self._update_overall_status(results)

        return results

    def _update_overall_status(self, component_results: Dict[str, HealthCheckResult]):
        """Update overall health status based on component results."""

        if not component_results:
            self.overall_status = HealthStatus.UNKNOWN
            return

        # Count status types
        status_counts = defaultdict(int)
        for result in component_results.values():
            status_counts[result.status] += 1

        total_components = len(component_results)

        # Determine overall status
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            if status_counts[HealthStatus.UNHEALTHY] >= total_components * 0.5:
                self.overall_status = HealthStatus.UNHEALTHY
            else:
                self.overall_status = HealthStatus.DEGRADED
        elif status_counts[HealthStatus.DEGRADED] > 0:
            self.overall_status = HealthStatus.DEGRADED
        elif status_counts[HealthStatus.HEALTHY] == total_components:
            self.overall_status = HealthStatus.HEALTHY
        else:
            self.overall_status = HealthStatus.UNKNOWN

        self.status_timestamp = datetime.now(timezone.utc)

        # Record metrics if available
        if self.metrics:
            component_info = {
                "overall_status": self.overall_status.value,
                "healthy_components": str(status_counts[HealthStatus.HEALTHY]),
                "degraded_components": str(status_counts[HealthStatus.DEGRADED]),
                "unhealthy_components": str(status_counts[HealthStatus.UNHEALTHY]),
                "last_check": self.status_timestamp.isoformat(),
            }
            self.metrics.update_component_status(component_info)

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all components."""

        summary = {
            "overall_status": self.overall_status.value,
            "last_updated": self.status_timestamp.isoformat(),
            "components": {},
            "trends": {},
        }

        # Component status summary
        for component in self.health_checks:
            if component in self.health_history and self.health_history[component]:
                latest = self.health_history[component][-1]
                summary["components"][component] = {
                    "status": latest.status.value,
                    "message": latest.message,
                    "last_check": latest.timestamp.isoformat(),
                    "response_time_ms": latest.response_time_ms,
                }
            else:
                summary["components"][component] = {
                    "status": "unknown",
                    "message": "No health check data available",
                }

        # Health trends
        for component, history in self.health_history.items():
            if len(history) >= 2:
                recent_checks = list(history)[-10:]  # Last 10 checks
                healthy_ratio = sum(
                    1 for check in recent_checks if check.status == HealthStatus.HEALTHY
                ) / len(recent_checks)

                if healthy_ratio >= 0.9:
                    trend = "stable"
                elif healthy_ratio >= 0.7:
                    trend = "degrading"
                else:
                    trend = "critical"

                summary["trends"][component] = {
                    "trend": trend,
                    "healthy_ratio": healthy_ratio,
                    "recent_checks": len(recent_checks),
                }

        return summary

    async def start_monitoring(self):
        """Start background health monitoring."""

        if self.monitoring_active:
            self.logger.warning("Health monitoring is already active")
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("Started background health monitoring")

    async def stop_monitoring(self):
        """Stop background health monitoring."""

        self.monitoring_active = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None

        self.logger.info("Stopped background health monitoring")

    async def _monitoring_loop(self):
        """Background monitoring loop."""

        try:
            while self.monitoring_active:
                # Check which components need health checks
                current_time = datetime.now(timezone.utc)
                components_to_check = []

                for component, interval in self.check_intervals.items():
                    last_check = self.last_check_times.get(component)

                    if (
                        last_check is None
                        or (current_time - last_check).total_seconds() >= interval
                    ):
                        components_to_check.append(component)

                # Perform health checks
                if components_to_check:
                    self.logger.debug(
                        f"Performing scheduled health checks for: {components_to_check}"
                    )

                    for component in components_to_check:
                        try:
                            await self.check_component_health(component)
                        except Exception as e:
                            self.logger.error(
                                f"Error during scheduled health check for {component}: {e}",
                                exception=e,
                            )

                # Sleep for a short interval before next iteration
                await asyncio.sleep(10)  # Check every 10 seconds

        except asyncio.CancelledError:
            self.logger.info("Health monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Health monitoring loop error: {e}", exception=e)


class AlertManager:
    """Alert management for Legacy System Whisperer."""

    def __init__(
        self,
        config: Config,
        logger: LegacySystemLogger,
        metrics: Optional[LegacySystemMetrics] = None,
    ):
        """Initialize alert manager."""
        self.config = config
        self.logger = logger
        self.metrics = metrics

        # Active alerts
        self.active_alerts: Dict[str, Alert] = {}

        # Alert rules and thresholds
        self.alert_rules: Dict[str, Dict[str, Any]] = {}

        # Alert handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []

        # Alert history
        self.alert_history: deque = deque(maxlen=1000)

        # Alert suppression
        self.suppression_rules: Dict[str, Dict[str, Any]] = {}

        # Setup default alert rules
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self):
        """Setup default alert rules."""

        # High anomaly detection rate
        self.alert_rules["high_anomaly_rate"] = {
            "metric": "anomalies_per_hour",
            "threshold": 10,
            "comparison": "greater_than",
            "severity": AlertSeverity.WARNING,
            "message": "High anomaly detection rate detected",
        }

        # Low system health score
        self.alert_rules["low_health_score"] = {
            "metric": "system_health_score",
            "threshold": 0.5,
            "comparison": "less_than",
            "severity": AlertSeverity.CRITICAL,
            "message": "System health score is critically low",
        }

        # LLM service failure rate
        self.alert_rules["high_llm_failure_rate"] = {
            "metric": "llm_failure_rate",
            "threshold": 0.1,  # 10%
            "comparison": "greater_than",
            "severity": AlertSeverity.WARNING,
            "message": "High LLM service failure rate",
        }

        # Component unhealthy
        self.alert_rules["component_unhealthy"] = {
            "metric": "component_health_status",
            "threshold": "unhealthy",
            "comparison": "equals",
            "severity": AlertSeverity.CRITICAL,
            "message": "Component is in unhealthy state",
        }

        # Memory usage high
        self.alert_rules["high_memory_usage"] = {
            "metric": "memory_usage_percent",
            "threshold": 0.9,  # 90%
            "comparison": "greater_than",
            "severity": AlertSeverity.WARNING,
            "message": "High memory usage detected",
        }

    def add_alert_rule(
        self,
        rule_name: str,
        metric: str,
        threshold: Union[float, str],
        comparison: str,
        severity: AlertSeverity,
        message: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Add custom alert rule."""

        self.alert_rules[rule_name] = {
            "metric": metric,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity,
            "message": message,
            "labels": labels or {},
        }

        self.logger.debug(f"Added alert rule: {rule_name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule."""
        self.alert_rules.pop(rule_name, None)
        self.logger.debug(f"Removed alert rule: {rule_name}")

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
        self.logger.debug("Added alert handler")

    def evaluate_metric(
        self,
        metric_name: str,
        metric_value: Union[float, str],
        labels: Optional[Dict[str, str]] = None,
    ):
        """Evaluate metric against alert rules."""

        labels = labels or {}

        for rule_name, rule in self.alert_rules.items():
            if rule["metric"] != metric_name:
                continue

            # Check if alert should be triggered
            should_alert = False
            comparison = rule["comparison"]
            threshold = rule["threshold"]

            try:
                if comparison == "greater_than":
                    should_alert = float(metric_value) > float(threshold)
                elif comparison == "less_than":
                    should_alert = float(metric_value) < float(threshold)
                elif comparison == "equals":
                    should_alert = str(metric_value) == str(threshold)
                elif comparison == "not_equals":
                    should_alert = str(metric_value) != str(threshold)
                elif comparison == "greater_equal":
                    should_alert = float(metric_value) >= float(threshold)
                elif comparison == "less_equal":
                    should_alert = float(metric_value) <= float(threshold)

            except (ValueError, TypeError):
                self.logger.warning(
                    f"Unable to evaluate alert rule {rule_name}: invalid metric value or threshold"
                )
                continue

            # Generate alert ID based on rule and labels
            alert_id = self._generate_alert_id(rule_name, labels)

            if should_alert:
                # Check if alert is already active
                if alert_id not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        alert_id=alert_id,
                        title=rule_name.replace("_", " ").title(),
                        description=rule["message"],
                        severity=rule["severity"],
                        source_component="legacy_system_whisperer",
                        metric_name=metric_name,
                        threshold_value=(
                            float(threshold)
                            if isinstance(threshold, (int, float))
                            else None
                        ),
                        current_value=(
                            float(metric_value)
                            if isinstance(metric_value, (int, float))
                            else None
                        ),
                        labels={**labels, **rule.get("labels", {})},
                    )

                    self._trigger_alert(alert)

            else:
                # Check if alert should be resolved
                if alert_id in self.active_alerts:
                    self._resolve_alert(alert_id)

    def _generate_alert_id(self, rule_name: str, labels: Dict[str, str]) -> str:
        """Generate unique alert ID."""
        import hashlib

        # Create deterministic ID based on rule and labels
        label_str = ",".join(sorted(f"{k}={v}" for k, v in labels.items()))
        id_string = f"{rule_name}:{label_str}"

        return hashlib.md5(id_string.encode()).hexdigest()[:12]

    def _trigger_alert(self, alert: Alert):
        """Trigger a new alert."""

        # Check suppression rules
        if self._is_suppressed(alert):
            self.logger.debug(f"Alert suppressed: {alert.alert_id}")
            return

        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert

        # Add to history
        self.alert_history.append(alert)

        # Log alert
        self.logger.warning(
            f"ALERT TRIGGERED: {alert.title}",
            extra_data={
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "description": alert.description,
                "labels": alert.labels,
            },
        )

        # Execute alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler error: {e}", exception=e)

        # Record metrics
        if self.metrics:
            self.metrics.record_error(
                error_category="alert",
                severity=alert.severity.value,
                component="alert_manager",
            )

    def _resolve_alert(self, alert_id: str):
        """Resolve an active alert."""

        if alert_id not in self.active_alerts:
            return

        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolution_timestamp = datetime.now(timezone.utc)

        # Remove from active alerts
        del self.active_alerts[alert_id]

        # Log resolution
        self.logger.info(
            f"ALERT RESOLVED: {alert.title}",
            extra_data={
                "alert_id": alert_id,
                "duration_seconds": (
                    alert.resolution_timestamp - alert.timestamp
                ).total_seconds(),
            },
        )

    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed."""

        # Check for suppression rules
        for rule_pattern, suppression in self.suppression_rules.items():
            if rule_pattern in alert.title or rule_pattern in alert.description:
                # Check time-based suppression
                if "suppress_until" in suppression:
                    if datetime.now(timezone.utc) < suppression["suppress_until"]:
                        return True

                # Check count-based suppression
                if "max_alerts_per_hour" in suppression:
                    recent_alerts = [
                        a
                        for a in self.alert_history
                        if (
                            a.title == alert.title
                            and datetime.now(timezone.utc) - a.timestamp
                            < timedelta(hours=1)
                        )
                    ]

                    if len(recent_alerts) >= suppression["max_alerts_per_hour"]:
                        return True

        return False

    def suppress_alert(
        self,
        pattern: str,
        suppress_until: Optional[datetime] = None,
        max_alerts_per_hour: Optional[int] = None,
    ):
        """Add alert suppression rule."""

        suppression = {}

        if suppress_until:
            suppression["suppress_until"] = suppress_until

        if max_alerts_per_hour:
            suppression["max_alerts_per_hour"] = max_alerts_per_hour

        self.suppression_rules[pattern] = suppression

        self.logger.info(f"Added alert suppression for pattern: {pattern}")

    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""

        now = datetime.now(timezone.utc)
        last_hour_alerts = [
            alert
            for alert in self.alert_history
            if now - alert.timestamp < timedelta(hours=1)
        ]

        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1

        return {
            "active_alerts": len(self.active_alerts),
            "alerts_last_hour": len(last_hour_alerts),
            "total_alerts_today": len(
                [
                    alert
                    for alert in self.alert_history
                    if now - alert.timestamp < timedelta(days=1)
                ]
            ),
            "severity_breakdown": dict(severity_counts),
            "oldest_active_alert": min(
                (alert.timestamp for alert in self.active_alerts.values()), default=None
            ),
        }


class LegacySystemMonitor:
    """Main monitoring coordinator for Legacy System Whisperer."""

    def __init__(
        self,
        config: Config,
        logger: LegacySystemLogger,
        base_metrics: Optional[AIEngineMetrics] = None,
    ):
        """Initialize Legacy System Monitor."""
        self.config = config
        self.logger = logger
        self.base_metrics = base_metrics

        # Initialize custom metrics
        self.metrics = LegacySystemMetrics()

        # Initialize sub-systems
        self.health_monitor = HealthMonitor(config, logger, self.metrics)
        self.alert_manager = AlertManager(config, logger, self.metrics)

        # Monitoring state
        self.monitoring_started = False

        # Resource monitoring
        self.resource_monitor_task: Optional[asyncio.Task] = None

        # Register default health checks
        self._register_default_health_checks()

        # Setup default alert handlers
        self._setup_default_alert_handlers()

    def _register_default_health_checks(self):
        """Register default health checks."""

        def service_health_check() -> HealthCheckResult:
            """Basic service health check."""
            return HealthCheckResult(
                component="legacy_whisperer_service",
                status=HealthStatus.HEALTHY,
                message="Service is running normally",
            )

        def memory_health_check() -> HealthCheckResult:
            """Memory usage health check."""
            try:
                import psutil

                memory = psutil.virtual_memory()

                if memory.percent > 90:
                    status = HealthStatus.UNHEALTHY
                    message = f"Memory usage critical: {memory.percent}%"
                elif memory.percent > 80:
                    status = HealthStatus.DEGRADED
                    message = f"Memory usage high: {memory.percent}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Memory usage normal: {memory.percent}%"

                return HealthCheckResult(
                    component="memory",
                    status=status,
                    message=message,
                    details={"usage_percent": memory.percent},
                )

            except ImportError:
                return HealthCheckResult(
                    component="memory",
                    status=HealthStatus.UNKNOWN,
                    message="psutil not available for memory monitoring",
                )

        # Register health checks
        self.health_monitor.register_health_check(
            "legacy_whisperer_service", service_health_check, interval_seconds=30
        )

        self.health_monitor.register_health_check(
            "memory", memory_health_check, interval_seconds=60
        )

    def _setup_default_alert_handlers(self):
        """Setup default alert handlers."""

        def log_alert_handler(alert: Alert):
            """Log alert to system logs."""
            log_level = (
                "critical" if alert.severity == AlertSeverity.EMERGENCY else "error"
            )

            self.logger.log_structured(
                getattr(self.logger, log_level.upper()),
                f"Alert: {alert.title} - {alert.description}",
                extra_data={
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "system_id": alert.system_id,
                    "metric": alert.metric_name,
                    "threshold": alert.threshold_value,
                    "current_value": alert.current_value,
                },
            )

        self.alert_manager.add_alert_handler(log_alert_handler)

    async def start_monitoring(self):
        """Start all monitoring components."""

        if self.monitoring_started:
            self.logger.warning("Monitoring already started")
            return

        try:
            # Start health monitoring
            await self.health_monitor.start_monitoring()

            # Start resource monitoring
            self.resource_monitor_task = asyncio.create_task(
                self._resource_monitoring_loop()
            )

            self.monitoring_started = True
            self.logger.info("Legacy System Whisperer monitoring started")

        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}", exception=e)
            raise

    async def stop_monitoring(self):
        """Stop all monitoring components."""

        try:
            # Stop health monitoring
            await self.health_monitor.stop_monitoring()

            # Stop resource monitoring
            if self.resource_monitor_task:
                self.resource_monitor_task.cancel()
                try:
                    await self.resource_monitor_task
                except asyncio.CancelledError:
                    pass
                self.resource_monitor_task = None

            self.monitoring_started = False
            self.logger.info("Legacy System Whisperer monitoring stopped")

        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}", exception=e)

    async def _resource_monitoring_loop(self):
        """Background resource monitoring loop."""

        try:
            while self.monitoring_started:
                try:
                    # Monitor system resources
                    await self._collect_resource_metrics()

                    # Sleep for monitoring interval
                    await asyncio.sleep(60)  # Monitor every minute

                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}", exception=e)
                    await asyncio.sleep(60)

        except asyncio.CancelledError:
            self.logger.info("Resource monitoring loop cancelled")

    async def _collect_resource_metrics(self):
        """Collect and record resource metrics."""

        try:
            import psutil

            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.record_resource_usage(
                component="legacy_whisperer",
                memory_bytes=memory.used,
                cpu_percent=psutil.cpu_percent(interval=1),
            )

            # Evaluate against alert rules
            self.alert_manager.evaluate_metric(
                "memory_usage_percent",
                memory.percent / 100.0,
                {"component": "legacy_whisperer"},
            )

        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            self.logger.error(f"Resource metrics collection error: {e}", exception=e)

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""

        return {
            "monitoring_active": self.monitoring_started,
            "health_summary": self.health_monitor.get_health_summary(),
            "alert_summary": self.alert_manager.get_alert_summary(),
            "metrics_available": PROMETHEUS_AVAILABLE,
            "registered_health_checks": len(self.health_monitor.health_checks),
            "active_alert_rules": len(self.alert_manager.alert_rules),
        }

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in exposition format."""

        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available\n"

        return generate_latest(self.metrics.registry)

    @contextmanager
    def monitor_operation(
        self, operation: str, component: str, system_id: Optional[str] = None
    ):
        """Context manager for monitoring operations."""

        start_time = time.time()

        try:
            yield

            # Operation successful
            duration = time.time() - start_time
            self.metrics.record_operation(operation, component, "success", duration)

        except Exception as e:
            # Operation failed
            duration = time.time() - start_time
            self.metrics.record_operation(operation, component, "failure", duration)

            # Record error
            error_category = getattr(e, "category", "unknown")
            error_severity = getattr(e, "severity", ErrorSeverity.MEDIUM)

            if hasattr(error_category, "value"):
                error_category = error_category.value
            if hasattr(error_severity, "value"):
                error_severity = error_severity.value

            self.metrics.record_error(error_category, error_severity, component)

            raise

    # Convenience methods for recording specific metrics
    def record_system_registration(self, system_context: LegacySystemContext):
        """Record system registration."""
        self.metrics.record_system_registration(
            system_context.system_type.value, system_context.criticality.value
        )

    def record_anomaly_detection(
        self, system_id: str, system_type: str, severity: str, duration_seconds: float
    ):
        """Record anomaly detection."""
        self.metrics.record_anomaly_detection(
            system_id, system_type, severity, duration_seconds
        )

    def record_knowledge_session(self, session_type: str, status: str):
        """Record knowledge capture session."""
        self.metrics.record_knowledge_session(session_type, status)

    def record_llm_request(
        self,
        provider: str,
        request_type: str,
        status: str,
        duration_seconds: float,
        tokens_used: int = 0,
    ):
        """Record LLM request."""
        self.metrics.record_llm_request(
            provider, request_type, status, duration_seconds, tokens_used
        )


def create_legacy_monitor(
    config: Config,
    logger: LegacySystemLogger,
    base_metrics: Optional[AIEngineMetrics] = None,
) -> LegacySystemMonitor:
    """Create Legacy System Whisperer monitor instance."""

    return LegacySystemMonitor(config, logger, base_metrics)
