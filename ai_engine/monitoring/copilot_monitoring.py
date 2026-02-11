"""
QBITEL Engine - Protocol Intelligence Copilot Monitoring
Advanced monitoring and observability for LLM-enhanced protocol analysis.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import psutil
import threading

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import structlog

from ..core.config import Config
from ..copilot.protocol_copilot import ProtocolIntelligenceCopilot
from ..llm.unified_llm_service import UnifiedLLMService

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types of metrics to collect."""

    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Definition of a monitoring metric."""

    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    quantiles: Optional[List[float]] = None


@dataclass
class Alert:
    """Monitoring alert."""

    level: AlertLevel
    title: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""

    timestamp: datetime
    copilot_queries_per_second: float
    average_response_time: float
    llm_provider_health: Dict[str, str]
    memory_usage_mb: float
    cpu_usage_percent: float
    active_sessions: int
    error_rate_percent: float
    cache_hit_rate_percent: float


class CopilotMonitoringService:
    """
    Comprehensive monitoring service for Protocol Intelligence Copilot.

    Provides metrics collection, alerting, performance monitoring,
    and health checking for all copilot components.
    """

    def __init__(
        self, config: Config, copilot: Optional[ProtocolIntelligenceCopilot] = None
    ):
        """Initialize monitoring service."""
        self.config = config
        self.copilot = copilot
        self.logger = structlog.get_logger(__name__)

        # Monitoring configuration
        self.metrics_enabled = True
        self.alerting_enabled = True
        self.performance_tracking_enabled = True
        self.health_check_interval = 30  # seconds
        self.metrics_collection_interval = 10  # seconds
        self.alert_cooldown_period = 300  # 5 minutes

        # Prometheus registry
        self.registry = CollectorRegistry()

        # Initialize metrics
        self._initialize_metrics()

        # State tracking
        self.is_initialized = False
        self.monitoring_tasks: List[asyncio.Task] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(
            maxlen=288
        )  # 24 hours at 5-min intervals
        self.active_alerts: Dict[str, Alert] = {}

        # Performance tracking
        self.query_times: deque = deque(maxlen=1000)
        self.error_counts: defaultdict = defaultdict(int)
        self.session_tracker: Dict[str, datetime] = {}

        # Thread-safe locks
        self.metrics_lock = threading.Lock()
        self.alerts_lock = threading.Lock()

        self.logger.info("Copilot monitoring service initialized")

    def _initialize_metrics(self):
        """Initialize Prometheus metrics."""

        # Core copilot metrics
        self.copilot_queries_total = Counter(
            "qbitel_copilot_queries_total",
            "Total copilot queries processed",
            ["query_type", "user_id", "status"],
            registry=self.registry,
        )

        self.copilot_query_duration = Histogram(
            "qbitel_copilot_query_duration_seconds",
            "Copilot query processing time",
            ["query_type", "llm_provider"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry,
        )

        self.copilot_confidence_score = Histogram(
            "qbitel_copilot_confidence_score",
            "Confidence scores of copilot responses",
            ["query_type"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        # LLM provider metrics
        self.llm_requests_total = Counter(
            "qbitel_llm_requests_total",
            "Total LLM provider requests",
            ["provider", "model", "status"],
            registry=self.registry,
        )

        self.llm_request_duration = Histogram(
            "qbitel_llm_request_duration_seconds",
            "LLM provider request duration",
            ["provider", "model"],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        self.llm_tokens_used = Counter(
            "qbitel_llm_tokens_used_total",
            "Total tokens used by LLM providers",
            ["provider", "model", "token_type"],
            registry=self.registry,
        )

        # RAG engine metrics
        self.rag_searches_total = Counter(
            "qbitel_rag_searches_total",
            "Total RAG searches performed",
            ["collection", "status"],
            registry=self.registry,
        )

        self.rag_search_duration = Histogram(
            "qbitel_rag_search_duration_seconds",
            "RAG search processing time",
            ["collection"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            registry=self.registry,
        )

        # System metrics
        self.active_sessions = Gauge(
            "qbitel_copilot_active_sessions",
            "Number of active copilot sessions",
            registry=self.registry,
        )

        self.memory_usage_bytes = Gauge(
            "qbitel_copilot_memory_usage_bytes",
            "Memory usage of copilot service",
            registry=self.registry,
        )

        self.cpu_usage_percent = Gauge(
            "qbitel_copilot_cpu_usage_percent",
            "CPU usage percentage of copilot service",
            registry=self.registry,
        )

        # Enhanced protocol discovery metrics
        self.enhanced_discoveries_total = Counter(
            "qbitel_enhanced_discoveries_total",
            "Total enhanced protocol discoveries",
            ["protocol_type", "llm_enhanced", "status"],
            registry=self.registry,
        )

        self.llm_analysis_duration = Histogram(
            "qbitel_llm_analysis_duration_seconds",
            "Duration of LLM analysis phases",
            ["analysis_type", "protocol_type"],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry,
        )

        # Cache metrics
        self.cache_operations_total = Counter(
            "qbitel_copilot_cache_operations_total",
            "Total cache operations",
            ["operation", "cache_type", "status"],
            registry=self.registry,
        )

        self.cache_hit_rate = Gauge(
            "qbitel_copilot_cache_hit_rate",
            "Cache hit rate percentage",
            ["cache_type"],
            registry=self.registry,
        )

        self.logger.info("Prometheus metrics initialized")

    async def initialize(self):
        """Initialize monitoring service."""
        if self.is_initialized:
            return

        try:
            # Start monitoring tasks
            await self._start_monitoring_tasks()

            # Initial system metrics collection
            await self._collect_system_metrics()

            self.is_initialized = True
            self.logger.info("Copilot monitoring service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring service: {e}")
            raise

    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        # Health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self.monitoring_tasks.append(health_task)

        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.monitoring_tasks.append(metrics_task)

        # Performance tracking task
        performance_task = asyncio.create_task(self._performance_tracking_loop())
        self.monitoring_tasks.append(performance_task)

        # Alert processing task
        alert_task = asyncio.create_task(self._alert_processing_loop())
        self.monitoring_tasks.append(alert_task)

        self.logger.info("Monitoring background tasks started")

    async def _health_check_loop(self):
        """Periodic health check loop."""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _metrics_collection_loop(self):
        """Periodic metrics collection loop."""
        while True:
            try:
                await self._collect_system_metrics()
                await self._collect_session_metrics()
                await asyncio.sleep(self.metrics_collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.metrics_collection_interval)

    async def _performance_tracking_loop(self):
        """Periodic performance tracking loop."""
        while True:
            try:
                await self._collect_performance_snapshot()
                await asyncio.sleep(300)  # Every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(300)

    async def _alert_processing_loop(self):
        """Process and manage alerts."""
        while True:
            try:
                await self._process_alerts()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(30)

    async def _perform_health_checks(self):
        """Perform comprehensive health checks."""
        health_status = {
            "copilot_service": "unknown",
            "llm_providers": {},
            "rag_engine": "unknown",
            "system_resources": "unknown",
        }

        try:
            # Check copilot service
            if self.copilot:
                copilot_health = self.copilot.get_health_status()
                health_status["copilot_service"] = "healthy"
                health_status["llm_providers"] = copilot_health.get(
                    "llm_service", {}
                ).get("providers", {})
                health_status["rag_engine"] = copilot_health.get(
                    "rag_engine", "unknown"
                )

            # Check system resources
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)

            if memory_percent > 90 or cpu_percent > 95:
                health_status["system_resources"] = "critical"
                await self._create_alert(
                    AlertLevel.CRITICAL,
                    "High Resource Usage",
                    f"Memory: {memory_percent}%, CPU: {cpu_percent}%",
                    {"memory_percent": memory_percent, "cpu_percent": cpu_percent},
                )
            elif memory_percent > 80 or cpu_percent > 85:
                health_status["system_resources"] = "warning"
                await self._create_alert(
                    AlertLevel.WARNING,
                    "Elevated Resource Usage",
                    f"Memory: {memory_percent}%, CPU: {cpu_percent}%",
                    {"memory_percent": memory_percent, "cpu_percent": cpu_percent},
                )
            else:
                health_status["system_resources"] = "healthy"

            # Check LLM provider health
            for provider, status in health_status["llm_providers"].items():
                if status != "healthy":
                    await self._create_alert(
                        AlertLevel.ERROR,
                        f"LLM Provider Issue",
                        f"Provider {provider} status: {status}",
                        {"provider": provider, "status": status},
                    )

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            await self._create_alert(
                AlertLevel.ERROR,
                "Health Check Failed",
                f"Health check error: {str(e)}",
                {"error": str(e)},
            )

    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # Memory usage
            memory_info = psutil.virtual_memory()
            self.memory_usage_bytes.set(memory_info.used)

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage_percent.set(cpu_percent)

            # Process-specific metrics if available
            try:
                process = psutil.Process()
                process_memory = process.memory_info().rss
                process_cpu = process.cpu_percent()

                # Log detailed process metrics
                self.logger.debug(
                    "Process metrics collected",
                    memory_mb=process_memory / 1024 / 1024,
                    cpu_percent=process_cpu,
                )
            except Exception:
                pass  # Process metrics not critical

        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")

    async def _collect_session_metrics(self):
        """Collect session-related metrics."""
        try:
            # Clean up expired sessions (older than 1 hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            expired_sessions = [
                session_id
                for session_id, timestamp in self.session_tracker.items()
                if timestamp < cutoff_time
            ]

            for session_id in expired_sessions:
                del self.session_tracker[session_id]

            # Update active sessions gauge
            self.active_sessions.set(len(self.session_tracker))

        except Exception as e:
            self.logger.error(f"Session metrics collection failed: {e}")

    async def _collect_performance_snapshot(self):
        """Collect comprehensive performance snapshot."""
        try:
            # Calculate performance metrics
            current_time = datetime.now()

            # Query rate calculation
            recent_queries = [
                t
                for t in self.query_times
                if current_time - datetime.fromtimestamp(t) < timedelta(minutes=1)
            ]
            queries_per_second = len(recent_queries) / 60.0

            # Average response time
            if self.query_times:
                avg_response_time = sum(
                    [time.time() - t for t in list(self.query_times)[-100:]]
                ) / min(100, len(self.query_times))
            else:
                avg_response_time = 0.0

            # Error rate calculation
            total_errors = sum(self.error_counts.values())
            total_requests = len(self.query_times)
            error_rate = (total_errors / max(1, total_requests)) * 100

            # LLM provider health
            llm_health = {}
            if self.copilot and self.copilot.llm_service:
                llm_health = self.copilot.llm_service.get_health_status().get(
                    "providers", {}
                )

            # Create performance snapshot
            snapshot = PerformanceMetrics(
                timestamp=current_time,
                copilot_queries_per_second=queries_per_second,
                average_response_time=avg_response_time,
                llm_provider_health=llm_health,
                memory_usage_mb=psutil.virtual_memory().used / 1024 / 1024,
                cpu_usage_percent=psutil.cpu_percent(),
                active_sessions=len(self.session_tracker),
                error_rate_percent=error_rate,
                cache_hit_rate_percent=0.0,  # Would be calculated from cache metrics
            )

            # Store snapshot
            with self.metrics_lock:
                self.performance_history.append(snapshot)

            # Check for performance alerts
            await self._check_performance_alerts(snapshot)

        except Exception as e:
            self.logger.error(f"Performance snapshot collection failed: {e}")

    async def _check_performance_alerts(self, snapshot: PerformanceMetrics):
        """Check performance metrics and create alerts if needed."""

        # High response time alert
        if snapshot.average_response_time > 10.0:
            await self._create_alert(
                AlertLevel.WARNING,
                "High Response Time",
                f"Average response time: {snapshot.average_response_time:.2f}s",
                {"response_time": snapshot.average_response_time},
            )

        # High error rate alert
        if snapshot.error_rate_percent > 5.0:
            await self._create_alert(
                AlertLevel.ERROR,
                "High Error Rate",
                f"Error rate: {snapshot.error_rate_percent:.1f}%",
                {"error_rate": snapshot.error_rate_percent},
            )

        # Low query rate (possible issue)
        if snapshot.copilot_queries_per_second < 0.1 and len(self.query_times) > 100:
            await self._create_alert(
                AlertLevel.WARNING,
                "Low Query Rate",
                f"Query rate: {snapshot.copilot_queries_per_second:.2f}/s",
                {"query_rate": snapshot.copilot_queries_per_second},
            )

    async def _create_alert(
        self, level: AlertLevel, title: str, description: str, metadata: Dict[str, Any]
    ):
        """Create and store alert."""
        alert_key = f"{title}_{level.value}"

        with self.alerts_lock:
            # Check cooldown period
            if alert_key in self.active_alerts:
                last_alert = self.active_alerts[alert_key]
                if (
                    datetime.now() - last_alert.timestamp
                ).total_seconds() < self.alert_cooldown_period:
                    return  # Skip duplicate alert within cooldown period

            # Create new alert
            alert = Alert(
                level=level,
                title=title,
                description=description,
                timestamp=datetime.now(),
                metadata=metadata,
            )

            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)

        # Log alert
        self.logger.warning(
            f"Alert created: {title}",
            level=level.value,
            description=description,
            metadata=metadata,
        )

        # Send alert to external systems if configured
        await self._send_alert_notification(alert)

    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification to external systems."""
        try:
            # This would integrate with external alerting systems
            # For now, just log structured alert data
            self.logger.info(
                "Alert notification",
                alert_level=alert.level.value,
                alert_title=alert.title,
                alert_description=alert.description,
                alert_timestamp=alert.timestamp.isoformat(),
                alert_metadata=alert.metadata,
            )
        except Exception as e:
            self.logger.error(f"Failed to send alert notification: {e}")

    async def _process_alerts(self):
        """Process active alerts and resolve expired ones."""
        current_time = datetime.now()
        resolved_alerts = []

        with self.alerts_lock:
            for alert_key, alert in list(self.active_alerts.items()):
                # Auto-resolve alerts older than 1 hour
                if (current_time - alert.timestamp).total_seconds() > 3600:
                    alert.resolved = True
                    alert.resolution_time = current_time
                    resolved_alerts.append(alert_key)

            # Remove resolved alerts from active list
            for alert_key in resolved_alerts:
                del self.active_alerts[alert_key]

        if resolved_alerts:
            self.logger.info(f"Auto-resolved {len(resolved_alerts)} alerts")

    # Public interface methods

    def record_copilot_query(
        self,
        query_type: str,
        user_id: str,
        status: str,
        duration: float,
        confidence: float,
        llm_provider: Optional[str] = None,
    ):
        """Record copilot query metrics."""
        try:
            with self.metrics_lock:
                # Update counters and histograms
                self.copilot_queries_total.labels(
                    query_type=query_type, user_id=user_id, status=status
                ).inc()

                if llm_provider:
                    self.copilot_query_duration.labels(
                        query_type=query_type, llm_provider=llm_provider
                    ).observe(duration)

                self.copilot_confidence_score.labels(query_type=query_type).observe(
                    confidence
                )

                # Track timing for performance analysis
                self.query_times.append(time.time())

                # Track session
                if user_id not in self.session_tracker:
                    self.session_tracker[user_id] = datetime.now()

                # Record errors
                if status == "error":
                    self.error_counts[query_type] += 1

        except Exception as e:
            self.logger.error(f"Failed to record copilot query metrics: {e}")

    def record_llm_request(
        self,
        provider: str,
        model: str,
        status: str,
        duration: float,
        tokens_used: int,
        token_type: str = "total",
    ):
        """Record LLM provider request metrics."""
        try:
            self.llm_requests_total.labels(
                provider=provider, model=model, status=status
            ).inc()

            self.llm_request_duration.labels(provider=provider, model=model).observe(
                duration
            )

            self.llm_tokens_used.labels(
                provider=provider, model=model, token_type=token_type
            ).inc(tokens_used)

        except Exception as e:
            self.logger.error(f"Failed to record LLM request metrics: {e}")

    def record_rag_search(self, collection: str, status: str, duration: float):
        """Record RAG search metrics."""
        try:
            self.rag_searches_total.labels(collection=collection, status=status).inc()

            self.rag_search_duration.labels(collection=collection).observe(duration)

        except Exception as e:
            self.logger.error(f"Failed to record RAG search metrics: {e}")

    def record_cache_operation(self, operation: str, cache_type: str, status: str):
        """Record cache operation metrics."""
        try:
            self.cache_operations_total.labels(
                operation=operation, cache_type=cache_type, status=status
            ).inc()

        except Exception as e:
            self.logger.error(f"Failed to record cache operation metrics: {e}")

    def update_cache_hit_rate(self, cache_type: str, hit_rate: float):
        """Update cache hit rate metric."""
        try:
            self.cache_hit_rate.labels(cache_type=cache_type).set(hit_rate * 100)
        except Exception as e:
            self.logger.error(f"Failed to update cache hit rate: {e}")

    def get_metrics_data(self) -> str:
        """Get Prometheus metrics data."""
        return generate_latest(self.registry)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self.metrics_lock:
            if not self.performance_history:
                return {"error": "No performance data available"}

            latest = self.performance_history[-1]

            # Calculate trends
            if len(self.performance_history) > 1:
                previous = self.performance_history[-2]
                response_time_trend = (
                    latest.average_response_time - previous.average_response_time
                )
                error_rate_trend = (
                    latest.error_rate_percent - previous.error_rate_percent
                )
            else:
                response_time_trend = 0.0
                error_rate_trend = 0.0

            return {
                "current_metrics": {
                    "queries_per_second": latest.copilot_queries_per_second,
                    "average_response_time": latest.average_response_time,
                    "error_rate_percent": latest.error_rate_percent,
                    "active_sessions": latest.active_sessions,
                    "memory_usage_mb": latest.memory_usage_mb,
                    "cpu_usage_percent": latest.cpu_usage_percent,
                },
                "trends": {
                    "response_time_trend": response_time_trend,
                    "error_rate_trend": error_rate_trend,
                },
                "llm_provider_status": latest.llm_provider_health,
                "timestamp": latest.timestamp.isoformat(),
            }

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        with self.alerts_lock:
            return {
                "active_alerts": len(self.active_alerts),
                "total_alerts_24h": len(
                    [
                        alert
                        for alert in self.alert_history
                        if (datetime.now() - alert.timestamp).total_seconds() < 86400
                    ]
                ),
                "alerts_by_level": {
                    level.value: len(
                        [
                            alert
                            for alert in self.alert_history
                            if alert.level == level
                            and (datetime.now() - alert.timestamp).total_seconds()
                            < 86400
                        ]
                    )
                    for level in AlertLevel
                },
                "recent_alerts": [
                    {
                        "level": alert.level.value,
                        "title": alert.title,
                        "description": alert.description,
                        "timestamp": alert.timestamp.isoformat(),
                        "resolved": alert.resolved,
                    }
                    for alert in list(self.alert_history)[-10:]
                ],
            }

    async def shutdown(self):
        """Shutdown monitoring service."""
        self.logger.info("Shutting down copilot monitoring service")

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)

        self.is_initialized = False
        self.logger.info("Copilot monitoring service shutdown complete")


# Factory function
def create_copilot_monitoring_service(
    config: Config, copilot: Optional[ProtocolIntelligenceCopilot] = None
) -> CopilotMonitoringService:
    """Create copilot monitoring service."""
    return CopilotMonitoringService(config, copilot)
