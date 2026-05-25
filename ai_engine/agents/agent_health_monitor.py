"""
QBITEL - Agent Health Monitor & Observability

Centralized health monitoring for the entire multi-agent system.

Features:
- Real-time agent health aggregation across all subsystems
- Circuit breaker dashboard integration
- LLM service health tracking
- Memory system health
- Automated alerting on degradation
- Health check endpoints for Kubernetes probes
- Historical health snapshots for trend analysis
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from prometheus_client import Counter, Gauge, Histogram, Info

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

HEALTH_CHECK_DURATION = Histogram(
    "qbitel_health_check_duration_seconds",
    "Health check execution time",
    ["component"],
)
HEALTH_STATUS = Gauge(
    "qbitel_health_status",
    "Component health status (0=unhealthy, 1=degraded, 2=healthy)",
    ["component"],
)
HEALTH_ALERTS = Counter(
    "qbitel_health_alerts_total",
    "Health alerts triggered",
    ["component", "severity"],
)
SYSTEM_INFO = Info(
    "qbitel_system",
    "QBITEL system information",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    component: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: float = 0.0
    consecutive_failures: int = 0


@dataclass
class HealthAlert:
    """A health alert event."""

    alert_id: str
    component: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False


@dataclass
class HealthSnapshot:
    """Point-in-time health snapshot for trend analysis."""

    timestamp: datetime
    overall_status: HealthStatus
    component_statuses: Dict[str, HealthStatus]
    open_circuits: int = 0
    total_agents: int = 0
    active_agents: int = 0
    error_agents: int = 0
    llm_healthy_models: int = 0
    memory_entries: int = 0


@dataclass
class HealthMonitorConfig:
    """Configuration for the health monitor."""

    # Check intervals
    health_check_interval_seconds: float = 30.0
    deep_check_interval_seconds: float = 300.0
    snapshot_interval_seconds: float = 60.0

    # Alerting
    enable_alerting: bool = True
    max_consecutive_failures_warning: int = 3
    max_consecutive_failures_critical: int = 5

    # History
    max_snapshots: int = 1440  # 24 hours at 1-minute intervals
    max_alerts: int = 1000

    # Thresholds
    agent_error_rate_threshold: float = 0.2  # 20% error agents = degraded
    circuit_open_threshold: int = 2  # 2+ open circuits = degraded
    llm_min_healthy_models: int = 1  # At least 1 healthy model required

    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Health Monitor
# ---------------------------------------------------------------------------


class AgentHealthMonitor:
    """
    Centralized health monitor for the QBITEL multi-agent system.

    Integrates with:
    - UnifiedAgentRegistry for agent health
    - CircuitBreakerRegistry for circuit health
    - AgentLLMService for LLM health
    - AgentMemoryManager for memory health
    - MultiAgentOrchestrator for orchestration health

    Usage:
        monitor = AgentHealthMonitor(config)
        monitor.register_component("orchestrator", orchestrator_health_check)
        monitor.register_component("memory", memory_health_check)
        await monitor.start()

        # Get health
        health = await monitor.get_health()
        # Returns: {"status": "healthy", "components": {...}, ...}

        # Kubernetes probe
        is_healthy = await monitor.is_healthy()  # for liveness
        is_ready = await monitor.is_ready()      # for readiness
    """

    def __init__(self, config: Optional[HealthMonitorConfig] = None):
        self.config = config or HealthMonitorConfig()

        # Component health checks — callable returns ComponentHealth
        self._health_checks: Dict[str, Callable] = {}

        # Current state
        self._component_health: Dict[str, ComponentHealth] = {}
        self._overall_status = HealthStatus.UNKNOWN

        # History
        self._snapshots: List[HealthSnapshot] = []
        self._alerts: List[HealthAlert] = []

        # Alert callbacks
        self._alert_callbacks: List[Callable] = []

        # Integration references (set after construction)
        self._agent_registry = None  # UnifiedAgentRegistry
        self._circuit_registry = None  # CircuitBreakerRegistry
        self._llm_service = None  # AgentLLMService
        self._memory_manager = None  # AgentMemoryManager
        self._orchestrator = None  # MultiAgentOrchestrator

        # Lifecycle
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._lock = asyncio.Lock()

        self.logger = logging.getLogger(f"{__name__}.HealthMonitor")

    # ------ Integration Setup ------

    def set_agent_registry(self, registry: Any) -> None:
        """Set the unified agent registry reference."""
        self._agent_registry = registry
        self.logger.info("Agent registry connected to health monitor")

    def set_circuit_registry(self, registry: Any) -> None:
        """Set the circuit breaker registry reference."""
        self._circuit_registry = registry
        self.logger.info("Circuit breaker registry connected to health monitor")

    def set_llm_service(self, service: Any) -> None:
        """Set the agent LLM service reference."""
        self._llm_service = service
        self.logger.info("LLM service connected to health monitor")

    def set_memory_manager(self, manager: Any) -> None:
        """Set the memory manager reference."""
        self._memory_manager = manager
        self.logger.info("Memory manager connected to health monitor")

    def set_orchestrator(self, orchestrator: Any) -> None:
        """Set the orchestrator reference."""
        self._orchestrator = orchestrator
        self.logger.info("Orchestrator connected to health monitor")

    # ------ Component Registration ------

    def register_component(
        self,
        name: str,
        health_check: Callable,
    ) -> None:
        """
        Register a component health check function.

        The function should be an async callable returning ComponentHealth.
        """
        self._health_checks[name] = health_check
        self.logger.info(f"Registered health check: {name}")

    # ------ Lifecycle ------

    async def start(self) -> None:
        """Start the health monitor background tasks."""
        if self._running:
            return

        self._running = True

        # Register built-in health checks
        self._register_builtin_checks()

        # Start background loops
        self._tasks.append(
            asyncio.create_task(self._periodic_health_check())
        )
        self._tasks.append(
            asyncio.create_task(self._periodic_snapshot())
        )

        # Set system info
        try:
            SYSTEM_INFO.info({
                "version": "1.0.0",
                "component": "agent_health_monitor",
            })
        except Exception:
            pass

        self.logger.info("Agent Health Monitor started")

    async def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.logger.info("Agent Health Monitor stopped")

    # ------ Health Checks ------

    async def check_health(self) -> Dict[str, Any]:
        """
        Run all health checks and return comprehensive health report.

        Returns a dictionary suitable for JSON serialization / API response.
        """
        start_time = time.time()
        components: Dict[str, ComponentHealth] = {}

        for name, check_fn in self._health_checks.items():
            check_start = time.time()
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    health = await asyncio.wait_for(check_fn(), timeout=10.0)
                else:
                    health = check_fn()

                health.response_time_ms = (time.time() - check_start) * 1000
                health.last_check = datetime.utcnow()

                # Reset consecutive failures on success
                if health.status != HealthStatus.UNHEALTHY:
                    health.consecutive_failures = 0
                else:
                    prev = self._component_health.get(name)
                    health.consecutive_failures = (
                        (prev.consecutive_failures + 1) if prev else 1
                    )

                components[name] = health

                # Update metrics
                status_val = {
                    HealthStatus.HEALTHY: 2,
                    HealthStatus.DEGRADED: 1,
                    HealthStatus.UNHEALTHY: 0,
                    HealthStatus.UNKNOWN: -1,
                }
                HEALTH_STATUS.labels(component=name).set(
                    status_val.get(health.status, -1)
                )
                HEALTH_CHECK_DURATION.labels(component=name).observe(
                    health.response_time_ms / 1000
                )

            except asyncio.TimeoutError:
                components[name] = ComponentHealth(
                    component=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Health check timed out",
                )
            except Exception as e:
                components[name] = ComponentHealth(
                    component=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check error: {str(e)}",
                )

        # Update stored state
        async with self._lock:
            self._component_health = components
            self._overall_status = self._calculate_overall(components)

        # Check for alerts
        if self.config.enable_alerting:
            await self._check_alerts(components)

        total_time = time.time() - start_time

        return {
            "status": self._overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "check_duration_ms": round(total_time * 1000, 1),
            "components": {
                name: {
                    "status": h.status.value,
                    "message": h.message,
                    "response_time_ms": round(h.response_time_ms, 1),
                    "details": h.details,
                }
                for name, h in components.items()
            },
        }

    async def is_healthy(self) -> bool:
        """Kubernetes liveness probe — is the system alive?"""
        return self._overall_status != HealthStatus.UNHEALTHY

    async def is_ready(self) -> bool:
        """Kubernetes readiness probe — is the system ready to serve?"""
        return self._overall_status == HealthStatus.HEALTHY

    # ------ Built-in Health Checks ------

    def _register_builtin_checks(self) -> None:
        """Register health checks for known integrated components."""
        if self._agent_registry is not None:
            self.register_component("agents", self._check_agents)

        if self._circuit_registry is not None:
            self.register_component("circuits", self._check_circuits)

        if self._llm_service is not None:
            self.register_component("llm", self._check_llm)

        if self._memory_manager is not None:
            self.register_component("memory", self._check_memory)

        if self._orchestrator is not None:
            self.register_component("orchestrator", self._check_orchestrator)

    async def _check_agents(self) -> ComponentHealth:
        """Check agent subsystem health."""
        if self._agent_registry is None:
            return ComponentHealth(
                component="agents",
                status=HealthStatus.UNKNOWN,
                message="Agent registry not connected",
            )

        dashboard = self._agent_registry.get_dashboard()
        total = dashboard.get("total_agents", 0)

        if total == 0:
            return ComponentHealth(
                component="agents",
                status=HealthStatus.DEGRADED,
                message="No agents registered",
                details=dashboard,
            )

        # Count error agents across subsystems
        error_count = 0
        for subsystem_info in dashboard.get("subsystems", {}).values():
            error_count += subsystem_info.get("error", 0)

        error_rate = error_count / total if total > 0 else 0

        if error_rate >= self.config.agent_error_rate_threshold:
            status = HealthStatus.DEGRADED
            msg = f"{error_count}/{total} agents in error state"
        else:
            status = HealthStatus.HEALTHY
            msg = f"{total} agents registered, {error_count} errors"

        return ComponentHealth(
            component="agents",
            status=status,
            message=msg,
            details=dashboard,
        )

    async def _check_circuits(self) -> ComponentHealth:
        """Check circuit breaker health."""
        if self._circuit_registry is None:
            return ComponentHealth(
                component="circuits",
                status=HealthStatus.UNKNOWN,
                message="Circuit registry not connected",
            )

        dashboard = self._circuit_registry.get_dashboard()
        open_count = dashboard.get("open_circuits", 0)

        if open_count >= self.config.circuit_open_threshold:
            return ComponentHealth(
                component="circuits",
                status=HealthStatus.DEGRADED,
                message=f"{open_count} circuits open",
                details=dashboard,
            )

        return ComponentHealth(
            component="circuits",
            status=HealthStatus.HEALTHY,
            message=f"{dashboard.get('total_breakers', 0)} circuits, {open_count} open",
            details=dashboard,
        )

    async def _check_llm(self) -> ComponentHealth:
        """Check LLM service health."""
        if self._llm_service is None:
            return ComponentHealth(
                component="llm",
                status=HealthStatus.UNKNOWN,
                message="LLM service not connected",
            )

        healthy_models = self._llm_service.get_healthy_models()
        circuit_dashboard = self._llm_service.get_circuit_dashboard()

        if len(healthy_models) < self.config.llm_min_healthy_models:
            return ComponentHealth(
                component="llm",
                status=HealthStatus.UNHEALTHY,
                message=f"Only {len(healthy_models)} healthy models (need {self.config.llm_min_healthy_models})",
                details={
                    "healthy_models": healthy_models,
                    "circuits": circuit_dashboard,
                },
            )

        return ComponentHealth(
            component="llm",
            status=HealthStatus.HEALTHY,
            message=f"{len(healthy_models)} healthy models available",
            details={
                "healthy_models": healthy_models,
                "token_usage": self._llm_service.get_token_usage(),
            },
        )

    async def _check_memory(self) -> ComponentHealth:
        """Check memory subsystem health."""
        if self._memory_manager is None:
            return ComponentHealth(
                component="memory",
                status=HealthStatus.UNKNOWN,
                message="Memory manager not connected",
            )

        stats = self._memory_manager.get_stats()
        return ComponentHealth(
            component="memory",
            status=HealthStatus.HEALTHY,
            message=(
                f"Episodic: {stats.get('episodic_count', 0)}, "
                f"Semantic: {stats.get('semantic_count', 0)}"
            ),
            details=stats,
        )

    async def _check_orchestrator(self) -> ComponentHealth:
        """Check orchestrator health."""
        if self._orchestrator is None:
            return ComponentHealth(
                component="orchestrator",
                status=HealthStatus.UNKNOWN,
                message="Orchestrator not connected",
            )

        is_running = getattr(self._orchestrator, "_running", False)
        stats = getattr(self._orchestrator, "stats", {})

        if not is_running:
            return ComponentHealth(
                component="orchestrator",
                status=HealthStatus.UNHEALTHY,
                message="Orchestrator is not running",
                details=stats,
            )

        return ComponentHealth(
            component="orchestrator",
            status=HealthStatus.HEALTHY,
            message=(
                f"Tasks: {stats.get('tasks_completed', 0)} completed, "
                f"{stats.get('tasks_failed', 0)} failed"
            ),
            details=stats,
        )

    # ------ Alerting ------

    async def _check_alerts(
        self, components: Dict[str, ComponentHealth]
    ) -> None:
        """Check for alert conditions and emit alerts."""
        import uuid as _uuid

        for name, health in components.items():
            if health.status == HealthStatus.UNHEALTHY:
                if health.consecutive_failures >= self.config.max_consecutive_failures_critical:
                    await self._emit_alert(HealthAlert(
                        alert_id=str(_uuid.uuid4()),
                        component=name,
                        severity=AlertSeverity.CRITICAL,
                        message=f"CRITICAL: {name} has been unhealthy for {health.consecutive_failures} checks",
                        details=health.details,
                    ))
                elif health.consecutive_failures >= self.config.max_consecutive_failures_warning:
                    await self._emit_alert(HealthAlert(
                        alert_id=str(_uuid.uuid4()),
                        component=name,
                        severity=AlertSeverity.WARNING,
                        message=f"WARNING: {name} unhealthy ({health.consecutive_failures} consecutive failures)",
                        details=health.details,
                    ))

    async def _emit_alert(self, alert: HealthAlert) -> None:
        """Emit a health alert."""
        self._alerts.append(alert)
        if len(self._alerts) > self.config.max_alerts:
            self._alerts = self._alerts[-self.config.max_alerts:]

        HEALTH_ALERTS.labels(
            component=alert.component,
            severity=alert.severity.value,
        ).inc()

        self.logger.warning(
            f"Health alert [{alert.severity.value}] {alert.component}: "
            f"{alert.message}"
        )

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")

    def on_alert(self, callback: Callable) -> None:
        """Register an alert callback."""
        self._alert_callbacks.append(callback)

    # ------ Background Loops ------

    async def _periodic_health_check(self) -> None:
        """Periodically run health checks."""
        while self._running:
            try:
                await self.check_health()
                await asyncio.sleep(self.config.health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Periodic health check error: {e}")
                await asyncio.sleep(self.config.health_check_interval_seconds)

    async def _periodic_snapshot(self) -> None:
        """Periodically capture health snapshots for trends."""
        while self._running:
            try:
                await asyncio.sleep(self.config.snapshot_interval_seconds)
                snapshot = self._capture_snapshot()
                self._snapshots.append(snapshot)
                if len(self._snapshots) > self.config.max_snapshots:
                    self._snapshots = self._snapshots[-self.config.max_snapshots:]
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Snapshot error: {e}")

    def _capture_snapshot(self) -> HealthSnapshot:
        """Capture a point-in-time health snapshot."""
        component_statuses = {
            name: h.status for name, h in self._component_health.items()
        }

        # Get agent stats
        total_agents = 0
        active_agents = 0
        error_agents = 0
        if self._agent_registry:
            dashboard = self._agent_registry.get_dashboard()
            total_agents = dashboard.get("total_agents", 0)
            for sub in dashboard.get("subsystems", {}).values():
                active_agents += sub.get("busy", 0)
                error_agents += sub.get("error", 0)

        # Get circuit stats
        open_circuits = 0
        if self._circuit_registry:
            open_circuits = len(self._circuit_registry.get_open_circuits())

        # Get LLM stats
        llm_healthy = 0
        if self._llm_service:
            llm_healthy = len(self._llm_service.get_healthy_models())

        # Get memory stats
        memory_entries = 0
        if self._memory_manager:
            stats = self._memory_manager.get_stats()
            memory_entries = (
                stats.get("episodic_count", 0) + stats.get("semantic_count", 0)
            )

        return HealthSnapshot(
            timestamp=datetime.utcnow(),
            overall_status=self._overall_status,
            component_statuses=component_statuses,
            open_circuits=open_circuits,
            total_agents=total_agents,
            active_agents=active_agents,
            error_agents=error_agents,
            llm_healthy_models=llm_healthy,
            memory_entries=memory_entries,
        )

    # ------ Helpers ------

    def _calculate_overall(
        self, components: Dict[str, ComponentHealth]
    ) -> HealthStatus:
        """Calculate overall system health from component statuses."""
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [h.status for h in components.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    # ------ Introspection ------

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get recent alerts, optionally filtered."""
        alerts = self._alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return [
            {
                "alert_id": a.alert_id,
                "component": a.component,
                "severity": a.severity.value,
                "message": a.message,
                "timestamp": a.timestamp.isoformat(),
                "acknowledged": a.acknowledged,
            }
            for a in alerts[-limit:]
        ]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_trend(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get health trend over the last N minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "status": s.overall_status.value,
                "total_agents": s.total_agents,
                "active_agents": s.active_agents,
                "error_agents": s.error_agents,
                "open_circuits": s.open_circuits,
                "llm_healthy_models": s.llm_healthy_models,
            }
            for s in self._snapshots
            if s.timestamp >= cutoff
        ]
