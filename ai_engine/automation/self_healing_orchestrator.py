"""
Self-Healing Orchestrator

Automated failure detection and recovery including:
- Health monitoring
- Automatic failover
- Service recovery
- Circuit breaker management
- Incident response automation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of managed components."""

    HSM = "hsm"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    API_ENDPOINT = "api_endpoint"
    KEY_SERVICE = "key_service"
    CERT_SERVICE = "cert_service"
    CRYPTO_PROVIDER = "crypto_provider"
    EXTERNAL_SERVICE = "external_service"


class RecoveryAction(Enum):
    """Types of recovery actions."""

    RESTART = "restart"
    FAILOVER = "failover"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CIRCUIT_BREAK = "circuit_break"
    RECONNECT = "reconnect"
    ROTATE_CREDENTIALS = "rotate_credentials"
    CLEAR_CACHE = "clear_cache"
    ALERT_ONLY = "alert_only"
    MANUAL_INTERVENTION = "manual_intervention"


class IncidentSeverity(Enum):
    """Incident severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check definition."""

    check_id: str
    name: str
    component_type: ComponentType
    check_function: Callable[[], bool]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    success_threshold: int = 1
    enabled: bool = True


@dataclass
class HealthResult:
    """Result of a health check."""

    check_id: str
    component_id: str
    status: HealthStatus
    checked_at: datetime
    response_time_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentState:
    """State of a monitored component."""

    component_id: str
    component_type: ComponentType
    name: str
    status: HealthStatus
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    total_failures: int = 0
    uptime_start: Optional[datetime] = None
    recovery_attempts: int = 0
    last_recovery: Optional[datetime] = None
    circuit_open: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        if not self.uptime_start:
            return 0
        return (datetime.utcnow() - self.uptime_start).total_seconds()

    def failure_rate(self) -> float:
        """Get failure rate as percentage."""
        if self.total_checks == 0:
            return 0
        return (self.total_failures / self.total_checks) * 100


@dataclass
class RecoveryPlan:
    """Recovery plan for a component."""

    plan_id: str
    component_type: ComponentType
    trigger_status: HealthStatus
    actions: List[RecoveryAction]
    max_attempts: int = 3
    cooldown_seconds: int = 300
    escalation_after_failures: int = 5
    escalation_action: RecoveryAction = RecoveryAction.MANUAL_INTERVENTION


@dataclass
class Incident:
    """Incident record."""

    incident_id: str
    component_id: str
    severity: IncidentSeverity
    title: str
    description: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    recovery_actions_taken: List[str] = field(default_factory=list)
    status: str = "open"
    assigned_to: Optional[str] = None
    root_cause: Optional[str] = None

    def duration_seconds(self) -> float:
        """Get incident duration."""
        end = self.resolved_at or datetime.utcnow()
        return (end - self.created_at).total_seconds()


@dataclass
class RecoveryEvent:
    """Recovery action event."""

    event_id: str
    component_id: str
    action: RecoveryAction
    triggered_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    message: str = ""
    incident_id: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker for component protection."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._failures = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = "closed"  # closed, open, half_open
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        with self._lock:
            if self._state == "open":
                # Check if recovery timeout has passed
                if self._last_failure_time:
                    elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = "half_open"
                        self._half_open_calls = 0
                        return False
                return True
            return False

    def record_success(self) -> None:
        """Record successful call."""
        with self._lock:
            if self._state == "half_open":
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = "closed"
                    self._failures = 0
            elif self._state == "closed":
                self._failures = max(0, self._failures - 1)

    def record_failure(self) -> None:
        """Record failed call."""
        with self._lock:
            self._failures += 1
            self._last_failure_time = datetime.utcnow()

            if self._state == "half_open":
                self._state = "open"
            elif self._failures >= self.failure_threshold:
                self._state = "open"

    def reset(self) -> None:
        """Reset circuit breaker."""
        with self._lock:
            self._failures = 0
            self._state = "closed"
            self._last_failure_time = None
            self._half_open_calls = 0

    @property
    def state(self) -> str:
        """Get current state."""
        return self._state


class SelfHealingOrchestrator:
    """
    Self-healing orchestrator for automated recovery.

    Capabilities:
    - Continuous health monitoring
    - Automatic failure detection
    - Recovery plan execution
    - Circuit breaker management
    - Incident tracking
    """

    def __init__(
        self,
        alert_callback: Optional[Callable[[Incident], None]] = None,
        recovery_callback: Optional[Callable[[RecoveryEvent], None]] = None,
    ):
        self._alert_callback = alert_callback
        self._recovery_callback = recovery_callback

        # Component tracking
        self._components: Dict[str, ComponentState] = {}
        self._health_checks: Dict[str, HealthCheck] = {}
        self._recovery_plans: Dict[str, RecoveryPlan] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # History
        self._health_history: Dict[str, deque] = {}  # component_id -> recent results
        self._incidents: Dict[str, Incident] = {}
        self._recovery_events: List[RecoveryEvent] = []

        # Threading
        self._lock = threading.RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Recovery handlers
        self._recovery_handlers: Dict[RecoveryAction, Callable] = {}

        # Initialize default recovery plans
        self._init_default_recovery_plans()

    def register_component(
        self,
        component_id: str,
        component_type: ComponentType,
        name: str,
        health_check: Optional[Callable[[], bool]] = None,
        check_interval: int = 30,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComponentState:
        """
        Register a component for monitoring.

        Args:
            component_id: Unique component ID
            component_type: Type of component
            name: Human-readable name
            health_check: Health check function
            check_interval: Check interval in seconds
            metadata: Optional metadata

        Returns:
            ComponentState
        """
        with self._lock:
            state = ComponentState(
                component_id=component_id,
                component_type=component_type,
                name=name,
                status=HealthStatus.UNKNOWN,
                uptime_start=datetime.utcnow(),
                metadata=metadata or {},
            )

            self._components[component_id] = state
            self._health_history[component_id] = deque(maxlen=100)

            # Create circuit breaker
            self._circuit_breakers[component_id] = CircuitBreaker()

            # Register health check if provided
            if health_check:
                check = HealthCheck(
                    check_id=f"{component_id}_check",
                    name=f"{name} Health Check",
                    component_type=component_type,
                    check_function=health_check,
                    interval_seconds=check_interval,
                )
                self._health_checks[check.check_id] = check

            logger.info(f"Registered component {component_id} ({component_type.value})")

            return state

    def unregister_component(self, component_id: str) -> bool:
        """Unregister a component."""
        with self._lock:
            if component_id in self._components:
                del self._components[component_id]
                self._health_history.pop(component_id, None)
                self._circuit_breakers.pop(component_id, None)

                # Remove associated health checks
                to_remove = [k for k, v in self._health_checks.items() if k.startswith(component_id)]
                for k in to_remove:
                    del self._health_checks[k]

                return True
            return False

    def add_health_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        with self._lock:
            self._health_checks[check.check_id] = check

    def add_recovery_plan(self, plan: RecoveryPlan) -> None:
        """Add a recovery plan."""
        with self._lock:
            self._recovery_plans[plan.plan_id] = plan

    def register_recovery_handler(
        self,
        action: RecoveryAction,
        handler: Callable[[str, ComponentState], bool],
    ) -> None:
        """Register a handler for a recovery action."""
        self._recovery_handlers[action] = handler

    def check_health(self, component_id: str) -> HealthResult:
        """
        Perform health check on a component.

        Args:
            component_id: Component to check

        Returns:
            HealthResult
        """
        with self._lock:
            if component_id not in self._components:
                return HealthResult(
                    check_id="unknown",
                    component_id=component_id,
                    status=HealthStatus.UNKNOWN,
                    checked_at=datetime.utcnow(),
                    response_time_ms=0,
                    message="Component not registered",
                )

            component = self._components[component_id]

            # Check circuit breaker
            circuit_breaker = self._circuit_breakers.get(component_id)
            if circuit_breaker and circuit_breaker.is_open:
                return HealthResult(
                    check_id=f"{component_id}_check",
                    component_id=component_id,
                    status=HealthStatus.CRITICAL,
                    checked_at=datetime.utcnow(),
                    response_time_ms=0,
                    message="Circuit breaker is open",
                )

            # Find health check
            check = next((c for c in self._health_checks.values() if c.check_id.startswith(component_id)), None)

            if not check or not check.enabled:
                return HealthResult(
                    check_id=f"{component_id}_check",
                    component_id=component_id,
                    status=HealthStatus.UNKNOWN,
                    checked_at=datetime.utcnow(),
                    response_time_ms=0,
                    message="No health check configured",
                )

            # Execute health check
            start_time = time.time()
            try:
                is_healthy = check.check_function()
                response_time = (time.time() - start_time) * 1000

                status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
                message = "OK" if is_healthy else "Health check failed"

            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                status = HealthStatus.CRITICAL
                message = f"Health check error: {str(e)}"
                is_healthy = False

            result = HealthResult(
                check_id=check.check_id,
                component_id=component_id,
                status=status,
                checked_at=datetime.utcnow(),
                response_time_ms=response_time,
                message=message,
            )

            # Update component state
            self._update_component_state(component, result, check)

            # Record in history
            self._health_history[component_id].append(result)

            # Handle failures
            if not is_healthy:
                circuit_breaker.record_failure()
                self._handle_failure(component, result)
            else:
                circuit_breaker.record_success()

            return result

    def _update_component_state(
        self,
        component: ComponentState,
        result: HealthResult,
        check: HealthCheck,
    ) -> None:
        """Update component state based on health result."""
        component.last_check = result.checked_at
        component.total_checks += 1

        if result.status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL):
            component.consecutive_failures += 1
            component.consecutive_successes = 0
            component.total_failures += 1

            # Update status based on failure threshold
            if component.consecutive_failures >= check.failure_threshold:
                component.status = HealthStatus.CRITICAL
            elif component.consecutive_failures > 0:
                component.status = HealthStatus.DEGRADED

        else:
            component.consecutive_successes += 1
            component.consecutive_failures = 0

            # Update status based on success threshold
            if component.consecutive_successes >= check.success_threshold:
                component.status = HealthStatus.HEALTHY

    def _handle_failure(
        self,
        component: ComponentState,
        result: HealthResult,
    ) -> None:
        """Handle component failure."""
        # Find recovery plan
        plan = self._find_recovery_plan(component)

        if not plan:
            logger.warning(f"No recovery plan for {component.component_id}")
            return

        # Check if we should attempt recovery
        if not self._should_attempt_recovery(component, plan):
            return

        # Create incident if not exists
        incident = self._get_or_create_incident(component, result)

        # Execute recovery actions
        for action in plan.actions:
            if component.recovery_attempts >= plan.max_attempts:
                # Escalate
                action = plan.escalation_action
                incident.severity = IncidentSeverity.CRITICAL

            event = self._execute_recovery(component, action, incident.incident_id)

            if event.success:
                logger.info(f"Recovery action {action.value} succeeded for {component.component_id}")
                break
            else:
                logger.warning(f"Recovery action {action.value} failed for {component.component_id}")

        # Update recovery tracking
        component.recovery_attempts += 1
        component.last_recovery = datetime.utcnow()

    def _find_recovery_plan(self, component: ComponentState) -> Optional[RecoveryPlan]:
        """Find recovery plan for component."""
        # Look for component-specific plan first
        specific_plan = self._recovery_plans.get(f"{component.component_id}_plan")
        if specific_plan:
            return specific_plan

        # Look for type-based plan
        for plan in self._recovery_plans.values():
            if plan.component_type == component.component_type:
                return plan

        return None

    def _should_attempt_recovery(
        self,
        component: ComponentState,
        plan: RecoveryPlan,
    ) -> bool:
        """Check if recovery should be attempted."""
        # Check cooldown
        if component.last_recovery:
            elapsed = (datetime.utcnow() - component.last_recovery).total_seconds()
            if elapsed < plan.cooldown_seconds:
                return False

        # Check max attempts
        if component.recovery_attempts >= plan.max_attempts + plan.escalation_after_failures:
            return False

        return True

    def _get_or_create_incident(
        self,
        component: ComponentState,
        result: HealthResult,
    ) -> Incident:
        """Get existing incident or create new one."""
        # Check for open incident
        for incident in self._incidents.values():
            if incident.component_id == component.component_id and incident.status == "open":
                return incident

        # Create new incident
        severity = IncidentSeverity.HIGH if result.status == HealthStatus.CRITICAL else IncidentSeverity.MEDIUM

        incident = Incident(
            incident_id=str(uuid.uuid4()),
            component_id=component.component_id,
            severity=severity,
            title=f"{component.name} failure",
            description=result.message,
            created_at=datetime.utcnow(),
        )

        self._incidents[incident.incident_id] = incident

        # Alert callback
        if self._alert_callback:
            try:
                self._alert_callback(incident)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Created incident {incident.incident_id} for {component.component_id}")

        return incident

    def _execute_recovery(
        self,
        component: ComponentState,
        action: RecoveryAction,
        incident_id: str,
    ) -> RecoveryEvent:
        """Execute a recovery action."""
        event = RecoveryEvent(
            event_id=str(uuid.uuid4()),
            component_id=component.component_id,
            action=action,
            triggered_at=datetime.utcnow(),
            incident_id=incident_id,
        )

        try:
            # Get handler
            handler = self._recovery_handlers.get(action)

            if handler:
                success = handler(component.component_id, component)
                event.success = success
                event.message = "Recovery action completed" if success else "Recovery action failed"
            else:
                # Default handling
                event.success = self._default_recovery(component, action)
                event.message = f"Default recovery for {action.value}"

            event.completed_at = datetime.utcnow()

        except Exception as e:
            event.success = False
            event.message = f"Recovery error: {str(e)}"
            event.completed_at = datetime.utcnow()

        self._recovery_events.append(event)

        # Update incident
        incident = self._incidents.get(incident_id)
        if incident:
            incident.recovery_actions_taken.append(f"{action.value}: {event.message}")

        # Recovery callback
        if self._recovery_callback:
            try:
                self._recovery_callback(event)
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")

        return event

    def _default_recovery(
        self,
        component: ComponentState,
        action: RecoveryAction,
    ) -> bool:
        """Default recovery handling."""
        if action == RecoveryAction.CIRCUIT_BREAK:
            circuit_breaker = self._circuit_breakers.get(component.component_id)
            if circuit_breaker:
                component.circuit_open = True
                return True

        elif action == RecoveryAction.ALERT_ONLY:
            return True  # Alert was already sent

        elif action == RecoveryAction.RECONNECT:
            # Attempt to reset state
            component.consecutive_failures = 0
            component.status = HealthStatus.UNKNOWN
            return True

        return False

    def resolve_incident(
        self,
        incident_id: str,
        root_cause: str = "",
    ) -> bool:
        """Resolve an incident."""
        with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return False

            incident.resolved_at = datetime.utcnow()
            incident.status = "resolved"
            incident.root_cause = root_cause

            # Reset component state
            component = self._components.get(incident.component_id)
            if component:
                component.recovery_attempts = 0
                circuit_breaker = self._circuit_breakers.get(component.component_id)
                if circuit_breaker:
                    circuit_breaker.reset()
                component.circuit_open = False

            logger.info(f"Resolved incident {incident_id}")

            return True

    def start_monitoring(
        self,
        check_interval_seconds: int = 30,
    ) -> None:
        """Start health monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._shutdown_event.clear()

        def monitor_loop():
            while not self._shutdown_event.is_set():
                try:
                    self._run_health_checks()
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")

                self._shutdown_event.wait(check_interval_seconds)

        self._monitor_thread = threading.Thread(
            target=monitor_loop,
            daemon=True,
            name="health-monitor",
        )
        self._monitor_thread.start()
        logger.info("Health monitoring started")

    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._shutdown_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")

    def _run_health_checks(self) -> None:
        """Run all health checks."""
        for component_id in list(self._components.keys()):
            try:
                self.check_health(component_id)
            except Exception as e:
                logger.error(f"Health check failed for {component_id}: {e}")

    def _init_default_recovery_plans(self) -> None:
        """Initialize default recovery plans."""
        # HSM recovery plan
        self._recovery_plans["hsm_recovery"] = RecoveryPlan(
            plan_id="hsm_recovery",
            component_type=ComponentType.HSM,
            trigger_status=HealthStatus.UNHEALTHY,
            actions=[
                RecoveryAction.RECONNECT,
                RecoveryAction.FAILOVER,
                RecoveryAction.CIRCUIT_BREAK,
            ],
            max_attempts=3,
            cooldown_seconds=60,
        )

        # Database recovery plan
        self._recovery_plans["database_recovery"] = RecoveryPlan(
            plan_id="database_recovery",
            component_type=ComponentType.DATABASE,
            trigger_status=HealthStatus.UNHEALTHY,
            actions=[
                RecoveryAction.RECONNECT,
                RecoveryAction.FAILOVER,
            ],
            max_attempts=5,
            cooldown_seconds=30,
        )

        # API endpoint recovery
        self._recovery_plans["api_recovery"] = RecoveryPlan(
            plan_id="api_recovery",
            component_type=ComponentType.API_ENDPOINT,
            trigger_status=HealthStatus.UNHEALTHY,
            actions=[
                RecoveryAction.CIRCUIT_BREAK,
                RecoveryAction.ALERT_ONLY,
            ],
            max_attempts=3,
            cooldown_seconds=120,
        )

        # Crypto provider recovery
        self._recovery_plans["crypto_recovery"] = RecoveryPlan(
            plan_id="crypto_recovery",
            component_type=ComponentType.CRYPTO_PROVIDER,
            trigger_status=HealthStatus.UNHEALTHY,
            actions=[
                RecoveryAction.RECONNECT,
                RecoveryAction.ROTATE_CREDENTIALS,
                RecoveryAction.FAILOVER,
            ],
            max_attempts=3,
            cooldown_seconds=60,
        )

    def get_component_status(self, component_id: str) -> Optional[ComponentState]:
        """Get component status."""
        return self._components.get(component_id)

    def get_all_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all components."""
        return {
            comp_id: {
                "name": comp.name,
                "type": comp.component_type.value,
                "status": comp.status.value,
                "uptime_seconds": comp.uptime_seconds(),
                "failure_rate": comp.failure_rate(),
                "consecutive_failures": comp.consecutive_failures,
                "circuit_open": comp.circuit_open,
            }
            for comp_id, comp in self._components.items()
        }

    def get_open_incidents(self) -> List[Incident]:
        """Get all open incidents."""
        return [i for i in self._incidents.values() if i.status == "open"]

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        return self._incidents.get(incident_id)

    def get_recovery_events(
        self,
        component_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[RecoveryEvent]:
        """Get recovery events."""
        events = self._recovery_events

        if component_id:
            events = [e for e in events if e.component_id == component_id]

        return events[-limit:]

    def get_health_history(
        self,
        component_id: str,
        limit: int = 100,
    ) -> List[HealthResult]:
        """Get health check history for a component."""
        history = self._health_history.get(component_id, deque())
        return list(history)[-limit:]

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        components = list(self._components.values())

        if not components:
            return {"status": "unknown", "message": "No components registered"}

        healthy = sum(1 for c in components if c.status == HealthStatus.HEALTHY)
        degraded = sum(1 for c in components if c.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for c in components if c.status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL))

        if unhealthy > 0:
            status = "critical" if any(c.status == HealthStatus.CRITICAL for c in components) else "unhealthy"
        elif degraded > 0:
            status = "degraded"
        elif healthy == len(components):
            status = "healthy"
        else:
            status = "unknown"

        return {
            "status": status,
            "total_components": len(components),
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "open_incidents": len(self.get_open_incidents()),
            "components": self.get_all_component_status(),
        }
