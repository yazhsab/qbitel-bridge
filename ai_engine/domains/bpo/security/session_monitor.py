"""
Agent Session Security Monitor Module

Real-time monitoring of agent sessions for security anomalies.

Detects:
- Unusual data access patterns (bulk customer lookups)
- Screen capture/screenshot attempts
- Clipboard copy of sensitive data
- USB device insertion
- Unauthorized application access
- Session sharing or replay attacks
- Idle session exploitation

This module provides continuous monitoring of agent desktop sessions
in BPO environments, detecting insider threats, compromised credentials,
and social engineering attacks against agents.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import logging
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SessionState(Enum):
    """State of an agent session."""

    INITIALIZING = auto()     # Session being set up
    ACTIVE = auto()           # Agent actively working
    IDLE = auto()             # Agent idle (no input)
    LOCKED = auto()           # Session locked (idle timeout)
    SUSPENDED = auto()        # Session suspended by policy
    TERMINATED = auto()       # Session terminated
    COMPROMISED = auto()      # Session flagged as compromised


class EventType(Enum):
    """Types of session events tracked by the monitor."""

    # Authentication events
    LOGIN = auto()
    LOGOUT = auto()
    MFA_CHALLENGE = auto()
    MFA_SUCCESS = auto()
    MFA_FAILURE = auto()
    PASSWORD_CHANGE = auto()
    SESSION_LOCK = auto()
    SESSION_UNLOCK = auto()

    # Data access events
    CUSTOMER_LOOKUP = auto()
    CUSTOMER_RECORD_VIEW = auto()
    CUSTOMER_RECORD_EDIT = auto()
    BULK_DATA_EXPORT = auto()
    REPORT_GENERATION = auto()
    SEARCH_QUERY = auto()

    # Desktop security events
    SCREEN_CAPTURE_ATTEMPT = auto()
    CLIPBOARD_COPY = auto()
    CLIPBOARD_PASTE = auto()
    PRINT_ATTEMPT = auto()
    USB_DEVICE_INSERT = auto()
    USB_DEVICE_REMOVE = auto()
    UNAUTHORIZED_APP_LAUNCH = auto()
    BROWSER_NAVIGATION = auto()
    FILE_DOWNLOAD = auto()
    FILE_UPLOAD = auto()

    # Session integrity events
    IP_ADDRESS_CHANGE = auto()
    USER_AGENT_CHANGE = auto()
    CONCURRENT_SESSION = auto()
    SESSION_REPLAY = auto()
    TOKEN_REUSE = auto()

    # Network events
    VPN_DISCONNECT = auto()
    NETWORK_CHANGE = auto()
    PROXY_DETECTED = auto()

    # Call events
    CALL_START = auto()
    CALL_END = auto()
    CALL_TRANSFER = auto()
    HOLD_START = auto()
    HOLD_END = auto()


class AnomalyType(Enum):
    """Types of anomalies detected in agent sessions."""

    BULK_DATA_ACCESS = auto()           # Mass customer record access
    UNUSUAL_ACCESS_PATTERN = auto()     # Access outside normal pattern
    DATA_EXFILTRATION_ATTEMPT = auto()  # Clipboard/screen/USB/print
    SESSION_INTEGRITY_VIOLATION = auto()  # IP change, replay, etc.
    UNAUTHORIZED_APPLICATION = auto()   # Banned application usage
    IDLE_SESSION_ABUSE = auto()         # Activity during idle period
    CREDENTIAL_SHARING = auto()         # Multiple locations
    OFF_HOURS_ACCESS = auto()           # Access outside work hours
    VELOCITY_ANOMALY = auto()           # Too many actions too fast
    GEOGRAPHIC_ANOMALY = auto()         # Impossible travel


class AlertSeverity(Enum):
    """Severity levels for session alerts."""

    INFO = (1, "Info")
    LOW = (2, "Low")
    MEDIUM = (3, "Medium")
    HIGH = (4, "High")
    CRITICAL = (5, "Critical")

    def __init__(self, level: int, display_name: str):
        self.level = level
        self.display_name = display_name


class AlertAction(Enum):
    """Actions taken in response to alerts."""

    LOG = auto()                # Log the event
    NOTIFY_SUPERVISOR = auto()  # Notify the agent's supervisor
    NOTIFY_SECURITY = auto()    # Notify the security team
    LOCK_SESSION = auto()       # Lock the agent's session
    TERMINATE_SESSION = auto()  # Terminate the session immediately
    REQUIRE_REAUTH = auto()     # Require re-authentication
    QUARANTINE = auto()         # Isolate the agent's session
    BLOCK_ACTION = auto()       # Block the specific action


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SessionEvent:
    """
    An event captured during an agent session.

    Events are the raw data that the anomaly detector analyzes.
    They capture what happened, when, and in what context.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    agent_id: str = ""
    tenant_id: str = ""
    event_type: EventType = EventType.LOGIN
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    source_ip: str = ""
    user_agent: str = ""
    application: str = ""
    data_classification: str = ""       # Data sensitivity involved
    success: bool = True
    blocked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for storage."""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "source_ip": self.source_ip,
            "application": self.application,
            "data_classification": self.data_classification,
            "success": self.success,
            "blocked": self.blocked,
        }


@dataclass
class SessionAlert:
    """
    An alert generated by the anomaly detector.

    Alerts represent potential security issues that require
    attention from supervisors or the security team.
    """

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    agent_id: str = ""
    tenant_id: str = ""
    anomaly_type: AnomalyType = AnomalyType.UNUSUAL_ACCESS_PATTERN
    severity: AlertSeverity = AlertSeverity.MEDIUM
    action: AlertAction = AlertAction.NOTIFY_SECURITY
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    related_events: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert for storage."""
        return {
            "alert_id": self.alert_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "anomaly_type": self.anomaly_type.name,
            "severity": self.severity.display_name,
            "action": self.action.name,
            "description": self.description,
            "evidence": self.evidence,
            "related_events": self.related_events,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class DataAccessPattern:
    """
    Tracks data access patterns for an agent.

    Used as a baseline for detecting unusual data access behavior.
    """

    agent_id: str = ""
    tenant_id: str = ""
    window_hours: int = 24

    # Access counts
    total_lookups: int = 0
    unique_customers_accessed: int = 0
    records_viewed: int = 0
    records_edited: int = 0
    exports_requested: int = 0
    searches_performed: int = 0

    # Timing
    first_access: Optional[datetime] = None
    last_access: Optional[datetime] = None
    peak_hour: int = 0
    access_hours: Set[int] = field(default_factory=set)

    # Baseline (rolling average)
    avg_lookups_per_shift: float = 0.0
    avg_unique_customers_per_shift: float = 0.0
    std_lookups_per_shift: float = 0.0

    # Accessed customer IDs (for tracking unique access)
    _accessed_customers: Set[str] = field(default_factory=set)

    def record_access(
        self, customer_id: str, access_type: str = "view"
    ) -> None:
        """Record a customer data access."""
        now = datetime.utcnow()
        self.total_lookups += 1
        self._accessed_customers.add(customer_id)
        self.unique_customers_accessed = len(self._accessed_customers)

        if access_type == "view":
            self.records_viewed += 1
        elif access_type == "edit":
            self.records_edited += 1
        elif access_type == "export":
            self.exports_requested += 1

        if self.first_access is None:
            self.first_access = now
        self.last_access = now
        self.access_hours.add(now.hour)


@dataclass
class SessionPolicy:
    """
    Security policy for agent sessions.

    Defines thresholds and rules for session monitoring
    and anomaly detection.
    """

    policy_id: str = ""
    policy_name: str = ""
    tenant_id: str = ""

    # Idle/timeout settings
    idle_timeout_minutes: int = 5
    max_session_duration_hours: int = 10
    lock_on_idle: bool = True

    # Data access thresholds
    max_customer_lookups_per_hour: int = 60
    max_unique_customers_per_hour: int = 40
    max_exports_per_shift: int = 5
    max_bulk_operations: int = 3

    # Desktop security
    block_screen_capture: bool = True
    block_clipboard_sensitive: bool = True
    block_usb_storage: bool = True
    block_printing: bool = True
    block_personal_email: bool = True
    block_cloud_storage: bool = True
    watermark_screen: bool = True

    # Allowed applications
    allowed_applications: Set[str] = field(default_factory=lambda: {
        "crm_client", "phone_client", "email_client",
        "knowledge_base", "ticket_system", "chat_client",
    })
    blocked_applications: Set[str] = field(default_factory=lambda: {
        "screen_recorder", "remote_desktop", "file_sharing",
        "personal_messenger", "torrent_client", "vpn_client",
    })

    # Session integrity
    allow_concurrent_sessions: bool = False
    max_concurrent_sessions: int = 1
    enforce_source_ip_consistency: bool = True
    enforce_user_agent_consistency: bool = True

    # Work hours
    enforce_work_hours: bool = True
    work_hours_start: int = 6
    work_hours_end: int = 22
    allowed_days: Set[int] = field(
        default_factory=lambda: {0, 1, 2, 3, 4, 5, 6}
    )

    # Alert escalation thresholds
    auto_lock_on_critical: bool = True
    auto_terminate_on_compromise: bool = True


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """
    Detects anomalies in agent session behavior.

    Uses rule-based detection combined with statistical baselines
    to identify unusual patterns that may indicate security threats.
    The detector maintains per-agent behavioral profiles and flags
    deviations from normal patterns.
    """

    def __init__(
        self,
        policy: SessionPolicy,
        *,
        sensitivity: float = 1.0,       # 0.5 = lenient, 2.0 = strict
        learning_period_days: int = 14,
    ):
        self.policy = policy
        self.sensitivity = sensitivity
        self.learning_period_days = learning_period_days

        # Agent baselines (agent_id -> DataAccessPattern)
        self._baselines: Dict[str, DataAccessPattern] = {}

        # Current shift data (agent_id -> DataAccessPattern)
        self._current_patterns: Dict[str, DataAccessPattern] = {}

        # Recent events window (agent_id -> list of events)
        self._event_windows: Dict[str, List[SessionEvent]] = {}

        # Known session IPs (session_id -> source_ip)
        self._session_ips: Dict[str, str] = {}

        # Known session user agents (session_id -> user_agent)
        self._session_user_agents: Dict[str, str] = {}

    def analyze_event(self, event: SessionEvent) -> List[SessionAlert]:
        """
        Analyze a session event for anomalies.

        Args:
            event: The session event to analyze.

        Returns:
            List of alerts generated by the analysis.
        """
        alerts: List[SessionAlert] = []

        # Track event
        self._track_event(event)

        # Run detectors
        alerts.extend(self._detect_bulk_access(event))
        alerts.extend(self._detect_data_exfiltration(event))
        alerts.extend(self._detect_session_integrity(event))
        alerts.extend(self._detect_unauthorized_app(event))
        alerts.extend(self._detect_off_hours(event))
        alerts.extend(self._detect_velocity_anomaly(event))

        return alerts

    def _detect_bulk_access(self, event: SessionEvent) -> List[SessionAlert]:
        """Detect bulk data access patterns."""
        alerts: List[SessionAlert] = []

        if event.event_type not in (
            EventType.CUSTOMER_LOOKUP,
            EventType.CUSTOMER_RECORD_VIEW,
            EventType.BULK_DATA_EXPORT,
            EventType.SEARCH_QUERY,
        ):
            return alerts

        # Update access pattern
        pattern = self._get_current_pattern(event.agent_id, event.tenant_id)
        customer_id = event.details.get("customer_id", "")
        access_type = "view"
        if event.event_type == EventType.CUSTOMER_RECORD_EDIT:
            access_type = "edit"
        elif event.event_type == EventType.BULK_DATA_EXPORT:
            access_type = "export"
        pattern.record_access(customer_id, access_type)

        # Check lookup rate
        recent_events = self._get_recent_events(event.agent_id, minutes=60)
        lookup_events = [
            e for e in recent_events
            if e.event_type in (
                EventType.CUSTOMER_LOOKUP,
                EventType.CUSTOMER_RECORD_VIEW,
            )
        ]

        if len(lookup_events) > self.policy.max_customer_lookups_per_hour * self.sensitivity:
            alerts.append(
                SessionAlert(
                    session_id=event.session_id,
                    agent_id=event.agent_id,
                    tenant_id=event.tenant_id,
                    anomaly_type=AnomalyType.BULK_DATA_ACCESS,
                    severity=AlertSeverity.HIGH,
                    action=AlertAction.NOTIFY_SECURITY,
                    description=(
                        f"Excessive customer lookups: {len(lookup_events)} "
                        f"in the last hour (threshold: "
                        f"{self.policy.max_customer_lookups_per_hour})"
                    ),
                    evidence={
                        "lookup_count": len(lookup_events),
                        "threshold": self.policy.max_customer_lookups_per_hour,
                        "unique_customers": pattern.unique_customers_accessed,
                        "window_minutes": 60,
                    },
                    related_events=[e.event_id for e in lookup_events[-10:]],
                )
            )

        # Check unique customer access
        if pattern.unique_customers_accessed > self.policy.max_unique_customers_per_hour * self.sensitivity:
            alerts.append(
                SessionAlert(
                    session_id=event.session_id,
                    agent_id=event.agent_id,
                    tenant_id=event.tenant_id,
                    anomaly_type=AnomalyType.BULK_DATA_ACCESS,
                    severity=AlertSeverity.HIGH,
                    action=AlertAction.NOTIFY_SECURITY,
                    description=(
                        f"Excessive unique customer access: "
                        f"{pattern.unique_customers_accessed} customers "
                        f"(threshold: {self.policy.max_unique_customers_per_hour})"
                    ),
                    evidence={
                        "unique_customers": pattern.unique_customers_accessed,
                        "threshold": self.policy.max_unique_customers_per_hour,
                    },
                    related_events=[event.event_id],
                )
            )

        # Check export count
        if event.event_type == EventType.BULK_DATA_EXPORT:
            if pattern.exports_requested > self.policy.max_exports_per_shift:
                alerts.append(
                    SessionAlert(
                        session_id=event.session_id,
                        agent_id=event.agent_id,
                        tenant_id=event.tenant_id,
                        anomaly_type=AnomalyType.DATA_EXFILTRATION_ATTEMPT,
                        severity=AlertSeverity.CRITICAL,
                        action=AlertAction.LOCK_SESSION,
                        description=(
                            f"Export limit exceeded: {pattern.exports_requested} "
                            f"exports (threshold: {self.policy.max_exports_per_shift})"
                        ),
                        evidence={
                            "export_count": pattern.exports_requested,
                            "threshold": self.policy.max_exports_per_shift,
                        },
                        related_events=[event.event_id],
                    )
                )

        return alerts

    def _detect_data_exfiltration(self, event: SessionEvent) -> List[SessionAlert]:
        """Detect potential data exfiltration attempts."""
        alerts: List[SessionAlert] = []

        exfiltration_events = {
            EventType.SCREEN_CAPTURE_ATTEMPT: (
                AnomalyType.DATA_EXFILTRATION_ATTEMPT,
                AlertSeverity.CRITICAL,
                AlertAction.LOCK_SESSION,
                "Screen capture attempt detected",
            ),
            EventType.CLIPBOARD_COPY: (
                AnomalyType.DATA_EXFILTRATION_ATTEMPT,
                AlertSeverity.MEDIUM,
                AlertAction.BLOCK_ACTION,
                "Clipboard copy of sensitive data",
            ),
            EventType.USB_DEVICE_INSERT: (
                AnomalyType.DATA_EXFILTRATION_ATTEMPT,
                AlertSeverity.HIGH,
                AlertAction.NOTIFY_SECURITY,
                "USB storage device inserted",
            ),
            EventType.PRINT_ATTEMPT: (
                AnomalyType.DATA_EXFILTRATION_ATTEMPT,
                AlertSeverity.MEDIUM,
                AlertAction.BLOCK_ACTION,
                "Print attempt with sensitive data",
            ),
            EventType.FILE_DOWNLOAD: (
                AnomalyType.DATA_EXFILTRATION_ATTEMPT,
                AlertSeverity.MEDIUM,
                AlertAction.NOTIFY_SUPERVISOR,
                "File download detected",
            ),
            EventType.FILE_UPLOAD: (
                AnomalyType.DATA_EXFILTRATION_ATTEMPT,
                AlertSeverity.HIGH,
                AlertAction.BLOCK_ACTION,
                "File upload detected",
            ),
        }

        if event.event_type in exfiltration_events:
            anomaly_type, severity, action, desc = exfiltration_events[event.event_type]

            # Check if the specific action is blocked by policy
            blocked = False
            if event.event_type == EventType.SCREEN_CAPTURE_ATTEMPT and self.policy.block_screen_capture:
                blocked = True
                severity = AlertSeverity.CRITICAL
            elif event.event_type == EventType.CLIPBOARD_COPY and self.policy.block_clipboard_sensitive:
                if event.data_classification in ("RESTRICTED", "CONFIDENTIAL"):
                    blocked = True
            elif event.event_type == EventType.USB_DEVICE_INSERT and self.policy.block_usb_storage:
                blocked = True
            elif event.event_type == EventType.PRINT_ATTEMPT and self.policy.block_printing:
                blocked = True

            alerts.append(
                SessionAlert(
                    session_id=event.session_id,
                    agent_id=event.agent_id,
                    tenant_id=event.tenant_id,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    action=action,
                    description=f"{desc} (blocked={blocked})",
                    evidence={
                        "event_type": event.event_type.name,
                        "blocked": blocked,
                        "data_classification": event.data_classification,
                        "application": event.application,
                        "details": event.details,
                    },
                    related_events=[event.event_id],
                )
            )

        return alerts

    def _detect_session_integrity(self, event: SessionEvent) -> List[SessionAlert]:
        """Detect session integrity violations (IP change, replay, etc.)."""
        alerts: List[SessionAlert] = []

        # IP address change detection
        if (
            self.policy.enforce_source_ip_consistency
            and event.source_ip
            and event.session_id
        ):
            known_ip = self._session_ips.get(event.session_id)
            if known_ip is None:
                self._session_ips[event.session_id] = event.source_ip
            elif known_ip != event.source_ip:
                alerts.append(
                    SessionAlert(
                        session_id=event.session_id,
                        agent_id=event.agent_id,
                        tenant_id=event.tenant_id,
                        anomaly_type=AnomalyType.SESSION_INTEGRITY_VIOLATION,
                        severity=AlertSeverity.CRITICAL,
                        action=AlertAction.REQUIRE_REAUTH,
                        description=(
                            f"Session IP address changed: "
                            f"{self._mask_ip(known_ip)} -> "
                            f"{self._mask_ip(event.source_ip)}"
                        ),
                        evidence={
                            "original_ip": self._mask_ip(known_ip),
                            "new_ip": self._mask_ip(event.source_ip),
                        },
                        related_events=[event.event_id],
                    )
                )
                # Update tracked IP
                self._session_ips[event.session_id] = event.source_ip

        # User agent change detection
        if (
            self.policy.enforce_user_agent_consistency
            and event.user_agent
            and event.session_id
        ):
            known_ua = self._session_user_agents.get(event.session_id)
            if known_ua is None:
                self._session_user_agents[event.session_id] = event.user_agent
            elif known_ua != event.user_agent:
                alerts.append(
                    SessionAlert(
                        session_id=event.session_id,
                        agent_id=event.agent_id,
                        tenant_id=event.tenant_id,
                        anomaly_type=AnomalyType.SESSION_INTEGRITY_VIOLATION,
                        severity=AlertSeverity.HIGH,
                        action=AlertAction.REQUIRE_REAUTH,
                        description="Session user agent changed (possible session hijack)",
                        evidence={
                            "original_ua_hash": hashlib.sha256(
                                known_ua.encode()
                            ).hexdigest()[:16],
                            "new_ua_hash": hashlib.sha256(
                                event.user_agent.encode()
                            ).hexdigest()[:16],
                        },
                        related_events=[event.event_id],
                    )
                )

        # Concurrent session detection
        if event.event_type == EventType.CONCURRENT_SESSION:
            if not self.policy.allow_concurrent_sessions:
                alerts.append(
                    SessionAlert(
                        session_id=event.session_id,
                        agent_id=event.agent_id,
                        tenant_id=event.tenant_id,
                        anomaly_type=AnomalyType.CREDENTIAL_SHARING,
                        severity=AlertSeverity.HIGH,
                        action=AlertAction.TERMINATE_SESSION,
                        description=(
                            "Concurrent session detected - "
                            "possible credential sharing"
                        ),
                        evidence=event.details,
                        related_events=[event.event_id],
                    )
                )

        return alerts

    def _detect_unauthorized_app(self, event: SessionEvent) -> List[SessionAlert]:
        """Detect unauthorized application usage."""
        alerts: List[SessionAlert] = []

        if event.event_type != EventType.UNAUTHORIZED_APP_LAUNCH:
            return alerts

        app_name = event.application or event.details.get("application", "unknown")

        if (
            app_name in self.policy.blocked_applications
            or (
                self.policy.allowed_applications
                and app_name not in self.policy.allowed_applications
            )
        ):
            is_blocked = app_name in self.policy.blocked_applications
            alerts.append(
                SessionAlert(
                    session_id=event.session_id,
                    agent_id=event.agent_id,
                    tenant_id=event.tenant_id,
                    anomaly_type=AnomalyType.UNAUTHORIZED_APPLICATION,
                    severity=(
                        AlertSeverity.HIGH if is_blocked else AlertSeverity.MEDIUM
                    ),
                    action=AlertAction.BLOCK_ACTION,
                    description=(
                        f"Unauthorized application launch: {app_name} "
                        f"({'blocked' if is_blocked else 'not in allowlist'})"
                    ),
                    evidence={
                        "application": app_name,
                        "is_blocked": is_blocked,
                    },
                    related_events=[event.event_id],
                )
            )

        return alerts

    def _detect_off_hours(self, event: SessionEvent) -> List[SessionAlert]:
        """Detect off-hours access."""
        alerts: List[SessionAlert] = []

        if not self.policy.enforce_work_hours:
            return alerts

        hour = event.timestamp.hour
        day = event.timestamp.weekday()

        is_off_hours = (
            hour < self.policy.work_hours_start
            or hour >= self.policy.work_hours_end
        )
        is_off_day = day not in self.policy.allowed_days

        if is_off_hours or is_off_day:
            alerts.append(
                SessionAlert(
                    session_id=event.session_id,
                    agent_id=event.agent_id,
                    tenant_id=event.tenant_id,
                    anomaly_type=AnomalyType.OFF_HOURS_ACCESS,
                    severity=AlertSeverity.MEDIUM,
                    action=AlertAction.NOTIFY_SUPERVISOR,
                    description=(
                        f"Off-hours access: {event.timestamp.strftime('%H:%M')} "
                        f"on {'weekend' if is_off_day and day >= 5 else event.timestamp.strftime('%A')}"
                    ),
                    evidence={
                        "hour": hour,
                        "day_of_week": day,
                        "work_hours": (
                            f"{self.policy.work_hours_start:02d}:00-"
                            f"{self.policy.work_hours_end:02d}:00"
                        ),
                    },
                    related_events=[event.event_id],
                )
            )

        return alerts

    def _detect_velocity_anomaly(self, event: SessionEvent) -> List[SessionAlert]:
        """Detect abnormally rapid actions."""
        alerts: List[SessionAlert] = []

        recent = self._get_recent_events(event.agent_id, minutes=1)
        if len(recent) > 30 * self.sensitivity:  # > 30 events per minute
            alerts.append(
                SessionAlert(
                    session_id=event.session_id,
                    agent_id=event.agent_id,
                    tenant_id=event.tenant_id,
                    anomaly_type=AnomalyType.VELOCITY_ANOMALY,
                    severity=AlertSeverity.MEDIUM,
                    action=AlertAction.NOTIFY_SECURITY,
                    description=(
                        f"Abnormal action velocity: {len(recent)} "
                        f"events in the last minute"
                    ),
                    evidence={
                        "event_count": len(recent),
                        "window_minutes": 1,
                        "threshold": int(30 * self.sensitivity),
                    },
                    related_events=[event.event_id],
                )
            )

        return alerts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _track_event(self, event: SessionEvent) -> None:
        """Track an event for pattern analysis."""
        agent_id = event.agent_id
        if agent_id not in self._event_windows:
            self._event_windows[agent_id] = []
        self._event_windows[agent_id].append(event)

        # Limit window size
        if len(self._event_windows[agent_id]) > 5000:
            self._event_windows[agent_id] = (
                self._event_windows[agent_id][-2500:]
            )

    def _get_recent_events(
        self, agent_id: str, minutes: int = 60
    ) -> List[SessionEvent]:
        """Get recent events for an agent within the given window."""
        if agent_id not in self._event_windows:
            return []
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            e for e in self._event_windows[agent_id]
            if e.timestamp >= cutoff
        ]

    def _get_current_pattern(
        self, agent_id: str, tenant_id: str
    ) -> DataAccessPattern:
        """Get or create the current shift data access pattern."""
        if agent_id not in self._current_patterns:
            self._current_patterns[agent_id] = DataAccessPattern(
                agent_id=agent_id,
                tenant_id=tenant_id,
            )
        return self._current_patterns[agent_id]

    @staticmethod
    def _mask_ip(ip: str) -> str:
        """Mask an IP address for logging."""
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.***. ***"
        return "***"


# ---------------------------------------------------------------------------
# Alert Manager
# ---------------------------------------------------------------------------


class AlertManager:
    """
    Manages session security alerts.

    Provides alert routing, deduplication, escalation,
    and acknowledgement tracking.
    """

    def __init__(
        self,
        *,
        alert_callback: Optional[Callable[[SessionAlert], None]] = None,
        dedup_window_seconds: int = 300,
        max_alerts_per_agent_per_hour: int = 50,
        escalation_threshold: int = 5,
    ):
        self.alert_callback = alert_callback
        self.dedup_window_seconds = dedup_window_seconds
        self.max_alerts_per_agent_per_hour = max_alerts_per_agent_per_hour
        self.escalation_threshold = escalation_threshold

        self._alerts: List[SessionAlert] = []
        self._agent_alert_counts: Dict[str, List[datetime]] = {}
        self._recent_alert_hashes: Dict[str, datetime] = {}
        self._stats = {
            "total_alerts": 0,
            "alerts_suppressed": 0,
            "alerts_escalated": 0,
        }

    def process_alert(self, alert: SessionAlert) -> bool:
        """
        Process a new alert, applying deduplication and rate limiting.

        Args:
            alert: The alert to process.

        Returns:
            True if the alert was delivered, False if suppressed.
        """
        # Deduplication
        alert_hash = self._compute_alert_hash(alert)
        if alert_hash in self._recent_alert_hashes:
            last_seen = self._recent_alert_hashes[alert_hash]
            if (datetime.utcnow() - last_seen).total_seconds() < self.dedup_window_seconds:
                self._stats["alerts_suppressed"] += 1
                return False
        self._recent_alert_hashes[alert_hash] = datetime.utcnow()

        # Rate limiting per agent
        agent_id = alert.agent_id
        if agent_id not in self._agent_alert_counts:
            self._agent_alert_counts[agent_id] = []

        now = datetime.utcnow()
        recent = [
            ts for ts in self._agent_alert_counts[agent_id]
            if (now - ts).total_seconds() < 3600
        ]
        self._agent_alert_counts[agent_id] = recent

        if len(recent) >= self.max_alerts_per_agent_per_hour:
            self._stats["alerts_suppressed"] += 1
            return False

        self._agent_alert_counts[agent_id].append(now)

        # Store alert
        self._alerts.append(alert)
        self._stats["total_alerts"] += 1

        # Check for escalation
        if len(recent) >= self.escalation_threshold:
            self._stats["alerts_escalated"] += 1
            alert.severity = AlertSeverity.CRITICAL
            alert.action = AlertAction.LOCK_SESSION

        # Deliver alert
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as exc:
                logger.error("Alert callback failed: %s", exc)

        logger.warning(
            "SESSION ALERT: agent=%s type=%s severity=%s desc=%s",
            alert.agent_id,
            alert.anomaly_type.name,
            alert.severity.display_name,
            alert.description,
        )

        return True

    def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str
    ) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                return True
        return False

    def get_open_alerts(
        self,
        *,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        min_severity: Optional[AlertSeverity] = None,
    ) -> List[SessionAlert]:
        """Get open (unresolved) alerts with optional filtering."""
        results = [a for a in self._alerts if not a.resolved]

        if agent_id:
            results = [a for a in results if a.agent_id == agent_id]
        if tenant_id:
            results = [a for a in results if a.tenant_id == tenant_id]
        if min_severity:
            results = [
                a for a in results if a.severity.level >= min_severity.level
            ]

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        return dict(self._stats)

    @staticmethod
    def _compute_alert_hash(alert: SessionAlert) -> str:
        """Compute a hash for alert deduplication."""
        key = (
            f"{alert.agent_id}:{alert.anomaly_type.name}:"
            f"{alert.session_id}"
        )
        return hashlib.sha256(key.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Agent Session Monitor (orchestrator)
# ---------------------------------------------------------------------------


class AgentSessionMonitor:
    """
    Real-time monitoring of agent sessions for security anomalies.

    Detects:
    - Unusual data access patterns (bulk customer lookups)
    - Screen capture/screenshot attempts
    - Clipboard copy of sensitive data
    - USB device insertion
    - Unauthorized application access
    - Session sharing or replay attacks
    - Idle session exploitation

    This is the main orchestrator that coordinates the anomaly detector,
    data access tracking, alert management, and session policy enforcement.

    Usage::

        policy = SessionPolicy(
            policy_id="SP-001",
            max_customer_lookups_per_hour=60,
            block_screen_capture=True,
        )
        monitor = AgentSessionMonitor(policy=policy)

        # Process an event
        alerts = monitor.process_event(event)
        for alert in alerts:
            print(f"ALERT: {alert.description}")

        # Get session status
        status = monitor.get_session_status("session-123")
    """

    def __init__(
        self,
        policy: SessionPolicy,
        *,
        alert_callback: Optional[Callable[[SessionAlert], None]] = None,
        sensitivity: float = 1.0,
    ):
        self.policy = policy
        self.anomaly_detector = AnomalyDetector(
            policy=policy,
            sensitivity=sensitivity,
        )
        self.alert_manager = AlertManager(
            alert_callback=alert_callback,
        )

        # Active sessions (session_id -> session state info)
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._stats = {
            "total_events_processed": 0,
            "total_alerts_generated": 0,
            "active_sessions": 0,
        }

    def process_event(self, event: SessionEvent) -> List[SessionAlert]:
        """
        Process a session event and return any generated alerts.

        This is the primary entry point for event processing. It:
        1. Updates session state
        2. Runs anomaly detection
        3. Manages alerts
        4. Enforces session policy

        Args:
            event: The session event to process.

        Returns:
            List of alerts generated by this event.
        """
        self._stats["total_events_processed"] += 1

        # Update session tracking
        self._update_session(event)

        # Run anomaly detection
        alerts = self.anomaly_detector.analyze_event(event)

        # Process alerts through alert manager
        delivered_alerts: List[SessionAlert] = []
        for alert in alerts:
            if self.alert_manager.process_alert(alert):
                delivered_alerts.append(alert)

                # Auto-actions based on policy
                self._enforce_alert_action(alert)

        self._stats["total_alerts_generated"] += len(delivered_alerts)

        return delivered_alerts

    def register_session(
        self,
        session_id: str,
        agent_id: str,
        tenant_id: str,
        source_ip: str = "",
    ) -> None:
        """Register a new agent session."""
        self._sessions[session_id] = {
            "session_id": session_id,
            "agent_id": agent_id,
            "tenant_id": tenant_id,
            "state": SessionState.ACTIVE,
            "started_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "source_ip": source_ip,
            "event_count": 0,
        }
        self._stats["active_sessions"] += 1
        logger.info(
            "Session registered: session=%s agent=%s tenant=%s",
            session_id,
            agent_id,
            tenant_id,
        )

    def terminate_session(
        self, session_id: str, reason: str = "manual"
    ) -> None:
        """Terminate an agent session."""
        if session_id in self._sessions:
            self._sessions[session_id]["state"] = SessionState.TERMINATED
            self._sessions[session_id]["terminated_at"] = datetime.utcnow()
            self._sessions[session_id]["termination_reason"] = reason
            self._stats["active_sessions"] = max(
                0, self._stats["active_sessions"] - 1
            )
            logger.info(
                "Session terminated: session=%s reason=%s",
                session_id,
                reason,
            )

    def lock_session(
        self, session_id: str, reason: str = "policy"
    ) -> None:
        """Lock an agent session."""
        if session_id in self._sessions:
            self._sessions[session_id]["state"] = SessionState.LOCKED
            self._sessions[session_id]["locked_at"] = datetime.utcnow()
            self._sessions[session_id]["lock_reason"] = reason
            logger.info(
                "Session locked: session=%s reason=%s",
                session_id,
                reason,
            )

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a session."""
        return self._sessions.get(session_id)

    def get_active_sessions(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all active sessions, optionally filtered by tenant."""
        sessions = [
            s for s in self._sessions.values()
            if s["state"] in (SessionState.ACTIVE, SessionState.IDLE)
        ]
        if tenant_id:
            sessions = [s for s in sessions if s["tenant_id"] == tenant_id]
        return sessions

    def get_statistics(self) -> Dict[str, Any]:
        """Get session monitor statistics."""
        stats = dict(self._stats)
        stats["alert_manager"] = self.alert_manager.get_statistics()
        return stats

    def _update_session(self, event: SessionEvent) -> None:
        """Update session tracking with a new event."""
        if event.session_id in self._sessions:
            session = self._sessions[event.session_id]
            session["last_activity"] = event.timestamp
            session["event_count"] = session.get("event_count", 0) + 1

            # Check idle timeout
            if session["state"] == SessionState.IDLE:
                session["state"] = SessionState.ACTIVE

    def _enforce_alert_action(self, alert: SessionAlert) -> None:
        """Enforce the action specified in an alert."""
        if alert.action == AlertAction.LOCK_SESSION and self.policy.auto_lock_on_critical:
            if alert.severity.level >= AlertSeverity.CRITICAL.level:
                self.lock_session(
                    alert.session_id,
                    reason=f"Auto-lock: {alert.description}",
                )

        elif alert.action == AlertAction.TERMINATE_SESSION and self.policy.auto_terminate_on_compromise:
            self.terminate_session(
                alert.session_id,
                reason=f"Auto-terminate: {alert.description}",
            )
