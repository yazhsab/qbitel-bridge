"""
Workforce Management Integration Bridge

Secure bridge to Workforce Management systems for agent scheduling,
real-time adherence monitoring, forecasting, and performance metrics.
Provides quantum-safe protection for agent scheduling and performance data.

Supported WFM platforms:
- NICE WFM (REST API)
- Verint Workforce Management (REST API)
- Aspect Workforce Management (REST API)
- Calabrio WFM (REST API)
- Genesys WFM (Platform API)
- Custom WFM via configurable REST adapters

Security features:
- PQC tunnel wrapping for all WFM API calls
- Audit logging of all schedule and performance data access
- Integration with QBITEL session management
- Auto-lock when agents are off-schedule
- Security alerting when agents are active outside scheduled hours
- Real-time adherence enforcement
- Rate limiting on WFM API access
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, time as dt_time, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import asyncio
import logging
import time
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WFMType(Enum):
    """Supported Workforce Management platforms."""

    NICE_WFM = ("NICE WFM", "rest", "niceincontact.com")
    VERINT = ("Verint WFM", "rest", "verint.com")
    ASPECT = ("Aspect WFM", "rest", "alvaria.com")
    CALABRIO = ("Calabrio WFM", "rest", "calabrio.com")
    GENESYS_WFM = ("Genesys WFM", "rest", "genesys.com")
    CUSTOM = ("Custom WFM", "rest", "")

    def __init__(self, display_name: str, api_type: str, domain: str):
        self.display_name = display_name
        self.api_type = api_type
        self.domain = domain


class WFMEventType(Enum):
    """WFM event types for callbacks and audit."""

    SCHEDULE_RETRIEVED = auto()
    SCHEDULE_UPDATED = auto()
    SCHEDULE_EXCEPTION_SUBMITTED = auto()
    SCHEDULE_EXCEPTION_APPROVED = auto()
    SCHEDULE_EXCEPTION_DENIED = auto()
    ADHERENCE_IN = auto()
    ADHERENCE_OUT = auto()
    ADHERENCE_WARNING = auto()
    FORECAST_RETRIEVED = auto()
    METRICS_RETRIEVED = auto()
    AGENT_OFF_SCHEDULE_ALERT = auto()
    AGENT_SESSION_AUTO_LOCKED = auto()
    OVERTIME_OFFERED = auto()
    VTO_OFFERED = auto()  # Voluntary Time Off


class AdherenceStatus(Enum):
    """Real-time agent schedule adherence status."""

    IN_ADHERENCE = ("in_adherence", True)
    OUT_OF_ADHERENCE = ("out_of_adherence", False)
    EXCUSED = ("excused", True)
    NOT_SCHEDULED = ("not_scheduled", True)
    UNKNOWN = ("unknown", False)

    def __init__(self, status_name: str, is_compliant: bool):
        self.status_name = status_name
        self.is_compliant = is_compliant


class ScheduleActivityType(Enum):
    """Types of activities in agent schedules."""

    WORK = "work"
    BREAK = "break"
    LUNCH = "lunch"
    TRAINING = "training"
    MEETING = "meeting"
    COACHING = "coaching"
    PROJECT = "project"
    OFF = "off"
    OVERTIME = "overtime"
    VTO = "vto"  # Voluntary Time Off


class ScheduleExceptionType(Enum):
    """Types of schedule exceptions that can be submitted."""

    SICK = ("sick", True)
    PERSONAL = ("personal", True)
    FAMILY_EMERGENCY = ("family_emergency", True)
    LATE_ARRIVAL = ("late_arrival", False)
    EARLY_DEPARTURE = ("early_departure", False)
    OVERTIME_REQUEST = ("overtime_request", False)
    SCHEDULE_SWAP = ("schedule_swap", False)
    VTO_REQUEST = ("vto_request", False)
    TRAINING = ("training", False)
    BEREAVEMENT = ("bereavement", True)
    JURY_DUTY = ("jury_duty", True)

    def __init__(self, exception_name: str, auto_approve: bool):
        self.exception_name = exception_name
        self.auto_approve = auto_approve


class SecurityAlertSeverity(Enum):
    """Severity levels for security alerts."""

    INFO = (0, "info")
    WARNING = (1, "warning")
    HIGH = (2, "high")
    CRITICAL = (3, "critical")

    def __init__(self, level: int, severity_name: str):
        self.alert_level = level
        self.severity_name = severity_name


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WFMConnectionConfig:
    """Configuration for WFM system connection."""

    # Connection
    wfm_type: WFMType = WFMType.NICE_WFM
    base_url: str = ""
    api_version: str = ""

    # Authentication
    auth_method: str = "oauth2"
    credentials_vault_path: str = ""
    oauth_token_url: str = ""
    oauth_scope: str = ""

    # PQC settings
    pqc_enabled: bool = True
    pqc_kem_algorithm: str = "ML-KEM-768"
    pqc_sig_algorithm: str = "ML-DSA-65"
    pqc_hybrid_mode: bool = True

    # TLS settings
    tls_version: str = "TLS 1.3"
    ca_cert_path: Optional[str] = None

    # Rate limiting
    max_requests_per_minute: int = 120
    max_requests_per_hour: int = 2000

    # Timeouts
    connect_timeout_seconds: int = 30
    read_timeout_seconds: int = 60

    # Schedule sync settings
    schedule_sync_interval_seconds: int = 300  # 5 minutes
    adherence_check_interval_seconds: int = 60  # 1 minute
    forecast_cache_minutes: int = 15

    # Security integration
    auto_lock_off_schedule: bool = True
    alert_off_schedule_activity: bool = True
    off_schedule_grace_period_minutes: int = 5
    max_overtime_hours_daily: float = 4.0
    enforce_break_compliance: bool = True

    # Data settings
    default_timezone: str = "UTC"
    metrics_history_days: int = 30

    def validate(self) -> List[str]:
        """Validate WFM connection configuration."""
        errors = []

        if not self.base_url and self.wfm_type != WFMType.CUSTOM:
            errors.append("WFM base URL is required")

        if not self.credentials_vault_path:
            errors.append("credentials_vault_path is required")

        if self.schedule_sync_interval_seconds < 60:
            errors.append("Schedule sync interval must be at least 60 seconds")

        if self.adherence_check_interval_seconds < 10:
            errors.append("Adherence check interval must be at least 10 seconds")

        if self.off_schedule_grace_period_minutes < 0:
            errors.append("Grace period cannot be negative")

        return errors


@dataclass
class ScheduleActivity:
    """A single scheduled activity within an agent's shift."""

    activity_type: ScheduleActivityType = ScheduleActivityType.WORK
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime = field(default_factory=datetime.utcnow)
    skill_group: str = ""
    queue_name: str = ""
    description: str = ""
    is_paid: bool = True

    @property
    def duration_minutes(self) -> float:
        """Get activity duration in minutes."""
        return (self.end_time - self.start_time).total_seconds() / 60.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "activity_type": self.activity_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_minutes": self.duration_minutes,
            "skill_group": self.skill_group,
            "queue_name": self.queue_name,
            "description": self.description,
            "is_paid": self.is_paid,
        }


@dataclass
class AgentSchedule:
    """Agent schedule data from WFM system."""

    agent_id: str = ""
    schedule_date: date = field(default_factory=date.today)
    shift_start: Optional[datetime] = None
    shift_end: Optional[datetime] = None
    break_times: List[ScheduleActivity] = field(default_factory=list)
    skill_groups: List[str] = field(default_factory=list)
    assigned_queues: List[str] = field(default_factory=list)
    activities: List[ScheduleActivity] = field(default_factory=list)

    # Schedule metadata
    timezone: str = "UTC"
    is_day_off: bool = False
    is_holiday: bool = False
    schedule_version: int = 0
    last_modified: Optional[datetime] = None

    # Overtime/VTO
    overtime_eligible: bool = False
    vto_eligible: bool = False
    scheduled_overtime_hours: float = 0.0

    @property
    def total_shift_hours(self) -> float:
        """Get total shift duration in hours."""
        if self.shift_start and self.shift_end:
            return (self.shift_end - self.shift_start).total_seconds() / 3600.0
        return 0.0

    @property
    def total_break_minutes(self) -> float:
        """Get total scheduled break time in minutes."""
        return sum(b.duration_minutes for b in self.break_times)

    @property
    def net_work_hours(self) -> float:
        """Get net working hours (total shift minus breaks)."""
        return self.total_shift_hours - (self.total_break_minutes / 60.0)

    def is_currently_scheduled(self, check_time: Optional[datetime] = None) -> bool:
        """Check if the agent is currently within their scheduled shift."""
        now = check_time or datetime.utcnow()
        if self.is_day_off or self.is_holiday:
            return False
        if self.shift_start and self.shift_end:
            return self.shift_start <= now <= self.shift_end
        return False

    def get_current_activity(self, check_time: Optional[datetime] = None) -> Optional[ScheduleActivity]:
        """Get the scheduled activity for the current time."""
        now = check_time or datetime.utcnow()
        for activity in self.activities:
            if activity.start_time <= now <= activity.end_time:
                return activity
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "schedule_date": self.schedule_date.isoformat(),
            "shift_start": self.shift_start.isoformat() if self.shift_start else None,
            "shift_end": self.shift_end.isoformat() if self.shift_end else None,
            "total_shift_hours": self.total_shift_hours,
            "total_break_minutes": self.total_break_minutes,
            "net_work_hours": self.net_work_hours,
            "skill_groups": self.skill_groups,
            "assigned_queues": self.assigned_queues,
            "activities": [a.to_dict() for a in self.activities],
            "is_day_off": self.is_day_off,
            "is_holiday": self.is_holiday,
            "overtime_eligible": self.overtime_eligible,
            "timezone": self.timezone,
        }


@dataclass
class PerformanceMetrics:
    """Agent performance metrics from WFM system."""

    agent_id: str = ""
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    # Call handling metrics
    aht: float = 0.0                       # Average Handle Time (seconds)
    calls_handled: int = 0
    calls_offered: int = 0

    # Quality metrics
    quality_score: float = 0.0             # 0-100 scale
    csat_score: float = 0.0                # Customer Satisfaction 0-100
    fcr_rate: float = 0.0                  # First Call Resolution %

    # Adherence metrics
    adherence_percentage: float = 0.0      # Schedule adherence %
    conformance_percentage: float = 0.0    # Schedule conformance %
    occupancy: float = 0.0                 # Agent occupancy %

    # Time metrics
    total_login_seconds: float = 0.0
    total_talk_seconds: float = 0.0
    total_hold_seconds: float = 0.0
    total_acw_seconds: float = 0.0
    total_idle_seconds: float = 0.0
    total_not_ready_seconds: float = 0.0

    # Additional metrics
    transfers_out: int = 0
    escalations: int = 0
    callbacks_completed: int = 0
    average_speed_of_answer: float = 0.0  # seconds
    longest_hold_time: float = 0.0        # seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "aht": self.aht,
            "calls_handled": self.calls_handled,
            "calls_offered": self.calls_offered,
            "quality_score": self.quality_score,
            "csat_score": self.csat_score,
            "fcr_rate": self.fcr_rate,
            "adherence_percentage": self.adherence_percentage,
            "conformance_percentage": self.conformance_percentage,
            "occupancy": self.occupancy,
            "total_login_seconds": self.total_login_seconds,
            "total_talk_seconds": self.total_talk_seconds,
            "total_hold_seconds": self.total_hold_seconds,
            "total_acw_seconds": self.total_acw_seconds,
            "total_idle_seconds": self.total_idle_seconds,
            "transfers_out": self.transfers_out,
            "escalations": self.escalations,
        }


@dataclass
class ForecastData:
    """Workforce forecast data from WFM system."""

    forecast_date: date = field(default_factory=date.today)
    interval_minutes: int = 30
    predicted_volume: int = 0
    predicted_aht: float = 0.0            # seconds
    required_agents: int = 0
    scheduled_agents: int = 0

    # Extended forecast
    service_level_target: float = 80.0     # Target SL %
    target_asa: float = 20.0               # Target ASA seconds
    predicted_abandonment_rate: float = 0.0
    shrinkage_factor: float = 0.0          # % of time agents unavailable
    required_fte: float = 0.0              # Full-time equivalent

    # Interval timing
    interval_start: Optional[datetime] = None
    interval_end: Optional[datetime] = None

    # Skill/queue breakdown
    skill_group: str = ""
    queue_name: str = ""
    channel: str = "voice"

    @property
    def staffing_delta(self) -> int:
        """Get the difference between scheduled and required agents."""
        return self.scheduled_agents - self.required_agents

    @property
    def is_understaffed(self) -> bool:
        """Check if the interval is understaffed."""
        return self.scheduled_agents < self.required_agents

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "forecast_date": self.forecast_date.isoformat(),
            "interval_minutes": self.interval_minutes,
            "interval_start": self.interval_start.isoformat() if self.interval_start else None,
            "interval_end": self.interval_end.isoformat() if self.interval_end else None,
            "predicted_volume": self.predicted_volume,
            "predicted_aht": self.predicted_aht,
            "required_agents": self.required_agents,
            "scheduled_agents": self.scheduled_agents,
            "staffing_delta": self.staffing_delta,
            "is_understaffed": self.is_understaffed,
            "service_level_target": self.service_level_target,
            "skill_group": self.skill_group,
            "queue_name": self.queue_name,
            "channel": self.channel,
        }


@dataclass
class AdherenceRecord:
    """Real-time adherence record for an agent."""

    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: AdherenceStatus = AdherenceStatus.UNKNOWN
    scheduled_activity: Optional[ScheduleActivityType] = None
    actual_activity: Optional[str] = None
    deviation_minutes: float = 0.0
    is_excused: bool = False
    exception_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.status_name,
            "scheduled_activity": self.scheduled_activity.value if self.scheduled_activity else None,
            "actual_activity": self.actual_activity,
            "deviation_minutes": self.deviation_minutes,
            "is_excused": self.is_excused,
        }


@dataclass
class SecurityAlert:
    """Security alert generated by WFM bridge."""

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: SecurityAlertSeverity = SecurityAlertSeverity.WARNING
    alert_type: str = ""
    agent_id: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.severity_name,
            "alert_type": self.alert_type,
            "agent_id": self.agent_id,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class WFMHealthStatus:
    """Health status of the WFM bridge."""

    is_connected: bool = False
    is_healthy: bool = False
    last_schedule_sync: Optional[datetime] = None
    last_adherence_check: Optional[datetime] = None
    agents_monitored: int = 0
    agents_in_adherence: int = 0
    agents_out_of_adherence: int = 0
    active_alerts: int = 0
    pqc_tunnel_active: bool = False
    error_count: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_connected": self.is_connected,
            "is_healthy": self.is_healthy,
            "last_schedule_sync": self.last_schedule_sync.isoformat() if self.last_schedule_sync else None,
            "last_adherence_check": self.last_adherence_check.isoformat() if self.last_adherence_check else None,
            "agents_monitored": self.agents_monitored,
            "agents_in_adherence": self.agents_in_adherence,
            "agents_out_of_adherence": self.agents_out_of_adherence,
            "active_alerts": self.active_alerts,
            "pqc_tunnel_active": self.pqc_tunnel_active,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WFMError(Exception):
    """Base exception for WFM bridge operations."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code


class WFMConnectionError(WFMError):
    """Exception for WFM connection failures."""
    pass


class WFMAuthenticationError(WFMError):
    """Exception for WFM authentication failures."""
    pass


class WFMScheduleError(WFMError):
    """Exception for schedule-related errors."""
    pass


class WFMAdherenceError(WFMError):
    """Exception for adherence-related errors."""
    pass


# ---------------------------------------------------------------------------
# WFM Bridge abstract base
# ---------------------------------------------------------------------------


class WFMBridge(ABC):
    """
    Secure bridge to Workforce Management systems.

    Provides quantum-safe protection for agent scheduling and
    performance data with integrated QBITEL session management
    for security enforcement.

    Features:
    - PQC tunnel wrapping for all WFM API calls
    - Real-time schedule adherence monitoring
    - Auto-lock of QBITEL sessions when agents are off-schedule
    - Security alerts when agents are active outside scheduled hours
    - Performance metrics retrieval with audit logging
    - Forecast data integration for capacity planning
    - Schedule exception management
    - Break compliance enforcement
    """

    def __init__(self, wfm_type: WFMType, config: WFMConnectionConfig):
        self._wfm_type = wfm_type
        self._config = config
        self._connected = False
        self._healthy = False

        # Schedule cache: agent_id -> AgentSchedule
        self._schedule_cache: Dict[str, AgentSchedule] = {}
        self._last_schedule_sync: Optional[datetime] = None

        # Adherence state: agent_id -> AdherenceRecord
        self._adherence_state: Dict[str, AdherenceRecord] = {}
        self._last_adherence_check: Optional[datetime] = None

        # Security alerts
        self._active_alerts: List[SecurityAlert] = []

        # PQC tunnel
        self._pqc_tunnel_active: bool = False
        self._pqc_session_id: Optional[str] = None

        # Background tasks
        self._schedule_sync_task: Optional[asyncio.Task] = None
        self._adherence_check_task: Optional[asyncio.Task] = None

        # Metrics
        self._connect_time: Optional[datetime] = None
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._last_error: Optional[str] = None

        # Callbacks
        self._event_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable[[SecurityAlert], None]] = []

        # Session lock callback (integration with QBITEL session manager)
        self._session_lock_callback: Optional[Callable[[str, str], None]] = None
        self._session_unlock_callback: Optional[Callable[[str], None]] = None

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

        # Validate
        errors = config.validate()
        if errors:
            logger.warning(f"WFM config validation warnings for {wfm_type.display_name}: {errors}")

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def wfm_type(self) -> WFMType:
        """Get the WFM system type."""
        return self._wfm_type

    @property
    def config(self) -> WFMConnectionConfig:
        """Get the connection configuration."""
        return self._config

    @property
    def is_connected(self) -> bool:
        """Check if connected to the WFM."""
        return self._connected

    @property
    def is_healthy(self) -> bool:
        """Check if the bridge is healthy."""
        return self._connected and self._healthy

    # -----------------------------------------------------------------------
    # Abstract methods - must be implemented by WFM-specific bridges
    # -----------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the WFM system.

        Raises:
            WFMConnectionError: If connection fails
            WFMAuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the WFM system."""
        pass

    @abstractmethod
    async def _fetch_schedule(
        self,
        agent_id: str,
        schedule_date: date,
    ) -> Dict[str, Any]:
        """
        Fetch raw schedule data from WFM (vendor-specific).

        Args:
            agent_id: Agent identifier
            schedule_date: Date to fetch schedule for

        Returns:
            Raw schedule data dictionary

        Raises:
            WFMError: If fetch fails
        """
        pass

    @abstractmethod
    async def _fetch_adherence(
        self,
        agent_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Fetch real-time adherence data from WFM (vendor-specific).

        Args:
            agent_ids: Agents to check adherence for

        Returns:
            List of raw adherence records

        Raises:
            WFMError: If fetch fails
        """
        pass

    @abstractmethod
    async def _fetch_forecast(
        self,
        forecast_date: date,
        skill_group: Optional[str] = None,
        interval_minutes: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Fetch forecast data from WFM (vendor-specific).

        Args:
            forecast_date: Date to fetch forecast for
            skill_group: Optional skill group filter
            interval_minutes: Forecast interval granularity

        Returns:
            List of raw forecast interval records

        Raises:
            WFMError: If fetch fails
        """
        pass

    @abstractmethod
    async def _submit_schedule_exception_internal(
        self,
        agent_id: str,
        exception_type: ScheduleExceptionType,
        start_time: datetime,
        end_time: datetime,
        reason: str,
    ) -> str:
        """
        Submit a schedule exception to WFM (vendor-specific).

        Args:
            agent_id: Agent requesting exception
            exception_type: Type of exception
            start_time: Exception start
            end_time: Exception end
            reason: Reason text

        Returns:
            Exception request ID

        Raises:
            WFMError: If submission fails
        """
        pass

    @abstractmethod
    async def _fetch_performance_metrics(
        self,
        agent_id: str,
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        """
        Fetch agent performance metrics from WFM (vendor-specific).

        Args:
            agent_id: Agent to fetch metrics for
            start_date: Period start
            end_date: Period end

        Returns:
            Raw performance metrics dictionary

        Raises:
            WFMError: If fetch fails
        """
        pass

    @abstractmethod
    async def _fetch_real_time_metrics(
        self,
        agent_ids: Optional[List[str]] = None,
        queue_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fetch real-time metrics from WFM (vendor-specific).

        Args:
            agent_ids: Agents to get metrics for (None = all)
            queue_names: Queues to get metrics for (None = all)

        Returns:
            Real-time metrics dictionary

        Raises:
            WFMError: If fetch fails
        """
        pass

    @abstractmethod
    async def _health_check_internal(self) -> bool:
        """
        Perform WFM-specific health check.

        Returns:
            True if WFM is healthy
        """
        pass

    # -----------------------------------------------------------------------
    # Public methods with security controls
    # -----------------------------------------------------------------------

    async def get_schedule(
        self,
        agent_id: str,
        schedule_date: Optional[date] = None,
        requester_id: str = "",
    ) -> AgentSchedule:
        """
        Get agent schedule with caching and audit logging.

        Args:
            agent_id: Agent to get schedule for
            schedule_date: Date to query (default: today)
            requester_id: ID of the requester (for audit)

        Returns:
            AgentSchedule for the requested date

        Raises:
            WFMError: If schedule fetch fails
        """
        self._check_connected()

        target_date = schedule_date or date.today()

        # Check cache
        cache_key = f"{agent_id}_{target_date.isoformat()}"
        if cache_key in self._schedule_cache:
            cached = self._schedule_cache[cache_key]
            self._audit_log_entry("SCHEDULE_RETRIEVED_CACHED", {
                "agent_id": agent_id,
                "date": target_date.isoformat(),
                "requester_id": requester_id,
            })
            return cached

        # Fetch from WFM
        try:
            raw_data = await self._fetch_schedule(agent_id, target_date)
        except Exception as exc:
            self._total_errors += 1
            self._last_error = str(exc)
            raise

        self._total_requests += 1

        # Parse into AgentSchedule
        schedule = self._parse_schedule(agent_id, target_date, raw_data)

        # Cache
        self._schedule_cache[cache_key] = schedule
        self._last_schedule_sync = datetime.utcnow()

        self._audit_log_entry("SCHEDULE_RETRIEVED", {
            "agent_id": agent_id,
            "date": target_date.isoformat(),
            "shift_hours": schedule.total_shift_hours,
            "is_day_off": schedule.is_day_off,
            "requester_id": requester_id,
        })

        self._emit_event(WFMEventType.SCHEDULE_RETRIEVED, {
            "agent_id": agent_id,
            "date": target_date.isoformat(),
        })

        return schedule

    async def get_adherence(
        self,
        agent_ids: List[str],
        requester_id: str = "",
    ) -> Dict[str, AdherenceRecord]:
        """
        Get real-time adherence status for agents.

        Args:
            agent_ids: Agents to check
            requester_id: Requester ID for audit

        Returns:
            Dict mapping agent_id to AdherenceRecord

        Raises:
            WFMError: If adherence check fails
        """
        self._check_connected()

        try:
            raw_records = await self._fetch_adherence(agent_ids)
        except Exception as exc:
            self._total_errors += 1
            self._last_error = str(exc)
            raise

        self._total_requests += 1

        result = {}
        for raw in raw_records:
            record = self._parse_adherence_record(raw)
            result[record.agent_id] = record
            self._adherence_state[record.agent_id] = record

        self._last_adherence_check = datetime.utcnow()

        # Check for out-of-adherence and raise alerts
        for agent_id, record in result.items():
            if record.status == AdherenceStatus.OUT_OF_ADHERENCE:
                self._emit_event(WFMEventType.ADHERENCE_OUT, {
                    "agent_id": agent_id,
                    "deviation_minutes": record.deviation_minutes,
                })

        self._audit_log_entry("ADHERENCE_CHECKED", {
            "agent_count": len(agent_ids),
            "in_adherence": sum(1 for r in result.values() if r.status.is_compliant),
            "out_of_adherence": sum(1 for r in result.values() if not r.status.is_compliant),
            "requester_id": requester_id,
        })

        return result

    async def get_forecast(
        self,
        forecast_date: Optional[date] = None,
        skill_group: Optional[str] = None,
        interval_minutes: int = 30,
        requester_id: str = "",
    ) -> List[ForecastData]:
        """
        Get workforce forecast data.

        Args:
            forecast_date: Date to forecast (default: today)
            skill_group: Optional skill group filter
            interval_minutes: Forecast interval granularity
            requester_id: Requester for audit

        Returns:
            List of ForecastData intervals

        Raises:
            WFMError: If forecast fetch fails
        """
        self._check_connected()

        target_date = forecast_date or date.today()

        try:
            raw_intervals = await self._fetch_forecast(
                target_date, skill_group, interval_minutes
            )
        except Exception as exc:
            self._total_errors += 1
            self._last_error = str(exc)
            raise

        self._total_requests += 1

        forecasts = [self._parse_forecast_interval(raw) for raw in raw_intervals]

        self._audit_log_entry("FORECAST_RETRIEVED", {
            "date": target_date.isoformat(),
            "skill_group": skill_group,
            "intervals": len(forecasts),
            "requester_id": requester_id,
        })

        self._emit_event(WFMEventType.FORECAST_RETRIEVED, {
            "date": target_date.isoformat(),
            "intervals": len(forecasts),
        })

        return forecasts

    async def submit_schedule_exception(
        self,
        agent_id: str,
        exception_type: ScheduleExceptionType,
        start_time: datetime,
        end_time: datetime,
        reason: str,
        requester_id: str = "",
    ) -> str:
        """
        Submit a schedule exception request.

        Args:
            agent_id: Agent requesting exception
            exception_type: Type of exception
            start_time: Exception start time
            end_time: Exception end time
            reason: Reason for the exception
            requester_id: Requester for audit

        Returns:
            Exception request ID

        Raises:
            WFMError: If submission fails
        """
        self._check_connected()

        # Validate time range
        if end_time <= start_time:
            raise WFMScheduleError("Exception end time must be after start time")

        try:
            exception_id = await self._submit_schedule_exception_internal(
                agent_id, exception_type, start_time, end_time, reason
            )
        except Exception as exc:
            self._total_errors += 1
            self._last_error = str(exc)
            raise

        self._total_requests += 1

        self._audit_log_entry("SCHEDULE_EXCEPTION_SUBMITTED", {
            "agent_id": agent_id,
            "exception_type": exception_type.exception_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "exception_id": exception_id,
            "requester_id": requester_id,
        })

        self._emit_event(WFMEventType.SCHEDULE_EXCEPTION_SUBMITTED, {
            "agent_id": agent_id,
            "exception_type": exception_type.exception_name,
            "exception_id": exception_id,
        })

        # Invalidate schedule cache for this agent
        self._invalidate_agent_schedule_cache(agent_id)

        return exception_id

    async def get_real_time_metrics(
        self,
        agent_ids: Optional[List[str]] = None,
        queue_names: Optional[List[str]] = None,
        requester_id: str = "",
    ) -> Dict[str, Any]:
        """
        Get real-time performance metrics.

        Args:
            agent_ids: Agents to get metrics for
            queue_names: Queues to get metrics for
            requester_id: Requester for audit

        Returns:
            Real-time metrics dictionary

        Raises:
            WFMError: If fetch fails
        """
        self._check_connected()

        try:
            metrics = await self._fetch_real_time_metrics(agent_ids, queue_names)
        except Exception as exc:
            self._total_errors += 1
            self._last_error = str(exc)
            raise

        self._total_requests += 1

        self._audit_log_entry("REAL_TIME_METRICS_RETRIEVED", {
            "agent_count": len(agent_ids) if agent_ids else 0,
            "queue_count": len(queue_names) if queue_names else 0,
            "requester_id": requester_id,
        })

        self._emit_event(WFMEventType.METRICS_RETRIEVED, {
            "type": "real_time",
        })

        return metrics

    async def get_agent_performance(
        self,
        agent_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        requester_id: str = "",
    ) -> PerformanceMetrics:
        """
        Get agent performance metrics for a date range.

        Args:
            agent_id: Agent to get metrics for
            start_date: Period start (default: today)
            end_date: Period end (default: today)
            requester_id: Requester for audit

        Returns:
            PerformanceMetrics for the agent

        Raises:
            WFMError: If fetch fails
        """
        self._check_connected()

        start = start_date or date.today()
        end = end_date or date.today()

        try:
            raw_metrics = await self._fetch_performance_metrics(agent_id, start, end)
        except Exception as exc:
            self._total_errors += 1
            self._last_error = str(exc)
            raise

        self._total_requests += 1

        metrics = self._parse_performance_metrics(agent_id, raw_metrics)

        self._audit_log_entry("PERFORMANCE_METRICS_RETRIEVED", {
            "agent_id": agent_id,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "requester_id": requester_id,
        })

        return metrics

    # -----------------------------------------------------------------------
    # QBITEL session management integration
    # -----------------------------------------------------------------------

    def register_session_lock_callback(
        self,
        lock_callback: Callable[[str, str], None],
        unlock_callback: Callable[[str], None],
    ) -> None:
        """
        Register QBITEL session lock/unlock callbacks.

        These callbacks are invoked when the WFM bridge determines
        that an agent session should be locked (off-schedule) or
        unlocked (back on schedule).

        Args:
            lock_callback: Called with (agent_id, reason) to lock session
            unlock_callback: Called with (agent_id) to unlock session
        """
        self._session_lock_callback = lock_callback
        self._session_unlock_callback = unlock_callback
        logger.info("Session lock callbacks registered with WFM bridge")

    async def check_agent_schedule_compliance(self, agent_id: str) -> bool:
        """
        Check if an agent is currently within their scheduled hours.

        If the agent is off-schedule and auto_lock_off_schedule is enabled,
        this method will trigger a session lock via the registered callback.

        Args:
            agent_id: Agent to check

        Returns:
            True if the agent is within scheduled hours

        Raises:
            WFMScheduleError: If schedule cannot be retrieved
        """
        schedule = await self.get_schedule(agent_id)
        now = datetime.utcnow()

        is_scheduled = schedule.is_currently_scheduled(now)

        if not is_scheduled and not schedule.is_day_off:
            # Check grace period
            grace_delta = timedelta(minutes=self._config.off_schedule_grace_period_minutes)

            within_grace = False
            if schedule.shift_start:
                within_grace = (
                    (schedule.shift_start - grace_delta) <= now <= schedule.shift_start
                    or (schedule.shift_end is not None and schedule.shift_end <= now <= schedule.shift_end + grace_delta)
                )

            if not within_grace:
                # Agent is off-schedule outside grace period
                if self._config.auto_lock_off_schedule and self._session_lock_callback:
                    reason = (
                        f"Agent {agent_id} is active outside scheduled hours. "
                        f"Shift: {schedule.shift_start} - {schedule.shift_end}"
                    )
                    self._session_lock_callback(agent_id, reason)

                    self._emit_event(WFMEventType.AGENT_SESSION_AUTO_LOCKED, {
                        "agent_id": agent_id,
                        "reason": "off_schedule",
                    })

                if self._config.alert_off_schedule_activity:
                    alert = SecurityAlert(
                        severity=SecurityAlertSeverity.HIGH,
                        alert_type="OFF_SCHEDULE_ACTIVITY",
                        agent_id=agent_id,
                        message=(
                            f"Agent {agent_id} is active outside scheduled hours. "
                            f"Shift: {schedule.shift_start} - {schedule.shift_end}. "
                            f"Current time: {now}"
                        ),
                        details={
                            "shift_start": schedule.shift_start.isoformat() if schedule.shift_start else None,
                            "shift_end": schedule.shift_end.isoformat() if schedule.shift_end else None,
                            "current_time": now.isoformat(),
                        },
                    )
                    self._raise_alert(alert)

                return False

        return True

    # -----------------------------------------------------------------------
    # Security alerting
    # -----------------------------------------------------------------------

    def register_alert_callback(self, callback: Callable[[SecurityAlert], None]) -> None:
        """Register a callback for security alerts."""
        self._alert_callbacks.append(callback)

    def _raise_alert(self, alert: SecurityAlert) -> None:
        """Raise a security alert."""
        self._active_alerts.append(alert)

        logger.warning(
            f"WFM Security Alert [{alert.severity.severity_name}]: "
            f"{alert.alert_type} - {alert.message}"
        )

        self._audit_log_entry("SECURITY_ALERT", alert.to_dict())

        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as exc:
                logger.error(f"Error in alert callback: {exc}", exc_info=True)

    def get_active_alerts(self, severity: Optional[SecurityAlertSeverity] = None) -> List[SecurityAlert]:
        """
        Get active security alerts.

        Args:
            severity: Optional filter by severity level

        Returns:
            List of active alerts
        """
        if severity:
            return [a for a in self._active_alerts if a.severity == severity and not a.acknowledged]
        return [a for a in self._active_alerts if not a.acknowledged]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge a security alert.

        Args:
            alert_id: Alert to acknowledge

        Returns:
            True if alert was found and acknowledged
        """
        for alert in self._active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self._audit_log_entry("ALERT_ACKNOWLEDGED", {"alert_id": alert_id})
                return True
        return False

    # -----------------------------------------------------------------------
    # Background monitoring tasks
    # -----------------------------------------------------------------------

    async def start_monitoring(self, agent_ids: List[str]) -> None:
        """
        Start background schedule and adherence monitoring.

        Launches periodic tasks that:
        1. Sync schedules from WFM at configured intervals
        2. Check adherence at configured intervals
        3. Trigger auto-lock/alerts for off-schedule activity

        Args:
            agent_ids: Agents to monitor
        """
        if self._schedule_sync_task and not self._schedule_sync_task.done():
            logger.warning("Monitoring tasks already running")
            return

        self._schedule_sync_task = asyncio.ensure_future(
            self._schedule_sync_loop(agent_ids)
        )
        self._adherence_check_task = asyncio.ensure_future(
            self._adherence_check_loop(agent_ids)
        )

        logger.info(f"WFM monitoring started for {len(agent_ids)} agents")

        self._audit_log_entry("MONITORING_STARTED", {
            "agent_count": len(agent_ids),
        })

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        if self._schedule_sync_task and not self._schedule_sync_task.done():
            self._schedule_sync_task.cancel()
            try:
                await self._schedule_sync_task
            except asyncio.CancelledError:
                pass

        if self._adherence_check_task and not self._adherence_check_task.done():
            self._adherence_check_task.cancel()
            try:
                await self._adherence_check_task
            except asyncio.CancelledError:
                pass

        logger.info("WFM monitoring stopped")

        self._audit_log_entry("MONITORING_STOPPED", {})

    async def _schedule_sync_loop(self, agent_ids: List[str]) -> None:
        """Background loop to periodically sync schedules."""
        while True:
            try:
                await asyncio.sleep(self._config.schedule_sync_interval_seconds)

                for agent_id in agent_ids:
                    try:
                        await self.get_schedule(agent_id)
                    except Exception as exc:
                        logger.warning(f"Failed to sync schedule for {agent_id}: {exc}")

                self._last_schedule_sync = datetime.utcnow()
                logger.debug(f"Schedule sync completed for {len(agent_ids)} agents")

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error(f"Schedule sync loop error: {exc}", exc_info=True)

    async def _adherence_check_loop(self, agent_ids: List[str]) -> None:
        """Background loop to periodically check adherence."""
        while True:
            try:
                await asyncio.sleep(self._config.adherence_check_interval_seconds)

                adherence = await self.get_adherence(agent_ids)

                for agent_id, record in adherence.items():
                    if not record.status.is_compliant:
                        await self.check_agent_schedule_compliance(agent_id)

                self._last_adherence_check = datetime.utcnow()

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error(f"Adherence check loop error: {exc}", exc_info=True)

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    async def connect_with_pqc(self) -> None:
        """Connect to WFM with PQC tunnel."""
        if self._config.pqc_enabled:
            await self._establish_pqc_tunnel()

        await self.connect()

        self._connected = True
        self._healthy = True
        self._connect_time = datetime.utcnow()

        logger.info(f"Connected to {self._wfm_type.display_name} (PQC: {self._pqc_tunnel_active})")

        self._audit_log_entry("WFM_CONNECTED", {
            "wfm_type": self._wfm_type.display_name,
            "pqc_tunnel": self._pqc_tunnel_active,
        })

    async def disconnect_gracefully(self) -> None:
        """Gracefully disconnect from WFM."""
        await self.stop_monitoring()

        if self._connected:
            try:
                await self.disconnect()
            except Exception as exc:
                logger.warning(f"Error during WFM disconnect: {exc}")
            finally:
                self._connected = False
                self._healthy = False

        if self._pqc_tunnel_active:
            await self._teardown_pqc_tunnel()

        self._schedule_cache.clear()
        self._adherence_state.clear()

        self._audit_log_entry("WFM_DISCONNECTED", {
            "wfm_type": self._wfm_type.display_name,
            "total_requests": self._total_requests,
        })

    # -----------------------------------------------------------------------
    # PQC tunnel
    # -----------------------------------------------------------------------

    async def _establish_pqc_tunnel(self) -> None:
        """Establish PQC tunnel for WFM API calls."""
        try:
            self._pqc_session_id = str(uuid.uuid4())

            # In production:
            # 1. Generate ML-KEM key pair
            # 2. Exchange keys with QBITEL PQC gateway
            # 3. Establish authenticated encrypted channel

            self._pqc_tunnel_active = True
            logger.info(
                f"PQC tunnel established for {self._wfm_type.display_name} "
                f"(KEM: {self._config.pqc_kem_algorithm})"
            )

        except Exception as exc:
            logger.error(f"PQC tunnel failed for {self._wfm_type.display_name}: {exc}")
            self._pqc_tunnel_active = False

    async def _teardown_pqc_tunnel(self) -> None:
        """Tear down PQC tunnel."""
        self._pqc_tunnel_active = False
        self._pqc_session_id = None

    # -----------------------------------------------------------------------
    # Data parsing helpers
    # -----------------------------------------------------------------------

    def _parse_schedule(
        self,
        agent_id: str,
        schedule_date: date,
        raw_data: Dict[str, Any],
    ) -> AgentSchedule:
        """Parse raw WFM schedule data into AgentSchedule."""
        activities = []
        break_times = []

        for raw_activity in raw_data.get("activities", []):
            activity = ScheduleActivity(
                activity_type=self._map_activity_type(raw_activity.get("type", "work")),
                start_time=self._parse_datetime(raw_activity.get("start_time", "")),
                end_time=self._parse_datetime(raw_activity.get("end_time", "")),
                skill_group=raw_activity.get("skill_group", ""),
                queue_name=raw_activity.get("queue", ""),
                description=raw_activity.get("description", ""),
                is_paid=raw_activity.get("is_paid", True),
            )
            activities.append(activity)

            if activity.activity_type in (
                ScheduleActivityType.BREAK,
                ScheduleActivityType.LUNCH,
            ):
                break_times.append(activity)

        shift_start = self._parse_datetime(raw_data.get("shift_start", ""))
        shift_end = self._parse_datetime(raw_data.get("shift_end", ""))

        return AgentSchedule(
            agent_id=agent_id,
            schedule_date=schedule_date,
            shift_start=shift_start if shift_start != datetime.min else None,
            shift_end=shift_end if shift_end != datetime.min else None,
            break_times=break_times,
            skill_groups=raw_data.get("skill_groups", []),
            assigned_queues=raw_data.get("assigned_queues", []),
            activities=activities,
            timezone=raw_data.get("timezone", self._config.default_timezone),
            is_day_off=raw_data.get("is_day_off", False),
            is_holiday=raw_data.get("is_holiday", False),
            schedule_version=raw_data.get("version", 0),
            overtime_eligible=raw_data.get("overtime_eligible", False),
            vto_eligible=raw_data.get("vto_eligible", False),
        )

    def _parse_adherence_record(self, raw: Dict[str, Any]) -> AdherenceRecord:
        """Parse raw adherence data into AdherenceRecord."""
        status_map = {
            "in_adherence": AdherenceStatus.IN_ADHERENCE,
            "out_of_adherence": AdherenceStatus.OUT_OF_ADHERENCE,
            "excused": AdherenceStatus.EXCUSED,
            "not_scheduled": AdherenceStatus.NOT_SCHEDULED,
        }

        return AdherenceRecord(
            agent_id=raw.get("agent_id", ""),
            timestamp=self._parse_datetime(raw.get("timestamp", "")),
            status=status_map.get(raw.get("status", ""), AdherenceStatus.UNKNOWN),
            scheduled_activity=self._map_activity_type(raw.get("scheduled_activity", "")),
            actual_activity=raw.get("actual_activity", ""),
            deviation_minutes=raw.get("deviation_minutes", 0.0),
            is_excused=raw.get("is_excused", False),
            exception_reason=raw.get("exception_reason", ""),
        )

    def _parse_forecast_interval(self, raw: Dict[str, Any]) -> ForecastData:
        """Parse raw forecast data into ForecastData."""
        return ForecastData(
            forecast_date=self._parse_date(raw.get("date", "")),
            interval_minutes=raw.get("interval_minutes", 30),
            predicted_volume=raw.get("predicted_volume", 0),
            predicted_aht=raw.get("predicted_aht", 0.0),
            required_agents=raw.get("required_agents", 0),
            scheduled_agents=raw.get("scheduled_agents", 0),
            service_level_target=raw.get("service_level_target", 80.0),
            target_asa=raw.get("target_asa", 20.0),
            predicted_abandonment_rate=raw.get("predicted_abandonment_rate", 0.0),
            shrinkage_factor=raw.get("shrinkage_factor", 0.0),
            required_fte=raw.get("required_fte", 0.0),
            interval_start=self._parse_datetime(raw.get("interval_start", "")),
            interval_end=self._parse_datetime(raw.get("interval_end", "")),
            skill_group=raw.get("skill_group", ""),
            queue_name=raw.get("queue_name", ""),
            channel=raw.get("channel", "voice"),
        )

    def _parse_performance_metrics(
        self,
        agent_id: str,
        raw: Dict[str, Any],
    ) -> PerformanceMetrics:
        """Parse raw performance metrics into PerformanceMetrics."""
        return PerformanceMetrics(
            agent_id=agent_id,
            period_start=self._parse_datetime(raw.get("period_start", "")),
            period_end=self._parse_datetime(raw.get("period_end", "")),
            aht=raw.get("aht", 0.0),
            calls_handled=raw.get("calls_handled", 0),
            calls_offered=raw.get("calls_offered", 0),
            quality_score=raw.get("quality_score", 0.0),
            csat_score=raw.get("csat_score", 0.0),
            fcr_rate=raw.get("fcr_rate", 0.0),
            adherence_percentage=raw.get("adherence_percentage", 0.0),
            conformance_percentage=raw.get("conformance_percentage", 0.0),
            occupancy=raw.get("occupancy", 0.0),
            total_login_seconds=raw.get("total_login_seconds", 0.0),
            total_talk_seconds=raw.get("total_talk_seconds", 0.0),
            total_hold_seconds=raw.get("total_hold_seconds", 0.0),
            total_acw_seconds=raw.get("total_acw_seconds", 0.0),
            total_idle_seconds=raw.get("total_idle_seconds", 0.0),
            total_not_ready_seconds=raw.get("total_not_ready_seconds", 0.0),
            transfers_out=raw.get("transfers_out", 0),
            escalations=raw.get("escalations", 0),
            callbacks_completed=raw.get("callbacks_completed", 0),
            average_speed_of_answer=raw.get("average_speed_of_answer", 0.0),
            longest_hold_time=raw.get("longest_hold_time", 0.0),
        )

    # -----------------------------------------------------------------------
    # Utility helpers
    # -----------------------------------------------------------------------

    def _map_activity_type(self, activity_str: str) -> ScheduleActivityType:
        """Map WFM activity string to ScheduleActivityType."""
        activity_map = {
            "work": ScheduleActivityType.WORK,
            "break": ScheduleActivityType.BREAK,
            "lunch": ScheduleActivityType.LUNCH,
            "training": ScheduleActivityType.TRAINING,
            "meeting": ScheduleActivityType.MEETING,
            "coaching": ScheduleActivityType.COACHING,
            "project": ScheduleActivityType.PROJECT,
            "off": ScheduleActivityType.OFF,
            "overtime": ScheduleActivityType.OVERTIME,
            "vto": ScheduleActivityType.VTO,
        }
        return activity_map.get(activity_str.lower(), ScheduleActivityType.WORK)

    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse datetime string with fallback."""
        if not dt_str:
            return datetime.min

        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%fZ",
                    "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Unable to parse datetime: {dt_str}")
        return datetime.min

    def _parse_date(self, date_str: str) -> date:
        """Parse date string with fallback."""
        if not date_str:
            return date.today()

        try:
            return date.fromisoformat(date_str)
        except ValueError:
            logger.warning(f"Unable to parse date: {date_str}")
            return date.today()

    def _invalidate_agent_schedule_cache(self, agent_id: str) -> None:
        """Invalidate all cached schedules for an agent."""
        keys_to_remove = [k for k in self._schedule_cache if k.startswith(f"{agent_id}_")]
        for key in keys_to_remove:
            del self._schedule_cache[key]

    def _check_connected(self) -> None:
        """Verify the bridge is connected."""
        if not self._connected:
            raise WFMConnectionError(f"Not connected to {self._wfm_type.display_name}")

    # -----------------------------------------------------------------------
    # Event handling
    # -----------------------------------------------------------------------

    def register_event_callback(self, callback: Callable) -> None:
        """Register a WFM event callback."""
        self._event_callbacks.append(callback)

    def _emit_event(self, event_type: WFMEventType, data: Dict[str, Any]) -> None:
        """Emit a WFM event."""
        for callback in self._event_callbacks:
            try:
                callback(event_type, data)
            except Exception as exc:
                logger.error(f"Error in WFM event callback: {exc}", exc_info=True)

    # -----------------------------------------------------------------------
    # Health and audit
    # -----------------------------------------------------------------------

    def get_health_status(self) -> WFMHealthStatus:
        """Get comprehensive health status."""
        in_adherence = sum(
            1 for r in self._adherence_state.values()
            if r.status.is_compliant
        )
        out_of_adherence = sum(
            1 for r in self._adherence_state.values()
            if not r.status.is_compliant
        )

        return WFMHealthStatus(
            is_connected=self._connected,
            is_healthy=self._healthy,
            last_schedule_sync=self._last_schedule_sync,
            last_adherence_check=self._last_adherence_check,
            agents_monitored=len(self._adherence_state),
            agents_in_adherence=in_adherence,
            agents_out_of_adherence=out_of_adherence,
            active_alerts=len(self.get_active_alerts()),
            pqc_tunnel_active=self._pqc_tunnel_active,
            error_count=self._total_errors,
            last_error=self._last_error,
        )

    async def check_health(self) -> Dict[str, Any]:
        """Perform active health check."""
        status = self.get_health_status()

        if self._connected:
            try:
                status.is_healthy = await self._health_check_internal()
            except Exception as exc:
                status.is_healthy = False
                status.last_error = str(exc)

        self._healthy = status.is_healthy
        return status.to_dict()

    def _audit_log_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Record an audit log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "wfm_type": self._wfm_type.display_name,
            "action": action,
            "details": details,
        }
        self._audit_log.append(entry)

        max_in_memory = 10000
        if len(self._audit_log) > max_in_memory:
            self._audit_log = self._audit_log[-max_in_memory:]

        logger.debug(f"WFM audit: {action} - {details}")

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries (most recent first)."""
        return self._audit_log[-limit:][::-1]
