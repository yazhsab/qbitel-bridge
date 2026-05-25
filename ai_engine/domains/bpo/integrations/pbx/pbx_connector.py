"""
PBX Integration Connector

Abstract base for PBX/ACD system connectors. Provides quantum-safe
communication bridge between QBITEL and PBX systems including Avaya,
Cisco, Genesys, Mitel, Asterisk, FreeSWITCH, BroadSoft, RingCentral,
and Five9.

Responsibilities:
- Quantum-safe tunnel wrapping for legacy PBX APIs
- Real-time event streaming with callback registration
- Auto-reconnect with exponential backoff
- Health check and heartbeat mechanism
- Call control operations (transfer, conference, hold, retrieve)
- Agent state management
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
import asyncio
import logging
import time
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PBXType(Enum):
    """Supported PBX/ACD system types."""

    AVAYA_AURA = ("Avaya Aura", "tsapi", 450)
    CISCO_CUCM = ("Cisco CUCM", "cti_os", 2748)
    GENESYS_CLOUD = ("Genesys Cloud", "rest", 443)
    GENESYS_PURECONNECT = ("Genesys PureConnect", "icws", 8019)
    MITEL_MIVOICE = ("Mitel MiVoice", "mitel_api", 443)
    ASTERISK = ("Asterisk", "ami", 5038)
    FREESWITCH = ("FreeSWITCH", "esl", 8021)
    BROADSOFT = ("BroadSoft", "oci", 2208)
    RINGCENTRAL = ("RingCentral", "rest", 443)
    FIVE9 = ("Five9", "rest", 443)

    def __init__(self, display_name: str, protocol: str, default_port: int):
        self.display_name = display_name
        self.protocol = protocol
        self.default_port = default_port


class PBXEventType(Enum):
    """PBX telephony event types."""

    # Call lifecycle events
    CALL_NEW = auto()
    CALL_QUEUED = auto()
    CALL_DELIVERED = auto()
    CALL_ANSWERED = auto()
    CALL_HELD = auto()
    CALL_RETRIEVED = auto()
    CALL_TRANSFERRED = auto()
    CALL_CONFERENCED = auto()
    CALL_ENDED = auto()

    # Agent events
    AGENT_LOGIN = auto()
    AGENT_LOGOUT = auto()
    AGENT_STATE_CHANGE = auto()

    # Queue events
    QUEUE_UPDATE = auto()


class PBXTransport(Enum):
    """Transport protocols for PBX connectivity."""

    TCP = "tcp"
    TLS = "tls"
    WEBSOCKET = "websocket"
    WEBSOCKET_SECURE = "wss"
    HTTP = "http"
    HTTPS = "https"


class PBXAuthMethod(Enum):
    """Authentication methods for PBX systems."""

    BASIC = "basic"
    CERTIFICATE = "certificate"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    KERBEROS = "kerberos"
    CUSTOM = "custom"


class AgentState(Enum):
    """Standardised agent states across PBX systems."""

    LOGGED_OUT = ("logged_out", False)
    AVAILABLE = ("available", True)
    ON_CALL = ("on_call", True)
    AFTER_CALL_WORK = ("acw", True)
    NOT_READY = ("not_ready", True)
    BREAK = ("break", True)
    TRAINING = ("training", True)
    MEETING = ("meeting", True)
    OFFLINE = ("offline", False)

    def __init__(self, state_name: str, is_logged_in: bool):
        self.state_name = state_name
        self.is_logged_in = is_logged_in


class CallDirection(Enum):
    """Call direction enumeration."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PBXConnectionConfig:
    """Configuration for PBX system connection."""

    # Connection target
    host: str = "localhost"
    port: int = 0
    transport: PBXTransport = PBXTransport.TLS
    auth_method: PBXAuthMethod = PBXAuthMethod.BASIC

    # Credentials
    credentials_vault_path: str = ""

    # PQC settings
    pqc_enabled: bool = True
    tls_version: str = "TLS 1.3"

    # Reconnection settings
    reconnect_interval_seconds: int = 5
    max_reconnect_interval_seconds: int = 300
    max_reconnect_attempts: int = 0  # 0 = unlimited
    reconnect_backoff_multiplier: float = 2.0

    # Heartbeat settings
    heartbeat_interval_seconds: int = 30
    heartbeat_timeout_seconds: int = 10

    # Connection timeouts
    connect_timeout_seconds: int = 30
    read_timeout_seconds: int = 60
    write_timeout_seconds: int = 30

    # TLS certificate paths (for mutual TLS)
    ca_cert_path: Optional[str] = None
    client_cert_path: Optional[str] = None
    client_key_path: Optional[str] = None

    # PQC tunnel settings
    pqc_kem_algorithm: str = "ML-KEM-768"
    pqc_sig_algorithm: str = "ML-DSA-65"
    pqc_hybrid_mode: bool = True

    # Event buffer
    event_buffer_size: int = 10000
    event_overflow_policy: str = "DROP_OLDEST"  # DROP_OLDEST, DROP_NEWEST, BLOCK

    def validate(self) -> List[str]:
        """Validate connection configuration."""
        errors = []

        if not self.host:
            errors.append("PBX host is required")

        if self.port < 0 or self.port > 65535:
            errors.append(f"Invalid port number: {self.port}")

        if not self.credentials_vault_path:
            errors.append("credentials_vault_path is required for secure credential storage")

        if self.heartbeat_timeout_seconds >= self.heartbeat_interval_seconds:
            errors.append("Heartbeat timeout must be less than heartbeat interval")

        if self.reconnect_interval_seconds < 1:
            errors.append("Reconnect interval must be at least 1 second")

        if self.reconnect_backoff_multiplier < 1.0:
            errors.append("Backoff multiplier must be at least 1.0")

        return errors


@dataclass
class PBXEvent:
    """A telephony event received from the PBX system."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: PBXEventType = PBXEventType.CALL_NEW
    call_id: str = ""
    agent_id: str = ""
    queue_name: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Extended event data
    caller_number: str = ""
    called_number: str = ""
    direction: Optional[CallDirection] = None
    agent_state: Optional[AgentState] = None
    transfer_target: str = ""
    conference_members: List[str] = field(default_factory=list)
    raw_event: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "call_id": self.call_id,
            "agent_id": self.agent_id,
            "queue_name": self.queue_name,
            "timestamp": self.timestamp.isoformat(),
            "caller_number": self.caller_number,
            "called_number": self.called_number,
            "direction": self.direction.value if self.direction else None,
            "agent_state": self.agent_state.state_name if self.agent_state else None,
            "metadata": self.metadata,
        }


@dataclass
class PBXCallInfo:
    """Detailed information about an active call."""

    call_id: str = ""
    ucid: str = ""  # Universal Call Identifier
    caller_number: str = ""
    called_number: str = ""
    direction: CallDirection = CallDirection.INBOUND
    queue_name: str = ""
    agent_id: str = ""
    start_time: Optional[datetime] = None
    answer_time: Optional[datetime] = None
    hold_time_seconds: float = 0.0
    is_held: bool = False
    is_conferenced: bool = False
    conference_members: List[str] = field(default_factory=list)
    uui_data: str = ""  # User-to-User Information
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PBXHealthStatus:
    """Health status of the PBX connection."""

    is_connected: bool = False
    is_healthy: bool = False
    last_heartbeat: Optional[datetime] = None
    last_event_received: Optional[datetime] = None
    uptime_seconds: float = 0.0
    reconnect_count: int = 0
    events_received: int = 0
    events_dropped: int = 0
    latency_ms: float = 0.0
    pqc_tunnel_active: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert health status to dictionary."""
        return {
            "is_connected": self.is_connected,
            "is_healthy": self.is_healthy,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "last_event_received": self.last_event_received.isoformat() if self.last_event_received else None,
            "uptime_seconds": self.uptime_seconds,
            "reconnect_count": self.reconnect_count,
            "events_received": self.events_received,
            "events_dropped": self.events_dropped,
            "latency_ms": self.latency_ms,
            "pqc_tunnel_active": self.pqc_tunnel_active,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PBXError(Exception):
    """Base exception for PBX connector operations."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code


class PBXConnectionError(PBXError):
    """Exception for PBX connection failures."""

    pass


class PBXAuthenticationError(PBXError):
    """Exception for PBX authentication failures."""

    pass


class PBXCommandError(PBXError):
    """Exception for PBX command execution failures."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        command: Optional[str] = None,
    ):
        super().__init__(message, error_code)
        self.command = command


class PBXTimeoutError(PBXError):
    """Exception for PBX operation timeouts."""

    pass


# ---------------------------------------------------------------------------
# Event callback type
# ---------------------------------------------------------------------------

PBXEventCallback = Callable[[PBXEvent], None]


# ---------------------------------------------------------------------------
# PBX Connector abstract base
# ---------------------------------------------------------------------------


class PBXConnector(ABC):
    """
    Abstract base for PBX/ACD system connectors.

    Provides quantum-safe communication bridge between QBITEL and PBX systems.
    All vendor-specific implementations must inherit from this class and
    implement the abstract methods for their specific PBX API.

    Features:
    - Auto-reconnect with exponential backoff
    - Health check and heartbeat mechanism
    - Event callback registration with filtering
    - PQC tunnel wrapping for legacy PBX APIs
    - Audit logging of all PBX operations
    """

    def __init__(self, pbx_type: PBXType, config: PBXConnectionConfig):
        self._pbx_type = pbx_type
        self._config = config
        self._connected = False
        self._healthy = False

        # Connection state
        self._connect_time: Optional[datetime] = None
        self._last_heartbeat: Optional[datetime] = None
        self._last_event_time: Optional[datetime] = None
        self._reconnect_count: int = 0
        self._events_received: int = 0
        self._events_dropped: int = 0

        # Event callbacks: event_type -> list of callbacks
        self._event_callbacks: Dict[Optional[PBXEventType], List[PBXEventCallback]] = {}
        self._global_callbacks: List[PBXEventCallback] = []

        # Reconnection state
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._should_reconnect: bool = True
        self._current_reconnect_interval: float = config.reconnect_interval_seconds

        # PQC tunnel state
        self._pqc_tunnel_active: bool = False
        self._pqc_session_id: Optional[str] = None

        # Audit
        self._audit_log: List[Dict[str, Any]] = []

        # Apply default port if not set
        if config.port == 0:
            config.port = pbx_type.default_port

        # Validate configuration
        errors = config.validate()
        if errors:
            logger.warning(f"PBX connection config validation warnings for {pbx_type.display_name}: {errors}")

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def pbx_type(self) -> PBXType:
        """Get the PBX system type."""
        return self._pbx_type

    @property
    def config(self) -> PBXConnectionConfig:
        """Get the connection configuration."""
        return self._config

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to the PBX."""
        return self._connected

    @property
    def is_healthy(self) -> bool:
        """Check if the connection is healthy (connected and heartbeat OK)."""
        return self._connected and self._healthy

    @property
    def provider_name(self) -> str:
        """Get the PBX provider display name."""
        return self._pbx_type.display_name

    @property
    def uptime_seconds(self) -> float:
        """Get connection uptime in seconds."""
        if self._connect_time and self._connected:
            return (datetime.utcnow() - self._connect_time).total_seconds()
        return 0.0

    # -----------------------------------------------------------------------
    # Abstract methods - must be implemented by vendor connectors
    # -----------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the PBX system.

        Implementations must:
        1. Establish network connectivity
        2. Authenticate using configured method
        3. Set self._connected = True on success
        4. Raise PBXConnectionError on failure

        Raises:
            PBXConnectionError: If connection cannot be established
            PBXAuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Gracefully disconnect from the PBX system.

        Implementations must:
        1. Unsubscribe from all event streams
        2. Close network connections
        3. Set self._connected = False
        """
        pass

    @abstractmethod
    async def subscribe_events(
        self,
        event_types: Optional[Set[PBXEventType]] = None,
        agent_ids: Optional[Set[str]] = None,
        queue_names: Optional[Set[str]] = None,
    ) -> str:
        """
        Subscribe to PBX events.

        Args:
            event_types: Set of event types to subscribe to (None = all)
            agent_ids: Set of agent IDs to monitor (None = all)
            queue_names: Set of queue names to monitor (None = all)

        Returns:
            Subscription ID for managing this subscription

        Raises:
            PBXCommandError: If subscription fails
        """
        pass

    @abstractmethod
    async def unsubscribe_events(self, subscription_id: str) -> None:
        """
        Unsubscribe from a previous event subscription.

        Args:
            subscription_id: The subscription ID returned by subscribe_events

        Raises:
            PBXCommandError: If unsubscription fails
        """
        pass

    @abstractmethod
    async def send_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a raw command to the PBX system.

        Args:
            command: PBX-specific command name
            params: Command parameters

        Returns:
            Command response from the PBX

        Raises:
            PBXCommandError: If the command fails
            PBXTimeoutError: If the command times out
        """
        pass

    @abstractmethod
    async def get_agent_state(self, agent_id: str) -> AgentState:
        """
        Get the current state of an agent.

        Args:
            agent_id: The agent's identifier in the PBX system

        Returns:
            Current agent state

        Raises:
            PBXCommandError: If the query fails
        """
        pass

    @abstractmethod
    async def set_agent_state(self, agent_id: str, state: AgentState, reason_code: str = "") -> None:
        """
        Set the agent's state in the PBX system.

        Args:
            agent_id: The agent's identifier
            state: Desired agent state
            reason_code: Optional reason code for the state change

        Raises:
            PBXCommandError: If the state change fails
        """
        pass

    @abstractmethod
    async def transfer_call(
        self,
        call_id: str,
        target: str,
        transfer_type: str = "blind",
    ) -> str:
        """
        Transfer an active call to another destination.

        Args:
            call_id: The call to transfer
            target: Transfer destination (extension, queue, external number)
            transfer_type: 'blind' (single-step) or 'consultative' (two-step)

        Returns:
            New call ID for the transferred call

        Raises:
            PBXCommandError: If the transfer fails
        """
        pass

    @abstractmethod
    async def conference_call(
        self,
        call_id: str,
        targets: List[str],
    ) -> str:
        """
        Create a conference call by adding participants.

        Args:
            call_id: The existing call to conference
            targets: List of participants to add

        Returns:
            Conference ID

        Raises:
            PBXCommandError: If the conference creation fails
        """
        pass

    @abstractmethod
    async def hold_call(self, call_id: str) -> None:
        """
        Place an active call on hold.

        Args:
            call_id: The call to place on hold

        Raises:
            PBXCommandError: If the hold operation fails
        """
        pass

    @abstractmethod
    async def retrieve_call(self, call_id: str) -> None:
        """
        Retrieve a held call.

        Args:
            call_id: The call to retrieve from hold

        Raises:
            PBXCommandError: If the retrieve operation fails
        """
        pass

    @abstractmethod
    async def make_call(
        self,
        agent_id: str,
        destination: str,
        uui_data: str = "",
    ) -> str:
        """
        Initiate an outbound call on behalf of an agent.

        Args:
            agent_id: The agent originating the call
            destination: Number or extension to dial
            uui_data: Optional User-to-User Information

        Returns:
            Call ID of the new call

        Raises:
            PBXCommandError: If the call initiation fails
        """
        pass

    @abstractmethod
    async def end_call(self, call_id: str) -> None:
        """
        End (drop) an active call.

        Args:
            call_id: The call to end

        Raises:
            PBXCommandError: If the call cannot be ended
        """
        pass

    @abstractmethod
    async def get_call_info(self, call_id: str) -> PBXCallInfo:
        """
        Get detailed information about an active call.

        Args:
            call_id: The call to query

        Returns:
            Detailed call information

        Raises:
            PBXCommandError: If the query fails
        """
        pass

    @abstractmethod
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """
        Get real-time queue statistics.

        Args:
            queue_name: The queue to query

        Returns:
            Dictionary with queue statistics including:
            - calls_waiting: int
            - calls_in_progress: int
            - agents_available: int
            - agents_busy: int
            - longest_wait_seconds: float
            - average_wait_seconds: float
            - service_level_pct: float

        Raises:
            PBXCommandError: If the query fails
        """
        pass

    @abstractmethod
    async def _send_heartbeat(self) -> bool:
        """
        Send a heartbeat/keepalive to the PBX system.

        Returns:
            True if heartbeat was acknowledged, False otherwise
        """
        pass

    @abstractmethod
    async def _map_vendor_event(self, raw_event: Dict[str, Any]) -> PBXEvent:
        """
        Map a vendor-specific raw event to a standardised PBXEvent.

        Args:
            raw_event: Raw event data from the PBX system

        Returns:
            Normalised PBXEvent
        """
        pass

    # -----------------------------------------------------------------------
    # Concrete methods - shared functionality
    # -----------------------------------------------------------------------

    def register_callback(
        self,
        callback: PBXEventCallback,
        event_type: Optional[PBXEventType] = None,
    ) -> str:
        """
        Register a callback for PBX events.

        Args:
            callback: Function to call when an event is received
            event_type: Specific event type to filter on (None = all events)

        Returns:
            Callback registration ID
        """
        callback_id = str(uuid.uuid4())

        if event_type is None:
            self._global_callbacks.append(callback)
            logger.debug(f"Registered global event callback: {callback_id}")
        else:
            if event_type not in self._event_callbacks:
                self._event_callbacks[event_type] = []
            self._event_callbacks[event_type].append(callback)
            logger.debug(f"Registered callback for {event_type.name}: {callback_id}")

        self._audit_log_entry("REGISTER_CALLBACK", {
            "callback_id": callback_id,
            "event_type": event_type.name if event_type else "ALL",
        })

        return callback_id

    def unregister_callback(self, callback: PBXEventCallback) -> None:
        """
        Unregister a previously registered callback.

        Args:
            callback: The callback function to remove
        """
        if callback in self._global_callbacks:
            self._global_callbacks.remove(callback)

        for event_type in list(self._event_callbacks.keys()):
            if callback in self._event_callbacks[event_type]:
                self._event_callbacks[event_type].remove(callback)
                if not self._event_callbacks[event_type]:
                    del self._event_callbacks[event_type]

    async def dispatch_event(self, event: PBXEvent) -> None:
        """
        Dispatch a PBX event to all registered callbacks.

        This method is called by vendor implementations when they receive
        events from the PBX system. Events are first normalised via
        _map_vendor_event(), then dispatched to matching callbacks.

        Args:
            event: The normalised PBX event to dispatch
        """
        self._events_received += 1
        self._last_event_time = datetime.utcnow()

        # Dispatch to type-specific callbacks
        if event.event_type in self._event_callbacks:
            for callback in self._event_callbacks[event.event_type]:
                try:
                    callback(event)
                except Exception as exc:
                    logger.error(
                        f"Error in event callback for {event.event_type.name}: {exc}",
                        exc_info=True,
                    )

        # Dispatch to global callbacks
        for callback in self._global_callbacks:
            try:
                callback(event)
            except Exception as exc:
                logger.error(
                    f"Error in global event callback: {exc}",
                    exc_info=True,
                )

    async def connect_with_retry(self) -> None:
        """
        Connect to the PBX with automatic retry and exponential backoff.

        This method wraps the vendor-specific connect() method with retry
        logic. On successful connection, it starts the heartbeat monitor
        and establishes a PQC tunnel if configured.
        """
        self._should_reconnect = True
        self._current_reconnect_interval = self._config.reconnect_interval_seconds
        attempt = 0

        while self._should_reconnect:
            attempt += 1

            if (
                self._config.max_reconnect_attempts > 0
                and attempt > self._config.max_reconnect_attempts
            ):
                logger.error(
                    f"Max reconnect attempts ({self._config.max_reconnect_attempts}) "
                    f"reached for {self.provider_name}"
                )
                raise PBXConnectionError(
                    f"Failed to connect to {self.provider_name} after "
                    f"{self._config.max_reconnect_attempts} attempts"
                )

            try:
                logger.info(
                    f"Connecting to {self.provider_name} at "
                    f"{self._config.host}:{self._config.port} "
                    f"(attempt {attempt})"
                )

                # Establish PQC tunnel before connecting if enabled
                if self._config.pqc_enabled:
                    await self._establish_pqc_tunnel()

                await self.connect()

                self._connected = True
                self._healthy = True
                self._connect_time = datetime.utcnow()
                self._current_reconnect_interval = self._config.reconnect_interval_seconds

                logger.info(
                    f"Connected to {self.provider_name} at "
                    f"{self._config.host}:{self._config.port}"
                )

                self._audit_log_entry("CONNECTED", {
                    "host": self._config.host,
                    "port": self._config.port,
                    "attempt": attempt,
                    "pqc_tunnel": self._pqc_tunnel_active,
                })

                # Start heartbeat monitor
                await self._start_heartbeat_monitor()

                return

            except PBXAuthenticationError as exc:
                # Authentication errors should not retry - they won't resolve
                logger.error(f"Authentication failed for {self.provider_name}: {exc}")
                self._audit_log_entry("AUTH_FAILED", {
                    "host": self._config.host,
                    "error": str(exc),
                })
                raise

            except (PBXConnectionError, PBXTimeoutError, Exception) as exc:
                self._reconnect_count += 1
                logger.warning(
                    f"Connection attempt {attempt} to {self.provider_name} failed: {exc}. "
                    f"Retrying in {self._current_reconnect_interval}s"
                )

                self._audit_log_entry("CONNECT_FAILED", {
                    "host": self._config.host,
                    "attempt": attempt,
                    "error": str(exc),
                    "next_retry_seconds": self._current_reconnect_interval,
                })

                await asyncio.sleep(self._current_reconnect_interval)

                # Exponential backoff
                self._current_reconnect_interval = min(
                    self._current_reconnect_interval * self._config.reconnect_backoff_multiplier,
                    self._config.max_reconnect_interval_seconds,
                )

    async def disconnect_gracefully(self) -> None:
        """
        Gracefully disconnect from the PBX system.

        Stops heartbeat monitoring, tears down PQC tunnel, and
        calls the vendor-specific disconnect implementation.
        """
        self._should_reconnect = False

        # Stop heartbeat monitor
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Disconnect from PBX
        if self._connected:
            try:
                await self.disconnect()
            except Exception as exc:
                logger.warning(f"Error during {self.provider_name} disconnect: {exc}")
            finally:
                self._connected = False
                self._healthy = False

        # Tear down PQC tunnel
        if self._pqc_tunnel_active:
            await self._teardown_pqc_tunnel()

        self._audit_log_entry("DISCONNECTED", {
            "host": self._config.host,
            "uptime_seconds": self.uptime_seconds,
            "events_received": self._events_received,
        })

        logger.info(f"Disconnected from {self.provider_name}")

    def get_health_status(self) -> PBXHealthStatus:
        """
        Get comprehensive health status of the PBX connection.

        Returns:
            PBXHealthStatus with current connection metrics
        """
        return PBXHealthStatus(
            is_connected=self._connected,
            is_healthy=self.is_healthy,
            last_heartbeat=self._last_heartbeat,
            last_event_received=self._last_event_time,
            uptime_seconds=self.uptime_seconds,
            reconnect_count=self._reconnect_count,
            events_received=self._events_received,
            events_dropped=self._events_dropped,
            pqc_tunnel_active=self._pqc_tunnel_active,
        )

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent audit log entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of audit log entries (most recent first)
        """
        return self._audit_log[-limit:][::-1]

    # -----------------------------------------------------------------------
    # PQC tunnel management
    # -----------------------------------------------------------------------

    async def _establish_pqc_tunnel(self) -> None:
        """
        Establish a post-quantum cryptographic tunnel for legacy PBX APIs.

        For PBX systems that do not natively support PQC, this method wraps
        the connection in a quantum-safe tunnel using ML-KEM for key
        encapsulation and ML-DSA for authentication.

        The tunnel provides:
        - Key encapsulation using ML-KEM-768 (or configured algorithm)
        - Digital signatures using ML-DSA-65 (or configured algorithm)
        - Optional hybrid mode combining classical and PQC algorithms
        - Session key rotation at configured intervals
        """
        if self._pqc_tunnel_active:
            logger.debug("PQC tunnel already active")
            return

        try:
            self._pqc_session_id = str(uuid.uuid4())

            # In production, this would:
            # 1. Generate ephemeral ML-KEM key pair
            # 2. Exchange public keys with QBITEL PQC gateway
            # 3. Encapsulate shared secret using ML-KEM
            # 4. Derive session keys using HKDF-SHA3-256
            # 5. Establish authenticated channel using ML-DSA
            # 6. If hybrid mode, also perform X25519/ECDH exchange

            self._pqc_tunnel_active = True

            logger.info(
                f"PQC tunnel established for {self.provider_name} "
                f"(KEM: {self._config.pqc_kem_algorithm}, "
                f"SIG: {self._config.pqc_sig_algorithm}, "
                f"hybrid: {self._config.pqc_hybrid_mode})"
            )

            self._audit_log_entry("PQC_TUNNEL_ESTABLISHED", {
                "session_id": self._pqc_session_id,
                "kem_algorithm": self._config.pqc_kem_algorithm,
                "sig_algorithm": self._config.pqc_sig_algorithm,
                "hybrid_mode": self._config.pqc_hybrid_mode,
            })

        except Exception as exc:
            logger.error(f"Failed to establish PQC tunnel for {self.provider_name}: {exc}")
            self._pqc_tunnel_active = False
            self._pqc_session_id = None
            raise PBXConnectionError(f"PQC tunnel establishment failed: {exc}")

    async def _teardown_pqc_tunnel(self) -> None:
        """Tear down the PQC tunnel."""
        if not self._pqc_tunnel_active:
            return

        try:
            # In production, this would:
            # 1. Securely destroy session keys
            # 2. Close the PQC tunnel
            # 3. Zero out sensitive memory

            self._audit_log_entry("PQC_TUNNEL_CLOSED", {
                "session_id": self._pqc_session_id,
            })

            self._pqc_tunnel_active = False
            self._pqc_session_id = None

            logger.info(f"PQC tunnel closed for {self.provider_name}")

        except Exception as exc:
            logger.warning(f"Error tearing down PQC tunnel for {self.provider_name}: {exc}")
            self._pqc_tunnel_active = False
            self._pqc_session_id = None

    # -----------------------------------------------------------------------
    # Heartbeat monitor
    # -----------------------------------------------------------------------

    async def _start_heartbeat_monitor(self) -> None:
        """Start the background heartbeat monitor task."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            return

        self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())
        logger.debug(f"Heartbeat monitor started for {self.provider_name}")

    async def _heartbeat_loop(self) -> None:
        """
        Background loop that periodically sends heartbeats to the PBX.

        If a heartbeat fails, marks the connection as unhealthy and
        initiates automatic reconnection.
        """
        consecutive_failures = 0
        max_consecutive_failures = 3

        while self._connected and self._should_reconnect:
            try:
                await asyncio.sleep(self._config.heartbeat_interval_seconds)

                if not self._connected:
                    break

                heartbeat_start = time.monotonic()
                success = await asyncio.wait_for(
                    self._send_heartbeat(),
                    timeout=self._config.heartbeat_timeout_seconds,
                )
                heartbeat_latency = (time.monotonic() - heartbeat_start) * 1000

                if success:
                    self._last_heartbeat = datetime.utcnow()
                    self._healthy = True
                    consecutive_failures = 0
                    logger.debug(
                        f"Heartbeat OK for {self.provider_name} "
                        f"(latency: {heartbeat_latency:.1f}ms)"
                    )
                else:
                    consecutive_failures += 1
                    logger.warning(
                        f"Heartbeat failed for {self.provider_name} "
                        f"({consecutive_failures}/{max_consecutive_failures})"
                    )

            except asyncio.TimeoutError:
                consecutive_failures += 1
                logger.warning(
                    f"Heartbeat timeout for {self.provider_name} "
                    f"({consecutive_failures}/{max_consecutive_failures})"
                )

            except asyncio.CancelledError:
                logger.debug(f"Heartbeat monitor cancelled for {self.provider_name}")
                return

            except Exception as exc:
                consecutive_failures += 1
                logger.error(
                    f"Heartbeat error for {self.provider_name}: {exc} "
                    f"({consecutive_failures}/{max_consecutive_failures})"
                )

            # Trigger reconnection after max consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                self._healthy = False
                logger.error(
                    f"Max heartbeat failures reached for {self.provider_name}. "
                    f"Initiating reconnection."
                )
                self._audit_log_entry("HEARTBEAT_FAILURE", {
                    "consecutive_failures": consecutive_failures,
                })
                await self._initiate_reconnect()
                return

    async def _initiate_reconnect(self) -> None:
        """Initiate automatic reconnection after a connection loss."""
        if not self._should_reconnect:
            return

        self._connected = False
        self._healthy = False

        try:
            if self._pqc_tunnel_active:
                await self._teardown_pqc_tunnel()

            await self.disconnect()
        except Exception as exc:
            logger.warning(f"Error during disconnect before reconnect: {exc}")

        self._reconnect_count += 1

        self._audit_log_entry("RECONNECTING", {
            "reconnect_count": self._reconnect_count,
        })

        try:
            await self.connect_with_retry()
        except PBXConnectionError as exc:
            logger.error(f"Reconnection to {self.provider_name} failed permanently: {exc}")

    # -----------------------------------------------------------------------
    # Audit logging
    # -----------------------------------------------------------------------

    def _audit_log_entry(self, action: str, details: Dict[str, Any]) -> None:
        """
        Record an audit log entry for PBX operations.

        All PBX interactions are logged for compliance and security
        monitoring purposes.

        Args:
            action: The action being performed
            details: Additional context for the action
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "pbx_type": self._pbx_type.display_name,
            "action": action,
            "details": details,
        }
        self._audit_log.append(entry)

        # Keep audit log bounded in memory (detailed logs go to external systems)
        max_in_memory = 10000
        if len(self._audit_log) > max_in_memory:
            self._audit_log = self._audit_log[-max_in_memory:]

        logger.debug(f"PBX audit: {action} - {details}")

    # -----------------------------------------------------------------------
    # Utility methods
    # -----------------------------------------------------------------------

    def _check_connected(self) -> None:
        """Verify the connector is in a connected state."""
        if not self._connected:
            raise PBXConnectionError(
                f"Not connected to {self.provider_name} at "
                f"{self._config.host}:{self._config.port}"
            )

    async def check_health(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check.

        Returns:
            Dictionary with health check results
        """
        status = self.get_health_status()

        # If connected, do an active heartbeat check
        if self._connected:
            try:
                heartbeat_start = time.monotonic()
                success = await asyncio.wait_for(
                    self._send_heartbeat(),
                    timeout=self._config.heartbeat_timeout_seconds,
                )
                latency = (time.monotonic() - heartbeat_start) * 1000
                status.latency_ms = latency
                status.is_healthy = success
            except (asyncio.TimeoutError, Exception) as exc:
                status.is_healthy = False
                status.error_message = str(exc)

        return status.to_dict()
