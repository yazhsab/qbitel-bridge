"""
Base SIEM Connector

Abstract base class and common types for SIEM integrations.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, Flag, auto
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SIEMCapability(Flag):
    """SIEM connector capabilities."""

    SEND_EVENTS = auto()  # Can send events
    QUERY_EVENTS = auto()  # Can query events
    QUERY_ALERTS = auto()  # Can query alerts
    CREATE_ALERTS = auto()  # Can create alerts
    STREAM_EVENTS = auto()  # Can stream events in real-time
    THREAT_INTEL = auto()  # Can exchange threat intelligence
    CORRELATION_RULES = auto()  # Can manage correlation rules
    CASES = auto()  # Can manage cases/incidents

    # Common combinations
    READ_ONLY = QUERY_EVENTS | QUERY_ALERTS
    WRITE_ONLY = SEND_EVENTS | CREATE_ALERTS
    FULL = SEND_EVENTS | QUERY_EVENTS | QUERY_ALERTS | CREATE_ALERTS | STREAM_EVENTS | THREAT_INTEL | CORRELATION_RULES | CASES


class ConnectionStatus(Enum):
    """Connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class EventSeverity(Enum):
    """Event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SIEMConfig:
    """Base configuration for SIEM connectors."""

    # Connection settings
    host: str = ""
    port: int = 443
    use_ssl: bool = True
    verify_ssl: bool = True

    # Authentication
    auth_type: str = "token"  # token, basic, oauth, api_key
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    api_key: Optional[str] = None

    # OAuth settings
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None

    # Connection behavior
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    connection_pool_size: int = 10

    # Event handling
    batch_size: int = 100
    batch_timeout_ms: int = 1000
    max_queue_size: int = 10000

    # Metadata
    source: str = "qbitel"
    source_type: str = "security:ai"
    index: Optional[str] = None

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.host:
            return False
        if self.auth_type == "token" and not self.token:
            return False
        if self.auth_type == "basic" and (not self.username or not self.password):
            return False
        if self.auth_type == "oauth" and (not self.client_id or not self.client_secret):
            return False
        return True


@dataclass
class SIEMEvent:
    """Represents an event to send to SIEM."""

    # Required fields
    event_type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Classification
    severity: EventSeverity = EventSeverity.INFO
    category: str = ""
    subcategory: str = ""

    # Source information
    source: str = "qbitel"
    source_type: str = "security:ai"
    host: str = ""

    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[str] = None

    # Identifiers
    event_id: str = ""
    correlation_id: str = ""
    parent_id: Optional[str] = None

    # User/entity information
    user: Optional[str] = None
    user_id: Optional[str] = None
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None

    # Asset information
    asset_id: Optional[str] = None
    asset_name: Optional[str] = None

    # Labels and tags
    labels: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category,
            "subcategory": self.subcategory,
            "source": self.source,
            "source_type": self.source_type,
            "host": self.host,
            "data": self.data,
            "raw": self.raw,
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "parent_id": self.parent_id,
            "user": self.user,
            "user_id": self.user_id,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "labels": self.labels,
            "tags": self.tags,
        }

    def to_cef(self) -> str:
        """Convert to Common Event Format (CEF)."""
        severity_map = {
            EventSeverity.DEBUG: 0,
            EventSeverity.INFO: 3,
            EventSeverity.LOW: 4,
            EventSeverity.MEDIUM: 6,
            EventSeverity.HIGH: 8,
            EventSeverity.CRITICAL: 10,
        }

        cef_severity = severity_map.get(self.severity, 5)

        # CEF header
        cef = f"CEF:0|QBITEL|AI-Security|1.0|{self.event_type}|{self.message}|{cef_severity}|"

        # Extension fields
        extensions = []
        if self.src_ip:
            extensions.append(f"src={self.src_ip}")
        if self.dst_ip:
            extensions.append(f"dst={self.dst_ip}")
        if self.user:
            extensions.append(f"suser={self.user}")
        if self.host:
            extensions.append(f"dhost={self.host}")
        if self.correlation_id:
            extensions.append(f"externalId={self.correlation_id}")

        extensions.append(f"rt={int(self.timestamp.timestamp() * 1000)}")

        cef += " ".join(extensions)
        return cef


@dataclass
class SIEMAlert:
    """Represents an alert from SIEM."""

    # Identity
    alert_id: str
    title: str
    description: str = ""

    # Classification
    severity: EventSeverity = EventSeverity.MEDIUM
    confidence: float = 0.0
    status: str = "new"  # new, investigating, resolved, closed

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    triggered_at: Optional[datetime] = None

    # Related data
    events: List[str] = field(default_factory=list)  # Event IDs
    entities: List[Dict[str, Any]] = field(default_factory=list)
    observables: List[Dict[str, Any]] = field(default_factory=list)

    # MITRE ATT&CK mapping
    tactics: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)

    # Detection info
    rule_id: Optional[str] = None
    rule_name: Optional[str] = None
    detection_source: str = ""

    # Response
    assigned_to: Optional[str] = None
    case_id: Optional[str] = None

    # Raw data
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "events": self.events,
            "entities": self.entities,
            "observables": self.observables,
            "tactics": self.tactics,
            "techniques": self.techniques,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "detection_source": self.detection_source,
            "assigned_to": self.assigned_to,
            "case_id": self.case_id,
        }


@dataclass
class SIEMQuery:
    """Query parameters for SIEM searches."""

    query: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Pagination
    limit: int = 100
    offset: int = 0

    # Sorting
    sort_by: str = "timestamp"
    sort_order: str = "desc"

    # Filters
    severity: Optional[List[EventSeverity]] = None
    categories: Optional[List[str]] = None
    sources: Optional[List[str]] = None

    # Output
    fields: Optional[List[str]] = None
    include_raw: bool = False

    # Query type
    query_type: str = "search"  # search, stats, timechart


@dataclass
class QueryResult:
    """Result from a SIEM query."""

    # Results
    events: List[SIEMEvent] = field(default_factory=list)
    alerts: List[SIEMAlert] = field(default_factory=list)
    aggregations: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    total_count: int = 0
    returned_count: int = 0
    query_time_ms: float = 0.0
    is_partial: bool = False

    # Pagination
    next_offset: Optional[int] = None
    has_more: bool = False

    # Raw response
    raw: Optional[Dict[str, Any]] = None


class BaseSIEMConnector(ABC):
    """
    Abstract base class for SIEM connectors.

    Provides common functionality and defines the interface
    that all SIEM connectors must implement.
    """

    def __init__(self, config: SIEMConfig):
        """
        Initialize connector.

        Args:
            config: SIEM configuration
        """
        self.config = config
        self._status = ConnectionStatus.DISCONNECTED
        self._capabilities = SIEMCapability(0)
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self._batch_task: Optional[asyncio.Task] = None
        self._callbacks: Dict[str, List[Callable]] = {
            "connected": [],
            "disconnected": [],
            "error": [],
            "event_sent": [],
            "alert_received": [],
        }

    @property
    def status(self) -> ConnectionStatus:
        """Get connection status."""
        return self._status

    @property
    def capabilities(self) -> SIEMCapability:
        """Get connector capabilities."""
        return self._capabilities

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._status == ConnectionStatus.CONNECTED

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to SIEM.

        Returns:
            True if connected successfully
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from SIEM."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check connection health.

        Returns:
            True if healthy
        """
        pass

    @abstractmethod
    async def send_event(self, event: SIEMEvent) -> bool:
        """
        Send a single event to SIEM.

        Args:
            event: Event to send

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    async def send_events(self, events: List[SIEMEvent]) -> int:
        """
        Send multiple events to SIEM.

        Args:
            events: Events to send

        Returns:
            Number of events sent successfully
        """
        pass

    @abstractmethod
    async def query(self, query: SIEMQuery) -> QueryResult:
        """
        Execute a query against SIEM.

        Args:
            query: Query parameters

        Returns:
            Query results
        """
        pass

    @abstractmethod
    async def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[List[EventSeverity]] = None,
        status: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SIEMAlert]:
        """
        Get alerts from SIEM.

        Args:
            start_time: Start of time range
            end_time: End of time range
            severity: Filter by severity
            status: Filter by status
            limit: Maximum alerts to return

        Returns:
            List of alerts
        """
        pass

    async def stream_events(
        self,
        query: Optional[SIEMQuery] = None,
    ) -> AsyncIterator[SIEMEvent]:
        """
        Stream events in real-time.

        Args:
            query: Optional query to filter events

        Yields:
            Events as they arrive
        """
        raise NotImplementedError("Streaming not supported by this connector")

    async def create_alert(
        self,
        title: str,
        description: str,
        severity: EventSeverity,
        events: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[SIEMAlert]:
        """
        Create an alert in SIEM.

        Args:
            title: Alert title
            description: Alert description
            severity: Alert severity
            events: Related event IDs
            **kwargs: Additional alert properties

        Returns:
            Created alert or None
        """
        raise NotImplementedError("Alert creation not supported by this connector")

    async def update_alert(
        self,
        alert_id: str,
        updates: Dict[str, Any],
    ) -> Optional[SIEMAlert]:
        """
        Update an alert.

        Args:
            alert_id: Alert ID
            updates: Fields to update

        Returns:
            Updated alert or None
        """
        raise NotImplementedError("Alert updates not supported by this connector")

    # Event batching

    async def queue_event(self, event: SIEMEvent) -> bool:
        """
        Queue an event for batched sending.

        Args:
            event: Event to queue

        Returns:
            True if queued successfully
        """
        try:
            self._event_queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event")
            return False

    async def start_batch_sender(self) -> None:
        """Start the batch event sender."""
        if self._batch_task:
            return

        self._batch_task = asyncio.create_task(self._batch_sender_loop())
        logger.info("Batch sender started")

    async def stop_batch_sender(self) -> None:
        """Stop the batch event sender."""
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
            self._batch_task = None

        # Flush remaining events
        await self._flush_queue()
        logger.info("Batch sender stopped")

    async def _batch_sender_loop(self) -> None:
        """Background loop for sending batched events."""
        while True:
            try:
                batch = await self._collect_batch()
                if batch:
                    sent = await self.send_events(batch)
                    logger.debug(f"Sent {sent}/{len(batch)} events")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch sender error: {e}")
                await asyncio.sleep(1)

    async def _collect_batch(self) -> List[SIEMEvent]:
        """Collect events into a batch."""
        batch = []
        deadline = asyncio.get_event_loop().time() + (self.config.batch_timeout_ms / 1000)

        while len(batch) < self.config.batch_size:
            try:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break

                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=remaining,
                )
                batch.append(event)
            except asyncio.TimeoutError:
                break

        return batch

    async def _flush_queue(self) -> None:
        """Flush all events in queue."""
        events = []
        while not self._event_queue.empty():
            try:
                events.append(self._event_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if events:
            await self.send_events(events)

    # Callbacks

    def on(self, event: str, callback: Callable) -> None:
        """Register event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit event to callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(*args, **kwargs))
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    # Retry logic

    async def _with_retry(
        self,
        operation: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute operation with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Operation failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        raise last_error
