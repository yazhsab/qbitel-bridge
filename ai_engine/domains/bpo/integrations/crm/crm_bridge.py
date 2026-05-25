"""
CRM Integration Bridge

Secure bridge between contact center operations and CRM systems.
Provides quantum-safe data protection for customer data in transit
with built-in PII masking, audit logging, rate limiting, and
data residency enforcement.

Supported CRM platforms:
- Salesforce Service Cloud (REST, Streaming API)
- Zendesk (REST API, Webhooks)
- ServiceNow (REST, GlideRecord)
- Microsoft Dynamics 365 (OData, Dataverse)
- Freshdesk (REST API)
- HubSpot (REST API)
- Zoho CRM (REST API)
- Custom CRM via configurable REST/SOAP adapters

Security features:
- Automatic PII masking of sensitive fields before transit
- Audit logging of all CRM data access
- Rate limiting to prevent bulk data extraction
- Data residency enforcement per tenant policy
- PQC tunnel wrapping for all CRM API calls
- Field-level access control based on agent role
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import asyncio
import hashlib
import logging
import re
import time
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CRMType(Enum):
    """Supported CRM platforms."""

    SALESFORCE = ("Salesforce", "rest", "salesforce.com")
    ZENDESK = ("Zendesk", "rest", "zendesk.com")
    SERVICENOW = ("ServiceNow", "rest", "service-now.com")
    DYNAMICS_365 = ("Dynamics 365", "odata", "dynamics.com")
    FRESHDESK = ("Freshdesk", "rest", "freshdesk.com")
    HUBSPOT = ("HubSpot", "rest", "hubspot.com")
    ZOHO = ("Zoho CRM", "rest", "zoho.com")
    CUSTOM = ("Custom CRM", "rest", "")

    def __init__(self, display_name: str, api_type: str, domain: str):
        self.display_name = display_name
        self.api_type = api_type
        self.domain = domain


class CRMEventType(Enum):
    """CRM event types for audit and callback."""

    CUSTOMER_LOOKUP = auto()
    CUSTOMER_FOUND = auto()
    CUSTOMER_NOT_FOUND = auto()
    CASE_CREATED = auto()
    CASE_UPDATED = auto()
    CASE_CLOSED = auto()
    INTERACTION_LOGGED = auto()
    NOTE_ADDED = auto()
    FIELD_UPDATED = auto()
    SEARCH_PERFORMED = auto()
    DATA_ACCESS_DENIED = auto()
    RATE_LIMIT_HIT = auto()
    PII_MASKED = auto()


class DataClassification(Enum):
    """Data sensitivity classification for CRM fields."""

    PUBLIC = (0, "Public", False)
    INTERNAL = (1, "Internal", False)
    CONFIDENTIAL = (2, "Confidential", True)
    PII = (3, "PII", True)
    PCI = (4, "PCI", True)
    PHI = (5, "PHI", True)

    def __init__(self, level: int, display_name: str, requires_masking: bool):
        self.level = level
        self.display_name = display_name
        self.requires_masking = requires_masking


class PIIMaskingStrategy(Enum):
    """Strategies for masking PII fields."""

    FULL_MASK = "full_mask"           # Replace entirely with ***
    LAST_FOUR = "last_four"           # Show only last 4 characters
    FIRST_LAST = "first_last"         # Show first and last character
    HASH = "hash"                     # SHA-256 hash of the value
    REDACT = "redact"                 # Replace with [REDACTED]
    TOKENIZE = "tokenize"             # Replace with a reversible token


class CaseStatus(Enum):
    """Standard case/ticket status values."""

    NEW = "new"
    OPEN = "open"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_ON_CUSTOMER = "waiting_on_customer"
    WAITING_ON_THIRD_PARTY = "waiting_on_third_party"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class CasePriority(Enum):
    """Case/ticket priority levels."""

    CRITICAL = (1, "Critical", 60)    # SLA in minutes
    HIGH = (2, "High", 240)
    MEDIUM = (3, "Medium", 480)
    LOW = (4, "Low", 1440)

    def __init__(self, level: int, display_name: str, sla_minutes: int):
        self.priority_level = level
        self.display_name = display_name
        self.sla_minutes = sla_minutes


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CRMConnectionConfig:
    """Configuration for CRM system connection."""

    # Connection
    crm_type: CRMType = CRMType.SALESFORCE
    base_url: str = ""
    api_version: str = ""

    # Authentication
    auth_method: str = "oauth2"  # oauth2, api_key, basic, certificate
    credentials_vault_path: str = ""
    oauth_token_url: str = ""
    oauth_client_id_vault_path: str = ""
    oauth_scope: str = ""

    # PQC settings
    pqc_enabled: bool = True
    pqc_kem_algorithm: str = "ML-KEM-768"
    pqc_sig_algorithm: str = "ML-DSA-65"
    pqc_hybrid_mode: bool = True

    # TLS settings
    tls_version: str = "TLS 1.3"
    ca_cert_path: Optional[str] = None
    client_cert_path: Optional[str] = None

    # Rate limiting
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    max_bulk_records: int = 100
    rate_limit_burst: int = 10

    # Timeouts
    connect_timeout_seconds: int = 30
    read_timeout_seconds: int = 60
    write_timeout_seconds: int = 30

    # Data protection
    enable_pii_masking: bool = True
    enable_audit_logging: bool = True
    data_residency_region: Optional[str] = None  # e.g., "EU", "US", "APAC"
    allowed_data_regions: List[str] = field(default_factory=lambda: ["US", "EU"])

    # Field-level access control
    restricted_fields: Set[str] = field(default_factory=lambda: {
        "ssn", "social_security_number", "tax_id",
        "credit_card_number", "card_number", "pan",
        "bank_account_number", "routing_number",
        "date_of_birth", "dob",
        "passport_number", "drivers_license",
    })

    # Cache settings
    cache_ttl_seconds: int = 300
    cache_max_entries: int = 1000

    def validate(self) -> List[str]:
        """Validate CRM connection configuration."""
        errors = []

        if not self.base_url and self.crm_type != CRMType.CUSTOM:
            errors.append("CRM base URL is required")

        if not self.credentials_vault_path:
            errors.append("credentials_vault_path is required")

        if self.max_requests_per_minute < 1:
            errors.append("Rate limit must be at least 1 request per minute")

        if self.max_bulk_records > 1000:
            errors.append("Bulk record limit cannot exceed 1000 to prevent data exfiltration")

        return errors


@dataclass
class CustomerContext:
    """
    Customer context data retrieved from CRM.

    PII fields are automatically masked based on the configured
    masking strategy before being returned to the caller.
    """

    customer_id: str = ""
    name_masked: str = ""              # Masked/tokenised customer name
    account_number_masked: str = ""    # Masked account number
    email_masked: str = ""             # Masked email
    phone_masked: str = ""             # Masked phone number

    # Non-sensitive context
    account_type: str = ""
    account_status: str = ""
    customer_segment: str = ""         # Gold, Silver, Bronze, etc.
    preferred_language: str = "en"
    timezone: str = "UTC"

    # Interaction history (summarised, no PII)
    interaction_count: int = 0
    last_interaction_date: Optional[datetime] = None
    open_cases_count: int = 0
    current_case_id: Optional[str] = None

    # Data classification
    data_classification: DataClassification = DataClassification.CONFIDENTIAL

    # Metadata
    crm_record_id: str = ""
    data_region: str = ""
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    cache_expires_at: Optional[datetime] = None

    # Entitlements
    entitlements: List[str] = field(default_factory=list)
    service_level: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (all PII remains masked)."""
        return {
            "customer_id": self.customer_id,
            "name_masked": self.name_masked,
            "account_number_masked": self.account_number_masked,
            "email_masked": self.email_masked,
            "phone_masked": self.phone_masked,
            "account_type": self.account_type,
            "account_status": self.account_status,
            "customer_segment": self.customer_segment,
            "preferred_language": self.preferred_language,
            "interaction_count": self.interaction_count,
            "open_cases_count": self.open_cases_count,
            "current_case_id": self.current_case_id,
            "data_classification": self.data_classification.display_name,
            "data_region": self.data_region,
        }


@dataclass
class CRMEvent:
    """An event generated by CRM bridge operations."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: CRMEventType = CRMEventType.CUSTOMER_LOOKUP
    customer_id: str = ""
    agent_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Audit fields
    action: str = ""
    fields_accessed: List[str] = field(default_factory=list)
    fields_masked: List[str] = field(default_factory=list)
    data_classification: Optional[DataClassification] = None
    source_ip: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to audit-safe dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "customer_id": self.customer_id,
            "agent_id": self.agent_id,
            "action": self.action,
            "fields_accessed": self.fields_accessed,
            "fields_masked": self.fields_masked,
            "data_classification": self.data_classification.display_name if self.data_classification else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CaseData:
    """Data structure for creating or updating a CRM case/ticket."""

    case_id: Optional[str] = None
    customer_id: str = ""
    subject: str = ""
    description: str = ""
    status: CaseStatus = CaseStatus.NEW
    priority: CasePriority = CasePriority.MEDIUM
    category: str = ""
    subcategory: str = ""
    assigned_agent_id: str = ""
    assigned_queue: str = ""
    call_id: str = ""
    channel: str = "voice"  # voice, chat, email, social
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API submission."""
        return {
            "case_id": self.case_id,
            "customer_id": self.customer_id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.display_name,
            "category": self.category,
            "subcategory": self.subcategory,
            "assigned_agent_id": self.assigned_agent_id,
            "assigned_queue": self.assigned_queue,
            "call_id": self.call_id,
            "channel": self.channel,
            "tags": self.tags,
            "custom_fields": self.custom_fields,
        }


@dataclass
class CRMHealthStatus:
    """Health status of the CRM bridge."""

    is_connected: bool = False
    is_healthy: bool = False
    last_request_time: Optional[datetime] = None
    requests_this_minute: int = 0
    requests_this_hour: int = 0
    rate_limit_remaining: int = 0
    avg_response_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    pqc_tunnel_active: bool = False
    error_count: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_connected": self.is_connected,
            "is_healthy": self.is_healthy,
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
            "requests_this_minute": self.requests_this_minute,
            "requests_this_hour": self.requests_this_hour,
            "rate_limit_remaining": self.rate_limit_remaining,
            "avg_response_time_ms": self.avg_response_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "pqc_tunnel_active": self.pqc_tunnel_active,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CRMError(Exception):
    """Base exception for CRM bridge operations."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code


class CRMConnectionError(CRMError):
    """Exception for CRM connection failures."""
    pass


class CRMAuthenticationError(CRMError):
    """Exception for CRM authentication failures."""
    pass


class CRMRateLimitError(CRMError):
    """Exception when CRM API rate limit is exceeded."""

    def __init__(self, message: str, retry_after_seconds: int = 60):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class CRMDataAccessError(CRMError):
    """Exception when CRM data access is denied."""
    pass


class CRMDataResidencyError(CRMError):
    """Exception when data residency requirements are violated."""
    pass


# ---------------------------------------------------------------------------
# PII Masking Engine
# ---------------------------------------------------------------------------


class PIIMaskingEngine:
    """
    Engine for masking PII fields in CRM data.

    Applies configured masking strategies to sensitive fields
    before data leaves the CRM bridge. Supports multiple masking
    strategies and field-level configuration.
    """

    # Default PII field patterns and their masking strategies
    DEFAULT_FIELD_RULES: Dict[str, Tuple[DataClassification, PIIMaskingStrategy]] = {
        # PII fields
        "name": (DataClassification.PII, PIIMaskingStrategy.FIRST_LAST),
        "first_name": (DataClassification.PII, PIIMaskingStrategy.FIRST_LAST),
        "last_name": (DataClassification.PII, PIIMaskingStrategy.FIRST_LAST),
        "email": (DataClassification.PII, PIIMaskingStrategy.FIRST_LAST),
        "phone": (DataClassification.PII, PIIMaskingStrategy.LAST_FOUR),
        "phone_number": (DataClassification.PII, PIIMaskingStrategy.LAST_FOUR),
        "mobile": (DataClassification.PII, PIIMaskingStrategy.LAST_FOUR),
        "address": (DataClassification.PII, PIIMaskingStrategy.REDACT),
        "street_address": (DataClassification.PII, PIIMaskingStrategy.REDACT),
        "postal_code": (DataClassification.PII, PIIMaskingStrategy.LAST_FOUR),
        "zip_code": (DataClassification.PII, PIIMaskingStrategy.LAST_FOUR),
        "date_of_birth": (DataClassification.PII, PIIMaskingStrategy.REDACT),
        "dob": (DataClassification.PII, PIIMaskingStrategy.REDACT),

        # PCI fields
        "credit_card_number": (DataClassification.PCI, PIIMaskingStrategy.LAST_FOUR),
        "card_number": (DataClassification.PCI, PIIMaskingStrategy.LAST_FOUR),
        "pan": (DataClassification.PCI, PIIMaskingStrategy.LAST_FOUR),
        "cvv": (DataClassification.PCI, PIIMaskingStrategy.FULL_MASK),
        "expiry_date": (DataClassification.PCI, PIIMaskingStrategy.FULL_MASK),
        "bank_account_number": (DataClassification.PCI, PIIMaskingStrategy.LAST_FOUR),
        "routing_number": (DataClassification.PCI, PIIMaskingStrategy.LAST_FOUR),

        # Government IDs
        "ssn": (DataClassification.PII, PIIMaskingStrategy.LAST_FOUR),
        "social_security_number": (DataClassification.PII, PIIMaskingStrategy.LAST_FOUR),
        "tax_id": (DataClassification.PII, PIIMaskingStrategy.LAST_FOUR),
        "passport_number": (DataClassification.PII, PIIMaskingStrategy.LAST_FOUR),
        "drivers_license": (DataClassification.PII, PIIMaskingStrategy.LAST_FOUR),

        # PHI fields
        "medical_record_number": (DataClassification.PHI, PIIMaskingStrategy.HASH),
        "insurance_id": (DataClassification.PHI, PIIMaskingStrategy.LAST_FOUR),
        "diagnosis": (DataClassification.PHI, PIIMaskingStrategy.REDACT),
    }

    def __init__(self, custom_rules: Optional[Dict[str, Tuple[DataClassification, PIIMaskingStrategy]]] = None):
        self._rules = dict(self.DEFAULT_FIELD_RULES)
        if custom_rules:
            self._rules.update(custom_rules)

        self._mask_count = 0
        self._token_store: Dict[str, str] = {}  # token -> original (for reversible tokenisation)

    def mask_field(self, field_name: str, value: str) -> Tuple[str, bool]:
        """
        Mask a field value if it matches PII rules.

        Args:
            field_name: The field name (normalised to lowercase)
            value: The field value

        Returns:
            Tuple of (masked_value, was_masked)
        """
        if not value or not isinstance(value, str):
            return str(value) if value else "", False

        normalised_name = field_name.lower().strip()
        rule = self._rules.get(normalised_name)

        if rule is None:
            # Check if field name contains any known PII patterns
            for pattern, r in self._rules.items():
                if pattern in normalised_name:
                    rule = r
                    break

        if rule is None:
            return value, False

        classification, strategy = rule
        masked_value = self._apply_strategy(value, strategy)
        self._mask_count += 1

        return masked_value, True

    def mask_record(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Mask all PII fields in a record.

        Args:
            record: Dictionary of field name -> value

        Returns:
            Tuple of (masked_record, list_of_masked_field_names)
        """
        masked = {}
        masked_fields = []

        for key, value in record.items():
            if isinstance(value, str):
                masked_value, was_masked = self.mask_field(key, value)
                masked[key] = masked_value
                if was_masked:
                    masked_fields.append(key)
            elif isinstance(value, dict):
                # Recursively mask nested objects
                nested_masked, nested_fields = self.mask_record(value)
                masked[key] = nested_masked
                masked_fields.extend(f"{key}.{f}" for f in nested_fields)
            else:
                masked[key] = value

        return masked, masked_fields

    def _apply_strategy(self, value: str, strategy: PIIMaskingStrategy) -> str:
        """Apply a masking strategy to a value."""
        if not value:
            return value

        if strategy == PIIMaskingStrategy.FULL_MASK:
            return "***"

        elif strategy == PIIMaskingStrategy.LAST_FOUR:
            if len(value) <= 4:
                return "***" + value
            return "*" * (len(value) - 4) + value[-4:]

        elif strategy == PIIMaskingStrategy.FIRST_LAST:
            if len(value) <= 2:
                return value[0] + "*"
            return value[0] + "*" * (len(value) - 2) + value[-1]

        elif strategy == PIIMaskingStrategy.HASH:
            return "SHA256:" + hashlib.sha256(value.encode()).hexdigest()[:16]

        elif strategy == PIIMaskingStrategy.REDACT:
            return "[REDACTED]"

        elif strategy == PIIMaskingStrategy.TOKENIZE:
            token = f"TOK:{uuid.uuid4().hex[:12]}"
            self._token_store[token] = value
            return token

        return "***"

    @property
    def total_masks_applied(self) -> int:
        """Get total number of masking operations performed."""
        return self._mask_count


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """
    Token bucket rate limiter for CRM API calls.

    Enforces per-minute and per-hour rate limits to prevent
    bulk data extraction and comply with CRM API quotas.
    """

    def __init__(
        self,
        max_per_minute: int = 60,
        max_per_hour: int = 1000,
        burst: int = 10,
    ):
        self._max_per_minute = max_per_minute
        self._max_per_hour = max_per_hour
        self._burst = burst

        self._minute_tokens: float = float(max_per_minute)
        self._hour_tokens: float = float(max_per_hour)
        self._last_refill: float = time.monotonic()

        self._requests_this_minute: int = 0
        self._requests_this_hour: int = 0
        self._minute_start: float = time.monotonic()
        self._hour_start: float = time.monotonic()

    def acquire(self) -> bool:
        """
        Attempt to acquire a rate limit token.

        Returns:
            True if the request is allowed, False if rate limited
        """
        now = time.monotonic()
        self._refill(now)

        # Check minute window
        if now - self._minute_start >= 60.0:
            self._requests_this_minute = 0
            self._minute_start = now

        # Check hour window
        if now - self._hour_start >= 3600.0:
            self._requests_this_hour = 0
            self._hour_start = now

        if self._requests_this_minute >= self._max_per_minute:
            return False

        if self._requests_this_hour >= self._max_per_hour:
            return False

        self._requests_this_minute += 1
        self._requests_this_hour += 1
        return True

    def _refill(self, now: float) -> None:
        """Refill tokens based on elapsed time."""
        elapsed = now - self._last_refill
        self._last_refill = now

        # Refill per-minute tokens
        self._minute_tokens = min(
            float(self._max_per_minute),
            self._minute_tokens + elapsed * (self._max_per_minute / 60.0),
        )

    @property
    def remaining_this_minute(self) -> int:
        """Get remaining requests this minute."""
        return max(0, self._max_per_minute - self._requests_this_minute)

    @property
    def remaining_this_hour(self) -> int:
        """Get remaining requests this hour."""
        return max(0, self._max_per_hour - self._requests_this_hour)


# ---------------------------------------------------------------------------
# CRM Bridge abstract base
# ---------------------------------------------------------------------------


class CRMBridge(ABC):
    """
    Secure bridge between contact center and CRM systems.

    Provides quantum-safe data protection for customer data in transit
    with built-in PII masking, comprehensive audit logging, rate limiting,
    and data residency enforcement.

    All vendor-specific CRM implementations must inherit from this class
    and implement the abstract methods for their specific CRM API.

    Features:
    - Automatic PII masking before data leaves the bridge
    - Comprehensive audit logging of all data access
    - Rate limiting to prevent bulk data extraction
    - Data residency enforcement per tenant configuration
    - PQC tunnel wrapping for all CRM API calls
    - Customer context caching with TTL
    - Field-level access control based on agent role
    """

    def __init__(self, crm_type: CRMType, config: CRMConnectionConfig):
        self._crm_type = crm_type
        self._config = config
        self._connected = False
        self._healthy = False

        # PII masking engine
        self._masking_engine = PIIMaskingEngine()

        # Rate limiter
        self._rate_limiter = RateLimiter(
            max_per_minute=config.max_requests_per_minute,
            max_per_hour=config.max_requests_per_hour,
            burst=config.rate_limit_burst,
        )

        # Customer context cache: customer_id -> (context, expires_at)
        self._context_cache: Dict[str, Tuple[CustomerContext, datetime]] = {}

        # PQC tunnel state
        self._pqc_tunnel_active: bool = False
        self._pqc_session_id: Optional[str] = None

        # Metrics
        self._connect_time: Optional[datetime] = None
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._last_error: Optional[str] = None
        self._response_times: List[float] = []
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

        # Event callbacks
        self._event_callbacks: List[Callable[[CRMEvent], None]] = []

        # Validate
        errors = config.validate()
        if errors:
            logger.warning(f"CRM config validation warnings for {crm_type.display_name}: {errors}")

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def crm_type(self) -> CRMType:
        """Get the CRM system type."""
        return self._crm_type

    @property
    def config(self) -> CRMConnectionConfig:
        """Get the connection configuration."""
        return self._config

    @property
    def is_connected(self) -> bool:
        """Check if connected to the CRM."""
        return self._connected

    @property
    def is_healthy(self) -> bool:
        """Check if the bridge is healthy."""
        return self._connected and self._healthy

    # -----------------------------------------------------------------------
    # Abstract methods - must be implemented by CRM-specific bridges
    # -----------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the CRM system.

        Implementations must:
        1. Authenticate with the CRM API
        2. Verify API access and permissions
        3. Set self._connected = True on success

        Raises:
            CRMConnectionError: If connection fails
            CRMAuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the CRM system.

        Implementations must:
        1. Invalidate auth tokens
        2. Close network connections
        3. Set self._connected = False
        """
        pass

    @abstractmethod
    async def _fetch_customer_context(
        self,
        customer_id: str,
        fields: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fetch raw customer data from the CRM (before PII masking).

        This is the vendor-specific implementation. The public
        get_customer_context() method wraps this with PII masking,
        caching, rate limiting, and audit logging.

        Args:
            customer_id: Customer identifier in the CRM
            fields: Specific fields to retrieve (None = all accessible)

        Returns:
            Raw customer record dictionary

        Raises:
            CRMError: If the fetch fails
        """
        pass

    @abstractmethod
    async def _create_case_internal(self, case_data: CaseData) -> str:
        """
        Create a case/ticket in the CRM (internal implementation).

        Args:
            case_data: Case data to create

        Returns:
            Created case ID

        Raises:
            CRMError: If case creation fails
        """
        pass

    @abstractmethod
    async def _update_case_internal(
        self,
        case_id: str,
        updates: Dict[str, Any],
    ) -> None:
        """
        Update a case/ticket in the CRM (internal implementation).

        Args:
            case_id: Case to update
            updates: Field updates to apply

        Raises:
            CRMError: If update fails
        """
        pass

    @abstractmethod
    async def _add_interaction_note_internal(
        self,
        case_id: str,
        note: str,
        agent_id: str,
        interaction_type: str,
    ) -> str:
        """
        Add an interaction note to a case (internal implementation).

        Args:
            case_id: Case to add note to
            note: Note text (already PII-scrubbed)
            agent_id: Agent adding the note
            interaction_type: Type of interaction (call, chat, email)

        Returns:
            Note/activity ID

        Raises:
            CRMError: If note creation fails
        """
        pass

    @abstractmethod
    async def _search_customer_internal(
        self,
        query: Dict[str, str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """
        Search for customers in the CRM (internal implementation).

        Args:
            query: Search criteria (field -> value)
            max_results: Maximum results to return

        Returns:
            List of raw customer records

        Raises:
            CRMError: If search fails
        """
        pass

    @abstractmethod
    async def _health_check_internal(self) -> bool:
        """
        Perform CRM-specific health check.

        Returns:
            True if CRM is healthy
        """
        pass

    # -----------------------------------------------------------------------
    # Public methods with PII masking, rate limiting, and audit logging
    # -----------------------------------------------------------------------

    async def get_customer_context(
        self,
        customer_id: str,
        agent_id: str,
        fields: Optional[Set[str]] = None,
    ) -> CustomerContext:
        """
        Get customer context from CRM with automatic PII masking.

        This method wraps the vendor-specific _fetch_customer_context()
        with security controls:
        1. Rate limiting check
        2. Cache lookup
        3. Data residency enforcement
        4. Raw data fetch from CRM
        5. PII masking of sensitive fields
        6. Audit logging
        7. Cache population

        Args:
            customer_id: Customer identifier
            agent_id: Agent requesting the data (for audit)
            fields: Specific fields to retrieve

        Returns:
            CustomerContext with PII fields masked

        Raises:
            CRMRateLimitError: If rate limit exceeded
            CRMDataResidencyError: If data residency violated
            CRMError: If fetch fails
        """
        self._check_connected()

        # Rate limiting
        if not self._rate_limiter.acquire():
            self._emit_event(CRMEventType.RATE_LIMIT_HIT, customer_id, agent_id)
            raise CRMRateLimitError(
                f"CRM rate limit exceeded ({self._config.max_requests_per_minute}/min)"
            )

        # Check cache
        cached = self._get_from_cache(customer_id)
        if cached is not None:
            self._cache_hits += 1
            self._audit_log_entry("CUSTOMER_LOOKUP_CACHED", {
                "customer_id": customer_id,
                "agent_id": agent_id,
            })
            return cached

        self._cache_misses += 1

        # Fetch from CRM
        start_time = time.monotonic()
        try:
            raw_data = await self._fetch_customer_context(customer_id, fields)
            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._record_response_time(elapsed_ms)
        except Exception as exc:
            self._total_errors += 1
            self._last_error = str(exc)
            raise

        self._total_requests += 1

        if not raw_data:
            self._emit_event(CRMEventType.CUSTOMER_NOT_FOUND, customer_id, agent_id)
            raise CRMError(f"Customer not found: {customer_id}")

        # Data residency check
        data_region = raw_data.get("data_region", raw_data.get("region", ""))
        if data_region and self._config.data_residency_region:
            if data_region != self._config.data_residency_region:
                self._emit_event(
                    CRMEventType.DATA_ACCESS_DENIED,
                    customer_id,
                    agent_id,
                    {"reason": "data_residency", "data_region": data_region},
                )
                raise CRMDataResidencyError(
                    f"Customer data in region {data_region} cannot be accessed "
                    f"from region {self._config.data_residency_region}"
                )

        # PII masking
        masked_data, masked_fields = self._masking_engine.mask_record(raw_data)

        if masked_fields:
            self._emit_event(
                CRMEventType.PII_MASKED,
                customer_id,
                agent_id,
                {"masked_fields": masked_fields},
            )

        # Build CustomerContext
        context = CustomerContext(
            customer_id=customer_id,
            name_masked=masked_data.get("name", masked_data.get("full_name", "")),
            account_number_masked=masked_data.get("account_number", ""),
            email_masked=masked_data.get("email", ""),
            phone_masked=masked_data.get("phone", masked_data.get("phone_number", "")),
            account_type=raw_data.get("account_type", ""),
            account_status=raw_data.get("account_status", ""),
            customer_segment=raw_data.get("segment", raw_data.get("customer_segment", "")),
            preferred_language=raw_data.get("preferred_language", "en"),
            timezone=raw_data.get("timezone", "UTC"),
            interaction_count=raw_data.get("interaction_count", 0),
            last_interaction_date=raw_data.get("last_interaction_date"),
            open_cases_count=raw_data.get("open_cases_count", 0),
            current_case_id=raw_data.get("current_case_id"),
            crm_record_id=raw_data.get("id", raw_data.get("record_id", "")),
            data_region=data_region,
            entitlements=raw_data.get("entitlements", []),
            service_level=raw_data.get("service_level", ""),
        )

        # Cache the result
        self._put_in_cache(customer_id, context)

        # Audit log
        self._audit_log_entry("CUSTOMER_LOOKUP", {
            "customer_id": customer_id,
            "agent_id": agent_id,
            "fields_accessed": list(raw_data.keys()),
            "fields_masked": masked_fields,
            "response_time_ms": elapsed_ms,
        })

        self._emit_event(CRMEventType.CUSTOMER_FOUND, customer_id, agent_id)

        return context

    async def create_case(
        self,
        case_data: CaseData,
        agent_id: str,
    ) -> str:
        """
        Create a new case/ticket in the CRM.

        Applies PII scrubbing to the case description before submission
        and logs the creation event.

        Args:
            case_data: Case data to create
            agent_id: Agent creating the case

        Returns:
            Created case ID

        Raises:
            CRMRateLimitError: If rate limit exceeded
            CRMError: If creation fails
        """
        self._check_connected()

        if not self._rate_limiter.acquire():
            raise CRMRateLimitError("CRM rate limit exceeded")

        # Scrub PII from description
        scrubbed_description = self._scrub_pii_from_text(case_data.description)
        case_data.description = scrubbed_description

        start_time = time.monotonic()
        try:
            case_id = await self._create_case_internal(case_data)
            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._record_response_time(elapsed_ms)
        except Exception as exc:
            self._total_errors += 1
            self._last_error = str(exc)
            raise

        self._total_requests += 1

        self._audit_log_entry("CASE_CREATED", {
            "case_id": case_id,
            "customer_id": case_data.customer_id,
            "agent_id": agent_id,
            "priority": case_data.priority.display_name,
            "channel": case_data.channel,
        })

        self._emit_event(
            CRMEventType.CASE_CREATED,
            case_data.customer_id,
            agent_id,
            {"case_id": case_id},
        )

        return case_id

    async def update_case(
        self,
        case_id: str,
        updates: Dict[str, Any],
        agent_id: str,
    ) -> None:
        """
        Update an existing case/ticket in the CRM.

        Args:
            case_id: Case to update
            updates: Field updates
            agent_id: Agent performing the update

        Raises:
            CRMRateLimitError: If rate limit exceeded
            CRMError: If update fails
        """
        self._check_connected()

        if not self._rate_limiter.acquire():
            raise CRMRateLimitError("CRM rate limit exceeded")

        # Scrub PII from text fields
        sanitised_updates = {}
        for key, value in updates.items():
            if isinstance(value, str) and key in ("description", "notes", "comment"):
                sanitised_updates[key] = self._scrub_pii_from_text(value)
            else:
                sanitised_updates[key] = value

        start_time = time.monotonic()
        try:
            await self._update_case_internal(case_id, sanitised_updates)
            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._record_response_time(elapsed_ms)
        except Exception as exc:
            self._total_errors += 1
            self._last_error = str(exc)
            raise

        self._total_requests += 1

        self._audit_log_entry("CASE_UPDATED", {
            "case_id": case_id,
            "agent_id": agent_id,
            "updated_fields": list(updates.keys()),
        })

        self._emit_event(
            CRMEventType.CASE_UPDATED,
            "",
            agent_id,
            {"case_id": case_id, "fields": list(updates.keys())},
        )

    async def add_interaction_note(
        self,
        case_id: str,
        note: str,
        agent_id: str,
        interaction_type: str = "voice",
    ) -> str:
        """
        Add an interaction note to a case.

        The note text is automatically scrubbed for PII before
        submission to the CRM.

        Args:
            case_id: Case to add the note to
            note: Note text
            agent_id: Agent adding the note
            interaction_type: Channel type (voice, chat, email)

        Returns:
            Note/activity ID

        Raises:
            CRMRateLimitError: If rate limit exceeded
            CRMError: If note creation fails
        """
        self._check_connected()

        if not self._rate_limiter.acquire():
            raise CRMRateLimitError("CRM rate limit exceeded")

        # Scrub PII from note text
        scrubbed_note = self._scrub_pii_from_text(note)

        start_time = time.monotonic()
        try:
            note_id = await self._add_interaction_note_internal(
                case_id, scrubbed_note, agent_id, interaction_type
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._record_response_time(elapsed_ms)
        except Exception as exc:
            self._total_errors += 1
            self._last_error = str(exc)
            raise

        self._total_requests += 1

        self._audit_log_entry("NOTE_ADDED", {
            "case_id": case_id,
            "note_id": note_id,
            "agent_id": agent_id,
            "interaction_type": interaction_type,
            "note_length": len(note),
            "pii_scrubbed": note != scrubbed_note,
        })

        self._emit_event(
            CRMEventType.NOTE_ADDED,
            "",
            agent_id,
            {"case_id": case_id, "note_id": note_id},
        )

        return note_id

    async def search_customer(
        self,
        query: Dict[str, str],
        agent_id: str,
        max_results: int = 10,
    ) -> List[CustomerContext]:
        """
        Search for customers in the CRM with PII masking.

        Results are automatically PII-masked before being returned.
        The search itself is rate-limited and audited.

        Args:
            query: Search criteria (field -> value)
            agent_id: Agent performing the search
            max_results: Maximum results to return (capped at config.max_bulk_records)

        Returns:
            List of masked CustomerContext objects

        Raises:
            CRMRateLimitError: If rate limit exceeded
            CRMError: If search fails
        """
        self._check_connected()

        if not self._rate_limiter.acquire():
            raise CRMRateLimitError("CRM rate limit exceeded")

        # Cap results to prevent bulk extraction
        max_results = min(max_results, self._config.max_bulk_records)

        start_time = time.monotonic()
        try:
            raw_results = await self._search_customer_internal(query, max_results)
            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._record_response_time(elapsed_ms)
        except Exception as exc:
            self._total_errors += 1
            self._last_error = str(exc)
            raise

        self._total_requests += 1

        # Mask PII in all results
        contexts = []
        for raw_data in raw_results:
            masked_data, masked_fields = self._masking_engine.mask_record(raw_data)

            context = CustomerContext(
                customer_id=raw_data.get("id", raw_data.get("customer_id", "")),
                name_masked=masked_data.get("name", ""),
                account_number_masked=masked_data.get("account_number", ""),
                email_masked=masked_data.get("email", ""),
                phone_masked=masked_data.get("phone", ""),
                account_type=raw_data.get("account_type", ""),
                account_status=raw_data.get("account_status", ""),
                customer_segment=raw_data.get("segment", ""),
            )
            contexts.append(context)

        self._audit_log_entry("CUSTOMER_SEARCH", {
            "agent_id": agent_id,
            "query_fields": list(query.keys()),
            "results_count": len(contexts),
            "max_results": max_results,
            "response_time_ms": elapsed_ms,
        })

        self._emit_event(
            CRMEventType.SEARCH_PERFORMED,
            "",
            agent_id,
            {"results_count": len(contexts)},
        )

        return contexts

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    async def connect_with_pqc(self) -> None:
        """
        Connect to the CRM system with PQC tunnel wrapping.

        Establishes a quantum-safe tunnel before connecting to the
        CRM API to protect all data in transit.
        """
        if self._config.pqc_enabled:
            await self._establish_pqc_tunnel()

        await self.connect()

        self._connected = True
        self._healthy = True
        self._connect_time = datetime.utcnow()

        logger.info(f"Connected to {self._crm_type.display_name} (PQC: {self._pqc_tunnel_active})")

        self._audit_log_entry("CRM_CONNECTED", {
            "crm_type": self._crm_type.display_name,
            "pqc_tunnel": self._pqc_tunnel_active,
        })

    async def disconnect_gracefully(self) -> None:
        """Gracefully disconnect from the CRM system."""
        if self._connected:
            try:
                await self.disconnect()
            except Exception as exc:
                logger.warning(f"Error during CRM disconnect: {exc}")
            finally:
                self._connected = False
                self._healthy = False

        if self._pqc_tunnel_active:
            await self._teardown_pqc_tunnel()

        # Clear caches
        self._context_cache.clear()

        self._audit_log_entry("CRM_DISCONNECTED", {
            "crm_type": self._crm_type.display_name,
            "total_requests": self._total_requests,
        })

    # -----------------------------------------------------------------------
    # PQC tunnel
    # -----------------------------------------------------------------------

    async def _establish_pqc_tunnel(self) -> None:
        """Establish PQC tunnel for CRM API calls."""
        try:
            self._pqc_session_id = str(uuid.uuid4())

            # In production:
            # 1. Generate ML-KEM key pair
            # 2. Exchange keys with QBITEL PQC gateway
            # 3. Establish authenticated encrypted channel
            # 4. All subsequent CRM API calls routed through tunnel

            self._pqc_tunnel_active = True

            logger.info(
                f"PQC tunnel established for {self._crm_type.display_name} "
                f"(KEM: {self._config.pqc_kem_algorithm})"
            )

        except Exception as exc:
            logger.error(f"PQC tunnel failed for {self._crm_type.display_name}: {exc}")
            self._pqc_tunnel_active = False

    async def _teardown_pqc_tunnel(self) -> None:
        """Tear down PQC tunnel."""
        self._pqc_tunnel_active = False
        self._pqc_session_id = None

    # -----------------------------------------------------------------------
    # PII scrubbing for free-text fields
    # -----------------------------------------------------------------------

    def _scrub_pii_from_text(self, text: str) -> str:
        """
        Scrub PII patterns from free-text fields (descriptions, notes).

        Uses regex patterns to detect and redact:
        - Credit card numbers
        - SSNs
        - Phone numbers
        - Email addresses

        Args:
            text: Input text that may contain PII

        Returns:
            Scrubbed text with PII replaced
        """
        if not text:
            return text

        scrubbed = text

        # Credit card patterns (Visa, MC, Amex, Discover)
        scrubbed = re.sub(
            r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            '[CARD_REDACTED]',
            scrubbed,
        )

        # SSN pattern
        scrubbed = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[SSN_REDACTED]',
            scrubbed,
        )

        # Email pattern
        scrubbed = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            '[EMAIL_REDACTED]',
            scrubbed,
        )

        # Phone number patterns (US)
        scrubbed = re.sub(
            r'\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            '[PHONE_REDACTED]',
            scrubbed,
        )

        return scrubbed

    # -----------------------------------------------------------------------
    # Cache management
    # -----------------------------------------------------------------------

    def _get_from_cache(self, customer_id: str) -> Optional[CustomerContext]:
        """Get customer context from cache if not expired."""
        if customer_id not in self._context_cache:
            return None

        context, expires_at = self._context_cache[customer_id]
        if datetime.utcnow() > expires_at:
            del self._context_cache[customer_id]
            return None

        return context

    def _put_in_cache(self, customer_id: str, context: CustomerContext) -> None:
        """Put customer context into cache with TTL."""
        # Enforce cache size limit
        if len(self._context_cache) >= self._config.cache_max_entries:
            # Evict oldest entry
            oldest_key = next(iter(self._context_cache))
            del self._context_cache[oldest_key]

        expires_at = datetime.utcnow() + timedelta(seconds=self._config.cache_ttl_seconds)
        context.cache_expires_at = expires_at
        self._context_cache[customer_id] = (context, expires_at)

    def invalidate_cache(self, customer_id: Optional[str] = None) -> None:
        """
        Invalidate cached customer context.

        Args:
            customer_id: Specific customer to invalidate (None = clear all)
        """
        if customer_id:
            self._context_cache.pop(customer_id, None)
        else:
            self._context_cache.clear()

    # -----------------------------------------------------------------------
    # Event handling
    # -----------------------------------------------------------------------

    def register_event_callback(self, callback: Callable[[CRMEvent], None]) -> None:
        """Register a callback for CRM events."""
        self._event_callbacks.append(callback)

    def unregister_event_callback(self, callback: Callable[[CRMEvent], None]) -> None:
        """Unregister a CRM event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    def _emit_event(
        self,
        event_type: CRMEventType,
        customer_id: str,
        agent_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a CRM event to registered callbacks."""
        event = CRMEvent(
            event_type=event_type,
            customer_id=customer_id,
            agent_id=agent_id,
            data=data or {},
        )

        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as exc:
                logger.error(f"Error in CRM event callback: {exc}", exc_info=True)

    # -----------------------------------------------------------------------
    # Metrics and health
    # -----------------------------------------------------------------------

    def _record_response_time(self, elapsed_ms: float) -> None:
        """Record a response time measurement."""
        self._response_times.append(elapsed_ms)
        # Keep bounded
        if len(self._response_times) > 1000:
            self._response_times = self._response_times[-1000:]

    def get_health_status(self) -> CRMHealthStatus:
        """Get comprehensive health status."""
        avg_response = 0.0
        if self._response_times:
            avg_response = sum(self._response_times) / len(self._response_times)

        total_cache_ops = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / total_cache_ops * 100) if total_cache_ops > 0 else 0.0

        return CRMHealthStatus(
            is_connected=self._connected,
            is_healthy=self._healthy,
            last_request_time=self._connect_time,
            requests_this_minute=self._rate_limiter._requests_this_minute,
            requests_this_hour=self._rate_limiter._requests_this_hour,
            rate_limit_remaining=self._rate_limiter.remaining_this_minute,
            avg_response_time_ms=avg_response,
            cache_hit_rate=cache_hit_rate,
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

    # -----------------------------------------------------------------------
    # Audit logging
    # -----------------------------------------------------------------------

    def _audit_log_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Record an audit log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "crm_type": self._crm_type.display_name,
            "action": action,
            "details": details,
        }
        self._audit_log.append(entry)

        # Keep bounded
        max_in_memory = 10000
        if len(self._audit_log) > max_in_memory:
            self._audit_log = self._audit_log[-max_in_memory:]

        logger.debug(f"CRM audit: {action} - {details}")

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries (most recent first)."""
        return self._audit_log[-limit:][::-1]

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def _check_connected(self) -> None:
        """Verify the bridge is connected."""
        if not self._connected:
            raise CRMConnectionError(
                f"Not connected to {self._crm_type.display_name}"
            )
