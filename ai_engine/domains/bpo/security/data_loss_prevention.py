"""
BPO Data Loss Prevention Module

DLP engine for preventing customer data exfiltration in BPO environments.

Monitors and prevents:
- Copy/paste of PII (SSN, credit cards, phone numbers)
- Screen scraping and unauthorized screenshots
- Email/chat exfiltration of customer data
- USB/removable media data theft
- Cloud storage uploads
- Print operations with sensitive data
- Voice channel data leakage (agent reading numbers aloud)

This module is designed for the unique challenges of BPO environments
where large numbers of agents handle sensitive customer data across
multiple channels (voice, chat, email) simultaneously.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import logging
import re
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DataType(Enum):
    """Types of sensitive data detected by the DLP engine."""

    CREDIT_CARD = (
        "credit_card",
        "Credit/Debit Card Number (PAN)",
        "PCI-DSS",
    )
    SSN = (
        "ssn",
        "Social Security Number",
        "PII",
    )
    PHONE = (
        "phone",
        "Phone Number",
        "PII",
    )
    EMAIL = (
        "email",
        "Email Address",
        "PII",
    )
    ADDRESS = (
        "address",
        "Physical/Mailing Address",
        "PII",
    )
    DOB = (
        "dob",
        "Date of Birth",
        "PII",
    )
    ACCOUNT_NUMBER = (
        "account_number",
        "Bank/Financial Account Number",
        "PCI/PII",
    )
    PHI = (
        "phi",
        "Protected Health Information",
        "HIPAA",
    )
    PASSPORT = (
        "passport",
        "Passport Number",
        "PII",
    )
    DRIVERS_LICENSE = (
        "drivers_license",
        "Driver's License Number",
        "PII",
    )
    TAX_ID = (
        "tax_id",
        "Tax Identification Number",
        "PII",
    )
    ROUTING_NUMBER = (
        "routing_number",
        "Bank Routing Number",
        "PCI",
    )
    CVV = (
        "cvv",
        "Card Verification Value",
        "PCI-DSS",
    )
    PIN = (
        "pin",
        "Personal Identification Number",
        "PCI-DSS",
    )
    API_KEY = (
        "api_key",
        "API Key or Token",
        "SECRET",
    )

    def __init__(self, type_id: str, display_name: str, category: str):
        self.type_id = type_id
        self.display_name = display_name
        self.category = category


class DLPAction(Enum):
    """Actions the DLP engine can take when a violation is detected."""

    BLOCK = (1, "Block the action entirely")
    ALERT = (2, "Allow but send security alert")
    LOG = (3, "Log the event for audit")
    MASK = (4, "Replace sensitive data with masked version")
    QUARANTINE = (5, "Quarantine the content for review")
    ENCRYPT = (6, "Encrypt the data before allowing")
    REDIRECT = (7, "Redirect to secure channel")

    def __init__(self, priority: int, description: str):
        self.priority = priority
        self.description = description


class DLPChannel(Enum):
    """Data exfiltration channels monitored by the DLP engine."""

    CLIPBOARD = auto()       # Copy/paste operations
    SCREEN_CAPTURE = auto()  # Screenshot/screen recording
    EMAIL = auto()           # Email (corporate or personal)
    CHAT = auto()            # Chat/messaging applications
    USB = auto()             # USB/removable media
    CLOUD_STORAGE = auto()   # Cloud storage uploads
    PRINT = auto()           # Print operations
    VOICE = auto()           # Voice channel (agent reading data)
    FILE_TRANSFER = auto()   # File transfer/download
    BROWSER = auto()         # Browser uploads/form submissions
    API = auto()             # API/webhook data transmission
    NETWORK = auto()         # Raw network exfiltration


class DLPSeverity(Enum):
    """Severity of DLP violations."""

    LOW = (1, "Low", "Minor policy infraction")
    MEDIUM = (2, "Medium", "Potential data exposure")
    HIGH = (3, "High", "Significant data leakage risk")
    CRITICAL = (4, "Critical", "Active data exfiltration")

    def __init__(self, level: int, display_name: str, description: str):
        self.level = level
        self.display_name = display_name
        self.description = description


class IncidentStatus(Enum):
    """Status of a DLP incident."""

    OPEN = auto()
    INVESTIGATING = auto()
    CONFIRMED = auto()
    FALSE_POSITIVE = auto()
    RESOLVED = auto()
    ESCALATED = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DLPRule:
    """
    A configurable DLP rule.

    Rules define what data types to watch for, in which channels,
    and what action to take when a violation is detected.
    """

    rule_id: str = ""
    name: str = ""
    description: str = ""
    enabled: bool = True
    priority: int = 100              # Lower = higher priority

    # What to detect
    data_type: DataType = DataType.CREDIT_CARD
    pattern: Optional[str] = None    # Custom regex pattern
    min_confidence: float = 0.8

    # Where to monitor
    channels: Set[DLPChannel] = field(default_factory=lambda: {
        DLPChannel.CLIPBOARD,
        DLPChannel.SCREEN_CAPTURE,
        DLPChannel.EMAIL,
        DLPChannel.CHAT,
        DLPChannel.USB,
        DLPChannel.CLOUD_STORAGE,
        DLPChannel.PRINT,
    })

    # What to do
    severity: DLPSeverity = DLPSeverity.HIGH
    action: DLPAction = DLPAction.BLOCK

    # Context
    tenant_ids: Set[str] = field(default_factory=set)  # Empty = all tenants
    agent_roles: Set[str] = field(default_factory=set)  # Empty = all roles

    # Thresholds
    max_occurrences_per_hour: int = 0      # 0 = trigger on first
    cooldown_seconds: int = 60

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DLPViolation:
    """
    A DLP violation detected by the engine.

    Captures what was detected, where, and what action was taken.
    """

    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    rule_name: str = ""
    agent_id: str = ""
    tenant_id: str = ""
    session_id: str = ""

    # What was detected
    data_type: DataType = DataType.CREDIT_CARD
    channel: DLPChannel = DLPChannel.CLIPBOARD
    severity: DLPSeverity = DLPSeverity.HIGH
    confidence: float = 0.0

    # Context
    masked_content: str = ""         # Masked version of detected data
    content_hash: str = ""           # SHA-256 hash for correlation
    source_application: str = ""
    destination: str = ""            # Where data was going (email, URL, etc.)

    # Action taken
    action_taken: DLPAction = DLPAction.BLOCK
    was_blocked: bool = False

    # Timestamps
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize violation for storage."""
        return {
            "violation_id": self.violation_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "data_type": self.data_type.type_id,
            "channel": self.channel.name,
            "severity": self.severity.display_name,
            "confidence": self.confidence,
            "masked_content": self.masked_content,
            "content_hash": self.content_hash,
            "source_application": self.source_application,
            "destination": self.destination,
            "action_taken": self.action_taken.name,
            "was_blocked": self.was_blocked,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class DLPIncident:
    """
    A DLP incident created from one or more violations.

    Incidents group related violations for investigation.
    """

    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    agent_id: str = ""
    status: IncidentStatus = IncidentStatus.OPEN
    severity: DLPSeverity = DLPSeverity.HIGH
    description: str = ""
    violations: List[str] = field(default_factory=list)  # violation IDs
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize incident for storage."""
        return {
            "incident_id": self.incident_id,
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "status": self.status.name,
            "severity": self.severity.display_name,
            "description": self.description,
            "violation_count": len(self.violations),
            "assigned_to": self.assigned_to,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": (
                self.resolved_at.isoformat() if self.resolved_at else None
            ),
        }


# ---------------------------------------------------------------------------
# PII Detector
# ---------------------------------------------------------------------------


class PIIDetector:
    """
    Regex and pattern-based PII detection engine.

    Detects various types of personally identifiable information
    in text data using regular expressions, checksum validation,
    and contextual analysis.
    """

    # Detection patterns by data type
    PATTERNS: Dict[DataType, Dict[str, Any]] = {
        DataType.CREDIT_CARD: {
            "regex": re.compile(
                r"(?<!\d)"
                r"(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"
                r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,7}"
                r"(?!\d)"
            ),
            "validator": "_validate_luhn",
            "min_length": 13,
            "max_length": 19,
        },
        DataType.SSN: {
            "regex": re.compile(
                r"\b(?!000|666|9\d{2})\d{3}"
                r"[\s\-]?"
                r"(?!00)\d{2}"
                r"[\s\-]?"
                r"(?!0000)\d{4}\b"
            ),
            "validator": "_validate_ssn",
            "min_length": 9,
            "max_length": 11,
        },
        DataType.PHONE: {
            "regex": re.compile(
                r"(?<!\d)"
                r"(?:\+?1[\s\-.]?)?"
                r"(?:\(?[2-9]\d{2}\)?[\s\-.]?)"
                r"[2-9]\d{2}[\s\-.]?\d{4}"
                r"(?!\d)"
            ),
            "validator": None,
            "min_length": 10,
            "max_length": 15,
        },
        DataType.EMAIL: {
            "regex": re.compile(
                r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
            ),
            "validator": None,
            "min_length": 5,
            "max_length": 254,
        },
        DataType.DOB: {
            "regex": re.compile(
                r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])"
                r"[/\-](?:19|20)\d{2}\b"
            ),
            "validator": None,
            "min_length": 8,
            "max_length": 10,
        },
        DataType.ACCOUNT_NUMBER: {
            "regex": re.compile(
                r"\b\d{8,17}\b"
            ),
            "validator": None,
            "min_length": 8,
            "max_length": 17,
            "context_required": True,  # Needs context keywords
            "context_keywords": [
                "account", "acct", "routing", "bank", "savings",
                "checking", "deposit",
            ],
        },
        DataType.CVV: {
            "regex": re.compile(
                r"(?i)(?:cvv|cvc|cvv2|cvc2|csv)[:\s]*(\d{3,4})"
            ),
            "validator": None,
            "min_length": 3,
            "max_length": 4,
        },
        DataType.PASSPORT: {
            "regex": re.compile(
                r"\b[A-Z]{1,2}\d{6,9}\b"
            ),
            "validator": None,
            "min_length": 7,
            "max_length": 11,
            "context_required": True,
            "context_keywords": ["passport", "travel", "document"],
        },
        DataType.DRIVERS_LICENSE: {
            "regex": re.compile(
                r"\b[A-Z]\d{3,8}\b"
            ),
            "validator": None,
            "min_length": 4,
            "max_length": 9,
            "context_required": True,
            "context_keywords": ["driver", "license", "dl", "dmv"],
        },
        DataType.TAX_ID: {
            "regex": re.compile(
                r"\b\d{2}[\-]\d{7}\b"
            ),
            "validator": None,
            "min_length": 9,
            "max_length": 10,
            "context_required": True,
            "context_keywords": ["tax", "ein", "tin", "itin", "fein"],
        },
        DataType.ROUTING_NUMBER: {
            "regex": re.compile(
                r"\b(?:0[1-9]|1[0-2]|2[1-9]|3[0-2]|6[1-9]|7[0-2]|80)\d{7}\b"
            ),
            "validator": "_validate_routing",
            "min_length": 9,
            "max_length": 9,
        },
    }

    def __init__(
        self,
        *,
        enabled_types: Optional[Set[DataType]] = None,
        min_confidence: float = 0.7,
        context_window_chars: int = 100,
    ):
        self._enabled_types: Set[DataType] = enabled_types or set(self.PATTERNS.keys())
        self._min_confidence = min_confidence
        self._context_window_chars = context_window_chars
        self._detection_counts: Dict[DataType, int] = {
            dt: 0 for dt in DataType
        }

    def detect(
        self, text: str, context: str = ""
    ) -> List[Tuple[DataType, str, float, int]]:
        """
        Detect PII in text.

        Args:
            text: The text to scan.
            context: Additional context for the scan.

        Returns:
            List of (data_type, masked_value, confidence, position) tuples.
        """
        detections: List[Tuple[DataType, str, float, int]] = []

        for data_type in self._enabled_types:
            pattern_info = self.PATTERNS.get(data_type)
            if not pattern_info:
                continue

            regex = pattern_info["regex"]
            requires_context = pattern_info.get("context_required", False)
            context_keywords = pattern_info.get("context_keywords", [])

            for match in regex.finditer(text):
                raw = match.group(0)
                digits_only = re.sub(r"[\s\-\(\)/.]", "", raw)

                # Length validation
                min_len = pattern_info.get("min_length", 0)
                max_len = pattern_info.get("max_length", 100)
                if not (min_len <= len(digits_only) <= max_len):
                    continue

                # Context check (if required)
                if requires_context:
                    surrounding = text[
                        max(0, match.start() - self._context_window_chars):
                        match.end() + self._context_window_chars
                    ].lower()
                    has_context = any(
                        kw in surrounding for kw in context_keywords
                    )
                    if not has_context:
                        continue

                # Custom validator
                validator_name = pattern_info.get("validator")
                confidence = 0.9  # Default high confidence
                if validator_name:
                    validator = getattr(self, validator_name, None)
                    if validator:
                        is_valid, conf = validator(digits_only)
                        if not is_valid:
                            continue
                        confidence = conf

                if confidence >= self._min_confidence:
                    masked = self._mask_value(digits_only, data_type)
                    detections.append(
                        (data_type, masked, confidence, match.start())
                    )
                    self._detection_counts[data_type] += 1

        return detections

    def redact(self, text: str) -> Tuple[str, int]:
        """
        Redact all PII from text.

        Args:
            text: The text to redact.

        Returns:
            Tuple of (redacted text, number of redactions).
        """
        detections = self.detect(text)
        result = text
        redaction_count = 0

        # Process detections in reverse order to maintain positions
        for data_type, masked, confidence, position in sorted(
            detections, key=lambda d: d[3], reverse=True
        ):
            # Find the original match at this position
            pattern_info = self.PATTERNS.get(data_type)
            if not pattern_info:
                continue

            regex = pattern_info["regex"]
            for match in regex.finditer(result):
                if abs(match.start() - position) < 5:
                    result = (
                        result[:match.start()]
                        + f"[{data_type.type_id.upper()}_REDACTED]"
                        + result[match.end():]
                    )
                    redaction_count += 1
                    break

        return result, redaction_count

    @staticmethod
    def _validate_luhn(number: str) -> Tuple[bool, float]:
        """Validate a number using the Luhn algorithm."""
        digits = re.sub(r"\D", "", number)
        if len(digits) < 13 or len(digits) > 19:
            return False, 0.0

        checksum = 0
        for i, d in enumerate(reversed(digits)):
            n = int(d)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            checksum += n

        is_valid = checksum % 10 == 0
        return is_valid, 0.95 if is_valid else 0.0

    @staticmethod
    def _validate_ssn(number: str) -> Tuple[bool, float]:
        """Validate SSN format."""
        digits = re.sub(r"\D", "", number)
        if len(digits) != 9:
            return False, 0.0

        # SSN rules: not 000, 666, 9xx; middle not 00; last not 0000
        area = int(digits[:3])
        group = int(digits[3:5])
        serial = int(digits[5:])

        if area == 0 or area == 666 or area >= 900:
            return False, 0.0
        if group == 0:
            return False, 0.0
        if serial == 0:
            return False, 0.0

        return True, 0.85

    @staticmethod
    def _validate_routing(number: str) -> Tuple[bool, float]:
        """Validate ABA routing number checksum."""
        digits = re.sub(r"\D", "", number)
        if len(digits) != 9:
            return False, 0.0

        # ABA checksum: 3*d1 + 7*d2 + d3 + 3*d4 + 7*d5 + d6 + 3*d7 + 7*d8 + d9 = 0 mod 10
        weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
        checksum = sum(
            int(d) * w for d, w in zip(digits, weights)
        )
        is_valid = checksum % 10 == 0
        return is_valid, 0.90 if is_valid else 0.0

    @staticmethod
    def _mask_value(value: str, data_type: DataType) -> str:
        """Mask a detected value based on its type."""
        digits = re.sub(r"\D", "", value)

        if data_type == DataType.CREDIT_CARD:
            if len(digits) >= 10:
                return digits[:6] + "*" * (len(digits) - 10) + digits[-4:]
            return "*" * len(digits)
        elif data_type == DataType.SSN:
            return f"***-**-{digits[-4:]}" if len(digits) >= 4 else "***"
        elif data_type == DataType.PHONE:
            if len(digits) >= 4:
                return "*" * (len(digits) - 4) + digits[-4:]
            return "***"
        elif data_type == DataType.EMAIL:
            parts = value.split("@")
            if len(parts) == 2 and len(parts[0]) > 2:
                return parts[0][:2] + "***@" + parts[1]
            return "***@***"
        elif data_type == DataType.ACCOUNT_NUMBER:
            if len(digits) >= 4:
                return "*" * (len(digits) - 4) + digits[-4:]
            return "***"
        else:
            return "*" * len(value)

    def get_statistics(self) -> Dict[str, int]:
        """Get detection statistics by data type."""
        return {
            dt.type_id: count
            for dt, count in self._detection_counts.items()
            if count > 0
        }


# ---------------------------------------------------------------------------
# Channel Monitor
# ---------------------------------------------------------------------------


class ChannelMonitor:
    """
    Monitors specific exfiltration channels for DLP violations.

    Each channel (clipboard, email, USB, etc.) has specific detection
    logic and policies.
    """

    def __init__(
        self,
        *,
        pii_detector: Optional[PIIDetector] = None,
        monitored_channels: Optional[Set[DLPChannel]] = None,
    ):
        self._pii_detector = pii_detector or PIIDetector()
        self._monitored_channels: Set[DLPChannel] = monitored_channels or {
            c for c in DLPChannel
        }
        self._channel_stats: Dict[DLPChannel, Dict[str, int]] = {
            c: {"scanned": 0, "violations": 0, "blocked": 0}
            for c in DLPChannel
        }

    def scan_channel_data(
        self,
        channel: DLPChannel,
        data: str,
        *,
        agent_id: str = "",
        session_id: str = "",
        destination: str = "",
        application: str = "",
    ) -> List[DLPViolation]:
        """
        Scan data from a specific channel for DLP violations.

        Args:
            channel: The channel the data came from.
            data: The textual data to scan.
            agent_id: The agent associated with the data.
            session_id: The session identifier.
            destination: Where the data is going.
            application: The application involved.

        Returns:
            List of DLP violations detected.
        """
        if channel not in self._monitored_channels:
            return []

        self._channel_stats[channel]["scanned"] += 1
        violations: List[DLPViolation] = []

        # Detect PII
        detections = self._pii_detector.detect(data)

        for data_type, masked_value, confidence, position in detections:
            content_hash = hashlib.sha256(
                data[position:position + 50].encode()
            ).hexdigest()

            violation = DLPViolation(
                agent_id=agent_id,
                session_id=session_id,
                data_type=data_type,
                channel=channel,
                severity=self._determine_severity(data_type, channel),
                confidence=confidence,
                masked_content=masked_value,
                content_hash=content_hash,
                source_application=application,
                destination=destination,
            )
            violations.append(violation)
            self._channel_stats[channel]["violations"] += 1

        return violations

    def is_channel_monitored(self, channel: DLPChannel) -> bool:
        """Check if a channel is currently monitored."""
        return channel in self._monitored_channels

    def enable_channel(self, channel: DLPChannel) -> None:
        """Enable monitoring for a channel."""
        self._monitored_channels.add(channel)

    def disable_channel(self, channel: DLPChannel) -> None:
        """Disable monitoring for a channel."""
        self._monitored_channels.discard(channel)

    def get_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get per-channel statistics."""
        return {
            ch.name: stats
            for ch, stats in self._channel_stats.items()
            if stats["scanned"] > 0
        }

    @staticmethod
    def _determine_severity(
        data_type: DataType, channel: DLPChannel
    ) -> DLPSeverity:
        """Determine severity based on data type and channel combination."""
        # High-risk data types
        high_risk_types = {
            DataType.CREDIT_CARD,
            DataType.SSN,
            DataType.PHI,
            DataType.CVV,
            DataType.PIN,
        }
        # High-risk channels
        high_risk_channels = {
            DLPChannel.USB,
            DLPChannel.CLOUD_STORAGE,
            DLPChannel.EMAIL,
            DLPChannel.FILE_TRANSFER,
        }

        if data_type in high_risk_types and channel in high_risk_channels:
            return DLPSeverity.CRITICAL
        elif data_type in high_risk_types:
            return DLPSeverity.HIGH
        elif channel in high_risk_channels:
            return DLPSeverity.MEDIUM
        else:
            return DLPSeverity.LOW


# ---------------------------------------------------------------------------
# Incident Reporter
# ---------------------------------------------------------------------------


class IncidentReporter:
    """
    Manages DLP incidents created from violations.

    Groups related violations into incidents, tracks their lifecycle,
    and provides reporting capabilities.
    """

    def __init__(
        self,
        *,
        auto_escalate_threshold: int = 5,
        incident_callback: Optional[Callable[[DLPIncident], None]] = None,
    ):
        self._auto_escalate_threshold = auto_escalate_threshold
        self._incident_callback = incident_callback
        self._incidents: Dict[str, DLPIncident] = {}
        self._violations: Dict[str, DLPViolation] = {}
        # agent_id -> incident_id mapping for grouping
        self._agent_incidents: Dict[str, str] = {}

    def report_violation(
        self, violation: DLPViolation
    ) -> DLPIncident:
        """
        Report a DLP violation and associate it with an incident.

        If the agent already has an open incident, the violation is
        added to it. Otherwise, a new incident is created.

        Args:
            violation: The DLP violation to report.

        Returns:
            The incident associated with this violation.
        """
        self._violations[violation.violation_id] = violation

        # Find or create incident for this agent
        incident = self._get_or_create_incident(violation)
        incident.violations.append(violation.violation_id)
        incident.updated_at = datetime.utcnow()

        # Update severity based on violation count
        if len(incident.violations) >= self._auto_escalate_threshold:
            incident.severity = DLPSeverity.CRITICAL
            incident.status = IncidentStatus.ESCALATED

        # Fire callback
        if self._incident_callback:
            try:
                self._incident_callback(incident)
            except Exception as exc:
                logger.error("Incident callback failed: %s", exc)

        logger.warning(
            "DLP violation reported: agent=%s type=%s channel=%s "
            "severity=%s incident=%s",
            violation.agent_id,
            violation.data_type.type_id,
            violation.channel.name,
            violation.severity.display_name,
            incident.incident_id,
        )

        return incident

    def get_incident(self, incident_id: str) -> Optional[DLPIncident]:
        """Get an incident by ID."""
        return self._incidents.get(incident_id)

    def get_open_incidents(
        self,
        *,
        tenant_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        min_severity: Optional[DLPSeverity] = None,
    ) -> List[DLPIncident]:
        """Get open incidents with optional filtering."""
        results = [
            i for i in self._incidents.values()
            if i.status in (
                IncidentStatus.OPEN,
                IncidentStatus.INVESTIGATING,
                IncidentStatus.ESCALATED,
            )
        ]

        if tenant_id:
            results = [i for i in results if i.tenant_id == tenant_id]
        if agent_id:
            results = [i for i in results if i.agent_id == agent_id]
        if min_severity:
            results = [
                i for i in results if i.severity.level >= min_severity.level
            ]

        return sorted(results, key=lambda i: i.severity.level, reverse=True)

    def resolve_incident(
        self,
        incident_id: str,
        resolution_notes: str,
        status: IncidentStatus = IncidentStatus.RESOLVED,
    ) -> bool:
        """Resolve an incident."""
        incident = self._incidents.get(incident_id)
        if not incident:
            return False

        incident.status = status
        incident.resolved_at = datetime.utcnow()
        incident.resolution_notes = resolution_notes
        incident.updated_at = datetime.utcnow()

        # Clear agent mapping if resolved
        if incident.agent_id in self._agent_incidents:
            if self._agent_incidents[incident.agent_id] == incident_id:
                del self._agent_incidents[incident.agent_id]

        logger.info(
            "DLP incident resolved: incident=%s status=%s",
            incident_id,
            status.name,
        )
        return True

    def assign_incident(
        self, incident_id: str, assigned_to: str
    ) -> bool:
        """Assign an incident to an investigator."""
        incident = self._incidents.get(incident_id)
        if not incident:
            return False

        incident.assigned_to = assigned_to
        incident.status = IncidentStatus.INVESTIGATING
        incident.updated_at = datetime.utcnow()
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get incident statistics."""
        total = len(self._incidents)
        open_count = sum(
            1 for i in self._incidents.values()
            if i.status in (
                IncidentStatus.OPEN,
                IncidentStatus.INVESTIGATING,
                IncidentStatus.ESCALATED,
            )
        )
        return {
            "total_incidents": total,
            "open_incidents": open_count,
            "resolved_incidents": total - open_count,
            "total_violations": len(self._violations),
        }

    def _get_or_create_incident(
        self, violation: DLPViolation
    ) -> DLPIncident:
        """Get an existing open incident or create a new one."""
        # Check for existing open incident for this agent
        existing_id = self._agent_incidents.get(violation.agent_id)
        if existing_id and existing_id in self._incidents:
            incident = self._incidents[existing_id]
            if incident.status in (
                IncidentStatus.OPEN,
                IncidentStatus.INVESTIGATING,
                IncidentStatus.ESCALATED,
            ):
                return incident

        # Create new incident
        incident = DLPIncident(
            tenant_id=violation.tenant_id,
            agent_id=violation.agent_id,
            severity=violation.severity,
            description=(
                f"DLP violation: {violation.data_type.display_name} "
                f"detected in {violation.channel.name}"
            ),
        )
        self._incidents[incident.incident_id] = incident
        self._agent_incidents[violation.agent_id] = incident.incident_id

        return incident


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """
    DLP policy evaluation engine.

    Evaluates DLP rules against detected violations and determines
    the appropriate action to take.
    """

    def __init__(self) -> None:
        self._rules: Dict[str, DLPRule] = {}
        self._evaluation_count: int = 0

    def add_rule(self, rule: DLPRule) -> None:
        """Add or update a DLP rule."""
        self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a DLP rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def get_rules(self) -> List[DLPRule]:
        """Get all rules sorted by priority."""
        return sorted(self._rules.values(), key=lambda r: r.priority)

    def evaluate(
        self,
        data_type: DataType,
        channel: DLPChannel,
        *,
        tenant_id: str = "",
        agent_role: str = "",
        confidence: float = 1.0,
    ) -> Tuple[DLPAction, Optional[DLPRule]]:
        """
        Evaluate rules and determine the action for a detection.

        Args:
            data_type: The type of data detected.
            channel: The channel where it was detected.
            tenant_id: The tenant context.
            agent_role: The agent's role.
            confidence: Detection confidence.

        Returns:
            Tuple of (action to take, matching rule or None).
        """
        self._evaluation_count += 1

        matching_rules: List[DLPRule] = []
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.data_type != data_type:
                continue
            if channel not in rule.channels:
                continue
            if confidence < rule.min_confidence:
                continue
            if rule.tenant_ids and tenant_id not in rule.tenant_ids:
                continue
            if rule.agent_roles and agent_role not in rule.agent_roles:
                continue
            matching_rules.append(rule)

        if not matching_rules:
            return DLPAction.LOG, None

        # Return the highest priority (lowest number) matching rule
        best_rule = min(matching_rules, key=lambda r: r.priority)
        return best_rule.action, best_rule

    def load_default_rules(self) -> None:
        """Load default DLP rules for BPO environments."""
        defaults = [
            DLPRule(
                rule_id="DLP-001",
                name="Block Credit Card Exfiltration",
                description="Block credit card data via all high-risk channels",
                data_type=DataType.CREDIT_CARD,
                channels={
                    DLPChannel.CLIPBOARD,
                    DLPChannel.EMAIL,
                    DLPChannel.CHAT,
                    DLPChannel.USB,
                    DLPChannel.CLOUD_STORAGE,
                    DLPChannel.PRINT,
                    DLPChannel.FILE_TRANSFER,
                },
                severity=DLPSeverity.CRITICAL,
                action=DLPAction.BLOCK,
                priority=10,
            ),
            DLPRule(
                rule_id="DLP-002",
                name="Block SSN Exfiltration",
                description="Block SSN data via all channels",
                data_type=DataType.SSN,
                severity=DLPSeverity.CRITICAL,
                action=DLPAction.BLOCK,
                priority=10,
            ),
            DLPRule(
                rule_id="DLP-003",
                name="Alert on Email/Phone Export",
                description="Alert when email addresses or phone numbers are copied",
                data_type=DataType.EMAIL,
                channels={DLPChannel.CLIPBOARD, DLPChannel.EMAIL, DLPChannel.CHAT},
                severity=DLPSeverity.MEDIUM,
                action=DLPAction.ALERT,
                priority=50,
            ),
            DLPRule(
                rule_id="DLP-004",
                name="Alert on Phone Number Export",
                description="Alert when phone numbers are copied or emailed",
                data_type=DataType.PHONE,
                channels={DLPChannel.CLIPBOARD, DLPChannel.EMAIL, DLPChannel.CHAT},
                severity=DLPSeverity.MEDIUM,
                action=DLPAction.ALERT,
                priority=50,
            ),
            DLPRule(
                rule_id="DLP-005",
                name="Block PHI Exfiltration",
                description="Block protected health information via all channels",
                data_type=DataType.PHI,
                severity=DLPSeverity.CRITICAL,
                action=DLPAction.BLOCK,
                priority=10,
            ),
            DLPRule(
                rule_id="DLP-006",
                name="Block CVV Capture",
                description="Block CVV data from being copied or stored",
                data_type=DataType.CVV,
                severity=DLPSeverity.CRITICAL,
                action=DLPAction.BLOCK,
                priority=5,
            ),
            DLPRule(
                rule_id="DLP-007",
                name="Block USB Data Transfer",
                description="Block all sensitive data to USB devices",
                data_type=DataType.CREDIT_CARD,
                channels={DLPChannel.USB},
                severity=DLPSeverity.CRITICAL,
                action=DLPAction.BLOCK,
                priority=5,
            ),
            DLPRule(
                rule_id="DLP-008",
                name="Mask Account Numbers in Logs",
                description="Mask account numbers when written to logs",
                data_type=DataType.ACCOUNT_NUMBER,
                channels={DLPChannel.API, DLPChannel.NETWORK},
                severity=DLPSeverity.HIGH,
                action=DLPAction.MASK,
                priority=30,
            ),
            DLPRule(
                rule_id="DLP-009",
                name="Block Screen Capture with PAN",
                description="Block screenshots containing card numbers",
                data_type=DataType.CREDIT_CARD,
                channels={DLPChannel.SCREEN_CAPTURE},
                severity=DLPSeverity.CRITICAL,
                action=DLPAction.BLOCK,
                priority=5,
            ),
            DLPRule(
                rule_id="DLP-010",
                name="Log Address Access",
                description="Log when physical addresses are accessed",
                data_type=DataType.ADDRESS,
                severity=DLPSeverity.LOW,
                action=DLPAction.LOG,
                priority=90,
            ),
        ]

        for rule in defaults:
            self._rules[rule.rule_id] = rule

        logger.info("Loaded %d default DLP rules", len(defaults))


# ---------------------------------------------------------------------------
# BPO Data Loss Prevention Engine (orchestrator)
# ---------------------------------------------------------------------------


class BPODataLossPreventionEngine:
    """
    DLP engine for preventing customer data exfiltration in BPO environments.

    Monitors and prevents:
    - Copy/paste of PII (SSN, credit cards, phone numbers)
    - Screen scraping and unauthorized screenshots
    - Email/chat exfiltration of customer data
    - USB/removable media data theft
    - Cloud storage uploads
    - Print operations with sensitive data
    - Voice channel data leakage (agent reading numbers aloud)

    This is the main orchestrator that coordinates the PII detector,
    channel monitor, policy engine, and incident reporter to provide
    comprehensive DLP coverage for BPO operations.

    Usage::

        dlp = BPODataLossPreventionEngine(tenant_id="tenant-001")
        dlp.load_default_rules()

        # Scan clipboard content
        result = dlp.scan(
            data="My card is 4111 1111 1111 1111",
            channel=DLPChannel.CLIPBOARD,
            agent_id="agent-001",
        )

        if result["violations"]:
            print(f"Blocked: {result['violations'][0].data_type.display_name}")
    """

    def __init__(
        self,
        tenant_id: str = "",
        *,
        enable_pii_detection: bool = True,
        enable_channel_monitoring: bool = True,
        enable_incident_tracking: bool = True,
        alert_callback: Optional[Callable[[DLPViolation], None]] = None,
        incident_callback: Optional[Callable[[DLPIncident], None]] = None,
    ):
        self.tenant_id = tenant_id
        self.alert_callback = alert_callback

        # Initialize sub-components
        self.pii_detector = PIIDetector()
        self.channel_monitor = ChannelMonitor(
            pii_detector=self.pii_detector,
        )
        self.policy_engine = PolicyEngine()
        self.incident_reporter = IncidentReporter(
            incident_callback=incident_callback,
        )

        self._enable_pii_detection = enable_pii_detection
        self._enable_channel_monitoring = enable_channel_monitoring
        self._enable_incident_tracking = enable_incident_tracking

        self._stats = {
            "total_scans": 0,
            "total_violations": 0,
            "total_blocks": 0,
            "total_alerts": 0,
        }

        logger.info(
            "BPODataLossPreventionEngine initialized: tenant=%s "
            "pii=%s channels=%s incidents=%s",
            tenant_id,
            enable_pii_detection,
            enable_channel_monitoring,
            enable_incident_tracking,
        )

    def load_default_rules(self) -> None:
        """Load default DLP rules."""
        self.policy_engine.load_default_rules()

    def scan(
        self,
        data: str,
        channel: DLPChannel,
        *,
        agent_id: str = "",
        session_id: str = "",
        tenant_id: Optional[str] = None,
        agent_role: str = "",
        destination: str = "",
        application: str = "",
    ) -> Dict[str, Any]:
        """
        Scan data for DLP violations.

        This is the primary entry point for DLP scanning. It:
        1. Detects PII in the data
        2. Evaluates policy rules
        3. Determines and applies actions
        4. Creates incidents if needed

        Args:
            data: The text data to scan.
            channel: The channel where the data was captured.
            agent_id: The agent associated with the data.
            session_id: The session identifier.
            tenant_id: The tenant context (uses default if None).
            agent_role: The agent's role for policy evaluation.
            destination: Where the data is being sent.
            application: The application involved.

        Returns:
            Dict with scan results including violations and actions.
        """
        effective_tenant = tenant_id or self.tenant_id
        self._stats["total_scans"] += 1

        result: Dict[str, Any] = {
            "scanned": True,
            "violations": [],
            "actions": [],
            "blocked": False,
            "masked_data": data,
        }

        # Detect violations through channel monitor
        violations = self.channel_monitor.scan_channel_data(
            channel=channel,
            data=data,
            agent_id=agent_id,
            session_id=session_id,
            destination=destination,
            application=application,
        )

        # Evaluate each violation against policy
        for violation in violations:
            violation.tenant_id = effective_tenant

            action, rule = self.policy_engine.evaluate(
                data_type=violation.data_type,
                channel=channel,
                tenant_id=effective_tenant,
                agent_role=agent_role,
                confidence=violation.confidence,
            )

            violation.action_taken = action
            violation.was_blocked = action == DLPAction.BLOCK

            if rule:
                violation.rule_id = rule.rule_id
                violation.rule_name = rule.name
                violation.severity = rule.severity

            result["violations"].append(violation)
            result["actions"].append(action.name)

            if action == DLPAction.BLOCK:
                result["blocked"] = True
                self._stats["total_blocks"] += 1

            if action in (DLPAction.ALERT, DLPAction.BLOCK):
                self._stats["total_alerts"] += 1
                if self.alert_callback:
                    try:
                        self.alert_callback(violation)
                    except Exception as exc:
                        logger.error("DLP alert callback failed: %s", exc)

            # Track incident
            if self._enable_incident_tracking:
                self.incident_reporter.report_violation(violation)

            self._stats["total_violations"] += 1

        # Apply masking if requested
        if violations and any(
            v.action_taken == DLPAction.MASK for v in violations
        ):
            masked, _ = self.pii_detector.redact(data)
            result["masked_data"] = masked

        return result

    def scan_clipboard(
        self,
        content: str,
        agent_id: str,
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Convenience method to scan clipboard content."""
        return self.scan(
            data=content,
            channel=DLPChannel.CLIPBOARD,
            agent_id=agent_id,
            session_id=session_id,
        )

    def scan_email(
        self,
        content: str,
        agent_id: str,
        destination: str,
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Convenience method to scan email content."""
        return self.scan(
            data=content,
            channel=DLPChannel.EMAIL,
            agent_id=agent_id,
            session_id=session_id,
            destination=destination,
        )

    def scan_chat_message(
        self,
        content: str,
        agent_id: str,
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Convenience method to scan chat message content."""
        return self.scan(
            data=content,
            channel=DLPChannel.CHAT,
            agent_id=agent_id,
            session_id=session_id,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall DLP engine statistics."""
        stats = dict(self._stats)
        stats["pii_detector"] = self.pii_detector.get_statistics()
        stats["channel_monitor"] = self.channel_monitor.get_statistics()
        stats["incident_reporter"] = self.incident_reporter.get_statistics()
        return stats

    def get_open_incidents(
        self,
        *,
        agent_id: Optional[str] = None,
        min_severity: Optional[DLPSeverity] = None,
    ) -> List[DLPIncident]:
        """Get open DLP incidents."""
        return self.incident_reporter.get_open_incidents(
            tenant_id=self.tenant_id,
            agent_id=agent_id,
            min_severity=min_severity,
        )
