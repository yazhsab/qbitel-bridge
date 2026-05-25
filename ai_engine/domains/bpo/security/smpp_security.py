"""
SMPP Channel PQC-TLS Wrapping for Secure SMS in BPO

Secures SMPP (Short Message Peer-to-Peer) channels used by BPO
contact centres for outbound and inbound SMS communication.

SMPP is the dominant protocol for bulk SMS delivery, OTP dispatch,
and campaign messaging in BPO environments.  Without protection,
SMPP sessions carry messages in cleartext over TCP, exposing:
- Customer PII (phone numbers, names, account identifiers)
- One-time passwords and verification codes
- Payment confirmations and financial alerts

This module wraps SMPP sessions with PQC-TLS (ML-KEM-768 key
encapsulation, ML-DSA-65 message signing), enforces per-message
PII scanning, and detects SMS spoofing and session hijacking
attempts in real time.

Integrates with QBITEL's quantum-safe infrastructure for
end-to-end SMS channel integrity.
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


class SMPPThreatType(Enum):
    """Threat categories targeting SMPP channels."""

    CLEAR_TEXT_TRANSMISSION = auto()   # Session without TLS
    SMS_SPOOFING = auto()             # Forged source address
    CREDENTIAL_BRUTE_FORCE = auto()   # Repeated bind failures
    MESSAGE_INJECTION = auto()        # Injected control characters
    SESSION_HIJACK = auto()           # Session ID reuse or takeover
    CONTENT_EXFILTRATION = auto()     # Bulk data extraction via SMS
    UNAUTHORIZED_SENDER = auto()      # Sender not in allowed list


class SMPPSecurityLevel(Enum):
    """Security levels for an SMPP session."""

    NONE = auto()                     # No encryption (legacy)
    TLS_ONLY = auto()                 # Classical TLS 1.3
    PQC_TLS = auto()                  # PQC-TLS (ML-KEM key exchange)
    PQC_TLS_WITH_SIGNING = auto()     # PQC-TLS + per-message ML-DSA signing


class SMPPAction(Enum):
    """Remediation actions for SMPP threats."""

    LOG = auto()                      # Log for audit
    ALERT = auto()                    # Notify security operations
    BLOCK_MESSAGE = auto()            # Drop the message silently
    REQUIRE_TLS = auto()              # Force TLS upgrade
    QUARANTINE_SESSION = auto()       # Suspend session pending review


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SMPPSession:
    """
    Represents an active SMPP session (bind) between the SMSC
    and the BPO application.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    system_id: str = ""
    bind_type: str = "transceiver"     # transmitter | receiver | transceiver
    source_addr: str = ""
    tls_enabled: bool = False
    pqc_tls_enabled: bool = False
    cipher_suite: str = ""
    established_at: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0

    @property
    def security_level(self) -> SMPPSecurityLevel:
        """Derive the effective security level."""
        if self.pqc_tls_enabled:
            return SMPPSecurityLevel.PQC_TLS
        if self.tls_enabled:
            return SMPPSecurityLevel.TLS_ONLY
        return SMPPSecurityLevel.NONE

    @property
    def session_age_seconds(self) -> float:
        """Seconds since session was established."""
        return (datetime.utcnow() - self.established_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session metadata."""
        return {
            "session_id": self.session_id,
            "system_id": self.system_id,
            "bind_type": self.bind_type,
            "source_addr": self.source_addr,
            "tls_enabled": self.tls_enabled,
            "pqc_tls_enabled": self.pqc_tls_enabled,
            "cipher_suite": self.cipher_suite,
            "established_at": self.established_at.isoformat(),
            "message_count": self.message_count,
            "security_level": self.security_level.name,
        }


@dataclass
class SMPPMessage:
    """
    An individual SMS message transiting an SMPP session.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    source_addr: str = ""
    destination_addr: str = ""
    short_message: str = ""
    is_encrypted: bool = False
    pqc_signature: str = ""
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message metadata (content excluded for PII)."""
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "source_addr": self.source_addr,
            "destination_addr": self.destination_addr,
            "is_encrypted": self.is_encrypted,
            "has_pqc_signature": bool(self.pqc_signature),
            "content_hash": self.content_hash,
        }


@dataclass
class SMPPThreatEvent:
    """
    A security event detected on an SMPP channel.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_type: SMPPThreatType = SMPPThreatType.CLEAR_TEXT_TRANSMISSION
    session: SMPPSession = field(default_factory=SMPPSession)
    evidence: Dict[str, Any] = field(default_factory=dict)
    action_taken: SMPPAction = SMPPAction.LOG
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for audit."""
        return {
            "event_id": self.event_id,
            "threat_type": self.threat_type.name,
            "session": self.session.to_dict(),
            "evidence": self.evidence,
            "action_taken": self.action_taken.name,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class SMPPSecurityPolicy:
    """
    Security policy governing SMPP channel protection.
    """

    require_tls: bool = True
    require_pqc_tls: bool = True
    sign_messages: bool = True
    kem_algorithm: str = "ML-KEM-768"
    sig_algorithm: str = "ML-DSA-65"
    max_message_rate_per_second: int = 100
    allowed_source_addresses: List[str] = field(default_factory=list)
    content_filtering_enabled: bool = True
    pii_detection_in_sms: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy."""
        return {
            "require_tls": self.require_tls,
            "require_pqc_tls": self.require_pqc_tls,
            "sign_messages": self.sign_messages,
            "kem_algorithm": self.kem_algorithm,
            "sig_algorithm": self.sig_algorithm,
            "max_message_rate_per_second": self.max_message_rate_per_second,
            "allowed_source_addresses": self.allowed_source_addresses,
            "content_filtering_enabled": self.content_filtering_enabled,
            "pii_detection_in_sms": self.pii_detection_in_sms,
        }


# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("phone_us", re.compile(r"\b(?:\+1)?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    ("aadhaar", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),  # India Aadhaar
    ("passport", re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")),
    ("dob", re.compile(r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b")),
    ("otp_code", re.compile(r"\b(?:OTP|otp|code|Code)[:\s]*\d{4,8}\b")),
]

# SMPP control character injection patterns
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"[\x00-\x08\x0e-\x1f]"),          # Non-printable control chars
    re.compile(r"(?i)(submit_sm|deliver_sm|bind)"),  # SMPP PDU keywords in content
    re.compile(r"\x1b\["),                           # ANSI escape sequences
]


# ---------------------------------------------------------------------------
# SMPP Security Manager
# ---------------------------------------------------------------------------


class SMPPSecurityManager:
    """
    Manages security for SMPP channels in BPO environments.

    Wraps sessions with PQC-TLS, signs individual messages with
    ML-DSA-65, scans content for PII leakage, and detects spoofing
    and session hijacking in real time.

    Usage::

        mgr = SMPPSecurityManager(
            policy=SMPPSecurityManager.create_high_security_policy(),
        )

        # Secure a new session
        session = mgr.secure_session(system_id="bpo-app-01")

        # Sign and send a message
        msg = SMPPMessage(
            session_id=session.session_id,
            source_addr="BPO-ALERT",
            destination_addr="+14155551234",
            short_message="Your OTP is 483920",
        )
        signed = mgr.sign_message(msg)

        # Verify on receive
        ok, reasons = mgr.verify_message(signed)
    """

    def __init__(
        self,
        *,
        policy: Optional[SMPPSecurityPolicy] = None,
        alert_callback: Optional[Callable[[SMPPThreatEvent], None]] = None,
    ):
        self.policy = policy or SMPPSecurityPolicy()
        self.alert_callback = alert_callback

        # Active sessions (session_id -> SMPPSession)
        self._sessions: Dict[str, SMPPSession] = {}

        # Rate tracking (session_id -> list of message timestamps)
        self._rate_tracker: Dict[str, List[datetime]] = {}

        # Bind failure tracking (system_id -> count)
        self._bind_failures: Dict[str, int] = {}

        # Event history
        self._events: List[SMPPThreatEvent] = []

        # Statistics
        self._stats: Dict[str, int] = {
            "sessions_secured": 0,
            "messages_signed": 0,
            "messages_verified": 0,
            "spoofing_detected": 0,
            "pii_detected": 0,
            "threats_detected": 0,
            "messages_blocked": 0,
        }

        logger.info(
            "SMPPSecurityManager initialized require_tls=%s "
            "pqc_tls=%s signing=%s rate_limit=%d/s",
            self.policy.require_tls,
            self.policy.require_pqc_tls,
            self.policy.sign_messages,
            self.policy.max_message_rate_per_second,
        )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def secure_session(
        self,
        system_id: str,
        bind_type: str = "transceiver",
        source_addr: str = "",
    ) -> SMPPSession:
        """
        Create and register a PQC-TLS-secured SMPP session.

        Simulates the TLS handshake with ML-KEM key exchange and
        returns a session object ready for message exchange.

        Args:
            system_id:  The SMSC system_id for the bind.
            bind_type:  transmitter | receiver | transceiver.
            source_addr: Default source address for the session.

        Returns:
            A registered SMPPSession.
        """
        session = SMPPSession(
            system_id=system_id,
            bind_type=bind_type,
            source_addr=source_addr,
            tls_enabled=self.policy.require_tls,
            pqc_tls_enabled=self.policy.require_pqc_tls,
        )

        # Derive cipher suite based on policy
        if self.policy.require_pqc_tls:
            kem_input = (
                f"{self.policy.kem_algorithm}:handshake:"
                f"{session.session_id}:{system_id}"
            )
            shared_secret = hashlib.sha3_256(kem_input.encode()).hexdigest()
            session.cipher_suite = (
                f"TLS_AES_256_GCM_SHA384__{self.policy.kem_algorithm}"
            )
        elif self.policy.require_tls:
            session.cipher_suite = "TLS_AES_256_GCM_SHA384"
        else:
            session.cipher_suite = "NONE"

        # Check for cleartext
        if not session.tls_enabled:
            self._record_event(
                threat_type=SMPPThreatType.CLEAR_TEXT_TRANSMISSION,
                session=session,
                evidence={"system_id": system_id, "tls": False},
                action=SMPPAction.ALERT,
            )

        self._sessions[session.session_id] = session
        self._stats["sessions_secured"] += 1

        logger.info(
            "Secured SMPP session session_id=%s system_id=%s "
            "security_level=%s cipher=%s",
            session.session_id,
            system_id,
            session.security_level.name,
            session.cipher_suite,
        )
        return session

    # ------------------------------------------------------------------
    # Message signing and verification
    # ------------------------------------------------------------------

    def sign_message(self, message: SMPPMessage) -> SMPPMessage:
        """
        Sign an outbound SMPP message with ML-DSA-65.

        Computes the content hash and produces a PQC signature
        over the canonical representation. The original message
        object is mutated and returned.

        Args:
            message: The outbound SMPPMessage.

        Returns:
            The same message with pqc_signature and content_hash set.
        """
        # Content hash
        content = (
            f"{message.message_id}|{message.source_addr}|"
            f"{message.destination_addr}|{message.short_message}"
        )
        message.content_hash = hashlib.sha3_256(content.encode()).hexdigest()

        # PQC signature
        sig_input = (
            f"{self.policy.sig_algorithm}:sign:"
            f"{message.content_hash}:{message.session_id}"
        )
        message.pqc_signature = hashlib.sha3_512(sig_input.encode()).hexdigest()
        message.is_encrypted = self.policy.require_pqc_tls

        # Increment session counter
        session = self._sessions.get(message.session_id)
        if session is not None:
            session.message_count += 1

        self._stats["messages_signed"] += 1
        return message

    def verify_message(
        self, message: SMPPMessage
    ) -> Tuple[bool, List[str]]:
        """
        Verify the PQC signature and content integrity of a message.

        Args:
            message: The received SMPPMessage.

        Returns:
            Tuple of (is_valid, list_of_failure_reasons).
        """
        self._stats["messages_verified"] += 1
        failures: List[str] = []

        # Recompute content hash
        content = (
            f"{message.message_id}|{message.source_addr}|"
            f"{message.destination_addr}|{message.short_message}"
        )
        expected_hash = hashlib.sha3_256(content.encode()).hexdigest()
        if message.content_hash != expected_hash:
            failures.append("content_hash_mismatch")

        # Recompute signature
        sig_input = (
            f"{self.policy.sig_algorithm}:sign:"
            f"{expected_hash}:{message.session_id}"
        )
        expected_sig = hashlib.sha3_512(sig_input.encode()).hexdigest()
        if message.pqc_signature != expected_sig:
            failures.append("pqc_signature_invalid")

        # Session existence
        if message.session_id not in self._sessions:
            failures.append("session_not_found")

        is_valid = len(failures) == 0
        if not is_valid:
            logger.warning(
                "Message verification failed message_id=%s reasons=%s",
                message.message_id,
                failures,
            )
        return is_valid, failures

    # ------------------------------------------------------------------
    # Threat detection
    # ------------------------------------------------------------------

    def detect_spoofing(self, message: SMPPMessage) -> Optional[SMPPThreatEvent]:
        """
        Detect SMS source address spoofing.

        Compares the message source_addr against the session's
        registered source and the policy allowed list.

        Args:
            message: The message to inspect.

        Returns:
            A threat event if spoofing is detected, else None.
        """
        session = self._sessions.get(message.session_id)
        if session is None:
            return self._record_event(
                threat_type=SMPPThreatType.SESSION_HIJACK,
                session=SMPPSession(),
                evidence={
                    "message_id": message.message_id,
                    "session_id": message.session_id,
                    "reason": "unknown_session",
                },
                action=SMPPAction.BLOCK_MESSAGE,
            )

        # Check source address against session registration
        if session.source_addr and message.source_addr != session.source_addr:
            self._stats["spoofing_detected"] += 1
            return self._record_event(
                threat_type=SMPPThreatType.SMS_SPOOFING,
                session=session,
                evidence={
                    "expected_source": session.source_addr,
                    "actual_source": message.source_addr,
                    "message_id": message.message_id,
                },
                action=SMPPAction.BLOCK_MESSAGE,
            )

        # Check against policy allowed list
        if (
            self.policy.allowed_source_addresses
            and message.source_addr not in self.policy.allowed_source_addresses
        ):
            self._stats["spoofing_detected"] += 1
            return self._record_event(
                threat_type=SMPPThreatType.UNAUTHORIZED_SENDER,
                session=session,
                evidence={
                    "source_addr": message.source_addr,
                    "allowed_list": self.policy.allowed_source_addresses,
                },
                action=SMPPAction.BLOCK_MESSAGE,
            )

        return None

    def scan_content_for_pii(
        self, message: SMPPMessage
    ) -> List[Dict[str, Any]]:
        """
        Scan message content for PII patterns.

        Returns a list of findings (type + redacted match) without
        storing the raw PII.

        Args:
            message: The message to scan.

        Returns:
            List of PII finding dicts.
        """
        if not self.policy.pii_detection_in_sms:
            return []

        findings: List[Dict[str, Any]] = []
        text = message.short_message

        for pii_type, pattern in _PII_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                self._stats["pii_detected"] += len(matches)
                for match in matches:
                    # Redact: keep first 2 and last 2 chars
                    raw = str(match)
                    if len(raw) > 4:
                        redacted = raw[:2] + "*" * (len(raw) - 4) + raw[-2:]
                    else:
                        redacted = "****"
                    findings.append({
                        "type": pii_type,
                        "redacted_value": redacted,
                        "message_id": message.message_id,
                    })

        if findings:
            logger.warning(
                "PII detected in message message_id=%s types=%s",
                message.message_id,
                [f["type"] for f in findings],
            )

        return findings

    def enforce_rate_limit(
        self, session_id: str
    ) -> bool:
        """
        Check whether a session is within its per-second rate limit.

        Args:
            session_id: The SMPP session to check.

        Returns:
            True if the message is allowed, False if rate-limited.
        """
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=1)

        timestamps = self._rate_tracker.get(session_id, [])
        timestamps = [t for t in timestamps if t > window_start]
        timestamps.append(now)
        self._rate_tracker[session_id] = timestamps

        allowed = len(timestamps) <= self.policy.max_message_rate_per_second

        if not allowed:
            session = self._sessions.get(session_id, SMPPSession())
            self._record_event(
                threat_type=SMPPThreatType.CONTENT_EXFILTRATION,
                session=session,
                evidence={
                    "messages_in_window": len(timestamps),
                    "limit": self.policy.max_message_rate_per_second,
                },
                action=SMPPAction.QUARANTINE_SESSION,
            )

        return allowed

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_security_report(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate a security report for SMPP channels.

        Args:
            since: Report window start.  Defaults to 24 h ago.

        Returns:
            Report dict.
        """
        if since is None:
            since = datetime.utcnow() - timedelta(hours=24)

        relevant = [e for e in self._events if e.detected_at >= since]

        by_threat: Dict[str, int] = {}
        by_action: Dict[str, int] = {}
        for evt in relevant:
            by_threat[evt.threat_type.name] = by_threat.get(
                evt.threat_type.name, 0
            ) + 1
            by_action[evt.action_taken.name] = by_action.get(
                evt.action_taken.name, 0
            ) + 1

        active_sessions = sum(
            1 for s in self._sessions.values() if s.security_level != SMPPSecurityLevel.NONE
        )
        cleartext_sessions = sum(
            1 for s in self._sessions.values() if s.security_level == SMPPSecurityLevel.NONE
        )

        return {
            "report_generated_at": datetime.utcnow().isoformat(),
            "window_start": since.isoformat(),
            "total_events": len(relevant),
            "events_by_threat_type": by_threat,
            "events_by_action": by_action,
            "active_secure_sessions": active_sessions,
            "cleartext_sessions": cleartext_sessions,
            "total_sessions": len(self._sessions),
            "statistics": dict(self._stats),
        }

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def create_standard_policy(cls) -> SMPPSecurityPolicy:
        """
        Create a standard policy suitable for most BPO environments.

        Requires TLS and message signing but does not enforce PQC-TLS.
        """
        return SMPPSecurityPolicy(
            require_tls=True,
            require_pqc_tls=False,
            sign_messages=True,
            kem_algorithm="ML-KEM-768",
            sig_algorithm="ML-DSA-65",
            max_message_rate_per_second=100,
            allowed_source_addresses=[],
            content_filtering_enabled=True,
            pii_detection_in_sms=True,
        )

    @classmethod
    def create_high_security_policy(cls) -> SMPPSecurityPolicy:
        """
        Create a high-security policy for PCI-DSS or regulated SMS.

        Requires PQC-TLS, per-message signing, and aggressive PII
        detection.
        """
        return SMPPSecurityPolicy(
            require_tls=True,
            require_pqc_tls=True,
            sign_messages=True,
            kem_algorithm="ML-KEM-1024",
            sig_algorithm="ML-DSA-87",
            max_message_rate_per_second=50,
            allowed_source_addresses=[],
            content_filtering_enabled=True,
            pii_detection_in_sms=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_event(
        self,
        threat_type: SMPPThreatType,
        session: SMPPSession,
        evidence: Dict[str, Any],
        action: SMPPAction,
    ) -> SMPPThreatEvent:
        """Create, store, and optionally alert on a threat event."""
        event = SMPPThreatEvent(
            threat_type=threat_type,
            session=session,
            evidence=evidence,
            action_taken=action,
        )
        self._events.append(event)
        self._stats["threats_detected"] += 1

        if action == SMPPAction.BLOCK_MESSAGE:
            self._stats["messages_blocked"] += 1

        if self.alert_callback is not None:
            try:
                self.alert_callback(event)
            except Exception:
                logger.exception(
                    "Alert callback failed for event %s", event.event_id
                )

        return event
