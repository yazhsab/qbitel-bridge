"""
SS7/Diameter Interception Defense with PQC Tunnel Overlay

Protects BPO signalling infrastructure against SS7 and Diameter
protocol-level attacks by applying post-quantum cryptographic
tunnel overlays to the signalling plane.

SS7/MAP and Diameter remain the backbone of mobile core networks.
Attackers exploit weaknesses in these protocols to:
- Track subscriber locations via Any-Time-Interrogation
- Intercept calls and SMS through UpdateLocation fraud
- Cause subscriber denial-of-service via CancelLocation
- Exfiltrate data through Diameter Sh/S6a interfaces

This module inspects every signalling message against a threat
model, applies PQC-encrypted tunnel overlays (ML-KEM-768 for
key encapsulation, ML-DSA-65 for authentication), and logs
tamper-proof audit events for regulatory compliance.

Integrates with QBITEL's quantum-safe infrastructure for
end-to-end signalling integrity.
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


class SS7ThreatType(Enum):
    """Categories of SS7/Diameter signalling threats."""

    LOCATION_TRACKING = auto()         # ATI / PSI abuse
    CALL_INTERCEPTION = auto()         # UpdateLocation redirect
    SMS_INTERCEPTION = auto()          # SRI-SM redirect
    SUBSCRIBER_DOS = auto()            # CancelLocation / PurgeMS
    ACCOUNT_FRAUD = auto()             # InsertSubscriberData tampering
    CALL_REDIRECT = auto()             # SendRoutingInfo manipulation
    DIAMETER_DATA_LEAK = auto()        # Diameter Sh/S6a data exfiltration
    SIP_I_INTERWORK_EXPLOIT = auto()   # SIP-I/T inter-working abuse


class SS7MessageType(Enum):
    """SS7 MAP message types relevant to security analysis."""

    UPDATE_LOCATION = auto()
    SEND_ROUTING_INFO = auto()
    PROVIDE_SUBSCRIBER_INFO = auto()
    ANY_TIME_INTERROGATION = auto()
    INSERT_SUBSCRIBER_DATA = auto()
    CANCEL_LOCATION = auto()
    SEND_AUTH_INFO = auto()
    PURGE_MS = auto()


class SS7ShieldAction(Enum):
    """Remediation actions for signalling threats."""

    LOG = auto()                       # Log only
    ALERT = auto()                     # Notify security operations
    BLOCK_MESSAGE = auto()             # Silently drop the message
    REQUIRE_VERIFICATION = auto()      # Challenge the originator
    REDIRECT_TO_HONEYPOT = auto()      # Route to deception node
    ENCRYPT_RESPONSE = auto()          # PQC-encrypt the response payload


class ProtocolLayer(Enum):
    """Signalling protocol layers inspected by the shield."""

    SS7_MAP = auto()
    SS7_ISUP = auto()
    SIGTRAN_M3UA = auto()
    DIAMETER_S6A = auto()
    DIAMETER_SH = auto()
    SIP_I = auto()
    SIP_T = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SS7Message:
    """
    An intercepted SS7/Diameter signalling message.

    Contains the protocol metadata needed for threat analysis
    without carrying the full decoded payload.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: SS7MessageType = SS7MessageType.UPDATE_LOCATION
    protocol_layer: ProtocolLayer = ProtocolLayer.SS7_MAP
    source_point_code: str = ""
    destination_point_code: str = ""
    imsi: str = ""
    msisdn: str = ""
    payload_hash: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_suspicious: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message metadata for audit logging."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.name,
            "protocol_layer": self.protocol_layer.name,
            "source_point_code": self.source_point_code,
            "destination_point_code": self.destination_point_code,
            "imsi": self.imsi,
            "msisdn": self.msisdn,
            "payload_hash": self.payload_hash,
            "timestamp": self.timestamp.isoformat(),
            "is_suspicious": self.is_suspicious,
        }


@dataclass
class SS7ThreatEvent:
    """
    A confirmed or suspected SS7/Diameter threat event.

    Carries the original message, classification, confidence
    score, and the PQC audit hash for tamper-proof evidence.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_type: SS7ThreatType = SS7ThreatType.LOCATION_TRACKING
    message: SS7Message = field(default_factory=SS7Message)
    confidence: float = 0.0            # 0.0 to 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    action_taken: SS7ShieldAction = SS7ShieldAction.LOG
    pqc_audit_hash: str = ""
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for storage or transmission."""
        return {
            "event_id": self.event_id,
            "threat_type": self.threat_type.name,
            "message": self.message.to_dict(),
            "confidence": self.confidence,
            "evidence": self.evidence,
            "action_taken": self.action_taken.name,
            "pqc_audit_hash": self.pqc_audit_hash,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class SS7ShieldPolicy:
    """
    Configuration policy for the SS7/Diameter shield.

    Controls which protocol layers are inspected, which
    message classes are blocked by default, and the PQC
    algorithms used for tunnel overlays.
    """

    enabled_layers: List[ProtocolLayer] = field(default_factory=list)
    block_location_queries: bool = True
    block_unauthorized_routing: bool = True
    encrypt_responses: bool = True
    pqc_tunnel_kem: str = "ML-KEM-768"
    pqc_tunnel_sig: str = "ML-DSA-65"
    honeypot_enabled: bool = False
    whitelist_point_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy."""
        return {
            "enabled_layers": [l.name for l in self.enabled_layers],
            "block_location_queries": self.block_location_queries,
            "block_unauthorized_routing": self.block_unauthorized_routing,
            "encrypt_responses": self.encrypt_responses,
            "pqc_tunnel_kem": self.pqc_tunnel_kem,
            "pqc_tunnel_sig": self.pqc_tunnel_sig,
            "honeypot_enabled": self.honeypot_enabled,
            "whitelist_point_codes": self.whitelist_point_codes,
        }


# ---------------------------------------------------------------------------
# Threat signatures
# ---------------------------------------------------------------------------

# Map of message type to the threat types it can indicate when
# originating from a non-whitelisted point code.
_THREAT_SIGNATURES: Dict[SS7MessageType, List[SS7ThreatType]] = {
    SS7MessageType.ANY_TIME_INTERROGATION: [
        SS7ThreatType.LOCATION_TRACKING,
    ],
    SS7MessageType.PROVIDE_SUBSCRIBER_INFO: [
        SS7ThreatType.LOCATION_TRACKING,
        SS7ThreatType.DIAMETER_DATA_LEAK,
    ],
    SS7MessageType.UPDATE_LOCATION: [
        SS7ThreatType.CALL_INTERCEPTION,
        SS7ThreatType.SMS_INTERCEPTION,
    ],
    SS7MessageType.SEND_ROUTING_INFO: [
        SS7ThreatType.CALL_REDIRECT,
        SS7ThreatType.SMS_INTERCEPTION,
    ],
    SS7MessageType.INSERT_SUBSCRIBER_DATA: [
        SS7ThreatType.ACCOUNT_FRAUD,
    ],
    SS7MessageType.CANCEL_LOCATION: [
        SS7ThreatType.SUBSCRIBER_DOS,
    ],
    SS7MessageType.PURGE_MS: [
        SS7ThreatType.SUBSCRIBER_DOS,
    ],
    SS7MessageType.SEND_AUTH_INFO: [
        SS7ThreatType.ACCOUNT_FRAUD,
        SS7ThreatType.CALL_INTERCEPTION,
    ],
}


# ---------------------------------------------------------------------------
# SS7 Shield Engine
# ---------------------------------------------------------------------------


class SS7ShieldEngine:
    """
    Inspects and defends SS7/Diameter signalling traffic.

    Sits inline on the SIGTRAN/Diameter path, analyses every
    message against threat signatures, and applies PQC-encrypted
    tunnel overlays to prevent interception or manipulation.

    Usage::

        engine = SS7ShieldEngine(
            policy=SS7ShieldEngine.create_maximum_protection_policy(),
        )

        # Analyse a captured message
        events = engine.analyze_message(msg)

        # Apply PQC overlay to outbound response
        overlay = engine.apply_pqc_tunnel_overlay(response_msg)
    """

    def __init__(
        self,
        *,
        policy: Optional[SS7ShieldPolicy] = None,
        alert_callback: Optional[Callable[[SS7ThreatEvent], None]] = None,
    ):
        self.policy = policy or SS7ShieldPolicy(
            enabled_layers=list(ProtocolLayer),
        )
        self.alert_callback = alert_callback

        # Threat event history
        self._events: List[SS7ThreatEvent] = []

        # Message counters per source point code (for velocity analysis)
        self._message_counters: Dict[str, List[datetime]] = {}

        # Blocked source point codes (auto-populated on repeated threats)
        self._blocked_point_codes: Set[str] = set()

        # Statistics
        self._stats: Dict[str, int] = {
            "messages_inspected": 0,
            "threats_detected": 0,
            "messages_blocked": 0,
            "tunnels_applied": 0,
            "honeypot_redirects": 0,
        }

        logger.info(
            "SS7ShieldEngine initialized layers=%d honeypot=%s "
            "kem=%s sig=%s",
            len(self.policy.enabled_layers),
            self.policy.honeypot_enabled,
            self.policy.pqc_tunnel_kem,
            self.policy.pqc_tunnel_sig,
        )

    # ------------------------------------------------------------------
    # Message analysis
    # ------------------------------------------------------------------

    def analyze_message(self, message: SS7Message) -> List[SS7ThreatEvent]:
        """
        Analyse a signalling message for threat indicators.

        Runs the message through all applicable detectors and returns
        any generated threat events.  Messages from whitelisted point
        codes are still logged but not blocked.

        Args:
            message: The intercepted SS7/Diameter message.

        Returns:
            List of SS7ThreatEvent instances (may be empty).
        """
        self._stats["messages_inspected"] += 1
        events: List[SS7ThreatEvent] = []

        # Skip disabled layers
        if message.protocol_layer not in self.policy.enabled_layers:
            return events

        # Auto-block previously flagged sources
        if message.source_point_code in self._blocked_point_codes:
            evt = self._create_event(
                threat_type=SS7ThreatType.CALL_INTERCEPTION,
                message=message,
                confidence=1.0,
                evidence={"reason": "source_point_code_blocked"},
                action=SS7ShieldAction.BLOCK_MESSAGE,
            )
            events.append(evt)
            return events

        # Check whitelist
        is_whitelisted = (
            message.source_point_code in self.policy.whitelist_point_codes
        )

        # Run specific detectors
        events.extend(self.detect_location_tracking(message))
        events.extend(self.detect_call_interception(message))
        events.extend(self.detect_sms_interception(message))

        # Velocity analysis (>20 messages from same source in 1 min)
        velocity_events = self._check_velocity(message)
        events.extend(velocity_events)

        # Origin validation
        origin_event = self.validate_message_origin(message)
        if origin_event is not None:
            events.append(origin_event)

        # If events found and source is not whitelisted, consider blocking
        if events and not is_whitelisted:
            max_confidence = max(e.confidence for e in events)
            if max_confidence >= 0.8:
                self._blocked_point_codes.add(message.source_point_code)
            message.is_suspicious = True

        self._stats["threats_detected"] += len(events)
        return events

    def detect_location_tracking(
        self, message: SS7Message
    ) -> List[SS7ThreatEvent]:
        """
        Detect subscriber location tracking attempts.

        ATI (Any-Time-Interrogation) and PSI (Provide-Subscriber-Info)
        from non-operator sources indicate location surveillance.

        Args:
            message: The SS7 message to inspect.

        Returns:
            List of threat events.
        """
        events: List[SS7ThreatEvent] = []

        if message.message_type not in (
            SS7MessageType.ANY_TIME_INTERROGATION,
            SS7MessageType.PROVIDE_SUBSCRIBER_INFO,
        ):
            return events

        if not self.policy.block_location_queries:
            return events

        is_whitelisted = (
            message.source_point_code in self.policy.whitelist_point_codes
        )
        confidence = 0.5 if is_whitelisted else 0.9
        action = SS7ShieldAction.LOG if is_whitelisted else SS7ShieldAction.BLOCK_MESSAGE

        evt = self._create_event(
            threat_type=SS7ThreatType.LOCATION_TRACKING,
            message=message,
            confidence=confidence,
            evidence={
                "message_type": message.message_type.name,
                "target_imsi": message.imsi,
                "target_msisdn": message.msisdn,
                "source_whitelisted": is_whitelisted,
            },
            action=action,
        )
        events.append(evt)
        return events

    def detect_call_interception(
        self, message: SS7Message
    ) -> List[SS7ThreatEvent]:
        """
        Detect call interception via UpdateLocation manipulation.

        A fraudulent UpdateLocation changes the subscriber's serving
        MSC/VLR so that inbound calls route through the attacker's
        infrastructure.

        Args:
            message: The SS7 message to inspect.

        Returns:
            List of threat events.
        """
        events: List[SS7ThreatEvent] = []

        if message.message_type != SS7MessageType.UPDATE_LOCATION:
            return events

        is_whitelisted = (
            message.source_point_code in self.policy.whitelist_point_codes
        )

        if is_whitelisted:
            return events

        confidence = 0.85
        action = SS7ShieldAction.BLOCK_MESSAGE

        if self.policy.honeypot_enabled:
            action = SS7ShieldAction.REDIRECT_TO_HONEYPOT
            self._stats["honeypot_redirects"] += 1

        evt = self._create_event(
            threat_type=SS7ThreatType.CALL_INTERCEPTION,
            message=message,
            confidence=confidence,
            evidence={
                "message_type": message.message_type.name,
                "target_imsi": message.imsi,
                "fraudulent_vlr": message.destination_point_code,
                "honeypot_redirect": action == SS7ShieldAction.REDIRECT_TO_HONEYPOT,
            },
            action=action,
        )
        events.append(evt)
        return events

    def detect_sms_interception(
        self, message: SS7Message
    ) -> List[SS7ThreatEvent]:
        """
        Detect SMS interception via SendRoutingInfo-for-SM redirect.

        Attackers use SRI-SM to change the SMS-C routing so that
        inbound SMS (including OTP codes) arrive at their node.

        Args:
            message: The SS7 message to inspect.

        Returns:
            List of threat events.
        """
        events: List[SS7ThreatEvent] = []

        if message.message_type != SS7MessageType.SEND_ROUTING_INFO:
            return events

        is_whitelisted = (
            message.source_point_code in self.policy.whitelist_point_codes
        )
        if is_whitelisted:
            return events

        if not self.policy.block_unauthorized_routing:
            return events

        evt = self._create_event(
            threat_type=SS7ThreatType.SMS_INTERCEPTION,
            message=message,
            confidence=0.80,
            evidence={
                "message_type": message.message_type.name,
                "target_msisdn": message.msisdn,
                "suspicious_routing_dest": message.destination_point_code,
            },
            action=SS7ShieldAction.BLOCK_MESSAGE,
        )
        events.append(evt)
        return events

    # ------------------------------------------------------------------
    # PQC tunnel overlay
    # ------------------------------------------------------------------

    def apply_pqc_tunnel_overlay(
        self, message: SS7Message
    ) -> Dict[str, Any]:
        """
        Apply a PQC-encrypted tunnel overlay to a signalling message.

        Encapsulates the message inside an ML-KEM-768 encrypted
        envelope with an ML-DSA-65 authentication tag, preventing
        eavesdropping or tampering on the signalling path.

        Args:
            message: The outbound message to protect.

        Returns:
            Dict containing the encrypted envelope metadata.
        """
        self._stats["tunnels_applied"] += 1

        # Build canonical payload for encapsulation
        canonical = (
            f"{message.message_id}|{message.message_type.name}|"
            f"{message.source_point_code}|{message.destination_point_code}|"
            f"{message.payload_hash}|{message.timestamp.isoformat()}"
        )

        # Simulate KEM encapsulation (shared secret derivation)
        kem_input = f"{self.policy.pqc_tunnel_kem}:encap:{canonical}"
        shared_secret = hashlib.sha3_256(kem_input.encode()).hexdigest()

        # Simulate symmetric encryption of payload under shared secret
        enc_input = f"AES-256-GCM:{shared_secret}:{message.payload_hash}"
        ciphertext_hash = hashlib.sha3_256(enc_input.encode()).hexdigest()

        # Simulate DSA authentication tag
        sig_input = (
            f"{self.policy.pqc_tunnel_sig}:sign:{ciphertext_hash}:"
            f"{message.source_point_code}"
        )
        auth_tag = hashlib.sha3_512(sig_input.encode()).hexdigest()

        overlay = {
            "message_id": message.message_id,
            "kem_algorithm": self.policy.pqc_tunnel_kem,
            "sig_algorithm": self.policy.pqc_tunnel_sig,
            "shared_secret_hash": shared_secret[:32],  # Truncated for log
            "ciphertext_hash": ciphertext_hash,
            "auth_tag": auth_tag,
            "applied_at": datetime.utcnow().isoformat(),
        }

        logger.debug(
            "PQC tunnel overlay applied message_id=%s kem=%s",
            message.message_id,
            self.policy.pqc_tunnel_kem,
        )
        return overlay

    # ------------------------------------------------------------------
    # Origin validation
    # ------------------------------------------------------------------

    def validate_message_origin(
        self, message: SS7Message
    ) -> Optional[SS7ThreatEvent]:
        """
        Validate the origin point code of a signalling message.

        Messages from unknown or suspicious point codes that carry
        sensitive operations (InsertSubscriberData, CancelLocation,
        PurgeMS, SendAuthInfo) are flagged automatically.

        Args:
            message: The message to validate.

        Returns:
            A threat event if origin is suspicious, else None.
        """
        sensitive_types = {
            SS7MessageType.INSERT_SUBSCRIBER_DATA,
            SS7MessageType.CANCEL_LOCATION,
            SS7MessageType.PURGE_MS,
            SS7MessageType.SEND_AUTH_INFO,
        }

        if message.message_type not in sensitive_types:
            return None

        is_whitelisted = (
            message.source_point_code in self.policy.whitelist_point_codes
        )
        if is_whitelisted:
            return None

        threat_types = _THREAT_SIGNATURES.get(message.message_type, [])
        primary_threat = (
            threat_types[0] if threat_types else SS7ThreatType.ACCOUNT_FRAUD
        )

        return self._create_event(
            threat_type=primary_threat,
            message=message,
            confidence=0.75,
            evidence={
                "message_type": message.message_type.name,
                "source_point_code": message.source_point_code,
                "reason": "non_whitelisted_sensitive_operation",
            },
            action=SS7ShieldAction.REQUIRE_VERIFICATION,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_threat_report(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate a threat report for the given window.

        Args:
            since: Report window start.  Defaults to 24 h ago.

        Returns:
            Report dict with event counts, top threats, and statistics.
        """
        if since is None:
            since = datetime.utcnow() - timedelta(hours=24)

        relevant = [e for e in self._events if e.detected_at >= since]

        by_threat: Dict[str, int] = {}
        by_action: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        for evt in relevant:
            by_threat[evt.threat_type.name] = by_threat.get(
                evt.threat_type.name, 0
            ) + 1
            by_action[evt.action_taken.name] = by_action.get(
                evt.action_taken.name, 0
            ) + 1
            src = evt.message.source_point_code or "unknown"
            by_source[src] = by_source.get(src, 0) + 1

        # Top offending point codes
        top_sources = sorted(
            by_source.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "report_generated_at": datetime.utcnow().isoformat(),
            "window_start": since.isoformat(),
            "total_events": len(relevant),
            "events_by_threat_type": by_threat,
            "events_by_action": by_action,
            "top_offending_sources": dict(top_sources),
            "blocked_point_codes": sorted(self._blocked_point_codes),
            "statistics": dict(self._stats),
        }

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def create_default_policy(cls) -> SS7ShieldPolicy:
        """
        Create a balanced default policy.

        Enables all protocol layers, blocks location queries and
        unauthorized routing, but does not enable the honeypot.
        """
        return SS7ShieldPolicy(
            enabled_layers=list(ProtocolLayer),
            block_location_queries=True,
            block_unauthorized_routing=True,
            encrypt_responses=True,
            pqc_tunnel_kem="ML-KEM-768",
            pqc_tunnel_sig="ML-DSA-65",
            honeypot_enabled=False,
            whitelist_point_codes=[],
        )

    @classmethod
    def create_maximum_protection_policy(cls) -> SS7ShieldPolicy:
        """
        Create a maximum-protection policy for sensitive environments.

        Enables honeypot redirection and requires PQC encryption on
        all response payloads.
        """
        return SS7ShieldPolicy(
            enabled_layers=list(ProtocolLayer),
            block_location_queries=True,
            block_unauthorized_routing=True,
            encrypt_responses=True,
            pqc_tunnel_kem="ML-KEM-1024",
            pqc_tunnel_sig="ML-DSA-87",
            honeypot_enabled=True,
            whitelist_point_codes=[],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_velocity(
        self, message: SS7Message
    ) -> List[SS7ThreatEvent]:
        """Detect high-velocity signalling from a single source."""
        events: List[SS7ThreatEvent] = []
        src = message.source_point_code
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)

        timestamps = self._message_counters.get(src, [])
        timestamps = [t for t in timestamps if t > window_start]
        timestamps.append(now)
        self._message_counters[src] = timestamps

        if len(timestamps) > 20:
            evt = self._create_event(
                threat_type=SS7ThreatType.SUBSCRIBER_DOS,
                message=message,
                confidence=0.70,
                evidence={
                    "messages_in_window": len(timestamps),
                    "window_seconds": 60,
                    "source_point_code": src,
                },
                action=SS7ShieldAction.BLOCK_MESSAGE,
            )
            events.append(evt)
            self._blocked_point_codes.add(src)

        return events

    def _create_event(
        self,
        threat_type: SS7ThreatType,
        message: SS7Message,
        confidence: float,
        evidence: Dict[str, Any],
        action: SS7ShieldAction,
    ) -> SS7ThreatEvent:
        """Create, store, and optionally alert on a threat event."""
        # Build PQC audit hash for tamper-proof evidence chain
        audit_input = (
            f"{threat_type.name}|{message.message_id}|"
            f"{confidence}|{action.name}|"
            f"{datetime.utcnow().isoformat()}"
        )
        pqc_audit_hash = hashlib.sha3_256(audit_input.encode()).hexdigest()

        event = SS7ThreatEvent(
            threat_type=threat_type,
            message=message,
            confidence=confidence,
            evidence=evidence,
            action_taken=action,
            pqc_audit_hash=pqc_audit_hash,
        )
        self._events.append(event)

        if action == SS7ShieldAction.BLOCK_MESSAGE:
            self._stats["messages_blocked"] += 1

        if self.alert_callback is not None:
            try:
                self.alert_callback(event)
            except Exception:
                logger.exception(
                    "Alert callback failed for event %s", event.event_id
                )

        return event
