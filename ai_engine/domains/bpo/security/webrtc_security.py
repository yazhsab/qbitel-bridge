"""
WebRTC PQC-Enhanced Security Module

Secures browser-based agent desktops using PQC-enhanced WebRTC.

Browser-based BPO desktops face unique attack surfaces:
- DTLS race conditions allowing MITM on media channels
- ICE candidate leakage exposing internal network topology
- TURN server hijacking for media interception
- SDP manipulation to downgrade encryption
- Data channel injection for command-and-control
- RTP packet injection for audio/video tampering

This module provides:
- SDP validation and sanitization with PQC cipher enforcement
- ICE candidate filtering to prevent IP leakage
- DTLS 1.3 enforcement with hybrid PQC key exchange
- Real-time threat detection for WebRTC sessions
- Browser fingerprint leak prevention

Integrates with QBITEL's quantum-safe infrastructure for
end-to-end protected agent communications.
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


class WebRTCThreatType(Enum):
    """Types of threats targeting WebRTC-based agent desktops."""

    DTLS_RACE_CONDITION = auto()       # MITM via DTLS handshake race
    ICE_CANDIDATE_LEAK = auto()        # Internal IP leakage through ICE
    TURN_SERVER_HIJACK = auto()        # TURN relay credential theft
    CODEC_DOWNGRADE = auto()           # Forced downgrade to weak codec
    DATA_CHANNEL_INJECTION = auto()    # Malicious data channel payloads
    BROWSER_FINGERPRINT_LEAK = auto()  # Agent browser fingerprint exposure
    SDP_MANIPULATION = auto()          # Tampered SDP offer/answer
    RELAY_ABUSE = auto()               # Unauthorized TURN relay usage
    RTP_PACKET_INJECTION = auto()      # Injected RTP media packets
    XSS_IN_AGENT_DESKTOP = auto()      # XSS targeting the agent desktop UI


class WebRTCSecurityLevel(Enum):
    """Security level for WebRTC session configuration."""

    STANDARD = auto()       # Industry-standard SRTP + DTLS
    ENHANCED = auto()       # Enhanced with strict SDP validation
    PQC_HYBRID = auto()     # Hybrid classical + PQC key exchange
    PQC_ONLY = auto()       # Full PQC (ML-KEM / ML-DSA only)


class WebRTCAction(Enum):
    """Actions to take when a WebRTC threat is detected."""

    LOG = auto()                       # Log event only
    ALERT = auto()                     # Alert security team
    BLOCK_CONNECTION = auto()          # Block the peer connection
    FORCE_RENEGOTIATION = auto()       # Force DTLS renegotiation
    TERMINATE_SESSION = auto()         # Terminate the agent session


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ICECandidate:
    """
    Represents an ICE candidate for WebRTC connectivity.

    Contains the candidate details needed for security analysis
    including IP exposure and relay status checks.
    """

    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "host"               # host, srflx, prflx, relay
    ip_address: str = ""
    port: int = 0
    protocol: str = "udp"            # udp or tcp
    is_private: bool = False
    is_relay: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize candidate for inspection or logging."""
        return {
            "candidate_id": self.candidate_id,
            "type": self.type,
            "ip_address": self.ip_address,
            "port": self.port,
            "protocol": self.protocol,
            "is_private": self.is_private,
            "is_relay": self.is_relay,
        }


@dataclass
class SDPSecurityProfile:
    """
    Security profile extracted from an SDP offer/answer.

    Captures the encryption, codec, and ICE configuration
    for security evaluation.
    """

    srtp_enabled: bool = True
    dtls_version: str = "1.2"
    cipher_suites: List[str] = field(default_factory=list)
    pqc_kem: str = ""
    ice_candidates_filtered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile for logging or comparison."""
        return {
            "srtp_enabled": self.srtp_enabled,
            "dtls_version": self.dtls_version,
            "cipher_suites": self.cipher_suites,
            "pqc_kem": self.pqc_kem,
            "ice_candidates_filtered": self.ice_candidates_filtered,
        }


@dataclass
class WebRTCThreatEvent:
    """
    A detected threat event in a WebRTC session.

    Contains classification, evidence, and the action taken
    in response to the threat.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_type: WebRTCThreatType = WebRTCThreatType.SDP_MANIPULATION
    severity: str = "medium"          # low, medium, high, critical
    session_id: str = ""
    agent_id: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    action_taken: Optional[WebRTCAction] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for storage or alerting."""
        return {
            "event_id": self.event_id,
            "threat_type": self.threat_type.name,
            "severity": self.severity,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "evidence": self.evidence,
            "detected_at": self.detected_at.isoformat(),
            "action_taken": self.action_taken.name if self.action_taken else None,
        }


@dataclass
class WebRTCSecurityPolicy:
    """
    Security policy governing WebRTC session behavior.

    Controls which security features are enforced and
    defines the allowed configuration parameters.
    """

    security_level: WebRTCSecurityLevel = WebRTCSecurityLevel.PQC_HYBRID
    filter_private_candidates: bool = True
    require_dtls_1_3: bool = True
    allowed_codecs: List[str] = field(
        default_factory=lambda: ["opus", "VP8", "VP9", "H264"],
    )
    pqc_dtls_enabled: bool = True
    origin_whitelist: List[str] = field(default_factory=list)
    max_data_channels: int = 2
    sdp_validation_strict: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy for storage or transmission."""
        return {
            "security_level": self.security_level.name,
            "filter_private_candidates": self.filter_private_candidates,
            "require_dtls_1_3": self.require_dtls_1_3,
            "allowed_codecs": self.allowed_codecs,
            "pqc_dtls_enabled": self.pqc_dtls_enabled,
            "origin_whitelist": self.origin_whitelist,
            "max_data_channels": self.max_data_channels,
            "sdp_validation_strict": self.sdp_validation_strict,
        }


# ---------------------------------------------------------------------------
# Private IP detection helpers
# ---------------------------------------------------------------------------

# RFC 1918 and RFC 4193 private address ranges
_PRIVATE_IP_PATTERNS = [
    re.compile(r"^10\."),                                      # 10.0.0.0/8
    re.compile(r"^172\.(1[6-9]|2[0-9]|3[0-1])\."),           # 172.16.0.0/12
    re.compile(r"^192\.168\."),                                # 192.168.0.0/16
    re.compile(r"^127\."),                                     # Loopback
    re.compile(r"^169\.254\."),                                # Link-local
    re.compile(r"^fc[0-9a-f]{2}:", re.IGNORECASE),            # IPv6 ULA
    re.compile(r"^fe80:", re.IGNORECASE),                      # IPv6 link-local
    re.compile(r"^::1$"),                                      # IPv6 loopback
]


def _is_private_ip(ip: str) -> bool:
    """Check if an IP address is private / internal."""
    for pattern in _PRIVATE_IP_PATTERNS:
        if pattern.match(ip):
            return True
    return False


# Weak cipher suites that should be rejected
_WEAK_CIPHER_SUITES: Set[str] = {
    "TLS_RSA_WITH_AES_128_CBC_SHA",
    "TLS_RSA_WITH_AES_256_CBC_SHA",
    "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
    "TLS_RSA_WITH_RC4_128_SHA",
    "TLS_RSA_WITH_RC4_128_MD5",
    "TLS_ECDHE_RSA_WITH_RC4_128_SHA",
    "TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA",
}


# ---------------------------------------------------------------------------
# WebRTC Security Monitor
# ---------------------------------------------------------------------------


class WebRTCSecurityMonitor:
    """
    Monitors and enforces security for WebRTC-based agent desktops.

    Validates SDP offers/answers, filters ICE candidates, checks
    DTLS configuration, and detects real-time threats to WebRTC
    sessions in the BPO environment.

    Usage::

        monitor = WebRTCSecurityMonitor(
            policy=WebRTCSecurityMonitor.create_pqc_policy(),
        )

        # Validate an SDP offer
        threats = monitor.validate_sdp(session_id, sdp_offer)
        if threats:
            for threat in threats:
                monitor.enforce_policy(threat)

        # Filter ICE candidates before sending
        safe_candidates = monitor.filter_ice_candidates(candidates)
    """

    def __init__(
        self,
        policy: Optional[WebRTCSecurityPolicy] = None,
        *,
        alert_callback: Optional[Callable[[WebRTCThreatEvent], None]] = None,
    ):
        self._policy = policy or self.create_pqc_policy()
        self._alert_callback = alert_callback

        # Session tracking
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._threat_history: List[WebRTCThreatEvent] = []

        # Statistics
        self._stats = {
            "total_sdp_validated": 0,
            "total_candidates_filtered": 0,
            "total_candidates_blocked": 0,
            "total_threats_detected": 0,
            "total_sessions_terminated": 0,
        }

        logger.info(
            "WebRTCSecurityMonitor initialized level=%s dtls_1_3=%s pqc=%s strict=%s",
            self._policy.security_level.name,
            self._policy.require_dtls_1_3,
            self._policy.pqc_dtls_enabled,
            self._policy.sdp_validation_strict,
        )

    # ------------------------------------------------------------------
    # SDP validation
    # ------------------------------------------------------------------

    def validate_sdp(
        self,
        session_id: str,
        sdp_text: str,
        agent_id: str = "",
    ) -> List[WebRTCThreatEvent]:
        """
        Validate an SDP offer or answer for security issues.

        Checks for SRTP enforcement, DTLS version, cipher suite
        strength, codec restrictions, and data channel limits.

        Args:
            session_id: The WebRTC session identifier.
            sdp_text: The raw SDP text to validate.
            agent_id: The agent associated with the session.

        Returns:
            List of detected threat events (empty if SDP is clean).
        """
        self._stats["total_sdp_validated"] += 1
        threats: List[WebRTCThreatEvent] = []

        logger.debug(
            "Validating SDP for session=%s agent=%s length=%d",
            session_id,
            agent_id,
            len(sdp_text),
        )

        # Parse SDP lines
        lines = sdp_text.strip().split("\n")
        sdp_data = self._parse_sdp_lines(lines)

        # Check 1: SRTP enforcement
        if not sdp_data.get("has_srtp", False):
            threats.append(WebRTCThreatEvent(
                threat_type=WebRTCThreatType.SDP_MANIPULATION,
                severity="critical",
                session_id=session_id,
                agent_id=agent_id,
                evidence={
                    "reason": "SRTP not enabled in SDP - media would be unencrypted",
                    "sdp_fragment": "Missing RTP/SAVPF or RTP/SAVP profile",
                },
            ))

        # Check 2: DTLS version
        dtls_version = sdp_data.get("dtls_version", "unknown")
        if self._policy.require_dtls_1_3 and dtls_version not in ("1.3", "unknown"):
            threats.append(WebRTCThreatEvent(
                threat_type=WebRTCThreatType.DTLS_RACE_CONDITION,
                severity="high",
                session_id=session_id,
                agent_id=agent_id,
                evidence={
                    "reason": f"DTLS version {dtls_version} does not meet minimum 1.3",
                    "required": "1.3",
                    "found": dtls_version,
                },
            ))

        # Check 3: Weak cipher suites
        cipher_suites = sdp_data.get("cipher_suites", [])
        weak_found = [cs for cs in cipher_suites if cs in _WEAK_CIPHER_SUITES]
        if weak_found:
            threats.append(WebRTCThreatEvent(
                threat_type=WebRTCThreatType.CODEC_DOWNGRADE,
                severity="high",
                session_id=session_id,
                agent_id=agent_id,
                evidence={
                    "reason": "Weak cipher suites present in SDP",
                    "weak_suites": weak_found,
                },
            ))

        # Check 4: Unauthorized codecs
        codecs = sdp_data.get("codecs", [])
        disallowed = [c for c in codecs if c not in self._policy.allowed_codecs]
        if disallowed and self._policy.sdp_validation_strict:
            threats.append(WebRTCThreatEvent(
                threat_type=WebRTCThreatType.CODEC_DOWNGRADE,
                severity="medium",
                session_id=session_id,
                agent_id=agent_id,
                evidence={
                    "reason": "Disallowed codecs found in SDP",
                    "disallowed": disallowed,
                    "allowed": self._policy.allowed_codecs,
                },
            ))

        # Check 5: Data channel count
        data_channel_count = sdp_data.get("data_channel_count", 0)
        if data_channel_count > self._policy.max_data_channels:
            threats.append(WebRTCThreatEvent(
                threat_type=WebRTCThreatType.DATA_CHANNEL_INJECTION,
                severity="high",
                session_id=session_id,
                agent_id=agent_id,
                evidence={
                    "reason": "Excessive data channels in SDP",
                    "found": data_channel_count,
                    "max_allowed": self._policy.max_data_channels,
                },
            ))

        # Check 6: PQC KEM presence (if required)
        if self._policy.pqc_dtls_enabled:
            pqc_kem = sdp_data.get("pqc_kem", "")
            if not pqc_kem and self._policy.security_level in (
                WebRTCSecurityLevel.PQC_HYBRID,
                WebRTCSecurityLevel.PQC_ONLY,
            ):
                threats.append(WebRTCThreatEvent(
                    threat_type=WebRTCThreatType.SDP_MANIPULATION,
                    severity="medium",
                    session_id=session_id,
                    agent_id=agent_id,
                    evidence={
                        "reason": "PQC KEM not present in SDP but required by policy",
                        "security_level": self._policy.security_level.name,
                    },
                ))

        # Track session
        self._sessions[session_id] = {
            "agent_id": agent_id,
            "sdp_validated_at": datetime.utcnow().isoformat(),
            "threats_found": len(threats),
            "sdp_data": sdp_data,
        }

        if threats:
            self._stats["total_threats_detected"] += len(threats)
            logger.warning(
                "SDP validation found %d threats for session=%s",
                len(threats),
                session_id,
            )

        return threats

    def _parse_sdp_lines(self, lines: List[str]) -> Dict[str, Any]:
        """Parse SDP lines into a structured dictionary for analysis."""
        result: Dict[str, Any] = {
            "has_srtp": False,
            "dtls_version": "unknown",
            "cipher_suites": [],
            "codecs": [],
            "data_channel_count": 0,
            "pqc_kem": "",
            "ice_candidates": [],
        }

        for line in lines:
            line = line.strip()

            # Check for SRTP
            if "RTP/SAVPF" in line or "RTP/SAVP" in line:
                result["has_srtp"] = True

            # Check for DTLS fingerprint (implies DTLS is being used)
            if line.startswith("a=fingerprint:"):
                if "sha-384" in line or "sha-512" in line:
                    result["dtls_version"] = "1.3"
                elif "sha-256" in line:
                    result["dtls_version"] = "1.2"

            # Extract codecs from rtpmap
            rtpmap_match = re.match(r"a=rtpmap:\d+\s+(\S+)/", line)
            if rtpmap_match:
                result["codecs"].append(rtpmap_match.group(1))

            # Count data channels
            if "application" in line and "webrtc-datachannel" in line.lower():
                result["data_channel_count"] += 1

            # Check for PQC KEM attribute
            if "a=pqc-kem:" in line:
                result["pqc_kem"] = line.split(":", 1)[1].strip()

            # Check cipher suites
            if "a=crypto:" in line:
                suite_match = re.search(r"a=crypto:\d+\s+(\S+)", line)
                if suite_match:
                    result["cipher_suites"].append(suite_match.group(1))

            # Collect ICE candidates
            if line.startswith("a=candidate:"):
                result["ice_candidates"].append(line)

        return result

    # ------------------------------------------------------------------
    # ICE candidate filtering
    # ------------------------------------------------------------------

    def filter_ice_candidates(
        self,
        candidates: List[ICECandidate],
    ) -> List[ICECandidate]:
        """
        Filter ICE candidates to prevent IP address leakage.

        Removes private/internal IP candidates that would expose
        the agent's network topology to external peers.

        Args:
            candidates: List of ICE candidates to filter.

        Returns:
            Filtered list containing only safe candidates.
        """
        self._stats["total_candidates_filtered"] += len(candidates)
        safe: List[ICECandidate] = []

        for candidate in candidates:
            is_private = _is_private_ip(candidate.ip_address)
            candidate.is_private = is_private
            candidate.is_relay = candidate.type == "relay"

            if is_private and self._policy.filter_private_candidates:
                self._stats["total_candidates_blocked"] += 1
                logger.debug(
                    "Blocked private ICE candidate: %s (%s)",
                    candidate.ip_address,
                    candidate.type,
                )
                continue

            safe.append(candidate)

        logger.info(
            "ICE filtering: %d/%d candidates passed",
            len(safe),
            len(candidates),
        )

        return safe

    # ------------------------------------------------------------------
    # DTLS security check
    # ------------------------------------------------------------------

    def check_dtls_security(
        self,
        session_id: str,
        dtls_version: str,
        cipher_suite: str,
        fingerprint: str,
    ) -> List[WebRTCThreatEvent]:
        """
        Check DTLS handshake parameters for security compliance.

        Validates the DTLS version, cipher suite strength, and
        certificate fingerprint against the security policy.

        Args:
            session_id: The WebRTC session identifier.
            dtls_version: The negotiated DTLS version string.
            cipher_suite: The negotiated cipher suite.
            fingerprint: The peer certificate fingerprint.

        Returns:
            List of detected threat events.
        """
        threats: List[WebRTCThreatEvent] = []
        agent_id = self._sessions.get(session_id, {}).get("agent_id", "")

        # Version check
        version_num = 0.0
        version_match = re.search(r"(\d+\.\d+)", dtls_version)
        if version_match:
            try:
                version_num = float(version_match.group(1))
            except ValueError:
                version_num = 0.0

        if self._policy.require_dtls_1_3 and version_num < 1.3:
            threats.append(WebRTCThreatEvent(
                threat_type=WebRTCThreatType.DTLS_RACE_CONDITION,
                severity="high",
                session_id=session_id,
                agent_id=agent_id,
                evidence={
                    "reason": f"DTLS {dtls_version} below required 1.3",
                    "negotiated_version": dtls_version,
                },
            ))

        # Cipher suite check
        if cipher_suite in _WEAK_CIPHER_SUITES:
            threats.append(WebRTCThreatEvent(
                threat_type=WebRTCThreatType.CODEC_DOWNGRADE,
                severity="high",
                session_id=session_id,
                agent_id=agent_id,
                evidence={
                    "reason": "Weak cipher suite negotiated for DTLS",
                    "cipher_suite": cipher_suite,
                },
            ))

        # Fingerprint validation (check format)
        valid_fp = bool(re.match(
            r"^(sha-256|sha-384|sha-512)\s+[0-9A-Fa-f:]+$",
            fingerprint,
        ))
        if not valid_fp:
            threats.append(WebRTCThreatEvent(
                threat_type=WebRTCThreatType.SDP_MANIPULATION,
                severity="critical",
                session_id=session_id,
                agent_id=agent_id,
                evidence={
                    "reason": "Invalid certificate fingerprint format",
                    "fingerprint": fingerprint[:50],
                },
            ))

        if threats:
            self._stats["total_threats_detected"] += len(threats)

        return threats

    # ------------------------------------------------------------------
    # Threat detection
    # ------------------------------------------------------------------

    def detect_threats(
        self,
        session_id: str,
        agent_id: str = "",
        *,
        rtp_packet_count: int = 0,
        rtp_ssrc_changes: int = 0,
        data_channel_messages: int = 0,
        origin_header: str = "",
        user_agent: str = "",
    ) -> List[WebRTCThreatEvent]:
        """
        Detect runtime threats in an active WebRTC session.

        Analyzes session behavior for anomalies including RTP
        injection, data channel abuse, origin violations, and
        browser fingerprint leakage.

        Args:
            session_id: The WebRTC session identifier.
            agent_id: The agent associated with the session.
            rtp_packet_count: Total RTP packets in the observation window.
            rtp_ssrc_changes: Number of SSRC changes (potential injection).
            data_channel_messages: Number of data channel messages.
            origin_header: The Origin header from the WebRTC signaling.
            user_agent: The User-Agent string from the agent browser.

        Returns:
            List of detected threat events.
        """
        threats: List[WebRTCThreatEvent] = []

        # RTP injection detection (excessive SSRC changes)
        if rtp_ssrc_changes > 3:
            threats.append(WebRTCThreatEvent(
                threat_type=WebRTCThreatType.RTP_PACKET_INJECTION,
                severity="high",
                session_id=session_id,
                agent_id=agent_id,
                evidence={
                    "reason": "Excessive SSRC changes indicate RTP injection",
                    "ssrc_changes": rtp_ssrc_changes,
                    "threshold": 3,
                    "packet_count": rtp_packet_count,
                },
            ))

        # Data channel abuse
        if data_channel_messages > 100:
            threats.append(WebRTCThreatEvent(
                threat_type=WebRTCThreatType.DATA_CHANNEL_INJECTION,
                severity="medium",
                session_id=session_id,
                agent_id=agent_id,
                evidence={
                    "reason": "Unusually high data channel message volume",
                    "message_count": data_channel_messages,
                    "threshold": 100,
                },
            ))

        # Origin validation
        if origin_header and self._policy.origin_whitelist:
            if origin_header not in self._policy.origin_whitelist:
                threats.append(WebRTCThreatEvent(
                    threat_type=WebRTCThreatType.XSS_IN_AGENT_DESKTOP,
                    severity="critical",
                    session_id=session_id,
                    agent_id=agent_id,
                    evidence={
                        "reason": "Origin not in whitelist (possible XSS or iframe attack)",
                        "origin": origin_header,
                        "whitelist": self._policy.origin_whitelist,
                    },
                ))

        # Browser fingerprint leak detection
        if user_agent and self._contains_sensitive_info(user_agent):
            threats.append(WebRTCThreatEvent(
                threat_type=WebRTCThreatType.BROWSER_FINGERPRINT_LEAK,
                severity="low",
                session_id=session_id,
                agent_id=agent_id,
                evidence={
                    "reason": "User-Agent contains detailed fingerprint information",
                    "user_agent_length": len(user_agent),
                },
            ))

        if threats:
            self._stats["total_threats_detected"] += len(threats)
            self._threat_history.extend(threats)

        return threats

    @staticmethod
    def _contains_sensitive_info(user_agent: str) -> bool:
        """Check if User-Agent contains excessive fingerprint details."""
        sensitive_patterns = [
            r"CPU\s+(iPhone|iPad)\s+OS\s+\d+",
            r"Build/\w+",
            r"OPR/\d+",
            r"Edg/\d+",
            r"CriOS/\d+",
        ]
        matches = sum(1 for p in sensitive_patterns if re.search(p, user_agent))
        return matches >= 2

    # ------------------------------------------------------------------
    # Policy enforcement
    # ------------------------------------------------------------------

    def enforce_policy(self, threat: WebRTCThreatEvent) -> WebRTCAction:
        """
        Enforce the security policy for a detected threat.

        Determines and applies the appropriate action based on the
        threat severity and the configured security level.

        Args:
            threat: The threat event to respond to.

        Returns:
            The action that was taken.
        """
        action: WebRTCAction

        if threat.severity == "critical":
            action = WebRTCAction.TERMINATE_SESSION
            self._stats["total_sessions_terminated"] += 1
        elif threat.severity == "high":
            if self._policy.security_level in (
                WebRTCSecurityLevel.PQC_HYBRID,
                WebRTCSecurityLevel.PQC_ONLY,
            ):
                action = WebRTCAction.BLOCK_CONNECTION
            else:
                action = WebRTCAction.FORCE_RENEGOTIATION
        elif threat.severity == "medium":
            action = WebRTCAction.ALERT
        else:
            action = WebRTCAction.LOG

        threat.action_taken = action

        logger.info(
            "Enforced policy: threat=%s severity=%s action=%s session=%s",
            threat.threat_type.name,
            threat.severity,
            action.name,
            threat.session_id,
        )

        if action in (WebRTCAction.TERMINATE_SESSION, WebRTCAction.BLOCK_CONNECTION):
            if self._alert_callback:
                self._alert_callback(threat)

        return action

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_session_report(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a security report for a session or all sessions.

        Args:
            session_id: If provided, report on that specific session.
                        If None, generate aggregate statistics.

        Returns:
            Dictionary containing the report data.
        """
        if session_id:
            session_data = self._sessions.get(session_id, {})
            session_threats = [
                t.to_dict() for t in self._threat_history
                if t.session_id == session_id
            ]
            return {
                "report_type": "session",
                "session_id": session_id,
                "session_data": session_data,
                "threats": session_threats,
                "threat_count": len(session_threats),
                "generated_at": datetime.utcnow().isoformat(),
            }

        return {
            "report_type": "aggregate",
            "statistics": dict(self._stats),
            "total_active_sessions": len(self._sessions),
            "total_threat_events": len(self._threat_history),
            "policy": self._policy.to_dict(),
            "generated_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create_pqc_policy(cls) -> WebRTCSecurityPolicy:
        """
        Create a PQC-hybrid security policy.

        Suitable for most BPO deployments that want quantum
        resistance while maintaining classical algorithm fallback.
        """
        return WebRTCSecurityPolicy(
            security_level=WebRTCSecurityLevel.PQC_HYBRID,
            filter_private_candidates=True,
            require_dtls_1_3=True,
            allowed_codecs=["opus", "VP8", "VP9", "H264"],
            pqc_dtls_enabled=True,
            origin_whitelist=[],
            max_data_channels=2,
            sdp_validation_strict=True,
        )
