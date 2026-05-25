"""
QBITEL - BPO Discovery Profile

AI-powered protocol discovery profile tuned for BPO/call center environments.
Provides feature extraction, classification, and fingerprinting capabilities
specialized for telephony, terminal, and contact center protocols.

Integrates with:
- ai_engine.discovery.protocol_discovery_orchestrator: DiscoveryRequest/Result
- ai_engine.discovery.statistical_analyzer: StructuralFeatures
- ai_engine.discovery.protocol_classifier: ProtocolClassifier
- ai_engine.discovery.protocol_signatures: ProtocolSignatureDatabase

BPO Protocol Categories:
- Voice/Telephony: SIP, SDP, RTP/SRTP, SRTP-PQC
- Terminal: TN3270e, TN5250
- Signaling: DTMF, SS7/ISUP, MGCP, H.323
- Contact Center: CTI (CSTA, TSAPI, Finesse), TAPI, JTAPI
- IVR: VoiceXML, MRCP, CCXML
- Integration: SMPP, XMPP
"""

import asyncio
import logging
import re
import struct
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics
BPO_DISCOVERY_REQUESTS = Counter(
    "qbitel_bpo_discovery_requests_total",
    "Total BPO protocol discovery requests",
    ["protocol_type", "result"],
)
BPO_DISCOVERY_DURATION = Histogram(
    "qbitel_bpo_discovery_duration_seconds",
    "BPO discovery processing duration",
    ["phase"],
)
BPO_DISCOVERY_CONFIDENCE = Histogram(
    "qbitel_bpo_discovery_confidence",
    "BPO discovery confidence distribution",
    ["protocol_type"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)


class BPOProtocolType(str, Enum):
    """BPO-specific protocol types."""

    # Voice/Telephony
    SIP = "sip"
    SIP_PQC_TLS = "sip_pqc_tls"
    SDP = "sdp"
    RTP = "rtp"
    SRTP = "srtp"
    SRTP_PQC = "srtp_pqc"

    # Terminal
    TN3270E = "tn3270e"
    TN5250 = "tn5250"

    # Signaling
    DTMF = "dtmf"
    SS7_ISUP = "ss7_isup"
    MGCP = "mgcp"
    H323 = "h323"

    # Contact Center
    CTI_CSTA = "cti_csta"
    CTI_TSAPI = "cti_tsapi"
    CTI_FINESSE = "cti_finesse"
    TAPI = "tapi"
    JTAPI = "jtapi"

    # IVR
    IVR_VXML = "ivr_vxml"
    MRCP = "mrcp"
    CCXML = "ccxml"

    # Integration
    SMPP = "smpp"
    XMPP = "xmpp"

    # Unknown
    UNKNOWN = "unknown"


class DiscoveryConfidence(str, Enum):
    """Confidence levels for discovery results."""
    HIGH = "high"       # >= 0.85 - Definitive match
    MEDIUM = "medium"   # >= 0.65 - Probable match
    LOW = "low"         # >= 0.40 - Possible match
    UNCERTAIN = "uncertain"  # < 0.40


@dataclass
class BPODiscoveryConfig:
    """Configuration for BPO protocol discovery."""

    # Confidence thresholds
    high_confidence_threshold: float = 0.85
    medium_confidence_threshold: float = 0.65
    low_confidence_threshold: float = 0.40

    # Feature extraction settings
    enable_deep_packet_inspection: bool = True
    enable_voice_pattern_analysis: bool = True
    enable_signaling_analysis: bool = True
    enable_terminal_detection: bool = True

    # Performance settings
    max_samples_per_session: int = 1000
    analysis_timeout_seconds: float = 30.0
    parallel_analysis: bool = True
    cache_results: bool = True

    # PBX vendor fingerprinting
    enable_vendor_fingerprinting: bool = True
    known_vendors: List[str] = field(default_factory=lambda: [
        "Avaya", "Cisco", "Genesys", "Mitel", "Asterisk",
        "FreePBX", "3CX", "RingCentral", "Five9", "NICE",
    ])

    # Port mappings for protocol hints
    port_protocol_hints: Dict[int, str] = field(default_factory=lambda: {
        5060: "sip",
        5061: "sip_pqc_tls",
        5004: "rtp",
        5005: "srtp",
        23: "tn3270e",
        992: "tn3270e",  # TN3270e over TLS
        2427: "mgcp",
        1720: "h323",
        2775: "smpp",
        5222: "xmpp",
        6060: "mrcp",
    })


@dataclass
class BPOFeatureVector:
    """Feature vector extracted from BPO protocol traffic."""

    # Statistical features
    entropy: float = 0.0
    ascii_ratio: float = 0.0
    null_ratio: float = 0.0
    printable_ratio: float = 0.0
    avg_message_size: float = 0.0

    # SIP-specific features
    sip_method_detected: bool = False
    sip_headers_count: int = 0
    sip_via_hops: int = 0
    sdp_body_present: bool = False
    sip_auth_type: Optional[str] = None

    # Voice/Media features
    rtp_payload_type: Optional[int] = None
    rtp_ssrc_count: int = 0
    dtmf_events_detected: int = 0
    codec_types: List[str] = field(default_factory=list)
    media_encryption: Optional[str] = None

    # Terminal features
    tn3270_data_stream: bool = False
    tn3270_function_keys: int = 0
    ebcdic_ratio: float = 0.0

    # CTI features
    csta_operations: List[str] = field(default_factory=list)
    xml_structure_detected: bool = False
    soap_envelope_detected: bool = False
    rest_api_detected: bool = False

    # IVR features
    vxml_tags_detected: int = 0
    mrcp_methods: List[str] = field(default_factory=list)
    grammar_references: int = 0

    # Vendor hints
    vendor_hints: List[str] = field(default_factory=list)
    user_agent_string: Optional[str] = None

    # PQC indicators
    pqc_key_exchange_detected: bool = False
    hybrid_tls_detected: bool = False
    ml_kem_detected: bool = False


@dataclass
class BPODiscoveryResult:
    """Result of BPO protocol discovery."""

    # Primary identification
    protocol_type: BPOProtocolType
    confidence: float
    confidence_level: DiscoveryConfidence

    # Feature vector
    features: BPOFeatureVector

    # Classification details
    match_reasons: List[str] = field(default_factory=list)
    alternative_protocols: List[Tuple[str, float]] = field(default_factory=list)

    # Vendor fingerprinting
    detected_vendor: Optional[str] = None
    vendor_confidence: float = 0.0
    vendor_details: Dict[str, Any] = field(default_factory=dict)

    # Security assessment
    encryption_detected: bool = False
    pqc_capable: bool = False
    security_concerns: List[str] = field(default_factory=list)

    # Metadata
    analysis_duration: float = 0.0
    samples_analyzed: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class BPOFeatureExtractor:
    """
    Feature extractor specialized for BPO/call center protocol traffic.

    Extracts protocol-specific features from network traffic samples
    to enable accurate classification of telephony, terminal, and
    contact center protocols.
    """

    # SIP method patterns
    SIP_METHODS = {
        b"INVITE", b"ACK", b"BYE", b"CANCEL", b"REGISTER",
        b"OPTIONS", b"PRACK", b"SUBSCRIBE", b"NOTIFY", b"PUBLISH",
        b"INFO", b"REFER", b"MESSAGE", b"UPDATE",
    }

    # SIP response pattern
    SIP_RESPONSE_RE = re.compile(rb"^SIP/2\.0\s+\d{3}\s+")

    # SIP header patterns
    SIP_HEADERS = {
        b"Via:", b"From:", b"To:", b"Call-ID:", b"CSeq:",
        b"Contact:", b"Content-Type:", b"Content-Length:",
        b"Max-Forwards:", b"Route:", b"Record-Route:",
        b"Authorization:", b"WWW-Authenticate:",
        b"Proxy-Authorization:", b"Proxy-Authenticate:",
    }

    # RTP header constants
    RTP_VERSION = 2
    RTP_MIN_HEADER_SIZE = 12

    # MGCP method patterns
    MGCP_VERBS = {b"CRCX", b"MDCX", b"DLCX", b"RQNT", b"NTFY", b"AUEP", b"AUCX", b"RSIP"}

    # CSTA XML patterns
    CSTA_PATTERNS = [
        re.compile(rb"<(?:Make|Answer|Clear|Hold|Retrieve|Transfer|Conference)Call"),
        re.compile(rb"<(?:MonitorStart|MonitorStop|SnapshotDevice|GetSwitchingFunction)"),
        re.compile(rb"xmlns[=:].*csta", re.IGNORECASE),
    ]

    # VoiceXML patterns
    VXML_PATTERNS = [
        re.compile(rb"<vxml\b", re.IGNORECASE),
        re.compile(rb"<(?:form|menu|field|block|filled|grammar|prompt)\b"),
        re.compile(rb"application/voicexml", re.IGNORECASE),
    ]

    # MRCP patterns
    MRCP_METHODS_RE = re.compile(
        rb"^(?:SPEAK|RECOGNIZE|DEFINE-GRAMMAR|SET-PARAMS|GET-PARAMS|STOP|PAUSE|RESUME)\b"
    )

    # TN3270e constants
    TN3270_EOR = b"\xff\xef"  # End of Record
    TN3270_IAC = 0xFF
    TN3270_COMMANDS = {0xF1, 0xF5, 0xF6, 0x7E, 0x6F}  # Write, Erase/Write, etc.

    # Vendor User-Agent patterns
    VENDOR_PATTERNS = {
        "Avaya": [re.compile(rb"Avaya", re.IGNORECASE), re.compile(rb"CM\d+\.\d+")],
        "Cisco": [re.compile(rb"Cisco", re.IGNORECASE), re.compile(rb"CUCM|Unified CM")],
        "Genesys": [re.compile(rb"Genesys", re.IGNORECASE), re.compile(rb"PureConnect|GCloud")],
        "Mitel": [re.compile(rb"Mitel", re.IGNORECASE), re.compile(rb"MiVoice")],
        "Asterisk": [re.compile(rb"Asterisk", re.IGNORECASE), re.compile(rb"FPBX|FreePBX")],
    }

    def __init__(self, config: Optional[BPODiscoveryConfig] = None):
        """Initialize the BPO feature extractor."""
        self.config = config or BPODiscoveryConfig()
        self.logger = logging.getLogger(f"{__name__}.BPOFeatureExtractor")

    def extract_features(self, samples: List[bytes]) -> BPOFeatureVector:
        """
        Extract BPO-specific feature vector from protocol samples.

        Args:
            samples: List of raw protocol message bytes

        Returns:
            BPOFeatureVector with extracted features
        """
        features = BPOFeatureVector()

        if not samples:
            return features

        # Statistical features
        features.entropy = self._calculate_entropy(samples)
        features.ascii_ratio = self._calculate_ascii_ratio(samples)
        features.null_ratio = self._calculate_null_ratio(samples)
        features.printable_ratio = self._calculate_printable_ratio(samples)
        features.avg_message_size = sum(len(s) for s in samples) / len(samples)

        # Protocol-specific features
        if self.config.enable_deep_packet_inspection:
            self._extract_sip_features(samples, features)
            self._extract_cti_features(samples, features)
            self._extract_ivr_features(samples, features)

        if self.config.enable_voice_pattern_analysis:
            self._extract_rtp_features(samples, features)
            self._extract_dtmf_features(samples, features)

        if self.config.enable_signaling_analysis:
            self._extract_mgcp_features(samples, features)
            self._extract_ss7_features(samples, features)

        if self.config.enable_terminal_detection:
            self._extract_tn3270_features(samples, features)

        # Vendor fingerprinting
        if self.config.enable_vendor_fingerprinting:
            self._extract_vendor_hints(samples, features)

        # PQC detection
        self._extract_pqc_features(samples, features)

        return features

    def _calculate_entropy(self, samples: List[bytes]) -> float:
        """Calculate Shannon entropy across all samples."""
        import math
        byte_counts = [0] * 256
        total = 0
        for sample in samples:
            for byte in sample:
                byte_counts[byte] += 1
                total += 1
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in byte_counts:
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        return entropy

    def _calculate_ascii_ratio(self, samples: List[bytes]) -> float:
        """Calculate ratio of ASCII printable bytes."""
        total = sum(len(s) for s in samples)
        if total == 0:
            return 0.0
        ascii_count = sum(
            1 for s in samples for b in s if 32 <= b <= 126
        )
        return ascii_count / total

    def _calculate_null_ratio(self, samples: List[bytes]) -> float:
        """Calculate ratio of null bytes."""
        total = sum(len(s) for s in samples)
        if total == 0:
            return 0.0
        null_count = sum(1 for s in samples for b in s if b == 0)
        return null_count / total

    def _calculate_printable_ratio(self, samples: List[bytes]) -> float:
        """Calculate ratio of printable bytes (including whitespace)."""
        total = sum(len(s) for s in samples)
        if total == 0:
            return 0.0
        printable_count = sum(
            1 for s in samples for b in s
            if 32 <= b <= 126 or b in (9, 10, 13)
        )
        return printable_count / total

    def _extract_sip_features(self, samples: List[bytes], features: BPOFeatureVector) -> None:
        """Extract SIP-specific features."""
        for sample in samples:
            # Check for SIP methods
            first_line = sample.split(b"\r\n", 1)[0] if b"\r\n" in sample else sample[:64]
            for method in self.SIP_METHODS:
                if first_line.startswith(method + b" "):
                    features.sip_method_detected = True
                    break

            # Check for SIP response
            if self.SIP_RESPONSE_RE.match(first_line):
                features.sip_method_detected = True

            # Count SIP headers
            for header in self.SIP_HEADERS:
                if header in sample:
                    features.sip_headers_count += 1

            # Count Via hops
            features.sip_via_hops += sample.count(b"Via:")

            # Check for SDP body
            if b"v=0\r\n" in sample and b"m=audio" in sample:
                features.sdp_body_present = True

            # Check auth type
            if b"Digest " in sample:
                features.sip_auth_type = "digest"
            elif b"Bearer " in sample:
                features.sip_auth_type = "bearer"

            # Extract User-Agent
            ua_match = re.search(rb"User-Agent:\s*(.+?)(?:\r\n|\r|\n)", sample)
            if ua_match:
                features.user_agent_string = ua_match.group(1).decode("utf-8", errors="replace")

    def _extract_rtp_features(self, samples: List[bytes], features: BPOFeatureVector) -> None:
        """Extract RTP/SRTP features from media packets."""
        ssrc_set: Set[int] = set()

        for sample in samples:
            if len(sample) < self.RTP_MIN_HEADER_SIZE:
                continue

            # Check RTP version (first 2 bits)
            version = (sample[0] >> 6) & 0x03
            if version != self.RTP_VERSION:
                continue

            # Extract payload type (7 bits after marker bit)
            payload_type = sample[1] & 0x7F
            if features.rtp_payload_type is None:
                features.rtp_payload_type = payload_type

            # Determine codec
            codec = self._rtp_payload_to_codec(payload_type)
            if codec and codec not in features.codec_types:
                features.codec_types.append(codec)

            # Extract SSRC
            ssrc = struct.unpack("!I", sample[8:12])[0]
            ssrc_set.add(ssrc)

            # Check for SRTP (10-byte auth tag at end is a hint)
            if len(sample) > 172:  # 160 bytes G.711 + 12 header + auth tag
                features.media_encryption = "srtp"

        features.rtp_ssrc_count = len(ssrc_set)

    def _rtp_payload_to_codec(self, pt: int) -> Optional[str]:
        """Map RTP payload type to codec name."""
        codec_map = {
            0: "G.711 μ-law (PCMU)",
            3: "GSM",
            4: "G.723",
            8: "G.711 A-law (PCMA)",
            9: "G.722",
            18: "G.729",
            96: "Dynamic (likely Opus)",
            97: "Dynamic (likely H.264)",
            101: "telephone-event (DTMF)",
        }
        return codec_map.get(pt)

    def _extract_dtmf_features(self, samples: List[bytes], features: BPOFeatureVector) -> None:
        """Extract DTMF event features (RFC 4733)."""
        for sample in samples:
            if len(sample) < 16:
                continue
            # Check for RTP with payload type 101 (telephone-event)
            if (sample[0] >> 6) & 0x03 == 2:
                pt = sample[1] & 0x7F
                if pt == 101:
                    features.dtmf_events_detected += 1

    def _extract_mgcp_features(self, samples: List[bytes], features: BPOFeatureVector) -> None:
        """Extract MGCP features."""
        for sample in samples:
            first_word = sample.split(b" ", 1)[0] if b" " in sample else b""
            if first_word in self.MGCP_VERBS:
                if "mgcp" not in [r.lower() for r in features.csta_operations]:
                    features.csta_operations.append("MGCP")

    def _extract_ss7_features(self, samples: List[bytes], features: BPOFeatureVector) -> None:
        """Extract SS7/ISUP features."""
        for sample in samples:
            # SS7 MTP3 has a specific header structure
            if len(sample) >= 5:
                # Check for ISUP message indicators
                si = sample[0] & 0x0F  # Service Indicator
                if si == 5:  # ISUP
                    features.csta_operations.append("SS7_ISUP")

    def _extract_tn3270_features(self, samples: List[bytes], features: BPOFeatureVector) -> None:
        """Extract TN3270e terminal features."""
        for sample in samples:
            # Check for TN3270 end-of-record marker
            if self.TN3270_EOR in sample:
                features.tn3270_data_stream = True

            # Check for TN3270 commands
            if len(sample) >= 2 and sample[0] == self.TN3270_IAC:
                if sample[1] in self.TN3270_COMMANDS:
                    features.tn3270_data_stream = True

            # Check for EBCDIC content
            ebcdic_bytes = sum(
                1 for b in sample if 0x40 <= b <= 0xFE and b not in range(0x7F, 0xA0)
            )
            if len(sample) > 0:
                features.ebcdic_ratio = max(features.ebcdic_ratio, ebcdic_bytes / len(sample))

            # Count function key indicators (AID bytes)
            aid_bytes = {0x7D, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
                         0xF8, 0xF9, 0xC1, 0xC2, 0xC3, 0xC4}
            for b in sample:
                if b in aid_bytes:
                    features.tn3270_function_keys += 1

    def _extract_cti_features(self, samples: List[bytes], features: BPOFeatureVector) -> None:
        """Extract CTI protocol features (CSTA, TSAPI, Finesse)."""
        for sample in samples:
            # CSTA XML patterns
            for pattern in self.CSTA_PATTERNS:
                if pattern.search(sample):
                    features.csta_operations.append("CSTA")
                    features.xml_structure_detected = True
                    break

            # Cisco Finesse REST API
            if b"finesse/api" in sample or b"<Finesse" in sample:
                features.rest_api_detected = True
                features.csta_operations.append("Finesse")

            # SOAP envelope detection (Avaya TSAPI uses SOAP)
            if b"<soap:Envelope" in sample or b"<SOAP-ENV:Envelope" in sample:
                features.soap_envelope_detected = True
                features.csta_operations.append("TSAPI")

            # Generic XML detection
            if sample.startswith(b"<?xml") or b"</" in sample[:100]:
                features.xml_structure_detected = True

    def _extract_ivr_features(self, samples: List[bytes], features: BPOFeatureVector) -> None:
        """Extract IVR protocol features (VoiceXML, MRCP, CCXML)."""
        for sample in samples:
            # VoiceXML detection
            for pattern in self.VXML_PATTERNS:
                if pattern.search(sample):
                    features.vxml_tags_detected += 1

            # MRCP method detection
            if self.MRCP_METHODS_RE.match(sample):
                method = sample.split(b" ", 1)[0].decode("utf-8", errors="replace")
                features.mrcp_methods.append(method)

            # CCXML detection
            if b"<ccxml" in sample.lower() or b"<eventprocessor" in sample.lower():
                features.vxml_tags_detected += 1

            # Grammar references
            if b"<grammar" in sample or b"application/srgs" in sample:
                features.grammar_references += 1

    def _extract_vendor_hints(self, samples: List[bytes], features: BPOFeatureVector) -> None:
        """Extract vendor fingerprinting hints."""
        for sample in samples:
            for vendor, patterns in self.VENDOR_PATTERNS.items():
                for pattern in patterns:
                    if pattern.search(sample):
                        if vendor not in features.vendor_hints:
                            features.vendor_hints.append(vendor)
                        break

    def _extract_pqc_features(self, samples: List[bytes], features: BPOFeatureVector) -> None:
        """Extract post-quantum cryptography indicators."""
        for sample in samples:
            # ML-KEM key exchange indicators
            if b"ML-KEM" in sample or b"MLKEM" in sample or b"kyber" in sample.lower():
                features.pqc_key_exchange_detected = True
                features.ml_kem_detected = True

            # Hybrid TLS indicators
            if b"hybrid" in sample.lower() and (b"x25519" in sample.lower() or b"mlkem" in sample.lower()):
                features.hybrid_tls_detected = True

            # PQC cipher suite indicators in SIP
            if b"algorithm=ML-KEM" in sample or b"a=crypto:.*PQC" in sample:
                features.pqc_key_exchange_detected = True


class BPOProtocolClassifier:
    """
    BPO protocol classifier using feature vectors.

    Classifies protocol traffic into specific BPO protocol types
    using a combination of rule-based and statistical approaches.
    """

    def __init__(self, config: Optional[BPODiscoveryConfig] = None):
        """Initialize the BPO protocol classifier."""
        self.config = config or BPODiscoveryConfig()
        self.logger = logging.getLogger(f"{__name__}.BPOProtocolClassifier")

    def classify(self, features: BPOFeatureVector) -> BPODiscoveryResult:
        """
        Classify protocol based on extracted features.

        Args:
            features: Extracted BPO feature vector

        Returns:
            BPODiscoveryResult with classification
        """
        candidates: List[Tuple[BPOProtocolType, float, List[str]]] = []

        # Check each protocol type
        candidates.append(self._score_sip(features))
        candidates.append(self._score_rtp(features))
        candidates.append(self._score_tn3270(features))
        candidates.append(self._score_cti(features))
        candidates.append(self._score_ivr(features))
        candidates.append(self._score_mgcp(features))
        candidates.append(self._score_smpp(features))

        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Best match
        best = candidates[0]
        protocol_type, confidence, reasons = best

        # Determine confidence level
        if confidence >= self.config.high_confidence_threshold:
            level = DiscoveryConfidence.HIGH
        elif confidence >= self.config.medium_confidence_threshold:
            level = DiscoveryConfidence.MEDIUM
        elif confidence >= self.config.low_confidence_threshold:
            level = DiscoveryConfidence.LOW
        else:
            level = DiscoveryConfidence.UNCERTAIN
            protocol_type = BPOProtocolType.UNKNOWN

        # Build alternative list
        alternatives = [
            (c[0].value, c[1]) for c in candidates[1:4] if c[1] > 0.1
        ]

        # Vendor detection
        detected_vendor = None
        vendor_confidence = 0.0
        if features.vendor_hints:
            detected_vendor = features.vendor_hints[0]
            vendor_confidence = min(0.9, 0.5 + 0.2 * len(features.vendor_hints))

        # Security assessment
        encryption_detected = (
            features.media_encryption is not None
            or features.pqc_key_exchange_detected
            or features.hybrid_tls_detected
        )
        security_concerns = self._assess_security(features)

        return BPODiscoveryResult(
            protocol_type=protocol_type,
            confidence=confidence,
            confidence_level=level,
            features=features,
            match_reasons=reasons,
            alternative_protocols=alternatives,
            detected_vendor=detected_vendor,
            vendor_confidence=vendor_confidence,
            encryption_detected=encryption_detected,
            pqc_capable=features.pqc_key_exchange_detected,
            security_concerns=security_concerns,
        )

    def _score_sip(self, features: BPOFeatureVector) -> Tuple[BPOProtocolType, float, List[str]]:
        """Score SIP protocol match."""
        score = 0.0
        reasons = []

        if features.sip_method_detected:
            score += 0.45
            reasons.append("SIP method or response detected")
        if features.sip_headers_count >= 3:
            score += 0.25
            reasons.append(f"SIP headers detected ({features.sip_headers_count})")
        if features.sip_via_hops > 0:
            score += 0.10
            reasons.append(f"Via hops: {features.sip_via_hops}")
        if features.sdp_body_present:
            score += 0.10
            reasons.append("SDP body present")
        if features.ascii_ratio > 0.8:
            score += 0.05
            reasons.append("High ASCII ratio (text protocol)")
        if features.user_agent_string:
            score += 0.05
            reasons.append(f"User-Agent: {features.user_agent_string}")

        # Determine PQC variant
        protocol_type = BPOProtocolType.SIP
        if features.pqc_key_exchange_detected or features.hybrid_tls_detected:
            protocol_type = BPOProtocolType.SIP_PQC_TLS
            reasons.append("PQC/hybrid TLS indicators detected")

        return (protocol_type, min(score, 1.0), reasons)

    def _score_rtp(self, features: BPOFeatureVector) -> Tuple[BPOProtocolType, float, List[str]]:
        """Score RTP/SRTP protocol match."""
        score = 0.0
        reasons = []

        if features.rtp_payload_type is not None:
            score += 0.40
            reasons.append(f"RTP payload type: {features.rtp_payload_type}")
        if features.rtp_ssrc_count > 0:
            score += 0.20
            reasons.append(f"SSRC streams: {features.rtp_ssrc_count}")
        if features.codec_types:
            score += 0.15
            reasons.append(f"Codecs: {', '.join(features.codec_types)}")
        if features.dtmf_events_detected > 0:
            score += 0.10
            reasons.append(f"DTMF events: {features.dtmf_events_detected}")
        if features.avg_message_size > 100 and features.avg_message_size < 300:
            score += 0.10
            reasons.append("Typical voice packet size")
        if features.entropy > 6.0:
            score += 0.05
            reasons.append("High entropy (encoded media)")

        protocol_type = BPOProtocolType.RTP
        if features.media_encryption == "srtp":
            protocol_type = BPOProtocolType.SRTP
            reasons.append("SRTP encryption detected")
        if features.pqc_key_exchange_detected:
            protocol_type = BPOProtocolType.SRTP_PQC
            reasons.append("PQC key exchange for SRTP")

        return (protocol_type, min(score, 1.0), reasons)

    def _score_tn3270(self, features: BPOFeatureVector) -> Tuple[BPOProtocolType, float, List[str]]:
        """Score TN3270e protocol match."""
        score = 0.0
        reasons = []

        if features.tn3270_data_stream:
            score += 0.50
            reasons.append("TN3270 data stream detected")
        if features.ebcdic_ratio > 0.3:
            score += 0.25
            reasons.append(f"EBCDIC content ratio: {features.ebcdic_ratio:.2f}")
        if features.tn3270_function_keys > 0:
            score += 0.15
            reasons.append(f"Function key AID bytes: {features.tn3270_function_keys}")
        if features.null_ratio > 0.1:
            score += 0.05
            reasons.append("Null byte padding (3270 buffers)")
        if not features.sip_method_detected and features.ascii_ratio < 0.5:
            score += 0.05
            reasons.append("Non-ASCII, non-SIP traffic")

        return (BPOProtocolType.TN3270E, min(score, 1.0), reasons)

    def _score_cti(self, features: BPOFeatureVector) -> Tuple[BPOProtocolType, float, List[str]]:
        """Score CTI protocol match."""
        score = 0.0
        reasons = []

        if "CSTA" in features.csta_operations:
            score += 0.50
            reasons.append("CSTA XML operations detected")
            protocol_type = BPOProtocolType.CTI_CSTA
        elif "TSAPI" in features.csta_operations:
            score += 0.45
            reasons.append("TSAPI SOAP operations detected")
            protocol_type = BPOProtocolType.CTI_TSAPI
        elif "Finesse" in features.csta_operations:
            score += 0.45
            reasons.append("Cisco Finesse REST API detected")
            protocol_type = BPOProtocolType.CTI_FINESSE
        else:
            protocol_type = BPOProtocolType.CTI_CSTA

        if features.xml_structure_detected:
            score += 0.20
            reasons.append("XML message structure")
        if features.soap_envelope_detected:
            score += 0.15
            reasons.append("SOAP envelope detected")
        if features.rest_api_detected:
            score += 0.15
            reasons.append("REST API detected")

        return (protocol_type, min(score, 1.0), reasons)

    def _score_ivr(self, features: BPOFeatureVector) -> Tuple[BPOProtocolType, float, List[str]]:
        """Score IVR protocol match."""
        score = 0.0
        reasons = []

        if features.vxml_tags_detected > 0:
            score += 0.45
            reasons.append(f"VoiceXML tags: {features.vxml_tags_detected}")
            protocol_type = BPOProtocolType.IVR_VXML
        elif features.mrcp_methods:
            score += 0.45
            reasons.append(f"MRCP methods: {', '.join(features.mrcp_methods)}")
            protocol_type = BPOProtocolType.MRCP
        else:
            protocol_type = BPOProtocolType.IVR_VXML

        if features.grammar_references > 0:
            score += 0.20
            reasons.append(f"Grammar references: {features.grammar_references}")
        if features.xml_structure_detected:
            score += 0.15
            reasons.append("XML structure")

        return (protocol_type, min(score, 1.0), reasons)

    def _score_mgcp(self, features: BPOFeatureVector) -> Tuple[BPOProtocolType, float, List[str]]:
        """Score MGCP protocol match."""
        score = 0.0
        reasons = []

        if "MGCP" in features.csta_operations:
            score += 0.50
            reasons.append("MGCP verb detected")
        if features.ascii_ratio > 0.9:
            score += 0.10
            reasons.append("High ASCII ratio (text protocol)")
        if features.sdp_body_present:
            score += 0.15
            reasons.append("SDP body (media description)")

        return (BPOProtocolType.MGCP, min(score, 1.0), reasons)

    def _score_smpp(self, features: BPOFeatureVector) -> Tuple[BPOProtocolType, float, List[str]]:
        """Score SMPP protocol match."""
        score = 0.0
        reasons = []

        # SMPP is binary with length-prefixed PDUs
        if features.ascii_ratio < 0.3 and features.entropy < 5.0:
            score += 0.15
            reasons.append("Binary protocol characteristics")
        # SMPP messages typically 100-500 bytes
        if 50 < features.avg_message_size < 600:
            score += 0.10
            reasons.append("Typical SMPP message size")

        return (BPOProtocolType.SMPP, min(score, 1.0), reasons)

    def _assess_security(self, features: BPOFeatureVector) -> List[str]:
        """Assess security concerns from features."""
        concerns = []

        if features.sip_method_detected and not features.sip_auth_type:
            concerns.append("SIP traffic without authentication detected")
        if features.rtp_payload_type is not None and features.media_encryption is None:
            concerns.append("Unencrypted RTP media stream detected")
        if features.dtmf_events_detected > 0 and features.media_encryption is None:
            concerns.append("DTMF digits transmitted without encryption (PCI-DSS risk)")
        if features.tn3270_data_stream and features.ebcdic_ratio > 0:
            concerns.append("TN3270e terminal data may contain sensitive mainframe data")
        if features.sip_method_detected and not features.pqc_key_exchange_detected:
            concerns.append("SIP signaling not using post-quantum key exchange")

        return concerns


class BPODiscoveryProfile:
    """
    High-level BPO discovery profile that orchestrates feature extraction,
    classification, and reporting for BPO protocol traffic.

    Usage:
        profile = BPODiscoveryProfile()
        result = await profile.discover(samples, source_port=5060)
    """

    def __init__(self, config: Optional[BPODiscoveryConfig] = None):
        """Initialize the BPO discovery profile."""
        self.config = config or BPODiscoveryConfig()
        self.extractor = BPOFeatureExtractor(self.config)
        self.classifier = BPOProtocolClassifier(self.config)
        self.logger = logging.getLogger(f"{__name__}.BPODiscoveryProfile")
        self._discovery_cache: Dict[str, BPODiscoveryResult] = {}

    async def discover(
        self,
        samples: List[bytes],
        source_port: Optional[int] = None,
        destination_port: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BPODiscoveryResult:
        """
        Perform BPO protocol discovery on traffic samples.

        Args:
            samples: Raw protocol message bytes
            source_port: Source port (for protocol hints)
            destination_port: Destination port (for protocol hints)
            metadata: Additional context metadata

        Returns:
            BPODiscoveryResult with protocol classification
        """
        import time
        start_time = time.time()

        # Check cache
        cache_key = self._compute_cache_key(samples)
        if self.config.cache_results and cache_key in self._discovery_cache:
            self.logger.debug(f"Cache hit for discovery: {cache_key[:16]}")
            return self._discovery_cache[cache_key]

        # Limit samples
        analysis_samples = samples[:self.config.max_samples_per_session]

        # Phase 1: Feature extraction
        with BPO_DISCOVERY_DURATION.labels(phase="feature_extraction").time():
            features = self.extractor.extract_features(analysis_samples)

        # Phase 2: Apply port hints
        if destination_port and destination_port in self.config.port_protocol_hints:
            hint = self.config.port_protocol_hints[destination_port]
            self.logger.debug(f"Port hint for {destination_port}: {hint}")

        # Phase 3: Classification
        with BPO_DISCOVERY_DURATION.labels(phase="classification").time():
            result = self.classifier.classify(features)

        # Enrich with metadata
        result.analysis_duration = time.time() - start_time
        result.samples_analyzed = len(analysis_samples)

        # Update metrics
        BPO_DISCOVERY_REQUESTS.labels(
            protocol_type=result.protocol_type.value,
            result=result.confidence_level.value,
        ).inc()
        BPO_DISCOVERY_CONFIDENCE.labels(
            protocol_type=result.protocol_type.value,
        ).observe(result.confidence)

        # Cache result
        if self.config.cache_results:
            self._discovery_cache[cache_key] = result
            # Evict old entries
            if len(self._discovery_cache) > 10000:
                oldest_keys = list(self._discovery_cache.keys())[:5000]
                for k in oldest_keys:
                    del self._discovery_cache[k]

        self.logger.info(
            f"BPO discovery: {result.protocol_type.value} "
            f"(confidence={result.confidence:.2f}, level={result.confidence_level.value}) "
            f"in {result.analysis_duration:.3f}s"
        )

        return result

    def _compute_cache_key(self, samples: List[bytes]) -> str:
        """Compute cache key from samples."""
        h = hashlib.sha256()
        for s in samples[:10]:
            h.update(s[:256])
        return h.hexdigest()

    def get_supported_protocols(self) -> List[str]:
        """Get list of supported BPO protocols."""
        return [p.value for p in BPOProtocolType if p != BPOProtocolType.UNKNOWN]

    def clear_cache(self) -> int:
        """Clear discovery cache. Returns number of entries cleared."""
        count = len(self._discovery_cache)
        self._discovery_cache.clear()
        return count
