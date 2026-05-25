"""
QBITEL - BPO Protocol Signatures

Pre-defined protocol signatures for BPO/call center protocol identification.
These signatures are registered with the platform's ProtocolSignatureDatabase
to enable rapid detection of telephony, terminal, and contact center protocols.

Signature coverage:
- SIP/SDP/SIP-PQC-TLS (voice signaling)
- RTP/SRTP/SRTP-PQC (media transport)
- TN3270e (IBM terminal emulation)
- CTI-CSTA/TSAPI/Finesse (computer telephony)
- IVR-VXML/MRCP/CCXML (interactive voice response)
- MGCP (media gateway control)
- H.323 (multimedia communications)
- SMPP (short message)
- SS7/ISUP (legacy signaling)
- DTMF-RFC4733 (telephone events)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class BPOProtocolCategory(str, Enum):
    """BPO-specific protocol categories."""

    VOICE_SIGNALING = "voice_signaling"
    MEDIA_TRANSPORT = "media_transport"
    TERMINAL_EMULATION = "terminal_emulation"
    CONTACT_CENTER = "contact_center"
    IVR = "ivr"
    GATEWAY_CONTROL = "gateway_control"
    LEGACY_SIGNALING = "legacy_signaling"
    MESSAGING = "messaging"


@dataclass
class BPOProtocolSignature:
    """
    BPO protocol signature for pattern-based identification.

    Compatible with the platform's ProtocolSignature dataclass
    but adds BPO-specific fields.
    """

    # Basic identification
    protocol_id: str
    protocol_name: str
    version: str = "1.0"
    category: BPOProtocolCategory = BPOProtocolCategory.VOICE_SIGNALING

    # Pattern matching
    magic_bytes: List[Tuple[int, bytes]] = field(default_factory=list)
    regex_patterns: List[str] = field(default_factory=list)
    byte_sequences: List[bytes] = field(default_factory=list)

    # Structural characteristics
    encoding: str = "ascii"
    framing: str = "delimiter_based"
    min_message_size: int = 0
    max_message_size: int = 65535
    typical_ports: List[int] = field(default_factory=list)

    # Byte distribution characteristics
    expected_entropy_range: Tuple[float, float] = (0.0, 8.0)
    ascii_ratio_range: Tuple[float, float] = (0.0, 1.0)

    # BPO-specific fields
    vendor_associations: List[str] = field(default_factory=list)
    compliance_implications: List[str] = field(default_factory=list)
    security_notes: List[str] = field(default_factory=list)
    pqc_variant: Optional[str] = None

    # Confidence weights
    pattern_weight: float = 0.5
    structural_weight: float = 0.3
    port_weight: float = 0.2

    # Description
    description: str = ""

    def matches(self, data: bytes, port: Optional[int] = None) -> Tuple[bool, float]:
        """
        Check if data matches this signature.

        Returns:
            Tuple of (matched, confidence)
        """
        score = 0.0
        checks = 0

        # Magic bytes check
        if self.magic_bytes:
            checks += 1
            for offset, magic in self.magic_bytes:
                if len(data) > offset + len(magic):
                    if data[offset:offset + len(magic)] == magic:
                        score += self.pattern_weight
                        break

        # Regex pattern check
        if self.regex_patterns:
            checks += 1
            for pattern in self.regex_patterns:
                try:
                    if re.search(pattern.encode() if isinstance(pattern, str) else pattern, data):
                        score += self.pattern_weight
                        break
                except re.error:
                    continue

        # Byte sequence check
        if self.byte_sequences:
            checks += 1
            for seq in self.byte_sequences:
                if seq in data:
                    score += self.structural_weight
                    break

        # Port check
        if port and self.typical_ports:
            checks += 1
            if port in self.typical_ports:
                score += self.port_weight

        # Message size check
        if self.min_message_size <= len(data) <= self.max_message_size:
            score += 0.05

        matched = score > 0.3
        confidence = min(score, 1.0)

        return (matched, confidence)


# =============================================================================
# BPO Protocol Signature Database
# =============================================================================

def _build_sip_signature() -> BPOProtocolSignature:
    """Build SIP protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_sip",
        protocol_name="SIP (Session Initiation Protocol)",
        version="2.0",
        category=BPOProtocolCategory.VOICE_SIGNALING,
        magic_bytes=[],
        regex_patterns=[
            r"^(?:INVITE|REGISTER|ACK|BYE|CANCEL|OPTIONS|PRACK|SUBSCRIBE|NOTIFY|PUBLISH|INFO|REFER|MESSAGE|UPDATE)\s+sip:",
            r"^SIP/2\.0\s+\d{3}\s+",
        ],
        byte_sequences=[
            b"SIP/2.0",
            b"Via: SIP/2.0",
            b"Call-ID:",
            b"CSeq:",
        ],
        encoding="ascii",
        framing="delimiter_based",
        min_message_size=100,
        max_message_size=65535,
        typical_ports=[5060, 5061],
        expected_entropy_range=(3.5, 6.0),
        ascii_ratio_range=(0.85, 1.0),
        vendor_associations=["Avaya", "Cisco", "Genesys", "Mitel", "Asterisk"],
        compliance_implications=["PCI-DSS 4.0", "TCPA", "HIPAA"],
        security_notes=[
            "Check for TLS/SRTP negotiation in SDP",
            "Monitor for SIP authentication bypass attempts",
            "Validate From/To header spoofing",
        ],
        description="SIP signaling for VoIP call setup, modification, and teardown",
    )


def _build_sip_pqc_tls_signature() -> BPOProtocolSignature:
    """Build SIP-PQC-TLS protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_sip_pqc_tls",
        protocol_name="SIP-PQC-TLS (Quantum-Safe SIP)",
        version="2.0-PQC",
        category=BPOProtocolCategory.VOICE_SIGNALING,
        regex_patterns=[
            r"^(?:INVITE|REGISTER)\s+sips?:",
            r"SIP/2\.0.*(?:ML-KEM|MLKEM|kyber|dilithium)",
        ],
        byte_sequences=[
            b"SIP/2.0",
            b"algorithm=ML-KEM",
            b"X-PQC-KeyExchange:",
        ],
        encoding="ascii",
        framing="delimiter_based",
        typical_ports=[5061, 5062],
        expected_entropy_range=(4.0, 7.0),
        ascii_ratio_range=(0.75, 1.0),
        pqc_variant="ML-KEM-768 + ML-DSA-65",
        security_notes=[
            "Verify hybrid key exchange completion",
            "Check ML-KEM parameter sizes",
            "Validate PQC certificate chain",
        ],
        description="SIP over quantum-safe TLS with ML-KEM key exchange and ML-DSA signatures",
    )


def _build_sdp_signature() -> BPOProtocolSignature:
    """Build SDP protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_sdp",
        protocol_name="SDP (Session Description Protocol)",
        version="1.0",
        category=BPOProtocolCategory.VOICE_SIGNALING,
        regex_patterns=[
            r"^v=0\r?\n",
            r"m=audio\s+\d+\s+(?:RTP|SRTP)",
        ],
        byte_sequences=[
            b"v=0\r\n",
            b"o=",
            b"s=",
            b"c=IN IP",
            b"m=audio",
        ],
        encoding="ascii",
        framing="delimiter_based",
        min_message_size=50,
        max_message_size=4096,
        typical_ports=[5060, 5061],
        expected_entropy_range=(3.0, 5.5),
        ascii_ratio_range=(0.90, 1.0),
        security_notes=[
            "Check for SRTP crypto attributes",
            "Validate media port ranges",
            "Monitor for unexpected codec negotiations",
        ],
        description="Session Description Protocol for media negotiation in SIP calls",
    )


def _build_rtp_signature() -> BPOProtocolSignature:
    """Build RTP protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_rtp",
        protocol_name="RTP (Real-time Transport Protocol)",
        version="2.0",
        category=BPOProtocolCategory.MEDIA_TRANSPORT,
        magic_bytes=[(0, b"\x80")],  # Version 2, no padding, no extension, no CSRC
        regex_patterns=[],
        byte_sequences=[],
        encoding="binary",
        framing="fixed_length",
        min_message_size=12,
        max_message_size=1500,
        typical_ports=[5004, 5005] + list(range(16384, 32768, 2)),
        expected_entropy_range=(5.0, 8.0),
        ascii_ratio_range=(0.0, 0.3),
        compliance_implications=["PCI-DSS 4.0 (unencrypted voice)"],
        security_notes=[
            "CRITICAL: Unencrypted RTP exposes voice content",
            "DTMF digits in RTP may contain credit card numbers",
            "Must be upgraded to SRTP for PCI-DSS compliance",
        ],
        description="Real-time Transport Protocol for voice media (unencrypted)",
    )


def _build_srtp_signature() -> BPOProtocolSignature:
    """Build SRTP protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_srtp",
        protocol_name="SRTP (Secure Real-time Transport Protocol)",
        version="2.0",
        category=BPOProtocolCategory.MEDIA_TRANSPORT,
        magic_bytes=[(0, b"\x80")],  # Same RTP header but with auth tag
        regex_patterns=[],
        byte_sequences=[],
        encoding="binary",
        framing="fixed_length",
        min_message_size=22,  # 12 header + 10 auth tag minimum
        max_message_size=1500,
        typical_ports=[5004, 5005] + list(range(16384, 32768, 2)),
        expected_entropy_range=(6.5, 8.0),
        ascii_ratio_range=(0.0, 0.15),
        compliance_implications=["PCI-DSS 4.0 (compliant with encryption)"],
        security_notes=[
            "Verify SRTP key derivation from SDP crypto attributes",
            "Check for proper AES-128/256 counter mode",
            "Validate authentication tag integrity",
        ],
        description="Secure RTP with AES encryption for voice media protection",
    )


def _build_srtp_pqc_signature() -> BPOProtocolSignature:
    """Build SRTP-PQC protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_srtp_pqc",
        protocol_name="SRTP-PQC (Quantum-Safe SRTP)",
        version="2.0-PQC",
        category=BPOProtocolCategory.MEDIA_TRANSPORT,
        magic_bytes=[(0, b"\x80")],
        regex_patterns=[],
        byte_sequences=[],
        encoding="binary",
        framing="fixed_length",
        min_message_size=22,
        max_message_size=1500,
        typical_ports=[5004, 5005],
        expected_entropy_range=(7.0, 8.0),
        pqc_variant="ML-KEM-768 derived keys",
        security_notes=[
            "Key material derived from ML-KEM key exchange",
            "Hybrid classical+PQC key derivation",
            "Quantum-resistant voice encryption",
        ],
        description="SRTP with post-quantum key exchange for future-proof voice encryption",
    )


def _build_tn3270e_signature() -> BPOProtocolSignature:
    """Build TN3270e protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_tn3270e",
        protocol_name="TN3270e (IBM Terminal Emulation)",
        version="3270E",
        category=BPOProtocolCategory.TERMINAL_EMULATION,
        magic_bytes=[],
        byte_sequences=[
            b"\xff\xef",       # End of Record
            b"\xff\xfb\x28",   # IAC WILL TN3270E
            b"\xff\xfd\x28",   # IAC DO TN3270E
        ],
        encoding="ebcdic",
        framing="header_defined",
        min_message_size=3,
        max_message_size=32768,
        typical_ports=[23, 992],
        expected_entropy_range=(3.0, 6.0),
        ascii_ratio_range=(0.1, 0.6),
        vendor_associations=["IBM"],
        compliance_implications=["SOX", "PCI-DSS 4.0", "HIPAA"],
        security_notes=[
            "TN3270e often carries mainframe transaction data",
            "Screen scraping may expose sensitive financial data",
            "Port 992 provides TLS encryption; port 23 is unencrypted",
            "Monitor for unauthorized mainframe access patterns",
        ],
        description="IBM TN3270 Enhanced terminal emulation for mainframe access in BPO operations",
    )


def _build_cti_csta_signature() -> BPOProtocolSignature:
    """Build CTI-CSTA protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_cti_csta",
        protocol_name="CTI-CSTA (Computer Supported Telecommunications)",
        version="ECMA-269",
        category=BPOProtocolCategory.CONTACT_CENTER,
        regex_patterns=[
            r"<(?:Make|Answer|Clear|Hold|Retrieve|Transfer|Conference)Call",
            r"<(?:MonitorStart|MonitorStop|SnapshotDevice)",
            r"xmlns.*csta",
        ],
        byte_sequences=[
            b"<MakeCall",
            b"<AnswerCall",
            b"<ClearCall",
            b"<MonitorStart",
        ],
        encoding="ascii",
        framing="delimiter_based",
        min_message_size=50,
        max_message_size=65535,
        typical_ports=[7001, 7002, 8443],
        expected_entropy_range=(3.5, 5.5),
        ascii_ratio_range=(0.85, 1.0),
        vendor_associations=["Avaya", "Mitel", "Siemens"],
        compliance_implications=["SOC 2 Type II"],
        security_notes=[
            "CTI commands can initiate/transfer calls - audit all operations",
            "Monitor for unauthorized MakeCall/Transfer operations",
            "Ensure TLS encryption for CTI connections",
        ],
        description="ECMA CSTA protocol for programmatic call control in contact centers",
    )


def _build_cti_tsapi_signature() -> BPOProtocolSignature:
    """Build CTI-TSAPI (Avaya) protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_cti_tsapi",
        protocol_name="CTI-TSAPI (Avaya Telephony Services API)",
        version="TSAPI",
        category=BPOProtocolCategory.CONTACT_CENTER,
        regex_patterns=[
            r"<SOAP-ENV:Envelope.*TSAPI",
            r"<soap:Envelope.*telephony",
        ],
        byte_sequences=[
            b"<SOAP-ENV:Envelope",
            b"TSAPI",
            b"AESConnection",
        ],
        encoding="ascii",
        framing="header_defined",
        typical_ports=[450, 1050, 8443],
        vendor_associations=["Avaya"],
        security_notes=[
            "TSAPI provides full PBX control - high security sensitivity",
            "Monitor AES connection establishment",
            "Validate TSAPI client certificates",
        ],
        description="Avaya TSAPI for AES (Application Enablement Services) integration",
    )


def _build_cti_finesse_signature() -> BPOProtocolSignature:
    """Build CTI-Finesse (Cisco) protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_cti_finesse",
        protocol_name="CTI-Finesse (Cisco Finesse REST API)",
        version="Finesse 12.x",
        category=BPOProtocolCategory.CONTACT_CENTER,
        regex_patterns=[
            r"finesse/api/(?:User|Dialog|Queue|Team)",
            r"<Finesse>",
            r"application/xml.*finesse",
        ],
        byte_sequences=[
            b"finesse/api",
            b"<Finesse",
            b"<Dialog>",
        ],
        encoding="ascii",
        framing="header_defined",
        typical_ports=[443, 8443, 8445],
        vendor_associations=["Cisco"],
        security_notes=[
            "Finesse REST API must use HTTPS",
            "Monitor agent state changes",
            "Validate SSO tokens for Finesse access",
        ],
        description="Cisco Finesse REST/XMPP API for contact center agent desktop",
    )


def _build_ivr_vxml_signature() -> BPOProtocolSignature:
    """Build IVR-VoiceXML protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_ivr_vxml",
        protocol_name="IVR-VoiceXML",
        version="2.1",
        category=BPOProtocolCategory.IVR,
        regex_patterns=[
            r"<vxml\b",
            r"application/voicexml",
            r"<(?:form|menu|field|block|filled|grammar|prompt)\b",
        ],
        byte_sequences=[
            b"<vxml",
            b"<?xml",
            b"<form",
            b"<grammar",
        ],
        encoding="ascii",
        framing="delimiter_based",
        min_message_size=50,
        max_message_size=65535,
        typical_ports=[80, 443, 8080],
        expected_entropy_range=(3.5, 5.5),
        ascii_ratio_range=(0.90, 1.0),
        compliance_implications=["TCPA", "PCI-DSS 4.0"],
        security_notes=[
            "IVR scripts may collect credit card numbers via DTMF",
            "Ensure DTMF input is masked/encrypted (PCI-DSS)",
            "Validate grammar files for injection attacks",
        ],
        description="VoiceXML documents for IVR menu trees and caller interaction",
    )


def _build_mrcp_signature() -> BPOProtocolSignature:
    """Build MRCP protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_mrcp",
        protocol_name="MRCP (Media Resource Control Protocol)",
        version="2.0",
        category=BPOProtocolCategory.IVR,
        regex_patterns=[
            r"^MRCP/2\.0",
            r"^(?:SPEAK|RECOGNIZE|DEFINE-GRAMMAR|SET-PARAMS|GET-PARAMS|STOP)\b.*MRCP",
        ],
        byte_sequences=[
            b"MRCP/2.0",
            b"SPEAK",
            b"RECOGNIZE",
        ],
        encoding="ascii",
        framing="header_defined",
        typical_ports=[6060, 6061, 9090],
        vendor_associations=["Nuance", "LumenVox", "Google"],
        security_notes=[
            "MRCP carries speech recognition results",
            "Recognized speech may contain sensitive data",
            "Ensure TLS for MRCP connections carrying PCI data",
        ],
        description="MRCP for speech synthesis and recognition in IVR systems",
    )


def _build_mgcp_signature() -> BPOProtocolSignature:
    """Build MGCP protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_mgcp",
        protocol_name="MGCP (Media Gateway Control Protocol)",
        version="1.0",
        category=BPOProtocolCategory.GATEWAY_CONTROL,
        regex_patterns=[
            r"^(?:CRCX|MDCX|DLCX|RQNT|NTFY|AUEP|AUCX|RSIP)\s+\d+",
            r"^200\s+\d+\s+OK",
        ],
        byte_sequences=[
            b"CRCX ",
            b"MDCX ",
            b"DLCX ",
            b"RQNT ",
        ],
        encoding="ascii",
        framing="delimiter_based",
        typical_ports=[2427, 2727],
        expected_entropy_range=(3.5, 5.5),
        ascii_ratio_range=(0.85, 1.0),
        vendor_associations=["Cisco", "AudioCodes"],
        security_notes=[
            "MGCP can create/modify media connections",
            "Ensure proper ACL on MGCP control ports",
            "Monitor for unauthorized connection creation (CRCX)",
        ],
        description="MGCP for media gateway control in VoIP infrastructure",
    )


def _build_h323_signature() -> BPOProtocolSignature:
    """Build H.323 protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_h323",
        protocol_name="H.323 (Multimedia Communications)",
        version="5.0",
        category=BPOProtocolCategory.VOICE_SIGNALING,
        magic_bytes=[(0, b"\x03")],  # TPKT version 3
        byte_sequences=[
            b"\x03\x00",  # TPKT header
        ],
        encoding="binary",
        framing="length_prefixed",
        min_message_size=10,
        max_message_size=65535,
        typical_ports=[1720, 1721],
        expected_entropy_range=(4.0, 7.0),
        ascii_ratio_range=(0.1, 0.5),
        vendor_associations=["Cisco", "Avaya", "Polycom"],
        security_notes=[
            "H.323 uses ASN.1/PER encoding - complex attack surface",
            "Monitor for H.225 call setup anomalies",
            "H.235 security profile should be enforced",
        ],
        description="H.323 multimedia framework for legacy VoIP systems",
    )


def _build_smpp_signature() -> BPOProtocolSignature:
    """Build SMPP protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_smpp",
        protocol_name="SMPP (Short Message Peer-to-Peer)",
        version="3.4",
        category=BPOProtocolCategory.MESSAGING,
        magic_bytes=[],
        byte_sequences=[],
        encoding="binary",
        framing="length_prefixed",
        min_message_size=16,
        max_message_size=65535,
        typical_ports=[2775, 2776],
        expected_entropy_range=(3.0, 6.0),
        ascii_ratio_range=(0.2, 0.7),
        compliance_implications=["TCPA"],
        security_notes=[
            "SMPP carries SMS messages - may contain OTP codes",
            "Monitor for bulk SMS abuse (TCPA compliance)",
            "Ensure TLS wrapping for SMPP connections",
        ],
        description="SMPP for SMS routing in contact center outbound messaging",
    )


def _build_ss7_isup_signature() -> BPOProtocolSignature:
    """Build SS7/ISUP protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_ss7_isup",
        protocol_name="SS7/ISUP (Signaling System 7)",
        version="ITU-T Q.763",
        category=BPOProtocolCategory.LEGACY_SIGNALING,
        magic_bytes=[],
        byte_sequences=[],
        encoding="binary",
        framing="length_prefixed",
        min_message_size=5,
        max_message_size=272,
        typical_ports=[],  # SS7 uses dedicated links, not TCP/IP ports
        expected_entropy_range=(3.0, 6.0),
        ascii_ratio_range=(0.0, 0.3),
        security_notes=[
            "CRITICAL: SS7 has no built-in security",
            "Monitor for SS7 location tracking attacks",
            "Watch for call interception via MAP/CAP manipulation",
            "Recommend migration to SIGTRAN/Diameter",
        ],
        description="Legacy SS7/ISUP signaling for PSTN interworking",
    )


def _build_dtmf_rfc4733_signature() -> BPOProtocolSignature:
    """Build DTMF RFC 4733 protocol signature."""
    return BPOProtocolSignature(
        protocol_id="bpo_dtmf_rfc4733",
        protocol_name="DTMF-RFC4733 (Telephone Events)",
        version="RFC 4733",
        category=BPOProtocolCategory.MEDIA_TRANSPORT,
        magic_bytes=[(0, b"\x80")],  # RTP header
        byte_sequences=[],
        encoding="binary",
        framing="fixed_length",
        min_message_size=16,  # 12 RTP header + 4 event payload
        max_message_size=20,
        typical_ports=list(range(16384, 32768, 2)),
        expected_entropy_range=(2.0, 5.0),
        compliance_implications=["PCI-DSS 4.0"],
        security_notes=[
            "CRITICAL PCI-DSS: DTMF carries credit card digits",
            "Must mask/suppress DTMF in recordings",
            "In-band DTMF must be encrypted end-to-end",
            "Use RFC 4733 out-of-band for easier PCI compliance",
        ],
        description="DTMF telephone events via RTP for digit collection in IVR/payment systems",
    )


# =============================================================================
# Signature Provider
# =============================================================================

class BPOProtocolSignatureProvider:
    """
    Provider for BPO protocol signatures.

    Builds and manages the complete set of BPO protocol signatures
    and provides methods to register them with the platform's
    ProtocolSignatureDatabase.
    """

    def __init__(self):
        """Initialize the BPO signature provider."""
        self.logger = logging.getLogger(f"{__name__}.BPOProtocolSignatureProvider")
        self._signatures: Dict[str, BPOProtocolSignature] = {}
        self._load_builtin_signatures()

    def _load_builtin_signatures(self) -> None:
        """Load all built-in BPO protocol signatures."""
        builders = [
            _build_sip_signature,
            _build_sip_pqc_tls_signature,
            _build_sdp_signature,
            _build_rtp_signature,
            _build_srtp_signature,
            _build_srtp_pqc_signature,
            _build_tn3270e_signature,
            _build_cti_csta_signature,
            _build_cti_tsapi_signature,
            _build_cti_finesse_signature,
            _build_ivr_vxml_signature,
            _build_mrcp_signature,
            _build_mgcp_signature,
            _build_h323_signature,
            _build_smpp_signature,
            _build_ss7_isup_signature,
            _build_dtmf_rfc4733_signature,
        ]

        for builder in builders:
            sig = builder()
            self._signatures[sig.protocol_id] = sig
            self.logger.debug(f"Loaded BPO signature: {sig.protocol_id} ({sig.protocol_name})")

        self.logger.info(f"Loaded {len(self._signatures)} BPO protocol signatures")

    def get_signature(self, protocol_id: str) -> Optional[BPOProtocolSignature]:
        """Get a specific protocol signature."""
        return self._signatures.get(protocol_id)

    def get_all_signatures(self) -> Dict[str, BPOProtocolSignature]:
        """Get all BPO protocol signatures."""
        return dict(self._signatures)

    def get_signatures_by_category(
        self, category: BPOProtocolCategory
    ) -> List[BPOProtocolSignature]:
        """Get signatures filtered by category."""
        return [s for s in self._signatures.values() if s.category == category]

    def get_signatures_by_vendor(self, vendor: str) -> List[BPOProtocolSignature]:
        """Get signatures associated with a specific vendor."""
        return [
            s for s in self._signatures.values()
            if vendor.lower() in [v.lower() for v in s.vendor_associations]
        ]

    def match_data(
        self, data: bytes, port: Optional[int] = None
    ) -> List[Tuple[BPOProtocolSignature, float]]:
        """
        Match data against all BPO signatures.

        Returns:
            List of (signature, confidence) tuples, sorted by confidence
        """
        matches = []
        for sig in self._signatures.values():
            matched, confidence = sig.matches(data, port)
            if matched:
                matches.append((sig, confidence))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def add_custom_signature(self, signature: BPOProtocolSignature) -> None:
        """Add a custom protocol signature."""
        self._signatures[signature.protocol_id] = signature
        self.logger.info(f"Added custom BPO signature: {signature.protocol_id}")

    def get_pci_relevant_signatures(self) -> List[BPOProtocolSignature]:
        """Get signatures relevant to PCI-DSS compliance."""
        return [
            s for s in self._signatures.values()
            if any("PCI" in c for c in s.compliance_implications)
            or any("PCI" in n for n in s.security_notes)
        ]

    def get_pqc_signatures(self) -> List[BPOProtocolSignature]:
        """Get post-quantum capable protocol signatures."""
        return [
            s for s in self._signatures.values()
            if s.pqc_variant is not None
        ]

    def to_platform_format(self) -> List[Dict[str, Any]]:
        """
        Convert BPO signatures to platform ProtocolSignature format.

        Returns format compatible with ProtocolSignatureDatabase.register_signature()
        """
        platform_sigs = []
        for sig in self._signatures.values():
            platform_sigs.append({
                "protocol_id": sig.protocol_id,
                "protocol_name": sig.protocol_name,
                "version": sig.version,
                "category": "application",  # Map to platform categories
                "magic_bytes": sig.magic_bytes,
                "regex_patterns": sig.regex_patterns,
                "byte_sequences": sig.byte_sequences,
                "encoding": sig.encoding,
                "framing": sig.framing,
                "min_message_size": sig.min_message_size,
                "max_message_size": sig.max_message_size,
                "typical_port": sig.typical_ports[0] if sig.typical_ports else None,
                "description": sig.description,
                "metadata": {
                    "bpo_category": sig.category.value,
                    "vendor_associations": sig.vendor_associations,
                    "compliance_implications": sig.compliance_implications,
                    "security_notes": sig.security_notes,
                    "pqc_variant": sig.pqc_variant,
                },
            })
        return platform_sigs


# Module-level convenience functions

_provider_instance: Optional[BPOProtocolSignatureProvider] = None


def get_bpo_signatures() -> BPOProtocolSignatureProvider:
    """Get the singleton BPO protocol signature provider."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = BPOProtocolSignatureProvider()
    return _provider_instance


def register_bpo_signatures(signature_database: Any) -> int:
    """
    Register all BPO signatures with the platform's ProtocolSignatureDatabase.

    Args:
        signature_database: Platform ProtocolSignatureDatabase instance

    Returns:
        Number of signatures registered
    """
    provider = get_bpo_signatures()
    platform_sigs = provider.to_platform_format()
    count = 0

    for sig_data in platform_sigs:
        try:
            if hasattr(signature_database, "register_signature"):
                signature_database.register_signature(sig_data)
            elif hasattr(signature_database, "add_signature"):
                signature_database.add_signature(sig_data)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to register BPO signature {sig_data['protocol_id']}: {e}")

    logger.info(f"Registered {count}/{len(platform_sigs)} BPO protocol signatures with platform")
    return count
