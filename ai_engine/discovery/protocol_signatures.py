"""
QBITEL - Protocol Signature Database

Pre-defined protocol signatures for rapid protocol identification without
requiring ML model training. Includes signatures for legacy mainframe protocols,
industrial protocols, and common network protocols.

This database enables:
- Fast protocol detection via pattern matching
- Signature-based classification for known protocols
- Hybrid classification (signatures + ML)
- Protocol discovery hints for unknown protocols
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Pattern
import hashlib

logger = logging.getLogger(__name__)


class ProtocolCategory(Enum):
    """Protocol categories for classification."""

    LEGACY_MAINFRAME = "legacy_mainframe"
    INDUSTRIAL = "industrial"
    FINANCIAL = "financial"
    NETWORK = "network"
    APPLICATION = "application"
    DATABASE = "database"
    MESSAGING = "messaging"
    IOT = "iot"
    HEALTHCARE = "healthcare"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


class EncodingType(Enum):
    """Data encoding types."""

    ASCII = "ascii"
    EBCDIC = "ebcdic"
    BINARY = "binary"
    UTF8 = "utf8"
    UTF16 = "utf16"
    MIXED = "mixed"


class FramingType(Enum):
    """Message framing types."""

    LENGTH_PREFIXED = "length_prefixed"
    DELIMITER_BASED = "delimiter_based"
    FIXED_LENGTH = "fixed_length"
    HEADER_DEFINED = "header_defined"
    NO_FRAMING = "no_framing"


@dataclass
class ProtocolSignature:
    """
    Protocol signature for pattern-based identification.

    Signatures contain multiple matching criteria:
    - Magic bytes at specific offsets
    - Regex patterns for text protocols
    - Byte distribution characteristics
    - Structural features
    """

    # Basic identification
    protocol_id: str
    protocol_name: str
    version: str = "1.0"
    category: ProtocolCategory = ProtocolCategory.UNKNOWN

    # Pattern matching
    magic_bytes: List[Tuple[int, bytes]] = field(default_factory=list)  # (offset, bytes)
    regex_patterns: List[str] = field(default_factory=list)
    byte_sequences: List[bytes] = field(default_factory=list)

    # Structural characteristics
    encoding: EncodingType = EncodingType.BINARY
    framing: FramingType = FramingType.NO_FRAMING
    min_message_size: int = 0
    max_message_size: int = 65535
    typical_port: Optional[int] = None

    # Byte distribution characteristics
    expected_entropy_range: Tuple[float, float] = (0.0, 8.0)
    ascii_ratio_range: Tuple[float, float] = (0.0, 1.0)
    null_ratio_range: Tuple[float, float] = (0.0, 1.0)

    # Header structure
    header_size: int = 0
    length_field_offset: Optional[int] = None
    length_field_size: int = 0
    length_includes_header: bool = False

    # Field markers
    field_delimiters: List[bytes] = field(default_factory=list)
    record_delimiters: List[bytes] = field(default_factory=list)

    # Metadata
    description: str = ""
    documentation_url: str = ""
    vendor: str = ""
    is_proprietary: bool = False
    requires_license: bool = False

    # Confidence weights for matching
    magic_bytes_weight: float = 0.4
    pattern_weight: float = 0.3
    structure_weight: float = 0.2
    statistics_weight: float = 0.1

    def __post_init__(self):
        """Compile regex patterns after initialization."""
        self._compiled_patterns: List[Pattern] = []
        for pattern in self.regex_patterns:
            try:
                self._compiled_patterns.append(re.compile(pattern.encode(), re.DOTALL))
            except re.error as e:
                logger.warning(f"Invalid regex pattern in {self.protocol_id}: {e}")


@dataclass
class SignatureMatch:
    """Result of signature matching."""

    signature: ProtocolSignature
    confidence: float
    match_details: Dict[str, Any] = field(default_factory=dict)
    matched_criteria: List[str] = field(default_factory=list)


class ProtocolSignatureDatabase:
    """
    Database of protocol signatures for rapid identification.

    Provides signature-based protocol detection that can work standalone
    or in combination with ML-based classification.
    """

    def __init__(self):
        self.signatures: Dict[str, ProtocolSignature] = {}
        self.category_index: Dict[ProtocolCategory, Set[str]] = {}
        self.magic_bytes_index: Dict[bytes, Set[str]] = {}
        self._load_builtin_signatures()

    def _load_builtin_signatures(self) -> None:
        """Load built-in protocol signatures."""
        builtin_signatures = [
            # ========== LEGACY MAINFRAME PROTOCOLS ==========
            ProtocolSignature(
                protocol_id="ibm_cics_tn3270",
                protocol_name="IBM CICS TN3270",
                category=ProtocolCategory.LEGACY_MAINFRAME,
                description="IBM 3270 terminal emulation protocol for CICS",
                encoding=EncodingType.EBCDIC,
                framing=FramingType.LENGTH_PREFIXED,
                magic_bytes=[(0, b"\xff\xef")],  # IAC EOR
                typical_port=23,
                header_size=5,
                expected_entropy_range=(3.0, 6.0),
                ascii_ratio_range=(0.0, 0.3),  # EBCDIC not ASCII
                field_delimiters=[b"\x11"],  # Set Buffer Address
                vendor="IBM",
            ),
            ProtocolSignature(
                protocol_id="ibm_cics_lu62",
                protocol_name="IBM LU 6.2 (APPC)",
                category=ProtocolCategory.LEGACY_MAINFRAME,
                description="IBM Advanced Program-to-Program Communication",
                encoding=EncodingType.EBCDIC,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(0, b"\x00\x00")],  # Transmission header
                min_message_size=6,
                header_size=6,
                expected_entropy_range=(2.5, 5.5),
                vendor="IBM",
            ),
            ProtocolSignature(
                protocol_id="ibm_ims_mfs",
                protocol_name="IBM IMS Message Format Services",
                category=ProtocolCategory.LEGACY_MAINFRAME,
                description="IBM IMS transaction protocol",
                encoding=EncodingType.EBCDIC,
                framing=FramingType.FIXED_LENGTH,
                expected_entropy_range=(2.0, 5.0),
                ascii_ratio_range=(0.0, 0.2),
                null_ratio_range=(0.1, 0.4),  # Often padded
                vendor="IBM",
            ),
            ProtocolSignature(
                protocol_id="ibm_mqseries",
                protocol_name="IBM MQ Series",
                category=ProtocolCategory.LEGACY_MAINFRAME,
                description="IBM MQ message queue protocol",
                encoding=EncodingType.MIXED,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(0, b"TSH ")],  # Transmission Segment Header
                header_size=28,
                length_field_offset=4,
                length_field_size=4,
                typical_port=1414,
                vendor="IBM",
            ),
            ProtocolSignature(
                protocol_id="unisys_dmsii",
                protocol_name="Unisys DMSII",
                category=ProtocolCategory.LEGACY_MAINFRAME,
                description="Unisys mainframe database access protocol",
                encoding=EncodingType.EBCDIC,
                framing=FramingType.LENGTH_PREFIXED,
                expected_entropy_range=(2.5, 5.0),
                vendor="Unisys",
            ),
            ProtocolSignature(
                protocol_id="cobol_copybook",
                protocol_name="COBOL Copybook Format",
                category=ProtocolCategory.LEGACY_MAINFRAME,
                description="Fixed-length COBOL record format",
                encoding=EncodingType.EBCDIC,
                framing=FramingType.FIXED_LENGTH,
                expected_entropy_range=(2.0, 4.5),
                null_ratio_range=(0.05, 0.3),  # Padded fields
                ascii_ratio_range=(0.0, 0.2),
            ),
            # ========== INDUSTRIAL PROTOCOLS ==========
            ProtocolSignature(
                protocol_id="modbus_tcp",
                protocol_name="Modbus TCP",
                category=ProtocolCategory.INDUSTRIAL,
                description="Industrial automation protocol over TCP",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(0, b"\x00\x00")],  # Transaction ID starts at 0
                min_message_size=12,
                max_message_size=260,
                header_size=7,  # MBAP header
                length_field_offset=4,
                length_field_size=2,
                typical_port=502,
                expected_entropy_range=(1.5, 4.5),
            ),
            ProtocolSignature(
                protocol_id="modbus_rtu",
                protocol_name="Modbus RTU",
                category=ProtocolCategory.INDUSTRIAL,
                description="Modbus over serial (RTU framing)",
                encoding=EncodingType.BINARY,
                framing=FramingType.DELIMITER_BASED,
                min_message_size=4,
                max_message_size=256,
                expected_entropy_range=(2.0, 5.0),
            ),
            ProtocolSignature(
                protocol_id="dnp3",
                protocol_name="DNP3 (Distributed Network Protocol)",
                category=ProtocolCategory.INDUSTRIAL,
                description="SCADA/power grid automation protocol",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(0, b"\x05\x64")],  # DNP3 start bytes
                min_message_size=10,
                header_size=10,
                typical_port=20000,
                expected_entropy_range=(2.0, 5.0),
            ),
            ProtocolSignature(
                protocol_id="iec61850_goose",
                protocol_name="IEC 61850 GOOSE",
                category=ProtocolCategory.INDUSTRIAL,
                description="Power grid substation automation",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(12, b"\x88\xb8")],  # EtherType
                expected_entropy_range=(2.5, 5.0),
            ),
            ProtocolSignature(
                protocol_id="bacnet_ip",
                protocol_name="BACnet/IP",
                category=ProtocolCategory.INDUSTRIAL,
                description="Building automation protocol",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(0, b"\x81")],  # BACnet Virtual Link Control
                typical_port=47808,
                expected_entropy_range=(2.0, 5.0),
            ),
            ProtocolSignature(
                protocol_id="opc_ua_binary",
                protocol_name="OPC UA Binary",
                category=ProtocolCategory.INDUSTRIAL,
                description="OPC Unified Architecture binary protocol",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(0, b"OPN"), (0, b"MSG"), (0, b"CLO")],
                header_size=8,
                length_field_offset=4,
                length_field_size=4,
                typical_port=4840,
            ),
            ProtocolSignature(
                protocol_id="profinet",
                protocol_name="PROFINET",
                category=ProtocolCategory.INDUSTRIAL,
                description="Industrial Ethernet protocol",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(12, b"\x88\x92")],  # EtherType
                expected_entropy_range=(2.0, 5.5),
            ),
            ProtocolSignature(
                protocol_id="ethernetip",
                protocol_name="EtherNet/IP",
                category=ProtocolCategory.INDUSTRIAL,
                description="Industrial Ethernet protocol (Rockwell)",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                typical_port=44818,
                header_size=24,
                expected_entropy_range=(2.0, 5.0),
            ),
            # ========== FINANCIAL PROTOCOLS ==========
            ProtocolSignature(
                protocol_id="iso8583",
                protocol_name="ISO 8583",
                category=ProtocolCategory.FINANCIAL,
                description="Financial transaction message format",
                encoding=EncodingType.MIXED,
                framing=FramingType.LENGTH_PREFIXED,
                min_message_size=20,
                header_size=2,  # Length header
                length_field_offset=0,
                length_field_size=2,
                expected_entropy_range=(3.0, 6.0),
            ),
            ProtocolSignature(
                protocol_id="fix",
                protocol_name="FIX Protocol",
                category=ProtocolCategory.FINANCIAL,
                description="Financial Information eXchange protocol",
                encoding=EncodingType.ASCII,
                framing=FramingType.DELIMITER_BASED,
                regex_patterns=[r"8=FIX\.\d+\.\d+\x01"],
                byte_sequences=[b"8=FIX"],
                field_delimiters=[b"\x01"],  # SOH delimiter
                expected_entropy_range=(4.0, 6.5),
                ascii_ratio_range=(0.8, 1.0),
            ),
            ProtocolSignature(
                protocol_id="swift_mt",
                protocol_name="SWIFT MT Messages",
                category=ProtocolCategory.FINANCIAL,
                description="SWIFT message format (MT)",
                encoding=EncodingType.ASCII,
                framing=FramingType.DELIMITER_BASED,
                regex_patterns=[r"\{1:F\d{2}[A-Z]{12}\d{4}\d{6}\}"],
                byte_sequences=[b"{1:F01", b"{1:F21"],
                record_delimiters=[b"\r\n"],
                expected_entropy_range=(4.0, 6.0),
                ascii_ratio_range=(0.9, 1.0),
            ),
            ProtocolSignature(
                protocol_id="iso20022_xml",
                protocol_name="ISO 20022 XML",
                category=ProtocolCategory.FINANCIAL,
                description="ISO 20022 financial messaging (XML)",
                encoding=EncodingType.UTF8,
                framing=FramingType.NO_FRAMING,
                regex_patterns=[
                    r"<\?xml.*\?>.*<Document.*xmlns.*iso20022",
                    r"<pain\.\d{3}\.\d{3}\.\d{2}>",
                ],
                byte_sequences=[b"<?xml", b"xmlns:"],
                expected_entropy_range=(4.5, 6.5),
                ascii_ratio_range=(0.95, 1.0),
            ),
            ProtocolSignature(
                protocol_id="nacha_ach",
                protocol_name="NACHA ACH File Format",
                category=ProtocolCategory.FINANCIAL,
                description="ACH batch file format",
                encoding=EncodingType.ASCII,
                framing=FramingType.FIXED_LENGTH,
                regex_patterns=[r"^1"],  # File header record type
                min_message_size=94,  # Record length
                expected_entropy_range=(4.0, 5.5),
                ascii_ratio_range=(0.95, 1.0),
            ),
            # ========== NETWORK PROTOCOLS ==========
            ProtocolSignature(
                protocol_id="http_1x",
                protocol_name="HTTP/1.x",
                category=ProtocolCategory.APPLICATION,
                description="HTTP protocol version 1.x",
                encoding=EncodingType.ASCII,
                framing=FramingType.DELIMITER_BASED,
                regex_patterns=[
                    r"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH) .* HTTP/1\.[01]",
                    r"^HTTP/1\.[01] \d{3}",
                ],
                byte_sequences=[b"GET ", b"POST ", b"HTTP/1."],
                record_delimiters=[b"\r\n\r\n"],
                typical_port=80,
                expected_entropy_range=(4.0, 6.5),
                ascii_ratio_range=(0.8, 1.0),
            ),
            ProtocolSignature(
                protocol_id="http_2",
                protocol_name="HTTP/2",
                category=ProtocolCategory.APPLICATION,
                description="HTTP protocol version 2",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(0, b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")],
                typical_port=443,
                header_size=9,  # Frame header
                expected_entropy_range=(3.0, 7.0),
            ),
            ProtocolSignature(
                protocol_id="tls_1x",
                protocol_name="TLS 1.x",
                category=ProtocolCategory.NETWORK,
                description="Transport Layer Security",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(0, b"\x16\x03")],  # TLS record
                header_size=5,
                length_field_offset=3,
                length_field_size=2,
                typical_port=443,
                expected_entropy_range=(6.0, 8.0),  # High entropy (encrypted)
            ),
            ProtocolSignature(
                protocol_id="ssh",
                protocol_name="SSH Protocol",
                category=ProtocolCategory.NETWORK,
                description="Secure Shell protocol",
                encoding=EncodingType.MIXED,
                framing=FramingType.LENGTH_PREFIXED,
                regex_patterns=[r"^SSH-2\.0-"],
                byte_sequences=[b"SSH-2.0-"],
                typical_port=22,
                expected_entropy_range=(5.0, 8.0),
            ),
            ProtocolSignature(
                protocol_id="dns",
                protocol_name="DNS Protocol",
                category=ProtocolCategory.NETWORK,
                description="Domain Name System protocol",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                min_message_size=12,
                header_size=12,
                typical_port=53,
                expected_entropy_range=(3.0, 6.0),
            ),
            # ========== MESSAGING PROTOCOLS ==========
            ProtocolSignature(
                protocol_id="amqp_091",
                protocol_name="AMQP 0-9-1",
                category=ProtocolCategory.MESSAGING,
                description="Advanced Message Queuing Protocol",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(0, b"AMQP\x00\x00\x09\x01")],
                typical_port=5672,
                expected_entropy_range=(3.0, 6.0),
            ),
            ProtocolSignature(
                protocol_id="kafka",
                protocol_name="Apache Kafka Protocol",
                category=ProtocolCategory.MESSAGING,
                description="Kafka message broker protocol",
                encoding=EncodingType.BINARY,
                framing=FramingType.LENGTH_PREFIXED,
                header_size=4,  # Request size
                length_field_offset=0,
                length_field_size=4,
                typical_port=9092,
                expected_entropy_range=(3.0, 6.5),
            ),
            ProtocolSignature(
                protocol_id="mqtt",
                protocol_name="MQTT",
                category=ProtocolCategory.IOT,
                description="MQ Telemetry Transport (IoT)",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(0, b"\x10")],  # CONNECT packet
                min_message_size=2,
                typical_port=1883,
                expected_entropy_range=(2.5, 6.0),
            ),
            # ========== HEALTHCARE PROTOCOLS ==========
            ProtocolSignature(
                protocol_id="hl7_v2",
                protocol_name="HL7 v2.x",
                category=ProtocolCategory.HEALTHCARE,
                description="Health Level 7 version 2 messages",
                encoding=EncodingType.ASCII,
                framing=FramingType.DELIMITER_BASED,
                regex_patterns=[r"^MSH\|"],
                byte_sequences=[b"MSH|"],
                field_delimiters=[b"|", b"^", b"~", b"\\", b"&"],
                record_delimiters=[b"\r"],
                expected_entropy_range=(4.0, 6.0),
                ascii_ratio_range=(0.9, 1.0),
            ),
            ProtocolSignature(
                protocol_id="dicom",
                protocol_name="DICOM",
                category=ProtocolCategory.HEALTHCARE,
                description="Digital Imaging and Communications in Medicine",
                encoding=EncodingType.BINARY,
                framing=FramingType.HEADER_DEFINED,
                magic_bytes=[(128, b"DICM")],  # DICOM prefix at offset 128
                typical_port=104,
                expected_entropy_range=(3.0, 7.0),
            ),
            ProtocolSignature(
                protocol_id="fhir_json",
                protocol_name="FHIR (JSON)",
                category=ProtocolCategory.HEALTHCARE,
                description="Fast Healthcare Interoperability Resources",
                encoding=EncodingType.UTF8,
                framing=FramingType.NO_FRAMING,
                regex_patterns=[r'"resourceType"\s*:\s*"[A-Z][a-zA-Z]+"'],
                byte_sequences=[b'"resourceType"'],
                expected_entropy_range=(4.5, 6.5),
                ascii_ratio_range=(0.95, 1.0),
            ),
            # ========== DATABASE PROTOCOLS ==========
            ProtocolSignature(
                protocol_id="postgresql",
                protocol_name="PostgreSQL Protocol",
                category=ProtocolCategory.DATABASE,
                description="PostgreSQL wire protocol",
                encoding=EncodingType.BINARY,
                framing=FramingType.LENGTH_PREFIXED,
                magic_bytes=[(0, b"\x00\x03\x00\x00")],  # StartupMessage
                header_size=4,
                length_field_offset=0,
                length_field_size=4,
                typical_port=5432,
                expected_entropy_range=(3.0, 6.0),
            ),
            ProtocolSignature(
                protocol_id="mysql",
                protocol_name="MySQL Protocol",
                category=ProtocolCategory.DATABASE,
                description="MySQL client/server protocol",
                encoding=EncodingType.BINARY,
                framing=FramingType.LENGTH_PREFIXED,
                header_size=4,
                length_field_offset=0,
                length_field_size=3,
                typical_port=3306,
                expected_entropy_range=(3.0, 6.5),
            ),
            ProtocolSignature(
                protocol_id="redis",
                protocol_name="Redis Protocol (RESP)",
                category=ProtocolCategory.DATABASE,
                description="Redis Serialization Protocol",
                encoding=EncodingType.ASCII,
                framing=FramingType.DELIMITER_BASED,
                regex_patterns=[r"^\*\d+\r\n\$\d+\r\n"],
                byte_sequences=[b"*", b"+OK\r\n", b"-ERR"],
                record_delimiters=[b"\r\n"],
                typical_port=6379,
                expected_entropy_range=(3.5, 5.5),
                ascii_ratio_range=(0.8, 1.0),
            ),
        ]

        # Register all signatures
        for signature in builtin_signatures:
            self.register_signature(signature)

        logger.info(
            f"Loaded {len(self.signatures)} built-in protocol signatures " f"across {len(self.category_index)} categories"
        )

    def register_signature(self, signature: ProtocolSignature) -> None:
        """Register a protocol signature."""
        self.signatures[signature.protocol_id] = signature

        # Update category index
        if signature.category not in self.category_index:
            self.category_index[signature.category] = set()
        self.category_index[signature.category].add(signature.protocol_id)

        # Update magic bytes index
        for offset, magic in signature.magic_bytes:
            key = magic[:4] if len(magic) >= 4 else magic
            if key not in self.magic_bytes_index:
                self.magic_bytes_index[key] = set()
            self.magic_bytes_index[key].add(signature.protocol_id)

    def identify_protocol(
        self,
        data: bytes,
        category_hint: Optional[ProtocolCategory] = None,
        top_n: int = 3,
    ) -> List[SignatureMatch]:
        """
        Identify protocol from data using signature matching.

        Args:
            data: Protocol message data
            category_hint: Optional category to prioritize
            top_n: Number of top matches to return

        Returns:
            List of signature matches sorted by confidence
        """
        if not data:
            return []

        matches: List[SignatureMatch] = []

        # Get candidate signatures
        candidates = self._get_candidate_signatures(data, category_hint)

        # Score each candidate
        for protocol_id in candidates:
            signature = self.signatures[protocol_id]
            confidence, details, criteria = self._score_signature_match(data, signature)

            if confidence > 0:
                matches.append(
                    SignatureMatch(
                        signature=signature,
                        confidence=confidence,
                        match_details=details,
                        matched_criteria=criteria,
                    )
                )

        # Sort by confidence and return top N
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:top_n]

    def _get_candidate_signatures(
        self,
        data: bytes,
        category_hint: Optional[ProtocolCategory] = None,
    ) -> Set[str]:
        """Get candidate signatures based on quick checks."""
        candidates = set()

        # Check magic bytes index for quick filtering
        for key_len in [4, 3, 2, 1]:
            if len(data) >= key_len:
                key = data[:key_len]
                if key in self.magic_bytes_index:
                    candidates.update(self.magic_bytes_index[key])

        # If category hint provided, prioritize those
        if category_hint and category_hint in self.category_index:
            category_ids = self.category_index[category_hint]
            candidates.update(category_ids)

        # If no candidates found, check all signatures
        if not candidates:
            candidates = set(self.signatures.keys())

        return candidates

    def _score_signature_match(
        self,
        data: bytes,
        signature: ProtocolSignature,
    ) -> Tuple[float, Dict[str, Any], List[str]]:
        """Score how well data matches a signature."""
        score = 0.0
        details = {}
        matched_criteria = []

        # Check magic bytes (highest weight)
        magic_score, magic_details = self._check_magic_bytes(data, signature)
        if magic_score > 0:
            score += magic_score * signature.magic_bytes_weight
            details["magic_bytes"] = magic_details
            matched_criteria.append("magic_bytes")

        # Check regex patterns
        pattern_score, pattern_details = self._check_patterns(data, signature)
        if pattern_score > 0:
            score += pattern_score * signature.pattern_weight
            details["patterns"] = pattern_details
            matched_criteria.append("patterns")

        # Check byte sequences
        seq_score, seq_details = self._check_byte_sequences(data, signature)
        if seq_score > 0:
            score += seq_score * signature.pattern_weight * 0.5
            details["byte_sequences"] = seq_details
            matched_criteria.append("byte_sequences")

        # Check structural characteristics
        struct_score, struct_details = self._check_structure(data, signature)
        if struct_score > 0:
            score += struct_score * signature.structure_weight
            details["structure"] = struct_details
            matched_criteria.append("structure")

        # Check statistical characteristics
        stats_score, stats_details = self._check_statistics(data, signature)
        if stats_score > 0:
            score += stats_score * signature.statistics_weight
            details["statistics"] = stats_details
            matched_criteria.append("statistics")

        # Normalize score to 0-1 range
        max_possible_score = (
            signature.magic_bytes_weight + signature.pattern_weight + signature.structure_weight + signature.statistics_weight
        )
        normalized_score = min(1.0, score / max_possible_score) if max_possible_score > 0 else 0.0

        return normalized_score, details, matched_criteria

    def _check_magic_bytes(self, data: bytes, signature: ProtocolSignature) -> Tuple[float, Dict[str, Any]]:
        """Check magic bytes match."""
        if not signature.magic_bytes:
            return 0.0, {}

        matches = 0
        details = {"matched": [], "expected": len(signature.magic_bytes)}

        for offset, magic in signature.magic_bytes:
            if offset + len(magic) <= len(data):
                if data[offset : offset + len(magic)] == magic:
                    matches += 1
                    details["matched"].append({"offset": offset, "magic": magic.hex()})

        if matches > 0:
            score = matches / len(signature.magic_bytes)
            return score, details

        return 0.0, details

    def _check_patterns(self, data: bytes, signature: ProtocolSignature) -> Tuple[float, Dict[str, Any]]:
        """Check regex pattern matches."""
        if not signature._compiled_patterns:
            return 0.0, {}

        matches = 0
        details = {"matched": [], "total_patterns": len(signature._compiled_patterns)}

        for i, pattern in enumerate(signature._compiled_patterns):
            if pattern.search(data):
                matches += 1
                details["matched"].append(i)

        if matches > 0:
            score = matches / len(signature._compiled_patterns)
            return score, details

        return 0.0, details

    def _check_byte_sequences(self, data: bytes, signature: ProtocolSignature) -> Tuple[float, Dict[str, Any]]:
        """Check for presence of expected byte sequences."""
        if not signature.byte_sequences:
            return 0.0, {}

        matches = 0
        details = {"matched": [], "total_sequences": len(signature.byte_sequences)}

        for seq in signature.byte_sequences:
            if seq in data:
                matches += 1
                details["matched"].append(seq.hex())

        if matches > 0:
            score = matches / len(signature.byte_sequences)
            return score, details

        return 0.0, details

    def _check_structure(self, data: bytes, signature: ProtocolSignature) -> Tuple[float, Dict[str, Any]]:
        """Check structural characteristics."""
        score = 0.0
        checks = 0
        details = {}

        # Check message size
        data_len = len(data)
        if signature.min_message_size > 0 or signature.max_message_size < 65535:
            checks += 1
            if signature.min_message_size <= data_len <= signature.max_message_size:
                score += 1.0
                details["size_match"] = True
            else:
                details["size_match"] = False

        # Check for field delimiters
        if signature.field_delimiters:
            checks += 1
            delimiter_found = any(d in data for d in signature.field_delimiters)
            if delimiter_found:
                score += 1.0
                details["delimiter_found"] = True
            else:
                details["delimiter_found"] = False

        # Check for record delimiters
        if signature.record_delimiters:
            checks += 1
            record_delim_found = any(d in data for d in signature.record_delimiters)
            if record_delim_found:
                score += 1.0
                details["record_delimiter_found"] = True
            else:
                details["record_delimiter_found"] = False

        if checks > 0:
            return score / checks, details

        return 0.0, details

    def _check_statistics(self, data: bytes, signature: ProtocolSignature) -> Tuple[float, Dict[str, Any]]:
        """Check statistical characteristics."""
        import numpy as np

        if not data:
            return 0.0, {}

        score = 0.0
        checks = 0
        details = {}

        # Calculate entropy
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        byte_probs = byte_counts / len(data)
        entropy = -np.sum(byte_probs * np.log2(byte_probs + 1e-10))

        # Check entropy range
        checks += 1
        if signature.expected_entropy_range[0] <= entropy <= signature.expected_entropy_range[1]:
            score += 1.0
            details["entropy_match"] = True
        details["entropy"] = float(entropy)

        # Check ASCII ratio
        ascii_count = sum(1 for b in data if 32 <= b <= 126)
        ascii_ratio = ascii_count / len(data)

        checks += 1
        if signature.ascii_ratio_range[0] <= ascii_ratio <= signature.ascii_ratio_range[1]:
            score += 1.0
            details["ascii_ratio_match"] = True
        details["ascii_ratio"] = float(ascii_ratio)

        # Check null ratio
        null_count = data.count(0)
        null_ratio = null_count / len(data)

        checks += 1
        if signature.null_ratio_range[0] <= null_ratio <= signature.null_ratio_range[1]:
            score += 1.0
            details["null_ratio_match"] = True
        details["null_ratio"] = float(null_ratio)

        return score / checks if checks > 0 else 0.0, details

    def get_signature(self, protocol_id: str) -> Optional[ProtocolSignature]:
        """Get a specific protocol signature."""
        return self.signatures.get(protocol_id)

    def get_signatures_by_category(self, category: ProtocolCategory) -> List[ProtocolSignature]:
        """Get all signatures in a category."""
        if category not in self.category_index:
            return []
        return [self.signatures[pid] for pid in self.category_index[category]]

    def get_all_protocol_ids(self) -> List[str]:
        """Get all registered protocol IDs."""
        return list(self.signatures.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        category_counts = {cat.value: len(ids) for cat, ids in self.category_index.items()}
        return {
            "total_signatures": len(self.signatures),
            "categories": category_counts,
            "magic_bytes_indexed": len(self.magic_bytes_index),
        }


# Global instance
_signature_database: Optional[ProtocolSignatureDatabase] = None


def get_signature_database() -> ProtocolSignatureDatabase:
    """Get global signature database instance."""
    global _signature_database
    if _signature_database is None:
        _signature_database = ProtocolSignatureDatabase()
    return _signature_database
