"""
Legacy System Fingerprinter

Identifies and characterizes legacy banking systems by analyzing:
- Data formats and encodings
- Communication patterns
- Protocol signatures
- System behaviors

Supports detection of:
- Mainframe systems (IBM z/OS, AS/400)
- Legacy middleware (CICS, IMS, MQ)
- Historical protocols
- Custom proprietary formats
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib


logger = logging.getLogger(__name__)


class LegacySystemType(Enum):
    """Types of legacy systems."""

    # IBM Mainframe
    IBM_ZOSF = "ibm_zos"
    IBM_AS400 = "ibm_as400"
    IBM_CICS = "ibm_cics"
    IBM_IMS = "ibm_ims"
    IBM_MQ = "ibm_mq"

    # Data formats
    COBOL_BATCH = "cobol_batch"
    VSAM = "vsam"
    DB2 = "db2"
    IMS_DB = "ims_db"

    # Communication
    SNA_LU62 = "sna_lu62"
    APPC = "appc"
    VTAM = "vtam"

    # Banking specific
    BASE24 = "base24"
    CONNEX = "connex"
    FISERV_DNA = "fiserv_dna"
    FIS_PROFILE = "fis_profile"
    JACK_HENRY = "jack_henry"
    TEMENOS_T24 = "temenos_t24"

    # Generic
    CUSTOM_PROPRIETARY = "custom_proprietary"
    UNKNOWN = "unknown"


class FingerprintConfidence(Enum):
    """Confidence levels for fingerprint matches."""

    DEFINITE = "definite"  # > 95%
    HIGH = "high"  # 80-95%
    MEDIUM = "medium"  # 60-80%
    LOW = "low"  # 40-60%
    UNCERTAIN = "uncertain"  # < 40%


@dataclass
class SystemCharacteristic:
    """A detected characteristic of a legacy system."""

    name: str
    value: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    category: str = "general"  # encoding, protocol, format, behavior


@dataclass
class EncodingProfile:
    """Encoding characteristics of data."""

    primary_encoding: str = "EBCDIC"
    secondary_encoding: Optional[str] = None
    packed_decimal: bool = False
    binary_integers: bool = False
    comp3_detected: bool = False
    zone_decimal: bool = False
    character_set: str = "EBCDIC-500"


@dataclass
class DataLayoutProfile:
    """Data layout characteristics."""

    record_type: str = "fixed"  # fixed, variable, undefined
    record_length: Optional[int] = None
    min_record_length: Optional[int] = None
    max_record_length: Optional[int] = None
    block_size: Optional[int] = None
    has_header: bool = False
    has_trailer: bool = False
    delimiter: Optional[str] = None
    field_count: int = 0


@dataclass
class CommunicationProfile:
    """Communication characteristics."""

    protocol: str = "unknown"
    transport: str = "unknown"
    message_format: str = "unknown"
    encoding: str = "unknown"
    supports_async: bool = False
    supports_batch: bool = True
    max_message_size: Optional[int] = None


@dataclass
class SystemFingerprint:
    """Complete fingerprint of a legacy system."""

    system_type: LegacySystemType
    confidence: FingerprintConfidence
    confidence_score: float  # 0.0 to 1.0

    # Detected characteristics
    characteristics: List[SystemCharacteristic] = field(default_factory=list)

    # Profiles
    encoding_profile: EncodingProfile = field(default_factory=EncodingProfile)
    data_layout: DataLayoutProfile = field(default_factory=DataLayoutProfile)
    communication: CommunicationProfile = field(default_factory=CommunicationProfile)

    # System info
    estimated_era: str = ""  # e.g., "1980s", "1990s", "2000s"
    likely_platform: str = ""
    likely_language: str = ""

    # Migration insights
    modernization_complexity: str = "unknown"  # low, medium, high, very_high
    recommended_target: str = ""
    migration_risks: List[str] = field(default_factory=list)
    migration_opportunities: List[str] = field(default_factory=list)

    # Metadata
    fingerprinted_at: datetime = field(default_factory=datetime.utcnow)
    sample_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "system_type": self.system_type.value,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "characteristics": [
                {
                    "name": c.name,
                    "value": c.value,
                    "confidence": c.confidence,
                    "category": c.category,
                }
                for c in self.characteristics
            ],
            "encoding": {
                "primary": self.encoding_profile.primary_encoding,
                "packed_decimal": self.encoding_profile.packed_decimal,
                "comp3": self.encoding_profile.comp3_detected,
            },
            "data_layout": {
                "record_type": self.data_layout.record_type,
                "record_length": self.data_layout.record_length,
                "field_count": self.data_layout.field_count,
            },
            "communication": {
                "protocol": self.communication.protocol,
                "transport": self.communication.transport,
            },
            "estimated_era": self.estimated_era,
            "likely_platform": self.likely_platform,
            "likely_language": self.likely_language,
            "modernization": {
                "complexity": self.modernization_complexity,
                "recommended_target": self.recommended_target,
                "risks": self.migration_risks,
                "opportunities": self.migration_opportunities,
            },
            "fingerprinted_at": self.fingerprinted_at.isoformat(),
        }


class LegacyFingerprinter:
    """
    AI-powered legacy system fingerprinter.

    Analyzes data samples and system artifacts to identify:
    - System type and platform
    - Data encoding and formats
    - Communication protocols
    - Era and technology stack
    """

    def __init__(self):
        self._signatures = self._build_signatures()
        self._ebcdic_table = self._build_ebcdic_table()

    def fingerprint(
        self,
        data: bytes,
        context: Optional[Dict[str, Any]] = None,
    ) -> SystemFingerprint:
        """
        Fingerprint legacy system from data sample.

        Args:
            data: Raw data sample (file content, message, etc.)
            context: Optional context (filename, source, etc.)

        Returns:
            SystemFingerprint with detected characteristics
        """
        # Initialize fingerprint
        fingerprint = SystemFingerprint(
            system_type=LegacySystemType.UNKNOWN,
            confidence=FingerprintConfidence.UNCERTAIN,
            confidence_score=0.0,
            sample_hash=hashlib.sha256(data).hexdigest()[:16],
        )

        # Analyze encoding
        self._analyze_encoding(data, fingerprint)

        # Analyze data layout
        self._analyze_layout(data, fingerprint)

        # Detect system type
        self._detect_system_type(data, fingerprint, context)

        # Analyze communication patterns if applicable
        if context and context.get("is_message"):
            self._analyze_communication(data, fingerprint)

        # Determine era
        self._estimate_era(fingerprint)

        # Calculate modernization complexity
        self._assess_modernization(fingerprint)

        return fingerprint

    def fingerprint_copybook(
        self,
        copybook_text: str,
    ) -> SystemFingerprint:
        """
        Fingerprint from COBOL copybook.

        Args:
            copybook_text: COBOL copybook source

        Returns:
            SystemFingerprint derived from copybook analysis
        """
        fingerprint = SystemFingerprint(
            system_type=LegacySystemType.COBOL_BATCH,
            confidence=FingerprintConfidence.HIGH,
            confidence_score=0.85,
        )

        # Parse copybook
        self._parse_copybook(copybook_text, fingerprint)

        # Detect platform from copybook features
        self._detect_platform_from_copybook(copybook_text, fingerprint)

        # Assess modernization
        self._assess_modernization(fingerprint)

        return fingerprint

    def fingerprint_jcl(
        self,
        jcl_text: str,
    ) -> SystemFingerprint:
        """
        Fingerprint from JCL (Job Control Language).

        Args:
            jcl_text: JCL source

        Returns:
            SystemFingerprint derived from JCL analysis
        """
        fingerprint = SystemFingerprint(
            system_type=LegacySystemType.IBM_ZOSF,
            confidence=FingerprintConfidence.HIGH,
            confidence_score=0.9,
        )

        fingerprint.likely_platform = "IBM z/OS"
        fingerprint.likely_language = "COBOL/JCL"

        # Parse JCL for dataset information
        self._parse_jcl(jcl_text, fingerprint)

        return fingerprint

    def compare_fingerprints(
        self,
        fp1: SystemFingerprint,
        fp2: SystemFingerprint,
    ) -> Dict[str, Any]:
        """
        Compare two fingerprints for compatibility analysis.

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint

        Returns:
            Compatibility report
        """
        # Encoding compatibility
        encoding_compatible = (
            fp1.encoding_profile.primary_encoding ==
            fp2.encoding_profile.primary_encoding
        )

        # Layout compatibility
        layout_compatible = (
            fp1.data_layout.record_type == fp2.data_layout.record_type
        )

        # Same platform
        same_platform = fp1.likely_platform == fp2.likely_platform

        compatibility_score = sum([
            0.4 if encoding_compatible else 0,
            0.3 if layout_compatible else 0,
            0.3 if same_platform else 0,
        ])

        return {
            "compatibility_score": compatibility_score,
            "encoding_compatible": encoding_compatible,
            "layout_compatible": layout_compatible,
            "same_platform": same_platform,
            "transformation_required": not encoding_compatible,
            "recommendations": self._generate_compatibility_recommendations(
                fp1, fp2, encoding_compatible, layout_compatible
            ),
        }

    # =========================================================================
    # Encoding Analysis
    # =========================================================================

    def _analyze_encoding(self, data: bytes, fingerprint: SystemFingerprint) -> None:
        """Analyze data encoding."""
        profile = EncodingProfile()

        # Check for EBCDIC
        ebcdic_score = self._calculate_ebcdic_score(data)
        ascii_score = self._calculate_ascii_score(data)

        if ebcdic_score > ascii_score:
            profile.primary_encoding = "EBCDIC"
            fingerprint.characteristics.append(SystemCharacteristic(
                name="encoding",
                value="EBCDIC",
                confidence=ebcdic_score,
                category="encoding",
                evidence=["Character frequency analysis indicates EBCDIC"],
            ))
        else:
            profile.primary_encoding = "ASCII"

        # Check for packed decimal (COMP-3)
        if self._detect_packed_decimal(data):
            profile.packed_decimal = True
            profile.comp3_detected = True
            fingerprint.characteristics.append(SystemCharacteristic(
                name="packed_decimal",
                value="COMP-3 detected",
                confidence=0.8,
                category="encoding",
            ))

        # Check for binary integers
        if self._detect_binary_integers(data):
            profile.binary_integers = True

        fingerprint.encoding_profile = profile

    def _calculate_ebcdic_score(self, data: bytes) -> float:
        """Calculate likelihood of EBCDIC encoding."""
        if not data:
            return 0.0

        # EBCDIC lowercase letters are in range 0x81-0x89, 0x91-0x99, 0xA2-0xA9
        # EBCDIC uppercase letters are in range 0xC1-0xC9, 0xD1-0xD9, 0xE2-0xE9
        # EBCDIC digits are in range 0xF0-0xF9

        ebcdic_chars = set(range(0x81, 0x8A)) | set(range(0x91, 0x9A)) | set(range(0xA2, 0xAA))
        ebcdic_chars |= set(range(0xC1, 0xCA)) | set(range(0xD1, 0xDA)) | set(range(0xE2, 0xEA))
        ebcdic_chars |= set(range(0xF0, 0xFA))  # Digits

        ebcdic_count = sum(1 for b in data if b in ebcdic_chars)

        return min(ebcdic_count / len(data), 1.0)

    def _calculate_ascii_score(self, data: bytes) -> float:
        """Calculate likelihood of ASCII encoding."""
        if not data:
            return 0.0

        # Printable ASCII range: 0x20-0x7E
        ascii_count = sum(1 for b in data if 0x20 <= b <= 0x7E or b in (0x0A, 0x0D, 0x09))

        return min(ascii_count / len(data), 1.0)

    def _detect_packed_decimal(self, data: bytes) -> bool:
        """Detect COMP-3 packed decimal fields."""
        # Packed decimal has sign nibble in last byte: 0xC (positive), 0xD (negative), 0xF (unsigned)
        for i in range(len(data) - 1):
            byte = data[i]
            # Check for valid packed decimal pattern
            high_nibble = (byte >> 4) & 0x0F
            low_nibble = byte & 0x0F

            if high_nibble <= 9 and low_nibble <= 9:
                next_byte = data[i + 1]
                sign = next_byte & 0x0F
                if sign in (0x0C, 0x0D, 0x0F):
                    return True

        return False

    def _detect_binary_integers(self, data: bytes) -> bool:
        """Detect binary integer fields."""
        # Look for patterns suggesting COMP (binary) fields
        # This is a heuristic - look for sequences of 2 or 4 bytes that could be integers
        if len(data) < 4:
            return False

        # Check for common binary patterns
        for i in range(0, len(data) - 3, 4):
            # Big-endian integer check
            val = int.from_bytes(data[i:i+4], 'big')
            if 0 < val < 1000000000:  # Reasonable integer value
                return True

        return False

    # =========================================================================
    # Layout Analysis
    # =========================================================================

    def _analyze_layout(self, data: bytes, fingerprint: SystemFingerprint) -> None:
        """Analyze data layout."""
        layout = DataLayoutProfile()

        # Check for fixed-length records
        lines = data.split(b'\n') if b'\n' in data else [data]

        if len(lines) > 1:
            lengths = [len(line) for line in lines if line]
            if lengths:
                # Check if all records same length
                if len(set(lengths)) == 1:
                    layout.record_type = "fixed"
                    layout.record_length = lengths[0]
                else:
                    layout.record_type = "variable"
                    layout.min_record_length = min(lengths)
                    layout.max_record_length = max(lengths)

                # Estimate field count based on common patterns
                layout.field_count = self._estimate_field_count(lines[0] if lines else b'')
        else:
            # Single record or no newlines
            layout.record_type = "fixed"
            layout.record_length = len(data)

        # Check for RDW (Record Descriptor Word)
        if len(data) >= 4:
            potential_rdw = int.from_bytes(data[0:2], 'big')
            if 4 <= potential_rdw <= len(data):
                layout.record_type = "variable"
                fingerprint.characteristics.append(SystemCharacteristic(
                    name="rdw_detected",
                    value="Variable length records with RDW",
                    confidence=0.7,
                    category="format",
                ))

        fingerprint.data_layout = layout

    def _estimate_field_count(self, record: bytes) -> int:
        """Estimate number of fields in a record."""
        # Look for field delimiters or boundaries
        # Heuristic: count transitions between character types

        if not record:
            return 0

        transitions = 0
        prev_type = None

        for byte in record:
            if byte in (0x40, 0x20):  # EBCDIC space or ASCII space
                curr_type = "space"
            elif 0xF0 <= byte <= 0xF9 or 0x30 <= byte <= 0x39:  # Digits
                curr_type = "digit"
            else:
                curr_type = "alpha"

            if prev_type and curr_type != prev_type:
                transitions += 1

            prev_type = curr_type

        return max(transitions // 2, 1)

    # =========================================================================
    # System Type Detection
    # =========================================================================

    def _detect_system_type(
        self,
        data: bytes,
        fingerprint: SystemFingerprint,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Detect legacy system type."""
        scores: Dict[LegacySystemType, float] = {}

        # Check context hints
        if context:
            filename = context.get("filename", "").lower()
            if ".cob" in filename or ".cpy" in filename:
                scores[LegacySystemType.COBOL_BATCH] = 0.5
            if ".jcl" in filename:
                scores[LegacySystemType.IBM_ZOSF] = 0.6
            if ".bms" in filename:
                scores[LegacySystemType.IBM_CICS] = 0.7
            if ".mfs" in filename:
                scores[LegacySystemType.IBM_IMS] = 0.7

        # Analyze content signatures
        text = data.decode("latin-1", errors="replace")

        for system_type, signatures in self._signatures.items():
            score = 0.0
            evidence = []

            for pattern, weight, description in signatures:
                if re.search(pattern, text, re.IGNORECASE):
                    score += weight
                    evidence.append(description)

            if score > 0:
                scores[system_type] = min(score, 1.0)
                if score > 0.3:
                    fingerprint.characteristics.append(SystemCharacteristic(
                        name=f"{system_type.value}_detection",
                        value="Detected",
                        confidence=score,
                        evidence=evidence,
                        category="system",
                    ))

        # Encoding-based inference
        if fingerprint.encoding_profile.primary_encoding == "EBCDIC":
            scores[LegacySystemType.IBM_ZOSF] = scores.get(LegacySystemType.IBM_ZOSF, 0) + 0.3

        if fingerprint.encoding_profile.comp3_detected:
            scores[LegacySystemType.COBOL_BATCH] = scores.get(LegacySystemType.COBOL_BATCH, 0) + 0.2

        # Get best match
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            fingerprint.system_type = best_type[0]
            fingerprint.confidence_score = best_type[1]
            fingerprint.confidence = self._score_to_confidence(best_type[1])

            # Set platform info
            self._set_platform_info(fingerprint)

    def _score_to_confidence(self, score: float) -> FingerprintConfidence:
        """Convert numeric score to confidence level."""
        if score >= 0.95:
            return FingerprintConfidence.DEFINITE
        elif score >= 0.8:
            return FingerprintConfidence.HIGH
        elif score >= 0.6:
            return FingerprintConfidence.MEDIUM
        elif score >= 0.4:
            return FingerprintConfidence.LOW
        else:
            return FingerprintConfidence.UNCERTAIN

    def _set_platform_info(self, fingerprint: SystemFingerprint) -> None:
        """Set platform information based on detected system type."""
        platform_map = {
            LegacySystemType.IBM_ZOSF: ("IBM z/OS", "COBOL/Assembler"),
            LegacySystemType.IBM_AS400: ("IBM AS/400 (iSeries)", "RPG/COBOL"),
            LegacySystemType.IBM_CICS: ("IBM z/OS with CICS", "COBOL"),
            LegacySystemType.IBM_IMS: ("IBM z/OS with IMS", "COBOL/PL/I"),
            LegacySystemType.IBM_MQ: ("IBM MQ", "Various"),
            LegacySystemType.COBOL_BATCH: ("Mainframe", "COBOL"),
            LegacySystemType.BASE24: ("ACI BASE24", "TAL/COBOL"),
            LegacySystemType.FISERV_DNA: ("Fiserv DNA", "Proprietary"),
            LegacySystemType.TEMENOS_T24: ("Temenos T24", "TAFL"),
        }

        if fingerprint.system_type in platform_map:
            platform, language = platform_map[fingerprint.system_type]
            fingerprint.likely_platform = platform
            fingerprint.likely_language = language

    def _build_signatures(self) -> Dict[LegacySystemType, List[Tuple[str, float, str]]]:
        """Build detection signatures for each system type."""
        return {
            LegacySystemType.IBM_CICS: [
                (r"EXEC\s+CICS", 0.8, "EXEC CICS command found"),
                (r"DFHCOMMAREA", 0.7, "DFHCOMMAREA reference"),
                (r"EIBCALEN", 0.6, "EIB field reference"),
                (r"DFHBMSCA", 0.6, "BMS attribute reference"),
            ],
            LegacySystemType.IBM_IMS: [
                (r"EXEC\s+DLI", 0.8, "EXEC DLI command found"),
                (r"PCB\s+MASK", 0.7, "PCB definition"),
                (r"GU\s+", 0.4, "IMS call (GU)"),
                (r"ISRT\s+", 0.4, "IMS call (ISRT)"),
            ],
            LegacySystemType.IBM_MQ: [
                (r"MQOPEN", 0.8, "MQ API call"),
                (r"MQPUT", 0.7, "MQ PUT"),
                (r"MQGET", 0.7, "MQ GET"),
                (r"MQMD", 0.6, "MQ Message Descriptor"),
            ],
            LegacySystemType.COBOL_BATCH: [
                (r"IDENTIFICATION\s+DIVISION", 0.6, "COBOL division"),
                (r"DATA\s+DIVISION", 0.5, "COBOL data division"),
                (r"PROCEDURE\s+DIVISION", 0.5, "COBOL procedure division"),
                (r"PIC\s+[X9]", 0.6, "PIC clause"),
                (r"WORKING-STORAGE", 0.5, "Working storage"),
            ],
            LegacySystemType.BASE24: [
                (r"BASE24", 0.9, "BASE24 reference"),
                (r"ISO8583", 0.5, "ISO 8583 reference"),
                (r"POSTILION", 0.4, "Postilion reference"),
            ],
            LegacySystemType.TEMENOS_T24: [
                (r"T24", 0.6, "T24 reference"),
                (r"TEMENOS", 0.8, "Temenos reference"),
                (r"TAFL", 0.7, "TAFL language"),
            ],
        }

    # =========================================================================
    # Communication Analysis
    # =========================================================================

    def _analyze_communication(self, data: bytes, fingerprint: SystemFingerprint) -> None:
        """Analyze communication patterns."""
        comm = CommunicationProfile()

        text = data.decode("latin-1", errors="replace")

        # Detect MQ
        if "MQMD" in text or "MQGET" in text:
            comm.protocol = "IBM MQ"
            comm.transport = "TCP/IP"
            comm.supports_async = True

        # Detect CICS
        if "EXEC CICS" in text:
            comm.protocol = "CICS"
            comm.transport = "SNA/TCP"

        # Detect HTTP/REST (modern wrapper)
        if "HTTP" in text or "REST" in text:
            comm.protocol = "HTTP"
            comm.transport = "TCP/IP"

        fingerprint.communication = comm

    # =========================================================================
    # Era Estimation
    # =========================================================================

    def _estimate_era(self, fingerprint: SystemFingerprint) -> None:
        """Estimate the era of the system."""
        era_indicators = {
            "1970s": ["EBCDIC", "VSAM", "IMS"],
            "1980s": ["CICS", "DB2", "COBOL-74", "SNA"],
            "1990s": ["COBOL-85", "MQ", "TCP/IP"],
            "2000s": ["XML", "web services", "Java"],
        }

        # Default based on system type
        era_map = {
            LegacySystemType.IBM_IMS: "1970s-1980s",
            LegacySystemType.IBM_CICS: "1980s-1990s",
            LegacySystemType.COBOL_BATCH: "1980s",
            LegacySystemType.IBM_MQ: "1990s",
            LegacySystemType.BASE24: "1980s-1990s",
        }

        fingerprint.estimated_era = era_map.get(
            fingerprint.system_type,
            "Unknown"
        )

    # =========================================================================
    # Modernization Assessment
    # =========================================================================

    def _assess_modernization(self, fingerprint: SystemFingerprint) -> None:
        """Assess modernization complexity and provide recommendations."""
        # Calculate complexity factors
        complexity_score = 0

        # Encoding complexity
        if fingerprint.encoding_profile.primary_encoding == "EBCDIC":
            complexity_score += 2
            fingerprint.migration_risks.append("EBCDIC to UTF-8 conversion required")

        if fingerprint.encoding_profile.comp3_detected:
            complexity_score += 1
            fingerprint.migration_risks.append("Packed decimal conversion needed")

        # Platform complexity
        if fingerprint.system_type in (LegacySystemType.IBM_CICS, LegacySystemType.IBM_IMS):
            complexity_score += 3
            fingerprint.migration_risks.append("Transaction monitor dependency")

        # Data layout complexity
        if fingerprint.data_layout.record_type == "variable":
            complexity_score += 1

        # Determine complexity level
        if complexity_score <= 2:
            fingerprint.modernization_complexity = "low"
        elif complexity_score <= 4:
            fingerprint.modernization_complexity = "medium"
        elif complexity_score <= 6:
            fingerprint.modernization_complexity = "high"
        else:
            fingerprint.modernization_complexity = "very_high"

        # Set recommended target
        target_map = {
            LegacySystemType.COBOL_BATCH: "Cloud-native microservices with ISO 20022",
            LegacySystemType.IBM_CICS: "Containerized services with API gateway",
            LegacySystemType.IBM_IMS: "Event-driven architecture with modern database",
            LegacySystemType.IBM_MQ: "Cloud messaging (AWS SQS, Azure Service Bus)",
        }

        fingerprint.recommended_target = target_map.get(
            fingerprint.system_type,
            "Cloud-native architecture"
        )

        # Opportunities
        fingerprint.migration_opportunities = [
            "PQC-ready cryptography",
            "Real-time processing",
            "API-first design",
            "Cloud scalability",
        ]

    # =========================================================================
    # Copybook and JCL Parsing
    # =========================================================================

    def _parse_copybook(self, text: str, fingerprint: SystemFingerprint) -> None:
        """Parse COBOL copybook for detailed analysis."""
        # Count PIC clauses
        pic_count = len(re.findall(r'PIC\s+', text, re.IGNORECASE))

        fingerprint.data_layout.field_count = pic_count

        # Detect COMP-3
        if re.search(r'COMP-3', text, re.IGNORECASE):
            fingerprint.encoding_profile.comp3_detected = True

        # Detect COMP (binary)
        if re.search(r'COMP\b', text, re.IGNORECASE):
            fingerprint.encoding_profile.binary_integers = True

        # Calculate record length from OCCURS and PIC
        fingerprint.characteristics.append(SystemCharacteristic(
            name="copybook_fields",
            value=str(pic_count),
            confidence=0.95,
            category="format",
        ))

    def _detect_platform_from_copybook(self, text: str, fingerprint: SystemFingerprint) -> None:
        """Detect platform specifics from copybook."""
        # IBM specific
        if re.search(r'SYNC|SYNCHRONIZED', text, re.IGNORECASE):
            fingerprint.likely_platform = "IBM Mainframe"

        # CICS specific
        if re.search(r'DFHCOMMAREA|DFHEIBLK', text, re.IGNORECASE):
            fingerprint.system_type = LegacySystemType.IBM_CICS

    def _parse_jcl(self, text: str, fingerprint: SystemFingerprint) -> None:
        """Parse JCL for system information."""
        # Extract DD statements
        dd_count = len(re.findall(r'//\w+\s+DD\s+', text))

        fingerprint.characteristics.append(SystemCharacteristic(
            name="jcl_datasets",
            value=str(dd_count),
            confidence=0.9,
            category="system",
        ))

        # Detect program names
        pgm_match = re.findall(r'PGM=(\w+)', text)
        if pgm_match:
            fingerprint.characteristics.append(SystemCharacteristic(
                name="programs",
                value=",".join(set(pgm_match)),
                confidence=0.95,
                category="system",
            ))

        # Detect CICS or IMS
        if any(p.startswith("DFH") for p in pgm_match):
            fingerprint.system_type = LegacySystemType.IBM_CICS
        elif any(p.startswith("DFS") for p in pgm_match):
            fingerprint.system_type = LegacySystemType.IBM_IMS

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_ebcdic_table(self) -> Dict[int, str]:
        """Build EBCDIC to ASCII translation table."""
        # Simplified table for common characters
        table = {}
        # Uppercase letters
        for i, c in enumerate("ABCDEFGHI"):
            table[0xC1 + i] = c
        for i, c in enumerate("JKLMNOPQR"):
            table[0xD1 + i] = c
        for i, c in enumerate("STUVWXYZ"):
            table[0xE2 + i] = c
        # Lowercase letters
        for i, c in enumerate("abcdefghi"):
            table[0x81 + i] = c
        for i, c in enumerate("jklmnopqr"):
            table[0x91 + i] = c
        for i, c in enumerate("stuvwxyz"):
            table[0xA2 + i] = c
        # Digits
        for i in range(10):
            table[0xF0 + i] = str(i)
        # Space
        table[0x40] = " "

        return table

    def _generate_compatibility_recommendations(
        self,
        fp1: SystemFingerprint,
        fp2: SystemFingerprint,
        encoding_compatible: bool,
        layout_compatible: bool,
    ) -> List[str]:
        """Generate compatibility recommendations."""
        recommendations = []

        if not encoding_compatible:
            recommendations.append(
                f"Implement encoding conversion from {fp1.encoding_profile.primary_encoding} "
                f"to {fp2.encoding_profile.primary_encoding}"
            )

        if not layout_compatible:
            recommendations.append(
                f"Transform record format from {fp1.data_layout.record_type} "
                f"to {fp2.data_layout.record_type}"
            )

        if fp1.encoding_profile.comp3_detected and not fp2.encoding_profile.comp3_detected:
            recommendations.append("Convert packed decimal (COMP-3) to standard numeric")

        return recommendations
