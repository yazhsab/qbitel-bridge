"""
Protocol Analyzer

Deep analysis engine for banking and financial protocols.
Analyzes protocol structures, behaviors, and capabilities.

Supported Protocols:
- ISO 20022 (pain, pacs, camt, etc.)
- SWIFT MT messages
- FIX protocol
- ACH/NACHA
- FedWire/FedNow
- SEPA
- EMV/chip cards
- 3D Secure
- Legacy mainframe protocols
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Types of protocols supported."""

    # Payment protocols
    ISO20022 = "iso20022"
    SWIFT_MT = "swift_mt"
    ACH_NACHA = "ach_nacha"
    FEDWIRE = "fedwire"
    FEDNOW = "fednow"
    SEPA = "sepa"

    # Trading protocols
    FIX = "fix"
    FIXML = "fixml"

    # Card protocols
    EMV = "emv"
    THREE_DS = "3ds"

    # Legacy protocols
    COBOL_COPYBOOK = "cobol_copybook"
    CICS_BMS = "cics_bms"
    IMS_MFS = "ims_mfs"
    EBCDIC_FIXED = "ebcdic_fixed"

    # Generic
    XML = "xml"
    JSON = "json"
    FIXED_WIDTH = "fixed_width"
    UNKNOWN = "unknown"


class ProtocolCapability(Enum):
    """Capabilities of analyzed protocols."""

    PAYMENT_INITIATION = "payment_initiation"
    PAYMENT_CLEARING = "payment_clearing"
    PAYMENT_SETTLEMENT = "payment_settlement"
    BALANCE_REPORTING = "balance_reporting"
    STATEMENT_GENERATION = "statement_generation"
    TRADE_EXECUTION = "trade_execution"
    MARKET_DATA = "market_data"
    CARD_AUTHORIZATION = "card_authorization"
    CARD_AUTHENTICATION = "card_authentication"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME_PROCESSING = "real_time_processing"
    INQUIRY = "inquiry"
    NOTIFICATION = "notification"


@dataclass
class ProtocolField:
    """Definition of a protocol field."""

    name: str
    data_type: str
    length: Optional[int] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    required: bool = False
    pattern: Optional[str] = None
    description: str = ""
    path: str = ""  # XPath or similar
    example: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    security_classification: str = "internal"  # public, internal, confidential, restricted


@dataclass
class ProtocolMessage:
    """Definition of a protocol message type."""

    name: str
    message_type: str
    protocol: ProtocolType
    fields: List[ProtocolField] = field(default_factory=list)
    capabilities: List[ProtocolCapability] = field(default_factory=list)
    description: str = ""
    version: str = ""
    namespace: Optional[str] = None


@dataclass
class SecurityCharacteristic:
    """Security characteristic of a protocol."""

    name: str
    present: bool
    strength: str = "unknown"  # none, weak, moderate, strong
    notes: str = ""
    recommendation: str = ""


@dataclass
class ProtocolAnalysisResult:
    """Result of protocol analysis."""

    protocol_type: ProtocolType
    confidence: float  # 0.0 to 1.0
    version: Optional[str] = None
    encoding: str = "UTF-8"

    # Structure analysis
    messages: List[ProtocolMessage] = field(default_factory=list)
    fields: List[ProtocolField] = field(default_factory=list)
    total_field_count: int = 0

    # Capabilities
    capabilities: Set[ProtocolCapability] = field(default_factory=set)

    # Security analysis
    security_characteristics: List[SecurityCharacteristic] = field(default_factory=list)
    pqc_ready: bool = False
    encryption_detected: bool = False
    signature_detected: bool = False

    # Compliance hints
    compliance_frameworks: List[str] = field(default_factory=list)
    pci_relevant_fields: List[str] = field(default_factory=list)

    # Quality metrics
    schema_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    # Analysis metadata
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    analysis_duration_ms: float = 0
    sample_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "protocol_type": self.protocol_type.value,
            "confidence": self.confidence,
            "version": self.version,
            "encoding": self.encoding,
            "message_count": len(self.messages),
            "field_count": self.total_field_count,
            "capabilities": [c.value for c in self.capabilities],
            "security": {
                "pqc_ready": self.pqc_ready,
                "encryption_detected": self.encryption_detected,
                "signature_detected": self.signature_detected,
                "characteristics": [
                    {"name": s.name, "present": s.present, "strength": s.strength} for s in self.security_characteristics
                ],
            },
            "compliance": {
                "frameworks": self.compliance_frameworks,
                "pci_relevant_fields": self.pci_relevant_fields,
            },
            "validation": {
                "schema_valid": self.schema_valid,
                "error_count": len(self.validation_errors),
            },
            "analyzed_at": self.analyzed_at.isoformat(),
            "analysis_duration_ms": self.analysis_duration_ms,
        }


class ProtocolAnalyzer:
    """
    AI-powered protocol analyzer for banking and financial systems.

    Capabilities:
    - Protocol type detection
    - Message structure analysis
    - Field extraction and classification
    - Security posture assessment
    - Compliance mapping
    - PQC readiness evaluation
    """

    def __init__(self):
        self._protocol_signatures = self._build_protocol_signatures()
        self._field_patterns = self._build_field_patterns()
        self._pci_field_patterns = self._build_pci_patterns()

    def analyze(
        self,
        data: bytes,
        hint: Optional[ProtocolType] = None,
    ) -> ProtocolAnalysisResult:
        """
        Analyze protocol data and extract intelligence.

        Args:
            data: Raw protocol data (message, schema, or sample)
            hint: Optional hint about expected protocol type

        Returns:
            ProtocolAnalysisResult with analysis details
        """
        start_time = datetime.utcnow()

        try:
            # Decode data
            text, encoding = self._detect_encoding_and_decode(data)

            # Detect protocol type
            protocol_type, confidence = self._detect_protocol_type(text, hint)

            # Create result
            result = ProtocolAnalysisResult(
                protocol_type=protocol_type,
                confidence=confidence,
                encoding=encoding,
                sample_hash=hashlib.sha256(data).hexdigest()[:16],
            )

            # Perform protocol-specific analysis
            if protocol_type == ProtocolType.ISO20022:
                self._analyze_iso20022(text, result)
            elif protocol_type == ProtocolType.SWIFT_MT:
                self._analyze_swift_mt(text, result)
            elif protocol_type == ProtocolType.FIX:
                self._analyze_fix(text, result)
            elif protocol_type == ProtocolType.ACH_NACHA:
                self._analyze_ach_nacha(text, result)
            elif protocol_type == ProtocolType.EMV:
                self._analyze_emv(text, result)
            elif protocol_type == ProtocolType.THREE_DS:
                self._analyze_3ds(text, result)
            elif protocol_type in (ProtocolType.COBOL_COPYBOOK, ProtocolType.EBCDIC_FIXED):
                self._analyze_legacy_fixed(text, result)
            else:
                self._analyze_generic(text, result)

            # Analyze security characteristics
            self._analyze_security(text, result)

            # Map compliance requirements
            self._map_compliance(result)

            # Calculate duration
            result.analysis_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return result

        except Exception as e:
            logger.error(f"Protocol analysis failed: {e}")
            return ProtocolAnalysisResult(
                protocol_type=ProtocolType.UNKNOWN,
                confidence=0.0,
                validation_errors=[str(e)],
            )

    def analyze_schema(
        self,
        schema_data: bytes,
        schema_type: str = "xsd",
    ) -> ProtocolAnalysisResult:
        """
        Analyze a protocol schema (XSD, JSON Schema, COBOL copybook).

        Args:
            schema_data: Schema definition
            schema_type: Type of schema (xsd, json_schema, copybook)

        Returns:
            ProtocolAnalysisResult with schema analysis
        """
        text, encoding = self._detect_encoding_and_decode(schema_data)

        if schema_type == "xsd":
            return self._analyze_xsd_schema(text)
        elif schema_type == "json_schema":
            return self._analyze_json_schema(text)
        elif schema_type == "copybook":
            return self._analyze_copybook(text)
        else:
            return self.analyze(schema_data)

    def compare_protocols(
        self,
        source: ProtocolAnalysisResult,
        target: ProtocolAnalysisResult,
    ) -> Dict[str, Any]:
        """
        Compare two protocol analyses to identify mapping opportunities.

        Args:
            source: Source protocol analysis
            target: Target protocol analysis

        Returns:
            Comparison report with field mappings and gaps
        """
        # Field matching
        field_mappings = []
        unmapped_source = []
        unmapped_target = []

        source_fields = {f.name.lower(): f for f in source.fields}
        target_fields = {f.name.lower(): f for f in target.fields}

        for name, source_field in source_fields.items():
            if name in target_fields:
                target_field = target_fields[name]
                field_mappings.append(
                    {
                        "source": source_field.name,
                        "target": target_field.name,
                        "type_compatible": source_field.data_type == target_field.data_type,
                        "confidence": 1.0,
                    }
                )
            else:
                # Try fuzzy matching
                best_match = self._find_best_field_match(source_field, target_fields.values())
                if best_match:
                    field_mappings.append(
                        {
                            "source": source_field.name,
                            "target": best_match[0].name,
                            "type_compatible": source_field.data_type == best_match[0].data_type,
                            "confidence": best_match[1],
                        }
                    )
                else:
                    unmapped_source.append(source_field.name)

        # Find unmapped target fields
        mapped_targets = {m["target"] for m in field_mappings}
        for name, field in target_fields.items():
            if field.name not in mapped_targets:
                unmapped_target.append(field.name)

        # Capability comparison
        capability_overlap = source.capabilities & target.capabilities
        source_only_capabilities = source.capabilities - target.capabilities
        target_only_capabilities = target.capabilities - source.capabilities

        return {
            "source_protocol": source.protocol_type.value,
            "target_protocol": target.protocol_type.value,
            "field_mappings": field_mappings,
            "unmapped_source_fields": unmapped_source,
            "unmapped_target_fields": unmapped_target,
            "mapping_coverage": len(field_mappings) / max(len(source_fields), 1),
            "capabilities": {
                "overlap": [c.value for c in capability_overlap],
                "source_only": [c.value for c in source_only_capabilities],
                "target_only": [c.value for c in target_only_capabilities],
            },
            "security_comparison": {
                "source_pqc_ready": source.pqc_ready,
                "target_pqc_ready": target.pqc_ready,
            },
            "migration_complexity": self._calculate_migration_complexity(field_mappings, unmapped_source, unmapped_target),
        }

    # =========================================================================
    # Protocol Detection
    # =========================================================================

    def _detect_protocol_type(
        self,
        text: str,
        hint: Optional[ProtocolType] = None,
    ) -> Tuple[ProtocolType, float]:
        """Detect protocol type from content."""
        scores: Dict[ProtocolType, float] = {}

        for protocol, signatures in self._protocol_signatures.items():
            score = 0.0
            for sig, weight in signatures:
                if sig in text:
                    score += weight
            scores[protocol] = min(score, 1.0)

        # If hint provided, boost that score
        if hint and hint in scores:
            scores[hint] = min(scores[hint] + 0.3, 1.0)

        # Get best match
        if not scores:
            return ProtocolType.UNKNOWN, 0.0

        best_protocol = max(scores.items(), key=lambda x: x[1])

        if best_protocol[1] < 0.2:
            return ProtocolType.UNKNOWN, best_protocol[1]

        return best_protocol[0], best_protocol[1]

    def _build_protocol_signatures(self) -> Dict[ProtocolType, List[Tuple[str, float]]]:
        """Build signature patterns for protocol detection."""
        return {
            ProtocolType.ISO20022: [
                ("urn:iso:std:iso:20022", 0.9),
                ("pain.", 0.6),
                ("pacs.", 0.6),
                ("camt.", 0.6),
                ("Document xmlns", 0.4),
                ("MsgId", 0.3),
                ("CreDtTm", 0.3),
            ],
            ProtocolType.SWIFT_MT: [
                ("{1:", 0.5),
                ("{2:", 0.3),
                ("{4:", 0.3),
                (":20:", 0.4),
                (":32A:", 0.4),
                (":50K:", 0.3),
                (":59:", 0.3),
            ],
            ProtocolType.FIX: [
                ("8=FIX", 0.9),
                ("35=", 0.4),
                ("49=", 0.3),
                ("56=", 0.3),
                ("\x01", 0.2),  # SOH delimiter
            ],
            ProtocolType.ACH_NACHA: [
                ("101 ", 0.6),  # File header
                ("501 ", 0.5),  # Batch header
                ("620", 0.4),
                ("NACHA", 0.3),
            ],
            ProtocolType.EMV: [
                ("9F26", 0.7),  # Application Cryptogram tag
                ("9F27", 0.5),
                ("5A", 0.3),  # PAN tag
                ("5F24", 0.3),  # Expiry tag
            ],
            ProtocolType.THREE_DS: [
                ("threeDSServerTransID", 0.8),
                ("acsTransID", 0.6),
                ("messageVersion", 0.4),
                ("deviceChannel", 0.4),
            ],
            ProtocolType.COBOL_COPYBOOK: [
                ("01 ", 0.4),
                ("05 ", 0.3),
                ("PIC ", 0.8),
                ("PIC X", 0.6),
                ("PIC 9", 0.6),
                ("COMP-3", 0.5),
            ],
            ProtocolType.SEPA: [
                ("CstmrCdtTrfInitn", 0.8),
                ("CstmrDrctDbtInitn", 0.8),
                ("SEPA", 0.5),
                ("BIC", 0.2),
                ("IBAN", 0.3),
            ],
        }

    # =========================================================================
    # Protocol-Specific Analyzers
    # =========================================================================

    def _analyze_iso20022(self, text: str, result: ProtocolAnalysisResult) -> None:
        """Analyze ISO 20022 message."""
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(text)

            # Detect message type from namespace
            ns_match = re.search(r'xmlns="([^"]+)"', text)
            if ns_match:
                ns = ns_match.group(1)
                result.version = ns

                # Determine capabilities from message type
                if "pain" in ns:
                    result.capabilities.add(ProtocolCapability.PAYMENT_INITIATION)
                    if "001" in ns:
                        result.messages.append(
                            ProtocolMessage(
                                name="CustomerCreditTransferInitiation",
                                message_type="pain.001",
                                protocol=ProtocolType.ISO20022,
                                capabilities=[ProtocolCapability.PAYMENT_INITIATION],
                            )
                        )
                elif "pacs" in ns:
                    result.capabilities.add(ProtocolCapability.PAYMENT_CLEARING)
                elif "camt" in ns:
                    result.capabilities.add(ProtocolCapability.BALANCE_REPORTING)
                    result.capabilities.add(ProtocolCapability.STATEMENT_GENERATION)

            # Extract fields
            self._extract_xml_fields(root, result, "")

            result.schema_valid = True

        except ET.ParseError as e:
            result.validation_errors.append(f"XML parse error: {e}")
            result.schema_valid = False

    def _analyze_swift_mt(self, text: str, result: ProtocolAnalysisResult) -> None:
        """Analyze SWIFT MT message."""
        # Detect message type
        mt_match = re.search(r"\{2:[IO](\d{3})", text)
        if mt_match:
            mt_type = mt_match.group(1)
            result.version = f"MT{mt_type}"

            # Map MT type to capabilities
            mt_capabilities = {
                "103": [ProtocolCapability.PAYMENT_CLEARING],
                "202": [ProtocolCapability.PAYMENT_CLEARING],
                "940": [ProtocolCapability.STATEMENT_GENERATION],
                "950": [ProtocolCapability.STATEMENT_GENERATION],
            }

            for cap in mt_capabilities.get(mt_type, []):
                result.capabilities.add(cap)

        # Extract fields
        field_pattern = re.compile(r":(\d{2}[A-Z]?):([^\n]+(?:\n(?!:)[^\n]+)*)")
        for match in field_pattern.finditer(text):
            tag = match.group(1)
            value = match.group(2).strip()

            field = ProtocolField(
                name=f"Field_{tag}",
                data_type="string",
                path=f":${tag}:",
                example=value[:50] if len(value) > 50 else value,
            )

            # Classify sensitive fields
            if tag in ("50K", "59", "50A", "59A"):  # Account holder fields
                field.security_classification = "confidential"
            elif tag in ("32A", "33B"):  # Amount fields
                field.security_classification = "restricted"

            result.fields.append(field)

        result.total_field_count = len(result.fields)

    def _analyze_fix(self, text: str, result: ProtocolAnalysisResult) -> None:
        """Analyze FIX protocol message."""
        # Detect FIX version
        version_match = re.search(r"8=FIX\.(\d+\.\d+)", text)
        if version_match:
            result.version = f"FIX.{version_match.group(1)}"

        # Detect message type
        msg_type_match = re.search(r"35=([A-Z0-9]+)", text)
        if msg_type_match:
            msg_type = msg_type_match.group(1)

            msg_type_map = {
                "D": ("NewOrderSingle", ProtocolCapability.TRADE_EXECUTION),
                "8": ("ExecutionReport", ProtocolCapability.TRADE_EXECUTION),
                "V": ("MarketDataRequest", ProtocolCapability.MARKET_DATA),
                "W": ("MarketDataSnapshot", ProtocolCapability.MARKET_DATA),
                "F": ("OrderCancelRequest", ProtocolCapability.TRADE_EXECUTION),
            }

            if msg_type in msg_type_map:
                name, capability = msg_type_map[msg_type]
                result.capabilities.add(capability)
                result.messages.append(
                    ProtocolMessage(
                        name=name,
                        message_type=msg_type,
                        protocol=ProtocolType.FIX,
                        capabilities=[capability],
                    )
                )

        # Extract FIX fields
        delimiter = "\x01" if "\x01" in text else "|"
        for field in text.split(delimiter):
            if "=" in field:
                tag, value = field.split("=", 1)
                result.fields.append(
                    ProtocolField(
                        name=f"Tag_{tag}",
                        data_type="string",
                        example=value[:30],
                    )
                )

        result.total_field_count = len(result.fields)

    def _analyze_ach_nacha(self, text: str, result: ProtocolAnalysisResult) -> None:
        """Analyze ACH/NACHA file."""
        result.capabilities.add(ProtocolCapability.BATCH_PROCESSING)
        result.capabilities.add(ProtocolCapability.PAYMENT_CLEARING)

        lines = text.split("\n")
        for line in lines:
            if len(line) >= 94:
                record_type = line[0]
                if record_type == "1":
                    result.messages.append(
                        ProtocolMessage(
                            name="FileHeaderRecord",
                            message_type="1",
                            protocol=ProtocolType.ACH_NACHA,
                        )
                    )
                elif record_type == "5":
                    result.messages.append(
                        ProtocolMessage(
                            name="BatchHeaderRecord",
                            message_type="5",
                            protocol=ProtocolType.ACH_NACHA,
                        )
                    )
                elif record_type == "6":
                    result.messages.append(
                        ProtocolMessage(
                            name="EntryDetailRecord",
                            message_type="6",
                            protocol=ProtocolType.ACH_NACHA,
                        )
                    )

        result.total_field_count = len(lines) * 10  # Approximate

    def _analyze_emv(self, text: str, result: ProtocolAnalysisResult) -> None:
        """Analyze EMV TLV data."""
        result.capabilities.add(ProtocolCapability.CARD_AUTHORIZATION)

        # Common EMV tags
        emv_tags = {
            "5A": ("ApplicationPAN", "restricted"),
            "5F24": ("ApplicationExpirationDate", "confidential"),
            "9F26": ("ApplicationCryptogram", "restricted"),
            "9F27": ("CryptogramInformationData", "internal"),
            "9F10": ("IssuerApplicationData", "restricted"),
        }

        for tag, (name, classification) in emv_tags.items():
            if tag in text:
                result.fields.append(
                    ProtocolField(
                        name=name,
                        data_type="hex",
                        path=f"Tag_{tag}",
                        security_classification=classification,
                    )
                )

        result.total_field_count = len(result.fields)

    def _analyze_3ds(self, text: str, result: ProtocolAnalysisResult) -> None:
        """Analyze 3D Secure message."""
        result.capabilities.add(ProtocolCapability.CARD_AUTHENTICATION)

        try:
            import json

            data = json.loads(text)

            # Detect 3DS version
            version = data.get("messageVersion", "2.1.0")
            result.version = f"3DS {version}"

            # Extract fields
            for key, value in data.items():
                field = ProtocolField(
                    name=key,
                    data_type=type(value).__name__,
                    path=f"$.{key}",
                )

                # Classify sensitive fields
                if key in ("acctNumber", "cardNumber"):
                    field.security_classification = "restricted"
                elif key in ("authenticationValue", "cavv"):
                    field.security_classification = "confidential"

                result.fields.append(field)

            result.total_field_count = len(result.fields)

        except (json.JSONDecodeError, TypeError):
            result.validation_errors.append("Invalid JSON for 3DS message")

    def _analyze_legacy_fixed(self, text: str, result: ProtocolAnalysisResult) -> None:
        """Analyze legacy fixed-width format."""
        result.capabilities.add(ProtocolCapability.BATCH_PROCESSING)

        lines = text.split("\n")
        if lines:
            # Analyze first line to detect field boundaries
            sample_line = lines[0]
            result.total_field_count = self._detect_fixed_fields(sample_line, result)

    def _analyze_generic(self, text: str, result: ProtocolAnalysisResult) -> None:
        """Generic analysis for unknown protocols."""
        # Try JSON
        try:
            import json

            data = json.loads(text)
            result.protocol_type = ProtocolType.JSON
            self._extract_json_fields(data, result, "$")
            return
        except (json.JSONDecodeError, TypeError):
            pass

        # Try XML
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(text)
            result.protocol_type = ProtocolType.XML
            self._extract_xml_fields(root, result, "")
            return
        except ET.ParseError:
            pass

        # Assume fixed-width
        result.protocol_type = ProtocolType.FIXED_WIDTH

    # =========================================================================
    # Security Analysis
    # =========================================================================

    def _analyze_security(self, text: str, result: ProtocolAnalysisResult) -> None:
        """Analyze security characteristics of the protocol."""
        # Check for encryption indicators
        encryption_patterns = [
            ("AES", "AES encryption"),
            ("RSA", "RSA encryption"),
            ("DES", "DES encryption (weak)"),
            ("3DES", "Triple DES encryption"),
            ("encrypt", "Generic encryption"),
        ]

        for pattern, name in encryption_patterns:
            if pattern.lower() in text.lower():
                result.encryption_detected = True
                result.security_characteristics.append(
                    SecurityCharacteristic(
                        name=name,
                        present=True,
                        strength="moderate" if "AES" in pattern else "weak",
                    )
                )

        # Check for signature indicators
        signature_patterns = [
            ("signature", "Digital signature"),
            ("HMAC", "HMAC authentication"),
            ("MAC", "Message authentication"),
            ("SHA-256", "SHA-256 hash"),
            ("SHA-512", "SHA-512 hash"),
        ]

        for pattern, name in signature_patterns:
            if pattern.lower() in text.lower():
                result.signature_detected = True
                result.security_characteristics.append(
                    SecurityCharacteristic(
                        name=name,
                        present=True,
                        strength="strong" if "SHA" in pattern else "moderate",
                    )
                )

        # Check PQC readiness
        pqc_patterns = ["ML-KEM", "ML-DSA", "Kyber", "Dilithium", "post-quantum"]
        result.pqc_ready = any(p.lower() in text.lower() for p in pqc_patterns)

        if not result.pqc_ready:
            result.security_characteristics.append(
                SecurityCharacteristic(
                    name="Post-Quantum Cryptography",
                    present=False,
                    strength="none",
                    recommendation="Consider PQC migration for quantum resistance",
                )
            )

        # Identify PCI-relevant fields
        for field in result.fields:
            if any(p in field.name.lower() for p in ["pan", "card", "cvv", "pin", "track"]):
                result.pci_relevant_fields.append(field.name)

    def _map_compliance(self, result: ProtocolAnalysisResult) -> None:
        """Map protocol to compliance frameworks."""
        # PCI DSS
        if result.pci_relevant_fields:
            result.compliance_frameworks.append("PCI-DSS")

        # Payment protocols
        if result.protocol_type in (
            ProtocolType.ISO20022,
            ProtocolType.SWIFT_MT,
            ProtocolType.ACH_NACHA,
        ):
            result.compliance_frameworks.append("SWIFT-CSP")

        # Card protocols
        if result.protocol_type in (ProtocolType.EMV, ProtocolType.THREE_DS):
            result.compliance_frameworks.append("PCI-DSS")
            result.compliance_frameworks.append("EMVCo")

        # Trading
        if result.protocol_type == ProtocolType.FIX:
            result.compliance_frameworks.append("MiFID-II")
            result.compliance_frameworks.append("SEC-17a-4")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _detect_encoding_and_decode(self, data: bytes) -> Tuple[str, str]:
        """Detect encoding and decode bytes to string."""
        # Try common encodings
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252", "ascii"]

        for encoding in encodings:
            try:
                text = data.decode(encoding)
                return text, encoding
            except (UnicodeDecodeError, LookupError):
                continue

        # Fall back to latin-1 which accepts any byte sequence
        return data.decode("latin-1", errors="replace"), "latin-1"

    def _extract_xml_fields(
        self,
        element: Any,
        result: ProtocolAnalysisResult,
        path: str,
    ) -> None:
        """Recursively extract fields from XML element."""
        current_path = f"{path}/{element.tag.split('}')[-1]}" if path else element.tag.split("}")[-1]

        # Extract text content
        if element.text and element.text.strip():
            result.fields.append(
                ProtocolField(
                    name=element.tag.split("}")[-1],
                    data_type="string",
                    path=current_path,
                    example=element.text.strip()[:30],
                )
            )
            result.total_field_count += 1

        # Extract attributes
        for attr_name, attr_value in element.attrib.items():
            result.fields.append(
                ProtocolField(
                    name=f"{element.tag.split('}')[-1]}@{attr_name}",
                    data_type="attribute",
                    path=f"{current_path}/@{attr_name}",
                    example=attr_value[:30],
                )
            )
            result.total_field_count += 1

        # Process children
        for child in element:
            self._extract_xml_fields(child, result, current_path)

    def _extract_json_fields(
        self,
        data: Any,
        result: ProtocolAnalysisResult,
        path: str,
    ) -> None:
        """Recursively extract fields from JSON data."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}"
                self._extract_json_fields(value, result, current_path)
        elif isinstance(data, list):
            if data:
                self._extract_json_fields(data[0], result, f"{path}[0]")
        else:
            result.fields.append(
                ProtocolField(
                    name=path.split(".")[-1],
                    data_type=type(data).__name__,
                    path=path,
                    example=str(data)[:30] if data else None,
                )
            )
            result.total_field_count += 1

    def _detect_fixed_fields(self, line: str, result: ProtocolAnalysisResult) -> int:
        """Detect field boundaries in fixed-width data."""
        # Simple heuristic: look for runs of spaces
        field_count = 0
        in_field = False

        for i, char in enumerate(line):
            if char != " " and not in_field:
                in_field = True
                field_count += 1
            elif char == " " and in_field:
                in_field = False

        return max(field_count, 1)

    def _find_best_field_match(
        self,
        source_field: ProtocolField,
        target_fields: List[ProtocolField],
    ) -> Optional[Tuple[ProtocolField, float]]:
        """Find best matching field using fuzzy matching."""
        best_match = None
        best_score = 0.0

        source_name = source_field.name.lower()

        for target in target_fields:
            target_name = target.name.lower()

            # Exact match
            if source_name == target_name:
                return (target, 1.0)

            # Partial match
            score = self._similarity_score(source_name, target_name)

            if score > best_score and score > 0.5:
                best_score = score
                best_match = target

        if best_match:
            return (best_match, best_score)
        return None

    def _similarity_score(self, s1: str, s2: str) -> float:
        """Calculate similarity score between two strings."""
        # Simple token overlap
        tokens1 = set(re.split(r"[_\s\-]", s1.lower()))
        tokens2 = set(re.split(r"[_\s\-]", s2.lower()))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def _calculate_migration_complexity(
        self,
        mappings: List[Dict],
        unmapped_source: List[str],
        unmapped_target: List[str],
    ) -> str:
        """Calculate migration complexity level."""
        coverage = len(mappings) / max(len(mappings) + len(unmapped_source), 1)
        type_incompatible = sum(1 for m in mappings if not m.get("type_compatible", True))

        if coverage > 0.9 and type_incompatible == 0:
            return "low"
        elif coverage > 0.7 and type_incompatible < 3:
            return "medium"
        elif coverage > 0.5:
            return "high"
        else:
            return "very_high"

    def _build_field_patterns(self) -> Dict[str, str]:
        """Build common field patterns."""
        return {
            "pan": r"\d{13,19}",
            "iban": r"[A-Z]{2}\d{2}[A-Z0-9]{4,30}",
            "bic": r"[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?",
            "amount": r"\d+(\.\d{2})?",
            "date_iso": r"\d{4}-\d{2}-\d{2}",
            "date_fix": r"\d{8}",
        }

    def _build_pci_patterns(self) -> List[str]:
        """Build PCI-relevant field patterns."""
        return [
            "pan",
            "primaryaccountnumber",
            "cardnumber",
            "cvv",
            "cvc",
            "cvv2",
            "cvc2",
            "pin",
            "pinblock",
            "track1",
            "track2",
            "expiry",
            "expirationdate",
        ]

    def _analyze_xsd_schema(self, text: str) -> ProtocolAnalysisResult:
        """Analyze XSD schema."""
        result = ProtocolAnalysisResult(
            protocol_type=ProtocolType.ISO20022,
            confidence=0.8,
        )

        # Extract element definitions
        element_pattern = re.compile(r'<xs:element\s+name="([^"]+)"')
        for match in element_pattern.finditer(text):
            result.fields.append(
                ProtocolField(
                    name=match.group(1),
                    data_type="element",
                )
            )

        result.total_field_count = len(result.fields)
        return result

    def _analyze_json_schema(self, text: str) -> ProtocolAnalysisResult:
        """Analyze JSON Schema."""
        import json

        result = ProtocolAnalysisResult(
            protocol_type=ProtocolType.JSON,
            confidence=0.8,
        )

        try:
            schema = json.loads(text)
            self._extract_json_schema_fields(schema, result)
        except json.JSONDecodeError:
            result.validation_errors.append("Invalid JSON schema")

        return result

    def _extract_json_schema_fields(
        self,
        schema: Dict,
        result: ProtocolAnalysisResult,
        path: str = "$",
    ) -> None:
        """Extract fields from JSON Schema."""
        if "properties" in schema:
            for name, prop in schema["properties"].items():
                result.fields.append(
                    ProtocolField(
                        name=name,
                        data_type=prop.get("type", "any"),
                        path=f"{path}.{name}",
                        required=name in schema.get("required", []),
                    )
                )
                result.total_field_count += 1

                # Recurse for nested objects
                if prop.get("type") == "object":
                    self._extract_json_schema_fields(prop, result, f"{path}.{name}")

    def _analyze_copybook(self, text: str) -> ProtocolAnalysisResult:
        """Analyze COBOL copybook."""
        result = ProtocolAnalysisResult(
            protocol_type=ProtocolType.COBOL_COPYBOOK,
            confidence=0.9,
        )

        # Parse PIC clauses
        pic_pattern = re.compile(r"(\d{2})\s+(\S+)\s+PIC\s+([X9\(\)V\-]+)", re.IGNORECASE)

        for match in pic_pattern.finditer(text):
            level = match.group(1)
            name = match.group(2)
            pic = match.group(3)

            result.fields.append(
                ProtocolField(
                    name=name,
                    data_type=self._pic_to_type(pic),
                    pattern=pic,
                )
            )

        result.total_field_count = len(result.fields)
        return result

    def _pic_to_type(self, pic: str) -> str:
        """Convert COBOL PIC clause to data type."""
        pic = pic.upper()
        if "X" in pic:
            return "string"
        elif "9" in pic and "V" in pic:
            return "decimal"
        elif "9" in pic:
            return "integer"
        else:
            return "binary"
