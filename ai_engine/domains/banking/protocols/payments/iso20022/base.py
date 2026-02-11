"""
ISO 20022 Base Classes

Core classes for ISO 20022 message parsing, building, and validation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union
from xml.etree import ElementTree as ET
import logging
import re
import hashlib

logger = logging.getLogger(__name__)


class ISO20022MessageType(Enum):
    """ISO 20022 message type categories."""

    # Payment Initiation (pain.*)
    PAIN_001 = ("pain.001", "CustomerCreditTransferInitiation", "Customer Credit Transfer Initiation")
    PAIN_002 = ("pain.002", "CustomerPaymentStatusReport", "Customer Payment Status Report")
    PAIN_007 = ("pain.007", "CustomerPaymentReversal", "Customer Payment Reversal")
    PAIN_008 = ("pain.008", "CustomerDirectDebitInitiation", "Customer Direct Debit Initiation")

    # Payments Clearing and Settlement (pacs.*)
    PACS_002 = ("pacs.002", "FIToFIPaymentStatusReport", "FI to FI Payment Status Report")
    PACS_003 = ("pacs.003", "FIToFICustomerDirectDebit", "FI to FI Customer Direct Debit")
    PACS_004 = ("pacs.004", "PaymentReturn", "Payment Return")
    PACS_008 = ("pacs.008", "FIToFICustomerCreditTransfer", "FI to FI Customer Credit Transfer")
    PACS_009 = ("pacs.009", "FinancialInstitutionCreditTransfer", "FI Credit Transfer")
    PACS_028 = ("pacs.028", "FIToFIPaymentStatusRequest", "FI to FI Payment Status Request")

    # Cash Management (camt.*)
    CAMT_052 = ("camt.052", "BankToCustomerAccountReport", "Bank to Customer Account Report")
    CAMT_053 = ("camt.053", "BankToCustomerStatement", "Bank to Customer Statement")
    CAMT_054 = ("camt.054", "BankToCustomerDebitCreditNotification", "Bank to Customer Debit/Credit Notification")
    CAMT_056 = ("camt.056", "FIToFIPaymentCancellationRequest", "FI to FI Payment Cancellation Request")
    CAMT_029 = ("camt.029", "ResolutionOfInvestigation", "Resolution of Investigation")

    def __init__(self, code: str, root_element: str, description: str):
        self.code = code
        self.root_element = root_element
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["ISO20022MessageType"]:
        """Get message type from code."""
        for msg_type in cls:
            if msg_type.code == code:
                return msg_type
        return None


class TransactionStatus(Enum):
    """Transaction status codes."""

    ACCEPTED = ("ACCP", "Accepted")
    ACCEPTED_SETTLEMENT_COMPLETED = ("ACSC", "Accepted Settlement Completed")
    ACCEPTED_SETTLEMENT_IN_PROGRESS = ("ACSP", "Accepted Settlement in Progress")
    ACCEPTED_TECHNICAL_VALIDATION = ("ACTC", "Accepted Technical Validation")
    ACCEPTED_WITH_CHANGE = ("ACWC", "Accepted with Change")
    PENDING = ("PDNG", "Pending")
    RECEIVED = ("RCVD", "Received")
    REJECTED = ("RJCT", "Rejected")

    def __init__(self, code: str, description: str):
        self.status_code = code
        self.description = description


@dataclass
class Amount:
    """Monetary amount with currency."""

    value: Decimal
    currency: str  # ISO 4217 currency code

    def __post_init__(self):
        if isinstance(self.value, (int, float)):
            self.value = Decimal(str(self.value))

    def to_xml_value(self) -> str:
        """Format for XML element value."""
        return f"{self.value:.2f}"

    @classmethod
    def from_xml(cls, element: ET.Element) -> "Amount":
        """Parse from XML element."""
        value = Decimal(element.text or "0")
        currency = element.get("Ccy", "USD")
        return cls(value=value, currency=currency)


@dataclass
class AccountIdentification:
    """Account identification."""

    iban: Optional[str] = None
    other_id: Optional[str] = None
    other_scheme: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        if self.iban:
            return {"IBAN": self.iban}
        return {"Othr": {"Id": self.other_id, "SchmeNm": self.other_scheme}}

    @classmethod
    def from_xml(cls, element: ET.Element, ns: Dict[str, str]) -> "AccountIdentification":
        """Parse from XML element."""
        iban_elem = element.find(".//IBAN", ns) or element.find("IBAN")
        if iban_elem is not None and iban_elem.text:
            return cls(iban=iban_elem.text)

        other_elem = element.find(".//Othr/Id", ns) or element.find("Othr/Id")
        if other_elem is not None and other_elem.text:
            return cls(other_id=other_elem.text)

        return cls()


@dataclass
class FinancialInstitution:
    """Financial institution identification."""

    bic: Optional[str] = None  # BIC/SWIFT code
    name: Optional[str] = None
    clearing_system_id: Optional[str] = None
    member_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.bic:
            result["BICFI"] = self.bic
        if self.name:
            result["Nm"] = self.name
        if self.clearing_system_id:
            result["ClrSysMmbId"] = {
                "ClrSysId": {"Cd": self.clearing_system_id},
                "MmbId": self.member_id,
            }
        return result

    @classmethod
    def from_xml(cls, element: ET.Element, ns: Dict[str, str]) -> "FinancialInstitution":
        """Parse from XML element."""
        bic_elem = element.find(".//BICFI", ns) or element.find("BICFI")
        name_elem = element.find(".//Nm", ns) or element.find("Nm")

        return cls(
            bic=bic_elem.text if bic_elem is not None else None,
            name=name_elem.text if name_elem is not None else None,
        )


@dataclass
class PartyIdentification:
    """Party (person or organization) identification."""

    name: Optional[str] = None
    postal_address: Optional[Dict[str, str]] = None
    organization_id: Optional[str] = None
    private_id: Optional[str] = None
    country: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.name:
            result["Nm"] = self.name
        if self.postal_address:
            result["PstlAdr"] = self.postal_address
        if self.organization_id:
            result["Id"] = {"OrgId": {"Othr": {"Id": self.organization_id}}}
        elif self.private_id:
            result["Id"] = {"PrvtId": {"Othr": {"Id": self.private_id}}}
        return result

    @classmethod
    def from_xml(cls, element: ET.Element, ns: Dict[str, str]) -> "PartyIdentification":
        """Parse from XML element."""
        name_elem = element.find(".//Nm", ns) or element.find("Nm")
        country_elem = element.find(".//Ctry", ns) or element.find("Ctry")

        return cls(
            name=name_elem.text if name_elem is not None else None,
            country=country_elem.text if country_elem is not None else None,
        )


@dataclass
class ISO20022Message(ABC):
    """Base class for ISO 20022 messages."""

    message_id: str = ""
    creation_datetime: datetime = field(default_factory=datetime.utcnow)
    message_type: Optional[ISO20022MessageType] = None

    # Namespace handling
    namespace: str = "urn:iso:std:iso:20022:tech:xsd"
    namespace_version: str = ""

    # Raw XML
    raw_xml: Optional[str] = None

    @property
    def full_namespace(self) -> str:
        """Get full namespace URI."""
        if self.message_type:
            return f"{self.namespace}:{self.message_type.code}.001.{self.namespace_version or '03'}"
        return self.namespace

    @abstractmethod
    def to_xml(self) -> str:
        """Serialize message to XML string."""
        pass

    @abstractmethod
    def validate(self) -> List[str]:
        """Validate message, return list of errors."""
        pass

    def get_hash(self) -> str:
        """Get hash of message content for integrity."""
        content = self.to_xml()
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "message_id": self.message_id,
            "creation_datetime": self.creation_datetime.isoformat(),
            "message_type": self.message_type.code if self.message_type else None,
        }


class ISO20022Parser(ABC):
    """Base parser for ISO 20022 messages."""

    # Common namespaces
    NAMESPACES = {
        "pain001": "urn:iso:std:iso:20022:tech:xsd:pain.001.001.03",
        "pain002": "urn:iso:std:iso:20022:tech:xsd:pain.002.001.03",
        "pacs008": "urn:iso:std:iso:20022:tech:xsd:pacs.008.001.02",
        "camt053": "urn:iso:std:iso:20022:tech:xsd:camt.053.001.02",
    }

    def __init__(self):
        self.errors: List[str] = []

    @abstractmethod
    def parse(self, xml_content: str) -> ISO20022Message:
        """Parse XML content to message object."""
        pass

    def _parse_xml(self, xml_content: str) -> ET.Element:
        """Parse XML string to element tree."""
        try:
            # Remove BOM if present
            if xml_content.startswith('\ufeff'):
                xml_content = xml_content[1:]

            return ET.fromstring(xml_content)
        except ET.ParseError as e:
            self.errors.append(f"XML parsing error: {e}")
            raise

    def _get_text(self, element: Optional[ET.Element], default: str = "") -> str:
        """Safely get element text."""
        if element is not None and element.text:
            return element.text.strip()
        return default

    def _get_datetime(self, element: Optional[ET.Element]) -> Optional[datetime]:
        """Parse datetime from element."""
        text = self._get_text(element)
        if not text:
            return None

        # Try common ISO 20022 datetime formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue

        self.errors.append(f"Unable to parse datetime: {text}")
        return None

    def _get_amount(self, element: Optional[ET.Element]) -> Optional[Amount]:
        """Parse amount from element."""
        if element is None:
            return None
        return Amount.from_xml(element)

    def _find_with_namespace(
        self,
        element: ET.Element,
        path: str,
        namespace: Optional[str] = None,
    ) -> Optional[ET.Element]:
        """Find element with or without namespace."""
        # Try with namespace
        if namespace:
            ns = {"ns": namespace}
            result = element.find(f".//ns:{path}", ns)
            if result is not None:
                return result

        # Try without namespace (local name match)
        for elem in element.iter():
            local_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if local_name == path:
                return elem

        return None


class ISO20022Builder(ABC):
    """Base builder for ISO 20022 messages."""

    def __init__(self, message_type: ISO20022MessageType, version: str = "03"):
        self.message_type = message_type
        self.version = version
        self.namespace = f"urn:iso:std:iso:20022:tech:xsd:{message_type.code}.001.{version}"

    @abstractmethod
    def build(self, **kwargs) -> str:
        """Build XML message from parameters."""
        pass

    def _create_root(self) -> ET.Element:
        """Create root document element."""
        root = ET.Element("Document")
        root.set("xmlns", self.namespace)
        return root

    def _add_element(
        self,
        parent: ET.Element,
        tag: str,
        text: Optional[str] = None,
        attribs: Optional[Dict[str, str]] = None,
    ) -> ET.Element:
        """Add child element."""
        elem = ET.SubElement(parent, tag)
        if text:
            elem.text = text
        if attribs:
            for key, value in attribs.items():
                elem.set(key, value)
        return elem

    def _add_amount(
        self,
        parent: ET.Element,
        tag: str,
        amount: Amount,
    ) -> ET.Element:
        """Add amount element with currency attribute."""
        elem = self._add_element(parent, tag, amount.to_xml_value())
        elem.set("Ccy", amount.currency)
        return elem

    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime for ISO 20022."""
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def _to_string(self, root: ET.Element) -> str:
        """Convert element tree to XML string."""
        return ET.tostring(root, encoding="unicode", xml_declaration=True)


class ISO20022Validator:
    """Validator for ISO 20022 messages."""

    # IBAN regex pattern
    IBAN_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$")

    # BIC regex pattern (8 or 11 characters)
    BIC_PATTERN = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$")

    # Currency code pattern (ISO 4217)
    CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_message(self, message: ISO20022Message) -> List[str]:
        """Validate ISO 20022 message."""
        self.errors = []
        self.warnings = []

        # Validate message ID
        if not message.message_id:
            self.errors.append("Message ID is required")
        elif len(message.message_id) > 35:
            self.errors.append("Message ID must not exceed 35 characters")

        # Validate creation datetime
        if not message.creation_datetime:
            self.errors.append("Creation datetime is required")
        elif message.creation_datetime > datetime.utcnow():
            self.warnings.append("Creation datetime is in the future")

        # Add message-specific validation
        self.errors.extend(message.validate())

        return self.errors

    def validate_iban(self, iban: str) -> bool:
        """Validate IBAN format and checksum."""
        if not iban:
            return False

        # Remove spaces
        iban = iban.replace(" ", "").upper()

        # Check format
        if not self.IBAN_PATTERN.match(iban):
            self.errors.append(f"Invalid IBAN format: {iban}")
            return False

        # Validate checksum (mod 97)
        rearranged = iban[4:] + iban[:4]
        numeric = ""
        for char in rearranged:
            if char.isdigit():
                numeric += char
            else:
                numeric += str(ord(char) - ord("A") + 10)

        if int(numeric) % 97 != 1:
            self.errors.append(f"Invalid IBAN checksum: {iban}")
            return False

        return True

    def validate_bic(self, bic: str) -> bool:
        """Validate BIC/SWIFT code format."""
        if not bic:
            return False

        bic = bic.upper()
        if not self.BIC_PATTERN.match(bic):
            self.errors.append(f"Invalid BIC format: {bic}")
            return False

        return True

    def validate_currency(self, currency: str) -> bool:
        """Validate ISO 4217 currency code."""
        if not currency:
            return False

        if not self.CURRENCY_PATTERN.match(currency.upper()):
            self.errors.append(f"Invalid currency code: {currency}")
            return False

        return True

    def validate_amount(self, amount: Amount) -> bool:
        """Validate amount."""
        if amount.value <= 0:
            self.errors.append("Amount must be positive")
            return False

        if not self.validate_currency(amount.currency):
            return False

        # Check decimal places (most currencies use 2)
        if abs(amount.value.as_tuple().exponent) > 2:
            self.warnings.append(f"Amount has more than 2 decimal places: {amount.value}")

        return True
