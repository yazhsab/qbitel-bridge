"""
FedWire Message Data Structures

Core data structures for FedWire Funds Transfer messages.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_codes import (
    TypeCode,
    TypeSubCode,
    BusinessFunctionCode,
    IDCode,
)


@dataclass
class Address:
    """Physical address for party information."""

    line1: str = ""
    line2: str = ""
    line3: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = "US"

    def to_lines(self) -> List[str]:
        """Convert address to list of lines."""
        lines = []
        if self.line1:
            lines.append(self.line1)
        if self.line2:
            lines.append(self.line2)
        if self.line3:
            lines.append(self.line3)

        city_state_zip = ""
        if self.city:
            city_state_zip = self.city
        if self.state:
            city_state_zip += f", {self.state}" if city_state_zip else self.state
        if self.postal_code:
            city_state_zip += f" {self.postal_code}" if city_state_zip else self.postal_code
        if city_state_zip:
            lines.append(city_state_zip)

        if self.country and self.country != "US":
            lines.append(self.country)

        return lines


@dataclass
class SenderInfo:
    """Sender Depository Institution information."""

    # ABA Routing Number (9 digits)
    routing_number: str = ""

    # Short name (up to 18 characters)
    short_name: str = ""

    # Address lines
    address: Address = field(default_factory=Address)

    def validate(self) -> List[str]:
        """Validate sender information."""
        errors = []
        if not self.routing_number:
            errors.append("Sender routing number is required")
        elif len(self.routing_number) != 9 or not self.routing_number.isdigit():
            errors.append("Sender routing number must be 9 digits")
        if not self.short_name:
            errors.append("Sender short name is required")
        return errors


@dataclass
class ReceiverInfo:
    """Receiver Depository Institution information."""

    # ABA Routing Number (9 digits)
    routing_number: str = ""

    # Short name (up to 18 characters)
    short_name: str = ""

    # Address lines
    address: Address = field(default_factory=Address)

    def validate(self) -> List[str]:
        """Validate receiver information."""
        errors = []
        if not self.routing_number:
            errors.append("Receiver routing number is required")
        elif len(self.routing_number) != 9 or not self.routing_number.isdigit():
            errors.append("Receiver routing number must be 9 digits")
        if not self.short_name:
            errors.append("Receiver short name is required")
        return errors


@dataclass
class FIInfo:
    """Financial Institution information (for intermediary/beneficiary FI)."""

    # ID type and number
    id_code: Optional[IDCode] = None
    identifier: str = ""

    # Name and address
    name: str = ""
    address: Address = field(default_factory=Address)

    def to_tag_value(self) -> str:
        """Convert to FedWire tag value format."""
        lines = []

        # First line: ID code and identifier
        if self.id_code and self.identifier:
            lines.append(f"{self.id_code.code}{self.identifier}")
        elif self.identifier:
            lines.append(self.identifier)

        # Name
        if self.name:
            lines.append(self.name)

        # Address
        lines.extend(self.address.to_lines())

        return "*".join(lines[:4])  # Max 4 lines


@dataclass
class BeneficiaryInfo:
    """Beneficiary information (tag 4200)."""

    # ID type and number
    id_code: Optional[IDCode] = None
    identifier: str = ""

    # Name
    name: str = ""

    # Address
    address: Address = field(default_factory=Address)

    def to_tag_value(self) -> str:
        """Convert to FedWire tag value format."""
        lines = []

        # First line: ID code and identifier
        if self.id_code and self.identifier:
            lines.append(f"{self.id_code.code}{self.identifier}")
        elif self.identifier:
            lines.append(self.identifier)

        # Name
        if self.name:
            lines.append(self.name)

        # Address
        lines.extend(self.address.to_lines())

        return "*".join(lines[:4])  # Max 4 lines


@dataclass
class OriginatorInfo:
    """Originator information (tag 5000)."""

    # ID type and number
    id_code: Optional[IDCode] = None
    identifier: str = ""

    # Name
    name: str = ""

    # Address
    address: Address = field(default_factory=Address)

    def to_tag_value(self) -> str:
        """Convert to FedWire tag value format."""
        lines = []

        # First line: ID code and identifier
        if self.id_code and self.identifier:
            lines.append(f"{self.id_code.code}{self.identifier}")
        elif self.identifier:
            lines.append(self.identifier)

        # Name
        if self.name:
            lines.append(self.name)

        # Address
        lines.extend(self.address.to_lines())

        return "*".join(lines[:4])  # Max 4 lines


@dataclass
class IntermediaryInfo:
    """Intermediary Financial Institution information (tag 4100)."""

    # ID type and number
    id_code: Optional[IDCode] = None
    identifier: str = ""

    # Name
    name: str = ""

    # Address
    address: Address = field(default_factory=Address)

    def to_tag_value(self) -> str:
        """Convert to FedWire tag value format."""
        lines = []

        # First line: ID code and identifier
        if self.id_code and self.identifier:
            lines.append(f"{self.id_code.code}{self.identifier}")
        elif self.identifier:
            lines.append(self.identifier)

        # Name
        if self.name:
            lines.append(self.name)

        # Address
        lines.extend(self.address.to_lines())

        return "*".join(lines[:4])  # Max 4 lines


@dataclass
class Charges:
    """Charges information (tag 3700)."""

    charge_details: str = ""  # "B" = Beneficiary, "S" = Shared
    send_amount: Optional[Decimal] = None
    currency: str = "USD"


@dataclass
class ExchangeRate:
    """Exchange rate information (tag 3720)."""

    rate: Decimal = Decimal("0")


@dataclass
class RemittanceInfo:
    """Remittance information structure."""

    # Related remittance (8200)
    remittance_location_method: str = ""
    remittance_location_edi: str = ""
    remittance_location_uri: str = ""

    # Originator/Beneficiary remittance (8300/8350)
    originator_name: str = ""
    originator_id_code: Optional[IDCode] = None
    originator_id: str = ""
    originator_address: Address = field(default_factory=Address)

    beneficiary_name: str = ""
    beneficiary_id_code: Optional[IDCode] = None
    beneficiary_id: str = ""
    beneficiary_address: Address = field(default_factory=Address)

    # Document info (8400)
    document_id_code: str = ""
    proprietary_code: str = ""
    document_id: str = ""

    # Amounts (8450, 8500, 8550)
    amount_paid: Optional[Decimal] = None
    amount_paid_currency: str = "USD"
    gross_amount: Optional[Decimal] = None
    gross_amount_currency: str = "USD"
    discount_amount: Optional[Decimal] = None
    discount_currency: str = "USD"

    # Adjustment (8600)
    adjustment_reason: str = ""
    credit_debit: str = ""
    adjustment_amount: Optional[Decimal] = None
    adjustment_currency: str = "USD"
    adjustment_info: str = ""

    # Date (8650)
    document_date: Optional[date] = None

    # Secondary doc (8700)
    secondary_doc_id_code: str = ""
    secondary_doc_proprietary: str = ""
    secondary_doc_id: str = ""
    secondary_doc_issuer: str = ""

    # Free text (8750)
    free_text_lines: List[str] = field(default_factory=list)


@dataclass
class OriginatorToBeneficiaryInfo:
    """Originator to Beneficiary Information (tag 6000)."""

    line1: str = ""
    line2: str = ""
    line3: str = ""
    line4: str = ""

    def to_tag_value(self) -> str:
        """Convert to FedWire tag value format."""
        lines = []
        if self.line1:
            lines.append(self.line1)
        if self.line2:
            lines.append(self.line2)
        if self.line3:
            lines.append(self.line3)
        if self.line4:
            lines.append(self.line4)
        return "*".join(lines)


@dataclass
class FIToFIInfo:
    """FI to FI Information (tag 6100)."""

    line1: str = ""
    line2: str = ""
    line3: str = ""
    line4: str = ""
    line5: str = ""
    line6: str = ""

    def to_tag_value(self) -> str:
        """Convert to FedWire tag value format."""
        lines = []
        if self.line1:
            lines.append(self.line1)
        if self.line2:
            lines.append(self.line2)
        if self.line3:
            lines.append(self.line3)
        if self.line4:
            lines.append(self.line4)
        if self.line5:
            lines.append(self.line5)
        if self.line6:
            lines.append(self.line6)
        return "*".join(lines)


@dataclass
class IMAD:
    """
    Input Message Accountability Data.

    Format: YYYYMMDDBBBBBBBBCC######
    - YYYYMMDD: Date
    - BBBBBBBB: Sender FedWire ID (8 chars)
    - CC: Cycle code
    - ######: Sequence number (6 digits)
    """

    input_date: date = field(default_factory=date.today)
    source_id: str = ""  # 8 character FedWire ID
    cycle_code: str = ""  # 2 characters
    sequence_number: str = ""  # 6 digits

    def to_string(self) -> str:
        """Convert IMAD to string format."""
        date_str = self.input_date.strftime("%Y%m%d")
        return f"{date_str}{self.source_id:8}{self.cycle_code:2}{self.sequence_number:>06}"

    @classmethod
    def from_string(cls, imad_str: str) -> "IMAD":
        """Parse IMAD from string."""
        if len(imad_str) != 22:
            raise ValueError(f"IMAD must be 22 characters, got {len(imad_str)}")

        input_date = datetime.strptime(imad_str[:8], "%Y%m%d").date()
        source_id = imad_str[8:16]
        cycle_code = imad_str[16:18]
        sequence_number = imad_str[18:22]

        return cls(
            input_date=input_date,
            source_id=source_id,
            cycle_code=cycle_code,
            sequence_number=sequence_number,
        )


@dataclass
class SenderSuppliedInfo:
    """
    Sender Supplied Information (tag 1500).

    Format information provided by the sender.
    """

    # Format version (2 characters, usually "30")
    format_version: str = "30"

    # User Request Correlation (4 characters)
    user_request_correlation: str = ""

    # Test/Production indicator (1 character: T or P)
    test_production: str = "P"

    # Message duplication code (1 character)
    message_dup_code: str = " "

    def to_string(self) -> str:
        """Convert to tag value string."""
        return (
            f"{self.format_version:2}"
            f"{self.user_request_correlation:4}"
            f"{self.test_production:1}"
            f"{self.message_dup_code:1}"
        )


@dataclass
class FedWireMessage:
    """
    Complete FedWire Funds Transfer Message.

    Represents all components of a FedWire message including mandatory
    and optional tags.
    """

    # ========== Mandatory Fields ==========

    # Sender Supplied Information (1500)
    sender_supplied: SenderSuppliedInfo = field(default_factory=SenderSuppliedInfo)

    # Type/Subtype (1510)
    type_code: TypeCode = TypeCode.BASIC_FUNDS_TRANSFER
    type_subcode: TypeSubCode = TypeSubCode.BASIC_TRANSFER

    # Input Message Accountability Data (1520)
    imad: IMAD = field(default_factory=IMAD)

    # Amount (2000) - in cents
    amount: Decimal = Decimal("0")
    currency: str = "USD"

    # Sender DI (3100)
    sender: SenderInfo = field(default_factory=SenderInfo)

    # Receiver DI (3400)
    receiver: ReceiverInfo = field(default_factory=ReceiverInfo)

    # Business Function Code (3600)
    business_function_code: BusinessFunctionCode = BusinessFunctionCode.BANK_TRANSFER

    # ========== Optional Fields ==========

    # Sender Reference (3320)
    sender_reference: str = ""

    # Previous Message Identifier (3500)
    previous_message_id: str = ""

    # Intermediary FI (4100)
    intermediary_fi: Optional[IntermediaryInfo] = None

    # Beneficiary FI (4000)
    beneficiary_fi: Optional[FIInfo] = None

    # Beneficiary (4200)
    beneficiary: Optional[BeneficiaryInfo] = None

    # Originator (5000)
    originator: Optional[OriginatorInfo] = None

    # Originator FI (5100)
    originator_fi: Optional[FIInfo] = None

    # Instructing FI (5200)
    instructing_fi: Optional[FIInfo] = None

    # Account debited/credited in drawdown (5400/5500)
    account_debited: str = ""
    account_credited: str = ""

    # Charges (3700)
    charges: Optional[Charges] = None

    # Instructed Amount (3710)
    instructed_amount: Optional[Decimal] = None
    instructed_currency: str = "USD"

    # Exchange Rate (3720)
    exchange_rate: Optional[Decimal] = None

    # Originator to Beneficiary Info (6000)
    originator_to_beneficiary: Optional[OriginatorToBeneficiaryInfo] = None

    # FI to FI Info (6100)
    fi_to_fi_info: Optional[FIToFIInfo] = None

    # Drawdown Debit Account Advice (6200)
    drawdown_advice_code: str = ""
    drawdown_advice_info: List[str] = field(default_factory=list)

    # Unstructured Addenda (7500)
    unstructured_addenda: str = ""

    # Remittance Information (8200-8750)
    remittance: Optional[RemittanceInfo] = None

    # Service Message (9000)
    service_message: str = ""

    # ========== Response/Output Fields ==========

    # Output Message Accountability Data (1120)
    omad: str = ""

    # Receipt timestamp (1110)
    receipt_timestamp: Optional[datetime] = None

    # Error information (1130)
    error_info: str = ""

    # Message disposition (1100)
    message_disposition: str = ""

    # ========== Metadata ==========

    # Raw message for reference
    raw_message: str = ""

    # Processing timestamps
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None

    def validate(self) -> List[str]:
        """
        Validate the FedWire message.

        Returns list of validation errors (empty if valid).
        """
        errors = []

        # Validate mandatory fields
        errors.extend(self.sender.validate())
        errors.extend(self.receiver.validate())

        # Validate amount
        if self.amount <= 0:
            errors.append("Amount must be greater than zero")
        if self.amount > Decimal("99999999999.99"):
            errors.append("Amount exceeds maximum allowed ($99,999,999,999.99)")

        # Validate IMAD
        if not self.imad.source_id:
            errors.append("IMAD source ID is required")

        # Validate business function code requirements
        bfc = self.business_function_code
        if bfc in [BusinessFunctionCode.CUSTOMER_TRANSFER, BusinessFunctionCode.CUSTOMER_TRANSFER_PLUS]:
            if not self.beneficiary:
                errors.append(f"{bfc.short_name} requires beneficiary information")

        if bfc == BusinessFunctionCode.CUSTOMER_TRANSFER_PLUS:
            if not self.originator:
                errors.append("CTP requires originator information")

        if bfc in [BusinessFunctionCode.DRAWDOWN_REQUEST, BusinessFunctionCode.DRAWDOWN_TRANSFER]:
            if not self.originator:
                errors.append(f"{bfc.short_name} requires originator information")
            if not self.originator_fi:
                errors.append(f"{bfc.short_name} requires originator FI information")
            if not self.beneficiary:
                errors.append(f"{bfc.short_name} requires beneficiary information")

        # Cover payment requirements
        if bfc == BusinessFunctionCode.COVER_PAYMENT:
            if not self.beneficiary:
                errors.append("Cover payment requires beneficiary")
            if not self.originator:
                errors.append("Cover payment requires originator")
            if not self.originator_fi:
                errors.append("Cover payment requires originator FI")

        return errors

    def get_amount_dollars(self) -> Decimal:
        """Get amount in dollars (from cents)."""
        return self.amount

    def set_amount_dollars(self, dollars: Decimal) -> None:
        """Set amount from dollars."""
        self.amount = dollars

    def to_dict(self) -> Dict:
        """Convert message to dictionary representation."""
        return {
            "type_code": self.type_code.code,
            "type_subcode": self.type_subcode.code,
            "business_function_code": self.business_function_code.code,
            "amount": str(self.amount),
            "currency": self.currency,
            "sender": {
                "routing_number": self.sender.routing_number,
                "short_name": self.sender.short_name,
            },
            "receiver": {
                "routing_number": self.receiver.routing_number,
                "short_name": self.receiver.short_name,
            },
            "imad": self.imad.to_string(),
            "sender_reference": self.sender_reference,
            "beneficiary": (
                {
                    "name": self.beneficiary.name,
                    "identifier": self.beneficiary.identifier,
                }
                if self.beneficiary
                else None
            ),
            "originator": (
                {
                    "name": self.originator.name,
                    "identifier": self.originator.identifier,
                }
                if self.originator
                else None
            ),
            "created_at": self.created_at.isoformat(),
        }
