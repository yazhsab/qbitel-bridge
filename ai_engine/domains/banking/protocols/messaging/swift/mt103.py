"""
MT103 - Single Customer Credit Transfer

The MT103 is the most commonly used SWIFT message for customer credit
transfers between financial institutions.

Supports:
- MT103: Standard single customer credit transfer
- MT103 STP: Straight Through Processing variant
- MT103 REMIT: Remittance information variant
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
import uuid
import re

from ai_engine.domains.banking.protocols.messaging.swift.swift_message import (
    SwiftMessage,
    SwiftBasicHeader,
    SwiftApplicationHeader,
    SwiftUserHeader,
    SwiftTextBlock,
    SwiftTrailerBlock,
    SwiftField,
    MessageDirection,
)
from ai_engine.domains.banking.protocols.messaging.swift.swift_codes import (
    BANK_OPERATION_CODES,
    CHARGE_CODES,
    INSTRUCTION_CODES,
    MT103_FIELDS,
)


@dataclass
class MT103Party:
    """Party information for MT103."""

    account: Optional[str] = None
    bic: Optional[str] = None
    name: str = ""
    address_line1: str = ""
    address_line2: str = ""
    address_line3: str = ""
    country: str = ""

    # Party identification (for 50F/59F format)
    party_identifier: Optional[str] = None
    name_address_type: Optional[str] = None  # Type of name/address

    def to_50k_format(self) -> str:
        """Format for field 50K (Ordering Customer)."""
        lines = []
        if self.account:
            lines.append(f"/{self.account}")
        if self.name:
            lines.append(self.name[:35])
        if self.address_line1:
            lines.append(self.address_line1[:35])
        if self.address_line2:
            lines.append(self.address_line2[:35])
        if self.address_line3:
            lines.append(self.address_line3[:35])
        return "\n".join(lines[:4])

    def to_59_format(self) -> str:
        """Format for field 59 (Beneficiary Customer)."""
        lines = []
        if self.account:
            lines.append(f"/{self.account}")
        if self.name:
            lines.append(self.name[:35])
        if self.address_line1:
            lines.append(self.address_line1[:35])
        if self.address_line2:
            lines.append(self.address_line2[:35])
        if self.address_line3:
            lines.append(self.address_line3[:35])
        return "\n".join(lines[:4])

    def to_bic_format(self) -> str:
        """Format with BIC only."""
        lines = []
        if self.account:
            lines.append(f"/{self.account}")
        if self.bic:
            lines.append(self.bic)
        return "\n".join(lines)


@dataclass
class MT103Agent:
    """Agent/Institution information for MT103."""

    bic: Optional[str] = None
    account: Optional[str] = None
    name: str = ""
    address_line1: str = ""
    address_line2: str = ""
    location: str = ""
    party_type: str = ""  # D=BIC, C=Account, etc.

    def to_a_format(self) -> str:
        """Format for option A (BIC)."""
        lines = []
        if self.party_type:
            lines.append(f"/{self.party_type}")
        if self.account:
            lines.append(f"/{self.account}")
        if self.bic:
            lines.append(self.bic)
        return "\n".join(lines) if lines else self.bic or ""

    def to_d_format(self) -> str:
        """Format for option D (Name and Address)."""
        lines = []
        if self.party_type:
            lines.append(f"/{self.party_type}")
        if self.account:
            lines.append(f"/{self.account}")
        if self.name:
            lines.append(self.name[:35])
        if self.address_line1:
            lines.append(self.address_line1[:35])
        if self.address_line2:
            lines.append(self.address_line2[:35])
        if self.location:
            lines.append(self.location[:35])
        return "\n".join(lines[:4])


@dataclass
class MT103Charges:
    """Charge information for MT103."""

    details: str = "SHA"  # BEN, OUR, SHA
    sender_charges: List[tuple] = field(default_factory=list)  # [(currency, amount)]
    receiver_charges: Optional[tuple] = None  # (currency, amount)

    @property
    def details_description(self) -> str:
        """Get description of charge details."""
        return CHARGE_CODES.get(self.details, self.details)


class MT103Field:
    """Field helpers for MT103."""

    @staticmethod
    def format_32a(value_date: date, currency: str, amount: Decimal) -> str:
        """Format field 32A (Value Date/Currency/Amount)."""
        date_str = value_date.strftime("%y%m%d")
        # Format amount with comma as decimal separator
        amount_str = f"{amount:,.2f}".replace(",", "").replace(".", ",")
        return f"{date_str}{currency}{amount_str}"

    @staticmethod
    def format_33b(currency: str, amount: Decimal) -> str:
        """Format field 33B (Currency/Instructed Amount)."""
        amount_str = f"{amount:,.2f}".replace(",", "").replace(".", ",")
        return f"{currency}{amount_str}"

    @staticmethod
    def parse_32a(value: str) -> tuple:
        """Parse field 32A into (date, currency, amount)."""
        if len(value) < 12:
            raise ValueError(f"Invalid 32A format: {value}")

        date_str = value[:6]
        currency = value[6:9]
        amount_str = value[9:].replace(",", ".")

        year = int(date_str[:2])
        year = 2000 + year if year < 50 else 1900 + year
        month = int(date_str[2:4])
        day = int(date_str[4:6])

        return date(year, month, day), currency, Decimal(amount_str)


@dataclass
class MT103Message:
    """
    MT103 Single Customer Credit Transfer message.

    This represents a fully parsed or constructed MT103 message.
    """

    # Required fields
    sender_reference: str = ""  # Field 20
    bank_operation_code: str = "CRED"  # Field 23B
    value_date: Optional[date] = None  # Field 32A
    currency: str = "USD"
    amount: Decimal = Decimal("0")

    # Ordering customer (field 50)
    ordering_customer: MT103Party = field(default_factory=MT103Party)
    ordering_customer_option: str = "K"  # A, F, or K

    # Beneficiary customer (field 59)
    beneficiary: MT103Party = field(default_factory=MT103Party)
    beneficiary_option: str = ""  # A, F, or blank

    # Charges (field 71A/71F/71G)
    charges: MT103Charges = field(default_factory=MT103Charges)

    # Optional agents/institutions
    ordering_institution: Optional[MT103Agent] = None  # Field 52
    senders_correspondent: Optional[MT103Agent] = None  # Field 53
    receivers_correspondent: Optional[MT103Agent] = None  # Field 54
    third_reimbursement_institution: Optional[MT103Agent] = None  # Field 55
    intermediary_institution: Optional[MT103Agent] = None  # Field 56
    account_with_institution: Optional[MT103Agent] = None  # Field 57

    # Optional fields
    time_indication: Optional[str] = None  # Field 13C
    instruction_codes: List[str] = field(default_factory=list)  # Field 23E
    transaction_type_code: Optional[str] = None  # Field 26T
    instructed_amount: Optional[Decimal] = None  # Field 33B
    instructed_currency: Optional[str] = None
    exchange_rate: Optional[Decimal] = None  # Field 36
    remittance_info: List[str] = field(default_factory=list)  # Field 70
    sender_to_receiver_info: List[str] = field(default_factory=list)  # Field 72
    regulatory_reporting: List[str] = field(default_factory=list)  # Field 77B
    envelope_contents: Optional[str] = None  # Field 77T

    # Header information
    sender_bic: str = ""
    receiver_bic: str = ""
    priority: str = "N"  # N=Normal, U=Urgent
    uetr: Optional[str] = None  # Unique End-to-end Transaction Reference

    # Metadata
    is_stp: bool = False  # STP variant
    is_remit: bool = False  # REMIT variant
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.uetr:
            self.uetr = str(uuid.uuid4())

    @property
    def message_type(self) -> str:
        """Get the specific MT103 variant."""
        if self.is_stp:
            return "103_STP"
        elif self.is_remit:
            return "103_REMIT"
        return "103"

    @property
    def operation_description(self) -> str:
        """Get description of bank operation code."""
        return BANK_OPERATION_CODES.get(self.bank_operation_code, self.bank_operation_code)

    def to_swift_message(self) -> SwiftMessage:
        """Convert to SwiftMessage object."""
        msg = SwiftMessage()

        # Block 1: Basic Header
        msg.basic_header = SwiftBasicHeader(
            application_id="F",
            service_id="01",
            lt_address=f"{self.sender_bic}XXXX" if len(self.sender_bic) == 8 else self.sender_bic,
            session_number="0000",
            sequence_number="000000",
        )

        # Block 2: Application Header
        msg.application_header = SwiftApplicationHeader(
            direction=MessageDirection.INPUT,
            message_type=self.message_type.replace("_", ""),
            destination_bic=f"{self.receiver_bic}XXXX" if len(self.receiver_bic) == 8 else self.receiver_bic,
            priority=self.priority,
        )

        # Block 3: User Header
        user_header_fields = {}
        if self.is_stp:
            user_header_fields["119"] = "STP"
        if self.is_remit:
            user_header_fields["119"] = "REMIT"
        if self.uetr:
            user_header_fields["121"] = self.uetr

        if user_header_fields:
            msg.user_header = SwiftUserHeader(fields=user_header_fields)

        # Block 4: Text Block
        self._build_text_block(msg.text_block)

        return msg

    def _build_text_block(self, text_block: SwiftTextBlock) -> None:
        """Build the text block fields."""
        # Field 20: Sender's Reference
        text_block.add_field("20", self.sender_reference[:16])

        # Field 13C: Time Indication (optional)
        if self.time_indication:
            text_block.add_field("13C", self.time_indication)

        # Field 23B: Bank Operation Code
        text_block.add_field("23B", self.bank_operation_code)

        # Field 23E: Instruction Code(s) (optional, repeatable)
        for code in self.instruction_codes:
            text_block.add_field("23E", code)

        # Field 26T: Transaction Type Code (optional)
        if self.transaction_type_code:
            text_block.add_field("26T", self.transaction_type_code)

        # Field 32A: Value Date/Currency/Amount
        if self.value_date:
            text_block.add_field(
                "32A",
                MT103Field.format_32a(self.value_date, self.currency, self.amount),
            )

        # Field 33B: Currency/Instructed Amount (optional)
        if self.instructed_amount and self.instructed_currency:
            text_block.add_field(
                "33B",
                MT103Field.format_33b(self.instructed_currency, self.instructed_amount),
            )

        # Field 36: Exchange Rate (optional)
        if self.exchange_rate:
            text_block.add_field("36", str(self.exchange_rate).replace(".", ","))

        # Field 50: Ordering Customer
        if self.ordering_customer_option == "A" and self.ordering_customer.bic:
            text_block.add_field("50", self.ordering_customer.to_bic_format(), "A")
        elif self.ordering_customer_option == "F":
            text_block.add_field("50", self.ordering_customer.to_50k_format(), "F")
        else:
            text_block.add_field("50", self.ordering_customer.to_50k_format(), "K")

        # Field 52: Ordering Institution (optional)
        if self.ordering_institution:
            if self.ordering_institution.bic:
                text_block.add_field("52", self.ordering_institution.to_a_format(), "A")
            else:
                text_block.add_field("52", self.ordering_institution.to_d_format(), "D")

        # Field 53: Sender's Correspondent (optional)
        if self.senders_correspondent:
            if self.senders_correspondent.bic:
                text_block.add_field("53", self.senders_correspondent.to_a_format(), "A")
            elif self.senders_correspondent.location:
                text_block.add_field("53", self.senders_correspondent.location, "B")
            else:
                text_block.add_field("53", self.senders_correspondent.to_d_format(), "D")

        # Field 54: Receiver's Correspondent (optional)
        if self.receivers_correspondent:
            if self.receivers_correspondent.bic:
                text_block.add_field("54", self.receivers_correspondent.to_a_format(), "A")
            elif self.receivers_correspondent.location:
                text_block.add_field("54", self.receivers_correspondent.location, "B")
            else:
                text_block.add_field("54", self.receivers_correspondent.to_d_format(), "D")

        # Field 55: Third Reimbursement Institution (optional)
        if self.third_reimbursement_institution:
            if self.third_reimbursement_institution.bic:
                text_block.add_field("55", self.third_reimbursement_institution.to_a_format(), "A")
            elif self.third_reimbursement_institution.location:
                text_block.add_field("55", self.third_reimbursement_institution.location, "B")
            else:
                text_block.add_field("55", self.third_reimbursement_institution.to_d_format(), "D")

        # Field 56: Intermediary Institution (optional)
        if self.intermediary_institution:
            if self.intermediary_institution.bic:
                text_block.add_field("56", self.intermediary_institution.to_a_format(), "A")
            elif self.intermediary_institution.account:
                text_block.add_field("56", f"/{self.intermediary_institution.account}", "C")
            else:
                text_block.add_field("56", self.intermediary_institution.to_d_format(), "D")

        # Field 57: Account With Institution (optional)
        if self.account_with_institution:
            if self.account_with_institution.bic:
                text_block.add_field("57", self.account_with_institution.to_a_format(), "A")
            elif self.account_with_institution.location:
                text_block.add_field("57", self.account_with_institution.location, "B")
            elif self.account_with_institution.account:
                text_block.add_field("57", f"/{self.account_with_institution.account}", "C")
            else:
                text_block.add_field("57", self.account_with_institution.to_d_format(), "D")

        # Field 59: Beneficiary Customer
        if self.beneficiary_option == "A" and self.beneficiary.bic:
            text_block.add_field("59", self.beneficiary.to_bic_format(), "A")
        elif self.beneficiary_option == "F":
            text_block.add_field("59", self.beneficiary.to_59_format(), "F")
        else:
            text_block.add_field("59", self.beneficiary.to_59_format())

        # Field 70: Remittance Information (optional)
        if self.remittance_info:
            remittance_text = "\n".join(line[:35] for line in self.remittance_info[:4])
            text_block.add_field("70", remittance_text)

        # Field 71A: Details of Charges
        text_block.add_field("71A", self.charges.details)

        # Field 71F: Sender's Charges (optional, repeatable)
        for currency, amount in self.charges.sender_charges:
            amount_str = f"{amount:,.2f}".replace(",", "").replace(".", ",")
            text_block.add_field("71F", f"{currency}{amount_str}")

        # Field 71G: Receiver's Charges (optional)
        if self.charges.receiver_charges:
            currency, amount = self.charges.receiver_charges
            amount_str = f"{amount:,.2f}".replace(",", "").replace(".", ",")
            text_block.add_field("71G", f"{currency}{amount_str}")

        # Field 72: Sender to Receiver Information (optional)
        if self.sender_to_receiver_info:
            s2r_text = "\n".join(line[:35] for line in self.sender_to_receiver_info[:6])
            text_block.add_field("72", s2r_text)

        # Field 77B: Regulatory Reporting (optional)
        if self.regulatory_reporting:
            reg_text = "\n".join(line[:35] for line in self.regulatory_reporting[:3])
            text_block.add_field("77B", reg_text)

        # Field 77T: Envelope Contents (optional, for REMIT variant)
        if self.envelope_contents:
            text_block.add_field("77T", self.envelope_contents)

    def to_swift(self) -> str:
        """Convert to SWIFT message string."""
        return self.to_swift_message().to_swift()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "message_type": f"MT{self.message_type}",
            "sender_reference": self.sender_reference,
            "sender_bic": self.sender_bic,
            "receiver_bic": self.receiver_bic,
            "uetr": self.uetr,
            "bank_operation_code": self.bank_operation_code,
            "value_date": self.value_date.isoformat() if self.value_date else None,
            "currency": self.currency,
            "amount": str(self.amount),
            "instructed_currency": self.instructed_currency,
            "instructed_amount": str(self.instructed_amount) if self.instructed_amount else None,
            "exchange_rate": str(self.exchange_rate) if self.exchange_rate else None,
            "ordering_customer": {
                "account": self.ordering_customer.account,
                "bic": self.ordering_customer.bic,
                "name": self.ordering_customer.name,
                "address": [
                    self.ordering_customer.address_line1,
                    self.ordering_customer.address_line2,
                    self.ordering_customer.address_line3,
                ],
            },
            "beneficiary": {
                "account": self.beneficiary.account,
                "bic": self.beneficiary.bic,
                "name": self.beneficiary.name,
                "address": [
                    self.beneficiary.address_line1,
                    self.beneficiary.address_line2,
                    self.beneficiary.address_line3,
                ],
            },
            "charges": {
                "details": self.charges.details,
                "sender_charges": self.charges.sender_charges,
                "receiver_charges": self.charges.receiver_charges,
            },
            "remittance_info": self.remittance_info,
            "is_stp": self.is_stp,
            "is_remit": self.is_remit,
            "created_at": self.created_at.isoformat(),
        }


class MT103Parser:
    """Parser for MT103 messages."""

    @classmethod
    def from_swift_message(cls, msg: SwiftMessage) -> MT103Message:
        """Parse MT103 from SwiftMessage object."""
        mt103 = MT103Message()

        # Header info
        mt103.sender_bic = msg.sender_bic
        mt103.receiver_bic = msg.receiver_bic
        mt103.priority = msg.application_header.priority

        # Check variant
        if msg.user_header:
            service_id = msg.user_header.fields.get("119", "")
            mt103.is_stp = service_id == "STP"
            mt103.is_remit = service_id == "REMIT"
            mt103.uetr = msg.user_header.uetr

        # Parse text block fields
        for field in msg.text_block.fields:
            cls._parse_field(mt103, field)

        return mt103

    @classmethod
    def _parse_field(cls, mt103: MT103Message, field: SwiftField) -> None:
        """Parse a single field into MT103 message."""
        tag = field.full_tag
        value = field.value

        if tag == "20":
            mt103.sender_reference = value

        elif tag == "13C":
            mt103.time_indication = value

        elif tag == "23B":
            mt103.bank_operation_code = value

        elif tag == "23E":
            mt103.instruction_codes.append(value)

        elif tag == "26T":
            mt103.transaction_type_code = value

        elif tag == "32A":
            parsed_date, currency, amount = MT103Field.parse_32a(value)
            mt103.value_date = parsed_date
            mt103.currency = currency
            mt103.amount = amount

        elif tag == "33B":
            mt103.instructed_currency = value[:3]
            mt103.instructed_amount = Decimal(value[3:].replace(",", "."))

        elif tag == "36":
            mt103.exchange_rate = Decimal(value.replace(",", "."))

        elif tag.startswith("50"):
            mt103.ordering_customer_option = tag[2:] if len(tag) > 2 else "K"
            cls._parse_party(mt103.ordering_customer, value)

        elif tag.startswith("52"):
            if mt103.ordering_institution is None:
                mt103.ordering_institution = MT103Agent()
            cls._parse_agent(mt103.ordering_institution, value, tag[2:] if len(tag) > 2 else "")

        elif tag.startswith("53"):
            if mt103.senders_correspondent is None:
                mt103.senders_correspondent = MT103Agent()
            cls._parse_agent(mt103.senders_correspondent, value, tag[2:] if len(tag) > 2 else "")

        elif tag.startswith("54"):
            if mt103.receivers_correspondent is None:
                mt103.receivers_correspondent = MT103Agent()
            cls._parse_agent(mt103.receivers_correspondent, value, tag[2:] if len(tag) > 2 else "")

        elif tag.startswith("55"):
            if mt103.third_reimbursement_institution is None:
                mt103.third_reimbursement_institution = MT103Agent()
            cls._parse_agent(mt103.third_reimbursement_institution, value, tag[2:] if len(tag) > 2 else "")

        elif tag.startswith("56"):
            if mt103.intermediary_institution is None:
                mt103.intermediary_institution = MT103Agent()
            cls._parse_agent(mt103.intermediary_institution, value, tag[2:] if len(tag) > 2 else "")

        elif tag.startswith("57"):
            if mt103.account_with_institution is None:
                mt103.account_with_institution = MT103Agent()
            cls._parse_agent(mt103.account_with_institution, value, tag[2:] if len(tag) > 2 else "")

        elif tag.startswith("59"):
            mt103.beneficiary_option = tag[2:] if len(tag) > 2 else ""
            cls._parse_party(mt103.beneficiary, value)

        elif tag == "70":
            mt103.remittance_info = value.split("\n")

        elif tag == "71A":
            mt103.charges.details = value

        elif tag == "71F":
            currency = value[:3]
            amount = Decimal(value[3:].replace(",", "."))
            mt103.charges.sender_charges.append((currency, amount))

        elif tag == "71G":
            currency = value[:3]
            amount = Decimal(value[3:].replace(",", "."))
            mt103.charges.receiver_charges = (currency, amount)

        elif tag == "72":
            mt103.sender_to_receiver_info = value.split("\n")

        elif tag == "77B":
            mt103.regulatory_reporting = value.split("\n")

        elif tag == "77T":
            mt103.envelope_contents = value

    @classmethod
    def _parse_party(cls, party: MT103Party, value: str) -> None:
        """Parse party information from field value."""
        lines = value.split("\n")
        address_lines = []

        for line in lines:
            if line.startswith("/"):
                if party.account is None:
                    party.account = line[1:]
                else:
                    address_lines.append(line)
            elif len(line) == 8 or len(line) == 11:
                # Could be BIC
                if re.match(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$", line):
                    party.bic = line
                else:
                    address_lines.append(line)
            else:
                address_lines.append(line)

        if address_lines:
            if not party.name and address_lines:
                party.name = address_lines[0]
                address_lines = address_lines[1:]
            if address_lines:
                party.address_line1 = address_lines[0]
            if len(address_lines) > 1:
                party.address_line2 = address_lines[1]
            if len(address_lines) > 2:
                party.address_line3 = address_lines[2]

    @classmethod
    def _parse_agent(cls, agent: MT103Agent, value: str, option: str) -> None:
        """Parse agent/institution information from field value."""
        lines = value.split("\n")
        address_lines = []

        for line in lines:
            if line.startswith("/") and len(line) == 2:
                agent.party_type = line[1]
            elif line.startswith("/"):
                if agent.account is None:
                    agent.account = line[1:]
                else:
                    address_lines.append(line)
            elif len(line) == 8 or len(line) == 11:
                # Could be BIC
                if re.match(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$", line):
                    agent.bic = line
                else:
                    address_lines.append(line)
            else:
                address_lines.append(line)

        if option == "B":
            agent.location = "\n".join(address_lines)
        elif address_lines:
            if not agent.name and address_lines:
                agent.name = address_lines[0]
                address_lines = address_lines[1:]
            if address_lines:
                agent.address_line1 = address_lines[0]
            if len(address_lines) > 1:
                agent.address_line2 = address_lines[1]


class MT103Builder:
    """Builder for creating MT103 messages."""

    def __init__(self):
        self._message = MT103Message()

    def reset(self) -> "MT103Builder":
        """Reset the builder."""
        self._message = MT103Message()
        return self

    def set_sender(self, bic: str) -> "MT103Builder":
        """Set sender BIC."""
        self._message.sender_bic = bic
        return self

    def set_receiver(self, bic: str) -> "MT103Builder":
        """Set receiver BIC."""
        self._message.receiver_bic = bic
        return self

    def set_reference(self, reference: str) -> "MT103Builder":
        """Set sender's reference (field 20)."""
        self._message.sender_reference = reference[:16]
        return self

    def set_operation_code(self, code: str) -> "MT103Builder":
        """Set bank operation code (field 23B)."""
        self._message.bank_operation_code = code
        return self

    def add_instruction_code(self, code: str, additional: str = "") -> "MT103Builder":
        """Add instruction code (field 23E)."""
        if additional:
            self._message.instruction_codes.append(f"{code}/{additional}")
        else:
            self._message.instruction_codes.append(code)
        return self

    def set_value_date(self, dt: date) -> "MT103Builder":
        """Set value date (field 32A)."""
        self._message.value_date = dt
        return self

    def set_amount(self, currency: str, amount: Decimal) -> "MT103Builder":
        """Set settlement currency and amount (field 32A)."""
        self._message.currency = currency
        self._message.amount = amount
        return self

    def set_instructed_amount(self, currency: str, amount: Decimal) -> "MT103Builder":
        """Set instructed currency and amount (field 33B)."""
        self._message.instructed_currency = currency
        self._message.instructed_amount = amount
        return self

    def set_exchange_rate(self, rate: Decimal) -> "MT103Builder":
        """Set exchange rate (field 36)."""
        self._message.exchange_rate = rate
        return self

    def set_ordering_customer(
        self,
        name: str,
        account: Optional[str] = None,
        address1: str = "",
        address2: str = "",
        address3: str = "",
        bic: Optional[str] = None,
        option: str = "K",
    ) -> "MT103Builder":
        """Set ordering customer (field 50)."""
        self._message.ordering_customer = MT103Party(
            account=account,
            bic=bic,
            name=name,
            address_line1=address1,
            address_line2=address2,
            address_line3=address3,
        )
        self._message.ordering_customer_option = option
        return self

    def set_beneficiary(
        self,
        name: str,
        account: Optional[str] = None,
        address1: str = "",
        address2: str = "",
        address3: str = "",
        bic: Optional[str] = None,
        option: str = "",
    ) -> "MT103Builder":
        """Set beneficiary customer (field 59)."""
        self._message.beneficiary = MT103Party(
            account=account,
            bic=bic,
            name=name,
            address_line1=address1,
            address_line2=address2,
            address_line3=address3,
        )
        self._message.beneficiary_option = option
        return self

    def set_ordering_institution(
        self, bic: Optional[str] = None, name: str = "", account: Optional[str] = None
    ) -> "MT103Builder":
        """Set ordering institution (field 52)."""
        self._message.ordering_institution = MT103Agent(bic=bic, name=name, account=account)
        return self

    def set_account_with_institution(
        self, bic: Optional[str] = None, name: str = "", account: Optional[str] = None
    ) -> "MT103Builder":
        """Set account with institution (field 57)."""
        self._message.account_with_institution = MT103Agent(bic=bic, name=name, account=account)
        return self

    def set_intermediary_institution(
        self, bic: Optional[str] = None, name: str = "", account: Optional[str] = None
    ) -> "MT103Builder":
        """Set intermediary institution (field 56)."""
        self._message.intermediary_institution = MT103Agent(bic=bic, name=name, account=account)
        return self

    def set_charges(
        self,
        details: str = "SHA",
        sender_charges: List[tuple] = None,
        receiver_charges: Optional[tuple] = None,
    ) -> "MT103Builder":
        """Set charges (fields 71A/71F/71G)."""
        self._message.charges = MT103Charges(
            details=details,
            sender_charges=sender_charges or [],
            receiver_charges=receiver_charges,
        )
        return self

    def set_remittance_info(self, *lines: str) -> "MT103Builder":
        """Set remittance information (field 70)."""
        self._message.remittance_info = list(lines[:4])
        return self

    def set_sender_to_receiver_info(self, *lines: str) -> "MT103Builder":
        """Set sender to receiver information (field 72)."""
        self._message.sender_to_receiver_info = list(lines[:6])
        return self

    def set_regulatory_reporting(self, *lines: str) -> "MT103Builder":
        """Set regulatory reporting (field 77B)."""
        self._message.regulatory_reporting = list(lines[:3])
        return self

    def set_priority(self, priority: str) -> "MT103Builder":
        """Set message priority (N=Normal, U=Urgent)."""
        self._message.priority = priority
        return self

    def set_stp(self, is_stp: bool = True) -> "MT103Builder":
        """Set STP variant."""
        self._message.is_stp = is_stp
        return self

    def set_remit(self, is_remit: bool = True) -> "MT103Builder":
        """Set REMIT variant."""
        self._message.is_remit = is_remit
        return self

    def set_uetr(self, uetr: str) -> "MT103Builder":
        """Set UETR."""
        self._message.uetr = uetr
        return self

    def build(self) -> MT103Message:
        """Build the MT103 message."""
        return self._message

    @classmethod
    def create_payment(
        cls,
        sender_bic: str,
        receiver_bic: str,
        reference: str,
        value_date: date,
        currency: str,
        amount: Decimal,
        ordering_customer_name: str,
        ordering_customer_account: str,
        beneficiary_name: str,
        beneficiary_account: str,
        charges: str = "SHA",
        remittance_info: str = "",
    ) -> MT103Message:
        """
        Create a simple MT103 payment message.

        Args:
            sender_bic: Sender bank BIC
            receiver_bic: Receiver bank BIC
            reference: Sender's reference
            value_date: Value/settlement date
            currency: Currency code
            amount: Payment amount
            ordering_customer_name: Debtor name
            ordering_customer_account: Debtor account
            beneficiary_name: Creditor name
            beneficiary_account: Creditor account
            charges: Charge details (BEN/OUR/SHA)
            remittance_info: Remittance information

        Returns:
            MT103Message
        """
        builder = cls()

        builder.set_sender(sender_bic)
        builder.set_receiver(receiver_bic)
        builder.set_reference(reference)
        builder.set_operation_code("CRED")
        builder.set_value_date(value_date)
        builder.set_amount(currency, amount)
        builder.set_ordering_customer(
            name=ordering_customer_name,
            account=ordering_customer_account,
        )
        builder.set_beneficiary(
            name=beneficiary_name,
            account=beneficiary_account,
        )
        builder.set_charges(details=charges)

        if remittance_info:
            builder.set_remittance_info(remittance_info)

        return builder.build()
