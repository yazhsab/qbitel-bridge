"""
MT202 - General Financial Institution Transfer

The MT202 is used for bank-to-bank transfers, typically for:
- Nostro/Vostro account movements
- Cover payments for MT103
- Interbank transfers

Supports:
- MT202: Standard financial institution transfer
- MT202 COV: Cover payment for customer credit transfer
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
    MT202_FIELDS,
)


@dataclass
class MT202Agent:
    """Agent/Institution information for MT202."""

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

    def to_b_format(self) -> str:
        """Format for option B (Location)."""
        lines = []
        if self.party_type:
            lines.append(f"/{self.party_type}")
        if self.account:
            lines.append(f"/{self.account}")
        if self.location:
            lines.append(self.location[:35])
        return "\n".join(lines)

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
        return "\n".join(lines[:4])


@dataclass
class MT202CoverInfo:
    """
    Cover payment information for MT202 COV.

    Contains underlying customer credit transfer details.
    """

    # Field 50: Ordering Customer
    ordering_customer_account: Optional[str] = None
    ordering_customer_bic: Optional[str] = None
    ordering_customer_name: str = ""
    ordering_customer_address: List[str] = field(default_factory=list)

    # Field 52: Ordering Institution
    ordering_institution_bic: Optional[str] = None
    ordering_institution_account: Optional[str] = None
    ordering_institution_name: str = ""

    # Field 56: Intermediary Institution
    intermediary_bic: Optional[str] = None
    intermediary_account: Optional[str] = None
    intermediary_name: str = ""

    # Field 57: Account With Institution
    account_with_bic: Optional[str] = None
    account_with_account: Optional[str] = None
    account_with_name: str = ""

    # Field 59: Beneficiary Customer
    beneficiary_account: Optional[str] = None
    beneficiary_bic: Optional[str] = None
    beneficiary_name: str = ""
    beneficiary_address: List[str] = field(default_factory=list)

    # Field 70: Remittance Information
    remittance_info: List[str] = field(default_factory=list)

    # Field 72: Sender to Receiver Information
    sender_to_receiver_info: List[str] = field(default_factory=list)

    # Field 33B: Currency/Instructed Amount
    instructed_currency: Optional[str] = None
    instructed_amount: Optional[Decimal] = None

    def to_sequence_b_fields(self) -> List[SwiftField]:
        """Generate sequence B fields for COV message."""
        fields = []

        # Field 50: Ordering Customer
        if self.ordering_customer_name or self.ordering_customer_account:
            lines = []
            if self.ordering_customer_account:
                lines.append(f"/{self.ordering_customer_account}")
            if self.ordering_customer_bic:
                lines.append(self.ordering_customer_bic)
            elif self.ordering_customer_name:
                lines.append(self.ordering_customer_name[:35])
                lines.extend(addr[:35] for addr in self.ordering_customer_address[:3])

            qualifier = "A" if self.ordering_customer_bic else "K"
            fields.append(SwiftField(tag="50", value="\n".join(lines[:4]), qualifier=qualifier))

        # Field 52: Ordering Institution
        if self.ordering_institution_bic:
            lines = []
            if self.ordering_institution_account:
                lines.append(f"/{self.ordering_institution_account}")
            lines.append(self.ordering_institution_bic)
            fields.append(SwiftField(tag="52", value="\n".join(lines), qualifier="A"))

        # Field 56: Intermediary Institution
        if self.intermediary_bic or self.intermediary_account:
            lines = []
            if self.intermediary_account:
                lines.append(f"/{self.intermediary_account}")
            if self.intermediary_bic:
                lines.append(self.intermediary_bic)
                fields.append(SwiftField(tag="56", value="\n".join(lines), qualifier="A"))
            elif self.intermediary_account:
                fields.append(SwiftField(tag="56", value=f"/{self.intermediary_account}", qualifier="C"))

        # Field 57: Account With Institution
        if self.account_with_bic or self.account_with_account:
            lines = []
            if self.account_with_account:
                lines.append(f"/{self.account_with_account}")
            if self.account_with_bic:
                lines.append(self.account_with_bic)
                fields.append(SwiftField(tag="57", value="\n".join(lines), qualifier="A"))
            elif self.account_with_account:
                fields.append(SwiftField(tag="57", value=f"/{self.account_with_account}", qualifier="C"))

        # Field 59: Beneficiary Customer
        if self.beneficiary_name or self.beneficiary_account:
            lines = []
            if self.beneficiary_account:
                lines.append(f"/{self.beneficiary_account}")
            if self.beneficiary_bic:
                lines.append(self.beneficiary_bic)
                fields.append(SwiftField(tag="59", value="\n".join(lines), qualifier="A"))
            else:
                if self.beneficiary_name:
                    lines.append(self.beneficiary_name[:35])
                    lines.extend(addr[:35] for addr in self.beneficiary_address[:3])
                fields.append(SwiftField(tag="59", value="\n".join(lines[:4]), qualifier=""))

        # Field 70: Remittance Information
        if self.remittance_info:
            lines = [line[:35] for line in self.remittance_info[:4]]
            fields.append(SwiftField(tag="70", value="\n".join(lines), qualifier=""))

        # Field 72: Sender to Receiver Information
        if self.sender_to_receiver_info:
            lines = [line[:35] for line in self.sender_to_receiver_info[:6]]
            fields.append(SwiftField(tag="72", value="\n".join(lines), qualifier=""))

        # Field 33B: Currency/Instructed Amount
        if self.instructed_currency and self.instructed_amount:
            amount_str = f"{self.instructed_amount:,.2f}".replace(",", "").replace(".", ",")
            fields.append(SwiftField(tag="33B", value=f"{self.instructed_currency}{amount_str}", qualifier=""))

        return fields


class MT202Field:
    """Field helpers for MT202."""

    @staticmethod
    def format_32a(value_date: date, currency: str, amount: Decimal) -> str:
        """Format field 32A (Value Date/Currency/Amount)."""
        date_str = value_date.strftime("%y%m%d")
        amount_str = f"{amount:,.2f}".replace(",", "").replace(".", ",")
        return f"{date_str}{currency}{amount_str}"

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
class MT202Message:
    """
    MT202 General Financial Institution Transfer message.

    This represents a fully parsed or constructed MT202 message.
    """

    # Required fields
    transaction_reference: str = ""  # Field 20
    related_reference: str = ""  # Field 21
    value_date: Optional[date] = None  # Field 32A
    currency: str = "USD"
    amount: Decimal = Decimal("0")

    # Beneficiary Institution (field 58) - REQUIRED
    beneficiary_institution: MT202Agent = field(default_factory=MT202Agent)
    beneficiary_option: str = "A"  # A or D

    # Optional agents/institutions
    ordering_institution: Optional[MT202Agent] = None  # Field 52
    senders_correspondent: Optional[MT202Agent] = None  # Field 53
    receivers_correspondent: Optional[MT202Agent] = None  # Field 54
    intermediary: Optional[MT202Agent] = None  # Field 56
    account_with_institution: Optional[MT202Agent] = None  # Field 57

    # Optional fields
    time_indication: Optional[str] = None  # Field 13C
    sender_to_receiver_info: List[str] = field(default_factory=list)  # Field 72

    # Header information
    sender_bic: str = ""
    receiver_bic: str = ""
    priority: str = "N"  # N=Normal, U=Urgent
    uetr: Optional[str] = None  # Unique End-to-end Transaction Reference

    # COV variant
    is_cov: bool = False
    cover_info: Optional[MT202CoverInfo] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.uetr:
            self.uetr = str(uuid.uuid4())

    @property
    def message_type(self) -> str:
        """Get the specific MT202 variant."""
        if self.is_cov:
            return "202_COV"
        return "202"

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
        if self.is_cov:
            user_header_fields["119"] = "COV"
        if self.uetr:
            user_header_fields["121"] = self.uetr

        if user_header_fields:
            msg.user_header = SwiftUserHeader(fields=user_header_fields)

        # Block 4: Text Block
        self._build_text_block(msg.text_block)

        return msg

    def _build_text_block(self, text_block: SwiftTextBlock) -> None:
        """Build the text block fields."""
        # Sequence A: General Information

        # Field 20: Transaction Reference Number
        text_block.add_field("20", self.transaction_reference[:16])

        # Field 21: Related Reference
        text_block.add_field("21", self.related_reference[:16])

        # Field 13C: Time Indication (optional)
        if self.time_indication:
            text_block.add_field("13C", self.time_indication)

        # Field 32A: Value Date/Currency/Amount
        if self.value_date:
            text_block.add_field(
                "32A",
                MT202Field.format_32a(self.value_date, self.currency, self.amount),
            )

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
                text_block.add_field("53", self.senders_correspondent.to_b_format(), "B")
            else:
                text_block.add_field("53", self.senders_correspondent.to_d_format(), "D")

        # Field 54: Receiver's Correspondent (optional)
        if self.receivers_correspondent:
            if self.receivers_correspondent.bic:
                text_block.add_field("54", self.receivers_correspondent.to_a_format(), "A")
            elif self.receivers_correspondent.location:
                text_block.add_field("54", self.receivers_correspondent.to_b_format(), "B")
            else:
                text_block.add_field("54", self.receivers_correspondent.to_d_format(), "D")

        # Field 56: Intermediary (optional)
        if self.intermediary:
            if self.intermediary.bic:
                text_block.add_field("56", self.intermediary.to_a_format(), "A")
            else:
                text_block.add_field("56", self.intermediary.to_d_format(), "D")

        # Field 57: Account With Institution (optional)
        if self.account_with_institution:
            if self.account_with_institution.bic:
                text_block.add_field("57", self.account_with_institution.to_a_format(), "A")
            elif self.account_with_institution.location:
                text_block.add_field("57", self.account_with_institution.to_b_format(), "B")
            else:
                text_block.add_field("57", self.account_with_institution.to_d_format(), "D")

        # Field 58: Beneficiary Institution (required)
        if self.beneficiary_option == "A" and self.beneficiary_institution.bic:
            text_block.add_field("58", self.beneficiary_institution.to_a_format(), "A")
        else:
            text_block.add_field("58", self.beneficiary_institution.to_d_format(), "D")

        # Field 72: Sender to Receiver Information (optional)
        if self.sender_to_receiver_info:
            s2r_text = "\n".join(line[:35] for line in self.sender_to_receiver_info[:6])
            text_block.add_field("72", s2r_text)

        # Sequence B: Underlying Customer Credit Transfer Details (for COV only)
        if self.is_cov and self.cover_info:
            for cov_field in self.cover_info.to_sequence_b_fields():
                text_block.fields.append(cov_field)

    def to_swift(self) -> str:
        """Convert to SWIFT message string."""
        return self.to_swift_message().to_swift()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "message_type": f"MT{self.message_type}",
            "transaction_reference": self.transaction_reference,
            "related_reference": self.related_reference,
            "sender_bic": self.sender_bic,
            "receiver_bic": self.receiver_bic,
            "uetr": self.uetr,
            "value_date": self.value_date.isoformat() if self.value_date else None,
            "currency": self.currency,
            "amount": str(self.amount),
            "beneficiary_institution": {
                "bic": self.beneficiary_institution.bic,
                "account": self.beneficiary_institution.account,
                "name": self.beneficiary_institution.name,
            },
            "is_cov": self.is_cov,
            "created_at": self.created_at.isoformat(),
        }

        if self.ordering_institution:
            result["ordering_institution"] = {
                "bic": self.ordering_institution.bic,
                "account": self.ordering_institution.account,
                "name": self.ordering_institution.name,
            }

        if self.is_cov and self.cover_info:
            result["cover_info"] = {
                "ordering_customer_name": self.cover_info.ordering_customer_name,
                "ordering_customer_account": self.cover_info.ordering_customer_account,
                "beneficiary_name": self.cover_info.beneficiary_name,
                "beneficiary_account": self.cover_info.beneficiary_account,
                "remittance_info": self.cover_info.remittance_info,
            }

        return result


class MT202Parser:
    """Parser for MT202 messages."""

    @classmethod
    def from_swift_message(cls, msg: SwiftMessage) -> MT202Message:
        """Parse MT202 from SwiftMessage object."""
        mt202 = MT202Message()

        # Header info
        mt202.sender_bic = msg.sender_bic
        mt202.receiver_bic = msg.receiver_bic
        mt202.priority = msg.application_header.priority

        # Check variant
        if msg.user_header:
            service_id = msg.user_header.fields.get("119", "")
            mt202.is_cov = service_id == "COV"
            mt202.uetr = msg.user_header.uetr

        # Parse text block fields
        in_sequence_b = False
        cover_info = MT202CoverInfo() if mt202.is_cov else None

        for field in msg.text_block.fields:
            # Detect sequence B (starts with field 50 in COV messages after field 58)
            if mt202.is_cov and field.tag == "50" and mt202.beneficiary_institution.bic:
                in_sequence_b = True

            if in_sequence_b and cover_info:
                cls._parse_cover_field(cover_info, field)
            else:
                cls._parse_field(mt202, field)

        if cover_info and mt202.is_cov:
            mt202.cover_info = cover_info

        return mt202

    @classmethod
    def _parse_field(cls, mt202: MT202Message, field: SwiftField) -> None:
        """Parse a single field into MT202 message."""
        tag = field.full_tag
        value = field.value

        if tag == "20":
            mt202.transaction_reference = value

        elif tag == "21":
            mt202.related_reference = value

        elif tag == "13C":
            mt202.time_indication = value

        elif tag == "32A":
            parsed_date, currency, amount = MT202Field.parse_32a(value)
            mt202.value_date = parsed_date
            mt202.currency = currency
            mt202.amount = amount

        elif tag.startswith("52"):
            if mt202.ordering_institution is None:
                mt202.ordering_institution = MT202Agent()
            cls._parse_agent(mt202.ordering_institution, value, tag[2:] if len(tag) > 2 else "")

        elif tag.startswith("53"):
            if mt202.senders_correspondent is None:
                mt202.senders_correspondent = MT202Agent()
            cls._parse_agent(mt202.senders_correspondent, value, tag[2:] if len(tag) > 2 else "")

        elif tag.startswith("54"):
            if mt202.receivers_correspondent is None:
                mt202.receivers_correspondent = MT202Agent()
            cls._parse_agent(mt202.receivers_correspondent, value, tag[2:] if len(tag) > 2 else "")

        elif tag.startswith("56"):
            if mt202.intermediary is None:
                mt202.intermediary = MT202Agent()
            cls._parse_agent(mt202.intermediary, value, tag[2:] if len(tag) > 2 else "")

        elif tag.startswith("57"):
            if mt202.account_with_institution is None:
                mt202.account_with_institution = MT202Agent()
            cls._parse_agent(mt202.account_with_institution, value, tag[2:] if len(tag) > 2 else "")

        elif tag.startswith("58"):
            mt202.beneficiary_option = tag[2:] if len(tag) > 2 else "A"
            cls._parse_agent(mt202.beneficiary_institution, value, mt202.beneficiary_option)

        elif tag == "72":
            mt202.sender_to_receiver_info = value.split("\n")

    @classmethod
    def _parse_cover_field(cls, cover_info: MT202CoverInfo, field: SwiftField) -> None:
        """Parse a field from sequence B (cover info)."""
        tag = field.full_tag
        value = field.value
        lines = value.split("\n")

        if tag.startswith("50"):
            for line in lines:
                if line.startswith("/"):
                    if cover_info.ordering_customer_account is None:
                        cover_info.ordering_customer_account = line[1:]
                elif len(line) == 8 or len(line) == 11:
                    if re.match(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$", line):
                        cover_info.ordering_customer_bic = line
                    else:
                        if not cover_info.ordering_customer_name:
                            cover_info.ordering_customer_name = line
                        else:
                            cover_info.ordering_customer_address.append(line)
                else:
                    if not cover_info.ordering_customer_name:
                        cover_info.ordering_customer_name = line
                    else:
                        cover_info.ordering_customer_address.append(line)

        elif tag.startswith("52"):
            for line in lines:
                if line.startswith("/"):
                    cover_info.ordering_institution_account = line[1:]
                elif re.match(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$", line):
                    cover_info.ordering_institution_bic = line
                else:
                    cover_info.ordering_institution_name = line

        elif tag.startswith("56"):
            for line in lines:
                if line.startswith("/"):
                    cover_info.intermediary_account = line[1:]
                elif re.match(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$", line):
                    cover_info.intermediary_bic = line
                else:
                    cover_info.intermediary_name = line

        elif tag.startswith("57"):
            for line in lines:
                if line.startswith("/"):
                    cover_info.account_with_account = line[1:]
                elif re.match(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$", line):
                    cover_info.account_with_bic = line
                else:
                    cover_info.account_with_name = line

        elif tag.startswith("59"):
            for line in lines:
                if line.startswith("/"):
                    if cover_info.beneficiary_account is None:
                        cover_info.beneficiary_account = line[1:]
                elif len(line) == 8 or len(line) == 11:
                    if re.match(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$", line):
                        cover_info.beneficiary_bic = line
                    else:
                        if not cover_info.beneficiary_name:
                            cover_info.beneficiary_name = line
                        else:
                            cover_info.beneficiary_address.append(line)
                else:
                    if not cover_info.beneficiary_name:
                        cover_info.beneficiary_name = line
                    else:
                        cover_info.beneficiary_address.append(line)

        elif tag == "70":
            cover_info.remittance_info = lines

        elif tag == "72":
            cover_info.sender_to_receiver_info = lines

        elif tag == "33B":
            cover_info.instructed_currency = value[:3]
            cover_info.instructed_amount = Decimal(value[3:].replace(",", "."))

    @classmethod
    def _parse_agent(cls, agent: MT202Agent, value: str, option: str) -> None:
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


class MT202Builder:
    """Builder for creating MT202 messages."""

    def __init__(self):
        self._message = MT202Message()

    def reset(self) -> "MT202Builder":
        """Reset the builder."""
        self._message = MT202Message()
        return self

    def set_sender(self, bic: str) -> "MT202Builder":
        """Set sender BIC."""
        self._message.sender_bic = bic
        return self

    def set_receiver(self, bic: str) -> "MT202Builder":
        """Set receiver BIC."""
        self._message.receiver_bic = bic
        return self

    def set_transaction_reference(self, reference: str) -> "MT202Builder":
        """Set transaction reference (field 20)."""
        self._message.transaction_reference = reference[:16]
        return self

    def set_related_reference(self, reference: str) -> "MT202Builder":
        """Set related reference (field 21)."""
        self._message.related_reference = reference[:16]
        return self

    def set_value_date(self, dt: date) -> "MT202Builder":
        """Set value date (field 32A)."""
        self._message.value_date = dt
        return self

    def set_amount(self, currency: str, amount: Decimal) -> "MT202Builder":
        """Set currency and amount (field 32A)."""
        self._message.currency = currency
        self._message.amount = amount
        return self

    def set_ordering_institution(
        self, bic: Optional[str] = None, name: str = "", account: Optional[str] = None
    ) -> "MT202Builder":
        """Set ordering institution (field 52)."""
        self._message.ordering_institution = MT202Agent(
            bic=bic, name=name, account=account
        )
        return self

    def set_senders_correspondent(
        self, bic: Optional[str] = None, location: str = "", account: Optional[str] = None
    ) -> "MT202Builder":
        """Set sender's correspondent (field 53)."""
        self._message.senders_correspondent = MT202Agent(
            bic=bic, location=location, account=account
        )
        return self

    def set_receivers_correspondent(
        self, bic: Optional[str] = None, location: str = "", account: Optional[str] = None
    ) -> "MT202Builder":
        """Set receiver's correspondent (field 54)."""
        self._message.receivers_correspondent = MT202Agent(
            bic=bic, location=location, account=account
        )
        return self

    def set_intermediary(
        self, bic: Optional[str] = None, name: str = "", account: Optional[str] = None
    ) -> "MT202Builder":
        """Set intermediary (field 56)."""
        self._message.intermediary = MT202Agent(
            bic=bic, name=name, account=account
        )
        return self

    def set_account_with_institution(
        self, bic: Optional[str] = None, name: str = "", account: Optional[str] = None
    ) -> "MT202Builder":
        """Set account with institution (field 57)."""
        self._message.account_with_institution = MT202Agent(
            bic=bic, name=name, account=account
        )
        return self

    def set_beneficiary_institution(
        self, bic: Optional[str] = None, name: str = "", account: Optional[str] = None
    ) -> "MT202Builder":
        """Set beneficiary institution (field 58)."""
        self._message.beneficiary_institution = MT202Agent(
            bic=bic, name=name, account=account
        )
        self._message.beneficiary_option = "A" if bic else "D"
        return self

    def set_sender_to_receiver_info(self, *lines: str) -> "MT202Builder":
        """Set sender to receiver information (field 72)."""
        self._message.sender_to_receiver_info = list(lines[:6])
        return self

    def set_priority(self, priority: str) -> "MT202Builder":
        """Set message priority (N=Normal, U=Urgent)."""
        self._message.priority = priority
        return self

    def set_cov(self, is_cov: bool = True) -> "MT202Builder":
        """Set COV variant."""
        self._message.is_cov = is_cov
        if is_cov and self._message.cover_info is None:
            self._message.cover_info = MT202CoverInfo()
        return self

    def set_cover_ordering_customer(
        self,
        name: str,
        account: Optional[str] = None,
        bic: Optional[str] = None,
        address: List[str] = None,
    ) -> "MT202Builder":
        """Set cover ordering customer."""
        if self._message.cover_info is None:
            self._message.cover_info = MT202CoverInfo()
        self._message.cover_info.ordering_customer_name = name
        self._message.cover_info.ordering_customer_account = account
        self._message.cover_info.ordering_customer_bic = bic
        self._message.cover_info.ordering_customer_address = address or []
        return self

    def set_cover_beneficiary(
        self,
        name: str,
        account: Optional[str] = None,
        bic: Optional[str] = None,
        address: List[str] = None,
    ) -> "MT202Builder":
        """Set cover beneficiary."""
        if self._message.cover_info is None:
            self._message.cover_info = MT202CoverInfo()
        self._message.cover_info.beneficiary_name = name
        self._message.cover_info.beneficiary_account = account
        self._message.cover_info.beneficiary_bic = bic
        self._message.cover_info.beneficiary_address = address or []
        return self

    def set_cover_remittance_info(self, *lines: str) -> "MT202Builder":
        """Set cover remittance info."""
        if self._message.cover_info is None:
            self._message.cover_info = MT202CoverInfo()
        self._message.cover_info.remittance_info = list(lines[:4])
        return self

    def set_uetr(self, uetr: str) -> "MT202Builder":
        """Set UETR."""
        self._message.uetr = uetr
        return self

    def build(self) -> MT202Message:
        """Build the MT202 message."""
        return self._message

    @classmethod
    def create_transfer(
        cls,
        sender_bic: str,
        receiver_bic: str,
        transaction_reference: str,
        related_reference: str,
        value_date: date,
        currency: str,
        amount: Decimal,
        beneficiary_bic: str,
    ) -> MT202Message:
        """
        Create a simple MT202 transfer message.

        Args:
            sender_bic: Sender bank BIC
            receiver_bic: Receiver bank BIC
            transaction_reference: Transaction reference
            related_reference: Related reference
            value_date: Value/settlement date
            currency: Currency code
            amount: Transfer amount
            beneficiary_bic: Beneficiary institution BIC

        Returns:
            MT202Message
        """
        builder = cls()

        builder.set_sender(sender_bic)
        builder.set_receiver(receiver_bic)
        builder.set_transaction_reference(transaction_reference)
        builder.set_related_reference(related_reference)
        builder.set_value_date(value_date)
        builder.set_amount(currency, amount)
        builder.set_beneficiary_institution(bic=beneficiary_bic)

        return builder.build()

    @classmethod
    def create_cover_payment(
        cls,
        sender_bic: str,
        receiver_bic: str,
        transaction_reference: str,
        related_reference: str,
        value_date: date,
        currency: str,
        amount: Decimal,
        beneficiary_bic: str,
        underlying_ordering_customer_name: str,
        underlying_ordering_customer_account: str,
        underlying_beneficiary_name: str,
        underlying_beneficiary_account: str,
        remittance_info: str = "",
    ) -> MT202Message:
        """
        Create an MT202 COV cover payment message.

        Args:
            sender_bic: Sender bank BIC
            receiver_bic: Receiver bank BIC
            transaction_reference: Transaction reference
            related_reference: Related reference (typically the MT103 reference)
            value_date: Value/settlement date
            currency: Currency code
            amount: Transfer amount
            beneficiary_bic: Beneficiary institution BIC
            underlying_ordering_customer_name: Customer name from MT103
            underlying_ordering_customer_account: Customer account from MT103
            underlying_beneficiary_name: Beneficiary name from MT103
            underlying_beneficiary_account: Beneficiary account from MT103
            remittance_info: Remittance information from MT103

        Returns:
            MT202Message (COV variant)
        """
        builder = cls()

        builder.set_sender(sender_bic)
        builder.set_receiver(receiver_bic)
        builder.set_transaction_reference(transaction_reference)
        builder.set_related_reference(related_reference)
        builder.set_value_date(value_date)
        builder.set_amount(currency, amount)
        builder.set_beneficiary_institution(bic=beneficiary_bic)
        builder.set_cov(True)
        builder.set_cover_ordering_customer(
            name=underlying_ordering_customer_name,
            account=underlying_ordering_customer_account,
        )
        builder.set_cover_beneficiary(
            name=underlying_beneficiary_name,
            account=underlying_beneficiary_account,
        )

        if remittance_info:
            builder.set_cover_remittance_info(remittance_info)

        return builder.build()
