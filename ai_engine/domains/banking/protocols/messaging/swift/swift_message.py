"""
SWIFT Message Data Structures

Base classes for SWIFT MT message representation including:
- Block structures (1-5)
- Field representation
- Message envelope
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


@dataclass
class SwiftField:
    """Represents a single SWIFT field."""

    tag: str
    value: str
    qualifier: str = ""  # Option letter (A, B, C, D, etc.)
    line_number: int = 0

    @property
    def full_tag(self) -> str:
        """Get full tag including qualifier."""
        if self.qualifier:
            return f"{self.tag}{self.qualifier}"
        return self.tag

    @property
    def lines(self) -> List[str]:
        """Get value as lines."""
        return self.value.split("\n")

    def to_swift(self) -> str:
        """Convert to SWIFT format."""
        return f":{self.full_tag}:{self.value}"

    def __str__(self) -> str:
        return self.to_swift()


@dataclass
class SwiftBlock:
    """Base class for SWIFT message blocks."""

    block_type: str
    content: str = ""

    def to_swift(self) -> str:
        """Convert to SWIFT block format."""
        return f"{{{self.block_type}:{self.content}}}"


@dataclass
class SwiftBasicHeader(SwiftBlock):
    """
    Block 1: Basic Header Block

    Format: {1:F01BANKBEBBAXXX0000000000}
    - F = Application ID (F=FIN, A=GPA, L=GPA Lite)
    - 01 = Service ID (01=FIN/GPA, 21=ACK/NAK)
    - BANKBEBBAXXX = Logical Terminal Address (LT Address)
    - 0000 = Session Number
    - 000000 = Sequence Number
    """

    block_type: str = "1"
    application_id: str = "F"  # F=FIN, A=GPA, L=GPA Lite
    service_id: str = "01"  # 01=FIN/GPA, 21=ACK/NAK
    lt_address: str = ""  # 12-char Logical Terminal Address
    session_number: str = "0000"
    sequence_number: str = "000000"

    @property
    def bic(self) -> str:
        """Extract BIC from LT address."""
        if len(self.lt_address) >= 8:
            return self.lt_address[:8]
        return self.lt_address

    @classmethod
    def from_content(cls, content: str) -> "SwiftBasicHeader":
        """Parse from block content."""
        header = cls()
        if len(content) >= 1:
            header.application_id = content[0]
        if len(content) >= 3:
            header.service_id = content[1:3]
        if len(content) >= 15:
            header.lt_address = content[3:15]
        if len(content) >= 19:
            header.session_number = content[15:19]
        if len(content) >= 25:
            header.sequence_number = content[19:25]
        header.content = content
        return header

    def to_swift(self) -> str:
        """Convert to SWIFT format."""
        content = (
            f"{self.application_id}"
            f"{self.service_id}"
            f"{self.lt_address}"
            f"{self.session_number}"
            f"{self.sequence_number}"
        )
        return f"{{1:{content}}}"


class MessageDirection(Enum):
    """Message direction indicator."""
    INPUT = "I"  # Message sent to SWIFT
    OUTPUT = "O"  # Message received from SWIFT


@dataclass
class SwiftApplicationHeader(SwiftBlock):
    """
    Block 2: Application Header Block

    Input format:  {2:I103BANKBEBBXXXXN}
    Output format: {2:O1031200010103BANKBEBBXXXX01200101031200N}

    - I/O = Input/Output indicator
    - 103 = Message Type
    - BANKBEBBXXXX = Destination/Sender BIC
    - N = Priority (N=Normal, S=System, U=Urgent)
    """

    block_type: str = "2"
    direction: MessageDirection = MessageDirection.INPUT
    message_type: str = ""  # 3-digit MT type
    destination_bic: str = ""  # For input messages
    sender_bic: str = ""  # For output messages
    priority: str = "N"  # N=Normal, S=System, U=Urgent
    delivery_monitoring: str = ""  # 1=Non-delivery warning, 2=Delivery notification
    obsolescence_period: str = ""  # 003 = 15 mins, 020 = 100 mins

    # Output message specific fields
    input_time: str = ""  # HHMM format
    input_date: str = ""  # YYMMDD format
    output_date: str = ""  # YYMMDD format
    output_time: str = ""  # HHMM format
    mir: str = ""  # Message Input Reference

    @classmethod
    def from_content(cls, content: str) -> "SwiftApplicationHeader":
        """Parse from block content."""
        header = cls()
        header.content = content

        if not content:
            return header

        header.direction = (
            MessageDirection.INPUT if content[0] == "I" else MessageDirection.OUTPUT
        )

        if header.direction == MessageDirection.INPUT:
            # Input format: I103BANKBEBBXXXXN
            if len(content) >= 4:
                header.message_type = content[1:4]
            if len(content) >= 16:
                header.destination_bic = content[4:16]
            if len(content) >= 17:
                header.priority = content[16]
            if len(content) >= 18:
                header.delivery_monitoring = content[17]
            if len(content) >= 21:
                header.obsolescence_period = content[18:21]
        else:
            # Output format: O1031200010103BANKBEBBXXXX01200101031200N
            if len(content) >= 4:
                header.message_type = content[1:4]
            if len(content) >= 8:
                header.input_time = content[4:8]
            if len(content) >= 14:
                header.input_date = content[8:14]
            if len(content) >= 26:
                header.sender_bic = content[14:26]
            if len(content) >= 32:
                header.mir = content[26:32]
            if len(content) >= 38:
                header.output_date = content[32:38]
            if len(content) >= 42:
                header.output_time = content[38:42]
            if len(content) >= 43:
                header.priority = content[42]

        return header

    def to_swift(self) -> str:
        """Convert to SWIFT format."""
        if self.direction == MessageDirection.INPUT:
            content = (
                f"I{self.message_type}"
                f"{self.destination_bic}"
                f"{self.priority}"
                f"{self.delivery_monitoring}"
                f"{self.obsolescence_period}"
            )
        else:
            content = (
                f"O{self.message_type}"
                f"{self.input_time}"
                f"{self.input_date}"
                f"{self.sender_bic}"
                f"{self.mir}"
                f"{self.output_date}"
                f"{self.output_time}"
                f"{self.priority}"
            )
        return f"{{2:{content.rstrip()}}}"


@dataclass
class SwiftUserHeader(SwiftBlock):
    """
    Block 3: User Header Block

    Format: {3:{113:SEPA}{108:MT103-001}{121:unique-id}}

    Contains optional banking priority, MUR, service ID, etc.
    """

    block_type: str = "3"
    fields: Dict[str, str] = field(default_factory=dict)

    @property
    def banking_priority(self) -> Optional[str]:
        """Get banking priority (tag 113)."""
        return self.fields.get("113")

    @property
    def mur(self) -> Optional[str]:
        """Get Message User Reference (tag 108)."""
        return self.fields.get("108")

    @property
    def service_type_id(self) -> Optional[str]:
        """Get Service Type Identifier (tag 111)."""
        return self.fields.get("111")

    @property
    def uetr(self) -> Optional[str]:
        """Get Unique End-to-end Transaction Reference (tag 121)."""
        return self.fields.get("121")

    @property
    def service_identifier(self) -> Optional[str]:
        """Get Service Identifier (tag 103)."""
        return self.fields.get("103")

    @classmethod
    def from_content(cls, content: str) -> "SwiftUserHeader":
        """Parse from block content."""
        header = cls()
        header.content = content

        # Parse sub-blocks like {113:SEPA}{108:MT103-001}
        import re

        pattern = r"\{(\d+):([^}]*)\}"
        matches = re.findall(pattern, content)
        for tag, value in matches:
            header.fields[tag] = value

        return header

    def to_swift(self) -> str:
        """Convert to SWIFT format."""
        if not self.fields:
            return ""

        sub_blocks = "".join(f"{{{tag}:{value}}}" for tag, value in self.fields.items())
        return f"{{3:{sub_blocks}}}"


@dataclass
class SwiftTextBlock(SwiftBlock):
    """
    Block 4: Text Block

    Contains the actual message fields.
    Format: {4:\n:20:REFERENCE\n:23B:CRED\n:32A:230101USD1000,00\n...}
    """

    block_type: str = "4"
    fields: List[SwiftField] = field(default_factory=list)

    def get_field(self, tag: str) -> Optional[SwiftField]:
        """Get first field by tag."""
        for f in self.fields:
            if f.tag == tag or f.full_tag == tag:
                return f
        return None

    def get_fields(self, tag: str) -> List[SwiftField]:
        """Get all fields matching tag."""
        return [f for f in self.fields if f.tag == tag or f.full_tag == tag]

    def get_field_value(self, tag: str) -> Optional[str]:
        """Get value of first field matching tag."""
        f = self.get_field(tag)
        return f.value if f else None

    def add_field(self, tag: str, value: str, qualifier: str = "") -> None:
        """Add a field to the text block."""
        self.fields.append(SwiftField(tag=tag, value=value, qualifier=qualifier))

    @classmethod
    def from_content(cls, content: str) -> "SwiftTextBlock":
        """Parse from block content."""
        text_block = cls()
        text_block.content = content

        # Split by field delimiter pattern
        lines = content.split("\n")
        current_tag = None
        current_value_lines = []

        for line in lines:
            if line.startswith(":"):
                # Save previous field
                if current_tag:
                    value = "\n".join(current_value_lines)
                    # Parse tag and qualifier
                    if len(current_tag) > 2 and current_tag[-1].isalpha():
                        base_tag = current_tag[:-1]
                        qualifier = current_tag[-1]
                    else:
                        base_tag = current_tag
                        qualifier = ""
                    text_block.fields.append(
                        SwiftField(tag=base_tag, value=value, qualifier=qualifier)
                    )

                # Parse new field
                colon_pos = line.find(":", 1)
                if colon_pos > 0:
                    current_tag = line[1:colon_pos]
                    current_value_lines = [line[colon_pos + 1 :]]
                else:
                    current_tag = None
                    current_value_lines = []
            elif current_tag:
                current_value_lines.append(line)

        # Save last field
        if current_tag:
            value = "\n".join(current_value_lines)
            if len(current_tag) > 2 and current_tag[-1].isalpha():
                base_tag = current_tag[:-1]
                qualifier = current_tag[-1]
            else:
                base_tag = current_tag
                qualifier = ""
            text_block.fields.append(
                SwiftField(tag=base_tag, value=value, qualifier=qualifier)
            )

        return text_block

    def to_swift(self) -> str:
        """Convert to SWIFT format."""
        field_lines = "\n".join(f.to_swift() for f in self.fields)
        return f"{{4:\n{field_lines}\n-}}"


@dataclass
class SwiftTrailerBlock(SwiftBlock):
    """
    Block 5: Trailer Block

    Contains authentication and checksum data.
    Format: {5:{MAC:00000000}{CHK:123456789ABC}}
    """

    block_type: str = "5"
    fields: Dict[str, str] = field(default_factory=dict)

    @property
    def mac(self) -> Optional[str]:
        """Get Message Authentication Code."""
        return self.fields.get("MAC")

    @property
    def checksum(self) -> Optional[str]:
        """Get checksum (CHK)."""
        return self.fields.get("CHK")

    @property
    def pde(self) -> Optional[str]:
        """Get Possible Duplicate Emission."""
        return self.fields.get("PDE")

    @property
    def dlm(self) -> Optional[str]:
        """Get Delayed Message."""
        return self.fields.get("DLM")

    @classmethod
    def from_content(cls, content: str) -> "SwiftTrailerBlock":
        """Parse from block content."""
        trailer = cls()
        trailer.content = content

        # Parse sub-blocks like {MAC:00000000}{CHK:123456789ABC}
        import re

        pattern = r"\{([A-Z]+):([^}]*)\}"
        matches = re.findall(pattern, content)
        for tag, value in matches:
            trailer.fields[tag] = value

        return trailer

    def to_swift(self) -> str:
        """Convert to SWIFT format."""
        if not self.fields:
            return "{5:}"

        sub_blocks = "".join(f"{{{tag}:{value}}}" for tag, value in self.fields.items())
        return f"{{5:{sub_blocks}}}"


@dataclass
class SwiftMessage:
    """
    Complete SWIFT MT message with all blocks.
    """

    basic_header: SwiftBasicHeader = field(default_factory=SwiftBasicHeader)
    application_header: SwiftApplicationHeader = field(
        default_factory=SwiftApplicationHeader
    )
    user_header: Optional[SwiftUserHeader] = None
    text_block: SwiftTextBlock = field(default_factory=SwiftTextBlock)
    trailer: Optional[SwiftTrailerBlock] = None

    # Metadata
    raw_message: str = ""
    parse_errors: List[str] = field(default_factory=list)

    @property
    def message_type(self) -> str:
        """Get message type (e.g., '103', '202')."""
        return self.application_header.message_type

    @property
    def sender_bic(self) -> str:
        """Get sender BIC."""
        if self.application_header.direction == MessageDirection.OUTPUT:
            return self.application_header.sender_bic[:8]
        return self.basic_header.bic

    @property
    def receiver_bic(self) -> str:
        """Get receiver BIC."""
        if self.application_header.direction == MessageDirection.INPUT:
            return self.application_header.destination_bic[:8]
        return ""

    @property
    def reference(self) -> Optional[str]:
        """Get transaction reference (field 20)."""
        return self.text_block.get_field_value("20")

    @property
    def uetr(self) -> Optional[str]:
        """Get UETR from user header."""
        if self.user_header:
            return self.user_header.uetr
        return None

    def get_field(self, tag: str) -> Optional[SwiftField]:
        """Get field from text block."""
        return self.text_block.get_field(tag)

    def get_field_value(self, tag: str) -> Optional[str]:
        """Get field value from text block."""
        return self.text_block.get_field_value(tag)

    def to_swift(self) -> str:
        """Convert to complete SWIFT message format."""
        parts = [
            self.basic_header.to_swift(),
            self.application_header.to_swift(),
        ]

        if self.user_header and self.user_header.fields:
            parts.append(self.user_header.to_swift())

        parts.append(self.text_block.to_swift())

        if self.trailer:
            parts.append(self.trailer.to_swift())

        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "message_type": self.message_type,
            "sender_bic": self.sender_bic,
            "receiver_bic": self.receiver_bic,
            "reference": self.reference,
            "uetr": self.uetr,
            "direction": self.application_header.direction.value,
            "priority": self.application_header.priority,
            "fields": {},
        }

        for f in self.text_block.fields:
            tag = f.full_tag
            if tag in result["fields"]:
                if isinstance(result["fields"][tag], list):
                    result["fields"][tag].append(f.value)
                else:
                    result["fields"][tag] = [result["fields"][tag], f.value]
            else:
                result["fields"][tag] = f.value

        if self.user_header:
            result["user_header"] = self.user_header.fields

        if self.trailer:
            result["trailer"] = self.trailer.fields

        return result

    def __str__(self) -> str:
        return f"SwiftMessage(MT{self.message_type}, ref={self.reference})"
