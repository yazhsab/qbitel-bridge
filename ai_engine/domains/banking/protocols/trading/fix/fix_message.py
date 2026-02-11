"""
FIX Message Data Structures

Base classes for FIX message representation including:
- Field and group structures
- Header and trailer
- Complete message
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal

from ai_engine.domains.banking.protocols.trading.fix.fix_codes import (
    FixVersion,
    FixMsgType,
    get_tag_name,
    FIX_TAG_NAMES,
)


# FIX field delimiter (SOH - Start of Header, ASCII 1)
SOH = "\x01"


@dataclass
class FixField:
    """Represents a single FIX field (tag=value)."""

    tag: int
    value: str

    @property
    def name(self) -> str:
        """Get the field name."""
        return get_tag_name(self.tag)

    @property
    def int_value(self) -> Optional[int]:
        """Get value as integer."""
        try:
            return int(self.value)
        except (ValueError, TypeError):
            return None

    @property
    def float_value(self) -> Optional[float]:
        """Get value as float."""
        try:
            return float(self.value)
        except (ValueError, TypeError):
            return None

    @property
    def decimal_value(self) -> Optional[Decimal]:
        """Get value as Decimal."""
        try:
            return Decimal(self.value)
        except Exception:
            return None

    @property
    def datetime_value(self) -> Optional[datetime]:
        """Get value as datetime (assumes FIX timestamp format)."""
        try:
            # FIX timestamp: YYYYMMDD-HH:MM:SS or YYYYMMDD-HH:MM:SS.sss
            if len(self.value) >= 17:
                if "." in self.value:
                    return datetime.strptime(self.value, "%Y%m%d-%H:%M:%S.%f")
                return datetime.strptime(self.value, "%Y%m%d-%H:%M:%S")
            elif len(self.value) == 8:
                return datetime.strptime(self.value, "%Y%m%d")
        except ValueError:
            pass
        return None

    def to_fix(self) -> str:
        """Convert to FIX format (tag=value)."""
        return f"{self.tag}={self.value}"

    def __str__(self) -> str:
        return f"{self.name}({self.tag})={self.value}"


@dataclass
class FixGroup:
    """
    Represents a repeating group in FIX.

    A group is defined by a count field followed by repeating
    instances of a set of fields.
    """

    count_tag: int  # Tag for the count field (e.g., NoOrders=73)
    entries: List[Dict[int, str]] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Get the number of entries."""
        return len(self.entries)

    def add_entry(self, fields: Dict[int, str]) -> None:
        """Add an entry to the group."""
        self.entries.append(fields)

    def get_entry(self, index: int) -> Optional[Dict[int, str]]:
        """Get entry by index."""
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def to_fields(self) -> List[FixField]:
        """Convert group to list of FixFields."""
        fields = [FixField(tag=self.count_tag, value=str(self.count))]

        for entry in self.entries:
            for tag, value in entry.items():
                fields.append(FixField(tag=tag, value=value))

        return fields


@dataclass
class FixHeader:
    """FIX message header (standard header fields)."""

    begin_string: str = "FIX.4.4"  # Tag 8
    body_length: int = 0  # Tag 9 (calculated)
    msg_type: str = ""  # Tag 35
    sender_comp_id: str = ""  # Tag 49
    target_comp_id: str = ""  # Tag 56
    msg_seq_num: int = 0  # Tag 34
    sending_time: Optional[datetime] = None  # Tag 52
    poss_dup_flag: bool = False  # Tag 43
    poss_resend: bool = False  # Tag 97
    orig_sending_time: Optional[datetime] = None  # Tag 122

    # Optional header fields
    on_behalf_of_comp_id: Optional[str] = None  # Tag 115
    on_behalf_of_sub_id: Optional[str] = None  # Tag 116
    deliver_to_comp_id: Optional[str] = None  # Tag 128
    deliver_to_sub_id: Optional[str] = None  # Tag 129
    sender_sub_id: Optional[str] = None  # Tag 50
    sender_location_id: Optional[str] = None  # Tag 142
    target_sub_id: Optional[str] = None  # Tag 57
    target_location_id: Optional[str] = None  # Tag 143

    @property
    def version(self) -> Optional[FixVersion]:
        """Get FIX version."""
        return FixVersion.from_string(self.begin_string)

    @property
    def message_type(self) -> Optional[FixMsgType]:
        """Get message type enum."""
        return FixMsgType.from_code(self.msg_type)

    def to_fields(self) -> List[FixField]:
        """Convert header to list of FixFields."""
        fields = [
            FixField(tag=8, value=self.begin_string),
            FixField(tag=9, value=str(self.body_length)),
            FixField(tag=35, value=self.msg_type),
            FixField(tag=49, value=self.sender_comp_id),
            FixField(tag=56, value=self.target_comp_id),
            FixField(tag=34, value=str(self.msg_seq_num)),
        ]

        if self.sending_time:
            fields.append(
                FixField(tag=52, value=self.sending_time.strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
            )

        if self.poss_dup_flag:
            fields.append(FixField(tag=43, value="Y"))

        if self.poss_resend:
            fields.append(FixField(tag=97, value="Y"))

        if self.orig_sending_time:
            fields.append(
                FixField(tag=122, value=self.orig_sending_time.strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
            )

        if self.on_behalf_of_comp_id:
            fields.append(FixField(tag=115, value=self.on_behalf_of_comp_id))

        if self.on_behalf_of_sub_id:
            fields.append(FixField(tag=116, value=self.on_behalf_of_sub_id))

        if self.deliver_to_comp_id:
            fields.append(FixField(tag=128, value=self.deliver_to_comp_id))

        if self.deliver_to_sub_id:
            fields.append(FixField(tag=129, value=self.deliver_to_sub_id))

        if self.sender_sub_id:
            fields.append(FixField(tag=50, value=self.sender_sub_id))

        if self.sender_location_id:
            fields.append(FixField(tag=142, value=self.sender_location_id))

        if self.target_sub_id:
            fields.append(FixField(tag=57, value=self.target_sub_id))

        if self.target_location_id:
            fields.append(FixField(tag=143, value=self.target_location_id))

        return fields


@dataclass
class FixTrailer:
    """FIX message trailer (standard trailer fields)."""

    checksum: str = ""  # Tag 10 (calculated)

    def to_fields(self) -> List[FixField]:
        """Convert trailer to list of FixFields."""
        return [FixField(tag=10, value=self.checksum)]


@dataclass
class FixMessage:
    """
    Complete FIX message with header, body, and trailer.
    """

    header: FixHeader = field(default_factory=FixHeader)
    body: List[FixField] = field(default_factory=list)
    trailer: FixTrailer = field(default_factory=FixTrailer)
    groups: Dict[int, FixGroup] = field(default_factory=dict)

    # Raw message and metadata
    raw_message: str = ""
    parse_errors: List[str] = field(default_factory=list)

    @property
    def msg_type(self) -> str:
        """Get message type code."""
        return self.header.msg_type

    @property
    def message_type(self) -> Optional[FixMsgType]:
        """Get message type enum."""
        return self.header.message_type

    @property
    def version(self) -> Optional[FixVersion]:
        """Get FIX version."""
        return self.header.version

    def get_field(self, tag: int) -> Optional[FixField]:
        """Get first field by tag."""
        for f in self.body:
            if f.tag == tag:
                return f
        return None

    def get_fields(self, tag: int) -> List[FixField]:
        """Get all fields matching tag."""
        return [f for f in self.body if f.tag == tag]

    def get_value(self, tag: int) -> Optional[str]:
        """Get value of first field matching tag."""
        f = self.get_field(tag)
        return f.value if f else None

    def get_int(self, tag: int) -> Optional[int]:
        """Get integer value of field."""
        f = self.get_field(tag)
        return f.int_value if f else None

    def get_float(self, tag: int) -> Optional[float]:
        """Get float value of field."""
        f = self.get_field(tag)
        return f.float_value if f else None

    def get_decimal(self, tag: int) -> Optional[Decimal]:
        """Get Decimal value of field."""
        f = self.get_field(tag)
        return f.decimal_value if f else None

    def get_datetime(self, tag: int) -> Optional[datetime]:
        """Get datetime value of field."""
        f = self.get_field(tag)
        return f.datetime_value if f else None

    def set_field(self, tag: int, value: Any) -> None:
        """Set a field value, replacing existing if present."""
        str_value = str(value)

        # Check if field exists
        for i, f in enumerate(self.body):
            if f.tag == tag:
                self.body[i] = FixField(tag=tag, value=str_value)
                return

        # Add new field
        self.body.append(FixField(tag=tag, value=str_value))

    def add_field(self, tag: int, value: Any) -> None:
        """Add a field (allows duplicates)."""
        self.body.append(FixField(tag=tag, value=str(value)))

    def remove_field(self, tag: int) -> bool:
        """Remove first field with tag. Returns True if removed."""
        for i, f in enumerate(self.body):
            if f.tag == tag:
                del self.body[i]
                return True
        return False

    def add_group(self, group: FixGroup) -> None:
        """Add a repeating group."""
        self.groups[group.count_tag] = group

    def get_group(self, count_tag: int) -> Optional[FixGroup]:
        """Get a repeating group by count tag."""
        return self.groups.get(count_tag)

    def calculate_body_length(self) -> int:
        """Calculate body length (tag 9)."""
        body_str = self._build_body_string()
        return len(body_str)

    def calculate_checksum(self) -> str:
        """Calculate checksum (tag 10)."""
        # Build message without checksum
        header_fields = self.header.to_fields()
        # Update body length
        self.header.body_length = self.calculate_body_length()
        header_fields[1] = FixField(tag=9, value=str(self.header.body_length))

        msg_str = ""
        for f in header_fields:
            msg_str += f.to_fix() + SOH

        msg_str += self._build_body_string()

        # Calculate checksum
        total = sum(ord(c) for c in msg_str) % 256
        return f"{total:03d}"

    def _build_body_string(self) -> str:
        """Build the body portion of the message."""
        body_str = ""

        for f in self.body:
            body_str += f.to_fix() + SOH

        # Add groups
        for group in self.groups.values():
            for gf in group.to_fields():
                body_str += gf.to_fix() + SOH

        return body_str

    def to_fix(self, delimiter: str = SOH) -> str:
        """
        Convert to FIX message string.

        Args:
            delimiter: Field delimiter (default is SOH)

        Returns:
            Complete FIX message string
        """
        # Calculate body length and checksum
        self.header.body_length = self.calculate_body_length()
        self.trailer.checksum = self.calculate_checksum()

        # Build message
        parts = []

        # Header
        for f in self.header.to_fields():
            parts.append(f.to_fix())

        # Body
        for f in self.body:
            parts.append(f.to_fix())

        # Groups
        for group in self.groups.values():
            for gf in group.to_fields():
                parts.append(gf.to_fix())

        # Trailer
        for f in self.trailer.to_fields():
            parts.append(f.to_fix())

        return delimiter.join(parts) + delimiter

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "msg_type": self.msg_type,
            "msg_type_name": self.message_type.description if self.message_type else None,
            "version": self.header.begin_string,
            "sender": self.header.sender_comp_id,
            "target": self.header.target_comp_id,
            "seq_num": self.header.msg_seq_num,
            "sending_time": (
                self.header.sending_time.isoformat() if self.header.sending_time else None
            ),
            "fields": {},
            "groups": {},
        }

        for f in self.body:
            name = f.name
            if name in result["fields"]:
                if isinstance(result["fields"][name], list):
                    result["fields"][name].append(f.value)
                else:
                    result["fields"][name] = [result["fields"][name], f.value]
            else:
                result["fields"][name] = f.value

        for count_tag, group in self.groups.items():
            result["groups"][get_tag_name(count_tag)] = {
                "count": group.count,
                "entries": group.entries,
            }

        return result

    def to_human_readable(self) -> str:
        """Convert to human-readable format."""
        lines = [
            f"FIX Message: {self.message_type.description if self.message_type else self.msg_type}",
            f"  Version: {self.header.begin_string}",
            f"  Sender: {self.header.sender_comp_id}",
            f"  Target: {self.header.target_comp_id}",
            f"  SeqNum: {self.header.msg_seq_num}",
            "",
            "Fields:",
        ]

        for f in self.body:
            lines.append(f"  {f.name}({f.tag}) = {f.value}")

        if self.groups:
            lines.append("")
            lines.append("Groups:")
            for count_tag, group in self.groups.items():
                lines.append(f"  {get_tag_name(count_tag)} ({group.count} entries):")
                for i, entry in enumerate(group.entries):
                    lines.append(f"    Entry {i + 1}:")
                    for tag, value in entry.items():
                        lines.append(f"      {get_tag_name(tag)}({tag}) = {value}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return f"FixMessage({self.msg_type}, {self.header.sender_comp_id}->{self.header.target_comp_id})"
