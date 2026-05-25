"""
TN3270e Terminal Protocol Data Structures

Data structures for TN3270e terminal protocol (RFC 2355) including:
- TN3270e message envelope with header
- 3270 data stream types
- Screen field representation with attribute detection
- Screen buffer for 3270 display emulation
- Security event tracking for suspicious terminal activity
"""

import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional


class TN3270eDataType(IntEnum):
    """
    TN3270e data types per RFC 2355.

    Identifies the type of data carried in the TN3270e message payload.
    """

    DATA_3270 = 0x00       # 3270 data stream
    SCS_DATA = 0x01        # SNA Character Stream data
    RESPONSE = 0x02        # Response to a request
    BIND_IMAGE = 0x03      # BIND image data
    UNBIND = 0x04          # UNBIND notification
    NVT_DATA = 0x05        # Network Virtual Terminal data
    REQUEST = 0x06         # Request message
    SSCP_LU_DATA = 0x07    # SSCP-LU session data


class TN3270eResponseFlag(IntEnum):
    """TN3270e response flags."""

    NO_RESPONSE = 0x00           # No response required
    ERROR_RESPONSE = 0x01        # Error response
    ALWAYS_RESPONSE = 0x02       # Always respond
    POSITIVE_RESPONSE = 0x00     # Positive response
    NEGATIVE_RESPONSE = 0x01     # Negative response


class TN3270eRequestFlag(IntEnum):
    """TN3270e request flags."""

    NO_REQUEST = 0x00   # Not a request
    ERR_COND_CLR = 0x00  # Error condition cleared


class AIDCode(IntEnum):
    """
    Attention Identifier codes for 3270 data streams.

    These identify which key the user pressed to transmit data.
    """

    NO_AID = 0x60
    ENTER = 0x7D
    PF1 = 0xF1
    PF2 = 0xF2
    PF3 = 0xF3
    PF4 = 0xF4
    PF5 = 0xF5
    PF6 = 0xF6
    PF7 = 0xF7
    PF8 = 0xF8
    PF9 = 0xF9
    PF10 = 0x7A
    PF11 = 0x7B
    PF12 = 0x7C
    PF13 = 0xC1
    PF14 = 0xC2
    PF15 = 0xC3
    PF16 = 0xC4
    PF17 = 0xC5
    PF18 = 0xC6
    PF19 = 0xC7
    PF20 = 0xC8
    PF21 = 0xC9
    PF22 = 0x4A
    PF23 = 0x4B
    PF24 = 0x4C
    PA1 = 0x6C
    PA2 = 0x6E
    PA3 = 0x6B
    CLEAR = 0x6D
    SYSREQ = 0xF0
    STRUCTURED_FIELD = 0x88


class FieldAttribute(IntEnum):
    """
    3270 field attribute bits.

    Encoded in the attribute byte that precedes each field on the screen.
    """

    PROTECTED = 0x20         # Field is protected (read-only)
    NUMERIC = 0x10           # Field accepts only numeric input
    DISPLAY_NOT_SELECTOR = 0x0C  # Display/selector pen detect
    INTENSIFIED = 0x08       # Intensified display
    HIDDEN = 0x0C            # Non-display (hidden) field
    MODIFIED = 0x01          # Modified Data Tag (MDT)


class OrderCode(IntEnum):
    """
    3270 data stream order codes.

    Orders control cursor positioning and field formatting.
    """

    START_FIELD = 0x1D           # Start Field (SF)
    START_FIELD_EXTENDED = 0x29  # Start Field Extended (SFE)
    SET_BUFFER_ADDRESS = 0x11   # Set Buffer Address (SBA)
    SET_ATTRIBUTE = 0x28        # Set Attribute (SA)
    MODIFY_FIELD = 0x2C         # Modify Field (MF)
    INSERT_CURSOR = 0x13        # Insert Cursor (IC)
    PROGRAM_TAB = 0x05          # Program Tab (PT)
    REPEAT_TO_ADDRESS = 0x3C    # Repeat to Address (RA)
    ERASE_UNPROTECTED = 0x12    # Erase Unprotected to Address (EUA)
    GRAPHIC_ESCAPE = 0x08       # Graphic Escape (GE)


class WriteCommand(IntEnum):
    """3270 write commands from host."""

    WRITE = 0xF1                     # Write
    ERASE_WRITE = 0xF5               # Erase/Write
    ERASE_WRITE_ALTERNATE = 0x7E     # Erase/Write Alternate
    ERASE_ALL_UNPROTECTED = 0x6F     # Erase All Unprotected
    WRITE_STRUCTURED_FIELD = 0xF3    # Write Structured Field


class SecurityEventType(Enum):
    """Types of security events detected in terminal sessions."""

    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    CREDENTIAL_STUFFING = "credential_stuffing"
    PROTECTED_FIELD_TAMPER = "protected_field_tamper"
    SENSITIVE_SCREEN_ACCESS = "sensitive_screen_access"
    SESSION_ANOMALY = "session_anomaly"
    RAPID_NAVIGATION = "rapid_navigation"
    BULK_DATA_READ = "bulk_data_read"
    UNAUTHORIZED_TRANSACTION = "unauthorized_transaction"


@dataclass
class TN3270eHeader:
    """
    TN3270e message header (5 bytes).

    Format:
        Byte 0: Data type
        Byte 1: Request flag
        Byte 2: Response flag
        Bytes 3-4: Sequence number (big-endian)
    """

    data_type: TN3270eDataType = TN3270eDataType.DATA_3270
    request_flag: int = 0x00
    response_flag: int = 0x00
    sequence_number: int = 0

    def to_bytes(self) -> bytes:
        """Serialize header to 5-byte wire format."""
        return struct.pack(
            "!BBBH",
            self.data_type,
            self.request_flag,
            self.response_flag,
            self.sequence_number,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "TN3270eHeader":
        """Deserialize header from 5-byte wire format."""
        if len(data) < 5:
            raise ValueError(f"TN3270e header requires 5 bytes, got {len(data)}")
        data_type, request_flag, response_flag, seq_num = struct.unpack(
            "!BBBH", data[:5]
        )
        return cls(
            data_type=TN3270eDataType(data_type),
            request_flag=request_flag,
            response_flag=response_flag,
            sequence_number=seq_num,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "data_type": self.data_type.name,
            "request_flag": self.request_flag,
            "response_flag": self.response_flag,
            "sequence_number": self.sequence_number,
        }

    def __str__(self) -> str:
        return (
            f"TN3270eHeader(type={self.data_type.name}, "
            f"seq={self.sequence_number})"
        )


@dataclass
class ScreenField:
    """
    Represents a single field on the 3270 display screen.

    Fields are defined by Start Field (SF) orders in the data stream
    and have attributes controlling their display and input behavior.
    """

    position: int = 0          # Buffer address of the field attribute byte
    length: int = 0            # Length of the field data area
    attribute_byte: int = 0    # Raw attribute byte value
    content: str = ""          # Current field content (EBCDIC decoded)
    field_id: str = ""         # Optional field identifier from the application

    @property
    def is_protected(self) -> bool:
        """Check if the field is protected (read-only)."""
        return bool(self.attribute_byte & FieldAttribute.PROTECTED)

    @property
    def is_hidden(self) -> bool:
        """Check if the field is hidden (non-display)."""
        return (self.attribute_byte & 0x0C) == FieldAttribute.HIDDEN

    @property
    def is_numeric(self) -> bool:
        """Check if the field only accepts numeric input."""
        return bool(self.attribute_byte & FieldAttribute.NUMERIC)

    @property
    def is_modified(self) -> bool:
        """Check if the field has been modified (MDT set)."""
        return bool(self.attribute_byte & FieldAttribute.MODIFIED)

    @property
    def is_intensified(self) -> bool:
        """Check if the field displays with intensified (bright) attribute."""
        display_bits = self.attribute_byte & 0x0C
        return display_bits == FieldAttribute.INTENSIFIED

    @property
    def is_input_field(self) -> bool:
        """Check if this is an unprotected input field."""
        return not self.is_protected

    @property
    def row(self) -> int:
        """Get the row number (0-based) for an 80-column screen."""
        return self.position // 80

    @property
    def column(self) -> int:
        """Get the column number (0-based) for an 80-column screen."""
        return self.position % 80

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "position": self.position,
            "row": self.row,
            "column": self.column,
            "length": self.length,
            "content": self.content if not self.is_hidden else "***HIDDEN***",
            "is_protected": self.is_protected,
            "is_hidden": self.is_hidden,
            "is_numeric": self.is_numeric,
            "is_modified": self.is_modified,
            "is_intensified": self.is_intensified,
            "field_id": self.field_id,
        }

    def __str__(self) -> str:
        prot = "P" if self.is_protected else "U"
        hidden = "H" if self.is_hidden else "V"
        mod = "M" if self.is_modified else "-"
        display_content = "***" if self.is_hidden else self.content[:20]
        return (
            f"ScreenField({self.row},{self.column} "
            f"[{prot}{hidden}{mod}] "
            f"len={self.length} "
            f"'{display_content}')"
        )


class ScreenBuffer:
    """
    Represents the 3270 display buffer.

    The buffer is a linear array of character cells that maps to the
    rectangular display. Standard sizes are 24x80 (Model 2) and
    32x132 (Model 5).
    """

    # Standard 3270 terminal model sizes
    MODEL_2_ROWS = 24
    MODEL_2_COLS = 80
    MODEL_5_ROWS = 32
    MODEL_5_COLS = 132

    def __init__(self, rows: int = 24, cols: int = 80):
        """
        Initialize a screen buffer.

        Args:
            rows: Number of rows (default 24 for Model 2)
            cols: Number of columns (default 80 for Model 2)
        """
        self.rows = rows
        self.cols = cols
        self.size = rows * cols
        self.buffer: bytearray = bytearray(self.size)
        self.attribute_buffer: bytearray = bytearray(self.size)
        self.fields: List[ScreenField] = []
        self.cursor_position: int = 0
        self.aid: int = AIDCode.NO_AID

    def clear(self) -> None:
        """Clear the entire screen buffer."""
        self.buffer = bytearray(self.size)
        self.attribute_buffer = bytearray(self.size)
        self.fields = []
        self.cursor_position = 0

    def get_position(self, row: int, col: int) -> int:
        """Convert row/column to buffer address."""
        if row < 0 or row >= self.rows:
            raise ValueError(f"Row {row} out of range (0-{self.rows - 1})")
        if col < 0 or col >= self.cols:
            raise ValueError(f"Column {col} out of range (0-{self.cols - 1})")
        return row * self.cols + col

    def get_row_col(self, position: int) -> tuple:
        """Convert buffer address to row/column tuple."""
        position = position % self.size
        return position // self.cols, position % self.cols

    def set_char(self, position: int, char: int) -> None:
        """Set a character at the specified buffer position."""
        if 0 <= position < self.size:
            self.buffer[position] = char

    def get_char(self, position: int) -> int:
        """Get the character at the specified buffer position."""
        if 0 <= position < self.size:
            return self.buffer[position]
        return 0x00

    def set_attribute(self, position: int, attribute: int) -> None:
        """Set the field attribute at the specified position."""
        if 0 <= position < self.size:
            self.attribute_buffer[position] = attribute

    def get_text(self, start: int, length: int) -> str:
        """
        Extract text from the buffer as ASCII.

        Converts EBCDIC buffer positions to readable ASCII text.
        Null bytes are treated as spaces.
        """
        result = []
        for i in range(length):
            pos = (start + i) % self.size
            byte_val = self.buffer[pos]
            if byte_val == 0x00:
                result.append(" ")
            elif 0x40 <= byte_val <= 0xFE:
                # EBCDIC to ASCII approximation for common characters
                result.append(_ebcdic_to_ascii(byte_val))
            else:
                result.append(" ")
        return "".join(result)

    def get_row_text(self, row: int) -> str:
        """Get the text content of an entire row."""
        start = row * self.cols
        return self.get_text(start, self.cols)

    def get_all_text(self) -> str:
        """Get the full screen content as a multi-line string."""
        lines = []
        for row in range(self.rows):
            lines.append(self.get_row_text(row))
        return "\n".join(lines)

    def get_input_fields(self) -> List[ScreenField]:
        """Get all unprotected (input) fields."""
        return [f for f in self.fields if f.is_input_field]

    def get_modified_fields(self) -> List[ScreenField]:
        """Get all fields with the Modified Data Tag set."""
        return [f for f in self.fields if f.is_modified]

    def get_hidden_fields(self) -> List[ScreenField]:
        """Get all hidden (non-display) fields, typically password fields."""
        return [f for f in self.fields if f.is_hidden]

    def find_field_at(self, position: int) -> Optional[ScreenField]:
        """Find the field containing the given buffer position."""
        for fld in self.fields:
            field_start = fld.position
            field_end = (field_start + fld.length) % self.size
            if field_start <= field_end:
                if field_start <= position < field_end:
                    return fld
            else:
                # Field wraps around the buffer
                if position >= field_start or position < field_end:
                    return fld
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert buffer state to dictionary representation."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "cursor_position": self.cursor_position,
            "cursor_row_col": self.get_row_col(self.cursor_position),
            "aid": AIDCode(self.aid).name if self.aid in [e.value for e in AIDCode] else hex(self.aid),
            "field_count": len(self.fields),
            "input_field_count": len(self.get_input_fields()),
            "modified_field_count": len(self.get_modified_fields()),
            "fields": [f.to_dict() for f in self.fields],
        }

    def __str__(self) -> str:
        return (
            f"ScreenBuffer({self.rows}x{self.cols}, "
            f"fields={len(self.fields)}, "
            f"cursor={self.cursor_position})"
        )


@dataclass
class SecurityEvent:
    """
    Represents a security event detected during terminal session monitoring.

    Tracks suspicious terminal activity for audit and alerting purposes.
    """

    event_type: SecurityEventType
    session_id: str = ""
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "WARNING"   # INFO, WARNING, CRITICAL
    description: str = ""
    screen_name: str = ""
    field_position: int = -1
    source_ip: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "description": self.description,
            "screen_name": self.screen_name,
            "field_position": self.field_position,
            "source_ip": self.source_ip,
            "details": self.details,
        }

    def __str__(self) -> str:
        return (
            f"SecurityEvent({self.event_type.value}, "
            f"severity={self.severity}, "
            f"session={self.session_id})"
        )


@dataclass
class TN3270eMessage:
    """
    Complete TN3270e protocol message.

    Combines the TN3270e header with the payload data and provides
    serialization support for wire format transmission.
    """

    header: TN3270eHeader = field(default_factory=TN3270eHeader)
    data: bytes = b""

    # Metadata
    message_type: str = ""       # Application-level message classification
    raw_bytes: bytes = b""       # Original raw bytes before parsing
    parse_errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def data_type(self) -> TN3270eDataType:
        """Get the data type from the header."""
        return self.header.data_type

    @property
    def sequence_number(self) -> int:
        """Get the sequence number from the header."""
        return self.header.sequence_number

    @property
    def is_3270_data(self) -> bool:
        """Check if this message carries 3270 data stream content."""
        return self.header.data_type == TN3270eDataType.DATA_3270

    @property
    def is_response(self) -> bool:
        """Check if this is a response message."""
        return self.header.data_type == TN3270eDataType.RESPONSE

    @property
    def is_bind(self) -> bool:
        """Check if this is a BIND image message."""
        return self.header.data_type == TN3270eDataType.BIND_IMAGE

    @property
    def is_unbind(self) -> bool:
        """Check if this is an UNBIND message."""
        return self.header.data_type == TN3270eDataType.UNBIND

    def to_bytes(self) -> bytes:
        """
        Serialize the complete TN3270e message to wire format.

        Returns:
            Bytes representation: 5-byte header + payload data
        """
        return self.header.to_bytes() + self.data

    @classmethod
    def from_bytes(cls, raw: bytes) -> "TN3270eMessage":
        """
        Deserialize a TN3270e message from wire format.

        Args:
            raw: Raw bytes from the wire (minimum 5 bytes for header)

        Returns:
            Parsed TN3270eMessage object
        """
        msg = cls()
        msg.raw_bytes = raw

        if len(raw) < 5:
            msg.parse_errors.append(
                f"Message too short for TN3270e header: {len(raw)} bytes"
            )
            return msg

        msg.header = TN3270eHeader.from_bytes(raw[:5])
        msg.data = raw[5:]
        return msg

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "header": self.header.to_dict(),
            "data_length": len(self.data),
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "parse_errors": self.parse_errors,
        }

    def __str__(self) -> str:
        return (
            f"TN3270eMessage(type={self.header.data_type.name}, "
            f"seq={self.header.sequence_number}, "
            f"data_len={len(self.data)})"
        )


# EBCDIC to ASCII conversion table for common characters
_EBCDIC_ASCII_MAP: Dict[int, str] = {
    0x40: " ",   # Space
    0x4B: ".",   # Period
    0x4C: "<",   # Less than
    0x4D: "(",   # Left paren
    0x4E: "+",   # Plus
    0x4F: "|",   # Pipe
    0x50: "&",   # Ampersand
    0x5A: "!",   # Exclamation
    0x5B: "$",   # Dollar
    0x5C: "*",   # Asterisk
    0x5D: ")",   # Right paren
    0x5E: ";",   # Semicolon
    0x60: "-",   # Minus/Hyphen
    0x61: "/",   # Slash
    0x6B: ",",   # Comma
    0x6C: "%",   # Percent
    0x6D: "_",   # Underscore
    0x6E: ">",   # Greater than
    0x6F: "?",   # Question mark
    0x7A: ":",   # Colon
    0x7B: "#",   # Hash
    0x7C: "@",   # At sign
    0x7D: "'",   # Apostrophe
    0x7E: "=",   # Equals
    0x7F: '"',   # Double quote
    # Lowercase letters a-i
    0x81: "a", 0x82: "b", 0x83: "c", 0x84: "d", 0x85: "e",
    0x86: "f", 0x87: "g", 0x88: "h", 0x89: "i",
    # Lowercase letters j-r
    0x91: "j", 0x92: "k", 0x93: "l", 0x94: "m", 0x95: "n",
    0x96: "o", 0x97: "p", 0x98: "q", 0x99: "r",
    # Lowercase letters s-z
    0xA2: "s", 0xA3: "t", 0xA4: "u", 0xA5: "v", 0xA6: "w",
    0xA7: "x", 0xA8: "y", 0xA9: "z",
    # Uppercase letters A-I
    0xC1: "A", 0xC2: "B", 0xC3: "C", 0xC4: "D", 0xC5: "E",
    0xC6: "F", 0xC7: "G", 0xC8: "H", 0xC9: "I",
    # Uppercase letters J-R
    0xD1: "J", 0xD2: "K", 0xD3: "L", 0xD4: "M", 0xD5: "N",
    0xD6: "O", 0xD7: "P", 0xD8: "Q", 0xD9: "R",
    # Uppercase letters S-Z
    0xE2: "S", 0xE3: "T", 0xE4: "U", 0xE5: "V", 0xE6: "W",
    0xE7: "X", 0xE8: "Y", 0xE9: "Z",
    # Digits 0-9
    0xF0: "0", 0xF1: "1", 0xF2: "2", 0xF3: "3", 0xF4: "4",
    0xF5: "5", 0xF6: "6", 0xF7: "7", 0xF8: "8", 0xF9: "9",
}


def _ebcdic_to_ascii(byte_val: int) -> str:
    """Convert a single EBCDIC byte to an ASCII character."""
    return _EBCDIC_ASCII_MAP.get(byte_val, " ")


def _ascii_to_ebcdic(char: str) -> int:
    """Convert a single ASCII character to an EBCDIC byte."""
    for ebcdic_val, ascii_char in _EBCDIC_ASCII_MAP.items():
        if ascii_char == char:
            return ebcdic_val
    return 0x40  # Default to space
