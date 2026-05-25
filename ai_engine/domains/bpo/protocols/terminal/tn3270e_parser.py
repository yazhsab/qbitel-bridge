"""
TN3270e Protocol Parser

Parses TN3270e terminal protocol messages including:
- TN3270e header extraction (5-byte envelope)
- 3270 data stream interpretation (structured fields, orders, attributes)
- AID byte identification (Enter, PF keys, PA keys)
- Telnet negotiation byte handling (IAC sequences)
- Screen buffer construction from data stream
- Sensitive data pattern detection in screen content
"""

import re
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ai_engine.domains.bpo.protocols.terminal.tn3270e_message import (
    TN3270eMessage,
    TN3270eHeader,
    TN3270eDataType,
    ScreenBuffer,
    ScreenField,
    AIDCode,
    OrderCode,
    WriteCommand,
    FieldAttribute,
    _ebcdic_to_ascii,
)


class TN3270eParseError(Exception):
    """Exception raised when TN3270e parsing fails."""

    def __init__(self, message: str, position: int = 0, raw_data: bytes = b""):
        super().__init__(message)
        self.position = position
        self.raw_data = raw_data


# Telnet protocol constants
IAC = 0xFF    # Interpret As Command
WILL = 0xFB   # Will option
WONT = 0xFC   # Won't option
DO = 0xFD     # Do option
DONT = 0xFE   # Don't option
SB = 0xFA     # Subnegotiation Begin
SE = 0xF0     # Subnegotiation End
EOR = 0xEF    # End of Record

# TN3270e telnet options
TN3270E = 0x28  # TN3270E option (RFC 2355)
BINARY = 0x00   # Binary transmission
EOR_OPT = 0x19  # End of Record option


@dataclass
class TelnetNegotiation:
    """Represents a telnet negotiation sequence."""

    command: int     # WILL, WONT, DO, DONT
    option: int      # The option being negotiated
    subneg_data: bytes = b""  # Subnegotiation data if applicable

    @property
    def command_name(self) -> str:
        """Get human-readable command name."""
        names = {WILL: "WILL", WONT: "WONT", DO: "DO", DONT: "DONT", SB: "SB"}
        return names.get(self.command, f"0x{self.command:02X}")

    @property
    def option_name(self) -> str:
        """Get human-readable option name."""
        names = {TN3270E: "TN3270E", BINARY: "BINARY", EOR_OPT: "EOR"}
        return names.get(self.option, f"0x{self.option:02X}")

    def __str__(self) -> str:
        return f"IAC {self.command_name} {self.option_name}"


@dataclass
class SensitiveDataMatch:
    """Represents a sensitive data pattern found in screen content."""

    pattern_type: str      # SSN, CREDIT_CARD, ACCOUNT_NUMBER, etc.
    position: int          # Buffer position where match starts
    row: int               # Screen row
    column: int            # Screen column
    length: int            # Length of the match
    masked_value: str      # Masked version of the matched value
    confidence: float      # Confidence score (0.0 to 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pattern_type": self.pattern_type,
            "position": self.position,
            "row": self.row,
            "column": self.column,
            "length": self.length,
            "masked_value": self.masked_value,
            "confidence": self.confidence,
        }


class TN3270eParser:
    """
    Parser for TN3270e terminal protocol data.

    Handles:
    - TN3270e header parsing (5-byte envelope)
    - Telnet negotiation sequences (IAC WILL/WONT/DO/DONT)
    - 3270 data stream orders (SF, SBA, SA, IC, etc.)
    - AID byte identification
    - Screen buffer population from write commands
    - Sensitive data pattern detection
    """

    # Sensitive data detection patterns
    SSN_PATTERN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
    CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
    PHONE_PATTERN = re.compile(r"\b\d{3}[-\s.]?\d{3}[-\s.]?\d{4}\b")
    ACCOUNT_PATTERN = re.compile(r"\b\d{8,17}\b")
    DATE_OF_BIRTH_PATTERN = re.compile(
        r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b"
    )

    # Screen names associated with sensitive data
    SENSITIVE_SCREEN_KEYWORDS = {
        "CUSTOMER", "ACCOUNT", "PAYMENT", "TRANSFER", "BALANCE",
        "SSN", "SOCIAL", "CREDIT", "DEBIT", "TRANSACTION",
        "PERSONAL", "CONFIDENTIAL", "RESTRICTED", "ACCT",
    }

    def __init__(self, strict: bool = True):
        """
        Initialize the TN3270e parser.

        Args:
            strict: If True, raise errors for malformed data
        """
        self.strict = strict

    def parse(self, raw_data: bytes) -> TN3270eMessage:
        """
        Parse a complete TN3270e message from raw bytes.

        Args:
            raw_data: Raw bytes from the wire (including TN3270e header)

        Returns:
            Parsed TN3270eMessage object

        Raises:
            TN3270eParseError: If parsing fails in strict mode
        """
        message = TN3270eMessage()
        message.raw_bytes = raw_data

        if len(raw_data) < 5:
            error_msg = (
                f"Data too short for TN3270e header: {len(raw_data)} bytes"
            )
            message.parse_errors.append(error_msg)
            if self.strict:
                raise TN3270eParseError(error_msg, 0, raw_data)
            return message

        # Parse header
        try:
            message.header = TN3270eHeader.from_bytes(raw_data[:5])
        except (ValueError, struct.error) as e:
            error_msg = f"Failed to parse TN3270e header: {e}"
            message.parse_errors.append(error_msg)
            if self.strict:
                raise TN3270eParseError(error_msg, 0, raw_data)
            return message

        message.data = raw_data[5:]
        return message

    def parse_telnet_stream(self, raw_data: bytes) -> Tuple[List[TelnetNegotiation], bytes]:
        """
        Separate telnet negotiation sequences from data.

        Scans through the raw byte stream, extracts IAC sequences,
        and returns the remaining clean data payload.

        Args:
            raw_data: Raw bytes potentially containing IAC sequences

        Returns:
            Tuple of (list of negotiations, cleaned data bytes)
        """
        negotiations: List[TelnetNegotiation] = []
        clean_data = bytearray()
        pos = 0

        while pos < len(raw_data):
            byte = raw_data[pos]

            if byte == IAC:
                if pos + 1 >= len(raw_data):
                    break

                next_byte = raw_data[pos + 1]

                if next_byte == IAC:
                    # Escaped IAC (literal 0xFF in data)
                    clean_data.append(IAC)
                    pos += 2
                elif next_byte in (WILL, WONT, DO, DONT):
                    # Three-byte negotiation: IAC WILL/WONT/DO/DONT option
                    if pos + 2 < len(raw_data):
                        option = raw_data[pos + 2]
                        negotiations.append(
                            TelnetNegotiation(command=next_byte, option=option)
                        )
                        pos += 3
                    else:
                        break
                elif next_byte == SB:
                    # Subnegotiation: IAC SB option ... IAC SE
                    sb_start = pos + 2
                    sb_end = self._find_subneg_end(raw_data, sb_start)
                    if sb_end >= 0 and sb_start < len(raw_data):
                        option = raw_data[sb_start]
                        subneg_data = bytes(raw_data[sb_start + 1 : sb_end])
                        negotiations.append(
                            TelnetNegotiation(
                                command=SB,
                                option=option,
                                subneg_data=subneg_data,
                            )
                        )
                        pos = sb_end + 2  # Skip past IAC SE
                    else:
                        break
                elif next_byte == EOR:
                    # End of Record marker
                    pos += 2
                else:
                    # Unknown IAC command, skip
                    pos += 2
            else:
                clean_data.append(byte)
                pos += 1

        return negotiations, bytes(clean_data)

    def _find_subneg_end(self, data: bytes, start: int) -> int:
        """Find the end of a telnet subnegotiation (IAC SE)."""
        pos = start
        while pos < len(data) - 1:
            if data[pos] == IAC and data[pos + 1] == SE:
                return pos
            pos += 1
        return -1

    def parse_3270_data_stream(
        self, data: bytes, screen: Optional[ScreenBuffer] = None
    ) -> ScreenBuffer:
        """
        Parse a 3270 data stream and populate a screen buffer.

        Interprets write commands, orders (SF, SBA, IC, etc.), and
        character data to build the current screen state.

        Args:
            data: 3270 data stream bytes (after TN3270e header)
            screen: Existing screen buffer to update, or None to create new

        Returns:
            Updated ScreenBuffer with parsed field and character data
        """
        if screen is None:
            screen = ScreenBuffer()

        if not data:
            return screen

        pos = 0

        # First byte is the write command (or WCC)
        if pos < len(data):
            write_cmd = data[pos]
            pos += 1

            if write_cmd in (
                WriteCommand.ERASE_WRITE,
                WriteCommand.ERASE_WRITE_ALTERNATE,
            ):
                screen.clear()
                if write_cmd == WriteCommand.ERASE_WRITE_ALTERNATE:
                    # Switch to alternate screen size (Model 5: 32x132)
                    screen = ScreenBuffer(
                        rows=ScreenBuffer.MODEL_5_ROWS,
                        cols=ScreenBuffer.MODEL_5_COLS,
                    )

            # Skip Write Control Character (WCC) if present
            if pos < len(data):
                _wcc = data[pos]
                pos += 1

        # Parse orders and data
        current_pos = screen.cursor_position
        current_field_start = -1
        current_attr = 0

        while pos < len(data):
            byte = data[pos]

            if byte == OrderCode.START_FIELD:
                # Start Field order: SF + attribute byte
                pos += 1
                if pos < len(data):
                    attr = data[pos]
                    pos += 1

                    # Save the current field if one was in progress
                    if current_field_start >= 0:
                        self._finalize_field(
                            screen, current_field_start, current_attr, current_pos
                        )

                    # Set attribute at current position
                    screen.set_attribute(current_pos, attr)
                    current_field_start = current_pos
                    current_attr = attr
                    current_pos = (current_pos + 1) % screen.size

            elif byte == OrderCode.START_FIELD_EXTENDED:
                # Start Field Extended: SFE + count + pairs
                pos += 1
                if pos < len(data):
                    pair_count = data[pos]
                    pos += 1
                    attr = 0
                    for _ in range(pair_count):
                        if pos + 1 < len(data):
                            attr_type = data[pos]
                            attr_value = data[pos + 1]
                            pos += 2
                            if attr_type == 0xC0:  # Basic field attribute
                                attr = attr_value
                        else:
                            break

                    if current_field_start >= 0:
                        self._finalize_field(
                            screen, current_field_start, current_attr, current_pos
                        )

                    screen.set_attribute(current_pos, attr)
                    current_field_start = current_pos
                    current_attr = attr
                    current_pos = (current_pos + 1) % screen.size

            elif byte == OrderCode.SET_BUFFER_ADDRESS:
                # Set Buffer Address: SBA + 2-byte address
                pos += 1
                if pos + 1 < len(data):
                    addr = self._decode_buffer_address(data[pos], data[pos + 1])
                    pos += 2
                    if 0 <= addr < screen.size:
                        current_pos = addr
                else:
                    break

            elif byte == OrderCode.INSERT_CURSOR:
                # Insert Cursor: set cursor position
                pos += 1
                screen.cursor_position = current_pos

            elif byte == OrderCode.PROGRAM_TAB:
                # Program Tab: advance to next unprotected field
                pos += 1

            elif byte == OrderCode.REPEAT_TO_ADDRESS:
                # Repeat to Address: RA + 2-byte address + char
                pos += 1
                if pos + 2 < len(data):
                    target_addr = self._decode_buffer_address(
                        data[pos], data[pos + 1]
                    )
                    repeat_char = data[pos + 2]
                    pos += 3
                    if 0 <= target_addr < screen.size:
                        while current_pos != target_addr:
                            screen.set_char(current_pos, repeat_char)
                            current_pos = (current_pos + 1) % screen.size
                else:
                    break

            elif byte == OrderCode.ERASE_UNPROTECTED:
                # Erase Unprotected to Address: EUA + 2-byte address
                pos += 1
                if pos + 1 < len(data):
                    target_addr = self._decode_buffer_address(
                        data[pos], data[pos + 1]
                    )
                    pos += 2
                    if 0 <= target_addr < screen.size:
                        while current_pos != target_addr:
                            fld = screen.find_field_at(current_pos)
                            if fld and not fld.is_protected:
                                screen.set_char(current_pos, 0x00)
                            current_pos = (current_pos + 1) % screen.size
                else:
                    break

            elif byte == OrderCode.SET_ATTRIBUTE:
                # Set Attribute: SA + type + value
                pos += 1
                if pos + 1 < len(data):
                    _attr_type = data[pos]
                    _attr_value = data[pos + 1]
                    pos += 2
                else:
                    break

            elif byte == OrderCode.MODIFY_FIELD:
                # Modify Field: MF + count + pairs
                pos += 1
                if pos < len(data):
                    pair_count = data[pos]
                    pos += 1
                    for _ in range(pair_count):
                        if pos + 1 < len(data):
                            pos += 2
                        else:
                            break

            elif byte == OrderCode.GRAPHIC_ESCAPE:
                # Graphic Escape: GE + character
                pos += 1
                if pos < len(data):
                    screen.set_char(current_pos, data[pos])
                    pos += 1
                    current_pos = (current_pos + 1) % screen.size

            else:
                # Regular display character
                screen.set_char(current_pos, byte)
                current_pos = (current_pos + 1) % screen.size
                pos += 1

        # Finalize last field
        if current_field_start >= 0:
            self._finalize_field(
                screen, current_field_start, current_attr, current_pos
            )

        # Populate field content strings
        self._populate_field_content(screen)

        return screen

    def _finalize_field(
        self,
        screen: ScreenBuffer,
        field_start: int,
        attribute: int,
        current_pos: int,
    ) -> None:
        """Create a ScreenField and add it to the screen buffer."""
        # Calculate field length (data area starts one position after attribute)
        data_start = (field_start + 1) % screen.size
        if current_pos >= data_start:
            length = current_pos - data_start
        else:
            length = screen.size - data_start + current_pos

        screen_field = ScreenField(
            position=field_start,
            length=length,
            attribute_byte=attribute,
        )
        screen.fields.append(screen_field)

    def _populate_field_content(self, screen: ScreenBuffer) -> None:
        """Extract text content for each field from the screen buffer."""
        for fld in screen.fields:
            data_start = (fld.position + 1) % screen.size
            fld.content = screen.get_text(data_start, fld.length)

    def _decode_buffer_address(self, byte1: int, byte2: int) -> int:
        """
        Decode a 2-byte 3270 buffer address.

        The address can be in 14-bit or 12-bit encoding depending
        on the high bits of byte1.
        """
        if byte1 & 0xC0 == 0x00:
            # 14-bit binary address
            return ((byte1 & 0x3F) << 8) | byte2
        else:
            # 12-bit encoded address
            return ((byte1 & 0x3F) << 6) | (byte2 & 0x3F)

    def parse_aid(self, data: bytes) -> Optional[AIDCode]:
        """
        Parse the AID (Attention Identifier) byte from inbound data.

        The AID byte is the first byte of data sent from the terminal
        and identifies which key the operator pressed.

        Args:
            data: Inbound 3270 data stream

        Returns:
            AIDCode enum value, or None if data is empty
        """
        if not data:
            return None

        aid_byte = data[0]
        try:
            return AIDCode(aid_byte)
        except ValueError:
            return None

    def parse_inbound_data(
        self, data: bytes, screen: ScreenBuffer
    ) -> Tuple[Optional[AIDCode], int, List[Tuple[int, str]]]:
        """
        Parse inbound data from the terminal (operator keystrokes).

        Inbound format: AID + cursor_address(2) + [SBA + address(2) + data]*

        Args:
            data: Inbound 3270 data stream bytes
            screen: Current screen buffer for context

        Returns:
            Tuple of (AID code, cursor position, list of (address, value) pairs)
        """
        if len(data) < 3:
            return None, 0, []

        aid = self.parse_aid(data)
        cursor_pos = self._decode_buffer_address(data[1], data[2])
        field_data: List[Tuple[int, str]] = []

        pos = 3
        current_addr = -1
        current_chars: List[str] = []

        while pos < len(data):
            byte = data[pos]

            if byte == OrderCode.SET_BUFFER_ADDRESS:
                # Save accumulated data for previous field
                if current_addr >= 0 and current_chars:
                    field_data.append((current_addr, "".join(current_chars)))

                pos += 1
                if pos + 1 < len(data):
                    current_addr = self._decode_buffer_address(
                        data[pos], data[pos + 1]
                    )
                    pos += 2
                    current_chars = []
                else:
                    break
            else:
                current_chars.append(_ebcdic_to_ascii(byte))
                pos += 1

        # Save last field data
        if current_addr >= 0 and current_chars:
            field_data.append((current_addr, "".join(current_chars)))

        return aid, cursor_pos, field_data

    def detect_sensitive_data(
        self, screen: ScreenBuffer
    ) -> List[SensitiveDataMatch]:
        """
        Scan the screen buffer for sensitive data patterns.

        Detects Social Security Numbers, credit card numbers, phone numbers,
        account numbers, and dates of birth in the screen content.

        Args:
            screen: Screen buffer to scan

        Returns:
            List of SensitiveDataMatch objects for each detection
        """
        matches: List[SensitiveDataMatch] = []
        full_text = screen.get_all_text()

        # Scan each row individually for pattern context
        for row_idx in range(screen.rows):
            row_text = screen.get_row_text(row_idx)

            # SSN detection
            for match in self.SSN_PATTERN.finditer(row_text):
                raw_digits = re.sub(r"[^\d]", "", match.group())
                if len(raw_digits) == 9 and not raw_digits.startswith("000"):
                    matches.append(SensitiveDataMatch(
                        pattern_type="SSN",
                        position=row_idx * screen.cols + match.start(),
                        row=row_idx,
                        column=match.start(),
                        length=len(match.group()),
                        masked_value=f"***-**-{raw_digits[-4:]}",
                        confidence=0.85,
                    ))

            # Credit card detection
            for match in self.CREDIT_CARD_PATTERN.finditer(row_text):
                raw_digits = re.sub(r"[^\d]", "", match.group())
                if len(raw_digits) in (15, 16) and self._luhn_check(raw_digits):
                    matches.append(SensitiveDataMatch(
                        pattern_type="CREDIT_CARD",
                        position=row_idx * screen.cols + match.start(),
                        row=row_idx,
                        column=match.start(),
                        length=len(match.group()),
                        masked_value=f"****-****-****-{raw_digits[-4:]}",
                        confidence=0.90,
                    ))

            # Phone number detection
            for match in self.PHONE_PATTERN.finditer(row_text):
                raw_digits = re.sub(r"[^\d]", "", match.group())
                if len(raw_digits) == 10:
                    matches.append(SensitiveDataMatch(
                        pattern_type="PHONE",
                        position=row_idx * screen.cols + match.start(),
                        row=row_idx,
                        column=match.start(),
                        length=len(match.group()),
                        masked_value=f"(***) ***-{raw_digits[-4:]}",
                        confidence=0.70,
                    ))

            # Date of birth detection
            for match in self.DATE_OF_BIRTH_PATTERN.finditer(row_text):
                # Check if preceded by DOB-related label
                prefix = row_text[max(0, match.start() - 15) : match.start()].upper()
                if any(kw in prefix for kw in ("DOB", "BIRTH", "BORN", "D.O.B")):
                    matches.append(SensitiveDataMatch(
                        pattern_type="DATE_OF_BIRTH",
                        position=row_idx * screen.cols + match.start(),
                        row=row_idx,
                        column=match.start(),
                        length=len(match.group()),
                        masked_value="**/**/****",
                        confidence=0.80,
                    ))

        # Account number detection in labeled fields
        for fld in screen.fields:
            if fld.is_protected and fld.content.strip():
                upper_content = fld.content.strip().upper()
                if any(kw in upper_content for kw in ("ACCT", "ACCOUNT", "ROUTING")):
                    # Check the next input field for account numbers
                    fld_idx = screen.fields.index(fld)
                    if fld_idx + 1 < len(screen.fields):
                        next_field = screen.fields[fld_idx + 1]
                        if not next_field.is_protected:
                            acct_match = self.ACCOUNT_PATTERN.search(
                                next_field.content.strip()
                            )
                            if acct_match:
                                raw = acct_match.group()
                                matches.append(SensitiveDataMatch(
                                    pattern_type="ACCOUNT_NUMBER",
                                    position=next_field.position,
                                    row=next_field.row,
                                    column=next_field.column,
                                    length=len(raw),
                                    masked_value=f"****{raw[-4:]}",
                                    confidence=0.75,
                                ))

        return matches

    def detect_sensitive_screen(self, screen: ScreenBuffer) -> bool:
        """
        Check if the current screen appears to contain sensitive data.

        Looks for keywords in protected (label) fields that indicate
        the screen is displaying customer, financial, or personal data.

        Args:
            screen: Screen buffer to analyze

        Returns:
            True if the screen contains sensitive data indicators
        """
        for fld in screen.fields:
            if fld.is_protected and fld.content.strip():
                upper_content = fld.content.strip().upper()
                for keyword in self.SENSITIVE_SCREEN_KEYWORDS:
                    if keyword in upper_content:
                        return True
        return False

    def extract_field_map(
        self, screen: ScreenBuffer
    ) -> Dict[str, ScreenField]:
        """
        Build a label-to-field map for the screen.

        For each protected (label) field, finds the adjacent unprotected
        (input) field and maps the label text to the input field.

        Args:
            screen: Screen buffer to analyze

        Returns:
            Dictionary mapping label text to the corresponding input field
        """
        field_map: Dict[str, ScreenField] = {}

        for idx, fld in enumerate(screen.fields):
            if fld.is_protected and fld.content.strip():
                label = fld.content.strip()
                # Look for the next unprotected field
                for next_idx in range(idx + 1, len(screen.fields)):
                    next_fld = screen.fields[next_idx]
                    if not next_fld.is_protected:
                        field_map[label] = next_fld
                        break
                    # Stop if we hit another label on a different row
                    if next_fld.row != fld.row and next_fld.is_protected:
                        break

        return field_map

    def _luhn_check(self, number: str) -> bool:
        """
        Validate a number using the Luhn algorithm.

        Used for credit card number verification.

        Args:
            number: Numeric string to validate

        Returns:
            True if the number passes the Luhn check
        """
        if not number.isdigit():
            return False

        digits = [int(d) for d in number]
        digits.reverse()

        total = 0
        for i, digit in enumerate(digits):
            if i % 2 == 1:
                doubled = digit * 2
                if doubled > 9:
                    doubled -= 9
                total += doubled
            else:
                total += digit

        return total % 10 == 0


def parse_tn3270e_message(raw_data: bytes, strict: bool = True) -> TN3270eMessage:
    """
    Convenience function to parse a TN3270e message.

    Args:
        raw_data: Raw bytes from the wire
        strict: If True, raise errors for malformed data

    Returns:
        Parsed TN3270eMessage object
    """
    parser = TN3270eParser(strict=strict)
    return parser.parse(raw_data)
