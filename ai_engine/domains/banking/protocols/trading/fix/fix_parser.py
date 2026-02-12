"""
FIX Message Parser

Parses FIX messages from raw text format into structured objects.
"""

import re
from typing import List, Optional, Tuple, Dict
from datetime import datetime

from ai_engine.domains.banking.protocols.trading.fix.fix_message import (
    FixMessage,
    FixHeader,
    FixTrailer,
    FixField,
    FixGroup,
    SOH,
)
from ai_engine.domains.banking.protocols.trading.fix.fix_codes import (
    FixVersion,
    FixMsgType,
)


class FixParseError(Exception):
    """Exception raised when parsing fails."""

    def __init__(self, message: str, position: int = 0, raw_content: str = ""):
        super().__init__(message)
        self.position = position
        self.raw_content = raw_content


# Tags that define repeating groups
GROUP_TAGS: Dict[int, List[int]] = {
    # NoOrders group
    73: [11, 37, 198, 526, 66, 756, 38, 799],
    # NoAllocs group
    78: [79, 661, 736, 467, 776, 539, 80, 993, 366, 467],
    # NoPartyIDs group
    453: [448, 447, 452, 802],
    # NoPartySubIDs group
    802: [523, 803],
    # NoMDEntryTypes group
    267: [269],
    # NoMDEntries group
    268: [
        269,
        270,
        271,
        272,
        273,
        274,
        275,
        276,
        277,
        278,
        279,
        280,
        281,
        282,
        283,
        284,
        285,
        286,
        287,
        288,
        289,
        290,
        291,
        292,
    ],
    # NoRelatedSym group
    146: [
        55,
        65,
        48,
        22,
        167,
        461,
        460,
        541,
        200,
        201,
        202,
        206,
        231,
        223,
        207,
        106,
        348,
        349,
        107,
        350,
        351,
        691,
        667,
        875,
        876,
        873,
        874,
    ],
    # NoSides group
    552: [
        54,
        37,
        198,
        11,
        526,
        66,
        756,
        453,
        229,
        1,
        660,
        581,
        589,
        590,
        591,
        70,
        78,
        79,
        80,
        467,
        81,
        575,
        576,
        577,
        578,
        579,
        376,
        377,
        528,
        529,
        582,
        583,
        336,
        625,
    ],
    # NoLegs group
    555: [
        600,
        601,
        602,
        603,
        604,
        605,
        606,
        607,
        608,
        609,
        610,
        611,
        612,
        613,
        614,
        615,
        616,
        617,
        618,
        619,
        620,
        621,
        622,
        623,
        624,
        556,
        740,
        739,
        955,
        956,
        687,
        690,
    ],
    # NoUnderlyings group
    711: [
        311,
        312,
        309,
        305,
        457,
        462,
        463,
        310,
        763,
        313,
        542,
        241,
        242,
        243,
        244,
        245,
        246,
        256,
        595,
        592,
        593,
        594,
        247,
        316,
        941,
        317,
        436,
        435,
        308,
        306,
        362,
        363,
        364,
        365,
        877,
        878,
        879,
        318,
        879,
        810,
        882,
        883,
        884,
        885,
        886,
    ],
    # NoPositions group
    702: [703, 704, 705, 706],
    # NoSecurityAltID group
    454: [455, 456],
}


class FixParser:
    """
    Parser for FIX messages.

    Handles:
    - Field extraction
    - Header and trailer parsing
    - Repeating group parsing
    - Checksum validation
    """

    def __init__(
        self,
        strict: bool = True,
        validate_checksum: bool = True,
        delimiter: str = SOH,
    ):
        """
        Initialize parser.

        Args:
            strict: If True, raise errors for validation failures
            validate_checksum: If True, validate message checksum
            delimiter: Field delimiter (default is SOH)
        """
        self.strict = strict
        self.validate_checksum = validate_checksum
        self.delimiter = delimiter

    def parse(self, raw_message: str) -> FixMessage:
        """
        Parse a FIX message from raw text.

        Args:
            raw_message: Raw FIX message text

        Returns:
            Parsed FixMessage object

        Raises:
            FixParseError: If parsing fails in strict mode
        """
        message = FixMessage()
        message.raw_message = raw_message

        # Handle different delimiters
        if self.delimiter != SOH and SOH not in raw_message:
            # Convert pipe or other delimiter to SOH for parsing
            raw_message = raw_message.replace(self.delimiter, SOH)

        # Extract fields
        fields = self._extract_fields(raw_message)

        if not fields:
            error = "No fields found in message"
            message.parse_errors.append(error)
            if self.strict:
                raise FixParseError(error, 0, raw_message)
            return message

        # Parse header
        self._parse_header(fields, message)

        # Validate checksum
        if self.validate_checksum:
            self._validate_checksum(raw_message, message)

        # Parse body (non-header, non-trailer fields)
        self._parse_body(fields, message)

        # Parse trailer
        self._parse_trailer(fields, message)

        return message

    def _extract_fields(self, raw_message: str) -> List[Tuple[int, str]]:
        """Extract all tag=value pairs from message."""
        fields = []

        # Split by SOH
        parts = raw_message.split(SOH)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if "=" not in part:
                continue

            try:
                eq_pos = part.index("=")
                tag = int(part[:eq_pos])
                value = part[eq_pos + 1 :]
                fields.append((tag, value))
            except (ValueError, IndexError):
                continue

        return fields

    def _parse_header(self, fields: List[Tuple[int, str]], message: FixMessage) -> None:
        """Parse standard header fields."""
        header_tags = {8, 9, 35, 49, 56, 34, 52, 43, 97, 122, 115, 116, 128, 129, 50, 142, 57, 143}

        for tag, value in fields:
            if tag not in header_tags:
                continue

            if tag == 8:
                message.header.begin_string = value
            elif tag == 9:
                try:
                    message.header.body_length = int(value)
                except ValueError:
                    message.parse_errors.append(f"Invalid body length: {value}")
            elif tag == 35:
                message.header.msg_type = value
            elif tag == 49:
                message.header.sender_comp_id = value
            elif tag == 56:
                message.header.target_comp_id = value
            elif tag == 34:
                try:
                    message.header.msg_seq_num = int(value)
                except ValueError:
                    message.parse_errors.append(f"Invalid sequence number: {value}")
            elif tag == 52:
                message.header.sending_time = self._parse_timestamp(value)
            elif tag == 43:
                message.header.poss_dup_flag = value == "Y"
            elif tag == 97:
                message.header.poss_resend = value == "Y"
            elif tag == 122:
                message.header.orig_sending_time = self._parse_timestamp(value)
            elif tag == 115:
                message.header.on_behalf_of_comp_id = value
            elif tag == 116:
                message.header.on_behalf_of_sub_id = value
            elif tag == 128:
                message.header.deliver_to_comp_id = value
            elif tag == 129:
                message.header.deliver_to_sub_id = value
            elif tag == 50:
                message.header.sender_sub_id = value
            elif tag == 142:
                message.header.sender_location_id = value
            elif tag == 57:
                message.header.target_sub_id = value
            elif tag == 143:
                message.header.target_location_id = value

    def _parse_body(self, fields: List[Tuple[int, str]], message: FixMessage) -> None:
        """Parse body fields and groups."""
        header_tags = {8, 9, 35, 49, 56, 34, 52, 43, 97, 122, 115, 116, 128, 129, 50, 142, 57, 143}
        trailer_tags = {10, 89, 93}

        i = 0
        while i < len(fields):
            tag, value = fields[i]

            # Skip header and trailer tags
            if tag in header_tags or tag in trailer_tags:
                i += 1
                continue

            # Check if this is a group count tag
            if tag in GROUP_TAGS:
                group, consumed = self._parse_group(fields, i, tag)
                if group:
                    message.add_group(group)
                    i += consumed
                    continue

            # Regular field
            message.body.append(FixField(tag=tag, value=value))
            i += 1

    def _parse_group(self, fields: List[Tuple[int, str]], start_index: int, count_tag: int) -> Tuple[Optional[FixGroup], int]:
        """Parse a repeating group."""
        try:
            count = int(fields[start_index][1])
        except (ValueError, IndexError):
            return None, 1

        if count <= 0:
            return None, 1

        group = FixGroup(count_tag=count_tag)
        group_field_tags = set(GROUP_TAGS.get(count_tag, []))

        if not group_field_tags:
            return None, 1

        # Find the first tag of each group instance
        first_tag = GROUP_TAGS[count_tag][0] if GROUP_TAGS[count_tag] else None

        i = start_index + 1
        consumed = 1
        current_entry: Dict[int, str] = {}
        entries_found = 0

        while i < len(fields) and entries_found < count:
            tag, value = fields[i]

            # Check if this tag belongs to the group
            if tag not in group_field_tags:
                # End of group
                break

            # Check if this is the start of a new entry
            if tag == first_tag and current_entry:
                group.entries.append(current_entry)
                entries_found += 1
                current_entry = {}

            current_entry[tag] = value
            i += 1
            consumed += 1

        # Add last entry
        if current_entry:
            group.entries.append(current_entry)
            entries_found += 1

        return group, consumed

    def _parse_trailer(self, fields: List[Tuple[int, str]], message: FixMessage) -> None:
        """Parse trailer fields."""
        for tag, value in fields:
            if tag == 10:
                message.trailer.checksum = value

    def _validate_checksum(self, raw_message: str, message: FixMessage) -> None:
        """Validate message checksum."""
        # Find checksum position
        checksum_pos = raw_message.rfind("10=")
        if checksum_pos == -1:
            message.parse_errors.append("Missing checksum")
            return

        # Calculate checksum of message up to checksum field
        msg_to_check = raw_message[:checksum_pos]

        # Sum all bytes
        total = sum(ord(c) for c in msg_to_check) % 256
        calculated = f"{total:03d}"

        if calculated != message.trailer.checksum:
            error = f"Checksum mismatch: expected {calculated}, got {message.trailer.checksum}"
            message.parse_errors.append(error)
            if self.strict:
                raise FixParseError(error, checksum_pos, raw_message)

    def _parse_timestamp(self, value: str) -> Optional[datetime]:
        """Parse FIX timestamp."""
        try:
            # Try with milliseconds
            if "." in value:
                return datetime.strptime(value, "%Y%m%d-%H:%M:%S.%f")
            # Try without milliseconds
            if len(value) == 17:
                return datetime.strptime(value, "%Y%m%d-%H:%M:%S")
            # Date only
            if len(value) == 8:
                return datetime.strptime(value, "%Y%m%d")
        except ValueError:
            pass
        return None


def parse_fix_message(
    raw_message: str,
    strict: bool = True,
    validate_checksum: bool = True,
    delimiter: str = SOH,
) -> FixMessage:
    """
    Convenience function to parse a FIX message.

    Args:
        raw_message: Raw FIX message text
        strict: If True, raise errors for validation failures
        validate_checksum: If True, validate message checksum
        delimiter: Field delimiter (default is SOH)

    Returns:
        Parsed FixMessage object
    """
    parser = FixParser(
        strict=strict,
        validate_checksum=validate_checksum,
        delimiter=delimiter,
    )
    return parser.parse(raw_message)


def parse_fix_from_pipe_delimited(raw_message: str) -> FixMessage:
    """
    Parse FIX message with pipe delimiter (commonly used for logging).

    Args:
        raw_message: Raw FIX message with | delimiter

    Returns:
        Parsed FixMessage object
    """
    return parse_fix_message(raw_message, delimiter="|", validate_checksum=False)
