"""
SWIFT Message Parser

Parses SWIFT MT messages from raw text format into structured objects.
"""

import re
from typing import Optional, Tuple
from dataclasses import dataclass

from ai_engine.domains.banking.protocols.messaging.swift.swift_message import (
    SwiftMessage,
    SwiftBasicHeader,
    SwiftApplicationHeader,
    SwiftUserHeader,
    SwiftTextBlock,
    SwiftTrailerBlock,
)


class SwiftParseError(Exception):
    """Exception raised when parsing fails."""

    def __init__(self, message: str, position: int = 0, raw_content: str = ""):
        super().__init__(message)
        self.position = position
        self.raw_content = raw_content


@dataclass
class BlockPosition:
    """Position of a block in the message."""

    block_type: str
    start: int
    end: int
    content: str


class SwiftParser:
    """
    Parser for SWIFT MT messages.

    Handles:
    - Block extraction (1-5)
    - Field parsing
    - Character set validation
    - Multi-line field values
    """

    # Block pattern: {n:content}
    BLOCK_PATTERN = re.compile(r"\{(\d):([^}]*(?:\{[^}]*\}[^}]*)*)\}")

    # Field pattern in text block: :TAG:value
    FIELD_PATTERN = re.compile(r":(\d{2}[A-Z]?):(.+?)(?=\n:|$)", re.DOTALL)

    # Alternative field pattern for multi-line
    FIELD_START_PATTERN = re.compile(r"^:(\d{2}[A-Z]?):")

    def __init__(self, strict: bool = True):
        """
        Initialize parser.

        Args:
            strict: If True, raise errors for validation failures
        """
        self.strict = strict

    def parse(self, raw_message: str) -> SwiftMessage:
        """
        Parse a SWIFT message from raw text.

        Args:
            raw_message: Raw SWIFT message text

        Returns:
            Parsed SwiftMessage object

        Raises:
            SwiftParseError: If parsing fails in strict mode
        """
        message = SwiftMessage()
        message.raw_message = raw_message

        # Normalize line endings
        raw_message = raw_message.replace("\r\n", "\n").replace("\r", "\n")

        # Extract blocks
        blocks = self._extract_blocks(raw_message)

        for block in blocks:
            try:
                if block.block_type == "1":
                    message.basic_header = SwiftBasicHeader.from_content(block.content)
                elif block.block_type == "2":
                    message.application_header = SwiftApplicationHeader.from_content(
                        block.content
                    )
                elif block.block_type == "3":
                    message.user_header = SwiftUserHeader.from_content(block.content)
                elif block.block_type == "4":
                    message.text_block = self._parse_text_block(block.content)
                elif block.block_type == "5":
                    message.trailer = SwiftTrailerBlock.from_content(block.content)
            except Exception as e:
                error_msg = f"Error parsing block {block.block_type}: {str(e)}"
                message.parse_errors.append(error_msg)
                if self.strict:
                    raise SwiftParseError(error_msg, block.start, raw_message)

        # Validate required blocks
        if not message.basic_header.content:
            message.parse_errors.append("Missing basic header (block 1)")
        if not message.application_header.content:
            message.parse_errors.append("Missing application header (block 2)")
        if not message.text_block.fields:
            message.parse_errors.append("Missing or empty text block (block 4)")

        if self.strict and message.parse_errors:
            raise SwiftParseError(
                f"Parse errors: {'; '.join(message.parse_errors)}", 0, raw_message
            )

        return message

    def _extract_blocks(self, raw_message: str) -> list[BlockPosition]:
        """Extract all blocks from the message."""
        blocks = []
        pos = 0

        while pos < len(raw_message):
            # Find start of block
            start = raw_message.find("{", pos)
            if start == -1:
                break

            # Get block type
            if start + 2 < len(raw_message) and raw_message[start + 2] == ":":
                block_type = raw_message[start + 1]
            else:
                pos = start + 1
                continue

            # Find matching closing brace
            end = self._find_block_end(raw_message, start)
            if end == -1:
                if self.strict:
                    raise SwiftParseError(
                        f"Unclosed block {block_type}", start, raw_message
                    )
                break

            # Extract content (after block_type:)
            content = raw_message[start + 3 : end]
            blocks.append(
                BlockPosition(
                    block_type=block_type,
                    start=start,
                    end=end + 1,
                    content=content,
                )
            )

            pos = end + 1

        return blocks

    def _find_block_end(self, text: str, start: int) -> int:
        """Find the end of a block, handling nested braces."""
        depth = 0
        pos = start

        while pos < len(text):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return pos
            pos += 1

        return -1

    def _parse_text_block(self, content: str) -> SwiftTextBlock:
        """Parse the text block (block 4) into fields."""
        text_block = SwiftTextBlock()
        text_block.content = content

        # Remove trailing dash if present
        content = content.rstrip()
        if content.endswith("-"):
            content = content[:-1].rstrip()

        # Split into lines and parse fields
        lines = content.split("\n")
        current_tag = None
        current_qualifier = ""
        current_value_lines = []
        line_number = 0

        for line in lines:
            line_number += 1
            line = line.rstrip()

            # Check if this is a new field
            match = self.FIELD_START_PATTERN.match(line)
            if match:
                # Save previous field
                if current_tag:
                    from ai_engine.domains.banking.protocols.messaging.swift.swift_message import (
                        SwiftField,
                    )

                    value = "\n".join(current_value_lines)
                    text_block.fields.append(
                        SwiftField(
                            tag=current_tag,
                            value=value,
                            qualifier=current_qualifier,
                            line_number=line_number - len(current_value_lines),
                        )
                    )

                # Start new field
                full_tag = match.group(1)
                if len(full_tag) > 2 and full_tag[-1].isalpha():
                    current_tag = full_tag[:-1]
                    current_qualifier = full_tag[-1]
                else:
                    current_tag = full_tag
                    current_qualifier = ""

                # Get value after :TAG:
                value_start = match.end()
                current_value_lines = [line[value_start:]]
            elif current_tag:
                # Continuation of previous field
                current_value_lines.append(line)

        # Save last field
        if current_tag:
            from ai_engine.domains.banking.protocols.messaging.swift.swift_message import (
                SwiftField,
            )

            value = "\n".join(current_value_lines)
            text_block.fields.append(
                SwiftField(
                    tag=current_tag,
                    value=value,
                    qualifier=current_qualifier,
                    line_number=line_number - len(current_value_lines) + 1,
                )
            )

        return text_block

    def parse_field_32a(self, value: str) -> Tuple[str, str, str]:
        """
        Parse field 32A (Value Date/Currency/Amount).

        Format: YYMMDDCCCAMOUNT (e.g., 230101USD1000,00)

        Returns:
            Tuple of (date, currency, amount)
        """
        if len(value) < 12:
            raise SwiftParseError(f"Invalid 32A format: {value}")

        date = value[:6]
        currency = value[6:9]
        amount = value[9:].replace(",", ".")

        return date, currency, amount

    def parse_field_50(self, value: str) -> dict:
        """
        Parse field 50 (Ordering Customer).

        Returns dict with 'account' and 'name_address' keys.
        """
        lines = value.split("\n")
        result = {"account": None, "name_address": []}

        for line in lines:
            if line.startswith("/"):
                if result["account"] is None:
                    result["account"] = line[1:]
                else:
                    result["name_address"].append(line)
            else:
                result["name_address"].append(line)

        return result

    def parse_field_59(self, value: str) -> dict:
        """
        Parse field 59 (Beneficiary Customer).

        Returns dict with 'account' and 'name_address' keys.
        """
        lines = value.split("\n")
        result = {"account": None, "name_address": []}

        for line in lines:
            if line.startswith("/"):
                if result["account"] is None:
                    result["account"] = line[1:]
                else:
                    result["name_address"].append(line)
            else:
                result["name_address"].append(line)

        return result

    def parse_bic(self, value: str) -> dict:
        """
        Parse a BIC from a field value.

        Format: [/account]BIC or just BIC
        """
        lines = value.split("\n")
        result = {"account": None, "bic": None}

        for line in lines:
            line = line.strip()
            if line.startswith("/"):
                # Account line
                result["account"] = line[1:]
            elif len(line) == 8 or len(line) == 11:
                # BIC (8 or 11 characters)
                result["bic"] = line
            elif len(line) > 0 and result["bic"] is None:
                # Might be BIC with account prefix
                result["bic"] = line

        return result


def parse_swift_message(raw_message: str, strict: bool = True) -> SwiftMessage:
    """
    Convenience function to parse a SWIFT message.

    Args:
        raw_message: Raw SWIFT message text
        strict: If True, raise errors for validation failures

    Returns:
        Parsed SwiftMessage object
    """
    parser = SwiftParser(strict=strict)
    return parser.parse(raw_message)
