"""
EMV TLV (Tag-Length-Value) Parser and Builder

Handles BER-TLV encoding/decoding for EMV data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class TlvTag:
    """Represents a TLV tag."""

    value: bytes
    is_constructed: bool = False
    is_primitive: bool = True

    @property
    def hex(self) -> str:
        """Get tag as hex string."""
        return self.value.hex().upper()

    @property
    def length(self) -> int:
        """Get tag length in bytes."""
        return len(self.value)

    @classmethod
    def from_hex(cls, hex_str: str) -> "TlvTag":
        """Create tag from hex string."""
        value = bytes.fromhex(hex_str)
        # Check if constructed (bit 6 of first byte)
        is_constructed = bool(value[0] & 0x20)
        return cls(value=value, is_constructed=is_constructed, is_primitive=not is_constructed)

    def __str__(self) -> str:
        return self.hex


@dataclass
class TlvData:
    """Represents a complete TLV data element."""

    tag: TlvTag
    length: int
    value: bytes
    children: List["TlvData"] = field(default_factory=list)

    @property
    def tag_hex(self) -> str:
        """Get tag as hex string."""
        return self.tag.hex

    @property
    def value_hex(self) -> str:
        """Get value as hex string."""
        return self.value.hex().upper()

    @property
    def is_constructed(self) -> bool:
        """Check if this is a constructed (template) tag."""
        return self.tag.is_constructed

    def get_child(self, tag: str) -> Optional["TlvData"]:
        """Get child by tag."""
        tag = tag.upper()
        for child in self.children:
            if child.tag_hex == tag:
                return child
        return None

    def get_all_children(self, tag: str) -> List["TlvData"]:
        """Get all children matching tag."""
        tag = tag.upper()
        return [child for child in self.children if child.tag_hex == tag]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "tag": self.tag_hex,
            "length": self.length,
            "value": self.value_hex,
        }
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result

    def __str__(self) -> str:
        return f"TLV({self.tag_hex}, len={self.length}, value={self.value_hex[:20]}...)"


class TlvParser:
    """
    Parser for BER-TLV encoded data.

    Handles:
    - Multi-byte tags
    - Multi-byte lengths
    - Constructed (template) tags
    - Nested structures
    """

    def __init__(self, strict: bool = True):
        """
        Initialize parser.

        Args:
            strict: If True, raise errors for invalid data
        """
        self.strict = strict

    def parse(self, data: Union[bytes, str]) -> List[TlvData]:
        """
        Parse TLV data.

        Args:
            data: TLV data as bytes or hex string

        Returns:
            List of TlvData elements
        """
        if isinstance(data, str):
            data = bytes.fromhex(data)

        return self._parse_tlv(data, 0, len(data))

    def _parse_tlv(self, data: bytes, offset: int, end: int) -> List[TlvData]:
        """Parse TLV data from offset to end."""
        elements = []

        while offset < end:
            # Skip padding bytes (0x00 or 0xFF)
            if data[offset] in (0x00, 0xFF):
                offset += 1
                continue

            # Parse tag
            tag, tag_len = self._parse_tag(data, offset)
            offset += tag_len

            if offset >= end:
                if self.strict:
                    raise ValueError(f"Unexpected end of data after tag {tag.hex}")
                break

            # Parse length
            length, len_bytes = self._parse_length(data, offset)
            offset += len_bytes

            if offset + length > end:
                if self.strict:
                    raise ValueError(
                        f"Length {length} exceeds remaining data for tag {tag.hex}"
                    )
                length = end - offset

            # Extract value
            value = data[offset : offset + length]

            # Create TLV element
            tlv = TlvData(tag=tag, length=length, value=value)

            # If constructed, parse children
            if tag.is_constructed and length > 0:
                try:
                    tlv.children = self._parse_tlv(value, 0, len(value))
                except Exception:
                    # If child parsing fails, keep value as-is
                    pass

            elements.append(tlv)
            offset += length

        return elements

    def _parse_tag(self, data: bytes, offset: int) -> Tuple[TlvTag, int]:
        """Parse a BER-TLV tag."""
        if offset >= len(data):
            raise ValueError("No data for tag")

        first_byte = data[offset]

        # Check if multi-byte tag (bits 5-1 all set)
        if (first_byte & 0x1F) == 0x1F:
            # Multi-byte tag
            tag_bytes = [first_byte]
            offset += 1

            while offset < len(data):
                byte = data[offset]
                tag_bytes.append(byte)
                offset += 1

                # Check if this is the last byte (bit 8 not set)
                if not (byte & 0x80):
                    break

            tag = TlvTag(value=bytes(tag_bytes))
            return tag, len(tag_bytes)
        else:
            # Single-byte tag
            tag = TlvTag(value=bytes([first_byte]))
            return tag, 1

    def _parse_length(self, data: bytes, offset: int) -> Tuple[int, int]:
        """Parse a BER-TLV length."""
        if offset >= len(data):
            raise ValueError("No data for length")

        first_byte = data[offset]

        if first_byte < 0x80:
            # Short form: single byte length
            return first_byte, 1

        elif first_byte == 0x80:
            # Indefinite form (not typically used in EMV)
            raise ValueError("Indefinite length not supported")

        elif first_byte == 0x81:
            # Long form: 1 additional byte
            if offset + 1 >= len(data):
                raise ValueError("Insufficient data for length")
            return data[offset + 1], 2

        elif first_byte == 0x82:
            # Long form: 2 additional bytes
            if offset + 2 >= len(data):
                raise ValueError("Insufficient data for length")
            return (data[offset + 1] << 8) | data[offset + 2], 3

        elif first_byte == 0x83:
            # Long form: 3 additional bytes
            if offset + 3 >= len(data):
                raise ValueError("Insufficient data for length")
            return (data[offset + 1] << 16) | (data[offset + 2] << 8) | data[offset + 3], 4

        else:
            raise ValueError(f"Unsupported length encoding: {first_byte:02X}")


class TlvBuilder:
    """
    Builder for BER-TLV encoded data.
    """

    def __init__(self):
        """Initialize builder."""
        self._elements: List[TlvData] = []

    def add(self, tag: str, value: Union[bytes, str]) -> "TlvBuilder":
        """
        Add a TLV element.

        Args:
            tag: Tag as hex string
            value: Value as bytes or hex string

        Returns:
            Self for chaining
        """
        if isinstance(value, str):
            value = bytes.fromhex(value)

        tlv_tag = TlvTag.from_hex(tag)
        self._elements.append(TlvData(tag=tlv_tag, length=len(value), value=value))
        return self

    def add_numeric(self, tag: str, value: int, length: int) -> "TlvBuilder":
        """
        Add a numeric TLV element (BCD encoded).

        Args:
            tag: Tag as hex string
            value: Numeric value
            length: Length in bytes

        Returns:
            Self for chaining
        """
        # Convert to BCD
        value_str = str(value).zfill(length * 2)
        value_bytes = bytes.fromhex(value_str)
        return self.add(tag, value_bytes)

    def add_alphanumeric(self, tag: str, value: str, max_length: int = None) -> "TlvBuilder":
        """
        Add an alphanumeric TLV element.

        Args:
            tag: Tag as hex string
            value: String value
            max_length: Maximum length (truncate if exceeded)

        Returns:
            Self for chaining
        """
        if max_length and len(value) > max_length:
            value = value[:max_length]
        value_bytes = value.encode("ascii")
        return self.add(tag, value_bytes)

    def add_constructed(self, tag: str, children: List[TlvData]) -> "TlvBuilder":
        """
        Add a constructed (template) TLV element.

        Args:
            tag: Tag as hex string
            children: Child TLV elements

        Returns:
            Self for chaining
        """
        tlv_tag = TlvTag.from_hex(tag)
        tlv_tag.is_constructed = True
        tlv_tag.is_primitive = False

        # Build child data
        child_data = b""
        for child in children:
            child_data += self._encode_tlv(child)

        self._elements.append(
            TlvData(
                tag=tlv_tag,
                length=len(child_data),
                value=child_data,
                children=children,
            )
        )
        return self

    def build(self) -> bytes:
        """Build the TLV data."""
        result = b""
        for element in self._elements:
            result += self._encode_tlv(element)
        return result

    def build_hex(self) -> str:
        """Build the TLV data as hex string."""
        return self.build().hex().upper()

    def _encode_tlv(self, tlv: TlvData) -> bytes:
        """Encode a single TLV element."""
        result = tlv.tag.value
        result += self._encode_length(tlv.length)
        result += tlv.value
        return result

    def _encode_length(self, length: int) -> bytes:
        """Encode a BER-TLV length."""
        if length < 0x80:
            return bytes([length])
        elif length < 0x100:
            return bytes([0x81, length])
        elif length < 0x10000:
            return bytes([0x82, (length >> 8) & 0xFF, length & 0xFF])
        else:
            return bytes([
                0x83,
                (length >> 16) & 0xFF,
                (length >> 8) & 0xFF,
                length & 0xFF,
            ])

    def reset(self) -> "TlvBuilder":
        """Reset the builder."""
        self._elements = []
        return self


def parse_tlv(data: Union[bytes, str], strict: bool = True) -> List[TlvData]:
    """
    Convenience function to parse TLV data.

    Args:
        data: TLV data as bytes or hex string
        strict: If True, raise errors for invalid data

    Returns:
        List of TlvData elements
    """
    parser = TlvParser(strict=strict)
    return parser.parse(data)


def build_tlv(elements: List[Tuple[str, Union[bytes, str]]]) -> bytes:
    """
    Convenience function to build TLV data.

    Args:
        elements: List of (tag, value) tuples

    Returns:
        TLV encoded bytes
    """
    builder = TlvBuilder()
    for tag, value in elements:
        builder.add(tag, value)
    return builder.build()


def tlv_to_dict(data: Union[bytes, str]) -> Dict[str, str]:
    """
    Parse TLV data to a flat dictionary of tag -> value.

    Args:
        data: TLV data as bytes or hex string

    Returns:
        Dictionary mapping tags to hex values
    """
    result = {}

    def _flatten(elements: List[TlvData], prefix: str = ""):
        for elem in elements:
            key = prefix + elem.tag_hex if prefix else elem.tag_hex
            result[key] = elem.value_hex

            if elem.children:
                _flatten(elem.children, f"{key}.")

    elements = parse_tlv(data, strict=False)
    _flatten(elements)
    return result
