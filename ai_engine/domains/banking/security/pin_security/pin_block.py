"""
PIN Block Formats

Implements various PIN block formats used in payment systems.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import secrets


class PINBlockFormat(Enum):
    """PIN Block formats."""

    ISO_0 = "ISO_0"  # ISO 9564-1 Format 0 (same as ANSI X9.8)
    ISO_1 = "ISO_1"  # ISO 9564-1 Format 1
    ISO_2 = "ISO_2"  # ISO 9564-1 Format 2
    ISO_3 = "ISO_3"  # ISO 9564-1 Format 3
    ISO_4 = "ISO_4"  # ISO 9564-1 Format 4 (AES)

    # Legacy formats
    ANSI_X98 = "ANSI_X98"  # Same as ISO Format 0
    VISA_1 = "VISA_1"  # Visa Format 1
    VISA_3 = "VISA_3"  # Visa Format 3 (same as ISO 0)
    ECI_1 = "ECI_1"  # ECI Format 1
    ECI_2 = "ECI_2"  # ECI Format 2
    ECI_3 = "ECI_3"  # ECI Format 3
    IBM_3621 = "IBM_3621"  # IBM 3621
    IBM_3624 = "IBM_3624"  # IBM 3624


@dataclass
class PINBlock:
    """
    PIN Block structure.

    Represents an encrypted or clear PIN block with associated metadata.
    """

    format: PINBlockFormat
    block_data: bytes  # 8 bytes for 3DES, 16 bytes for AES
    pan: Optional[str] = None  # Required for formats that XOR with PAN

    # Encryption info
    is_encrypted: bool = False
    key_identifier: Optional[str] = None

    @classmethod
    def create_iso_format_0(cls, pin: str, pan: str) -> "PINBlock":
        """
        Create ISO Format 0 PIN Block.

        Format: 0 L P P P P P/F P/F P/F P/F P/F P/F P/F P/F P/F
        Where:
            0 = Format code
            L = PIN length (4-12)
            P = PIN digit
            F = Fill (0xF)

        XOR with PAN block: 0000 PPPP PPPP PPPP
        """
        if not 4 <= len(pin) <= 12:
            raise ValueError("PIN must be 4-12 digits")

        if not pin.isdigit():
            raise ValueError("PIN must be numeric")

        # Build PIN block (clear)
        # Format: 0 + length + PIN + padding with F
        pin_len = f"{len(pin):01X}"
        padded_pin = pin.ljust(14, "F")
        clear_pin_block = f"0{pin_len}{padded_pin}"

        # Build PAN block
        # Format: 0000 + rightmost 12 digits of PAN excluding check digit
        pan_digits = pan.replace(" ", "").replace("-", "")
        if len(pan_digits) < 13:
            raise ValueError("PAN must be at least 13 digits")

        pan_block_digits = pan_digits[-13:-1]  # Rightmost 12 excluding check digit
        pan_block = f"0000{pan_block_digits}"

        # XOR PIN block with PAN block
        pin_block_int = int(clear_pin_block, 16)
        pan_block_int = int(pan_block, 16)
        result_int = pin_block_int ^ pan_block_int
        result_hex = f"{result_int:016X}"

        return cls(
            format=PINBlockFormat.ISO_0,
            block_data=bytes.fromhex(result_hex),
            pan=pan,
            is_encrypted=False,
        )

    @classmethod
    def create_iso_format_1(cls, pin: str, transaction_number: int) -> "PINBlock":
        """
        Create ISO Format 1 PIN Block.

        Format: 1 L P P P P R R R R R R R R R R
        Where:
            1 = Format code
            L = PIN length
            P = PIN digit (right justified)
            R = Random padding

        Does not require PAN.
        """
        if not 4 <= len(pin) <= 12:
            raise ValueError("PIN must be 4-12 digits")

        # Generate random padding
        random_bytes = secrets.token_bytes(8)
        random_hex = random_bytes.hex().upper()

        # Build block
        pin_len = f"{len(pin):01X}"
        block = f"1{pin_len}{pin.ljust(14 - len(pin), random_hex[:14 - len(pin)])}"

        # Use remaining random for padding
        if len(block) < 16:
            block += random_hex[:(16 - len(block))]

        return cls(
            format=PINBlockFormat.ISO_1,
            block_data=bytes.fromhex(block[:16]),
            is_encrypted=False,
        )

    @classmethod
    def create_iso_format_2(cls, pin: str) -> "PINBlock":
        """
        Create ISO Format 2 PIN Block.

        Format: 2 L P P P P F F F F F F F F F F
        Where:
            2 = Format code
            L = PIN length
            P = PIN digit
            F = Fill (0xF)

        Used for chip card PIN verification.
        Does not require PAN.
        """
        if not 4 <= len(pin) <= 12:
            raise ValueError("PIN must be 4-12 digits")

        pin_len = f"{len(pin):01X}"
        padded_pin = pin.ljust(14, "F")
        block = f"2{pin_len}{padded_pin}"

        return cls(
            format=PINBlockFormat.ISO_2,
            block_data=bytes.fromhex(block),
            is_encrypted=False,
        )

    @classmethod
    def create_iso_format_3(cls, pin: str, pan: str) -> "PINBlock":
        """
        Create ISO Format 3 PIN Block.

        Similar to Format 0 but with random padding instead of 0xF.

        Format: 3 L P P P P R R R R R R R R R R
        XOR with PAN block.
        """
        if not 4 <= len(pin) <= 12:
            raise ValueError("PIN must be 4-12 digits")

        # Generate random padding (hex digits A-F only)
        random_padding = ""
        while len(random_padding) < 14 - len(pin):
            b = secrets.token_bytes(1)[0]
            h = f"{b:02X}"
            for c in h:
                if c in "ABCDEF" and len(random_padding) < 14 - len(pin):
                    random_padding += c

        pin_len = f"{len(pin):01X}"
        clear_pin_block = f"3{pin_len}{pin}{random_padding}"

        # Build PAN block
        pan_digits = pan.replace(" ", "").replace("-", "")
        pan_block_digits = pan_digits[-13:-1]
        pan_block = f"0000{pan_block_digits}"

        # XOR
        pin_block_int = int(clear_pin_block, 16)
        pan_block_int = int(pan_block, 16)
        result_int = pin_block_int ^ pan_block_int
        result_hex = f"{result_int:016X}"

        return cls(
            format=PINBlockFormat.ISO_3,
            block_data=bytes.fromhex(result_hex),
            pan=pan,
            is_encrypted=False,
        )

    @classmethod
    def create_iso_format_4(cls, pin: str, pan: str) -> "PINBlock":
        """
        Create ISO Format 4 PIN Block (AES).

        16-byte block for AES encryption.

        Format:
            Block A: 4 L P P P P P/A P/A P/A P/A P/A P/A P/A P/A P/A A A A A A A A A A A A A A A A A
            XOR with:
            Block B: PAN block (16 bytes)
        """
        if not 4 <= len(pin) <= 12:
            raise ValueError("PIN must be 4-12 digits")

        pin_len = f"{len(pin):01X}"

        # Generate random fill (hex A-F only)
        fill_needed = 30 - len(pin)
        fill = ""
        while len(fill) < fill_needed:
            b = secrets.token_bytes(1)[0]
            h = f"{b:02X}"
            for c in h:
                if c in "ABCDEF" and len(fill) < fill_needed:
                    fill += c

        # Build PIN block A
        block_a = f"4{pin_len}{pin}{fill}"

        # Build PAN block B
        pan_digits = pan.replace(" ", "").replace("-", "")
        pan_len = f"{len(pan_digits):02X}"
        pan_block = f"{pan_len}{pan_digits}".ljust(32, "0")

        # XOR
        block_a_int = int(block_a, 16)
        block_b_int = int(pan_block[:32], 16)
        result_int = block_a_int ^ block_b_int
        result_hex = f"{result_int:032X}"

        return cls(
            format=PINBlockFormat.ISO_4,
            block_data=bytes.fromhex(result_hex),
            pan=pan,
            is_encrypted=False,
        )

    def extract_pin(self, pan: Optional[str] = None) -> str:
        """
        Extract PIN from clear PIN block.

        Args:
            pan: PAN for formats that require it

        Returns:
            Extracted PIN
        """
        if self.is_encrypted:
            raise ValueError("Cannot extract PIN from encrypted block")

        block_hex = self.block_data.hex().upper()

        if self.format == PINBlockFormat.ISO_0:
            if not pan and not self.pan:
                raise ValueError("PAN required for ISO Format 0")
            pan = pan or self.pan

            # Rebuild PAN block
            pan_digits = pan.replace(" ", "").replace("-", "")
            pan_block = f"0000{pan_digits[-13:-1]}"

            # XOR to get clear PIN block
            pin_block_int = int(block_hex, 16)
            pan_block_int = int(pan_block, 16)
            clear_int = pin_block_int ^ pan_block_int
            clear_hex = f"{clear_int:016X}"

            # Extract PIN
            pin_len = int(clear_hex[1], 16)
            return clear_hex[2:2 + pin_len]

        elif self.format == PINBlockFormat.ISO_1:
            pin_len = int(block_hex[1], 16)
            return block_hex[2:2 + pin_len]

        elif self.format == PINBlockFormat.ISO_2:
            pin_len = int(block_hex[1], 16)
            return block_hex[2:2 + pin_len]

        elif self.format == PINBlockFormat.ISO_3:
            if not pan and not self.pan:
                raise ValueError("PAN required for ISO Format 3")
            pan = pan or self.pan

            # XOR with PAN block
            pan_digits = pan.replace(" ", "").replace("-", "")
            pan_block = f"0000{pan_digits[-13:-1]}"

            pin_block_int = int(block_hex, 16)
            pan_block_int = int(pan_block, 16)
            clear_int = pin_block_int ^ pan_block_int
            clear_hex = f"{clear_int:016X}"

            pin_len = int(clear_hex[1], 16)
            return clear_hex[2:2 + pin_len]

        elif self.format == PINBlockFormat.ISO_4:
            if not pan and not self.pan:
                raise ValueError("PAN required for ISO Format 4")
            pan = pan or self.pan

            # XOR with PAN block
            pan_digits = pan.replace(" ", "").replace("-", "")
            pan_len_hex = f"{len(pan_digits):02X}"
            pan_block = f"{pan_len_hex}{pan_digits}".ljust(32, "0")

            block_int = int(block_hex, 16)
            pan_int = int(pan_block[:32], 16)
            clear_int = block_int ^ pan_int
            clear_hex = f"{clear_int:032X}"

            pin_len = int(clear_hex[1], 16)
            return clear_hex[2:2 + pin_len]

        else:
            raise ValueError(f"Extraction not implemented for format: {self.format}")

    def to_hex(self) -> str:
        """Return block as hex string."""
        return self.block_data.hex().upper()

    @classmethod
    def from_hex(cls, hex_string: str, format: PINBlockFormat, pan: Optional[str] = None) -> "PINBlock":
        """Create PIN block from hex string."""
        return cls(
            format=format,
            block_data=bytes.fromhex(hex_string),
            pan=pan,
            is_encrypted=False,
        )
