"""
QBITEL - Sample Protocol Data Generator

Generates synthetic legacy protocol samples for testing protocol discovery
and classification capabilities.

Protocol Types:
- IBM 3270 TN3270 (Terminal Emulation)
- IBM MQ Series Messages
- COBOL Fixed-Length Records (EBCDIC)
- ISO 8583 Financial Messages
- Modbus TCP Industrial Protocol
"""

import struct
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta
import hashlib


# =============================================================================
# EBCDIC Encoding Tables (US EBCDIC Code Page 037)
# =============================================================================

ASCII_TO_EBCDIC = {
    ' ': 0x40, '.': 0x4B, '<': 0x4C, '(': 0x4D, '+': 0x4E, '|': 0x4F,
    '&': 0x50, '!': 0x5A, '$': 0x5B, '*': 0x5C, ')': 0x5D, ';': 0x5E,
    '-': 0x60, '/': 0x61, ',': 0x6B, '%': 0x6C, '_': 0x6D, '>': 0x6E,
    '?': 0x6F, ':': 0x7A, '#': 0x7B, '@': 0x7C, "'": 0x7D, '=': 0x7E,
    '"': 0x7F, 'a': 0x81, 'b': 0x82, 'c': 0x83, 'd': 0x84, 'e': 0x85,
    'f': 0x86, 'g': 0x87, 'h': 0x88, 'i': 0x89, 'j': 0x91, 'k': 0x92,
    'l': 0x93, 'm': 0x94, 'n': 0x95, 'o': 0x96, 'p': 0x97, 'q': 0x98,
    'r': 0x99, 's': 0xA2, 't': 0xA3, 'u': 0xA4, 'v': 0xA5, 'w': 0xA6,
    'x': 0xA7, 'y': 0xA8, 'z': 0xA9, 'A': 0xC1, 'B': 0xC2, 'C': 0xC3,
    'D': 0xC4, 'E': 0xC5, 'F': 0xC6, 'G': 0xC7, 'H': 0xC8, 'I': 0xC9,
    'J': 0xD1, 'K': 0xD2, 'L': 0xD3, 'M': 0xD4, 'N': 0xD5, 'O': 0xD6,
    'P': 0xD7, 'Q': 0xD8, 'R': 0xD9, 'S': 0xE2, 'T': 0xE3, 'U': 0xE4,
    'V': 0xE5, 'W': 0xE6, 'X': 0xE7, 'Y': 0xE8, 'Z': 0xE9,
    '0': 0xF0, '1': 0xF1, '2': 0xF2, '3': 0xF3, '4': 0xF4,
    '5': 0xF5, '6': 0xF6, '7': 0xF7, '8': 0xF8, '9': 0xF9,
}


def ascii_to_ebcdic(text: str) -> bytes:
    """Convert ASCII string to EBCDIC bytes."""
    result = bytearray()
    for char in text:
        if char in ASCII_TO_EBCDIC:
            result.append(ASCII_TO_EBCDIC[char])
        else:
            result.append(0x40)  # Space for unknown chars
    return bytes(result)


def pad_ebcdic(text: str, length: int) -> bytes:
    """Pad text to fixed length in EBCDIC."""
    text = text[:length]  # Truncate if too long
    padded = text.ljust(length)
    return ascii_to_ebcdic(padded)


# =============================================================================
# TN3270 Protocol Samples (IBM 3270 Terminal)
# =============================================================================

class TN3270Generator:
    """Generate TN3270 protocol samples."""

    # 3270 Command Codes
    WRITE = 0xF1
    ERASE_WRITE = 0xF5
    ERASE_ALL_UNPROTECTED = 0x6F
    READ_BUFFER = 0xF2
    READ_MODIFIED = 0xF6

    # Write Control Character (WCC)
    WCC_RESET = 0xC3

    # Orders
    SBA = 0x11  # Set Buffer Address
    SF = 0x1D   # Start Field
    SA = 0x28   # Set Attribute
    IC = 0x13   # Insert Cursor

    @classmethod
    def generate_write_screen(cls, rows: int = 24, cols: int = 80) -> bytes:
        """Generate a 3270 write screen command."""
        data = bytearray()

        # Command and WCC
        data.append(cls.ERASE_WRITE)
        data.append(cls.WCC_RESET)

        # Add some field data
        # Set Buffer Address to row 1, col 1
        addr = cls._encode_buffer_address(1, 1, cols)
        data.append(cls.SBA)
        data.extend(addr)

        # Start protected field
        data.append(cls.SF)
        data.append(0xE0)  # Protected, display

        # Add screen title
        title = "QBITEL BANKING SYSTEM - MAIN MENU"
        data.extend(ascii_to_ebcdic(title))

        # Add menu options
        for i, option in enumerate([
            "1. Account Inquiry",
            "2. Transfer Funds",
            "3. Payment Processing",
            "4. Customer Maintenance",
            "5. Reports",
            "PF3=Exit  PF12=Cancel"
        ], start=3):
            addr = cls._encode_buffer_address(i + 2, 5, cols)
            data.append(cls.SBA)
            data.extend(addr)
            data.append(cls.SF)
            data.append(0xE0)  # Protected
            data.extend(ascii_to_ebcdic(option))

        # Add input field
        addr = cls._encode_buffer_address(15, 5, cols)
        data.append(cls.SBA)
        data.extend(addr)
        data.extend(ascii_to_ebcdic("Selection: "))
        data.append(cls.SF)
        data.append(0xC0)  # Unprotected, display

        # Insert cursor
        data.append(cls.IC)

        return bytes(data)

    @classmethod
    def _encode_buffer_address(cls, row: int, col: int, cols: int = 80) -> bytes:
        """Encode row/col to 3270 buffer address."""
        addr = (row - 1) * cols + (col - 1)
        # 12-bit encoding
        high = (addr >> 6) & 0x3F
        low = addr & 0x3F
        # Add 0x40 offset for valid 3270 address bytes
        return bytes([high | 0x40, low | 0x40])

    @classmethod
    def generate_read_modified(cls) -> bytes:
        """Generate a read modified command response."""
        data = bytearray()

        # AID byte (Enter key)
        data.append(0x7D)

        # Cursor address
        data.extend(cls._encode_buffer_address(15, 17, 80))

        # Field data (user input)
        data.append(cls.SBA)
        data.extend(cls._encode_buffer_address(15, 17, 80))
        data.extend(ascii_to_ebcdic("1"))

        return bytes(data)


# =============================================================================
# IBM MQ Series Protocol Samples
# =============================================================================

class MQSeriesGenerator:
    """Generate IBM MQ Series message samples."""

    @classmethod
    def generate_put_message(cls, queue_name: str, message: str) -> bytes:
        """Generate an MQ PUT message."""
        data = bytearray()

        # TSH (Transmission Segment Header) - 28 bytes
        data.extend(b"TSH ")  # Eye catcher
        data.extend(struct.pack(">I", len(message) + 100))  # Segment length
        data.extend(struct.pack(">H", 1))  # Segment type (data)
        data.extend(struct.pack(">H", 0))  # Control flags
        data.extend(b"\x00" * 16)  # Reserved

        # MQMD (Message Descriptor) simplified
        data.extend(b"MD  ")  # Structure ID
        data.extend(struct.pack(">I", 2))  # Version
        data.extend(struct.pack(">I", 8))  # Report options
        data.extend(struct.pack(">I", 0))  # Message type
        data.extend(struct.pack(">I", 0))  # Expiry
        data.extend(struct.pack(">I", 0))  # Feedback

        # Queue name (48 bytes, space padded)
        data.extend(pad_ebcdic(queue_name, 48))

        # Message data
        data.extend(message.encode('utf-8'))

        return bytes(data)

    @classmethod
    def generate_get_message(cls, queue_name: str) -> bytes:
        """Generate an MQ GET request."""
        data = bytearray()

        # TSH Header
        data.extend(b"TSH ")
        data.extend(struct.pack(">I", 80))  # Segment length
        data.extend(struct.pack(">H", 2))  # Segment type (get request)
        data.extend(struct.pack(">H", 0))  # Control flags
        data.extend(b"\x00" * 16)

        # GMO (Get Message Options)
        data.extend(b"GMO ")
        data.extend(struct.pack(">I", 1))  # Version
        data.extend(struct.pack(">I", 1))  # Options (wait)
        data.extend(struct.pack(">I", 30000))  # Wait interval (30s)

        # Queue name
        data.extend(pad_ebcdic(queue_name, 48))

        return bytes(data)


# =============================================================================
# COBOL Fixed-Length Record Samples
# =============================================================================

class COBOLRecordGenerator:
    """Generate COBOL fixed-length record samples."""

    @classmethod
    def generate_customer_record(
        cls,
        customer_id: int = None,
        name: str = None,
        balance: float = None
    ) -> bytes:
        """Generate a 500-byte customer master record."""
        data = bytearray()

        # Header (20 bytes)
        data.extend(ascii_to_ebcdic("AC"))  # Record type (Active)
        data.extend(ascii_to_ebcdic("01"))  # Version
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        data.extend(ascii_to_ebcdic(timestamp))  # Timestamp
        data.extend(b"\x00\x00")  # Filler

        # Identity (35 bytes)
        cust_id = customer_id or random.randint(1000000000, 9999999999)
        data.extend(ascii_to_ebcdic(f"{cust_id:010d}"))  # Customer ID
        data.extend(ascii_to_ebcdic("I"))  # Customer type (Individual)
        ssn = random.randint(100000000, 999999999)
        data.extend(ascii_to_ebcdic(f"{ssn:09d}"))  # SSN
        data.extend(pad_ebcdic("", 15))  # Tax ID

        # Name (61 bytes)
        name = name or "SMITH"
        data.extend(pad_ebcdic(name.upper(), 30))  # Last name
        data.extend(pad_ebcdic("JOHN", 20))  # First name
        data.extend(ascii_to_ebcdic("A"))  # Middle init
        data.extend(pad_ebcdic("MR", 5))  # Title
        data.extend(pad_ebcdic("", 5))  # Suffix

        # Address (120 bytes)
        data.extend(pad_ebcdic("123 MAIN STREET", 40))
        data.extend(pad_ebcdic("APT 4B", 40))
        data.extend(pad_ebcdic("NEW YORK", 25))
        data.extend(ascii_to_ebcdic("NY"))
        data.extend(pad_ebcdic("10001", 10))
        data.extend(ascii_to_ebcdic("USA"))

        # Contact (80 bytes)
        data.extend(ascii_to_ebcdic("2125551234"))  # Home phone
        data.extend(ascii_to_ebcdic("2125559876"))  # Work phone
        data.extend(ascii_to_ebcdic("9175551111"))  # Mobile
        data.extend(pad_ebcdic("john.smith@email.com", 50))  # Email

        # Financial (35 bytes) - COMP-3 packed decimal
        data.extend(ascii_to_ebcdic("750"))  # Credit score
        data.extend(cls._pack_decimal(50000.00, 9, 2))  # Credit limit
        balance = balance or random.uniform(-5000, 100000)
        data.extend(cls._pack_decimal(balance, 11, 2))  # Balance
        data.extend(cls._pack_decimal(45000.00, 9, 2))  # Available credit
        data.extend(cls._pack_decimal(1500.00, 9, 2))  # Last payment
        data.extend(ascii_to_ebcdic("20240115"))  # Last payment date

        # Relationship (22 bytes)
        data.extend(ascii_to_ebcdic("20150320"))  # Open date
        data.extend(ascii_to_ebcdic("00123"))  # Branch code
        data.extend(ascii_to_ebcdic("001234"))  # Officer ID
        data.extend(ascii_to_ebcdic("PR"))  # Segment (Premium)
        data.extend(ascii_to_ebcdic("2"))  # Risk rating

        # Flags (10 bytes)
        data.extend(ascii_to_ebcdic("Y"))  # KYC verified
        data.extend(ascii_to_ebcdic("N"))  # AML flag
        data.extend(ascii_to_ebcdic("N"))  # Fraud flag
        data.extend(ascii_to_ebcdic("N"))  # Deceased flag
        data.extend(ascii_to_ebcdic("N"))  # Bankrupt flag
        data.extend(b"\x00" * 5)  # Filler

        # Pad to 500 bytes
        current_len = len(data)
        if current_len < 500:
            data.extend(b"\x00" * (500 - current_len))

        return bytes(data[:500])

    @classmethod
    def _pack_decimal(cls, value: float, digits: int, decimals: int) -> bytes:
        """Pack a decimal value as COMP-3 (packed decimal)."""
        # COMP-3: each byte holds 2 digits, last nibble is sign (C=+, D=-)
        is_negative = value < 0
        value = abs(value)

        # Convert to string without decimal point
        int_value = int(value * (10 ** decimals))
        str_value = f"{int_value:0{digits + decimals}d}"

        # Calculate packed length
        packed_len = (digits + decimals + 2) // 2

        # Pack digits
        result = bytearray()
        padded = str_value.zfill(packed_len * 2 - 1)

        for i in range(0, len(padded) - 1, 2):
            high = int(padded[i])
            low = int(padded[i + 1])
            result.append((high << 4) | low)

        # Last byte: digit + sign
        last_digit = int(padded[-1])
        sign = 0x0D if is_negative else 0x0C
        result.append((last_digit << 4) | sign)

        return bytes(result)

    @classmethod
    def generate_transaction_record(
        cls,
        txn_type: str = "DB",
        amount: float = None
    ) -> bytes:
        """Generate a 300-byte transaction record."""
        data = bytearray()

        # Header (16 bytes)
        data.extend(ascii_to_ebcdic(txn_type))  # Record type
        data.extend(ascii_to_ebcdic("01"))  # Version
        seq = random.randint(100000000000, 999999999999)
        data.extend(ascii_to_ebcdic(f"{seq:012d}"))  # Sequence

        # Identity (48 bytes)
        txn_id = hashlib.md5(str(random.random()).encode()).hexdigest()[:20]
        data.extend(pad_ebcdic(txn_id.upper(), 20))  # Transaction ID
        ref = f"REF{random.randint(1000000000, 9999999999):010d}"
        data.extend(pad_ebcdic(ref, 16))  # Reference number
        data.extend(ascii_to_ebcdic(f"{random.randint(10000000, 99999999):08d}"))  # Batch ID
        data.extend(ascii_to_ebcdic("ATM "))  # Source system

        # Account info (28 bytes)
        data.extend(ascii_to_ebcdic(f"{random.randint(100000000000, 999999999999):012d}"))  # From account
        data.extend(ascii_to_ebcdic(f"{random.randint(100000000000, 999999999999):012d}"))  # To account
        data.extend(ascii_to_ebcdic("CH"))  # From type (Checking)
        data.extend(ascii_to_ebcdic("SV"))  # To type (Savings)

        # Amount info (32 bytes)
        amount = amount or random.uniform(10, 5000)
        data.extend(cls._pack_decimal(amount, 13, 2))  # Amount
        data.extend(ascii_to_ebcdic("USD"))  # Currency
        data.extend(cls._pack_decimal(1.0, 5, 6))  # Exchange rate
        data.extend(cls._pack_decimal(amount, 13, 2))  # Local amount
        data.extend(cls._pack_decimal(2.50, 7, 2))  # Fee

        # Datetime (33 bytes)
        now = datetime.now()
        data.extend(ascii_to_ebcdic(now.strftime("%Y%m%d")))  # Date
        data.extend(ascii_to_ebcdic(now.strftime("%H%M%S")))  # Time
        data.extend(ascii_to_ebcdic(now.strftime("%Y%m%d")))  # Posting date
        data.extend(ascii_to_ebcdic(now.strftime("%Y%m%d")))  # Value date
        data.extend(ascii_to_ebcdic("EST"))  # Timezone

        # Location (52 bytes)
        data.extend(ascii_to_ebcdic("00123"))  # Branch code
        data.extend(pad_ebcdic("ATM00042", 8))  # Terminal ID
        data.extend(pad_ebcdic("MERCH123456789", 15))  # Merchant ID
        data.extend(ascii_to_ebcdic("5411"))  # MCC code (grocery)
        data.extend(pad_ebcdic("NEW YORK", 20))  # City
        data.extend(ascii_to_ebcdic("NY"))  # State
        data.extend(ascii_to_ebcdic("USA"))  # Country

        # Authorization (16 bytes)
        data.extend(pad_ebcdic("A12345", 6))  # Auth code
        data.extend(ascii_to_ebcdic("00"))  # Response (approved)
        data.extend(ascii_to_ebcdic("123456"))  # Auth time
        data.extend(ascii_to_ebcdic("M"))  # CVV result
        data.extend(ascii_to_ebcdic("Y"))  # AVS result

        # Description (80 bytes)
        data.extend(pad_ebcdic("ATM WITHDRAWAL - MAIN ST BRANCH", 40))
        data.extend(pad_ebcdic("", 40))

        # Audit (44 bytes)
        data.extend(pad_ebcdic("SYSTEM01", 8))  # Created by
        data.extend(ascii_to_ebcdic(now.strftime("%Y%m%d%H%M%S")))  # Created timestamp
        data.extend(pad_ebcdic("", 8))  # Modified by
        data.extend(ascii_to_ebcdic("00000000000000"))  # Modified timestamp

        # Pad to 300 bytes
        current_len = len(data)
        if current_len < 300:
            data.extend(b"\x00" * (300 - current_len))

        return bytes(data[:300])


# =============================================================================
# ISO 8583 Financial Message Samples
# =============================================================================

class ISO8583Generator:
    """Generate ISO 8583 financial message samples."""

    @classmethod
    def generate_authorization_request(
        cls,
        card_number: str = None,
        amount: int = None
    ) -> bytes:
        """Generate an ISO 8583 authorization request (0100)."""
        data = bytearray()

        # Length header (2 bytes)
        # Will be filled in at the end

        # MTI (Message Type Indicator) - 4 bytes ASCII
        data.extend(b"0100")

        # Primary Bitmap - 8 bytes (fields 1-64)
        # Indicating which fields are present
        bitmap = 0
        bitmap |= (1 << (64 - 2))   # Field 2: PAN
        bitmap |= (1 << (64 - 3))   # Field 3: Processing Code
        bitmap |= (1 << (64 - 4))   # Field 4: Amount
        bitmap |= (1 << (64 - 11))  # Field 11: STAN
        bitmap |= (1 << (64 - 12))  # Field 12: Time
        bitmap |= (1 << (64 - 13))  # Field 13: Date
        bitmap |= (1 << (64 - 14))  # Field 14: Expiry
        bitmap |= (1 << (64 - 22))  # Field 22: POS Entry Mode
        bitmap |= (1 << (64 - 35))  # Field 35: Track 2
        bitmap |= (1 << (64 - 41))  # Field 41: Terminal ID
        bitmap |= (1 << (64 - 42))  # Field 42: Merchant ID
        bitmap |= (1 << (64 - 49))  # Field 49: Currency Code

        data.extend(struct.pack(">Q", bitmap))

        # Field 2: PAN (Primary Account Number) - LLVAR
        card = card_number or f"4{random.randint(100000000000000, 999999999999999):015d}"
        data.append(len(card))
        data.extend(card.encode('ascii'))

        # Field 3: Processing Code - 6 digits
        data.extend(b"000000")  # Purchase

        # Field 4: Transaction Amount - 12 digits
        amt = amount or random.randint(1000, 500000)  # Cents
        data.extend(f"{amt:012d}".encode('ascii'))

        # Field 11: STAN - 6 digits
        stan = random.randint(1, 999999)
        data.extend(f"{stan:06d}".encode('ascii'))

        # Field 12: Time - 6 digits HHMMSS
        now = datetime.now()
        data.extend(now.strftime("%H%M%S").encode('ascii'))

        # Field 13: Date - 4 digits MMDD
        data.extend(now.strftime("%m%d").encode('ascii'))

        # Field 14: Expiry Date - 4 digits YYMM
        expiry = now + timedelta(days=365 * 3)
        data.extend(expiry.strftime("%y%m").encode('ascii'))

        # Field 22: POS Entry Mode - 3 digits
        data.extend(b"051")  # Chip read

        # Field 35: Track 2 Data - LLVAR
        track2 = f"{card}={expiry.strftime('%y%m')}101"
        data.append(len(track2))
        data.extend(track2.encode('ascii'))

        # Field 41: Terminal ID - 8 chars
        data.extend(b"TERM0001")

        # Field 42: Merchant ID - 15 chars
        data.extend(b"MERCHANT0000001")

        # Field 49: Currency Code - 3 digits
        data.extend(b"840")  # USD

        # Prepend length
        msg_len = len(data)
        result = struct.pack(">H", msg_len) + bytes(data)

        return result


# =============================================================================
# Modbus TCP Protocol Samples
# =============================================================================

class ModbusTCPGenerator:
    """Generate Modbus TCP protocol samples."""

    @classmethod
    def generate_read_holding_registers(
        cls,
        unit_id: int = 1,
        start_addr: int = 0,
        quantity: int = 10
    ) -> bytes:
        """Generate a Modbus TCP read holding registers request."""
        data = bytearray()

        # MBAP Header (7 bytes)
        transaction_id = random.randint(1, 65535)
        data.extend(struct.pack(">H", transaction_id))  # Transaction ID
        data.extend(struct.pack(">H", 0))  # Protocol ID (0 = Modbus)
        data.extend(struct.pack(">H", 6))  # Length (remaining bytes)
        data.append(unit_id)  # Unit ID

        # PDU (Protocol Data Unit)
        data.append(0x03)  # Function code: Read Holding Registers
        data.extend(struct.pack(">H", start_addr))  # Starting address
        data.extend(struct.pack(">H", quantity))  # Quantity

        return bytes(data)

    @classmethod
    def generate_write_single_register(
        cls,
        unit_id: int = 1,
        address: int = 0,
        value: int = 0
    ) -> bytes:
        """Generate a Modbus TCP write single register request."""
        data = bytearray()

        # MBAP Header
        transaction_id = random.randint(1, 65535)
        data.extend(struct.pack(">H", transaction_id))
        data.extend(struct.pack(">H", 0))
        data.extend(struct.pack(">H", 6))
        data.append(unit_id)

        # PDU
        data.append(0x06)  # Function code: Write Single Register
        data.extend(struct.pack(">H", address))
        data.extend(struct.pack(">H", value))

        return bytes(data)


# =============================================================================
# Sample Data Generator
# =============================================================================

def generate_protocol_samples(protocol_type: str, count: int = 10) -> List[bytes]:
    """
    Generate sample protocol data for testing.

    Args:
        protocol_type: One of 'tn3270', 'mq', 'cobol_customer', 'cobol_txn',
                      'iso8583', 'modbus'
        count: Number of samples to generate

    Returns:
        List of protocol message bytes
    """
    samples = []

    for _ in range(count):
        if protocol_type == "tn3270":
            if random.random() < 0.7:
                samples.append(TN3270Generator.generate_write_screen())
            else:
                samples.append(TN3270Generator.generate_read_modified())

        elif protocol_type == "mq":
            queue_name = random.choice([
                "CUSTOMER.REQUEST.QUEUE",
                "PAYMENT.PROCESS.QUEUE",
                "NOTIFICATION.OUT.QUEUE",
            ])
            if random.random() < 0.6:
                message = f'{{"action": "inquiry", "customer_id": {random.randint(1000, 9999)}}}'
                samples.append(MQSeriesGenerator.generate_put_message(queue_name, message))
            else:
                samples.append(MQSeriesGenerator.generate_get_message(queue_name))

        elif protocol_type == "cobol_customer":
            samples.append(COBOLRecordGenerator.generate_customer_record())

        elif protocol_type == "cobol_txn":
            txn_type = random.choice(["DB", "CR", "RV", "AJ"])
            samples.append(COBOLRecordGenerator.generate_transaction_record(txn_type))

        elif protocol_type == "iso8583":
            samples.append(ISO8583Generator.generate_authorization_request())

        elif protocol_type == "modbus":
            if random.random() < 0.7:
                samples.append(ModbusTCPGenerator.generate_read_holding_registers(
                    unit_id=random.randint(1, 10),
                    start_addr=random.randint(0, 100),
                    quantity=random.randint(1, 20)
                ))
            else:
                samples.append(ModbusTCPGenerator.generate_write_single_register(
                    unit_id=random.randint(1, 10),
                    address=random.randint(0, 100),
                    value=random.randint(0, 65535)
                ))

        else:
            raise ValueError(f"Unknown protocol type: {protocol_type}")

    return samples


def get_all_sample_protocols() -> Dict[str, List[bytes]]:
    """Get sample data for all supported protocol types."""
    return {
        "tn3270": generate_protocol_samples("tn3270", 20),
        "mq_series": generate_protocol_samples("mq", 20),
        "cobol_customer": generate_protocol_samples("cobol_customer", 20),
        "cobol_transaction": generate_protocol_samples("cobol_txn", 20),
        "iso8583": generate_protocol_samples("iso8583", 20),
        "modbus_tcp": generate_protocol_samples("modbus", 20),
    }


if __name__ == "__main__":
    # Generate samples for testing
    print("Generating sample protocol data...")

    samples = get_all_sample_protocols()

    for protocol_name, protocol_samples in samples.items():
        print(f"\n{protocol_name}: {len(protocol_samples)} samples")
        if protocol_samples:
            sample = protocol_samples[0]
            print(f"  Sample size: {len(sample)} bytes")
            print(f"  First 32 bytes: {sample[:32].hex()}")
