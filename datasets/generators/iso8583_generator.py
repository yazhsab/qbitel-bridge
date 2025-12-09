"""
ISO 8583 Protocol Message Generator

Generates realistic ISO 8583 financial transaction messages for training
the protocol discovery and field detection models.

ISO 8583 Structure:
- MTI (Message Type Indicator): 4 bytes
- Bitmap: 8 or 16 bytes (primary + secondary)
- Data Elements: Variable based on bitmap
"""

import json
import random
import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib


class ISO8583Generator:
    """Generate realistic ISO 8583 messages for ML training."""

    # Message Type Indicators
    MTI_TYPES = {
        "0100": "authorization_request",
        "0110": "authorization_response",
        "0200": "financial_request",
        "0210": "financial_response",
        "0400": "reversal_request",
        "0410": "reversal_response",
        "0420": "reversal_advice",
        "0430": "reversal_advice_response",
        "0800": "network_management_request",
        "0810": "network_management_response",
    }

    # Data Element Definitions (subset of full spec)
    DATA_ELEMENTS = {
        2: {"name": "pan", "type": "LLVAR", "max_len": 19, "description": "Primary Account Number"},
        3: {"name": "processing_code", "type": "N", "len": 6, "description": "Processing Code"},
        4: {"name": "amount", "type": "N", "len": 12, "description": "Transaction Amount"},
        7: {"name": "transmission_datetime", "type": "N", "len": 10, "description": "Transmission Date/Time"},
        11: {"name": "stan", "type": "N", "len": 6, "description": "System Trace Audit Number"},
        12: {"name": "local_time", "type": "N", "len": 6, "description": "Local Transaction Time"},
        13: {"name": "local_date", "type": "N", "len": 4, "description": "Local Transaction Date"},
        14: {"name": "expiry_date", "type": "N", "len": 4, "description": "Card Expiry Date"},
        18: {"name": "merchant_type", "type": "N", "len": 4, "description": "Merchant Category Code"},
        22: {"name": "pos_entry_mode", "type": "N", "len": 3, "description": "POS Entry Mode"},
        23: {"name": "card_sequence", "type": "N", "len": 3, "description": "Card Sequence Number"},
        25: {"name": "pos_condition", "type": "N", "len": 2, "description": "POS Condition Code"},
        26: {"name": "pos_pin_capture", "type": "N", "len": 2, "description": "POS PIN Capture Code"},
        32: {"name": "acquiring_inst", "type": "LLVAR", "max_len": 11, "description": "Acquiring Institution ID"},
        35: {"name": "track2", "type": "LLVAR", "max_len": 37, "description": "Track 2 Data"},
        37: {"name": "retrieval_ref", "type": "AN", "len": 12, "description": "Retrieval Reference Number"},
        38: {"name": "auth_code", "type": "AN", "len": 6, "description": "Authorization Code"},
        39: {"name": "response_code", "type": "AN", "len": 2, "description": "Response Code"},
        41: {"name": "terminal_id", "type": "ANS", "len": 8, "description": "Card Acceptor Terminal ID"},
        42: {"name": "merchant_id", "type": "ANS", "len": 15, "description": "Card Acceptor ID"},
        43: {"name": "merchant_name", "type": "ANS", "len": 40, "description": "Card Acceptor Name/Location"},
        49: {"name": "currency_code", "type": "N", "len": 3, "description": "Transaction Currency Code"},
        52: {"name": "pin_block", "type": "B", "len": 8, "description": "PIN Block"},
        55: {"name": "emv_data", "type": "LLLVAR", "max_len": 999, "description": "EMV Data"},
        60: {"name": "additional_data", "type": "LLLVAR", "max_len": 999, "description": "Additional Data"},
        63: {"name": "private_data", "type": "LLLVAR", "max_len": 999, "description": "Private Data"},
    }

    # Response codes
    RESPONSE_CODES = {
        "00": "Approved",
        "01": "Refer to issuer",
        "03": "Invalid merchant",
        "05": "Do not honor",
        "12": "Invalid transaction",
        "13": "Invalid amount",
        "14": "Invalid card number",
        "51": "Insufficient funds",
        "54": "Expired card",
        "55": "Incorrect PIN",
        "61": "Exceeds withdrawal limit",
        "91": "Issuer unavailable",
        "96": "System malfunction",
    }

    # Merchant Category Codes (MCC)
    MCC_CODES = ["5411", "5812", "5912", "5541", "5942", "5311", "5651", "5732", "5999", "4121"]

    # Currency codes
    CURRENCY_CODES = ["840", "978", "826", "392", "156", "036", "124", "756", "702", "344"]

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.stan_counter = random.randint(1, 999999)

    def generate_pan(self) -> str:
        """Generate a valid test PAN with Luhn checksum."""
        # Test card prefixes
        prefixes = ["4111111111", "5500000000", "340000000", "6011000000"]
        prefix = random.choice(prefixes)

        # Generate random digits
        remaining = 15 - len(prefix)
        number = prefix + "".join([str(random.randint(0, 9)) for _ in range(remaining)])

        # Calculate Luhn checksum
        digits = [int(d) for d in number]
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(divmod(d * 2, 10))
        check_digit = (10 - (checksum % 10)) % 10

        return number + str(check_digit)

    def generate_track2(self, pan: str, expiry: str) -> str:
        """Generate Track 2 data."""
        service_code = "101"  # International, normal authorization
        discretionary = "".join([str(random.randint(0, 9)) for _ in range(8)])
        return f"{pan}={expiry}{service_code}{discretionary}"

    def generate_emv_data(self) -> bytes:
        """Generate realistic EMV TLV data."""
        tags = []

        # 9F26 - Application Cryptogram
        ac = bytes([random.randint(0, 255) for _ in range(8)])
        tags.append(b"\x9F\x26\x08" + ac)

        # 9F27 - Cryptogram Information Data
        tags.append(b"\x9F\x27\x01" + bytes([random.choice([0x00, 0x40, 0x80])]))

        # 9F10 - Issuer Application Data
        iad = bytes([random.randint(0, 255) for _ in range(7)])
        tags.append(b"\x9F\x10\x07" + iad)

        # 9F37 - Unpredictable Number
        un = bytes([random.randint(0, 255) for _ in range(4)])
        tags.append(b"\x9F\x37\x04" + un)

        # 9F36 - Application Transaction Counter
        atc = struct.pack(">H", random.randint(1, 65535))
        tags.append(b"\x9F\x36\x02" + atc)

        # 95 - Terminal Verification Results
        tvr = bytes([random.randint(0, 255) for _ in range(5)])
        tags.append(b"\x95\x05" + tvr)

        # 9A - Transaction Date
        today = datetime.now()
        date_bcd = bytes([
            int(str(today.year)[2:4]),
            today.month,
            today.day
        ])
        tags.append(b"\x9A\x03" + date_bcd)

        # 9C - Transaction Type
        tags.append(b"\x9C\x01" + bytes([0x00]))  # Purchase

        return b"".join(tags)

    def encode_bitmap(self, fields: List[int]) -> bytes:
        """Encode field presence as bitmap."""
        bitmap = [0] * 128
        for field in fields:
            if 1 <= field <= 128:
                bitmap[field - 1] = 1

        # Check if secondary bitmap needed
        needs_secondary = any(f > 64 for f in fields)
        if needs_secondary:
            bitmap[0] = 1  # Set bit 1 to indicate secondary bitmap

        # Convert to bytes
        primary = 0
        for i in range(64):
            if bitmap[i]:
                primary |= (1 << (63 - i))

        result = struct.pack(">Q", primary)

        if needs_secondary:
            secondary = 0
            for i in range(64, 128):
                if bitmap[i]:
                    secondary |= (1 << (127 - i))
            result += struct.pack(">Q", secondary)

        return result

    def encode_field(self, field_num: int, value: str) -> Tuple[bytes, Dict]:
        """Encode a single field and return bytes with metadata."""
        spec = self.DATA_ELEMENTS.get(field_num)
        if not spec:
            return b"", {}

        field_type = spec["type"]
        metadata = {
            "field": field_num,
            "name": spec["name"],
            "value": value,
            "type": field_type,
        }

        if field_type == "N":
            # Numeric, right-justified, zero-padded
            length = spec["len"]
            encoded = value.zfill(length).encode("ascii")
            metadata["offset"] = None  # Will be set later
            metadata["length"] = length
            return encoded, metadata

        elif field_type in ("AN", "ANS"):
            # Alphanumeric, left-justified, space-padded
            length = spec["len"]
            encoded = value.ljust(length)[:length].encode("ascii")
            metadata["length"] = length
            return encoded, metadata

        elif field_type == "LLVAR":
            # Variable length with 2-digit length prefix
            max_len = spec["max_len"]
            truncated = value[:max_len]
            length_prefix = f"{len(truncated):02d}".encode("ascii")
            encoded = length_prefix + truncated.encode("ascii")
            metadata["length"] = len(truncated)
            metadata["length_prefix"] = 2
            return encoded, metadata

        elif field_type == "LLLVAR":
            # Variable length with 3-digit length prefix
            max_len = spec["max_len"]
            if isinstance(value, bytes):
                truncated = value[:max_len]
                length_prefix = f"{len(truncated):03d}".encode("ascii")
                encoded = length_prefix + truncated
            else:
                truncated = value[:max_len]
                length_prefix = f"{len(truncated):03d}".encode("ascii")
                encoded = length_prefix + truncated.encode("ascii")
            metadata["length"] = len(truncated) if isinstance(truncated, bytes) else len(truncated)
            metadata["length_prefix"] = 3
            return encoded, metadata

        elif field_type == "B":
            # Binary
            length = spec["len"]
            if isinstance(value, bytes):
                encoded = value[:length].ljust(length, b"\x00")
            else:
                encoded = bytes.fromhex(value)[:length].ljust(length, b"\x00")
            metadata["length"] = length
            return encoded, metadata

        return b"", {}

    def generate_authorization_request(self) -> Tuple[bytes, Dict]:
        """Generate an authorization request (0100)."""
        mti = "0100"
        now = datetime.now()

        # Generate field values
        pan = self.generate_pan()
        expiry = f"{(now.year + random.randint(1, 5)) % 100:02d}{random.randint(1, 12):02d}"
        amount = random.randint(100, 999999)  # Amount in cents

        self.stan_counter = (self.stan_counter + 1) % 1000000

        fields_data = {
            2: pan,
            3: f"{random.choice(['00', '01', '20'])}0000",  # Processing code
            4: str(amount),
            7: now.strftime("%m%d%H%M%S"),
            11: f"{self.stan_counter:06d}",
            12: now.strftime("%H%M%S"),
            13: now.strftime("%m%d"),
            14: expiry,
            18: random.choice(self.MCC_CODES),
            22: f"{random.choice(['05', '07', '90', '91'])}1",
            25: f"{random.randint(0, 99):02d}",
            32: f"{random.randint(100000, 999999)}",
            35: self.generate_track2(pan, expiry),
            41: f"TERM{random.randint(1000, 9999):04d}",
            42: f"MERCH{random.randint(100000000, 999999999):09d}",
            43: f"MERCHANT NAME {random.randint(1, 999):03d}".ljust(40),
            49: random.choice(self.CURRENCY_CODES),
            52: bytes([random.randint(0, 255) for _ in range(8)]),
        }

        # Add EMV data for chip transactions
        if fields_data[22].startswith("05"):
            fields_data[55] = self.generate_emv_data()

        return self._build_message(mti, fields_data)

    def generate_authorization_response(self, request_data: Dict = None) -> Tuple[bytes, Dict]:
        """Generate an authorization response (0110)."""
        mti = "0110"
        now = datetime.now()

        # Use request data if provided, otherwise generate new
        if request_data:
            pan = request_data.get("pan", self.generate_pan())
            amount = request_data.get("amount", random.randint(100, 999999))
            stan = request_data.get("stan", f"{self.stan_counter:06d}")
        else:
            pan = self.generate_pan()
            amount = random.randint(100, 999999)
            self.stan_counter = (self.stan_counter + 1) % 1000000
            stan = f"{self.stan_counter:06d}"

        # Weighted response codes (mostly approvals)
        response_code = random.choices(
            list(self.RESPONSE_CODES.keys()),
            weights=[70, 2, 1, 5, 2, 1, 3, 3, 2, 5, 2, 2, 2],
            k=1
        )[0]

        fields_data = {
            2: pan,
            3: f"{random.choice(['00', '01', '20'])}0000",
            4: str(amount),
            7: now.strftime("%m%d%H%M%S"),
            11: stan,
            12: now.strftime("%H%M%S"),
            13: now.strftime("%m%d"),
            37: f"{random.randint(100000000000, 999999999999)}",
            38: f"{random.randint(100000, 999999):06d}" if response_code == "00" else "",
            39: response_code,
            41: f"TERM{random.randint(1000, 9999):04d}",
            42: f"MERCH{random.randint(100000000, 999999999):09d}",
            49: random.choice(self.CURRENCY_CODES),
        }

        return self._build_message(mti, fields_data)

    def generate_financial_request(self) -> Tuple[bytes, Dict]:
        """Generate a financial request (0200)."""
        mti = "0200"
        now = datetime.now()

        pan = self.generate_pan()
        expiry = f"{(now.year + random.randint(1, 5)) % 100:02d}{random.randint(1, 12):02d}"
        amount = random.randint(100, 9999999)

        self.stan_counter = (self.stan_counter + 1) % 1000000

        fields_data = {
            2: pan,
            3: f"{random.choice(['00', '01', '20', '30'])}0000",
            4: str(amount),
            7: now.strftime("%m%d%H%M%S"),
            11: f"{self.stan_counter:06d}",
            12: now.strftime("%H%M%S"),
            13: now.strftime("%m%d"),
            14: expiry,
            18: random.choice(self.MCC_CODES),
            22: f"{random.choice(['05', '07', '90', '91'])}1",
            23: f"{random.randint(1, 999):03d}",
            25: f"{random.randint(0, 99):02d}",
            26: f"{random.randint(0, 12):02d}",
            32: f"{random.randint(100000, 999999)}",
            35: self.generate_track2(pan, expiry),
            37: f"{random.randint(100000000000, 999999999999)}",
            41: f"TERM{random.randint(1000, 9999):04d}",
            42: f"MERCH{random.randint(100000000, 999999999):09d}",
            43: f"MERCHANT NAME {random.randint(1, 999):03d}".ljust(40),
            49: random.choice(self.CURRENCY_CODES),
            52: bytes([random.randint(0, 255) for _ in range(8)]),
        }

        if fields_data[22].startswith("05"):
            fields_data[55] = self.generate_emv_data()

        return self._build_message(mti, fields_data)

    def generate_reversal_request(self) -> Tuple[bytes, Dict]:
        """Generate a reversal request (0400)."""
        mti = "0400"
        now = datetime.now()

        pan = self.generate_pan()
        amount = random.randint(100, 999999)

        self.stan_counter = (self.stan_counter + 1) % 1000000

        fields_data = {
            2: pan,
            3: f"{random.choice(['00', '20'])}0000",
            4: str(amount),
            7: now.strftime("%m%d%H%M%S"),
            11: f"{self.stan_counter:06d}",
            12: now.strftime("%H%M%S"),
            13: now.strftime("%m%d"),
            32: f"{random.randint(100000, 999999)}",
            37: f"{random.randint(100000000000, 999999999999)}",
            38: f"{random.randint(100000, 999999):06d}",
            41: f"TERM{random.randint(1000, 9999):04d}",
            42: f"MERCH{random.randint(100000000, 999999999):09d}",
            49: random.choice(self.CURRENCY_CODES),
        }

        return self._build_message(mti, fields_data)

    def generate_network_management(self) -> Tuple[bytes, Dict]:
        """Generate a network management request (0800)."""
        mti = "0800"
        now = datetime.now()

        self.stan_counter = (self.stan_counter + 1) % 1000000

        # Network management codes
        network_codes = ["001", "301", "302", "303"]  # Sign-on, echo, key exchange

        fields_data = {
            7: now.strftime("%m%d%H%M%S"),
            11: f"{self.stan_counter:06d}",
            12: now.strftime("%H%M%S"),
            13: now.strftime("%m%d"),
            41: f"TERM{random.randint(1000, 9999):04d}",
            60: random.choice(network_codes),
        }

        return self._build_message(mti, fields_data)

    def _build_message(self, mti: str, fields_data: Dict) -> Tuple[bytes, Dict]:
        """Build complete ISO 8583 message with metadata."""
        message_parts = []
        field_metadata = []
        current_offset = 0

        # MTI
        mti_bytes = mti.encode("ascii")
        message_parts.append(mti_bytes)
        field_metadata.append({
            "field": "mti",
            "name": "message_type_indicator",
            "value": mti,
            "offset": current_offset,
            "length": 4,
            "type": "N"
        })
        current_offset += 4

        # Bitmap
        field_numbers = sorted(fields_data.keys())
        bitmap = self.encode_bitmap(field_numbers)
        message_parts.append(bitmap)
        field_metadata.append({
            "field": "bitmap",
            "name": "bitmap",
            "value": bitmap.hex(),
            "offset": current_offset,
            "length": len(bitmap),
            "type": "B",
            "fields_present": field_numbers
        })
        current_offset += len(bitmap)

        # Data elements
        for field_num in field_numbers:
            value = fields_data[field_num]
            encoded, meta = self.encode_field(field_num, value)
            if encoded:
                message_parts.append(encoded)
                meta["offset"] = current_offset
                field_metadata.append(meta)
                current_offset += len(encoded)

        message = b"".join(message_parts)

        metadata = {
            "protocol": "iso8583",
            "mti": mti,
            "message_type": self.MTI_TYPES.get(mti, "unknown"),
            "mti_description": self.MTI_TYPES.get(mti, "unknown"),
            "timestamp": datetime.now().isoformat(),
            "message_length": len(message),
            "fields": field_metadata,
            "field_count": len(field_numbers),
            "hash": hashlib.sha256(message).hexdigest()
        }

        return message, metadata

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate a complete dataset of ISO 8583 messages."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generators = [
            ("auth_request", self.generate_authorization_request, 0.30),
            ("auth_response", self.generate_authorization_response, 0.25),
            ("financial_request", self.generate_financial_request, 0.25),
            ("reversal_request", self.generate_reversal_request, 0.10),
            ("network_mgmt", self.generate_network_management, 0.10),
        ]

        dataset_metadata = {
            "protocol": "iso8583",
            "version": "1987/1993/2003",
            "total_samples": num_samples,
            "samples_by_type": {},
            "generated_at": datetime.now().isoformat(),
        }

        sample_idx = 0
        for msg_type, generator, ratio in generators:
            count = int(num_samples * ratio)
            dataset_metadata["samples_by_type"][msg_type] = count

            for i in range(count):
                message, metadata = generator()
                metadata["sample_index"] = sample_idx
                metadata["message_type"] = msg_type

                # Save binary message
                bin_path = output_path / f"iso8583_{msg_type}_{sample_idx:06d}.bin"
                with open(bin_path, "wb") as f:
                    f.write(message)

                # Save metadata
                meta_path = output_path / f"iso8583_{msg_type}_{sample_idx:06d}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        # Save dataset metadata
        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate ISO 8583 dataset."""
    generator = ISO8583Generator(seed=42)

    output_dir = Path(__file__).parent.parent / "protocols" / "iso8583"

    print("Generating ISO 8583 dataset...")
    metadata = generator.generate_dataset(
        num_samples=1000,
        output_dir=str(output_dir)
    )

    print(f"Generated {metadata['total_samples']} samples")
    print(f"Output directory: {output_dir}")
    for msg_type, count in metadata["samples_by_type"].items():
        print(f"  - {msg_type}: {count} samples")


if __name__ == "__main__":
    main()
