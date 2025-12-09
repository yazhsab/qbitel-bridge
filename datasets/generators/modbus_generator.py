"""
Modbus Protocol Message Generator

Generates realistic Modbus TCP/RTU messages for training
the protocol discovery and field detection models.

Modbus TCP Structure:
- Transaction ID: 2 bytes
- Protocol ID: 2 bytes (0x0000 for Modbus)
- Length: 2 bytes
- Unit ID: 1 byte
- Function Code: 1 byte
- Data: Variable
"""

import json
import random
import struct
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib


class ModbusGenerator:
    """Generate realistic Modbus TCP/RTU messages for ML training."""

    # Function codes
    FUNCTION_CODES = {
        0x01: {"name": "read_coils", "request_len": 4, "description": "Read Coils"},
        0x02: {"name": "read_discrete_inputs", "request_len": 4, "description": "Read Discrete Inputs"},
        0x03: {"name": "read_holding_registers", "request_len": 4, "description": "Read Holding Registers"},
        0x04: {"name": "read_input_registers", "request_len": 4, "description": "Read Input Registers"},
        0x05: {"name": "write_single_coil", "request_len": 4, "description": "Write Single Coil"},
        0x06: {"name": "write_single_register", "request_len": 4, "description": "Write Single Register"},
        0x0F: {"name": "write_multiple_coils", "request_len": "variable", "description": "Write Multiple Coils"},
        0x10: {"name": "write_multiple_registers", "request_len": "variable", "description": "Write Multiple Registers"},
        0x17: {"name": "read_write_multiple_registers", "request_len": "variable", "description": "Read/Write Multiple Registers"},
    }

    # Exception codes
    EXCEPTION_CODES = {
        0x01: "Illegal Function",
        0x02: "Illegal Data Address",
        0x03: "Illegal Data Value",
        0x04: "Slave Device Failure",
        0x05: "Acknowledge",
        0x06: "Slave Device Busy",
    }

    # Typical register ranges for different device types
    DEVICE_PROFILES = {
        "plc": {
            "coils": (0, 9999),
            "discrete_inputs": (10000, 19999),
            "holding_registers": (40000, 49999),
            "input_registers": (30000, 39999),
        },
        "sensor": {
            "holding_registers": (0, 100),
            "input_registers": (0, 50),
        },
        "motor_drive": {
            "holding_registers": (0, 500),
            "input_registers": (0, 200),
        },
        "power_meter": {
            "holding_registers": (0, 300),
            "input_registers": (0, 150),
        },
    }

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.transaction_id = random.randint(1, 65535)

    def _get_next_transaction_id(self) -> int:
        """Get next transaction ID."""
        self.transaction_id = (self.transaction_id + 1) % 65536
        return self.transaction_id

    def _calculate_crc16(self, data: bytes) -> bytes:
        """Calculate Modbus CRC16."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return struct.pack("<H", crc)

    def generate_read_coils_request(self, unit_id: int = None) -> Tuple[bytes, Dict]:
        """Generate Read Coils request (FC 0x01)."""
        unit_id = unit_id or random.randint(1, 247)
        start_address = random.randint(0, 9900)
        quantity = random.randint(1, min(2000, 9999 - start_address))

        transaction_id = self._get_next_transaction_id()
        protocol_id = 0x0000
        length = 6  # Unit ID + FC + Data

        # Build MBAP header + PDU
        message = struct.pack(
            ">HHHBBHH",
            transaction_id,
            protocol_id,
            length,
            unit_id,
            0x01,  # Function code
            start_address,
            quantity
        )

        metadata = {
            "protocol": "modbus_tcp",
            "message_type": "read_coils_request",
            "timestamp": datetime.now().isoformat(),
            "function_code": 0x01,
            "function_name": "read_coils",
            "transaction_id": transaction_id,
            "unit_id": unit_id,
            "start_address": start_address,
            "quantity": quantity,
            "is_request": True,
            "fields": [
                {"name": "transaction_id", "offset": 0, "length": 2, "value": transaction_id},
                {"name": "protocol_id", "offset": 2, "length": 2, "value": protocol_id},
                {"name": "length", "offset": 4, "length": 2, "value": length},
                {"name": "unit_id", "offset": 6, "length": 1, "value": unit_id},
                {"name": "function_code", "offset": 7, "length": 1, "value": 0x01},
                {"name": "start_address", "offset": 8, "length": 2, "value": start_address},
                {"name": "quantity", "offset": 10, "length": 2, "value": quantity},
            ]
        }

        return message, metadata

    def generate_read_coils_response(self, request_meta: Dict = None) -> Tuple[bytes, Dict]:
        """Generate Read Coils response."""
        if request_meta:
            transaction_id = request_meta.get("transaction_id", self._get_next_transaction_id())
            unit_id = request_meta.get("unit_id", random.randint(1, 247))
            quantity = request_meta.get("quantity", random.randint(1, 100))
        else:
            transaction_id = self._get_next_transaction_id()
            unit_id = random.randint(1, 247)
            quantity = random.randint(1, 100)

        byte_count = (quantity + 7) // 8
        coil_status = bytes([random.randint(0, 255) for _ in range(byte_count)])

        length = 3 + byte_count
        message = struct.pack(
            ">HHHBBB",
            transaction_id,
            0x0000,
            length,
            unit_id,
            0x01,
            byte_count
        ) + coil_status

        metadata = {
            "protocol": "modbus_tcp",
            "message_type": "read_coils_response",
            "timestamp": datetime.now().isoformat(),
            "function_code": 0x01,
            "function_name": "read_coils",
            "transaction_id": transaction_id,
            "unit_id": unit_id,
            "byte_count": byte_count,
            "is_request": False,
            "fields": [
                {"name": "transaction_id", "offset": 0, "length": 2, "value": transaction_id},
                {"name": "protocol_id", "offset": 2, "length": 2, "value": 0},
                {"name": "length", "offset": 4, "length": 2, "value": length},
                {"name": "unit_id", "offset": 6, "length": 1, "value": unit_id},
                {"name": "function_code", "offset": 7, "length": 1, "value": 0x01},
                {"name": "byte_count", "offset": 8, "length": 1, "value": byte_count},
                {"name": "coil_status", "offset": 9, "length": byte_count, "value": coil_status.hex()},
            ]
        }

        return message, metadata

    def generate_read_holding_registers_request(self, unit_id: int = None) -> Tuple[bytes, Dict]:
        """Generate Read Holding Registers request (FC 0x03)."""
        unit_id = unit_id or random.randint(1, 247)
        start_address = random.randint(40000, 49900)
        quantity = random.randint(1, min(125, 49999 - start_address))

        transaction_id = self._get_next_transaction_id()

        message = struct.pack(
            ">HHHBBHH",
            transaction_id,
            0x0000,
            6,
            unit_id,
            0x03,
            start_address,
            quantity
        )

        metadata = {
            "protocol": "modbus_tcp",
            "message_type": "read_holding_registers_request",
            "timestamp": datetime.now().isoformat(),
            "function_code": 0x03,
            "function_name": "read_holding_registers",
            "transaction_id": transaction_id,
            "unit_id": unit_id,
            "start_address": start_address,
            "quantity": quantity,
            "is_request": True,
            "fields": [
                {"name": "transaction_id", "offset": 0, "length": 2, "value": transaction_id},
                {"name": "protocol_id", "offset": 2, "length": 2, "value": 0},
                {"name": "length", "offset": 4, "length": 2, "value": 6},
                {"name": "unit_id", "offset": 6, "length": 1, "value": unit_id},
                {"name": "function_code", "offset": 7, "length": 1, "value": 0x03},
                {"name": "start_address", "offset": 8, "length": 2, "value": start_address},
                {"name": "quantity", "offset": 10, "length": 2, "value": quantity},
            ]
        }

        return message, metadata

    def generate_read_holding_registers_response(self, request_meta: Dict = None) -> Tuple[bytes, Dict]:
        """Generate Read Holding Registers response."""
        if request_meta:
            transaction_id = request_meta.get("transaction_id", self._get_next_transaction_id())
            unit_id = request_meta.get("unit_id", random.randint(1, 247))
            quantity = request_meta.get("quantity", random.randint(1, 50))
        else:
            transaction_id = self._get_next_transaction_id()
            unit_id = random.randint(1, 247)
            quantity = random.randint(1, 50)

        byte_count = quantity * 2
        register_values = bytes([random.randint(0, 255) for _ in range(byte_count)])

        length = 3 + byte_count
        message = struct.pack(
            ">HHHBBB",
            transaction_id,
            0x0000,
            length,
            unit_id,
            0x03,
            byte_count
        ) + register_values

        metadata = {
            "protocol": "modbus_tcp",
            "message_type": "read_holding_registers_response",
            "timestamp": datetime.now().isoformat(),
            "function_code": 0x03,
            "function_name": "read_holding_registers",
            "transaction_id": transaction_id,
            "unit_id": unit_id,
            "byte_count": byte_count,
            "register_count": quantity,
            "is_request": False,
            "fields": [
                {"name": "transaction_id", "offset": 0, "length": 2, "value": transaction_id},
                {"name": "protocol_id", "offset": 2, "length": 2, "value": 0},
                {"name": "length", "offset": 4, "length": 2, "value": length},
                {"name": "unit_id", "offset": 6, "length": 1, "value": unit_id},
                {"name": "function_code", "offset": 7, "length": 1, "value": 0x03},
                {"name": "byte_count", "offset": 8, "length": 1, "value": byte_count},
                {"name": "register_values", "offset": 9, "length": byte_count, "value": register_values.hex()},
            ]
        }

        return message, metadata

    def generate_write_single_register_request(self, unit_id: int = None) -> Tuple[bytes, Dict]:
        """Generate Write Single Register request (FC 0x06)."""
        unit_id = unit_id or random.randint(1, 247)
        register_address = random.randint(40000, 49999)
        register_value = random.randint(0, 65535)

        transaction_id = self._get_next_transaction_id()

        message = struct.pack(
            ">HHHBBHH",
            transaction_id,
            0x0000,
            6,
            unit_id,
            0x06,
            register_address,
            register_value
        )

        metadata = {
            "protocol": "modbus_tcp",
            "message_type": "write_single_register_request",
            "timestamp": datetime.now().isoformat(),
            "function_code": 0x06,
            "function_name": "write_single_register",
            "transaction_id": transaction_id,
            "unit_id": unit_id,
            "register_address": register_address,
            "register_value": register_value,
            "is_request": True,
            "fields": [
                {"name": "transaction_id", "offset": 0, "length": 2, "value": transaction_id},
                {"name": "protocol_id", "offset": 2, "length": 2, "value": 0},
                {"name": "length", "offset": 4, "length": 2, "value": 6},
                {"name": "unit_id", "offset": 6, "length": 1, "value": unit_id},
                {"name": "function_code", "offset": 7, "length": 1, "value": 0x06},
                {"name": "register_address", "offset": 8, "length": 2, "value": register_address},
                {"name": "register_value", "offset": 10, "length": 2, "value": register_value},
            ]
        }

        return message, metadata

    def generate_write_multiple_registers_request(self, unit_id: int = None) -> Tuple[bytes, Dict]:
        """Generate Write Multiple Registers request (FC 0x10)."""
        unit_id = unit_id or random.randint(1, 247)
        start_address = random.randint(40000, 49900)
        quantity = random.randint(1, min(123, 49999 - start_address))
        byte_count = quantity * 2
        register_values = bytes([random.randint(0, 255) for _ in range(byte_count)])

        transaction_id = self._get_next_transaction_id()
        length = 7 + byte_count

        message = struct.pack(
            ">HHHBBHHB",
            transaction_id,
            0x0000,
            length,
            unit_id,
            0x10,
            start_address,
            quantity,
            byte_count
        ) + register_values

        metadata = {
            "protocol": "modbus_tcp",
            "message_type": "write_multiple_registers_request",
            "timestamp": datetime.now().isoformat(),
            "function_code": 0x10,
            "function_name": "write_multiple_registers",
            "transaction_id": transaction_id,
            "unit_id": unit_id,
            "start_address": start_address,
            "quantity": quantity,
            "byte_count": byte_count,
            "is_request": True,
            "fields": [
                {"name": "transaction_id", "offset": 0, "length": 2, "value": transaction_id},
                {"name": "protocol_id", "offset": 2, "length": 2, "value": 0},
                {"name": "length", "offset": 4, "length": 2, "value": length},
                {"name": "unit_id", "offset": 6, "length": 1, "value": unit_id},
                {"name": "function_code", "offset": 7, "length": 1, "value": 0x10},
                {"name": "start_address", "offset": 8, "length": 2, "value": start_address},
                {"name": "quantity", "offset": 10, "length": 2, "value": quantity},
                {"name": "byte_count", "offset": 12, "length": 1, "value": byte_count},
                {"name": "register_values", "offset": 13, "length": byte_count, "value": register_values.hex()},
            ]
        }

        return message, metadata

    def generate_exception_response(self, function_code: int = None, unit_id: int = None) -> Tuple[bytes, Dict]:
        """Generate Modbus exception response."""
        unit_id = unit_id or random.randint(1, 247)
        function_code = function_code or random.choice([0x01, 0x03, 0x06, 0x10])
        exception_code = random.choice(list(self.EXCEPTION_CODES.keys()))

        transaction_id = self._get_next_transaction_id()

        message = struct.pack(
            ">HHHBBB",
            transaction_id,
            0x0000,
            3,
            unit_id,
            function_code | 0x80,  # Exception flag
            exception_code
        )

        metadata = {
            "protocol": "modbus_tcp",
            "message_type": "exception_response",
            "timestamp": datetime.now().isoformat(),
            "function_code": function_code | 0x80,
            "function_name": f"exception_{self.FUNCTION_CODES.get(function_code, {}).get('name', 'unknown')}",
            "transaction_id": transaction_id,
            "unit_id": unit_id,
            "exception_code": exception_code,
            "exception_message": self.EXCEPTION_CODES.get(exception_code, "Unknown"),
            "is_request": False,
            "is_exception": True,
            "fields": [
                {"name": "transaction_id", "offset": 0, "length": 2, "value": transaction_id},
                {"name": "protocol_id", "offset": 2, "length": 2, "value": 0},
                {"name": "length", "offset": 4, "length": 2, "value": 3},
                {"name": "unit_id", "offset": 6, "length": 1, "value": unit_id},
                {"name": "function_code", "offset": 7, "length": 1, "value": function_code | 0x80},
                {"name": "exception_code", "offset": 8, "length": 1, "value": exception_code},
            ]
        }

        return message, metadata

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate a complete dataset of Modbus messages."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generators = [
            ("read_coils_req", self.generate_read_coils_request, 0.15),
            ("read_coils_resp", self.generate_read_coils_response, 0.10),
            ("read_holding_req", self.generate_read_holding_registers_request, 0.20),
            ("read_holding_resp", self.generate_read_holding_registers_response, 0.15),
            ("write_single_req", self.generate_write_single_register_request, 0.15),
            ("write_multiple_req", self.generate_write_multiple_registers_request, 0.15),
            ("exception", self.generate_exception_response, 0.10),
        ]

        dataset_metadata = {
            "protocol": "modbus_tcp",
            "version": "TCP/IP",
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
                metadata["hash"] = hashlib.sha256(message).hexdigest()

                bin_path = output_path / f"modbus_{msg_type}_{sample_idx:06d}.bin"
                with open(bin_path, "wb") as f:
                    f.write(message)

                meta_path = output_path / f"modbus_{msg_type}_{sample_idx:06d}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate Modbus dataset."""
    generator = ModbusGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "protocols" / "modbus"

    print("Generating Modbus TCP dataset...")
    metadata = generator.generate_dataset(num_samples=1000, output_dir=str(output_dir))

    print(f"Generated {metadata['total_samples']} samples")
    print(f"Output directory: {output_dir}")
    for msg_type, count in metadata["samples_by_type"].items():
        print(f"  - {msg_type}: {count} samples")


if __name__ == "__main__":
    main()
