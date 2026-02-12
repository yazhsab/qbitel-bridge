"""
Secure Modbus Communications

PQC-enhanced security for Modbus protocol:
- Modbus/TCP
- Modbus RTU over TCP
- Modbus RTU over serial (with gateway)

Based on:
- Modbus/TCP Security (draft)
- IEC 62351 principles

Challenges:
- Original Modbus has no security
- Very limited message size (256 bytes max)
- Real-time requirements
- Thousands of deployed devices
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Callable

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Metrics
MODBUS_AUTH_OPS = Counter(
    "modbus_secure_auth_operations_total", "Total Modbus secure operations", ["operation", "function_code"]
)

MODBUS_AUTH_LATENCY = Histogram("modbus_secure_latency_ms", "Modbus secure operation latency", buckets=[0.1, 0.5, 1, 2, 5, 10])

MODBUS_ACTIVE_SESSIONS = Gauge("modbus_secure_active_sessions", "Number of active Modbus secure sessions")


class ModbusFunctionCode(Enum):
    """Modbus function codes."""

    READ_COILS = 0x01
    READ_DISCRETE_INPUTS = 0x02
    READ_HOLDING_REGISTERS = 0x03
    READ_INPUT_REGISTERS = 0x04
    WRITE_SINGLE_COIL = 0x05
    WRITE_SINGLE_REGISTER = 0x06
    WRITE_MULTIPLE_COILS = 0x0F
    WRITE_MULTIPLE_REGISTERS = 0x10

    # Security extension function codes
    SECURITY_HANDSHAKE = 0x5A  # Custom: Security handshake
    AUTHENTICATED_READ = 0x5B  # Custom: Authenticated read
    AUTHENTICATED_WRITE = 0x5C  # Custom: Authenticated write


class ModbusRole(Enum):
    """Modbus security roles."""

    VIEWER = auto()  # Read-only access
    OPERATOR = auto()  # Read + limited write
    ENGINEER = auto()  # Full read/write
    ADMIN = auto()  # Full access + config


@dataclass
class ModbusSecurityProfile:
    """Security profile for Modbus device."""

    device_id: str
    unit_id: int
    role: ModbusRole

    # Security settings
    require_authentication: bool = True
    require_encryption: bool = False
    max_session_age: int = 3600
    max_messages_per_session: int = 100000

    # Function code permissions
    allowed_function_codes: List[ModbusFunctionCode] = field(default_factory=list)


@dataclass
class ModbusSecureSession:
    """Secure Modbus session."""

    session_id: bytes
    client_id: str
    server_unit_id: int
    profile: ModbusSecurityProfile

    # Key material
    authentication_key: bytes
    encryption_key: Optional[bytes] = None

    # Session state
    sequence_number: int = 0
    created_at: float = field(default_factory=time.time)
    message_count: int = 0

    def is_valid(self) -> bool:
        """Check if session is still valid."""
        age = time.time() - self.created_at
        return age < self.profile.max_session_age and self.message_count < self.profile.max_messages_per_session

    def next_sequence(self) -> int:
        """Get next sequence number."""
        seq = self.sequence_number
        self.sequence_number = (self.sequence_number + 1) & 0xFFFFFFFF
        self.message_count += 1
        return seq


@dataclass
class ModbusSecureMessage:
    """Secure Modbus message."""

    unit_id: int
    function_code: ModbusFunctionCode
    data: bytes
    session_id: bytes
    sequence: int
    mac: bytes


class ModbusAuthenticator:
    """
    Authenticator for Modbus security.

    Provides:
    - Session establishment
    - Message authentication
    - Optional encryption
    """

    def __init__(self, use_pqc: bool = True):
        self.use_pqc = use_pqc
        self._sessions: Dict[bytes, ModbusSecureSession] = {}

        logger.info(f"Modbus authenticator initialized: PQC={use_pqc}")

    async def create_session(
        self,
        client_id: str,
        profile: ModbusSecurityProfile,
    ) -> ModbusSecureSession:
        """
        Create secure session.

        In production, would involve key exchange with server.
        """
        session_id = secrets.token_bytes(8)

        if self.use_pqc:
            master_key = await self._pqc_key_exchange()
        else:
            master_key = secrets.token_bytes(32)

        # Derive session keys
        auth_key = hashlib.sha256(master_key + b"auth").digest()
        enc_key = hashlib.sha256(master_key + b"enc").digest() if profile.require_encryption else None

        session = ModbusSecureSession(
            session_id=session_id,
            client_id=client_id,
            server_unit_id=profile.unit_id,
            profile=profile,
            authentication_key=auth_key,
            encryption_key=enc_key,
        )

        self._sessions[session_id] = session
        MODBUS_ACTIVE_SESSIONS.set(len(self._sessions))

        logger.debug(f"Created Modbus session: {session_id.hex()}")
        return session

    async def _pqc_key_exchange(self) -> bytes:
        """Perform PQC key exchange."""
        from ai_engine.crypto.mlkem import MlKemEngine, MlKemSecurityLevel

        engine = MlKemEngine(MlKemSecurityLevel.MLKEM_512)  # Use 512 for constrained devices
        keypair = await engine.generate_keypair()
        ct, ss = await engine.encapsulate(keypair.public_key)
        return ss.data

    async def authenticate_request(
        self,
        session: ModbusSecureSession,
        function_code: ModbusFunctionCode,
        data: bytes,
    ) -> ModbusSecureMessage:
        """
        Create authenticated Modbus request.

        Args:
            session: Active session
            function_code: Modbus function code
            data: Request data

        Returns:
            Authenticated message
        """
        if not session.is_valid():
            raise ValueError("Session expired")

        # Check function code permission
        if session.profile.allowed_function_codes:
            if function_code not in session.profile.allowed_function_codes:
                raise PermissionError(f"Function code not allowed: {function_code.name}")

        sequence = session.next_sequence()

        # Build MAC input
        mac_input = (
            session.session_id
            + session.server_unit_id.to_bytes(1, "big")
            + function_code.value.to_bytes(1, "big")
            + sequence.to_bytes(4, "big")
            + data
        )

        mac = hmac.new(session.authentication_key, mac_input, hashlib.sha256).digest()[
            :8
        ]  # Truncate to 8 bytes for Modbus constraints

        MODBUS_AUTH_OPS.labels(operation="authenticate_request", function_code=function_code.name).inc()

        return ModbusSecureMessage(
            unit_id=session.server_unit_id,
            function_code=function_code,
            data=data,
            session_id=session.session_id,
            sequence=sequence,
            mac=mac,
        )

    async def verify_request(
        self,
        message: ModbusSecureMessage,
    ) -> bool:
        """
        Verify authenticated request.

        Returns True if authentication is valid.
        """
        session = self._sessions.get(message.session_id)
        if not session or not session.is_valid():
            logger.warning(f"Invalid or expired session: {message.session_id.hex()}")
            return False

        # Verify MAC
        mac_input = (
            message.session_id
            + message.unit_id.to_bytes(1, "big")
            + message.function_code.value.to_bytes(1, "big")
            + message.sequence.to_bytes(4, "big")
            + message.data
        )

        expected_mac = hmac.new(session.authentication_key, mac_input, hashlib.sha256).digest()[:8]

        if not secrets.compare_digest(message.mac, expected_mac):
            logger.warning("MAC verification failed")
            return False

        MODBUS_AUTH_OPS.labels(operation="verify_request", function_code=message.function_code.name).inc()

        return True

    def get_session(self, session_id: bytes) -> Optional[ModbusSecureSession]:
        """Get session by ID."""
        return self._sessions.get(session_id)


class SecureModbusClient:
    """
    Secure Modbus TCP client.

    Wraps standard Modbus client with security.
    """

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 1,
        use_pqc: bool = True,
    ):
        self.host = host
        self.port = port
        self.unit_id = unit_id

        self.authenticator = ModbusAuthenticator(use_pqc=use_pqc)
        self._session: Optional[ModbusSecureSession] = None

        logger.info(f"Secure Modbus client: {host}:{port}")

    async def connect(
        self,
        client_id: str,
        role: ModbusRole = ModbusRole.OPERATOR,
    ) -> None:
        """Establish secure connection."""
        profile = ModbusSecurityProfile(
            device_id=f"{self.host}:{self.unit_id}",
            unit_id=self.unit_id,
            role=role,
            allowed_function_codes=self._get_role_permissions(role),
        )

        self._session = await self.authenticator.create_session(client_id, profile)
        logger.info(f"Connected to {self.host}:{self.port}")

    def _get_role_permissions(self, role: ModbusRole) -> List[ModbusFunctionCode]:
        """Get allowed function codes for role."""
        if role == ModbusRole.VIEWER:
            return [
                ModbusFunctionCode.READ_COILS,
                ModbusFunctionCode.READ_DISCRETE_INPUTS,
                ModbusFunctionCode.READ_HOLDING_REGISTERS,
                ModbusFunctionCode.READ_INPUT_REGISTERS,
            ]
        elif role == ModbusRole.OPERATOR:
            return [
                ModbusFunctionCode.READ_COILS,
                ModbusFunctionCode.READ_DISCRETE_INPUTS,
                ModbusFunctionCode.READ_HOLDING_REGISTERS,
                ModbusFunctionCode.READ_INPUT_REGISTERS,
                ModbusFunctionCode.WRITE_SINGLE_COIL,
                ModbusFunctionCode.WRITE_SINGLE_REGISTER,
            ]
        else:
            return list(ModbusFunctionCode)

    async def read_holding_registers(
        self,
        address: int,
        count: int,
    ) -> List[int]:
        """Read holding registers with authentication."""
        if not self._session:
            raise RuntimeError("Not connected")

        # Build request data
        data = struct.pack(">HH", address, count)

        message = await self.authenticator.authenticate_request(
            self._session,
            ModbusFunctionCode.READ_HOLDING_REGISTERS,
            data,
        )

        # In production, would send to server and parse response
        # Here we return placeholder
        return [0] * count

    async def write_single_register(
        self,
        address: int,
        value: int,
    ) -> bool:
        """Write single register with authentication."""
        if not self._session:
            raise RuntimeError("Not connected")

        data = struct.pack(">HH", address, value)

        message = await self.authenticator.authenticate_request(
            self._session,
            ModbusFunctionCode.WRITE_SINGLE_REGISTER,
            data,
        )

        MODBUS_AUTH_OPS.labels(operation="write", function_code="WRITE_SINGLE_REGISTER").inc()

        return True

    async def disconnect(self) -> None:
        """Close connection."""
        self._session = None
        logger.info(f"Disconnected from {self.host}:{self.port}")


class SecureModbusServer:
    """
    Secure Modbus TCP server.

    Handles authenticated requests from clients.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 502,
        unit_id: int = 1,
        use_pqc: bool = True,
    ):
        self.host = host
        self.port = port
        self.unit_id = unit_id

        self.authenticator = ModbusAuthenticator(use_pqc=use_pqc)
        self._handlers: Dict[ModbusFunctionCode, Callable] = {}

        # Simulated register storage
        self._holding_registers: Dict[int, int] = {}
        self._input_registers: Dict[int, int] = {}
        self._coils: Dict[int, bool] = {}

        logger.info(f"Secure Modbus server: {host}:{port}")

    def register_handler(
        self,
        function_code: ModbusFunctionCode,
        handler: Callable,
    ) -> None:
        """Register handler for function code."""
        self._handlers[function_code] = handler

    async def handle_request(
        self,
        message: ModbusSecureMessage,
    ) -> Optional[bytes]:
        """
        Handle authenticated request.

        Returns response data if valid, None if authentication fails.
        """
        start = time.time()

        # Verify authentication
        if not await self.authenticator.verify_request(message):
            return None

        # Get handler
        handler = self._handlers.get(message.function_code)
        if handler:
            response = await handler(message.data)
        else:
            response = await self._default_handler(message)

        elapsed_ms = (time.time() - start) * 1000
        MODBUS_AUTH_LATENCY.observe(elapsed_ms)

        return response

    async def _default_handler(
        self,
        message: ModbusSecureMessage,
    ) -> bytes:
        """Default handler for standard function codes."""
        fc = message.function_code

        if fc == ModbusFunctionCode.READ_HOLDING_REGISTERS:
            address, count = struct.unpack(">HH", message.data[:4])
            values = [self._holding_registers.get(address + i, 0) for i in range(count)]
            return struct.pack(">" + "H" * count, *values)

        elif fc == ModbusFunctionCode.WRITE_SINGLE_REGISTER:
            address, value = struct.unpack(">HH", message.data[:4])
            self._holding_registers[address] = value
            return message.data

        elif fc == ModbusFunctionCode.READ_COILS:
            address, count = struct.unpack(">HH", message.data[:4])
            # Pack coil values
            values = [self._coils.get(address + i, False) for i in range(count)]
            # Convert to bytes
            byte_count = (count + 7) // 8
            result = bytearray(byte_count)
            for i, v in enumerate(values):
                if v:
                    result[i // 8] |= 1 << (i % 8)
            return bytes([byte_count]) + bytes(result)

        return b""

    def set_register(self, address: int, value: int) -> None:
        """Set holding register value."""
        self._holding_registers[address] = value

    def set_coil(self, address: int, value: bool) -> None:
        """Set coil value."""
        self._coils[address] = value


def create_modbus_security_profile(
    device_id: str,
    unit_id: int,
    role: ModbusRole,
    require_encryption: bool = False,
) -> ModbusSecurityProfile:
    """Create Modbus security profile with sensible defaults."""
    allowed_codes = []

    if role in (ModbusRole.VIEWER, ModbusRole.OPERATOR, ModbusRole.ENGINEER, ModbusRole.ADMIN):
        allowed_codes.extend(
            [
                ModbusFunctionCode.READ_COILS,
                ModbusFunctionCode.READ_DISCRETE_INPUTS,
                ModbusFunctionCode.READ_HOLDING_REGISTERS,
                ModbusFunctionCode.READ_INPUT_REGISTERS,
            ]
        )

    if role in (ModbusRole.OPERATOR, ModbusRole.ENGINEER, ModbusRole.ADMIN):
        allowed_codes.extend(
            [
                ModbusFunctionCode.WRITE_SINGLE_COIL,
                ModbusFunctionCode.WRITE_SINGLE_REGISTER,
            ]
        )

    if role in (ModbusRole.ENGINEER, ModbusRole.ADMIN):
        allowed_codes.extend(
            [
                ModbusFunctionCode.WRITE_MULTIPLE_COILS,
                ModbusFunctionCode.WRITE_MULTIPLE_REGISTERS,
            ]
        )

    return ModbusSecurityProfile(
        device_id=device_id,
        unit_id=unit_id,
        role=role,
        require_authentication=True,
        require_encryption=require_encryption,
        allowed_function_codes=allowed_codes,
    )
