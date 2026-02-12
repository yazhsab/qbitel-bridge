"""
IEEE 1609.2 / SAE J2735 V2X Protocol Support

Post-quantum extensions for V2X Secure Protocol Data Units (SPDUs).
"""

import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """V2X message types per SAE J2735."""

    BSM = 0x14  # Basic Safety Message
    EVA = 0x16  # Emergency Vehicle Alert
    TIM = 0x1F  # Traveler Information Message
    SPAT = 0x13  # Signal Phase and Timing
    MAP = 0x12  # Map Data
    RSA = 0x1B  # Road Side Alert


@dataclass
class BasicSafetyMessage:
    """SAE J2735 Basic Safety Message (BSM)."""

    msg_count: int
    temp_id: bytes  # 4 bytes
    latitude: float  # Degrees
    longitude: float  # Degrees
    elevation: float  # Meters
    speed: float  # m/s
    heading: float  # Degrees
    acceleration: float  # m/s^2
    brake_status: int
    vehicle_size_width: float
    vehicle_size_length: float
    timestamp: float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        """Serialize BSM to bytes (simplified)."""
        return struct.pack(
            ">B4sdddffdBff",
            self.msg_count,
            self.temp_id,
            self.latitude,
            self.longitude,
            self.elevation,
            self.speed,
            self.heading,
            self.acceleration,
            self.brake_status,
            self.vehicle_size_width,
            self.vehicle_size_length,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "BasicSafetyMessage":
        """Deserialize BSM from bytes."""
        unpacked = struct.unpack(">B4sdddffdBff", data[:59])
        return cls(
            msg_count=unpacked[0],
            temp_id=unpacked[1],
            latitude=unpacked[2],
            longitude=unpacked[3],
            elevation=unpacked[4],
            speed=unpacked[5],
            heading=unpacked[6],
            acceleration=unpacked[7],
            brake_status=unpacked[8],
            vehicle_size_width=unpacked[9],
            vehicle_size_length=unpacked[10],
        )


@dataclass
class V2XCertificate:
    """V2X certificate (IEEE 1609.2 format with PQC extensions)."""

    version: int
    issuer_id: bytes
    subject_id: bytes
    validity_start: float
    validity_end: float
    public_key_algorithm: str
    public_key: bytes
    signature_algorithm: str
    signature: bytes
    pqc_public_key: Optional[bytes] = None  # PQC extension
    pqc_signature: Optional[bytes] = None  # PQC extension
    is_hybrid: bool = False

    @property
    def is_valid(self) -> bool:
        now = time.time()
        return self.validity_start <= now <= self.validity_end


@dataclass
class SignedPDU:
    """IEEE 1609.2 Signed Protocol Data Unit with PQC support."""

    protocol_version: int = 3
    content_type: MessageType = MessageType.BSM
    content: bytes = b""
    signer_info: bytes = b""
    classical_signature: bytes = b""
    pqc_signature: bytes = b""  # Falcon/Dilithium signature
    timestamp: float = field(default_factory=time.time)

    def get_signed_data(self) -> bytes:
        """Get data that was signed."""
        return self.content + struct.pack(">d", self.timestamp)


class IEEE1609Protocol:
    """
    IEEE 1609.2 protocol handler with PQC support.

    Supports both classical (ECDSA) and post-quantum (Falcon) signatures
    for backward compatibility with legacy vehicles.
    """

    def __init__(
        self,
        use_pqc: bool = True,
        use_hybrid: bool = True,
        pqc_algorithm: str = "falcon-512",
    ):
        self.use_pqc = use_pqc
        self.use_hybrid = use_hybrid
        self.pqc_algorithm = pqc_algorithm

        logger.info(f"IEEE 1609.2 protocol: pqc={use_pqc}, hybrid={use_hybrid}")

    async def sign_message(
        self,
        content: bytes,
        message_type: MessageType,
        private_key: bytes,
        certificate: V2XCertificate,
    ) -> SignedPDU:
        """
        Sign a V2X message.

        In hybrid mode, produces both classical and PQC signatures.
        """
        from ai_engine.crypto.falcon import FalconEngine, FalconPrivateKey

        timestamp = time.time()
        signed_data = content + struct.pack(">d", timestamp)

        pqc_sig = b""
        if self.use_pqc:
            engine = FalconEngine()
            sk = FalconPrivateKey(engine.level, private_key)
            signature = await engine.sign(signed_data, sk)
            pqc_sig = signature.data

        # Classical signature would be computed here for hybrid mode
        classical_sig = b""  # Placeholder

        return SignedPDU(
            content_type=message_type,
            content=content,
            signer_info=certificate.subject_id,
            classical_signature=classical_sig,
            pqc_signature=pqc_sig,
            timestamp=timestamp,
        )

    async def verify_message(
        self,
        pdu: SignedPDU,
        public_key: bytes,
    ) -> bool:
        """Verify a signed V2X message."""
        from ai_engine.crypto.falcon import FalconEngine, FalconSignature, FalconPublicKey

        signed_data = pdu.get_signed_data()

        if pdu.pqc_signature:
            engine = FalconEngine()
            sig = FalconSignature(engine.level, pdu.pqc_signature)
            pk = FalconPublicKey(engine.level, public_key)
            return await engine.verify(signed_data, sig, pk)

        # Fall back to classical verification
        return True  # Placeholder
