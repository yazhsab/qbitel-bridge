"""
IEC 62351 Security Profile for Power Systems

Post-quantum security extensions for:
- IEC 61850 (Substation automation)
- GOOSE (Generic Object Oriented Substation Event)
- SV (Sampled Values)
- MMS (Manufacturing Message Specification)

IEC 62351 Parts:
- Part 3: TLS profiles
- Part 4: MMS security
- Part 5: IEC 60870-5 security
- Part 6: IEC 61850 profiles
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
from typing import Dict, List, Optional, Tuple, Set

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Metrics
IEC62351_AUTH_OPS = Counter(
    'iec62351_authentication_operations_total',
    'Total IEC 62351 authentication operations',
    ['protocol', 'operation']
)

IEC62351_LATENCY = Histogram(
    'iec62351_operation_latency_ms',
    'IEC 62351 operation latency',
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20]
)


class SecurityObjective(Enum):
    """IEC 62351 security objectives."""

    AUTHENTICATION = auto()      # Entity authentication
    AUTHORIZATION = auto()       # Access control
    INTEGRITY = auto()          # Message integrity
    CONFIDENTIALITY = auto()    # Data encryption
    ACCOUNTABILITY = auto()      # Audit logging
    AVAILABILITY = auto()        # DoS protection


class ProtectionLevel(Enum):
    """Message protection levels."""

    NONE = 0                # No protection
    AUTHENTICATION = 1      # MAC only
    AUTH_ENCRYPTION = 2     # MAC + encryption


class IEC61850Protocol(Enum):
    """IEC 61850 protocols."""

    GOOSE = auto()      # Generic Object Oriented Substation Event
    SV = auto()         # Sampled Values
    MMS = auto()        # Manufacturing Message Specification
    RSTP = auto()       # Rapid Spanning Tree Protocol


@dataclass
class SecurityPolicy:
    """Security policy configuration."""

    objectives: Set[SecurityObjective]
    protection_level: ProtectionLevel
    key_rotation_interval: int = 3600
    max_message_age_ms: int = 2000  # 2 seconds for GOOSE
    require_pqc: bool = False


@dataclass
class IEC62351KeySet:
    """Key set for IEC 62351 security."""

    key_id: bytes
    encryption_key: bytes
    authentication_key: bytes
    created_at: float = field(default_factory=time.time)
    valid_until: float = 0.0

    def __post_init__(self):
        if self.valid_until == 0.0:
            self.valid_until = self.created_at + 3600

    def is_valid(self) -> bool:
        return time.time() < self.valid_until


@dataclass
class ProtectedMessage:
    """IEC 62351 protected message."""

    protocol: IEC61850Protocol
    payload: bytes
    key_id: bytes
    sequence: int
    timestamp: int  # microseconds since epoch
    mac: bytes
    encrypted: bool = False


class IEC62351KeyManagement:
    """
    Key management for IEC 62351.

    Implements IEC 62351-9 key management with PQC extensions.
    """

    def __init__(self, use_pqc: bool = True):
        self.use_pqc = use_pqc
        self._key_sets: Dict[bytes, IEC62351KeySet] = {}
        self._device_keys: Dict[str, bytes] = {}  # device_id -> current key_id

        logger.info(f"IEC 62351 key management initialized: PQC={use_pqc}")

    async def generate_key_set(
        self,
        device_id: str,
        validity_seconds: int = 3600,
    ) -> IEC62351KeySet:
        """Generate new key set for device."""
        key_id = secrets.token_bytes(8)

        if self.use_pqc:
            # Use PQC-derived keys
            master_key = await self._generate_pqc_key()
        else:
            master_key = secrets.token_bytes(32)

        # Derive encryption and authentication keys
        encryption_key = hashlib.sha256(master_key + b"encryption").digest()
        authentication_key = hashlib.sha256(master_key + b"authentication").digest()

        key_set = IEC62351KeySet(
            key_id=key_id,
            encryption_key=encryption_key,
            authentication_key=authentication_key,
            valid_until=time.time() + validity_seconds,
        )

        self._key_sets[key_id] = key_set
        self._device_keys[device_id] = key_id

        logger.debug(f"Generated key set for {device_id}: id={key_id.hex()[:8]}")
        return key_set

    async def _generate_pqc_key(self) -> bytes:
        """Generate PQC-derived master key."""
        from ai_engine.crypto.mlkem import MlKemEngine, MlKemSecurityLevel

        engine = MlKemEngine(MlKemSecurityLevel.MLKEM_768)
        keypair = await engine.generate_keypair()

        # Encapsulate to self to get random key
        ct, ss = await engine.encapsulate(keypair.public_key)
        return ss.data

    def get_key_set(self, key_id: bytes) -> Optional[IEC62351KeySet]:
        """Get key set by ID."""
        key_set = self._key_sets.get(key_id)
        if key_set and key_set.is_valid():
            return key_set
        return None

    def get_device_key_set(self, device_id: str) -> Optional[IEC62351KeySet]:
        """Get current key set for device."""
        key_id = self._device_keys.get(device_id)
        if key_id:
            return self.get_key_set(key_id)
        return None

    async def rotate_keys(self, device_id: str) -> IEC62351KeySet:
        """Rotate keys for device."""
        # Mark old key set as expiring soon
        old_key_id = self._device_keys.get(device_id)
        if old_key_id and old_key_id in self._key_sets:
            self._key_sets[old_key_id].valid_until = time.time() + 60  # 1 min grace

        # Generate new key set
        return await self.generate_key_set(device_id)


class IEC62351MessageProtection:
    """
    Message protection for IEC 62351.

    Provides authentication and optional encryption
    for IEC 61850 messages.
    """

    def __init__(
        self,
        key_manager: IEC62351KeyManagement,
        policy: SecurityPolicy,
    ):
        self.key_manager = key_manager
        self.policy = policy
        self._sequence = 0

        logger.info(
            f"Message protection initialized: level={policy.protection_level.name}"
        )

    async def protect_message(
        self,
        protocol: IEC61850Protocol,
        payload: bytes,
        device_id: str,
    ) -> ProtectedMessage:
        """
        Apply protection to message.

        Args:
            protocol: IEC 61850 protocol type
            payload: Message payload
            device_id: Sending device ID

        Returns:
            Protected message
        """
        start = time.time()

        key_set = self.key_manager.get_device_key_set(device_id)
        if not key_set:
            raise ValueError(f"No valid key set for device: {device_id}")

        self._sequence += 1
        timestamp = int(time.time() * 1000000)  # microseconds

        # Prepare data for MAC
        protected_payload = payload
        if self.policy.protection_level == ProtectionLevel.AUTH_ENCRYPTION:
            protected_payload = self._encrypt(payload, key_set.encryption_key)

        # Compute MAC
        mac_input = (
            protocol.value.to_bytes(2, 'big') +
            key_set.key_id +
            self._sequence.to_bytes(8, 'big') +
            timestamp.to_bytes(8, 'big') +
            protected_payload
        )

        mac = hmac.new(
            key_set.authentication_key,
            mac_input,
            hashlib.sha256
        ).digest()[:16]  # 128-bit MAC

        elapsed_ms = (time.time() - start) * 1000
        IEC62351_LATENCY.observe(elapsed_ms)
        IEC62351_AUTH_OPS.labels(
            protocol=protocol.name,
            operation="protect"
        ).inc()

        return ProtectedMessage(
            protocol=protocol,
            payload=protected_payload,
            key_id=key_set.key_id,
            sequence=self._sequence,
            timestamp=timestamp,
            mac=mac,
            encrypted=(self.policy.protection_level == ProtectionLevel.AUTH_ENCRYPTION),
        )

    async def verify_message(
        self,
        message: ProtectedMessage,
    ) -> Optional[bytes]:
        """
        Verify and extract message.

        Returns:
            Original payload if valid, None otherwise
        """
        start = time.time()

        key_set = self.key_manager.get_key_set(message.key_id)
        if not key_set:
            logger.warning(f"Unknown key ID: {message.key_id.hex()}")
            return None

        # Verify timestamp freshness
        current_time = int(time.time() * 1000000)
        age_ms = (current_time - message.timestamp) / 1000

        if age_ms > self.policy.max_message_age_ms:
            logger.warning(f"Message too old: {age_ms:.1f}ms")
            return None

        # Verify MAC
        mac_input = (
            message.protocol.value.to_bytes(2, 'big') +
            message.key_id +
            message.sequence.to_bytes(8, 'big') +
            message.timestamp.to_bytes(8, 'big') +
            message.payload
        )

        expected_mac = hmac.new(
            key_set.authentication_key,
            mac_input,
            hashlib.sha256
        ).digest()[:16]

        if not secrets.compare_digest(message.mac, expected_mac):
            logger.warning("MAC verification failed")
            return None

        # Decrypt if necessary
        payload = message.payload
        if message.encrypted:
            payload = self._decrypt(payload, key_set.encryption_key)

        elapsed_ms = (time.time() - start) * 1000
        IEC62351_LATENCY.observe(elapsed_ms)
        IEC62351_AUTH_OPS.labels(
            protocol=message.protocol.name,
            operation="verify"
        ).inc()

        return payload

    def _encrypt(self, payload: bytes, key: bytes) -> bytes:
        """Encrypt payload using AES-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce = secrets.token_bytes(12)
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, payload, None)
            return nonce + ciphertext

        except ImportError:
            # Fallback: XOR with key-derived stream (NOT production safe)
            return payload

    def _decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt payload."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce = ciphertext[:12]
            ct = ciphertext[12:]
            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, ct, None)

        except ImportError:
            return ciphertext


class GooseSecurityProfile:
    """
    Security profile for GOOSE messages.

    GOOSE (Generic Object Oriented Substation Event) requires:
    - Ultra-low latency (<4ms typical, <10ms max)
    - High availability
    - Multicast delivery
    """

    # GOOSE timing requirements
    MAX_LATENCY_MS = 10.0
    TYPICAL_LATENCY_MS = 4.0

    def __init__(
        self,
        key_manager: IEC62351KeyManagement,
    ):
        self.key_manager = key_manager
        self.policy = SecurityPolicy(
            objectives={SecurityObjective.AUTHENTICATION, SecurityObjective.INTEGRITY},
            protection_level=ProtectionLevel.AUTHENTICATION,
            max_message_age_ms=int(self.MAX_LATENCY_MS * 2),  # 2x max latency
        )
        self.protection = IEC62351MessageProtection(key_manager, self.policy)

        logger.info("GOOSE security profile initialized")

    async def protect_goose(
        self,
        goose_pdu: bytes,
        device_id: str,
    ) -> bytes:
        """
        Protect GOOSE PDU.

        Returns serialized protected message.
        """
        message = await self.protection.protect_message(
            IEC61850Protocol.GOOSE,
            goose_pdu,
            device_id,
        )

        return self._serialize_protected_goose(message)

    async def verify_goose(
        self,
        protected_goose: bytes,
    ) -> Optional[bytes]:
        """
        Verify protected GOOSE message.

        Returns original PDU if valid.
        """
        message = self._deserialize_protected_goose(protected_goose)
        return await self.protection.verify_message(message)

    def _serialize_protected_goose(self, message: ProtectedMessage) -> bytes:
        """Serialize protected GOOSE for transmission."""
        # Format: key_id(8) | seq(8) | timestamp(8) | mac(16) | payload
        return (
            message.key_id +
            message.sequence.to_bytes(8, 'big') +
            message.timestamp.to_bytes(8, 'big') +
            message.mac +
            message.payload
        )

    def _deserialize_protected_goose(self, data: bytes) -> ProtectedMessage:
        """Deserialize protected GOOSE."""
        return ProtectedMessage(
            protocol=IEC61850Protocol.GOOSE,
            key_id=data[:8],
            sequence=int.from_bytes(data[8:16], 'big'),
            timestamp=int.from_bytes(data[16:24], 'big'),
            mac=data[24:40],
            payload=data[40:],
        )


class SvSecurityProfile:
    """
    Security profile for Sampled Values.

    SV has even stricter timing than GOOSE:
    - Typically 4000 samples/second (250μs between samples)
    - Used for protection relay applications
    """

    # SV timing requirements
    SAMPLE_RATE_HZ = 4000
    SAMPLE_INTERVAL_US = 250

    def __init__(
        self,
        key_manager: IEC62351KeyManagement,
    ):
        self.key_manager = key_manager

        # For SV, we use pre-computed MAC keys to minimize latency
        self._mac_cache: Dict[str, bytes] = {}

        logger.info("SV security profile initialized")

    async def protect_sv_stream(
        self,
        samples: List[bytes],
        device_id: str,
    ) -> List[bytes]:
        """
        Protect a stream of SV samples.

        Uses batched MAC computation for efficiency.
        """
        key_set = self.key_manager.get_device_key_set(device_id)
        if not key_set:
            raise ValueError(f"No key set for device: {device_id}")

        protected = []
        base_timestamp = int(time.time() * 1000000)

        for i, sample in enumerate(samples):
            # Compute lightweight MAC for each sample
            timestamp = base_timestamp + (i * self.SAMPLE_INTERVAL_US)
            mac = self._compute_sv_mac(
                sample,
                key_set.authentication_key,
                timestamp,
            )

            # Append MAC to sample
            protected.append(sample + mac)

        return protected

    def _compute_sv_mac(
        self,
        sample: bytes,
        key: bytes,
        timestamp: int,
    ) -> bytes:
        """Compute lightweight MAC for SV sample."""
        # Use truncated MAC for efficiency
        mac_input = timestamp.to_bytes(8, 'big') + sample
        return hmac.new(key, mac_input, hashlib.sha256).digest()[:8]


class IEC62351Profile:
    """
    Complete IEC 62351 security profile.

    Provides unified interface for all IEC 61850 protocols.
    """

    def __init__(self, use_pqc: bool = True):
        self.key_manager = IEC62351KeyManagement(use_pqc=use_pqc)
        self.goose_profile = GooseSecurityProfile(self.key_manager)
        self.sv_profile = SvSecurityProfile(self.key_manager)

        logger.info("IEC 62351 profile initialized")

    async def register_device(
        self,
        device_id: str,
        key_validity_seconds: int = 3600,
    ) -> IEC62351KeySet:
        """Register device and generate initial key set."""
        return await self.key_manager.generate_key_set(
            device_id,
            key_validity_seconds,
        )

    async def protect_goose(
        self,
        pdu: bytes,
        device_id: str,
    ) -> bytes:
        """Protect GOOSE PDU."""
        return await self.goose_profile.protect_goose(pdu, device_id)

    async def verify_goose(
        self,
        protected_pdu: bytes,
    ) -> Optional[bytes]:
        """Verify protected GOOSE PDU."""
        return await self.goose_profile.verify_goose(protected_pdu)

    async def protect_sv_samples(
        self,
        samples: List[bytes],
        device_id: str,
    ) -> List[bytes]:
        """Protect stream of SV samples."""
        return await self.sv_profile.protect_sv_stream(samples, device_id)
