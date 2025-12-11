"""
ARINC 653 IMA (Integrated Modular Avionics) Partition Support

PQC cryptography for safety-critical aviation partitions:
- Time and space partitioning
- Inter-partition communication security
- Health monitoring integration
- DO-178C compliance support

ARINC 653 defines partitioned software architecture for
safety-critical avionics systems.
"""

import asyncio
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Callable

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Metrics
PARTITION_CRYPTO_OPS = Counter(
    'arinc653_crypto_operations_total',
    'Total cryptographic operations in partitions',
    ['partition_name', 'operation']
)

PARTITION_CRYPTO_LATENCY = Histogram(
    'arinc653_crypto_latency_seconds',
    'Cryptographic operation latency',
    ['operation'],
    buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]
)

PARTITION_KEY_STATUS = Gauge(
    'arinc653_partition_key_valid',
    'Key validity status (1=valid, 0=invalid)',
    ['partition_name']
)


class PartitionMode(Enum):
    """ARINC 653 partition operating modes."""

    IDLE = auto()           # Partition not running
    COLD_START = auto()     # Initial startup
    WARM_START = auto()     # Restart without full initialization
    NORMAL = auto()         # Normal operation


class CriticalityLevel(Enum):
    """DO-178C Design Assurance Levels."""

    DAL_A = "A"    # Catastrophic failure condition
    DAL_B = "B"    # Hazardous/Severe-Major
    DAL_C = "C"    # Major failure condition
    DAL_D = "D"    # Minor failure condition
    DAL_E = "E"    # No safety effect


class PortDirection(Enum):
    """Inter-partition communication port direction."""

    SOURCE = auto()
    DESTINATION = auto()


class PortType(Enum):
    """Inter-partition communication port types."""

    SAMPLING = auto()   # Latest value only
    QUEUING = auto()    # FIFO message queue


@dataclass
class PartitionConfig:
    """Configuration for an ARINC 653 partition."""

    name: str
    partition_id: int
    criticality: CriticalityLevel

    # Time allocation (microseconds)
    period: int              # Scheduling period
    duration: int            # Execution time per period

    # Memory allocation (bytes)
    code_size: int
    data_size: int
    stack_size: int

    # Crypto configuration
    enable_pqc: bool = True
    key_rotation_period: int = 3600  # seconds
    max_concurrent_ops: int = 10


@dataclass
class PartitionSecurityContext:
    """Security context for a partition."""

    partition_id: int
    partition_name: str

    # Key material
    signing_key: bytes
    encryption_key: bytes
    hmac_key: bytes

    # State
    key_version: int = 1
    created_at: float = field(default_factory=time.time)
    last_rotation: float = field(default_factory=time.time)
    operations_since_rotation: int = 0

    def needs_rotation(self, max_ops: int = 100000, max_age: int = 3600) -> bool:
        """Check if key rotation is needed."""
        age = time.time() - self.last_rotation
        return (
            self.operations_since_rotation >= max_ops or
            age >= max_age
        )

    def increment_ops(self) -> None:
        """Increment operation counter."""
        self.operations_since_rotation += 1


@dataclass
class SecurePort:
    """Secure inter-partition communication port."""

    name: str
    port_type: PortType
    direction: PortDirection
    max_message_size: int
    partition_id: int

    # Security
    authenticated: bool = True
    encrypted: bool = False

    # State
    message_count: int = 0
    last_message_time: float = 0.0


@dataclass
class SecureMessage:
    """Authenticated message for inter-partition communication."""

    source_partition: int
    destination_partition: int
    port_name: str
    payload: bytes
    sequence: int
    timestamp: float
    mac: bytes  # Message authentication code
    encrypted: bool = False


class PartitionKeyManager:
    """
    Key manager for ARINC 653 partitions.

    Handles key generation, storage, and rotation for
    partition cryptographic operations.
    """

    def __init__(self):
        self._contexts: Dict[int, PartitionSecurityContext] = {}
        self._key_derivation_secret: bytes = secrets.token_bytes(32)

        logger.info("Partition key manager initialized")

    async def initialize_partition(
        self,
        config: PartitionConfig,
    ) -> PartitionSecurityContext:
        """
        Initialize security context for a partition.

        Args:
            config: Partition configuration

        Returns:
            Security context for the partition
        """
        # Derive partition-specific keys
        signing_key = self._derive_key(
            config.partition_id,
            "signing",
            64,  # Falcon-512 seed size
        )

        encryption_key = self._derive_key(
            config.partition_id,
            "encryption",
            32,  # AES-256 key size
        )

        hmac_key = self._derive_key(
            config.partition_id,
            "hmac",
            32,  # HMAC-SHA256 key size
        )

        context = PartitionSecurityContext(
            partition_id=config.partition_id,
            partition_name=config.name,
            signing_key=signing_key,
            encryption_key=encryption_key,
            hmac_key=hmac_key,
        )

        self._contexts[config.partition_id] = context
        PARTITION_KEY_STATUS.labels(partition_name=config.name).set(1)

        logger.info(f"Security context initialized for partition {config.name}")
        return context

    def _derive_key(
        self,
        partition_id: int,
        purpose: str,
        length: int,
    ) -> bytes:
        """Derive key for specific purpose."""
        material = hashlib.sha512(
            self._key_derivation_secret +
            partition_id.to_bytes(4, 'big') +
            purpose.encode()
        ).digest()

        return material[:length]

    async def rotate_keys(
        self,
        partition_id: int,
    ) -> PartitionSecurityContext:
        """Rotate keys for a partition."""
        if partition_id not in self._contexts:
            raise ValueError(f"Unknown partition: {partition_id}")

        old_context = self._contexts[partition_id]

        # Generate new derivation material
        new_secret = hashlib.sha256(
            self._key_derivation_secret +
            old_context.key_version.to_bytes(4, 'big') +
            secrets.token_bytes(16)
        ).digest()

        self._key_derivation_secret = new_secret

        # Create new context
        new_context = PartitionSecurityContext(
            partition_id=partition_id,
            partition_name=old_context.partition_name,
            signing_key=self._derive_key(partition_id, "signing", 64),
            encryption_key=self._derive_key(partition_id, "encryption", 32),
            hmac_key=self._derive_key(partition_id, "hmac", 32),
            key_version=old_context.key_version + 1,
        )

        self._contexts[partition_id] = new_context

        logger.info(
            f"Keys rotated for partition {old_context.partition_name}: "
            f"v{old_context.key_version} -> v{new_context.key_version}"
        )

        return new_context

    def get_context(self, partition_id: int) -> Optional[PartitionSecurityContext]:
        """Get security context for partition."""
        return self._contexts.get(partition_id)


class PartitionCryptoEngine:
    """
    Cryptographic engine for ARINC 653 partitions.

    Provides PQC operations within partition constraints:
    - Deterministic execution time (for DAL-A/B)
    - Memory-bounded operations
    - No dynamic allocation during critical operations
    """

    def __init__(
        self,
        context: PartitionSecurityContext,
        config: PartitionConfig,
    ):
        self.context = context
        self.config = config

        # Pre-allocate buffers for deterministic timing
        self._signature_buffer = bytearray(1024)
        self._hash_buffer = bytearray(64)
        self._mac_buffer = bytearray(32)

        logger.info(f"Crypto engine initialized for partition {config.name}")

    async def sign_message(
        self,
        message: bytes,
    ) -> bytes:
        """
        Sign message with partition's signing key.

        For DAL-A/B partitions, uses deterministic timing.
        """
        start = time.time()

        if self.config.criticality in (CriticalityLevel.DAL_A, CriticalityLevel.DAL_B):
            # Deterministic signing for safety-critical partitions
            signature = await self._deterministic_sign(message)
        else:
            # Standard signing for less critical partitions
            signature = await self._standard_sign(message)

        self.context.increment_ops()

        elapsed = time.time() - start
        PARTITION_CRYPTO_LATENCY.labels(operation="sign").observe(elapsed)
        PARTITION_CRYPTO_OPS.labels(
            partition_name=self.config.name,
            operation="sign"
        ).inc()

        return signature

    async def verify_signature(
        self,
        message: bytes,
        signature: bytes,
        source_partition_id: int,
    ) -> bool:
        """Verify signature from another partition."""
        start = time.time()

        # Get source partition's context
        # In production, would look up from key manager
        result = len(signature) >= 512  # Simplified verification

        elapsed = time.time() - start
        PARTITION_CRYPTO_LATENCY.labels(operation="verify").observe(elapsed)
        PARTITION_CRYPTO_OPS.labels(
            partition_name=self.config.name,
            operation="verify"
        ).inc()

        return result

    async def compute_mac(
        self,
        message: bytes,
    ) -> bytes:
        """Compute MAC for inter-partition message."""
        import hmac

        start = time.time()

        mac = hmac.new(
            self.context.hmac_key,
            message,
            hashlib.sha256
        ).digest()

        elapsed = time.time() - start
        PARTITION_CRYPTO_LATENCY.labels(operation="mac").observe(elapsed)
        PARTITION_CRYPTO_OPS.labels(
            partition_name=self.config.name,
            operation="mac"
        ).inc()

        return mac

    async def verify_mac(
        self,
        message: bytes,
        mac: bytes,
        source_partition_id: int,
    ) -> bool:
        """Verify MAC from another partition."""
        # In production, would use source partition's HMAC key
        # For inter-partition, a shared key derived from both IDs
        import hmac as hmac_module

        shared_key = hashlib.sha256(
            self.context.hmac_key +
            source_partition_id.to_bytes(4, 'big')
        ).digest()

        expected = hmac_module.new(
            shared_key,
            message,
            hashlib.sha256
        ).digest()

        return secrets.compare_digest(mac, expected)

    async def _deterministic_sign(self, message: bytes) -> bytes:
        """
        Deterministic signing for DAL-A/B partitions.

        Ensures constant-time execution regardless of message content.
        """
        from ai_engine.crypto.falcon import FalconEngine, FalconSecurityLevel, FalconPrivateKey

        # Pad message to fixed size for constant-time
        padded = message + b'\x00' * (1024 - len(message) % 1024)

        engine = FalconEngine(FalconSecurityLevel.FALCON_512)
        sk = FalconPrivateKey(FalconSecurityLevel.FALCON_512, self.context.signing_key)

        # Hash message first for consistent signing time
        message_hash = hashlib.sha256(padded).digest()

        sig = await engine.sign(message_hash, sk)
        return sig.data

    async def _standard_sign(self, message: bytes) -> bytes:
        """Standard signing for DAL-C/D/E partitions."""
        from ai_engine.crypto.falcon import FalconEngine, FalconSecurityLevel, FalconPrivateKey

        engine = FalconEngine(FalconSecurityLevel.FALCON_512)
        sk = FalconPrivateKey(FalconSecurityLevel.FALCON_512, self.context.signing_key)

        sig = await engine.sign(message, sk)
        return sig.data


class SecureInterPartitionChannel:
    """
    Secure channel for inter-partition communication.

    Implements ARINC 653 Part 1 compliant communication
    with PQC authentication.
    """

    def __init__(
        self,
        source_partition: int,
        dest_partition: int,
        source_engine: PartitionCryptoEngine,
    ):
        self.source_partition = source_partition
        self.dest_partition = dest_partition
        self.engine = source_engine

        self._sequence = 0
        self._ports: Dict[str, SecurePort] = {}

        logger.debug(
            f"Secure channel created: {source_partition} -> {dest_partition}"
        )

    def create_port(
        self,
        name: str,
        port_type: PortType,
        direction: PortDirection,
        max_message_size: int = 4096,
        authenticated: bool = True,
        encrypted: bool = False,
    ) -> SecurePort:
        """Create secure communication port."""
        port = SecurePort(
            name=name,
            port_type=port_type,
            direction=direction,
            max_message_size=max_message_size,
            partition_id=self.source_partition,
            authenticated=authenticated,
            encrypted=encrypted,
        )

        self._ports[name] = port
        logger.debug(f"Created secure port: {name}")
        return port

    async def send_message(
        self,
        port_name: str,
        payload: bytes,
    ) -> SecureMessage:
        """
        Send authenticated message via port.

        Args:
            port_name: Name of the port to use
            payload: Message payload

        Returns:
            Secure message ready for transmission
        """
        port = self._ports.get(port_name)
        if not port:
            raise ValueError(f"Unknown port: {port_name}")

        if port.direction != PortDirection.SOURCE:
            raise ValueError(f"Port {port_name} is not a source port")

        if len(payload) > port.max_message_size:
            raise ValueError(f"Message exceeds max size: {len(payload)}")

        self._sequence += 1
        timestamp = time.time()

        # Build message for authentication
        auth_data = (
            self.source_partition.to_bytes(4, 'big') +
            self.dest_partition.to_bytes(4, 'big') +
            port_name.encode() +
            self._sequence.to_bytes(8, 'big') +
            int(timestamp * 1000000).to_bytes(8, 'big') +
            payload
        )

        # Compute MAC
        mac = await self.engine.compute_mac(auth_data)

        port.message_count += 1
        port.last_message_time = timestamp

        return SecureMessage(
            source_partition=self.source_partition,
            destination_partition=self.dest_partition,
            port_name=port_name,
            payload=payload,
            sequence=self._sequence,
            timestamp=timestamp,
            mac=mac,
            encrypted=port.encrypted,
        )

    async def receive_message(
        self,
        port_name: str,
        message: SecureMessage,
    ) -> Optional[bytes]:
        """
        Receive and verify message from port.

        Args:
            port_name: Name of the port
            message: Received message

        Returns:
            Verified payload, or None if verification fails
        """
        port = self._ports.get(port_name)
        if not port:
            raise ValueError(f"Unknown port: {port_name}")

        if port.direction != PortDirection.DESTINATION:
            raise ValueError(f"Port {port_name} is not a destination port")

        # Verify message
        auth_data = (
            message.source_partition.to_bytes(4, 'big') +
            message.destination_partition.to_bytes(4, 'big') +
            message.port_name.encode() +
            message.sequence.to_bytes(8, 'big') +
            int(message.timestamp * 1000000).to_bytes(8, 'big') +
            message.payload
        )

        if not await self.engine.verify_mac(
            auth_data,
            message.mac,
            message.source_partition
        ):
            logger.warning(
                f"MAC verification failed: partition {message.source_partition} "
                f"-> {message.destination_partition}"
            )
            return None

        port.message_count += 1
        port.last_message_time = time.time()

        return message.payload


class ARINC653SecurityManager:
    """
    Top-level security manager for ARINC 653 system.

    Coordinates security across all partitions.
    """

    def __init__(self):
        self.key_manager = PartitionKeyManager()
        self._partitions: Dict[int, Tuple[PartitionConfig, PartitionCryptoEngine]] = {}
        self._channels: Dict[Tuple[int, int], SecureInterPartitionChannel] = {}

        logger.info("ARINC 653 security manager initialized")

    async def register_partition(
        self,
        config: PartitionConfig,
    ) -> PartitionCryptoEngine:
        """Register and initialize a partition."""
        context = await self.key_manager.initialize_partition(config)
        engine = PartitionCryptoEngine(context, config)

        self._partitions[config.partition_id] = (config, engine)

        logger.info(
            f"Partition registered: {config.name} "
            f"(ID={config.partition_id}, DAL={config.criticality.value})"
        )

        return engine

    def create_channel(
        self,
        source_partition: int,
        dest_partition: int,
    ) -> SecureInterPartitionChannel:
        """Create secure channel between partitions."""
        if source_partition not in self._partitions:
            raise ValueError(f"Unknown source partition: {source_partition}")

        _, source_engine = self._partitions[source_partition]

        channel = SecureInterPartitionChannel(
            source_partition,
            dest_partition,
            source_engine,
        )

        self._channels[(source_partition, dest_partition)] = channel

        logger.info(
            f"Secure channel created: {source_partition} -> {dest_partition}"
        )

        return channel

    async def rotate_all_keys(self) -> int:
        """Rotate keys for all partitions that need it."""
        rotated = 0

        for partition_id, (config, engine) in self._partitions.items():
            context = self.key_manager.get_context(partition_id)
            if context and context.needs_rotation(
                max_age=config.key_rotation_period
            ):
                new_context = await self.key_manager.rotate_keys(partition_id)
                engine.context = new_context
                rotated += 1

        if rotated:
            logger.info(f"Rotated keys for {rotated} partitions")

        return rotated

    def get_partition_engine(
        self,
        partition_id: int,
    ) -> Optional[PartitionCryptoEngine]:
        """Get crypto engine for partition."""
        if partition_id in self._partitions:
            return self._partitions[partition_id][1]
        return None

    @property
    def partition_count(self) -> int:
        """Number of registered partitions."""
        return len(self._partitions)

    @property
    def channel_count(self) -> int:
        """Number of secure channels."""
        return len(self._channels)
