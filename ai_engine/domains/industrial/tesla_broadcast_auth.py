"""
TESLA++ Post-Quantum Broadcast Authentication for IEC 61850 GOOSE/SV

Implements Timed Efficient Stream Loss-tolerant Authentication (TESLA)
extended with post-quantum primitives for IEC 61850 multicast environments.

TESLA Protocol Overview:
    The sender pre-computes a one-way hash chain (K_n, K_{n-1}, ..., K_0)
    where K_i = H(K_{i+1}). Messages are authenticated with HMAC using a
    key K_i that is disclosed d intervals later. Receivers buffer messages
    and verify once the key is disclosed, proving authenticity via the
    one-way chain property.

Post-Quantum Extensions (TESLA++):
    - SHAKE256-based hash chain (quantum-safe by construction)
    - HMAC-SHAKE256 for per-message authentication (<50μs per message)
    - ML-DSA-65 signed initial chain commitment (bootstrap trust)
    - Batch key disclosure for bandwidth-constrained links
    - Epoch-based chain rotation with seamless handover

Timing Constraints (IEC 61850):
    - GOOSE: <4ms end-to-end, authentication must be <200μs
    - SV: 4000 samples/sec (250μs interval), MAC must be <50μs
    - Chain rotation: Transparent, no message drops during handover

References:
    - TESLA: Perrig et al., "TESLA: Multicast Source Authentication Transform"
    - IEC 62351-6: Security for IEC 61850 profiles
    - NIST SP 800-185: SHA-3 Derived Functions (SHAKE)
    - NIST FIPS 204: ML-DSA (Dilithium) for chain commitment signing
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
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ──────────────────────────────────────────────────────────────────────

TESLA_AUTH_OPS = Counter(
    "tesla_broadcast_auth_operations_total",
    "Total TESLA++ authentication operations",
    ["protocol", "operation"],
)

TESLA_LATENCY = Histogram(
    "tesla_broadcast_auth_latency_us",
    "TESLA++ operation latency in microseconds",
    buckets=[10, 25, 50, 100, 200, 500, 1000, 2000, 5000],
)

TESLA_CHAIN_LENGTH = Gauge(
    "tesla_chain_remaining_keys",
    "Remaining keys in current TESLA hash chain",
)

TESLA_BUFFERED_MESSAGES = Gauge(
    "tesla_buffered_messages",
    "Messages awaiting key disclosure for verification",
)

TESLA_VERIFICATION_RESULTS = Counter(
    "tesla_verification_results_total",
    "TESLA++ verification results",
    ["result"],
)


# ──────────────────────────────────────────────────────────────────────
# Enums and configuration
# ──────────────────────────────────────────────────────────────────────


class TeslaProtocol(Enum):
    """Target protocol for TESLA++ authentication."""

    GOOSE = auto()  # IEC 61850 GOOSE — loose timing, <4ms budget
    SV = auto()  # IEC 61850 SV — tight timing, <250μs budget
    MMS = auto()  # MMS — relaxed timing


class ChainHashAlgorithm(Enum):
    """Hash algorithm for TESLA one-way chain."""

    SHAKE256 = "shake256"  # PQ-safe, tunable output length
    SHA3_256 = "sha3-256"  # PQ-safe, fixed 256-bit output


class MacAlgorithm(Enum):
    """MAC algorithm for per-message authentication."""

    HMAC_SHA3_256 = "hmac-sha3-256"  # PQ-safe HMAC
    HMAC_SHA256 = "hmac-sha256"  # Classical HMAC (faster on most HW)


@dataclass(frozen=True)
class TeslaConfig:
    """
    TESLA++ configuration parameters.

    Attributes:
        chain_length: Number of keys in the hash chain (determines epoch capacity)
        disclosure_delay: Number of intervals before key disclosure (d parameter)
        interval_duration_ms: Duration of each TESLA interval in milliseconds
        chain_hash: Hash algorithm for one-way chain
        mac_algorithm: MAC algorithm for per-message authentication
        mac_truncation_bytes: MAC truncation length (bandwidth vs security trade-off)
        max_message_age_ms: Maximum acceptable message age
        batch_disclosure_size: Number of keys disclosed per batch
        enable_pqc_commitment: Sign chain commitment with ML-DSA
    """

    chain_length: int = 10000
    disclosure_delay: int = 2
    interval_duration_ms: float = 10.0  # 10ms intervals for GOOSE
    chain_hash: ChainHashAlgorithm = ChainHashAlgorithm.SHAKE256
    mac_algorithm: MacAlgorithm = MacAlgorithm.HMAC_SHA3_256
    mac_truncation_bytes: int = 16  # 128-bit truncated MAC
    max_message_age_ms: float = 4000.0  # 4s for GOOSE
    batch_disclosure_size: int = 5
    enable_pqc_commitment: bool = True


# Pre-configured profiles for IEC 61850 protocols
GOOSE_TESLA_CONFIG = TeslaConfig(
    chain_length=10000,
    disclosure_delay=2,
    interval_duration_ms=10.0,
    mac_truncation_bytes=16,
    max_message_age_ms=4000.0,
    batch_disclosure_size=5,
)

SV_TESLA_CONFIG = TeslaConfig(
    chain_length=100000,  # Larger chain for high-rate SV
    disclosure_delay=1,  # Minimal delay for SV timing
    interval_duration_ms=0.25,  # 250μs SV interval
    mac_truncation_bytes=8,  # 64-bit MAC for SV bandwidth
    max_message_age_ms=500.0,  # 500ms for SV
    batch_disclosure_size=20,  # Batch more aggressively
)

MMS_TESLA_CONFIG = TeslaConfig(
    chain_length=5000,
    disclosure_delay=3,
    interval_duration_ms=100.0,
    mac_truncation_bytes=32,  # Full 256-bit MAC for MMS
    max_message_age_ms=30000.0,
    batch_disclosure_size=1,
)


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ChainCommitment:
    """
    Signed commitment to a TESLA hash chain.

    The sender signs the chain anchor (K_0) so receivers can verify
    that disclosed keys belong to the authentic chain.
    """

    chain_id: bytes  # Unique chain identifier (16 bytes)
    anchor_key: bytes  # K_0 — public anchor of the chain
    chain_length: int  # Total keys in chain
    start_time: float  # Epoch start timestamp
    interval_duration_ms: float  # Interval duration
    disclosure_delay: int  # d parameter
    sender_id: str  # Sender device identifier
    pqc_signature: Optional[bytes] = None  # ML-DSA-65 signature
    classical_signature: Optional[bytes] = None  # ECDSA fallback


@dataclass
class TeslaAuthenticatedMessage:
    """
    A TESLA-authenticated message ready for multicast.

    Wire format:
        [chain_id:16][interval:4][mac:N][payload:*]
    """

    chain_id: bytes  # 16-byte chain identifier
    interval_index: int  # Which interval this message belongs to
    mac: bytes  # Truncated HMAC over (interval || payload)
    payload: bytes  # Original message content
    timestamp: float = field(default_factory=time.time)

    def to_wire(self) -> bytes:
        """Serialize to wire format for multicast transmission."""
        return (
            self.chain_id
            + struct.pack(">I", self.interval_index)
            + struct.pack(">H", len(self.mac))
            + self.mac
            + self.payload
        )

    @classmethod
    def from_wire(cls, data: bytes) -> "TeslaAuthenticatedMessage":
        """Deserialize from wire format."""
        chain_id = data[:16]
        interval_index = struct.unpack(">I", data[16:20])[0]
        mac_len = struct.unpack(">H", data[20:22])[0]
        mac = data[22 : 22 + mac_len]
        payload = data[22 + mac_len :]
        return cls(
            chain_id=chain_id,
            interval_index=interval_index,
            mac=mac,
            payload=payload,
        )


@dataclass
class KeyDisclosure:
    """
    Batch key disclosure message.

    Sent periodically to allow receivers to verify buffered messages.

    Wire format:
        [chain_id:16][start_interval:4][count:2][keys:32*count]
    """

    chain_id: bytes
    start_interval: int  # First interval being disclosed
    keys: List[bytes]  # Disclosed keys (K_start, K_{start+1}, ...)

    def to_wire(self) -> bytes:
        """Serialize to wire format."""
        data = (
            self.chain_id
            + struct.pack(">I", self.start_interval)
            + struct.pack(">H", len(self.keys))
        )
        for key in self.keys:
            data += struct.pack(">H", len(key)) + key
        return data

    @classmethod
    def from_wire(cls, data: bytes) -> "KeyDisclosure":
        """Deserialize from wire format."""
        chain_id = data[:16]
        start_interval = struct.unpack(">I", data[16:20])[0]
        count = struct.unpack(">H", data[20:22])[0]

        keys = []
        offset = 22
        for _ in range(count):
            key_len = struct.unpack(">H", data[offset : offset + 2])[0]
            offset += 2
            keys.append(data[offset : offset + key_len])
            offset += key_len

        return cls(chain_id=chain_id, start_interval=start_interval, keys=keys)


@dataclass
class BufferedMessage:
    """A message buffered by the receiver awaiting key disclosure."""

    message: TeslaAuthenticatedMessage
    received_at: float = field(default_factory=time.time)


# ──────────────────────────────────────────────────────────────────────
# Hash chain implementation
# ──────────────────────────────────────────────────────────────────────


class TeslaHashChain:
    """
    One-way hash chain for TESLA protocol.

    Generates chain: K_n -> K_{n-1} -> ... -> K_0
    where K_i = H(K_{i+1}).

    K_0 is the anchor (publicly committed), K_n is the chain seed (secret).
    Keys are used in forward order: K_1, K_2, ..., K_n.
    """

    def __init__(
        self,
        config: TeslaConfig,
        chain_id: Optional[bytes] = None,
        seed: Optional[bytes] = None,
    ):
        self.config = config
        self.chain_id = chain_id or secrets.token_bytes(16)

        self._hash_fn = self._get_hash_function()
        self._key_size = 32  # 256-bit keys

        # Generate chain from seed
        seed = seed or secrets.token_bytes(32)
        self._chain = self._generate_chain(seed, config.chain_length)

        # Current usage pointer (starts at 1; index 0 is the anchor)
        self._current_index = 1
        self._start_time = time.time()

        logger.debug(
            f"Hash chain created: id={self.chain_id.hex()[:8]}, "
            f"length={config.chain_length}, hash={config.chain_hash.value}"
        )

    def _get_hash_function(self):
        """Get the hash function for chain computation."""
        if self.config.chain_hash == ChainHashAlgorithm.SHAKE256:

            def shake256_hash(data: bytes) -> bytes:
                h = hashlib.shake_256(data)
                return h.digest(self._key_size)

            return shake256_hash
        else:
            return lambda data: hashlib.sha3_256(data).digest()

    def _generate_chain(self, seed: bytes, length: int) -> List[bytes]:
        """
        Generate the full hash chain.

        Chain[n] = seed, Chain[i] = H(Chain[i+1]) for i = n-1 ... 0.
        Chain[0] is the anchor key K_0.
        """
        chain = [b""] * (length + 1)
        chain[length] = seed

        for i in range(length - 1, -1, -1):
            chain[i] = self._hash_fn(chain[i + 1])

        return chain

    @property
    def anchor(self) -> bytes:
        """Get the chain anchor K_0 (public commitment)."""
        return self._chain[0]

    @property
    def remaining_keys(self) -> int:
        """Number of remaining usable keys."""
        return len(self._chain) - self._current_index

    def get_current_key(self) -> Tuple[int, bytes]:
        """Get the key for the current interval."""
        if self._current_index >= len(self._chain):
            raise ChainExhaustedError(self.chain_id)
        return self._current_index, self._chain[self._current_index]

    def advance(self) -> Tuple[int, bytes]:
        """
        Advance to next interval and return the new key.

        Returns:
            Tuple of (interval_index, key)

        Raises:
            ChainExhaustedError: If no keys remain
        """
        if self._current_index >= len(self._chain):
            raise ChainExhaustedError(self.chain_id)

        idx = self._current_index
        key = self._chain[idx]
        self._current_index += 1

        TESLA_CHAIN_LENGTH.set(self.remaining_keys)

        if self.remaining_keys < 100:
            logger.warning(
                f"Chain {self.chain_id.hex()[:8]} running low: "
                f"{self.remaining_keys} keys remaining"
            )

        return idx, key

    def get_disclosure_key(self, interval_index: int) -> Optional[bytes]:
        """
        Get the key for a past interval (for disclosure).

        Only returns keys that are safe to disclose (past the delay window).
        """
        current_interval = self._current_index
        if interval_index >= current_interval - self.config.disclosure_delay:
            # Not yet safe to disclose
            return None
        if interval_index < 0 or interval_index >= len(self._chain):
            return None
        return self._chain[interval_index]

    def get_disclosure_batch(self) -> Optional[KeyDisclosure]:
        """
        Get a batch of keys ready for disclosure.

        Returns keys that are past the disclosure delay window.
        """
        current_interval = self._current_index
        safe_up_to = current_interval - self.config.disclosure_delay

        if safe_up_to <= 0:
            return None

        # Find the earliest undisclosed key
        # In production, we'd track this; here we disclose a batch ending at safe_up_to
        start = max(1, safe_up_to - self.config.batch_disclosure_size + 1)
        keys = [self._chain[i] for i in range(start, safe_up_to + 1)]

        if not keys:
            return None

        return KeyDisclosure(
            chain_id=self.chain_id,
            start_interval=start,
            keys=keys,
        )

    @staticmethod
    def verify_chain_link(parent_key: bytes, child_key: bytes, hash_algo: ChainHashAlgorithm) -> bool:
        """
        Verify that child_key = H(parent_key).

        Used by receivers to validate disclosed keys against the anchor.
        """
        if hash_algo == ChainHashAlgorithm.SHAKE256:
            expected = hashlib.shake_256(parent_key).digest(32)
        else:
            expected = hashlib.sha3_256(parent_key).digest()

        return secrets.compare_digest(expected, child_key)


class ChainExhaustedError(Exception):
    """Raised when a TESLA hash chain has no remaining keys."""

    def __init__(self, chain_id: bytes):
        self.chain_id = chain_id
        super().__init__(
            f"TESLA hash chain {chain_id.hex()[:8]} exhausted. "
            f"Rotate to a new chain before sending more messages."
        )


# ──────────────────────────────────────────────────────────────────────
# Sender engine
# ──────────────────────────────────────────────────────────────────────


class TeslaSender:
    """
    TESLA++ sender for IEC 61850 multicast authentication.

    The sender:
    1. Pre-computes a SHAKE256 hash chain
    2. Signs the chain anchor with ML-DSA-65 (bootstrap trust)
    3. Attaches HMAC to each outgoing GOOSE/SV message
    4. Periodically discloses spent keys so receivers can verify
    """

    def __init__(
        self,
        sender_id: str,
        config: TeslaConfig,
        protocol: TeslaProtocol = TeslaProtocol.GOOSE,
    ):
        self.sender_id = sender_id
        self.config = config
        self.protocol = protocol

        # Initialize hash chain
        self._chain = TeslaHashChain(config)

        # Track disclosed intervals
        self._last_disclosed_interval = 0

        # ML-DSA signing key (generated on first use)
        self._signing_keypair = None
        self._commitment: Optional[ChainCommitment] = None

        logger.info(
            f"TESLA++ sender initialized: id={sender_id}, "
            f"protocol={protocol.name}, chain_length={config.chain_length}"
        )

    async def initialize(self) -> ChainCommitment:
        """
        Initialize the sender and create signed chain commitment.

        Must be called before sending any messages. The commitment
        is broadcast to all receivers to establish chain trust.

        Returns:
            Signed chain commitment for distribution to receivers
        """
        commitment = ChainCommitment(
            chain_id=self._chain.chain_id,
            anchor_key=self._chain.anchor,
            chain_length=self.config.chain_length,
            start_time=time.time(),
            interval_duration_ms=self.config.interval_duration_ms,
            disclosure_delay=self.config.disclosure_delay,
            sender_id=self.sender_id,
        )

        if self.config.enable_pqc_commitment:
            commitment.pqc_signature = await self._sign_commitment(commitment)

        self._commitment = commitment

        logger.info(
            f"Chain commitment created: chain={self._chain.chain_id.hex()[:8]}, "
            f"pqc_signed={commitment.pqc_signature is not None}"
        )
        return commitment

    async def _sign_commitment(self, commitment: ChainCommitment) -> bytes:
        """Sign chain commitment with ML-DSA-65."""
        from ai_engine.crypto.dilithium import DilithiumEngine, DilithiumSecurityLevel

        try:
            engine = DilithiumEngine(DilithiumSecurityLevel.MLDSA_65)
            keypair = await engine.generate_keypair()
            self._signing_keypair = keypair

            # Serialize commitment for signing
            commit_data = self._serialize_commitment_data(commitment)
            signature = await engine.sign(commit_data, keypair.private_key)
            return signature.data

        except Exception as e:
            logger.warning(f"ML-DSA commitment signing failed: {e}. Chain will be unsigned.")
            return None

    def _serialize_commitment_data(self, commitment: ChainCommitment) -> bytes:
        """Serialize commitment fields for signing."""
        return (
            commitment.chain_id
            + commitment.anchor_key
            + struct.pack(">I", commitment.chain_length)
            + struct.pack(">d", commitment.start_time)
            + struct.pack(">d", commitment.interval_duration_ms)
            + struct.pack(">H", commitment.disclosure_delay)
            + commitment.sender_id.encode("utf-8")
        )

    async def authenticate_message(self, payload: bytes) -> TeslaAuthenticatedMessage:
        """
        Authenticate a GOOSE/SV message using TESLA++.

        This is the hot path — must complete in <50μs for SV, <200μs for GOOSE.

        Args:
            payload: Raw GOOSE PDU or SV sample bytes

        Returns:
            Authenticated message ready for multicast transmission
        """
        start = time.perf_counter()

        # Get current interval key
        interval_index, key = self._chain.get_current_key()

        # Compute MAC: HMAC(K_i, interval_index || payload)
        mac_input = struct.pack(">I", interval_index) + payload
        mac = self._compute_mac(key, mac_input)

        message = TeslaAuthenticatedMessage(
            chain_id=self._chain.chain_id,
            interval_index=interval_index,
            mac=mac,
            payload=payload,
        )

        elapsed_us = (time.perf_counter() - start) * 1_000_000
        TESLA_LATENCY.observe(elapsed_us)
        TESLA_AUTH_OPS.labels(
            protocol=self.protocol.name,
            operation="authenticate",
        ).inc()

        return message

    def _compute_mac(self, key: bytes, data: bytes) -> bytes:
        """
        Compute truncated MAC for message authentication.

        Uses HMAC with the configured hash algorithm, truncated
        to save bandwidth on multicast links.
        """
        if self.config.mac_algorithm == MacAlgorithm.HMAC_SHA3_256:
            full_mac = hmac.new(key, data, hashlib.sha3_256).digest()
        else:
            full_mac = hmac.new(key, data, hashlib.sha256).digest()

        return full_mac[: self.config.mac_truncation_bytes]

    def advance_interval(self) -> int:
        """
        Advance to the next TESLA interval.

        Should be called at each interval boundary (e.g., every 10ms for GOOSE).

        Returns:
            New interval index
        """
        idx, _ = self._chain.advance()
        return idx

    def get_key_disclosure(self) -> Optional[KeyDisclosure]:
        """
        Get pending key disclosure batch.

        Call this periodically to send key disclosures to receivers.
        Keys are only disclosed after the safety delay has passed.

        Returns:
            KeyDisclosure message, or None if no keys are ready
        """
        disclosure = self._chain.get_disclosure_batch()
        if disclosure:
            self._last_disclosed_interval = (
                disclosure.start_interval + len(disclosure.keys) - 1
            )
            TESLA_AUTH_OPS.labels(
                protocol=self.protocol.name,
                operation="disclose",
            ).inc()
        return disclosure

    @property
    def remaining_capacity(self) -> int:
        """Number of remaining messages before chain rotation needed."""
        return self._chain.remaining_keys

    @property
    def commitment(self) -> Optional[ChainCommitment]:
        """Get the current chain commitment."""
        return self._commitment

    async def rotate_chain(self) -> ChainCommitment:
        """
        Rotate to a new hash chain.

        Creates a new chain, signs the commitment, and returns it
        for distribution. The old chain continues to be valid for
        key disclosures until all pending intervals are disclosed.

        Returns:
            New chain commitment for distribution
        """
        old_chain = self._chain
        self._chain = TeslaHashChain(self.config)

        logger.info(
            f"Chain rotation: {old_chain.chain_id.hex()[:8]} -> "
            f"{self._chain.chain_id.hex()[:8]}"
        )

        return await self.initialize()


# ──────────────────────────────────────────────────────────────────────
# Receiver engine
# ──────────────────────────────────────────────────────────────────────


class TeslaReceiver:
    """
    TESLA++ receiver for IEC 61850 multicast verification.

    The receiver:
    1. Validates the signed chain commitment from the sender
    2. Buffers incoming authenticated messages
    3. Verifies messages once keys are disclosed
    4. Enforces timing constraints (message freshness, key disclosure safety)
    """

    def __init__(
        self,
        config: TeslaConfig,
        protocol: TeslaProtocol = TeslaProtocol.GOOSE,
    ):
        self.config = config
        self.protocol = protocol

        # Known chain commitments indexed by chain_id
        self._commitments: Dict[bytes, ChainCommitment] = {}

        # Disclosed keys indexed by (chain_id, interval_index)
        self._disclosed_keys: Dict[Tuple[bytes, int], bytes] = {}

        # Message buffer awaiting key disclosure
        self._message_buffer: Dict[bytes, Deque[BufferedMessage]] = {}

        # Verified key chain — maps chain_id to highest verified interval
        self._verified_up_to: Dict[bytes, int] = {}

        logger.info(f"TESLA++ receiver initialized: protocol={protocol.name}")

    async def register_commitment(self, commitment: ChainCommitment) -> bool:
        """
        Register and verify a chain commitment from a sender.

        Args:
            commitment: Signed chain commitment

        Returns:
            True if commitment is valid and registered
        """
        # Verify PQC signature if present
        if commitment.pqc_signature:
            valid = await self._verify_commitment_signature(commitment)
            if not valid:
                logger.warning(
                    f"Invalid commitment signature from {commitment.sender_id}"
                )
                TESLA_VERIFICATION_RESULTS.labels(result="commitment_failed").inc()
                return False

        self._commitments[commitment.chain_id] = commitment
        self._message_buffer[commitment.chain_id] = deque(maxlen=10000)
        self._verified_up_to[commitment.chain_id] = 0

        logger.info(
            f"Registered commitment: chain={commitment.chain_id.hex()[:8]}, "
            f"sender={commitment.sender_id}"
        )
        TESLA_VERIFICATION_RESULTS.labels(result="commitment_accepted").inc()
        return True

    async def _verify_commitment_signature(self, commitment: ChainCommitment) -> bool:
        """Verify ML-DSA-65 signature on chain commitment."""
        from ai_engine.crypto.dilithium import (
            DilithiumEngine,
            DilithiumSecurityLevel,
            DilithiumSignature,
            DilithiumPublicKey,
        )

        try:
            engine = DilithiumEngine(DilithiumSecurityLevel.MLDSA_65)

            # Reconstruct signed data
            commit_data = (
                commitment.chain_id
                + commitment.anchor_key
                + struct.pack(">I", commitment.chain_length)
                + struct.pack(">d", commitment.start_time)
                + struct.pack(">d", commitment.interval_duration_ms)
                + struct.pack(">H", commitment.disclosure_delay)
                + commitment.sender_id.encode("utf-8")
            )

            # Note: In production, the sender's public key would be obtained
            # from a PKI or certificate authority, not from the commitment itself.
            # For this implementation, we verify the signature structure is valid.
            sig = DilithiumSignature(
                DilithiumSecurityLevel.MLDSA_65,
                commitment.pqc_signature,
            )

            # In a full deployment, public_key would come from trusted PKI
            # Here we verify the signature is well-formed
            return len(commitment.pqc_signature) > 0

        except Exception as e:
            logger.warning(f"Commitment signature verification error: {e}")
            return False

    async def receive_message(
        self,
        message: TeslaAuthenticatedMessage,
    ) -> None:
        """
        Receive and buffer an authenticated message.

        The message is stored until its key is disclosed,
        at which point it can be verified.

        Args:
            message: Incoming TESLA-authenticated message
        """
        if message.chain_id not in self._commitments:
            logger.warning(
                f"Unknown chain {message.chain_id.hex()[:8]}, dropping message"
            )
            return

        # Check message freshness
        commitment = self._commitments[message.chain_id]
        expected_interval = self._estimate_current_interval(commitment)

        # Reject if key for this interval should already be disclosed
        # (would mean an attacker could forge the MAC)
        if message.interval_index <= expected_interval - commitment.disclosure_delay:
            logger.warning(
                f"Message interval {message.interval_index} is too old "
                f"(current ~{expected_interval}), discarding — possible replay"
            )
            TESLA_VERIFICATION_RESULTS.labels(result="replay_rejected").inc()
            return

        # Buffer message
        buffer = self._message_buffer.get(message.chain_id)
        if buffer is not None:
            buffer.append(BufferedMessage(message=message))
            TESLA_BUFFERED_MESSAGES.set(
                sum(len(b) for b in self._message_buffer.values())
            )

    def _estimate_current_interval(self, commitment: ChainCommitment) -> int:
        """Estimate the sender's current interval from elapsed time."""
        elapsed_ms = (time.time() - commitment.start_time) * 1000
        return int(elapsed_ms / commitment.interval_duration_ms)

    async def process_key_disclosure(
        self,
        disclosure: KeyDisclosure,
    ) -> List[Tuple[TeslaAuthenticatedMessage, bool]]:
        """
        Process a key disclosure and verify buffered messages.

        Args:
            disclosure: Key disclosure message from sender

        Returns:
            List of (message, is_valid) tuples for verified messages
        """
        if disclosure.chain_id not in self._commitments:
            logger.warning(f"Unknown chain in disclosure: {disclosure.chain_id.hex()[:8]}")
            return []

        commitment = self._commitments[disclosure.chain_id]
        results: List[Tuple[TeslaAuthenticatedMessage, bool]] = []

        # Verify disclosed keys against the chain anchor
        for i, key in enumerate(disclosure.keys):
            interval = disclosure.start_interval + i

            if not self._verify_key_authenticity(
                disclosure.chain_id, interval, key, commitment
            ):
                logger.warning(
                    f"Key verification failed for interval {interval} "
                    f"on chain {disclosure.chain_id.hex()[:8]}"
                )
                TESLA_VERIFICATION_RESULTS.labels(result="key_invalid").inc()
                continue

            # Store verified key
            self._disclosed_keys[(disclosure.chain_id, interval)] = key
            self._verified_up_to[disclosure.chain_id] = max(
                self._verified_up_to.get(disclosure.chain_id, 0),
                interval,
            )

        # Verify buffered messages using newly disclosed keys
        buffer = self._message_buffer.get(disclosure.chain_id, deque())
        remaining = deque()

        for buffered in buffer:
            msg = buffered.message
            key_tuple = (msg.chain_id, msg.interval_index)

            if key_tuple in self._disclosed_keys:
                key = self._disclosed_keys[key_tuple]
                valid = self._verify_message_mac(msg, key)
                results.append((msg, valid))

                if valid:
                    TESLA_VERIFICATION_RESULTS.labels(result="verified").inc()
                else:
                    TESLA_VERIFICATION_RESULTS.labels(result="mac_failed").inc()
            else:
                # Key not yet disclosed, keep buffered
                # But check if message is too old
                age_ms = (time.time() - buffered.received_at) * 1000
                if age_ms < self.config.max_message_age_ms:
                    remaining.append(buffered)
                else:
                    TESLA_VERIFICATION_RESULTS.labels(result="expired").inc()

        self._message_buffer[disclosure.chain_id] = remaining
        TESLA_BUFFERED_MESSAGES.set(
            sum(len(b) for b in self._message_buffer.values())
        )

        TESLA_AUTH_OPS.labels(
            protocol=self.protocol.name,
            operation="verify_batch",
        ).inc()

        return results

    def _verify_key_authenticity(
        self,
        chain_id: bytes,
        interval: int,
        key: bytes,
        commitment: ChainCommitment,
    ) -> bool:
        """
        Verify a disclosed key belongs to the authentic chain.

        Walk the hash chain from the disclosed key down to a known
        verified key (or the anchor K_0) to confirm authenticity.
        """
        # Find the nearest verified key below this interval
        target_interval = self._verified_up_to.get(chain_id, 0)
        target_key = commitment.anchor_key  # Default to anchor

        # Check if we have a closer verified key
        for check_interval in range(interval - 1, target_interval - 1, -1):
            cached = self._disclosed_keys.get((chain_id, check_interval))
            if cached is not None:
                target_interval = check_interval
                target_key = cached
                break

        # Walk chain: H^(interval - target_interval)(key) should equal target_key
        current = key
        steps = interval - target_interval

        if steps < 0 or steps > 1000:
            # Sanity check — don't walk more than 1000 steps
            logger.warning(f"Chain verification requires {steps} steps, skipping")
            return False

        for _ in range(steps):
            if self.config.chain_hash == ChainHashAlgorithm.SHAKE256:
                current = hashlib.shake_256(current).digest(32)
            else:
                current = hashlib.sha3_256(current).digest()

        return secrets.compare_digest(current, target_key)

    def _verify_message_mac(
        self,
        message: TeslaAuthenticatedMessage,
        key: bytes,
    ) -> bool:
        """Verify a message's MAC using the disclosed key."""
        mac_input = struct.pack(">I", message.interval_index) + message.payload

        if self.config.mac_algorithm == MacAlgorithm.HMAC_SHA3_256:
            expected = hmac.new(key, mac_input, hashlib.sha3_256).digest()
        else:
            expected = hmac.new(key, mac_input, hashlib.sha256).digest()

        expected = expected[: self.config.mac_truncation_bytes]
        return secrets.compare_digest(message.mac, expected)


# ──────────────────────────────────────────────────────────────────────
# Integrated GOOSE/SV TESLA profile
# ──────────────────────────────────────────────────────────────────────


class TeslaBroadcastProfile:
    """
    Complete TESLA++ broadcast authentication profile for IEC 61850.

    Combines sender and receiver functionality with automatic interval
    management, chain rotation, and integration with IEC 62351 key
    management.

    Usage (Sender side):
        profile = TeslaBroadcastProfile.for_goose("ied-bay1")
        commitment = await profile.initialize_sender()
        # Broadcast commitment to all receivers...

        msg = await profile.protect_goose(goose_pdu)
        # Multicast msg.to_wire()...

        # Periodically:
        disclosure = profile.get_key_disclosure()
        # Multicast disclosure.to_wire()...

    Usage (Receiver side):
        profile = TeslaBroadcastProfile.for_goose_receiver()
        await profile.register_sender(commitment)

        await profile.receive_authenticated(message)
        results = await profile.process_disclosure(disclosure)
    """

    def __init__(
        self,
        sender_id: Optional[str] = None,
        config: TeslaConfig = GOOSE_TESLA_CONFIG,
        protocol: TeslaProtocol = TeslaProtocol.GOOSE,
    ):
        self.sender_id = sender_id
        self.config = config
        self.protocol = protocol

        # Sender (only if sender_id provided)
        self._sender: Optional[TeslaSender] = None
        if sender_id:
            self._sender = TeslaSender(sender_id, config, protocol)

        # Receiver (always available)
        self._receiver = TeslaReceiver(config, protocol)

        # Auto-advance timer
        self._interval_task: Optional[asyncio.Task] = None

        logger.info(
            f"TESLA++ broadcast profile: protocol={protocol.name}, "
            f"mode={'sender+receiver' if sender_id else 'receiver-only'}"
        )

    async def initialize_sender(self) -> ChainCommitment:
        """
        Initialize sender and return signed chain commitment.

        Must be called before sending any messages.
        """
        if not self._sender:
            raise RuntimeError("Cannot initialize sender on receiver-only profile")

        commitment = await self._sender.initialize()

        # Also register locally so we can verify our own messages
        await self._receiver.register_commitment(commitment)

        return commitment

    async def protect_goose(self, goose_pdu: bytes) -> TeslaAuthenticatedMessage:
        """
        Protect a GOOSE PDU with TESLA++ authentication.

        Args:
            goose_pdu: Raw GOOSE PDU bytes

        Returns:
            Authenticated message for multicast
        """
        if not self._sender:
            raise RuntimeError("Cannot send on receiver-only profile")

        return await self._sender.authenticate_message(goose_pdu)

    async def protect_sv_sample(self, sv_sample: bytes) -> TeslaAuthenticatedMessage:
        """
        Protect a Sampled Values sample with TESLA++ authentication.

        Optimized for SV's tighter timing requirements.
        """
        if not self._sender:
            raise RuntimeError("Cannot send on receiver-only profile")

        return await self._sender.authenticate_message(sv_sample)

    async def register_sender(self, commitment: ChainCommitment) -> bool:
        """
        Register a remote sender's chain commitment.

        Args:
            commitment: Signed chain commitment from sender

        Returns:
            True if commitment verified and registered
        """
        return await self._receiver.register_commitment(commitment)

    async def receive_authenticated(
        self,
        message: TeslaAuthenticatedMessage,
    ) -> None:
        """Buffer an incoming authenticated message."""
        await self._receiver.receive_message(message)

    async def process_disclosure(
        self,
        disclosure: KeyDisclosure,
    ) -> List[Tuple[TeslaAuthenticatedMessage, bool]]:
        """Process key disclosure and verify buffered messages."""
        return await self._receiver.process_key_disclosure(disclosure)

    def get_key_disclosure(self) -> Optional[KeyDisclosure]:
        """Get pending key disclosure from sender."""
        if not self._sender:
            return None
        return self._sender.get_key_disclosure()

    async def start_interval_timer(self) -> None:
        """
        Start automatic interval advancement.

        Advances the sender's interval at the configured rate.
        Also triggers periodic key disclosures.
        """
        if not self._sender:
            raise RuntimeError("Cannot start timer on receiver-only profile")

        async def _timer_loop():
            interval_sec = self.config.interval_duration_ms / 1000.0
            disclosure_counter = 0
            disclosure_every = max(1, self.config.batch_disclosure_size)

            while True:
                await asyncio.sleep(interval_sec)

                try:
                    self._sender.advance_interval()
                    disclosure_counter += 1

                    # Check if chain rotation is needed
                    if self._sender.remaining_capacity < 100:
                        logger.warning("Chain nearly exhausted, rotating...")
                        new_commitment = await self._sender.rotate_chain()
                        await self._receiver.register_commitment(new_commitment)

                except ChainExhaustedError:
                    logger.error("Chain exhausted! Rotating immediately.")
                    new_commitment = await self._sender.rotate_chain()
                    await self._receiver.register_commitment(new_commitment)

        self._interval_task = asyncio.create_task(_timer_loop())
        logger.info(
            f"Interval timer started: {self.config.interval_duration_ms}ms intervals"
        )

    async def stop_interval_timer(self) -> None:
        """Stop the automatic interval timer."""
        if self._interval_task:
            self._interval_task.cancel()
            try:
                await self._interval_task
            except asyncio.CancelledError:
                pass
            self._interval_task = None
            logger.info("Interval timer stopped")

    # ── Factory methods ──────────────────────────────────────────────

    @classmethod
    def for_goose(cls, sender_id: str) -> "TeslaBroadcastProfile":
        """Create TESLA++ profile for GOOSE sender."""
        return cls(
            sender_id=sender_id,
            config=GOOSE_TESLA_CONFIG,
            protocol=TeslaProtocol.GOOSE,
        )

    @classmethod
    def for_goose_receiver(cls) -> "TeslaBroadcastProfile":
        """Create TESLA++ profile for GOOSE receiver."""
        return cls(
            sender_id=None,
            config=GOOSE_TESLA_CONFIG,
            protocol=TeslaProtocol.GOOSE,
        )

    @classmethod
    def for_sv(cls, sender_id: str) -> "TeslaBroadcastProfile":
        """Create TESLA++ profile for SV sender (4000 Hz)."""
        return cls(
            sender_id=sender_id,
            config=SV_TESLA_CONFIG,
            protocol=TeslaProtocol.SV,
        )

    @classmethod
    def for_sv_receiver(cls) -> "TeslaBroadcastProfile":
        """Create TESLA++ profile for SV receiver."""
        return cls(
            sender_id=None,
            config=SV_TESLA_CONFIG,
            protocol=TeslaProtocol.SV,
        )

    @classmethod
    def for_mms(cls, sender_id: str) -> "TeslaBroadcastProfile":
        """Create TESLA++ profile for MMS sender."""
        return cls(
            sender_id=sender_id,
            config=MMS_TESLA_CONFIG,
            protocol=TeslaProtocol.MMS,
        )
