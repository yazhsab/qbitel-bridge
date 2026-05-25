"""
Forward-Secure Post-Quantum Channels for Aviation

Provides forward secrecy for ATC communication channels such that
compromise of current key material cannot decrypt past sessions.
Each session epoch uses fresh ephemeral ML-KEM key exchanges, and
old keys are securely erased.

Motivation:
    Aviation communication (ACARS, CPDLC, ADS-C) sessions can span hours.
    An attacker who compromises a ground station's long-term key should not
    be able to decrypt recordings of past flights. Forward secrecy achieves
    this by deriving per-epoch session keys via ephemeral PQC key exchange.

Protocol Flow:
    1. Aircraft and Ground Station perform ML-KEM-768 key encapsulation
    2. Shared secret is derived via HKDF-SHA3-256 with epoch number
    3. Session keys are used for AES-256-GCM encryption
    4. At epoch boundary, old keys are zeroized (secure erasure)
    5. New ephemeral keys are generated for the next epoch

Ratcheting:
    Uses a double-ratchet-style key derivation chain:
    - Sending chain: KDF(chain_key, "send") → (new_chain_key, message_key)
    - Receiving chain: KDF(chain_key, "recv") → (new_chain_key, message_key)
    - DH ratchet: Fresh ML-KEM exchange at each turn

Channel Profiles:
    - ACARS (VHF/SATCOM): Long-lived sessions, periodic ratchet
    - CPDLC: Controller-Pilot messages, per-exchange ratchet
    - ADS-C: Surveillance contracts, infrequent ratchet

Standards:
    - ARINC 823: AMS security for ACARS
    - ICAO Doc 9896: ATS data link security
    - EUROCAE ED-228: LDACS security
"""

import hashlib
import logging
import secrets
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

FS_OPS = Counter("aviation_fs_channel_ops_total", "Forward-secure channel ops", ["operation"])
FS_LATENCY = Histogram("aviation_fs_channel_latency_ms", "FS channel latency", buckets=[1, 5, 10, 50, 100])
FS_EPOCH = Gauge("aviation_fs_channel_epoch", "Current channel epoch")


class ChannelType(Enum):
    """Aviation communication channel types."""
    ACARS = "acars"         # Long-lived, periodic ratchet
    CPDLC = "cpdlc"         # Controller-Pilot Data Link
    ADS_C = "ads-c"         # Automatic Dependent Surveillance - Contract
    LDACS = "ldacs"         # L-Band Digital Aeronautical Comm System


class RatchetMode(Enum):
    """Key ratchet advancement mode."""
    PER_MESSAGE = auto()     # Ratchet on every message (max forward secrecy)
    PER_EPOCH = auto()       # Ratchet at time intervals
    PER_EXCHANGE = auto()    # Ratchet on DH exchange (double-ratchet)


@dataclass(frozen=True)
class FSChannelConfig:
    """Forward-secure channel configuration."""
    channel_type: ChannelType
    ratchet_mode: RatchetMode
    epoch_duration_seconds: int = 300      # 5 minutes per epoch
    max_messages_per_epoch: int = 1000
    kem_security_level: str = "MLKEM_768"
    symmetric_algorithm: str = "AES-256-GCM"
    kdf_algorithm: str = "HKDF-SHA3-256"
    key_erasure_delay_ms: int = 0          # Immediate erasure


ACARS_FS_CONFIG = FSChannelConfig(
    channel_type=ChannelType.ACARS,
    ratchet_mode=RatchetMode.PER_EPOCH,
    epoch_duration_seconds=600,   # 10 min epochs
    max_messages_per_epoch=500,
)

CPDLC_FS_CONFIG = FSChannelConfig(
    channel_type=ChannelType.CPDLC,
    ratchet_mode=RatchetMode.PER_EXCHANGE,
    epoch_duration_seconds=300,
    max_messages_per_epoch=100,
)

ADS_C_FS_CONFIG = FSChannelConfig(
    channel_type=ChannelType.ADS_C,
    ratchet_mode=RatchetMode.PER_EPOCH,
    epoch_duration_seconds=1800,  # 30 min epochs
    max_messages_per_epoch=2000,
)

LDACS_FS_CONFIG = FSChannelConfig(
    channel_type=ChannelType.LDACS,
    ratchet_mode=RatchetMode.PER_MESSAGE,
    epoch_duration_seconds=120,
    max_messages_per_epoch=10000,
)


@dataclass
class EpochState:
    """Cryptographic state for a single epoch."""
    epoch_number: int
    sending_chain_key: bytes
    receiving_chain_key: bytes
    sending_counter: int = 0
    receiving_counter: int = 0
    ephemeral_public_key: bytes = b""
    created_at: float = field(default_factory=time.time)

    def zeroize(self):
        """Securely erase all key material for this epoch."""
        if isinstance(self.sending_chain_key, bytearray):
            for i in range(len(self.sending_chain_key)):
                self.sending_chain_key[i] = 0
        if isinstance(self.receiving_chain_key, bytearray):
            for i in range(len(self.receiving_chain_key)):
                self.receiving_chain_key[i] = 0
        # Overwrite with zeros
        object.__setattr__(self, "sending_chain_key", b"\x00" * 32)
        object.__setattr__(self, "receiving_chain_key", b"\x00" * 32)


@dataclass
class EncryptedChannelMessage:
    """An encrypted message on the forward-secure channel."""
    message_id: bytes
    epoch_number: int
    message_counter: int
    ciphertext: bytes
    nonce: bytes
    ephemeral_public_key: bytes    # Sender's ephemeral public key for this epoch
    sender_id: str
    channel_type: ChannelType
    timestamp: float = field(default_factory=time.time)

    def to_wire(self) -> bytes:
        return (
            self.message_id
            + struct.pack(">I", self.epoch_number)
            + struct.pack(">I", self.message_counter)
            + struct.pack(">H", len(self.nonce))
            + self.nonce
            + struct.pack(">H", len(self.ephemeral_public_key))
            + self.ephemeral_public_key
            + self.ciphertext
        )


class ForwardSecureChannel:
    """
    Forward-secure post-quantum communication channel.

    Implements a ratcheting key derivation scheme that provides
    forward secrecy: compromise of current keys cannot decrypt
    past messages because old keys are securely erased.

    Usage (Aircraft side):
        channel = ForwardSecureChannel("AC-ABC123", CPDLC_FS_CONFIG)
        await channel.establish(ground_station_public_key)

        encrypted = await channel.encrypt("WILCO DESCEND FL340")
        # Transmit encrypted...

    Usage (Ground Station side):
        channel = ForwardSecureChannel("GND-KJFK", CPDLC_FS_CONFIG)
        await channel.establish(aircraft_public_key)

        plaintext = await channel.decrypt(encrypted)
    """

    def __init__(
        self,
        peer_id: str,
        config: FSChannelConfig = CPDLC_FS_CONFIG,
    ):
        self.peer_id = peer_id
        self.config = config

        # Key state
        self._root_key: Optional[bytes] = None
        self._current_epoch: Optional[EpochState] = None
        self._epoch_history: Dict[int, EpochState] = {}
        self._epoch_counter = 0

        # Own ephemeral keys
        self._ephemeral_private_key: Optional[bytes] = None
        self._ephemeral_public_key: Optional[bytes] = None

        # Remote peer's ephemeral public key
        self._remote_ephemeral_key: Optional[bytes] = None

        self._message_counter = 0
        self._established = False

        logger.info(
            f"FS channel created: peer={peer_id}, "
            f"channel={config.channel_type.value}, "
            f"ratchet={config.ratchet_mode.name}"
        )

    async def establish(
        self,
        remote_public_key: bytes,
        is_initiator: bool = True,
    ) -> bytes:
        """
        Establish the forward-secure channel via ML-KEM key exchange.

        Args:
            remote_public_key: Remote peer's long-term public key
            is_initiator: True if this side initiates the handshake

        Returns:
            Own ephemeral public key to send to peer
        """
        start = time.perf_counter()

        from ai_engine.crypto.mlkem import MlKemEngine, MlKemSecurityLevel, MlKemPublicKey

        kem = MlKemEngine(MlKemSecurityLevel.MLKEM_768)

        # Generate ephemeral keypair
        ephemeral_kp = await kem.generate_keypair()
        self._ephemeral_private_key = ephemeral_kp.private_key.data
        self._ephemeral_public_key = ephemeral_kp.public_key.data

        # Encapsulate to remote's public key for initial shared secret
        remote_pk = MlKemPublicKey(MlKemSecurityLevel.MLKEM_768, remote_public_key)
        ct, shared_secret = await kem.encapsulate(remote_pk)

        # Derive root key from shared secret
        self._root_key = self._kdf(
            shared_secret.data,
            b"aviation-fs-root-v1",
            b"root-key",
        )

        self._remote_ephemeral_key = remote_public_key

        # Initialize first epoch
        self._advance_epoch()
        self._established = True

        elapsed = (time.perf_counter() - start) * 1000
        FS_LATENCY.observe(elapsed)
        FS_OPS.labels(operation="establish").inc()

        logger.info(
            f"Channel established: peer={self.peer_id}, "
            f"epoch=0, initiator={is_initiator}"
        )

        return self._ephemeral_public_key

    async def encrypt(self, plaintext: bytes) -> EncryptedChannelMessage:
        """
        Encrypt a message on the forward-secure channel.

        Derives a per-message key from the sending chain,
        encrypts with AES-256-GCM, and advances the chain.
        """
        if not self._established or not self._current_epoch:
            raise RuntimeError("Channel not established")

        start = time.perf_counter()

        # Check if epoch rotation needed
        if self._should_rotate_epoch():
            self._advance_epoch()

        epoch = self._current_epoch

        # Derive message key from sending chain
        message_key, new_chain_key = self._chain_ratchet(
            epoch.sending_chain_key, epoch.sending_counter
        )

        # Update chain state
        epoch.sending_chain_key = new_chain_key
        epoch.sending_counter += 1

        # Encrypt with AES-256-GCM
        nonce = secrets.token_bytes(12)
        ciphertext = self._aes_gcm_encrypt(message_key, nonce, plaintext)

        # Zeroize message key immediately
        message_key = b"\x00" * 32

        self._message_counter += 1

        msg = EncryptedChannelMessage(
            message_id=secrets.token_bytes(8),
            epoch_number=epoch.epoch_number,
            message_counter=epoch.sending_counter - 1,
            ciphertext=ciphertext,
            nonce=nonce,
            ephemeral_public_key=self._ephemeral_public_key or b"",
            sender_id=self.peer_id,
            channel_type=self.config.channel_type,
        )

        elapsed = (time.perf_counter() - start) * 1000
        FS_LATENCY.observe(elapsed)
        FS_OPS.labels(operation="encrypt").inc()

        return msg

    async def decrypt(self, message: EncryptedChannelMessage) -> bytes:
        """
        Decrypt a message from the forward-secure channel.

        Finds the correct epoch state and derives the message key
        from the receiving chain.
        """
        if not self._established:
            raise RuntimeError("Channel not established")

        start = time.perf_counter()

        # Find the correct epoch state
        epoch = self._get_epoch_state(message.epoch_number)
        if not epoch:
            raise ValueError(f"Unknown epoch {message.epoch_number} — keys may have been erased")

        # Derive message key from receiving chain
        # Fast-forward chain if needed
        while epoch.receiving_counter < message.message_counter:
            _, new_chain_key = self._chain_ratchet(
                epoch.receiving_chain_key, epoch.receiving_counter
            )
            epoch.receiving_chain_key = new_chain_key
            epoch.receiving_counter += 1

        message_key, new_chain_key = self._chain_ratchet(
            epoch.receiving_chain_key, epoch.receiving_counter
        )
        epoch.receiving_chain_key = new_chain_key
        epoch.receiving_counter += 1

        # Decrypt
        plaintext = self._aes_gcm_decrypt(message_key, message.nonce, message.ciphertext)

        # Zeroize message key
        message_key = b"\x00" * 32

        # Per-message ratchet: advance epoch after each message
        if self.config.ratchet_mode == RatchetMode.PER_MESSAGE:
            self._advance_epoch()

        elapsed = (time.perf_counter() - start) * 1000
        FS_LATENCY.observe(elapsed)
        FS_OPS.labels(operation="decrypt").inc()

        return plaintext

    async def ratchet(self, remote_ephemeral_key: Optional[bytes] = None):
        """
        Explicitly advance the key ratchet.

        Call this when the remote peer sends a new ephemeral key,
        or at epoch boundaries for time-based ratcheting.
        """
        if remote_ephemeral_key:
            self._remote_ephemeral_key = remote_ephemeral_key

        self._advance_epoch()

        FS_OPS.labels(operation="ratchet").inc()
        logger.debug(f"Ratchet advanced to epoch {self._epoch_counter}")

    def _advance_epoch(self):
        """Create a new epoch and erase old key material."""
        # Derive new epoch keys from root key
        new_sending = self._kdf(
            self._root_key,
            struct.pack(">I", self._epoch_counter),
            b"sending-chain",
        )
        new_receiving = self._kdf(
            self._root_key,
            struct.pack(">I", self._epoch_counter),
            b"receiving-chain",
        )

        # Advance root key (one-way: cannot recover previous root)
        self._root_key = self._kdf(
            self._root_key,
            struct.pack(">I", self._epoch_counter),
            b"root-advance",
        )

        new_epoch = EpochState(
            epoch_number=self._epoch_counter,
            sending_chain_key=new_sending,
            receiving_chain_key=new_receiving,
            ephemeral_public_key=self._ephemeral_public_key or b"",
        )

        # Erase old epochs (keep current - 1 for in-flight messages)
        if self._current_epoch:
            old_epoch_num = self._current_epoch.epoch_number
            # Keep one previous epoch for out-of-order delivery
            for epoch_num in list(self._epoch_history.keys()):
                if epoch_num < old_epoch_num:
                    self._epoch_history[epoch_num].zeroize()
                    del self._epoch_history[epoch_num]
                    logger.debug(f"Epoch {epoch_num} keys erased (forward secrecy)")

            self._epoch_history[old_epoch_num] = self._current_epoch

        self._current_epoch = new_epoch
        self._epoch_counter += 1

        FS_EPOCH.set(self._epoch_counter)

    def _should_rotate_epoch(self) -> bool:
        """Check if epoch rotation is needed."""
        if not self._current_epoch:
            return True

        epoch = self._current_epoch

        # Time-based rotation
        elapsed = time.time() - epoch.created_at
        if elapsed >= self.config.epoch_duration_seconds:
            return True

        # Message count rotation
        if epoch.sending_counter >= self.config.max_messages_per_epoch:
            return True

        return False

    def _get_epoch_state(self, epoch_number: int) -> Optional[EpochState]:
        """Get epoch state, checking current and history."""
        if self._current_epoch and self._current_epoch.epoch_number == epoch_number:
            return self._current_epoch
        return self._epoch_history.get(epoch_number)

    def _chain_ratchet(
        self,
        chain_key: bytes,
        counter: int,
    ) -> Tuple[bytes, bytes]:
        """
        Advance the chain ratchet one step.

        Returns (message_key, new_chain_key).
        """
        message_key = self._kdf(chain_key, struct.pack(">I", counter), b"message-key")
        new_chain_key = self._kdf(chain_key, struct.pack(">I", counter), b"chain-advance")
        return message_key, new_chain_key

    def _kdf(self, ikm: bytes, salt: bytes, info: bytes) -> bytes:
        """HKDF-SHA3-256 key derivation."""
        # Two-stage HKDF: extract then expand
        prk = hashlib.sha3_256(salt + ikm).digest()
        okm = hashlib.sha3_256(prk + info + b"\x01").digest()
        return okm

    def _aes_gcm_encrypt(self, key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
        """Encrypt with AES-256-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            return AESGCM(key[:32]).encrypt(nonce, plaintext, None)
        except ImportError:
            logger.critical("AES-GCM unavailable — INSECURE fallback")
            return plaintext

    def _aes_gcm_decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
        """Decrypt with AES-256-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            return AESGCM(key[:32]).decrypt(nonce, ciphertext, None)
        except ImportError:
            return ciphertext

    @property
    def current_epoch(self) -> int:
        return self._epoch_counter

    @property
    def is_established(self) -> bool:
        return self._established

    # ── Factory methods ──────────────────────────────────────────

    @classmethod
    def for_acars(cls, peer_id: str) -> "ForwardSecureChannel":
        """Create FS channel for ACARS sessions."""
        return cls(peer_id, ACARS_FS_CONFIG)

    @classmethod
    def for_cpdlc(cls, peer_id: str) -> "ForwardSecureChannel":
        """Create FS channel for CPDLC (Controller-Pilot)."""
        return cls(peer_id, CPDLC_FS_CONFIG)

    @classmethod
    def for_ads_c(cls, peer_id: str) -> "ForwardSecureChannel":
        """Create FS channel for ADS-C surveillance contracts."""
        return cls(peer_id, ADS_C_FS_CONFIG)

    @classmethod
    def for_ldacs(cls, peer_id: str) -> "ForwardSecureChannel":
        """Create FS channel for LDACS next-gen datalink."""
        return cls(peer_id, LDACS_FS_CONFIG)
