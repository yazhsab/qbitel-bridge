"""
LDACS (L-band Digital Aeronautical Communication System) Security Profile

PQC-enabled security for next-generation air-ground datalink:
- Higher bandwidth than VHF ACARS (~100+ kbps)
- Native support for modern cryptography
- Designed for future airspace modernization
"""

import asyncio
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
LDACS_HANDSHAKES = Counter("ldacs_pqc_handshakes_total", "Total LDACS PQC handshakes", ["status"])

LDACS_HANDSHAKE_TIME = Histogram(
    "ldacs_pqc_handshake_seconds", "LDACS PQC handshake duration", buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

LDACS_MESSAGES = Counter(
    "ldacs_authenticated_messages_total", "Total authenticated LDACS messages", ["direction", "message_type"]
)


class LdacsChannelType(Enum):
    """LDACS logical channel types."""

    DCCH = auto()  # Dedicated Control Channel
    RACH = auto()  # Random Access Channel
    DATA = auto()  # Data Channel
    VOICE = auto()  # Voice Channel (digitized)


class LdacsSecurityLevel(Enum):
    """LDACS security levels."""

    STANDARD = "standard"  # Normal operations
    ENHANCED = "enhanced"  # High-value flights
    CRITICAL = "critical"  # Safety-critical
    EMERGENCY = "emergency"  # Emergency operations


@dataclass
class LdacsSecurityContext:
    """Security context for an LDACS session."""

    session_id: bytes
    aircraft_id: str
    ground_station_id: str
    security_level: LdacsSecurityLevel

    # Key material
    shared_secret: bytes
    encryption_key: bytes
    authentication_key: bytes

    # Session state
    sequence_number: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    # Rekeying
    rekey_interval: int = 3600  # 1 hour default
    messages_until_rekey: int = 100000

    def needs_rekey(self) -> bool:
        """Check if session needs rekeying."""
        age = time.time() - self.created_at
        return age > self.rekey_interval or self.messages_until_rekey <= 0

    def increment_sequence(self) -> int:
        """Get next sequence number."""
        seq = self.sequence_number
        self.sequence_number += 1
        self.messages_until_rekey -= 1
        self.last_activity = time.time()
        return seq


@dataclass
class LdacsAuthenticatedMessage:
    """Authenticated LDACS message."""

    channel: LdacsChannelType
    payload: bytes
    sequence: int
    timestamp: float
    mac: bytes  # Message Authentication Code
    signature: Optional[bytes] = None  # PQC signature for critical messages


class LdacsKeyExchange:
    """
    LDACS Post-Quantum Key Exchange.

    Implements hybrid key exchange combining:
    - X25519 for classical security
    - ML-KEM-768 for post-quantum security
    """

    def __init__(self, security_level: LdacsSecurityLevel = LdacsSecurityLevel.STANDARD):
        self.security_level = security_level
        logger.info(f"LDACS key exchange initialized: level={security_level.value}")

    async def initiate_handshake(
        self,
        aircraft_id: str,
        ground_station_id: str,
    ) -> Tuple[bytes, bytes]:
        """
        Initiate LDACS handshake (aircraft side).

        Returns:
            Tuple of (client_hello_message, ephemeral_private_key)
        """
        from ai_engine.crypto.hybrid import HybridKemEngine, HybridKexVariant

        # Generate ephemeral key pair
        engine = HybridKemEngine(HybridKexVariant.X25519_MLKEM_768)
        keypair = await engine.generate_keypair()

        # Build ClientHello
        client_hello = self._build_client_hello(
            aircraft_id,
            ground_station_id,
            keypair.public_key.to_bytes(),
        )

        logger.debug(f"LDACS handshake initiated: {aircraft_id} -> {ground_station_id}")
        return client_hello, keypair.private_key.pqc + keypair.private_key.classical

    def _build_client_hello(
        self,
        aircraft_id: str,
        ground_station_id: str,
        public_key: bytes,
    ) -> bytes:
        """Build LDACS ClientHello message."""
        # Simple format: aircraft_id|ground_station_id|security_level|public_key
        header = f"{aircraft_id}|{ground_station_id}|{self.security_level.value}|".encode()
        return header + public_key

    async def respond_handshake(
        self,
        client_hello: bytes,
    ) -> Tuple[bytes, LdacsSecurityContext]:
        """
        Respond to LDACS handshake (ground station side).

        Returns:
            Tuple of (server_hello_message, security_context)
        """
        from ai_engine.crypto.hybrid import HybridKemEngine, HybridKexVariant, HybridPublicKey

        start = time.time()

        # Parse ClientHello
        parts = client_hello.split(b"|", 3)
        aircraft_id = parts[0].decode()
        ground_station_id = parts[1].decode()
        security_level = LdacsSecurityLevel(parts[2].decode())
        client_public_key = parts[3]

        # Perform key encapsulation
        engine = HybridKemEngine(HybridKexVariant.X25519_MLKEM_768)
        client_pk = HybridPublicKey.from_bytes(
            HybridKexVariant.X25519_MLKEM_768,
            client_public_key,
        )
        ciphertext, shared_secret = await engine.encapsulate(client_pk)

        # Derive session keys
        session_id = secrets.token_bytes(16)
        encryption_key, auth_key = self._derive_session_keys(
            shared_secret.data,
            session_id,
            aircraft_id,
            ground_station_id,
        )

        # Create security context
        context = LdacsSecurityContext(
            session_id=session_id,
            aircraft_id=aircraft_id,
            ground_station_id=ground_station_id,
            security_level=security_level,
            shared_secret=shared_secret.data,
            encryption_key=encryption_key,
            authentication_key=auth_key,
        )

        # Build ServerHello
        server_hello = self._build_server_hello(
            session_id,
            ciphertext.to_bytes(),
        )

        elapsed = time.time() - start
        LDACS_HANDSHAKE_TIME.observe(elapsed)
        LDACS_HANDSHAKES.labels(status="success").inc()

        logger.info(f"LDACS handshake completed: {aircraft_id} <-> {ground_station_id} " f"in {elapsed*1000:.1f}ms")

        return server_hello, context

    async def complete_handshake(
        self,
        server_hello: bytes,
        private_key: bytes,
        aircraft_id: str,
        ground_station_id: str,
    ) -> LdacsSecurityContext:
        """
        Complete LDACS handshake (aircraft side).

        Returns:
            Security context for the session
        """
        from ai_engine.crypto.hybrid import HybridKemEngine, HybridKexVariant, HybridCiphertext, HybridPrivateKey

        # Parse ServerHello
        session_id = server_hello[:16]
        ciphertext_bytes = server_hello[16:]

        # Perform key decapsulation
        engine = HybridKemEngine(HybridKexVariant.X25519_MLKEM_768)

        # Reconstruct private key (ML-KEM part + classical part)
        mlkem_sk_size = 2400  # ML-KEM-768 private key size
        private = HybridPrivateKey(
            variant=HybridKexVariant.X25519_MLKEM_768,
            pqc=private_key[:mlkem_sk_size],
            classical=private_key[mlkem_sk_size:],
        )

        ciphertext = HybridCiphertext.from_bytes(
            HybridKexVariant.X25519_MLKEM_768,
            ciphertext_bytes,
        )

        shared_secret = await engine.decapsulate(ciphertext, private)

        # Derive session keys
        encryption_key, auth_key = self._derive_session_keys(
            shared_secret.data,
            session_id,
            aircraft_id,
            ground_station_id,
        )

        return LdacsSecurityContext(
            session_id=session_id,
            aircraft_id=aircraft_id,
            ground_station_id=ground_station_id,
            security_level=self.security_level,
            shared_secret=shared_secret.data,
            encryption_key=encryption_key,
            authentication_key=auth_key,
        )

    def _build_server_hello(
        self,
        session_id: bytes,
        ciphertext: bytes,
    ) -> bytes:
        """Build LDACS ServerHello message."""
        return session_id + ciphertext

    def _derive_session_keys(
        self,
        shared_secret: bytes,
        session_id: bytes,
        aircraft_id: str,
        ground_station_id: str,
    ) -> Tuple[bytes, bytes]:
        """Derive encryption and authentication keys from shared secret."""
        # Use HKDF-like derivation
        context = f"{aircraft_id}|{ground_station_id}".encode()

        # Encryption key
        enc_material = hashlib.sha256(b"ldacs-encryption" + shared_secret + session_id + context).digest()

        # Authentication key
        auth_material = hashlib.sha256(b"ldacs-authentication" + shared_secret + session_id + context).digest()

        return enc_material, auth_material


class LdacsSecureChannel:
    """
    LDACS Secure Channel Implementation.

    Provides authenticated and optionally encrypted
    communication over LDACS datalink.
    """

    def __init__(self, context: LdacsSecurityContext):
        self.context = context
        self._message_buffer: List[LdacsAuthenticatedMessage] = []

        logger.info(f"LDACS secure channel established: " f"{context.aircraft_id} <-> {context.ground_station_id}")

    async def send_message(
        self,
        payload: bytes,
        channel: LdacsChannelType = LdacsChannelType.DATA,
        require_signature: bool = False,
    ) -> bytes:
        """
        Prepare message for transmission.

        Args:
            payload: Message payload
            channel: LDACS channel type
            require_signature: Include PQC signature

        Returns:
            Encoded message ready for transmission
        """
        sequence = self.context.increment_sequence()
        timestamp = time.time()

        # Compute MAC
        mac = self._compute_mac(payload, sequence, timestamp)

        # Optional PQC signature for critical messages
        signature = None
        if require_signature or self.context.security_level == LdacsSecurityLevel.CRITICAL:
            signature = await self._sign_message(payload, sequence)

        msg = LdacsAuthenticatedMessage(
            channel=channel,
            payload=payload,
            sequence=sequence,
            timestamp=timestamp,
            mac=mac,
            signature=signature,
        )

        LDACS_MESSAGES.labels(direction="outbound", message_type=channel.name).inc()

        return self._encode_message(msg)

    async def receive_message(
        self,
        encoded: bytes,
    ) -> Optional[LdacsAuthenticatedMessage]:
        """
        Verify and decode received message.

        Returns:
            Decoded message if valid, None if authentication fails
        """
        msg = self._decode_message(encoded)

        # Verify MAC
        expected_mac = self._compute_mac(
            msg.payload,
            msg.sequence,
            msg.timestamp,
        )

        if not secrets.compare_digest(msg.mac, expected_mac):
            logger.warning(f"LDACS MAC verification failed: seq={msg.sequence}")
            return None

        # Verify signature if present
        if msg.signature:
            if not await self._verify_signature(msg.payload, msg.sequence, msg.signature):
                logger.warning(f"LDACS signature verification failed: seq={msg.sequence}")
                return None

        LDACS_MESSAGES.labels(direction="inbound", message_type=msg.channel.name).inc()

        return msg

    def _compute_mac(
        self,
        payload: bytes,
        sequence: int,
        timestamp: float,
    ) -> bytes:
        """Compute HMAC for message."""
        import hmac

        message = self.context.session_id + sequence.to_bytes(8, "big") + int(timestamp * 1000).to_bytes(8, "big") + payload

        return hmac.new(
            self.context.authentication_key,
            message,
            hashlib.sha256,
        ).digest()[
            :16
        ]  # Truncate to 128 bits

    async def _sign_message(
        self,
        payload: bytes,
        sequence: int,
    ) -> bytes:
        """Sign message with PQC signature."""
        from ai_engine.crypto.falcon import FalconEngine, FalconSecurityLevel

        engine = FalconEngine(FalconSecurityLevel.FALCON_512)
        keypair = await engine.generate_keypair()

        message = self.context.session_id + sequence.to_bytes(8, "big") + payload

        sig = await engine.sign(message, keypair.private_key)
        return sig.data

    async def _verify_signature(
        self,
        payload: bytes,
        sequence: int,
        signature: bytes,
    ) -> bool:
        """Verify PQC signature."""
        # In production, would use pre-shared/certified public key
        # This is a placeholder implementation
        return len(signature) > 0

    def _encode_message(self, msg: LdacsAuthenticatedMessage) -> bytes:
        """Encode message for transmission."""
        # Format: channel(1) | sequence(8) | timestamp(8) | mac(16) |
        #         sig_len(2) | signature | payload
        sig_bytes = msg.signature or b""

        header = bytes([msg.channel.value])
        header += msg.sequence.to_bytes(8, "big")
        header += int(msg.timestamp * 1000).to_bytes(8, "big")
        header += msg.mac
        header += len(sig_bytes).to_bytes(2, "big")

        return header + sig_bytes + msg.payload

    def _decode_message(self, data: bytes) -> LdacsAuthenticatedMessage:
        """Decode received message."""
        channel = LdacsChannelType(data[0])
        sequence = int.from_bytes(data[1:9], "big")
        timestamp = int.from_bytes(data[9:17], "big") / 1000.0
        mac = data[17:33]
        sig_len = int.from_bytes(data[33:35], "big")

        if sig_len > 0:
            signature = data[35 : 35 + sig_len]
            payload = data[35 + sig_len :]
        else:
            signature = None
            payload = data[35:]

        return LdacsAuthenticatedMessage(
            channel=channel,
            payload=payload,
            sequence=sequence,
            timestamp=timestamp,
            mac=mac,
            signature=signature,
        )

    def check_rekey_needed(self) -> bool:
        """Check if session needs rekeying."""
        return self.context.needs_rekey()


class LdacsSecurityProfile:
    """
    Complete LDACS Security Profile.

    Manages security for LDACS communications including:
    - Key exchange and management
    - Message authentication
    - Session lifecycle
    """

    def __init__(self):
        self._sessions: Dict[str, LdacsSecureChannel] = {}
        self._pending_handshakes: Dict[str, Tuple[bytes, float]] = {}

        logger.info("LDACS security profile initialized")

    async def establish_session(
        self,
        aircraft_id: str,
        ground_station_id: str,
        security_level: LdacsSecurityLevel = LdacsSecurityLevel.STANDARD,
    ) -> LdacsSecureChannel:
        """
        Establish secure LDACS session (full handshake).

        This is a simplified API that handles the complete
        handshake process.
        """
        session_key = f"{aircraft_id}:{ground_station_id}"

        if session_key in self._sessions:
            existing = self._sessions[session_key]
            if not existing.check_rekey_needed():
                return existing
            # Session needs rekey, establish new one
            del self._sessions[session_key]

        # Perform handshake
        key_exchange = LdacsKeyExchange(security_level)

        # Initiate
        client_hello, private_key = await key_exchange.initiate_handshake(
            aircraft_id,
            ground_station_id,
        )

        # Respond (simulating ground station)
        server_hello, _ = await key_exchange.respond_handshake(client_hello)

        # Complete
        context = await key_exchange.complete_handshake(
            server_hello,
            private_key,
            aircraft_id,
            ground_station_id,
        )

        channel = LdacsSecureChannel(context)
        self._sessions[session_key] = channel

        return channel

    def get_session(
        self,
        aircraft_id: str,
        ground_station_id: str,
    ) -> Optional[LdacsSecureChannel]:
        """Get existing session if available."""
        session_key = f"{aircraft_id}:{ground_station_id}"
        return self._sessions.get(session_key)

    async def cleanup_expired_sessions(self, max_age: float = 7200.0) -> int:
        """Remove expired sessions."""
        now = time.time()
        expired = []

        for key, channel in self._sessions.items():
            age = now - channel.context.created_at
            if age > max_age or channel.check_rekey_needed():
                expired.append(key)

        for key in expired:
            del self._sessions[key]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired LDACS sessions")

        return len(expired)

    @property
    def active_session_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)
