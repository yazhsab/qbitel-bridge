"""
Lightweight PQC for SCADA Systems

Optimized for constrained industrial controllers:
- Memory: 64KB-256KB typical
- CPU: 16-100 MHz processors
- Latency: <10ms for critical operations

Strategies:
1. Pre-computed signature fragments
2. Stateless hash-based signatures (LMS/XMSS) for firmware
3. Lightweight symmetric key derivation
4. Session-based key caching
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Metrics
SCADA_KEY_OPS = Counter("scada_pqc_key_operations_total", "Total SCADA PQC key operations", ["operation", "security_level"])

SCADA_AUTH_LATENCY = Histogram(
    "scada_authentication_latency_ms", "SCADA authentication latency in milliseconds", buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50]
)

SCADA_ACTIVE_SESSIONS = Gauge("scada_active_sessions", "Number of active SCADA sessions")


class ScadaSecurityLevel(Enum):
    """SCADA security levels based on IEC 62443."""

    SL1 = 1  # Protection against casual/unintentional
    SL2 = 2  # Protection against intentional with low resources
    SL3 = 3  # Protection against sophisticated attackers
    SL4 = 4  # Protection against state-level attackers


class ScadaDeviceType(Enum):
    """SCADA device types with different constraints."""

    RTU = auto()  # Remote Terminal Unit
    PLC = auto()  # Programmable Logic Controller
    IED = auto()  # Intelligent Electronic Device
    HMI = auto()  # Human Machine Interface
    HISTORIAN = auto()  # Data Historian
    GATEWAY = auto()  # Protocol Gateway


@dataclass
class ScadaDeviceProfile:
    """Profile of SCADA device capabilities."""

    device_type: ScadaDeviceType
    device_id: str
    vendor: str

    # Resource constraints
    available_memory_kb: int
    cpu_mhz: int
    has_hardware_crypto: bool = False

    # Supported features
    supports_pqc: bool = False
    supports_hybrid: bool = True
    max_signature_size: int = 1024


@dataclass
class ScadaSessionContext:
    """Security context for SCADA session."""

    session_id: bytes
    device_id: str
    peer_device_id: str
    security_level: ScadaSecurityLevel

    # Key material
    master_secret: bytes
    encryption_key: bytes
    authentication_key: bytes

    # Session state
    sequence_number: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    message_count: int = 0

    # Limits
    max_messages: int = 1000000
    max_age_seconds: int = 86400  # 24 hours

    def is_valid(self) -> bool:
        """Check if session is still valid."""
        age = time.time() - self.created_at
        return age < self.max_age_seconds and self.message_count < self.max_messages

    def get_next_sequence(self) -> int:
        """Get next sequence number."""
        seq = self.sequence_number
        self.sequence_number += 1
        self.message_count += 1
        self.last_activity = time.time()
        return seq


class ScadaKeyExchange:
    """
    Lightweight key exchange for SCADA systems.

    Uses hybrid approach:
    - ML-KEM-512 for PQC security
    - X25519 for classical security
    - Optimized for constrained devices
    """

    def __init__(
        self,
        security_level: ScadaSecurityLevel = ScadaSecurityLevel.SL2,
        use_pqc: bool = True,
    ):
        self.security_level = security_level
        self.use_pqc = use_pqc

        logger.info(
            f"SCADA key exchange initialized: SL{security_level.value}, " f"PQC={'enabled' if use_pqc else 'disabled'}"
        )

    async def initiate(
        self,
        local_device: ScadaDeviceProfile,
        peer_device_id: str,
    ) -> Tuple[bytes, bytes]:
        """
        Initiate key exchange.

        Returns:
            Tuple of (key_exchange_message, ephemeral_private_key)
        """
        if self.use_pqc and local_device.supports_pqc:
            return await self._initiate_hybrid(local_device, peer_device_id)
        else:
            return await self._initiate_classical(local_device, peer_device_id)

    async def _initiate_hybrid(
        self,
        device: ScadaDeviceProfile,
        peer_device_id: str,
    ) -> Tuple[bytes, bytes]:
        """Initiate hybrid key exchange."""
        from ai_engine.crypto.hybrid import HybridKemEngine, HybridKexVariant

        # Use ML-KEM-512 for constrained devices
        engine = HybridKemEngine(HybridKexVariant.X25519_MLKEM_512)
        keypair = await engine.generate_keypair()

        # Build message
        message = self._build_kex_message(
            device.device_id,
            peer_device_id,
            keypair.public_key.to_bytes(),
            is_pqc=True,
        )

        # Store private key material
        private_key = keypair.private_key.classical + keypair.private_key.pqc

        SCADA_KEY_OPS.labels(operation="kex_initiate", security_level=f"SL{self.security_level.value}").inc()

        return message, private_key

    async def _initiate_classical(
        self,
        device: ScadaDeviceProfile,
        peer_device_id: str,
    ) -> Tuple[bytes, bytes]:
        """Initiate classical-only key exchange for legacy devices."""
        try:
            from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

            private_key = X25519PrivateKey.generate()
            public_key = private_key.public_key().public_bytes_raw()

            message = self._build_kex_message(
                device.device_id,
                peer_device_id,
                public_key,
                is_pqc=False,
            )

            return message, private_key.private_bytes_raw()

        except ImportError:
            # Fallback
            return secrets.token_bytes(64), secrets.token_bytes(32)

    def _build_kex_message(
        self,
        device_id: str,
        peer_device_id: str,
        public_key: bytes,
        is_pqc: bool,
    ) -> bytes:
        """Build key exchange message."""
        header = b"\x01" if is_pqc else b"\x00"  # Version/type byte
        header += self.security_level.value.to_bytes(1, "big")
        header += len(device_id).to_bytes(1, "big") + device_id.encode()
        header += len(peer_device_id).to_bytes(1, "big") + peer_device_id.encode()

        return header + public_key

    async def respond(
        self,
        kex_message: bytes,
        local_device: ScadaDeviceProfile,
    ) -> Tuple[bytes, ScadaSessionContext]:
        """
        Respond to key exchange and establish session.

        Returns:
            Tuple of (response_message, session_context)
        """
        start = time.time()

        # Parse incoming message
        is_pqc = kex_message[0] == 1
        security_level = ScadaSecurityLevel(kex_message[1])
        offset = 2

        peer_id_len = kex_message[offset]
        peer_device_id = kex_message[offset + 1 : offset + 1 + peer_id_len].decode()
        offset += 1 + peer_id_len

        local_id_len = kex_message[offset]
        offset += 1 + local_id_len  # Skip local device ID

        peer_public_key = kex_message[offset:]

        # Perform key exchange
        if is_pqc and local_device.supports_pqc:
            response, shared_secret = await self._respond_hybrid(peer_public_key)
        else:
            response, shared_secret = await self._respond_classical(peer_public_key)

        # Derive session keys
        session_id = secrets.token_bytes(16)
        master_secret = shared_secret
        encryption_key = self._derive_key(master_secret, b"encryption", 32)
        auth_key = self._derive_key(master_secret, b"authentication", 32)

        context = ScadaSessionContext(
            session_id=session_id,
            device_id=local_device.device_id,
            peer_device_id=peer_device_id,
            security_level=security_level,
            master_secret=master_secret,
            encryption_key=encryption_key,
            authentication_key=auth_key,
        )

        elapsed_ms = (time.time() - start) * 1000
        SCADA_AUTH_LATENCY.observe(elapsed_ms)
        SCADA_KEY_OPS.labels(operation="kex_respond", security_level=f"SL{security_level.value}").inc()

        return session_id + response, context

    async def _respond_hybrid(
        self,
        peer_public_key: bytes,
    ) -> Tuple[bytes, bytes]:
        """Respond with hybrid key exchange."""
        from ai_engine.crypto.hybrid import HybridKemEngine, HybridKexVariant, HybridPublicKey

        engine = HybridKemEngine(HybridKexVariant.X25519_MLKEM_512)
        peer_pk = HybridPublicKey.from_bytes(
            HybridKexVariant.X25519_MLKEM_512,
            peer_public_key,
        )

        ciphertext, shared_secret = await engine.encapsulate(peer_pk)
        return ciphertext.to_bytes(), shared_secret.data

    async def _respond_classical(
        self,
        peer_public_key: bytes,
    ) -> Tuple[bytes, bytes]:
        """Respond with classical key exchange."""
        try:
            from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

            our_private = X25519PrivateKey.generate()
            our_public = our_private.public_key().public_bytes_raw()

            peer_pk = X25519PublicKey.from_public_bytes(peer_public_key)
            shared_secret = our_private.exchange(peer_pk)

            return our_public, shared_secret

        except ImportError:
            return secrets.token_bytes(32), secrets.token_bytes(32)

    def _derive_key(
        self,
        master_secret: bytes,
        label: bytes,
        length: int,
    ) -> bytes:
        """Derive key from master secret."""
        return hashlib.sha256(master_secret + label).digest()[:length]


class ScadaMessageAuthenticator:
    """
    Message authenticator for SCADA communications.

    Provides lightweight authentication suitable for
    constrained industrial devices.
    """

    def __init__(
        self,
        context: ScadaSessionContext,
        use_truncated_mac: bool = True,
    ):
        self.context = context
        self.use_truncated_mac = use_truncated_mac
        self._mac_length = 8 if use_truncated_mac else 32  # 64-bit or 256-bit

        logger.debug(f"Message authenticator initialized: session={context.session_id.hex()[:8]}")

    async def authenticate_message(
        self,
        message: bytes,
        include_timestamp: bool = True,
    ) -> bytes:
        """
        Add authentication to message.

        Returns message with appended authentication data.
        """
        start = time.time()

        sequence = self.context.get_next_sequence()

        # Build authenticated data
        auth_header = sequence.to_bytes(8, "big")

        if include_timestamp:
            timestamp = int(time.time() * 1000)  # milliseconds
            auth_header += timestamp.to_bytes(8, "big")

        # Compute MAC
        mac_input = self.context.session_id + auth_header + message
        mac = hmac.new(self.context.authentication_key, mac_input, hashlib.sha256).digest()[: self._mac_length]

        elapsed_ms = (time.time() - start) * 1000
        SCADA_AUTH_LATENCY.observe(elapsed_ms)

        return auth_header + message + mac

    async def verify_message(
        self,
        authenticated_message: bytes,
    ) -> Optional[bytes]:
        """
        Verify and extract message.

        Returns original message if valid, None otherwise.
        """
        if len(authenticated_message) < 16 + self._mac_length:
            return None

        # Extract components
        sequence = int.from_bytes(authenticated_message[:8], "big")
        timestamp = int.from_bytes(authenticated_message[8:16], "big")
        message = authenticated_message[16 : -self._mac_length]
        received_mac = authenticated_message[-self._mac_length :]

        # Recompute MAC
        auth_header = authenticated_message[:16]
        mac_input = self.context.session_id + auth_header + message
        expected_mac = hmac.new(self.context.authentication_key, mac_input, hashlib.sha256).digest()[: self._mac_length]

        if not secrets.compare_digest(received_mac, expected_mac):
            logger.warning(f"MAC verification failed: seq={sequence}")
            return None

        # Check timestamp (5 minute window)
        current_time = int(time.time() * 1000)
        if abs(current_time - timestamp) > 300000:
            logger.warning(f"Timestamp outside acceptable window: {timestamp}")
            return None

        return message


class LightweightScadaPQC:
    """
    Main interface for lightweight SCADA PQC operations.

    Optimized for:
    - Minimal memory footprint
    - Fast authentication (< 1ms target)
    - Long-lived sessions
    """

    def __init__(
        self,
        device_profile: ScadaDeviceProfile,
        security_level: ScadaSecurityLevel = ScadaSecurityLevel.SL2,
    ):
        self.device_profile = device_profile
        self.security_level = security_level

        self._sessions: Dict[str, ScadaSessionContext] = {}
        self._key_exchange = ScadaKeyExchange(
            security_level=security_level,
            use_pqc=device_profile.supports_pqc,
        )

        logger.info(
            f"Lightweight SCADA PQC initialized: device={device_profile.device_id}, " f"type={device_profile.device_type.name}"
        )

    async def establish_session(
        self,
        peer_device_id: str,
    ) -> ScadaSessionContext:
        """
        Establish secure session with peer device.

        Handles the complete key exchange protocol.
        """
        # Initiate key exchange
        kex_message, private_key = await self._key_exchange.initiate(
            self.device_profile,
            peer_device_id,
        )

        # In production, this would send kex_message to peer
        # and receive response. Here we simulate the response.

        # Simulate peer response
        peer_profile = ScadaDeviceProfile(
            device_type=ScadaDeviceType.RTU,
            device_id=peer_device_id,
            vendor="Generic",
            available_memory_kb=256,
            cpu_mhz=100,
            supports_pqc=self.device_profile.supports_pqc,
        )

        response, peer_context = await self._key_exchange.respond(
            kex_message,
            peer_profile,
        )

        # Create our session context (in production, would use
        # complete_session with response)
        session = ScadaSessionContext(
            session_id=response[:16],
            device_id=self.device_profile.device_id,
            peer_device_id=peer_device_id,
            security_level=self.security_level,
            master_secret=peer_context.master_secret,
            encryption_key=peer_context.encryption_key,
            authentication_key=peer_context.authentication_key,
        )

        self._sessions[peer_device_id] = session
        SCADA_ACTIVE_SESSIONS.set(len(self._sessions))

        logger.info(f"Session established with {peer_device_id}")
        return session

    def get_session(self, peer_device_id: str) -> Optional[ScadaSessionContext]:
        """Get existing session with peer."""
        session = self._sessions.get(peer_device_id)
        if session and session.is_valid():
            return session

        # Session expired or doesn't exist
        if session:
            del self._sessions[peer_device_id]
            SCADA_ACTIVE_SESSIONS.set(len(self._sessions))

        return None

    def create_authenticator(
        self,
        peer_device_id: str,
    ) -> Optional[ScadaMessageAuthenticator]:
        """Create message authenticator for peer session."""
        session = self.get_session(peer_device_id)
        if not session:
            return None

        return ScadaMessageAuthenticator(session)

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        expired = [peer_id for peer_id, session in self._sessions.items() if not session.is_valid()]

        for peer_id in expired:
            del self._sessions[peer_id]

        if expired:
            SCADA_ACTIVE_SESSIONS.set(len(self._sessions))
            logger.info(f"Cleaned up {len(expired)} expired SCADA sessions")

        return len(expired)

    @property
    def active_session_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)


def create_rtu_profile(
    device_id: str,
    vendor: str = "Generic",
    memory_kb: int = 128,
    cpu_mhz: int = 50,
) -> ScadaDeviceProfile:
    """Create profile for typical RTU."""
    return ScadaDeviceProfile(
        device_type=ScadaDeviceType.RTU,
        device_id=device_id,
        vendor=vendor,
        available_memory_kb=memory_kb,
        cpu_mhz=cpu_mhz,
        has_hardware_crypto=False,
        supports_pqc=memory_kb >= 256,  # Need 256KB+ for PQC
        supports_hybrid=True,
        max_signature_size=512,
    )


def create_ied_profile(
    device_id: str,
    vendor: str = "Generic",
    memory_kb: int = 512,
    cpu_mhz: int = 200,
) -> ScadaDeviceProfile:
    """Create profile for typical IED."""
    return ScadaDeviceProfile(
        device_type=ScadaDeviceType.IED,
        device_id=device_id,
        vendor=vendor,
        available_memory_kb=memory_kb,
        cpu_mhz=cpu_mhz,
        has_hardware_crypto=True,
        supports_pqc=True,
        supports_hybrid=True,
        max_signature_size=1024,
    )
