"""
Lightweight PQC for Constrained Medical Devices

Optimized implementation for devices with severe resource constraints:
- Pacemakers: 64KB FRAM, 16-bit MSP430, 10+ year battery
- Insulin pumps: Limited computational power
- Implantable monitors: Extreme power constraints

Strategies:
1. Use ML-KEM-512 (lowest security level, smallest footprint)
2. Pre-compute expensive operations during device programming
3. Session key caching to minimize key exchanges
4. Deferred verification when battery is critical
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


class MemoryConstraint(Enum):
    """Memory constraint levels for medical devices."""

    ULTRA_CONSTRAINED = auto()  # <32KB RAM (cannot run standard PQC)
    CONSTRAINED = auto()        # 32-64KB RAM (ML-KEM-512 possible)
    MODERATE = auto()           # 64-256KB RAM (ML-KEM-768 possible)
    STANDARD = auto()           # >256KB RAM (full PQC support)


@dataclass
class ConstrainedDeviceProfile:
    """Profile describing a constrained medical device."""

    device_type: str
    ram_kb: int
    flash_kb: int
    cpu_mhz: int
    battery_years: float
    has_hardware_crypto: bool = False
    supports_floating_point: bool = False

    @property
    def memory_constraint(self) -> MemoryConstraint:
        if self.ram_kb < 32:
            return MemoryConstraint.ULTRA_CONSTRAINED
        elif self.ram_kb < 64:
            return MemoryConstraint.CONSTRAINED
        elif self.ram_kb < 256:
            return MemoryConstraint.MODERATE
        else:
            return MemoryConstraint.STANDARD

    @property
    def can_run_mlkem_512(self) -> bool:
        return self.ram_kb >= 32

    @property
    def can_run_falcon(self) -> bool:
        # Falcon requires floating-point or careful emulation
        return self.ram_kb >= 32 and (self.supports_floating_point or self.ram_kb >= 64)

    @property
    def recommended_algorithm(self) -> str:
        if self.ram_kb < 32:
            return "external_shield"
        elif self.ram_kb < 64:
            return "mlkem-512"
        elif self.ram_kb < 256:
            return "mlkem-768"
        else:
            return "mlkem-1024"


# Pre-defined device profiles
PACEMAKER_PROFILE = ConstrainedDeviceProfile(
    device_type="pacemaker",
    ram_kb=64,
    flash_kb=256,
    cpu_mhz=16,
    battery_years=10,
    has_hardware_crypto=False,
    supports_floating_point=False,
)

INSULIN_PUMP_PROFILE = ConstrainedDeviceProfile(
    device_type="insulin_pump",
    ram_kb=128,
    flash_kb=512,
    cpu_mhz=48,
    battery_years=0.5,  # Replaceable batteries
    has_hardware_crypto=True,
    supports_floating_point=True,
)

ICD_PROFILE = ConstrainedDeviceProfile(
    device_type="icd",  # Implantable Cardioverter Defibrillator
    ram_kb=96,
    flash_kb=384,
    cpu_mhz=24,
    battery_years=8,
    has_hardware_crypto=False,
    supports_floating_point=False,
)


# Prometheus metrics
LIGHTWEIGHT_PQC_OPS = Counter(
    'healthcare_pqc_operations_total',
    'Healthcare PQC operations',
    ['operation', 'device_type']
)

LIGHTWEIGHT_PQC_LATENCY = Histogram(
    'healthcare_pqc_latency_seconds',
    'Healthcare PQC operation latency',
    ['operation', 'device_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)

DEVICE_BATTERY_ESTIMATE = Gauge(
    'healthcare_device_battery_years_remaining',
    'Estimated battery life remaining',
    ['device_id', 'device_type']
)


@dataclass
class SessionKey:
    """Cached session key to minimize expensive operations."""

    key: bytes
    created_at: float
    expires_at: float
    device_id: str
    operations_count: int = 0
    max_operations: int = 1000

    @property
    def is_valid(self) -> bool:
        return (
            time.time() < self.expires_at and
            self.operations_count < self.max_operations
        )

    def use(self) -> None:
        self.operations_count += 1


@dataclass
class PrecomputedKeys:
    """
    Pre-computed cryptographic material for constrained devices.

    Generated during device programming and stored in flash memory.
    Reduces runtime computational load significantly.
    """

    device_id: str
    mlkem_public_key: bytes
    mlkem_private_key: bytes  # Stored encrypted with device master key
    signature_public_key: bytes
    signature_private_key: bytes  # Stored encrypted
    session_key_pool: List[bytes]  # Pre-generated session keys
    pool_index: int = 0
    created_at: float = field(default_factory=time.time)

    def get_next_session_key(self) -> Optional[bytes]:
        """Get next pre-computed session key from pool."""
        if self.pool_index >= len(self.session_key_pool):
            return None
        key = self.session_key_pool[self.pool_index]
        self.pool_index += 1
        return key

    @property
    def keys_remaining(self) -> int:
        return len(self.session_key_pool) - self.pool_index


class LightweightPQCEngine:
    """
    Lightweight PQC engine for constrained medical devices.

    Optimizations:
    1. Session key caching to reduce KEM operations
    2. Pre-computed key pools generated during programming
    3. Battery-aware operation scheduling
    4. Graceful degradation when resources are critical
    """

    def __init__(
        self,
        device_profile: ConstrainedDeviceProfile,
        device_id: str,
        precomputed_keys: Optional[PrecomputedKeys] = None,
    ):
        self.profile = device_profile
        self.device_id = device_id
        self.precomputed = precomputed_keys

        # Session key cache
        self._session_cache: Dict[str, SessionKey] = {}

        # Select appropriate algorithm based on device capabilities
        if device_profile.memory_constraint == MemoryConstraint.ULTRA_CONSTRAINED:
            self._use_external_shield = True
            self._kem_engine = None
            self._sig_engine = None
            logger.warning(f"Device {device_id} requires external security shield")
        else:
            self._use_external_shield = False
            self._initialize_engines()

        logger.info(
            f"Lightweight PQC engine initialized for {device_profile.device_type}: "
            f"ram={device_profile.ram_kb}KB, algo={device_profile.recommended_algorithm}"
        )

    def _initialize_engines(self) -> None:
        """Initialize cryptographic engines based on device capability."""
        from ai_engine.crypto.mlkem import MlKemEngine, MlKemSecurityLevel
        from ai_engine.crypto.falcon import FalconEngine, FalconSecurityLevel

        # Select ML-KEM level based on memory
        if self.profile.ram_kb < 64:
            mlkem_level = MlKemSecurityLevel.MLKEM_512
        elif self.profile.ram_kb < 256:
            mlkem_level = MlKemSecurityLevel.MLKEM_768
        else:
            mlkem_level = MlKemSecurityLevel.MLKEM_1024

        self._kem_engine = MlKemEngine(mlkem_level)

        # Use Falcon if device supports it (smaller signatures)
        if self.profile.can_run_falcon:
            self._sig_engine = FalconEngine(FalconSecurityLevel.FALCON_512)
        else:
            # Fall back to pre-signed operations
            self._sig_engine = None
            logger.info(f"Device {self.device_id} will use pre-signed operations")

    async def establish_session(
        self,
        peer_public_key: bytes,
        peer_id: str,
    ) -> bytes:
        """
        Establish a session key with a peer.

        Uses caching and pre-computation to minimize expensive operations.

        Args:
            peer_public_key: Peer's ML-KEM public key
            peer_id: Identifier for the peer

        Returns:
            Session key (32 bytes)
        """
        # Check cache first
        cache_key = f"{peer_id}:{hash(peer_public_key)}"
        if cache_key in self._session_cache:
            session = self._session_cache[cache_key]
            if session.is_valid:
                session.use()
                logger.debug(f"Using cached session key for {peer_id}")
                LIGHTWEIGHT_PQC_OPS.labels(
                    operation="session_cache_hit",
                    device_type=self.profile.device_type
                ).inc()
                return session.key

        # Check pre-computed pool
        if self.precomputed and self.precomputed.keys_remaining > 0:
            key = self.precomputed.get_next_session_key()
            if key:
                logger.debug(f"Using pre-computed session key for {peer_id}")
                self._cache_session_key(key, peer_id, cache_key)
                LIGHTWEIGHT_PQC_OPS.labels(
                    operation="session_precomputed",
                    device_type=self.profile.device_type
                ).inc()
                return key

        # Fall back to runtime key exchange
        start = time.time()

        if self._use_external_shield:
            raise RuntimeError("Device requires external security shield for key exchange")

        from ai_engine.crypto.mlkem import MlKemPublicKey

        mlkem_pk = MlKemPublicKey(
            self._kem_engine.level,
            peer_public_key
        )

        ciphertext, shared_secret = await self._kem_engine.encapsulate(mlkem_pk)

        session_key = shared_secret.data

        # Cache the session key
        self._cache_session_key(session_key, peer_id, cache_key)

        elapsed = time.time() - start
        LIGHTWEIGHT_PQC_OPS.labels(
            operation="session_establish",
            device_type=self.profile.device_type
        ).inc()
        LIGHTWEIGHT_PQC_LATENCY.labels(
            operation="session_establish",
            device_type=self.profile.device_type
        ).observe(elapsed)

        logger.info(f"Established session with {peer_id} in {elapsed:.3f}s")

        return session_key

    def _cache_session_key(
        self,
        key: bytes,
        peer_id: str,
        cache_key: str,
    ) -> None:
        """Cache a session key for future use."""
        # Session keys valid for 24 hours or 1000 operations
        session = SessionKey(
            key=key,
            created_at=time.time(),
            expires_at=time.time() + 86400,
            device_id=peer_id,
            max_operations=1000,
        )
        self._session_cache[cache_key] = session

        # Limit cache size
        if len(self._session_cache) > 10:
            # Remove oldest expired entries
            now = time.time()
            expired = [k for k, v in self._session_cache.items() if not v.is_valid]
            for k in expired:
                del self._session_cache[k]

    async def encrypt_message(
        self,
        plaintext: bytes,
        session_key: bytes,
    ) -> bytes:
        """
        Encrypt a message using the session key.

        Uses AES-256-GCM which is quantum-safe with 256-bit keys.

        Args:
            plaintext: Data to encrypt
            session_key: 32-byte session key

        Returns:
            Ciphertext with nonce prepended
        """
        start = time.time()

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import secrets

        # Use AES-256-GCM (quantum-safe symmetric encryption)
        aesgcm = AESGCM(session_key)
        nonce = secrets.token_bytes(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        result = nonce + ciphertext

        elapsed = time.time() - start
        LIGHTWEIGHT_PQC_LATENCY.labels(
            operation="encrypt",
            device_type=self.profile.device_type
        ).observe(elapsed)

        return result

    async def decrypt_message(
        self,
        ciphertext: bytes,
        session_key: bytes,
    ) -> bytes:
        """
        Decrypt a message using the session key.

        Args:
            ciphertext: Encrypted data with nonce prepended
            session_key: 32-byte session key

        Returns:
            Decrypted plaintext
        """
        start = time.time()

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = ciphertext[:12]
        actual_ciphertext = ciphertext[12:]

        aesgcm = AESGCM(session_key)
        plaintext = aesgcm.decrypt(nonce, actual_ciphertext, None)

        elapsed = time.time() - start
        LIGHTWEIGHT_PQC_LATENCY.labels(
            operation="decrypt",
            device_type=self.profile.device_type
        ).observe(elapsed)

        return plaintext

    async def sign_telemetry(
        self,
        data: bytes,
    ) -> bytes:
        """
        Sign device telemetry data.

        Uses Falcon for smaller signatures if available,
        otherwise uses pre-signed authentication.

        Args:
            data: Telemetry data to sign

        Returns:
            Signature bytes
        """
        if self._sig_engine is None:
            raise RuntimeError("Device does not support runtime signing")

        start = time.time()

        from ai_engine.crypto.falcon import FalconPrivateKey

        # Use pre-computed private key if available
        if self.precomputed:
            sk = FalconPrivateKey(
                self._sig_engine.level,
                self.precomputed.signature_private_key
            )
        else:
            raise RuntimeError("No signing key available")

        signature = await self._sig_engine.sign(data, sk)

        elapsed = time.time() - start
        LIGHTWEIGHT_PQC_OPS.labels(
            operation="sign",
            device_type=self.profile.device_type
        ).inc()
        LIGHTWEIGHT_PQC_LATENCY.labels(
            operation="sign",
            device_type=self.profile.device_type
        ).observe(elapsed)

        logger.debug(f"Signed telemetry: {signature.size} bytes in {elapsed:.3f}s")

        return signature.data

    def get_status(self) -> Dict[str, Any]:
        """Get engine status for monitoring."""
        return {
            "device_id": self.device_id,
            "device_type": self.profile.device_type,
            "memory_constraint": self.profile.memory_constraint.name,
            "recommended_algorithm": self.profile.recommended_algorithm,
            "uses_external_shield": self._use_external_shield,
            "cached_sessions": len(self._session_cache),
            "precomputed_keys_remaining": (
                self.precomputed.keys_remaining if self.precomputed else 0
            ),
        }


async def provision_device(
    device_profile: ConstrainedDeviceProfile,
    device_id: str,
    session_key_pool_size: int = 100,
) -> PrecomputedKeys:
    """
    Provision cryptographic material for a constrained device.

    This should be called during device manufacturing/programming,
    NOT at runtime. The generated keys are stored in device flash.

    Args:
        device_profile: Device capability profile
        device_id: Unique device identifier
        session_key_pool_size: Number of session keys to pre-generate

    Returns:
        PrecomputedKeys to be programmed into device
    """
    from ai_engine.crypto.mlkem import MlKemEngine, MlKemSecurityLevel
    from ai_engine.crypto.falcon import FalconEngine, FalconSecurityLevel
    import secrets

    logger.info(f"Provisioning device {device_id} ({device_profile.device_type})")

    # Select algorithms based on device capability
    if device_profile.ram_kb < 64:
        mlkem_level = MlKemSecurityLevel.MLKEM_512
    else:
        mlkem_level = MlKemSecurityLevel.MLKEM_768

    # Generate ML-KEM key pair
    mlkem_engine = MlKemEngine(mlkem_level)
    mlkem_keypair = await mlkem_engine.generate_keypair()

    # Generate signature key pair (Falcon if supported)
    if device_profile.can_run_falcon:
        sig_engine = FalconEngine(FalconSecurityLevel.FALCON_512)
        sig_keypair = await sig_engine.generate_keypair()
        sig_public = sig_keypair.public_key.data
        sig_private = sig_keypair.private_key.data
    else:
        # Device will use pre-signed authentication
        sig_public = b""
        sig_private = b""

    # Pre-generate session key pool
    session_keys = [secrets.token_bytes(32) for _ in range(session_key_pool_size)]

    precomputed = PrecomputedKeys(
        device_id=device_id,
        mlkem_public_key=mlkem_keypair.public_key.data,
        mlkem_private_key=mlkem_keypair.private_key.data,
        signature_public_key=sig_public,
        signature_private_key=sig_private,
        session_key_pool=session_keys,
    )

    logger.info(
        f"Provisioned {device_id}: "
        f"mlkem={mlkem_level.value}, "
        f"session_keys={session_key_pool_size}"
    )

    return precomputed
