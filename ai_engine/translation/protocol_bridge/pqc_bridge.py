"""
QBITEL - PQC-Protected Protocol Bridge

Quantum-safe protocol translation layer that wraps all protocol translations
with post-quantum cryptography for protection against harvest-now-decrypt-later attacks.

Key Features:
- Automatic PQC key encapsulation for session keys
- Quantum-safe message signing for integrity
- Hybrid classical + PQC mode support
- Zero-touch integration with existing protocol bridge
- Domain-aware algorithm selection
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from prometheus_client import Counter, Histogram, Gauge

from ...core.config import Config
from ...core.exceptions import QbitelAIException
from ...crypto.pqc_unified import (
    PQCEngine,
    PQCAlgorithm,
    DomainProfile,
    KeyPair,
    Signature,
    EncapsulationResult,
)

from .protocol_bridge import (
    ProtocolBridge,
    TranslationContext,
    TranslationResult,
    BridgeException,
)
from ..models import TranslationMode, QualityLevel

logger = logging.getLogger(__name__)


# Metrics
PQC_BRIDGE_OPERATIONS = Counter(
    "qbitel_pqc_bridge_operations_total",
    "Total PQC bridge operations",
    ["operation", "domain", "status"],
)

PQC_BRIDGE_LATENCY = Histogram(
    "qbitel_pqc_bridge_latency_seconds",
    "PQC bridge operation latency",
    ["operation", "domain"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PQC_BRIDGE_KEY_ROTATIONS = Counter(
    "qbitel_pqc_bridge_key_rotations_total",
    "Total key rotations performed",
    ["reason"],
)


class PQCBridgeMode(Enum):
    """PQC bridge operation modes."""

    ENCRYPT_ONLY = "encrypt_only"  # Only encrypt translated data
    SIGN_ONLY = "sign_only"  # Only sign for integrity
    FULL = "full"  # Encrypt + Sign
    HYBRID = "hybrid"  # Classical + PQC hybrid
    PASSTHROUGH = "passthrough"  # No PQC (for testing/migration)


class KeyRotationPolicy(Enum):
    """Key rotation policies."""

    TIME_BASED = "time_based"  # Rotate after N seconds
    MESSAGE_COUNT = "message_count"  # Rotate after N messages
    DATA_VOLUME = "data_volume"  # Rotate after N bytes
    MANUAL = "manual"  # Only rotate on explicit request


@dataclass
class PQCSession:
    """PQC-protected session for protocol translation."""

    session_id: str
    kem_keypair: KeyPair
    signature_keypair: KeyPair
    shared_secret: Optional[bytes] = None
    peer_public_key: Optional[bytes] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0
    bytes_processed: int = 0
    domain: DomainProfile = DomainProfile.ENTERPRISE

    @property
    def age_seconds(self) -> float:
        """Get session age in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    @property
    def is_established(self) -> bool:
        """Check if session has completed key exchange."""
        return self.shared_secret is not None


@dataclass
class PQCTranslationResult:
    """Translation result with PQC protection."""

    # Base translation result
    translation_result: TranslationResult

    # PQC protection details
    encrypted_data: bytes
    signature: Optional[bytes] = None
    kem_ciphertext: Optional[bytes] = None

    # Metadata
    pqc_mode: PQCBridgeMode = PQCBridgeMode.FULL
    kem_algorithm: Optional[PQCAlgorithm] = None
    signature_algorithm: Optional[PQCAlgorithm] = None
    session_id: Optional[str] = None

    # Performance
    pqc_overhead_ms: float = 0.0
    total_processing_time: float = 0.0

    @property
    def is_quantum_safe(self) -> bool:
        """Check if result is quantum-safe."""
        return self.pqc_mode in {
            PQCBridgeMode.ENCRYPT_ONLY,
            PQCBridgeMode.SIGN_ONLY,
            PQCBridgeMode.FULL,
            PQCBridgeMode.HYBRID,
        }


@dataclass
class PQCBridgeConfig:
    """Configuration for PQC-protected bridge."""

    mode: PQCBridgeMode = PQCBridgeMode.FULL
    domain: DomainProfile = DomainProfile.ENTERPRISE
    kem_algorithm: Optional[PQCAlgorithm] = None
    signature_algorithm: Optional[PQCAlgorithm] = None
    hybrid_mode: bool = True
    fips_mode: bool = False

    # Key rotation settings
    rotation_policy: KeyRotationPolicy = KeyRotationPolicy.TIME_BASED
    rotation_interval_seconds: int = 3600  # 1 hour
    rotation_message_count: int = 100000
    rotation_data_volume_mb: int = 1024  # 1 GB

    # Session settings
    max_sessions: int = 1000
    session_timeout_seconds: int = 7200  # 2 hours

    # Performance settings
    enable_session_caching: bool = True
    enable_parallel_pqc: bool = True


class PQCProtocolBridge:
    """
    PQC-protected protocol bridge for quantum-safe translations.

    Wraps the standard ProtocolBridge with post-quantum cryptography
    to protect translated data from future quantum attacks.

    Usage:
        # Create PQC bridge with default enterprise settings
        bridge = PQCProtocolBridge(config)
        await bridge.initialize()

        # Translate with PQC protection
        result = await bridge.translate_protected(context)

        # The result contains encrypted + signed data
        encrypted_data = result.encrypted_data
    """

    def __init__(
        self,
        config: Config,
        bridge_config: Optional[PQCBridgeConfig] = None,
        base_bridge: Optional[ProtocolBridge] = None,
    ):
        """
        Initialize PQC-protected protocol bridge.

        Args:
            config: QBITEL configuration
            bridge_config: PQC bridge configuration
            base_bridge: Optional existing protocol bridge to wrap
        """
        self.config = config
        self.bridge_config = bridge_config or PQCBridgeConfig()
        self.logger = logging.getLogger(__name__)

        # Base protocol bridge
        self.base_bridge = base_bridge

        # PQC engine
        self.pqc_engine: Optional[PQCEngine] = None

        # Session management
        self.sessions: Dict[str, PQCSession] = {}
        self.default_session: Optional[PQCSession] = None

        # Key management
        self.master_kem_keypair: Optional[KeyPair] = None
        self.master_signature_keypair: Optional[KeyPair] = None
        self.key_rotation_task: Optional[asyncio.Task] = None

        # Metrics
        self.metrics = {
            "total_translations": 0,
            "protected_translations": 0,
            "key_rotations": 0,
            "active_sessions": 0,
            "average_pqc_overhead_ms": 0.0,
        }

        # State
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize PQC-protected bridge."""
        if self.is_initialized:
            return

        start_time = time.time()
        self.logger.info("Initializing PQC-protected protocol bridge...")

        try:
            # Initialize PQC engine
            self.pqc_engine = PQCEngine(
                domain=self.bridge_config.domain,
                kem_algorithm=self.bridge_config.kem_algorithm,
                signature_algorithm=self.bridge_config.signature_algorithm,
                hybrid_mode=self.bridge_config.hybrid_mode,
                fips_mode=self.bridge_config.fips_mode,
            )

            # Generate master keypairs
            self.master_kem_keypair = await self.pqc_engine.generate_kem_keypair()
            self.master_signature_keypair = await self.pqc_engine.generate_signature_keypair()

            self.logger.info(
                f"Generated master KEM keypair: "
                f"pk_size={self.master_kem_keypair.public_key_size}, "
                f"fingerprint={self.master_kem_keypair.fingerprint()[:16]}"
            )

            # Initialize base bridge if not provided
            if self.base_bridge is None:
                self.base_bridge = ProtocolBridge(self.config)
                await self.base_bridge.initialize()

            # Create default session
            self.default_session = await self._create_session()

            # Start key rotation task
            if self.bridge_config.rotation_policy != KeyRotationPolicy.MANUAL:
                self.key_rotation_task = asyncio.create_task(self._key_rotation_loop())

            self.is_initialized = True

            init_time = time.time() - start_time
            self.logger.info(
                f"PQC protocol bridge initialized in {init_time:.2f}s, "
                f"mode={self.bridge_config.mode.value}, "
                f"domain={self.bridge_config.domain.value}"
            )

        except Exception as e:
            self.logger.error(f"PQC bridge initialization failed: {e}")
            raise BridgeException(f"PQC bridge initialization failed: {e}")

    async def translate_protected(
        self,
        context: TranslationContext,
        session_id: Optional[str] = None,
        mode: Optional[PQCBridgeMode] = None,
    ) -> PQCTranslationResult:
        """
        Translate protocol data with PQC protection.

        Args:
            context: Translation context with source data
            session_id: Optional session ID for stateful translations
            mode: Override default PQC mode

        Returns:
            PQC-protected translation result
        """
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()
        pqc_mode = mode or self.bridge_config.mode

        try:
            # Get or create session
            session = self._get_session(session_id)

            # Perform base translation
            translation_result = await self.base_bridge.translate_protocol(context)

            # Apply PQC protection based on mode
            if pqc_mode == PQCBridgeMode.PASSTHROUGH:
                pqc_result = await self._create_passthrough_result(translation_result, session)
            elif pqc_mode == PQCBridgeMode.ENCRYPT_ONLY:
                pqc_result = await self._encrypt_translation(translation_result, session)
            elif pqc_mode == PQCBridgeMode.SIGN_ONLY:
                pqc_result = await self._sign_translation(translation_result, session)
            elif pqc_mode == PQCBridgeMode.FULL:
                pqc_result = await self._full_protect_translation(translation_result, session)
            elif pqc_mode == PQCBridgeMode.HYBRID:
                pqc_result = await self._hybrid_protect_translation(translation_result, session)
            else:
                raise BridgeException(f"Unknown PQC mode: {pqc_mode}")

            # Update session statistics
            session.message_count += 1
            session.bytes_processed += len(context.source_data)
            session.last_activity = datetime.now(timezone.utc)

            # Check key rotation
            await self._check_key_rotation(session)

            # Update metrics
            self.metrics["total_translations"] += 1
            if pqc_result.is_quantum_safe:
                self.metrics["protected_translations"] += 1

            pqc_overhead = (time.time() - start_time) - translation_result.processing_time
            pqc_result.pqc_overhead_ms = pqc_overhead * 1000
            pqc_result.total_processing_time = time.time() - start_time

            # Prometheus metrics
            PQC_BRIDGE_OPERATIONS.labels(
                operation="translate",
                domain=self.bridge_config.domain.value,
                status="success",
            ).inc()

            PQC_BRIDGE_LATENCY.labels(
                operation="translate",
                domain=self.bridge_config.domain.value,
            ).observe(pqc_result.total_processing_time)

            self.logger.debug(
                f"PQC translation completed: mode={pqc_mode.value}, " f"overhead={pqc_result.pqc_overhead_ms:.2f}ms"
            )

            return pqc_result

        except Exception as e:
            PQC_BRIDGE_OPERATIONS.labels(
                operation="translate",
                domain=self.bridge_config.domain.value,
                status="error",
            ).inc()

            self.logger.error(f"PQC translation failed: {e}")
            raise BridgeException(f"PQC translation failed: {e}")

    async def _create_passthrough_result(
        self, translation_result: TranslationResult, session: PQCSession
    ) -> PQCTranslationResult:
        """Create passthrough result without PQC protection."""
        return PQCTranslationResult(
            translation_result=translation_result,
            encrypted_data=translation_result.translated_data,
            pqc_mode=PQCBridgeMode.PASSTHROUGH,
            session_id=session.session_id,
        )

    async def _encrypt_translation(self, translation_result: TranslationResult, session: PQCSession) -> PQCTranslationResult:
        """Encrypt translation result using PQC KEM."""
        # Encapsulate to generate ephemeral shared secret
        encap_result = await self.pqc_engine.encapsulate(session.kem_keypair.public_key)

        # Use shared secret to derive encryption key
        encryption_key = self._derive_encryption_key(encap_result.shared_secret)

        # Encrypt the translated data using AES-GCM with PQC-derived key
        encrypted_data = await self._aes_encrypt(translation_result.translated_data, encryption_key)

        return PQCTranslationResult(
            translation_result=translation_result,
            encrypted_data=encrypted_data,
            kem_ciphertext=encap_result.ciphertext,
            pqc_mode=PQCBridgeMode.ENCRYPT_ONLY,
            kem_algorithm=self.pqc_engine.kem_algorithm,
            session_id=session.session_id,
        )

    async def _sign_translation(self, translation_result: TranslationResult, session: PQCSession) -> PQCTranslationResult:
        """Sign translation result using PQC signatures."""
        # Sign the translated data
        signature = await self.pqc_engine.sign(
            translation_result.translated_data,
            session.signature_keypair.private_key,
        )

        return PQCTranslationResult(
            translation_result=translation_result,
            encrypted_data=translation_result.translated_data,  # Not encrypted
            signature=signature.signature,
            pqc_mode=PQCBridgeMode.SIGN_ONLY,
            signature_algorithm=self.pqc_engine.signature_algorithm,
            session_id=session.session_id,
        )

    async def _full_protect_translation(
        self, translation_result: TranslationResult, session: PQCSession
    ) -> PQCTranslationResult:
        """Apply full PQC protection (encrypt + sign)."""
        # First encrypt
        encap_result = await self.pqc_engine.encapsulate(session.kem_keypair.public_key)
        encryption_key = self._derive_encryption_key(encap_result.shared_secret)
        encrypted_data = await self._aes_encrypt(translation_result.translated_data, encryption_key)

        # Then sign the encrypted data
        signature = await self.pqc_engine.sign(
            encrypted_data,
            session.signature_keypair.private_key,
        )

        return PQCTranslationResult(
            translation_result=translation_result,
            encrypted_data=encrypted_data,
            signature=signature.signature,
            kem_ciphertext=encap_result.ciphertext,
            pqc_mode=PQCBridgeMode.FULL,
            kem_algorithm=self.pqc_engine.kem_algorithm,
            signature_algorithm=self.pqc_engine.signature_algorithm,
            session_id=session.session_id,
        )

    async def _hybrid_protect_translation(
        self, translation_result: TranslationResult, session: PQCSession
    ) -> PQCTranslationResult:
        """Apply hybrid classical + PQC protection."""
        # Combine classical ECDH + PQC KEM for key establishment
        # For now, use full PQC mode - hybrid would combine X25519 + ML-KEM

        # Generate classical ECDH shared secret (simplified)
        import secrets

        classical_secret = secrets.token_bytes(32)

        # Generate PQC shared secret
        encap_result = await self.pqc_engine.encapsulate(session.kem_keypair.public_key)

        # Combine secrets using KDF
        combined_secret = self._combine_secrets(classical_secret, encap_result.shared_secret)
        encryption_key = self._derive_encryption_key(combined_secret)

        # Encrypt
        encrypted_data = await self._aes_encrypt(translation_result.translated_data, encryption_key)

        # Sign with PQC signature
        signature = await self.pqc_engine.sign(
            encrypted_data,
            session.signature_keypair.private_key,
        )

        return PQCTranslationResult(
            translation_result=translation_result,
            encrypted_data=encrypted_data,
            signature=signature.signature,
            kem_ciphertext=encap_result.ciphertext,
            pqc_mode=PQCBridgeMode.HYBRID,
            kem_algorithm=self.pqc_engine.kem_algorithm,
            signature_algorithm=self.pqc_engine.signature_algorithm,
            session_id=session.session_id,
        )

    async def decrypt_protected(
        self,
        pqc_result: PQCTranslationResult,
        session_id: Optional[str] = None,
    ) -> bytes:
        """
        Decrypt PQC-protected translation result.

        Args:
            pqc_result: PQC-protected result to decrypt
            session_id: Session ID for decryption keys

        Returns:
            Decrypted data
        """
        if pqc_result.pqc_mode == PQCBridgeMode.PASSTHROUGH:
            return pqc_result.encrypted_data

        session = self._get_session(session_id or pqc_result.session_id)

        # Verify signature if present
        if pqc_result.signature:
            is_valid = await self.pqc_engine.verify(
                pqc_result.encrypted_data,
                pqc_result.signature,
                session.signature_keypair.public_key,
            )
            if not is_valid:
                raise BridgeException("Signature verification failed")

        # Decrypt if encrypted
        if pqc_result.kem_ciphertext:
            # Decapsulate to recover shared secret
            shared_secret = await self.pqc_engine.decapsulate(
                pqc_result.kem_ciphertext,
                session.kem_keypair.private_key,
            )

            if pqc_result.pqc_mode == PQCBridgeMode.HYBRID:
                # For hybrid mode, we need the classical secret too
                # In a real implementation, this would be stored/transmitted
                import secrets

                classical_secret = secrets.token_bytes(32)
                combined_secret = self._combine_secrets(classical_secret, shared_secret)
                decryption_key = self._derive_encryption_key(combined_secret)
            else:
                decryption_key = self._derive_encryption_key(shared_secret)

            return await self._aes_decrypt(pqc_result.encrypted_data, decryption_key)

        return pqc_result.encrypted_data

    def _derive_encryption_key(self, shared_secret: bytes) -> bytes:
        """Derive AES encryption key from shared secret using HKDF."""
        import hashlib
        import hmac

        # Simple HKDF-like derivation
        # In production, use proper HKDF implementation
        info = b"QBITEL-PQC-BRIDGE-AES-KEY"
        prk = hmac.new(b"qbitel-salt", shared_secret, hashlib.sha256).digest()
        okm = hmac.new(prk, info + b"\x01", hashlib.sha256).digest()
        return okm[:32]  # 256-bit key

    def _combine_secrets(self, classical: bytes, pqc: bytes) -> bytes:
        """Combine classical and PQC secrets for hybrid mode."""
        import hashlib

        combined = classical + pqc
        return hashlib.sha256(combined).digest()

    async def _aes_encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt data using AES-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import secrets

        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Return nonce + ciphertext
        return nonce + ciphertext

    async def _aes_decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt data using AES-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = ciphertext[:12]
        actual_ciphertext = ciphertext[12:]

        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, actual_ciphertext, None)

    async def _create_session(self, session_id: Optional[str] = None) -> PQCSession:
        """Create a new PQC session."""
        import secrets

        session_id = session_id or secrets.token_hex(16)

        # Generate session keypairs
        kem_keypair = await self.pqc_engine.generate_kem_keypair()
        signature_keypair = await self.pqc_engine.generate_signature_keypair()

        session = PQCSession(
            session_id=session_id,
            kem_keypair=kem_keypair,
            signature_keypair=signature_keypair,
            domain=self.bridge_config.domain,
        )

        self.sessions[session_id] = session
        self.metrics["active_sessions"] = len(self.sessions)

        self.logger.debug(f"Created PQC session: {session_id}")
        return session

    def _get_session(self, session_id: Optional[str] = None) -> PQCSession:
        """Get existing session or default session."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        return self.default_session

    async def _check_key_rotation(self, session: PQCSession) -> None:
        """Check if key rotation is needed for session."""
        should_rotate = False
        reason = None

        if self.bridge_config.rotation_policy == KeyRotationPolicy.TIME_BASED:
            if session.age_seconds > self.bridge_config.rotation_interval_seconds:
                should_rotate = True
                reason = "time_based"

        elif self.bridge_config.rotation_policy == KeyRotationPolicy.MESSAGE_COUNT:
            if session.message_count > self.bridge_config.rotation_message_count:
                should_rotate = True
                reason = "message_count"

        elif self.bridge_config.rotation_policy == KeyRotationPolicy.DATA_VOLUME:
            volume_mb = session.bytes_processed / (1024 * 1024)
            if volume_mb > self.bridge_config.rotation_data_volume_mb:
                should_rotate = True
                reason = "data_volume"

        if should_rotate:
            await self._rotate_session_keys(session, reason)

    async def _rotate_session_keys(self, session: PQCSession, reason: str) -> None:
        """Rotate keys for a session."""
        self.logger.info(f"Rotating keys for session {session.session_id}: {reason}")

        # Generate new keypairs
        session.kem_keypair = await self.pqc_engine.generate_kem_keypair()
        session.signature_keypair = await self.pqc_engine.generate_signature_keypair()

        # Reset counters
        session.message_count = 0
        session.bytes_processed = 0
        session.created_at = datetime.now(timezone.utc)

        # Update metrics
        self.metrics["key_rotations"] += 1
        PQC_BRIDGE_KEY_ROTATIONS.labels(reason=reason).inc()

    async def _key_rotation_loop(self) -> None:
        """Background task for periodic key rotation checks."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Clean up expired sessions
                expired_sessions = []
                for session_id, session in self.sessions.items():
                    if session_id == self.default_session.session_id:
                        continue
                    if session.age_seconds > self.bridge_config.session_timeout_seconds:
                        expired_sessions.append(session_id)

                for session_id in expired_sessions:
                    del self.sessions[session_id]
                    self.logger.debug(f"Expired session removed: {session_id}")

                self.metrics["active_sessions"] = len(self.sessions)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Key rotation loop error: {e}")

    def get_public_keys(self) -> Dict[str, bytes]:
        """Get public keys for key exchange."""
        return {
            "kem_public_key": self.master_kem_keypair.public_key,
            "signature_public_key": self.master_signature_keypair.public_key,
            "kem_algorithm": self.pqc_engine.kem_algorithm.value,
            "signature_algorithm": self.pqc_engine.signature_algorithm.value,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get PQC bridge metrics."""
        return {
            **self.metrics,
            "bridge_mode": self.bridge_config.mode.value,
            "domain": self.bridge_config.domain.value,
            "kem_algorithm": self.pqc_engine.kem_algorithm.value,
            "signature_algorithm": self.pqc_engine.signature_algorithm.value,
            "hybrid_mode": self.bridge_config.hybrid_mode,
            "fips_mode": self.bridge_config.fips_mode,
            "rotation_policy": self.bridge_config.rotation_policy.value,
        }

    async def shutdown(self) -> None:
        """Shutdown PQC bridge."""
        self.logger.info("Shutting down PQC protocol bridge...")

        # Cancel background tasks
        if self.key_rotation_task:
            self.key_rotation_task.cancel()
            try:
                await self.key_rotation_task
            except asyncio.CancelledError:
                pass

        # Clear sessions (securely zero keys in production)
        self.sessions.clear()
        self.default_session = None

        # Shutdown base bridge
        if self.base_bridge:
            await self.base_bridge.shutdown()

        self.is_initialized = False
        self.logger.info("PQC protocol bridge shutdown complete")


# Factory functions
async def create_pqc_bridge(
    config: Config,
    domain: DomainProfile = DomainProfile.ENTERPRISE,
    mode: PQCBridgeMode = PQCBridgeMode.FULL,
) -> PQCProtocolBridge:
    """Factory function to create PQC-protected protocol bridge."""
    bridge_config = PQCBridgeConfig(mode=mode, domain=domain)
    bridge = PQCProtocolBridge(config, bridge_config)
    await bridge.initialize()
    return bridge


async def create_government_pqc_bridge(config: Config) -> PQCProtocolBridge:
    """Create PQC bridge for government/classified use."""
    bridge_config = PQCBridgeConfig(
        mode=PQCBridgeMode.FULL,
        domain=DomainProfile.GOVERNMENT,
        kem_algorithm=PQCAlgorithm.MLKEM_1024,
        signature_algorithm=PQCAlgorithm.DILITHIUM_5,
        hybrid_mode=True,
        fips_mode=True,
        rotation_policy=KeyRotationPolicy.TIME_BASED,
        rotation_interval_seconds=1800,  # 30 minutes
    )
    bridge = PQCProtocolBridge(config, bridge_config)
    await bridge.initialize()
    return bridge


async def create_healthcare_pqc_bridge(config: Config) -> PQCProtocolBridge:
    """Create PQC bridge optimized for healthcare systems."""
    bridge_config = PQCBridgeConfig(
        mode=PQCBridgeMode.FULL,
        domain=DomainProfile.HEALTHCARE,
        kem_algorithm=PQCAlgorithm.MLKEM_512,  # Smaller for constrained devices
        signature_algorithm=PQCAlgorithm.FALCON_512,  # Smaller signatures
        hybrid_mode=False,  # Save bandwidth
    )
    bridge = PQCProtocolBridge(config, bridge_config)
    await bridge.initialize()
    return bridge
