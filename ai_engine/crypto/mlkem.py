"""
ML-KEM (Module-Lattice Key Encapsulation Mechanism) - NIST FIPS 203

Python implementation supporting all security levels:
- ML-KEM-512: Security Level 1 (128-bit)
- ML-KEM-768: Security Level 3 (192-bit) - Default for TLS 1.3
- ML-KEM-1024: Security Level 5 (256-bit)

Security:
    Set strict_mode=True (default in production) to prevent fallback to
    random-byte generation when crypto libraries are unavailable.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from .providers import ProviderRegistry, _GLOBAL_STRICT_MODE

logger = logging.getLogger(__name__)


class PQCProviderUnavailableError(Exception):
    """Raised when no real PQC crypto provider is available and strict mode is enabled."""

    def __init__(self, algorithm: str, operation: str):
        self.algorithm = algorithm
        self.operation = operation
        super().__init__(
            f"CRITICAL: No PQC crypto provider available for {algorithm} {operation}. "
            f"Install 'kyber-py' or 'liboqs-python' for production use. "
            f"Set QBITEL_PQC_ALLOW_FALLBACK=1 to allow insecure test fallback (NEVER in production)."
        )


class MlKemSecurityLevel(Enum):
    """ML-KEM security levels per FIPS 203."""

    MLKEM_512 = "ml-kem-512"
    MLKEM_768 = "ml-kem-768"
    MLKEM_1024 = "ml-kem-1024"

    @property
    def public_key_size(self) -> int:
        sizes = {
            MlKemSecurityLevel.MLKEM_512: 800,
            MlKemSecurityLevel.MLKEM_768: 1184,
            MlKemSecurityLevel.MLKEM_1024: 1568,
        }
        return sizes[self]

    @property
    def private_key_size(self) -> int:
        sizes = {
            MlKemSecurityLevel.MLKEM_512: 1632,
            MlKemSecurityLevel.MLKEM_768: 2400,
            MlKemSecurityLevel.MLKEM_1024: 3168,
        }
        return sizes[self]

    @property
    def ciphertext_size(self) -> int:
        sizes = {
            MlKemSecurityLevel.MLKEM_512: 768,
            MlKemSecurityLevel.MLKEM_768: 1088,
            MlKemSecurityLevel.MLKEM_1024: 1568,
        }
        return sizes[self]

    @property
    def shared_secret_size(self) -> int:
        return 32  # All levels produce 256-bit shared secret

    @property
    def nist_level(self) -> int:
        levels = {
            MlKemSecurityLevel.MLKEM_512: 1,
            MlKemSecurityLevel.MLKEM_768: 3,
            MlKemSecurityLevel.MLKEM_1024: 5,
        }
        return levels[self]


@dataclass
class MlKemPublicKey:
    """ML-KEM public key."""

    level: MlKemSecurityLevel
    data: bytes

    def __post_init__(self):
        if len(self.data) != self.level.public_key_size:
            raise ValueError(f"Invalid public key size: expected {self.level.public_key_size}, " f"got {len(self.data)}")

    def to_bytes(self) -> bytes:
        return self.data


@dataclass
class MlKemPrivateKey:
    """ML-KEM private key with secure handling."""

    level: MlKemSecurityLevel
    data: bytes

    def __post_init__(self):
        if len(self.data) != self.level.private_key_size:
            raise ValueError(f"Invalid private key size: expected {self.level.private_key_size}, " f"got {len(self.data)}")

    def to_bytes(self) -> bytes:
        return self.data

    def __del__(self):
        # Zeroize on deletion
        if hasattr(self, "data") and isinstance(self.data, bytearray):
            for i in range(len(self.data)):
                self.data[i] = 0


@dataclass
class MlKemKeyPair:
    """ML-KEM key pair."""

    level: MlKemSecurityLevel
    public_key: MlKemPublicKey
    private_key: MlKemPrivateKey
    created_at: float = field(default_factory=time.time)


@dataclass
class MlKemCiphertext:
    """ML-KEM ciphertext."""

    level: MlKemSecurityLevel
    data: bytes

    def __post_init__(self):
        if len(self.data) != self.level.ciphertext_size:
            raise ValueError(f"Invalid ciphertext size: expected {self.level.ciphertext_size}, " f"got {len(self.data)}")

    def to_bytes(self) -> bytes:
        return self.data


@dataclass
class MlKemSharedSecret:
    """ML-KEM shared secret."""

    data: bytes

    def __post_init__(self):
        if len(self.data) != 32:
            raise ValueError(f"Invalid shared secret size: expected 32, got {len(self.data)}")

    def to_bytes(self) -> bytes:
        return self.data

    def __del__(self):
        # Zeroize on deletion
        if hasattr(self, "data") and isinstance(self.data, bytearray):
            for i in range(len(self.data)):
                self.data[i] = 0


class MlKemEngine:
    """
    ML-KEM cryptographic engine.

    Supports all three security levels with automatic provider selection.

    Args:
        level: Security level (512, 768, or 1024)
        provider: Crypto provider ("kyber-py", "liboqs", or None for auto)
        strict_mode: If True (default), raises PQCProviderUnavailableError when
            no real crypto library is available instead of silently falling back
            to random bytes. Set to False ONLY for unit testing.
    """

    def __init__(
        self,
        level: MlKemSecurityLevel = MlKemSecurityLevel.MLKEM_768,
        provider: Optional[str] = None,
        strict_mode: Optional[bool] = None,
    ):
        self.level = level
        self.strict_mode = strict_mode if strict_mode is not None else _GLOBAL_STRICT_MODE
        self.provider = provider or self._detect_provider()

        if self.provider == "fallback" and self.strict_mode:
            raise PQCProviderUnavailableError(level.value, "initialization")

        if self.provider == "fallback":
            logger.warning(
                f"ML-KEM engine using INSECURE fallback for {level.value}. "
                f"This MUST NOT be used in production."
            )

        self._engine = self._initialize_engine()

        logger.info(f"ML-KEM engine initialized: level={level.value}, provider={self.provider}")

    def _detect_provider(self) -> str:
        """Detect available crypto provider via the unified registry."""
        return ProviderRegistry().get_kem_provider(strict_mode=False)

    def _initialize_engine(self):
        """Initialize the underlying crypto engine."""
        if self.provider == "kyber-py":
            from kyber import Kyber512, Kyber768, Kyber1024

            engines = {
                MlKemSecurityLevel.MLKEM_512: Kyber512,
                MlKemSecurityLevel.MLKEM_768: Kyber768,
                MlKemSecurityLevel.MLKEM_1024: Kyber1024,
            }
            return engines[self.level]
        elif self.provider == "liboqs":
            return None  # Use OQS directly in methods
        else:
            return None

    async def generate_keypair(self) -> MlKemKeyPair:
        """Generate a new ML-KEM key pair."""
        start = time.time()

        if self.provider == "kyber-py":
            pk, sk = self._engine.keygen()
            keypair = MlKemKeyPair(
                level=self.level,
                public_key=MlKemPublicKey(self.level, bytes(pk)),
                private_key=MlKemPrivateKey(self.level, bytes(sk)),
            )
        elif self.provider == "liboqs":
            import oqs

            kem_name = {
                MlKemSecurityLevel.MLKEM_512: "Kyber512",
                MlKemSecurityLevel.MLKEM_768: "Kyber768",
                MlKemSecurityLevel.MLKEM_1024: "Kyber1024",
            }[self.level]

            with oqs.KeyEncapsulation(kem_name) as kem:
                pk = kem.generate_keypair()
                sk = kem.export_secret_key()

            keypair = MlKemKeyPair(
                level=self.level,
                public_key=MlKemPublicKey(self.level, pk),
                private_key=MlKemPrivateKey(self.level, sk),
            )
        else:
            # INSECURE fallback for testing only — strict_mode check already happened in __init__
            import secrets

            logger.critical(
                f"ML-KEM keygen using INSECURE random fallback for {self.level.value}. "
                f"Keys have NO quantum-safe security."
            )
            keypair = MlKemKeyPair(
                level=self.level,
                public_key=MlKemPublicKey(self.level, secrets.token_bytes(self.level.public_key_size)),
                private_key=MlKemPrivateKey(self.level, secrets.token_bytes(self.level.private_key_size)),
            )

        logger.debug(f"Generated ML-KEM keypair in {time.time() - start:.3f}s")
        return keypair

    async def encapsulate(
        self,
        public_key: MlKemPublicKey,
    ) -> Tuple[MlKemCiphertext, MlKemSharedSecret]:
        """
        Encapsulate a shared secret.

        Args:
            public_key: Recipient's public key

        Returns:
            Tuple of (ciphertext, shared_secret)
        """
        if public_key.level != self.level:
            raise ValueError(f"Security level mismatch: engine is {self.level}, key is {public_key.level}")

        start = time.time()

        if self.provider == "kyber-py":
            ct, ss = self._engine.enc(public_key.data)
            ciphertext = MlKemCiphertext(self.level, bytes(ct))
            shared_secret = MlKemSharedSecret(bytes(ss))
        elif self.provider == "liboqs":
            import oqs

            kem_name = {
                MlKemSecurityLevel.MLKEM_512: "Kyber512",
                MlKemSecurityLevel.MLKEM_768: "Kyber768",
                MlKemSecurityLevel.MLKEM_1024: "Kyber1024",
            }[self.level]

            with oqs.KeyEncapsulation(kem_name) as kem:
                ct, ss = kem.encap_secret(public_key.data)

            ciphertext = MlKemCiphertext(self.level, ct)
            shared_secret = MlKemSharedSecret(ss)
        else:
            import secrets

            logger.critical(
                f"ML-KEM encapsulate using INSECURE random fallback for {self.level.value}. "
                f"Shared secret has NO quantum-safe security."
            )
            ciphertext = MlKemCiphertext(self.level, secrets.token_bytes(self.level.ciphertext_size))
            shared_secret = MlKemSharedSecret(secrets.token_bytes(32))

        logger.debug(f"Encapsulated in {time.time() - start:.3f}s")
        return ciphertext, shared_secret

    async def decapsulate(
        self,
        ciphertext: MlKemCiphertext,
        private_key: MlKemPrivateKey,
    ) -> MlKemSharedSecret:
        """
        Decapsulate to recover the shared secret.

        Args:
            ciphertext: Ciphertext from encapsulation
            private_key: Recipient's private key

        Returns:
            Shared secret
        """
        if ciphertext.level != self.level or private_key.level != self.level:
            raise ValueError("Security level mismatch")

        start = time.time()

        if self.provider == "kyber-py":
            ss = self._engine.dec(ciphertext.data, private_key.data)
            shared_secret = MlKemSharedSecret(bytes(ss))
        elif self.provider == "liboqs":
            import oqs

            kem_name = {
                MlKemSecurityLevel.MLKEM_512: "Kyber512",
                MlKemSecurityLevel.MLKEM_768: "Kyber768",
                MlKemSecurityLevel.MLKEM_1024: "Kyber1024",
            }[self.level]

            with oqs.KeyEncapsulation(kem_name, private_key.data) as kem:
                ss = kem.decap_secret(ciphertext.data)

            shared_secret = MlKemSharedSecret(ss)
        else:
            import secrets

            logger.critical(
                f"ML-KEM decapsulate using INSECURE random fallback for {self.level.value}. "
                f"Shared secret has NO quantum-safe security."
            )
            shared_secret = MlKemSharedSecret(secrets.token_bytes(32))

        logger.debug(f"Decapsulated in {time.time() - start:.3f}s")
        return shared_secret

    @classmethod
    def for_tls_hybrid(cls) -> "MlKemEngine":
        """Create engine optimized for TLS 1.3 hybrid (ML-KEM-768)."""
        return cls(MlKemSecurityLevel.MLKEM_768)

    @classmethod
    def for_constrained_devices(cls) -> "MlKemEngine":
        """Create engine for constrained devices (ML-KEM-512)."""
        return cls(MlKemSecurityLevel.MLKEM_512)

    @classmethod
    def for_maximum_security(cls) -> "MlKemEngine":
        """Create engine for maximum security (ML-KEM-1024)."""
        return cls(MlKemSecurityLevel.MLKEM_1024)
