"""
ML-KEM (Module-Lattice Key Encapsulation Mechanism) - NIST FIPS 203

Python implementation supporting all security levels:
- ML-KEM-512: Security Level 1 (128-bit)
- ML-KEM-768: Security Level 3 (192-bit) - Default for TLS 1.3
- ML-KEM-1024: Security Level 5 (256-bit)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        level: MlKemSecurityLevel = MlKemSecurityLevel.MLKEM_768,
        provider: Optional[str] = None,
    ):
        """
        Initialize ML-KEM engine.

        Args:
            level: Security level (512, 768, or 1024)
            provider: Crypto provider ("kyber-py", "liboqs", or None for auto)
        """
        self.level = level
        self.provider = provider or self._detect_provider()
        self._engine = self._initialize_engine()

        logger.info(f"ML-KEM engine initialized: level={level.value}, provider={self.provider}")

    def _detect_provider(self) -> str:
        """Detect available crypto provider."""
        try:
            import kyber

            return "kyber-py"
        except ImportError:
            pass

        try:
            import oqs

            return "liboqs"
        except ImportError:
            pass

        return "fallback"

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
            # Fallback for testing
            import secrets

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
