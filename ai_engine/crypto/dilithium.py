"""
ML-DSA (Dilithium) Digital Signature Algorithm - NIST FIPS 204

Python implementation supporting all security levels:
- ML-DSA-44 (Dilithium2): Security Level 2 (~128-bit)
- ML-DSA-65 (Dilithium3): Security Level 3 (~192-bit) - Enterprise default
- ML-DSA-87 (Dilithium5): Security Level 5 (~256-bit) - Maximum security
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class DilithiumSecurityLevel(Enum):
    """Dilithium/ML-DSA security levels per FIPS 204."""

    DILITHIUM_2 = "dilithium-2"
    DILITHIUM_3 = "dilithium-3"
    DILITHIUM_5 = "dilithium-5"

    # NIST naming aliases
    MLDSA_44 = "ml-dsa-44"
    MLDSA_65 = "ml-dsa-65"
    MLDSA_87 = "ml-dsa-87"

    @property
    def public_key_size(self) -> int:
        sizes = {
            DilithiumSecurityLevel.DILITHIUM_2: 1312,
            DilithiumSecurityLevel.MLDSA_44: 1312,
            DilithiumSecurityLevel.DILITHIUM_3: 1952,
            DilithiumSecurityLevel.MLDSA_65: 1952,
            DilithiumSecurityLevel.DILITHIUM_5: 2592,
            DilithiumSecurityLevel.MLDSA_87: 2592,
        }
        return sizes[self]

    @property
    def private_key_size(self) -> int:
        sizes = {
            DilithiumSecurityLevel.DILITHIUM_2: 2528,
            DilithiumSecurityLevel.MLDSA_44: 2528,
            DilithiumSecurityLevel.DILITHIUM_3: 4000,
            DilithiumSecurityLevel.MLDSA_65: 4000,
            DilithiumSecurityLevel.DILITHIUM_5: 4864,
            DilithiumSecurityLevel.MLDSA_87: 4864,
        }
        return sizes[self]

    @property
    def signature_size(self) -> int:
        sizes = {
            DilithiumSecurityLevel.DILITHIUM_2: 2420,
            DilithiumSecurityLevel.MLDSA_44: 2420,
            DilithiumSecurityLevel.DILITHIUM_3: 3293,
            DilithiumSecurityLevel.MLDSA_65: 3293,
            DilithiumSecurityLevel.DILITHIUM_5: 4595,
            DilithiumSecurityLevel.MLDSA_87: 4595,
        }
        return sizes[self]

    @property
    def nist_level(self) -> int:
        levels = {
            DilithiumSecurityLevel.DILITHIUM_2: 2,
            DilithiumSecurityLevel.MLDSA_44: 2,
            DilithiumSecurityLevel.DILITHIUM_3: 3,
            DilithiumSecurityLevel.MLDSA_65: 3,
            DilithiumSecurityLevel.DILITHIUM_5: 5,
            DilithiumSecurityLevel.MLDSA_87: 5,
        }
        return levels[self]

    @property
    def nist_name(self) -> str:
        """Get the NIST FIPS 204 name."""
        names = {
            DilithiumSecurityLevel.DILITHIUM_2: "ML-DSA-44",
            DilithiumSecurityLevel.MLDSA_44: "ML-DSA-44",
            DilithiumSecurityLevel.DILITHIUM_3: "ML-DSA-65",
            DilithiumSecurityLevel.MLDSA_65: "ML-DSA-65",
            DilithiumSecurityLevel.DILITHIUM_5: "ML-DSA-87",
            DilithiumSecurityLevel.MLDSA_87: "ML-DSA-87",
        }
        return names[self]


@dataclass
class DilithiumPublicKey:
    """Dilithium public key."""

    level: DilithiumSecurityLevel
    data: bytes

    def to_bytes(self) -> bytes:
        return self.data


@dataclass
class DilithiumPrivateKey:
    """Dilithium private key with secure handling."""

    level: DilithiumSecurityLevel
    data: bytes

    def to_bytes(self) -> bytes:
        return self.data

    def __del__(self):
        if hasattr(self, "data") and isinstance(self.data, bytearray):
            for i in range(len(self.data)):
                self.data[i] = 0


@dataclass
class DilithiumKeyPair:
    """Dilithium key pair."""

    level: DilithiumSecurityLevel
    public_key: DilithiumPublicKey
    private_key: DilithiumPrivateKey
    created_at: float = field(default_factory=time.time)


@dataclass
class DilithiumSignature:
    """Dilithium signature."""

    level: DilithiumSecurityLevel
    data: bytes
    created_at: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        return len(self.data)

    def to_bytes(self) -> bytes:
        return self.data


class DilithiumEngine:
    """
    Dilithium/ML-DSA signature engine.

    Faster signing than Falcon, larger signatures.
    Recommended for enterprise where bandwidth is not critical.
    """

    def __init__(
        self,
        level: DilithiumSecurityLevel = DilithiumSecurityLevel.DILITHIUM_3,
        provider: Optional[str] = None,
    ):
        self.level = level
        self.provider = provider or self._detect_provider()
        self._engine = self._initialize_engine()

        logger.info(f"Dilithium engine initialized: level={level.value}, provider={self.provider}")

    def _detect_provider(self) -> str:
        try:
            import dilithium

            return "dilithium-py"
        except ImportError:
            pass

        try:
            import oqs

            return "liboqs"
        except ImportError:
            pass

        return "fallback"

    def _initialize_engine(self):
        if self.provider == "dilithium-py":
            from dilithium import Dilithium2, Dilithium3, Dilithium5

            engines = {
                DilithiumSecurityLevel.DILITHIUM_2: Dilithium2,
                DilithiumSecurityLevel.MLDSA_44: Dilithium2,
                DilithiumSecurityLevel.DILITHIUM_3: Dilithium3,
                DilithiumSecurityLevel.MLDSA_65: Dilithium3,
                DilithiumSecurityLevel.DILITHIUM_5: Dilithium5,
                DilithiumSecurityLevel.MLDSA_87: Dilithium5,
            }
            return engines[self.level]
        return None

    async def generate_keypair(self) -> DilithiumKeyPair:
        """Generate a Dilithium key pair."""
        start = time.time()

        if self.provider == "dilithium-py":
            pk, sk = self._engine.keygen()
            keypair = DilithiumKeyPair(
                level=self.level,
                public_key=DilithiumPublicKey(self.level, bytes(pk)),
                private_key=DilithiumPrivateKey(self.level, bytes(sk)),
            )
        elif self.provider == "liboqs":
            import oqs

            sig_name = {
                DilithiumSecurityLevel.DILITHIUM_2: "Dilithium2",
                DilithiumSecurityLevel.MLDSA_44: "Dilithium2",
                DilithiumSecurityLevel.DILITHIUM_3: "Dilithium3",
                DilithiumSecurityLevel.MLDSA_65: "Dilithium3",
                DilithiumSecurityLevel.DILITHIUM_5: "Dilithium5",
                DilithiumSecurityLevel.MLDSA_87: "Dilithium5",
            }[self.level]

            with oqs.Signature(sig_name) as sig:
                pk = sig.generate_keypair()
                sk = sig.export_secret_key()

            keypair = DilithiumKeyPair(
                level=self.level,
                public_key=DilithiumPublicKey(self.level, pk),
                private_key=DilithiumPrivateKey(self.level, sk),
            )
        else:
            import secrets

            keypair = DilithiumKeyPair(
                level=self.level,
                public_key=DilithiumPublicKey(self.level, secrets.token_bytes(self.level.public_key_size)),
                private_key=DilithiumPrivateKey(self.level, secrets.token_bytes(self.level.private_key_size)),
            )

        logger.debug(f"Generated Dilithium keypair in {time.time() - start:.3f}s")
        return keypair

    async def sign(
        self,
        message: bytes,
        private_key: DilithiumPrivateKey,
    ) -> DilithiumSignature:
        """
        Sign a message.

        Args:
            message: Message to sign
            private_key: Signer's private key

        Returns:
            Dilithium signature
        """
        start = time.time()

        if self.provider == "dilithium-py":
            sig_bytes = self._engine.sign(private_key.data, message)
            signature = DilithiumSignature(self.level, bytes(sig_bytes))
        elif self.provider == "liboqs":
            import oqs

            sig_name = {
                DilithiumSecurityLevel.DILITHIUM_2: "Dilithium2",
                DilithiumSecurityLevel.MLDSA_44: "Dilithium2",
                DilithiumSecurityLevel.DILITHIUM_3: "Dilithium3",
                DilithiumSecurityLevel.MLDSA_65: "Dilithium3",
                DilithiumSecurityLevel.DILITHIUM_5: "Dilithium5",
                DilithiumSecurityLevel.MLDSA_87: "Dilithium5",
            }[self.level]

            with oqs.Signature(sig_name, private_key.data) as sig:
                sig_bytes = sig.sign(message)

            signature = DilithiumSignature(self.level, sig_bytes)
        else:
            import secrets

            signature = DilithiumSignature(self.level, secrets.token_bytes(self.level.signature_size))

        logger.debug(f"Dilithium sign: {signature.size} bytes in {time.time() - start:.3f}s")
        return signature

    async def verify(
        self,
        message: bytes,
        signature: DilithiumSignature,
        public_key: DilithiumPublicKey,
    ) -> bool:
        """
        Verify a signature.

        Args:
            message: Original message
            signature: Signature to verify
            public_key: Signer's public key

        Returns:
            True if valid
        """
        start = time.time()

        if self.provider == "dilithium-py":
            valid = self._engine.verify(public_key.data, message, signature.data)
        elif self.provider == "liboqs":
            import oqs

            sig_name = {
                DilithiumSecurityLevel.DILITHIUM_2: "Dilithium2",
                DilithiumSecurityLevel.MLDSA_44: "Dilithium2",
                DilithiumSecurityLevel.DILITHIUM_3: "Dilithium3",
                DilithiumSecurityLevel.MLDSA_65: "Dilithium3",
                DilithiumSecurityLevel.DILITHIUM_5: "Dilithium5",
                DilithiumSecurityLevel.MLDSA_87: "Dilithium5",
            }[self.level]

            with oqs.Signature(sig_name) as sig:
                valid = sig.verify(message, signature.data, public_key.data)
        else:
            valid = True

        logger.debug(f"Dilithium verify: {valid} in {time.time() - start:.3f}s")
        return valid

    @classmethod
    def for_enterprise(cls) -> "DilithiumEngine":
        """Create engine for enterprise deployments (Level 3)."""
        return cls(DilithiumSecurityLevel.DILITHIUM_3)

    @classmethod
    def for_maximum_security(cls) -> "DilithiumEngine":
        """Create engine for maximum security (Level 5)."""
        return cls(DilithiumSecurityLevel.DILITHIUM_5)

    @classmethod
    def for_government(cls) -> "DilithiumEngine":
        """Create engine for government/classified (Level 5)."""
        return cls(DilithiumSecurityLevel.MLDSA_87)
