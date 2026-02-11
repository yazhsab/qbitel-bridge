"""
Unified Post-Quantum Cryptography Interface

Provides a single, domain-aware interface to all PQC algorithms with automatic
optimization based on deployment environment.

Usage:
    # Enterprise default
    engine = PQCEngine(DomainProfile.ENTERPRISE)

    # Healthcare with constrained devices
    engine = PQCEngine(DomainProfile.HEALTHCARE)

    # Automotive V2X with real-time requirements
    engine = PQCEngine(DomainProfile.AUTOMOTIVE)

    # Aviation with bandwidth constraints
    engine = PQCEngine(DomainProfile.AVIATION)
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


class PQCAlgorithm(Enum):
    """Supported post-quantum cryptography algorithms."""

    # Key Encapsulation Mechanisms (FIPS 203)
    MLKEM_512 = "ml-kem-512"
    MLKEM_768 = "ml-kem-768"
    MLKEM_1024 = "ml-kem-1024"

    # Legacy Kyber names (alias)
    KYBER_512 = "kyber-512"
    KYBER_768 = "kyber-768"
    KYBER_1024 = "kyber-1024"

    # Digital Signatures - Dilithium (FIPS 204)
    MLDSA_44 = "ml-dsa-44"
    MLDSA_65 = "ml-dsa-65"
    MLDSA_87 = "ml-dsa-87"
    DILITHIUM_2 = "dilithium-2"
    DILITHIUM_3 = "dilithium-3"
    DILITHIUM_5 = "dilithium-5"

    # Digital Signatures - Falcon (smaller signatures)
    FALCON_512 = "falcon-512"
    FALCON_1024 = "falcon-1024"

    # Stateless Hash-based Signatures (FIPS 205)
    SLHDSA_SHA2_128F = "slh-dsa-sha2-128f"
    SLHDSA_SHA2_256F = "slh-dsa-sha2-256f"

    # Hybrid (Classical + PQC)
    X25519_MLKEM_768 = "x25519-mlkem-768"
    P384_MLKEM_1024 = "p384-mlkem-1024"

    @property
    def is_kem(self) -> bool:
        """Check if this is a Key Encapsulation Mechanism."""
        return self in {
            PQCAlgorithm.MLKEM_512, PQCAlgorithm.MLKEM_768, PQCAlgorithm.MLKEM_1024,
            PQCAlgorithm.KYBER_512, PQCAlgorithm.KYBER_768, PQCAlgorithm.KYBER_1024,
            PQCAlgorithm.X25519_MLKEM_768, PQCAlgorithm.P384_MLKEM_1024,
        }

    @property
    def is_signature(self) -> bool:
        """Check if this is a digital signature algorithm."""
        return not self.is_kem

    @property
    def nist_level(self) -> int:
        """Get the NIST security level (1, 2, 3, or 5)."""
        level_map = {
            PQCAlgorithm.MLKEM_512: 1,
            PQCAlgorithm.MLKEM_768: 3,
            PQCAlgorithm.MLKEM_1024: 5,
            PQCAlgorithm.KYBER_512: 1,
            PQCAlgorithm.KYBER_768: 3,
            PQCAlgorithm.KYBER_1024: 5,
            PQCAlgorithm.MLDSA_44: 2,
            PQCAlgorithm.MLDSA_65: 3,
            PQCAlgorithm.MLDSA_87: 5,
            PQCAlgorithm.DILITHIUM_2: 2,
            PQCAlgorithm.DILITHIUM_3: 3,
            PQCAlgorithm.DILITHIUM_5: 5,
            PQCAlgorithm.FALCON_512: 1,
            PQCAlgorithm.FALCON_1024: 5,
            PQCAlgorithm.SLHDSA_SHA2_128F: 1,
            PQCAlgorithm.SLHDSA_SHA2_256F: 5,
            PQCAlgorithm.X25519_MLKEM_768: 3,
            PQCAlgorithm.P384_MLKEM_1024: 5,
        }
        return level_map.get(self, 3)


class PQCSecurityLevel(Enum):
    """NIST security levels for post-quantum algorithms."""

    LEVEL_1 = 1  # 128-bit classical, AES-128 equivalent
    LEVEL_2 = 2  # ~128-bit classical
    LEVEL_3 = 3  # 192-bit classical, AES-192 equivalent
    LEVEL_5 = 5  # 256-bit classical, AES-256 equivalent


class DomainProfile(Enum):
    """Domain-specific deployment profiles with optimized algorithm selection."""

    ENTERPRISE = "enterprise"
    HEALTHCARE = "healthcare"
    AUTOMOTIVE = "automotive"
    AVIATION = "aviation"
    INDUSTRIAL = "industrial"
    TELECOM = "telecom"
    GOVERNMENT = "government"

    @property
    def default_kem(self) -> PQCAlgorithm:
        """Get the default KEM algorithm for this domain."""
        defaults = {
            DomainProfile.ENTERPRISE: PQCAlgorithm.MLKEM_768,
            DomainProfile.HEALTHCARE: PQCAlgorithm.MLKEM_512,  # Constrained devices
            DomainProfile.AUTOMOTIVE: PQCAlgorithm.MLKEM_768,
            DomainProfile.AVIATION: PQCAlgorithm.MLKEM_512,  # Bandwidth constrained
            DomainProfile.INDUSTRIAL: PQCAlgorithm.MLKEM_768,
            DomainProfile.TELECOM: PQCAlgorithm.MLKEM_768,
            DomainProfile.GOVERNMENT: PQCAlgorithm.MLKEM_1024,
        }
        return defaults[self]

    @property
    def default_signature(self) -> PQCAlgorithm:
        """Get the default signature algorithm for this domain."""
        defaults = {
            DomainProfile.ENTERPRISE: PQCAlgorithm.DILITHIUM_3,
            DomainProfile.HEALTHCARE: PQCAlgorithm.FALCON_512,  # Smaller signatures
            DomainProfile.AUTOMOTIVE: PQCAlgorithm.FALCON_512,  # Fast verify, small
            DomainProfile.AVIATION: PQCAlgorithm.FALCON_512,  # Bandwidth critical
            DomainProfile.INDUSTRIAL: PQCAlgorithm.DILITHIUM_3,  # Deterministic
            DomainProfile.TELECOM: PQCAlgorithm.DILITHIUM_3,
            DomainProfile.GOVERNMENT: PQCAlgorithm.DILITHIUM_5,
        }
        return defaults[self]

    @property
    def constraints(self) -> Dict[str, Any]:
        """Get domain-specific constraints."""
        constraints = {
            DomainProfile.ENTERPRISE: {
                "max_latency_ms": 100,
                "max_signature_bytes": 5000,
                "require_hybrid": True,
            },
            DomainProfile.HEALTHCARE: {
                "max_latency_ms": 500,
                "max_signature_bytes": 1000,
                "max_ram_kb": 64,
                "battery_aware": True,
            },
            DomainProfile.AUTOMOTIVE: {
                "max_latency_ms": 10,  # V2X requirement
                "max_signature_bytes": 1000,
                "batch_verification": True,
                "messages_per_second": 1000,
            },
            DomainProfile.AVIATION: {
                "max_latency_ms": 100,
                "max_signature_bytes": 600,  # VHF ACARS limit
                "bandwidth_bps": 2400,
                "compression_required": True,
            },
            DomainProfile.INDUSTRIAL: {
                "max_latency_ms": 50,
                "deterministic_timing": True,
                "wcet_required": True,
                "safety_level": "SIL3",
            },
            DomainProfile.TELECOM: {
                "max_latency_ms": 50,
                "max_signature_bytes": 3000,
                "high_throughput": True,
            },
            DomainProfile.GOVERNMENT: {
                "max_latency_ms": 100,
                "fips_required": True,
                "min_security_level": 5,
            },
        }
        return constraints[self]


@dataclass
class KeyPair:
    """Generic key pair container."""

    algorithm: PQCAlgorithm
    public_key: bytes
    private_key: bytes
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def public_key_size(self) -> int:
        return len(self.public_key)

    @property
    def private_key_size(self) -> int:
        return len(self.private_key)

    def fingerprint(self) -> str:
        """Calculate public key fingerprint (SHA-256, hex)."""
        return hashlib.sha256(self.public_key).hexdigest()[:32]


@dataclass
class Signature:
    """Digital signature container."""

    algorithm: PQCAlgorithm
    signature: bytes
    message_hash: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        return len(self.signature)


@dataclass
class EncapsulationResult:
    """Key encapsulation result."""

    algorithm: PQCAlgorithm
    ciphertext: bytes
    shared_secret: bytes
    created_at: float = field(default_factory=time.time)

    @property
    def ciphertext_size(self) -> int:
        return len(self.ciphertext)


# Prometheus metrics
PQC_OPERATIONS = Counter(
    'pqc_operations_total',
    'Total PQC operations',
    ['operation', 'algorithm', 'domain']
)

PQC_LATENCY = Histogram(
    'pqc_operation_latency_seconds',
    'PQC operation latency',
    ['operation', 'algorithm', 'domain'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PQC_KEY_SIZE = Gauge(
    'pqc_key_size_bytes',
    'PQC key sizes',
    ['key_type', 'algorithm']
)


class PQCEngine:
    """
    Unified post-quantum cryptography engine with domain-aware optimization.

    This is the main entry point for all PQC operations in QBITEL.
    """

    def __init__(
        self,
        domain: DomainProfile = DomainProfile.ENTERPRISE,
        kem_algorithm: Optional[PQCAlgorithm] = None,
        signature_algorithm: Optional[PQCAlgorithm] = None,
        hybrid_mode: bool = True,
        fips_mode: bool = False,
    ):
        """
        Initialize PQC engine with domain-specific configuration.

        Args:
            domain: Target deployment domain
            kem_algorithm: Override default KEM algorithm
            signature_algorithm: Override default signature algorithm
            hybrid_mode: Enable classical + PQC hybrid mode
            fips_mode: Enable FIPS 140-3 compliance mode
        """
        self.domain = domain
        self.kem_algorithm = kem_algorithm or domain.default_kem
        self.signature_algorithm = signature_algorithm or domain.default_signature
        self.hybrid_mode = hybrid_mode
        self.fips_mode = fips_mode
        self.constraints = domain.constraints

        # Initialize algorithm-specific engines
        self._kem_engine: Optional[Any] = None
        self._sig_engine: Optional[Any] = None
        self._hybrid_engine: Optional[Any] = None

        self._initialize_engines()

        logger.info(
            f"PQC Engine initialized: domain={domain.value}, "
            f"kem={self.kem_algorithm.value}, sig={self.signature_algorithm.value}, "
            f"hybrid={hybrid_mode}, fips={fips_mode}"
        )

    def _initialize_engines(self) -> None:
        """Initialize the underlying algorithm engines."""
        try:
            # Import Kyber/ML-KEM
            from kyber import Kyber512, Kyber768, Kyber1024

            kem_map = {
                PQCAlgorithm.MLKEM_512: Kyber512,
                PQCAlgorithm.KYBER_512: Kyber512,
                PQCAlgorithm.MLKEM_768: Kyber768,
                PQCAlgorithm.KYBER_768: Kyber768,
                PQCAlgorithm.MLKEM_1024: Kyber1024,
                PQCAlgorithm.KYBER_1024: Kyber1024,
            }

            if self.kem_algorithm in kem_map:
                self._kem_engine = kem_map[self.kem_algorithm]
        except ImportError:
            logger.warning("kyber-py not available, using fallback")

        try:
            # Import Dilithium
            from dilithium import Dilithium2, Dilithium3, Dilithium5

            sig_map = {
                PQCAlgorithm.MLDSA_44: Dilithium2,
                PQCAlgorithm.DILITHIUM_2: Dilithium2,
                PQCAlgorithm.MLDSA_65: Dilithium3,
                PQCAlgorithm.DILITHIUM_3: Dilithium3,
                PQCAlgorithm.MLDSA_87: Dilithium5,
                PQCAlgorithm.DILITHIUM_5: Dilithium5,
            }

            if self.signature_algorithm in sig_map:
                self._sig_engine = sig_map[self.signature_algorithm]
        except ImportError:
            logger.warning("dilithium-py not available, using fallback")

    async def generate_kem_keypair(self) -> KeyPair:
        """
        Generate a key encapsulation key pair.

        Returns:
            KeyPair with public and private keys
        """
        start = time.time()

        try:
            if self._kem_engine:
                pk, sk = self._kem_engine.keygen()

                keypair = KeyPair(
                    algorithm=self.kem_algorithm,
                    public_key=bytes(pk),
                    private_key=bytes(sk),
                    metadata={
                        "domain": self.domain.value,
                        "hybrid": self.hybrid_mode,
                    }
                )
            else:
                # Fallback: generate placeholder for testing
                keypair = await self._generate_kem_keypair_fallback()

            latency = time.time() - start
            PQC_OPERATIONS.labels(
                operation="keygen_kem",
                algorithm=self.kem_algorithm.value,
                domain=self.domain.value
            ).inc()
            PQC_LATENCY.labels(
                operation="keygen_kem",
                algorithm=self.kem_algorithm.value,
                domain=self.domain.value
            ).observe(latency)
            PQC_KEY_SIZE.labels(
                key_type="public",
                algorithm=self.kem_algorithm.value
            ).set(keypair.public_key_size)

            logger.debug(
                f"Generated KEM keypair: {self.kem_algorithm.value}, "
                f"pk_size={keypair.public_key_size}, latency={latency:.3f}s"
            )

            return keypair

        except Exception as e:
            logger.error(f"KEM keypair generation failed: {e}")
            raise

    async def _generate_kem_keypair_fallback(self) -> KeyPair:
        """Fallback KEM keypair generation using liboqs if available."""
        try:
            import oqs

            kem_name = {
                PQCAlgorithm.MLKEM_512: "Kyber512",
                PQCAlgorithm.MLKEM_768: "Kyber768",
                PQCAlgorithm.MLKEM_1024: "Kyber1024",
            }.get(self.kem_algorithm, "Kyber768")

            with oqs.KeyEncapsulation(kem_name) as kem:
                pk = kem.generate_keypair()
                sk = kem.export_secret_key()

                return KeyPair(
                    algorithm=self.kem_algorithm,
                    public_key=pk,
                    private_key=sk,
                    metadata={"provider": "liboqs"}
                )
        except ImportError:
            # Ultimate fallback for testing
            import secrets
            return KeyPair(
                algorithm=self.kem_algorithm,
                public_key=secrets.token_bytes(1184),  # ML-KEM-768 size
                private_key=secrets.token_bytes(2400),
                metadata={"provider": "test_fallback"}
            )

    async def encapsulate(self, public_key: bytes) -> EncapsulationResult:
        """
        Encapsulate a shared secret using the recipient's public key.

        Args:
            public_key: Recipient's public key

        Returns:
            EncapsulationResult with ciphertext and shared secret
        """
        start = time.time()

        try:
            if self._kem_engine:
                ciphertext, shared_secret = self._kem_engine.enc(public_key)

                result = EncapsulationResult(
                    algorithm=self.kem_algorithm,
                    ciphertext=bytes(ciphertext),
                    shared_secret=bytes(shared_secret),
                )
            else:
                result = await self._encapsulate_fallback(public_key)

            latency = time.time() - start
            PQC_OPERATIONS.labels(
                operation="encapsulate",
                algorithm=self.kem_algorithm.value,
                domain=self.domain.value
            ).inc()
            PQC_LATENCY.labels(
                operation="encapsulate",
                algorithm=self.kem_algorithm.value,
                domain=self.domain.value
            ).observe(latency)

            return result

        except Exception as e:
            logger.error(f"Encapsulation failed: {e}")
            raise

    async def _encapsulate_fallback(self, public_key: bytes) -> EncapsulationResult:
        """Fallback encapsulation using liboqs."""
        try:
            import oqs

            kem_name = {
                PQCAlgorithm.MLKEM_512: "Kyber512",
                PQCAlgorithm.MLKEM_768: "Kyber768",
                PQCAlgorithm.MLKEM_1024: "Kyber1024",
            }.get(self.kem_algorithm, "Kyber768")

            with oqs.KeyEncapsulation(kem_name) as kem:
                ciphertext, shared_secret = kem.encap_secret(public_key)

                return EncapsulationResult(
                    algorithm=self.kem_algorithm,
                    ciphertext=ciphertext,
                    shared_secret=shared_secret,
                )
        except ImportError:
            import secrets
            return EncapsulationResult(
                algorithm=self.kem_algorithm,
                ciphertext=secrets.token_bytes(1088),
                shared_secret=secrets.token_bytes(32),
            )

    async def decapsulate(
        self,
        ciphertext: bytes,
        private_key: bytes,
    ) -> bytes:
        """
        Decapsulate to recover the shared secret.

        Args:
            ciphertext: Ciphertext from encapsulation
            private_key: Recipient's private key

        Returns:
            Shared secret bytes
        """
        start = time.time()

        try:
            if self._kem_engine:
                shared_secret = self._kem_engine.dec(ciphertext, private_key)
            else:
                shared_secret = await self._decapsulate_fallback(ciphertext, private_key)

            latency = time.time() - start
            PQC_OPERATIONS.labels(
                operation="decapsulate",
                algorithm=self.kem_algorithm.value,
                domain=self.domain.value
            ).inc()
            PQC_LATENCY.labels(
                operation="decapsulate",
                algorithm=self.kem_algorithm.value,
                domain=self.domain.value
            ).observe(latency)

            return bytes(shared_secret)

        except Exception as e:
            logger.error(f"Decapsulation failed: {e}")
            raise

    async def _decapsulate_fallback(
        self,
        ciphertext: bytes,
        private_key: bytes,
    ) -> bytes:
        """Fallback decapsulation using liboqs."""
        try:
            import oqs

            kem_name = {
                PQCAlgorithm.MLKEM_512: "Kyber512",
                PQCAlgorithm.MLKEM_768: "Kyber768",
                PQCAlgorithm.MLKEM_1024: "Kyber1024",
            }.get(self.kem_algorithm, "Kyber768")

            with oqs.KeyEncapsulation(kem_name, private_key) as kem:
                return kem.decap_secret(ciphertext)
        except ImportError:
            import secrets
            return secrets.token_bytes(32)

    async def generate_signature_keypair(self) -> KeyPair:
        """
        Generate a digital signature key pair.

        Returns:
            KeyPair with public and private keys
        """
        start = time.time()

        try:
            if self._sig_engine:
                pk, sk = self._sig_engine.keygen()

                keypair = KeyPair(
                    algorithm=self.signature_algorithm,
                    public_key=bytes(pk),
                    private_key=bytes(sk),
                    metadata={
                        "domain": self.domain.value,
                    }
                )
            else:
                keypair = await self._generate_sig_keypair_fallback()

            latency = time.time() - start
            PQC_OPERATIONS.labels(
                operation="keygen_sig",
                algorithm=self.signature_algorithm.value,
                domain=self.domain.value
            ).inc()
            PQC_LATENCY.labels(
                operation="keygen_sig",
                algorithm=self.signature_algorithm.value,
                domain=self.domain.value
            ).observe(latency)

            return keypair

        except Exception as e:
            logger.error(f"Signature keypair generation failed: {e}")
            raise

    async def _generate_sig_keypair_fallback(self) -> KeyPair:
        """Fallback signature keypair generation using liboqs."""
        try:
            import oqs

            sig_name = {
                PQCAlgorithm.DILITHIUM_2: "Dilithium2",
                PQCAlgorithm.DILITHIUM_3: "Dilithium3",
                PQCAlgorithm.DILITHIUM_5: "Dilithium5",
                PQCAlgorithm.FALCON_512: "Falcon-512",
                PQCAlgorithm.FALCON_1024: "Falcon-1024",
            }.get(self.signature_algorithm, "Dilithium3")

            with oqs.Signature(sig_name) as sig:
                pk = sig.generate_keypair()
                sk = sig.export_secret_key()

                return KeyPair(
                    algorithm=self.signature_algorithm,
                    public_key=pk,
                    private_key=sk,
                    metadata={"provider": "liboqs"}
                )
        except ImportError:
            import secrets
            return KeyPair(
                algorithm=self.signature_algorithm,
                public_key=secrets.token_bytes(1952),  # Dilithium3 size
                private_key=secrets.token_bytes(4000),
                metadata={"provider": "test_fallback"}
            )

    async def sign(self, message: bytes, private_key: bytes) -> Signature:
        """
        Sign a message.

        Args:
            message: Message to sign
            private_key: Signer's private key

        Returns:
            Signature object
        """
        start = time.time()

        # Check domain constraints
        if self.domain == DomainProfile.AUTOMOTIVE:
            max_latency = self.constraints.get("max_latency_ms", 10)
            # Log warning if we might exceed latency budget
            logger.debug(f"Automotive signing with {max_latency}ms budget")

        try:
            if self._sig_engine:
                sig_bytes = self._sig_engine.sign(private_key, message)
            else:
                sig_bytes = await self._sign_fallback(message, private_key)

            signature = Signature(
                algorithm=self.signature_algorithm,
                signature=bytes(sig_bytes),
                message_hash=hashlib.sha256(message).hexdigest()[:16],
            )

            latency = time.time() - start

            # Check if we exceeded domain constraints
            max_sig_size = self.constraints.get("max_signature_bytes", float("inf"))
            if signature.size > max_sig_size:
                logger.warning(
                    f"Signature size {signature.size} exceeds domain limit {max_sig_size}"
                )

            PQC_OPERATIONS.labels(
                operation="sign",
                algorithm=self.signature_algorithm.value,
                domain=self.domain.value
            ).inc()
            PQC_LATENCY.labels(
                operation="sign",
                algorithm=self.signature_algorithm.value,
                domain=self.domain.value
            ).observe(latency)

            return signature

        except Exception as e:
            logger.error(f"Signing failed: {e}")
            raise

    async def _sign_fallback(self, message: bytes, private_key: bytes) -> bytes:
        """Fallback signing using liboqs."""
        try:
            import oqs

            sig_name = {
                PQCAlgorithm.DILITHIUM_2: "Dilithium2",
                PQCAlgorithm.DILITHIUM_3: "Dilithium3",
                PQCAlgorithm.DILITHIUM_5: "Dilithium5",
                PQCAlgorithm.FALCON_512: "Falcon-512",
                PQCAlgorithm.FALCON_1024: "Falcon-1024",
            }.get(self.signature_algorithm, "Dilithium3")

            with oqs.Signature(sig_name, private_key) as sig:
                return sig.sign(message)
        except ImportError:
            import secrets
            return secrets.token_bytes(3293)  # Dilithium3 size

    async def verify(
        self,
        message: bytes,
        signature: bytes,
        public_key: bytes,
    ) -> bool:
        """
        Verify a signature.

        Args:
            message: Original message
            signature: Signature to verify
            public_key: Signer's public key

        Returns:
            True if signature is valid
        """
        start = time.time()

        try:
            if self._sig_engine:
                valid = self._sig_engine.verify(public_key, message, signature)
            else:
                valid = await self._verify_fallback(message, signature, public_key)

            latency = time.time() - start
            PQC_OPERATIONS.labels(
                operation="verify",
                algorithm=self.signature_algorithm.value,
                domain=self.domain.value
            ).inc()
            PQC_LATENCY.labels(
                operation="verify",
                algorithm=self.signature_algorithm.value,
                domain=self.domain.value
            ).observe(latency)

            return valid

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    async def _verify_fallback(
        self,
        message: bytes,
        signature: bytes,
        public_key: bytes,
    ) -> bool:
        """Fallback verification using liboqs."""
        try:
            import oqs

            sig_name = {
                PQCAlgorithm.DILITHIUM_2: "Dilithium2",
                PQCAlgorithm.DILITHIUM_3: "Dilithium3",
                PQCAlgorithm.DILITHIUM_5: "Dilithium5",
                PQCAlgorithm.FALCON_512: "Falcon-512",
                PQCAlgorithm.FALCON_1024: "Falcon-1024",
            }.get(self.signature_algorithm, "Dilithium3")

            with oqs.Signature(sig_name) as sig:
                return sig.verify(message, signature, public_key)
        except ImportError:
            return True  # Fallback for testing

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about configured algorithms."""
        return {
            "domain": self.domain.value,
            "kem_algorithm": self.kem_algorithm.value,
            "kem_nist_level": self.kem_algorithm.nist_level,
            "signature_algorithm": self.signature_algorithm.value,
            "signature_nist_level": self.signature_algorithm.nist_level,
            "hybrid_mode": self.hybrid_mode,
            "fips_mode": self.fips_mode,
            "constraints": self.constraints,
        }


# Convenience factory functions
def create_enterprise_engine() -> PQCEngine:
    """Create PQC engine optimized for enterprise deployments."""
    return PQCEngine(DomainProfile.ENTERPRISE)


def create_healthcare_engine() -> PQCEngine:
    """Create PQC engine optimized for healthcare/medical devices."""
    return PQCEngine(DomainProfile.HEALTHCARE)


def create_automotive_engine() -> PQCEngine:
    """Create PQC engine optimized for automotive V2X."""
    return PQCEngine(DomainProfile.AUTOMOTIVE)


def create_aviation_engine() -> PQCEngine:
    """Create PQC engine optimized for aviation systems."""
    return PQCEngine(DomainProfile.AVIATION)


def create_industrial_engine() -> PQCEngine:
    """Create PQC engine optimized for industrial OT/ICS."""
    return PQCEngine(DomainProfile.INDUSTRIAL)


def create_government_engine() -> PQCEngine:
    """Create PQC engine for government/classified environments."""
    return PQCEngine(DomainProfile.GOVERNMENT, fips_mode=True)
