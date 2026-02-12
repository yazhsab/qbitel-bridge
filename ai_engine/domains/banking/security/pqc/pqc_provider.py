"""
PQC Provider

Unified provider interface for post-quantum cryptographic operations.
Manages algorithm selection, key lifecycle, and migration support.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ai_engine.domains.banking.security.pqc.ml_kem import (
    MLKEM512,
    MLKEM768,
    MLKEM1024,
    MLKEMKeyPair,
    MLKEMEncapsulation,
    create_ml_kem,
)
from ai_engine.domains.banking.security.pqc.ml_dsa import (
    MLDSA44,
    MLDSA65,
    MLDSA87,
    MLDSAKeyPair,
    MLDSASignature,
    create_ml_dsa,
)
from ai_engine.domains.banking.security.pqc.hybrid import (
    HybridKEM,
    HybridSigner,
    HybridKeyPair,
    HybridEncapsulation,
    HybridSignature,
    HybridMode,
)


class PQCAlgorithm(Enum):
    """Supported PQC algorithms."""

    # Pure PQC - KEM
    ML_KEM_512 = ("ML-KEM-512", "kem", "level1")
    ML_KEM_768 = ("ML-KEM-768", "kem", "level3")
    ML_KEM_1024 = ("ML-KEM-1024", "kem", "level5")

    # Pure PQC - Signatures
    ML_DSA_44 = ("ML-DSA-44", "sig", "level2")
    ML_DSA_65 = ("ML-DSA-65", "sig", "level3")
    ML_DSA_87 = ("ML-DSA-87", "sig", "level5")

    # Hybrid KEM
    ECDH_ML_KEM_768 = ("ECDH-ML-KEM-768", "hybrid_kem", "level3")
    ECDH_ML_KEM_1024 = ("ECDH-ML-KEM-1024", "hybrid_kem", "level5")
    RSA_ML_KEM_768 = ("RSA-ML-KEM-768", "hybrid_kem", "level3")

    # Hybrid Signatures
    ECDSA_ML_DSA_65 = ("ECDSA-ML-DSA-65", "hybrid_sig", "level3")
    ECDSA_ML_DSA_87 = ("ECDSA-ML-DSA-87", "hybrid_sig", "level5")
    RSA_ML_DSA_65 = ("RSA-ML-DSA-65", "hybrid_sig", "level3")

    def __init__(self, name: str, op_type: str, security_level: str):
        self.algorithm_name = name
        self.operation_type = op_type
        self.security_level = security_level

    @property
    def is_kem(self) -> bool:
        return self.operation_type in ("kem", "hybrid_kem")

    @property
    def is_signature(self) -> bool:
        return self.operation_type in ("sig", "hybrid_sig")

    @property
    def is_hybrid(self) -> bool:
        return self.operation_type.startswith("hybrid")


@dataclass
class PQCConfig:
    """Configuration for PQC provider."""

    # Default algorithms
    default_kem: PQCAlgorithm = PQCAlgorithm.ML_KEM_768
    default_signature: PQCAlgorithm = PQCAlgorithm.ML_DSA_65

    # Hybrid mode settings
    prefer_hybrid: bool = True
    hybrid_kem: PQCAlgorithm = PQCAlgorithm.ECDH_ML_KEM_768
    hybrid_signature: PQCAlgorithm = PQCAlgorithm.ECDSA_ML_DSA_65

    # Security settings
    minimum_security_level: str = "level3"
    allow_pure_pqc: bool = True
    allow_pure_classical: bool = False  # For migration only

    # Key management
    key_rotation_days: int = 365
    auto_upgrade_algorithms: bool = True

    # Audit
    log_all_operations: bool = True


@dataclass
class PQCKeyInfo:
    """Information about a PQC key."""

    key_id: str
    algorithm: PQCAlgorithm
    created_at: datetime
    public_key_hash: str
    is_hybrid: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class PQCProvider:
    """
    Unified provider for post-quantum cryptographic operations.

    Features:
    - Algorithm selection and management
    - Key generation for all PQC algorithms
    - Encapsulation/decapsulation operations
    - Signing/verification operations
    - Hybrid mode support
    - Migration support from classical to PQC
    """

    def __init__(self, config: Optional[PQCConfig] = None):
        """
        Initialize PQC provider.

        Args:
            config: Provider configuration
        """
        self.config = config or PQCConfig()
        self._keys: Dict[str, Tuple[Any, PQCKeyInfo]] = {}
        self._operation_log: List[Dict[str, Any]] = []

        # Initialize algorithm instances
        self._kem_instances: Dict[PQCAlgorithm, Any] = {}
        self._dsa_instances: Dict[PQCAlgorithm, Any] = {}
        self._hybrid_kem_instances: Dict[PQCAlgorithm, HybridKEM] = {}
        self._hybrid_sig_instances: Dict[PQCAlgorithm, HybridSigner] = {}

    def generate_kem_keypair(
        self,
        algorithm: Optional[PQCAlgorithm] = None,
        key_id: Optional[str] = None,
    ) -> Tuple[str, Any]:
        """
        Generate a KEM key pair.

        Args:
            algorithm: KEM algorithm (defaults to config)
            key_id: Optional key identifier

        Returns:
            Tuple of (key_id, key_pair)
        """
        if algorithm is None:
            algorithm = self.config.hybrid_kem if self.config.prefer_hybrid else self.config.default_kem

        if not algorithm.is_kem:
            raise ValueError(f"Algorithm {algorithm} is not a KEM")

        # Generate key based on algorithm type
        if algorithm.is_hybrid:
            keypair = self._get_hybrid_kem(algorithm).generate_keypair()
        else:
            keypair = self._get_kem(algorithm).generate_keypair()

        # Generate key ID
        if key_id is None:
            import uuid

            key_id = str(uuid.uuid4())

        # Create key info
        pk_bytes = (
            keypair.public_key.to_bytes() if hasattr(keypair.public_key, "to_bytes") else bytes(keypair.public_key.key_bytes)
        )
        key_info = PQCKeyInfo(
            key_id=key_id,
            algorithm=algorithm,
            created_at=datetime.utcnow(),
            public_key_hash=hashlib.sha256(pk_bytes).hexdigest()[:16],
            is_hybrid=algorithm.is_hybrid,
        )

        self._keys[key_id] = (keypair, key_info)
        self._log_operation("generate_kem_keypair", algorithm, key_id)

        return key_id, keypair

    def generate_signature_keypair(
        self,
        algorithm: Optional[PQCAlgorithm] = None,
        key_id: Optional[str] = None,
    ) -> Tuple[str, Any]:
        """
        Generate a signature key pair.

        Args:
            algorithm: Signature algorithm (defaults to config)
            key_id: Optional key identifier

        Returns:
            Tuple of (key_id, key_pair)
        """
        if algorithm is None:
            algorithm = self.config.hybrid_signature if self.config.prefer_hybrid else self.config.default_signature

        if not algorithm.is_signature:
            raise ValueError(f"Algorithm {algorithm} is not a signature algorithm")

        # Generate key based on algorithm type
        if algorithm.is_hybrid:
            keypair = self._get_hybrid_signer(algorithm).generate_keypair()
        else:
            keypair = self._get_dsa(algorithm).generate_keypair()

        # Generate key ID
        if key_id is None:
            import uuid

            key_id = str(uuid.uuid4())

        # Create key info
        pk_bytes = (
            keypair.public_key.to_bytes() if hasattr(keypair.public_key, "to_bytes") else bytes(keypair.public_key.key_bytes)
        )
        key_info = PQCKeyInfo(
            key_id=key_id,
            algorithm=algorithm,
            created_at=datetime.utcnow(),
            public_key_hash=hashlib.sha256(pk_bytes).hexdigest()[:16],
            is_hybrid=algorithm.is_hybrid,
        )

        self._keys[key_id] = (keypair, key_info)
        self._log_operation("generate_signature_keypair", algorithm, key_id)

        return key_id, keypair

    def encapsulate(
        self,
        public_key: Any,
        algorithm: Optional[PQCAlgorithm] = None,
    ) -> Union[MLKEMEncapsulation, HybridEncapsulation]:
        """
        Encapsulate a shared secret.

        Args:
            public_key: Recipient's public key
            algorithm: Algorithm to use (inferred if not provided)

        Returns:
            Encapsulation result
        """
        if algorithm is None:
            algorithm = self._infer_kem_algorithm(public_key)

        if algorithm.is_hybrid:
            result = self._get_hybrid_kem(algorithm).encapsulate(public_key)
        else:
            result = self._get_kem(algorithm).encapsulate(public_key)

        self._log_operation("encapsulate", algorithm)
        return result

    def decapsulate(
        self,
        ciphertext: Any,
        private_key: Any,
        algorithm: Optional[PQCAlgorithm] = None,
    ) -> bytes:
        """
        Decapsulate to recover shared secret.

        Args:
            ciphertext: Encapsulation ciphertext
            private_key: Recipient's private key
            algorithm: Algorithm to use

        Returns:
            Shared secret (32 bytes)
        """
        if algorithm is None:
            algorithm = self._infer_kem_algorithm(private_key)

        if algorithm.is_hybrid:
            result = self._get_hybrid_kem(algorithm).decapsulate(ciphertext, private_key)
        else:
            result = self._get_kem(algorithm).decapsulate(ciphertext, private_key)

        self._log_operation("decapsulate", algorithm)
        return result

    def sign(
        self,
        message: bytes,
        private_key: Any,
        algorithm: Optional[PQCAlgorithm] = None,
        context: bytes = b"",
    ) -> Union[MLDSASignature, HybridSignature]:
        """
        Sign a message.

        Args:
            message: Message to sign
            private_key: Signer's private key
            algorithm: Algorithm to use
            context: Optional context for domain separation

        Returns:
            Signature
        """
        if algorithm is None:
            algorithm = self._infer_sig_algorithm(private_key)

        if algorithm.is_hybrid:
            result = self._get_hybrid_signer(algorithm).sign(message, private_key, context)
        else:
            result = self._get_dsa(algorithm).sign(message, private_key, context)

        self._log_operation("sign", algorithm, data_size=len(message))
        return result

    def verify(
        self,
        message: bytes,
        signature: Any,
        public_key: Any,
        algorithm: Optional[PQCAlgorithm] = None,
        context: bytes = b"",
    ) -> bool:
        """
        Verify a signature.

        Args:
            message: Original message
            signature: Signature to verify
            public_key: Signer's public key
            algorithm: Algorithm to use
            context: Context used during signing

        Returns:
            True if valid
        """
        if algorithm is None:
            algorithm = self._infer_sig_algorithm(public_key)

        if algorithm.is_hybrid:
            result = self._get_hybrid_signer(algorithm).verify(message, signature, public_key, context)
        else:
            result = self._get_dsa(algorithm).verify(message, signature, public_key, context)

        self._log_operation("verify", algorithm, success=result)
        return result

    def get_key(self, key_id: str) -> Optional[Tuple[Any, PQCKeyInfo]]:
        """Get key by ID."""
        return self._keys.get(key_id)

    def list_keys(self) -> List[PQCKeyInfo]:
        """List all managed keys."""
        return [info for _, info in self._keys.values()]

    def get_recommended_algorithm(
        self,
        operation: str,
        security_level: str = "level3",
    ) -> PQCAlgorithm:
        """
        Get recommended algorithm for operation and security level.

        Args:
            operation: "kem" or "signature"
            security_level: "level1", "level2", "level3", or "level5"

        Returns:
            Recommended algorithm
        """
        if self.config.prefer_hybrid:
            if operation == "kem":
                if security_level == "level5":
                    return PQCAlgorithm.ECDH_ML_KEM_1024
                return PQCAlgorithm.ECDH_ML_KEM_768
            else:
                if security_level == "level5":
                    return PQCAlgorithm.ECDSA_ML_DSA_87
                return PQCAlgorithm.ECDSA_ML_DSA_65
        else:
            if operation == "kem":
                level_map = {
                    "level1": PQCAlgorithm.ML_KEM_512,
                    "level3": PQCAlgorithm.ML_KEM_768,
                    "level5": PQCAlgorithm.ML_KEM_1024,
                }
            else:
                level_map = {
                    "level2": PQCAlgorithm.ML_DSA_44,
                    "level3": PQCAlgorithm.ML_DSA_65,
                    "level5": PQCAlgorithm.ML_DSA_87,
                }
            return level_map.get(security_level, list(level_map.values())[1])

    def get_migration_plan(
        self,
        current_algorithm: str,
        target_security_level: str = "level3",
    ) -> Dict[str, Any]:
        """
        Get migration plan from classical to PQC algorithm.

        Args:
            current_algorithm: Current classical algorithm
            target_security_level: Target PQC security level

        Returns:
            Migration plan with steps
        """
        classical_to_pqc = {
            # KEM equivalents
            "RSA-2048": PQCAlgorithm.ML_KEM_768,
            "RSA-3072": PQCAlgorithm.ML_KEM_768,
            "RSA-4096": PQCAlgorithm.ML_KEM_1024,
            "ECDH-P256": PQCAlgorithm.ML_KEM_768,
            "ECDH-P384": PQCAlgorithm.ML_KEM_768,
            "ECDH-P521": PQCAlgorithm.ML_KEM_1024,
            # Signature equivalents
            "RSA-PSS": PQCAlgorithm.ML_DSA_65,
            "ECDSA-P256": PQCAlgorithm.ML_DSA_65,
            "ECDSA-P384": PQCAlgorithm.ML_DSA_65,
            "ECDSA-P521": PQCAlgorithm.ML_DSA_87,
        }

        target_pqc = classical_to_pqc.get(current_algorithm)
        if not target_pqc:
            target_pqc = self.get_recommended_algorithm(
                "kem" if "RSA" in current_algorithm or "ECDH" in current_algorithm else "signature", target_security_level
            )

        # Get hybrid intermediate
        if target_pqc.is_kem:
            hybrid_intermediate = PQCAlgorithm.ECDH_ML_KEM_768
        else:
            hybrid_intermediate = PQCAlgorithm.ECDSA_ML_DSA_65

        return {
            "current_algorithm": current_algorithm,
            "target_algorithm": target_pqc.algorithm_name,
            "target_security_level": target_security_level,
            "recommended_path": [
                {
                    "phase": 1,
                    "name": "Hybrid Transition",
                    "algorithm": hybrid_intermediate.algorithm_name,
                    "description": "Deploy hybrid mode for backward compatibility",
                },
                {
                    "phase": 2,
                    "name": "Pure PQC",
                    "algorithm": target_pqc.algorithm_name,
                    "description": "Transition to pure PQC when ecosystem ready",
                },
            ],
            "compatibility_notes": [
                "Hybrid mode maintains security if either algorithm is broken",
                "Pure PQC requires all parties to support ML-KEM/ML-DSA",
                "Key sizes increase significantly (plan for storage/bandwidth)",
            ],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            "total_keys": len(self._keys),
            "operations_logged": len(self._operation_log),
            "algorithms_used": list(set(op.get("algorithm", "unknown") for op in self._operation_log)),
            "hybrid_preference": self.config.prefer_hybrid,
            "default_kem": self.config.default_kem.algorithm_name,
            "default_signature": self.config.default_signature.algorithm_name,
        }

    def _get_kem(self, algorithm: PQCAlgorithm):
        """Get or create KEM instance."""
        if algorithm not in self._kem_instances:
            level = algorithm.algorithm_name.split("-")[-1]
            self._kem_instances[algorithm] = create_ml_kem(level)
        return self._kem_instances[algorithm]

    def _get_dsa(self, algorithm: PQCAlgorithm):
        """Get or create DSA instance."""
        if algorithm not in self._dsa_instances:
            level = algorithm.algorithm_name.split("-")[-1]
            self._dsa_instances[algorithm] = create_ml_dsa(level)
        return self._dsa_instances[algorithm]

    def _get_hybrid_kem(self, algorithm: PQCAlgorithm) -> HybridKEM:
        """Get or create hybrid KEM instance."""
        if algorithm not in self._hybrid_kem_instances:
            mode_map = {
                PQCAlgorithm.ECDH_ML_KEM_768: HybridMode.ECDH_MLKEM_768,
                PQCAlgorithm.ECDH_ML_KEM_1024: HybridMode.ECDH_MLKEM_1024,
                PQCAlgorithm.RSA_ML_KEM_768: HybridMode.RSA_MLKEM_768,
            }
            self._hybrid_kem_instances[algorithm] = HybridKEM(mode_map[algorithm])
        return self._hybrid_kem_instances[algorithm]

    def _get_hybrid_signer(self, algorithm: PQCAlgorithm) -> HybridSigner:
        """Get or create hybrid signer instance."""
        if algorithm not in self._hybrid_sig_instances:
            mode_map = {
                PQCAlgorithm.ECDSA_ML_DSA_65: HybridMode.ECDSA_MLDSA_65,
                PQCAlgorithm.ECDSA_ML_DSA_87: HybridMode.ECDSA_MLDSA_87,
                PQCAlgorithm.RSA_ML_DSA_65: HybridMode.RSA_MLDSA_65,
            }
            self._hybrid_sig_instances[algorithm] = HybridSigner(mode_map[algorithm])
        return self._hybrid_sig_instances[algorithm]

    def _infer_kem_algorithm(self, key: Any) -> PQCAlgorithm:
        """Infer KEM algorithm from key type."""
        if isinstance(key, HybridKeyPair) or hasattr(key, "mode"):
            return self.config.hybrid_kem
        return self.config.default_kem

    def _infer_sig_algorithm(self, key: Any) -> PQCAlgorithm:
        """Infer signature algorithm from key type."""
        if isinstance(key, HybridKeyPair) or hasattr(key, "mode"):
            return self.config.hybrid_signature
        return self.config.default_signature

    def _log_operation(
        self,
        operation: str,
        algorithm: PQCAlgorithm,
        key_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log operation for audit."""
        if self.config.log_all_operations:
            self._operation_log.append(
                {
                    "operation": operation,
                    "algorithm": algorithm.algorithm_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "key_id": key_id,
                    **kwargs,
                }
            )
