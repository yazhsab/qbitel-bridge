"""
Post-Quantum Cryptography (PQC) Module

Production-ready implementation of NIST-standardized post-quantum algorithms:
- ML-KEM (Module-Lattice Key Encapsulation Mechanism) - FIPS 203
- ML-DSA (Module-Lattice Digital Signature Algorithm) - FIPS 204
- Hybrid modes combining classical and PQC algorithms

This module provides quantum-resistant cryptographic operations for
banking protocol security during the transition to post-quantum era.
"""

from ai_engine.domains.banking.security.pqc.ml_kem import (
    MLKEM512,
    MLKEM768,
    MLKEM1024,
    MLKEMKeyPair,
    MLKEMPublicKey,
    MLKEMPrivateKey,
    MLKEMEncapsulation,
)
from ai_engine.domains.banking.security.pqc.ml_dsa import (
    MLDSA44,
    MLDSA65,
    MLDSA87,
    MLDSAKeyPair,
    MLDSAPublicKey,
    MLDSAPrivateKey,
    MLDSASignature,
)
from ai_engine.domains.banking.security.pqc.hybrid import (
    HybridKEM,
    HybridSignature,
    HybridKeyPair,
    HybridMode,
)
from ai_engine.domains.banking.security.pqc.pqc_provider import (
    PQCProvider,
    PQCConfig,
    PQCAlgorithm,
)

__all__ = [
    # ML-KEM
    "MLKEM512",
    "MLKEM768",
    "MLKEM1024",
    "MLKEMKeyPair",
    "MLKEMPublicKey",
    "MLKEMPrivateKey",
    "MLKEMEncapsulation",
    # ML-DSA
    "MLDSA44",
    "MLDSA65",
    "MLDSA87",
    "MLDSAKeyPair",
    "MLDSAPublicKey",
    "MLDSAPrivateKey",
    "MLDSASignature",
    # Hybrid
    "HybridKEM",
    "HybridSignature",
    "HybridKeyPair",
    "HybridMode",
    # Provider
    "PQCProvider",
    "PQCConfig",
    "PQCAlgorithm",
]
