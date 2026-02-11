"""
QBITEL - Post-Quantum Cryptography Module

This package provides a unified interface for post-quantum cryptography
operations across all NIST-standardized algorithms and domain-specific
optimizations for healthcare, automotive, aviation, and industrial environments.

Algorithms supported:
- ML-KEM (FIPS 203): Key Encapsulation at levels 512, 768, 1024
- ML-DSA (FIPS 204): Digital Signatures (Dilithium) at levels 2, 3, 5
- SLH-DSA (FIPS 205): Stateless Hash-based Signatures (SPHINCS+)
- Falcon: Compact signatures for bandwidth-constrained environments
- Hybrid: X25519MLKEM768, P384MLKEM1024 for TLS 1.3

Advanced primitives:
- Zero-Knowledge Proofs (ZKP): Privacy-preserving verification
- Verifiable Random Functions (VRF): Verifiable randomness generation
- Threshold Signatures: Distributed signing (t-of-n)

Domain-specific modules:
- Healthcare: Constrained device support, FHIR integration
- Automotive: V2X real-time signatures, batch verification
- Aviation: Bandwidth-optimized signatures, LDACS support
- Industrial: Deterministic timing for safety-critical systems
"""

from .pqc_unified import (
    PQCEngine,
    PQCAlgorithm,
    PQCSecurityLevel,
    DomainProfile,
    KeyPair,
    Signature,
    EncapsulationResult,
)

from .mlkem import (
    MlKemEngine,
    MlKemSecurityLevel,
    MlKemKeyPair,
    MlKemPublicKey,
    MlKemPrivateKey,
    MlKemCiphertext,
    MlKemSharedSecret,
)

from .falcon import (
    FalconEngine,
    FalconSecurityLevel,
    FalconKeyPair,
    FalconSignature,
    FalconBatchVerifier,
)

from .dilithium import (
    DilithiumEngine,
    DilithiumSecurityLevel,
    DilithiumKeyPair,
    DilithiumSignature,
)

from .hybrid import (
    HybridKemEngine,
    HybridKexVariant,
    HybridKeyPair,
    HybridPublicKey,
    HybridCiphertext,
    HybridSharedSecret,
)

from .zkp import (
    ZKPEngine,
    ZKPType,
    ZKProof,
    RangeProof,
    Commitment,
    CommitmentScheme,
    SchnorrProtocol,
    RangeProofSystem,
    MembershipProof,
    IdentityProofSystem,
)

from .vrf import (
    VRFEngine,
    VRFSecurityLevel,
    VRFKeyPair,
    VRFOutput,
    VRFProof,
    DistributedVRF,
    RandomBeacon,
    LeaderElection,
    create_vrf_keypair,
    evaluate_vrf,
    verify_vrf,
)

from .threshold import (
    ThresholdSignatureScheme,
    ThresholdConfig,
    ThresholdSetup,
    ThresholdKeyShare,
    ThresholdSigner,
    ThresholdCombiner,
    ThresholdVerifier,
    ThresholdSignature,
    SignatureShare,
    ThresholdScheme,
    SecretSharing,
    create_threshold_scheme,
)

__all__ = [
    # Unified interface
    "PQCEngine",
    "PQCAlgorithm",
    "PQCSecurityLevel",
    "DomainProfile",
    "KeyPair",
    "Signature",
    "EncapsulationResult",
    # ML-KEM
    "MlKemEngine",
    "MlKemSecurityLevel",
    "MlKemKeyPair",
    "MlKemPublicKey",
    "MlKemPrivateKey",
    "MlKemCiphertext",
    "MlKemSharedSecret",
    # Falcon
    "FalconEngine",
    "FalconSecurityLevel",
    "FalconKeyPair",
    "FalconSignature",
    "FalconBatchVerifier",
    # Dilithium
    "DilithiumEngine",
    "DilithiumSecurityLevel",
    "DilithiumKeyPair",
    "DilithiumSignature",
    # Hybrid
    "HybridKemEngine",
    "HybridKexVariant",
    "HybridKeyPair",
    "HybridPublicKey",
    "HybridCiphertext",
    "HybridSharedSecret",
    # Zero-Knowledge Proofs
    "ZKPEngine",
    "ZKPType",
    "ZKProof",
    "RangeProof",
    "Commitment",
    "CommitmentScheme",
    "SchnorrProtocol",
    "RangeProofSystem",
    "MembershipProof",
    "IdentityProofSystem",
    # Verifiable Random Functions
    "VRFEngine",
    "VRFSecurityLevel",
    "VRFKeyPair",
    "VRFOutput",
    "VRFProof",
    "DistributedVRF",
    "RandomBeacon",
    "LeaderElection",
    "create_vrf_keypair",
    "evaluate_vrf",
    "verify_vrf",
    # Threshold Signatures
    "ThresholdSignatureScheme",
    "ThresholdConfig",
    "ThresholdSetup",
    "ThresholdKeyShare",
    "ThresholdSigner",
    "ThresholdCombiner",
    "ThresholdVerifier",
    "ThresholdSignature",
    "SignatureShare",
    "ThresholdScheme",
    "SecretSharing",
    "create_threshold_scheme",
]

__version__ = "1.0.0"
