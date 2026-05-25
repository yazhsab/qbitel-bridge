"""
QBITEL - Post-Quantum Cryptography Module

This package provides a unified interface for post-quantum cryptography
operations across all NIST-standardized algorithms and domain-specific
optimizations for healthcare, automotive, aviation, and industrial environments.

Algorithms supported:
- ML-KEM (FIPS 203): Key Encapsulation at levels 512, 768, 1024
- ML-DSA (FIPS 204): Digital Signatures (Dilithium) at levels 2, 3, 5
- SLH-DSA (FIPS 205): Stateless Hash-based Signatures (SPHINCS+)
- LMS (NIST SP 800-208): Stateful Hash-based Signatures (RFC 8554)
- XMSS (NIST SP 800-208): eXtended Merkle Signature Scheme (RFC 8391)
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
- CNSA 2.0 Defense: NSA-mandated algorithm suite

Security:
    By default, strict mode is enabled. PQC engines will REFUSE to start
    if no real cryptographic library (kyber-py, dilithium-py, liboqs) is
    available. Set QBITEL_PQC_ALLOW_FALLBACK=1 environment variable to
    allow insecure test fallbacks (NEVER in production).
"""

from .pqc_unified import (
    PQCEngine,
    PQCAlgorithm,
    PQCSecurityLevel,
    DomainProfile,
    KeyPair,
    Signature,
    EncapsulationResult,
    create_cnsa2_engine,
)

from .mlkem import PQCProviderUnavailableError

from .providers import (
    CryptoProvider,
    ProviderRegistry,
    ProviderTier,
    PROVIDER_TIERS,
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

from .lms_xmss import (
    # LMS (RFC 8554)
    LmsEngine,
    LmsAlgorithm,
    LmotsAlgorithm,
    LmsKeyPair,
    LmsPublicKey,
    LmsPrivateKey,
    LmsSignature,
    # XMSS (RFC 8391)
    XmssEngine,
    XmssAlgorithm,
    XmssKeyPair,
    XmssPublicKey,
    XmssPrivateKey,
    XmssSignature,
    # State Management
    StateBackend,
    FileStateBackend,
    InMemoryStateBackend,
    StateExhaustedError,
    StateLockError,
    # CNSA 2.0
    CNSA2Profile,
    create_cnsa2_lms_engine,
    create_cnsa2_xmss_engine,
)

from .agility import (
    CryptoAgilityNegotiator,
    CryptoCapability,
    CryptoPolicy,
    NegotiatedSuite,
    AlgorithmDescriptor,
    AlgorithmFamily,
    AlgorithmStatus,
    SecurityEra,
    NegotiationFailedError,
    ALGORITHM_REGISTRY,
    TRANSITIONAL_POLICY,
    POST_QUANTUM_POLICY,
    CNSA2_POLICY,
    LEGACY_COMPATIBLE_POLICY,
)

from .quantum_threat_scoring import (
    QuantumThreatScorer,
    CryptoAsset,
    ThreatAssessment,
    PortfolioAssessment,
    DataSensitivity,
    MigrationPhase,
    RiskLevel,
    AlgorithmCategory,
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
    "create_cnsa2_engine",
    # Strict mode / errors
    "PQCProviderUnavailableError",
    # Provider registry
    "CryptoProvider",
    "ProviderRegistry",
    "ProviderTier",
    "PROVIDER_TIERS",
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
    # LMS (RFC 8554) — CNSA 2.0 stateful signatures
    "LmsEngine",
    "LmsAlgorithm",
    "LmotsAlgorithm",
    "LmsKeyPair",
    "LmsPublicKey",
    "LmsPrivateKey",
    "LmsSignature",
    # XMSS (RFC 8391) — CNSA 2.0 stateful signatures
    "XmssEngine",
    "XmssAlgorithm",
    "XmssKeyPair",
    "XmssPublicKey",
    "XmssPrivateKey",
    "XmssSignature",
    # State Management (for LMS/XMSS)
    "StateBackend",
    "FileStateBackend",
    "InMemoryStateBackend",
    "StateExhaustedError",
    "StateLockError",
    # CNSA 2.0 Compliance
    "CNSA2Profile",
    "create_cnsa2_lms_engine",
    "create_cnsa2_xmss_engine",
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
    # Crypto Agility Negotiation
    "CryptoAgilityNegotiator",
    "CryptoCapability",
    "CryptoPolicy",
    "NegotiatedSuite",
    "AlgorithmDescriptor",
    "AlgorithmFamily",
    "AlgorithmStatus",
    "SecurityEra",
    "NegotiationFailedError",
    "ALGORITHM_REGISTRY",
    "TRANSITIONAL_POLICY",
    "POST_QUANTUM_POLICY",
    "CNSA2_POLICY",
    "LEGACY_COMPATIBLE_POLICY",
    # Quantum Threat Scoring
    "QuantumThreatScorer",
    "CryptoAsset",
    "ThreatAssessment",
    "PortfolioAssessment",
    "DataSensitivity",
    "MigrationPhase",
    "RiskLevel",
    "AlgorithmCategory",
]

__version__ = "1.0.0"
