"""
Automotive V2X Domain PQC Module

Optimized for Vehicle-to-Everything (V2X) communication:
- Sub-10ms signature verification latency
- 1000+ messages/second verification throughput
- Batch verification for dense traffic scenarios
- IEEE 1609.2 / SAE J2735 protocol support
- Backward compatibility with legacy vehicles

Key features:
- Falcon-512 for compact signatures (666 bytes vs 2420 for Dilithium)
- Parallel batch verification using SIMD
- P2PCD adaptation for large certificates
- SCMS (Security Credential Management System) integration
"""

from .batch_verification import (
    V2XBatchVerifier,
    BatchVerificationResult,
    VerificationMetrics,
)

from .v2x_protocol import (
    IEEE1609Protocol,
    BasicSafetyMessage,
    SignedPDU,
    V2XCertificate,
)

from .implicit_certificates import (
    PQImplicitCertificate,
    ImplicitCertificateAuthority,
    ButterflyKeyExpansion,
)

from .scms_integration import (
    SCMSClient,
    PseudonymCertificate,
    EnrollmentCertificate,
)

from .fleet_compatibility import (
    LegacyVehicleAdapter,
    HybridModeManager,
    CompatibilityProfile,
)

from .misbehavior_detection import (
    MisbehaviorAuthority,
    LinkableRingSigner,
    MisbehaviorReport,
    MisbehaviorScore,
    MisbehaviorType,
    SeverityLevel,
    MisbehaviorAction,
    LinkableRingSignature,
)

from .v2x_group_signatures import (
    V2XGroupSignatureProtocol,
    GroupManager,
    GroupSigner,
    GroupVerifier,
    GroupPublicKey,
    MemberPrivateKey,
    GroupSignature,
    GroupSignatureConfig,
    GroupSignatureScheme,
    LinkabilityWindow,
    RevocationEntry,
    RevocationReason,
    BatchVerifyResult,
    OpeningResult,
    GroupFullError,
    V2X_HIGHWAY_CONFIG,
    V2X_URBAN_CONFIG,
    V2X_INTERSECTION_CONFIG,
)

__all__ = [
    # Batch Verification
    "V2XBatchVerifier",
    "BatchVerificationResult",
    "VerificationMetrics",
    # Protocol
    "IEEE1609Protocol",
    "BasicSafetyMessage",
    "SignedPDU",
    "V2XCertificate",
    # Implicit Certificates
    "PQImplicitCertificate",
    "ImplicitCertificateAuthority",
    "ButterflyKeyExpansion",
    # SCMS
    "SCMSClient",
    "PseudonymCertificate",
    "EnrollmentCertificate",
    # Fleet Compatibility
    "LegacyVehicleAdapter",
    "HybridModeManager",
    "CompatibilityProfile",
    # V2X Group Signatures
    "V2XGroupSignatureProtocol",
    "GroupManager",
    "GroupSigner",
    "GroupVerifier",
    "GroupPublicKey",
    "MemberPrivateKey",
    "GroupSignature",
    "GroupSignatureConfig",
    "GroupSignatureScheme",
    "LinkabilityWindow",
    "RevocationEntry",
    "RevocationReason",
    "BatchVerifyResult",
    "OpeningResult",
    "GroupFullError",
    "V2X_HIGHWAY_CONFIG",
    "V2X_URBAN_CONFIG",
    "V2X_INTERSECTION_CONFIG",
    # Misbehavior Detection
    "MisbehaviorAuthority",
    "LinkableRingSigner",
    "MisbehaviorReport",
    "MisbehaviorScore",
    "MisbehaviorType",
    "SeverityLevel",
    "MisbehaviorAction",
    "LinkableRingSignature",
]
