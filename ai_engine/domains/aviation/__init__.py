"""
Aviation Domain PQC Module

Optimized for bandwidth-constrained aviation channels:
- VHF ACARS: 2.4 kbps
- Classic SATCOM: 600 bps
- LDACS: ~100 kbps (next-generation)
- ADS-B: Currently unencrypted/unauthenticated

Key features:
- Signature compression for bandwidth-constrained channels
- LDACS (L-band Digital Aeronautical Communications) integration
- ADS-B authentication with spoofing detection
- ARINC 653 partition support for IMA systems
- DO-326A/DO-356A certification evidence generation

Compression strategies:
- Zstandard dictionary compression (60-80% reduction)
- Delta encoding for sequential messages
- Session-based signature aggregation
"""

from .signature_compression import (
    SignatureCompressor,
    CompressionStrategy,
    CompressedSignature,
    DictionaryManager,
    create_acars_compressor,
    create_satcom_compressor,
    create_ldacs_compressor,
)

from .ldacs_profile import (
    LdacsSecurityProfile,
    LdacsKeyExchange,
    LdacsSecureChannel,
    LdacsSecurityContext,
    LdacsSecurityLevel,
    LdacsChannelType,
    LdacsAuthenticatedMessage,
)

from .adsb_authenticator import (
    AdsbAuthenticator,
    AdsbAuthenticationMode,
    AdsbAuthToken,
    AdsbAuthTokenGenerator,
    AdsbPosition,
    AdsbMessageType,
    AircraftTrackingState,
)

from .arinc653_partition import (
    ARINC653SecurityManager,
    PartitionCryptoEngine,
    PartitionSecurityContext,
    PartitionConfig,
    PartitionKeyManager,
    SecureInterPartitionChannel,
    SecurePort,
    SecureMessage,
    CriticalityLevel,
    PartitionMode,
    PortDirection,
    PortType,
)

from .certification_support import (
    PQCCertificationManager,
    CertificationRequirement,
    CertificationEvidence,
    CertificationStandard,
    TestCase,
    AlgorithmValidation,
    DesignAssuranceLevel,
    SecurityAssuranceLevel,
    RequirementType,
    create_pqc_aviation_requirements,
    create_pqc_test_cases,
)

__all__ = [
    # Compression
    "SignatureCompressor",
    "CompressionStrategy",
    "CompressedSignature",
    "DictionaryManager",
    "create_acars_compressor",
    "create_satcom_compressor",
    "create_ldacs_compressor",
    # LDACS
    "LdacsSecurityProfile",
    "LdacsKeyExchange",
    "LdacsSecureChannel",
    "LdacsSecurityContext",
    "LdacsSecurityLevel",
    "LdacsChannelType",
    "LdacsAuthenticatedMessage",
    # ADS-B
    "AdsbAuthenticator",
    "AdsbAuthenticationMode",
    "AdsbAuthToken",
    "AdsbAuthTokenGenerator",
    "AdsbPosition",
    "AdsbMessageType",
    "AircraftTrackingState",
    # ARINC 653
    "ARINC653SecurityManager",
    "PartitionCryptoEngine",
    "PartitionSecurityContext",
    "PartitionConfig",
    "PartitionKeyManager",
    "SecureInterPartitionChannel",
    "SecurePort",
    "SecureMessage",
    "CriticalityLevel",
    "PartitionMode",
    "PortDirection",
    "PortType",
    # Certification
    "PQCCertificationManager",
    "CertificationRequirement",
    "CertificationEvidence",
    "CertificationStandard",
    "TestCase",
    "AlgorithmValidation",
    "DesignAssuranceLevel",
    "SecurityAssuranceLevel",
    "RequirementType",
    "create_pqc_aviation_requirements",
    "create_pqc_test_cases",
]
