"""
Healthcare Domain PQC Module

Optimized for medical device constraints:
- Ultra-low memory (<64KB RAM for pacemakers)
- 10+ year battery life requirements
- FDA 510(k) / PMA cybersecurity compliance
- HL7 FHIR security profile integration
- External security shield for legacy devices

Key features:
- Lightweight ML-KEM-512 for key exchange
- Falcon-512 for smaller signatures
- Battery-aware cryptographic scheduling
- HIPAA-compliant key management
"""

from .lightweight_pqc import (
    LightweightPQCEngine,
    ConstrainedDeviceProfile,
    MemoryConstraint,
)

from .security_shield import (
    MedicalDeviceSecurityShield,
    LegacyDeviceProxy,
    ShieldConfiguration,
)

from .fhir_pqc_profile import (
    FHIRPQCSecurityProfile,
    SmartOnFHIRPQC,
    UDAPPQCExtension,
)

from .battery_scheduler import (
    BatteryAwareCryptoScheduler,
    PowerProfile,
    CryptoOperation,
)

from .fda_compliance import (
    FDAComplianceValidator,
    CybersecurityRequirement,
    PremarketSubmission,
)

from .ehr_proxy_reencryption import (
    EHRProxyReEncryption,
    ProxyReEncryptionKey,
    EncryptedEHRRecord,
    ReEncryptedRecord,
    RecordCategory,
    ConsentType,
    ReEncryptionDeniedError,
)

from .homomorphic_vitals import (
    HomomorphicVitalEngine,
    EncryptedVital,
    EncryptedAggregate,
    DecryptedStatistic,
    VitalType,
    HEKeyPair,
)

from .verifiable_credentials import (
    HealthCredentialIssuer,
    HealthCredentialHolder,
    HealthCredentialVerifier,
    VerifiableCredential,
    VerifiablePresentation,
    CredentialType,
    CredentialSubject,
)

__all__ = [
    # Lightweight PQC
    "LightweightPQCEngine",
    "ConstrainedDeviceProfile",
    "MemoryConstraint",
    # Security Shield
    "MedicalDeviceSecurityShield",
    "LegacyDeviceProxy",
    "ShieldConfiguration",
    # FHIR Integration
    "FHIRPQCSecurityProfile",
    "SmartOnFHIRPQC",
    "UDAPPQCExtension",
    # Battery Scheduling
    "BatteryAwareCryptoScheduler",
    "PowerProfile",
    "CryptoOperation",
    # FDA Compliance
    "FDAComplianceValidator",
    "CybersecurityRequirement",
    "PremarketSubmission",
    # EHR Proxy Re-Encryption
    "EHRProxyReEncryption",
    "ProxyReEncryptionKey",
    "EncryptedEHRRecord",
    "ReEncryptedRecord",
    "RecordCategory",
    "ConsentType",
    "ReEncryptionDeniedError",
    # Homomorphic Vital Sign Analytics
    "HomomorphicVitalEngine",
    "EncryptedVital",
    "EncryptedAggregate",
    "DecryptedStatistic",
    "VitalType",
    "HEKeyPair",
    # Verifiable Health Credentials
    "HealthCredentialIssuer",
    "HealthCredentialHolder",
    "HealthCredentialVerifier",
    "VerifiableCredential",
    "VerifiablePresentation",
    "CredentialType",
    "CredentialSubject",
]
