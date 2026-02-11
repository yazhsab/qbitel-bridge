"""
HSM (Hardware Security Module) Integration

Provides abstraction layer for HSM operations supporting:
- Multiple HSM vendors (Thales, Futurex, Utimaco, SoftHSM)
- Cloud HSM providers (AWS CloudHSM, Azure Dedicated/Managed HSM, GCP Cloud HSM)
- PKCS#11 interface
- Key generation and storage
- Cryptographic operations
- Post-quantum cryptography support
- Connection pooling and load balancing
"""

from ai_engine.domains.banking.security.hsm.hsm_types import (
    HSMKeyType,
    HSMAlgorithm,
    HSMConfig,
    HSMKeyHandle,
    HSMError,
    HSMConnectionError,
    HSMOperationError,
    HSMKeyNotFoundError,
    HSMAuthenticationError,
    HSMCapabilityError,
    HSMKeyExistsError,
)
from ai_engine.domains.banking.security.hsm.hsm_provider import (
    HSMProvider,
    HSMSession,
    EncryptionResult,
    DecryptionResult,
    SignatureResult,
    VerificationResult,
    KEMEncapsulationResult,
)
from ai_engine.domains.banking.security.hsm.soft_hsm import SoftHSM
from ai_engine.domains.banking.security.hsm.thales_hsm import ThalesHSM
from ai_engine.domains.banking.security.hsm.futurex_hsm import FuturexHSM
from ai_engine.domains.banking.security.hsm.hsm_factory import create_hsm_provider

# Cloud HSM providers
from ai_engine.domains.banking.security.hsm.cloud import (
    # AWS
    AWSCloudHSMProvider,
    AWSCloudHSMConfig,
    # Azure
    AzureDedicatedHSMProvider,
    AzureHSMConfig,
    AzureManagedHSMProvider,
    # GCP
    GCPCloudHSMProvider,
    GCPHSMConfig,
    # Connection pooling
    HSMConnectionPool,
    HSMPoolConfig,
    HSMLoadBalancer,
    PooledHSMSession,
)

__all__ = [
    # Types
    "HSMKeyType",
    "HSMAlgorithm",
    "HSMConfig",
    "HSMKeyHandle",
    "HSMError",
    "HSMConnectionError",
    "HSMOperationError",
    "HSMKeyNotFoundError",
    "HSMAuthenticationError",
    "HSMCapabilityError",
    "HSMKeyExistsError",
    # Provider base
    "HSMProvider",
    "HSMSession",
    "EncryptionResult",
    "DecryptionResult",
    "SignatureResult",
    "VerificationResult",
    "KEMEncapsulationResult",
    # On-premise implementations
    "SoftHSM",
    "ThalesHSM",
    "FuturexHSM",
    # Cloud providers
    "AWSCloudHSMProvider",
    "AWSCloudHSMConfig",
    "AzureDedicatedHSMProvider",
    "AzureHSMConfig",
    "AzureManagedHSMProvider",
    "GCPCloudHSMProvider",
    "GCPHSMConfig",
    # Connection pooling
    "HSMConnectionPool",
    "HSMPoolConfig",
    "HSMLoadBalancer",
    "PooledHSMSession",
    # Factory
    "create_hsm_provider",
]
