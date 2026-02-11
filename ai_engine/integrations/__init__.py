"""
QBITEL - Ecosystem Integrations

External system integrations for PQC deployment:
- Cloud HSM: AWS CloudHSM, Azure Key Vault, GCP Cloud HSM
- Network: 5G/6G security integration
- Enterprise: LDAP, SAML, OAuth integration
- Compliance: FIPS 140-3, Common Criteria integration
"""

from .cloud_hsm import (
    CloudHSMProvider,
    CloudHSMManager,
    AWSCloudHSMAdapter,
    AzureKeyVaultAdapter,
    GCPCloudHSMAdapter,
    HSMKeyHandle,
    HSMSession,
)

from .telecom_5g import (
    FiveGSecurityManager,
    SUPIConcealment,
    NEFAuthenticator,
    NetworkSliceSecurityProfile,
    UEAuthenticator,
    FiveGKeyHierarchy,
)

__all__ = [
    # Cloud HSM
    "CloudHSMProvider",
    "CloudHSMManager",
    "AWSCloudHSMAdapter",
    "AzureKeyVaultAdapter",
    "GCPCloudHSMAdapter",
    "HSMKeyHandle",
    "HSMSession",
    # 5G/6G
    "FiveGSecurityManager",
    "SUPIConcealment",
    "NEFAuthenticator",
    "NetworkSliceSecurityProfile",
    "UEAuthenticator",
    "FiveGKeyHierarchy",
]
