"""
Cloud HSM Provider Implementations

Production-ready HSM integrations for major cloud providers:
- AWS CloudHSM
- Azure Dedicated HSM
- GCP Cloud HSM

All implementations follow the HSMProvider interface and provide:
- Connection pooling and failover
- PQC support (where available)
- Audit logging
- Health monitoring
"""

from ai_engine.domains.banking.security.hsm.cloud.aws_cloudhsm import (
    AWSCloudHSMProvider,
    AWSCloudHSMConfig,
)
from ai_engine.domains.banking.security.hsm.cloud.azure_hsm import (
    AzureDedicatedHSMProvider,
    AzureHSMConfig,
    AzureManagedHSMProvider,
)
from ai_engine.domains.banking.security.hsm.cloud.gcp_hsm import (
    GCPCloudHSMProvider,
    GCPHSMConfig,
)
from ai_engine.domains.banking.security.hsm.cloud.hsm_pool import (
    HSMConnectionPool,
    HSMPoolConfig,
    HSMLoadBalancer,
    PooledHSMSession,
)

__all__ = [
    # AWS
    "AWSCloudHSMProvider",
    "AWSCloudHSMConfig",
    # Azure
    "AzureDedicatedHSMProvider",
    "AzureHSMConfig",
    "AzureManagedHSMProvider",
    # GCP
    "GCPCloudHSMProvider",
    "GCPHSMConfig",
    # Connection pooling
    "HSMConnectionPool",
    "HSMPoolConfig",
    "HSMLoadBalancer",
    "PooledHSMSession",
]
