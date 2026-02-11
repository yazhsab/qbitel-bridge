"""
HSM Factory

Factory function for creating HSM provider instances based on configuration.
Supports both on-premise and cloud HSM providers.
"""

from typing import List, Optional, Union

from ai_engine.domains.banking.security.hsm.hsm_types import HSMConfig, HSMError
from ai_engine.domains.banking.security.hsm.hsm_provider import HSMProvider
from ai_engine.domains.banking.security.hsm.soft_hsm import SoftHSM
from ai_engine.domains.banking.security.hsm.thales_hsm import ThalesHSM
from ai_engine.domains.banking.security.hsm.futurex_hsm import FuturexHSM

# Cloud providers
from ai_engine.domains.banking.security.hsm.cloud.aws_cloudhsm import (
    AWSCloudHSMProvider,
    AWSCloudHSMConfig,
)
from ai_engine.domains.banking.security.hsm.cloud.azure_hsm import (
    AzureDedicatedHSMProvider,
    AzureManagedHSMProvider,
    AzureHSMConfig,
)
from ai_engine.domains.banking.security.hsm.cloud.gcp_hsm import (
    GCPCloudHSMProvider,
    GCPHSMConfig,
)
from ai_engine.domains.banking.security.hsm.cloud.hsm_pool import (
    HSMConnectionPool,
    HSMPoolConfig,
    HSMLoadBalancer,
    LoadBalanceStrategy,
)


def create_hsm_provider(
    config: Union[HSMConfig, AWSCloudHSMConfig, AzureHSMConfig, GCPHSMConfig]
) -> HSMProvider:
    """
    Create an HSM provider based on configuration.

    Supports:
    - On-premise: softhsm, thales, futurex, utimaco
    - Cloud: aws_cloudhsm, azure_dedicated_hsm, azure_managed_hsm, gcp_cloud_hsm

    Args:
        config: HSM configuration (type depends on provider)

    Returns:
        HSMProvider instance

    Raises:
        HSMError: If provider type is unknown or configuration is invalid
    """
    # Validate configuration
    errors = config.validate()
    if errors:
        raise HSMError(f"Invalid configuration: {', '.join(errors)}")

    provider_type = config.provider_type.lower()

    # On-premise providers
    if provider_type == "softhsm":
        return SoftHSM(config)
    elif provider_type == "thales":
        return ThalesHSM(config)
    elif provider_type == "futurex":
        return FuturexHSM(config)
    elif provider_type == "utimaco":
        raise HSMError("Utimaco HSM support not yet implemented")

    # Cloud providers
    elif provider_type == "aws_cloudhsm":
        if not isinstance(config, AWSCloudHSMConfig):
            raise HSMError("AWS CloudHSM requires AWSCloudHSMConfig")
        return AWSCloudHSMProvider(config)

    elif provider_type in ("azure_dedicated_hsm", "azure_hsm"):
        if not isinstance(config, AzureHSMConfig):
            raise HSMError("Azure HSM requires AzureHSMConfig")
        return AzureDedicatedHSMProvider(config)

    elif provider_type == "azure_managed_hsm":
        if not isinstance(config, AzureHSMConfig):
            raise HSMError("Azure Managed HSM requires AzureHSMConfig")
        return AzureManagedHSMProvider(config)

    elif provider_type == "gcp_cloud_hsm":
        if not isinstance(config, GCPHSMConfig):
            raise HSMError("GCP Cloud HSM requires GCPHSMConfig")
        return GCPCloudHSMProvider(config)

    else:
        raise HSMError(f"Unknown HSM provider type: {provider_type}")


def create_softhsm(enable_pqc: bool = True) -> SoftHSM:
    """
    Create a SoftHSM instance for development/testing.

    Args:
        enable_pqc: Enable post-quantum cryptography support

    Returns:
        SoftHSM instance
    """
    config = HSMConfig(
        provider_type="softhsm",
        enable_pqc=enable_pqc,
    )
    return SoftHSM(config)


def create_aws_cloudhsm(
    cluster_id: str,
    crypto_user_password: str,
    aws_region: str = "us-east-1",
    **kwargs,
) -> AWSCloudHSMProvider:
    """
    Create an AWS CloudHSM provider.

    Args:
        cluster_id: AWS CloudHSM cluster ID
        crypto_user_password: Crypto user password
        aws_region: AWS region
        **kwargs: Additional configuration options

    Returns:
        AWSCloudHSMProvider instance
    """
    config = AWSCloudHSMConfig(
        cluster_id=cluster_id,
        crypto_user_password=crypto_user_password,
        aws_region=aws_region,
        **kwargs,
    )
    return AWSCloudHSMProvider(config)


def create_azure_hsm(
    hsm_name: str,
    use_managed_identity: bool = True,
    vault_url: Optional[str] = None,
    managed: bool = False,
    **kwargs,
) -> Union[AzureDedicatedHSMProvider, AzureManagedHSMProvider]:
    """
    Create an Azure HSM provider.

    Args:
        hsm_name: Name of the HSM
        use_managed_identity: Use managed identity authentication
        vault_url: Vault URL for Managed HSM
        managed: If True, create Managed HSM provider
        **kwargs: Additional configuration options

    Returns:
        Azure HSM provider instance
    """
    config = AzureHSMConfig(
        hsm_name=hsm_name,
        use_managed_identity=use_managed_identity,
        vault_url=vault_url,
        **kwargs,
    )

    if managed or vault_url:
        return AzureManagedHSMProvider(config)
    else:
        return AzureDedicatedHSMProvider(config)


def create_gcp_hsm(
    project_id: str,
    key_ring_id: str = "qbitel-hsm-keyring",
    location: str = "us-east1",
    protection_level: str = "HSM",
    **kwargs,
) -> GCPCloudHSMProvider:
    """
    Create a GCP Cloud HSM provider.

    Args:
        project_id: GCP project ID
        key_ring_id: Key ring ID
        location: GCP location
        protection_level: HSM or SOFTWARE
        **kwargs: Additional configuration options

    Returns:
        GCPCloudHSMProvider instance
    """
    config = GCPHSMConfig(
        project_id=project_id,
        key_ring_id=key_ring_id,
        location=location,
        protection_level=protection_level,
        **kwargs,
    )
    return GCPCloudHSMProvider(config)


def create_hsm_pool(
    providers: List[HSMProvider],
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
    min_connections: int = 2,
    max_connections: int = 10,
    **kwargs,
) -> HSMConnectionPool:
    """
    Create an HSM connection pool with multiple providers.

    Args:
        providers: List of HSM providers
        strategy: Load balancing strategy
        min_connections: Minimum connections per provider
        max_connections: Maximum connections per provider
        **kwargs: Additional pool configuration

    Returns:
        HSMConnectionPool instance
    """
    config = HSMPoolConfig(
        min_connections=min_connections,
        max_connections=max_connections,
        load_balance_strategy=strategy,
        **kwargs,
    )
    return HSMConnectionPool(providers, config)


def create_hsm_load_balancer(
    providers: List[HSMProvider],
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.FAILOVER,
    **kwargs,
) -> HSMLoadBalancer:
    """
    Create an HSM load balancer for high availability.

    Args:
        providers: List of HSM providers (first is primary for failover strategy)
        strategy: Load balancing strategy
        **kwargs: Additional configuration

    Returns:
        HSMLoadBalancer instance
    """
    config = HSMPoolConfig(
        load_balance_strategy=strategy,
        **kwargs,
    )
    return HSMLoadBalancer(providers, config)
