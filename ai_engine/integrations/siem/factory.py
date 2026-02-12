"""
SIEM Connector Factory

Factory pattern for creating SIEM connectors.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Type, Union

from ai_engine.integrations.siem.base import BaseSIEMConnector, SIEMConfig
from ai_engine.integrations.siem.splunk import SplunkConnector, SplunkConfig
from ai_engine.integrations.siem.sentinel import SentinelConnector, SentinelConfig
from ai_engine.integrations.siem.chronicle import ChronicleConnector, ChronicleConfig

logger = logging.getLogger(__name__)


class SIEMType(Enum):
    """Supported SIEM platforms."""

    SPLUNK = "splunk"
    SENTINEL = "sentinel"
    CHRONICLE = "chronicle"


# Registry of connector classes
_CONNECTOR_REGISTRY: Dict[SIEMType, Type[BaseSIEMConnector]] = {
    SIEMType.SPLUNK: SplunkConnector,
    SIEMType.SENTINEL: SentinelConnector,
    SIEMType.CHRONICLE: ChronicleConnector,
}

# Registry of config classes
_CONFIG_REGISTRY: Dict[SIEMType, Type[SIEMConfig]] = {
    SIEMType.SPLUNK: SplunkConfig,
    SIEMType.SENTINEL: SentinelConfig,
    SIEMType.CHRONICLE: ChronicleConfig,
}


class SIEMConnectorFactory:
    """
    Factory for creating SIEM connectors.

    Example:
        # Create from type and config dict
        connector = SIEMConnectorFactory.create(
            "splunk",
            {
                "host": "splunk.example.com",
                "hec": {"hec_token": "your-token"},
            }
        )

        # Create with typed config
        config = SplunkConfig(host="splunk.example.com")
        connector = SIEMConnectorFactory.create("splunk", config)

        # Create from environment
        connector = SIEMConnectorFactory.from_env("SIEM")
    """

    @staticmethod
    def create(
        siem_type: Union[str, SIEMType],
        config: Union[SIEMConfig, Dict[str, Any]],
    ) -> BaseSIEMConnector:
        """
        Create a SIEM connector.

        Args:
            siem_type: SIEM platform type
            config: Configuration (dict or typed config object)

        Returns:
            SIEM connector instance
        """
        # Normalize type
        if isinstance(siem_type, str):
            try:
                siem_type = SIEMType(siem_type.lower())
            except ValueError:
                raise ValueError(f"Unknown SIEM type: {siem_type}. " f"Supported: {[t.value for t in SIEMType]}")

        # Get connector class
        connector_class = _CONNECTOR_REGISTRY.get(siem_type)
        if not connector_class:
            raise ValueError(f"No connector registered for {siem_type}")

        # Create config if needed
        if isinstance(config, dict):
            config_class = _CONFIG_REGISTRY.get(siem_type, SIEMConfig)
            config = SIEMConnectorFactory._dict_to_config(config, config_class)

        # Validate config
        if not config.validate():
            raise ValueError("Invalid configuration")

        logger.info(f"Creating {siem_type.value} connector")
        return connector_class(config)

    @staticmethod
    def from_env(
        prefix: str = "SIEM",
        siem_type: Optional[Union[str, SIEMType]] = None,
    ) -> BaseSIEMConnector:
        """
        Create connector from environment variables.

        Environment variables:
            {PREFIX}_TYPE: SIEM type (splunk, sentinel, chronicle)
            {PREFIX}_HOST: SIEM host
            {PREFIX}_PORT: SIEM port
            {PREFIX}_TOKEN: API token
            ... (platform-specific variables)

        Args:
            prefix: Environment variable prefix
            siem_type: Override SIEM type

        Returns:
            SIEM connector instance
        """
        import os

        # Determine SIEM type
        if siem_type is None:
            type_str = os.getenv(f"{prefix}_TYPE")
            if not type_str:
                raise ValueError(f"{prefix}_TYPE environment variable not set")
            siem_type = SIEMType(type_str.lower())
        elif isinstance(siem_type, str):
            siem_type = SIEMType(siem_type.lower())

        # Build config from environment
        config_dict = {
            "host": os.getenv(f"{prefix}_HOST", ""),
            "port": int(os.getenv(f"{prefix}_PORT", "443")),
            "use_ssl": os.getenv(f"{prefix}_USE_SSL", "true").lower() == "true",
            "verify_ssl": os.getenv(f"{prefix}_VERIFY_SSL", "true").lower() == "true",
            "token": os.getenv(f"{prefix}_TOKEN"),
            "username": os.getenv(f"{prefix}_USERNAME"),
            "password": os.getenv(f"{prefix}_PASSWORD"),
            "api_key": os.getenv(f"{prefix}_API_KEY"),
        }

        # Platform-specific environment variables
        if siem_type == SIEMType.SPLUNK:
            config_dict["hec"] = {
                "hec_token": os.getenv(f"{prefix}_HEC_TOKEN", ""),
            }
            config_dict["index"] = os.getenv(f"{prefix}_INDEX", "main")

        elif siem_type == SIEMType.SENTINEL:
            config_dict["workspace_id"] = os.getenv(f"{prefix}_WORKSPACE_ID", "")
            config_dict["log_analytics_key"] = os.getenv(f"{prefix}_LOG_ANALYTICS_KEY", "")
            config_dict["tenant_id"] = os.getenv(f"{prefix}_TENANT_ID", "")
            config_dict["client_id"] = os.getenv(f"{prefix}_CLIENT_ID", "")
            config_dict["client_secret"] = os.getenv(f"{prefix}_CLIENT_SECRET", "")
            config_dict["subscription_id"] = os.getenv(f"{prefix}_SUBSCRIPTION_ID", "")
            config_dict["resource_group"] = os.getenv(f"{prefix}_RESOURCE_GROUP", "")
            config_dict["workspace_name"] = os.getenv(f"{prefix}_WORKSPACE_NAME", "")

        elif siem_type == SIEMType.CHRONICLE:
            config_dict["project_id"] = os.getenv(f"{prefix}_PROJECT_ID", "")
            config_dict["instance_id"] = os.getenv(f"{prefix}_INSTANCE_ID", "")
            config_dict["region"] = os.getenv(f"{prefix}_REGION", "us")
            config_dict["service_account_file"] = os.getenv(f"{prefix}_SERVICE_ACCOUNT_FILE")

        # Filter out None values
        config_dict = {k: v for k, v in config_dict.items() if v is not None}

        return SIEMConnectorFactory.create(siem_type, config_dict)

    @staticmethod
    def register(
        siem_type: SIEMType,
        connector_class: Type[BaseSIEMConnector],
        config_class: Type[SIEMConfig] = SIEMConfig,
    ) -> None:
        """
        Register a custom SIEM connector.

        Args:
            siem_type: SIEM type enum
            connector_class: Connector implementation class
            config_class: Configuration class
        """
        _CONNECTOR_REGISTRY[siem_type] = connector_class
        _CONFIG_REGISTRY[siem_type] = config_class
        logger.info(f"Registered SIEM connector: {siem_type.value}")

    @staticmethod
    def list_supported() -> list:
        """List supported SIEM platforms."""
        return [t.value for t in _CONNECTOR_REGISTRY.keys()]

    @staticmethod
    def _dict_to_config(
        config_dict: Dict[str, Any],
        config_class: Type[SIEMConfig],
    ) -> SIEMConfig:
        """Convert dictionary to typed config object."""
        # Handle nested config objects
        if config_class == SplunkConfig and "hec" in config_dict:
            from ai_engine.integrations.siem.splunk import SplunkHECConfig

            hec_dict = config_dict.pop("hec", {})
            if isinstance(hec_dict, dict):
                config_dict["hec"] = SplunkHECConfig(**hec_dict)

        # Create config object
        # Filter to only valid fields
        import dataclasses

        if dataclasses.is_dataclass(config_class):
            valid_fields = {f.name for f in dataclasses.fields(config_class)}
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
            return config_class(**filtered_dict)

        return config_class(**config_dict)


# Convenience function
def create_siem_connector(
    siem_type: Union[str, SIEMType],
    **config_kwargs,
) -> BaseSIEMConnector:
    """
    Convenience function to create a SIEM connector.

    Args:
        siem_type: SIEM platform type
        **config_kwargs: Configuration parameters

    Returns:
        SIEM connector instance
    """
    return SIEMConnectorFactory.create(siem_type, config_kwargs)
