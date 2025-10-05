"""
CRONOS AI Engine - Security Orchestrator Integrations

Enterprise integrations with SIEM, ticketing, communication platforms,
and network security tools for comprehensive security orchestration.
"""

__all__ = []

# Base Connector
try:
    from .base_connector import (
        BaseIntegrationConnector,
        HTTPIntegrationConnector,
        IntegrationResult,
        IntegrationType,
        IntegrationConfig,
    )
    __all__.extend([
        "BaseIntegrationConnector",
        "HTTPIntegrationConnector",
        "IntegrationResult",
        "IntegrationType",
        "IntegrationConfig",
    ])
except Exception:  # pragma: no cover
    pass

# SIEM Connectors
try:
    from .siem_connector import SIEMConnector, SplunkConnector, QRadarConnector
    __all__.extend(["SIEMConnector", "SplunkConnector", "QRadarConnector"])
except Exception:  # pragma: no cover
    pass

# Ticketing Connectors
try:
    from .ticketing_connector import TicketingConnector, ServiceNowConnector, JiraConnector
    __all__.extend(["TicketingConnector", "ServiceNowConnector", "JiraConnector"])
except Exception:  # pragma: no cover
    pass

# Communication Connectors
try:
    from .communication_connector import (
        CommunicationConnector,
        SlackConnector,
        EmailConnector,
    )
    __all__.extend(["CommunicationConnector", "SlackConnector", "EmailConnector"])
except Exception:  # pragma: no cover
    pass

# Network Security Connectors
try:
    from .network_security_connector import (
        NetworkSecurityConnector,
        FirewallConnector,
        IDSConnector,
    )
    __all__.extend(["NetworkSecurityConnector", "FirewallConnector", "IDSConnector"])
except Exception:  # pragma: no cover
    pass

# Integration Manager
try:
    from .integration_manager import IntegrationManager, get_integration_manager
    __all__.extend(["IntegrationManager", "get_integration_manager"])
except Exception:  # pragma: no cover
    pass
