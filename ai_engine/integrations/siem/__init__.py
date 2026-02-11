"""
SIEM Integration Module

Native connectors for enterprise SIEM platforms:
- Splunk (HEC, REST API)
- Microsoft Sentinel (Log Analytics, Graph API)
- Google Chronicle (SIEM, SOAR)
- Generic Syslog/CEF

Features:
- Bidirectional data flow (ingest and query)
- Real-time event streaming
- Alert forwarding
- Threat intelligence sharing
- Correlation rule synchronization

Usage:
    from ai_engine.integrations.siem import SIEMConnectorFactory

    # Create Splunk connector
    splunk = SIEMConnectorFactory.create("splunk", config)
    await splunk.connect()

    # Send events
    await splunk.send_events(events)

    # Query alerts
    alerts = await splunk.query_alerts(time_range="24h")
"""

from ai_engine.integrations.siem.base import (
    BaseSIEMConnector,
    SIEMConfig,
    SIEMEvent,
    SIEMAlert,
    SIEMQuery,
    QueryResult,
    ConnectionStatus,
    SIEMCapability,
)
from ai_engine.integrations.siem.splunk import (
    SplunkConnector,
    SplunkConfig,
    SplunkHECConfig,
)
from ai_engine.integrations.siem.sentinel import (
    SentinelConnector,
    SentinelConfig,
)
from ai_engine.integrations.siem.chronicle import (
    ChronicleConnector,
    ChronicleConfig,
)
from ai_engine.integrations.siem.factory import (
    SIEMConnectorFactory,
    SIEMType,
)

__all__ = [
    # Base classes
    "BaseSIEMConnector",
    "SIEMConfig",
    "SIEMEvent",
    "SIEMAlert",
    "SIEMQuery",
    "QueryResult",
    "ConnectionStatus",
    "SIEMCapability",
    # Splunk
    "SplunkConnector",
    "SplunkConfig",
    "SplunkHECConfig",
    # Sentinel
    "SentinelConnector",
    "SentinelConfig",
    # Chronicle
    "ChronicleConnector",
    "ChronicleConfig",
    # Factory
    "SIEMConnectorFactory",
    "SIEMType",
]
