"""
PBX/ACD Integration Connectors

Provides quantum-safe communication bridges between QBITEL and
PBX/ACD systems for call control, agent state management, and
real-time event streaming.
"""

from typing import List

__all__: List[str] = [
    "PBXConnector",
    "PBXType",
    "PBXConnectionConfig",
    "PBXEvent",
    "PBXEventType",
]


def __getattr__(name: str):
    """Lazy import PBX connector classes."""
    if name in ("PBXConnector", "PBXType", "PBXConnectionConfig",
                "PBXEvent", "PBXEventType"):
        from ai_engine.domains.bpo.integrations.pbx.pbx_connector import (
            PBXConnector,
            PBXType,
            PBXConnectionConfig,
            PBXEvent,
            PBXEventType,
        )
        _map = {
            "PBXConnector": PBXConnector,
            "PBXType": PBXType,
            "PBXConnectionConfig": PBXConnectionConfig,
            "PBXEvent": PBXEvent,
            "PBXEventType": PBXEventType,
        }
        return _map[name]

    if name == "AvayaConnector":
        from ai_engine.domains.bpo.integrations.pbx.avaya_connector import AvayaConnector
        return AvayaConnector

    if name == "CiscoConnector":
        from ai_engine.domains.bpo.integrations.pbx.cisco_connector import CiscoConnector
        return CiscoConnector

    raise AttributeError(f"module 'ai_engine.domains.bpo.integrations.pbx' has no attribute '{name}'")
