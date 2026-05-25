"""
CRM Integration Bridges

Provides quantum-safe data bridges between contact center operations
and Customer Relationship Management platforms with built-in PII
protection and audit logging.
"""

from typing import List

__all__: List[str] = [
    "CRMBridge",
    "CRMType",
    "CRMConnectionConfig",
    "CustomerContext",
    "CRMEvent",
]


def __getattr__(name: str):
    """Lazy import CRM bridge classes."""
    if name in ("CRMBridge", "CRMType", "CRMConnectionConfig",
                "CustomerContext", "CRMEvent"):
        from ai_engine.domains.bpo.integrations.crm.crm_bridge import (
            CRMBridge,
            CRMType,
            CRMConnectionConfig,
            CustomerContext,
            CRMEvent,
        )
        _map = {
            "CRMBridge": CRMBridge,
            "CRMType": CRMType,
            "CRMConnectionConfig": CRMConnectionConfig,
            "CustomerContext": CustomerContext,
            "CRMEvent": CRMEvent,
        }
        return _map[name]

    raise AttributeError(f"module 'ai_engine.domains.bpo.integrations.crm' has no attribute '{name}'")
