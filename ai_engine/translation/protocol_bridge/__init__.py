"""
Protocol Bridge Module

Real-time protocol-to-protocol translation with:
- Streaming translation support
- Caching and performance optimization
- Multiple protocol format support
- Post-Quantum Cryptography protection
"""

from ai_engine.translation.protocol_bridge.protocol_bridge import (
    ProtocolBridge,
    TranslationContext,
    TranslationResult,
    BridgeConnection,
    BridgeException,
)

from ai_engine.translation.protocol_bridge.pqc_bridge import (
    PQCProtocolBridge,
    PQCBridgeMode,
    PQCBridgeConfig,
    PQCSession,
    PQCTranslationResult,
    KeyRotationPolicy,
    create_pqc_bridge,
    create_government_pqc_bridge,
    create_healthcare_pqc_bridge,
)

__all__ = [
    # Base bridge
    "ProtocolBridge",
    "TranslationContext",
    "TranslationResult",
    "BridgeConnection",
    "BridgeException",
    # PQC bridge
    "PQCProtocolBridge",
    "PQCBridgeMode",
    "PQCBridgeConfig",
    "PQCSession",
    "PQCTranslationResult",
    "KeyRotationPolicy",
    # Factory functions
    "create_pqc_bridge",
    "create_government_pqc_bridge",
    "create_healthcare_pqc_bridge",
]
