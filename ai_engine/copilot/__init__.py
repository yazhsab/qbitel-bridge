"""
CRONOS AI - Protocol Intelligence Copilot
Natural language interface for protocol analysis and cybersecurity intelligence.
"""

__all__ = []

try:
    from .protocol_copilot import (
        ProtocolIntelligenceCopilot,
        CopilotQuery,
        CopilotResponse,
    )

    __all__.extend(["ProtocolIntelligenceCopilot", "CopilotQuery", "CopilotResponse"])
except Exception:  # pragma: no cover
    ProtocolIntelligenceCopilot = None  # type: ignore
    CopilotQuery = None  # type: ignore
    CopilotResponse = None  # type: ignore

try:
    from .context_manager import ConversationContextManager

    __all__.append("ConversationContextManager")
except Exception:  # pragma: no cover
    ConversationContextManager = None  # type: ignore

try:
    from .protocol_knowledge_base import ProtocolKnowledgeBase

    __all__.append("ProtocolKnowledgeBase")
except Exception:  # pragma: no cover
    ProtocolKnowledgeBase = None  # type: ignore
