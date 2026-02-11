"""
Translation Module

Protocol translation and code generation capabilities:
- Protocol bridge for format conversion
- Code generation from protocol specifications
- API generation from protocol definitions
"""

from ai_engine.translation.protocol_bridge import (
    ProtocolBridge,
    ProtocolMapping,
    TranslationResult,
)
from ai_engine.translation.code_generation import (
    CodeGenerator,
    GeneratedCode,
)
from ai_engine.translation.api_generation import (
    APIGenerator,
    APIDefinition,
)

__all__ = [
    "ProtocolBridge",
    "ProtocolMapping",
    "TranslationResult",
    "CodeGenerator",
    "GeneratedCode",
    "APIGenerator",
    "APIDefinition",
]
