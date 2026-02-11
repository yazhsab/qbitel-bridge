"""
Code Generation Module

Generate code from protocol specifications:
- Parser generation
- Serializer generation
- Validator generation
- SDK generation
"""

from ai_engine.translation.code_generation.code_generator import (
    CodeGenerator,
    GeneratedCode,
    CodeTemplate,
    TargetLanguage,
)

__all__ = [
    "CodeGenerator",
    "GeneratedCode",
    "CodeTemplate",
    "TargetLanguage",
]
