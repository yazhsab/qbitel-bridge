"""
CRONOS AI - LLM Module
Unified LLM services and advanced AI capabilities.
"""

__all__ = []

try:
    from .unified_llm_service import (
        UnifiedLLMService,
        get_llm_service,
        LLMRequest,
        LLMResponse
    )

    __all__.extend([
        'UnifiedLLMService',
        'get_llm_service',
        'LLMRequest',
        'LLMResponse',
    ])
except Exception:  # pragma: no cover - optional dependency failures
    UnifiedLLMService = None  # type: ignore

try:
    from .rag_engine import (
        RAGEngine,
        RAGDocument,
        RAGQueryResult
    )

    __all__.extend([
        'RAGEngine',
        'RAGDocument',
        'RAGQueryResult',
    ])
except Exception:  # pragma: no cover - optional dependency failures
    RAGEngine = None  # type: ignore

try:
    from .legacy_whisperer import (
        LegacySystemWhisperer,
        ProtocolSpecification,
        AdapterCode,
        Explanation,
        AdapterLanguage,
        ProtocolComplexity,
        ModernizationRisk,
        create_legacy_whisperer
    )

    __all__.extend([
        'LegacySystemWhisperer',
        'ProtocolSpecification',
        'AdapterCode',
        'Explanation',
        'AdapterLanguage',
        'ProtocolComplexity',
        'ModernizationRisk',
        'create_legacy_whisperer',
    ])
except Exception:  # pragma: no cover - optional dependency failures
    LegacySystemWhisperer = None  # type: ignore
