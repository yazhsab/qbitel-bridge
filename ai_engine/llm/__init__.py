"""
QBITEL - LLM Module
Unified LLM services and advanced AI capabilities.

Updated for 2024-2025 with:
- Latest model versions (GPT-4o, Claude 4, Llama 3.2)
- Function calling / Tool use
- Structured JSON output
- Hybrid search RAG with multi-query expansion
- Contextual compression
- LangGraph workflows
- LLM observability with Langfuse
- Token cost tracking
- Semantic caching
- Intelligent model routing with cost optimization
- Prompt templates with versioning and A/B testing
- Guardrails for safety and validation
"""

__all__ = []

try:
    from .unified_llm_service import (
        UnifiedLLMService,
        get_llm_service,
        LLMRequest,
        LLMResponse,
        # New: Tool/Function calling
        ToolDefinition,
        ToolParameter,
        ToolCall,
        ToolResult,
        # New: Response formats
        ResponseFormat,
        # New: Model configuration
        ModelConfig,
        # New: Pre-built tools
        PROTOCOL_ANALYSIS_TOOLS,
        SECURITY_ANALYSIS_TOOLS,
        COMPLIANCE_TOOLS,
    )

    __all__.extend(
        [
            "UnifiedLLMService",
            "get_llm_service",
            "LLMRequest",
            "LLMResponse",
            "ToolDefinition",
            "ToolParameter",
            "ToolCall",
            "ToolResult",
            "ResponseFormat",
            "ModelConfig",
            "PROTOCOL_ANALYSIS_TOOLS",
            "SECURITY_ANALYSIS_TOOLS",
            "COMPLIANCE_TOOLS",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    UnifiedLLMService = None  # type: ignore

try:
    from .rag_engine import (
        RAGEngine,
        RAGDocument,
        RAGQueryResult,
        # New: Search modes
        SearchMode,
        # New: Configuration
        EmbeddingModelConfig,
        RerankerModelConfig,
    )

    __all__.extend(
        [
            "RAGEngine",
            "RAGDocument",
            "RAGQueryResult",
            "SearchMode",
            "EmbeddingModelConfig",
            "RerankerModelConfig",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    RAGEngine = None  # type: ignore

# New: LangGraph Workflow Orchestration
try:
    from .langgraph_workflows import (
        WorkflowManager,
        WorkflowStatus,
        WorkflowCheckpoint,
        ProtocolAnalysisState,
        SecurityOrchestrationState,
        ComplianceCheckState,
        get_workflow_manager,
        set_workflow_manager,
    )

    __all__.extend(
        [
            "WorkflowManager",
            "WorkflowStatus",
            "WorkflowCheckpoint",
            "ProtocolAnalysisState",
            "SecurityOrchestrationState",
            "ComplianceCheckState",
            "get_workflow_manager",
            "set_workflow_manager",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    WorkflowManager = None  # type: ignore

# New: LLM Observability
try:
    from .observability import (
        LLMObservabilityManager,
        LLMTrace,
        TraceStatus,
        TraceMetadata,
        TokenUsage,
        PromptVersion,
        TokenCostConfig,
        CostTracker,
        get_observability_manager,
        configure_observability,
    )

    __all__.extend(
        [
            "LLMObservabilityManager",
            "LLMTrace",
            "TraceStatus",
            "TraceMetadata",
            "TokenUsage",
            "PromptVersion",
            "TokenCostConfig",
            "CostTracker",
            "get_observability_manager",
            "configure_observability",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    LLMObservabilityManager = None  # type: ignore

try:
    from .legacy_whisperer import (
        LegacySystemWhisperer,
        ProtocolSpecification,
        AdapterCode,
        Explanation,
        AdapterLanguage,
        ProtocolComplexity,
        ModernizationRisk,
        create_legacy_whisperer,
    )

    __all__.extend(
        [
            "LegacySystemWhisperer",
            "ProtocolSpecification",
            "AdapterCode",
            "Explanation",
            "AdapterLanguage",
            "ProtocolComplexity",
            "ModernizationRisk",
            "create_legacy_whisperer",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    LegacySystemWhisperer = None  # type: ignore

# New: Advanced RAG (Multi-query expansion, Contextual compression)
try:
    from .advanced_rag import (
        QueryExpander,
        QueryExpansionStrategy,
        ExpandedQuery,
        ContextualCompressor,
        CompressionStrategy,
        CompressedContext,
        FusionRetriever,
        MultiQueryResult,
        AdvancedRAGPipeline,
        create_advanced_rag_pipeline,
    )

    __all__.extend(
        [
            "QueryExpander",
            "QueryExpansionStrategy",
            "ExpandedQuery",
            "ContextualCompressor",
            "CompressionStrategy",
            "CompressedContext",
            "FusionRetriever",
            "MultiQueryResult",
            "AdvancedRAGPipeline",
            "create_advanced_rag_pipeline",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    AdvancedRAGPipeline = None  # type: ignore

# New: Semantic Caching
try:
    from .semantic_cache import (
        SemanticCache,
        CacheEntry,
        CacheHitResult,
        CacheHitType,
        CacheStats,
        CachedLLMService,
        get_semantic_cache,
        configure_semantic_cache,
    )

    __all__.extend(
        [
            "SemanticCache",
            "CacheEntry",
            "CacheHitResult",
            "CacheHitType",
            "CacheStats",
            "CachedLLMService",
            "get_semantic_cache",
            "configure_semantic_cache",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    SemanticCache = None  # type: ignore

# New: Intelligent Model Routing
try:
    from .model_routing import (
        IntelligentRouter,
        ModelSpec,
        ModelTier,
        TaskComplexity,
        RoutingDecision,
        BudgetConfig,
        BudgetStatus,
        BudgetManager,
        ComplexityAssessor,
        RoutedLLMService,
        get_router,
        configure_router,
    )

    __all__.extend(
        [
            "IntelligentRouter",
            "ModelSpec",
            "ModelTier",
            "TaskComplexity",
            "RoutingDecision",
            "BudgetConfig",
            "BudgetStatus",
            "BudgetManager",
            "ComplexityAssessor",
            "RoutedLLMService",
            "get_router",
            "configure_router",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    IntelligentRouter = None  # type: ignore

# New: Prompt Templates with Versioning
try:
    from .prompt_templates import (
        PromptTemplateManager,
        PromptTemplate,
        TemplateVersion,
        TemplateVariable,
        TemplateStatus,
        VariableType,
        RenderedPrompt,
        ABTestConfig,
        ABTestResult,
        get_template_manager,
        configure_template_manager,
    )

    __all__.extend(
        [
            "PromptTemplateManager",
            "PromptTemplate",
            "TemplateVersion",
            "TemplateVariable",
            "TemplateStatus",
            "VariableType",
            "RenderedPrompt",
            "ABTestConfig",
            "ABTestResult",
            "get_template_manager",
            "configure_template_manager",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    PromptTemplateManager = None  # type: ignore

# New: Guardrails for Safety and Validation
try:
    from .guardrails import (
        GuardrailManager,
        GuardrailResult,
        GuardrailViolation,
        GuardrailType,
        SeverityLevel,
        ViolationType,
        ContentFilter,
        PIIDetector,
        PIIPattern,
        OutputValidator,
        HallucinationDetector,
        RateLimiter,
        RateLimitConfig,
        GuardedLLMService,
        GuardrailException,
        get_guardrail_manager,
        configure_guardrails,
    )

    __all__.extend(
        [
            "GuardrailManager",
            "GuardrailResult",
            "GuardrailViolation",
            "GuardrailType",
            "SeverityLevel",
            "ViolationType",
            "ContentFilter",
            "PIIDetector",
            "PIIPattern",
            "OutputValidator",
            "HallucinationDetector",
            "RateLimiter",
            "RateLimitConfig",
            "GuardedLLMService",
            "GuardrailException",
            "get_guardrail_manager",
            "configure_guardrails",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    GuardrailManager = None  # type: ignore
