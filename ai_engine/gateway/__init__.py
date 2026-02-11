"""
QBITEL Gateway - Centralized AI/LLM Management Layer

Enterprise-grade AI Gateway providing:
- Centralized LLM routing with intelligent provider selection
- Semantic caching for 30-50% cost reduction
- Rate limiting and quota management
- Cost tracking and budget enforcement
- Request/response logging and analytics
- Automatic failover and circuit breaking
- Prompt versioning and management

Usage:
    from ai_engine.gateway import AIGateway, get_ai_gateway

    # Initialize gateway
    gateway = await get_ai_gateway()

    # Make requests through gateway
    response = await gateway.complete(
        prompt="Analyze this protocol",
        domain="protocol_copilot",
        cache_enabled=True
    )
"""

from .ai_gateway import (
    AIGateway,
    GatewayConfig,
    GatewayRequest,
    GatewayResponse,
    get_ai_gateway,
    initialize_ai_gateway,
    shutdown_ai_gateway,
)

from .semantic_cache import (
    SemanticCache,
    CacheConfig,
    CacheEntry,
    CacheStats,
)

from .cost_tracker import (
    CostTracker,
    UsageRecord,
    CostReport,
    BudgetAlert,
)

from .rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
)

from .prompt_registry import (
    PromptRegistry,
    PromptTemplate,
    PromptVersion,
)

__all__ = [
    # Gateway
    "AIGateway",
    "GatewayConfig",
    "GatewayRequest",
    "GatewayResponse",
    "get_ai_gateway",
    "initialize_ai_gateway",
    "shutdown_ai_gateway",
    # Cache
    "SemanticCache",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    # Cost
    "CostTracker",
    "UsageRecord",
    "CostReport",
    "BudgetAlert",
    # Rate Limiting
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitExceeded",
    # Prompts
    "PromptRegistry",
    "PromptTemplate",
    "PromptVersion",
]
