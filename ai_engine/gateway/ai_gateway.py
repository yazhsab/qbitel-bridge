"""
QBITEL Gateway - Main Gateway Implementation

Centralized AI/LLM management layer that coordinates:
- Intelligent routing to LLM providers
- Semantic caching for cost reduction
- Rate limiting and quota management
- Cost tracking and budget enforcement
- Prompt versioning and A/B testing
- Request/response logging and analytics

This gateway sits between application code and the LLM service,
providing a unified interface with enterprise features.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, AsyncIterator, Type
from datetime import datetime, timezone
from enum import Enum
from contextlib import asynccontextmanager

from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge

from ..llm.unified_llm_service import (
    UnifiedLLMService,
    LLMRequest,
    LLMResponse,
    LLMProvider,
    ResponseFormat,
    ToolDefinition,
    get_llm_service,
)
from ..core.config import Config, get_config
from ..core.circuit_breakers import circuit_breakers, CircuitOpenError

from .semantic_cache import SemanticCache, CacheConfig, CacheEntry
from .cost_tracker import CostTracker, BudgetConfig, UsageRecord, BudgetAlert
from .rate_limiter import RateLimiter, RateLimitConfig, RateLimitExceeded
from .prompt_registry import PromptRegistry, PromptTemplate, PromptVersion

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================

GATEWAY_REQUESTS = Counter(
    "qbitel_gateway_requests_total",
    "Total gateway requests",
    ["domain", "status", "cached"],
)
GATEWAY_LATENCY = Histogram(
    "qbitel_gateway_request_duration_seconds",
    "Gateway request duration",
    ["domain", "cached"],
    buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60],
)
GATEWAY_ACTIVE_REQUESTS = Gauge(
    "qbitel_gateway_active_requests",
    "Currently active requests",
    ["domain"],
)
GATEWAY_PROVIDER_USAGE = Counter(
    "qbitel_gateway_provider_usage_total",
    "Provider usage count",
    ["provider", "domain"],
)


# =============================================================================
# Data Classes
# =============================================================================

class RequestPriority(str, Enum):
    """Request priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GatewayConfig:
    """Configuration for the AI Gateway."""

    # Cache configuration
    cache_enabled: bool = True
    cache_config: Optional[CacheConfig] = None

    # Cost tracking
    cost_tracking_enabled: bool = True
    budget_config: Optional[BudgetConfig] = None

    # Rate limiting
    rate_limiting_enabled: bool = True
    rate_limit_config: Optional[RateLimitConfig] = None

    # Prompt registry
    prompt_registry_enabled: bool = True

    # Fallback settings
    fallback_enabled: bool = True
    fallback_providers: List[str] = field(default_factory=lambda: ["ollama_local"])

    # Request settings
    default_timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Redis URL for distributed features
    redis_url: Optional[str] = None

    # Logging
    log_requests: bool = True
    log_responses: bool = False  # Can be verbose


@dataclass
class GatewayRequest:
    """Request to the AI Gateway."""

    # Core request
    prompt: str
    domain: str

    # Optional parameters
    system_prompt: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 2000
    temperature: float = 0.3

    # Gateway-specific
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: RequestPriority = RequestPriority.NORMAL

    # Caching
    cache_enabled: bool = True
    cache_ttl: Optional[int] = None

    # Provider selection
    preferred_provider: Optional[str] = None
    model_override: Optional[str] = None

    # Structured output
    response_format: ResponseFormat = ResponseFormat.TEXT
    json_schema: Optional[Dict[str, Any]] = None
    response_model: Optional[Type[BaseModel]] = None

    # Tools
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[str] = None

    # Prompt template
    prompt_template: Optional[str] = None
    template_variables: Dict[str, Any] = field(default_factory=dict)

    # Streaming
    stream: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GatewayResponse:
    """Response from the AI Gateway."""

    # Core response
    content: str
    request_id: str
    domain: str

    # Provider info
    provider: str
    model: str

    # Token usage
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Performance
    latency_ms: float
    cache_hit: bool = False

    # Cost
    estimated_cost: float = 0.0

    # Prompt version (if using registry)
    prompt_version: Optional[str] = None

    # Structured output
    parsed_response: Optional[Dict[str, Any]] = None
    validated_model: Optional[BaseModel] = None

    # Tool calls
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.validated_model:
            result["validated_model"] = self.validated_model.model_dump()
        return result


# =============================================================================
# AI Gateway
# =============================================================================

class AIGateway:
    """
    Centralized AI Gateway for QBITEL.

    Provides enterprise-grade LLM management with caching, rate limiting,
    cost tracking, and intelligent routing.
    """

    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self.logger = logging.getLogger(__name__)

        # Core LLM service
        self._llm_service: Optional[UnifiedLLMService] = None

        # Gateway components
        self._cache: Optional[SemanticCache] = None
        self._cost_tracker: Optional[CostTracker] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._prompt_registry: Optional[PromptRegistry] = None

        # State
        self._initialized = False
        self._active_requests: Dict[str, GatewayRequest] = {}

        # Request queue for priority handling
        self._request_queues: Dict[RequestPriority, asyncio.Queue] = {
            RequestPriority.CRITICAL: asyncio.Queue(),
            RequestPriority.HIGH: asyncio.Queue(),
            RequestPriority.NORMAL: asyncio.Queue(),
            RequestPriority.LOW: asyncio.Queue(),
        }

    async def initialize(self):
        """Initialize the gateway and all components."""
        if self._initialized:
            return

        self.logger.info("Initializing AI Gateway...")

        # Get config
        app_config = get_config()

        # Initialize LLM service
        self._llm_service = get_llm_service()
        if hasattr(self._llm_service, "initialize"):
            await self._llm_service.initialize()

        # Initialize semantic cache
        if self.config.cache_enabled:
            cache_config = self.config.cache_config or CacheConfig(
                redis_url=self.config.redis_url
            )
            self._cache = SemanticCache(cache_config)
            await self._cache.initialize()
            self.logger.info("Semantic cache initialized")

        # Initialize cost tracker
        if self.config.cost_tracking_enabled:
            budget_config = self.config.budget_config or BudgetConfig()
            self._cost_tracker = CostTracker(budget_config, self.config.redis_url)
            await self._cost_tracker.initialize()

            # Register alert callback
            self._cost_tracker.register_alert_callback(self._handle_budget_alert)
            self.logger.info("Cost tracker initialized")

        # Initialize rate limiter
        if self.config.rate_limiting_enabled:
            rate_config = self.config.rate_limit_config or RateLimitConfig(
                redis_url=self.config.redis_url
            )
            self._rate_limiter = RateLimiter(rate_config)
            await self._rate_limiter.initialize()
            self.logger.info("Rate limiter initialized")

        # Initialize prompt registry
        if self.config.prompt_registry_enabled:
            self._prompt_registry = PromptRegistry(self.config.redis_url)
            await self._prompt_registry.initialize()
            self.logger.info("Prompt registry initialized")

        self._initialized = True
        self.logger.info("AI Gateway initialized successfully")

    async def shutdown(self):
        """Shutdown the gateway gracefully."""
        self.logger.info("Shutting down AI Gateway...")

        # Wait for active requests to complete (with timeout)
        if self._active_requests:
            self.logger.info(f"Waiting for {len(self._active_requests)} active requests...")
            await asyncio.sleep(5)  # Give requests time to complete

        # Shutdown components
        if self._cache:
            await self._cache.shutdown()
        if self._cost_tracker:
            await self._cost_tracker.shutdown()
        if self._rate_limiter:
            await self._rate_limiter.shutdown()
        if self._prompt_registry:
            await self._prompt_registry.shutdown()

        if self._llm_service and hasattr(self._llm_service, "shutdown"):
            await self._llm_service.shutdown()

        self._initialized = False
        self.logger.info("AI Gateway shutdown complete")

    async def complete(self, request: GatewayRequest) -> GatewayResponse:
        """
        Process a completion request through the gateway.

        This is the main entry point for all LLM requests.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        cache_hit = False

        # Track active request
        self._active_requests[request.request_id] = request
        GATEWAY_ACTIVE_REQUESTS.labels(domain=request.domain).inc()

        try:
            # Log request
            if self.config.log_requests:
                self.logger.info(
                    f"Gateway request: id={request.request_id}, "
                    f"domain={request.domain}, priority={request.priority.value}"
                )

            # Rate limiting
            if self._rate_limiter:
                await self._rate_limiter.acquire(
                    domain=request.domain,
                    user_id=request.user_id,
                    wait=request.priority != RequestPriority.LOW,
                )

            # Budget check
            fallback_model = None
            if self._cost_tracker:
                # Estimate cost
                estimated_cost = self._estimate_cost(request)
                allowed, fallback_model = await self._cost_tracker.check_budget(
                    domain=request.domain,
                    estimated_cost=estimated_cost,
                )
                if not allowed:
                    raise Exception("Budget exceeded and blocking enabled")

            # Resolve prompt template
            prompt = request.prompt
            system_prompt = request.system_prompt
            prompt_version = None

            if self._prompt_registry and request.prompt_template:
                system_prompt, prompt, version = await self._prompt_registry.get_prompt(
                    prompt_name=request.prompt_template,
                    variables={**request.template_variables, "query": request.prompt},
                    request_id=request.request_id,
                )
                prompt_version = version.version_id

            # Check cache
            if request.cache_enabled and self._cache and not request.stream:
                cache_entry = await self._cache.get(
                    prompt=prompt,
                    domain=request.domain,
                    system_prompt=system_prompt,
                )

                if cache_entry:
                    cache_hit = True
                    response = self._create_cached_response(request, cache_entry, prompt_version)

                    # Record usage (cached)
                    if self._cost_tracker:
                        await self._cost_tracker.record_usage(
                            request_id=request.request_id,
                            domain=request.domain,
                            model=cache_entry.model,
                            input_tokens=0,
                            output_tokens=0,
                            latency_ms=(time.time() - start_time) * 1000,
                            user_id=request.user_id,
                            cached=True,
                        )

                    GATEWAY_REQUESTS.labels(
                        domain=request.domain,
                        status="success",
                        cached="true",
                    ).inc()

                    return response

            # Make LLM request
            model = fallback_model or request.model_override
            llm_response = await self._make_llm_request(request, prompt, system_prompt, model)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Create response
            response = self._create_response(request, llm_response, latency_ms, prompt_version)

            # Store in cache
            if request.cache_enabled and self._cache and not request.stream:
                await self._cache.put(
                    prompt=prompt,
                    response=llm_response.content,
                    domain=request.domain,
                    model=llm_response.provider,
                    tokens_used=llm_response.tokens_used,
                    system_prompt=system_prompt,
                    ttl=request.cache_ttl,
                )

            # Record usage
            if self._cost_tracker:
                await self._cost_tracker.record_usage(
                    request_id=request.request_id,
                    domain=request.domain,
                    model=llm_response.provider,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=latency_ms,
                    user_id=request.user_id,
                    cached=False,
                )

            # Record prompt performance
            if self._prompt_registry and prompt_version:
                await self._prompt_registry.record_performance(
                    prompt_name=request.prompt_template,
                    version_id=prompt_version,
                    tokens_used=response.total_tokens,
                    latency_ms=latency_ms,
                    success=True,
                )

            # Track metrics
            GATEWAY_REQUESTS.labels(
                domain=request.domain,
                status="success",
                cached="false",
            ).inc()
            GATEWAY_LATENCY.labels(
                domain=request.domain,
                cached="false",
            ).observe(latency_ms / 1000)
            GATEWAY_PROVIDER_USAGE.labels(
                provider=response.provider,
                domain=request.domain,
            ).inc()

            return response

        except RateLimitExceeded as e:
            GATEWAY_REQUESTS.labels(
                domain=request.domain,
                status="rate_limited",
                cached="false",
            ).inc()
            raise

        except CircuitOpenError as e:
            GATEWAY_REQUESTS.labels(
                domain=request.domain,
                status="circuit_open",
                cached="false",
            ).inc()
            raise

        except Exception as e:
            GATEWAY_REQUESTS.labels(
                domain=request.domain,
                status="error",
                cached="false",
            ).inc()
            self.logger.error(f"Gateway request failed: {e}")
            raise

        finally:
            # Remove from active requests
            self._active_requests.pop(request.request_id, None)
            GATEWAY_ACTIVE_REQUESTS.labels(domain=request.domain).dec()

    async def complete_stream(
        self,
        request: GatewayRequest,
    ) -> AsyncIterator[str]:
        """
        Process a streaming completion request.

        Yields content chunks as they arrive.
        """
        if not self._initialized:
            await self.initialize()

        request.stream = True
        request.cache_enabled = False  # Disable caching for streams

        # Track active request
        self._active_requests[request.request_id] = request
        GATEWAY_ACTIVE_REQUESTS.labels(domain=request.domain).inc()

        try:
            # Rate limiting
            if self._rate_limiter:
                await self._rate_limiter.acquire(
                    domain=request.domain,
                    user_id=request.user_id,
                )

            # Resolve prompt
            prompt = request.prompt
            system_prompt = request.system_prompt

            if self._prompt_registry and request.prompt_template:
                system_prompt, prompt, _ = await self._prompt_registry.get_prompt(
                    prompt_name=request.prompt_template,
                    variables={**request.template_variables, "query": request.prompt},
                    request_id=request.request_id,
                )

            # Create LLM request
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain=request.domain,
                context=request.context,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system_prompt=system_prompt,
                stream=True,
            )

            # Stream response
            async for chunk in self._llm_service.stream_request(llm_request):
                yield chunk

            GATEWAY_REQUESTS.labels(
                domain=request.domain,
                status="success",
                cached="false",
            ).inc()

        except Exception as e:
            GATEWAY_REQUESTS.labels(
                domain=request.domain,
                status="error",
                cached="false",
            ).inc()
            raise

        finally:
            self._active_requests.pop(request.request_id, None)
            GATEWAY_ACTIVE_REQUESTS.labels(domain=request.domain).dec()

    async def _make_llm_request(
        self,
        request: GatewayRequest,
        prompt: str,
        system_prompt: Optional[str],
        model_override: Optional[str] = None,
    ) -> LLMResponse:
        """Make the actual LLM request with retry logic."""
        llm_request = LLMRequest(
            prompt=prompt,
            feature_domain=request.domain,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=system_prompt,
            stream=request.stream,
            tools=request.tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format,
            json_schema=request.json_schema,
            response_model=request.response_model,
            model_override=model_override,
        )

        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = await asyncio.wait_for(
                    self._llm_service.process_request(llm_request),
                    timeout=self.config.default_timeout,
                )
                return response

            except asyncio.TimeoutError:
                last_error = Exception(f"Request timeout after {self.config.default_timeout}s")
                self.logger.warning(
                    f"LLM request timeout (attempt {attempt + 1}/{self.config.max_retries})"
                )

            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )

            # Wait before retry
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        raise last_error or Exception("LLM request failed")

    def _create_response(
        self,
        request: GatewayRequest,
        llm_response: LLMResponse,
        latency_ms: float,
        prompt_version: Optional[str],
    ) -> GatewayResponse:
        """Create a gateway response from LLM response."""
        # Estimate token split (rough approximation if not provided)
        total_tokens = llm_response.tokens_used
        output_tokens = len(llm_response.content.split()) * 1.3  # Rough estimate
        input_tokens = max(0, total_tokens - int(output_tokens))

        # Estimate cost
        estimated_cost = 0.0
        if self._cost_tracker:
            pricing = self._cost_tracker.get_pricing(llm_response.provider)
            estimated_cost = pricing.calculate_cost(input_tokens, int(output_tokens))

        return GatewayResponse(
            content=llm_response.content,
            request_id=request.request_id,
            domain=request.domain,
            provider=llm_response.provider,
            model=llm_response.provider,
            input_tokens=input_tokens,
            output_tokens=int(output_tokens),
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cache_hit=False,
            estimated_cost=estimated_cost,
            prompt_version=prompt_version,
            parsed_response=llm_response.parsed_response,
            validated_model=llm_response.validated_model,
            tool_calls=[asdict(tc) for tc in llm_response.tool_calls]
            if llm_response.tool_calls else None,
            metadata=llm_response.metadata or {},
        )

    def _create_cached_response(
        self,
        request: GatewayRequest,
        cache_entry: CacheEntry,
        prompt_version: Optional[str],
    ) -> GatewayResponse:
        """Create a gateway response from cached entry."""
        return GatewayResponse(
            content=cache_entry.response,
            request_id=request.request_id,
            domain=request.domain,
            provider=cache_entry.model,
            model=cache_entry.model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0.0,
            cache_hit=True,
            estimated_cost=0.0,
            prompt_version=prompt_version,
            metadata={"cache_key": cache_entry.cache_key},
        )

    def _estimate_cost(self, request: GatewayRequest) -> float:
        """Estimate the cost of a request."""
        # Rough estimation based on prompt length
        input_tokens = len(request.prompt.split()) * 1.3
        output_tokens = request.max_tokens * 0.5  # Assume 50% of max

        if self._cost_tracker:
            # Use default model pricing
            pricing = self._cost_tracker.get_pricing("gpt-4o")
            return pricing.calculate_cost(int(input_tokens), int(output_tokens))

        return 0.01  # Default estimate

    def _handle_budget_alert(self, alert: BudgetAlert):
        """Handle budget alerts."""
        self.logger.warning(f"Budget alert: {alert.message}")
        # Could integrate with notification systems here

    # ==========================================================================
    # Convenience Methods
    # ==========================================================================

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            stats = self._cache.get_stats()
            return asdict(stats)
        return {}

    async def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        if self._cost_tracker:
            return await self._cost_tracker.get_usage_summary()
        return {}

    async def get_rate_limit_status(self, domain: str) -> Dict[str, Any]:
        """Get rate limit status for a domain."""
        if self._rate_limiter:
            return await self._rate_limiter.get_status(domain)
        return {}

    async def invalidate_cache(
        self,
        domain: Optional[str] = None,
    ):
        """Invalidate cache entries."""
        if self._cache:
            await self._cache.invalidate(domain=domain)

    @property
    def is_initialized(self) -> bool:
        """Check if gateway is initialized."""
        return self._initialized

    @property
    def active_request_count(self) -> int:
        """Get count of active requests."""
        return len(self._active_requests)


# =============================================================================
# Global Gateway Instance
# =============================================================================

_gateway_instance: Optional[AIGateway] = None
_gateway_lock = asyncio.Lock()


async def get_ai_gateway() -> AIGateway:
    """Get the global AI Gateway instance."""
    global _gateway_instance

    if _gateway_instance is None:
        async with _gateway_lock:
            if _gateway_instance is None:
                _gateway_instance = AIGateway()
                await _gateway_instance.initialize()

    return _gateway_instance


async def initialize_ai_gateway(config: Optional[GatewayConfig] = None) -> AIGateway:
    """Initialize the global AI Gateway with custom config."""
    global _gateway_instance

    async with _gateway_lock:
        if _gateway_instance is not None:
            await _gateway_instance.shutdown()

        _gateway_instance = AIGateway(config)
        await _gateway_instance.initialize()

    return _gateway_instance


async def shutdown_ai_gateway():
    """Shutdown the global AI Gateway."""
    global _gateway_instance

    async with _gateway_lock:
        if _gateway_instance is not None:
            await _gateway_instance.shutdown()
            _gateway_instance = None


@asynccontextmanager
async def ai_gateway_context(config: Optional[GatewayConfig] = None):
    """Context manager for AI Gateway lifecycle."""
    gateway = await initialize_ai_gateway(config)
    try:
        yield gateway
    finally:
        await shutdown_ai_gateway()
