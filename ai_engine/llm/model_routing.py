"""
CRONOS AI - Intelligent Model Routing and Cost Optimization
Smart routing between LLM models based on complexity, cost, and quality requirements.

Features:
- Complexity-based model selection
- Cost-optimized fallback chains
- Quality-aware routing
- Budget management
- A/B testing support
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel


class ModelTier(Enum):
    """Model capability tiers."""
    PREMIUM = "premium"  # Most capable, highest cost (GPT-4o, Claude Opus)
    STANDARD = "standard"  # Good balance (GPT-4o-mini, Claude Sonnet)
    ECONOMY = "economy"  # Cost-effective (Haiku, local models)
    LOCAL = "local"  # Free, privacy-focused (Ollama)


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"  # Basic Q&A, formatting
    MODERATE = "moderate"  # Analysis, summarization
    COMPLEX = "complex"  # Multi-step reasoning, code generation
    CRITICAL = "critical"  # Highest quality required


@dataclass
class ModelSpec:
    """Specification for an LLM model."""
    model_id: str
    provider: str
    tier: ModelTier
    cost_per_1k_input: float  # USD per 1K input tokens
    cost_per_1k_output: float  # USD per 1K output tokens
    max_tokens: int
    context_window: int
    capabilities: List[str]  # e.g., ["code", "vision", "tools"]
    quality_score: float  # 0-1 quality rating
    latency_ms: float  # Average latency
    is_available: bool = True


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    selected_model: ModelSpec
    fallback_chain: List[ModelSpec]
    routing_reason: str
    estimated_cost: float
    complexity_assessment: TaskComplexity
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BudgetConfig:
    """Budget configuration for cost management."""
    daily_limit_usd: float = 100.0
    monthly_limit_usd: float = 2000.0
    per_request_limit_usd: float = 1.0
    alert_threshold_percent: float = 80.0
    enforce_limits: bool = True


@dataclass
class BudgetStatus:
    """Current budget status."""
    daily_spent: float = 0.0
    monthly_spent: float = 0.0
    daily_remaining: float = 0.0
    monthly_remaining: float = 0.0
    is_within_budget: bool = True
    alert_triggered: bool = False


# Default model specifications (2024-2025 pricing)
DEFAULT_MODELS = [
    ModelSpec(
        model_id="gpt-4o",
        provider="openai",
        tier=ModelTier.PREMIUM,
        cost_per_1k_input=2.50,
        cost_per_1k_output=10.00,
        max_tokens=16384,
        context_window=128000,
        capabilities=["code", "vision", "tools", "json"],
        quality_score=0.95,
        latency_ms=800,
    ),
    ModelSpec(
        model_id="gpt-4o-mini",
        provider="openai",
        tier=ModelTier.STANDARD,
        cost_per_1k_input=0.15,
        cost_per_1k_output=0.60,
        max_tokens=16384,
        context_window=128000,
        capabilities=["code", "vision", "tools", "json"],
        quality_score=0.88,
        latency_ms=400,
    ),
    ModelSpec(
        model_id="claude-sonnet-4-5-20250929",
        provider="anthropic",
        tier=ModelTier.PREMIUM,
        cost_per_1k_input=3.00,
        cost_per_1k_output=15.00,
        max_tokens=8192,
        context_window=200000,
        capabilities=["code", "tools", "long_context"],
        quality_score=0.94,
        latency_ms=900,
    ),
    ModelSpec(
        model_id="claude-3-5-haiku-20241022",
        provider="anthropic",
        tier=ModelTier.ECONOMY,
        cost_per_1k_input=0.25,
        cost_per_1k_output=1.25,
        max_tokens=8192,
        context_window=200000,
        capabilities=["code", "tools"],
        quality_score=0.82,
        latency_ms=300,
    ),
    ModelSpec(
        model_id="llama3.2",
        provider="ollama",
        tier=ModelTier.LOCAL,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        max_tokens=4096,
        context_window=8192,
        capabilities=["code"],
        quality_score=0.75,
        latency_ms=200,
    ),
    ModelSpec(
        model_id="qwen2.5",
        provider="ollama",
        tier=ModelTier.LOCAL,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        max_tokens=4096,
        context_window=32768,
        capabilities=["code", "long_context"],
        quality_score=0.78,
        latency_ms=250,
    ),
]


class ComplexityAssessor:
    """
    Assesses the complexity of a task/query for routing decisions.

    Uses heuristics and optional LLM-based assessment.
    """

    def __init__(self, llm_assess_func: Optional[Callable] = None):
        self.llm_assess = llm_assess_func
        self.logger = logging.getLogger(__name__)

        # Complexity indicators
        self.complex_indicators = [
            "step by step",
            "analyze",
            "compare",
            "evaluate",
            "multi-step",
            "complex",
            "comprehensive",
            "detailed analysis",
            "trade-offs",
            "architecture",
            "design",
            "implement",
            "optimize",
        ]

        self.simple_indicators = [
            "what is",
            "define",
            "list",
            "summarize",
            "translate",
            "format",
            "convert",
            "explain briefly",
        ]

        self.critical_indicators = [
            "security",
            "vulnerability",
            "critical",
            "production",
            "sensitive",
            "compliance",
            "audit",
            "legal",
        ]

    def assess(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        feature_domain: str = "",
    ) -> TaskComplexity:
        """
        Assess the complexity of a query.

        Args:
            query: The query/prompt to assess
            context: Optional context information
            feature_domain: Feature domain (affects complexity assessment)

        Returns:
            TaskComplexity level
        """
        query_lower = query.lower()
        context = context or {}

        # Check for critical indicators first
        for indicator in self.critical_indicators:
            if indicator in query_lower:
                return TaskComplexity.CRITICAL

        # Check feature domain
        critical_domains = ["security_orchestrator", "compliance_reporter"]
        if feature_domain in critical_domains:
            return TaskComplexity.COMPLEX

        # Count complexity indicators
        complex_count = sum(
            1 for ind in self.complex_indicators if ind in query_lower
        )
        simple_count = sum(
            1 for ind in self.simple_indicators if ind in query_lower
        )

        # Length-based heuristics
        query_length = len(query)
        context_size = sum(len(str(v)) for v in context.values())

        # Multi-part queries (containing "and", numbered lists)
        has_multiple_parts = (
            " and " in query_lower and query_lower.count(" and ") >= 2
        ) or any(f"{i}." in query or f"{i})" in query for i in range(1, 5))

        # Scoring
        complexity_score = 0

        if complex_count > 1:
            complexity_score += 2
        elif complex_count == 1:
            complexity_score += 1

        if simple_count > 0:
            complexity_score -= 1

        if query_length > 500:
            complexity_score += 1
        if context_size > 2000:
            complexity_score += 1

        if has_multiple_parts:
            complexity_score += 1

        # Code-related tasks are inherently more complex
        code_indicators = ["code", "implement", "function", "class", "debug"]
        if any(ind in query_lower for ind in code_indicators):
            complexity_score += 1

        # Map score to complexity
        if complexity_score <= 0:
            return TaskComplexity.SIMPLE
        elif complexity_score <= 2:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.COMPLEX

    async def assess_with_llm(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[TaskComplexity, str]:
        """
        Use LLM to assess complexity (more accurate but slower).

        Returns:
            Tuple of (complexity, reasoning)
        """
        if self.llm_assess is None:
            return self.assess(query, context), "heuristic assessment"

        prompt = f"""Assess the complexity of this task on a scale:
- SIMPLE: Basic Q&A, formatting, simple lookups
- MODERATE: Analysis, summarization, single-step reasoning
- COMPLEX: Multi-step reasoning, code generation, detailed analysis
- CRITICAL: Security-critical, compliance, production systems

Task: {query[:500]}

Respond with just the complexity level (SIMPLE, MODERATE, COMPLEX, or CRITICAL) and a brief reason.
Format: LEVEL: reason"""

        try:
            response = await self.llm_assess(prompt)
            response_upper = response.upper()

            if "CRITICAL" in response_upper:
                return TaskComplexity.CRITICAL, response
            elif "COMPLEX" in response_upper:
                return TaskComplexity.COMPLEX, response
            elif "MODERATE" in response_upper:
                return TaskComplexity.MODERATE, response
            else:
                return TaskComplexity.SIMPLE, response

        except Exception as e:
            self.logger.warning(f"LLM complexity assessment failed: {e}")
            return self.assess(query, context), "fallback heuristic"


class BudgetManager:
    """
    Manages LLM spending budgets.

    Tracks costs and enforces limits.
    """

    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()
        self.logger = logging.getLogger(__name__)

        # Spending trackers
        self._daily_spent: float = 0.0
        self._monthly_spent: float = 0.0
        self._daily_reset: datetime = datetime.now()
        self._monthly_reset: datetime = datetime.now()

        # Request history for detailed tracking
        self._request_history: List[Dict[str, Any]] = []

    def record_cost(
        self,
        cost: float,
        model_id: str,
        tokens_used: int,
    ) -> None:
        """Record a cost incurrence."""
        now = datetime.now()

        # Reset daily counter if new day
        if now.date() > self._daily_reset.date():
            self._daily_spent = 0.0
            self._daily_reset = now

        # Reset monthly counter if new month
        if now.month != self._monthly_reset.month or now.year != self._monthly_reset.year:
            self._monthly_spent = 0.0
            self._monthly_reset = now

        self._daily_spent += cost
        self._monthly_spent += cost

        self._request_history.append({
            "timestamp": now,
            "cost": cost,
            "model_id": model_id,
            "tokens_used": tokens_used,
        })

        # Keep only last 1000 requests
        if len(self._request_history) > 1000:
            self._request_history = self._request_history[-1000:]

    def check_budget(self, estimated_cost: float = 0.0) -> BudgetStatus:
        """Check current budget status."""
        daily_remaining = self.config.daily_limit_usd - self._daily_spent
        monthly_remaining = self.config.monthly_limit_usd - self._monthly_spent

        is_within_budget = (
            daily_remaining >= estimated_cost and
            monthly_remaining >= estimated_cost and
            estimated_cost <= self.config.per_request_limit_usd
        )

        alert_triggered = (
            (self._daily_spent / self.config.daily_limit_usd * 100) >= self.config.alert_threshold_percent or
            (self._monthly_spent / self.config.monthly_limit_usd * 100) >= self.config.alert_threshold_percent
        )

        return BudgetStatus(
            daily_spent=self._daily_spent,
            monthly_spent=self._monthly_spent,
            daily_remaining=max(0, daily_remaining),
            monthly_remaining=max(0, monthly_remaining),
            is_within_budget=is_within_budget if self.config.enforce_limits else True,
            alert_triggered=alert_triggered,
        )

    def get_cost_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get cost analytics for the specified period."""
        cutoff = datetime.now() - timedelta(days=days)

        recent_requests = [
            r for r in self._request_history
            if r["timestamp"] >= cutoff
        ]

        if not recent_requests:
            return {
                "total_cost": 0.0,
                "total_requests": 0,
                "avg_cost_per_request": 0.0,
                "by_model": {},
            }

        total_cost = sum(r["cost"] for r in recent_requests)
        by_model = {}

        for r in recent_requests:
            model = r["model_id"]
            if model not in by_model:
                by_model[model] = {"cost": 0.0, "requests": 0, "tokens": 0}
            by_model[model]["cost"] += r["cost"]
            by_model[model]["requests"] += 1
            by_model[model]["tokens"] += r["tokens_used"]

        return {
            "total_cost": total_cost,
            "total_requests": len(recent_requests),
            "avg_cost_per_request": total_cost / len(recent_requests),
            "by_model": by_model,
            "period_days": days,
        }


class IntelligentRouter:
    """
    Intelligent model router with cost optimization and fallback chains.

    Features:
    - Complexity-based routing
    - Cost-aware model selection
    - Capability matching
    - Fallback chain generation
    - A/B testing support
    """

    def __init__(
        self,
        models: Optional[List[ModelSpec]] = None,
        budget_config: Optional[BudgetConfig] = None,
        complexity_assessor: Optional[ComplexityAssessor] = None,
        prefer_local: bool = False,
        quality_threshold: float = 0.7,
    ):
        self.models = models or DEFAULT_MODELS
        self.budget_manager = BudgetManager(budget_config)
        self.complexity_assessor = complexity_assessor or ComplexityAssessor()
        self.prefer_local = prefer_local
        self.quality_threshold = quality_threshold
        self.logger = logging.getLogger(__name__)

        # Build model lookup
        self._model_by_id = {m.model_id: m for m in self.models}

        # Tier preferences by complexity
        self._complexity_to_tiers = {
            TaskComplexity.SIMPLE: [ModelTier.ECONOMY, ModelTier.LOCAL, ModelTier.STANDARD],
            TaskComplexity.MODERATE: [ModelTier.STANDARD, ModelTier.ECONOMY, ModelTier.PREMIUM],
            TaskComplexity.COMPLEX: [ModelTier.PREMIUM, ModelTier.STANDARD],
            TaskComplexity.CRITICAL: [ModelTier.PREMIUM],
        }

    def route(
        self,
        query: str,
        feature_domain: str = "",
        context: Optional[Dict[str, Any]] = None,
        required_capabilities: Optional[List[str]] = None,
        max_cost: Optional[float] = None,
        prefer_provider: Optional[str] = None,
        force_complexity: Optional[TaskComplexity] = None,
    ) -> RoutingDecision:
        """
        Route a request to the best model.

        Args:
            query: The query/prompt
            feature_domain: Feature domain for routing
            context: Optional context
            required_capabilities: Required model capabilities
            max_cost: Maximum cost per request
            prefer_provider: Preferred provider
            force_complexity: Override complexity assessment

        Returns:
            RoutingDecision with selected model and fallback chain
        """
        # Assess complexity
        if force_complexity:
            complexity = force_complexity
        else:
            complexity = self.complexity_assessor.assess(query, context, feature_domain)

        # Check budget
        budget_status = self.budget_manager.check_budget()

        # Get candidate models
        candidates = self._get_candidates(
            complexity=complexity,
            required_capabilities=required_capabilities or [],
            max_cost=max_cost,
            prefer_provider=prefer_provider,
            budget_status=budget_status,
        )

        if not candidates:
            # Fallback to any available model
            candidates = [m for m in self.models if m.is_available]

        if not candidates:
            raise ValueError("No available models for routing")

        # Select best model
        selected = candidates[0]

        # Build fallback chain
        fallback_chain = candidates[1:4]  # Next 3 best candidates

        # Estimate cost
        estimated_tokens = len(query.split()) * 1.5 + 500  # Rough estimate
        estimated_cost = (
            (estimated_tokens / 1000) * selected.cost_per_1k_input +
            (estimated_tokens / 1000) * selected.cost_per_1k_output * 2
        )

        return RoutingDecision(
            selected_model=selected,
            fallback_chain=fallback_chain,
            routing_reason=self._build_routing_reason(
                selected, complexity, budget_status
            ),
            estimated_cost=estimated_cost,
            complexity_assessment=complexity,
        )

    def _get_candidates(
        self,
        complexity: TaskComplexity,
        required_capabilities: List[str],
        max_cost: Optional[float],
        prefer_provider: Optional[str],
        budget_status: BudgetStatus,
    ) -> List[ModelSpec]:
        """Get candidate models sorted by preference."""
        preferred_tiers = self._complexity_to_tiers[complexity]

        candidates = []
        for model in self.models:
            if not model.is_available:
                continue

            # Check capabilities
            if required_capabilities:
                if not all(cap in model.capabilities for cap in required_capabilities):
                    continue

            # Check cost constraints
            if max_cost:
                estimated_cost = (model.cost_per_1k_input + model.cost_per_1k_output) / 2
                if estimated_cost > max_cost:
                    continue

            # Check budget
            if not budget_status.is_within_budget and model.tier != ModelTier.LOCAL:
                continue

            # Check quality threshold
            if model.quality_score < self.quality_threshold and complexity != TaskComplexity.SIMPLE:
                continue

            candidates.append(model)

        # Sort candidates
        def sort_key(m: ModelSpec) -> Tuple[int, float, float, float]:
            tier_priority = (
                preferred_tiers.index(m.tier)
                if m.tier in preferred_tiers
                else len(preferred_tiers)
            )
            provider_priority = 0 if prefer_provider and m.provider == prefer_provider else 1
            cost = m.cost_per_1k_input + m.cost_per_1k_output
            quality = -m.quality_score  # Negative for descending order

            # Prefer local if configured
            if self.prefer_local and m.tier == ModelTier.LOCAL:
                tier_priority = 0

            return (tier_priority, provider_priority, cost, quality)

        candidates.sort(key=sort_key)
        return candidates

    def _build_routing_reason(
        self,
        model: ModelSpec,
        complexity: TaskComplexity,
        budget_status: BudgetStatus,
    ) -> str:
        """Build human-readable routing reason."""
        reasons = [
            f"Selected {model.model_id} ({model.provider})",
            f"for {complexity.value} complexity task",
        ]

        if model.tier == ModelTier.LOCAL:
            reasons.append("using local model for privacy/cost")
        elif model.tier == ModelTier.ECONOMY:
            reasons.append("optimized for cost")
        elif model.tier == ModelTier.PREMIUM:
            reasons.append("using premium tier for quality")

        if budget_status.alert_triggered:
            reasons.append("(budget alert active)")

        return " ".join(reasons)

    def update_model_availability(self, model_id: str, is_available: bool) -> None:
        """Update model availability status."""
        if model_id in self._model_by_id:
            self._model_by_id[model_id].is_available = is_available

    def get_model_spec(self, model_id: str) -> Optional[ModelSpec]:
        """Get model specification by ID."""
        return self._model_by_id.get(model_id)

    def add_model(self, model: ModelSpec) -> None:
        """Add a new model to the router."""
        self.models.append(model)
        self._model_by_id[model.model_id] = model


class RoutedLLMService:
    """
    Wrapper that adds intelligent routing to any LLM service.

    Usage:
        routed_service = RoutedLLMService(llm_service, router_config)
        response = await routed_service.process_request(request)
    """

    def __init__(
        self,
        llm_service,
        router: Optional[IntelligentRouter] = None,
        enable_routing: bool = True,
        track_costs: bool = True,
    ):
        self.llm_service = llm_service
        self.router = router or IntelligentRouter()
        self.enable_routing = enable_routing
        self.track_costs = track_costs
        self.logger = logging.getLogger(__name__)

    async def process_request(self, request) -> Any:
        """
        Process LLM request with intelligent routing.

        Args:
            request: LLM request (LLMRequest or similar)

        Returns:
            LLM response from selected model
        """
        if not self.enable_routing:
            return await self.llm_service.process_request(request)

        # Get routing decision
        query = getattr(request, "prompt", str(request))
        feature_domain = getattr(request, "feature_domain", "")
        context = getattr(request, "context", None)

        decision = self.router.route(
            query=query,
            feature_domain=feature_domain,
            context=context,
        )

        self.logger.debug(
            f"Routing decision: {decision.selected_model.model_id} "
            f"(complexity: {decision.complexity_assessment.value})"
        )

        # Set model override on request
        if hasattr(request, "model_override"):
            request.model_override = decision.selected_model.model_id

        # Try selected model and fallbacks
        last_error = None
        models_to_try = [decision.selected_model] + decision.fallback_chain

        for model in models_to_try:
            try:
                if hasattr(request, "model_override"):
                    request.model_override = model.model_id

                response = await self.llm_service.process_request(request)

                # Track cost
                if self.track_costs:
                    tokens = getattr(response, "tokens_used", 0)
                    cost = (
                        (tokens / 1000) * model.cost_per_1k_input +
                        (tokens / 1000) * model.cost_per_1k_output
                    )
                    self.router.budget_manager.record_cost(cost, model.model_id, tokens)

                return response

            except Exception as e:
                last_error = e
                self.logger.warning(f"Model {model.model_id} failed: {e}")
                self.router.update_model_availability(model.model_id, False)

        raise last_error or Exception("All models failed")

    def get_budget_status(self) -> BudgetStatus:
        """Get current budget status."""
        return self.router.budget_manager.check_budget()

    def get_cost_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get cost analytics."""
        return self.router.budget_manager.get_cost_analytics(days)


# Global router instance
_global_router: Optional[IntelligentRouter] = None


def get_router() -> IntelligentRouter:
    """Get or create global router instance."""
    global _global_router
    if _global_router is None:
        _global_router = IntelligentRouter()
    return _global_router


def configure_router(**config) -> IntelligentRouter:
    """Configure and return global router."""
    global _global_router
    _global_router = IntelligentRouter(**config)
    return _global_router
