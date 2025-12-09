"""
CRONOS AI - LLM Observability and Tracing
Comprehensive observability for LLM operations with Langfuse integration.

This module provides:
- LLM request/response tracing
- Token cost tracking
- Latency monitoring
- Error tracking and debugging
- Prompt versioning support
"""

import asyncio
import logging
import time
import json
import uuid
import hashlib
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager
from functools import wraps

# Langfuse imports with fallback
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    observe = None
    langfuse_context = None

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# =============================================================================
# Metrics
# =============================================================================

LLM_TRACE_COUNTER = Counter(
    "cronos_llm_traces_total",
    "Total LLM traces",
    ["provider", "operation", "status"]
)
LLM_COST_COUNTER = Counter(
    "cronos_llm_cost_dollars_total",
    "Total LLM cost in dollars",
    ["provider", "model"]
)
LLM_LATENCY_HISTOGRAM = Histogram(
    "cronos_llm_latency_seconds",
    "LLM operation latency",
    ["provider", "model", "operation"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)
LLM_TOKEN_HISTOGRAM = Histogram(
    "cronos_llm_tokens",
    "Token usage per request",
    ["provider", "model", "token_type"],
    buckets=[10, 50, 100, 500, 1000, 2000, 4000, 8000, 16000, 32000]
)
LLM_ERROR_COUNTER = Counter(
    "cronos_llm_errors_total",
    "Total LLM errors",
    ["provider", "error_type"]
)
ACTIVE_TRACES_GAUGE = Gauge(
    "cronos_llm_active_traces",
    "Currently active LLM traces"
)

# =============================================================================
# Token Cost Configuration (2024-2025 Pricing)
# =============================================================================

class TokenCostConfig:
    """Token pricing configuration for various models."""

    # OpenAI Pricing (per 1M tokens) - Updated December 2024
    OPENAI_COSTS = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    # Anthropic Pricing (per 1M tokens) - Updated December 2024
    ANTHROPIC_COSTS = {
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    }

    # Ollama/Local models - No cost
    LOCAL_COSTS = {
        "llama3.2": {"input": 0.0, "output": 0.0},
        "qwen2.5": {"input": 0.0, "output": 0.0},
        "mistral": {"input": 0.0, "output": 0.0},
    }

    @classmethod
    def get_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request."""
        # Check OpenAI models
        if model in cls.OPENAI_COSTS:
            costs = cls.OPENAI_COSTS[model]
            input_cost = (input_tokens / 1_000_000) * costs["input"]
            output_cost = (output_tokens / 1_000_000) * costs["output"]
            return input_cost + output_cost

        # Check Anthropic models
        if model in cls.ANTHROPIC_COSTS:
            costs = cls.ANTHROPIC_COSTS[model]
            input_cost = (input_tokens / 1_000_000) * costs["input"]
            output_cost = (output_tokens / 1_000_000) * costs["output"]
            return input_cost + output_cost

        # Check local models
        if model in cls.LOCAL_COSTS:
            return 0.0

        # Unknown model - return 0
        return 0.0

    @classmethod
    def get_provider(cls, model: str) -> str:
        """Determine provider from model name."""
        if model in cls.OPENAI_COSTS or model.startswith("gpt"):
            return "openai"
        if model in cls.ANTHROPIC_COSTS or model.startswith("claude"):
            return "anthropic"
        return "local"


# =============================================================================
# Data Classes
# =============================================================================

class TraceStatus(Enum):
    """Status of a trace."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class TokenUsage:
    """Token usage breakdown."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost(self) -> float:
        """Calculate cost (needs model info)."""
        return 0.0  # Set externally


@dataclass
class TraceMetadata:
    """Metadata for a trace."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMTrace:
    """Complete trace of an LLM operation."""
    trace_id: str
    parent_trace_id: Optional[str] = None
    operation: str = "generation"
    provider: str = "unknown"
    model: str = "unknown"

    # Request info
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    request_params: Dict[str, Any] = field(default_factory=dict)

    # Response info
    response: Optional[str] = None
    response_metadata: Dict[str, Any] = field(default_factory=dict)

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    latency_ms: float = 0.0

    # Cost
    cost_usd: float = 0.0

    # Status
    status: TraceStatus = TraceStatus.RUNNING
    error: Optional[str] = None

    # Metadata
    metadata: TraceMetadata = field(default_factory=TraceMetadata)

    def complete(
        self,
        response: str,
        input_tokens: int,
        output_tokens: int,
        response_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark trace as completed."""
        self.end_time = datetime.utcnow()
        self.response = response
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.latency_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = TraceStatus.COMPLETED
        self.response_metadata = response_metadata or {}

        # Calculate cost
        self.cost_usd = TokenCostConfig.get_cost(
            self.model, input_tokens, output_tokens
        )

    def fail(self, error: str) -> None:
        """Mark trace as failed."""
        self.end_time = datetime.utcnow()
        self.latency_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = TraceStatus.FAILED
        self.error = error


@dataclass
class PromptVersion:
    """Versioned prompt for tracking."""
    prompt_id: str
    version: int
    content: str
    system_prompt: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            content_str = f"{self.system_prompt or ''}{self.content}"
            self.hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]


# =============================================================================
# Observability Manager
# =============================================================================

class LLMObservabilityManager:
    """
    Manages LLM observability with Langfuse integration.

    Features:
    - Trace tracking and storage
    - Token cost calculation
    - Latency monitoring
    - Prompt versioning
    - Error tracking
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Langfuse client
        self.langfuse_client: Optional[Any] = None
        self._init_langfuse()

        # Local trace storage (for when Langfuse unavailable)
        self.traces: Dict[str, LLMTrace] = {}
        self.trace_history: List[LLMTrace] = []
        self.max_history_size = self.config.get("max_history_size", 10000)

        # Prompt versioning
        self.prompt_versions: Dict[str, List[PromptVersion]] = {}

        # Cost tracking
        self.cost_tracker = CostTracker()

        # Active trace tracking
        self._active_traces: Dict[str, LLMTrace] = {}

    def _init_langfuse(self) -> None:
        """Initialize Langfuse client."""
        if not LANGFUSE_AVAILABLE:
            self.logger.info("Langfuse not available. Using local trace storage.")
            return

        public_key = self.config.get("langfuse_public_key")
        secret_key = self.config.get("langfuse_secret_key")
        host = self.config.get("langfuse_host", "https://cloud.langfuse.com")

        if public_key and secret_key:
            try:
                self.langfuse_client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                self.logger.info("Langfuse client initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Langfuse: {e}")
                self.langfuse_client = None
        else:
            self.logger.info("Langfuse credentials not configured. Using local storage.")

    # =========================================================================
    # Tracing API
    # =========================================================================

    def start_trace(
        self,
        operation: str,
        provider: str,
        model: str,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        request_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[TraceMetadata] = None,
        parent_trace_id: Optional[str] = None
    ) -> LLMTrace:
        """Start a new trace."""
        trace_id = str(uuid.uuid4())

        trace = LLMTrace(
            trace_id=trace_id,
            parent_trace_id=parent_trace_id,
            operation=operation,
            provider=provider,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            request_params=request_params or {},
            metadata=metadata or TraceMetadata()
        )

        # Store trace
        self.traces[trace_id] = trace
        self._active_traces[trace_id] = trace
        ACTIVE_TRACES_GAUGE.inc()

        # Send to Langfuse if available
        if self.langfuse_client:
            try:
                self.langfuse_client.trace(
                    id=trace_id,
                    name=operation,
                    metadata={
                        "provider": provider,
                        "model": model,
                        **(metadata.custom_attributes if metadata else {})
                    },
                    tags=metadata.tags if metadata else [],
                    user_id=metadata.user_id if metadata else None,
                    session_id=metadata.session_id if metadata else None
                )
            except Exception as e:
                self.logger.warning(f"Failed to send trace to Langfuse: {e}")

        LLM_TRACE_COUNTER.labels(
            provider=provider,
            operation=operation,
            status="started"
        ).inc()

        return trace

    def end_trace(
        self,
        trace_id: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        response_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """End a trace successfully."""
        trace = self.traces.get(trace_id)
        if not trace:
            self.logger.warning(f"Trace {trace_id} not found")
            return

        # Complete the trace
        trace.complete(response, input_tokens, output_tokens, response_metadata)

        # Remove from active traces
        self._active_traces.pop(trace_id, None)
        ACTIVE_TRACES_GAUGE.dec()

        # Add to history
        self._add_to_history(trace)

        # Update metrics
        LLM_TRACE_COUNTER.labels(
            provider=trace.provider,
            operation=trace.operation,
            status="completed"
        ).inc()

        LLM_LATENCY_HISTOGRAM.labels(
            provider=trace.provider,
            model=trace.model,
            operation=trace.operation
        ).observe(trace.latency_ms / 1000)  # Convert to seconds

        LLM_TOKEN_HISTOGRAM.labels(
            provider=trace.provider,
            model=trace.model,
            token_type="input"
        ).observe(input_tokens)

        LLM_TOKEN_HISTOGRAM.labels(
            provider=trace.provider,
            model=trace.model,
            token_type="output"
        ).observe(output_tokens)

        LLM_COST_COUNTER.labels(
            provider=trace.provider,
            model=trace.model
        ).inc(trace.cost_usd)

        # Update cost tracker
        self.cost_tracker.record_cost(
            provider=trace.provider,
            model=trace.model,
            cost=trace.cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        # Send to Langfuse if available
        if self.langfuse_client:
            try:
                self.langfuse_client.generation(
                    trace_id=trace_id,
                    name=trace.operation,
                    model=trace.model,
                    input=trace.prompt or json.dumps(trace.messages),
                    output=response,
                    usage={
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": input_tokens + output_tokens
                    },
                    metadata=response_metadata
                )
            except Exception as e:
                self.logger.warning(f"Failed to send generation to Langfuse: {e}")

    def fail_trace(self, trace_id: str, error: str) -> None:
        """Mark a trace as failed."""
        trace = self.traces.get(trace_id)
        if not trace:
            self.logger.warning(f"Trace {trace_id} not found")
            return

        trace.fail(error)

        # Remove from active traces
        self._active_traces.pop(trace_id, None)
        ACTIVE_TRACES_GAUGE.dec()

        # Add to history
        self._add_to_history(trace)

        # Update metrics
        LLM_TRACE_COUNTER.labels(
            provider=trace.provider,
            operation=trace.operation,
            status="failed"
        ).inc()

        LLM_ERROR_COUNTER.labels(
            provider=trace.provider,
            error_type=type(error).__name__ if isinstance(error, Exception) else "unknown"
        ).inc()

    def _add_to_history(self, trace: LLMTrace) -> None:
        """Add trace to history with size limit."""
        self.trace_history.append(trace)

        # Trim history if needed
        if len(self.trace_history) > self.max_history_size:
            self.trace_history = self.trace_history[-self.max_history_size:]

    # =========================================================================
    # Context Manager and Decorator
    # =========================================================================

    @asynccontextmanager
    async def trace_llm_call(
        self,
        operation: str,
        provider: str,
        model: str,
        **kwargs
    ):
        """Context manager for tracing LLM calls."""
        trace = self.start_trace(
            operation=operation,
            provider=provider,
            model=model,
            **kwargs
        )

        try:
            yield trace
        except Exception as e:
            self.fail_trace(trace.trace_id, str(e))
            raise

    def trace_decorator(
        self,
        operation: str,
        provider: str,
        model: str
    ):
        """Decorator for tracing LLM calls."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                trace = self.start_trace(
                    operation=operation,
                    provider=provider,
                    model=model
                )

                try:
                    result = await func(*args, **kwargs)

                    # Try to extract token usage from result
                    if hasattr(result, 'tokens_used'):
                        self.end_trace(
                            trace.trace_id,
                            response=str(result),
                            input_tokens=getattr(result, 'input_tokens', 0),
                            output_tokens=getattr(result, 'output_tokens', result.tokens_used)
                        )
                    else:
                        self.end_trace(
                            trace.trace_id,
                            response=str(result),
                            input_tokens=0,
                            output_tokens=0
                        )

                    return result
                except Exception as e:
                    self.fail_trace(trace.trace_id, str(e))
                    raise

            return wrapper
        return decorator

    # =========================================================================
    # Prompt Versioning
    # =========================================================================

    def register_prompt(
        self,
        prompt_id: str,
        content: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PromptVersion:
        """Register a new prompt version."""
        if prompt_id not in self.prompt_versions:
            self.prompt_versions[prompt_id] = []

        # Calculate new version number
        version = len(self.prompt_versions[prompt_id]) + 1

        prompt_version = PromptVersion(
            prompt_id=prompt_id,
            version=version,
            content=content,
            system_prompt=system_prompt,
            metadata=metadata or {}
        )

        self.prompt_versions[prompt_id].append(prompt_version)

        self.logger.info(f"Registered prompt {prompt_id} version {version}")

        return prompt_version

    def get_prompt_version(
        self,
        prompt_id: str,
        version: Optional[int] = None
    ) -> Optional[PromptVersion]:
        """Get a specific prompt version (default: latest)."""
        versions = self.prompt_versions.get(prompt_id)
        if not versions:
            return None

        if version is None:
            return versions[-1]  # Latest

        for v in versions:
            if v.version == version:
                return v

        return None

    def list_prompt_versions(self, prompt_id: str) -> List[PromptVersion]:
        """List all versions of a prompt."""
        return self.prompt_versions.get(prompt_id, [])

    # =========================================================================
    # Analytics
    # =========================================================================

    def get_trace(self, trace_id: str) -> Optional[LLMTrace]:
        """Get a specific trace."""
        return self.traces.get(trace_id)

    def get_recent_traces(
        self,
        limit: int = 100,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status: Optional[TraceStatus] = None
    ) -> List[LLMTrace]:
        """Get recent traces with optional filtering."""
        traces = self.trace_history.copy()

        if provider:
            traces = [t for t in traces if t.provider == provider]
        if model:
            traces = [t for t in traces if t.model == model]
        if status:
            traces = [t for t in traces if t.status == status]

        return traces[-limit:]

    def get_cost_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get cost summary for a time period."""
        return self.cost_tracker.get_summary(start_time, end_time)

    def get_latency_stats(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get latency statistics."""
        traces = self.trace_history

        if provider:
            traces = [t for t in traces if t.provider == provider]
        if model:
            traces = [t for t in traces if t.model == model]

        if not traces:
            return {"count": 0, "avg_ms": 0, "p50_ms": 0, "p90_ms": 0, "p99_ms": 0}

        latencies = sorted([t.latency_ms for t in traces if t.latency_ms > 0])

        return {
            "count": len(latencies),
            "avg_ms": sum(latencies) / len(latencies),
            "p50_ms": latencies[len(latencies) // 2] if latencies else 0,
            "p90_ms": latencies[int(len(latencies) * 0.9)] if latencies else 0,
            "p99_ms": latencies[int(len(latencies) * 0.99)] if latencies else 0,
            "min_ms": min(latencies) if latencies else 0,
            "max_ms": max(latencies) if latencies else 0
        }

    def flush(self) -> None:
        """Flush any pending traces to Langfuse."""
        if self.langfuse_client:
            try:
                self.langfuse_client.flush()
            except Exception as e:
                self.logger.warning(f"Failed to flush Langfuse: {e}")

    def shutdown(self) -> None:
        """Shutdown observability manager."""
        self.flush()
        if self.langfuse_client:
            try:
                self.langfuse_client.shutdown()
            except Exception as e:
                self.logger.warning(f"Failed to shutdown Langfuse: {e}")


# =============================================================================
# Cost Tracker
# =============================================================================

class CostTracker:
    """Tracks LLM costs with time-based aggregation."""

    def __init__(self):
        self.costs: List[Dict[str, Any]] = []

    def record_cost(
        self,
        provider: str,
        model: str,
        cost: float,
        input_tokens: int,
        output_tokens: int
    ) -> None:
        """Record a cost entry."""
        self.costs.append({
            "timestamp": datetime.utcnow(),
            "provider": provider,
            "model": model,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        })

    def get_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get cost summary for a time period."""
        costs = self.costs

        if start_time:
            costs = [c for c in costs if c["timestamp"] >= start_time]
        if end_time:
            costs = [c for c in costs if c["timestamp"] <= end_time]

        if not costs:
            return {
                "total_cost": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_provider": {},
                "by_model": {}
            }

        # Aggregate by provider
        by_provider: Dict[str, Dict[str, Any]] = {}
        for c in costs:
            provider = c["provider"]
            if provider not in by_provider:
                by_provider[provider] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "requests": 0
                }
            by_provider[provider]["cost"] += c["cost"]
            by_provider[provider]["input_tokens"] += c["input_tokens"]
            by_provider[provider]["output_tokens"] += c["output_tokens"]
            by_provider[provider]["requests"] += 1

        # Aggregate by model
        by_model: Dict[str, Dict[str, Any]] = {}
        for c in costs:
            model = c["model"]
            if model not in by_model:
                by_model[model] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "requests": 0
                }
            by_model[model]["cost"] += c["cost"]
            by_model[model]["input_tokens"] += c["input_tokens"]
            by_model[model]["output_tokens"] += c["output_tokens"]
            by_model[model]["requests"] += 1

        return {
            "total_cost": sum(c["cost"] for c in costs),
            "total_input_tokens": sum(c["input_tokens"] for c in costs),
            "total_output_tokens": sum(c["output_tokens"] for c in costs),
            "total_requests": len(costs),
            "by_provider": by_provider,
            "by_model": by_model,
            "period": {
                "start": min(c["timestamp"] for c in costs).isoformat() if costs else None,
                "end": max(c["timestamp"] for c in costs).isoformat() if costs else None
            }
        }

    def get_daily_costs(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily cost breakdown."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent_costs = [c for c in self.costs if c["timestamp"] >= cutoff]

        # Group by day
        daily: Dict[str, Dict[str, Any]] = {}
        for c in recent_costs:
            day = c["timestamp"].strftime("%Y-%m-%d")
            if day not in daily:
                daily[day] = {"date": day, "cost": 0.0, "requests": 0}
            daily[day]["cost"] += c["cost"]
            daily[day]["requests"] += 1

        return sorted(daily.values(), key=lambda x: x["date"])


# =============================================================================
# Global Instance
# =============================================================================

_observability_manager: Optional[LLMObservabilityManager] = None


def get_observability_manager() -> LLMObservabilityManager:
    """Get or create the global observability manager."""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = LLMObservabilityManager()
    return _observability_manager


def configure_observability(config: Dict[str, Any]) -> LLMObservabilityManager:
    """Configure and return the global observability manager."""
    global _observability_manager
    _observability_manager = LLMObservabilityManager(config)
    return _observability_manager
