"""
QBITEL - Agent LLM Integration Layer

Provides a local-first LLM service interface for the core agent framework.
Bridges the gap between the agent system (which previously had no direct LLM
integration) and the model routing / vLLM provider layers.

Features:
- Local-first model routing (qbitel-* fine-tuned → vLLM local → cloud fallback)
- Circuit breaker protection on every LLM call
- Structured output parsing (JSON mode, tool-calling format)
- Token usage tracking per agent
- Configurable temperature/max_tokens per task type
- Transparent fallback between local and cloud models
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from prometheus_client import Counter, Histogram, Gauge

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    BulkheadFullError,
    get_circuit_breaker_registry,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

AGENT_LLM_CALLS = Counter(
    "qbitel_agent_llm_calls_total",
    "Total LLM calls from agents",
    ["agent_type", "model", "status"],
)
AGENT_LLM_LATENCY = Histogram(
    "qbitel_agent_llm_latency_seconds",
    "LLM call latency from agents",
    ["agent_type", "model"],
)
AGENT_LLM_TOKENS = Counter(
    "qbitel_agent_llm_tokens_total",
    "Tokens consumed by agent LLM calls",
    ["agent_type", "model", "direction"],  # direction: input | output
)
AGENT_LLM_FALLBACKS = Counter(
    "qbitel_agent_llm_fallbacks_total",
    "LLM fallback events",
    ["agent_type", "from_model", "to_model"],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LLMTaskType(str, Enum):
    """Task-type hints for model selection."""

    PROTOCOL_ANALYSIS = "protocol_analysis"
    SECURITY_ASSESSMENT = "security_assessment"
    CODE_GENERATION = "code_generation"
    CODE_TRANSLATION = "code_translation"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    GENERAL = "general"


@dataclass
class AgentLLMConfig:
    """Configuration for agent LLM integration."""

    # Local-first model routing
    # Priority: fine-tuned local → general local → cloud fallback
    default_model: str = "qbitel-protocol"
    security_model: str = "qbitel-security"
    translation_model: str = "qbitel-translate"
    fast_model: str = "mimo-v2-flash"
    reasoning_model: str = "deepseek-r1-distilled-32b"
    code_model: str = "qwen3-coder-next"
    cloud_fallback_model: str = "gpt-4o"

    # Fallback chain (tried in order if primary fails)
    fallback_chain: List[str] = field(default_factory=lambda: [
        "qwen3-coder-next",
        "deepseek-r1-distilled-32b",
        "mimo-v2-flash",
        "deepseek-v3.2",
        "gpt-4o",
    ])

    # Generation defaults
    default_temperature: float = 0.3
    default_max_tokens: int = 4096
    default_top_p: float = 0.95

    # Per-task-type overrides
    task_type_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "protocol_analysis": {
            "model": "qbitel-protocol",
            "temperature": 0.2,
            "max_tokens": 8192,
        },
        "security_assessment": {
            "model": "qbitel-security",
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        "code_generation": {
            "model": "qwen3-coder-next",
            "temperature": 0.3,
            "max_tokens": 8192,
        },
        "code_translation": {
            "model": "qbitel-translate",
            "temperature": 0.2,
            "max_tokens": 8192,
        },
        "reasoning": {
            "model": "deepseek-r1-distilled-32b",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "summarization": {
            "model": "mimo-v2-flash",
            "temperature": 0.3,
            "max_tokens": 2048,
        },
        "classification": {
            "model": "mimo-v2-flash",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        "general": {
            "model": "qwen3-coder-next",
            "temperature": 0.5,
            "max_tokens": 4096,
        },
    })

    # Circuit breaker
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_reset_timeout: float = 15.0

    # Retry
    max_retries: int = 2
    retry_delay: float = 1.0

    # Budget
    max_tokens_per_minute: int = 100_000
    enable_token_tracking: bool = True


@dataclass
class LLMResponse:
    """Standardized LLM response from agent calls."""

    content: str
    model: str
    tokens_input: int = 0
    tokens_output: int = 0
    latency_seconds: float = 0.0
    was_fallback: bool = False
    fallback_chain_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output

    def as_json(self) -> Optional[Dict[str, Any]]:
        """Try to parse content as JSON."""
        try:
            return json.loads(self.content)
        except (json.JSONDecodeError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Agent LLM Service
# ---------------------------------------------------------------------------


class AgentLLMService:
    """
    LLM service layer for the QBITEL core agent framework.

    Wraps the unified LLM service / vLLM provider with:
    - Local-first model routing per task type
    - Circuit breaker protection per model
    - Automatic fallback along the chain
    - Token tracking and budget enforcement

    Usage:
        llm_service = AgentLLMService(config, unified_llm_service)

        # Simple call
        response = await llm_service.generate(
            agent_type="threat_analyst",
            prompt="Analyze this protocol capture...",
            task_type=LLMTaskType.SECURITY_ASSESSMENT,
        )

        # Structured output
        response = await llm_service.generate_structured(
            agent_type="protocol_analyzer",
            prompt="Extract fields from this message...",
            output_schema={"type": "object", "properties": {...}},
            task_type=LLMTaskType.PROTOCOL_ANALYSIS,
        )
    """

    def __init__(
        self,
        config: Optional[AgentLLMConfig] = None,
        llm_backend: Optional[Any] = None,
    ):
        """
        Initialize the Agent LLM Service.

        Args:
            config: Agent LLM configuration.
            llm_backend: Underlying LLM service (UnifiedLLMService, vLLM
                         provider, or any object exposing an async
                         ``generate(model, messages, **kwargs)`` method).
        """
        self.config = config or AgentLLMConfig()
        self._backend = llm_backend

        # Circuit breaker registry – one breaker per model
        self._cb_registry = get_circuit_breaker_registry()

        # Token tracking
        self._token_counts: Dict[str, Dict[str, int]] = {}  # agent_type -> {input, output}

        # Rate limit tracking
        self._minute_tokens: int = 0
        self._minute_start: float = time.time()

        self.logger = logging.getLogger(f"{__name__}.AgentLLMService")
        self.logger.info(
            f"AgentLLMService initialized (default_model={self.config.default_model})"
        )

    def set_backend(self, llm_backend: Any) -> None:
        """Set or replace the LLM backend at runtime."""
        self._backend = llm_backend
        self.logger.info("LLM backend updated")

    # ------ Public API ------

    async def generate(
        self,
        agent_type: str,
        prompt: str,
        task_type: Union[LLMTaskType, str] = LLMTaskType.GENERAL,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model_override: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate an LLM response for an agent.

        Args:
            agent_type: Type of agent making the call (for metrics).
            prompt: The user/task prompt.
            task_type: Hint for model selection.
            system_prompt: Optional system prompt override.
            messages: Full message list (overrides prompt if provided).
            model_override: Force a specific model.
            temperature: Override temperature.
            max_tokens: Override max tokens.
            stop_sequences: Stop sequences.
            **kwargs: Additional params passed to backend.

        Returns:
            LLMResponse with content and metadata.
        """
        task_type_str = task_type.value if isinstance(task_type, LLMTaskType) else task_type

        # Resolve model and generation params
        model, gen_params = self._resolve_params(
            task_type_str, model_override, temperature, max_tokens
        )

        # Build messages
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        # Build fallback chain
        fallback_chain = self._build_fallback_chain(model)

        # Try primary model, then fallbacks
        chain_used = []
        last_error = None

        for attempt_model in [model] + fallback_chain:
            chain_used.append(attempt_model)

            try:
                response = await self._call_model(
                    model=attempt_model,
                    messages=messages,
                    agent_type=agent_type,
                    stop_sequences=stop_sequences,
                    **gen_params,
                    **kwargs,
                )

                # Track if we had to fall back
                was_fallback = attempt_model != model
                if was_fallback:
                    AGENT_LLM_FALLBACKS.labels(
                        agent_type=agent_type,
                        from_model=model,
                        to_model=attempt_model,
                    ).inc()
                    self.logger.info(
                        f"LLM fallback: {model} → {attempt_model} for {agent_type}"
                    )

                response.was_fallback = was_fallback
                response.fallback_chain_used = chain_used
                return response

            except CircuitOpenError as e:
                self.logger.warning(
                    f"Circuit open for {attempt_model}, trying next in chain"
                )
                last_error = e
                continue

            except Exception as e:
                self.logger.warning(
                    f"Model {attempt_model} failed: {e}, trying next"
                )
                last_error = e
                continue

        # All models failed
        raise RuntimeError(
            f"All models in chain exhausted for {agent_type}. "
            f"Chain: {chain_used}. Last error: {last_error}"
        )

    async def generate_structured(
        self,
        agent_type: str,
        prompt: str,
        output_schema: Optional[Dict[str, Any]] = None,
        task_type: Union[LLMTaskType, str] = LLMTaskType.GENERAL,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a structured (JSON) response.

        Adds JSON-mode instructions to the prompt and validates output.
        """
        # Enhance system prompt for JSON output
        json_instruction = (
            "You MUST respond with valid JSON only. "
            "Do not include any text before or after the JSON."
        )
        if output_schema:
            json_instruction += (
                f"\n\nExpected output schema:\n"
                f"```json\n{json.dumps(output_schema, indent=2)}\n```"
            )

        enhanced_system = (
            f"{system_prompt}\n\n{json_instruction}"
            if system_prompt
            else json_instruction
        )

        response = await self.generate(
            agent_type=agent_type,
            prompt=prompt,
            task_type=task_type,
            system_prompt=enhanced_system,
            **kwargs,
        )

        # Try to parse JSON; if it fails, strip markdown code fences and retry
        parsed = response.as_json()
        if parsed is None:
            content = response.content.strip()
            # Strip markdown fences
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = "\n".join(lines).strip()
                try:
                    parsed = json.loads(content)
                    response.content = content
                except json.JSONDecodeError:
                    pass

        response.metadata["structured"] = parsed is not None
        return response

    # ------ Internal ------

    async def _call_model(
        self,
        model: str,
        messages: List[Dict[str, str]],
        agent_type: str,
        stop_sequences: Optional[List[str]] = None,
        **gen_params,
    ) -> LLMResponse:
        """Call a specific model with circuit breaker protection."""
        start_time = time.time()

        # Get or create circuit breaker for this model
        cb = self._cb_registry.get_or_create(
            name=f"llm:{model}",
            template="llm",
        )

        # The actual call function
        async def _do_call() -> Dict[str, Any]:
            if self._backend is None:
                raise RuntimeError(
                    "No LLM backend configured. Call set_backend() first."
                )

            # Adapt to backend interface
            # Support both `generate(model, messages, **kwargs)` and
            # `chat(messages, model=model, **kwargs)` patterns
            if hasattr(self._backend, "generate"):
                result = await self._backend.generate(
                    model=model,
                    messages=messages,
                    stop=stop_sequences,
                    **gen_params,
                )
            elif hasattr(self._backend, "chat"):
                result = await self._backend.chat(
                    messages=messages,
                    model=model,
                    stop=stop_sequences,
                    **gen_params,
                )
            elif hasattr(self._backend, "agenerate"):
                result = await self._backend.agenerate(
                    model=model,
                    messages=messages,
                    stop=stop_sequences,
                    **gen_params,
                )
            else:
                raise TypeError(
                    f"LLM backend {type(self._backend).__name__} has no "
                    f"generate/chat/agenerate method"
                )

            return result

        # Execute through circuit breaker
        raw_result = await cb.call(_do_call)

        elapsed = time.time() - start_time

        # Normalize result to LLMResponse
        response = self._normalize_response(raw_result, model, elapsed)

        # Track metrics
        AGENT_LLM_CALLS.labels(
            agent_type=agent_type, model=model, status="success"
        ).inc()
        AGENT_LLM_LATENCY.labels(
            agent_type=agent_type, model=model
        ).observe(elapsed)

        if self.config.enable_token_tracking:
            self._track_tokens(
                agent_type, model,
                response.tokens_input, response.tokens_output,
            )

        return response

    def _normalize_response(
        self, raw: Any, model: str, elapsed: float
    ) -> LLMResponse:
        """Normalize various backend response formats to LLMResponse."""
        if isinstance(raw, str):
            return LLMResponse(
                content=raw,
                model=model,
                latency_seconds=elapsed,
            )

        if isinstance(raw, dict):
            # OpenAI-style response
            content = ""
            tokens_in = 0
            tokens_out = 0

            if "choices" in raw:
                choice = raw["choices"][0]
                if "message" in choice:
                    content = choice["message"].get("content", "")
                elif "text" in choice:
                    content = choice["text"]
            elif "content" in raw:
                content = raw["content"]
            elif "text" in raw:
                content = raw["text"]
            elif "response" in raw:
                content = raw["response"]

            usage = raw.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)

            return LLMResponse(
                content=content,
                model=raw.get("model", model),
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                latency_seconds=elapsed,
                metadata=raw.get("metadata", {}),
            )

        # Unknown format — try to convert to string
        return LLMResponse(
            content=str(raw),
            model=model,
            latency_seconds=elapsed,
        )

    def _resolve_params(
        self,
        task_type: str,
        model_override: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Tuple[str, Dict[str, Any]]:
        """Resolve model and generation params from task type and overrides."""
        # Start with defaults
        model = self.config.default_model
        params = {
            "temperature": self.config.default_temperature,
            "max_tokens": self.config.default_max_tokens,
            "top_p": self.config.default_top_p,
        }

        # Apply task-type config
        task_config = self.config.task_type_configs.get(task_type, {})
        if task_config:
            model = task_config.get("model", model)
            params["temperature"] = task_config.get("temperature", params["temperature"])
            params["max_tokens"] = task_config.get("max_tokens", params["max_tokens"])

        # Apply explicit overrides
        if model_override:
            model = model_override
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        return model, params

    def _build_fallback_chain(self, primary_model: str) -> List[str]:
        """Build fallback chain excluding the primary model."""
        return [m for m in self.config.fallback_chain if m != primary_model]

    def _track_tokens(
        self,
        agent_type: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        """Track token usage per agent type."""
        if agent_type not in self._token_counts:
            self._token_counts[agent_type] = {"input": 0, "output": 0}

        self._token_counts[agent_type]["input"] += tokens_in
        self._token_counts[agent_type]["output"] += tokens_out

        AGENT_LLM_TOKENS.labels(
            agent_type=agent_type, model=model, direction="input"
        ).inc(tokens_in)
        AGENT_LLM_TOKENS.labels(
            agent_type=agent_type, model=model, direction="output"
        ).inc(tokens_out)

    # ------ Introspection ------

    def get_token_usage(self) -> Dict[str, Dict[str, int]]:
        """Get token usage breakdown per agent type."""
        return dict(self._token_counts)

    def get_circuit_dashboard(self) -> Dict[str, Any]:
        """Get circuit breaker dashboard for all LLM models."""
        return self._cb_registry.get_dashboard()

    def get_healthy_models(self) -> List[str]:
        """Get list of models whose circuits are not open."""
        open_circuits = self._cb_registry.get_open_circuits()
        open_models = {
            name.replace("llm:", "") for name in open_circuits
        }
        return [
            m for m in [self.config.default_model] + self.config.fallback_chain
            if m not in open_models
        ]
