"""
QBITEL - vLLM Provider Integration

High-performance LLM serving with vLLM for production deployments.

vLLM provides:
- PagedAttention for 24x better throughput
- Continuous batching for high efficiency
- Speculative decoding for faster generation
- Tensor parallelism for large models
- OpenAI-compatible API

This provider integrates vLLM-served models into the QBITEL LLM service.
"""

import asyncio
import logging
import time
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, AsyncIterator, Union
from enum import Enum

import aiohttp
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================

VLLM_REQUESTS = Counter(
    "qbitel_vllm_requests_total",
    "Total vLLM requests",
    ["model", "status"],
)
VLLM_LATENCY = Histogram(
    "qbitel_vllm_request_duration_seconds",
    "vLLM request latency",
    ["model"],
    buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120],
)
VLLM_TOKENS_PER_SECOND = Histogram(
    "qbitel_vllm_tokens_per_second",
    "vLLM tokens generated per second",
    ["model"],
    buckets=[10, 25, 50, 100, 200, 500, 1000],
)
VLLM_QUEUE_SIZE = Gauge(
    "qbitel_vllm_queue_size",
    "vLLM request queue size",
    ["model"],
)
VLLM_ACTIVE_REQUESTS = Gauge(
    "qbitel_vllm_active_requests",
    "Currently active vLLM requests",
    ["model"],
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VLLMModelConfig:
    """Configuration for a vLLM-served model."""

    model_name: str
    endpoint: str  # e.g., "http://vllm-server:8000"

    # Model settings
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50

    # Performance settings
    max_concurrent_requests: int = 100
    request_timeout: float = 120.0
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Features
    supports_streaming: bool = True
    supports_function_calling: bool = False  # Model dependent

    # Cost (per 1K tokens, for local deployments)
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0


# Pre-configured models for vLLM
VLLM_MODELS = {
    "llama-3.2-70b": VLLMModelConfig(
        model_name="meta-llama/Llama-3.2-70B-Instruct",
        endpoint="http://vllm-llama:8000",
        max_tokens=8192,
        supports_function_calling=True,
    ),
    "llama-3.2-8b": VLLMModelConfig(
        model_name="meta-llama/Llama-3.2-8B-Instruct",
        endpoint="http://vllm-llama-8b:8000",
        max_tokens=8192,
    ),
    "mistral-7b": VLLMModelConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        endpoint="http://vllm-mistral:8000",
        max_tokens=32768,
    ),
    "codellama-34b": VLLMModelConfig(
        model_name="codellama/CodeLlama-34b-Instruct-hf",
        endpoint="http://vllm-codellama:8000",
        max_tokens=16384,
    ),
    "qwen-72b": VLLMModelConfig(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        endpoint="http://vllm-qwen:8000",
        max_tokens=32768,
        supports_function_calling=True,
    ),
    "deepseek-coder-33b": VLLMModelConfig(
        model_name="deepseek-ai/deepseek-coder-33b-instruct",
        endpoint="http://vllm-deepseek:8000",
        max_tokens=16384,
    ),
}


@dataclass
class VLLMProviderConfig:
    """Configuration for vLLM provider."""

    # Default endpoints (can be overridden per model)
    default_endpoint: str = "http://localhost:8000"

    # Model configurations
    models: Dict[str, VLLMModelConfig] = field(default_factory=lambda: VLLM_MODELS.copy())

    # Default model selection
    default_model: str = "llama-3.2-8b"
    code_model: str = "codellama-34b"
    fast_model: str = "llama-3.2-8b"
    powerful_model: str = "llama-3.2-70b"

    # Connection settings
    connection_pool_size: int = 100
    keepalive_timeout: float = 30.0

    # Health check
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0


# =============================================================================
# Request/Response Models
# =============================================================================

@dataclass
class VLLMRequest:
    """Request to vLLM server."""

    prompt: str
    model: str
    max_tokens: int = 2000
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    stream: bool = False
    stop: Optional[List[str]] = None

    # Chat format
    messages: Optional[List[Dict[str, str]]] = None
    system_prompt: Optional[str] = None

    # Advanced
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    best_of: int = 1
    use_beam_search: bool = False
    skip_special_tokens: bool = True

    # Request metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible format for vLLM."""
        if self.messages:
            # Chat completion format
            messages = self.messages.copy()
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            return {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": self.stream,
                "stop": self.stop,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
            }
        else:
            # Completion format
            prompt = self.prompt
            if self.system_prompt:
                prompt = f"{self.system_prompt}\n\n{prompt}"

            return {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": self.stream,
                "stop": self.stop,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
            }


@dataclass
class VLLMResponse:
    """Response from vLLM server."""

    content: str
    model: str
    request_id: str

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Performance
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Metadata
    finish_reason: str = "stop"
    logprobs: Optional[List[float]] = None


# =============================================================================
# vLLM Provider
# =============================================================================

class VLLMProvider:
    """
    High-performance LLM provider using vLLM.

    Provides OpenAI-compatible API access to vLLM-served models with:
    - Automatic retries and failover
    - Connection pooling
    - Streaming support
    - Health monitoring
    - Comprehensive metrics
    """

    def __init__(self, config: Optional[VLLMProviderConfig] = None):
        self.config = config or VLLMProviderConfig()
        self.logger = logging.getLogger(__name__)

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        # Model health tracking
        self._model_health: Dict[str, bool] = {}
        self._health_check_task: Optional[asyncio.Task] = None

        # Request tracking
        self._active_requests: Dict[str, VLLMRequest] = {}

        # Semaphores for concurrency control
        self._semaphores: Dict[str, asyncio.Semaphore] = {}

        self._initialized = False

    async def initialize(self):
        """Initialize the vLLM provider."""
        if self._initialized:
            return

        # Create HTTP session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.config.connection_pool_size,
            keepalive_timeout=self.config.keepalive_timeout,
        )
        self._session = aiohttp.ClientSession(connector=connector)

        # Initialize semaphores for each model
        for model_id, model_config in self.config.models.items():
            self._semaphores[model_id] = asyncio.Semaphore(
                model_config.max_concurrent_requests
            )
            self._model_health[model_id] = True  # Assume healthy initially

        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        self._initialized = True
        self.logger.info(f"vLLM provider initialized with {len(self.config.models)} models")

    async def shutdown(self):
        """Shutdown the vLLM provider."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()

        self._initialized = False
        self.logger.info("vLLM provider shutdown")

    async def complete(self, request: VLLMRequest) -> VLLMResponse:
        """
        Generate a completion using vLLM.

        Args:
            request: The completion request

        Returns:
            VLLMResponse with generated content
        """
        if not self._initialized:
            await self.initialize()

        model_id = self._resolve_model(request.model)
        model_config = self.config.models.get(model_id)

        if not model_config:
            raise ValueError(f"Unknown model: {request.model}")

        # Check model health
        if not self._model_health.get(model_id, True):
            # Try to find a fallback
            fallback = self._get_fallback_model(model_id)
            if fallback:
                self.logger.warning(
                    f"Model {model_id} unhealthy, using fallback {fallback}"
                )
                model_id = fallback
                model_config = self.config.models[fallback]
            else:
                raise Exception(f"Model {model_id} is unavailable")

        # Acquire semaphore for concurrency control
        semaphore = self._semaphores.get(model_id)
        if semaphore:
            await semaphore.acquire()

        VLLM_ACTIVE_REQUESTS.labels(model=model_id).inc()
        self._active_requests[request.request_id] = request

        try:
            start_time = time.time()

            # Make request with retries
            response = await self._make_request_with_retry(
                request, model_config
            )

            latency_ms = (time.time() - start_time) * 1000

            # Calculate tokens per second
            tokens_per_second = 0.0
            if response.completion_tokens > 0 and latency_ms > 0:
                tokens_per_second = response.completion_tokens / (latency_ms / 1000)

            response.latency_ms = latency_ms
            response.tokens_per_second = tokens_per_second

            # Record metrics
            VLLM_REQUESTS.labels(model=model_id, status="success").inc()
            VLLM_LATENCY.labels(model=model_id).observe(latency_ms / 1000)
            VLLM_TOKENS_PER_SECOND.labels(model=model_id).observe(tokens_per_second)

            return response

        except Exception as e:
            VLLM_REQUESTS.labels(model=model_id, status="error").inc()
            raise

        finally:
            self._active_requests.pop(request.request_id, None)
            VLLM_ACTIVE_REQUESTS.labels(model=model_id).dec()
            if semaphore:
                semaphore.release()

    async def stream(self, request: VLLMRequest) -> AsyncIterator[str]:
        """
        Stream a completion using vLLM.

        Args:
            request: The completion request

        Yields:
            Content chunks as they're generated
        """
        if not self._initialized:
            await self.initialize()

        model_id = self._resolve_model(request.model)
        model_config = self.config.models.get(model_id)

        if not model_config:
            raise ValueError(f"Unknown model: {request.model}")

        if not model_config.supports_streaming:
            # Fall back to non-streaming
            response = await self.complete(request)
            yield response.content
            return

        request.stream = True
        endpoint = f"{model_config.endpoint}/v1/chat/completions"
        payload = request.to_openai_format()

        try:
            async with self._session.post(
                endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=model_config.request_timeout),
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            raise

    async def _make_request_with_retry(
        self,
        request: VLLMRequest,
        model_config: VLLMModelConfig,
    ) -> VLLMResponse:
        """Make a request with retry logic."""
        last_error = None

        for attempt in range(model_config.retry_attempts):
            try:
                return await self._make_request(request, model_config)

            except aiohttp.ClientError as e:
                last_error = e
                self.logger.warning(
                    f"vLLM request failed (attempt {attempt + 1}): {e}"
                )

                if attempt < model_config.retry_attempts - 1:
                    await asyncio.sleep(
                        model_config.retry_delay * (attempt + 1)
                    )

            except asyncio.TimeoutError:
                last_error = Exception("Request timeout")
                self.logger.warning(
                    f"vLLM request timeout (attempt {attempt + 1})"
                )

                if attempt < model_config.retry_attempts - 1:
                    await asyncio.sleep(model_config.retry_delay)

        raise last_error or Exception("vLLM request failed")

    async def _make_request(
        self,
        request: VLLMRequest,
        model_config: VLLMModelConfig,
    ) -> VLLMResponse:
        """Make a single request to vLLM server."""
        # Determine endpoint based on request format
        if request.messages:
            endpoint = f"{model_config.endpoint}/v1/chat/completions"
        else:
            endpoint = f"{model_config.endpoint}/v1/completions"

        payload = request.to_openai_format()
        payload["model"] = model_config.model_name  # Use full model name

        async with self._session.post(
            endpoint,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=model_config.request_timeout),
        ) as response:
            response.raise_for_status()
            data = await response.json()

            # Parse response
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]

                if "message" in choice:
                    content = choice["message"].get("content", "")
                else:
                    content = choice.get("text", "")

                usage = data.get("usage", {})

                return VLLMResponse(
                    content=content,
                    model=data.get("model", request.model),
                    request_id=data.get("id", request.request_id),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    finish_reason=choice.get("finish_reason", "stop"),
                )

            raise Exception("Invalid vLLM response format")

    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to model ID."""
        # Check for exact match
        if model in self.config.models:
            return model

        # Check aliases
        aliases = {
            "default": self.config.default_model,
            "fast": self.config.fast_model,
            "code": self.config.code_model,
            "powerful": self.config.powerful_model,
            "llama": "llama-3.2-8b",
            "mistral": "mistral-7b",
            "codellama": "codellama-34b",
            "qwen": "qwen-72b",
            "deepseek": "deepseek-coder-33b",
        }

        return aliases.get(model, self.config.default_model)

    def _get_fallback_model(self, model_id: str) -> Optional[str]:
        """Get a fallback model when primary is unhealthy."""
        # Define fallback chains
        fallback_chains = {
            "llama-3.2-70b": ["llama-3.2-8b", "mistral-7b"],
            "llama-3.2-8b": ["mistral-7b"],
            "codellama-34b": ["deepseek-coder-33b", "llama-3.2-8b"],
            "qwen-72b": ["llama-3.2-70b", "llama-3.2-8b"],
        }

        fallbacks = fallback_chains.get(model_id, [])
        for fallback in fallbacks:
            if self._model_health.get(fallback, False):
                return fallback

        # Last resort: any healthy model
        for mid, healthy in self._model_health.items():
            if healthy and mid != model_id:
                return mid

        return None

    async def _health_check_loop(self):
        """Background task to check model health."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._check_all_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")

    async def _check_all_models(self):
        """Check health of all configured models."""
        for model_id, model_config in self.config.models.items():
            try:
                healthy = await self._check_model_health(model_config)
                self._model_health[model_id] = healthy

                if not healthy:
                    self.logger.warning(f"Model {model_id} is unhealthy")

            except Exception as e:
                self._model_health[model_id] = False
                self.logger.error(f"Health check failed for {model_id}: {e}")

    async def _check_model_health(self, model_config: VLLMModelConfig) -> bool:
        """Check health of a single model."""
        try:
            endpoint = f"{model_config.endpoint}/health"

            async with self._session.get(
                endpoint,
                timeout=aiohttp.ClientTimeout(
                    total=self.config.health_check_timeout
                ),
            ) as response:
                return response.status == 200

        except Exception:
            return False

    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models."""
        status = {}
        for model_id, config in self.config.models.items():
            status[model_id] = {
                "healthy": self._model_health.get(model_id, False),
                "endpoint": config.endpoint,
                "model_name": config.model_name,
                "max_tokens": config.max_tokens,
                "supports_streaming": config.supports_streaming,
                "supports_function_calling": config.supports_function_calling,
            }
        return status


# =============================================================================
# Global Provider Instance
# =============================================================================

_vllm_provider: Optional[VLLMProvider] = None
_vllm_lock = asyncio.Lock()


async def get_vllm_provider() -> VLLMProvider:
    """Get the global vLLM provider instance."""
    global _vllm_provider

    if _vllm_provider is None:
        async with _vllm_lock:
            if _vllm_provider is None:
                _vllm_provider = VLLMProvider()
                await _vllm_provider.initialize()

    return _vllm_provider


async def shutdown_vllm_provider():
    """Shutdown the global vLLM provider."""
    global _vllm_provider

    async with _vllm_lock:
        if _vllm_provider is not None:
            await _vllm_provider.shutdown()
            _vllm_provider = None
