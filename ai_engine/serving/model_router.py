"""
Model Router for Intelligent Request Routing

Provides intelligent routing for model inference requests:
- A/B testing and canary deployments
- Load balancing across replicas
- Weighted traffic splitting
- Fallback routing
- Latency-based routing
- Request deduplication
"""

import asyncio
import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy types."""

    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_CONNECTIONS = "least_connections"
    LATENCY_BASED = "latency_based"
    HASH_BASED = "hash_based"  # Consistent hashing
    RANDOM = "random"
    CANARY = "canary"
    FAILOVER = "failover"


@dataclass
class ModelEndpoint:
    """Represents a model serving endpoint."""

    name: str
    url: str
    weight: float = 1.0
    priority: int = 0  # Lower = higher priority for failover

    # Health status
    healthy: bool = True
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0

    # Metrics
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0

    # Metadata
    version: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def update_latency(self, latency_ms: float) -> None:
        """Update rolling average latency."""
        alpha = 0.1  # Exponential moving average factor
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms


@dataclass
class TrafficSplit:
    """Traffic split configuration for A/B testing."""

    # Traffic percentages (must sum to 100)
    splits: Dict[str, int]  # endpoint_name -> percentage

    # Optional: split by user/session
    split_key: Optional[str] = None  # Header or parameter to use

    # Sticky sessions
    sticky: bool = True
    sticky_ttl_seconds: int = 3600


@dataclass
class RoutingConfig:
    """Configuration for model routing."""

    strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN
    traffic_split: Optional[TrafficSplit] = None

    # Health check configuration
    health_check_interval: int = 30
    health_check_timeout: int = 10
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2

    # Circuit breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5  # Failures before opening
    circuit_breaker_timeout: int = 30  # Seconds before half-open

    # Retry configuration
    max_retries: int = 2
    retry_delay_ms: int = 100
    retry_on_status: List[int] = field(default_factory=lambda: [502, 503, 504])

    # Timeout
    request_timeout: int = 300

    # Request deduplication
    dedup_enabled: bool = True
    dedup_window_ms: int = 100


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for endpoint protection."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None

    def record_success(self) -> None:
        """Record successful request."""
        self.failure_count = 0
        self.last_success_time = datetime.utcnow()
        self.state = CircuitState.CLOSED

    def record_failure(self, threshold: int) -> None:
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= threshold:
            self.state = CircuitState.OPEN

    def should_allow(self, timeout: int) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= timeout:
                    self.state = CircuitState.HALF_OPEN
                    return True
            return False

        # Half-open: allow one request
        return True


class ModelRouter:
    """
    Intelligent model request router.

    Features:
    - Multiple routing strategies
    - A/B testing and canary deployments
    - Health checking and circuit breaking
    - Request deduplication
    - Retry with backoff

    Example:
        router = ModelRouter()

        # Add endpoints
        router.add_endpoint(ModelEndpoint(
            name="model-v1",
            url="http://model-v1.svc:8000",
            weight=80
        ))
        router.add_endpoint(ModelEndpoint(
            name="model-v2",
            url="http://model-v2.svc:8000",
            weight=20
        ))

        # Route request
        endpoint = await router.route(
            request_id="req-123",
            context={"user_id": "user-456"}
        )

        # Make request to endpoint.url
    """

    def __init__(self, config: Optional[RoutingConfig] = None):
        """Initialize router."""
        self.config = config or RoutingConfig()
        self._endpoints: Dict[str, ModelEndpoint] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._sticky_sessions: Dict[str, Tuple[str, datetime]] = {}
        self._round_robin_index = 0
        self._dedup_cache: Dict[str, Tuple[str, float]] = {}  # hash -> (endpoint, time)
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None

        logger.info(f"Model router initialized with strategy: {self.config.strategy}")

    def add_endpoint(self, endpoint: ModelEndpoint) -> None:
        """Add a model endpoint."""
        self._endpoints[endpoint.name] = endpoint
        self._circuit_breakers[endpoint.name] = CircuitBreaker()
        logger.info(f"Added endpoint: {endpoint.name} ({endpoint.url})")

    def remove_endpoint(self, name: str) -> None:
        """Remove a model endpoint."""
        if name in self._endpoints:
            del self._endpoints[name]
            del self._circuit_breakers[name]
            logger.info(f"Removed endpoint: {name}")

    def update_endpoint(self, endpoint: ModelEndpoint) -> None:
        """Update an existing endpoint."""
        if endpoint.name in self._endpoints:
            self._endpoints[endpoint.name] = endpoint
            logger.info(f"Updated endpoint: {endpoint.name}")

    async def route(
        self,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        force_endpoint: Optional[str] = None,
    ) -> ModelEndpoint:
        """
        Route a request to an appropriate endpoint.

        Args:
            request_id: Unique request identifier
            context: Request context (user_id, headers, etc.)
            force_endpoint: Force routing to specific endpoint

        Returns:
            Selected ModelEndpoint
        """
        context = context or {}

        # Force endpoint if specified
        if force_endpoint and force_endpoint in self._endpoints:
            return self._endpoints[force_endpoint]

        # Check deduplication
        if self.config.dedup_enabled and request_id:
            cached = self._check_dedup(request_id, context)
            if cached:
                return cached

        # Get healthy endpoints
        healthy_endpoints = await self._get_healthy_endpoints()
        if not healthy_endpoints:
            raise RuntimeError("No healthy endpoints available")

        # Select endpoint based on strategy
        endpoint = await self._select_endpoint(healthy_endpoints, context)

        # Cache for deduplication
        if self.config.dedup_enabled and request_id:
            self._cache_dedup(request_id, context, endpoint)

        # Update metrics
        endpoint.active_connections += 1
        endpoint.total_requests += 1

        return endpoint

    async def record_result(
        self,
        endpoint_name: str,
        success: bool,
        latency_ms: float,
        status_code: Optional[int] = None,
    ) -> None:
        """
        Record request result for an endpoint.

        Args:
            endpoint_name: Endpoint name
            success: Whether request succeeded
            latency_ms: Request latency
            status_code: HTTP status code
        """
        endpoint = self._endpoints.get(endpoint_name)
        if not endpoint:
            return

        endpoint.active_connections = max(0, endpoint.active_connections - 1)
        endpoint.update_latency(latency_ms)

        circuit = self._circuit_breakers.get(endpoint_name)
        if circuit:
            if success:
                circuit.record_success()
            else:
                circuit.record_failure(self.config.circuit_breaker_threshold)
                endpoint.total_errors += 1

    async def _get_healthy_endpoints(self) -> List[ModelEndpoint]:
        """Get list of healthy endpoints."""
        healthy = []
        for name, endpoint in self._endpoints.items():
            if not endpoint.healthy:
                continue

            # Check circuit breaker
            circuit = self._circuit_breakers.get(name)
            if circuit and self.config.circuit_breaker_enabled:
                if not circuit.should_allow(self.config.circuit_breaker_timeout):
                    continue

            healthy.append(endpoint)

        return healthy

    async def _select_endpoint(
        self,
        endpoints: List[ModelEndpoint],
        context: Dict[str, Any],
    ) -> ModelEndpoint:
        """Select endpoint based on routing strategy."""

        # Check for traffic split (A/B testing)
        if self.config.traffic_split:
            return await self._select_with_traffic_split(endpoints, context)

        strategy = self.config.strategy

        if strategy == RoutingStrategy.ROUND_ROBIN:
            return await self._round_robin(endpoints)

        elif strategy == RoutingStrategy.WEIGHTED:
            return await self._weighted_random(endpoints)

        elif strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return await self._least_connections(endpoints)

        elif strategy == RoutingStrategy.LATENCY_BASED:
            return await self._latency_based(endpoints)

        elif strategy == RoutingStrategy.HASH_BASED:
            return await self._hash_based(endpoints, context)

        elif strategy == RoutingStrategy.RANDOM:
            return random.choice(endpoints)

        elif strategy == RoutingStrategy.FAILOVER:
            return await self._failover(endpoints)

        elif strategy == RoutingStrategy.CANARY:
            return await self._canary(endpoints, context)

        else:
            return random.choice(endpoints)

    async def _select_with_traffic_split(
        self,
        endpoints: List[ModelEndpoint],
        context: Dict[str, Any],
    ) -> ModelEndpoint:
        """Select endpoint based on traffic split configuration."""
        split = self.config.traffic_split

        # Check sticky session
        if split.sticky:
            sticky_key = self._get_sticky_key(context, split.split_key)
            if sticky_key:
                cached = self._sticky_sessions.get(sticky_key)
                if cached:
                    endpoint_name, cached_time = cached
                    if (datetime.utcnow() - cached_time).total_seconds() < split.sticky_ttl_seconds:
                        endpoint = self._endpoints.get(endpoint_name)
                        if endpoint and endpoint in endpoints:
                            return endpoint

        # Select based on split percentages
        endpoint_names = list(split.splits.keys())
        weights = [split.splits[name] for name in endpoint_names]

        # Filter to only healthy endpoints
        available_endpoints = [
            (name, weight) for name, weight in zip(endpoint_names, weights) if name in [e.name for e in endpoints]
        ]

        if not available_endpoints:
            return random.choice(endpoints)

        names, weights = zip(*available_endpoints)
        selected_name = random.choices(names, weights=weights)[0]
        endpoint = self._endpoints[selected_name]

        # Store sticky session
        if split.sticky:
            sticky_key = self._get_sticky_key(context, split.split_key)
            if sticky_key:
                self._sticky_sessions[sticky_key] = (selected_name, datetime.utcnow())

        return endpoint

    async def _round_robin(
        self,
        endpoints: List[ModelEndpoint],
    ) -> ModelEndpoint:
        """Round-robin selection."""
        async with self._lock:
            endpoint = endpoints[self._round_robin_index % len(endpoints)]
            self._round_robin_index += 1
            return endpoint

    async def _weighted_random(
        self,
        endpoints: List[ModelEndpoint],
    ) -> ModelEndpoint:
        """Weighted random selection."""
        weights = [e.weight for e in endpoints]
        return random.choices(endpoints, weights=weights)[0]

    async def _least_connections(
        self,
        endpoints: List[ModelEndpoint],
    ) -> ModelEndpoint:
        """Select endpoint with least active connections."""
        return min(endpoints, key=lambda e: e.active_connections)

    async def _latency_based(
        self,
        endpoints: List[ModelEndpoint],
    ) -> ModelEndpoint:
        """Select endpoint with lowest latency."""
        # Filter out endpoints with no latency data
        with_latency = [e for e in endpoints if e.avg_latency_ms > 0]
        if not with_latency:
            return random.choice(endpoints)

        # Use inverse latency as weight for probabilistic selection
        weights = [1.0 / max(e.avg_latency_ms, 1) for e in with_latency]
        return random.choices(with_latency, weights=weights)[0]

    async def _hash_based(
        self,
        endpoints: List[ModelEndpoint],
        context: Dict[str, Any],
    ) -> ModelEndpoint:
        """Consistent hash-based selection."""
        # Get hash key from context
        hash_key = context.get("user_id") or context.get("session_id") or str(random.random())
        hash_value = int(hashlib.md5(str(hash_key).encode()).hexdigest(), 16)
        index = hash_value % len(endpoints)
        return endpoints[index]

    async def _failover(
        self,
        endpoints: List[ModelEndpoint],
    ) -> ModelEndpoint:
        """Priority-based failover selection."""
        # Sort by priority (lower = higher priority)
        sorted_endpoints = sorted(endpoints, key=lambda e: e.priority)
        return sorted_endpoints[0]

    async def _canary(
        self,
        endpoints: List[ModelEndpoint],
        context: Dict[str, Any],
    ) -> ModelEndpoint:
        """Canary deployment selection."""
        # Find canary and stable endpoints
        canary = [e for e in endpoints if e.tags.get("canary") == "true"]
        stable = [e for e in endpoints if e.tags.get("canary") != "true"]

        if not canary:
            return random.choice(stable) if stable else random.choice(endpoints)

        if not stable:
            return random.choice(canary)

        # Route small percentage to canary
        canary_percent = int(canary[0].tags.get("canary_percent", "10"))
        if random.randint(1, 100) <= canary_percent:
            return random.choice(canary)
        return random.choice(stable)

    def _get_sticky_key(
        self,
        context: Dict[str, Any],
        split_key: Optional[str],
    ) -> Optional[str]:
        """Get key for sticky session."""
        if split_key and split_key in context:
            return str(context[split_key])
        return context.get("user_id") or context.get("session_id")

    def _check_dedup(
        self,
        request_id: str,
        context: Dict[str, Any],
    ) -> Optional[ModelEndpoint]:
        """Check deduplication cache."""
        # Create hash of request
        hash_data = f"{request_id}:{sorted(context.items())}"
        request_hash = hashlib.md5(hash_data.encode()).hexdigest()

        cached = self._dedup_cache.get(request_hash)
        if cached:
            endpoint_name, cache_time = cached
            if (time.time() - cache_time) * 1000 < self.config.dedup_window_ms:
                return self._endpoints.get(endpoint_name)

        return None

    def _cache_dedup(
        self,
        request_id: str,
        context: Dict[str, Any],
        endpoint: ModelEndpoint,
    ) -> None:
        """Cache endpoint selection for deduplication."""
        hash_data = f"{request_id}:{sorted(context.items())}"
        request_hash = hashlib.md5(hash_data.encode()).hexdigest()
        self._dedup_cache[request_hash] = (endpoint.name, time.time())

        # Cleanup old entries
        self._cleanup_dedup_cache()

    def _cleanup_dedup_cache(self) -> None:
        """Remove old dedup cache entries."""
        current_time = time.time()
        window_seconds = self.config.dedup_window_ms / 1000

        expired = [key for key, (_, cache_time) in self._dedup_cache.items() if current_time - cache_time > window_seconds]

        for key in expired:
            del self._dedup_cache[key]

    async def start_health_checks(self) -> None:
        """Start periodic health checks."""
        if self._health_check_task:
            return

        async def health_check_loop():
            while True:
                try:
                    await self._run_health_checks()
                    await asyncio.sleep(self.config.health_check_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    await asyncio.sleep(5)

        self._health_check_task = asyncio.create_task(health_check_loop())
        logger.info("Health check task started")

    async def stop_health_checks(self) -> None:
        """Stop health checks."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Health check task stopped")

    async def _run_health_checks(self) -> None:
        """Run health checks on all endpoints."""
        tasks = [self._check_endpoint_health(endpoint) for endpoint in self._endpoints.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_endpoint_health(self, endpoint: ModelEndpoint) -> None:
        """Check health of a single endpoint."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{endpoint.url}/v2/health/ready",
                    timeout=self.config.health_check_timeout,
                )
                is_healthy = response.status_code == 200

        except ImportError:
            # No httpx, assume healthy
            is_healthy = True
        except Exception:
            is_healthy = False

        endpoint.last_health_check = datetime.utcnow()

        if is_healthy:
            endpoint.consecutive_failures = 0
            if not endpoint.healthy:
                # Check healthy threshold
                endpoint.healthy = True
                logger.info(f"Endpoint {endpoint.name} is now healthy")
        else:
            endpoint.consecutive_failures += 1
            if endpoint.consecutive_failures >= self.config.unhealthy_threshold:
                if endpoint.healthy:
                    endpoint.healthy = False
                    logger.warning(f"Endpoint {endpoint.name} is now unhealthy")

    def get_metrics(self) -> Dict[str, Any]:
        """Get router metrics."""
        return {
            "endpoints": {
                name: {
                    "url": e.url,
                    "healthy": e.healthy,
                    "active_connections": e.active_connections,
                    "total_requests": e.total_requests,
                    "total_errors": e.total_errors,
                    "avg_latency_ms": e.avg_latency_ms,
                    "weight": e.weight,
                }
                for name, e in self._endpoints.items()
            },
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                }
                for name, cb in self._circuit_breakers.items()
            },
            "strategy": self.config.strategy.value,
        }
