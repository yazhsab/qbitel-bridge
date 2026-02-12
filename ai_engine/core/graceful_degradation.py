"""
QBITEL Engine - Graceful Degradation Service

Provides graceful degradation strategies when services fail:
- Fallback implementations
- Cached responses
- Reduced functionality modes
- Health-based routing

Usage:
    from ai_engine.core.graceful_degradation import DegradationService, degradation_service

    # Register a fallback
    degradation_service.register_fallback(
        "llm_completion",
        fallback_fn=cached_completion,
        cache_ttl=300
    )

    # Use with automatic fallback
    result = await degradation_service.execute_with_fallback(
        "llm_completion",
        primary_fn=lambda: llm.complete(prompt),
    )
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from prometheus_client import Counter, Gauge, Histogram

from ai_engine.core.circuit_breakers import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    circuit_breakers,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Prometheus Metrics
# =============================================================================

DEGRADATION_MODE = Gauge(
    "qbitel_degradation_mode",
    "Current degradation mode (0=normal, 1=degraded, 2=minimal)",
    ["service"],
)

FALLBACK_EXECUTIONS = Counter(
    "qbitel_fallback_executions_total",
    "Total fallback executions",
    ["service", "fallback_type"],
)

CACHE_HITS = Counter(
    "qbitel_degradation_cache_hits_total",
    "Cache hits during degradation",
    ["service"],
)

CACHE_MISSES = Counter(
    "qbitel_degradation_cache_misses_total",
    "Cache misses during degradation",
    ["service"],
)

DEGRADATION_DURATION = Histogram(
    "qbitel_degradation_duration_seconds",
    "Time spent in degraded mode",
    ["service"],
    buckets=[1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600],
)


# =============================================================================
# Degradation Modes
# =============================================================================


class DegradationMode(str, Enum):
    """Service degradation modes."""

    NORMAL = "normal"  # Full functionality
    DEGRADED = "degraded"  # Reduced functionality, using fallbacks
    MINIMAL = "minimal"  # Minimum viable service
    OFFLINE = "offline"  # Service unavailable


# =============================================================================
# Fallback Strategies
# =============================================================================


class FallbackStrategy(str, Enum):
    """Fallback strategies for degraded mode."""

    CACHE = "cache"  # Return cached response
    STATIC = "static"  # Return static default
    REDUCED = "reduced"  # Execute with reduced functionality
    QUEUE = "queue"  # Queue for later processing
    REDIRECT = "redirect"  # Redirect to alternate service
    FAIL_OPEN = "fail_open"  # Allow through without service
    FAIL_CLOSED = "fail_closed"  # Reject with error


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CacheEntry:
    """Cached response entry."""

    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return datetime.now(timezone.utc) > self.expires_at


@dataclass
class FallbackConfig:
    """Configuration for a fallback handler."""

    name: str
    strategy: FallbackStrategy
    fallback_fn: Optional[Callable] = None
    static_value: Any = None
    cache_ttl: int = 300  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0
    enabled: bool = True
    priority: int = 0  # Higher priority fallbacks tried first


@dataclass
class ServiceHealth:
    """Health status for a service."""

    service_name: str
    mode: DegradationMode
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    degraded_since: Optional[datetime] = None
    error_rate: float = 0.0
    latency_p99: float = 0.0


# =============================================================================
# Response Cache
# =============================================================================


class ResponseCache:
    """
    Cache for storing fallback responses.

    Provides LRU caching with TTL support.
    """

    def __init__(self, max_size: int = 10000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    def _make_key(self, service: str, *args, **kwargs) -> str:
        """Generate cache key from service name and arguments."""
        key_data = {
            "service": service,
            "args": args,
            "kwargs": kwargs,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    async def get(self, service: str, *args, **kwargs) -> Optional[Any]:
        """Get cached value if exists and not expired."""
        key = self._make_key(service, *args, **kwargs)

        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                CACHE_MISSES.labels(service=service).inc()
                return None

            if entry.is_expired:
                del self._cache[key]
                CACHE_MISSES.labels(service=service).inc()
                return None

            entry.hit_count += 1
            CACHE_HITS.labels(service=service).inc()
            return entry.value

    async def set(
        self,
        service: str,
        value: Any,
        *args,
        ttl: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Store value in cache."""
        key = self._make_key(service, *args, **kwargs)
        ttl = ttl or self.default_ttl

        async with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self.max_size:
                await self._evict_oldest()

            now = datetime.now(timezone.utc)
            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl),
            )

    async def _evict_oldest(self) -> None:
        """Evict oldest entries to make room."""
        if not self._cache:
            return

        # Sort by created_at and remove oldest 10%
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at,
        )
        evict_count = max(1, len(sorted_keys) // 10)

        for key in sorted_keys[:evict_count]:
            del self._cache[key]

    async def invalidate(self, service: str, *args, **kwargs) -> None:
        """Invalidate specific cache entry."""
        key = self._make_key(service, *args, **kwargs)
        async with self._lock:
            if key in self._cache:
                del self._cache[key]

    async def clear(self, service: Optional[str] = None) -> None:
        """Clear cache entries."""
        async with self._lock:
            if service is None:
                self._cache.clear()
            else:
                keys_to_delete = [k for k, v in self._cache.items() if service in k]
                for key in keys_to_delete:
                    del self._cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hit_count for e in self._cache.values())
        expired = sum(1 for e in self._cache.values() if e.is_expired)

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "expired_entries": expired,
        }


# =============================================================================
# Degradation Service
# =============================================================================


class DegradationService:
    """
    Graceful degradation service.

    Manages fallbacks and degradation modes for all services.
    """

    def __init__(
        self,
        cache_max_size: int = 10000,
        cache_default_ttl: int = 300,
        failure_threshold: int = 5,
        recovery_threshold: int = 3,
    ):
        self.cache = ResponseCache(cache_max_size, cache_default_ttl)
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold

        self._fallbacks: Dict[str, List[FallbackConfig]] = {}
        self._health: Dict[str, ServiceHealth] = {}
        self._lock = asyncio.Lock()

        # Track degradation timing
        self._degradation_start: Dict[str, float] = {}

    def register_fallback(
        self,
        service_name: str,
        strategy: FallbackStrategy = FallbackStrategy.CACHE,
        fallback_fn: Optional[Callable] = None,
        static_value: Any = None,
        cache_ttl: int = 300,
        priority: int = 0,
    ) -> None:
        """
        Register a fallback for a service.

        Args:
            service_name: Name of the service
            strategy: Fallback strategy to use
            fallback_fn: Function to call for fallback
            static_value: Static value to return
            cache_ttl: Cache TTL in seconds
            priority: Priority (higher = tried first)
        """
        config = FallbackConfig(
            name=f"{service_name}_{strategy.value}",
            strategy=strategy,
            fallback_fn=fallback_fn,
            static_value=static_value,
            cache_ttl=cache_ttl,
            priority=priority,
        )

        if service_name not in self._fallbacks:
            self._fallbacks[service_name] = []

        self._fallbacks[service_name].append(config)
        self._fallbacks[service_name].sort(key=lambda x: -x.priority)

        # Initialize health tracking
        if service_name not in self._health:
            self._health[service_name] = ServiceHealth(
                service_name=service_name,
                mode=DegradationMode.NORMAL,
            )
            DEGRADATION_MODE.labels(service=service_name).set(0)

        logger.info(f"Registered fallback for '{service_name}': {strategy.value} (priority={priority})")

    async def execute_with_fallback(
        self,
        service_name: str,
        primary_fn: Callable[[], T],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute function with automatic fallback on failure.

        Args:
            service_name: Name of the service
            primary_fn: Primary function to execute
            *args: Arguments for caching key
            **kwargs: Keyword arguments for caching key

        Returns:
            Result from primary function or fallback
        """
        health = self._health.get(service_name)

        # Try primary function
        try:
            if asyncio.iscoroutinefunction(primary_fn):
                result = await primary_fn()
            else:
                result = primary_fn()

            # Success - update health and cache result
            await self._record_success(service_name)

            # Cache successful result for future fallback
            await self.cache.set(service_name, result, *args, **kwargs)

            return result

        except Exception as e:
            logger.warning(f"Primary function failed for '{service_name}': {e}")
            await self._record_failure(service_name, e)

            # Try fallbacks
            return await self._execute_fallbacks(service_name, e, *args, **kwargs)

    async def _execute_fallbacks(
        self,
        service_name: str,
        original_error: Exception,
        *args,
        **kwargs,
    ) -> Any:
        """Execute fallback strategies in priority order."""
        fallbacks = self._fallbacks.get(service_name, [])

        if not fallbacks:
            logger.error(f"No fallbacks registered for '{service_name}'")
            raise original_error

        for fallback in fallbacks:
            if not fallback.enabled:
                continue

            try:
                result = await self._execute_fallback(service_name, fallback, *args, **kwargs)
                FALLBACK_EXECUTIONS.labels(
                    service=service_name,
                    fallback_type=fallback.strategy.value,
                ).inc()
                return result

            except Exception as e:
                logger.warning(f"Fallback '{fallback.name}' failed for '{service_name}': {e}")
                continue

        # All fallbacks failed
        logger.error(f"All fallbacks failed for '{service_name}'")
        raise original_error

    async def _execute_fallback(
        self,
        service_name: str,
        fallback: FallbackConfig,
        *args,
        **kwargs,
    ) -> Any:
        """Execute a specific fallback strategy."""
        if fallback.strategy == FallbackStrategy.CACHE:
            cached = await self.cache.get(service_name, *args, **kwargs)
            if cached is not None:
                return cached
            raise ValueError("No cached value available")

        elif fallback.strategy == FallbackStrategy.STATIC:
            if fallback.static_value is not None:
                return fallback.static_value
            raise ValueError("No static value configured")

        elif fallback.strategy == FallbackStrategy.REDUCED:
            if fallback.fallback_fn is not None:
                if asyncio.iscoroutinefunction(fallback.fallback_fn):
                    return await fallback.fallback_fn(*args, **kwargs)
                return fallback.fallback_fn(*args, **kwargs)
            raise ValueError("No fallback function configured")

        elif fallback.strategy == FallbackStrategy.QUEUE:
            # Queue for later processing
            await self._queue_for_later(service_name, *args, **kwargs)
            return {"status": "queued", "message": "Request queued for later processing"}

        elif fallback.strategy == FallbackStrategy.REDIRECT:
            if fallback.fallback_fn is not None:
                if asyncio.iscoroutinefunction(fallback.fallback_fn):
                    return await fallback.fallback_fn(*args, **kwargs)
                return fallback.fallback_fn(*args, **kwargs)
            raise ValueError("No redirect handler configured")

        elif fallback.strategy == FallbackStrategy.FAIL_OPEN:
            return None  # Allow through

        elif fallback.strategy == FallbackStrategy.FAIL_CLOSED:
            raise ValueError(f"Service '{service_name}' is unavailable")

        raise ValueError(f"Unknown fallback strategy: {fallback.strategy}")

    async def _queue_for_later(
        self,
        service_name: str,
        *args,
        **kwargs,
    ) -> None:
        """Queue request for later processing."""
        # In production, this would push to a message queue
        logger.info(f"Queuing request for '{service_name}' for later processing")

    async def _record_success(self, service_name: str) -> None:
        """Record successful service call."""
        async with self._lock:
            if service_name not in self._health:
                self._health[service_name] = ServiceHealth(
                    service_name=service_name,
                    mode=DegradationMode.NORMAL,
                )

            health = self._health[service_name]
            health.last_success = datetime.now(timezone.utc)
            health.consecutive_successes += 1
            health.consecutive_failures = 0

            # Check for recovery
            if health.mode != DegradationMode.NORMAL and health.consecutive_successes >= self.recovery_threshold:
                await self._recover_service(service_name)

    async def _record_failure(
        self,
        service_name: str,
        error: Exception,
    ) -> None:
        """Record failed service call."""
        async with self._lock:
            if service_name not in self._health:
                self._health[service_name] = ServiceHealth(
                    service_name=service_name,
                    mode=DegradationMode.NORMAL,
                )

            health = self._health[service_name]
            health.last_failure = datetime.now(timezone.utc)
            health.consecutive_failures += 1
            health.consecutive_successes = 0

            # Check for degradation
            if health.mode == DegradationMode.NORMAL and health.consecutive_failures >= self.failure_threshold:
                await self._degrade_service(service_name)
            elif health.mode == DegradationMode.DEGRADED and health.consecutive_failures >= self.failure_threshold * 2:
                await self._minimize_service(service_name)

    async def _degrade_service(self, service_name: str) -> None:
        """Transition service to degraded mode."""
        health = self._health[service_name]
        health.mode = DegradationMode.DEGRADED
        health.degraded_since = datetime.now(timezone.utc)

        self._degradation_start[service_name] = time.time()
        DEGRADATION_MODE.labels(service=service_name).set(1)

        logger.warning(f"Service '{service_name}' entering DEGRADED mode")

    async def _minimize_service(self, service_name: str) -> None:
        """Transition service to minimal mode."""
        health = self._health[service_name]
        health.mode = DegradationMode.MINIMAL

        DEGRADATION_MODE.labels(service=service_name).set(2)

        logger.error(f"Service '{service_name}' entering MINIMAL mode")

    async def _recover_service(self, service_name: str) -> None:
        """Recover service to normal mode."""
        health = self._health[service_name]
        old_mode = health.mode
        health.mode = DegradationMode.NORMAL
        health.degraded_since = None

        # Record degradation duration
        if service_name in self._degradation_start:
            duration = time.time() - self._degradation_start[service_name]
            DEGRADATION_DURATION.labels(service=service_name).observe(duration)
            del self._degradation_start[service_name]

        DEGRADATION_MODE.labels(service=service_name).set(0)

        logger.info(f"Service '{service_name}' recovered from {old_mode.value} to NORMAL")

    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get health status for a service."""
        return self._health.get(service_name)

    def get_all_health(self) -> Dict[str, ServiceHealth]:
        """Get health status for all services."""
        return self._health.copy()

    def get_degraded_services(self) -> List[str]:
        """Get list of services currently in degraded mode."""
        return [name for name, health in self._health.items() if health.mode != DegradationMode.NORMAL]

    async def force_mode(
        self,
        service_name: str,
        mode: DegradationMode,
    ) -> None:
        """Force a service into a specific mode."""
        async with self._lock:
            if service_name not in self._health:
                self._health[service_name] = ServiceHealth(
                    service_name=service_name,
                    mode=mode,
                )
            else:
                self._health[service_name].mode = mode

            mode_values = {
                DegradationMode.NORMAL: 0,
                DegradationMode.DEGRADED: 1,
                DegradationMode.MINIMAL: 2,
                DegradationMode.OFFLINE: 3,
            }
            DEGRADATION_MODE.labels(service=service_name).set(mode_values[mode])

            logger.info(f"Service '{service_name}' forced to {mode.value} mode")

    def get_status(self) -> Dict[str, Any]:
        """Get overall degradation service status."""
        degraded = self.get_degraded_services()
        cache_stats = self.cache.get_stats()

        return {
            "total_services": len(self._health),
            "degraded_services": degraded,
            "degraded_count": len(degraded),
            "cache": cache_stats,
            "health": {
                name: {
                    "mode": h.mode.value,
                    "consecutive_failures": h.consecutive_failures,
                    "consecutive_successes": h.consecutive_successes,
                    "degraded_since": (h.degraded_since.isoformat() if h.degraded_since else None),
                }
                for name, h in self._health.items()
            },
        }


# =============================================================================
# Decorator
# =============================================================================


def with_graceful_degradation(
    service_name: str,
    fallback_value: Any = None,
    cache_result: bool = True,
    cache_ttl: int = 300,
) -> Callable:
    """
    Decorator for graceful degradation.

    Args:
        service_name: Name of the service
        fallback_value: Default value if all fallbacks fail
        cache_result: Whether to cache successful results
        cache_ttl: Cache TTL in seconds

    Example:
        @with_graceful_degradation("user_service", fallback_value=[])
        async def get_users():
            return await user_api.list_users()
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await degradation_service.execute_with_fallback(
                    service_name,
                    lambda: func(*args, **kwargs),
                    *args,
                    **kwargs,
                )
                return result
            except Exception:
                if fallback_value is not None:
                    return fallback_value
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Global Instance
# =============================================================================

# Create global degradation service
degradation_service = DegradationService()

# Register default fallbacks for common services
degradation_service.register_fallback(
    "llm",
    strategy=FallbackStrategy.CACHE,
    cache_ttl=600,
    priority=10,
)

degradation_service.register_fallback(
    "llm",
    strategy=FallbackStrategy.REDUCED,
    fallback_fn=lambda prompt, **kwargs: {
        "response": "Service temporarily unavailable. Please try again.",
        "degraded": True,
    },
    priority=5,
)

degradation_service.register_fallback(
    "database",
    strategy=FallbackStrategy.CACHE,
    cache_ttl=60,
    priority=10,
)

degradation_service.register_fallback(
    "external_api",
    strategy=FallbackStrategy.CACHE,
    cache_ttl=300,
    priority=10,
)

degradation_service.register_fallback(
    "external_api",
    strategy=FallbackStrategy.QUEUE,
    priority=5,
)

degradation_service.register_fallback(
    "discovery",
    strategy=FallbackStrategy.CACHE,
    cache_ttl=3600,  # Cache discovery results longer
    priority=10,
)


# =============================================================================
# Health Check Integration
# =============================================================================


async def check_degradation_health() -> Dict[str, Any]:
    """Health check for degradation service."""
    status = degradation_service.get_status()
    degraded = status["degraded_services"]

    health_status = "healthy"
    if len(degraded) > 0:
        health_status = "degraded"
    if len(degraded) > len(status["health"]) // 2:
        health_status = "unhealthy"

    return {
        "status": health_status,
        "degraded_services": degraded,
        "total_services": status["total_services"],
        "cache_size": status["cache"]["size"],
    }
