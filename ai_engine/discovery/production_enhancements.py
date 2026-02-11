"""
QBITEL Engine - Protocol Discovery Production Enhancements

This module provides production-ready enhancements for the protocol discovery engine,
including advanced monitoring, caching, error handling, and performance optimization.
"""

import asyncio
import logging
import time
import hashlib
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from enum import Enum
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

from prometheus_client import Counter, Histogram, Gauge, Summary
import redis
from circuitbreaker import circuit
import structlog

from ..core.config import Config
from ..core.exceptions import ProtocolException, ModelException


# ============================================================================
# ENHANCED METRICS
# ============================================================================


class DiscoveryMetrics:
    """Comprehensive metrics for protocol discovery operations."""

    def __init__(self):
        # Discovery operation metrics
        self.discovery_requests_total = Counter(
            "qbitel_discovery_requests_total",
            "Total protocol discovery requests",
            ["protocol_type", "status", "cache_hit"],
        )

        self.discovery_duration_seconds = Histogram(
            "qbitel_discovery_duration_seconds",
            "Protocol discovery duration",
            ["protocol_type", "phase"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        self.discovery_confidence_score = Histogram(
            "qbitel_discovery_confidence_score",
            "Discovery confidence scores",
            ["protocol_type"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # Component-specific metrics
        self.statistical_analysis_duration = Histogram(
            "qbitel_statistical_analysis_duration_seconds",
            "Statistical analysis duration",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )

        self.grammar_learning_duration = Histogram(
            "qbitel_grammar_learning_duration_seconds",
            "Grammar learning duration",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )

        self.parser_generation_duration = Histogram(
            "qbitel_parser_generation_duration_seconds",
            "Parser generation duration",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        )

        self.classification_duration = Histogram(
            "qbitel_classification_duration_seconds",
            "Protocol classification duration",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        )

        self.validation_duration = Histogram(
            "qbitel_validation_duration_seconds",
            "Message validation duration",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        )

        # Cache metrics
        self.cache_operations_total = Counter(
            "qbitel_cache_operations_total",
            "Total cache operations",
            ["operation", "cache_type", "status"],
        )

        self.cache_size_bytes = Gauge(
            "qbitel_cache_size_bytes", "Current cache size in bytes", ["cache_type"]
        )

        self.cache_hit_rate = Gauge(
            "qbitel_cache_hit_rate", "Cache hit rate", ["cache_type"]
        )

        # Resource metrics
        self.active_discoveries = Gauge(
            "qbitel_active_discoveries", "Number of active discovery operations"
        )

        self.protocol_profiles_count = Gauge(
            "qbitel_protocol_profiles_count", "Number of learned protocol profiles"
        )

        self.model_memory_usage_bytes = Gauge(
            "qbitel_model_memory_usage_bytes",
            "Memory usage by ML models",
            ["model_type"],
        )

        # Error metrics
        self.discovery_errors_total = Counter(
            "qbitel_discovery_errors_total",
            "Total discovery errors",
            ["error_type", "component"],
        )

        self.circuit_breaker_state = Gauge(
            "qbitel_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
            ["component"],
        )

        # LLM integration metrics
        self.llm_analysis_requests_total = Counter(
            "qbitel_llm_analysis_requests_total",
            "Total LLM analysis requests",
            ["analysis_type", "status"],
        )

        self.llm_analysis_duration_seconds = Histogram(
            "qbitel_llm_analysis_duration_seconds",
            "LLM analysis duration",
            ["analysis_type"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )

        self.llm_token_usage = Counter(
            "qbitel_llm_token_usage_total",
            "Total LLM tokens used",
            ["provider", "operation"],
        )


# ============================================================================
# DISTRIBUTED CACHING
# ============================================================================


class CacheBackend(Enum):
    """Cache backend types."""

    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[float] = None
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class DistributedCache:
    """
    Production-grade distributed caching system.

    Supports multiple backends (memory, Redis) with automatic failover,
    TTL management, and cache warming strategies.
    """

    def __init__(
        self,
        backend: CacheBackend = CacheBackend.HYBRID,
        redis_url: Optional[str] = None,
        max_memory_size: int = 1024 * 1024 * 1024,  # 1GB
        default_ttl: float = 3600.0,  # 1 hour
        metrics: Optional[DiscoveryMetrics] = None,
    ):
        self.backend = backend
        self.max_memory_size = max_memory_size
        self.default_ttl = default_ttl
        self.metrics = metrics or DiscoveryMetrics()
        self.logger = structlog.get_logger(__name__)

        # Memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_cache_lock = threading.RLock()
        self.current_memory_size = 0

        # Redis cache
        self.redis_client: Optional[redis.Redis] = None
        if backend in [CacheBackend.REDIS, CacheBackend.HYBRID] and redis_url:
            try:
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                self.redis_client.ping()
                self.logger.info("Redis cache connected", url=redis_url)
            except Exception as e:
                self.logger.warning(
                    "Redis connection failed, using memory only", error=str(e)
                )
                self.redis_client = None

        # Cache statistics
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0, "evictions": 0}

        # LRU tracking
        self.access_order: deque = deque(maxlen=10000)

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time()

        try:
            # Try memory cache first
            with self.memory_cache_lock:
                if key in self.memory_cache:
                    entry = self.memory_cache[key]

                    if entry.is_expired():
                        del self.memory_cache[key]
                        self.current_memory_size -= entry.size_bytes
                        self.metrics.cache_operations_total.labels(
                            operation="get", cache_type="memory", status="expired"
                        ).inc()
                    else:
                        entry.accessed_at = time.time()
                        entry.access_count += 1
                        self.access_order.append(key)
                        self.stats["hits"] += 1

                        self.metrics.cache_operations_total.labels(
                            operation="get", cache_type="memory", status="hit"
                        ).inc()

                        self._update_cache_metrics()
                        return entry.value

            # Try Redis if available
            if self.redis_client and self.backend in [
                CacheBackend.REDIS,
                CacheBackend.HYBRID,
            ]:
                try:
                    value_bytes = self.redis_client.get(key)
                    if value_bytes:
                        value = pickle.loads(value_bytes)
                        self.stats["hits"] += 1

                        self.metrics.cache_operations_total.labels(
                            operation="get", cache_type="redis", status="hit"
                        ).inc()

                        # Promote to memory cache
                        await self._set_memory(key, value, self.default_ttl)

                        self._update_cache_metrics()
                        return value
                except Exception as e:
                    self.logger.warning("Redis get failed", key=key, error=str(e))

            # Cache miss
            self.stats["misses"] += 1
            self.metrics.cache_operations_total.labels(
                operation="get", cache_type="all", status="miss"
            ).inc()

            self._update_cache_metrics()
            return None

        finally:
            duration = time.time() - start_time
            if duration > 0.1:  # Log slow cache operations
                self.logger.warning("Slow cache get", key=key, duration=duration)

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        ttl = ttl or self.default_ttl

        try:
            # Set in memory cache
            success_memory = await self._set_memory(key, value, ttl)

            # Set in Redis if available
            success_redis = True
            if self.redis_client and self.backend in [
                CacheBackend.REDIS,
                CacheBackend.HYBRID,
            ]:
                try:
                    value_bytes = pickle.dumps(value)
                    self.redis_client.setex(key, int(ttl), value_bytes)

                    self.metrics.cache_operations_total.labels(
                        operation="set", cache_type="redis", status="success"
                    ).inc()
                except Exception as e:
                    self.logger.warning("Redis set failed", key=key, error=str(e))
                    success_redis = False

            self.stats["sets"] += 1
            return success_memory or success_redis

        except Exception as e:
            self.logger.error("Cache set failed", key=key, error=str(e))
            self.metrics.cache_operations_total.labels(
                operation="set", cache_type="all", status="error"
            ).inc()
            return False

    async def _set_memory(self, key: str, value: Any, ttl: float) -> bool:
        """Set value in memory cache with LRU eviction."""
        try:
            value_bytes = pickle.dumps(value)
            size_bytes = len(value_bytes)

            with self.memory_cache_lock:
                # Evict if necessary
                while (
                    self.current_memory_size + size_bytes > self.max_memory_size
                    and self.memory_cache
                ):
                    await self._evict_lru()

                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    access_count=0,
                    ttl=ttl,
                    size_bytes=size_bytes,
                )

                # Remove old entry if exists
                if key in self.memory_cache:
                    old_entry = self.memory_cache[key]
                    self.current_memory_size -= old_entry.size_bytes

                self.memory_cache[key] = entry
                self.current_memory_size += size_bytes
                self.access_order.append(key)

                self.metrics.cache_operations_total.labels(
                    operation="set", cache_type="memory", status="success"
                ).inc()

                self.metrics.cache_size_bytes.labels(cache_type="memory").set(
                    self.current_memory_size
                )

                return True

        except Exception as e:
            self.logger.error("Memory cache set failed", key=key, error=str(e))
            return False

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.access_order:
            # Fallback: evict oldest entry
            if self.memory_cache:
                oldest_key = min(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k].accessed_at,
                )
                entry = self.memory_cache.pop(oldest_key)
                self.current_memory_size -= entry.size_bytes
                self.stats["evictions"] += 1
            return

        # Find LRU entry
        while self.access_order:
            lru_key = self.access_order.popleft()
            if lru_key in self.memory_cache:
                entry = self.memory_cache.pop(lru_key)
                self.current_memory_size -= entry.size_bytes
                self.stats["evictions"] += 1

                self.metrics.cache_operations_total.labels(
                    operation="evict", cache_type="memory", status="success"
                ).inc()
                break

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        deleted = False

        # Delete from memory
        with self.memory_cache_lock:
            if key in self.memory_cache:
                entry = self.memory_cache.pop(key)
                self.current_memory_size -= entry.size_bytes
                deleted = True

        # Delete from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
                deleted = True
            except Exception as e:
                self.logger.warning("Redis delete failed", key=key, error=str(e))

        if deleted:
            self.stats["deletes"] += 1
            self.metrics.cache_operations_total.labels(
                operation="delete", cache_type="all", status="success"
            ).inc()

        return deleted

    async def clear(self) -> None:
        """Clear all cache entries."""
        with self.memory_cache_lock:
            self.memory_cache.clear()
            self.current_memory_size = 0
            self.access_order.clear()

        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                self.logger.warning("Redis flush failed", error=str(e))

        self.logger.info("Cache cleared")

    def _update_cache_metrics(self) -> None:
        """Update cache metrics."""
        total_ops = self.stats["hits"] + self.stats["misses"]
        if total_ops > 0:
            hit_rate = self.stats["hits"] / total_ops
            self.metrics.cache_hit_rate.labels(cache_type="all").set(hit_rate)

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired entries."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Run every minute

                with self.memory_cache_lock:
                    expired_keys = [
                        key
                        for key, entry in self.memory_cache.items()
                        if entry.is_expired()
                    ]

                    for key in expired_keys:
                        entry = self.memory_cache.pop(key)
                        self.current_memory_size -= entry.size_bytes

                    if expired_keys:
                        self.logger.info(
                            "Cleaned up expired cache entries", count=len(expired_keys)
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Cache cleanup failed", error=str(e))

    async def shutdown(self) -> None:
        """Shutdown cache system."""
        self._shutdown_event.set()

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self.redis_client:
            self.redis_client.close()

        self.logger.info("Cache system shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_ops = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_ops if total_ops > 0 else 0.0

        return {
            "backend": self.backend.value,
            "memory_cache_size": len(self.memory_cache),
            "memory_usage_bytes": self.current_memory_size,
            "memory_usage_mb": self.current_memory_size / (1024 * 1024),
            "max_memory_mb": self.max_memory_size / (1024 * 1024),
            "redis_connected": self.redis_client is not None,
            "hit_rate": hit_rate,
            **self.stats,
        }


# ============================================================================
# ENHANCED ERROR HANDLING
# ============================================================================


class DiscoveryError(Exception):
    """Base exception for discovery errors."""

    def __init__(
        self, message: str, component: str, recoverable: bool = True, **kwargs
    ):
        super().__init__(message)
        self.component = component
        self.recoverable = recoverable
        self.metadata = kwargs
        self.timestamp = datetime.utcnow()


class StatisticalAnalysisError(DiscoveryError):
    """Error in statistical analysis."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="statistical_analyzer", **kwargs)


class GrammarLearningError(DiscoveryError):
    """Error in grammar learning."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="grammar_learner", **kwargs)


class ParserGenerationError(DiscoveryError):
    """Error in parser generation."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="parser_generator", **kwargs)


class ClassificationError(DiscoveryError):
    """Error in protocol classification."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="classifier", **kwargs)


class ValidationError(DiscoveryError):
    """Error in message validation."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="validator", **kwargs)


class ErrorRecoveryStrategy:
    """Strategy for error recovery."""

    @staticmethod
    async def retry_with_backoff(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: bool = True,
    ) -> Any:
        """Retry function with exponential backoff."""
        import random

        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise

                delay = min(base_delay * (2**attempt), max_delay)
                if jitter:
                    delay *= 0.5 + random.random()

                logging.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s",
                    exc_info=True,
                )
                await asyncio.sleep(delay)

    @staticmethod
    def with_fallback(primary_func: Callable, fallback_func: Callable) -> Callable:
        """Execute function with fallback."""

        async def wrapper(*args, **kwargs):
            try:
                return await primary_func(*args, **kwargs)
            except Exception as e:
                logging.warning(f"Primary function failed, using fallback: {e}")
                return await fallback_func(*args, **kwargs)

        return wrapper


# ============================================================================
# PRODUCTION CONFIGURATION
# ============================================================================


@dataclass
class ProductionConfig:
    """Production-specific configuration."""

    # Performance
    enable_caching: bool = True
    cache_backend: CacheBackend = CacheBackend.HYBRID
    redis_url: Optional[str] = None
    max_cache_size_mb: int = 1024
    cache_ttl_seconds: float = 3600.0

    # Reliability
    enable_circuit_breakers: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    max_retries: int = 3
    retry_base_delay: float = 1.0

    # Monitoring
    enable_detailed_metrics: bool = True
    enable_distributed_tracing: bool = True
    log_level: str = "INFO"
    structured_logging: bool = True

    # Security
    enable_input_validation: bool = True
    max_message_size_mb: int = 10
    rate_limit_per_minute: int = 1000
    enable_audit_logging: bool = True

    # Resource Management
    max_concurrent_discoveries: int = 100
    worker_threads: int = 8
    gpu_memory_fraction: float = 0.5

    # Model Management
    enable_model_versioning: bool = True
    model_checkpoint_interval: int = 1000
    max_model_versions: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "performance": {
                "enable_caching": self.enable_caching,
                "cache_backend": self.cache_backend.value,
                "max_cache_size_mb": self.max_cache_size_mb,
                "cache_ttl_seconds": self.cache_ttl_seconds,
            },
            "reliability": {
                "enable_circuit_breakers": self.enable_circuit_breakers,
                "max_retries": self.max_retries,
            },
            "monitoring": {
                "enable_detailed_metrics": self.enable_detailed_metrics,
                "enable_distributed_tracing": self.enable_distributed_tracing,
                "log_level": self.log_level,
            },
            "security": {
                "enable_input_validation": self.enable_input_validation,
                "max_message_size_mb": self.max_message_size_mb,
                "rate_limit_per_minute": self.rate_limit_per_minute,
            },
            "resources": {
                "max_concurrent_discoveries": self.max_concurrent_discoveries,
                "worker_threads": self.worker_threads,
            },
        }


# ============================================================================
# HEALTH CHECK SYSTEM
# ============================================================================


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a component."""

    name: str
    status: HealthStatus
    message: str
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.component_checks: Dict[str, Callable] = {}

    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.component_checks[name] = check_func

    async def check_all(self) -> Dict[str, ComponentHealth]:
        """Run all health checks."""
        results = {}

        for name, check_func in self.component_checks.items():
            try:
                start_time = time.time()
                status, message, metadata = await check_func()
                latency_ms = (time.time() - start_time) * 1000

                results[name] = ComponentHealth(
                    name=name,
                    status=status,
                    message=message,
                    latency_ms=latency_ms,
                    metadata=metadata,
                )
            except Exception as e:
                results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}",
                    metadata={"error": str(e)},
                )

        return results

    async def is_healthy(self) -> bool:
        """Check if system is healthy."""
        results = await self.check_all()
        return all(
            health.status != HealthStatus.UNHEALTHY for health in results.values()
        )

    async def is_ready(self) -> bool:
        """Check if system is ready to serve requests."""
        results = await self.check_all()
        critical_components = ["cache", "models"]

        return all(
            results.get(comp, ComponentHealth("", HealthStatus.UNHEALTHY, "")).status
            != HealthStatus.UNHEALTHY
            for comp in critical_components
            if comp in results
        )


# Export all production enhancements
__all__ = [
    "DiscoveryMetrics",
    "CacheBackend",
    "DistributedCache",
    "DiscoveryError",
    "StatisticalAnalysisError",
    "GrammarLearningError",
    "ParserGenerationError",
    "ClassificationError",
    "ValidationError",
    "ErrorRecoveryStrategy",
    "ProductionConfig",
    "HealthStatus",
    "ComponentHealth",
    "HealthChecker",
]
