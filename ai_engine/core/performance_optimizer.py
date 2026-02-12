"""
Performance optimization module for QBITEL Bridge.
Provides advanced caching, memory management, and computational optimizations.
"""

import asyncio
import time
import threading
from typing import Dict, Any, Optional, Callable, Tuple, List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from collections import OrderedDict
import weakref
import gc
import psutil
import numpy as np
from functools import wraps, lru_cache
import hashlib
import pickle
import logging
import redis
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Metrics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage: int = 0
    avg_access_time: float = 0.0
    hit_rate: float = 0.0


@dataclass
class PerformanceMetrics:
    """Overall performance metrics."""

    total_requests: int = 0
    avg_response_time: float = 0.0
    throughput_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_threads: int = 0
    cache_metrics: Dict[str, CacheMetrics] = field(default_factory=dict)


class LRUCacheWithTTL:
    """High-performance LRU cache with TTL and memory management."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
        self._metrics = CacheMetrics()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            current_time = time.time()

            # Check if key exists and not expired
            if key in self._cache:
                if current_time - self._timestamps[key] < self.ttl:
                    # Move to end (most recently used)
                    value = self._cache.pop(key)
                    self._cache[key] = value
                    self._metrics.hits += 1
                    return value
                else:
                    # Expired, remove
                    del self._cache[key]
                    del self._timestamps[key]
                    self._metrics.evictions += 1

            self._metrics.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            current_time = time.time()

            if key in self._cache:
                # Update existing
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                # Evict oldest
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
                self._metrics.evictions += 1

            self._cache[key] = value
            self._timestamps[key] = current_time

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._metrics.evictions += len(self._cache)

    def size(self) -> int:
        return len(self._cache)

    def get_metrics(self) -> CacheMetrics:
        with self._lock:
            total_requests = self._metrics.hits + self._metrics.misses
            self._metrics.hit_rate = self._metrics.hits / total_requests if total_requests > 0 else 0.0
            self._metrics.memory_usage = len(pickle.dumps(self._cache))
            return self._metrics


class RedisCache:
    """Redis-based distributed cache for cluster deployments."""

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl
        self._metrics = CacheMetrics()

    async def get(self, key: str) -> Optional[Any]:
        try:
            start_time = time.time()
            data = self.redis_client.get(key)
            access_time = time.time() - start_time

            if data:
                self._metrics.hits += 1
                self._metrics.avg_access_time = (
                    self._metrics.avg_access_time * (self._metrics.hits - 1) + access_time
                ) / self._metrics.hits
                return pickle.loads(data.encode())
            else:
                self._metrics.misses += 1
                return None
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            self._metrics.misses += 1
            return None

    async def put(self, key: str, value: Any) -> None:
        try:
            data = pickle.dumps(value)
            self.redis_client.setex(key, self.ttl, data)
        except Exception as e:
            logger.error(f"Redis cache put error: {e}")

    def get_metrics(self) -> CacheMetrics:
        info = self.redis_client.info()
        self._metrics.memory_usage = info.get("used_memory", 0)
        total_requests = self._metrics.hits + self._metrics.misses
        self._metrics.hit_rate = self._metrics.hits / total_requests if total_requests > 0 else 0.0
        return self._metrics


class ComputationCache:
    """High-performance cache for expensive computations."""

    def __init__(self, max_size: int = 1000, use_redis: bool = False, redis_url: str = None):
        self.local_cache = LRUCacheWithTTL(max_size)
        self.redis_cache = RedisCache(redis_url) if use_redis and redis_url else None

    def cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function call."""
        key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    async def get_or_compute(self, key: str, compute_func: Callable, *args, **kwargs) -> Any:
        """Get from cache or compute and store."""
        # Try local cache first
        result = self.local_cache.get(key)
        if result is not None:
            return result

        # Try Redis cache if available
        if self.redis_cache:
            result = await self.redis_cache.get(key)
            if result is not None:
                # Store in local cache for faster access
                self.local_cache.put(key, result)
                return result

        # Compute and cache
        if asyncio.iscoroutinefunction(compute_func):
            result = await compute_func(*args, **kwargs)
        else:
            result = compute_func(*args, **kwargs)

        # Store in both caches
        self.local_cache.put(key, result)
        if self.redis_cache:
            await self.redis_cache.put(key, result)

        return result


class MemoryPool:
    """Memory pool for efficient buffer management."""

    def __init__(self, buffer_size: int = 8192, pool_size: int = 100):
        self.buffer_size = buffer_size
        self.pool = []
        self.in_use = set()
        self._lock = threading.Lock()

        # Pre-allocate buffers
        for _ in range(pool_size):
            self.pool.append(bytearray(buffer_size))

    def get_buffer(self) -> bytearray:
        """Get a buffer from the pool."""
        with self._lock:
            if self.pool:
                buffer = self.pool.pop()
                self.in_use.add(id(buffer))
                return buffer
            else:
                # Pool exhausted, create new buffer
                buffer = bytearray(self.buffer_size)
                self.in_use.add(id(buffer))
                return buffer

    def return_buffer(self, buffer: bytearray) -> None:
        """Return a buffer to the pool."""
        with self._lock:
            buffer_id = id(buffer)
            if buffer_id in self.in_use:
                self.in_use.remove(buffer_id)
                # Clear buffer and return to pool
                buffer[:] = b"\x00" * len(buffer)
                self.pool.append(buffer)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "pool_size": len(self.pool),
                "in_use": len(self.in_use),
                "buffer_size": self.buffer_size,
            }


class ThreadPoolManager:
    """Optimized thread pool management for concurrent operations."""

    def __init__(self, max_threads: int = None):
        self.max_threads = max_threads or min(32, (psutil.cpu_count() or 1) * 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=psutil.cpu_count())
        self._active_tasks = 0
        self._completed_tasks = 0

    async def run_cpu_bound(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-bound task in process pool."""
        loop = asyncio.get_event_loop()
        self._active_tasks += 1
        try:
            result = await loop.run_in_executor(self.process_pool, func, *args, **kwargs)
            self._completed_tasks += 1
            return result
        finally:
            self._active_tasks -= 1

    async def run_io_bound(self, func: Callable, *args, **kwargs) -> Any:
        """Run I/O-bound task in thread pool."""
        loop = asyncio.get_event_loop()
        self._active_tasks += 1
        try:
            result = await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
            self._completed_tasks += 1
            return result
        finally:
            self._active_tasks -= 1

    def get_stats(self) -> dict:
        return {
            "max_threads": self.max_threads,
            "active_tasks": self._active_tasks,
            "completed_tasks": self._completed_tasks,
            "thread_pool_size": (self.thread_pool._threads.__len__() if hasattr(self.thread_pool, "_threads") else 0),
        }

    def shutdown(self):
        """Shutdown thread pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class BatchProcessor:
    """Efficient batch processing for high-throughput operations."""

    def __init__(self, batch_size: int = 1000, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._batch = []
        self._last_flush = time.time()
        self._lock = asyncio.Lock()
        self._processors = {}

    async def add_item(self, processor_id: str, item: Any, process_func: Callable):
        """Add item to batch for processing."""
        async with self._lock:
            if processor_id not in self._processors:
                self._processors[processor_id] = {
                    "batch": [],
                    "func": process_func,
                    "last_flush": time.time(),
                }

            processor = self._processors[processor_id]
            processor["batch"].append(item)

            # Check if batch is ready for processing
            current_time = time.time()
            should_flush = (
                len(processor["batch"]) >= self.batch_size or current_time - processor["last_flush"] >= self.flush_interval
            )

            if should_flush:
                await self._flush_batch(processor_id)

    async def _flush_batch(self, processor_id: str):
        """Flush batch for processing."""
        if processor_id not in self._processors:
            return

        processor = self._processors[processor_id]
        if not processor["batch"]:
            return

        batch = processor["batch"]
        processor["batch"] = []
        processor["last_flush"] = time.time()

        # Process batch
        try:
            if asyncio.iscoroutinefunction(processor["func"]):
                await processor["func"](batch)
            else:
                processor["func"](batch)
        except Exception as e:
            logger.error(f"Batch processing error for {processor_id}: {e}")

    async def flush_all(self):
        """Flush all pending batches."""
        async with self._lock:
            for processor_id in list(self._processors.keys()):
                await self._flush_batch(processor_id)


class PerformanceOptimizer:
    """Main performance optimization coordinator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Initialize components
        self.computation_cache = ComputationCache(
            max_size=config.get("cache_size", 10000),
            use_redis=config.get("use_redis", False),
            redis_url=config.get("redis_url"),
        )

        self.memory_pool = MemoryPool(
            buffer_size=config.get("buffer_size", 8192),
            pool_size=config.get("pool_size", 100),
        )

        self.thread_manager = ThreadPoolManager(max_threads=config.get("max_threads"))

        self.batch_processor = BatchProcessor(
            batch_size=config.get("batch_size", 1000),
            flush_interval=config.get("flush_interval", 1.0),
        )

        # Performance metrics
        self._metrics = PerformanceMetrics()
        self._start_time = time.time()
        self._request_times = []

    def cached_computation(self, cache_key: Optional[str] = None):
        """Decorator for caching expensive computations."""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key = cache_key or self.computation_cache.cache_key(func.__name__, args, kwargs)

                # Get or compute
                return await self.computation_cache.get_or_compute(key, func, *args, **kwargs)

            return wrapper

        return decorator

    @asynccontextmanager
    async def performance_context(self):
        """Context manager for performance tracking."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._request_times.append(duration)
            self._metrics.total_requests += 1

            # Update metrics
            if len(self._request_times) > 1000:
                self._request_times = self._request_times[-1000:]  # Keep last 1000

            self._metrics.avg_response_time = np.mean(self._request_times)
            self._metrics.throughput_per_sec = self._metrics.total_requests / (time.time() - self._start_time)

    def get_system_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        # System metrics
        process = psutil.Process()
        self._metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        self._metrics.cpu_usage_percent = process.cpu_percent()
        self._metrics.active_threads = process.num_threads()

        # Cache metrics
        self._metrics.cache_metrics = {
            "computation": self.computation_cache.local_cache.get_metrics(),
            "memory_pool": self.memory_pool.get_stats(),
            "thread_pool": self.thread_manager.get_stats(),
        }

        return self._metrics

    async def optimize_memory(self):
        """Perform memory optimization."""
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")

        # Clear caches if memory usage is high
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        if memory_mb > 1000:  # > 1GB
            logger.warning(f"High memory usage: {memory_mb:.2f}MB, clearing caches")
            self.computation_cache.local_cache.clear()

            # Request Redis cache cleanup if available
            if self.computation_cache.redis_cache:
                try:
                    self.computation_cache.redis_cache.redis_client.flushdb()
                except Exception as e:
                    logger.error(f"Redis cache clear error: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        metrics = self.get_system_metrics()

        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time,
            "performance": {
                "avg_response_time_ms": metrics.avg_response_time * 1000,
                "throughput_per_sec": metrics.throughput_per_sec,
                "total_requests": metrics.total_requests,
            },
            "resources": {
                "memory_usage_mb": metrics.memory_usage_mb,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "active_threads": metrics.active_threads,
            },
            "caches": {
                name: {
                    "hit_rate": cache.hit_rate if hasattr(cache, "hit_rate") else 0.0,
                    "size": (
                        cache.get("pool_size", cache.get("hits", 0)) if isinstance(cache, dict) else getattr(cache, "hits", 0)
                    ),
                }
                for name, cache in metrics.cache_metrics.items()
            },
        }

        # Determine health status
        if metrics.cpu_usage_percent > 90:
            health_status["status"] = "degraded"
            health_status["warnings"] = ["High CPU usage"]
        elif metrics.memory_usage_mb > 2000:  # > 2GB
            health_status["status"] = "degraded"
            health_status["warnings"] = ["High memory usage"]
        elif metrics.avg_response_time > 1.0:  # > 1 second
            health_status["status"] = "degraded"
            health_status["warnings"] = ["High response times"]

        return health_status

    def shutdown(self):
        """Shutdown performance optimizer."""
        logger.info("Shutting down performance optimizer")
        self.thread_manager.shutdown()


# Global performance optimizer instance
_performance_optimizer = None


def get_performance_optimizer(
    config: Optional[Dict[str, Any]] = None,
) -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(config)
    return _performance_optimizer


def shutdown_performance_optimizer():
    """Shutdown global performance optimizer."""
    global _performance_optimizer
    if _performance_optimizer:
        _performance_optimizer.shutdown()
        _performance_optimizer = None


# Convenience decorators
def cached(cache_key: Optional[str] = None):
    """Decorator for caching function results."""
    optimizer = get_performance_optimizer()
    return optimizer.cached_computation(cache_key)


async def with_performance_tracking(func: Callable, *args, **kwargs):
    """Execute function with performance tracking."""
    optimizer = get_performance_optimizer()
    async with optimizer.performance_context():
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
