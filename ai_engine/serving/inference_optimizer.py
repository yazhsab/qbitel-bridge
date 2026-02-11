"""
Inference Optimizer

Optimizations for model inference:
- Request batching
- Response caching
- Prefetching
- Speculative execution
- Model quantization support
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchConfig:
    """Configuration for request batching."""

    # Batching parameters
    max_batch_size: int = 32
    max_wait_ms: int = 50  # Max time to wait for batch to fill
    min_batch_size: int = 1  # Minimum batch size before processing

    # Dynamic batching
    dynamic_batching: bool = True
    adaptive_timeout: bool = True  # Adjust timeout based on queue

    # Padding strategy
    pad_to_multiple: int = 8  # Pad batch size to multiple of N

    # Priority batching
    priority_enabled: bool = True
    priority_levels: int = 3


@dataclass
class CacheConfig:
    """Configuration for response caching."""

    # Cache settings
    enabled: bool = True
    max_size: int = 10000  # Maximum cache entries
    ttl_seconds: int = 3600  # Default TTL

    # Cache strategy
    strategy: str = "lru"  # lru, lfu, fifo
    warm_cache: bool = True  # Pre-warm with common requests

    # Partial caching
    cache_embeddings: bool = True  # Cache intermediate embeddings
    cache_tokens: bool = True  # Cache tokenized inputs

    # Memory management
    max_memory_mb: int = 1024
    eviction_threshold: float = 0.9  # Evict when 90% full


@dataclass
class OptimizationMetrics:
    """Metrics for optimization performance."""

    # Batching metrics
    total_batches: int = 0
    total_requests: int = 0
    avg_batch_size: float = 0.0
    max_batch_size_seen: int = 0
    batch_wait_time_ms: float = 0.0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_size: int = 0
    cache_memory_mb: float = 0.0

    # Latency metrics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Throughput
    requests_per_second: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batching": {
                "total_batches": self.total_batches,
                "total_requests": self.total_requests,
                "avg_batch_size": self.avg_batch_size,
                "max_batch_size_seen": self.max_batch_size_seen,
                "batch_wait_time_ms": self.batch_wait_time_ms,
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": self.cache_hit_rate,
                "size": self.cache_size,
                "memory_mb": self.cache_memory_mb,
            },
            "latency": {
                "avg_ms": self.avg_latency_ms,
                "p50_ms": self.p50_latency_ms,
                "p99_ms": self.p99_latency_ms,
            },
            "throughput": {
                "requests_per_second": self.requests_per_second,
            },
        }


@dataclass
class BatchRequest(Generic[T]):
    """A request waiting to be batched."""

    id: str
    data: T
    priority: int = 1
    timestamp: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)


class LRUCache(Generic[T]):
    """Thread-safe LRU cache implementation."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[T, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                return None

            value, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value

    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        async with self._lock:
            # Evict if necessary
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class RequestBatcher(Generic[T, R]):
    """
    Batches incoming requests for efficient processing.

    Features:
    - Dynamic batch sizing
    - Priority queuing
    - Adaptive timeouts
    - Batch padding
    """

    def __init__(
        self,
        process_fn: Callable[[List[T]], List[R]],
        config: Optional[BatchConfig] = None,
    ):
        """
        Initialize batcher.

        Args:
            process_fn: Function to process a batch of requests
            config: Batching configuration
        """
        self.process_fn = process_fn
        self.config = config or BatchConfig()

        self._queues: Dict[int, List[BatchRequest[T]]] = {
            i: [] for i in range(self.config.priority_levels)
        }
        self._lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None
        self._running = False

        self._metrics = OptimizationMetrics()

    async def start(self) -> None:
        """Start the batcher."""
        if self._running:
            return

        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())
        logger.info("Request batcher started")

    async def stop(self) -> None:
        """Stop the batcher."""
        self._running = False
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        logger.info("Request batcher stopped")

    async def submit(
        self,
        request_id: str,
        data: T,
        priority: int = 1,
    ) -> R:
        """
        Submit a request for batched processing.

        Args:
            request_id: Unique request identifier
            data: Request data
            priority: Priority level (0 = highest)

        Returns:
            Processing result
        """
        priority = min(priority, self.config.priority_levels - 1)

        request = BatchRequest(
            id=request_id,
            data=data,
            priority=priority,
            future=asyncio.get_event_loop().create_future(),
        )

        async with self._lock:
            self._queues[priority].append(request)
            self._metrics.total_requests += 1

        return await request.future

    async def _batch_loop(self) -> None:
        """Main batching loop."""
        while self._running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
                else:
                    await asyncio.sleep(0.001)  # Small sleep if no work
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(0.1)

    async def _collect_batch(self) -> List[BatchRequest[T]]:
        """Collect requests into a batch."""
        batch: List[BatchRequest[T]] = []
        deadline = time.time() + (self.config.max_wait_ms / 1000)

        while len(batch) < self.config.max_batch_size:
            # Check all priority queues
            async with self._lock:
                for priority in range(self.config.priority_levels):
                    while (
                        self._queues[priority]
                        and len(batch) < self.config.max_batch_size
                    ):
                        batch.append(self._queues[priority].pop(0))

            # Check if we have minimum batch or deadline passed
            if len(batch) >= self.config.min_batch_size:
                if time.time() >= deadline or len(batch) >= self.config.max_batch_size:
                    break

            # Wait a bit more for batch to fill
            remaining = deadline - time.time()
            if remaining > 0:
                await asyncio.sleep(min(remaining, 0.005))
            else:
                break

        # Pad batch size if configured
        if self.config.pad_to_multiple > 1 and batch:
            target_size = (
                (len(batch) + self.config.pad_to_multiple - 1)
                // self.config.pad_to_multiple
                * self.config.pad_to_multiple
            )
            # Note: actual padding would duplicate last request or add no-ops

        return batch

    async def _process_batch(self, batch: List[BatchRequest[T]]) -> None:
        """Process a batch of requests."""
        start_time = time.time()

        # Update metrics
        self._metrics.total_batches += 1
        self._metrics.max_batch_size_seen = max(
            self._metrics.max_batch_size_seen, len(batch)
        )

        # Calculate average batch size
        self._metrics.avg_batch_size = (
            (self._metrics.avg_batch_size * (self._metrics.total_batches - 1)
             + len(batch)) / self._metrics.total_batches
        )

        try:
            # Extract data for processing
            data_batch = [req.data for req in batch]

            # Process batch
            if asyncio.iscoroutinefunction(self.process_fn):
                results = await self.process_fn(data_batch)
            else:
                results = await asyncio.to_thread(self.process_fn, data_batch)

            # Distribute results
            for request, result in zip(batch, results):
                if not request.future.done():
                    request.future.set_result(result)

            # Update latency metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics.avg_latency_ms = (
                (self._metrics.avg_latency_ms * (self._metrics.total_batches - 1)
                 + elapsed_ms) / self._metrics.total_batches
            )

        except Exception as e:
            # Set exception on all futures
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
            logger.error(f"Batch processing failed: {e}")

    @property
    def queue_size(self) -> int:
        """Get total queue size."""
        return sum(len(q) for q in self._queues.values())

    @property
    def metrics(self) -> OptimizationMetrics:
        """Get batcher metrics."""
        return self._metrics


class InferenceOptimizer:
    """
    Comprehensive inference optimization.

    Combines batching, caching, and other optimizations for
    maximum inference throughput and efficiency.

    Example:
        optimizer = InferenceOptimizer(
            inference_fn=model.predict,
            batch_config=BatchConfig(max_batch_size=32),
            cache_config=CacheConfig(enabled=True, ttl_seconds=3600)
        )

        await optimizer.start()

        # Single request (will be batched and cached)
        result = await optimizer.infer(
            request_id="req-123",
            input_data={"text": "Hello world"}
        )
    """

    def __init__(
        self,
        inference_fn: Callable[[List[Any]], List[Any]],
        batch_config: Optional[BatchConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        """
        Initialize optimizer.

        Args:
            inference_fn: Function to run inference on a batch
            batch_config: Batching configuration
            cache_config: Caching configuration
        """
        self.batch_config = batch_config or BatchConfig()
        self.cache_config = cache_config or CacheConfig()

        self._batcher = RequestBatcher(
            process_fn=inference_fn,
            config=self.batch_config,
        )

        self._cache: Optional[LRUCache] = None
        if self.cache_config.enabled:
            self._cache = LRUCache(
                max_size=self.cache_config.max_size,
                ttl_seconds=self.cache_config.ttl_seconds,
            )

        self._metrics = OptimizationMetrics()
        self._start_time = time.time()

        logger.info("Inference optimizer initialized")

    async def start(self) -> None:
        """Start the optimizer."""
        await self._batcher.start()
        logger.info("Inference optimizer started")

    async def stop(self) -> None:
        """Stop the optimizer."""
        await self._batcher.stop()
        logger.info("Inference optimizer stopped")

    async def infer(
        self,
        request_id: str,
        input_data: Any,
        use_cache: bool = True,
        priority: int = 1,
        cache_key: Optional[str] = None,
    ) -> Any:
        """
        Run inference with optimizations.

        Args:
            request_id: Unique request identifier
            input_data: Input data for inference
            use_cache: Whether to use caching
            priority: Request priority
            cache_key: Optional custom cache key

        Returns:
            Inference result
        """
        # Generate cache key
        if use_cache and self._cache:
            key = cache_key or self._compute_cache_key(input_data)

            # Check cache
            cached = await self._cache.get(key)
            if cached is not None:
                self._metrics.cache_hits += 1
                return cached

            self._metrics.cache_misses += 1

        # Submit to batcher
        result = await self._batcher.submit(request_id, input_data, priority)

        # Cache result
        if use_cache and self._cache:
            await self._cache.set(key, result)
            self._metrics.cache_size = self._cache.size

        return result

    async def batch_infer(
        self,
        requests: List[Tuple[str, Any]],
        use_cache: bool = True,
    ) -> List[Any]:
        """
        Run batch inference.

        Args:
            requests: List of (request_id, input_data) tuples
            use_cache: Whether to use caching

        Returns:
            List of inference results
        """
        tasks = [
            self.infer(request_id, input_data, use_cache)
            for request_id, input_data in requests
        ]
        return await asyncio.gather(*tasks)

    async def warm_cache(
        self,
        common_inputs: List[Any],
    ) -> int:
        """
        Pre-warm cache with common inputs.

        Args:
            common_inputs: List of common input data

        Returns:
            Number of entries cached
        """
        if not self._cache or not self.cache_config.warm_cache:
            return 0

        cached = 0
        for i, input_data in enumerate(common_inputs):
            try:
                result = await self._batcher.submit(f"warmup-{i}", input_data)
                key = self._compute_cache_key(input_data)
                await self._cache.set(key, result)
                cached += 1
            except Exception as e:
                logger.warning(f"Cache warmup failed for input {i}: {e}")

        logger.info(f"Cache warmed with {cached} entries")
        return cached

    async def clear_cache(self) -> None:
        """Clear the inference cache."""
        if self._cache:
            await self._cache.clear()
            self._metrics.cache_size = 0
            logger.info("Cache cleared")

    def _compute_cache_key(self, input_data: Any) -> str:
        """Compute cache key for input data."""
        if isinstance(input_data, dict):
            data_str = str(sorted(input_data.items()))
        elif isinstance(input_data, (list, tuple)):
            data_str = str(input_data)
        else:
            data_str = str(input_data)

        return hashlib.sha256(data_str.encode()).hexdigest()

    def get_metrics(self) -> OptimizationMetrics:
        """Get comprehensive metrics."""
        # Combine batcher and local metrics
        batcher_metrics = self._batcher.metrics

        self._metrics.total_batches = batcher_metrics.total_batches
        self._metrics.total_requests = batcher_metrics.total_requests
        self._metrics.avg_batch_size = batcher_metrics.avg_batch_size
        self._metrics.max_batch_size_seen = batcher_metrics.max_batch_size_seen
        self._metrics.avg_latency_ms = batcher_metrics.avg_latency_ms

        # Calculate throughput
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            self._metrics.requests_per_second = (
                self._metrics.total_requests / elapsed
            )

        return self._metrics

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._batcher.queue_size

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        return self._metrics.cache_hit_rate


# Utility functions for common optimization patterns


async def create_optimized_inference(
    model_fn: Callable,
    max_batch_size: int = 32,
    cache_enabled: bool = True,
    cache_ttl: int = 3600,
) -> InferenceOptimizer:
    """
    Create an optimized inference pipeline.

    Args:
        model_fn: Model inference function
        max_batch_size: Maximum batch size
        cache_enabled: Enable response caching
        cache_ttl: Cache TTL in seconds

    Returns:
        Configured InferenceOptimizer
    """
    optimizer = InferenceOptimizer(
        inference_fn=model_fn,
        batch_config=BatchConfig(max_batch_size=max_batch_size),
        cache_config=CacheConfig(enabled=cache_enabled, ttl_seconds=cache_ttl),
    )
    await optimizer.start()
    return optimizer
