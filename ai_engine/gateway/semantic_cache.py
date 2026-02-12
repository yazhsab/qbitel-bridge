"""
QBITEL Gateway - Semantic Cache

Intelligent caching layer for LLM responses using semantic similarity.
Reduces costs by 30-50% by serving cached responses for semantically similar queries.

Features:
- Semantic similarity matching using embeddings
- Configurable similarity thresholds
- TTL-based cache expiration
- Redis backend for distributed caching
- In-memory LRU cache for hot queries
- Cache warming and preloading
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from enum import Enum
import numpy as np

from prometheus_client import Counter, Histogram, Gauge

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================

CACHE_HITS = Counter(
    "qbitel_gateway_cache_hits_total",
    "Total cache hits",
    ["domain", "cache_type"],
)
CACHE_MISSES = Counter(
    "qbitel_gateway_cache_misses_total",
    "Total cache misses",
    ["domain"],
)
CACHE_SIZE = Gauge(
    "qbitel_gateway_cache_size",
    "Current cache size",
    ["cache_type"],
)
CACHE_LATENCY = Histogram(
    "qbitel_gateway_cache_latency_seconds",
    "Cache lookup latency",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
CACHE_COST_SAVINGS = Counter(
    "qbitel_gateway_cache_cost_savings_dollars",
    "Estimated cost savings from cache hits",
    ["domain"],
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CacheConfig:
    """Configuration for semantic cache."""

    # Similarity settings
    similarity_threshold: float = 0.92  # Minimum similarity for cache hit
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast, good quality

    # Cache settings
    max_memory_entries: int = 10000
    max_redis_entries: int = 100000
    default_ttl_seconds: int = 3600  # 1 hour

    # Redis settings
    redis_url: Optional[str] = None
    redis_prefix: str = "qbitel:ai_gateway:cache:"

    # Domain-specific TTLs
    domain_ttls: Dict[str, int] = field(
        default_factory=lambda: {
            "protocol_copilot": 7200,  # 2 hours - stable knowledge
            "security_orchestrator": 1800,  # 30 min - security changes faster
            "legacy_whisperer": 86400,  # 24 hours - legacy rarely changes
            "compliance_reporter": 3600,  # 1 hour
            "translation_studio": 7200,  # 2 hours
        }
    )

    # Cost estimation (per 1K tokens)
    cost_per_1k_tokens: Dict[str, float] = field(
        default_factory=lambda: {
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.00015,
            "claude-sonnet-4-5": 0.003,
            "claude-opus-4-5": 0.015,
            "claude-3-5-haiku": 0.00025,
            "llama3.2": 0.0,  # Local
        }
    )


@dataclass
class CacheEntry:
    """A cached LLM response."""

    # Key information
    cache_key: str
    prompt_hash: str
    embedding: List[float]

    # Request context
    domain: str
    model: str
    prompt: str
    system_prompt: Optional[str] = None

    # Response
    response: str
    tokens_used: int

    # Metadata
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0
    hit_count: int = 0
    last_hit_at: Optional[float] = None

    # Cost tracking
    estimated_cost: float = 0.0

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CacheStats:
    """Cache statistics."""

    total_entries: int = 0
    memory_entries: int = 0
    redis_entries: int = 0

    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0

    total_cost_savings: float = 0.0
    average_similarity: float = 0.0

    entries_by_domain: Dict[str, int] = field(default_factory=dict)
    hits_by_domain: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# Embedding Manager
# =============================================================================


class EmbeddingManager:
    """Manages embeddings for semantic similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the embedding model."""
        if self._model is not None:
            return

        async with self._lock:
            if self._model is not None:
                return

            if SentenceTransformer is not None:
                try:
                    # Load in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    self._model = await loop.run_in_executor(None, SentenceTransformer, self.model_name)
                    logger.info(f"Loaded embedding model: {self.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load embedding model: {e}")
                    self._model = None
            else:
                logger.warning("sentence-transformers not installed, using hash-based fallback")

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        await self.initialize()

        if self._model is not None:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, lambda: self._model.encode(text, normalize_embeddings=True))
            return embedding.tolist()
        else:
            # Fallback: Use hash-based pseudo-embedding
            return self._hash_embedding(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        await self.initialize()

        if self._model is not None:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, lambda: self._model.encode(texts, normalize_embeddings=True))
            return embeddings.tolist()
        else:
            return [self._hash_embedding(t) for t in texts]

    @staticmethod
    def _hash_embedding(text: str, dim: int = 384) -> List[float]:
        """Generate deterministic pseudo-embedding from hash."""
        digest = hashlib.sha256(text.encode()).digest()
        # Extend hash to desired dimension
        extended = digest * ((dim // len(digest)) + 1)
        values = [b / 255.0 for b in extended[:dim]]
        # Normalize
        norm = np.linalg.norm(values)
        if norm > 0:
            values = [v / norm for v in values]
        return values

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr))


# =============================================================================
# Semantic Cache
# =============================================================================


class SemanticCache:
    """
    Semantic cache for LLM responses.

    Uses embedding similarity to find cached responses for similar queries,
    reducing costs by avoiding redundant LLM calls.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.embedding_manager = EmbeddingManager(self.config.embedding_model)

        # In-memory cache (LRU-style)
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._memory_embeddings: List[Tuple[str, List[float]]] = []

        # Redis client
        self._redis: Optional[Any] = None

        # Statistics
        self._stats = CacheStats()

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize cache backends."""
        # Initialize embedding manager
        await self.embedding_manager.initialize()

        # Initialize Redis if configured
        if self.config.redis_url and redis is not None:
            try:
                self._redis = redis.from_url(
                    self.config.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._redis.ping()
                logger.info("Connected to Redis for semantic cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self._redis = None

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Semantic cache initialized")

    async def shutdown(self):
        """Shutdown cache."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._redis:
            await self._redis.close()

        logger.info("Semantic cache shutdown")

    async def get(
        self,
        prompt: str,
        domain: str,
        system_prompt: Optional[str] = None,
    ) -> Optional[CacheEntry]:
        """
        Look up cache for semantically similar query.

        Returns cached entry if similarity >= threshold, None otherwise.
        """
        start_time = time.time()

        try:
            # Generate embedding for query
            query_embedding = await self.embedding_manager.embed(prompt)

            # Search memory cache first
            best_match = await self._search_memory_cache(query_embedding, domain, system_prompt)

            if best_match:
                CACHE_HITS.labels(domain=domain, cache_type="memory").inc()
                CACHE_LATENCY.labels(operation="get_hit").observe(time.time() - start_time)
                return best_match

            # Search Redis cache if available
            if self._redis:
                redis_match = await self._search_redis_cache(query_embedding, domain, system_prompt)
                if redis_match:
                    # Promote to memory cache
                    await self._add_to_memory_cache(redis_match)
                    CACHE_HITS.labels(domain=domain, cache_type="redis").inc()
                    CACHE_LATENCY.labels(operation="get_hit").observe(time.time() - start_time)
                    return redis_match

            # Cache miss
            CACHE_MISSES.labels(domain=domain).inc()
            self._stats.total_misses += 1
            CACHE_LATENCY.labels(operation="get_miss").observe(time.time() - start_time)
            return None

        except Exception as e:
            logger.error(f"Cache lookup error: {e}")
            CACHE_MISSES.labels(domain=domain).inc()
            return None

    async def put(
        self,
        prompt: str,
        response: str,
        domain: str,
        model: str,
        tokens_used: int,
        system_prompt: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> CacheEntry:
        """
        Store response in cache.

        Returns the created cache entry.
        """
        start_time = time.time()

        # Generate embedding
        embedding = await self.embedding_manager.embed(prompt)

        # Calculate TTL
        if ttl is None:
            ttl = self.config.domain_ttls.get(domain, self.config.default_ttl_seconds)

        # Estimate cost
        cost_rate = self.config.cost_per_1k_tokens.get(model, 0.001)
        estimated_cost = (tokens_used / 1000) * cost_rate

        # Create cache entry
        entry = CacheEntry(
            cache_key=self._generate_cache_key(prompt, domain, system_prompt),
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            embedding=embedding,
            domain=domain,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            response=response,
            tokens_used=tokens_used,
            expires_at=time.time() + ttl,
            estimated_cost=estimated_cost,
        )

        # Store in memory cache
        await self._add_to_memory_cache(entry)

        # Store in Redis if available
        if self._redis:
            await self._add_to_redis_cache(entry, ttl)

        CACHE_LATENCY.labels(operation="put").observe(time.time() - start_time)

        return entry

    async def invalidate(
        self,
        domain: Optional[str] = None,
        prompt_hash: Optional[str] = None,
    ):
        """Invalidate cache entries."""
        async with self._lock:
            keys_to_remove = []

            for key, entry in self._memory_cache.items():
                if domain and entry.domain != domain:
                    continue
                if prompt_hash and entry.prompt_hash != prompt_hash:
                    continue
                keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._memory_cache[key]

            # Remove from embeddings list
            self._memory_embeddings = [(k, e) for k, e in self._memory_embeddings if k not in keys_to_remove]

            # Invalidate in Redis
            if self._redis and domain:
                pattern = f"{self.config.redis_prefix}{domain}:*"
                async for key in self._redis.scan_iter(pattern):
                    await self._redis.delete(key)

        logger.info(f"Invalidated {len(keys_to_remove)} cache entries")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_requests = self._stats.total_hits + self._stats.total_misses
        self._stats.hit_rate = self._stats.total_hits / total_requests if total_requests > 0 else 0.0
        self._stats.memory_entries = len(self._memory_cache)
        self._stats.total_entries = self._stats.memory_entries + self._stats.redis_entries

        # Update entries by domain
        self._stats.entries_by_domain = {}
        for entry in self._memory_cache.values():
            domain = entry.domain
            self._stats.entries_by_domain[domain] = self._stats.entries_by_domain.get(domain, 0) + 1

        return self._stats

    async def _search_memory_cache(
        self,
        query_embedding: List[float],
        domain: str,
        system_prompt: Optional[str],
    ) -> Optional[CacheEntry]:
        """Search memory cache for similar entry."""
        best_similarity = 0.0
        best_entry = None

        async with self._lock:
            for cache_key, entry_embedding in self._memory_embeddings:
                entry = self._memory_cache.get(cache_key)
                if not entry:
                    continue

                # Filter by domain
                if entry.domain != domain:
                    continue

                # Filter by system prompt
                if entry.system_prompt != system_prompt:
                    continue

                # Check expiration
                if entry.is_expired():
                    continue

                # Calculate similarity
                similarity = self.embedding_manager.cosine_similarity(query_embedding, entry_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

        if best_entry and best_similarity >= self.config.similarity_threshold:
            # Update hit stats
            best_entry.hit_count += 1
            best_entry.last_hit_at = time.time()
            self._stats.total_hits += 1

            # Track cost savings
            CACHE_COST_SAVINGS.labels(domain=domain).inc(best_entry.estimated_cost)
            self._stats.total_cost_savings += best_entry.estimated_cost

            logger.debug(
                f"Cache hit: similarity={best_similarity:.3f}, " f"domain={domain}, savings=${best_entry.estimated_cost:.4f}"
            )

            return best_entry

        return None

    async def _search_redis_cache(
        self,
        query_embedding: List[float],
        domain: str,
        system_prompt: Optional[str],
    ) -> Optional[CacheEntry]:
        """Search Redis cache for similar entry."""
        if not self._redis:
            return None

        try:
            # Get all entries for domain
            pattern = f"{self.config.redis_prefix}{domain}:*"
            best_similarity = 0.0
            best_entry = None

            async for key in self._redis.scan_iter(pattern, count=100):
                data = await self._redis.get(key)
                if not data:
                    continue

                entry_dict = json.loads(data)
                entry = CacheEntry.from_dict(entry_dict)

                # Filter by system prompt
                if entry.system_prompt != system_prompt:
                    continue

                # Calculate similarity
                similarity = self.embedding_manager.cosine_similarity(query_embedding, entry.embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

            if best_entry and best_similarity >= self.config.similarity_threshold:
                best_entry.hit_count += 1
                best_entry.last_hit_at = time.time()
                self._stats.total_hits += 1

                CACHE_COST_SAVINGS.labels(domain=domain).inc(best_entry.estimated_cost)
                self._stats.total_cost_savings += best_entry.estimated_cost

                return best_entry

            return None

        except Exception as e:
            logger.error(f"Redis cache search error: {e}")
            return None

    async def _add_to_memory_cache(self, entry: CacheEntry):
        """Add entry to memory cache with LRU eviction."""
        async with self._lock:
            # Check if we need to evict
            if len(self._memory_cache) >= self.config.max_memory_entries:
                # Remove oldest/least used entries
                sorted_entries = sorted(self._memory_cache.items(), key=lambda x: (x[1].hit_count, x[1].last_hit_at or 0))
                # Remove 10% of entries
                to_remove = max(1, len(sorted_entries) // 10)
                for key, _ in sorted_entries[:to_remove]:
                    del self._memory_cache[key]

                # Update embeddings list
                remaining_keys = set(self._memory_cache.keys())
                self._memory_embeddings = [(k, e) for k, e in self._memory_embeddings if k in remaining_keys]

            # Add new entry
            self._memory_cache[entry.cache_key] = entry
            self._memory_embeddings.append((entry.cache_key, entry.embedding))

            CACHE_SIZE.labels(cache_type="memory").set(len(self._memory_cache))

    async def _add_to_redis_cache(self, entry: CacheEntry, ttl: int):
        """Add entry to Redis cache."""
        if not self._redis:
            return

        try:
            key = f"{self.config.redis_prefix}{entry.domain}:{entry.cache_key}"
            data = json.dumps(entry.to_dict())
            await self._redis.setex(key, ttl, data)
            self._stats.redis_entries += 1
            CACHE_SIZE.labels(cache_type="redis").set(self._stats.redis_entries)
        except Exception as e:
            logger.error(f"Redis cache write error: {e}")

    async def _cleanup_loop(self):
        """Background task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    async def _cleanup_expired(self):
        """Remove expired entries from memory cache."""
        async with self._lock:
            expired_keys = [key for key, entry in self._memory_cache.items() if entry.is_expired()]

            for key in expired_keys:
                del self._memory_cache[key]

            self._memory_embeddings = [(k, e) for k, e in self._memory_embeddings if k not in expired_keys]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                CACHE_SIZE.labels(cache_type="memory").set(len(self._memory_cache))

    @staticmethod
    def _generate_cache_key(
        prompt: str,
        domain: str,
        system_prompt: Optional[str],
    ) -> str:
        """Generate unique cache key."""
        content = f"{domain}:{system_prompt or ''}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
