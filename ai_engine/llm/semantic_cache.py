"""
QBITEL - Semantic Caching for LLM Responses
Intelligent caching that matches semantically similar queries.

Features:
- Semantic similarity-based cache lookup
- Configurable similarity thresholds
- TTL-based expiration
- Cost savings tracking
- Cache analytics
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError:
    _SentenceTransformer = None


class CacheHitType(Enum):
    """Type of cache hit."""

    EXACT = "exact"  # Exact query match
    SEMANTIC = "semantic"  # Semantically similar query
    MISS = "miss"  # No cache hit


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    query: str
    query_embedding: List[float]
    response: str
    response_metadata: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = None
    tokens_saved: int = 0
    cost_saved: float = 0.0
    feature_domain: str = ""


@dataclass
class CacheHitResult:
    """Result of a cache lookup."""

    hit_type: CacheHitType
    entry: Optional[CacheEntry]
    similarity_score: float = 0.0
    lookup_time: float = 0.0


@dataclass
class CacheStats:
    """Cache statistics."""

    total_queries: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    total_tokens_saved: int = 0
    total_cost_saved: float = 0.0
    avg_similarity_score: float = 0.0
    cache_size: int = 0
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None


class _FallbackEmbedder:
    """Fallback embedder using hash-based pseudo-embeddings."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return np.array([self._hash_embed(text) for text in texts])

    def _hash_embed(self, text: str) -> List[float]:
        """Generate deterministic pseudo-embedding from text hash."""
        digest = hashlib.sha256(text.lower().encode()).digest()
        floats = [byte / 255.0 for byte in digest]
        repeats = (self.dimension + len(floats) - 1) // len(floats)
        return (floats * repeats)[: self.dimension]


class SemanticCache:
    """
    Semantic cache for LLM responses using embedding similarity.

    Features:
    - Exact match lookup (fast, hash-based)
    - Semantic similarity lookup (embedding-based)
    - Configurable similarity threshold
    - TTL-based expiration
    - Cost and token savings tracking
    - Cache eviction policies
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.92,
        default_ttl_seconds: int = 3600,
        max_cache_size: int = 10000,
        enable_semantic_lookup: bool = True,
        eviction_policy: str = "lru",  # "lru", "lfu", "fifo"
    ):
        self.logger = logging.getLogger(__name__)
        self.similarity_threshold = similarity_threshold
        self.default_ttl = timedelta(seconds=default_ttl_seconds)
        self.max_cache_size = max_cache_size
        self.enable_semantic_lookup = enable_semantic_lookup
        self.eviction_policy = eviction_policy

        # Initialize embedding model
        if _SentenceTransformer is not None:
            try:
                self.embedder = _SentenceTransformer(embedding_model)
                self.logger.info(f"Semantic cache using model: {embedding_model}")
            except Exception as e:
                self.logger.warning(f"Failed to load embedding model: {e}, using fallback")
                self.embedder = _FallbackEmbedder()
        else:
            self.logger.info("SentenceTransformers not available, using fallback embedder")
            self.embedder = _FallbackEmbedder()

        # Cache storage
        self._exact_cache: Dict[str, CacheEntry] = {}  # hash -> entry
        self._semantic_cache: List[CacheEntry] = []  # list for similarity search
        self._embedding_matrix: Optional[np.ndarray] = None

        # Statistics
        self._stats = CacheStats()

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    def _hash_query(self, query: str, feature_domain: str = "") -> str:
        """Generate hash for exact match lookup."""
        combined = f"{feature_domain}:{query}".lower().strip()
        return hashlib.sha256(combined.encode()).hexdigest()

    async def get(
        self,
        query: str,
        feature_domain: str = "",
        similarity_threshold: Optional[float] = None,
    ) -> CacheHitResult:
        """
        Look up query in cache.

        Args:
            query: Query string to look up
            feature_domain: Feature domain for scoped caching
            similarity_threshold: Override default similarity threshold

        Returns:
            CacheHitResult with hit type and cached entry if found
        """
        start_time = time.time()
        threshold = similarity_threshold or self.similarity_threshold

        async with self._lock:
            self._stats.total_queries += 1

            # Step 1: Try exact match
            query_hash = self._hash_query(query, feature_domain)
            if query_hash in self._exact_cache:
                entry = self._exact_cache[query_hash]

                # Check expiration
                if entry.expires_at > datetime.now():
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    self._stats.exact_hits += 1

                    return CacheHitResult(
                        hit_type=CacheHitType.EXACT, entry=entry, similarity_score=1.0, lookup_time=time.time() - start_time
                    )
                else:
                    # Expired, remove from cache
                    del self._exact_cache[query_hash]

            # Step 2: Try semantic match if enabled
            if self.enable_semantic_lookup and self._semantic_cache:
                query_embedding = self.embedder.encode([query])[0]

                best_match = None
                best_score = 0.0

                for entry in self._semantic_cache:
                    if entry.expires_at <= datetime.now():
                        continue

                    # Filter by domain if specified
                    if feature_domain and entry.feature_domain != feature_domain:
                        continue

                    score = self._cosine_similarity(query_embedding, np.array(entry.query_embedding))

                    if score > best_score and score >= threshold:
                        best_score = score
                        best_match = entry

                if best_match is not None:
                    best_match.access_count += 1
                    best_match.last_accessed = datetime.now()
                    self._stats.semantic_hits += 1
                    self._update_avg_similarity(best_score)

                    return CacheHitResult(
                        hit_type=CacheHitType.SEMANTIC,
                        entry=best_match,
                        similarity_score=best_score,
                        lookup_time=time.time() - start_time,
                    )

            # No cache hit
            self._stats.misses += 1
            return CacheHitResult(
                hit_type=CacheHitType.MISS, entry=None, similarity_score=0.0, lookup_time=time.time() - start_time
            )

    async def put(
        self,
        query: str,
        response: str,
        response_metadata: Optional[Dict[str, Any]] = None,
        feature_domain: str = "",
        ttl_seconds: Optional[int] = None,
        tokens_used: int = 0,
        cost: float = 0.0,
    ) -> CacheEntry:
        """
        Add a new entry to the cache.

        Args:
            query: Original query string
            response: LLM response to cache
            response_metadata: Additional metadata about the response
            feature_domain: Feature domain for scoped caching
            ttl_seconds: Time-to-live override
            tokens_used: Number of tokens used (for savings tracking)
            cost: Cost of the LLM call (for savings tracking)

        Returns:
            Created CacheEntry
        """
        async with self._lock:
            # Evict if at capacity
            if len(self._semantic_cache) >= self.max_cache_size:
                await self._evict()

            # Generate embedding
            query_embedding = self.embedder.encode([query])[0].tolist()

            # Create entry
            now = datetime.now()
            ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else self.default_ttl

            entry = CacheEntry(
                query=query,
                query_embedding=query_embedding,
                response=response,
                response_metadata=response_metadata or {},
                created_at=now,
                expires_at=now + ttl,
                last_accessed=now,
                tokens_saved=tokens_used,
                cost_saved=cost,
                feature_domain=feature_domain,
            )

            # Add to exact cache
            query_hash = self._hash_query(query, feature_domain)
            self._exact_cache[query_hash] = entry

            # Add to semantic cache
            self._semantic_cache.append(entry)

            # Update stats
            self._stats.cache_size = len(self._semantic_cache)
            if self._stats.oldest_entry is None:
                self._stats.oldest_entry = now
            self._stats.newest_entry = now

            return entry

    async def invalidate(
        self,
        query: Optional[str] = None,
        feature_domain: Optional[str] = None,
        older_than: Optional[datetime] = None,
    ) -> int:
        """
        Invalidate cache entries matching criteria.

        Args:
            query: Specific query to invalidate (exact match)
            feature_domain: Invalidate all entries for this domain
            older_than: Invalidate entries created before this time

        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            count = 0

            if query is not None:
                # Invalidate specific query
                query_hash = self._hash_query(query, feature_domain or "")
                if query_hash in self._exact_cache:
                    del self._exact_cache[query_hash]
                    count += 1

                # Remove from semantic cache
                self._semantic_cache = [
                    e
                    for e in self._semantic_cache
                    if e.query != query or (feature_domain and e.feature_domain != feature_domain)
                ]

            elif feature_domain is not None:
                # Invalidate all entries for domain
                before_count = len(self._semantic_cache)
                self._semantic_cache = [e for e in self._semantic_cache if e.feature_domain != feature_domain]
                count = before_count - len(self._semantic_cache)

                # Clean exact cache
                hashes_to_remove = [h for h, e in self._exact_cache.items() if e.feature_domain == feature_domain]
                for h in hashes_to_remove:
                    del self._exact_cache[h]

            elif older_than is not None:
                # Invalidate old entries
                before_count = len(self._semantic_cache)
                self._semantic_cache = [e for e in self._semantic_cache if e.created_at >= older_than]
                count = before_count - len(self._semantic_cache)

                # Clean exact cache
                hashes_to_remove = [h for h, e in self._exact_cache.items() if e.created_at < older_than]
                for h in hashes_to_remove:
                    del self._exact_cache[h]

            self._stats.cache_size = len(self._semantic_cache)
            return count

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._exact_cache.clear()
            self._semantic_cache.clear()
            self._stats = CacheStats()

    async def cleanup_expired(self) -> int:
        """Remove expired entries from cache."""
        async with self._lock:
            now = datetime.now()

            # Clean semantic cache
            before_count = len(self._semantic_cache)
            self._semantic_cache = [e for e in self._semantic_cache if e.expires_at > now]
            removed = before_count - len(self._semantic_cache)

            # Clean exact cache
            expired_hashes = [h for h, e in self._exact_cache.items() if e.expires_at <= now]
            for h in expired_hashes:
                del self._exact_cache[h]

            self._stats.cache_size = len(self._semantic_cache)
            return removed

    async def _evict(self) -> None:
        """Evict entries based on eviction policy."""
        if not self._semantic_cache:
            return

        if self.eviction_policy == "lru":
            # Evict least recently used
            self._semantic_cache.sort(key=lambda e: e.last_accessed or e.created_at)
        elif self.eviction_policy == "lfu":
            # Evict least frequently used
            self._semantic_cache.sort(key=lambda e: e.access_count)
        elif self.eviction_policy == "fifo":
            # Evict oldest first
            self._semantic_cache.sort(key=lambda e: e.created_at)

        # Remove 10% of entries
        evict_count = max(1, len(self._semantic_cache) // 10)
        evicted = self._semantic_cache[:evict_count]
        self._semantic_cache = self._semantic_cache[evict_count:]

        # Clean exact cache
        evicted_queries = {(e.query, e.feature_domain) for e in evicted}
        self._exact_cache = {h: e for h, e in self._exact_cache.items() if (e.query, e.feature_domain) not in evicted_queries}

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        stats = CacheStats(
            total_queries=self._stats.total_queries,
            exact_hits=self._stats.exact_hits,
            semantic_hits=self._stats.semantic_hits,
            misses=self._stats.misses,
            total_tokens_saved=sum(e.tokens_saved * e.access_count for e in self._semantic_cache),
            total_cost_saved=sum(e.cost_saved * e.access_count for e in self._semantic_cache),
            avg_similarity_score=self._stats.avg_similarity_score,
            cache_size=len(self._semantic_cache),
            oldest_entry=self._stats.oldest_entry,
            newest_entry=self._stats.newest_entry,
        )
        return stats

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def _update_avg_similarity(self, new_score: float) -> None:
        """Update running average similarity score."""
        total_semantic = self._stats.semantic_hits
        if total_semantic == 0:
            self._stats.avg_similarity_score = new_score
        else:
            # Running average
            self._stats.avg_similarity_score = (
                self._stats.avg_similarity_score * (total_semantic - 1) + new_score
            ) / total_semantic


class CachedLLMService:
    """
    Wrapper that adds semantic caching to any LLM service.

    Usage:
        cached_service = CachedLLMService(llm_service, cache_config)
        response = await cached_service.process_request(request)
    """

    def __init__(
        self,
        llm_service,
        cache: Optional[SemanticCache] = None,
        enable_caching: bool = True,
        cache_domains: Optional[List[str]] = None,
        skip_cache_for_tools: bool = True,
    ):
        self.llm_service = llm_service
        self.cache = cache or SemanticCache()
        self.enable_caching = enable_caching
        self.cache_domains = set(cache_domains) if cache_domains else None
        self.skip_cache_for_tools = skip_cache_for_tools
        self.logger = logging.getLogger(__name__)

    async def process_request(self, request) -> Any:
        """
        Process LLM request with caching.

        Args:
            request: LLM request (LLMRequest or similar)

        Returns:
            LLM response (cached or fresh)
        """
        # Check if caching should be skipped
        if not self.enable_caching:
            return await self.llm_service.process_request(request)

        if self.skip_cache_for_tools and getattr(request, "tools", None):
            return await self.llm_service.process_request(request)

        domain = getattr(request, "feature_domain", "")
        if self.cache_domains and domain not in self.cache_domains:
            return await self.llm_service.process_request(request)

        # Try cache lookup
        query = getattr(request, "prompt", str(request))
        cache_result = await self.cache.get(query, feature_domain=domain)

        if cache_result.hit_type != CacheHitType.MISS:
            self.logger.debug(f"Cache {cache_result.hit_type.value} hit " f"(similarity: {cache_result.similarity_score:.3f})")

            # Reconstruct response from cache
            entry = cache_result.entry
            return self._reconstruct_response(entry)

        # Cache miss - call LLM
        response = await self.llm_service.process_request(request)

        # Cache the response
        tokens = getattr(response, "tokens_used", 0)
        cost = self._estimate_cost(response)

        await self.cache.put(
            query=query,
            response=getattr(response, "content", str(response)),
            response_metadata={
                "provider": getattr(response, "provider", "unknown"),
                "tokens_used": tokens,
                "confidence": getattr(response, "confidence", 0.0),
            },
            feature_domain=domain,
            tokens_used=tokens,
            cost=cost,
        )

        return response

    def _reconstruct_response(self, entry: CacheEntry) -> Any:
        """Reconstruct LLM response from cache entry."""
        # Import here to avoid circular imports
        try:
            from .unified_llm_service import LLMResponse

            return LLMResponse(
                content=entry.response,
                provider=entry.response_metadata.get("provider", "cache"),
                tokens_used=0,  # No tokens used for cached response
                processing_time=0.0,
                confidence=entry.response_metadata.get("confidence", 1.0),
                metadata={
                    "cached": True,
                    "cache_hit_type": "semantic",
                    "original_created_at": entry.created_at.isoformat(),
                },
            )
        except ImportError:
            # Return as dictionary if LLMResponse not available
            return {
                "content": entry.response,
                "provider": "cache",
                "cached": True,
                "metadata": entry.response_metadata,
            }

    def _estimate_cost(self, response) -> float:
        """Estimate cost of LLM response."""
        tokens = getattr(response, "tokens_used", 0)
        provider = getattr(response, "provider", "")

        # Approximate cost per 1K tokens
        cost_per_1k = {
            "openai_gpt4": 0.03,
            "anthropic_claude": 0.015,
            "ollama_local": 0.0,
        }

        rate = cost_per_1k.get(provider, 0.01)
        return (tokens / 1000) * rate

    async def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.cache.get_stats()

    async def clear_cache(self) -> None:
        """Clear the cache."""
        await self.cache.clear()


# Global cache instance
_global_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get or create global semantic cache instance."""
    global _global_semantic_cache
    if _global_semantic_cache is None:
        _global_semantic_cache = SemanticCache()
    return _global_semantic_cache


def configure_semantic_cache(**config) -> SemanticCache:
    """Configure and return global semantic cache."""
    global _global_semantic_cache
    _global_semantic_cache = SemanticCache(**config)
    return _global_semantic_cache
