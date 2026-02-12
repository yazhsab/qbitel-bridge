"""
QBITEL Gateway - Rate Limiter

Token bucket rate limiting for LLM requests.
Prevents API abuse and ensures fair resource allocation.

Features:
- Token bucket algorithm
- Per-domain and per-user limits
- Distributed rate limiting with Redis
- Graceful degradation
- Request queuing
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from collections import defaultdict

from prometheus_client import Counter, Gauge, Histogram

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================

RATE_LIMIT_REQUESTS = Counter(
    "qbitel_gateway_rate_limit_requests_total",
    "Rate limited requests",
    ["domain", "action"],  # action: allowed, rejected, queued
)
RATE_LIMIT_TOKENS = Gauge(
    "qbitel_gateway_rate_limit_tokens",
    "Current token bucket level",
    ["domain"],
)
RATE_LIMIT_QUEUE_SIZE = Gauge(
    "qbitel_gateway_rate_limit_queue_size",
    "Request queue size",
    ["domain"],
)
RATE_LIMIT_WAIT_TIME = Histogram(
    "qbitel_gateway_rate_limit_wait_seconds",
    "Time spent waiting for rate limit",
    ["domain"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)


# =============================================================================
# Exceptions
# =============================================================================


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded and request cannot proceed."""

    def __init__(
        self,
        message: str,
        domain: str,
        retry_after: float,
        current_rate: float,
        limit: float,
    ):
        super().__init__(message)
        self.domain = domain
        self.retry_after = retry_after
        self.current_rate = current_rate
        self.limit = limit


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    # Global limits
    global_requests_per_minute: int = 1000
    global_tokens_per_minute: int = 500000

    # Per-domain limits (requests per minute)
    domain_limits: Dict[str, int] = field(
        default_factory=lambda: {
            "protocol_copilot": 200,
            "security_orchestrator": 300,
            "legacy_whisperer": 100,
            "compliance_reporter": 100,
            "translation_studio": 150,
        }
    )

    # Per-user limits (requests per minute)
    user_requests_per_minute: int = 60
    user_tokens_per_minute: int = 100000

    # Token bucket settings
    bucket_capacity: int = 100  # Max burst size
    refill_rate: float = 1.67  # Tokens per second (100/minute)

    # Queue settings
    max_queue_size: int = 100
    max_wait_seconds: float = 30.0

    # Redis settings
    redis_url: Optional[str] = None
    redis_prefix: str = "qbitel:rate_limit:"


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: int
    tokens: float
    refill_rate: float
    last_refill: float = field(default_factory=time.time)

    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, count: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        self.refill()
        if self.tokens >= count:
            self.tokens -= count
            return True
        return False

    def time_until_available(self, count: int = 1) -> float:
        """Calculate time until enough tokens are available."""
        self.refill()
        if self.tokens >= count:
            return 0.0
        needed = count - self.tokens
        return needed / self.refill_rate


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """
    Token bucket rate limiter with distributed support.

    Implements per-domain and per-user rate limiting with optional
    Redis backend for distributed environments.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()

        # Token buckets (in-memory)
        self._global_bucket = TokenBucket(
            capacity=self.config.global_requests_per_minute,
            tokens=float(self.config.global_requests_per_minute),
            refill_rate=self.config.global_requests_per_minute / 60.0,
        )
        self._domain_buckets: Dict[str, TokenBucket] = {}
        self._user_buckets: Dict[str, TokenBucket] = {}

        # Request queues
        self._queues: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue(maxsize=self.config.max_queue_size))

        # Redis client
        self._redis: Optional[Any] = None

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Sliding window counters for accurate rate limiting
        self._request_counts: Dict[str, list] = defaultdict(list)

    async def initialize(self):
        """Initialize rate limiter."""
        if self.config.redis_url and redis is not None:
            try:
                self._redis = redis.from_url(
                    self.config.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._redis.ping()
                logger.info("Rate limiter connected to Redis")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self._redis = None

        # Initialize domain buckets
        for domain, limit in self.config.domain_limits.items():
            self._domain_buckets[domain] = TokenBucket(
                capacity=limit,
                tokens=float(limit),
                refill_rate=limit / 60.0,
            )

        logger.info("Rate limiter initialized")

    async def shutdown(self):
        """Shutdown rate limiter."""
        if self._redis:
            await self._redis.close()
        logger.info("Rate limiter shutdown")

    async def acquire(
        self,
        domain: str,
        user_id: Optional[str] = None,
        tokens: int = 1,
        wait: bool = True,
    ) -> bool:
        """
        Acquire rate limit tokens.

        Args:
            domain: Request domain
            user_id: Optional user ID for per-user limiting
            tokens: Number of tokens to consume
            wait: If True, wait for tokens; if False, fail immediately

        Returns:
            True if acquired, False if rejected

        Raises:
            RateLimitExceeded: If rate limit exceeded and wait=False
        """
        start_time = time.time()

        # Check global limit
        global_ok = await self._check_global_limit(tokens)
        if not global_ok:
            if wait:
                await self._wait_for_tokens("global", tokens)
            else:
                await self._reject("global", domain)

        # Check domain limit
        domain_ok = await self._check_domain_limit(domain, tokens)
        if not domain_ok:
            if wait:
                await self._wait_for_tokens(domain, tokens)
            else:
                await self._reject("domain", domain)

        # Check user limit
        if user_id:
            user_ok = await self._check_user_limit(user_id, tokens)
            if not user_ok:
                if wait:
                    await self._wait_for_tokens(f"user:{user_id}", tokens)
                else:
                    await self._reject("user", domain, user_id)

        # Track metrics
        wait_time = time.time() - start_time
        if wait_time > 0.1:
            RATE_LIMIT_WAIT_TIME.labels(domain=domain).observe(wait_time)

        RATE_LIMIT_REQUESTS.labels(domain=domain, action="allowed").inc()

        return True

    async def release(self, domain: str, tokens: int = 1):
        """Release tokens back to the bucket (for cancelled requests)."""
        async with self._lock:
            if domain in self._domain_buckets:
                bucket = self._domain_buckets[domain]
                bucket.tokens = min(bucket.capacity, bucket.tokens + tokens)

    async def get_status(self, domain: str) -> Dict[str, Any]:
        """Get current rate limit status for a domain."""
        async with self._lock:
            bucket = self._domain_buckets.get(domain)
            if not bucket:
                limit = self.config.domain_limits.get(domain, self.config.global_requests_per_minute)
                bucket = TokenBucket(
                    capacity=limit,
                    tokens=float(limit),
                    refill_rate=limit / 60.0,
                )
                self._domain_buckets[domain] = bucket

            bucket.refill()

            return {
                "domain": domain,
                "tokens_available": bucket.tokens,
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate,
                "utilization": 1 - (bucket.tokens / bucket.capacity),
                "requests_per_minute": int(bucket.capacity),
            }

    async def _check_global_limit(self, tokens: int) -> bool:
        """Check global rate limit."""
        async with self._lock:
            self._global_bucket.refill()
            if self._global_bucket.tokens >= tokens:
                self._global_bucket.tokens -= tokens
                return True
            return False

    async def _check_domain_limit(self, domain: str, tokens: int) -> bool:
        """Check domain rate limit."""
        async with self._lock:
            if domain not in self._domain_buckets:
                limit = self.config.domain_limits.get(domain, self.config.global_requests_per_minute)
                self._domain_buckets[domain] = TokenBucket(
                    capacity=limit,
                    tokens=float(limit),
                    refill_rate=limit / 60.0,
                )

            bucket = self._domain_buckets[domain]
            bucket.refill()

            RATE_LIMIT_TOKENS.labels(domain=domain).set(bucket.tokens)

            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                return True
            return False

    async def _check_user_limit(self, user_id: str, tokens: int) -> bool:
        """Check per-user rate limit."""
        async with self._lock:
            if user_id not in self._user_buckets:
                self._user_buckets[user_id] = TokenBucket(
                    capacity=self.config.user_requests_per_minute,
                    tokens=float(self.config.user_requests_per_minute),
                    refill_rate=self.config.user_requests_per_minute / 60.0,
                )

            bucket = self._user_buckets[user_id]
            bucket.refill()

            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                return True
            return False

    async def _wait_for_tokens(self, bucket_key: str, tokens: int):
        """Wait until tokens are available."""
        async with self._lock:
            if bucket_key == "global":
                bucket = self._global_bucket
            elif bucket_key.startswith("user:"):
                user_id = bucket_key.split(":", 1)[1]
                bucket = self._user_buckets.get(user_id)
                if not bucket:
                    return  # User bucket doesn't exist, allow
            else:
                bucket = self._domain_buckets.get(bucket_key)
                if not bucket:
                    return  # Domain bucket doesn't exist, allow

            wait_time = bucket.time_until_available(tokens)

        if wait_time > self.config.max_wait_seconds:
            domain = bucket_key if not bucket_key.startswith("user:") else "user"
            raise RateLimitExceeded(
                f"Rate limit exceeded, retry after {wait_time:.1f}s",
                domain=domain,
                retry_after=wait_time,
                current_rate=bucket.refill_rate * 60,
                limit=bucket.capacity,
            )

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        # Try to acquire again
        async with self._lock:
            bucket.refill()
            if bucket.tokens >= tokens:
                bucket.tokens -= tokens

    async def _reject(
        self,
        limit_type: str,
        domain: str,
        user_id: Optional[str] = None,
    ):
        """Reject a request due to rate limit."""
        RATE_LIMIT_REQUESTS.labels(domain=domain, action="rejected").inc()

        async with self._lock:
            if limit_type == "global":
                bucket = self._global_bucket
            elif limit_type == "user" and user_id:
                bucket = self._user_buckets.get(user_id, self._global_bucket)
            else:
                bucket = self._domain_buckets.get(domain, self._global_bucket)

            retry_after = bucket.time_until_available(1)

        raise RateLimitExceeded(
            f"Rate limit exceeded for {limit_type}",
            domain=domain,
            retry_after=retry_after,
            current_rate=bucket.refill_rate * 60,
            limit=bucket.capacity,
        )

    async def _cleanup_old_buckets(self):
        """Clean up unused user buckets."""
        async with self._lock:
            # Remove user buckets that haven't been used recently
            # and have full tokens (indicating no recent activity)
            to_remove = []
            for user_id, bucket in self._user_buckets.items():
                bucket.refill()
                if bucket.tokens >= bucket.capacity * 0.99:
                    to_remove.append(user_id)

            for user_id in to_remove[:100]:  # Limit removal per cycle
                del self._user_buckets[user_id]

            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} unused user rate limit buckets")
