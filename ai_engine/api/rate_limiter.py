"""
CRONOS AI Engine - Advanced Rate Limiting
Production-ready rate limiting with Redis backend and multiple strategies.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import redis.asyncio as redis
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    burst_size: int = 200
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    enable_per_user: bool = True
    enable_per_ip: bool = True
    enable_per_endpoint: bool = True
    whitelist_ips: list = None
    blacklist_ips: list = None
    
    def __post_init__(self):
        if self.whitelist_ips is None:
            self.whitelist_ips = []
        if self.blacklist_ips is None:
            self.blacklist_ips = []


class RedisRateLimiter:
    """
    Redis-backed rate limiter with multiple strategies.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        config: Optional[RateLimitConfig] = None
    ):
        self.redis_url = redis_url
        self.config = config or RateLimitConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            self.logger.info("Rate limiter Redis connection established")
        except Exception as e:
            self.logger.error(f"Failed to initialize rate limiter: {e}")
            raise
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: Unique identifier (user_id, ip, etc.)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            (is_allowed, metadata)
        """
        if not self.redis_client:
            # Fall back to allowing request if Redis is unavailable
            self.logger.warning("Redis unavailable, allowing request")
            return True, {}
        
        try:
            if self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._sliding_window_check(identifier, limit, window_seconds)
            elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._token_bucket_check(identifier, limit, window_seconds)
            elif self.config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                return await self._leaky_bucket_check(identifier, limit, window_seconds)
            else:  # FIXED_WINDOW
                return await self._fixed_window_check(identifier, limit, window_seconds)
                
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request on error
            return True, {}
    
    async def _sliding_window_check(
        self,
        identifier: str,
        limit: int,
        window_seconds: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window rate limit check."""
        key = f"rate_limit:sliding:{identifier}"
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiry
        pipe.expire(key, window_seconds)
        
        results = await pipe.execute()
        current_count = results[1]
        
        is_allowed = current_count < limit
        remaining = max(0, limit - current_count - 1)
        
        metadata = {
            "limit": limit,
            "remaining": remaining,
            "reset": int(current_time + window_seconds),
            "current_count": current_count
        }
        
        return is_allowed, metadata
    
    async def _fixed_window_check(
        self,
        identifier: str,
        limit: int,
        window_seconds: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window rate limit check."""
        current_window = int(time.time() / window_seconds)
        key = f"rate_limit:fixed:{identifier}:{current_window}"
        
        # Increment counter
        count = await self.redis_client.incr(key)
        
        # Set expiry on first request
        if count == 1:
            await self.redis_client.expire(key, window_seconds)
        
        is_allowed = count <= limit
        remaining = max(0, limit - count)
        reset_time = (current_window + 1) * window_seconds
        
        metadata = {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "current_count": count
        }
        
        return is_allowed, metadata
    
    async def _token_bucket_check(
        self,
        identifier: str,
        limit: int,
        window_seconds: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket rate limit check."""
        key = f"rate_limit:token:{identifier}"
        current_time = time.time()
        
        # Get current bucket state
        bucket_data = await self.redis_client.hgetall(key)
        
        if not bucket_data:
            # Initialize bucket
            tokens = limit - 1
            last_refill = current_time
        else:
            tokens = float(bucket_data.get("tokens", limit))
            last_refill = float(bucket_data.get("last_refill", current_time))
            
            # Refill tokens based on time elapsed
            time_elapsed = current_time - last_refill
            refill_rate = limit / window_seconds
            tokens_to_add = time_elapsed * refill_rate
            tokens = min(limit, tokens + tokens_to_add)
            last_refill = current_time
        
        is_allowed = tokens >= 1
        
        if is_allowed:
            tokens -= 1
        
        # Update bucket state
        await self.redis_client.hset(
            key,
            mapping={
                "tokens": str(tokens),
                "last_refill": str(last_refill)
            }
        )
        await self.redis_client.expire(key, window_seconds * 2)
        
        metadata = {
            "limit": limit,
            "remaining": int(tokens),
            "reset": int(last_refill + window_seconds),
            "tokens": tokens
        }
        
        return is_allowed, metadata
    
    async def _leaky_bucket_check(
        self,
        identifier: str,
        limit: int,
        window_seconds: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Leaky bucket rate limit check."""
        key = f"rate_limit:leaky:{identifier}"
        current_time = time.time()
        
        # Get current bucket state
        bucket_data = await self.redis_client.hgetall(key)
        
        if not bucket_data:
            # Initialize bucket
            water_level = 1
            last_leak = current_time
        else:
            water_level = float(bucket_data.get("water_level", 0))
            last_leak = float(bucket_data.get("last_leak", current_time))
            
            # Leak water based on time elapsed
            time_elapsed = current_time - last_leak
            leak_rate = limit / window_seconds
            water_leaked = time_elapsed * leak_rate
            water_level = max(0, water_level - water_leaked)
            last_leak = current_time
            
            # Add new request
            water_level += 1
        
        is_allowed = water_level <= limit
        
        # Update bucket state
        await self.redis_client.hset(
            key,
            mapping={
                "water_level": str(water_level),
                "last_leak": str(last_leak)
            }
        )
        await self.redis_client.expire(key, window_seconds * 2)
        
        metadata = {
            "limit": limit,
            "remaining": max(0, int(limit - water_level)),
            "reset": int(last_leak + window_seconds),
            "water_level": water_level
        }
        
        return is_allowed, metadata
    
    async def reset_limit(self, identifier: str):
        """Reset rate limit for identifier."""
        if self.redis_client:
            keys = await self.redis_client.keys(f"rate_limit:*:{identifier}*")
            if keys:
                await self.redis_client.delete(*keys)
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


class AdvancedRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting middleware with multiple strategies.
    """
    
    def __init__(
        self,
        app,
        rate_limiter: RedisRateLimiter,
        config: Optional[RateLimitConfig] = None
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.config = config or RateLimitConfig()
        self.logger = logging.getLogger(__name__)
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        
        # Get client identifier
        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)
        endpoint = request.url.path
        
        # Check whitelist/blacklist
        if client_ip in self.config.whitelist_ips:
            return await call_next(request)
        
        if client_ip in self.config.blacklist_ips:
            return Response(
                content="Access denied",
                status_code=403
            )
        
        # Check rate limits
        rate_limit_checks = []
        
        # Per-IP rate limit
        if self.config.enable_per_ip:
            rate_limit_checks.append(
                ("ip", client_ip, self.config.requests_per_minute, 60)
            )
            rate_limit_checks.append(
                ("ip_hour", client_ip, self.config.requests_per_hour, 3600)
            )
        
        # Per-user rate limit
        if self.config.enable_per_user and user_id:
            rate_limit_checks.append(
                ("user", user_id, self.config.requests_per_minute * 10, 60)
            )
            rate_limit_checks.append(
                ("user_hour", user_id, self.config.requests_per_hour * 10, 3600)
            )
        
        # Per-endpoint rate limit
        if self.config.enable_per_endpoint:
            endpoint_key = f"{client_ip}:{endpoint}"
            rate_limit_checks.append(
                ("endpoint", endpoint_key, self.config.requests_per_minute, 60)
            )
        
        # Check all rate limits
        for check_type, identifier, limit, window in rate_limit_checks:
            is_allowed, metadata = await self.rate_limiter.check_rate_limit(
                identifier, limit, window
            )
            
            if not is_allowed:
                self.logger.warning(
                    f"Rate limit exceeded for {check_type}: {identifier}",
                    extra={
                        "client_ip": client_ip,
                        "user_id": user_id,
                        "endpoint": endpoint,
                        "limit_type": check_type
                    }
                )
                
                # Return 429 with rate limit headers
                return Response(
                    content=f"Rate limit exceeded. Try again in {metadata.get('reset', 60)} seconds.",
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    headers={
                        "X-RateLimit-Limit": str(metadata.get("limit", limit)),
                        "X-RateLimit-Remaining": str(metadata.get("remaining", 0)),
                        "X-RateLimit-Reset": str(metadata.get("reset", 0)),
                        "Retry-After": str(window)
                    }
                )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        if rate_limit_checks:
            _, identifier, limit, window = rate_limit_checks[0]
            _, metadata = await self.rate_limiter.check_rate_limit(identifier, limit, window)
            
            response.headers["X-RateLimit-Limit"] = str(metadata.get("limit", limit))
            response.headers["X-RateLimit-Remaining"] = str(metadata.get("remaining", 0))
            response.headers["X-RateLimit-Reset"] = str(metadata.get("reset", 0))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check X-Forwarded-For header (for proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request."""
        # Try to get from request state (set by auth middleware)
        if hasattr(request.state, "user_id"):
            return request.state.user_id
        
        # Try to get from JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Would decode JWT here
            pass
        
        return None


# Global rate limiter instance
_rate_limiter: Optional[RedisRateLimiter] = None


async def get_rate_limiter(
    redis_url: Optional[str] = None,
    config: Optional[RateLimitConfig] = None
) -> RedisRateLimiter:
    """Get or create global rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RedisRateLimiter(
            redis_url=redis_url or "redis://localhost:6379/0",
            config=config
        )
        await _rate_limiter.initialize()
    
    return _rate_limiter