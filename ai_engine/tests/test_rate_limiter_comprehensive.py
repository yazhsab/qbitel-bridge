"""
Comprehensive Unit Tests for api/rate_limiter.py
Ensures 100% code coverage for rate limiting functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import Request, Response
from starlette.datastructures import Headers

from ai_engine.api.rate_limiter import (
    RateLimitStrategy,
    RateLimitConfig,
    RedisRateLimiter,
    RateLimiter,
    AdvancedRateLimitMiddleware,
    get_rate_limiter,
)


class TestRateLimitStrategy:
    """Test RateLimitStrategy enumeration."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert RateLimitStrategy.FIXED_WINDOW.value == "fixed_window"
        assert RateLimitStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert RateLimitStrategy.TOKEN_BUCKET.value == "token_bucket"
        assert RateLimitStrategy.LEAKY_BUCKET.value == "leaky_bucket"


class TestRateLimitConfig:
    """Test RateLimitConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 100
        assert config.requests_per_hour == 1000
        assert config.burst_size == 200
        assert config.strategy == RateLimitStrategy.SLIDING_WINDOW
        assert config.enable_per_user is True
        assert config.enable_per_ip is True
        assert config.enable_per_endpoint is True
        assert config.whitelist_ips == []
        assert config.blacklist_ips == []

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_minute=50,
            requests_per_hour=500,
            burst_size=100,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            whitelist_ips=["127.0.0.1"],
            blacklist_ips=["192.168.1.1"],
        )
        assert config.requests_per_minute == 50
        assert config.requests_per_hour == 500
        assert config.burst_size == 100
        assert config.strategy == RateLimitStrategy.TOKEN_BUCKET
        assert "127.0.0.1" in config.whitelist_ips
        assert "192.168.1.1" in config.blacklist_ips


class TestRedisRateLimiter:
    """Test RedisRateLimiter class."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.pipeline = Mock(return_value=AsyncMock())
        return mock_client

    @pytest.fixture
    def rate_limiter(self):
        """Create RedisRateLimiter instance."""
        return RedisRateLimiter(redis_url="redis://localhost:6379/0")

    @pytest.mark.asyncio
    async def test_initialize_success(self, rate_limiter, mock_redis):
        """Test successful Redis initialization."""
        with patch("ai_engine.api.rate_limiter.redis.from_url", return_value=mock_redis):
            await rate_limiter.initialize()
            assert rate_limiter.redis_client is not None
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, rate_limiter):
        """Test Redis initialization failure."""
        with patch(
            "ai_engine.api.rate_limiter.redis.from_url",
            side_effect=Exception("Connection failed"),
        ):
            with pytest.raises(Exception):
                await rate_limiter.initialize()

    @pytest.mark.asyncio
    async def test_check_rate_limit_no_redis(self, rate_limiter):
        """Test rate limit check without Redis."""
        is_allowed, metadata = await rate_limiter.check_rate_limit("test_user", 100, 60)
        assert is_allowed is True
        assert metadata == {}

    @pytest.mark.asyncio
    async def test_sliding_window_check_allowed(self, rate_limiter, mock_redis):
        """Test sliding window check - request allowed."""
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[None, 5, None, None])
        mock_redis.pipeline.return_value = mock_pipeline

        rate_limiter.redis_client = mock_redis
        rate_limiter.config.strategy = RateLimitStrategy.SLIDING_WINDOW

        is_allowed, metadata = await rate_limiter.check_rate_limit("test_user", 100, 60)
        assert is_allowed is True
        assert metadata["limit"] == 100
        assert metadata["remaining"] >= 0

    @pytest.mark.asyncio
    async def test_sliding_window_check_denied(self, rate_limiter, mock_redis):
        """Test sliding window check - request denied."""
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[None, 100, None, None])
        mock_redis.pipeline.return_value = mock_pipeline

        rate_limiter.redis_client = mock_redis
        rate_limiter.config.strategy = RateLimitStrategy.SLIDING_WINDOW

        is_allowed, metadata = await rate_limiter.check_rate_limit("test_user", 100, 60)
        assert is_allowed is False
        assert metadata["remaining"] == 0

    @pytest.mark.asyncio
    async def test_fixed_window_check_allowed(self, rate_limiter, mock_redis):
        """Test fixed window check - request allowed."""
        mock_redis.incr = AsyncMock(return_value=1)
        mock_redis.expire = AsyncMock()

        rate_limiter.redis_client = mock_redis
        rate_limiter.config.strategy = RateLimitStrategy.FIXED_WINDOW

        is_allowed, metadata = await rate_limiter.check_rate_limit("test_user", 100, 60)
        assert is_allowed is True
        assert metadata["limit"] == 100
        assert metadata["remaining"] == 99

    @pytest.mark.asyncio
    async def test_fixed_window_check_denied(self, rate_limiter, mock_redis):
        """Test fixed window check - request denied."""
        mock_redis.incr = AsyncMock(return_value=101)

        rate_limiter.redis_client = mock_redis
        rate_limiter.config.strategy = RateLimitStrategy.FIXED_WINDOW

        is_allowed, metadata = await rate_limiter.check_rate_limit("test_user", 100, 60)
        assert is_allowed is False
        assert metadata["remaining"] == 0

    @pytest.mark.asyncio
    async def test_token_bucket_check_initial(self, rate_limiter, mock_redis):
        """Test token bucket check - initial request."""
        mock_redis.hgetall = AsyncMock(return_value={})
        mock_redis.hset = AsyncMock()
        mock_redis.expire = AsyncMock()

        rate_limiter.redis_client = mock_redis
        rate_limiter.config.strategy = RateLimitStrategy.TOKEN_BUCKET

        is_allowed, metadata = await rate_limiter.check_rate_limit("test_user", 100, 60)
        assert is_allowed is True
        assert "tokens" in metadata

    @pytest.mark.asyncio
    async def test_token_bucket_check_refill(self, rate_limiter, mock_redis):
        """Test token bucket check - with refill."""
        current_time = time.time()
        mock_redis.hgetall = AsyncMock(return_value={"tokens": "50", "last_refill": str(current_time - 30)})
        mock_redis.hset = AsyncMock()
        mock_redis.expire = AsyncMock()

        rate_limiter.redis_client = mock_redis
        rate_limiter.config.strategy = RateLimitStrategy.TOKEN_BUCKET

        is_allowed, metadata = await rate_limiter.check_rate_limit("test_user", 100, 60)
        assert is_allowed is True

    @pytest.mark.asyncio
    async def test_token_bucket_check_no_tokens(self, rate_limiter, mock_redis):
        """Test token bucket check - no tokens available."""
        current_time = time.time()
        mock_redis.hgetall = AsyncMock(return_value={"tokens": "0", "last_refill": str(current_time)})
        mock_redis.hset = AsyncMock()
        mock_redis.expire = AsyncMock()

        rate_limiter.redis_client = mock_redis
        rate_limiter.config.strategy = RateLimitStrategy.TOKEN_BUCKET

        is_allowed, metadata = await rate_limiter.check_rate_limit("test_user", 100, 60)
        assert is_allowed is False

    @pytest.mark.asyncio
    async def test_leaky_bucket_check_initial(self, rate_limiter, mock_redis):
        """Test leaky bucket check - initial request."""
        mock_redis.hgetall = AsyncMock(return_value={})
        mock_redis.hset = AsyncMock()
        mock_redis.expire = AsyncMock()

        rate_limiter.redis_client = mock_redis
        rate_limiter.config.strategy = RateLimitStrategy.LEAKY_BUCKET

        is_allowed, metadata = await rate_limiter.check_rate_limit("test_user", 100, 60)
        assert is_allowed is True
        assert "water_level" in metadata

    @pytest.mark.asyncio
    async def test_leaky_bucket_check_with_leak(self, rate_limiter, mock_redis):
        """Test leaky bucket check - with water leak."""
        current_time = time.time()
        mock_redis.hgetall = AsyncMock(return_value={"water_level": "50", "last_leak": str(current_time - 30)})
        mock_redis.hset = AsyncMock()
        mock_redis.expire = AsyncMock()

        rate_limiter.redis_client = mock_redis
        rate_limiter.config.strategy = RateLimitStrategy.LEAKY_BUCKET

        is_allowed, metadata = await rate_limiter.check_rate_limit("test_user", 100, 60)
        assert is_allowed is True

    @pytest.mark.asyncio
    async def test_leaky_bucket_check_overflow(self, rate_limiter, mock_redis):
        """Test leaky bucket check - bucket overflow."""
        current_time = time.time()
        mock_redis.hgetall = AsyncMock(return_value={"water_level": "100", "last_leak": str(current_time)})
        mock_redis.hset = AsyncMock()
        mock_redis.expire = AsyncMock()

        rate_limiter.redis_client = mock_redis
        rate_limiter.config.strategy = RateLimitStrategy.LEAKY_BUCKET

        is_allowed, metadata = await rate_limiter.check_rate_limit("test_user", 100, 60)
        assert is_allowed is False

    @pytest.mark.asyncio
    async def test_check_rate_limit_error_fail_open(self, rate_limiter, mock_redis):
        """Test rate limit check error - fail open."""
        mock_redis.pipeline.side_effect = Exception("Redis error")
        rate_limiter.redis_client = mock_redis

        is_allowed, metadata = await rate_limiter.check_rate_limit("test_user", 100, 60)
        assert is_allowed is True
        assert metadata == {}

    @pytest.mark.asyncio
    async def test_reset_limit(self, rate_limiter, mock_redis):
        """Test resetting rate limit."""
        mock_redis.keys = AsyncMock(return_value=["rate_limit:sliding:test_user"])
        mock_redis.delete = AsyncMock()

        rate_limiter.redis_client = mock_redis
        await rate_limiter.reset_limit("test_user")
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_limit_no_keys(self, rate_limiter, mock_redis):
        """Test resetting rate limit with no keys."""
        mock_redis.keys = AsyncMock(return_value=[])

        rate_limiter.redis_client = mock_redis
        await rate_limiter.reset_limit("test_user")

    @pytest.mark.asyncio
    async def test_close(self, rate_limiter, mock_redis):
        """Test closing Redis connection."""
        rate_limiter.redis_client = mock_redis
        await rate_limiter.close()
        mock_redis.close.assert_called_once()


class TestRateLimiter:
    """Test simplified RateLimiter class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.security = Mock()
        config.security.rate_limit_per_minute = 100
        return config

    @pytest.fixture
    def rate_limiter(self, mock_config):
        """Create RateLimiter instance."""
        return RateLimiter(mock_config)

    @pytest.mark.asyncio
    async def test_initialize_with_existing_client(self, rate_limiter):
        """Test initialization with existing Redis client."""
        mock_redis = AsyncMock()
        rate_limiter.redis_client = mock_redis
        await rate_limiter.initialize()
        assert rate_limiter.redis_client is mock_redis

    @pytest.mark.asyncio
    async def test_initialize_create_client(self, rate_limiter):
        """Test initialization creating new Redis client."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        with patch("ai_engine.api.rate_limiter.redis.from_url", return_value=mock_redis):
            await rate_limiter.initialize()
            assert rate_limiter.redis_client is not None
            assert rate_limiter._owns_client is True

    @pytest.mark.asyncio
    async def test_initialize_failure(self, rate_limiter):
        """Test initialization failure."""
        with patch(
            "ai_engine.api.rate_limiter.redis.from_url",
            side_effect=Exception("Connection failed"),
        ):
            await rate_limiter.initialize()
            assert rate_limiter.redis_client is None

    @pytest.mark.asyncio
    async def test_shutdown_owned_client(self, rate_limiter):
        """Test shutdown with owned client."""
        mock_redis = AsyncMock()
        rate_limiter.redis_client = mock_redis
        rate_limiter._owns_client = True

        await rate_limiter.shutdown()
        mock_redis.close.assert_called_once()
        assert rate_limiter.redis_client is None

    @pytest.mark.asyncio
    async def test_shutdown_not_owned_client(self, rate_limiter):
        """Test shutdown with non-owned client."""
        mock_redis = AsyncMock()
        rate_limiter.redis_client = mock_redis
        rate_limiter._owns_client = False

        await rate_limiter.shutdown()
        mock_redis.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_rate_limit_no_redis(self, rate_limiter):
        """Test rate limit check without Redis."""
        result = await rate_limiter.check_rate_limit("test_user")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, rate_limiter):
        """Test rate limit check - allowed."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="50")
        mock_redis.incr = AsyncMock()
        mock_redis.expire = AsyncMock()
        rate_limiter.redis_client = mock_redis

        result = await rate_limiter.check_rate_limit("test_user")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_denied(self, rate_limiter):
        """Test rate limit check - denied."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="100")
        rate_limiter.redis_client = mock_redis

        result = await rate_limiter.check_rate_limit("test_user")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_rate_limit_error(self, rate_limiter):
        """Test rate limit check with error."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=Exception("Redis error"))
        rate_limiter.redis_client = mock_redis

        result = await rate_limiter.check_rate_limit("test_user")
        assert result is True

    @pytest.mark.asyncio
    async def test_reset_rate_limit(self, rate_limiter):
        """Test resetting rate limit."""
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock()
        rate_limiter.redis_client = mock_redis

        await rate_limiter.reset_rate_limit("test_user")
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_rate_limit_no_redis(self, rate_limiter):
        """Test resetting rate limit without Redis."""
        await rate_limiter.reset_rate_limit("test_user")

    @pytest.mark.asyncio
    async def test_reset_rate_limit_error(self, rate_limiter):
        """Test resetting rate limit with error."""
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(side_effect=Exception("Redis error"))
        rate_limiter.redis_client = mock_redis

        await rate_limiter.reset_rate_limit("test_user")

    @pytest.mark.asyncio
    async def test_get_count_none(self, rate_limiter):
        """Test getting count when value is None."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        rate_limiter.redis_client = mock_redis

        count = await rate_limiter._get_count("test_key")
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_count_bytes(self, rate_limiter):
        """Test getting count when value is bytes."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=b"42")
        rate_limiter.redis_client = mock_redis

        count = await rate_limiter._get_count("test_key")
        assert count == 42

    @pytest.mark.asyncio
    async def test_get_count_string(self, rate_limiter):
        """Test getting count when value is string."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="42")
        rate_limiter.redis_client = mock_redis

        count = await rate_limiter._get_count("test_key")
        assert count == 42

    @pytest.mark.asyncio
    async def test_get_count_invalid(self, rate_limiter):
        """Test getting count with invalid value."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="invalid")
        rate_limiter.redis_client = mock_redis

        count = await rate_limiter._get_count("test_key")
        assert count == 0

    def test_build_key(self, rate_limiter):
        """Test building rate limit key."""
        key = rate_limiter._build_key("test_user")
        assert key == "rate_limit:test_user"

    def test_rate_config_from_security_config(self):
        """Test rate config initialization from security config."""
        config = Mock()
        config.security = Mock()
        config.security.rate_limit_per_minute = 50

        limiter = RateLimiter(config)
        assert limiter.rate_config.requests_per_minute == 50


class TestAdvancedRateLimitMiddleware:
    """Test AdvancedRateLimitMiddleware class."""

    @pytest.fixture
    def mock_rate_limiter(self):
        """Create mock rate limiter."""
        limiter = AsyncMock()
        limiter.check_rate_limit = AsyncMock(return_value=(True, {"limit": 100, "remaining": 99, "reset": 60}))
        return limiter

    @pytest.fixture
    def middleware(self, mock_rate_limiter):
        """Create middleware instance."""
        app = Mock()
        config = RateLimitConfig()
        return AdvancedRateLimitMiddleware(app, mock_rate_limiter, config)

    @pytest.mark.asyncio
    async def test_dispatch_whitelisted_ip(self, middleware, mock_rate_limiter):
        """Test dispatch with whitelisted IP."""
        middleware.config.whitelist_ips = ["127.0.0.1"]

        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/api/test"
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = Headers({})

        call_next = AsyncMock(return_value=Response(content="OK"))

        response = await middleware.dispatch(request, call_next)
        assert response.body == b"OK"
        mock_rate_limiter.check_rate_limit.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_blacklisted_ip(self, middleware):
        """Test dispatch with blacklisted IP."""
        middleware.config.blacklist_ips = ["192.168.1.1"]

        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/api/test"
        request.client = Mock()
        request.client.host = "192.168.1.1"
        request.headers = Headers({})

        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 403
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_rate_limit_exceeded(self, middleware, mock_rate_limiter):
        """Test dispatch with rate limit exceeded."""
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=(False, {"limit": 100, "remaining": 0, "reset": 60}))

        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/api/test"
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = Headers({})
        request.state = Mock()

        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 429
        assert "X-RateLimit-Limit" in response.headers
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_success_with_headers(self, middleware, mock_rate_limiter):
        """Test successful dispatch with rate limit headers."""
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/api/test"
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = Headers({})
        request.state = Mock()

        mock_response = Response(content="OK")
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    def test_get_client_ip_x_forwarded_for(self, middleware):
        """Test getting client IP from X-Forwarded-For header."""
        request = Mock(spec=Request)
        request.headers = Headers({"X-Forwarded-For": "192.168.1.1, 10.0.0.1"})
        request.client = Mock()
        request.client.host = "127.0.0.1"

        ip = middleware._get_client_ip(request)
        assert ip == "192.168.1.1"

    def test_get_client_ip_x_real_ip(self, middleware):
        """Test getting client IP from X-Real-IP header."""
        request = Mock(spec=Request)
        request.headers = Headers({"X-Real-IP": "192.168.1.1"})
        request.client = Mock()
        request.client.host = "127.0.0.1"

        ip = middleware._get_client_ip(request)
        assert ip == "192.168.1.1"

    def test_get_client_ip_direct(self, middleware):
        """Test getting client IP directly."""
        request = Mock(spec=Request)
        request.headers = Headers({})
        request.client = Mock()
        request.client.host = "192.168.1.1"

        ip = middleware._get_client_ip(request)
        assert ip == "192.168.1.1"

    def test_get_client_ip_no_client(self, middleware):
        """Test getting client IP when no client."""
        request = Mock(spec=Request)
        request.headers = Headers({})
        request.client = None

        ip = middleware._get_client_ip(request)
        assert ip == "unknown"

    def test_get_user_id_from_state(self, middleware):
        """Test getting user ID from request state."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.user_id = "user123"
        request.headers = Headers({})

        user_id = middleware._get_user_id(request)
        assert user_id == "user123"

    def test_get_user_id_from_bearer_token(self, middleware):
        """Test getting user ID from Bearer token."""
        request = Mock(spec=Request)
        request.state = Mock(spec=[])
        request.headers = Headers({"Authorization": "Bearer token123"})

        user_id = middleware._get_user_id(request)
        assert user_id is None  # Would decode JWT in real implementation

    def test_get_user_id_none(self, middleware):
        """Test getting user ID when not available."""
        request = Mock(spec=Request)
        request.state = Mock(spec=[])
        request.headers = Headers({})

        user_id = middleware._get_user_id(request)
        assert user_id is None


class TestGlobalRateLimiter:
    """Test global rate limiter functions."""

    @pytest.mark.asyncio
    async def test_get_rate_limiter_singleton(self):
        """Test get_rate_limiter returns singleton."""
        with patch("ai_engine.api.rate_limiter._rate_limiter", None):
            mock_limiter = AsyncMock()
            mock_limiter.initialize = AsyncMock()

            with patch("ai_engine.api.rate_limiter.RedisRateLimiter", return_value=mock_limiter):
                limiter1 = await get_rate_limiter()
                limiter2 = await get_rate_limiter()
                assert limiter1 is limiter2

    @pytest.mark.asyncio
    async def test_get_rate_limiter_with_config(self):
        """Test get_rate_limiter with custom config."""
        with patch("ai_engine.api.rate_limiter._rate_limiter", None):
            mock_limiter = AsyncMock()
            mock_limiter.initialize = AsyncMock()

            config = RateLimitConfig(requests_per_minute=50)

            with patch("ai_engine.api.rate_limiter.RedisRateLimiter", return_value=mock_limiter) as mock_class:
                await get_rate_limiter(config=config)
                mock_class.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
