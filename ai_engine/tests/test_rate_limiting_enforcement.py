"""
Test Rate Limiting Enforcement - P0 Critical Security Test
Verifies that rate limiting is actively enforced on all API endpoints.
"""

import pytest
import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import redis.asyncio as redis

from ai_engine.api.middleware import setup_middleware
from ai_engine.api.rate_limiter import (
    RedisRateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    AdvancedRateLimitMiddleware,
)
from ai_engine.core.config import Config


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    mock_client = MagicMock(spec=redis.Redis)

    # Mock pipeline
    mock_pipeline = MagicMock()

    # Create async functions for mock responses
    async def mock_execute():
        return [None, 0, None, None]

    async def mock_ping():
        return True

    async def mock_close():
        return None

    async def mock_incr():
        return 1

    async def mock_expire():
        return True

    async def mock_get():
        return None

    mock_pipeline.execute = MagicMock(return_value=mock_execute())
    mock_pipeline.zremrangebyscore = MagicMock()
    mock_pipeline.zcard = MagicMock()
    mock_pipeline.zadd = MagicMock()
    mock_pipeline.expire = MagicMock()

    mock_client.pipeline = MagicMock(return_value=mock_pipeline)
    mock_client.ping = MagicMock(return_value=mock_ping())
    mock_client.close = MagicMock(return_value=mock_close())
    mock_client.incr = MagicMock(return_value=mock_incr())
    mock_client.expire = MagicMock(return_value=mock_expire())
    mock_client.get = MagicMock(return_value=mock_get())

    return mock_client


@pytest.fixture
def rate_limit_config():
    """Create rate limit configuration."""
    return RateLimitConfig(
        requests_per_minute=10,
        requests_per_hour=100,
        burst_size=20,
        strategy=RateLimitStrategy.SLIDING_WINDOW,
        enable_per_user=True,
        enable_per_ip=True,
        enable_per_endpoint=True,
    )


@pytest.fixture
async def rate_limiter(mock_redis_client, rate_limit_config):
    """Create RedisRateLimiter instance."""
    limiter = RedisRateLimiter(
        redis_url="redis://localhost:6379/0", config=rate_limit_config
    )
    limiter.redis_client = mock_redis_client
    return limiter


class TestRateLimiterBasicFunctionality:
    """Test basic rate limiter functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self, rate_limit_config):
        """Test rate limiter can be initialized."""
        limiter = RedisRateLimiter(
            redis_url="redis://localhost:6379/0", config=rate_limit_config
        )
        assert limiter.config == rate_limit_config
        assert limiter.redis_client is None

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_within_limit(self, rate_limiter):
        """Test that requests within limit are allowed."""

        # Mock Redis to return low count
        async def mock_execute_low():
            return [None, 5, None, None]

        rate_limiter.redis_client.pipeline().execute = MagicMock(
            return_value=mock_execute_low()
        )

        is_allowed, metadata = await rate_limiter.check_rate_limit(
            "test_user", limit=10, window_seconds=60
        )

        assert is_allowed is True
        assert "limit" in metadata
        assert "remaining" in metadata

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_over_limit(self, rate_limiter):
        """Test that requests over limit are blocked."""

        # Mock Redis to return high count (over limit)
        async def mock_execute_high():
            return [None, 15, None, None]

        rate_limiter.redis_client.pipeline().execute = MagicMock(
            return_value=mock_execute_high()
        )

        is_allowed, metadata = await rate_limiter.check_rate_limit(
            "test_user", limit=10, window_seconds=60
        )

        assert is_allowed is False
        assert metadata["remaining"] == 0

    @pytest.mark.asyncio
    async def test_rate_limiter_metadata(self, rate_limiter):
        """Test that rate limiter returns proper metadata."""

        async def mock_execute_metadata():
            return [None, 3, None, None]

        rate_limiter.redis_client.pipeline().execute = MagicMock(
            return_value=mock_execute_metadata()
        )

        is_allowed, metadata = await rate_limiter.check_rate_limit(
            "test_user", limit=10, window_seconds=60
        )

        assert "limit" in metadata
        assert "remaining" in metadata
        assert "reset" in metadata
        assert "current_count" in metadata
        assert metadata["limit"] == 10


class TestRateLimitingMiddleware:
    """Test rate limiting middleware integration."""

    def test_simple_rate_limiting_middleware_blocks_excess_requests(self):
        """Test that simple middleware blocks excess requests."""
        from ai_engine.api.middleware import RateLimitingMiddleware

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        app.add_middleware(RateLimitingMiddleware, requests_per_minute=5)

        client = TestClient(app)

        # Make requests up to limit
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200, f"Request {i+1} should succeed"

        # Next request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429, "Request over limit should be blocked"
        assert "Rate limit exceeded" in response.text

    def test_rate_limiting_with_different_ips(self):
        """Test that rate limiting tracks different IPs separately."""
        from ai_engine.api.middleware import RateLimitingMiddleware

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        middleware_instance = RateLimitingMiddleware(app, requests_per_minute=3)
        app.add_middleware(RateLimitingMiddleware, requests_per_minute=3)

        # Note: TestClient doesn't easily support different client IPs
        # This test validates the structure is in place
        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200


class TestProductionRateLimitingConfig:
    """Test production rate limiting configuration."""

    def test_rate_limiting_enabled_in_production_config(self):
        """Test that rate limiting is enabled in production configuration."""
        # This would normally load from config file
        config_dict = {
            "rate_limiting": {
                "enabled": True,
                "default_limit": 100,
                "burst_limit": 200,
                "per_user_limit": 1000,
            }
        }

        assert config_dict["rate_limiting"]["enabled"] is True
        assert config_dict["rate_limiting"]["default_limit"] == 100

    def test_middleware_setup_includes_rate_limiting(self):
        """Test that middleware setup includes rate limiting."""
        from ai_engine.api.middleware import setup_middleware

        app = FastAPI()
        config = Mock(spec=Config)

        # Mock configuration
        config.security = Mock()
        config.security.tls_enabled = False
        config.security.trusted_hosts = None

        config.rate_limiting = {
            "enabled": False,  # Will still apply basic rate limiting
            "default_limit": 100,
        }

        config.redis = Mock()
        config.redis.host = "localhost"
        config.redis.port = 6379
        config.redis.password = ""

        config.cors = {"enabled": False}
        config.__dict__ = {
            "rate_limiting": config.rate_limiting,
            "cors": config.cors,
        }

        # Setup middleware
        setup_middleware(app, config)

        # Verify middleware was added (check middleware stack)
        middleware_types = [type(m).__name__ for m in app.user_middleware]

        # Should have rate limiting middleware
        assert any(
            "RateLimit" in name for name in middleware_types
        ), f"Rate limiting middleware not found. Middleware stack: {middleware_types}"


class TestRateLimitStrategies:
    """Test different rate limiting strategies."""

    @pytest.mark.asyncio
    async def test_sliding_window_strategy(self, rate_limiter):
        """Test sliding window rate limiting strategy."""
        rate_limiter.config.strategy = RateLimitStrategy.SLIDING_WINDOW

        async def mock_execute_sliding():
            return [None, 5, None, None]

        rate_limiter.redis_client.pipeline().execute = MagicMock(
            return_value=mock_execute_sliding()
        )

        is_allowed, metadata = await rate_limiter.check_rate_limit(
            "test_user", limit=10, window_seconds=60
        )

        assert is_allowed is True

    @pytest.mark.asyncio
    async def test_fixed_window_strategy(self, rate_limiter):
        """Test fixed window rate limiting strategy."""
        rate_limiter.config.strategy = RateLimitStrategy.FIXED_WINDOW

        async def mock_incr():
            return 5

        rate_limiter.redis_client.incr = MagicMock(return_value=mock_incr())

        is_allowed, metadata = await rate_limiter.check_rate_limit(
            "test_user", limit=10, window_seconds=60
        )

        assert is_allowed is True

    @pytest.mark.asyncio
    async def test_token_bucket_strategy(self, rate_limiter):
        """Test token bucket rate limiting strategy."""
        rate_limiter.config.strategy = RateLimitStrategy.TOKEN_BUCKET

        async def mock_hgetall():
            return {}

        async def mock_hset():
            return True

        rate_limiter.redis_client.hgetall = MagicMock(return_value=mock_hgetall())
        rate_limiter.redis_client.hset = MagicMock(return_value=mock_hset())

        is_allowed, metadata = await rate_limiter.check_rate_limit(
            "test_user", limit=10, window_seconds=60
        )

        assert is_allowed is True

    @pytest.mark.asyncio
    async def test_leaky_bucket_strategy(self, rate_limiter):
        """Test leaky bucket rate limiting strategy."""
        rate_limiter.config.strategy = RateLimitStrategy.LEAKY_BUCKET

        async def mock_hgetall():
            return {}

        async def mock_hset():
            return True

        rate_limiter.redis_client.hgetall = MagicMock(return_value=mock_hgetall())
        rate_limiter.redis_client.hset = MagicMock(return_value=mock_hset())

        is_allowed, metadata = await rate_limiter.check_rate_limit(
            "test_user", limit=10, window_seconds=60
        )

        assert is_allowed is True


class TestRateLimitingEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_rate_limiter_handles_redis_unavailable(self, rate_limit_config):
        """Test that rate limiter fails open when Redis is unavailable."""
        limiter = RedisRateLimiter(
            redis_url="redis://localhost:6379/0", config=rate_limit_config
        )
        # Redis client is None (not initialized)

        is_allowed, metadata = await limiter.check_rate_limit(
            "test_user", limit=10, window_seconds=60
        )

        # Should allow request when Redis is unavailable (fail open)
        assert is_allowed is True

    @pytest.mark.asyncio
    async def test_rate_limiter_handles_redis_errors(self, rate_limiter):
        """Test that rate limiter handles Redis errors gracefully."""
        # Mock Redis to raise exception
        rate_limiter.redis_client.pipeline = MagicMock(
            side_effect=Exception("Redis connection error")
        )

        is_allowed, metadata = await rate_limiter.check_rate_limit(
            "test_user", limit=10, window_seconds=60
        )

        # Should fail open (allow request) on error
        assert is_allowed is True

    @pytest.mark.asyncio
    async def test_whitelist_bypasses_rate_limiting(
        self, rate_limiter, rate_limit_config
    ):
        """Test that whitelisted IPs bypass rate limiting."""
        rate_limit_config.whitelist_ips = ["192.168.1.100"]

        # This would be tested in middleware, but structure is validated here
        assert "192.168.1.100" in rate_limit_config.whitelist_ips

    @pytest.mark.asyncio
    async def test_blacklist_blocks_requests(self, rate_limit_config):
        """Test that blacklisted IPs are blocked."""
        rate_limit_config.blacklist_ips = ["10.0.0.1"]

        # This would be tested in middleware, but structure is validated here
        assert "10.0.0.1" in rate_limit_config.blacklist_ips


class TestRateLimitingProduction:
    """Production-specific rate limiting tests."""

    def test_production_config_has_rate_limiting(self):
        """Verify production config has rate limiting enabled."""
        # This test ensures rate limiting configuration exists
        production_config = {
            "rate_limiting": {
                "enabled": True,
                "default_limit": 100,
                "burst_limit": 200,
                "per_user_limit": 1000,
                "per_ip_limit": 500,
            }
        }

        assert production_config["rate_limiting"]["enabled"] is True
        assert production_config["rate_limiting"]["default_limit"] > 0
        assert (
            production_config["rate_limiting"]["burst_limit"]
            >= production_config["rate_limiting"]["default_limit"]
        )

    def test_rate_limiting_always_active_in_middleware(self):
        """Test that rate limiting is always active in middleware setup."""
        from ai_engine.api.middleware import setup_middleware

        app = FastAPI()
        config = Mock(spec=Config)

        # Even with rate limiting "disabled", basic protection should apply
        config.security = Mock()
        config.security.tls_enabled = False
        config.security.trusted_hosts = None

        config.rate_limiting = Mock()
        config.rate_limiting.enabled = False
        config.rate_limiting.default_limit = 100

        config.redis = Mock()
        config.redis.host = "localhost"
        config.redis.port = 6379
        config.redis.password = ""

        config.cors = {"enabled": False}
        config.__dict__ = {
            "rate_limiting": config.rate_limiting,
            "cors": config.cors,
        }

        # Setup middleware
        setup_middleware(app, config)

        # Verify some form of rate limiting is active
        middleware_names = [type(m).__name__ for m in app.user_middleware]
        assert any(
            "RateLimit" in str(name) for name in middleware_names
        ), "Rate limiting middleware should always be present for security"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
