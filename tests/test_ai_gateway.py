"""
Tests for QBITEL Gateway

Comprehensive test suite for the AI Gateway including:
- Semantic cache functionality
- Cost tracking
- Rate limiting
- Prompt registry
- Gateway integration
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from dataclasses import asdict

# Import gateway components
from ai_engine.gateway.semantic_cache import (
    SemanticCache,
    CacheConfig,
    CacheEntry,
    EmbeddingManager,
)
from ai_engine.gateway.cost_tracker import (
    CostTracker,
    BudgetConfig,
    UsageRecord,
    ModelPricing,
    MODEL_PRICING,
)
from ai_engine.gateway.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
    TokenBucket,
)
from ai_engine.gateway.prompt_registry import (
    PromptRegistry,
    PromptTemplate,
    PromptVersion,
    PromptStatus,
)
from ai_engine.gateway.ai_gateway import (
    AIGateway,
    GatewayConfig,
    GatewayRequest,
    GatewayResponse,
    RequestPriority,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cache_config():
    """Create a test cache config."""
    return CacheConfig(
        similarity_threshold=0.9,
        max_memory_entries=100,
        default_ttl_seconds=60,
        redis_url=None,  # Use in-memory only
    )


@pytest.fixture
def budget_config():
    """Create a test budget config."""
    return BudgetConfig(
        daily_limit=10.0,
        monthly_limit=100.0,
        per_request_limit=1.0,
        block_on_exceed=False,
        fallback_to_local=True,
    )


@pytest.fixture
def rate_limit_config():
    """Create a test rate limit config."""
    return RateLimitConfig(
        global_requests_per_minute=100,
        user_requests_per_minute=10,
        bucket_capacity=10,
        refill_rate=1.0,
        redis_url=None,
    )


@pytest.fixture
def gateway_config(cache_config, budget_config, rate_limit_config):
    """Create a test gateway config."""
    return GatewayConfig(
        cache_enabled=True,
        cache_config=cache_config,
        cost_tracking_enabled=True,
        budget_config=budget_config,
        rate_limiting_enabled=True,
        rate_limit_config=rate_limit_config,
        prompt_registry_enabled=True,
        redis_url=None,
    )


# =============================================================================
# Token Bucket Tests
# =============================================================================

class TestTokenBucket:
    """Tests for token bucket rate limiting."""

    def test_initial_tokens(self):
        """Test initial token count."""
        bucket = TokenBucket(capacity=10, tokens=10.0, refill_rate=1.0)
        assert bucket.tokens == 10.0

    def test_consume_tokens(self):
        """Test consuming tokens."""
        bucket = TokenBucket(capacity=10, tokens=10.0, refill_rate=1.0)
        assert bucket.consume(5) is True
        assert bucket.tokens == 5.0

    def test_consume_insufficient_tokens(self):
        """Test consuming more tokens than available."""
        bucket = TokenBucket(capacity=10, tokens=2.0, refill_rate=1.0)
        assert bucket.consume(5) is False
        assert bucket.tokens == 2.0  # Unchanged

    def test_refill(self):
        """Test token refill over time."""
        bucket = TokenBucket(
            capacity=10,
            tokens=0.0,
            refill_rate=10.0,  # 10 tokens per second
            last_refill=time.time() - 0.5,  # 0.5 seconds ago
        )
        bucket.refill()
        # Should have refilled ~5 tokens
        assert 4.0 <= bucket.tokens <= 6.0

    def test_refill_capped_at_capacity(self):
        """Test that refill doesn't exceed capacity."""
        bucket = TokenBucket(
            capacity=10,
            tokens=8.0,
            refill_rate=10.0,
            last_refill=time.time() - 1.0,
        )
        bucket.refill()
        assert bucket.tokens == 10.0  # Capped at capacity

    def test_time_until_available(self):
        """Test calculating time until tokens available."""
        bucket = TokenBucket(capacity=10, tokens=0.0, refill_rate=2.0)
        wait_time = bucket.time_until_available(4)
        assert 1.9 <= wait_time <= 2.1  # Should be ~2 seconds


# =============================================================================
# Embedding Manager Tests
# =============================================================================

class TestEmbeddingManager:
    """Tests for embedding manager."""

    @pytest.mark.asyncio
    async def test_hash_embedding(self):
        """Test fallback hash-based embedding."""
        embedding = EmbeddingManager._hash_embedding("test query", dim=384)
        assert len(embedding) == 384
        assert all(isinstance(v, float) for v in embedding)

    @pytest.mark.asyncio
    async def test_hash_embedding_deterministic(self):
        """Test that hash embeddings are deterministic."""
        emb1 = EmbeddingManager._hash_embedding("same text")
        emb2 = EmbeddingManager._hash_embedding("same text")
        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_hash_embedding_different_texts(self):
        """Test that different texts produce different embeddings."""
        emb1 = EmbeddingManager._hash_embedding("text one")
        emb2 = EmbeddingManager._hash_embedding("text two")
        assert emb1 != emb2

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec = [1.0, 0.0, 0.0]
        sim = EmbeddingManager.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = EmbeddingManager.cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.001


# =============================================================================
# Semantic Cache Tests
# =============================================================================

class TestSemanticCache:
    """Tests for semantic cache."""

    @pytest.mark.asyncio
    async def test_cache_initialization(self, cache_config):
        """Test cache initialization."""
        cache = SemanticCache(cache_config)
        await cache.initialize()
        assert cache._memory_cache == {}
        await cache.shutdown()

    @pytest.mark.asyncio
    async def test_cache_put_and_get(self, cache_config):
        """Test storing and retrieving from cache."""
        cache = SemanticCache(cache_config)
        await cache.initialize()

        # Store entry
        await cache.put(
            prompt="What is the weather?",
            response="It's sunny today.",
            domain="test",
            model="gpt-4o",
            tokens_used=100,
        )

        # Retrieve with same prompt
        entry = await cache.get(
            prompt="What is the weather?",
            domain="test",
        )

        assert entry is not None
        assert entry.response == "It's sunny today."
        await cache.shutdown()

    @pytest.mark.asyncio
    async def test_cache_miss_different_domain(self, cache_config):
        """Test cache miss when domain differs."""
        cache = SemanticCache(cache_config)
        await cache.initialize()

        await cache.put(
            prompt="Test prompt",
            response="Test response",
            domain="domain_a",
            model="gpt-4o",
            tokens_used=50,
        )

        entry = await cache.get(
            prompt="Test prompt",
            domain="domain_b",  # Different domain
        )

        assert entry is None
        await cache.shutdown()

    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache_config):
        """Test cache entry expiration."""
        cache_config.default_ttl_seconds = 1  # 1 second TTL
        cache = SemanticCache(cache_config)
        await cache.initialize()

        await cache.put(
            prompt="Expiring prompt",
            response="Expiring response",
            domain="test",
            model="gpt-4o",
            tokens_used=50,
        )

        # Wait for expiration
        await asyncio.sleep(1.5)

        entry = await cache.get(prompt="Expiring prompt", domain="test")
        assert entry is None
        await cache.shutdown()

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_config):
        """Test cache statistics."""
        cache = SemanticCache(cache_config)
        await cache.initialize()

        await cache.put(
            prompt="Stats test",
            response="Response",
            domain="test",
            model="gpt-4o",
            tokens_used=50,
        )

        stats = cache.get_stats()
        assert stats.memory_entries == 1
        await cache.shutdown()


# =============================================================================
# Cost Tracker Tests
# =============================================================================

class TestCostTracker:
    """Tests for cost tracking."""

    @pytest.mark.asyncio
    async def test_cost_tracker_initialization(self, budget_config):
        """Test cost tracker initialization."""
        tracker = CostTracker(budget_config)
        await tracker.initialize()
        summary = await tracker.get_usage_summary()
        assert summary["daily"]["total"] == 0.0
        await tracker.shutdown()

    @pytest.mark.asyncio
    async def test_record_usage(self, budget_config):
        """Test recording usage."""
        tracker = CostTracker(budget_config)
        await tracker.initialize()

        record = await tracker.record_usage(
            request_id="test-123",
            domain="test",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=500.0,
        )

        assert record.total_cost > 0
        assert record.input_tokens == 1000
        assert record.output_tokens == 500

        summary = await tracker.get_usage_summary()
        assert summary["daily"]["total"] > 0
        await tracker.shutdown()

    @pytest.mark.asyncio
    async def test_cached_request_no_cost(self, budget_config):
        """Test that cached requests have zero cost."""
        tracker = CostTracker(budget_config)
        await tracker.initialize()

        record = await tracker.record_usage(
            request_id="cached-123",
            domain="test",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cached=True,
        )

        assert record.total_cost == 0.0
        await tracker.shutdown()

    @pytest.mark.asyncio
    async def test_budget_check(self, budget_config):
        """Test budget checking."""
        tracker = CostTracker(budget_config)
        await tracker.initialize()

        allowed, fallback = await tracker.check_budget(
            domain="test",
            estimated_cost=0.01,
        )

        assert allowed is True
        assert fallback is None
        await tracker.shutdown()

    @pytest.mark.asyncio
    async def test_budget_exceeded_fallback(self, budget_config):
        """Test fallback when budget exceeded."""
        budget_config.daily_limit = 0.001  # Very low limit
        tracker = CostTracker(budget_config)
        await tracker.initialize()

        # Record usage to exceed budget
        await tracker.record_usage(
            request_id="big-request",
            domain="test",
            model="gpt-4o",
            input_tokens=10000,
            output_tokens=5000,
        )

        allowed, fallback = await tracker.check_budget(
            domain="test",
            estimated_cost=0.01,
        )

        assert allowed is True  # Still allowed due to fallback
        assert fallback == "llama3.2"  # Fallback to local
        await tracker.shutdown()

    def test_model_pricing(self):
        """Test model pricing lookup."""
        tracker = CostTracker()
        pricing = tracker.get_pricing("gpt-4o")
        assert pricing.provider == "openai"
        assert pricing.input_cost_per_1k > 0

    def test_local_model_zero_cost(self):
        """Test that local models have zero cost."""
        pricing = MODEL_PRICING["llama3.2"]
        assert pricing.is_local is True
        cost = pricing.calculate_cost(1000, 500)
        assert cost == 0.0


# =============================================================================
# Rate Limiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self, rate_limit_config):
        """Test rate limiter initialization."""
        limiter = RateLimiter(rate_limit_config)
        await limiter.initialize()
        status = await limiter.get_status("test")
        assert status["capacity"] > 0
        await limiter.shutdown()

    @pytest.mark.asyncio
    async def test_acquire_success(self, rate_limit_config):
        """Test successful token acquisition."""
        limiter = RateLimiter(rate_limit_config)
        await limiter.initialize()

        result = await limiter.acquire(domain="test", user_id="user1")
        assert result is True
        await limiter.shutdown()

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, rate_limit_config):
        """Test rate limit exceeded."""
        rate_limit_config.bucket_capacity = 2
        rate_limit_config.refill_rate = 0.01  # Very slow refill
        limiter = RateLimiter(rate_limit_config)
        await limiter.initialize()

        # Exhaust tokens
        await limiter.acquire(domain="test_exceeded")
        await limiter.acquire(domain="test_exceeded")

        # This should fail
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire(domain="test_exceeded", wait=False)

        await limiter.shutdown()

    @pytest.mark.asyncio
    async def test_rate_limit_per_user(self, rate_limit_config):
        """Test per-user rate limiting."""
        rate_limit_config.user_requests_per_minute = 2
        limiter = RateLimiter(rate_limit_config)
        await limiter.initialize()

        # User 1 requests
        await limiter.acquire(domain="test", user_id="user1")
        await limiter.acquire(domain="test", user_id="user1")

        # User 1 should be rate limited
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire(domain="test", user_id="user1", wait=False)

        # User 2 should still be allowed
        result = await limiter.acquire(domain="test", user_id="user2", wait=False)
        assert result is True

        await limiter.shutdown()


# =============================================================================
# Prompt Registry Tests
# =============================================================================

class TestPromptRegistry:
    """Tests for prompt registry."""

    @pytest.mark.asyncio
    async def test_registry_initialization(self):
        """Test prompt registry initialization."""
        registry = PromptRegistry()
        await registry.initialize()
        templates = await registry.list_templates()
        assert len(templates) > 0  # Default templates loaded
        await registry.shutdown()

    @pytest.mark.asyncio
    async def test_register_template(self):
        """Test registering a new template."""
        registry = PromptRegistry()
        await registry.initialize()

        template = PromptTemplate(
            name="custom_template",
            domain="test",
            description="A custom template",
        )
        await registry.register_template(template)

        retrieved = await registry.get_template("custom_template")
        assert retrieved is not None
        assert retrieved.name == "custom_template"
        await registry.shutdown()

    @pytest.mark.asyncio
    async def test_create_version(self):
        """Test creating a prompt version."""
        registry = PromptRegistry()
        await registry.initialize()

        # Register template first
        template = PromptTemplate(name="versioned", domain="test")
        await registry.register_template(template)

        # Create version
        version = await registry.create_version(
            prompt_name="versioned",
            system_prompt="You are a helpful assistant.",
            user_prompt_template="Answer: {query}",
            description="Version 1",
            make_active=True,
        )

        assert version.version_number == 1
        assert version.status == PromptStatus.ACTIVE
        await registry.shutdown()

    @pytest.mark.asyncio
    async def test_render_prompt(self):
        """Test rendering a prompt with variables."""
        registry = PromptRegistry()
        await registry.initialize()

        template = PromptTemplate(name="render_test", domain="test")
        await registry.register_template(template)

        await registry.create_version(
            prompt_name="render_test",
            system_prompt="You are {role}.",
            user_prompt_template="Question: {query}",
            make_active=True,
        )

        system, user, version = await registry.get_prompt(
            prompt_name="render_test",
            variables={"role": "an expert", "query": "What is AI?"},
            request_id="test-123",
        )

        assert system == "You are an expert."
        assert user == "Question: What is AI?"
        await registry.shutdown()

    @pytest.mark.asyncio
    async def test_ab_testing(self):
        """Test A/B testing between versions."""
        registry = PromptRegistry()
        await registry.initialize()

        template = PromptTemplate(name="ab_test", domain="test")
        await registry.register_template(template)

        # Create two versions
        v1 = await registry.create_version(
            prompt_name="ab_test",
            system_prompt="Version 1",
            user_prompt_template="{query}",
        )
        v2 = await registry.create_version(
            prompt_name="ab_test",
            system_prompt="Version 2",
            user_prompt_template="{query}",
        )

        # Start A/B test
        await registry.start_ab_test(
            prompt_name="ab_test",
            version_ids=[v1.version_id, v2.version_id],
            traffic_split=[50.0, 50.0],
        )

        template = await registry.get_template("ab_test")
        assert template.ab_test_active is True
        assert len(template.ab_test_versions) == 2

        await registry.shutdown()


# =============================================================================
# Gateway Integration Tests
# =============================================================================

class TestAIGateway:
    """Integration tests for AI Gateway."""

    @pytest.mark.asyncio
    async def test_gateway_initialization(self, gateway_config):
        """Test gateway initialization."""
        gateway = AIGateway(gateway_config)
        await gateway.initialize()
        assert gateway.is_initialized is True
        await gateway.shutdown()

    @pytest.mark.asyncio
    async def test_gateway_request_cached(self, gateway_config):
        """Test gateway request with caching."""
        gateway = AIGateway(gateway_config)
        await gateway.initialize()

        # Mock LLM service
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_response.provider = "gpt-4o"
        mock_response.tokens_used = 100
        mock_response.confidence = 0.95
        mock_response.metadata = {}
        mock_response.tool_calls = None
        mock_response.parsed_response = None
        mock_response.validated_model = None

        gateway._llm_service = Mock()
        gateway._llm_service.process_request = AsyncMock(return_value=mock_response)

        # First request - cache miss
        request1 = GatewayRequest(
            prompt="Test query",
            domain="test",
            cache_enabled=True,
        )
        response1 = await gateway.complete(request1)
        assert response1.cache_hit is False

        # Second request - cache hit
        request2 = GatewayRequest(
            prompt="Test query",
            domain="test",
            cache_enabled=True,
        )
        response2 = await gateway.complete(request2)
        assert response2.cache_hit is True

        await gateway.shutdown()

    @pytest.mark.asyncio
    async def test_gateway_cost_tracking(self, gateway_config):
        """Test gateway cost tracking."""
        gateway = AIGateway(gateway_config)
        await gateway.initialize()

        # Mock LLM service
        mock_response = Mock()
        mock_response.content = "Response"
        mock_response.provider = "gpt-4o"
        mock_response.tokens_used = 500
        mock_response.confidence = 0.9
        mock_response.metadata = {}
        mock_response.tool_calls = None
        mock_response.parsed_response = None
        mock_response.validated_model = None

        gateway._llm_service = Mock()
        gateway._llm_service.process_request = AsyncMock(return_value=mock_response)

        request = GatewayRequest(
            prompt="Track cost",
            domain="test",
            cache_enabled=False,
        )
        await gateway.complete(request)

        summary = await gateway.get_cost_summary()
        assert summary["daily"]["total"] > 0

        await gateway.shutdown()

    @pytest.mark.asyncio
    async def test_gateway_cache_stats(self, gateway_config):
        """Test gateway cache statistics."""
        gateway = AIGateway(gateway_config)
        await gateway.initialize()

        stats = await gateway.get_cache_stats()
        assert "total_entries" in stats or "memory_entries" in stats

        await gateway.shutdown()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
