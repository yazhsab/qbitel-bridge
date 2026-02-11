"""
QBITEL Engine - Circuit Breaker Tests

Tests for the circuit breaker infrastructure.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai_engine.core.circuit_breakers import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
    CircuitOpenError,
    circuit_breakers,
    with_circuit_breaker,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig(name="test")

        assert config.name == "test"
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_calls == 3
        assert config.excluded_exceptions == set()
        assert config.included_exceptions is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            name="custom",
            failure_threshold=3,
            recovery_timeout=30.0,
            excluded_exceptions={ValueError},
        )

        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert ValueError in config.excluded_exceptions


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker for testing."""
        config = CircuitBreakerConfig(
            name="test_breaker",
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout=1.0,  # Short timeout for testing
            half_open_max_calls=2,
        )
        return CircuitBreaker(config)

    def test_initial_state(self, breaker):
        """Test initial circuit state is closed."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open
        assert not breaker.is_half_open

    @pytest.mark.asyncio
    async def test_successful_call(self, breaker):
        """Test successful call through circuit breaker."""
        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_failed_call(self, breaker):
        """Test failed call through circuit breaker."""
        async def fail_func():
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            await breaker.call(fail_func)

        assert breaker._failure_count == 1
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, breaker):
        """Test circuit opens after failure threshold is reached."""
        async def fail_func():
            raise RuntimeError("Test error")

        # Cause failures up to threshold
        for _ in range(breaker.config.failure_threshold):
            with pytest.raises(RuntimeError):
                await breaker.call(fail_func)

        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self, breaker):
        """Test open circuit rejects new calls."""
        async def fail_func():
            raise RuntimeError("Test error")

        # Open the circuit
        for _ in range(breaker.config.failure_threshold):
            with pytest.raises(RuntimeError):
                await breaker.call(fail_func)

        # New calls should be rejected
        with pytest.raises(CircuitOpenError) as exc_info:
            await breaker.call(fail_func)

        assert exc_info.value.circuit_name == "test_breaker"
        assert exc_info.value.time_until_retry >= 0

    @pytest.mark.asyncio
    async def test_half_open_after_recovery_timeout(self, breaker):
        """Test circuit transitions to half-open after recovery timeout."""
        async def fail_func():
            raise RuntimeError("Test error")

        # Open the circuit
        for _ in range(breaker.config.failure_threshold):
            with pytest.raises(RuntimeError):
                await breaker.call(fail_func)

        assert breaker.is_open

        # Wait for recovery timeout
        await asyncio.sleep(breaker.config.recovery_timeout + 0.1)

        # Next call should be allowed (half-open)
        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        # After success in half-open, we're still half-open until threshold
        assert breaker._success_count == 1

    @pytest.mark.asyncio
    async def test_half_open_closes_after_successes(self, breaker):
        """Test circuit closes after success threshold in half-open."""
        async def fail_func():
            raise RuntimeError("Test error")

        # Open the circuit
        for _ in range(breaker.config.failure_threshold):
            with pytest.raises(RuntimeError):
                await breaker.call(fail_func)

        # Wait for recovery timeout
        await asyncio.sleep(breaker.config.recovery_timeout + 0.1)

        async def success_func():
            return "success"

        # Succeed enough times to close
        for _ in range(breaker.config.success_threshold):
            await breaker.call(success_func)

        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_half_open_reopens_on_failure(self, breaker):
        """Test circuit reopens on failure in half-open state."""
        async def fail_func():
            raise RuntimeError("Test error")

        # Open the circuit
        for _ in range(breaker.config.failure_threshold):
            with pytest.raises(RuntimeError):
                await breaker.call(fail_func)

        # Wait for recovery timeout
        await asyncio.sleep(breaker.config.recovery_timeout + 0.1)

        # Fail in half-open state
        with pytest.raises(RuntimeError):
            await breaker.call(fail_func)

        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_excluded_exceptions_not_counted(self, breaker):
        """Test excluded exceptions don't count toward failures."""
        breaker.config.excluded_exceptions = {ValueError}

        async def value_error_func():
            raise ValueError("Excluded error")

        for _ in range(breaker.config.failure_threshold + 1):
            with pytest.raises(ValueError):
                await breaker.call(value_error_func)

        # Circuit should still be closed
        assert breaker.is_closed
        assert breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_success(self, breaker):
        """Test using circuit breaker as context manager - success."""
        async with breaker:
            result = "success"

        assert result == "success"
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_context_manager_failure(self, breaker):
        """Test using circuit breaker as context manager - failure."""
        with pytest.raises(RuntimeError):
            async with breaker:
                raise RuntimeError("Test error")

        assert breaker._failure_count == 1

    @pytest.mark.asyncio
    async def test_reset(self, breaker):
        """Test manual reset of circuit breaker."""
        async def fail_func():
            raise RuntimeError("Test error")

        # Open the circuit
        for _ in range(breaker.config.failure_threshold):
            with pytest.raises(RuntimeError):
                await breaker.call(fail_func)

        assert breaker.is_open

        # Reset
        await breaker.reset()

        assert breaker.is_closed
        assert breaker._failure_count == 0

    def test_get_status(self, breaker):
        """Test getting circuit breaker status."""
        status = breaker.get_status()

        assert status["name"] == "test_breaker"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert "config" in status
        assert status["config"]["failure_threshold"] == 3


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_register_breaker(self):
        """Test registering a circuit breaker."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(name="test")

        breaker = registry.register(config)

        assert breaker is not None
        assert "test" in registry

    def test_get_breaker(self):
        """Test getting a registered breaker."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(name="test")
        registry.register(config)

        breaker = registry.get("test")
        assert breaker is not None

        missing = registry.get("nonexistent")
        assert missing is None

    def test_getitem_breaker(self):
        """Test getting breaker with bracket notation."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(name="test")
        registry.register(config)

        breaker = registry["test"]
        assert breaker is not None

        with pytest.raises(KeyError):
            _ = registry["nonexistent"]

    def test_duplicate_registration(self):
        """Test registering same name twice returns existing."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(name="test")

        breaker1 = registry.register(config)
        breaker2 = registry.register(config)

        assert breaker1 is breaker2

    def test_get_all_status(self):
        """Test getting status of all breakers."""
        registry = CircuitBreakerRegistry()
        registry.register(CircuitBreakerConfig(name="test1"))
        registry.register(CircuitBreakerConfig(name="test2"))

        status = registry.get_all_status()

        assert "test1" in status
        assert "test2" in status

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """Test resetting all breakers."""
        registry = CircuitBreakerRegistry()
        registry.register(CircuitBreakerConfig(name="test1", failure_threshold=1))
        registry.register(CircuitBreakerConfig(name="test2", failure_threshold=1))

        # Open both breakers
        async def fail_func():
            raise RuntimeError()

        with pytest.raises(RuntimeError):
            await registry["test1"].call(fail_func)
        with pytest.raises(RuntimeError):
            await registry["test2"].call(fail_func)

        # Reset all
        await registry.reset_all()

        assert registry["test1"].is_closed
        assert registry["test2"].is_closed


class TestWithCircuitBreakerDecorator:
    """Tests for the with_circuit_breaker decorator."""

    def setup_method(self):
        """Set up test fixtures."""
        # Register a test circuit breaker
        if "test_decorator" not in circuit_breakers:
            circuit_breakers.register(
                CircuitBreakerConfig(
                    name="test_decorator",
                    failure_threshold=2,
                    recovery_timeout=1.0,
                )
            )
        # Reset before each test using asyncio.run
        asyncio.run(circuit_breakers["test_decorator"].reset())

    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Test decorator with successful function."""
        @with_circuit_breaker("test_decorator")
        async def success_func():
            return "success"

        result = await success_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_failure(self):
        """Test decorator with failing function."""
        @with_circuit_breaker("test_decorator")
        async def fail_func():
            raise RuntimeError("Error")

        with pytest.raises(RuntimeError):
            await fail_func()

    @pytest.mark.asyncio
    async def test_decorator_with_fallback(self):
        """Test decorator with fallback function."""
        call_count = 0

        @with_circuit_breaker("test_decorator", fallback=lambda: "fallback")
        async def fail_func():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Error")

        # First two calls fail but circuit is still closed
        with pytest.raises(RuntimeError):
            await fail_func()
        with pytest.raises(RuntimeError):
            await fail_func()

        # Circuit is now open, should return fallback
        result = await fail_func()
        assert result == "fallback"
        assert call_count == 2  # Function not called when circuit open

    @pytest.mark.asyncio
    async def test_decorator_unknown_breaker(self):
        """Test decorator with unknown circuit breaker name."""
        @with_circuit_breaker("nonexistent")
        async def func():
            return "success"

        # Should execute without protection
        result = await func()
        assert result == "success"


class TestGlobalCircuitBreakers:
    """Tests for pre-configured global circuit breakers."""

    def test_default_breakers_registered(self):
        """Test that default circuit breakers are registered."""
        assert "llm" in circuit_breakers
        assert "database" in circuit_breakers
        assert "redis" in circuit_breakers
        assert "discovery" in circuit_breakers
        assert "external_api" in circuit_breakers

    def test_llm_breaker_config(self):
        """Test LLM circuit breaker configuration."""
        llm_breaker = circuit_breakers["llm"]
        assert llm_breaker.config.failure_threshold == 3
        assert llm_breaker.config.recovery_timeout == 60.0
        assert ValueError in llm_breaker.config.excluded_exceptions

    def test_discovery_breaker_config(self):
        """Test discovery circuit breaker configuration."""
        discovery_breaker = circuit_breakers["discovery"]
        assert discovery_breaker.config.failure_threshold == 5
        assert discovery_breaker.config.recovery_timeout == 60.0
