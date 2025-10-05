"""
Tests for comprehensive error handling system.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from ai_engine.core.error_handling import (
    ErrorHandler,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    ErrorContext,
    ErrorRecord,
    RetryConfig,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreaker,
    RetryManager,
    HealthMonitor,
    handle_errors,
    error_context,
)
from ai_engine.core.exceptions import (
    CronosAIException,
    ProtocolException,
    ModelException,
    InferenceException,
)


class TestErrorSeverityAndCategory:
    """Test error classification enums."""

    def test_error_severity_values(self):
        """Test error severity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_error_category_values(self):
        """Test error category enum values."""
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.MODEL_INFERENCE.value == "model_inference"
        assert ErrorCategory.SECURITY.value == "security"

    def test_recovery_strategy_values(self):
        """Test recovery strategy enum values."""
        assert RecoveryStrategy.RETRY.value == "retry"
        assert RecoveryStrategy.FALLBACK.value == "fallback"
        assert RecoveryStrategy.CIRCUIT_BREAK.value == "circuit_break"
        assert RecoveryStrategy.FAIL_FAST.value == "fail_fast"


class TestErrorContext:
    """Test ErrorContext dataclass."""

    def test_error_context_creation(self):
        """Test creating error context."""
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            request_id="req-123",
            user_id="user-456",
        )

        assert context.component == "test_component"
        assert context.operation == "test_operation"
        assert context.request_id == "req-123"
        assert context.user_id == "user-456"

    def test_error_context_with_additional_data(self):
        """Test error context with additional data."""
        context = ErrorContext(
            component="api",
            operation="get_data",
            additional_data={"endpoint": "/api/users", "method": "GET"},
        )

        assert context.additional_data["endpoint"] == "/api/users"
        assert context.additional_data["method"] == "GET"


class TestErrorRecord:
    """Test ErrorRecord dataclass."""

    def test_error_record_creation(self):
        """Test creating error record."""
        context = ErrorContext(component="test", operation="op")
        record = ErrorRecord(
            error_id="err-123",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
            component="test",
            operation="op",
            exception_type="ConnectionError",
            exception_message="Connection failed",
            stack_trace="...",
            context=context,
        )

        assert record.error_id == "err-123"
        assert record.severity == ErrorSeverity.HIGH
        assert record.category == ErrorCategory.NETWORK
        assert record.exception_type == "ConnectionError"

    def test_error_record_to_dict(self):
        """Test error record conversion to dictionary."""
        context = ErrorContext(component="test", operation="op")
        record = ErrorRecord(
            error_id="err-123",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
            component="test",
            operation="op",
            exception_type="ValueError",
            exception_message="Invalid value",
            stack_trace="...",
            context=context,
            recovery_attempted=True,
            recovery_successful=True,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        result = record.to_dict()
        assert result["error_id"] == "err-123"
        assert result["severity"] == "medium"
        assert result["category"] == "data_validation"
        assert result["recovery_attempted"] is True
        assert result["recovery_successful"] is True
        assert result["recovery_strategy"] == "retry"


class TestRetryManager:
    """Test RetryManager class."""

    @pytest.mark.asyncio
    async def test_retry_success_on_first_attempt(self):
        """Test successful execution on first attempt."""
        retry_manager = RetryManager(max_retries=3)

        async def success_func():
            return "success"

        result = await retry_manager.retry(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test successful execution after failures."""
        retry_manager = RetryManager(max_retries=3, base_delay=0.01)
        attempt_count = [0]

        async def eventually_succeeds():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await retry_manager.retry(eventually_succeeds)
        assert result == "success"
        assert attempt_count[0] == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test retry attempts exhausted."""
        retry_manager = RetryManager(max_retries=2, base_delay=0.01)

        async def always_fails():
            raise ConnectionError("Permanent failure")

        with pytest.raises(ConnectionError, match="Permanent failure"):
            await retry_manager.retry(always_fails)

    @pytest.mark.asyncio
    async def test_retry_non_retryable_exception(self):
        """Test non-retryable exception fails immediately."""
        retry_manager = RetryManager(
            max_retries=3, retryable_exceptions=(ConnectionError,)
        )

        async def raises_value_error():
            raise ValueError("Non-retryable")

        with pytest.raises(ValueError, match="Non-retryable"):
            await retry_manager.retry(raises_value_error)

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        retry_manager = RetryManager(
            max_retries=3, base_delay=1.0, backoff_factor=2.0, jitter=False
        )

        delay1 = retry_manager._get_delay(1)
        delay2 = retry_manager._get_delay(2)
        delay3 = retry_manager._get_delay(3)

        assert delay1 == 1.0  # 1.0 * 2^0
        assert delay2 == 2.0  # 1.0 * 2^1
        assert delay3 == 4.0  # 1.0 * 2^2

    def test_retry_should_retry_logic(self):
        """Test retry decision logic."""
        retry_manager = RetryManager(
            retryable_exceptions=(ConnectionError, TimeoutError)
        )

        assert retry_manager._should_retry(ConnectionError())
        assert retry_manager._should_retry(TimeoutError())
        assert not retry_manager._should_retry(ValueError())


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state allows calls."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        async def success_func():
            return "success"

        assert cb.state == CircuitBreakerState.CLOSED
        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        config = CircuitBreakerConfig(
            failure_threshold=3, expected_exception_types={ConnectionError}
        )
        cb = CircuitBreaker(config)

        async def failing_func():
            raise ConnectionError("Service unavailable")

        # Trigger failures to open circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(failing_func)

        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_blocks_calls(self):
        """Test open circuit breaker blocks calls."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0)
        cb = CircuitBreaker(config)

        async def failing_func():
            raise ConnectionError("Fail")

        # Open the circuit
        with pytest.raises(ConnectionError):
            await cb.call(failing_func)

        assert cb.state == CircuitBreakerState.OPEN

        # Subsequent calls should fail immediately
        async def any_func():
            return "result"

        with pytest.raises(CronosAIException, match="Circuit breaker is OPEN"):
            await cb.call(any_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_transition(self):
        """Test circuit breaker transitions to half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,  # Short timeout for testing
            half_open_max_calls=2,
        )
        cb = CircuitBreaker(config)

        async def failing_func():
            raise ConnectionError("Fail")

        # Open the circuit
        with pytest.raises(ConnectionError):
            await cb.call(failing_func)

        assert cb.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next call should transition to half-open
        async def success_func():
            return "recovered"

        result = await cb.call(success_func)
        assert result == "recovered"
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_to_closed(self):
        """Test half-open circuit transitions to closed on success."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker(config)

        async def fail_then_succeed():
            raise ConnectionError("Fail")

        # Open the circuit
        with pytest.raises(ConnectionError):
            await cb.call(fail_then_succeed)

        # Wait for recovery
        await asyncio.sleep(0.15)

        async def success_func():
            return "success"

        # Should transition from HALF_OPEN to CLOSED
        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0


class TestErrorHandler:
    """Test ErrorHandler class."""

    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler(enable_persistent_storage=False, enable_sentry=False)

        assert len(handler.error_records) == 0
        assert len(handler.error_stats) == 0
        assert len(handler.circuit_breakers) == 0

    def test_classify_error(self):
        """Test error classification."""
        handler = ErrorHandler(enable_persistent_storage=False, enable_sentry=False)

        # Test specific exception types
        severity, category, strategy = handler.classify_error(ConnectionError())
        assert severity == ErrorSeverity.MEDIUM
        assert category == ErrorCategory.NETWORK
        assert strategy == RecoveryStrategy.RETRY

        severity, category, strategy = handler.classify_error(ModelException("Test"))
        assert severity == ErrorSeverity.HIGH
        assert category == ErrorCategory.MODEL_INFERENCE
        assert strategy == RecoveryStrategy.FALLBACK

    @pytest.mark.asyncio
    async def test_handle_error_logging(self):
        """Test error handling and logging."""
        handler = ErrorHandler(enable_persistent_storage=False, enable_sentry=False)

        context = ErrorContext(component="test", operation="test_op")
        exception = ValueError("Test error")

        success, result = await handler.handle_error(exception, context)

        # Should log the error
        assert len(handler.error_records) == 1
        record = handler.error_records[0]
        assert record.component == "test"
        assert record.operation == "test_op"
        assert record.exception_type == "ValueError"

    @pytest.mark.asyncio
    async def test_handle_error_with_retry_recovery(self):
        """Test error handling with retry recovery."""
        handler = ErrorHandler(enable_persistent_storage=False, enable_sentry=False)

        context = ErrorContext(component="test", operation="test_op")

        # Create a recovery function that succeeds
        async def recovery_func():
            return "recovered"

        exception = InferenceException("Inference failed")

        success, result = await handler.handle_error(
            exception, context, recovery_func=recovery_func
        )

        assert success is True
        assert result == "recovered"

    def test_get_circuit_breaker(self):
        """Test getting/creating circuit breaker."""
        handler = ErrorHandler(enable_persistent_storage=False, enable_sentry=False)

        cb1 = handler.get_circuit_breaker("component1")
        cb2 = handler.get_circuit_breaker("component1")

        # Should return same instance
        assert cb1 is cb2
        assert "component1" in handler.circuit_breakers

    def test_set_retry_config(self):
        """Test setting retry configuration."""
        handler = ErrorHandler(enable_persistent_storage=False, enable_sentry=False)

        config = RetryConfig(max_attempts=5, base_delay=2.0)
        handler.set_retry_config("component1", config)

        assert "component1" in handler.retry_configs
        assert handler.retry_configs["component1"].max_attempts == 5
        assert handler.retry_configs["component1"].base_delay == 2.0

    def test_get_error_statistics(self):
        """Test getting error statistics."""
        handler = ErrorHandler(enable_persistent_storage=False, enable_sentry=False)

        # Manually add error stats
        handler.error_stats["high_network"] = 5
        handler.component_errors["api"]["ConnectionError"] = 3

        stats = handler.get_error_statistics()

        assert stats["total_errors"] == 0  # No actual records
        assert "high_network" in stats["error_by_severity"]
        assert "api" in stats["error_by_component"]

    @pytest.mark.asyncio
    async def test_get_component_health(self):
        """Test component health status."""
        handler = ErrorHandler(enable_persistent_storage=False, enable_sentry=False)

        # Add some error records
        context = ErrorContext(component="api", operation="request")
        for _ in range(5):
            error = ErrorRecord(
                error_id=f"err-{_}",
                timestamp=time.time(),
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.NETWORK,
                component="api",
                operation="request",
                exception_type="ConnectionError",
                exception_message="Failed",
                stack_trace="...",
                context=context,
            )
            handler.error_records.append(error)

        health = handler.get_component_health("api")

        assert health["component"] == "api"
        assert health["health_status"] in ["healthy", "degraded", "unhealthy"]
        assert health["recent_error_count"] > 0


class TestHandleErrorsDecorator:
    """Test handle_errors decorator."""

    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Test decorator with successful function."""

        @handle_errors(component="test")
        async def success_func():
            return "success"

        result = await success_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_with_error(self):
        """Test decorator handles errors."""

        @handle_errors(component="test", operation="failing_op")
        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await failing_func()


class TestErrorContextManager:
    """Test error_context context manager."""

    @pytest.mark.asyncio
    async def test_error_context_success(self):
        """Test error context with successful operation."""
        async with error_context("test", "operation") as ctx:
            assert ctx.component == "test"
            assert ctx.operation == "operation"

    @pytest.mark.asyncio
    async def test_error_context_with_error(self):
        """Test error context handles errors."""
        with pytest.raises(ValueError):
            async with error_context("test", "failing_op"):
                raise ValueError("Test error")


class TestHealthMonitor:
    """Test HealthMonitor class."""

    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        handler = ErrorHandler(enable_persistent_storage=False, enable_sentry=False)
        monitor = HealthMonitor(handler)

        assert monitor.error_handler is handler
        assert len(monitor.health_checks) == 0

    @pytest.mark.asyncio
    async def test_register_health_check(self):
        """Test registering health check."""
        handler = ErrorHandler(enable_persistent_storage=False, enable_sentry=False)
        monitor = HealthMonitor(handler)

        async def custom_health_check():
            return {"status": "healthy"}

        monitor.register_health_check("component1", custom_health_check)

        assert "component1" in monitor.health_checks

    @pytest.mark.asyncio
    async def test_check_system_health(self):
        """Test system-wide health check."""
        handler = ErrorHandler(enable_persistent_storage=False, enable_sentry=False)
        monitor = HealthMonitor(handler)

        # Register a health check
        async def health_check():
            return {"custom_metric": "ok"}

        monitor.register_health_check("test_component", health_check)

        # Check system health
        health = await monitor.check_system_health()

        assert "overall_status" in health
        assert "components" in health
        assert "error_statistics" in health


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_retry_config_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.backoff_strategy == "exponential"

    def test_retry_config_custom(self):
        """Test custom retry configuration."""
        config = RetryConfig(max_attempts=5, base_delay=2.0, backoff_strategy="linear")

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.backoff_strategy == "linear"


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass."""

    def test_circuit_breaker_config_defaults(self):
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_calls == 3

    def test_circuit_breaker_config_custom(self):
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(failure_threshold=10, recovery_timeout=30.0)

        assert config.failure_threshold == 10
        assert config.recovery_timeout == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
