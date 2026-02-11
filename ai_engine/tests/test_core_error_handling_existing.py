"""
Comprehensive tests for ai_engine.core.error_handling module.

This module provides error handling, logging, and recovery mechanisms
for the QBITEL Engine.
"""

import pytest
import asyncio
import logging
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import traceback

from ai_engine.core.error_handling import (
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    ErrorContext,
    ErrorRecord,
    RetryConfig,
    CircuitBreakerConfig,
    CircuitBreakerState,
    RetryManager,
    CircuitBreaker,
    ErrorHandler,
    HealthMonitor,
)


class TestErrorSeverity:
    """Test ErrorSeverity enum."""

    def test_error_severity_values(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_error_severity_comparison(self):
        """Test ErrorSeverity comparison."""
        assert ErrorSeverity.LOW < ErrorSeverity.MEDIUM
        assert ErrorSeverity.MEDIUM < ErrorSeverity.HIGH
        assert ErrorSeverity.HIGH < ErrorSeverity.CRITICAL


class TestErrorCategory:
    """Test ErrorCategory enum."""

    def test_error_category_values(self):
        """Test ErrorCategory enum values."""
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.AUTHENTICATION.value == "authentication"
        assert ErrorCategory.AUTHORIZATION.value == "authorization"
        assert ErrorCategory.DATA_VALIDATION.value == "data_validation"
        assert ErrorCategory.MODEL_INFERENCE.value == "model_inference"
        assert ErrorCategory.RESOURCE_EXHAUSTION.value == "resource_exhaustion"


class TestRecoveryStrategy:
    """Test RecoveryStrategy enum."""

    def test_recovery_strategy_values(self):
        """Test RecoveryStrategy enum values."""
        assert RecoveryStrategy.RETRY.value == "retry"
        assert RecoveryStrategy.FALLBACK.value == "fallback"
        assert RecoveryStrategy.IGNORE.value == "ignore"
        assert RecoveryStrategy.ESCALATE.value == "escalate"
        assert RecoveryStrategy.RESTART.value == "restart"


class TestErrorContext:
    """Test ErrorContext class."""

    def test_initialization(self):
        """Test ErrorContext initialization."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
        )

        assert context.operation == "test_operation"
        assert context.component == "test_component"
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.category == ErrorCategory.DATA_VALIDATION
        assert context.timestamp is not None
        assert context.metadata == {}

    def test_initialization_with_metadata(self):
        """Test ErrorContext initialization with metadata."""
        metadata = {"user_id": "123", "request_id": "abc"}
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
            metadata=metadata,
        )

        assert context.metadata == metadata

    def test_initialization_with_recovery_strategy(self):
        """Test ErrorContext initialization with recovery strategy."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        assert context.recovery_strategy == RecoveryStrategy.RETRY

    def test_to_dict(self):
        """Test converting ErrorContext to dictionary."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
            metadata={"key": "value"},
        )

        context_dict = context.to_dict()

        assert context_dict["operation"] == "test_operation"
        assert context_dict["component"] == "test_component"
        assert context_dict["severity"] == "medium"
        assert context_dict["category"] == "data_validation"
        assert context_dict["metadata"]["key"] == "value"
        assert "timestamp" in context_dict

    def test_from_dict(self):
        """Test creating ErrorContext from dictionary."""
        context_dict = {
            "operation": "test_operation",
            "component": "test_component",
            "severity": "medium",
            "category": "data_validation",
            "metadata": {"key": "value"},
            "timestamp": datetime.now().isoformat(),
        }

        context = ErrorContext.from_dict(context_dict)

        assert context.operation == "test_operation"
        assert context.component == "test_component"
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.category == ErrorCategory.DATA_VALIDATION
        assert context.metadata["key"] == "value"


class TestErrorRecord:
    """Test ErrorRecord class."""

    def test_initialization(self):
        """Test ErrorRecord initialization."""
        error = ValueError("Test error")
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
        )

        record = ErrorRecord(
            error=error, context=context, handled=True, recovered=False
        )

        assert record.error == error
        assert record.context == context
        assert record.handled is True
        assert record.recovered is False
        assert record.timestamp is not None
        assert record.id is not None

    def test_initialization_with_custom_id(self):
        """Test ErrorRecord initialization with custom ID."""
        error = ValueError("Test error")
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
        )

        custom_id = "custom_error_id"
        record = ErrorRecord(
            error=error, context=context, handled=True, recovered=False, id=custom_id
        )

        assert record.id == custom_id

    def test_to_dict(self):
        """Test converting ErrorRecord to dictionary."""
        error = ValueError("Test error")
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
        )

        record = ErrorRecord(
            error=error, context=context, handled=True, recovered=False
        )

        record_dict = record.to_dict()

        assert record_dict["error_type"] == "ValueError"
        assert record_dict["error_message"] == "Test error"
        assert record_dict["operation"] == "test_operation"
        assert record_dict["component"] == "test_component"
        assert record_dict["severity"] == "medium"
        assert record_dict["category"] == "data_validation"
        assert record_dict["handled"] is True
        assert record_dict["recovered"] is False
        assert "timestamp" in record_dict
        assert "id" in record_dict

    def test_from_dict(self):
        """Test creating ErrorRecord from dictionary."""
        record_dict = {
            "error_type": "ValueError",
            "error_message": "Test error",
            "operation": "test_operation",
            "component": "test_component",
            "severity": "medium",
            "category": "data_validation",
            "handled": True,
            "recovered": False,
            "timestamp": datetime.now().isoformat(),
            "id": "test_id",
        }

        record = ErrorRecord.from_dict(record_dict)

        assert record.error_type == "ValueError"
        assert record.error_message == "Test error"
        assert record.operation == "test_operation"
        assert record.component == "test_component"
        assert record.severity == ErrorSeverity.MEDIUM
        assert record.category == ErrorCategory.DATA_VALIDATION
        assert record.handled is True
        assert record.recovered is False
        assert record.id == "test_id"


class TestRetryConfig:
    """Test RetryConfig class."""

    def test_initialization(self):
        """Test RetryConfig initialization."""
        config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True,
        )

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 10.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_initialization_with_defaults(self):
        """Test RetryConfig initialization with defaults."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 10.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_validation(self):
        """Test RetryConfig validation."""
        # Valid config
        config = RetryConfig(max_retries=3, base_delay=1.0)
        assert config.validate() is True

        # Invalid config - negative max_retries
        config = RetryConfig(max_retries=-1)
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            config.validate()

        # Invalid config - negative base_delay
        config = RetryConfig(base_delay=-1.0)
        with pytest.raises(ValueError, match="base_delay must be positive"):
            config.validate()

    def test_calculate_delay(self):
        """Test delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0)

        # Test exponential backoff
        delay1 = config.calculate_delay(1)
        delay2 = config.calculate_delay(2)
        delay3 = config.calculate_delay(3)

        assert delay1 == 1.0  # base_delay
        assert delay2 == 2.0  # base_delay * exponential_base
        assert delay3 == 4.0  # base_delay * exponential_base^2

    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(base_delay=1.0, jitter=True)

        delay = config.calculate_delay(1)

        # Delay should be between 0.5 and 1.5 (base_delay ± 50%)
        assert 0.5 <= delay <= 1.5

    def test_calculate_delay_max_delay(self):
        """Test delay calculation with max delay limit."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=2.0)

        delay = config.calculate_delay(5)  # Would be 16.0 without max_delay

        assert delay == 2.0  # Limited by max_delay


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig class."""

    def test_initialization(self):
        """Test CircuitBreakerConfig initialization."""
        config = CircuitBreakerConfig(
            failure_threshold=5, timeout=60.0, expected_exception=ConnectionError
        )

        assert config.failure_threshold == 5
        assert config.timeout == 60.0
        assert config.expected_exception == ConnectionError

    def test_initialization_with_defaults(self):
        """Test CircuitBreakerConfig initialization with defaults."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.timeout == 60.0
        assert config.expected_exception == Exception

    def test_validation(self):
        """Test CircuitBreakerConfig validation."""
        # Valid config
        config = CircuitBreakerConfig(failure_threshold=5, timeout=60.0)
        assert config.validate() is True

        # Invalid config - negative failure_threshold
        config = CircuitBreakerConfig(failure_threshold=-1)
        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            config.validate()

        # Invalid config - negative timeout
        config = CircuitBreakerConfig(timeout=-1.0)
        with pytest.raises(ValueError, match="timeout must be positive"):
            config.validate()


class TestCircuitBreakerState:
    """Test CircuitBreakerState enum."""

    def test_circuit_breaker_state_values(self):
        """Test CircuitBreakerState enum values."""
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"


class TestRetryManager:
    """Test RetryManager class."""

    @pytest.fixture
    def retry_manager(self):
        """Create RetryManager instance."""
        return RetryManager()

    def test_initialization(self, retry_manager):
        """Test RetryManager initialization."""
        assert retry_manager.config is not None
        assert retry_manager._retry_count == 0
        assert retry_manager._last_attempt_time is None

    def test_retry_operation_success_first_attempt(self, retry_manager):
        """Test retry operation that succeeds on first attempt."""

        def successful_operation():
            return "success"

        result = retry_manager.retry_operation(successful_operation)

        assert result == "success"
        assert retry_manager._retry_count == 0

    def test_retry_operation_success_after_retries(self, retry_manager):
        """Test retry operation that succeeds after retries."""
        attempt_count = 0

        def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = retry_manager.retry_operation(flaky_operation)

        assert result == "success"
        assert attempt_count == 3
        assert retry_manager._retry_count == 2

    def test_retry_operation_max_retries_exceeded(self, retry_manager):
        """Test retry operation that exceeds max retries."""

        def always_failing_operation():
            raise ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            retry_manager.retry_operation(always_failing_operation)

        assert retry_manager._retry_count == retry_manager.config.max_retries

    def test_retry_operation_with_custom_exception(self, retry_manager):
        """Test retry operation with custom exception handling."""

        def operation_with_custom_exception():
            raise ValueError("Custom error")

        with pytest.raises(ValueError):
            retry_manager.retry_operation(
                operation_with_custom_exception,
                retry_exceptions=(ConnectionError, TimeoutError),
            )

    def test_retry_operation_with_callback(self, retry_manager):
        """Test retry operation with callback."""
        callback_called = False

        def callback(attempt, error):
            nonlocal callback_called
            callback_called = True
            assert attempt > 0
            assert isinstance(error, ConnectionError)

        def failing_operation():
            raise ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            retry_manager.retry_operation(failing_operation, retry_callback=callback)

        assert callback_called is True

    def test_reset_retry_count(self, retry_manager):
        """Test resetting retry count."""
        retry_manager._retry_count = 5

        retry_manager.reset_retry_count()

        assert retry_manager._retry_count == 0

    def test_get_retry_count(self, retry_manager):
        """Test getting retry count."""
        retry_manager._retry_count = 3

        count = retry_manager.get_retry_count()

        assert count == 3

    def test_is_retry_exhausted(self, retry_manager):
        """Test checking if retries are exhausted."""
        retry_manager._retry_count = retry_manager.config.max_retries

        assert retry_manager.is_retry_exhausted() is True

        retry_manager._retry_count = 1

        assert retry_manager.is_retry_exhausted() is False


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create CircuitBreaker instance."""
        return CircuitBreaker()

    def test_initialization(self, circuit_breaker):
        """Test CircuitBreaker initialization."""
        assert circuit_breaker.config is not None
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.last_failure_time is None

    def test_call_success(self, circuit_breaker):
        """Test successful circuit breaker call."""

        def successful_operation():
            return "success"

        result = circuit_breaker.call(successful_operation)

        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_call_failure_below_threshold(self, circuit_breaker):
        """Test circuit breaker call with failures below threshold."""

        def failing_operation():
            raise ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            circuit_breaker.call(failing_operation)

        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 1

    def test_call_failure_above_threshold(self, circuit_breaker):
        """Test circuit breaker call with failures above threshold."""

        def failing_operation():
            raise ConnectionError("Connection failed")

        # Trigger failures to open circuit breaker
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ConnectionError):
                circuit_breaker.call(failing_operation)

        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count == circuit_breaker.config.failure_threshold

    def test_call_circuit_open(self, circuit_breaker):
        """Test circuit breaker call when circuit is open."""
        # Open the circuit breaker
        circuit_breaker.state = CircuitBreakerState.OPEN
        circuit_breaker.last_failure_time = datetime.now()

        def any_operation():
            return "should not be called"

        with pytest.raises(Exception, match="Circuit breaker is open"):
            circuit_breaker.call(any_operation)

    def test_call_circuit_half_open(self, circuit_breaker):
        """Test circuit breaker call when circuit is half-open."""
        # Set circuit to half-open
        circuit_breaker.state = CircuitBreakerState.HALF_OPEN

        def successful_operation():
            return "success"

        result = circuit_breaker.call(successful_operation)

        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_call_circuit_half_open_failure(self, circuit_breaker):
        """Test circuit breaker call when circuit is half-open and fails."""
        # Set circuit to half-open
        circuit_breaker.state = CircuitBreakerState.HALF_OPEN

        def failing_operation():
            raise ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            circuit_breaker.call(failing_operation)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

    def test_reset_circuit_breaker(self, circuit_breaker):
        """Test resetting circuit breaker."""
        # Open the circuit breaker
        circuit_breaker.state = CircuitBreakerState.OPEN
        circuit_breaker.failure_count = 5

        circuit_breaker.reset()

        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_get_state(self, circuit_breaker):
        """Test getting circuit breaker state."""
        state = circuit_breaker.get_state()

        assert state["state"] == "closed"
        assert state["failure_count"] == 0
        assert state["failure_threshold"] == circuit_breaker.config.failure_threshold
        assert state["timeout"] == circuit_breaker.config.timeout

    def test_is_circuit_open(self, circuit_breaker):
        """Test checking if circuit is open."""
        assert circuit_breaker.is_circuit_open() is False

        circuit_breaker.state = CircuitBreakerState.OPEN

        assert circuit_breaker.is_circuit_open() is True

    def test_is_circuit_closed(self, circuit_breaker):
        """Test checking if circuit is closed."""
        assert circuit_breaker.is_circuit_closed() is True

        circuit_breaker.state = CircuitBreakerState.OPEN

        assert circuit_breaker.is_circuit_closed() is False

    def test_is_circuit_half_open(self, circuit_breaker):
        """Test checking if circuit is half-open."""
        assert circuit_breaker.is_circuit_half_open() is False

        circuit_breaker.state = CircuitBreakerState.HALF_OPEN

        assert circuit_breaker.is_circuit_half_open() is True


class TestErrorHandler:
    """Test ErrorHandler class."""

    @pytest.fixture
    def error_handler(self):
        """Create ErrorHandler instance."""
        return ErrorHandler()

    def test_initialization(self, error_handler):
        """Test ErrorHandler initialization."""
        assert error_handler.retry_manager is not None
        assert error_handler.circuit_breaker is not None
        assert error_handler.health_monitor is not None
        assert error_handler._error_records == []

    def test_handle_error_success(self, error_handler):
        """Test successful error handling."""
        error = ValueError("Test error")
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
        )

        with patch.object(error_handler, "_log_error") as mock_log:
            with patch.object(error_handler, "_record_metrics") as mock_metrics:
                result = error_handler.handle_error(error, context)

                assert result is not None
                assert result.error == error
                assert result.context == context
                assert result.timestamp is not None
                assert result.handled is True

                mock_log.assert_called_once()
                mock_metrics.assert_called_once()

    def test_handle_error_with_recovery(self, error_handler):
        """Test error handling with recovery strategy."""
        error = ConnectionError("Connection failed")
        context = ErrorContext(
            operation="database_connection",
            component="database",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        with patch.object(error_handler, "_attempt_recovery") as mock_recovery:
            mock_recovery.return_value = True

            result = error_handler.handle_error(error, context)

            assert result.recovered is True
            mock_recovery.assert_called_once_with(error, context)

    def test_handle_error_circuit_breaker_open(self, error_handler):
        """Test error handling when circuit breaker is open."""
        error = ConnectionError("Connection failed")
        context = ErrorContext(
            operation="database_connection",
            component="database",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
        )

        # Simulate circuit breaker being open
        error_handler.circuit_breaker.state = CircuitBreakerState.OPEN

        with patch.object(error_handler, "_log_error") as mock_log:
            result = error_handler.handle_error(error, context)

            assert result.circuit_breaker_open is True
            mock_log.assert_called_once()

    def test_handle_error_with_retry(self, error_handler):
        """Test error handling with retry mechanism."""
        error = TimeoutError("Operation timed out")
        context = ErrorContext(
            operation="api_call",
            component="api_client",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        with patch.object(error_handler, "_retry_operation") as mock_retry:
            mock_retry.return_value = True

            result = error_handler.handle_error(error, context)

            assert result.retried is True
            mock_retry.assert_called_once_with(error, context)

    def test_log_error(self, error_handler):
        """Test error logging functionality."""
        error = ValueError("Test error")
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
        )

        with patch("ai_engine.core.error_handling.logging.getLogger") as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log

            error_handler._log_error(error, context)

            mock_log.error.assert_called_once()
            call_args = mock_log.error.call_args[0][0]
            assert "Test error" in call_args
            assert "test_operation" in call_args

    def test_record_metrics(self, error_handler):
        """Test metrics recording functionality."""
        error = ValueError("Test error")
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
        )

        with patch.object(error_handler.health_monitor, "record_error") as mock_record:
            error_handler._record_metrics(error, context)

            mock_record.assert_called_once_with(
                error_type=type(error).__name__,
                component=context.component,
                severity=context.severity.value,
                category=context.category.value,
            )

    def test_attempt_recovery_success(self, error_handler):
        """Test successful recovery attempt."""
        error = ConnectionError("Connection failed")
        context = ErrorContext(
            operation="database_connection",
            component="database",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        with patch.object(error_handler.retry_manager, "retry_operation") as mock_retry:
            mock_retry.return_value = True

            result = error_handler._attempt_recovery(error, context)

            assert result is True
            mock_retry.assert_called_once_with(error, context)

    def test_attempt_recovery_failure(self, error_handler):
        """Test failed recovery attempt."""
        error = ConnectionError("Connection failed")
        context = ErrorContext(
            operation="database_connection",
            component="database",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        with patch.object(error_handler.retry_manager, "retry_operation") as mock_retry:
            mock_retry.return_value = False

            result = error_handler._attempt_recovery(error, context)

            assert result is False

    def test_retry_operation_success(self, error_handler):
        """Test successful retry operation."""
        error = TimeoutError("Operation timed out")
        context = ErrorContext(
            operation="api_call",
            component="api_client",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        with patch.object(error_handler.retry_manager, "retry_operation") as mock_retry:
            mock_retry.return_value = True

            result = error_handler._retry_operation(error, context)

            assert result is True
            mock_retry.assert_called_once_with(error, context)

    def test_retry_operation_failure(self, error_handler):
        """Test failed retry operation."""
        error = TimeoutError("Operation timed out")
        context = ErrorContext(
            operation="api_call",
            component="api_client",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        with patch.object(error_handler.retry_manager, "retry_operation") as mock_retry:
            mock_retry.return_value = False

            result = error_handler._retry_operation(error, context)

            assert result is False

    def test_get_error_records(self, error_handler):
        """Test getting error records."""
        error = ValueError("Test error")
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
        )

        error_handler.handle_error(error, context)

        records = error_handler.get_error_records()

        assert len(records) == 1
        assert records[0].error == error
        assert records[0].context == context

    def test_clear_error_records(self, error_handler):
        """Test clearing error records."""
        error = ValueError("Test error")
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_VALIDATION,
        )

        error_handler.handle_error(error, context)
        assert len(error_handler._error_records) == 1

        error_handler.clear_error_records()
        assert len(error_handler._error_records) == 0

    def test_get_error_statistics(self, error_handler):
        """Test getting error statistics."""
        # Generate some errors
        for _ in range(5):
            error = ValueError("Test error")
            context = ErrorContext(
                operation="test_operation",
                component="test_component",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.DATA_VALIDATION,
            )

            error_handler.handle_error(error, context)

        stats = error_handler.get_error_statistics()

        assert stats["total_errors"] == 5
        assert stats["error_types"]["ValueError"] == 5
        assert stats["components"]["test_component"] == 5
        assert stats["severities"]["medium"] == 5
        assert stats["categories"]["data_validation"] == 5


class TestHealthMonitor:
    """Test HealthMonitor class."""

    @pytest.fixture
    def health_monitor(self):
        """Create HealthMonitor instance."""
        return HealthMonitor()

    def test_initialization(self, health_monitor):
        """Test HealthMonitor initialization."""
        assert health_monitor._error_counts == {}
        assert health_monitor._error_rates == {}
        assert health_monitor._last_reset is not None

    def test_record_error(self, health_monitor):
        """Test recording error metrics."""
        health_monitor.record_error(
            error_type="ValueError",
            component="test_component",
            severity="medium",
            category="data_validation",
        )

        key = ("ValueError", "test_component", "medium", "data_validation")
        assert key in health_monitor._error_counts
        assert health_monitor._error_counts[key] == 1

    def test_record_error_multiple(self, health_monitor):
        """Test recording multiple errors."""
        for _ in range(5):
            health_monitor.record_error(
                error_type="ValueError",
                component="test_component",
                severity="medium",
                category="data_validation",
            )

        key = ("ValueError", "test_component", "medium", "data_validation")
        assert health_monitor._error_counts[key] == 5

    def test_get_error_statistics(self, health_monitor):
        """Test getting error statistics."""
        # Record some errors
        health_monitor.record_error(
            "ValueError", "component1", "medium", "data_validation"
        )
        health_monitor.record_error(
            "ValueError", "component1", "medium", "data_validation"
        )
        health_monitor.record_error(
            "TypeError", "component2", "high", "model_inference"
        )

        stats = health_monitor.get_error_statistics()

        assert stats["total_errors"] == 3
        assert stats["error_types"]["ValueError"] == 2
        assert stats["error_types"]["TypeError"] == 1
        assert stats["components"]["component1"] == 2
        assert stats["components"]["component2"] == 1

    def test_reset_metrics(self, health_monitor):
        """Test resetting metrics."""
        health_monitor.record_error(
            "ValueError", "test_component", "medium", "data_validation"
        )

        assert len(health_monitor._error_counts) > 0

        health_monitor.reset_metrics()

        assert len(health_monitor._error_counts) == 0
        assert health_monitor._last_reset is not None

    def test_calculate_error_rate(self, health_monitor):
        """Test calculating error rate."""
        # Record errors over time
        for _ in range(10):
            health_monitor.record_error(
                "ValueError", "test_component", "medium", "data_validation"
            )

        rate = health_monitor.calculate_error_rate("ValueError", "test_component")

        assert rate > 0
        assert rate <= 1.0

    def test_get_health_status(self, health_monitor):
        """Test getting health status."""
        # Record some errors
        health_monitor.record_error(
            "ValueError", "test_component", "medium", "data_validation"
        )

        status = health_monitor.get_health_status()

        assert status["status"] == "healthy"
        assert status["total_errors"] == 1
        assert status["error_rate"] > 0

    def test_get_health_status_unhealthy(self, health_monitor):
        """Test getting health status when unhealthy."""
        # Record many errors to make it unhealthy
        for _ in range(100):
            health_monitor.record_error(
                "ValueError", "test_component", "medium", "data_validation"
            )

        status = health_monitor.get_health_status()

        assert status["status"] == "unhealthy"
        assert status["total_errors"] == 100


class TestErrorHandlingIntegration:
    """Integration tests for error handling system."""

    @pytest.mark.asyncio
    async def test_full_error_handling_flow(self):
        """Test complete error handling flow."""
        error_handler = ErrorHandler()

        # Test error handling
        error = ConnectionError("Connection failed")
        context = ErrorContext(
            operation="database_connection",
            component="database",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        result = error_handler.handle_error(error, context)

        assert result is not None
        assert result.error == error
        assert result.context == context
        assert result.handled is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with error handling."""
        error_handler = ErrorHandler()

        # Simulate multiple failures to open circuit breaker
        for _ in range(error_handler.circuit_breaker.config.failure_threshold):
            error = ConnectionError("Connection failed")
            context = ErrorContext(
                operation="database_connection",
                component="database",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.NETWORK,
            )

            result = error_handler.handle_error(error, context)

            if result.circuit_breaker_open:
                break

        # Verify circuit breaker is open
        assert error_handler.circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_metrics_integration(self):
        """Test metrics integration with error handling."""
        error_handler = ErrorHandler()

        # Generate some errors
        for _ in range(5):
            error = ValueError("Test error")
            context = ErrorContext(
                operation="test_operation",
                component="test_component",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.DATA_VALIDATION,
            )

            error_handler.handle_error(error, context)

        # Check metrics
        stats = error_handler.get_error_statistics()

        assert stats["total_errors"] == 5
        assert stats["error_types"]["ValueError"] == 5
        assert stats["components"]["test_component"] == 5

    @pytest.mark.asyncio
    async def test_retry_integration(self):
        """Test retry integration with error handling."""
        error_handler = ErrorHandler()

        # Test retry mechanism
        error = TimeoutError("Operation timed out")
        context = ErrorContext(
            operation="api_call",
            component="api_client",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        result = error_handler.handle_error(error, context)

        assert result.retried is True
        assert result.recovered is False  # Retry will fail for this test
