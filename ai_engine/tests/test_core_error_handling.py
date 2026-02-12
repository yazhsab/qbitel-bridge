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
    ErrorHandler,
    RetryManager,
    ErrorLogger,
    ErrorMetrics,
    CircuitBreaker,
    RetryManager,
    ErrorContext,
    ErrorSeverity,
    ErrorCategory,
    ErrorHandlerConfig,
    RecoveryStrategy,
    ErrorHandlerException,
)


class TestErrorHandler:
    """Test ErrorHandler class."""

    @pytest.fixture
    def error_handler(self):
        """Create ErrorHandler instance."""
        config = ErrorHandlerConfig(
            max_retries=3,
            retry_delay=1.0,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=60.0,
            enable_metrics=True,
            log_level=logging.ERROR,
        )
        return ErrorHandler(config)

    def test_initialization(self, error_handler):
        """Test ErrorHandler initialization."""
        assert error_handler.config.max_retries == 3
        assert error_handler.config.retry_delay == 1.0
        assert error_handler.config.circuit_breaker_threshold == 5
        assert error_handler.config.circuit_breaker_timeout == 60.0
        assert error_handler.config.enable_metrics is True
        assert error_handler.config.log_level == logging.ERROR

    def test_handle_error_success(self, error_handler):
        """Test successful error handling."""
        error = ValueError("Test error")
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
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
            category=ErrorCategory.CONNECTION,
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
            category=ErrorCategory.CONNECTION,
        )

        # Simulate circuit breaker being open
        error_handler.circuit_breaker.state = "open"

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
            category=ErrorCategory.TIMEOUT,
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
            category=ErrorCategory.VALIDATION,
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
            category=ErrorCategory.VALIDATION,
        )

        with patch.object(error_handler.metrics, "record_error") as mock_record:
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
            category=ErrorCategory.CONNECTION,
            recovery_strategy=RecoveryStrategy.RECONNECT,
        )

        with patch.object(error_handler.recovery_manager, "attempt_recovery") as mock_recovery:
            mock_recovery.return_value = True

            result = error_handler._attempt_recovery(error, context)

            assert result is True
            mock_recovery.assert_called_once_with(error, context)

    def test_attempt_recovery_failure(self, error_handler):
        """Test failed recovery attempt."""
        error = ConnectionError("Connection failed")
        context = ErrorContext(
            operation="database_connection",
            component="database",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONNECTION,
            recovery_strategy=RecoveryStrategy.RECONNECT,
        )

        with patch.object(error_handler.recovery_manager, "attempt_recovery") as mock_recovery:
            mock_recovery.return_value = False

            result = error_handler._attempt_recovery(error, context)

            assert result is False

    def test_retry_operation_success(self, error_handler):
        """Test successful retry operation."""
        error = TimeoutError("Operation timed out")
        context = ErrorContext(
            operation="api_call",
            component="api_client",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.TIMEOUT,
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
            category=ErrorCategory.TIMEOUT,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        with patch.object(error_handler.retry_manager, "retry_operation") as mock_retry:
            mock_retry.return_value = False

            result = error_handler._retry_operation(error, context)

            assert result is False


class TestErrorRecoveryManager:
    """Test ErrorRecoveryManager class."""

    @pytest.fixture
    def recovery_manager(self):
        """Create ErrorRecoveryManager instance."""
        return ErrorRecoveryManager()

    def test_initialization(self, recovery_manager):
        """Test ErrorRecoveryManager initialization."""
        assert recovery_manager.recovery_strategies == {}
        assert recovery_manager.recovery_history == []

    def test_register_recovery_strategy(self, recovery_manager):
        """Test registering recovery strategy."""

        def mock_recovery(error, context):
            return True

        recovery_manager.register_recovery_strategy(ErrorCategory.CONNECTION, RecoveryStrategy.RECONNECT, mock_recovery)

        assert ErrorCategory.CONNECTION in recovery_manager.recovery_strategies
        assert RecoveryStrategy.RECONNECT in recovery_manager.recovery_strategies[ErrorCategory.CONNECTION]

    def test_attempt_recovery_success(self, recovery_manager):
        """Test successful recovery attempt."""

        def mock_recovery(error, context):
            return True

        recovery_manager.register_recovery_strategy(ErrorCategory.CONNECTION, RecoveryStrategy.RECONNECT, mock_recovery)

        error = ConnectionError("Connection failed")
        context = ErrorContext(
            operation="database_connection",
            component="database",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONNECTION,
            recovery_strategy=RecoveryStrategy.RECONNECT,
        )

        result = recovery_manager.attempt_recovery(error, context)

        assert result is True
        assert len(recovery_manager.recovery_history) == 1
        assert recovery_manager.recovery_history[0].success is True

    def test_attempt_recovery_failure(self, recovery_manager):
        """Test failed recovery attempt."""

        def mock_recovery(error, context):
            return False

        recovery_manager.register_recovery_strategy(ErrorCategory.CONNECTION, RecoveryStrategy.RECONNECT, mock_recovery)

        error = ConnectionError("Connection failed")
        context = ErrorContext(
            operation="database_connection",
            component="database",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONNECTION,
            recovery_strategy=RecoveryStrategy.RECONNECT,
        )

        result = recovery_manager.attempt_recovery(error, context)

        assert result is False
        assert len(recovery_manager.recovery_history) == 1
        assert recovery_manager.recovery_history[0].success is False

    def test_attempt_recovery_no_strategy(self, recovery_manager):
        """Test recovery attempt with no registered strategy."""
        error = ConnectionError("Connection failed")
        context = ErrorContext(
            operation="database_connection",
            component="database",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONNECTION,
            recovery_strategy=RecoveryStrategy.RECONNECT,
        )

        result = recovery_manager.attempt_recovery(error, context)

        assert result is False

    def test_get_recovery_statistics(self, recovery_manager):
        """Test getting recovery statistics."""
        # Add some recovery history
        recovery_manager.recovery_history = [
            Mock(success=True, timestamp=datetime.now()),
            Mock(success=False, timestamp=datetime.now()),
            Mock(success=True, timestamp=datetime.now()),
        ]

        stats = recovery_manager.get_recovery_statistics()

        assert stats["total_attempts"] == 3
        assert stats["successful_recoveries"] == 2
        assert stats["failed_recoveries"] == 1
        assert stats["success_rate"] == 2 / 3


class TestErrorLogger:
    """Test ErrorLogger class."""

    @pytest.fixture
    def error_logger(self):
        """Create ErrorLogger instance."""
        return ErrorLogger()

    def test_initialization(self, error_logger):
        """Test ErrorLogger initialization."""
        assert error_logger.logger is not None
        assert error_logger.log_formatter is not None

    def test_log_error(self, error_logger):
        """Test error logging."""
        error = ValueError("Test error")
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
        )

        with patch.object(error_logger.logger, "error") as mock_log:
            error_logger.log_error(error, context)

            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "Test error" in call_args
            assert "test_operation" in call_args

    def test_log_error_with_traceback(self, error_logger):
        """Test error logging with traceback."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            error = e
            context = ErrorContext(
                operation="test_operation",
                component="test_component",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.VALIDATION,
            )

            with patch.object(error_logger.logger, "error") as mock_log:
                error_logger.log_error(error, context, include_traceback=True)

                mock_log.assert_called_once()
                call_args = mock_log.call_args[0][0]
                assert "Test error" in call_args
                assert "Traceback" in call_args

    def test_log_error_structured(self, error_logger):
        """Test structured error logging."""
        error = ValueError("Test error")
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
        )

        with patch.object(error_logger.logger, "error") as mock_log:
            error_logger.log_error_structured(error, context)

            mock_log.assert_called_once()
            # Verify structured logging format
            call_args = mock_log.call_args[0][0]
            assert isinstance(call_args, dict)
            assert call_args["error_type"] == "ValueError"
            assert call_args["error_message"] == "Test error"
            assert call_args["operation"] == "test_operation"
            assert call_args["component"] == "test_component"


class TestErrorMetrics:
    """Test ErrorMetrics class."""

    @pytest.fixture
    def error_metrics(self):
        """Create ErrorMetrics instance."""
        return ErrorMetrics()

    def test_initialization(self, error_metrics):
        """Test ErrorMetrics initialization."""
        assert error_metrics.error_counts == {}
        assert error_metrics.error_rates == {}
        assert error_metrics.last_reset is not None

    def test_record_error(self, error_metrics):
        """Test recording error metrics."""
        error_metrics.record_error(
            error_type="ValueError",
            component="test_component",
            severity="medium",
            category="validation",
        )

        key = ("ValueError", "test_component", "medium", "validation")
        assert key in error_metrics.error_counts
        assert error_metrics.error_counts[key] == 1

    def test_record_error_multiple(self, error_metrics):
        """Test recording multiple errors."""
        for _ in range(5):
            error_metrics.record_error(
                error_type="ValueError",
                component="test_component",
                severity="medium",
                category="validation",
            )

        key = ("ValueError", "test_component", "medium", "validation")
        assert error_metrics.error_counts[key] == 5

    def test_get_error_statistics(self, error_metrics):
        """Test getting error statistics."""
        # Record some errors
        error_metrics.record_error("ValueError", "component1", "medium", "validation")
        error_metrics.record_error("ValueError", "component1", "medium", "validation")
        error_metrics.record_error("TypeError", "component2", "high", "runtime")

        stats = error_metrics.get_error_statistics()

        assert stats["total_errors"] == 3
        assert stats["error_types"]["ValueError"] == 2
        assert stats["error_types"]["TypeError"] == 1
        assert stats["components"]["component1"] == 2
        assert stats["components"]["component2"] == 1

    def test_reset_metrics(self, error_metrics):
        """Test resetting metrics."""
        error_metrics.record_error("ValueError", "test_component", "medium", "validation")

        assert len(error_metrics.error_counts) > 0

        error_metrics.reset_metrics()

        assert len(error_metrics.error_counts) == 0
        assert error_metrics.last_reset is not None

    def test_calculate_error_rate(self, error_metrics):
        """Test calculating error rate."""
        # Record errors over time
        for _ in range(10):
            error_metrics.record_error("ValueError", "test_component", "medium", "validation")

        rate = error_metrics.calculate_error_rate("ValueError", "test_component")

        assert rate > 0
        assert rate <= 1.0


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create CircuitBreaker instance."""
        return CircuitBreaker(failure_threshold=5, timeout=60.0, expected_exception=ConnectionError)

    def test_initialization(self, circuit_breaker):
        """Test CircuitBreaker initialization."""
        assert circuit_breaker.failure_threshold == 5
        assert circuit_breaker.timeout == 60.0
        assert circuit_breaker.expected_exception == ConnectionError
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0

    def test_call_success(self, circuit_breaker):
        """Test successful circuit breaker call."""

        def successful_operation():
            return "success"

        result = circuit_breaker.call(successful_operation)

        assert result == "success"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0

    def test_call_failure_below_threshold(self, circuit_breaker):
        """Test circuit breaker call with failures below threshold."""

        def failing_operation():
            raise ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            circuit_breaker.call(failing_operation)

        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 1

    def test_call_failure_above_threshold(self, circuit_breaker):
        """Test circuit breaker call with failures above threshold."""

        def failing_operation():
            raise ConnectionError("Connection failed")

        # Trigger failures to open circuit breaker
        for _ in range(5):
            with pytest.raises(ConnectionError):
                circuit_breaker.call(failing_operation)

        assert circuit_breaker.state == "open"
        assert circuit_breaker.failure_count == 5

    def test_call_circuit_open(self, circuit_breaker):
        """Test circuit breaker call when circuit is open."""
        # Open the circuit breaker
        circuit_breaker.state = "open"
        circuit_breaker.last_failure_time = datetime.now()

        def any_operation():
            return "should not be called"

        with pytest.raises(Exception, match="Circuit breaker is open"):
            circuit_breaker.call(any_operation)

    def test_call_circuit_half_open(self, circuit_breaker):
        """Test circuit breaker call when circuit is half-open."""
        # Set circuit to half-open
        circuit_breaker.state = "half-open"

        def successful_operation():
            return "success"

        result = circuit_breaker.call(successful_operation)

        assert result == "success"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0

    def test_call_circuit_half_open_failure(self, circuit_breaker):
        """Test circuit breaker call when circuit is half-open and fails."""
        # Set circuit to half-open
        circuit_breaker.state = "half-open"

        def failing_operation():
            raise ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            circuit_breaker.call(failing_operation)

        assert circuit_breaker.state == "open"

    def test_reset_circuit_breaker(self, circuit_breaker):
        """Test resetting circuit breaker."""
        # Open the circuit breaker
        circuit_breaker.state = "open"
        circuit_breaker.failure_count = 5

        circuit_breaker.reset()

        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0

    def test_get_state(self, circuit_breaker):
        """Test getting circuit breaker state."""
        state = circuit_breaker.get_state()

        assert state["state"] == "closed"
        assert state["failure_count"] == 0
        assert state["failure_threshold"] == 5
        assert state["timeout"] == 60.0


class TestRetryManager:
    """Test RetryManager class."""

    @pytest.fixture
    def retry_manager(self):
        """Create RetryManager instance."""
        return RetryManager(max_retries=3, base_delay=1.0, max_delay=10.0, exponential_base=2.0)

    def test_initialization(self, retry_manager):
        """Test RetryManager initialization."""
        assert retry_manager.max_retries == 3
        assert retry_manager.base_delay == 1.0
        assert retry_manager.max_delay == 10.0
        assert retry_manager.exponential_base == 2.0

    def test_retry_operation_success_first_attempt(self, retry_manager):
        """Test retry operation that succeeds on first attempt."""

        def successful_operation():
            return "success"

        result = retry_manager.retry_operation(successful_operation)

        assert result == "success"

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

    def test_retry_operation_max_retries_exceeded(self, retry_manager):
        """Test retry operation that exceeds max retries."""

        def always_failing_operation():
            raise ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            retry_manager.retry_operation(always_failing_operation)

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

    def test_calculate_delay(self, retry_manager):
        """Test delay calculation."""
        # Test exponential backoff
        delay1 = retry_manager._calculate_delay(1)
        delay2 = retry_manager._calculate_delay(2)
        delay3 = retry_manager._calculate_delay(3)

        assert delay1 == 1.0  # base_delay
        assert delay2 == 2.0  # base_delay * exponential_base
        assert delay3 == 4.0  # base_delay * exponential_base^2

    def test_calculate_delay_with_jitter(self, retry_manager):
        """Test delay calculation with jitter."""
        delay = retry_manager._calculate_delay(1, jitter=True)

        # Delay should be between 0.5 and 1.5 (base_delay ± 50%)
        assert 0.5 <= delay <= 1.5


class TestErrorContext:
    """Test ErrorContext class."""

    def test_initialization(self):
        """Test ErrorContext initialization."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
        )

        assert context.operation == "test_operation"
        assert context.component == "test_component"
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.category == ErrorCategory.VALIDATION
        assert context.timestamp is not None
        assert context.metadata == {}

    def test_initialization_with_metadata(self):
        """Test ErrorContext initialization with metadata."""
        metadata = {"user_id": "123", "request_id": "abc"}
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            metadata=metadata,
        )

        assert context.metadata == metadata

    def test_initialization_with_recovery_strategy(self):
        """Test ErrorContext initialization with recovery strategy."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        assert context.recovery_strategy == RecoveryStrategy.RETRY

    def test_to_dict(self):
        """Test converting ErrorContext to dictionary."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            metadata={"key": "value"},
        )

        context_dict = context.to_dict()

        assert context_dict["operation"] == "test_operation"
        assert context_dict["component"] == "test_component"
        assert context_dict["severity"] == "medium"
        assert context_dict["category"] == "validation"
        assert context_dict["metadata"]["key"] == "value"
        assert "timestamp" in context_dict

    def test_from_dict(self):
        """Test creating ErrorContext from dictionary."""
        context_dict = {
            "operation": "test_operation",
            "component": "test_component",
            "severity": "medium",
            "category": "validation",
            "metadata": {"key": "value"},
            "timestamp": datetime.now().isoformat(),
        }

        context = ErrorContext.from_dict(context_dict)

        assert context.operation == "test_operation"
        assert context.component == "test_component"
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.category == ErrorCategory.VALIDATION
        assert context.metadata["key"] == "value"


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
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.CONNECTION.value == "connection"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.RUNTIME.value == "runtime"
        assert ErrorCategory.SECURITY.value == "security"
        assert ErrorCategory.RESOURCE.value == "resource"


class TestRecoveryStrategy:
    """Test RecoveryStrategy enum."""

    def test_recovery_strategy_values(self):
        """Test RecoveryStrategy enum values."""
        assert RecoveryStrategy.RETRY.value == "retry"
        assert RecoveryStrategy.RECONNECT.value == "reconnect"
        assert RecoveryStrategy.FALLBACK.value == "fallback"
        assert RecoveryStrategy.IGNORE.value == "ignore"
        assert RecoveryStrategy.ESCALATE.value == "escalate"


class TestErrorHandlerConfig:
    """Test ErrorHandlerConfig class."""

    def test_initialization(self):
        """Test ErrorHandlerConfig initialization."""
        config = ErrorHandlerConfig(
            max_retries=3,
            retry_delay=1.0,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=60.0,
            enable_metrics=True,
            log_level=logging.ERROR,
        )

        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_timeout == 60.0
        assert config.enable_metrics is True
        assert config.log_level == logging.ERROR

    def test_default_values(self):
        """Test ErrorHandlerConfig default values."""
        config = ErrorHandlerConfig()

        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_timeout == 60.0
        assert config.enable_metrics is True
        assert config.log_level == logging.ERROR


class TestErrorHandlerException:
    """Test ErrorHandlerException class."""

    def test_initialization(self):
        """Test ErrorHandlerException initialization."""
        error = ErrorHandlerException("Test error", error_code="TEST_ERROR")

        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"

    def test_initialization_with_context(self):
        """Test ErrorHandlerException initialization with context."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
        )

        error = ErrorHandlerException("Test error", context=context)

        assert str(error) == "Test error"
        assert error.context == context


class TestErrorHandlingIntegration:
    """Integration tests for error handling system."""

    @pytest.mark.asyncio
    async def test_full_error_handling_flow(self):
        """Test complete error handling flow."""
        config = ErrorHandlerConfig(
            max_retries=2,
            retry_delay=0.1,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=1.0,
            enable_metrics=True,
        )

        error_handler = ErrorHandler(config)

        # Register recovery strategy
        def mock_recovery(error, context):
            return True

        error_handler.recovery_manager.register_recovery_strategy(
            ErrorCategory.CONNECTION, RecoveryStrategy.RECONNECT, mock_recovery
        )

        # Test error handling
        error = ConnectionError("Connection failed")
        context = ErrorContext(
            operation="database_connection",
            component="database",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONNECTION,
            recovery_strategy=RecoveryStrategy.RECONNECT,
        )

        result = error_handler.handle_error(error, context)

        assert result is not None
        assert result.error == error
        assert result.context == context
        assert result.handled is True
        assert result.recovered is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with error handling."""
        config = ErrorHandlerConfig(
            max_retries=1,
            retry_delay=0.1,
            circuit_breaker_threshold=2,
            circuit_breaker_timeout=1.0,
        )

        error_handler = ErrorHandler(config)

        # Simulate multiple failures to open circuit breaker
        for _ in range(3):
            error = ConnectionError("Connection failed")
            context = ErrorContext(
                operation="database_connection",
                component="database",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.CONNECTION,
            )

            result = error_handler.handle_error(error, context)

            if result.circuit_breaker_open:
                break

        # Verify circuit breaker is open
        assert error_handler.circuit_breaker.state == "open"

    @pytest.mark.asyncio
    async def test_metrics_integration(self):
        """Test metrics integration with error handling."""
        config = ErrorHandlerConfig(enable_metrics=True)
        error_handler = ErrorHandler(config)

        # Generate some errors
        for _ in range(5):
            error = ValueError("Test error")
            context = ErrorContext(
                operation="test_operation",
                component="test_component",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.VALIDATION,
            )

            error_handler.handle_error(error, context)

        # Check metrics
        stats = error_handler.metrics.get_error_statistics()

        assert stats["total_errors"] == 5
        assert stats["error_types"]["ValueError"] == 5
        assert stats["components"]["test_component"] == 5
