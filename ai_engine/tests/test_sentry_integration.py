"""
CRONOS AI Engine - Sentry Integration Tests

Comprehensive test suite for Sentry error tracking and monitoring.
"""

import pytest
import os
import logging
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, Optional

from ai_engine.core.sentry_integration import (
    SentryErrorTracker,
    SentryPerformanceTracker,
    SentryUserTracker,
    SentryContextManager,
)
from ai_engine.core.error_handling import ErrorRecord, ErrorSeverity
from ai_engine.core.exceptions import CronosAIException


class TestSentryErrorTracker:
    """Test SentryErrorTracker functionality."""

    @pytest.fixture
    def mock_sentry_sdk(self):
        """Mock sentry_sdk module."""
        with patch('ai_engine.core.sentry_integration.sentry_sdk') as mock_sdk:
            mock_sdk.init = Mock()
            mock_sdk.capture_exception = Mock()
            mock_sdk.capture_message = Mock()
            mock_sdk.add_breadcrumb = Mock()
            mock_sdk.set_tag = Mock()
            mock_sdk.set_context = Mock()
            mock_sdk.set_user = Mock()
            mock_sdk.set_level = Mock()
            mock_sdk.push_scope = Mock()
            mock_sdk.pop_scope = Mock()
            mock_sdk.configure_scope = Mock()
            mock_sdk.get_current_hub = Mock()
            mock_sdk.flush = Mock()
            yield mock_sdk

    @pytest.fixture
    def sentry_tracker(self, mock_sentry_sdk):
        """Create SentryErrorTracker instance."""
        return SentryErrorTracker(
            dsn="https://test@sentry.io/123456",
            environment="testing",
            release="1.0.0-test",
            traces_sample_rate=0.1,
            profiles_sample_rate=0.1,
            enable_tracing=True
        )

    def test_sentry_tracker_initialization(self, sentry_tracker, mock_sentry_sdk):
        """Test SentryErrorTracker initialization."""
        assert sentry_tracker.dsn == "https://test@sentry.io/123456"
        assert sentry_tracker.environment == "testing"
        assert sentry_tracker.release == "1.0.0-test"
        assert sentry_tracker.traces_sample_rate == 0.1
        assert sentry_tracker.profiles_sample_rate == 0.1
        assert sentry_tracker.enable_tracing is True
        
        # Verify Sentry was initialized
        mock_sentry_sdk.init.assert_called_once()

    def test_sentry_tracker_initialization_with_env_vars(self, mock_sentry_sdk):
        """Test SentryErrorTracker initialization with environment variables."""
        with patch.dict(os.environ, {
            'SENTRY_DSN': 'https://env@sentry.io/789',
            'CRONOS_AI_VERSION': '2.0.0-env'
        }):
            tracker = SentryErrorTracker()
            
            assert tracker.dsn == "https://env@sentry.io/789"
            assert tracker.release == "2.0.0-env"

    def test_sentry_tracker_initialization_without_dsn(self, mock_sentry_sdk):
        """Test SentryErrorTracker initialization without DSN."""
        with patch.dict(os.environ, {}, clear=True):
            tracker = SentryErrorTracker()
            
            assert tracker.dsn is None
            # Should not initialize Sentry without DSN
            mock_sentry_sdk.init.assert_not_called()

    def test_capture_error_record(self, sentry_tracker, mock_sentry_sdk):
        """Test capturing error record."""
        error_record = ErrorRecord(
            error_id="test-error-123",
            timestamp=None,
            severity=ErrorSeverity.ERROR,
            category=None,
            component="test_component",
            operation="test_operation",
            exception_type="ValueError",
            exception_message="Test error message",
            stack_trace="Traceback...",
            context={"key": "value"},
            recovery_attempted=False,
            recovery_successful=False,
            recovery_strategy=None,
            retry_count=0,
            metadata=None
        )
        
        sentry_tracker.capture_error_record(error_record)
        
        # Verify Sentry operations were called
        mock_sentry_sdk.set_tag.assert_called()
        mock_sentry_sdk.set_context.assert_called()
        mock_sentry_sdk.capture_message.assert_called()

    def test_capture_exception(self, sentry_tracker, mock_sentry_sdk):
        """Test capturing exception."""
        exception = ValueError("Test exception")
        
        sentry_tracker.capture_exception(exception)
        
        mock_sentry_sdk.capture_exception.assert_called_once_with(exception)

    def test_capture_message(self, sentry_tracker, mock_sentry_sdk):
        """Test capturing message."""
        message = "Test message"
        level = "error"
        
        sentry_tracker.capture_message(message, level)
        
        mock_sentry_sdk.capture_message.assert_called_once_with(message, level=level)

    def test_add_breadcrumb(self, sentry_tracker, mock_sentry_sdk):
        """Test adding breadcrumb."""
        breadcrumb = {
            "message": "Test breadcrumb",
            "category": "test",
            "level": "info"
        }
        
        sentry_tracker.add_breadcrumb(breadcrumb)
        
        mock_sentry_sdk.add_breadcrumb.assert_called_once_with(breadcrumb)

    def test_set_user_context(self, sentry_tracker, mock_sentry_sdk):
        """Test setting user context."""
        user_context = {
            "id": "user123",
            "username": "testuser",
            "email": "test@example.com"
        }
        
        sentry_tracker.set_user_context(user_context)
        
        mock_sentry_sdk.set_user.assert_called_once_with(user_context)

    def test_set_custom_context(self, sentry_tracker, mock_sentry_sdk):
        """Test setting custom context."""
        context_name = "custom_context"
        context_data = {"key": "value"}
        
        sentry_tracker.set_custom_context(context_name, context_data)
        
        mock_sentry_sdk.set_context.assert_called_once_with(context_name, context_data)

    def test_set_tag(self, sentry_tracker, mock_sentry_sdk):
        """Test setting tag."""
        key = "test_tag"
        value = "test_value"
        
        sentry_tracker.set_tag(key, value)
        
        mock_sentry_sdk.set_tag.assert_called_once_with(key, value)

    def test_flush(self, sentry_tracker, mock_sentry_sdk):
        """Test flushing Sentry."""
        sentry_tracker.flush()
        
        mock_sentry_sdk.flush.assert_called_once()

    def test_capture_error_record_with_recovery(self, sentry_tracker, mock_sentry_sdk):
        """Test capturing error record with recovery information."""
        error_record = ErrorRecord(
            error_id="test-error-recovery",
            timestamp=None,
            severity=ErrorSeverity.ERROR,
            category=None,
            component="test_component",
            operation="test_operation",
            exception_type="ValueError",
            exception_message="Test error with recovery",
            stack_trace="Traceback...",
            context={"key": "value"},
            recovery_attempted=True,
            recovery_successful=True,
            recovery_strategy=None,
            retry_count=3,
            metadata={"recovery_info": "successful"}
        )
        
        sentry_tracker.capture_error_record(error_record)
        
        # Verify recovery information was set
        mock_sentry_sdk.set_tag.assert_called()
        mock_sentry_sdk.set_context.assert_called()

    def test_capture_error_record_critical_severity(self, sentry_tracker, mock_sentry_sdk):
        """Test capturing error record with critical severity."""
        error_record = ErrorRecord(
            error_id="test-error-critical",
            timestamp=None,
            severity=ErrorSeverity.CRITICAL,
            category=None,
            component="critical_component",
            operation="critical_operation",
            exception_type="SystemError",
            exception_message="Critical system error",
            stack_trace="Traceback...",
            context={"critical": True},
            recovery_attempted=False,
            recovery_successful=False,
            recovery_strategy=None,
            retry_count=0,
            metadata=None
        )
        
        sentry_tracker.capture_error_record(error_record)
        
        # Verify critical severity was handled
        mock_sentry_sdk.set_tag.assert_called()
        mock_sentry_sdk.capture_message.assert_called()


class TestSentryPerformanceTracker:
    """Test SentryPerformanceTracker functionality."""

    @pytest.fixture
    def mock_sentry_sdk(self):
        """Mock sentry_sdk module."""
        with patch('ai_engine.core.sentry_integration.sentry_sdk') as mock_sdk:
            mock_sdk.start_transaction = Mock()
            mock_sdk.start_span = Mock()
            mock_sdk.set_measurement = Mock()
            mock_sdk.set_tag = Mock()
            mock_sdk.set_context = Mock()
            yield mock_sdk

    @pytest.fixture
    def performance_tracker(self, mock_sentry_sdk):
        """Create SentryPerformanceTracker instance."""
        return SentryPerformanceTracker()

    def test_start_transaction(self, performance_tracker, mock_sentry_sdk):
        """Test starting a transaction."""
        mock_transaction = Mock()
        mock_sentry_sdk.start_transaction.return_value = mock_transaction
        
        result = performance_tracker.start_transaction(
            name="test_transaction",
            op="test_operation"
        )
        
        assert result == mock_transaction
        mock_sentry_sdk.start_transaction.assert_called_once_with(
            name="test_transaction",
            op="test_operation"
        )

    def test_start_span(self, performance_tracker, mock_sentry_sdk):
        """Test starting a span."""
        mock_span = Mock()
        mock_sentry_sdk.start_span.return_value = mock_span
        
        result = performance_tracker.start_span(
            transaction=Mock(),
            name="test_span",
            op="test_operation"
        )
        
        assert result == mock_span
        mock_sentry_sdk.start_span.assert_called_once()

    def test_set_measurement(self, performance_tracker, mock_sentry_sdk):
        """Test setting measurement."""
        performance_tracker.set_measurement("test_metric", 123.45)
        
        mock_sentry_sdk.set_measurement.assert_called_once_with("test_metric", 123.45)

    def test_track_function_performance(self, performance_tracker, mock_sentry_sdk):
        """Test tracking function performance."""
        mock_transaction = Mock()
        mock_span = Mock()
        mock_sentry_sdk.start_transaction.return_value = mock_transaction
        mock_sentry_sdk.start_span.return_value = mock_span
        
        def test_function():
            return "test_result"
        
        result = performance_tracker.track_function_performance(
            test_function,
            transaction_name="test_transaction",
            span_name="test_span"
        )
        
        assert result == "test_result"
        mock_transaction.finish.assert_called_once()
        mock_span.finish.assert_called_once()

    def test_track_async_function_performance(self, performance_tracker, mock_sentry_sdk):
        """Test tracking async function performance."""
        mock_transaction = Mock()
        mock_span = Mock()
        mock_sentry_sdk.start_transaction.return_value = mock_transaction
        mock_sentry_sdk.start_span.return_value = mock_span
        
        async def test_async_function():
            return "async_result"
        
        import asyncio
        result = asyncio.run(performance_tracker.track_async_function_performance(
            test_async_function,
            transaction_name="test_async_transaction",
            span_name="test_async_span"
        ))
        
        assert result == "async_result"
        mock_transaction.finish.assert_called_once()
        mock_span.finish.assert_called_once()


class TestSentryUserTracker:
    """Test SentryUserTracker functionality."""

    @pytest.fixture
    def mock_sentry_sdk(self):
        """Mock sentry_sdk module."""
        with patch('ai_engine.core.sentry_integration.sentry_sdk') as mock_sdk:
            mock_sdk.set_user = Mock()
            mock_sdk.set_tag = Mock()
            mock_sdk.set_context = Mock()
            yield mock_sdk

    @pytest.fixture
    def user_tracker(self, mock_sentry_sdk):
        """Create SentryUserTracker instance."""
        return SentryUserTracker()

    def test_set_user_info(self, user_tracker, mock_sentry_sdk):
        """Test setting user information."""
        user_info = {
            "id": "user123",
            "username": "testuser",
            "email": "test@example.com",
            "ip_address": "192.168.1.1"
        }
        
        user_tracker.set_user_info(user_info)
        
        mock_sentry_sdk.set_user.assert_called_once_with(user_info)

    def test_set_user_activity(self, user_tracker, mock_sentry_sdk):
        """Test setting user activity."""
        activity = {
            "action": "login",
            "timestamp": "2023-01-01T00:00:00Z",
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0"
        }
        
        user_tracker.set_user_activity(activity)
        
        mock_sentry_sdk.set_context.assert_called_once_with("user_activity", activity)

    def test_set_user_permissions(self, user_tracker, mock_sentry_sdk):
        """Test setting user permissions."""
        permissions = ["read", "write", "admin"]
        
        user_tracker.set_user_permissions(permissions)
        
        mock_sentry_sdk.set_tag.assert_called_once_with("user_permissions", "read,write,admin")

    def test_clear_user_context(self, user_tracker, mock_sentry_sdk):
        """Test clearing user context."""
        user_tracker.clear_user_context()
        
        mock_sentry_sdk.set_user.assert_called_once_with(None)


class TestSentryContextManager:
    """Test SentryContextManager functionality."""

    @pytest.fixture
    def mock_sentry_sdk(self):
        """Mock sentry_sdk module."""
        with patch('ai_engine.core.sentry_integration.sentry_sdk') as mock_sdk:
            mock_sdk.push_scope = Mock()
            mock_sdk.pop_scope = Mock()
            mock_sdk.configure_scope = Mock()
            mock_sdk.set_tag = Mock()
            mock_sdk.set_context = Mock()
            yield mock_sdk

    @pytest.fixture
    def context_manager(self, mock_sentry_sdk):
        """Create SentryContextManager instance."""
        return SentryContextManager()

    def test_push_scope(self, context_manager, mock_sentry_sdk):
        """Test pushing scope."""
        context_manager.push_scope()
        
        mock_sentry_sdk.push_scope.assert_called_once()

    def test_pop_scope(self, context_manager, mock_sentry_sdk):
        """Test popping scope."""
        context_manager.pop_scope()
        
        mock_sentry_sdk.pop_scope.assert_called_once()

    def test_configure_scope(self, context_manager, mock_sentry_sdk):
        """Test configuring scope."""
        def scope_callback(scope):
            scope.set_tag("test", "value")
        
        context_manager.configure_scope(scope_callback)
        
        mock_sentry_sdk.configure_scope.assert_called_once_with(scope_callback)

    def test_context_manager_usage(self, context_manager, mock_sentry_sdk):
        """Test using context manager."""
        with context_manager as scope:
            scope.set_tag("test", "value")
            scope.set_context("test_context", {"key": "value"})
        
        mock_sentry_sdk.push_scope.assert_called_once()
        mock_sentry_sdk.pop_scope.assert_called_once()

    def test_context_manager_with_exception(self, context_manager, mock_sentry_sdk):
        """Test context manager with exception."""
        with pytest.raises(ValueError):
            with context_manager as scope:
                scope.set_tag("test", "value")
                raise ValueError("Test exception")
        
        mock_sentry_sdk.push_scope.assert_called_once()
        mock_sentry_sdk.pop_scope.assert_called_once()


class TestSentryIntegrationIntegration:
    """Integration tests for Sentry components."""

    @pytest.mark.asyncio
    async def test_sentry_integration_workflow(self):
        """Test complete Sentry integration workflow."""
        with patch('ai_engine.core.sentry_integration.sentry_sdk') as mock_sdk:
            # Initialize tracker
            tracker = SentryErrorTracker(
                dsn="https://test@sentry.io/123456",
                environment="testing"
            )
            
            # Set user context
            tracker.set_user_context({
                "id": "user123",
                "username": "testuser"
            })
            
            # Add breadcrumb
            tracker.add_breadcrumb({
                "message": "User performed action",
                "category": "user_action"
            })
            
            # Capture error
            error_record = ErrorRecord(
                error_id="integration-test-error",
                timestamp=None,
                severity=ErrorSeverity.ERROR,
                category=None,
                component="integration_test",
                operation="test_operation",
                exception_type="IntegrationError",
                exception_message="Integration test error",
                stack_trace="Traceback...",
                context={"test": "integration"},
                recovery_attempted=False,
                recovery_successful=False,
                recovery_strategy=None,
                retry_count=0,
                metadata=None
            )
            
            tracker.capture_error_record(error_record)
            
            # Verify all operations were called
            mock_sdk.init.assert_called_once()
            mock_sdk.set_user.assert_called_once()
            mock_sdk.add_breadcrumb.assert_called_once()
            mock_sdk.set_tag.assert_called()
            mock_sdk.set_context.assert_called()
            mock_sdk.capture_message.assert_called_once()

    def test_sentry_performance_tracking_workflow(self):
        """Test complete performance tracking workflow."""
        with patch('ai_engine.core.sentry_integration.sentry_sdk') as mock_sdk:
            mock_transaction = Mock()
            mock_span = Mock()
            mock_sdk.start_transaction.return_value = mock_transaction
            mock_sdk.start_span.return_value = mock_span
            
            # Initialize performance tracker
            perf_tracker = SentryPerformanceTracker()
            
            # Start transaction
            transaction = perf_tracker.start_transaction(
                name="test_workflow",
                op="workflow"
            )
            
            # Start span
            span = perf_tracker.start_span(
                transaction=transaction,
                name="test_operation",
                op="operation"
            )
            
            # Set measurement
            perf_tracker.set_measurement("operation_duration", 1.5)
            
            # Finish span and transaction
            span.finish()
            transaction.finish()
            
            # Verify operations
            mock_sdk.start_transaction.assert_called_once()
            mock_sdk.start_span.assert_called_once()
            mock_sdk.set_measurement.assert_called_once()
            mock_span.finish.assert_called_once()
            mock_transaction.finish.assert_called_once()

    def test_sentry_error_handling(self):
        """Test Sentry error handling when Sentry is not available."""
        with patch('ai_engine.core.sentry_integration.sentry_sdk', side_effect=ImportError("Sentry not available")):
            # Should not raise exception when Sentry is not available
            tracker = SentryErrorTracker()
            
            # Operations should not fail
            tracker.capture_message("Test message")
            tracker.set_tag("test", "value")
            tracker.flush()

    def test_sentry_configuration_validation(self):
        """Test Sentry configuration validation."""
        with patch('ai_engine.core.sentry_integration.sentry_sdk') as mock_sdk:
            # Test with invalid DSN
            tracker = SentryErrorTracker(dsn="invalid-dsn")
            
            # Should still initialize but with warnings
            assert tracker.dsn == "invalid-dsn"
            
            # Test with invalid sample rates
            tracker = SentryErrorTracker(
                dsn="https://test@sentry.io/123456",
                traces_sample_rate=1.5,  # Invalid: > 1.0
                profiles_sample_rate=-0.1  # Invalid: < 0.0
            )
            
            # Should clamp to valid ranges
            assert tracker.traces_sample_rate == 1.0
            assert tracker.profiles_sample_rate == 0.0
