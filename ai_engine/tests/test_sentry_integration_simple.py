"""
QBITEL Engine - Sentry Integration Simple Tests

Simple test suite for Sentry error tracking and monitoring.
"""

import pytest
import os
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from ai_engine.core.sentry_integration import (
    SentryErrorTracker,
)
from ai_engine.core.error_handling import ErrorRecord, ErrorSeverity
from ai_engine.core.exceptions import QbitelAIException


class TestSentryErrorTracker:
    """Test SentryErrorTracker functionality."""

    @pytest.fixture
    def mock_sentry_sdk(self):
        """Mock sentry_sdk module."""
        with patch("ai_engine.core.sentry_integration.sentry_sdk") as mock_sdk:
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
            enable_tracing=True,
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
        with patch.dict(
            os.environ,
            {
                "SENTRY_DSN": "https://env@sentry.io/789",
                "QBITEL_AI_VERSION": "2.0.0-env",
            },
        ):
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
            metadata=None,
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
        breadcrumb = {"message": "Test breadcrumb", "category": "test", "level": "info"}

        sentry_tracker.add_breadcrumb(breadcrumb)

        mock_sentry_sdk.add_breadcrumb.assert_called_once_with(breadcrumb)

    def test_set_user_context(self, sentry_tracker, mock_sentry_sdk):
        """Test setting user context."""
        user_context = {
            "id": "user123",
            "username": "testuser",
            "email": "test@example.com",
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
            metadata={"recovery_info": "successful"},
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
            metadata=None,
        )

        sentry_tracker.capture_error_record(error_record)

        # Verify critical severity was handled
        mock_sentry_sdk.set_tag.assert_called()
        mock_sentry_sdk.capture_message.assert_called()

    def test_initialize_success(self, sentry_tracker, mock_sentry_sdk):
        """Test successful Sentry initialization."""
        result = sentry_tracker.initialize()

        assert result is True
        assert sentry_tracker.initialized is True
        mock_sentry_sdk.init.assert_called()

    def test_initialize_without_dsn(self, mock_sentry_sdk):
        """Test Sentry initialization without DSN."""
        tracker = SentryErrorTracker(dsn=None)

        result = tracker.initialize()

        assert result is False
        assert tracker.initialized is False
        mock_sentry_sdk.init.assert_not_called()

    def test_initialize_with_error(self, mock_sentry_sdk):
        """Test Sentry initialization with error."""
        mock_sentry_sdk.init.side_effect = Exception("Initialization failed")

        tracker = SentryErrorTracker(dsn="https://test@sentry.io/123456")

        result = tracker.initialize()

        assert result is False
        assert tracker.initialized is False

    def test_before_send_filter(self, sentry_tracker):
        """Test before_send filter function."""
        event = {
            "event_id": "test-event-123",
            "level": "error",
            "message": "Test error message",
            "tags": {"component": "test"},
            "extra": {"debug_info": "test"},
        }

        # Test that filter returns the event
        filtered_event = sentry_tracker._before_send(event, None)

        assert filtered_event == event

    def test_before_send_filter_with_exception(self, sentry_tracker):
        """Test before_send filter with exception."""
        event = {
            "event_id": "test-event-456",
            "level": "error",
            "message": "Test error message",
            "exception": {"values": [{"type": "ValueError", "value": "Test error"}]},
        }

        # Test that filter returns the event
        filtered_event = sentry_tracker._before_send(event, None)

        assert filtered_event == event

    def test_before_breadcrumb_filter(self, sentry_tracker):
        """Test before_breadcrumb filter function."""
        breadcrumb = {
            "message": "Test breadcrumb",
            "category": "test",
            "level": "info",
            "timestamp": "2023-01-01T00:00:00Z",
        }

        # Test that filter returns the breadcrumb
        filtered_breadcrumb = sentry_tracker._before_breadcrumb(breadcrumb, None)

        assert filtered_breadcrumb == breadcrumb

    def test_before_breadcrumb_filter_http(self, sentry_tracker):
        """Test before_breadcrumb filter with HTTP breadcrumb."""
        breadcrumb = {
            "message": "HTTP request",
            "category": "http",
            "level": "info",
            "data": {
                "url": "https://example.com/api",
                "method": "GET",
                "status_code": 200,
            },
        }

        # Test that filter returns the breadcrumb
        filtered_breadcrumb = sentry_tracker._before_breadcrumb(breadcrumb, None)

        assert filtered_breadcrumb == breadcrumb

    def test_sentry_integration_workflow(self, mock_sentry_sdk):
        """Test complete Sentry integration workflow."""
        # Initialize tracker
        tracker = SentryErrorTracker(dsn="https://test@sentry.io/123456", environment="testing")

        # Set user context
        tracker.set_user_context({"id": "user123", "username": "testuser"})

        # Add breadcrumb
        tracker.add_breadcrumb({"message": "User performed action", "category": "user_action"})

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
            metadata=None,
        )

        tracker.capture_error_record(error_record)

        # Verify all operations were called
        mock_sentry_sdk.init.assert_called_once()
        mock_sentry_sdk.set_user.assert_called_once()
        mock_sentry_sdk.add_breadcrumb.assert_called_once()
        mock_sentry_sdk.set_tag.assert_called()
        mock_sentry_sdk.set_context.assert_called()
        mock_sentry_sdk.capture_message.assert_called_once()

    def test_sentry_error_handling(self):
        """Test Sentry error handling when Sentry is not available."""
        with patch(
            "ai_engine.core.sentry_integration.sentry_sdk",
            side_effect=ImportError("Sentry not available"),
        ):
            # Should not raise exception when Sentry is not available
            tracker = SentryErrorTracker()

            # Operations should not fail
            tracker.capture_message("Test message")
            tracker.set_tag("test", "value")
            tracker.flush()

    def test_sentry_configuration_validation(self, mock_sentry_sdk):
        """Test Sentry configuration validation."""
        # Test with invalid DSN
        tracker = SentryErrorTracker(dsn="invalid-dsn")

        # Should still initialize but with warnings
        assert tracker.dsn == "invalid-dsn"

        # Test with invalid sample rates
        tracker = SentryErrorTracker(
            dsn="https://test@sentry.io/123456",
            traces_sample_rate=1.5,  # Invalid: > 1.0
            profiles_sample_rate=-0.1,  # Invalid: < 0.0
        )

        # Should clamp to valid ranges
        assert tracker.traces_sample_rate == 1.0
        assert tracker.profiles_sample_rate == 0.0

    def test_sentry_tracker_context_manager(self, sentry_tracker, mock_sentry_sdk):
        """Test Sentry tracker as context manager."""
        with sentry_tracker as tracker:
            tracker.set_tag("context", "test")
            tracker.capture_message("Context test message")

        # Verify operations were called
        mock_sentry_sdk.set_tag.assert_called()
        mock_sentry_sdk.capture_message.assert_called()

    def test_sentry_tracker_performance_tracking(self, sentry_tracker, mock_sentry_sdk):
        """Test Sentry performance tracking."""
        with sentry_tracker.start_transaction("test_transaction", "test_operation") as transaction:
            with sentry_tracker.start_span(transaction, "test_span", "test_operation") as span:
                span.set_tag("test", "value")
                span.set_data("test_data", "value")

        # Verify transaction and span operations
        mock_sentry_sdk.set_tag.assert_called()
        mock_sentry_sdk.set_context.assert_called()

    def test_sentry_tracker_custom_fingerprint(self, sentry_tracker, mock_sentry_sdk):
        """Test Sentry custom fingerprint."""
        error_record = ErrorRecord(
            error_id="test-error-fingerprint",
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
            metadata=None,
        )

        # Set custom fingerprint
        sentry_tracker.set_custom_fingerprint(["test", "fingerprint"])
        sentry_tracker.capture_error_record(error_record)

        # Verify fingerprint was set
        mock_sentry_sdk.set_tag.assert_called()
        mock_sentry_sdk.capture_message.assert_called()
