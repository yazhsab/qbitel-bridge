"""
Basic test suite for sentry_integration.py - tests only what actually exists.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging

from ai_engine.core.sentry_integration import (
    SentryErrorTracker,
    get_sentry_tracker,
    initialize_sentry,
)
from ai_engine.core.error_handling import (
    ErrorRecord,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
)


class TestSentryErrorTracker:
    """Test SentryErrorTracker with basic functionality."""

    def test_initialization_with_dsn(self):
        """Test initialization with DSN."""
        tracker = SentryErrorTracker(
            dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
        )

        assert tracker.dsn == "https://test@sentry.io/123"
        assert tracker.environment == "test"
        assert tracker.release == "1.0.0"
        assert tracker.initialized is False

    def test_initialization_without_dsn(self):
        """Test initialization without DSN."""
        with patch.dict("os.environ", {"SENTRY_DSN": "https://env@sentry.io/456"}):
            tracker = SentryErrorTracker()

            assert tracker.dsn == "https://env@sentry.io/456"
            assert tracker.environment == "production"
            assert tracker.initialized is False

    def test_initialization_with_version_from_env(self):
        """Test initialization with version from environment."""
        with patch.dict("os.environ", {"CRONOS_AI_VERSION": "2.0.0"}):
            tracker = SentryErrorTracker()

            assert tracker.release == "2.0.0"

    def test_initialization_without_dsn_from_env(self):
        """Test initialization without DSN from environment."""
        with patch.dict("os.environ", {}, clear=True):
            tracker = SentryErrorTracker()

            assert tracker.dsn is None
            assert tracker.environment == "production"

    def test_initialize_success(self):
        """Test successful initialization."""
        with patch("ai_engine.core.sentry_integration.sentry_sdk") as mock_sdk:
            tracker = SentryErrorTracker(
                dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
            )

            result = tracker.initialize()

            assert result is True
            assert tracker.initialized is True
            mock_sdk.init.assert_called_once()

    def test_initialize_without_dsn(self):
        """Test initialization without DSN."""
        tracker = SentryErrorTracker(dsn=None)

        result = tracker.initialize()

        assert result is False
        assert tracker.initialized is False

    def test_initialize_failure(self):
        """Test initialization failure."""
        with patch(
            "ai_engine.core.sentry_integration.sentry_sdk.init",
            side_effect=Exception("Init failed"),
        ):
            tracker = SentryErrorTracker(
                dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
            )

            result = tracker.initialize()

            assert result is False
            assert tracker.initialized is False

    def test_capture_exception_not_initialized(self):
        """Test capturing exceptions when not initialized."""
        tracker = SentryErrorTracker()

        exception = ValueError("Test error")
        result = tracker.capture_exception(exception)

        assert result is None

    def test_capture_exception_initialized(self):
        """Test capturing exceptions when initialized."""
        with patch("ai_engine.core.sentry_integration.sentry_sdk") as mock_sdk:
            tracker = SentryErrorTracker(
                dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
            )
            tracker.initialized = True

            exception = ValueError("Test error")
            result = tracker.capture_exception(exception)

            assert result is not None
            mock_sdk.capture_exception.assert_called_once_with(exception)

    def test_capture_exception_with_context(self):
        """Test capturing exceptions with context."""
        with patch("ai_engine.core.sentry_integration.sentry_sdk") as mock_sdk:
            tracker = SentryErrorTracker(
                dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
            )
            tracker.initialized = True

            exception = ValueError("Test error")
            context = {"test": "context"}
            tags = {"test_tag": "test_value"}

            result = tracker.capture_exception(exception, context=context, tags=tags)

            assert result is not None
            mock_sdk.capture_exception.assert_called_once_with(exception)

    def test_add_breadcrumb_not_initialized(self):
        """Test adding breadcrumbs when not initialized."""
        tracker = SentryErrorTracker()

        # Should not raise an exception
        tracker.add_breadcrumb("Test breadcrumb", category="test")

    def test_add_breadcrumb_initialized(self):
        """Test adding breadcrumbs when initialized."""
        with patch("ai_engine.core.sentry_integration.sentry_sdk") as mock_sdk:
            tracker = SentryErrorTracker(
                dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
            )
            tracker.initialized = True

            tracker.add_breadcrumb("Test breadcrumb", category="test")

            mock_sdk.add_breadcrumb.assert_called_once_with(
                message="Test breadcrumb", category="test", level="info", data={}
            )

    def test_set_user_not_initialized(self):
        """Test setting user when not initialized."""
        tracker = SentryErrorTracker()

        # Should not raise an exception
        tracker.set_user(user_id="123", username="testuser")

    def test_set_user_initialized(self):
        """Test setting user when initialized."""
        with patch("ai_engine.core.sentry_integration.sentry_sdk") as mock_sdk:
            tracker = SentryErrorTracker(
                dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
            )
            tracker.initialized = True

            tracker.set_user(user_id="123", username="testuser")

            mock_sdk.set_user.assert_called_once_with(
                {"id": "123", "username": "testuser", "email": None, "ip_address": None}
            )

    def test_set_tag_not_initialized(self):
        """Test setting tag when not initialized."""
        tracker = SentryErrorTracker()

        # Should not raise an exception
        tracker.set_tag("test_tag", "test_value")

    def test_set_tag_initialized(self):
        """Test setting tag when initialized."""
        with patch("ai_engine.core.sentry_integration.sentry_sdk") as mock_sdk:
            tracker = SentryErrorTracker(
                dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
            )
            tracker.initialized = True

            tracker.set_tag("test_tag", "test_value")

            mock_sdk.set_tag.assert_called_once_with("test_tag", "test_value")

    def test_set_context_not_initialized(self):
        """Test setting context when not initialized."""
        tracker = SentryErrorTracker()

        # Should not raise an exception
        tracker.set_context("test_context", {"test": "value"})

    def test_set_context_initialized(self):
        """Test setting context when initialized."""
        with patch("ai_engine.core.sentry_integration.sentry_sdk") as mock_sdk:
            tracker = SentryErrorTracker(
                dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
            )
            tracker.initialized = True

            context = {"test": "value"}
            tracker.set_context("test_context", context)

            mock_sdk.set_context.assert_called_once_with("test_context", context)

    def test_flush_not_initialized(self):
        """Test flushing when not initialized."""
        tracker = SentryErrorTracker()

        # Should not raise an exception
        tracker.flush()

    def test_flush_initialized(self):
        """Test flushing when initialized."""
        with patch("ai_engine.core.sentry_integration.sentry_sdk") as mock_sdk:
            tracker = SentryErrorTracker(
                dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
            )
            tracker.initialized = True

            tracker.flush(timeout=5.0)

            mock_sdk.flush.assert_called_once_with(timeout=5.0)

    def test_close_not_initialized(self):
        """Test closing when not initialized."""
        tracker = SentryErrorTracker()

        # Should not raise an exception
        tracker.close()

    def test_close_initialized(self):
        """Test closing when initialized."""
        with patch("ai_engine.core.sentry_integration.sentry_sdk") as mock_sdk:
            tracker = SentryErrorTracker(
                dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
            )
            tracker.initialized = True

            tracker.close()

            mock_sdk.flush.assert_called_once_with(timeout=5.0)
            assert tracker.initialized is False

    def test_severity_to_sentry_level(self):
        """Test severity to Sentry level conversion."""
        tracker = SentryErrorTracker()

        assert tracker._severity_to_sentry_level(ErrorSeverity.LOW) == "info"
        assert tracker._severity_to_sentry_level(ErrorSeverity.MEDIUM) == "warning"
        assert tracker._severity_to_sentry_level(ErrorSeverity.HIGH) == "error"
        assert tracker._severity_to_sentry_level(ErrorSeverity.CRITICAL) == "fatal"

    def test_logger_initialization(self):
        """Test that logger is properly initialized."""
        tracker = SentryErrorTracker()

        assert isinstance(tracker.logger, logging.Logger)
        assert tracker.logger.name == "ai_engine.core.sentry_integration"

    def test_default_configuration(self):
        """Test default configuration values."""
        tracker = SentryErrorTracker()

        assert tracker.traces_sample_rate == 0.1
        assert tracker.profiles_sample_rate == 0.1
        assert tracker.enable_tracing is True

    def test_custom_configuration(self):
        """Test custom configuration values."""
        tracker = SentryErrorTracker(
            traces_sample_rate=0.5, profiles_sample_rate=0.2, enable_tracing=False
        )

        assert tracker.traces_sample_rate == 0.5
        assert tracker.profiles_sample_rate == 0.2
        assert tracker.enable_tracing is False


class TestGetSentryTracker:
    """Test the get_sentry_tracker function."""

    def test_get_sentry_tracker_creation(self):
        """Test that get_sentry_tracker creates a new instance."""
        with patch("ai_engine.core.sentry_integration._sentry_tracker", None):
            tracker = get_sentry_tracker()

            assert tracker is not None
            assert isinstance(tracker, SentryErrorTracker)

    def test_get_sentry_tracker_existing(self):
        """Test that get_sentry_tracker returns existing instance."""
        existing_tracker = SentryErrorTracker()

        with patch(
            "ai_engine.core.sentry_integration._sentry_tracker", existing_tracker
        ):
            tracker = get_sentry_tracker()

            assert tracker is existing_tracker


class TestInitializeSentry:
    """Test the initialize_sentry function."""

    def test_initialize_sentry(self):
        """Test initialize_sentry function."""
        with patch(
            "ai_engine.core.sentry_integration.get_sentry_tracker"
        ) as mock_get_tracker:
            mock_tracker = Mock()
            mock_tracker.initialized = True
            mock_get_tracker.return_value = mock_tracker

            result = initialize_sentry()

            assert result is True
            mock_get_tracker.assert_called_once()
