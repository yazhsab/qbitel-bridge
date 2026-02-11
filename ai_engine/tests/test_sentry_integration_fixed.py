"""
Test suite for sentry_integration.py with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging

from ai_engine.core.sentry_integration import SentryErrorTracker


class TestSentryErrorTracker:
    """Test SentryErrorTracker with mocked dependencies."""

    @pytest.fixture
    def mock_sentry_sdk(self):
        """Mock sentry_sdk module."""
        with patch("ai_engine.core.sentry_integration.sentry_sdk") as mock_sdk:
            mock_sdk.init = Mock()
            mock_sdk.capture_exception = Mock()
            mock_sdk.capture_message = Mock()
            mock_sdk.add_breadcrumb = Mock()
            mock_sdk.set_tag = Mock()
            mock_sdk.set_user = Mock()
            mock_sdk.set_context = Mock()
            mock_sdk.set_extra = Mock()
            mock_sdk.set_level = Mock()
            mock_sdk.flush = Mock()
            mock_sdk.get_current_hub = Mock()
            yield mock_sdk

    @pytest.fixture
    def sentry_tracker(self, mock_sentry_sdk):
        """Create SentryErrorTracker instance."""
        return SentryErrorTracker(
            dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
        )

    def test_initialization_with_dsn(self, mock_sentry_sdk):
        """Test initialization with DSN."""
        tracker = SentryErrorTracker(
            dsn="https://test@sentry.io/123", environment="test", release="1.0.0"
        )

        assert tracker.dsn == "https://test@sentry.io/123"
        assert tracker.environment == "test"
        assert tracker.release == "1.0.0"
        assert tracker.initialized is False

    def test_initialization_without_dsn(self, mock_sentry_sdk):
        """Test initialization without DSN."""
        with patch.dict("os.environ", {"SENTRY_DSN": "https://env@sentry.io/456"}):
            tracker = SentryErrorTracker()

            assert tracker.dsn == "https://env@sentry.io/456"
            assert tracker.environment == "production"
            assert tracker.initialized is False

    def test_initialization_with_version_from_env(self, mock_sentry_sdk):
        """Test initialization with version from environment."""
        with patch.dict("os.environ", {"QBITEL_AI_VERSION": "2.0.0"}):
            tracker = SentryErrorTracker()

            assert tracker.release == "2.0.0"

    def test_initialize_success(self, sentry_tracker, mock_sentry_sdk):
        """Test successful initialization."""
        sentry_tracker.initialize()

        assert sentry_tracker.initialized is True
        mock_sentry_sdk.init.assert_called_once()

        # Check that init was called with correct parameters
        call_args = mock_sentry_sdk.init.call_args
        assert call_args[1]["dsn"] == "https://test@sentry.io/123"
        assert call_args[1]["environment"] == "test"
        assert call_args[1]["release"] == "1.0.0"

    def test_initialize_without_dsn(self, mock_sentry_sdk):
        """Test initialization without DSN."""
        tracker = SentryErrorTracker(dsn=None)

        with pytest.raises(ValueError, match="Sentry DSN is required"):
            tracker.initialize()

    def test_initialize_already_initialized(self, sentry_tracker, mock_sentry_sdk):
        """Test initialization when already initialized."""
        sentry_tracker.initialized = True

        sentry_tracker.initialize()

        # Should not call init again
        mock_sentry_sdk.init.assert_not_called()

    def test_capture_exception(self, sentry_tracker, mock_sentry_sdk):
        """Test capturing exceptions."""
        sentry_tracker.initialized = True

        exception = ValueError("Test error")
        sentry_tracker.capture_exception(exception)

        mock_sentry_sdk.capture_exception.assert_called_once_with(exception)

    def test_capture_exception_not_initialized(self, sentry_tracker, mock_sentry_sdk):
        """Test capturing exceptions when not initialized."""
        exception = ValueError("Test error")
        sentry_tracker.capture_exception(exception)

        # Should not capture when not initialized
        mock_sentry_sdk.capture_exception.assert_not_called()

    def test_capture_message(self, sentry_tracker, mock_sentry_sdk):
        """Test capturing messages."""
        sentry_tracker.initialized = True

        sentry_tracker.capture_message("Test message", level="info")

        mock_sentry_sdk.capture_message.assert_called_once_with(
            "Test message", level="info"
        )

    def test_capture_message_not_initialized(self, sentry_tracker, mock_sentry_sdk):
        """Test capturing messages when not initialized."""
        sentry_tracker.capture_message("Test message")

        # Should not capture when not initialized
        mock_sentry_sdk.capture_message.assert_not_called()

    def test_add_breadcrumb(self, sentry_tracker, mock_sentry_sdk):
        """Test adding breadcrumbs."""
        sentry_tracker.initialized = True

        sentry_tracker.add_breadcrumb("Test breadcrumb", category="test")

        mock_sentry_sdk.add_breadcrumb.assert_called_once_with(
            message="Test breadcrumb", category="test"
        )

    def test_set_tag(self, sentry_tracker, mock_sentry_sdk):
        """Test setting tags."""
        sentry_tracker.initialized = True

        sentry_tracker.set_tag("test_tag", "test_value")

        mock_sentry_sdk.set_tag.assert_called_once_with("test_tag", "test_value")

    def test_set_user(self, sentry_tracker, mock_sentry_sdk):
        """Test setting user context."""
        sentry_tracker.initialized = True

        user_info = {"id": "123", "username": "testuser"}
        sentry_tracker.set_user(user_info)

        mock_sentry_sdk.set_user.assert_called_once_with(user_info)

    def test_set_context(self, sentry_tracker, mock_sentry_sdk):
        """Test setting context."""
        sentry_tracker.initialized = True

        context = {"component": "test", "operation": "test_op"}
        sentry_tracker.set_context("test_context", context)

        mock_sentry_sdk.set_context.assert_called_once_with("test_context", context)

    def test_set_extra(self, sentry_tracker, mock_sentry_sdk):
        """Test setting extra data."""
        sentry_tracker.initialized = True

        sentry_tracker.set_extra("test_extra", "test_value")

        mock_sentry_sdk.set_extra.assert_called_once_with("test_extra", "test_value")

    def test_set_level(self, sentry_tracker, mock_sentry_sdk):
        """Test setting log level."""
        sentry_tracker.initialized = True

        sentry_tracker.set_level("error")

        mock_sentry_sdk.set_level.assert_called_once_with("error")

    def test_flush(self, sentry_tracker, mock_sentry_sdk):
        """Test flushing events."""
        sentry_tracker.initialized = True

        sentry_tracker.flush()

        mock_sentry_sdk.flush.assert_called_once()

    def test_get_current_hub(self, sentry_tracker, mock_sentry_sdk):
        """Test getting current hub."""
        sentry_tracker.initialized = True

        result = sentry_tracker.get_current_hub()

        mock_sentry_sdk.get_current_hub.assert_called_once()
        assert result == mock_sentry_sdk.get_current_hub.return_value

    def test_logger_initialization(self, sentry_tracker):
        """Test that logger is properly initialized."""
        assert isinstance(sentry_tracker.logger, logging.Logger)
        assert sentry_tracker.logger.name == "ai_engine.core.sentry_integration"

    def test_default_configuration(self, mock_sentry_sdk):
        """Test default configuration values."""
        tracker = SentryErrorTracker()

        assert tracker.traces_sample_rate == 0.1
        assert tracker.profiles_sample_rate == 0.1
        assert tracker.enable_tracing is True
        assert tracker.access_token_expire_minutes == 30
        assert tracker.refresh_token_expire_days == 7

    def test_custom_configuration(self, mock_sentry_sdk):
        """Test custom configuration values."""
        tracker = SentryErrorTracker(
            traces_sample_rate=0.5, profiles_sample_rate=0.2, enable_tracing=False
        )

        assert tracker.traces_sample_rate == 0.5
        assert tracker.profiles_sample_rate == 0.2
        assert tracker.enable_tracing is False
