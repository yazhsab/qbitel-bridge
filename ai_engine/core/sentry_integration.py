"""
QBITEL Engine - Sentry Integration
Enterprise error tracking and monitoring with Sentry.
"""

import logging
import os
from typing import Dict, Any, Optional, List
import sentry_sdk
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration

from .error_handling import ErrorRecord, ErrorSeverity
from .exceptions import QbitelAIException

logger = logging.getLogger(__name__)


class SentryErrorTracker:
    """
    Sentry integration for comprehensive error tracking and monitoring.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        environment: str = "production",
        release: Optional[str] = None,
        traces_sample_rate: float = 0.1,
        profiles_sample_rate: float = 0.1,
        enable_tracing: bool = True,
    ):
        """
        Initialize Sentry integration.

        Args:
            dsn: Sentry DSN (Data Source Name)
            environment: Environment name (production, staging, development)
            release: Release version
            traces_sample_rate: Percentage of transactions to trace (0.0 to 1.0)
            profiles_sample_rate: Percentage of transactions to profile (0.0 to 1.0)
            enable_tracing: Enable performance tracing
        """
        self.dsn = dsn or os.getenv("SENTRY_DSN")
        self.environment = environment
        self.release = release or os.getenv("QBITEL_AI_VERSION", "1.0.0")
        self.traces_sample_rate = traces_sample_rate
        self.profiles_sample_rate = profiles_sample_rate
        self.enable_tracing = enable_tracing
        self.initialized = False

        self.logger = logging.getLogger(__name__)

    def initialize(self) -> bool:
        """
        Initialize Sentry SDK with all integrations.

        Returns:
            True if initialized successfully
        """
        if not self.dsn:
            self.logger.warning("Sentry DSN not provided, error tracking disabled")
            return False

        try:
            # Configure logging integration
            logging_integration = LoggingIntegration(
                level=logging.INFO,  # Capture info and above as breadcrumbs
                event_level=logging.ERROR,  # Send errors as events
            )

            # Initialize Sentry
            sentry_sdk.init(
                dsn=self.dsn,
                environment=self.environment,
                release=self.release,
                traces_sample_rate=(
                    self.traces_sample_rate if self.enable_tracing else 0.0
                ),
                profiles_sample_rate=(
                    self.profiles_sample_rate if self.enable_tracing else 0.0
                ),
                integrations=[
                    FastApiIntegration(transaction_style="endpoint"),
                    AsyncioIntegration(),
                    logging_integration,
                    RedisIntegration(),
                    SqlalchemyIntegration(),
                ],
                # Set custom tags
                default_integrations=True,
                attach_stacktrace=True,
                send_default_pii=False,  # Don't send PII by default
                max_breadcrumbs=100,
                before_send=self._before_send,
                before_breadcrumb=self._before_breadcrumb,
            )

            # Set global tags
            sentry_sdk.set_tag("service", "qbitel-engine")
            sentry_sdk.set_tag("component", "ai-engine")

            self.initialized = True
            self.logger.info(f"Sentry initialized for environment: {self.environment}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Sentry: {e}")
            return False

    def _before_send(
        self, event: Dict[str, Any], hint: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Filter and modify events before sending to Sentry.

        Args:
            event: Sentry event dictionary
            hint: Additional context

        Returns:
            Modified event or None to drop the event
        """
        # Filter out low-severity errors in production
        if self.environment == "production":
            if event.get("level") == "info":
                return None

        # Add custom context
        if "exception" in hint:
            exc = hint["exception"]
            if isinstance(exc, QbitelAIException):
                event["tags"] = event.get("tags", {})
                event["tags"]["qbitel_exception"] = True

        return event

    def _before_breadcrumb(
        self, crumb: Dict[str, Any], hint: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Filter and modify breadcrumbs before adding to event.

        Args:
            crumb: Breadcrumb dictionary
            hint: Additional context

        Returns:
            Modified breadcrumb or None to drop it
        """
        # Filter out noisy breadcrumbs
        if crumb.get("category") == "query" and crumb.get("message", "").startswith(
            "SELECT 1"
        ):
            return None

        return crumb

    def capture_error_record(self, error_record: ErrorRecord) -> Optional[str]:
        """
        Capture error record in Sentry.

        Args:
            error_record: Error record to capture

        Returns:
            Sentry event ID or None
        """
        if not self.initialized:
            return None

        try:
            # Set context
            with sentry_sdk.push_scope() as scope:
                # Set tags
                scope.set_tag("component", error_record.component)
                scope.set_tag("operation", error_record.operation)
                scope.set_tag("severity", error_record.severity.value)
                scope.set_tag("category", error_record.category.value)
                scope.set_tag("exception_type", error_record.exception_type)

                # Set context
                scope.set_context(
                    "error_record",
                    {
                        "error_id": error_record.error_id,
                        "timestamp": error_record.timestamp,
                        "recovery_attempted": error_record.recovery_attempted,
                        "recovery_successful": error_record.recovery_successful,
                        "recovery_strategy": (
                            error_record.recovery_strategy.value
                            if error_record.recovery_strategy
                            else None
                        ),
                        "retry_count": error_record.retry_count,
                    },
                )

                scope.set_context(
                    "request_context",
                    {
                        "request_id": error_record.context.request_id,
                        "user_id": error_record.context.user_id,
                        "session_id": error_record.context.session_id,
                        "trace_id": error_record.context.trace_id,
                    },
                )

                # Set extra data
                scope.set_extra("additional_data", error_record.context.additional_data)
                scope.set_extra("metadata", error_record.metadata)

                # Set level based on severity
                level = self._severity_to_sentry_level(error_record.severity)
                scope.level = level

                # Capture exception
                event_id = sentry_sdk.capture_message(
                    f"{error_record.exception_type}: {error_record.exception_message}",
                    level=level,
                )

                return event_id

        except Exception as e:
            self.logger.error(f"Failed to capture error in Sentry: {e}")
            return None

    def capture_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        level: str = "error",
    ) -> Optional[str]:
        """
        Capture exception in Sentry with custom context.

        Args:
            exception: Exception to capture
            context: Additional context
            tags: Custom tags
            level: Sentry level (error, warning, info)

        Returns:
            Sentry event ID or None
        """
        if not self.initialized:
            return None

        try:
            with sentry_sdk.push_scope() as scope:
                # Set tags
                if tags:
                    for key, value in tags.items():
                        scope.set_tag(key, value)

                # Set context
                if context:
                    scope.set_context("custom", context)

                # Set level
                scope.level = level

                # Capture exception
                event_id = sentry_sdk.capture_exception(exception)
                return event_id

        except Exception as e:
            self.logger.error(f"Failed to capture exception in Sentry: {e}")
            return None

    def add_breadcrumb(
        self,
        message: str,
        category: str = "default",
        level: str = "info",
        data: Optional[Dict[str, Any]] = None,
    ):
        """
        Add breadcrumb for debugging context.

        Args:
            message: Breadcrumb message
            category: Breadcrumb category
            level: Breadcrumb level
            data: Additional data
        """
        if not self.initialized:
            return

        try:
            sentry_sdk.add_breadcrumb(
                message=message, category=category, level=level, data=data or {}
            )
        except Exception as e:
            self.logger.error(f"Failed to add breadcrumb: {e}")

    def set_user(
        self,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        email: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        """
        Set user context for error tracking.

        Args:
            user_id: User ID
            username: Username
            email: User email
            ip_address: User IP address
        """
        if not self.initialized:
            return

        try:
            sentry_sdk.set_user(
                {
                    "id": user_id,
                    "username": username,
                    "email": email,
                    "ip_address": ip_address,
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to set user context: {e}")

    def set_tag(self, key: str, value: str):
        """Set custom tag."""
        if self.initialized:
            sentry_sdk.set_tag(key, value)

    def set_context(self, key: str, value: Dict[str, Any]):
        """Set custom context."""
        if self.initialized:
            sentry_sdk.set_context(key, value)

    def start_transaction(self, name: str, op: str = "task") -> Any:
        """
        Start performance transaction.

        Args:
            name: Transaction name
            op: Operation type

        Returns:
            Transaction object
        """
        if not self.initialized or not self.enable_tracing:
            return None

        return sentry_sdk.start_transaction(name=name, op=op)

    def _severity_to_sentry_level(self, severity: ErrorSeverity) -> str:
        """Convert error severity to Sentry level."""
        severity_map = {
            ErrorSeverity.LOW: "info",
            ErrorSeverity.MEDIUM: "warning",
            ErrorSeverity.HIGH: "error",
            ErrorSeverity.CRITICAL: "fatal",
        }
        return severity_map.get(severity, "error")

    def flush(self, timeout: float = 2.0):
        """
        Flush pending events to Sentry.

        Args:
            timeout: Timeout in seconds
        """
        if self.initialized:
            sentry_sdk.flush(timeout=timeout)

    def close(self):
        """Close Sentry client."""
        if self.initialized:
            sentry_sdk.flush(timeout=5.0)
            self.initialized = False
            self.logger.info("Sentry client closed")


# Global Sentry tracker instance
_sentry_tracker: Optional[SentryErrorTracker] = None


def get_sentry_tracker(
    dsn: Optional[str] = None,
    environment: Optional[str] = None,
    release: Optional[str] = None,
) -> SentryErrorTracker:
    """Get or create global Sentry tracker instance."""
    global _sentry_tracker

    if _sentry_tracker is None:
        _sentry_tracker = SentryErrorTracker(
            dsn=dsn,
            environment=environment or os.getenv("QBITEL_ENVIRONMENT", "production"),
            release=release,
        )
        _sentry_tracker.initialize()

    return _sentry_tracker


def initialize_sentry(
    dsn: Optional[str] = None,
    environment: Optional[str] = None,
    release: Optional[str] = None,
) -> bool:
    """
    Initialize Sentry for the application.

    Args:
        dsn: Sentry DSN
        environment: Environment name
        release: Release version

    Returns:
        True if initialized successfully
    """
    tracker = get_sentry_tracker(dsn, environment, release)
    return tracker.initialized
