"""
CRONOS AI - Sentry APM Integration
Production-ready error tracking and performance monitoring with Sentry.
"""

import logging
import os
from typing import Optional, Dict, Any, Callable
from functools import wraps
import sentry_sdk
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.threading import ThreadingIntegration

logger = logging.getLogger(__name__)


class SentryConfig:
    """Sentry configuration."""

    def __init__(self):
        # Core configuration
        self.dsn = os.getenv('SENTRY_DSN', '')
        self.environment = os.getenv('CRONOS_ENVIRONMENT', 'development')
        self.release = os.getenv('CRONOS_VERSION', '1.0.0')

        # Performance monitoring
        self.traces_sample_rate = float(os.getenv('SENTRY_TRACES_SAMPLE_RATE', '0.1'))
        self.profiles_sample_rate = float(os.getenv('SENTRY_PROFILES_SAMPLE_RATE', '0.1'))

        # Error filtering
        self.enabled = os.getenv('SENTRY_ENABLED', 'true').lower() == 'true'
        self.send_default_pii = os.getenv('SENTRY_SEND_PII', 'false').lower() == 'true'

        # Custom tags
        self.tags = {
            'app': 'cronos-ai',
            'component': 'ai-engine',
        }


def initialize_sentry(config: Optional[SentryConfig] = None) -> bool:
    """
    Initialize Sentry SDK with comprehensive integrations.

    Args:
        config: Sentry configuration (uses environment variables if None)

    Returns:
        True if initialized successfully, False otherwise
    """
    if config is None:
        config = SentryConfig()

    if not config.enabled:
        logger.info("Sentry is disabled")
        return False

    if not config.dsn:
        logger.warning("SENTRY_DSN not set - Sentry will not be initialized")
        return False

    try:
        # Configure logging integration
        logging_integration = LoggingIntegration(
            level=logging.INFO,  # Capture info and above as breadcrumbs
            event_level=logging.ERROR  # Send errors as events
        )

        # Initialize Sentry
        sentry_sdk.init(
            dsn=config.dsn,
            environment=config.environment,
            release=config.release,

            # Performance monitoring
            traces_sample_rate=config.traces_sample_rate,
            profiles_sample_rate=config.profiles_sample_rate,

            # Integrations
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                AsyncioIntegration(),
                SqlalchemyIntegration(),
                RedisIntegration(),
                logging_integration,
                ThreadingIntegration(propagate_hub=True),
            ],

            # Privacy settings
            send_default_pii=config.send_default_pii,

            # Error filtering
            before_send=before_send_filter,
            before_send_transaction=before_send_transaction_filter,

            # Additional options
            attach_stacktrace=True,
            max_breadcrumbs=50,
            debug=config.environment == 'development',
        )

        # Set custom tags
        for key, value in config.tags.items():
            sentry_sdk.set_tag(key, value)

        logger.info(
            f"✅ Sentry initialized successfully "
            f"(environment: {config.environment}, "
            f"traces: {config.traces_sample_rate*100}%, "
            f"profiles: {config.profiles_sample_rate*100}%)"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")
        return False


def shutdown_sentry():
    """Shutdown Sentry and flush remaining events."""
    try:
        client = sentry_sdk.Hub.current.client
        if client:
            client.close(timeout=2.0)
            logger.info("Sentry shutdown complete")
    except Exception as e:
        logger.error(f"Error shutting down Sentry: {e}")


def before_send_filter(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter events before sending to Sentry.

    Args:
        event: Sentry event
        hint: Event hint with exception info

    Returns:
        Modified event or None to drop the event
    """
    # Filter out known noise
    if 'exc_info' in hint:
        exc_type, exc_value, tb = hint['exc_info']

        # Ignore specific exceptions
        ignored_exceptions = [
            'asyncio.CancelledError',
            'ConnectionResetError',
            'BrokenPipeError',
        ]

        if exc_type.__name__ in ignored_exceptions:
            return None

    # Filter out health check errors
    if 'request' in event:
        url = event['request'].get('url', '')
        if '/health' in url or '/metrics' in url:
            return None

    # Scrub sensitive data
    if event.get('request', {}).get('headers'):
        headers = event['request']['headers']
        sensitive_headers = ['Authorization', 'X-API-Key', 'Cookie']
        for header in sensitive_headers:
            if header in headers:
                headers[header] = '[Filtered]'

    return event


def before_send_transaction_filter(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter transactions before sending to Sentry.

    Args:
        event: Transaction event
        hint: Event hint

    Returns:
        Modified event or None to drop the transaction
    """
    # Don't track health check transactions
    transaction_name = event.get('transaction', '')
    if transaction_name in ['/health', '/metrics', '/ready']:
        return None

    return event


def capture_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = 'error',
    fingerprint: Optional[list] = None
) -> Optional[str]:
    """
    Capture an error with additional context.

    Args:
        error: Exception to capture
        context: Additional context data
        level: Error level (error, warning, info)
        fingerprint: Custom fingerprint for grouping

    Returns:
        Event ID if captured, None otherwise
    """
    try:
        with sentry_sdk.push_scope() as scope:
            # Set level
            scope.level = level

            # Add context
            if context:
                for key, value in context.items():
                    scope.set_context(key, value)

            # Set fingerprint for custom grouping
            if fingerprint:
                scope.fingerprint = fingerprint

            # Capture exception
            event_id = sentry_sdk.capture_exception(error)
            logger.debug(f"Captured error in Sentry: {event_id}")
            return event_id

    except Exception as e:
        logger.error(f"Failed to capture error in Sentry: {e}")
        return None


def capture_message(
    message: str,
    level: str = 'info',
    context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Capture a message with optional context.

    Args:
        message: Message to capture
        level: Message level (debug, info, warning, error, fatal)
        context: Additional context data

    Returns:
        Event ID if captured, None otherwise
    """
    try:
        with sentry_sdk.push_scope() as scope:
            scope.level = level

            if context:
                for key, value in context.items():
                    scope.set_context(key, value)

            event_id = sentry_sdk.capture_message(message, level=level)
            return event_id

    except Exception as e:
        logger.error(f"Failed to capture message in Sentry: {e}")
        return None


def add_breadcrumb(
    message: str,
    category: str = 'default',
    level: str = 'info',
    data: Optional[Dict[str, Any]] = None
):
    """
    Add a breadcrumb for debugging context.

    Args:
        message: Breadcrumb message
        category: Category (e.g., 'query', 'http', 'auth')
        level: Level (debug, info, warning, error)
        data: Additional data
    """
    try:
        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {}
        )
    except Exception as e:
        logger.error(f"Failed to add breadcrumb: {e}")


def set_user(user_id: str, username: Optional[str] = None, email: Optional[str] = None):
    """
    Set user context for error tracking.

    Args:
        user_id: User ID
        username: Username (optional)
        email: Email (optional, will be filtered if PII is disabled)
    """
    try:
        user_data = {'id': user_id}
        if username:
            user_data['username'] = username
        if email and SentryConfig().send_default_pii:
            user_data['email'] = email

        sentry_sdk.set_user(user_data)
    except Exception as e:
        logger.error(f"Failed to set user context: {e}")


def set_tag(key: str, value: str):
    """
    Set a custom tag for filtering/grouping.

    Args:
        key: Tag key
        value: Tag value
    """
    try:
        sentry_sdk.set_tag(key, value)
    except Exception as e:
        logger.error(f"Failed to set tag: {e}")


def set_context(name: str, context: Dict[str, Any]):
    """
    Set additional context data.

    Args:
        name: Context name
        context: Context data
    """
    try:
        sentry_sdk.set_context(name, context)
    except Exception as e:
        logger.error(f"Failed to set context: {e}")


def monitor_performance(transaction_name: str, operation: str = 'function'):
    """
    Decorator to monitor function performance.

    Args:
        transaction_name: Name of the transaction
        operation: Operation type (function, http, db, etc.)

    Usage:
        @monitor_performance('protocol_discovery', 'ai_inference')
        async def discover_protocol(data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with sentry_sdk.start_transaction(
                op=operation,
                name=transaction_name
            ) as transaction:
                try:
                    result = await func(*args, **kwargs)
                    transaction.set_status("ok")
                    return result
                except Exception as e:
                    transaction.set_status("internal_error")
                    capture_error(e, context={
                        'transaction': transaction_name,
                        'operation': operation
                    })
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with sentry_sdk.start_transaction(
                op=operation,
                name=transaction_name
            ) as transaction:
                try:
                    result = func(*args, **kwargs)
                    transaction.set_status("ok")
                    return result
                except Exception as e:
                    transaction.set_status("internal_error")
                    capture_error(e, context={
                        'transaction': transaction_name,
                        'operation': operation
                    })
                    raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def trace_span(operation: str, description: Optional[str] = None):
    """
    Decorator to trace a span within a transaction.

    Args:
        operation: Operation name
        description: Span description

    Usage:
        @trace_span('database.query', 'fetch_protocols')
        def get_protocols():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with sentry_sdk.start_span(
                op=operation,
                description=description or func.__name__
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status("ok")
                    return result
                except Exception as e:
                    span.set_status("internal_error")
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with sentry_sdk.start_span(
                op=operation,
                description=description or func.__name__
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status("ok")
                    return result
                except Exception as e:
                    span.set_status("internal_error")
                    raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class SentryMiddleware:
    """
    Custom middleware for enhanced Sentry integration.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Start transaction
        transaction = sentry_sdk.start_transaction(
            op="http.server",
            name=f"{scope['method']} {scope['path']}",
            source="route"
        )

        # Add request context
        sentry_sdk.set_context("request", {
            "method": scope["method"],
            "path": scope["path"],
            "query_string": scope.get("query_string", b"").decode(),
        })

        try:
            with sentry_sdk.Hub(sentry_sdk.Hub.current):
                with transaction:
                    await self.app(scope, receive, send)
                    transaction.set_status("ok")
        except Exception as e:
            transaction.set_status("internal_error")
            capture_error(e)
            raise


# Health check helper
def get_sentry_health() -> Dict[str, Any]:
    """
    Get Sentry health status.

    Returns:
        Health status dict
    """
    try:
        client = sentry_sdk.Hub.current.client
        if client and client.is_healthy():
            return {
                "status": "healthy",
                "dsn_configured": bool(client.dsn),
                "enabled": True
            }
        elif not SentryConfig().enabled:
            return {
                "status": "disabled",
                "enabled": False
            }
        else:
            return {
                "status": "unhealthy",
                "enabled": True
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "enabled": False
        }


# Example usage context manager
class sentry_transaction:
    """
    Context manager for manual transaction creation.

    Usage:
        with sentry_transaction('custom_operation', 'my_task'):
            # Your code here
            pass
    """

    def __init__(self, operation: str, name: str):
        self.operation = operation
        self.name = name
        self.transaction = None

    def __enter__(self):
        self.transaction = sentry_sdk.start_transaction(
            op=self.operation,
            name=self.name
        )
        self.transaction.__enter__()
        return self.transaction

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.transaction.set_status("ok")
        else:
            self.transaction.set_status("internal_error")
            capture_error(exc_val)

        self.transaction.__exit__(exc_type, exc_val, exc_tb)
        return False
