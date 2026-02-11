"""
QBITEL - Translation Studio Logging Infrastructure
Enterprise-grade structured logging for protocol translation, API generation, and SDK creation.
"""

import logging
import json
import time
import sys
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import traceback
import threading
from functools import wraps
import uuid

from prometheus_client import Counter, Histogram, Gauge
from ..core.structured_logging import StructuredLogger, LogLevel

from .exceptions import (
    TranslationStudioException,
    ErrorCategory,
    ErrorSeverity,
    ErrorContext,
)

# Prometheus metrics for logging
LOG_MESSAGES_COUNTER = Counter(
    "qbitel_translation_log_messages_total",
    "Total log messages by level and component",
    ["level", "component", "operation"],
)

ERROR_COUNTER = Counter(
    "qbitel_translation_errors_total",
    "Total errors by category and severity",
    ["category", "severity", "component"],
)

OPERATION_DURATION = Histogram(
    "qbitel_translation_operation_duration_seconds",
    "Duration of translation operations",
    ["component", "operation"],
)

ACTIVE_OPERATIONS = Gauge(
    "qbitel_translation_active_operations",
    "Number of active operations",
    ["component", "operation"],
)


class LogComponent(Enum):
    """Components for structured logging."""

    DISCOVERY = "discovery"
    API_GENERATION = "api_generation"
    CODE_GENERATION = "code_generation"
    PROTOCOL_BRIDGE = "protocol_bridge"
    RAG_ENGINE = "rag_engine"
    API_ENDPOINTS = "api_endpoints"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    CACHE = "cache"
    METRICS = "metrics"
    HEALTH_CHECK = "health_check"


class LogOperation(Enum):
    """Operations for structured logging."""

    # Discovery operations
    PROTOCOL_ANALYSIS = "protocol_analysis"
    FIELD_DETECTION = "field_detection"
    SEMANTIC_ANALYSIS = "semantic_analysis"

    # API Generation operations
    SCHEMA_GENERATION = "schema_generation"
    ENDPOINT_GENERATION = "endpoint_generation"
    OPENAPI_GENERATION = "openapi_generation"

    # Code Generation operations
    SDK_GENERATION = "sdk_generation"
    TEMPLATE_RENDERING = "template_rendering"
    CODE_VALIDATION = "code_validation"

    # Protocol Bridge operations
    PROTOCOL_TRANSLATION = "protocol_translation"
    DATA_MAPPING = "data_mapping"
    QUALITY_ASSESSMENT = "quality_assessment"

    # RAG operations
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    CONTEXT_ENHANCEMENT = "context_enhancement"
    SIMILARITY_SEARCH = "similarity_search"

    # API operations
    REQUEST_PROCESSING = "request_processing"
    RESPONSE_GENERATION = "response_generation"
    FILE_UPLOAD = "file_upload"

    # General operations
    INITIALIZATION = "initialization"
    CONFIGURATION = "configuration"
    CLEANUP = "cleanup"


@dataclass
class LogContext:
    """Context information for structured logging."""

    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[LogComponent] = None
    operation: Optional[LogOperation] = None

    # Translation specific context
    protocol_type: Optional[str] = None
    api_style: Optional[str] = None
    target_language: Optional[str] = None
    confidence: Optional[float] = None
    processing_stage: Optional[str] = None

    # Performance context
    start_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None

    # Additional metadata
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.start_time is None:
            self.start_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                if isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value
        return result


class TranslationStudioLogger:
    """Enterprise-grade logger for Translation Studio operations."""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}

        # Initialize structured logger
        self.logger = StructuredLogger(name)

        # Configure log levels and formats
        self._setup_logging()

        # Performance tracking
        self.active_operations: Dict[str, LogContext] = {}
        self.operation_lock = threading.RLock()

    def _setup_logging(self):
        """Setup logging configuration."""
        # Configure log level
        log_level = self.config.get("log_level", "INFO").upper()
        if hasattr(logging, log_level):
            self.logger.logger.setLevel(getattr(logging, log_level))

        # Configure log format for translation studio
        self.log_format = {
            "timestamp": "%(asctime)s",
            "level": "%(levelname)s",
            "component": "translation_studio",
            "logger": "%(name)s",
            "message": "%(message)s",
            "module": "%(module)s",
            "function": "%(funcName)s",
            "line": "%(lineno)d",
        }

    def _create_log_entry(
        self,
        level: str,
        message: str,
        context: Optional[LogContext] = None,
        exception: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create structured log entry."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "component": "translation_studio",
            "logger": self.name,
            "message": message,
        }

        # Add context information
        if context:
            entry["context"] = context.to_dict()

            # Update metrics
            if context.component and context.operation:
                LOG_MESSAGES_COUNTER.labels(
                    level=level.lower(),
                    component=context.component.value,
                    operation=context.operation.value,
                ).inc()

        # Add exception information
        if exception:
            entry["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc(),
            }

            # Handle TranslationStudioException
            if isinstance(exception, TranslationStudioException):
                entry["exception"].update(
                    {
                        "error_code": exception.error_code,
                        "category": exception.category.value,
                        "severity": exception.severity.value,
                        "details": exception.details,
                    }
                )

                # Update error metrics
                ERROR_COUNTER.labels(
                    category=exception.category.value,
                    severity=exception.severity.value,
                    component=(
                        context.component.value
                        if context and context.component
                        else "unknown"
                    ),
                ).inc()

        # Add extra fields
        if extra:
            entry["extra"] = extra

        return entry

    def debug(
        self,
        message: str,
        context: Optional[LogContext] = None,
        exception: Optional[Exception] = None,
        **kwargs,
    ):
        """Log debug message."""
        entry = self._create_log_entry("DEBUG", message, context, exception, kwargs)
        self.logger.debug(json.dumps(entry))

    def info(
        self,
        message: str,
        context: Optional[LogContext] = None,
        exception: Optional[Exception] = None,
        **kwargs,
    ):
        """Log info message."""
        entry = self._create_log_entry("INFO", message, context, exception, kwargs)
        self.logger.info(json.dumps(entry))

    def warning(
        self,
        message: str,
        context: Optional[LogContext] = None,
        exception: Optional[Exception] = None,
        **kwargs,
    ):
        """Log warning message."""
        entry = self._create_log_entry("WARNING", message, context, exception, kwargs)
        self.logger.warning(json.dumps(entry))

    def error(
        self,
        message: str,
        context: Optional[LogContext] = None,
        exception: Optional[Exception] = None,
        **kwargs,
    ):
        """Log error message."""
        entry = self._create_log_entry("ERROR", message, context, exception, kwargs)
        self.logger.error(json.dumps(entry))

    def critical(
        self,
        message: str,
        context: Optional[LogContext] = None,
        exception: Optional[Exception] = None,
        **kwargs,
    ):
        """Log critical message."""
        entry = self._create_log_entry("CRITICAL", message, context, exception, kwargs)
        self.logger.critical(json.dumps(entry))

    def log_operation_start(
        self,
        operation: LogOperation,
        component: LogComponent,
        context: Optional[LogContext] = None,
    ) -> str:
        """Log operation start and return operation ID."""
        operation_id = str(uuid.uuid4())

        if not context:
            context = LogContext()

        context.operation = operation
        context.component = component
        context.start_time = time.time()

        with self.operation_lock:
            self.active_operations[operation_id] = context

        # Update metrics
        ACTIVE_OPERATIONS.labels(
            component=component.value, operation=operation.value
        ).inc()

        self.info(
            f"Operation started: {operation.value}",
            context=context,
            operation_id=operation_id,
        )

        return operation_id

    def log_operation_end(
        self,
        operation_id: str,
        success: bool = True,
        result_summary: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Log operation completion."""
        with self.operation_lock:
            context = self.active_operations.pop(operation_id, None)

        if not context:
            self.warning(f"Operation end logged for unknown operation: {operation_id}")
            return

        # Calculate duration
        end_time = time.time()
        context.duration = end_time - context.start_time

        # Update metrics
        if context.component and context.operation:
            ACTIVE_OPERATIONS.labels(
                component=context.component.value, operation=context.operation.value
            ).dec()

            OPERATION_DURATION.labels(
                component=context.component.value, operation=context.operation.value
            ).observe(context.duration)

        # Log completion
        status = "completed successfully" if success else "failed"
        message = f"Operation {status}: {context.operation.value if context.operation else 'unknown'}"
        if result_summary:
            message += f" - {result_summary}"

        log_method = self.info if success else self.error
        log_method(
            message,
            context=context,
            operation_id=operation_id,
            duration=context.duration,
            metrics=metrics or {},
        )

    def log_performance_metrics(
        self,
        operation: str,
        metrics: Dict[str, Any],
        context: Optional[LogContext] = None,
    ):
        """Log performance metrics."""
        self.info(
            f"Performance metrics for {operation}",
            context=context,
            performance_metrics=metrics,
        )

    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[LogContext] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log security-related events."""
        log_entry = {
            "security_event": True,
            "event_type": event_type,
            "severity": severity.value,
            "description": description,
        }

        if details:
            log_entry["security_details"] = details

        # Log at appropriate level based on severity
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.critical(
                f"SECURITY EVENT: {description}", context=context, **log_entry
            )
        elif severity == ErrorSeverity.MEDIUM:
            self.error(f"Security event: {description}", context=context, **log_entry)
        else:
            self.warning(
                f"Security notice: {description}", context=context, **log_entry
            )

    def log_api_request(
        self,
        method: str,
        endpoint: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        status_code: Optional[int] = None,
        duration: Optional[float] = None,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None,
    ):
        """Log API request with standard fields."""
        context = LogContext(
            request_id=request_id,
            user_id=user_id,
            component=LogComponent.API_ENDPOINTS,
            operation=LogOperation.REQUEST_PROCESSING,
        )

        self.info(
            f"API Request: {method} {endpoint}",
            context=context,
            api_request={
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "duration": duration,
                "request_size": request_size,
                "response_size": response_size,
            },
        )

    def log_translation_metrics(
        self,
        protocol_type: str,
        confidence: float,
        processing_time: float,
        context: Optional[LogContext] = None,
    ):
        """Log translation-specific metrics."""
        if not context:
            context = LogContext()

        context.protocol_type = protocol_type
        context.confidence = confidence
        context.duration = processing_time

        self.info(
            f"Translation metrics: {protocol_type} (confidence: {confidence:.2f})",
            context=context,
            translation_metrics={
                "protocol_type": protocol_type,
                "confidence": confidence,
                "processing_time": processing_time,
            },
        )


# Global logger instances
_loggers: Dict[str, TranslationStudioLogger] = {}
_logger_lock = threading.RLock()


def get_logger(name: str, config: Dict[str, Any] = None) -> TranslationStudioLogger:
    """Get or create a logger instance."""
    with _logger_lock:
        if name not in _loggers:
            _loggers[name] = TranslationStudioLogger(name, config)
        return _loggers[name]


def create_context(
    request_id: str = None,
    user_id: str = None,
    component: LogComponent = None,
    operation: LogOperation = None,
    **kwargs,
) -> LogContext:
    """Create a log context with common fields."""
    return LogContext(
        request_id=request_id,
        user_id=user_id,
        component=component,
        operation=operation,
        **kwargs,
    )


# Decorators for automatic logging


def log_operation(
    component: LogComponent,
    operation: LogOperation,
    log_args: bool = False,
    log_result: bool = False,
    logger_name: str = None,
):
    """Decorator to automatically log operation start/end."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger_instance = get_logger(
                logger_name or f"{component.value}.{operation.value}"
            )

            # Extract context from kwargs if available
            context = kwargs.pop("log_context", None) or LogContext()
            context.component = component
            context.operation = operation

            operation_id = logger_instance.log_operation_start(
                operation, component, context
            )

            try:
                # Log arguments if requested
                if log_args:
                    logger_instance.debug(
                        f"Function arguments for {func.__name__}",
                        context=context,
                        function_args={
                            "args": [
                                str(arg)[:100] for arg in args
                            ],  # Truncate long args
                            "kwargs": {k: str(v)[:100] for k, v in kwargs.items()},
                        },
                    )

                # Execute function
                result = await func(*args, **kwargs)

                # Log result if requested
                result_summary = None
                if log_result and result:
                    if hasattr(result, "__dict__"):
                        result_summary = f"Result: {type(result).__name__}"
                    else:
                        result_summary = f"Result: {str(result)[:100]}"

                logger_instance.log_operation_end(operation_id, True, result_summary)
                return result

            except Exception as e:
                logger_instance.log_operation_end(operation_id, False)
                logger_instance.error(
                    f"Operation failed: {func.__name__}", context=context, exception=e
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger_instance = get_logger(
                logger_name or f"{component.value}.{operation.value}"
            )

            # Extract context from kwargs if available
            context = kwargs.pop("log_context", None) or LogContext()
            context.component = component
            context.operation = operation

            operation_id = logger_instance.log_operation_start(
                operation, component, context
            )

            try:
                # Log arguments if requested
                if log_args:
                    logger_instance.debug(
                        f"Function arguments for {func.__name__}",
                        context=context,
                        function_args={
                            "args": [str(arg)[:100] for arg in args],
                            "kwargs": {k: str(v)[:100] for k, v in kwargs.items()},
                        },
                    )

                # Execute function
                result = func(*args, **kwargs)

                # Log result if requested
                result_summary = None
                if log_result and result:
                    if hasattr(result, "__dict__"):
                        result_summary = f"Result: {type(result).__name__}"
                    else:
                        result_summary = f"Result: {str(result)[:100]}"

                logger_instance.log_operation_end(operation_id, True, result_summary)
                return result

            except Exception as e:
                logger_instance.log_operation_end(operation_id, False)
                logger_instance.error(
                    f"Operation failed: {func.__name__}", context=context, exception=e
                )
                raise

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def log_exceptions(
    component: LogComponent, logger_name: str = None, re_raise: bool = True
):
    """Decorator to automatically log exceptions."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger_instance = get_logger(logger_name or component.value)

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = kwargs.get("log_context") or LogContext(component=component)
                logger_instance.error(
                    f"Exception in {func.__name__}", context=context, exception=e
                )
                if re_raise:
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger_instance = get_logger(logger_name or component.value)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = kwargs.get("log_context") or LogContext(component=component)
                logger_instance.error(
                    f"Exception in {func.__name__}", context=context, exception=e
                )
                if re_raise:
                    raise

        # Return appropriate wrapper
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Audit logging utilities


class AuditLogger:
    """Specialized logger for audit events."""

    def __init__(self):
        self.logger = get_logger("audit")

    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log user actions for audit purposes."""
        context = LogContext(
            user_id=user_id,
            component=LogComponent.AUTHENTICATION,
            metadata={"audit": True},
        )

        self.logger.info(
            f"User action: {action} on {resource} - {result}",
            context=context,
            audit_event={
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "result": result,
                "details": details or {},
            },
        )

    def log_data_access(
        self,
        user_id: str,
        data_type: str,
        access_type: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log data access events."""
        context = LogContext(
            user_id=user_id,
            component=LogComponent.AUTHENTICATION,
            metadata={"audit": True, "data_access": True},
        )

        result = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"Data access: {access_type} {data_type} - {result}",
            context=context,
            data_access_event={
                "user_id": user_id,
                "data_type": data_type,
                "access_type": access_type,
                "success": success,
                "details": details or {},
            },
        )


# Initialize audit logger
audit_logger = AuditLogger()


__all__ = [
    "TranslationStudioLogger",
    "LogContext",
    "LogComponent",
    "LogOperation",
    "get_logger",
    "create_context",
    "log_operation",
    "log_exceptions",
    "audit_logger",
]
