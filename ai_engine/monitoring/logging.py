"""
CRONOS AI Engine - Structured Logging

This module provides structured logging capabilities with distributed
tracing, contextual information, and comprehensive log management.
"""

import logging
import json
import time
import threading
import asyncio
import uuid
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
import traceback
import sys
import os
from pathlib import Path

from ..core.config import Config

try:
    from ..core.exceptions import LoggingException
except ImportError:  # pragma: no cover - fallback when symbol missing

    class LoggingException(Exception):
        """Fallback logging exception used when core definition is unavailable."""

        pass


class LogLevel(str, Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log format types."""

    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"


@dataclass
class TraceContext:
    """Distributed tracing context."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    tags: Dict[str, Any] = field(default_factory=dict)
    baggage: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "duration_ms": (time.time() - self.start_time) * 1000,
            "tags": self.tags,
            "baggage": self.baggage,
        }


@dataclass
class LoggerConfig:
    """Configuration for structured logger."""

    name: str
    level: LogLevel = LogLevel.INFO
    format_type: LogFormat = LogFormat.STRUCTURED
    enable_console: bool = True
    enable_file: bool = True
    log_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_json_format: bool = False
    include_trace_info: bool = True
    include_system_info: bool = True
    sensitive_fields: List[str] = field(
        default_factory=lambda: ["password", "token", "api_key"]
    )
    custom_fields: Dict[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """
    Structured log formatter that outputs JSON or structured text.

    This formatter adds contextual information including trace context,
    system information, and custom fields to log records.
    """

    def __init__(
        self,
        config: LoggerConfig,
        trace_context_provider: Optional[Callable[[], Optional[TraceContext]]] = None,
    ):
        """Initialize structured formatter."""
        super().__init__()
        self.config = config
        self.trace_context_provider = trace_context_provider

        # System information
        self.system_info = {
            "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "process_id": os.getpid(),
            "thread_id": threading.get_ident(),
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured data."""
        # Base log data
        log_data = {
            "timestamp": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
        }

        # Add system info if enabled
        if self.config.include_system_info:
            log_data["system"] = self.system_info.copy()

        # Add trace context if available
        if self.config.include_trace_info and self.trace_context_provider:
            trace_context = self.trace_context_provider()
            if trace_context:
                log_data["trace"] = trace_context.to_dict()

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add custom fields
        if self.config.custom_fields:
            log_data.update(self.config.custom_fields)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                # Filter sensitive fields
                if key.lower() in [
                    field.lower() for field in self.config.sensitive_fields
                ]:
                    log_data[key] = "[REDACTED]"
                else:
                    log_data[key] = value

        # Format output
        if self.config.format_type == LogFormat.JSON:
            return json.dumps(log_data, default=str, separators=(",", ":"))
        elif self.config.format_type == LogFormat.STRUCTURED:
            # Custom structured format
            base_info = f"{log_data['timestamp']:.3f} [{log_data['level']:8}] {log_data['logger']} - {log_data['message']}"

            extras = []
            if "trace" in log_data and log_data["trace"].get("trace_id"):
                extras.append(f"trace_id={log_data['trace']['trace_id']}")
            if "exception" in log_data:
                extras.append(f"exception={log_data['exception']['type']}")

            if extras:
                return f"{base_info} | {' '.join(extras)}"
            else:
                return base_info
        else:
            # Fallback to text format
            return f"{log_data['timestamp']:.3f} [{log_data['level']}] {log_data['message']}"


class StructuredLogger:
    """
    Advanced structured logger with distributed tracing support.

    This logger provides structured logging capabilities with contextual
    information, distributed tracing integration, and performance monitoring.
    """

    def __init__(self, config: LoggerConfig):
        """Initialize structured logger."""
        self.config = config
        self.logger = logging.getLogger(config.name)
        self.logger.setLevel(getattr(logging, config.level.value))

        # Trace context storage (thread-local)
        self._trace_contexts: Dict[int, TraceContext] = {}
        self._trace_lock = threading.RLock()

        # Performance tracking
        self._log_counts: Dict[str, int] = {}
        self._log_times: Dict[str, List[float]] = {}

        # Setup handlers
        self._setup_handlers()

        # Prevent duplicate log messages
        self.logger.propagate = False

    def _setup_handlers(self) -> None:
        """Setup logging handlers."""
        # Clear existing handlers
        self.logger.handlers.clear()

        formatter = StructuredFormatter(self.config, self._get_current_trace_context)

        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if self.config.enable_file:
            log_file = self.config.log_file or f"{self.config.name}.log"

            # Create log directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Use rotating file handler
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _get_current_trace_context(self) -> Optional[TraceContext]:
        """Get current thread's trace context."""
        thread_id = threading.get_ident()
        with self._trace_lock:
            return self._trace_contexts.get(thread_id)

    def _set_trace_context(self, context: TraceContext) -> None:
        """Set trace context for current thread."""
        thread_id = threading.get_ident()
        with self._trace_lock:
            self._trace_contexts[thread_id] = context

    def _clear_trace_context(self) -> None:
        """Clear trace context for current thread."""
        thread_id = threading.get_ident()
        with self._trace_lock:
            self._trace_contexts.pop(thread_id, None)

    def _track_log_performance(self, level: str, duration: float) -> None:
        """Track logging performance metrics."""
        self._log_counts[level] = self._log_counts.get(level, 0) + 1

        if level not in self._log_times:
            self._log_times[level] = []
        self._log_times[level].append(duration)

        # Keep only last 1000 measurements
        if len(self._log_times[level]) > 1000:
            self._log_times[level] = self._log_times[level][-1000:]

    @contextmanager
    def trace_span(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for creating distributed trace spans."""
        # Generate IDs
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())

        # Create trace context
        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            tags=tags or {},
        )

        # Set context
        old_context = self._get_current_trace_context()
        self._set_trace_context(context)

        try:
            self.debug(
                f"Started operation: {operation_name}",
                trace_id=trace_id,
                span_id=span_id,
                operation=operation_name,
            )
            yield context
            self.debug(
                f"Completed operation: {operation_name}",
                trace_id=trace_id,
                span_id=span_id,
                operation=operation_name,
                duration_ms=(time.time() - context.start_time) * 1000,
            )
        except Exception as e:
            self.error(
                f"Failed operation: {operation_name}",
                trace_id=trace_id,
                span_id=span_id,
                operation=operation_name,
                error=str(e),
                exc_info=True,
            )
            raise
        finally:
            # Restore previous context
            if old_context:
                self._set_trace_context(old_context)
            else:
                self._clear_trace_context()

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        start_time = time.time()
        self.logger.debug(message, extra=kwargs)
        self._track_log_performance("DEBUG", time.time() - start_time)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        start_time = time.time()
        self.logger.info(message, extra=kwargs)
        self._track_log_performance("INFO", time.time() - start_time)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        start_time = time.time()
        self.logger.warning(message, extra=kwargs)
        self._track_log_performance("WARNING", time.time() - start_time)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        start_time = time.time()
        self.logger.error(message, extra=kwargs)
        self._track_log_performance("ERROR", time.time() - start_time)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        start_time = time.time()
        self.logger.critical(message, extra=kwargs)
        self._track_log_performance("CRITICAL", time.time() - start_time)

    def log_model_inference(
        self,
        model_name: str,
        input_size: int,
        inference_time_ms: float,
        success: bool,
        confidence: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Log model inference operation."""
        log_data = {
            "model_name": model_name,
            "input_size_bytes": input_size,
            "inference_time_ms": inference_time_ms,
            "success": success,
            "operation_type": "model_inference",
        }

        if confidence is not None:
            log_data["confidence"] = confidence

        log_data.update(kwargs)

        if success:
            self.info(f"Model inference completed: {model_name}", **log_data)
        else:
            self.error(f"Model inference failed: {model_name}", **log_data)

    def log_protocol_discovery(
        self,
        discovered_protocol: Optional[str],
        confidence: float,
        processing_time_ms: float,
        data_size: int,
        **kwargs,
    ) -> None:
        """Log protocol discovery operation."""
        log_data = {
            "discovered_protocol": discovered_protocol or "unknown",
            "confidence": confidence,
            "processing_time_ms": processing_time_ms,
            "data_size_bytes": data_size,
            "operation_type": "protocol_discovery",
        }

        log_data.update(kwargs)

        if discovered_protocol:
            self.info(
                f"Protocol discovered: {discovered_protocol} (confidence: {confidence:.2f})",
                **log_data,
            )
        else:
            self.warning(
                f"No protocol discovered (confidence: {confidence:.2f})", **log_data
            )

    def log_field_detection(
        self,
        field_count: int,
        processing_time_ms: float,
        protocol_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log field detection operation."""
        log_data = {
            "field_count": field_count,
            "processing_time_ms": processing_time_ms,
            "protocol_type": protocol_type or "unknown",
            "operation_type": "field_detection",
        }

        log_data.update(kwargs)

        self.info(
            f"Field detection completed: {field_count} fields detected", **log_data
        )

    def log_anomaly_detection(
        self,
        is_anomalous: bool,
        anomaly_score: float,
        processing_time_ms: float,
        threshold: float,
        **kwargs,
    ) -> None:
        """Log anomaly detection operation."""
        log_data = {
            "is_anomalous": is_anomalous,
            "anomaly_score": anomaly_score,
            "processing_time_ms": processing_time_ms,
            "threshold": threshold,
            "operation_type": "anomaly_detection",
        }

        log_data.update(kwargs)

        if is_anomalous:
            self.warning(
                f"Anomaly detected: score={anomaly_score:.3f}, threshold={threshold:.3f}",
                **log_data,
            )
        else:
            self.info(
                f"No anomaly detected: score={anomaly_score:.3f}, threshold={threshold:.3f}",
                **log_data,
            )

    def log_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        processing_time_ms: float,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log API request."""
        log_data = {
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "processing_time_ms": processing_time_ms,
            "operation_type": "api_request",
        }

        if request_id:
            log_data["request_id"] = request_id
        if user_id:
            log_data["user_id"] = user_id

        log_data.update(kwargs)

        if status_code < 400:
            self.info(f"API request: {method} {endpoint} -> {status_code}", **log_data)
        elif status_code < 500:
            self.warning(
                f"Client error: {method} {endpoint} -> {status_code}", **log_data
            )
        else:
            self.error(
                f"Server error: {method} {endpoint} -> {status_code}", **log_data
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get logging performance statistics."""
        stats = {"log_counts": self._log_counts.copy(), "average_log_times": {}}

        for level, times in self._log_times.items():
            if times:
                stats["average_log_times"][level] = {
                    "count": len(times),
                    "avg_ms": (sum(times) * 1000) / len(times),
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                }

        return stats

    def configure_level(self, level: LogLevel) -> None:
        """Dynamically change log level."""
        self.config.level = level
        self.logger.setLevel(getattr(logging, level.value))

    def add_custom_field(self, key: str, value: Any) -> None:
        """Add a custom field to all future log messages."""
        self.config.custom_fields[key] = value

    def remove_custom_field(self, key: str) -> None:
        """Remove a custom field."""
        self.config.custom_fields.pop(key, None)


# Global logger registry
_logger_registry: Dict[str, StructuredLogger] = {}
_registry_lock = threading.RLock()


def setup_logging(config: Config) -> None:
    """Setup global logging configuration."""
    global _logger_registry

    # Main logger configuration
    main_config = LoggerConfig(
        name="cronos_ai",
        level=LogLevel(getattr(config, "log_level", "INFO").upper()),
        format_type=LogFormat(getattr(config, "log_format", "structured")),
        enable_console=getattr(config, "enable_console_logging", True),
        enable_file=getattr(config, "enable_file_logging", True),
        log_file=getattr(config, "log_file", "cronos_ai.log"),
        include_trace_info=getattr(config, "enable_tracing", True),
        include_system_info=getattr(config, "include_system_info", True),
    )

    # Create main logger
    main_logger = StructuredLogger(main_config)

    with _registry_lock:
        _logger_registry["cronos_ai"] = main_logger

    # Create component-specific loggers
    components = [
        "cronos_ai.engine",
        "cronos_ai.discovery",
        "cronos_ai.detection",
        "cronos_ai.anomaly",
        "cronos_ai.training",
        "cronos_ai.models",
        "cronos_ai.api",
        "cronos_ai.monitoring",
    ]

    for component in components:
        component_config = LoggerConfig(
            name=component,
            level=main_config.level,
            format_type=main_config.format_type,
            enable_console=main_config.enable_console,
            enable_file=main_config.enable_file,
            log_file=main_config.log_file,
            include_trace_info=main_config.include_trace_info,
            include_system_info=main_config.include_system_info,
        )

        component_logger = StructuredLogger(component_config)

        with _registry_lock:
            _logger_registry[component] = component_logger


def get_logger(name: str = "cronos_ai") -> StructuredLogger:
    """Get a logger instance by name."""
    with _registry_lock:
        if name not in _logger_registry:
            # Create default logger if not found
            config = LoggerConfig(name=name)
            logger = StructuredLogger(config)
            _logger_registry[name] = logger

        return _logger_registry[name]


def configure_all_loggers(level: LogLevel) -> None:
    """Configure log level for all loggers."""
    with _registry_lock:
        for logger in _logger_registry.values():
            logger.configure_level(level)


def get_all_logger_stats() -> Dict[str, Any]:
    """Get performance stats for all loggers."""
    with _registry_lock:
        return {
            name: logger.get_performance_stats()
            for name, logger in _logger_registry.items()
        }


# Convenience functions for common logging patterns


def log_function_call(logger: StructuredLogger):
    """Decorator for logging function calls."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with logger.trace_span(f"{func.__module__}.{func.__name__}"):
                logger.debug(
                    f"Calling function: {func.__name__}",
                    function=func.__name__,
                    module=func.__module__,
                )
                try:
                    result = func(*args, **kwargs)
                    logger.debug(f"Function completed: {func.__name__}")
                    return result
                except Exception as e:
                    logger.error(
                        f"Function failed: {func.__name__}", error=str(e), exc_info=True
                    )
                    raise

        return wrapper

    return decorator


async def log_async_function_call(logger: StructuredLogger):
    """Decorator for logging async function calls."""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            with logger.trace_span(f"{func.__module__}.{func.__name__}"):
                logger.debug(
                    f"Calling async function: {func.__name__}",
                    function=func.__name__,
                    module=func.__module__,
                )
                try:
                    result = await func(*args, **kwargs)
                    logger.debug(f"Async function completed: {func.__name__}")
                    return result
                except Exception as e:
                    logger.error(
                        f"Async function failed: {func.__name__}",
                        error=str(e),
                        exc_info=True,
                    )
                    raise

        return wrapper

    return decorator
