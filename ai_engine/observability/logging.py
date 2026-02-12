"""
Structured Logging

Production-ready structured logging with:
- Correlation IDs for request tracking
- JSON output for log aggregation
- Log levels and filtering
- Context propagation
- Sensitive data masking

Integrates with:
- ELK Stack
- Splunk
- CloudWatch
- Datadog
"""

import logging
import json
import sys
import threading
import traceback
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Union
import uuid
import re


class LogLevel(Enum):
    """Log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        """Get numeric level."""
        return {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }[self.value]


@dataclass
class LogContext:
    """Context for structured logging."""

    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    service_name: str = "qbitel"
    environment: str = "production"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "correlation_id": self.correlation_id,
            "service": self.service_name,
            "environment": self.environment,
        }

        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.user_id:
            result["user_id"] = self.user_id
        if self.session_id:
            result["session_id"] = self.session_id
        if self.request_id:
            result["request_id"] = self.request_id
        if self.extra:
            result.update(self.extra)

        return result


class SensitiveDataMasker:
    """Masks sensitive data in log output."""

    # Patterns for sensitive data
    PATTERNS = [
        # Credit card numbers (basic pattern)
        (re.compile(r"\b\d{13,19}\b"), "[CARD_NUMBER]"),
        # PAN with separators
        (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[CARD_NUMBER]"),
        # CVV
        (re.compile(r"\bcvv[:\s]*\d{3,4}\b", re.IGNORECASE), "cvv: [CVV]"),
        # Password fields
        (re.compile(r"password['\"]?\s*[=:]\s*['\"]?[^'\",\s]+", re.IGNORECASE), "password: [REDACTED]"),
        # API keys
        (re.compile(r"api[_-]?key['\"]?\s*[=:]\s*['\"]?[\w-]+", re.IGNORECASE), "api_key: [REDACTED]"),
        # Bearer tokens
        (re.compile(r"bearer\s+[\w\-._~+/]+=*", re.IGNORECASE), "Bearer [TOKEN]"),
        # SSN
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
        # Email addresses (partial mask)
        (re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b"), "[EMAIL]"),
    ]

    # Keys to mask in dictionaries
    SENSITIVE_KEYS = {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "credentials",
        "private_key",
        "pan",
        "card_number",
        "cvv",
        "cvc",
        "pin",
        "ssn",
        "social_security",
    }

    @classmethod
    def mask(cls, data: Any) -> Any:
        """
        Mask sensitive data.

        Args:
            data: Data to mask

        Returns:
            Masked data
        """
        if isinstance(data, str):
            return cls._mask_string(data)
        elif isinstance(data, dict):
            return cls._mask_dict(data)
        elif isinstance(data, (list, tuple)):
            return type(data)(cls.mask(item) for item in data)
        return data

    @classmethod
    def _mask_string(cls, text: str) -> str:
        """Mask sensitive patterns in string."""
        result = text
        for pattern, replacement in cls.PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    @classmethod
    def _mask_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive keys in dictionary."""
        result = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in cls.SENSITIVE_KEYS):
                result[key] = "[REDACTED]"
            else:
                result[key] = cls.mask(value)
        return result


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def __init__(
        self,
        service_name: str = "qbitel",
        environment: str = "production",
        mask_sensitive: bool = True,
    ):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.mask_sensitive = mask_sensitive

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
        }

        # Add source info
        log_entry["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add context if available
        if hasattr(record, "context"):
            log_entry["context"] = record.context

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry["extra"] = record.extra_fields

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": traceback.format_exception(*record.exc_info),
            }

        # Mask sensitive data
        if self.mask_sensitive:
            log_entry = SensitiveDataMasker.mask(log_entry)

        return json.dumps(log_entry, default=str)


class StructuredLogger:
    """
    Structured logger with context support.

    Provides:
    - Structured JSON output
    - Correlation ID tracking
    - Sensitive data masking
    - Context propagation
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        context: Optional[LogContext] = None,
    ):
        self.name = name
        self._context = context or LogContext()
        self._context_var = threading.local()

        # Get or create underlying logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.level)

    @property
    def context(self) -> LogContext:
        """Get current context."""
        return getattr(self._context_var, "context", None) or self._context

    @context.setter
    def context(self, ctx: LogContext) -> None:
        """Set current context."""
        self._context_var.context = ctx

    @contextmanager
    def with_context(
        self,
        **kwargs: Any,
    ) -> Generator["StructuredLogger", None, None]:
        """Context manager for temporary context."""
        old_context = self.context
        new_context = LogContext(
            correlation_id=kwargs.get("correlation_id", old_context.correlation_id),
            trace_id=kwargs.get("trace_id", old_context.trace_id),
            span_id=kwargs.get("span_id", old_context.span_id),
            user_id=kwargs.get("user_id", old_context.user_id),
            service_name=old_context.service_name,
            environment=old_context.environment,
            extra={**old_context.extra, **kwargs.get("extra", {})},
        )
        self.context = new_context
        try:
            yield self
        finally:
            self.context = old_context

    def _log(
        self,
        level: LogLevel,
        message: str,
        exc_info: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Internal log method."""
        record = self._logger.makeRecord(
            self.name,
            level.level,
            "",  # pathname (will be overwritten)
            0,  # lineno (will be overwritten)
            message,
            (),
            exc_info,
        )

        # Add context
        record.context = self.context.to_dict()

        # Add extra fields
        if kwargs:
            record.extra_fields = kwargs

        self._logger.handle(record)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, exc_info=exc_info, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._log(LogLevel.ERROR, message, exc_info=sys.exc_info(), **kwargs)

    # Banking-specific logging methods
    def log_transaction(
        self,
        transaction_id: str,
        protocol: str,
        amount: float,
        currency: str,
        status: str,
        **kwargs: Any,
    ) -> None:
        """Log a transaction event."""
        self.info(
            f"Transaction {transaction_id}: {status}",
            transaction_id=transaction_id,
            protocol=protocol,
            amount=amount,
            currency=currency,
            status=status,
            event_type="transaction",
            **kwargs,
        )

    def log_crypto_operation(
        self,
        operation: str,
        algorithm: str,
        duration_ms: float,
        success: bool,
        **kwargs: Any,
    ) -> None:
        """Log a cryptographic operation."""
        level = LogLevel.INFO if success else LogLevel.WARNING
        self._log(
            level,
            f"Crypto operation {operation}: {'success' if success else 'failed'}",
            operation=operation,
            algorithm=algorithm,
            duration_ms=duration_ms,
            success=success,
            event_type="crypto_operation",
            **kwargs,
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        **kwargs: Any,
    ) -> None:
        """Log a security event."""
        level = {
            "low": LogLevel.INFO,
            "medium": LogLevel.WARNING,
            "high": LogLevel.ERROR,
            "critical": LogLevel.CRITICAL,
        }.get(severity.lower(), LogLevel.WARNING)

        self._log(
            level,
            f"Security event: {event_type}",
            event_type=event_type,
            severity=severity,
            description=description,
            **kwargs,
        )


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}
_loggers_lock = threading.Lock()


def get_logger(
    name: str = "qbitel",
    level: LogLevel = LogLevel.INFO,
) -> StructuredLogger:
    """Get or create a structured logger."""
    with _loggers_lock:
        if name not in _loggers:
            _loggers[name] = StructuredLogger(name, level)
        return _loggers[name]


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    format_json: bool = True,
    service_name: str = "qbitel",
    environment: str = "production",
    mask_sensitive: bool = True,
    output: str = "stdout",
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Minimum log level
        format_json: Use JSON formatting
        service_name: Service name in logs
        environment: Environment name
        mask_sensitive: Mask sensitive data
        output: Output destination (stdout, stderr, or filename)
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    if output == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    elif output == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(output)

    # Set formatter
    if format_json:
        formatter = JsonFormatter(service_name, environment, mask_sensitive)
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    handler.setLevel(level.level)

    # Configure root logger
    root_logger.setLevel(level.level)
    root_logger.addHandler(handler)

    # Log configuration
    logger = get_logger("qbitel.logging")
    logger.info(
        "Logging configured",
        level=level.value,
        format="json" if format_json else "text",
        output=output,
    )
