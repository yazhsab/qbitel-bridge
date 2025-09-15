"""
CRONOS AI Engine - Structured Logging System

This module provides enterprise-grade structured logging with correlation IDs,
performance metrics, security logging, and integration with monitoring systems.
"""

import asyncio
import logging
import logging.config
import json
import time
import traceback
import uuid
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import sys
import os
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import threading

from .exceptions import CronosAIException


class LogLevel(Enum):
    """Enhanced log levels for structured logging."""
    TRACE = "TRACE"      # Very detailed debugging
    DEBUG = "DEBUG"      # Debugging information
    INFO = "INFO"        # General information
    WARN = "WARN"        # Warning messages
    ERROR = "ERROR"      # Error conditions
    CRITICAL = "CRITICAL"  # Critical failures
    SECURITY = "SECURITY"  # Security-related events
    PERFORMANCE = "PERFORMANCE"  # Performance metrics
    AUDIT = "AUDIT"      # Audit trail events


class LogCategory(Enum):
    """Log categories for filtering and routing."""
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    AUDIT = "audit"
    PROTOCOL_DISCOVERY = "protocol_discovery"
    MODEL_INFERENCE = "model_inference"
    DATA_PROCESSING = "data_processing"
    EXTERNAL_API = "external_api"
    USER_ACTION = "user_action"


@dataclass
class LogContext:
    """Context information for structured logging."""
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    correlation_id: Optional[str] = None
    additional_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Remove None values
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class LogEvent:
    """Structured log event."""
    timestamp: float
    level: LogLevel
    category: LogCategory
    message: str
    component: str
    operation: Optional[str] = None
    context: Optional[LogContext] = None
    exception: Optional[Exception] = None
    performance_metrics: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'timestamp': self.timestamp,
            'iso_timestamp': datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat(),
            'level': self.level.value,
            'category': self.category.value,
            'message': self.message,
            'component': self.component,
            'operation': self.operation,
            'metadata': self.metadata
        }
        
        if self.context:
            result['context'] = self.context.to_dict()
        
        if self.exception:
            result['exception'] = {
                'type': type(self.exception).__name__,
                'message': str(self.exception),
                'traceback': traceback.format_exception(type(self.exception), self.exception, self.exception.__traceback__)
            }
        
        if self.performance_metrics:
            result['performance_metrics'] = self.performance_metrics
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Extract structured data from record
        context = getattr(record, 'context', None)
        category = getattr(record, 'category', LogCategory.SYSTEM)
        performance_metrics = getattr(record, 'performance_metrics', None)
        metadata = getattr(record, 'metadata', {})
        
        # Create log event
        log_event = LogEvent(
            timestamp=record.created,
            level=LogLevel(record.levelname),
            category=category,
            message=record.getMessage(),
            component=getattr(record, 'component', record.name),
            operation=getattr(record, 'operation', None),
            context=context,
            exception=record.exc_info[1] if record.exc_info else None,
            performance_metrics=performance_metrics,
            metadata=metadata
        )
        
        return log_event.to_json()


class PerformanceLogger:
    """Logger specifically for performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.operation_times: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
    
    @contextmanager
    def time_operation(self, operation_name: str, context: Optional[LogContext] = None):
        """Context manager to time operations."""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            
            # Record timing
            with self.lock:
                if operation_name not in self.operation_times:
                    self.operation_times[operation_name] = []
                self.operation_times[operation_name].append(duration)
                
                # Keep only last 1000 measurements
                if len(self.operation_times[operation_name]) > 1000:
                    self.operation_times[operation_name] = self.operation_times[operation_name][-1000:]
            
            # Log performance metric
            self.logger.info(
                f"Operation {operation_name} completed",
                extra={
                    'category': LogCategory.PERFORMANCE,
                    'performance_metrics': {
                        'operation': operation_name,
                        'duration_seconds': duration,
                        'duration_ms': duration * 1000
                    },
                    'context': context
                }
            )
    
    def get_operation_stats(self, operation_name: str) -> Optional[Dict[str, float]]:
        """Get statistics for an operation."""
        with self.lock:
            times = self.operation_times.get(operation_name)
            if not times:
                return None
            
            return {
                'count': len(times),
                'avg_duration': sum(times) / len(times),
                'min_duration': min(times),
                'max_duration': max(times),
                'total_duration': sum(times)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        with self.lock:
            return {
                operation: self.get_operation_stats(operation)
                for operation in self.operation_times.keys()
            }


class SecurityLogger:
    """Logger specifically for security events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_authentication_attempt(
        self,
        user_id: Optional[str],
        success: bool,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        context: Optional[LogContext] = None
    ):
        """Log authentication attempt."""
        message = f"Authentication {'successful' if success else 'failed'}"
        if user_id:
            message += f" for user {user_id}"
        
        self.logger.info(
            message,
            extra={
                'category': LogCategory.SECURITY,
                'metadata': {
                    'event_type': 'authentication',
                    'success': success,
                    'user_id': user_id,
                    'source_ip': source_ip,
                    'user_agent': user_agent
                },
                'context': context
            }
        )
    
    def log_authorization_check(
        self,
        user_id: Optional[str],
        resource: str,
        action: str,
        granted: bool,
        context: Optional[LogContext] = None
    ):
        """Log authorization check."""
        result = "granted" if granted else "denied"
        message = f"Access {result} for user {user_id} to {action} on {resource}"
        
        self.logger.info(
            message,
            extra={
                'category': LogCategory.SECURITY,
                'metadata': {
                    'event_type': 'authorization',
                    'user_id': user_id,
                    'resource': resource,
                    'action': action,
                    'granted': granted
                },
                'context': context
            }
        )
    
    def log_security_violation(
        self,
        violation_type: str,
        description: str,
        severity: str = "medium",
        context: Optional[LogContext] = None
    ):
        """Log security violation."""
        self.logger.warning(
            f"Security violation: {description}",
            extra={
                'category': LogCategory.SECURITY,
                'metadata': {
                    'event_type': 'security_violation',
                    'violation_type': violation_type,
                    'severity': severity,
                    'description': description
                },
                'context': context
            }
        )


class AuditLogger:
    """Logger specifically for audit trail."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_data_access(
        self,
        user_id: Optional[str],
        resource_type: str,
        resource_id: str,
        action: str,
        context: Optional[LogContext] = None
    ):
        """Log data access event."""
        message = f"User {user_id} performed {action} on {resource_type}:{resource_id}"
        
        self.logger.info(
            message,
            extra={
                'category': LogCategory.AUDIT,
                'metadata': {
                    'event_type': 'data_access',
                    'user_id': user_id,
                    'resource_type': resource_type,
                    'resource_id': resource_id,
                    'action': action
                },
                'context': context
            }
        )
    
    def log_configuration_change(
        self,
        user_id: Optional[str],
        component: str,
        setting: str,
        old_value: Any,
        new_value: Any,
        context: Optional[LogContext] = None
    ):
        """Log configuration change."""
        message = f"Configuration change in {component}: {setting} changed"
        
        self.logger.info(
            message,
            extra={
                'category': LogCategory.AUDIT,
                'metadata': {
                    'event_type': 'configuration_change',
                    'user_id': user_id,
                    'component': component,
                    'setting': setting,
                    'old_value': str(old_value),
                    'new_value': str(new_value)
                },
                'context': context
            }
        )
    
    def log_model_training(
        self,
        user_id: Optional[str],
        model_name: str,
        training_data_size: int,
        training_duration: float,
        model_accuracy: Optional[float] = None,
        context: Optional[LogContext] = None
    ):
        """Log model training event."""
        message = f"Model {model_name} trained with {training_data_size} samples"
        
        self.logger.info(
            message,
            extra={
                'category': LogCategory.AUDIT,
                'metadata': {
                    'event_type': 'model_training',
                    'user_id': user_id,
                    'model_name': model_name,
                    'training_data_size': training_data_size,
                    'training_duration': training_duration,
                    'model_accuracy': model_accuracy
                },
                'context': context
            }
        )


class StructuredLogger:
    """Main structured logger with specialized sub-loggers."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        
        # Set up base logger
        self.logger = logging.getLogger(name)
        self._setup_logger()
        
        # Specialized loggers
        self.performance = PerformanceLogger(self.logger)
        self.security = SecurityLogger(self.logger)
        self.audit = AuditLogger(self.logger)
        
        # Context tracking
        self._context_stack: List[LogContext] = []
        self._local = threading.local()
    
    def _setup_logger(self):
        """Setup logger with structured formatter."""
        if self.logger.handlers:
            return  # Already configured
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(console_handler)
        
        # Create file handler if configured
        if 'log_file' in self.config:
            file_handler = logging.FileHandler(self.config['log_file'])
            file_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(file_handler)
        
        # Set log level
        log_level = self.config.get('log_level', 'INFO')
        self.logger.setLevel(getattr(logging, log_level.upper()))
    
    def push_context(self, context: LogContext):
        """Push context onto stack."""
        if not hasattr(self._local, 'contexts'):
            self._local.contexts = []
        self._local.contexts.append(context)
    
    def pop_context(self) -> Optional[LogContext]:
        """Pop context from stack."""
        if hasattr(self._local, 'contexts') and self._local.contexts:
            return self._local.contexts.pop()
        return None
    
    def get_current_context(self) -> Optional[LogContext]:
        """Get current context."""
        if hasattr(self._local, 'contexts') and self._local.contexts:
            return self._local.contexts[-1]
        return None
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for logging context."""
        ctx = LogContext(**kwargs)
        self.push_context(ctx)
        try:
            yield ctx
        finally:
            self.pop_context()
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[LogContext] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """Internal logging method."""
        # Use current context if none provided
        if context is None:
            context = self.get_current_context()
        
        # Log with extra fields
        extra = {
            'category': category,
            'component': component or self.name,
            'operation': operation,
            'context': context,
            'performance_metrics': performance_metrics,
            'metadata': metadata or {}
        }
        
        # Map LogLevel to logging level
        level_map = {
            LogLevel.TRACE: logging.DEBUG,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARN: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
            LogLevel.SECURITY: logging.WARNING,
            LogLevel.PERFORMANCE: logging.INFO,
            LogLevel.AUDIT: logging.INFO
        }
        
        log_level = level_map.get(level, logging.INFO)
        
        if exception:
            self.logger.log(log_level, message, extra=extra, exc_info=exception)
        else:
            self.logger.log(log_level, message, extra=extra)
    
    def trace(self, message: str, **kwargs):
        """Log trace message."""
        self._log(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warn(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARN, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message (alias)."""
        self.warn(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)


class LoggerFactory:
    """Factory for creating structured loggers."""
    
    _loggers: Dict[str, StructuredLogger] = {}
    _default_config: Dict[str, Any] = {
        'log_level': 'INFO',
        'log_file': None
    }
    
    @classmethod
    def get_logger(cls, name: str, config: Optional[Dict[str, Any]] = None) -> StructuredLogger:
        """Get or create a structured logger."""
        if name not in cls._loggers:
            logger_config = {**cls._default_config}
            if config:
                logger_config.update(config)
            cls._loggers[name] = StructuredLogger(name, logger_config)
        
        return cls._loggers[name]
    
    @classmethod
    def configure_logging(cls, config: Dict[str, Any]):
        """Configure logging system."""
        cls._default_config.update(config)
        
        # Apply configuration to existing loggers
        for logger in cls._loggers.values():
            logger.config.update(config)
            logger._setup_logger()
    
    @classmethod
    def get_all_loggers(cls) -> Dict[str, StructuredLogger]:
        """Get all created loggers."""
        return cls._loggers.copy()


# Convenience function to get logger
def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> StructuredLogger:
    """Get a structured logger."""
    return LoggerFactory.get_logger(name, config)


# Default logging configuration
DEFAULT_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'structured': {
            '()': StructuredFormatter,
            'include_context': True
        },
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'structured',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'structured',
            'filename': 'cronos_ai.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        'cronos.ai': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}


def configure_logging(config: Optional[Dict[str, Any]] = None):
    """Configure the logging system."""
    logging_config = config or DEFAULT_LOGGING_CONFIG
    
    # Apply logging configuration
    logging.config.dictConfig(logging_config)
    
    # Configure structured logger factory
    LoggerFactory.configure_logging({
        'log_level': logging_config.get('root', {}).get('level', 'INFO'),
        'log_file': None  # File logging handled by dictConfig
    })


# Initialize logging with default configuration
if not logging.getLogger().handlers:
    configure_logging()