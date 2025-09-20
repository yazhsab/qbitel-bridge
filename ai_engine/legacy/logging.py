"""
CRONOS AI Engine - Legacy System Whisperer Logging

Enhanced logging system for Legacy System Whisperer feature.
Provides structured, context-aware logging with enterprise-grade capabilities.
"""

import logging
import json
import time
import traceback
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import uuid
from contextlib import contextmanager, asynccontextmanager

from ..core.structured_logging import get_logger
from .exceptions import (
    LegacySystemWhispererException,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext
)
from .models import LegacySystemContext


class LogLevel(Enum):
    """Enhanced log levels for Legacy System Whisperer."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"  # Special audit log level
    SECURITY = "SECURITY"  # Security-related events


class LogCategory(Enum):
    """Categories for structured logging."""
    SYSTEM_REGISTRATION = "system_registration"
    ANOMALY_DETECTION = "anomaly_detection"
    FAILURE_PREDICTION = "failure_prediction"
    KNOWLEDGE_CAPTURE = "knowledge_capture"
    DECISION_SUPPORT = "decision_support"
    MAINTENANCE_SCHEDULING = "maintenance_scheduling"
    PERFORMANCE_MONITORING = "performance_monitoring"
    LLM_INTEGRATION = "llm_integration"
    SERVICE_LIFECYCLE = "service_lifecycle"
    SECURITY_EVENT = "security_event"
    AUDIT_TRAIL = "audit_trail"
    USER_ACTION = "user_action"
    SYSTEM_HEALTH = "system_health"
    DATA_PROCESSING = "data_processing"


@dataclass
class LogContext:
    """Structured context for legacy system logging."""
    
    # Core identifiers
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    system_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    
    # Categorization
    category: Optional[LogCategory] = None
    severity: Optional[ErrorSeverity] = None
    
    # Timing information
    timestamp: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # System context
    system_type: Optional[str] = None
    system_name: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    
    # Performance metrics
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Business context
    business_impact: Optional[str] = None
    compliance_relevance: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())[:8]
        
        if self.metadata is None:
            self.metadata = {}
        
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log context to dictionary."""
        data = asdict(self)
        
        # Convert enums to strings
        if self.category:
            data["category"] = self.category.value
        if self.severity:
            data["severity"] = self.severity.value
        if self.timestamp:
            data["timestamp"] = self.timestamp.isoformat()
        
        # Remove None values
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class AuditLogEntry:
    """Structured audit log entry for compliance and security tracking."""
    
    # Core audit information
    audit_id: str
    timestamp: datetime
    event_type: str
    actor: str  # User, system, or service performing the action
    action: str  # What action was performed
    resource: str  # What resource was affected
    outcome: str  # SUCCESS, FAILURE, PARTIAL
    
    # Context information
    system_id: Optional[str] = None
    session_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Change tracking
    previous_value: Optional[Any] = None
    new_value: Optional[Any] = None
    changes: Optional[Dict[str, Any]] = None
    
    # Compliance and security
    compliance_tags: Optional[List[str]] = None
    risk_level: Optional[str] = None
    data_classification: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None
    
    def __post_init__(self):
        """Initialize audit entry."""
        if not self.audit_id:
            self.audit_id = f"audit_{uuid.uuid4().hex[:12]}"
        
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc)
        
        if self.metadata is None:
            self.metadata = {}
        
        if self.compliance_tags is None:
            self.compliance_tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        
        # Serialize complex values
        if self.previous_value is not None:
            data["previous_value"] = json.dumps(self.previous_value, default=str)
        if self.new_value is not None:
            data["new_value"] = json.dumps(self.new_value, default=str)
        
        return {k: v for k, v in data.items() if v is not None}


class LegacySystemLogger:
    """Enhanced logger for Legacy System Whisperer with structured logging capabilities."""
    
    def __init__(
        self,
        name: str,
        base_logger: Optional[logging.Logger] = None,
        enable_audit_logging: bool = True,
        enable_performance_tracking: bool = True,
        enable_security_logging: bool = True
    ):
        """Initialize Legacy System Logger."""
        
        self.name = name
        self.base_logger = base_logger or get_logger(name)
        self.enable_audit_logging = enable_audit_logging
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_security_logging = enable_security_logging
        
        # Performance tracking
        self.operation_stats = {}
        self.error_counts = {}
        
        # Audit logging
        self.audit_entries: List[AuditLogEntry] = []
        self.max_audit_entries = 10000  # In-memory limit
        
        # Context stack for nested operations
        self.context_stack: List[LogContext] = []
        
        # Setup specialized loggers
        self.audit_logger = logging.getLogger(f"{name}.audit")
        self.security_logger = logging.getLogger(f"{name}.security")
        self.performance_logger = logging.getLogger(f"{name}.performance")
        
        # Configure formatters
        self._setup_formatters()
    
    def _setup_formatters(self) -> None:
        """Setup specialized formatters for different log types."""
        
        # Structured JSON formatter for audit logs
        audit_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "audit_entry": %(message)s}'
        )
        
        # Security event formatter
        security_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "SECURITY", '
            '"logger": "%(name)s", "security_event": %(message)s}'
        )
        
        # Performance metrics formatter
        performance_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "PERFORMANCE", '
            '"logger": "%(name)s", "performance_data": %(message)s}'
        )
        
        # Apply formatters if handlers exist
        for handler in self.audit_logger.handlers:
            handler.setFormatter(audit_formatter)
        
        for handler in self.security_logger.handlers:
            handler.setFormatter(security_formatter)
        
        for handler in self.performance_logger.handlers:
            handler.setFormatter(performance_formatter)
    
    def with_context(self, **kwargs) -> "ContextualLogger":
        """Create a contextual logger with additional context."""
        context = LogContext(**kwargs)
        return ContextualLogger(self, context)
    
    def with_system_context(self, system_context: LegacySystemContext) -> "ContextualLogger":
        """Create a contextual logger with system context."""
        context = LogContext(
            system_id=system_context.system_id,
            system_name=system_context.system_name,
            system_type=system_context.system_type.value,
            metadata={
                "system_version": system_context.version,
                "criticality": system_context.criticality.value,
                "location": system_context.location
            }
        )
        return ContextualLogger(self, context)
    
    def log_structured(
        self,
        level: Union[LogLevel, int],
        message: str,
        context: Optional[LogContext] = None,
        exception: Optional[Exception] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log structured message with context."""
        
        # Convert LogLevel enum to logging level
        if isinstance(level, LogLevel):
            log_level = getattr(logging, level.value, logging.INFO)
        else:
            log_level = level
        
        # Build log record
        log_data = {
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger_name": self.name
        }
        
        # Add context if provided
        if context:
            log_data.update(context.to_dict())
        
        # Add exception information
        if exception:
            log_data.update({
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "stack_trace": traceback.format_exc()
            })
            
            # Add Legacy System Whisperer exception details
            if isinstance(exception, LegacySystemWhispererException):
                error_context = exception.get_error_context()
                log_data["error_context"] = asdict(error_context)
        
        # Add extra data
        if extra_data:
            log_data["extra"] = extra_data
        
        # Log to appropriate logger
        if level == LogLevel.AUDIT and self.enable_audit_logging:
            self.audit_logger.info(json.dumps(log_data))
        elif level == LogLevel.SECURITY and self.enable_security_logging:
            self.security_logger.warning(json.dumps(log_data))
        else:
            # Use base logger with structured format
            self.base_logger.log(log_level, json.dumps(log_data))
        
        # Update statistics
        self._update_log_statistics(level, context, exception)
    
    def log_audit_event(self, audit_entry: AuditLogEntry) -> None:
        """Log audit event for compliance tracking."""
        
        if not self.enable_audit_logging:
            return
        
        # Add to in-memory storage
        self.audit_entries.append(audit_entry)
        
        # Maintain size limit
        if len(self.audit_entries) > self.max_audit_entries:
            self.audit_entries = self.audit_entries[-self.max_audit_entries:]
        
        # Log audit entry
        self.audit_logger.info(json.dumps(audit_entry.to_dict()))
        
        # Log to main logger as well for visibility
        self.log_structured(
            LogLevel.INFO,
            f"Audit Event: {audit_entry.action} on {audit_entry.resource}",
            LogContext(
                category=LogCategory.AUDIT_TRAIL,
                metadata={
                    "audit_id": audit_entry.audit_id,
                    "actor": audit_entry.actor,
                    "outcome": audit_entry.outcome
                }
            )
        )
    
    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[LogContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security-related event."""
        
        if not self.enable_security_logging:
            return
        
        security_data = {
            "event_type": event_type,
            "description": description,
            "severity": severity.value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if context:
            security_data.update(context.to_dict())
        
        if metadata:
            security_data["metadata"] = metadata
        
        # Log security event
        self.security_logger.warning(json.dumps(security_data))
        
        # Also log audit entry for security events
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            audit_entry = AuditLogEntry(
                audit_id=f"sec_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(timezone.utc),
                event_type="security_event",
                actor="system",
                action=event_type,
                resource="legacy_system_whisperer",
                outcome="ALERT",
                risk_level=severity.value,
                metadata=metadata or {}
            )
            self.log_audit_event(audit_entry)
    
    def log_performance_metric(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log performance metrics for operations."""
        
        if not self.enable_performance_tracking:
            return
        
        performance_data = {
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if metadata:
            performance_data["metadata"] = metadata
        
        # Update operation statistics
        if operation not in self.operation_stats:
            self.operation_stats[operation] = {
                "count": 0,
                "total_duration": 0.0,
                "success_count": 0,
                "failure_count": 0,
                "avg_duration": 0.0
            }
        
        stats = self.operation_stats[operation]
        stats["count"] += 1
        stats["total_duration"] += duration_ms
        stats["avg_duration"] = stats["total_duration"] / stats["count"]
        
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
        
        # Add statistics to performance data
        performance_data["operation_stats"] = stats.copy()
        
        # Log performance data
        self.performance_logger.info(json.dumps(performance_data))
    
    def _update_log_statistics(
        self,
        level: LogLevel,
        context: Optional[LogContext],
        exception: Optional[Exception]
    ) -> None:
        """Update internal logging statistics."""
        
        # Count errors by category
        if exception and context and context.category:
            category_key = context.category.value
            if category_key not in self.error_counts:
                self.error_counts[category_key] = 0
            self.error_counts[category_key] += 1
    
    @contextmanager
    def operation_context(
        self,
        operation: str,
        system_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracking operation performance and logging."""
        
        start_time = time.time()
        context = LogContext(
            operation=operation,
            system_id=system_id,
            metadata=metadata,
            component=self.name
        )
        
        self.context_stack.append(context)
        
        try:
            self.log_structured(
                LogLevel.DEBUG,
                f"Starting operation: {operation}",
                context
            )
            
            yield context
            
            # Operation completed successfully
            duration_ms = (time.time() - start_time) * 1000
            context.duration_ms = duration_ms
            
            self.log_structured(
                LogLevel.DEBUG,
                f"Completed operation: {operation} in {duration_ms:.2f}ms",
                context
            )
            
            self.log_performance_metric(operation, duration_ms, True, metadata)
            
        except Exception as e:
            # Operation failed
            duration_ms = (time.time() - start_time) * 1000
            context.duration_ms = duration_ms
            
            self.log_structured(
                LogLevel.ERROR,
                f"Failed operation: {operation} after {duration_ms:.2f}ms",
                context,
                exception=e
            )
            
            self.log_performance_metric(operation, duration_ms, False, metadata)
            raise
            
        finally:
            if self.context_stack and self.context_stack[-1] == context:
                self.context_stack.pop()
    
    @asynccontextmanager
    async def async_operation_context(
        self,
        operation: str,
        system_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Async context manager for tracking operation performance and logging."""
        
        start_time = time.time()
        context = LogContext(
            operation=operation,
            system_id=system_id,
            metadata=metadata,
            component=self.name
        )
        
        self.context_stack.append(context)
        
        try:
            self.log_structured(
                LogLevel.DEBUG,
                f"Starting async operation: {operation}",
                context
            )
            
            yield context
            
            # Operation completed successfully
            duration_ms = (time.time() - start_time) * 1000
            context.duration_ms = duration_ms
            
            self.log_structured(
                LogLevel.DEBUG,
                f"Completed async operation: {operation} in {duration_ms:.2f}ms",
                context
            )
            
            self.log_performance_metric(operation, duration_ms, True, metadata)
            
        except Exception as e:
            # Operation failed
            duration_ms = (time.time() - start_time) * 1000
            context.duration_ms = duration_ms
            
            self.log_structured(
                LogLevel.ERROR,
                f"Failed async operation: {operation} after {duration_ms:.2f}ms",
                context,
                exception=e
            )
            
            self.log_performance_metric(operation, duration_ms, False, metadata)
            raise
            
        finally:
            if self.context_stack and self.context_stack[-1] == context:
                self.context_stack.pop()
    
    def get_current_context(self) -> Optional[LogContext]:
        """Get current logging context from stack."""
        return self.context_stack[-1] if self.context_stack else None
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get operation performance statistics."""
        return {
            "operation_stats": self.operation_stats.copy(),
            "error_counts": self.error_counts.copy(),
            "total_operations": sum(stats["count"] for stats in self.operation_stats.values()),
            "total_errors": sum(self.error_counts.values()),
            "audit_entries_count": len(self.audit_entries)
        }
    
    def reset_statistics(self) -> None:
        """Reset internal statistics."""
        self.operation_stats.clear()
        self.error_counts.clear()
        self.audit_entries.clear()
    
    # Convenience methods for common log levels
    def trace(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log trace message."""
        self.log_structured(LogLevel.TRACE, message, context, **kwargs)
    
    def debug(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log debug message."""
        self.log_structured(LogLevel.DEBUG, message, context, **kwargs)
    
    def info(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log info message."""
        self.log_structured(LogLevel.INFO, message, context, **kwargs)
    
    def warning(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log warning message."""
        self.log_structured(LogLevel.WARNING, message, context, **kwargs)
    
    def error(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log error message."""
        self.log_structured(LogLevel.ERROR, message, context, **kwargs)
    
    def critical(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log critical message."""
        self.log_structured(LogLevel.CRITICAL, message, context, **kwargs)


class ContextualLogger:
    """Logger with pre-configured context."""
    
    def __init__(self, base_logger: LegacySystemLogger, context: LogContext):
        """Initialize contextual logger."""
        self.base_logger = base_logger
        self.context = context
    
    def log_structured(
        self,
        level: Union[LogLevel, int],
        message: str,
        additional_context: Optional[LogContext] = None,
        **kwargs
    ) -> None:
        """Log with combined context."""
        
        # Merge contexts
        final_context = LogContext(**asdict(self.context))
        if additional_context:
            # Update with additional context
            for key, value in asdict(additional_context).items():
                if value is not None:
                    setattr(final_context, key, value)
        
        self.base_logger.log_structured(level, message, final_context, **kwargs)
    
    # Convenience methods
    def trace(self, message: str, **kwargs):
        self.log_structured(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self.log_structured(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.log_structured(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log_structured(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log_structured(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.log_structured(LogLevel.CRITICAL, message, **kwargs)
    
    def with_context(self, **kwargs) -> "ContextualLogger":
        """Create new contextual logger with additional context."""
        new_context = LogContext(**asdict(self.context))
        for key, value in kwargs.items():
            if value is not None:
                setattr(new_context, key, value)
        return ContextualLogger(self.base_logger, new_context)


def create_legacy_logger(
    name: str,
    enable_audit: bool = True,
    enable_performance: bool = True,
    enable_security: bool = True
) -> LegacySystemLogger:
    """Create a Legacy System Whisperer logger instance."""
    
    return LegacySystemLogger(
        name=name,
        enable_audit_logging=enable_audit,
        enable_performance_tracking=enable_performance,
        enable_security_logging=enable_security
    )


def setup_legacy_logging_handlers(
    log_dir: Union[str, Path],
    service_name: str = "legacy-system-whisperer"
) -> None:
    """Setup file handlers for Legacy System Whisperer logging."""
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Main application log
    main_handler = logging.FileHandler(log_path / f"{service_name}.log")
    main_handler.setLevel(logging.DEBUG)
    
    # Audit log
    audit_handler = logging.FileHandler(log_path / f"{service_name}-audit.log")
    audit_handler.setLevel(logging.INFO)
    
    # Security log
    security_handler = logging.FileHandler(log_path / f"{service_name}-security.log")
    security_handler.setLevel(logging.WARNING)
    
    # Performance log
    performance_handler = logging.FileHandler(log_path / f"{service_name}-performance.log")
    performance_handler.setLevel(logging.INFO)
    
    # Add handlers to appropriate loggers
    main_logger = logging.getLogger(service_name)
    main_logger.addHandler(main_handler)
    
    audit_logger = logging.getLogger(f"{service_name}.audit")
    audit_logger.addHandler(audit_handler)
    
    security_logger = logging.getLogger(f"{service_name}.security")
    security_logger.addHandler(security_handler)
    
    performance_logger = logging.getLogger(f"{service_name}.performance")
    performance_logger.addHandler(performance_handler)


# Decorator for automatic operation logging
def log_operation(
    operation_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    logger: Optional[LegacySystemLogger] = None
):
    """Decorator to automatically log function operations."""
    
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                op_logger = logger or create_legacy_logger(func.__module__)
                
                context = LogContext(
                    operation=op_name,
                    component=func.__module__
                )
                
                if log_args:
                    context.metadata = context.metadata or {}
                    context.metadata["args"] = str(args)
                    context.metadata["kwargs"] = str(kwargs)
                
                async with op_logger.async_operation_context(op_name, metadata=context.metadata):
                    result = await func(*args, **kwargs)
                    
                    if log_result:
                        op_logger.debug(f"Operation {op_name} result: {result}")
                    
                    return result
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                op_logger = logger or create_legacy_logger(func.__module__)
                
                context = LogContext(
                    operation=op_name,
                    component=func.__module__
                )
                
                if log_args:
                    context.metadata = context.metadata or {}
                    context.metadata["args"] = str(args)
                    context.metadata["kwargs"] = str(kwargs)
                
                with op_logger.operation_context(op_name, metadata=context.metadata):
                    result = func(*args, **kwargs)
                    
                    if log_result:
                        op_logger.debug(f"Operation {op_name} result: {result}")
                    
                    return result
            
            return sync_wrapper
    
    return decorator


# Global logger registry
_loggers: Dict[str, LegacySystemLogger] = {}


def get_legacy_logger(name: str) -> LegacySystemLogger:
    """Get or create a Legacy System Whisperer logger."""
    
    if name not in _loggers:
        _loggers[name] = create_legacy_logger(name)
    
    return _loggers[name]


def shutdown_legacy_logging() -> None:
    """Shutdown all Legacy System Whisperer loggers and handlers."""
    
    for logger in _loggers.values():
        # Reset statistics
        logger.reset_statistics()
        
        # Close handlers
        for handler in logger.base_logger.handlers:
            handler.close()
        
        for handler in logger.audit_logger.handlers:
            handler.close()
        
        for handler in logger.security_logger.handlers:
            handler.close()
        
        for handler in logger.performance_logger.handlers:
            handler.close()
    
    _loggers.clear()