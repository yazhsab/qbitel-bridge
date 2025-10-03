"""
CRONOS AI Engine - Comprehensive Error Handling System

This module provides enterprise-grade error handling, recovery strategies,
and structured logging for the protocol discovery system.
"""

import asyncio
import functools
import logging
import sys
import traceback
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable, Type
from enum import Enum
import json
import uuid
import random
from collections import Counter

from .exceptions import CronosAIException, ProtocolException, ModelException, InferenceException


class ErrorSeverity(Enum):
    """Error severity levels for categorization and handling."""
    LOW = "low"              # Minor issues, system can continue
    MEDIUM = "medium"        # Significant issues, may impact performance
    HIGH = "high"           # Serious issues, component may fail
    CRITICAL = "critical"    # System-threatening issues, immediate attention required


class ErrorCategory(Enum):
    """Error categories for classification and routing."""
    CONFIGURATION = "configuration"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_VALIDATION = "data_validation"
    MODEL_INFERENCE = "model_inference"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_DEPENDENCY = "external_dependency"
    INTERNAL_LOGIC = "internal_logic"
    PERFORMANCE = "performance"
    SECURITY = "security"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"                    # Retry the operation
    FALLBACK = "fallback"             # Use fallback mechanism
    CIRCUIT_BREAK = "circuit_break"   # Stop calling failing service
    DEGRADE = "degrade"               # Reduce functionality
    FAIL_FAST = "fail_fast"           # Fail immediately
    IGNORE = "ignore"                 # Log and continue


@dataclass
class ErrorContext:
    """Context information for error handling."""
    component: str
    operation: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """Detailed error record for tracking and analysis."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    exception_type: str
    exception_message: str
    stack_trace: str
    context: ErrorContext
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error record to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'category': self.category.value,
            'component': self.component,
            'operation': self.operation,
            'exception_type': self.exception_type,
            'exception_message': self.exception_message,
            'stack_trace': self.stack_trace,
            'context': {
                'component': self.context.component,
                'operation': self.context.operation,
                'request_id': self.context.request_id,
                'user_id': self.context.user_id,
                'session_id': self.context.session_id,
                'trace_id': self.context.trace_id,
                'additional_data': self.context.additional_data
            },
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'recovery_strategy': self.recovery_strategy.value if self.recovery_strategy else None,
            'retry_count': self.retry_count,
            'metadata': self.metadata
        }


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # exponential, linear, constant
    retryable_exceptions: Set[Type[Exception]] = field(default_factory=lambda: {
        ConnectionError, TimeoutError, InferenceException
    })


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    expected_exception_types: Set[Type[Exception]] = field(default_factory=lambda: {
        ConnectionError, TimeoutError
    })


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is recovered


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise CronosAIException("Circuit breaker is OPEN")
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise CronosAIException("Circuit breaker HALF_OPEN call limit reached")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker transitioned to CLOSED after recovery")
            
            return result
            
        except Exception as e:
            if type(e) in self.config.expected_exception_types:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.OPEN
                    self.logger.warning("Circuit breaker transitioned back to OPEN")
                elif self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    self.logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
            
            raise


class ErrorHandler:
    """
    Comprehensive error handling system with recovery strategies,
    circuit breakers, and detailed error tracking.
    """
    
    def __init__(self, enable_persistent_storage: bool = True, enable_sentry: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_records: deque = deque(maxlen=10000)  # Keep last 10k errors
        self.error_stats = defaultdict(int)
        self.component_errors = defaultdict(lambda: defaultdict(int))
        
        # Recovery mechanisms
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_configs: Dict[str, RetryConfig] = {}
        
        # Error classification rules
        self.classification_rules = self._initialize_classification_rules()
        
        # External integrations
        self.enable_persistent_storage = enable_persistent_storage
        self.enable_sentry = enable_sentry
        self.persistent_storage = None
        self.sentry_tracker = None
        
    def _initialize_classification_rules(self) -> Dict[Type[Exception], Tuple[ErrorSeverity, ErrorCategory, RecoveryStrategy]]:
        """Initialize error classification rules."""
        return {
            # Configuration errors
            ValueError: (ErrorSeverity.HIGH, ErrorCategory.CONFIGURATION, RecoveryStrategy.FAIL_FAST),
            KeyError: (ErrorSeverity.HIGH, ErrorCategory.CONFIGURATION, RecoveryStrategy.FAIL_FAST),
            
            # Network errors
            ConnectionError: (ErrorSeverity.MEDIUM, ErrorCategory.NETWORK, RecoveryStrategy.RETRY),
            TimeoutError: (ErrorSeverity.MEDIUM, ErrorCategory.NETWORK, RecoveryStrategy.RETRY),
            
            # Model errors
            ModelException: (ErrorSeverity.HIGH, ErrorCategory.MODEL_INFERENCE, RecoveryStrategy.FALLBACK),
            InferenceException: (ErrorSeverity.MEDIUM, ErrorCategory.MODEL_INFERENCE, RecoveryStrategy.RETRY),
            
            # Protocol errors
            ProtocolException: (ErrorSeverity.MEDIUM, ErrorCategory.DATA_VALIDATION, RecoveryStrategy.DEGRADE),
            
            # Resource errors
            MemoryError: (ErrorSeverity.CRITICAL, ErrorCategory.RESOURCE_EXHAUSTION, RecoveryStrategy.CIRCUIT_BREAK),
            
            # Generic errors
            Exception: (ErrorSeverity.MEDIUM, ErrorCategory.INTERNAL_LOGIC, RecoveryStrategy.RETRY),
        }
    
    def classify_error(self, exception: Exception) -> Tuple[ErrorSeverity, ErrorCategory, RecoveryStrategy]:
        """Classify an error and determine handling strategy."""
        exc_type = type(exception)
        
        # Direct match
        if exc_type in self.classification_rules:
            return self.classification_rules[exc_type]
        
        # Check inheritance hierarchy
        for base_type, (severity, category, strategy) in self.classification_rules.items():
            if issubclass(exc_type, base_type):
                return severity, category, strategy
        
        # Default classification
        return ErrorSeverity.MEDIUM, ErrorCategory.INTERNAL_LOGIC, RecoveryStrategy.RETRY
    
    async def handle_error(
        self,
        exception: Exception,
        context: ErrorContext,
        recovery_func: Optional[Callable] = None
    ) -> Tuple[bool, Any]:
        """
        Handle an error with appropriate recovery strategy.
        
        Returns:
            (recovery_successful, result_or_none)
        """
        error_id = str(uuid.uuid4())
        severity, category, strategy = self.classify_error(exception)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=time.time(),
            severity=severity,
            category=category,
            component=context.component,
            operation=context.operation,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            context=context
        )
        
        # Log error
        log_level = self._get_log_level(severity)
        self.logger.log(
            log_level,
            f"Error {error_id} in {context.component}.{context.operation}: {exception}",
            extra={
                'error_id': error_id,
                'severity': severity.value,
                'category': category.value,
                'context': context.additional_data
            }
        )
        
        # Update statistics
        self.error_stats[f"{severity.value}_{category.value}"] += 1
        self.component_errors[context.component][type(exception).__name__] += 1
        
        # Attempt recovery
        recovery_successful = False
        result = None
        
        try:
            if strategy == RecoveryStrategy.RETRY and recovery_func:
                result = await self._retry_with_backoff(recovery_func, context)
                recovery_successful = True
                error_record.recovery_attempted = True
                error_record.recovery_successful = True
                error_record.recovery_strategy = strategy
                
            elif strategy == RecoveryStrategy.FALLBACK and recovery_func:
                result = await self._execute_fallback(recovery_func, context, exception)
                recovery_successful = True
                error_record.recovery_attempted = True
                error_record.recovery_successful = True
                error_record.recovery_strategy = strategy
                
            elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
                # Circuit breaker will be handled at the caller level
                pass
                
            elif strategy == RecoveryStrategy.DEGRADE:
                # Return degraded result
                result = await self._degrade_gracefully(context, exception)
                recovery_successful = True
                error_record.recovery_attempted = True
                error_record.recovery_successful = True
                error_record.recovery_strategy = strategy
                
            elif strategy == RecoveryStrategy.IGNORE:
                recovery_successful = True
                error_record.recovery_attempted = True
                error_record.recovery_successful = True
                error_record.recovery_strategy = strategy
                
        except Exception as recovery_error:
            self.logger.error(
                f"Recovery failed for error {error_id}: {recovery_error}",
                extra={'original_error': str(exception), 'recovery_error': str(recovery_error)}
            )
            error_record.recovery_attempted = True
            error_record.recovery_successful = False
        
        # Store error record
        self.error_records.append(error_record)
        
        # Store in persistent storage
        if self.enable_persistent_storage and self.persistent_storage:
            try:
                await self.persistent_storage.store_error(error_record)
            except Exception as storage_error:
                self.logger.error(f"Failed to store error in persistent storage: {storage_error}")
        
        # Send to Sentry
        if self.enable_sentry and self.sentry_tracker:
            try:
                self.sentry_tracker.capture_error_record(error_record)
            except Exception as sentry_error:
                self.logger.error(f"Failed to send error to Sentry: {sentry_error}")
        
        return recovery_successful, result
    
    async def _retry_with_backoff(
        self,
        func: Callable,
        context: ErrorContext,
        max_attempts: int = 3
    ) -> Any:
        """Retry function with exponential backoff."""
        config = self.retry_configs.get(context.component, RetryConfig())
        
        last_exception = None
        for attempt in range(config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    return func()
                    
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if type(e) not in config.retryable_exceptions:
                    raise
                
                if attempt < config.max_attempts - 1:
                    # Calculate backoff delay
                    if config.backoff_strategy == "exponential":
                        delay = min(
                            config.base_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )
                    elif config.backoff_strategy == "linear":
                        delay = min(config.base_delay * (attempt + 1), config.max_delay)
                    else:  # constant
                        delay = config.base_delay
                    
                    # Add jitter if enabled
                    if config.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    self.logger.debug(
                        f"Retry attempt {attempt + 1}/{config.max_attempts} "
                        f"for {context.component}.{context.operation} after {delay:.2f}s"
                    )
                    
                    await asyncio.sleep(delay)
        
        # All retries failed
        if last_exception:
            raise last_exception
    
    async def _execute_fallback(
        self,
        fallback_func: Callable,
        context: ErrorContext,
        original_exception: Exception
    ) -> Any:
        """Execute fallback function."""
        self.logger.info(
            f"Executing fallback for {context.component}.{context.operation} "
            f"due to {type(original_exception).__name__}"
        )
        
        try:
            if asyncio.iscoroutinefunction(fallback_func):
                return await fallback_func()
            else:
                return fallback_func()
        except Exception as fallback_error:
            self.logger.error(
                f"Fallback also failed: {fallback_error}",
                extra={'original_error': str(original_exception)}
            )
            raise
    
    async def _degrade_gracefully(
        self,
        context: ErrorContext,
        exception: Exception
    ) -> Any:
        """Provide degraded functionality."""
        self.logger.warning(
            f"Degrading functionality for {context.component}.{context.operation} "
            f"due to {type(exception).__name__}"
        )
        
        # Return appropriate degraded result based on component
        if context.component == "protocol_classifier":
            return {"protocol_type": "unknown", "confidence": 0.0}
        elif context.component == "grammar_learner":
            return None  # Skip grammar learning
        elif context.component == "parser_generator":
            return None  # Skip parser generation
        else:
            return None
    
    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """Get logging level for error severity."""
        level_map = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return level_map[severity]
    
    def get_circuit_breaker(self, component: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for component."""
        if component not in self.circuit_breakers:
            cb_config = config or CircuitBreakerConfig()
            self.circuit_breakers[component] = CircuitBreaker(cb_config)
        
        return self.circuit_breakers[component]
    
    def set_retry_config(self, component: str, config: RetryConfig) -> None:
        """Set retry configuration for component."""
        self.retry_configs[component] = config
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            'total_errors': len(self.error_records),
            'error_by_severity': dict(self.error_stats),
            'error_by_component': {
                comp: dict(errors) for comp, errors in self.component_errors.items()
            },
            'circuit_breaker_states': {
                comp: cb.state.value for comp, cb in self.circuit_breakers.items()
            },
            'recent_errors': [
                record.to_dict() for record in list(self.error_records)[-10:]
            ]
        }
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status for a specific component."""
        recent_errors = [
            record for record in self.error_records 
            if record.component == component and 
            time.time() - record.timestamp < 3600  # Last hour
        ]
        
        error_rate = len(recent_errors) / 60.0  # Errors per minute
        
        # Determine health status
        if error_rate > 1.0:
            health_status = "unhealthy"
        elif error_rate > 0.5:
            health_status = "degraded"
        else:
            health_status = "healthy"
        
        return {
            'component': component,
            'health_status': health_status,
            'error_rate_per_minute': error_rate,
            'recent_error_count': len(recent_errors),
            'circuit_breaker_state': (
                self.circuit_breakers[component].state.value 
                if component in self.circuit_breakers else "not_configured"
            ),
            'dominant_error_types': [
                error_type for error_type, count in 
                Counter([r.exception_type for r in recent_errors]).most_common(5)
            ]
        }
    
    async def initialize_integrations(
        self,
        redis_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
        sentry_dsn: Optional[str] = None,
        environment: Optional[str] = None
    ):
        """Initialize external integrations for error tracking."""
        try:
            # Initialize persistent storage
            if self.enable_persistent_storage:
                from .error_storage import get_error_storage
                self.persistent_storage = await get_error_storage(redis_url, postgres_url)
                self.logger.info("Persistent error storage initialized")
            
            # Initialize Sentry
            if self.enable_sentry:
                from .sentry_integration import get_sentry_tracker
                self.sentry_tracker = get_sentry_tracker(sentry_dsn, environment)
                self.logger.info("Sentry error tracking initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize error tracking integrations: {e}")
    
    async def get_aggregated_errors(
        self,
        component: Optional[str] = None,
        time_window_hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get aggregated errors from persistent storage.
        
        Args:
            component: Filter by component name
            time_window_hours: Time window in hours
            limit: Maximum number of errors to return
            
        Returns:
            List of error records
        """
        if not self.persistent_storage:
            # Fall back to in-memory records
            cutoff_time = time.time() - (time_window_hours * 3600)
            errors = [
                record.to_dict() for record in self.error_records
                if record.timestamp >= cutoff_time
            ]
            if component:
                errors = [e for e in errors if e['component'] == component]
            return errors[:limit]
        
        try:
            if component:
                since = time.time() - (time_window_hours * 3600)
                return await self.persistent_storage.get_errors_by_component(
                    component, limit, since
                )
            else:
                # Get statistics and recent errors
                stats = await self.persistent_storage.get_error_statistics(time_window_hours)
                return stats
        except Exception as e:
            self.logger.error(f"Failed to get aggregated errors: {e}")
            return []
    
    async def cleanup_old_errors(self):
        """Clean up old error records from persistent storage."""
        if self.persistent_storage:
            try:
                deleted_count = await self.persistent_storage.cleanup_old_errors()
                self.logger.info(f"Cleaned up {deleted_count} old error records")
                return deleted_count
            except Exception as e:
                self.logger.error(f"Failed to cleanup old errors: {e}")
                return 0
        return 0


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(
    component: str,
    operation: str = None,
    recovery_func: Optional[Callable] = None,
    circuit_breaker: bool = False,
    **context_data
):
    """
    Decorator for comprehensive error handling.
    
    Args:
        component: Component name for error tracking
        operation: Operation name (defaults to function name)
        recovery_func: Function to call for recovery
        circuit_breaker: Whether to use circuit breaker
        **context_data: Additional context data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            context = ErrorContext(
                component=component,
                operation=op_name,
                request_id=context_data.get('request_id'),
                trace_id=context_data.get('trace_id'),
                additional_data=context_data
            )
            
            try:
                if circuit_breaker:
                    cb = error_handler.get_circuit_breaker(component)
                    return await cb.call(func, *args, **kwargs)
                else:
                    return await func(*args, **kwargs)
                    
            except Exception as e:
                recovery_successful, result = await error_handler.handle_error(
                    e, context, recovery_func
                )
                
                if recovery_successful:
                    return result
                else:
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            context = ErrorContext(
                component=component,
                operation=op_name,
                request_id=context_data.get('request_id'),
                trace_id=context_data.get('trace_id'),
                additional_data=context_data
            )
            
            try:
                if circuit_breaker:
                    # For sync functions, we can't use async circuit breaker
                    return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                # For sync functions, we need to handle errors synchronously
                error_handler.logger.error(
                    f"Error in {component}.{op_name}: {e}",
                    extra={'context': context_data}
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@asynccontextmanager
async def error_context(
    component: str,
    operation: str,
    **context_data
):
    """
    Context manager for error handling with automatic cleanup.
    
    Usage:
        async with error_context("component", "operation") as ctx:
            # Your code here
            pass
    """
    context = ErrorContext(
        component=component,
        operation=operation,
        request_id=context_data.get('request_id'),
        trace_id=context_data.get('trace_id'),
        additional_data=context_data
    )
    
    start_time = time.time()
    
    try:
        yield context
        
        # Log successful completion
        duration = time.time() - start_time
        logging.getLogger(component).debug(
            f"Operation {operation} completed successfully in {duration:.3f}s"
        )
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Handle error
        await error_handler.handle_error(e, context)
        
        logging.getLogger(component).error(
            f"Operation {operation} failed after {duration:.3f}s: {e}"
        )
        
        raise


class HealthMonitor:
    """Health monitoring system for components."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.logger = logging.getLogger(f"{__name__}.HealthMonitor")
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.health_checks: Dict[str, Callable] = {}
        
    def register_health_check(self, component: str, health_check_func: Callable) -> None:
        """Register a health check function for a component."""
        self.health_checks[component] = health_check_func
        self.logger.info(f"Registered health check for component: {component}")
    
    async def check_component_health(self, component: str) -> Dict[str, Any]:
        """Check health of a specific component."""
        health_info = self.error_handler.get_component_health(component)
        
        # Run custom health check if available
        if component in self.health_checks:
            try:
                custom_health = await self.health_checks[component]()
                health_info.update(custom_health)
            except Exception as e:
                health_info['custom_health_check_error'] = str(e)
        
        self.component_health[component] = health_info
        return health_info
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        component_healths = {}
        
        # Get list of components from error handler
        components = set()
        for record in self.error_handler.error_records:
            components.add(record.component)
        
        # Add registered health check components
        components.update(self.health_checks.keys())
        
        # Check each component
        for component in components:
            component_healths[component] = await self.check_component_health(component)
        
        # Determine overall health
        unhealthy_components = [
            comp for comp, health in component_healths.items()
            if health['health_status'] in ['unhealthy', 'degraded']
        ]
        
        if not unhealthy_components:
            overall_status = "healthy"
        elif len(unhealthy_components) <= len(component_healths) * 0.3:  # Less than 30% unhealthy
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            'overall_status': overall_status,
            'timestamp': time.time(),
            'components': component_healths,
            'unhealthy_components': unhealthy_components,
            'error_statistics': self.error_handler.get_error_statistics()
        }


# Global health monitor
health_monitor = HealthMonitor(error_handler)