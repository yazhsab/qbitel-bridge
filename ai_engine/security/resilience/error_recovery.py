"""
CRONOS AI Engine - Error Recovery Manager Implementation

Enterprise-grade error recovery with automated recovery strategies,
fallback mechanisms, and graceful degradation for the Zero-Touch Security Orchestrator.
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

from ..logging import get_security_logger, SecurityLogType, LogLevel


class RecoveryStrategy(str, Enum):
    """Types of recovery strategies."""
    RETRY = "retry"                    # Simple retry
    FALLBACK = "fallback"             # Use fallback operation
    CIRCUIT_BREAKER = "circuit_breaker" # Circuit breaker pattern
    DEGRADED_MODE = "degraded_mode"    # Operate in degraded mode
    FAIL_FAST = "fail_fast"           # Fail immediately
    COMPENSATE = "compensate"         # Execute compensating action
    QUEUE_FOR_LATER = "queue_for_later" # Queue for later processing


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryResult(str, Enum):
    """Result of recovery attempt."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    DEGRADED = "degraded"
    DEFERRED = "deferred"


@dataclass
class ErrorContext:
    """Context information about an error."""
    error: Exception
    operation_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'error_type': type(self.error).__name__,
            'error_message': str(self.error),
            'operation_name': self.operation_name,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'metadata': self.metadata,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


@dataclass
class RecoveryAction:
    """Definition of a recovery action."""
    strategy: RecoveryStrategy
    handler: Optional[Callable] = None
    fallback_operation: Optional[Callable] = None
    compensating_action: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class RecoveryPlan:
    """Recovery plan with multiple strategies."""
    name: str
    actions: List[RecoveryAction] = field(default_factory=list)
    max_recovery_attempts: int = 3
    recovery_timeout: float = 300.0  # 5 minutes
    escalation_threshold: int = 5
    enabled: bool = True


@dataclass
class RecoveryExecutionResult:
    """Result of executing a recovery plan."""
    success: bool
    result: RecoveryResult
    message: str
    strategy_used: Optional[RecoveryStrategy] = None
    execution_time: float = 0.0
    recovered_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecoveryHandler(ABC):
    """Abstract base class for recovery handlers."""
    
    @abstractmethod
    async def handle_recovery(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> RecoveryExecutionResult:
        """Handle recovery for an error."""
        pass


class RetryRecoveryHandler(RecoveryHandler):
    """Handles retry-based recovery."""
    
    async def handle_recovery(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> RecoveryExecutionResult:
        """Execute retry recovery."""
        
        if error_context.retry_count >= error_context.max_retries:
            return RecoveryExecutionResult(
                success=False,
                result=RecoveryResult.FAILURE,
                message="Maximum retries exceeded",
                strategy_used=RecoveryStrategy.RETRY
            )
        
        try:
            # Execute the original operation again
            if recovery_action.handler:
                result = await recovery_action.handler()
                
                return RecoveryExecutionResult(
                    success=True,
                    result=RecoveryResult.SUCCESS,
                    message="Retry successful",
                    strategy_used=RecoveryStrategy.RETRY,
                    recovered_data=result
                )
            else:
                return RecoveryExecutionResult(
                    success=False,
                    result=RecoveryResult.FAILURE,
                    message="No retry handler provided",
                    strategy_used=RecoveryStrategy.RETRY
                )
                
        except Exception as e:
            return RecoveryExecutionResult(
                success=False,
                result=RecoveryResult.FAILURE,
                message=f"Retry failed: {str(e)}",
                strategy_used=RecoveryStrategy.RETRY,
                metadata={'retry_error': str(e)}
            )


class FallbackRecoveryHandler(RecoveryHandler):
    """Handles fallback-based recovery."""
    
    async def handle_recovery(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> RecoveryExecutionResult:
        """Execute fallback recovery."""
        
        if not recovery_action.fallback_operation:
            return RecoveryExecutionResult(
                success=False,
                result=RecoveryResult.FAILURE,
                message="No fallback operation provided",
                strategy_used=RecoveryStrategy.FALLBACK
            )
        
        try:
            # Execute fallback operation
            if asyncio.iscoroutinefunction(recovery_action.fallback_operation):
                result = await recovery_action.fallback_operation()
            else:
                result = recovery_action.fallback_operation()
            
            return RecoveryExecutionResult(
                success=True,
                result=RecoveryResult.DEGRADED,
                message="Fallback operation successful",
                strategy_used=RecoveryStrategy.FALLBACK,
                recovered_data=result
            )
            
        except Exception as e:
            return RecoveryExecutionResult(
                success=False,
                result=RecoveryResult.FAILURE,
                message=f"Fallback failed: {str(e)}",
                strategy_used=RecoveryStrategy.FALLBACK,
                metadata={'fallback_error': str(e)}
            )


class CompensatingRecoveryHandler(RecoveryHandler):
    """Handles compensating action recovery."""
    
    async def handle_recovery(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> RecoveryExecutionResult:
        """Execute compensating action recovery."""
        
        if not recovery_action.compensating_action:
            return RecoveryExecutionResult(
                success=False,
                result=RecoveryResult.FAILURE,
                message="No compensating action provided",
                strategy_used=RecoveryStrategy.COMPENSATE
            )
        
        try:
            # Execute compensating action
            if asyncio.iscoroutinefunction(recovery_action.compensating_action):
                result = await recovery_action.compensating_action()
            else:
                result = recovery_action.compensating_action()
            
            return RecoveryExecutionResult(
                success=True,
                result=RecoveryResult.SUCCESS,
                message="Compensating action successful",
                strategy_used=RecoveryStrategy.COMPENSATE,
                recovered_data=result
            )
            
        except Exception as e:
            return RecoveryExecutionResult(
                success=False,
                result=RecoveryResult.FAILURE,
                message=f"Compensating action failed: {str(e)}",
                strategy_used=RecoveryStrategy.COMPENSATE,
                metadata={'compensation_error': str(e)}
            )


class DegradedModeHandler(RecoveryHandler):
    """Handles degraded mode recovery."""
    
    async def handle_recovery(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> RecoveryExecutionResult:
        """Execute degraded mode recovery."""
        
        # Degraded mode typically means reduced functionality
        degraded_result = recovery_action.metadata.get('degraded_result')
        
        return RecoveryExecutionResult(
            success=True,
            result=RecoveryResult.DEGRADED,
            message="Operating in degraded mode",
            strategy_used=RecoveryStrategy.DEGRADED_MODE,
            recovered_data=degraded_result,
            metadata={'degraded_mode': True}
        )


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and execution for the 
    Zero-Touch Security Orchestrator.
    """
    
    def __init__(self):
        self.logger = get_security_logger("cronos.security.resilience.error_recovery")
        
        # Recovery plans
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        
        # Recovery handlers
        self.recovery_handlers: Dict[RecoveryStrategy, RecoveryHandler] = {
            RecoveryStrategy.RETRY: RetryRecoveryHandler(),
            RecoveryStrategy.FALLBACK: FallbackRecoveryHandler(),
            RecoveryStrategy.COMPENSATE: CompensatingRecoveryHandler(),
            RecoveryStrategy.DEGRADED_MODE: DegradedModeHandler()
        }
        
        # Error patterns and their associated recovery plans
        self.error_patterns: Dict[str, str] = {}  # error pattern -> plan name
        
        # Metrics
        self._total_recovery_attempts = 0
        self._successful_recoveries = 0
        self._failed_recoveries = 0
        self._recovery_history: List[Dict[str, Any]] = []
        
        # Create default recovery plans
        self._create_default_plans()
    
    def _create_default_plans(self):
        """Create default recovery plans."""
        
        # Network error recovery plan
        network_plan = RecoveryPlan(
            name="network_errors",
            actions=[
                RecoveryAction(strategy=RecoveryStrategy.RETRY),
                RecoveryAction(strategy=RecoveryStrategy.FALLBACK),
                RecoveryAction(strategy=RecoveryStrategy.DEGRADED_MODE)
            ]
        )
        self.register_recovery_plan(network_plan)
        
        # Database error recovery plan
        database_plan = RecoveryPlan(
            name="database_errors", 
            actions=[
                RecoveryAction(strategy=RecoveryStrategy.RETRY),
                RecoveryAction(strategy=RecoveryStrategy.CIRCUIT_BREAKER),
                RecoveryAction(strategy=RecoveryStrategy.QUEUE_FOR_LATER)
            ]
        )
        self.register_recovery_plan(database_plan)
        
        # Authentication error recovery plan
        auth_plan = RecoveryPlan(
            name="authentication_errors",
            actions=[
                RecoveryAction(strategy=RecoveryStrategy.RETRY),
                RecoveryAction(strategy=RecoveryStrategy.FAIL_FAST)
            ]
        )
        self.register_recovery_plan(auth_plan)
        
        # Critical system error recovery plan
        critical_plan = RecoveryPlan(
            name="critical_errors",
            actions=[
                RecoveryAction(strategy=RecoveryStrategy.COMPENSATE),
                RecoveryAction(strategy=RecoveryStrategy.DEGRADED_MODE),
                RecoveryAction(strategy=RecoveryStrategy.FAIL_FAST)
            ]
        )
        self.register_recovery_plan(critical_plan)
        
        # Register error patterns
        self.register_error_pattern("ConnectionError", "network_errors")
        self.register_error_pattern("TimeoutError", "network_errors")
        self.register_error_pattern("DatabaseError", "database_errors")
        self.register_error_pattern("AuthenticationError", "authentication_errors")
        self.register_error_pattern("SecurityError", "critical_errors")
    
    def register_recovery_plan(self, plan: RecoveryPlan):
        """Register a recovery plan."""
        
        self.recovery_plans[plan.name] = plan
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Registered recovery plan: {plan.name}",
            level=LogLevel.INFO,
            metadata={
                'plan_name': plan.name,
                'actions_count': len(plan.actions),
                'max_attempts': plan.max_recovery_attempts
            }
        )
    
    def register_error_pattern(self, error_pattern: str, plan_name: str):
        """Register an error pattern to recovery plan mapping."""
        
        if plan_name not in self.recovery_plans:
            raise ValueError(f"Recovery plan '{plan_name}' not found")
        
        self.error_patterns[error_pattern] = plan_name
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Registered error pattern: {error_pattern} -> {plan_name}",
            level=LogLevel.INFO
        )
    
    def register_recovery_handler(self, strategy: RecoveryStrategy, handler: RecoveryHandler):
        """Register a custom recovery handler."""
        
        self.recovery_handlers[strategy] = handler
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Registered recovery handler for strategy: {strategy.value}",
            level=LogLevel.INFO
        )
    
    def _identify_recovery_plan(self, error_context: ErrorContext) -> Optional[RecoveryPlan]:
        """Identify the appropriate recovery plan for an error."""
        
        error_type = type(error_context.error).__name__
        error_message = str(error_context.error).lower()
        
        # Check exact type match first
        if error_type in self.error_patterns:
            plan_name = self.error_patterns[error_type]
            return self.recovery_plans.get(plan_name)
        
        # Check for partial matches in error message
        for pattern, plan_name in self.error_patterns.items():
            if pattern.lower() in error_message:
                return self.recovery_plans.get(plan_name)
        
        # Check severity-based fallback
        if error_context.severity == ErrorSeverity.CRITICAL:
            return self.recovery_plans.get("critical_errors")
        elif error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.MEDIUM]:
            return self.recovery_plans.get("network_errors")  # Default fallback
        
        return None
    
    async def recover_from_error(
        self,
        error: Exception,
        operation_name: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> RecoveryExecutionResult:
        """
        Attempt to recover from an error using appropriate recovery strategies.
        
        Args:
            error: The exception that occurred
            operation_name: Name of the operation that failed
            severity: Severity of the error
            metadata: Additional context metadata
            max_retries: Maximum number of retries allowed
            
        Returns:
            RecoveryExecutionResult with recovery outcome
        """
        
        start_time = asyncio.get_event_loop().time()
        self._total_recovery_attempts += 1
        
        # Create error context
        error_context = ErrorContext(
            error=error,
            operation_name=operation_name,
            severity=severity,
            metadata=metadata or {},
            max_retries=max_retries
        )
        
        self.logger.log_security_event(
            SecurityLogType.PERFORMANCE_METRIC,
            f"Starting error recovery for operation: {operation_name}",
            level=LogLevel.WARNING,
            metadata=error_context.to_dict()
        )
        
        # Identify recovery plan
        recovery_plan = self._identify_recovery_plan(error_context)
        
        if not recovery_plan or not recovery_plan.enabled:
            result = RecoveryExecutionResult(
                success=False,
                result=RecoveryResult.FAILURE,
                message="No recovery plan available",
                execution_time=asyncio.get_event_loop().time() - start_time
            )
            
            self._record_recovery_result(error_context, result)
            return result
        
        # Execute recovery actions in order
        for attempt in range(recovery_plan.max_recovery_attempts):
            for action in recovery_plan.actions:
                if not action.enabled:
                    continue
                
                handler = self.recovery_handlers.get(action.strategy)
                if not handler:
                    self.logger.log_security_event(
                        SecurityLogType.PERFORMANCE_METRIC,
                        f"No handler for recovery strategy: {action.strategy.value}",
                        level=LogLevel.WARNING
                    )
                    continue
                
                try:
                    # Execute recovery action
                    recovery_result = await asyncio.wait_for(
                        handler.handle_recovery(error_context, action),
                        timeout=recovery_plan.recovery_timeout
                    )
                    
                    recovery_result.execution_time = asyncio.get_event_loop().time() - start_time
                    
                    if recovery_result.success:
                        self._successful_recoveries += 1
                        
                        self.logger.log_security_event(
                            SecurityLogType.PERFORMANCE_METRIC,
                            f"Recovery successful using {recovery_result.strategy_used.value} strategy",
                            level=LogLevel.INFO,
                            metadata={
                                'operation': operation_name,
                                'strategy': recovery_result.strategy_used.value,
                                'attempt': attempt + 1,
                                'execution_time': recovery_result.execution_time
                            }
                        )
                        
                        self._record_recovery_result(error_context, recovery_result)
                        return recovery_result
                
                except asyncio.TimeoutError:
                    self.logger.log_security_event(
                        SecurityLogType.PERFORMANCE_METRIC,
                        f"Recovery action timeout: {action.strategy.value}",
                        level=LogLevel.WARNING,
                        metadata={
                            'operation': operation_name,
                            'strategy': action.strategy.value,
                            'timeout': recovery_plan.recovery_timeout
                        }
                    )
                    continue
                
                except Exception as recovery_error:
                    self.logger.log_security_event(
                        SecurityLogType.PERFORMANCE_METRIC,
                        f"Recovery action failed: {action.strategy.value} - {str(recovery_error)}",
                        level=LogLevel.ERROR,
                        metadata={
                            'operation': operation_name,
                            'strategy': action.strategy.value,
                            'recovery_error': str(recovery_error)
                        }
                    )
                    continue
        
        # All recovery attempts failed
        self._failed_recoveries += 1
        
        result = RecoveryExecutionResult(
            success=False,
            result=RecoveryResult.FAILURE,
            message="All recovery strategies failed",
            execution_time=asyncio.get_event_loop().time() - start_time
        )
        
        self.logger.log_security_event(
            SecurityLogType.PERFORMANCE_METRIC,
            f"Error recovery failed for operation: {operation_name}",
            level=LogLevel.ERROR,
            metadata={
                'operation': operation_name,
                'total_attempts': recovery_plan.max_recovery_attempts,
                'execution_time': result.execution_time
            },
            error_code="ERROR_RECOVERY_FAILED"
        )
        
        self._record_recovery_result(error_context, result)
        return result
    
    def _record_recovery_result(
        self,
        error_context: ErrorContext,
        recovery_result: RecoveryExecutionResult
    ):
        """Record recovery result for metrics and analysis."""
        
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'error_context': error_context.to_dict(),
            'recovery_result': {
                'success': recovery_result.success,
                'result': recovery_result.result.value,
                'message': recovery_result.message,
                'strategy_used': recovery_result.strategy_used.value if recovery_result.strategy_used else None,
                'execution_time': recovery_result.execution_time
            }
        }
        
        self._recovery_history.append(record)
        
        # Keep only recent history
        if len(self._recovery_history) > 1000:
            self._recovery_history = self._recovery_history[-500:]
    
    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get recovery metrics and statistics."""
        
        success_rate = (
            self._successful_recoveries / self._total_recovery_attempts
            if self._total_recovery_attempts > 0 else 0
        )
        
        # Analyze recovery patterns
        strategy_usage = {}
        for record in self._recovery_history[-100:]:  # Last 100 records
            strategy = record['recovery_result']['strategy_used']
            if strategy:
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        return {
            'total_recovery_attempts': self._total_recovery_attempts,
            'successful_recoveries': self._successful_recoveries,
            'failed_recoveries': self._failed_recoveries,
            'success_rate': success_rate,
            'registered_plans': len(self.recovery_plans),
            'error_patterns': len(self.error_patterns),
            'strategy_usage': strategy_usage,
            'recovery_plans': {
                name: {
                    'actions_count': len(plan.actions),
                    'max_attempts': plan.max_recovery_attempts,
                    'enabled': plan.enabled
                }
                for name, plan in self.recovery_plans.items()
            }
        }
    
    def get_recovery_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent recovery history."""
        return self._recovery_history[-limit:]
    
    def reset_metrics(self):
        """Reset recovery metrics."""
        self._total_recovery_attempts = 0
        self._successful_recoveries = 0
        self._failed_recoveries = 0
        self._recovery_history.clear()
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Reset error recovery metrics",
            level=LogLevel.INFO
        )


# Global error recovery manager instance
_error_recovery_manager: Optional[ErrorRecoveryManager] = None

def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager instance."""
    global _error_recovery_manager
    if _error_recovery_manager is None:
        _error_recovery_manager = ErrorRecoveryManager()
    return _error_recovery_manager