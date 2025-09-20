"""
CRONOS AI Engine - Circuit Breaker Implementation

Enterprise-grade circuit breaker pattern to prevent cascading failures
and provide graceful degradation for the Zero-Touch Security Orchestrator.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Optional, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from ..logging import get_security_logger, SecurityLogType, LogLevel


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # for half-open state
    timeout: float = 30.0  # request timeout
    expected_exception: type = Exception
    
    # Advanced settings
    sliding_window_size: int = 100
    minimum_requests: int = 10
    failure_rate_threshold: float = 0.5  # 50%
    slow_call_duration_threshold: float = 5.0  # seconds
    slow_call_rate_threshold: float = 0.5  # 50%


@dataclass
class CallResult:
    """Result of a circuit breaker call."""
    success: bool
    duration: float
    exception: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CircuitBreakerException(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, message: str, state: CircuitBreakerState):
        super().__init__(message)
        self.state = state


class CircuitBreaker:
    """
    Implementation of the Circuit Breaker pattern.
    
    Prevents cascading failures by stopping calls to failing services
    and providing graceful degradation mechanisms.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = get_security_logger(f"cronos.security.resilience.circuit_breaker.{name}")
        
        # State management
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._next_attempt_time: Optional[datetime] = None
        
        # Call history for sliding window
        self._call_history: List[CallResult] = []
        self._lock = asyncio.Lock()
        
        # Metrics
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._rejected_calls = 0
        self._slow_calls = 0
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Circuit breaker '{name}' initialized",
            level=LogLevel.INFO,
            metadata={
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold
            }
        )
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count
    
    @property
    def success_count(self) -> int:
        """Get current success count."""
        return self._success_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            'name': self.name,
            'state': self._state.value,
            'total_calls': self._total_calls,
            'successful_calls': self._successful_calls,
            'failed_calls': self._failed_calls,
            'rejected_calls': self._rejected_calls,
            'slow_calls': self._slow_calls,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
            'failure_rate': self._calculate_failure_rate(),
            'slow_call_rate': self._calculate_slow_call_rate(),
            'last_failure_time': self._last_failure_time.isoformat() if self._last_failure_time else None,
            'next_attempt_time': self._next_attempt_time.isoformat() if self._next_attempt_time else None
        }
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function call through the circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerException: When circuit is open
            Original exception: When function fails
        """
        
        async with self._lock:
            self._total_calls += 1
            
            # Check if we should reject the call
            if await self._should_reject_call():
                self._rejected_calls += 1
                
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Circuit breaker '{self.name}' rejected call - state: {self._state.value}",
                    level=LogLevel.WARNING,
                    metadata={
                        'state': self._state.value,
                        'failure_count': self._failure_count,
                        'rejected_calls': self._rejected_calls
                    }
                )
                
                raise CircuitBreakerException(
                    f"Circuit breaker '{self.name}' is {self._state.value}",
                    self._state
                )
        
        # Execute the function call
        start_time = time.time()
        call_result = None
        
        try:
            # Apply timeout if configured
            if self.config.timeout > 0:
                result = await asyncio.wait_for(
                    self._execute_async_call(func, *args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = await self._execute_async_call(func, *args, **kwargs)
            
            duration = time.time() - start_time
            call_result = CallResult(success=True, duration=duration)
            
            await self._record_success(call_result)
            return result
            
        except asyncio.TimeoutError as e:
            duration = time.time() - start_time
            call_result = CallResult(success=False, duration=duration, exception=e)
            await self._record_failure(call_result)
            raise
            
        except Exception as e:
            duration = time.time() - start_time
            call_result = CallResult(success=False, duration=duration, exception=e)
            
            if isinstance(e, self.config.expected_exception):
                await self._record_failure(call_result)
            else:
                # Unexpected exception, treat as failure but don't count towards threshold
                async with self._lock:
                    self._failed_calls += 1
                    self._call_history.append(call_result)
                    self._cleanup_old_calls()
            
            raise
    
    async def _execute_async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function call, handling both sync and async functions."""
        
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Execute sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def _should_reject_call(self) -> bool:
        """Determine if call should be rejected based on current state."""
        
        current_time = datetime.utcnow()
        
        if self._state == CircuitBreakerState.CLOSED:
            return False
        elif self._state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (self._next_attempt_time and 
                current_time >= self._next_attempt_time):
                self._state = CircuitBreakerState.HALF_OPEN
                self._success_count = 0
                
                self.logger.log_security_event(
                    SecurityLogType.CONFIGURATION_CHANGE,
                    f"Circuit breaker '{self.name}' transitioning to HALF_OPEN",
                    level=LogLevel.INFO
                )
                
                return False
            else:
                return True
        elif self._state == CircuitBreakerState.HALF_OPEN:
            return False
        
        return False
    
    async def _record_success(self, call_result: CallResult):
        """Record a successful call."""
        
        async with self._lock:
            self._successful_calls += 1
            self._call_history.append(call_result)
            self._cleanup_old_calls()
            
            # Check if call was slow
            if call_result.duration > self.config.slow_call_duration_threshold:
                self._slow_calls += 1
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._last_failure_time = None
                    self._next_attempt_time = None
                    
                    self.logger.log_security_event(
                        SecurityLogType.CONFIGURATION_CHANGE,
                        f"Circuit breaker '{self.name}' recovered - transitioning to CLOSED",
                        level=LogLevel.INFO,
                        metadata={'success_count': self._success_count}
                    )
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    self._failure_count = max(0, self._failure_count - 1)
    
    async def _record_failure(self, call_result: CallResult):
        """Record a failed call."""
        
        async with self._lock:
            self._failed_calls += 1
            self._call_history.append(call_result)
            self._cleanup_old_calls()
            
            # Check if call was slow
            if call_result.duration > self.config.slow_call_duration_threshold:
                self._slow_calls += 1
            
            if self._state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]:
                self._failure_count += 1
                self._last_failure_time = datetime.utcnow()
                
                # Check if we should open the circuit
                should_open = False
                
                if self._state == CircuitBreakerState.HALF_OPEN:
                    # Any failure in half-open state opens the circuit
                    should_open = True
                elif self._state == CircuitBreakerState.CLOSED:
                    # Check thresholds for closed state
                    if len(self._call_history) >= self.config.minimum_requests:
                        failure_rate = self._calculate_failure_rate()
                        slow_call_rate = self._calculate_slow_call_rate()
                        
                        should_open = (
                            self._failure_count >= self.config.failure_threshold or
                            failure_rate >= self.config.failure_rate_threshold or
                            slow_call_rate >= self.config.slow_call_rate_threshold
                        )
                    else:
                        # Use simple failure count threshold when not enough data
                        should_open = self._failure_count >= self.config.failure_threshold
                
                if should_open:
                    self._state = CircuitBreakerState.OPEN
                    self._next_attempt_time = (
                        datetime.utcnow() + 
                        timedelta(seconds=self.config.recovery_timeout)
                    )
                    
                    self.logger.log_security_event(
                        SecurityLogType.CONFIGURATION_CHANGE,
                        f"Circuit breaker '{self.name}' opened due to failures",
                        level=LogLevel.ERROR,
                        metadata={
                            'failure_count': self._failure_count,
                            'failure_rate': self._calculate_failure_rate(),
                            'slow_call_rate': self._calculate_slow_call_rate(),
                            'next_attempt_time': self._next_attempt_time.isoformat()
                        },
                        error_code="CIRCUIT_BREAKER_OPENED"
                    )
    
    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate from sliding window."""
        
        if not self._call_history:
            return 0.0
        
        failed_calls = sum(1 for call in self._call_history if not call.success)
        return failed_calls / len(self._call_history)
    
    def _calculate_slow_call_rate(self) -> float:
        """Calculate slow call rate from sliding window."""
        
        if not self._call_history:
            return 0.0
        
        slow_calls = sum(
            1 for call in self._call_history 
            if call.duration > self.config.slow_call_duration_threshold
        )
        return slow_calls / len(self._call_history)
    
    def _cleanup_old_calls(self):
        """Remove old calls from history to maintain sliding window size."""
        
        if len(self._call_history) > self.config.sliding_window_size:
            # Remove oldest calls
            excess = len(self._call_history) - self.config.sliding_window_size
            self._call_history = self._call_history[excess:]
    
    async def reset(self):
        """Reset circuit breaker to closed state."""
        
        async with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._next_attempt_time = None
            
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Circuit breaker '{self.name}' manually reset",
                level=LogLevel.INFO
            )
    
    async def force_open(self):
        """Force circuit breaker to open state."""
        
        async with self._lock:
            self._state = CircuitBreakerState.OPEN
            self._next_attempt_time = (
                datetime.utcnow() + 
                timedelta(seconds=self.config.recovery_timeout)
            )
            
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Circuit breaker '{self.name}' manually opened",
                level=LogLevel.WARNING
            )
    
    def __str__(self) -> str:
        return f"CircuitBreaker(name={self.name}, state={self._state.value})"
    
    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name}, state={self._state.value}, "
            f"failures={self._failure_count}, successes={self._success_count})"
        )


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = get_security_logger("cronos.security.resilience.circuit_breaker_manager")
    
    def get_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
            
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Created circuit breaker: {name}",
                level=LogLevel.INFO
            )
        
        return self.circuit_breakers[name]
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        
        return {
            name: cb.get_metrics()
            for name, cb in self.circuit_breakers.items()
        }
    
    async def reset_all(self):
        """Reset all circuit breakers."""
        
        for cb in self.circuit_breakers.values():
            await cb.reset()
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Reset {len(self.circuit_breakers)} circuit breakers",
            level=LogLevel.INFO
        )
    
    def list_circuit_breakers(self) -> List[str]:
        """List all circuit breaker names."""
        return list(self.circuit_breakers.keys())


# Global circuit breaker manager instance
_circuit_breaker_manager: Optional[CircuitBreakerManager] = None

def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager instance."""
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()
    return _circuit_breaker_manager