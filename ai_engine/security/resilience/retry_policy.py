"""
CRONOS AI Engine - Retry Policy Implementation

Enterprise-grade retry mechanisms with exponential backoff, jitter,
and intelligent failure classification for the Zero-Touch Security Orchestrator.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, List, Dict, Type
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..logging import get_security_logger, SecurityLogType, LogLevel


class RetryResult(str, Enum):
    """Result of a retry attempt."""
    SUCCESS = "success"
    FAILURE = "failure"
    EXHAUSTED = "exhausted"
    ABORTED = "aborted"


class FailureClassification(str, Enum):
    """Classification of failures for retry decisions."""
    TRANSIENT = "transient"      # Temporary failure, should retry
    PERMANENT = "permanent"      # Permanent failure, don't retry
    RATE_LIMITED = "rate_limited" # Rate limited, retry with backoff
    TIMEOUT = "timeout"          # Timeout, may retry with longer timeout
    UNKNOWN = "unknown"          # Unknown failure type


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    delay: float
    exception: Optional[Exception] = None
    timestamp: datetime = None
    duration: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class RetryStats:
    """Statistics for retry operations."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_delay: float = 0.0
    average_delay: float = 0.0
    max_delay: float = 0.0
    success_rate: float = 0.0
    
    def update(self, attempt: RetryAttempt, success: bool):
        """Update statistics with a new attempt."""
        self.total_attempts += 1
        if success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1
        
        self.total_delay += attempt.delay
        self.average_delay = self.total_delay / self.total_attempts
        self.max_delay = max(self.max_delay, attempt.delay)
        self.success_rate = self.successful_attempts / self.total_attempts


class BackoffStrategy(ABC):
    """Abstract base class for backoff strategies."""
    
    @abstractmethod
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate delay for given attempt number."""
        pass


class LinearBackoff(BackoffStrategy):
    """Linear backoff strategy."""
    
    def __init__(self, multiplier: float = 1.0):
        self.multiplier = multiplier
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate linear backoff delay."""
        return base_delay * attempt * self.multiplier


class ExponentialBackoff(BackoffStrategy):
    """Exponential backoff strategy with jitter."""
    
    def __init__(
        self, 
        multiplier: float = 2.0,
        max_delay: float = 300.0,  # 5 minutes max
        jitter: bool = True,
        jitter_factor: float = 0.1
    ):
        self.multiplier = multiplier
        self.max_delay = max_delay
        self.jitter = jitter
        self.jitter_factor = jitter_factor
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        
        # Calculate exponential delay
        delay = base_delay * (self.multiplier ** (attempt - 1))
        
        # Apply max delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter and delay > 0:
            jitter_amount = delay * self.jitter_factor
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay + jitter)
        
        return delay


class FixedBackoff(BackoffStrategy):
    """Fixed delay backoff strategy."""
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Return fixed delay regardless of attempt number."""
        return base_delay


class RetryPolicy:
    """
    Configurable retry policy with intelligent failure classification
    and multiple backoff strategies.
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        backoff_strategy: Optional[BackoffStrategy] = None,
        max_total_time: Optional[float] = None,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        non_retryable_exceptions: Optional[List[Type[Exception]]] = None,
        failure_classifier: Optional[Callable[[Exception], FailureClassification]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff_strategy = backoff_strategy or ExponentialBackoff()
        self.max_total_time = max_total_time
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError,
            TimeoutError,
            OSError,
            asyncio.TimeoutError
        ]
        self.non_retryable_exceptions = non_retryable_exceptions or [
            ValueError,
            TypeError,
            AttributeError,
            KeyError
        ]
        self.failure_classifier = failure_classifier or self._default_failure_classifier
        
        self.logger = get_security_logger("cronos.security.resilience.retry_policy")
        self._stats = RetryStats()
    
    def _default_failure_classifier(self, exception: Exception) -> FailureClassification:
        """Default failure classification logic."""
        
        exception_type = type(exception)
        
        # Check explicit non-retryable exceptions
        if any(isinstance(exception, exc_type) for exc_type in self.non_retryable_exceptions):
            return FailureClassification.PERMANENT
        
        # Check explicit retryable exceptions
        if any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions):
            return FailureClassification.TRANSIENT
        
        # Classify based on exception type and message
        exception_str = str(exception).lower()
        
        if isinstance(exception, (ConnectionError, OSError)):
            return FailureClassification.TRANSIENT
        elif isinstance(exception, (TimeoutError, asyncio.TimeoutError)):
            return FailureClassification.TIMEOUT
        elif "rate limit" in exception_str or "throttl" in exception_str:
            return FailureClassification.RATE_LIMITED
        elif "not found" in exception_str or "unauthorized" in exception_str:
            return FailureClassification.PERMANENT
        else:
            return FailureClassification.UNKNOWN
    
    def should_retry(self, attempt: int, exception: Exception, total_time: float) -> bool:
        """Determine if we should retry based on the failure."""
        
        # Check attempt limit
        if attempt >= self.max_attempts:
            return False
        
        # Check total time limit
        if self.max_total_time and total_time >= self.max_total_time:
            return False
        
        # Classify the failure
        classification = self.failure_classifier(exception)
        
        # Don't retry permanent failures
        if classification == FailureClassification.PERMANENT:
            return False
        
        # Always retry transient and timeout failures (within limits)
        if classification in [FailureClassification.TRANSIENT, FailureClassification.TIMEOUT]:
            return True
        
        # For rate limited failures, always retry with backoff
        if classification == FailureClassification.RATE_LIMITED:
            return True
        
        # For unknown failures, be conservative and retry
        if classification == FailureClassification.UNKNOWN:
            return True
        
        return True
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries exhausted
        """
        
        attempts = []
        start_time = time.time()
        last_exception = None
        
        for attempt_number in range(1, self.max_attempts + 1):
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, func, *args, **kwargs)
                
                # Success - update stats and return
                attempt = RetryAttempt(
                    attempt_number=attempt_number,
                    delay=0.0 if attempt_number == 1 else attempts[-1].delay,
                    duration=time.time() - (start_time + sum(a.delay for a in attempts))
                )
                attempts.append(attempt)
                self._stats.update(attempt, success=True)
                
                if attempt_number > 1:
                    self.logger.log_security_event(
                        SecurityLogType.PERFORMANCE_METRIC,
                        f"Retry succeeded on attempt {attempt_number}",
                        level=LogLevel.INFO,
                        metadata={
                            'attempts': attempt_number,
                            'total_time': time.time() - start_time,
                            'function': func.__name__ if hasattr(func, '__name__') else str(func)
                        }
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                total_time = time.time() - start_time
                
                # Check if we should retry
                if not self.should_retry(attempt_number, e, total_time):
                    attempt = RetryAttempt(
                        attempt_number=attempt_number,
                        delay=0.0,
                        exception=e,
                        duration=time.time() - (start_time + sum(a.delay for a in attempts))
                    )
                    attempts.append(attempt)
                    self._stats.update(attempt, success=False)
                    
                    failure_classification = self.failure_classifier(e)
                    
                    self.logger.log_security_event(
                        SecurityLogType.PERFORMANCE_METRIC,
                        f"Retry exhausted after {attempt_number} attempts - {failure_classification.value} failure",
                        level=LogLevel.ERROR,
                        metadata={
                            'attempts': attempt_number,
                            'total_time': total_time,
                            'failure_classification': failure_classification.value,
                            'exception_type': type(e).__name__,
                            'exception_message': str(e),
                            'function': func.__name__ if hasattr(func, '__name__') else str(func)
                        },
                        error_code="RETRY_EXHAUSTED"
                    )
                    
                    break
                
                # Calculate delay for next attempt
                if attempt_number < self.max_attempts:
                    delay = self.backoff_strategy.calculate_delay(attempt_number, self.base_delay)
                    
                    # Adjust delay based on failure classification
                    failure_classification = self.failure_classifier(e)
                    if failure_classification == FailureClassification.RATE_LIMITED:
                        # Increase delay for rate limited failures
                        delay *= 2
                    elif failure_classification == FailureClassification.TIMEOUT:
                        # Smaller delay for timeouts
                        delay *= 0.5
                    
                    attempt = RetryAttempt(
                        attempt_number=attempt_number,
                        delay=delay,
                        exception=e,
                        duration=time.time() - (start_time + sum(a.delay for a in attempts))
                    )
                    attempts.append(attempt)
                    self._stats.update(attempt, success=False)
                    
                    self.logger.log_security_event(
                        SecurityLogType.PERFORMANCE_METRIC,
                        f"Retry attempt {attempt_number} failed, retrying in {delay:.2f}s",
                        level=LogLevel.WARNING,
                        metadata={
                            'attempt': attempt_number,
                            'delay': delay,
                            'failure_classification': failure_classification.value,
                            'exception_type': type(e).__name__,
                            'function': func.__name__ if hasattr(func, '__name__') else str(func)
                        }
                    )
                    
                    # Wait before retry
                    if delay > 0:
                        await asyncio.sleep(delay)
                else:
                    # Last attempt failed
                    attempt = RetryAttempt(
                        attempt_number=attempt_number,
                        delay=0.0,
                        exception=e,
                        duration=time.time() - (start_time + sum(a.delay for a in attempts))
                    )
                    attempts.append(attempt)
                    self._stats.update(attempt, success=False)
        
        # All retries exhausted, raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Retry logic failed unexpectedly")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return {
            'max_attempts': self.max_attempts,
            'base_delay': self.base_delay,
            'backoff_strategy': type(self.backoff_strategy).__name__,
            'stats': {
                'total_attempts': self._stats.total_attempts,
                'successful_attempts': self._stats.successful_attempts,
                'failed_attempts': self._stats.failed_attempts,
                'total_delay': self._stats.total_delay,
                'average_delay': self._stats.average_delay,
                'max_delay': self._stats.max_delay,
                'success_rate': self._stats.success_rate
            }
        }
    
    def reset_stats(self):
        """Reset retry statistics."""
        self._stats = RetryStats()


class RetryManager:
    """Manages multiple retry policies."""
    
    def __init__(self):
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.logger = get_security_logger("cronos.security.resilience.retry_manager")
    
    def create_policy(
        self,
        name: str,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        backoff_strategy: Optional[BackoffStrategy] = None,
        **kwargs
    ) -> RetryPolicy:
        """Create and register a new retry policy."""
        
        policy = RetryPolicy(
            max_attempts=max_attempts,
            base_delay=base_delay,
            backoff_strategy=backoff_strategy,
            **kwargs
        )
        
        self.retry_policies[name] = policy
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Created retry policy: {name}",
            level=LogLevel.INFO,
            metadata={
                'max_attempts': max_attempts,
                'base_delay': base_delay,
                'backoff_strategy': type(backoff_strategy).__name__ if backoff_strategy else 'ExponentialBackoff'
            }
        )
        
        return policy
    
    def get_policy(self, name: str) -> Optional[RetryPolicy]:
        """Get retry policy by name."""
        return self.retry_policies.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all retry policies."""
        return {
            name: policy.get_stats()
            for name, policy in self.retry_policies.items()
        }
    
    def reset_all_stats(self):
        """Reset statistics for all retry policies."""
        for policy in self.retry_policies.values():
            policy.reset_stats()
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Reset stats for {len(self.retry_policies)} retry policies",
            level=LogLevel.INFO
        )


# Pre-configured retry policies for common scenarios
class CommonRetryPolicies:
    """Pre-configured retry policies for common use cases."""
    
    @staticmethod
    def create_fast_retry() -> RetryPolicy:
        """Fast retry for quick operations."""
        return RetryPolicy(
            max_attempts=3,
            base_delay=0.1,
            backoff_strategy=ExponentialBackoff(multiplier=2.0, max_delay=5.0)
        )
    
    @staticmethod
    def create_standard_retry() -> RetryPolicy:
        """Standard retry for most operations."""
        return RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            backoff_strategy=ExponentialBackoff(multiplier=2.0, max_delay=30.0)
        )
    
    @staticmethod
    def create_slow_retry() -> RetryPolicy:
        """Slow retry for expensive operations."""
        return RetryPolicy(
            max_attempts=5,
            base_delay=5.0,
            backoff_strategy=ExponentialBackoff(multiplier=2.0, max_delay=300.0),
            max_total_time=600.0  # 10 minutes max
        )
    
    @staticmethod
    def create_api_retry() -> RetryPolicy:
        """Retry policy optimized for API calls."""
        return RetryPolicy(
            max_attempts=4,
            base_delay=1.0,
            backoff_strategy=ExponentialBackoff(
                multiplier=2.0, 
                max_delay=60.0,
                jitter=True,
                jitter_factor=0.1
            ),
            retryable_exceptions=[
                ConnectionError,
                TimeoutError,
                OSError,
                asyncio.TimeoutError
            ]
        )
    
    @staticmethod
    def create_database_retry() -> RetryPolicy:
        """Retry policy optimized for database operations."""
        return RetryPolicy(
            max_attempts=3,
            base_delay=2.0,
            backoff_strategy=LinearBackoff(multiplier=1.5),
            max_total_time=30.0
        )


# Global retry manager instance
_retry_manager: Optional[RetryManager] = None

def get_retry_manager() -> RetryManager:
    """Get global retry manager instance."""
    global _retry_manager
    if _retry_manager is None:
        _retry_manager = RetryManager()
    return _retry_manager