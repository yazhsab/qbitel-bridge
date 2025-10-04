"""
CRONOS AI Engine - Timeout Manager Implementation

Enterprise-grade timeout management with configurable policies,
adaptive timeouts, and graceful handling for the Zero-Touch Security Orchestrator.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager

from ..logging import get_security_logger, SecurityLogType, LogLevel


class TimeoutStrategy(str, Enum):
    """Timeout strategy types."""

    FIXED = "fixed"  # Fixed timeout value
    ADAPTIVE = "adaptive"  # Adaptive based on historical performance
    PERCENTILE = "percentile"  # Based on response time percentiles
    CASCADING = "cascading"  # Cascading timeouts with fallbacks


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior."""

    name: str
    default_timeout: float = 30.0
    strategy: TimeoutStrategy = TimeoutStrategy.FIXED
    min_timeout: float = 1.0
    max_timeout: float = 300.0
    percentile: float = 95.0  # For percentile-based timeouts
    adaptation_factor: float = 1.5  # For adaptive timeouts
    history_size: int = 100
    enabled: bool = True


@dataclass
class TimeoutResult:
    """Result of a timeout-managed operation."""

    success: bool
    duration: float
    timeout_used: float
    timeout_occurred: bool = False
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TimeoutHistory:
    """Tracks timeout history for adaptive timeout calculations."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.durations: List[float] = []
        self.timeouts: List[float] = []
        self.successes: List[bool] = []
        self.timestamps: List[datetime] = []

    def add_result(self, duration: float, timeout: float, success: bool):
        """Add a result to the history."""

        self.durations.append(duration)
        self.timeouts.append(timeout)
        self.successes.append(success)
        self.timestamps.append(datetime.utcnow())

        # Maintain max size
        if len(self.durations) > self.max_size:
            self.durations.pop(0)
            self.timeouts.pop(0)
            self.successes.pop(0)
            self.timestamps.pop(0)

    def get_percentile_duration(self, percentile: float = 95.0) -> Optional[float]:
        """Get percentile duration from history."""

        if not self.durations:
            return None

        sorted_durations = sorted(self.durations)
        index = int((percentile / 100.0) * len(sorted_durations))
        index = min(index, len(sorted_durations) - 1)
        return sorted_durations[index]

    def get_average_duration(self) -> Optional[float]:
        """Get average duration from history."""

        if not self.durations:
            return None

        return sum(self.durations) / len(self.durations)

    def get_success_rate(self) -> float:
        """Get success rate from history."""

        if not self.successes:
            return 0.0

        return sum(self.successes) / len(self.successes)

    def get_timeout_rate(self) -> float:
        """Get timeout rate from history."""

        if not self.durations:
            return 0.0

        timeout_count = sum(1 for d, t in zip(self.durations, self.timeouts) if d >= t)

        return timeout_count / len(self.durations)


class TimeoutPolicy:
    """
    Manages timeout behavior for operations with configurable strategies
    and adaptive timeout calculation.
    """

    def __init__(self, config: TimeoutConfig):
        self.config = config
        self.logger = get_security_logger(
            f"cronos.security.resilience.timeout.{config.name}"
        )

        # History tracking
        self.history = TimeoutHistory(config.history_size)

        # Current timeout value (for adaptive strategies)
        self._current_timeout = config.default_timeout

        # Statistics
        self._total_operations = 0
        self._successful_operations = 0
        self._timeout_operations = 0
        self._failed_operations = 0

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Timeout policy '{config.name}' initialized",
            level=LogLevel.INFO,
            metadata={
                "strategy": config.strategy.value,
                "default_timeout": config.default_timeout,
                "min_timeout": config.min_timeout,
                "max_timeout": config.max_timeout,
            },
        )

    def calculate_timeout(self, override_timeout: Optional[float] = None) -> float:
        """Calculate timeout value based on strategy and history."""

        if override_timeout is not None:
            return max(
                self.config.min_timeout, min(override_timeout, self.config.max_timeout)
            )

        if not self.config.enabled:
            return self.config.max_timeout

        if self.config.strategy == TimeoutStrategy.FIXED:
            return self.config.default_timeout

        elif self.config.strategy == TimeoutStrategy.ADAPTIVE:
            return self._calculate_adaptive_timeout()

        elif self.config.strategy == TimeoutStrategy.PERCENTILE:
            return self._calculate_percentile_timeout()

        elif self.config.strategy == TimeoutStrategy.CASCADING:
            return self._calculate_cascading_timeout()

        else:
            return self.config.default_timeout

    def _calculate_adaptive_timeout(self) -> float:
        """Calculate adaptive timeout based on recent performance."""

        avg_duration = self.history.get_average_duration()

        if avg_duration is None:
            return self.config.default_timeout

        # Adapt timeout based on recent average with safety factor
        adaptive_timeout = avg_duration * self.config.adaptation_factor

        # Consider success rate - increase timeout if success rate is low
        success_rate = self.history.get_success_rate()
        if success_rate < 0.8:  # If success rate below 80%
            adaptive_timeout *= 1.5

        # Bound the timeout
        return max(
            self.config.min_timeout, min(adaptive_timeout, self.config.max_timeout)
        )

    def _calculate_percentile_timeout(self) -> float:
        """Calculate timeout based on response time percentile."""

        percentile_duration = self.history.get_percentile_duration(
            self.config.percentile
        )

        if percentile_duration is None:
            return self.config.default_timeout

        # Add buffer to percentile duration
        timeout = percentile_duration * 1.2  # 20% buffer

        return max(self.config.min_timeout, min(timeout, self.config.max_timeout))

    def _calculate_cascading_timeout(self) -> float:
        """Calculate cascading timeout with fallback levels."""

        # Start with adaptive timeout as base
        base_timeout = self._calculate_adaptive_timeout()

        # Adjust based on recent timeout rate
        timeout_rate = self.history.get_timeout_rate()

        if timeout_rate > 0.2:  # If more than 20% timeouts
            # Increase timeout progressively
            multiplier = 1.0 + (timeout_rate * 2)  # Up to 3x for 100% timeout rate
            base_timeout *= multiplier

        return max(self.config.min_timeout, min(base_timeout, self.config.max_timeout))

    async def execute_with_timeout(
        self,
        operation: Callable,
        *args,
        timeout_override: Optional[float] = None,
        **kwargs,
    ) -> TimeoutResult:
        """
        Execute an operation with timeout management.

        Args:
            operation: Async or sync function to execute
            *args: Operation arguments
            timeout_override: Optional timeout override
            **kwargs: Operation keyword arguments

        Returns:
            TimeoutResult with execution details
        """

        timeout = self.calculate_timeout(timeout_override)
        start_time = time.time()
        self._total_operations += 1

        try:
            if asyncio.iscoroutinefunction(operation):
                result = await asyncio.wait_for(
                    operation(*args, **kwargs), timeout=timeout
                )
            else:
                # Execute sync function in thread pool with timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, operation, *args, **kwargs),
                    timeout=timeout,
                )

            duration = time.time() - start_time

            # Record success
            self.history.add_result(duration, timeout, True)
            self._successful_operations += 1

            timeout_result = TimeoutResult(
                success=True, duration=duration, timeout_used=timeout
            )

            self.logger.log_security_event(
                SecurityLogType.PERFORMANCE_METRIC,
                f"Operation completed within timeout",
                level=LogLevel.DEBUG,
                metadata={
                    "policy_name": self.config.name,
                    "duration": duration,
                    "timeout": timeout,
                    "strategy": self.config.strategy.value,
                },
            )

            return timeout_result

        except asyncio.TimeoutError:
            duration = time.time() - start_time

            # Record timeout
            self.history.add_result(duration, timeout, False)
            self._timeout_operations += 1
            self._failed_operations += 1

            timeout_result = TimeoutResult(
                success=False,
                duration=duration,
                timeout_used=timeout,
                timeout_occurred=True,
                error_message=f"Operation timed out after {timeout:.2f}s",
            )

            self.logger.log_security_event(
                SecurityLogType.PERFORMANCE_METRIC,
                f"Operation timeout in policy '{self.config.name}'",
                level=LogLevel.WARNING,
                metadata={
                    "policy_name": self.config.name,
                    "timeout": timeout,
                    "strategy": self.config.strategy.value,
                    "timeout_rate": self.history.get_timeout_rate(),
                },
            )

            return timeout_result

        except Exception as e:
            duration = time.time() - start_time

            # Record failure
            self.history.add_result(duration, timeout, False)
            self._failed_operations += 1

            timeout_result = TimeoutResult(
                success=False,
                duration=duration,
                timeout_used=timeout,
                error_message=str(e),
            )

            self.logger.log_security_event(
                SecurityLogType.PERFORMANCE_METRIC,
                f"Operation failed in policy '{self.config.name}': {str(e)}",
                level=LogLevel.ERROR,
                metadata={
                    "policy_name": self.config.name,
                    "duration": duration,
                    "timeout": timeout,
                    "error": str(e),
                },
            )

            return timeout_result

    def get_metrics(self) -> Dict[str, Any]:
        """Get timeout policy metrics."""

        return {
            "policy_name": self.config.name,
            "strategy": self.config.strategy.value,
            "current_timeout": self._current_timeout,
            "default_timeout": self.config.default_timeout,
            "min_timeout": self.config.min_timeout,
            "max_timeout": self.config.max_timeout,
            "enabled": self.config.enabled,
            "statistics": {
                "total_operations": self._total_operations,
                "successful_operations": self._successful_operations,
                "timeout_operations": self._timeout_operations,
                "failed_operations": self._failed_operations,
                "success_rate": (
                    self._successful_operations / self._total_operations
                    if self._total_operations > 0
                    else 0
                ),
                "timeout_rate": (
                    self._timeout_operations / self._total_operations
                    if self._total_operations > 0
                    else 0
                ),
                "failure_rate": (
                    self._failed_operations / self._total_operations
                    if self._total_operations > 0
                    else 0
                ),
            },
            "history_metrics": {
                "average_duration": self.history.get_average_duration(),
                "percentile_duration": self.history.get_percentile_duration(
                    self.config.percentile
                ),
                "historical_success_rate": self.history.get_success_rate(),
                "historical_timeout_rate": self.history.get_timeout_rate(),
                "history_size": len(self.history.durations),
            },
        }

    def reset_history(self):
        """Reset timeout history."""
        self.history = TimeoutHistory(self.config.history_size)
        self._current_timeout = self.config.default_timeout

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Reset timeout history for policy '{self.config.name}'",
            level=LogLevel.INFO,
        )


@asynccontextmanager
async def timeout_context(timeout: float, name: str = "operation"):
    """
    Context manager for simple timeout handling.

    Args:
        timeout: Timeout in seconds
        name: Operation name for logging
    """

    logger = get_security_logger("cronos.security.resilience.timeout_context")
    start_time = time.time()

    try:
        yield
        duration = time.time() - start_time

        logger.log_security_event(
            SecurityLogType.PERFORMANCE_METRIC,
            f"Timeout context '{name}' completed",
            level=LogLevel.DEBUG,
            metadata={"duration": duration, "timeout": timeout},
        )

    except asyncio.TimeoutError:
        duration = time.time() - start_time

        logger.log_security_event(
            SecurityLogType.PERFORMANCE_METRIC,
            f"Timeout context '{name}' timed out",
            level=LogLevel.WARNING,
            metadata={"duration": duration, "timeout": timeout},
        )

        raise


class TimeoutManager:
    """
    Manages multiple timeout policies and provides centralized timeout management
    for the Zero-Touch Security Orchestrator.
    """

    def __init__(self):
        self.policies: Dict[str, TimeoutPolicy] = {}
        self.logger = get_security_logger("cronos.security.resilience.timeout_manager")

        # Default policies
        self._create_default_policies()

    def _create_default_policies(self):
        """Create default timeout policies for common scenarios."""

        # Fast operations (API calls, simple queries)
        self.create_policy(
            name="fast",
            default_timeout=5.0,
            strategy=TimeoutStrategy.ADAPTIVE,
            min_timeout=1.0,
            max_timeout=15.0,
        )

        # Standard operations (most business logic)
        self.create_policy(
            name="standard",
            default_timeout=30.0,
            strategy=TimeoutStrategy.PERCENTILE,
            min_timeout=5.0,
            max_timeout=60.0,
            percentile=95.0,
        )

        # Slow operations (heavy processing, file I/O)
        self.create_policy(
            name="slow",
            default_timeout=120.0,
            strategy=TimeoutStrategy.CASCADING,
            min_timeout=30.0,
            max_timeout=300.0,
        )

        # Critical operations (security-critical, cannot fail)
        self.create_policy(
            name="critical",
            default_timeout=60.0,
            strategy=TimeoutStrategy.FIXED,
            min_timeout=60.0,
            max_timeout=60.0,
        )

    def create_policy(
        self,
        name: str,
        default_timeout: float = 30.0,
        strategy: TimeoutStrategy = TimeoutStrategy.FIXED,
        **kwargs,
    ) -> TimeoutPolicy:
        """Create a new timeout policy."""

        config = TimeoutConfig(
            name=name, default_timeout=default_timeout, strategy=strategy, **kwargs
        )

        policy = TimeoutPolicy(config)
        self.policies[name] = policy

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Created timeout policy: {name}",
            level=LogLevel.INFO,
            metadata={
                "policy_name": name,
                "default_timeout": default_timeout,
                "strategy": strategy.value,
            },
        )

        return policy

    def get_policy(self, name: str) -> Optional[TimeoutPolicy]:
        """Get timeout policy by name."""
        return self.policies.get(name)

    def remove_policy(self, name: str) -> bool:
        """Remove a timeout policy."""

        if name in self.policies:
            del self.policies[name]

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Removed timeout policy: {name}",
                level=LogLevel.INFO,
            )

            return True

        return False

    async def execute_with_policy(
        self,
        policy_name: str,
        operation: Callable,
        *args,
        timeout_override: Optional[float] = None,
        **kwargs,
    ) -> TimeoutResult:
        """Execute operation with specified timeout policy."""

        policy = self.get_policy(policy_name)

        if not policy:
            # Use default policy if specified policy not found
            policy = self.get_policy("standard")

            if not policy:
                raise ValueError(
                    f"Timeout policy '{policy_name}' not found and no default available"
                )

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Using default timeout policy instead of '{policy_name}'",
                level=LogLevel.WARNING,
            )

        return await policy.execute_with_timeout(
            operation, *args, timeout_override=timeout_override, **kwargs
        )

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all timeout policies."""

        return {name: policy.get_metrics() for name, policy in self.policies.items()}

    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global timeout manager metrics."""

        total_operations = sum(p._total_operations for p in self.policies.values())
        successful_operations = sum(
            p._successful_operations for p in self.policies.values()
        )
        timeout_operations = sum(p._timeout_operations for p in self.policies.values())
        failed_operations = sum(p._failed_operations for p in self.policies.values())

        return {
            "total_policies": len(self.policies),
            "policy_names": list(self.policies.keys()),
            "global_statistics": {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "timeout_operations": timeout_operations,
                "failed_operations": failed_operations,
                "global_success_rate": (
                    successful_operations / total_operations
                    if total_operations > 0
                    else 0
                ),
                "global_timeout_rate": (
                    timeout_operations / total_operations if total_operations > 0 else 0
                ),
                "global_failure_rate": (
                    failed_operations / total_operations if total_operations > 0 else 0
                ),
            },
            "policies": self.get_all_metrics(),
        }

    def reset_all_history(self):
        """Reset history for all timeout policies."""

        for policy in self.policies.values():
            policy.reset_history()

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Reset history for {len(self.policies)} timeout policies",
            level=LogLevel.INFO,
        )


# Context manager class for timeout management
class TimeoutContext:
    """Context manager for timeout-managed operations."""

    def __init__(
        self,
        manager: TimeoutManager,
        policy_name: str = "standard",
        timeout_override: Optional[float] = None,
        operation_name: str = "unknown",
    ):
        self.manager = manager
        self.policy_name = policy_name
        self.timeout_override = timeout_override
        self.operation_name = operation_name
        self.start_time: Optional[float] = None

    async def __aenter__(self):
        """Enter timeout context."""
        self.start_time = time.time()

        policy = self.manager.get_policy(self.policy_name)
        if policy:
            timeout = policy.calculate_timeout(self.timeout_override)
        else:
            timeout = self.timeout_override or 30.0

        # Create timeout task
        self.timeout_task = asyncio.create_task(asyncio.sleep(timeout))

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit timeout context."""

        if not self.timeout_task.done():
            self.timeout_task.cancel()

        if self.start_time:
            duration = time.time() - self.start_time

            if exc_type == asyncio.TimeoutError:
                # Log timeout
                logger = get_security_logger(
                    "cronos.security.resilience.timeout_context"
                )
                logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Operation '{self.operation_name}' timed out",
                    level=LogLevel.WARNING,
                    metadata={
                        "operation": self.operation_name,
                        "duration": duration,
                        "policy": self.policy_name,
                    },
                )


# Global timeout manager instance
_timeout_manager: Optional[TimeoutManager] = None


def get_timeout_manager() -> TimeoutManager:
    """Get global timeout manager instance."""
    global _timeout_manager
    if _timeout_manager is None:
        _timeout_manager = TimeoutManager()
    return _timeout_manager
