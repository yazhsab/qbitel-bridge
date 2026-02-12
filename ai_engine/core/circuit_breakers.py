"""
QBITEL Engine - Circuit Breaker Infrastructure

Provides circuit breaker patterns to prevent cascade failures across services.
Uses pybreaker with Prometheus metrics integration.

Circuit States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is failing, requests fail immediately
- HALF_OPEN: Testing if service recovered

Usage:
    from ai_engine.core.circuit_breakers import circuit_breakers, with_circuit_breaker

    # As decorator
    @with_circuit_breaker("llm")
    async def call_llm(prompt: str):
        ...

    # As context manager
    async with circuit_breakers["database"]:
        await db.execute(...)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================

CIRCUIT_STATE = Gauge(
    "qbitel_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["name"],
)

CIRCUIT_FAILURES = Counter(
    "qbitel_circuit_breaker_failures_total",
    "Total circuit breaker failure count",
    ["name", "exception_type"],
)

CIRCUIT_SUCCESSES = Counter(
    "qbitel_circuit_breaker_successes_total",
    "Total circuit breaker success count",
    ["name"],
)

CIRCUIT_REJECTIONS = Counter(
    "qbitel_circuit_breaker_rejections_total",
    "Requests rejected due to open circuit",
    ["name"],
)

CIRCUIT_STATE_CHANGES = Counter(
    "qbitel_circuit_breaker_state_changes_total",
    "Total circuit breaker state transitions",
    ["name", "from_state", "to_state"],
)

CIRCUIT_CALL_DURATION = Histogram(
    "qbitel_circuit_breaker_call_duration_seconds",
    "Duration of calls through circuit breaker",
    ["name", "status"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
)


# =============================================================================
# Circuit Breaker States
# =============================================================================


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


# =============================================================================
# Exceptions
# =============================================================================


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and request is rejected."""

    def __init__(self, circuit_name: str, time_until_retry: float):
        self.circuit_name = circuit_name
        self.time_until_retry = time_until_retry
        super().__init__(f"Circuit '{circuit_name}' is open. Retry in {time_until_retry:.1f}s")


# =============================================================================
# Circuit Breaker Configuration
# =============================================================================


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    name: str
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    recovery_timeout: float = 60.0  # Seconds before half-open attempt
    half_open_max_calls: int = 3  # Max calls in half-open state
    excluded_exceptions: Set[Type[Exception]] = field(default_factory=set)
    included_exceptions: Optional[Set[Type[Exception]]] = None  # If set, only these count


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================


class CircuitBreaker:
    """
    Async circuit breaker implementation.

    Prevents cascade failures by failing fast when a service is unhealthy.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.name = config.name

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        # History for debugging
        self._failure_history: List[Dict[str, Any]] = []
        self._max_history = 100

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Initialize metrics
        CIRCUIT_STATE.labels(name=self.name).set(0)  # Start closed

        logger.info(f"Circuit breaker '{self.name}' initialized")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)."""
        return self._state == CircuitState.HALF_OPEN

    async def __aenter__(self):
        """Context manager entry."""
        await self._before_call()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_val is not None:
            await self._on_failure(exc_val)
        else:
            await self._on_success()
        return False  # Don't suppress exceptions

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Any exception from the function
        """
        await self._before_call()

        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._on_success()
            duration = time.time() - start_time
            CIRCUIT_CALL_DURATION.labels(name=self.name, status="success").observe(duration)
            return result

        except Exception as e:
            await self._on_failure(e)
            duration = time.time() - start_time
            CIRCUIT_CALL_DURATION.labels(name=self.name, status="failure").observe(duration)
            raise

    async def _before_call(self) -> None:
        """Check circuit state before allowing a call."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.recovery_timeout:
                        # Transition to half-open
                        self._transition_to(CircuitState.HALF_OPEN)
                        return

                # Still open, reject the call
                time_until_retry = self.config.recovery_timeout - (time.time() - (self._last_failure_time or time.time()))
                CIRCUIT_REJECTIONS.labels(name=self.name).inc()
                raise CircuitOpenError(self.name, max(0, time_until_retry))

            if self._state == CircuitState.HALF_OPEN:
                # Check if we've exceeded half-open call limit
                if self._half_open_calls >= self.config.half_open_max_calls:
                    # Too many concurrent calls, reject
                    CIRCUIT_REJECTIONS.labels(name=self.name).inc()
                    raise CircuitOpenError(self.name, 5.0)  # Suggest retry in 5s
                self._half_open_calls += 1

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            CIRCUIT_SUCCESSES.labels(name=self.name).inc()

            if self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
                return

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                self._half_open_calls = max(0, self._half_open_calls - 1)

                if self._success_count >= self.config.success_threshold:
                    # Enough successes, close the circuit
                    self._transition_to(CircuitState.CLOSED)

    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed call."""
        async with self._lock:
            # Check if this exception should be counted
            if not self._should_count_exception(exception):
                return

            CIRCUIT_FAILURES.labels(name=self.name, exception_type=type(exception).__name__).inc()

            # Record failure
            self._failure_count += 1
            self._last_failure_time = time.time()
            self._record_failure(exception)

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                self._half_open_calls = max(0, self._half_open_calls - 1)
                self._transition_to(CircuitState.OPEN)

    def _should_count_exception(self, exception: Exception) -> bool:
        """Check if exception should count toward failure threshold."""
        exc_type = type(exception)

        # Check excluded exceptions
        if exc_type in self.config.excluded_exceptions:
            return False

        for excluded in self.config.excluded_exceptions:
            if isinstance(exception, excluded):
                return False

        # Check included exceptions (if specified)
        if self.config.included_exceptions is not None:
            if exc_type not in self.config.included_exceptions:
                for included in self.config.included_exceptions:
                    if isinstance(exception, included):
                        return True
                return False

        return True

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state

        # Update metrics
        state_values = {
            CircuitState.CLOSED: 0,
            CircuitState.OPEN: 1,
            CircuitState.HALF_OPEN: 2,
        }
        CIRCUIT_STATE.labels(name=self.name).set(state_values[new_state])
        CIRCUIT_STATE_CHANGES.labels(name=self.name, from_state=old_state.value, to_state=new_state.value).inc()

        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.OPEN:
            self._last_failure_time = time.time()

        logger.info(f"Circuit breaker '{self.name}' transitioned: {old_state.value} -> {new_state.value}")

    def _record_failure(self, exception: Exception) -> None:
        """Record failure in history."""
        self._failure_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(exception).__name__,
                "message": str(exception)[:200],
            }
        )
        # Keep history bounded
        if len(self._failure_history) > self._max_history:
            self._failure_history = self._failure_history[-self._max_history :]

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": (
                datetime.fromtimestamp(self._last_failure_time, timezone.utc).isoformat() if self._last_failure_time else None
            ),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "recovery_timeout": self.config.recovery_timeout,
            },
            "recent_failures": self._failure_history[-5:],
        }

    async def reset(self) -> None:
        """Manually reset circuit to closed state."""
        async with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_history.clear()
            logger.info(f"Circuit breaker '{self.name}' manually reset")


# =============================================================================
# Circuit Breaker Registry
# =============================================================================


class CircuitBreakerRegistry:
    """
    Registry for managing circuit breakers across the application.
    """

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def register(self, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker."""
        if config.name in self._breakers:
            logger.warning(f"Circuit breaker '{config.name}' already registered")
            return self._breakers[config.name]

        breaker = CircuitBreaker(config)
        self._breakers[config.name] = breaker
        return breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def __getitem__(self, name: str) -> CircuitBreaker:
        """Get circuit breaker by name (raises KeyError if not found)."""
        if name not in self._breakers:
            raise KeyError(f"Circuit breaker '{name}' not registered")
        return self._breakers[name]

    def __contains__(self, name: str) -> bool:
        """Check if circuit breaker is registered."""
        return name in self._breakers

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() for name, breaker in self._breakers.items()}

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()


# =============================================================================
# Global Registry with Pre-configured Breakers
# =============================================================================

# Create global registry
circuit_breakers = CircuitBreakerRegistry()

# Register default circuit breakers
circuit_breakers.register(
    CircuitBreakerConfig(
        name="llm",
        failure_threshold=3,
        recovery_timeout=60.0,
        half_open_max_calls=2,
        # Don't trip on validation errors
        excluded_exceptions={ValueError, TypeError},
    )
)

circuit_breakers.register(
    CircuitBreakerConfig(
        name="database",
        failure_threshold=5,
        recovery_timeout=30.0,
        half_open_max_calls=3,
    )
)

circuit_breakers.register(
    CircuitBreakerConfig(
        name="redis",
        failure_threshold=3,
        recovery_timeout=15.0,
        half_open_max_calls=2,
    )
)

circuit_breakers.register(
    CircuitBreakerConfig(
        name="discovery",
        failure_threshold=5,
        recovery_timeout=60.0,
        half_open_max_calls=2,
    )
)

circuit_breakers.register(
    CircuitBreakerConfig(
        name="external_api",
        failure_threshold=5,
        recovery_timeout=120.0,
        half_open_max_calls=1,
    )
)


# =============================================================================
# Decorator
# =============================================================================

T = TypeVar("T")


def with_circuit_breaker(
    circuit_name: str,
    fallback: Optional[Callable[..., T]] = None,
) -> Callable:
    """
    Decorator to wrap function with circuit breaker.

    Args:
        circuit_name: Name of the circuit breaker to use
        fallback: Optional fallback function if circuit is open

    Example:
        @with_circuit_breaker("llm")
        async def call_llm(prompt: str):
            ...

        @with_circuit_breaker("database", fallback=lambda: [])
        async def get_items():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            breaker = circuit_breakers.get(circuit_name)
            if breaker is None:
                logger.warning(f"Circuit breaker '{circuit_name}' not found, executing without protection")
                return await func(*args, **kwargs)

            try:
                return await breaker.call(func, *args, **kwargs)
            except CircuitOpenError:
                if fallback is not None:
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    return fallback(*args, **kwargs)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to run in event loop
            breaker = circuit_breakers.get(circuit_name)
            if breaker is None:
                return func(*args, **kwargs)

            loop = asyncio.get_event_loop()
            return loop.run_until_complete(breaker.call(func, *args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
