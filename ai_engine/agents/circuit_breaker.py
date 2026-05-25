"""
QBITEL - Circuit Breaker Pattern for Agent Resilience

Implements the circuit breaker pattern to protect agents from cascading
failures. When an external dependency (LLM, tool, downstream agent) starts
failing, the circuit opens and fast-fails subsequent calls, preventing
resource exhaustion and letting the dependency recover.

States:
    CLOSED  → normal operation, calls pass through
    OPEN    → dependency is failing, calls are short-circuited
    HALF_OPEN → trial call is allowed to test recovery

Features:
- Per-dependency circuit breakers
- Configurable failure thresholds
- Exponential backoff on retries
- Health metrics and alerting
- Async-native design
- Bulkhead isolation (concurrency limiter)
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

CIRCUIT_STATE = Gauge(
    "qbitel_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half_open, 2=open)",
    ["circuit_name"],
)
CIRCUIT_CALLS = Counter(
    "qbitel_circuit_breaker_calls_total",
    "Total calls through circuit breaker",
    ["circuit_name", "outcome"],  # success, failure, rejected
)
CIRCUIT_TRANSITIONS = Counter(
    "qbitel_circuit_breaker_transitions_total",
    "Circuit state transitions",
    ["circuit_name", "from_state", "to_state"],
)
CIRCUIT_RESPONSE_TIME = Histogram(
    "qbitel_circuit_breaker_response_seconds",
    "Call response time through circuit breaker",
    ["circuit_name"],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data classes
# ---------------------------------------------------------------------------


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FailureType(str, Enum):
    """Classification of failures for circuit breaker decisions."""

    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker instance."""

    name: str

    # Failure thresholds
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes in half-open before closing
    failure_rate_threshold: float = 0.5  # Failure rate to trigger opening

    # Timing
    reset_timeout_seconds: float = 30.0  # Time in OPEN before trying HALF_OPEN
    call_timeout_seconds: float = 60.0  # Timeout for individual calls
    sliding_window_seconds: float = 120.0  # Window for failure rate calculation

    # Backoff
    enable_exponential_backoff: bool = True
    max_backoff_seconds: float = 300.0  # 5 minutes max backoff
    backoff_multiplier: float = 2.0

    # Bulkhead (concurrency limiting)
    max_concurrent_calls: int = 10  # 0 = unlimited
    max_wait_seconds: float = 5.0  # Max wait for semaphore

    # Exceptions that should NOT count as failures
    ignored_exceptions: List[type] = field(default_factory=list)

    # Exceptions that ALWAYS count as failures
    record_exceptions: List[type] = field(default_factory=list)

    # Fallback
    fallback_function: Optional[Callable] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerMetrics:
    """Runtime metrics for a circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    timeout_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    open_count: int = 0  # Times circuit opened
    half_open_count: int = 0
    total_response_time: float = 0.0

    # Sliding window for failure rate
    _recent_outcomes: List[Tuple[datetime, bool]] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate from recent outcomes."""
        if not self._recent_outcomes:
            return 0.0
        failures = sum(1 for _, success in self._recent_outcomes if not success)
        return failures / len(self._recent_outcomes)

    @property
    def average_response_time(self) -> float:
        """Average response time."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_response_time / self.successful_calls

    def record_outcome(self, success: bool, window_seconds: float) -> None:
        """Record an outcome and prune old entries."""
        now = datetime.utcnow()
        self._recent_outcomes.append((now, success))

        # Prune entries outside the sliding window
        cutoff = now - timedelta(seconds=window_seconds)
        self._recent_outcomes = [
            (ts, s) for ts, s in self._recent_outcomes if ts >= cutoff
        ]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    def __init__(self, circuit_name: str, message: str):
        self.circuit_name = circuit_name
        super().__init__(f"CircuitBreaker[{circuit_name}]: {message}")


class CircuitOpenError(CircuitBreakerError):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, circuit_name: str, remaining_seconds: float):
        self.remaining_seconds = remaining_seconds
        super().__init__(
            circuit_name,
            f"Circuit is OPEN. Retry after {remaining_seconds:.1f}s",
        )


class BulkheadFullError(CircuitBreakerError):
    """Raised when the concurrency limiter is saturated."""

    def __init__(self, circuit_name: str, max_concurrent: int):
        self.max_concurrent = max_concurrent
        super().__init__(
            circuit_name,
            f"Bulkhead full ({max_concurrent} concurrent calls)",
        )


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """
    Async circuit breaker with bulkhead isolation.

    Usage:

        cb = CircuitBreaker(CircuitBreakerConfig(name="llm"))

        try:
            result = await cb.call(some_async_function, arg1, arg2)
        except CircuitOpenError:
            # Handle fast-fail
            ...
        except BulkheadFullError:
            # Handle concurrency limit
            ...

    Or as a decorator:

        @cb.protect
        async def call_llm(prompt: str) -> str:
            ...
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()

        # Backoff tracking
        self._current_backoff = config.reset_timeout_seconds
        self._open_since: Optional[datetime] = None

        # Bulkhead semaphore
        self._semaphore: Optional[asyncio.Semaphore] = None
        if config.max_concurrent_calls > 0:
            self._semaphore = asyncio.Semaphore(config.max_concurrent_calls)

        # Event listeners
        self._on_state_change: List[Callable] = []
        self._on_failure: List[Callable] = []
        self._on_success: List[Callable] = []

        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        CIRCUIT_STATE.labels(circuit_name=config.name).set(0)

    # ------ Public API ------

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Async callable to execute.
            *args, **kwargs: Arguments for the callable.

        Returns:
            Result of the function.

        Raises:
            CircuitOpenError: If circuit is open.
            BulkheadFullError: If concurrency limit reached.
        """
        # Phase 1: Check circuit state
        await self._check_state()

        # Phase 2: Acquire bulkhead permit
        if self._semaphore:
            try:
                acquired = await asyncio.wait_for(
                    self._acquire_semaphore(),
                    timeout=self.config.max_wait_seconds,
                )
                if not acquired:
                    raise BulkheadFullError(
                        self.config.name, self.config.max_concurrent_calls
                    )
            except asyncio.TimeoutError:
                self.metrics.rejected_calls += 1
                CIRCUIT_CALLS.labels(
                    circuit_name=self.config.name, outcome="rejected"
                ).inc()
                raise BulkheadFullError(
                    self.config.name, self.config.max_concurrent_calls
                )

        # Phase 3: Execute call
        start_time = time.time()
        try:
            if self.config.call_timeout_seconds > 0:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.call_timeout_seconds,
                )
            else:
                result = await func(*args, **kwargs)

            elapsed = time.time() - start_time
            await self._on_call_success(elapsed)
            return result

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self.metrics.timeout_calls += 1
            await self._on_call_failure(elapsed, FailureType.TIMEOUT)
            raise

        except Exception as exc:
            elapsed = time.time() - start_time

            # Check if this exception should be ignored
            if self._should_ignore(exc):
                await self._on_call_success(elapsed)
                raise

            failure_type = self._classify_failure(exc)
            await self._on_call_failure(elapsed, failure_type)
            raise

        finally:
            if self._semaphore:
                self._semaphore.release()

    def protect(self, func: Callable) -> Callable:
        """
        Decorator to protect an async function with this circuit breaker.

        Usage:
            @circuit_breaker.protect
            async def call_llm(prompt: str) -> str:
                ...
        """

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)

        return wrapper

    async def call_with_fallback(
        self,
        func: Callable,
        fallback: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute with automatic fallback if circuit is open or call fails.

        Args:
            func: Primary function to call.
            fallback: Fallback function if primary fails.
            *args, **kwargs: Arguments for both functions.
        """
        try:
            return await self.call(func, *args, **kwargs)
        except (CircuitOpenError, BulkheadFullError):
            self.logger.info(
                f"Using fallback due to circuit state ({self.state.value})"
            )
            return await fallback(*args, **kwargs)
        except Exception as e:
            self.logger.warning(
                f"Primary call failed ({e}), using fallback"
            )
            return await fallback(*args, **kwargs)

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self._current_backoff = self.config.reset_timeout_seconds
        self._open_since = None
        self.metrics.consecutive_failures = 0
        self.metrics.consecutive_successes = 0
        self._update_state_metric()
        self.logger.info(f"Circuit manually reset from {old_state.value}")

    # ------ State Management ------

    async def _check_state(self) -> None:
        """Check circuit state and handle transitions."""
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return  # Normal operation

            if self.state == CircuitState.OPEN:
                # Check if enough time has passed to try half-open
                if self._open_since:
                    elapsed = (datetime.utcnow() - self._open_since).total_seconds()
                    if elapsed >= self._current_backoff:
                        self._transition_to(CircuitState.HALF_OPEN)
                        return

                remaining = self._current_backoff
                if self._open_since:
                    remaining -= (
                        datetime.utcnow() - self._open_since
                    ).total_seconds()

                raise CircuitOpenError(self.config.name, max(0, remaining))

            if self.state == CircuitState.HALF_OPEN:
                return  # Allow trial call

    async def _on_call_success(self, elapsed: float) -> None:
        """Handle a successful call."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.total_response_time += elapsed
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = datetime.utcnow()
            self.metrics.record_outcome(True, self.config.sliding_window_seconds)

            CIRCUIT_CALLS.labels(
                circuit_name=self.config.name, outcome="success"
            ).inc()
            CIRCUIT_RESPONSE_TIME.labels(
                circuit_name=self.config.name
            ).observe(elapsed)

            # In HALF_OPEN, check if enough successes to close
            if self.state == CircuitState.HALF_OPEN:
                if (
                    self.metrics.consecutive_successes
                    >= self.config.success_threshold
                ):
                    self._transition_to(CircuitState.CLOSED)
                    # Reset backoff on recovery
                    self._current_backoff = self.config.reset_timeout_seconds

        # Notify listeners
        for listener in self._on_success:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(self.config.name, elapsed)
                else:
                    listener(self.config.name, elapsed)
            except Exception:
                pass

    async def _on_call_failure(
        self, elapsed: float, failure_type: FailureType
    ) -> None:
        """Handle a failed call."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = datetime.utcnow()
            self.metrics.record_outcome(False, self.config.sliding_window_seconds)

            CIRCUIT_CALLS.labels(
                circuit_name=self.config.name, outcome="failure"
            ).inc()

            # Determine whether to open the circuit
            should_open = False

            # Threshold-based: consecutive failures
            if (
                self.metrics.consecutive_failures
                >= self.config.failure_threshold
            ):
                should_open = True

            # Rate-based: failure rate in sliding window
            if (
                self.metrics.failure_rate
                >= self.config.failure_rate_threshold
                and len(self.metrics._recent_outcomes) >= 10
            ):
                should_open = True

            # In HALF_OPEN, any failure re-opens immediately
            if self.state == CircuitState.HALF_OPEN:
                should_open = True

            if should_open and self.state != CircuitState.OPEN:
                self._transition_to(CircuitState.OPEN)

                # Exponential backoff for repeated openings
                if self.config.enable_exponential_backoff:
                    self._current_backoff = min(
                        self._current_backoff * self.config.backoff_multiplier,
                        self.config.max_backoff_seconds,
                    )

        # Notify listeners
        for listener in self._on_failure:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(self.config.name, failure_type, elapsed)
                else:
                    listener(self.config.name, failure_type, elapsed)
            except Exception:
                pass

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new circuit state."""
        old_state = self.state
        if old_state == new_state:
            return

        self.state = new_state
        self.metrics.last_state_change = datetime.utcnow()

        if new_state == CircuitState.OPEN:
            self._open_since = datetime.utcnow()
            self.metrics.open_count += 1
        elif new_state == CircuitState.HALF_OPEN:
            self.metrics.half_open_count += 1
            self.metrics.consecutive_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._open_since = None

        self._update_state_metric()

        CIRCUIT_TRANSITIONS.labels(
            circuit_name=self.config.name,
            from_state=old_state.value,
            to_state=new_state.value,
        ).inc()

        self.logger.warning(
            f"Circuit [{self.config.name}] transition: "
            f"{old_state.value} → {new_state.value}"
        )

        # Notify state change listeners
        for listener in self._on_state_change:
            try:
                if asyncio.iscoroutinefunction(listener):
                    asyncio.create_task(
                        listener(self.config.name, old_state, new_state)
                    )
                else:
                    listener(self.config.name, old_state, new_state)
            except Exception:
                pass

    # ------ Helpers ------

    def _should_ignore(self, exc: Exception) -> bool:
        """Check if an exception should be ignored (not counted as failure)."""
        return any(isinstance(exc, t) for t in self.config.ignored_exceptions)

    def _classify_failure(self, exc: Exception) -> FailureType:
        """Classify an exception into a failure type."""
        exc_name = type(exc).__name__.lower()

        if "timeout" in exc_name:
            return FailureType.TIMEOUT
        if "connection" in exc_name or "connect" in exc_name:
            return FailureType.CONNECTION_ERROR
        if "ratelimit" in exc_name or "rate_limit" in exc_name or "429" in str(exc):
            return FailureType.RATE_LIMIT
        if any(code in str(exc) for code in ["500", "502", "503", "504"]):
            return FailureType.SERVER_ERROR
        if "validation" in exc_name or "invalid" in exc_name:
            return FailureType.VALIDATION_ERROR

        return FailureType.UNKNOWN

    async def _acquire_semaphore(self) -> bool:
        """Acquire the bulkhead semaphore."""
        if self._semaphore:
            await self._semaphore.acquire()
            return True
        return True

    def _update_state_metric(self) -> None:
        """Update Prometheus state gauge."""
        state_map = {
            CircuitState.CLOSED: 0,
            CircuitState.HALF_OPEN: 1,
            CircuitState.OPEN: 2,
        }
        CIRCUIT_STATE.labels(circuit_name=self.config.name).set(
            state_map[self.state]
        )

    # ------ Event Registration ------

    def on_state_change(self, callback: Callable) -> None:
        """Register a callback for state changes."""
        self._on_state_change.append(callback)

    def on_failure(self, callback: Callable) -> None:
        """Register a callback for failures."""
        self._on_failure.append(callback)

    def on_success(self, callback: Callable) -> None:
        """Register a callback for successes."""
        self._on_success.append(callback)

    # ------ Introspection ------

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "rejected_calls": self.metrics.rejected_calls,
            "timeout_calls": self.metrics.timeout_calls,
            "consecutive_failures": self.metrics.consecutive_failures,
            "consecutive_successes": self.metrics.consecutive_successes,
            "failure_rate": round(self.metrics.failure_rate, 3),
            "average_response_time": round(
                self.metrics.average_response_time, 3
            ),
            "open_count": self.metrics.open_count,
            "current_backoff": round(self._current_backoff, 1),
            "last_failure": (
                self.metrics.last_failure_time.isoformat()
                if self.metrics.last_failure_time
                else None
            ),
            "last_success": (
                self.metrics.last_success_time.isoformat()
                if self.metrics.last_success_time
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Circuit Breaker Registry
# ---------------------------------------------------------------------------


class CircuitBreakerRegistry:
    """
    Central registry for all circuit breakers in the system.

    Provides:
    - Named circuit breaker lookup
    - Global health dashboard
    - Bulk operations (reset all, etc.)
    - Default config templates for common patterns
    """

    # Pre-defined config templates
    TEMPLATES: Dict[str, Dict[str, Any]] = {
        "llm": {
            "failure_threshold": 3,
            "success_threshold": 2,
            "reset_timeout_seconds": 15.0,
            "call_timeout_seconds": 120.0,
            "max_concurrent_calls": 8,
            "enable_exponential_backoff": True,
            "max_backoff_seconds": 300.0,
        },
        "tool": {
            "failure_threshold": 5,
            "success_threshold": 3,
            "reset_timeout_seconds": 10.0,
            "call_timeout_seconds": 60.0,
            "max_concurrent_calls": 20,
            "enable_exponential_backoff": False,
        },
        "agent": {
            "failure_threshold": 3,
            "success_threshold": 2,
            "reset_timeout_seconds": 30.0,
            "call_timeout_seconds": 300.0,
            "max_concurrent_calls": 5,
            "enable_exponential_backoff": True,
            "max_backoff_seconds": 120.0,
        },
        "external_api": {
            "failure_threshold": 5,
            "success_threshold": 3,
            "reset_timeout_seconds": 60.0,
            "call_timeout_seconds": 30.0,
            "max_concurrent_calls": 10,
            "enable_exponential_backoff": True,
            "max_backoff_seconds": 600.0,
        },
    }

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.Registry")

    def get_or_create(
        self,
        name: str,
        template: Optional[str] = None,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """
        Get an existing circuit breaker or create a new one.

        Args:
            name: Unique name for the circuit breaker.
            template: One of the predefined templates (llm, tool, agent, external_api).
            config: Full custom config (overrides template).

        Returns:
            CircuitBreaker instance.
        """
        if name in self._breakers:
            return self._breakers[name]

        if config is None:
            # Use template if specified
            if template and template in self.TEMPLATES:
                template_vals = self.TEMPLATES[template]
                config = CircuitBreakerConfig(name=name, **template_vals)
            else:
                config = CircuitBreakerConfig(name=name)

        breaker = CircuitBreaker(config)
        self._breakers[name] = breaker
        self.logger.info(
            f"Created circuit breaker: {name} "
            f"(threshold={config.failure_threshold})"
        )
        return breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def remove(self, name: str) -> bool:
        """Remove a circuit breaker."""
        if name in self._breakers:
            del self._breakers[name]
            return True
        return False

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        for breaker in self._breakers.values():
            breaker.reset()
        self.logger.info(f"Reset all {len(self._breakers)} circuit breakers")

    def get_dashboard(self) -> Dict[str, Any]:
        """Get health dashboard for all circuit breakers."""
        breakers_status = {}
        open_count = 0
        half_open_count = 0
        total_failures = 0

        for name, breaker in self._breakers.items():
            status = breaker.get_status()
            breakers_status[name] = status

            if breaker.state == CircuitState.OPEN:
                open_count += 1
            elif breaker.state == CircuitState.HALF_OPEN:
                half_open_count += 1
            total_failures += status["failed_calls"]

        return {
            "total_breakers": len(self._breakers),
            "open_circuits": open_count,
            "half_open_circuits": half_open_count,
            "closed_circuits": len(self._breakers) - open_count - half_open_count,
            "total_failures": total_failures,
            "breakers": breakers_status,
            "health": "degraded" if open_count > 0 else "healthy",
        }

    def get_open_circuits(self) -> List[str]:
        """Get names of all open circuit breakers."""
        return [
            name
            for name, breaker in self._breakers.items()
            if breaker.state == CircuitState.OPEN
        ]

    def __len__(self) -> int:
        return len(self._breakers)

    def __contains__(self, name: str) -> bool:
        return name in self._breakers


# ---------------------------------------------------------------------------
# Retry Helper with Circuit Breaker
# ---------------------------------------------------------------------------


async def retry_with_circuit_breaker(
    func: Callable,
    circuit_breaker: CircuitBreaker,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    *args,
    **kwargs,
) -> Any:
    """
    Execute a function with retry logic and circuit breaker protection.

    Args:
        func: Async callable to execute.
        circuit_breaker: Circuit breaker to use.
        max_retries: Maximum number of retries.
        retry_delay: Initial delay between retries in seconds.
        backoff_factor: Multiplier for delay between retries.
        *args, **kwargs: Arguments for the callable.

    Returns:
        Result of the function.

    Raises:
        The last exception if all retries fail.
    """
    last_exception = None
    delay = retry_delay

    for attempt in range(max_retries + 1):
        try:
            return await circuit_breaker.call(func, *args, **kwargs)

        except CircuitOpenError:
            # Circuit is open — do not retry, let caller handle
            raise

        except BulkheadFullError:
            # Concurrency limit — wait and retry
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= backoff_factor
                continue
            raise

        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay *= backoff_factor
            else:
                raise

    raise last_exception  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Singleton global registry
# ---------------------------------------------------------------------------

# Module-level singleton for convenience
_global_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get or create the global CircuitBreakerRegistry singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CircuitBreakerRegistry()
    return _global_registry
