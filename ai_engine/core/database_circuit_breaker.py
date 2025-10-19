"""
CRONOS AI Engine - Database Circuit Breaker Integration

Integrates circuit breaker pattern with database operations to prevent
cascading failures when database is experiencing issues.
"""

import logging
from typing import Optional, Callable, Any, TypeVar, ParamSpec
from functools import wraps

from sqlalchemy.exc import (
    OperationalError,
    DBAPIError,
    DatabaseError,
    TimeoutError as SQLTimeoutError,
)

from ai_engine.security.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerException,
    CircuitBreakerState,
)

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


class DatabaseCircuitBreakerManager:
    """
    Manages circuit breakers for database operations.

    Features:
    - Automatic circuit breaker per database operation type
    - Configurable failure thresholds
    - Graceful degradation when database is down
    - Metrics for monitoring
    """

    def __init__(self):
        """Initialize database circuit breaker manager."""
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig(
            failure_threshold=5,  # Open after 5 consecutive failures
            recovery_timeout=30,  # Try again after 30 seconds
            success_threshold=2,  # Close after 2 successful calls in half-open
            timeout=10.0,  # 10 second timeout for database operations
            expected_exception=Exception,
            # Sliding window settings
            sliding_window_size=50,
            minimum_requests=10,
            failure_rate_threshold=0.5,  # 50% failure rate
            slow_call_duration_threshold=5.0,  # >5s is considered slow
            slow_call_rate_threshold=0.3,  # 30% slow calls
        )

    def get_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker for a specific database operation.

        Args:
            name: Circuit breaker name (e.g., "database_read", "database_write")
            config: Optional custom configuration

        Returns:
            CircuitBreaker: Circuit breaker instance
        """
        if name not in self._circuit_breakers:
            breaker_config = config or self._default_config
            self._circuit_breakers[name] = CircuitBreaker(
                name=f"db_{name}",
                config=breaker_config
            )
            logger.info(f"Created circuit breaker for database operation: {name}")

        return self._circuit_breakers[name]

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """
        Get states of all circuit breakers.

        Returns:
            dict: Circuit breaker states and metrics
        """
        return {
            name: {
                "state": breaker._state.value,
                "failure_count": breaker._failure_count,
                "success_count": breaker._success_count,
                "total_calls": breaker._total_calls,
                "successful_calls": breaker._successful_calls,
                "failed_calls": breaker._failed_calls,
                "rejected_calls": breaker._rejected_calls,
                "slow_calls": breaker._slow_calls,
            }
            for name, breaker in self._circuit_breakers.items()
        }


# Global circuit breaker manager
_db_circuit_breaker_manager: Optional[DatabaseCircuitBreakerManager] = None


def get_db_circuit_breaker_manager() -> DatabaseCircuitBreakerManager:
    """
    Get global database circuit breaker manager.

    Returns:
        DatabaseCircuitBreakerManager: Circuit breaker manager
    """
    global _db_circuit_breaker_manager
    if _db_circuit_breaker_manager is None:
        _db_circuit_breaker_manager = DatabaseCircuitBreakerManager()
    return _db_circuit_breaker_manager


def with_database_circuit_breaker(
    name: str = "default",
    config: Optional[CircuitBreakerConfig] = None,
    fallback: Optional[Callable] = None,
):
    """
    Decorator to protect database operations with circuit breaker.

    Usage:
        @with_database_circuit_breaker(name="read_users")
        async def get_all_users(db: AsyncSession):
            result = await db.execute(select(User))
            return result.scalars().all()

        # With fallback
        async def get_cached_users():
            return get_from_cache("users")

        @with_database_circuit_breaker(
            name="read_users",
            fallback=get_cached_users
        )
        async def get_all_users(db: AsyncSession):
            # ... database operation ...
            pass

    Args:
        name: Circuit breaker name (default: "default")
        config: Optional custom circuit breaker configuration
        fallback: Optional fallback function when circuit is open

    Returns:
        Decorated function with circuit breaker protection
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            manager = get_db_circuit_breaker_manager()
            circuit_breaker = manager.get_circuit_breaker(name, config)

            try:
                # Call function with circuit breaker protection
                result = await circuit_breaker.call(func, *args, **kwargs)
                return result

            except CircuitBreakerException as e:
                # Circuit is open - use fallback if available
                logger.warning(
                    f"Circuit breaker '{name}' is {e.state.value} for {func.__name__}. "
                    f"Database may be experiencing issues."
                )

                if fallback:
                    logger.info(f"Using fallback for {func.__name__}")
                    try:
                        return await fallback(*args, **kwargs)
                    except Exception as fb_error:
                        logger.error(f"Fallback also failed for {func.__name__}: {fb_error}")
                        raise

                # No fallback available
                raise RuntimeError(
                    f"Database operation '{func.__name__}' unavailable due to circuit breaker. "
                    f"State: {e.state.value}"
                ) from e

            except (OperationalError, DBAPIError, DatabaseError, SQLTimeoutError) as e:
                # Database-specific errors
                logger.error(
                    f"Database error in {func.__name__} (circuit: {name}): {e}"
                )
                raise

        return wrapper
    return decorator


async def is_database_available(circuit_breaker_name: str = "default") -> bool:
    """
    Check if database is available based on circuit breaker state.

    Args:
        circuit_breaker_name: Name of circuit breaker to check

    Returns:
        bool: True if database is available (circuit closed or half-open)
    """
    manager = get_db_circuit_breaker_manager()
    circuit_breaker = manager.get_circuit_breaker(circuit_breaker_name)

    return circuit_breaker._state in (
        CircuitBreakerState.CLOSED,
        CircuitBreakerState.HALF_OPEN
    )


async def reset_database_circuit_breaker(circuit_breaker_name: str = "default") -> None:
    """
    Manually reset a database circuit breaker.

    Use this after manually verifying database is operational.

    Args:
        circuit_breaker_name: Name of circuit breaker to reset
    """
    manager = get_db_circuit_breaker_manager()
    circuit_breaker = manager.get_circuit_breaker(circuit_breaker_name)

    await circuit_breaker.reset()
    logger.info(f"Database circuit breaker '{circuit_breaker_name}' manually reset")


def get_database_health_status() -> dict[str, Any]:
    """
    Get health status of all database circuit breakers.

    Returns:
        dict: Health status with circuit breaker states
    """
    from datetime import datetime

    manager = get_db_circuit_breaker_manager()
    states = manager.get_all_states()

    # Determine overall health
    all_closed = all(
        state["state"] == CircuitBreakerState.CLOSED.value
        for state in states.values()
    )

    any_open = any(
        state["state"] == CircuitBreakerState.OPEN.value
        for state in states.values()
    )

    if all_closed:
        overall_status = "healthy"
    elif any_open:
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    return {
        "overall_status": overall_status,
        "circuit_breakers": states,
        "timestamp": datetime.utcnow().isoformat(),
    }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from sqlalchemy.ext.asyncio import AsyncSession

    # Example protected database operation
    @with_database_circuit_breaker(name="read_users")
    async def get_users(db: AsyncSession):
        """Example database read operation."""
        from sqlalchemy import select
        result = await db.execute(select("SELECT 1"))
        return result.fetchall()

    # Example with fallback
    async def get_cached_users():
        """Fallback to cache when database is down."""
        return [{"id": 1, "name": "cached_user"}]

    @with_database_circuit_breaker(
        name="read_users_with_fallback",
        fallback=get_cached_users
    )
    async def get_users_with_fallback(db: AsyncSession):
        """Example with fallback to cache."""
        from sqlalchemy import select
        result = await db.execute(select("SELECT * FROM users"))
        return result.fetchall()

    # Check health
    async def check_health():
        status = get_database_health_status()
        print(f"Database health: {status}")

    asyncio.run(check_health())
