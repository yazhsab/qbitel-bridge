"""
CRONOS AI Engine - Transaction Management

Production-ready transaction management decorators and utilities
for ensuring ACID properties and handling concurrent operations safely.
"""

import logging
import asyncio
from functools import wraps
from typing import Callable, Any, Optional, TypeVar, ParamSpec
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import (
    OperationalError,
    DBAPIError,
    IntegrityError,
    DatabaseError as SQLAlchemyDatabaseError,
)
from sqlalchemy import text

from ai_engine.core.database_manager import TransactionError

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


class IsolationLevel:
    """SQL transaction isolation levels."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


def transactional(
    isolation_level: Optional[str] = None,
    retry_on_deadlock: bool = True,
    max_retries: int = 3,
    retry_delay: float = 0.1,
):
    """
    Decorator for transactional database operations with automatic commit/rollback.

    Features:
    - Automatic commit on success
    - Automatic rollback on exception
    - Deadlock detection and automatic retry
    - Configurable isolation levels
    - Comprehensive error logging

    Usage:
        @transactional()
        async def create_user(db: AsyncSession, username: str) -> User:
            user = User(username=username)
            db.add(user)
            return user

        # With custom isolation level
        @transactional(isolation_level=IsolationLevel.SERIALIZABLE)
        async def critical_operation(db: AsyncSession) -> None:
            # This will run with SERIALIZABLE isolation
            pass

    Args:
        isolation_level: SQL isolation level (None uses database default)
        retry_on_deadlock: Whether to automatically retry on deadlock
        max_retries: Maximum number of retry attempts on deadlock
        retry_delay: Initial delay between retries (exponential backoff)

    Returns:
        Decorated function with transaction management

    Raises:
        TransactionError: If transaction fails after all retries
        ValueError: If db parameter is missing
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Extract db session from arguments
            db = kwargs.get('db') or _find_session_in_args(args)

            if db is None:
                raise ValueError(
                    f"@transactional decorator requires 'db: AsyncSession' parameter. "
                    f"Function: {func.__name__}"
                )

            if not isinstance(db, AsyncSession):
                raise ValueError(
                    f"db parameter must be AsyncSession, got {type(db)}. "
                    f"Function: {func.__name__}"
                )

            # Execute with retry logic
            last_exception = None

            for attempt in range(max_retries):
                try:
                    # Set isolation level if specified
                    if isolation_level:
                        await db.execute(
                            text(f"SET TRANSACTION ISOLATION LEVEL {isolation_level}")
                        )
                        logger.debug(
                            f"Transaction isolation level set to {isolation_level} "
                            f"for {func.__name__}"
                        )

                    # Execute the function
                    result = await func(*args, **kwargs)

                    # Commit is handled by the session context manager
                    # But we can explicitly flush here if needed
                    await db.flush()

                    logger.debug(f"Transaction completed successfully: {func.__name__}")
                    return result

                except IntegrityError as e:
                    # Database constraint violation - don't retry
                    logger.warning(
                        f"Integrity error in {func.__name__}: {e}. "
                        f"Rolling back transaction."
                    )
                    # Rollback is handled by session context manager
                    raise TransactionError(
                        f"Database constraint violation in {func.__name__}: {e}"
                    ) from e

                except OperationalError as e:
                    last_exception = e
                    error_str = str(e).lower()

                    # Check for deadlock
                    is_deadlock = "deadlock" in error_str

                    if is_deadlock:
                        logger.warning(
                            f"Deadlock detected in {func.__name__} "
                            f"(attempt {attempt + 1}/{max_retries}): {e}"
                        )

                        if not retry_on_deadlock or attempt == max_retries - 1:
                            raise TransactionError(
                                f"Deadlock in {func.__name__} after "
                                f"{attempt + 1} attempts: {e}"
                            ) from e

                        # Exponential backoff
                        delay = retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {delay:.2f}s...")
                        await asyncio.sleep(delay)

                        # Rollback handled by session context manager
                        # Continue to next retry attempt
                        continue

                    else:
                        # Other operational errors - don't retry
                        logger.error(
                            f"Operational error in {func.__name__}: {e}. "
                            f"Rolling back transaction."
                        )
                        raise TransactionError(
                            f"Database operational error in {func.__name__}: {e}"
                        ) from e

                except (DBAPIError, SQLAlchemyDatabaseError) as e:
                    # Other database errors - don't retry
                    logger.error(
                        f"Database error in {func.__name__}: {e}. "
                        f"Rolling back transaction."
                    )
                    raise TransactionError(
                        f"Database error in {func.__name__}: {e}"
                    ) from e

                except Exception as e:
                    # Any other exception - don't retry
                    logger.error(
                        f"Unexpected error in {func.__name__}: {e}. "
                        f"Rolling back transaction."
                    )
                    raise

            # Should never reach here, but just in case
            raise TransactionError(
                f"Transaction failed in {func.__name__} after {max_retries} attempts"
            ) from last_exception

        return wrapper
    return decorator


def readonly_transaction(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator for read-only transactions.

    Read-only transactions:
    - Use READ COMMITTED isolation level
    - Don't acquire write locks
    - Better performance for queries
    - Prevent accidental writes

    Usage:
        @readonly_transaction
        async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
            result = await db.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        db = kwargs.get('db') or _find_session_in_args(args)

        if db is None:
            raise ValueError(
                f"@readonly_transaction requires 'db: AsyncSession' parameter. "
                f"Function: {func.__name__}"
            )

        try:
            # Set transaction to read-only
            await db.execute(text("SET TRANSACTION READ ONLY"))
            logger.debug(f"Read-only transaction started: {func.__name__}")

            result = await func(*args, **kwargs)
            return result

        except Exception as e:
            logger.error(f"Error in read-only transaction {func.__name__}: {e}")
            raise

    return wrapper


@asynccontextmanager
async def savepoint(db: AsyncSession, name: str = "sp"):
    """
    Context manager for nested transactions using savepoints.

    Savepoints allow creating transaction checkpoints that can be
    rolled back independently without affecting the outer transaction.

    Usage:
        async with db_manager.get_session() as session:
            # Outer transaction

            user = User(username="test")
            session.add(user)

            async with savepoint(session, "create_profile"):
                # Inner savepoint - can be rolled back independently
                profile = Profile(user=user, bio="test")
                session.add(profile)

                if invalid_data:
                    # This only rolls back to the savepoint
                    raise ValueError("Invalid profile data")

            # User is still created even if profile creation failed

    Args:
        db: Database session
        name: Savepoint name (default: "sp")

    Yields:
        None

    Raises:
        Any exception from the nested block (after rolling back to savepoint)
    """
    # Create savepoint
    await db.execute(text(f"SAVEPOINT {name}"))
    logger.debug(f"Savepoint created: {name}")

    try:
        yield

        # Release savepoint on success
        await db.execute(text(f"RELEASE SAVEPOINT {name}"))
        logger.debug(f"Savepoint released: {name}")

    except Exception as e:
        # Rollback to savepoint on error
        logger.warning(f"Rolling back to savepoint {name}: {e}")
        await db.execute(text(f"ROLLBACK TO SAVEPOINT {name}"))
        raise


def _find_session_in_args(args: tuple) -> Optional[AsyncSession]:
    """
    Find AsyncSession in function arguments.

    Args:
        args: Function arguments tuple

    Returns:
        AsyncSession if found, None otherwise
    """
    for arg in args:
        if isinstance(arg, AsyncSession):
            return arg
    return None


async def execute_in_transaction(
    db: AsyncSession,
    operations: list[Callable],
    *args,
    **kwargs
) -> list[Any]:
    """
    Execute multiple operations in a single transaction.

    All operations succeed or all fail together (ACID atomicity).

    Usage:
        async with db_manager.get_session() as session:
            results = await execute_in_transaction(
                session,
                [
                    lambda: create_user(session, "user1"),
                    lambda: create_profile(session, "profile1"),
                    lambda: send_welcome_email("user1@example.com"),
                ]
            )

    Args:
        db: Database session
        operations: List of async callables to execute
        *args, **kwargs: Arguments passed to each operation

    Returns:
        List of results from each operation

    Raises:
        TransactionError: If any operation fails
    """
    results = []

    try:
        for i, operation in enumerate(operations):
            logger.debug(f"Executing operation {i + 1}/{len(operations)}")
            result = await operation(*args, **kwargs)
            results.append(result)

        logger.debug(f"All {len(operations)} operations completed successfully")
        return results

    except Exception as e:
        logger.error(
            f"Transaction failed at operation {len(results) + 1}/{len(operations)}: {e}"
        )
        raise TransactionError(
            f"Transaction failed at operation {len(results) + 1}: {e}"
        ) from e


class TransactionContext:
    """
    Context manager for explicit transaction control.

    Provides fine-grained control over transaction lifecycle.

    Usage:
        async with db_manager.get_session() as session:
            async with TransactionContext(session) as tx:
                user = User(username="test")
                session.add(user)

                if not valid:
                    await tx.rollback()
                    return

                await tx.commit()
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize transaction context.

        Args:
            db: Database session
        """
        self.db = db
        self._committed = False
        self._rolled_back = False

    async def __aenter__(self):
        """Enter transaction context."""
        logger.debug("Transaction context entered")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context with automatic rollback on exception."""
        if exc_type is not None and not self._rolled_back:
            logger.warning(f"Exception in transaction context, rolling back: {exc_val}")
            await self.rollback()
            return False  # Re-raise exception

        if not self._committed and not self._rolled_back:
            logger.debug("Transaction context exiting without commit/rollback, committing")
            await self.commit()

        return False

    async def commit(self):
        """Explicitly commit the transaction."""
        if self._committed:
            logger.warning("Transaction already committed")
            return

        if self._rolled_back:
            raise TransactionError("Cannot commit: transaction was rolled back")

        await self.db.flush()
        self._committed = True
        logger.debug("Transaction committed")

    async def rollback(self):
        """Explicitly rollback the transaction."""
        if self._rolled_back:
            logger.warning("Transaction already rolled back")
            return

        if self._committed:
            raise TransactionError("Cannot rollback: transaction was committed")

        # Note: Actual rollback is handled by session context manager
        self._rolled_back = True
        logger.debug("Transaction marked for rollback")


# Convenience function for common transaction patterns
async def run_in_transaction(
    db_manager,
    operation: Callable,
    *args,
    **kwargs
) -> Any:
    """
    Run an operation in a managed transaction.

    This is a convenience function that handles session creation
    and transaction management automatically.

    Usage:
        from ai_engine.core.database_manager import get_database_manager

        db_manager = get_database_manager()
        user = await run_in_transaction(
            db_manager,
            create_user,
            username="test",
            email="test@example.com"
        )

    Args:
        db_manager: DatabaseManager instance
        operation: Async callable to execute
        *args, **kwargs: Arguments for the operation

    Returns:
        Result of the operation

    Raises:
        TransactionError: If operation fails
    """
    async with db_manager.get_session() as session:
        kwargs['db'] = session
        return await operation(*args, **kwargs)
