"""
CRONOS AI Engine - Database Connection Pool Manager

Production-ready database session management with connection pooling,
transaction management, circuit breaker integration, and comprehensive monitoring.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Dict, Any, Callable
from datetime import datetime, timedelta
import time

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.exc import (
    OperationalError,
    DBAPIError,
    IntegrityError,
    DatabaseError,
)
from sqlalchemy import event, text
from prometheus_client import Counter, Histogram, Gauge

from ai_engine.core.config import DatabaseConfig

logger = logging.getLogger(__name__)

# Prometheus Metrics
DB_CONNECTIONS_TOTAL = Gauge(
    "db_connections_total",
    "Total number of database connections in the pool",
)
DB_CONNECTIONS_ACTIVE = Gauge(
    "db_connections_active",
    "Number of active database connections",
)
DB_CONNECTIONS_IDLE = Gauge(
    "db_connections_idle",
    "Number of idle database connections",
)
DB_CONNECTION_ERRORS = Counter(
    "db_connection_errors_total",
    "Total number of database connection errors",
    ["error_type"],
)
DB_QUERY_DURATION = Histogram(
    "db_query_duration_seconds",
    "Database query execution time in seconds",
    ["query_type"],
)
DB_TRANSACTIONS_TOTAL = Counter(
    "db_transactions_total",
    "Total number of database transactions",
    ["status"],  # committed, rolled_back
)
DB_DEADLOCKS_TOTAL = Counter(
    "db_deadlocks_total",
    "Total number of database deadlock errors",
)
DB_CONNECTION_POOL_EXHAUSTED = Counter(
    "db_connection_pool_exhausted_total",
    "Number of times connection pool was exhausted",
)


class DatabaseError(Exception):
    """Base exception for database errors."""
    pass


class ConnectionPoolExhausted(DatabaseError):
    """Raised when connection pool is exhausted."""
    pass


class TransactionError(DatabaseError):
    """Raised when transaction fails."""
    pass


class DatabaseManager:
    """
    Production-ready database connection pool manager.

    Features:
    - Async SQLAlchemy engine with connection pooling
    - Automatic connection health checks (pool_pre_ping)
    - Connection recycling to prevent stale connections
    - Comprehensive metrics and monitoring
    - Transaction management with automatic commit/rollback
    - Graceful shutdown with connection draining
    - Circuit breaker integration ready
    """

    def __init__(self, config: DatabaseConfig, environment: str = "production"):
        """
        Initialize database manager.

        Args:
            config: Database configuration
            environment: Environment (production, staging, development)
        """
        self.config = config
        self.environment = environment
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
        self._active_sessions: Dict[str, AsyncSession] = {}
        self._session_start_times: Dict[str, float] = {}
        self._shutdown_event = asyncio.Event()

        logger.info(f"Initializing DatabaseManager for {environment} environment")

    async def initialize(self) -> None:
        """Initialize the database engine and session factory."""
        if self._initialized:
            logger.warning("DatabaseManager already initialized")
            return

        try:
            # Build database URL
            db_url = self._build_database_url()

            # Configure connection pool based on environment
            pool_config = self._get_pool_config()

            # Create async engine
            self.engine = create_async_engine(
                db_url,
                **pool_config,
                echo=self.config.echo,
                future=True,
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,  # Important for async usage
                autocommit=False,
                autoflush=False,
            )

            # Register event listeners for monitoring
            self._register_event_listeners()

            # Verify connectivity
            await self._verify_connectivity()

            self._initialized = True
            logger.info(
                f"✅ Database initialized successfully "
                f"(pool_size={self.config.pool_size}, "
                f"max_overflow={self.config.max_overflow})"
            )

        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            DB_CONNECTION_ERRORS.labels(error_type="initialization").inc()
            raise DatabaseError(f"Database initialization failed: {e}") from e

    def _build_database_url(self) -> str:
        """
        Build async PostgreSQL connection URL.

        Password is never logged and exists only in memory during connection setup.
        """
        # Use asyncpg driver for async support
        # NOTE: Password is only in memory here, not logged
        return (
            f"postgresql+asyncpg://{self.config.username}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        )

    def _get_pool_config(self) -> Dict[str, Any]:
        """Get connection pool configuration based on environment."""
        if self.environment == "test":
            # Use NullPool for testing to avoid connection reuse issues
            return {
                "poolclass": NullPool,
            }

        # Production configuration with QueuePool
        return {
            "poolclass": QueuePool,
            "pool_size": self.config.pool_size,
            "max_overflow": self.config.max_overflow,
            "pool_timeout": self.config.pool_timeout,
            "pool_recycle": self.config.pool_recycle,
            "pool_pre_ping": True,  # CRITICAL: Verify connections before use
        }

    def _register_event_listeners(self) -> None:
        """Register SQLAlchemy event listeners for monitoring."""
        if not self.engine:
            return

        @event.listens_for(self.engine.sync_engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Track connection creation."""
            logger.debug("Database connection established")
            DB_CONNECTIONS_TOTAL.inc()

        @event.listens_for(self.engine.sync_engine, "close")
        def receive_close(dbapi_conn, connection_record):
            """Track connection closure."""
            logger.debug("Database connection closed")
            DB_CONNECTIONS_TOTAL.dec()

        @event.listens_for(self.engine.sync_engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Track connection check-in (returned to pool)."""
            DB_CONNECTIONS_ACTIVE.dec()
            DB_CONNECTIONS_IDLE.inc()

        @event.listens_for(self.engine.sync_engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Track connection check-out (acquired from pool)."""
            DB_CONNECTIONS_IDLE.dec()
            DB_CONNECTIONS_ACTIVE.inc()

    async def _verify_connectivity(self) -> None:
        """Verify database connectivity."""
        if not self.engine:
            raise DatabaseError("Engine not initialized")

        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                row = result.fetchone()
                if row[0] != 1:
                    raise DatabaseError("Connectivity test failed")
            logger.info("✅ Database connectivity verified")
        except Exception as e:
            logger.error(f"❌ Database connectivity test failed: {e}")
            raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with automatic transaction management.

        Usage:
            async with db_manager.get_session() as session:
                result = await session.execute(...)
                # Automatic commit on success, rollback on exception

        Yields:
            AsyncSession: Database session

        Raises:
            DatabaseError: If session cannot be created
            ConnectionPoolExhausted: If connection pool is full
        """
        if not self._initialized:
            raise DatabaseError("DatabaseManager not initialized. Call initialize() first.")

        if self._shutdown_event.is_set():
            raise DatabaseError("DatabaseManager is shutting down")

        session: Optional[AsyncSession] = None
        session_id = str(id(asyncio.current_task()))
        start_time = time.time()

        try:
            # Create session
            session = self.session_factory()
            self._active_sessions[session_id] = session
            self._session_start_times[session_id] = start_time

            yield session

            # Commit transaction on success
            await session.commit()
            DB_TRANSACTIONS_TOTAL.labels(status="committed").inc()

        except IntegrityError as e:
            # Database constraint violation (unique, foreign key, etc.)
            if session:
                await session.rollback()
            DB_TRANSACTIONS_TOTAL.labels(status="rolled_back").inc()
            logger.warning(f"Database integrity error: {e}")
            raise TransactionError(f"Database constraint violation: {e}") from e

        except OperationalError as e:
            # Connection issues, deadlocks, etc.
            if session:
                await session.rollback()
            DB_TRANSACTIONS_TOTAL.labels(status="rolled_back").inc()

            # Check for specific error types
            if "deadlock" in str(e).lower():
                DB_DEADLOCKS_TOTAL.inc()
                logger.warning(f"Database deadlock detected: {e}")
            elif "connection" in str(e).lower():
                DB_CONNECTION_ERRORS.labels(error_type="operational").inc()
                logger.error(f"Database connection error: {e}")

            raise TransactionError(f"Database operational error: {e}") from e

        except DBAPIError as e:
            # Low-level database API errors
            if session:
                await session.rollback()
            DB_TRANSACTIONS_TOTAL.labels(status="rolled_back").inc()
            DB_CONNECTION_ERRORS.labels(error_type="dbapi").inc()
            logger.error(f"Database API error: {e}")
            raise TransactionError(f"Database API error: {e}") from e

        except Exception as e:
            # Any other exception
            if session:
                await session.rollback()
            DB_TRANSACTIONS_TOTAL.labels(status="rolled_back").inc()
            logger.error(f"Unexpected error in database session: {e}")
            raise

        finally:
            # Cleanup
            if session:
                await session.close()

            # Track session duration
            duration = time.time() - start_time
            DB_QUERY_DURATION.labels(query_type="transaction").observe(duration)

            # Warn on long-held sessions
            if duration > 30:
                logger.warning(
                    f"Long-held database session: {duration:.2f}s "
                    f"(session_id={session_id})"
                )

            # Remove from tracking
            self._active_sessions.pop(session_id, None)
            self._session_start_times.pop(session_id, None)

    async def execute_with_retry(
        self,
        operation: Callable,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute a database operation with automatic retry on transient errors.

        Args:
            operation: Async function to execute
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
            *args, **kwargs: Arguments to pass to operation

        Returns:
            Result of the operation

        Raises:
            TransactionError: If all retries are exhausted
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                return await operation(*args, **kwargs)
            except (OperationalError, DBAPIError) as e:
                last_exception = e

                # Check if error is retryable
                error_str = str(e).lower()
                is_retryable = any(
                    keyword in error_str
                    for keyword in ["deadlock", "timeout", "connection", "lock"]
                )

                if not is_retryable or attempt == max_retries - 1:
                    raise TransactionError(
                        f"Database operation failed after {attempt + 1} attempts: {e}"
                    ) from e

                # Exponential backoff
                delay = retry_delay * (2 ** attempt)
                logger.warning(
                    f"Database operation failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)

        # Should never reach here, but just in case
        raise TransactionError(
            f"Database operation failed after {max_retries} attempts"
        ) from last_exception

    async def get_pool_status(self) -> Dict[str, Any]:
        """
        Get current connection pool status.

        Returns:
            Dict with pool metrics
        """
        if not self.engine:
            return {"initialized": False}

        pool = self.engine.pool

        return {
            "initialized": True,
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total_connections": pool.size() + pool.overflow(),
            "active_sessions": len(self._active_sessions),
        }

    async def wait_for_active_connections(self, timeout: int = 30) -> None:
        """
        Wait for all active database connections to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Raises:
            TimeoutError: If timeout is exceeded
        """
        logger.info(f"Waiting for {len(self._active_sessions)} active sessions to complete...")

        start_time = time.time()
        while self._active_sessions:
            if time.time() - start_time > timeout:
                logger.error(
                    f"Timeout waiting for active sessions. "
                    f"Remaining: {len(self._active_sessions)}"
                )
                raise TimeoutError(
                    f"Timeout waiting for active database sessions. "
                    f"Remaining: {len(self._active_sessions)}"
                )

            # Check for stuck sessions
            now = time.time()
            for session_id, start_time in list(self._session_start_times.items()):
                duration = now - start_time
                if duration > 60:
                    logger.warning(
                        f"Long-running session detected: {session_id} "
                        f"({duration:.2f}s)"
                    )

            await asyncio.sleep(0.5)

        logger.info("✅ All active sessions completed")

    async def dispose(self) -> None:
        """
        Dispose of the database engine and close all connections.
        Should be called during application shutdown.
        """
        if not self._initialized:
            logger.warning("DatabaseManager not initialized, nothing to dispose")
            return

        logger.info("Shutting down DatabaseManager...")
        self._shutdown_event.set()

        try:
            # Wait for active sessions to complete
            await self.wait_for_active_connections(timeout=30)
        except TimeoutError:
            logger.warning("Some sessions did not complete in time, forcing shutdown")

        # Dispose engine
        if self.engine:
            await self.engine.dispose()
            logger.info("✅ Database engine disposed")

        self._initialized = False
        self.engine = None
        self.session_factory = None
        logger.info("✅ DatabaseManager shutdown complete")


# Global database manager instance (initialized during app startup)
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance.

    Returns:
        DatabaseManager: Global database manager

    Raises:
        RuntimeError: If database manager is not initialized
    """
    global _db_manager
    if _db_manager is None:
        raise RuntimeError(
            "Database manager not initialized. "
            "Call initialize_database_manager() during app startup."
        )
    return _db_manager


async def initialize_database_manager(
    config: DatabaseConfig,
    environment: str = "production"
) -> DatabaseManager:
    """
    Initialize the global database manager.

    Args:
        config: Database configuration
        environment: Environment name

    Returns:
        DatabaseManager: Initialized database manager
    """
    global _db_manager

    if _db_manager is not None:
        logger.warning("Database manager already initialized")
        return _db_manager

    _db_manager = DatabaseManager(config, environment)
    await _db_manager.initialize()

    return _db_manager


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Usage:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()

    Yields:
        AsyncSession: Database session
    """
    db_manager = get_database_manager()
    async with db_manager.get_session() as session:
        yield session
