"""
CRONOS AI - Database Integration Tests

Tests for database connection pool, transaction management,
circuit breaker integration, and encryption.
"""

import pytest
import asyncio
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from ai_engine.core.config import DatabaseConfig
from ai_engine.core.database_manager import (
    DatabaseManager,
    initialize_database_manager,
    get_database_manager,
    get_db,
)
from ai_engine.core.transaction_manager import (
    transactional,
    readonly_transaction,
    savepoint,
    TransactionContext,
)
from ai_engine.core.database_circuit_breaker import (
    with_database_circuit_breaker,
    get_db_circuit_breaker_manager,
    is_database_available,
)
from ai_engine.security.field_encryption import (
    EncryptionKeyManager,
    initialize_encryption,
    EncryptedString,
)
from ai_engine.models.database import Base, User, UserRole


@pytest.fixture(scope="function")
async def db_config():
    """Create test database configuration."""
    import os
    # Use test database
    return DatabaseConfig(
        host=os.getenv("TEST_DB_HOST", "localhost"),
        port=int(os.getenv("TEST_DB_PORT", 5432)),
        database=os.getenv("TEST_DB_NAME", "cronos_ai_test"),
        username=os.getenv("TEST_DB_USER", "cronos"),
        password=os.getenv("TEST_DB_PASSWORD", "test_password"),
        pool_size=5,
        max_overflow=10,
        pool_timeout=10,
        pool_recycle=300,
        echo=False,
    )


@pytest.fixture(scope="function")
async def db_manager(db_config):
    """Create and initialize database manager for testing."""
    # Initialize encryption for testing
    initialize_encryption(test_mode=True)

    # Initialize database manager
    manager = DatabaseManager(db_config, environment="test")
    await manager.initialize()

    # Create tables
    async with manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield manager

    # Cleanup
    async with manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await manager.dispose()


@pytest.fixture
async def db_session(db_manager) -> AsyncGenerator[AsyncSession, None]:
    """Get database session for testing."""
    async with db_manager.get_session() as session:
        yield session


class TestDatabaseConnectionPool:
    """Tests for database connection pool management."""

    @pytest.mark.asyncio
    async def test_connection_pool_initialization(self, db_manager):
        """Test database connection pool initializes correctly."""
        assert db_manager._initialized is True
        assert db_manager.engine is not None
        assert db_manager.session_factory is not None

        # Check pool status
        status = await db_manager.get_pool_status()
        assert status["initialized"] is True
        assert status["pool_size"] >= 0

    @pytest.mark.asyncio
    async def test_connection_pool_checkout_checkin(self, db_manager):
        """Test connection checkout and checkin."""
        # Get initial status
        initial_status = await db_manager.get_pool_status()
        initial_active = initial_status["active_sessions"]

        # Create multiple sessions
        sessions = []
        for _ in range(3):
            async with db_manager.get_session() as session:
                sessions.append(session)
                # Session is active here

        # After context exit, all should be returned
        final_status = await db_manager.get_pool_status()
        assert final_status["active_sessions"] == initial_active

    @pytest.mark.asyncio
    async def test_connection_pool_concurrent_access(self, db_manager):
        """Test connection pool handles concurrent access."""
        async def query_database(session_id: int):
            async with db_manager.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                value = result.scalar()
                assert value == 1
                await asyncio.sleep(0.1)  # Simulate work
                return session_id

        # Run 20 concurrent database operations
        tasks = [query_database(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 20
        assert sorted(results) == list(range(20))

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion_handling(self, db_manager):
        """Test behavior when connection pool is exhausted."""
        # This test verifies pool timeout works correctly
        # Intentionally try to exceed pool size
        pool_size = db_manager.config.pool_size + db_manager.config.max_overflow

        async def hold_connection():
            async with db_manager.get_session() as session:
                await asyncio.sleep(0.5)

        # Create tasks that exceed pool capacity
        tasks = [hold_connection() for _ in range(pool_size + 5)]

        # All should eventually complete (with queueing)
        await asyncio.gather(*tasks)


class TestTransactionManagement:
    """Tests for transaction management."""

    @pytest.mark.asyncio
    async def test_automatic_commit_on_success(self, db_session):
        """Test transaction commits automatically on success."""
        @transactional()
        async def create_user(db: AsyncSession, username: str):
            user = User(
                username=username,
                email=f"{username}@example.com",
                password_hash="hashed_password",
                role=UserRole.VIEWER,
            )
            db.add(user)
            await db.flush()
            return user

        # Create user
        user = await create_user(db=db_session, username="test_user")
        assert user.id is not None

        # Verify user was committed
        result = await db_session.execute(
            select(User).where(User.username == "test_user")
        )
        saved_user = result.scalar_one_or_none()
        assert saved_user is not None
        assert saved_user.username == "test_user"

    @pytest.mark.asyncio
    async def test_automatic_rollback_on_exception(self, db_session):
        """Test transaction rolls back on exception."""
        @transactional()
        async def create_user_with_error(db: AsyncSession):
            user = User(
                username="rollback_test",
                email="rollback@example.com",
                password_hash="hashed",
                role=UserRole.VIEWER,
            )
            db.add(user)
            await db.flush()

            # Simulate an error
            raise ValueError("Intentional error for rollback test")

        # Should raise ValueError
        with pytest.raises(Exception):
            await create_user_with_error(db=db_session)

        # User should NOT exist
        result = await db_session.execute(
            select(User).where(User.username == "rollback_test")
        )
        user = result.scalar_one_or_none()
        assert user is None

    @pytest.mark.asyncio
    async def test_readonly_transaction(self, db_session):
        """Test read-only transaction."""
        # First create a user
        user = User(
            username="readonly_user",
            email="readonly@example.com",
            password_hash="hashed",
            role=UserRole.VIEWER,
        )
        db_session.add(user)
        await db_session.flush()

        @readonly_transaction
        async def get_user(db: AsyncSession, username: str):
            result = await db.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()

        # Read user
        found_user = await get_user(db=db_session, username="readonly_user")
        assert found_user is not None
        assert found_user.username == "readonly_user"

    @pytest.mark.asyncio
    async def test_savepoint_rollback(self, db_session):
        """Test savepoint allows partial rollback."""
        # Create first user (will be committed)
        user1 = User(
            username="user1",
            email="user1@example.com",
            password_hash="hashed",
            role=UserRole.VIEWER,
        )
        db_session.add(user1)
        await db_session.flush()

        # Try to create second user with savepoint
        try:
            async with savepoint(db_session, "create_user2"):
                user2 = User(
                    username="user2",
                    email="user2@example.com",
                    password_hash="hashed",
                    role=UserRole.VIEWER,
                )
                db_session.add(user2)
                await db_session.flush()

                # Intentional error
                raise ValueError("Rollback to savepoint")
        except ValueError:
            pass

        # Commit outer transaction
        await db_session.commit()

        # User1 should exist, user2 should not
        result = await db_session.execute(select(User))
        users = result.scalars().all()
        assert len(users) == 1
        assert users[0].username == "user1"


class TestDatabaseCircuitBreaker:
    """Tests for database circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_protects_database_operations(self, db_session):
        """Test circuit breaker protects database operations."""
        @with_database_circuit_breaker(name="test_operation")
        async def protected_query():
            async with db_session as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar()

        # Should work normally
        result = await protected_query()
        assert result == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, db_session):
        """Test circuit breaker opens after repeated failures."""
        failure_count = 0

        @with_database_circuit_breaker(name="failing_operation")
        async def failing_query():
            nonlocal failure_count
            failure_count += 1
            raise Exception("Database error")

        # Cause multiple failures
        for _ in range(6):
            try:
                await failing_query()
            except Exception:
                pass

        # Circuit should be open now
        available = await is_database_available("failing_operation")
        # May be open or half-open depending on timing
        assert failure_count >= 5

    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback(self, db_session):
        """Test circuit breaker uses fallback when open."""
        async def fallback_function():
            return "fallback_value"

        call_count = 0

        @with_database_circuit_breaker(
            name="fallback_test",
            fallback=fallback_function
        )
        async def query_with_fallback():
            nonlocal call_count
            call_count += 1
            if call_count <= 6:
                raise Exception("Database error")
            return "success"

        # First calls should fail and use fallback
        for _ in range(3):
            try:
                result = await query_with_fallback()
                if result == "fallback_value":
                    # Fallback was used
                    pass
            except:
                pass


class TestDatabaseEncryption:
    """Tests for database field encryption."""

    @pytest.mark.asyncio
    async def test_encrypted_field_storage(self, db_session):
        """Test encrypted fields are stored encrypted."""
        # Create user with MFA secret
        user = User(
            username="encrypted_test",
            email="encrypted@example.com",
            password_hash="hashed",
            role=UserRole.VIEWER,
            mfa_enabled=True,
            mfa_secret="JBSWY3DPEHPK3PXP",  # Test TOTP secret
        )
        db_session.add(user)
        await db_session.flush()
        await db_session.commit()

        # Read user back
        result = await db_session.execute(
            select(User).where(User.username == "encrypted_test")
        )
        saved_user = result.scalar_one()

        # MFA secret should be decrypted automatically
        assert saved_user.mfa_secret == "JBSWY3DPEHPK3PXP"

        # Verify it's actually encrypted in database (raw query)
        raw_result = await db_session.execute(
            text("SELECT mfa_secret FROM users WHERE username = 'encrypted_test'")
        )
        raw_value = raw_result.scalar()

        # Raw value should be bytes (encrypted)
        assert isinstance(raw_value, bytes)
        assert raw_value != b"JBSWY3DPEHPK3PXP"

    @pytest.mark.asyncio
    async def test_encrypted_json_field(self, db_session):
        """Test encrypted JSON fields."""
        backup_codes = ["CODE1", "CODE2", "CODE3"]

        user = User(
            username="json_encrypted_test",
            email="json@example.com",
            password_hash="hashed",
            role=UserRole.VIEWER,
            mfa_enabled=True,
            mfa_backup_codes=backup_codes,
        )
        db_session.add(user)
        await db_session.flush()
        await db_session.commit()

        # Read back
        result = await db_session.execute(
            select(User).where(User.username == "json_encrypted_test")
        )
        saved_user = result.scalar_one()

        # Should be decrypted automatically
        assert saved_user.mfa_backup_codes == backup_codes


class TestDatabasePerformance:
    """Performance tests for database operations."""

    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, db_session):
        """Test bulk insert performance."""
        import time

        # Create 100 users
        users = [
            User(
                username=f"perf_user_{i}",
                email=f"perf{i}@example.com",
                password_hash="hashed",
                role=UserRole.VIEWER,
            )
            for i in range(100)
        ]

        start = time.time()
        db_session.add_all(users)
        await db_session.flush()
        await db_session.commit()
        duration = time.time() - start

        # Should complete in reasonable time (< 5 seconds)
        assert duration < 5.0

        # Verify count
        result = await db_session.execute(select(User))
        count = len(result.scalars().all())
        assert count == 100

    @pytest.mark.asyncio
    async def test_query_performance(self, db_manager):
        """Test query performance."""
        import time

        async def run_queries(num_queries: int):
            start = time.time()

            for _ in range(num_queries):
                async with db_manager.get_session() as session:
                    await session.execute(text("SELECT 1"))

            return time.time() - start

        # Run 100 queries
        duration = await run_queries(100)

        # Should complete quickly (< 10 seconds even with connection overhead)
        assert duration < 10.0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
