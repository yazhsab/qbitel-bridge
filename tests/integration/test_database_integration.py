"""
QBITEL - Database Integration Tests

This module tests database connectivity, operations, and data persistence
for the Legacy System Whisperer component.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Test: Database Connection
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.asyncio
class TestDatabaseConnection:
    """Tests for database connectivity."""

    async def test_database_connection_pool(self, test_database: Dict[str, Any]):
        """Test database connection pool initialization."""
        if test_database.get("mock"):
            pytest.skip(f"Database not available: {test_database.get('error', 'mock mode')}")

        pool = test_database["pool"]
        assert pool is not None

        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1

    async def test_database_health_check(self, test_database: Dict[str, Any]):
        """Test database health check query."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]

        async with pool.acquire() as conn:
            # Test basic connectivity
            version = await conn.fetchval("SELECT version()")
            assert "PostgreSQL" in version

            # Test TimescaleDB extension (if available)
            try:
                ts_version = await conn.fetchval(
                    "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
                )
                logger.info(f"TimescaleDB version: {ts_version}")
            except Exception:
                logger.warning("TimescaleDB extension not installed")

    async def test_connection_pool_limits(self, test_database: Dict[str, Any]):
        """Test connection pool handles concurrent connections."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]

        # Acquire multiple connections concurrently
        async def use_connection(i: int) -> int:
            async with pool.acquire() as conn:
                await asyncio.sleep(0.1)  # Simulate work
                result = await conn.fetchval(f"SELECT {i}")
                return result

        tasks = [use_connection(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert results == [0, 1, 2, 3, 4]


# =============================================================================
# Test: Legacy System CRUD Operations
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.asyncio
class TestLegacySystemCRUD:
    """Tests for Legacy System CRUD operations."""

    async def test_create_legacy_system(self, test_database: Dict[str, Any]):
        """Test creating a legacy system record."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]

        async with pool.acquire() as conn:
            # Insert a legacy system
            system_id = "TEST-SYS-001"
            result = await conn.fetchrow("""
                INSERT INTO test_schema.legacy_systems
                (system_id, name, system_type, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id, system_id, created_at
            """, system_id, "Test Mainframe", "mainframe", json.dumps({"version": "1.0"}))

            assert result["system_id"] == system_id
            assert result["created_at"] is not None

            # Cleanup
            await conn.execute(
                "DELETE FROM test_schema.legacy_systems WHERE system_id = $1",
                system_id
            )

    async def test_read_legacy_system(self, test_database: Dict[str, Any]):
        """Test reading a legacy system record."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]

        async with pool.acquire() as conn:
            # Insert test data
            system_id = "TEST-SYS-002"
            await conn.execute("""
                INSERT INTO test_schema.legacy_systems
                (system_id, name, system_type)
                VALUES ($1, $2, $3)
            """, system_id, "Test System", "as400")

            # Read it back
            result = await conn.fetchrow("""
                SELECT * FROM test_schema.legacy_systems
                WHERE system_id = $1
            """, system_id)

            assert result["name"] == "Test System"
            assert result["system_type"] == "as400"

            # Cleanup
            await conn.execute(
                "DELETE FROM test_schema.legacy_systems WHERE system_id = $1",
                system_id
            )

    async def test_update_legacy_system(self, test_database: Dict[str, Any]):
        """Test updating a legacy system record."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]

        async with pool.acquire() as conn:
            # Insert test data
            system_id = "TEST-SYS-003"
            await conn.execute("""
                INSERT INTO test_schema.legacy_systems
                (system_id, name, system_type)
                VALUES ($1, $2, $3)
            """, system_id, "Original Name", "mainframe")

            # Update the record
            await conn.execute("""
                UPDATE test_schema.legacy_systems
                SET name = $2, updated_at = NOW()
                WHERE system_id = $1
            """, system_id, "Updated Name")

            # Verify update
            result = await conn.fetchrow("""
                SELECT name, updated_at FROM test_schema.legacy_systems
                WHERE system_id = $1
            """, system_id)

            assert result["name"] == "Updated Name"

            # Cleanup
            await conn.execute(
                "DELETE FROM test_schema.legacy_systems WHERE system_id = $1",
                system_id
            )

    async def test_delete_legacy_system(self, test_database: Dict[str, Any]):
        """Test deleting a legacy system record."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]

        async with pool.acquire() as conn:
            # Insert test data
            system_id = "TEST-SYS-004"
            await conn.execute("""
                INSERT INTO test_schema.legacy_systems
                (system_id, name, system_type)
                VALUES ($1, $2, $3)
            """, system_id, "To Delete", "mainframe")

            # Delete the record
            await conn.execute("""
                DELETE FROM test_schema.legacy_systems
                WHERE system_id = $1
            """, system_id)

            # Verify deletion
            result = await conn.fetchrow("""
                SELECT * FROM test_schema.legacy_systems
                WHERE system_id = $1
            """, system_id)

            assert result is None


# =============================================================================
# Test: Analysis Results Storage
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.asyncio
class TestAnalysisResultsStorage:
    """Tests for storing and retrieving analysis results."""

    async def test_store_analysis_result(self, test_database: Dict[str, Any]):
        """Test storing an analysis result."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]

        async with pool.acquire() as conn:
            # Create a system first
            system_id = "TEST-ANALYSIS-001"
            await conn.execute("""
                INSERT INTO test_schema.legacy_systems
                (system_id, name, system_type)
                VALUES ($1, $2, $3)
                ON CONFLICT (system_id) DO NOTHING
            """, system_id, "Analysis Test System", "mainframe")

            # Store analysis result
            analysis_result = {
                "complexity_score": 0.75,
                "patterns_found": ["indexed-file", "comp-3"],
                "recommendations": ["Migrate to REST API"]
            }

            result = await conn.fetchrow("""
                INSERT INTO test_schema.analysis_results
                (system_id, analysis_type, result, confidence)
                VALUES ($1, $2, $3, $4)
                RETURNING id, created_at
            """, system_id, "cobol_analysis", json.dumps(analysis_result), 0.95)

            assert result["id"] is not None
            assert result["created_at"] is not None

            # Cleanup
            await conn.execute(
                "DELETE FROM test_schema.analysis_results WHERE system_id = $1",
                system_id
            )
            await conn.execute(
                "DELETE FROM test_schema.legacy_systems WHERE system_id = $1",
                system_id
            )

    async def test_query_analysis_history(self, test_database: Dict[str, Any]):
        """Test querying analysis history for a system."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]

        async with pool.acquire() as conn:
            # Create a system
            system_id = "TEST-HISTORY-001"
            await conn.execute("""
                INSERT INTO test_schema.legacy_systems
                (system_id, name, system_type)
                VALUES ($1, $2, $3)
                ON CONFLICT (system_id) DO NOTHING
            """, system_id, "History Test System", "mainframe")

            # Insert multiple analysis results
            for i in range(5):
                await conn.execute("""
                    INSERT INTO test_schema.analysis_results
                    (system_id, analysis_type, result, confidence)
                    VALUES ($1, $2, $3, $4)
                """, system_id, f"analysis_{i}", json.dumps({"iteration": i}), 0.8 + i * 0.02)

            # Query history
            results = await conn.fetch("""
                SELECT * FROM test_schema.analysis_results
                WHERE system_id = $1
                ORDER BY created_at DESC
            """, system_id)

            assert len(results) == 5

            # Cleanup
            await conn.execute(
                "DELETE FROM test_schema.analysis_results WHERE system_id = $1",
                system_id
            )
            await conn.execute(
                "DELETE FROM test_schema.legacy_systems WHERE system_id = $1",
                system_id
            )


# =============================================================================
# Test: TimescaleDB Features
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.asyncio
class TestTimescaleDBFeatures:
    """Tests for TimescaleDB-specific features."""

    async def test_timescale_hypertable_creation(self, test_database: Dict[str, Any]):
        """Test creating a TimescaleDB hypertable."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]

        async with pool.acquire() as conn:
            # Check if TimescaleDB is available
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
            except Exception:
                pytest.skip("TimescaleDB not available")

            # Create a metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_schema.system_metrics (
                    time TIMESTAMPTZ NOT NULL,
                    system_id VARCHAR(255) NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DOUBLE PRECISION,
                    labels JSONB DEFAULT '{}'
                )
            """)

            # Convert to hypertable
            try:
                await conn.execute("""
                    SELECT create_hypertable(
                        'test_schema.system_metrics',
                        'time',
                        if_not_exists => TRUE
                    )
                """)

                # Insert some test data
                for i in range(100):
                    await conn.execute("""
                        INSERT INTO test_schema.system_metrics
                        (time, system_id, metric_name, metric_value)
                        VALUES ($1, $2, $3, $4)
                    """,
                        datetime.now() - timedelta(hours=i),
                        "TEST-SYS",
                        "cpu_usage",
                        50.0 + i * 0.5
                    )

                # Query with time bucket (TimescaleDB feature)
                results = await conn.fetch("""
                    SELECT
                        time_bucket('1 hour', time) AS bucket,
                        AVG(metric_value) AS avg_value
                    FROM test_schema.system_metrics
                    WHERE system_id = 'TEST-SYS'
                    GROUP BY bucket
                    ORDER BY bucket DESC
                    LIMIT 10
                """)

                assert len(results) > 0

            finally:
                # Cleanup
                await conn.execute("DROP TABLE IF EXISTS test_schema.system_metrics")


# =============================================================================
# Test: Database Transactions
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.asyncio
class TestDatabaseTransactions:
    """Tests for database transaction handling."""

    async def test_transaction_commit(self, test_database: Dict[str, Any]):
        """Test transaction commit."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]

        async with pool.acquire() as conn:
            async with conn.transaction():
                system_id = "TRANS-TEST-001"
                await conn.execute("""
                    INSERT INTO test_schema.legacy_systems
                    (system_id, name, system_type)
                    VALUES ($1, $2, $3)
                """, system_id, "Transaction Test", "mainframe")

            # Verify data persisted after transaction
            result = await conn.fetchrow("""
                SELECT * FROM test_schema.legacy_systems
                WHERE system_id = $1
            """, system_id)

            assert result is not None
            assert result["name"] == "Transaction Test"

            # Cleanup
            await conn.execute(
                "DELETE FROM test_schema.legacy_systems WHERE system_id = $1",
                system_id
            )

    async def test_transaction_rollback(self, test_database: Dict[str, Any]):
        """Test transaction rollback on error."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]

        async with pool.acquire() as conn:
            system_id = "TRANS-ROLLBACK-001"

            try:
                async with conn.transaction():
                    await conn.execute("""
                        INSERT INTO test_schema.legacy_systems
                        (system_id, name, system_type)
                        VALUES ($1, $2, $3)
                    """, system_id, "Rollback Test", "mainframe")

                    # Force an error
                    raise ValueError("Simulated error")
            except ValueError:
                pass

            # Verify data was not persisted
            result = await conn.fetchrow("""
                SELECT * FROM test_schema.legacy_systems
                WHERE system_id = $1
            """, system_id)

            assert result is None


# =============================================================================
# Test: Database Performance
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.slow
@pytest.mark.asyncio
class TestDatabasePerformance:
    """Performance tests for database operations."""

    async def test_bulk_insert_performance(self, test_database: Dict[str, Any]):
        """Test bulk insert performance."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]
        num_records = 1000

        async with pool.acquire() as conn:
            # Prepare test data
            records = [
                (f"PERF-{i:05d}", f"System {i}", "mainframe")
                for i in range(num_records)
            ]

            # Bulk insert
            start_time = asyncio.get_event_loop().time()

            await conn.executemany("""
                INSERT INTO test_schema.legacy_systems
                (system_id, name, system_type)
                VALUES ($1, $2, $3)
            """, records)

            elapsed = asyncio.get_event_loop().time() - start_time

            logger.info(f"Inserted {num_records} records in {elapsed:.2f} seconds")
            logger.info(f"Rate: {num_records / elapsed:.2f} records/second")

            # Verify count
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM test_schema.legacy_systems
                WHERE system_id LIKE 'PERF-%'
            """)

            assert count == num_records

            # Cleanup
            await conn.execute("""
                DELETE FROM test_schema.legacy_systems
                WHERE system_id LIKE 'PERF-%'
            """)

    async def test_concurrent_read_performance(self, test_database: Dict[str, Any]):
        """Test concurrent read performance."""
        if test_database.get("mock"):
            pytest.skip("Database not available")

        pool = test_database["pool"]
        num_concurrent = 50

        async with pool.acquire() as conn:
            # Create test record
            system_id = "CONCURRENT-READ-TEST"
            await conn.execute("""
                INSERT INTO test_schema.legacy_systems
                (system_id, name, system_type)
                VALUES ($1, $2, $3)
                ON CONFLICT (system_id) DO NOTHING
            """, system_id, "Concurrent Test", "mainframe")

        async def read_system():
            async with pool.acquire() as conn:
                return await conn.fetchrow("""
                    SELECT * FROM test_schema.legacy_systems
                    WHERE system_id = $1
                """, system_id)

        start_time = asyncio.get_event_loop().time()
        tasks = [read_system() for _ in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        elapsed = asyncio.get_event_loop().time() - start_time

        logger.info(f"Completed {num_concurrent} concurrent reads in {elapsed:.2f} seconds")

        assert all(r["system_id"] == system_id for r in results)

        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM test_schema.legacy_systems WHERE system_id = $1",
                system_id
            )
