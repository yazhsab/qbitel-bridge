"""
QBITEL - Integration Test Configuration

This module provides fixtures and configuration for integration testing
of the Legacy System Whisperer and related components.
"""

import os
import asyncio
import pytest
import logging
from typing import AsyncGenerator, Generator, Dict, Any
from unittest.mock import AsyncMock, MagicMock

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set test environment
os.environ.setdefault("QBITEL_AI_ENVIRONMENT", "testing")
os.environ.setdefault("QBITEL_AI_DEBUG", "true")
os.environ.setdefault("QBITEL_AI_LOG_LEVEL", "DEBUG")


# =============================================================================
# Event Loop Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def database_url() -> str:
    """Get database URL for testing."""
    return os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql://qbitel:qbitel123@localhost:5432/qbitel_test"
    )


@pytest.fixture(scope="session")
async def test_database(database_url: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Set up test database connection.

    This fixture:
    1. Creates a connection pool
    2. Runs migrations
    3. Yields the connection
    4. Cleans up after tests
    """
    try:
        import asyncpg

        # Create connection pool
        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=10,
            command_timeout=60
        )

        # Run any needed setup
        async with pool.acquire() as conn:
            # Create test schema
            await conn.execute("""
                CREATE SCHEMA IF NOT EXISTS test_schema;
            """)

            # Create necessary tables for testing
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_schema.legacy_systems (
                    id SERIAL PRIMARY KEY,
                    system_id VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    system_type VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_schema.analysis_results (
                    id SERIAL PRIMARY KEY,
                    system_id VARCHAR(255) REFERENCES test_schema.legacy_systems(system_id),
                    analysis_type VARCHAR(50) NOT NULL,
                    result JSONB NOT NULL,
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

        yield {"pool": pool, "url": database_url}

        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute("DROP SCHEMA IF EXISTS test_schema CASCADE;")

        await pool.close()

    except ImportError:
        logger.warning("asyncpg not installed, using mock database")
        yield {"pool": None, "url": database_url, "mock": True}
    except Exception as e:
        logger.warning(f"Could not connect to database: {e}, using mock")
        yield {"pool": None, "url": database_url, "mock": True, "error": str(e)}


# =============================================================================
# Redis Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def redis_url() -> str:
    """Get Redis URL for testing."""
    return os.environ.get(
        "TEST_REDIS_URL",
        "redis://localhost:6379/15"  # Use DB 15 for testing
    )


@pytest.fixture(scope="session")
async def test_redis(redis_url: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Set up test Redis connection."""
    try:
        import redis.asyncio as redis

        client = redis.from_url(redis_url, decode_responses=True)
        await client.ping()

        # Clear test database
        await client.flushdb()

        yield {"client": client, "url": redis_url}

        # Cleanup
        await client.flushdb()
        await client.close()

    except ImportError:
        logger.warning("redis not installed, using mock")
        yield {"client": None, "url": redis_url, "mock": True}
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}, using mock")
        yield {"client": None, "url": redis_url, "mock": True, "error": str(e)}


# =============================================================================
# LLM Service Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """Standard mock LLM response."""
    return {
        "content": "This is a mock LLM response for testing.",
        "model": "mock-model",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


@pytest.fixture
def mock_llm_service(mock_llm_response: Dict[str, Any]) -> MagicMock:
    """Create a mock LLM service for testing without real API calls."""
    service = MagicMock()
    service.generate = AsyncMock(return_value=mock_llm_response)
    service.is_healthy = AsyncMock(return_value=True)
    service.get_provider_status = MagicMock(return_value={
        "openai": {"healthy": True},
        "anthropic": {"healthy": True},
        "ollama": {"healthy": False}
    })
    return service


@pytest.fixture
async def real_llm_service() -> AsyncGenerator[Any, None]:
    """
    Create a real LLM service for integration testing.

    Only use this fixture when you have valid API keys configured.
    """
    try:
        from ai_engine.llm.unified_llm_service import UnifiedLLMService

        service = UnifiedLLMService()
        await service.initialize()

        yield service

        await service.shutdown()

    except ImportError:
        logger.warning("UnifiedLLMService not available, yielding None")
        yield None
    except Exception as e:
        logger.warning(f"Could not initialize LLM service: {e}")
        yield None


# =============================================================================
# Legacy Whisperer Fixtures
# =============================================================================

@pytest.fixture
async def legacy_whisperer(mock_llm_service: MagicMock) -> AsyncGenerator[Any, None]:
    """Create a Legacy Whisperer instance for testing."""
    try:
        from ai_engine.llm.legacy_whisperer import LegacySystemWhisperer

        whisperer = LegacySystemWhisperer(llm_service=mock_llm_service)

        yield whisperer

    except ImportError:
        logger.warning("LegacySystemWhisperer not available")
        yield None


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_cobol_code() -> str:
    """Sample COBOL code for testing."""
    return """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. CUSTMAST.
       AUTHOR. QBITEL-TEST.
       DATE-WRITTEN. 2024-01-15.
      *
      * CUSTOMER MASTER FILE MAINTENANCE PROGRAM
      *
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUSTOMER-FILE
               ASSIGN TO 'CUSTMAST'
               ORGANIZATION IS INDEXED
               ACCESS MODE IS DYNAMIC
               RECORD KEY IS CUST-ID
               FILE STATUS IS WS-FILE-STATUS.
      *
       DATA DIVISION.
       FILE SECTION.
       FD CUSTOMER-FILE.
       01 CUSTOMER-RECORD.
           05 CUST-ID              PIC 9(8).
           05 CUST-NAME            PIC X(30).
           05 CUST-ADDRESS         PIC X(50).
           05 CUST-BALANCE         PIC S9(9)V99 COMP-3.
           05 CUST-CREDIT-LIMIT    PIC S9(9)V99 COMP-3.
           05 CUST-STATUS          PIC X(1).
               88 CUST-ACTIVE      VALUE 'A'.
               88 CUST-INACTIVE    VALUE 'I'.
               88 CUST-SUSPENDED   VALUE 'S'.
      *
       WORKING-STORAGE SECTION.
       01 WS-FILE-STATUS          PIC XX.
       01 WS-EOF                  PIC 9 VALUE 0.
           88 END-OF-FILE         VALUE 1.
      *
       PROCEDURE DIVISION.
       MAIN-LOGIC.
           PERFORM OPEN-FILES
           PERFORM PROCESS-RECORDS UNTIL END-OF-FILE
           PERFORM CLOSE-FILES
           STOP RUN.
      *
       OPEN-FILES.
           OPEN I-O CUSTOMER-FILE.
           IF WS-FILE-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING FILE: ' WS-FILE-STATUS
               STOP RUN
           END-IF.
      *
       PROCESS-RECORDS.
           READ CUSTOMER-FILE NEXT
               AT END SET END-OF-FILE TO TRUE
               NOT AT END
                   PERFORM UPDATE-CUSTOMER
           END-READ.
      *
       UPDATE-CUSTOMER.
           IF CUST-BALANCE > CUST-CREDIT-LIMIT
               SET CUST-SUSPENDED TO TRUE
               REWRITE CUSTOMER-RECORD
           END-IF.
      *
       CLOSE-FILES.
           CLOSE CUSTOMER-FILE.
    """


@pytest.fixture
def sample_protocol_traffic() -> list:
    """Sample protocol traffic for testing."""
    return [
        bytes.fromhex("00 1A 00 00 00 01 C3 E4 E2 E3 F0 F0 F0 F1 00 00 00 64".replace(" ", "")),
        bytes.fromhex("00 1A 00 00 00 02 C3 E4 E2 E3 F0 F0 F0 F2 00 00 00 C8".replace(" ", "")),
        bytes.fromhex("00 1A 00 00 00 03 C3 E4 E2 E3 F0 F0 F0 F3 00 00 01 2C".replace(" ", "")),
    ]


@pytest.fixture
def sample_system_metrics() -> Dict[str, Any]:
    """Sample system metrics for testing."""
    return {
        "system_id": "MAINFRAME-001",
        "timestamp": "2024-01-15T10:30:00Z",
        "cpu_utilization": 75.5,
        "memory_usage": 82.3,
        "disk_io_read": 1500,
        "disk_io_write": 800,
        "transaction_count": 50000,
        "response_time_ms": 45.2,
        "error_rate": 0.001,
        "active_connections": 250
    }


# =============================================================================
# API Client Fixtures
# =============================================================================

@pytest.fixture
def api_base_url() -> str:
    """Base URL for API testing."""
    return os.environ.get("TEST_API_URL", "http://localhost:8000")


@pytest.fixture
async def api_client(api_base_url: str) -> AsyncGenerator[Any, None]:
    """Create an async HTTP client for API testing."""
    try:
        import httpx

        async with httpx.AsyncClient(
            base_url=api_base_url,
            timeout=30.0,
            headers={"Content-Type": "application/json"}
        ) as client:
            yield client

    except ImportError:
        logger.warning("httpx not installed, using mock client")
        mock_client = MagicMock()
        mock_client.get = AsyncMock()
        mock_client.post = AsyncMock()
        yield mock_client


# =============================================================================
# Test Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "requires_db: mark test as requiring database"
    )
    config.addinivalue_line(
        "markers", "requires_redis: mark test as requiring redis"
    )
    config.addinivalue_line(
        "markers", "requires_llm: mark test as requiring LLM API"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "uc1: mark test as UC1 (Legacy Mainframe Modernization) flow"
    )


# =============================================================================
# UC1 Protocol Sample Fixtures
# =============================================================================

@pytest.fixture
def sample_tn3270_data() -> bytes:
    """Sample TN3270 terminal data stream."""
    return bytes([
        0xF5, 0xC3,  # Write command + WCC
        0x11, 0x40, 0x40,  # SBA Row 1, Col 1
        0x1D, 0xF0,  # SF unprotected
        0xC3, 0xE4, 0xE2, 0xE3,  # "CUST" in EBCDIC
        0xD6, 0xD4, 0xC5, 0xD9,  # "OMER" in EBCDIC
    ])


@pytest.fixture
def sample_mq_series_data() -> bytes:
    """Sample IBM MQ Series message header."""
    return bytes([
        0x4D, 0x44, 0x20, 0x20,  # StrucId: "MD  "
        0x00, 0x00, 0x00, 0x02,  # Version: 2
        0x00, 0x00, 0x00, 0x08,  # Report
        0x00, 0x00, 0x00, 0x08,  # MsgType
    ])


@pytest.fixture
def sample_iso8583_data() -> bytes:
    """Sample ISO 8583 financial message."""
    return bytes([
        0x02, 0x00,  # MTI: 0200
        0x30, 0x20, 0x05, 0x80, 0x20, 0xC0, 0x00, 0x04,  # Bitmap
        0x16,  # PAN length
        0x34, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31,
        0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31,
    ])


@pytest.fixture
def sample_modbus_tcp_data() -> bytes:
    """Sample Modbus TCP request."""
    return bytes([
        0x00, 0x01,  # Transaction ID
        0x00, 0x00,  # Protocol ID
        0x00, 0x06,  # Length
        0x01,        # Unit ID
        0x03,        # Function: Read Holding Registers
        0x00, 0x00,  # Starting address
        0x00, 0x0A,  # Quantity (10)
    ])


@pytest.fixture
def sample_cobol_ebcdic_record() -> bytes:
    """Sample COBOL record with EBCDIC encoding and packed decimals."""
    record = bytearray(500)
    # Record type "01"
    record[0:2] = bytes([0xF0, 0xF1])
    # Customer ID "0000000001"
    for i in range(9):
        record[2+i] = 0xF0
    record[11] = 0xF1
    # First name "JOHN" in EBCDIC
    record[12:16] = bytes([0xD1, 0xD6, 0xC8, 0xD5])
    # Fill with spaces
    for i in range(16, 37):
        record[i] = 0x40
    # Packed decimal balance: $12,345.67 (COMP-3)
    record[200:208] = bytes([0x00, 0x00, 0x00, 0x01, 0x23, 0x45, 0x67, 0x0C])
    return bytes(record)


# =============================================================================
# Protocol Signature Database Fixture
# =============================================================================

@pytest.fixture
def protocol_signature_db():
    """Get the protocol signature database."""
    try:
        from ai_engine.discovery.protocol_signatures import get_signature_database
        return get_signature_database()
    except ImportError:
        return None


# =============================================================================
# PQC Bridge Fixture
# =============================================================================

@pytest.fixture
def pqc_bridge_factory():
    """Factory for creating PQC protocol bridges."""
    try:
        from ai_engine.translation.protocol_bridge import (
            create_pqc_bridge,
            PQCBridgeMode,
        )

        def create_bridge(source="COBOL_RECORD", target="JSON", mode=PQCBridgeMode.FULL):
            return create_pqc_bridge(
                source_protocol=source,
                target_protocol=target,
                mode=mode,
            )

        return create_bridge
    except ImportError:
        return None
