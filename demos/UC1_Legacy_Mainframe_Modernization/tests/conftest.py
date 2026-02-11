"""
Pytest configuration and fixtures for UC1 Demo tests.
"""

import os
import sys
import asyncio
import pytest
from typing import AsyncGenerator, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEMO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DEMO_ROOT / "backend"))

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["AUTH_REQUIRED"] = "false"
os.environ["DB_DRIVER"] = "sqlite"
os.environ["DB_NAME"] = "test_uc1"


# =============================================================================
# Event Loop Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Application Fixtures
# =============================================================================

@pytest.fixture
def test_client():
    """Create test client for API testing."""
    from fastapi.testclient import TestClient

    # Import after path setup
    try:
        from production_app import app
        return TestClient(app)
    except ImportError:
        pytest.skip("Production app not available")


@pytest.fixture
async def async_client():
    """Create async test client."""
    from httpx import AsyncClient, ASGITransport

    try:
        from production_app import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    except ImportError:
        pytest.skip("Production app not available")


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_service():
    """Create mock LLM service."""
    mock = AsyncMock()

    # Mock process_request
    mock.process_request.return_value = MagicMock(
        content='{"program_name": "TESTPROG", "complexity_score": 0.5}',
        parsed_response={"program_name": "TESTPROG", "complexity_score": 0.5},
        provider="mock",
        model="mock-model",
        tokens_used=100,
    )

    mock.initialize = AsyncMock()
    mock.shutdown = AsyncMock()
    mock.primary_provider = MagicMock(value="mock")
    mock.get_current_provider.return_value = "mock"
    mock.get_health_status.return_value = {"status": "healthy"}

    return mock


@pytest.fixture
def mock_legacy_whisperer():
    """Create mock Legacy Whisperer."""
    from dataclasses import dataclass

    @dataclass
    class MockProtocolSpec:
        protocol_name: str = "MockProtocol"
        version: str = "1.0"
        description: str = "Mock protocol specification"
        complexity: MagicMock = MagicMock(value="moderate")
        fields: list = None
        message_types: list = None
        patterns: list = None
        is_binary: bool = True
        is_stateful: bool = False
        uses_encryption: bool = False
        has_checksums: bool = False
        confidence_score: float = 0.85
        documentation: str = "Mock documentation"

        def __post_init__(self):
            self.fields = self.fields or []
            self.message_types = self.message_types or []
            self.patterns = self.patterns or []

    @dataclass
    class MockAdapter:
        adapter_code: str = "# Mock adapter code"
        test_code: str = "# Mock test code"
        documentation: str = "Mock documentation"
        dependencies: list = None
        code_quality_score: float = 0.8

        def __post_init__(self):
            self.dependencies = self.dependencies or []

    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.shutdown = AsyncMock()
    mock.reverse_engineer_protocol = AsyncMock(return_value=MockProtocolSpec())
    mock.generate_adapter_code = AsyncMock(return_value=MockAdapter())
    mock.get_statistics.return_value = {"analysis_cache_size": 0, "adapter_cache_size": 0}

    return mock


@pytest.fixture
def mock_legacy_service():
    """Create mock Legacy System Whisperer Service."""
    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.shutdown = AsyncMock()
    mock.registered_systems = {}
    mock.get_service_metrics.return_value = {
        "requests_processed": 0,
        "predictions_generated": 0,
    }

    async def mock_register(system_context, enable_monitoring=True):
        mock.registered_systems[system_context.system_id] = system_context
        return {"capabilities_enabled": ["cobol_analysis", "protocol_analysis"]}

    mock.register_legacy_system = mock_register

    return mock


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_cobol_source():
    """Sample COBOL source code for testing."""
    return """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. TESTPROG.
       AUTHOR. TEST.

       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUSTOMER-FILE ASSIGN TO 'CUSTMAST.DAT'
               ORGANIZATION IS INDEXED
               ACCESS MODE IS DYNAMIC
               RECORD KEY IS CUST-ID
               FILE STATUS IS WS-FILE-STATUS.

       DATA DIVISION.
       FILE SECTION.
       FD CUSTOMER-FILE.
       01 CUSTOMER-RECORD.
           05 CUST-ID             PIC X(10).
           05 CUST-NAME           PIC X(50).
           05 CUST-BALANCE        PIC 9(9)V99.
           05 CUST-STATUS         PIC X(1).

       WORKING-STORAGE SECTION.
       01 WS-FILE-STATUS         PIC XX.
       01 WS-EOF                 PIC 9 VALUE 0.

       PROCEDURE DIVISION.
       MAIN-PROCEDURE.
           PERFORM INITIALIZATION
           PERFORM PROCESS-RECORDS UNTIL WS-EOF = 1
           PERFORM CLEANUP
           STOP RUN.

       INITIALIZATION.
           OPEN INPUT CUSTOMER-FILE.

       PROCESS-RECORDS.
           READ CUSTOMER-FILE
               AT END SET WS-EOF TO 1
               NOT AT END PERFORM PROCESS-SINGLE-RECORD
           END-READ.

       PROCESS-SINGLE-RECORD.
           DISPLAY "Processing: " CUST-ID.

       CLEANUP.
           CLOSE CUSTOMER-FILE.
    """


@pytest.fixture
def sample_protocol_samples():
    """Sample protocol data for testing."""
    return [
        "01020304000a48656c6c6f576f726c64",  # Header + "HelloWorld"
        "01020304000b48656c6c6f576f726c6432",  # Header + "HelloWorld2"
        "01020304000c48656c6c6f576f726c643132",  # Header + "HelloWorld12"
        "01020304000648656c6c6f21",  # Header + "Hello!"
        "0102030400054d7367203031",  # Header + "Msg 01"
        "0102030400054d7367203032",  # Header + "Msg 02"
        "0102030400054d7367203033",  # Header + "Msg 03"
        "0102030400054d7367203034",  # Header + "Msg 04"
        "0102030400054d7367203035",  # Header + "Msg 05"
        "0102030400054d7367203036",  # Header + "Msg 06"
    ]


@pytest.fixture
def sample_system_registration():
    """Sample system registration data."""
    return {
        "system_name": "Test Mainframe",
        "system_type": "mainframe",
        "manufacturer": "IBM",
        "model": "z15",
        "version": "2.4",
        "location": "Data Center A",
        "criticality": "high",
        "business_function": "Core Banking",
        "metadata": {
            "environment": "production",
            "owner": "IT Operations",
        },
    }


@pytest.fixture
def sample_modernization_request():
    """Sample modernization request data."""
    return {
        "system_id": "test-system-123",
        "target_technology": "python-fastapi",
        "include_code_generation": True,
        "include_risk_assessment": True,
        "objectives": ["improve_performance", "reduce_cost", "increase_maintainability"],
    }


@pytest.fixture
def sample_knowledge_capture():
    """Sample knowledge capture data."""
    return {
        "expert_id": "expert-001",
        "system_id": "test-system-123",
        "session_type": "interview",
        "knowledge_input": """
        When the batch job NIGHTLY01 runs longer than usual, it's often because
        the customer file has grown beyond 100,000 records. The solution is to
        run the archive job ARCH01 first to move inactive records to cold storage.
        This behavior has been consistent for the past 5 years.
        """,
        "context": {
            "job_name": "NIGHTLY01",
            "typical_duration_minutes": 45,
            "max_acceptable_duration_minutes": 90,
        },
    }


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
async def test_database():
    """Create test database."""
    try:
        from database import DatabaseManager, DatabaseConfig

        config = DatabaseConfig(
            driver="sqlite",
            database="test_uc1_db",
        )
        db = DatabaseManager(config)
        await db.initialize()
        yield db
        await db.shutdown()

        # Cleanup test database
        import os
        db_path = os.path.join(os.path.dirname(__file__), "..", "backend", "test_uc1_db.db")
        if os.path.exists(db_path):
            os.remove(db_path)
    except ImportError:
        pytest.skip("Database module not available")


# =============================================================================
# Authentication Fixtures
# =============================================================================

@pytest.fixture
def auth_headers():
    """Create authentication headers."""
    return {
        "X-API-Key": "demo-key-for-testing-only",
    }


@pytest.fixture
def admin_headers():
    """Create admin authentication headers."""
    return {
        "X-API-Key": os.getenv("ADMIN_API_KEY", "admin-test-key"),
    }


# =============================================================================
# Utility Functions
# =============================================================================

def assert_response_success(response, expected_status=200):
    """Assert response is successful."""
    assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}: {response.text}"


def assert_response_error(response, expected_status):
    """Assert response is an error."""
    assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}"
    assert "error" in response.json() or "detail" in response.json()
