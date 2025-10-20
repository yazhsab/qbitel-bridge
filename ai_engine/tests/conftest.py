"""
CRONOS AI - Pytest Configuration and Fixtures

This module provides shared fixtures and configuration for all tests.
"""

# Import mocks FIRST before any other imports
from . import conftest_mocks  # noqa: F401

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock
from prometheus_client import REGISTRY

from ai_engine.core.config import Config


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test function."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    # Clean up any remaining tasks
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
        pass
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = Config()
    config.environment = "testing"
    config.model_path = "test_models"
    config.data_path = "test_data"
    config.device = "cpu"
    config.health_check_interval = 5
    config.health_check_timeout = 2
    return config


@pytest.fixture
def mock_database():
    """Mock database connection."""
    db = AsyncMock()
    db.execute = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchrow = AsyncMock(return_value=None)
    db.fetchval = AsyncMock(return_value=None)
    return db


@pytest.fixture
def mock_redis():
    """Mock Redis connection."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=0)
    redis.expire = AsyncMock(return_value=True)
    return redis


@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    llm = AsyncMock()
    llm.process_request = AsyncMock()
    llm.process_request.return_value.content = "Mock LLM response"
    return llm


@pytest.fixture(autouse=True)
def mock_etcd3(monkeypatch):
    """Mock etcd3 to avoid protobuf issues."""
    from unittest.mock import MagicMock
    import sys

    # Create comprehensive etcd3 mock
    mock_etcd = MagicMock()
    mock_client = MagicMock()
    mock_etcd.client = MagicMock(return_value=mock_client)

    # Mock the module
    sys.modules["etcd3"] = mock_etcd
    sys.modules["etcd3.etcdrpc"] = MagicMock()
    sys.modules["etcd3.etcdrpc.rpc_pb2"] = MagicMock()
    sys.modules["etcd3.etcdrpc.kv_pb2"] = MagicMock()

    yield mock_etcd

    # Cleanup
    for module in [
        "etcd3",
        "etcd3.etcdrpc",
        "etcd3.etcdrpc.rpc_pb2",
        "etcd3.etcdrpc.kv_pb2",
    ]:
        if module in sys.modules:
            del sys.modules[module]


@pytest.fixture
def mock_password_hash():
    """Mock password hashing for authentication tests."""
    from unittest.mock import patch

    with patch("passlib.hash.bcrypt.hash") as mock_hash:
        with patch("passlib.hash.bcrypt.verify") as mock_verify:
            mock_hash.return_value = "$2b$12$test_hashed_password_mock_value_12345"
            mock_verify.return_value = True
            yield {"hash": mock_hash, "verify": mock_verify}


@pytest.fixture
def mock_auth_service():
    """Mock authentication service functions."""
    from unittest.mock import patch

    mocks = {}
    patches = []

    # Try to patch various auth locations
    auth_modules = [
        "ai_engine.api.auth",
        "ai_engine.api.auth_enterprise",
    ]

    for module in auth_modules:
        try:
            p1 = patch(f"{module}.hash_password", return_value="$2b$12$hashed")
            p2 = patch(f"{module}.verify_password", return_value=True)
            p3 = patch(f"{module}.get_api_key", return_value="test_api_key_12345")

            mocks[f"{module}.hash"] = p1.start()
            mocks[f"{module}.verify"] = p2.start()
            mocks[f"{module}.api_key"] = p3.start()

            patches.extend([p1, p2, p3])
        except (ImportError, AttributeError):
            pass

    yield mocks

    for p in patches:
        try:
            p.stop()
        except:
            pass


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "chaos: marks tests as chaos engineering tests")


@pytest.fixture(autouse=True)
def clear_prometheus_registry():
    """Clear Prometheus registry before each test to avoid duplicate metrics."""
    # Get list of collectors to remove
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield
    # Clean up after test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


# Test data generators
def generate_sample_packet_data() -> bytes:
    """Generate sample packet data for testing."""
    return b"GET /api/v1/test HTTP/1.1\r\nHost: example.com\r\n\r\n"


def generate_sample_modbus_data() -> str:
    """Generate sample Modbus data for testing."""
    return "0103000000024C0B"
