"""
CRONOS AI - Pytest Configuration and Fixtures

This module provides shared fixtures and configuration for all tests.
"""

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
