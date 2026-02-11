"""
QBITEL - Smoke Test Configuration

Pytest configuration for smoke tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Configure pytest for smoke tests."""
    config.addinivalue_line(
        "markers", "smoke: mark test as a smoke test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )


@pytest.fixture(scope="session")
def project_root():
    """Return project root path."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def is_ci():
    """Check if running in CI environment."""
    ci_vars = ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL"]
    return any(os.environ.get(var) for var in ci_vars)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Set test environment variables
    os.environ.setdefault("QBITEL_ENV", "test")
    os.environ.setdefault("QBITEL_LOG_LEVEL", "WARNING")
    os.environ.setdefault("QBITEL_DEBUG", "false")

    yield

    # Cleanup if needed


@pytest.fixture
def mock_database():
    """Mock database for tests that don't need real DB."""
    class MockDB:
        def execute(self, query):
            return None

        def fetchall(self):
            return []

    return MockDB()


@pytest.fixture
def mock_redis():
    """Mock Redis for tests that don't need real Redis."""
    class MockRedis:
        def __init__(self):
            self._data = {}

        def get(self, key):
            return self._data.get(key)

        def set(self, key, value, ex=None):
            self._data[key] = value

        def delete(self, key):
            self._data.pop(key, None)

        def ping(self):
            return True

    return MockRedis()


# Skip slow tests in CI unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    if os.environ.get("CI") and not os.environ.get("RUN_SLOW_TESTS"):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests in CI")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
