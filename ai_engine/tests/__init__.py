"""Shared test utilities and defaults for ai_engine tests."""

import asyncio
import os
from typing import Generator

import pytest


DEFAULT_ENV_FLAG = "CRONOS_TEST_ENV"


def setup_test_environment() -> None:
    """Apply baseline environment variables for tests."""

    os.environ.setdefault(DEFAULT_ENV_FLAG, "1")
    os.environ.setdefault("CRONOS_ENV", "testing")


def cleanup_test_environment() -> None:
    """Clean up environment variables set during tests."""

    os.environ.pop(DEFAULT_ENV_FLAG, None)


class TestConfig:
    """Common testing defaults used across suites."""

    PERFORMANCE_THRESHOLD_MS: float = 250.0
    DEFAULT_TIMEOUT_SECONDS: float = 5.0
    MAX_CONCURRENT_REQUESTS: int = 20

    @pytest.fixture(scope="session")
    def event_loop(self) -> Generator[asyncio.AbstractEventLoop, None, None]:
        """Provide a dedicated event loop for async tests."""
        loop = asyncio.new_event_loop()
        try:
            yield loop
        finally:
            loop.close()
