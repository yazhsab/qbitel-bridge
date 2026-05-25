"""Shared fixtures for crypto unit tests."""

import os
import pytest
from unittest.mock import patch


@pytest.fixture
def pqc_fallback_mode(monkeypatch):
    """Enable fallback mode (allow insecure providers for testing).

    Also patches the module-level _GLOBAL_STRICT_MODE cached at import time
    in each crypto module so that factory methods (which don't accept
    strict_mode) use fallback instead of raising.
    """
    monkeypatch.setenv("QBITEL_PQC_ALLOW_FALLBACK", "1")

    # Patch the cached module-level strict-mode flag in every module that
    # copies it from providers at import time.
    import ai_engine.crypto.providers as _providers
    import ai_engine.crypto.mlkem as _mlkem
    import ai_engine.crypto.falcon as _falcon
    monkeypatch.setattr(_providers, "_GLOBAL_STRICT_MODE", False)
    monkeypatch.setattr(_mlkem, "_GLOBAL_STRICT_MODE", False)
    monkeypatch.setattr(_falcon, "_GLOBAL_STRICT_MODE", False)

    # Reset the provider registry singleton so it re-detects
    from ai_engine.crypto.providers import ProviderRegistry
    ProviderRegistry.reset()
    yield
    ProviderRegistry.reset()


@pytest.fixture
def pqc_strict_mode(monkeypatch):
    """Ensure strict mode is active (no fallback allowed)."""
    monkeypatch.delenv("QBITEL_PQC_ALLOW_FALLBACK", raising=False)
    from ai_engine.crypto.providers import ProviderRegistry
    ProviderRegistry.reset()
    yield
    ProviderRegistry.reset()


@pytest.fixture
def experimental_crypto_allowed(monkeypatch):
    """Enable experimental crypto modules."""
    monkeypatch.setenv("QBITEL_ALLOW_EXPERIMENTAL_CRYPTO", "1")
    yield


@pytest.fixture
def mock_no_providers(monkeypatch):
    """Mock environment where NO PQC providers are installed."""
    monkeypatch.setenv("QBITEL_PQC_ALLOW_FALLBACK", "1")
    from ai_engine.crypto.providers import ProviderRegistry
    ProviderRegistry.reset()

    import builtins
    original_import = builtins.__import__

    blocked = {"oqs", "pqcrypto", "kyber", "dilithium"}

    def mock_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"Mocked: {name} not available")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        ProviderRegistry.reset()
        yield

    ProviderRegistry.reset()
