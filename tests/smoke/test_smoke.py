"""
QBITEL - Smoke Tests

Quick validation tests for CI/CD pipelines:
- Basic import validation
- Configuration loading
- Database connectivity
- API endpoint availability
- Critical service health

These tests should complete quickly (< 60 seconds total).
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Import Tests
# =============================================================================


class TestImports:
    """Test that all critical modules can be imported."""

    def test_import_ai_engine(self):
        """Test ai_engine package imports."""
        import ai_engine

        assert ai_engine is not None

    def test_import_core_modules(self):
        """Test core module imports."""
        from ai_engine.core import circuit_breakers
        from ai_engine.core import constants
        from ai_engine.core import error_codes
        from ai_engine.core import feature_flags

        assert circuit_breakers is not None
        assert constants is not None
        assert error_codes is not None
        assert feature_flags is not None

    def test_import_discovery_modules(self):
        """Test discovery module imports."""
        from ai_engine.discovery import ProtocolDiscoveryOrchestrator

        assert ProtocolDiscoveryOrchestrator is not None

    def test_import_llm_modules(self):
        """Test LLM module imports."""
        from ai_engine.llm import LegacyWhisperer

        assert LegacyWhisperer is not None

    def test_import_api_modules(self):
        """Test API module imports."""
        from ai_engine.api import rest

        assert rest is not None

    def test_import_models(self):
        """Test model imports."""
        from ai_engine.models import registry

        assert registry is not None

    def test_import_sandbox(self):
        """Test sandbox module imports."""
        from ai_engine.sandbox import FirecrackerSandbox, SandboxConfig

        assert FirecrackerSandbox is not None
        assert SandboxConfig is not None

    def test_import_serving(self):
        """Test serving module imports."""
        from ai_engine.serving import KServeManager, ModelRouter

        assert KServeManager is not None
        assert ModelRouter is not None

    def test_import_siem(self):
        """Test SIEM integration imports."""
        from ai_engine.integrations.siem import SIEMConnectorFactory

        assert SIEMConnectorFactory is not None

    def test_import_observability(self):
        """Test observability imports."""
        from ai_engine.observability import metrics, tracing

        assert metrics is not None
        assert tracing is not None


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_environment_variables_exist(self):
        """Test required environment variables can be loaded."""
        # These should have defaults or be optional
        env_vars = [
            "QBITEL_LOG_LEVEL",
            "QBITEL_DEBUG",
        ]

        for var in env_vars:
            # Just verify we can access them (may be None)
            value = os.environ.get(var)
            # We're not asserting they exist, just that access works

    def test_config_files_exist(self):
        """Test critical configuration files exist."""
        config_files = [
            PROJECT_ROOT / "pyproject.toml",
            PROJECT_ROOT / "ops/monitoring/prometheus/alerts.yml",
        ]

        for config_file in config_files:
            assert config_file.exists(), f"Config file missing: {config_file}"

    def test_feature_flags_load(self):
        """Test feature flags can be loaded."""
        from ai_engine.core.feature_flags import feature_flags

        # Should not raise
        flags = feature_flags.get_all_flags()
        assert isinstance(flags, dict)

    def test_error_codes_load(self):
        """Test error codes are properly defined."""
        from ai_engine.core.error_codes import ErrorCode

        # Verify error codes exist
        assert hasattr(ErrorCode, "VALIDATION_ERROR")
        assert hasattr(ErrorCode, "INTERNAL_ERROR")


# =============================================================================
# Component Health Tests
# =============================================================================


class TestComponentHealth:
    """Test individual component health."""

    def test_circuit_breakers_initialize(self):
        """Test circuit breakers initialize correctly."""
        from ai_engine.core.circuit_breakers import circuit_breakers

        # Verify default breakers are registered
        assert "llm" in circuit_breakers
        assert "database" in circuit_breakers
        assert "redis" in circuit_breakers

    def test_circuit_breaker_status(self):
        """Test circuit breaker status reporting."""
        from ai_engine.core.circuit_breakers import circuit_breakers

        status = circuit_breakers.get_all_status()
        assert isinstance(status, dict)

        # All breakers should start closed
        for name, breaker_status in status.items():
            assert breaker_status["state"] == "closed"

    def test_graceful_degradation_initialize(self):
        """Test graceful degradation service initializes."""
        from ai_engine.core.graceful_degradation import degradation_service

        status = degradation_service.get_status()
        assert isinstance(status, dict)
        assert "total_services" in status
        assert "cache" in status

    def test_metrics_exportable(self):
        """Test Prometheus metrics can be exported."""
        from prometheus_client import generate_latest, REGISTRY

        # Should not raise
        metrics_output = generate_latest(REGISTRY)
        assert len(metrics_output) > 0


# =============================================================================
# API Tests
# =============================================================================


class TestAPISmoke:
    """Smoke tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from ai_engine.api.rest import app

        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health endpoint responds."""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # May be unhealthy in test env

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint responds."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    def test_openapi_spec(self, client):
        """Test OpenAPI spec is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert "openapi" in spec
        assert "paths" in spec


# =============================================================================
# Database Tests
# =============================================================================


class TestDatabaseSmoke:
    """Smoke tests for database connectivity."""

    @pytest.mark.skipif(
        not os.environ.get("DATABASE_URL"),
        reason="DATABASE_URL not set",
    )
    def test_database_connection(self):
        """Test database connection can be established."""
        from sqlalchemy import create_engine, text

        engine = create_engine(os.environ["DATABASE_URL"])
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    @pytest.mark.skipif(
        not os.environ.get("REDIS_URL"),
        reason="REDIS_URL not set",
    )
    def test_redis_connection(self):
        """Test Redis connection can be established."""
        import redis

        r = redis.from_url(os.environ["REDIS_URL"])
        assert r.ping()


# =============================================================================
# Security Tests
# =============================================================================


class TestSecuritySmoke:
    """Smoke tests for security components."""

    def test_security_middleware_loads(self):
        """Test security middleware can be loaded."""
        from ai_engine.api.security_middleware import (
            SecurityConfig,
            SecurityHeadersMiddleware,
            setup_security_middleware,
        )

        assert SecurityConfig is not None
        assert SecurityHeadersMiddleware is not None
        assert setup_security_middleware is not None

    def test_security_config_defaults(self):
        """Test security config has safe defaults."""
        from ai_engine.api.security_middleware import SecurityConfig

        config = SecurityConfig()

        # Verify secure defaults
        assert config.force_https is True
        assert config.hsts_max_age >= 31536000  # At least 1 year
        assert config.frame_options == "DENY"
        assert config.content_type_nosniff is True


# =============================================================================
# Integration Smoke Tests
# =============================================================================


class TestIntegrationSmoke:
    """Smoke tests for integrations."""

    def test_siem_connector_factory(self):
        """Test SIEM connector factory works."""
        from ai_engine.integrations.siem import SIEMConnectorFactory

        # Should list available connectors
        available = SIEMConnectorFactory.get_available_connectors()
        assert "splunk" in available
        assert "sentinel" in available
        assert "chronicle" in available

    def test_sdk_generator_loads(self):
        """Test SDK generator can be loaded."""
        from sdk_generator import SDKGenerator

        assert SDKGenerator is not None

    def test_kserve_manager_loads(self):
        """Test KServe manager can be loaded."""
        from ai_engine.serving.kserve_manager import KServeManager

        assert KServeManager is not None


# =============================================================================
# Performance Baseline Tests
# =============================================================================


class TestPerformanceBaseline:
    """Basic performance tests to catch regressions."""

    def test_import_time(self):
        """Test module import time is reasonable."""
        import time

        start = time.time()
        import ai_engine  # noqa: F401

        duration = time.time() - start
        # Should import in under 5 seconds
        assert duration < 5.0, f"Import took too long: {duration:.2f}s"

    def test_config_load_time(self):
        """Test configuration loading is fast."""
        import time

        from ai_engine.core.feature_flags import FeatureFlagService

        start = time.time()
        service = FeatureFlagService()
        _ = service.get_all_flags()
        duration = time.time() - start

        # Should complete in under 1 second
        assert duration < 1.0, f"Config load took too long: {duration:.2f}s"


# =============================================================================
# Async Smoke Tests
# =============================================================================


class TestAsyncSmoke:
    """Smoke tests for async components."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_async(self):
        """Test circuit breaker works with async functions."""
        from ai_engine.core.circuit_breakers import circuit_breakers

        breaker = circuit_breakers["llm"]

        # Test successful call
        async def success_fn():
            return "success"

        result = await breaker.call(success_fn)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_degradation_service_async(self):
        """Test degradation service works with async."""
        from ai_engine.core.graceful_degradation import degradation_service

        async def primary_fn():
            return {"data": "test"}

        result = await degradation_service.execute_with_fallback(
            "test_service",
            primary_fn,
        )
        assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_cache_async(self):
        """Test response cache works."""
        from ai_engine.core.graceful_degradation import ResponseCache

        cache = ResponseCache()

        await cache.set("test_service", "test_value", "arg1")
        result = await cache.get("test_service", "arg1")

        assert result == "test_value"


# =============================================================================
# File Structure Tests
# =============================================================================


class TestFileStructure:
    """Test project file structure is correct."""

    def test_required_directories_exist(self):
        """Test required directories exist."""
        required_dirs = [
            "ai_engine",
            "ai_engine/api",
            "ai_engine/core",
            "ai_engine/discovery",
            "ai_engine/llm",
            "ai_engine/models",
            "ops",
            "tests",
        ]

        for dir_path in required_dirs:
            full_path = PROJECT_ROOT / dir_path
            assert full_path.is_dir(), f"Directory missing: {dir_path}"

    def test_required_files_exist(self):
        """Test required files exist."""
        required_files = [
            "pyproject.toml",
            "ai_engine/__init__.py",
            "ai_engine/api/rest.py",
        ]

        for file_path in required_files:
            full_path = PROJECT_ROOT / file_path
            assert full_path.is_file(), f"File missing: {file_path}"

    def test_no_syntax_errors(self):
        """Test Python files have no syntax errors."""
        import ast

        python_files = list(PROJECT_ROOT.glob("ai_engine/**/*.py"))[:20]  # Sample

        for py_file in python_files:
            try:
                with open(py_file, "r") as f:
                    source = f.read()
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file}: {e}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
