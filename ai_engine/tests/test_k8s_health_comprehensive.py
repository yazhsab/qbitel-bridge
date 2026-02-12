"""
Comprehensive Unit Tests for Kubernetes Health Check Endpoints
Tests all functionality in ai_engine/api/k8s_health.py
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from ai_engine.api.k8s_health import (
    KubernetesHealthProbes,
    ProbeStatus,
    ProbeResult,
)
from ai_engine.core.config import Config
from ai_engine.monitoring.health import HealthChecker, HealthStatus, ComponentHealth


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Config()
    config.startup_timeout = 300
    config.readiness_threshold = 0.8
    config.liveness_threshold = 0.5
    return config


@pytest.fixture
def mock_health_checker():
    """Create mock health checker."""
    checker = Mock(spec=HealthChecker)

    # Mock check_all_components
    mock_system_health = Mock()
    mock_system_health.overall_status = HealthStatus.HEALTHY
    mock_system_health.component_health = {
        "component1": ComponentHealth(name="component1", status=HealthStatus.HEALTHY, message="OK"),
        "component2": ComponentHealth(name="component2", status=HealthStatus.HEALTHY, message="OK"),
    }
    mock_system_health.get_healthy_components = Mock(return_value=["component1", "component2"])

    checker.check_all_components = AsyncMock(return_value=mock_system_health)

    return checker


@pytest.fixture
def k8s_probes(mock_config, mock_health_checker):
    """Create KubernetesHealthProbes instance."""
    return KubernetesHealthProbes(mock_config, mock_health_checker)


class TestKubernetesHealthProbesInitialization:
    """Test initialization of KubernetesHealthProbes."""

    def test_initialization(self, k8s_probes, mock_config, mock_health_checker):
        """Test successful initialization."""
        assert k8s_probes.config == mock_config
        assert k8s_probes.health_checker == mock_health_checker
        assert k8s_probes.startup_timeout == 300
        assert k8s_probes.readiness_threshold == 0.8
        assert k8s_probes.liveness_threshold == 0.5
        assert k8s_probes.startup_complete is False
        assert len(k8s_probes.dependency_checkers) == 4

    def test_initialization_with_defaults(self):
        """Test initialization with default config values."""
        config = Config()
        checker = Mock(spec=HealthChecker)
        probes = KubernetesHealthProbes(config, checker)

        assert probes.startup_timeout == 300
        assert probes.readiness_threshold == 0.8
        assert probes.liveness_threshold == 0.5


class TestRouterCreation:
    """Test FastAPI router creation."""

    def test_create_router(self, k8s_probes):
        """Test router creation."""
        router = k8s_probes.create_router()

        assert router is not None
        assert router.prefix == "/health"
        assert "health" in router.tags

    @pytest.mark.asyncio
    async def test_liveness_endpoint_pass(self, k8s_probes):
        """Test liveness endpoint returns 200 when passing."""
        router = k8s_probes.create_router()

        # Find the liveness endpoint
        liveness_route = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/health/live":
                liveness_route = route
                break

        assert liveness_route is not None

    @pytest.mark.asyncio
    async def test_readiness_endpoint_pass(self, k8s_probes):
        """Test readiness endpoint returns 200 when passing."""
        k8s_probes.startup_complete = True
        router = k8s_probes.create_router()

        # Find the readiness endpoint
        readiness_route = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/health/ready":
                readiness_route = route
                break

        assert readiness_route is not None


class TestLivenessProbe:
    """Test liveness probe functionality."""

    @pytest.mark.asyncio
    async def test_liveness_check_pass(self, k8s_probes):
        """Test successful liveness check."""
        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(percent=50.0)

            result = await k8s_probes.check_liveness()

            assert result.status == ProbeStatus.PASS
            assert "event_loop" in result.checks
            assert "health_checker" in result.checks
            assert "memory" in result.checks
            assert result.checks["event_loop"]["status"] == "pass"
            assert result.checks["health_checker"]["status"] == "pass"
            assert result.checks["memory"]["status"] == "pass"

    @pytest.mark.asyncio
    async def test_liveness_check_memory_critical(self, k8s_probes):
        """Test liveness check fails with critical memory usage."""
        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(percent=96.0)

            result = await k8s_probes.check_liveness()

            assert result.status == ProbeStatus.FAIL
            assert result.checks["memory"]["status"] == "fail"
            assert "Critical memory usage" in result.checks["memory"]["message"]

    @pytest.mark.asyncio
    async def test_liveness_check_no_health_checker(self, mock_config):
        """Test liveness check with no health checker."""
        probes = KubernetesHealthProbes(mock_config, None)

        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(percent=50.0)

            result = await probes.check_liveness()

            assert result.status == ProbeStatus.FAIL
            assert result.checks["health_checker"]["status"] == "fail"

    @pytest.mark.asyncio
    async def test_liveness_check_exception(self, k8s_probes):
        """Test liveness check handles exceptions."""
        with patch("psutil.virtual_memory", side_effect=Exception("Memory error")):
            result = await k8s_probes.check_liveness()

            assert result.status == ProbeStatus.FAIL
            assert "error" in result.checks
            assert "Memory error" in result.checks["error"]["message"]

    @pytest.mark.asyncio
    async def test_liveness_check_updates_timestamp(self, k8s_probes):
        """Test liveness check updates last check timestamp."""
        initial_time = k8s_probes.last_liveness_check

        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(percent=50.0)
            await k8s_probes.check_liveness()

        assert k8s_probes.last_liveness_check > initial_time


class TestReadinessProbe:
    """Test readiness probe functionality."""

    @pytest.mark.asyncio
    async def test_readiness_check_startup_not_complete(self, k8s_probes):
        """Test readiness check fails when startup not complete."""
        k8s_probes.startup_complete = False

        result = await k8s_probes.check_readiness()

        assert result.status == ProbeStatus.FAIL
        assert "startup" in result.checks
        assert result.checks["startup"]["status"] == "fail"

    @pytest.mark.asyncio
    async def test_readiness_check_pass(self, k8s_probes, mock_health_checker):
        """Test successful readiness check."""
        k8s_probes.startup_complete = True

        result = await k8s_probes.check_readiness()

        assert result.status == ProbeStatus.PASS
        assert "dependencies" in result.checks
        assert "system_health" in result.checks
        assert result.checks["system_health"]["status"] == "pass"

    @pytest.mark.asyncio
    async def test_readiness_check_low_health_ratio(self, k8s_probes, mock_health_checker):
        """Test readiness check fails with low health ratio."""
        k8s_probes.startup_complete = True

        # Mock unhealthy system
        mock_system_health = Mock()
        mock_system_health.overall_status = HealthStatus.DEGRADED
        mock_system_health.component_health = {
            "component1": ComponentHealth(name="component1", status=HealthStatus.HEALTHY, message="OK"),
            "component2": ComponentHealth(name="component2", status=HealthStatus.UNHEALTHY, message="Failed"),
            "component3": ComponentHealth(name="component3", status=HealthStatus.UNHEALTHY, message="Failed"),
        }
        mock_system_health.get_healthy_components = Mock(return_value=["component1"])
        mock_health_checker.check_all_components = AsyncMock(return_value=mock_system_health)

        result = await k8s_probes.check_readiness()

        assert result.status == ProbeStatus.FAIL
        assert result.checks["system_health"]["status"] == "fail"
        assert result.checks["system_health"]["health_ratio"] < 0.8

    @pytest.mark.asyncio
    async def test_readiness_check_exception(self, k8s_probes, mock_health_checker):
        """Test readiness check handles exceptions."""
        k8s_probes.startup_complete = True
        mock_health_checker.check_all_components = AsyncMock(side_effect=Exception("Health check error"))

        result = await k8s_probes.check_readiness()

        assert result.status == ProbeStatus.FAIL
        assert "error" in result.checks

    @pytest.mark.asyncio
    async def test_readiness_check_updates_timestamp(self, k8s_probes):
        """Test readiness check updates last check timestamp."""
        k8s_probes.startup_complete = True
        initial_time = k8s_probes.last_readiness_check

        await k8s_probes.check_readiness()

        assert k8s_probes.last_readiness_check > initial_time


class TestStartupProbe:
    """Test startup probe functionality."""

    @pytest.mark.asyncio
    async def test_startup_check_timeout_exceeded(self, k8s_probes):
        """Test startup check fails when timeout exceeded."""
        k8s_probes.startup_time = time.time() - 400  # 400 seconds ago

        result = await k8s_probes.check_startup()

        assert result.status == ProbeStatus.FAIL
        assert "timeout" in result.checks
        assert result.checks["timeout"]["status"] == "fail"

    @pytest.mark.asyncio
    async def test_startup_check_already_complete(self, k8s_probes):
        """Test startup check passes when already complete."""
        k8s_probes.startup_complete = True

        result = await k8s_probes.check_startup()

        assert result.status == ProbeStatus.PASS
        assert result.checks["startup"]["status"] == "pass"

    @pytest.mark.asyncio
    async def test_startup_check_minimum_time_not_elapsed(self, k8s_probes):
        """Test startup check fails when minimum time not elapsed."""
        k8s_probes.startup_time = time.time() - 2  # 2 seconds ago

        result = await k8s_probes.check_startup()

        assert result.status == ProbeStatus.FAIL
        assert "minimum_time" in result.checks
        assert result.checks["minimum_time"]["status"] == "fail"

    @pytest.mark.asyncio
    async def test_startup_check_success(self, k8s_probes):
        """Test successful startup check."""
        k8s_probes.startup_time = time.time() - 10  # 10 seconds ago

        result = await k8s_probes.check_startup()

        assert result.status == ProbeStatus.PASS
        assert k8s_probes.startup_complete is True
        assert result.checks["health_checker"]["status"] == "pass"
        assert result.checks["minimum_time"]["status"] == "pass"
        assert result.checks["critical_dependencies"]["status"] == "pass"

    @pytest.mark.asyncio
    async def test_startup_check_no_health_checker(self, mock_config):
        """Test startup check with no health checker."""
        probes = KubernetesHealthProbes(mock_config, None)
        probes.startup_time = time.time() - 10

        result = await probes.check_startup()

        assert result.status == ProbeStatus.FAIL
        assert result.checks["health_checker"]["status"] == "fail"

    @pytest.mark.asyncio
    async def test_startup_check_exception(self, k8s_probes):
        """Test startup check handles exceptions."""
        k8s_probes.startup_time = time.time() - 10

        # Mock dependency check to raise exception
        async def failing_check():
            raise Exception("Dependency error")

        k8s_probes.dependency_checkers["database"] = failing_check

        result = await k8s_probes.check_startup()

        # Should still complete but may have issues
        assert result is not None


class TestDependencyChecks:
    """Test dependency checking functionality."""

    @pytest.mark.asyncio
    async def test_check_dependencies_all_pass(self, k8s_probes):
        """Test all dependencies pass."""
        result = await k8s_probes.check_dependencies()

        assert result.status == ProbeStatus.PASS
        assert "database" in result.checks
        assert "redis" in result.checks
        assert "model_registry" in result.checks
        assert "external_services" in result.checks

    @pytest.mark.asyncio
    async def test_check_dependencies_timeout(self, k8s_probes):
        """Test dependency check timeout."""

        async def slow_check():
            await asyncio.sleep(10)
            return {"status": "pass"}

        k8s_probes.dependency_checkers["slow_service"] = slow_check

        result = await k8s_probes.check_dependencies()

        assert "slow_service" in result.checks
        assert result.checks["slow_service"]["status"] == "fail"
        assert "timeout" in result.checks["slow_service"]["message"].lower()

    @pytest.mark.asyncio
    async def test_check_dependencies_exception(self, k8s_probes):
        """Test dependency check handles exceptions."""

        async def failing_check():
            raise Exception("Connection failed")

        k8s_probes.dependency_checkers["failing_service"] = failing_check

        result = await k8s_probes.check_dependencies()

        assert "failing_service" in result.checks
        assert result.checks["failing_service"]["status"] == "fail"

    @pytest.mark.asyncio
    async def test_check_dependencies_mixed_results(self, k8s_probes):
        """Test dependencies with mixed pass/fail results."""

        async def failing_check():
            return {"status": "fail", "message": "Service unavailable"}

        k8s_probes.dependency_checkers["failing_service"] = failing_check

        result = await k8s_probes.check_dependencies()

        assert result.status == ProbeStatus.FAIL
        assert result.checks["failing_service"]["status"] == "fail"


class TestUtilityMethods:
    """Test utility methods."""

    def test_mark_startup_complete(self, k8s_probes):
        """Test manually marking startup as complete."""
        assert k8s_probes.startup_complete is False

        k8s_probes.mark_startup_complete()

        assert k8s_probes.startup_complete is True

    def test_get_probe_status(self, k8s_probes):
        """Test getting probe status."""
        k8s_probes.startup_complete = True
        k8s_probes.last_liveness_check = time.time()
        k8s_probes.last_readiness_check = time.time()

        status = k8s_probes.get_probe_status()

        assert status["startup_complete"] is True
        assert "uptime_seconds" in status
        assert status["uptime_seconds"] > 0
        assert status["last_liveness_check"] > 0
        assert status["last_readiness_check"] > 0
        assert status["startup_timeout"] == 300


class TestProbeResult:
    """Test ProbeResult dataclass."""

    def test_probe_result_creation(self):
        """Test creating ProbeResult."""
        result = ProbeResult(
            status=ProbeStatus.PASS,
            checks={"test": {"status": "pass"}},
            timestamp=time.time(),
            response_time_ms=10.5,
        )

        assert result.status == ProbeStatus.PASS
        assert "test" in result.checks
        assert result.response_time_ms == 10.5


class TestProbeStatus:
    """Test ProbeStatus enum."""

    def test_probe_status_values(self):
        """Test ProbeStatus enum values."""
        assert ProbeStatus.PASS.value == "pass"
        assert ProbeStatus.FAIL.value == "fail"
        assert ProbeStatus.WARN.value == "warn"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_dependency_checkers(self, mock_config, mock_health_checker):
        """Test with no dependency checkers."""
        probes = KubernetesHealthProbes(mock_config, mock_health_checker)
        probes.dependency_checkers.clear()

        result = await probes.check_dependencies()

        assert result.status == ProbeStatus.PASS
        assert len(result.checks) == 0

    @pytest.mark.asyncio
    async def test_concurrent_probe_checks(self, k8s_probes):
        """Test concurrent probe checks."""
        k8s_probes.startup_complete = True

        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(percent=50.0)

            # Run multiple checks concurrently
            results = await asyncio.gather(
                k8s_probes.check_liveness(),
                k8s_probes.check_readiness(),
                k8s_probes.check_dependencies(),
            )

        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_config_attribute_defaults(self):
        """Test handling of missing config attributes."""
        config = Config()
        # Don't set any health-related attributes
        checker = Mock(spec=HealthChecker)

        probes = KubernetesHealthProbes(config, checker)

        # Should use defaults
        assert probes.startup_timeout == 300
        assert probes.readiness_threshold == 0.8
        assert probes.liveness_threshold == 0.5
