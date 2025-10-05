"""
Tests for ai_engine/api/k8s_health.py - Kubernetes Health Checks
"""

import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from ai_engine.core.config import Config


class TestKubernetesHealthProbes:
    """Test suite for KubernetesHealthProbes."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock(spec=Config)
        config.startup_timeout = 300
        config.readiness_threshold = 0.8
        config.liveness_threshold = 0.5
        return config

    @pytest.fixture
    def mock_health_checker(self):
        """Create mock health checker."""
        from ai_engine.monitoring.health import HealthChecker, HealthStatus, SystemHealth
        
        checker = Mock(spec=HealthChecker)
        system_health = Mock(spec=SystemHealth)
        system_health.overall_status = HealthStatus.HEALTHY
        system_health.component_health = {"component1": Mock(), "component2": Mock()}
        system_health.get_healthy_components.return_value = ["component1", "component2"]
        
        checker.check_all_components = AsyncMock(return_value=system_health)
        return checker

    @pytest.fixture
    def health_probes(self, mock_config, mock_health_checker):
        """Create KubernetesHealthProbes instance."""
        from ai_engine.api.k8s_health import KubernetesHealthProbes
        return KubernetesHealthProbes(mock_config, mock_health_checker)

    def test_probes_initialization(self, health_probes, mock_config):
        """Test probes initialization."""
        assert health_probes.config == mock_config
        assert health_probes.startup_timeout == 300
        assert health_probes.readiness_threshold == 0.8
        assert health_probes.liveness_threshold == 0.5
        assert health_probes.startup_complete is False

    def test_create_router(self, health_probes):
        """Test router creation."""
        router = health_probes.create_router()
        
        assert router is not None
        assert router.prefix == "/health"

    @pytest.mark.asyncio
    async def test_check_liveness_pass(self, health_probes):
        """Test liveness check passes."""
        from ai_engine.api.k8s_health import ProbeStatus
        
        result = await health_probes.check_liveness()
        
        assert result.status == ProbeStatus.PASS
        assert "event_loop" in result.checks
        assert "health_checker" in result.checks
        assert "memory" in result.checks

    @pytest.mark.asyncio
    async def test_check_liveness_no_health_checker(self, health_probes):
        """Test liveness check with no health checker."""
        from ai_engine.api.k8s_health import ProbeStatus
        
        health_probes.health_checker = None
        result = await health_probes.check_liveness()
        
        assert result.status == ProbeStatus.FAIL
        assert result.checks["health_checker"]["status"] == "fail"

    @pytest.mark.asyncio
    async def test_check_liveness_exception(self, health_probes):
        """Test liveness check handles exceptions."""
        from ai_engine.api.k8s_health import ProbeStatus
        
        with patch('psutil.virtual_memory', side_effect=Exception("Memory error")):
            result = await health_probes.check_liveness()
            
            assert result.status == ProbeStatus.FAIL
            assert "error" in result.checks

    @pytest.mark.asyncio
    async def test_check_readiness_not_started(self, health_probes):
        """Test readiness check when startup not complete."""
        from ai_engine.api.k8s_health import ProbeStatus
        
        health_probes.startup_complete = False
        result = await health_probes.check_readiness()
        
        assert result.status == ProbeStatus.FAIL
        assert result.checks["startup"]["status"] == "fail"

    @pytest.mark.asyncio
    async def test_check_readiness_pass(self, health_probes, mock_health_checker):
        """Test readiness check passes."""
        from ai_engine.api.k8s_health import ProbeStatus
        
        health_probes.startup_complete = True
        result = await health_probes.check_readiness()
        
        assert result.status == ProbeStatus.PASS
        assert "dependencies" in result.checks
        assert "system_health" in result.checks

    @pytest.mark.asyncio
    async def test_check_readiness_low_health_ratio(self, health_probes, mock_health_checker):
        """Test readiness check with low health ratio."""
        from ai_engine.api.k8s_health import ProbeStatus
        from ai_engine.monitoring.health import SystemHealth, HealthStatus
        
        health_probes.startup_complete = True
        
        # Mock unhealthy system
        system_health = Mock(spec=SystemHealth)
        system_health.overall_status = HealthStatus.DEGRADED
        system_health.component_health = {"c1": Mock(), "c2": Mock(), "c3": Mock(), "c4": Mock()}
        system_health.get_healthy_components.return_value = ["c1"]  # Only 1 of 4 healthy
        
        mock_health_checker.check_all_components = AsyncMock(return_value=system_health)
        
        result = await health_probes.check_readiness()
        
        assert result.status == ProbeStatus.FAIL

    @pytest.mark.asyncio
    async def test_check_startup_timeout_exceeded(self, health_probes):
        """Test startup check with timeout exceeded."""
        from ai_engine.api.k8s_health import ProbeStatus
        
        health_probes.startup_time = time.time() - 400  # Exceeded 300s timeout
        result = await health_probes.check_startup()
        
        assert result.status == ProbeStatus.FAIL
        assert "timeout" in result.checks

    @pytest.mark.asyncio
    async def test_check_startup_already_complete(self, health_probes):
        """Test startup check when already complete."""
        from ai_engine.api.k8s_health import ProbeStatus
        
        health_probes.startup_complete = True
        result = await health_probes.check_startup()
        
        assert result.status == ProbeStatus.PASS
        assert result.checks["startup"]["status"] == "pass"

    @pytest.mark.asyncio
    async def test_check_startup_minimum_time_not_elapsed(self, health_probes):
        """Test startup check before minimum time elapsed."""
        from ai_engine.api.k8s_health import ProbeStatus
        
        health_probes.startup_time = time.time() - 2  # Only 2 seconds
        result = await health_probes.check_startup()
        
        assert result.status == ProbeStatus.FAIL
        assert result.checks["minimum_time"]["status"] == "fail"

    @pytest.mark.asyncio
    async def test_check_startup_success(self, health_probes):
        """Test successful startup check."""
        from ai_engine.api.k8s_health import ProbeStatus
        
        health_probes.startup_time = time.time() - 10  # 10 seconds elapsed
        result = await health_probes.check_startup()
        
        # Should mark startup as complete
        assert health_probes.startup_complete is True

    @pytest.mark.asyncio
    async def test_check_dependencies(self, health_probes):
        """Test dependency checks."""
        from ai_engine.api.k8s_health import ProbeStatus
        
        result = await health_probes.check_dependencies()
        
        assert result.status == ProbeStatus.PASS
        assert "database" in result.checks
        assert "redis" in result.checks
        assert "model_registry" in result.checks
        assert "external_services" in result.checks

    @pytest.mark.asyncio
    async def test_check_dependencies_timeout(self, health_probes):
        """Test dependency check with timeout."""
        from ai_engine.api.k8s_health import ProbeStatus
        import asyncio
        
        # Mock a slow dependency check
        async def slow_check():
            await asyncio.sleep(10)
            return {"status": "pass"}
        
        health_probes.dependency_checkers["slow_dep"] = slow_check
        
        result = await health_probes.check_dependencies()
        
        # Should have timeout entry
        assert "slow_dep" in result.checks
        assert result.checks["slow_dep"]["status"] == "fail"

    def test_mark_startup_complete(self, health_probes):
        """Test manually marking startup as complete."""
        health_probes.mark_startup_complete()
        
        assert health_probes.startup_complete is True

    def test_get_probe_status(self, health_probes):
        """Test getting probe status."""
        status = health_probes.get_probe_status()
        
        assert "startup_complete" in status
        assert "uptime_seconds" in status
        assert "last_liveness_check" in status
        assert "last_readiness_check" in status
        assert "startup_timeout" in status


class TestProbeEndpoints:
    """Test suite for probe endpoints."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock(spec=Config)
        config.startup_timeout = 300
        config.readiness_threshold = 0.8
        config.liveness_threshold = 0.5
        return config

    @pytest.fixture
    def mock_health_checker(self):
        """Create mock health checker."""
        from ai_engine.monitoring.health import HealthChecker, HealthStatus, SystemHealth
        
        checker = Mock(spec=HealthChecker)
        system_health = Mock(spec=SystemHealth)
        system_health.overall_status = HealthStatus.HEALTHY
        system_health.component_health = {"component1": Mock(), "component2": Mock()}
        system_health.get_healthy_components.return_value = ["component1", "component2"]
        
        checker.check_all_components = AsyncMock(return_value=system_health)
        return checker

    @pytest.mark.asyncio
    async def test_liveness_endpoint_pass(self, mock_config, mock_health_checker):
        """Test liveness endpoint returns 200 when passing."""
        from ai_engine.api.k8s_health import KubernetesHealthProbes
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        probes = KubernetesHealthProbes(mock_config, mock_health_checker)
        router = probes.create_router()
        
        app = FastAPI()
        app.include_router(router)
        
        client = TestClient(app)
        response = client.get("/health/live")
        
        assert response.status_code == 200
        assert response.json()["status"] == "pass"

    @pytest.mark.asyncio
    async def test_readiness_endpoint_not_ready(self, mock_config, mock_health_checker):
        """Test readiness endpoint returns 503 when not ready."""
        from ai_engine.api.k8s_health import KubernetesHealthProbes
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        probes = KubernetesHealthProbes(mock_config, mock_health_checker)
        probes.startup_complete = False
        router = probes.create_router()
        
        app = FastAPI()
        app.include_router(router)
        
        client = TestClient(app)
        response = client.get("/health/ready")
        
        assert response.status_code == 503
        assert response.json()["status"] == "fail"

    @pytest.mark.asyncio
    async def test_startup_endpoint(self, mock_config, mock_health_checker):
        """Test startup endpoint."""
        from ai_engine.api.k8s_health import KubernetesHealthProbes
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        probes = KubernetesHealthProbes(mock_config, mock_health_checker)
        probes.startup_time = time.time() - 10
        router = probes.create_router()
        
        app = FastAPI()
        app.include_router(router)
        
        client = TestClient(app)
        response = client.get("/health/startup")
        
        assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_dependencies_endpoint(self, mock_config, mock_health_checker):
        """Test dependencies endpoint."""
        from ai_engine.api.k8s_health import KubernetesHealthProbes
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        probes = KubernetesHealthProbes(mock_config, mock_health_checker)
        router = probes.create_router()
        
        app = FastAPI()
        app.include_router(router)
        
        client = TestClient(app)
        response = client.get("/health/dependencies")
        
        assert response.status_code == 200
        assert "checks" in response.json()


class TestProbeResult:
    """Test suite for ProbeResult dataclass."""

    def test_probe_result_creation(self):
        """Test ProbeResult creation."""
        from ai_engine.api.k8s_health import ProbeResult, ProbeStatus
        
        result = ProbeResult(
            status=ProbeStatus.PASS,
            checks={"test": {"status": "pass"}},
            timestamp=time.time(),
            response_time_ms=10.5
        )
        
        assert result.status == ProbeStatus.PASS
        assert "test" in result.checks
        assert result.response_time_ms == 10.5


class TestProbeStatus:
    """Test suite for ProbeStatus enum."""

    def test_probe_status_values(self):
        """Test ProbeStatus enum values."""
        from ai_engine.api.k8s_health import ProbeStatus
        
        assert ProbeStatus.PASS.value == "pass"
        assert ProbeStatus.FAIL.value == "fail"
        assert ProbeStatus.WARN.value == "warn"


class TestDependencyCheckers:
    """Test suite for dependency checker functions."""

    @pytest.fixture
    def health_probes(self, mock_config, mock_health_checker):
        """Create KubernetesHealthProbes instance."""
        from ai_engine.api.k8s_health import KubernetesHealthProbes
        return KubernetesHealthProbes(mock_config, mock_health_checker)

    @pytest.mark.asyncio
    async def test_database_checker(self, health_probes):
        """Test database dependency checker."""
        checker = health_probes.dependency_checkers["database"]
        result = await checker()
        
        assert "status" in result
        assert "critical" in result
        assert result["critical"] is True

    @pytest.mark.asyncio
    async def test_redis_checker(self, health_probes):
        """Test Redis dependency checker."""
        checker = health_probes.dependency_checkers["redis"]
        result = await checker()
        
        assert "status" in result
        assert "critical" in result
        assert result["critical"] is False

    @pytest.mark.asyncio
    async def test_model_registry_checker(self, health_probes):
        """Test model registry dependency checker."""
        checker = health_probes.dependency_checkers["model_registry"]
        result = await checker()
        
        assert "status" in result
        assert "critical" in result

    @pytest.mark.asyncio
    async def test_external_services_checker(self, health_probes):
        """Test external services dependency checker."""
        checker = health_probes.dependency_checkers["external_services"]
        result = await checker()
        
        assert "status" in result
        assert "critical" in result
        assert result["critical"] is False

    @pytest.fixture
    def mock_health_checker(self):
        """Create mock health checker."""
        from ai_engine.monitoring.health import HealthChecker, HealthStatus, SystemHealth
        
        checker = Mock(spec=HealthChecker)
        system_health = Mock(spec=SystemHealth)
        system_health.overall_status = HealthStatus.HEALTHY
        system_health.component_health = {"component1": Mock(), "component2": Mock()}
        system_health.get_healthy_components.return_value = ["component1", "component2"]
        
        checker.check_all_components = AsyncMock(return_value=system_health)
        return checker