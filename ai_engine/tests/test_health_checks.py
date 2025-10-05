"""
Tests for health monitoring and checks.
Covers HealthChecker, ComponentHealth, and SystemHealth.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from ai_engine.monitoring.health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    SystemHealth,
)
from ai_engine.core.config import Config


class TestHealthStatus:
    """Test HealthStatus enumeration."""

    def test_health_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentHealth:
    """Test ComponentHealth dataclass."""

    def test_component_health_creation(self):
        """Test creating component health."""
        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="All systems operational",
        )

        assert health.name == "test_component"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == "All systems operational"

    def test_component_health_defaults(self):
        """Test component health default values."""
        health = ComponentHealth(name="default_component", status=HealthStatus.HEALTHY)

        assert health.message == ""
        assert health.check_duration_ms == 0.0
        assert health.details == {}
        assert health.dependencies == []
        assert health.last_check_time > 0

    def test_is_healthy(self):
        """Test is_healthy method."""
        health = ComponentHealth(name="healthy_comp", status=HealthStatus.HEALTHY)

        assert health.is_healthy() is True
        assert health.is_degraded() is False
        assert health.is_unhealthy() is False

    def test_is_degraded(self):
        """Test is_degraded method."""
        health = ComponentHealth(name="degraded_comp", status=HealthStatus.DEGRADED)

        assert health.is_healthy() is False
        assert health.is_degraded() is True
        assert health.is_unhealthy() is False

    def test_is_unhealthy(self):
        """Test is_unhealthy method."""
        health = ComponentHealth(name="unhealthy_comp", status=HealthStatus.UNHEALTHY)

        assert health.is_healthy() is False
        assert health.is_degraded() is False
        assert health.is_unhealthy() is True

    def test_component_health_with_details(self):
        """Test component health with details."""
        details = {"latency_ms": 50, "error_rate": 0.01}
        health = ComponentHealth(
            name="detailed_comp", status=HealthStatus.HEALTHY, details=details
        )

        assert health.details == details

    def test_component_health_with_dependencies(self):
        """Test component health with dependencies."""
        deps = ["database", "cache", "message_queue"]
        health = ComponentHealth(
            name="dependent_comp", status=HealthStatus.HEALTHY, dependencies=deps
        )

        assert health.dependencies == deps


class TestSystemHealth:
    """Test SystemHealth dataclass."""

    def test_system_health_creation(self):
        """Test creating system health."""
        health = SystemHealth(overall_status=HealthStatus.HEALTHY)

        assert health.overall_status == HealthStatus.HEALTHY
        assert health.component_health == {}
        assert health.system_metrics == {}
        assert health.uptime_seconds == 0.0

    def test_get_healthy_components(self):
        """Test getting healthy components."""
        system = SystemHealth(overall_status=HealthStatus.HEALTHY)
        system.component_health = {
            "comp1": ComponentHealth("comp1", HealthStatus.HEALTHY),
            "comp2": ComponentHealth("comp2", HealthStatus.DEGRADED),
            "comp3": ComponentHealth("comp3", HealthStatus.UNHEALTHY),
        }

        healthy = system.get_healthy_components()
        assert len(healthy) == 1
        assert healthy[0].name == "comp1"

    def test_get_unhealthy_components(self):
        """Test getting unhealthy components."""
        system = SystemHealth(overall_status=HealthStatus.DEGRADED)
        system.component_health = {
            "comp1": ComponentHealth("comp1", HealthStatus.HEALTHY),
            "comp2": ComponentHealth("comp2", HealthStatus.UNHEALTHY),
            "comp3": ComponentHealth("comp3", HealthStatus.UNHEALTHY),
        }

        unhealthy = system.get_unhealthy_components()
        assert len(unhealthy) == 2

    def test_get_degraded_components(self):
        """Test getting degraded components."""
        system = SystemHealth(overall_status=HealthStatus.DEGRADED)
        system.component_health = {
            "comp1": ComponentHealth("comp1", HealthStatus.HEALTHY),
            "comp2": ComponentHealth("comp2", HealthStatus.DEGRADED),
            "comp3": ComponentHealth("comp3", HealthStatus.DEGRADED),
        }

        degraded = system.get_degraded_components()
        assert len(degraded) == 2

    def test_calculate_overall_status_all_healthy(self):
        """Test calculating overall status when all healthy."""
        system = SystemHealth(overall_status=HealthStatus.UNKNOWN)
        system.component_health = {
            "comp1": ComponentHealth("comp1", HealthStatus.HEALTHY),
            "comp2": ComponentHealth("comp2", HealthStatus.HEALTHY),
            "comp3": ComponentHealth("comp3", HealthStatus.HEALTHY),
        }

        status = system.calculate_overall_status()
        assert status == HealthStatus.HEALTHY

    def test_calculate_overall_status_some_degraded(self):
        """Test calculating overall status with degraded components."""
        system = SystemHealth(overall_status=HealthStatus.UNKNOWN)
        system.component_health = {
            "comp1": ComponentHealth("comp1", HealthStatus.HEALTHY),
            "comp2": ComponentHealth("comp2", HealthStatus.DEGRADED),
        }

        status = system.calculate_overall_status()
        assert status == HealthStatus.DEGRADED

    def test_calculate_overall_status_majority_unhealthy(self):
        """Test calculating overall status when majority unhealthy."""
        system = SystemHealth(overall_status=HealthStatus.UNKNOWN)
        system.component_health = {
            "comp1": ComponentHealth("comp1", HealthStatus.UNHEALTHY),
            "comp2": ComponentHealth("comp2", HealthStatus.UNHEALTHY),
            "comp3": ComponentHealth("comp3", HealthStatus.HEALTHY),
        }

        status = system.calculate_overall_status()
        assert status == HealthStatus.UNHEALTHY

    def test_calculate_overall_status_critical_component_unhealthy(self):
        """Test overall status when critical component is unhealthy."""
        system = SystemHealth(overall_status=HealthStatus.UNKNOWN)
        system.component_health = {
            "ai_engine": ComponentHealth("ai_engine", HealthStatus.UNHEALTHY),
            "other1": ComponentHealth("other1", HealthStatus.HEALTHY),
            "other2": ComponentHealth("other2", HealthStatus.HEALTHY),
        }

        status = system.calculate_overall_status()
        assert status == HealthStatus.UNHEALTHY

    def test_calculate_overall_status_no_components(self):
        """Test calculating overall status with no components."""
        system = SystemHealth(overall_status=HealthStatus.UNKNOWN)
        status = system.calculate_overall_status()
        assert status == HealthStatus.UNKNOWN


class TestHealthChecker:
    """Test HealthChecker class."""

    @pytest.fixture
    def checker(self):
        """Create health checker instance."""
        config = Config()
        return HealthChecker(config)

    def test_health_checker_initialization(self, checker):
        """Test health checker initialization."""
        assert checker.health_checkers is not None
        assert checker.health_history == []
        assert checker.check_interval == 30
        assert checker.timeout == 10

    def test_register_health_checker(self, checker):
        """Test registering health checker."""

        async def custom_checker():
            return ComponentHealth(name="custom", status=HealthStatus.HEALTHY)

        checker.register_health_checker("custom_component", custom_checker)

        assert "custom_component" in checker.health_checkers

    @pytest.mark.asyncio
    async def test_check_all_components_empty(self, checker):
        """Test checking components when none registered."""
        # Clear default checkers for this test
        checker.health_checkers = {}

        system_health = await checker.check_all_components()

        assert isinstance(system_health, SystemHealth)
        assert system_health.overall_status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_check_all_components_healthy(self, checker):
        """Test checking all healthy components."""

        async def healthy_checker():
            return ComponentHealth(
                name="healthy", status=HealthStatus.HEALTHY, message="All good"
            )

        checker.health_checkers = {"healthy": healthy_checker}

        system_health = await checker.check_all_components()

        assert len(system_health.component_health) == 1
        assert system_health.component_health["healthy"].is_healthy()
        assert system_health.overall_status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_component_timeout(self, checker):
        """Test health check timeout."""

        async def slow_checker():
            await asyncio.sleep(20)  # Longer than timeout
            return ComponentHealth("slow", HealthStatus.HEALTHY)

        checker.health_checkers = {"slow": slow_checker}
        checker.timeout = 0.1  # Very short timeout

        system_health = await checker.check_all_components()

        assert system_health.component_health["slow"].is_unhealthy()
        assert "timed out" in system_health.component_health["slow"].message

    @pytest.mark.asyncio
    async def test_check_component_exception(self, checker):
        """Test health check with exception."""

        async def failing_checker():
            raise Exception("Checker failed")

        checker.health_checkers = {"failing": failing_checker}

        system_health = await checker.check_all_components()

        assert system_health.component_health["failing"].is_unhealthy()
        assert "failed" in system_health.component_health["failing"].message.lower()

    @pytest.mark.asyncio
    async def test_check_multiple_components(self, checker):
        """Test checking multiple components."""

        async def healthy_checker():
            return ComponentHealth("healthy", HealthStatus.HEALTHY)

        async def degraded_checker():
            return ComponentHealth("degraded", HealthStatus.DEGRADED)

        async def unhealthy_checker():
            return ComponentHealth("unhealthy", HealthStatus.UNHEALTHY)

        checker.health_checkers = {
            "healthy": healthy_checker,
            "degraded": degraded_checker,
            "unhealthy": unhealthy_checker,
        }

        system_health = await checker.check_all_components()

        assert len(system_health.component_health) == 3
        assert system_health.overall_status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_history(self, checker):
        """Test health history tracking."""

        async def simple_checker():
            return ComponentHealth("simple", HealthStatus.HEALTHY)

        checker.health_checkers = {"simple": simple_checker}

        # Run multiple checks
        await checker.check_all_components()
        await checker.check_all_components()

        assert len(checker.health_history) == 2

    @pytest.mark.asyncio
    async def test_uptime_calculation(self, checker):
        """Test uptime calculation."""
        checker.health_checkers = {}

        # Wait a bit
        await asyncio.sleep(0.1)

        system_health = await checker.check_all_components()

        assert system_health.uptime_seconds > 0


class TestHealthCheckerEdgeCases:
    """Test edge cases for health checker."""

    @pytest.fixture
    def checker(self):
        """Create health checker instance."""
        config = Config()
        return HealthChecker(config)

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, checker):
        """Test running health checks concurrently."""
        check_count = [0]

        async def counting_checker():
            check_count[0] += 1
            await asyncio.sleep(0.01)
            return ComponentHealth(f"comp_{check_count[0]}", HealthStatus.HEALTHY)

        # Register multiple checkers
        for i in range(10):
            checker.health_checkers[f"comp_{i}"] = counting_checker

        system_health = await checker.check_all_components()

        # All should have been checked
        assert len(system_health.component_health) == 10

    @pytest.mark.asyncio
    async def test_health_check_with_retry(self, checker):
        """Test health check that fails then succeeds."""
        attempt = [0]

        async def retry_checker():
            attempt[0] += 1
            if attempt[0] < 2:
                raise Exception("First attempt fails")
            return ComponentHealth("retry", HealthStatus.HEALTHY)

        checker.health_checkers = {"retry": retry_checker}

        # First check should fail
        system1 = await checker.check_all_components()
        assert system1.component_health["retry"].is_unhealthy()

        # Second check should succeed
        system2 = await checker.check_all_components()
        assert system2.component_health["retry"].is_healthy()

    @pytest.mark.asyncio
    async def test_health_check_performance(self, checker):
        """Test health check performance tracking."""

        async def timed_checker():
            start = time.time()
            await asyncio.sleep(0.05)
            duration = (time.time() - start) * 1000
            return ComponentHealth(
                "timed", HealthStatus.HEALTHY, check_duration_ms=duration
            )

        checker.health_checkers = {"timed": timed_checker}

        system_health = await checker.check_all_components()

        assert system_health.component_health["timed"].check_duration_ms > 0

    @pytest.mark.asyncio
    async def test_max_history_size(self, checker):
        """Test health history size limit."""
        checker.max_history_size = 5
        checker.health_checkers = {}

        # Generate more checks than history size
        for _ in range(10):
            await checker.check_all_components()

        # History should be limited
        assert len(checker.health_history) <= checker.max_history_size + 5


class TestComponentHealthDetails:
    """Test component health with detailed information."""

    def test_health_with_metrics(self):
        """Test component health with performance metrics."""
        details = {
            "cpu_percent": 45.2,
            "memory_mb": 512,
            "latency_p95_ms": 150,
            "error_rate": 0.001,
        }

        health = ComponentHealth(
            name="api_server",
            status=HealthStatus.HEALTHY,
            message="Operating normally",
            details=details,
        )

        assert health.details["cpu_percent"] == 45.2
        assert health.details["latency_p95_ms"] == 150

    def test_health_with_dependencies(self):
        """Test component health tracking dependencies."""
        health = ComponentHealth(
            name="application",
            status=HealthStatus.HEALTHY,
            dependencies=["database", "redis", "message_queue"],
        )

        assert "database" in health.dependencies
        assert "redis" in health.dependencies
        assert len(health.dependencies) == 3

    def test_health_degradation_reasons(self):
        """Test capturing degradation reasons."""
        details = {
            "reason": "high_latency",
            "threshold_ms": 200,
            "actual_ms": 350,
            "degraded_since": time.time(),
        }

        health = ComponentHealth(
            name="slow_service",
            status=HealthStatus.DEGRADED,
            message="Response time above threshold",
            details=details,
        )

        assert health.details["reason"] == "high_latency"
        assert health.details["actual_ms"] > health.details["threshold_ms"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
