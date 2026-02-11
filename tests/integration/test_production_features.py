"""
QBITEL - Integration Tests for Production Features

Tests for newly implemented production-ready features:
- Security middleware
- Circuit breakers
- Graceful degradation
- Health checks
- Incident response
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Security Middleware Tests
# =============================================================================


class TestSecurityMiddleware:
    """Tests for security middleware."""

    def test_security_config_defaults(self):
        """Test SecurityConfig has secure defaults."""
        from ai_engine.api.security_middleware import SecurityConfig

        config = SecurityConfig()

        # HTTPS/HSTS
        assert config.force_https is True
        assert config.hsts_max_age >= 31536000  # At least 1 year
        assert config.hsts_include_subdomains is True

        # Content Security Policy
        assert config.csp_enabled is True
        assert "default-src" in config.csp_directives
        assert "'self'" in config.csp_directives["default-src"]

        # Frame options
        assert config.frame_options == "DENY"

        # Content type options
        assert config.content_type_nosniff is True

        # XSS protection
        assert "1" in config.xss_protection

    def test_security_headers_middleware_init(self):
        """Test SecurityHeadersMiddleware initialization."""
        from fastapi import FastAPI
        from ai_engine.api.security_middleware import (
            SecurityConfig,
            SecurityHeadersMiddleware,
        )

        app = FastAPI()
        config = SecurityConfig()

        middleware = SecurityHeadersMiddleware(app, config)

        assert middleware.config == config

    @pytest.mark.asyncio
    async def test_security_headers_added_to_response(self):
        """Test security headers are added to responses."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ai_engine.api.security_middleware import (
            SecurityConfig,
            setup_security_middleware,
        )

        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        config = SecurityConfig(force_https=False)  # Disable for testing
        setup_security_middleware(app, config)

        client = TestClient(app)
        response = client.get("/test")

        # Verify security headers
        assert "X-Frame-Options" in response.headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "Referrer-Policy" in response.headers

    def test_cors_configuration(self):
        """Test CORS is properly configured."""
        from ai_engine.api.security_middleware import SecurityConfig

        config = SecurityConfig()

        assert config.cors_enabled is True
        assert len(config.cors_origins) > 0
        assert "GET" in config.cors_allow_methods
        assert "POST" in config.cors_allow_methods


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreakers:
    """Tests for circuit breaker implementation."""

    def test_circuit_breaker_registry(self):
        """Test circuit breaker registry."""
        from ai_engine.core.circuit_breakers import circuit_breakers

        # Verify default breakers are registered
        assert "llm" in circuit_breakers
        assert "database" in circuit_breakers
        assert "redis" in circuit_breakers
        assert "discovery" in circuit_breakers
        assert "external_api" in circuit_breakers

    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        from ai_engine.core.circuit_breakers import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            name="test_breaker",
            failure_threshold=2,
            recovery_timeout=1.0,
        )
        breaker = CircuitBreaker(config)

        # Initial state is closed
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_circuit_breaker_success_flow(self):
        """Test circuit breaker with successful calls."""
        from ai_engine.core.circuit_breakers import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            name="test_success",
            failure_threshold=3,
        )
        breaker = CircuitBreaker(config)

        async def success_fn():
            return "success"

        result = await breaker.call(success_fn)

        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_opens(self):
        """Test circuit breaker opens after failures."""
        from ai_engine.core.circuit_breakers import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            name="test_failure",
            failure_threshold=2,
            recovery_timeout=60.0,
        )
        breaker = CircuitBreaker(config)

        async def failing_fn():
            raise Exception("Test failure")

        # Fail twice to reach threshold
        for _ in range(2):
            try:
                await breaker.call(failing_fn)
            except Exception:
                pass

        # Circuit should now be open
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator."""
        from ai_engine.core.circuit_breakers import with_circuit_breaker

        @with_circuit_breaker("llm")
        async def decorated_fn():
            return "decorated_result"

        result = await decorated_fn()
        assert result == "decorated_result"

    def test_circuit_breaker_status(self):
        """Test circuit breaker status reporting."""
        from ai_engine.core.circuit_breakers import circuit_breakers

        status = circuit_breakers.get_all_status()

        assert isinstance(status, dict)
        for name, breaker_status in status.items():
            assert "state" in breaker_status
            assert "failure_count" in breaker_status
            assert "config" in breaker_status


# =============================================================================
# Graceful Degradation Tests
# =============================================================================


class TestGracefulDegradation:
    """Tests for graceful degradation service."""

    def test_degradation_service_init(self):
        """Test degradation service initialization."""
        from ai_engine.core.graceful_degradation import (
            DegradationService,
            degradation_service,
        )

        assert degradation_service is not None
        assert isinstance(degradation_service, DegradationService)

    def test_fallback_registration(self):
        """Test fallback registration."""
        from ai_engine.core.graceful_degradation import (
            DegradationService,
            FallbackStrategy,
        )

        service = DegradationService()

        service.register_fallback(
            "test_service",
            strategy=FallbackStrategy.STATIC,
            static_value={"default": "value"},
        )

        assert "test_service" in service._fallbacks

    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self):
        """Test successful execution without fallback."""
        from ai_engine.core.graceful_degradation import DegradationService

        service = DegradationService()

        async def primary_fn():
            return {"result": "primary"}

        result = await service.execute_with_fallback(
            "test_service",
            primary_fn,
        )

        assert result == {"result": "primary"}

    @pytest.mark.asyncio
    async def test_execute_with_fallback_uses_cache(self):
        """Test fallback uses cached value on failure."""
        from ai_engine.core.graceful_degradation import (
            DegradationService,
            FallbackStrategy,
        )

        service = DegradationService()

        # Register cache fallback
        service.register_fallback(
            "cache_test",
            strategy=FallbackStrategy.CACHE,
        )

        # First call succeeds and caches
        async def success_fn():
            return {"cached": "value"}

        await service.execute_with_fallback("cache_test", success_fn)

        # Second call fails but uses cache
        async def fail_fn():
            raise Exception("Primary failed")

        result = await service.execute_with_fallback("cache_test", fail_fn)
        assert result == {"cached": "value"}

    def test_response_cache(self):
        """Test response cache operations."""
        from ai_engine.core.graceful_degradation import ResponseCache

        cache = ResponseCache()

        # Test cache key generation
        key1 = cache._make_key("service", "arg1", kwarg="value")
        key2 = cache._make_key("service", "arg1", kwarg="value")
        key3 = cache._make_key("service", "arg2")

        assert key1 == key2
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_response_cache_async(self):
        """Test async cache operations."""
        from ai_engine.core.graceful_degradation import ResponseCache

        cache = ResponseCache()

        await cache.set("test_service", "test_value", "arg1")
        result = await cache.get("test_service", "arg1")

        assert result == "test_value"

    def test_degradation_modes(self):
        """Test degradation mode tracking."""
        from ai_engine.core.graceful_degradation import (
            DegradationMode,
            DegradationService,
        )

        service = DegradationService()

        # Initial mode should be normal
        health = service.get_service_health("test_service")
        # New services start without health tracking
        assert health is None

    def test_get_degraded_services(self):
        """Test getting list of degraded services."""
        from ai_engine.core.graceful_degradation import degradation_service

        degraded = degradation_service.get_degraded_services()
        assert isinstance(degraded, list)


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthChecks:
    """Tests for health check service."""

    def test_health_service_init(self):
        """Test health service initialization."""
        from ai_engine.api.health_checks import HealthCheckService

        service = HealthCheckService()
        assert service is not None

    @pytest.mark.asyncio
    async def test_system_resources_check(self):
        """Test system resources health check."""
        from ai_engine.api.health_checks import HealthCheckService, HealthStatus

        service = HealthCheckService()
        result = await service.check_system_resources()

        assert result.name == "system_resources"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNKNOWN]
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_network_check(self):
        """Test network connectivity check."""
        from ai_engine.api.health_checks import HealthCheckService, HealthStatus

        service = HealthCheckService()
        result = await service.check_network_connectivity()

        assert result.name == "network"
        # May be degraded in CI environment
        assert result.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]

    @pytest.mark.asyncio
    async def test_check_all(self):
        """Test comprehensive health check."""
        from ai_engine.api.health_checks import HealthCheckService

        service = HealthCheckService()
        result = await service.check_all()

        assert result.components is not None
        assert len(result.components) > 0
        assert result.latency_ms >= 0

    def test_health_status_enum(self):
        """Test health status enumeration."""
        from ai_engine.api.health_checks import HealthStatus

        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    @pytest.mark.asyncio
    async def test_custom_endpoint_check(self):
        """Test custom endpoint health check."""
        from ai_engine.api.health_checks import HealthCheckService, HealthStatus

        service = HealthCheckService()

        # Test with a reliable endpoint
        result = await service.check_custom_endpoint(
            name="httpbin",
            url="https://httpbin.org/status/200",
            expected_status=200,
        )

        # May fail in CI without network
        assert result.name == "httpbin"


# =============================================================================
# Incident Response Tests
# =============================================================================


class TestIncidentResponse:
    """Tests for incident response runbook."""

    def test_incident_response_init(self):
        """Test incident response runbook initialization."""
        from ops.operational.incident_response_runbook import IncidentResponseRunbook

        runbook = IncidentResponseRunbook()
        assert runbook is not None
        assert runbook.contacts is not None

    def test_severity_levels(self):
        """Test incident severity levels."""
        from ops.operational.incident_response_runbook import IncidentSeverity

        assert IncidentSeverity.SEV1.value == "sev1"
        assert IncidentSeverity.SEV2.value == "sev2"
        assert IncidentSeverity.SEV3.value == "sev3"
        assert IncidentSeverity.SEV4.value == "sev4"

    def test_incident_types(self):
        """Test incident types."""
        from ops.operational.incident_response_runbook import IncidentType

        assert IncidentType.SERVICE_OUTAGE.value == "service_outage"
        assert IncidentType.SECURITY_INCIDENT.value == "security_incident"
        assert IncidentType.DATA_CORRUPTION.value == "data_corruption"

    @pytest.mark.asyncio
    async def test_create_incident(self):
        """Test incident creation."""
        from ops.operational.incident_response_runbook import (
            IncidentResponseRunbook,
            IncidentSeverity,
            IncidentType,
            IncidentStatus,
        )

        runbook = IncidentResponseRunbook()

        incident = await runbook.create_incident(
            title="Test Incident",
            severity=IncidentSeverity.SEV3,
            incident_type=IncidentType.SERVICE_OUTAGE,
            description="Test description",
            affected_services=["test-service"],
            impact="Test impact",
        )

        assert incident.incident_id.startswith("INC-")
        assert incident.status == IncidentStatus.DETECTED
        assert len(incident.timeline) > 0

    def test_get_runbook(self):
        """Test getting runbook for incident type."""
        from ops.operational.incident_response_runbook import (
            IncidentResponseRunbook,
            IncidentType,
        )

        runbook = IncidentResponseRunbook()

        service_outage_runbook = runbook.get_runbook(IncidentType.SERVICE_OUTAGE)

        assert "title" in service_outage_runbook
        assert "steps" in service_outage_runbook
        assert len(service_outage_runbook["steps"]) > 0

    def test_escalation_paths(self):
        """Test escalation paths are defined."""
        from ops.operational.incident_response_runbook import (
            IncidentResponseRunbook,
            IncidentSeverity,
        )

        runbook = IncidentResponseRunbook()

        # SEV1 should have escalation path
        sev1_path = runbook.ESCALATION_PATHS[IncidentSeverity.SEV1]
        assert len(sev1_path) > 0
        assert "contacts" in sev1_path[0]

    @pytest.mark.asyncio
    async def test_acknowledge_incident(self):
        """Test incident acknowledgment."""
        from ops.operational.incident_response_runbook import (
            IncidentResponseRunbook,
            IncidentSeverity,
            IncidentType,
            IncidentStatus,
        )

        runbook = IncidentResponseRunbook()

        incident = await runbook.create_incident(
            title="Ack Test",
            severity=IncidentSeverity.SEV3,
            incident_type=IncidentType.SERVICE_OUTAGE,
            description="Test",
            affected_services=["test"],
            impact="Test",
        )

        await runbook.acknowledge_incident(
            incident.incident_id,
            "test-responder",
        )

        assert incident.status == IncidentStatus.ACKNOWLEDGED
        assert incident.acknowledged_at is not None
        assert "test-responder" in incident.responders


# =============================================================================
# DR Testing Tests
# =============================================================================


class TestDRTesting:
    """Tests for DR testing framework."""

    def test_dr_framework_init(self):
        """Test DR testing framework initialization."""
        from ops.operational.dr_testing import DRTestingFramework

        framework = DRTestingFramework()
        assert framework is not None
        assert framework.config is not None

    def test_test_categories(self):
        """Test DR test categories."""
        from ops.operational.dr_testing import TestCategory

        assert TestCategory.BACKUP.value == "backup"
        assert TestCategory.FAILOVER.value == "failover"
        assert TestCategory.DATA_INTEGRITY.value == "data_integrity"
        assert TestCategory.RECOVERY_TIME.value == "recovery_time"

    def test_test_result_enum(self):
        """Test DR test result enum."""
        from ops.operational.dr_testing import TestResult

        assert TestResult.PASS.value == "pass"
        assert TestResult.FAIL.value == "fail"
        assert TestResult.SKIP.value == "skip"
        assert TestResult.ERROR.value == "error"

    @pytest.mark.asyncio
    async def test_run_backup_tests(self):
        """Test running backup tests."""
        from ops.operational.dr_testing import DRTestingFramework, TestCategory

        framework = DRTestingFramework()
        results = await framework.run_test_category(TestCategory.BACKUP)

        assert len(results) > 0
        for result in results:
            assert result.category == TestCategory.BACKUP

    @pytest.mark.asyncio
    async def test_run_runbook_tests(self):
        """Test running runbook validation tests."""
        from ops.operational.dr_testing import DRTestingFramework, TestCategory

        framework = DRTestingFramework()
        results = await framework.run_test_category(TestCategory.RUNBOOK)

        assert len(results) > 0


# =============================================================================
# Production Readiness Tests
# =============================================================================


class TestProductionReadiness:
    """Tests for production readiness checker."""

    def test_readiness_checker_init(self):
        """Test production readiness checker initialization."""
        from scripts.check_production_readiness import ProductionReadinessChecker

        checker = ProductionReadinessChecker()
        assert checker is not None
        assert checker.config is not None

    def test_check_categories(self):
        """Test check categories are defined."""
        from scripts.check_production_readiness import CheckCategory

        assert CheckCategory.INFRASTRUCTURE.value == "infrastructure"
        assert CheckCategory.SECURITY.value == "security"
        assert CheckCategory.MONITORING.value == "monitoring"
        assert CheckCategory.CICD.value == "cicd"

    @pytest.mark.asyncio
    async def test_run_checks(self):
        """Test running production readiness checks."""
        from scripts.check_production_readiness import ProductionReadinessChecker

        checker = ProductionReadinessChecker()
        report = await checker.run_all_checks()

        assert report is not None
        assert report.checks is not None
        assert report.score >= 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_degradation(self):
        """Test circuit breaker integrates with graceful degradation."""
        from ai_engine.core.circuit_breakers import circuit_breakers
        from ai_engine.core.graceful_degradation import degradation_service

        # Both services should be initialized
        assert circuit_breakers is not None
        assert degradation_service is not None

        # Get status from both
        cb_status = circuit_breakers.get_all_status()
        deg_status = degradation_service.get_status()

        assert isinstance(cb_status, dict)
        assert isinstance(deg_status, dict)

    @pytest.mark.asyncio
    async def test_health_checks_all_components(self):
        """Test health checks cover all components."""
        from ai_engine.api.health_checks import health_service

        result = await health_service.check_all()

        # Verify all expected components are checked
        expected_components = ["database", "redis", "llm_service", "system_resources", "network"]

        for component in expected_components:
            assert component in result.components


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
