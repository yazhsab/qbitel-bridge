"""
QBITEL Engine - Production Readiness Tests
Comprehensive test suite for production readiness features.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Import modules to test
from ai_engine.core.error_handling import (
    ErrorHandler,
    ErrorRecord,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    ErrorContext,
    CircuitBreaker,
    CircuitBreakerConfig,
)
from ai_engine.core.config_validator import ConfigValidator, ValidationSeverity


class TestErrorStorage:
    """Test persistent error storage."""

    @pytest.mark.asyncio
    async def test_error_storage_initialization(self):
        """Test error storage initialization."""
        from ai_engine.core.error_storage import PersistentErrorStorage

        storage = PersistentErrorStorage(
            redis_url="redis://localhost:6379/0",
            postgres_url="postgresql+asyncpg://test:test@localhost/test",
        )

        assert storage is not None
        assert storage.redis_ttl == 86400
        assert storage.postgres_retention_days == 90

    @pytest.mark.asyncio
    async def test_error_record_serialization(self):
        """Test error record serialization."""
        context = ErrorContext(
            component="test", operation="test_op", request_id="req-123"
        )

        error_record = ErrorRecord(
            error_id="err-123",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
            component="test",
            operation="test_op",
            exception_type="TestError",
            exception_message="Test message",
            stack_trace="test stack trace",
            context=context,
        )

        error_dict = error_record.to_dict()

        assert error_dict["error_id"] == "err-123"
        assert error_dict["severity"] == "high"
        assert error_dict["category"] == "network"
        assert error_dict["component"] == "test"


class TestSentryIntegration:
    """Test Sentry integration."""

    def test_sentry_tracker_initialization(self):
        """Test Sentry tracker initialization."""
        from ai_engine.core.sentry_integration import SentryErrorTracker

        tracker = SentryErrorTracker(
            dsn="https://test@sentry.io/123", environment="test", traces_sample_rate=0.1
        )

        assert tracker.dsn == "https://test@sentry.io/123"
        assert tracker.environment == "test"
        assert tracker.traces_sample_rate == 0.1

    def test_severity_mapping(self):
        """Test error severity to Sentry level mapping."""
        from ai_engine.core.sentry_integration import SentryErrorTracker

        tracker = SentryErrorTracker()

        assert tracker._severity_to_sentry_level(ErrorSeverity.LOW) == "info"
        assert tracker._severity_to_sentry_level(ErrorSeverity.MEDIUM) == "warning"
        assert tracker._severity_to_sentry_level(ErrorSeverity.HIGH) == "error"
        assert tracker._severity_to_sentry_level(ErrorSeverity.CRITICAL) == "fatal"


class TestCircuitBreaker:
    """Test circuit breaker implementation."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        async def success_func():
            return "success"

        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state.value == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        config = CircuitBreakerConfig(
            failure_threshold=3, expected_exception_types={ConnectionError}
        )
        cb = CircuitBreaker(config)

        async def failing_func():
            raise ConnectionError("Test error")

        # Trigger failures
        for _ in range(3):
            try:
                await cb.call(failing_func)
            except ConnectionError:
                pass

        assert cb.state.value == "open"
        assert cb.failure_count >= 3


class TestRateLimiter:
    """Test rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        from ai_engine.api.rate_limiter import RedisRateLimiter, RateLimitConfig

        config = RateLimitConfig(requests_per_minute=100, burst_size=200)

        limiter = RedisRateLimiter(redis_url="redis://localhost:6379/0", config=config)

        assert limiter.config.requests_per_minute == 100
        assert limiter.config.burst_size == 200

    @pytest.mark.asyncio
    async def test_rate_limit_strategies(self):
        """Test different rate limiting strategies."""
        from ai_engine.api.rate_limiter import RateLimitStrategy

        strategies = [
            RateLimitStrategy.FIXED_WINDOW,
            RateLimitStrategy.SLIDING_WINDOW,
            RateLimitStrategy.TOKEN_BUCKET,
            RateLimitStrategy.LEAKY_BUCKET,
        ]

        assert len(strategies) == 4
        assert RateLimitStrategy.SLIDING_WINDOW.value == "sliding_window"


class TestConfigValidator:
    """Test configuration validation."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = ConfigValidator(environment="production")
        assert validator.environment == "production"
        assert len(validator.issues) == 0

    def test_database_validation(self):
        """Test database configuration validation."""
        validator = ConfigValidator(environment="production")

        # Invalid config - missing SSL
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "test",
                "username": "test",
                "ssl_mode": "disable",
            }
        }

        validator._validate_database(config)

        # Should have error about SSL
        ssl_errors = [
            issue for issue in validator.issues if "ssl" in issue.message.lower()
        ]
        assert len(ssl_errors) > 0

    def test_security_validation(self):
        """Test security configuration validation."""
        validator = ConfigValidator(environment="production")

        # Invalid config - no TLS
        config = {
            "security": {
                "tls_enabled": False,
                "api_key_enabled": False,
                "jwt_enabled": False,
            }
        }

        validator._validate_security(config)

        # Should have multiple errors
        assert len(validator.issues) > 0

        # Check for TLS error
        tls_errors = [
            issue
            for issue in validator.issues
            if issue.severity == ValidationSeverity.ERROR
            and "tls" in issue.field.lower()
        ]
        assert len(tls_errors) > 0

    def test_hardcoded_secrets_detection(self):
        """Test detection of hardcoded secrets."""
        validator = ConfigValidator(environment="production")

        config = {
            "database": {"password": "hardcoded-password"},  # Should be detected
            "security": {"jwt_secret": "my-secret-key"},  # Should be detected
        }

        validator._check_hardcoded_secrets(config)

        # Should detect hardcoded secrets
        secret_errors = [
            issue for issue in validator.issues if "hardcoded" in issue.message.lower()
        ]
        assert len(secret_errors) >= 2


class TestSecurityHeaders:
    """Test security headers middleware."""

    @pytest.mark.asyncio
    async def test_security_headers_added(self):
        """Test that security headers are added to responses."""
        from fastapi import FastAPI, Response
        from fastapi.testclient import TestClient
        from ai_engine.api.middleware import SecurityHeadersMiddleware

        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware, enable_hsts=True, enable_csp=True)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "Strict-Transport-Security" in response.headers
        assert "Content-Security-Policy" in response.headers


class TestErrorHandler:
    """Test enhanced error handler."""

    @pytest.mark.asyncio
    async def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler(enable_persistent_storage=True, enable_sentry=True)

        assert handler.enable_persistent_storage is True
        assert handler.enable_sentry is True
        assert len(handler.error_records) == 0

    @pytest.mark.asyncio
    async def test_error_classification(self):
        """Test error classification."""
        handler = ErrorHandler()

        # Test different exception types
        severity, category, strategy = handler.classify_error(ConnectionError())
        assert severity == ErrorSeverity.MEDIUM
        assert category == ErrorCategory.NETWORK
        assert strategy == RecoveryStrategy.RETRY

        severity, category, strategy = handler.classify_error(MemoryError())
        assert severity == ErrorSeverity.CRITICAL
        assert category == ErrorCategory.RESOURCE_EXHAUSTION
        assert strategy == RecoveryStrategy.CIRCUIT_BREAK

    @pytest.mark.asyncio
    async def test_error_aggregation(self):
        """Test error aggregation."""
        handler = ErrorHandler(enable_persistent_storage=False)

        # Add some test errors
        for i in range(5):
            context = ErrorContext(component="test", operation="test_op")
            error = ErrorRecord(
                error_id=f"err-{i}",
                timestamp=time.time(),
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.NETWORK,
                component="test",
                operation="test_op",
                exception_type="TestError",
                exception_message=f"Error {i}",
                stack_trace="test",
                context=context,
            )
            handler.error_records.append(error)

        # Get aggregated errors
        errors = await handler.get_aggregated_errors(
            component="test", time_window_hours=1
        )

        assert len(errors) == 5


class TestProductionConfig:
    """Test production configuration."""

    def test_production_config_exists(self):
        """Test that production config file exists."""
        from pathlib import Path

        config_path = Path("config/environments/production.yaml")
        assert config_path.exists()

    def test_production_config_valid(self):
        """Test that production config is valid."""
        import yaml
        from pathlib import Path

        config_path = Path("config/environments/production.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["environment"] == "production"
        assert config["debug"] is False
        assert config["security"]["tls_enabled"] is True
        assert config["rate_limiting"]["enabled"] is True


class TestTLSConfiguration:
    """Test TLS/SSL configuration."""

    def test_tls_config_exists(self):
        """Test that TLS config exists."""
        from pathlib import Path

        tls_config = Path("ops/deploy/kubernetes/production/tls-config.yaml")
        assert tls_config.exists()

    def test_tls_config_valid(self):
        """Test TLS configuration is valid."""
        import yaml
        from pathlib import Path

        config_path = Path("ops/deploy/kubernetes/production/tls-config.yaml")
        with open(config_path) as f:
            configs = list(yaml.safe_load_all(f))

        # Should have multiple resources
        assert len(configs) > 0

        # Check for ClusterIssuer
        issuers = [c for c in configs if c.get("kind") == "ClusterIssuer"]
        assert len(issuers) > 0

        # Check for Certificate
        certs = [c for c in configs if c.get("kind") == "Certificate"]
        assert len(certs) > 0


# Integration Tests
class TestIntegration:
    """Integration tests for production readiness."""

    @pytest.mark.asyncio
    async def test_full_error_handling_flow(self):
        """Test complete error handling flow."""
        handler = ErrorHandler(enable_persistent_storage=False, enable_sentry=False)

        context = ErrorContext(
            component="integration_test",
            operation="test_operation",
            request_id="req-123",
        )

        # Simulate error
        try:
            raise ConnectionError("Test connection error")
        except Exception as e:
            recovery_successful, result = await handler.handle_error(e, context, None)

        # Verify error was recorded
        assert len(handler.error_records) > 0
        latest_error = handler.error_records[-1]
        assert latest_error.component == "integration_test"
        assert latest_error.exception_type == "ConnectionError"

    def test_config_validation_flow(self):
        """Test configuration validation flow."""
        validator = ConfigValidator(environment="production")

        # Valid production config
        valid_config = {
            "environment": "production",
            "debug": False,
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "test",
                "username": "test",
                "ssl_mode": "require",
                "connection_pool_size": 20,
            },
            "security": {
                "tls_enabled": True,
                "api_key_enabled": True,
                "jwt_enabled": True,
                "audit_logging_enabled": True,
            },
            "rate_limiting": {"enabled": True, "default_limit": 100},
            "monitoring": {
                "metrics_enabled": True,
                "prometheus_enabled": True,
                "alerting_enabled": True,
            },
        }

        is_valid, issues = validator.validate_config(valid_config)

        # Should have some warnings but no errors
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0 or all("password" in e.field.lower() for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
