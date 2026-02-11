"""
Production Observability Module

Comprehensive observability for enterprise security platforms:
- Structured logging with correlation
- Metrics collection (Prometheus-compatible)
- Distributed tracing (OpenTelemetry-compatible)
- Health checks and readiness probes
- Audit logging for compliance

Components:
- MetricsCollector: Prometheus-style metrics
- TracingProvider: Distributed tracing
- StructuredLogger: Correlation-aware logging
- HealthRegistry: Health and readiness checks
- AuditLogger: Compliance-ready audit trails

Usage:
    from ai_engine.observability import initialize_observability, get_logger

    # Initialize all observability components at startup
    await initialize_observability(config)

    # Get structured logger
    logger = get_logger(__name__)
    logger.info("Operation completed", extra={"user_id": "123"})
"""

import os
import logging
from typing import Any, Dict, Optional

from ai_engine.observability.metrics import (
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    Summary,
    MetricLabels,
    get_metrics_collector,
)
from ai_engine.observability.tracing import (
    TracingProvider,
    Span,
    SpanContext,
    trace_method,
    get_tracer,
)
from ai_engine.observability.logging import (
    StructuredLogger,
    LogLevel,
    LogContext,
    get_logger,
    configure_logging,
)
from ai_engine.observability.health import (
    HealthRegistry,
    HealthCheck,
    HealthStatus,
    LivenessCheck,
    ReadinessCheck,
    get_health_registry,
)
from ai_engine.observability.audit import (
    AuditLogger,
    AuditEvent,
    AuditCategory,
    AuditSeverity,
    AuditOutcome,
    AuditActor,
    AuditResource,
    get_audit_logger,
)
from ai_engine.observability.tracing import (
    SpanKind,
    SpanStatus,
)

__all__ = [
    # Initialization
    "initialize_observability",
    "shutdown_observability",
    "ObservabilityConfig",
    # Metrics
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "MetricLabels",
    "get_metrics_collector",
    # Tracing
    "TracingProvider",
    "Span",
    "SpanContext",
    "trace_method",
    "get_tracer",
    # Logging
    "StructuredLogger",
    "LogLevel",
    "LogContext",
    "get_logger",
    "configure_logging",
    # Health
    "HealthRegistry",
    "HealthCheck",
    "HealthStatus",
    "LivenessCheck",
    "ReadinessCheck",
    "get_health_registry",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditCategory",
    "AuditSeverity",
    "AuditOutcome",
    "AuditActor",
    "AuditResource",
    "get_audit_logger",
    # Tracing extras
    "SpanKind",
    "SpanStatus",
]


# =============================================================================
# Configuration
# =============================================================================


class ObservabilityConfig:
    """Configuration for observability components."""

    def __init__(
        self,
        service_name: str = "qbitel",
        environment: str = None,
        log_level: str = None,
        log_format: str = None,
        tracing_enabled: bool = True,
        metrics_enabled: bool = True,
        audit_enabled: bool = True,
        jaeger_endpoint: str = None,
        otlp_endpoint: str = None,
        sentry_dsn: str = None,
    ):
        self.service_name = service_name
        self.environment = environment or os.getenv("QBITEL_AI_ENVIRONMENT", "production")
        self.log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
        self.log_format = log_format or os.getenv(
            "LOG_FORMAT", "json" if self.environment == "production" else "text"
        )
        self.tracing_enabled = tracing_enabled and os.getenv("TRACING_ENABLED", "true").lower() == "true"
        self.metrics_enabled = metrics_enabled
        self.audit_enabled = audit_enabled
        self.jaeger_endpoint = jaeger_endpoint or os.getenv("JAEGER_ENDPOINT")
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTLP_ENDPOINT")
        self.sentry_dsn = sentry_dsn or os.getenv("SENTRY_DSN")


# =============================================================================
# Global State
# =============================================================================

_initialized = False
_config: Optional[ObservabilityConfig] = None
_metrics: Optional[MetricsCollector] = None
_tracer: Optional[TracingProvider] = None
_audit: Optional[AuditLogger] = None
_health: Optional[HealthRegistry] = None


# =============================================================================
# Initialization
# =============================================================================


async def initialize_observability(
    config: ObservabilityConfig = None,
    service_name: str = "qbitel",
    environment: str = None,
) -> Dict[str, Any]:
    """
    Initialize all observability components.

    This should be called once at application startup.

    Args:
        config: ObservabilityConfig instance (optional)
        service_name: Service name for tracing/metrics
        environment: Deployment environment

    Returns:
        Dictionary with initialization status of each component
    """
    global _initialized, _config, _metrics, _tracer, _audit, _health

    if _initialized:
        return {"status": "already_initialized"}

    # Create or use provided config
    if config is None:
        config = ObservabilityConfig(
            service_name=service_name,
            environment=environment,
        )
    _config = config

    results = {}
    _logger = logging.getLogger(__name__)

    # 1. Configure structured logging
    try:
        configure_logging(
            level=config.log_level,
            json_output=config.log_format == "json",
        )
        results["logging"] = {
            "status": "initialized",
            "level": config.log_level,
            "format": config.log_format,
        }
        _logger.info(f"Logging configured: level={config.log_level}, format={config.log_format}")
    except Exception as e:
        results["logging"] = {"status": "error", "error": str(e)}
        _logger.error(f"Failed to configure logging: {e}")

    # 2. Initialize metrics collector
    if config.metrics_enabled:
        try:
            _metrics = get_metrics_collector()

            # Register standard application metrics
            _register_standard_metrics(_metrics)

            results["metrics"] = {
                "status": "initialized",
                "namespace": config.service_name,
            }
            _logger.info("Metrics collector initialized")
        except Exception as e:
            results["metrics"] = {"status": "error", "error": str(e)}
            _logger.error(f"Failed to initialize metrics: {e}")
    else:
        results["metrics"] = {"status": "disabled"}

    # 3. Initialize distributed tracing
    if config.tracing_enabled:
        try:
            _tracer = get_tracer()
            results["tracing"] = {
                "status": "initialized",
                "service_name": config.service_name,
            }
            _logger.info("Distributed tracing initialized")
        except Exception as e:
            results["tracing"] = {"status": "error", "error": str(e)}
            _logger.error(f"Failed to initialize tracing: {e}")
    else:
        results["tracing"] = {"status": "disabled"}

    # 4. Initialize audit logger
    if config.audit_enabled:
        try:
            _audit = get_audit_logger()
            results["audit"] = {"status": "initialized"}
            _logger.info("Audit logger initialized")
        except Exception as e:
            results["audit"] = {"status": "error", "error": str(e)}
            _logger.error(f"Failed to initialize audit logger: {e}")
    else:
        results["audit"] = {"status": "disabled"}

    # 5. Initialize health registry
    try:
        _health = get_health_registry()

        # Register standard health checks
        _register_standard_health_checks(_health)

        results["health"] = {"status": "initialized"}
        _logger.info("Health registry initialized")
    except Exception as e:
        results["health"] = {"status": "error", "error": str(e)}
        _logger.error(f"Failed to initialize health registry: {e}")

    # 6. Initialize Sentry error tracking (if configured)
    if config.sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

            sentry_sdk.init(
                dsn=config.sentry_dsn,
                environment=config.environment,
                traces_sample_rate=0.1 if config.environment == "production" else 1.0,
                profiles_sample_rate=0.1 if config.environment == "production" else 1.0,
                integrations=[
                    FastApiIntegration(),
                    SqlalchemyIntegration(),
                ],
            )
            results["sentry"] = {"status": "initialized"}
            _logger.info("Sentry error tracking initialized")
        except ImportError:
            results["sentry"] = {"status": "skipped", "reason": "sentry-sdk not installed"}
        except Exception as e:
            results["sentry"] = {"status": "error", "error": str(e)}
            _logger.error(f"Failed to initialize Sentry: {e}")
    else:
        results["sentry"] = {"status": "disabled", "reason": "SENTRY_DSN not configured"}

    _initialized = True

    _logger.info(
        f"Observability stack initialized for {config.service_name} ({config.environment})",
        extra={"components": results},
    )

    return {
        "status": "initialized",
        "service_name": config.service_name,
        "environment": config.environment,
        "components": results,
    }


async def shutdown_observability():
    """Shutdown all observability components gracefully."""
    global _initialized, _metrics, _tracer, _audit, _health

    if not _initialized:
        return

    _logger = logging.getLogger(__name__)
    _logger.info("Shutting down observability stack")

    # Flush metrics
    if _metrics:
        try:
            _metrics.flush()
            _logger.debug("Metrics flushed")
        except Exception as e:
            _logger.error(f"Error flushing metrics: {e}")

    # Flush tracing spans
    if _tracer:
        try:
            _tracer.flush()
            _logger.debug("Traces flushed")
        except Exception as e:
            _logger.error(f"Error flushing traces: {e}")

    # Flush audit logs
    if _audit:
        try:
            _audit.flush()
            _logger.debug("Audit logs flushed")
        except Exception as e:
            _logger.error(f"Error flushing audit logs: {e}")

    _initialized = False
    _logger.info("Observability stack shutdown complete")


# =============================================================================
# Standard Metrics Registration
# =============================================================================


def _register_standard_metrics(metrics: MetricsCollector):
    """Register standard application metrics."""
    # HTTP metrics
    metrics.create_counter(
        "http_requests",
        "Total HTTP requests",
        labels=["method", "endpoint", "status_code"],
    )
    metrics.create_histogram(
        "http_request_duration_seconds",
        "HTTP request duration",
        labels=["method", "endpoint"],
    )

    # LLM metrics
    metrics.create_counter(
        "llm_requests",
        "Total LLM requests",
        labels=["provider", "model", "status"],
    )
    metrics.create_histogram(
        "llm_request_duration_seconds",
        "LLM request duration",
        labels=["provider", "model"],
    )
    metrics.create_counter(
        "llm_tokens",
        "Total LLM tokens used",
        labels=["provider", "model", "direction"],
    )

    # Security metrics
    metrics.create_counter(
        "security_events",
        "Total security events",
        labels=["event_type", "severity"],
    )
    metrics.create_counter(
        "security_decisions",
        "Total security decisions",
        labels=["decision_type", "action"],
    )
    metrics.create_gauge(
        "active_threats",
        "Number of active threats",
    )

    # Protocol discovery metrics
    metrics.create_counter(
        "protocol_discoveries",
        "Total protocol discoveries",
        labels=["protocol_type", "status"],
    )
    metrics.create_histogram(
        "discovery_duration_seconds",
        "Protocol discovery duration",
        labels=["protocol_type"],
    )

    # Database metrics
    metrics.create_gauge(
        "db_connections_active",
        "Active database connections",
    )
    metrics.create_counter(
        "db_queries",
        "Total database queries",
        labels=["operation", "table"],
    )


def _register_standard_health_checks(health: HealthRegistry):
    """Register standard health checks."""
    # These are placeholder registrations - actual check functions
    # should be provided by the respective service modules
    pass
