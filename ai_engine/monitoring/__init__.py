"""
QBITEL Engine - Monitoring Module

This module provides comprehensive monitoring, logging, and observability
capabilities for the AI Engine.
"""

__all__ = []

try:
    from .metrics import (
        MetricsCollector,
        PrometheusMetrics,
        CustomMetrics,
        AIEngineMetrics,
    )

    __all__.extend(
        [
            "MetricsCollector",
            "PrometheusMetrics",
            "CustomMetrics",
            "AIEngineMetrics",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    MetricsCollector = None  # type: ignore

try:
    from .logging import (
        StructuredLogger,
        LoggerConfig,
        setup_logging,
        get_logger,
        TraceContext,
    )

    __all__.extend(
        [
            "StructuredLogger",
            "LoggerConfig",
            "setup_logging",
            "get_logger",
            "TraceContext",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    StructuredLogger = None  # type: ignore

try:
    from .health import HealthChecker, HealthStatus, ComponentHealth, SystemHealth

    __all__.extend(
        [
            "HealthChecker",
            "HealthStatus",
            "ComponentHealth",
            "SystemHealth",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    HealthChecker = None  # type: ignore

try:
    from .observability import ObservabilityManager, TracingProvider, DistributedTracing

    __all__.extend(
        [
            "ObservabilityManager",
            "TracingProvider",
            "DistributedTracing",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    ObservabilityManager = None  # type: ignore

try:
    from .alerts import AlertManager, AlertRule, AlertChannel, NotificationProvider

    __all__.extend(
        [
            "AlertManager",
            "AlertRule",
            "AlertChannel",
            "NotificationProvider",
        ]
    )
except Exception:  # pragma: no cover - optional dependency failures
    AlertManager = None  # type: ignore
