"""
CRONOS AI Engine - Zero-Touch Security Orchestrator

Enterprise-grade autonomous security response system with legacy protocol awareness.
"""

__all__ = []

# Core Models
try:
    from .models import (
        SecurityEvent,
        ThreatAnalysis,
        AutomatedResponse,
        SecurityContext,
        LegacySystem,
        QuarantineResult,
        ResponseAction,
        SecurityMetrics,
        ThreatIntelligence,
    )
    __all__.extend([
        "SecurityEvent",
        "ThreatAnalysis",
        "AutomatedResponse",
        "SecurityContext",
        "LegacySystem",
        "QuarantineResult",
        "ResponseAction",
        "SecurityMetrics",
        "ThreatIntelligence",
    ])
except Exception:  # pragma: no cover
    pass

# Core Services
try:
    from .decision_engine import ZeroTouchDecisionEngine
    __all__.append("ZeroTouchDecisionEngine")
except Exception:  # pragma: no cover
    ZeroTouchDecisionEngine = None  # type: ignore

try:
    from .legacy_response import LegacyAwareResponseManager
    __all__.append("LegacyAwareResponseManager")
except Exception:  # pragma: no cover
    LegacyAwareResponseManager = None  # type: ignore

try:
    from .threat_analyzer import ThreatAnalyzer
    __all__.append("ThreatAnalyzer")
except Exception:  # pragma: no cover
    ThreatAnalyzer = None  # type: ignore

try:
    from .security_service import SecurityOrchestratorService
    __all__.append("SecurityOrchestratorService")
except Exception:  # pragma: no cover
    SecurityOrchestratorService = None  # type: ignore

# Integration Components
try:
    from .integrations import (
        BaseIntegrationConnector,
        IntegrationType,
        IntegrationConfig,
        IntegrationResult,
        SIEMConnector,
        SplunkConnector,
        QRadarConnector,
        TicketingConnector,
        ServiceNowConnector,
        JiraConnector,
        CommunicationConnector,
        SlackConnector,
        EmailConnector,
        NetworkSecurityConnector,
        FirewallConnector,
        IDSConnector,
        IntegrationManager,
        get_integration_manager,
    )
    __all__.extend([
        "BaseIntegrationConnector",
        "IntegrationType",
        "IntegrationConfig",
        "IntegrationResult",
        "SIEMConnector",
        "SplunkConnector",
        "QRadarConnector",
        "TicketingConnector",
        "ServiceNowConnector",
        "JiraConnector",
        "CommunicationConnector",
        "SlackConnector",
        "EmailConnector",
        "NetworkSecurityConnector",
        "FirewallConnector",
        "IDSConnector",
        "IntegrationManager",
        "get_integration_manager",
    ])
except Exception:  # pragma: no cover
    pass

# Resilience Components
try:
    from .resilience import (
        # Circuit Breaker
        CircuitBreaker,
        CircuitBreakerState,
        CircuitBreakerConfig,
        CircuitBreakerManager,
        get_circuit_breaker_manager,
        # Retry Policies
        RetryPolicy,
        ExponentialBackoff,
        LinearBackoff,
        FixedBackoff,
        RetryManager,
        CommonRetryPolicies,
        get_retry_manager,
        # Bulkhead Isolation
        BulkheadManager,
        ResourcePool,
        ResourceContext,
        AllocationStrategy,
        get_bulkhead_manager,
        # Health Checking
        HealthChecker,
        HealthCheck,
        BasicHealthCheck,
        HTTPHealthCheck,
        DatabaseHealthCheck,
        HealthStatus,
        CheckType,
        get_health_checker,
        # Timeout Management
        TimeoutManager,
        TimeoutPolicy,
        TimeoutContext,
        TimeoutStrategy,
        timeout_context,
        get_timeout_manager,
        # Error Recovery
        ErrorRecoveryManager,
        RecoveryStrategy,
        RecoveryPlan,
        RecoveryAction,
        ErrorSeverity,
        get_error_recovery_manager,
        # Main Resilience Manager
        ResilienceManager,
        ResilienceConfig,
        ResilienceLevel,
        get_resilience_manager,
    )
    __all__.extend([
        # Circuit Breaker
        "CircuitBreaker",
        "CircuitBreakerState",
        "CircuitBreakerConfig",
        "CircuitBreakerManager",
        "get_circuit_breaker_manager",
        # Retry Policies
        "RetryPolicy",
        "ExponentialBackoff",
        "LinearBackoff",
        "FixedBackoff",
        "RetryManager",
        "CommonRetryPolicies",
        "get_retry_manager",
        # Bulkhead Isolation
        "BulkheadManager",
        "ResourcePool",
        "ResourceContext",
        "AllocationStrategy",
        "get_bulkhead_manager",
        # Health Checking
        "HealthChecker",
        "HealthCheck",
        "BasicHealthCheck",
        "HTTPHealthCheck",
        "DatabaseHealthCheck",
        "HealthStatus",
        "CheckType",
        "get_health_checker",
        # Timeout Management
        "TimeoutManager",
        "TimeoutPolicy",
        "TimeoutContext",
        "TimeoutStrategy",
        "timeout_context",
        "get_timeout_manager",
        # Error Recovery
        "ErrorRecoveryManager",
        "RecoveryStrategy",
        "RecoveryPlan",
        "RecoveryAction",
        "ErrorSeverity",
        "get_error_recovery_manager",
        # Main Resilience Manager
        "ResilienceManager",
        "ResilienceConfig",
        "ResilienceLevel",
        "get_resilience_manager",
    ])
except Exception:  # pragma: no cover
    pass
