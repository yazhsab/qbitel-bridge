"""
Zero-Touch Automation Engine

Self-configuring, self-healing automation for enterprise security:
- Protocol adapter generation
- Configuration automation
- Key lifecycle management
- Certificate automation
- Self-healing orchestration
- Compliance automation

Components:
- ProtocolAdapterGenerator: Auto-generates protocol adapters from analysis
- ConfigurationManager: Self-configuring security infrastructure
- KeyLifecycleManager: Automated key rotation and management
- CertificateAutomation: Auto-renewal and certificate provisioning
- SelfHealingOrchestrator: Detects and recovers from failures
- AutomationEngine: Main orchestration engine combining all components
"""

from ai_engine.automation.protocol_adapter_generator import (
    ProtocolAdapterGenerator,
    GeneratedAdapter,
    AdapterConfig,
    FieldMapping,
    TransformationType,
)
from ai_engine.automation.configuration_manager import (
    ConfigurationManager,
    SecurityConfiguration,
    ConfigurationChange,
    ConfigurationType,
    ConfigurationSource,
)
from ai_engine.automation.key_lifecycle_manager import (
    KeyLifecycleManager,
    KeyPolicy,
    KeyMetadata,
    KeyRotationEvent,
    KeyState,
    KeyType,
)
from ai_engine.automation.certificate_automation import (
    CertificateAutomation,
    CertificatePolicy,
    Certificate,
    CertificateRequest,
    RenewalEvent,
    CertificateType,
    CertificateState,
    RevocationReason,
)
from ai_engine.automation.self_healing_orchestrator import (
    SelfHealingOrchestrator,
    HealthCheck,
    HealthResult,
    HealthStatus,
    ComponentState,
    ComponentType,
    RecoveryAction,
    RecoveryPlan,
    RecoveryEvent,
    Incident,
    IncidentSeverity,
    CircuitBreaker,
)
from ai_engine.automation.automation_engine import (
    AutomationEngine,
    AutomationConfig,
    AutomationStatus,
    AutomationTask,
    create_automation_engine,
)

__all__ = [
    # Protocol adapters
    "ProtocolAdapterGenerator",
    "GeneratedAdapter",
    "AdapterConfig",
    "FieldMapping",
    "TransformationType",
    # Configuration
    "ConfigurationManager",
    "SecurityConfiguration",
    "ConfigurationChange",
    "ConfigurationType",
    "ConfigurationSource",
    # Key management
    "KeyLifecycleManager",
    "KeyPolicy",
    "KeyMetadata",
    "KeyRotationEvent",
    "KeyState",
    "KeyType",
    # Certificates
    "CertificateAutomation",
    "CertificatePolicy",
    "Certificate",
    "CertificateRequest",
    "RenewalEvent",
    "CertificateType",
    "CertificateState",
    "RevocationReason",
    # Self-healing
    "SelfHealingOrchestrator",
    "HealthCheck",
    "HealthResult",
    "HealthStatus",
    "ComponentState",
    "ComponentType",
    "RecoveryAction",
    "RecoveryPlan",
    "RecoveryEvent",
    "Incident",
    "IncidentSeverity",
    "CircuitBreaker",
    # Main engine
    "AutomationEngine",
    "AutomationConfig",
    "AutomationStatus",
    "AutomationTask",
    "create_automation_engine",
]
