"""
CRONOS AI Engine - Security Orchestrator Configuration Management

Enterprise-grade configuration management with environment-specific overrides and validation.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pathlib import Path

from ..core.config import Config
from ..core.exceptions import ConfigurationException


class SecurityConfigError(ConfigurationException):
    """Security configuration specific error."""

    pass


class QuarantineMethod(str, Enum):
    """Available quarantine methods."""

    NETWORK_SEGMENTATION = "network_segmentation"
    VLAN_ISOLATION = "vlan_isolation"
    FIREWALL_RULES = "firewall_rules"
    ENHANCED_MONITORING = "enhanced_monitoring"
    MONITORING_ONLY = "monitoring_only"


class IntegrationType(str, Enum):
    """Integration types."""

    SIEM = "siem"
    TICKETING = "ticketing"
    COMMUNICATIONS = "communications"
    NETWORK_SECURITY = "network_security"


@dataclass
class ConfidenceThresholds:
    """Confidence thresholds for automated decisions."""

    auto_execute: float = 0.95
    auto_approve: float = 0.85
    escalate_threshold: float = 0.50

    def validate(self) -> None:
        """Validate threshold values."""
        thresholds = [self.auto_execute, self.auto_approve, self.escalate_threshold]

        for threshold in thresholds:
            if not 0.0 <= threshold <= 1.0:
                raise SecurityConfigError(f"Invalid confidence threshold: {threshold}")

        if self.auto_execute < self.auto_approve:
            raise SecurityConfigError(
                "auto_execute threshold must be >= auto_approve threshold"
            )

        if self.auto_approve < self.escalate_threshold:
            raise SecurityConfigError(
                "auto_approve threshold must be >= escalate_threshold"
            )


@dataclass
class ResponseRiskScores:
    """Risk scores for different response types."""

    alert_security_team: float = 0.0
    enable_monitoring: float = 0.1
    log_retention_increase: float = 0.1
    block_ip: float = 0.3
    deploy_honeypot: float = 0.3
    redirect_traffic: float = 0.4
    virtual_patch: float = 0.5
    reset_credentials: float = 0.6
    network_segmentation: float = 0.7
    quarantine: float = 0.8
    isolate_system: float = 0.8
    shutdown_service: float = 0.9
    patch_vulnerability: float = 0.9

    def validate(self) -> None:
        """Validate risk scores."""
        for field_name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise SecurityConfigError(
                    f"Invalid risk score for {field_name}: {value}"
                )


@dataclass
class DecisionEngineConfig:
    """Configuration for the decision engine."""

    confidence_thresholds: ConfidenceThresholds = field(
        default_factory=ConfidenceThresholds
    )
    response_risk_scores: ResponseRiskScores = field(default_factory=ResponseRiskScores)
    enable_learning: bool = True
    learning_rate: float = 0.01
    min_historical_samples: int = 100
    llm_timeout_seconds: int = 30
    llm_retry_attempts: int = 3
    fallback_to_rules: bool = True

    def validate(self) -> None:
        """Validate configuration."""
        self.confidence_thresholds.validate()
        self.response_risk_scores.validate()

        if not 0.0 < self.learning_rate <= 1.0:
            raise SecurityConfigError(f"Invalid learning rate: {self.learning_rate}")

        if self.min_historical_samples < 10:
            raise SecurityConfigError("min_historical_samples must be at least 10")


@dataclass
class ProtocolSettings:
    """Protocol-specific settings."""

    max_downtime_seconds: Optional[int] = None
    preserve_message_integrity: bool = False
    require_graceful_shutdown: bool = False
    transaction_integrity_check: bool = False
    rollback_on_failure: bool = False
    no_network_disruption: bool = False
    industrial_safety_mode: bool = False
    session_preservation: bool = False
    mainframe_aware_operations: bool = False
    max_concurrent_changes: int = 1
    prohibited_actions: List[str] = field(default_factory=list)


@dataclass
class DependencyAnalysisConfig:
    """Dependency analysis configuration."""

    enabled: bool = True
    max_dependency_depth: int = 5
    critical_path_protection: bool = True
    cascade_impact_threshold: float = 0.7

    def validate(self) -> None:
        """Validate configuration."""
        if self.max_dependency_depth < 1 or self.max_dependency_depth > 10:
            raise SecurityConfigError("max_dependency_depth must be between 1 and 10")

        if not 0.0 <= self.cascade_impact_threshold <= 1.0:
            raise SecurityConfigError(
                "cascade_impact_threshold must be between 0.0 and 1.0"
            )


@dataclass
class LegacyResponseConfig:
    """Configuration for legacy system responses."""

    max_concurrent_quarantines: int = 5
    quarantine_timeout_hours: int = 72
    require_approval_for_critical: bool = True
    protocol_settings: Dict[str, ProtocolSettings] = field(default_factory=dict)
    dependency_analysis: DependencyAnalysisConfig = field(
        default_factory=DependencyAnalysisConfig
    )
    quarantine_methods: Dict[str, QuarantineMethod] = field(
        default_factory=lambda: {
            "default": QuarantineMethod.NETWORK_SEGMENTATION,
            "critical_systems": QuarantineMethod.ENHANCED_MONITORING,
            "legacy_industrial": QuarantineMethod.MONITORING_ONLY,
        }
    )

    def validate(self) -> None:
        """Validate configuration."""
        if self.max_concurrent_quarantines < 1:
            raise SecurityConfigError("max_concurrent_quarantines must be at least 1")

        if (
            self.quarantine_timeout_hours < 1 or self.quarantine_timeout_hours > 720
        ):  # Max 30 days
            raise SecurityConfigError(
                "quarantine_timeout_hours must be between 1 and 720"
            )

        self.dependency_analysis.validate()


@dataclass
class MLModelConfig:
    """ML model configuration."""

    enabled: bool = True
    model_path: str = ""
    confidence_threshold: float = 0.7
    batch_size: int = 32
    update_frequency_hours: int = 24

    def validate(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise SecurityConfigError(
                "confidence_threshold must be between 0.0 and 1.0"
            )

        if self.batch_size < 1 or self.batch_size > 1024:
            raise SecurityConfigError("batch_size must be between 1 and 1024")


@dataclass
class ThreatIntelligenceConfig:
    """Threat intelligence configuration."""

    enabled: bool = True
    feeds: List[Dict[str, Any]] = field(default_factory=list)
    correlation_threshold: float = 0.8
    max_intelligence_age_days: int = 30
    auto_expire_indicators: bool = True

    def validate(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.correlation_threshold <= 1.0:
            raise SecurityConfigError(
                "correlation_threshold must be between 0.0 and 1.0"
            )

        if self.max_intelligence_age_days < 1 or self.max_intelligence_age_days > 365:
            raise SecurityConfigError(
                "max_intelligence_age_days must be between 1 and 365"
            )


@dataclass
class ThreatAnalyzerConfig:
    """Configuration for threat analyzer."""

    ml_models: Dict[str, MLModelConfig] = field(default_factory=dict)
    feature_extraction: Dict[str, Any] = field(default_factory=dict)
    threat_intelligence: ThreatIntelligenceConfig = field(
        default_factory=ThreatIntelligenceConfig
    )
    context_analysis: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""
        for model_name, model_config in self.ml_models.items():
            model_config.validate()

        self.threat_intelligence.validate()


@dataclass
class BusinessImpactConfig:
    """Business impact assessment configuration."""

    financial_models: Dict[str, Any] = field(default_factory=dict)
    system_criticality: Dict[str, Any] = field(default_factory=dict)
    compliance_frameworks: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""
        # Validate financial models
        if "base_incident_cost" in self.financial_models:
            if self.financial_models["base_incident_cost"] < 0:
                raise SecurityConfigError("base_incident_cost cannot be negative")


@dataclass
class IntegrationConfig:
    """Integration configuration."""

    siem: Dict[str, Any] = field(default_factory=dict)
    ticketing: Dict[str, Any] = field(default_factory=dict)
    communications: Dict[str, Any] = field(default_factory=dict)
    network_security: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate integration configuration."""
        # Validate SIEM config
        if self.siem.get("enabled", False):
            required_fields = ["connector_type", "endpoint"]
            for field in required_fields:
                if not self.siem.get(field):
                    raise SecurityConfigError(
                        f"SIEM integration missing required field: {field}"
                    )


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""

    prometheus: Dict[str, Any] = field(default_factory=dict)
    grafana: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    health_checks: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate monitoring configuration."""
        # Validate Prometheus config
        if self.prometheus.get("enabled", False):
            port = self.prometheus.get("port", 9090)
            if not isinstance(port, int) or port < 1024 or port > 65535:
                raise SecurityConfigError(
                    "Prometheus port must be between 1024 and 65535"
                )


@dataclass
class PerformanceConfig:
    """Performance and scaling configuration."""

    max_memory_mb: int = 4096
    max_cpu_cores: int = 4
    thread_pool_size: int = 10
    analysis_cache: Dict[str, Any] = field(default_factory=dict)
    threat_intel_cache: Dict[str, Any] = field(default_factory=dict)
    rate_limits: Dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate performance configuration."""
        if self.max_memory_mb < 512:
            raise SecurityConfigError("max_memory_mb must be at least 512")

        if self.max_cpu_cores < 1:
            raise SecurityConfigError("max_cpu_cores must be at least 1")

        if self.thread_pool_size < 1 or self.thread_pool_size > 100:
            raise SecurityConfigError("thread_pool_size must be between 1 and 100")


@dataclass
class SecurityConfig:
    """Security and compliance configuration."""

    encryption: Dict[str, Any] = field(default_factory=dict)
    auth: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)
    privacy: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate security configuration."""
        # Validate authentication config
        if self.auth.get("required", True):
            if not self.auth.get("method"):
                raise SecurityConfigError(
                    "Authentication method must be specified when auth is required"
                )


@dataclass
class SecurityOrchestratorConfig:
    """Main security orchestrator configuration."""

    enabled: bool = True
    max_concurrent_incidents: int = 50
    incident_retention_hours: int = 24
    auto_cleanup_enabled: bool = True

    # Component configurations
    decision_engine: DecisionEngineConfig = field(default_factory=DecisionEngineConfig)
    legacy_response: LegacyResponseConfig = field(default_factory=LegacyResponseConfig)
    threat_analyzer: ThreatAnalyzerConfig = field(default_factory=ThreatAnalyzerConfig)
    business_impact: BusinessImpactConfig = field(default_factory=BusinessImpactConfig)
    integrations: IntegrationConfig = field(default_factory=IntegrationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def validate(self) -> None:
        """Validate entire configuration."""
        if self.max_concurrent_incidents < 1 or self.max_concurrent_incidents > 1000:
            raise SecurityConfigError(
                "max_concurrent_incidents must be between 1 and 1000"
            )

        if self.incident_retention_hours < 1 or self.incident_retention_hours > 720:
            raise SecurityConfigError(
                "incident_retention_hours must be between 1 and 720"
            )

        # Validate all component configurations
        self.decision_engine.validate()
        self.legacy_response.validate()
        self.threat_analyzer.validate()
        self.business_impact.validate()
        self.integrations.validate()
        self.monitoring.validate()
        self.performance.validate()
        self.security.validate()


class SecurityConfigManager:
    """
    Manages security orchestrator configuration with environment-specific overrides.
    """

    def __init__(
        self, config_path: Optional[str] = None, environment: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)

        # Determine configuration path
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default to security config in config directory
            self.config_path = (
                Path(__file__).parent.parent.parent
                / "config"
                / "security"
                / "zero-touch-security-config.yaml"
            )

        # Determine environment
        self.environment = environment or os.getenv("CRONOS_ENVIRONMENT", "development")

        # Configuration cache
        self._config_cache: Optional[SecurityOrchestratorConfig] = None
        self._config_timestamp: Optional[float] = None

        self.logger.info(
            f"Security Config Manager initialized for environment: {self.environment}"
        )

    def load_config(self, force_reload: bool = False) -> SecurityOrchestratorConfig:
        """
        Load and validate security orchestrator configuration.

        Args:
            force_reload: Force reload from disk even if cached

        Returns:
            Validated security orchestrator configuration
        """

        # Check if we can use cached config
        if not force_reload and self._config_cache and self._config_timestamp:
            if self.config_path.exists():
                file_mtime = self.config_path.stat().st_mtime
                if file_mtime <= self._config_timestamp:
                    return self._config_cache

        try:
            self.logger.info(f"Loading security config from: {self.config_path}")

            # Load YAML configuration
            with open(self.config_path, "r") as file:
                raw_config = yaml.safe_load(file)

            # Extract security orchestrator config
            security_config_dict = raw_config.get("security_orchestrator", {})

            # Apply environment-specific overrides
            environment_overrides = raw_config.get("environments", {}).get(
                self.environment, {}
            )
            if environment_overrides:
                security_config_dict = self._merge_config_dict(
                    security_config_dict,
                    environment_overrides.get("security_orchestrator", {}),
                )

            # Create configuration object
            config = self._create_config_from_dict(security_config_dict)

            # Validate configuration
            config.validate()

            # Cache the configuration
            self._config_cache = config
            self._config_timestamp = time.time() if self.config_path.exists() else None

            self.logger.info("Security configuration loaded and validated successfully")
            return config

        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise SecurityConfigError(
                f"Configuration file not found: {self.config_path}"
            )

        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise SecurityConfigError(f"Configuration file parsing error: {e}")

        except Exception as e:
            self.logger.error(f"Configuration loading error: {e}")
            raise SecurityConfigError(f"Configuration loading failed: {e}")

    def _create_config_from_dict(
        self, config_dict: Dict[str, Any]
    ) -> SecurityOrchestratorConfig:
        """Create configuration object from dictionary."""

        # Create component configurations
        decision_engine_dict = config_dict.get("decision_engine", {})
        decision_engine = DecisionEngineConfig(
            confidence_thresholds=ConfidenceThresholds(
                **decision_engine_dict.get("confidence_thresholds", {})
            ),
            response_risk_scores=ResponseRiskScores(
                **decision_engine_dict.get("response_risk_scores", {})
            ),
            enable_learning=decision_engine_dict.get("enable_learning", True),
            learning_rate=decision_engine_dict.get("learning_rate", 0.01),
            min_historical_samples=decision_engine_dict.get(
                "min_historical_samples", 100
            ),
            llm_timeout_seconds=decision_engine_dict.get("llm_timeout_seconds", 30),
            llm_retry_attempts=decision_engine_dict.get("llm_retry_attempts", 3),
            fallback_to_rules=decision_engine_dict.get("fallback_to_rules", True),
        )

        # Create legacy response config
        legacy_response_dict = config_dict.get("legacy_response", {})

        # Protocol settings
        protocol_settings = {}
        for protocol, settings in legacy_response_dict.get(
            "protocol_settings", {}
        ).items():
            protocol_settings[protocol] = ProtocolSettings(**settings)

        legacy_response = LegacyResponseConfig(
            max_concurrent_quarantines=legacy_response_dict.get(
                "max_concurrent_quarantines", 5
            ),
            quarantine_timeout_hours=legacy_response_dict.get(
                "quarantine_timeout_hours", 72
            ),
            require_approval_for_critical=legacy_response_dict.get(
                "require_approval_for_critical", True
            ),
            protocol_settings=protocol_settings,
            dependency_analysis=DependencyAnalysisConfig(
                **legacy_response_dict.get("dependency_analysis", {})
            ),
            quarantine_methods={
                k: QuarantineMethod(v)
                for k, v in legacy_response_dict.get("quarantine_methods", {}).items()
            },
        )

        # Create threat analyzer config
        threat_analyzer_dict = config_dict.get("threat_analyzer", {})

        ml_models = {}
        for model_name, model_dict in threat_analyzer_dict.get("ml_models", {}).items():
            ml_models[model_name] = MLModelConfig(**model_dict)

        threat_analyzer = ThreatAnalyzerConfig(
            ml_models=ml_models,
            feature_extraction=threat_analyzer_dict.get("feature_extraction", {}),
            threat_intelligence=ThreatIntelligenceConfig(
                **threat_analyzer_dict.get("threat_intelligence", {})
            ),
            context_analysis=threat_analyzer_dict.get("context_analysis", {}),
        )

        # Create other component configs
        business_impact = BusinessImpactConfig(**config_dict.get("business_impact", {}))
        integrations = IntegrationConfig(**config_dict.get("integrations", {}))
        monitoring = MonitoringConfig(**config_dict.get("monitoring", {}))
        performance = PerformanceConfig(**config_dict.get("performance", {}))
        security = SecurityConfig(**config_dict.get("security", {}))

        # Create main configuration
        return SecurityOrchestratorConfig(
            enabled=config_dict.get("enabled", True),
            max_concurrent_incidents=config_dict.get("max_concurrent_incidents", 50),
            incident_retention_hours=config_dict.get("incident_retention_hours", 24),
            auto_cleanup_enabled=config_dict.get("auto_cleanup_enabled", True),
            decision_engine=decision_engine,
            legacy_response=legacy_response,
            threat_analyzer=threat_analyzer,
            business_impact=business_impact,
            integrations=integrations,
            monitoring=monitoring,
            performance=performance,
            security=security,
        )

    def _merge_config_dict(
        self, base_dict: Dict[str, Any], override_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""

        result = base_dict.copy()

        for key, value in override_dict.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_config_dict(result[key], value)
            else:
                result[key] = value

        return result

    def get_protocol_settings(self, protocol: str) -> Optional[ProtocolSettings]:
        """Get protocol-specific settings."""
        config = self.load_config()
        return config.legacy_response.protocol_settings.get(protocol)

    def get_confidence_thresholds(self) -> ConfidenceThresholds:
        """Get decision confidence thresholds."""
        config = self.load_config()
        return config.decision_engine.confidence_thresholds

    def get_response_risk_score(self, response_type: str) -> float:
        """Get risk score for a response type."""
        config = self.load_config()
        return getattr(
            config.decision_engine.response_risk_scores, response_type.lower(), 0.5
        )

    def is_integration_enabled(self, integration_type: str) -> bool:
        """Check if an integration is enabled."""
        config = self.load_config()
        integration_config = getattr(config.integrations, integration_type.lower(), {})
        return (
            integration_config.get("enabled", False)
            if isinstance(integration_config, dict)
            else False
        )

    def get_quarantine_method(self, system_type: str = "default") -> QuarantineMethod:
        """Get quarantine method for a system type."""
        config = self.load_config()
        return config.legacy_response.quarantine_methods.get(
            system_type, QuarantineMethod.NETWORK_SEGMENTATION
        )

    def reload_config(self) -> SecurityOrchestratorConfig:
        """Force reload configuration from disk."""
        return self.load_config(force_reload=True)

    def validate_config_file(self) -> bool:
        """Validate configuration file without loading into cache."""
        try:
            temp_config = self.load_config(force_reload=True)
            temp_config.validate()
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def export_config_schema(self) -> Dict[str, Any]:
        """Export configuration schema for documentation."""

        def get_field_info(dataclass_type):
            """Get field information from dataclass."""
            fields = {}
            for field in dataclass_type.__dataclass_fields__.values():
                fields[field.name] = {
                    "type": str(field.type),
                    "default": (
                        field.default
                        if field.default != field.default_factory
                        else "factory"
                    ),
                    "required": field.default == field.default_factory
                    and not callable(field.default_factory),
                }
            return fields

        return {
            "SecurityOrchestratorConfig": get_field_info(SecurityOrchestratorConfig),
            "DecisionEngineConfig": get_field_info(DecisionEngineConfig),
            "LegacyResponseConfig": get_field_info(LegacyResponseConfig),
            "ThreatAnalyzerConfig": get_field_info(ThreatAnalyzerConfig),
            "ConfidenceThresholds": get_field_info(ConfidenceThresholds),
            "ResponseRiskScores": get_field_info(ResponseRiskScores),
            "ProtocolSettings": get_field_info(ProtocolSettings),
        }


# Global configuration manager instance
_config_manager: Optional[SecurityConfigManager] = None


def get_security_config_manager(
    config_path: Optional[str] = None, environment: Optional[str] = None
) -> SecurityConfigManager:
    """Get global security configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = SecurityConfigManager(config_path, environment)
    return _config_manager


def get_security_config(force_reload: bool = False) -> SecurityOrchestratorConfig:
    """Get security orchestrator configuration."""
    manager = get_security_config_manager()
    return manager.load_config(force_reload=force_reload)


def reload_security_config() -> SecurityOrchestratorConfig:
    """Reload security configuration from disk."""
    manager = get_security_config_manager()
    return manager.reload_config()
