"""
CRONOS AI Engine - Legacy System Whisperer Configuration

Enterprise-grade configuration management for Legacy System Whisperer feature.
Extends the existing configuration system with legacy-specific settings.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum

from ..core.config import Config as BaseConfig
from ..core.exceptions import ConfigurationException


class LegacySystemType(str, Enum):
    """Supported legacy system types."""
    MAINFRAME = "mainframe"
    COBOL = "cobol"
    SCADA = "scada"
    MEDICAL_DEVICE = "medical_device"
    PLC = "plc"
    DCS = "dcs"
    LEGACY_DATABASE = "legacy_database"
    EMBEDDED_SYSTEM = "embedded_system"
    PROPRIETARY_PROTOCOL = "proprietary_protocol"


class PredictionHorizonConfig(str, Enum):
    """Prediction horizon configuration options."""
    SHORT_TERM = "short_term"  # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"  # 1-6 months
    STRATEGIC = "strategic"  # 6+ months


class KnowledgeSourceType(str, Enum):
    """Types of knowledge sources for capture."""
    EXPERT_INTERVIEW = "expert_interview"
    DOCUMENTATION_MINING = "documentation_mining"
    CODE_ANALYSIS = "code_analysis"
    LOG_ANALYSIS = "log_analysis"
    INCIDENT_ANALYSIS = "incident_analysis"
    MAINTENANCE_RECORDS = "maintenance_records"


@dataclass
class AnomalyDetectionConfig:
    """Configuration for enhanced anomaly detection."""
    
    # Detection thresholds
    anomaly_threshold: float = 0.95
    critical_anomaly_threshold: float = 0.99
    warning_threshold: float = 0.85
    
    # Model parameters
    vae_latent_dim: int = 64
    vae_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    vae_learning_rate: float = 1e-4
    vae_batch_size: int = 32
    vae_epochs: int = 100
    
    # Data processing
    sequence_length: int = 100
    feature_normalization: bool = True
    outlier_removal: bool = True
    outlier_std_threshold: float = 3.0
    
    # LLM integration
    llm_analysis_enabled: bool = True
    llm_confidence_threshold: float = 0.8
    llm_analysis_batch_size: int = 10
    
    # Performance tuning
    detection_interval_seconds: int = 300  # 5 minutes
    batch_processing_size: int = 1000
    max_concurrent_detections: int = 5
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Alert configuration
    enable_real_time_alerts: bool = True
    alert_cooldown_minutes: int = 15
    escalation_threshold_count: int = 3
    
    def validate(self) -> None:
        """Validate anomaly detection configuration."""
        errors = []
        
        if not 0.5 <= self.anomaly_threshold <= 1.0:
            errors.append("Anomaly threshold must be between 0.5 and 1.0")
        
        if self.critical_anomaly_threshold <= self.anomaly_threshold:
            errors.append("Critical anomaly threshold must be higher than anomaly threshold")
        
        if self.vae_latent_dim <= 0:
            errors.append("VAE latent dimension must be positive")
        
        if self.sequence_length <= 0:
            errors.append("Sequence length must be positive")
        
        if errors:
            raise ConfigurationException(f"Anomaly detection config validation failed: {'; '.join(errors)}")


@dataclass
class PredictiveAnalyticsConfig:
    """Configuration for predictive analytics components."""
    
    # Failure prediction
    failure_prediction_enabled: bool = True
    prediction_horizons: List[PredictionHorizonConfig] = field(
        default_factory=lambda: [
            PredictionHorizonConfig.SHORT_TERM,
            PredictionHorizonConfig.MEDIUM_TERM,
            PredictionHorizonConfig.LONG_TERM
        ]
    )
    min_historical_data_days: int = 30
    prediction_confidence_threshold: float = 0.75
    
    # Performance monitoring
    performance_monitoring_enabled: bool = True
    performance_baseline_days: int = 30
    performance_degradation_threshold: float = 0.1  # 10% degradation
    performance_alert_threshold: float = 0.2  # 20% degradation
    
    # Maintenance scheduling
    maintenance_optimization_enabled: bool = True
    maintenance_window_hours: List[int] = field(default_factory=lambda: [2, 3, 4])  # 2-4 AM
    maintenance_blackout_days: List[str] = field(default_factory=lambda: ["friday", "saturday"])
    resource_utilization_target: float = 0.8  # 80% resource utilization
    
    # Time series analysis
    time_series_model: str = "lstm"  # lstm, arima, prophet
    time_series_lookback_days: int = 90
    time_series_forecast_days: int = 30
    seasonal_decomposition: bool = True
    trend_detection: bool = True
    
    # LLM-enhanced prediction
    llm_prediction_enhancement: bool = True
    llm_context_window_days: int = 7
    llm_pattern_analysis: bool = True
    
    def validate(self) -> None:
        """Validate predictive analytics configuration."""
        errors = []
        
        if self.min_historical_data_days < 7:
            errors.append("Minimum historical data must be at least 7 days")
        
        if not 0.5 <= self.prediction_confidence_threshold <= 1.0:
            errors.append("Prediction confidence threshold must be between 0.5 and 1.0")
        
        if not 0.0 <= self.resource_utilization_target <= 1.0:
            errors.append("Resource utilization target must be between 0.0 and 1.0")
        
        if self.time_series_lookback_days < self.min_historical_data_days:
            errors.append("Time series lookback must be at least minimum historical data period")
        
        if errors:
            raise ConfigurationException(f"Predictive analytics config validation failed: {'; '.join(errors)}")


@dataclass
class KnowledgeCaptureConfig:
    """Configuration for knowledge capture system."""
    
    # Knowledge sources
    enabled_sources: List[KnowledgeSourceType] = field(
        default_factory=lambda: [
            KnowledgeSourceType.EXPERT_INTERVIEW,
            KnowledgeSourceType.DOCUMENTATION_MINING,
            KnowledgeSourceType.LOG_ANALYSIS
        ]
    )
    
    # Expert interview settings
    session_timeout_minutes: int = 120
    max_concurrent_sessions: int = 10
    session_auto_save_interval: int = 300  # 5 minutes
    expert_validation_required: bool = True
    
    # Knowledge processing
    llm_processing_enabled: bool = True
    knowledge_validation_threshold: float = 0.8
    auto_categorization: bool = True
    duplicate_detection: bool = True
    similarity_threshold: float = 0.85
    
    # Storage and retrieval
    knowledge_retention_days: int = 1825  # 5 years
    knowledge_versioning: bool = True
    full_text_search: bool = True
    semantic_search: bool = True
    
    # Quality control
    peer_review_required: bool = True
    minimum_confidence_score: float = 0.7
    knowledge_approval_workflow: bool = True
    audit_trail_enabled: bool = True
    
    # Performance settings
    batch_processing_size: int = 100
    indexing_enabled: bool = True
    cache_frequently_accessed: bool = True
    cache_size_mb: int = 500
    
    def validate(self) -> None:
        """Validate knowledge capture configuration."""
        errors = []
        
        if self.session_timeout_minutes <= 0:
            errors.append("Session timeout must be positive")
        
        if not 0.0 <= self.knowledge_validation_threshold <= 1.0:
            errors.append("Knowledge validation threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.similarity_threshold <= 1.0:
            errors.append("Similarity threshold must be between 0.0 and 1.0")
        
        if self.knowledge_retention_days <= 0:
            errors.append("Knowledge retention period must be positive")
        
        if errors:
            raise ConfigurationException(f"Knowledge capture config validation failed: {'; '.join(errors)}")


@dataclass
class DecisionSupportConfig:
    """Configuration for decision support system."""
    
    # Recommendation engine
    recommendation_enabled: bool = True
    max_recommendations: int = 10
    recommendation_diversity: float = 0.3  # Encourage diverse recommendations
    confidence_threshold: float = 0.7
    
    # Impact assessment
    impact_assessment_enabled: bool = True
    financial_impact_calculation: bool = True
    risk_assessment_enabled: bool = True
    compliance_impact_check: bool = True
    
    # Decision categories
    supported_decision_types: List[str] = field(
        default_factory=lambda: [
            "maintenance_planning",
            "upgrade_decision",
            "risk_mitigation",
            "resource_allocation",
            "system_retirement",
            "emergency_response"
        ]
    )
    
    # Action planning
    action_planning_enabled: bool = True
    detailed_timelines: bool = True
    resource_allocation_planning: bool = True
    risk_mitigation_planning: bool = True
    
    # LLM integration
    llm_recommendation_enhancement: bool = True
    llm_explanation_generation: bool = True
    multi_llm_consensus: bool = False  # Use multiple LLMs for critical decisions
    
    # Performance tuning
    decision_cache_ttl_hours: int = 24
    parallel_analysis: bool = True
    max_concurrent_analyses: int = 3
    
    def validate(self) -> None:
        """Validate decision support configuration."""
        errors = []
        
        if self.max_recommendations <= 0:
            errors.append("Maximum recommendations must be positive")
        
        if not 0.0 <= self.recommendation_diversity <= 1.0:
            errors.append("Recommendation diversity must be between 0.0 and 1.0")
        
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append("Confidence threshold must be between 0.0 and 1.0")
        
        if self.decision_cache_ttl_hours <= 0:
            errors.append("Decision cache TTL must be positive")
        
        if errors:
            raise ConfigurationException(f"Decision support config validation failed: {'; '.join(errors)}")


@dataclass
class LLMIntegrationConfig:
    """Configuration for LLM service integration."""
    
    # Primary LLM provider
    primary_provider: str = "openai"  # openai, anthropic, ollama
    fallback_providers: List[str] = field(default_factory=lambda: ["anthropic", "ollama"])
    
    # Request settings
    max_tokens: int = 4000
    temperature: float = 0.1  # Low temperature for consistent analysis
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 2
    
    # Rate limiting
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    burst_limit: int = 20
    
    # Quality control
    response_validation: bool = True
    minimum_response_length: int = 50
    maximum_response_length: int = 10000
    content_filtering: bool = True
    
    # Caching
    response_caching: bool = True
    cache_ttl_hours: int = 6
    cache_size_mb: int = 1000
    
    # Domain-specific prompts
    use_domain_prompts: bool = True
    prompt_versioning: bool = True
    prompt_optimization: bool = True
    
    def validate(self) -> None:
        """Validate LLM integration configuration."""
        errors = []
        
        if self.max_tokens <= 0:
            errors.append("Max tokens must be positive")
        
        if not 0.0 <= self.temperature <= 2.0:
            errors.append("Temperature must be between 0.0 and 2.0")
        
        if self.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
        
        if self.requests_per_minute <= 0:
            errors.append("Requests per minute must be positive")
        
        if errors:
            raise ConfigurationException(f"LLM integration config validation failed: {'; '.join(errors)}")


@dataclass
class MonitoringObservabilityConfig:
    """Configuration for monitoring and observability."""
    
    # Metrics collection
    metrics_collection_enabled: bool = True
    metrics_collection_interval: int = 60  # seconds
    custom_metrics_enabled: bool = True
    
    # Prometheus integration
    prometheus_enabled: bool = True
    prometheus_port: int = 8090
    prometheus_path: str = "/legacy-metrics"
    
    # Logging configuration
    structured_logging: bool = True
    log_level: str = "INFO"
    log_rotation: bool = True
    max_log_size_mb: int = 100
    log_retention_days: int = 30
    
    # Health checks
    health_check_enabled: bool = True
    health_check_interval: int = 30  # seconds
    component_health_checks: bool = True
    
    # Performance monitoring
    performance_monitoring: bool = True
    latency_tracking: bool = True
    throughput_tracking: bool = True
    error_rate_tracking: bool = True
    
    # Alerting thresholds
    high_latency_ms: float = 1000
    error_rate_threshold: float = 0.05  # 5%
    system_failure_alert: bool = True
    maintenance_due_alert: bool = True
    knowledge_gap_alert: bool = True
    
    # Distributed tracing
    tracing_enabled: bool = True
    tracing_sample_rate: float = 0.1  # 10% sampling
    trace_export_endpoint: str = "http://localhost:14268/api/traces"
    
    def validate(self) -> None:
        """Validate monitoring configuration."""
        errors = []
        
        if self.metrics_collection_interval <= 0:
            errors.append("Metrics collection interval must be positive")
        
        if not 1024 <= self.prometheus_port <= 65535:
            errors.append("Prometheus port must be a valid port number")
        
        if not 0.0 <= self.error_rate_threshold <= 1.0:
            errors.append("Error rate threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.tracing_sample_rate <= 1.0:
            errors.append("Tracing sample rate must be between 0.0 and 1.0")
        
        if errors:
            raise ConfigurationException(f"Monitoring config validation failed: {'; '.join(errors)}")


@dataclass
class SecurityConfig:
    """Security configuration for Legacy System Whisperer."""
    
    # Data encryption
    encrypt_sensitive_data: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    
    # Access control
    role_based_access: bool = True
    multi_factor_auth: bool = False  # Optional for internal systems
    session_timeout_minutes: int = 480  # 8 hours
    
    # Data privacy
    anonymize_system_data: bool = True
    data_masking: bool = True
    pii_detection: bool = True
    gdpr_compliance: bool = True
    
    # Audit and compliance
    audit_logging: bool = True
    compliance_reporting: bool = True
    data_retention_enforcement: bool = True
    
    # Secure communication
    tls_enabled: bool = True
    certificate_validation: bool = True
    secure_llm_communication: bool = True
    
    def validate(self) -> None:
        """Validate security configuration."""
        errors = []
        
        if self.key_rotation_days <= 0:
            errors.append("Key rotation period must be positive")
        
        if self.session_timeout_minutes <= 0:
            errors.append("Session timeout must be positive")
        
        if errors:
            raise ConfigurationException(f"Security config validation failed: {'; '.join(errors)}")


@dataclass
class LegacySystemWhispererConfig:
    """Main configuration for Legacy System Whisperer feature."""
    
    # Feature enablement
    enabled: bool = True
    service_name: str = "legacy-system-whisperer"
    version: str = "1.0.0"
    
    # Supported system types
    supported_system_types: List[LegacySystemType] = field(
        default_factory=lambda: [
            LegacySystemType.MAINFRAME,
            LegacySystemType.COBOL,
            LegacySystemType.SCADA,
            LegacySystemType.MEDICAL_DEVICE
        ]
    )
    
    # Component configurations
    anomaly_detection: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)
    predictive_analytics: PredictiveAnalyticsConfig = field(default_factory=PredictiveAnalyticsConfig)
    knowledge_capture: KnowledgeCaptureConfig = field(default_factory=KnowledgeCaptureConfig)
    decision_support: DecisionSupportConfig = field(default_factory=DecisionSupportConfig)
    llm_integration: LLMIntegrationConfig = field(default_factory=LLMIntegrationConfig)
    monitoring: MonitoringObservabilityConfig = field(default_factory=MonitoringObservabilityConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # System limits
    max_registered_systems: int = 1000
    max_concurrent_analyses: int = 20
    max_knowledge_sessions: int = 50
    
    # Performance tuning
    async_processing: bool = True
    background_task_workers: int = 4
    queue_size_limit: int = 10000
    memory_limit_mb: int = 2048
    
    # Integration settings
    integrate_with_existing_monitoring: bool = True
    export_metrics_to_prometheus: bool = True
    use_existing_llm_service: bool = True
    use_existing_database: bool = True
    
    def validate(self) -> None:
        """Validate the complete Legacy System Whisperer configuration."""
        errors = []
        
        # Validate limits
        if self.max_registered_systems <= 0:
            errors.append("Maximum registered systems must be positive")
        
        if self.max_concurrent_analyses <= 0:
            errors.append("Maximum concurrent analyses must be positive")
        
        if self.memory_limit_mb <= 512:
            errors.append("Memory limit must be at least 512 MB")
        
        # Validate component configurations
        try:
            self.anomaly_detection.validate()
        except ConfigurationException as e:
            errors.append(f"Anomaly detection: {e}")
        
        try:
            self.predictive_analytics.validate()
        except ConfigurationException as e:
            errors.append(f"Predictive analytics: {e}")
        
        try:
            self.knowledge_capture.validate()
        except ConfigurationException as e:
            errors.append(f"Knowledge capture: {e}")
        
        try:
            self.decision_support.validate()
        except ConfigurationException as e:
            errors.append(f"Decision support: {e}")
        
        try:
            self.llm_integration.validate()
        except ConfigurationException as e:
            errors.append(f"LLM integration: {e}")
        
        try:
            self.monitoring.validate()
        except ConfigurationException as e:
            errors.append(f"Monitoring: {e}")
        
        try:
            self.security.validate()
        except ConfigurationException as e:
            errors.append(f"Security: {e}")
        
        if errors:
            raise ConfigurationException(f"Legacy System Whisperer config validation failed: {'; '.join(errors)}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegacySystemWhispererConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Update basic settings
        if "enabled" in data:
            config.enabled = data["enabled"]
        if "service_name" in data:
            config.service_name = data["service_name"]
        
        # Update component configurations
        if "anomaly_detection" in data:
            config.anomaly_detection = AnomalyDetectionConfig(**data["anomaly_detection"])
        if "predictive_analytics" in data:
            config.predictive_analytics = PredictiveAnalyticsConfig(**data["predictive_analytics"])
        if "knowledge_capture" in data:
            config.knowledge_capture = KnowledgeCaptureConfig(**data["knowledge_capture"])
        if "decision_support" in data:
            config.decision_support = DecisionSupportConfig(**data["decision_support"])
        if "llm_integration" in data:
            config.llm_integration = LLMIntegrationConfig(**data["llm_integration"])
        if "monitoring" in data:
            config.monitoring = MonitoringObservabilityConfig(**data["monitoring"])
        if "security" in data:
            config.security = SecurityConfig(**data["security"])
        
        # Update system limits
        for key in ["max_registered_systems", "max_concurrent_analyses", "memory_limit_mb"]:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "service_name": self.service_name,
            "version": self.version,
            "supported_system_types": [stype.value for stype in self.supported_system_types],
            
            "max_registered_systems": self.max_registered_systems,
            "max_concurrent_analyses": self.max_concurrent_analyses,
            "memory_limit_mb": self.memory_limit_mb,
            
            "anomaly_detection": {
                "anomaly_threshold": self.anomaly_detection.anomaly_threshold,
                "llm_analysis_enabled": self.anomaly_detection.llm_analysis_enabled,
                "detection_interval_seconds": self.anomaly_detection.detection_interval_seconds,
            },
            
            "predictive_analytics": {
                "failure_prediction_enabled": self.predictive_analytics.failure_prediction_enabled,
                "prediction_horizons": [h.value for h in self.predictive_analytics.prediction_horizons],
                "min_historical_data_days": self.predictive_analytics.min_historical_data_days,
            },
            
            "knowledge_capture": {
                "enabled_sources": [src.value for src in self.knowledge_capture.enabled_sources],
                "llm_processing_enabled": self.knowledge_capture.llm_processing_enabled,
                "knowledge_retention_days": self.knowledge_capture.knowledge_retention_days,
            },
            
            "decision_support": {
                "recommendation_enabled": self.decision_support.recommendation_enabled,
                "max_recommendations": self.decision_support.max_recommendations,
                "llm_recommendation_enhancement": self.decision_support.llm_recommendation_enhancement,
            },
            
            "monitoring": {
                "metrics_collection_enabled": self.monitoring.metrics_collection_enabled,
                "prometheus_enabled": self.monitoring.prometheus_enabled,
                "health_check_enabled": self.monitoring.health_check_enabled,
            }
        }


def extend_base_config_with_legacy(base_config: BaseConfig) -> BaseConfig:
    """
    Extend the base CRONOS AI configuration with Legacy System Whisperer settings.
    
    Args:
        base_config: Base configuration to extend
        
    Returns:
        Extended configuration with Legacy System Whisperer settings
    """
    
    # Add legacy system whisperer configuration
    if not hasattr(base_config, 'legacy_system_whisperer'):
        base_config.legacy_system_whisperer = LegacySystemWhispererConfig()
    
    return base_config


def load_legacy_config_from_file(config_path: Union[str, Path]) -> LegacySystemWhispererConfig:
    """Load Legacy System Whisperer configuration from YAML file."""
    import yaml
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationException(f"Legacy configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        # Extract legacy system whisperer configuration
        legacy_data = config_data.get("legacy_system_whisperer", {})
        
        return LegacySystemWhispererConfig.from_dict(legacy_data)
        
    except yaml.YAMLError as e:
        raise ConfigurationException(f"Invalid YAML in legacy configuration file: {e}")
    except Exception as e:
        raise ConfigurationException(f"Error loading legacy configuration file: {e}")


def load_legacy_config_from_env(prefix: str = "CRONOS_AI_LEGACY_") -> LegacySystemWhispererConfig:
    """Load Legacy System Whisperer configuration from environment variables."""
    config = LegacySystemWhispererConfig()
    
    # Load basic settings
    config.enabled = os.getenv(f"{prefix}ENABLED", str(config.enabled)).lower() == "true"
    config.service_name = os.getenv(f"{prefix}SERVICE_NAME", config.service_name)
    
    # Load anomaly detection settings
    if os.getenv(f"{prefix}ANOMALY_THRESHOLD"):
        config.anomaly_detection.anomaly_threshold = float(os.getenv(f"{prefix}ANOMALY_THRESHOLD", config.anomaly_detection.anomaly_threshold))
    
    # Load predictive analytics settings
    if os.getenv(f"{prefix}PREDICTION_ENABLED"):
        config.predictive_analytics.failure_prediction_enabled = os.getenv(f"{prefix}PREDICTION_ENABLED", str(config.predictive_analytics.failure_prediction_enabled)).lower() == "true"
    
    # Load LLM integration settings
    if os.getenv(f"{prefix}LLM_PROVIDER"):
        config.llm_integration.primary_provider = os.getenv(f"{prefix}LLM_PROVIDER", config.llm_integration.primary_provider)
    
    if os.getenv(f"{prefix}LLM_MAX_TOKENS"):
        config.llm_integration.max_tokens = int(os.getenv(f"{prefix}LLM_MAX_TOKENS", str(config.llm_integration.max_tokens)))
    
    # Load system limits
    if os.getenv(f"{prefix}MAX_SYSTEMS"):
        config.max_registered_systems = int(os.getenv(f"{prefix}MAX_SYSTEMS", str(config.max_registered_systems)))
    
    if os.getenv(f"{prefix}MEMORY_LIMIT_MB"):
        config.memory_limit_mb = int(os.getenv(f"{prefix}MEMORY_LIMIT_MB", str(config.memory_limit_mb)))
    
    return config


def get_default_legacy_config() -> LegacySystemWhispererConfig:
    """Get default Legacy System Whisperer configuration."""
    return LegacySystemWhispererConfig()


def validate_legacy_config(config: LegacySystemWhispererConfig) -> None:
    """Validate Legacy System Whisperer configuration and raise exceptions for invalid settings."""
    config.validate()


# Configuration factory functions
def create_production_legacy_config() -> LegacySystemWhispererConfig:
    """Create production-optimized Legacy System Whisperer configuration."""
    config = LegacySystemWhispererConfig()
    
    # Production optimizations
    config.anomaly_detection.anomaly_threshold = 0.98
    config.anomaly_detection.enable_real_time_alerts = True
    config.anomaly_detection.cache_ttl_seconds = 7200  # 2 hours
    
    config.predictive_analytics.min_historical_data_days = 90  # More data for production
    config.predictive_analytics.prediction_confidence_threshold = 0.85
    
    config.knowledge_capture.peer_review_required = True
    config.knowledge_capture.audit_trail_enabled = True
    
    config.decision_support.multi_llm_consensus = True  # Use multiple LLMs for critical decisions
    config.decision_support.detailed_timelines = True
    
    config.llm_integration.temperature = 0.05  # Very low temperature for consistency
    config.llm_integration.response_validation = True
    
    config.monitoring.metrics_collection_interval = 30  # More frequent monitoring
    config.monitoring.health_check_interval = 15
    
    config.security.encrypt_sensitive_data = True
    config.security.audit_logging = True
    config.security.compliance_reporting = True
    
    # Higher limits for production
    config.max_registered_systems = 5000
    config.max_concurrent_analyses = 50
    config.memory_limit_mb = 8192  # 8GB
    
    return config


def create_development_legacy_config() -> LegacySystemWhispererConfig:
    """Create development-optimized Legacy System Whisperer configuration."""
    config = LegacySystemWhispererConfig()
    
    # Development settings
    config.anomaly_detection.anomaly_threshold = 0.85
    config.anomaly_detection.cache_ttl_seconds = 300  # 5 minutes
    
    config.predictive_analytics.min_historical_data_days = 7  # Less data needed for testing
    
    config.knowledge_capture.peer_review_required = False
    config.knowledge_capture.session_timeout_minutes = 30  # Shorter sessions
    
    config.decision_support.multi_llm_consensus = False
    
    config.llm_integration.temperature = 0.2  # Slightly higher for experimentation
    config.llm_integration.timeout_seconds = 15  # Faster timeouts
    
    config.monitoring.metrics_collection_interval = 120  # Less frequent
    
    config.security.multi_factor_auth = False
    config.security.gdpr_compliance = False  # Not needed for dev
    
    # Lower limits for development
    config.max_registered_systems = 100
    config.max_concurrent_analyses = 5
    config.memory_limit_mb = 1024  # 1GB
    
    return config