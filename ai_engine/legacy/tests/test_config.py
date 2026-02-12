"""
Tests for Legacy System Whisperer Configuration

Test suite for configuration management and validation.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, Mock

from ..config import (
    # Enums
    LegacySystemType,
    PredictionHorizonConfig,
    KnowledgeSourceType,
    # Config Classes
    AnomalyDetectionConfig,
    PredictiveAnalyticsConfig,
    KnowledgeCaptureConfig,
    DecisionSupportConfig,
    LLMIntegrationConfig,
    MonitoringObservabilityConfig,
    SecurityConfig,
    LegacySystemWhispererConfig,
    # Factory functions
    extend_base_config_with_legacy,
    load_legacy_config_from_file,
    load_legacy_config_from_env,
    get_default_legacy_config,
    validate_legacy_config,
    create_production_legacy_config,
    create_development_legacy_config,
)
from ...core.config import Config
from ...core.exceptions import ConfigurationException


class TestLegacySystemType:
    """Test LegacySystemType enum."""

    @pytest.mark.unit
    def test_legacy_system_type_values(self):
        """Test LegacySystemType enum values."""
        assert LegacySystemType.MAINFRAME.value == "mainframe"
        assert LegacySystemType.COBOL.value == "cobol"
        assert LegacySystemType.SCADA.value == "scada"
        assert LegacySystemType.MEDICAL_DEVICE.value == "medical_device"
        assert LegacySystemType.PLC.value == "plc"
        assert LegacySystemType.DCS.value == "dcs"
        assert LegacySystemType.LEGACY_DATABASE.value == "legacy_database"
        assert LegacySystemType.EMBEDDED_SYSTEM.value == "embedded_system"
        assert LegacySystemType.PROPRIETARY_PROTOCOL.value == "proprietary_protocol"


class TestAnomalyDetectionConfig:
    """Test AnomalyDetectionConfig."""

    @pytest.mark.unit
    def test_default_anomaly_config(self):
        """Test default anomaly detection configuration."""
        config = AnomalyDetectionConfig()

        assert config.anomaly_threshold == 0.95
        assert config.critical_anomaly_threshold == 0.99
        assert config.warning_threshold == 0.85
        assert config.vae_latent_dim == 64
        assert config.vae_hidden_dims == [512, 256, 128]
        assert config.vae_learning_rate == 1e-4
        assert config.vae_batch_size == 32
        assert config.vae_epochs == 100
        assert config.sequence_length == 100
        assert config.feature_normalization is True
        assert config.llm_analysis_enabled is True
        assert config.detection_interval_seconds == 300
        assert config.enable_real_time_alerts is True

    @pytest.mark.unit
    def test_anomaly_config_validation_success(self):
        """Test successful anomaly config validation."""
        config = AnomalyDetectionConfig(
            anomaly_threshold=0.90,
            critical_anomaly_threshold=0.95,
            vae_latent_dim=32,
            sequence_length=50,
        )

        # Should not raise exception
        config.validate()

    @pytest.mark.unit
    def test_anomaly_config_validation_invalid_threshold(self):
        """Test anomaly config validation with invalid threshold."""
        config = AnomalyDetectionConfig(anomaly_threshold=0.4)  # Below 0.5

        with pytest.raises(
            ConfigurationException,
            match="Anomaly threshold must be between 0.5 and 1.0",
        ):
            config.validate()

    @pytest.mark.unit
    def test_anomaly_config_validation_critical_threshold_too_low(self):
        """Test validation with critical threshold lower than anomaly threshold."""
        config = AnomalyDetectionConfig(
            anomaly_threshold=0.95,
            critical_anomaly_threshold=0.90,  # Lower than anomaly threshold
        )

        with pytest.raises(ConfigurationException, match="Critical anomaly threshold must be higher"):
            config.validate()

    @pytest.mark.unit
    def test_anomaly_config_validation_negative_latent_dim(self):
        """Test validation with negative latent dimension."""
        config = AnomalyDetectionConfig(vae_latent_dim=-10)

        with pytest.raises(ConfigurationException, match="VAE latent dimension must be positive"):
            config.validate()

    @pytest.mark.unit
    def test_anomaly_config_validation_negative_sequence_length(self):
        """Test validation with negative sequence length."""
        config = AnomalyDetectionConfig(sequence_length=-5)

        with pytest.raises(ConfigurationException, match="Sequence length must be positive"):
            config.validate()


class TestPredictiveAnalyticsConfig:
    """Test PredictiveAnalyticsConfig."""

    @pytest.mark.unit
    def test_default_predictive_config(self):
        """Test default predictive analytics configuration."""
        config = PredictiveAnalyticsConfig()

        assert config.failure_prediction_enabled is True
        assert PredictionHorizonConfig.SHORT_TERM in config.prediction_horizons
        assert PredictionHorizonConfig.MEDIUM_TERM in config.prediction_horizons
        assert PredictionHorizonConfig.LONG_TERM in config.prediction_horizons
        assert config.min_historical_data_days == 30
        assert config.prediction_confidence_threshold == 0.75
        assert config.performance_monitoring_enabled is True
        assert config.maintenance_optimization_enabled is True
        assert config.time_series_model == "lstm"
        assert config.llm_prediction_enhancement is True

    @pytest.mark.unit
    def test_predictive_config_validation_success(self):
        """Test successful predictive config validation."""
        config = PredictiveAnalyticsConfig(
            min_historical_data_days=14,
            prediction_confidence_threshold=0.8,
            resource_utilization_target=0.75,
            time_series_lookback_days=60,
        )

        # Should not raise exception
        config.validate()

    @pytest.mark.unit
    def test_predictive_config_validation_insufficient_historical_data(self):
        """Test validation with insufficient historical data."""
        config = PredictiveAnalyticsConfig(min_historical_data_days=5)  # Less than 7

        with pytest.raises(
            ConfigurationException,
            match="Minimum historical data must be at least 7 days",
        ):
            config.validate()

    @pytest.mark.unit
    def test_predictive_config_validation_invalid_confidence_threshold(self):
        """Test validation with invalid confidence threshold."""
        config = PredictiveAnalyticsConfig(prediction_confidence_threshold=1.5)  # Over 1.0

        with pytest.raises(
            ConfigurationException,
            match="Prediction confidence threshold must be between 0.5 and 1.0",
        ):
            config.validate()

    @pytest.mark.unit
    def test_predictive_config_validation_invalid_resource_utilization(self):
        """Test validation with invalid resource utilization target."""
        config = PredictiveAnalyticsConfig(resource_utilization_target=1.2)  # Over 1.0

        with pytest.raises(
            ConfigurationException,
            match="Resource utilization target must be between 0.0 and 1.0",
        ):
            config.validate()


class TestKnowledgeCaptureConfig:
    """Test KnowledgeCaptureConfig."""

    @pytest.mark.unit
    def test_default_knowledge_config(self):
        """Test default knowledge capture configuration."""
        config = KnowledgeCaptureConfig()

        assert KnowledgeSourceType.EXPERT_INTERVIEW in config.enabled_sources
        assert KnowledgeSourceType.DOCUMENTATION_MINING in config.enabled_sources
        assert KnowledgeSourceType.LOG_ANALYSIS in config.enabled_sources
        assert config.session_timeout_minutes == 120
        assert config.max_concurrent_sessions == 10
        assert config.llm_processing_enabled is True
        assert config.knowledge_validation_threshold == 0.8
        assert config.duplicate_detection is True
        assert config.knowledge_retention_days == 1825  # 5 years
        assert config.peer_review_required is True
        assert config.audit_trail_enabled is True

    @pytest.mark.unit
    def test_knowledge_config_validation_success(self):
        """Test successful knowledge config validation."""
        config = KnowledgeCaptureConfig(
            session_timeout_minutes=60,
            knowledge_validation_threshold=0.9,
            similarity_threshold=0.8,
            knowledge_retention_days=730,
        )

        # Should not raise exception
        config.validate()

    @pytest.mark.unit
    def test_knowledge_config_validation_negative_timeout(self):
        """Test validation with negative session timeout."""
        config = KnowledgeCaptureConfig(session_timeout_minutes=-30)

        with pytest.raises(ConfigurationException, match="Session timeout must be positive"):
            config.validate()

    @pytest.mark.unit
    def test_knowledge_config_validation_invalid_validation_threshold(self):
        """Test validation with invalid validation threshold."""
        config = KnowledgeCaptureConfig(knowledge_validation_threshold=1.5)  # Over 1.0

        with pytest.raises(
            ConfigurationException,
            match="Knowledge validation threshold must be between 0.0 and 1.0",
        ):
            config.validate()


class TestDecisionSupportConfig:
    """Test DecisionSupportConfig."""

    @pytest.mark.unit
    def test_default_decision_config(self):
        """Test default decision support configuration."""
        config = DecisionSupportConfig()

        assert config.recommendation_enabled is True
        assert config.max_recommendations == 10
        assert config.recommendation_diversity == 0.3
        assert config.confidence_threshold == 0.7
        assert config.impact_assessment_enabled is True
        assert config.action_planning_enabled is True
        assert config.llm_recommendation_enhancement is True
        assert config.multi_llm_consensus is False
        assert "maintenance_planning" in config.supported_decision_types
        assert "upgrade_decision" in config.supported_decision_types

    @pytest.mark.unit
    def test_decision_config_validation_success(self):
        """Test successful decision config validation."""
        config = DecisionSupportConfig(
            max_recommendations=5,
            recommendation_diversity=0.5,
            confidence_threshold=0.8,
            decision_cache_ttl_hours=12,
        )

        # Should not raise exception
        config.validate()

    @pytest.mark.unit
    def test_decision_config_validation_negative_max_recommendations(self):
        """Test validation with negative max recommendations."""
        config = DecisionSupportConfig(max_recommendations=-5)

        with pytest.raises(ConfigurationException, match="Maximum recommendations must be positive"):
            config.validate()

    @pytest.mark.unit
    def test_decision_config_validation_invalid_diversity(self):
        """Test validation with invalid recommendation diversity."""
        config = DecisionSupportConfig(recommendation_diversity=1.5)  # Over 1.0

        with pytest.raises(
            ConfigurationException,
            match="Recommendation diversity must be between 0.0 and 1.0",
        ):
            config.validate()


class TestLLMIntegrationConfig:
    """Test LLMIntegrationConfig."""

    @pytest.mark.unit
    def test_default_llm_config(self):
        """Test default LLM integration configuration."""
        config = LLMIntegrationConfig()

        assert config.primary_provider == "openai"
        assert "anthropic" in config.fallback_providers
        assert "ollama" in config.fallback_providers
        assert config.max_tokens == 4000
        assert config.temperature == 0.1
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert config.requests_per_minute == 100
        assert config.response_caching is True
        assert config.use_domain_prompts is True

    @pytest.mark.unit
    def test_llm_config_validation_success(self):
        """Test successful LLM config validation."""
        config = LLMIntegrationConfig(max_tokens=2000, temperature=0.2, timeout_seconds=15, requests_per_minute=50)

        # Should not raise exception
        config.validate()

    @pytest.mark.unit
    def test_llm_config_validation_negative_max_tokens(self):
        """Test validation with negative max tokens."""
        config = LLMIntegrationConfig(max_tokens=-100)

        with pytest.raises(ConfigurationException, match="Max tokens must be positive"):
            config.validate()

    @pytest.mark.unit
    def test_llm_config_validation_invalid_temperature(self):
        """Test validation with invalid temperature."""
        config = LLMIntegrationConfig(temperature=2.5)  # Over 2.0

        with pytest.raises(ConfigurationException, match="Temperature must be between 0.0 and 2.0"):
            config.validate()


class TestLegacySystemWhispererConfig:
    """Test main LegacySystemWhispererConfig."""

    @pytest.mark.unit
    def test_default_main_config(self):
        """Test default main configuration."""
        config = LegacySystemWhispererConfig()

        assert config.enabled is True
        assert config.service_name == "legacy-system-whisperer"
        assert config.version == "1.0.0"
        assert LegacySystemType.MAINFRAME in config.supported_system_types
        assert LegacySystemType.COBOL in config.supported_system_types
        assert LegacySystemType.SCADA in config.supported_system_types
        assert LegacySystemType.MEDICAL_DEVICE in config.supported_system_types
        assert config.max_registered_systems == 1000
        assert config.max_concurrent_analyses == 20
        assert config.memory_limit_mb == 2048
        assert config.async_processing is True
        assert config.use_existing_llm_service is True

    @pytest.mark.unit
    def test_main_config_validation_success(self):
        """Test successful main config validation."""
        config = LegacySystemWhispererConfig(max_registered_systems=500, max_concurrent_analyses=10, memory_limit_mb=1024)

        # Should not raise exception
        config.validate()

    @pytest.mark.unit
    def test_main_config_validation_negative_max_systems(self):
        """Test validation with negative max systems."""
        config = LegacySystemWhispererConfig(max_registered_systems=-10)

        with pytest.raises(ConfigurationException, match="Maximum registered systems must be positive"):
            config.validate()

    @pytest.mark.unit
    def test_main_config_validation_insufficient_memory(self):
        """Test validation with insufficient memory limit."""
        config = LegacySystemWhispererConfig(memory_limit_mb=256)  # Below 512 MB

        with pytest.raises(ConfigurationException, match="Memory limit must be at least 512 MB"):
            config.validate()

    @pytest.mark.unit
    def test_main_config_component_validation_cascade(self):
        """Test that component validation errors cascade up."""
        config = LegacySystemWhispererConfig()

        # Set invalid component config
        config.anomaly_detection.anomaly_threshold = 0.3  # Invalid

        with pytest.raises(ConfigurationException, match="Anomaly detection"):
            config.validate()

    @pytest.mark.unit
    def test_main_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "enabled": False,
            "service_name": "test-legacy-service",
            "max_registered_systems": 100,
            "anomaly_detection": {
                "anomaly_threshold": 0.85,
                "llm_analysis_enabled": False,
            },
            "llm_integration": {"primary_provider": "anthropic", "max_tokens": 2000},
        }

        config = LegacySystemWhispererConfig.from_dict(config_dict)

        assert config.enabled is False
        assert config.service_name == "test-legacy-service"
        assert config.max_registered_systems == 100
        assert config.anomaly_detection.anomaly_threshold == 0.85
        assert config.anomaly_detection.llm_analysis_enabled is False
        assert config.llm_integration.primary_provider == "anthropic"
        assert config.llm_integration.max_tokens == 2000

    @pytest.mark.unit
    def test_main_config_to_dict(self):
        """Test converting config to dictionary."""
        config = LegacySystemWhispererConfig(enabled=True, max_registered_systems=500)

        config_dict = config.to_dict()

        assert config_dict["enabled"] is True
        assert config_dict["max_registered_systems"] == 500
        assert "anomaly_detection" in config_dict
        assert "predictive_analytics" in config_dict
        assert "knowledge_capture" in config_dict
        assert "decision_support" in config_dict
        assert "monitoring" in config_dict


class TestConfigFileLoading:
    """Test configuration file loading."""

    @pytest.mark.unit
    def test_load_config_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "legacy_system_whisperer": {
                "enabled": True,
                "service_name": "test-service",
                "max_registered_systems": 200,
                "anomaly_detection": {
                    "anomaly_threshold": 0.88,
                    "detection_interval_seconds": 120,
                },
                "llm_integration": {"primary_provider": "ollama", "temperature": 0.05},
            }
        }

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Load config from file
            config = load_legacy_config_from_file(temp_path)

            assert config.enabled is True
            assert config.service_name == "test-service"
            assert config.max_registered_systems == 200
            assert config.anomaly_detection.anomaly_threshold == 0.88
            assert config.anomaly_detection.detection_interval_seconds == 120
            assert config.llm_integration.primary_provider == "ollama"
            assert config.llm_integration.temperature == 0.05

        finally:
            # Clean up temp file
            Path(temp_path).unlink()

    @pytest.mark.unit
    def test_load_config_from_missing_file(self):
        """Test loading config from missing file raises exception."""
        with pytest.raises(ConfigurationException, match="Legacy configuration file not found"):
            load_legacy_config_from_file("/nonexistent/config.yaml")

    @pytest.mark.unit
    def test_load_config_from_invalid_yaml(self):
        """Test loading config from invalid YAML file."""
        # Create temporary invalid YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:\n  - unclosed bracket [")
            temp_path = f.name

        try:
            with pytest.raises(ConfigurationException, match="Invalid YAML"):
                load_legacy_config_from_file(temp_path)
        finally:
            Path(temp_path).unlink()


class TestEnvironmentVariableLoading:
    """Test configuration loading from environment variables."""

    @pytest.mark.unit
    def test_load_config_from_env_defaults(self):
        """Test loading config from environment with defaults."""
        config = load_legacy_config_from_env()

        # Should have default values
        assert config.enabled is True
        assert config.service_name == "legacy-system-whisperer"
        assert config.anomaly_detection.anomaly_threshold == 0.95

    @pytest.mark.unit
    def test_load_config_from_env_with_values(self):
        """Test loading config from environment with set values."""
        env_vars = {
            "QBITEL_AI_LEGACY_ENABLED": "false",
            "QBITEL_AI_LEGACY_SERVICE_NAME": "env-test-service",
            "QBITEL_AI_LEGACY_ANOMALY_THRESHOLD": "0.80",
            "QBITEL_AI_LEGACY_PREDICTION_ENABLED": "false",
            "QBITEL_AI_LEGACY_LLM_PROVIDER": "anthropic",
            "QBITEL_AI_LEGACY_LLM_MAX_TOKENS": "2000",
            "QBITEL_AI_LEGACY_MAX_SYSTEMS": "100",
            "QBITEL_AI_LEGACY_MEMORY_LIMIT_MB": "1024",
        }

        with patch.dict("os.environ", env_vars):
            config = load_legacy_config_from_env()

            assert config.enabled is False
            assert config.service_name == "env-test-service"
            assert config.anomaly_detection.anomaly_threshold == 0.80
            assert config.predictive_analytics.failure_prediction_enabled is False
            assert config.llm_integration.primary_provider == "anthropic"
            assert config.llm_integration.max_tokens == 2000
            assert config.max_registered_systems == 100
            assert config.memory_limit_mb == 1024


class TestConfigFactories:
    """Test configuration factory functions."""

    @pytest.mark.unit
    def test_get_default_legacy_config(self):
        """Test getting default configuration."""
        config = get_default_legacy_config()

        assert isinstance(config, LegacySystemWhispererConfig)
        assert config.enabled is True
        assert config.service_name == "legacy-system-whisperer"

    @pytest.mark.unit
    def test_create_production_legacy_config(self):
        """Test creating production configuration."""
        config = create_production_legacy_config()

        # Production optimizations
        assert config.anomaly_detection.anomaly_threshold == 0.98
        assert config.anomaly_detection.enable_real_time_alerts is True
        assert config.anomaly_detection.cache_ttl_seconds == 7200
        assert config.predictive_analytics.min_historical_data_days == 90
        assert config.predictive_analytics.prediction_confidence_threshold == 0.85
        assert config.knowledge_capture.peer_review_required is True
        assert config.decision_support.multi_llm_consensus is True
        assert config.llm_integration.temperature == 0.05
        assert config.security.encrypt_sensitive_data is True

        # Higher limits
        assert config.max_registered_systems == 5000
        assert config.max_concurrent_analyses == 50
        assert config.memory_limit_mb == 8192

    @pytest.mark.unit
    def test_create_development_legacy_config(self):
        """Test creating development configuration."""
        config = create_development_legacy_config()

        # Development settings
        assert config.anomaly_detection.anomaly_threshold == 0.85
        assert config.anomaly_detection.cache_ttl_seconds == 300
        assert config.predictive_analytics.min_historical_data_days == 7
        assert config.knowledge_capture.peer_review_required is False
        assert config.decision_support.multi_llm_consensus is False
        assert config.llm_integration.temperature == 0.2
        assert config.security.gdpr_compliance is False

        # Lower limits
        assert config.max_registered_systems == 100
        assert config.max_concurrent_analyses == 5
        assert config.memory_limit_mb == 1024

    @pytest.mark.unit
    def test_extend_base_config_with_legacy(self):
        """Test extending base config with legacy settings."""
        base_config = Config()

        # Should not have legacy config initially
        assert not hasattr(base_config, "legacy_system_whisperer")

        # Extend with legacy config
        extended_config = extend_base_config_with_legacy(base_config)

        # Should now have legacy config
        assert hasattr(extended_config, "legacy_system_whisperer")
        assert isinstance(extended_config.legacy_system_whisperer, LegacySystemWhispererConfig)

    @pytest.mark.unit
    def test_validate_legacy_config_success(self):
        """Test successful config validation."""
        config = get_default_legacy_config()

        # Should not raise exception
        validate_legacy_config(config)

    @pytest.mark.unit
    def test_validate_legacy_config_failure(self):
        """Test config validation failure."""
        config = get_default_legacy_config()
        config.max_registered_systems = -10  # Invalid

        with pytest.raises(ConfigurationException):
            validate_legacy_config(config)


class TestConfigIntegration:
    """Test configuration integration scenarios."""

    @pytest.mark.integration
    def test_full_config_lifecycle(self):
        """Test complete configuration lifecycle."""
        # 1. Create config
        config = LegacySystemWhispererConfig(service_name="integration-test", max_registered_systems=50)

        # 2. Validate config
        config.validate()

        # 3. Serialize config
        config_dict = config.to_dict()
        assert config_dict["service_name"] == "integration-test"
        assert config_dict["max_registered_systems"] == 50

        # 4. Recreate from dict
        recreated_config = LegacySystemWhispererConfig.from_dict(config_dict)

        # 5. Verify recreated config
        assert recreated_config.service_name == "integration-test"
        assert recreated_config.max_registered_systems == 50

        # 6. Validate recreated config
        recreated_config.validate()

    @pytest.mark.integration
    def test_config_with_all_components(self):
        """Test configuration with all components configured."""
        config_dict = {
            "enabled": True,
            "service_name": "full-test-service",
            "max_registered_systems": 200,
            "anomaly_detection": {
                "anomaly_threshold": 0.90,
                "llm_analysis_enabled": True,
                "detection_interval_seconds": 180,
                "enable_real_time_alerts": True,
            },
            "predictive_analytics": {
                "failure_prediction_enabled": True,
                "min_historical_data_days": 21,
                "prediction_confidence_threshold": 0.80,
                "llm_prediction_enhancement": True,
            },
            "knowledge_capture": {
                "session_timeout_minutes": 90,
                "llm_processing_enabled": True,
                "knowledge_validation_threshold": 0.85,
                "peer_review_required": True,
            },
            "decision_support": {
                "recommendation_enabled": True,
                "max_recommendations": 8,
                "confidence_threshold": 0.75,
                "llm_recommendation_enhancement": True,
            },
            "llm_integration": {
                "primary_provider": "anthropic",
                "max_tokens": 3000,
                "temperature": 0.15,
                "response_caching": True,
            },
            "monitoring": {
                "metrics_collection_enabled": True,
                "prometheus_enabled": True,
                "health_check_enabled": True,
            },
            "security": {
                "encrypt_sensitive_data": True,
                "audit_logging": True,
                "gdpr_compliance": True,
            },
        }

        # Create config from comprehensive dict
        config = LegacySystemWhispererConfig.from_dict(config_dict)

        # Validate all components
        config.validate()

        # Verify all settings
        assert config.service_name == "full-test-service"
        assert config.max_registered_systems == 200
        assert config.anomaly_detection.anomaly_threshold == 0.90
        assert config.predictive_analytics.min_historical_data_days == 21
        assert config.knowledge_capture.session_timeout_minutes == 90
        assert config.decision_support.max_recommendations == 8
        assert config.llm_integration.primary_provider == "anthropic"
        assert config.monitoring.metrics_collection_enabled is True
        assert config.security.encrypt_sensitive_data is True


# Performance tests for configuration
class TestConfigPerformance:
    """Test configuration performance characteristics."""

    @pytest.mark.performance
    def test_config_creation_performance(self):
        """Test configuration creation performance."""
        import time

        start_time = time.time()

        # Create 100 configurations
        configs = []
        for i in range(100):
            config = LegacySystemWhispererConfig(service_name=f"perf_test_{i}", max_registered_systems=100 + i)
            configs.append(config)

        creation_time = time.time() - start_time

        # Should create 100 configs in reasonable time (< 0.1 seconds)
        assert creation_time < 0.1
        assert len(configs) == 100

    @pytest.mark.performance
    def test_config_validation_performance(self):
        """Test configuration validation performance."""
        import time

        config = create_production_legacy_config()

        start_time = time.time()

        # Validate 100 times
        for _ in range(100):
            config.validate()

        validation_time = time.time() - start_time

        # Should validate 100 times in reasonable time (< 0.1 seconds)
        assert validation_time < 0.1

    @pytest.mark.performance
    def test_config_serialization_performance(self):
        """Test configuration serialization performance."""
        import time

        config = create_production_legacy_config()

        start_time = time.time()

        # Serialize 100 times
        for _ in range(100):
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict)

        serialization_time = time.time() - start_time

        # Should serialize 100 times in reasonable time (< 0.1 seconds)
        assert serialization_time < 0.1


# Utility functions for config testing
def create_test_config_file(config_dict: dict) -> str:
    """Create temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        return f.name


def cleanup_test_config_file(file_path: str):
    """Clean up test config file."""
    Path(file_path).unlink(missing_ok=True)


def assert_config_equals(config1: LegacySystemWhispererConfig, config2: LegacySystemWhispererConfig):
    """Assert that two configurations are equal."""
    assert config1.enabled == config2.enabled
    assert config1.service_name == config2.service_name
    assert config1.max_registered_systems == config2.max_registered_systems
    assert config1.anomaly_detection.anomaly_threshold == config2.anomaly_detection.anomaly_threshold
    assert config1.llm_integration.primary_provider == config2.llm_integration.primary_provider
