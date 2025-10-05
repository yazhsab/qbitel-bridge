"""
Tests for custom exception classes.
Quick coverage wins by testing all exception types.
"""

import pytest

from ai_engine.core.exceptions import (
    CronosAIException,
    ConfigurationException,
    ModelException,
    ModelVersionException,
    DiscoveryException,
    InferenceException,
    TrainingException,
    DataException,
    ProtocolException,
    FieldDetectionException,
    AnomalyDetectionException,
    FeatureExtractionException,
    ModelRegistryException,
    ServingException,
    AIEngineException,
    ValidationException,
    ConfigException,
    PolicyException,
    ComplianceException,
    SecurityException,
    LLMException,
    TranslationException,
    ObservabilityException,
    MonitoringException,
    LoggingException,
    AlertException,
    HealthCheckException,
    SLAViolationException,
    DeploymentException,
    EnsembleException,
)


class TestCronosAIException:
    """Test base CronosAIException."""

    def test_basic_exception(self):
        """Test basic exception creation."""
        exc = CronosAIException("Test error")
        assert str(exc) == "Test error (Code: CRONOS_AI_ERROR)"
        assert exc.message == "Test error"
        assert exc.error_code == "CRONOS_AI_ERROR"
        assert exc.context == {}

    def test_exception_with_code(self):
        """Test exception with custom error code."""
        exc = CronosAIException("Test error", error_code="CUSTOM_CODE")
        assert exc.error_code == "CUSTOM_CODE"

    def test_exception_with_context(self):
        """Test exception with context data."""
        context = {"key": "value", "count": 42}
        exc = CronosAIException("Test error", context=context)
        assert exc.context == context
        assert "Context:" in str(exc)


class TestConfigurationException:
    """Test ConfigurationException."""

    def test_configuration_exception(self):
        """Test configuration exception creation."""
        exc = ConfigurationException("Invalid config")
        assert exc.error_code == "CONFIG_ERROR"
        assert exc.message == "Invalid config"

    def test_configuration_exception_with_key(self):
        """Test configuration exception with config key."""
        exc = ConfigurationException("Invalid value", config_key="database.host")
        assert exc.context["config_key"] == "database.host"


class TestModelException:
    """Test ModelException."""

    def test_model_exception(self):
        """Test model exception creation."""
        exc = ModelException("Model failed")
        assert exc.error_code == "MODEL_ERROR"

    def test_model_exception_with_details(self):
        """Test model exception with details."""
        exc = ModelException("Model failed", model_name="cnn_classifier", model_version="v1.2")
        assert exc.context["model_name"] == "cnn_classifier"
        assert exc.context["model_version"] == "v1.2"


class TestModelVersionException:
    """Test ModelVersionException."""

    def test_model_version_exception(self):
        """Test model version exception."""
        exc = ModelVersionException("Version not found")
        assert exc.error_code == "MODEL_VERSION_ERROR"

    def test_model_version_exception_with_details(self):
        """Test model version exception with details."""
        exc = ModelVersionException("Version conflict", version_id="v2.0", model_name="classifier")
        assert exc.context["version_id"] == "v2.0"
        assert exc.context["model_name"] == "classifier"


class TestDiscoveryException:
    """Test DiscoveryException."""

    def test_discovery_exception(self):
        """Test discovery exception."""
        exc = DiscoveryException("Discovery failed")
        assert exc.error_code == "DISCOVERY_ERROR"

    def test_discovery_exception_with_request_id(self):
        """Test discovery exception with request ID."""
        exc = DiscoveryException("Discovery failed", request_id="req-123")
        assert exc.context["request_id"] == "req-123"


class TestInferenceException:
    """Test InferenceException."""

    def test_inference_exception(self):
        """Test inference exception."""
        exc = InferenceException("Inference failed")
        assert exc.error_code == "INFERENCE_ERROR"

    def test_inference_exception_with_details(self):
        """Test inference exception with details."""
        exc = InferenceException("Invalid input", model_name="bert", input_shape=(32, 512))
        assert exc.context["model_name"] == "bert"
        assert exc.context["input_shape"] == (32, 512)


class TestTrainingException:
    """Test TrainingException."""

    def test_training_exception(self):
        """Test training exception."""
        exc = TrainingException("Training failed")
        assert exc.error_code == "TRAINING_ERROR"

    def test_training_exception_with_details(self):
        """Test training exception with training details."""
        exc = TrainingException("NaN loss", epoch=5, batch_idx=100, loss_value=float('inf'))
        assert exc.context["epoch"] == 5
        assert exc.context["batch_idx"] == 100
        assert exc.context["loss_value"] == float('inf')


class TestDataException:
    """Test DataException."""

    def test_data_exception(self):
        """Test data exception."""
        exc = DataException("Data processing failed")
        assert exc.error_code == "DATA_ERROR"

    def test_data_exception_with_details(self):
        """Test data exception with details."""
        exc = DataException("Invalid records", data_source="file.csv", record_count=1000)
        assert exc.context["data_source"] == "file.csv"
        assert exc.context["record_count"] == 1000


class TestProtocolException:
    """Test ProtocolException."""

    def test_protocol_exception(self):
        """Test protocol exception."""
        exc = ProtocolException("Protocol parsing failed")
        assert exc.error_code == "PROTOCOL_ERROR"

    def test_protocol_exception_with_details(self):
        """Test protocol exception with details."""
        exc = ProtocolException("Invalid packet", protocol_type="HTTP", packet_length=1024)
        assert exc.context["protocol_type"] == "HTTP"
        assert exc.context["packet_length"] == 1024


class TestFieldDetectionException:
    """Test FieldDetectionException."""

    def test_field_detection_exception(self):
        """Test field detection exception."""
        exc = FieldDetectionException("Field detection failed")
        assert exc.error_code == "FIELD_DETECTION_ERROR"

    def test_field_detection_exception_with_details(self):
        """Test field detection exception with details."""
        exc = FieldDetectionException("Too many fields", message_length=256, detected_fields=50)
        assert exc.context["message_length"] == 256
        assert exc.context["detected_fields"] == 50


class TestAnomalyDetectionException:
    """Test AnomalyDetectionException."""

    def test_anomaly_detection_exception(self):
        """Test anomaly detection exception."""
        exc = AnomalyDetectionException("Anomaly detection failed")
        assert exc.error_code == "ANOMALY_DETECTION_ERROR"

    def test_anomaly_detection_exception_with_details(self):
        """Test anomaly detection exception with details."""
        exc = AnomalyDetectionException("High anomaly score", detector_type="VAE", anomaly_score=0.95)
        assert exc.context["detector_type"] == "VAE"
        assert exc.context["anomaly_score"] == 0.95


class TestFeatureExtractionException:
    """Test FeatureExtractionException."""

    def test_feature_extraction_exception(self):
        """Test feature extraction exception."""
        exc = FeatureExtractionException("Feature extraction failed")
        assert exc.error_code == "FEATURE_EXTRACTION_ERROR"

    def test_feature_extraction_exception_with_details(self):
        """Test feature extraction exception with details."""
        exc = FeatureExtractionException("Invalid input", feature_type="spectral", input_size=1024)
        assert exc.context["feature_type"] == "spectral"
        assert exc.context["input_size"] == 1024


class TestModelRegistryException:
    """Test ModelRegistryException."""

    def test_model_registry_exception(self):
        """Test model registry exception."""
        exc = ModelRegistryException("Registry operation failed")
        assert exc.error_code == "MODEL_REGISTRY_ERROR"

    def test_model_registry_exception_with_details(self):
        """Test model registry exception with details."""
        exc = ModelRegistryException("Connection failed", registry_url="http://registry", operation="push")
        assert exc.context["registry_url"] == "http://registry"
        assert exc.context["operation"] == "push"


class TestServingException:
    """Test ServingException."""

    def test_serving_exception(self):
        """Test serving exception."""
        exc = ServingException("Serving failed")
        assert exc.error_code == "SERVING_ERROR"

    def test_serving_exception_with_details(self):
        """Test serving exception with details."""
        exc = ServingException("Request timeout", endpoint="/predict", request_id="req-456")
        assert exc.context["endpoint"] == "/predict"
        assert exc.context["request_id"] == "req-456"


class TestAIEngineException:
    """Test AIEngineException."""

    def test_ai_engine_exception(self):
        """Test AI engine exception."""
        exc = AIEngineException("Engine failed")
        assert exc.error_code == "AI_ENGINE_ERROR"
        assert str(exc) == "Engine failed"

    def test_ai_engine_exception_with_component(self):
        """Test AI engine exception with component."""
        exc = AIEngineException("Component failed", component="orchestrator")
        assert exc.context["component"] == "orchestrator"


class TestValidationException:
    """Test ValidationException."""

    def test_validation_exception(self):
        """Test validation exception."""
        exc = ValidationException("Validation failed")
        assert exc.context["component"] == "validation"

    def test_validation_exception_with_details(self):
        """Test validation exception with details."""
        details = {"min": 0, "max": 100, "actual": 150}
        exc = ValidationException("Value out of range", field="age", details=details)
        assert exc.context["field"] == "age"
        assert exc.context["details"] == details


class TestConfigException:
    """Test ConfigException."""

    def test_config_exception(self):
        """Test config exception."""
        exc = ConfigException("Config service error")
        assert exc.error_code == "CONFIG_SERVICE_ERROR"

    def test_config_exception_with_key(self):
        """Test config exception with key."""
        exc = ConfigException("Key not found", key="app.feature_flag")
        assert exc.context["config_key"] == "app.feature_flag"


class TestPolicyException:
    """Test PolicyException."""

    def test_policy_exception(self):
        """Test policy exception."""
        exc = PolicyException("Policy violation")
        assert exc.error_code == "POLICY_ERROR"

    def test_policy_exception_with_id(self):
        """Test policy exception with policy ID."""
        exc = PolicyException("Policy not found", policy_id="pol-123")
        assert exc.context["policy_id"] == "pol-123"


class TestComplianceException:
    """Test ComplianceException."""

    def test_compliance_exception(self):
        """Test compliance exception."""
        exc = ComplianceException("Compliance check failed")
        assert exc.error_code == "COMPLIANCE_ERROR"

    def test_compliance_exception_with_framework(self):
        """Test compliance exception with framework."""
        exc = ComplianceException("GDPR violation", framework="GDPR")
        assert exc.context["framework"] == "GDPR"


class TestSecurityException:
    """Test SecurityException."""

    def test_security_exception(self):
        """Test security exception."""
        exc = SecurityException("Security violation")
        assert exc.error_code == "SECURITY_ERROR"

    def test_security_exception_with_domain(self):
        """Test security exception with domain."""
        exc = SecurityException("Authentication failed", domain="auth")
        assert exc.context["domain"] == "auth"


class TestLLMException:
    """Test LLMException."""

    def test_llm_exception(self):
        """Test LLM exception."""
        exc = LLMException("LLM request failed")
        assert exc.error_code == "LLM_ERROR"

    def test_llm_exception_with_provider(self):
        """Test LLM exception with provider."""
        exc = LLMException("Rate limit exceeded", provider="openai")
        assert exc.context["provider"] == "openai"


class TestTranslationException:
    """Test TranslationException."""

    def test_translation_exception(self):
        """Test translation exception."""
        exc = TranslationException("Translation failed")
        assert exc.error_code == "TRANSLATION_ERROR"

    def test_translation_exception_with_id(self):
        """Test translation exception with ID."""
        exc = TranslationException("Translation not found", translation_id="trans-789")
        assert exc.context["translation_id"] == "trans-789"


class TestObservabilityException:
    """Test ObservabilityException."""

    def test_observability_exception(self):
        """Test observability exception."""
        exc = ObservabilityException("Observability failed")
        assert exc.error_code == "OBSERVABILITY_ERROR"

    def test_observability_exception_with_subsystem(self):
        """Test observability exception with subsystem."""
        exc = ObservabilityException("Metric collection failed", subsystem="metrics")
        assert exc.context["subsystem"] == "metrics"


class TestMonitoringException:
    """Test MonitoringException."""

    def test_monitoring_exception(self):
        """Test monitoring exception."""
        exc = MonitoringException("Monitoring failed")
        assert exc.context["subsystem"] == "monitoring"

    def test_monitoring_exception_with_metric(self):
        """Test monitoring exception with metric."""
        exc = MonitoringException("Metric not found", metric="cpu_usage")
        assert exc.context["metric"] == "cpu_usage"


class TestLoggingException:
    """Test LoggingException."""

    def test_logging_exception(self):
        """Test logging exception."""
        exc = LoggingException("Logging failed")
        assert exc.context["subsystem"] == "logging"

    def test_logging_exception_with_sink(self):
        """Test logging exception with sink."""
        exc = LoggingException("Sink connection failed", sink="elasticsearch")
        assert exc.context["sink"] == "elasticsearch"


class TestAlertException:
    """Test AlertException."""

    def test_alert_exception(self):
        """Test alert exception."""
        exc = AlertException("Alert failed")
        assert exc.context["subsystem"] == "alerts"

    def test_alert_exception_with_id(self):
        """Test alert exception with alert ID."""
        exc = AlertException("Alert not found", alert_id="alert-456")
        assert exc.context["alert_id"] == "alert-456"


class TestHealthCheckException:
    """Test HealthCheckException."""

    def test_health_check_exception(self):
        """Test health check exception."""
        exc = HealthCheckException("Health check failed")
        assert exc.context["subsystem"] == "health"

    def test_health_check_exception_with_component(self):
        """Test health check exception with component."""
        exc = HealthCheckException("Component unhealthy", component="database")
        assert exc.context["component"] == "database"


class TestSLAViolationException:
    """Test SLAViolationException."""

    def test_sla_violation_exception(self):
        """Test SLA violation exception."""
        exc = SLAViolationException("SLA violated")
        assert exc.error_code == "SLA_VIOLATION"

    def test_sla_violation_exception_with_threshold(self):
        """Test SLA violation exception with threshold."""
        exc = SLAViolationException("Response time exceeded", sla_threshold_ms=100.0)
        assert exc.context["sla_threshold_ms"] == 100.0


class TestDeploymentException:
    """Test DeploymentException."""

    def test_deployment_exception(self):
        """Test deployment exception."""
        exc = DeploymentException("Deployment failed")
        assert exc.error_code == "DEPLOYMENT_ERROR"

    def test_deployment_exception_with_id(self):
        """Test deployment exception with deployment ID."""
        exc = DeploymentException("Rollback required", deployment_id="deploy-789")
        assert exc.context["deployment_id"] == "deploy-789"


class TestEnsembleException:
    """Test EnsembleException."""

    def test_ensemble_exception(self):
        """Test ensemble exception."""
        exc = EnsembleException("Ensemble failed")
        assert exc.error_code == "ENSEMBLE_ERROR"

    def test_ensemble_exception_with_id(self):
        """Test ensemble exception with ensemble ID."""
        exc = EnsembleException("Model disagreement", ensemble_id="ens-456")
        assert exc.context["ensemble_id"] == "ens-456"


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""

    def test_all_inherit_from_cronos_ai_exception(self):
        """Test that all exceptions inherit from CronosAIException."""
        exceptions = [
            ConfigurationException,
            ModelException,
            InferenceException,
            DataException,
            ProtocolException,
            SecurityException,
            ComplianceException,
        ]

        for exc_class in exceptions:
            exc = exc_class("Test")
            assert isinstance(exc, CronosAIException)
            assert isinstance(exc, Exception)

    def test_observability_exceptions_inherit_correctly(self):
        """Test observability exception hierarchy."""
        exc = MonitoringException("Test")
        assert isinstance(exc, ObservabilityException)
        assert isinstance(exc, CronosAIException)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
