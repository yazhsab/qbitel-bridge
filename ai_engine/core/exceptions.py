"""
CRONOS AI Engine - Custom Exceptions

This module defines custom exception classes for the AI Engine.
"""

from typing import Any, Dict, Optional


class CronosAIException(Exception):
    """Base exception for all CRONOS AI Engine errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "CRONOS_AI_ERROR"
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} (Code: {self.error_code}, Context: {self.context})"
        return f"{self.message} (Code: {self.error_code})"


class ConfigurationException(CronosAIException):
    """Exception raised for configuration-related errors."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message,
            error_code="CONFIG_ERROR",
            context={"config_key": config_key} if config_key else {},
        )


class ModelException(CronosAIException):
    """Exception raised for model-related errors."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ):
        context = {}
        if model_name:
            context["model_name"] = model_name
        if model_version:
            context["model_version"] = model_version

        super().__init__(message, error_code="MODEL_ERROR", context=context)


class ModelVersionException(CronosAIException):
    """Exception raised for model version management issues."""

    def __init__(
        self,
        message: str,
        version_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        context = {}
        if version_id:
            context["version_id"] = version_id
        if model_name:
            context["model_name"] = model_name

        super().__init__(message, error_code="MODEL_VERSION_ERROR", context=context)


class DiscoveryException(CronosAIException):
    """Exception raised during protocol discovery workflows."""

    def __init__(self, message: str, request_id: Optional[str] = None):
        context = {"request_id": request_id} if request_id else {}
        super().__init__(message, error_code="DISCOVERY_ERROR", context=context)


class InferenceException(CronosAIException):
    """Exception raised during model inference."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        input_shape: Optional[tuple] = None,
    ):
        context = {}
        if model_name:
            context["model_name"] = model_name
        if input_shape:
            context["input_shape"] = input_shape

        super().__init__(message, error_code="INFERENCE_ERROR", context=context)


class TrainingException(CronosAIException):
    """Exception raised during model training."""

    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
        loss_value: Optional[float] = None,
    ):
        context = {}
        if epoch is not None:
            context["epoch"] = epoch
        if batch_idx is not None:
            context["batch_idx"] = batch_idx
        if loss_value is not None:
            context["loss_value"] = loss_value

        super().__init__(message, error_code="TRAINING_ERROR", context=context)


class DataException(CronosAIException):
    """Exception raised for data processing errors."""

    def __init__(
        self,
        message: str,
        data_source: Optional[str] = None,
        record_count: Optional[int] = None,
    ):
        context = {}
        if data_source:
            context["data_source"] = data_source
        if record_count is not None:
            context["record_count"] = record_count

        super().__init__(message, error_code="DATA_ERROR", context=context)


class ProtocolException(CronosAIException):
    """Exception raised for protocol analysis errors."""

    def __init__(
        self,
        message: str,
        protocol_type: Optional[str] = None,
        packet_length: Optional[int] = None,
    ):
        context = {}
        if protocol_type:
            context["protocol_type"] = protocol_type
        if packet_length is not None:
            context["packet_length"] = packet_length

        super().__init__(message, error_code="PROTOCOL_ERROR", context=context)


class FieldDetectionException(CronosAIException):
    """Exception raised for field detection errors."""

    def __init__(
        self,
        message: str,
        message_length: Optional[int] = None,
        detected_fields: Optional[int] = None,
    ):
        context = {}
        if message_length is not None:
            context["message_length"] = message_length
        if detected_fields is not None:
            context["detected_fields"] = detected_fields

        super().__init__(message, error_code="FIELD_DETECTION_ERROR", context=context)


class AnomalyDetectionException(CronosAIException):
    """Exception raised for anomaly detection errors."""

    def __init__(
        self,
        message: str,
        detector_type: Optional[str] = None,
        anomaly_score: Optional[float] = None,
    ):
        context = {}
        if detector_type:
            context["detector_type"] = detector_type
        if anomaly_score is not None:
            context["anomaly_score"] = anomaly_score

        super().__init__(message, error_code="ANOMALY_DETECTION_ERROR", context=context)


class FeatureExtractionException(CronosAIException):
    """Exception raised for feature extraction errors."""

    def __init__(
        self,
        message: str,
        feature_type: Optional[str] = None,
        input_size: Optional[int] = None,
    ):
        context = {}
        if feature_type:
            context["feature_type"] = feature_type
        if input_size is not None:
            context["input_size"] = input_size

        super().__init__(
            message, error_code="FEATURE_EXTRACTION_ERROR", context=context
        )


class ModelRegistryException(CronosAIException):
    """Exception raised for model registry operations."""

    def __init__(
        self,
        message: str,
        registry_url: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        context = {}
        if registry_url:
            context["registry_url"] = registry_url
        if operation:
            context["operation"] = operation

        super().__init__(message, error_code="MODEL_REGISTRY_ERROR", context=context)


class ServingException(CronosAIException):
    """Exception raised during model serving operations."""

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        context = {}
        if endpoint:
            context["endpoint"] = endpoint
        if request_id:
            context["request_id"] = request_id

        super().__init__(message, error_code="SERVING_ERROR", context=context)


class AIEngineException(CronosAIException):
    """High-level exception for AI Engine orchestration failures."""

    def __init__(self, message: str, component: Optional[str] = None):
        context = {"component": component} if component else {}
        super().__init__(message, error_code="AI_ENGINE_ERROR", context=context)

    def __str__(self) -> str:  # pragma: no cover - simple override
        return self.message


class ValidationException(AIEngineException):
    """Exception raised when validation logic fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        context: Dict[str, Any] = {}
        if field:
            context["field"] = field
        if details:
            context["details"] = details

        super().__init__(message, component="validation")
        self.context.update(context)


class ConfigException(CronosAIException):
    """Exception raised for configuration service errors."""

    def __init__(self, message: str, key: Optional[str] = None):
        context = {"config_key": key} if key else {}
        super().__init__(message, error_code="CONFIG_SERVICE_ERROR", context=context)


class PolicyException(CronosAIException):
    """Exception raised within the policy engine domain."""

    def __init__(self, message: str, policy_id: Optional[str] = None):
        context = {"policy_id": policy_id} if policy_id else {}
        super().__init__(message, error_code="POLICY_ERROR", context=context)


class ComplianceException(CronosAIException):
    """Exception raised for compliance assessment operations."""

    def __init__(self, message: str, framework: Optional[str] = None):
        context = {"framework": framework} if framework else {}
        super().__init__(message, error_code="COMPLIANCE_ERROR", context=context)


class SecurityException(CronosAIException):
    """Exception raised for security orchestration failures."""

    def __init__(self, message: str, domain: Optional[str] = None):
        context = {"domain": domain} if domain else {}
        super().__init__(message, error_code="SECURITY_ERROR", context=context)


class LLMException(CronosAIException):
    """LLM-related exception shared across providers."""

    def __init__(self, message: str, provider: Optional[str] = None):
        context = {"provider": provider} if provider else {}
        super().__init__(message, error_code="LLM_ERROR", context=context)


class TranslationException(CronosAIException):
    """Exception raised by translation studio endpoints."""

    def __init__(self, message: str, translation_id: Optional[str] = None):
        context = {"translation_id": translation_id} if translation_id else {}
        super().__init__(message, error_code="TRANSLATION_ERROR", context=context)


class ObservabilityException(CronosAIException):
    """Exception raised for observability subsystem failures."""

    def __init__(self, message: str, subsystem: Optional[str] = None):
        context = {"subsystem": subsystem} if subsystem else {}
        super().__init__(message, error_code="OBSERVABILITY_ERROR", context=context)


class MonitoringException(ObservabilityException):
    """Exception for monitoring metric collection issues."""

    def __init__(self, message: str, metric: Optional[str] = None):
        super().__init__(message, subsystem="monitoring")
        if metric:
            self.context["metric"] = metric


class LoggingException(ObservabilityException):
    """Exception for logging pipeline failures."""

    def __init__(self, message: str, sink: Optional[str] = None):
        super().__init__(message, subsystem="logging")
        if sink:
            self.context["sink"] = sink


class AlertException(ObservabilityException):
    """Exception raised while managing alerts."""

    def __init__(self, message: str, alert_id: Optional[str] = None):
        super().__init__(message, subsystem="alerts")
        if alert_id:
            self.context["alert_id"] = alert_id


class HealthCheckException(ObservabilityException):
    """Exception raised when health checks fail."""

    def __init__(self, message: str, component: Optional[str] = None):
        super().__init__(message, subsystem="health")
        if component:
            self.context["component"] = component


class SLAViolationException(CronosAIException):
    """Exception raised when service level objectives are violated."""

    def __init__(self, message: str, sla_threshold_ms: Optional[float] = None):
        context = {"sla_threshold_ms": sla_threshold_ms} if sla_threshold_ms else {}
        super().__init__(message, error_code="SLA_VIOLATION", context=context)


class DeploymentException(CronosAIException):
    """Exception raised for deployment lifecycle errors."""

    def __init__(self, message: str, deployment_id: Optional[str] = None):
        context = {"deployment_id": deployment_id} if deployment_id else {}
        super().__init__(message, error_code="DEPLOYMENT_ERROR", context=context)


class EnsembleException(CronosAIException):
    """Exception raised for ensemble model coordination errors."""

    def __init__(self, message: str, ensemble_id: Optional[str] = None):
        context = {"ensemble_id": ensemble_id} if ensemble_id else {}
        super().__init__(message, error_code="ENSEMBLE_ERROR", context=context)


class ThreatAnalysisException(CronosAIException):
    """Exception raised for threat analysis operations."""

    def __init__(self, message: str, threat_type: Optional[str] = None):
        context = {"threat_type": threat_type} if threat_type else {}
        super().__init__(message, error_code="THREAT_ANALYSIS_ERROR", context=context)
