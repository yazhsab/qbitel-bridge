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
