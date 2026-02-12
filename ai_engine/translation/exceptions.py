"""
QBITEL - Translation Studio Exceptions
Enterprise-grade exception handling for protocol translation, API generation, and SDK creation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import traceback
from datetime import datetime, timezone


class ErrorCategory(Enum):
    """Error categories for translation operations."""

    PROTOCOL_DISCOVERY = "protocol_discovery"
    API_GENERATION = "api_generation"
    CODE_GENERATION = "code_generation"
    PROTOCOL_TRANSLATION = "protocol_translation"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    RESOURCE_LIMIT = "resource_limit"
    EXTERNAL_SERVICE = "external_service"
    DATA_CORRUPTION = "data_corruption"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    INTERNAL = "internal"


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for errors."""

    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    protocol_type: Optional[str] = None
    api_style: Optional[str] = None
    target_language: Optional[str] = None
    processing_stage: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TranslationStudioException(Exception):
    """Base exception for Translation Studio operations."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        error_code: Optional[str] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
        self.error_code = error_code or self._generate_error_code()
        self.retry_after = retry_after
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)
        self.traceback_str = traceback.format_exc() if cause else None

    def _generate_error_code(self) -> str:
        """Generate a unique error code."""
        import uuid

        return f"TS_{self.category.value.upper()}_{str(uuid.uuid4())[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": {
                "request_id": self.context.request_id,
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "component": self.context.component,
                "operation": self.context.operation,
                "protocol_type": self.context.protocol_type,
                "api_style": self.context.api_style,
                "target_language": self.context.target_language,
                "processing_stage": self.context.processing_stage,
                "metadata": self.context.metadata,
            },
            "details": self.details,
            "retry_after": self.retry_after,
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback_str,
        }

    def get_http_status_code(self) -> int:
        """Get appropriate HTTP status code for this exception."""
        status_mapping = {
            ErrorCategory.AUTHENTICATION: 401,
            ErrorCategory.PERMISSION: 403,
            ErrorCategory.NOT_FOUND: 404,
            ErrorCategory.CONFLICT: 409,
            ErrorCategory.VALIDATION: 400,
            ErrorCategory.RATE_LIMIT: 429,
            ErrorCategory.RESOURCE_LIMIT: 413,
            ErrorCategory.TIMEOUT: 408,
            ErrorCategory.EXTERNAL_SERVICE: 502,
            ErrorCategory.NETWORK: 503,
            ErrorCategory.CONFIGURATION: 500,
            ErrorCategory.DATA_CORRUPTION: 500,
            ErrorCategory.INTERNAL: 500,
        }
        return status_mapping.get(self.category, 500)


# Protocol Discovery Exceptions


class ProtocolDiscoveryException(TranslationStudioException):
    """Base exception for protocol discovery operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.PROTOCOL_DISCOVERY, **kwargs)


class ProtocolNotSupportedException(ProtocolDiscoveryException):
    """Exception raised when a protocol is not supported."""

    def __init__(self, protocol_name: str, supported_protocols: List[str] = None, **kwargs):
        self.protocol_name = protocol_name
        self.supported_protocols = supported_protocols or []

        message = f"Protocol '{protocol_name}' is not supported"
        if self.supported_protocols:
            message += f". Supported protocols: {', '.join(self.supported_protocols)}"

        super().__init__(
            message,
            details={
                "protocol_name": protocol_name,
                "supported_protocols": self.supported_protocols,
            },
            **kwargs,
        )


class ProtocolAnalysisException(ProtocolDiscoveryException):
    """Exception raised during protocol analysis."""

    def __init__(self, analysis_stage: str, **kwargs):
        self.analysis_stage = analysis_stage
        super().__init__(
            f"Protocol analysis failed at stage: {analysis_stage}",
            details={"analysis_stage": analysis_stage},
            **kwargs,
        )


class InsufficientProtocolDataException(ProtocolDiscoveryException):
    """Exception raised when insufficient data is provided for protocol discovery."""

    def __init__(self, required_samples: int, provided_samples: int, **kwargs):
        self.required_samples = required_samples
        self.provided_samples = provided_samples

        super().__init__(
            f"Insufficient protocol data. Required: {required_samples}, Provided: {provided_samples}",
            details={
                "required_samples": required_samples,
                "provided_samples": provided_samples,
            },
            **kwargs,
        )


# API Generation Exceptions


class APIGenerationException(TranslationStudioException):
    """Base exception for API generation operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.API_GENERATION, **kwargs)


class APISpecificationInvalidException(APIGenerationException):
    """Exception raised when API specification is invalid."""

    def __init__(self, validation_errors: List[str], **kwargs):
        self.validation_errors = validation_errors

        super().__init__(
            f"API specification validation failed: {'; '.join(validation_errors)}",
            details={"validation_errors": validation_errors},
            **kwargs,
        )


class UnsupportedAPIStyleException(APIGenerationException):
    """Exception raised for unsupported API styles."""

    def __init__(self, api_style: str, supported_styles: List[str] = None, **kwargs):
        self.api_style = api_style
        self.supported_styles = supported_styles or []

        message = f"API style '{api_style}' is not supported"
        if self.supported_styles:
            message += f". Supported styles: {', '.join(self.supported_styles)}"

        super().__init__(
            message,
            details={"api_style": api_style, "supported_styles": self.supported_styles},
            **kwargs,
        )


class SecurityConfigurationException(APIGenerationException):
    """Exception raised for security configuration errors."""

    def __init__(self, security_issue: str, **kwargs):
        super().__init__(
            f"Security configuration error: {security_issue}",
            severity=ErrorSeverity.HIGH,
            details={"security_issue": security_issue},
            **kwargs,
        )


# Code Generation Exceptions


class CodeGenerationException(TranslationStudioException):
    """Base exception for code generation operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CODE_GENERATION, **kwargs)


class UnsupportedLanguageException(CodeGenerationException):
    """Exception raised for unsupported programming languages."""

    def __init__(self, language: str, supported_languages: List[str] = None, **kwargs):
        self.language = language
        self.supported_languages = supported_languages or []

        message = f"Programming language '{language}' is not supported"
        if self.supported_languages:
            message += f". Supported languages: {', '.join(self.supported_languages)}"

        super().__init__(
            message,
            details={
                "language": language,
                "supported_languages": self.supported_languages,
            },
            **kwargs,
        )


class CodeCompilationException(CodeGenerationException):
    """Exception raised when generated code fails compilation/validation."""

    def __init__(self, language: str, compilation_errors: List[str], **kwargs):
        self.language = language
        self.compilation_errors = compilation_errors

        super().__init__(
            f"Generated {language} code failed compilation: {'; '.join(compilation_errors)}",
            details={"language": language, "compilation_errors": compilation_errors},
            **kwargs,
        )


class TemplateRenderingException(CodeGenerationException):
    """Exception raised during template rendering."""

    def __init__(self, template_name: str, rendering_error: str, **kwargs):
        self.template_name = template_name
        self.rendering_error = rendering_error

        super().__init__(
            f"Template '{template_name}' rendering failed: {rendering_error}",
            details={
                "template_name": template_name,
                "rendering_error": rendering_error,
            },
            **kwargs,
        )


# Protocol Translation Exceptions


class ProtocolTranslationException(TranslationStudioException):
    """Base exception for protocol translation operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.PROTOCOL_TRANSLATION, **kwargs)


class TranslationMappingException(ProtocolTranslationException):
    """Exception raised when protocol mapping fails."""

    def __init__(self, source_protocol: str, target_protocol: str, mapping_error: str, **kwargs):
        self.source_protocol = source_protocol
        self.target_protocol = target_protocol
        self.mapping_error = mapping_error

        super().__init__(
            f"Translation mapping failed from {source_protocol} to {target_protocol}: {mapping_error}",
            details={
                "source_protocol": source_protocol,
                "target_protocol": target_protocol,
                "mapping_error": mapping_error,
            },
            **kwargs,
        )


class DataIntegrityException(ProtocolTranslationException):
    """Exception raised when data integrity is compromised during translation."""

    def __init__(self, integrity_check: str, **kwargs):
        super().__init__(
            f"Data integrity check failed: {integrity_check}",
            severity=ErrorSeverity.HIGH,
            details={"integrity_check": integrity_check},
            **kwargs,
        )


class TranslationQualityException(ProtocolTranslationException):
    """Exception raised when translation quality is below threshold."""

    def __init__(self, quality_score: float, min_threshold: float, **kwargs):
        self.quality_score = quality_score
        self.min_threshold = min_threshold

        super().__init__(
            f"Translation quality {quality_score:.2f} below minimum threshold {min_threshold:.2f}",
            details={"quality_score": quality_score, "min_threshold": min_threshold},
            **kwargs,
        )


# Validation Exceptions


class ValidationException(TranslationStudioException):
    """Base exception for validation errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)


class SchemaValidationException(ValidationException):
    """Exception raised for schema validation errors."""

    def __init__(self, schema_errors: List[str], **kwargs):
        self.schema_errors = schema_errors

        super().__init__(
            f"Schema validation failed: {'; '.join(schema_errors)}",
            details={"schema_errors": schema_errors},
            **kwargs,
        )


class InputValidationException(ValidationException):
    """Exception raised for input validation errors."""

    def __init__(self, field_name: str, validation_error: str, **kwargs):
        self.field_name = field_name
        self.validation_error = validation_error

        super().__init__(
            f"Input validation failed for field '{field_name}': {validation_error}",
            details={"field_name": field_name, "validation_error": validation_error},
            **kwargs,
        )


# Resource and Rate Limiting Exceptions


class ResourceLimitException(TranslationStudioException):
    """Exception raised when resource limits are exceeded."""

    def __init__(self, resource_type: str, limit: int, current_usage: int, **kwargs):
        self.resource_type = resource_type
        self.limit = limit
        self.current_usage = current_usage

        super().__init__(
            f"{resource_type} limit exceeded: {current_usage}/{limit}",
            category=ErrorCategory.RESOURCE_LIMIT,
            details={
                "resource_type": resource_type,
                "limit": limit,
                "current_usage": current_usage,
            },
            **kwargs,
        )


class RateLimitException(TranslationStudioException):
    """Exception raised when rate limits are exceeded."""

    def __init__(self, operation: str, limit: int, window: int, retry_after: int = 60, **kwargs):
        self.operation = operation
        self.limit = limit
        self.window = window

        super().__init__(
            f"Rate limit exceeded for {operation}: {limit} requests per {window}s",
            category=ErrorCategory.RATE_LIMIT,
            retry_after=retry_after,
            details={"operation": operation, "limit": limit, "window": window},
            **kwargs,
        )


# External Service Exceptions


class ExternalServiceException(TranslationStudioException):
    """Exception raised for external service errors."""

    def __init__(self, service_name: str, service_error: str, **kwargs):
        self.service_name = service_name
        self.service_error = service_error

        super().__init__(
            f"External service '{service_name}' error: {service_error}",
            category=ErrorCategory.EXTERNAL_SERVICE,
            details={"service_name": service_name, "service_error": service_error},
            **kwargs,
        )


class LLMServiceException(ExternalServiceException):
    """Exception raised for LLM service errors."""

    def __init__(self, llm_provider: str, llm_error: str, **kwargs):
        super().__init__(service_name=f"LLM_{llm_provider}", service_error=llm_error, **kwargs)


class RAGEngineException(ExternalServiceException):
    """Exception raised for RAG engine errors."""

    def __init__(self, rag_error: str, **kwargs):
        super().__init__(service_name="RAG_Engine", service_error=rag_error, **kwargs)


# Timeout and Network Exceptions


class TimeoutException(TranslationStudioException):
    """Exception raised for timeout errors."""

    def __init__(self, operation: str, timeout_seconds: int, **kwargs):
        self.operation = operation
        self.timeout_seconds = timeout_seconds

        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds}s",
            category=ErrorCategory.TIMEOUT,
            details={"operation": operation, "timeout_seconds": timeout_seconds},
            **kwargs,
        )


class NetworkException(TranslationStudioException):
    """Exception raised for network-related errors."""

    def __init__(self, network_error: str, **kwargs):
        super().__init__(
            f"Network error: {network_error}",
            category=ErrorCategory.NETWORK,
            details={"network_error": network_error},
            **kwargs,
        )


# Configuration Exceptions


class ConfigurationException(TranslationStudioException):
    """Exception raised for configuration errors."""

    def __init__(self, config_issue: str, **kwargs):
        super().__init__(
            f"Configuration error: {config_issue}",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            details={"config_issue": config_issue},
            **kwargs,
        )


# Helper Functions


def create_error_context(
    request_id: str = None,
    user_id: str = None,
    component: str = None,
    operation: str = None,
    **kwargs,
) -> ErrorContext:
    """Create an ErrorContext with common fields."""
    return ErrorContext(
        request_id=request_id,
        user_id=user_id,
        component=component,
        operation=operation,
        **kwargs,
    )


def handle_external_exception(
    exception: Exception,
    context: ErrorContext,
    default_category: ErrorCategory = ErrorCategory.INTERNAL,
) -> TranslationStudioException:
    """Convert external exceptions to TranslationStudioException."""
    if isinstance(exception, TranslationStudioException):
        return exception

    # Map common external exceptions
    exception_mapping = {
        ValueError: ErrorCategory.VALIDATION,
        TypeError: ErrorCategory.VALIDATION,
        KeyError: ErrorCategory.NOT_FOUND,
        FileNotFoundError: ErrorCategory.NOT_FOUND,
        PermissionError: ErrorCategory.PERMISSION,
        TimeoutError: ErrorCategory.TIMEOUT,
        ConnectionError: ErrorCategory.NETWORK,
        ImportError: ErrorCategory.CONFIGURATION,
        AttributeError: ErrorCategory.CONFIGURATION,
    }

    category = exception_mapping.get(type(exception), default_category)

    return TranslationStudioException(message=str(exception), category=category, context=context, cause=exception)


def raise_for_validation_errors(errors: List[str], context: ErrorContext = None):
    """Raise ValidationException if errors exist."""
    if errors:
        raise ValidationException(
            f"Validation failed: {'; '.join(errors)}",
            context=context,
            details={"validation_errors": errors},
        )


def raise_for_insufficient_confidence(
    confidence: float,
    minimum_confidence: float,
    operation: str,
    context: ErrorContext = None,
):
    """Raise exception for insufficient confidence scores."""
    if confidence < minimum_confidence:
        raise TranslationQualityException(
            quality_score=confidence,
            min_threshold=minimum_confidence,
            context=context,
            details={"operation": operation},
        )


__all__ = [
    # Enums
    "ErrorCategory",
    "ErrorSeverity",
    # Base classes
    "ErrorContext",
    "TranslationStudioException",
    # Protocol Discovery exceptions
    "ProtocolDiscoveryException",
    "ProtocolNotSupportedException",
    "ProtocolAnalysisException",
    "InsufficientProtocolDataException",
    # API Generation exceptions
    "APIGenerationException",
    "APISpecificationInvalidException",
    "UnsupportedAPIStyleException",
    "SecurityConfigurationException",
    # Code Generation exceptions
    "CodeGenerationException",
    "UnsupportedLanguageException",
    "CodeCompilationException",
    "TemplateRenderingException",
    # Protocol Translation exceptions
    "ProtocolTranslationException",
    "TranslationMappingException",
    "DataIntegrityException",
    "TranslationQualityException",
    # Validation exceptions
    "ValidationException",
    "SchemaValidationException",
    "InputValidationException",
    # Resource exceptions
    "ResourceLimitException",
    "RateLimitException",
    # External service exceptions
    "ExternalServiceException",
    "LLMServiceException",
    "RAGEngineException",
    # Timeout and Network exceptions
    "TimeoutException",
    "NetworkException",
    # Configuration exceptions
    "ConfigurationException",
    # Helper functions
    "create_error_context",
    "handle_external_exception",
    "raise_for_validation_errors",
    "raise_for_insufficient_confidence",
]
