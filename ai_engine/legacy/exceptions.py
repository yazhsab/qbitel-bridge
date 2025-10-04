"""
CRONOS AI Engine - Legacy System Whisperer Exceptions

Comprehensive exception handling for Legacy System Whisperer feature.
Provides structured error handling with proper categorization and context.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..core.exceptions import CronosAIException


class ErrorSeverity(Enum):
    """Error severity levels for Legacy System Whisperer."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification and handling."""

    SYSTEM_REGISTRATION = "system_registration"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTION_ANALYSIS = "prediction_analysis"
    KNOWLEDGE_CAPTURE = "knowledge_capture"
    DECISION_SUPPORT = "decision_support"
    MAINTENANCE_SCHEDULING = "maintenance_scheduling"
    LLM_SERVICE = "llm_service"
    DATA_PROCESSING = "data_processing"
    CONFIGURATION = "configuration"
    EXTERNAL_INTEGRATION = "external_integration"
    RESOURCE_LIMITATION = "resource_limitation"
    VALIDATION = "validation"


@dataclass
class ErrorContext:
    """Structured error context for detailed error information."""

    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    system_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    recovery_suggestions: Optional[List[str]] = None


class LegacySystemWhispererException(CronosAIException):
    """Base exception for Legacy System Whisperer feature."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.DATA_PROCESSING,
        system_id: Optional[str] = None,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        cause: Optional[Exception] = None,
    ):
        """Initialize Legacy System Whisperer exception."""
        super().__init__(message)

        self.severity = severity
        self.category = category
        self.system_id = system_id
        self.operation = operation
        self.component = component
        self.additional_context = additional_context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.cause = cause
        self.timestamp = datetime.now()

        # Generate unique error ID
        import uuid

        self.error_id = str(uuid.uuid4())[:8]

    def get_error_context(self) -> ErrorContext:
        """Get structured error context."""
        return ErrorContext(
            error_id=self.error_id,
            timestamp=self.timestamp,
            severity=self.severity,
            category=self.category,
            system_id=self.system_id,
            operation=self.operation,
            component=self.component,
            additional_context=self.additional_context,
            recovery_suggestions=self.recovery_suggestions,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_id": self.error_id,
            "message": str(self),
            "severity": self.severity.value,
            "category": self.category.value,
            "system_id": self.system_id,
            "operation": self.operation,
            "component": self.component,
            "timestamp": self.timestamp.isoformat(),
            "additional_context": self.additional_context,
            "recovery_suggestions": self.recovery_suggestions,
            "cause": str(self.cause) if self.cause else None,
        }


class SystemRegistrationException(LegacySystemWhispererException):
    """Exception for system registration errors."""

    def __init__(
        self,
        message: str,
        system_id: Optional[str] = None,
        system_name: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM_REGISTRATION,
            system_id=system_id,
            component="system_registry",
            additional_context={
                "system_name": system_name,
                "validation_errors": validation_errors or [],
            },
            recovery_suggestions=[
                "Verify system context information is complete",
                "Check system ID uniqueness",
                "Validate system type and criticality settings",
                "Ensure required system metadata is provided",
            ],
            **kwargs,
        )


class AnomalyDetectionException(LegacySystemWhispererException):
    """Exception for anomaly detection errors."""

    def __init__(
        self,
        message: str,
        system_id: Optional[str] = None,
        detector_type: Optional[str] = None,
        model_version: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.ANOMALY_DETECTION,
            system_id=system_id,
            component="enhanced_detector",
            additional_context={
                "detector_type": detector_type,
                "model_version": model_version,
            },
            recovery_suggestions=[
                "Check input data format and completeness",
                "Verify model initialization status",
                "Validate system context availability",
                "Consider fallback detection methods",
            ],
            **kwargs,
        )


class PredictionAnalysisException(LegacySystemWhispererException):
    """Exception for prediction analysis errors."""

    def __init__(
        self,
        message: str,
        system_id: Optional[str] = None,
        prediction_type: Optional[str] = None,
        time_horizon: Optional[str] = None,
        data_points: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.PREDICTION_ANALYSIS,
            system_id=system_id,
            component="failure_predictor",
            additional_context={
                "prediction_type": prediction_type,
                "time_horizon": time_horizon,
                "data_points": data_points,
            },
            recovery_suggestions=[
                "Verify sufficient historical data availability",
                "Check time series data quality and completeness",
                "Validate prediction horizon parameters",
                "Consider adjusting prediction model parameters",
            ],
            **kwargs,
        )


class KnowledgeCaptureException(LegacySystemWhispererException):
    """Exception for knowledge capture errors."""

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        expert_id: Optional[str] = None,
        knowledge_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.KNOWLEDGE_CAPTURE,
            component="knowledge_capture",
            additional_context={
                "session_id": session_id,
                "expert_id": expert_id,
                "knowledge_type": knowledge_type,
            },
            recovery_suggestions=[
                "Verify expert input format and content",
                "Check LLM service availability and connectivity",
                "Validate knowledge session state",
                "Consider manual knowledge entry as fallback",
            ],
            **kwargs,
        )


class DecisionSupportException(LegacySystemWhispererException):
    """Exception for decision support errors."""

    def __init__(
        self,
        message: str,
        system_id: Optional[str] = None,
        decision_category: Optional[str] = None,
        analysis_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.DECISION_SUPPORT,
            system_id=system_id,
            component="decision_support",
            additional_context={
                "decision_category": decision_category,
                "analysis_type": analysis_type,
            },
            recovery_suggestions=[
                "Verify decision context completeness",
                "Check system analysis data availability",
                "Validate decision objectives and constraints",
                "Consider simplified decision criteria",
            ],
            **kwargs,
        )


class MaintenanceSchedulingException(LegacySystemWhispererException):
    """Exception for maintenance scheduling errors."""

    def __init__(
        self,
        message: str,
        system_id: Optional[str] = None,
        maintenance_count: Optional[int] = None,
        optimization_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.MAINTENANCE_SCHEDULING,
            system_id=system_id,
            component="maintenance_scheduler",
            additional_context={
                "maintenance_count": maintenance_count,
                "optimization_type": optimization_type,
            },
            recovery_suggestions=[
                "Verify maintenance request format and completeness",
                "Check resource constraint validity",
                "Validate business scheduling constraints",
                "Consider simplified scheduling approach",
            ],
            **kwargs,
        )


class LLMServiceException(LegacySystemWhispererException):
    """Exception for LLM service integration errors."""

    def __init__(
        self,
        message: str,
        llm_provider: Optional[str] = None,
        request_type: Optional[str] = None,
        response_status: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.LLM_SERVICE,
            component="llm_integration",
            additional_context={
                "llm_provider": llm_provider,
                "request_type": request_type,
                "response_status": response_status,
            },
            recovery_suggestions=[
                "Check LLM service connectivity and authentication",
                "Verify request format and parameters",
                "Consider alternative LLM provider",
                "Use cached responses if available",
                "Implement fallback logic for critical operations",
            ],
            **kwargs,
        )


class DataProcessingException(LegacySystemWhispererException):
    """Exception for data processing errors."""

    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        processing_stage: Optional[str] = None,
        data_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_PROCESSING,
            component="data_processor",
            additional_context={
                "data_type": data_type,
                "processing_stage": processing_stage,
                "data_size": data_size,
            },
            recovery_suggestions=[
                "Verify input data format and structure",
                "Check data completeness and quality",
                "Validate processing parameters",
                "Consider data preprocessing steps",
                "Implement data validation checkpoints",
            ],
            **kwargs,
        )


class ConfigurationException(LegacySystemWhispererException):
    """Exception for configuration errors."""

    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        config_parameter: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            component="configuration",
            additional_context={
                "config_section": config_section,
                "config_parameter": config_parameter,
            },
            recovery_suggestions=[
                "Verify configuration file syntax and format",
                "Check required configuration parameters",
                "Validate configuration values and ranges",
                "Use default configuration values if available",
                "Consult configuration documentation",
            ],
            **kwargs,
        )


class ExternalIntegrationException(LegacySystemWhispererException):
    """Exception for external system integration errors."""

    def __init__(
        self,
        message: str,
        integration_type: Optional[str] = None,
        external_service: Optional[str] = None,
        connection_status: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_INTEGRATION,
            component="external_integration",
            additional_context={
                "integration_type": integration_type,
                "external_service": external_service,
                "connection_status": connection_status,
            },
            recovery_suggestions=[
                "Check external service availability and status",
                "Verify network connectivity and authentication",
                "Validate integration configuration",
                "Consider retry mechanisms with exponential backoff",
                "Implement circuit breaker patterns",
            ],
            **kwargs,
        )


class ResourceLimitationException(LegacySystemWhispererException):
    """Exception for resource limitation errors."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[str] = None,
        limit: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE_LIMITATION,
            severity=ErrorSeverity.HIGH,
            component="resource_manager",
            additional_context={
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit,
            },
            recovery_suggestions=[
                "Monitor and optimize resource usage",
                "Consider resource cleanup and optimization",
                "Implement resource pooling and reuse",
                "Scale resources if possible",
                "Implement graceful degradation",
            ],
            **kwargs,
        )


class ValidationException(LegacySystemWhispererException):
    """Exception for validation errors."""

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        field_name: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            component="validator",
            additional_context={
                "validation_type": validation_type,
                "field_name": field_name,
                "validation_errors": validation_errors or [],
            },
            recovery_suggestions=[
                "Check input data format and required fields",
                "Verify data types and value ranges",
                "Validate business rules and constraints",
                "Review validation error messages",
                "Correct input data and retry operation",
            ],
            **kwargs,
        )


class ErrorHandler:
    """Centralized error handling for Legacy System Whisperer."""

    def __init__(self, logger):
        """Initialize error handler."""
        self.logger = logger
        self.error_statistics = {
            "total_errors": 0,
            "by_severity": {severity.value: 0 for severity in ErrorSeverity},
            "by_category": {category.value: 0 for category in ErrorCategory},
            "by_component": {},
        }

    def handle_exception(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """
        Handle exception with proper logging and statistics tracking.

        Args:
            exception: The exception to handle
            context: Additional context information

        Returns:
            Structured error context
        """

        if isinstance(exception, LegacySystemWhispererException):
            return self._handle_legacy_exception(exception, context)
        else:
            return self._handle_generic_exception(exception, context)

    def _handle_legacy_exception(
        self,
        exception: LegacySystemWhispererException,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorContext:
        """Handle Legacy System Whisperer specific exceptions."""

        error_context = exception.get_error_context()

        # Update statistics
        self._update_error_statistics(exception)

        # Log with appropriate level based on severity
        log_level = self._get_log_level(exception.severity)
        self.logger.log(
            log_level,
            f"Legacy System Whisperer Error [{error_context.error_id}]: {exception}",
            extra={
                "error_context": error_context,
                "exception_dict": exception.to_dict(),
                "additional_context": context,
            },
        )

        return error_context

    def _handle_generic_exception(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """Handle generic exceptions."""

        import uuid

        error_context = ErrorContext(
            error_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_PROCESSING,
            additional_context=context,
            recovery_suggestions=[
                "Check logs for detailed error information",
                "Verify system state and configuration",
                "Contact support if problem persists",
            ],
        )

        # Update statistics
        self.error_statistics["total_errors"] += 1
        self.error_statistics["by_severity"][ErrorSeverity.MEDIUM.value] += 1
        self.error_statistics["by_category"][ErrorCategory.DATA_PROCESSING.value] += 1

        # Log error
        self.logger.error(
            f"Unhandled Exception [{error_context.error_id}]: {exception}",
            extra={
                "error_context": error_context,
                "exception_type": type(exception).__name__,
                "additional_context": context,
            },
            exc_info=True,
        )

        return error_context

    def _update_error_statistics(
        self, exception: LegacySystemWhispererException
    ) -> None:
        """Update error statistics."""

        self.error_statistics["total_errors"] += 1
        self.error_statistics["by_severity"][exception.severity.value] += 1
        self.error_statistics["by_category"][exception.category.value] += 1

        if exception.component:
            if exception.component not in self.error_statistics["by_component"]:
                self.error_statistics["by_component"][exception.component] = 0
            self.error_statistics["by_component"][exception.component] += 1

    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """Get logging level based on error severity."""

        import logging

        level_mapping = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }

        return level_mapping.get(severity, logging.ERROR)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return self.error_statistics.copy()

    def reset_statistics(self) -> None:
        """Reset error statistics."""
        self.error_statistics = {
            "total_errors": 0,
            "by_severity": {severity.value: 0 for severity in ErrorSeverity},
            "by_category": {category.value: 0 for category in ErrorCategory},
            "by_component": {},
        }


def create_error_handler(logger) -> ErrorHandler:
    """Create error handler instance."""
    return ErrorHandler(logger)


def handle_with_recovery(
    operation_func, recovery_func=None, max_retries: int = 3, logger=None
):
    """
    Decorator for handling operations with recovery mechanisms.

    Args:
        operation_func: Function to execute
        recovery_func: Recovery function to call on error
        max_retries: Maximum number of retry attempts
        logger: Logger instance
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except LegacySystemWhispererException as e:
                    last_exception = e

                    if logger:
                        logger.warning(
                            f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}",
                            extra={"error_context": e.get_error_context()},
                        )

                    # Try recovery if available and not last attempt
                    if recovery_func and attempt < max_retries:
                        try:
                            await recovery_func(e, attempt)
                        except Exception as recovery_error:
                            if logger:
                                logger.error(f"Recovery failed: {recovery_error}")

                    # If last attempt, re-raise
                    if attempt == max_retries:
                        raise e

                except Exception as e:
                    # For non-Legacy System Whisperer exceptions, convert and re-raise
                    legacy_exception = LegacySystemWhispererException(
                        message=f"Unexpected error in {func.__name__}: {e}",
                        severity=ErrorSeverity.HIGH,
                        operation=func.__name__,
                        cause=e,
                    )
                    raise legacy_exception

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator
