"""
QBITEL Engine - FastAPI Exception Handlers

This module provides centralized exception handling for the REST API,
converting QbitelAIException and other exceptions to standardized JSON responses.
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional
from http import HTTPStatus

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError

from ..core.exceptions import (
    QbitelAIException,
    DiscoveryException,
    ProtocolException,
    FieldDetectionException,
    AnomalyDetectionException,
    LLMException,
    ValidationException,
    SecurityException,
    ComplianceException,
    ConfigurationException,
    ModelException,
    InferenceException,
)
from ..core.error_codes import (
    ErrorCode,
    get_http_status_for_error,
    is_retryable,
)
from ..core.circuit_breakers import CircuitOpenError

logger = logging.getLogger(__name__)


# Exception to ErrorCode mapping
EXCEPTION_TO_ERROR_CODE: Dict[type, ErrorCode] = {
    DiscoveryException: ErrorCode.DISCOVERY_FAILED,
    ProtocolException: ErrorCode.PROTOCOL_ERROR,
    FieldDetectionException: ErrorCode.FIELD_DETECTION_FAILED,
    AnomalyDetectionException: ErrorCode.ANOMALY_DETECTION_FAILED,
    LLMException: ErrorCode.LLM_PROVIDER_ERROR,
    ValidationException: ErrorCode.VALIDATION_ERROR,
    SecurityException: ErrorCode.UNAUTHORIZED,
    ComplianceException: ErrorCode.COMPLIANCE_ERROR,
    ConfigurationException: ErrorCode.CONFIGURATION_ERROR,
    ModelException: ErrorCode.MODEL_NOT_LOADED,
    InferenceException: ErrorCode.FIELD_DETECTION_FAILED,
}


def get_error_code_for_exception(exc: Exception) -> ErrorCode:
    """Get the appropriate error code for an exception."""
    exc_type = type(exc)

    # Direct match
    if exc_type in EXCEPTION_TO_ERROR_CODE:
        return EXCEPTION_TO_ERROR_CODE[exc_type]

    # Check inheritance
    for base_type, error_code in EXCEPTION_TO_ERROR_CODE.items():
        if isinstance(exc, base_type):
            return error_code

    # Default to internal error
    return ErrorCode.INTERNAL_ERROR


def create_error_response(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    request_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a standardized error response body."""
    http_status = get_http_status_for_error(error_code)

    response = {
        "error": {
            "code": int(error_code),
            "name": error_code.name,
            "message": message,
            "http_status": http_status.value,
            "timestamp": time.time(),
        }
    }

    if details:
        response["error"]["details"] = details

    if correlation_id:
        response["error"]["correlation_id"] = correlation_id

    if request_path:
        response["error"]["path"] = request_path

    if is_retryable(error_code):
        response["error"]["retryable"] = True
        response["error"]["retry_after"] = 60  # Default retry after 60 seconds

    return response


async def qbitel_exception_handler(
    request: Request, exc: QbitelAIException
) -> JSONResponse:
    """Handle QbitelAIException and its subclasses."""
    correlation_id = getattr(request.state, "correlation_id", None)
    error_code = get_error_code_for_exception(exc)
    http_status = get_http_status_for_error(error_code)

    # Log the error
    logger.error(
        f"QbitelAIException: {error_code.name} - {exc.message}",
        extra={
            "error_code": int(error_code),
            "correlation_id": correlation_id,
            "path": request.url.path,
            "context": exc.context,
        },
    )

    response_body = create_error_response(
        error_code=error_code,
        message=exc.message,
        details=exc.context if exc.context else None,
        correlation_id=correlation_id,
        request_path=request.url.path,
    )

    headers = {}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    if is_retryable(error_code):
        headers["Retry-After"] = "60"

    return JSONResponse(
        status_code=http_status.value,
        content=response_body,
        headers=headers,
    )


async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Handle FastAPI/Starlette HTTPException."""
    correlation_id = getattr(request.state, "correlation_id", None)

    # Map HTTP status to error code
    status_to_error_code = {
        400: ErrorCode.VALIDATION_ERROR,
        401: ErrorCode.UNAUTHORIZED,
        403: ErrorCode.INSUFFICIENT_PERMISSIONS,
        404: ErrorCode.DISCOVERY_SESSION_NOT_FOUND,
        405: ErrorCode.NOT_IMPLEMENTED,
        408: ErrorCode.TIMEOUT,
        413: ErrorCode.PAYLOAD_TOO_LARGE,
        415: ErrorCode.UNSUPPORTED_CONTENT_TYPE,
        422: ErrorCode.INVALID_INPUT,
        429: ErrorCode.LLM_RATE_LIMITED,
        500: ErrorCode.INTERNAL_ERROR,
        501: ErrorCode.NOT_IMPLEMENTED,
        502: ErrorCode.EXTERNAL_SERVICE_ERROR,
        503: ErrorCode.SERVICE_UNAVAILABLE,
        504: ErrorCode.TIMEOUT,
    }

    error_code = status_to_error_code.get(exc.status_code, ErrorCode.INTERNAL_ERROR)

    logger.warning(
        f"HTTPException: {exc.status_code} - {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "correlation_id": correlation_id,
            "path": request.url.path,
        },
    )

    response_body = create_error_response(
        error_code=error_code,
        message=str(exc.detail) if exc.detail else HTTPStatus(exc.status_code).phrase,
        correlation_id=correlation_id,
        request_path=request.url.path,
    )

    headers = {}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return JSONResponse(
        status_code=exc.status_code,
        content=response_body,
        headers=headers,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    correlation_id = getattr(request.state, "correlation_id", None)

    # Extract validation error details
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error.get("loc", [])),
            "message": error.get("msg", "Validation error"),
            "type": error.get("type", "unknown"),
        })

    logger.warning(
        f"Validation error: {len(errors)} field(s) invalid",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
            "errors": errors,
        },
    )

    response_body = create_error_response(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        details={"validation_errors": errors},
        correlation_id=correlation_id,
        request_path=request.url.path,
    )

    headers = {}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return JSONResponse(
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY.value,
        content=response_body,
        headers=headers,
    )


async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    correlation_id = getattr(request.state, "correlation_id", None)

    # Log full stack trace for unexpected errors
    logger.exception(
        f"Unhandled exception: {type(exc).__name__} - {str(exc)}",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
            "exception_type": type(exc).__name__,
        },
    )

    # Don't expose internal error details in production
    response_body = create_error_response(
        error_code=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred",
        details={"exception_type": type(exc).__name__},
        correlation_id=correlation_id,
        request_path=request.url.path,
    )

    headers = {}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        content=response_body,
        headers=headers,
    )


async def timeout_exception_handler(
    request: Request, exc: TimeoutError
) -> JSONResponse:
    """Handle timeout errors."""
    correlation_id = getattr(request.state, "correlation_id", None)

    logger.warning(
        f"Request timeout: {request.url.path}",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
        },
    )

    response_body = create_error_response(
        error_code=ErrorCode.TIMEOUT,
        message="Request timed out",
        correlation_id=correlation_id,
        request_path=request.url.path,
    )

    headers = {
        "Retry-After": "60",
    }
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return JSONResponse(
        status_code=HTTPStatus.GATEWAY_TIMEOUT.value,
        content=response_body,
        headers=headers,
    )


async def connection_exception_handler(
    request: Request, exc: ConnectionError
) -> JSONResponse:
    """Handle connection errors."""
    correlation_id = getattr(request.state, "correlation_id", None)

    logger.error(
        f"Connection error: {str(exc)}",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
        },
    )

    response_body = create_error_response(
        error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
        message="Service connection failed",
        correlation_id=correlation_id,
        request_path=request.url.path,
    )

    headers = {
        "Retry-After": "30",
    }
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return JSONResponse(
        status_code=HTTPStatus.BAD_GATEWAY.value,
        content=response_body,
        headers=headers,
    )


async def circuit_breaker_exception_handler(
    request: Request, exc: CircuitOpenError
) -> JSONResponse:
    """Handle circuit breaker open errors."""
    correlation_id = getattr(request.state, "correlation_id", None)

    logger.warning(
        f"Circuit breaker open: {exc.circuit_name}",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
            "circuit_name": exc.circuit_name,
            "retry_in_seconds": exc.time_until_retry,
        },
    )

    response_body = create_error_response(
        error_code=ErrorCode.CIRCUIT_BREAKER_OPEN,
        message=f"Service temporarily unavailable (circuit breaker: {exc.circuit_name})",
        details={
            "circuit_name": exc.circuit_name,
            "retry_after_seconds": int(exc.time_until_retry),
        },
        correlation_id=correlation_id,
        request_path=request.url.path,
    )

    headers = {
        "Retry-After": str(int(exc.time_until_retry)),
    }
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return JSONResponse(
        status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        content=response_body,
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI application."""

    # Register QbitelAIException handler (and all subclasses)
    app.add_exception_handler(QbitelAIException, qbitel_exception_handler)

    # Register HTTPException handler
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # Register validation error handler
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Register circuit breaker handler
    app.add_exception_handler(CircuitOpenError, circuit_breaker_exception_handler)

    # Register timeout handler
    app.add_exception_handler(TimeoutError, timeout_exception_handler)

    # Register connection error handler
    app.add_exception_handler(ConnectionError, connection_exception_handler)

    # Register generic exception handler (catch-all)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Exception handlers registered successfully")
