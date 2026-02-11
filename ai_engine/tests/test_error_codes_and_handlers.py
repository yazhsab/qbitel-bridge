"""
QBITEL Engine - Error Codes and Exception Handlers Tests

Unit tests for numeric error codes, HTTP status mapping, and FastAPI exception handlers.
"""

import pytest
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ai_engine.core.error_codes import (
    ErrorCode,
    get_http_status_for_error,
    get_error_code_info,
    is_client_error,
    is_server_error,
    is_retryable,
)
from ai_engine.core.exceptions import (
    QbitelAIException,
    DiscoveryException,
    ProtocolException,
    FieldDetectionException,
    LLMException,
    ValidationException,
    SecurityException,
)
from ai_engine.api.exception_handlers import (
    get_error_code_for_exception,
    create_error_response,
    register_exception_handlers,
)
from ai_engine.core.constants import (
    MAX_PAYLOAD_SIZE_DEFAULT,
    MAX_PROTOCOL_MESSAGE_SIZE,
    RATE_LIMIT_DEFAULT,
    REQUEST_TIMEOUT_DEFAULT,
)


class TestErrorCodes:
    """Tests for error code definitions and mappings."""

    def test_error_code_ranges_discovery(self):
        """Test discovery error codes are in 1000-1999 range."""
        discovery_codes = [
            ErrorCode.DISCOVERY_FAILED,
            ErrorCode.DISCOVERY_TIMEOUT,
            ErrorCode.INVALID_PROTOCOL_DATA,
            ErrorCode.GRAMMAR_LEARNING_FAILED,
            ErrorCode.PARSER_GENERATION_FAILED,
        ]
        for code in discovery_codes:
            assert 1000 <= code < 2000, f"{code.name} should be in 1000-1999 range"

    def test_error_code_ranges_detection(self):
        """Test detection error codes are in 2000-2999 range."""
        detection_codes = [
            ErrorCode.FIELD_DETECTION_FAILED,
            ErrorCode.MODEL_NOT_LOADED,
            ErrorCode.INVALID_MESSAGE_FORMAT,
            ErrorCode.ANOMALY_DETECTION_FAILED,
        ]
        for code in detection_codes:
            assert 2000 <= code < 3000, f"{code.name} should be in 2000-2999 range"

    def test_error_code_ranges_llm(self):
        """Test LLM error codes are in 3000-3999 range."""
        llm_codes = [
            ErrorCode.LLM_UNAVAILABLE,
            ErrorCode.LLM_TIMEOUT,
            ErrorCode.LLM_RATE_LIMITED,
            ErrorCode.LLM_CONTEXT_TOO_LONG,
        ]
        for code in llm_codes:
            assert 3000 <= code < 4000, f"{code.name} should be in 3000-3999 range"

    def test_error_code_ranges_auth(self):
        """Test auth error codes are in 4000-4999 range."""
        auth_codes = [
            ErrorCode.UNAUTHORIZED,
            ErrorCode.INVALID_TOKEN,
            ErrorCode.TOKEN_EXPIRED,
            ErrorCode.INSUFFICIENT_PERMISSIONS,
        ]
        for code in auth_codes:
            assert 4000 <= code < 5000, f"{code.name} should be in 4000-4999 range"

    def test_error_code_ranges_validation(self):
        """Test validation error codes are in 5000-5999 range."""
        validation_codes = [
            ErrorCode.VALIDATION_ERROR,
            ErrorCode.INVALID_INPUT,
            ErrorCode.PAYLOAD_TOO_LARGE,
        ]
        for code in validation_codes:
            assert 5000 <= code < 6000, f"{code.name} should be in 5000-5999 range"

    def test_error_code_ranges_system(self):
        """Test system error codes are in 9000-9999 range."""
        system_codes = [
            ErrorCode.INTERNAL_ERROR,
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.TIMEOUT,
        ]
        for code in system_codes:
            assert 9000 <= code < 10000, f"{code.name} should be in 9000-9999 range"


class TestHttpStatusMapping:
    """Tests for error code to HTTP status mapping."""

    def test_auth_errors_map_to_401_403(self):
        """Test authentication errors map to 401/403."""
        assert get_http_status_for_error(ErrorCode.UNAUTHORIZED) == HTTPStatus.UNAUTHORIZED
        assert get_http_status_for_error(ErrorCode.INVALID_TOKEN) == HTTPStatus.UNAUTHORIZED
        assert get_http_status_for_error(ErrorCode.TOKEN_EXPIRED) == HTTPStatus.UNAUTHORIZED
        assert get_http_status_for_error(ErrorCode.INSUFFICIENT_PERMISSIONS) == HTTPStatus.FORBIDDEN

    def test_validation_errors_map_correctly(self):
        """Test validation errors map to appropriate 4xx codes."""
        assert get_http_status_for_error(ErrorCode.VALIDATION_ERROR) == HTTPStatus.BAD_REQUEST
        assert get_http_status_for_error(ErrorCode.INVALID_INPUT) == HTTPStatus.BAD_REQUEST
        assert get_http_status_for_error(ErrorCode.PAYLOAD_TOO_LARGE) == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        assert get_http_status_for_error(ErrorCode.UNSUPPORTED_CONTENT_TYPE) == HTTPStatus.UNSUPPORTED_MEDIA_TYPE

    def test_server_errors_map_to_5xx(self):
        """Test server errors map to 5xx status codes."""
        assert get_http_status_for_error(ErrorCode.INTERNAL_ERROR) == HTTPStatus.INTERNAL_SERVER_ERROR
        assert get_http_status_for_error(ErrorCode.SERVICE_UNAVAILABLE) == HTTPStatus.SERVICE_UNAVAILABLE
        assert get_http_status_for_error(ErrorCode.TIMEOUT) == HTTPStatus.GATEWAY_TIMEOUT

    def test_llm_errors_map_correctly(self):
        """Test LLM errors map to appropriate status codes."""
        assert get_http_status_for_error(ErrorCode.LLM_UNAVAILABLE) == HTTPStatus.SERVICE_UNAVAILABLE
        assert get_http_status_for_error(ErrorCode.LLM_RATE_LIMITED) == HTTPStatus.TOO_MANY_REQUESTS
        assert get_http_status_for_error(ErrorCode.LLM_TIMEOUT) == HTTPStatus.GATEWAY_TIMEOUT
        assert get_http_status_for_error(ErrorCode.LLM_PROVIDER_ERROR) == HTTPStatus.BAD_GATEWAY


class TestErrorCodeInfo:
    """Tests for error code info retrieval."""

    def test_get_error_code_info_structure(self):
        """Test error code info has correct structure."""
        info = get_error_code_info(ErrorCode.DISCOVERY_FAILED)

        assert "code" in info
        assert "name" in info
        assert "http_status" in info
        assert "http_status_phrase" in info

    def test_get_error_code_info_values(self):
        """Test error code info returns correct values."""
        info = get_error_code_info(ErrorCode.DISCOVERY_FAILED)

        assert info["code"] == 1000
        assert info["name"] == "DISCOVERY_FAILED"
        assert info["http_status"] == 500


class TestErrorCategorization:
    """Tests for error categorization helpers."""

    def test_is_client_error_true(self):
        """Test is_client_error returns True for 4xx errors."""
        client_errors = [
            ErrorCode.VALIDATION_ERROR,
            ErrorCode.UNAUTHORIZED,
            ErrorCode.INVALID_INPUT,
            ErrorCode.PAYLOAD_TOO_LARGE,
            ErrorCode.INSUFFICIENT_PERMISSIONS,
        ]
        for code in client_errors:
            assert is_client_error(code) is True, f"{code.name} should be client error"

    def test_is_client_error_false(self):
        """Test is_client_error returns False for 5xx errors."""
        server_errors = [
            ErrorCode.INTERNAL_ERROR,
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.DISCOVERY_FAILED,
        ]
        for code in server_errors:
            assert is_client_error(code) is False, f"{code.name} should not be client error"

    def test_is_server_error_true(self):
        """Test is_server_error returns True for 5xx errors."""
        server_errors = [
            ErrorCode.INTERNAL_ERROR,
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.DISCOVERY_FAILED,
            ErrorCode.DATABASE_ERROR,
        ]
        for code in server_errors:
            assert is_server_error(code) is True, f"{code.name} should be server error"

    def test_is_server_error_false(self):
        """Test is_server_error returns False for 4xx errors."""
        client_errors = [
            ErrorCode.VALIDATION_ERROR,
            ErrorCode.UNAUTHORIZED,
        ]
        for code in client_errors:
            assert is_server_error(code) is False, f"{code.name} should not be server error"

    def test_is_retryable_true(self):
        """Test is_retryable returns True for transient errors."""
        retryable = [
            ErrorCode.LLM_RATE_LIMITED,
            ErrorCode.LLM_TIMEOUT,
            ErrorCode.TIMEOUT,
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.CIRCUIT_BREAKER_OPEN,
            ErrorCode.NETWORK_ERROR,
        ]
        for code in retryable:
            assert is_retryable(code) is True, f"{code.name} should be retryable"

    def test_is_retryable_false(self):
        """Test is_retryable returns False for permanent errors."""
        non_retryable = [
            ErrorCode.VALIDATION_ERROR,
            ErrorCode.UNAUTHORIZED,
            ErrorCode.INTERNAL_ERROR,
            ErrorCode.INVALID_INPUT,
        ]
        for code in non_retryable:
            assert is_retryable(code) is False, f"{code.name} should not be retryable"


class TestExceptionToErrorCodeMapping:
    """Tests for mapping exceptions to error codes."""

    def test_discovery_exception_mapping(self):
        """Test DiscoveryException maps to DISCOVERY_FAILED."""
        exc = DiscoveryException("Test error")
        code = get_error_code_for_exception(exc)
        assert code == ErrorCode.DISCOVERY_FAILED

    def test_protocol_exception_mapping(self):
        """Test ProtocolException maps to PROTOCOL_ERROR."""
        exc = ProtocolException("Test error")
        code = get_error_code_for_exception(exc)
        assert code == ErrorCode.PROTOCOL_ERROR

    def test_field_detection_exception_mapping(self):
        """Test FieldDetectionException maps to FIELD_DETECTION_FAILED."""
        exc = FieldDetectionException("Test error")
        code = get_error_code_for_exception(exc)
        assert code == ErrorCode.FIELD_DETECTION_FAILED

    def test_llm_exception_mapping(self):
        """Test LLMException maps to LLM_PROVIDER_ERROR."""
        exc = LLMException("Test error")
        code = get_error_code_for_exception(exc)
        assert code == ErrorCode.LLM_PROVIDER_ERROR

    def test_unknown_exception_mapping(self):
        """Test unknown exceptions map to INTERNAL_ERROR."""
        exc = RuntimeError("Unknown error")
        code = get_error_code_for_exception(exc)
        assert code == ErrorCode.INTERNAL_ERROR


class TestErrorResponseCreation:
    """Tests for error response creation."""

    def test_basic_error_response_structure(self):
        """Test basic error response has correct structure."""
        response = create_error_response(
            error_code=ErrorCode.VALIDATION_ERROR,
            message="Invalid input",
        )

        assert "error" in response
        error = response["error"]
        assert error["code"] == 5000
        assert error["name"] == "VALIDATION_ERROR"
        assert error["message"] == "Invalid input"
        assert error["http_status"] == 400
        assert "timestamp" in error

    def test_error_response_with_details(self):
        """Test error response includes details when provided."""
        response = create_error_response(
            error_code=ErrorCode.VALIDATION_ERROR,
            message="Validation failed",
            details={"field": "packet_data", "reason": "exceeds size limit"},
        )

        assert response["error"]["details"]["field"] == "packet_data"
        assert response["error"]["details"]["reason"] == "exceeds size limit"

    def test_error_response_with_correlation_id(self):
        """Test error response includes correlation ID."""
        response = create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Something went wrong",
            correlation_id="corr-123-abc",
        )

        assert response["error"]["correlation_id"] == "corr-123-abc"

    def test_error_response_with_request_path(self):
        """Test error response includes request path."""
        response = create_error_response(
            error_code=ErrorCode.DISCOVERY_FAILED,
            message="Discovery failed",
            request_path="/api/v1/discover",
        )

        assert response["error"]["path"] == "/api/v1/discover"

    def test_retryable_error_includes_retry_info(self):
        """Test retryable errors include retry information."""
        response = create_error_response(
            error_code=ErrorCode.LLM_RATE_LIMITED,
            message="Rate limited",
        )

        assert response["error"]["retryable"] is True
        assert response["error"]["retry_after"] == 60


class TestConstants:
    """Tests for constants values."""

    def test_size_limits_positive(self):
        """Test size limits are positive values."""
        assert MAX_PAYLOAD_SIZE_DEFAULT > 0
        assert MAX_PROTOCOL_MESSAGE_SIZE > 0

    def test_size_limits_reasonable(self):
        """Test size limits are within reasonable bounds."""
        # At least 1MB
        assert MAX_PAYLOAD_SIZE_DEFAULT >= 1024 * 1024
        # No more than 100MB
        assert MAX_PAYLOAD_SIZE_DEFAULT <= 100 * 1024 * 1024

    def test_rate_limits_positive(self):
        """Test rate limits are positive."""
        assert RATE_LIMIT_DEFAULT > 0

    def test_timeouts_positive(self):
        """Test timeouts are positive."""
        assert REQUEST_TIMEOUT_DEFAULT > 0


class TestFastAPIExceptionHandlerIntegration:
    """Integration tests for FastAPI exception handlers."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI application."""
        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test/qbitel-error")
        async def trigger_qbitel_error():
            raise DiscoveryException("Protocol discovery failed", request_id="req-123")

        @app.get("/test/http-error")
        async def trigger_http_error():
            raise HTTPException(status_code=404, detail="Resource not found")

        @app.get("/test/validation-error")
        async def trigger_validation_error():
            raise ValidationException("Field validation failed", field="test_field")

        @app.get("/test/llm-error")
        async def trigger_llm_error():
            raise LLMException("LLM provider unavailable", provider="ollama")

        @app.get("/test/generic-error")
        async def trigger_generic_error():
            raise RuntimeError("Unexpected runtime error")

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app, raise_server_exceptions=False)

    def test_qbitel_exception_response_format(self, client):
        """Test QbitelAIException produces standardized response."""
        response = client.get("/test/qbitel-error")

        assert response.status_code == 500
        data = response.json()

        assert "error" in data
        assert data["error"]["code"] == 1000  # DISCOVERY_FAILED
        assert data["error"]["name"] == "DISCOVERY_FAILED"
        assert "discovery failed" in data["error"]["message"].lower()

    def test_http_exception_response_format(self, client):
        """Test HTTPException produces standardized response."""
        response = client.get("/test/http-error")

        assert response.status_code == 404
        data = response.json()

        assert "error" in data
        assert data["error"]["message"] == "Resource not found"

    def test_llm_exception_response_format(self, client):
        """Test LLMException produces standardized response."""
        response = client.get("/test/llm-error")

        # LLMException maps to LLM_PROVIDER_ERROR which is 502
        data = response.json()

        assert "error" in data
        assert "llm" in data["error"]["message"].lower() or "unavailable" in data["error"]["message"].lower()

    def test_generic_exception_response_format(self, client):
        """Test generic exceptions produce safe 500 response."""
        response = client.get("/test/generic-error")

        assert response.status_code == 500
        data = response.json()

        assert "error" in data
        assert data["error"]["code"] == 9000  # INTERNAL_ERROR
        # Should not expose internal error details
        assert "runtime" not in data["error"]["message"].lower()


class TestQbitelAIExceptionClass:
    """Tests for QbitelAIException class behavior."""

    def test_exception_basic_creation(self):
        """Test basic exception creation."""
        exc = QbitelAIException("Test error message")

        assert exc.message == "Test error message"
        assert exc.error_code == "QBITEL_AI_ERROR"
        assert exc.context == {}

    def test_exception_with_error_code(self):
        """Test exception with custom error code."""
        exc = QbitelAIException("Test error", error_code="CUSTOM_CODE")

        assert exc.error_code == "CUSTOM_CODE"

    def test_exception_with_context(self):
        """Test exception with context data."""
        exc = QbitelAIException(
            "Test error",
            context={"request_id": "123", "user_id": "user-456"}
        )

        assert exc.context["request_id"] == "123"
        assert exc.context["user_id"] == "user-456"

    def test_exception_str_representation(self):
        """Test exception string representation."""
        exc = QbitelAIException("Test error", error_code="TEST_CODE")
        str_repr = str(exc)

        assert "Test error" in str_repr
        assert "TEST_CODE" in str_repr

    def test_discovery_exception_specialization(self):
        """Test DiscoveryException has correct error code."""
        exc = DiscoveryException("Discovery failed", request_id="req-123")

        assert exc.error_code == "DISCOVERY_ERROR"
        assert exc.context["request_id"] == "req-123"

    def test_protocol_exception_specialization(self):
        """Test ProtocolException includes protocol info."""
        exc = ProtocolException(
            "Protocol error",
            protocol_type="ISO-8583",
            packet_length=1024
        )

        assert exc.error_code == "PROTOCOL_ERROR"
        assert exc.context["protocol_type"] == "ISO-8583"
        assert exc.context["packet_length"] == 1024

    def test_llm_exception_specialization(self):
        """Test LLMException includes provider info."""
        exc = LLMException("Provider error", provider="ollama")

        assert exc.error_code == "LLM_ERROR"
        assert exc.context["provider"] == "ollama"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
