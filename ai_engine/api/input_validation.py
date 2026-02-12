"""
QBITEL Engine - Input Validation Framework

Production-grade input validation with payload size limits, sanitization,
and protection against common attacks.
"""

import re
import logging
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
VALIDATION_ERRORS = Counter(
    "input_validation_errors_total",
    "Total input validation errors",
    ["error_type", "endpoint"],
)
PAYLOAD_SIZE = Histogram(
    "request_payload_size_bytes",
    "Request payload size in bytes",
    ["method", "endpoint"],
    buckets=[100, 1000, 10_000, 100_000, 1_000_000, 10_000_000],
)


class ValidationError(Exception):
    """Input validation error."""

    pass


class InputValidator:
    """
    Validates and sanitizes input data.

    Features:
    - Payload size limits
    - SQL injection prevention
    - XSS prevention
    - Command injection prevention
    - Path traversal prevention
    - Content-type validation
    """

    # Maximum payload sizes (bytes)
    MAX_PAYLOAD_SIZES = {
        "default": 10 * 1024 * 1024,  # 10MB
        "json": 5 * 1024 * 1024,  # 5MB
        "form": 1 * 1024 * 1024,  # 1MB
        "file": 50 * 1024 * 1024,  # 50MB
    }

    # Dangerous patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bunion\b.*\bselect\b)",
        r"(\bor\b\s+1\s*=\s*1)",
        r"(\band\b\s+1\s*=\s*1)",
        r"(';.*--)",
        r"(\bdrop\b.*\btable\b)",
        r"(\bexec\b.*\()",
        r"(\binsert\b.*\binto\b)",
        r"(\bupdate\b.*\bset\b)",
        r"(\bdelete\b.*\bfrom\b)",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # onclick, onload, etc.
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$()]",
        r"\$\([^)]*\)",  # Command substitution
        r"`[^`]*`",  # Backticks
    ]

    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.",
        r"%2e%2e",
        r"~root",
        r"/etc/",
        r"/var/",
    ]

    def __init__(self):
        """Initialize input validator."""
        # Compile regex patterns for performance
        self.sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS]
        self.xss_patterns = [re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS]
        self.cmd_patterns = [re.compile(p) for p in self.COMMAND_INJECTION_PATTERNS]
        self.path_patterns = [re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL_PATTERNS]

    def validate_payload_size(
        self,
        content: bytes,
        content_type: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> bool:
        """
        Validate payload size.

        Args:
            content: Request content
            content_type: Content-Type header
            max_size: Override max size

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If payload too large
        """
        size = len(content)

        # Determine max size based on content type
        if max_size:
            limit = max_size
        elif content_type:
            if "json" in content_type.lower():
                limit = self.MAX_PAYLOAD_SIZES["json"]
            elif "form" in content_type.lower():
                limit = self.MAX_PAYLOAD_SIZES["form"]
            elif "multipart" in content_type.lower():
                limit = self.MAX_PAYLOAD_SIZES["file"]
            else:
                limit = self.MAX_PAYLOAD_SIZES["default"]
        else:
            limit = self.MAX_PAYLOAD_SIZES["default"]

        if size > limit:
            raise ValidationError(f"Payload too large: {size} bytes (max: {limit} bytes)")

        return True

    def check_sql_injection(self, value: str) -> bool:
        """
        Check for SQL injection patterns.

        Args:
            value: String to check

        Returns:
            bool: True if suspicious patterns found
        """
        for pattern in self.sql_patterns:
            if pattern.search(value):
                logger.warning(f"Potential SQL injection detected: {pattern.pattern}")
                return True
        return False

    def check_xss(self, value: str) -> bool:
        """
        Check for XSS patterns.

        Args:
            value: String to check

        Returns:
            bool: True if suspicious patterns found
        """
        for pattern in self.xss_patterns:
            if pattern.search(value):
                logger.warning(f"Potential XSS detected: {pattern.pattern}")
                return True
        return False

    def check_command_injection(self, value: str) -> bool:
        """
        Check for command injection patterns.

        Args:
            value: String to check

        Returns:
            bool: True if suspicious patterns found
        """
        for pattern in self.cmd_patterns:
            if pattern.search(value):
                logger.warning(f"Potential command injection detected: {pattern.pattern}")
                return True
        return False

    def check_path_traversal(self, value: str) -> bool:
        """
        Check for path traversal patterns.

        Args:
            value: String to check

        Returns:
            bool: True if suspicious patterns found
        """
        for pattern in self.path_patterns:
            if pattern.search(value):
                logger.warning(f"Potential path traversal detected: {pattern.pattern}")
                return True
        return False

    def sanitize_string(self, value: str, max_length: Optional[int] = None, allow_html: bool = False) -> str:
        """
        Sanitize string input.

        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML tags

        Returns:
            str: Sanitized string

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value)}")

        # Check length
        if max_length and len(value) > max_length:
            raise ValidationError(f"String too long: {len(value)} chars (max: {max_length})")

        # Check for SQL injection
        if self.check_sql_injection(value):
            VALIDATION_ERRORS.labels(error_type="sql_injection", endpoint="unknown").inc()
            raise ValidationError("Potential SQL injection detected")

        # Check for XSS (unless HTML is explicitly allowed)
        if not allow_html and self.check_xss(value):
            VALIDATION_ERRORS.labels(error_type="xss", endpoint="unknown").inc()
            raise ValidationError("Potential XSS detected")

        # Check for command injection
        if self.check_command_injection(value):
            VALIDATION_ERRORS.labels(error_type="command_injection", endpoint="unknown").inc()
            raise ValidationError("Potential command injection detected")

        # Check for path traversal
        if self.check_path_traversal(value):
            VALIDATION_ERRORS.labels(error_type="path_traversal", endpoint="unknown").inc()
            raise ValidationError("Potential path traversal detected")

        # Basic sanitization (strip control characters)
        sanitized = "".join(char for char in value if ord(char) >= 32 or char in "\n\r\t")

        return sanitized

    def validate_json(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate JSON against schema.

        Args:
            data: JSON data
            schema: Validation schema

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        for field, constraints in schema.items():
            if constraints.get("required", False) and field not in data:
                raise ValidationError(f"Required field missing: {field}")

            if field in data:
                value = data[field]
                field_type = constraints.get("type")

                # Type validation
                if field_type and not isinstance(value, field_type):
                    raise ValidationError(
                        f"Invalid type for {field}: expected {field_type.__name__}, " f"got {type(value).__name__}"
                    )

                # String validation
                if isinstance(value, str):
                    max_length = constraints.get("max_length")
                    self.sanitize_string(value, max_length=max_length)

                # Numeric validation
                if isinstance(value, (int, float)):
                    min_val = constraints.get("min")
                    max_val = constraints.get("max")
                    if min_val is not None and value < min_val:
                        raise ValidationError(f"{field} below minimum: {value} < {min_val}")
                    if max_val is not None and value > max_val:
                        raise ValidationError(f"{field} above maximum: {value} > {max_val}")

                # List validation
                if isinstance(value, list):
                    max_items = constraints.get("max_items")
                    if max_items and len(value) > max_items:
                        raise ValidationError(f"{field} has too many items: {len(value)} > {max_items}")

        return True


class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce payload size limits.

    This prevents DoS attacks via large payloads.
    """

    def __init__(self, app, max_size: int = 10 * 1024 * 1024):
        """
        Initialize middleware.

        Args:
            app: FastAPI application
            max_size: Maximum payload size in bytes (default: 10MB)
        """
        super().__init__(app)
        self.max_size = max_size
        self.validator = InputValidator()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with payload size validation."""
        # Get content length
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                size = int(content_length)

                # Record payload size
                PAYLOAD_SIZE.labels(method=request.method, endpoint=request.url.path).observe(size)

                # Check size
                if size > self.max_size:
                    logger.warning(f"Payload too large: {size} bytes from {request.client.host} " f"to {request.url.path}")
                    VALIDATION_ERRORS.labels(error_type="payload_too_large", endpoint=request.url.path).inc()

                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"Payload too large: {size} bytes (max: {self.max_size} bytes)",
                            "error_code": "PAYLOAD_TOO_LARGE",
                        },
                        headers={"Retry-After": "60"},
                    )

            except ValueError:
                logger.warning(f"Invalid Content-Length header: {content_length}")

        # Process request
        response = await call_next(request)
        return response


class ContentTypeValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate Content-Type header.

    Ensures requests have appropriate Content-Type for their payload.
    """

    ALLOWED_CONTENT_TYPES = {
        "POST": [
            "application/json",
            "multipart/form-data",
            "application/x-www-form-urlencoded",
        ],
        "PUT": ["application/json", "multipart/form-data"],
        "PATCH": ["application/json"],
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with Content-Type validation."""
        # Only validate methods with body
        if request.method in self.ALLOWED_CONTENT_TYPES:
            content_type = request.headers.get("content-type", "").split(";")[0].strip()

            if content_type:
                allowed = self.ALLOWED_CONTENT_TYPES[request.method]

                # Check if content-type is allowed
                if not any(ct in content_type for ct in allowed):
                    logger.warning(f"Invalid Content-Type: {content_type} for {request.method} " f"{request.url.path}")
                    VALIDATION_ERRORS.labels(error_type="invalid_content_type", endpoint=request.url.path).inc()

                    return JSONResponse(
                        status_code=415,
                        content={
                            "detail": f"Unsupported Content-Type: {content_type}",
                            "allowed": allowed,
                            "error_code": "UNSUPPORTED_MEDIA_TYPE",
                        },
                    )

        response = await call_next(request)
        return response


# Global validator instance
_input_validator: Optional[InputValidator] = None


def get_input_validator() -> InputValidator:
    """
    Get global input validator instance.

    Returns:
        InputValidator: Global instance
    """
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator
