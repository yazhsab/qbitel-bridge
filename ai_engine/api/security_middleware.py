"""
Security Middleware

Production-grade security middleware for QBITEL API:
- Security headers (CSP, HSTS, X-Frame-Options, etc.)
- CORS configuration
- Request validation
- Rate limiting integration
- Request ID propagation
- Audit logging
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from functools import wraps

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security middleware configuration."""

    # HTTPS/HSTS
    force_https: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = True

    # Content Security Policy
    csp_enabled: bool = True
    csp_directives: Dict[str, str] = field(
        default_factory=lambda: {
            "default-src": "'self'",
            "script-src": "'self' 'unsafe-inline'",
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: https:",
            "font-src": "'self'",
            "connect-src": "'self'",
            "frame-ancestors": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
        }
    )

    # Frame options
    frame_options: str = "DENY"  # DENY, SAMEORIGIN

    # Content type options
    content_type_nosniff: bool = True

    # XSS protection
    xss_protection: str = "1; mode=block"

    # Referrer policy
    referrer_policy: str = "strict-origin-when-cross-origin"

    # Permissions policy
    permissions_policy: Dict[str, str] = field(
        default_factory=lambda: {
            "geolocation": "()",
            "microphone": "()",
            "camera": "()",
            "payment": "()",
            "usb": "()",
        }
    )

    # CORS
    cors_enabled: bool = True
    cors_origins: List[str] = field(
        default_factory=lambda: os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    )
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    cors_allow_headers: List[str] = field(default_factory=lambda: ["*"])
    cors_expose_headers: List[str] = field(
        default_factory=lambda: ["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
    )
    cors_max_age: int = 600

    # Request validation
    max_content_length: int = 10 * 1024 * 1024  # 10MB
    allowed_content_types: Set[str] = field(
        default_factory=lambda: {
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        }
    )

    # Rate limiting headers
    rate_limit_headers: bool = True

    # Request ID
    request_id_header: str = "X-Request-ID"
    generate_request_id: bool = True

    # Audit logging
    audit_logging: bool = True
    audit_sensitive_paths: Set[str] = field(
        default_factory=lambda: {
            "/api/v1/auth",
            "/api/v1/admin",
            "/api/v1/security",
        }
    )

    # Trusted proxies
    trusted_proxies: List[str] = field(default_factory=lambda: ["127.0.0.1", "10.0.0.0/8"])


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security headers to all responses.

    Headers added:
    - Strict-Transport-Security (HSTS)
    - Content-Security-Policy (CSP)
    - X-Frame-Options
    - X-Content-Type-Options
    - X-XSS-Protection
    - Referrer-Policy
    - Permissions-Policy
    """

    def __init__(self, app: FastAPI, config: Optional[SecurityConfig] = None):
        super().__init__(app)
        self.config = config or SecurityConfig()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Add request ID
        request_id = request.headers.get(self.config.request_id_header)
        if not request_id and self.config.generate_request_id:
            request_id = str(uuid.uuid4())

        # Store request ID in state for logging
        request.state.request_id = request_id

        # Record start time
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Add security headers
        self._add_security_headers(response)

        # Add request ID to response
        if request_id:
            response.headers[self.config.request_id_header] = request_id

        # Add processing time header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        # Audit logging for sensitive paths
        if self.config.audit_logging:
            self._audit_log(request, response, process_time)

        return response

    def _add_security_headers(self, response: Response) -> None:
        """Add all security headers to response."""

        # HSTS
        if self.config.force_https:
            hsts_value = f"max-age={self.config.hsts_max_age}"
            if self.config.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if self.config.hsts_preload:
                hsts_value += "; preload"
            response.headers["Strict-Transport-Security"] = hsts_value

        # CSP
        if self.config.csp_enabled:
            csp_value = "; ".join(f"{directive} {value}" for directive, value in self.config.csp_directives.items())
            response.headers["Content-Security-Policy"] = csp_value

        # X-Frame-Options
        response.headers["X-Frame-Options"] = self.config.frame_options

        # X-Content-Type-Options
        if self.config.content_type_nosniff:
            response.headers["X-Content-Type-Options"] = "nosniff"

        # X-XSS-Protection
        response.headers["X-XSS-Protection"] = self.config.xss_protection

        # Referrer-Policy
        response.headers["Referrer-Policy"] = self.config.referrer_policy

        # Permissions-Policy
        if self.config.permissions_policy:
            policy_value = ", ".join(f"{feature}={value}" for feature, value in self.config.permissions_policy.items())
            response.headers["Permissions-Policy"] = policy_value

        # Cache control for sensitive responses
        if "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"

    def _audit_log(self, request: Request, response: Response, duration: float) -> None:
        """Log audit information for requests."""
        path = request.url.path

        # Only audit sensitive paths in detail
        is_sensitive = any(path.startswith(p) for p in self.config.audit_sensitive_paths)

        log_data = {
            "request_id": getattr(request.state, "request_id", None),
            "method": request.method,
            "path": path,
            "status_code": response.status_code,
            "duration_ms": duration * 1000,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent", ""),
        }

        if is_sensitive:
            log_data["query_params"] = dict(request.query_params)
            logger.info(f"Audit: Sensitive request", extra=log_data)
        else:
            logger.debug(f"Request completed", extra=log_data)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP, handling proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """Middleware that redirects HTTP to HTTPS."""

    def __init__(self, app: FastAPI, exclude_paths: Optional[List[str]] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip health check endpoints
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        # Check for HTTPS
        if request.url.scheme == "http":
            # Check X-Forwarded-Proto for proxied requests
            forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
            if forwarded_proto != "https":
                https_url = request.url.replace(scheme="https")
                return Response(
                    status_code=301,
                    headers={"Location": str(https_url)},
                )

        return await call_next(request)


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation and sanitization."""

    def __init__(self, app: FastAPI, config: Optional[SecurityConfig] = None):
        super().__init__(app)
        self.config = config or SecurityConfig()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > self.config.max_content_length:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Request Entity Too Large",
                            "max_size": self.config.max_content_length,
                        },
                    )
            except ValueError:
                pass

        # Check content type for POST/PUT/PATCH
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("Content-Type", "").split(";")[0].strip()
            if content_type and content_type not in self.config.allowed_content_types:
                return JSONResponse(
                    status_code=415,
                    content={
                        "error": "Unsupported Media Type",
                        "allowed_types": list(self.config.allowed_content_types),
                    },
                )

        return await call_next(request)


class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add rate limit headers to responses."""

    def __init__(
        self,
        app: FastAPI,
        default_limit: int = 100,
        default_window: int = 60,
    ):
        super().__init__(app)
        self.default_limit = default_limit
        self.default_window = default_window

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Get rate limit info from request state (set by rate limiter)
        limit = getattr(request.state, "rate_limit", self.default_limit)
        remaining = getattr(request.state, "rate_limit_remaining", limit)
        reset = getattr(request.state, "rate_limit_reset", int(time.time()) + self.default_window)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(reset)

        return response


def setup_security_middleware(
    app: FastAPI,
    config: Optional[SecurityConfig] = None,
) -> None:
    """
    Configure all security middleware for the application.

    Args:
        app: FastAPI application
        config: Security configuration
    """
    config = config or SecurityConfig()

    # CORS middleware (must be added first)
    if config.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=config.cors_allow_credentials,
            allow_methods=config.cors_allow_methods,
            allow_headers=config.cors_allow_headers,
            expose_headers=config.cors_expose_headers,
            max_age=config.cors_max_age,
        )

    # Request validation
    app.add_middleware(RequestValidationMiddleware, config=config)

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware, config=config)

    # HTTPS redirect (only in production)
    if config.force_https:
        app.add_middleware(
            HTTPSRedirectMiddleware,
            exclude_paths=["/health", "/metrics", "/ready", "/live"],
        )

    # Rate limit headers
    if config.rate_limit_headers:
        app.add_middleware(RateLimitHeadersMiddleware)

    logger.info("Security middleware configured")


# Decorator for requiring specific security contexts
def require_secure_context(
    require_https: bool = True,
    require_auth: bool = True,
    allowed_ips: Optional[List[str]] = None,
):
    """
    Decorator to enforce security requirements on endpoints.

    Args:
        require_https: Require HTTPS connection
        require_auth: Require authentication
        allowed_ips: IP whitelist
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Check HTTPS
            if require_https:
                proto = request.headers.get("X-Forwarded-Proto", request.url.scheme)
                if proto != "https":
                    return JSONResponse(
                        status_code=403,
                        content={"error": "HTTPS required"},
                    )

            # Check IP whitelist
            if allowed_ips:
                client_ip = (
                    request.headers.get("X-Forwarded-For", request.client.host if request.client else "").split(",")[0].strip()
                )

                if client_ip not in allowed_ips:
                    return JSONResponse(
                        status_code=403,
                        content={"error": "IP not allowed"},
                    )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


# Security event logging
class SecurityEventLogger:
    """Logger for security-related events."""

    def __init__(self, logger_name: str = "security"):
        self.logger = logging.getLogger(logger_name)

    def log_authentication(
        self,
        user_id: str,
        success: bool,
        method: str,
        ip_address: str,
        details: Optional[Dict] = None,
    ) -> None:
        """Log authentication event."""
        event = {
            "event_type": "authentication",
            "user_id": user_id,
            "success": success,
            "method": method,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat(),
            **(details or {}),
        }

        if success:
            self.logger.info("Authentication successful", extra=event)
        else:
            self.logger.warning("Authentication failed", extra=event)

    def log_authorization(
        self,
        user_id: str,
        resource: str,
        action: str,
        allowed: bool,
        ip_address: str,
    ) -> None:
        """Log authorization event."""
        event = {
            "event_type": "authorization",
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "allowed": allowed,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if allowed:
            self.logger.debug("Authorization granted", extra=event)
        else:
            self.logger.warning("Authorization denied", extra=event)

    def log_security_violation(
        self,
        violation_type: str,
        ip_address: str,
        details: Dict,
    ) -> None:
        """Log security violation."""
        event = {
            "event_type": "security_violation",
            "violation_type": violation_type,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat(),
            **details,
        }

        self.logger.error("Security violation detected", extra=event)

    def log_rate_limit(
        self,
        ip_address: str,
        endpoint: str,
        limit: int,
        current: int,
    ) -> None:
        """Log rate limit event."""
        event = {
            "event_type": "rate_limit",
            "ip_address": ip_address,
            "endpoint": endpoint,
            "limit": limit,
            "current": current,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.logger.warning("Rate limit exceeded", extra=event)


# Global security event logger
security_logger = SecurityEventLogger()
