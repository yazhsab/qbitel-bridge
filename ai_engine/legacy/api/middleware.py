"""
CRONOS AI Engine - Legacy System Whisperer API Middleware

Custom middleware for request processing, logging, rate limiting, and security.
"""

import time
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from contextvars import ContextVar
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.status import HTTP_429_TOO_MANY_REQUESTS, HTTP_500_INTERNAL_SERVER_ERROR

from ...core.config import Config, get_config
from ..logging import get_legacy_logger, LogContext, LogCategory
from ..exceptions import LegacySystemWhispererException
from ..monitoring import LegacySystemMonitor
from .auth import User, get_auth, log_security_event


# Request context storage
request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


class RequestContext:
    """Request context data structure."""

    def __init__(self):
        self.request_id: str = str(uuid.uuid4())
        self.start_time: float = time.time()
        self.user: Optional[User] = None
        self.client_ip: Optional[str] = None
        self.user_agent: Optional[str] = None
        self.endpoint: Optional[str] = None
        self.method: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "start_time": self.start_time,
            "user_id": self.user.user_id if self.user else None,
            "username": self.user.username if self.user else None,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "endpoint": self.endpoint,
            "method": self.method,
            "metadata": self.metadata,
        }


def get_request_context() -> RequestContext:
    """Get current request context."""
    ctx_data = request_context.get({})
    if not ctx_data:
        return RequestContext()

    # Reconstruct context from stored data
    ctx = RequestContext()
    ctx.request_id = ctx_data.get("request_id", ctx.request_id)
    ctx.start_time = ctx_data.get("start_time", ctx.start_time)
    ctx.client_ip = ctx_data.get("client_ip")
    ctx.user_agent = ctx_data.get("user_agent")
    ctx.endpoint = ctx_data.get("endpoint")
    ctx.method = ctx_data.get("method")
    ctx.metadata = ctx_data.get("metadata", {})

    return ctx


def set_request_context(ctx: RequestContext) -> None:
    """Set request context."""
    request_context.set(ctx.to_dict())


class LegacySystemMiddleware(BaseHTTPMiddleware):
    """Main middleware for Legacy System Whisperer API."""

    def __init__(
        self, app, config: Config, monitor: Optional[LegacySystemMonitor] = None
    ):
        super().__init__(app)
        self.config = config
        self.monitor = monitor
        self.logger = get_legacy_logger(__name__)

        # Rate limiting storage
        self.rate_limit_cache: Dict[str, List[float]] = {}

        # Request tracking
        self.active_requests: Dict[str, RequestContext] = {}

        self.logger.info("Legacy System Whisperer middleware initialized")

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request through middleware pipeline."""

        # Create request context
        ctx = RequestContext()
        ctx.client_ip = self._get_client_ip(request)
        ctx.user_agent = request.headers.get("user-agent")
        ctx.endpoint = str(request.url.path)
        ctx.method = request.method

        # Set context
        set_request_context(ctx)
        self.active_requests[ctx.request_id] = ctx

        try:
            # Pre-processing
            await self._pre_process(request, ctx)

            # Process request
            response = await call_next(request)

            # Post-processing
            await self._post_process(request, response, ctx)

            return response

        except Exception as e:
            # Error handling
            return await self._handle_error(request, e, ctx)

        finally:
            # Cleanup
            self.active_requests.pop(ctx.request_id, None)

    async def _pre_process(self, request: Request, ctx: RequestContext) -> None:
        """Pre-process request."""

        # Log incoming request
        self.logger.debug(
            f"Incoming request: {ctx.method} {ctx.endpoint}",
            context=LogContext(
                request_id=ctx.request_id,
                operation="request_start",
                component="api_middleware",
                category=LogCategory.USER_ACTION,
                metadata={
                    "method": ctx.method,
                    "endpoint": ctx.endpoint,
                    "client_ip": ctx.client_ip,
                    "user_agent": ctx.user_agent,
                },
            ),
        )

        # Rate limiting check
        if self._should_apply_rate_limiting(request):
            await self._check_rate_limit(request, ctx)

        # Security headers validation
        await self._validate_security_headers(request, ctx)

        # Record request metrics
        if self.monitor:
            # Would record request start metrics
            pass

    async def _post_process(
        self, request: Request, response: Response, ctx: RequestContext
    ) -> None:
        """Post-process response."""

        processing_time = time.time() - ctx.start_time

        # Add response headers
        response.headers["X-Request-ID"] = ctx.request_id
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"
        response.headers["X-API-Version"] = "1.0.0"

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Log response
        self.logger.info(
            f"Request completed: {ctx.method} {ctx.endpoint} - {response.status_code}",
            context=LogContext(
                request_id=ctx.request_id,
                operation="request_complete",
                component="api_middleware",
                category=LogCategory.USER_ACTION,
                duration_ms=processing_time * 1000,
                metadata={
                    "status_code": response.status_code,
                    "processing_time_ms": processing_time * 1000,
                    "user_id": ctx.user.user_id if ctx.user else None,
                },
            ),
        )

        # Record response metrics
        if self.monitor:
            self.monitor.record_operation(
                operation="api_request",
                component="api_middleware",
                status="success" if response.status_code < 400 else "error",
                duration_seconds=processing_time,
            )

    async def _handle_error(
        self, request: Request, error: Exception, ctx: RequestContext
    ) -> Response:
        """Handle request errors."""

        processing_time = time.time() - ctx.start_time

        # Log error
        self.logger.error(
            f"Request failed: {ctx.method} {ctx.endpoint}",
            context=LogContext(
                request_id=ctx.request_id,
                operation="request_error",
                component="api_middleware",
                category=LogCategory.USER_ACTION,
                duration_ms=processing_time * 1000,
                metadata={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "user_id": ctx.user.user_id if ctx.user else None,
                },
            ),
            exception=error,
        )

        # Record error metrics
        if self.monitor:
            self.monitor.record_operation(
                operation="api_request",
                component="api_middleware",
                status="error",
                duration_seconds=processing_time,
            )

        # Security event logging
        if ctx.user:
            log_security_event(
                event_type="api_error",
                user=ctx.user,
                resource=ctx.endpoint,
                action=ctx.method,
                outcome="failure",
                details={
                    "error_type": type(error).__name__,
                    "processing_time_ms": processing_time * 1000,
                },
            )

        # Return appropriate error response
        if isinstance(error, HTTPException):
            return await self._create_error_response(
                status_code=error.status_code,
                message=error.detail,
                request_id=ctx.request_id,
            )
        elif isinstance(error, LegacySystemWhispererException):
            return await self._create_error_response(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                message=str(error),
                request_id=ctx.request_id,
                error_code=error.category.value if hasattr(error, "category") else None,
            )
        else:
            return await self._create_error_response(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                message="Internal server error",
                request_id=ctx.request_id,
            )

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to client host
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    def _should_apply_rate_limiting(self, request: Request) -> bool:
        """Check if rate limiting should be applied to request."""
        # Skip rate limiting for health checks
        if request.url.path.endswith("/health"):
            return False

        # Skip for metrics endpoint
        if request.url.path.endswith("/metrics"):
            return False

        return True

    async def _check_rate_limit(self, request: Request, ctx: RequestContext) -> None:
        """Check request against rate limits."""
        client_id = ctx.client_ip

        # Get rate limit settings from config
        limit = self.config.security.rate_limit_per_minute
        window_seconds = 60

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # Initialize client history if needed
        if client_id not in self.rate_limit_cache:
            self.rate_limit_cache[client_id] = []

        # Clean old requests
        self.rate_limit_cache[client_id] = [
            timestamp
            for timestamp in self.rate_limit_cache[client_id]
            if timestamp > cutoff_time
        ]

        # Check limit
        if len(self.rate_limit_cache[client_id]) >= limit:
            self.logger.warning(
                f"Rate limit exceeded for client {client_id}",
                context=LogContext(
                    request_id=ctx.request_id,
                    category=LogCategory.SECURITY_EVENT,
                    metadata={
                        "client_ip": client_id,
                        "current_requests": len(self.rate_limit_cache[client_id]),
                        "limit": limit,
                    },
                ),
            )

            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests per minute",
            )

        # Add current request
        self.rate_limit_cache[client_id].append(current_time)

    async def _validate_security_headers(
        self, request: Request, ctx: RequestContext
    ) -> None:
        """Validate security-related headers."""

        # Check for required headers in production
        if self.config.environment.value == "production":
            # Require HTTPS in production (would check X-Forwarded-Proto in real deployment)
            pass

        # Validate Content-Type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                self.logger.warning(
                    f"Invalid content-type: {content_type}",
                    context=LogContext(
                        request_id=ctx.request_id, category=LogCategory.SECURITY_EVENT
                    ),
                )

    async def _create_error_response(
        self,
        status_code: int,
        message: str,
        request_id: str,
        error_code: Optional[str] = None,
    ) -> Response:
        """Create standardized error response."""

        from fastapi.responses import JSONResponse

        error_data = {
            "success": False,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
        }

        if error_code:
            error_data["error_code"] = error_code

        return JSONResponse(
            status_code=status_code,
            content=error_data,
            headers={
                "X-Request-ID": request_id,
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
            },
        )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Specialized middleware for request logging."""

    def __init__(self, app):
        super().__init__(app)
        self.logger = get_legacy_logger("request_logger")

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Log request details."""

        ctx = get_request_context()

        # Log request body for POST/PUT (be careful with sensitive data)
        if request.method in ["POST", "PUT", "PATCH"]:
            if request.headers.get("content-type", "").startswith("application/json"):
                # In production, would filter sensitive fields
                try:
                    body = await request.body()
                    if len(body) < 10000:  # Only log small payloads
                        ctx.metadata["request_body_size"] = len(body)
                except Exception:
                    pass

        response = await call_next(request)

        # Log response details
        ctx.metadata["response_size"] = response.headers.get(
            "content-length", "unknown"
        )

        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security-focused middleware."""

    def __init__(self, app, config: Config):
        super().__init__(app)
        self.config = config
        self.logger = get_legacy_logger("security_middleware")

        # Suspicious activity tracking
        self.suspicious_ips: Dict[str, int] = {}
        self.blocked_ips: set = set()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Apply security checks."""

        ctx = get_request_context()

        # Check blocked IPs
        if ctx.client_ip in self.blocked_ips:
            self.logger.warning(
                f"Request from blocked IP: {ctx.client_ip}",
                context=LogContext(
                    request_id=ctx.request_id, category=LogCategory.SECURITY_EVENT
                ),
            )
            raise HTTPException(status_code=403, detail="Access denied")

        # Suspicious activity detection
        await self._check_suspicious_activity(request, ctx)

        # Process request
        response = await call_next(request)

        # Add security headers
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response

    async def _check_suspicious_activity(
        self, request: Request, ctx: RequestContext
    ) -> None:
        """Check for suspicious request patterns."""

        suspicious_patterns = [
            # Common attack patterns
            "../",
            "etc/passwd",
            "cmd.exe",
            "<script>",
            "javascript:",
            "union select",
            "drop table",
            "insert into",
        ]

        # Check URL path
        url_path = str(request.url.path).lower()
        for pattern in suspicious_patterns:
            if pattern in url_path:
                self._flag_suspicious_activity(
                    ctx.client_ip, f"Suspicious URL pattern: {pattern}"
                )
                break

        # Check headers
        for header_name, header_value in request.headers.items():
            header_value_lower = header_value.lower()
            for pattern in suspicious_patterns:
                if pattern in header_value_lower:
                    self._flag_suspicious_activity(
                        ctx.client_ip, f"Suspicious header: {header_name}"
                    )
                    break

    def _flag_suspicious_activity(self, client_ip: str, reason: str) -> None:
        """Flag suspicious activity for IP."""

        if client_ip not in self.suspicious_ips:
            self.suspicious_ips[client_ip] = 0

        self.suspicious_ips[client_ip] += 1

        self.logger.warning(
            f"Suspicious activity from {client_ip}: {reason}",
            context=LogContext(
                category=LogCategory.SECURITY_EVENT,
                metadata={
                    "client_ip": client_ip,
                    "reason": reason,
                    "count": self.suspicious_ips[client_ip],
                },
            ),
        )

        # Block IP after multiple suspicious activities
        if self.suspicious_ips[client_ip] >= 5:
            self.blocked_ips.add(client_ip)
            self.logger.error(f"IP blocked due to suspicious activity: {client_ip}")


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting API metrics."""

    def __init__(self, app, monitor: Optional[LegacySystemMonitor] = None):
        super().__init__(app)
        self.monitor = monitor

        # Request counters
        self.request_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Collect request metrics."""

        ctx = get_request_context()
        endpoint = ctx.endpoint

        # Count request
        if endpoint not in self.request_counts:
            self.request_counts[endpoint] = 0
        self.request_counts[endpoint] += 1

        start_time = time.time()

        try:
            response = await call_next(request)

            # Record successful request
            if self.monitor:
                duration = time.time() - start_time
                self.monitor.record_operation(
                    operation="api_endpoint",
                    component="api",
                    status="success",
                    duration_seconds=duration,
                )

            return response

        except Exception as e:
            # Count error
            if endpoint not in self.error_counts:
                self.error_counts[endpoint] = 0
            self.error_counts[endpoint] += 1

            # Record error
            if self.monitor:
                duration = time.time() - start_time
                self.monitor.record_operation(
                    operation="api_endpoint",
                    component="api",
                    status="error",
                    duration_seconds=duration,
                )

            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return {
            "request_counts": self.request_counts.copy(),
            "error_counts": self.error_counts.copy(),
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
        }


class CORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware for cross-origin requests."""

    def __init__(
        self,
        app,
        allowed_origins: List[str] = None,
        allowed_methods: List[str] = None,
        allowed_headers: List[str] = None,
        allow_credentials: bool = True,
    ):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "OPTIONS",
        ]
        self.allowed_headers = allowed_headers or ["*"]
        self.allow_credentials = allow_credentials

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Handle CORS."""

        origin = request.headers.get("origin")

        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
        else:
            response = await call_next(request)

        # Add CORS headers
        if origin and (origin in self.allowed_origins or "*" in self.allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = origin

        response.headers["Access-Control-Allow-Methods"] = ", ".join(
            self.allowed_methods
        )
        response.headers["Access-Control-Allow-Headers"] = ", ".join(
            self.allowed_headers
        )

        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"

        return response


# Middleware factory functions
def create_legacy_middleware(
    config: Config, monitor: Optional[LegacySystemMonitor] = None
) -> LegacySystemMiddleware:
    """Create Legacy System Whisperer middleware instance."""
    return LegacySystemMiddleware(None, config, monitor)


def create_security_middleware(config: Config) -> SecurityMiddleware:
    """Create security middleware instance."""
    return SecurityMiddleware(None, config)


def create_metrics_middleware(
    monitor: Optional[LegacySystemMonitor] = None,
) -> MetricsMiddleware:
    """Create metrics middleware instance."""
    return MetricsMiddleware(None, monitor)


# Context utilities
def get_current_user_from_context() -> Optional[User]:
    """Get current user from request context."""
    ctx = get_request_context()
    return ctx.user


def get_request_id() -> str:
    """Get current request ID."""
    ctx = get_request_context()
    return ctx.request_id


def add_context_metadata(key: str, value: Any) -> None:
    """Add metadata to current request context."""
    ctx = get_request_context()
    ctx.metadata[key] = value
    set_request_context(ctx)


def get_context_metadata(key: str, default: Any = None) -> Any:
    """Get metadata from current request context."""
    ctx = get_request_context()
    return ctx.metadata.get(key, default)


# Request timing utilities
class RequestTimer:
    """Context manager for timing request operations."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            add_context_metadata(f"{self.operation_name}_duration_ms", duration * 1000)


# Health check utilities
def is_healthy() -> bool:
    """Check if service is healthy based on middleware metrics."""
    # Would check various health indicators
    return True


def get_middleware_status() -> Dict[str, Any]:
    """Get middleware status information."""
    return {
        "middleware_active": True,
        "active_requests": len(request_context.get({}).get("active_requests", {})),
        "blocked_ips": 0,  # Would get from security middleware
        "total_requests": 0,  # Would get from metrics middleware
        "last_updated": datetime.now().isoformat(),
    }
