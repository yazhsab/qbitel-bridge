"""
CRONOS AI Engine - API Middleware

This module provides custom middleware for the FastAPI application.
"""

import logging
import time
import uuid
from typing import Callable, Dict, Any, Optional
import json

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

from .auth import check_rate_limit, get_rate_limit_info
from ..core.exceptions import AuthenticationException


# Prometheus metrics
request_count = Counter(
    'cronos_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code']
)

request_duration = Histogram(
    'cronos_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

active_requests = Gauge(
    'cronos_api_active_requests',
    'Number of active API requests'
)

rate_limit_exceeded = Counter(
    'cronos_api_rate_limit_exceeded_total',
    'Total number of rate limit exceeded errors',
    ['endpoint']
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request/response logging.
    
    This middleware logs all API requests and responses with
    detailed information for monitoring and debugging.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger("cronos.api.requests")
        
        # Configure structured logging
        self.logger.setLevel(logging.INFO)
        
        # Request tracking
        self.active_requests_count = 0
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and response with logging."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Track active requests
        self.active_requests_count += 1
        active_requests.set(self.active_requests_count)
        
        # Start timing
        start_time = time.time()
        
        # Extract request information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        content_length = request.headers.get("content-length", "0")
        
        # Log request start
        self.logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "content_length": content_length,
                "headers": dict(request.headers),
                "timestamp": time.time()
            }
        )
        
        # Process request
        response = None
        error = None
        
        try:
            response = await call_next(request)
            
        except Exception as e:
            error = e
            self.logger.error(
                "Request failed with exception",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            
            # Create error response
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id
                }
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update metrics
        self.active_requests_count -= 1
        active_requests.set(self.active_requests_count)
        
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(processing_time)
        
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{processing_time:.4f}"
        
        # Log response
        log_level = logging.ERROR if response.status_code >= 400 else logging.INFO
        
        self.logger.log(
            log_level,
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "status_code": response.status_code,
                "processing_time_ms": processing_time * 1000,
                "response_size": len(response.body) if hasattr(response, 'body') else None,
                "client_ip": client_ip,
                "error": str(error) if error else None
            }
        )
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check X-Forwarded-For header (proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header (nginx)
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API rate limiting.
    
    This middleware implements rate limiting per client/API key
    to prevent abuse and ensure fair usage.
    """
    
    def __init__(self, app, rate_limit: int = 100, time_window: int = 60):
        super().__init__(app)
        self.logger = logging.getLogger("cronos.api.ratelimit")
        self.rate_limit = rate_limit
        self.time_window = time_window
        
        # Exempt paths (health checks, etc.)
        self.exempt_paths = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply rate limiting to requests."""
        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_identifier(request)
        
        # Check rate limit
        if not check_rate_limit(client_id):
            # Rate limit exceeded
            rate_limit_exceeded.labels(endpoint=request.url.path).inc()
            
            rate_info = get_rate_limit_info(client_id)
            
            self.logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_id": client_id,
                    "path": request.url.path,
                    "rate_limit": self.rate_limit,
                    "time_window": self.time_window
                }
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.rate_limit} requests per {self.time_window} seconds",
                    "retry_after": int(rate_info["reset_time"] - time.time())
                },
                headers={
                    "X-RateLimit-Limit": str(self.rate_limit),
                    "X-RateLimit-Remaining": str(rate_info["remaining_requests"]),
                    "X-RateLimit-Reset": str(int(rate_info["reset_time"])),
                    "Retry-After": str(int(rate_info["reset_time"] - time.time()))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        rate_info = get_rate_limit_info(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining_requests"])
        response.headers["X-RateLimit-Reset"] = str(int(rate_info["reset_time"]))
        
        return response
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get API key from Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token.startswith("cronos_ai_"):
                return f"api_key:{token[:20]}"  # Use first 20 chars of API key
            else:
                return f"jwt:{hash(token) % 1000000}"  # Hash JWT for anonymity
        
        # Fallback to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for security headers and validation.
    
    This middleware adds security headers and performs
    basic security validation on requests.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger("cronos.api.security")
        
        # Security configuration
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.allowed_origins = ["*"]  # Configure based on deployment
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply security measures to requests."""
        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            self.logger.warning(
                "Request size exceeded limit",
                extra={
                    "content_length": content_length,
                    "max_size": self.max_request_size,
                    "path": request.url.path
                }
            )
            
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "Request entity too large",
                    "max_size_mb": self.max_request_size // (1024 * 1024)
                }
            )
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                if request.url.path not in ["/docs", "/redoc"]:  # Allow docs endpoints
                    return JSONResponse(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        content={
                            "error": "Unsupported media type",
                            "expected": "application/json"
                        }
                    )
        
        # Check for suspicious patterns
        suspicious_patterns = ["../", "..\\", "<script", "javascript:", "data:"]
        url_path = str(request.url)
        
        for pattern in suspicious_patterns:
            if pattern in url_path.lower():
                self.logger.warning(
                    "Suspicious request pattern detected",
                    extra={
                        "pattern": pattern,
                        "url": url_path,
                        "client_ip": request.client.host if request.client else "unknown"
                    }
                )
                
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": "Invalid request"}
                )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting detailed metrics.
    
    This middleware collects comprehensive metrics about
    API usage, performance, and system health.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger("cronos.api.metrics")
        
        # Additional metrics
        self.error_count = Counter(
            'cronos_api_errors_total',
            'Total number of API errors',
            ['endpoint', 'error_type', 'status_code']
        )
        
        self.request_size = Histogram(
            'cronos_api_request_size_bytes',
            'Size of API requests in bytes',
            ['endpoint']
        )
        
        self.response_size = Histogram(
            'cronos_api_response_size_bytes',
            'Size of API responses in bytes',
            ['endpoint', 'status_code']
        )
        
        self.endpoint_usage = Counter(
            'cronos_api_endpoint_usage_total',
            'Usage count per endpoint',
            ['endpoint', 'method']
        )
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Collect metrics for requests."""
        endpoint = request.url.path
        method = request.method
        
        # Record endpoint usage
        self.endpoint_usage.labels(endpoint=endpoint, method=method).inc()
        
        # Record request size
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                self.request_size.labels(endpoint=endpoint).observe(size)
            except ValueError:
                pass
        
        # Process request
        response = await call_next(request)
        
        # Record response size
        response_body = getattr(response, 'body', b'')
        if response_body:
            response_size = len(response_body)
            self.response_size.labels(
                endpoint=endpoint,
                status_code=response.status_code
            ).observe(response_size)
        
        # Record errors
        if response.status_code >= 400:
            error_type = "client_error" if response.status_code < 500 else "server_error"
            self.error_count.labels(
                endpoint=endpoint,
                error_type=error_type,
                status_code=response.status_code
            ).inc()
        
        return response


class CorsMiddleware(BaseHTTPMiddleware):
    """
    Custom CORS middleware with enhanced logging.
    
    This middleware handles CORS requests with detailed
    logging and configurable security policies.
    """
    
    def __init__(
        self,
        app,
        allowed_origins: list = None,
        allowed_methods: list = None,
        allowed_headers: list = None,
        allow_credentials: bool = True
    ):
        super().__init__(app)
        self.logger = logging.getLogger("cronos.api.cors")
        
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = allowed_headers or ["*"]
        self.allow_credentials = allow_credentials
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Handle CORS requests."""
        origin = request.headers.get("origin")
        
        # Handle preflight OPTIONS request
        if request.method == "OPTIONS":
            response = Response()
            
            # Check if origin is allowed
            if self._is_origin_allowed(origin):
                response.headers["Access-Control-Allow-Origin"] = origin or "*"
                response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
                response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
                
                if self.allow_credentials:
                    response.headers["Access-Control-Allow-Credentials"] = "true"
                
                response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours
                
                self.logger.debug(
                    "CORS preflight request handled",
                    extra={
                        "origin": origin,
                        "method": request.headers.get("access-control-request-method"),
                        "headers": request.headers.get("access-control-request-headers")
                    }
                )
            else:
                self.logger.warning(
                    "CORS preflight request blocked",
                    extra={"origin": origin, "allowed_origins": self.allowed_origins}
                )
                response.status_code = 403
            
            return response
        
        # Process actual request
        response = await call_next(request)
        
        # Add CORS headers to response
        if self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
            
            # Expose headers
            response.headers["Access-Control-Expose-Headers"] = (
                "X-Request-ID, X-Processing-Time, X-RateLimit-Limit, "
                "X-RateLimit-Remaining, X-RateLimit-Reset"
            )
        
        return response
    
    def _is_origin_allowed(self, origin: Optional[str]) -> bool:
        """Check if origin is allowed."""
        if not origin:
            return True  # Allow requests without origin (e.g., curl)
        
        if "*" in self.allowed_origins:
            return True
        
        return origin in self.allowed_origins


class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for response compression.
    
    This middleware compresses responses based on client
    capabilities and content type.
    """
    
    def __init__(self, app, minimum_size: int = 500):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compressible_types = {
            "application/json",
            "application/xml",
            "text/html",
            "text/plain",
            "text/css",
            "application/javascript"
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply compression to responses."""
        response = await call_next(request)
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding:
            return response
        
        # Check content type
        content_type = response.headers.get("content-type", "").split(";")[0]
        if content_type not in self.compressible_types:
            return response
        
        # Check response size
        body = getattr(response, 'body', b'')
        if len(body) < self.minimum_size:
            return response
        
        # Compress response (simplified - use proper compression in production)
        import gzip
        compressed_body = gzip.compress(body)
        
        if len(compressed_body) < len(body):
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Content-Length"] = str(len(compressed_body))
            # Update response body (implementation depends on response type)
        
        return response


def setup_middleware(app):
    """
    Setup all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Order matters - middleware is applied in reverse order
    
    # Compression (outermost)
    app.add_middleware(CompressionMiddleware)
    
    # CORS
    app.add_middleware(
        CorsMiddleware,
        allowed_origins=["*"],  # Configure based on security requirements
        allowed_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allowed_headers=["*"],
        allow_credentials=True
    )
    
    # Security headers
    app.add_middleware(SecurityMiddleware)
    
    # Metrics collection
    app.add_middleware(MetricsMiddleware)
    
    # Rate limiting
    app.add_middleware(RateLimitMiddleware, rate_limit=100, time_window=60)
    
    # Request/response logging (innermost)
    app.add_middleware(LoggingMiddleware)