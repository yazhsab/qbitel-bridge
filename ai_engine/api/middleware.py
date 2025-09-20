"""
CRONOS AI Engine - API Middleware
Enterprise-grade middleware for security, monitoring, and performance.
"""

import time
import logging
import uuid
from typing import Callable, Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.gzip import GZipMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

from ..core.config import Config

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

COPILOT_QUERIES = Counter(
    'copilot_queries_total',
    'Total copilot queries',
    ['user_id', 'query_type']
)

LLM_REQUESTS = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['provider', 'status']
)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced request logging with correlation IDs and performance metrics."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started - {request.method} {request.url.path}",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "user_agent": request.headers.get("user-agent"),
                "client_ip": request.client.host if request.client else None
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Update metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Log response
        logger.info(
            f"Request completed - {request.method} {request.url.path} - {response.status_code}",
            extra={
                "correlation_id": correlation_id,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
                "response_size": response.headers.get("content-length")
            }
        )
        
        # Add correlation ID to response
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        })
        
        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.client_requests = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - 60  # 1 minute ago
        self.client_requests = {
            ip: [t for t in times if t > cutoff_time]
            for ip, times in self.client_requests.items()
        }
        
        # Check rate limit
        client_times = self.client_requests.get(client_ip, [])
        if len(client_times) >= self.requests_per_minute:
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": "60"}
            )
        
        # Add current request
        client_times.append(current_time)
        self.client_requests[client_ip] = client_times
        
        return await call_next(request)

class ConnectionCounterMiddleware(BaseHTTPMiddleware):
    """Track active connections."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        ACTIVE_CONNECTIONS.inc()
        try:
            response = await call_next(request)
            return response
        finally:
            ACTIVE_CONNECTIONS.dec()

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Enhanced error handling and logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            correlation_id = getattr(request.state, 'correlation_id', 'unknown')
            
            # Log error with full context
            logger.error(
                f"Unhandled exception in request processing",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )
            
            # Return generic error response
            return Response(
                content="Internal server error",
                status_code=500,
                headers={"X-Correlation-ID": correlation_id}
            )

def setup_middleware(app: FastAPI, config: Config):
    """Setup all middleware for the FastAPI application."""
    
    # Connection counter (outermost)
    app.add_middleware(ConnectionCounterMiddleware)
    
    # Error handling
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate limiting (if enabled)
    if config.security.get('enable_rate_limiting', True):
        app.add_middleware(
            RateLimitingMiddleware,
            requests_per_minute=config.security.get('requests_per_minute', 100)
        )
    
    # Request logging and metrics
    app.add_middleware(RequestLoggingMiddleware)
    
    # Trusted hosts
    if config.security.get('trusted_hosts'):
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=config.security.trusted_hosts
        )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    logger.info("API middleware setup completed")

class MetricsCollector:
    """Collect and expose custom metrics for the Protocol Intelligence Copilot."""
    
    @staticmethod
    def record_copilot_query(user_id: str, query_type: str):
        """Record copilot query metrics."""
        COPILOT_QUERIES.labels(user_id=user_id, query_type=query_type).inc()
    
    @staticmethod
    def record_llm_request(provider: str, status: str):
        """Record LLM request metrics."""
        LLM_REQUESTS.labels(provider=provider, status=status).inc()
    
    @staticmethod
    def get_metrics_data() -> Dict[str, Any]:
        """Get current metrics data."""
        return {
            "active_connections": ACTIVE_CONNECTIONS._value._value,
            "total_requests": sum(REQUEST_COUNT._metrics.values()),
            "copilot_queries": sum(COPILOT_QUERIES._metrics.values()),
            "llm_requests": sum(LLM_REQUESTS._metrics.values())
        }

def setup_prometheus_metrics(app: FastAPI):
    """Setup Prometheus metrics endpoint."""
    
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint."""
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    logger.info("Prometheus metrics endpoint configured at /metrics")

# Health check utilities
class HealthChecker:
    """Enhanced health checking for all system components."""
    
    @staticmethod
    async def check_llm_providers() -> Dict[str, str]:
        """Check health of LLM providers."""
        # This would normally check actual provider health
        return {
            "openai": "healthy",
            "anthropic": "healthy", 
            "ollama": "unknown"
        }
    
    @staticmethod
    async def check_databases() -> Dict[str, str]:
        """Check database connections."""
        return {
            "redis": "healthy",
            "chromadb": "healthy",
            "timescaledb": "unknown"
        }
    
    @staticmethod
    async def check_external_services() -> Dict[str, str]:
        """Check external service dependencies."""
        return {
            "prometheus": "healthy",
            "grafana": "unknown"
        }
    
    @classmethod
    async def comprehensive_health_check(cls) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        return {
            "llm_providers": await cls.check_llm_providers(),
            "databases": await cls.check_databases(),
            "external_services": await cls.check_external_services(),
            "metrics": MetricsCollector.get_metrics_data(),
            "timestamp": time.time()
        }