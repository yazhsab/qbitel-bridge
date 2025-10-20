"""
CRONOS AI Engine - API Module

This module provides REST and gRPC API interfaces for the AI Engine.
"""

from .rest import AIEngineAPI, create_app, create_ai_engine_api
from .grpc import AIEngineGRPCService, GRPCServer, AIEngineGRPCClient, run_grpc_server
from .schemas import (
    ProtocolDiscoveryRequest,
    ProtocolDiscoveryResponse,
    FieldDetectionRequest,
    FieldDetectionResponse,
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    DataFormat,
    ProtocolType,
    DetectionLevel,
)
from .auth import (
    get_current_user,
    get_api_key,
    initialize_auth,
    require_permission,
    require_role,
    login,
    refresh_access_token,
    logout,
)
from .middleware import (
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    RateLimitingMiddleware,
    ConnectionCounterMiddleware,
    ErrorHandlingMiddleware,
    setup_middleware,
)

# Marketplace (optional import - may not be available in all deployments)
try:
    from .marketplace_endpoints import router as marketplace_router
    from .marketplace_schemas import (
        ProtocolSearchRequest,
        ProtocolSubmitRequest,
        ProtocolPurchaseRequest,
        ReviewSubmitRequest,
    )
    _marketplace_available = True
except ImportError:
    _marketplace_available = False
    marketplace_router = None

__all__ = [
    # REST API
    "AIEngineAPI",
    "create_app",
    "create_ai_engine_api",
    # gRPC API
    "AIEngineGRPCService",
    "GRPCServer",
    "AIEngineGRPCClient",
    "run_grpc_server",
    # Schemas
    "ProtocolDiscoveryRequest",
    "ProtocolDiscoveryResponse",
    "FieldDetectionRequest",
    "FieldDetectionResponse",
    "AnomalyDetectionRequest",
    "AnomalyDetectionResponse",
    "DataFormat",
    "ProtocolType",
    "DetectionLevel",
    # Authentication
    "get_current_user",
    "get_api_key",
    "initialize_auth",
    "require_permission",
    "require_role",
    "login",
    "refresh_access_token",
    "logout",
    # Middleware
    "RequestLoggingMiddleware",
    "SecurityHeadersMiddleware",
    "RateLimitingMiddleware",
    "ConnectionCounterMiddleware",
    "ErrorHandlingMiddleware",
    "setup_middleware",
]

# Add marketplace exports if available
if _marketplace_available:
    __all__.extend([
        "marketplace_router",
        "ProtocolSearchRequest",
        "ProtocolSubmitRequest",
        "ProtocolPurchaseRequest",
        "ReviewSubmitRequest",
    ])
