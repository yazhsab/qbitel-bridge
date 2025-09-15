"""
CRONOS AI Engine - API Module

This module provides REST and gRPC API interfaces for the AI Engine.
"""

from .rest import app as rest_app, AIEngineAPI, create_app, run_server
from .grpc import AIEngineGRPCService, GRPCServer, AIEngineGRPCClient, run_grpc_server
from .schemas import (
    ProtocolDiscoveryRequest,
    ProtocolDiscoveryResponse,
    FieldDetectionRequest,
    FieldDetectionResponse,
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    ModelRegistrationRequest,
    ModelRegistrationResponse,
    EngineStatusResponse,
    ErrorResponse,
    BatchProcessingRequest,
    BatchProcessingResponse,
    TrainingRequest,
    TrainingResponse,
    HealthCheckResponse,
    PredictionRequest,
    PredictionResponse,
    DataFormat,
    ProtocolType,
    DetectionLevel,
    ModelType
)
from .auth import (
    authenticate_request,
    get_api_key,
    initialize_auth,
    APIKeyManager,
    JWTManager,
    RateLimiter
)
from .middleware import (
    RateLimitMiddleware,
    LoggingMiddleware,
    SecurityMiddleware,
    MetricsMiddleware,
    CorsMiddleware,
    CompressionMiddleware,
    setup_middleware
)

__all__ = [
    # REST API
    "rest_app",
    "AIEngineAPI",
    "create_app",
    "run_server",
    
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
    "ModelRegistrationRequest",
    "ModelRegistrationResponse",
    "EngineStatusResponse",
    "ErrorResponse",
    "BatchProcessingRequest",
    "BatchProcessingResponse",
    "TrainingRequest",
    "TrainingResponse",
    "HealthCheckResponse",
    "PredictionRequest",
    "PredictionResponse",
    "DataFormat",
    "ProtocolType",
    "DetectionLevel",
    "ModelType",
    
    # Authentication
    "authenticate_request",
    "get_api_key",
    "initialize_auth",
    "APIKeyManager",
    "JWTManager",
    "RateLimiter",
    
    # Middleware
    "RateLimitMiddleware",
    "LoggingMiddleware",
    "SecurityMiddleware",
    "MetricsMiddleware",
    "CorsMiddleware",
    "CompressionMiddleware",
    "setup_middleware"
]