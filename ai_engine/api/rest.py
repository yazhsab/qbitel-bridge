"""
CRONOS AI Engine - Enhanced REST API with Protocol Intelligence Copilot
Extended FastAPI implementation with LLM-enhanced protocol analysis capabilities.
"""

import asyncio
import base64
import binascii
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from ..core.config import Config, get_config
from ..core.engine import CronosAIEngine
from ..copilot.protocol_copilot import (
    create_protocol_copilot,
    ProtocolIntelligenceCopilot,
)
from ..llm.rag_engine import RAGDocument
from ..llm.unified_llm_service import (
    initialize_llm_service,
    shutdown_llm_service,
)
from ..llm.translation_studio import (
    initialize_translation_studio,
    shutdown_translation_studio,
)
from ..monitoring.alerts import initialize_alert_manager, shutdown_alert_manager
from ..monitoring.metrics import MetricsCollector
from ..llm.security_orchestrator import (
    initialize_security_orchestrator,
    shutdown_security_orchestrator,
)
from ..policy.policy_engine import get_policy_engine
from .copilot_endpoints import router as copilot_router
from .translation_studio_endpoints import router as translation_router
from .security_orchestrator_endpoints import router as security_router
from .auth import get_current_user, verify_token
from .middleware import setup_middleware
from .schemas import *

logger = logging.getLogger(__name__)

# Global instances
_ai_engine: Optional[CronosAIEngine] = None
_protocol_copilot: Optional[ProtocolIntelligenceCopilot] = None
_metrics_collector: Optional["MetricsCollector"] = None


def create_app(config: Config = None) -> FastAPI:
    """Create and configure the FastAPI application with Protocol Intelligence Copilot."""
    if config is None:
        config = get_config()

    # Create FastAPI app
    app = FastAPI(
        title="CRONOS AI Engine with Protocol Intelligence Copilot",
        description="Enterprise-grade AI-powered protocol analysis and cybersecurity intelligence",
        version="2.0.0",
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
    )

    # Setup CORS with safe defaults that respect configured origins
    security_cfg = getattr(config, "security", None)
    raw_cors = getattr(security_cfg, "cors_origins", ["*"])
    if isinstance(raw_cors, (list, tuple, set)):
        cors_origins = list(raw_cors)
    elif raw_cors:
        cors_origins = [str(raw_cors)]
    else:
        cors_origins = ["*"]
    allow_credentials = "*" not in cors_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup additional middleware
    setup_middleware(app, config)

    # Include routers
    app.include_router(copilot_router)
    app.include_router(translation_router)
    app.include_router(security_router)

    # Enhanced API endpoints
    @app.get("/")
    async def root():
        """Root endpoint with copilot information."""
        return {
            "message": "CRONOS AI Engine with Protocol Intelligence Copilot",
            "version": "2.0.0",
            "features": [
                "Protocol Discovery and Analysis",
                "AI-Powered Field Detection",
                "Anomaly Detection",
                "Protocol Intelligence Copilot",
                "Natural Language Queries",
                "Real-time WebSocket Communication",
                "Enterprise Security",
                "Compliance Reporting",
            ],
            "endpoints": {
                "copilot": "/api/v1/copilot",
                "websocket": "/api/v1/copilot/ws",
                "docs": "/docs",
                "health": "/health",
            },
        }

    @app.get("/health")
    async def health_check():
        """Enhanced health check including copilot status."""
        global _ai_engine, _protocol_copilot

        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": app.version,
            "checks": [],
            "services": {
                "ai_engine": "unknown",
                "protocol_copilot": "unknown",
                "llm_service": "unknown",
                "rag_engine": "unknown",
            },
        }

        try:
            # Check AI Engine
            if _ai_engine:
                health_data["services"]["ai_engine"] = "healthy"
                health_data["checks"].append(
                    {
                        "component": "ai_engine",
                        "status": "healthy",
                        "details": "Engine initialized",
                    }
                )
            else:
                health_data["services"]["ai_engine"] = "not_initialized"
                health_data["checks"].append(
                    {
                        "component": "ai_engine",
                        "status": "unhealthy",
                        "details": "Engine not initialized",
                    }
                )

            # Check Protocol Copilot
            if _protocol_copilot:
                copilot_health = _protocol_copilot.get_health_status()
                health_data["services"]["protocol_copilot"] = "healthy"
                health_data["services"]["llm_service"] = copilot_health.get(
                    "llm_service", {}
                ).get("providers", {})
                health_data["services"]["rag_engine"] = copilot_health.get(
                    "rag_engine", "unknown"
                )
            else:
                health_data["services"]["protocol_copilot"] = "not_initialized"
                health_data["checks"].append(
                    {
                        "component": "protocol_copilot",
                        "status": "degraded",
                        "details": "Copilot not initialized",
                    }
                )

            if _protocol_copilot and _protocol_copilot.get_health_status():
                health_data["checks"].append(
                    {
                        "component": "llm_service",
                        "status": "healthy",
                        "details": "LLM service available",
                    }
                )
            else:
                health_data["checks"].append(
                    {
                        "component": "llm_service",
                        "status": "unknown",
                        "details": "LLM service not initialized",
                    }
                )

        except Exception as e:
            health_data["status"] = "degraded"
            health_data["error"] = str(e)

        if any(check["status"] in {"unhealthy", "degraded"} for check in health_data["checks"]):
            health_data["status"] = "degraded"

        return health_data

    # Enhanced protocol discovery endpoint
    @app.post("/api/v1/discover")
    async def discover_protocol_enhanced(
        request: ProtocolDiscoveryRequest,
        background_tasks: BackgroundTasks,
        current_user: Dict[str, Any] = Depends(get_current_user),
    ):
        """Enhanced protocol discovery with LLM analysis."""
        global _ai_engine, _protocol_copilot

        try:
            packet_data = request.packet_data
            try:
                if isinstance(packet_data, str):
                    packet_data = base64.b64decode(packet_data.encode("utf-8"), validate=True)
                else:
                    packet_data = base64.b64decode(packet_data, validate=True)
            except (ValueError, binascii.Error):
                raise HTTPException(
                    status_code=422, detail="Invalid base64 packet data"
                )

            if not _ai_engine:
                raise HTTPException(status_code=503, detail="AI Engine not initialized")

            start_time = time.time()

            result = await _ai_engine.discover_protocol(packet_data, request.metadata)

            processing_time = result.get("processing_time")
            if processing_time is None:
                processing_time = time.time() - start_time

            # Enhance with LLM analysis if copilot is available
            enhanced_analysis = None
            if _protocol_copilot and request.enable_llm_analysis:
                try:
                    from ..copilot.protocol_copilot import CopilotQuery

                    query = CopilotQuery(
                        query=f"Analyze this {result['protocol_type']} protocol data",
                        user_id=current_user.get("user_id", "system"),
                        session_id=request.session_id or "discovery_session",
                        packet_data=packet_data,
                    )

                    copilot_response = await _protocol_copilot.process_query(query)
                    enhanced_analysis = {
                        "llm_analysis": copilot_response.response,
                        "confidence": copilot_response.confidence,
                        "suggestions": copilot_response.suggestions,
                        "security_implications": "Analyzed via LLM",
                    }

                except Exception as e:
                    logger.warning(f"LLM analysis failed: {e}")
                    enhanced_analysis = {"error": "LLM analysis unavailable"}

            # Combine results
            enhanced_result = {
                "protocol_type": result.get("protocol_type", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "structure": result.get("structure", {}),
                "grammar": result.get("grammar"),
                "metadata": result.get("metadata", request.metadata or {}),
                "processing_time": processing_time,
                "enhanced_analysis": enhanced_analysis,
                "llm_enabled": _protocol_copilot is not None,
            }

            # Background task for learning
            background_tasks.add_task(
                update_knowledge_base, result, request.packet_data
            )

            return enhanced_result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Enhanced protocol discovery failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # LLM-enhanced field detection
    @app.post("/api/v1/detect-fields")
    async def detect_fields_enhanced(
        request: FieldDetectionRequest,
        current_user: Dict[str, Any] = Depends(get_current_user),
    ):
        """Enhanced field detection with LLM interpretation."""
        global _ai_engine, _protocol_copilot

        if not _ai_engine:
            raise HTTPException(status_code=503, detail="AI Engine not initialized")

        try:
            start_time = time.time()
            fields = await _ai_engine.detect_fields(
                request.message_data, request.protocol_type
            )

            processing_time = time.time() - start_time

            # Enhance with LLM interpretation
            llm_interpretation = None
            if _protocol_copilot and request.enable_llm_analysis:
                try:
                    from ..copilot.protocol_copilot import CopilotQuery

                    query = CopilotQuery(
                        query=f"Interpret these detected fields in {request.protocol_type or 'unknown'} protocol: {fields}",
                        user_id=current_user.get("user_id", "system"),
                        session_id=request.session_id or "field_session",
                        packet_data=request.message_data,
                    )

                    copilot_response = await _protocol_copilot.process_query(query)
                    llm_interpretation = {
                        "interpretation": copilot_response.response,
                        "confidence": copilot_response.confidence,
                        "field_semantics": "Analyzed via LLM",
                    }

                except Exception as e:
                    logger.warning(f"LLM field interpretation failed: {e}")

            normalized_fields: List[Dict[str, Any]] = []
            for field in fields:
                normalized_fields.append(
                    {
                        "field_id": field.get("id"),
                        "field_name": field.get("name", ""),
                        "start_offset": field.get("start"),
                        "end_offset": field.get("end"),
                        "field_type": field.get("type", "unknown"),
                        "confidence": field.get("confidence"),
                        "examples": field.get("examples", []),
                    }
                )

            return {
                "detected_fields": normalized_fields,
                "total_fields": len(normalized_fields),
                "llm_interpretation": llm_interpretation,
                "processing_time": processing_time,
            }

        except Exception as e:
            logger.error(f"Enhanced field detection failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Startup and shutdown events
    @app.on_event("startup")
    async def startup():
        """Initialize AI Engine, Protocol Copilot, Translation Studio, and Security Orchestrator."""
        global _ai_engine, _protocol_copilot, _metrics_collector

        try:
            logger.info("Starting CRONOS AI Engine with all services...")

            # Initialize AI Engine
            _ai_engine = CronosAIEngine(config)
            await _ai_engine.initialize()

            # Initialize LLM Service and set as global singleton
            llm_service = await initialize_llm_service(config)

            # Initialize Protocol Copilot
            _protocol_copilot = await create_protocol_copilot()

            # Initialize Translation Studio
            _metrics_collector = MetricsCollector(config)
            await _metrics_collector.start()  # Start metrics collection
            await initialize_translation_studio(config, llm_service, _metrics_collector)

            # Initialize Security Orchestrator
            alert_manager = await initialize_alert_manager(config)
            policy_engine = get_policy_engine()
            await initialize_security_orchestrator(
                config, llm_service, alert_manager, policy_engine
            )

            logger.info("CRONOS AI Engine with all services started successfully")

        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            raise

    @app.on_event("shutdown")
    async def shutdown():
        """Shutdown services gracefully."""
        global _ai_engine, _protocol_copilot, _metrics_collector

        try:
            logger.info("Shutting down CRONOS AI Engine and all services...")

            # Shutdown Translation Studio
            await shutdown_translation_studio()

            # Shutdown Security Orchestrator
            await shutdown_security_orchestrator()

            # Shutdown Alert Manager
            await shutdown_alert_manager()

            # Shutdown Protocol Copilot
            if _protocol_copilot:
                await _protocol_copilot.shutdown()

            # Shutdown LLM Service using proper lifecycle management
            await shutdown_llm_service()

            # Shutdown Metrics Collector
            if _metrics_collector:
                await _metrics_collector.shutdown()
                logger.info("Metrics collector shutdown complete")

            # Shutdown AI Engine
            if _ai_engine:
                await _ai_engine.shutdown()

            logger.info("Services shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    return app


# Enhanced request/response models
class ProtocolDiscoveryRequest(BaseModel):
    """Enhanced protocol discovery request."""

    packet_data: bytes = Field(..., description="Raw packet data for analysis")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    enable_llm_analysis: bool = Field(False, description="Enable LLM-enhanced analysis")
    session_id: Optional[str] = Field(None, description="Session ID for context")


class FieldDetectionRequest(BaseModel):
    """Enhanced field detection request."""

    message_data: bytes = Field(..., description="Protocol message data")
    protocol_type: Optional[str] = Field(None, description="Known protocol type")
    enable_llm_analysis: bool = Field(False, description="Enable LLM interpretation")
    session_id: Optional[str] = Field(None, description="Session ID for context")


# Background tasks
async def update_knowledge_base(result: Dict[str, Any], packet_data: bytes):
    """Update knowledge base with discovery results."""
    try:
        global _protocol_copilot

        if _protocol_copilot and result.get("confidence", 0) > 0.8:
            discovery_doc = RAGDocument(
                id=f"protocol-{result['protocol_type']}-{uuid.uuid4().hex}",
                content=f"Discovered {result['protocol_type']} protocol with {result['confidence']} confidence",
                metadata={
                    "protocol": result["protocol_type"],
                    "confidence": result["confidence"],
                    "discovery_time": datetime.utcnow().isoformat(),
                },
                created_at=datetime.utcnow(),
            )

            await _protocol_copilot.rag_engine.add_documents(
                "protocol_knowledge", [discovery_doc]
            )

    except Exception as e:
        logger.error(f"Failed to update knowledge base: {e}")


class AIEngineAPI:
    """Main API class for the AI Engine with Protocol Copilot integration."""

    def __init__(self, config: Config):
        self.config = config
        self.app = create_app(config)

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server."""
        uvicorn.run(self.app, host=host, port=port, log_level="info", access_log=True)


# Factory function
def create_ai_engine_api(config: Config = None) -> AIEngineAPI:
    """Create AI Engine API with Protocol Copilot."""
    if config is None:
        config = get_config()
    return AIEngineAPI(config)
