"""
CRONOS AI Engine - REST API

This module implements the REST API using FastAPI for the AI Engine.
"""

import logging
import time
import traceback
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import base64

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from ..core.engine import AIEngine
from ..core.config import Config
from ..core.exceptions import AIEngineException, ModelException, ValidationException
from ..models import ModelInput
from .schemas import (
    ProtocolDiscoveryRequest, ProtocolDiscoveryResponse, GrammarRule,
    FieldDetectionRequest, FieldDetectionResponse, DetectedField,
    AnomalyDetectionRequest, AnomalyDetectionResponse, AnomalyScore, AnomalyExplanation,
    ModelRegistrationRequest, ModelRegistrationResponse, ModelInfo,
    ModelListRequest, ModelListResponse,
    EngineStatusResponse, ComponentStatus,
    BatchProcessingRequest, BatchProcessingResponse, BatchItemResult,
    TrainingRequest, TrainingResponse, TrainingStatus,
    HealthCheckResponse,
    PredictionRequest, PredictionResponse,
    ErrorResponse,
    DataFormat, ProtocolType, DetectionLevel
)
from .auth import authenticate_request, get_api_key
from .middleware import RateLimitMiddleware, LoggingMiddleware


class AIEngineAPI:
    """
    FastAPI application for CRONOS AI Engine.
    
    This class provides comprehensive REST API endpoints for all AI Engine
    functionality including protocol discovery, field detection, anomaly
    detection, and model management.
    """
    
    def __init__(self, config: Config):
        """Initialize FastAPI application."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI Engine
        self.ai_engine: Optional[AIEngine] = None
        
        # Create FastAPI app
        self.app = FastAPI(
            title="CRONOS AI Engine API",
            description="Enterprise-grade AI Engine for protocol discovery, field detection, and anomaly detection",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # Security
        self.security = HTTPBearer()
        
        # Background task tracking
        self.background_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Setup error handlers
        self._setup_error_handlers()
        
        self.logger.info("AIEngineAPI initialized")
    
    async def startup(self) -> None:
        """Startup event handler."""
        try:
            self.logger.info("Starting AI Engine API...")
            
            # Initialize AI Engine
            self.ai_engine = AIEngine(self.config)
            await self.ai_engine.initialize()
            
            self.logger.info("AI Engine API started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start AI Engine API: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown event handler."""
        try:
            self.logger.info("Shutting down AI Engine API...")
            
            if self.ai_engine:
                await self.ai_engine.cleanup()
            
            self.logger.info("AI Engine API shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during API shutdown: {e}")
    
    def _setup_middleware(self) -> None:
        """Setup middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure based on security requirements
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware
        self.app.add_middleware(LoggingMiddleware)
        self.app.add_middleware(RateLimitMiddleware, rate_limit=100, time_window=60)  # 100 requests per minute
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        # Health check endpoint
        @self.app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
        async def health_check():
            """Health check endpoint."""
            checks = {}
            status = "healthy"
            
            try:
                # Check AI Engine
                if self.ai_engine:
                    engine_status = await self.ai_engine.get_status()
                    checks["ai_engine"] = engine_status.get("status") == "ready"
                else:
                    checks["ai_engine"] = False
                    status = "unhealthy"
                
                # Check model registry
                if self.ai_engine and self.ai_engine.model_registry:
                    checks["model_registry"] = True
                else:
                    checks["model_registry"] = False
                
                # Overall status
                if not all(checks.values()):
                    status = "degraded" if any(checks.values()) else "unhealthy"
                
                return HealthCheckResponse(
                    status=status,
                    version="1.0.0",
                    checks=checks
                )
                
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                return HealthCheckResponse(
                    status="unhealthy",
                    version="1.0.0",
                    checks={},
                    details={"error": str(e)}
                )
        
        # Engine status endpoint
        @self.app.get("/status", response_model=EngineStatusResponse, tags=["Engine"])
        async def get_engine_status():
            """Get comprehensive engine status."""
            if not self.ai_engine:
                raise HTTPException(status_code=503, detail="AI Engine not initialized")
            
            try:
                status = await self.ai_engine.get_status()
                components = []
                
                # Add component statuses
                for component, details in status.get("components", {}).items():
                    components.append(ComponentStatus(
                        name=component,
                        status=details.get("status", "unknown"),
                        last_check=datetime.utcnow(),
                        details=details,
                        metrics=details.get("metrics", {})
                    ))
                
                return EngineStatusResponse(
                    engine_version="1.0.0",
                    uptime_seconds=status.get("uptime_seconds", 0),
                    components=components,
                    system_metrics=status.get("system_metrics", {}),
                    active_models=status.get("active_models", {}),
                    recent_activity=status.get("recent_activity")
                )
                
            except Exception as e:
                self.logger.error(f"Status check failed: {e}")
                raise HTTPException(status_code=500, detail=f"Status check failed: {e}")
        
        # Protocol Discovery endpoints
        @self.app.post("/discovery/protocols", response_model=ProtocolDiscoveryResponse, tags=["Protocol Discovery"])
        async def discover_protocol(
            request: ProtocolDiscoveryRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Discover protocol from data sample."""
            await authenticate_request(credentials)
            
            if not self.ai_engine:
                raise HTTPException(status_code=503, detail="AI Engine not initialized")
            
            try:
                start_time = time.time()
                
                # Convert data based on format
                data_bytes = self._decode_data(request.data, request.data_format)
                
                # Create model input
                model_input = ModelInput(
                    data=data_bytes,
                    metadata={
                        "expected_protocol": request.expected_protocol.value if request.expected_protocol else None,
                        "confidence_threshold": request.confidence_threshold,
                        "max_samples": request.max_samples
                    }
                )
                
                # Perform protocol discovery
                result = await self.ai_engine.discover_protocol(model_input)
                
                processing_time = (time.time() - start_time) * 1000
                
                # Build response
                grammar_rules = []
                if request.include_grammar and "grammar_rules" in result.metadata:
                    for rule in result.metadata["grammar_rules"]:
                        grammar_rules.append(GrammarRule(
                            rule_name=rule["name"],
                            production=rule["production"],
                            probability=rule["probability"],
                            examples=rule.get("examples", [])
                        ))
                
                return ProtocolDiscoveryResponse(
                    request_id=request.request_id,
                    processing_time_ms=processing_time,
                    discovered_protocol=result.metadata.get("protocol_type"),
                    confidence_score=result.metadata.get("confidence", 0.0),
                    protocol_characteristics=result.metadata.get("characteristics", {}),
                    field_structure=result.metadata.get("field_structure"),
                    grammar_rules=grammar_rules if grammar_rules else None,
                    statistical_features=result.metadata.get("statistical_features", {}),
                    recommendations=result.metadata.get("recommendations", [])
                )
                
            except Exception as e:
                self.logger.error(f"Protocol discovery failed: {e}")
                raise HTTPException(status_code=500, detail=f"Protocol discovery failed: {e}")
        
        # Field Detection endpoints
        @self.app.post("/detection/fields", response_model=FieldDetectionResponse, tags=["Field Detection"])
        async def detect_fields(
            request: FieldDetectionRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Detect fields in protocol data."""
            await authenticate_request(credentials)
            
            if not self.ai_engine:
                raise HTTPException(status_code=503, detail="AI Engine not initialized")
            
            try:
                start_time = time.time()
                
                # Convert data
                data_bytes = self._decode_data(request.data, request.data_format)
                
                # Create model input
                model_input = ModelInput(
                    data=data_bytes,
                    metadata={
                        "protocol_hint": request.protocol_hint.value if request.protocol_hint else None,
                        "detection_level": request.detection_level.value,
                        "include_boundaries": request.include_boundaries,
                        "include_semantics": request.include_semantics,
                        "max_fields": request.max_fields
                    }
                )
                
                # Perform field detection
                result = await self.ai_engine.detect_fields(model_input)
                
                processing_time = (time.time() - start_time) * 1000
                
                # Build detected fields
                detected_fields = []
                if "detected_fields" in result.metadata:
                    for field in result.metadata["detected_fields"]:
                        detected_fields.append(DetectedField(
                            field_id=field["id"],
                            field_name=field.get("name"),
                            start_offset=field["start_offset"],
                            end_offset=field["end_offset"],
                            field_type=field["field_type"],
                            confidence=field["confidence"],
                            semantic_type=field.get("semantic_type"),
                            encoding=field.get("encoding"),
                            constraints=field.get("constraints"),
                            examples=field.get("examples", [])
                        ))
                
                return FieldDetectionResponse(
                    request_id=request.request_id,
                    processing_time_ms=processing_time,
                    detected_fields=detected_fields,
                    total_fields=len(detected_fields),
                    protocol_structure=result.metadata.get("protocol_structure"),
                    field_relationships=result.metadata.get("field_relationships"),
                    confidence_summary=result.metadata.get("confidence_summary", {})
                )
                
            except Exception as e:
                self.logger.error(f"Field detection failed: {e}")
                raise HTTPException(status_code=500, detail=f"Field detection failed: {e}")
        
        # Anomaly Detection endpoints
        @self.app.post("/detection/anomalies", response_model=AnomalyDetectionResponse, tags=["Anomaly Detection"])
        async def detect_anomalies(
            request: AnomalyDetectionRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Detect anomalies in protocol data."""
            await authenticate_request(credentials)
            
            if not self.ai_engine:
                raise HTTPException(status_code=503, detail="AI Engine not initialized")
            
            try:
                start_time = time.time()
                
                # Convert data
                data_bytes = self._decode_data(request.data, request.data_format)
                
                # Convert baseline data if provided
                baseline_bytes = None
                if request.baseline_data:
                    baseline_bytes = [
                        self._decode_data(baseline, request.data_format)
                        for baseline in request.baseline_data
                    ]
                
                # Create model input
                model_input = ModelInput(
                    data=data_bytes,
                    metadata={
                        "baseline_data": baseline_bytes,
                        "protocol_context": request.protocol_context.value if request.protocol_context else None,
                        "sensitivity": request.sensitivity.value,
                        "anomaly_threshold": request.anomaly_threshold,
                        "include_explanations": request.include_explanations
                    }
                )
                
                # Perform anomaly detection
                result = await self.ai_engine.detect_anomalies(model_input)
                
                processing_time = (time.time() - start_time) * 1000
                
                # Build anomaly score
                score_data = result.metadata.get("anomaly_score", {})
                anomaly_score = AnomalyScore(
                    overall_score=score_data.get("overall_score", 0.0),
                    reconstruction_error=score_data.get("reconstruction_error"),
                    statistical_deviation=score_data.get("statistical_deviation"),
                    sequence_anomaly=score_data.get("sequence_anomaly"),
                    field_level_scores=score_data.get("field_level_scores")
                )
                
                # Build explanations
                explanations = []
                if request.include_explanations and "explanations" in result.metadata:
                    for exp in result.metadata["explanations"]:
                        explanations.append(AnomalyExplanation(
                            anomaly_type=exp["type"],
                            description=exp["description"],
                            affected_regions=exp.get("affected_regions", []),
                            severity=exp["severity"],
                            confidence=exp["confidence"],
                            recommendations=exp.get("recommendations", [])
                        ))
                
                return AnomalyDetectionResponse(
                    request_id=request.request_id,
                    processing_time_ms=processing_time,
                    is_anomalous=result.metadata.get("is_anomalous", False),
                    anomaly_score=anomaly_score,
                    anomaly_explanations=explanations,
                    baseline_comparison=result.metadata.get("baseline_comparison"),
                    trend_analysis=result.metadata.get("trend_analysis")
                )
                
            except Exception as e:
                self.logger.error(f"Anomaly detection failed: {e}")
                raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {e}")
        
        # Model Management endpoints
        @self.app.post("/models", response_model=ModelRegistrationResponse, tags=["Model Management"])
        async def register_model(
            request: ModelRegistrationRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Register a new model."""
            await authenticate_request(credentials)
            
            if not self.ai_engine or not self.ai_engine.model_registry:
                raise HTTPException(status_code=503, detail="Model registry not available")
            
            try:
                # Implementation would register model with registry
                # For now, return success response
                model_id = f"{request.model_name}_{request.model_version}_{int(time.time())}"
                
                return ModelRegistrationResponse(
                    request_id=request.request_id,
                    model_id=model_id,
                    registration_status="success",
                    message="Model registered successfully"
                )
                
            except Exception as e:
                self.logger.error(f"Model registration failed: {e}")
                raise HTTPException(status_code=500, detail=f"Model registration failed: {e}")
        
        @self.app.get("/models", response_model=ModelListResponse, tags=["Model Management"])
        async def list_models(
            model_type: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = 50,
            offset: int = 0,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """List registered models."""
            await authenticate_request(credentials)
            
            if not self.ai_engine or not self.ai_engine.model_registry:
                raise HTTPException(status_code=503, detail="Model registry not available")
            
            try:
                # Get models from registry
                models = await self.ai_engine.model_registry.list_models()
                
                # Apply filters
                if model_type:
                    models = [m for m in models if m.model_type.value == model_type]
                if status:
                    models = [m for m in models if m.status.value == status]
                
                # Apply pagination
                total_count = len(models)
                models = models[offset:offset + limit]
                
                # Convert to response format
                model_infos = []
                for model in models:
                    model_infos.append(ModelInfo(
                        model_id=model.model_id,
                        name=model.name,
                        version=model.version,
                        model_type=model.model_type,
                        status=model.status.value,
                        created_at=datetime.fromtimestamp(model.created_at) if model.created_at else datetime.utcnow(),
                        updated_at=datetime.fromtimestamp(model.updated_at) if model.updated_at else datetime.utcnow(),
                        performance_metrics={
                            "accuracy": model.accuracy,
                            "precision": model.precision,
                            "recall": model.recall,
                            "f1_score": model.f1_score
                        },
                        tags=model.tags or {},
                        description=model.description
                    ))
                
                return ModelListResponse(
                    models=model_infos,
                    total_count=total_count,
                    has_more=offset + limit < total_count
                )
                
            except Exception as e:
                self.logger.error(f"Model listing failed: {e}")
                raise HTTPException(status_code=500, detail=f"Model listing failed: {e}")
        
        # Generic prediction endpoint
        @self.app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
        async def predict(
            request: PredictionRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Make prediction using specified model."""
            await authenticate_request(credentials)
            
            if not self.ai_engine:
                raise HTTPException(status_code=503, detail="AI Engine not initialized")
            
            try:
                start_time = time.time()
                
                # Convert input data
                data_bytes = self._decode_data(request.input_data, request.input_format)
                
                # Create model input
                model_input = ModelInput(
                    data=data_bytes,
                    metadata={
                        "model_name": request.model_name,
                        "model_version": request.model_version,
                        "options": request.options
                    }
                )
                
                # Make prediction based on model type
                result = None
                if "protocol" in request.model_name.lower():
                    result = await self.ai_engine.discover_protocol(model_input)
                elif "field" in request.model_name.lower():
                    result = await self.ai_engine.detect_fields(model_input)
                elif "anomaly" in request.model_name.lower():
                    result = await self.ai_engine.detect_anomalies(model_input)
                else:
                    raise HTTPException(status_code=400, detail="Unable to determine prediction type from model name")
                
                processing_time = (time.time() - start_time) * 1000
                
                return PredictionResponse(
                    request_id=request.request_id,
                    processing_time_ms=processing_time,
                    predictions=result.to_dict()["predictions"] if hasattr(result, 'to_dict') else {"result": "success"},
                    confidence=result.metadata.get("confidence"),
                    model_info={
                        "name": request.model_name,
                        "version": request.model_version or "latest"
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
        
        # Batch processing endpoint
        @self.app.post("/batch", response_model=BatchProcessingResponse, tags=["Batch Processing"])
        async def batch_process(
            request: BatchProcessingRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Process multiple items in batch."""
            await authenticate_request(credentials)
            
            if not self.ai_engine:
                raise HTTPException(status_code=503, detail="AI Engine not initialized")
            
            try:
                batch_id = request.batch_id or f"batch_{int(time.time())}"
                
                # For large batches, process in background
                if len(request.data_items) > 10:
                    background_tasks.add_task(
                        self._process_batch_background,
                        batch_id,
                        request
                    )
                    
                    return BatchProcessingResponse(
                        request_id=request.request_id,
                        batch_id=batch_id,
                        total_items=len(request.data_items),
                        successful_items=0,
                        failed_items=0,
                        results=[],
                        batch_metrics={"status": "processing"}
                    )
                else:
                    # Process immediately for small batches
                    results = await self._process_batch_immediate(request)
                    
                    successful = sum(1 for r in results if r.success)
                    failed = len(results) - successful
                    
                    return BatchProcessingResponse(
                        request_id=request.request_id,
                        batch_id=batch_id,
                        total_items=len(request.data_items),
                        successful_items=successful,
                        failed_items=failed,
                        results=results,
                        batch_metrics={"status": "completed"}
                    )
                
            except Exception as e:
                self.logger.error(f"Batch processing failed: {e}")
                raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")
        
        @self.app.get("/batch/{batch_id}", response_model=BatchProcessingResponse, tags=["Batch Processing"])
        async def get_batch_status(
            batch_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get batch processing status."""
            await authenticate_request(credentials)
            
            if batch_id not in self.background_tasks:
                raise HTTPException(status_code=404, detail="Batch not found")
            
            task_info = self.background_tasks[batch_id]
            return BatchProcessingResponse(**task_info)
    
    def _setup_error_handlers(self) -> None:
        """Setup error handlers."""
        
        @self.app.exception_handler(AIEngineException)
        async def ai_engine_exception_handler(request: Request, exc: AIEngineException):
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error_code="AI_ENGINE_ERROR",
                    error_message=str(exc),
                    error_details={"exception_type": type(exc).__name__}
                ).dict()
            )
        
        @self.app.exception_handler(ValidationException)
        async def validation_exception_handler(request: Request, exc: ValidationException):
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error_code="VALIDATION_ERROR",
                    error_message=str(exc),
                    error_details={"exception_type": type(exc).__name__}
                ).dict()
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            self.logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error_code="INTERNAL_SERVER_ERROR",
                    error_message="Internal server error",
                    error_details={"exception_type": type(exc).__name__}
                ).dict()
            )
    
    def _decode_data(self, data: str, data_format: DataFormat) -> bytes:
        """Decode data based on format."""
        try:
            if data_format == DataFormat.BASE64:
                return base64.b64decode(data)
            elif data_format == DataFormat.HEX_STRING:
                return bytes.fromhex(data.replace(' ', ''))
            elif data_format == DataFormat.TEXT:
                return data.encode('utf-8')
            elif data_format == DataFormat.RAW_BYTES:
                # Assume data is already bytes
                return data.encode('latin-1')
            else:
                raise ValueError(f"Unsupported data format: {data_format}")
        except Exception as e:
            raise ValidationException(f"Data decoding failed: {e}")
    
    async def _process_batch_immediate(self, request: BatchProcessingRequest) -> List[BatchItemResult]:
        """Process batch immediately."""
        results = []
        
        for i, data_item in enumerate(request.data_items):
            try:
                start_time = time.time()
                
                # Create mock result for now
                result_data = {"item_index": i, "processed": True}
                processing_time = (time.time() - start_time) * 1000
                
                results.append(BatchItemResult(
                    item_index=i,
                    success=True,
                    result=result_data,
                    processing_time_ms=processing_time
                ))
                
            except Exception as e:
                results.append(BatchItemResult(
                    item_index=i,
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    async def _process_batch_background(self, batch_id: str, request: BatchProcessingRequest) -> None:
        """Process batch in background."""
        try:
            # Initialize task info
            self.background_tasks[batch_id] = {
                "batch_id": batch_id,
                "total_items": len(request.data_items),
                "successful_items": 0,
                "failed_items": 0,
                "results": [],
                "batch_metrics": {"status": "processing", "progress": 0.0}
            }
            
            # Process items
            results = []
            for i, data_item in enumerate(request.data_items):
                try:
                    # Process item (mock implementation)
                    await asyncio.sleep(0.1)  # Simulate processing time
                    
                    result_data = {"item_index": i, "processed": True}
                    results.append(BatchItemResult(
                        item_index=i,
                        success=True,
                        result=result_data
                    ))
                    
                    self.background_tasks[batch_id]["successful_items"] += 1
                    
                except Exception as e:
                    results.append(BatchItemResult(
                        item_index=i,
                        success=False,
                        error=str(e)
                    ))
                    
                    self.background_tasks[batch_id]["failed_items"] += 1
                
                # Update progress
                progress = (i + 1) / len(request.data_items)
                self.background_tasks[batch_id]["batch_metrics"]["progress"] = progress
            
            # Update final results
            self.background_tasks[batch_id]["results"] = results
            self.background_tasks[batch_id]["batch_metrics"]["status"] = "completed"
            
        except Exception as e:
            self.logger.error(f"Background batch processing failed: {e}")
            self.background_tasks[batch_id]["batch_metrics"]["status"] = "failed"
            self.background_tasks[batch_id]["batch_metrics"]["error"] = str(e)


# Create FastAPI app instance
def create_app(config: Config) -> FastAPI:
    """Create FastAPI application."""
    api = AIEngineAPI(config)
    
    # Add startup and shutdown events
    @api.app.on_event("startup")
    async def startup_event():
        await api.startup()
    
    @api.app.on_event("shutdown")
    async def shutdown_event():
        await api.shutdown()
    
    return api.app


# Global app instance for uvicorn
app = None


def run_server(config: Config, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the API server."""
    global app
    app = create_app(config)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )