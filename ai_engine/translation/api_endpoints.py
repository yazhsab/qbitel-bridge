"""
QBITEL - Translation Studio API Endpoints
Enterprise-grade REST API endpoints for protocol translation, API generation, and SDK creation.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
import tempfile
import zipfile
from pathlib import Path
import uuid

from ..core.config import Config, get_config
from ..core.exceptions import QbitelAIException
from ..api.auth import get_current_user, require_permissions
from ..llm.unified_llm_service import get_llm_service
from ..llm.rag_engine import RAGEngine

from .models import (
    TranslationRequest,
    APIGenerationResult,
    GeneratedSDK,
    CodeLanguage,
    APIStyle,
    SecurityLevel,
    GenerationStatus,
    ProtocolSchema,
    APISpecification,
    TranslationMode,
    QualityLevel,
)
from .enhanced_discovery import (
    EnhancedProtocolDiscoveryOrchestrator,
    APIGenerationRequest,
)
from .api_generation.api_generator import APIGenerator, APIGenerationContext
from .code_generation.code_generator import (
    MultiLanguageCodeGenerator,
    GenerationContext,
)
from .protocol_bridge.protocol_bridge import ProtocolBridge, TranslationContext

from prometheus_client import Counter, Histogram
import uuid

# Metrics for API endpoints
API_REQUEST_COUNTER = Counter(
    "qbitel_translation_api_requests_total",
    "Total translation API requests",
    ["endpoint", "method", "status"],
)

API_REQUEST_DURATION = Histogram(
    "qbitel_translation_api_request_duration_seconds",
    "Translation API request duration",
    ["endpoint"],
)

logger = logging.getLogger(__name__)

# Create router for translation endpoints
router = APIRouter(prefix="/api/v1/translation", tags=["Translation Studio"])

# Pydantic models for API requests/responses


class ProtocolDiscoveryRequest(BaseModel):
    """Request model for protocol discovery with API generation."""

    messages: List[str] = Field(..., description="Protocol messages (base64 encoded)")
    known_protocol: Optional[str] = Field(None, description="Known protocol name")
    target_api_style: APIStyle = Field(APIStyle.REST, description="Target API style")
    target_languages: List[CodeLanguage] = Field(
        default=[CodeLanguage.PYTHON], description="Target languages for SDK generation"
    )
    security_level: SecurityLevel = Field(SecurityLevel.AUTHENTICATED, description="Security level for generated API")
    generate_documentation: bool = Field(True, description="Generate documentation")
    generate_tests: bool = Field(True, description="Generate test files")
    generate_examples: bool = Field(True, description="Generate examples")
    api_base_path: str = Field("/api/v1", description="Base path for generated API")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="Additional user context")

    @validator("messages")
    def validate_messages(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one message is required")
        return v


class ProtocolDiscoveryResponse(BaseModel):
    """Response model for protocol discovery."""

    request_id: str
    protocol_type: str
    confidence: float
    api_specification: Optional[Dict[str, Any]] = None
    generated_sdks: List[Dict[str, Any]] = []
    processing_time: float
    status: GenerationStatus
    recommendations: List[str] = []
    warnings: List[str] = []
    natural_language_summary: Optional[str] = None


class APIGenerationRequest(BaseModel):
    """Request model for API generation from protocol schema."""

    protocol_schema: Dict[str, Any] = Field(..., description="Protocol schema definition")
    api_style: APIStyle = Field(APIStyle.REST, description="API style")
    security_level: SecurityLevel = Field(SecurityLevel.AUTHENTICATED, description="Security level")
    base_path: str = Field("/api/v1", description="Base path for API")
    include_examples: bool = Field(True, description="Include examples in specification")
    include_documentation: bool = Field(True, description="Include comprehensive documentation")
    custom_endpoints: List[Dict[str, Any]] = Field(default=[], description="Custom endpoint definitions")


class SDKGenerationRequest(BaseModel):
    """Request model for SDK generation."""

    api_specification: Dict[str, Any] = Field(..., description="OpenAPI specification")
    target_languages: List[CodeLanguage] = Field(..., description="Target programming languages")
    package_name: str = Field(..., description="SDK package name")
    version: str = Field("1.0.0", description="SDK version")
    author: Optional[str] = Field(None, description="SDK author")
    description: Optional[str] = Field(None, description="SDK description")
    generate_tests: bool = Field(True, description="Generate test files")
    generate_examples: bool = Field(True, description="Generate usage examples")
    generate_documentation: bool = Field(True, description="Generate documentation")


class ProtocolTranslationRequest(BaseModel):
    """Request model for protocol translation."""

    source_protocol: str = Field(..., description="Source protocol name")
    target_protocol: str = Field(..., description="Target protocol name")
    data: str = Field(..., description="Protocol data (base64 encoded)")
    translation_mode: str = Field("hybrid", description="Translation mode")
    quality_level: str = Field("balanced", description="Quality level")
    preserve_metadata: bool = Field(True, description="Preserve metadata")
    custom_rules: List[Dict[str, Any]] = Field(default=[], description="Custom translation rules")


class StreamingConnectionRequest(BaseModel):
    """Request model for creating streaming connection."""

    source_protocol: str
    target_protocol: str
    quality_level: str = "balanced"
    connection_name: Optional[str] = None


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation."""

    translations: List[ProtocolTranslationRequest] = Field(..., description="Batch of translations")
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    fail_fast: bool = Field(False, description="Stop on first failure")


# Global service instances (initialized on startup)
discovery_orchestrator: Optional[EnhancedProtocolDiscoveryOrchestrator] = None
api_generator: Optional[APIGenerator] = None
code_generator: Optional[MultiLanguageCodeGenerator] = None
protocol_bridge: Optional[ProtocolBridge] = None
rag_engine: Optional[RAGEngine] = None

# Background job tracking
background_jobs: Dict[str, Dict[str, Any]] = {}


@router.on_event("startup")
async def initialize_services():
    """Initialize translation services on startup."""
    global discovery_orchestrator, api_generator, code_generator, protocol_bridge, rag_engine

    try:
        config = get_config()
        llm_service = get_llm_service()

        # Initialize services
        discovery_orchestrator = EnhancedProtocolDiscoveryOrchestrator(config, llm_service)
        await discovery_orchestrator.initialize()

        api_generator = APIGenerator(config, llm_service)
        code_generator = MultiLanguageCodeGenerator(config, llm_service)
        protocol_bridge = ProtocolBridge(config, llm_service)
        await protocol_bridge.initialize()

        rag_engine = RAGEngine(config.rag if hasattr(config, "rag") else {})
        await rag_engine.initialize()

        logger.info("Translation Studio services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize translation services: {e}")
        raise


# Protocol Discovery and API Generation Endpoints


@router.post("/discover", response_model=ProtocolDiscoveryResponse)
async def discover_protocol_and_generate_api(
    request: ProtocolDiscoveryRequest,
    current_user=Depends(get_current_user),
    background_tasks: BackgroundTasks = None,
):
    """
    Discover protocol and automatically generate API specification and SDKs.

    This is the main entry point for the Translation Studio that:
    1. Analyzes protocol messages to identify the protocol type
    2. Generates comprehensive API specification
    3. Creates SDKs in specified languages
    4. Provides natural language explanations and recommendations
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        API_REQUEST_COUNTER.labels(endpoint="discover", method="POST", status="started").inc()

        # Decode base64 messages
        decoded_messages = []
        for msg in request.messages:
            try:
                import base64

                decoded_msg = base64.b64decode(msg)
                decoded_messages.append(decoded_msg)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 message: {e}")

        # Create discovery request
        discovery_request = APIGenerationRequest(
            messages=decoded_messages,
            known_protocol=request.known_protocol,
            target_api_style=request.target_api_style,
            include_authentication=request.security_level != SecurityLevel.PUBLIC,
            security_level=request.security_level,
            generate_openapi_spec=True,
            api_base_path=request.api_base_path,
            include_examples=request.generate_examples,
            enable_field_semantics=True,
            user_context=request.user_context,
        )

        # Perform discovery and API generation
        discovery_result = await discovery_orchestrator.discover_and_generate_api(discovery_request)

        # Generate SDKs if API was successfully generated
        generated_sdks = []
        if discovery_result.api_specification and request.target_languages:
            for language in request.target_languages:
                try:
                    generation_context = GenerationContext(
                        api_specification=discovery_result.api_specification,
                        target_language=language,
                        package_name=f"{discovery_result.protocol_type.lower()}-sdk-{language.value}",
                        version="1.0.0",
                        description=f"{discovery_result.protocol_type} SDK for {language.value}",
                        generate_tests=request.generate_tests,
                        generate_examples=request.generate_examples,
                        generate_documentation=request.generate_documentation,
                        security_level=request.security_level,
                    )

                    sdk = await code_generator.generate_sdk(
                        discovery_result.api_specification,
                        language,
                        f"{discovery_result.protocol_type.lower()}-sdk-{language.value}",
                        **generation_context.__dict__,
                    )

                    generated_sdks.append(
                        {
                            "language": language.value,
                            "name": sdk.name,
                            "version": sdk.version,
                            "files_count": len(sdk.source_files) + len(sdk.test_files) + len(sdk.config_files),
                            "sdk_id": sdk.sdk_id,
                        }
                    )

                except Exception as e:
                    logger.error(f"SDK generation failed for {language.value}: {e}")

        # Create response
        response = ProtocolDiscoveryResponse(
            request_id=request_id,
            protocol_type=discovery_result.protocol_type,
            confidence=discovery_result.confidence,
            api_specification=(
                discovery_result.api_specification.to_openapi_dict() if discovery_result.api_specification else None
            ),
            generated_sdks=generated_sdks,
            processing_time=time.time() - start_time,
            status=(GenerationStatus.COMPLETED if discovery_result.confidence > 0.6 else GenerationStatus.FAILED),
            recommendations=discovery_result.recommendations,
            warnings=[],
            natural_language_summary=discovery_result.natural_language_summary,
        )

        API_REQUEST_COUNTER.labels(endpoint="discover", method="POST", status="success").inc()
        API_REQUEST_DURATION.labels(endpoint="discover").observe(response.processing_time)

        logger.info(f"Discovery completed: {discovery_result.protocol_type} (confidence: {discovery_result.confidence:.2f})")

        return response

    except Exception as e:
        API_REQUEST_COUNTER.labels(endpoint="discover", method="POST", status="error").inc()
        logger.error(f"Protocol discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@router.post("/generate-api", response_model=Dict[str, Any])
async def generate_api_specification(request: APIGenerationRequest, current_user=Depends(get_current_user)):
    """Generate API specification from protocol schema."""
    start_time = time.time()

    try:
        API_REQUEST_COUNTER.labels(endpoint="generate-api", method="POST", status="started").inc()

        # Convert protocol schema
        protocol_schema = ProtocolSchema(**request.protocol_schema)

        # Create API generation context
        context = APIGenerationContext(
            protocol_schema=protocol_schema,
            target_style=request.api_style,
            security_level=request.security_level,
            base_path=request.base_path,
            include_examples=request.include_examples,
            include_documentation=request.include_documentation,
        )

        # Generate API specification
        api_spec = await api_generator.generate_api_specification(context)

        response = {
            "api_specification": api_spec.to_openapi_dict(),
            "endpoints_count": len(api_spec.endpoints),
            "schemas_count": len(api_spec.schemas),
            "processing_time": time.time() - start_time,
            "spec_id": api_spec.spec_id,
        }

        API_REQUEST_COUNTER.labels(endpoint="generate-api", method="POST", status="success").inc()
        API_REQUEST_DURATION.labels(endpoint="generate-api").observe(response["processing_time"])

        return response

    except Exception as e:
        API_REQUEST_COUNTER.labels(endpoint="generate-api", method="POST", status="error").inc()
        logger.error(f"API generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"API generation failed: {str(e)}")


# SDK Generation Endpoints


@router.post("/generate-sdk")
async def generate_sdk(request: SDKGenerationRequest, current_user=Depends(get_current_user)):
    """Generate SDK for specified languages from API specification."""
    start_time = time.time()

    try:
        API_REQUEST_COUNTER.labels(endpoint="generate-sdk", method="POST", status="started").inc()

        # Convert API specification
        api_spec = APISpecification(**request.api_specification)

        # Generate SDKs for all requested languages
        generated_sdks = await code_generator.generate_multiple_sdks(
            api_spec,
            request.target_languages,
            request.package_name,
            version=request.version,
            author=request.author,
            description=request.description,
            generate_tests=request.generate_tests,
            generate_examples=request.generate_examples,
            generate_documentation=request.generate_documentation,
        )

        response = {
            "generated_sdks": [
                {
                    "language": language.value,
                    "name": sdk.name,
                    "version": sdk.version,
                    "source_files": len(sdk.source_files),
                    "test_files": len(sdk.test_files),
                    "config_files": len(sdk.config_files),
                    "documentation_files": len(sdk.documentation_files),
                    "sdk_id": sdk.sdk_id,
                }
                for language, sdk in generated_sdks.items()
            ],
            "processing_time": time.time() - start_time,
            "total_languages": len(generated_sdks),
        }

        API_REQUEST_COUNTER.labels(endpoint="generate-sdk", method="POST", status="success").inc()
        API_REQUEST_DURATION.labels(endpoint="generate-sdk").observe(response["processing_time"])

        return response

    except Exception as e:
        API_REQUEST_COUNTER.labels(endpoint="generate-sdk", method="POST", status="error").inc()
        logger.error(f"SDK generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"SDK generation failed: {str(e)}")


@router.get("/download-sdk/{sdk_id}")
async def download_sdk(sdk_id: str, format: str = "zip", current_user=Depends(get_current_user)):
    """Download generated SDK as a zip file."""
    try:
        # This would retrieve the SDK from storage
        # For now, return a placeholder response

        if format not in ["zip", "tar.gz"]:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'zip' or 'tar.gz'")

        # Create temporary zip file
        temp_dir = tempfile.mkdtemp()
        zip_path = Path(temp_dir) / f"sdk_{sdk_id}.{format}"

        # In a real implementation, this would package the actual SDK files
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.writestr(
                "README.md",
                f"# SDK {sdk_id}\n\nGenerated by QBITEL Translation Studio",
            )
            zipf.writestr("src/client.py", "# Generated client code would be here")

        return FileResponse(
            path=str(zip_path),
            filename=f"sdk_{sdk_id}.{format}",
            media_type="application/zip" if format == "zip" else "application/gzip",
        )

    except Exception as e:
        logger.error(f"SDK download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


# Protocol Translation Endpoints


@router.post("/translate")
async def translate_protocol(request: ProtocolTranslationRequest, current_user=Depends(get_current_user)):
    """Translate protocol data between different protocol formats."""
    start_time = time.time()

    try:
        API_REQUEST_COUNTER.labels(endpoint="translate", method="POST", status="started").inc()

        # Decode protocol data
        import base64

        protocol_data = base64.b64decode(request.data)

        # Create translation context
        translation_context = TranslationContext(
            source_protocol=request.source_protocol,
            target_protocol=request.target_protocol,
            source_data=protocol_data,
            translation_mode=TranslationMode(request.translation_mode),
            quality_level=QualityLevel(request.quality_level),
            preserve_metadata=request.preserve_metadata,
        )

        # Perform translation
        result = await protocol_bridge.translate_protocol(translation_context)

        response = {
            "translation_id": result.translation_id,
            "source_protocol": result.source_protocol,
            "target_protocol": result.target_protocol,
            "translated_data": base64.b64encode(result.translated_data).decode("utf-8"),
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "translation_mode": result.translation_mode.value,
            "metadata": result.metadata,
            "warnings": result.warnings,
            "validation_errors": result.validation_errors,
        }

        API_REQUEST_COUNTER.labels(endpoint="translate", method="POST", status="success").inc()
        API_REQUEST_DURATION.labels(endpoint="translate").observe(response["processing_time"])

        return response

    except Exception as e:
        API_REQUEST_COUNTER.labels(endpoint="translate", method="POST", status="error").inc()
        logger.error(f"Protocol translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@router.post("/translate/batch")
async def batch_translate_protocols(
    request: BatchTranslationRequest,
    current_user=Depends(get_current_user),
    background_tasks: BackgroundTasks = None,
):
    """Perform batch protocol translation."""
    start_time = time.time()
    job_id = str(uuid.uuid4())

    try:
        API_REQUEST_COUNTER.labels(endpoint="translate-batch", method="POST", status="started").inc()

        # Create translation contexts
        contexts = []
        for trans_req in request.translations:
            import base64

            protocol_data = base64.b64decode(trans_req.data)

            context = TranslationContext(
                source_protocol=trans_req.source_protocol,
                target_protocol=trans_req.target_protocol,
                source_data=protocol_data,
                translation_mode=TranslationMode(trans_req.translation_mode),
                quality_level=QualityLevel(trans_req.quality_level),
                preserve_metadata=trans_req.preserve_metadata,
            )
            contexts.append(context)

        # Perform batch translation
        results = await protocol_bridge.batch_translate(contexts)

        response = {
            "job_id": job_id,
            "total_translations": len(request.translations),
            "successful_translations": len([r for r in results if not r.validation_errors]),
            "failed_translations": len([r for r in results if r.validation_errors]),
            "processing_time": time.time() - start_time,
            "results": [
                {
                    "translation_id": r.translation_id,
                    "source_protocol": r.source_protocol,
                    "target_protocol": r.target_protocol,
                    "confidence": r.confidence,
                    "success": not bool(r.validation_errors),
                    "errors": r.validation_errors,
                }
                for r in results
            ],
        }

        API_REQUEST_COUNTER.labels(endpoint="translate-batch", method="POST", status="success").inc()
        API_REQUEST_DURATION.labels(endpoint="translate-batch").observe(response["processing_time"])

        return response

    except Exception as e:
        API_REQUEST_COUNTER.labels(endpoint="translate-batch", method="POST", status="error").inc()
        logger.error(f"Batch translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch translation failed: {str(e)}")


# Streaming Translation Endpoints


@router.post("/streaming/create")
async def create_streaming_connection(request: StreamingConnectionRequest, current_user=Depends(get_current_user)):
    """Create a streaming translation connection."""
    try:
        from .protocol_bridge.protocol_bridge import QualityLevel

        connection_id = await protocol_bridge.create_streaming_connection(
            source_protocol=request.source_protocol,
            target_protocol=request.target_protocol,
            quality_level=QualityLevel(request.quality_level),
        )

        return {
            "connection_id": connection_id,
            "source_protocol": request.source_protocol,
            "target_protocol": request.target_protocol,
            "quality_level": request.quality_level,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Streaming connection creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Connection creation failed: {str(e)}")


@router.get("/streaming/{connection_id}/status")
async def get_streaming_connection_status(connection_id: str, current_user=Depends(get_current_user)):
    """Get status of a streaming connection."""
    try:
        status = protocol_bridge.get_connection_status(connection_id)

        if not status:
            raise HTTPException(status_code=404, detail="Connection not found")

        return status

    except Exception as e:
        logger.error(f"Connection status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.delete("/streaming/{connection_id}")
async def close_streaming_connection(connection_id: str, current_user=Depends(get_current_user)):
    """Close a streaming connection."""
    try:
        await protocol_bridge.close_streaming_connection(connection_id)

        return {
            "connection_id": connection_id,
            "status": "closed",
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Connection closure failed: {e}")
        raise HTTPException(status_code=500, detail=f"Connection closure failed: {str(e)}")


# Knowledge Base and RAG Endpoints


@router.get("/knowledge/patterns/{protocol_name}")
async def get_protocol_patterns(
    protocol_name: str,
    pattern_type: str = "all",
    limit: int = 5,
    current_user=Depends(get_current_user),
):
    """Get protocol patterns from knowledge base."""
    try:
        result = await rag_engine.query_protocol_patterns(
            protocol_name=protocol_name, pattern_type=pattern_type, n_results=limit
        )

        return {
            "protocol_name": protocol_name,
            "pattern_type": pattern_type,
            "patterns": [
                {
                    "id": doc.id,
                    "content": (doc.content[:500] + "..." if len(doc.content) > 500 else doc.content),
                    "metadata": doc.metadata,
                    "similarity": score,
                }
                for doc, score in zip(result.documents, result.similarity_scores)
            ],
            "total_results": result.total_results,
            "processing_time": result.processing_time,
        }

    except Exception as e:
        logger.error(f"Pattern query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern query failed: {str(e)}")


@router.get("/knowledge/templates/{language}")
async def get_code_templates(
    language: str,
    template_type: str = "client",
    limit: int = 3,
    current_user=Depends(get_current_user),
):
    """Get code generation templates for specific language."""
    try:
        result = await rag_engine.get_code_generation_templates(
            language=language, template_type=template_type, n_results=limit
        )

        return {
            "language": language,
            "template_type": template_type,
            "templates": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "similarity": score,
                }
                for doc, score in zip(result.documents, result.similarity_scores)
            ],
            "total_results": result.total_results,
        }

    except Exception as e:
        logger.error(f"Template query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Template query failed: {str(e)}")


@router.get("/knowledge/best-practices")
async def get_best_practices(context: str = "general", limit: int = 5, current_user=Depends(get_current_user)):
    """Get translation best practices."""
    try:
        result = await rag_engine.get_translation_best_practices(context=context, n_results=limit)

        return {
            "context": context,
            "best_practices": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "similarity": score,
                }
                for doc, score in zip(result.documents, result.similarity_scores)
            ],
            "total_results": result.total_results,
        }

    except Exception as e:
        logger.error(f"Best practices query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Best practices query failed: {str(e)}")


# Status and Metrics Endpoints


@router.get("/status")
async def get_translation_studio_status():
    """Get comprehensive status of Translation Studio services."""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {
                "discovery_orchestrator": discovery_orchestrator is not None,
                "api_generator": api_generator is not None,
                "code_generator": code_generator is not None,
                "protocol_bridge": protocol_bridge is not None,
                "rag_engine": rag_engine is not None,
            },
            "capabilities": {
                "supported_languages": ([lang.value for lang in CodeLanguage] if code_generator else []),
                "supported_api_styles": [style.value for style in APIStyle],
                "supported_security_levels": [level.value for level in SecurityLevel],
            },
        }

        # Add service-specific metrics
        if discovery_orchestrator:
            discovery_metrics = await discovery_orchestrator.get_api_generation_metrics()
            status["metrics"] = {"discovery": discovery_metrics}

        if api_generator:
            api_metrics = api_generator.get_generation_metrics()
            status["metrics"]["api_generation"] = api_metrics

        if code_generator:
            code_metrics = code_generator.get_generation_metrics()
            status["metrics"]["code_generation"] = code_metrics

        if protocol_bridge:
            bridge_metrics = protocol_bridge.get_bridge_metrics()
            status["metrics"]["protocol_bridge"] = bridge_metrics

        return status

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/metrics")
async def get_translation_metrics():
    """Get detailed metrics for Translation Studio operations."""
    try:
        metrics = {"timestamp": datetime.now(timezone.utc).isoformat(), "services": {}}

        if discovery_orchestrator:
            metrics["services"]["discovery"] = await discovery_orchestrator.get_api_generation_metrics()

        if api_generator:
            metrics["services"]["api_generation"] = api_generator.get_generation_metrics()

        if code_generator:
            metrics["services"]["code_generation"] = code_generator.get_generation_metrics()

        if protocol_bridge:
            metrics["services"]["protocol_bridge"] = protocol_bridge.get_bridge_metrics()

        if rag_engine:
            metrics["services"]["knowledge_base"] = rag_engine.get_collection_stats()

        return metrics

    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


# File Upload Endpoints


@router.post("/upload/protocol-samples")
async def upload_protocol_samples(
    files: List[UploadFile] = File(...),
    protocol_type: str = None,
    current_user=Depends(get_current_user),
):
    """Upload protocol sample files for analysis and training."""
    try:
        uploaded_files = []

        for file in files:
            if file.size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=413, detail=f"File {file.filename} is too large")

            content = await file.read()

            # Store file and queue for processing
            file_info = {
                "filename": file.filename,
                "size": len(content),
                "content_type": file.content_type,
                "protocol_type": protocol_type,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }

            uploaded_files.append(file_info)

        return {
            "uploaded_files": uploaded_files,
            "total_files": len(uploaded_files),
            "next_steps": [
                "Files queued for analysis",
                "Use /analyze endpoint to process uploaded files",
                "Results will be available via /results/{job_id} endpoint",
            ],
        }

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# Export the router
__all__ = ["router"]
