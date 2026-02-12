"""
QBITEL - Protocol Translation Studio API Endpoints
FastAPI endpoints for protocol translation services.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import base64

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..llm.translation_studio import (
    ProtocolTranslationStudio,
    get_translation_studio,
    ProtocolSpecification,
    ProtocolField,
    ProtocolType,
    FieldType,
    TranslationRules,
    PerformanceData,
)
from ..core.config import get_config
from ..core.exceptions import TranslationException

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/translation", tags=["translation"])


# Request/Response Models
class TranslateRequest(BaseModel):
    """Protocol translation request."""

    source_protocol: str = Field(..., description="Source protocol identifier")
    target_protocol: str = Field(..., description="Target protocol identifier")
    message: str = Field(..., description="Base64-encoded message to translate")
    validate: bool = Field(default=True, description="Validate translation result")


class TranslateResponse(BaseModel):
    """Protocol translation response."""

    translation_id: str
    source_protocol: str
    target_protocol: str
    translated_message: str  # Base64-encoded
    success: bool
    latency_ms: float
    validation_passed: bool
    errors: List[str] = []
    warnings: List[str] = []
    timestamp: str


class ProtocolFieldModel(BaseModel):
    """Protocol field model."""

    name: str
    field_type: str
    offset: Optional[int] = None
    length: Optional[int] = None
    required: bool = True
    default_value: Any = None
    description: str = ""


class ProtocolSpecModel(BaseModel):
    """Protocol specification model."""

    protocol_id: str
    protocol_type: str
    version: str
    name: str
    description: str
    fields: List[ProtocolFieldModel]
    encoding: str = "utf-8"
    byte_order: str = "big"


class GenerateRulesRequest(BaseModel):
    """Generate translation rules request."""

    source_protocol_id: str
    target_protocol_id: str


class GenerateRulesResponse(BaseModel):
    """Generate translation rules response."""

    rules_id: str
    source_protocol: str
    target_protocol: str
    rules_count: int
    accuracy: float
    created_at: str


class OptimizeRequest(BaseModel):
    """Optimize translation request."""

    rules_id: str
    performance_data: Dict[str, Any]


class OptimizeResponse(BaseModel):
    """Optimize translation response."""

    original_rules_id: str
    optimized_rules_id: str
    optimizations_applied: List[str]
    performance_improvement: float
    accuracy_improvement: float


class StatisticsResponse(BaseModel):
    """Translation studio statistics."""

    total_translations: int
    successful_translations: int
    failed_translations: int
    rules_generated: int
    optimizations_applied: int
    average_accuracy: float
    average_latency_ms: float
    cached_rules: int
    registered_protocols: int
    timestamp: str


# Dependency to get translation studio
async def get_studio() -> ProtocolTranslationStudio:
    """Get translation studio instance."""
    studio = get_translation_studio()
    if not studio:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Translation studio not initialized",
        )
    return studio


# Endpoints
@router.post("/translate", response_model=TranslateResponse)
async def translate_protocol(request: TranslateRequest, studio: ProtocolTranslationStudio = Depends(get_studio)):
    """
    Translate message from source protocol to target protocol.

    This endpoint performs real-time protocol translation with:
    - Automatic protocol detection and parsing
    - Rule-based translation
    - Validation of translated message
    - Performance metrics tracking
    """
    import time

    start_time = time.time()

    try:
        logger.info(f"Translation request: {request.source_protocol} -> {request.target_protocol}")

        # Decode message
        try:
            message_bytes = base64.b64decode(request.message)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64-encoded message: {e}",
            )

        # Perform translation
        translated_bytes = await studio.translate_protocol(request.source_protocol, request.target_protocol, message_bytes)

        # Calculate actual latency
        actual_latency_ms = (time.time() - start_time) * 1000

        # Perform validation if requested
        validation_passed = True
        if request.validate:
            # Get protocol specs for validation
            source_spec = await studio._get_protocol_spec(request.source_protocol)
            target_spec = await studio._get_protocol_spec(request.target_protocol)
            rules = await studio._get_translation_rules(source_spec, target_spec)

            validation_result = await studio._validate_translation(
                message_bytes, translated_bytes, source_spec, target_spec, rules
            )
            validation_passed = validation_result["passed"]

        # Encode result
        translated_b64 = base64.b64encode(translated_bytes).decode("utf-8")

        return TranslateResponse(
            translation_id=f"trans_{datetime.utcnow().timestamp()}",
            source_protocol=request.source_protocol,
            target_protocol=request.target_protocol,
            translated_message=translated_b64,
            success=True,
            latency_ms=actual_latency_ms,
            validation_passed=validation_passed,
            timestamp=datetime.utcnow().isoformat(),
        )

    except TranslationException as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}",
        )


@router.post("/rules/generate", response_model=GenerateRulesResponse)
async def generate_translation_rules(
    request: GenerateRulesRequest,
    background_tasks: BackgroundTasks,
    studio: ProtocolTranslationStudio = Depends(get_studio),
):
    """
    Generate translation rules between two protocols using LLM.

    This endpoint:
    - Analyzes source and target protocol specifications
    - Uses LLM to generate intelligent translation rules
    - Creates test cases for validation
    - Caches rules for future use
    """
    try:
        logger.info(f"Generating rules: {request.source_protocol_id} -> {request.target_protocol_id}")

        # Get protocol specifications
        source_spec = await studio._get_protocol_spec(request.source_protocol_id)
        target_spec = await studio._get_protocol_spec(request.target_protocol_id)

        # Generate rules
        rules = await studio.generate_translation_rules(source_spec, target_spec)

        return GenerateRulesResponse(
            rules_id=rules.rules_id,
            source_protocol=source_spec.protocol_id,
            target_protocol=target_spec.protocol_id,
            rules_count=len(rules.rules),
            accuracy=rules.accuracy,
            created_at=rules.created_at.isoformat(),
        )

    except Exception as e:
        logger.error(f"Rule generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate rules: {str(e)}",
        )


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_translation(request: OptimizeRequest, studio: ProtocolTranslationStudio = Depends(get_studio)):
    """
    Optimize translation rules for better performance.

    This endpoint:
    - Analyzes current performance metrics
    - Identifies bottlenecks
    - Uses LLM to suggest optimizations
    - Applies and benchmarks improvements
    """
    try:
        logger.info(f"Optimizing rules: {request.rules_id}")

        # Get rules from cache
        rules = None
        for cached_rules in studio.translation_rules_cache.values():
            if cached_rules.rules_id == request.rules_id:
                rules = cached_rules
                break

        if not rules:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rules not found: {request.rules_id}",
            )

        # Create performance data
        perf_data = PerformanceData(
            total_translations=request.performance_data.get("total_translations", 1000),
            successful_translations=request.performance_data.get("successful_translations", 950),
            failed_translations=request.performance_data.get("failed_translations", 50),
            average_latency_ms=request.performance_data.get("average_latency_ms", 2.0),
            p50_latency_ms=request.performance_data.get("p50_latency_ms", 1.5),
            p95_latency_ms=request.performance_data.get("p95_latency_ms", 3.0),
            p99_latency_ms=request.performance_data.get("p99_latency_ms", 5.0),
            throughput_per_second=request.performance_data.get("throughput_per_second", 50000),
            error_rate=request.performance_data.get("error_rate", 0.05),
            accuracy=request.performance_data.get("accuracy", 0.95),
        )

        # Optimize
        optimized = await studio.optimize_translation(rules, perf_data)

        return OptimizeResponse(
            original_rules_id=rules.rules_id,
            optimized_rules_id=optimized.optimized_rules.rules_id,
            optimizations_applied=optimized.optimizations_applied,
            performance_improvement=optimized.performance_improvement,
            accuracy_improvement=optimized.accuracy_improvement,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize: {str(e)}",
        )


@router.post("/protocols/register", status_code=status.HTTP_201_CREATED)
async def register_protocol(spec: ProtocolSpecModel, studio: ProtocolTranslationStudio = Depends(get_studio)):
    """
    Register a new protocol specification.

    This endpoint allows registering custom protocol specifications
    for translation support.
    """
    try:
        logger.info(f"Registering protocol: {spec.protocol_id}")

        # Convert to ProtocolSpecification
        fields = [
            ProtocolField(
                name=f.name,
                field_type=FieldType(f.field_type),
                offset=f.offset,
                length=f.length,
                required=f.required,
                default_value=f.default_value,
                description=f.description,
            )
            for f in spec.fields
        ]

        protocol_spec = ProtocolSpecification(
            protocol_id=spec.protocol_id,
            protocol_type=ProtocolType(spec.protocol_type),
            version=spec.version,
            name=spec.name,
            description=spec.description,
            fields=fields,
            encoding=spec.encoding,
            byte_order=spec.byte_order,
        )

        await studio.register_protocol(protocol_spec)

        return {
            "message": f"Protocol registered: {spec.protocol_id}",
            "protocol_id": spec.protocol_id,
        }

    except Exception as e:
        logger.error(f"Protocol registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register protocol: {str(e)}",
        )


@router.get("/protocols", response_model=List[str])
async def list_protocols(studio: ProtocolTranslationStudio = Depends(get_studio)):
    """
    List all registered protocol specifications.
    """
    return list(studio.protocol_specs.keys())


@router.get("/protocols/{protocol_id}", response_model=ProtocolSpecModel)
async def get_protocol(protocol_id: str, studio: ProtocolTranslationStudio = Depends(get_studio)):
    """
    Get protocol specification details.
    """
    try:
        spec = await studio._get_protocol_spec(protocol_id)

        return ProtocolSpecModel(
            protocol_id=spec.protocol_id,
            protocol_type=spec.protocol_type.value,
            version=spec.version,
            name=spec.name,
            description=spec.description,
            fields=[
                ProtocolFieldModel(
                    name=f.name,
                    field_type=f.field_type.value,
                    offset=f.offset,
                    length=f.length,
                    required=f.required,
                    default_value=f.default_value,
                    description=f.description,
                )
                for f in spec.fields
            ],
            encoding=spec.encoding,
            byte_order=spec.byte_order,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Protocol not found: {protocol_id}",
        )


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(studio: ProtocolTranslationStudio = Depends(get_studio)):
    """
    Get translation studio statistics and metrics.
    """
    stats = studio.get_statistics()

    return StatisticsResponse(
        total_translations=stats["total_translations"],
        successful_translations=stats["successful_translations"],
        failed_translations=stats["failed_translations"],
        rules_generated=stats["rules_generated"],
        optimizations_applied=stats["optimizations_applied"],
        average_accuracy=stats["average_accuracy"],
        average_latency_ms=stats["average_latency_ms"],
        cached_rules=stats["cached_rules"],
        registered_protocols=stats["registered_protocols"],
        timestamp=stats["timestamp"],
    )


@router.get("/health")
async def health_check(studio: ProtocolTranslationStudio = Depends(get_studio)):
    """
    Health check endpoint for translation studio.
    """
    return {
        "status": "healthy",
        "service": "protocol_translation_studio",
        "version": "1.0.0",
        "registered_protocols": len(studio.protocol_specs),
        "cached_rules": len(studio.translation_rules_cache),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.delete("/cache/clear")
async def clear_cache(studio: ProtocolTranslationStudio = Depends(get_studio)):
    """
    Clear translation rules cache.
    """
    cache_size = len(studio.translation_rules_cache)
    studio.translation_rules_cache.clear()

    return {
        "message": "Cache cleared",
        "rules_cleared": cache_size,
        "timestamp": datetime.utcnow().isoformat(),
    }
