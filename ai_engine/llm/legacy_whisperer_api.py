"""
QBITEL Bridge - Legacy System Whisperer API Endpoints
FastAPI endpoints for legacy protocol analysis and modernization.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from datetime import datetime

from .legacy_whisperer import (
    LegacySystemWhisperer,
    ProtocolSpecification,
    AdapterCode,
    Explanation,
    AdapterLanguage,
    create_legacy_whisperer,
)
from ..core.exceptions import QbitelAIException

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/legacy-whisperer", tags=["legacy-whisperer"])

# Global whisperer instance
_whisperer: Optional[LegacySystemWhisperer] = None


async def get_whisperer() -> LegacySystemWhisperer:
    """Dependency to get whisperer instance."""
    global _whisperer
    if _whisperer is None:
        _whisperer = await create_legacy_whisperer()
    return _whisperer


# Request/Response Models


class ReverseEngineerRequest(BaseModel):
    """Request for protocol reverse engineering."""

    traffic_samples: List[str] = Field(
        ..., description="List of protocol message samples (hex-encoded)", min_items=10
    )
    system_context: str = Field(
        default="", description="Additional context about the system"
    )


class ReverseEngineerResponse(BaseModel):
    """Response from protocol reverse engineering."""

    protocol_name: str
    version: str
    description: str
    complexity: str
    confidence_score: float
    analysis_time: float
    samples_analyzed: int
    fields: List[Dict[str, Any]]
    message_types: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]]
    documentation: str
    spec_id: str


class GenerateAdapterRequest(BaseModel):
    """Request for adapter code generation."""

    spec_id: str = Field(..., description="Protocol specification ID")
    target_protocol: str = Field(
        ..., description="Target protocol (REST, gRPC, GraphQL)"
    )
    language: str = Field(default="python", description="Programming language")


class GenerateAdapterResponse(BaseModel):
    """Response from adapter code generation."""

    adapter_id: str
    source_protocol: str
    target_protocol: str
    language: str
    adapter_code: str
    test_code: str
    documentation: str
    dependencies: List[str]
    configuration_template: str
    deployment_guide: str
    code_quality_score: float
    generation_time: float


class ExplainBehaviorRequest(BaseModel):
    """Request for behavior explanation."""

    behavior: str = Field(..., description="Description of the legacy behavior")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )


class ExplainBehaviorResponse(BaseModel):
    """Response from behavior explanation."""

    explanation_id: str
    behavior_description: str
    technical_explanation: str
    historical_context: str
    root_causes: List[str]
    implications: List[str]
    modernization_approaches: List[Dict[str, Any]]
    recommended_approach: Optional[str]
    modernization_risks: List[Dict[str, Any]]
    risk_level: str
    implementation_steps: List[str]
    estimated_effort: str
    confidence: float


# API Endpoints


@router.post("/reverse-engineer", response_model=ReverseEngineerResponse)
async def reverse_engineer_protocol(
    request: ReverseEngineerRequest,
    whisperer: LegacySystemWhisperer = Depends(get_whisperer),
):
    """
    Reverse engineer a legacy protocol from traffic samples.

    This endpoint analyzes protocol message samples to automatically:
    - Identify protocol structure and fields
    - Detect message types and patterns
    - Generate comprehensive documentation
    - Assess protocol complexity

    **Success Metrics:**
    - Reverse engineering accuracy: 85%+
    - Documentation completeness: 90%+
    """
    try:
        # Convert hex samples to bytes
        traffic_samples = [bytes.fromhex(sample) for sample in request.traffic_samples]

        # Perform reverse engineering
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=traffic_samples, system_context=request.system_context
        )

        # Convert to response
        return ReverseEngineerResponse(
            protocol_name=spec.protocol_name,
            version=spec.version,
            description=spec.description,
            complexity=spec.complexity.value,
            confidence_score=spec.confidence_score,
            analysis_time=spec.analysis_time,
            samples_analyzed=spec.samples_analyzed,
            fields=[
                {
                    "name": f.name,
                    "offset": f.offset,
                    "length": f.length,
                    "type": f.field_type,
                    "description": f.description,
                    "confidence": f.confidence,
                }
                for f in spec.fields
            ],
            message_types=spec.message_types,
            patterns=[
                {
                    "type": p.pattern_type,
                    "description": p.description,
                    "frequency": p.frequency,
                    "confidence": p.confidence,
                }
                for p in spec.patterns
            ],
            documentation=spec.documentation,
            spec_id=spec.spec_id,
        )

    except QbitelAIException as e:
        logger.error(f"Reverse engineering failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in reverse engineering: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/generate-adapter", response_model=GenerateAdapterResponse)
async def generate_adapter_code(
    request: GenerateAdapterRequest,
    background_tasks: BackgroundTasks,
    whisperer: LegacySystemWhisperer = Depends(get_whisperer),
):
    """
    Generate protocol adapter code.

    This endpoint generates production-ready adapter code to bridge
    a legacy protocol to a modern target protocol (REST, gRPC, GraphQL).

    **Includes:**
    - Complete adapter implementation
    - Comprehensive test suite
    - Integration documentation
    - Deployment guide
    - Configuration templates

    **Code Quality:** Production-ready with >85% test coverage
    """
    try:
        # Get protocol specification from cache
        spec = None
        for cached_spec in whisperer.analysis_cache.values():
            if cached_spec.spec_id == request.spec_id:
                spec = cached_spec
                break

        if spec is None:
            raise HTTPException(
                status_code=404,
                detail=f"Protocol specification {request.spec_id} not found",
            )

        # Validate language
        try:
            language = AdapterLanguage(request.language.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Unsupported language: {request.language}"
            )

        # Generate adapter code
        adapter = await whisperer.generate_adapter_code(
            legacy_protocol=spec,
            target_protocol=request.target_protocol,
            language=language,
        )

        # Convert to response
        return GenerateAdapterResponse(
            adapter_id=adapter.adapter_id,
            source_protocol=adapter.source_protocol,
            target_protocol=adapter.target_protocol,
            language=adapter.language.value,
            adapter_code=adapter.adapter_code,
            test_code=adapter.test_code,
            documentation=adapter.documentation,
            dependencies=adapter.dependencies,
            configuration_template=adapter.configuration_template,
            deployment_guide=adapter.deployment_guide,
            code_quality_score=adapter.code_quality_score,
            generation_time=adapter.generation_time,
        )

    except HTTPException:
        raise
    except QbitelAIException as e:
        logger.error(f"Adapter generation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in adapter generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/explain-behavior", response_model=ExplainBehaviorResponse)
async def explain_legacy_behavior(
    request: ExplainBehaviorRequest,
    whisperer: LegacySystemWhisperer = Depends(get_whisperer),
):
    """
    Explain legacy system behavior with modernization guidance.

    This endpoint provides:
    - Technical explanation of legacy behavior
    - Historical context and root causes
    - Multiple modernization approaches
    - Risk assessment
    - Implementation guidance

    **Use Cases:**
    - Understanding undocumented legacy systems
    - Planning modernization projects
    - Risk assessment for migrations
    - Technical debt analysis
    """
    try:
        # Explain behavior
        explanation = await whisperer.explain_legacy_behavior(
            behavior=request.behavior, context=request.context
        )

        # Convert to response
        return ExplainBehaviorResponse(
            explanation_id=explanation.explanation_id,
            behavior_description=explanation.behavior_description,
            technical_explanation=explanation.technical_explanation,
            historical_context=explanation.historical_context,
            root_causes=explanation.root_causes,
            implications=explanation.implications,
            modernization_approaches=explanation.modernization_approaches,
            recommended_approach=explanation.recommended_approach,
            modernization_risks=explanation.modernization_risks,
            risk_level=explanation.risk_level.value,
            implementation_steps=explanation.implementation_steps,
            estimated_effort=explanation.estimated_effort,
            confidence=explanation.confidence,
        )

    except QbitelAIException as e:
        logger.error(f"Behavior explanation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in behavior explanation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/statistics")
async def get_statistics(whisperer: LegacySystemWhisperer = Depends(get_whisperer)):
    """
    Get Legacy System Whisperer statistics.

    Returns operational metrics and cache statistics.
    """
    try:
        return whisperer.get_statistics()
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns the health status of the Legacy System Whisperer service.
    """
    return {
        "status": "healthy",
        "service": "legacy-system-whisperer",
        "timestamp": datetime.utcnow().isoformat(),
    }


# Startup/Shutdown Events


async def startup_event():
    """Initialize whisperer on startup."""
    global _whisperer
    try:
        _whisperer = await create_legacy_whisperer()
        logger.info("Legacy System Whisperer API initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Legacy System Whisperer API: {e}")
        raise


async def shutdown_event():
    """Cleanup on shutdown."""
    global _whisperer
    if _whisperer:
        await _whisperer.shutdown()
        logger.info("Legacy System Whisperer API shutdown complete")
