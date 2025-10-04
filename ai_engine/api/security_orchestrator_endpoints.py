"""
CRONOS AI - Zero-Touch Security Orchestrator API Endpoints
RESTful API for security automation and threat management.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from ..core.config import Config, get_config
from ..core.exceptions import SecurityException
from ..llm.security_orchestrator import (
    ZeroTouchSecurityOrchestrator,
    get_security_orchestrator,
    SecurityEvent,
    SecurityRequirements,
    ThreatData,
    ThreatType,
    ThreatSeverity,
    ResponseAction,
    IncidentStatus,
)
from ..llm.unified_llm_service import get_llm_service
from ..monitoring.alerts import get_alert_manager
from ..policy.policy_engine import get_policy_engine

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(
    prefix="/api/v1/security",
    tags=["security-orchestrator"],
    responses={404: {"description": "Not found"}},
)


# Pydantic models for API requests/responses
class SecurityEventRequest(BaseModel):
    """Security event submission request."""

    event_type: str = Field(..., description="Type of security event")
    severity: str = Field(..., description="Event severity level")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    destination_ip: Optional[str] = Field(None, description="Destination IP address")
    user_id: Optional[str] = Field(None, description="User ID associated with event")
    resource: Optional[str] = Field(None, description="Affected resource")
    description: str = Field(..., description="Event description")
    indicators: List[str] = Field(default_factory=list, description="Threat indicators")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw event data")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SecurityPolicyRequest(BaseModel):
    """Security policy generation request."""

    framework: str = Field(
        ..., description="Security framework (NIST, ISO27001, CIS, etc.)"
    )
    controls: List[str] = Field(..., description="Security controls to implement")
    risk_level: str = Field(..., description="Risk level (low, medium, high, critical)")
    compliance_requirements: List[str] = Field(
        ..., description="Compliance requirements"
    )
    business_context: Dict[str, Any] = Field(
        default_factory=dict, description="Business context"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict, description="Implementation constraints"
    )


class ThreatIntelligenceRequest(BaseModel):
    """Threat intelligence analysis request."""

    source: str = Field(..., description="Intelligence source")
    threat_indicators: List[str] = Field(..., description="Threat indicators")
    threat_actors: List[str] = Field(
        default_factory=list, description="Known threat actors"
    )
    attack_patterns: List[str] = Field(
        default_factory=list, description="Attack patterns"
    )
    vulnerabilities: List[str] = Field(
        default_factory=list, description="Vulnerabilities"
    )
    confidence: float = Field(0.7, ge=0.0, le=1.0, description="Confidence level")
    raw_data: Dict[str, Any] = Field(
        default_factory=dict, description="Raw intelligence data"
    )


class SecurityResponseModel(BaseModel):
    """Security response model."""

    response_id: str
    event_id: str
    actions_taken: List[str]
    success: bool
    execution_time: float
    details: str
    blocked_ips: List[str]
    isolated_systems: List[str]
    alerts_generated: List[str]
    timestamp: str
    metadata: Dict[str, Any]


class ThreatAnalysisModel(BaseModel):
    """Threat analysis model."""

    threat_id: str
    threat_type: str
    severity: str
    confidence: float
    risk_score: float
    attack_vector: str
    affected_assets: List[str]
    indicators_of_compromise: List[str]
    analysis_summary: str
    recommended_actions: List[str]
    mitigation_strategies: List[str]
    timestamp: str
    metadata: Dict[str, Any]


class SecurityPolicyModel(BaseModel):
    """Security policy model."""

    policy_id: str
    name: str
    description: str
    policy_type: str
    rules: List[Dict[str, Any]]
    enforcement_level: str
    scope: List[str]
    created_at: str
    metadata: Dict[str, Any]


# Dependency to get security orchestrator
async def get_orchestrator() -> ZeroTouchSecurityOrchestrator:
    """Get security orchestrator instance."""
    orchestrator = get_security_orchestrator()
    if not orchestrator:
        # Orchestrator must be initialized during startup
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Security orchestrator not initialized. Service is starting up or failed to initialize.",
        )

    return orchestrator


@router.post(
    "/events/detect-and-respond",
    response_model=SecurityResponseModel,
    status_code=status.HTTP_200_OK,
    summary="Detect and respond to security event",
    description="Submit a security event for automated threat detection and response",
)
async def detect_and_respond(
    event_request: SecurityEventRequest,
    background_tasks: BackgroundTasks,
    orchestrator: ZeroTouchSecurityOrchestrator = Depends(get_orchestrator),
):
    """
    Detect threats and execute automated response for a security event.

    This endpoint:
    - Analyzes the security event using LLM
    - Determines threat level and required actions
    - Executes automated response (block, isolate, alert, etc.)
    - Creates incident record
    - Generates alerts for high-severity threats
    """
    try:
        # Create SecurityEvent object
        security_event = SecurityEvent(
            event_id=f"EVT-{datetime.utcnow().timestamp()}",
            event_type=ThreatType(event_request.event_type),
            severity=ThreatSeverity(event_request.severity),
            timestamp=datetime.utcnow(),
            source_ip=event_request.source_ip,
            destination_ip=event_request.destination_ip,
            user_id=event_request.user_id,
            resource=event_request.resource,
            description=event_request.description,
            raw_data=event_request.raw_data,
            indicators=event_request.indicators,
            metadata=event_request.metadata,
        )

        # Process event
        response = await orchestrator.detect_and_respond(security_event)

        # Convert to response model
        return SecurityResponseModel(
            response_id=response.response_id,
            event_id=response.event_id,
            actions_taken=[action.value for action in response.actions_taken],
            success=response.success,
            execution_time=response.execution_time,
            details=response.details,
            blocked_ips=response.blocked_ips,
            isolated_systems=response.isolated_systems,
            alerts_generated=response.alerts_generated,
            timestamp=response.timestamp.isoformat(),
            metadata=response.metadata,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )
    except SecurityException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security operation failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in detect_and_respond: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/policies/generate",
    response_model=List[SecurityPolicyModel],
    status_code=status.HTTP_201_CREATED,
    summary="Generate security policies",
    description="Generate security policies based on framework and requirements",
)
async def generate_security_policies(
    policy_request: SecurityPolicyRequest,
    orchestrator: ZeroTouchSecurityOrchestrator = Depends(get_orchestrator),
):
    """
    Generate security policies automatically based on requirements.

    This endpoint:
    - Analyzes security requirements
    - Generates policies aligned with specified framework
    - Validates policies against best practices
    - Creates implementation guide
    """
    try:
        # Create SecurityRequirements object
        requirements = SecurityRequirements(
            requirement_id=f"REQ-{datetime.utcnow().timestamp()}",
            framework=policy_request.framework,
            controls=policy_request.controls,
            risk_level=policy_request.risk_level,
            compliance_requirements=policy_request.compliance_requirements,
            business_context=policy_request.business_context,
            constraints=policy_request.constraints,
        )

        # Generate policies
        policies = await orchestrator.generate_security_policies(requirements)

        # Convert to response models
        return [
            SecurityPolicyModel(
                policy_id=policy.policy_id,
                name=policy.name,
                description=policy.description,
                policy_type=policy.policy_type,
                rules=policy.rules,
                enforcement_level=policy.enforcement_level,
                scope=policy.scope,
                created_at=policy.created_at.isoformat(),
                metadata=policy.metadata,
            )
            for policy in policies
        ]

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )
    except SecurityException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Policy generation failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in generate_security_policies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/threat-intelligence/analyze",
    response_model=ThreatAnalysisModel,
    status_code=status.HTTP_200_OK,
    summary="Analyze threat intelligence",
    description="Analyze threat intelligence data and provide actionable insights",
)
async def analyze_threat_intelligence(
    intel_request: ThreatIntelligenceRequest,
    orchestrator: ZeroTouchSecurityOrchestrator = Depends(get_orchestrator),
):
    """
    Analyze threat intelligence data.

    This endpoint:
    - Correlates threat indicators
    - Assesses impact on infrastructure
    - Generates detection strategies
    - Provides mitigation recommendations
    """
    try:
        # Create ThreatData object
        threat_data = ThreatData(
            data_id=f"TI-{datetime.utcnow().timestamp()}",
            source=intel_request.source,
            threat_indicators=intel_request.threat_indicators,
            threat_actors=intel_request.threat_actors,
            attack_patterns=intel_request.attack_patterns,
            vulnerabilities=intel_request.vulnerabilities,
            timestamp=datetime.utcnow(),
            confidence=intel_request.confidence,
            raw_data=intel_request.raw_data,
        )

        # Analyze threat intelligence
        analysis = await orchestrator.threat_intelligence_analysis(threat_data)

        # Convert to response model
        return ThreatAnalysisModel(
            threat_id=analysis.threat_id,
            threat_type=analysis.threat_type.value,
            severity=analysis.severity.value,
            confidence=analysis.confidence,
            risk_score=analysis.risk_score,
            attack_vector=analysis.attack_vector,
            affected_assets=analysis.affected_assets,
            indicators_of_compromise=analysis.indicators_of_compromise,
            analysis_summary=analysis.analysis_summary,
            recommended_actions=analysis.recommended_actions,
            mitigation_strategies=analysis.mitigation_strategies,
            timestamp=analysis.timestamp.isoformat(),
            metadata=analysis.metadata,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )
    except SecurityException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Threat analysis failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in analyze_threat_intelligence: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/posture/assess",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Assess security posture",
    description="Get comprehensive security posture assessment",
)
async def assess_security_posture(
    orchestrator: ZeroTouchSecurityOrchestrator = Depends(get_orchestrator),
):
    """
    Assess overall security posture.

    Returns:
    - Overall security status
    - Active threats by severity
    - Detection accuracy metrics
    - Recent incidents
    - Security recommendations
    """
    try:
        assessment = await orchestrator.assess_security_posture()
        return assessment

    except SecurityException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security posture assessment failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in assess_security_posture: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/incidents",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="List security incidents",
    description="Get list of security incidents with optional filtering",
)
async def list_incidents(
    status_filter: Optional[str] = None,
    severity_filter: Optional[str] = None,
    limit: int = 50,
    orchestrator: ZeroTouchSecurityOrchestrator = Depends(get_orchestrator),
):
    """
    List security incidents with optional filtering.

    Query Parameters:
    - status_filter: Filter by incident status
    - severity_filter: Filter by severity level
    - limit: Maximum number of incidents to return
    """
    try:
        incidents = list(orchestrator.active_incidents.values())

        # Apply filters
        if status_filter:
            incidents = [i for i in incidents if i.status.value == status_filter]

        if severity_filter:
            incidents = [i for i in incidents if i.severity.value == severity_filter]

        # Sort by created_at (newest first) and limit
        incidents.sort(key=lambda x: x.created_at, reverse=True)
        incidents = incidents[:limit]

        # Convert to dict
        return {
            "total": len(incidents),
            "incidents": [
                {
                    "incident_id": i.incident_id,
                    "title": i.title,
                    "description": i.description,
                    "severity": i.severity.value,
                    "status": i.status.value,
                    "created_at": i.created_at.isoformat(),
                    "updated_at": i.updated_at.isoformat(),
                    "resolved_at": i.resolved_at.isoformat() if i.resolved_at else None,
                    "event_count": len(i.events),
                    "assigned_to": i.assigned_to,
                }
                for i in incidents
            ],
        }

    except Exception as e:
        logger.error(f"Unexpected error in list_incidents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/incidents/{incident_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get incident details",
    description="Get detailed information about a specific security incident",
)
async def get_incident(
    incident_id: str,
    orchestrator: ZeroTouchSecurityOrchestrator = Depends(get_orchestrator),
):
    """Get detailed information about a specific security incident."""
    try:
        if incident_id not in orchestrator.active_incidents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id} not found",
            )

        incident = orchestrator.active_incidents[incident_id]

        return {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "description": incident.description,
            "severity": incident.severity.value,
            "status": incident.status.value,
            "created_at": incident.created_at.isoformat(),
            "updated_at": incident.updated_at.isoformat(),
            "resolved_at": (
                incident.resolved_at.isoformat() if incident.resolved_at else None
            ),
            "assigned_to": incident.assigned_to,
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "severity": e.severity.value,
                    "timestamp": e.timestamp.isoformat(),
                    "source_ip": e.source_ip,
                    "destination_ip": e.destination_ip,
                    "description": e.description,
                }
                for e in incident.events
            ],
            "analysis": (
                {
                    "threat_id": incident.analysis.threat_id,
                    "threat_type": incident.analysis.threat_type.value,
                    "confidence": incident.analysis.confidence,
                    "risk_score": incident.analysis.risk_score,
                    "attack_vector": incident.analysis.attack_vector,
                    "analysis_summary": incident.analysis.analysis_summary,
                    "recommended_actions": incident.analysis.recommended_actions,
                }
                if incident.analysis
                else None
            ),
            "response": (
                {
                    "response_id": incident.response.response_id,
                    "actions_taken": [a.value for a in incident.response.actions_taken],
                    "success": incident.response.success,
                    "details": incident.response.details,
                    "blocked_ips": incident.response.blocked_ips,
                    "isolated_systems": incident.response.isolated_systems,
                }
                if incident.response
                else None
            ),
            "metadata": incident.metadata,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_incident: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/statistics",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get security statistics",
    description="Get comprehensive security orchestrator statistics",
)
async def get_statistics(
    orchestrator: ZeroTouchSecurityOrchestrator = Depends(get_orchestrator),
):
    """Get comprehensive security orchestrator statistics."""
    try:
        stats = orchestrator.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"Unexpected error in get_statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/health",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check security orchestrator health status",
)
async def health_check(
    orchestrator: ZeroTouchSecurityOrchestrator = Depends(get_orchestrator),
):
    """Health check endpoint for security orchestrator."""
    try:
        return {
            "status": "healthy",
            "service": "zero-touch-security-orchestrator",
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": orchestrator.get_statistics(),
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "zero-touch-security-orchestrator",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
