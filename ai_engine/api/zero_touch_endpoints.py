"""
QBITEL Engine - Zero-Touch Security Decision Engine API Endpoints

REST API endpoints for the autonomous security decision engine.
Provides interfaces for:
- Real-time threat analysis and response
- Decision audit and review
- Manual approval workflows
- Configuration management
- Metrics and reporting
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge

from ..api.auth import get_current_user
from ..security.decision_engine import ZeroTouchDecisionEngine
from ..security.models import (
    SecurityEvent,
    ThreatAnalysis,
    AutomatedResponse,
    ResponseAction,
)
from ..core.circuit_breakers import with_circuit_breaker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/zero-touch", tags=["Zero-Touch Security"])

# =============================================================================
# Prometheus Metrics
# =============================================================================

ZERO_TOUCH_REQUESTS = Counter(
    "qbitel_zero_touch_api_requests_total",
    "Total Zero-Touch API requests",
    ["endpoint", "status"],
)

ZERO_TOUCH_LATENCY = Histogram(
    "qbitel_zero_touch_api_latency_seconds",
    "Zero-Touch API latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)

ACTIVE_INCIDENTS = Gauge(
    "qbitel_zero_touch_active_incidents",
    "Number of active security incidents",
)

# =============================================================================
# Global Decision Engine Instance
# =============================================================================

_decision_engine: Optional[ZeroTouchDecisionEngine] = None


def get_decision_engine() -> ZeroTouchDecisionEngine:
    """Get the decision engine instance."""
    global _decision_engine
    if _decision_engine is None:
        raise HTTPException(
            status_code=503, detail="Zero-Touch Decision Engine not initialized"
        )
    return _decision_engine


async def initialize_decision_engine(config: Any) -> ZeroTouchDecisionEngine:
    """Initialize the decision engine."""
    global _decision_engine
    _decision_engine = ZeroTouchDecisionEngine(config)
    logger.info("Zero-Touch Decision Engine initialized")
    return _decision_engine


async def shutdown_decision_engine():
    """Shutdown the decision engine."""
    global _decision_engine
    if _decision_engine:
        _decision_engine = None
        logger.info("Zero-Touch Decision Engine shutdown")


# =============================================================================
# Request/Response Models
# =============================================================================


class SecurityEventRequest(BaseModel):
    """Request model for submitting a security event."""

    event_type: str = Field(..., description="Type of security event")
    severity: str = Field(..., description="Event severity (critical/high/medium/low)")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    destination_ip: Optional[str] = Field(None, description="Destination IP address")
    source_port: Optional[int] = Field(None, description="Source port")
    destination_port: Optional[int] = Field(None, description="Destination port")
    protocol: Optional[str] = Field(None, description="Network protocol")
    user_id: Optional[str] = Field(None, description="Associated user ID")
    asset_id: Optional[str] = Field(None, description="Affected asset ID")
    asset_type: Optional[str] = Field(None, description="Type of affected asset")
    indicators: List[str] = Field(default_factory=list, description="IOC indicators")
    raw_event: Dict[str, Any] = Field(default_factory=dict, description="Raw event data")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "intrusion_attempt",
                "severity": "high",
                "source_ip": "192.168.1.100",
                "destination_ip": "10.0.0.50",
                "destination_port": 22,
                "protocol": "SSH",
                "asset_id": "server-prod-001",
                "asset_type": "linux_server",
                "indicators": ["failed_login_brute_force", "suspicious_user_agent"],
                "raw_event": {"attempts": 50, "duration_seconds": 120},
            }
        }


class ThreatAnalysisResponse(BaseModel):
    """Response model for threat analysis."""

    event_id: str
    threat_level: str
    confidence: float
    threat_type: Optional[str]
    mitre_techniques: List[str]
    affected_assets: List[str]
    blast_radius: int
    analysis_summary: str
    recommended_actions: List[str]


class DecisionResponse(BaseModel):
    """Response model for automated decision."""

    decision_id: str
    event_id: str
    decision: str  # auto_execute, auto_approve, escalate
    confidence: float
    response_plan: Dict[str, Any]
    actions: List[Dict[str, Any]]
    execution_status: str
    requires_approval: bool
    escalation_reason: Optional[str]
    processing_time_ms: int


class ManualApprovalRequest(BaseModel):
    """Request for manual approval of a pending action."""

    decision_id: str = Field(..., description="ID of the decision to approve")
    action_ids: List[str] = Field(..., description="IDs of actions to approve")
    approved: bool = Field(..., description="Whether to approve or reject")
    comments: Optional[str] = Field(None, description="Reviewer comments")


class ConfigurationUpdate(BaseModel):
    """Request to update decision engine configuration."""

    auto_execute_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence threshold for auto-execution"
    )
    auto_approve_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence threshold for auto-approval"
    )
    escalation_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence threshold for escalation"
    )
    enable_business_hour_constraints: Optional[bool] = Field(
        None, description="Enable business hour constraints for disruptive actions"
    )
    high_risk_actions: Optional[List[str]] = Field(
        None, description="Actions requiring extra approval"
    )


class MetricsResponse(BaseModel):
    """Response model for decision engine metrics."""

    total_events_processed: int
    auto_executed: int
    auto_approved: int
    escalated: int
    manual_approvals: int
    average_response_time_ms: float
    success_rate: float
    false_positive_rate: float
    coverage_by_threat_type: Dict[str, int]
    decision_distribution: Dict[str, float]
    recent_decisions: List[Dict[str, Any]]


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("/analyze", response_model=ThreatAnalysisResponse)
@with_circuit_breaker("llm")
async def analyze_security_event(
    request: SecurityEventRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    engine: ZeroTouchDecisionEngine = Depends(get_decision_engine),
):
    """
    Analyze a security event and return threat assessment.

    This endpoint performs comprehensive threat analysis without taking action.
    Use /respond for full automated response.
    """
    import time

    start_time = time.time()
    ZERO_TOUCH_REQUESTS.labels(endpoint="analyze", status="started").inc()

    try:
        # Convert request to SecurityEvent
        event = SecurityEvent(
            event_id=str(uuid4()),
            event_type=request.event_type,
            timestamp=datetime.now(timezone.utc),
            severity=request.severity,
            source_ip=request.source_ip,
            destination_ip=request.destination_ip,
            source_port=request.source_port,
            destination_port=request.destination_port,
            protocol=request.protocol,
            user_id=request.user_id,
            asset_id=request.asset_id,
            asset_type=request.asset_type,
            indicators=request.indicators,
            raw_event=request.raw_event,
            context=request.context,
        )

        # Perform analysis
        analysis = await engine._analyze_threat(event)

        processing_time = (time.time() - start_time) * 1000
        ZERO_TOUCH_LATENCY.labels(endpoint="analyze").observe(processing_time / 1000)
        ZERO_TOUCH_REQUESTS.labels(endpoint="analyze", status="success").inc()

        return ThreatAnalysisResponse(
            event_id=event.event_id,
            threat_level=analysis.threat_level,
            confidence=analysis.confidence,
            threat_type=analysis.threat_type,
            mitre_techniques=analysis.mitre_techniques or [],
            affected_assets=analysis.affected_assets or [],
            blast_radius=analysis.blast_radius or 0,
            analysis_summary=analysis.analysis_summary or "",
            recommended_actions=analysis.recommended_actions or [],
        )

    except Exception as e:
        ZERO_TOUCH_REQUESTS.labels(endpoint="analyze", status="error").inc()
        logger.error(f"Threat analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/respond", response_model=DecisionResponse)
@with_circuit_breaker("llm")
async def respond_to_security_event(
    request: SecurityEventRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    engine: ZeroTouchDecisionEngine = Depends(get_decision_engine),
):
    """
    Analyze and respond to a security event autonomously.

    The decision engine will:
    1. Analyze the threat
    2. Assess business impact
    3. Generate response options
    4. Make autonomous decision (or escalate)
    5. Execute approved actions
    """
    import time

    start_time = time.time()
    ZERO_TOUCH_REQUESTS.labels(endpoint="respond", status="started").inc()

    try:
        # Convert request to SecurityEvent
        event = SecurityEvent(
            event_id=str(uuid4()),
            event_type=request.event_type,
            timestamp=datetime.now(timezone.utc),
            severity=request.severity,
            source_ip=request.source_ip,
            destination_ip=request.destination_ip,
            source_port=request.source_port,
            destination_port=request.destination_port,
            protocol=request.protocol,
            user_id=request.user_id,
            asset_id=request.asset_id,
            asset_type=request.asset_type,
            indicators=request.indicators,
            raw_event=request.raw_event,
            context=request.context,
        )

        # Full analysis and response
        response = await engine.analyze_and_respond(event)

        processing_time = int((time.time() - start_time) * 1000)
        ZERO_TOUCH_LATENCY.labels(endpoint="respond").observe(processing_time / 1000)
        ZERO_TOUCH_REQUESTS.labels(endpoint="respond", status="success").inc()

        # Update active incidents gauge
        ACTIVE_INCIDENTS.inc()
        background_tasks.add_task(_decrement_active_incidents_after_resolution, response)

        return DecisionResponse(
            decision_id=response.response_id,
            event_id=event.event_id,
            decision=response.decision_type,
            confidence=response.confidence,
            response_plan={
                "primary_actions": [a.action_type for a in response.primary_actions],
                "fallback_actions": [a.action_type for a in response.fallback_actions],
                "estimated_impact": response.estimated_impact,
            },
            actions=[
                {
                    "action_id": a.action_id,
                    "action_type": a.action_type,
                    "target": a.target,
                    "status": a.status,
                    "requires_approval": a.requires_approval,
                }
                for a in response.primary_actions
            ],
            execution_status=response.execution_status,
            requires_approval=any(a.requires_approval for a in response.primary_actions),
            escalation_reason=response.escalation_reason,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        ZERO_TOUCH_REQUESTS.labels(endpoint="respond", status="error").inc()
        logger.error(f"Security response failed: {e}")
        raise HTTPException(status_code=500, detail=f"Response failed: {str(e)}")


@router.get("/decisions")
async def list_decisions(
    status: Optional[str] = Query(None, description="Filter by status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    engine: ZeroTouchDecisionEngine = Depends(get_decision_engine),
):
    """
    List security decisions with optional filtering.

    Returns a paginated list of decisions made by the engine.
    """
    ZERO_TOUCH_REQUESTS.labels(endpoint="list_decisions", status="success").inc()

    # Get decision history from engine
    decisions = engine.get_decision_history(
        status=status,
        severity=severity,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )

    return {
        "total": len(decisions),
        "limit": limit,
        "offset": offset,
        "decisions": decisions,
    }


@router.get("/decisions/{decision_id}")
async def get_decision(
    decision_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    engine: ZeroTouchDecisionEngine = Depends(get_decision_engine),
):
    """
    Get detailed information about a specific decision.

    Includes full analysis, actions taken, and audit trail.
    """
    ZERO_TOUCH_REQUESTS.labels(endpoint="get_decision", status="success").inc()

    decision = engine.get_decision(decision_id)
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")

    return decision


@router.post("/approve")
async def approve_action(
    request: ManualApprovalRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    engine: ZeroTouchDecisionEngine = Depends(get_decision_engine),
):
    """
    Manually approve or reject pending actions.

    Used for actions that require human approval before execution.
    """
    ZERO_TOUCH_REQUESTS.labels(endpoint="approve", status="started").inc()

    try:
        result = await engine.process_approval(
            decision_id=request.decision_id,
            action_ids=request.action_ids,
            approved=request.approved,
            approver_id=current_user.get("user_id"),
            comments=request.comments,
        )

        ZERO_TOUCH_REQUESTS.labels(endpoint="approve", status="success").inc()
        return result

    except ValueError as e:
        ZERO_TOUCH_REQUESTS.labels(endpoint="approve", status="error").inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        ZERO_TOUCH_REQUESTS.labels(endpoint="approve", status="error").inc()
        logger.error(f"Approval processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Approval failed: {str(e)}")


@router.get("/pending-approvals")
async def list_pending_approvals(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    engine: ZeroTouchDecisionEngine = Depends(get_decision_engine),
):
    """
    List actions pending manual approval.

    Returns actions that were escalated and require human review.
    """
    ZERO_TOUCH_REQUESTS.labels(endpoint="pending_approvals", status="success").inc()

    pending = engine.get_pending_approvals(severity=severity, limit=limit)

    return {
        "total": len(pending),
        "pending_approvals": pending,
    }


@router.get("/config")
async def get_configuration(
    current_user: Dict[str, Any] = Depends(get_current_user),
    engine: ZeroTouchDecisionEngine = Depends(get_decision_engine),
):
    """
    Get current decision engine configuration.

    Returns thresholds, enabled features, and operational parameters.
    """
    ZERO_TOUCH_REQUESTS.labels(endpoint="get_config", status="success").inc()

    return engine.get_configuration()


@router.put("/config")
async def update_configuration(
    request: ConfigurationUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    engine: ZeroTouchDecisionEngine = Depends(get_decision_engine),
):
    """
    Update decision engine configuration.

    Allows tuning of thresholds and operational parameters.
    Requires admin privileges.
    """
    # Check admin privileges
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=403, detail="Admin privileges required to update configuration"
        )

    ZERO_TOUCH_REQUESTS.labels(endpoint="update_config", status="started").inc()

    try:
        updated = engine.update_configuration(
            auto_execute_threshold=request.auto_execute_threshold,
            auto_approve_threshold=request.auto_approve_threshold,
            escalation_threshold=request.escalation_threshold,
            enable_business_hour_constraints=request.enable_business_hour_constraints,
            high_risk_actions=request.high_risk_actions,
        )

        ZERO_TOUCH_REQUESTS.labels(endpoint="update_config", status="success").inc()

        logger.info(
            f"Configuration updated by {current_user.get('user_id')}: {request.model_dump(exclude_none=True)}"
        )

        return {"message": "Configuration updated", "configuration": updated}

    except Exception as e:
        ZERO_TOUCH_REQUESTS.labels(endpoint="update_config", status="error").inc()
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Configuration update failed: {str(e)}"
        )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    time_range: str = Query("24h", description="Time range (1h, 24h, 7d, 30d)"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    engine: ZeroTouchDecisionEngine = Depends(get_decision_engine),
):
    """
    Get decision engine metrics and statistics.

    Returns aggregated metrics over the specified time range.
    """
    ZERO_TOUCH_REQUESTS.labels(endpoint="metrics", status="success").inc()

    metrics = engine.get_metrics(time_range=time_range)

    return MetricsResponse(
        total_events_processed=metrics.get("total_events", 0),
        auto_executed=metrics.get("auto_executed", 0),
        auto_approved=metrics.get("auto_approved", 0),
        escalated=metrics.get("escalated", 0),
        manual_approvals=metrics.get("manual_approvals", 0),
        average_response_time_ms=metrics.get("avg_response_time_ms", 0.0),
        success_rate=metrics.get("success_rate", 0.0),
        false_positive_rate=metrics.get("false_positive_rate", 0.0),
        coverage_by_threat_type=metrics.get("by_threat_type", {}),
        decision_distribution=metrics.get("decision_distribution", {}),
        recent_decisions=metrics.get("recent_decisions", []),
    )


@router.get("/health")
async def health_check(
    engine: ZeroTouchDecisionEngine = Depends(get_decision_engine),
):
    """
    Health check for the Zero-Touch Decision Engine.

    Returns operational status and component health.
    """
    ZERO_TOUCH_REQUESTS.labels(endpoint="health", status="success").inc()

    health = engine.get_health_status()

    return {
        "status": health.get("status", "unknown"),
        "components": health.get("components", {}),
        "last_decision_time": health.get("last_decision_time"),
        "uptime_seconds": health.get("uptime_seconds", 0),
        "version": "1.0.0",
    }


@router.post("/simulate")
async def simulate_response(
    request: SecurityEventRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    engine: ZeroTouchDecisionEngine = Depends(get_decision_engine),
):
    """
    Simulate response to a security event without executing actions.

    Useful for testing and validating decision logic.
    """
    ZERO_TOUCH_REQUESTS.labels(endpoint="simulate", status="started").inc()

    try:
        event = SecurityEvent(
            event_id=str(uuid4()),
            event_type=request.event_type,
            timestamp=datetime.now(timezone.utc),
            severity=request.severity,
            source_ip=request.source_ip,
            destination_ip=request.destination_ip,
            source_port=request.source_port,
            destination_port=request.destination_port,
            protocol=request.protocol,
            user_id=request.user_id,
            asset_id=request.asset_id,
            asset_type=request.asset_type,
            indicators=request.indicators,
            raw_event=request.raw_event,
            context=request.context,
        )

        # Run simulation (no execution)
        simulation = await engine.simulate_response(event)

        ZERO_TOUCH_REQUESTS.labels(endpoint="simulate", status="success").inc()

        return {
            "simulation_id": str(uuid4()),
            "event": event.model_dump(),
            "predicted_decision": simulation.get("decision"),
            "predicted_confidence": simulation.get("confidence"),
            "predicted_actions": simulation.get("actions", []),
            "risk_assessment": simulation.get("risk_assessment"),
            "estimated_impact": simulation.get("estimated_impact"),
            "warnings": simulation.get("warnings", []),
        }

    except Exception as e:
        ZERO_TOUCH_REQUESTS.labels(endpoint="simulate", status="error").inc()
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


# =============================================================================
# Background Tasks
# =============================================================================


async def _decrement_active_incidents_after_resolution(response: AutomatedResponse):
    """Decrement active incidents counter when resolved."""
    import asyncio

    # Wait for potential resolution (simplified - in production, use event-driven)
    await asyncio.sleep(300)  # 5 minutes
    ACTIVE_INCIDENTS.dec()
