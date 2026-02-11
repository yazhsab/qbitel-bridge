"""
QBITEL - Threat Intelligence Platform API Endpoints

REST API endpoints for threat intelligence operations including IOC queries,
MITRE ATT&CK lookups, threat hunting, and feed management.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..core.config import Config
from ..core.exceptions import QbitelAIException
from ..security.models import SecurityEvent
from ..threat_intelligence import (
    ThreatIntelligenceManager,
    get_threat_intelligence_manager,
    HuntCampaign,
)
from .middleware import get_current_user, get_config


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/threat-intel", tags=["Threat Intelligence"])


# Request/Response Models


class EnrichEventRequest(BaseModel):
    """Request to enrich security event with TIP data."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of security event")
    severity: str = Field(..., description="Event severity level")
    description: str = Field(..., description="Event description")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    destination_ip: Optional[str] = Field(None, description="Destination IP address")
    source_port: Optional[int] = Field(None, description="Source port")
    destination_port: Optional[int] = Field(None, description="Destination port")
    protocol: Optional[str] = Field(None, description="Network protocol")
    user_id: Optional[str] = Field(None, description="User ID if available")
    additional_context: Optional[Dict[str, Any]] = Field(
        None, description="Additional event context"
    )


class EnrichEventResponse(BaseModel):
    """Response with TIP enrichment data."""

    event_id: str
    timestamp: str
    ioc_matches: List[Dict[str, Any]]
    ttp_mapping: Optional[Dict[str, Any]]
    threat_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized threat score 0-1"
    )
    recommendations: List[str]
    processing_time_ms: float


class QueryIOCsRequest(BaseModel):
    """Request to query IOCs."""

    indicator_type: Optional[str] = Field(
        None, description="Type of indicator (ipv4-addr, domain-name, etc)"
    )
    labels: Optional[List[str]] = Field(None, description="Filter by labels")
    min_confidence: int = Field(0, ge=0, le=100, description="Minimum confidence score")
    limit: int = Field(100, ge=1, le=1000, description="Maximum results to return")


class QueryIOCsResponse(BaseModel):
    """Response with IOC query results."""

    indicators: List[Dict[str, Any]]
    total: int
    query_time_ms: float


class QueryTechniquesRequest(BaseModel):
    """Request to query MITRE ATT&CK techniques."""

    query: Optional[str] = Field(
        None, description="Search query (technique name, ID, or description)"
    )
    tactic: Optional[str] = Field(None, description="Filter by tactic")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")


class QueryTechniquesResponse(BaseModel):
    """Response with technique query results."""

    techniques: List[Dict[str, Any]]
    total: int
    query_time_ms: float


class ExecuteHuntRequest(BaseModel):
    """Request to execute threat hunting campaign."""

    hypotheses: Optional[List[str]] = Field(
        None, description="Specific hypothesis IDs to hunt (None = all)"
    )
    time_range_hours: int = Field(
        24, ge=1, le=168, description="Time range to hunt in hours"
    )
    description: Optional[str] = Field(None, description="Campaign description")


class ExecuteHuntResponse(BaseModel):
    """Response with hunt campaign results."""

    campaign_id: str
    start_time: str
    end_time: str
    hypotheses_tested: int
    findings_count: int
    high_severity_findings: int
    findings: List[Dict[str, Any]]
    execution_time_ms: float


class FeedStatusResponse(BaseModel):
    """Response with IOC feed status."""

    feeds: List[Dict[str, Any]]
    total_feeds: int
    total_indicators: int


class UpdateFeedResponse(BaseModel):
    """Response after updating IOC feed."""

    feed_id: str
    indicators_added: int
    update_time: str
    update_duration_ms: float


class CoverageReportResponse(BaseModel):
    """Response with detection coverage report."""

    timestamp: str
    mitre_attack_coverage: Optional[Dict[str, Any]]
    ioc_feed_status: List[Dict[str, Any]]


class HealthCheckResponse(BaseModel):
    """Response with TIP health status."""

    status: str
    initialized: bool
    components: Dict[str, bool]
    ioc_feeds: int
    indicators_cached: int
    attack_techniques: int
    hunt_hypotheses: int


# API Endpoints


@router.post("/enrich", response_model=EnrichEventResponse)
async def enrich_security_event(
    request: EnrichEventRequest,
    config: Config = Depends(get_config),
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """
    Enrich security event with threat intelligence data.

    Processes the event through the TIP pipeline to:
    - Check against known IOCs
    - Map to MITRE ATT&CK techniques
    - Calculate threat score
    - Generate recommendations
    """
    start_time = datetime.utcnow()

    try:
        # Get TIP manager
        tip_manager = get_threat_intelligence_manager(config)

        if not tip_manager._initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Threat Intelligence Platform not initialized",
            )

        # Convert request to SecurityEvent
        event = SecurityEvent(
            event_id=request.event_id,
            event_type=request.event_type,
            severity=request.severity,
            description=request.description,
            source_ip=request.source_ip or "unknown",
            destination_ip=request.destination_ip or "unknown",
            source_port=request.source_port or 0,
            destination_port=request.destination_port or 0,
            protocol=request.protocol or "unknown",
            user_id=request.user_id or "unknown",
            timestamp=datetime.utcnow(),
            raw_data=request.additional_context or {},
        )

        # Process through TIP
        enrichment = await tip_manager.process_security_event(event)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return EnrichEventResponse(
            event_id=enrichment["event_id"],
            timestamp=enrichment["timestamp"],
            ioc_matches=enrichment["ioc_matches"],
            ttp_mapping=enrichment["ttp_mapping"],
            threat_score=enrichment["threat_score"],
            recommendations=enrichment["recommendations"],
            processing_time_ms=processing_time,
        )

    except QbitelAIException as e:
        logger.error(f"TIP enrichment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TIP enrichment failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in TIP enrichment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during TIP enrichment",
        )


@router.post("/iocs/query", response_model=QueryIOCsResponse)
async def query_iocs(
    request: QueryIOCsRequest,
    config: Config = Depends(get_config),
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """
    Query Indicators of Compromise (IOCs).

    Search IOCs by type, labels, and confidence threshold.
    Returns matching indicators with metadata.
    """
    start_time = datetime.utcnow()

    try:
        tip_manager = get_threat_intelligence_manager(config)

        if not tip_manager._initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Threat Intelligence Platform not initialized",
            )

        result = await tip_manager.query_threat_intelligence(
            query_type="iocs",
            query_params={
                "indicator_type": request.indicator_type,
                "labels": request.labels,
                "min_confidence": request.min_confidence,
                "limit": request.limit,
            },
        )

        query_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return QueryIOCsResponse(
            indicators=result["indicators"],
            total=result["total"],
            query_time_ms=query_time,
        )

    except QbitelAIException as e:
        logger.error(f"IOC query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"IOC query failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in IOC query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during IOC query",
        )


@router.post("/techniques/query", response_model=QueryTechniquesResponse)
async def query_techniques(
    request: QueryTechniquesRequest,
    config: Config = Depends(get_config),
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """
    Query MITRE ATT&CK techniques.

    Search techniques by name, ID, description, or tactic.
    Returns matching techniques with full details.
    """
    start_time = datetime.utcnow()

    try:
        tip_manager = get_threat_intelligence_manager(config)

        if not tip_manager._initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Threat Intelligence Platform not initialized",
            )

        result = await tip_manager.query_threat_intelligence(
            query_type="techniques",
            query_params={
                "query": request.query,
                "tactic": request.tactic,
                "limit": request.limit,
            },
        )

        query_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return QueryTechniquesResponse(
            techniques=result["techniques"],
            total=result["total"],
            query_time_ms=query_time,
        )

    except QbitelAIException as e:
        logger.error(f"Technique query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Technique query failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in technique query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during technique query",
        )


@router.post("/hunt", response_model=ExecuteHuntResponse)
async def execute_threat_hunt(
    request: ExecuteHuntRequest,
    config: Config = Depends(get_config),
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """
    Execute automated threat hunting campaign.

    Runs hypothesis-driven threat hunt across security event history.
    Returns findings with severity assessment and recommendations.
    """
    start_time = datetime.utcnow()

    try:
        tip_manager = get_threat_intelligence_manager(config)

        if not tip_manager._initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Threat Intelligence Platform not initialized",
            )

        campaign = await tip_manager.execute_threat_hunt(
            hypotheses=request.hypotheses,
            time_range_hours=request.time_range_hours,
        )

        # Count high severity findings
        high_severity = sum(
            1
            for finding in campaign.findings
            if finding.severity in ["high", "critical"]
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ExecuteHuntResponse(
            campaign_id=campaign.campaign_id,
            start_time=campaign.start_time.isoformat(),
            end_time=(
                campaign.end_time.isoformat()
                if campaign.end_time
                else datetime.utcnow().isoformat()
            ),
            hypotheses_tested=len(campaign.hypotheses_tested),
            findings_count=len(campaign.findings),
            high_severity_findings=high_severity,
            findings=[finding.to_dict() for finding in campaign.findings],
            execution_time_ms=execution_time,
        )

    except QbitelAIException as e:
        logger.error(f"Threat hunt execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Threat hunt execution failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in threat hunt execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during threat hunt execution",
        )


@router.get("/feeds", response_model=FeedStatusResponse)
async def get_feed_status(
    config: Config = Depends(get_config),
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """
    Get IOC feed status.

    Returns status of all configured IOC feeds including:
    - Feed name and source
    - Enabled status
    - Last update time
    - Indicators count
    """
    try:
        tip_manager = get_threat_intelligence_manager(config)

        if not tip_manager._initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Threat Intelligence Platform not initialized",
            )

        feeds = []
        total_indicators = 0

        if tip_manager.stix_client:
            for feed_id, feed in tip_manager.stix_client.ioc_feeds.items():
                feeds.append(
                    {
                        "feed_id": feed_id,
                        "name": feed.name,
                        "source": feed.source,
                        "feed_type": feed.feed_type,
                        "enabled": feed.enabled,
                        "last_update": (
                            feed.last_update.isoformat() if feed.last_update else None
                        ),
                        "update_interval_hours": feed.update_interval_hours,
                        "indicators_count": feed.indicators_count,
                    }
                )
                total_indicators += feed.indicators_count

        return FeedStatusResponse(
            feeds=feeds,
            total_feeds=len(feeds),
            total_indicators=total_indicators,
        )

    except QbitelAIException as e:
        logger.error(f"Failed to get feed status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feed status: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error getting feed status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error getting feed status",
        )


@router.post("/feeds/{feed_id}/update", response_model=UpdateFeedResponse)
async def update_feed(
    feed_id: str,
    config: Config = Depends(get_config),
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """
    Update specific IOC feed.

    Manually trigger update of an IOC feed from its source.
    Returns number of indicators added/updated.
    """
    start_time = datetime.utcnow()

    try:
        tip_manager = get_threat_intelligence_manager(config)

        if not tip_manager._initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Threat Intelligence Platform not initialized",
            )

        results = await tip_manager.update_ioc_feeds(feed_ids=[feed_id])

        if feed_id not in results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feed not found: {feed_id}",
            )

        indicators_added = results[feed_id]
        update_duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        return UpdateFeedResponse(
            feed_id=feed_id,
            indicators_added=indicators_added,
            update_time=datetime.utcnow().isoformat(),
            update_duration_ms=update_duration,
        )

    except HTTPException:
        raise
    except QbitelAIException as e:
        logger.error(f"Feed update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feed update failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error updating feed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during feed update",
        )


@router.get("/coverage", response_model=CoverageReportResponse)
async def get_coverage_report(
    config: Config = Depends(get_config),
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """
    Get detection coverage report.

    Returns comprehensive coverage report including:
    - MITRE ATT&CK coverage by tactic
    - IOC feed status and statistics
    - Overall detection capability assessment
    """
    try:
        tip_manager = get_threat_intelligence_manager(config)

        if not tip_manager._initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Threat Intelligence Platform not initialized",
            )

        report = tip_manager.get_coverage_report()

        return CoverageReportResponse(
            timestamp=report["timestamp"],
            mitre_attack_coverage=report["mitre_attack_coverage"],
            ioc_feed_status=report["ioc_feed_status"],
        )

    except QbitelAIException as e:
        logger.error(f"Failed to get coverage report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get coverage report: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error getting coverage report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error getting coverage report",
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    config: Config = Depends(get_config),
):
    """
    TIP health check.

    Returns health status of all TIP components:
    - Initialization status
    - Component availability
    - Resource statistics
    """
    try:
        tip_manager = get_threat_intelligence_manager(config)

        health = tip_manager.get_health_status()

        overall_status = "healthy" if health["initialized"] else "degraded"

        return HealthCheckResponse(
            status=overall_status,
            initialized=health["initialized"],
            components=health["components"],
            ioc_feeds=health["ioc_feeds"],
            indicators_cached=health["indicators_cached"],
            attack_techniques=health["attack_techniques"],
            hunt_hypotheses=health["hunt_hypotheses"],
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthCheckResponse(
            status="unhealthy",
            initialized=False,
            components={},
            ioc_feeds=0,
            indicators_cached=0,
            attack_techniques=0,
            hunt_hypotheses=0,
        )
