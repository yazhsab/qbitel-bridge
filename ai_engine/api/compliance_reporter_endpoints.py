"""
QBITEL - Autonomous Compliance Reporter API Endpoints

RESTful API endpoints for compliance reporting, monitoring, and audit evidence generation.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
import asyncio

from ..compliance.compliance_reporter import (
    AutonomousComplianceReporter,
    ComplianceStandard,
    ComplianceAlert,
    AuditRequest,
    AuditEvidence,
    MonitoringFrequency,
    AlertSeverity,
    ContinuousMonitoringConfig,
    get_compliance_reporter,
)
from ..compliance.report_generator import ReportFormat, ReportType
from ..core.config import get_config
from ..monitoring.enterprise_metrics import get_enterprise_metrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/compliance", tags=["compliance-reporter"])
metrics = get_enterprise_metrics()


# Request/Response Models
class ComplianceReportRequest(BaseModel):
    """Request model for compliance report generation."""

    protocol: str = Field(..., description="Protocol or system being assessed")
    standard: str = Field(
        ..., description="Compliance standard (GDPR, SOC2, HIPAA, PCI-DSS)"
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict, description="Evidence data for assessment"
    )
    report_type: str = Field(default="detailed_technical", description="Type of report")
    format: str = Field(
        default="pdf", description="Output format (pdf, json, html, excel)"
    )


class ComplianceReportResponse(BaseModel):
    """Response model for compliance report."""

    report_id: str
    framework: str
    report_type: str
    format: str
    generated_date: str
    file_name: str
    file_size: int
    compliance_score: float
    risk_score: float
    download_url: str
    metadata: Dict[str, Any]


class MonitoringConfigRequest(BaseModel):
    """Request model for monitoring configuration."""

    enabled: bool = True
    frequency: str = Field(default="hourly", description="Monitoring frequency")
    frameworks: List[str] = Field(
        default_factory=list, description="Frameworks to monitor"
    )
    protocols: List[str] = Field(
        default_factory=list, description="Protocols to monitor"
    )
    auto_remediation: bool = False
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)


class AuditRequestModel(BaseModel):
    """Request model for audit evidence generation."""

    auditor: str = Field(..., description="Auditor name or ID")
    framework: str = Field(..., description="Compliance framework")
    requirements: List[str] = Field(
        default_factory=list, description="Specific requirements"
    )
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    evidence_types: List[str] = Field(default_factory=list)
    format: str = "comprehensive"


class AuditEvidenceResponse(BaseModel):
    """Response model for audit evidence."""

    evidence_id: str
    request_id: str
    framework: str
    generated_date: str
    evidence_count: int
    verification_status: str
    digital_signature: str
    download_url: str


class ComplianceAlertResponse(BaseModel):
    """Response model for compliance alert."""

    alert_id: str
    timestamp: str
    severity: str
    standard: str
    requirement_id: str
    requirement_title: str
    violation_description: str
    recommended_actions: List[str]
    auto_remediation_available: bool


# API Endpoints


@router.post("/reports/generate", response_model=ComplianceReportResponse)
async def generate_compliance_report(
    request: ComplianceReportRequest, background_tasks: BackgroundTasks
):
    """
    Generate comprehensive compliance report.

    **Success Metrics:**
    - Generation time: <10 minutes
    - Compliance accuracy: 95%+
    - Audit pass rate: 98%+
    """
    try:
        logger.info(
            f"Generating {request.standard} compliance report for {request.protocol}"
        )

        # Get compliance reporter instance
        reporter = await get_compliance_reporter()

        # Map string values to enums
        report_type = ReportType[request.report_type.upper()]
        report_format = ReportFormat[request.format.upper()]

        # Generate report
        report = await reporter.generate_compliance_report(
            protocol=request.protocol,
            standard=request.standard,
            evidence=request.evidence,
            report_type=report_type,
            format=report_format,
        )

        # Record metrics
        metrics.increment_protocol_discovery_counter(
            "compliance_api_reports_generated_total",
            labels={"standard": request.standard, "format": request.format},
        )

        # Create response
        response = ComplianceReportResponse(
            report_id=report.report_id,
            framework=report.framework,
            report_type=report.report_type.value,
            format=report.format.value,
            generated_date=report.generated_date.isoformat(),
            file_name=report.file_name,
            file_size=report.file_size,
            compliance_score=report.metadata.get("compliance_score", 0.0),
            risk_score=report.metadata.get("risk_score", 0.0),
            download_url=f"/api/v1/compliance/reports/{report.report_id}/download",
            metadata=report.metadata,
        )

        return response

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Report generation failed: {str(e)}"
        )


@router.get("/reports/{report_id}/download")
async def download_compliance_report(report_id: str):
    """Download generated compliance report."""
    try:
        # In production, this would retrieve the report from storage
        # For now, return a placeholder response
        raise HTTPException(
            status_code=501,
            detail="Report download endpoint - implementation requires storage backend",
        )

    except Exception as e:
        logger.error(f"Report download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/start")
async def start_continuous_monitoring(
    config: MonitoringConfigRequest, background_tasks: BackgroundTasks
):
    """
    Start continuous compliance monitoring.

    **Features:**
    - Real-time compliance checking
    - Automated alert generation
    - Trend analysis
    - Predictive compliance issues
    """
    try:
        logger.info(
            f"Starting continuous monitoring for {len(config.frameworks)} frameworks"
        )

        # Get compliance reporter
        reporter = await get_compliance_reporter()

        # Map frequency string to enum
        frequency = MonitoringFrequency[config.frequency.upper()]

        # Create monitoring configuration
        monitoring_config = ContinuousMonitoringConfig(
            enabled=config.enabled,
            frequency=frequency,
            frameworks=config.frameworks,
            auto_remediation=config.auto_remediation,
            alert_thresholds=config.alert_thresholds,
        )

        # Update reporter configuration
        reporter.monitoring_config = monitoring_config

        # Start monitoring in background
        async def monitor_task():
            async for alert in reporter.continuous_compliance_monitoring(
                protocols=config.protocols,
                standards=config.frameworks,
                config=monitoring_config,
            ):
                logger.info(f"Alert generated: {alert.alert_id}")

        background_tasks.add_task(monitor_task)

        return {
            "status": "monitoring_started",
            "frameworks": config.frameworks,
            "frequency": config.frequency,
            "auto_remediation": config.auto_remediation,
        }

    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/status")
async def get_monitoring_status():
    """Get current monitoring status and statistics."""
    try:
        reporter = await get_compliance_reporter()
        status = reporter.get_service_status()

        return {
            "monitoring_active": status["monitoring_active"],
            "active_monitors": status["active_monitors"],
            "alert_queue_size": status["alert_queue_size"],
            "frameworks_monitored": len(status["active_monitors"]),
            "performance_metrics": status["performance_metrics"],
        }

    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/alerts", response_model=List[ComplianceAlertResponse])
async def get_compliance_alerts(
    limit: int = Query(default=100, le=1000),
    severity: Optional[str] = None,
    framework: Optional[str] = None,
):
    """Get recent compliance alerts."""
    try:
        reporter = await get_compliance_reporter()

        # Get alerts from queue (non-blocking)
        alerts = []
        try:
            for _ in range(min(limit, reporter.alert_queue.qsize())):
                alert = reporter.alert_queue.get_nowait()

                # Filter by severity if specified
                if severity and alert.severity.value != severity.lower():
                    continue

                # Filter by framework if specified
                if framework and alert.standard != framework:
                    continue

                alerts.append(
                    ComplianceAlertResponse(
                        alert_id=alert.alert_id,
                        timestamp=alert.timestamp.isoformat(),
                        severity=alert.severity.value,
                        standard=alert.standard,
                        requirement_id=alert.requirement_id,
                        requirement_title=alert.requirement_title,
                        violation_description=alert.violation_description,
                        recommended_actions=alert.recommended_actions,
                        auto_remediation_available=alert.auto_remediation_available,
                    )
                )
        except asyncio.QueueEmpty:
            pass

        return alerts

    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audit/evidence", response_model=AuditEvidenceResponse)
async def generate_audit_evidence(request: AuditRequestModel):
    """
    Generate comprehensive audit evidence package.

    **Features:**
    - Collect relevant logs
    - Generate evidence documentation
    - Create audit trail
    - Prepare for auditor review
    """
    try:
        logger.info(f"Generating audit evidence for {request.framework}")

        # Get compliance reporter
        reporter = await get_compliance_reporter()

        # Parse dates
        start_date = (
            datetime.fromisoformat(request.start_date)
            if request.start_date
            else datetime.utcnow() - timedelta(days=30)
        )
        end_date = (
            datetime.fromisoformat(request.end_date)
            if request.end_date
            else datetime.utcnow()
        )

        # Create audit request
        audit_request = AuditRequest(
            request_id=f"AUDIT-{int(datetime.utcnow().timestamp())}",
            auditor=request.auditor,
            framework=request.framework,
            requirements=request.requirements,
            start_date=start_date,
            end_date=end_date,
            evidence_types=request.evidence_types,
            format=request.format,
        )

        # Generate evidence
        evidence = await reporter.generate_audit_evidence(audit_request)

        # Record metrics
        metrics.increment_protocol_discovery_counter(
            "compliance_api_audit_evidence_generated_total",
            labels={"framework": request.framework},
        )

        # Create response
        response = AuditEvidenceResponse(
            evidence_id=evidence.evidence_id,
            request_id=evidence.request_id,
            framework=evidence.framework,
            generated_date=evidence.generated_date.isoformat(),
            evidence_count=len(evidence.evidence_items),
            verification_status=evidence.verification_status,
            digital_signature=evidence.digital_signature,
            download_url=f"/api/v1/compliance/audit/evidence/{evidence.evidence_id}/download",
        )

        return response

    except Exception as e:
        logger.error(f"Audit evidence generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/evidence/{evidence_id}/download")
async def download_audit_evidence(evidence_id: str):
    """Download audit evidence package."""
    try:
        # In production, this would retrieve the evidence from storage
        raise HTTPException(
            status_code=501,
            detail="Evidence download endpoint - implementation requires storage backend",
        )

    except Exception as e:
        logger.error(f"Evidence download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/frameworks")
async def list_compliance_frameworks():
    """List available compliance frameworks."""
    try:
        reporter = await get_compliance_reporter()
        frameworks = reporter.regulatory_kb.get_available_frameworks()

        framework_details = []
        for framework in frameworks:
            metadata = reporter.regulatory_kb.get_framework_metadata(framework)
            framework_details.append(
                {
                    "id": framework,
                    "name": metadata.get("full_name", framework),
                    "applicable_to": metadata.get("applicable_to", []),
                    "last_updated": metadata.get("last_updated", ""),
                    "next_review": metadata.get("next_review", ""),
                }
            )

        return {"frameworks": framework_details, "total_count": len(framework_details)}

    except Exception as e:
        logger.error(f"Failed to list frameworks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/metrics")
async def get_performance_metrics():
    """
    Get compliance reporter performance metrics.

    **Success Metrics:**
    - Report generation time: <10 minutes
    - Compliance accuracy: 95%+
    - Audit pass rate: 98%+
    """
    try:
        reporter = await get_compliance_reporter()
        metrics = reporter.get_performance_metrics()

        return {
            "report_generation": metrics["report_generation"],
            "compliance_accuracy": metrics["compliance_accuracy"],
            "audit_pass_rate": metrics["audit_pass_rate"],
            "monitoring": metrics["monitoring"],
            "service_status": metrics["service_status"],
            "success_criteria": {
                "report_generation_target": "<600 seconds",
                "report_generation_met": metrics["report_generation"]["target_met"],
                "compliance_accuracy_target": "≥95%",
                "compliance_accuracy_met": metrics["compliance_accuracy"]["target_met"],
                "audit_pass_rate_target": "≥98%",
                "audit_pass_rate_met": metrics["audit_pass_rate"]["target_met"],
            },
        }

    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for compliance reporter service."""
    try:
        reporter = await get_compliance_reporter()
        status = reporter.get_service_status()

        return {
            "status": "healthy" if status["service_running"] else "unhealthy",
            "service_running": status["service_running"],
            "monitoring_active": status["monitoring_active"],
            "llm_service": status["llm_service_status"],
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


# Include router in main API
def include_compliance_reporter_routes(app):
    """Include compliance reporter routes in FastAPI app."""
    app.include_router(router)
    logger.info("Compliance Reporter API endpoints registered")
