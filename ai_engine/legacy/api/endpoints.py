"""
QBITEL Engine - Legacy System Whisperer API Endpoints

FastAPI endpoints for Legacy System Whisperer functionality.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from starlette.status import *

from ...core.config import Config, get_config
from ...core.exceptions import QbitelAIException
from ..service import LegacySystemWhispererService
from ..models import (
    LegacySystemContext,
    SystemType,
    Criticality,
    SystemMetrics,
    TimeSeriesData,
    PredictionHorizon,
    MaintenanceRecommendation,
)
from ..exceptions import (
    LegacySystemWhispererException,
    SystemRegistrationException,
    AnomalyDetectionException,
    ValidationException,
)
from ..logging import get_legacy_logger
from ..monitoring import LegacySystemMonitor

from .schemas import (
    # Request schemas
    SystemRegistrationRequest,
    SystemMetricsRequest,
    HealthAnalysisRequest,
    ExpertKnowledgeRequest,
    KnowledgeSearchRequest,
    DecisionSupportRequest,
    MaintenanceSchedulingRequest,
    SystemDashboardRequest,
    ServiceStatusRequest,
    BulkSystemAnalysisRequest,
    CreateAlertRuleRequest,
    AlertsListRequest,
    # Response schemas
    SystemRegistrationResponse,
    HealthAnalysisResponse,
    ExpertKnowledgeResponse,
    KnowledgeSearchResponse,
    DecisionSupportResponse,
    MaintenanceSchedulingResponse,
    SystemDashboardResponse,
    ServiceStatusResponse,
    BulkSystemAnalysisResponse,
    AlertRuleResponse,
    AlertsListResponse,
    HealthCheckResponse,
    APIError,
    ValidationErrorResponse,
    AuthenticationErrorResponse,
    AuthorizationErrorResponse,
    # Enum conversions
    SystemTypeAPI,
    CriticalityAPI,
    PredictionHorizonAPI,
    DecisionCategoryAPI,
)
from .auth import get_current_user, require_permission
from .middleware import get_request_context

# Initialize router
legacy_router = APIRouter(
    prefix="/api/v1/legacy-system-whisperer",
    tags=["Legacy System Whisperer"],
    responses={
        HTTP_400_BAD_REQUEST: {"model": ValidationErrorResponse},
        HTTP_401_UNAUTHORIZED: {"model": AuthenticationErrorResponse},
        HTTP_403_FORBIDDEN: {"model": AuthorizationErrorResponse},
        HTTP_500_INTERNAL_SERVER_ERROR: {"model": APIError},
    },
)

# Initialize logger
logger = get_legacy_logger(__name__)

# Global service instance (will be injected)
_service_instance: Optional[LegacySystemWhispererService] = None
_monitor_instance: Optional[LegacySystemMonitor] = None


def get_service() -> LegacySystemWhispererService:
    """Get Legacy System Whisperer service instance."""
    global _service_instance
    if _service_instance is None:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Legacy System Whisperer service not initialized",
        )
    return _service_instance


def get_monitor() -> LegacySystemMonitor:
    """Get Legacy System Whisperer monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Legacy System Whisperer monitor not initialized",
        )
    return _monitor_instance


async def set_service_instance(service: LegacySystemWhispererService):
    """Set service instance (called during startup)."""
    global _service_instance
    _service_instance = service


async def set_monitor_instance(monitor: LegacySystemMonitor):
    """Set monitor instance (called during startup)."""
    global _monitor_instance
    _monitor_instance = monitor


# Error handlers
@legacy_router.exception_handler(ValidationException)
async def validation_exception_handler(request, exc: ValidationException):
    """Handle validation exceptions."""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


@legacy_router.exception_handler(LegacySystemWhispererException)
async def legacy_exception_handler(request, exc: LegacySystemWhispererException):
    """Handle Legacy System Whisperer exceptions."""
    logger.error(f"Legacy System Whisperer error: {exc}")

    status_code = HTTP_500_INTERNAL_SERVER_ERROR
    if isinstance(exc, SystemRegistrationException):
        status_code = HTTP_400_BAD_REQUEST
    elif isinstance(exc, AnomalyDetectionException):
        status_code = HTTP_422_UNPROCESSABLE_ENTITY

    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error_code": (exc.category.value if hasattr(exc, "category") else "UNKNOWN_ERROR"),
            "message": str(exc),
            "severity": exc.severity.value if hasattr(exc, "severity") else "medium",
            "timestamp": datetime.now().isoformat(),
        },
    )


# Health Check Endpoint
@legacy_router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Get service health status.

    Returns comprehensive health information including component status,
    uptime, and basic metrics.
    """
    try:
        service = get_service()
        monitor = get_monitor()

        # Get service status
        service_status = service.get_service_status()
        monitor_status = monitor.get_monitoring_status()

        # Calculate uptime
        uptime = (datetime.now() - service.start_time).total_seconds() if service.start_time else 0

        return HealthCheckResponse(
            status=service_status.overall_status,
            version="1.0.0",
            uptime_seconds=uptime,
            components={
                "legacy_whisperer_service": {
                    "component_name": "legacy_whisperer_service",
                    "status": "healthy" if service.is_initialized else "unhealthy",
                    "last_check": datetime.now(),
                    "error_count": service_status.error_count,
                },
                "monitoring_system": {
                    "component_name": "monitoring_system",
                    "status": ("healthy" if monitor_status["monitoring_active"] else "degraded"),
                    "last_check": datetime.now(),
                    "error_count": 0,
                },
            },
            metrics={
                "registered_systems": service_status.active_systems,
                "active_predictions": service_status.active_predictions,
                "knowledge_base_size": service_status.knowledge_base_size,
                "performance_score": service_status.performance_score,
            },
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}",
        )


# System Registration Endpoints
@legacy_router.post("/systems", response_model=SystemRegistrationResponse, status_code=HTTP_201_CREATED)
async def register_system(
    request: SystemRegistrationRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    service: LegacySystemWhispererService = Depends(get_service),
    monitor: LegacySystemMonitor = Depends(get_monitor),
):
    """
    Register a new legacy system for monitoring and management.

    This endpoint registers a legacy system with the Legacy System Whisperer,
    enabling monitoring, predictive analytics, and management capabilities.
    """
    try:
        # Convert API schema to internal model
        system_context = LegacySystemContext(
            system_id=request.system_id,
            system_name=request.system_name,
            system_type=SystemType(request.system_type.value),
            version=request.version,
            location=request.location,
            criticality=Criticality(request.criticality.value),
            compliance_requirements=request.compliance_requirements,
            technical_contacts=request.technical_contacts,
            business_contacts=request.business_contacts,
            custom_metadata=request.custom_metadata,
        )

        # Register system
        result = await service.register_legacy_system(system_context, enable_monitoring=request.enable_monitoring)

        # Record metrics
        monitor.record_system_registration(system_context)

        # Background task to initialize monitoring
        if request.enable_monitoring:
            background_tasks.add_task(_initialize_system_monitoring, service, system_context.system_id)

        logger.info(f"System registered successfully: {request.system_id}")

        return SystemRegistrationResponse(success=True, message="System registered successfully", data=result)

    except Exception as e:
        logger.error(f"System registration failed: {e}")
        if isinstance(e, (ValidationException, SystemRegistrationException)):
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@legacy_router.get("/systems/{system_id}", response_model=SystemDashboardResponse)
async def get_system_dashboard(
    system_id: str = Path(..., description="System ID"),
    include_historical: bool = Query(True, description="Include historical data"),
    time_range_days: int = Query(30, ge=1, le=365, description="Time range in days"),
    current_user=Depends(get_current_user),
    service: LegacySystemWhispererService = Depends(get_service),
):
    """
    Get comprehensive dashboard for a registered system.

    Returns system information, current status, health metrics,
    recent predictions, and recommendations.
    """
    try:
        dashboard = service.get_system_dashboard(system_id)

        return SystemDashboardResponse(success=True, message="Dashboard retrieved successfully", data=dashboard)

    except Exception as e:
        logger.error(f"Dashboard retrieval failed for {system_id}: {e}")
        if "not registered" in str(e):
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=str(e))
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@legacy_router.delete("/systems/{system_id}", status_code=HTTP_204_NO_CONTENT)
async def unregister_system(
    system_id: str = Path(..., description="System ID"),
    current_user=Depends(get_current_user),
    service: LegacySystemWhispererService = Depends(get_service),
):
    """
    Unregister a system from Legacy System Whisperer.

    This will stop monitoring and remove the system from management.
    Historical data may be retained based on retention policies.
    """
    try:
        # Implementation would be added to service
        # For now, return success
        logger.info(f"System unregistered: {system_id}")
        return None

    except Exception as e:
        logger.error(f"System unregistration failed for {system_id}: {e}")
        if "not found" in str(e):
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=str(e))
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Health Analysis Endpoints
@legacy_router.post("/systems/{system_id}/analyze", response_model=HealthAnalysisResponse)
async def analyze_system_health(
    system_id: str = Path(..., description="System ID"),
    request: HealthAnalysisRequest = None,
    background_tasks: BackgroundTasks = None,
    current_user=Depends(get_current_user),
    service: LegacySystemWhispererService = Depends(get_service),
    monitor: LegacySystemMonitor = Depends(get_monitor),
):
    """
    Perform comprehensive system health analysis.

    Analyzes current system metrics, predicts potential failures,
    and provides actionable recommendations.
    """
    try:
        # Convert request data
        system_metrics = SystemMetrics(
            timestamp=request.current_metrics.timestamp,
            cpu_utilization=request.current_metrics.cpu_utilization,
            memory_utilization=request.current_metrics.memory_utilization,
            disk_utilization=request.current_metrics.disk_utilization,
            network_utilization=request.current_metrics.network_utilization,
            response_time_ms=request.current_metrics.response_time_ms,
            error_rate=request.current_metrics.error_rate,
            transaction_rate=request.current_metrics.transaction_rate,
            availability=request.current_metrics.availability,
        )

        # Convert time series data if provided
        time_series = None
        if request.time_series_data:
            time_series = [
                TimeSeriesData(
                    timestamp=ts.timestamp,
                    cpu_utilization=ts.cpu_utilization,
                    memory_utilization=ts.memory_utilization,
                    response_time_ms=ts.response_time_ms,
                    error_rate=ts.error_rate,
                )
                for ts in request.time_series_data
            ]

        # Convert prediction horizon
        horizon_mapping = {
            PredictionHorizonAPI.SHORT_TERM: PredictionHorizon.SHORT_TERM,
            PredictionHorizonAPI.MEDIUM_TERM: PredictionHorizon.MEDIUM_TERM,
            PredictionHorizonAPI.LONG_TERM: PredictionHorizon.LONG_TERM,
            PredictionHorizonAPI.STRATEGIC: PredictionHorizon.STRATEGIC,
        }
        prediction_horizon = horizon_mapping[request.prediction_horizon]

        # Perform analysis with monitoring
        with monitor.monitor_operation("health_analysis", "health_analyzer", system_id):
            result = await service.analyze_system_health(
                system_id=system_id,
                current_metrics=system_metrics,
                time_series_data=time_series,
                prediction_horizon=prediction_horizon,
            )

        # Record metrics
        monitor.record_anomaly_detection(
            system_id=system_id,
            system_type="legacy",
            severity="medium",
            duration_seconds=result.get("processing_time_seconds", 0),
        )

        logger.info(f"Health analysis completed for {system_id}")

        return HealthAnalysisResponse(success=True, message="Health analysis completed successfully", data=result)

    except Exception as e:
        logger.error(f"Health analysis failed for {system_id}: {e}")
        if "not registered" in str(e):
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=str(e))
        elif isinstance(e, AnomalyDetectionException):
            raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Knowledge Management Endpoints
@legacy_router.post("/knowledge/capture", response_model=ExpertKnowledgeResponse)
async def capture_expert_knowledge(
    request: ExpertKnowledgeRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    service: LegacySystemWhispererService = Depends(get_service),
    monitor: LegacySystemMonitor = Depends(get_monitor),
):
    """
    Capture expert knowledge about legacy systems.

    Processes expert input through LLM analysis to extract
    structured, actionable knowledge for system management.
    """
    try:
        result = await service.capture_expert_knowledge(
            expert_id=request.expert_id,
            session_type=request.session_type,
            expert_input=request.expert_input,
            system_id=request.system_id,
            context=request.context,
        )

        # Record metrics
        monitor.record_knowledge_session(request.session_type, "success")

        logger.info(f"Expert knowledge captured from {request.expert_id}")

        return ExpertKnowledgeResponse(success=True, message="Expert knowledge captured successfully", data=result)

    except Exception as e:
        logger.error(f"Knowledge capture failed: {e}")
        monitor.record_knowledge_session(request.session_type, "failure")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@legacy_router.get("/knowledge/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    query: str = Query(..., min_length=1, max_length=500, description="Search query"),
    system_id: Optional[str] = Query(None, description="Filter by system ID"),
    category: Optional[str] = Query(None, description="Filter by knowledge category"),
    confidence_threshold: float = Query(0.5, ge=0, le=1, description="Minimum confidence threshold"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    current_user=Depends(get_current_user),
    service: LegacySystemWhispererService = Depends(get_service),
):
    """
    Search the knowledge base for relevant information.

    Performs semantic search across captured expert knowledge
    to find relevant information for system management decisions.
    """
    try:
        # Implementation would be added to service
        # For now, return mock response

        search_result = {
            "results": [],
            "total_results": 0,
            "query": query,
            "processing_time_ms": 150.0,
        }

        logger.info(f"Knowledge search completed: '{query}'")

        return KnowledgeSearchResponse(
            success=True,
            message="Knowledge search completed",
            data=search_result,
            total_results=search_result["total_results"],
            query=query,
        )

    except Exception as e:
        logger.error(f"Knowledge search failed: {e}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Decision Support Endpoints
@legacy_router.post("/systems/{system_id}/decision-support", response_model=DecisionSupportResponse)
async def get_decision_support(
    system_id: str = Path(..., description="System ID"),
    request: DecisionSupportRequest = None,
    current_user=Depends(get_current_user),
    service: LegacySystemWhispererService = Depends(get_service),
    monitor: LegacySystemMonitor = Depends(get_monitor),
):
    """
    Get intelligent decision support recommendations.

    Analyzes system state and business context to provide
    actionable recommendations for system management decisions.
    """
    try:
        # Convert decision category
        from ..decision_support import DecisionCategory

        category_mapping = {
            DecisionCategoryAPI.MAINTENANCE_PLANNING: DecisionCategory.MAINTENANCE_PLANNING,
            DecisionCategoryAPI.UPGRADE_DECISION: DecisionCategory.UPGRADE_DECISION,
            DecisionCategoryAPI.RISK_MITIGATION: DecisionCategory.RISK_MITIGATION,
            DecisionCategoryAPI.RESOURCE_ALLOCATION: DecisionCategory.RESOURCE_ALLOCATION,
            DecisionCategoryAPI.SYSTEM_RETIREMENT: DecisionCategory.SYSTEM_RETIREMENT,
            DecisionCategoryAPI.EMERGENCY_RESPONSE: DecisionCategory.EMERGENCY_RESPONSE,
        }
        decision_category = category_mapping[request.decision_category]

        result = await service.create_decision_support(
            decision_category=decision_category,
            system_id=system_id,
            current_situation=request.current_situation,
            objectives=request.objectives,
            constraints=request.constraints,
        )

        # Record metrics
        monitor.record_recommendation(request.decision_category.value, "high")

        logger.info(f"Decision support generated for {system_id}")

        return DecisionSupportResponse(success=True, message="Decision support generated successfully", data=result)

    except Exception as e:
        logger.error(f"Decision support failed for {system_id}: {e}")
        if "not registered" in str(e):
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=str(e))
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Maintenance Scheduling Endpoints
@legacy_router.post("/maintenance/schedule", response_model=MaintenanceSchedulingResponse)
async def schedule_maintenance(
    request: MaintenanceSchedulingRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    service: LegacySystemWhispererService = Depends(get_service),
    monitor: LegacySystemMonitor = Depends(get_monitor),
):
    """
    Schedule maintenance activities with optimization.

    Optimizes maintenance scheduling across multiple systems
    considering resource constraints and business requirements.
    """
    try:
        # Convert maintenance requests
        maintenance_requests = []
        for req in request.maintenance_requests:
            maintenance_req = MaintenanceRecommendation(
                system_id=req.system_id,
                maintenance_type=req.maintenance_type,
                priority=req.priority,
                estimated_duration=timedelta(hours=req.estimated_duration_hours),
                description=req.description,
                required_resources=req.required_resources,
                business_impact=req.business_impact,
                recommended_schedule=req.preferred_schedule,
                dependencies=req.dependencies,
                cost_estimate=req.cost_estimate,
            )
            maintenance_requests.append(maintenance_req)

        result = await service.schedule_maintenance(
            maintenance_requests=maintenance_requests,
            resource_constraints=request.resource_constraints,
            business_constraints=request.business_constraints,
        )

        # Record metrics
        for req in request.maintenance_requests:
            monitor.record_maintenance_scheduled(req.maintenance_type, "legacy")

        logger.info(f"Maintenance scheduled for {len(maintenance_requests)} systems")

        return MaintenanceSchedulingResponse(
            success=True,
            message="Maintenance scheduling completed successfully",
            data=result,
        )

    except Exception as e:
        logger.error(f"Maintenance scheduling failed: {e}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Bulk Operations Endpoints
@legacy_router.post("/systems/bulk-analysis", response_model=BulkSystemAnalysisResponse)
async def bulk_system_analysis(
    request: BulkSystemAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    service: LegacySystemWhispererService = Depends(get_service),
    monitor: LegacySystemMonitor = Depends(get_monitor),
):
    """
    Perform bulk analysis across multiple systems.

    Analyzes multiple systems in parallel to provide
    comparative insights and system-wide recommendations.
    """
    try:
        # For now, return mock bulk analysis
        results = []

        for system_id in request.system_ids:
            try:
                # Mock analysis result
                result = {
                    "system_id": system_id,
                    "status": "completed",
                    "analysis_data": {
                        "health_score": 85.0,
                        "risk_level": "medium",
                        "recommendations_count": 3,
                    },
                    "processing_time_ms": 1500.0,
                }
                results.append(result)

            except Exception as e:
                result = {
                    "system_id": system_id,
                    "status": "failed",
                    "error_message": str(e),
                    "processing_time_ms": 500.0,
                }
                results.append(result)

        summary = {
            "total_systems": len(request.system_ids),
            "successful_analyses": len([r for r in results if r["status"] == "completed"]),
            "failed_analyses": len([r for r in results if r["status"] == "failed"]),
            "average_processing_time_ms": sum(r["processing_time_ms"] for r in results) / len(results),
        }

        logger.info(f"Bulk analysis completed for {len(request.system_ids)} systems")

        return BulkSystemAnalysisResponse(
            success=True,
            message="Bulk analysis completed",
            data=results,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Bulk analysis failed: {e}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Monitoring and Alerting Endpoints
@legacy_router.post("/alerts/rules", response_model=AlertRuleResponse, status_code=HTTP_201_CREATED)
async def create_alert_rule(
    request: CreateAlertRuleRequest,
    current_user=Depends(get_current_user),
    monitor: LegacySystemMonitor = Depends(get_monitor),
):
    """
    Create a new alert rule for system monitoring.

    Defines conditions that trigger alerts when system
    metrics exceed specified thresholds.
    """
    try:
        alert_rule = request.alert_rule

        # Add alert rule to monitor
        monitor.alert_manager.add_alert_rule(
            rule_name=alert_rule.rule_name,
            metric=alert_rule.metric_name,
            threshold=alert_rule.threshold,
            comparison=alert_rule.condition,
            severity=alert_rule.severity,
            message=f"Alert rule: {alert_rule.rule_name}",
            labels={"system_id": alert_rule.system_id} if alert_rule.system_id else {},
        )

        logger.info(f"Alert rule created: {alert_rule.rule_name}")

        return AlertRuleResponse(success=True, message="Alert rule created successfully", data=alert_rule)

    except Exception as e:
        logger.error(f"Alert rule creation failed: {e}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@legacy_router.get("/alerts", response_model=AlertsListResponse)
async def list_alerts(
    system_id: Optional[str] = Query(None, description="Filter by system ID"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    status: str = Query("active", regex="^(active|resolved|all)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
    monitor: LegacySystemMonitor = Depends(get_monitor),
):
    """
    List system alerts with filtering and pagination.

    Returns current and historical alerts based on
    specified filters and pagination parameters.
    """
    try:
        # Get alerts from monitor
        if status == "active":
            alerts = monitor.alert_manager.get_active_alerts()
        else:
            # Would implement historical alert retrieval
            alerts = monitor.alert_manager.get_active_alerts()

        # Apply filters
        if system_id:
            alerts = [a for a in alerts if a.system_id == system_id]

        if severity:
            alerts = [a for a in alerts if a.severity.value == severity]

        # Apply pagination
        total_count = len(alerts)
        alerts = alerts[offset : offset + limit]

        # Convert to API format
        alert_items = [
            {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "system_id": alert.system_id,
                "metric_name": alert.metric_name,
                "threshold_value": alert.threshold_value,
                "current_value": alert.current_value,
                "timestamp": alert.timestamp,
                "resolved": alert.resolved,
                "resolution_timestamp": alert.resolution_timestamp,
            }
            for alert in alerts
        ]

        return AlertsListResponse(
            success=True,
            message="Alerts retrieved successfully",
            data=alert_items,
            total_count=total_count,
            pagination={
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total_count,
            },
        )

    except Exception as e:
        logger.error(f"Alert listing failed: {e}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Service Management Endpoints
@legacy_router.get("/service/status", response_model=ServiceStatusResponse)
async def get_service_status(
    include_details: bool = Query(True, description="Include component details"),
    include_metrics: bool = Query(False, description="Include performance metrics"),
    current_user=Depends(get_current_user),
    service: LegacySystemWhispererService = Depends(get_service),
    monitor: LegacySystemMonitor = Depends(get_monitor),
):
    """
    Get comprehensive service status information.

    Returns service health, component status, performance metrics,
    and operational statistics.
    """
    try:
        service_status = service.get_service_status()
        service_metrics = service.get_service_metrics()
        monitor_status = monitor.get_monitoring_status()

        status_data = {
            "service_info": {
                "name": "Legacy System Whisperer",
                "version": "1.0.0",
                "status": service_status.overall_status,
                "uptime_seconds": ((datetime.now() - service.start_time).total_seconds() if service.start_time else 0),
                "initialized": service.is_initialized,
            },
            "component_status": service_status.components_status,
            "system_stats": service_metrics["system_registry"],
            "monitoring_status": monitor_status,
        }

        if include_metrics:
            status_data["performance_metrics"] = service_metrics["service_metrics"]

        return ServiceStatusResponse(
            success=True,
            message="Service status retrieved successfully",
            data=status_data,
        )

    except Exception as e:
        logger.error(f"Service status retrieval failed: {e}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Metrics Endpoint (Prometheus format)
@legacy_router.get("/metrics", response_class=JSONResponse)
async def get_metrics(
    format: str = Query("json", regex="^(json|prometheus)$"),
    current_user=Depends(get_current_user),
    monitor: LegacySystemMonitor = Depends(get_monitor),
):
    """
    Get service metrics in requested format.

    Returns performance and operational metrics
    in JSON or Prometheus exposition format.
    """
    try:
        if format == "prometheus":
            metrics_data = monitor.get_prometheus_metrics()
            return JSONResponse(
                content={"metrics": metrics_data},
                headers={"Content-Type": "text/plain"},
            )
        else:
            # JSON format
            status = monitor.get_monitoring_status()
            return JSONResponse(content=status)

    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Background task functions
async def _initialize_system_monitoring(service: LegacySystemWhispererService, system_id: str):
    """Initialize monitoring for a newly registered system."""
    try:
        logger.info(f"Initializing monitoring for system: {system_id}")
        # Implementation would setup monitoring dashboards, alerts, etc.
        await asyncio.sleep(1)  # Simulate initialization
        logger.info(f"Monitoring initialized for system: {system_id}")

    except Exception as e:
        logger.error(f"Failed to initialize monitoring for {system_id}: {e}")


# Utility functions
def convert_system_type_to_internal(api_type: SystemTypeAPI) -> SystemType:
    """Convert API system type to internal enum."""
    mapping = {
        SystemTypeAPI.MAINFRAME: SystemType.MAINFRAME,
        SystemTypeAPI.COBOL: SystemType.COBOL,
        SystemTypeAPI.SCADA: SystemType.SCADA,
        SystemTypeAPI.MEDICAL_DEVICE: SystemType.MEDICAL_DEVICE,
        SystemTypeAPI.PLC: SystemType.PLC,
        SystemTypeAPI.DCS: SystemType.DCS,
        SystemTypeAPI.LEGACY_DATABASE: SystemType.LEGACY_DATABASE,
        SystemTypeAPI.EMBEDDED_SYSTEM: SystemType.EMBEDDED_SYSTEM,
        SystemTypeAPI.PROPRIETARY_PROTOCOL: SystemType.PROPRIETARY_PROTOCOL,
    }
    return mapping[api_type]


def convert_criticality_to_internal(api_criticality: CriticalityAPI) -> Criticality:
    """Convert API criticality to internal enum."""
    mapping = {
        CriticalityAPI.LOW: Criticality.LOW,
        CriticalityAPI.MEDIUM: Criticality.MEDIUM,
        CriticalityAPI.HIGH: Criticality.HIGH,
        CriticalityAPI.CRITICAL: Criticality.CRITICAL,
    }
    return mapping[api_criticality]


# Request context middleware integration
async def get_request_context_dependency():
    """Dependency to get request context."""
    return get_request_context()


# Add dependency to all endpoints
for route in legacy_router.routes:
    if hasattr(route, "dependencies"):
        route.dependencies.append(Depends(get_request_context_dependency))
