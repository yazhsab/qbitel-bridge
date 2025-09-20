"""
CRONOS AI Engine - Legacy System Whisperer API Schemas

Pydantic schemas for API request/response validation.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator

from ..models import SystemType, Criticality, FailureType, SeverityLevel, MaintenanceType, PredictionHorizon


# Enums for API
class SystemTypeAPI(str, Enum):
    """API enum for system types."""
    MAINFRAME = "mainframe"
    COBOL = "cobol"
    SCADA = "scada"
    MEDICAL_DEVICE = "medical_device"
    PLC = "plc"
    DCS = "dcs"
    LEGACY_DATABASE = "legacy_database"
    EMBEDDED_SYSTEM = "embedded_system"
    PROPRIETARY_PROTOCOL = "proprietary_protocol"


class CriticalityAPI(str, Enum):
    """API enum for system criticality."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PredictionHorizonAPI(str, Enum):
    """API enum for prediction horizons."""
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    STRATEGIC = "strategic"


class DecisionCategoryAPI(str, Enum):
    """API enum for decision categories."""
    MAINTENANCE_PLANNING = "maintenance_planning"
    UPGRADE_DECISION = "upgrade_decision"
    RISK_MITIGATION = "risk_mitigation"
    RESOURCE_ALLOCATION = "resource_allocation"
    SYSTEM_RETIREMENT = "system_retirement"
    EMERGENCY_RESPONSE = "emergency_response"


# Base API Response Schema
class APIResponse(BaseModel):
    """Base API response schema."""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class APIError(APIResponse):
    """API error response schema."""
    success: bool = False
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


# System Registration Schemas
class SystemRegistrationRequest(BaseModel):
    """Request schema for system registration."""
    system_id: str = Field(..., min_length=1, max_length=100)
    system_name: str = Field(..., min_length=1, max_length=200)
    system_type: SystemTypeAPI
    version: Optional[str] = Field(None, max_length=50)
    location: Optional[str] = Field(None, max_length=100)
    criticality: CriticalityAPI = CriticalityAPI.MEDIUM
    compliance_requirements: List[str] = Field(default_factory=list)
    technical_contacts: List[str] = Field(default_factory=list)
    business_contacts: List[str] = Field(default_factory=list)
    enable_monitoring: bool = True
    custom_metadata: Optional[Dict[str, Any]] = None

    @validator('technical_contacts', 'business_contacts')
    def validate_email_contacts(cls, v):
        """Validate email format in contact lists."""
        import re
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        for email in v:
            if not email_pattern.match(email):
                raise ValueError(f'Invalid email format: {email}')
        return v


class SystemRegistrationResponse(APIResponse):
    """Response schema for system registration."""
    data: Dict[str, Any] = Field(..., description="Registration details")


# System Metrics Schemas
class SystemMetricsRequest(BaseModel):
    """Request schema for system metrics."""
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    cpu_utilization: float = Field(..., ge=0, le=100)
    memory_utilization: float = Field(..., ge=0, le=100)
    disk_utilization: Optional[float] = Field(None, ge=0, le=100)
    network_utilization: Optional[float] = Field(None, ge=0, le=100)
    response_time_ms: Optional[float] = Field(None, ge=0)
    error_rate: Optional[float] = Field(None, ge=0, le=1)
    transaction_rate: Optional[float] = Field(None, ge=0)
    availability: Optional[float] = Field(None, ge=0, le=1)
    custom_metrics: Optional[Dict[str, float]] = None


class TimeSeriesDataPoint(BaseModel):
    """Time series data point schema."""
    timestamp: datetime
    cpu_utilization: Optional[float] = Field(None, ge=0, le=100)
    memory_utilization: Optional[float] = Field(None, ge=0, le=100)
    response_time_ms: Optional[float] = Field(None, ge=0)
    error_rate: Optional[float] = Field(None, ge=0, le=1)
    custom_metrics: Optional[Dict[str, float]] = None


# Health Analysis Schemas
class HealthAnalysisRequest(BaseModel):
    """Request schema for system health analysis."""
    system_id: str
    current_metrics: SystemMetricsRequest
    time_series_data: Optional[List[TimeSeriesDataPoint]] = None
    prediction_horizon: PredictionHorizonAPI = PredictionHorizonAPI.MEDIUM_TERM
    include_recommendations: bool = True
    analysis_depth: str = Field(default="standard", regex="^(basic|standard|comprehensive)$")


class FailurePredictionData(BaseModel):
    """Failure prediction data schema."""
    system_id: str
    failure_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    predicted_failure_types: List[str]
    time_to_failure_days: Optional[int] = Field(None, ge=0)
    severity: str
    contributing_factors: List[str]
    recommended_actions: List[str]
    prediction_timestamp: datetime


class PerformanceAnalysisData(BaseModel):
    """Performance analysis data schema."""
    performance_score: float = Field(..., ge=0, le=100)
    status: str
    degradation_indicators: List[str]
    trend_analysis: Optional[Dict[str, Any]] = None
    baseline_comparison: Optional[Dict[str, Any]] = None


class HealthAnalysisResponse(APIResponse):
    """Response schema for health analysis."""
    data: Dict[str, Any] = Field(..., description="Health analysis results")


# Knowledge Capture Schemas
class ExpertKnowledgeRequest(BaseModel):
    """Request schema for expert knowledge capture."""
    expert_id: str = Field(..., min_length=1, max_length=100)
    session_type: str = Field(..., min_length=1, max_length=50)
    expert_input: str = Field(..., min_length=1, max_length=10000)
    system_id: Optional[str] = None
    knowledge_category: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    validate_knowledge: bool = True


class ExpertKnowledgeResponse(APIResponse):
    """Response schema for expert knowledge capture."""
    data: Dict[str, Any] = Field(..., description="Knowledge capture results")


class KnowledgeSearchRequest(BaseModel):
    """Request schema for knowledge search."""
    query: str = Field(..., min_length=1, max_length=500)
    system_id: Optional[str] = None
    knowledge_category: Optional[str] = None
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    max_results: int = Field(default=10, ge=1, le=100)
    include_metadata: bool = True


class KnowledgeItem(BaseModel):
    """Knowledge item schema."""
    knowledge_id: str
    title: str
    content: str
    category: str
    confidence_level: str
    source: str
    relevance_score: Optional[float] = None
    related_systems: List[str]
    tags: List[str]
    creation_timestamp: datetime
    last_updated: datetime


class KnowledgeSearchResponse(APIResponse):
    """Response schema for knowledge search."""
    data: Dict[str, Any] = Field(..., description="Knowledge search results")
    total_results: int
    query: str


# Decision Support Schemas
class DecisionSupportRequest(BaseModel):
    """Request schema for decision support."""
    decision_category: DecisionCategoryAPI
    system_id: str
    current_situation: Dict[str, Any]
    objectives: List[str]
    constraints: Optional[Dict[str, Any]] = None
    risk_tolerance: str = Field(default="medium", regex="^(low|medium|high)$")
    stakeholders: Optional[List[str]] = None
    timeline: Optional[str] = None
    budget_constraints: Optional[float] = Field(None, ge=0)


class ActionRecommendation(BaseModel):
    """Action recommendation schema."""
    action_id: str
    title: str
    description: str
    priority: str
    estimated_effort: Optional[str] = None
    estimated_cost: Optional[float] = None
    expected_benefits: List[str]
    risks: List[str]
    dependencies: List[str]
    timeline: Optional[str] = None
    resource_requirements: List[str]


class BusinessImpactData(BaseModel):
    """Business impact data schema."""
    financial_impact: Optional[float] = None
    operational_impact: str
    compliance_impact: Optional[str] = None
    risk_level: str
    stakeholder_impact: Dict[str, str]
    timeline_impact: Optional[str] = None


class DecisionSupportResponse(APIResponse):
    """Response schema for decision support."""
    data: Dict[str, Any] = Field(..., description="Decision support results")


# Maintenance Scheduling Schemas
class MaintenanceRequest(BaseModel):
    """Maintenance request schema."""
    system_id: str
    maintenance_type: str
    priority: str = Field(..., regex="^(low|medium|high|critical)$")
    description: str = Field(..., min_length=1, max_length=1000)
    estimated_duration_hours: float = Field(..., gt=0)
    required_resources: List[str]
    preferred_schedule: Optional[datetime] = None
    deadline: Optional[datetime] = None
    business_impact: Optional[str] = None
    cost_estimate: Optional[float] = Field(None, ge=0)
    dependencies: List[str] = Field(default_factory=list)


class MaintenanceSchedulingRequest(BaseModel):
    """Request schema for maintenance scheduling."""
    maintenance_requests: List[MaintenanceRequest]
    resource_constraints: Optional[Dict[str, Any]] = None
    business_constraints: Optional[Dict[str, Any]] = None
    optimization_criteria: List[str] = Field(default_factory=lambda: ["minimize_cost", "maximize_availability"])
    scheduling_horizon_days: int = Field(default=90, ge=1, le=365)


class ScheduledMaintenanceItem(BaseModel):
    """Scheduled maintenance item schema."""
    maintenance_id: str
    system_id: str
    maintenance_type: str
    scheduled_start: datetime
    scheduled_end: datetime
    assigned_resources: List[str]
    optimization_score: float
    conflicts_resolved: int
    priority_score: float


class MaintenanceSchedulingResponse(APIResponse):
    """Response schema for maintenance scheduling."""
    data: Dict[str, Any] = Field(..., description="Maintenance scheduling results")


# Dashboard and Reporting Schemas
class SystemDashboardRequest(BaseModel):
    """Request schema for system dashboard."""
    system_id: str
    include_historical_data: bool = True
    time_range_days: int = Field(default=30, ge=1, le=365)
    include_predictions: bool = True
    include_recommendations: bool = True


class DashboardSummary(BaseModel):
    """Dashboard summary schema."""
    system_info: Dict[str, Any]
    current_status: str
    health_score: float
    last_analysis: Optional[datetime] = None
    active_alerts: int
    scheduled_maintenance: int
    recent_recommendations: int
    knowledge_base_items: int


class SystemDashboardResponse(APIResponse):
    """Response schema for system dashboard."""
    data: DashboardSummary


class ServiceStatusRequest(BaseModel):
    """Request schema for service status."""
    include_component_details: bool = True
    include_metrics: bool = False
    include_recent_activity: bool = False


class ComponentStatus(BaseModel):
    """Component status schema."""
    component_name: str
    status: str
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_count: int = 0
    details: Optional[Dict[str, Any]] = None


class ServiceStatusResponse(APIResponse):
    """Response schema for service status."""
    data: Dict[str, Any] = Field(..., description="Service status information")


# Bulk Operations Schemas
class BulkSystemAnalysisRequest(BaseModel):
    """Request schema for bulk system analysis."""
    system_ids: List[str] = Field(..., min_items=1, max_items=50)
    analysis_type: str = Field(default="health", regex="^(health|performance|risk|compliance)$")
    prediction_horizon: PredictionHorizonAPI = PredictionHorizonAPI.MEDIUM_TERM
    include_recommendations: bool = True
    parallel_processing: bool = True


class BulkAnalysisResult(BaseModel):
    """Bulk analysis result schema."""
    system_id: str
    status: str
    analysis_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[float] = None


class BulkSystemAnalysisResponse(APIResponse):
    """Response schema for bulk system analysis."""
    data: List[BulkAnalysisResult]
    summary: Dict[str, Any]


# Configuration Schemas
class SystemConfigurationUpdate(BaseModel):
    """System configuration update schema."""
    system_id: str
    configuration_updates: Dict[str, Any]
    update_reason: str = Field(..., min_length=1, max_length=500)
    effective_date: Optional[datetime] = None
    rollback_plan: Optional[str] = None


class ConfigurationUpdateResponse(APIResponse):
    """Response schema for configuration updates."""
    data: Dict[str, Any] = Field(..., description="Configuration update results")


# Export/Import Schemas
class SystemExportRequest(BaseModel):
    """Request schema for system data export."""
    system_ids: List[str] = Field(..., min_items=1)
    export_types: List[str] = Field(..., min_items=1)  # e.g., ["configuration", "history", "patterns"]
    date_range: Optional[Dict[str, datetime]] = None
    format: str = Field(default="json", regex="^(json|csv|xml)$")
    include_sensitive_data: bool = False


class SystemExportResponse(APIResponse):
    """Response schema for system data export."""
    data: Dict[str, Any] = Field(..., description="Export data")
    export_id: str
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None


class SystemImportRequest(BaseModel):
    """Request schema for system data import."""
    import_data: Dict[str, Any]
    import_type: str = Field(..., regex="^(configuration|history|patterns|knowledge)$")
    validation_mode: str = Field(default="strict", regex="^(strict|lenient|skip)$")
    merge_strategy: str = Field(default="update", regex="^(update|replace|merge)$")
    dry_run: bool = False


class ImportResult(BaseModel):
    """Import result schema."""
    item_id: str
    status: str
    message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class SystemImportResponse(APIResponse):
    """Response schema for system data import."""
    data: List[ImportResult]
    summary: Dict[str, Any]


# Monitoring and Alerting Schemas
class AlertRule(BaseModel):
    """Alert rule schema."""
    rule_id: Optional[str] = None
    rule_name: str = Field(..., min_length=1, max_length=100)
    system_id: Optional[str] = None
    metric_name: str
    condition: str = Field(..., regex="^(greater_than|less_than|equals|not_equals)$")
    threshold: Union[float, str]
    severity: str = Field(..., regex="^(info|warning|critical|emergency)$")
    enabled: bool = True
    notification_channels: List[str] = Field(default_factory=list)
    cooldown_minutes: int = Field(default=15, ge=1)


class CreateAlertRuleRequest(BaseModel):
    """Request schema for creating alert rules."""
    alert_rule: AlertRule


class AlertRuleResponse(APIResponse):
    """Response schema for alert rule operations."""
    data: AlertRule


class AlertsListRequest(BaseModel):
    """Request schema for listing alerts."""
    system_id: Optional[str] = None
    severity: Optional[str] = None
    status: str = Field(default="active", regex="^(active|resolved|all)$")
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="timestamp", regex="^(timestamp|severity|system_id)$")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$")


class AlertItem(BaseModel):
    """Alert item schema."""
    alert_id: str
    title: str
    description: str
    severity: str
    system_id: Optional[str] = None
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    timestamp: datetime
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


class AlertsListResponse(APIResponse):
    """Response schema for alerts listing."""
    data: List[AlertItem]
    total_count: int
    pagination: Dict[str, Any]


# Validation and Error Schemas
class ValidationError(BaseModel):
    """Validation error schema."""
    field: str
    message: str
    invalid_value: Optional[Any] = None


class ValidationErrorResponse(APIError):
    """Validation error response schema."""
    error_code: str = "VALIDATION_ERROR"
    validation_errors: List[ValidationError]


class RateLimitErrorResponse(APIError):
    """Rate limit error response schema."""
    error_code: str = "RATE_LIMIT_EXCEEDED"
    retry_after_seconds: int
    limit: int
    window_seconds: int


class AuthenticationErrorResponse(APIError):
    """Authentication error response schema."""
    error_code: str = "AUTHENTICATION_FAILED"
    auth_scheme: str
    realm: Optional[str] = None


class AuthorizationErrorResponse(APIError):
    """Authorization error response schema."""
    error_code: str = "AUTHORIZATION_FAILED"
    required_permission: str
    user_permissions: List[str]


# Health Check Schema
class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., regex="^(healthy|degraded|unhealthy)$")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    uptime_seconds: float
    components: Dict[str, ComponentStatus]
    metrics: Optional[Dict[str, float]] = None


# API Metadata Schema
class APIMetadata(BaseModel):
    """API metadata schema."""
    version: str
    title: str = "Legacy System Whisperer API"
    description: str = "REST API for Legacy System Management and Predictive Analytics"
    contact: Dict[str, str] = Field(default_factory=lambda: {
        "name": "CRONOS AI Engine Team",
        "email": "support@cronos-ai.com"
    })
    license: Dict[str, str] = Field(default_factory=lambda: {
        "name": "Proprietary",
        "url": "https://cronos-ai.com/license"
    })
    supported_features: List[str] = Field(default_factory=lambda: [
        "system_registration",
        "health_analysis", 
        "failure_prediction",
        "knowledge_capture",
        "decision_support",
        "maintenance_scheduling",
        "monitoring_alerts"
    ])


# Utility functions for schema validation
def validate_system_id(system_id: str) -> bool:
    """Validate system ID format."""
    import re
    pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
    return bool(pattern.match(system_id) and len(system_id) <= 100)


def validate_time_range(start_time: datetime, end_time: datetime) -> bool:
    """Validate time range."""
    return start_time < end_time and (end_time - start_time).days <= 365


def validate_metric_value(value: float, metric_type: str) -> bool:
    """Validate metric value based on type."""
    if metric_type in ["cpu_utilization", "memory_utilization", "disk_utilization", "network_utilization"]:
        return 0 <= value <= 100
    elif metric_type in ["error_rate", "availability"]:
        return 0 <= value <= 1
    elif metric_type in ["response_time_ms", "transaction_rate"]:
        return value >= 0
    return True


# Create custom validator decorators
def validate_email_list(email_list: List[str]) -> List[str]:
    """Validate list of email addresses."""
    import re
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    for email in email_list:
        if not email_pattern.match(email):
            raise ValueError(f'Invalid email format: {email}')
    
    return email_list


def validate_json_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate JSON data structure."""
    import json
    
    try:
        # Attempt to serialize and deserialize to validate JSON structure
        json.dumps(data)
        return data
    except (TypeError, ValueError) as e:
        raise ValueError(f'Invalid JSON data: {e}')