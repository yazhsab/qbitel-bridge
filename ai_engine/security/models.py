"""
CRONOS AI Engine - Zero-Touch Security Orchestrator Models

Enterprise-grade data models for autonomous security operations.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from datetime import datetime, timedelta
import ipaddress
import uuid

from ..core.exceptions import CronosAIException


class SecurityEventType(str, Enum):
    """Types of security events."""

    MALWARE_DETECTION = "malware_detection"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    DATA_EXFILTRATION = "data_exfiltration"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    VULNERABILITY_EXPLOIT = "vulnerability_exploit"
    INSIDER_THREAT = "insider_threat"
    RANSOMWARE_ACTIVITY = "ransomware_activity"
    PROTOCOL_VIOLATION = "protocol_violation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_NETWORK_ACTIVITY = "suspicious_network_activity"
    CONFIGURATION_CHANGE = "configuration_change"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    COMMAND_AND_CONTROL = "command_and_control"
    ZERO_DAY_EXPLOIT = "zero_day_exploit"


class ThreatLevel(str, Enum):
    """Threat severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ConfidenceLevel(str, Enum):
    """Confidence levels for automated decisions."""

    VERY_HIGH = "very_high"  # 95%+
    HIGH = "high"  # 85-94%
    MEDIUM = "medium"  # 70-84%
    LOW = "low"  # 50-69%
    VERY_LOW = "very_low"  # <50%


class ResponseType(str, Enum):
    """Types of automated responses."""

    QUARANTINE = "quarantine"
    BLOCK_IP = "block_ip"
    ISOLATE_SYSTEM = "isolate_system"
    PATCH_VULNERABILITY = "patch_vulnerability"
    RESET_CREDENTIALS = "reset_credentials"
    ENABLE_MONITORING = "enable_monitoring"
    BACKUP_SYSTEM = "backup_system"
    ALERT_SECURITY_TEAM = "alert_security_team"
    SHUTDOWN_SERVICE = "shutdown_service"
    REDIRECT_TRAFFIC = "redirect_traffic"
    DEPLOY_HONEYPOT = "deploy_honeypot"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    VIRTUAL_PATCH = "virtual_patch"
    NETWORK_SEGMENTATION = "network_segmentation"
    LOG_RETENTION_INCREASE = "log_retention_increase"


class SystemCriticality(str, Enum):
    """Business criticality levels."""

    MISSION_CRITICAL = "mission_critical"
    BUSINESS_CRITICAL = "business_critical"
    IMPORTANT = "important"
    STANDARD = "standard"
    LOW_PRIORITY = "low_priority"


class ProtocolType(str, Enum):
    """Legacy protocol types."""

    HL7_MLLP = "hl7_mllp"
    ISO8583 = "iso8583"
    MODBUS = "modbus"
    TN3270E = "tn3270e"
    SWIFT = "swift"
    FIX = "fix"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


@dataclass
class SecurityEvent:
    """Core security event structure."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SecurityEventType = SecurityEventType.ANOMALOUS_BEHAVIOR
    threat_level: ThreatLevel = ThreatLevel.MEDIUM

    # Event details
    title: str = ""
    description: str = ""
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    affected_systems: List[str] = field(default_factory=list)
    affected_protocols: List[ProtocolType] = field(default_factory=list)

    # Technical details
    attack_vectors: List[str] = field(default_factory=list)
    indicators_of_compromise: List[str] = field(default_factory=list)
    network_artifacts: Dict[str, Any] = field(default_factory=dict)
    file_artifacts: Dict[str, Any] = field(default_factory=dict)

    # Context
    business_context: Dict[str, Any] = field(default_factory=dict)
    compliance_implications: List[str] = field(default_factory=list)
    potential_impact: Dict[str, Any] = field(default_factory=dict)

    # Detection metadata
    detection_source: str = "ai_engine"
    detection_method: str = "ml_model"
    confidence_score: float = 0.0
    false_positive_likelihood: float = 0.0

    # Timing
    event_timestamp: datetime = field(default_factory=datetime.now)
    detection_timestamp: datetime = field(default_factory=datetime.now)
    first_observed: Optional[datetime] = None
    last_observed: Optional[datetime] = None

    # Correlation
    related_events: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    parent_incident: Optional[str] = None

    # Raw data
    raw_data: Optional[bytes] = None
    log_entries: List[str] = field(default_factory=list)
    packet_data: Optional[bytes] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LegacySystem:
    """Legacy system information for security context."""

    system_id: str
    system_name: str
    system_type: str
    protocol_types: List[ProtocolType] = field(default_factory=list)

    # Network information
    ip_addresses: List[str] = field(default_factory=list)
    network_segments: List[str] = field(default_factory=list)
    exposed_ports: List[int] = field(default_factory=list)

    # Business context
    criticality: SystemCriticality = SystemCriticality.STANDARD
    business_function: str = ""
    compliance_requirements: List[str] = field(default_factory=list)
    data_classification: str = "internal"

    # Technical characteristics
    operating_system: Optional[str] = None
    vendor: Optional[str] = None
    version: Optional[str] = None
    patch_level: Optional[str] = None
    last_patched: Optional[datetime] = None

    # Dependencies
    dependent_systems: List[str] = field(default_factory=list)
    dependency_systems: List[str] = field(default_factory=list)
    integration_points: List[Dict[str, Any]] = field(default_factory=list)

    # Security posture
    security_controls: List[str] = field(default_factory=list)
    known_vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    risk_score: float = 0.0

    # Operational constraints
    maintenance_windows: List[Dict[str, Any]] = field(default_factory=list)
    uptime_requirements: float = 99.0  # Percentage
    change_freeze_periods: List[Tuple[datetime, datetime]] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatAnalysis:
    """Comprehensive threat analysis results."""

    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = ""

    # Analysis results
    threat_classification: SecurityEventType = SecurityEventType.ANOMALOUS_BEHAVIOR
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_score: float = 0.0

    # Threat intelligence
    threat_actor: Optional[str] = None
    attack_campaign: Optional[str] = None
    ttps: List[str] = field(default_factory=list)  # Tactics, Techniques, Procedures
    mitre_attack_techniques: List[str] = field(default_factory=list)

    # Impact assessment
    potential_damage: Dict[str, Any] = field(default_factory=dict)
    affected_assets: List[str] = field(default_factory=list)
    business_impact_score: float = 0.0
    financial_impact_estimate: Optional[float] = None

    # Technical analysis
    attack_timeline: List[Dict[str, Any]] = field(default_factory=list)
    root_cause_analysis: Optional[str] = None
    attack_methodology: List[str] = field(default_factory=list)

    # Risk factors
    exploitability_score: float = 0.0
    prevalence_score: float = 0.0
    detectability_score: float = 0.0

    # Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    short_term_actions: List[str] = field(default_factory=list)
    long_term_actions: List[str] = field(default_factory=list)

    # Analysis metadata
    analyzer_version: str = "1.0.0"
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0
    data_sources: List[str] = field(default_factory=list)

    # Expert insights (from LLM)
    expert_analysis: Optional[str] = None
    similar_incidents: List[str] = field(default_factory=list)
    contextual_factors: List[str] = field(default_factory=list)


@dataclass
class ResponseAction:
    """Individual response action definition."""

    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ResponseType = ResponseType.ALERT_SECURITY_TEAM

    # Action details
    title: str = ""
    description: str = ""
    priority: int = 1  # 1 = highest priority

    # Execution parameters
    target_systems: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300

    # Prerequisites
    required_permissions: List[str] = field(default_factory=list)
    prerequisite_actions: List[str] = field(default_factory=list)
    safety_checks: List[str] = field(default_factory=list)

    # Risk assessment
    risk_level: ThreatLevel = ThreatLevel.LOW
    potential_side_effects: List[str] = field(default_factory=list)
    rollback_procedure: Optional[str] = None

    # Business impact
    business_impact: Dict[str, Any] = field(default_factory=dict)
    estimated_downtime: Optional[int] = None  # seconds
    user_impact: Optional[str] = None

    # Execution tracking
    status: str = "pending"  # pending, executing, completed, failed, skipped
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # Approval workflow
    requires_approval: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutomatedResponse:
    """Complete automated response plan."""

    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = ""
    analysis_id: str = ""

    # Response strategy
    response_strategy: str = "contain_and_investigate"
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_score: float = 0.0

    # Actions
    actions: List[ResponseAction] = field(default_factory=list)
    parallel_actions: List[List[str]] = field(
        default_factory=list
    )  # action_ids that can run in parallel
    sequential_actions: List[str] = field(
        default_factory=list
    )  # action_ids that must run sequentially

    # Timing constraints
    execution_window: Optional[Tuple[datetime, datetime]] = None
    max_execution_time: int = 3600  # seconds
    escalation_timeout: int = 1800  # seconds

    # Approval and authorization
    auto_execute: bool = False
    requires_human_approval: bool = True
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    # Risk management
    overall_risk_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    safety_constraints: List[str] = field(default_factory=list)

    # Business impact
    estimated_business_impact: Dict[str, Any] = field(default_factory=dict)
    affected_business_processes: List[str] = field(default_factory=list)
    compliance_considerations: List[str] = field(default_factory=list)

    # Execution tracking
    status: str = (
        "pending"  # pending, approved, executing, completed, failed, cancelled
    )
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    success_rate: float = 0.0

    # Results
    execution_results: List[Dict[str, Any]] = field(default_factory=list)
    effectiveness_score: Optional[float] = None
    lessons_learned: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "zero_touch_engine"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuarantineResult:
    """Result of quarantine operation."""

    quarantine_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    system_id: str = ""

    # Quarantine details
    quarantine_type: str = "network_isolation"
    status: str = "pending"  # pending, active, released, failed
    severity_level: ThreatLevel = ThreatLevel.MEDIUM

    # Implementation details
    isolation_method: str = "vlan_isolation"
    affected_interfaces: List[str] = field(default_factory=list)
    blocked_protocols: List[ProtocolType] = field(default_factory=list)
    allowed_exceptions: List[str] = field(default_factory=list)

    # Safety measures
    dependency_checks: List[Dict[str, Any]] = field(default_factory=list)
    rollback_plan: Optional[str] = None
    monitoring_enabled: bool = True

    # Business considerations
    business_impact_assessment: Dict[str, Any] = field(default_factory=dict)
    stakeholder_notifications: List[str] = field(default_factory=list)

    # Timing
    quarantine_start: Optional[datetime] = None
    quarantine_end: Optional[datetime] = None
    planned_duration: Optional[int] = None  # seconds
    auto_release: bool = False

    # Results
    success: bool = False
    error_message: Optional[str] = None
    side_effects: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityContext:
    """Security context for decision making."""

    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Environmental context
    current_threat_level: ThreatLevel = ThreatLevel.MEDIUM
    active_incidents: List[str] = field(default_factory=list)
    recent_attacks: List[Dict[str, Any]] = field(default_factory=list)

    # System state
    system_health: Dict[str, Any] = field(default_factory=dict)
    network_topology: Dict[str, Any] = field(default_factory=dict)
    critical_systems: List[str] = field(default_factory=list)

    # Business context
    business_hours: bool = True
    maintenance_windows: List[Dict[str, Any]] = field(default_factory=list)
    compliance_mode: str = "standard"  # standard, audit, lockdown

    # Threat intelligence
    current_campaigns: List[str] = field(default_factory=list)
    threat_indicators: List[str] = field(default_factory=list)
    vulnerability_status: Dict[str, Any] = field(default_factory=dict)

    # Operational constraints
    staffing_level: str = "normal"  # minimal, normal, enhanced
    budget_constraints: Dict[str, Any] = field(default_factory=dict)
    risk_tolerance: str = "moderate"  # low, moderate, high

    # Historical data
    similar_incident_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    false_positive_rates: Dict[str, float] = field(default_factory=dict)
    response_effectiveness: Dict[str, float] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatIntelligence:
    """Threat intelligence information."""

    intelligence_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Source information
    source: str = ""
    source_reliability: str = "medium"  # low, medium, high, very_high
    classification: str = "internal"  # public, internal, confidential, secret

    # Threat data
    threat_type: SecurityEventType = SecurityEventType.ANOMALOUS_BEHAVIOR
    threat_actors: List[str] = field(default_factory=list)
    campaigns: List[str] = field(default_factory=list)

    # Technical indicators
    iocs: List[Dict[str, Any]] = field(default_factory=list)  # Indicators of Compromise
    ttps: List[str] = field(default_factory=list)
    attack_patterns: List[str] = field(default_factory=list)

    # Contextual information
    target_sectors: List[str] = field(default_factory=list)
    target_regions: List[str] = field(default_factory=list)
    active_period: Optional[Tuple[datetime, datetime]] = None

    # Assessment
    severity: ThreatLevel = ThreatLevel.MEDIUM
    confidence: float = 0.0
    relevance_score: float = 0.0

    # Temporal data
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityMetrics:
    """Security metrics and KPIs."""

    metrics_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Detection metrics
    total_events: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0

    # Response metrics
    mean_response_time: float = 0.0  # seconds
    automated_responses: int = 0
    human_escalations: int = 0
    successful_containments: int = 0

    # Business impact
    prevented_incidents: int = 0
    estimated_savings: float = 0.0
    downtime_prevented: float = 0.0  # hours

    # System performance
    processing_latency: float = 0.0  # milliseconds
    throughput: float = 0.0  # events per second
    accuracy: float = 0.0

    # Operational metrics
    analyst_workload_reduction: float = 0.0  # percentage
    time_to_detection: float = 0.0  # seconds
    time_to_response: float = 0.0  # seconds

    # Metadata
    period_start: datetime = field(default_factory=datetime.now)
    period_end: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Exception classes for security operations
class SecurityException(CronosAIException):
    """Base security orchestrator exception."""

    pass


class ThreatAnalysisException(SecurityException):
    """Exception during threat analysis."""

    pass


class ResponseExecutionException(SecurityException):
    """Exception during response execution."""

    pass


class QuarantineException(SecurityException):
    """Exception during quarantine operations."""

    pass


class SecurityConfigurationException(SecurityException):
    """Exception in security configuration."""

    pass


# Validation functions
def validate_ip_address(ip_str: str) -> bool:
    """Validate IP address format."""
    try:
        ipaddress.ip_address(ip_str)
        return True
    except ValueError:
        return False


def validate_confidence_score(score: float) -> bool:
    """Validate confidence score is between 0.0 and 1.0."""
    return 0.0 <= score <= 1.0


def calculate_risk_score(
    threat_level: ThreatLevel, system_criticality: SystemCriticality, confidence: float
) -> float:
    """Calculate overall risk score."""
    threat_weights = {
        ThreatLevel.CRITICAL: 1.0,
        ThreatLevel.HIGH: 0.8,
        ThreatLevel.MEDIUM: 0.6,
        ThreatLevel.LOW: 0.4,
        ThreatLevel.INFO: 0.2,
    }

    criticality_weights = {
        SystemCriticality.MISSION_CRITICAL: 1.0,
        SystemCriticality.BUSINESS_CRITICAL: 0.8,
        SystemCriticality.IMPORTANT: 0.6,
        SystemCriticality.STANDARD: 0.4,
        SystemCriticality.LOW_PRIORITY: 0.2,
    }

    base_risk = threat_weights.get(threat_level, 0.5) * criticality_weights.get(
        system_criticality, 0.5
    )
    confidence_adjusted_risk = base_risk * confidence

    return min(1.0, confidence_adjusted_risk)
