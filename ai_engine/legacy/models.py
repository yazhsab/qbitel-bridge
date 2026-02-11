"""
QBITEL Engine - Legacy System Whisperer Models

Data models and structures for legacy system management.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import numpy as np


class SystemType(str, Enum):
    """Types of legacy systems."""

    MAINFRAME = "mainframe"
    COBOL_SYSTEM = "cobol_system"
    INDUSTRIAL_SCADA = "industrial_scada"
    MEDICAL_DEVICE = "medical_device"
    FINANCIAL_LEGACY = "financial_legacy"
    MANUFACTURING_CONTROL = "manufacturing_control"
    TELECOM_SWITCH = "telecom_switch"
    EMBEDDED_SYSTEM = "embedded_system"
    DATABASE_LEGACY = "database_legacy"
    UNKNOWN = "unknown"


class FailureType(str, Enum):
    """Types of potential system failures."""

    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_CORRUPTION = "software_corruption"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_LEAK = "memory_leak"
    DISK_FAILURE = "disk_failure"
    NETWORK_TIMEOUT = "network_timeout"
    CONFIGURATION_DRIFT = "configuration_drift"
    SECURITY_BREACH = "security_breach"
    DATA_CORRUPTION = "data_corruption"
    CAPACITY_OVERRUN = "capacity_overrun"


class SeverityLevel(str, Enum):
    """Severity levels for issues and recommendations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MaintenanceType(str, Enum):
    """Types of maintenance activities."""

    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"
    ROUTINE = "routine"


@dataclass
class LegacySystemContext:
    """Context information about a legacy system."""

    system_id: str
    system_name: str
    system_type: SystemType
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    version: Optional[str] = None
    installation_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    location: Optional[str] = None
    criticality: SeverityLevel = SeverityLevel.MEDIUM

    # System specifications
    cpu_cores: Optional[int] = None
    memory_gb: Optional[float] = None
    disk_capacity_gb: Optional[float] = None
    network_interfaces: List[str] = field(default_factory=list)

    # Operational context
    operating_system: Optional[str] = None
    runtime_environment: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    connected_systems: List[str] = field(default_factory=list)

    # Business context
    business_function: Optional[str] = None
    users_count: Optional[int] = None
    transactions_per_day: Optional[int] = None
    uptime_requirement: float = 99.0  # Percentage

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemBehaviorPattern:
    """Represents a learned behavior pattern of a legacy system."""

    pattern_id: str
    system_id: str
    pattern_type: str  # "normal", "warning", "error", "maintenance"

    # Pattern characteristics
    description: str
    frequency: str  # "hourly", "daily", "weekly", "monthly", "irregular"
    duration_minutes: Optional[float] = None
    time_of_occurrence: Optional[str] = (
        None  # "morning", "evening", "night", "business_hours"
    )

    # Pattern metrics
    cpu_utilization_pattern: Optional[List[float]] = field(default_factory=list)
    memory_usage_pattern: Optional[List[float]] = field(default_factory=list)
    disk_io_pattern: Optional[List[float]] = field(default_factory=list)
    network_traffic_pattern: Optional[List[float]] = field(default_factory=list)

    # Pattern indicators
    leading_indicators: List[str] = field(default_factory=list)
    concurrent_patterns: List[str] = field(default_factory=list)
    following_patterns: List[str] = field(default_factory=list)

    # Statistical properties
    confidence_score: float = 0.0
    occurrence_count: int = 0
    last_seen: Optional[datetime] = None
    first_seen: Optional[datetime] = None

    # Expert knowledge
    expert_notes: Optional[str] = None
    business_impact: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SystemFailurePrediction:
    """Prediction of potential system failure."""

    prediction_id: str
    system_id: str
    failure_type: FailureType
    severity: SeverityLevel

    # Prediction details
    probability: float  # 0.0 to 1.0
    time_to_failure_hours: Optional[float] = None
    confidence_interval: Optional[tuple] = None  # (lower, upper) bounds

    # Contributing factors
    primary_indicators: List[str] = field(default_factory=list)
    secondary_indicators: List[str] = field(default_factory=list)
    historical_precedents: List[str] = field(default_factory=list)

    # Analysis results
    anomaly_score: float = 0.0
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    pattern_deviations: List[str] = field(default_factory=list)

    # Expert insights
    expert_analysis: Optional[str] = None
    tribal_knowledge_references: List[str] = field(default_factory=list)

    # Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    monitoring_recommendations: List[str] = field(default_factory=list)
    preventive_measures: List[str] = field(default_factory=list)

    # Business impact
    estimated_downtime_hours: Optional[float] = None
    business_impact_score: float = 0.0
    affected_users: Optional[int] = None
    revenue_impact_estimate: Optional[float] = None

    # Metadata
    model_version: str = "1.0.0"
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    expiry_timestamp: Optional[datetime] = None
    created_by: str = "legacy_whisperer"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaintenanceRecommendation:
    """Maintenance recommendation for a legacy system."""

    recommendation_id: str
    system_id: str
    maintenance_type: MaintenanceType
    priority: SeverityLevel

    # Recommendation details
    title: str
    description: str
    rationale: str
    expected_duration_hours: float
    estimated_cost: Optional[float] = None

    # Scheduling
    recommended_start_time: Optional[datetime] = None
    recommended_completion_time: Optional[datetime] = None
    maintenance_window_hours: Optional[int] = None
    blackout_periods: List[tuple] = field(default_factory=list)  # [(start, end), ...]

    # Prerequisites
    required_resources: List[str] = field(default_factory=list)
    required_expertise: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Risk assessment
    risk_level: SeverityLevel = SeverityLevel.MEDIUM
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    rollback_plan: Optional[str] = None

    # Expected outcomes
    expected_benefits: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    kpi_improvements: Dict[str, float] = field(default_factory=dict)

    # Implementation details
    step_by_step_procedure: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    safety_precautions: List[str] = field(default_factory=list)

    # Tracking
    status: str = (
        "pending"  # pending, approved, scheduled, in_progress, completed, cancelled
    )
    approved_by: Optional[str] = None
    scheduled_by: Optional[str] = None
    executed_by: Optional[str] = None

    # Results (filled after execution)
    actual_duration_hours: Optional[float] = None
    actual_cost: Optional[float] = None
    success_score: Optional[float] = None
    lessons_learned: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "legacy_whisperer"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormalizedKnowledge:
    """Formalized tribal knowledge about legacy systems."""

    knowledge_id: str
    system_id: Optional[str] = None  # Can be system-specific or general
    knowledge_type: str = (
        "general"  # "behavior", "troubleshooting", "maintenance", "configuration"
    )

    # Knowledge content
    title: str = ""
    description: str = ""
    detailed_explanation: str = ""

    # Structured knowledge
    behaviors: Dict[str, Any] = field(default_factory=dict)
    failure_indicators: List[str] = field(default_factory=list)
    maintenance_procedures: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Conditions and triggers
    conditions: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)

    # Solutions and actions
    recommended_actions: List[str] = field(default_factory=list)
    troubleshooting_steps: List[str] = field(default_factory=list)
    workarounds: List[str] = field(default_factory=list)

    # Validation and reliability
    confidence_score: float = 0.0
    validation_count: int = 0
    success_rate: float = 0.0
    last_validated: Optional[datetime] = None

    # Source information
    source_expert: Optional[str] = None
    source_document: Optional[str] = None
    source_type: str = "interview"  # "interview", "document", "observation", "incident"

    # Usage tracking
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    effectiveness_ratings: List[float] = field(default_factory=list)

    # Relationships
    related_knowledge: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    supersedes: List[str] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    status: str = "active"  # "active", "deprecated", "under_review"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """Real-time metrics for a legacy system."""

    system_id: str
    timestamp: datetime

    # Performance metrics
    cpu_utilization: float  # 0.0 to 100.0
    memory_utilization: float  # 0.0 to 100.0
    disk_utilization: float  # 0.0 to 100.0
    network_throughput: float  # bytes per second

    # System health indicators
    response_time_ms: Optional[float] = None
    transaction_rate: Optional[float] = None
    error_rate: Optional[float] = None
    uptime_seconds: Optional[float] = None

    # Resource usage
    disk_io_rate: Optional[float] = None
    network_connections: Optional[int] = None
    active_processes: Optional[int] = None
    queue_depth: Optional[int] = None

    # Custom metrics (system-specific)
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    # Quality indicators
    data_quality_score: Optional[float] = None
    completeness_score: Optional[float] = None

    # Metadata
    collection_method: str = "agent"  # "agent", "snmp", "api", "log_parser"
    source: str = "system_monitor"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoricalPatternDatabase:
    """Container for historical patterns and their metadata."""

    patterns: Dict[str, SystemBehaviorPattern] = field(default_factory=dict)
    pattern_relationships: Dict[str, List[str]] = field(default_factory=dict)
    system_patterns: Dict[str, List[str]] = field(
        default_factory=dict
    )  # system_id -> pattern_ids

    def get_patterns(
        self, system_id: Optional[str] = None
    ) -> List[SystemBehaviorPattern]:
        """Get patterns for a specific system or all patterns."""
        if system_id:
            pattern_ids = self.system_patterns.get(system_id, [])
            return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
        return list(self.patterns.values())

    def add_pattern(self, pattern: SystemBehaviorPattern) -> None:
        """Add a new pattern to the database."""
        self.patterns[pattern.pattern_id] = pattern

        # Update system patterns mapping
        if pattern.system_id not in self.system_patterns:
            self.system_patterns[pattern.system_id] = []

        if pattern.pattern_id not in self.system_patterns[pattern.system_id]:
            self.system_patterns[pattern.system_id].append(pattern.pattern_id)

    def get_related_patterns(self, pattern_id: str) -> List[SystemBehaviorPattern]:
        """Get patterns related to a specific pattern."""
        related_ids = self.pattern_relationships.get(pattern_id, [])
        return [self.patterns[pid] for pid in related_ids if pid in self.patterns]
