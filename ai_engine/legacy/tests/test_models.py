"""
Tests for Legacy System Whisperer Models

Test suite for data models and structures.
"""

import pytest
from datetime import datetime, timedelta
from dataclasses import FrozenInstanceError
from typing import List

from ..models import (
    # Enums
    SystemType,
    Criticality,
    FailureType,
    SeverityLevel,
    MaintenanceType,
    PredictionHorizon,
    DecisionCategory,
    KnowledgeConfidenceLevel,
    # Data Models
    LegacySystemContext,
    SystemMetrics,
    TimeSeriesData,
    SystemFailurePrediction,
    MaintenanceRecommendation,
    SystemBehaviorPattern,
    FormalizedKnowledge,
    DecisionContext,
    ActionRecommendation,
    BusinessImpactAssessment,
)
from ..exceptions import ValidationException


class TestEnums:
    """Test enumeration classes."""

    @pytest.mark.unit
    def test_system_type_values(self):
        """Test SystemType enum values."""
        assert SystemType.MAINFRAME.value == "mainframe"
        assert SystemType.COBOL.value == "cobol"
        assert SystemType.SCADA.value == "scada"
        assert SystemType.MEDICAL_DEVICE.value == "medical_device"
        assert SystemType.PLC.value == "plc"
        assert SystemType.DCS.value == "dcs"

    @pytest.mark.unit
    def test_criticality_values(self):
        """Test Criticality enum values."""
        assert Criticality.LOW.value == "low"
        assert Criticality.MEDIUM.value == "medium"
        assert Criticality.HIGH.value == "high"
        assert Criticality.CRITICAL.value == "critical"

    @pytest.mark.unit
    def test_failure_type_values(self):
        """Test FailureType enum values."""
        assert FailureType.HARDWARE_FAILURE.value == "hardware_failure"
        assert FailureType.SOFTWARE_FAILURE.value == "software_failure"
        assert FailureType.PERFORMANCE_DEGRADATION.value == "performance_degradation"
        assert FailureType.CONNECTIVITY_ISSUE.value == "connectivity_issue"

    @pytest.mark.unit
    def test_prediction_horizon_values(self):
        """Test PredictionHorizon enum values."""
        assert PredictionHorizon.SHORT_TERM == 7  # days
        assert PredictionHorizon.MEDIUM_TERM == 30  # days
        assert PredictionHorizon.LONG_TERM == 90  # days
        assert PredictionHorizon.STRATEGIC == 180  # days


class TestLegacySystemContext:
    """Test LegacySystemContext data model."""

    @pytest.fixture
    def valid_context_data(self):
        """Create valid context data."""
        return {
            "system_id": "test_system_001",
            "system_name": "Test Legacy System",
            "system_type": SystemType.MAINFRAME,
            "version": "z/OS 2.5",
            "location": "datacenter_a",
            "criticality": Criticality.HIGH,
            "compliance_requirements": ["sox", "hipaa"],
            "technical_contacts": ["admin@test.com"],
            "business_contacts": ["business@test.com"],
        }

    @pytest.mark.unit
    def test_create_valid_context(self, valid_context_data):
        """Test creating valid system context."""
        context = LegacySystemContext(**valid_context_data)

        assert context.system_id == "test_system_001"
        assert context.system_name == "Test Legacy System"
        assert context.system_type == SystemType.MAINFRAME
        assert context.version == "z/OS 2.5"
        assert context.location == "datacenter_a"
        assert context.criticality == Criticality.HIGH
        assert "sox" in context.compliance_requirements
        assert "hipaa" in context.compliance_requirements
        assert "admin@test.com" in context.technical_contacts
        assert "business@test.com" in context.business_contacts

    @pytest.mark.unit
    def test_context_minimal_required_fields(self):
        """Test context with only required fields."""
        context = LegacySystemContext(
            system_id="minimal_system",
            system_name="Minimal System",
            system_type=SystemType.SCADA,
        )

        assert context.system_id == "minimal_system"
        assert context.system_name == "Minimal System"
        assert context.system_type == SystemType.SCADA
        assert context.version is None
        assert context.location is None
        assert context.criticality == Criticality.MEDIUM  # default value
        assert context.compliance_requirements == []
        assert context.technical_contacts == []
        assert context.business_contacts == []

    @pytest.mark.unit
    def test_context_validation_empty_system_id(self):
        """Test validation with empty system ID."""
        with pytest.raises(ValidationException, match="System ID cannot be empty"):
            LegacySystemContext.validate_system_context(
                {
                    "system_id": "",
                    "system_name": "Test System",
                    "system_type": SystemType.MAINFRAME,
                }
            )

    @pytest.mark.unit
    def test_context_validation_empty_system_name(self):
        """Test validation with empty system name."""
        with pytest.raises(ValidationException, match="System name cannot be empty"):
            LegacySystemContext.validate_system_context(
                {
                    "system_id": "test_001",
                    "system_name": "",
                    "system_type": SystemType.MAINFRAME,
                }
            )

    @pytest.mark.unit
    def test_context_validation_invalid_email(self):
        """Test validation with invalid email contact."""
        with pytest.raises(ValidationException, match="Invalid email format"):
            LegacySystemContext.validate_system_context(
                {
                    "system_id": "test_001",
                    "system_name": "Test System",
                    "system_type": SystemType.MAINFRAME,
                    "technical_contacts": ["invalid-email"],
                }
            )

    @pytest.mark.unit
    def test_context_to_dict(self, valid_context_data):
        """Test context serialization to dictionary."""
        context = LegacySystemContext(**valid_context_data)
        context_dict = context.to_dict()

        assert context_dict["system_id"] == "test_system_001"
        assert context_dict["system_type"] == "mainframe"
        assert context_dict["criticality"] == "high"
        assert isinstance(context_dict["compliance_requirements"], list)
        assert isinstance(context_dict["technical_contacts"], list)


class TestSystemMetrics:
    """Test SystemMetrics data model."""

    @pytest.fixture
    def valid_metrics_data(self):
        """Create valid metrics data."""
        return {
            "timestamp": datetime.now(),
            "cpu_utilization": 65.5,
            "memory_utilization": 78.2,
            "disk_utilization": 45.8,
            "network_utilization": 23.4,
            "response_time_ms": 150.0,
            "error_rate": 0.02,
            "transaction_rate": 1250.0,
            "availability": 0.9995,
        }

    @pytest.mark.unit
    def test_create_valid_metrics(self, valid_metrics_data):
        """Test creating valid system metrics."""
        metrics = SystemMetrics(**valid_metrics_data)

        assert metrics.cpu_utilization == 65.5
        assert metrics.memory_utilization == 78.2
        assert metrics.disk_utilization == 45.8
        assert metrics.network_utilization == 23.4
        assert metrics.response_time_ms == 150.0
        assert metrics.error_rate == 0.02
        assert metrics.transaction_rate == 1250.0
        assert metrics.availability == 0.9995
        assert isinstance(metrics.timestamp, datetime)

    @pytest.mark.unit
    def test_metrics_validation_negative_values(self):
        """Test metrics validation with negative values."""
        with pytest.raises(ValidationException, match="Utilization values must be non-negative"):
            SystemMetrics.validate_metrics(
                {
                    "cpu_utilization": -10.0,
                    "memory_utilization": 50.0,
                    "timestamp": datetime.now(),
                }
            )

    @pytest.mark.unit
    def test_metrics_validation_utilization_over_100(self):
        """Test metrics validation with utilization over 100%."""
        with pytest.raises(ValidationException, match="Utilization values cannot exceed 100%"):
            SystemMetrics.validate_metrics(
                {
                    "cpu_utilization": 150.0,  # Over 100%
                    "memory_utilization": 50.0,
                    "timestamp": datetime.now(),
                }
            )

    @pytest.mark.unit
    def test_metrics_validation_negative_response_time(self):
        """Test metrics validation with negative response time."""
        with pytest.raises(ValidationException, match="Response time must be non-negative"):
            SystemMetrics.validate_metrics(
                {
                    "cpu_utilization": 50.0,
                    "response_time_ms": -100.0,
                    "timestamp": datetime.now(),
                }
            )

    @pytest.mark.unit
    def test_metrics_validation_invalid_error_rate(self):
        """Test metrics validation with invalid error rate."""
        with pytest.raises(ValidationException, match="Error rate must be between 0 and 1"):
            SystemMetrics.validate_metrics(
                {
                    "cpu_utilization": 50.0,
                    "error_rate": 1.5,  # Over 1.0
                    "timestamp": datetime.now(),
                }
            )

    @pytest.mark.unit
    def test_metrics_validation_invalid_availability(self):
        """Test metrics validation with invalid availability."""
        with pytest.raises(ValidationException, match="Availability must be between 0 and 1"):
            SystemMetrics.validate_metrics(
                {
                    "cpu_utilization": 50.0,
                    "availability": 1.1,  # Over 1.0
                    "timestamp": datetime.now(),
                }
            )

    @pytest.mark.unit
    def test_metrics_to_feature_vector(self, valid_metrics_data):
        """Test metrics conversion to feature vector."""
        metrics = SystemMetrics(**valid_metrics_data)
        feature_vector = metrics.to_feature_vector()

        expected_features = [
            "cpu_utilization",
            "memory_utilization",
            "disk_utilization",
            "network_utilization",
            "response_time_ms",
            "error_rate",
            "transaction_rate",
            "availability",
        ]

        for feature in expected_features:
            assert feature in feature_vector
            assert isinstance(feature_vector[feature], (int, float))


class TestTimeSeriesData:
    """Test TimeSeriesData model."""

    @pytest.mark.unit
    def test_create_time_series_data(self):
        """Test creating time series data."""
        timestamp = datetime.now()
        data = TimeSeriesData(
            timestamp=timestamp,
            cpu_utilization=75.0,
            memory_utilization=80.0,
            response_time_ms=200.0,
            error_rate=0.01,
        )

        assert data.timestamp == timestamp
        assert data.cpu_utilization == 75.0
        assert data.memory_utilization == 80.0
        assert data.response_time_ms == 200.0
        assert data.error_rate == 0.01

    @pytest.mark.unit
    def test_time_series_immutable(self):
        """Test that TimeSeriesData is immutable."""
        data = TimeSeriesData(timestamp=datetime.now(), cpu_utilization=75.0)

        with pytest.raises(FrozenInstanceError):
            data.cpu_utilization = 80.0


class TestSystemFailurePrediction:
    """Test SystemFailurePrediction model."""

    @pytest.fixture
    def sample_prediction_data(self):
        """Create sample prediction data."""
        return {
            "system_id": "test_system_001",
            "prediction_timestamp": datetime.now(),
            "failure_probability": 0.15,
            "confidence": 0.85,
            "predicted_failure_types": [FailureType.HARDWARE_FAILURE],
            "time_to_failure_days": 30,
            "severity": SeverityLevel.MEDIUM,
            "contributing_factors": ["high_temperature", "increased_error_rate"],
            "recommended_actions": ["schedule_maintenance", "monitor_closely"],
        }

    @pytest.mark.unit
    def test_create_failure_prediction(self, sample_prediction_data):
        """Test creating failure prediction."""
        prediction = SystemFailurePrediction(**sample_prediction_data)

        assert prediction.system_id == "test_system_001"
        assert prediction.failure_probability == 0.15
        assert prediction.confidence == 0.85
        assert FailureType.HARDWARE_FAILURE in prediction.predicted_failure_types
        assert prediction.time_to_failure_days == 30
        assert prediction.severity == SeverityLevel.MEDIUM
        assert "high_temperature" in prediction.contributing_factors
        assert "schedule_maintenance" in prediction.recommended_actions

    @pytest.mark.unit
    def test_prediction_validation_invalid_probability(self):
        """Test prediction validation with invalid probability."""
        with pytest.raises(ValidationException, match="Failure probability must be between 0 and 1"):
            SystemFailurePrediction.validate_prediction(
                {
                    "system_id": "test_001",
                    "failure_probability": 1.5,  # Invalid
                    "confidence": 0.8,
                }
            )

    @pytest.mark.unit
    def test_prediction_validation_invalid_confidence(self):
        """Test prediction validation with invalid confidence."""
        with pytest.raises(ValidationException, match="Confidence must be between 0 and 1"):
            SystemFailurePrediction.validate_prediction(
                {
                    "system_id": "test_001",
                    "failure_probability": 0.2,
                    "confidence": -0.1,  # Invalid
                }
            )

    @pytest.mark.unit
    def test_prediction_get_risk_level(self, sample_prediction_data):
        """Test risk level calculation."""
        # High risk prediction
        high_risk_data = sample_prediction_data.copy()
        high_risk_data["failure_probability"] = 0.8
        high_risk_prediction = SystemFailurePrediction(**high_risk_data)
        assert high_risk_prediction.get_risk_level() == "high"

        # Medium risk prediction
        medium_risk_data = sample_prediction_data.copy()
        medium_risk_data["failure_probability"] = 0.4
        medium_risk_prediction = SystemFailurePrediction(**medium_risk_data)
        assert medium_risk_prediction.get_risk_level() == "medium"

        # Low risk prediction
        low_risk_data = sample_prediction_data.copy()
        low_risk_data["failure_probability"] = 0.1
        low_risk_prediction = SystemFailurePrediction(**low_risk_data)
        assert low_risk_prediction.get_risk_level() == "low"


class TestMaintenanceRecommendation:
    """Test MaintenanceRecommendation model."""

    @pytest.fixture
    def sample_maintenance_data(self):
        """Create sample maintenance data."""
        return {
            "system_id": "test_system_001",
            "maintenance_type": MaintenanceType.PREVENTIVE,
            "priority": "high",
            "estimated_duration": timedelta(hours=4),
            "description": "Replace aging hardware components",
            "required_resources": ["technician", "replacement_parts"],
            "business_impact": "minimal",
            "recommended_schedule": datetime.now() + timedelta(days=7),
            "dependencies": ["system_002"],
            "cost_estimate": 15000.0,
        }

    @pytest.mark.unit
    def test_create_maintenance_recommendation(self, sample_maintenance_data):
        """Test creating maintenance recommendation."""
        recommendation = MaintenanceRecommendation(**sample_maintenance_data)

        assert recommendation.system_id == "test_system_001"
        assert recommendation.maintenance_type == MaintenanceType.PREVENTIVE
        assert recommendation.priority == "high"
        assert recommendation.estimated_duration == timedelta(hours=4)
        assert "Replace aging hardware" in recommendation.description
        assert "technician" in recommendation.required_resources
        assert recommendation.business_impact == "minimal"
        assert recommendation.cost_estimate == 15000.0

    @pytest.mark.unit
    def test_maintenance_validation_negative_duration(self):
        """Test maintenance validation with negative duration."""
        with pytest.raises(ValidationException, match="Estimated duration must be positive"):
            MaintenanceRecommendation.validate_maintenance(
                {
                    "system_id": "test_001",
                    "maintenance_type": MaintenanceType.PREVENTIVE,
                    "estimated_duration": timedelta(hours=-2),  # Negative
                }
            )

    @pytest.mark.unit
    def test_maintenance_validation_negative_cost(self):
        """Test maintenance validation with negative cost."""
        with pytest.raises(ValidationException, match="Cost estimate must be non-negative"):
            MaintenanceRecommendation.validate_maintenance(
                {
                    "system_id": "test_001",
                    "maintenance_type": MaintenanceType.PREVENTIVE,
                    "estimated_duration": timedelta(hours=2),
                    "cost_estimate": -1000.0,  # Negative
                }
            )


class TestSystemBehaviorPattern:
    """Test SystemBehaviorPattern model."""

    @pytest.mark.unit
    def test_create_behavior_pattern(self):
        """Test creating behavior pattern."""
        pattern = SystemBehaviorPattern(
            pattern_id="pattern_001",
            system_id="test_system_001",
            pattern_type="performance_degradation",
            description="CPU utilization spikes during batch processing",
            frequency="daily",
            confidence=0.85,
            first_observed=datetime.now() - timedelta(days=30),
            last_observed=datetime.now(),
            occurrence_count=25,
            associated_metrics=["cpu_utilization", "response_time_ms"],
            triggers=["batch_job_start"],
            business_impact="medium",
        )

        assert pattern.pattern_id == "pattern_001"
        assert pattern.system_id == "test_system_001"
        assert pattern.pattern_type == "performance_degradation"
        assert pattern.confidence == 0.85
        assert pattern.occurrence_count == 25
        assert "cpu_utilization" in pattern.associated_metrics
        assert "batch_job_start" in pattern.triggers


class TestFormalizedKnowledge:
    """Test FormalizedKnowledge model."""

    @pytest.mark.unit
    def test_create_formalized_knowledge(self):
        """Test creating formalized knowledge."""
        knowledge = FormalizedKnowledge(
            knowledge_id="knowledge_001",
            title="System Maintenance Procedures",
            category="maintenance",
            content="Detailed maintenance procedures for legacy system",
            confidence_level=KnowledgeConfidenceLevel.HIGH,
            source="expert_interview",
            expert_id="expert_001",
            validation_status="validated",
            tags=["maintenance", "procedures", "legacy"],
            related_systems=["system_001", "system_002"],
            creation_timestamp=datetime.now(),
            last_updated=datetime.now(),
            version="1.0",
        )

        assert knowledge.knowledge_id == "knowledge_001"
        assert knowledge.title == "System Maintenance Procedures"
        assert knowledge.category == "maintenance"
        assert knowledge.confidence_level == KnowledgeConfidenceLevel.HIGH
        assert knowledge.source == "expert_interview"
        assert "maintenance" in knowledge.tags
        assert "system_001" in knowledge.related_systems


class TestDecisionContext:
    """Test DecisionContext model."""

    @pytest.mark.unit
    def test_create_decision_context(self):
        """Test creating decision context."""
        system_context = LegacySystemContext(
            system_id="test_system",
            system_name="Test System",
            system_type=SystemType.MAINFRAME,
        )

        decision_context = DecisionContext(
            decision_id="decision_001",
            decision_category=DecisionCategory.MAINTENANCE_PLANNING,
            system_context=system_context,
            current_situation={"performance_degraded": True},
            constraints={"budget": 50000, "maintenance_window": "weekend"},
            objectives=["improve_performance", "reduce_cost"],
            stakeholders=["operations", "management"],
            risk_tolerance="medium",
        )

        assert decision_context.decision_id == "decision_001"
        assert decision_context.decision_category == DecisionCategory.MAINTENANCE_PLANNING
        assert decision_context.system_context.system_id == "test_system"
        assert decision_context.current_situation["performance_degraded"] is True
        assert decision_context.constraints["budget"] == 50000
        assert "improve_performance" in decision_context.objectives
        assert "operations" in decision_context.stakeholders


class TestDataModelIntegration:
    """Test integration between different data models."""

    @pytest.mark.unit
    def test_prediction_to_maintenance_recommendation(self):
        """Test converting prediction to maintenance recommendation."""
        # Create a failure prediction
        prediction = SystemFailurePrediction(
            system_id="test_system_001",
            prediction_timestamp=datetime.now(),
            failure_probability=0.7,
            confidence=0.9,
            predicted_failure_types=[FailureType.HARDWARE_FAILURE],
            time_to_failure_days=15,
            severity=SeverityLevel.HIGH,
            recommended_actions=["replace_hardware", "schedule_maintenance"],
        )

        # Convert to maintenance recommendation
        maintenance_rec = MaintenanceRecommendation.from_failure_prediction(prediction)

        assert maintenance_rec.system_id == prediction.system_id
        assert maintenance_rec.maintenance_type == MaintenanceType.CORRECTIVE
        assert maintenance_rec.priority == "high"  # Based on high severity
        assert "hardware" in maintenance_rec.description.lower()

    @pytest.mark.unit
    def test_metrics_time_series_consistency(self):
        """Test consistency between SystemMetrics and TimeSeriesData."""
        timestamp = datetime.now()

        # Create system metrics
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_utilization=65.0,
            memory_utilization=78.0,
            response_time_ms=150.0,
            error_rate=0.02,
        )

        # Create equivalent time series data
        ts_data = TimeSeriesData(
            timestamp=timestamp,
            cpu_utilization=65.0,
            memory_utilization=78.0,
            response_time_ms=150.0,
            error_rate=0.02,
        )

        # Should have same key values
        assert metrics.timestamp == ts_data.timestamp
        assert metrics.cpu_utilization == ts_data.cpu_utilization
        assert metrics.memory_utilization == ts_data.memory_utilization
        assert metrics.response_time_ms == ts_data.response_time_ms
        assert metrics.error_rate == ts_data.error_rate

    @pytest.mark.unit
    def test_system_context_with_multiple_models(self):
        """Test system context used across multiple models."""
        # Create system context
        system_context = LegacySystemContext(
            system_id="integrated_test_system",
            system_name="Integrated Test System",
            system_type=SystemType.SCADA,
            criticality=Criticality.CRITICAL,
        )

        # Use in failure prediction
        prediction = SystemFailurePrediction(
            system_id=system_context.system_id,
            prediction_timestamp=datetime.now(),
            failure_probability=0.3,
            confidence=0.8,
        )

        # Use in maintenance recommendation
        maintenance = MaintenanceRecommendation(
            system_id=system_context.system_id,
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority="medium",
            estimated_duration=timedelta(hours=2),
        )

        # Use in behavior pattern
        pattern = SystemBehaviorPattern(
            pattern_id="pattern_001",
            system_id=system_context.system_id,
            pattern_type="normal_operation",
            confidence=0.9,
        )

        # All should reference the same system
        assert prediction.system_id == system_context.system_id
        assert maintenance.system_id == system_context.system_id
        assert pattern.system_id == system_context.system_id

        # Critical system should have higher maintenance priority
        if system_context.criticality == Criticality.CRITICAL:
            assert maintenance.priority in ["high", "critical", "medium"]


# Performance and stress tests for data models
class TestModelPerformance:
    """Test data model performance characteristics."""

    @pytest.mark.performance
    def test_large_time_series_creation(self):
        """Test creating large time series datasets."""
        import time

        start_time = time.time()

        # Create 1000 time series data points
        time_series = []
        base_time = datetime.now()

        for i in range(1000):
            ts_data = TimeSeriesData(
                timestamp=base_time + timedelta(hours=i),
                cpu_utilization=50.0 + (i % 40),
                memory_utilization=60.0 + (i % 30),
                response_time_ms=100.0 + (i % 100),
                error_rate=0.01 + (i % 10) * 0.001,
            )
            time_series.append(ts_data)

        creation_time = time.time() - start_time

        # Should create 1000 records in reasonable time (< 1 second)
        assert creation_time < 1.0
        assert len(time_series) == 1000

    @pytest.mark.performance
    def test_model_serialization_performance(self):
        """Test model serialization performance."""
        import time

        # Create complex system context
        context = LegacySystemContext(
            system_id="perf_test_system",
            system_name="Performance Test System",
            system_type=SystemType.MAINFRAME,
            criticality=Criticality.HIGH,
            compliance_requirements=["sox", "hipaa", "gdpr", "pci", "iso27001"],
            technical_contacts=[f"tech{i}@test.com" for i in range(10)],
            business_contacts=[f"biz{i}@test.com" for i in range(10)],
        )

        # Test serialization performance
        start_time = time.time()

        for _ in range(100):
            context_dict = context.to_dict()
            assert isinstance(context_dict, dict)

        serialization_time = time.time() - start_time

        # Should serialize 100 times in reasonable time (< 0.1 seconds)
        assert serialization_time < 0.1


# Utility functions for model testing
def create_test_system_context(system_id: str = None) -> LegacySystemContext:
    """Create a test system context."""
    system_id = system_id or f"test_system_{int(datetime.now().timestamp())}"

    return LegacySystemContext(
        system_id=system_id,
        system_name=f"Test System {system_id}",
        system_type=SystemType.MAINFRAME,
        version="test_version",
        location="test_location",
        criticality=Criticality.MEDIUM,
        compliance_requirements=["test_compliance"],
        technical_contacts=["tech@test.com"],
        business_contacts=["biz@test.com"],
    )


def create_test_system_metrics() -> SystemMetrics:
    """Create test system metrics."""
    return SystemMetrics(
        timestamp=datetime.now(),
        cpu_utilization=50.0,
        memory_utilization=60.0,
        disk_utilization=40.0,
        network_utilization=30.0,
        response_time_ms=100.0,
        error_rate=0.01,
        transaction_rate=1000.0,
        availability=0.999,
    )


def create_test_failure_prediction(
    system_id: str = "test_system",
) -> SystemFailurePrediction:
    """Create test failure prediction."""
    return SystemFailurePrediction(
        system_id=system_id,
        prediction_timestamp=datetime.now(),
        failure_probability=0.3,
        confidence=0.8,
        predicted_failure_types=[FailureType.HARDWARE_FAILURE],
        time_to_failure_days=30,
        severity=SeverityLevel.MEDIUM,
    )
