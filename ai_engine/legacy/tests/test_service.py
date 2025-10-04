"""
Tests for Legacy System Whisperer Service

Comprehensive test suite for the main service integration layer.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

from ...core.config import Config
from ...core.exceptions import CronosAIException
from ...llm.unified_llm_service import UnifiedLLMService
from ...monitoring.metrics import AIEngineMetrics

from ..service import LegacySystemWhispererService, ServiceHealthStatus
from ..models import (
    LegacySystemContext,
    SystemType,
    Criticality,
    SystemMetrics,
    TimeSeriesData,
    PredictionHorizon,
)
from ..config import LegacySystemWhispererConfig
from ..logging import LegacySystemLogger
from ..exceptions import LegacySystemWhispererException


class TestLegacySystemWhispererService:
    """Test suite for LegacySystemWhispererService."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        config.legacy_system_whisperer = LegacySystemWhispererConfig()
        return config

    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        service = Mock(spec=UnifiedLLMService)
        service.process_request = AsyncMock()
        return service

    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics service."""
        return Mock(spec=AIEngineMetrics)

    @pytest.fixture
    def service(self, config):
        """Create service instance for testing."""
        return LegacySystemWhispererService(config)

    @pytest.fixture
    async def initialized_service(self, service, mock_llm_service, mock_metrics):
        """Create and initialize service for testing."""
        await service.initialize(mock_llm_service, mock_metrics)
        return service

    @pytest.fixture
    def sample_system_context(self):
        """Create sample system context for testing."""
        return LegacySystemContext(
            system_id="test_system_001",
            system_name="Test Legacy System",
            system_type=SystemType.MAINFRAME,
            version="z/OS 2.5",
            location="datacenter_a",
            criticality=Criticality.HIGH,
            compliance_requirements=["sox", "hipaa"],
            technical_contacts=["admin@test.com"],
            business_contacts=["business@test.com"],
        )

    @pytest.fixture
    def sample_system_metrics(self):
        """Create sample system metrics for testing."""
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_utilization=65.5,
            memory_utilization=78.2,
            disk_utilization=45.8,
            network_utilization=23.4,
            response_time_ms=150.0,
            error_rate=0.02,
            transaction_rate=1250.0,
            availability=0.9995,
        )

    @pytest.mark.unit
    def test_service_creation(self, config):
        """Test service can be created successfully."""
        service = LegacySystemWhispererService(config)

        assert service.config == config
        assert not service.is_initialized
        assert service.start_time is None
        assert len(service.registered_systems) == 0
        assert service.service_metrics["requests_processed"] == 0

    @pytest.mark.unit
    async def test_service_initialization(
        self, service, mock_llm_service, mock_metrics
    ):
        """Test service initialization."""
        await service.initialize(mock_llm_service, mock_metrics)

        assert service.is_initialized
        assert service.llm_service == mock_llm_service
        assert service.metrics == mock_metrics
        assert service.start_time is not None
        assert service.enhanced_detector is not None
        assert service.knowledge_capture is not None
        assert service.failure_predictor is not None
        assert service.performance_monitor is not None
        assert service.maintenance_scheduler is not None
        assert service.recommendation_engine is not None
        assert service.impact_assessor is not None
        assert service.action_planner is not None

    @pytest.mark.unit
    async def test_double_initialization_warning(
        self, service, mock_llm_service, mock_metrics, caplog
    ):
        """Test that double initialization shows warning."""
        await service.initialize(mock_llm_service, mock_metrics)
        await service.initialize(mock_llm_service, mock_metrics)

        assert "Service already initialized" in caplog.text

    @pytest.mark.unit
    async def test_register_legacy_system_success(
        self, initialized_service, sample_system_context
    ):
        """Test successful system registration."""
        result = await initialized_service.register_legacy_system(
            sample_system_context, enable_monitoring=True
        )

        assert result["system_id"] == sample_system_context.system_id
        assert result["system_name"] == sample_system_context.system_name
        assert result["system_type"] == sample_system_context.system_type.value
        assert result["monitoring_enabled"] is True
        assert "registration_time" in result
        assert "capabilities_enabled" in result

        # Check system was registered
        assert sample_system_context.system_id in initialized_service.registered_systems
        assert (
            initialized_service.active_monitoring[sample_system_context.system_id]
            is True
        )

    @pytest.mark.unit
    async def test_register_system_invalid_context(self, initialized_service):
        """Test registration with invalid system context."""
        invalid_context = LegacySystemContext(
            system_id="",  # Invalid empty system_id
            system_name="",  # Invalid empty name
            system_type=SystemType.MAINFRAME,
        )

        with pytest.raises(CronosAIException, match="Invalid system context"):
            await initialized_service.register_legacy_system(invalid_context)

    @pytest.mark.unit
    async def test_analyze_system_health_unregistered(
        self, initialized_service, sample_system_metrics
    ):
        """Test health analysis for unregistered system."""
        with pytest.raises(CronosAIException, match="System .* not registered"):
            await initialized_service.analyze_system_health(
                "unregistered_system", sample_system_metrics
            )

    @pytest.mark.unit
    async def test_analyze_system_health_success(
        self, initialized_service, sample_system_context, sample_system_metrics
    ):
        """Test successful system health analysis."""
        # Register system first
        await initialized_service.register_legacy_system(sample_system_context)

        # Mock the enhanced detector
        mock_prediction = Mock()
        mock_prediction.failure_probability = 0.15
        mock_prediction.confidence = 0.85
        initialized_service.enhanced_detector.predict_system_failure = AsyncMock(
            return_value=mock_prediction
        )

        # Mock performance monitor
        mock_performance = {"performance_score": 78.5, "status": "good"}
        initialized_service.performance_monitor.analyze_performance = AsyncMock(
            return_value=mock_performance
        )

        result = await initialized_service.analyze_system_health(
            sample_system_context.system_id, sample_system_metrics
        )

        assert result["system_id"] == sample_system_context.system_id
        assert "analysis_timestamp" in result
        assert "overall_health_score" in result
        assert "analysis_results" in result
        assert "processing_time_seconds" in result
        assert result["overall_health_score"] > 0

    @pytest.mark.unit
    async def test_capture_expert_knowledge_success(self, initialized_service):
        """Test successful expert knowledge capture."""
        # Mock knowledge capture
        mock_session_id = "session_123"
        mock_capture_result = {"status": "success", "items_captured": 3}
        mock_session_result = {"total_knowledge": 15, "quality_score": 0.92}

        initialized_service.knowledge_capture.start_expert_session = AsyncMock(
            return_value=mock_session_id
        )
        initialized_service.knowledge_capture.capture_expert_input = AsyncMock(
            return_value=mock_capture_result
        )
        initialized_service.knowledge_capture.finalize_session = AsyncMock(
            return_value=mock_session_result
        )

        result = await initialized_service.capture_expert_knowledge(
            expert_id="expert_001",
            session_type="maintenance_knowledge",
            expert_input="System requires maintenance every 6 months",
            system_id="test_system_001",
        )

        assert result["status"] == "success"
        assert result["session_id"] == mock_session_id
        assert result["capture_result"] == mock_capture_result
        assert result["session_result"] == mock_session_result

        # Check metrics updated
        assert initialized_service.service_metrics["knowledge_items_captured"] == 1

    @pytest.mark.unit
    async def test_schedule_maintenance_success(self, initialized_service):
        """Test successful maintenance scheduling."""
        from ..models import MaintenanceRecommendation

        # Mock maintenance requests
        maintenance_requests = [
            MaintenanceRecommendation(
                system_id="test_system_001",
                maintenance_type="preventive",
                priority="medium",
                estimated_duration=timedelta(hours=4),
                description="Routine system maintenance",
            )
        ]

        # Mock scheduler
        mock_schedule_result = {
            "optimized_schedule": [
                {
                    "maintenance_id": "maint_001",
                    "scheduled_time": datetime.now() + timedelta(days=7),
                    "estimated_duration": timedelta(hours=4),
                }
            ],
            "optimization_score": 0.87,
            "conflicts_resolved": 2,
        }

        initialized_service.maintenance_scheduler.optimize_maintenance_schedule = (
            AsyncMock(return_value=mock_schedule_result)
        )
        initialized_service.maintenance_scheduler.add_scheduled_maintenance = Mock()

        result = await initialized_service.schedule_maintenance(maintenance_requests)

        assert result == mock_schedule_result
        assert initialized_service.service_metrics["maintenance_scheduled"] == 1

    @pytest.mark.unit
    async def test_create_decision_support_success(
        self, initialized_service, sample_system_context
    ):
        """Test successful decision support creation."""
        from ..decision_support import DecisionCategory

        # Register system first
        await initialized_service.register_legacy_system(sample_system_context)

        # Mock recommendation engine
        mock_recommendations = Mock()
        mock_recommendations.recommended_actions = [
            {"action": "upgrade_memory", "priority": "high"},
            {"action": "schedule_maintenance", "priority": "medium"},
        ]

        initialized_service.recommendation_engine.generate_recommendations = AsyncMock(
            return_value=mock_recommendations
        )

        # Mock impact assessor
        mock_impact = {"financial_impact": 50000, "risk_level": "medium"}
        initialized_service.impact_assessor.assess_business_impact = AsyncMock(
            return_value=mock_impact
        )

        result = await initialized_service.create_decision_support(
            decision_category=DecisionCategory.MAINTENANCE_PLANNING,
            system_id=sample_system_context.system_id,
            current_situation={"performance_degraded": True},
            objectives=["improve_performance", "reduce_risk"],
        )

        assert result["status"] == "success"
        assert "decision_recommendations" in result
        assert initialized_service.service_metrics["recommendations_created"] == 1

    @pytest.mark.unit
    def test_get_system_dashboard_unregistered(self, initialized_service):
        """Test dashboard for unregistered system raises exception."""
        with pytest.raises(CronosAIException, match="System .* not registered"):
            initialized_service.get_system_dashboard("unregistered_system")

    @pytest.mark.unit
    async def test_get_system_dashboard_success(
        self, initialized_service, sample_system_context
    ):
        """Test successful system dashboard retrieval."""
        # Register system first
        await initialized_service.register_legacy_system(sample_system_context)

        # Mock enhanced detector summary
        initialized_service.enhanced_detector.get_system_health_summary = Mock(
            return_value={
                "pattern_count": 15,
                "recent_predictions": 3,
                "health_score": 85.2,
            }
        )

        dashboard = initialized_service.get_system_dashboard(
            sample_system_context.system_id
        )

        assert dashboard["system_info"]["system_id"] == sample_system_context.system_id
        assert (
            dashboard["system_info"]["system_name"] == sample_system_context.system_name
        )
        assert (
            dashboard["system_info"]["system_type"]
            == sample_system_context.system_type.value
        )
        assert "current_status" in dashboard
        assert "health_score" in dashboard
        assert "pattern_count" in dashboard

    @pytest.mark.unit
    def test_get_service_status(self, initialized_service):
        """Test service status retrieval."""
        status = initialized_service.get_service_status()

        assert isinstance(status, ServiceHealthStatus)
        assert status.active_systems >= 0
        assert status.active_predictions >= 0
        assert status.last_health_check is not None

    @pytest.mark.unit
    def test_get_service_metrics(self, initialized_service):
        """Test service metrics retrieval."""
        metrics = initialized_service.get_service_metrics()

        assert "service_metrics" in metrics
        assert "system_registry" in metrics
        assert "component_status" in metrics
        assert "uptime" in metrics
        assert "last_updated" in metrics

        # Check service metrics structure
        service_metrics = metrics["service_metrics"]
        assert "requests_processed" in service_metrics
        assert "predictions_generated" in service_metrics
        assert "recommendations_created" in service_metrics

    @pytest.mark.unit
    async def test_shutdown_success(self, initialized_service, sample_system_context):
        """Test successful service shutdown."""
        # Register a system first
        await initialized_service.register_legacy_system(sample_system_context)
        assert (
            initialized_service.active_monitoring[sample_system_context.system_id]
            is True
        )

        # Shutdown service
        await initialized_service.shutdown()

        assert not initialized_service.is_initialized
        assert (
            initialized_service.active_monitoring[sample_system_context.system_id]
            is False
        )

    @pytest.mark.unit
    def test_calculate_overall_health_score_no_data(self, initialized_service):
        """Test health score calculation with no analysis data."""
        score = initialized_service._calculate_overall_health_score({})
        assert score == 50.0  # Neutral score

    @pytest.mark.unit
    def test_calculate_overall_health_score_with_data(self, initialized_service):
        """Test health score calculation with analysis data."""
        analysis_results = {
            "performance": {"performance_score": 80},
            "failure_prediction": {"failure_probability": 0.2},
            "failure_analysis": {"anomaly_analysis": {"anomaly_score": 0.1}},
        }

        score = initialized_service._calculate_overall_health_score(analysis_results)

        # Should be weighted average of scores
        # Performance: 80, Failure: 80 (1-0.2)*100, Anomaly: 90 (1-0.1)*100
        # Weighted: (80*0.4 + 80*0.4 + 90*0.2) = 82
        assert 80 <= score <= 85

    @pytest.mark.unit
    def test_update_service_metrics_success(self, initialized_service):
        """Test service metrics update."""
        initial_requests = initialized_service.service_metrics["requests_processed"]

        initialized_service._update_service_metrics("test_operation", 0.5, True)

        assert (
            initialized_service.service_metrics["requests_processed"]
            == initial_requests + 1
        )
        assert initialized_service.service_metrics["average_response_time"] == 0.5
        assert initialized_service.service_metrics["error_rate"] == 0.0

    @pytest.mark.unit
    def test_update_service_metrics_failure(self, initialized_service):
        """Test service metrics update on failure."""
        initial_requests = initialized_service.service_metrics["requests_processed"]

        initialized_service._update_service_metrics("test_operation", 1.0, False)

        assert (
            initialized_service.service_metrics["requests_processed"]
            == initial_requests + 1
        )
        assert (
            initialized_service.service_metrics["error_rate"] == 1.0
        )  # 100% error rate for single failure

    @pytest.mark.integration
    async def test_end_to_end_system_lifecycle(
        self, initialized_service, sample_system_context, sample_system_metrics
    ):
        """Test complete system lifecycle from registration to analysis."""
        # 1. Register system
        registration_result = await initialized_service.register_legacy_system(
            sample_system_context, enable_monitoring=True
        )
        assert (
            registration_result["status"] == "success"
            or "registration_time" in registration_result
        )

        # 2. Mock dependencies for health analysis
        mock_prediction = Mock()
        mock_prediction.failure_probability = 0.1
        mock_prediction.confidence = 0.9
        initialized_service.enhanced_detector.predict_system_failure = AsyncMock(
            return_value=mock_prediction
        )

        mock_performance = {"performance_score": 85.0, "status": "good"}
        initialized_service.performance_monitor.analyze_performance = AsyncMock(
            return_value=mock_performance
        )

        # 3. Analyze system health
        health_result = await initialized_service.analyze_system_health(
            sample_system_context.system_id, sample_system_metrics
        )
        assert health_result["system_id"] == sample_system_context.system_id
        assert health_result["overall_health_score"] > 0

        # 4. Get dashboard
        dashboard = initialized_service.get_system_dashboard(
            sample_system_context.system_id
        )
        assert dashboard["system_info"]["system_id"] == sample_system_context.system_id

        # 5. Check service metrics
        metrics = initialized_service.get_service_metrics()
        assert metrics["system_registry"]["registered_systems"] == 1
        assert metrics["system_registry"]["monitored_systems"] == 1

    @pytest.mark.performance
    async def test_concurrent_system_registration(self, initialized_service):
        """Test concurrent system registration performance."""
        systems = []
        for i in range(10):
            system = LegacySystemContext(
                system_id=f"concurrent_test_{i:03d}",
                system_name=f"Test System {i}",
                system_type=SystemType.MAINFRAME if i % 2 == 0 else SystemType.SCADA,
                criticality=Criticality.MEDIUM,
            )
            systems.append(system)

        # Register systems concurrently
        tasks = [
            initialized_service.register_legacy_system(system, enable_monitoring=False)
            for system in systems
        ]

        results = await asyncio.gather(*tasks)

        # All registrations should succeed
        assert len(results) == 10
        assert all("system_id" in result for result in results)
        assert len(initialized_service.registered_systems) == 10

    @pytest.mark.security
    async def test_system_isolation(self, initialized_service):
        """Test that systems are properly isolated."""
        # Create two systems
        system1 = LegacySystemContext(
            system_id="system_001",
            system_name="System 1",
            system_type=SystemType.MAINFRAME,
            criticality=Criticality.HIGH,
        )

        system2 = LegacySystemContext(
            system_id="system_002",
            system_name="System 2",
            system_type=SystemType.SCADA,
            criticality=Criticality.CRITICAL,
        )

        # Register both systems
        await initialized_service.register_legacy_system(system1)
        await initialized_service.register_legacy_system(system2)

        # Verify systems are isolated
        dashboard1 = initialized_service.get_system_dashboard("system_001")
        dashboard2 = initialized_service.get_system_dashboard("system_002")

        assert dashboard1["system_info"]["system_id"] == "system_001"
        assert dashboard2["system_info"]["system_id"] == "system_002"
        assert dashboard1["system_info"]["system_type"] == "mainframe"
        assert dashboard2["system_info"]["system_type"] == "scada"

    @pytest.mark.unit
    async def test_error_handling_initialization_failure(
        self, service, mock_llm_service
    ):
        """Test error handling during initialization failure."""
        # Mock component initialization failure
        with patch.object(
            service, "_initialize_components", side_effect=Exception("Init failed")
        ):
            with pytest.raises(
                CronosAIException, match="Service initialization failed"
            ):
                await service.initialize(mock_llm_service)

        assert not service.is_initialized


@pytest.mark.unit
class TestServiceHealthStatus:
    """Test ServiceHealthStatus data class."""

    def test_service_health_status_creation(self):
        """Test ServiceHealthStatus creation."""
        status = ServiceHealthStatus(
            overall_status="healthy",
            components_status={"component1": "healthy"},
            last_health_check=datetime.now(),
            active_systems=5,
            active_predictions=10,
            knowledge_base_size=100,
            performance_score=85.5,
        )

        assert status.overall_status == "healthy"
        assert status.components_status["component1"] == "healthy"
        assert status.active_systems == 5
        assert status.active_predictions == 10
        assert status.knowledge_base_size == 100
        assert status.performance_score == 85.5
        assert status.error_count == 0
        assert status.warnings == []


# Test fixtures and utilities
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def create_test_time_series_data(days: int = 30) -> list:
    """Create test time series data."""
    data = []
    base_time = datetime.now() - timedelta(days=days)

    for i in range(days * 24):  # Hourly data
        timestamp = base_time + timedelta(hours=i)
        data.append(
            TimeSeriesData(
                timestamp=timestamp,
                cpu_utilization=50.0 + (i % 20),
                memory_utilization=60.0 + (i % 15),
                response_time_ms=100.0 + (i % 50),
                error_rate=0.01 + (i % 10) * 0.001,
            )
        )

    return data


def assert_service_initialized(service: LegacySystemWhispererService):
    """Assert that service is properly initialized."""
    assert service.is_initialized
    assert service.llm_service is not None
    assert service.enhanced_detector is not None
    assert service.knowledge_capture is not None
    assert service.failure_predictor is not None
    assert service.performance_monitor is not None
    assert service.maintenance_scheduler is not None
    assert service.recommendation_engine is not None
    assert service.impact_assessor is not None
    assert service.action_planner is not None
