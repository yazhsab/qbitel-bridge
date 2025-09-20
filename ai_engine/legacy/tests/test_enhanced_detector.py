"""
Tests for Legacy System Enhanced Anomaly Detector

Test suite for enhanced anomaly detection with LLM integration.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

from ..enhanced_detector import (
    EnhancedLegacySystemDetector,
    LegacySystemLLMAnalyzer,
    HistoricalPatternDatabase,
    ExpertKnowledgeBase
)
from ..models import (
    LegacySystemContext, SystemType, Criticality,
    SystemFailurePrediction, SystemBehaviorPattern,
    PredictionHorizon, FailureType, SeverityLevel
)
from ...core.config import Config
from ...llm.unified_llm_service import UnifiedLLMService, LLMRequest
from ...monitoring.metrics import AIEngineMetrics
from ..exceptions import LegacySystemWhispererException, AnomalyDetectionException
from ..config import LegacySystemWhispererConfig


class TestEnhancedLegacySystemDetector:
    """Test suite for EnhancedLegacySystemDetector."""
    
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
    def detector(self, config):
        """Create detector instance for testing."""
        return EnhancedLegacySystemDetector(config)
    
    @pytest.fixture
    async def initialized_detector(self, detector, mock_llm_service, mock_metrics):
        """Create and initialize detector for testing."""
        await detector.initialize_enhanced(mock_llm_service, mock_metrics)
        return detector
    
    @pytest.fixture
    def sample_system_context(self):
        """Create sample system context."""
        return LegacySystemContext(
            system_id="test_mainframe_001",
            system_name="Test Mainframe System",
            system_type=SystemType.MAINFRAME,
            version="z/OS 2.5",
            location="datacenter_a",
            criticality=Criticality.HIGH,
            compliance_requirements=["sox", "hipaa"],
            technical_contacts=["admin@test.com"],
            business_contacts=["business@test.com"]
        )
    
    @pytest.fixture
    def sample_system_data(self):
        """Create sample system data for analysis."""
        return {
            "cpu_utilization": 75.5,
            "memory_utilization": 82.3,
            "disk_utilization": 45.8,
            "response_time_ms": 250.0,
            "error_rate": 0.025,
            "transaction_rate": 1150.0
        }
    
    @pytest.mark.unit
    def test_detector_creation(self, config):
        """Test detector can be created successfully."""
        detector = EnhancedLegacySystemDetector(config)
        
        assert detector.config == config
        assert not detector.is_enhanced_initialized
        assert detector.llm_service is None
        assert detector.metrics is None
        assert detector.llm_analyzer is None
        assert detector.historical_patterns is None
        assert detector.expert_knowledge_base is None
    
    @pytest.mark.unit
    async def test_detector_initialization(self, detector, mock_llm_service, mock_metrics):
        """Test detector enhanced initialization."""
        await detector.initialize_enhanced(mock_llm_service, mock_metrics)
        
        assert detector.is_enhanced_initialized
        assert detector.llm_service == mock_llm_service
        assert detector.metrics == mock_metrics
        assert detector.llm_analyzer is not None
        assert detector.historical_patterns is not None
        assert detector.expert_knowledge_base is not None
        assert detector.system_contexts == {}
        assert detector.analysis_history == {}
    
    @pytest.mark.unit
    async def test_double_initialization_warning(
        self, 
        detector, 
        mock_llm_service, 
        mock_metrics, 
        caplog
    ):
        """Test that double initialization shows warning."""
        await detector.initialize_enhanced(mock_llm_service, mock_metrics)
        await detector.initialize_enhanced(mock_llm_service, mock_metrics)
        
        assert "Enhanced detector already initialized" in caplog.text
    
    @pytest.mark.unit
    async def test_add_system_context(self, initialized_detector, sample_system_context):
        """Test adding system context."""
        initialized_detector.add_system_context(sample_system_context)
        
        assert sample_system_context.system_id in initialized_detector.system_contexts
        stored_context = initialized_detector.system_contexts[sample_system_context.system_id]
        assert stored_context.system_name == sample_system_context.system_name
        assert stored_context.system_type == sample_system_context.system_type
        assert stored_context.criticality == sample_system_context.criticality
    
    @pytest.mark.unit
    async def test_predict_system_failure_success(
        self,
        initialized_detector,
        sample_system_context,
        sample_system_data
    ):
        """Test successful system failure prediction."""
        # Add system context
        initialized_detector.add_system_context(sample_system_context)
        
        # Mock base VAE detector
        with patch.object(initialized_detector, 'detect_anomaly') as mock_detect:
            mock_detect.return_value = 0.85  # High anomaly score
            
            # Mock LLM analyzer
            mock_llm_analysis = {
                "failure_probability": 0.75,
                "confidence": 0.88,
                "predicted_failure_types": ["hardware_failure"],
                "contributing_factors": ["high_cpu_usage", "memory_pressure"],
                "time_to_failure_estimate": 15,
                "severity": "high",
                "recommended_actions": ["immediate_investigation", "schedule_maintenance"]
            }
            
            initialized_detector.llm_analyzer.analyze_patterns = AsyncMock(
                return_value=mock_llm_analysis
            )
            
            # Perform prediction
            prediction = await initialized_detector.predict_system_failure(
                sample_system_data,
                sample_system_context,
                prediction_horizon=30
            )
            
            # Verify prediction
            assert isinstance(prediction, SystemFailurePrediction)
            assert prediction.system_id == sample_system_context.system_id
            assert prediction.failure_probability > 0
            assert prediction.confidence > 0
            assert len(prediction.predicted_failure_types) > 0
            assert len(prediction.contributing_factors) > 0
            assert len(prediction.recommended_actions) > 0
    
    @pytest.mark.unit
    async def test_predict_system_failure_no_context(
        self,
        initialized_detector,
        sample_system_data
    ):
        """Test failure prediction without system context."""
        with pytest.raises(AnomalyDetectionException, match="System context not found"):
            await initialized_detector.predict_system_failure(
                sample_system_data,
                system_context=None,  # No context provided
                prediction_horizon=30
            )
    
    @pytest.mark.unit
    async def test_predict_system_failure_detector_error(
        self,
        initialized_detector,
        sample_system_context,
        sample_system_data
    ):
        """Test failure prediction with detector error."""
        # Add system context
        initialized_detector.add_system_context(sample_system_context)
        
        # Mock base detector failure
        with patch.object(initialized_detector, 'detect_anomaly') as mock_detect:
            mock_detect.side_effect = Exception("Detector error")
            
            with pytest.raises(AnomalyDetectionException, match="Enhanced anomaly detection failed"):
                await initialized_detector.predict_system_failure(
                    sample_system_data,
                    sample_system_context,
                    prediction_horizon=30
                )
    
    @pytest.mark.unit
    async def test_enhanced_analysis_integration(
        self,
        initialized_detector,
        sample_system_context,
        sample_system_data
    ):
        """Test integration between anomaly detection and LLM analysis."""
        # Add system context
        initialized_detector.add_system_context(sample_system_context)
        
        # Mock historical patterns
        mock_patterns = [
            SystemBehaviorPattern(
                pattern_id="pattern_001",
                system_id=sample_system_context.system_id,
                pattern_type="cpu_spike",
                description="CPU spikes during batch processing",
                confidence=0.9
            )
        ]
        initialized_detector.historical_patterns.get_patterns.return_value = mock_patterns
        
        # Mock expert knowledge
        mock_knowledge = {
            "maintenance_indicators": ["high_response_time", "increased_errors"],
            "failure_patterns": ["gradual_performance_degradation"],
            "recommended_thresholds": {"cpu_critical": 85.0, "memory_critical": 90.0}
        }
        initialized_detector.expert_knowledge_base = mock_knowledge
        
        # Mock anomaly detection
        with patch.object(initialized_detector, 'detect_anomaly') as mock_detect:
            mock_detect.return_value = 0.75
            
            # Mock LLM analysis with pattern and knowledge integration
            mock_llm_response = {
                "failure_probability": 0.65,
                "confidence": 0.82,
                "analysis": "System shows signs of gradual performance degradation",
                "pattern_matches": ["cpu_spike"],
                "expert_knowledge_applied": ["maintenance_indicators"],
                "severity": "medium"
            }
            
            initialized_detector.llm_analyzer.analyze_patterns = AsyncMock(
                return_value=mock_llm_response
            )
            
            # Perform enhanced analysis
            prediction = await initialized_detector.predict_system_failure(
                sample_system_data,
                sample_system_context,
                prediction_horizon=30
            )
            
            # Verify LLM analyzer was called with correct parameters
            initialized_detector.llm_analyzer.analyze_patterns.assert_called_once()
            call_args = initialized_detector.llm_analyzer.analyze_patterns.call_args
            
            assert call_args[1]["current_data"] == sample_system_data
            assert call_args[1]["historical_patterns"] == mock_patterns
            assert call_args[1]["expert_knowledge"] == mock_knowledge
            
            # Verify prediction incorporates LLM analysis
            assert prediction.failure_probability == 0.65
            assert prediction.confidence == 0.82
    
    @pytest.mark.unit
    def test_get_system_health_summary(self, initialized_detector, sample_system_context):
        """Test system health summary retrieval."""
        system_id = sample_system_context.system_id
        
        # Add system context
        initialized_detector.add_system_context(sample_system_context)
        
        # Add some mock analysis history
        initialized_detector.analysis_history[system_id] = [
            {"timestamp": datetime.now(), "anomaly_score": 0.3, "failure_probability": 0.2},
            {"timestamp": datetime.now(), "anomaly_score": 0.5, "failure_probability": 0.4},
            {"timestamp": datetime.now(), "anomaly_score": 0.8, "failure_probability": 0.7}
        ]
        
        summary = initialized_detector.get_system_health_summary(system_id)
        
        assert "pattern_count" in summary
        assert "recent_predictions" in summary
        assert "health_score" in summary
        assert "latest_analysis" in summary
        assert summary["recent_predictions"] == 3
        
        # Health score should be calculated based on recent analyses
        assert 0 <= summary["health_score"] <= 100
    
    @pytest.mark.unit
    def test_get_system_health_summary_no_system(self, initialized_detector):
        """Test health summary for non-existent system."""
        summary = initialized_detector.get_system_health_summary("nonexistent_system")
        
        # Should return basic summary with zeros
        assert summary["pattern_count"] == 0
        assert summary["recent_predictions"] == 0
        assert summary["health_score"] == 50.0  # Neutral score


class TestLegacySystemLLMAnalyzer:
    """Test suite for LegacySystemLLMAnalyzer."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        service = Mock(spec=UnifiedLLMService)
        service.process_request = AsyncMock()
        return service
    
    @pytest.fixture
    def analyzer(self, mock_llm_service):
        """Create LLM analyzer for testing."""
        return LegacySystemLLMAnalyzer(mock_llm_service)
    
    @pytest.fixture
    def sample_current_data(self):
        """Create sample current system data."""
        return {
            "cpu_utilization": 85.0,
            "memory_utilization": 78.0,
            "response_time_ms": 300.0,
            "error_rate": 0.05
        }
    
    @pytest.fixture
    def sample_historical_patterns(self):
        """Create sample historical patterns."""
        return [
            SystemBehaviorPattern(
                pattern_id="pattern_001",
                system_id="test_system",
                pattern_type="performance_degradation",
                description="Gradual performance decline over time",
                confidence=0.85
            ),
            SystemBehaviorPattern(
                pattern_id="pattern_002", 
                system_id="test_system",
                pattern_type="error_spike",
                description="Periodic error rate increases",
                confidence=0.75
            )
        ]
    
    @pytest.fixture
    def sample_expert_knowledge(self):
        """Create sample expert knowledge."""
        return {
            "critical_thresholds": {"cpu": 90.0, "memory": 85.0, "response_time": 500.0},
            "failure_indicators": ["sustained_high_cpu", "memory_leaks", "network_timeouts"],
            "maintenance_patterns": ["monthly_restart", "quarterly_upgrade"],
            "business_impact_factors": ["peak_hours", "batch_processing_time"]
        }
    
    @pytest.mark.unit
    async def test_analyze_patterns_success(
        self,
        analyzer,
        sample_current_data,
        sample_historical_patterns,
        sample_expert_knowledge
    ):
        """Test successful pattern analysis."""
        # Mock LLM response
        mock_llm_response = Mock()
        mock_llm_response.content = """{
            "failure_probability": 0.72,
            "confidence": 0.85,
            "predicted_failure_types": ["performance_degradation", "resource_exhaustion"],
            "contributing_factors": ["high_cpu_utilization", "increasing_response_time"],
            "time_to_failure_estimate": 10,
            "severity": "high",
            "recommended_actions": ["investigate_cpu_usage", "monitor_memory_trends"],
            "pattern_analysis": {
                "matched_patterns": ["performance_degradation"],
                "trend_direction": "declining",
                "anomaly_significance": "high"
            },
            "expert_knowledge_insights": {
                "threshold_violations": ["cpu_approaching_critical"],
                "recommended_interventions": ["immediate_investigation"]
            }
        }"""
        
        analyzer.llm_service.process_request.return_value = mock_llm_response
        
        # Analyze patterns
        result = await analyzer.analyze_patterns(
            current_data=sample_current_data,
            historical_patterns=sample_historical_patterns,
            expert_knowledge=sample_expert_knowledge
        )
        
        # Verify LLM service was called
        analyzer.llm_service.process_request.assert_called_once()
        call_args = analyzer.llm_service.process_request.call_args[0][0]
        assert isinstance(call_args, LLMRequest)
        assert "legacy system analysis" in call_args.prompt.lower()
        
        # Verify parsed response
        assert result["failure_probability"] == 0.72
        assert result["confidence"] == 0.85
        assert "performance_degradation" in result["predicted_failure_types"]
        assert "high_cpu_utilization" in result["contributing_factors"]
        assert result["time_to_failure_estimate"] == 10
        assert result["severity"] == "high"
        assert len(result["recommended_actions"]) == 2
    
    @pytest.mark.unit
    async def test_analyze_patterns_llm_failure(
        self,
        analyzer,
        sample_current_data,
        sample_historical_patterns,
        sample_expert_knowledge
    ):
        """Test pattern analysis with LLM service failure."""
        # Mock LLM service failure
        analyzer.llm_service.process_request.side_effect = Exception("LLM service error")
        
        with pytest.raises(LegacySystemWhispererException, match="LLM analysis failed"):
            await analyzer.analyze_patterns(
                current_data=sample_current_data,
                historical_patterns=sample_historical_patterns,
                expert_knowledge=sample_expert_knowledge
            )
    
    @pytest.mark.unit
    async def test_analyze_patterns_invalid_json_response(
        self,
        analyzer,
        sample_current_data,
        sample_historical_patterns,
        sample_expert_knowledge
    ):
        """Test pattern analysis with invalid JSON response."""
        # Mock invalid LLM response
        mock_llm_response = Mock()
        mock_llm_response.content = "Invalid JSON response"
        
        analyzer.llm_service.process_request.return_value = mock_llm_response
        
        with pytest.raises(LegacySystemWhispererException, match="Failed to parse LLM response"):
            await analyzer.analyze_patterns(
                current_data=sample_current_data,
                historical_patterns=sample_historical_patterns,
                expert_knowledge=sample_expert_knowledge
            )
    
    @pytest.mark.unit
    def test_build_analysis_prompt(
        self,
        analyzer,
        sample_current_data,
        sample_historical_patterns,
        sample_expert_knowledge
    ):
        """Test analysis prompt building."""
        prompt = analyzer._build_analysis_prompt(
            sample_current_data,
            sample_historical_patterns,
            sample_expert_knowledge
        )
        
        # Verify prompt contains key elements
        assert "legacy system analysis" in prompt.lower()
        assert "current system metrics" in prompt.lower()
        assert "historical patterns" in prompt.lower()
        assert "expert knowledge" in prompt.lower()
        assert "failure probability" in prompt.lower()
        assert str(sample_current_data["cpu_utilization"]) in prompt
        assert "performance_degradation" in prompt  # From historical patterns
        assert "critical_thresholds" in prompt  # From expert knowledge
    
    @pytest.mark.unit
    def test_parse_llm_response_success(self, analyzer):
        """Test successful LLM response parsing."""
        response_content = """{
            "failure_probability": 0.65,
            "confidence": 0.80,
            "predicted_failure_types": ["hardware_failure"],
            "contributing_factors": ["temperature_increase"],
            "severity": "medium",
            "recommended_actions": ["monitor_temperature"]
        }"""
        
        result = analyzer._parse_llm_response(response_content)
        
        assert result["failure_probability"] == 0.65
        assert result["confidence"] == 0.80
        assert result["predicted_failure_types"] == ["hardware_failure"]
        assert result["contributing_factors"] == ["temperature_increase"]
        assert result["severity"] == "medium"
        assert result["recommended_actions"] == ["monitor_temperature"]
    
    @pytest.mark.unit
    def test_parse_llm_response_missing_fields(self, analyzer):
        """Test LLM response parsing with missing fields."""
        response_content = """{
            "failure_probability": 0.65
        }"""
        
        result = analyzer._parse_llm_response(response_content)
        
        # Should provide defaults for missing fields
        assert result["failure_probability"] == 0.65
        assert result["confidence"] == 0.5  # Default
        assert result["predicted_failure_types"] == []
        assert result["contributing_factors"] == []
        assert result["recommended_actions"] == []


class TestHistoricalPatternDatabase:
    """Test suite for HistoricalPatternDatabase."""
    
    @pytest.fixture
    def pattern_db(self):
        """Create pattern database for testing."""
        return HistoricalPatternDatabase()
    
    @pytest.fixture
    def sample_patterns(self):
        """Create sample patterns for testing."""
        return [
            SystemBehaviorPattern(
                pattern_id="pattern_001",
                system_id="system_001",
                pattern_type="cpu_spike",
                description="CPU spikes during batch processing",
                frequency="daily",
                confidence=0.9,
                occurrence_count=25
            ),
            SystemBehaviorPattern(
                pattern_id="pattern_002",
                system_id="system_001", 
                pattern_type="memory_leak",
                description="Gradual memory usage increase",
                frequency="weekly",
                confidence=0.85,
                occurrence_count=12
            ),
            SystemBehaviorPattern(
                pattern_id="pattern_003",
                system_id="system_002",
                pattern_type="network_timeout",
                description="Network timeouts during peak hours",
                frequency="daily",
                confidence=0.75,
                occurrence_count=18
            )
        ]
    
    @pytest.mark.unit
    def test_add_pattern(self, pattern_db, sample_patterns):
        """Test adding pattern to database."""
        pattern = sample_patterns[0]
        
        pattern_db.add_pattern(pattern)
        
        assert len(pattern_db.patterns) == 1
        stored_pattern = pattern_db.patterns[0]
        assert stored_pattern.pattern_id == pattern.pattern_id
        assert stored_pattern.system_id == pattern.system_id
        assert stored_pattern.pattern_type == pattern.pattern_type
    
    @pytest.mark.unit
    def test_get_patterns_all(self, pattern_db, sample_patterns):
        """Test getting all patterns."""
        for pattern in sample_patterns:
            pattern_db.add_pattern(pattern)
        
        all_patterns = pattern_db.get_patterns()
        
        assert len(all_patterns) == 3
        pattern_ids = [p.pattern_id for p in all_patterns]
        assert "pattern_001" in pattern_ids
        assert "pattern_002" in pattern_ids
        assert "pattern_003" in pattern_ids
    
    @pytest.mark.unit
    def test_get_patterns_by_system(self, pattern_db, sample_patterns):
        """Test getting patterns for specific system."""
        for pattern in sample_patterns:
            pattern_db.add_pattern(pattern)
        
        system_patterns = pattern_db.get_patterns(system_id="system_001")
        
        assert len(system_patterns) == 2
        for pattern in system_patterns:
            assert pattern.system_id == "system_001"
    
    @pytest.mark.unit
    def test_get_patterns_by_type(self, pattern_db, sample_patterns):
        """Test getting patterns by type."""
        for pattern in sample_patterns:
            pattern_db.add_pattern(pattern)
        
        cpu_patterns = pattern_db.get_patterns(pattern_type="cpu_spike")
        
        assert len(cpu_patterns) == 1
        assert cpu_patterns[0].pattern_type == "cpu_spike"
    
    @pytest.mark.unit
    def test_get_patterns_by_confidence_threshold(self, pattern_db, sample_patterns):
        """Test getting patterns by confidence threshold."""
        for pattern in sample_patterns:
            pattern_db.add_pattern(pattern)
        
        high_confidence_patterns = pattern_db.get_patterns(min_confidence=0.8)
        
        assert len(high_confidence_patterns) == 2
        for pattern in high_confidence_patterns:
            assert pattern.confidence >= 0.8
    
    @pytest.mark.unit
    def test_update_pattern_confidence(self, pattern_db, sample_patterns):
        """Test updating pattern confidence."""
        pattern = sample_patterns[0]
        pattern_db.add_pattern(pattern)
        
        # Update confidence
        pattern_db.update_pattern_confidence(pattern.pattern_id, 0.95)
        
        updated_pattern = pattern_db.get_patterns()[0]
        assert updated_pattern.confidence == 0.95
    
    @pytest.mark.unit
    def test_remove_pattern(self, pattern_db, sample_patterns):
        """Test removing pattern from database."""
        pattern = sample_patterns[0]
        pattern_db.add_pattern(pattern)
        
        assert len(pattern_db.patterns) == 1
        
        pattern_db.remove_pattern(pattern.pattern_id)
        
        assert len(pattern_db.patterns) == 0
    
    @pytest.mark.unit
    def test_get_pattern_summary(self, pattern_db, sample_patterns):
        """Test getting pattern database summary."""
        for pattern in sample_patterns:
            pattern_db.add_pattern(pattern)
        
        summary = pattern_db.get_pattern_summary()
        
        assert summary["total_patterns"] == 3
        assert "system_001" in summary["patterns_by_system"]
        assert "system_002" in summary["patterns_by_system"]
        assert summary["patterns_by_system"]["system_001"] == 2
        assert summary["patterns_by_system"]["system_002"] == 1
        assert "cpu_spike" in summary["patterns_by_type"]
        assert "memory_leak" in summary["patterns_by_type"]
        assert "network_timeout" in summary["patterns_by_type"]


class TestExpertKnowledgeBase:
    """Test suite for ExpertKnowledgeBase."""
    
    @pytest.mark.unit
    def test_expert_knowledge_base_creation(self):
        """Test expert knowledge base creation."""
        knowledge_base = ExpertKnowledgeBase()
        
        # Should have default knowledge categories
        assert "critical_thresholds" in knowledge_base.knowledge
        assert "failure_indicators" in knowledge_base.knowledge
        assert "maintenance_patterns" in knowledge_base.knowledge
        assert "recovery_procedures" in knowledge_base.knowledge
    
    @pytest.mark.unit
    def test_add_knowledge_item(self):
        """Test adding knowledge item."""
        knowledge_base = ExpertKnowledgeBase()
        
        knowledge_base.add_knowledge_item(
            "custom_category",
            "custom_key",
            "custom_value"
        )
        
        assert "custom_category" in knowledge_base.knowledge
        assert knowledge_base.knowledge["custom_category"]["custom_key"] == "custom_value"
    
    @pytest.mark.unit
    def test_get_knowledge_by_category(self):
        """Test getting knowledge by category."""
        knowledge_base = ExpertKnowledgeBase()
        
        thresholds = knowledge_base.get_knowledge("critical_thresholds")
        
        assert isinstance(thresholds, dict)
        assert "cpu_critical" in thresholds
        assert "memory_critical" in thresholds
    
    @pytest.mark.unit
    def test_get_all_knowledge(self):
        """Test getting all knowledge."""
        knowledge_base = ExpertKnowledgeBase()
        
        all_knowledge = knowledge_base.get_all_knowledge()
        
        assert isinstance(all_knowledge, dict)
        assert "critical_thresholds" in all_knowledge
        assert "failure_indicators" in all_knowledge
        assert "maintenance_patterns" in all_knowledge
    
    @pytest.mark.unit
    def test_update_knowledge_category(self):
        """Test updating entire knowledge category."""
        knowledge_base = ExpertKnowledgeBase()
        
        new_thresholds = {
            "cpu_warning": 70.0,
            "cpu_critical": 85.0,
            "memory_warning": 75.0,
            "memory_critical": 90.0
        }
        
        knowledge_base.update_knowledge_category("critical_thresholds", new_thresholds)
        
        updated_thresholds = knowledge_base.get_knowledge("critical_thresholds")
        assert updated_thresholds == new_thresholds


class TestIntegrationScenarios:
    """Test integration scenarios for enhanced detector."""
    
    @pytest.mark.integration
    async def test_end_to_end_failure_prediction(self):
        """Test complete end-to-end failure prediction flow."""
        # Setup
        config = Config()
        config.legacy_system_whisperer = LegacySystemWhispererConfig()
        
        detector = EnhancedLegacySystemDetector(config)
        
        # Mock dependencies
        mock_llm_service = Mock(spec=UnifiedLLMService)
        mock_llm_response = Mock()
        mock_llm_response.content = """{
            "failure_probability": 0.68,
            "confidence": 0.85,
            "predicted_failure_types": ["performance_degradation"],
            "contributing_factors": ["high_resource_utilization"],
            "time_to_failure_estimate": 20,
            "severity": "medium",
            "recommended_actions": ["monitor_resources", "schedule_maintenance"]
        }"""
        mock_llm_service.process_request = AsyncMock(return_value=mock_llm_response)
        
        mock_metrics = Mock(spec=AIEngineMetrics)
        
        # Initialize detector
        await detector.initialize_enhanced(mock_llm_service, mock_metrics)
        
        # Create system context
        system_context = LegacySystemContext(
            system_id="integration_test_system",
            system_name="Integration Test System",
            system_type=SystemType.MAINFRAME,
            criticality=Criticality.HIGH
        )
        detector.add_system_context(system_context)
        
        # Add historical patterns
        pattern = SystemBehaviorPattern(
            pattern_id="test_pattern",
            system_id=system_context.system_id,
            pattern_type="performance_decline",
            confidence=0.8
        )
        detector.historical_patterns.add_pattern(pattern)
        
        # Mock base anomaly detection
        with patch.object(detector, 'detect_anomaly') as mock_detect:
            mock_detect.return_value = 0.7
            
            # Perform prediction
            system_data = {
                "cpu_utilization": 80.0,
                "memory_utilization": 75.0,
                "response_time_ms": 200.0,
                "error_rate": 0.03
            }
            
            prediction = await detector.predict_system_failure(
                system_data,
                system_context,
                prediction_horizon=30
            )
            
            # Verify complete prediction
            assert prediction.system_id == system_context.system_id
            assert prediction.failure_probability == 0.68
            assert prediction.confidence == 0.85
            assert "performance_degradation" in prediction.predicted_failure_types
            assert "high_resource_utilization" in prediction.contributing_factors
            assert prediction.time_to_failure_days == 20
            assert prediction.severity.value == "medium"
            assert len(prediction.recommended_actions) == 2
    
    @pytest.mark.performance
    async def test_detector_performance_with_multiple_systems(self):
        """Test detector performance with multiple systems."""
        import time
        
        # Setup
        config = Config()
        config.legacy_system_whisperer = LegacySystemWhispererConfig()
        detector = EnhancedLegacySystemDetector(config)
        
        # Mock fast LLM service
        mock_llm_service = Mock(spec=UnifiedLLMService)
        mock_llm_response = Mock()
        mock_llm_response.content = '{"failure_probability": 0.3, "confidence": 0.8}'
        mock_llm_service.process_request = AsyncMock(return_value=mock_llm_response)
        
        mock_metrics = Mock(spec=AIEngineMetrics)
        
        await detector.initialize_enhanced(mock_llm_service, mock_metrics)
        
        # Create multiple system contexts
        systems = []
        for i in range(10):
            system = LegacySystemContext(
                system_id=f"perf_test_system_{i}",
                system_name=f"Performance Test System {i}",
                system_type=SystemType.MAINFRAME,
                criticality=Criticality.MEDIUM
            )
            systems.append(system)
            detector.add_system_context(system)
        
        # Mock base detector
        with patch.object(detector, 'detect_anomaly') as mock_detect:
            mock_detect.return_value = 0.5
            
            # Measure prediction performance
            start_time = time.time()
            
            system_data = {
                "cpu_utilization": 60.0,
                "memory_utilization": 50.0,
                "response_time_ms": 100.0,
                "error_rate": 0.01
            }
            
            # Perform predictions for all systems
            predictions = []
            for system in systems:
                prediction = await detector.predict_system_failure(
                    system_data,
                    system,
                    prediction_horizon=30
                )
                predictions.append(prediction)
            
            total_time = time.time() - start_time
            
            # Should process 10 systems in reasonable time (< 5 seconds)
            assert total_time < 5.0
            assert len(predictions) == 10
            
            # All predictions should be valid
            for prediction in predictions:
                assert prediction.failure_probability >= 0
                assert prediction.confidence >= 0


# Utility functions for testing
def create_mock_system_data(
    cpu: float = 50.0,
    memory: float = 60.0,
    response_time: float = 100.0,
    error_rate: float = 0.01
) -> Dict[str, float]:
    """Create mock system data for testing."""
    return {
        "cpu_utilization": cpu,
        "memory_utilization": memory,
        "disk_utilization": 40.0,
        "network_utilization": 30.0,
        "response_time_ms": response_time,
        "error_rate": error_rate,
        "transaction_rate": 1000.0
    }


def create_test_behavior_pattern(
    pattern_id: str,
    system_id: str,
    pattern_type: str = "test_pattern",
    confidence: float = 0.8
) -> SystemBehaviorPattern:
    """Create test behavior pattern."""
    return SystemBehaviorPattern(
        pattern_id=pattern_id,
        system_id=system_id,
        pattern_type=pattern_type,
        description=f"Test pattern for {system_id}",
        confidence=confidence,
        occurrence_count=10
    )


def assert_valid_failure_prediction(prediction: SystemFailurePrediction):
    """Assert that failure prediction is valid."""
    assert prediction.system_id is not None
    assert 0 <= prediction.failure_probability <= 1
    assert 0 <= prediction.confidence <= 1
    assert len(prediction.predicted_failure_types) >= 0
    assert len(prediction.contributing_factors) >= 0
    assert len(prediction.recommended_actions) >= 0
    assert prediction.prediction_timestamp is not None