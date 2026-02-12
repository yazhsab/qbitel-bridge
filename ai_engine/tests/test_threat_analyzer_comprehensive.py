"""
Comprehensive Unit Tests for Threat Analyzer
Tests for ai_engine/security/threat_analyzer.py
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from ai_engine.security.threat_analyzer import (
    ThreatAnalyzer,
    ThreatClassificationModel,
    FeatureExtractor,
)
from ai_engine.security.models import (
    SecurityEvent,
    ThreatAnalysis,
    SecurityContext,
    LegacySystem,
    SecurityEventType,
    ThreatLevel,
    ConfidenceLevel,
    ThreatIntelligence,
    SystemCriticality,
    ProtocolType,
)
from ai_engine.core.config import Config
from ai_engine.core.exceptions import ThreatAnalysisException
from ai_engine.models.base import ModelInput, ModelOutput


class TestFeatureExtractor:
    """Test FeatureExtractor class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Mock(spec=Config)

    @pytest.fixture
    def extractor(self, config):
        """Create feature extractor instance."""
        return FeatureExtractor(config)

    def test_extractor_initialization(self, extractor):
        """Test feature extractor initialization."""
        assert extractor.config is not None
        assert extractor.max_sequence_length == 512
        assert extractor.vocab_size == 10000

    def test_extract_event_features(self, extractor):
        """Test extracting features from security event."""
        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.MALWARE_DETECTION,
            threat_level=ThreatLevel.HIGH,
            confidence_score=0.9,
            description="Malware detected on system",
            event_timestamp=datetime.utcnow(),
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            affected_systems=["server1", "server2"],
            indicators_of_compromise=["hash123", "domain.com"],
            attack_vectors=["phishing", "exploit"],
        )

        features = extractor.extract_event_features(event)

        assert isinstance(features, np.ndarray)
        assert features.shape == (256,)
        assert features.dtype == np.float32

    def test_extract_basic_features(self, extractor):
        """Test extracting basic event features."""
        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.INTRUSION_ATTEMPT,
            threat_level=ThreatLevel.MEDIUM,
            confidence_score=0.8,
            description="Test event",
            event_timestamp=datetime.utcnow(),
        )

        features = extractor._extract_basic_features(event)

        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, float) for f in features)

    def test_extract_network_features(self, extractor):
        """Test extracting network features."""
        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.NETWORK_SCAN,
            threat_level=ThreatLevel.LOW,
            confidence_score=0.7,
            description="Network scan detected",
            event_timestamp=datetime.utcnow(),
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            network_artifacts=["packet1", "packet2"],
            affected_protocols=["tcp", "http"],
        )

        features = extractor._extract_network_features(event)

        assert isinstance(features, list)
        assert len(features) > 0

    def test_extract_temporal_features(self, extractor):
        """Test extracting temporal features."""
        now = datetime.utcnow()
        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
            threat_level=ThreatLevel.MEDIUM,
            confidence_score=0.75,
            description="Anomaly detected",
            event_timestamp=now,
            detection_timestamp=now + timedelta(minutes=5),
        )

        features = extractor._extract_temporal_features(event)

        assert isinstance(features, list)
        assert len(features) > 0

    def test_extract_content_features(self, extractor):
        """Test extracting content features."""
        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.MALWARE_DETECTION,
            threat_level=ThreatLevel.HIGH,
            confidence_score=0.9,
            description="Malware detected with suspicious behavior",
            event_timestamp=datetime.utcnow(),
            indicators_of_compromise=["hash1", "hash2", "hash3"],
            attack_vectors=["exploit", "phishing"],
            affected_systems=["server1"],
            file_artifacts=["malware.exe"],
        )

        features = extractor._extract_content_features(event)

        assert isinstance(features, list)
        assert len(features) > 0

    def test_encode_ip_address_ipv4(self, extractor):
        """Test encoding IPv4 address."""
        ip_features = extractor._encode_ip_address("192.168.1.100")

        assert len(ip_features) == 4
        assert all(0 <= f <= 1 for f in ip_features)

    def test_encode_ip_address_invalid(self, extractor):
        """Test encoding invalid IP address."""
        ip_features = extractor._encode_ip_address("invalid")

        assert len(ip_features) == 4
        assert all(f == 0.0 for f in ip_features)

    def test_extract_text_features(self, extractor):
        """Test extracting text features."""
        text = "Malware detected with suspicious ransomware activity"

        features = extractor._extract_text_features(text)

        assert isinstance(features, list)
        assert len(features) > 0


class TestThreatClassificationModel:
    """Test ThreatClassificationModel class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Mock(spec=Config)

    @pytest.fixture
    def model(self, config):
        """Create classification model instance."""
        return ThreatClassificationModel(config)

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.num_classes == len(SecurityEventType)
        assert model.feature_dim == 256
        assert model.features is not None
        assert model.classifier is not None

    def test_model_forward(self, model):
        """Test model forward pass."""
        import torch

        x = torch.randn(10, 256)
        output = model.forward(x)

        assert output.shape == (10, model.num_classes)
        # Softmax output should sum to 1
        assert torch.allclose(output.sum(dim=1), torch.ones(10), atol=1e-5)

    def test_model_predict(self, model):
        """Test model prediction."""
        import torch

        input_data = ModelInput(data=torch.randn(1, 256))

        output = model.predict(input_data)

        assert isinstance(output, ModelOutput)
        assert output.predictions is not None
        assert output.confidence is not None
        assert "predicted_class" in output.metadata

    def test_model_validate_input(self, model):
        """Test input validation."""
        import torch

        valid_input = ModelInput(data=torch.randn(1, 256))
        assert model.validate_input(valid_input) is True

        invalid_input = ModelInput(data=torch.randn(1, 128))
        assert model.validate_input(invalid_input) is False

    def test_model_get_input_schema(self, model):
        """Test getting input schema."""
        schema = model.get_input_schema()

        assert schema["type"] == "tensor"
        assert schema["shape"] == [-1, 256]

    def test_model_get_output_schema(self, model):
        """Test getting output schema."""
        schema = model.get_output_schema()

        assert "predictions" in schema
        assert schema["predictions"]["type"] == "tensor"


class TestThreatAnalyzer:
    """Test ThreatAnalyzer class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Mock(spec=Config)

    @pytest.fixture
    def analyzer(self, config):
        """Create threat analyzer instance."""
        return ThreatAnalyzer(config)

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config is not None
        assert analyzer.llm_service is None
        assert analyzer.classification_model is None
        assert isinstance(analyzer.threat_intelligence, dict)
        assert isinstance(analyzer.analysis_cache, dict)
        assert analyzer._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_analyzer(self, analyzer):
        """Test analyzer initialization."""
        with patch("ai_engine.security.threat_analyzer.get_llm_service") as mock_llm:
            mock_llm_instance = AsyncMock()
            mock_llm_instance._initialized = False
            mock_llm.return_value = mock_llm_instance

            with patch.object(analyzer, "_load_threat_intelligence", new_callable=AsyncMock):
                await analyzer.initialize()

                assert analyzer._initialized is True
                assert analyzer.llm_service is not None
                assert analyzer.classification_model is not None

    @pytest.mark.asyncio
    async def test_initialize_analyzer_failure(self, analyzer):
        """Test analyzer initialization failure."""
        with patch(
            "ai_engine.security.threat_analyzer.get_llm_service",
            side_effect=Exception("Init failed"),
        ):
            with pytest.raises(ThreatAnalysisException):
                await analyzer.initialize()

    @pytest.mark.asyncio
    async def test_analyze_threat(self, analyzer):
        """Test threat analysis."""
        # Setup
        analyzer._initialized = True
        analyzer.llm_service = AsyncMock()
        analyzer.llm_service.process_request = AsyncMock(return_value=Mock(content="Analysis complete"))
        analyzer.classification_model = Mock()

        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.MALWARE_DETECTION,
            threat_level=ThreatLevel.HIGH,
            confidence_score=0.9,
            description="Malware detected",
            event_timestamp=datetime.utcnow(),
            affected_systems=["server1"],
            indicators_of_compromise=["hash123"],
            attack_vectors=["exploit"],
        )

        with patch.object(analyzer, "_perform_ml_classification", new_callable=AsyncMock) as mock_ml:
            mock_ml.return_value = {
                "predicted_type": SecurityEventType.MALWARE_DETECTION,
                "confidence": 0.9,
                "ml_confidence": 0.9,
            }

            with patch.object(analyzer, "_assess_threat_severity", new_callable=AsyncMock) as mock_severity:
                mock_severity.return_value = {
                    "threat_level": ThreatLevel.HIGH,
                    "severity_score": 0.8,
                    "base_severity": ThreatLevel.HIGH,
                    "adjustments": {},
                }

                with patch.object(analyzer, "_analyze_context", new_callable=AsyncMock) as mock_context:
                    mock_context.return_value = {
                        "llm_analysis": "Context analysis",
                        "context_score": 0.7,
                        "key_factors": [],
                        "business_impact_indicators": [],
                    }

                    with patch.object(
                        analyzer,
                        "_correlate_threat_intelligence",
                        new_callable=AsyncMock,
                    ) as mock_intel:
                        mock_intel.return_value = {
                            "correlations": [],
                            "relevance_score": 0.5,
                            "matched_intelligence_count": 0,
                            "threat_actors": [],
                            "associated_campaigns": [],
                        }

                        with patch.object(analyzer, "_assess_business_impact", new_callable=AsyncMock) as mock_impact:
                            mock_impact.return_value = {
                                "business_impact_score": 0.6,
                                "impact_factors": [],
                                "estimated_financial_impact": 50000,
                                "operational_impact_level": "high",
                                "recovery_time_estimate": 8,
                            }

                            result = await analyzer.analyze_threat(event)

                            assert isinstance(result, ThreatAnalysis)
                            assert result.event_id == "test_123"
                            assert result.threat_level == ThreatLevel.HIGH

    @pytest.mark.asyncio
    async def test_analyze_threat_with_cache(self, analyzer):
        """Test threat analysis with cache hit."""
        analyzer._initialized = True

        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.MALWARE_DETECTION,
            threat_level=ThreatLevel.HIGH,
            confidence_score=0.9,
            description="Malware detected",
            event_timestamp=datetime.utcnow(),
        )

        # Create cached analysis
        cached_analysis = ThreatAnalysis(
            event_id="test_123",
            threat_classification=SecurityEventType.MALWARE_DETECTION,
            threat_level=ThreatLevel.HIGH,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.9,
        )

        cache_key = analyzer._create_cache_key(event)
        analyzer.analysis_cache[cache_key] = cached_analysis

        result = await analyzer.analyze_threat(event)

        assert result == cached_analysis

    @pytest.mark.asyncio
    async def test_analyze_threat_not_initialized(self, analyzer):
        """Test analysis when not initialized."""
        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.MALWARE_DETECTION,
            threat_level=ThreatLevel.HIGH,
            confidence_score=0.9,
            description="Test",
            event_timestamp=datetime.utcnow(),
        )

        with patch.object(analyzer, "initialize", new_callable=AsyncMock):
            with patch.object(analyzer, "_perform_ml_classification", side_effect=Exception("Failed")):
                with pytest.raises(ThreatAnalysisException):
                    await analyzer.analyze_threat(event)

    def test_rule_based_classification(self, analyzer):
        """Test rule-based classification fallback."""
        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.MALWARE_DETECTION,
            threat_level=ThreatLevel.HIGH,
            confidence_score=0.9,
            description="Test",
            event_timestamp=datetime.utcnow(),
            indicators_of_compromise=["ioc1", "ioc2"],
            attack_vectors=["vector1"],
        )

        result = analyzer._rule_based_classification(event)

        assert result["predicted_type"] == SecurityEventType.MALWARE_DETECTION
        assert result["confidence"] > 0.6

    def test_threat_level_to_score(self, analyzer):
        """Test converting threat level to score."""
        assert analyzer._threat_level_to_score(ThreatLevel.CRITICAL) == 1.0
        assert analyzer._threat_level_to_score(ThreatLevel.HIGH) == 0.8
        assert analyzer._threat_level_to_score(ThreatLevel.MEDIUM) == 0.6
        assert analyzer._threat_level_to_score(ThreatLevel.LOW) == 0.4
        assert analyzer._threat_level_to_score(ThreatLevel.INFO) == 0.2

    def test_score_to_threat_level(self, analyzer):
        """Test converting score to threat level."""
        assert analyzer._score_to_threat_level(0.95) == ThreatLevel.CRITICAL
        assert analyzer._score_to_threat_level(0.75) == ThreatLevel.HIGH
        assert analyzer._score_to_threat_level(0.55) == ThreatLevel.MEDIUM
        assert analyzer._score_to_threat_level(0.35) == ThreatLevel.LOW
        assert analyzer._score_to_threat_level(0.15) == ThreatLevel.INFO

    def test_confidence_score_to_level(self, analyzer):
        """Test converting confidence score to level."""
        assert analyzer._confidence_score_to_level(0.96) == ConfidenceLevel.VERY_HIGH
        assert analyzer._confidence_score_to_level(0.88) == ConfidenceLevel.HIGH
        assert analyzer._confidence_score_to_level(0.75) == ConfidenceLevel.MEDIUM
        assert analyzer._confidence_score_to_level(0.55) == ConfidenceLevel.LOW
        assert analyzer._confidence_score_to_level(0.40) == ConfidenceLevel.VERY_LOW

    def test_calculate_context_severity_multiplier(self, analyzer):
        """Test calculating context severity multiplier."""
        context = SecurityContext(
            current_threat_level=ThreatLevel.CRITICAL,
            active_incidents=["inc1", "inc2", "inc3", "inc4"],
            business_hours=True,
        )

        multiplier = analyzer._calculate_context_severity_multiplier(context)

        assert multiplier > 1.0

    def test_calculate_exploitability_score(self, analyzer):
        """Test calculating exploitability score."""
        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.ZERO_DAY_EXPLOIT,
            threat_level=ThreatLevel.CRITICAL,
            confidence_score=0.95,
            description="Zero-day exploit",
            event_timestamp=datetime.utcnow(),
            indicators_of_compromise=["ioc" + str(i) for i in range(10)],
            attack_vectors=["v" + str(i) for i in range(5)],
        )

        score = analyzer._calculate_exploitability_score(event)

        assert 0 <= score <= 1.0
        assert score > 0.5  # Should be high for zero-day

    def test_generate_immediate_actions(self, analyzer):
        """Test generating immediate actions."""
        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.RANSOMWARE_ACTIVITY,
            threat_level=ThreatLevel.CRITICAL,
            confidence_score=0.95,
            description="Ransomware detected",
            event_timestamp=datetime.utcnow(),
            source_ip="192.168.1.100",
        )

        severity = {"threat_level": ThreatLevel.CRITICAL, "severity_score": 0.95}

        actions = analyzer._generate_immediate_actions(event, severity)

        assert isinstance(actions, list)
        assert len(actions) > 0
        assert any("Alert" in action for action in actions)

    def test_map_to_mitre_techniques(self, analyzer):
        """Test mapping to MITRE ATT&CK techniques."""
        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.DATA_EXFILTRATION,
            threat_level=ThreatLevel.HIGH,
            confidence_score=0.9,
            description="Data exfiltration detected",
            event_timestamp=datetime.utcnow(),
        )

        techniques = analyzer._map_to_mitre_techniques(event)

        assert isinstance(techniques, list)
        assert len(techniques) > 0
        assert all(t.startswith("T") for t in techniques)

    def test_infer_attack_methodology(self, analyzer):
        """Test inferring attack methodology."""
        event = SecurityEvent(
            event_id="test_123",
            event_type=SecurityEventType.RANSOMWARE_ACTIVITY,
            threat_level=ThreatLevel.CRITICAL,
            confidence_score=0.95,
            description="Ransomware attack",
            event_timestamp=datetime.utcnow(),
            attack_vectors=["phishing", "exploit"],
        )

        methodology = analyzer._infer_attack_methodology(event)

        assert isinstance(methodology, list)
        assert len(methodology) > 0

    @pytest.mark.asyncio
    async def test_update_threat_intelligence(self, analyzer):
        """Test updating threat intelligence."""
        intelligence = ThreatIntelligence(
            source="test_feed",
            threat_type=SecurityEventType.MALWARE_DETECTION,
            threat_actors=["APT28"],
            campaigns=["Campaign1"],
            iocs=[{"type": "ip", "value": "1.2.3.4"}],
            ttps=["ttp1"],
            severity=ThreatLevel.HIGH,
            confidence=0.9,
        )

        await analyzer.update_threat_intelligence(intelligence)

        assert intelligence.intelligence_id in analyzer.threat_intelligence

    def test_get_cached_analysis(self, analyzer):
        """Test getting cached analysis."""
        analysis = ThreatAnalysis(
            event_id="test_123",
            threat_classification=SecurityEventType.MALWARE_DETECTION,
            threat_level=ThreatLevel.HIGH,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.9,
        )

        cache_key = "test_key"
        analyzer.analysis_cache[cache_key] = analysis

        result = analyzer.get_cached_analysis(cache_key)

        assert result == analysis

    def test_clear_analysis_cache(self, analyzer):
        """Test clearing analysis cache."""
        analyzer.analysis_cache["key1"] = Mock()
        analyzer.analysis_cache["key2"] = Mock()

        analyzer.clear_analysis_cache()

        assert len(analyzer.analysis_cache) == 0

    @pytest.mark.asyncio
    async def test_get_analyzer_metrics(self, analyzer):
        """Test getting analyzer metrics."""
        analyzer.analysis_cache["key1"] = Mock()
        analyzer.threat_intelligence["intel1"] = Mock()
        analyzer.classification_model = Mock()

        metrics = await analyzer.get_analyzer_metrics()

        assert "total_analyses" in metrics
        assert "threat_intelligence_entries" in metrics
        assert "classification_model_loaded" in metrics
        assert metrics["total_analyses"] == 1
        assert metrics["threat_intelligence_entries"] == 1

    @pytest.mark.asyncio
    async def test_shutdown_analyzer(self, analyzer):
        """Test shutting down analyzer."""
        analyzer._initialized = True
        analyzer.analysis_cache["key1"] = Mock()

        await analyzer.shutdown()

        assert analyzer._initialized is False
        assert len(analyzer.analysis_cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
