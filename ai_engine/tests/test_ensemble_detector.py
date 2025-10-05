"""
Tests for ai_engine/anomaly/ensemble_detector.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime


class TestAnomalyResult:
    """Test suite for AnomalyResult dataclass."""

    def test_anomaly_result_creation(self):
        """Test AnomalyResult creation."""
        from ai_engine.anomaly.ensemble_detector import AnomalyResult
        
        result = AnomalyResult(
            score=0.8,
            confidence=0.9,
            is_anomalous=True,
            individual_scores={"test": 0.8},
            explanation="Test anomaly"
        )
        
        assert result.score == 0.8
        assert result.confidence == 0.9
        assert result.is_anomalous is True

    def test_anomaly_result_to_dict(self):
        """Test AnomalyResult to_dict conversion."""
        from ai_engine.anomaly.ensemble_detector import AnomalyResult
        
        result = AnomalyResult(
            score=0.7,
            confidence=0.85,
            is_anomalous=False
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["score"] == 0.7
        assert result_dict["is_anomalous"] is False
        assert "timestamp" in result_dict


class TestEnsembleAnomalyDetector:
    """Test suite for EnsembleAnomalyDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        from ai_engine.anomaly.ensemble_detector import EnsembleAnomalyDetector
        return EnsembleAnomalyDetector()

    @pytest.mark.asyncio
    async def test_detect_with_numpy_array(self, detector):
        """Test detection with numpy array."""
        features = np.random.randn(100)
        result = await detector.detect(features)
        
        assert "score" in result
        assert "confidence" in result
        assert "is_anomalous" in result
        assert isinstance(result["is_anomalous"], bool)

    @pytest.mark.asyncio
    async def test_detect_with_list(self, detector):
        """Test detection with list."""
        features = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = await detector.detect(features)
        
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_detect_with_dict_features_key(self, detector):
        """Test detection with dict containing 'features' key."""
        features = {"features": [0.1, 0.2, 0.3, 0.4]}
        result = await detector.detect(features)
        
        assert "score" in result

    @pytest.mark.asyncio
    async def test_detect_with_dict_vector_key(self, detector):
        """Test detection with dict containing 'vector' key."""
        features = {"vector": [0.5, 0.6, 0.7]}
        result = await detector.detect(features)
        
        assert "score" in result

    @pytest.mark.asyncio
    async def test_detect_with_dict_values(self, detector):
        """Test detection with dict values."""
        features = {"a": 0.1, "b": 0.2, "c": 0.3}
        result = await detector.detect(features)
        
        assert "score" in result

    @pytest.mark.asyncio
    async def test_detect_with_context(self, detector):
        """Test detection with context."""
        features = np.random.randn(50)
        context = {"source": "192.168.1.1"}
        
        result = await detector.detect(features, context=context)
        
        assert "explanation" in result
        assert "192.168.1.1" in result["explanation"]

    @pytest.mark.asyncio
    async def test_detect_empty_features(self, detector):
        """Test detection with empty features."""
        from ai_engine.core.exceptions import AnomalyDetectionException
        
        features = []
        
        with pytest.raises(AnomalyDetectionException, match="Empty feature vector"):
            await detector.detect(features)

    @pytest.mark.asyncio
    async def test_detect_batch(self, detector):
        """Test batch detection."""
        batch_features = [
            np.random.randn(50),
            np.random.randn(50),
            np.random.randn(50)
        ]
        
        results = await detector.detect_batch(batch_features)
        
        assert len(results) == 3
        assert all("score" in r for r in results)

    @pytest.mark.asyncio
    async def test_update_baseline(self, detector):
        """Test baseline update."""
        samples = [b"test data 1", b"test data 2", b"test data 3"]
        
        await detector.update_baseline(samples)
        
        assert detector._baseline_mean is not None
        assert detector._baseline_std is not None

    @pytest.mark.asyncio
    async def test_update_baseline_empty(self, detector):
        """Test baseline update with empty samples."""
        from ai_engine.core.exceptions import AnomalyDetectionException
        
        with pytest.raises(AnomalyDetectionException, match="No samples supplied"):
            await detector.update_baseline([])

    @pytest.mark.asyncio
    async def test_detect_with_baseline(self, detector):
        """Test detection after baseline update."""
        # Update baseline
        samples = [b"normal data" * 10 for _ in range(10)]
        await detector.update_baseline(samples)
        
        # Detect with normal data
        features = np.random.randn(50) * 0.1 + 0.5
        result = await detector.detect(features)
        
        assert "score" in result

    def test_normalize_features_numpy(self, detector):
        """Test feature normalization with numpy array."""
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = detector._normalize_features(features)
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_normalize_features_constant(self, detector):
        """Test feature normalization with constant values."""
        features = np.array([5.0, 5.0, 5.0])
        normalized = detector._normalize_features(features)
        
        assert np.all(normalized == 0.0)

    def test_bytes_to_vector(self, detector):
        """Test bytes to vector conversion."""
        data = b"test data"
        vector = detector._bytes_to_vector(data)
        
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32

    def test_statistical_deviation_no_baseline(self, detector):
        """Test statistical deviation without baseline."""
        vector = np.random.randn(100)
        score = detector._statistical_deviation(vector)
        
        assert 0.0 <= score <= 1.0

    def test_statistical_deviation_with_baseline(self, detector):
        """Test statistical deviation with baseline."""
        detector._baseline_mean = 0.5
        detector._baseline_std = 0.1
        
        vector = np.random.randn(100) * 0.1 + 0.5
        score = detector._statistical_deviation(vector)
        
        assert 0.0 <= score <= 1.0

    def test_volatility_score(self, detector):
        """Test volatility score calculation."""
        vector = np.random.randn(100)
        score = detector._volatility_score(vector)
        
        assert 0.0 <= score <= 1.0

    def test_volatility_score_small_vector(self, detector):
        """Test volatility score with small vector."""
        vector = np.array([1.0])
        score = detector._volatility_score(vector)
        
        assert score == 0.0

    def test_pattern_change_score(self, detector):
        """Test pattern change score."""
        vector = np.random.randn(100)
        score = detector._pattern_change_score(vector)
        
        assert 0.0 <= score <= 1.0

    def test_pattern_change_score_small_vector(self, detector):
        """Test pattern change score with small vector."""
        vector = np.array([1.0, 2.0])
        score = detector._pattern_change_score(vector)
        
        assert score == 0.0

    def test_confidence_calculation(self, detector):
        """Test confidence calculation."""
        vector = np.random.randn(100)
        confidence = detector._confidence(vector)
        
        assert 0.0 <= confidence <= 1.0

    def test_build_explanation_anomalous(self, detector):
        """Test explanation building for anomaly."""
        scores = {"statistical": 0.8, "volatility": 0.7, "pattern": 0.6}
        explanation = detector._build_explanation(True, scores, None)
        
        assert "anomalous" in explanation

    def test_build_explanation_normal(self, detector):
        """Test explanation building for normal."""
        scores = {"statistical": 0.2, "volatility": 0.1, "pattern": 0.15}
        explanation = detector._build_explanation(False, scores, None)
        
        assert "normal" in explanation

    def test_build_explanation_with_context(self, detector):
        """Test explanation with context."""
        scores = {"statistical": 0.5}
        context = {"source": "10.0.0.1"}
        explanation = detector._build_explanation(True, scores, context)
        
        assert "10.0.0.1" in explanation

    @pytest.mark.asyncio
    async def test_detect_high_anomaly_score(self, detector):
        """Test detection with high anomaly score."""
        # Create features that should trigger anomaly
        features = np.ones(100) * 10.0  # Extreme values
        result = await detector.detect(features)
        
        assert "score" in result
        assert "is_anomalous" in result

    @pytest.mark.asyncio
    async def test_individual_scores_in_result(self, detector):
        """Test that individual scores are included in result."""
        features = np.random.randn(50)
        result = await detector.detect(features)
        
        assert "individual_scores" in result
        assert "statistical" in result["individual_scores"]
        assert "volatility" in result["individual_scores"]
        assert "pattern" in result["individual_scores"]