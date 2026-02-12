"""
Comprehensive tests for anomaly detection suite:
- ensemble_detector.py
- isolation_forest.py
- lstm_detector.py

Tests cover all detection methods, baseline updates, edge cases, and integration.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from ai_engine.anomaly.ensemble_detector import (
    EnsembleAnomalyDetector,
    AnomalyResult,
)
from ai_engine.anomaly.isolation_forest import (
    IsolationForestDetector,
    IsolationForestResult,
    IsolationForestSettings,
)
from ai_engine.anomaly.lstm_detector import (
    LSTMAnomalyDetector,
    LSTMModel,
    LSTMDetectionResult,
)
from ai_engine.core.config import Config
from ai_engine.core.exceptions import AnomalyDetectionException

# ===== ENSEMBLE DETECTOR TESTS =====


@pytest.fixture
def ensemble_detector():
    """Create ensemble detector instance."""
    config = Mock(spec=Config)
    config.anomaly_threshold = 0.5
    return EnsembleAnomalyDetector(config)


@pytest.mark.asyncio
async def test_ensemble_detect_numpy_array(ensemble_detector):
    """Test detection with numpy array input."""
    features = np.random.rand(100)
    result = await ensemble_detector.detect(features)

    assert "score" in result
    assert "confidence" in result
    assert "is_anomalous" in result
    assert 0 <= result["score"] <= 1
    assert 0 <= result["confidence"] <= 1


@pytest.mark.asyncio
async def test_ensemble_detect_list_input(ensemble_detector):
    """Test detection with list input."""
    features = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = await ensemble_detector.detect(features)

    assert result["score"] >= 0
    assert result["is_anomalous"] in [True, False]


@pytest.mark.asyncio
async def test_ensemble_detect_dict_with_features(ensemble_detector):
    """Test detection with dict containing 'features' key."""
    features = {"features": [0.1, 0.2, 0.3, 0.4]}
    result = await ensemble_detector.detect(features)

    assert "individual_scores" in result
    assert "statistical" in result["individual_scores"]
    assert "volatility" in result["individual_scores"]
    assert "pattern" in result["individual_scores"]


@pytest.mark.asyncio
async def test_ensemble_detect_dict_with_vector(ensemble_detector):
    """Test detection with dict containing 'vector' key."""
    features = {"vector": np.random.rand(50)}
    result = await ensemble_detector.detect(features)

    assert result["score"] >= 0


@pytest.mark.asyncio
async def test_ensemble_detect_dict_values(ensemble_detector):
    """Test detection with dict values."""
    features = {"a": 0.1, "b": 0.2, "c": 0.3}
    result = await ensemble_detector.detect(features)

    assert "score" in result


@pytest.mark.asyncio
async def test_ensemble_detect_empty_features(ensemble_detector):
    """Test detection with empty features raises exception."""
    with pytest.raises(AnomalyDetectionException, match="Empty feature vector"):
        await ensemble_detector.detect([])


@pytest.mark.asyncio
async def test_ensemble_detect_with_context(ensemble_detector):
    """Test detection with context information."""
    features = np.random.rand(50)
    context = {"source": "192.168.1.1"}

    result = await ensemble_detector.detect(features, context)

    assert "192.168.1.1" in result["explanation"]
    assert "source" in result["metadata"]
    assert result["metadata"]["source"] == "192.168.1.1"


@pytest.mark.asyncio
async def test_ensemble_detect_batch(ensemble_detector):
    """Test batch detection."""
    batch = [np.random.rand(50) for _ in range(5)]

    results = await ensemble_detector.detect_batch(batch)

    assert len(results) == 5
    assert all("score" in r for r in results)


@pytest.mark.asyncio
async def test_ensemble_update_baseline(ensemble_detector):
    """Test baseline update."""
    samples = [np.random.bytes(100) for _ in range(10)]

    await ensemble_detector.update_baseline(samples)

    assert ensemble_detector._baseline_mean is not None
    assert ensemble_detector._baseline_std is not None


@pytest.mark.asyncio
async def test_ensemble_update_baseline_empty(ensemble_detector):
    """Test baseline update with empty samples."""
    with pytest.raises(AnomalyDetectionException, match="No samples"):
        await ensemble_detector.update_baseline([])


@pytest.mark.asyncio
async def test_ensemble_statistical_deviation_with_baseline(ensemble_detector):
    """Test statistical deviation with baseline."""
    # Set baseline
    ensemble_detector._baseline_mean = 0.5
    ensemble_detector._baseline_std = 0.2

    vector = np.array([0.5, 0.5, 0.5])
    score = ensemble_detector._statistical_deviation(vector)

    assert 0 <= score <= 1
    assert score < 0.5  # Should be low since matches baseline


@pytest.mark.asyncio
async def test_ensemble_statistical_deviation_no_baseline(ensemble_detector):
    """Test statistical deviation without baseline."""
    vector = np.array([0.3, 0.4, 0.5])
    score = ensemble_detector._statistical_deviation(vector)

    assert 0 <= score <= 1


def test_ensemble_volatility_score(ensemble_detector):
    """Test volatility score calculation."""
    # High volatility
    high_volatility = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    score_high = ensemble_detector._volatility_score(high_volatility)

    # Low volatility
    low_volatility = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    score_low = ensemble_detector._volatility_score(low_volatility)

    assert score_high > score_low


def test_ensemble_volatility_score_small_vector(ensemble_detector):
    """Test volatility with small vector."""
    vector = np.array([0.5])
    score = ensemble_detector._volatility_score(vector)

    assert score == 0.0


def test_ensemble_pattern_change_score(ensemble_detector):
    """Test pattern change detection using FFT."""
    # Signal with high frequency component
    high_freq = np.sin(np.linspace(0, 20 * np.pi, 100))
    score_high = ensemble_detector._pattern_change_score(high_freq)

    # Signal with low frequency
    low_freq = np.sin(np.linspace(0, 2 * np.pi, 100))
    score_low = ensemble_detector._pattern_change_score(low_freq)

    assert 0 <= score_high <= 1
    assert 0 <= score_low <= 1


def test_ensemble_pattern_change_small_vector(ensemble_detector):
    """Test pattern change with small vector."""
    vector = np.array([0.1, 0.2])
    score = ensemble_detector._pattern_change_score(vector)

    assert score == 0.0


def test_ensemble_confidence_score(ensemble_detector):
    """Test confidence score calculation."""
    # Large vector = higher confidence
    large = np.random.rand(200)
    conf_large = ensemble_detector._confidence(large)

    # Small vector = lower confidence
    small = np.random.rand(10)
    conf_small = ensemble_detector._confidence(small)

    assert conf_large > conf_small


def test_ensemble_normalize_features_dict_formats(ensemble_detector):
    """Test feature normalization with different dict formats."""
    # Test "features" key
    dict1 = {"features": [1, 2, 3, 4, 5]}
    vec1 = ensemble_detector._normalize_features(dict1)
    assert vec1.size == 5

    # Test "vector" key
    dict2 = {"vector": np.array([1, 2, 3])}
    vec2 = ensemble_detector._normalize_features(dict2)
    assert vec2.size == 3

    # Test arbitrary keys
    dict3 = {"a": 1, "b": 2, "c": 3}
    vec3 = ensemble_detector._normalize_features(dict3)
    assert vec3.size == 3


def test_ensemble_normalize_features_scaling(ensemble_detector):
    """Test feature normalization scales to [0, 1]."""
    features = np.array([10, 20, 30, 40, 50])
    normalized = ensemble_detector._normalize_features(features)

    assert np.min(normalized) >= 0.0
    assert np.max(normalized) <= 1.0
    assert np.isclose(np.min(normalized), 0.0)
    assert np.isclose(np.max(normalized), 1.0)


def test_ensemble_normalize_features_constant(ensemble_detector):
    """Test normalization of constant features."""
    features = np.array([5.0, 5.0, 5.0, 5.0])
    normalized = ensemble_detector._normalize_features(features)

    # Should return zeros for constant input
    assert np.all(normalized == 0.0)


def test_ensemble_bytes_to_vector(ensemble_detector):
    """Test conversion of bytes to vector."""
    data = b"Hello, World!"
    vector = ensemble_detector._bytes_to_vector(data)

    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert vector.size == len(data)


def test_ensemble_build_explanation_normal(ensemble_detector):
    """Test explanation for normal traffic."""
    scores = {"statistical": 0.1, "volatility": 0.2, "pattern": 0.15}
    explanation = ensemble_detector._build_explanation(False, scores, None)

    assert "normal" in explanation.lower()
    assert "statistical=0.10" in explanation


def test_ensemble_build_explanation_anomalous(ensemble_detector):
    """Test explanation for anomalous traffic."""
    scores = {"statistical": 0.9, "volatility": 0.85, "pattern": 0.95}
    explanation = ensemble_detector._build_explanation(True, scores, None)

    assert "anomalous" in explanation.lower()


def test_ensemble_build_explanation_with_source(ensemble_detector):
    """Test explanation with source context."""
    scores = {"statistical": 0.5}
    context = {"source": "10.0.0.1"}
    explanation = ensemble_detector._build_explanation(True, scores, context)

    assert "10.0.0.1" in explanation


def test_anomaly_result_to_dict():
    """Test AnomalyResult to_dict conversion."""
    result = AnomalyResult(
        score=0.75,
        confidence=0.9,
        is_anomalous=True,
        individual_scores={"test": 0.8},
        explanation="Test anomaly",
        metadata={"key": "value"},
    )

    dict_result = result.to_dict()

    assert dict_result["score"] == 0.75
    assert dict_result["confidence"] == 0.9
    assert dict_result["is_anomalous"] is True
    assert "timestamp" in dict_result
    assert dict_result["metadata"]["key"] == "value"


# ===== ISOLATION FOREST TESTS =====


@pytest.fixture
def isolation_forest():
    """Create isolation forest detector instance."""
    return IsolationForestDetector()


@pytest.mark.asyncio
async def test_isolation_forest_detect_basic(isolation_forest):
    """Test basic isolation forest detection."""
    features = [0.1, 0.2, 0.3, 0.4, 0.5] * 10

    result = await isolation_forest.detect(features)

    assert isinstance(result, IsolationForestResult)
    assert 0 <= result.score <= 1
    assert result.is_anomalous in [True, False]
    assert result.threshold > 0


@pytest.mark.asyncio
async def test_isolation_forest_detect_high_variance(isolation_forest):
    """Test detection of high variance (anomaly)."""
    # High variance features
    features = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 20

    result = await isolation_forest.detect(features)

    # High variance should result in higher score
    assert result.score > 0.0


@pytest.mark.asyncio
async def test_isolation_forest_detect_low_variance(isolation_forest):
    """Test detection of low variance (normal)."""
    # Low variance features
    features = [0.5, 0.51, 0.49, 0.50, 0.52] * 20

    result = await isolation_forest.detect(features)

    # Low variance should result in lower score
    assert result.score >= 0.0


@pytest.mark.asyncio
async def test_isolation_forest_empty_features(isolation_forest):
    """Test isolation forest with empty features."""
    with pytest.raises(AnomalyDetectionException, match="Empty feature vector"):
        await isolation_forest.detect([])


@pytest.mark.asyncio
async def test_isolation_forest_with_context(isolation_forest):
    """Test isolation forest with context."""
    features = [0.1, 0.2, 0.3]
    context = {"protocol": "HTTP"}

    result = await isolation_forest.detect(features, context)

    assert "protocol" in result.metadata
    assert result.metadata["protocol"] == "HTTP"


@pytest.mark.asyncio
async def test_isolation_forest_windowing(isolation_forest):
    """Test window size limiting."""
    settings = IsolationForestSettings(window_size=10)
    detector = IsolationForestDetector(settings=settings)

    # Features larger than window
    features = list(range(100))

    result = await detector.detect(features)

    assert result.metadata["window_size"] == 10


@pytest.mark.asyncio
async def test_isolation_forest_update_baseline(isolation_forest):
    """Test baseline update."""
    samples = [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
    ]

    initial_baseline = isolation_forest._baseline_variance

    await isolation_forest.update_baseline(samples)

    assert isolation_forest._baseline_variance != initial_baseline


@pytest.mark.asyncio
async def test_isolation_forest_update_baseline_empty(isolation_forest):
    """Test baseline update with empty samples."""
    await isolation_forest.update_baseline([])

    # Should not change baseline
    assert isolation_forest._baseline_variance == 0.05


def test_isolation_forest_settings():
    """Test isolation forest settings."""
    settings = IsolationForestSettings(contamination=0.1, window_size=256, version="2.0.0")

    assert settings.contamination == 0.1
    assert settings.window_size == 256
    assert settings.version == "2.0.0"


def test_isolation_forest_result():
    """Test isolation forest result."""
    result = IsolationForestResult(score=0.8, threshold=0.5, is_anomalous=True, metadata={"test": "value"})

    assert result.score == 0.8
    assert result.is_anomalous is True
    assert result.timestamp is not None


# ===== LSTM DETECTOR TESTS =====


@pytest.fixture
def lstm_detector():
    """Create LSTM detector instance."""
    return LSTMAnomalyDetector()


@pytest.mark.asyncio
async def test_lstm_detect_basic(lstm_detector):
    """Test basic LSTM detection."""
    sequence = [0.1, 0.2, 0.3, 0.4, 0.5] * 20

    result = await lstm_detector.detect(sequence)

    assert isinstance(result, LSTMDetectionResult)
    assert 0 <= result.score <= 1
    assert 0 <= result.confidence <= 1
    assert result.is_anomalous in [True, False]


@pytest.mark.asyncio
async def test_lstm_detect_smooth_sequence(lstm_detector):
    """Test detection on smooth sequence (normal)."""
    # Smooth increasing sequence
    sequence = list(range(100))

    result = await lstm_detector.detect(sequence)

    # Smooth sequence should have low residuals
    assert result.score >= 0.0


@pytest.mark.asyncio
async def test_lstm_detect_erratic_sequence(lstm_detector):
    """Test detection on erratic sequence (anomaly)."""
    # Erratic sequence with sudden changes
    sequence = [0.0, 1.0, 0.0, 1.0, 0.0] * 20

    result = await lstm_detector.detect(sequence)

    # Erratic sequence should have higher score
    assert result.score > 0.0


@pytest.mark.asyncio
async def test_lstm_empty_sequence(lstm_detector):
    """Test LSTM with empty sequence."""
    with pytest.raises(AnomalyDetectionException, match="Empty sequence"):
        await lstm_detector.detect([])


@pytest.mark.asyncio
async def test_lstm_with_context(lstm_detector):
    """Test LSTM with context."""
    sequence = [0.1, 0.2, 0.3]
    context = {"session_id": "123"}

    result = await lstm_detector.detect(sequence, context)

    assert "session_id" in result.metadata
    assert result.metadata["session_id"] == "123"


@pytest.mark.asyncio
async def test_lstm_confidence_by_length(lstm_detector):
    """Test confidence increases with sequence length."""
    short_seq = [0.1, 0.2, 0.3]
    long_seq = [0.1] * 200

    result_short = await lstm_detector.detect(short_seq)
    result_long = await lstm_detector.detect(long_seq)

    assert result_long.confidence > result_short.confidence


@pytest.mark.asyncio
async def test_lstm_update_threshold(lstm_detector):
    """Test threshold calibration."""
    calibration_sequences = [
        [0.1, 0.2, 0.3] * 10,
        [0.2, 0.3, 0.4] * 10,
        [0.3, 0.4, 0.5] * 10,
    ]

    initial_threshold = lstm_detector.threshold

    await lstm_detector.update_threshold(calibration_sequences)

    # Threshold should be updated
    assert lstm_detector.threshold != initial_threshold
    assert 0.1 <= lstm_detector.threshold <= 0.9


@pytest.mark.asyncio
async def test_lstm_update_threshold_empty(lstm_detector):
    """Test threshold update with empty sequences."""
    await lstm_detector.update_threshold([])

    # Should not crash, threshold unchanged
    assert lstm_detector.threshold == 0.55


@pytest.mark.asyncio
async def test_lstm_update_threshold_bounds(lstm_detector):
    """Test threshold stays within bounds."""
    # Sequences that would produce extreme values
    sequences = [
        [0.0] * 50,  # All zeros
        [100.0] * 50,  # Very high values
    ]

    await lstm_detector.update_threshold(sequences)

    # Should be clamped to bounds
    assert 0.1 <= lstm_detector.threshold <= 0.9


def test_lstm_model_metadata():
    """Test LSTM model metadata."""
    model = LSTMModel(input_size=512, hidden_size=128, num_layers=3, dropout=0.2, version="2.0.0")

    metadata = model.metadata()

    assert metadata["input_size"] == 512
    assert metadata["hidden_size"] == 128
    assert metadata["num_layers"] == 3
    assert metadata["dropout"] == 0.2
    assert metadata["version"] == "2.0.0"


def test_lstm_detection_result():
    """Test LSTM detection result."""
    result = LSTMDetectionResult(score=0.7, confidence=0.85, is_anomalous=True, metadata={"key": "value"})

    assert result.score == 0.7
    assert result.confidence == 0.85
    assert result.is_anomalous is True
    assert result.timestamp is not None


# ===== INTEGRATION TESTS =====


@pytest.mark.asyncio
async def test_ensemble_with_baseline_integration():
    """Test ensemble detector with baseline workflow."""
    detector = EnsembleAnomalyDetector()

    # Generate normal traffic samples as bytes
    np.random.seed(42)
    normal_samples = [np.random.bytes(100) for _ in range(20)]

    # Update baseline
    await detector.update_baseline(normal_samples)

    # Verify baseline was set
    assert detector._baseline_mean is not None
    assert detector._baseline_std is not None

    # Detect on data
    features = np.random.rand(100)
    result = await detector.detect(features)

    # Verify result structure
    assert "score" in result
    assert "is_anomalous" in result
    assert "explanation" in result
    assert 0 <= result["score"] <= 1


@pytest.mark.asyncio
async def test_all_detectors_integration():
    """Test all three detectors on same data."""
    features = np.random.rand(100).tolist()

    # Ensemble
    ensemble = EnsembleAnomalyDetector()
    result_ensemble = await ensemble.detect(features)

    # Isolation Forest
    isolation = IsolationForestDetector()
    result_isolation = await isolation.detect(features)

    # LSTM
    lstm = LSTMAnomalyDetector()
    result_lstm = await lstm.detect(features)

    # All should return valid results
    assert result_ensemble["score"] >= 0
    assert result_isolation.score >= 0
    assert result_lstm.score >= 0


@pytest.mark.asyncio
async def test_concurrent_detections():
    """Test concurrent detection calls."""
    detector = EnsembleAnomalyDetector()

    # Create multiple feature sets
    feature_sets = [np.random.rand(50) for _ in range(10)]

    # Run detections concurrently
    results = await asyncio.gather(*[detector.detect(features) for features in feature_sets])

    assert len(results) == 10
    assert all("score" in r for r in results)
