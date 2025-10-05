"""
Tests for ai_engine/anomaly/lstm_detector.py
"""

import pytest
import numpy as np
from unittest.mock import Mock


class TestLSTMModel:
    """Test suite for LSTMModel dataclass."""

    def test_model_defaults(self):
        """Test default model configuration."""
        from ai_engine.anomaly.lstm_detector import LSTMModel

        model = LSTMModel()

        assert model.input_size == 256
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.dropout == 0.1
        assert model.version == "1.0.0"

    def test_model_custom(self):
        """Test custom model configuration."""
        from ai_engine.anomaly.lstm_detector import LSTMModel

        model = LSTMModel(
            input_size=512, hidden_size=128, num_layers=3, dropout=0.2, version="2.0.0"
        )

        assert model.input_size == 512
        assert model.hidden_size == 128
        assert model.num_layers == 3

    def test_model_metadata(self):
        """Test model metadata generation."""
        from ai_engine.anomaly.lstm_detector import LSTMModel

        model = LSTMModel()
        metadata = model.metadata()

        assert isinstance(metadata, dict)
        assert "input_size" in metadata
        assert "hidden_size" in metadata
        assert "num_layers" in metadata
        assert "dropout" in metadata
        assert "version" in metadata


class TestLSTMDetectionResult:
    """Test suite for LSTMDetectionResult."""

    def test_result_creation(self):
        """Test result creation."""
        from ai_engine.anomaly.lstm_detector import LSTMDetectionResult

        result = LSTMDetectionResult(score=0.6, confidence=0.8, is_anomalous=True)

        assert result.score == 0.6
        assert result.confidence == 0.8
        assert result.is_anomalous is True
        assert result.timestamp is not None


class TestLSTMAnomalyDetector:
    """Test suite for LSTMAnomalyDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        from ai_engine.anomaly.lstm_detector import LSTMAnomalyDetector

        return LSTMAnomalyDetector()

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.lstm_anomaly_threshold = 0.6
        return config

    @pytest.fixture
    def detector_with_config(self, mock_config):
        """Create detector with config."""
        from ai_engine.anomaly.lstm_detector import LSTMAnomalyDetector

        return LSTMAnomalyDetector(mock_config)

    @pytest.mark.asyncio
    async def test_detect_normal_sequence(self, detector):
        """Test detection with normal sequence."""
        sequence = [0.1, 0.2, 0.15, 0.18, 0.12, 0.16] * 10
        result = await detector.detect(sequence)

        assert isinstance(result.score, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.is_anomalous, bool)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_detect_anomalous_sequence(self, detector):
        """Test detection with anomalous sequence."""
        # Create sequence with sudden spike
        sequence = [0.1] * 50 + [10.0] * 10 + [0.1] * 50
        result = await detector.detect(sequence)

        assert result.score >= 0.0

    @pytest.mark.asyncio
    async def test_detect_empty_sequence(self, detector):
        """Test detection with empty sequence."""
        from ai_engine.core.exceptions import AnomalyDetectionException

        with pytest.raises(AnomalyDetectionException, match="Empty sequence"):
            await detector.detect([])

    @pytest.mark.asyncio
    async def test_detect_with_context(self, detector):
        """Test detection with context."""
        sequence = [0.5] * 50
        context = {"protocol": "HTTP"}

        result = await detector.detect(sequence, context=context)

        assert "protocol" in result.metadata
        assert result.metadata["protocol"] == "HTTP"

    @pytest.mark.asyncio
    async def test_detect_short_sequence(self, detector):
        """Test detection with short sequence."""
        sequence = [0.1, 0.2, 0.3]
        result = await detector.detect(sequence)

        assert result.score >= 0.0
        assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_detect_long_sequence(self, detector):
        """Test detection with long sequence."""
        sequence = list(np.random.randn(200))
        result = await detector.detect(sequence)

        assert result.score >= 0.0
        # Longer sequences should have higher confidence
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_confidence_increases_with_length(self, detector):
        """Test that confidence increases with sequence length."""
        short_seq = [0.1] * 10
        long_seq = [0.1] * 200

        short_result = await detector.detect(short_seq)
        long_result = await detector.detect(long_seq)

        assert long_result.confidence > short_result.confidence

    @pytest.mark.asyncio
    async def test_update_threshold(self, detector):
        """Test threshold update."""
        calibration_sequences = [
            [0.1 + i * 0.01 for i in range(20)],
            [0.2 + i * 0.01 for i in range(20)],
            [0.15 + i * 0.01 for i in range(20)],
        ]

        original_threshold = detector.threshold
        await detector.update_threshold(calibration_sequences)

        # Threshold should be updated
        assert detector.threshold != original_threshold
        assert 0.1 <= detector.threshold <= 0.9

    @pytest.mark.asyncio
    async def test_update_threshold_empty_sequences(self, detector):
        """Test threshold update with empty sequences."""
        original_threshold = detector.threshold
        await detector.update_threshold([])

        # Threshold should remain unchanged
        assert detector.threshold == original_threshold

    @pytest.mark.asyncio
    async def test_update_threshold_with_empty_vectors(self, detector):
        """Test threshold update with some empty vectors."""
        calibration_sequences = [[0.1, 0.2, 0.3], [], [0.4, 0.5, 0.6]]

        await detector.update_threshold(calibration_sequences)

        # Should handle empty vectors gracefully
        assert 0.1 <= detector.threshold <= 0.9

    @pytest.mark.asyncio
    async def test_threshold_bounds(self, detector):
        """Test that threshold stays within bounds."""
        # Very low variance sequences
        low_var_sequences = [[0.1] * 20 for _ in range(5)]
        await detector.update_threshold(low_var_sequences)
        assert detector.threshold >= 0.1

        # Very high variance sequences
        high_var_sequences = [list(range(100)) for _ in range(5)]
        await detector.update_threshold(high_var_sequences)
        assert detector.threshold <= 0.9

    @pytest.mark.asyncio
    async def test_metadata_includes_model_info(self, detector):
        """Test that metadata includes model information."""
        sequence = [0.1] * 50
        result = await detector.detect(sequence)

        assert "input_size" in result.metadata
        assert "hidden_size" in result.metadata
        assert "num_layers" in result.metadata
        assert "dropout" in result.metadata
        assert "version" in result.metadata

    @pytest.mark.asyncio
    async def test_custom_threshold_from_config(self, detector_with_config):
        """Test detector with custom threshold from config."""
        assert detector_with_config.threshold == 0.6

    @pytest.mark.asyncio
    async def test_anomaly_detection_threshold(self, detector):
        """Test anomaly detection based on threshold."""
        # Set a low threshold
        detector.threshold = 0.3

        # High residual sequence
        sequence = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 20
        result = await detector.detect(sequence)

        # Should be detected as anomalous if score > threshold
        if result.score > detector.threshold:
            assert result.is_anomalous is True

    @pytest.mark.asyncio
    async def test_rolling_mean_calculation(self, detector):
        """Test that rolling mean is calculated correctly."""
        # Constant sequence should have low residuals
        sequence = [0.5] * 100
        result = await detector.detect(sequence)

        # Score should be low for constant sequence
        assert result.score < 0.5

    @pytest.mark.asyncio
    async def test_detect_after_threshold_update(self, detector):
        """Test detection after threshold calibration."""
        # Calibrate with normal sequences
        calibration = [[0.1 + i * 0.01 for i in range(20)] for _ in range(5)]
        await detector.update_threshold(calibration)

        # Detect with similar sequence
        sequence = [0.12 + i * 0.01 for i in range(20)]
        result = await detector.detect(sequence)

        assert result.score >= 0.0
        assert result.is_anomalous is not None

    @pytest.mark.asyncio
    async def test_numpy_array_input(self, detector):
        """Test detection with numpy array input."""
        sequence = np.random.randn(50)
        result = await detector.detect(sequence)

        assert result.score >= 0.0

    @pytest.mark.asyncio
    async def test_model_version_in_metadata(self, detector):
        """Test that model version is in metadata."""
        sequence = [0.1] * 20
        result = await detector.detect(sequence)

        assert result.metadata["version"] == "1.0.0"
