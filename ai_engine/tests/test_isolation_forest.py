"""
Tests for ai_engine/anomaly/isolation_forest.py
"""

import pytest
import numpy as np
from unittest.mock import Mock


class TestIsolationForestSettings:
    """Test suite for IsolationForestSettings."""

    def test_settings_defaults(self):
        """Test default settings."""
        from ai_engine.anomaly.isolation_forest import IsolationForestSettings
        
        settings = IsolationForestSettings()
        
        assert settings.contamination == 0.05
        assert settings.window_size == 128
        assert settings.version == "1.0.0"

    def test_settings_custom(self):
        """Test custom settings."""
        from ai_engine.anomaly.isolation_forest import IsolationForestSettings
        
        settings = IsolationForestSettings(
            contamination=0.1,
            window_size=256,
            version="2.0.0"
        )
        
        assert settings.contamination == 0.1
        assert settings.window_size == 256
        assert settings.version == "2.0.0"


class TestIsolationForestResult:
    """Test suite for IsolationForestResult."""

    def test_result_creation(self):
        """Test result creation."""
        from ai_engine.anomaly.isolation_forest import IsolationForestResult
        
        result = IsolationForestResult(
            score=0.7,
            threshold=0.5,
            is_anomalous=True
        )
        
        assert result.score == 0.7
        assert result.threshold == 0.5
        assert result.is_anomalous is True
        assert result.timestamp is not None


class TestIsolationForestDetector:
    """Test suite for IsolationForestDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        from ai_engine.anomaly.isolation_forest import IsolationForestDetector
        return IsolationForestDetector()

    @pytest.fixture
    def detector_with_settings(self):
        """Create detector with custom settings."""
        from ai_engine.anomaly.isolation_forest import (
            IsolationForestDetector,
            IsolationForestSettings
        )
        settings = IsolationForestSettings(contamination=0.1, window_size=64)
        return IsolationForestDetector(settings=settings)

    @pytest.mark.asyncio
    async def test_detect_normal(self, detector):
        """Test detection with normal features."""
        features = [0.1, 0.2, 0.15, 0.18, 0.12] * 20
        result = await detector.detect(features)
        
        assert isinstance(result.score, float)
        assert isinstance(result.is_anomalous, bool)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_detect_anomalous(self, detector):
        """Test detection with anomalous features."""
        # High variance features
        features = list(range(100))
        result = await detector.detect(features)
        
        assert result.score >= 0.0
        assert "variance" in result.metadata

    @pytest.mark.asyncio
    async def test_detect_empty_features(self, detector):
        """Test detection with empty features."""
        from ai_engine.core.exceptions import AnomalyDetectionException
        
        with pytest.raises(AnomalyDetectionException, match="Empty feature vector"):
            await detector.detect([])

    @pytest.mark.asyncio
    async def test_detect_with_context(self, detector):
        """Test detection with context."""
        features = [0.5] * 50
        context = {"source": "test"}
        
        result = await detector.detect(features, context=context)
        
        assert "source" in result.metadata
        assert result.metadata["source"] == "test"

    @pytest.mark.asyncio
    async def test_detect_small_window(self, detector):
        """Test detection with features smaller than window size."""
        features = [0.1, 0.2, 0.3]
        result = await detector.detect(features)
        
        assert result.metadata["window_size"] == 3

    @pytest.mark.asyncio
    async def test_detect_large_window(self, detector):
        """Test detection with features larger than window size."""
        features = list(np.random.randn(200))
        result = await detector.detect(features)
        
        assert result.metadata["window_size"] == detector.settings.window_size

    @pytest.mark.asyncio
    async def test_update_baseline(self, detector):
        """Test baseline update."""
        samples = [
            [0.1, 0.2, 0.15],
            [0.12, 0.18, 0.14],
            [0.11, 0.19, 0.16]
        ]
        
        await detector.update_baseline(samples)
        
        assert detector._baseline_variance > 0

    @pytest.mark.asyncio
    async def test_update_baseline_empty_samples(self, detector):
        """Test baseline update with empty samples."""
        await detector.update_baseline([])
        
        # Should not crash, baseline remains default
        assert detector._baseline_variance == 0.05

    @pytest.mark.asyncio
    async def test_update_baseline_with_empty_vectors(self, detector):
        """Test baseline update with some empty vectors."""
        samples = [
            [0.1, 0.2],
            [],
            [0.3, 0.4]
        ]
        
        await detector.update_baseline(samples)
        
        assert detector._baseline_variance > 0

    @pytest.mark.asyncio
    async def test_threshold_calculation(self, detector):
        """Test threshold calculation."""
        features = [0.5] * 100
        result = await detector.detect(features)
        
        assert result.threshold > 0
        assert result.threshold >= detector._baseline_variance

    @pytest.mark.asyncio
    async def test_score_normalization(self, detector):
        """Test that score is normalized between 0 and 1."""
        features = list(np.random.randn(100) * 10)
        result = await detector.detect(features)
        
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_metadata_includes_settings(self, detector):
        """Test that metadata includes settings."""
        features = [0.1] * 50
        result = await detector.detect(features)
        
        assert "settings" in result.metadata
        assert result.metadata["settings"] == detector.settings

    @pytest.mark.asyncio
    async def test_custom_contamination(self, detector_with_settings):
        """Test detector with custom contamination."""
        features = [0.1] * 50
        result = await detector_with_settings.detect(features)
        
        assert detector_with_settings.settings.contamination == 0.1

    @pytest.mark.asyncio
    async def test_custom_window_size(self, detector_with_settings):
        """Test detector with custom window size."""
        features = list(range(100))
        result = await detector_with_settings.detect(features)
        
        assert result.metadata["window_size"] == 64

    @pytest.mark.asyncio
    async def test_detect_after_baseline_update(self, detector):
        """Test detection after baseline update."""
        # Update baseline with normal data
        samples = [[0.1 + i * 0.01 for i in range(10)] for _ in range(5)]
        await detector.update_baseline(samples)
        
        # Detect with similar data
        features = [0.15 + i * 0.01 for i in range(10)]
        result = await detector.detect(features)
        
        assert result.score >= 0.0