"""
Tests for Model Drift Monitor.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, Mock

from ai_engine.explainability.drift_monitor import (
    ModelDriftMonitor,
    DriftMonitorException,
)
from ai_engine.explainability.database import AIDecisionAudit, ModelDriftMetric


class TestModelDriftMonitor:
    """Tests for ModelDriftMonitor."""

    @pytest.fixture
    def mock_session(self):
        """Create mock async session for testing."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def drift_monitor(self, mock_session):
        """Create drift monitor for testing."""
        monitor = ModelDriftMonitor(
            async_session=mock_session,
            drift_threshold=0.05,
            min_sample_size=100,
            comparison_window_hours=24,
        )
        return monitor

    def test_initialization(self, drift_monitor):
        """Test drift monitor initialization."""
        assert drift_monitor.drift_threshold == 0.05
        assert drift_monitor.min_sample_size == 100
        assert drift_monitor.comparison_window_hours == 24

    def test_custom_thresholds(self):
        """Test drift monitor with custom thresholds."""
        session = AsyncMock()
        monitor = ModelDriftMonitor(
            async_session=session,
            drift_threshold=0.10,
            min_sample_size=50,
            comparison_window_hours=48,
        )

        assert monitor.drift_threshold == 0.10
        assert monitor.min_sample_size == 50
        assert monitor.comparison_window_hours == 48

    @pytest.mark.asyncio
    async def test_calculate_accuracy(self, drift_monitor):
        """Test accuracy calculation from decisions."""
        # Mock decisions with predictions and ground truth
        decisions = [
            MagicMock(
                decision_output={"result": "HTTP"},
                additional_metadata={"true_label": "HTTP"},
            ),
            MagicMock(
                decision_output={"result": "TLS"},
                additional_metadata={"true_label": "TLS"},
            ),
            MagicMock(
                decision_output={"result": "HTTP"},
                additional_metadata={"true_label": "TLS"},  # Wrong
            ),
            MagicMock(
                decision_output={"result": "SSH"},
                additional_metadata={"true_label": "HTTP"},  # Wrong
            ),
        ]

        # 2 out of 4 correct = 50% accuracy
        accuracy = drift_monitor._calculate_accuracy(decisions)
        assert accuracy == pytest.approx(0.5, rel=0.01)

    @pytest.mark.asyncio
    async def test_calculate_accuracy_all_correct(self, drift_monitor):
        """Test accuracy calculation with 100% correct predictions."""
        decisions = [
            MagicMock(
                decision_output={"result": "HTTP"},
                additional_metadata={"true_label": "HTTP"},
            ),
            MagicMock(
                decision_output={"result": "TLS"},
                additional_metadata={"true_label": "TLS"},
            ),
        ]

        accuracy = drift_monitor._calculate_accuracy(decisions)
        assert accuracy == 1.0

    @pytest.mark.asyncio
    async def test_calculate_accuracy_empty(self, drift_monitor):
        """Test accuracy calculation with empty decision list."""
        decisions = []
        accuracy = drift_monitor._calculate_accuracy(decisions)
        assert accuracy == 0.0

    @pytest.mark.asyncio
    async def test_calculate_average_confidence(self, drift_monitor):
        """Test average confidence calculation."""
        decisions = [
            MagicMock(confidence_score=0.95),
            MagicMock(confidence_score=0.90),
            MagicMock(confidence_score=0.85),
            MagicMock(confidence_score=0.80),
        ]

        avg_conf = drift_monitor._calculate_average_confidence(decisions)
        assert avg_conf == pytest.approx(0.875, rel=0.01)

    @pytest.mark.asyncio
    async def test_calculate_distribution(self, drift_monitor):
        """Test prediction distribution calculation."""
        decisions = [
            MagicMock(decision_output={"result": "HTTP"}),
            MagicMock(decision_output={"result": "HTTP"}),
            MagicMock(decision_output={"result": "TLS"}),
            MagicMock(decision_output={"result": "SSH"}),
        ]

        distribution = drift_monitor._calculate_distribution(decisions)

        assert distribution["HTTP"] == pytest.approx(0.5, rel=0.01)  # 2/4
        assert distribution["TLS"] == pytest.approx(0.25, rel=0.01)  # 1/4
        assert distribution["SSH"] == pytest.approx(0.25, rel=0.01)  # 1/4

    @pytest.mark.asyncio
    async def test_kl_divergence_identical(self, drift_monitor):
        """Test KL divergence with identical distributions."""
        p = {"HTTP": 0.5, "TLS": 0.5}
        q = {"HTTP": 0.5, "TLS": 0.5}

        kl_div = drift_monitor._kl_divergence(p, q)
        assert kl_div == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_kl_divergence_different(self, drift_monitor):
        """Test KL divergence with different distributions."""
        p = {"HTTP": 0.5, "TLS": 0.5}
        q = {"HTTP": 0.7, "TLS": 0.3}

        kl_div = drift_monitor._kl_divergence(p, q)
        assert kl_div > 0  # Should be positive
        assert np.isfinite(kl_div)  # Should not be infinite

    @pytest.mark.asyncio
    async def test_kl_divergence_new_category(self, drift_monitor):
        """Test KL divergence when new category appears."""
        p = {"HTTP": 0.5, "TLS": 0.5}
        q = {"HTTP": 0.4, "TLS": 0.4, "SSH": 0.2}  # New category

        kl_div = drift_monitor._kl_divergence(p, q)
        assert np.isfinite(kl_div)

    @pytest.mark.asyncio
    async def test_calculate_drift_score_no_drift(self, drift_monitor):
        """Test drift score calculation when no drift."""
        baseline_accuracy = 0.95
        recent_accuracy = 0.94
        baseline_conf = 0.90
        recent_conf = 0.89
        baseline_dist = {"HTTP": 0.5, "TLS": 0.5}
        recent_dist = {"HTTP": 0.51, "TLS": 0.49}

        drift_score = drift_monitor._calculate_drift_score(
            baseline_accuracy=baseline_accuracy,
            recent_accuracy=recent_accuracy,
            baseline_confidence=baseline_conf,
            recent_confidence=recent_conf,
            baseline_distribution=baseline_dist,
            recent_distribution=recent_dist,
        )

        # Should be small drift
        assert drift_score < 0.05  # Below threshold

    @pytest.mark.asyncio
    async def test_calculate_drift_score_high_drift(self, drift_monitor):
        """Test drift score calculation with high drift."""
        baseline_accuracy = 0.95
        recent_accuracy = 0.75  # 20% drop
        baseline_conf = 0.90
        recent_conf = 0.70  # 20% drop
        baseline_dist = {"HTTP": 0.5, "TLS": 0.5}
        recent_dist = {"HTTP": 0.9, "TLS": 0.1}  # Major shift

        drift_score = drift_monitor._calculate_drift_score(
            baseline_accuracy=baseline_accuracy,
            recent_accuracy=recent_accuracy,
            baseline_confidence=baseline_conf,
            recent_confidence=recent_conf,
            baseline_distribution=baseline_dist,
            recent_distribution=recent_dist,
        )

        # Should be high drift
        assert drift_score > 0.05  # Above threshold

    @pytest.mark.asyncio
    async def test_check_drift_insufficient_samples(self, drift_monitor):
        """Test drift check with insufficient samples."""
        # Mock query to return too few baseline decisions
        mock_execute = AsyncMock()
        mock_execute.scalars.return_value.all.return_value = [
            MagicMock() for _ in range(50)  # Less than min_sample_size (100)
        ]
        drift_monitor.async_session.execute = AsyncMock(return_value=mock_execute)

        with pytest.raises(DriftMonitorException, match="Insufficient baseline data"):
            await drift_monitor.check_drift(
                model_name="test_model",
                model_version="1.0.0",
            )

    @pytest.mark.asyncio
    async def test_generate_drift_summary_no_drift(self, drift_monitor):
        """Test drift summary when no drift detected."""
        summary = drift_monitor._generate_drift_summary(
            model_name="test_model",
            model_version="1.0.0",
            baseline_accuracy=0.95,
            recent_accuracy=0.94,
            baseline_confidence=0.90,
            recent_confidence=0.89,
            drift_score=0.02,
            drift_detected=False,
        )

        assert "No significant drift" in summary or "No drift detected" in summary
        assert "test_model" in summary
        assert "2" in summary or "0.02" in summary  # Drift score

    @pytest.mark.asyncio
    async def test_generate_drift_summary_with_drift(self, drift_monitor):
        """Test drift summary when drift detected."""
        summary = drift_monitor._generate_drift_summary(
            model_name="test_model",
            model_version="1.0.0",
            baseline_accuracy=0.95,
            recent_accuracy=0.75,
            baseline_confidence=0.90,
            recent_confidence=0.70,
            drift_score=0.15,
            drift_detected=True,
        )

        assert "DRIFT DETECTED" in summary or "drift detected" in summary.lower()
        assert "test_model" in summary
        assert "15" in summary or "0.15" in summary  # Drift score

    def test_drift_threshold_validation(self):
        """Test that drift threshold is properly set."""
        session = AsyncMock()

        # Valid thresholds
        monitor1 = ModelDriftMonitor(session, drift_threshold=0.01)
        assert monitor1.drift_threshold == 0.01

        monitor2 = ModelDriftMonitor(session, drift_threshold=0.20)
        assert monitor2.drift_threshold == 0.20

    @pytest.mark.asyncio
    async def test_min_sample_size_enforced(self, drift_monitor):
        """Test that minimum sample size is enforced."""
        # This should be tested in the check_drift method
        assert drift_monitor.min_sample_size == 100

    def test_comparison_window_hours(self, drift_monitor):
        """Test comparison window configuration."""
        assert drift_monitor.comparison_window_hours == 24

    @pytest.mark.asyncio
    async def test_confidence_distribution_consistency(self, drift_monitor):
        """Test that confidence values are in valid range."""
        decisions = [
            MagicMock(confidence_score=0.95),
            MagicMock(confidence_score=0.50),
            MagicMock(confidence_score=1.00),
            MagicMock(confidence_score=0.00),
        ]

        avg_conf = drift_monitor._calculate_average_confidence(decisions)

        # Average should be in valid range
        assert 0.0 <= avg_conf <= 1.0

    @pytest.mark.asyncio
    async def test_prediction_distribution_sums_to_one(self, drift_monitor):
        """Test that prediction distribution probabilities sum to 1."""
        decisions = [
            MagicMock(decision_output={"result": "HTTP"}),
            MagicMock(decision_output={"result": "HTTP"}),
            MagicMock(decision_output={"result": "TLS"}),
            MagicMock(decision_output={"result": "SSH"}),
        ]

        distribution = drift_monitor._calculate_distribution(decisions)

        # Sum of probabilities should be 1.0
        total = sum(distribution.values())
        assert total == pytest.approx(1.0, rel=0.01)
