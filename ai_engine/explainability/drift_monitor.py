"""
QBITEL - Model Drift Monitor

Monitors AI model performance over time and alerts on degradation.
Detects concept drift, data drift, and model performance degradation.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from .database import AIDecisionAudit, ModelDriftMetric
from .metrics import update_drift_metrics, record_drift_alert
from ..core.exceptions import QbitelAIException

logger = logging.getLogger(__name__)


class DriftMonitorException(QbitelAIException):
    """Exception raised by drift monitor."""

    pass


class ModelDriftMonitor:
    """
    Monitor and detect model performance drift.

    Features:
    - Performance metric tracking (accuracy, precision, recall)
    - Distribution shift detection (prediction distribution)
    - Confidence degradation monitoring
    - Automated alerting on drift detection
    - Baseline comparison with statistical tests
    """

    def __init__(
        self,
        async_session: AsyncSession,
        drift_threshold: float = 0.05,  # 5% drift triggers alert
        min_sample_size: int = 100,
        comparison_window_hours: int = 24,
    ):
        """
        Initialize drift monitor.

        Args:
            async_session: SQLAlchemy async session factory
            drift_threshold: Threshold for drift detection (0-1)
            min_sample_size: Minimum samples required for drift calculation
            comparison_window_hours: Hours of data to use for drift comparison
        """
        self.async_session = async_session
        self.drift_threshold = drift_threshold
        self.min_sample_size = min_sample_size
        self.comparison_window_hours = comparison_window_hours

        # Cache for baseline metrics
        self._baseline_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"Initialized ModelDriftMonitor with threshold={drift_threshold}, "
            f"min_samples={min_sample_size}"
        )

    async def check_drift(
        self,
        model_name: str,
        model_version: str,
        force_baseline_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Check for model drift.

        Args:
            model_name: Model to check
            model_version: Model version
            force_baseline_refresh: Force refresh of baseline metrics

        Returns:
            Drift report dictionary
        """
        logger.info(f"Checking drift for {model_name} v{model_version}")

        # Get baseline metrics
        baseline = await self._get_baseline_metrics(
            model_name,
            model_version,
            force_refresh=force_baseline_refresh,
        )

        if not baseline:
            logger.warning(
                f"No baseline metrics for {model_name}, skipping drift check"
            )
            return {
                "drift_detected": False,
                "reason": "no_baseline",
                "message": "Insufficient historical data for drift detection",
            }

        # Get recent metrics
        recent_start = datetime.now(timezone.utc) - timedelta(
            hours=self.comparison_window_hours
        )
        recent_metrics = await self._calculate_recent_metrics(
            model_name,
            model_version,
            recent_start,
        )

        if recent_metrics["sample_size"] < self.min_sample_size:
            logger.warning(
                f"Insufficient recent samples ({recent_metrics['sample_size']}) "
                f"for drift detection"
            )
            return {
                "drift_detected": False,
                "reason": "insufficient_samples",
                "sample_size": recent_metrics["sample_size"],
            }

        # Calculate drift scores
        drift_scores = self._calculate_drift_scores(baseline, recent_metrics)

        # Determine if drift detected
        drift_detected = drift_scores["overall_drift"] > self.drift_threshold

        # Create drift report
        drift_report = {
            "model_name": model_name,
            "model_version": model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drift_detected": drift_detected,
            "drift_score": drift_scores["overall_drift"],
            "drift_threshold": self.drift_threshold,
            "baseline_date": baseline["timestamp"].isoformat(),
            "comparison_window_hours": self.comparison_window_hours,
            "sample_size": recent_metrics["sample_size"],
            "metrics": {
                "accuracy_drift": drift_scores.get("accuracy_drift", 0),
                "confidence_drift": drift_scores.get("confidence_drift", 0),
                "prediction_distribution_drift": drift_scores.get(
                    "distribution_drift", 0
                ),
            },
            "baseline_metrics": {
                "accuracy": baseline.get("accuracy"),
                "avg_confidence": baseline.get("avg_confidence"),
            },
            "recent_metrics": {
                "accuracy": recent_metrics.get("accuracy"),
                "avg_confidence": recent_metrics.get("avg_confidence"),
            },
        }

        # Log drift metric to database
        await self._log_drift_metric(
            model_name=model_name,
            model_version=model_version,
            drift_score=drift_scores["overall_drift"],
            drift_detected=drift_detected,
            drift_details=drift_report,
            baseline_date=baseline["timestamp"],
            sample_size=recent_metrics["sample_size"],
        )

        # Update Prometheus metrics
        update_drift_metrics(
            model_name=model_name,
            model_version=model_version,
            drift_score=drift_scores["overall_drift"],
            accuracy=recent_metrics.get("accuracy", 0),
            avg_confidence=recent_metrics.get("avg_confidence", 0),
        )

        # Trigger alert if drift detected
        if drift_detected:
            await self._trigger_drift_alert(model_name, drift_report)

        return drift_report

    async def _get_baseline_metrics(
        self,
        model_name: str,
        model_version: str,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get baseline metrics for model.

        Uses cached baseline unless force_refresh=True.
        Baseline is calculated from the first 30 days of production data.
        """
        cache_key = f"{model_name}:{model_version}"

        if not force_refresh and cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]

        # Calculate baseline from first 30 days of data
        async with self.async_session() as session:
            # Get earliest timestamp for this model
            result = await session.execute(
                select(AIDecisionAudit)
                .where(
                    and_(
                        AIDecisionAudit.model_name == model_name,
                        AIDecisionAudit.model_version == model_version,
                    )
                )
                .order_by(AIDecisionAudit.timestamp.asc())
                .limit(1)
            )
            earliest_record = result.scalar_one_or_none()

            if not earliest_record:
                return None

            baseline_start = earliest_record.timestamp
            baseline_end = baseline_start + timedelta(days=30)

            # Get baseline decisions
            result = await session.execute(
                select(AIDecisionAudit).where(
                    and_(
                        AIDecisionAudit.model_name == model_name,
                        AIDecisionAudit.model_version == model_version,
                        AIDecisionAudit.timestamp >= baseline_start,
                        AIDecisionAudit.timestamp <= baseline_end,
                    )
                )
            )
            baseline_decisions = result.scalars().all()

            if len(baseline_decisions) < self.min_sample_size:
                return None

            # Calculate baseline metrics
            baseline_metrics = self._calculate_metrics_from_decisions(
                baseline_decisions
            )
            baseline_metrics["timestamp"] = baseline_start

            # Cache baseline
            self._baseline_cache[cache_key] = baseline_metrics

            logger.info(
                f"Calculated baseline for {model_name} v{model_version}: "
                f"{len(baseline_decisions)} samples, accuracy={baseline_metrics.get('accuracy', 0):.3f}"
            )

            return baseline_metrics

    async def _calculate_recent_metrics(
        self,
        model_name: str,
        model_version: str,
        start_time: datetime,
    ) -> Dict[str, Any]:
        """Calculate metrics for recent time window."""
        async with self.async_session() as session:
            result = await session.execute(
                select(AIDecisionAudit).where(
                    and_(
                        AIDecisionAudit.model_name == model_name,
                        AIDecisionAudit.model_version == model_version,
                        AIDecisionAudit.timestamp >= start_time,
                    )
                )
            )
            recent_decisions = result.scalars().all()

            return self._calculate_metrics_from_decisions(recent_decisions)

    def _calculate_metrics_from_decisions(
        self,
        decisions: List[AIDecisionAudit],
    ) -> Dict[str, Any]:
        """Calculate performance metrics from decision records."""
        if not decisions:
            return {"sample_size": 0}

        # Calculate accuracy (using human review as ground truth)
        reviewed_decisions = [d for d in decisions if d.human_reviewed]
        if reviewed_decisions:
            correct_decisions = sum(
                1 for d in reviewed_decisions if not d.human_override
            )
            accuracy = correct_decisions / len(reviewed_decisions)
        else:
            accuracy = None  # No ground truth available

        # Calculate average confidence
        avg_confidence = np.mean([d.confidence_score for d in decisions])

        # Calculate confidence distribution
        confidence_scores = [d.confidence_score for d in decisions]
        confidence_distribution = {
            "mean": float(np.mean(confidence_scores)),
            "std": float(np.std(confidence_scores)),
            "min": float(np.min(confidence_scores)),
            "max": float(np.max(confidence_scores)),
            "q25": float(np.percentile(confidence_scores, 25)),
            "q50": float(np.percentile(confidence_scores, 50)),
            "q75": float(np.percentile(confidence_scores, 75)),
        }

        # Calculate prediction distribution
        prediction_counts = defaultdict(int)
        for d in decisions:
            output = d.decision_output.get("result", "unknown")
            prediction_counts[str(output)] += 1

        prediction_distribution = {
            k: v / len(decisions) for k, v in prediction_counts.items()
        }

        return {
            "sample_size": len(decisions),
            "accuracy": accuracy,
            "avg_confidence": float(avg_confidence),
            "confidence_distribution": confidence_distribution,
            "prediction_distribution": prediction_distribution,
        }

    def _calculate_drift_scores(
        self,
        baseline: Dict[str, Any],
        recent: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Calculate drift scores by comparing baseline and recent metrics.

        Returns:
            Dictionary of drift scores (0-1, higher = more drift)
        """
        drift_scores = {}

        # Accuracy drift (if available)
        if baseline.get("accuracy") is not None and recent.get("accuracy") is not None:
            accuracy_drift = abs(baseline["accuracy"] - recent["accuracy"])
            drift_scores["accuracy_drift"] = accuracy_drift
        else:
            drift_scores["accuracy_drift"] = 0

        # Confidence drift
        confidence_drift = abs(baseline["avg_confidence"] - recent["avg_confidence"])
        drift_scores["confidence_drift"] = confidence_drift

        # Prediction distribution drift (KL divergence)
        baseline_dist = baseline.get("prediction_distribution", {})
        recent_dist = recent.get("prediction_distribution", {})

        if baseline_dist and recent_dist:
            distribution_drift = self._calculate_kl_divergence(
                baseline_dist,
                recent_dist,
            )
            drift_scores["distribution_drift"] = distribution_drift
        else:
            drift_scores["distribution_drift"] = 0

        # Overall drift score (weighted average)
        weights = {
            "accuracy_drift": 0.4,
            "confidence_drift": 0.3,
            "distribution_drift": 0.3,
        }

        overall_drift = sum(drift_scores.get(k, 0) * w for k, w in weights.items())
        drift_scores["overall_drift"] = overall_drift

        return drift_scores

    def _calculate_kl_divergence(
        self,
        p_dist: Dict[str, float],
        q_dist: Dict[str, float],
    ) -> float:
        """
        Calculate KL divergence between two probability distributions.

        KL(P || Q) = sum(P(i) * log(P(i) / Q(i)))
        """
        # Get all classes
        all_classes = set(p_dist.keys()) | set(q_dist.keys())

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10

        kl_div = 0.0
        for cls in all_classes:
            p = p_dist.get(cls, 0) + epsilon
            q = q_dist.get(cls, 0) + epsilon
            kl_div += p * np.log(p / q)

        # Normalize to 0-1 range (cap at 1.0 for large divergences)
        return min(float(kl_div), 1.0)

    async def _log_drift_metric(
        self,
        model_name: str,
        model_version: str,
        drift_score: float,
        drift_detected: bool,
        drift_details: Dict[str, Any],
        baseline_date: datetime,
        sample_size: int,
    ):
        """Log drift metric to database."""
        async with self.async_session() as session:
            drift_metric = ModelDriftMetric(
                id=uuid4(),
                timestamp=datetime.now(timezone.utc),
                model_name=model_name,
                model_version=model_version,
                accuracy=drift_details["recent_metrics"].get("accuracy"),
                precision=None,  # Not calculated yet
                recall=None,  # Not calculated yet
                f1_score=None,  # Not calculated yet
                average_confidence=drift_details["recent_metrics"]["avg_confidence"],
                prediction_distribution=drift_details["recent_metrics"].get(
                    "prediction_distribution"
                ),
                confidence_distribution=drift_details["recent_metrics"].get(
                    "confidence_distribution"
                ),
                drift_score=drift_score,
                drift_detected=drift_detected,
                drift_details=drift_details,
                baseline_date=baseline_date,
                comparison_window_hours=self.comparison_window_hours,
                sample_size=sample_size,
                alert_triggered=drift_detected,
                alert_timestamp=datetime.now(timezone.utc) if drift_detected else None,
            )

            session.add(drift_metric)
            await session.commit()

            logger.info(
                f"Logged drift metric for {model_name} (drift_score={drift_score:.3f})"
            )

    async def _trigger_drift_alert(
        self,
        model_name: str,
        drift_report: Dict[str, Any],
    ):
        """Trigger alert for detected drift."""
        logger.warning(
            f"DRIFT ALERT: {model_name} - drift_score={drift_report['drift_score']:.3f} "
            f"(threshold={self.drift_threshold})"
        )

        # Record Prometheus metric
        record_drift_alert(
            model_name=model_name,
            alert_type="performance_degradation",
        )

        # TODO: Send notification (email, Slack, PagerDuty)
        # This would integrate with existing alerting infrastructure

    # Helper methods for testing
    def _calculate_accuracy(self, decisions: List[Any]) -> float:
        """
        Calculate accuracy from decision objects or (predicted, actual) tuples.
        Helper method for testing and external use.

        Args:
            decisions: List of AIDecisionAudit objects or (predicted, actual) tuples
        """
        if not decisions:
            return 0.0

        # Check if these are AIDecisionAudit objects or tuples
        first_item = decisions[0]
        if isinstance(first_item, tuple):
            # Handle tuples (predicted, actual)
            correct = sum(1 for pred, actual in decisions if pred == actual)
        else:
            # Handle AIDecisionAudit objects
            correct = 0
            for decision in decisions:
                predicted = decision.decision_output.get("result")
                actual = decision.additional_metadata.get("true_label")
                if predicted == actual:
                    correct += 1

        return correct / len(decisions)

    def _calculate_average_confidence(self, confidence_scores: List[float]) -> float:
        """
        Calculate average confidence score.
        Helper method for testing and external use.
        """
        if not confidence_scores:
            return 0.0
        return float(np.mean(confidence_scores))

    def _calculate_distribution(self, predictions: List[Any]) -> Dict[str, float]:
        """
        Calculate prediction distribution.
        Helper method for testing and external use.
        """
        if not predictions:
            return {}
        counts = defaultdict(int)
        for pred in predictions:
            counts[str(pred)] += 1
        total = len(predictions)
        return {k: v / total for k, v in counts.items()}

    def _kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """
        Calculate KL divergence. Alias for _calculate_kl_divergence.
        Helper method for testing and external use.
        """
        return self._calculate_kl_divergence(p, q)

    def _calculate_drift_score(
        self, baseline: Dict[str, Any], recent: Dict[str, Any]
    ) -> float:
        """
        Calculate overall drift score. Alias for _calculate_drift_scores.
        Helper method for testing and external use.
        """
        drift_scores = self._calculate_drift_scores(baseline, recent)
        return drift_scores["overall_drift"]

    def _generate_drift_summary(
        self,
        drift_detected: bool,
        drift_score: float,
        metrics: Dict[str, Any],
    ) -> str:
        """
        Generate human-readable drift summary.
        Helper method for testing and external use.
        """
        if not drift_detected:
            return (
                f"No significant drift detected (score={drift_score:.3f}). "
                f"Model performance is stable."
            )

        return (
            f"DRIFT DETECTED: drift_score={drift_score:.3f} exceeds threshold. "
            f"Accuracy: {metrics.get('accuracy', 0):.3f}, "
            f"Avg Confidence: {metrics.get('avg_confidence', 0):.3f}. "
            f"Model may require retraining."
        )
