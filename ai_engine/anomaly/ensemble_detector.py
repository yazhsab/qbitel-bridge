"""CRONOS AI Engine - Ensemble Anomaly Detector

Lightweight ensemble detector that combines statistical heuristics with
pluggable anomaly detection backends. The implementation intentionally
keeps dependencies minimal while exposing a stable interface used by the
rest of the platform and the accompanying test-suite.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

from ..core.config import Config
from ..core.exceptions import AnomalyDetectionException


@dataclass
class AnomalyResult:
    """Aggregate anomaly detection result produced by the ensemble."""

    score: float
    confidence: float
    is_anomalous: bool
    individual_scores: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation compatible with engine contract."""
        return {
            "score": self.score,
            "confidence": self.confidence,
            "is_anomalous": self.is_anomalous,
            "individual_scores": self.individual_scores,
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class EnsembleAnomalyDetector:
    """Heuristic ensemble anomaly detector used in tests and demos."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = getattr(self.config, "logger", None)
        self.threshold = getattr(self.config, "anomaly_threshold", 0.5)
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None

    async def detect(
        self,
        features: Union[np.ndarray, Iterable[float], Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Detect anomalies and return a dictionary for engine consumption."""
        result = await self._detect_internal(features, context)
        return result.to_dict()

    async def detect_batch(
        self,
        batch_features: Iterable[Union[np.ndarray, Iterable[float], Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Detect anomalies for a batch of feature vectors."""
        tasks = [self._detect_internal(features, context) for features in batch_features]
        results = await asyncio.gather(*tasks)
        return [result.to_dict() for result in results]

    async def update_baseline(self, samples: Iterable[bytes]) -> None:
        """Update baseline statistics used by heuristic scoring."""
        vectors = [self._bytes_to_vector(sample) for sample in samples if sample]
        if not vectors:
            raise AnomalyDetectionException("No samples supplied for baseline update")

        stacked = np.concatenate(vectors)
        self._baseline_mean = float(np.mean(stacked))
        self._baseline_std = float(np.std(stacked) + 1e-6)

    async def _detect_internal(
        self,
        features: Union[np.ndarray, Iterable[float], Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> AnomalyResult:
        """Internal coroutine returning an ``AnomalyResult`` instance."""
        vector = self._normalize_features(features)
        if vector.size == 0:
            raise AnomalyDetectionException("Empty feature vector supplied")

        statistical_score = self._statistical_deviation(vector)
        volatility_score = self._volatility_score(vector)
        pattern_score = self._pattern_change_score(vector)

        individual_scores = {
            "statistical": statistical_score,
            "volatility": volatility_score,
            "pattern": pattern_score,
        }

        score = float(np.clip(np.mean(list(individual_scores.values())), 0.0, 1.0))
        confidence = self._confidence(vector)
        is_anomalous = score > self.threshold

        explanation = self._build_explanation(is_anomalous, individual_scores, context)

        metadata = {}
        if context:
            metadata.update(context)

        return AnomalyResult(
            score=score,
            confidence=confidence,
            is_anomalous=is_anomalous,
            individual_scores=individual_scores,
            explanation=explanation,
            metadata=metadata,
        )

    def _normalize_features(
        self, features: Union[np.ndarray, Iterable[float], Dict[str, Any]]
    ) -> np.ndarray:
        """Convert supported feature input formats into a Numpy vector."""
        if isinstance(features, dict):
            if "features" in features:
                arr = np.asarray(features["features"], dtype=np.float32)
            elif "vector" in features:
                arr = np.asarray(features["vector"], dtype=np.float32)
            else:
                arr = np.asarray(list(features.values()), dtype=np.float32)
        elif isinstance(features, np.ndarray):
            arr = features.astype(np.float32).flatten()
        else:
            arr = np.asarray(list(features), dtype=np.float32)

        if arr.size == 0:
            return arr

        # Scale between 0 and 1 for stability
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if math.isclose(min_val, max_val):
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val + 1e-9)

    def _bytes_to_vector(self, sample: bytes) -> np.ndarray:
        """Convert raw bytes into a numerical vector."""
        return np.frombuffer(sample, dtype=np.uint8).astype(np.float32)

    def _statistical_deviation(self, vector: np.ndarray) -> float:
        """Compute deviation from the established baseline."""
        if self._baseline_mean is None or self._baseline_std is None:
            baseline_mean = 0.5
            baseline_std = 0.25
        else:
            baseline_mean = self._baseline_mean
            baseline_std = self._baseline_std

        current_mean = float(np.mean(vector))
        z_score = abs(current_mean - baseline_mean) / (baseline_std + 1e-6)
        return float(np.clip(z_score / 10.0, 0.0, 1.0))

    def _volatility_score(self, vector: np.ndarray) -> float:
        """Estimate volatility using rolling differences."""
        if vector.size < 2:
            return 0.0
        diffs = np.abs(np.diff(vector))
        volatility = float(np.mean(diffs))
        return float(np.clip(volatility, 0.0, 1.0))

    def _pattern_change_score(self, vector: np.ndarray) -> float:
        """Detect abrupt pattern changes using FFT energy distribution."""
        if vector.size < 4:
            return 0.0
        spectrum = np.fft.rfft(vector - np.mean(vector))
        energy = np.abs(spectrum) ** 2
        high_freq_energy = float(np.sum(energy[int(len(energy) / 3) :]))
        total_energy = float(np.sum(energy) + 1e-9)
        ratio = high_freq_energy / total_energy
        return float(np.clip(ratio, 0.0, 1.0))

    def _confidence(self, vector: np.ndarray) -> float:
        """Derive a confidence score based on vector density."""
        confidence = 1 - math.exp(-vector.size / 128.0)
        return float(np.clip(confidence, 0.0, 1.0))

    def _build_explanation(
        self,
        is_anomalous: bool,
        scores: Dict[str, float],
        context: Optional[Dict[str, Any]],
    ) -> str:
        status = "anomalous" if is_anomalous else "normal"
        score_parts = ", ".join(f"{name}={value:.2f}" for name, value in scores.items())
        if context and "source" in context:
            return f"Traffic from {context['source']} classified as {status} ({score_parts})"
        return f"Sample classified as {status} ({score_parts})"


__all__ = ["EnsembleAnomalyDetector", "AnomalyResult"]
