"""CRONOS AI Engine - LSTM Anomaly Detector (stub).

This lightweight implementation captures temporal dynamics using a simple
statistical surrogate so tests have a deterministic interface to exercise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional
from datetime import datetime, timezone

import numpy as np

from ..core.config import Config
from ..core.exceptions import AnomalyDetectionException


@dataclass
class LSTMModel:
    """Configuration snapshot for the detector."""

    input_size: int = 256
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    version: str = "1.0.0"

    def metadata(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "version": self.version,
        }


@dataclass
class LSTMDetectionResult:
    """Temporal anomaly detection outcome."""

    score: float
    confidence: float
    is_anomalous: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class LSTMAnomalyDetector:
    """Simplified LSTM-inspired anomaly detector."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model = LSTMModel()
        self.threshold = getattr(self.config, "lstm_anomaly_threshold", 0.55)

    async def detect(
        self,
        sequence: Iterable[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> LSTMDetectionResult:
        """Detect anomalies in a sequence-like input."""
        vector = np.asarray(list(sequence), dtype=np.float32)
        if vector.size == 0:
            raise AnomalyDetectionException("Empty sequence passed to LSTM detector")

        rolling_mean = np.convolve(vector, np.ones(5) / 5.0, mode="valid")
        residuals = np.abs(vector[-len(rolling_mean) :] - rolling_mean)
        score = float(np.clip(np.mean(residuals), 0.0, 1.0))

        confidence = float(np.clip(1 - np.exp(-vector.size / 64.0), 0.0, 1.0))
        is_anomalous = score > self.threshold

        metadata = self.model.metadata()
        if context:
            metadata.update(context)

        return LSTMDetectionResult(
            score=score,
            confidence=confidence,
            is_anomalous=is_anomalous,
            metadata=metadata,
        )

    async def update_threshold(
        self, calibration_sequences: Iterable[Iterable[float]]
    ) -> None:
        """Calibrate anomaly threshold using sample sequences."""
        scores = []
        for sequence in calibration_sequences:
            vector = np.asarray(list(sequence), dtype=np.float32)
            if vector.size == 0:
                continue
            rolling_mean = np.convolve(vector, np.ones(5) / 5.0, mode="valid")
            residuals = np.abs(vector[-len(rolling_mean) :] - rolling_mean)
            scores.append(np.mean(residuals))

        if scores:
            baseline = float(np.mean(scores))
            self.threshold = min(max(baseline * 1.5, 0.1), 0.9)


__all__ = ["LSTMAnomalyDetector", "LSTMModel", "LSTMDetectionResult"]
