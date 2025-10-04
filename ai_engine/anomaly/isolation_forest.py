"""CRONOS AI Engine - Isolation Forest Detector (heuristic stub)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import numpy as np

from ..core.config import Config
from ..core.exceptions import AnomalyDetectionException


@dataclass
class IsolationForestSettings:
    """Configuration for the heuristic isolation forest detector."""

    contamination: float = 0.05
    window_size: int = 128
    version: str = "1.0.0"


@dataclass
class IsolationForestResult:
    """Detection result produced by the heuristic isolation forest."""

    score: float
    threshold: float
    is_anomalous: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class IsolationForestDetector:
    """Variance-based isolation heuristic approximating an isolation forest."""

    def __init__(self, config: Optional[Config] = None, *, settings: Optional[IsolationForestSettings] = None):
        self.config = config or Config()
        self.settings = settings or IsolationForestSettings()
        self._baseline_variance: float = 0.05

    async def detect(
        self,
        features: Iterable[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> IsolationForestResult:
        vector = np.asarray(list(features), dtype=np.float32)
        if vector.size == 0:
            raise AnomalyDetectionException("Empty feature vector supplied to isolation forest")

        window = min(self.settings.window_size, vector.size)
        sliding = vector[-window:]
        variance = float(np.var(sliding))

        threshold = max(self._baseline_variance * (1 + self.settings.contamination * 5), 1e-3)
        is_anomalous = variance > threshold
        score = float(np.clip(variance / (threshold + 1e-9), 0.0, 2.0))
        score = min(score / 2.0, 1.0)

        metadata = {
            "variance": variance,
            "window_size": window,
            "baseline_variance": self._baseline_variance,
            "settings": self.settings,
        }
        if context:
            metadata.update(context)

        return IsolationForestResult(
            score=score,
            threshold=threshold,
            is_anomalous=is_anomalous,
            metadata=metadata,
        )

    async def update_baseline(self, samples: Iterable[Iterable[float]]) -> None:
        variances = []
        for sample in samples:
            vector = np.asarray(list(sample), dtype=np.float32)
            if vector.size:
                variances.append(np.var(vector))
        if variances:
            self._baseline_variance = float(np.mean(variances))


__all__ = ["IsolationForestDetector", "IsolationForestResult", "IsolationForestSettings"]
