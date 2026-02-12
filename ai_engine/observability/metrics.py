"""
Metrics Collection

Prometheus-compatible metrics collection for:
- Request/response metrics
- Cryptographic operation metrics
- Key/certificate lifecycle metrics
- System health metrics
- Business metrics (transactions, volumes)

Supports:
- Counters (monotonically increasing)
- Gauges (can go up or down)
- Histograms (value distributions)
- Summaries (quantile calculations)
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import re

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricLabels:
    """Labels for metric identification."""

    labels: Dict[str, str] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.labels.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MetricLabels):
            return False
        return self.labels == other.labels

    def to_prometheus(self) -> str:
        """Convert labels to Prometheus format."""
        if not self.labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
        return "{" + ",".join(parts) + "}"


@dataclass
class MetricValue:
    """A single metric value with labels."""

    value: float
    labels: MetricLabels
    timestamp: float = field(default_factory=time.time)


class Counter:
    """
    A counter metric (monotonically increasing).

    Use for:
    - Request counts
    - Error counts
    - Processed items
    """

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: Dict[MetricLabels, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
        """Increment the counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")

        label_obj = MetricLabels(labels or {})
        with self._lock:
            self._values[label_obj] += value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        label_obj = MetricLabels(labels or {})
        return self._values.get(label_obj, 0.0)

    def labels(self, **kwargs: str) -> "CounterChild":
        """Return a child counter with labels."""
        return CounterChild(self, kwargs)

    def collect(self) -> List[MetricValue]:
        """Collect all metric values."""
        with self._lock:
            return [MetricValue(value=v, labels=l) for l, v in self._values.items()]


class CounterChild:
    """Child counter with bound labels."""

    def __init__(self, parent: Counter, labels: Dict[str, str]):
        self._parent = parent
        self._labels = labels

    def inc(self, value: float = 1.0) -> None:
        """Increment the counter."""
        self._parent.inc(self._labels, value)


class Gauge:
    """
    A gauge metric (can go up or down).

    Use for:
    - Current active connections
    - Queue depth
    - Temperature
    - Memory usage
    """

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: Dict[MetricLabels, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge value."""
        label_obj = MetricLabels(labels or {})
        with self._lock:
            self._values[label_obj] = value

    def inc(self, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
        """Increment the gauge."""
        label_obj = MetricLabels(labels or {})
        with self._lock:
            self._values[label_obj] += value

    def dec(self, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
        """Decrement the gauge."""
        label_obj = MetricLabels(labels or {})
        with self._lock:
            self._values[label_obj] -= value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        label_obj = MetricLabels(labels or {})
        return self._values.get(label_obj, 0.0)

    def labels(self, **kwargs: str) -> "GaugeChild":
        """Return a child gauge with labels."""
        return GaugeChild(self, kwargs)

    def set_function(
        self,
        func: Callable[[], float],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set gauge to track a function's return value."""
        # Store function for later collection
        label_obj = MetricLabels(labels or {})
        self._values[label_obj] = func()

    def collect(self) -> List[MetricValue]:
        """Collect all metric values."""
        with self._lock:
            return [MetricValue(value=v, labels=l) for l, v in self._values.items()]


class GaugeChild:
    """Child gauge with bound labels."""

    def __init__(self, parent: Gauge, labels: Dict[str, str]):
        self._parent = parent
        self._labels = labels

    def set(self, value: float) -> None:
        self._parent.set(value, self._labels)

    def inc(self, value: float = 1.0) -> None:
        self._parent.inc(self._labels, value)

    def dec(self, value: float = 1.0) -> None:
        self._parent.dec(self._labels, value)


class Histogram:
    """
    A histogram metric (value distribution).

    Use for:
    - Request latencies
    - Response sizes
    - Batch sizes
    """

    DEFAULT_BUCKETS = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        float("inf"),
    )

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._buckets: Dict[MetricLabels, Dict[float, int]] = defaultdict(lambda: {b: 0 for b in self.buckets})
        self._sums: Dict[MetricLabels, float] = defaultdict(float)
        self._counts: Dict[MetricLabels, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value."""
        label_obj = MetricLabels(labels or {})
        with self._lock:
            self._sums[label_obj] += value
            self._counts[label_obj] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._buckets[label_obj][bucket] += 1

    def labels(self, **kwargs: str) -> "HistogramChild":
        """Return a child histogram with labels."""
        return HistogramChild(self, kwargs)

    def time(self, labels: Optional[Dict[str, str]] = None) -> "HistogramTimer":
        """Context manager for timing."""
        return HistogramTimer(self, labels)

    def collect(self) -> List[MetricValue]:
        """Collect all metric values."""
        values = []
        with self._lock:
            for label_obj in set(self._buckets.keys()) | set(self._sums.keys()):
                # Add bucket values
                for bucket, count in self._buckets[label_obj].items():
                    bucket_labels = MetricLabels(
                        {
                            **label_obj.labels,
                            "le": str(bucket) if bucket != float("inf") else "+Inf",
                        }
                    )
                    values.append(MetricValue(value=count, labels=bucket_labels))

                # Add sum
                sum_labels = MetricLabels({**label_obj.labels, "aggregate": "sum"})
                values.append(MetricValue(value=self._sums[label_obj], labels=sum_labels))

                # Add count
                count_labels = MetricLabels({**label_obj.labels, "aggregate": "count"})
                values.append(MetricValue(value=self._counts[label_obj], labels=count_labels))

        return values


class HistogramChild:
    """Child histogram with bound labels."""

    def __init__(self, parent: Histogram, labels: Dict[str, str]):
        self._parent = parent
        self._labels = labels

    def observe(self, value: float) -> None:
        self._parent.observe(value, self._labels)

    def time(self) -> "HistogramTimer":
        return HistogramTimer(self._parent, self._labels)


class HistogramTimer:
    """Context manager for timing operations."""

    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]]):
        self._histogram = histogram
        self._labels = labels
        self._start: Optional[float] = None

    def __enter__(self) -> "HistogramTimer":
        self._start = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._start is not None:
            duration = time.time() - self._start
            self._histogram.observe(duration, self._labels)


class Summary:
    """
    A summary metric (quantile calculations).

    Use for:
    - Request latencies with quantiles
    - Response sizes with quantiles
    """

    DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99)

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        quantiles: Optional[Tuple[float, ...]] = None,
        max_age_seconds: int = 600,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.quantiles = quantiles or self.DEFAULT_QUANTILES
        self.max_age_seconds = max_age_seconds
        self._observations: Dict[MetricLabels, List[Tuple[float, float]]] = defaultdict(list)
        self._sums: Dict[MetricLabels, float] = defaultdict(float)
        self._counts: Dict[MetricLabels, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value."""
        label_obj = MetricLabels(labels or {})
        now = time.time()
        cutoff = now - self.max_age_seconds

        with self._lock:
            # Add new observation
            self._observations[label_obj].append((now, value))
            self._sums[label_obj] += value
            self._counts[label_obj] += 1

            # Clean old observations
            self._observations[label_obj] = [(t, v) for t, v in self._observations[label_obj] if t > cutoff]

    def labels(self, **kwargs: str) -> "SummaryChild":
        """Return a child summary with labels."""
        return SummaryChild(self, kwargs)

    def collect(self) -> List[MetricValue]:
        """Collect all metric values."""
        values = []
        with self._lock:
            for label_obj, observations in self._observations.items():
                if not observations:
                    continue

                # Sort observations by value for quantile calculation
                sorted_values = sorted(v for _, v in observations)

                # Calculate quantiles
                for quantile in self.quantiles:
                    idx = int(len(sorted_values) * quantile)
                    idx = min(idx, len(sorted_values) - 1)
                    quantile_labels = MetricLabels(
                        {
                            **label_obj.labels,
                            "quantile": str(quantile),
                        }
                    )
                    values.append(
                        MetricValue(
                            value=sorted_values[idx],
                            labels=quantile_labels,
                        )
                    )

                # Add sum and count
                sum_labels = MetricLabels({**label_obj.labels, "aggregate": "sum"})
                values.append(MetricValue(value=self._sums[label_obj], labels=sum_labels))

                count_labels = MetricLabels({**label_obj.labels, "aggregate": "count"})
                values.append(MetricValue(value=self._counts[label_obj], labels=count_labels))

        return values


class SummaryChild:
    """Child summary with bound labels."""

    def __init__(self, parent: Summary, labels: Dict[str, str]):
        self._parent = parent
        self._labels = labels

    def observe(self, value: float) -> None:
        self._parent.observe(value, self._labels)


class MetricsCollector:
    """
    Central metrics collector.

    Provides:
    - Metric registration and collection
    - Prometheus-format export
    - OpenMetrics-format export
    - Built-in system metrics
    """

    def __init__(
        self,
        namespace: str = "qbitel",
        include_system_metrics: bool = True,
    ):
        self.namespace = namespace
        self._metrics: Dict[str, Union[Counter, Gauge, Histogram, Summary]] = {}
        self._lock = threading.Lock()

        # Register built-in metrics
        if include_system_metrics:
            self._register_system_metrics()

    def counter(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
    ) -> Counter:
        """Create or get a counter metric."""
        full_name = f"{self.namespace}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Counter(full_name, description, label_names)
            return self._metrics[full_name]  # type: ignore

    def gauge(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
    ) -> Gauge:
        """Create or get a gauge metric."""
        full_name = f"{self.namespace}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Gauge(full_name, description, label_names)
            return self._metrics[full_name]  # type: ignore

    def histogram(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        """Create or get a histogram metric."""
        full_name = f"{self.namespace}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Histogram(full_name, description, label_names, buckets)
            return self._metrics[full_name]  # type: ignore

    def summary(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        quantiles: Optional[Tuple[float, ...]] = None,
    ) -> Summary:
        """Create or get a summary metric."""
        full_name = f"{self.namespace}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Summary(full_name, description, label_names, quantiles)
            return self._metrics[full_name]  # type: ignore

    def collect_all(self) -> Dict[str, List[MetricValue]]:
        """Collect all metric values."""
        result = {}
        with self._lock:
            for name, metric in self._metrics.items():
                result[name] = metric.collect()
        return result

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        metrics = self.collect_all()

        for name, values in metrics.items():
            metric = self._metrics.get(name)
            if metric:
                # Type header
                metric_type = self._get_metric_type(metric)
                lines.append(f"# HELP {name} {metric.description}")
                lines.append(f"# TYPE {name} {metric_type}")

                # Values
                for value in values:
                    label_str = value.labels.to_prometheus()
                    lines.append(f"{name}{label_str} {value.value}")

        return "\n".join(lines)

    def _get_metric_type(self, metric: Any) -> str:
        """Get Prometheus type string for metric."""
        if isinstance(metric, Counter):
            return "counter"
        elif isinstance(metric, Gauge):
            return "gauge"
        elif isinstance(metric, Histogram):
            return "histogram"
        elif isinstance(metric, Summary):
            return "summary"
        return "untyped"

    def _register_system_metrics(self) -> None:
        """Register built-in system metrics."""
        import os
        import sys

        # Process info
        self.gauge(
            "process_start_time_seconds",
            "Start time of the process since unix epoch in seconds",
        ).set(time.time())

        # Python info
        info_gauge = self.gauge(
            "python_info",
            "Python version info",
            ["version", "implementation"],
        )
        info_gauge.set(
            1.0,
            {
                "version": sys.version.split()[0],
                "implementation": sys.implementation.name,
            },
        )


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics_collector(
    namespace: str = "qbitel",
    include_system_metrics: bool = True,
) -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    with _metrics_lock:
        if _metrics_collector is None:
            _metrics_collector = MetricsCollector(namespace, include_system_metrics)
        return _metrics_collector


# Pre-defined banking metrics
def create_banking_metrics(collector: Optional[MetricsCollector] = None) -> Dict[str, Any]:
    """Create standard banking metrics."""
    mc = collector or get_metrics_collector()

    return {
        # Transaction metrics
        "transactions_total": mc.counter(
            "transactions_total",
            "Total number of transactions processed",
            ["protocol", "status"],
        ),
        "transaction_duration_seconds": mc.histogram(
            "transaction_duration_seconds",
            "Transaction processing duration in seconds",
            ["protocol"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        ),
        # Cryptographic operations
        "crypto_operations_total": mc.counter(
            "crypto_operations_total",
            "Total cryptographic operations",
            ["operation", "algorithm"],
        ),
        "crypto_operation_duration_seconds": mc.histogram(
            "crypto_operation_duration_seconds",
            "Cryptographic operation duration",
            ["operation", "algorithm"],
        ),
        # Key management
        "keys_total": mc.gauge(
            "keys_total",
            "Total number of managed keys",
            ["type", "state"],
        ),
        "key_rotations_total": mc.counter(
            "key_rotations_total",
            "Total key rotations performed",
            ["type", "reason"],
        ),
        # Certificate management
        "certificates_total": mc.gauge(
            "certificates_total",
            "Total number of managed certificates",
            ["type", "state"],
        ),
        "certificate_days_to_expiry": mc.gauge(
            "certificate_days_to_expiry",
            "Days until certificate expiry",
            ["common_name"],
        ),
        # HSM metrics
        "hsm_operations_total": mc.counter(
            "hsm_operations_total",
            "Total HSM operations",
            ["provider", "operation"],
        ),
        "hsm_operation_duration_seconds": mc.histogram(
            "hsm_operation_duration_seconds",
            "HSM operation duration",
            ["provider", "operation"],
        ),
        "hsm_connection_pool_size": mc.gauge(
            "hsm_connection_pool_size",
            "HSM connection pool size",
            ["provider"],
        ),
        # Verification metrics
        "verifications_total": mc.counter(
            "verifications_total",
            "Total verification operations",
            ["type", "result"],
        ),
        # Health metrics
        "component_health": mc.gauge(
            "component_health",
            "Component health status (1=healthy, 0=unhealthy)",
            ["component"],
        ),
        "incidents_total": mc.counter(
            "incidents_total",
            "Total incidents created",
            ["severity"],
        ),
    }
