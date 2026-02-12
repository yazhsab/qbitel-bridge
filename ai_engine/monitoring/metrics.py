"""
QBITEL Engine - Metrics Collection

This module provides comprehensive metrics collection and monitoring
for the AI Engine using Prometheus and custom metrics.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from contextlib import contextmanager

import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from prometheus_client.core import CollectorRegistry

try:  # Optional dependency for system metrics
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    psutil = None  # type: ignore[assignment]

try:  # Optional dependency for GPU metrics
    import torch  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from ..core.config import Config

try:  # Optional custom exception
    from ..core.exceptions import MonitoringException
except ImportError:  # pragma: no cover - fallback when symbol missing

    class MonitoringException(Exception):
        """Fallback monitoring exception used when core definition is unavailable."""

        pass


class MetricType(str, Enum):
    """Metric types supported by the system."""

    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricConfig:
    """Configuration for a metric."""

    name: str
    help: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: str = ""
    namespace: str = "qbitel"


class MetricsCollector:
    """
    Base metrics collector for QBITEL Engine.

    This class provides a foundation for collecting and managing
    various metrics throughout the AI Engine lifecycle.
    """

    def __init__(self, config: Config):
        """Initialize metrics collector."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Metric storage
        self.metrics: Dict[str, Any] = {}
        self.custom_metrics: Dict[str, Any] = {}

        # Thread safety
        self.lock = threading.RLock()

        # Collection state
        self.collection_enabled = True
        self.collection_interval = getattr(config, "metrics_collection_interval", 15)  # seconds

        # Background collection task
        self._collection_task: Optional[asyncio.Task] = None
        self._stop_collection = asyncio.Event()

        self.logger.info("MetricsCollector initialized")

    async def initialize(self) -> None:
        """Maintain compatibility with call sites expecting initialize."""
        await self.start()

    def register_metric(self, metric_config: MetricConfig, metric_type: MetricType) -> Any:
        """Register a new metric."""
        with self.lock:
            if metric_config.name in self.metrics:
                self.logger.warning(f"Metric {metric_config.name} already registered")
                return self.metrics[metric_config.name]

            try:
                if metric_type == MetricType.COUNTER:
                    metric = Counter(
                        metric_config.name,
                        metric_config.help,
                        labelnames=metric_config.labels,
                        namespace=metric_config.namespace,
                    )
                elif metric_type == MetricType.HISTOGRAM:
                    buckets = metric_config.buckets or prometheus_client.DEFAULT_BUCKETS
                    metric = Histogram(
                        metric_config.name,
                        metric_config.help,
                        labelnames=metric_config.labels,
                        buckets=buckets,
                        namespace=metric_config.namespace,
                    )
                elif metric_type == MetricType.GAUGE:
                    metric = Gauge(
                        metric_config.name,
                        metric_config.help,
                        labelnames=metric_config.labels,
                        namespace=metric_config.namespace,
                    )
                elif metric_type == MetricType.SUMMARY:
                    metric = Summary(
                        metric_config.name,
                        metric_config.help,
                        labelnames=metric_config.labels,
                        namespace=metric_config.namespace,
                    )
                elif metric_type == MetricType.INFO:
                    metric = Info(
                        metric_config.name,
                        metric_config.help,
                        labelnames=metric_config.labels,
                        namespace=metric_config.namespace,
                    )
                else:
                    raise MonitoringException(f"Unsupported metric type: {metric_type}")

                self.metrics[metric_config.name] = metric
                self.logger.info(f"Registered metric: {metric_config.name} ({metric_type})")
                return metric

            except Exception as e:
                self.logger.error(f"Failed to register metric {metric_config.name}: {e}")
                raise MonitoringException(f"Metric registration failed: {e}")

    def get_metric(self, name: str) -> Optional[Any]:
        """Get a registered metric by name."""
        with self.lock:
            return self.metrics.get(name)

    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
        """Increment a counter metric."""
        metric = self.get_metric(name)
        if metric and hasattr(metric, "inc"):
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        metric = self.get_metric(name)
        if metric and hasattr(metric, "set"):
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value in a histogram metric."""
        metric = self.get_metric(name)
        if metric and hasattr(metric, "observe"):
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)

    def observe_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value in a summary metric."""
        metric = self.get_metric(name)
        if metric and hasattr(metric, "observe"):
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)

    @contextmanager
    def time_operation(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe_histogram(metric_name, duration, labels)

    async def start(self) -> None:
        """Start metrics collector and background collection."""
        await self.start_collection()
        self.logger.info("MetricsCollector started")

    async def start_collection(self) -> None:
        """Start background metrics collection."""
        if self._collection_task:
            self.logger.warning("Metrics collection already started")
            return

        self._collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Started background metrics collection")

    async def shutdown(self) -> None:
        """Shutdown metrics collector."""
        await self.stop_collection()
        self.logger.info("MetricsCollector shutdown complete")

    async def stop_collection(self) -> None:
        """Stop background metrics collection."""
        if not self._collection_task:
            return

        self._stop_collection.set()

        try:
            await asyncio.wait_for(self._collection_task, timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("Metrics collection task did not stop gracefully, cancelling...")
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        self._collection_task = None
        self._stop_collection.clear()
        self.logger.info("Stopped background metrics collection")

    async def _collection_loop(self) -> None:
        """Background collection loop."""
        while not self._stop_collection.is_set():
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            if psutil is None:
                self.logger.debug("psutil not available; skipping system metrics collection")
                return

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("system_cpu_percent", cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_percent", memory.percent)
            self.set_gauge("system_memory_available_bytes", memory.available)
            self.set_gauge("system_memory_used_bytes", memory.used)

            # Disk metrics
            disk = psutil.disk_usage("/")
            self.set_gauge("system_disk_percent", (disk.used / disk.total) * 100)
            self.set_gauge("system_disk_free_bytes", disk.free)

            # GPU metrics (if available)
            await self._collect_gpu_metrics()

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    async def _collect_gpu_metrics(self) -> None:
        """Collect GPU metrics if available."""
        try:
            if torch is None:
                self.logger.debug("PyTorch not available; skipping GPU metrics collection")
                return

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                self.set_gauge("gpu_device_count", device_count)

                for i in range(device_count):
                    # Memory metrics
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    max_memory = torch.cuda.max_memory_allocated(i)

                    labels = {"device": str(i)}
                    self.set_gauge("gpu_memory_allocated_bytes", memory_allocated, labels)
                    self.set_gauge("gpu_memory_reserved_bytes", memory_reserved, labels)
                    self.set_gauge("gpu_max_memory_allocated_bytes", max_memory, labels)

                    # Utilization (would require nvidia-ml-py for detailed metrics)
                    # For now, just indicate GPU is available
                    self.set_gauge("gpu_available", 1, labels)

        except Exception as e:
            self.logger.error(f"Error collecting GPU metrics: {e}")

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        with self.lock:
            return self.metrics.copy()

    def clear_metrics(self) -> None:
        """Clear all metrics."""
        with self.lock:
            self.metrics.clear()
            self.logger.info("Cleared all metrics")


class PrometheusMetrics(MetricsCollector):
    """
    Prometheus-specific metrics collector.

    This class extends the base metrics collector with Prometheus-specific
    functionality and standard metric registration.
    """

    def __init__(self, config: Config):
        """Initialize Prometheus metrics collector."""
        super().__init__(config)

        # Prometheus registry
        self.registry = CollectorRegistry()

        # HTTP server for metrics endpoint
        self.metrics_server = None
        self.metrics_port = getattr(config, "metrics_port", 9090)

        # Initialize standard metrics
        self._initialize_standard_metrics()

        self.logger.info("PrometheusMetrics initialized")

    def _initialize_standard_metrics(self) -> None:
        """Initialize standard Prometheus metrics."""
        # Request metrics
        self.register_metric(
            MetricConfig(
                name="http_requests_total",
                help="Total number of HTTP requests",
                labels=["method", "endpoint", "status_code"],
            ),
            MetricType.COUNTER,
        )

        self.register_metric(
            MetricConfig(
                name="http_request_duration_seconds",
                help="HTTP request duration in seconds",
                labels=["method", "endpoint"],
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            ),
            MetricType.HISTOGRAM,
        )

        # AI Engine specific metrics
        self.register_metric(
            MetricConfig(
                name="protocol_discovery_requests_total",
                help="Total protocol discovery requests",
                labels=["protocol_type", "success"],
            ),
            MetricType.COUNTER,
        )

        self.register_metric(
            MetricConfig(
                name="protocol_discovery_duration_seconds",
                help="Protocol discovery duration in seconds",
                labels=["protocol_type"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
            ),
            MetricType.HISTOGRAM,
        )

        self.register_metric(
            MetricConfig(
                name="field_detection_requests_total",
                help="Total field detection requests",
                labels=["protocol_type", "success"],
            ),
            MetricType.COUNTER,
        )

        self.register_metric(
            MetricConfig(
                name="field_detection_duration_seconds",
                help="Field detection duration in seconds",
                labels=["protocol_type"],
            ),
            MetricType.HISTOGRAM,
        )

        self.register_metric(
            MetricConfig(
                name="anomaly_detection_requests_total",
                help="Total anomaly detection requests",
                labels=["anomaly_detected", "success"],
            ),
            MetricType.COUNTER,
        )

        self.register_metric(
            MetricConfig(
                name="anomaly_detection_duration_seconds",
                help="Anomaly detection duration in seconds",
            ),
            MetricType.HISTOGRAM,
        )

        # Model metrics
        self.register_metric(
            MetricConfig(
                name="model_inference_requests_total",
                help="Total model inference requests",
                labels=["model_name", "model_version", "success"],
            ),
            MetricType.COUNTER,
        )

        self.register_metric(
            MetricConfig(
                name="model_inference_duration_seconds",
                help="Model inference duration in seconds",
                labels=["model_name", "model_version"],
            ),
            MetricType.HISTOGRAM,
        )

        # System metrics
        self.register_metric(
            MetricConfig(name="system_cpu_percent", help="System CPU usage percentage"),
            MetricType.GAUGE,
        )

        self.register_metric(
            MetricConfig(name="system_memory_percent", help="System memory usage percentage"),
            MetricType.GAUGE,
        )

        self.register_metric(
            MetricConfig(
                name="system_memory_available_bytes",
                help="Available system memory in bytes",
            ),
            MetricType.GAUGE,
        )

        self.register_metric(
            MetricConfig(name="system_memory_used_bytes", help="Used system memory in bytes"),
            MetricType.GAUGE,
        )

        self.register_metric(
            MetricConfig(name="gpu_device_count", help="Number of available GPU devices"),
            MetricType.GAUGE,
        )

        self.register_metric(
            MetricConfig(
                name="gpu_memory_allocated_bytes",
                help="GPU memory allocated in bytes",
                labels=["device"],
            ),
            MetricType.GAUGE,
        )

    def start_metrics_server(self) -> None:
        """Start Prometheus metrics HTTP server."""
        try:
            from prometheus_client import start_http_server

            start_http_server(self.metrics_port, registry=self.registry)
            self.logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
            raise MonitoringException(f"Metrics server startup failed: {e}")

    def stop_metrics_server(self) -> None:
        """Stop Prometheus metrics HTTP server."""
        # Prometheus client doesn't provide a direct way to stop the server
        # In production, this would be handled by process lifecycle
        self.logger.info("Metrics server stop requested")


class CustomMetrics:
    """
    Custom metrics collector for application-specific metrics.

    This class provides functionality for collecting and managing
    custom business logic metrics that are specific to the AI Engine.
    """

    def __init__(self):
        """Initialize custom metrics collector."""
        self.logger = logging.getLogger(__name__)

        # Custom metric storage
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}

        # Thread safety
        self.lock = threading.RLock()

        self.logger.info("CustomMetrics initialized")

    def increment_custom_counter(self, name: str, value: int = 1) -> None:
        """Increment a custom counter."""
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + value

    def set_custom_gauge(self, name: str, value: float) -> None:
        """Set a custom gauge value."""
        with self.lock:
            self.gauges[name] = value

    def add_histogram_observation(self, name: str, value: float) -> None:
        """Add an observation to a custom histogram."""
        with self.lock:
            if name not in self.histograms:
                self.histograms[name] = []
            self.histograms[name].append(value)

            # Keep only last 1000 observations
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]

    def get_custom_metrics(self) -> Dict[str, Any]:
        """Get all custom metrics."""
        with self.lock:
            metrics = {
                "counters": self.counters.copy(),
                "gauges": self.gauges.copy(),
                "histograms": {},
            }

            # Calculate histogram statistics
            for name, values in self.histograms.items():
                if values:
                    metrics["histograms"][name] = {
                        "count": len(values),
                        "sum": sum(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                    }

            return metrics

    def clear_custom_metrics(self) -> None:
        """Clear all custom metrics."""
        with self.lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.logger.info("Cleared all custom metrics")


class AIEngineMetrics:
    """
    AI Engine specific metrics collector.

    This class provides high-level metrics collection specifically
    tailored for AI Engine operations and performance monitoring.
    """

    def __init__(self, prometheus_metrics: PrometheusMetrics):
        """Initialize AI Engine metrics."""
        self.prometheus = prometheus_metrics
        self.custom = CustomMetrics()
        self.logger = logging.getLogger(__name__)

        # Operation tracking
        self.operation_start_times: Dict[str, float] = {}

        self.logger.info("AIEngineMetrics initialized")

    @contextmanager
    def track_protocol_discovery(self, protocol_hint: Optional[str] = None):
        """Context manager for tracking protocol discovery operations."""
        operation_id = f"protocol_discovery_{int(time.time() * 1000000)}"
        start_time = time.time()
        self.operation_start_times[operation_id] = start_time

        success = False
        discovered_protocol = "unknown"

        try:
            yield
            success = True
        except Exception as e:
            self.logger.error(f"Protocol discovery failed: {e}")
            success = False
            raise
        finally:
            duration = time.time() - start_time

            # Record metrics
            labels = {
                "protocol_type": discovered_protocol,
                "success": str(success).lower(),
            }
            self.prometheus.increment_counter("protocol_discovery_requests_total", labels)

            duration_labels = {"protocol_type": discovered_protocol}
            self.prometheus.observe_histogram("protocol_discovery_duration_seconds", duration, duration_labels)

            # Custom metrics
            self.custom.increment_custom_counter("protocol_discovery_total")
            if success:
                self.custom.increment_custom_counter("protocol_discovery_success")
            else:
                self.custom.increment_custom_counter("protocol_discovery_failures")

            self.custom.add_histogram_observation("protocol_discovery_latency", duration)

            # Cleanup
            self.operation_start_times.pop(operation_id, None)

    @contextmanager
    def track_field_detection(self, protocol_type: Optional[str] = None):
        """Context manager for tracking field detection operations."""
        operation_id = f"field_detection_{int(time.time() * 1000000)}"
        start_time = time.time()
        self.operation_start_times[operation_id] = start_time

        success = False
        protocol = protocol_type or "unknown"

        try:
            yield
            success = True
        except Exception as e:
            self.logger.error(f"Field detection failed: {e}")
            success = False
            raise
        finally:
            duration = time.time() - start_time

            # Record metrics
            labels = {"protocol_type": protocol, "success": str(success).lower()}
            self.prometheus.increment_counter("field_detection_requests_total", labels)

            duration_labels = {"protocol_type": protocol}
            self.prometheus.observe_histogram("field_detection_duration_seconds", duration, duration_labels)

            # Custom metrics
            self.custom.increment_custom_counter("field_detection_total")
            if success:
                self.custom.increment_custom_counter("field_detection_success")
            else:
                self.custom.increment_custom_counter("field_detection_failures")

            self.custom.add_histogram_observation("field_detection_latency", duration)

            # Cleanup
            self.operation_start_times.pop(operation_id, None)

    @contextmanager
    def track_anomaly_detection(self):
        """Context manager for tracking anomaly detection operations."""
        operation_id = f"anomaly_detection_{int(time.time() * 1000000)}"
        start_time = time.time()
        self.operation_start_times[operation_id] = start_time

        success = False
        anomaly_detected = False

        try:
            yield
            success = True
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            success = False
            raise
        finally:
            duration = time.time() - start_time

            # Record metrics
            labels = {
                "anomaly_detected": str(anomaly_detected).lower(),
                "success": str(success).lower(),
            }
            self.prometheus.increment_counter("anomaly_detection_requests_total", labels)

            self.prometheus.observe_histogram("anomaly_detection_duration_seconds", duration)

            # Custom metrics
            self.custom.increment_custom_counter("anomaly_detection_total")
            if success:
                self.custom.increment_custom_counter("anomaly_detection_success")
                if anomaly_detected:
                    self.custom.increment_custom_counter("anomalies_detected")
            else:
                self.custom.increment_custom_counter("anomaly_detection_failures")

            self.custom.add_histogram_observation("anomaly_detection_latency", duration)

            # Cleanup
            self.operation_start_times.pop(operation_id, None)

    @contextmanager
    def track_model_inference(self, model_name: str, model_version: str = "unknown"):
        """Context manager for tracking model inference operations."""
        operation_id = f"model_inference_{int(time.time() * 1000000)}"
        start_time = time.time()
        self.operation_start_times[operation_id] = start_time

        success = False

        try:
            yield
            success = True
        except Exception as e:
            self.logger.error(f"Model inference failed for {model_name}:{model_version}: {e}")
            success = False
            raise
        finally:
            duration = time.time() - start_time

            # Record metrics
            labels = {
                "model_name": model_name,
                "model_version": model_version,
                "success": str(success).lower(),
            }
            self.prometheus.increment_counter("model_inference_requests_total", labels)

            duration_labels = {"model_name": model_name, "model_version": model_version}
            self.prometheus.observe_histogram("model_inference_duration_seconds", duration, duration_labels)

            # Custom metrics
            model_key = f"{model_name}:{model_version}"
            self.custom.increment_custom_counter(f"model_inference_total_{model_key}")
            if success:
                self.custom.increment_custom_counter(f"model_inference_success_{model_key}")
            else:
                self.custom.increment_custom_counter(f"model_inference_failures_{model_key}")

            self.custom.add_histogram_observation(f"model_inference_latency_{model_key}", duration)

            # Cleanup
            self.operation_start_times.pop(operation_id, None)

    def record_data_processing_metrics(self, data_size: int, processing_time: float) -> None:
        """Record data processing metrics."""
        self.custom.add_histogram_observation("data_processing_size_bytes", float(data_size))
        self.custom.add_histogram_observation("data_processing_time_seconds", processing_time)

        # Calculate throughput
        if processing_time > 0:
            throughput = data_size / processing_time  # bytes per second
            self.custom.add_histogram_observation("data_processing_throughput_bps", throughput)

    def record_model_accuracy(self, model_name: str, accuracy: float) -> None:
        """Record model accuracy metrics."""
        self.custom.set_custom_gauge(f"model_accuracy_{model_name}", accuracy)

    def get_operation_summary(self) -> Dict[str, Any]:
        """Get summary of current operations."""
        current_time = time.time()
        active_operations = {}

        for operation_id, start_time in self.operation_start_times.items():
            duration = current_time - start_time
            operation_type = operation_id.split("_")[0]

            if operation_type not in active_operations:
                active_operations[operation_type] = []

            active_operations[operation_type].append(
                {
                    "operation_id": operation_id,
                    "duration": duration,
                    "start_time": start_time,
                }
            )

        return {
            "active_operations": active_operations,
            "total_active": len(self.operation_start_times),
            "custom_metrics": self.custom.get_custom_metrics(),
        }

    async def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external consumption."""
        return {
            "prometheus_metrics": self.prometheus.get_all_metrics(),
            "custom_metrics": self.custom.get_custom_metrics(),
            "operation_summary": self.get_operation_summary(),
            "timestamp": time.time(),
        }
