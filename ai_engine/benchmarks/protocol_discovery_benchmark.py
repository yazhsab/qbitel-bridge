"""
CRONOS AI Engine - Protocol Discovery Benchmarking Framework

This module provides comprehensive benchmarking capabilities for protocol
discovery systems, including performance, accuracy, and scalability testing.
"""

import asyncio
import logging
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
import concurrent.futures

import numpy as np
import torch
from prometheus_client import Counter, Histogram, Gauge

from ..core.config import Config
from ..discovery.production_protocol_discovery import (
    ProductionProtocolDiscovery,
    DiscoveryRequest,
    DiscoveryResult,
    QualityMode,
)


class BenchmarkType(str, Enum):
    """Types of benchmarks."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    SCALABILITY = "scalability"
    SLA_COMPLIANCE = "sla_compliance"
    RESOURCE_USAGE = "resource_usage"
    STRESS_TEST = "stress_test"


class LoadPattern(str, Enum):
    """Load patterns for stress testing."""

    CONSTANT = "constant"
    RAMP_UP = "ramp_up"
    SPIKE = "spike"
    WAVE = "wave"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    benchmark_type: BenchmarkType
    duration_seconds: int = 60
    num_requests: int = 1000
    concurrent_requests: int = 10
    warmup_requests: int = 100
    sla_threshold_ms: int = 100
    quality_mode: QualityMode = QualityMode.BALANCED
    load_pattern: LoadPattern = LoadPattern.CONSTANT
    collect_metrics: bool = True
    save_results: bool = True
    output_dir: str = "benchmark_results"


@dataclass
class LatencyMetrics:
    """Latency benchmark metrics."""

    mean_ms: float
    median_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float


@dataclass
class ThroughputMetrics:
    """Throughput benchmark metrics."""

    requests_per_second: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    average_latency_ms: float


@dataclass
class AccuracyMetrics:
    """Accuracy benchmark metrics."""

    total_samples: int
    correct_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Dict[str, Dict[str, int]]


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    cpu_percent_mean: float
    cpu_percent_max: float
    memory_mb_mean: float
    memory_mb_max: float
    gpu_memory_mb_mean: Optional[float] = None
    gpu_memory_mb_max: Optional[float] = None
    gpu_utilization_mean: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    benchmark_type: BenchmarkType
    config: BenchmarkConfig
    latency_metrics: Optional[LatencyMetrics] = None
    throughput_metrics: Optional[ThroughputMetrics] = None
    accuracy_metrics: Optional[AccuracyMetrics] = None
    resource_metrics: Optional[ResourceMetrics] = None
    sla_compliance_rate: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProtocolDiscoveryBenchmark:
    """
    Comprehensive benchmarking framework for protocol discovery.

    This class provides various benchmarking capabilities including:
    - Latency measurements (p50, p90, p95, p99)
    - Throughput testing
    - Accuracy evaluation
    - SLA compliance testing
    - Resource usage monitoring
    - Stress testing with various load patterns
    """

    def __init__(self, discovery_system: ProductionProtocolDiscovery, config: Config):
        """Initialize benchmark framework."""
        self.discovery_system = discovery_system
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Test data
        self.test_data: List[Tuple[bytes, str]] = []  # (packet_data, expected_protocol)

        # Results storage
        self.results: List[BenchmarkResult] = []

        # Resource monitoring
        self.resource_monitor = ResourceMonitor()

        self.logger.info("ProtocolDiscoveryBenchmark initialized")

    def load_test_data(self, test_data_path: str) -> None:
        """Load test data from file."""
        try:
            with open(test_data_path, "r") as f:
                data = json.load(f)

            for item in data:
                packet_data = bytes.fromhex(item["packet_hex"])
                expected_protocol = item["protocol"]
                self.test_data.append((packet_data, expected_protocol))

            self.logger.info(f"Loaded {len(self.test_data)} test samples")

        except Exception as e:
            self.logger.error(f"Failed to load test data: {e}")
            raise

    def generate_synthetic_data(
        self, num_samples: int = 1000, protocols: Optional[List[str]] = None
    ) -> None:
        """Generate synthetic test data."""
        if protocols is None:
            protocols = ["HTTP", "DNS", "TLS", "SSH", "FTP"]

        self.test_data = []

        for i in range(num_samples):
            # Generate random packet data
            packet_size = np.random.randint(64, 1500)
            packet_data = np.random.bytes(packet_size)

            # Assign random protocol
            protocol = np.random.choice(protocols)

            self.test_data.append((packet_data, protocol))

        self.logger.info(f"Generated {num_samples} synthetic test samples")

    async def run_latency_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run latency benchmark.

        Measures latency distribution including p50, p90, p95, p99 percentiles.
        """
        self.logger.info("Starting latency benchmark")

        latencies = []

        # Warmup
        await self._warmup(config.warmup_requests)

        # Run benchmark
        for i in range(config.num_requests):
            if not self.test_data:
                self.generate_synthetic_data(config.num_requests)

            packet_data, _ = self.test_data[i % len(self.test_data)]

            request = DiscoveryRequest(
                request_id=f"latency_bench_{i}",
                packet_data=packet_data,
                quality_mode=config.quality_mode,
            )

            start_time = time.time()
            try:
                await self.discovery_system.discover_with_sla(
                    request, sla_ms=config.sla_threshold_ms
                )
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)

            except Exception as e:
                self.logger.warning(f"Request failed: {e}")

        # Calculate metrics
        latency_metrics = LatencyMetrics(
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            p50_ms=np.percentile(latencies, 50),
            p90_ms=np.percentile(latencies, 90),
            p95_ms=np.percentile(latencies, 95),
            p99_ms=np.percentile(latencies, 99),
            p999_ms=np.percentile(latencies, 99.9),
            min_ms=min(latencies),
            max_ms=max(latencies),
            std_dev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        )

        result = BenchmarkResult(
            benchmark_type=BenchmarkType.LATENCY,
            config=config,
            latency_metrics=latency_metrics,
            metadata={
                "total_requests": len(latencies),
                "quality_mode": config.quality_mode.value,
            },
        )

        self.results.append(result)
        self.logger.info(
            f"Latency benchmark completed: p50={latency_metrics.p50_ms:.2f}ms, "
            f"p95={latency_metrics.p95_ms:.2f}ms, p99={latency_metrics.p99_ms:.2f}ms"
        )

        return result

    async def run_throughput_benchmark(
        self, config: BenchmarkConfig
    ) -> BenchmarkResult:
        """
        Run throughput benchmark.

        Measures requests per second under concurrent load.
        """
        self.logger.info(
            f"Starting throughput benchmark with {config.concurrent_requests} concurrent requests"
        )

        if not self.test_data:
            self.generate_synthetic_data(config.num_requests)

        # Warmup
        await self._warmup(config.warmup_requests)

        successful = 0
        failed = 0
        total_latency = 0.0

        start_time = time.time()

        # Create request batches
        batches = []
        for i in range(0, config.num_requests, config.concurrent_requests):
            batch = []
            for j in range(config.concurrent_requests):
                idx = (i + j) % len(self.test_data)
                packet_data, _ = self.test_data[idx]

                request = DiscoveryRequest(
                    request_id=f"throughput_bench_{i}_{j}",
                    packet_data=packet_data,
                    quality_mode=config.quality_mode,
                )
                batch.append(request)
            batches.append(batch)

        # Execute batches concurrently
        for batch in batches:
            tasks = [
                self.discovery_system.discover_with_sla(req, config.sla_threshold_ms)
                for req in batch
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                else:
                    successful += 1
                    total_latency += result.processing_time_ms

        duration = time.time() - start_time

        throughput_metrics = ThroughputMetrics(
            requests_per_second=config.num_requests / duration,
            total_requests=config.num_requests,
            successful_requests=successful,
            failed_requests=failed,
            duration_seconds=duration,
            average_latency_ms=total_latency / successful if successful > 0 else 0.0,
        )

        result = BenchmarkResult(
            benchmark_type=BenchmarkType.THROUGHPUT,
            config=config,
            throughput_metrics=throughput_metrics,
            metadata={
                "concurrent_requests": config.concurrent_requests,
                "quality_mode": config.quality_mode.value,
            },
        )

        self.results.append(result)
        self.logger.info(
            f"Throughput benchmark completed: {throughput_metrics.requests_per_second:.2f} req/s"
        )

        return result

    async def run_accuracy_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run accuracy benchmark.

        Evaluates prediction accuracy against ground truth labels.
        """
        self.logger.info("Starting accuracy benchmark")

        if not self.test_data:
            raise ValueError("Test data with labels required for accuracy benchmark")

        predictions = []
        ground_truth = []

        for i, (packet_data, expected_protocol) in enumerate(
            self.test_data[: config.num_requests]
        ):
            request = DiscoveryRequest(
                request_id=f"accuracy_bench_{i}",
                packet_data=packet_data,
                quality_mode=config.quality_mode,
            )

            try:
                result = await self.discovery_system.discover_with_sla(
                    request, sla_ms=config.sla_threshold_ms
                )
                predictions.append(result.protocol_type)
                ground_truth.append(expected_protocol)

            except Exception as e:
                self.logger.warning(f"Prediction failed: {e}")

        # Calculate metrics
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        accuracy = correct / len(predictions) if predictions else 0.0

        # Calculate per-class metrics (simplified)
        unique_protocols = set(ground_truth)
        confusion_matrix = {
            protocol: {p: 0 for p in unique_protocols} for protocol in unique_protocols
        }

        for pred, true in zip(predictions, ground_truth):
            if pred in confusion_matrix[true]:
                confusion_matrix[true][pred] += 1

        # Calculate precision, recall, F1 (macro-averaged)
        precision_scores = []
        recall_scores = []

        for protocol in unique_protocols:
            tp = confusion_matrix[protocol].get(protocol, 0)
            fp = sum(
                confusion_matrix[p].get(protocol, 0)
                for p in unique_protocols
                if p != protocol
            )
            fn = sum(
                confusion_matrix[protocol].get(p, 0)
                for p in unique_protocols
                if p != protocol
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            precision_scores.append(precision)
            recall_scores.append(recall)

        avg_precision = statistics.mean(precision_scores) if precision_scores else 0.0
        avg_recall = statistics.mean(recall_scores) if recall_scores else 0.0
        f1 = (
            2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0.0
        )

        accuracy_metrics = AccuracyMetrics(
            total_samples=len(predictions),
            correct_predictions=correct,
            accuracy=accuracy,
            precision=avg_precision,
            recall=avg_recall,
            f1_score=f1,
            confusion_matrix=confusion_matrix,
        )

        result = BenchmarkResult(
            benchmark_type=BenchmarkType.ACCURACY,
            config=config,
            accuracy_metrics=accuracy_metrics,
            metadata={
                "num_protocols": len(unique_protocols),
                "quality_mode": config.quality_mode.value,
            },
        )

        self.results.append(result)
        self.logger.info(
            f"Accuracy benchmark completed: accuracy={accuracy:.2%}, F1={f1:.2%}"
        )

        return result

    async def run_sla_compliance_benchmark(
        self, config: BenchmarkConfig
    ) -> BenchmarkResult:
        """
        Run SLA compliance benchmark.

        Tests system's ability to meet SLA guarantees under load.
        """
        self.logger.info(
            f"Starting SLA compliance benchmark (threshold: {config.sla_threshold_ms}ms)"
        )

        if not self.test_data:
            self.generate_synthetic_data(config.num_requests)

        sla_met_count = 0
        sla_violated_count = 0

        for i in range(config.num_requests):
            packet_data, _ = self.test_data[i % len(self.test_data)]

            request = DiscoveryRequest(
                request_id=f"sla_bench_{i}",
                packet_data=packet_data,
                quality_mode=config.quality_mode,
            )

            try:
                result = await self.discovery_system.discover_with_sla(
                    request, sla_ms=config.sla_threshold_ms
                )

                if result.sla_met:
                    sla_met_count += 1
                else:
                    sla_violated_count += 1

            except Exception as e:
                self.logger.warning(f"Request failed: {e}")
                sla_violated_count += 1

        compliance_rate = sla_met_count / (sla_met_count + sla_violated_count)

        result = BenchmarkResult(
            benchmark_type=BenchmarkType.SLA_COMPLIANCE,
            config=config,
            sla_compliance_rate=compliance_rate,
            metadata={
                "sla_met": sla_met_count,
                "sla_violated": sla_violated_count,
                "threshold_ms": config.sla_threshold_ms,
            },
        )

        self.results.append(result)
        self.logger.info(
            f"SLA compliance benchmark completed: {compliance_rate:.2%} compliance rate"
        )

        return result

    async def run_resource_usage_benchmark(
        self, config: BenchmarkConfig
    ) -> BenchmarkResult:
        """
        Run resource usage benchmark.

        Monitors CPU, memory, and GPU usage during operation.
        """
        self.logger.info("Starting resource usage benchmark")

        if not self.test_data:
            self.generate_synthetic_data(config.num_requests)

        # Start resource monitoring
        self.resource_monitor.start()

        # Run requests
        for i in range(config.num_requests):
            packet_data, _ = self.test_data[i % len(self.test_data)]

            request = DiscoveryRequest(
                request_id=f"resource_bench_{i}",
                packet_data=packet_data,
                quality_mode=config.quality_mode,
            )

            try:
                await self.discovery_system.discover_with_sla(
                    request, sla_ms=config.sla_threshold_ms
                )
            except Exception as e:
                self.logger.warning(f"Request failed: {e}")

        # Stop monitoring and get metrics
        resource_metrics = self.resource_monitor.stop()

        result = BenchmarkResult(
            benchmark_type=BenchmarkType.RESOURCE_USAGE,
            config=config,
            resource_metrics=resource_metrics,
            metadata={"quality_mode": config.quality_mode.value},
        )

        self.results.append(result)
        self.logger.info(
            f"Resource usage benchmark completed: "
            f"CPU={resource_metrics.cpu_percent_mean:.1f}%, "
            f"Memory={resource_metrics.memory_mb_mean:.1f}MB"
        )

        return result

    async def run_stress_test(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run stress test with configurable load patterns.

        Tests system behavior under various load conditions.
        """
        self.logger.info(
            f"Starting stress test with {config.load_pattern.value} load pattern"
        )

        if not self.test_data:
            self.generate_synthetic_data(config.num_requests * 2)

        # Generate load pattern
        load_schedule = self._generate_load_pattern(config)

        results_data = []

        for time_point, concurrent_load in load_schedule:
            # Execute concurrent requests
            tasks = []
            for i in range(concurrent_load):
                idx = len(results_data) % len(self.test_data)
                packet_data, _ = self.test_data[idx]

                request = DiscoveryRequest(
                    request_id=f"stress_test_{time_point}_{i}",
                    packet_data=packet_data,
                    quality_mode=config.quality_mode,
                )

                tasks.append(
                    self.discovery_system.discover_with_sla(
                        request, config.sla_threshold_ms
                    )
                )

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in batch_results:
                if not isinstance(res, Exception):
                    results_data.append(
                        {
                            "time": time_point,
                            "latency_ms": res.processing_time_ms,
                            "sla_met": res.sla_met,
                            "load": concurrent_load,
                        }
                    )

        # Analyze results
        latencies = [r["latency_ms"] for r in results_data]
        sla_met = sum(1 for r in results_data if r["sla_met"])

        latency_metrics = LatencyMetrics(
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            p50_ms=np.percentile(latencies, 50),
            p90_ms=np.percentile(latencies, 90),
            p95_ms=np.percentile(latencies, 95),
            p99_ms=np.percentile(latencies, 99),
            p999_ms=np.percentile(latencies, 99.9),
            min_ms=min(latencies),
            max_ms=max(latencies),
            std_dev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        )

        result = BenchmarkResult(
            benchmark_type=BenchmarkType.STRESS_TEST,
            config=config,
            latency_metrics=latency_metrics,
            sla_compliance_rate=sla_met / len(results_data),
            metadata={
                "load_pattern": config.load_pattern.value,
                "max_concurrent_load": max(load for _, load in load_schedule),
                "total_requests": len(results_data),
            },
        )

        self.results.append(result)
        self.logger.info(
            f"Stress test completed: {len(results_data)} requests processed"
        )

        return result

    async def run_full_benchmark_suite(
        self, base_config: BenchmarkConfig
    ) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        self.logger.info("Starting full benchmark suite")

        results = []

        # Latency benchmark
        latency_config = BenchmarkConfig(**asdict(base_config))
        latency_config.benchmark_type = BenchmarkType.LATENCY
        results.append(await self.run_latency_benchmark(latency_config))

        # Throughput benchmark
        throughput_config = BenchmarkConfig(**asdict(base_config))
        throughput_config.benchmark_type = BenchmarkType.THROUGHPUT
        results.append(await self.run_throughput_benchmark(throughput_config))

        # Accuracy benchmark (if test data has labels)
        if self.test_data:
            accuracy_config = BenchmarkConfig(**asdict(base_config))
            accuracy_config.benchmark_type = BenchmarkType.ACCURACY
            results.append(await self.run_accuracy_benchmark(accuracy_config))

        # SLA compliance benchmark
        sla_config = BenchmarkConfig(**asdict(base_config))
        sla_config.benchmark_type = BenchmarkType.SLA_COMPLIANCE
        results.append(await self.run_sla_compliance_benchmark(sla_config))

        # Resource usage benchmark
        resource_config = BenchmarkConfig(**asdict(base_config))
        resource_config.benchmark_type = BenchmarkType.RESOURCE_USAGE
        results.append(await self.run_resource_usage_benchmark(resource_config))

        self.logger.info("Full benchmark suite completed")

        return results

    def save_results(self, output_path: Optional[str] = None) -> None:
        """Save benchmark results to file."""
        if output_path is None:
            output_path = f"benchmark_results_{int(time.time())}.json"

        results_data = [asdict(result) for result in self.results]

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        self.logger.info(f"Benchmark results saved to {output_path}")

    def generate_report(self) -> str:
        """Generate human-readable benchmark report."""
        report_lines = ["=" * 80, "PROTOCOL DISCOVERY BENCHMARK REPORT", "=" * 80, ""]

        for result in self.results:
            report_lines.append(f"\n{result.benchmark_type.value.upper()} BENCHMARK")
            report_lines.append("-" * 80)

            if result.latency_metrics:
                m = result.latency_metrics
                report_lines.extend(
                    [
                        f"  Mean Latency:    {m.mean_ms:.2f} ms",
                        f"  Median Latency:  {m.median_ms:.2f} ms",
                        f"  P95 Latency:     {m.p95_ms:.2f} ms",
                        f"  P99 Latency:     {m.p99_ms:.2f} ms",
                        f"  Min/Max:         {m.min_ms:.2f} / {m.max_ms:.2f} ms",
                    ]
                )

            if result.throughput_metrics:
                m = result.throughput_metrics
                report_lines.extend(
                    [
                        f"  Throughput:      {m.requests_per_second:.2f} req/s",
                        f"  Total Requests:  {m.total_requests}",
                        f"  Success Rate:    {m.successful_requests/m.total_requests:.2%}",
                        f"  Avg Latency:     {m.average_latency_ms:.2f} ms",
                    ]
                )

            if result.accuracy_metrics:
                m = result.accuracy_metrics
                report_lines.extend(
                    [
                        f"  Accuracy:        {m.accuracy:.2%}",
                        f"  Precision:       {m.precision:.2%}",
                        f"  Recall:          {m.recall:.2%}",
                        f"  F1 Score:        {m.f1_score:.2%}",
                    ]
                )

            if result.sla_compliance_rate is not None:
                report_lines.append(
                    f"  SLA Compliance:  {result.sla_compliance_rate:.2%}"
                )

            if result.resource_metrics:
                m = result.resource_metrics
                report_lines.extend(
                    [
                        f"  CPU Usage:       {m.cpu_percent_mean:.1f}% (max: {m.cpu_percent_max:.1f}%)",
                        f"  Memory Usage:    {m.memory_mb_mean:.1f} MB (max: {m.memory_mb_max:.1f} MB)",
                    ]
                )

                if m.gpu_memory_mb_mean:
                    report_lines.append(
                        f"  GPU Memory:      {m.gpu_memory_mb_mean:.1f} MB"
                    )

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)

    async def _warmup(self, num_requests: int) -> None:
        """Warmup the system before benchmarking."""
        self.logger.info(f"Warming up with {num_requests} requests")

        if not self.test_data:
            self.generate_synthetic_data(num_requests)

        for i in range(num_requests):
            packet_data, _ = self.test_data[i % len(self.test_data)]
            request = DiscoveryRequest(
                request_id=f"warmup_{i}", packet_data=packet_data
            )

            try:
                await self.discovery_system.discover_with_sla(request, sla_ms=1000)
            except:
                pass

    def _generate_load_pattern(
        self, config: BenchmarkConfig
    ) -> List[Tuple[float, int]]:
        """Generate load pattern for stress testing."""
        schedule = []
        duration = config.duration_seconds
        max_load = config.concurrent_requests

        if config.load_pattern == LoadPattern.CONSTANT:
            # Constant load
            for t in range(duration):
                schedule.append((t, max_load))

        elif config.load_pattern == LoadPattern.RAMP_UP:
            # Gradual ramp up
            for t in range(duration):
                load = int(max_load * (t / duration))
                schedule.append((t, max(1, load)))

        elif config.load_pattern == LoadPattern.SPIKE:
            # Sudden spike in the middle
            for t in range(duration):
                if duration // 3 <= t < 2 * duration // 3:
                    load = max_load
                else:
                    load = max_load // 4
                schedule.append((t, load))

        elif config.load_pattern == LoadPattern.WAVE:
            # Sinusoidal wave pattern
            for t in range(duration):
                load = int(
                    max_load * (0.5 + 0.5 * np.sin(2 * np.pi * t / (duration / 3)))
                )
                schedule.append((t, max(1, load)))

        return schedule


class ResourceMonitor:
    """Monitor system resource usage."""

    def __init__(self):
        """Initialize resource monitor."""
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.metrics = {
            "cpu_percent": [],
            "memory_mb": [],
            "gpu_memory_mb": [],
            "gpu_utilization": [],
        }
        self._monitor_task = None

    def start(self) -> None:
        """Start monitoring resources."""
        import psutil

        self.monitoring = True
        self.metrics = {key: [] for key in self.metrics.keys()}

        async def monitor_loop():
            while self.monitoring:
                # CPU and memory
                self.metrics["cpu_percent"].append(psutil.cpu_percent(interval=0.1))
                memory = psutil.virtual_memory()
                self.metrics["memory_mb"].append(memory.used / (1024 * 1024))

                # GPU metrics
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    self.metrics["gpu_memory_mb"].append(gpu_memory)

                await asyncio.sleep(1)

        self._monitor_task = asyncio.create_task(monitor_loop())

    def stop(self) -> ResourceMetrics:
        """Stop monitoring and return metrics."""
        self.monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()

        return ResourceMetrics(
            cpu_percent_mean=(
                statistics.mean(self.metrics["cpu_percent"])
                if self.metrics["cpu_percent"]
                else 0.0
            ),
            cpu_percent_max=(
                max(self.metrics["cpu_percent"]) if self.metrics["cpu_percent"] else 0.0
            ),
            memory_mb_mean=(
                statistics.mean(self.metrics["memory_mb"])
                if self.metrics["memory_mb"]
                else 0.0
            ),
            memory_mb_max=(
                max(self.metrics["memory_mb"]) if self.metrics["memory_mb"] else 0.0
            ),
            gpu_memory_mb_mean=(
                statistics.mean(self.metrics["gpu_memory_mb"])
                if self.metrics["gpu_memory_mb"]
                else None
            ),
            gpu_memory_mb_max=(
                max(self.metrics["gpu_memory_mb"])
                if self.metrics["gpu_memory_mb"]
                else None
            ),
            gpu_utilization_mean=(
                statistics.mean(self.metrics["gpu_utilization"])
                if self.metrics["gpu_utilization"]
                else None
            ),
        )
