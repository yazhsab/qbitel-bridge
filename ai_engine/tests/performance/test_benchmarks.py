"""
Performance Benchmarks for QBITEL
Tests throughput, latency, and resource usage of critical components
"""

import pytest
import time
import secrets
import asyncio
from typing import List, Dict
import statistics

from ai_engine.cloud_native.service_mesh.istio.qkd_certificate_manager import (
    QuantumCertificateManager,
    CertificateAlgorithm
)
from ai_engine.cloud_native.service_mesh.envoy.traffic_encryption import (
    TrafficEncryptionManager
)
from ai_engine.cloud_native.container_security.signing.dilithium_signer import (
    DilithiumSigner
)
from ai_engine.cloud_native.event_streaming.kafka.secure_producer import (
    SecureKafkaProducer
)


@pytest.mark.performance
@pytest.mark.benchmark
class TestCryptographyPerformance:
    """Benchmark cryptographic operations"""

    def test_kyber_1024_key_generation_performance(self, benchmark):
        """Benchmark Kyber-1024 key pair generation"""
        cert_manager = QuantumCertificateManager()

        result = benchmark(
            cert_manager._generate_key_pair,
            CertificateAlgorithm.KYBER_1024
        )

        # Verify result
        private_key, public_key = result
        assert len(private_key) == 3168
        assert len(public_key) == 1568

        # Performance expectations: < 10ms on modern hardware
        stats = benchmark.stats
        assert stats['mean'] < 0.01, "Key generation should be < 10ms"

    def test_kyber_768_key_generation_performance(self, benchmark):
        """Benchmark Kyber-768 key pair generation"""
        cert_manager = QuantumCertificateManager()

        result = benchmark(
            cert_manager._generate_key_pair,
            CertificateAlgorithm.KYBER_768
        )

        private_key, public_key = result
        assert len(private_key) == 2400
        assert len(public_key) == 1184

    def test_dilithium_5_signature_generation_performance(self, benchmark):
        """Benchmark Dilithium-5 signature generation"""
        signer = DilithiumSigner(security_level=5)
        data = b"Test data for signature performance benchmark" * 10

        result = benchmark(signer.sign, data)

        assert result is not None
        assert len(result) > 0

    def test_dilithium_5_verification_performance(self, benchmark):
        """Benchmark Dilithium-5 signature verification"""
        signer = DilithiumSigner(security_level=5)
        data = b"Test data for verification performance benchmark" * 10
        signature = signer.sign(data)

        result = benchmark(signer.verify, data, signature)

        assert result is True

    def test_aes_256_gcm_encryption_performance(self, benchmark):
        """Benchmark AES-256-GCM encryption (1KB blocks)"""
        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)
        key = secrets.token_bytes(32)
        data = secrets.token_bytes(1024)  # 1KB

        result = benchmark(encryption_manager._encrypt_data, data, key)

        ciphertext, tag, nonce = result
        assert len(tag) == 16
        assert len(nonce) == 12

        # Performance expectation: < 1ms for 1KB
        stats = benchmark.stats
        assert stats['mean'] < 0.001, "AES-GCM encryption should be < 1ms for 1KB"

    def test_aes_256_gcm_decryption_performance(self, benchmark):
        """Benchmark AES-256-GCM decryption (1KB blocks)"""
        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)
        key = secrets.token_bytes(32)
        data = secrets.token_bytes(1024)
        ciphertext, tag, nonce = encryption_manager._encrypt_data(data, key)

        result = benchmark(
            encryption_manager._decrypt_data,
            ciphertext, tag, nonce, key
        )

        assert result == data

    def test_large_data_encryption_performance(self, benchmark):
        """Benchmark encryption of 1MB data blocks"""
        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)
        key = secrets.token_bytes(32)
        data = secrets.token_bytes(1024 * 1024)  # 1MB

        result = benchmark(encryption_manager._encrypt_data, data, key)

        ciphertext, tag, nonce = result
        assert len(ciphertext) == len(data)

        # Performance expectation: < 100ms for 1MB
        stats = benchmark.stats
        assert stats['mean'] < 0.1, "Large data encryption should be < 100ms"


@pytest.mark.performance
@pytest.mark.benchmark
class TestThroughputBenchmarks:
    """Benchmark throughput of various components"""

    def test_encryption_throughput(self):
        """Measure encryption throughput (MB/s)"""
        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)
        key = secrets.token_bytes(32)

        # Test with varying data sizes
        data_sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB
        results = {}

        for size in data_sizes:
            data = secrets.token_bytes(size)

            # Measure time for 100 iterations
            iterations = 100
            start_time = time.perf_counter()

            for _ in range(iterations):
                encryption_manager._encrypt_data(data, key)

            elapsed_time = time.perf_counter() - start_time

            # Calculate throughput
            total_bytes = size * iterations
            throughput_mbps = (total_bytes / elapsed_time) / (1024 * 1024)

            results[size] = throughput_mbps

        # Log results
        for size, throughput in results.items():
            print(f"Throughput for {size} bytes: {throughput:.2f} MB/s")

        # Performance expectation: > 100 MB/s for 1MB blocks
        assert results[1048576] > 100, "Should achieve > 100 MB/s for 1MB blocks"

    def test_signature_throughput(self):
        """Measure signature generation throughput (signatures/second)"""
        signer = DilithiumSigner(security_level=5)
        data = b"Test data for throughput measurement" * 10

        # Measure signatures per second
        iterations = 50
        start_time = time.perf_counter()

        for _ in range(iterations):
            signer.sign(data)

        elapsed_time = time.perf_counter() - start_time
        throughput = iterations / elapsed_time

        print(f"Signature throughput: {throughput:.2f} signatures/second")

        # Performance expectation: > 100 signatures/second
        assert throughput > 100, "Should achieve > 100 signatures/second"

    def test_certificate_generation_throughput(self):
        """Measure certificate generation throughput"""
        cert_manager = QuantumCertificateManager()

        iterations = 20
        start_time = time.perf_counter()

        for i in range(iterations):
            cert_manager.generate_certificate(
                f"service-{i}",
                "qbitel-service-mesh",
                [f"service-{i}.qbitel-service-mesh.svc.cluster.local"]
            )

        elapsed_time = time.perf_counter() - start_time
        throughput = iterations / elapsed_time

        print(f"Certificate generation: {throughput:.2f} certs/second")

        # Performance expectation: > 10 certificates/second
        assert throughput > 10, "Should achieve > 10 certificates/second"


@pytest.mark.performance
@pytest.mark.benchmark
class TestLatencyBenchmarks:
    """Benchmark latency of critical operations"""

    def test_encryption_latency_distribution(self):
        """Measure encryption latency distribution (P50, P95, P99)"""
        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)
        key = secrets.token_bytes(32)
        data = secrets.token_bytes(1024)  # 1KB

        # Collect latency samples
        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            encryption_manager._encrypt_data(data, key)
            latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms

        # Calculate percentiles
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile

        print(f"Encryption Latency - P50: {p50:.3f}ms, P95: {p95:.3f}ms, P99: {p99:.3f}ms")

        # Performance expectations
        assert p50 < 1.0, "P50 latency should be < 1ms"
        assert p95 < 2.0, "P95 latency should be < 2ms"
        assert p99 < 5.0, "P99 latency should be < 5ms"

    def test_signature_latency_distribution(self):
        """Measure signature generation latency distribution"""
        signer = DilithiumSigner(security_level=5)
        data = b"Test data for latency measurement" * 10

        latencies = []
        for _ in range(500):
            start = time.perf_counter()
            signer.sign(data)
            latencies.append((time.perf_counter() - start) * 1000)

        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]
        p99 = statistics.quantiles(latencies, n=100)[98]

        print(f"Signature Latency - P50: {p50:.3f}ms, P95: {p95:.3f}ms, P99: {p99:.3f}ms")

        # Performance expectations (signatures are heavier than symmetric crypto)
        assert p50 < 10.0, "P50 signature latency should be < 10ms"
        assert p99 < 50.0, "P99 signature latency should be < 50ms"


@pytest.mark.performance
@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Test performance at scale"""

    def test_concurrent_encryption_scalability(self):
        """Test encryption performance under concurrent load"""
        import concurrent.futures

        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)

        def encrypt_task(task_id: int) -> float:
            key = secrets.token_bytes(32)
            data = secrets.token_bytes(1024)

            start = time.perf_counter()
            for _ in range(100):
                encryption_manager._encrypt_data(data, key)
            return time.perf_counter() - start

        # Test with increasing concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        results = {}

        for concurrency in concurrency_levels:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                start_time = time.perf_counter()
                futures = [executor.submit(encrypt_task, i) for i in range(concurrency)]
                concurrent.futures.wait(futures)
                total_time = time.perf_counter() - start_time

                total_operations = concurrency * 100
                throughput = total_operations / total_time

                results[concurrency] = throughput

        for concurrency, throughput in results.items():
            print(f"Concurrency {concurrency}: {throughput:.2f} ops/sec")

        # Throughput should scale reasonably with concurrency
        assert results[10] > results[1] * 5, "Should scale with concurrency"

    def test_memory_efficiency(self):
        """Test memory usage during high-volume operations"""
        import tracemalloc

        tracemalloc.start()

        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)
        key = secrets.token_bytes(32)

        # Perform many encryption operations
        for _ in range(10000):
            data = secrets.token_bytes(1024)
            encryption_manager._encrypt_data(data, key)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        print(f"Peak memory usage: {peak_mb:.2f} MB")

        # Should not use excessive memory (< 100MB for 10k operations)
        assert peak_mb < 100, "Memory usage should be < 100MB"


@pytest.mark.performance
@pytest.mark.benchmark
class TestKafkaProducerPerformance:
    """Benchmark Kafka producer performance"""

    @pytest.mark.skip(reason="Requires running Kafka cluster")
    def test_kafka_producer_throughput(self):
        """Measure Kafka producer throughput with encryption"""
        producer = SecureKafkaProducer(
            bootstrap_servers="localhost:9092",
            enable_encryption=True
        )

        # Send 10,000 messages
        messages_count = 10000
        message_size = 1024  # 1KB

        start_time = time.perf_counter()

        for i in range(messages_count):
            message = secrets.token_bytes(message_size)
            producer.send_encrypted_message("test-topic", message)

        elapsed_time = time.perf_counter() - start_time

        throughput_msgs = messages_count / elapsed_time
        throughput_mb = (messages_count * message_size) / elapsed_time / (1024 * 1024)

        print(f"Kafka throughput: {throughput_msgs:.2f} msg/s, {throughput_mb:.2f} MB/s")

        # Performance expectation: > 10,000 msg/s
        assert throughput_msgs > 10000, "Should achieve > 10k msg/s"


def run_all_benchmarks():
    """Run all performance benchmarks and generate report"""
    pytest.main([
        __file__,
        "-v",
        "--benchmark-only",
        "--benchmark-min-rounds=10",
        "--benchmark-warmup=on",
        "--benchmark-json=benchmark_results.json"
    ])


if __name__ == "__main__":
    run_all_benchmarks()
