"""
Performance tests for Kyber post-quantum cryptographic operations.
Tests key generation, encapsulation, and decapsulation performance.
"""

import pytest
import time
import statistics
from typing import List, Dict
from oqs import KeyEncapsulation


class TestKyberPerformance:
    """Performance benchmarks for Kyber post-quantum key encapsulation."""

    ITERATIONS = 100
    KYBER_VARIANTS = ["Kyber512", "Kyber768", "Kyber1024"]

    @pytest.fixture
    def performance_metrics(self):
        """Initialize metrics collection."""
        return {"key_generation": [], "encapsulation": [], "decapsulation": [], "total_time": []}

    def measure_key_generation(self, algorithm: str, iterations: int) -> List[float]:
        """Measure key generation performance."""
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            kem = KeyEncapsulation(algorithm)
            public_key = kem.generate_keypair()
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # Convert to milliseconds
        return timings

    def measure_encapsulation(self, kem: KeyEncapsulation, public_key: bytes, iterations: int) -> List[float]:
        """Measure encapsulation performance."""
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            ciphertext, shared_secret = kem.encap_secret(public_key)
            end = time.perf_counter()
            timings.append((end - start) * 1000)
        return timings

    def measure_decapsulation(self, kem: KeyEncapsulation, ciphertext: bytes, iterations: int) -> List[float]:
        """Measure decapsulation performance."""
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            shared_secret = kem.decap_secret(ciphertext)
            end = time.perf_counter()
            timings.append((end - start) * 1000)
        return timings

    @pytest.mark.parametrize("algorithm", KYBER_VARIANTS)
    def test_kyber_key_generation_performance(self, algorithm: str, performance_metrics: Dict):
        """Test Kyber key generation performance across variants."""
        timings = self.measure_key_generation(algorithm, self.ITERATIONS)

        avg_time = statistics.mean(timings)
        median_time = statistics.median(timings)
        std_dev = statistics.stdev(timings)
        min_time = min(timings)
        max_time = max(timings)

        print(f"\n{algorithm} Key Generation Performance:")
        print(f"  Average: {avg_time:.3f} ms")
        print(f"  Median: {median_time:.3f} ms")
        print(f"  Std Dev: {std_dev:.3f} ms")
        print(f"  Min: {min_time:.3f} ms")
        print(f"  Max: {max_time:.3f} ms")

        # Performance assertions (adjust thresholds as needed)
        assert avg_time < 5.0, f"{algorithm} key generation too slow: {avg_time:.3f} ms"
        assert std_dev < 2.0, f"{algorithm} key generation inconsistent: {std_dev:.3f} ms"

    @pytest.mark.parametrize("algorithm", KYBER_VARIANTS)
    def test_kyber_encapsulation_performance(self, algorithm: str, performance_metrics: Dict):
        """Test Kyber encapsulation performance."""
        kem = KeyEncapsulation(algorithm)
        public_key = kem.generate_keypair()

        timings = self.measure_encapsulation(kem, public_key, self.ITERATIONS)

        avg_time = statistics.mean(timings)
        median_time = statistics.median(timings)
        std_dev = statistics.stdev(timings)

        print(f"\n{algorithm} Encapsulation Performance:")
        print(f"  Average: {avg_time:.3f} ms")
        print(f"  Median: {median_time:.3f} ms")
        print(f"  Std Dev: {std_dev:.3f} ms")

        # Performance assertions
        assert avg_time < 3.0, f"{algorithm} encapsulation too slow: {avg_time:.3f} ms"
        assert std_dev < 1.5, f"{algorithm} encapsulation inconsistent: {std_dev:.3f} ms"

    @pytest.mark.parametrize("algorithm", KYBER_VARIANTS)
    def test_kyber_decapsulation_performance(self, algorithm: str, performance_metrics: Dict):
        """Test Kyber decapsulation performance."""
        kem = KeyEncapsulation(algorithm)
        public_key = kem.generate_keypair()
        ciphertext, _ = kem.encap_secret(public_key)

        timings = self.measure_decapsulation(kem, ciphertext, self.ITERATIONS)

        avg_time = statistics.mean(timings)
        median_time = statistics.median(timings)
        std_dev = statistics.stdev(timings)

        print(f"\n{algorithm} Decapsulation Performance:")
        print(f"  Average: {avg_time:.3f} ms")
        print(f"  Median: {median_time:.3f} ms")
        print(f"  Std Dev: {std_dev:.3f} ms")

        # Performance assertions
        assert avg_time < 3.0, f"{algorithm} decapsulation too slow: {avg_time:.3f} ms"
        assert std_dev < 1.5, f"{algorithm} decapsulation inconsistent: {std_dev:.3f} ms"

    @pytest.mark.parametrize("algorithm", KYBER_VARIANTS)
    def test_kyber_full_cycle_performance(self, algorithm: str):
        """Test complete Kyber key exchange cycle performance."""
        timings = []

        for _ in range(self.ITERATIONS):
            start = time.perf_counter()

            # Key generation
            kem = KeyEncapsulation(algorithm)
            public_key = kem.generate_keypair()

            # Encapsulation
            ciphertext, shared_secret_sender = kem.encap_secret(public_key)

            # Decapsulation
            shared_secret_receiver = kem.decap_secret(ciphertext)

            end = time.perf_counter()
            timings.append((end - start) * 1000)

            # Verify correctness
            assert shared_secret_sender == shared_secret_receiver

        avg_time = statistics.mean(timings)
        percentile_95 = statistics.quantiles(timings, n=20)[18]  # 95th percentile

        print(f"\n{algorithm} Full Cycle Performance:")
        print(f"  Average: {avg_time:.3f} ms")
        print(f"  95th Percentile: {percentile_95:.3f} ms")

        # Performance assertions for complete cycle
        assert avg_time < 10.0, f"{algorithm} full cycle too slow: {avg_time:.3f} ms"
        assert percentile_95 < 15.0, f"{algorithm} 95th percentile too high: {percentile_95:.3f} ms"

    def test_kyber_throughput(self):
        """Test Kyber operations throughput."""
        algorithm = "Kyber768"  # Standard recommended variant
        kem = KeyEncapsulation(algorithm)

        # Measure key generations per second
        start = time.perf_counter()
        count = 0
        while time.perf_counter() - start < 1.0:
            public_key = kem.generate_keypair()
            count += 1

        keygen_per_sec = count
        print(f"\n{algorithm} Throughput:")
        print(f"  Key Generations/sec: {keygen_per_sec}")

        # Should achieve reasonable throughput
        assert keygen_per_sec > 100, f"Throughput too low: {keygen_per_sec} ops/sec"

    @pytest.mark.benchmark
    def test_kyber_memory_efficiency(self):
        """Test memory usage during Kyber operations."""
        import tracemalloc

        algorithm = "Kyber768"
        tracemalloc.start()

        # Perform operations
        kem = KeyEncapsulation(algorithm)
        public_key = kem.generate_keypair()
        ciphertext, shared_secret = kem.encap_secret(public_key)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nKyber768 Memory Usage:")
        print(f"  Current: {current / 1024:.2f} KB")
        print(f"  Peak: {peak / 1024:.2f} KB")

        # Memory should be reasonable (under 1MB for basic operations)
        assert peak < 1024 * 1024, f"Memory usage too high: {peak / 1024:.2f} KB"
