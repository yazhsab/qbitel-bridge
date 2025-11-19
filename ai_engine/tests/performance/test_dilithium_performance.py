"""
Performance tests for Dilithium post-quantum digital signature operations.
Tests key generation, signing, and verification performance.
"""

import pytest
import time
import statistics
from typing import List, Dict
from oqs import Signature


class TestDilithiumPerformance:
    """Performance benchmarks for Dilithium post-quantum signatures."""

    ITERATIONS = 100
    DILITHIUM_VARIANTS = ['Dilithium2', 'Dilithium3', 'Dilithium5']
    TEST_MESSAGE = b"Performance test message for Dilithium signature scheme"

    @pytest.fixture
    def performance_metrics(self):
        """Initialize metrics collection."""
        return {
            'key_generation': [],
            'signing': [],
            'verification': [],
            'total_time': []
        }

    def measure_key_generation(self, algorithm: str, iterations: int) -> List[float]:
        """Measure key generation performance."""
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            sig = Signature(algorithm)
            public_key = sig.generate_keypair()
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # Convert to milliseconds
        return timings

    def measure_signing(self, sig: Signature, message: bytes, iterations: int) -> List[float]:
        """Measure signing performance."""
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            signature = sig.sign(message)
            end = time.perf_counter()
            timings.append((end - start) * 1000)
        return timings

    def measure_verification(self, sig: Signature, message: bytes, signature: bytes,
                           public_key: bytes, iterations: int) -> List[float]:
        """Measure verification performance."""
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            is_valid = sig.verify(message, signature, public_key)
            end = time.perf_counter()
            timings.append((end - start) * 1000)
            assert is_valid, "Signature verification failed"
        return timings

    @pytest.mark.parametrize("algorithm", DILITHIUM_VARIANTS)
    def test_dilithium_key_generation_performance(self, algorithm: str, performance_metrics: Dict):
        """Test Dilithium key generation performance across variants."""
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

        # Performance assertions (adjust thresholds based on security level)
        max_acceptable = 10.0 if algorithm == 'Dilithium5' else 7.0
        assert avg_time < max_acceptable, f"{algorithm} key generation too slow: {avg_time:.3f} ms"
        assert std_dev < 3.0, f"{algorithm} key generation inconsistent: {std_dev:.3f} ms"

    @pytest.mark.parametrize("algorithm", DILITHIUM_VARIANTS)
    def test_dilithium_signing_performance(self, algorithm: str, performance_metrics: Dict):
        """Test Dilithium signing performance."""
        sig = Signature(algorithm)
        public_key = sig.generate_keypair()

        timings = self.measure_signing(sig, self.TEST_MESSAGE, self.ITERATIONS)

        avg_time = statistics.mean(timings)
        median_time = statistics.median(timings)
        std_dev = statistics.stdev(timings)

        print(f"\n{algorithm} Signing Performance:")
        print(f"  Average: {avg_time:.3f} ms")
        print(f"  Median: {median_time:.3f} ms")
        print(f"  Std Dev: {std_dev:.3f} ms")

        # Signing should be fast
        max_acceptable = 5.0 if algorithm == 'Dilithium5' else 3.0
        assert avg_time < max_acceptable, f"{algorithm} signing too slow: {avg_time:.3f} ms"
        assert std_dev < 2.0, f"{algorithm} signing inconsistent: {std_dev:.3f} ms"

    @pytest.mark.parametrize("algorithm", DILITHIUM_VARIANTS)
    def test_dilithium_verification_performance(self, algorithm: str, performance_metrics: Dict):
        """Test Dilithium signature verification performance."""
        sig = Signature(algorithm)
        public_key = sig.generate_keypair()
        signature = sig.sign(self.TEST_MESSAGE)

        timings = self.measure_verification(sig, self.TEST_MESSAGE, signature,
                                           public_key, self.ITERATIONS)

        avg_time = statistics.mean(timings)
        median_time = statistics.median(timings)
        std_dev = statistics.stdev(timings)

        print(f"\n{algorithm} Verification Performance:")
        print(f"  Average: {avg_time:.3f} ms")
        print(f"  Median: {median_time:.3f} ms")
        print(f"  Std Dev: {std_dev:.3f} ms")

        # Verification should be very fast
        max_acceptable = 3.0 if algorithm == 'Dilithium5' else 2.0
        assert avg_time < max_acceptable, f"{algorithm} verification too slow: {avg_time:.3f} ms"
        assert std_dev < 1.5, f"{algorithm} verification inconsistent: {std_dev:.3f} ms"

    @pytest.mark.parametrize("algorithm", DILITHIUM_VARIANTS)
    def test_dilithium_full_cycle_performance(self, algorithm: str):
        """Test complete Dilithium sign-verify cycle performance."""
        timings = []

        for _ in range(self.ITERATIONS):
            start = time.perf_counter()

            # Key generation
            sig = Signature(algorithm)
            public_key = sig.generate_keypair()

            # Signing
            signature = sig.sign(self.TEST_MESSAGE)

            # Verification
            is_valid = sig.verify(self.TEST_MESSAGE, signature, public_key)

            end = time.perf_counter()
            timings.append((end - start) * 1000)

            # Verify correctness
            assert is_valid, "Signature verification failed"

        avg_time = statistics.mean(timings)
        percentile_95 = statistics.quantiles(timings, n=20)[18]  # 95th percentile

        print(f"\n{algorithm} Full Cycle Performance:")
        print(f"  Average: {avg_time:.3f} ms")
        print(f"  95th Percentile: {percentile_95:.3f} ms")

        # Performance assertions for complete cycle
        max_acceptable = 15.0 if algorithm == 'Dilithium5' else 12.0
        assert avg_time < max_acceptable, f"{algorithm} full cycle too slow: {avg_time:.3f} ms"

    def test_dilithium_signing_throughput(self):
        """Test Dilithium signing throughput."""
        algorithm = 'Dilithium3'  # Standard recommended variant
        sig = Signature(algorithm)
        public_key = sig.generate_keypair()

        # Measure signatures per second
        start = time.perf_counter()
        count = 0
        while time.perf_counter() - start < 1.0:
            signature = sig.sign(self.TEST_MESSAGE)
            count += 1

        signs_per_sec = count
        print(f"\n{algorithm} Signing Throughput: {signs_per_sec} signatures/sec")

        # Should achieve reasonable throughput
        assert signs_per_sec > 200, f"Signing throughput too low: {signs_per_sec} ops/sec"

    def test_dilithium_verification_throughput(self):
        """Test Dilithium verification throughput."""
        algorithm = 'Dilithium3'
        sig = Signature(algorithm)
        public_key = sig.generate_keypair()
        signature = sig.sign(self.TEST_MESSAGE)

        # Measure verifications per second
        start = time.perf_counter()
        count = 0
        while time.perf_counter() - start < 1.0:
            is_valid = sig.verify(self.TEST_MESSAGE, signature, public_key)
            count += 1

        verifications_per_sec = count
        print(f"\n{algorithm} Verification Throughput: {verifications_per_sec} verifications/sec")

        # Verification should be fast
        assert verifications_per_sec > 300, f"Verification throughput too low: {verifications_per_sec} ops/sec"

    @pytest.mark.parametrize("message_size", [100, 1024, 10240, 102400])
    def test_dilithium_variable_message_size_performance(self, message_size: int):
        """Test Dilithium performance with different message sizes."""
        algorithm = 'Dilithium3'
        sig = Signature(algorithm)
        public_key = sig.generate_keypair()

        message = b"x" * message_size

        # Measure signing
        signing_timings = []
        for _ in range(50):
            start = time.perf_counter()
            signature = sig.sign(message)
            end = time.perf_counter()
            signing_timings.append((end - start) * 1000)

        avg_signing = statistics.mean(signing_timings)

        # Measure verification
        verification_timings = []
        for _ in range(50):
            start = time.perf_counter()
            is_valid = sig.verify(message, signature, public_key)
            end = time.perf_counter()
            verification_timings.append((end - start) * 1000)

        avg_verification = statistics.mean(verification_timings)

        print(f"\n{algorithm} Performance (Message size: {message_size} bytes):")
        print(f"  Signing: {avg_signing:.3f} ms")
        print(f"  Verification: {avg_verification:.3f} ms")

        # Performance should scale reasonably with message size
        # Note: Dilithium hashes the message, so larger messages take longer
        assert avg_signing < 20.0, f"Signing too slow for {message_size} bytes"
        assert avg_verification < 20.0, f"Verification too slow for {message_size} bytes"

    @pytest.mark.benchmark
    def test_dilithium_memory_efficiency(self):
        """Test memory usage during Dilithium operations."""
        import tracemalloc

        algorithm = 'Dilithium3'
        tracemalloc.start()

        # Perform operations
        sig = Signature(algorithm)
        public_key = sig.generate_keypair()
        signature = sig.sign(self.TEST_MESSAGE)
        is_valid = sig.verify(self.TEST_MESSAGE, signature, public_key)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nDilithium3 Memory Usage:")
        print(f"  Current: {current / 1024:.2f} KB")
        print(f"  Peak: {peak / 1024:.2f} KB")

        # Memory should be reasonable
        assert peak < 2 * 1024 * 1024, f"Memory usage too high: {peak / 1024:.2f} KB"
