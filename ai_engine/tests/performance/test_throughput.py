"""
Performance tests for system throughput.
Tests message processing, encryption, and communication throughput.
"""

import pytest
import time
import statistics
import threading
import queue
from typing import List, Dict
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestThroughput:
    """Throughput performance tests for various system components."""

    DURATION = 5.0  # seconds
    MESSAGE_SIZES = [256, 1024, 4096, 16384]  # bytes

    @pytest.fixture
    def test_messages(self):
        """Generate test messages of various sizes."""
        return {size: os.urandom(size) for size in self.MESSAGE_SIZES}

    def measure_encryption_throughput(self, message_size: int, duration: float) -> Dict:
        """Measure encryption throughput over a time period."""
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        key = ChaCha20Poly1305.generate_key()
        cipher = ChaCha20Poly1305(key)
        message = os.urandom(message_size)

        bytes_encrypted = 0
        operations = 0
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration:
            nonce = os.urandom(12)
            ciphertext = cipher.encrypt(nonce, message, None)
            bytes_encrypted += len(message)
            operations += 1

        elapsed = time.perf_counter() - start_time

        return {
            "bytes_encrypted": bytes_encrypted,
            "operations": operations,
            "elapsed": elapsed,
            "throughput_mbps": (bytes_encrypted / elapsed) / (1024 * 1024),
            "ops_per_sec": operations / elapsed,
        }

    def measure_decryption_throughput(self, message_size: int, duration: float) -> Dict:
        """Measure decryption throughput over a time period."""
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        key = ChaCha20Poly1305.generate_key()
        cipher = ChaCha20Poly1305(key)
        message = os.urandom(message_size)
        nonce = os.urandom(12)
        ciphertext = cipher.encrypt(nonce, message, None)

        bytes_decrypted = 0
        operations = 0
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration:
            plaintext = cipher.decrypt(nonce, ciphertext, None)
            bytes_decrypted += len(plaintext)
            operations += 1

        elapsed = time.perf_counter() - start_time

        return {
            "bytes_decrypted": bytes_decrypted,
            "operations": operations,
            "elapsed": elapsed,
            "throughput_mbps": (bytes_decrypted / elapsed) / (1024 * 1024),
            "ops_per_sec": operations / elapsed,
        }

    @pytest.mark.parametrize("message_size", MESSAGE_SIZES)
    def test_encryption_throughput(self, message_size: int):
        """Test encryption throughput for different message sizes."""
        metrics = self.measure_encryption_throughput(message_size, self.DURATION)

        print(f"\nEncryption Throughput ({message_size} bytes):")
        print(f"  Throughput: {metrics['throughput_mbps']:.2f} MB/s")
        print(f"  Operations/sec: {metrics['ops_per_sec']:.0f}")
        print(f"  Total operations: {metrics['operations']}")

        # Minimum acceptable throughput based on message size
        min_throughput = 10.0 if message_size >= 4096 else 5.0
        assert (
            metrics["throughput_mbps"] > min_throughput
        ), f"Encryption throughput too low: {metrics['throughput_mbps']:.2f} MB/s"

    @pytest.mark.parametrize("message_size", MESSAGE_SIZES)
    def test_decryption_throughput(self, message_size: int):
        """Test decryption throughput for different message sizes."""
        metrics = self.measure_decryption_throughput(message_size, self.DURATION)

        print(f"\nDecryption Throughput ({message_size} bytes):")
        print(f"  Throughput: {metrics['throughput_mbps']:.2f} MB/s")
        print(f"  Operations/sec: {metrics['ops_per_sec']:.0f}")
        print(f"  Total operations: {metrics['operations']}")

        min_throughput = 10.0 if message_size >= 4096 else 5.0
        assert (
            metrics["throughput_mbps"] > min_throughput
        ), f"Decryption throughput too low: {metrics['throughput_mbps']:.2f} MB/s"

    def test_concurrent_encryption_throughput(self):
        """Test encryption throughput with concurrent operations."""
        message_size = 1024
        num_threads = 4
        duration = 3.0

        results = []

        def worker():
            return self.measure_encryption_throughput(message_size, duration)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            for future in as_completed(futures):
                results.append(future.result())

        total_throughput = sum(r["throughput_mbps"] for r in results)
        total_ops = sum(r["ops_per_sec"] for r in results)

        print(f"\nConcurrent Encryption Throughput ({num_threads} threads):")
        print(f"  Total Throughput: {total_throughput:.2f} MB/s")
        print(f"  Total Operations/sec: {total_ops:.0f}")
        print(f"  Per-thread Throughput: {total_throughput/num_threads:.2f} MB/s")

        # With 4 threads, should achieve at least 3x single-thread throughput
        single_thread = self.measure_encryption_throughput(message_size, 1.0)
        assert total_throughput > single_thread["throughput_mbps"] * 3, "Concurrent throughput scaling insufficient"

    def test_quantum_signature_throughput(self):
        """Test quantum-safe signature generation throughput."""
        from oqs import Signature

        algorithm = "Dilithium3"
        message = b"Test message for throughput measurement"
        duration = 3.0

        sig = Signature(algorithm)
        public_key = sig.generate_keypair()

        operations = 0
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration:
            signature = sig.sign(message)
            operations += 1

        elapsed = time.perf_counter() - start_time
        signatures_per_sec = operations / elapsed

        print(f"\n{algorithm} Signature Throughput:")
        print(f"  Signatures/sec: {signatures_per_sec:.0f}")
        print(f"  Total signatures: {operations}")

        assert signatures_per_sec > 200, f"Signature throughput too low: {signatures_per_sec:.0f} ops/sec"

    def test_quantum_verification_throughput(self):
        """Test quantum-safe signature verification throughput."""
        from oqs import Signature

        algorithm = "Dilithium3"
        message = b"Test message for throughput measurement"
        duration = 3.0

        sig = Signature(algorithm)
        public_key = sig.generate_keypair()
        signature = sig.sign(message)

        operations = 0
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration:
            is_valid = sig.verify(message, signature, public_key)
            operations += 1

        elapsed = time.perf_counter() - start_time
        verifications_per_sec = operations / elapsed

        print(f"\n{algorithm} Verification Throughput:")
        print(f"  Verifications/sec: {verifications_per_sec:.0f}")
        print(f"  Total verifications: {operations}")

        assert verifications_per_sec > 300, f"Verification throughput too low: {verifications_per_sec:.0f} ops/sec"

    def test_message_queue_throughput(self):
        """Test message queue throughput."""
        message_queue = queue.Queue(maxsize=1000)
        messages_sent = 0
        messages_received = 0
        stop_flag = threading.Event()

        def producer():
            nonlocal messages_sent
            while not stop_flag.is_set():
                try:
                    message_queue.put(os.urandom(256), timeout=0.001)
                    messages_sent += 1
                except queue.Full:
                    pass

        def consumer():
            nonlocal messages_received
            while not stop_flag.is_set():
                try:
                    message = message_queue.get(timeout=0.001)
                    messages_received += 1
                except queue.Empty:
                    pass

        # Start threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        start_time = time.perf_counter()
        producer_thread.start()
        consumer_thread.start()

        # Run for duration
        time.sleep(3.0)
        stop_flag.set()

        producer_thread.join()
        consumer_thread.join()
        elapsed = time.perf_counter() - start_time

        throughput = messages_received / elapsed

        print(f"\nMessage Queue Throughput:")
        print(f"  Messages/sec: {throughput:.0f}")
        print(f"  Total sent: {messages_sent}")
        print(f"  Total received: {messages_received}")
        print(f"  Queue efficiency: {(messages_received/messages_sent)*100:.1f}%")

        assert throughput > 10000, f"Queue throughput too low: {throughput:.0f} msgs/sec"
        assert messages_received / messages_sent > 0.95, "Message loss too high"

    def test_parallel_key_generation_throughput(self):
        """Test throughput of parallel quantum key generation."""
        from oqs import KeyEncapsulation

        algorithm = "Kyber768"
        num_workers = 4
        duration = 3.0

        def generate_keys():
            count = 0
            start = time.perf_counter()
            while time.perf_counter() - start < duration:
                kem = KeyEncapsulation(algorithm)
                public_key = kem.generate_keypair()
                count += 1
            return count

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(generate_keys) for _ in range(num_workers)]
            results = [future.result() for future in as_completed(futures)]

        total_keys = sum(results)
        keys_per_sec = total_keys / duration

        print(f"\nParallel Key Generation Throughput ({num_workers} workers):")
        print(f"  Keys/sec: {keys_per_sec:.0f}")
        print(f"  Total keys generated: {total_keys}")
        print(f"  Per-worker average: {keys_per_sec/num_workers:.0f}")

        assert keys_per_sec > 400, f"Parallel key generation too slow: {keys_per_sec:.0f} keys/sec"

    @pytest.mark.benchmark
    def test_sustained_throughput_stability(self):
        """Test throughput stability over extended period."""
        message_size = 1024
        test_duration = 10.0
        sample_interval = 1.0

        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        key = ChaCha20Poly1305.generate_key()
        cipher = ChaCha20Poly1305(key)
        message = os.urandom(message_size)

        throughput_samples = []
        start_time = time.perf_counter()
        last_sample_time = start_time
        operations_in_interval = 0

        while time.perf_counter() - start_time < test_duration:
            nonce = os.urandom(12)
            ciphertext = cipher.encrypt(nonce, message, None)
            operations_in_interval += 1

            current_time = time.perf_counter()
            if current_time - last_sample_time >= sample_interval:
                interval_duration = current_time - last_sample_time
                ops_per_sec = operations_in_interval / interval_duration
                throughput_mbps = (operations_in_interval * message_size / interval_duration) / (1024 * 1024)
                throughput_samples.append(throughput_mbps)

                operations_in_interval = 0
                last_sample_time = current_time

        avg_throughput = statistics.mean(throughput_samples)
        std_dev = statistics.stdev(throughput_samples)
        coefficient_of_variation = (std_dev / avg_throughput) * 100

        print(f"\nSustained Throughput Stability ({test_duration:.0f}s):")
        print(f"  Average: {avg_throughput:.2f} MB/s")
        print(f"  Std Dev: {std_dev:.2f} MB/s")
        print(f"  CV: {coefficient_of_variation:.1f}%")
        print(f"  Samples: {throughput_samples}")

        # Throughput should be stable (CV < 20%)
        assert coefficient_of_variation < 20, f"Throughput too unstable: CV={coefficient_of_variation:.1f}%"
