"""
Performance tests for system latency.
Tests end-to-end latency, processing delays, and response times.
"""

import pytest
import time
import statistics
from typing import List, Dict
import os
from oqs import KeyEncapsulation, Signature


class TestLatency:
    """Latency performance tests for various system operations."""

    ITERATIONS = 200
    PERCENTILES = [50, 90, 95, 99]

    def calculate_percentiles(self, timings: List[float]) -> Dict[int, float]:
        """Calculate percentile values from timing data."""
        sorted_timings = sorted(timings)
        percentiles = {}

        for p in self.PERCENTILES:
            index = int(len(sorted_timings) * p / 100)
            percentiles[p] = sorted_timings[min(index, len(sorted_timings) - 1)]

        return percentiles

    def measure_operation_latency(self, operation_func, iterations: int) -> List[float]:
        """Measure latency of a given operation."""
        timings = []

        for _ in range(iterations):
            start = time.perf_counter()
            operation_func()
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # Convert to milliseconds

        return timings

    def test_key_encapsulation_latency(self):
        """Test latency of quantum key encapsulation."""
        algorithm = 'Kyber768'

        def key_encapsulation():
            kem = KeyEncapsulation(algorithm)
            public_key = kem.generate_keypair()
            ciphertext, shared_secret = kem.encap_secret(public_key)
            decrypted_secret = kem.decap_secret(ciphertext)

        timings = self.measure_operation_latency(key_encapsulation, self.ITERATIONS)
        percentiles = self.calculate_percentiles(timings)

        avg_latency = statistics.mean(timings)
        min_latency = min(timings)
        max_latency = max(timings)

        print(f"\n{algorithm} Key Encapsulation Latency:")
        print(f"  Average: {avg_latency:.3f} ms")
        print(f"  Min: {min_latency:.3f} ms")
        print(f"  Max: {max_latency:.3f} ms")
        for p, value in percentiles.items():
            print(f"  P{p}: {value:.3f} ms")

        # Latency assertions
        assert avg_latency < 10.0, f"Average latency too high: {avg_latency:.3f} ms"
        assert percentiles[95] < 15.0, f"P95 latency too high: {percentiles[95]:.3f} ms"
        assert percentiles[99] < 20.0, f"P99 latency too high: {percentiles[99]:.3f} ms"

    def test_signature_generation_latency(self):
        """Test latency of quantum signature generation."""
        algorithm = 'Dilithium3'
        message = b"Test message for latency measurement"

        def sign_operation():
            sig = Signature(algorithm)
            public_key = sig.generate_keypair()
            signature = sig.sign(message)

        timings = self.measure_operation_latency(sign_operation, self.ITERATIONS)
        percentiles = self.calculate_percentiles(timings)

        avg_latency = statistics.mean(timings)

        print(f"\n{algorithm} Signature Generation Latency:")
        print(f"  Average: {avg_latency:.3f} ms")
        for p, value in percentiles.items():
            print(f"  P{p}: {value:.3f} ms")

        assert avg_latency < 12.0, f"Signature latency too high: {avg_latency:.3f} ms"
        assert percentiles[99] < 20.0, f"P99 signature latency too high: {percentiles[99]:.3f} ms"

    def test_signature_verification_latency(self):
        """Test latency of quantum signature verification."""
        algorithm = 'Dilithium3'
        message = b"Test message for latency measurement"

        # Pre-generate signature
        sig = Signature(algorithm)
        public_key = sig.generate_keypair()
        signature = sig.sign(message)

        def verify_operation():
            is_valid = sig.verify(message, signature, public_key)
            assert is_valid

        timings = self.measure_operation_latency(verify_operation, self.ITERATIONS)
        percentiles = self.calculate_percentiles(timings)

        avg_latency = statistics.mean(timings)

        print(f"\n{algorithm} Signature Verification Latency:")
        print(f"  Average: {avg_latency:.3f} ms")
        for p, value in percentiles.items():
            print(f"  P{p}: {value:.3f} ms")

        assert avg_latency < 3.0, f"Verification latency too high: {avg_latency:.3f} ms"
        assert percentiles[99] < 5.0, f"P99 verification latency too high: {percentiles[99]:.3f} ms"

    def test_encryption_latency(self):
        """Test latency of symmetric encryption operations."""
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        key = ChaCha20Poly1305.generate_key()
        cipher = ChaCha20Poly1305(key)
        message = os.urandom(1024)

        def encrypt_decrypt():
            nonce = os.urandom(12)
            ciphertext = cipher.encrypt(nonce, message, None)
            plaintext = cipher.decrypt(nonce, ciphertext, None)

        timings = self.measure_operation_latency(encrypt_decrypt, self.ITERATIONS)
        percentiles = self.calculate_percentiles(timings)

        avg_latency = statistics.mean(timings)

        print(f"\nChaCha20-Poly1305 Encryption/Decryption Latency (1KB):")
        print(f"  Average: {avg_latency:.3f} ms")
        for p, value in percentiles.items():
            print(f"  P{p}: {value:.3f} ms")

        assert avg_latency < 1.0, f"Encryption latency too high: {avg_latency:.3f} ms"
        assert percentiles[99] < 2.0, f"P99 encryption latency too high: {percentiles[99]:.3f} ms"

    @pytest.mark.parametrize("message_size", [256, 1024, 4096, 16384])
    def test_encryption_latency_by_size(self, message_size: int):
        """Test encryption latency for different message sizes."""
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        key = ChaCha20Poly1305.generate_key()
        cipher = ChaCha20Poly1305(key)
        message = os.urandom(message_size)

        encryption_timings = []
        decryption_timings = []

        for _ in range(100):
            nonce = os.urandom(12)

            # Measure encryption
            start = time.perf_counter()
            ciphertext = cipher.encrypt(nonce, message, None)
            end = time.perf_counter()
            encryption_timings.append((end - start) * 1000)

            # Measure decryption
            start = time.perf_counter()
            plaintext = cipher.decrypt(nonce, ciphertext, None)
            end = time.perf_counter()
            decryption_timings.append((end - start) * 1000)

        avg_enc = statistics.mean(encryption_timings)
        avg_dec = statistics.mean(decryption_timings)
        p99_enc = self.calculate_percentiles(encryption_timings)[99]
        p99_dec = self.calculate_percentiles(decryption_timings)[99]

        print(f"\nEncryption Latency ({message_size} bytes):")
        print(f"  Encryption avg: {avg_enc:.3f} ms (P99: {p99_enc:.3f} ms)")
        print(f"  Decryption avg: {avg_dec:.3f} ms (P99: {p99_dec:.3f} ms)")

        # Latency should scale reasonably with size
        max_acceptable = (message_size / 1024) * 0.5  # 0.5ms per KB
        assert avg_enc < max_acceptable, f"Encryption latency too high for {message_size} bytes"
        assert avg_dec < max_acceptable, f"Decryption latency too high for {message_size} bytes"

    def test_key_generation_latency(self):
        """Test latency of quantum key generation."""
        algorithm = 'Kyber768'

        def key_generation():
            kem = KeyEncapsulation(algorithm)
            public_key = kem.generate_keypair()

        timings = self.measure_operation_latency(key_generation, self.ITERATIONS)
        percentiles = self.calculate_percentiles(timings)

        avg_latency = statistics.mean(timings)

        print(f"\n{algorithm} Key Generation Latency:")
        print(f"  Average: {avg_latency:.3f} ms")
        for p, value in percentiles.items():
            print(f"  P{p}: {value:.3f} ms")

        assert avg_latency < 5.0, f"Key generation latency too high: {avg_latency:.3f} ms"
        assert percentiles[99] < 10.0, f"P99 key generation latency too high: {percentiles[99]:.3f} ms"

    def test_end_to_end_secure_message_latency(self):
        """Test end-to-end latency for secure message exchange."""
        message = b"Secure message for end-to-end latency test"

        def secure_message_exchange():
            # Key exchange
            kem = KeyEncapsulation('Kyber768')
            public_key = kem.generate_keypair()
            ciphertext, shared_secret = kem.encap_secret(public_key)

            # Derive encryption key from shared secret
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            from cryptography.hazmat.backends import default_backend

            kdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'',
                backend=default_backend()
            )
            encryption_key = kdf.derive(shared_secret)

            # Encrypt message
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
            cipher = ChaCha20Poly1305(encryption_key)
            nonce = os.urandom(12)
            encrypted_message = cipher.encrypt(nonce, message, None)

            # Sign the encrypted message
            sig = Signature('Dilithium3')
            sig_public_key = sig.generate_keypair()
            signature = sig.sign(encrypted_message)

            # Verify signature
            is_valid = sig.verify(encrypted_message, signature, sig_public_key)

            # Decrypt message
            receiver_secret = kem.decap_secret(ciphertext)
            receiver_key = kdf.derive(receiver_secret)
            cipher2 = ChaCha20Poly1305(receiver_key)
            decrypted_message = cipher2.decrypt(nonce, encrypted_message, None)

            assert is_valid
            assert decrypted_message == message

        timings = self.measure_operation_latency(secure_message_exchange, 50)
        percentiles = self.calculate_percentiles(timings)

        avg_latency = statistics.mean(timings)

        print(f"\nEnd-to-End Secure Message Latency:")
        print(f"  Average: {avg_latency:.3f} ms")
        for p, value in percentiles.items():
            print(f"  P{p}: {value:.3f} ms")

        # Complete secure exchange should complete within acceptable time
        assert avg_latency < 30.0, f"E2E latency too high: {avg_latency:.3f} ms"
        assert percentiles[95] < 40.0, f"P95 E2E latency too high: {percentiles[95]:.3f} ms"

    def test_latency_jitter(self):
        """Test latency jitter (variability) for critical operations."""
        algorithm = 'Kyber768'

        def operation():
            kem = KeyEncapsulation(algorithm)
            public_key = kem.generate_keypair()
            ciphertext, shared_secret = kem.encap_secret(public_key)

        timings = self.measure_operation_latency(operation, 500)

        avg_latency = statistics.mean(timings)
        std_dev = statistics.stdev(timings)
        jitter = (std_dev / avg_latency) * 100  # Coefficient of variation

        print(f"\nLatency Jitter Analysis ({algorithm}):")
        print(f"  Average: {avg_latency:.3f} ms")
        print(f"  Std Dev: {std_dev:.3f} ms")
        print(f"  Jitter (CV): {jitter:.1f}%")

        # Jitter should be reasonable (< 30%)
        assert jitter < 30, f"Latency jitter too high: {jitter:.1f}%"

    def test_cold_start_latency(self):
        """Test cold start latency (first operation)."""
        from oqs import KeyEncapsulation

        # Measure first operation (cold start)
        start = time.perf_counter()
        kem = KeyEncapsulation('Kyber768')
        public_key = kem.generate_keypair()
        cold_start_time = (time.perf_counter() - start) * 1000

        # Measure warm operations
        warm_timings = []
        for _ in range(10):
            start = time.perf_counter()
            kem = KeyEncapsulation('Kyber768')
            public_key = kem.generate_keypair()
            warm_timings.append((time.perf_counter() - start) * 1000)

        avg_warm = statistics.mean(warm_timings)
        overhead = ((cold_start_time - avg_warm) / avg_warm) * 100

        print(f"\nCold Start Latency:")
        print(f"  Cold start: {cold_start_time:.3f} ms")
        print(f"  Warm average: {avg_warm:.3f} ms")
        print(f"  Overhead: {overhead:.1f}%")

        # Cold start overhead should be reasonable
        assert cold_start_time < 50.0, f"Cold start too slow: {cold_start_time:.3f} ms"

    @pytest.mark.benchmark
    def test_tail_latency_analysis(self):
        """Comprehensive tail latency analysis."""
        algorithm = 'Kyber768'

        def operation():
            kem = KeyEncapsulation(algorithm)
            public_key = kem.generate_keypair()
            ciphertext, shared_secret = kem.encap_secret(public_key)

        # Large sample size for accurate tail latency
        timings = self.measure_operation_latency(operation, 1000)

        sorted_timings = sorted(timings)
        tail_percentiles = {
            'P50': sorted_timings[499],
            'P90': sorted_timings[899],
            'P95': sorted_timings[949],
            'P99': sorted_timings[989],
            'P99.9': sorted_timings[998],
            'Max': max(timings)
        }

        print(f"\nTail Latency Analysis ({algorithm}, n=1000):")
        for label, value in tail_percentiles.items():
            print(f"  {label}: {value:.3f} ms")

        # Tail latency assertions
        assert tail_percentiles['P99'] < 15.0, f"P99 too high: {tail_percentiles['P99']:.3f} ms"
        assert tail_percentiles['P99.9'] < 25.0, f"P99.9 too high: {tail_percentiles['P99.9']:.3f} ms"
        assert tail_percentiles['Max'] < 50.0, f"Max latency too high: {tail_percentiles['Max']:.3f} ms"
