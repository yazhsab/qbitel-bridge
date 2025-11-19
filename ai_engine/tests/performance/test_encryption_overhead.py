"""
Performance tests for encryption overhead analysis.
Compares quantum-safe encryption with classical encryption methods.
"""

import pytest
import time
import statistics
from typing import List, Dict, Tuple
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os


class TestEncryptionOverhead:
    """Performance overhead analysis for quantum-safe vs classical encryption."""

    ITERATIONS = 100
    MESSAGE_SIZES = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB

    @pytest.fixture
    def test_data(self):
        """Generate test data of various sizes."""
        return {
            size: os.urandom(size) for size in self.MESSAGE_SIZES
        }

    def measure_aes_gcm_encryption(self, data: bytes, iterations: int) -> Tuple[List[float], List[float]]:
        """Measure AES-GCM encryption and decryption performance."""
        key = os.urandom(32)  # 256-bit key
        encryption_timings = []
        decryption_timings = []

        for _ in range(iterations):
            # Encryption
            nonce = os.urandom(12)
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce),
                backend=default_backend()
            )

            start = time.perf_counter()
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            tag = encryptor.tag
            end = time.perf_counter()
            encryption_timings.append((end - start) * 1000)

            # Decryption
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce, tag),
                backend=default_backend()
            )

            start = time.perf_counter()
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            end = time.perf_counter()
            decryption_timings.append((end - start) * 1000)

        return encryption_timings, decryption_timings

    def measure_chacha20_poly1305_encryption(self, data: bytes, iterations: int) -> Tuple[List[float], List[float]]:
        """Measure ChaCha20-Poly1305 encryption and decryption performance."""
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        key = ChaCha20Poly1305.generate_key()
        chacha = ChaCha20Poly1305(key)
        encryption_timings = []
        decryption_timings = []

        for _ in range(iterations):
            nonce = os.urandom(12)

            # Encryption
            start = time.perf_counter()
            ciphertext = chacha.encrypt(nonce, data, None)
            end = time.perf_counter()
            encryption_timings.append((end - start) * 1000)

            # Decryption
            start = time.perf_counter()
            plaintext = chacha.decrypt(nonce, ciphertext, None)
            end = time.perf_counter()
            decryption_timings.append((end - start) * 1000)

        return encryption_timings, decryption_timings

    @pytest.mark.parametrize("message_size", MESSAGE_SIZES)
    def test_aes_gcm_baseline_performance(self, message_size: int, test_data: Dict):
        """Establish baseline AES-GCM performance metrics."""
        data = test_data[message_size]
        enc_timings, dec_timings = self.measure_aes_gcm_encryption(data, self.ITERATIONS)

        avg_enc = statistics.mean(enc_timings)
        avg_dec = statistics.mean(dec_timings)
        throughput_enc = (message_size / (avg_enc / 1000)) / (1024 * 1024)  # MB/s
        throughput_dec = (message_size / (avg_dec / 1000)) / (1024 * 1024)  # MB/s

        print(f"\nAES-GCM Performance ({message_size / 1024:.0f} KB):")
        print(f"  Encryption: {avg_enc:.3f} ms ({throughput_enc:.2f} MB/s)")
        print(f"  Decryption: {avg_dec:.3f} ms ({throughput_dec:.2f} MB/s)")

        # Store baseline for comparison
        assert avg_enc > 0, "Invalid encryption timing"
        assert avg_dec > 0, "Invalid decryption timing"

    @pytest.mark.parametrize("message_size", MESSAGE_SIZES)
    def test_chacha20_baseline_performance(self, message_size: int, test_data: Dict):
        """Establish baseline ChaCha20-Poly1305 performance metrics."""
        data = test_data[message_size]
        enc_timings, dec_timings = self.measure_chacha20_poly1305_encryption(data, self.ITERATIONS)

        avg_enc = statistics.mean(enc_timings)
        avg_dec = statistics.mean(dec_timings)
        throughput_enc = (message_size / (avg_enc / 1000)) / (1024 * 1024)  # MB/s
        throughput_dec = (message_size / (avg_dec / 1000)) / (1024 * 1024)  # MB/s

        print(f"\nChaCha20-Poly1305 Performance ({message_size / 1024:.0f} KB):")
        print(f"  Encryption: {avg_enc:.3f} ms ({throughput_enc:.2f} MB/s)")
        print(f"  Decryption: {avg_dec:.3f} ms ({throughput_dec:.2f} MB/s)")

        assert avg_enc > 0, "Invalid encryption timing"
        assert avg_dec > 0, "Invalid decryption timing"

    def test_encryption_overhead_comparison(self, test_data: Dict):
        """Compare encryption overhead across different algorithms and message sizes."""
        results = {}

        for size in self.MESSAGE_SIZES:
            data = test_data[size]

            # Measure AES-GCM
            aes_enc, aes_dec = self.measure_aes_gcm_encryption(data, 50)

            # Measure ChaCha20
            chacha_enc, chacha_dec = self.measure_chacha20_poly1305_encryption(data, 50)

            results[size] = {
                'aes_gcm': {
                    'encryption': statistics.mean(aes_enc),
                    'decryption': statistics.mean(aes_dec)
                },
                'chacha20': {
                    'encryption': statistics.mean(chacha_enc),
                    'decryption': statistics.mean(chacha_dec)
                }
            }

        # Print comparison table
        print("\nEncryption Overhead Comparison:")
        print("Size\t\tAlgorithm\tEncryption\tDecryption")
        print("-" * 60)

        for size, metrics in results.items():
            size_kb = size / 1024
            for algo, timings in metrics.items():
                print(f"{size_kb:.0f} KB\t\t{algo}\t\t{timings['encryption']:.3f} ms\t{timings['decryption']:.3f} ms")

        # Verify overhead is acceptable
        for size, metrics in results.items():
            aes_enc = metrics['aes_gcm']['encryption']
            chacha_enc = metrics['chacha20']['encryption']

            # Overhead should not exceed 50% difference
            overhead_ratio = max(aes_enc, chacha_enc) / min(aes_enc, chacha_enc)
            assert overhead_ratio < 1.5, f"Encryption overhead too high at {size} bytes: {overhead_ratio:.2f}x"

    def test_quantum_key_exchange_overhead(self):
        """Test overhead of quantum-safe key exchange vs classical ECDH."""
        from oqs import KeyEncapsulation
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        iterations = 50

        # Classical ECDH
        ecdh_timings = []
        for _ in range(iterations):
            start = time.perf_counter()

            # Generate key pairs
            private_key1 = ec.generate_private_key(ec.SECP256R1(), default_backend())
            private_key2 = ec.generate_private_key(ec.SECP256R1(), default_backend())

            # Perform key exchange
            shared_key1 = private_key1.exchange(ec.ECDH(), private_key2.public_key())
            shared_key2 = private_key2.exchange(ec.ECDH(), private_key1.public_key())

            # Derive key
            kdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'', backend=default_backend())
            key = kdf.derive(shared_key1)

            end = time.perf_counter()
            ecdh_timings.append((end - start) * 1000)

        # Quantum-safe Kyber
        kyber_timings = []
        for _ in range(iterations):
            start = time.perf_counter()

            kem = KeyEncapsulation('Kyber768')
            public_key = kem.generate_keypair()
            ciphertext, shared_secret1 = kem.encap_secret(public_key)
            shared_secret2 = kem.decap_secret(ciphertext)

            end = time.perf_counter()
            kyber_timings.append((end - start) * 1000)

        avg_ecdh = statistics.mean(ecdh_timings)
        avg_kyber = statistics.mean(kyber_timings)
        overhead = ((avg_kyber - avg_ecdh) / avg_ecdh) * 100

        print("\nQuantum Key Exchange Overhead:")
        print(f"  Classical ECDH: {avg_ecdh:.3f} ms")
        print(f"  Kyber768: {avg_kyber:.3f} ms")
        print(f"  Overhead: {overhead:.1f}%")

        # Kyber overhead should be reasonable (typically 2-3x slower)
        assert overhead < 500, f"Quantum key exchange overhead too high: {overhead:.1f}%"

    def test_signature_verification_overhead(self):
        """Test overhead of quantum-safe signatures vs classical ECDSA."""
        from oqs import Signature
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes

        message = b"Test message for signature verification overhead analysis"
        iterations = 50

        # Classical ECDSA
        ecdsa_sign_timings = []
        ecdsa_verify_timings = []

        for _ in range(iterations):
            private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())

            start = time.perf_counter()
            signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
            end = time.perf_counter()
            ecdsa_sign_timings.append((end - start) * 1000)

            public_key = private_key.public_key()
            start = time.perf_counter()
            public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
            end = time.perf_counter()
            ecdsa_verify_timings.append((end - start) * 1000)

        # Quantum-safe Dilithium
        dilithium_sign_timings = []
        dilithium_verify_timings = []

        for _ in range(iterations):
            sig = Signature('Dilithium3')
            public_key = sig.generate_keypair()

            start = time.perf_counter()
            signature = sig.sign(message)
            end = time.perf_counter()
            dilithium_sign_timings.append((end - start) * 1000)

            start = time.perf_counter()
            sig.verify(message, signature, public_key)
            end = time.perf_counter()
            dilithium_verify_timings.append((end - start) * 1000)

        avg_ecdsa_sign = statistics.mean(ecdsa_sign_timings)
        avg_ecdsa_verify = statistics.mean(ecdsa_verify_timings)
        avg_dilithium_sign = statistics.mean(dilithium_sign_timings)
        avg_dilithium_verify = statistics.mean(dilithium_verify_timings)

        sign_overhead = ((avg_dilithium_sign - avg_ecdsa_sign) / avg_ecdsa_sign) * 100
        verify_overhead = ((avg_dilithium_verify - avg_ecdsa_verify) / avg_ecdsa_verify) * 100

        print("\nSignature Verification Overhead:")
        print(f"  ECDSA Signing: {avg_ecdsa_sign:.3f} ms")
        print(f"  Dilithium3 Signing: {avg_dilithium_sign:.3f} ms")
        print(f"  Sign Overhead: {sign_overhead:.1f}%")
        print(f"  ECDSA Verification: {avg_ecdsa_verify:.3f} ms")
        print(f"  Dilithium3 Verification: {avg_dilithium_verify:.3f} ms")
        print(f"  Verify Overhead: {verify_overhead:.1f}%")

        # Overhead should be acceptable for production use
        assert sign_overhead < 1000, f"Signature overhead too high: {sign_overhead:.1f}%"
        assert verify_overhead < 1000, f"Verification overhead too high: {verify_overhead:.1f}%"

    @pytest.mark.benchmark
    def test_cpu_utilization_overhead(self):
        """Test CPU utilization during encryption operations."""
        import psutil

        data = os.urandom(1024 * 1024)  # 1MB

        # Measure CPU before
        cpu_before = psutil.cpu_percent(interval=0.1)

        # Perform intensive encryption
        key = os.urandom(32)
        for _ in range(100):
            nonce = os.urandom(12)
            cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()

        # Measure CPU after
        cpu_after = psutil.cpu_percent(interval=0.1)
        cpu_delta = cpu_after - cpu_before

        print(f"\nCPU Utilization During Encryption:")
        print(f"  Before: {cpu_before:.1f}%")
        print(f"  After: {cpu_after:.1f}%")
        print(f"  Delta: {cpu_delta:.1f}%")

        # CPU utilization should be reasonable
        assert cpu_after < 100, "CPU utilization at maximum"
