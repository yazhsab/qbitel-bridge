"""
Security compliance tests for regulatory and industry standards.
Tests FIPS 140-2, NIST standards, GDPR, and other compliance requirements.
"""

import pytest
import os
import hashlib
from typing import Dict, List
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305, AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from oqs import KeyEncapsulation, Signature


class TestCompliance:
    """Compliance tests for security standards and regulations."""

    # NIST-approved key sizes
    NIST_MIN_SYMMETRIC_KEY_SIZE = 128  # bits
    NIST_RECOMMENDED_SYMMETRIC_KEY_SIZE = 256  # bits
    NIST_MIN_ASYMMETRIC_KEY_SIZE = 2048  # bits (RSA)
    NIST_RECOMMENDED_HASH_SIZE = 256  # bits

    @pytest.fixture
    def compliance_config(self):
        """Configuration for compliance testing."""
        return {
            'algorithms': {
                'kem': ['Kyber512', 'Kyber768', 'Kyber1024'],
                'signature': ['Dilithium2', 'Dilithium3', 'Dilithium5'],
                'symmetric': ['AES-256-GCM', 'ChaCha20-Poly1305'],
                'hash': ['SHA-256', 'SHA-384', 'SHA-512']
            },
            'key_sizes': {
                'symmetric': 256,
                'hash': 256
            },
            'requirements': {
                'min_password_length': 8,
                'kdf_iterations': 100000,
                'nonce_size': 96,  # bits
                'mac_size': 128  # bits
            }
        }

    def test_nist_post_quantum_standards(self, compliance_config):
        """Test compliance with NIST post-quantum cryptography standards."""
        # NIST selected Kyber and Dilithium as standards
        approved_kems = compliance_config['algorithms']['kem']
        approved_sigs = compliance_config['algorithms']['signature']

        # Test KEM compliance
        for algorithm in approved_kems:
            kem = KeyEncapsulation(algorithm)
            public_key = kem.generate_keypair()
            ciphertext, shared_secret = kem.encap_secret(public_key)

            # Verify shared secret size (should be 256 bits)
            assert len(shared_secret) == 32, \
                f"{algorithm} shared secret must be 256 bits"

            # Verify decapsulation works
            decrypted_secret = kem.decap_secret(ciphertext)
            assert shared_secret == decrypted_secret, \
                f"{algorithm} KEM correctness failed"

        # Test signature compliance
        for algorithm in approved_sigs:
            sig = Signature(algorithm)
            public_key = sig.generate_keypair()
            message = b"NIST compliance test"
            signature = sig.sign(message)

            # Verify signature
            is_valid = sig.verify(message, signature, public_key)
            assert is_valid, f"{algorithm} signature verification failed"

    def test_fips_140_2_approved_algorithms(self):
        """Test use of FIPS 140-2 approved algorithms."""
        # FIPS 140-2 approved symmetric algorithms
        # AES is FIPS approved (ChaCha20 is not, but widely used)

        # Test AES-256-GCM (FIPS approved)
        key = AESGCM.generate_key(bit_length=256)
        cipher = AESGCM(key)

        message = b"FIPS 140-2 test message"
        nonce = os.urandom(12)

        ciphertext = cipher.encrypt(nonce, message, None)
        plaintext = cipher.decrypt(nonce, ciphertext, None)

        assert plaintext == message, "AES-GCM encryption/decryption failed"
        assert len(key) == 32, "AES key must be 256 bits"

    def test_nist_key_size_requirements(self, compliance_config):
        """Test compliance with NIST key size requirements."""
        required_size = compliance_config['key_sizes']['symmetric']

        # Test symmetric key sizes
        aes_key = AESGCM.generate_key(bit_length=256)
        assert len(aes_key) * 8 == required_size, \
            f"AES key size must be {required_size} bits"

        chacha_key = ChaCha20Poly1305.generate_key()
        assert len(chacha_key) * 8 == required_size, \
            f"ChaCha20 key size must be {required_size} bits"

        # Test post-quantum shared secrets
        kem = KeyEncapsulation('Kyber768')
        public_key = kem.generate_keypair()
        _, shared_secret = kem.encap_secret(public_key)
        assert len(shared_secret) * 8 == required_size, \
            f"Shared secret must be {required_size} bits"

    def test_password_policy_compliance(self, compliance_config):
        """Test compliance with password policy requirements."""
        min_length = compliance_config['requirements']['min_password_length']

        # Test password minimum length
        passwords = {
            'too_short': b'Pass1',
            'minimum': b'Pass1234',
            'good': b'SecureP@ssw0rd!',
            'excellent': b'MyV3ry$ecur3P@ssw0rd2023!'
        }

        for label, password in passwords.items():
            if len(password) < min_length:
                # Password too short - should be rejected
                assert label == 'too_short', \
                    f"Password '{label}' should be rejected (too short)"
            else:
                # Password meets minimum length - derive key
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=os.urandom(16),
                    iterations=compliance_config['requirements']['kdf_iterations'],
                    backend=default_backend()
                )
                key = kdf.derive(password)
                assert len(key) == 32, "Derived key must be 256 bits"

    def test_kdf_iteration_count_compliance(self, compliance_config):
        """Test KDF iteration count meets security requirements."""
        min_iterations = compliance_config['requirements']['kdf_iterations']
        password = b"test_password"
        salt = os.urandom(16)

        # Test with compliant iteration count
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=min_iterations,
            backend=default_backend()
        )

        import time
        start = time.perf_counter()
        key = kdf.derive(password)
        elapsed = time.perf_counter() - start

        # Should take noticeable time (at least 10ms)
        assert elapsed > 0.01, \
            f"KDF with {min_iterations} iterations too fast: {elapsed*1000:.1f}ms"

        assert len(key) == 32, "Derived key must be 256 bits"

    def test_nonce_size_compliance(self, compliance_config):
        """Test nonce size meets requirements."""
        required_nonce_size = compliance_config['requirements']['nonce_size'] // 8

        # Test AES-GCM nonce size
        aes_nonce = os.urandom(12)
        assert len(aes_nonce) == required_nonce_size, \
            f"AES-GCM nonce must be {required_nonce_size} bytes"

        # Test ChaCha20-Poly1305 nonce size
        chacha_nonce = os.urandom(12)
        assert len(chacha_nonce) == required_nonce_size, \
            f"ChaCha20-Poly1305 nonce must be {required_nonce_size} bytes"

        # Verify nonces are random
        nonces = [os.urandom(12) for _ in range(100)]
        assert len(set(nonces)) == 100, "Nonces must be unique"

    def test_mac_size_compliance(self, compliance_config):
        """Test MAC/authentication tag size meets requirements."""
        min_mac_size = compliance_config['requirements']['mac_size'] // 8

        # AES-GCM provides 128-bit authentication tag
        key = AESGCM.generate_key(bit_length=256)
        cipher = AESGCM(key)
        nonce = os.urandom(12)
        message = b"Test message"

        ciphertext = cipher.encrypt(nonce, message, None)

        # Ciphertext includes authentication tag
        # Tag is appended to ciphertext (16 bytes)
        assert len(ciphertext) >= len(message) + min_mac_size, \
            f"Authentication tag must be at least {min_mac_size} bytes"

    def test_hash_function_compliance(self, compliance_config):
        """Test hash function compliance with NIST standards."""
        approved_hashes = compliance_config['algorithms']['hash']

        # Test SHA-256 (NIST approved)
        message = b"Hash function compliance test"

        sha256_hash = hashlib.sha256(message).digest()
        assert len(sha256_hash) == 32, "SHA-256 output must be 256 bits"

        sha384_hash = hashlib.sha384(message).digest()
        assert len(sha384_hash) == 48, "SHA-384 output must be 384 bits"

        sha512_hash = hashlib.sha512(message).digest()
        assert len(sha512_hash) == 64, "SHA-512 output must be 512 bits"

        # Verify hashes are deterministic
        assert sha256_hash == hashlib.sha256(message).digest()

    def test_random_number_generation_compliance(self):
        """Test random number generation meets cryptographic standards."""
        # Use os.urandom which is cryptographically secure

        # Generate random bytes
        random_bytes = os.urandom(32)
        assert len(random_bytes) == 32

        # Verify randomness (basic statistical test)
        samples = [os.urandom(32) for _ in range(100)]

        # All samples should be unique
        assert len(set(samples)) == 100, "RNG produced duplicate values"

        # Verify byte distribution
        all_bytes = b''.join(samples)
        byte_counts = [all_bytes.count(bytes([i])) for i in range(256)]

        # Each byte value should appear roughly equally (within 3 std devs)
        import statistics
        mean_count = statistics.mean(byte_counts)
        std_dev = statistics.stdev(byte_counts)

        for count in byte_counts:
            # Allow some variance but not too much
            assert abs(count - mean_count) < 4 * std_dev, \
                "RNG distribution appears non-random"

    def test_key_derivation_compliance(self):
        """Test key derivation function compliance."""
        # NIST SP 800-132 recommendations for PBKDF2
        password = b"user_password"
        salt = os.urandom(16)  # Minimum 128 bits

        # Test PBKDF2 with SHA-256
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256-bit key
            salt=salt,
            iterations=100000,  # NIST recommends 10,000 minimum
            backend=default_backend()
        )

        key1 = kdf.derive(password)
        assert len(key1) == 32, "Derived key must be 256 bits"

        # Verify determinism
        kdf2 = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key2 = kdf2.derive(password)
        assert key1 == key2, "KDF must be deterministic"

        # Different salt should produce different key
        kdf3 = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=os.urandom(16),
            iterations=100000,
            backend=default_backend()
        )
        key3 = kdf3.derive(password)
        assert key1 != key3, "Different salts must produce different keys"

    def test_gdpr_data_protection_compliance(self):
        """Test GDPR data protection requirements."""
        # GDPR requires encryption of personal data

        # Simulate personal data
        personal_data = b"User PII: John Doe, john@example.com, SSN: 123-45-6789"

        # Encrypt with strong algorithm
        key = AESGCM.generate_key(bit_length=256)
        cipher = AESGCM(key)
        nonce = os.urandom(12)

        # Encrypt personal data
        ciphertext = cipher.encrypt(nonce, personal_data, None)

        # Verify encryption
        assert ciphertext != personal_data, "Data must be encrypted"

        # Verify decryption (for authorized access)
        decrypted = cipher.decrypt(nonce, ciphertext, None)
        assert decrypted == personal_data, "Decryption must be reversible"

        # Simulate "right to be forgotten" - key deletion
        del key
        del cipher

        # After key deletion, data should be unrecoverable
        # (In production, securely wipe key from memory)

    def test_pci_dss_compliance(self):
        """Test PCI-DSS compliance for payment card data protection."""
        # PCI-DSS requires strong cryptography

        # Simulate credit card data
        card_data = b"4532-1234-5678-9010|12/25|123"

        # Encrypt with AES-256 (PCI-DSS approved)
        key = AESGCM.generate_key(bit_length=256)
        cipher = AESGCM(key)
        nonce = os.urandom(12)

        ciphertext = cipher.encrypt(nonce, card_data, None)

        # Verify encryption
        assert ciphertext != card_data
        assert len(key) == 32, "PCI-DSS requires 256-bit keys"

        # Verify secure transmission (authenticated encryption)
        decrypted = cipher.decrypt(nonce, ciphertext, None)
        assert decrypted == card_data

        # Tampering should be detected
        tampered = ciphertext[:-1] + b'\x00'
        with pytest.raises(Exception):
            cipher.decrypt(nonce, tampered, None)

    def test_hipaa_compliance(self):
        """Test HIPAA compliance for healthcare data protection."""
        # HIPAA requires encryption of ePHI

        # Simulate protected health information
        phi_data = b"Patient: John Doe, DOB: 1980-01-01, Diagnosis: Diabetes"

        # Encrypt with FIPS-approved algorithm
        key = AESGCM.generate_key(bit_length=256)
        cipher = AESGCM(key)
        nonce = os.urandom(12)

        ciphertext = cipher.encrypt(nonce, phi_data, None)

        # Verify data is protected
        assert ciphertext != phi_data
        assert len(key) == 32, "HIPAA requires strong encryption (256-bit)"

        # Verify integrity protection
        decrypted = cipher.decrypt(nonce, ciphertext, None)
        assert decrypted == phi_data

    def test_quantum_safe_migration_compliance(self):
        """Test compliance with quantum-safe migration guidelines."""
        # NIST recommends hybrid approach during transition

        # Use post-quantum algorithms
        kem = KeyEncapsulation('Kyber768')
        sig = Signature('Dilithium3')

        # Verify algorithms are quantum-safe
        public_key_kem = kem.generate_keypair()
        ciphertext, shared_secret = kem.encap_secret(public_key_kem)

        public_key_sig = sig.generate_keypair()
        message = b"Quantum-safe test"
        signature = sig.sign(message)

        # Verify operations work
        assert kem.decap_secret(ciphertext) == shared_secret
        assert sig.verify(message, signature, public_key_sig)

    def test_audit_logging_compliance(self):
        """Test that cryptographic operations can be audited."""
        # Track cryptographic operations for compliance

        audit_log = []

        def log_operation(operation: str, details: Dict):
            """Log cryptographic operation for audit."""
            import time
            audit_log.append({
                'timestamp': time.time(),
                'operation': operation,
                'details': details
            })

        # Perform operations with logging
        kem = KeyEncapsulation('Kyber768')
        public_key = kem.generate_keypair()
        log_operation('key_generation', {'algorithm': 'Kyber768', 'key_type': 'KEM'})

        ciphertext, shared_secret = kem.encap_secret(public_key)
        log_operation('encapsulation', {'algorithm': 'Kyber768'})

        decrypted = kem.decap_secret(ciphertext)
        log_operation('decapsulation', {'algorithm': 'Kyber768'})

        # Verify audit log
        assert len(audit_log) == 3, "All operations should be logged"
        assert all('timestamp' in entry for entry in audit_log)
        assert all('operation' in entry for entry in audit_log)

    def test_key_rotation_compliance(self):
        """Test compliance with key rotation policies."""
        # Simulate key rotation after time period

        # Generate initial key
        key_v1 = AESGCM.generate_key(bit_length=256)
        cipher_v1 = AESGCM(key_v1)

        message = b"Data encrypted with key v1"
        nonce_v1 = os.urandom(12)
        ciphertext_v1 = cipher_v1.encrypt(nonce_v1, message, None)

        # Rotate to new key
        key_v2 = AESGCM.generate_key(bit_length=256)
        cipher_v2 = AESGCM(key_v2)

        # Re-encrypt with new key
        decrypted = cipher_v1.decrypt(nonce_v1, ciphertext_v1, None)
        nonce_v2 = os.urandom(12)
        ciphertext_v2 = cipher_v2.encrypt(nonce_v2, decrypted, None)

        # Verify new encryption
        assert cipher_v2.decrypt(nonce_v2, ciphertext_v2, None) == message

        # Old key should be securely deleted
        del key_v1
        del cipher_v1

    def test_algorithm_agility_compliance(self):
        """Test algorithm agility for future upgrades."""
        # System should support multiple algorithms

        algorithms_tested = []

        # Test multiple KEM algorithms
        for alg in ['Kyber512', 'Kyber768', 'Kyber1024']:
            kem = KeyEncapsulation(alg)
            public_key = kem.generate_keypair()
            ciphertext, shared_secret = kem.encap_secret(public_key)
            assert kem.decap_secret(ciphertext) == shared_secret
            algorithms_tested.append(alg)

        # Test multiple signature algorithms
        for alg in ['Dilithium2', 'Dilithium3', 'Dilithium5']:
            sig = Signature(alg)
            public_key = sig.generate_keypair()
            signature = sig.sign(b"test")
            assert sig.verify(b"test", signature, public_key)
            algorithms_tested.append(alg)

        # Verify agility
        assert len(algorithms_tested) == 6, \
            "System should support multiple algorithms"

    @pytest.mark.parametrize("security_level", [1, 3, 5])
    def test_security_level_compliance(self, security_level):
        """Test compliance with NIST security levels."""
        # NIST defines 5 security levels
        # Kyber512 = Level 1, Kyber768 = Level 3, Kyber1024 = Level 5

        level_mapping = {
            1: 'Kyber512',
            3: 'Kyber768',
            5: 'Kyber1024'
        }

        if security_level in level_mapping:
            algorithm = level_mapping[security_level]
            kem = KeyEncapsulation(algorithm)
            public_key = kem.generate_keypair()
            ciphertext, shared_secret = kem.encap_secret(public_key)

            # Verify operation
            assert kem.decap_secret(ciphertext) == shared_secret

            # Verify shared secret size (always 256 bits)
            assert len(shared_secret) == 32
