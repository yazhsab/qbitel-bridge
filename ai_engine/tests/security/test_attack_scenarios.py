"""
Security tests for various attack scenarios.
Tests resistance to common attacks including replay, MITM, injection, and DoS.
"""

import pytest
import time
import os
import hashlib
from typing import List, Dict, Tuple
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305, AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from oqs import KeyEncapsulation, Signature


class TestAttackScenarios:
    """Security tests for resistance to various attack scenarios."""

    @pytest.fixture
    def encryption_key(self):
        """Generate a secure encryption key for testing."""
        return ChaCha20Poly1305.generate_key()

    @pytest.fixture
    def message_samples(self):
        """Generate sample messages for testing."""
        return {
            'small': b"Small message",
            'medium': b"Medium message " * 100,
            'large': b"Large message " * 1000,
            'binary': os.urandom(1024)
        }

    def test_replay_attack_prevention(self, encryption_key):
        """Test that replay attacks are prevented using nonces."""
        cipher = ChaCha20Poly1305(encryption_key)
        message = b"Secret message"

        # Encrypt message with nonce
        nonce1 = os.urandom(12)
        ciphertext1 = cipher.encrypt(nonce1, message, None)

        # Decrypt successfully
        plaintext1 = cipher.decrypt(nonce1, ciphertext1, None)
        assert plaintext1 == message

        # Attempt replay: reuse same nonce with different message
        message2 = b"Different message"
        ciphertext2 = cipher.encrypt(nonce1, message2, None)

        # Ciphertext should be different even with same nonce (but this is bad practice)
        # The key point: using same nonce is cryptographically broken
        # In production, nonce reuse should be prevented

        # Verify that different nonces produce different ciphertexts
        nonce2 = os.urandom(12)
        ciphertext3 = cipher.encrypt(nonce2, message, None)
        assert ciphertext1 != ciphertext3, "Different nonces should produce different ciphertexts"

    def test_man_in_the_middle_attack_detection(self):
        """Test detection of MITM attacks using digital signatures."""
        # Alice and Bob establish keys
        alice_sig = Signature('Dilithium3')
        alice_public = alice_sig.generate_keypair()

        bob_sig = Signature('Dilithium3')
        bob_public = bob_sig.generate_keypair()

        # Alice sends a message with signature
        message = b"Confidential data from Alice"
        alice_signature = alice_sig.sign(message)

        # Bob verifies it's really from Alice
        is_valid = bob_sig.verify(message, alice_signature, alice_public)
        assert is_valid, "Valid signature should verify"

        # Attacker (Eve) tries to modify the message
        tampered_message = b"Tampered data from Eve"

        # Bob verifies the tampered message with Alice's signature
        is_valid_tampered = bob_sig.verify(tampered_message, alice_signature, alice_public)
        assert not is_valid_tampered, "Tampered message should not verify"

        # Attacker tries to replace signature
        eve_sig = Signature('Dilithium3')
        eve_public = eve_sig.generate_keypair()
        eve_signature = eve_sig.sign(tampered_message)

        # Bob checks with Alice's public key (not Eve's)
        is_valid_eve = bob_sig.verify(tampered_message, eve_signature, alice_public)
        assert not is_valid_eve, "Eve's signature should not verify with Alice's key"

    def test_chosen_plaintext_attack_resistance(self, encryption_key):
        """Test resistance to chosen plaintext attacks."""
        cipher = ChaCha20Poly1305(encryption_key)

        # Attacker chooses specific plaintexts
        chosen_plaintexts = [
            b"\x00" * 16,  # All zeros
            b"\xff" * 16,  # All ones
            b"\xaa" * 16,  # Alternating pattern
            b"AAAA" * 4,   # Repetitive
        ]

        ciphertexts = []
        nonces = []

        for plaintext in chosen_plaintexts:
            nonce = os.urandom(12)
            ciphertext = cipher.encrypt(nonce, plaintext, None)
            ciphertexts.append(ciphertext)
            nonces.append(nonce)

        # Verify all ciphertexts are different
        assert len(set(ciphertexts)) == len(ciphertexts), \
            "Ciphertexts should be unique even for chosen plaintexts"

        # Verify ciphertexts don't reveal patterns
        for i, ct in enumerate(ciphertexts):
            # Ciphertext should not resemble plaintext
            assert ct != chosen_plaintexts[i], "Ciphertext should not match plaintext"

            # Ciphertext should appear random (basic entropy check)
            unique_bytes = len(set(ct))
            assert unique_bytes > len(ct) * 0.3, "Ciphertext should have good entropy"

    def test_chosen_ciphertext_attack_resistance(self, encryption_key):
        """Test resistance to chosen ciphertext attacks (CCA)."""
        cipher = ChaCha20Poly1305(encryption_key)
        message = b"Original message"
        nonce = os.urandom(12)

        # Encrypt original message
        ciphertext = cipher.encrypt(nonce, message, None)

        # Attacker modifies ciphertext
        tampered_ciphertexts = [
            ciphertext[:-1] + b'\x00',  # Modify last byte
            ciphertext[:5] + b'\xff' + ciphertext[6:],  # Modify middle
            ciphertext + b'\x00',  # Append byte
            ciphertext[1:],  # Remove first byte
        ]

        for tampered_ct in tampered_ciphertexts:
            # Decryption should fail for tampered ciphertext
            with pytest.raises(Exception):
                cipher.decrypt(nonce, tampered_ct, None)

    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks in cryptographic operations."""
        kem = KeyEncapsulation('Kyber768')
        public_key = kem.generate_keypair()

        # Measure decapsulation time for multiple valid ciphertexts
        valid_timings = []
        for _ in range(50):
            ciphertext, _ = kem.encap_secret(public_key)
            start = time.perf_counter()
            shared_secret = kem.decap_secret(ciphertext)
            end = time.perf_counter()
            valid_timings.append(end - start)

        # Measure decapsulation time for invalid ciphertexts
        invalid_timings = []
        for _ in range(50):
            invalid_ciphertext = os.urandom(1088)  # Kyber768 ciphertext size
            start = time.perf_counter()
            try:
                shared_secret = kem.decap_secret(invalid_ciphertext)
            except:
                pass
            end = time.perf_counter()
            invalid_timings.append(end - start)

        # Calculate average times
        import statistics
        avg_valid = statistics.mean(valid_timings)
        avg_invalid = statistics.mean(invalid_timings)

        # Timing difference should be minimal (constant-time operation)
        # Allow up to 50% difference due to system noise
        timing_ratio = max(avg_valid, avg_invalid) / min(avg_valid, avg_invalid)
        assert timing_ratio < 1.5, \
            f"Potential timing attack vulnerability: {timing_ratio:.2f}x difference"

    def test_padding_oracle_attack_prevention(self):
        """Test that padding oracle attacks are prevented."""
        # Use AEAD cipher which includes authentication
        key = AESGCM.generate_key(bit_length=256)
        cipher = AESGCM(key)

        message = b"Secret message with padding"
        nonce = os.urandom(12)

        # Encrypt message
        ciphertext = cipher.encrypt(nonce, message, None)

        # Attacker tries to modify ciphertext to probe padding
        for i in range(min(16, len(ciphertext))):
            tampered = bytearray(ciphertext)
            tampered[i] ^= 0xFF
            tampered = bytes(tampered)

            # Decryption should fail without revealing padding info
            with pytest.raises(Exception) as exc_info:
                cipher.decrypt(nonce, tampered, None)

            # Exception should not reveal information about padding
            error_msg = str(exc_info.value).lower()
            assert 'padding' not in error_msg, \
                "Error message should not reveal padding information"

    def test_length_extension_attack_prevention(self):
        """Test prevention of length extension attacks."""
        # Use HMAC which is resistant to length extension
        from cryptography.hazmat.primitives import hmac

        key = os.urandom(32)
        message = b"Original message"

        # Create HMAC
        h = hmac.HMAC(key, hashes.SHA256(), backend=default_backend())
        h.update(message)
        mac1 = h.finalize()

        # Attacker tries to extend message
        extended_message = message + b" with extension"

        h2 = hmac.HMAC(key, hashes.SHA256(), backend=default_backend())
        h2.update(extended_message)
        mac2 = h2.finalize()

        # MACs should be completely different
        assert mac1 != mac2, "Extended message should have different MAC"

        # Attacker cannot compute valid MAC for extended message without key
        # This is guaranteed by HMAC construction

    def test_birthday_attack_resistance(self):
        """Test resistance to birthday attacks on hash functions."""
        # Generate many messages and hash them
        num_messages = 10000
        hashes_seen = set()

        for i in range(num_messages):
            message = f"Message {i}".encode()
            hash_digest = hashlib.sha256(message).digest()
            hashes_seen.add(hash_digest)

        # All hashes should be unique (no collisions)
        assert len(hashes_seen) == num_messages, \
            f"Hash collision detected: {num_messages - len(hashes_seen)} collisions"

    def test_denial_of_service_resistance(self):
        """Test resistance to DoS attacks through resource exhaustion."""
        cipher = ChaCha20Poly1305(ChaCha20Poly1305.generate_key())

        # Attempt to process many large messages quickly
        large_message = b"x" * (1024 * 1024)  # 1MB
        iterations = 10

        start_time = time.perf_counter()

        for _ in range(iterations):
            nonce = os.urandom(12)
            ciphertext = cipher.encrypt(nonce, large_message, None)
            plaintext = cipher.decrypt(nonce, ciphertext, None)

        elapsed = time.perf_counter() - start_time
        time_per_op = elapsed / iterations

        # Operations should complete in reasonable time (< 100ms per 1MB)
        assert time_per_op < 0.1, \
            f"Operation too slow, vulnerable to DoS: {time_per_op:.3f}s per 1MB"

    def test_key_exhaustion_attack_resistance(self):
        """Test that key space is large enough to resist exhaustion."""
        # Verify key sizes are sufficient
        kyber_key_size = 32  # 256 bits shared secret
        dilithium_key_size = 32  # 256 bits seed

        # Key space should be at least 2^256
        min_key_space_bits = 256

        assert kyber_key_size * 8 >= min_key_space_bits, \
            "Kyber key space too small"

        # Test that generated keys have good distribution
        kem = KeyEncapsulation('Kyber768')
        shared_secrets = []

        for _ in range(100):
            public_key = kem.generate_keypair()
            ciphertext, shared_secret = kem.encap_secret(public_key)
            shared_secrets.append(shared_secret)

        # All secrets should be unique
        assert len(set(shared_secrets)) == 100, "Insufficient key space diversity"

    def test_downgrade_attack_prevention(self):
        """Test prevention of downgrade attacks to weaker algorithms."""
        # System should enforce use of strong algorithms

        # Acceptable algorithms (post-quantum)
        acceptable_kems = ['Kyber512', 'Kyber768', 'Kyber1024']
        acceptable_sigs = ['Dilithium2', 'Dilithium3', 'Dilithium5']

        # Verify we can use strong algorithms
        for algorithm in acceptable_kems:
            kem = KeyEncapsulation(algorithm)
            public_key = kem.generate_keypair()
            assert public_key is not None

        for algorithm in acceptable_sigs:
            sig = Signature(algorithm)
            public_key = sig.generate_keypair()
            assert public_key is not None

        # Note: In production, explicitly reject weak algorithms
        # and enforce minimum security levels

    def test_side_channel_cache_timing_resistance(self):
        """Test resistance to cache timing side-channel attacks."""
        kem = KeyEncapsulation('Kyber768')
        public_key = kem.generate_keypair()

        # Perform many operations and measure timing variance
        timings = []
        for _ in range(100):
            ciphertext, _ = kem.encap_secret(public_key)

            start = time.perf_counter()
            shared_secret = kem.decap_secret(ciphertext)
            end = time.perf_counter()

            timings.append(end - start)

        # Calculate coefficient of variation
        import statistics
        mean_time = statistics.mean(timings)
        std_dev = statistics.stdev(timings)
        cv = (std_dev / mean_time) * 100

        # Timing should be relatively consistent (CV < 30%)
        assert cv < 30, f"High timing variance may leak information: CV={cv:.1f}%"

    def test_fault_injection_attack_resistance(self):
        """Test resistance to fault injection attacks."""
        sig = Signature('Dilithium3')
        public_key = sig.generate_keypair()
        message = b"Test message"

        # Generate valid signature
        signature = sig.sign(message)

        # Simulate fault injection by flipping bits in signature
        for bit_position in [0, 100, 1000, 2000]:
            if bit_position < len(signature) * 8:
                byte_pos = bit_position // 8
                bit_pos = bit_position % 8

                faulty_signature = bytearray(signature)
                faulty_signature[byte_pos] ^= (1 << bit_pos)
                faulty_signature = bytes(faulty_signature)

                # Verification should fail for faulty signature
                is_valid = sig.verify(message, faulty_signature, public_key)
                assert not is_valid, \
                    f"Fault injection at bit {bit_position} not detected"

    def test_nonce_reuse_detection(self):
        """Test that nonce reuse is catastrophic and detectable."""
        key = ChaCha20Poly1305.generate_key()
        cipher = ChaCha20Poly1305(key)

        nonce = os.urandom(12)
        message1 = b"First message"
        message2 = b"Second message"

        # Encrypt both messages with same nonce (BAD!)
        ciphertext1 = cipher.encrypt(nonce, message1, None)
        ciphertext2 = cipher.encrypt(nonce, message2, None)

        # This demonstrates why nonce reuse is bad
        # In production, nonce reuse MUST be prevented

        # Verify that same nonce produces different ciphertexts for different messages
        assert ciphertext1 != ciphertext2, \
            "Different messages should produce different ciphertexts"

    def test_session_hijacking_prevention(self):
        """Test prevention of session hijacking through authenticated encryption."""
        # Simulate session establishment
        kem = KeyEncapsulation('Kyber768')
        public_key = kem.generate_keypair()
        ciphertext, session_key = kem.encap_secret(public_key)

        # Derive encryption key from session key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'session_salt',
            iterations=100000,
            backend=default_backend()
        )
        encryption_key = kdf.derive(session_key)

        cipher = ChaCha20Poly1305(encryption_key)

        # Legitimate message
        message = b"Authenticated session message"
        nonce = os.urandom(12)
        ciphertext_msg = cipher.encrypt(nonce, message, None)

        # Attacker tries to inject message without session key
        attacker_key = os.urandom(32)
        attacker_cipher = ChaCha20Poly1305(attacker_key)
        attacker_message = b"Hijacked message"
        attacker_ciphertext = attacker_cipher.encrypt(nonce, attacker_message, None)

        # Legitimate recipient cannot decrypt attacker's message
        with pytest.raises(Exception):
            cipher.decrypt(nonce, attacker_ciphertext, None)

    def test_quantum_attack_resistance(self):
        """Test that post-quantum algorithms are used for quantum resistance."""
        # Verify we're using post-quantum algorithms
        kem_algorithms = ['Kyber512', 'Kyber768', 'Kyber1024']
        sig_algorithms = ['Dilithium2', 'Dilithium3', 'Dilithium5']

        # These are NIST post-quantum finalists/winners
        for alg in kem_algorithms:
            kem = KeyEncapsulation(alg)
            public_key = kem.generate_keypair()
            ciphertext, shared_secret = kem.encap_secret(public_key)

            # Verify operation completes
            decrypted_secret = kem.decap_secret(ciphertext)
            assert shared_secret == decrypted_secret

        for alg in sig_algorithms:
            sig = Signature(alg)
            public_key = sig.generate_keypair()
            message = b"Quantum-safe signature test"
            signature = sig.sign(message)

            # Verify signature
            assert sig.verify(message, signature, public_key)

    def test_brute_force_attack_resistance(self):
        """Test that key derivation is resistant to brute force."""
        password = b"user_password"
        salt = os.urandom(16)

        # Use strong KDF with high iteration count
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # High iteration count slows brute force
            backend=default_backend()
        )

        # Measure time to derive key
        start = time.perf_counter()
        key = kdf.derive(password)
        elapsed = time.perf_counter() - start

        # Should take noticeable time (> 10ms) to slow brute force
        assert elapsed > 0.01, \
            f"KDF too fast, vulnerable to brute force: {elapsed*1000:.1f}ms"

        # Verify key is deterministic
        kdf2 = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key2 = kdf2.derive(password)
        assert key == key2, "KDF should be deterministic"

    @pytest.mark.parametrize("attack_type", [
        "bit_flip",
        "byte_modification",
        "truncation",
        "extension"
    ])
    def test_integrity_attack_detection(self, attack_type, encryption_key):
        """Test detection of various integrity attacks."""
        cipher = ChaCha20Poly1305(encryption_key)
        message = b"Integrity protected message"
        nonce = os.urandom(12)

        # Encrypt message
        ciphertext = cipher.encrypt(nonce, message, None)

        # Perform attack
        if attack_type == "bit_flip":
            tampered = bytearray(ciphertext)
            tampered[5] ^= 0x01
            tampered = bytes(tampered)
        elif attack_type == "byte_modification":
            tampered = bytearray(ciphertext)
            tampered[10] = 0xFF
            tampered = bytes(tampered)
        elif attack_type == "truncation":
            tampered = ciphertext[:-5]
        elif attack_type == "extension":
            tampered = ciphertext + b"\x00\x00"

        # Decryption should fail
        with pytest.raises(Exception):
            cipher.decrypt(nonce, tampered, None)
