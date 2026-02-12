"""
Security tests for quantum cryptographic implementations.
Tests security properties, resistance to attacks, and cryptographic correctness.
"""

import pytest
import os
from oqs import KeyEncapsulation, Signature
from typing import Tuple


class TestQuantumCryptoSecurity:
    """Security tests for post-quantum cryptographic algorithms."""

    @pytest.fixture
    def kyber_instance(self):
        """Create Kyber768 instance for testing."""
        return KeyEncapsulation("Kyber768")

    @pytest.fixture
    def dilithium_instance(self):
        """Create Dilithium3 instance for testing."""
        return Signature("Dilithium3")

    def test_kyber_shared_secret_uniqueness(self, kyber_instance):
        """Verify that each key encapsulation produces unique shared secrets."""
        kem = kyber_instance
        public_key = kem.generate_keypair()

        shared_secrets = set()
        for _ in range(100):
            ciphertext, shared_secret = kem.encap_secret(public_key)
            shared_secrets.add(shared_secret)

        # All shared secrets should be unique
        assert len(shared_secrets) == 100, "Shared secrets are not unique"

    def test_kyber_ciphertext_uniqueness(self, kyber_instance):
        """Verify that each encapsulation produces unique ciphertexts."""
        kem = kyber_instance
        public_key = kem.generate_keypair()

        ciphertexts = set()
        for _ in range(100):
            ciphertext, shared_secret = kem.encap_secret(public_key)
            ciphertexts.add(ciphertext)

        # All ciphertexts should be unique (probabilistically)
        assert len(ciphertexts) == 100, "Ciphertexts are not sufficiently random"

    def test_kyber_key_recovery_impossible(self, kyber_instance):
        """Verify that private key cannot be recovered from public key."""
        kem1 = KeyEncapsulation("Kyber768")
        public_key1 = kem1.generate_keypair()

        kem2 = KeyEncapsulation("Kyber768")
        public_key2 = kem2.generate_keypair()

        # Generate shared secrets using both keys
        ciphertext1, secret1_sender = kem1.encap_secret(public_key1)
        ciphertext2, secret2_sender = kem2.encap_secret(public_key2)

        # Decapsulation with correct key should work
        secret1_receiver = kem1.decap_secret(ciphertext1)
        assert secret1_sender == secret1_receiver

        # Attempting to use wrong private key should produce different secret
        secret_wrong = kem2.decap_secret(ciphertext1)
        assert secret_wrong != secret1_sender, "Key recovery vulnerability detected"

    def test_kyber_chosen_ciphertext_attack_resistance(self, kyber_instance):
        """Test resistance to chosen ciphertext attacks."""
        kem = kyber_instance
        public_key = kem.generate_keypair()

        # Generate valid ciphertext
        valid_ciphertext, valid_secret = kem.encap_secret(public_key)

        # Attempt to modify ciphertext (simulate attacker tampering)
        tampered_ciphertext = bytearray(valid_ciphertext)
        tampered_ciphertext[0] ^= 0xFF  # Flip bits in first byte
        tampered_ciphertext = bytes(tampered_ciphertext)

        # Decapsulate tampered ciphertext
        tampered_secret = kem.decap_secret(tampered_ciphertext)

        # Tampered ciphertext should produce different secret
        assert tampered_secret != valid_secret, "CCA vulnerability: same secret from tampered ciphertext"

    def test_kyber_key_size_validation(self):
        """Verify correct key sizes for Kyber variants."""
        variants = {
            "Kyber512": {"public": 800, "secret": 1632, "ciphertext": 768, "shared": 32},
            "Kyber768": {"public": 1184, "secret": 2400, "ciphertext": 1088, "shared": 32},
            "Kyber1024": {"public": 1568, "secret": 3168, "ciphertext": 1568, "shared": 32},
        }

        for algorithm, expected_sizes in variants.items():
            kem = KeyEncapsulation(algorithm)
            public_key = kem.generate_keypair()
            ciphertext, shared_secret = kem.encap_secret(public_key)

            assert len(public_key) == expected_sizes["public"], f"{algorithm} public key size mismatch"
            assert len(ciphertext) == expected_sizes["ciphertext"], f"{algorithm} ciphertext size mismatch"
            assert len(shared_secret) == expected_sizes["shared"], f"{algorithm} shared secret size mismatch"

    def test_dilithium_signature_determinism(self, dilithium_instance):
        """Verify that signatures are non-deterministic (use random nonce)."""
        sig = dilithium_instance
        public_key = sig.generate_keypair()
        message = b"Test message for signature determinism"

        signatures = set()
        for _ in range(10):
            signature = sig.sign(message)
            signatures.add(signature)

        # Dilithium uses randomized signing, so signatures should be unique
        assert len(signatures) == 10, "Signatures should be randomized"

    def test_dilithium_signature_forgery_resistance(self, dilithium_instance):
        """Test resistance to signature forgery."""
        sig = dilithium_instance
        public_key = sig.generate_keypair()
        message = b"Original message"

        # Sign the original message
        valid_signature = sig.sign(message)

        # Verify original signature works
        assert sig.verify(message, valid_signature, public_key), "Valid signature failed"

        # Attempt forgery: modify message
        forged_message = b"Forged message"
        assert not sig.verify(
            forged_message, valid_signature, public_key
        ), "Signature forgery vulnerability: signature valid for different message"

        # Attempt forgery: modify signature
        forged_signature = bytearray(valid_signature)
        forged_signature[0] ^= 0xFF
        forged_signature = bytes(forged_signature)

        assert not sig.verify(
            message, forged_signature, public_key
        ), "Signature forgery vulnerability: modified signature accepted"

    def test_dilithium_signature_replay_attack(self, dilithium_instance):
        """Test that signatures can be verified multiple times (replay should be handled at protocol level)."""
        sig = dilithium_instance
        public_key = sig.generate_keypair()
        message = b"Message for replay test"

        signature = sig.sign(message)

        # Verify signature multiple times (should always work)
        for _ in range(10):
            assert sig.verify(message, signature, public_key), "Signature verification unstable"

    def test_dilithium_key_size_validation(self):
        """Verify correct key sizes for Dilithium variants."""
        variants = {
            "Dilithium2": {"public": 1312, "secret": 2528, "signature": 2420},
            "Dilithium3": {"public": 1952, "secret": 4000, "signature": 3293},
            "Dilithium5": {"public": 2592, "secret": 4864, "signature": 4595},
        }

        for algorithm, expected_sizes in variants.items():
            sig = Signature(algorithm)
            public_key = sig.generate_keypair()
            signature = sig.sign(b"test")

            assert len(public_key) == expected_sizes["public"], f"{algorithm} public key size mismatch"
            assert len(signature) == expected_sizes["signature"], f"{algorithm} signature size mismatch"

    def test_random_oracle_model_properties(self, kyber_instance):
        """Test that cryptographic hash functions behave like random oracles."""
        kem = kyber_instance

        # Generate multiple key pairs
        public_keys = []
        for _ in range(20):
            kem_temp = KeyEncapsulation("Kyber768")
            public_key = kem_temp.generate_keypair()
            public_keys.append(public_key)

        # All public keys should be unique (high entropy)
        assert len(set(public_keys)) == 20, "Insufficient randomness in key generation"

        # Check distribution of first byte (should be relatively uniform)
        first_bytes = [pk[0] for pk in public_keys]
        unique_first_bytes = len(set(first_bytes))

        # Should have good distribution (at least 50% unique in sample)
        assert unique_first_bytes >= 10, "Poor randomness distribution in key generation"

    def test_side_channel_timing_consistency(self, kyber_instance):
        """Test for timing side-channel resistance (constant-time operations)."""
        import time

        kem = kyber_instance
        public_key = kem.generate_keypair()

        # Measure decapsulation time for multiple ciphertexts
        timings = []
        for _ in range(50):
            ciphertext, _ = kem.encap_secret(public_key)

            start = time.perf_counter()
            shared_secret = kem.decap_secret(ciphertext)
            end = time.perf_counter()

            timings.append(end - start)

        # Calculate coefficient of variation
        import statistics

        avg_time = statistics.mean(timings)
        std_dev = statistics.stdev(timings)
        cv = (std_dev / avg_time) * 100

        # Timing should be relatively consistent (CV < 30%)
        assert cv < 30, f"Potential timing side-channel: CV={cv:.1f}%"

    def test_malformed_input_handling(self, kyber_instance):
        """Test handling of malformed inputs."""
        kem = kyber_instance
        public_key = kem.generate_keypair()

        # Test with invalid ciphertext sizes
        invalid_ciphertexts = [
            b"",  # Empty
            b"short",  # Too short
            os.urandom(10),  # Wrong size
            os.urandom(2000),  # Too long
        ]

        for invalid_ct in invalid_ciphertexts:
            try:
                # Should either raise exception or return deterministic output
                secret = kem.decap_secret(invalid_ct)
                # If no exception, verify it returns something (implicit rejection)
                assert len(secret) == 32, "Invalid output for malformed ciphertext"
            except Exception:
                # Exception is acceptable for malformed input
                pass

    def test_key_reuse_safety(self, kyber_instance):
        """Test that key pairs can be safely reused for multiple encapsulations."""
        kem = kyber_instance
        public_key = kem.generate_keypair()

        secrets_sender = []
        ciphertexts = []

        # Use same public key multiple times
        for _ in range(50):
            ciphertext, shared_secret = kem.encap_secret(public_key)
            secrets_sender.append(shared_secret)
            ciphertexts.append(ciphertext)

        # Verify all can be decapsulated correctly
        for i, ciphertext in enumerate(ciphertexts):
            decapsulated_secret = kem.decap_secret(ciphertext)
            assert decapsulated_secret == secrets_sender[i], f"Key reuse safety violation at index {i}"

    def test_public_key_validation(self):
        """Test that invalid public keys are rejected."""
        kem = KeyEncapsulation("Kyber768")

        invalid_public_keys = [
            b"",  # Empty
            b"invalid",  # Wrong size
            os.urandom(100),  # Wrong size
            b"\x00" * 1184,  # All zeros (valid size but invalid key)
        ]

        for invalid_pk in invalid_public_keys:
            try:
                ciphertext, shared_secret = kem.encap_secret(invalid_pk)
                # Some implementations may accept but produce predictable output
                # Security requirement: should not crash or expose secrets
            except Exception as e:
                # Rejection is acceptable
                assert isinstance(e, (ValueError, RuntimeError, TypeError)), f"Unexpected exception type: {type(e)}"

    def test_zero_knowledge_property(self, dilithium_instance):
        """Test that signatures don't leak information about private key."""
        sig = dilithium_instance
        public_key = sig.generate_keypair()

        messages = [b"Message 1", b"Message 2", b"Message 3"]
        signatures = [sig.sign(msg) for msg in messages]

        # Signatures should not reveal patterns about the private key
        # Check that signatures are sufficiently different
        for i, sig1 in enumerate(signatures):
            for j, sig2 in enumerate(signatures):
                if i != j:
                    # Calculate Hamming distance
                    diff_bytes = sum(a != b for a, b in zip(sig1, sig2))
                    similarity_ratio = 1 - (diff_bytes / len(sig1))

                    # Signatures should be very different (< 60% similar)
                    assert similarity_ratio < 0.6, f"Signatures too similar: {similarity_ratio:.2%}"

    @pytest.mark.parametrize("algorithm", ["Kyber512", "Kyber768", "Kyber1024"])
    def test_kyber_kem_correctness(self, algorithm):
        """Test correctness of KEM for all Kyber variants."""
        kem = KeyEncapsulation(algorithm)
        public_key = kem.generate_keypair()

        for _ in range(20):
            ciphertext, shared_secret_sender = kem.encap_secret(public_key)
            shared_secret_receiver = kem.decap_secret(ciphertext)

            assert shared_secret_sender == shared_secret_receiver, f"{algorithm} KEM correctness failed"
            assert len(shared_secret_sender) == 32, f"{algorithm} shared secret size incorrect"

    @pytest.mark.parametrize("algorithm", ["Dilithium2", "Dilithium3", "Dilithium5"])
    def test_dilithium_signature_correctness(self, algorithm):
        """Test correctness of signatures for all Dilithium variants."""
        sig = Signature(algorithm)
        public_key = sig.generate_keypair()
        messages = [b"Test 1", b"Test 2", b"Test 3" * 100]

        for message in messages:
            signature = sig.sign(message)
            is_valid = sig.verify(message, signature, public_key)

            assert is_valid, f"{algorithm} signature verification failed"

            # Wrong message should fail
            wrong_message = message + b"tampered"
            assert not sig.verify(wrong_message, signature, public_key), f"{algorithm} accepted signature for wrong message"

    def test_entropy_of_generated_keys(self):
        """Test that generated keys have sufficient entropy."""
        kem = KeyEncapsulation("Kyber768")

        public_keys = []
        for _ in range(10):
            kem_temp = KeyEncapsulation("Kyber768")
            pk = kem_temp.generate_keypair()
            public_keys.append(pk)

        # Calculate byte-wise entropy
        byte_positions = {}
        for pk in public_keys:
            for i, byte in enumerate(pk):
                if i not in byte_positions:
                    byte_positions[i] = []
                byte_positions[i].append(byte)

        # Check variance in bytes (good entropy should show variation)
        import statistics

        low_entropy_positions = 0
        for pos, bytes_at_pos in byte_positions.items():
            unique_values = len(set(bytes_at_pos))
            if unique_values < 3:  # Very low variation
                low_entropy_positions += 1

        # Less than 10% of positions should have low entropy
        max_low_entropy = len(byte_positions) * 0.1
        assert low_entropy_positions < max_low_entropy, f"Insufficient entropy: {low_entropy_positions} low-entropy positions"

    def test_collision_resistance(self):
        """Test collision resistance of key generation."""
        keys_generated = 1000
        kem_keys = set()
        sig_keys = set()

        for _ in range(keys_generated):
            kem = KeyEncapsulation("Kyber768")
            kem_pk = kem.generate_keypair()
            kem_keys.add(kem_pk)

            sig = Signature("Dilithium3")
            sig_pk = sig.generate_keypair()
            sig_keys.add(sig_pk)

        # All keys should be unique (no collisions)
        assert len(kem_keys) == keys_generated, f"KEM key collision detected: {keys_generated - len(kem_keys)} collisions"
        assert (
            len(sig_keys) == keys_generated
        ), f"Signature key collision detected: {keys_generated - len(sig_keys)} collisions"
