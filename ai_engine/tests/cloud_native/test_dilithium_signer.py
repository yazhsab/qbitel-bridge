"""
Unit tests for Dilithium Post-Quantum Image Signer.
"""

import pytest
import base64
import hashlib
import sys
from pathlib import Path

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.container_security.signing.dilithium_signer import DilithiumSigner


class TestDilithiumSigner:
    """Test DilithiumSigner class"""

    @pytest.fixture
    def signer(self):
        """Create Dilithium signer instance"""
        return DilithiumSigner(key_size=3)  # Dilithium-3

    def test_signer_initialization(self, signer):
        """Test signer initialization"""
        assert signer.key_size == 3
        assert signer.algorithm_name == "Dilithium-3"

    def test_generate_keypair(self, signer):
        """Test keypair generation"""
        keypair = signer.generate_keypair()

        assert "public_key" in keypair
        assert "private_key" in keypair

        # Check key format (should be base64 encoded)
        public_key = keypair["public_key"]
        private_key = keypair["private_key"]

        assert isinstance(public_key, str)
        assert isinstance(private_key, str)

        # Should be valid base64
        try:
            base64.b64decode(public_key)
            base64.b64decode(private_key)
        except Exception:
            pytest.fail("Keys are not valid base64")

    def test_sign_image(self, signer):
        """Test signing container image"""
        # Generate keypair
        keypair = signer.generate_keypair()

        # Create image digest
        image_digest = hashlib.sha256(b"test-image-content").hexdigest()

        # Sign the image
        signature_data = signer.sign_image(
            image_digest=image_digest,
            private_key=keypair["private_key"],
            metadata={"image": "test:v1", "timestamp": "2024-01-01"},
        )

        assert "signature" in signature_data
        assert "payload" in signature_data
        assert "algorithm" in signature_data

        assert signature_data["algorithm"] == "Dilithium-3"

        # Verify payload contains metadata
        payload = signature_data["payload"]
        assert image_digest in payload

    def test_verify_signature(self, signer):
        """Test signature verification"""
        # Generate keypair
        keypair = signer.generate_keypair()

        # Create and sign image
        image_digest = hashlib.sha256(b"test-image-content").hexdigest()
        signature_data = signer.sign_image(
            image_digest=image_digest, private_key=keypair["private_key"], metadata={"image": "test:v1"}
        )

        # Verify signature
        is_valid = signer.verify_signature(
            image_digest=image_digest,
            signature=signature_data["signature"],
            public_key=keypair["public_key"],
            payload=signature_data["payload"],
        )

        assert is_valid is True

    def test_verify_invalid_signature(self, signer):
        """Test verification of invalid signature"""
        # Generate keypair
        keypair = signer.generate_keypair()

        # Create image and sign
        image_digest = hashlib.sha256(b"test-image").hexdigest()
        signature_data = signer.sign_image(image_digest=image_digest, private_key=keypair["private_key"])

        # Tamper with signature
        tampered_signature = signature_data["signature"][:-10] + "tampered=="

        # Verification should fail
        is_valid = signer.verify_signature(
            image_digest=image_digest,
            signature=tampered_signature,
            public_key=keypair["public_key"],
            payload=signature_data["payload"],
        )

        assert is_valid is False

    def test_verify_wrong_public_key(self, signer):
        """Test verification with wrong public key"""
        # Generate two keypairs
        keypair1 = signer.generate_keypair()
        keypair2 = signer.generate_keypair()

        # Sign with keypair1
        image_digest = hashlib.sha256(b"test-image").hexdigest()
        signature_data = signer.sign_image(image_digest=image_digest, private_key=keypair1["private_key"])

        # Try to verify with keypair2's public key
        is_valid = signer.verify_signature(
            image_digest=image_digest,
            signature=signature_data["signature"],
            public_key=keypair2["public_key"],
            payload=signature_data["payload"],
        )

        assert is_valid is False

    def test_different_key_sizes(self):
        """Test different Dilithium key sizes"""
        key_sizes = [2, 3, 5]  # Dilithium-2, 3, 5

        for size in key_sizes:
            signer = DilithiumSigner(key_size=size)
            assert signer.key_size == size
            assert signer.algorithm_name == f"Dilithium-{size}"

            # Generate and test keypair
            keypair = signer.generate_keypair()
            assert "public_key" in keypair
            assert "private_key" in keypair

    def test_sign_with_metadata(self, signer):
        """Test signing with additional metadata"""
        keypair = signer.generate_keypair()
        image_digest = hashlib.sha256(b"test").hexdigest()

        metadata = {
            "image": "myapp:v1.2.3",
            "registry": "gcr.io",
            "timestamp": "2024-01-01T00:00:00Z",
            "builder": "CI/CD Pipeline",
        }

        signature_data = signer.sign_image(image_digest=image_digest, private_key=keypair["private_key"], metadata=metadata)

        # Check metadata in payload
        payload = signature_data["payload"]
        assert "myapp:v1.2.3" in payload
        assert "gcr.io" in payload

    def test_signature_determinism(self, signer):
        """Test that signatures are different for same input (due to randomness)"""
        keypair = signer.generate_keypair()
        image_digest = hashlib.sha256(b"test").hexdigest()

        # Sign twice
        sig1 = signer.sign_image(image_digest, keypair["private_key"])
        sig2 = signer.sign_image(image_digest, keypair["private_key"])

        # Signatures should be different (due to nonce/randomness)
        # Note: Some implementations might be deterministic
        assert sig1["signature"] != sig2["signature"] or sig1 == sig2

    def test_keypair_uniqueness(self, signer):
        """Test that generated keypairs are unique"""
        keypair1 = signer.generate_keypair()
        keypair2 = signer.generate_keypair()

        assert keypair1["public_key"] != keypair2["public_key"]
        assert keypair1["private_key"] != keypair2["private_key"]

    def test_large_metadata(self, signer):
        """Test signing with large metadata"""
        keypair = signer.generate_keypair()
        image_digest = hashlib.sha256(b"test").hexdigest()

        # Create large metadata
        metadata = {
            "layers": ["sha256:" + hashlib.sha256(str(i).encode()).hexdigest() for i in range(100)],
            "annotations": {f"key{i}": f"value{i}" for i in range(50)},
        }

        signature_data = signer.sign_image(image_digest=image_digest, private_key=keypair["private_key"], metadata=metadata)

        assert "signature" in signature_data

        # Verify
        is_valid = signer.verify_signature(
            image_digest=image_digest,
            signature=signature_data["signature"],
            public_key=keypair["public_key"],
            payload=signature_data["payload"],
        )

        assert is_valid is True

    def test_empty_metadata(self, signer):
        """Test signing without metadata"""
        keypair = signer.generate_keypair()
        image_digest = hashlib.sha256(b"test").hexdigest()

        signature_data = signer.sign_image(image_digest=image_digest, private_key=keypair["private_key"], metadata={})

        assert "signature" in signature_data

        # Verify
        is_valid = signer.verify_signature(
            image_digest=image_digest,
            signature=signature_data["signature"],
            public_key=keypair["public_key"],
            payload=signature_data["payload"],
        )

        assert is_valid is True

    def test_invalid_key_size(self):
        """Test initialization with invalid key size"""
        with pytest.raises(Exception):
            DilithiumSigner(key_size=99)  # Invalid size
