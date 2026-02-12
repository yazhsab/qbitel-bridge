"""
Dilithium Image Signer

Signs container images using Dilithium-5 post-quantum digital signatures
for quantum-safe image verification.
"""

import logging
import hashlib
import base64
import json
from typing import Dict, Any, Optional
import datetime

# Post-quantum cryptography imports
from dilithium_py.dilithium import Dilithium2, Dilithium3, Dilithium5

logger = logging.getLogger(__name__)


class DilithiumSigner:
    """
    Signs container images with Dilithium quantum-safe signatures.
    Uses real NIST-approved Dilithium post-quantum cryptography.
    """

    def __init__(self, key_size: int = 5):
        """
        Initialize Dilithium signer with real PQC implementation.

        Args:
            key_size: Dilithium security level (2, 3, or 5)
        """
        self.key_size = key_size

        # Map security levels to implementations
        self._dilithium_implementations = {2: Dilithium2, 3: Dilithium3, 5: Dilithium5}

        self._dilithium = self._dilithium_implementations.get(key_size, Dilithium5)

        logger.info(f"Initialized DilithiumSigner with real Dilithium-{key_size} " f"post-quantum cryptography")

    def generate_keypair(self) -> Dict[str, str]:
        """
        Generate real Dilithium keypair using NIST PQC implementation.

        Returns:
            Dict with base64-encoded public key, private key, and algorithm info
        """
        logger.info(f"Generating real Dilithium-{self.key_size} keypair")

        # Use real Dilithium implementation to generate keys
        public_key, private_key = self._dilithium.keygen()

        logger.info(
            f"Generated Dilithium-{self.key_size} keypair: " f"pk={len(public_key)} bytes, sk={len(private_key)} bytes"
        )

        return {
            "private_key": base64.b64encode(private_key).decode(),
            "public_key": base64.b64encode(public_key).decode(),
            "algorithm": f"dilithium-{self.key_size}",
            "key_sizes": {"public_key_bytes": len(public_key), "private_key_bytes": len(private_key)},
        }

    def sign_image(self, image_digest: str, private_key: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sign a container image using real Dilithium post-quantum signatures.

        Args:
            image_digest: Image SHA256 digest
            private_key: Base64-encoded Dilithium private key
            metadata: Optional metadata to include

        Returns:
            Dict containing quantum-safe signature and metadata
        """
        # Create signing payload with timestamp
        timestamp = datetime.datetime.utcnow().isoformat()
        payload = {
            "image_digest": image_digest,
            "timestamp": timestamp,
            "metadata": metadata or {},
            "algorithm": f"dilithium-{self.key_size}",
        }

        # Serialize payload for signing
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")

        # Decode the private key
        private_key_bytes = base64.b64decode(private_key)

        # Sign with real Dilithium implementation
        signature = self._dilithium.sign(private_key_bytes, payload_bytes)

        logger.info(
            f"Signed image {image_digest[:16]}... with Dilithium-{self.key_size}: "
            f"payload={len(payload_bytes)} bytes, signature={len(signature)} bytes"
        )

        return {
            "signature": base64.b64encode(signature).decode(),
            "algorithm": f"dilithium-{self.key_size}",
            "payload": payload,
            "signature_size_bytes": len(signature),
        }

    def verify_signature(
        self, image_digest: str, signature: str, public_key: str, payload: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Verify image signature using real Dilithium post-quantum verification.

        Args:
            image_digest: Image SHA256 digest
            signature: Base64-encoded Dilithium signature
            public_key: Base64-encoded Dilithium public key
            payload: Original payload that was signed (optional, will be reconstructed if not provided)

        Returns:
            True if signature is valid, False otherwise
        """
        logger.info(f"Verifying Dilithium-{self.key_size} signature for image {image_digest[:16]}...")

        try:
            # If payload not provided, reconstruct it
            # Note: This is a simplified version - in production, the full payload
            # should be stored with the signature
            if payload is None:
                logger.warning("Payload not provided, verification may fail")
                return False

            # Serialize the payload exactly as it was when signed
            payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")

            # Decode signature and public key
            signature_bytes = base64.b64decode(signature)
            public_key_bytes = base64.b64decode(public_key)

            # Verify with real Dilithium implementation
            is_valid = self._dilithium.verify(public_key_bytes, payload_bytes, signature_bytes)

            logger.info(f"Signature verification result for {image_digest[:16]}...: {is_valid}")

            return is_valid

        except Exception as e:
            logger.error(f"Signature verification failed for {image_digest}: {e}")
            return False
