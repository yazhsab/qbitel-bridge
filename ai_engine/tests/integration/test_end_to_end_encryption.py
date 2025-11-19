"""
Integration tests for End-to-End Encryption.
Tests quantum-safe encryption across the entire stack.
"""
import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.service_mesh.istio.qkd_certificate_manager import (
    QuantumCertificateManager,
    CertificateAlgorithm
)
from cloud_native.container_security.signing.dilithium_signer import DilithiumSigner
from cloud_native.event_streaming.kafka.secure_producer import SecureKafkaProducer
from unittest.mock import patch


class TestEndToEndEncryption:
    """Test end-to-end quantum-safe encryption"""

    def test_quantum_certificate_chain(self):
        """Test quantum-safe certificate chain"""
        cert_manager = QuantumCertificateManager(
            key_algorithm=CertificateAlgorithm.KYBER_1024,
            signature_algorithm=CertificateAlgorithm.DILITHIUM_5
        )

        # Root CA
        root_ca = cert_manager.generate_root_ca("CN=Root CA", 10)

        # Intermediate CA
        intermediate = cert_manager.generate_service_certificate(
            service_name="intermediate-ca",
            namespace="istio-system",
            ca_cert=root_ca["certificate"],
            ca_signature_private_key=root_ca["signature_private_key"]
        )

        # Service certificate
        service_cert = cert_manager.generate_service_certificate(
            service_name="my-service",
            namespace="production",
            ca_cert=intermediate["certificate"],
            ca_signature_private_key=root_ca["signature_private_key"]
        )

        assert root_ca is not None
        assert intermediate is not None
        assert service_cert is not None

    @patch('kafka.KafkaProducer')
    def test_message_encryption_chain(self, mock_kafka):
        """Test message encryption with quantum-safe algorithms"""
        producer = SecureKafkaProducer(
            bootstrap_servers=["localhost:9092"],
            topic="encrypted-channel",
            enable_quantum_encryption=True
        )

        # Encrypt and send message
        message = {"sensitive": "data", "user_id": "12345"}
        encrypted = producer._encrypt_message(str(message).encode())

        assert encrypted is not None
        assert encrypted != str(message).encode()
