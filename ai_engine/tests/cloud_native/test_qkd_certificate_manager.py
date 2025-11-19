"""
Unit tests for Quantum Key Distribution Certificate Manager.
"""
import pytest
import base64
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.service_mesh.istio.qkd_certificate_manager import (
    CertificateAlgorithm,
    CertificateMetadata,
    QuantumCertificateManager
)


class TestCertificateAlgorithm:
    """Test CertificateAlgorithm enum"""

    def test_algorithm_values(self):
        """Test available algorithms"""
        assert CertificateAlgorithm.KYBER_512 is not None
        assert CertificateAlgorithm.KYBER_768 is not None
        assert CertificateAlgorithm.KYBER_1024 is not None
        assert CertificateAlgorithm.DILITHIUM_2 is not None
        assert CertificateAlgorithm.DILITHIUM_3 is not None
        assert CertificateAlgorithm.DILITHIUM_5 is not None


class TestCertificateMetadata:
    """Test CertificateMetadata dataclass"""

    def test_metadata_creation(self):
        """Test creating certificate metadata"""
        now = datetime.utcnow()
        expiry = now + timedelta(days=365)

        metadata = CertificateMetadata(
            subject="CN=test-service",
            issuer="CN=root-ca",
            serial_number="123456",
            not_before=now,
            not_after=expiry,
            key_algorithm=CertificateAlgorithm.KYBER_1024,
            signature_algorithm=CertificateAlgorithm.DILITHIUM_5
        )

        assert metadata.subject == "CN=test-service"
        assert metadata.key_algorithm == CertificateAlgorithm.KYBER_1024
        assert metadata.signature_algorithm == CertificateAlgorithm.DILITHIUM_5


class TestQuantumCertificateManager:
    """Test QuantumCertificateManager class"""

    @pytest.fixture
    def cert_manager(self):
        """Create certificate manager instance"""
        return QuantumCertificateManager(
            key_algorithm=CertificateAlgorithm.KYBER_1024,
            signature_algorithm=CertificateAlgorithm.DILITHIUM_5,
            cert_validity_days=365,
            rotation_threshold_days=30,
            auto_rotation=True
        )

    def test_manager_initialization(self, cert_manager):
        """Test manager initialization"""
        assert cert_manager.key_algorithm == CertificateAlgorithm.KYBER_1024
        assert cert_manager.signature_algorithm == CertificateAlgorithm.DILITHIUM_5
        assert cert_manager.cert_validity_days == 365
        assert cert_manager.rotation_threshold_days == 30
        assert cert_manager.auto_rotation is True

    def test_generate_root_ca(self, cert_manager):
        """Test root CA generation"""
        root_ca = cert_manager.generate_root_ca(
            subject="CN=Cronos Root CA,O=Cronos AI",
            validity_years=10
        )

        assert "certificate" in root_ca
        assert "public_key" in root_ca
        assert "private_key" in root_ca
        assert "signature_public_key" in root_ca
        assert "signature_private_key" in root_ca
        assert "metadata" in root_ca

        # Check certificate format
        cert = root_ca["certificate"]
        assert cert.startswith("-----BEGIN CERTIFICATE-----")
        assert cert.endswith("-----END CERTIFICATE-----\n")

        # Check metadata
        metadata = root_ca["metadata"]
        assert "CN=Cronos Root CA" in metadata["subject"]

    def test_generate_service_certificate(self, cert_manager):
        """Test service certificate generation"""
        # Generate root CA first
        root_ca = cert_manager.generate_root_ca(
            subject="CN=Root CA",
            validity_years=10
        )

        # Generate service certificate
        service_cert = cert_manager.generate_service_certificate(
            service_name="test-service",
            namespace="default",
            ca_cert=root_ca["certificate"],
            ca_signature_private_key=root_ca["signature_private_key"],
            sans=["test-service.default.svc.cluster.local"]
        )

        assert "certificate" in service_cert
        assert "public_key" in service_cert
        assert "private_key" in service_cert
        assert "metadata" in service_cert

        # Check SAN in metadata
        metadata = service_cert["metadata"]
        assert "test-service.default.svc.cluster.local" in metadata.get("sans", [])

    def test_create_kubernetes_secret(self, cert_manager):
        """Test Kubernetes secret creation"""
        root_ca = cert_manager.generate_root_ca("CN=Root CA", 10)
        service_cert = cert_manager.generate_service_certificate(
            service_name="test-service",
            namespace="default",
            ca_cert=root_ca["certificate"],
            ca_signature_private_key=root_ca["signature_private_key"]
        )

        secret = cert_manager.create_kubernetes_secret(
            cert_data=service_cert,
            secret_name="test-service-cert",
            namespace="default"
        )

        assert secret["apiVersion"] == "v1"
        assert secret["kind"] == "Secret"
        assert secret["type"] == "kubernetes.io/tls"
        assert secret["metadata"]["name"] == "test-service-cert"
        assert secret["metadata"]["namespace"] == "default"

        # Check data fields
        data = secret["data"]
        assert "tls.crt" in data
        assert "tls.key" in data
        assert "ca.crt" in data

        # Verify base64 encoding
        for key, value in data.items():
            decoded = base64.b64decode(value)
            assert len(decoded) > 0

    def test_check_rotation_needed(self, cert_manager):
        """Test certificate rotation check"""
        root_ca = cert_manager.generate_root_ca("CN=Root CA", 10)
        service_cert = cert_manager.generate_service_certificate(
            service_name="test-service",
            namespace="default",
            ca_cert=root_ca["certificate"],
            ca_signature_private_key=root_ca["signature_private_key"]
        )

        # New certificate should not need rotation
        needs_rotation = cert_manager.check_rotation_needed(service_cert)
        assert needs_rotation is False

        # Test with expired certificate
        expired_metadata = service_cert["metadata"].copy()
        expired_metadata["not_after"] = (
            datetime.utcnow() - timedelta(days=1)
        ).isoformat()

        expired_cert = service_cert.copy()
        expired_cert["metadata"] = expired_metadata

        needs_rotation = cert_manager.check_rotation_needed(expired_cert)
        assert needs_rotation is True

    def test_rotate_certificate(self, cert_manager):
        """Test certificate rotation"""
        root_ca = cert_manager.generate_root_ca("CN=Root CA", 10)

        # Generate initial certificate
        original_cert = cert_manager.generate_service_certificate(
            service_name="test-service",
            namespace="default",
            ca_cert=root_ca["certificate"],
            ca_signature_private_key=root_ca["signature_private_key"]
        )

        # Store in cache
        cert_manager._certificate_cache = {
            ("test-service", "default"): original_cert
        }

        # Rotate certificate
        new_cert = cert_manager.rotate_certificate(
            service_name="test-service",
            namespace="default",
            ca_cert=root_ca["certificate"],
            ca_signature_private_key=root_ca["signature_private_key"]
        )

        # Verify new certificate is different
        assert new_cert["certificate"] != original_cert["certificate"]
        assert new_cert["private_key"] != original_cert["private_key"]

        # Verify cache updated
        cached = cert_manager.get_certificate("test-service", "default")
        assert cached == new_cert

    def test_pem_encoding_decoding(self, cert_manager):
        """Test PEM encoding/decoding"""
        test_data = b"test certificate data"
        label = "CERTIFICATE"

        # Encode
        pem = cert_manager._encode_pem(test_data, label)
        assert pem.startswith(f"-----BEGIN {label}-----")
        assert pem.endswith(f"-----END {label}-----\n")

        # Decode
        decoded = cert_manager._decode_pem(pem)
        assert decoded == test_data

    def test_different_key_algorithms(self):
        """Test different key algorithms"""
        algorithms = [
            CertificateAlgorithm.KYBER_512,
            CertificateAlgorithm.KYBER_768,
            CertificateAlgorithm.KYBER_1024
        ]

        for algo in algorithms:
            manager = QuantumCertificateManager(
                key_algorithm=algo,
                signature_algorithm=CertificateAlgorithm.DILITHIUM_3
            )

            root_ca = manager.generate_root_ca("CN=Test CA", 5)
            assert root_ca is not None
            assert "certificate" in root_ca

    def test_certificate_cache(self, cert_manager):
        """Test certificate caching"""
        root_ca = cert_manager.generate_root_ca("CN=Root CA", 10)

        # Generate and cache certificate
        service_cert = cert_manager.generate_service_certificate(
            service_name="cached-service",
            namespace="default",
            ca_cert=root_ca["certificate"],
            ca_signature_private_key=root_ca["signature_private_key"]
        )

        # Retrieve from cache
        cached = cert_manager.get_certificate("cached-service", "default")
        assert cached == service_cert

        # Non-existent certificate
        missing = cert_manager.get_certificate("missing-service", "default")
        assert missing is None

    def test_auto_rotation(self):
        """Test auto-rotation feature"""
        manager = QuantumCertificateManager(
            auto_rotation=True,
            rotation_threshold_days=30
        )
        assert manager.auto_rotation is True

        manager_no_rotation = QuantumCertificateManager(auto_rotation=False)
        assert manager_no_rotation.auto_rotation is False
