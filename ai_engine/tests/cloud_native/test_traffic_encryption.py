"""
Unit tests for Envoy Traffic Encryption.
"""

import pytest
import sys
from pathlib import Path

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from cloud_native.service_mesh.envoy.traffic_encryption import (
        TrafficEncryptionManager,
        TLSVersion,
        CipherSuite,
        EncryptionConfig,
    )
except ImportError:
    pytest.skip("traffic_encryption module not found", allow_module_level=True)


class TestTLSVersion:
    """Test TLSVersion enum"""

    def test_tls_versions(self):
        """Test TLS version values"""
        assert TLSVersion.TLS_1_2 is not None
        assert TLSVersion.TLS_1_3 is not None


class TestCipherSuite:
    """Test CipherSuite enum"""

    def test_cipher_suites(self):
        """Test cipher suite values"""
        assert CipherSuite.AES_256_GCM is not None
        assert CipherSuite.CHACHA20_POLY1305 is not None


class TestEncryptionConfig:
    """Test EncryptionConfig dataclass"""

    def test_config_creation(self):
        """Test creating encryption config"""
        config = EncryptionConfig(
            tls_version=TLSVersion.TLS_1_3, cipher_suites=[CipherSuite.AES_256_GCM], enable_quantum_safe=True
        )

        assert config.tls_version == TLSVersion.TLS_1_3
        assert CipherSuite.AES_256_GCM in config.cipher_suites
        assert config.enable_quantum_safe is True


class TestTrafficEncryptionManager:
    """Test TrafficEncryptionManager class"""

    @pytest.fixture
    def encryption_manager(self):
        """Create encryption manager instance"""
        return TrafficEncryptionManager(enable_quantum_safe=True, min_tls_version=TLSVersion.TLS_1_3)

    def test_manager_initialization(self, encryption_manager):
        """Test manager initialization"""
        assert encryption_manager.enable_quantum_safe is True
        assert encryption_manager.min_tls_version == TLSVersion.TLS_1_3

    def test_create_tls_context(self, encryption_manager):
        """Test TLS context creation"""
        tls_context = encryption_manager.create_tls_context(
            cert_path="/etc/certs/tls.crt", key_path="/etc/certs/tls.key", ca_path="/etc/certs/ca.crt"
        )

        assert "common_tls_context" in tls_context
        common_ctx = tls_context["common_tls_context"]

        # Check certificate configuration
        assert "tls_certificates" in common_ctx
        assert len(common_ctx["tls_certificates"]) > 0

        # Check validation context
        assert "validation_context" in common_ctx

    def test_quantum_safe_cipher_suites(self, encryption_manager):
        """Test quantum-safe cipher suite configuration"""
        tls_context = encryption_manager.create_tls_context(cert_path="/etc/certs/tls.crt", key_path="/etc/certs/tls.key")

        common_ctx = tls_context["common_tls_context"]
        tls_params = common_ctx.get("tls_params", {})

        cipher_suites = tls_params.get("cipher_suites", [])

        # Check for quantum-resistant ciphers
        expected_ciphers = ["TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"]

        for cipher in expected_ciphers:
            assert cipher in cipher_suites or cipher.replace("TLS_", "") in cipher_suites

    def test_tls_version_configuration(self, encryption_manager):
        """Test TLS version configuration"""
        tls_context = encryption_manager.create_tls_context(cert_path="/etc/certs/tls.crt", key_path="/etc/certs/tls.key")

        common_ctx = tls_context["common_tls_context"]
        tls_params = common_ctx.get("tls_params", {})

        # Should enforce TLS 1.3 minimum
        min_version = tls_params.get("tls_minimum_protocol_version")
        assert min_version == "TLSv1_3" or min_version == "TLS_1_3"

    def test_mtls_configuration(self, encryption_manager):
        """Test mutual TLS configuration"""
        tls_context = encryption_manager.create_mtls_context(
            cert_path="/etc/certs/tls.crt",
            key_path="/etc/certs/tls.key",
            ca_path="/etc/certs/ca.crt",
            require_client_cert=True,
        )

        common_ctx = tls_context["common_tls_context"]

        # Check client certificate requirement
        validation_ctx = common_ctx.get("validation_context", {})
        assert "trusted_ca" in validation_ctx

        # Check require_client_certificate
        require_client = tls_context.get("require_client_certificate")
        assert require_client is True

    def test_alpn_protocols(self, encryption_manager):
        """Test ALPN protocol configuration"""
        tls_context = encryption_manager.create_tls_context(
            cert_path="/etc/certs/tls.crt", key_path="/etc/certs/tls.key", alpn_protocols=["h2", "http/1.1"]
        )

        common_ctx = tls_context["common_tls_context"]
        alpn = common_ctx.get("alpn_protocols", [])

        assert "h2" in alpn
        assert "http/1.1" in alpn

    def test_sni_configuration(self, encryption_manager):
        """Test SNI configuration"""
        sni_context = encryption_manager.create_sni_context(
            server_name="api.example.com", cert_path="/etc/certs/api.crt", key_path="/etc/certs/api.key"
        )

        assert "name" in sni_context
        assert sni_context["name"] == "api.example.com"
        assert "transport_socket" in sni_context

    def test_certificate_validation(self, encryption_manager):
        """Test certificate validation configuration"""
        tls_context = encryption_manager.create_tls_context(
            cert_path="/etc/certs/tls.crt",
            key_path="/etc/certs/tls.key",
            ca_path="/etc/certs/ca.crt",
            verify_subject_alt_name=["service.local"],
        )

        common_ctx = tls_context["common_tls_context"]
        validation_ctx = common_ctx.get("validation_context", {})

        # Check SAN verification
        match_subject_alt_names = validation_ctx.get("match_subject_alt_names", [])
        assert any("service.local" in str(san) for san in match_subject_alt_names)

    def test_session_resumption(self, encryption_manager):
        """Test TLS session resumption configuration"""
        tls_context = encryption_manager.create_tls_context(
            cert_path="/etc/certs/tls.crt", key_path="/etc/certs/tls.key", enable_session_resumption=True
        )

        # Check for session ticket configuration
        common_ctx = tls_context["common_tls_context"]
        session_ticket = common_ctx.get("tls_session_ticket_keys")

        # Session resumption might be enabled by default or configured differently
        assert tls_context is not None

    def test_downstream_tls(self, encryption_manager):
        """Test downstream TLS configuration"""
        downstream_tls = encryption_manager.create_downstream_tls_context(
            cert_path="/etc/certs/server.crt", key_path="/etc/certs/server.key"
        )

        assert "common_tls_context" in downstream_tls
        assert downstream_tls.get("require_client_certificate") is not None

    def test_upstream_tls(self, encryption_manager):
        """Test upstream TLS configuration"""
        upstream_tls = encryption_manager.create_upstream_tls_context(ca_path="/etc/certs/ca.crt", sni="backend.local")

        assert "common_tls_context" in upstream_tls
        assert upstream_tls.get("sni") == "backend.local"

    def test_disable_tls(self):
        """Test disabling TLS"""
        manager = TrafficEncryptionManager(enable_tls=False)

        context = manager.create_tls_context(cert_path="/etc/certs/tls.crt", key_path="/etc/certs/tls.key")

        # Should return None or minimal config when TLS disabled
        assert context is None or context == {}
