"""
Integration tests for Istio and Envoy.
"""
import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.service_mesh.istio.mtls_config import MutualTLSConfigurator, MTLSMode
from cloud_native.service_mesh.istio.qkd_certificate_manager import (
    QuantumCertificateManager,
    CertificateAlgorithm
)
from cloud_native.service_mesh.envoy.xds_server import EnvoyXDSServer
from cloud_native.service_mesh.envoy.traffic_encryption import TrafficEncryptionManager, TLSVersion


class TestIstioEnvoyIntegration:
    """Test Istio and Envoy integration"""

    @pytest.mark.asyncio
    async def test_mtls_with_quantum_certificates(self):
        """Test mTLS with quantum-safe certificates"""
        # Generate quantum certificates
        cert_manager = QuantumCertificateManager(
            key_algorithm=CertificateAlgorithm.KYBER_1024,
            signature_algorithm=CertificateAlgorithm.DILITHIUM_5
        )

        root_ca = cert_manager.generate_root_ca("CN=Istio CA", 10)
        service_cert = cert_manager.generate_service_certificate(
            service_name="test-service",
            namespace="default",
            ca_cert=root_ca["certificate"],
            ca_signature_private_key=root_ca["signature_private_key"]
        )

        # Create Kubernetes secret
        secret = cert_manager.create_kubernetes_secret(
            cert_data=service_cert,
            secret_name="test-service-cert",
            namespace="default"
        )

        # Configure Istio mTLS
        mtls_config = MutualTLSConfigurator(default_mode=MTLSMode.STRICT)
        dest_rule = mtls_config.create_destination_rule(
            name="test-service",
            host="test-service.default.svc.cluster.local",
            namespace="default",
            enable_quantum_tls=True
        )

        # Configure Envoy TLS
        encryption_mgr = TrafficEncryptionManager(
            enable_quantum_safe=True,
            min_tls_version=TLSVersion.TLS_1_3
        )

        tls_context = encryption_mgr.create_mtls_context(
            cert_path="/etc/certs/tls.crt",
            key_path="/etc/certs/tls.key",
            ca_path="/etc/certs/ca.crt",
            require_client_cert=True
        )

        assert secret is not None
        assert dest_rule is not None
        assert tls_context is not None

    @pytest.mark.asyncio
    async def test_sidecar_injection_with_xds(self):
        """Test sidecar injection with xDS configuration"""
        from cloud_native.service_mesh.istio.sidecar_injector import SidecarInjector

        # Create sidecar injector
        injector = SidecarInjector(
            namespace="istio-system",
            enable_quantum_crypto=True
        )

        # Inject sidecar
        pod = {
            "metadata": {"name": "test-pod", "namespace": "default"},
            "spec": {"containers": [{"name": "app", "image": "nginx:latest"}]}
        }

        injected_pod = injector.inject_sidecar(pod)

        # Create xDS resources for the sidecar
        xds_server = EnvoyXDSServer(port=15010)

        listener = await xds_server.create_listener(
            name="virtualInbound",
            address="0.0.0.0",
            port=15006,
            enable_quantum_filter=True
        )

        cluster = await xds_server.create_cluster(
            name="outbound|80||nginx.default.svc.cluster.local",
            endpoints=["127.0.0.1:80"],
            lb_policy="ROUND_ROBIN"
        )

        assert injected_pod is not None
        assert listener is not None
        assert cluster is not None
