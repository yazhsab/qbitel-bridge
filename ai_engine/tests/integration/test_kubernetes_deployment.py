"""
Integration tests for Kubernetes Deployment.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.service_mesh.istio.sidecar_injector import SidecarInjector
from cloud_native.container_security.admission_control.webhook_server import (
    AdmissionWebhookServer,
    SecurityPolicy
)
from cloud_native.service_mesh.istio.mtls_config import MutualTLSConfigurator, MTLSMode


class TestKubernetesDeployment:
    """Test Kubernetes deployment with security"""

    def test_pod_deployment_pipeline(self):
        """Test pod deployment with security checks"""
        # Admission webhook
        webhook = AdmissionWebhookServer(port=8443, tls_cert_path="/tmp/tls.crt", tls_key_path="/tmp/tls.key")
        policy = SecurityPolicy(
            name="production",
            allowed_registries=["gcr.io"],
            require_signature=True
        )
        webhook.add_policy(policy)

        # Pod spec
        pod = {
            "metadata": {"name": "app-pod", "namespace": "production"},
            "spec": {"containers": [{"name": "app", "image": "gcr.io/myapp:v1"}]}
        }

        # Validate
        result = webhook.validate_pod(pod, "production")

        # If allowed, inject sidecar
        if result["allowed"]:
            injector = SidecarInjector(enable_quantum_crypto=True)
            injected_pod = injector.inject_sidecar(pod)

            # Configure mTLS
            mtls = MutualTLSConfigurator(default_mode=MTLSMode.STRICT)
            peer_auth = mtls.create_peer_authentication(
                name="app-pod-mtls",
                namespace="production",
                mode=MTLSMode.STRICT
            )

            assert injected_pod is not None
            assert peer_auth is not None
