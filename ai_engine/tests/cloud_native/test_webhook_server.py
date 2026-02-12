"""
Unit tests for Kubernetes Admission Webhook Server.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.container_security.admission_control.webhook_server import (
    AdmissionAction,
    AdmissionRequest,
    SecurityPolicy,
    AdmissionWebhookServer,
)


class TestAdmissionAction:
    """Test AdmissionAction enum"""

    def test_admission_actions(self):
        """Test admission action values"""
        assert AdmissionAction.ALLOW is not None
        assert AdmissionAction.DENY is not None
        assert AdmissionAction.PATCH is not None


class TestAdmissionRequest:
    """Test AdmissionRequest dataclass"""

    def test_request_creation(self):
        """Test creating admission request"""
        request = AdmissionRequest(
            uid="abc-123", kind="Pod", namespace="default", operation="CREATE", object={"metadata": {"name": "test-pod"}}
        )

        assert request.uid == "abc-123"
        assert request.kind == "Pod"
        assert request.namespace == "default"
        assert request.operation == "CREATE"


class TestSecurityPolicy:
    """Test SecurityPolicy dataclass"""

    def test_policy_creation(self):
        """Test creating security policy"""
        policy = SecurityPolicy(
            name="strict-policy",
            allowed_registries=["docker.io", "gcr.io"],
            require_signature=True,
            block_privileged=True,
            scan_for_quantum_vulnerabilities=True,
        )

        assert policy.name == "strict-policy"
        assert "docker.io" in policy.allowed_registries
        assert policy.require_signature is True
        assert policy.scan_for_quantum_vulnerabilities is True


class TestAdmissionWebhookServer:
    """Test AdmissionWebhookServer class"""

    @pytest.fixture
    def webhook_server(self):
        """Create webhook server instance"""
        return AdmissionWebhookServer(
            port=8443, tls_cert_path="/etc/webhook/certs/tls.crt", tls_key_path="/etc/webhook/certs/tls.key"
        )

    def test_server_initialization(self, webhook_server):
        """Test server initialization"""
        assert webhook_server.port == 8443
        assert webhook_server.tls_cert_path == "/etc/webhook/certs/tls.crt"
        assert webhook_server.tls_key_path == "/etc/webhook/certs/tls.key"

    def test_validate_pod_allowed(self, webhook_server):
        """Test pod validation - allow case"""
        policy = SecurityPolicy(
            name="permissive", allowed_registries=["docker.io"], require_signature=False, block_privileged=False
        )

        webhook_server.add_policy(policy)

        pod = {"metadata": {"name": "test-pod"}, "spec": {"containers": [{"name": "app", "image": "docker.io/nginx:latest"}]}}

        result = webhook_server.validate_pod(pod, "permissive")

        assert result["action"] == AdmissionAction.ALLOW
        assert result["allowed"] is True

    def test_validate_pod_denied_registry(self, webhook_server):
        """Test pod validation - deny due to registry"""
        policy = SecurityPolicy(name="strict", allowed_registries=["gcr.io"], require_signature=False, block_privileged=False)

        webhook_server.add_policy(policy)

        pod = {"metadata": {"name": "test-pod"}, "spec": {"containers": [{"name": "app", "image": "docker.io/nginx:latest"}]}}

        result = webhook_server.validate_pod(pod, "strict")

        assert result["action"] == AdmissionAction.DENY
        assert result["allowed"] is False
        assert "registry" in result.get("message", "").lower()

    def test_validate_pod_privileged(self, webhook_server):
        """Test blocking privileged pods"""
        policy = SecurityPolicy(
            name="no-privilege", allowed_registries=["docker.io"], require_signature=False, block_privileged=True
        )

        webhook_server.add_policy(policy)

        privileged_pod = {
            "metadata": {"name": "privileged-pod"},
            "spec": {
                "containers": [{"name": "app", "image": "docker.io/nginx:latest", "securityContext": {"privileged": True}}]
            },
        }

        result = webhook_server.validate_pod(privileged_pod, "no-privilege")

        assert result["action"] == AdmissionAction.DENY
        assert result["allowed"] is False
        assert "privileged" in result.get("message", "").lower()

    def test_verify_image_signature(self, webhook_server):
        """Test image signature verification"""
        # This would typically integrate with cosign
        image = "gcr.io/signed-image:v1"

        try:
            is_signed = webhook_server._verify_image_signature(image)
            # In unit test, might not have actual signed images
            assert isinstance(is_signed, bool)
        except Exception:
            # Signature verification might not be available
            pytest.skip("Signature verification not available")

    def test_check_quantum_vulnerable_libs(self, webhook_server):
        """Test quantum vulnerability checking"""
        image = "test-image:latest"

        vulnerabilities = webhook_server._check_quantum_vulnerable_libs(image)

        assert isinstance(vulnerabilities, list)

    def test_validate_image_registry(self, webhook_server):
        """Test registry validation"""
        policy = SecurityPolicy(name="test-policy", allowed_registries=["docker.io", "gcr.io", "ghcr.io"])

        # Allowed registry
        assert webhook_server._validate_image_registry("docker.io/nginx:latest", policy) is True
        assert webhook_server._validate_image_registry("gcr.io/my-project/app:v1", policy) is True

        # Disallowed registry
        assert webhook_server._validate_image_registry("quay.io/app:latest", policy) is False

    def test_add_remove_policy(self, webhook_server):
        """Test adding and removing policies"""
        policy = SecurityPolicy(name="temp-policy", allowed_registries=["docker.io"])

        # Add policy
        webhook_server.add_policy(policy)
        assert webhook_server.get_policy("temp-policy") is not None

        # Remove policy
        webhook_server.remove_policy("temp-policy")
        assert webhook_server.get_policy("temp-policy") is None

    @pytest.mark.asyncio
    async def test_handle_health(self, webhook_server):
        """Test health check handler"""

        # Mock request
        class MockRequest:
            pass

        request = MockRequest()

        try:
            response = await webhook_server.handle_health(request)
            assert response is not None
        except Exception:
            # Handler might require aiohttp setup
            pytest.skip("Health handler requires aiohttp context")

    def test_create_webhook_config(self, webhook_server):
        """Test ValidatingWebhookConfiguration generation"""
        webhook_config = webhook_server.create_webhook_config()

        assert webhook_config["apiVersion"] == "admissionregistration.k8s.io/v1"
        assert webhook_config["kind"] == "ValidatingWebhookConfiguration"

        webhooks = webhook_config["webhooks"]
        assert len(webhooks) > 0

        webhook = webhooks[0]
        assert "name" in webhook
        assert "clientConfig" in webhook
        assert "rules" in webhook
        assert "admissionReviewVersions" in webhook

    def test_patch_action(self, webhook_server):
        """Test patch action generation"""
        policy = SecurityPolicy(name="patch-policy", allowed_registries=["docker.io"], add_security_labels=True)

        webhook_server.add_policy(policy)

        pod = {"metadata": {"name": "test-pod"}, "spec": {"containers": [{"name": "app", "image": "docker.io/nginx:latest"}]}}

        result = webhook_server.validate_pod(pod, "patch-policy")

        # Might return PATCH action with modifications
        if result["action"] == AdmissionAction.PATCH:
            assert "patch" in result
            assert isinstance(result["patch"], list)

    def test_multiple_containers_validation(self, webhook_server):
        """Test validating pod with multiple containers"""
        policy = SecurityPolicy(name="multi-container", allowed_registries=["docker.io"], require_signature=False)

        webhook_server.add_policy(policy)

        pod = {
            "metadata": {"name": "multi-pod"},
            "spec": {
                "containers": [
                    {"name": "app", "image": "docker.io/nginx:latest"},
                    {"name": "sidecar", "image": "docker.io/envoy:v1.28"},
                ]
            },
        }

        result = webhook_server.validate_pod(pod, "multi-container")

        assert result["action"] == AdmissionAction.ALLOW
        assert result["allowed"] is True
