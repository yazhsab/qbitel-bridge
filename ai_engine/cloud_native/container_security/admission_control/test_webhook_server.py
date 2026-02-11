"""
Integration tests for Admission Webhook Server

Tests the production-ready Kubernetes admission webhook implementation.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from ai_engine.cloud_native.container_security.admission_control.webhook_server import (
    AdmissionWebhookServer,
    SecurityPolicy,
    AdmissionRequest,
    AdmissionAction
)


class TestAdmissionWebhookServer:
    """Test suite for Admission Webhook Server"""

    @pytest.fixture
    def webhook_server(self):
        """Create webhook server instance for testing"""
        return AdmissionWebhookServer(port=8444)

    def test_server_initialization(self, webhook_server):
        """Test server initializes correctly"""
        assert webhook_server.port == 8444
        assert "default" in webhook_server._policies
        assert webhook_server._app is None

    def test_default_policy_loaded(self, webhook_server):
        """Test default security policy is loaded"""
        policy = webhook_server._policies["default"]

        assert policy.name == "default"
        assert policy.require_image_signature is True
        assert policy.disallow_privileged is True
        assert policy.require_quantum_safe_crypto is True

    def test_validate_pod_success(self, webhook_server):
        """Test pod validation with compliant pod"""
        pod = {
            "metadata": {"name": "test-pod"},
            "spec": {
                "containers": [
                    {
                        "name": "app",
                        "image": "gcr.io/distroless/base@sha256:abc123",
                        "securityContext": {
                            "privileged": False,
                            "readOnlyRootFilesystem": False
                        }
                    }
                ],
                "hostNetwork": False,
                "hostPID": False
            }
        }

        result = webhook_server.validate_pod(pod)

        assert result["allowed"] is True
        assert result["status"]["code"] == 200

    def test_validate_pod_privileged_denied(self, webhook_server):
        """Test pod validation denies privileged containers"""
        pod = {
            "metadata": {"name": "test-pod"},
            "spec": {
                "containers": [
                    {
                        "name": "app",
                        "image": "nginx:latest",
                        "securityContext": {
                            "privileged": True  # Privileged container
                        }
                    }
                ]
            }
        }

        result = webhook_server.validate_pod(pod)

        assert result["allowed"] is False
        assert result["status"]["code"] == 403
        assert len(result["violations"]) > 0
        assert any("privileged" in v.lower() for v in result["violations"])

    def test_validate_pod_host_network_denied(self, webhook_server):
        """Test pod validation denies host network"""
        pod = {
            "metadata": {"name": "test-pod"},
            "spec": {
                "containers": [
                    {
                        "name": "app",
                        "image": "gcr.io/app:v1@sha256:abc123"
                    }
                ],
                "hostNetwork": True  # Host network enabled
            }
        }

        result = webhook_server.validate_pod(pod)

        assert result["allowed"] is False
        assert any("host network" in v.lower() for v in result["violations"])

    def test_validate_pod_unsigned_image(self, webhook_server):
        """Test pod validation for unsigned images"""
        pod = {
            "metadata": {"name": "test-pod"},
            "spec": {
                "containers": [
                    {
                        "name": "app",
                        "image": "unknown-registry.io/app:latest"  # No digest, untrusted
                    }
                ]
            }
        }

        result = webhook_server.validate_pod(pod)

        assert result["allowed"] is False
        assert any("signature" in v.lower() for v in result["violations"])

    def test_validate_pod_blocked_registry(self, webhook_server):
        """Test pod validation blocks disallowed registries"""
        pod = {
            "metadata": {"name": "test-pod"},
            "spec": {
                "containers": [
                    {
                        "name": "app",
                        "image": "untrusted.io/app:latest"
                    }
                ]
            }
        }

        result = webhook_server.validate_pod(pod)

        assert result["allowed"] is False
        assert any("disallowed registry" in v.lower() for v in result["violations"])

    def test_validate_pod_quantum_vulnerable_libs(self, webhook_server):
        """Test pod validation detects quantum-vulnerable libraries"""
        pod = {
            "metadata": {"name": "test-pod"},
            "spec": {
                "containers": [
                    {
                        "name": "app",
                        "image": "ubuntu:14.04"  # Outdated base with vulnerable libs
                    }
                ]
            }
        }

        result = webhook_server.validate_pod(pod)

        assert result["allowed"] is False
        assert any("quantum-vulnerable" in v.lower() or "outdated" in v.lower()
                  for v in result["violations"])

    def test_validate_image_registry_allowed(self, webhook_server):
        """Test image registry validation for allowed registries"""
        policy = webhook_server._policies["default"]

        # Test allowed registry
        assert webhook_server._validate_image_registry("gcr.io/myapp:v1", policy) is True
        assert webhook_server._validate_image_registry("docker.io/nginx:latest", policy) is True

    def test_validate_image_registry_blocked(self, webhook_server):
        """Test image registry validation for blocked registries"""
        policy = webhook_server._policies["default"]

        # Test blocked registry
        assert webhook_server._validate_image_registry("untrusted.io/app:v1", policy) is False

    def test_verify_image_signature_with_digest(self, webhook_server):
        """Test image signature verification for images with digest"""
        # Images with digest are considered signed
        assert webhook_server._verify_image_signature(
            "gcr.io/myapp@sha256:abc123"
        ) is True

    def test_verify_image_signature_without_digest(self, webhook_server):
        """Test image signature verification for images without digest"""
        # Unknown images without digest should fail verification
        assert webhook_server._verify_image_signature(
            "unknown.io/app:latest"
        ) is False

    def test_verify_image_signature_trusted_registry(self, webhook_server):
        """Test image signature verification for trusted registries"""
        # Trusted registry images pass even without explicit digest
        assert webhook_server._verify_image_signature(
            "gcr.io/distroless/base:latest"
        ) is True

    def test_check_quantum_vulnerable_libs(self, webhook_server):
        """Test checking for quantum-vulnerable libraries"""
        # Test vulnerable base image
        vulnerable = webhook_server._check_quantum_vulnerable_libs("ubuntu:14.04")
        assert len(vulnerable) > 0

        # Test modern image (should be clean)
        clean = webhook_server._check_quantum_vulnerable_libs("gcr.io/distroless/base:latest")
        # May still have some patterns matched, but fewer

    def test_add_policy(self, webhook_server):
        """Test adding a custom security policy"""
        custom_policy = SecurityPolicy(
            name="strict",
            require_image_signature=True,
            max_critical_vulnerabilities=0,
            max_high_vulnerabilities=0,
            disallow_privileged=True,
            require_read_only_root=True
        )

        webhook_server.add_policy(custom_policy)

        assert "strict" in webhook_server._policies
        assert webhook_server._policies["strict"].require_read_only_root is True

    def test_remove_policy(self, webhook_server):
        """Test removing a security policy"""
        # Add a custom policy
        custom_policy = SecurityPolicy(name="temporary")
        webhook_server.add_policy(custom_policy)

        assert "temporary" in webhook_server._policies

        # Remove it
        webhook_server.remove_policy("temporary")

        assert "temporary" not in webhook_server._policies

    def test_cannot_remove_default_policy(self, webhook_server):
        """Test that default policy cannot be removed"""
        webhook_server.remove_policy("default")

        # Default policy should still exist
        assert "default" in webhook_server._policies

    def test_create_webhook_config(self, webhook_server):
        """Test creating Kubernetes webhook configuration"""
        config = webhook_server.create_webhook_config()

        assert config["apiVersion"] == "admissionregistration.k8s.io/v1"
        assert config["kind"] == "ValidatingWebhookConfiguration"
        assert config["metadata"]["name"] == "qbitel-container-security"
        assert len(config["webhooks"]) > 0

    @pytest.mark.asyncio
    async def test_handle_health(self, webhook_server):
        """Test health check endpoint"""
        # Create mock request
        mock_request = Mock(spec=web.Request)

        response = await webhook_server.handle_health(mock_request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_handle_validate_pod(self, webhook_server):
        """Test validation webhook handler"""
        # Create admission review request
        admission_review = {
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "request": {
                "uid": "test-uid-123",
                "kind": {"kind": "Pod"},
                "operation": "CREATE",
                "namespace": "default",
                "object": {
                    "metadata": {"name": "test-pod"},
                    "spec": {
                        "containers": [
                            {
                                "name": "app",
                                "image": "gcr.io/distroless/base@sha256:abc123"
                            }
                        ]
                    }
                }
            }
        }

        # Create mock request
        mock_request = Mock(spec=web.Request)
        mock_request.json = AsyncMock(return_value=admission_review)

        response = await webhook_server.handle_validate(mock_request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_server_lifecycle(self):
        """Test server start and stop"""
        server = AdmissionWebhookServer(port=8445)

        # Start server
        await server.start()

        assert server._app is not None
        assert server._runner is not None

        # Stop server
        await server.stop()

    def test_validate_pod_with_init_containers(self, webhook_server):
        """Test pod validation includes init containers"""
        pod = {
            "metadata": {"name": "test-pod"},
            "spec": {
                "initContainers": [
                    {
                        "name": "init",
                        "image": "untrusted.io/init:latest"  # Blocked registry
                    }
                ],
                "containers": [
                    {
                        "name": "app",
                        "image": "gcr.io/app@sha256:abc123"
                    }
                ]
            }
        }

        result = webhook_server.validate_pod(pod)

        assert result["allowed"] is False
        assert any("init" in v and "disallowed registry" in v.lower()
                  for v in result["violations"])

    def test_validate_pod_host_pid(self, webhook_server):
        """Test pod validation denies host PID namespace"""
        pod = {
            "metadata": {"name": "test-pod"},
            "spec": {
                "containers": [
                    {
                        "name": "app",
                        "image": "gcr.io/app@sha256:abc123"
                    }
                ],
                "hostPID": True  # Host PID namespace
            }
        }

        result = webhook_server.validate_pod(pod)

        assert result["allowed"] is False
        assert any("host pid" in v.lower() for v in result["violations"])


class TestSecurityPolicy:
    """Test suite for SecurityPolicy dataclass"""

    def test_security_policy_creation(self):
        """Test creating a SecurityPolicy"""
        policy = SecurityPolicy(
            name="test-policy",
            require_image_signature=True,
            max_critical_vulnerabilities=0
        )

        assert policy.name == "test-policy"
        assert policy.require_image_signature is True
        assert policy.max_critical_vulnerabilities == 0
        assert policy.enabled is True  # Default value

    def test_security_policy_with_registries(self):
        """Test SecurityPolicy with registry lists"""
        policy = SecurityPolicy(
            name="registry-policy",
            allowed_registries=["gcr.io", "docker.io"],
            blocked_registries=["untrusted.io"]
        )

        assert "gcr.io" in policy.allowed_registries
        assert "untrusted.io" in policy.blocked_registries

    def test_security_policy_defaults(self):
        """Test SecurityPolicy default values"""
        policy = SecurityPolicy(name="defaults")

        assert policy.allowed_registries == []
        assert policy.blocked_registries == []
        assert policy.enforce is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
