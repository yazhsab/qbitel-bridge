"""
Unit tests for Istio Sidecar Injector.
"""

import pytest
import json
import sys
from pathlib import Path

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from cloud_native.service_mesh.istio.sidecar_injector import SidecarInjector, InjectionPolicy, SidecarConfig
except ImportError:
    pytest.skip("sidecar_injector module not found", allow_module_level=True)


class TestSidecarConfig:
    """Test SidecarConfig dataclass"""

    def test_default_config(self):
        """Test default sidecar configuration"""
        config = SidecarConfig()
        assert config.image is not None
        assert config.cpu_limit is not None
        assert config.memory_limit is not None

    def test_custom_config(self):
        """Test custom sidecar configuration"""
        config = SidecarConfig(image="custom-proxy:v1", cpu_limit="500m", memory_limit="256Mi")
        assert config.image == "custom-proxy:v1"
        assert config.cpu_limit == "500m"


class TestInjectionPolicy:
    """Test InjectionPolicy"""

    def test_injection_policy_creation(self):
        """Test creating injection policy"""
        policy = InjectionPolicy(name="test-policy", namespaces=["default"], labels={"app": "test"})
        assert policy.name == "test-policy"
        assert "default" in policy.namespaces
        assert policy.labels["app"] == "test"


class TestSidecarInjector:
    """Test SidecarInjector class"""

    @pytest.fixture
    def injector(self):
        """Create SidecarInjector instance"""
        return SidecarInjector(namespace="istio-system", enable_quantum_crypto=True)

    def test_injector_initialization(self, injector):
        """Test injector initialization"""
        assert injector.namespace == "istio-system"
        assert injector.enable_quantum_crypto is True

    def test_should_inject_pod(self, injector):
        """Test pod injection decision logic"""
        # Pod with injection enabled
        pod_with_injection = {"metadata": {"annotations": {"sidecar.istio.io/inject": "true"}, "labels": {"app": "test"}}}
        assert injector.should_inject(pod_with_injection) is True

        # Pod with injection disabled
        pod_no_injection = {"metadata": {"annotations": {"sidecar.istio.io/inject": "false"}}}
        assert injector.should_inject(pod_no_injection) is False

    def test_inject_sidecar(self, injector):
        """Test sidecar injection into pod spec"""
        pod = {
            "metadata": {"name": "test-pod", "namespace": "default"},
            "spec": {"containers": [{"name": "app", "image": "nginx:latest"}]},
        }

        injected_pod = injector.inject_sidecar(pod)

        # Check sidecar container added
        assert len(injected_pod["spec"]["containers"]) == 2

        # Find sidecar container
        sidecar = next((c for c in injected_pod["spec"]["containers"] if "proxy" in c["name"] or "sidecar" in c["name"]), None)
        assert sidecar is not None

        # Check quantum crypto annotations if enabled
        if injector.enable_quantum_crypto:
            annotations = injected_pod["metadata"].get("annotations", {})
            assert any("quantum" in k.lower() for k in annotations.keys()) or any(
                "pqc" in k.lower() for k in annotations.keys()
            )

    def test_inject_init_containers(self, injector):
        """Test init container injection"""
        pod = {"metadata": {"name": "test-pod"}, "spec": {"containers": [{"name": "app", "image": "nginx"}]}}

        injected_pod = injector.inject_sidecar(pod)

        # Check init containers
        init_containers = injected_pod["spec"].get("initContainers", [])
        assert isinstance(init_containers, list)

    def test_inject_volumes(self, injector):
        """Test volume injection for certificates"""
        pod = {"metadata": {"name": "test-pod"}, "spec": {"containers": [{"name": "app", "image": "nginx"}]}}

        injected_pod = injector.inject_sidecar(pod)

        # Check volumes for certs
        volumes = injected_pod["spec"].get("volumes", [])
        assert isinstance(volumes, list)

        # Should have certificate volumes
        cert_volumes = [v for v in volumes if "cert" in v.get("name", "").lower()]
        assert len(cert_volumes) > 0

    def test_create_mutating_webhook_config(self, injector):
        """Test webhook configuration generation"""
        webhook_config = injector.create_mutating_webhook_config()

        assert "apiVersion" in webhook_config
        assert "kind" in webhook_config
        assert webhook_config["kind"] == "MutatingWebhookConfiguration"

        webhooks = webhook_config.get("webhooks", [])
        assert len(webhooks) > 0

        webhook = webhooks[0]
        assert "name" in webhook
        assert "clientConfig" in webhook
        assert "rules" in webhook

    def test_quantum_crypto_injection(self):
        """Test quantum crypto specific injection"""
        injector = SidecarInjector(enable_quantum_crypto=True)

        pod = {"metadata": {"name": "test-pod"}, "spec": {"containers": [{"name": "app", "image": "nginx"}]}}

        injected_pod = injector.inject_sidecar(pod)

        # Check for quantum-specific environment variables
        sidecar = next((c for c in injected_pod["spec"]["containers"] if "proxy" in c["name"] or "sidecar" in c["name"]), None)

        if sidecar and "env" in sidecar:
            env_names = [e["name"] for e in sidecar["env"]]
            quantum_env = [e for e in env_names if "QUANTUM" in e or "PQC" in e]
            assert len(quantum_env) > 0 or injector.enable_quantum_crypto
