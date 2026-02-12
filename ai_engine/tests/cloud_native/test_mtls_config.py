"""
Unit tests for Istio Mutual TLS Configuration.
"""

import pytest
import sys
from pathlib import Path

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.service_mesh.istio.mtls_config import (
    MTLSMode,
    MTLSPolicy,
    MutualTLSConfigurator,
    create_istio_mtls_config_map,
)


class TestMTLSMode:
    """Test MTLSMode enum"""

    def test_mtls_modes(self):
        """Test MTLS mode values"""
        assert MTLSMode.STRICT is not None
        assert MTLSMode.PERMISSIVE is not None
        assert MTLSMode.DISABLE is not None


class TestMTLSPolicy:
    """Test MTLSPolicy dataclass"""

    def test_policy_creation(self):
        """Test creating mTLS policy"""
        policy = MTLSPolicy(name="test-policy", namespace="default", mode=MTLSMode.STRICT, selector={"app": "test"})

        assert policy.name == "test-policy"
        assert policy.namespace == "default"
        assert policy.mode == MTLSMode.STRICT
        assert policy.selector["app"] == "test"


class TestMutualTLSConfigurator:
    """Test MutualTLSConfigurator class"""

    @pytest.fixture
    def configurator(self):
        """Create MutualTLSConfigurator instance"""
        return MutualTLSConfigurator(default_mode=MTLSMode.STRICT, namespace="istio-system")

    def test_configurator_initialization(self, configurator):
        """Test configurator initialization"""
        assert configurator.default_mode == MTLSMode.STRICT
        assert configurator.namespace == "istio-system"

    def test_create_peer_authentication(self, configurator):
        """Test PeerAuthentication resource creation"""
        peer_auth = configurator.create_peer_authentication(
            name="test-auth", namespace="default", mode=MTLSMode.STRICT, selector={"app": "test"}
        )

        assert peer_auth["apiVersion"] == "security.istio.io/v1beta1"
        assert peer_auth["kind"] == "PeerAuthentication"
        assert peer_auth["metadata"]["name"] == "test-auth"
        assert peer_auth["metadata"]["namespace"] == "default"

        spec = peer_auth["spec"]
        assert spec["mtls"]["mode"] == "STRICT"
        assert spec["selector"]["matchLabels"]["app"] == "test"

    def test_create_destination_rule(self, configurator):
        """Test DestinationRule creation"""
        dest_rule = configurator.create_destination_rule(
            name="test-rule", host="test-service.default.svc.cluster.local", namespace="default", enable_quantum_tls=True
        )

        assert dest_rule["apiVersion"] == "networking.istio.io/v1beta1"
        assert dest_rule["kind"] == "DestinationRule"
        assert dest_rule["metadata"]["name"] == "test-rule"
        assert dest_rule["spec"]["host"] == "test-service.default.svc.cluster.local"

        # Check TLS settings
        tls = dest_rule["spec"]["trafficPolicy"]["tls"]
        assert tls["mode"] == "ISTIO_MUTUAL"

        # Check quantum-safe cipher suites if enabled
        if configurator.default_mode == MTLSMode.STRICT:
            cipher_suites = tls.get("cipherSuites", [])
            assert "TLS_AES_256_GCM_SHA384" in cipher_suites

    def test_create_authorization_policy(self, configurator):
        """Test AuthorizationPolicy creation"""
        rules = [
            {
                "from": [{"source": {"principals": ["cluster.local/ns/default/sa/frontend"]}}],
                "to": [{"operation": {"methods": ["GET"]}}],
            }
        ]

        auth_policy = configurator.create_authorization_policy(
            name="test-authz", namespace="default", action="ALLOW", rules=rules, selector={"app": "backend"}
        )

        assert auth_policy["apiVersion"] == "security.istio.io/v1beta1"
        assert auth_policy["kind"] == "AuthorizationPolicy"
        assert auth_policy["spec"]["action"] == "ALLOW"
        assert len(auth_policy["spec"]["rules"]) == 1

    def test_create_mesh_wide_mtls(self, configurator):
        """Test mesh-wide mTLS policy"""
        mesh_policy = configurator.create_mesh_wide_mtls(MTLSMode.STRICT)

        assert mesh_policy["metadata"]["name"] == "default"
        assert mesh_policy["metadata"]["namespace"] == configurator.namespace
        assert mesh_policy["spec"]["mtls"]["mode"] == "STRICT"

        # Mesh-wide policy should not have selector
        assert "selector" not in mesh_policy["spec"]

    def test_create_namespace_mtls(self, configurator):
        """Test namespace-scoped mTLS"""
        namespace_policy = configurator.create_namespace_mtls(namespace="production", mode=MTLSMode.STRICT)

        assert namespace_policy["metadata"]["namespace"] == "production"
        assert namespace_policy["spec"]["mtls"]["mode"] == "STRICT"

    def test_create_workload_mtls(self, configurator):
        """Test workload-specific mTLS"""
        workload_policy = configurator.create_workload_mtls(
            name="payment-service", namespace="production", labels={"app": "payment", "version": "v1"}, mode=MTLSMode.STRICT
        )

        assert workload_policy["metadata"]["name"] == "payment-service"
        assert workload_policy["spec"]["selector"]["matchLabels"]["app"] == "payment"
        assert workload_policy["spec"]["mtls"]["mode"] == "STRICT"

    def test_create_zero_trust_policy(self, configurator):
        """Test zero-trust policy creation"""
        allowed_services = ["frontend", "api-gateway"]

        zero_trust = configurator.create_zero_trust_policy(namespace="production", allowed_services=allowed_services)

        # Should create both PeerAuthentication and AuthorizationPolicy
        assert "peer_authentication" in zero_trust
        assert "authorization_policy" in zero_trust

        # Check strict mTLS
        peer_auth = zero_trust["peer_authentication"]
        assert peer_auth["spec"]["mtls"]["mode"] == "STRICT"

        # Check default DENY policy
        authz_policy = zero_trust["authorization_policy"]
        assert authz_policy["spec"]["action"] == "DENY"

    def test_verify_mtls_configuration(self, configurator):
        """Test mTLS configuration verification"""
        # This would typically check if mTLS is properly configured
        # For unit tests, we verify the method exists and returns expected format
        result = configurator.verify_mtls_configuration(source_service="frontend", dest_service="backend", namespace="default")

        assert isinstance(result, dict)
        assert "mtls_enabled" in result
        assert "mode" in result

    def test_permissive_mode(self):
        """Test permissive mode configuration"""
        configurator = MutualTLSConfigurator(default_mode=MTLSMode.PERMISSIVE)

        peer_auth = configurator.create_peer_authentication(
            name="permissive-auth", namespace="default", mode=MTLSMode.PERMISSIVE
        )

        assert peer_auth["spec"]["mtls"]["mode"] == "PERMISSIVE"

    def test_disable_mtls(self):
        """Test disabling mTLS"""
        configurator = MutualTLSConfigurator(default_mode=MTLSMode.DISABLE)

        peer_auth = configurator.create_peer_authentication(name="no-mtls", namespace="default", mode=MTLSMode.DISABLE)

        assert peer_auth["spec"]["mtls"]["mode"] == "DISABLE"

    def test_quantum_tls_cipher_suites(self, configurator):
        """Test quantum-safe cipher suite configuration"""
        dest_rule = configurator.create_destination_rule(
            name="quantum-rule", host="secure-service", namespace="default", enable_quantum_tls=True
        )

        tls = dest_rule["spec"]["trafficPolicy"]["tls"]
        cipher_suites = tls.get("cipherSuites", [])

        # Check for quantum-resistant ciphers
        expected_ciphers = ["TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256", "TLS_AES_128_GCM_SHA256"]

        for cipher in expected_ciphers:
            assert cipher in cipher_suites

    def test_subset_configurations(self, configurator):
        """Test destination rule with subsets"""
        subset_configs = [{"name": "v1", "labels": {"version": "v1"}}, {"name": "v2", "labels": {"version": "v2"}}]

        dest_rule = configurator.create_destination_rule(
            name="versioned-service", host="my-service", namespace="default", subset_configs=subset_configs
        )

        subsets = dest_rule["spec"].get("subsets", [])
        assert len(subsets) == 2
        assert subsets[0]["name"] == "v1"
        assert subsets[1]["name"] == "v2"


class TestConfigMapHelper:
    """Test ConfigMap helper function"""

    def test_create_istio_mtls_config_map(self):
        """Test Istio mTLS ConfigMap generation"""
        config_map = create_istio_mtls_config_map(namespace="istio-system")

        assert config_map["apiVersion"] == "v1"
        assert config_map["kind"] == "ConfigMap"
        assert config_map["metadata"]["namespace"] == "istio-system"
        assert "data" in config_map
