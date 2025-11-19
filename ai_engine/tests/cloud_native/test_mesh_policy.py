"""
Unit tests for Istio Mesh Policy management.
"""
import pytest
import sys
from pathlib import Path

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from cloud_native.service_mesh.istio.mesh_policy import (
        MeshPolicyManager,
        PolicyType,
        TrafficPolicy
    )
except ImportError:
    pytest.skip("mesh_policy module not found", allow_module_level=True)


class TestPolicyType:
    """Test PolicyType enum"""

    def test_policy_types(self):
        """Test available policy types"""
        assert PolicyType.RATE_LIMIT is not None
        assert PolicyType.CIRCUIT_BREAKER is not None
        assert PolicyType.RETRY is not None
        assert PolicyType.TIMEOUT is not None


class TestTrafficPolicy:
    """Test TrafficPolicy dataclass"""

    def test_policy_creation(self):
        """Test creating traffic policy"""
        policy = TrafficPolicy(
            name="test-policy",
            policy_type=PolicyType.RATE_LIMIT,
            config={"requests_per_second": 100}
        )

        assert policy.name == "test-policy"
        assert policy.policy_type == PolicyType.RATE_LIMIT
        assert policy.config["requests_per_second"] == 100


class TestMeshPolicyManager:
    """Test MeshPolicyManager class"""

    @pytest.fixture
    def policy_manager(self):
        """Create MeshPolicyManager instance"""
        return MeshPolicyManager(namespace="istio-system")

    def test_manager_initialization(self, policy_manager):
        """Test manager initialization"""
        assert policy_manager.namespace == "istio-system"
        assert hasattr(policy_manager, "policies")

    def test_create_rate_limit_policy(self, policy_manager):
        """Test rate limit policy creation"""
        policy = policy_manager.create_rate_limit_policy(
            name="api-rate-limit",
            service="api-service",
            requests_per_second=100,
            burst=50
        )

        assert policy["kind"] == "EnvoyFilter"
        assert "api-rate-limit" in policy["metadata"]["name"]

    def test_create_circuit_breaker_policy(self, policy_manager):
        """Test circuit breaker policy creation"""
        policy = policy_manager.create_circuit_breaker_policy(
            name="backend-cb",
            service="backend-service",
            max_connections=100,
            max_pending_requests=50,
            max_requests=200,
            consecutive_errors=5
        )

        assert policy["kind"] == "DestinationRule"
        assert "backend-cb" in policy["metadata"]["name"]

        # Check circuit breaker settings
        traffic_policy = policy["spec"]["trafficPolicy"]
        assert "connectionPool" in traffic_policy
        assert "outlierDetection" in traffic_policy

    def test_create_retry_policy(self, policy_manager):
        """Test retry policy creation"""
        policy = policy_manager.create_retry_policy(
            name="service-retry",
            service="flaky-service",
            num_retries=3,
            retry_on="5xx,reset,connect-failure",
            per_try_timeout="2s"
        )

        assert policy["kind"] == "VirtualService"
        http_routes = policy["spec"]["http"]
        assert len(http_routes) > 0

        retry_config = http_routes[0]["retries"]
        assert retry_config["attempts"] == 3
        assert retry_config["perTryTimeout"] == "2s"

    def test_create_timeout_policy(self, policy_manager):
        """Test timeout policy creation"""
        policy = policy_manager.create_timeout_policy(
            name="service-timeout",
            service="slow-service",
            request_timeout="10s",
            idle_timeout="300s"
        )

        assert policy["kind"] == "VirtualService"
        http_routes = policy["spec"]["http"]
        assert http_routes[0]["timeout"] == "10s"

    def test_add_policy(self, policy_manager):
        """Test adding policy to manager"""
        policy = TrafficPolicy(
            name="test-policy",
            policy_type=PolicyType.RATE_LIMIT,
            config={"requests_per_second": 100}
        )

        policy_manager.add_policy(policy)
        retrieved = policy_manager.get_policy("test-policy")

        assert retrieved == policy

    def test_remove_policy(self, policy_manager):
        """Test removing policy"""
        policy = TrafficPolicy(
            name="temp-policy",
            policy_type=PolicyType.TIMEOUT,
            config={"timeout": "5s"}
        )

        policy_manager.add_policy(policy)
        assert policy_manager.get_policy("temp-policy") is not None

        policy_manager.remove_policy("temp-policy")
        assert policy_manager.get_policy("temp-policy") is None

    def test_list_policies(self, policy_manager):
        """Test listing all policies"""
        policies = [
            TrafficPolicy("policy1", PolicyType.RATE_LIMIT, {}),
            TrafficPolicy("policy2", PolicyType.RETRY, {}),
            TrafficPolicy("policy3", PolicyType.CIRCUIT_BREAKER, {})
        ]

        for policy in policies:
            policy_manager.add_policy(policy)

        all_policies = policy_manager.list_policies()
        assert len(all_policies) >= 3

    def test_fault_injection_policy(self, policy_manager):
        """Test fault injection policy"""
        policy = policy_manager.create_fault_injection_policy(
            name="chaos-test",
            service="test-service",
            delay_percent=10,
            delay_duration="5s",
            abort_percent=5,
            abort_status=503
        )

        assert policy["kind"] == "VirtualService"
        http_routes = policy["spec"]["http"]
        fault = http_routes[0]["fault"]

        assert fault["delay"]["percentage"]["value"] == 10
        assert fault["delay"]["fixedDelay"] == "5s"
        assert fault["abort"]["percentage"]["value"] == 5
        assert fault["abort"]["httpStatus"] == 503

    def test_traffic_mirroring_policy(self, policy_manager):
        """Test traffic mirroring policy"""
        policy = policy_manager.create_traffic_mirror_policy(
            name="mirror-test",
            service="production-service",
            mirror_service="test-service",
            mirror_percentage=10
        )

        assert policy["kind"] == "VirtualService"
        http_routes = policy["spec"]["http"]

        mirror = http_routes[0]["mirror"]
        assert mirror["host"] == "test-service"
        assert http_routes[0]["mirrorPercentage"]["value"] == 10

    def test_canary_deployment_policy(self, policy_manager):
        """Test canary deployment policy"""
        policy = policy_manager.create_canary_policy(
            name="canary-deploy",
            service="my-service",
            stable_version="v1",
            canary_version="v2",
            canary_weight=20
        )

        assert policy["kind"] == "VirtualService"
        http_routes = policy["spec"]["http"]
        destinations = http_routes[0]["route"]

        # Check weight distribution
        v1_dest = next(d for d in destinations if d["destination"]["subset"] == "v1")
        v2_dest = next(d for d in destinations if d["destination"]["subset"] == "v2")

        assert v1_dest["weight"] == 80
        assert v2_dest["weight"] == 20

    def test_header_based_routing(self, policy_manager):
        """Test header-based routing policy"""
        policy = policy_manager.create_header_routing_policy(
            name="header-route",
            service="api-service",
            header_name="x-api-version",
            header_value="v2",
            target_version="v2"
        )

        assert policy["kind"] == "VirtualService"
        http_routes = policy["spec"]["http"]

        # Check match condition
        match = http_routes[0]["match"][0]
        assert match["headers"]["x-api-version"]["exact"] == "v2"

        # Check destination
        assert http_routes[0]["route"][0]["destination"]["subset"] == "v2"
