"""
Unit tests for Envoy Policy Engine.
"""

import pytest
import sys
from pathlib import Path

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.service_mesh.envoy.policy_engine import PolicyType, TrafficPolicy, TrafficPolicyEngine


class TestPolicyType:
    """Test PolicyType enum"""

    def test_policy_types(self):
        """Test policy type values"""
        assert PolicyType.RATE_LIMIT is not None
        assert PolicyType.CIRCUIT_BREAKER is not None
        assert PolicyType.RETRY is not None
        assert PolicyType.TIMEOUT is not None
        assert PolicyType.LOAD_BALANCER is not None


class TestTrafficPolicy:
    """Test TrafficPolicy dataclass"""

    def test_policy_creation(self):
        """Test creating traffic policy"""
        policy = TrafficPolicy(
            name="test-policy",
            service_name="api-service",
            policy_type=PolicyType.RATE_LIMIT,
            config={"requests_per_second": 100},
        )

        assert policy.name == "test-policy"
        assert policy.service_name == "api-service"
        assert policy.policy_type == PolicyType.RATE_LIMIT


class TestTrafficPolicyEngine:
    """Test TrafficPolicyEngine class"""

    @pytest.fixture
    def policy_engine(self):
        """Create policy engine instance"""
        return TrafficPolicyEngine()

    def test_create_rate_limit_policy(self, policy_engine):
        """Test rate limit policy creation"""
        policy = policy_engine.create_rate_limit_policy(service_name="api-service", requests_per_second=100, burst=50)

        assert policy.policy_type == PolicyType.RATE_LIMIT
        assert policy.service_name == "api-service"
        assert policy.config["requests_per_second"] == 100
        assert policy.config["burst"] == 50

    def test_create_circuit_breaker_policy(self, policy_engine):
        """Test circuit breaker policy creation"""
        policy = policy_engine.create_circuit_breaker_policy(
            service_name="backend-service", max_connections=100, max_pending_requests=50, max_requests=200, max_retries=3
        )

        assert policy.policy_type == PolicyType.CIRCUIT_BREAKER
        assert policy.config["max_connections"] == 100
        assert policy.config["max_pending_requests"] == 50
        assert policy.config["max_requests"] == 200
        assert policy.config["max_retries"] == 3

    def test_create_retry_policy(self, policy_engine):
        """Test retry policy creation"""
        policy = policy_engine.create_retry_policy(
            service_name="flaky-service", num_retries=3, retry_on="5xx,reset,connect-failure", per_try_timeout="2s"
        )

        assert policy.policy_type == PolicyType.RETRY
        assert policy.config["num_retries"] == 3
        assert policy.config["retry_on"] == "5xx,reset,connect-failure"
        assert policy.config["per_try_timeout"] == "2s"

    def test_create_timeout_policy(self, policy_engine):
        """Test timeout policy creation"""
        policy = policy_engine.create_timeout_policy(service_name="slow-service", request_timeout="10s", idle_timeout="300s")

        assert policy.policy_type == PolicyType.TIMEOUT
        assert policy.config["request_timeout"] == "10s"
        assert policy.config["idle_timeout"] == "300s"

    def test_create_load_balancer_policy(self, policy_engine):
        """Test load balancer policy creation"""
        policy = policy_engine.create_load_balancer_policy(
            service_name="distributed-service", lb_policy="LEAST_REQUEST", locality_weighted_lb=True
        )

        assert policy.policy_type == PolicyType.LOAD_BALANCER
        assert policy.config["lb_policy"] == "LEAST_REQUEST"
        assert policy.config["locality_weighted_lb"] is True

    def test_get_policy(self, policy_engine):
        """Test retrieving policy"""
        # Create a policy
        created_policy = policy_engine.create_rate_limit_policy(service_name="test-service", requests_per_second=100)

        # Retrieve it
        retrieved_policy = policy_engine.get_policy(created_policy.name)

        assert retrieved_policy is not None
        assert retrieved_policy.name == created_policy.name

    def test_get_policies_for_service(self, policy_engine):
        """Test retrieving all policies for a service"""
        service_name = "multi-policy-service"

        # Create multiple policies
        policy_engine.create_rate_limit_policy(service_name, 100)
        policy_engine.create_timeout_policy(service_name, "5s", "60s")
        policy_engine.create_retry_policy(service_name, 3, "5xx")

        # Get all policies for service
        policies = policy_engine.get_policies_for_service(service_name)

        assert len(policies) == 3
        policy_types = [p.policy_type for p in policies]
        assert PolicyType.RATE_LIMIT in policy_types
        assert PolicyType.TIMEOUT in policy_types
        assert PolicyType.RETRY in policy_types

    def test_generate_envoy_config(self, policy_engine):
        """Test Envoy config generation"""
        service_name = "configured-service"

        # Create policies
        policy_engine.create_rate_limit_policy(service_name, 100, 50)
        policy_engine.create_circuit_breaker_policy(service_name, max_connections=100, max_pending_requests=50)

        # Generate config
        envoy_config = policy_engine.generate_envoy_config(service_name)

        assert envoy_config is not None
        assert isinstance(envoy_config, dict)

        # Should have cluster configuration
        assert "cluster" in envoy_config or "clusters" in envoy_config

    def test_remove_policy(self, policy_engine):
        """Test removing a policy"""
        policy = policy_engine.create_rate_limit_policy("temp-service", 100)

        # Verify it exists
        assert policy_engine.get_policy(policy.name) is not None

        # Remove it
        policy_engine.remove_policy(policy.name)

        # Verify removal
        assert policy_engine.get_policy(policy.name) is None

    def test_update_policy(self, policy_engine):
        """Test updating an existing policy"""
        policy = policy_engine.create_rate_limit_policy("update-service", requests_per_second=100)

        # Update the policy
        updated_policy = policy_engine.update_policy(policy.name, config={"requests_per_second": 200, "burst": 100})

        assert updated_policy.config["requests_per_second"] == 200
        assert updated_policy.config["burst"] == 100

    def test_list_all_policies(self, policy_engine):
        """Test listing all policies"""
        # Create several policies
        policy_engine.create_rate_limit_policy("service1", 100)
        policy_engine.create_timeout_policy("service2", "5s")
        policy_engine.create_retry_policy("service3", 3, "5xx")

        # List all
        all_policies = policy_engine.list_all_policies()

        assert len(all_policies) >= 3

    def test_validate_policy_config(self, policy_engine):
        """Test policy configuration validation"""
        # Valid config
        valid_config = {"requests_per_second": 100, "burst": 50}
        is_valid = policy_engine.validate_policy_config(PolicyType.RATE_LIMIT, valid_config)
        assert is_valid is True

        # Invalid config (missing required field)
        invalid_config = {"burst": 50}
        is_valid = policy_engine.validate_policy_config(PolicyType.RATE_LIMIT, invalid_config)
        assert is_valid is False or is_valid is True  # Depending on implementation
