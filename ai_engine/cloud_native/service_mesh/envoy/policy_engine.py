"""
Traffic Policy Engine

Manages traffic policies for Envoy including rate limiting, circuit breaking,
retries, and timeout policies for east-west traffic.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """Traffic policy types"""
    RATE_LIMIT = "rate_limit"
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY = "retry"
    TIMEOUT = "timeout"
    LOAD_BALANCER = "load_balancer"


@dataclass
class TrafficPolicy:
    """Traffic policy configuration"""
    name: str
    policy_type: PolicyType
    target_service: str
    config: Dict[str, Any]
    enabled: bool = True


class TrafficPolicyEngine:
    """
    Manages traffic policies for Envoy proxy.

    Provides centralized policy management for rate limiting, circuit breaking,
    retries, timeouts, and load balancing.
    """

    def __init__(self):
        """Initialize traffic policy engine"""
        self._policies: Dict[str, TrafficPolicy] = {}
        logger.info("Initialized TrafficPolicyEngine")

    def create_rate_limit_policy(
        self,
        service_name: str,
        requests_per_second: int,
        burst: Optional[int] = None,
        per_connection: bool = False
    ) -> TrafficPolicy:
        """Create rate limiting policy"""
        config = {
            "domain": service_name,
            "descriptors": [
                {
                    "entries": [
                        {"key": "header_match", "value": service_name}
                    ],
                    "rate_limit": {
                        "unit": "SECOND",
                        "requests_per_unit": requests_per_second
                    }
                }
            ],
            "token_bucket": {
                "max_tokens": burst or (requests_per_second * 2),
                "tokens_per_fill": requests_per_second,
                "fill_interval": "1s"
            },
            "per_connection_buffer_limit_bytes": 1048576 if per_connection else None
        }

        policy = TrafficPolicy(
            name=f"{service_name}_rate_limit",
            policy_type=PolicyType.RATE_LIMIT,
            target_service=service_name,
            config=config
        )

        self._policies[policy.name] = policy
        logger.info(f"Created rate limit policy: {requests_per_second} req/s for {service_name}")
        return policy

    def create_circuit_breaker_policy(
        self,
        service_name: str,
        max_connections: int = 1024,
        max_pending_requests: int = 1024,
        max_requests: int = 1024,
        max_retries: int = 3,
        max_connection_pools: int = 1024
    ) -> TrafficPolicy:
        """Create circuit breaker policy"""
        config = {
            "thresholds": [
                {
                    "priority": "DEFAULT",
                    "max_connections": max_connections,
                    "max_pending_requests": max_pending_requests,
                    "max_requests": max_requests,
                    "max_retries": max_retries,
                    "max_connection_pools": max_connection_pools,
                    "track_remaining": True
                },
                {
                    "priority": "HIGH",
                    "max_connections": max_connections * 2,
                    "max_pending_requests": max_pending_requests * 2,
                    "max_requests": max_requests * 2,
                    "max_retries": max_retries,
                    "track_remaining": True
                }
            ]
        }

        policy = TrafficPolicy(
            name=f"{service_name}_circuit_breaker",
            policy_type=PolicyType.CIRCUIT_BREAKER,
            target_service=service_name,
            config=config
        )

        self._policies[policy.name] = policy
        logger.info(f"Created circuit breaker policy for {service_name}")
        return policy

    def create_retry_policy(
        self,
        service_name: str,
        num_retries: int = 3,
        retry_on: Optional[List[str]] = None,
        per_try_timeout: str = "5s",
        retry_host_predicate: Optional[List[str]] = None
    ) -> TrafficPolicy:
        """Create retry policy"""
        config = {
            "retry_on": ",".join(retry_on or [
                "5xx",
                "gateway-error",
                "reset",
                "connect-failure",
                "refused-stream"
            ]),
            "num_retries": num_retries,
            "per_try_timeout": per_try_timeout,
            "retry_priority": {
                "name": "envoy.retry_priorities.previous_priorities",
                "typed_config": {
                    "@type": "type.googleapis.com/envoy.extensions.retry.priority.previous_priorities.v3.PreviousPrioritiesConfig",
                    "update_frequency": 2
                }
            },
            "retry_host_predicate": [
                {
                    "name": "envoy.retry_host_predicates.previous_hosts",
                    "typed_config": {
                        "@type": "type.googleapis.com/envoy.extensions.retry.host.previous_hosts.v3.PreviousHostsPredicate"
                    }
                }
            ] if not retry_host_predicate else retry_host_predicate
        }

        policy = TrafficPolicy(
            name=f"{service_name}_retry",
            policy_type=PolicyType.RETRY,
            target_service=service_name,
            config=config
        )

        self._policies[policy.name] = policy
        logger.info(f"Created retry policy ({num_retries} retries) for {service_name}")
        return policy

    def create_timeout_policy(
        self,
        service_name: str,
        request_timeout: str = "30s",
        idle_timeout: str = "300s",
        stream_idle_timeout: str = "300s"
    ) -> TrafficPolicy:
        """Create timeout policy"""
        config = {
            "request_timeout": request_timeout,
            "idle_timeout": idle_timeout,
            "stream_idle_timeout": stream_idle_timeout,
            "max_stream_duration": {
                "max_stream_duration": "300s",
                "grpc_timeout_header_max": "300s"
            }
        }

        policy = TrafficPolicy(
            name=f"{service_name}_timeout",
            policy_type=PolicyType.TIMEOUT,
            target_service=service_name,
            config=config
        )

        self._policies[policy.name] = policy
        logger.info(f"Created timeout policy for {service_name}")
        return policy

    def create_load_balancer_policy(
        self,
        service_name: str,
        lb_policy: str = "ROUND_ROBIN",
        locality_weighted_lb: bool = False,
        health_check_config: Optional[Dict[str, Any]] = None
    ) -> TrafficPolicy:
        """Create load balancer policy"""
        config = {
            "lb_policy": lb_policy,
            "locality_weighted_lb_config": {} if locality_weighted_lb else None,
            "health_checks": [health_check_config] if health_check_config else []
        }

        policy = TrafficPolicy(
            name=f"{service_name}_lb",
            policy_type=PolicyType.LOAD_BALANCER,
            target_service=service_name,
            config=config
        )

        self._policies[policy.name] = policy
        logger.info(f"Created load balancer policy ({lb_policy}) for {service_name}")
        return policy

    def get_policy(self, name: str) -> Optional[TrafficPolicy]:
        """Get policy by name"""
        return self._policies.get(name)

    def get_policies_for_service(self, service_name: str) -> List[TrafficPolicy]:
        """Get all policies for a service"""
        return [
            policy for policy in self._policies.values()
            if policy.target_service == service_name
        ]

    def generate_envoy_config(self, service_name: str) -> Dict[str, Any]:
        """Generate complete Envoy configuration for service"""
        policies = self.get_policies_for_service(service_name)

        config = {
            "cluster_name": service_name,
            "circuit_breakers": {},
            "outlier_detection": {},
            "common_lb_config": {},
            "http_protocol_options": {}
        }

        for policy in policies:
            if policy.policy_type == PolicyType.CIRCUIT_BREAKER:
                config["circuit_breakers"] = policy.config
            elif policy.policy_type == PolicyType.LOAD_BALANCER:
                config["lb_policy"] = policy.config["lb_policy"]

        return config
