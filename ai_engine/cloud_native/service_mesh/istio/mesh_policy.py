"""
Mesh Policy Manager

Manages service mesh security policies including traffic policies,
authorization rules, and network policies for Istio.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TrafficDirection(Enum):
    """Traffic direction for policies"""
    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"
    BOTH = "BOTH"


class PolicyAction(Enum):
    """Policy enforcement actions"""
    ALLOW = "ALLOW"
    DENY = "DENY"
    AUDIT = "AUDIT"
    CUSTOM = "CUSTOM"


@dataclass
class ServicePolicy:
    """Service-level security policy"""
    service_name: str
    namespace: str
    allowed_sources: List[str]
    allowed_operations: List[str]
    require_mtls: bool = True
    require_jwt: bool = False
    rate_limit: Optional[int] = None


class MeshPolicyManager:
    """
    Manages Istio service mesh security policies.

    Provides centralized policy management for traffic control,
    authentication, authorization, and rate limiting.
    """

    def __init__(self, namespace: str = "qbitel-system"):
        """
        Initialize the Mesh Policy Manager.

        Args:
            namespace: Kubernetes namespace for QBITEL
        """
        self.namespace = namespace
        self._policies: Dict[str, ServicePolicy] = {}
        logger.info(f"Initialized MeshPolicyManager in namespace: {namespace}")

    def create_traffic_policy(
        self,
        name: str,
        target_host: str,
        namespace: str = "default",
        load_balancer: str = "ROUND_ROBIN",
        enable_circuit_breaker: bool = True,
        enable_retry: bool = True
    ) -> Dict[str, Any]:
        """
        Create traffic management policy (DestinationRule).

        Args:
            name: Policy name
            target_host: Target service host
            namespace: Target namespace
            load_balancer: Load balancing algorithm
            enable_circuit_breaker: Enable circuit breaker
            enable_retry: Enable automatic retries

        Returns:
            Dict containing DestinationRule manifest
        """
        dest_rule = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "DestinationRule",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "qbitel",
                    "managed-by": "mesh-policy-manager"
                }
            },
            "spec": {
                "host": target_host,
                "trafficPolicy": {
                    "loadBalancer": {
                        "simple": load_balancer
                    }
                }
            }
        }

        # Add circuit breaker if enabled
        if enable_circuit_breaker:
            dest_rule["spec"]["trafficPolicy"]["outlierDetection"] = {
                "consecutiveErrors": 5,
                "interval": "30s",
                "baseEjectionTime": "30s",
                "maxEjectionPercent": 50,
                "minHealthPercent": 40
            }

        # Add retry policy if enabled
        if enable_retry:
            dest_rule["spec"]["trafficPolicy"]["connectionPool"] = {
                "http": {
                    "http1MaxPendingRequests": 100,
                    "http2MaxRequests": 100,
                    "maxRequestsPerConnection": 2,
                    "maxRetries": 3
                }
            }

        logger.info(f"Created traffic policy {name} for {target_host}")
        return dest_rule

    def create_rate_limit_policy(
        self,
        name: str,
        namespace: str,
        selector: Dict[str, str],
        requests_per_second: int,
        burst: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create rate limiting policy using Envoy filter.

        Args:
            name: Policy name
            namespace: Target namespace
            selector: Workload selector labels
            requests_per_second: Maximum requests per second
            burst: Burst allowance (defaults to 2x requests_per_second)

        Returns:
            Dict containing EnvoyFilter manifest
        """
        if burst is None:
            burst = requests_per_second * 2

        envoy_filter = {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "EnvoyFilter",
            "metadata": {
                "name": f"{name}-rate-limit",
                "namespace": namespace,
                "labels": {
                    "app": "qbitel",
                    "component": "rate-limiter"
                }
            },
            "spec": {
                "workloadSelector": {
                    "labels": selector
                },
                "configPatches": [
                    {
                        "applyTo": "HTTP_FILTER",
                        "match": {
                            "context": "SIDECAR_INBOUND",
                            "listener": {
                                "filterChain": {
                                    "filter": {
                                        "name": "envoy.filters.network.http_connection_manager"
                                    }
                                }
                            }
                        },
                        "patch": {
                            "operation": "INSERT_BEFORE",
                            "value": {
                                "name": "envoy.filters.http.local_ratelimit",
                                "typed_config": {
                                    "@type": "type.googleapis.com/envoy.extensions.filters.http.local_ratelimit.v3.LocalRateLimit",
                                    "stat_prefix": "http_local_rate_limiter",
                                    "token_bucket": {
                                        "max_tokens": burst,
                                        "tokens_per_fill": requests_per_second,
                                        "fill_interval": "1s"
                                    },
                                    "filter_enabled": {
                                        "runtime_key": "local_rate_limit_enabled",
                                        "default_value": {
                                            "numerator": 100,
                                            "denominator": "HUNDRED"
                                        }
                                    },
                                    "filter_enforced": {
                                        "runtime_key": "local_rate_limit_enforced",
                                        "default_value": {
                                            "numerator": 100,
                                            "denominator": "HUNDRED"
                                        }
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        }

        logger.info(
            f"Created rate limit policy {name}: "
            f"{requests_per_second} req/s, burst {burst}"
        )
        return envoy_filter

    def create_service_authorization(
        self,
        service: ServicePolicy
    ) -> Dict[str, Any]:
        """
        Create authorization policy for a service.

        Args:
            service: Service policy configuration

        Returns:
            Dict containing AuthorizationPolicy manifest
        """
        rules = []

        # Create rules for each allowed source
        for source in service.allowed_sources:
            rule = {
                "from": [
                    {
                        "source": {
                            "principals": [
                                f"cluster.local/ns/{service.namespace}/sa/{source}"
                            ]
                        }
                    }
                ]
            }

            # Add operation restrictions if specified
            if service.allowed_operations:
                rule["to"] = [
                    {
                        "operation": {
                            "methods": service.allowed_operations
                        }
                    }
                ]

            rules.append(rule)

        auth_policy = {
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "AuthorizationPolicy",
            "metadata": {
                "name": f"{service.service_name}-authz",
                "namespace": service.namespace,
                "labels": {
                    "app": "qbitel",
                    "service": service.service_name
                }
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "app": service.service_name
                    }
                },
                "action": "ALLOW",
                "rules": rules
            }
        }

        logger.info(f"Created authorization policy for {service.service_name}")
        return auth_policy

    def create_jwt_authentication(
        self,
        name: str,
        namespace: str,
        issuer: str,
        jwks_uri: str,
        selector: Optional[Dict[str, str]] = None,
        audiences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create JWT authentication policy.

        Args:
            name: Policy name
            namespace: Target namespace
            issuer: JWT issuer
            jwks_uri: JWKS endpoint URI
            selector: Workload selector labels
            audiences: List of valid audiences

        Returns:
            Dict containing RequestAuthentication manifest
        """
        req_auth = {
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "RequestAuthentication",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "qbitel",
                    "component": "jwt-auth"
                }
            },
            "spec": {
                "jwtRules": [
                    {
                        "issuer": issuer,
                        "jwksUri": jwks_uri
                    }
                ]
            }
        }

        # Add selector if provided
        if selector:
            req_auth["spec"]["selector"] = {
                "matchLabels": selector
            }

        # Add audiences if provided
        if audiences:
            req_auth["spec"]["jwtRules"][0]["audiences"] = audiences

        logger.info(f"Created JWT authentication policy {name}")
        return req_auth

    def create_egress_policy(
        self,
        name: str,
        namespace: str,
        hosts: List[str],
        ports: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Create egress traffic policy (ServiceEntry).

        Args:
            name: Policy name
            namespace: Source namespace
            hosts: List of external hosts to allow
            ports: List of allowed ports

        Returns:
            Dict containing ServiceEntry manifest
        """
        service_entry = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "ServiceEntry",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "qbitel",
                    "component": "egress-policy"
                }
            },
            "spec": {
                "hosts": hosts,
                "location": "MESH_EXTERNAL",
                "resolution": "DNS"
            }
        }

        # Add ports if specified
        if ports:
            service_entry["spec"]["ports"] = [
                {
                    "number": port,
                    "name": f"https-{port}",
                    "protocol": "HTTPS"
                }
                for port in ports
            ]

        logger.info(f"Created egress policy {name} for hosts: {hosts}")
        return service_entry

    def create_sidecar_config(
        self,
        name: str,
        namespace: str,
        egress_hosts: Optional[List[str]] = None,
        ingress_listeners: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create Sidecar configuration for fine-grained traffic control.

        Args:
            name: Sidecar config name
            namespace: Target namespace
            egress_hosts: List of allowed egress hosts
            ingress_listeners: Custom ingress listener configurations

        Returns:
            Dict containing Sidecar manifest
        """
        sidecar = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "Sidecar",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "qbitel",
                    "component": "sidecar-config"
                }
            },
            "spec": {
                "egress": [
                    {
                        "hosts": egress_hosts or [
                            "./*",  # Current namespace
                            "istio-system/*"  # Istio system services
                        ]
                    }
                ]
            }
        }

        # Add custom ingress listeners if provided
        if ingress_listeners:
            sidecar["spec"]["ingress"] = ingress_listeners

        logger.info(f"Created sidecar configuration {name}")
        return sidecar

    def create_network_policy(
        self,
        name: str,
        namespace: str,
        pod_selector: Dict[str, str],
        ingress_rules: Optional[List[Dict[str, Any]]] = None,
        egress_rules: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create Kubernetes NetworkPolicy for additional security layer.

        Args:
            name: Policy name
            namespace: Target namespace
            pod_selector: Pod selector labels
            ingress_rules: Ingress rule specifications
            egress_rules: Egress rule specifications

        Returns:
            Dict containing NetworkPolicy manifest
        """
        network_policy = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "qbitel",
                    "managed-by": "mesh-policy-manager"
                }
            },
            "spec": {
                "podSelector": {
                    "matchLabels": pod_selector
                },
                "policyTypes": []
            }
        }

        # Add ingress rules
        if ingress_rules:
            network_policy["spec"]["ingress"] = ingress_rules
            network_policy["spec"]["policyTypes"].append("Ingress")

        # Add egress rules
        if egress_rules:
            network_policy["spec"]["egress"] = egress_rules
            network_policy["spec"]["policyTypes"].append("Egress")

        logger.info(f"Created network policy {name}")
        return network_policy

    def register_service_policy(self, policy: ServicePolicy):
        """
        Register a service policy for management.

        Args:
            policy: Service policy to register
        """
        key = f"{policy.namespace}/{policy.service_name}"
        self._policies[key] = policy
        logger.info(f"Registered service policy: {key}")

    def get_service_policy(
        self,
        service_name: str,
        namespace: str
    ) -> Optional[ServicePolicy]:
        """
        Retrieve registered service policy.

        Args:
            service_name: Service name
            namespace: Namespace

        Returns:
            ServicePolicy or None if not found
        """
        key = f"{namespace}/{service_name}"
        return self._policies.get(key)

    def create_complete_service_policies(
        self,
        policy: ServicePolicy
    ) -> List[Dict[str, Any]]:
        """
        Create all necessary policies for a service.

        Args:
            policy: Service policy configuration

        Returns:
            List of policy manifests
        """
        manifests = []

        # 1. Authorization policy
        auth_policy = self.create_service_authorization(policy)
        manifests.append(auth_policy)

        # 2. Rate limiting (if specified)
        if policy.rate_limit:
            rate_limit = self.create_rate_limit_policy(
                name=policy.service_name,
                namespace=policy.namespace,
                selector={"app": policy.service_name},
                requests_per_second=policy.rate_limit
            )
            manifests.append(rate_limit)

        # 3. Traffic policy with circuit breaker
        traffic_policy = self.create_traffic_policy(
            name=f"{policy.service_name}-traffic",
            target_host=f"{policy.service_name}.{policy.namespace}.svc.cluster.local",
            namespace=policy.namespace
        )
        manifests.append(traffic_policy)

        # Register the policy
        self.register_service_policy(policy)

        logger.info(
            f"Created {len(manifests)} policies for service {policy.service_name}"
        )
        return manifests
