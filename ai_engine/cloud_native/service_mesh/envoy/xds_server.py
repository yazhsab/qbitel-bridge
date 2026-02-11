"""
Envoy xDS Server

Implements the Envoy xDS (Discovery Service) protocol for dynamic configuration
of quantum-safe encryption filters and traffic policies.
"""

import asyncio
import logging
import hashlib
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import grpc
from concurrent import futures

logger = logging.getLogger(__name__)

# Import Envoy xDS protocol buffers
try:
    from envoy.service.discovery.v3 import discovery_pb2, discovery_pb2_grpc
    from envoy.config.listener.v3 import listener_pb2
    from envoy.config.cluster.v3 import cluster_pb2
    from envoy.config.route.v3 import route_pb2
    from envoy.config.endpoint.v3 import endpoint_pb2
    from envoy.extensions.transport_sockets.tls.v3 import secret_pb2
    from google.protobuf import any_pb2
    ENVOY_PROTOS_AVAILABLE = True
except ImportError:
    ENVOY_PROTOS_AVAILABLE = False
    # Create mock classes for when protos are not available
    class MockServicer:
        pass
    discovery_pb2_grpc = type('discovery_pb2_grpc', (), {
        'AggregatedDiscoveryServiceServicer': MockServicer
    })()
    any_pb2 = type('any_pb2', (), {'Any': dict})()
    logger.warning("Envoy protobuf packages not available. Install with: pip install xds-protos grpcio-tools")


class XDSResourceType(Enum):
    """Envoy xDS resource types"""
    LISTENER = "type.googleapis.com/envoy.config.listener.v3.Listener"
    CLUSTER = "type.googleapis.com/envoy.config.cluster.v3.Cluster"
    ROUTE = "type.googleapis.com/envoy.config.route.v3.RouteConfiguration"
    ENDPOINT = "type.googleapis.com/envoy.config.endpoint.v3.ClusterLoadAssignment"
    SECRET = "type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.Secret"


@dataclass
class XDSResource:
    """Represents an xDS resource"""
    name: str
    version: str
    resource_type: XDSResourceType
    config: Dict[str, Any]
    last_updated: float = field(default_factory=time.time)


@dataclass
class EnvoyNode:
    """Represents an Envoy proxy node"""
    id: str
    cluster: str
    metadata: Dict[str, Any]
    locality: Optional[Dict[str, str]] = None
    build_version: str = ""


class EnvoyXDSServer(discovery_pb2_grpc.AggregatedDiscoveryServiceServicer if ENVOY_PROTOS_AVAILABLE else object):
    """
    Envoy xDS (Discovery Service) Server for dynamic configuration.

    Implements the xDS protocol to dynamically configure Envoy proxies with
    quantum-safe encryption filters, traffic policies, and service discovery.
    """

    def __init__(
        self,
        port: int = 18000,
        enable_ads: bool = True,
        cache_ttl: int = 300
    ):
        """
        Initialize the Envoy xDS server.

        Args:
            port: Port for xDS gRPC server
            enable_ads: Enable Aggregated Discovery Service
            cache_ttl: Cache TTL in seconds
        """
        self.port = port
        self.enable_ads = enable_ads
        self.cache_ttl = cache_ttl

        # Resource storage
        self._listeners: Dict[str, XDSResource] = {}
        self._clusters: Dict[str, XDSResource] = {}
        self._routes: Dict[str, XDSResource] = {}
        self._endpoints: Dict[str, XDSResource] = {}
        self._secrets: Dict[str, XDSResource] = {}

        # Connected nodes
        self._nodes: Dict[str, EnvoyNode] = {}

        # Subscriptions (node_id -> resource_names)
        self._subscriptions: Dict[str, Set[str]] = {}

        # Version tracking
        self._version_counter = 0

        # gRPC server
        self._server: Optional[grpc.aio.Server] = None
        self._running = False

        logger.info(f"Initialized EnvoyXDSServer on port {port}")

    async def start(self):
        """Start the gRPC xDS server"""
        if not ENVOY_PROTOS_AVAILABLE:
            logger.error("Cannot start xDS server: Envoy protobuf packages not installed")
            raise RuntimeError("Envoy protobuf packages required. Install with: pip install envoy-data-plane-api")

        logger.info(f"Starting gRPC xDS server on port {self.port}")

        # Initialize default resources
        await self._initialize_default_resources()

        # Create gRPC server with thread pool
        self._server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
            ]
        )

        # Register ADS service
        discovery_pb2_grpc.add_AggregatedDiscoveryServiceServicer_to_server(
            self, self._server
        )

        # Add insecure port (in production, use TLS)
        self._server.add_insecure_port(f'[::]:{self.port}')

        # Start server
        await self._server.start()
        self._running = True

        logger.info(f"gRPC xDS server started on port {self.port}")
        logger.info("Envoy proxies can now connect for dynamic configuration")

        # Wait for termination
        try:
            await self._server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Shutting down xDS server...")
            await self.stop()

    async def stop(self):
        """Stop the gRPC xDS server"""
        if self._server and self._running:
            logger.info("Stopping xDS server...")
            await self._server.stop(grace=5.0)
            self._running = False
            logger.info("xDS server stopped")

    async def StreamAggregatedResources(
        self,
        request_iterator,
        context: grpc.aio.ServicerContext
    ):
        """
        Handle bidirectional streaming for ADS (Aggregated Discovery Service).

        This is the main xDS protocol handler that Envoy proxies use to get
        their dynamic configuration.
        """
        node_id = None

        try:
            async for request in request_iterator:
                # Extract node information
                if request.node and request.node.id:
                    node_id = request.node.id
                    if node_id not in self._nodes:
                        # Register new node
                        node = EnvoyNode(
                            id=node_id,
                            cluster=request.node.cluster or "unknown",
                            metadata=dict(request.node.metadata) if request.node.metadata else {},
                            build_version=request.node.user_agent_version or ""
                        )
                        self.register_node(node)

                # Determine resource type
                type_url = request.type_url
                resource_names = list(request.resource_names) if request.resource_names else []

                logger.info(f"xDS request from node {node_id}: {type_url}, resources: {resource_names}")

                # Get resources based on type
                resources = self._get_resources_by_type_url(type_url, node_id, resource_names)

                # Build discovery response
                response = discovery_pb2.DiscoveryResponse(
                    version_info=str(self._version_counter),
                    resources=resources,
                    type_url=type_url,
                    nonce=self._generate_nonce()
                )

                # Send response
                yield response

                logger.info(f"Sent {len(resources)} resources to node {node_id}")

        except Exception as e:
            logger.error(f"Error in StreamAggregatedResources for node {node_id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    def _get_resources_by_type_url(
        self,
        type_url: str,
        node_id: Optional[str],
        resource_names: List[str]
    ) -> List[Any]:
        """
        Get resources by type URL and serialize them.

        Args:
            type_url: xDS resource type URL
            node_id: Node requesting resources
            resource_names: Specific resource names requested

        Returns:
            List of serialized resources as Any protobufs
        """
        resources = []

        # Map type URL to resource store
        resource_map = {
            XDSResourceType.LISTENER.value: self._listeners,
            XDSResourceType.CLUSTER.value: self._clusters,
            XDSResourceType.ROUTE.value: self._routes,
            XDSResourceType.ENDPOINT.value: self._endpoints,
            XDSResourceType.SECRET.value: self._secrets,
        }

        resource_store = resource_map.get(type_url, {})

        # Filter by requested names or return all
        if resource_names:
            selected_resources = [
                res for name, res in resource_store.items()
                if name in resource_names
            ]
        else:
            selected_resources = list(resource_store.values())

        # Serialize resources to Any protos
        for resource in selected_resources:
            any_resource = any_pb2.Any()
            # In production, you would serialize the actual protobuf message
            # For now, we'll pack the JSON config
            any_resource.type_url = type_url
            any_resource.value = json.dumps(resource.config).encode('utf-8')
            resources.append(any_resource)

        return resources

    def _generate_nonce(self) -> str:
        """Generate a unique nonce for xDS response"""
        import uuid
        return str(uuid.uuid4())

    async def _initialize_default_resources(self):
        """Initialize default xDS resources"""
        # Create default listener with quantum-safe filter
        await self.create_listener(
            name="quantum_listener",
            address="0.0.0.0",
            port=15001,
            enable_quantum_filter=True
        )

        logger.info("Default xDS resources initialized")

    async def create_listener(
        self,
        name: str,
        address: str,
        port: int,
        enable_quantum_filter: bool = True,
        filter_chains: Optional[List[Dict[str, Any]]] = None
    ) -> XDSResource:
        """
        Create an Envoy listener resource.

        Args:
            name: Listener name
            address: Bind address
            port: Bind port
            enable_quantum_filter: Enable quantum-safe encryption filter
            filter_chains: Custom filter chain configurations

        Returns:
            XDSResource for the listener
        """
        listener_config = {
            "name": name,
            "address": {
                "socket_address": {
                    "address": address,
                    "port_value": port
                }
            },
            "filter_chains": filter_chains or self._create_quantum_filter_chain()
        }

        version = self._get_next_version()
        resource = XDSResource(
            name=name,
            version=version,
            resource_type=XDSResourceType.LISTENER,
            config=listener_config
        )

        self._listeners[name] = resource
        logger.info(f"Created listener {name} on {address}:{port} (v{version})")

        return resource

    def _create_quantum_filter_chain(self) -> List[Dict[str, Any]]:
        """Create filter chain with quantum-safe encryption"""
        return [
            {
                "filters": [
                    {
                        "name": "envoy.filters.network.http_connection_manager",
                        "typed_config": {
                            "@type": "type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager",
                            "stat_prefix": "ingress_http",
                            "codec_type": "AUTO",
                            "route_config": {
                                "name": "local_route",
                                "virtual_hosts": [
                                    {
                                        "name": "local_service",
                                        "domains": ["*"],
                                        "routes": [
                                            {
                                                "match": {"prefix": "/"},
                                                "route": {"cluster": "local_service"}
                                            }
                                        ]
                                    }
                                ]
                            },
                            "http_filters": [
                                {
                                    "name": "qbitel.filters.http.quantum_encryption",
                                    "typed_config": {
                                        "@type": "type.googleapis.com/qbitel.extensions.filters.http.quantum_encryption.v3.QuantumEncryption",
                                        "key_algorithm": "kyber-1024",
                                        "signature_algorithm": "dilithium-5",
                                        "enable_metrics": True,
                                        "metrics_prefix": "quantum_encryption"
                                    }
                                },
                                {
                                    "name": "envoy.filters.http.router",
                                    "typed_config": {
                                        "@type": "type.googleapis.com/envoy.extensions.filters.http.router.v3.Router"
                                    }
                                }
                            ]
                        }
                    }
                ],
                "transport_socket": {
                    "name": "envoy.transport_sockets.tls",
                    "typed_config": {
                        "@type": "type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.DownstreamTlsContext",
                        "common_tls_context": {
                            "tls_params": {
                                "tls_minimum_protocol_version": "TLSv1_3",
                                "tls_maximum_protocol_version": "TLSv1_3",
                                "cipher_suites": [
                                    "TLS_AES_256_GCM_SHA384",
                                    "TLS_CHACHA20_POLY1305_SHA256"
                                ]
                            },
                            "tls_certificate_sds_secret_configs": [
                                {
                                    "name": "qbitel_quantum_cert",
                                    "sds_config": {
                                        "api_config_source": {
                                            "api_type": "GRPC",
                                            "grpc_services": [
                                                {
                                                    "envoy_grpc": {
                                                        "cluster_name": "sds_cluster"
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        ]

    async def create_cluster(
        self,
        name: str,
        endpoints: List[Dict[str, Any]],
        lb_policy: str = "ROUND_ROBIN",
        enable_quantum_tls: bool = True,
        circuit_breaker: Optional[Dict[str, Any]] = None
    ) -> XDSResource:
        """
        Create an Envoy cluster resource.

        Args:
            name: Cluster name
            endpoints: List of endpoint configurations
            lb_policy: Load balancing policy (ROUND_ROBIN, LEAST_REQUEST, etc.)
            enable_quantum_tls: Enable quantum-safe TLS
            circuit_breaker: Circuit breaker configuration

        Returns:
            XDSResource for the cluster
        """
        cluster_config = {
            "name": name,
            "type": "STRICT_DNS",
            "lb_policy": lb_policy,
            "load_assignment": {
                "cluster_name": name,
                "endpoints": [
                    {
                        "lb_endpoints": [
                            {
                                "endpoint": {
                                    "address": {
                                        "socket_address": ep
                                    }
                                }
                            }
                            for ep in endpoints
                        ]
                    }
                ]
            },
            "connect_timeout": "5s"
        }

        # Add circuit breaker if specified
        if circuit_breaker:
            cluster_config["circuit_breakers"] = circuit_breaker
        else:
            cluster_config["circuit_breakers"] = {
                "thresholds": [
                    {
                        "priority": "DEFAULT",
                        "max_connections": 10000,
                        "max_pending_requests": 10000,
                        "max_requests": 10000,
                        "max_retries": 3
                    }
                ]
            }

        # Add quantum-safe TLS if enabled
        if enable_quantum_tls:
            cluster_config["transport_socket"] = self._create_quantum_transport_socket()

        version = self._get_next_version()
        resource = XDSResource(
            name=name,
            version=version,
            resource_type=XDSResourceType.CLUSTER,
            config=cluster_config
        )

        self._clusters[name] = resource
        logger.info(f"Created cluster {name} with {len(endpoints)} endpoints (v{version})")

        return resource

    def _create_quantum_transport_socket(self) -> Dict[str, Any]:
        """Create quantum-safe TLS transport socket configuration"""
        return {
            "name": "envoy.transport_sockets.tls",
            "typed_config": {
                "@type": "type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext",
                "common_tls_context": {
                    "tls_params": {
                        "tls_minimum_protocol_version": "TLSv1_3",
                        "tls_maximum_protocol_version": "TLSv1_3",
                        "cipher_suites": [
                            "TLS_AES_256_GCM_SHA384",
                            "TLS_CHACHA20_POLY1305_SHA256"
                        ]
                    },
                    "tls_certificate_sds_secret_configs": [
                        {
                            "name": "qbitel_quantum_client_cert",
                            "sds_config": {
                                "api_config_source": {
                                    "api_type": "GRPC",
                                    "grpc_services": [
                                        {
                                            "envoy_grpc": {
                                                "cluster_name": "sds_cluster"
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    ],
                    "validation_context_sds_secret_config": {
                        "name": "qbitel_quantum_validation_context",
                        "sds_config": {
                            "api_config_source": {
                                "api_type": "GRPC",
                                "grpc_services": [
                                    {
                                        "envoy_grpc": {
                                            "cluster_name": "sds_cluster"
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }

    async def create_route(
        self,
        name: str,
        virtual_hosts: List[Dict[str, Any]]
    ) -> XDSResource:
        """
        Create an Envoy route configuration.

        Args:
            name: Route configuration name
            virtual_hosts: List of virtual host configurations

        Returns:
            XDSResource for the route
        """
        route_config = {
            "name": name,
            "virtual_hosts": virtual_hosts
        }

        version = self._get_next_version()
        resource = XDSResource(
            name=name,
            version=version,
            resource_type=XDSResourceType.ROUTE,
            config=route_config
        )

        self._routes[name] = resource
        logger.info(f"Created route {name} with {len(virtual_hosts)} virtual hosts (v{version})")

        return resource

    async def create_secret(
        self,
        name: str,
        certificate: str,
        private_key: str,
        certificate_chain: Optional[str] = None
    ) -> XDSResource:
        """
        Create an Envoy secret resource for quantum-safe certificates.

        Args:
            name: Secret name
            certificate: PEM-encoded certificate
            private_key: PEM-encoded private key
            certificate_chain: Optional certificate chain

        Returns:
            XDSResource for the secret
        """
        secret_config = {
            "name": name,
            "tls_certificate": {
                "certificate_chain": {
                    "inline_string": certificate_chain or certificate
                },
                "private_key": {
                    "inline_string": private_key
                }
            }
        }

        version = self._get_next_version()
        resource = XDSResource(
            name=name,
            version=version,
            resource_type=XDSResourceType.SECRET,
            config=secret_config
        )

        self._secrets[name] = resource
        logger.info(f"Created secret {name} (v{version})")

        return resource

    def register_node(self, node: EnvoyNode):
        """
        Register an Envoy proxy node.

        Args:
            node: EnvoyNode to register
        """
        self._nodes[node.id] = node
        self._subscriptions[node.id] = set()
        logger.info(f"Registered Envoy node {node.id} in cluster {node.cluster}")

    def subscribe(self, node_id: str, resource_names: List[str]):
        """
        Subscribe a node to specific resources.

        Args:
            node_id: Node ID
            resource_names: List of resource names to subscribe to
        """
        if node_id not in self._subscriptions:
            self._subscriptions[node_id] = set()

        self._subscriptions[node_id].update(resource_names)
        logger.info(f"Node {node_id} subscribed to {len(resource_names)} resources")

    def get_resources_for_node(
        self,
        node_id: str,
        resource_type: XDSResourceType
    ) -> List[XDSResource]:
        """
        Get resources for a specific node and type.

        Args:
            node_id: Node ID
            resource_type: Type of resources to retrieve

        Returns:
            List of XDSResource objects
        """
        if node_id not in self._subscriptions:
            return []

        # Get the appropriate resource store
        resource_store = {
            XDSResourceType.LISTENER: self._listeners,
            XDSResourceType.CLUSTER: self._clusters,
            XDSResourceType.ROUTE: self._routes,
            XDSResourceType.ENDPOINT: self._endpoints,
            XDSResourceType.SECRET: self._secrets
        }.get(resource_type, {})

        # Filter by subscriptions
        subscribed_names = self._subscriptions[node_id]
        if not subscribed_names:
            # If no specific subscriptions, return all
            return list(resource_store.values())

        return [
            resource for name, resource in resource_store.items()
            if name in subscribed_names
        ]

    def _get_next_version(self) -> str:
        """Generate next version number"""
        self._version_counter += 1
        return str(self._version_counter)

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get current snapshot of all xDS resources.

        Returns:
            Dict containing all resources
        """
        return {
            "listeners": {name: res.config for name, res in self._listeners.items()},
            "clusters": {name: res.config for name, res in self._clusters.items()},
            "routes": {name: res.config for name, res in self._routes.items()},
            "endpoints": {name: res.config for name, res in self._endpoints.items()},
            "secrets": {name: res.config for name, res in self._secrets.items()},
            "version": str(self._version_counter),
            "nodes": len(self._nodes)
        }

    async def update_listener(
        self,
        name: str,
        config_updates: Dict[str, Any]
    ) -> Optional[XDSResource]:
        """
        Update an existing listener configuration.

        Args:
            name: Listener name
            config_updates: Configuration updates to apply

        Returns:
            Updated XDSResource or None if not found
        """
        if name not in self._listeners:
            logger.warning(f"Listener {name} not found for update")
            return None

        listener = self._listeners[name]
        listener.config.update(config_updates)
        listener.version = self._get_next_version()
        listener.last_updated = time.time()

        logger.info(f"Updated listener {name} to version {listener.version}")
        return listener

    async def delete_resource(
        self,
        resource_type: XDSResourceType,
        name: str
    ) -> bool:
        """
        Delete a resource.

        Args:
            resource_type: Type of resource
            name: Resource name

        Returns:
            True if deleted, False if not found
        """
        resource_store = {
            XDSResourceType.LISTENER: self._listeners,
            XDSResourceType.CLUSTER: self._clusters,
            XDSResourceType.ROUTE: self._routes,
            XDSResourceType.ENDPOINT: self._endpoints,
            XDSResourceType.SECRET: self._secrets
        }.get(resource_type)

        if resource_store and name in resource_store:
            del resource_store[name]
            logger.info(f"Deleted {resource_type.value} resource {name}")
            return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get xDS server statistics.

        Returns:
            Dict containing server statistics
        """
        return {
            "listeners": len(self._listeners),
            "clusters": len(self._clusters),
            "routes": len(self._routes),
            "endpoints": len(self._endpoints),
            "secrets": len(self._secrets),
            "connected_nodes": len(self._nodes),
            "total_subscriptions": sum(len(subs) for subs in self._subscriptions.values()),
            "version": self._version_counter,
            "uptime_seconds": time.time()  # Would need actual start time
        }
