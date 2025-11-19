"""
Integration tests for Envoy xDS Server

Tests the production-ready gRPC xDS server implementation.
"""

import pytest
import asyncio
import grpc
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from ai_engine.cloud_native.service_mesh.envoy.xds_server import (
    EnvoyXDSServer,
    XDSResourceType,
    EnvoyNode,
    XDSResource
)


class TestEnvoyXDSServer:
    """Test suite for Envoy xDS Server"""

    @pytest.fixture
    async def xds_server(self):
        """Create xDS server instance for testing"""
        server = EnvoyXDSServer(port=18001, enable_ads=True)
        yield server
        # Cleanup
        if server._server:
            await server.stop()

    @pytest.mark.asyncio
    async def test_server_initialization(self, xds_server):
        """Test server initializes correctly"""
        assert xds_server.port == 18001
        assert xds_server.enable_ads is True
        assert xds_server._version_counter == 0
        assert len(xds_server._listeners) == 0
        assert len(xds_server._clusters) == 0

    @pytest.mark.asyncio
    async def test_create_listener(self, xds_server):
        """Test creating a listener resource"""
        listener = await xds_server.create_listener(
            name="test_listener",
            address="0.0.0.0",
            port=8080,
            enable_quantum_filter=True
        )

        assert listener.name == "test_listener"
        assert listener.resource_type == XDSResourceType.LISTENER
        assert listener.config["address"]["socket_address"]["port_value"] == 8080
        assert "filter_chains" in listener.config

    @pytest.mark.asyncio
    async def test_create_cluster(self, xds_server):
        """Test creating a cluster resource"""
        endpoints = [
            {"address": "192.168.1.1", "port_value": 8080},
            {"address": "192.168.1.2", "port_value": 8080}
        ]

        cluster = await xds_server.create_cluster(
            name="test_cluster",
            endpoints=endpoints,
            lb_policy="ROUND_ROBIN",
            enable_quantum_tls=True
        )

        assert cluster.name == "test_cluster"
        assert cluster.resource_type == XDSResourceType.CLUSTER
        assert cluster.config["lb_policy"] == "ROUND_ROBIN"
        assert "transport_socket" in cluster.config

    @pytest.mark.asyncio
    async def test_register_node(self, xds_server):
        """Test registering an Envoy node"""
        node = EnvoyNode(
            id="test-node-1",
            cluster="test-cluster",
            metadata={"region": "us-west-1"}
        )

        xds_server.register_node(node)

        assert "test-node-1" in xds_server._nodes
        assert xds_server._nodes["test-node-1"].cluster == "test-cluster"

    @pytest.mark.asyncio
    async def test_subscribe_resources(self, xds_server):
        """Test resource subscription"""
        node_id = "test-node-1"
        resources = ["listener1", "listener2"]

        xds_server.subscribe(node_id, resources)

        assert node_id in xds_server._subscriptions
        assert len(xds_server._subscriptions[node_id]) == 2
        assert "listener1" in xds_server._subscriptions[node_id]

    @pytest.mark.asyncio
    async def test_get_resources_for_node(self, xds_server):
        """Test retrieving resources for a node"""
        # Create test resources
        await xds_server.create_listener(
            name="test_listener",
            address="0.0.0.0",
            port=8080
        )

        # Subscribe node to resources
        node_id = "test-node-1"
        xds_server.subscribe(node_id, ["test_listener"])

        # Get resources
        resources = xds_server.get_resources_for_node(
            node_id,
            XDSResourceType.LISTENER
        )

        assert len(resources) == 1
        assert resources[0].name == "test_listener"

    @pytest.mark.asyncio
    async def test_update_listener(self, xds_server):
        """Test updating listener configuration"""
        # Create initial listener
        await xds_server.create_listener(
            name="test_listener",
            address="0.0.0.0",
            port=8080
        )

        initial_version = xds_server._listeners["test_listener"].version

        # Update listener
        updated = await xds_server.update_listener(
            "test_listener",
            {"updated": True}
        )

        assert updated is not None
        assert updated.version != initial_version
        assert updated.config["updated"] is True

    @pytest.mark.asyncio
    async def test_delete_resource(self, xds_server):
        """Test deleting a resource"""
        # Create listener
        await xds_server.create_listener(
            name="test_listener",
            address="0.0.0.0",
            port=8080
        )

        assert "test_listener" in xds_server._listeners

        # Delete listener
        deleted = await xds_server.delete_resource(
            XDSResourceType.LISTENER,
            "test_listener"
        )

        assert deleted is True
        assert "test_listener" not in xds_server._listeners

    @pytest.mark.asyncio
    async def test_get_snapshot(self, xds_server):
        """Test getting server snapshot"""
        # Create some resources
        await xds_server.create_listener(
            name="listener1",
            address="0.0.0.0",
            port=8080
        )

        await xds_server.create_cluster(
            name="cluster1",
            endpoints=[{"address": "192.168.1.1", "port_value": 8080}]
        )

        snapshot = xds_server.get_snapshot()

        assert "listeners" in snapshot
        assert "clusters" in snapshot
        assert len(snapshot["listeners"]) == 1
        assert len(snapshot["clusters"]) == 1
        assert snapshot["version"] == str(xds_server._version_counter)

    @pytest.mark.asyncio
    async def test_get_statistics(self, xds_server):
        """Test getting server statistics"""
        # Create resources
        await xds_server.create_listener(
            name="listener1",
            address="0.0.0.0",
            port=8080
        )

        # Register node
        node = EnvoyNode(id="node1", cluster="cluster1", metadata={})
        xds_server.register_node(node)

        stats = xds_server.get_statistics()

        assert stats["listeners"] == 1
        assert stats["connected_nodes"] == 1
        assert "version" in stats

    @pytest.mark.asyncio
    async def test_quantum_filter_chain(self, xds_server):
        """Test quantum-safe filter chain creation"""
        filter_chain = xds_server._create_quantum_filter_chain()

        assert len(filter_chain) > 0
        assert "filters" in filter_chain[0]

        # Check for quantum encryption filter
        http_filters = filter_chain[0]["filters"][0]["typed_config"]["http_filters"]
        quantum_filter = next(
            (f for f in http_filters if "quantum_encryption" in f["name"]),
            None
        )

        assert quantum_filter is not None
        assert "kyber-1024" in quantum_filter["typed_config"]["key_algorithm"]

    @pytest.mark.asyncio
    async def test_quantum_transport_socket(self, xds_server):
        """Test quantum-safe TLS transport socket"""
        transport_socket = xds_server._create_quantum_transport_socket()

        assert transport_socket["name"] == "envoy.transport_sockets.tls"
        assert "common_tls_context" in transport_socket["typed_config"]

        tls_context = transport_socket["typed_config"]["common_tls_context"]
        assert tls_context["tls_params"]["tls_minimum_protocol_version"] == "TLSv1_3"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not hasattr(EnvoyXDSServer, '_server'),
        reason="gRPC server not available"
    )
    async def test_grpc_server_start_stop(self):
        """Test gRPC server lifecycle"""
        server = EnvoyXDSServer(port=18002)

        # Start server in background
        start_task = asyncio.create_task(server.start())

        # Wait a bit for server to start
        await asyncio.sleep(0.5)

        # Check server is running
        assert server._running is True
        assert server._server is not None

        # Stop server
        await server.stop()
        assert server._running is False

        # Cancel the start task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass

    def test_version_increment(self, xds_server):
        """Test version counter increments correctly"""
        initial_version = xds_server._version_counter

        v1 = xds_server._get_next_version()
        assert int(v1) == initial_version + 1

        v2 = xds_server._get_next_version()
        assert int(v2) == initial_version + 2

    @pytest.mark.asyncio
    async def test_create_secret(self, xds_server):
        """Test creating a secret resource"""
        secret = await xds_server.create_secret(
            name="test_cert",
            certificate="-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----",
            private_key="-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----"
        )

        assert secret.name == "test_cert"
        assert secret.resource_type == XDSResourceType.SECRET
        assert "tls_certificate" in secret.config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
