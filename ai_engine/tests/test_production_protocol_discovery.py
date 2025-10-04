"""
CRONOS AI Engine - Production Protocol Discovery Tests

Integration tests for production protocol discovery system.
"""

import pytest
import asyncio
import time
from typing import List

from ..discovery.production_protocol_discovery import (
    ProductionProtocolDiscovery,
    DiscoveryRequest,
    DiscoveryResult,
    QualityMode,
    DeploymentType,
    ExplainableDiscovery,
    ModelVersionManager,
)
from ..core.config import Config
from ..core.exceptions import SLAViolationException


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
async def discovery_system(config):
    """Create and initialize discovery system."""
    system = ProductionProtocolDiscovery(config)
    await system.initialize()
    return system


@pytest.fixture
def sample_packet_data():
    """Generate sample packet data."""
    return b"\x47\x45\x54\x20\x2f\x20\x48\x54\x54\x50\x2f\x31\x2e\x31"  # GET / HTTP/1.1


class TestSLADiscovery:
    """Test SLA-aware discovery functionality."""

    @pytest.mark.asyncio
    async def test_sla_met(self, discovery_system, sample_packet_data):
        """Test successful SLA compliance."""
        request = DiscoveryRequest(
            request_id="test_sla_met",
            packet_data=sample_packet_data,
            quality_mode=QualityMode.FAST,
        )

        result = await discovery_system.discover_with_sla(request, sla_ms=1000)

        assert result is not None
        assert result.request_id == "test_sla_met"
        assert result.processing_time_ms < 1000
        assert result.sla_met is True

    @pytest.mark.asyncio
    async def test_sla_violation_fallback(self, discovery_system, sample_packet_data):
        """Test SLA violation with fallback."""
        request = DiscoveryRequest(
            request_id="test_sla_violation",
            packet_data=sample_packet_data,
            quality_mode=QualityMode.ACCURATE,
        )

        # Use very tight SLA to force violation
        result = await discovery_system.discover_with_sla(request, sla_ms=1)

        assert result is not None
        assert result.fallback_used is True

    @pytest.mark.asyncio
    async def test_quality_modes(self, discovery_system, sample_packet_data):
        """Test different quality modes."""
        modes = [QualityMode.FAST, QualityMode.BALANCED, QualityMode.ACCURATE]
        results = []

        for mode in modes:
            request = DiscoveryRequest(
                request_id=f"test_quality_{mode.value}",
                packet_data=sample_packet_data,
                quality_mode=mode,
            )

            result = await discovery_system.discover_with_sla(request, sla_ms=5000)
            results.append(result)

        # Verify all modes work
        assert all(r is not None for r in results)
        assert all(r.quality_mode in modes for r in results)


class TestExplainableDiscovery:
    """Test explainable AI functionality."""

    @pytest.mark.asyncio
    async def test_explanation_generation(self, discovery_system, sample_packet_data):
        """Test explanation generation."""
        request = DiscoveryRequest(
            request_id="test_explanation",
            packet_data=sample_packet_data,
            require_explanation=True,
        )

        result = await discovery_system.discover_with_explainability(request)

        assert result is not None
        assert result.explanation is not None
        assert len(result.explanation.feature_importances) > 0
        assert len(result.explanation.reasoning) > 0

    @pytest.mark.asyncio
    async def test_feature_importance(self, discovery_system, sample_packet_data):
        """Test feature importance calculation."""
        request = DiscoveryRequest(
            request_id="test_features",
            packet_data=sample_packet_data,
            require_explanation=True,
        )

        result = await discovery_system.discover_with_explainability(request)

        assert result.explanation is not None
        importances = result.explanation.feature_importances

        # Verify importances are sorted
        scores = [f.importance_score for f in importances]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_decision_paths(self, discovery_system, sample_packet_data):
        """Test decision path extraction."""
        request = DiscoveryRequest(
            request_id="test_paths",
            packet_data=sample_packet_data,
            require_explanation=True,
        )

        result = await discovery_system.discover_with_explainability(request)

        assert result.explanation is not None
        paths = result.explanation.decision_paths

        # Verify paths are sequential
        if paths:
            steps = [p.step for p in paths]
            assert steps == list(range(len(steps)))


class TestModelVersioning:
    """Test model versioning and deployment."""

    def test_version_registration(self, config):
        """Test model version registration."""
        manager = ModelVersionManager(config)

        version = manager.register_version(
            version_id="v1.0.0",
            model_path="models/test_v1.pt",
            deployment_type=DeploymentType.PRIMARY,
        )

        assert version.version_id == "v1.0.0"
        assert version.deployment_type == DeploymentType.PRIMARY
        assert version.is_active is True

    def test_version_activation(self, config):
        """Test version activation."""
        manager = ModelVersionManager(config)

        manager.register_version("v1.0.0", "models/v1.pt")
        manager.register_version("v2.0.0", "models/v2.pt")

        manager.activate_version("v1.0.0")
        assert manager.active_version == "v1.0.0"

        manager.activate_version("v2.0.0")
        assert manager.active_version == "v2.0.0"
        assert manager.versions["v1.0.0"].is_active is False

    def test_rollback(self, config):
        """Test version rollback."""
        manager = ModelVersionManager(config)

        manager.register_version("v1.0.0", "models/v1.pt")
        manager.activate_version("v1.0.0")

        manager.register_version("v2.0.0", "models/v2.pt")
        manager.activate_version("v2.0.0")

        # Rollback to v1.0.0
        success = manager.rollback_to_version("v1.0.0")

        assert success is True
        assert manager.active_version == "v1.0.0"

    def test_canary_deployment(self, config):
        """Test canary deployment."""
        manager = ModelVersionManager(config)

        manager.register_version("v1.0.0", "models/v1.pt")
        manager.activate_version("v1.0.0")

        manager.register_version("v2.0.0", "models/v2.pt")
        success = manager.create_canary_deployment("v2.0.0", traffic_percentage=10.0)

        assert success is True
        assert "v2.0.0" in manager.canary_deployments
        assert manager.versions["v2.0.0"].traffic_percentage == 10.0

    def test_ab_testing(self, config):
        """Test A/B testing setup."""
        manager = ModelVersionManager(config)

        manager.register_version("v1.0.0", "models/v1.pt")
        manager.register_version("v2.0.0", "models/v2.pt")

        ab_test = manager.create_ab_test(
            test_id="test_v1_vs_v2",
            version_a="v1.0.0",
            version_b="v2.0.0",
            traffic_split=0.5,
        )

        assert ab_test.test_id == "test_v1_vs_v2"
        assert ab_test.traffic_split == 0.5
        assert ab_test.is_active is True

    @pytest.mark.asyncio
    async def test_version_selection(self, discovery_system, sample_packet_data):
        """Test version selection for requests."""
        # Register multiple versions
        discovery_system.version_manager.register_version("v1.0.0", "models/v1.pt")
        discovery_system.version_manager.register_version("v2.0.0", "models/v2.pt")
        discovery_system.version_manager.activate_version("v1.0.0")

        # Test explicit version request
        request = DiscoveryRequest(
            request_id="test_version_select",
            packet_data=sample_packet_data,
            model_version="v2.0.0",
        )

        selected = discovery_system.version_manager.select_version_for_request(request)
        assert selected == "v2.0.0"


class TestConcurrency:
    """Test concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, discovery_system, sample_packet_data):
        """Test handling multiple concurrent requests."""
        num_requests = 50

        async def make_request(i):
            request = DiscoveryRequest(
                request_id=f"concurrent_{i}", packet_data=sample_packet_data
            )
            return await discovery_system.discover_with_sla(request, sla_ms=5000)

        # Execute requests concurrently
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all succeeded
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == num_requests

    @pytest.mark.asyncio
    async def test_load_handling(self, discovery_system, sample_packet_data):
        """Test system under load."""
        num_batches = 10
        batch_size = 20

        for batch in range(num_batches):
            tasks = []
            for i in range(batch_size):
                request = DiscoveryRequest(
                    request_id=f"load_test_{batch}_{i}", packet_data=sample_packet_data
                )
                tasks.append(discovery_system.discover_with_sla(request, sla_ms=2000))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Most requests should succeed
            successful = [r for r in results if not isinstance(r, Exception)]
            assert len(successful) >= batch_size * 0.9  # 90% success rate


class TestCaching:
    """Test result caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, discovery_system, sample_packet_data):
        """Test cache hit scenario."""
        request1 = DiscoveryRequest(
            request_id="cache_test_1", packet_data=sample_packet_data
        )

        # First request - cache miss
        result1 = await discovery_system.discover_with_sla(request1, sla_ms=5000)

        # Check if result is cached
        cached = discovery_system._get_cached_result(sample_packet_data)
        assert cached is not None

    @pytest.mark.asyncio
    async def test_cache_expiration(self, discovery_system, sample_packet_data):
        """Test cache expiration."""
        # Set short TTL for testing
        original_ttl = discovery_system.cache_ttl
        discovery_system.cache_ttl = 1  # 1 second

        request = DiscoveryRequest(
            request_id="cache_expire_test", packet_data=sample_packet_data
        )

        # Make request
        await discovery_system.discover_with_sla(request, sla_ms=5000)

        # Wait for cache to expire
        await asyncio.sleep(2)

        # Check cache is expired
        cached = discovery_system._get_cached_result(sample_packet_data)
        assert cached is None

        # Restore original TTL
        discovery_system.cache_ttl = original_ttl


class TestMetrics:
    """Test metrics collection."""

    @pytest.mark.asyncio
    async def test_metrics_recording(self, discovery_system, sample_packet_data):
        """Test metrics are recorded."""
        initial_count = discovery_system.request_count

        request = DiscoveryRequest(
            request_id="metrics_test", packet_data=sample_packet_data
        )

        await discovery_system.discover_with_sla(request, sla_ms=5000)

        # Verify metrics updated
        stats = discovery_system.get_statistics()
        assert stats["total_requests"] >= initial_count

    @pytest.mark.asyncio
    async def test_version_metrics(self, discovery_system, sample_packet_data):
        """Test version-specific metrics."""
        version_id = "v1.0.0"

        request = DiscoveryRequest(
            request_id="version_metrics_test",
            packet_data=sample_packet_data,
            model_version=version_id,
        )

        await discovery_system.discover_with_sla(request, sla_ms=5000)

        # Check version metrics
        perf = discovery_system.version_manager.get_version_performance(version_id)
        assert len(perf) > 0


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_invalid_packet_data(self, discovery_system):
        """Test handling of invalid packet data."""
        request = DiscoveryRequest(
            request_id="invalid_data_test", packet_data=b""  # Empty data
        )

        # Should handle gracefully
        result = await discovery_system.discover_with_sla(request, sla_ms=5000)
        assert result is not None

    @pytest.mark.asyncio
    async def test_timeout_handling(self, discovery_system, sample_packet_data):
        """Test timeout handling."""
        request = DiscoveryRequest(
            request_id="timeout_test",
            packet_data=sample_packet_data,
            quality_mode=QualityMode.ACCURATE,
        )

        # Very tight timeout
        result = await discovery_system.discover_with_sla(request, sla_ms=1)

        # Should use fallback
        assert result.fallback_used is True


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, discovery_system, sample_packet_data):
        """Test complete discovery workflow."""
        # Create request with all features
        request = DiscoveryRequest(
            request_id="e2e_test",
            packet_data=sample_packet_data,
            quality_mode=QualityMode.BALANCED,
            require_explanation=True,
            sla_ms=500,
        )

        # Execute discovery
        result = await discovery_system.discover_with_explainability(request)

        # Verify complete result
        assert result is not None
        assert result.request_id == "e2e_test"
        assert result.protocol_type is not None
        assert result.confidence >= 0.0
        assert result.explanation is not None
        assert result.processing_time_ms > 0
        assert result.model_version is not None

    @pytest.mark.asyncio
    async def test_production_scenario(self, discovery_system):
        """Test realistic production scenario."""
        # Simulate production traffic
        protocols = ["HTTP", "DNS", "TLS", "SSH"]
        num_requests = 100

        results = []

        for i in range(num_requests):
            # Generate varied packet data
            packet_data = bytes([i % 256 for _ in range(64)])

            request = DiscoveryRequest(
                request_id=f"prod_test_{i}",
                packet_data=packet_data,
                quality_mode=QualityMode.BALANCED,
            )

            result = await discovery_system.discover_with_sla(request, sla_ms=200)
            results.append(result)

        # Analyze results
        successful = [r for r in results if r.sla_met]
        sla_compliance = len(successful) / len(results)

        # Should have high SLA compliance
        assert sla_compliance >= 0.95  # 95% compliance

        # Check statistics
        stats = discovery_system.get_statistics()
        assert stats["total_requests"] >= num_requests
        assert stats["sla_compliance_rate"] >= 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
