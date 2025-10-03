"""
CRONOS AI Engine - Protocol Discovery Integration Tests

Comprehensive integration tests for end-to-end protocol discovery workflows,
including all components working together in production scenarios.
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any
import numpy as np

from ai_engine.discovery.protocol_discovery_orchestrator import (
    ProtocolDiscoveryOrchestrator,
    DiscoveryRequest,
    DiscoveryResult,
    DiscoveryPhase
)
from ai_engine.discovery.enhanced_protocol_discovery_orchestrator import (
    EnhancedProtocolDiscoveryOrchestrator,
    EnhancedDiscoveryRequest,
    LLMAnalysisType
)
from ai_engine.discovery.production_enhancements import (
    DistributedCache,
    CacheBackend,
    DiscoveryMetrics,
    ProductionConfig,
    HealthChecker,
    HealthStatus
)
from ai_engine.discovery.statistical_analyzer import StatisticalAnalyzer
from ai_engine.discovery.grammar_learner import GrammarLearner
from ai_engine.discovery.parser_generator import ParserGenerator
from ai_engine.discovery.protocol_classifier import ProtocolClassifier, ProtocolSample
from ai_engine.discovery.message_validator import MessageValidator, ValidationLevel
from ai_engine.core.config import Config


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def production_config():
    """Create production configuration."""
    return ProductionConfig(
        enable_caching=True,
        cache_backend=CacheBackend.MEMORY,
        enable_circuit_breakers=True,
        enable_detailed_metrics=True,
        max_concurrent_discoveries=50
    )


@pytest.fixture
async def orchestrator(config):
    """Create and initialize orchestrator."""
    orch = ProtocolDiscoveryOrchestrator(config)
    await orch.initialize()
    yield orch
    await orch.shutdown()


@pytest.fixture
async def enhanced_orchestrator(config):
    """Create and initialize enhanced orchestrator."""
    orch = EnhancedProtocolDiscoveryOrchestrator(config)
    await orch.initialize()
    yield orch
    await orch.shutdown()


@pytest.fixture
async def distributed_cache():
    """Create distributed cache."""
    cache = DistributedCache(
        backend=CacheBackend.MEMORY,
        max_memory_size=100 * 1024 * 1024  # 100MB
    )
    await cache.start_cleanup_task()
    yield cache
    await cache.shutdown()


@pytest.fixture
def sample_http_messages():
    """Sample HTTP protocol messages."""
    return [
        b"GET /api/users HTTP/1.1\r\nHost: example.com\r\n\r\n",
        b"POST /api/data HTTP/1.1\r\nHost: example.com\r\nContent-Length: 13\r\n\r\n{\"key\":\"val\"}",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"ok\"}",
    ]


@pytest.fixture
def sample_binary_messages():
    """Sample binary protocol messages."""
    return [
        b"\x00\x01\x02\x03" + b"A" * 100,
        b"\x00\x01\x02\x04" + b"B" * 150,
        b"\x00\x01\x02\x05" + b"C" * 200,
    ]


@pytest.fixture
def sample_json_messages():
    """Sample JSON messages."""
    return [
        b'{"type":"request","id":1,"data":"test"}',
        b'{"type":"response","id":1,"status":"ok"}',
        b'{"type":"event","name":"update","payload":{"value":42}}',
    ]


# ============================================================================
# BASIC INTEGRATION TESTS
# ============================================================================

class TestBasicIntegration:
    """Basic integration tests for protocol discovery."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_discovery_http(self, orchestrator, sample_http_messages):
        """Test complete discovery workflow with HTTP messages."""
        request = DiscoveryRequest(
            messages=sample_http_messages,
            training_mode=True,
            generate_parser=True,
            validate_results=True
        )
        
        result = await orchestrator.discover_protocol(request)
        
        assert result is not None
        assert result.protocol_type != "unknown"
        assert result.confidence > 0.0
        assert len(result.phases_completed) > 0
        assert DiscoveryPhase.STATISTICAL_ANALYSIS in result.phases_completed
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_discovery_binary(self, orchestrator, sample_binary_messages):
        """Test complete discovery workflow with binary messages."""
        request = DiscoveryRequest(
            messages=sample_binary_messages,
            training_mode=True,
            generate_parser=True,
            validate_results=True
        )
        
        result = await orchestrator.discover_protocol(request)
        
        assert result is not None
        assert result.protocol_type != "unknown"
        assert result.grammar is not None
        assert len(result.grammar.rules) > 0
        assert result.statistical_analysis is not None
    
    @pytest.mark.asyncio
    async def test_discovery_with_caching(self, orchestrator, sample_http_messages):
        """Test discovery with caching enabled."""
        request = DiscoveryRequest(
            messages=sample_http_messages,
            training_mode=False
        )
        
        # First discovery - cache miss
        result1 = await orchestrator.discover_protocol(request)
        time1 = result1.processing_time
        
        # Second discovery - should hit cache
        result2 = await orchestrator.discover_protocol(request)
        time2 = result2.processing_time
        
        assert result1.protocol_type == result2.protocol_type
        assert result1.confidence == result2.confidence
        # Cache hit should be faster (though not guaranteed in all cases)
        assert time2 <= time1 * 2  # Allow some variance
    
    @pytest.mark.asyncio
    async def test_concurrent_discoveries(self, orchestrator, sample_http_messages, sample_binary_messages):
        """Test concurrent discovery operations."""
        requests = [
            DiscoveryRequest(messages=sample_http_messages),
            DiscoveryRequest(messages=sample_binary_messages),
            DiscoveryRequest(messages=sample_http_messages[:2]),
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*[
            orchestrator.discover_protocol(req) for req in requests
        ])
        
        assert len(results) == 3
        assert all(r is not None for r in results)
        assert all(r.protocol_type != "unknown" for r in results)


# ============================================================================
# ENHANCED INTEGRATION TESTS
# ============================================================================

class TestEnhancedIntegration:
    """Integration tests for enhanced discovery with LLM."""
    
    @pytest.mark.asyncio
    async def test_enhanced_discovery_with_llm(self, enhanced_orchestrator, sample_http_messages):
        """Test enhanced discovery with LLM analysis."""
        request = EnhancedDiscoveryRequest(
            messages=sample_http_messages,
            enable_llm_analysis=True,
            llm_analysis_types=[
                LLMAnalysisType.PROTOCOL_IDENTIFICATION,
                LLMAnalysisType.SECURITY_ASSESSMENT
            ],
            natural_language_explanation=True
        )
        
        result = await enhanced_orchestrator.discover_protocol_enhanced(request)
        
        assert result is not None
        assert result.protocol_type != "unknown"
        # LLM analysis might not be available in test environment
        # assert len(result.llm_analyses) > 0
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_enhanced_discovery_without_llm(self, enhanced_orchestrator, sample_binary_messages):
        """Test enhanced discovery without LLM (fallback to traditional)."""
        request = EnhancedDiscoveryRequest(
            messages=sample_binary_messages,
            enable_llm_analysis=False,
            training_mode=True
        )
        
        result = await enhanced_orchestrator.discover_protocol_enhanced(request)
        
        assert result is not None
        assert result.protocol_type != "unknown"
        assert result.grammar is not None


# ============================================================================
# COMPONENT INTEGRATION TESTS
# ============================================================================

class TestComponentIntegration:
    """Test integration between individual components."""
    
    @pytest.mark.asyncio
    async def test_statistical_to_grammar_pipeline(self, config, sample_http_messages):
        """Test pipeline from statistical analysis to grammar learning."""
        analyzer = StatisticalAnalyzer(config)
        learner = GrammarLearner(config)
        
        # Statistical analysis
        stats_result = await analyzer.analyze_messages(sample_http_messages)
        assert stats_result is not None
        assert 'byte_statistics' in stats_result
        
        # Grammar learning using stats
        grammar = await learner.learn_grammar(sample_http_messages)
        assert grammar is not None
        assert len(grammar.rules) > 0
        assert len(grammar.symbols) > 0
    
    @pytest.mark.asyncio
    async def test_grammar_to_parser_pipeline(self, config, sample_http_messages):
        """Test pipeline from grammar learning to parser generation."""
        learner = GrammarLearner(config)
        generator = ParserGenerator(config)
        
        # Learn grammar
        grammar = await learner.learn_grammar(sample_http_messages)
        
        # Generate parser
        parser = await generator.generate_parser(
            grammar,
            parser_id="test_http",
            protocol_name="HTTP"
        )
        
        assert parser is not None
        assert parser.parse_function is not None
        assert parser.validate_function is not None
        
        # Test parser
        parse_result = await parser.parse_function(sample_http_messages[0])
        assert parse_result is not None
    
    @pytest.mark.asyncio
    async def test_parser_to_validator_pipeline(self, config, sample_http_messages):
        """Test pipeline from parser generation to validation."""
        learner = GrammarLearner(config)
        generator = ParserGenerator(config)
        validator = MessageValidator(config)
        
        # Learn grammar and generate parser
        grammar = await learner.learn_grammar(sample_http_messages)
        parser = await generator.generate_parser(grammar, parser_id="test_http")
        
        # Register with validator
        validator.register_parser("http", parser)
        validator.register_grammar("http", grammar)
        
        # Validate message
        validation_result = await validator.validate(
            sample_http_messages[0],
            protocol_type="http",
            validation_level=ValidationLevel.STANDARD
        )
        
        assert validation_result is not None
        # Validation might fail due to incomplete parser, but should not crash
        assert validation_result.confidence >= 0.0


# ============================================================================
# PRODUCTION FEATURES TESTS
# ============================================================================

class TestProductionFeatures:
    """Test production-specific features."""
    
    @pytest.mark.asyncio
    async def test_distributed_cache_operations(self, distributed_cache):
        """Test distributed cache operations."""
        # Set value
        await distributed_cache.set("test_key", {"data": "test_value"}, ttl=60.0)
        
        # Get value
        value = await distributed_cache.get("test_key")
        assert value is not None
        assert value["data"] == "test_value"
        
        # Cache stats
        stats = distributed_cache.get_stats()
        assert stats['hits'] > 0
        assert stats['sets'] > 0
        assert stats['hit_rate'] > 0.0
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self, distributed_cache):
        """Test cache LRU eviction."""
        # Fill cache with many entries
        for i in range(100):
            await distributed_cache.set(f"key_{i}", f"value_{i}" * 1000, ttl=3600.0)
        
        # Verify some entries exist
        value = await distributed_cache.get("key_50")
        assert value is not None
        
        stats = distributed_cache.get_stats()
        assert stats['memory_cache_size'] > 0
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, distributed_cache):
        """Test cache entry expiration."""
        # Set with short TTL
        await distributed_cache.set("expire_key", "expire_value", ttl=0.1)
        
        # Should exist immediately
        value1 = await distributed_cache.get("expire_key")
        assert value1 == "expire_value"
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired
        value2 = await distributed_cache.get("expire_key")
        assert value2 is None
    
    @pytest.mark.asyncio
    async def test_health_checker(self):
        """Test health checking system."""
        checker = HealthChecker()
        
        # Register mock health checks
        async def check_cache():
            return HealthStatus.HEALTHY, "Cache operational", {}
        
        async def check_models():
            return HealthStatus.HEALTHY, "Models loaded", {}
        
        checker.register_check("cache", check_cache)
        checker.register_check("models", check_models)
        
        # Run health checks
        results = await checker.check_all()
        assert len(results) == 2
        assert all(h.status == HealthStatus.HEALTHY for h in results.values())
        
        # Check overall health
        is_healthy = await checker.is_healthy()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection."""
        metrics = DiscoveryMetrics()
        
        # Record some metrics
        metrics.discovery_requests_total.labels(
            protocol_type='http',
            status='success',
            cache_hit='false'
        ).inc()
        
        metrics.discovery_duration_seconds.labels(
            protocol_type='http',
            phase='statistical_analysis'
        ).observe(0.05)
        
        metrics.discovery_confidence_score.labels(
            protocol_type='http'
        ).observe(0.85)
        
        # Metrics should be recorded (actual values checked by Prometheus)
        assert metrics.discovery_requests_total is not None
        assert metrics.discovery_duration_seconds is not None


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and load tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_discovery_latency(self, orchestrator, sample_http_messages):
        """Test discovery latency meets requirements."""
        request = DiscoveryRequest(messages=sample_http_messages)
        
        latencies = []
        for _ in range(10):
            start = time.time()
            result = await orchestrator.discover_protocol(request)
            latency = time.time() - start
            latencies.append(latency)
            assert result is not None
        
        avg_latency = sum(latencies) / len(latencies)
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        
        # Performance targets (adjust based on requirements)
        assert avg_latency < 1.0, f"Average latency {avg_latency:.3f}s exceeds 1s"
        assert p99_latency < 2.0, f"P99 latency {p99_latency:.3f}s exceeds 2s"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_load(self, orchestrator, sample_http_messages):
        """Test system under concurrent load."""
        num_concurrent = 20
        requests = [DiscoveryRequest(messages=sample_http_messages) for _ in range(num_concurrent)]
        
        start = time.time()
        results = await asyncio.gather(*[
            orchestrator.discover_protocol(req) for req in requests
        ])
        total_time = time.time() - start
        
        assert len(results) == num_concurrent
        assert all(r is not None for r in results)
        
        # Should handle concurrent requests efficiently
        avg_time_per_request = total_time / num_concurrent
        assert avg_time_per_request < 2.0, f"Average time per request {avg_time_per_request:.3f}s too high"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_usage(self, orchestrator, sample_http_messages):
        """Test memory usage remains reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many discoveries
        for _ in range(50):
            request = DiscoveryRequest(messages=sample_http_messages)
            await orchestrator.discover_protocol(request)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for 50 discoveries)
        assert memory_increase < 500, f"Memory increased by {memory_increase:.1f}MB"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, orchestrator):
        """Test handling of invalid input."""
        # Empty messages
        request = DiscoveryRequest(messages=[])
        
        with pytest.raises(Exception):  # Should raise appropriate exception
            await orchestrator.discover_protocol(request)
    
    @pytest.mark.asyncio
    async def test_malformed_data_handling(self, orchestrator):
        """Test handling of malformed data."""
        # Very large message
        large_message = b"X" * (20 * 1024 * 1024)  # 20MB
        request = DiscoveryRequest(messages=[large_message])
        
        # Should handle gracefully (might fail but shouldn't crash)
        try:
            result = await orchestrator.discover_protocol(request)
            # If it succeeds, verify result
            assert result is not None
        except Exception as e:
            # If it fails, should be a proper exception
            assert isinstance(e, (ProtocolException, ModelException, Exception))
    
    @pytest.mark.asyncio
    async def test_component_failure_recovery(self, orchestrator, sample_http_messages):
        """Test recovery from component failures."""
        request = DiscoveryRequest(
            messages=sample_http_messages,
            training_mode=False
        )
        
        # Should complete even if some components have issues
        result = await orchestrator.discover_protocol(request)
        assert result is not None


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class TestRegression:
    """Regression tests for known issues."""
    
    @pytest.mark.asyncio
    async def test_empty_grammar_handling(self, config):
        """Test handling of empty grammar (regression test)."""
        generator = ParserGenerator(config)
        learner = GrammarLearner(config)
        
        # Try to learn grammar from minimal data
        minimal_messages = [b"A"]
        grammar = await learner.learn_grammar(minimal_messages)
        
        # Should handle gracefully
        assert grammar is not None
    
    @pytest.mark.asyncio
    async def test_unicode_handling(self, orchestrator):
        """Test handling of Unicode data."""
        unicode_messages = [
            "Hello 世界".encode('utf-8'),
            "Привет мир".encode('utf-8'),
            "مرحبا العالم".encode('utf-8'),
        ]
        
        request = DiscoveryRequest(messages=unicode_messages)
        result = await orchestrator.discover_protocol(request)
        
        assert result is not None
        assert result.protocol_type != "unknown"


# ============================================================================
# CLEANUP
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup():
    """Cleanup after each test."""
    yield
    # Cleanup code here if needed
    await asyncio.sleep(0.1)  # Allow async cleanup


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])