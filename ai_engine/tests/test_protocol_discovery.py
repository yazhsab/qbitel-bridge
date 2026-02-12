"""
Comprehensive test suite for QBITEL Bridge System.
Tests all core components with unit, integration, and performance testing.
"""

import pytest
import pytest_asyncio
import asyncio
import numpy as np
import time
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional

# Import all components to test
from ai_engine.discovery.statistical_analyzer import (
    StatisticalAnalyzer,
    TrafficPattern,
    FieldBoundary,
)
from ai_engine.discovery.grammar_learner import GrammarLearner, PCFGRule, Grammar
from ai_engine.discovery.parser_generator import ParserGenerator, GeneratedParser
from ai_engine.discovery.protocol_classifier import (
    ProtocolClassifier,
    ClassificationResult,
)
from ai_engine.discovery.message_validator import (
    MessageValidator,
    ValidationResult,
    ValidationRule,
)
from ai_engine.discovery.protocol_discovery_orchestrator import (
    ProtocolDiscoveryOrchestrator,
    DiscoveryResult,
)
from ai_engine.core.error_handling import (
    ErrorHandler,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    RetryManager,
)
from ai_engine.core.config import Config
from ai_engine.core.structured_logging import StructuredLogger
from ai_engine.core.performance_optimizer import PerformanceOptimizer, ComputationCache
from ai_engine.monitoring.enterprise_metrics import EnterpriseMetrics, MetricsCollector


class TestStatisticalAnalyzer:
    """Test suite for StatisticalAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return StatisticalAnalyzer()

    @pytest.fixture
    def sample_traffic(self):
        """Generate sample network traffic for testing."""
        return [
            b"GET /api/v1/users HTTP/1.1\r\nHost: example.com\r\n\r\n",
            b"POST /api/v1/data HTTP/1.1\r\nContent-Length: 123\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n",
            b"\x01\x02\x03\x04\x05\x06\x07\x08",  # Binary protocol
            b"CONNECT server.example.com:443 HTTP/1.1\r\n\r\n",
        ]

    @pytest.mark.asyncio
    async def test_analyze_traffic_basic(self, analyzer, sample_traffic):
        """Test basic traffic analysis."""
        result = await analyzer.analyze_traffic(sample_traffic)

        assert isinstance(result, TrafficPattern)
        assert result.total_messages == len(sample_traffic)
        assert result.message_lengths is not None
        assert len(result.message_lengths) == len(sample_traffic)
        assert result.entropy > 0
        assert result.binary_ratio >= 0 and result.binary_ratio <= 1

    @pytest.mark.asyncio
    async def test_calculate_entropy(self, analyzer):
        """Test entropy calculation."""
        # High entropy (random data)
        random_data = os.urandom(1000)
        entropy_high = await analyzer._calculate_entropy(random_data)

        # Low entropy (repeated pattern)
        pattern_data = b"ABCD" * 250
        entropy_low = await analyzer._calculate_entropy(pattern_data)

        assert entropy_high > entropy_low
        assert entropy_high > 6.0  # Should be close to 8 for random data
        assert entropy_low < 3.0  # Should be much lower for patterns

    @pytest.mark.asyncio
    async def test_detect_field_boundaries(self, analyzer):
        """Test field boundary detection."""
        messages = [
            b"USER:alice|PASS:secret123|CMD:login",
            b"USER:bob|PASS:password456|CMD:logout",
            b"USER:charlie|PASS:test789|CMD:status",
        ]

        boundaries = await analyzer.detect_field_boundaries(messages)

        assert len(boundaries) > 0
        # Should detect the pipe separators
        assert any(b.separator == b"|" for b in boundaries)
        # Should detect consistent positions
        assert all(b.confidence > 0.5 for b in boundaries)

    @pytest.mark.asyncio
    async def test_detect_patterns_http(self, analyzer):
        """Test HTTP pattern detection."""
        http_messages = [
            b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n",
            b"POST /api/data HTTP/1.1\r\nContent-Length: 100\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n",
        ]

        patterns = await analyzer.detect_patterns(http_messages)

        assert len(patterns) > 0
        # Should detect HTTP-like patterns
        http_patterns = [p for p in patterns if b"HTTP" in p.pattern]
        assert len(http_patterns) > 0

    @pytest.mark.asyncio
    async def test_performance_large_dataset(self, analyzer):
        """Test performance with large dataset."""
        # Generate large dataset
        large_dataset = [os.urandom(100) for _ in range(1000)]

        start_time = time.time()
        result = await analyzer.analyze_traffic(large_dataset)
        duration = time.time() - start_time

        assert duration < 10.0  # Should complete within 10 seconds
        assert result.total_messages == 1000
        assert result.processing_time < 10.0


class TestGrammarLearner:
    """Test suite for GrammarLearner."""

    @pytest.fixture
    def learner(self, test_config):
        return GrammarLearner(test_config)

    @pytest.fixture
    def sample_patterns(self):
        """Sample patterns for grammar learning."""
        return [
            b"GET /path HTTP/1.1",
            b"POST /api HTTP/1.1",
            b"PUT /data HTTP/1.1",
            b"DELETE /item HTTP/1.1",
        ]

    @pytest.mark.asyncio
    async def test_learn_pcfg_basic(self, learner, sample_patterns):
        """Test basic PCFG learning."""
        grammar = await learner.learn_pcfg(sample_patterns)

        assert isinstance(grammar, Grammar)
        assert len(grammar.rules) > 0
        assert grammar.start_symbol is not None

        # Should have learned HTTP method patterns
        method_rules = [r for r in grammar.rules if "METHOD" in str(r)]
        assert len(method_rules) > 0

    @pytest.mark.asyncio
    async def test_refine_with_em(self, learner, sample_patterns):
        """Test EM algorithm refinement."""
        # Learn initial grammar
        initial_grammar = await learner.learn_pcfg(sample_patterns)

        # Refine with EM
        refined_grammar = await learner.refine_with_em(initial_grammar, sample_patterns, max_iterations=5)

        assert isinstance(refined_grammar, Grammar)
        assert len(refined_grammar.rules) >= len(initial_grammar.rules)

        # Check that probabilities sum to 1 for each non-terminal
        non_terminals = set(r.lhs for r in refined_grammar.rules)
        for nt in non_terminals:
            nt_rules = [r for r in refined_grammar.rules if r.lhs == nt]
            total_prob = sum(r.probability for r in nt_rules)
            assert abs(total_prob - 1.0) < 0.01  # Should sum to 1 (within tolerance)

    @pytest.mark.asyncio
    async def test_identify_semantic_components(self, learner):
        """Test semantic component identification."""
        messages = [
            b"LOGIN alice password123",
            b"LOGIN bob secretkey456",
            b"LOGOUT alice",
            b"STATUS online",
        ]

        components = await learner.identify_semantic_components(messages)

        assert len(components) > 0
        # Should identify command component
        command_components = [c for c in components if c.component_type == "COMMAND"]
        assert len(command_components) > 0

        # Should identify user component
        user_components = [c for c in components if c.component_type == "USER"]
        assert len(user_components) > 0

    @pytest.mark.asyncio
    async def test_optimize_grammar(self, learner, sample_patterns):
        """Test grammar optimization."""
        grammar = await learner.learn_pcfg(sample_patterns)
        optimized = await learner.optimize_grammar(grammar, sample_patterns)

        assert isinstance(optimized, Grammar)
        # Optimized grammar should have fewer or equal rules (merged redundant ones)
        assert len(optimized.rules) <= len(grammar.rules)


class TestParserGenerator:
    """Test suite for ParserGenerator."""

    @pytest.fixture
    def generator(self, test_config):
        return ParserGenerator(test_config)

    @pytest.fixture
    def sample_grammar(self):
        """Sample grammar for parser generation."""
        rules = [
            PCFGRule("S", ["METHOD", "PATH", "VERSION"], 1.0),
            PCFGRule("METHOD", [b"GET"], 0.4),
            PCFGRule("METHOD", [b"POST"], 0.3),
            PCFGRule("METHOD", [b"PUT"], 0.3),
            PCFGRule("PATH", [b"/api"], 0.6),
            PCFGRule("PATH", [b"/data"], 0.4),
            PCFGRule("VERSION", [b"HTTP/1.1"], 1.0),
        ]
        return Grammar("S", rules)

    @pytest.mark.asyncio
    async def test_generate_parser_basic(self, generator, sample_grammar):
        """Test basic parser generation."""
        parser = await generator.generate_parser(sample_grammar, "test_protocol")

        assert isinstance(parser, GeneratedParser)
        assert parser.protocol_name == "test_protocol"
        assert parser.grammar == sample_grammar
        assert parser.parse_function is not None

    @pytest.mark.asyncio
    async def test_parse_with_generated_parser(self, generator, sample_grammar):
        """Test parsing with generated parser."""
        parser = await generator.generate_parser(sample_grammar, "http_like")

        # Test valid message
        valid_message = b"GET /api HTTP/1.1"
        result = await parser.parse(valid_message)

        assert result.success
        assert result.parsed_fields is not None
        assert "METHOD" in result.parsed_fields
        assert result.parsed_fields["METHOD"] == b"GET"

    @pytest.mark.asyncio
    async def test_parser_performance(self, generator, sample_grammar):
        """Test parser performance."""
        parser = await generator.generate_parser(sample_grammar, "perf_test")

        # Test with many messages
        messages = [b"GET /api HTTP/1.1"] * 1000

        start_time = time.time()
        results = []
        for msg in messages:
            result = await parser.parse(msg)
            results.append(result)
        duration = time.time() - start_time

        assert duration < 5.0  # Should parse 1000 messages in < 5 seconds
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_compile_parser_code(self, generator, sample_grammar):
        """Test parser code compilation."""
        code = await generator._compile_parser_code(sample_grammar, "compiled_test")

        assert isinstance(code, str)
        assert "def parse" in code
        assert "class CompiledTestParser" in code

        # Should be valid Python code
        compile(code, "<test>", "exec")  # Should not raise exception


class TestProtocolClassifier:
    """Test suite for ProtocolClassifier."""

    @pytest.fixture
    def classifier(self):
        config = {
            "model_cache_dir": tempfile.mkdtemp(),
            "use_gpu": False,  # Disable GPU for tests
            "ensemble_size": 2,  # Smaller ensemble for faster tests
        }
        return ProtocolClassifier(config)

    @pytest.fixture
    def training_data(self):
        """Sample training data."""
        return {
            "http": [
                b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n",
                b"POST /api/data HTTP/1.1\r\nContent-Length: 100\r\n\r\n",
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n",
            ],
            "smtp": [
                b"HELO example.com\r\n",
                b"MAIL FROM:<user@example.com>\r\n",
                b"RCPT TO:<dest@example.com>\r\n",
            ],
            "custom": [
                b"CMD:LOGIN|USER:alice|PASS:secret",
                b"CMD:DATA|SIZE:1024|CHECKSUM:abc123",
                b"CMD:QUIT|STATUS:OK",
            ],
        }

    @pytest.mark.asyncio
    async def test_train_models(self, classifier, training_data):
        """Test model training."""
        await classifier.train_models(training_data)

        # Should have trained models
        assert hasattr(classifier, "_cnn_model")
        assert hasattr(classifier, "_lstm_model")
        assert hasattr(classifier, "_rf_model")

        # Should have protocol mappings
        assert len(classifier.protocol_to_id) > 0
        assert len(classifier.id_to_protocol) > 0

    @pytest.mark.asyncio
    async def test_classify_message(self, classifier, training_data):
        """Test message classification."""
        # Train first
        await classifier.train_models(training_data)

        # Test HTTP message
        http_msg = b"GET /test HTTP/1.1\r\nHost: test.com\r\n\r\n"
        prediction = await classifier.classify_message(http_msg)

        assert isinstance(prediction, ClassificationResult)
        assert prediction.protocol in training_data.keys()
        assert prediction.confidence > 0.0 and prediction.confidence <= 1.0
        assert len(prediction.class_probabilities) == len(training_data)

    @pytest.mark.asyncio
    async def test_ensemble_prediction(self, classifier, training_data):
        """Test ensemble prediction accuracy."""
        await classifier.train_models(training_data)

        # Test multiple messages
        test_messages = [
            (b"GET /api HTTP/1.1\r\n\r\n", "http"),
            (b"HELO server.com\r\n", "smtp"),
            (b"CMD:STATUS|USER:test", "custom"),
        ]

        correct_predictions = 0
        for message, expected_protocol in test_messages:
            prediction = await classifier.classify_message(message)
            if prediction.protocol == expected_protocol:
                correct_predictions += 1

        # Should get at least 60% accuracy on simple test cases
        accuracy = correct_predictions / len(test_messages)
        assert accuracy >= 0.6

    @pytest.mark.asyncio
    async def test_model_persistence(self, classifier, training_data):
        """Test model saving and loading."""
        await classifier.train_models(training_data)

        # Save models
        model_path = os.path.join(classifier.config["model_cache_dir"], "test_models")
        await classifier.save_models(model_path)

        # Create new classifier and load
        new_classifier = ProtocolClassifier(classifier.config)
        await new_classifier.load_models(model_path)

        # Test that loaded classifier works
        test_msg = b"GET /test HTTP/1.1\r\n\r\n"
        prediction = await new_classifier.classify_message(test_msg)
        assert isinstance(prediction, ClassificationResult)


class TestMessageValidator:
    """Test suite for MessageValidator."""

    @pytest.fixture
    def validator(self, test_config):
        return MessageValidator(test_config)

    @pytest.fixture
    def sample_rules(self):
        """Sample validation rules."""
        return [
            ValidationRule(
                name="http_method",
                rule_type="regex",
                pattern=rb"^(GET|POST|PUT|DELETE|HEAD|OPTIONS)",
                required=True,
            ),
            ValidationRule(
                name="http_version",
                rule_type="regex",
                pattern=rb"HTTP/\d\.\d",
                required=True,
            ),
            ValidationRule(
                name="content_length",
                rule_type="custom",
                custom_validator=lambda msg: b"Content-Length:" in msg or len(msg) < 100,
                required=False,
            ),
        ]

    @pytest.mark.asyncio
    async def test_validate_message_basic(self, validator, sample_rules):
        """Test basic message validation."""
        # Add rules
        for rule in sample_rules:
            validator.add_rule(rule)

        # Test valid message
        valid_msg = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
        result = await validator.validate_message(valid_msg)

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.passed_rules) >= 2  # Should pass http_method and http_version
        assert len(result.failed_rules) == 0

    @pytest.mark.asyncio
    async def test_validate_message_invalid(self, validator, sample_rules):
        """Test validation with invalid message."""
        for rule in sample_rules:
            validator.add_rule(rule)

        # Test invalid message (no HTTP method)
        invalid_msg = b"INVALID MESSAGE FORMAT"
        result = await validator.validate_message(invalid_msg)

        assert not result.is_valid
        assert len(result.failed_rules) > 0
        assert "http_method" in result.failed_rules

    @pytest.mark.asyncio
    async def test_semantic_validation(self, validator):
        """Test semantic validation."""
        # Add semantic rule
        semantic_rule = ValidationRule(
            name="login_semantics",
            rule_type="semantic",
            semantic_check=lambda fields: "username" in fields and "password" in fields,
            required=True,
        )
        validator.add_rule(semantic_rule)

        # Mock parsed fields
        with patch.object(validator, "_extract_semantic_fields") as mock_extract:
            mock_extract.return_value = {"username": "alice", "password": "secret"}

            message = b"LOGIN alice secret"
            result = await validator.validate_message(message)

            assert result.is_valid
            assert "login_semantics" in result.passed_rules

    @pytest.mark.asyncio
    async def test_validation_performance(self, validator, sample_rules):
        """Test validation performance."""
        for rule in sample_rules:
            validator.add_rule(rule)

        # Test with many messages
        messages = [b"GET /test HTTP/1.1\r\n\r\n"] * 1000

        start_time = time.time()
        results = []
        for msg in messages:
            result = await validator.validate_message(msg)
            results.append(result)
        duration = time.time() - start_time

        assert duration < 10.0  # Should validate 1000 messages in < 10 seconds
        assert all(r.is_valid for r in results)


class TestProtocolDiscoveryOrchestrator:
    """Test suite for ProtocolDiscoveryOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        config = {
            "cache_ttl": 300,
            "max_cache_size": 1000,
            "enable_adaptive_learning": True,
        }
        return ProtocolDiscoveryOrchestrator(config)

    @pytest.fixture
    def sample_traffic_data(self):
        """Sample traffic for full discovery testing."""
        return [
            b"GET /api/users HTTP/1.1\r\nHost: api.example.com\r\n\r\n",
            b"POST /api/login HTTP/1.1\r\nContent-Type: application/json\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n",
            b"CMD:LOGIN|USER:alice|PASS:secret123",
            b"CMD:DATA|SIZE:1024|PAYLOAD:abcdef",
            b"STATUS:OK|SESSION:abc123",
        ]

    @pytest.mark.asyncio
    async def test_discover_protocol_full_pipeline(self, orchestrator, sample_traffic_data):
        """Test full protocol discovery pipeline."""
        result = await orchestrator.discover_protocol(sample_traffic_data)

        assert isinstance(result, DiscoveryResult)
        assert result.success
        assert result.discovered_protocols is not None
        assert len(result.discovered_protocols) > 0

        # Should have discovered at least one protocol
        protocol = result.discovered_protocols[0]
        assert hasattr(protocol, "name")
        assert hasattr(protocol, "confidence")
        assert hasattr(protocol, "parser")

    @pytest.mark.asyncio
    async def test_caching_behavior(self, orchestrator, sample_traffic_data):
        """Test that caching improves performance."""
        # First discovery (no cache)
        start_time = time.time()
        result1 = await orchestrator.discover_protocol(sample_traffic_data)
        duration1 = time.time() - start_time

        # Second discovery (should use cache)
        start_time = time.time()
        result2 = await orchestrator.discover_protocol(sample_traffic_data)
        duration2 = time.time() - start_time

        # Second call should be faster due to caching
        assert duration2 < duration1
        assert result1.discovered_protocols == result2.discovered_protocols

    @pytest.mark.asyncio
    async def test_adaptive_learning(self, orchestrator, sample_traffic_data):
        """Test adaptive learning functionality."""
        # Enable adaptive learning
        orchestrator.config["enable_adaptive_learning"] = True

        # First discovery
        result1 = await orchestrator.discover_protocol(sample_traffic_data)

        # Provide feedback (simulate correct classification)
        for protocol in result1.discovered_protocols:
            await orchestrator.provide_feedback(protocol.name, True, sample_traffic_data[:3])

        # Second discovery should have improved confidence
        result2 = await orchestrator.discover_protocol(sample_traffic_data)

        # Compare confidence scores (should be equal or improved)
        for p1, p2 in zip(result1.discovered_protocols, result2.discovered_protocols):
            if p1.name == p2.name:
                assert p2.confidence >= p1.confidence


class TestErrorHandling:
    """Test suite for error handling components."""

    @pytest.fixture
    def error_handler(self):
        return ErrorHandler()

    @pytest.fixture
    def circuit_breaker(self):
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0)
        return CircuitBreaker(config)

    @pytest.mark.asyncio
    async def test_circuit_breaker_states(self, circuit_breaker):
        """Test circuit breaker state transitions."""
        # Initially closed
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

        # Simulate failures
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(lambda: 1 / 0)  # Will raise ZeroDivisionError

        # Should be open now
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Should be half-open
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Successful call should close it
        result = await circuit_breaker.call(lambda: "success")
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_retry_manager(self):
        """Test retry manager functionality."""
        retry_manager = RetryManager(max_retries=3, base_delay=0.1)

        # Test successful retry
        call_count = 0

        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"

        result = await retry_manager.retry(flaky_function)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_error_classification(self, error_handler):
        """Test error classification."""
        # Network error
        network_error = ConnectionError("Connection refused")
        classification = error_handler.classify_error(network_error)
        assert classification.category == "network"
        assert classification.severity in ["medium", "high"]

        # Validation error
        validation_error = ValueError("Invalid input")
        classification = error_handler.classify_error(validation_error)
        assert classification.category == "validation"
        assert classification.severity in ["low", "medium"]


class TestPerformanceOptimizer:
    """Test suite for performance optimization."""

    @pytest.fixture
    def optimizer(self):
        config = {"cache_size": 100, "use_redis": False, "max_threads": 4}
        return PerformanceOptimizer(config)

    @pytest.mark.asyncio
    async def test_computation_cache(self, optimizer):
        """Test computation caching."""
        call_count = 0

        @optimizer.cached_computation()
        async def expensive_computation(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate expensive operation
            return x * 2

        # First call
        result1 = await expensive_computation(5)
        assert result1 == 10
        assert call_count == 1

        # Second call (should use cache)
        result2 = await expensive_computation(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment

    @pytest.mark.asyncio
    async def test_performance_tracking(self, optimizer):
        """Test performance metrics tracking."""
        async with optimizer.performance_context():
            await asyncio.sleep(0.1)  # Simulate work

        metrics = optimizer.get_system_metrics()
        assert metrics.total_requests >= 1
        assert metrics.avg_response_time > 0

    def test_memory_pool(self, optimizer):
        """Test memory pool functionality."""
        # Get buffer from pool
        buffer = optimizer.memory_pool.get_buffer()
        assert isinstance(buffer, bytearray)
        assert len(buffer) == optimizer.memory_pool.buffer_size

        # Return buffer
        optimizer.memory_pool.return_buffer(buffer)

        # Get stats
        stats = optimizer.memory_pool.get_stats()
        assert "pool_size" in stats
        assert "in_use" in stats


class TestIntegration:
    """Integration tests for the complete system."""

    @pytest_asyncio.fixture
    async def full_system(self, test_config):
        """Setup complete protocol discovery system."""
        # Initialize all components
        orchestrator = ProtocolDiscoveryOrchestrator(test_config)
        optimizer = PerformanceOptimizer()
        metrics = EnterpriseMetrics()

        # Start systems
        await metrics.start()

        yield {"orchestrator": orchestrator, "optimizer": optimizer, "metrics": metrics}

        # Cleanup
        await metrics.stop()

    @pytest.mark.asyncio
    async def test_end_to_end_discovery(self, full_system):
        """Test end-to-end protocol discovery."""
        orchestrator = full_system["orchestrator"]

        # Sample mixed protocol traffic
        mixed_traffic = [
            # HTTP traffic
            b"GET /api/v1/status HTTP/1.1\r\nHost: api.example.com\r\n\r\n",
            b"POST /api/v1/login HTTP/1.1\r\nContent-Type: application/json\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n",
            # Custom protocol traffic
            b"CMD:CONNECT|HOST:server.example.com|PORT:443",
            b"CMD:AUTH|USER:alice|TOKEN:abc123def456",
            b"RESPONSE:OK|SESSION:session789",
            # Binary protocol traffic
            b"\x01\x02\x03\x04BINARY_HEADER\x05\x06\x07\x08",
            b"\x09\x0a\x0bDATA_PAYLOAD\x0c\x0d\x0e\x0f",
        ]

        # Perform discovery
        result = await orchestrator.discover_protocol(mixed_traffic)

        # Verify results
        assert result.success
        assert len(result.discovered_protocols) > 0

        # Should identify at least HTTP and custom protocols
        protocol_names = [p.name for p in result.discovered_protocols]
        assert any("http" in name.lower() or "web" in name.lower() for name in protocol_names)

        # Test each discovered protocol's parser
        for protocol in result.discovered_protocols:
            if protocol.parser:
                # Test parsing with relevant messages
                relevant_messages = [msg for msg in mixed_traffic if protocol.confidence > 0.5]
                if relevant_messages:
                    parse_result = await protocol.parser.parse(relevant_messages[0])
                    # Parser should at least attempt to parse
                    assert hasattr(parse_result, "success")

    @pytest.mark.asyncio
    async def test_system_resilience(self, full_system):
        """Test system behavior under stress and errors."""
        orchestrator = full_system["orchestrator"]
        metrics = full_system["metrics"]

        # Generate large amount of varied traffic
        stress_traffic = []
        for i in range(100):
            if i % 3 == 0:
                stress_traffic.append(f"GET /page{i} HTTP/1.1\r\nHost: test{i}.com\r\n\r\n".encode())
            elif i % 3 == 1:
                stress_traffic.append(f"CMD:DATA{i}|SIZE:{len(str(i))}|PAYLOAD:test{i}".encode())
            else:
                stress_traffic.append(os.urandom(50))  # Random binary data

        # Test discovery under stress
        start_time = time.time()
        result = await orchestrator.discover_protocol(stress_traffic)
        duration = time.time() - start_time

        # Should complete within reasonable time
        assert duration < 30.0  # 30 seconds max for 100 messages
        assert result.success

        # System should remain healthy
        dashboard_metrics = metrics.get_dashboard_metrics()
        assert dashboard_metrics["health"]["status"] in ["healthy", "degraded"]

    @pytest.mark.asyncio
    async def test_concurrent_discoveries(self, full_system):
        """Test concurrent protocol discoveries."""
        orchestrator = full_system["orchestrator"]

        # Different traffic patterns
        traffic_patterns = [
            [b"GET /api HTTP/1.1\r\n\r\n"] * 10,
            [b"CMD:LOGIN|USER:test"] * 10,
            [os.urandom(20) for _ in range(10)],
        ]

        # Run concurrent discoveries
        tasks = [orchestrator.discover_protocol(traffic) for traffic in traffic_patterns]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert not isinstance(result, Exception)
            assert result.success


# Performance and load testing
class TestPerformance:
    """Performance and load testing."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_throughput_benchmark(self):
        """Benchmark overall system throughput."""
        orchestrator = ProtocolDiscoveryOrchestrator(Config())

        # Generate test data
        test_messages = []
        for i in range(1000):
            if i % 2 == 0:
                test_messages.append(f"GET /test{i} HTTP/1.1\r\nHost: test.com\r\n\r\n".encode())
            else:
                test_messages.append(f"CMD:TEST{i}|DATA:payload{i}".encode())

        # Benchmark discovery
        start_time = time.time()
        result = await orchestrator.discover_protocol(test_messages)
        duration = time.time() - start_time

        throughput = len(test_messages) / duration

        # Should process at least 100 messages per second
        assert throughput >= 100
        assert result.success

        print(f"Throughput: {throughput:.2f} messages/second")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage(self):
        """Test memory usage under load."""
        import tracemalloc

        tracemalloc.start()

        orchestrator = ProtocolDiscoveryOrchestrator(Config())

        # Generate large dataset
        large_dataset = [os.urandom(1000) for _ in range(1000)]

        # Process dataset
        await orchestrator.discover_protocol(large_dataset)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (< 500MB peak)
        assert peak < 500 * 1024 * 1024

        print(f"Memory usage - Current: {current / 1024 / 1024:.2f} MB, Peak: {peak / 1024 / 1024:.2f} MB")


# Test configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "integration: mark test as integration test")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
