"""
CRONOS AI - Legacy System Whisperer Tests
Comprehensive test suite for legacy protocol analysis and modernization.
"""

import pytest
import asyncio
from typing import List
from datetime import datetime

from ai_engine.llm.legacy_whisperer import (
    LegacySystemWhisperer,
    ProtocolSpecification,
    AdapterCode,
    Explanation,
    AdapterLanguage,
    ProtocolComplexity,
    ModernizationRisk,
    LegacyWhispererException,
)


@pytest.fixture
async def whisperer():
    """Create Legacy System Whisperer instance."""
    whisperer = LegacySystemWhisperer()
    await whisperer.initialize()
    yield whisperer
    await whisperer.shutdown()


@pytest.fixture
def sample_traffic():
    """Generate sample protocol traffic."""
    # Simple binary protocol with header + payload
    samples = []
    for i in range(20):
        # Magic number (4 bytes) + length (2 bytes) + payload
        magic = b"\x42\x43\x44\x45"
        length = (10 + i).to_bytes(2, "big")
        payload = bytes([i] * (10 + i))
        samples.append(magic + length + payload)
    return samples


@pytest.fixture
def sample_text_traffic():
    """Generate sample text protocol traffic."""
    samples = []
    for i in range(15):
        message = f"CMD:{i:03d}|DATA:test_data_{i}|END\n".encode("ascii")
        samples.append(message)
    return samples


class TestProtocolReverseEngineering:
    """Test protocol reverse engineering functionality."""

    @pytest.mark.asyncio
    async def test_reverse_engineer_binary_protocol(self, whisperer, sample_traffic):
        """Test reverse engineering of binary protocol."""
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context="Test binary protocol"
        )

        assert isinstance(spec, ProtocolSpecification)
        assert spec.protocol_name
        assert spec.confidence_score > 0.0
        assert spec.samples_analyzed == len(sample_traffic)
        assert len(spec.fields) > 0
        assert len(spec.patterns) > 0
        assert spec.is_binary is True

    @pytest.mark.asyncio
    async def test_reverse_engineer_text_protocol(self, whisperer, sample_text_traffic):
        """Test reverse engineering of text protocol."""
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_text_traffic, system_context="Test text protocol"
        )

        assert isinstance(spec, ProtocolSpecification)
        assert spec.samples_analyzed == len(sample_text_traffic)
        assert spec.is_binary is False

    @pytest.mark.asyncio
    async def test_insufficient_samples_error(self, whisperer):
        """Test error handling for insufficient samples."""
        with pytest.raises(LegacyWhispererException) as exc_info:
            await whisperer.reverse_engineer_protocol(
                traffic_samples=[b"test"] * 5, system_context=""  # Less than minimum
            )
        assert "Insufficient samples" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_protocol_caching(self, whisperer, sample_traffic):
        """Test that protocol specifications are cached."""
        # First call
        spec1 = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context="Test"
        )

        # Second call with same data should use cache
        spec2 = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context="Test"
        )

        assert spec1.spec_id == spec2.spec_id
        assert len(whisperer.analysis_cache) > 0

    @pytest.mark.asyncio
    async def test_pattern_detection(self, whisperer, sample_traffic):
        """Test pattern detection in traffic samples."""
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context=""
        )

        # Should detect magic number pattern
        magic_patterns = [p for p in spec.patterns if p.pattern_type == "magic_number"]
        assert len(magic_patterns) > 0

        # Check pattern confidence
        for pattern in spec.patterns:
            assert 0.0 <= pattern.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_complexity_assessment(self, whisperer, sample_traffic):
        """Test protocol complexity assessment."""
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context=""
        )

        assert spec.complexity in [
            ProtocolComplexity.SIMPLE,
            ProtocolComplexity.MODERATE,
            ProtocolComplexity.COMPLEX,
            ProtocolComplexity.HIGHLY_COMPLEX,
        ]

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, whisperer, sample_traffic):
        """Test confidence score calculation."""
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context=""
        )

        assert 0.0 <= spec.confidence_score <= 1.0
        # With 20 samples, should have reasonable confidence
        assert spec.confidence_score > 0.3


class TestAdapterCodeGeneration:
    """Test protocol adapter code generation."""

    @pytest.mark.asyncio
    async def test_generate_python_adapter(self, whisperer, sample_traffic):
        """Test Python adapter code generation."""
        # First reverse engineer the protocol
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context="Test protocol"
        )

        # Generate adapter
        adapter = await whisperer.generate_adapter_code(
            legacy_protocol=spec,
            target_protocol="REST",
            language=AdapterLanguage.PYTHON,
        )

        assert isinstance(adapter, AdapterCode)
        assert adapter.source_protocol == spec.protocol_name
        assert adapter.target_protocol == "REST"
        assert adapter.language == AdapterLanguage.PYTHON
        assert len(adapter.adapter_code) > 0
        assert len(adapter.test_code) > 0
        assert len(adapter.documentation) > 0
        assert adapter.code_quality_score > 0.0

    @pytest.mark.asyncio
    async def test_generate_multiple_languages(self, whisperer, sample_traffic):
        """Test adapter generation for multiple languages."""
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context="Test"
        )

        languages = [AdapterLanguage.PYTHON, AdapterLanguage.JAVA, AdapterLanguage.GO]

        for language in languages:
            adapter = await whisperer.generate_adapter_code(
                legacy_protocol=spec, target_protocol="gRPC", language=language
            )

            assert adapter.language == language
            assert len(adapter.adapter_code) > 0

    @pytest.mark.asyncio
    async def test_adapter_caching(self, whisperer, sample_traffic):
        """Test that adapter code is cached."""
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context="Test"
        )

        # First generation
        adapter1 = await whisperer.generate_adapter_code(
            legacy_protocol=spec,
            target_protocol="REST",
            language=AdapterLanguage.PYTHON,
        )

        # Second generation should use cache
        adapter2 = await whisperer.generate_adapter_code(
            legacy_protocol=spec,
            target_protocol="REST",
            language=AdapterLanguage.PYTHON,
        )

        assert adapter1.adapter_id == adapter2.adapter_id
        assert len(whisperer.adapter_cache) > 0

    @pytest.mark.asyncio
    async def test_adapter_dependencies(self, whisperer, sample_traffic):
        """Test dependency extraction from generated code."""
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context="Test"
        )

        adapter = await whisperer.generate_adapter_code(
            legacy_protocol=spec,
            target_protocol="REST",
            language=AdapterLanguage.PYTHON,
        )

        assert isinstance(adapter.dependencies, list)
        # Should have some dependencies for REST adapter
        # (actual dependencies depend on LLM generation)

    @pytest.mark.asyncio
    async def test_configuration_template(self, whisperer, sample_traffic):
        """Test configuration template generation."""
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context="Test"
        )

        adapter = await whisperer.generate_adapter_code(
            legacy_protocol=spec,
            target_protocol="REST",
            language=AdapterLanguage.PYTHON,
        )

        assert len(adapter.configuration_template) > 0
        assert spec.protocol_name in adapter.configuration_template


class TestBehaviorExplanation:
    """Test legacy behavior explanation functionality."""

    @pytest.mark.asyncio
    async def test_explain_simple_behavior(self, whisperer):
        """Test explanation of simple legacy behavior."""
        explanation = await whisperer.explain_legacy_behavior(
            behavior="System uses fixed-width records with EBCDIC encoding",
            context={"system_type": "mainframe", "era": "1980s"},
        )

        assert isinstance(explanation, Explanation)
        assert len(explanation.technical_explanation) > 0
        assert len(explanation.modernization_approaches) > 0
        assert explanation.risk_level in [
            ModernizationRisk.LOW,
            ModernizationRisk.MEDIUM,
            ModernizationRisk.HIGH,
            ModernizationRisk.CRITICAL,
        ]
        assert 0.0 <= explanation.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_modernization_approaches(self, whisperer):
        """Test modernization approach suggestions."""
        explanation = await whisperer.explain_legacy_behavior(
            behavior="Synchronous batch processing with overnight runs",
            context={"system": "financial"},
        )

        assert len(explanation.modernization_approaches) > 0

        for approach in explanation.modernization_approaches:
            assert "name" in approach or "description" in approach

    @pytest.mark.asyncio
    async def test_risk_assessment(self, whisperer):
        """Test risk assessment for modernization."""
        explanation = await whisperer.explain_legacy_behavior(
            behavior="Proprietary binary protocol with no documentation",
            context={"criticality": "high"},
        )

        assert len(explanation.modernization_risks) > 0
        assert explanation.risk_level is not None

    @pytest.mark.asyncio
    async def test_implementation_guidance(self, whisperer):
        """Test implementation guidance generation."""
        explanation = await whisperer.explain_legacy_behavior(
            behavior="Legacy authentication using custom token format", context={}
        )

        assert len(explanation.implementation_steps) > 0
        assert len(explanation.estimated_effort) > 0

    @pytest.mark.asyncio
    async def test_recommended_approach_selection(self, whisperer):
        """Test that a recommended approach is selected."""
        explanation = await whisperer.explain_legacy_behavior(
            behavior="Monolithic application with tight coupling",
            context={"size": "large"},
        )

        if len(explanation.modernization_approaches) > 0:
            assert explanation.recommended_approach is not None


class TestStatisticsAndHealth:
    """Test statistics and health monitoring."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, whisperer):
        """Test statistics retrieval."""
        stats = whisperer.get_statistics()

        assert isinstance(stats, dict)
        assert "analysis_cache_size" in stats
        assert "adapter_cache_size" in stats
        assert "min_samples_required" in stats
        assert "confidence_threshold" in stats

    @pytest.mark.asyncio
    async def test_cache_size_limits(self, whisperer, sample_traffic):
        """Test that cache respects size limits."""
        # Generate many specifications to test cache limit
        for i in range(whisperer.max_cache_size + 5):
            modified_samples = [s + bytes([i]) for s in sample_traffic]
            await whisperer.reverse_engineer_protocol(
                traffic_samples=modified_samples, system_context=f"Test {i}"
            )

        stats = whisperer.get_statistics()
        assert stats["analysis_cache_size"] <= whisperer.max_cache_size


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_samples(self, whisperer):
        """Test handling of empty sample list."""
        with pytest.raises(LegacyWhispererException):
            await whisperer.reverse_engineer_protocol(
                traffic_samples=[], system_context=""
            )

    @pytest.mark.asyncio
    async def test_malformed_samples(self, whisperer):
        """Test handling of malformed samples."""
        # Very short samples
        samples = [b"x" for _ in range(15)]

        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=samples, system_context="Malformed test"
        )

        # Should still produce a specification, even if low confidence
        assert isinstance(spec, ProtocolSpecification)

    @pytest.mark.asyncio
    async def test_unicode_in_context(self, whisperer, sample_traffic):
        """Test handling of Unicode in context."""
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic,
            system_context="Test with émojis 🚀 and ünïcödé",
        )

        assert isinstance(spec, ProtocolSpecification)

    @pytest.mark.asyncio
    async def test_very_large_samples(self, whisperer):
        """Test handling of very large samples."""
        # Create large samples
        large_samples = [bytes([i % 256] * 10000) for i in range(15)]

        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=large_samples, system_context="Large samples test"
        )

        assert isinstance(spec, ProtocolSpecification)


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_modernization_workflow(self, whisperer, sample_traffic):
        """Test complete workflow from analysis to adapter generation."""
        # Step 1: Reverse engineer protocol
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context="Legacy system"
        )

        assert spec.confidence_score > 0.0

        # Step 2: Generate adapter
        adapter = await whisperer.generate_adapter_code(
            legacy_protocol=spec,
            target_protocol="REST",
            language=AdapterLanguage.PYTHON,
        )

        assert len(adapter.adapter_code) > 0

        # Step 3: Explain behavior
        explanation = await whisperer.explain_legacy_behavior(
            behavior=f"Legacy protocol: {spec.protocol_name}",
            context={"protocol_spec": spec.spec_id},
        )

        assert len(explanation.modernization_approaches) > 0

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, whisperer, sample_traffic):
        """Test concurrent protocol analysis operations."""
        # Create multiple analysis tasks
        tasks = []
        for i in range(5):
            modified_samples = [s + bytes([i]) for s in sample_traffic]
            task = whisperer.reverse_engineer_protocol(
                traffic_samples=modified_samples, system_context=f"Concurrent test {i}"
            )
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, ProtocolSpecification)


# Performance Tests


class TestPerformance:
    """Performance and scalability tests."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_sample_set_performance(self, whisperer):
        """Test performance with large sample set."""
        import time

        # Generate 100 samples
        samples = [bytes([i % 256] * 100) for i in range(100)]

        start_time = time.time()
        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=samples, system_context="Performance test"
        )
        elapsed = time.time() - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 60.0  # 60 seconds max
        assert spec.samples_analyzed == 100

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_adapter_generation_performance(self, whisperer, sample_traffic):
        """Test adapter generation performance."""
        import time

        spec = await whisperer.reverse_engineer_protocol(
            traffic_samples=sample_traffic, system_context="Performance test"
        )

        start_time = time.time()
        adapter = await whisperer.generate_adapter_code(
            legacy_protocol=spec,
            target_protocol="REST",
            language=AdapterLanguage.PYTHON,
        )
        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 120.0  # 2 minutes max
        assert len(adapter.adapter_code) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
