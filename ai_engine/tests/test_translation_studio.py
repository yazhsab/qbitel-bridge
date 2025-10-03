"""
CRONOS AI - Protocol Translation Studio Tests
Comprehensive test suite for protocol translation functionality.
"""

import json
import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import struct

from ai_engine.llm.translation_studio import (
    ProtocolTranslationStudio,
    ProtocolSpecification,
    ProtocolField,
    ProtocolType,
    FieldType,
    TranslationRule,
    TranslationRules,
    TranslationStrategy,
    PerformanceData,
    TranslationException
)
from ai_engine.core.config import Config
from ai_engine.llm.unified_llm_service import UnifiedLLMService, LLMResponse


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service."""
    service = Mock(spec=UnifiedLLMService)
    service.process_request = AsyncMock()
    return service


@pytest_asyncio.fixture
async def translation_studio(config, mock_llm_service):
    """Create translation studio instance."""
    studio = ProtocolTranslationStudio(
        config=config,
        llm_service=mock_llm_service
    )
    return studio


@pytest.fixture
def http_spec():
    """Create HTTP protocol specification."""
    return ProtocolSpecification(
        protocol_id="http",
        protocol_type=ProtocolType.HTTP,
        version="1.1",
        name="HTTP",
        description="HTTP Protocol",
        fields=[
            ProtocolField(
                name="method",
                field_type=FieldType.STRING,
                length=10,
                required=True
            ),
            ProtocolField(
                name="path",
                field_type=FieldType.STRING,
                length=256,
                required=True
            )
        ]
    )


@pytest.fixture
def mqtt_spec():
    """Create MQTT protocol specification."""
    return ProtocolSpecification(
        protocol_id="mqtt",
        protocol_type=ProtocolType.MQTT,
        version="3.1.1",
        name="MQTT",
        description="MQTT Protocol",
        fields=[
            ProtocolField(
                name="message_type",
                field_type=FieldType.INTEGER,
                length=1,
                required=True
            ),
            ProtocolField(
                name="topic",
                field_type=FieldType.STRING,
                length=256,
                required=True
            ),
            ProtocolField(
                name="payload",
                field_type=FieldType.BINARY,
                required=True
            )
        ]
    )


class TestProtocolTranslationStudio:
    """Test Protocol Translation Studio functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, translation_studio):
        """Test studio initialization."""
        assert translation_studio is not None
        assert len(translation_studio.protocol_specs) > 0
        assert 'http' in translation_studio.protocol_specs
        assert 'mqtt' in translation_studio.protocol_specs
        assert 'modbus' in translation_studio.protocol_specs

    @pytest.mark.asyncio
    async def test_processing_step_trims_and_lowercases(self, translation_studio):
        """Ensure preprocessing descriptors apply string transformations."""
        payload = {
            'method': '  GET  ',
            'topic': 'Main/Queue ',
        }

        trimmed = await translation_studio._apply_processing_step(payload, 'trim_strings')
        assert trimmed['method'] == 'GET'
        assert trimmed['topic'] == 'Main/Queue'

        lowered = await translation_studio._apply_processing_step(
            trimmed,
            json.dumps({'operation': 'lowercase', 'fields': ['method']})
        )
        assert lowered['method'] == 'get'
        # Field not targeted should remain unchanged
        assert lowered['topic'] == 'Main/Queue'

    @pytest.mark.asyncio
    async def test_processing_step_type_conversion_and_rename(self, translation_studio):
        """Processing steps should convert types and rename fields reliably."""
        payload = {
            'port': '8080',
            'payload': b'data',
        }

        converted = await translation_studio._apply_processing_step(
            payload,
            json.dumps({'operation': 'convert_type', 'fields': ['port'], 'type': 'int'})
        )
        assert converted['port'] == 8080

        renamed = await translation_studio._apply_processing_step(
            converted,
            'rename_field:from=payload,to=body'
        )
        assert 'payload' not in renamed
        assert renamed['body'] == b'data'
    
    @pytest.mark.asyncio
    async def test_register_protocol(self, translation_studio, http_spec):
        """Test protocol registration."""
        custom_spec = ProtocolSpecification(
            protocol_id="custom",
            protocol_type=ProtocolType.CUSTOM,
            version="1.0",
            name="Custom Protocol",
            description="Test protocol",
            fields=[]
        )
        
        await translation_studio.register_protocol(custom_spec)
        assert 'custom' in translation_studio.protocol_specs
    
    @pytest.mark.asyncio
    async def test_parse_message(self, translation_studio):
        """Test message parsing."""
        # Create a simple test message
        spec = ProtocolSpecification(
            protocol_id="test",
            protocol_type=ProtocolType.CUSTOM,
            version="1.0",
            name="Test",
            description="Test protocol",
            fields=[
                ProtocolField(
                    name="id",
                    field_type=FieldType.INTEGER,
                    length=4,
                    required=True
                ),
                ProtocolField(
                    name="name",
                    field_type=FieldType.STRING,
                    length=10,
                    required=True
                )
            ]
        )
        
        # Create test message
        message = struct.pack('>I', 12345) + b'TestName\x00\x00'
        
        parsed = await translation_studio._parse_message(message, spec)
        
        assert parsed['id'] == 12345
        assert parsed['name'] == 'TestName'
    
    @pytest.mark.asyncio
    async def test_generate_message(self, translation_studio):
        """Test message generation."""
        spec = ProtocolSpecification(
            protocol_id="test",
            protocol_type=ProtocolType.CUSTOM,
            version="1.0",
            name="Test",
            description="Test protocol",
            fields=[
                ProtocolField(
                    name="id",
                    field_type=FieldType.INTEGER,
                    length=4,
                    required=True
                ),
                ProtocolField(
                    name="name",
                    field_type=FieldType.STRING,
                    length=10,
                    required=True
                )
            ]
        )
        
        data = {
            'id': 12345,
            'name': 'TestName'
        }
        
        message = await translation_studio._generate_message(data, spec)
        
        assert len(message) >= 14  # 4 bytes + 10 bytes
        assert struct.unpack('>I', message[0:4])[0] == 12345
    
    @pytest.mark.asyncio
    async def test_generate_translation_rules(
        self,
        translation_studio,
        mock_llm_service,
        http_spec,
        mqtt_spec
    ):
        """Test translation rule generation."""
        # Mock LLM response
        mock_llm_service.process_request.return_value = LLMResponse(
            content='{"rules": [{"source_field": "method", "target_field": "message_type", "strategy": "transformation"}], "preprocessing": [], "postprocessing": [], "error_handling": {}, "performance_hints": {}, "test_cases": []}',
            provider="test",
            tokens_used=100,
            processing_time=0.5,
            confidence=0.9
        )
        
        rules = await translation_studio.generate_translation_rules(
            http_spec,
            mqtt_spec
        )
        
        assert rules is not None
        assert len(rules.rules) > 0
        assert rules.source_protocol == http_spec
        assert rules.target_protocol == mqtt_spec
    
    @pytest.mark.asyncio
    async def test_apply_translation_rules(self, translation_studio, http_spec, mqtt_spec):
        """Test applying translation rules."""
        # Create simple rules
        rules = TranslationRules(
            rules_id="test_rules",
            source_protocol=http_spec,
            target_protocol=mqtt_spec,
            rules=[
                TranslationRule(
                    rule_id="rule1",
                    source_field="method",
                    target_field="message_type",
                    strategy=TranslationStrategy.DIRECT_MAPPING
                ),
                TranslationRule(
                    rule_id="rule2",
                    source_field="path",
                    target_field="topic",
                    strategy=TranslationStrategy.DIRECT_MAPPING
                )
            ]
        )
        
        source_data = {
            'method': 'GET',
            'path': '/api/test'
        }
        
        result = await translation_studio._apply_translation_rules(source_data, rules)
        
        assert 'message_type' in result
        assert 'topic' in result
    
    @pytest.mark.asyncio
    async def test_optimize_translation(
        self,
        translation_studio,
        mock_llm_service,
        http_spec,
        mqtt_spec
    ):
        """Test translation optimization."""
        # Create test rules
        rules = TranslationRules(
            rules_id="test_rules",
            source_protocol=http_spec,
            target_protocol=mqtt_spec,
            rules=[
                TranslationRule(
                    rule_id="rule1",
                    source_field="method",
                    target_field="message_type",
                    strategy=TranslationStrategy.DIRECT_MAPPING
                )
            ],
            accuracy=0.95
        )
        
        # Create performance data
        perf_data = PerformanceData(
            total_translations=1000,
            successful_translations=950,
            failed_translations=50,
            average_latency_ms=2.0,
            p50_latency_ms=1.5,
            p95_latency_ms=3.0,
            p99_latency_ms=5.0,
            throughput_per_second=50000,
            error_rate=0.05,
            accuracy=0.95
        )
        
        # Mock LLM response
        mock_llm_service.process_request.return_value = LLMResponse(
            content='{"optimizations": [{"type": "rule_consolidation", "description": "test"}], "optimized_rules": [], "performance_hints": {}}',
            provider="test",
            tokens_used=100,
            processing_time=0.5,
            confidence=0.9
        )
        
        optimized = await translation_studio.optimize_translation(rules, perf_data)
        
        assert optimized is not None
        assert optimized.original_rules == rules
        assert len(optimized.optimizations_applied) > 0
    
    @pytest.mark.asyncio
    async def test_validate_translation(self, translation_studio, mqtt_spec):
        """Test translation validation."""
        source_message = b'test_source'
        target_message = struct.pack('B', 1) + b'test_topic'.ljust(256, b'\x00') + b'payload'
        
        result = await translation_studio._validate_translation(
            source_message,
            target_message,
            mqtt_spec,
            mqtt_spec,
            TranslationRules(
                rules_id="test",
                source_protocol=mqtt_spec,
                target_protocol=mqtt_spec,
                rules=[]
            )
        )
        
        assert 'passed' in result
        assert 'errors' in result
    
    @pytest.mark.asyncio
    async def test_statistics(self, translation_studio):
        """Test statistics retrieval."""
        stats = translation_studio.get_statistics()
        
        assert 'total_translations' in stats
        assert 'successful_translations' in stats
        assert 'failed_translations' in stats
        assert 'rules_generated' in stats
        assert 'cached_rules' in stats
        assert 'registered_protocols' in stats
    
    @pytest.mark.asyncio
    async def test_cache_management(self, translation_studio, http_spec, mqtt_spec):
        """Test translation rules caching."""
        cache_key = translation_studio._get_rules_cache_key(http_spec, mqtt_spec)
        
        assert cache_key is not None
        assert isinstance(cache_key, str)
        
        # Test cache key consistency
        cache_key2 = translation_studio._get_rules_cache_key(http_spec, mqtt_spec)
        assert cache_key == cache_key2
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, translation_studio):
        """Test performance metrics recording."""
        translation_studio._record_translation_metrics(
            "http",
            "mqtt",
            1.5,
            True
        )
        
        assert len(translation_studio.latency_samples) > 0
        assert 'http:mqtt' in translation_studio.latency_samples
    
    @pytest.mark.asyncio
    async def test_error_handling(self, translation_studio):
        """Test error handling in translation."""
        with pytest.raises(TranslationException):
            # Try to translate with invalid protocol
            await translation_studio.translate_protocol(
                "invalid_protocol",
                "another_invalid",
                b"test"
            )
    
    @pytest.mark.asyncio
    async def test_transformation_application(self, translation_studio):
        """Test transformation application."""
        # Test simple transformation
        result = await translation_studio._apply_transformation(
            10,
            "value * 2"
        )
        assert result == 20
        
        # Test string transformation
        result = await translation_studio._apply_transformation(
            "test",
            "value.upper()"
        )
        assert result == "TEST"
    
    @pytest.mark.asyncio
    async def test_validation_expression(self, translation_studio):
        """Test validation expression evaluation."""
        # Test valid expression
        result = await translation_studio._validate_value(10, "value > 5")
        assert result is True
        
        # Test invalid expression
        result = await translation_studio._validate_value(3, "value > 5")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_condition_checking(self, translation_studio):
        """Test condition checking."""
        data = {'field1': 10, 'field2': 'test'}
        
        # Test valid condition
        result = await translation_studio._check_conditions(
            data,
            ["field1 > 5", "field2 == 'test'"]
        )
        assert result is True
        
        # Test invalid condition
        result = await translation_studio._check_conditions(
            data,
            ["field1 < 5"]
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_bottleneck_analysis(self, translation_studio, http_spec, mqtt_spec):
        """Test bottleneck analysis."""
        rules = TranslationRules(
            rules_id="test",
            source_protocol=http_spec,
            target_protocol=mqtt_spec,
            rules=[TranslationRule(
                rule_id=f"rule{i}",
                source_field="test",
                target_field="test",
                strategy=TranslationStrategy.DIRECT_MAPPING
            ) for i in range(150)]  # Many rules to trigger bottleneck
        )
        
        perf_data = PerformanceData(
            total_translations=1000,
            successful_translations=900,
            failed_translations=100,
            average_latency_ms=5.0,  # High latency
            p50_latency_ms=4.0,
            p95_latency_ms=8.0,
            p99_latency_ms=10.0,
            throughput_per_second=10000,  # Low throughput
            error_rate=0.1,  # High error rate
            accuracy=0.85
        )
        
        bottlenecks = await translation_studio._analyze_bottlenecks(rules, perf_data)
        
        assert len(bottlenecks) > 0
        assert any('latency' in b.lower() for b in bottlenecks)
        assert any('throughput' in b.lower() for b in bottlenecks)


class TestProtocolSpecifications:
    """Test protocol specification functionality."""
    
    def test_protocol_field_creation(self):
        """Test protocol field creation."""
        field = ProtocolField(
            name="test_field",
            field_type=FieldType.STRING,
            length=100,
            required=True,
            description="Test field"
        )
        
        assert field.name == "test_field"
        assert field.field_type == FieldType.STRING
        assert field.length == 100
        assert field.required is True
    
    def test_protocol_specification_creation(self):
        """Test protocol specification creation."""
        spec = ProtocolSpecification(
            protocol_id="test",
            protocol_type=ProtocolType.CUSTOM,
            version="1.0",
            name="Test Protocol",
            description="Test",
            fields=[]
        )
        
        assert spec.protocol_id == "test"
        assert spec.protocol_type == ProtocolType.CUSTOM
        assert spec.version == "1.0"


class TestTranslationRules:
    """Test translation rules functionality."""
    
    def test_translation_rule_creation(self):
        """Test translation rule creation."""
        rule = TranslationRule(
            rule_id="test_rule",
            source_field="source",
            target_field="target",
            strategy=TranslationStrategy.DIRECT_MAPPING
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.source_field == "source"
        assert rule.target_field == "target"
        assert rule.strategy == TranslationStrategy.DIRECT_MAPPING
    
    def test_translation_rules_creation(self, http_spec, mqtt_spec):
        """Test translation rules set creation."""
        rules = TranslationRules(
            rules_id="test_rules",
            source_protocol=http_spec,
            target_protocol=mqtt_spec,
            rules=[],
            accuracy=0.95
        )
        
        assert rules.rules_id == "test_rules"
        assert rules.source_protocol == http_spec
        assert rules.target_protocol == mqtt_spec
        assert rules.accuracy == 0.95


@pytest.mark.integration
class TestIntegration:
    """Integration tests for translation studio."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_translation(
        self,
        translation_studio,
        mock_llm_service
    ):
        """Test end-to-end translation flow."""
        # Mock LLM responses
        mock_llm_service.process_request.return_value = LLMResponse(
            content='{"rules": [{"source_field": "method", "target_field": "message_type", "strategy": "direct_mapping"}], "preprocessing": [], "postprocessing": [], "error_handling": {}, "performance_hints": {}, "test_cases": []}',
            provider="test",
            tokens_used=100,
            processing_time=0.5,
            confidence=0.9
        )
        
        # This would be a full integration test with real protocols
        # For now, we verify the studio is properly initialized
        assert translation_studio is not None
        assert len(translation_studio.protocol_specs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
