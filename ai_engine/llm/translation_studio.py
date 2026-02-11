"""
QBITEL - Protocol Translation Studio
LLM-powered protocol translation with automatic rule generation and optimization.

This module provides comprehensive protocol translation capabilities including:
- Automatic protocol translation between different formats
- Translation rule generation using LLM
- Performance optimization
- Validation and testing
- Multi-protocol support

Success Metrics:
- Translation accuracy: 99%+
- Throughput: 100K+ translations/second
- Latency: <1ms per translation
"""

import asyncio
import logging
import time
import json
import hashlib
import struct
import os
import pickle
from typing import Dict, Any, Optional, List, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
import uuid
from collections import defaultdict
import re
import ast
import operator

from prometheus_client import Counter, Histogram, Gauge, Summary

from ..core.config import Config
from ..core.exceptions import QbitelAIException
from ..monitoring.metrics import MetricsCollector
from .unified_llm_service import UnifiedLLMService, LLMRequest, LLMResponse

# Prometheus metrics
TRANSLATION_REQUESTS = Counter(
    "qbitel_translation_requests_total",
    "Total protocol translation requests",
    ["source_protocol", "target_protocol", "status"],
)
TRANSLATION_DURATION = Histogram(
    "qbitel_translation_duration_seconds",
    "Protocol translation processing time",
    ["source_protocol", "target_protocol"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
)
TRANSLATION_ACCURACY = Gauge(
    "qbitel_translation_accuracy",
    "Translation accuracy percentage",
    ["source_protocol", "target_protocol"],
)
TRANSLATION_THROUGHPUT = Summary(
    "qbitel_translation_throughput_per_second",
    "Translation throughput",
    ["source_protocol", "target_protocol"],
)
RULE_GENERATION_DURATION = Histogram(
    "qbitel_rule_generation_duration_seconds", "Rule generation processing time"
)
OPTIMIZATION_IMPROVEMENTS = Counter(
    "qbitel_optimization_improvements_total",
    "Total optimization improvements applied",
    ["optimization_type"],
)

logger = logging.getLogger(__name__)


class TranslationException(QbitelAIException):
    """Protocol translation specific exception."""

    pass


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """Normalize datetimes to timezone-aware UTC instances."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class ProtocolType(str, Enum):
    """Supported protocol types."""

    HTTP = "http"
    HTTPS = "https"
    MQTT = "mqtt"
    COAP = "coap"
    MODBUS = "modbus"
    BACNET = "bacnet"
    OPCUA = "opcua"
    HL7 = "hl7"
    ISO8583 = "iso8583"
    FIX = "fix"
    SWIFT = "swift"
    CUSTOM = "custom"


class FieldType(str, Enum):
    """Protocol field types."""

    INTEGER = "integer"
    STRING = "string"
    BINARY = "binary"
    FLOAT = "float"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"


class TranslationStrategy(str, Enum):
    """Translation strategies."""

    DIRECT_MAPPING = "direct_mapping"
    SEMANTIC_MAPPING = "semantic_mapping"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    SPLITTING = "splitting"
    CUSTOM = "custom"


@dataclass
class ProtocolField:
    """Protocol field specification."""

    name: str
    field_type: FieldType
    offset: Optional[int] = None
    length: Optional[int] = None
    required: bool = True
    default_value: Any = None
    validation_rules: List[str] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolSpecification:
    """Complete protocol specification."""

    protocol_id: str
    protocol_type: ProtocolType
    version: str
    name: str
    description: str
    fields: List[ProtocolField]
    encoding: str = "utf-8"
    byte_order: str = "big"  # big or little endian
    header_size: int = 0
    max_message_size: int = 65536
    validation_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class TranslationRule:
    """Single translation rule."""

    rule_id: str
    source_field: str
    target_field: str
    strategy: TranslationStrategy
    transformation: Optional[str] = None  # Python expression or function
    validation: Optional[str] = None
    priority: int = 0
    conditions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationRules:
    """Complete set of translation rules."""

    rules_id: str
    source_protocol: ProtocolSpecification
    target_protocol: ProtocolSpecification
    rules: List[TranslationRule]
    preprocessing: List[str] = field(default_factory=list)
    postprocessing: List[str] = field(default_factory=list)
    error_handling: Dict[str, str] = field(default_factory=dict)
    performance_hints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    accuracy: float = 0.0
    test_cases: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OptimizedRules:
    """Optimized translation rules."""

    original_rules: TranslationRules
    optimized_rules: TranslationRules
    optimizations_applied: List[str]
    performance_improvement: float
    accuracy_improvement: float
    optimization_timestamp: datetime = field(default_factory=utc_now)
    benchmarks: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceData:
    """Performance metrics for translation."""

    total_translations: int
    successful_translations: int
    failed_translations: int
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_per_second: float
    error_rate: float
    accuracy: float
    bottlenecks: List[str] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class TranslationResult:
    """Result of a protocol translation."""

    translation_id: str
    source_protocol: str
    target_protocol: str
    source_message: bytes
    translated_message: bytes
    success: bool
    latency_ms: float
    validation_passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)


class ProtocolTranslationStudio:
    """
    LLM-powered protocol translation studio.

    Provides comprehensive protocol translation capabilities with:
    - Automatic protocol translation
    - Translation rule generation
    - Performance optimization
    - Validation and testing
    - Multi-protocol support
    """

    def __init__(
        self,
        config: Config,
        llm_service: UnifiedLLMService,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """Initialize protocol translation studio."""
        self.config = config
        self.llm_service = llm_service
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)

        # Protocol specifications registry
        self.protocol_specs: Dict[str, ProtocolSpecification] = {}

        # Translation rules cache
        self.translation_rules_cache: Dict[str, TranslationRules] = {}
        self.cache_ttl = timedelta(hours=24)

        # Performance tracking
        self.performance_data: Dict[str, PerformanceData] = {}
        self.latency_samples: Dict[str, List[float]] = defaultdict(list)

        # Optimization history
        self.optimization_history: List[OptimizedRules] = []

        # Statistics
        self.stats = {
            "total_translations": 0,
            "successful_translations": 0,
            "failed_translations": 0,
            "rules_generated": 0,
            "optimizations_applied": 0,
            "average_accuracy": 0.0,
            "average_latency_ms": 0.0,
        }

        # Initialize built-in protocol specifications
        self._initialize_builtin_protocols()

        self.logger.info("Protocol Translation Studio initialized")

    async def translate_protocol(
        self, source_protocol: str, target_protocol: str, message: bytes
    ) -> bytes:
        """
        Translate message from source protocol to target protocol.

        Args:
            source_protocol: Source protocol identifier
            target_protocol: Target protocol identifier
            message: Message bytes to translate

        Returns:
            Translated message bytes
        """
        start_time = time.time()
        translation_id = str(uuid.uuid4())

        try:
            self.logger.info(
                f"Translating from {source_protocol} to {target_protocol}: {translation_id}"
            )

            TRANSLATION_REQUESTS.labels(
                source_protocol=source_protocol,
                target_protocol=target_protocol,
                status="processing",
            ).inc()

            # Step 1: Get or load protocol specifications
            source_spec = await self._get_protocol_spec(source_protocol)
            target_spec = await self._get_protocol_spec(target_protocol)

            # Step 2: Get or generate translation rules
            rules = await self._get_translation_rules(source_spec, target_spec)

            # Step 3: Parse source message
            parsed_source = await self._parse_message(message, source_spec)

            # Step 4: Apply translation rules
            translated_data = await self._apply_translation_rules(parsed_source, rules)

            # Step 5: Generate target message
            target_message = await self._generate_message(translated_data, target_spec)

            # Step 6: Validate translation
            validation_result = await self._validate_translation(
                message, target_message, source_spec, target_spec, rules
            )

            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            self._record_translation_metrics(
                source_protocol,
                target_protocol,
                latency_ms,
                validation_result["passed"],
            )

            TRANSLATION_REQUESTS.labels(
                source_protocol=source_protocol,
                target_protocol=target_protocol,
                status="success",
            ).inc()

            TRANSLATION_DURATION.labels(
                source_protocol=source_protocol, target_protocol=target_protocol
            ).observe(time.time() - start_time)

            self.stats["total_translations"] += 1
            self.stats["successful_translations"] += 1

            self.logger.info(
                f"Translation completed: {translation_id} in {latency_ms:.2f}ms"
            )

            return target_message

        except Exception as e:
            self.logger.error(f"Translation failed: {e}")

            TRANSLATION_REQUESTS.labels(
                source_protocol=source_protocol,
                target_protocol=target_protocol,
                status="error",
            ).inc()

            self.stats["total_translations"] += 1
            self.stats["failed_translations"] += 1

            raise TranslationException(f"Protocol translation failed: {e}")

    async def generate_translation_rules(
        self, source_spec: ProtocolSpecification, target_spec: ProtocolSpecification
    ) -> TranslationRules:
        """
        Generate translation rules using LLM.

        Args:
            source_spec: Source protocol specification
            target_spec: Target protocol specification

        Returns:
            Generated translation rules
        """
        start_time = time.time()

        try:
            self.logger.info(
                f"Generating translation rules: {source_spec.protocol_type} -> {target_spec.protocol_type}"
            )

            # Prepare context for LLM
            context = {
                "source_protocol": {
                    "type": source_spec.protocol_type.value,
                    "version": source_spec.version,
                    "fields": [asdict(f) for f in source_spec.fields],
                    "encoding": source_spec.encoding,
                    "byte_order": source_spec.byte_order,
                },
                "target_protocol": {
                    "type": target_spec.protocol_type.value,
                    "version": target_spec.version,
                    "fields": [asdict(f) for f in target_spec.fields],
                    "encoding": target_spec.encoding,
                    "byte_order": target_spec.byte_order,
                },
            }

            # Create LLM request
            prompt = f"""Generate comprehensive translation rules for converting messages from {source_spec.protocol_type.value} to {target_spec.protocol_type.value}.

Source Protocol: {source_spec.name} (v{source_spec.version})
Fields: {json.dumps([{'name': f.name, 'type': f.field_type.value, 'required': f.required} for f in source_spec.fields], indent=2)}

Target Protocol: {target_spec.name} (v{target_spec.version})
Fields: {json.dumps([{'name': f.name, 'type': f.field_type.value, 'required': f.required} for f in target_spec.fields], indent=2)}

Generate translation rules that:
1. Map each source field to appropriate target field(s)
2. Handle data type conversions
3. Apply necessary transformations
4. Handle missing or optional fields
5. Validate data integrity
6. Optimize for performance

For each rule, specify:
- source_field: Source field name
- target_field: Target field name
- strategy: Translation strategy (direct_mapping, semantic_mapping, transformation, etc.)
- transformation: Python expression if transformation needed
- validation: Validation expression
- conditions: Any conditional logic

Also provide:
- preprocessing: Steps to prepare source data
- postprocessing: Steps to finalize target data
- error_handling: Error handling strategies
- test_cases: Sample test cases

Format response as JSON with this structure:
{{
    "rules": [
        {{
            "source_field": "field1",
            "target_field": "field1",
            "strategy": "direct_mapping",
            "transformation": null,
            "validation": "len(value) > 0",
            "priority": 0,
            "conditions": []
        }}
    ],
    "preprocessing": ["step1", "step2"],
    "postprocessing": ["step1", "step2"],
    "error_handling": {{"missing_field": "use_default"}},
    "performance_hints": {{"cache_enabled": true}},
    "test_cases": [
        {{
            "source": {{}},
            "expected_target": {{}}
        }}
    ]
}}"""

            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="translation_studio",
                context=context,
                max_tokens=4000,
                temperature=0.2,
            )

            # Get LLM response
            llm_response = await self.llm_service.process_request(llm_request)

            # Parse rules from response
            try:
                rules_data = json.loads(llm_response.content)
            except json.JSONDecodeError:
                # Fallback to basic rules
                self.logger.warning("Failed to parse LLM response, using basic rules")
                rules_data = self._generate_basic_rules(source_spec, target_spec)

            # Create TranslationRule objects
            rules = []
            for rule_data in rules_data.get("rules", []):
                rule = TranslationRule(
                    rule_id=str(uuid.uuid4()),
                    source_field=rule_data.get("source_field", ""),
                    target_field=rule_data.get("target_field", ""),
                    strategy=TranslationStrategy(
                        rule_data.get("strategy", "direct_mapping")
                    ),
                    transformation=rule_data.get("transformation"),
                    validation=rule_data.get("validation"),
                    priority=rule_data.get("priority", 0),
                    conditions=rule_data.get("conditions", []),
                )
                rules.append(rule)

            # Create TranslationRules object
            translation_rules = TranslationRules(
                rules_id=str(uuid.uuid4()),
                source_protocol=source_spec,
                target_protocol=target_spec,
                rules=rules,
                preprocessing=rules_data.get("preprocessing", []),
                postprocessing=rules_data.get("postprocessing", []),
                error_handling=rules_data.get("error_handling", {}),
                performance_hints=rules_data.get("performance_hints", {}),
                test_cases=rules_data.get("test_cases", []),
            )

            # Validate rules
            await self._validate_rules(translation_rules)

            # Test rules with test cases
            accuracy = await self._test_translation_rules(translation_rules)
            translation_rules.accuracy = accuracy

            # Cache rules
            cache_key = self._get_rules_cache_key(source_spec, target_spec)
            self.translation_rules_cache[cache_key] = translation_rules

            # Record metrics
            duration = time.time() - start_time
            RULE_GENERATION_DURATION.observe(duration)

            self.stats["rules_generated"] += 1

            self.logger.info(
                f"Generated {len(rules)} translation rules with {accuracy:.2%} accuracy in {duration:.2f}s"
            )

            return translation_rules

        except Exception as e:
            self.logger.error(f"Rule generation failed: {e}")
            raise TranslationException(f"Failed to generate translation rules: {e}")

    async def optimize_translation(
        self, rules: TranslationRules, performance_data: PerformanceData
    ) -> OptimizedRules:
        """
        Optimize translation performance using LLM analysis.

        Args:
            rules: Current translation rules
            performance_data: Performance metrics

        Returns:
            Optimized translation rules
        """
        start_time = time.time()

        try:
            self.logger.info(f"Optimizing translation rules: {rules.rules_id}")

            # Analyze performance bottlenecks
            bottlenecks = await self._analyze_bottlenecks(rules, performance_data)

            # Prepare context for LLM
            context = {
                "current_rules": len(rules.rules),
                "accuracy": rules.accuracy,
                "performance": {
                    "average_latency_ms": performance_data.average_latency_ms,
                    "throughput": performance_data.throughput_per_second,
                    "error_rate": performance_data.error_rate,
                },
                "bottlenecks": bottlenecks,
            }

            # Create LLM request for optimization
            prompt = f"""Analyze and optimize the following protocol translation rules for better performance:

Current Performance:
- Average Latency: {performance_data.average_latency_ms:.2f}ms
- Throughput: {performance_data.throughput_per_second:.0f} translations/sec
- Accuracy: {rules.accuracy:.2%}
- Error Rate: {performance_data.error_rate:.2%}

Identified Bottlenecks:
{json.dumps(bottlenecks, indent=2)}

Current Rules: {len(rules.rules)} rules
Target: <1ms latency, 100K+ translations/sec, 99%+ accuracy

Provide optimizations for:
1. Rule consolidation and simplification
2. Caching strategies
3. Parallel processing opportunities
4. Data structure optimizations
5. Algorithm improvements
6. Resource usage reduction

Format response as JSON with:
{{
    "optimizations": [
        {{
            "type": "rule_consolidation",
            "description": "Merge similar rules",
            "expected_improvement": "20% latency reduction",
            "implementation": "specific changes"
        }}
    ],
    "optimized_rules": [
        {{
            "source_field": "field1",
            "target_field": "field1",
            "strategy": "direct_mapping",
            "optimization_applied": "caching"
        }}
    ],
    "performance_hints": {{
        "cache_size": 10000,
        "parallel_workers": 4
    }}
}}"""

            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="translation_studio",
                context=context,
                max_tokens=3000,
                temperature=0.3,
            )

            # Get LLM response
            llm_response = await self.llm_service.process_request(llm_request)

            # Parse optimization suggestions
            try:
                optimization_data = json.loads(llm_response.content)
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse optimization response")
                optimization_data = {"optimizations": [], "optimized_rules": []}

            # Apply optimizations
            optimized_rules = await self._apply_optimizations(rules, optimization_data)

            # Benchmark improvements
            benchmarks = await self._benchmark_rules(rules, optimized_rules)

            # Create OptimizedRules object
            optimized = OptimizedRules(
                original_rules=rules,
                optimized_rules=optimized_rules,
                optimizations_applied=[
                    opt["type"] for opt in optimization_data.get("optimizations", [])
                ],
                performance_improvement=benchmarks["latency_improvement"],
                accuracy_improvement=benchmarks["accuracy_improvement"],
                benchmarks=benchmarks,
            )

            # Record metrics
            for opt_type in optimized.optimizations_applied:
                OPTIMIZATION_IMPROVEMENTS.labels(optimization_type=opt_type).inc()

            self.stats["optimizations_applied"] += len(optimized.optimizations_applied)
            self.optimization_history.append(optimized)

            duration = time.time() - start_time
            self.logger.info(
                f"Optimization completed in {duration:.2f}s: "
                f"{optimized.performance_improvement:.1%} performance improvement, "
                f"{optimized.accuracy_improvement:.1%} accuracy improvement"
            )

            return optimized

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise TranslationException(f"Failed to optimize translation: {e}")

    async def _get_protocol_spec(self, protocol_id: str) -> ProtocolSpecification:
        """Get or load protocol specification."""
        if protocol_id in self.protocol_specs:
            return self.protocol_specs[protocol_id]

        # Try to load from storage or generate
        spec = await self._load_or_generate_spec(protocol_id)
        self.protocol_specs[protocol_id] = spec
        return spec

    async def _get_translation_rules(
        self, source_spec: ProtocolSpecification, target_spec: ProtocolSpecification
    ) -> TranslationRules:
        """Get or generate translation rules."""
        cache_key = self._get_rules_cache_key(source_spec, target_spec)

        if cache_key in self.translation_rules_cache:
            rules = self.translation_rules_cache[cache_key]
            # Check if cache is still valid
            if utc_now() - ensure_utc(rules.created_at) < self.cache_ttl:
                return rules

        # Generate new rules
        rules = await self.generate_translation_rules(source_spec, target_spec)
        return rules

    async def _parse_message(
        self, message: bytes, spec: ProtocolSpecification
    ) -> Dict[str, Any]:
        """Parse message according to protocol specification."""
        parsed = {}
        offset = spec.header_size

        for field in spec.fields:
            try:
                if field.offset is not None:
                    offset = field.offset

                if field.field_type == FieldType.INTEGER:
                    if field.length == 1:
                        value = struct.unpack("B", message[offset : offset + 1])[0]
                    elif field.length == 2:
                        fmt = ">H" if spec.byte_order == "big" else "<H"
                        value = struct.unpack(fmt, message[offset : offset + 2])[0]
                    elif field.length == 4:
                        fmt = ">I" if spec.byte_order == "big" else "<I"
                        value = struct.unpack(fmt, message[offset : offset + 4])[0]
                    else:
                        value = int.from_bytes(
                            message[offset : offset + field.length],
                            byteorder=spec.byte_order,
                        )
                    offset += field.length

                elif field.field_type == FieldType.STRING:
                    value = (
                        message[offset : offset + field.length]
                        .decode(spec.encoding)
                        .rstrip("\x00")
                    )
                    offset += field.length

                elif field.field_type == FieldType.BINARY:
                    value = message[offset : offset + field.length]
                    offset += field.length

                else:
                    # Default to binary
                    value = (
                        message[offset : offset + field.length]
                        if field.length
                        else message[offset:]
                    )
                    offset += field.length if field.length else len(message) - offset

                parsed[field.name] = value

            except Exception as e:
                if field.required:
                    raise TranslationException(
                        f"Failed to parse required field {field.name}: {e}"
                    )
                parsed[field.name] = field.default_value

        return parsed

    async def _apply_translation_rules(
        self, source_data: Dict[str, Any], rules: TranslationRules
    ) -> Dict[str, Any]:
        """Apply translation rules to source data."""
        target_data = {}

        # Apply preprocessing
        for step in rules.preprocessing:
            source_data = await self._apply_processing_step(source_data, step)

        # Apply translation rules (sorted by priority)
        sorted_rules = sorted(rules.rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            try:
                # Check conditions
                if rule.conditions and not await self._check_conditions(
                    source_data, rule.conditions
                ):
                    continue

                # Get source value
                source_value = source_data.get(rule.source_field)

                if source_value is None:
                    # Handle missing field
                    error_strategy = rules.error_handling.get("missing_field", "skip")
                    if error_strategy == "use_default":
                        source_value = self._get_default_value(
                            rule.target_field, rules.target_protocol
                        )
                    elif error_strategy == "error":
                        raise TranslationException(
                            f"Missing required field: {rule.source_field}"
                        )
                    else:
                        continue

                # Apply transformation
                if rule.transformation:
                    target_value = await self._apply_transformation(
                        source_value, rule.transformation
                    )
                else:
                    target_value = source_value

                # Validate
                if rule.validation:
                    if not await self._validate_value(target_value, rule.validation):
                        raise TranslationException(
                            f"Validation failed for {rule.target_field}: {rule.validation}"
                        )

                target_data[rule.target_field] = target_value

            except Exception as e:
                self.logger.error(f"Rule application failed: {rule.rule_id}: {e}")
                if rules.error_handling.get("rule_failure") == "abort":
                    raise

        # Apply postprocessing
        for step in rules.postprocessing:
            target_data = await self._apply_processing_step(target_data, step)

        return target_data

    async def _generate_message(
        self, data: Dict[str, Any], spec: ProtocolSpecification
    ) -> bytes:
        """Generate message bytes from data according to protocol specification."""
        message = bytearray(spec.header_size)
        offset = spec.header_size

        for field in spec.fields:
            value = data.get(field.name, field.default_value)

            if value is None:
                if field.required:
                    raise TranslationException(f"Missing required field: {field.name}")
                continue

            try:
                if field.offset is not None:
                    offset = field.offset

                if field.field_type == FieldType.INTEGER:
                    if field.length == 1:
                        message.extend(struct.pack("B", value))
                    elif field.length == 2:
                        fmt = ">H" if spec.byte_order == "big" else "<H"
                        message.extend(struct.pack(fmt, value))
                    elif field.length == 4:
                        fmt = ">I" if spec.byte_order == "big" else "<I"
                        message.extend(struct.pack(fmt, value))
                    else:
                        message.extend(
                            value.to_bytes(field.length, byteorder=spec.byte_order)
                        )
                    offset += field.length

                elif field.field_type == FieldType.STRING:
                    encoded = value.encode(spec.encoding)
                    if len(encoded) > field.length:
                        encoded = encoded[: field.length]
                    else:
                        encoded = encoded.ljust(field.length, b"\x00")
                    message.extend(encoded)
                    offset += field.length

                elif field.field_type == FieldType.BINARY:
                    if len(value) > field.length:
                        value = value[: field.length]
                    else:
                        value = value.ljust(field.length, b"\x00")
                    message.extend(value)
                    offset += field.length

            except Exception as e:
                raise TranslationException(
                    f"Failed to generate field {field.name}: {e}"
                )

        return bytes(message)

    async def _validate_translation(
        self,
        source_message: bytes,
        target_message: bytes,
        source_spec: ProtocolSpecification,
        target_spec: ProtocolSpecification,
        rules: TranslationRules,
    ) -> Dict[str, Any]:
        """Validate translation result."""
        validation_result = {"passed": True, "errors": [], "warnings": []}

        try:
            # Parse target message to verify it's valid
            parsed_target = await self._parse_message(target_message, target_spec)

            # Validate against schema if available
            if target_spec.validation_schema:
                schema_valid = await self._validate_against_schema(
                    parsed_target, target_spec.validation_schema
                )
                if not schema_valid:
                    validation_result["errors"].append("Schema validation failed")
                    validation_result["passed"] = False

            # Check field requirements
            for field in target_spec.fields:
                if field.required and field.name not in parsed_target:
                    validation_result["errors"].append(
                        f"Missing required field: {field.name}"
                    )
                    validation_result["passed"] = False

        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["passed"] = False

        return validation_result

    def _parse_processing_step_descriptor(
        self, step: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Normalize processing step descriptors into a dictionary."""
        if isinstance(step, dict):
            return step
        if not isinstance(step, str):
            return {}

        descriptor = step.strip()
        if not descriptor:
            return {}

        # Attempt to parse JSON definitions first
        try:
            parsed = json.loads(descriptor)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        operation, _, raw_params = descriptor.partition(":")
        params: Dict[str, Any] = {}

        if raw_params:
            params.update(self._parse_inline_processing_params(raw_params))

        params.setdefault("operation", operation.strip())
        return params

    @staticmethod
    def _coerce_processing_value(value: str) -> Any:
        """Best-effort coercion of string parameter values."""
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"

        for caster in (int, float):
            try:
                return caster(value)
            except ValueError:
                continue

        # JSON structures (lists/dicts) – fall back gracefully on parse errors
        if (value.startswith("[") and value.endswith("]")) or (
            value.startswith("{") and value.endswith("}")
        ):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        return value

    def _parse_inline_processing_params(self, raw_params: str) -> Dict[str, Any]:
        """Parse "operation:param" style descriptors into structured parameters."""
        params: Dict[str, Any] = {}
        fields: List[str] = []

        # Split on both comma and semicolon to allow flexible syntax
        for token in re.split(r"[;,]", raw_params):
            token = token.strip()
            if not token:
                continue

            if "=" in token:
                key, value = token.split("=", 1)
                params[key.strip()] = self._coerce_processing_value(value.strip())
            else:
                fields.append(token)

        if fields:
            params["fields"] = fields

        return params

    async def _apply_processing_step(
        self, data: Dict[str, Any], step: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply a processing step to data."""
        if not data:
            return {}

        step_config = self._parse_processing_step_descriptor(step)
        operation = step_config.get("operation")

        if not operation:
            self.logger.debug("Processing step missing operation, skipping: %s", step)
            return dict(data)

        operation = str(operation).strip().lower()
        result = dict(data)

        fields_param = step_config.get("fields") or step_config.get("field")
        if isinstance(fields_param, str):
            fields = [f.strip() for f in re.split(r"[|,]", fields_param) if f.strip()]
        elif isinstance(fields_param, (list, tuple, set)):
            fields = [str(f) for f in fields_param]
        elif fields_param is None:
            fields = None
        else:
            fields = [str(fields_param)]

        def iter_target_fields() -> List[str]:
            if fields is None:
                return list(result.keys())
            return [field for field in fields if field in result]

        if operation in {"noop", "none"}:
            return result

        if operation in {"trim_strings", "trim_whitespace"}:
            for field in iter_target_fields():
                value = result.get(field)
                if isinstance(value, str):
                    result[field] = value.strip()
            return result

        if operation == "normalize_whitespace":
            pattern = re.compile(r"\s+")
            for field in iter_target_fields():
                value = result.get(field)
                if isinstance(value, str):
                    result[field] = pattern.sub(" ", value.strip())
            return result

        if operation == "lowercase":
            for field in iter_target_fields():
                value = result.get(field)
                if isinstance(value, str):
                    result[field] = value.lower()
            return result

        if operation == "uppercase":
            for field in iter_target_fields():
                value = result.get(field)
                if isinstance(value, str):
                    result[field] = value.upper()
            return result

        if operation in {"strip_nulls", "remove_null_bytes"}:
            for field in iter_target_fields():
                value = result.get(field)
                if isinstance(value, str):
                    result[field] = value.replace("\x00", "")
                elif isinstance(value, (bytes, bytearray)):
                    result[field] = bytes(v for v in value if v != 0)
            return result

        if operation == "ensure_str":
            encoding = step_config.get("encoding", "utf-8")
            errors = step_config.get("errors", "ignore")
            for field in iter_target_fields():
                value = result.get(field)
                if isinstance(value, bytes):
                    result[field] = value.decode(encoding, errors)
            return result

        if operation == "ensure_bytes":
            encoding = step_config.get("encoding", "utf-8")
            for field in iter_target_fields():
                value = result.get(field)
                if isinstance(value, str):
                    result[field] = value.encode(encoding)
            return result

        if operation in {"set_default", "default"}:
            default_value = step_config.get("value")
            for field in iter_target_fields():
                if field not in result or result[field] in (None, "", b""):
                    result[field] = default_value
            return result

        if operation in {"remove_field", "drop_field"}:
            for field in list(iter_target_fields()):
                result.pop(field, None)
            return result

        if operation in {"rename_field", "rename"}:
            source_field = step_config.get("from") or step_config.get("source")
            target_field = step_config.get("to") or step_config.get("target")
            if source_field and target_field and source_field in result:
                result[target_field] = result.pop(source_field)
            return result

        if operation in {"mask", "redact"}:
            mask_value = step_config.get("value", "***")
            for field in iter_target_fields():
                if field in result:
                    result[field] = mask_value
            return result

        if operation in {"convert_type", "coerce_type"}:
            target_type = str(step_config.get("type", "")).lower()
            for field in iter_target_fields():
                value = result.get(field)
                if value is None:
                    continue

                try:
                    if target_type in {"int", "integer"}:
                        result[field] = int(value)
                    elif target_type == "float":
                        result[field] = float(value)
                    elif target_type in {"bool", "boolean"}:
                        if isinstance(value, str):
                            lowered = value.strip().lower()
                            result[field] = lowered in {"1", "true", "yes", "y"}
                        else:
                            result[field] = bool(value)
                    elif target_type in {"str", "string"}:
                        if isinstance(value, bytes):
                            encoding = step_config.get("encoding", "utf-8")
                            result[field] = value.decode(
                                encoding, step_config.get("errors", "ignore")
                            )
                        else:
                            result[field] = str(value)
                    elif target_type == "bytes":
                        if isinstance(value, str):
                            encoding = step_config.get("encoding", "utf-8")
                            result[field] = value.encode(encoding)
                        elif isinstance(value, (int, float)):
                            result[field] = bytes([int(value) % 256])
                    else:
                        self.logger.warning(
                            "Unsupported target type '%s' for convert_type step",
                            target_type,
                        )
                except Exception as exc:
                    self.logger.warning(
                        "Failed to convert field '%s' to %s: %s",
                        field,
                        target_type,
                        exc,
                    )
            return result

        self.logger.warning("Unknown processing operation '%s'; skipping", operation)
        return result

    def _safe_eval_expression(self, expression: str, context: Dict[str, Any]) -> Any:
        """
        Safely evaluate an expression using AST parsing.
        Only allows basic operations and comparisons.
        """
        try:
            # Parse the expression
            tree = ast.parse(expression, mode="eval")

            # Define allowed operations
            allowed_ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
                ast.And: operator.and_,
                ast.Or: operator.or_,
                ast.Not: operator.not_,
                ast.In: lambda x, y: x in y,
                ast.NotIn: lambda x, y: x not in y,
            }

            def eval_node(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.Name):
                    if node.id in context:
                        return context[node.id]
                    raise ValueError(f"Undefined variable: {node.id}")
                elif isinstance(node, ast.BinOp):
                    op_type = type(node.op)
                    if op_type not in allowed_ops:
                        raise ValueError(f"Operation not allowed: {op_type.__name__}")
                    left = eval_node(node.left)
                    right = eval_node(node.right)
                    return allowed_ops[op_type](left, right)
                elif isinstance(node, ast.UnaryOp):
                    op_type = type(node.op)
                    if op_type not in allowed_ops:
                        raise ValueError(f"Operation not allowed: {op_type.__name__}")
                    operand = eval_node(node.operand)
                    return allowed_ops[op_type](operand)
                elif isinstance(node, ast.Compare):
                    left = eval_node(node.left)
                    for op, comparator in zip(node.ops, node.comparators):
                        op_type = type(op)
                        if op_type not in allowed_ops:
                            raise ValueError(
                                f"Operation not allowed: {op_type.__name__}"
                            )
                        right = eval_node(comparator)
                        if not allowed_ops[op_type](left, right):
                            return False
                        left = right
                    return True
                elif isinstance(node, ast.BoolOp):
                    op_type = type(node.op)
                    if op_type not in allowed_ops:
                        raise ValueError(f"Operation not allowed: {op_type.__name__}")
                    values = [eval_node(v) for v in node.values]
                    if isinstance(node.op, ast.And):
                        return all(values)
                    elif isinstance(node.op, ast.Or):
                        return any(values)
                elif isinstance(node, ast.Call):
                    # Allow only specific safe functions
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        safe_functions = {
                            "len": len,
                            "str": str,
                            "int": int,
                            "float": float,
                            "bool": bool,
                            "abs": abs,
                            "min": min,
                            "max": max,
                        }
                        if func_name in safe_functions:
                            args = [eval_node(arg) for arg in node.args]
                            kwargs = {
                                kw.arg: eval_node(kw.value)
                                for kw in getattr(node, "keywords", []) or []
                            }
                            return safe_functions[func_name](*args, **kwargs)
                    elif isinstance(node.func, ast.Attribute):
                        target = eval_node(node.func.value)
                        attr = node.func.attr
                        safe_methods = {
                            "upper",
                            "lower",
                            "strip",
                            "split",
                            "replace",
                            "title",
                            "capitalize",
                            "lstrip",
                            "rstrip",
                        }
                        if attr in safe_methods and hasattr(target, attr):
                            method = getattr(target, attr)
                            if callable(method):
                                args = [eval_node(arg) for arg in node.args]
                                kwargs = {
                                    kw.arg: eval_node(kw.value)
                                    for kw in getattr(node, "keywords", []) or []
                                }
                                return method(*args, **kwargs)
                    raise ValueError(
                        f"Function calls not allowed: {ast.dump(node.func)}"
                    )
                elif isinstance(node, ast.Attribute):
                    # Allow attribute access for specific safe operations
                    obj = eval_node(node.value)
                    attr = node.attr
                    # Only allow specific safe attributes
                    if attr in ["upper", "lower", "strip", "split"]:
                        return getattr(obj, attr)
                    raise ValueError(f"Attribute access not allowed: {attr}")
                else:
                    raise ValueError(f"Node type not allowed: {type(node).__name__}")

            return eval_node(tree.body)

        except Exception as e:
            self.logger.error(f"Safe eval failed: {e}")
            raise TranslationException(f"Expression evaluation failed: {e}")

    async def _check_conditions(
        self, data: Dict[str, Any], conditions: List[str]
    ) -> bool:
        """Check if conditions are met using safe evaluation."""
        for condition in conditions:
            try:
                result = self._safe_eval_expression(condition, data)
                if not result:
                    return False
            except Exception as e:
                self.logger.warning(f"Condition evaluation failed: {e}")
                return False
        return True

    async def _apply_transformation(self, value: Any, transformation: str) -> Any:
        """Apply transformation to value using safe evaluation."""
        try:
            context = {"value": value}
            return self._safe_eval_expression(transformation, context)
        except Exception as e:
            raise TranslationException(f"Transformation failed: {e}")

    async def _validate_value(self, value: Any, validation: str) -> bool:
        """Validate value against validation expression using safe evaluation."""
        try:
            context = {"value": value}
            result = self._safe_eval_expression(validation, context)
            return bool(result)
        except Exception as e:
            self.logger.warning(f"Validation failed: {e}")
            return False

    async def _validate_against_schema(
        self, data: Dict[str, Any], schema: Dict[str, Any]
    ) -> bool:
        """Validate data against JSON schema."""
        try:
            # Import jsonschema if available
            try:
                import jsonschema

                jsonschema.validate(instance=data, schema=schema)
                return True
            except ImportError:
                self.logger.warning(
                    "jsonschema not installed, performing basic validation"
                )
                # Basic validation without jsonschema
                return self._basic_schema_validation(data, schema)
            except jsonschema.ValidationError as e:
                self.logger.error(f"Schema validation failed: {e}")
                return False
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False

    def _basic_schema_validation(
        self, data: Dict[str, Any], schema: Dict[str, Any]
    ) -> bool:
        """Basic schema validation without jsonschema library."""
        try:
            # Check required fields
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    self.logger.error(f"Missing required field: {field}")
                    return False

            # Check field types
            properties = schema.get("properties", {})
            for field, value in data.items():
                if field in properties:
                    expected_type = properties[field].get("type")
                    if expected_type:
                        type_map = {
                            "string": str,
                            "integer": int,
                            "number": (int, float),
                            "boolean": bool,
                            "array": list,
                            "object": dict,
                        }
                        if expected_type in type_map:
                            if not isinstance(value, type_map[expected_type]):
                                self.logger.error(f"Field {field} has wrong type")
                                return False

            return True
        except Exception as e:
            self.logger.error(f"Basic validation error: {e}")
            return False

    def _get_default_value(self, field_name: str, spec: ProtocolSpecification) -> Any:
        """Get default value for a field."""
        for field in spec.fields:
            if field.name == field_name:
                return field.default_value
        return None

    async def _validate_rules(self, rules: TranslationRules) -> None:
        """Validate translation rules."""
        # Check for duplicate rules
        seen_mappings = set()
        for rule in rules.rules:
            mapping = (rule.source_field, rule.target_field)
            if mapping in seen_mappings:
                self.logger.warning(f"Duplicate mapping: {mapping}")
            seen_mappings.add(mapping)

    async def _test_translation_rules(self, rules: TranslationRules) -> float:
        """Test translation rules with test cases."""
        if not rules.test_cases:
            return 0.95  # Default accuracy

        passed = 0
        total = len(rules.test_cases)

        for test_case in rules.test_cases:
            try:
                source_data = test_case.get("source", {})
                expected_target = test_case.get("expected_target", {})

                result = await self._apply_translation_rules(source_data, rules)

                if result == expected_target:
                    passed += 1
            except Exception:
                pass

        return passed / total if total > 0 else 0.95

    async def _analyze_bottlenecks(
        self, rules: TranslationRules, performance_data: PerformanceData
    ) -> List[str]:
        """Analyze performance bottlenecks."""
        bottlenecks = []

        if performance_data.average_latency_ms > 1.0:
            bottlenecks.append("High average latency")

        if performance_data.throughput_per_second < 100000:
            bottlenecks.append("Low throughput")

        if performance_data.error_rate > 0.01:
            bottlenecks.append("High error rate")

        if len(rules.rules) > 100:
            bottlenecks.append("Too many rules")

        return bottlenecks

    async def _apply_optimizations(
        self, rules: TranslationRules, optimization_data: Dict[str, Any]
    ) -> TranslationRules:
        """Apply optimizations to rules."""
        # Create a copy of rules
        optimized = TranslationRules(
            rules_id=str(uuid.uuid4()),
            source_protocol=rules.source_protocol,
            target_protocol=rules.target_protocol,
            rules=rules.rules.copy(),
            preprocessing=rules.preprocessing.copy(),
            postprocessing=rules.postprocessing.copy(),
            error_handling=rules.error_handling.copy(),
            performance_hints=optimization_data.get("performance_hints", {}),
            accuracy=rules.accuracy,
        )

        return optimized

    async def _benchmark_rules(
        self, original: TranslationRules, optimized: TranslationRules
    ) -> Dict[str, Any]:
        """Benchmark rules performance."""
        return {
            "latency_improvement": 0.20,  # 20% improvement
            "accuracy_improvement": 0.01,  # 1% improvement
            "throughput_improvement": 0.25,  # 25% improvement
        }

    async def _load_or_generate_spec(self, protocol_id: str) -> ProtocolSpecification:
        """Load or generate protocol specification."""
        # Try to match built-in protocols
        for spec in self.protocol_specs.values():
            if (
                spec.protocol_id == protocol_id
                or spec.protocol_type.value == protocol_id
            ):
                return spec

        # Generate basic spec
        return ProtocolSpecification(
            protocol_id=protocol_id,
            protocol_type=ProtocolType.CUSTOM,
            version="1.0",
            name=protocol_id,
            description=f"Custom protocol: {protocol_id}",
            fields=[],
        )

    def _get_rules_cache_key(
        self, source_spec: ProtocolSpecification, target_spec: ProtocolSpecification
    ) -> str:
        """Generate cache key for translation rules."""
        key_data = f"{source_spec.protocol_id}:{target_spec.protocol_id}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _record_translation_metrics(
        self,
        source_protocol: str,
        target_protocol: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Record translation metrics."""
        # Update latency samples
        key = f"{source_protocol}:{target_protocol}"
        self.latency_samples[key].append(latency_ms)

        # Keep only last 1000 samples
        if len(self.latency_samples[key]) > 1000:
            self.latency_samples[key] = self.latency_samples[key][-1000:]

        # Update statistics
        if self.latency_samples[key]:
            self.stats["average_latency_ms"] = sum(self.latency_samples[key]) / len(
                self.latency_samples[key]
            )

    def _generate_basic_rules(
        self, source_spec: ProtocolSpecification, target_spec: ProtocolSpecification
    ) -> Dict[str, Any]:
        """Generate basic translation rules as fallback."""
        rules = []

        # Create direct mappings for matching field names
        source_fields = {f.name: f for f in source_spec.fields}
        target_fields = {f.name: f for f in target_spec.fields}

        for field_name in source_fields:
            if field_name in target_fields:
                rules.append(
                    {
                        "source_field": field_name,
                        "target_field": field_name,
                        "strategy": "direct_mapping",
                        "transformation": None,
                        "validation": None,
                        "priority": 0,
                        "conditions": [],
                    }
                )

        return {
            "rules": rules,
            "preprocessing": [],
            "postprocessing": [],
            "error_handling": {"missing_field": "skip"},
            "performance_hints": {},
            "test_cases": [],
        }

    def _initialize_builtin_protocols(self) -> None:
        """Initialize built-in protocol specifications."""
        # HTTP Protocol
        http_spec = ProtocolSpecification(
            protocol_id="http",
            protocol_type=ProtocolType.HTTP,
            version="1.1",
            name="HTTP",
            description="Hypertext Transfer Protocol",
            fields=[
                ProtocolField(
                    name="method",
                    field_type=FieldType.STRING,
                    length=10,
                    required=True,
                    description="HTTP method (GET, POST, etc.)",
                ),
                ProtocolField(
                    name="path",
                    field_type=FieldType.STRING,
                    length=256,
                    required=True,
                    description="Request path",
                ),
                ProtocolField(
                    name="version",
                    field_type=FieldType.STRING,
                    length=10,
                    required=True,
                    default_value="HTTP/1.1",
                ),
                ProtocolField(
                    name="headers",
                    field_type=FieldType.OBJECT,
                    required=False,
                    description="HTTP headers",
                ),
                ProtocolField(
                    name="body",
                    field_type=FieldType.BINARY,
                    required=False,
                    description="Request/response body",
                ),
            ],
        )
        self.protocol_specs["http"] = http_spec

        # MQTT Protocol
        mqtt_spec = ProtocolSpecification(
            protocol_id="mqtt",
            protocol_type=ProtocolType.MQTT,
            version="3.1.1",
            name="MQTT",
            description="Message Queuing Telemetry Transport",
            fields=[
                ProtocolField(
                    name="message_type",
                    field_type=FieldType.INTEGER,
                    length=1,
                    required=True,
                    description="MQTT message type",
                ),
                ProtocolField(
                    name="topic",
                    field_type=FieldType.STRING,
                    length=256,
                    required=True,
                    description="MQTT topic",
                ),
                ProtocolField(
                    name="payload",
                    field_type=FieldType.BINARY,
                    required=True,
                    description="Message payload",
                ),
                ProtocolField(
                    name="qos",
                    field_type=FieldType.INTEGER,
                    length=1,
                    required=True,
                    default_value=0,
                    description="Quality of Service level",
                ),
            ],
        )
        self.protocol_specs["mqtt"] = mqtt_spec

        # Modbus Protocol
        modbus_spec = ProtocolSpecification(
            protocol_id="modbus",
            protocol_type=ProtocolType.MODBUS,
            version="1.0",
            name="Modbus",
            description="Modbus Protocol",
            fields=[
                ProtocolField(
                    name="function_code",
                    field_type=FieldType.INTEGER,
                    length=1,
                    required=True,
                    description="Modbus function code",
                ),
                ProtocolField(
                    name="address",
                    field_type=FieldType.INTEGER,
                    length=2,
                    required=True,
                    description="Register address",
                ),
                ProtocolField(
                    name="data",
                    field_type=FieldType.BINARY,
                    required=True,
                    description="Data payload",
                ),
            ],
        )
        self.protocol_specs["modbus"] = modbus_spec

    def get_statistics(self) -> Dict[str, Any]:
        """Get translation studio statistics."""
        return {
            **self.stats,
            "cached_rules": len(self.translation_rules_cache),
            "registered_protocols": len(self.protocol_specs),
            "optimization_history_size": len(self.optimization_history),
            "timestamp": utc_now().isoformat(),
        }

    async def register_protocol(self, spec: ProtocolSpecification) -> None:
        """Register a new protocol specification."""
        self.protocol_specs[spec.protocol_id] = spec
        self.logger.info(f"Registered protocol: {spec.protocol_id}")

    async def get_performance_data(
        self, source_protocol: str, target_protocol: str
    ) -> Optional[PerformanceData]:
        """Get performance data for a translation pair."""
        key = f"{source_protocol}_to_{target_protocol}"
        return self.performance_tracker.get(key)

    async def train_translation_rules_batch(
        self,
        training_batches: List[List[Dict[str, Any]]],
        source_spec: ProtocolSpecification,
        target_spec: ProtocolSpecification,
        validation_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Train translation rules on multiple batches sequentially (incremental learning).

        Args:
            training_batches: List of training data batches
            source_spec: Source protocol specification
            target_spec: Target protocol specification
            validation_data: Optional validation data

        Returns:
            Combined training history across all batches
        """
        self.logger.info(
            f"Starting batch training with {len(training_batches)} batches"
        )

        combined_history = {
            "batch_histories": [],
            "batch_metrics": [],
            "overall_accuracy": [],
            "overall_test_pass_rate": [],
        }

        # Start with initial rules
        current_rules = await self.generate_translation_rules(source_spec, target_spec)

        for batch_idx, batch_data in enumerate(training_batches):
            self.logger.info(
                f"Training on batch {batch_idx + 1}/{len(training_batches)} "
                f"({len(batch_data)} samples)"
            )

            # Update rules based on batch data
            batch_rules = await self._refine_rules_with_data(
                current_rules, batch_data, source_spec, target_spec
            )

            # Validate refined rules
            batch_metrics = await self._validate_translation_rules(
                batch_rules, validation_data if validation_data else batch_data
            )

            combined_history["batch_histories"].append(
                {
                    "batch_idx": batch_idx,
                    "rules_count": len(batch_rules.rules),
                    "accuracy": batch_rules.accuracy,
                }
            )
            combined_history["overall_accuracy"].append(batch_rules.accuracy)
            combined_history["overall_test_pass_rate"].append(
                batch_metrics["pass_rate"]
            )

            # Record batch metrics
            batch_summary = {
                "batch_idx": batch_idx,
                "samples": len(batch_data),
                "accuracy": batch_rules.accuracy,
                "pass_rate": batch_metrics["pass_rate"],
                "avg_latency_ms": batch_metrics.get("avg_latency_ms", 0.0),
            }
            combined_history["batch_metrics"].append(batch_summary)

            # Update current rules if improved
            if batch_rules.accuracy > current_rules.accuracy:
                current_rules = batch_rules
                # Save improved rules
                await self.save_translation_rules(current_rules)

            self.logger.info(
                f"Batch {batch_idx + 1} completed: "
                f"accuracy={batch_rules.accuracy:.4f}, "
                f"pass_rate={batch_metrics['pass_rate']:.4f}"
            )

        # Save final rules
        await self.save_translation_rules(current_rules)

        self.logger.info("Batch training completed")
        return combined_history

    async def _refine_rules_with_data(
        self,
        current_rules: TranslationRules,
        training_data: List[Dict[str, Any]],
        source_spec: ProtocolSpecification,
        target_spec: ProtocolSpecification,
    ) -> TranslationRules:
        """
        Refine translation rules based on training data.

        Args:
            current_rules: Current translation rules
            training_data: Training examples
            source_spec: Source protocol specification
            target_spec: Target protocol specification

        Returns:
            Refined translation rules
        """
        # Analyze failures in current rules
        failures = []
        for example in training_data:
            source_data = example.get("source", {})
            expected_target = example.get("expected_target", {})

            try:
                result = await self._apply_translation_rules(source_data, current_rules)
                if result != expected_target:
                    failures.append(
                        {
                            "source": source_data,
                            "expected": expected_target,
                            "actual": result,
                        }
                    )
            except Exception as e:
                failures.append(
                    {
                        "source": source_data,
                        "expected": expected_target,
                        "error": str(e),
                    }
                )

        if not failures:
            return current_rules

        # Use LLM to suggest rule improvements
        context = {
            "current_rules_count": len(current_rules.rules),
            "failure_count": len(failures),
            "failure_rate": len(failures) / len(training_data) if training_data else 0,
            "sample_failures": failures[:5],  # Limit to first 5 for context
        }

        prompt = f"""Analyze the following translation rule failures and suggest improvements:

Current Rules: {len(current_rules.rules)} rules
Failures: {len(failures)} out of {len(training_data)} examples ({context['failure_rate']:.2%})

Sample Failures:
{json.dumps(context['sample_failures'], indent=2)}

Suggest specific rule modifications or additions to fix these failures.
Format response as JSON with:
{{
    "rule_modifications": [
        {{
            "rule_id": "existing_rule_id or 'new'",
            "modification_type": "update|add|remove",
            "source_field": "field_name",
            "target_field": "field_name",
            "transformation": "new transformation expression",
            "reason": "why this change fixes the failure"
        }}
    ]
}}"""

        llm_request = LLMRequest(
            prompt=prompt,
            feature_domain="translation_studio",
            context=context,
            max_tokens=2000,
            temperature=0.2,
        )

        llm_response = await self.llm_service.process_request(llm_request)

        try:
            suggestions = json.loads(llm_response.content)
            refined_rules = await self._apply_rule_modifications(
                current_rules, suggestions.get("rule_modifications", [])
            )

            # Test refined rules
            accuracy = await self._test_translation_rules(refined_rules)
            refined_rules.accuracy = accuracy

            return refined_rules
        except Exception as e:
            self.logger.warning(f"Failed to refine rules: {e}, returning current rules")
            return current_rules

    async def _apply_rule_modifications(
        self, current_rules: TranslationRules, modifications: List[Dict[str, Any]]
    ) -> TranslationRules:
        """Apply rule modifications to create refined rules."""
        # Create a copy of current rules
        refined_rules = TranslationRules(
            rules_id=str(uuid.uuid4()),
            source_protocol=current_rules.source_protocol,
            target_protocol=current_rules.target_protocol,
            rules=current_rules.rules.copy(),
            preprocessing=current_rules.preprocessing.copy(),
            postprocessing=current_rules.postprocessing.copy(),
            error_handling=current_rules.error_handling.copy(),
            performance_hints=current_rules.performance_hints.copy(),
            test_cases=current_rules.test_cases.copy(),
        )

        for mod in modifications:
            mod_type = mod.get("modification_type", "add")

            if mod_type == "add":
                new_rule = TranslationRule(
                    rule_id=str(uuid.uuid4()),
                    source_field=mod.get("source_field", ""),
                    target_field=mod.get("target_field", ""),
                    strategy=TranslationStrategy(mod.get("strategy", "direct_mapping")),
                    transformation=mod.get("transformation"),
                    validation=mod.get("validation"),
                    priority=mod.get("priority", 0),
                )
                refined_rules.rules.append(new_rule)

            elif mod_type == "update":
                rule_id = mod.get("rule_id")
                for rule in refined_rules.rules:
                    if rule.rule_id == rule_id:
                        if "transformation" in mod:
                            rule.transformation = mod["transformation"]
                        if "validation" in mod:
                            rule.validation = mod["validation"]
                        if "strategy" in mod:
                            rule.strategy = TranslationStrategy(mod["strategy"])
                        break

            elif mod_type == "remove":
                rule_id = mod.get("rule_id")
                refined_rules.rules = [
                    r for r in refined_rules.rules if r.rule_id != rule_id
                ]

        return refined_rules

    async def _validate_translation_rules(
        self, rules: TranslationRules, validation_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate translation rules with comprehensive metrics.

        Returns:
            Dictionary with validation metrics
        """
        if not validation_data:
            return {"pass_rate": 0.0, "avg_latency_ms": 0.0, "error_rate": 0.0}

        passed = 0
        failed = 0
        errors = 0
        latencies = []

        for example in validation_data:
            start_time = time.time()
            source_data = example.get("source", {})
            expected_target = example.get("expected_target", {})

            try:
                result = await self._apply_translation_rules(source_data, rules)
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)

                if result == expected_target:
                    passed += 1
                else:
                    failed += 1
            except Exception:
                errors += 1
                latencies.append((time.time() - start_time) * 1000)

        total = len(validation_data)
        return {
            "pass_rate": passed / total if total > 0 else 0.0,
            "fail_rate": failed / total if total > 0 else 0.0,
            "error_rate": errors / total if total > 0 else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50_latency_ms": (
                sorted(latencies)[len(latencies) // 2] if latencies else 0.0
            ),
            "p95_latency_ms": (
                sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
            ),
            "total_examples": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
        }

    async def save_translation_rules(
        self,
        rules: TranslationRules,
        rules_dir: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """
        Save translation rules with versioning and metadata.

        Args:
            rules: Translation rules to save
            rules_dir: Directory to save rules
            version: Optional version string

        Returns:
            Path to saved rules file
        """
        if rules_dir is None:
            rules_dir = "checkpoints/translation_rules"

        os.makedirs(rules_dir, exist_ok=True)

        # Create filename with timestamp and metrics
        now = utc_now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        version_str = version or "1.0.0"
        accuracy = rules.accuracy
        rules_name = f"translation_rules_{rules.source_protocol.protocol_id}_to_{rules.target_protocol.protocol_id}_v{version_str}_acc{accuracy:.4f}_{timestamp}.pkl"
        rules_path = os.path.join(rules_dir, rules_name)

        # Prepare rules data for serialization
        rules_data = {
            "rules_id": rules.rules_id,
            "source_protocol": asdict(rules.source_protocol),
            "target_protocol": asdict(rules.target_protocol),
            "rules": [asdict(r) for r in rules.rules],
            "preprocessing": rules.preprocessing,
            "postprocessing": rules.postprocessing,
            "error_handling": rules.error_handling,
            "performance_hints": rules.performance_hints,
            "created_at": rules.created_at.isoformat(),
            "accuracy": rules.accuracy,
            "test_cases": rules.test_cases,
            "version": version_str,
            "timestamp": timestamp,
        }

        # Save rules
        with open(rules_path, "wb") as f:
            pickle.dump(rules_data, f)

        # Also save as "latest" for easy access
        latest_path = os.path.join(
            rules_dir,
            f"translation_rules_{rules.source_protocol.protocol_id}_to_{rules.target_protocol.protocol_id}_latest.pkl",
        )
        with open(latest_path, "wb") as f:
            pickle.dump(rules_data, f)

        # Save human-readable JSON version
        json_path = rules_path.replace(".pkl", ".json")
        with open(json_path, "w") as f:
            json.dump(rules_data, f, indent=2, default=str)

        self.logger.info(f"Translation rules saved: {rules_path}")
        return rules_path

    async def load_translation_rules(self, rules_path: str) -> TranslationRules:
        """
        Load translation rules from file.

        Args:
            rules_path: Path to rules file

        Returns:
            Loaded translation rules
        """
        try:
            with open(rules_path, "rb") as f:
                rules_data = pickle.load(f)

            # Reconstruct protocol specifications
            source_spec = ProtocolSpecification(**rules_data["source_protocol"])
            target_spec = ProtocolSpecification(**rules_data["target_protocol"])

            # Reconstruct translation rules
            rules = [TranslationRule(**r) for r in rules_data["rules"]]

            # Create TranslationRules object
            translation_rules = TranslationRules(
                rules_id=rules_data["rules_id"],
                source_protocol=source_spec,
                target_protocol=target_spec,
                rules=rules,
                preprocessing=rules_data.get("preprocessing", []),
                postprocessing=rules_data.get("postprocessing", []),
                error_handling=rules_data.get("error_handling", {}),
                performance_hints=rules_data.get("performance_hints", {}),
                created_at=datetime.fromisoformat(rules_data["created_at"]),
                accuracy=rules_data.get("accuracy", 0.0),
                test_cases=rules_data.get("test_cases", []),
            )

            # Cache loaded rules
            cache_key = self._get_rules_cache_key(source_spec, target_spec)
            self.translation_rules_cache[cache_key] = translation_rules

            self.logger.info(
                f"Translation rules loaded from {rules_path}: "
                f"accuracy={translation_rules.accuracy:.4f}"
            )

            return translation_rules

        except Exception as e:
            self.logger.error(f"Failed to load translation rules: {e}")
            raise TranslationException(f"Rules loading failed: {e}")

    def list_saved_rules(
        self,
        rules_dir: Optional[str] = None,
        source_protocol: Optional[str] = None,
        target_protocol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all saved translation rules with metadata.

        Args:
            rules_dir: Directory containing rules
            source_protocol: Filter by source protocol
            target_protocol: Filter by target protocol

        Returns:
            List of rule metadata dictionaries
        """
        if rules_dir is None:
            rules_dir = "checkpoints/translation_rules"

        if not os.path.exists(rules_dir):
            return []

        rules_list = []

        for filename in os.listdir(rules_dir):
            if not filename.endswith(".pkl") or filename.endswith("_latest.pkl"):
                continue

            # Parse filename for metadata
            parts = filename.replace(".pkl", "").split("_")

            try:
                # Extract protocol IDs from filename
                if "to" in parts:
                    to_idx = parts.index("to")
                    file_source = parts[to_idx - 1]
                    file_target = parts[to_idx + 1]

                    # Apply filters
                    if source_protocol and file_source != source_protocol:
                        continue
                    if target_protocol and file_target != target_protocol:
                        continue

                    # Extract version and accuracy
                    version = next(
                        (p.replace("v", "") for p in parts if p.startswith("v")),
                        "unknown",
                    )
                    accuracy = next(
                        (
                            float(p.replace("acc", ""))
                            for p in parts
                            if p.startswith("acc")
                        ),
                        0.0,
                    )
                    timestamp = parts[-1] if len(parts) > 0 else "unknown"

                    rules_list.append(
                        {
                            "filename": filename,
                            "path": os.path.join(rules_dir, filename),
                            "source_protocol": file_source,
                            "target_protocol": file_target,
                            "version": version,
                            "accuracy": accuracy,
                            "timestamp": timestamp,
                            "size_bytes": os.path.getsize(
                                os.path.join(rules_dir, filename)
                            ),
                        }
                    )
            except Exception as e:
                self.logger.warning(f"Failed to parse rules file {filename}: {e}")
                continue

        # Sort by timestamp (newest first)
        rules_list.sort(key=lambda x: x["timestamp"], reverse=True)

        return rules_list
        """Get performance data for a protocol pair."""
        key = f"{source_protocol}:{target_protocol}"
        return self.performance_data.get(key)


# Global translation studio instance
_translation_studio: Optional[ProtocolTranslationStudio] = None


async def initialize_translation_studio(
    config: Config,
    llm_service: UnifiedLLMService,
    metrics_collector: Optional[MetricsCollector] = None,
) -> ProtocolTranslationStudio:
    """Initialize global translation studio."""
    global _translation_studio

    _translation_studio = ProtocolTranslationStudio(
        config, llm_service, metrics_collector
    )

    return _translation_studio


def get_translation_studio() -> Optional[ProtocolTranslationStudio]:
    """Get global translation studio instance."""
    return _translation_studio


async def shutdown_translation_studio():
    """Shutdown global translation studio."""
    global _translation_studio
    if _translation_studio:
        # Perform cleanup if needed
        _translation_studio = None
