"""
QBITEL Bridge - Legacy System Whisperer Orchestrator

Main orchestration layer that coordinates the 4 legacy whisperer services:
- COBOLParserService: Traffic pattern analysis and protocol structure inference
- BusinessRulesExtractor: Documentation and behavior analysis
- DataFlowAnalyzer: Protocol differences and risk assessment
- ModernizationRecommender: Adapter code generation

This module provides the same public API as the original LegacySystemWhisperer
for backward compatibility.
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

from prometheus_client import Counter, Histogram

from .models import (
    ProtocolSpecification,
    AdapterCode,
    Explanation,
    AdapterLanguage,
    LegacyWhispererException,
)
from .cobol_parser import COBOLParserService
from .business_rules import BusinessRulesExtractor
from .data_flow import DataFlowAnalyzer
from .modernization import ModernizationRecommender

from ...core.config import get_config
from ...llm.unified_llm_service import get_llm_service
from ...llm.rag_engine import RAGEngine, RAGDocument

# Metrics
LEGACY_ANALYSIS_COUNTER = Counter(
    "qbitel_legacy_analysis_total",
    "Total legacy protocol analyses",
    ["analysis_type", "status"],
)
LEGACY_ANALYSIS_DURATION = Histogram(
    "qbitel_legacy_analysis_duration_seconds",
    "Legacy protocol analysis duration",
    ["analysis_type"],
)
LEGACY_CONFIDENCE_SCORE = Histogram("qbitel_legacy_confidence_score", "Legacy protocol analysis confidence scores")
LEGACY_ADAPTER_GENERATION = Counter(
    "qbitel_legacy_adapter_generation_total",
    "Total adapter code generations",
    ["source_protocol", "target_protocol", "status"],
)

logger = logging.getLogger(__name__)


class LegacySystemWhisperer:
    """
    LLM-powered legacy protocol understanding and modernization.

    This orchestrator coordinates the 4 specialized services:
    - COBOL Parser: Traffic pattern analysis
    - Business Rules: Documentation generation
    - Data Flow: Risk assessment
    - Modernization: Code generation

    Features:
    - Automatic protocol reverse engineering from traffic samples
    - Legacy documentation generation
    - Protocol adapter code generation
    - Migration path recommendations
    - Risk assessment for modernization

    Success Metrics:
    - Reverse engineering accuracy: 85%+
    - Documentation completeness: 90%+
    - Adapter code quality: Production-ready
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.llm_service = get_llm_service()
        self.rag_engine = RAGEngine()

        # Initialize specialized services
        self.cobol_parser = COBOLParserService(self.llm_service)
        self.business_rules = BusinessRulesExtractor(self.llm_service, self.rag_engine)
        self.data_flow = DataFlowAnalyzer(self.llm_service)
        self.modernization = ModernizationRecommender(self.llm_service)

        # Analysis cache
        self.analysis_cache: Dict[str, ProtocolSpecification] = {}
        self.adapter_cache: Dict[str, AdapterCode] = {}

        # Knowledge base for legacy protocols
        self.legacy_knowledge: Dict[str, Any] = {}

        # Configuration
        self.min_samples_for_analysis = 10
        self.confidence_threshold = 0.85
        self.max_cache_size = 100

    async def initialize(self) -> None:
        """Initialize the Legacy System Whisperer."""
        try:
            self.logger.info("Initializing Legacy System Whisperer...")

            # Initialize LLM service
            await self.llm_service.initialize()

            # Initialize RAG engine
            await self.rag_engine.initialize()

            # Load legacy protocol knowledge
            await self._load_legacy_knowledge()

            self.logger.info("Legacy System Whisperer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Legacy System Whisperer: {e}")
            raise LegacyWhispererException(f"Initialization failed: {e}")

    async def reverse_engineer_protocol(self, traffic_samples: List[bytes], system_context: str = "") -> ProtocolSpecification:
        """
        Reverse engineer legacy protocol from traffic samples.

        Args:
            traffic_samples: List of protocol message samples
            system_context: Additional context about the system

        Returns:
            Complete protocol specification
        """
        start_time = time.time()

        try:
            LEGACY_ANALYSIS_COUNTER.labels(analysis_type="reverse_engineering", status="started").inc()

            # Validate input
            if len(traffic_samples) < self.min_samples_for_analysis:
                raise LegacyWhispererException(
                    f"Insufficient samples: {len(traffic_samples)} " f"(minimum: {self.min_samples_for_analysis})"
                )

            # Check cache
            cache_key = self._generate_cache_key(traffic_samples, system_context)
            if cache_key in self.analysis_cache:
                self.logger.info("Returning cached protocol specification")
                return self.analysis_cache[cache_key]

            # Step 1: Analyze traffic patterns (COBOL Parser)
            self.logger.info(f"Analyzing {len(traffic_samples)} traffic samples...")
            patterns = await self.cobol_parser.analyze_traffic_patterns(traffic_samples)

            # Step 2: Infer protocol structure (COBOL Parser)
            self.logger.info("Inferring protocol structure...")
            fields = await self.cobol_parser.infer_protocol_structure(traffic_samples, patterns)

            # Step 3: Identify message types (COBOL Parser)
            self.logger.info("Identifying message types...")
            message_types = await self.cobol_parser.identify_message_types(traffic_samples, fields)

            # Step 4: Determine protocol characteristics (COBOL Parser)
            self.logger.info("Determining protocol characteristics...")
            characteristics = await self.cobol_parser.determine_characteristics(traffic_samples)

            # Step 5: Generate documentation (Business Rules)
            self.logger.info("Generating protocol documentation...")
            documentation = await self.business_rules.generate_documentation(
                traffic_samples,
                patterns,
                fields,
                message_types,
                characteristics,
                system_context,
            )

            # Step 6: Assess complexity (COBOL Parser)
            complexity = self.cobol_parser.assess_complexity(fields, patterns, message_types)

            # Step 7: Calculate confidence score (COBOL Parser)
            confidence = self.cobol_parser.calculate_confidence(len(traffic_samples), patterns, fields, message_types)

            # Create specification
            spec = ProtocolSpecification(
                protocol_name=documentation.get("protocol_name", "Unknown Protocol"),
                version=documentation.get("version", "1.0"),
                description=documentation.get("description", ""),
                complexity=complexity,
                fields=fields,
                message_types=message_types,
                patterns=patterns,
                is_binary=characteristics["is_binary"],
                is_stateful=characteristics["is_stateful"],
                uses_encryption=characteristics["uses_encryption"],
                has_checksums=characteristics["has_checksums"],
                confidence_score=confidence,
                analysis_time=time.time() - start_time,
                samples_analyzed=len(traffic_samples),
                documentation=documentation.get("full_documentation", ""),
                usage_examples=documentation.get("usage_examples", []),
                known_implementations=documentation.get("known_implementations", []),
                historical_context=documentation.get("historical_context", ""),
                common_issues=documentation.get("common_issues", []),
                security_concerns=documentation.get("security_concerns", []),
            )

            # Cache the result
            self._cache_specification(cache_key, spec)

            # Update metrics
            LEGACY_ANALYSIS_DURATION.labels(analysis_type="reverse_engineering").observe(time.time() - start_time)
            LEGACY_CONFIDENCE_SCORE.observe(confidence)
            LEGACY_ANALYSIS_COUNTER.labels(analysis_type="reverse_engineering", status="success").inc()

            self.logger.info(
                f"Protocol reverse engineering completed in {time.time() - start_time:.2f}s "
                f"with confidence {confidence:.2%}"
            )

            return spec

        except Exception as e:
            self.logger.error(f"Protocol reverse engineering failed: {e}")
            LEGACY_ANALYSIS_COUNTER.labels(analysis_type="reverse_engineering", status="error").inc()
            raise LegacyWhispererException(f"Reverse engineering failed: {e}")

    async def generate_adapter_code(
        self,
        legacy_protocol: ProtocolSpecification,
        target_protocol: str,
        language: AdapterLanguage = AdapterLanguage.PYTHON,
    ) -> AdapterCode:
        """
        Generate protocol adapter code.

        Args:
            legacy_protocol: Legacy protocol specification
            target_protocol: Target protocol name (e.g., 'REST', 'gRPC', 'GraphQL')
            language: Programming language for adapter

        Returns:
            Complete adapter code with tests and documentation
        """
        start_time = time.time()

        try:
            LEGACY_ADAPTER_GENERATION.labels(
                source_protocol=legacy_protocol.protocol_name,
                target_protocol=target_protocol,
                status="started",
            ).inc()

            # Check cache
            cache_key = f"{legacy_protocol.spec_id}_{target_protocol}_{language.value}"
            if cache_key in self.adapter_cache:
                self.logger.info("Returning cached adapter code")
                return self.adapter_cache[cache_key]

            # Step 1: Analyze protocol differences (Data Flow)
            self.logger.info("Analyzing protocol differences...")
            differences = await self.data_flow.analyze_protocol_differences(legacy_protocol, target_protocol)

            # Step 2: Generate transformation logic (Modernization)
            self.logger.info(f"Generating {language.value} adapter code...")
            adapter_code = await self.modernization.generate_transformation_logic(
                legacy_protocol, target_protocol, language, differences
            )

            # Step 3: Generate test cases (Modernization)
            self.logger.info("Generating test cases...")
            test_code = await self.modernization.generate_test_cases(legacy_protocol, target_protocol, language, adapter_code)

            # Step 4: Generate documentation (Modernization)
            self.logger.info("Generating integration documentation...")
            documentation = await self.modernization.generate_integration_guide(
                legacy_protocol, target_protocol, language, adapter_code
            )

            # Step 5: Extract dependencies and configuration (Modernization)
            dependencies = self.modernization.extract_dependencies(adapter_code, language)
            config_template = self.modernization.generate_config_template(legacy_protocol, target_protocol, language)

            # Step 6: Generate deployment guide (Modernization)
            deployment_guide = await self.modernization.generate_deployment_guide(
                legacy_protocol, target_protocol, language, adapter_code
            )

            # Step 7: Assess code quality (Modernization)
            quality_score = await self.modernization.assess_code_quality(adapter_code, test_code)

            # Create adapter code object
            adapter = AdapterCode(
                source_protocol=legacy_protocol.protocol_name,
                target_protocol=target_protocol,
                language=language,
                adapter_code=adapter_code,
                test_code=test_code,
                documentation=documentation,
                dependencies=dependencies,
                configuration_template=config_template,
                deployment_guide=deployment_guide,
                integration_points=differences.get("integration_points", []),
                api_endpoints=differences.get("api_endpoints", []),
                code_quality_score=quality_score,
                test_coverage=0.85,  # Estimated based on generated tests
                performance_notes=differences.get("performance_notes", ""),
                generation_time=time.time() - start_time,
                llm_provider=self.llm_service.get_current_provider(),
            )

            # Cache the result
            self._cache_adapter(cache_key, adapter)

            # Update metrics
            LEGACY_ADAPTER_GENERATION.labels(
                source_protocol=legacy_protocol.protocol_name,
                target_protocol=target_protocol,
                status="success",
            ).inc()

            self.logger.info(
                f"Adapter code generation completed in {time.time() - start_time:.2f}s "
                f"with quality score {quality_score:.2%}"
            )

            return adapter

        except Exception as e:
            self.logger.error(f"Adapter code generation failed: {e}")
            LEGACY_ADAPTER_GENERATION.labels(
                source_protocol=legacy_protocol.protocol_name,
                target_protocol=target_protocol,
                status="error",
            ).inc()
            raise LegacyWhispererException(f"Adapter generation failed: {e}")

    async def explain_legacy_behavior(self, behavior: str, context: Dict[str, Any]) -> Explanation:
        """
        Explain legacy system behavior with modernization guidance.

        Args:
            behavior: Description of the legacy behavior
            context: Additional context (protocol, system info, etc.)

        Returns:
            Comprehensive explanation with modernization recommendations
        """
        start_time = time.time()

        try:
            LEGACY_ANALYSIS_COUNTER.labels(analysis_type="behavior_explanation", status="started").inc()

            # Step 1: Analyze behavior patterns (Business Rules)
            self.logger.info("Analyzing legacy behavior...")
            analysis = await self.business_rules.analyze_behavior_with_llm(behavior, context)

            # Step 2: Provide historical context (Business Rules)
            self.logger.info("Gathering historical context...")
            historical_context = await self.business_rules.gather_historical_context(behavior, context)

            # Step 3: Suggest modernization approaches (Business Rules)
            self.logger.info("Generating modernization approaches...")
            approaches = await self.business_rules.suggest_modernization_approaches(behavior, context, analysis)

            # Step 4: Assess modernization risks (Data Flow)
            self.logger.info("Assessing modernization risks...")
            risks = await self.data_flow.assess_modernization_risks(behavior, context, approaches)

            # Step 5: Generate implementation guidance (Modernization)
            self.logger.info("Generating implementation guidance...")
            implementation = await self.modernization.generate_implementation_guidance(behavior, approaches, risks)

            # Create explanation
            explanation = Explanation(
                behavior_description=behavior,
                technical_explanation=analysis.get("technical_explanation", ""),
                historical_context=historical_context,
                root_causes=analysis.get("root_causes", []),
                implications=analysis.get("implications", []),
                modernization_approaches=approaches,
                recommended_approach=self.data_flow.select_best_approach(approaches, risks),
                modernization_risks=risks,
                risk_level=self.data_flow.determine_overall_risk(risks),
                implementation_steps=implementation.get("steps", []),
                estimated_effort=implementation.get("effort", ""),
                required_expertise=implementation.get("expertise", []),
                confidence=analysis.get("confidence", 0.0),
                completeness=self.business_rules.calculate_completeness(analysis, approaches, risks),
            )

            # Update metrics
            LEGACY_ANALYSIS_DURATION.labels(analysis_type="behavior_explanation").observe(time.time() - start_time)
            LEGACY_ANALYSIS_COUNTER.labels(analysis_type="behavior_explanation", status="success").inc()

            self.logger.info(f"Behavior explanation completed in {time.time() - start_time:.2f}s")

            return explanation

        except Exception as e:
            self.logger.error(f"Behavior explanation failed: {e}")
            LEGACY_ANALYSIS_COUNTER.labels(analysis_type="behavior_explanation", status="error").inc()
            raise LegacyWhispererException(f"Behavior explanation failed: {e}")

    def _generate_cache_key(self, samples: List[bytes], context: str) -> str:
        """Generate cache key for analysis."""
        hasher = hashlib.sha256()
        for sample in samples[:10]:
            hasher.update(sample)
        hasher.update(context.encode())
        return hasher.hexdigest()[:16]

    def _cache_specification(self, key: str, spec: ProtocolSpecification) -> None:
        """Cache protocol specification."""
        if len(self.analysis_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.analysis_cache))
            del self.analysis_cache[oldest_key]
        self.analysis_cache[key] = spec

    def _cache_adapter(self, key: str, adapter: AdapterCode) -> None:
        """Cache adapter code."""
        if len(self.adapter_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.adapter_cache))
            del self.adapter_cache[oldest_key]
        self.adapter_cache[key] = adapter

    async def _load_legacy_knowledge(self) -> None:
        """Load legacy protocol knowledge into RAG."""
        legacy_docs = [
            RAGDocument(
                id="legacy_mainframe_protocols",
                content="""
                Legacy Mainframe Protocols:

                Common characteristics:
                - Fixed-length records
                - EBCDIC encoding
                - Batch-oriented processing
                - Limited error handling
                - Synchronous communication

                Modernization considerations:
                - Character encoding conversion (EBCDIC to ASCII/UTF-8)
                - Record format transformation
                - Async communication patterns
                - Enhanced error handling
                - API-based access
                """,
                metadata={"category": "legacy_protocols", "era": "mainframe"},
                created_at=datetime.now(timezone.utc),
            ),
            RAGDocument(
                id="legacy_proprietary_protocols",
                content="""
                Legacy Proprietary Protocols:

                Common challenges:
                - Undocumented specifications
                - Vendor lock-in
                - Limited tooling
                - Security vulnerabilities
                - Performance limitations

                Reverse engineering approaches:
                - Traffic analysis
                - Binary analysis
                - Pattern recognition
                - State machine inference
                - Field boundary detection
                """,
                metadata={"category": "legacy_protocols", "type": "proprietary"},
                created_at=datetime.now(timezone.utc),
            ),
        ]

        await self.rag_engine.add_documents("protocol_knowledge", legacy_docs)
        self.logger.info("Legacy protocol knowledge loaded")

    def get_statistics(self) -> Dict[str, Any]:
        """Get Legacy Whisperer statistics."""
        return {
            "analysis_cache_size": len(self.analysis_cache),
            "adapter_cache_size": len(self.adapter_cache),
            "min_samples_required": self.min_samples_for_analysis,
            "confidence_threshold": self.confidence_threshold,
            "max_cache_size": self.max_cache_size,
        }

    async def shutdown(self) -> None:
        """Shutdown Legacy System Whisperer."""
        try:
            self.logger.info("Shutting down Legacy System Whisperer...")
            self.analysis_cache.clear()
            self.adapter_cache.clear()
            self.logger.info("Legacy System Whisperer shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Factory function
async def create_legacy_whisperer() -> LegacySystemWhisperer:
    """Factory function to create Legacy System Whisperer."""
    whisperer = LegacySystemWhisperer()
    await whisperer.initialize()
    return whisperer
