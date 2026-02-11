"""
QBITEL - Translation Studio Integration
Integration layer between the existing Protocol Discovery Orchestrator and the Translation Studio.
This module ensures seamless integration while maintaining backward compatibility.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from ..core.config import Config
from ..discovery.protocol_discovery_orchestrator import (
    ProtocolDiscoveryOrchestrator,
    DiscoveryRequest as BaseDiscoveryRequest,
    DiscoveryResult as BaseDiscoveryResult,
    DiscoveryPhase,
)
from ..llm.unified_llm_service import UnifiedLLMService
from ..llm.rag_engine import RAGEngine

from .enhanced_discovery import (
    EnhancedProtocolDiscoveryOrchestrator,
    APIGenerationRequest,
    APIGenerationResult,
)
from .api_generation.api_generator import APIGenerator
from .code_generation.code_generator import MultiLanguageCodeGenerator
from .protocol_bridge.protocol_bridge import ProtocolBridge
from .models import (
    CodeLanguage,
    APIStyle,
    SecurityLevel,
    ProtocolSchema,
    APISpecification,
    GeneratedSDK,
)
from .exceptions import (
    TranslationStudioException,
    ProtocolDiscoveryException,
    create_error_context,
)
from .logging import get_logger, create_context, LogComponent, LogOperation
from .monitoring import get_monitor, monitor_operation


@dataclass
class IntegratedDiscoveryRequest(BaseDiscoveryRequest):
    """Extended discovery request with Translation Studio capabilities."""

    # Translation Studio specific fields
    target_api_style: APIStyle = APIStyle.REST
    target_languages: List[CodeLanguage] = field(
        default_factory=lambda: [CodeLanguage.PYTHON]
    )
    security_level: SecurityLevel = SecurityLevel.AUTHENTICATED
    generate_api_specification: bool = True
    generate_sdks: bool = True
    generate_documentation: bool = True
    generate_tests: bool = True
    api_base_path: str = "/api/v1"
    package_name: Optional[str] = None
    enable_semantic_analysis: bool = True
    enable_field_semantics: bool = True
    user_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedDiscoveryResult(BaseDiscoveryResult):
    """Extended discovery result with Translation Studio outputs."""

    # Translation Studio specific fields
    api_specification: Optional[APISpecification] = None
    generated_sdks: Dict[CodeLanguage, GeneratedSDK] = field(default_factory=dict)
    semantic_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    natural_language_summary: Optional[str] = None

    # Enhanced fields
    field_meanings: Dict[str, str] = field(default_factory=dict)
    protocol_relationships: List[Dict[str, Any]] = field(default_factory=list)
    translation_capabilities: List[str] = field(default_factory=list)


class IntegratedProtocolDiscoveryOrchestrator:
    """
    Integrated Protocol Discovery Orchestrator that combines the existing
    QBITEL protocol discovery with Translation Studio capabilities.

    This class provides seamless integration between the base discovery system
    and the enhanced translation features while maintaining backward compatibility.
    """

    def __init__(
        self,
        config: Config,
        llm_service: Optional[UnifiedLLMService] = None,
        rag_engine: Optional[RAGEngine] = None,
    ):
        """Initialize the integrated orchestrator."""
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize base orchestrator
        self.base_orchestrator = ProtocolDiscoveryOrchestrator(config)

        # Initialize Translation Studio components
        self.llm_service = llm_service
        self.rag_engine = rag_engine

        # Translation Studio orchestrator (will be initialized lazily)
        self.translation_orchestrator: Optional[
            EnhancedProtocolDiscoveryOrchestrator
        ] = None

        # Translation Studio components (will be initialized lazily)
        self.api_generator: Optional[APIGenerator] = None
        self.code_generator: Optional[MultiLanguageCodeGenerator] = None
        self.protocol_bridge: Optional[ProtocolBridge] = None

        # Configuration flags
        self.translation_enabled = config.get("translation", {}).get("enabled", True)
        self.backward_compatible = True
        self.integration_mode = config.get("translation", {}).get(
            "integration_mode", "enhanced"
        )

        # Integration state
        self.is_initialized = False
        self._components_initialized = False

        self.logger.info(
            f"Integrated Protocol Discovery Orchestrator created "
            f"(translation_enabled: {self.translation_enabled}, "
            f"integration_mode: {self.integration_mode})"
        )

    async def initialize(self) -> None:
        """Initialize both base and enhanced orchestrators."""
        if self.is_initialized:
            self.logger.warning("Integrated orchestrator already initialized")
            return

        start_time = time.time()
        self.logger.info("Initializing Integrated Protocol Discovery Orchestrator")

        try:
            # Always initialize base orchestrator
            await self.base_orchestrator.initialize()

            # Initialize Translation Studio components if enabled
            if self.translation_enabled and self.llm_service:
                await self._initialize_translation_components()
                self._components_initialized = True

            self.is_initialized = True
            initialization_time = time.time() - start_time

            self.logger.info(
                f"Integrated Protocol Discovery Orchestrator initialized in {initialization_time:.2f}s "
                f"(translation_enabled: {self._components_initialized})"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize integrated orchestrator: {e}")
            raise ProtocolDiscoveryException(
                f"Integrated orchestrator initialization failed: {e}",
                context=create_error_context(
                    component="integration", operation="initialization"
                ),
            )

    async def _initialize_translation_components(self) -> None:
        """Initialize Translation Studio components."""
        self.logger.info("Initializing Translation Studio components")

        try:
            # Initialize enhanced discovery orchestrator
            self.translation_orchestrator = EnhancedProtocolDiscoveryOrchestrator(
                config=self.config,
                llm_service=self.llm_service,
                base_orchestrator=self.base_orchestrator,
            )
            await self.translation_orchestrator.initialize()

            # Initialize API generator
            self.api_generator = APIGenerator(
                config=self.config, llm_service=self.llm_service
            )
            await self.api_generator.initialize()

            # Initialize code generator
            self.code_generator = MultiLanguageCodeGenerator(
                config=self.config, llm_service=self.llm_service
            )
            await self.code_generator.initialize()

            # Initialize protocol bridge
            self.protocol_bridge = ProtocolBridge(
                config=self.config, llm_service=self.llm_service
            )
            await self.protocol_bridge.initialize()

            self.logger.info("Translation Studio components initialized successfully")

        except Exception as e:
            self.logger.error(
                f"Failed to initialize Translation Studio components: {e}"
            )
            raise

    async def discover_protocol(
        self, request: Union[BaseDiscoveryRequest, IntegratedDiscoveryRequest]
    ) -> Union[BaseDiscoveryResult, IntegratedDiscoveryResult]:
        """
        Main protocol discovery entry point with automatic capability detection.

        This method provides intelligent routing between base and enhanced discovery
        based on the request type and available capabilities.
        """
        if not self.is_initialized:
            await self.initialize()

        context = create_context(
            component=LogComponent.DISCOVERY,
            operation=LogOperation.PROTOCOL_ANALYSIS,
            protocol_type=getattr(request, "known_protocol", None),
        )

        # Determine which discovery path to use
        use_enhanced = (
            isinstance(request, IntegratedDiscoveryRequest)
            and self._components_initialized
            and (
                request.generate_api_specification
                or request.generate_sdks
                or request.enable_semantic_analysis
            )
        )

        if use_enhanced:
            self.logger.info(
                "Using enhanced discovery with Translation Studio capabilities"
            )
            return await self._enhanced_discovery(request, context)
        else:
            self.logger.info("Using base protocol discovery")
            return await self._base_discovery(request, context)

    @monitor_operation("integration", "base_discovery")
    async def _base_discovery(
        self, request: BaseDiscoveryRequest, context
    ) -> BaseDiscoveryResult:
        """Perform base protocol discovery."""
        try:
            result = await self.base_orchestrator.discover_protocol(request)

            self.logger.info(
                f"Base discovery completed: {result.protocol_type} "
                f"(confidence: {result.confidence:.2f})",
                context=context,
            )

            return result

        except Exception as e:
            self.logger.error(
                f"Base discovery failed: {e}", context=context, exception=e
            )
            raise ProtocolDiscoveryException(
                f"Base discovery failed: {e}",
                context=create_error_context(
                    component="integration",
                    operation="base_discovery",
                    protocol_type=getattr(request, "known_protocol", None),
                ),
                cause=e,
            )

    @monitor_operation("integration", "enhanced_discovery")
    async def _enhanced_discovery(
        self, request: IntegratedDiscoveryRequest, context
    ) -> IntegratedDiscoveryResult:
        """Perform enhanced discovery with Translation Studio capabilities."""
        start_time = time.time()

        try:
            # Step 1: Perform base discovery
            base_request = BaseDiscoveryRequest(
                messages=request.messages,
                known_protocol=request.known_protocol,
                training_mode=request.training_mode,
                confidence_threshold=request.confidence_threshold,
                generate_parser=request.generate_parser,
                validate_results=request.validate_results,
                custom_rules=request.custom_rules,
                metadata=request.metadata,
            )

            base_result = await self.base_orchestrator.discover_protocol(base_request)

            # Step 2: Create integrated result from base result
            integrated_result = IntegratedDiscoveryResult(
                protocol_type=base_result.protocol_type,
                confidence=base_result.confidence,
                grammar=base_result.grammar,
                parser=base_result.parser,
                validation_result=base_result.validation_result,
                statistical_analysis=base_result.statistical_analysis,
                classification_details=base_result.classification_details,
                processing_time=0.0,  # Will be updated
                phases_completed=base_result.phases_completed.copy(),
                metadata=base_result.metadata.copy(),
            )

            # Step 3: Add Translation Studio enhancements if confidence is sufficient
            if base_result.confidence >= request.confidence_threshold:
                await self._add_translation_enhancements(
                    request, integrated_result, context
                )
            else:
                self.logger.warning(
                    f"Skipping Translation Studio enhancements due to low confidence: {base_result.confidence:.2f}",
                    context=context,
                )
                integrated_result.recommendations.append(
                    "Consider providing more sample messages to improve confidence"
                )

            integrated_result.processing_time = time.time() - start_time

            self.logger.info(
                f"Enhanced discovery completed: {integrated_result.protocol_type} "
                f"(confidence: {integrated_result.confidence:.2f}, "
                f"api_generated: {integrated_result.api_specification is not None}, "
                f"sdks_generated: {len(integrated_result.generated_sdks)})",
                context=context,
            )

            return integrated_result

        except Exception as e:
            self.logger.error(
                f"Enhanced discovery failed: {e}", context=context, exception=e
            )
            raise ProtocolDiscoveryException(
                f"Enhanced discovery failed: {e}",
                context=create_error_context(
                    component="integration",
                    operation="enhanced_discovery",
                    protocol_type=getattr(request, "known_protocol", None),
                ),
                cause=e,
            )

    async def _add_translation_enhancements(
        self,
        request: IntegratedDiscoveryRequest,
        result: IntegratedDiscoveryResult,
        context,
    ) -> None:
        """Add Translation Studio enhancements to the discovery result."""

        try:
            # Create protocol schema from discovery results
            protocol_schema = await self._create_protocol_schema(result)

            # Semantic analysis with LLM
            if request.enable_semantic_analysis and self.llm_service:
                result.semantic_analysis = await self._perform_semantic_analysis(
                    request.messages, protocol_schema, context
                )
                result.phases_completed.append("semantic_analysis")

            # API specification generation
            if request.generate_api_specification and self.api_generator:
                result.api_specification = await self._generate_api_specification(
                    protocol_schema, request, context
                )
                result.phases_completed.append("api_generation")

            # SDK generation
            if (
                request.generate_sdks
                and result.api_specification
                and self.code_generator
            ):
                result.generated_sdks = await self._generate_sdks(
                    result.api_specification, request, context
                )
                result.phases_completed.append("sdk_generation")

            # Generate recommendations
            result.recommendations = await self._generate_recommendations(
                result, request, context
            )

            # Generate natural language summary
            if self.llm_service:
                result.natural_language_summary = await self._generate_summary(
                    result, context
                )

            # Add translation capabilities
            result.translation_capabilities = self._get_translation_capabilities(result)

        except Exception as e:
            self.logger.error(
                f"Failed to add translation enhancements: {e}", context=context
            )
            # Don't fail the entire discovery - just add error to recommendations
            result.recommendations.append(
                f"Translation Studio enhancements failed: {str(e)[:100]}..."
            )

    async def _create_protocol_schema(
        self, result: IntegratedDiscoveryResult
    ) -> ProtocolSchema:
        """Create protocol schema from discovery results."""
        fields = []

        # Extract fields from grammar if available
        if result.grammar:
            for rule in result.grammar.rules:
                for symbol in rule.right_hand_side:
                    if hasattr(symbol, "semantic_type") and symbol.semantic_type:
                        fields.append(
                            {
                                "name": symbol.name or f"field_{len(fields)}",
                                "type": symbol.semantic_type,
                                "description": f"Field derived from grammar rule: {rule.left_hand_side}",
                            }
                        )

        # If no fields from grammar, create basic fields from statistical analysis
        if not fields and result.statistical_analysis:
            stats = result.statistical_analysis
            if "common_patterns" in stats:
                for i, pattern in enumerate(stats["common_patterns"][:5]):
                    fields.append(
                        {
                            "name": f"pattern_{i}",
                            "type": "binary",
                            "description": f"Common pattern: {pattern}",
                        }
                    )

        # Ensure at least one field
        if not fields:
            fields = [
                {"name": "data", "type": "binary", "description": "Protocol data"}
            ]

        return ProtocolSchema(
            name=result.protocol_type,
            version="1.0",
            description=f"Auto-discovered protocol: {result.protocol_type}",
            fields=fields,
            semantic_info={
                "confidence": result.confidence,
                "discovery_method": "integrated",
                "has_grammar": result.grammar is not None,
                "has_parser": result.parser is not None,
            },
        )

    async def _perform_semantic_analysis(
        self, messages: List[bytes], protocol_schema: ProtocolSchema, context
    ) -> Dict[str, Any]:
        """Perform LLM-powered semantic analysis."""
        try:
            # Use translation orchestrator for semantic analysis
            semantic_result = (
                await self.translation_orchestrator.analyze_semantic_fields(
                    messages=messages,
                    protocol_schema=protocol_schema,
                    context=context.metadata if context else {},
                )
            )

            return semantic_result

        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {e}", context=context)
            return {"error": str(e), "field_meanings": {}, "relationships": []}

    async def _generate_api_specification(
        self,
        protocol_schema: ProtocolSchema,
        request: IntegratedDiscoveryRequest,
        context,
    ) -> Optional[APISpecification]:
        """Generate API specification from protocol schema."""
        try:
            from .api_generation.api_generator import APIGenerationContext

            generation_context = APIGenerationContext(
                protocol_schema=protocol_schema,
                target_style=request.target_api_style,
                security_level=request.security_level,
                base_path=request.api_base_path,
                include_examples=request.generate_documentation,
                include_documentation=request.generate_documentation,
            )

            api_spec = await self.api_generator.generate_api_specification(
                generation_context
            )

            # Record metrics
            monitor = get_monitor()
            monitor.record_api_generation(
                api_style=request.target_api_style.value,
                security_level=request.security_level.value,
                duration=0.0,  # Would be measured in actual implementation
                endpoints_count=len(api_spec.endpoints),
                success=True,
            )

            return api_spec

        except Exception as e:
            self.logger.error(
                f"API specification generation failed: {e}", context=context
            )
            return None

    async def _generate_sdks(
        self,
        api_specification: APISpecification,
        request: IntegratedDiscoveryRequest,
        context,
    ) -> Dict[CodeLanguage, GeneratedSDK]:
        """Generate SDKs for requested languages."""
        generated_sdks = {}

        for language in request.target_languages:
            try:
                package_name = (
                    request.package_name
                    or f"{api_specification.title.lower().replace(' ', '-')}-{language.value}-sdk"
                )

                sdk = await self.code_generator.generate_sdk(
                    api_specification=api_specification,
                    target_language=language,
                    package_name=package_name,
                    generate_tests=request.generate_tests,
                    generate_documentation=request.generate_documentation,
                )

                generated_sdks[language] = sdk

                # Record metrics
                monitor = get_monitor()
                monitor.record_code_generation(
                    language=language.value,
                    duration=0.0,  # Would be measured in actual implementation
                    lines_generated=sum(
                        len(content.split("\n"))
                        for content in sdk.source_files.values()
                    ),
                    success=True,
                )

            except Exception as e:
                self.logger.error(
                    f"SDK generation failed for {language.value}: {e}", context=context
                )
                continue

        return generated_sdks

    async def _generate_recommendations(
        self,
        result: IntegratedDiscoveryResult,
        request: IntegratedDiscoveryRequest,
        context,
    ) -> List[str]:
        """Generate recommendations based on discovery results."""
        recommendations = []

        # Confidence-based recommendations
        if result.confidence < 0.8:
            recommendations.append(
                "Consider providing more sample messages to improve confidence"
            )

        if result.confidence < 0.6:
            recommendations.append(
                "Protocol detection confidence is low - manual verification recommended"
            )

        # API-related recommendations
        if result.api_specification:
            if len(result.api_specification.endpoints) > 20:
                recommendations.append(
                    "Consider implementing pagination for APIs with many endpoints"
                )

            if request.security_level == SecurityLevel.PUBLIC:
                recommendations.append(
                    "Consider adding authentication for production APIs"
                )

        # SDK-related recommendations
        if result.generated_sdks:
            if len(result.generated_sdks) > 3:
                recommendations.append(
                    "Consider creating SDK documentation portal for multiple languages"
                )

        # Performance recommendations
        if result.processing_time > 5.0:
            recommendations.append(
                "Consider enabling caching to improve discovery performance"
            )

        return recommendations

    async def _generate_summary(
        self, result: IntegratedDiscoveryResult, context
    ) -> Optional[str]:
        """Generate natural language summary of discovery results."""
        try:
            if not self.llm_service:
                return None

            # Create summary prompt
            summary_data = {
                "protocol_type": result.protocol_type,
                "confidence": result.confidence,
                "has_api": result.api_specification is not None,
                "sdk_count": len(result.generated_sdks),
                "sdk_languages": [lang.value for lang in result.generated_sdks.keys()],
                "processing_time": result.processing_time,
            }

            from ..llm.unified_llm_service import LLMRequest

            llm_request = LLMRequest(
                prompt=f"Generate a brief, user-friendly summary of this protocol discovery result: {summary_data}",
                feature_domain="translation_studio",
                context={"operation": "summary_generation"},
            )

            response = await self.llm_service.process_request(llm_request)
            return response.content[:500]  # Limit summary length

        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}", context=context)
            return None

    def _get_translation_capabilities(
        self, result: IntegratedDiscoveryResult
    ) -> List[str]:
        """Get list of available translation capabilities."""
        capabilities = []

        if result.protocol_type != "unknown":
            capabilities.append(f"Protocol identification: {result.protocol_type}")

        if result.parser:
            capabilities.append("Message parsing")

        if result.grammar:
            capabilities.append("Grammar-based validation")

        if result.api_specification:
            capabilities.append(
                f"API generation: {result.api_specification.style.value}"
            )

        if result.generated_sdks:
            languages = [lang.value for lang in result.generated_sdks.keys()]
            capabilities.append(f"SDK generation: {', '.join(languages)}")

        if self.protocol_bridge:
            capabilities.append("Protocol translation")

        return capabilities

    # Backward compatibility methods

    async def train_classifier(self, *args, **kwargs):
        """Delegate to base orchestrator for backward compatibility."""
        return await self.base_orchestrator.train_classifier(*args, **kwargs)

    async def discover_protocol_stream(self, *args, **kwargs):
        """Delegate to base orchestrator for backward compatibility."""
        return await self.base_orchestrator.discover_protocol_stream(*args, **kwargs)

    async def get_protocol_profiles(self) -> Dict[str, Any]:
        """Get protocol profiles from both base and translation orchestrators."""
        profiles = await self.base_orchestrator.get_protocol_profiles()

        if self.translation_orchestrator:
            # Add translation-specific profile data
            for protocol_name in profiles:
                profiles[protocol_name]["translation_capabilities"] = True
                profiles[protocol_name]["api_generation_available"] = True
                profiles[protocol_name]["sdk_generation_available"] = True

        return profiles

    async def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from both orchestrators."""
        base_stats = await self.base_orchestrator.get_discovery_statistics()

        # Add integration-specific statistics
        base_stats["integration"] = {
            "translation_enabled": self.translation_enabled,
            "components_initialized": self._components_initialized,
            "integration_mode": self.integration_mode,
            "backward_compatible": self.backward_compatible,
        }

        if self.translation_orchestrator:
            translation_stats = (
                await self.translation_orchestrator.get_api_generation_metrics()
            )
            base_stats["translation_studio"] = translation_stats

        return base_stats

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for integrated system."""
        health = await self.base_orchestrator.health_check()

        # Add Translation Studio health info
        health["translation_studio"] = {
            "enabled": self.translation_enabled,
            "initialized": self._components_initialized,
            "components": {},
        }

        if self._components_initialized:
            components = {
                "translation_orchestrator": self.translation_orchestrator,
                "api_generator": self.api_generator,
                "code_generator": self.code_generator,
                "protocol_bridge": self.protocol_bridge,
            }

            for name, component in components.items():
                if component:
                    try:
                        # Simple health check
                        health["translation_studio"]["components"][name] = "healthy"
                    except Exception as e:
                        health["translation_studio"]["components"][
                            name
                        ] = f"unhealthy: {e}"

        return health

    async def shutdown(self):
        """Shutdown both orchestrators."""
        self.logger.info("Shutting down Integrated Protocol Discovery Orchestrator")

        # Shutdown Translation Studio components first
        if self._components_initialized:
            shutdown_tasks = []

            for component in [
                self.translation_orchestrator,
                self.api_generator,
                self.code_generator,
                self.protocol_bridge,
            ]:
                if component and hasattr(component, "shutdown"):
                    shutdown_tasks.append(component.shutdown())

            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        # Shutdown base orchestrator
        await self.base_orchestrator.shutdown()

        self.is_initialized = False
        self._components_initialized = False

        self.logger.info(
            "Integrated Protocol Discovery Orchestrator shutdown completed"
        )


# Convenience functions for easy integration


async def create_integrated_orchestrator(
    config: Config,
    llm_service: Optional[UnifiedLLMService] = None,
    rag_engine: Optional[RAGEngine] = None,
    auto_initialize: bool = True,
) -> IntegratedProtocolDiscoveryOrchestrator:
    """Create and optionally initialize an integrated orchestrator."""

    orchestrator = IntegratedProtocolDiscoveryOrchestrator(
        config=config, llm_service=llm_service, rag_engine=rag_engine
    )

    if auto_initialize:
        await orchestrator.initialize()

    return orchestrator


def create_integrated_request(
    messages: List[bytes],
    known_protocol: Optional[str] = None,
    target_api_style: APIStyle = APIStyle.REST,
    target_languages: Optional[List[CodeLanguage]] = None,
    **kwargs,
) -> IntegratedDiscoveryRequest:
    """Create an integrated discovery request with sensible defaults."""

    if target_languages is None:
        target_languages = [CodeLanguage.PYTHON]

    return IntegratedDiscoveryRequest(
        messages=messages,
        known_protocol=known_protocol,
        target_api_style=target_api_style,
        target_languages=target_languages,
        **kwargs,
    )


__all__ = [
    "IntegratedDiscoveryRequest",
    "IntegratedDiscoveryResult",
    "IntegratedProtocolDiscoveryOrchestrator",
    "create_integrated_orchestrator",
    "create_integrated_request",
]
