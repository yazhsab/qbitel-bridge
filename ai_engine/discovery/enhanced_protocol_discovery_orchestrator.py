"""
QBITEL Engine - Enhanced Protocol Discovery Orchestrator with LLM Integration
Extended version with Protocol Intelligence Copilot integration for natural language analysis.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from .protocol_discovery_orchestrator import (
    ProtocolDiscoveryOrchestrator,
    DiscoveryRequest,
    DiscoveryResult,
    DiscoveryPhase,
    ProtocolProfile,
)
from ..core.config import Config
from ..core.exceptions import ProtocolException, ModelException
from ..copilot.protocol_copilot import (
    ProtocolIntelligenceCopilot,
    CopilotQuery,
    CopilotResponse,
)
from ..llm.unified_llm_service import UnifiedLLMService
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Enhanced metrics
LLM_ANALYSIS_COUNTER = Counter(
    "qbitel_llm_protocol_analysis_total",
    "Total LLM protocol analyses",
    ["protocol_type", "analysis_type", "status"],
)

LLM_ANALYSIS_DURATION = Histogram(
    "qbitel_llm_analysis_duration_seconds", "LLM analysis duration", ["analysis_type"]
)


class LLMAnalysisType(Enum):
    """Types of LLM analysis performed."""

    PROTOCOL_IDENTIFICATION = "protocol_identification"
    SECURITY_ASSESSMENT = "security_assessment"
    COMPLIANCE_CHECK = "compliance_check"
    ANOMALY_EXPLANATION = "anomaly_explanation"
    FIELD_SEMANTICS = "field_semantics"
    VULNERABILITY_ANALYSIS = "vulnerability_analysis"


@dataclass
class EnhancedDiscoveryRequest(DiscoveryRequest):
    """Enhanced discovery request with LLM analysis options."""

    enable_llm_analysis: bool = True
    llm_analysis_types: List[LLMAnalysisType] = field(
        default_factory=lambda: [
            LLMAnalysisType.PROTOCOL_IDENTIFICATION,
            LLMAnalysisType.SECURITY_ASSESSMENT,
        ]
    )
    natural_language_explanation: bool = True
    security_focus: bool = False
    compliance_frameworks: Optional[List[str]] = None
    user_context: Optional[Dict[str, Any]] = None
    conversation_session_id: Optional[str] = None


@dataclass
class LLMAnalysisResult:
    """Result of LLM analysis."""

    analysis_type: LLMAnalysisType
    response: str
    confidence: float
    recommendations: List[str] = field(default_factory=list)
    security_implications: Optional[str] = None
    compliance_notes: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0
    llm_provider: Optional[str] = None


@dataclass
class EnhancedDiscoveryResult(DiscoveryResult):
    """Enhanced discovery result with LLM analysis."""

    llm_analyses: Dict[LLMAnalysisType, LLMAnalysisResult] = field(default_factory=dict)
    natural_language_summary: Optional[str] = None
    security_assessment: Optional[str] = None
    compliance_status: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    conversation_context: Optional[Dict[str, Any]] = None


class EnhancedProtocolDiscoveryOrchestrator(ProtocolDiscoveryOrchestrator):
    """
    Enhanced Protocol Discovery Orchestrator with LLM Intelligence.

    Extends the base orchestrator with natural language analysis capabilities,
    providing human-readable explanations, security assessments, and
    intelligent recommendations for protocol discoveries.
    """

    def __init__(
        self,
        config: Config,
        protocol_copilot: Optional[ProtocolIntelligenceCopilot] = None,
    ):
        """Initialize enhanced orchestrator with LLM capabilities."""
        super().__init__(config)
        self.protocol_copilot = protocol_copilot
        self.llm_service: Optional[UnifiedLLMService] = None

        # Enhanced configuration
        self.enable_llm_by_default = True
        self.llm_confidence_threshold = 0.6
        self.max_llm_retries = 2
        self.llm_timeout = 30.0

        # LLM analysis cache
        self.llm_analysis_cache: Dict[str, Dict[LLMAnalysisType, LLMAnalysisResult]] = (
            {}
        )

        self.logger.info(
            "Enhanced Protocol Discovery Orchestrator initialized with LLM capabilities"
        )

    async def initialize(self) -> None:
        """Initialize enhanced orchestrator with LLM services."""
        await super().initialize()

        try:
            # Initialize LLM service if not provided via copilot
            if not self.protocol_copilot:
                from ..copilot.protocol_copilot import create_protocol_copilot

                self.protocol_copilot = await create_protocol_copilot()

            if self.protocol_copilot:
                self.llm_service = self.protocol_copilot.llm_service
                self.logger.info(
                    "LLM services initialized for enhanced protocol discovery"
                )
            else:
                self.logger.warning(
                    "LLM services not available - falling back to traditional discovery"
                )

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM services: {e}")
            # Continue without LLM capabilities

    async def discover_protocol_enhanced(
        self, request: EnhancedDiscoveryRequest
    ) -> EnhancedDiscoveryResult:
        """
        Enhanced protocol discovery with LLM analysis.

        Args:
            request: Enhanced discovery request with LLM options

        Returns:
            Comprehensive discovery result with LLM insights
        """
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Perform traditional discovery first
            traditional_request = DiscoveryRequest(
                messages=request.messages,
                known_protocol=request.known_protocol,
                training_mode=request.training_mode,
                confidence_threshold=request.confidence_threshold,
                generate_parser=request.generate_parser,
                validate_results=request.validate_results,
                custom_rules=request.custom_rules,
                metadata=request.metadata,
            )

            traditional_result = await super().discover_protocol(traditional_request)

            # Convert to enhanced result
            enhanced_result = EnhancedDiscoveryResult(
                protocol_type=traditional_result.protocol_type,
                confidence=traditional_result.confidence,
                grammar=traditional_result.grammar,
                parser=traditional_result.parser,
                validation_result=traditional_result.validation_result,
                statistical_analysis=traditional_result.statistical_analysis,
                classification_details=traditional_result.classification_details,
                processing_time=traditional_result.processing_time,
                phases_completed=traditional_result.phases_completed,
                metadata=traditional_result.metadata,
            )

            # Perform LLM analysis if enabled and available
            if request.enable_llm_analysis and self.protocol_copilot:
                await self._perform_llm_analysis(request, enhanced_result)

            # Generate natural language summary
            if request.natural_language_explanation:
                await self._generate_natural_language_summary(enhanced_result)

            # Update processing time
            enhanced_result.processing_time = time.time() - start_time

            self.logger.info(
                f"Enhanced protocol discovery completed: {enhanced_result.protocol_type} "
                f"with {len(enhanced_result.llm_analyses)} LLM analyses "
                f"(time: {enhanced_result.processing_time:.2f}s)"
            )

            return enhanced_result

        except Exception as e:
            self.logger.error(f"Enhanced protocol discovery failed: {e}")
            # Return basic result with error
            return EnhancedDiscoveryResult(
                protocol_type="unknown",
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)},
            )

    async def _perform_llm_analysis(
        self, request: EnhancedDiscoveryRequest, result: EnhancedDiscoveryResult
    ) -> None:
        """Perform comprehensive LLM analysis on discovery results."""

        for analysis_type in request.llm_analysis_types:
            analysis_start = time.time()

            try:
                # Check cache first
                cache_key = self._generate_llm_cache_key(
                    request.messages, analysis_type
                )
                if cache_key in self.llm_analysis_cache:
                    cached_analyses = self.llm_analysis_cache[cache_key]
                    if analysis_type in cached_analyses:
                        result.llm_analyses[analysis_type] = cached_analyses[
                            analysis_type
                        ]
                        continue

                # Perform LLM analysis
                llm_result = await self._execute_llm_analysis(
                    analysis_type, request, result
                )

                if llm_result:
                    llm_result.processing_time = time.time() - analysis_start
                    result.llm_analyses[analysis_type] = llm_result

                    # Cache result
                    if cache_key not in self.llm_analysis_cache:
                        self.llm_analysis_cache[cache_key] = {}
                    self.llm_analysis_cache[cache_key][analysis_type] = llm_result

                    # Update metrics
                    LLM_ANALYSIS_COUNTER.labels(
                        protocol_type=result.protocol_type,
                        analysis_type=analysis_type.value,
                        status="success",
                    ).inc()

                    LLM_ANALYSIS_DURATION.labels(
                        analysis_type=analysis_type.value
                    ).observe(llm_result.processing_time)

            except Exception as e:
                self.logger.error(f"LLM analysis {analysis_type.value} failed: {e}")
                LLM_ANALYSIS_COUNTER.labels(
                    protocol_type=result.protocol_type,
                    analysis_type=analysis_type.value,
                    status="error",
                ).inc()

    async def _execute_llm_analysis(
        self,
        analysis_type: LLMAnalysisType,
        request: EnhancedDiscoveryRequest,
        result: EnhancedDiscoveryResult,
    ) -> Optional[LLMAnalysisResult]:
        """Execute specific type of LLM analysis."""

        try:
            # Prepare context for LLM
            context = {
                "protocol_type": result.protocol_type,
                "confidence": result.confidence,
                "statistical_analysis": result.statistical_analysis,
                "classification_details": (
                    result.classification_details.__dict__
                    if result.classification_details
                    else None
                ),
                "validation_result": (
                    result.validation_result.__dict__
                    if result.validation_result
                    else None
                ),
                "message_count": len(request.messages),
                "user_context": request.user_context or {},
            }

            # Generate appropriate query based on analysis type
            query_text = self._generate_analysis_query(analysis_type, request, result)

            # Create copilot query
            copilot_query = CopilotQuery(
                query=query_text,
                query_type=self._map_analysis_to_copilot_type(analysis_type),
                user_id=(
                    request.user_context.get("user_id", "system")
                    if request.user_context
                    else "system"
                ),
                session_id=request.conversation_session_id
                or f"discovery_{int(time.time())}",
                context=context,
                packet_data=request.messages[0] if request.messages else None,
                enable_learning=True,
            )

            # Execute query via copilot
            copilot_response = await self.protocol_copilot.process_query(copilot_query)

            # Convert to LLM analysis result
            llm_result = LLMAnalysisResult(
                analysis_type=analysis_type,
                response=copilot_response.response,
                confidence=copilot_response.confidence,
                recommendations=copilot_response.suggestions,
                sources=copilot_response.sources,
                llm_provider=(
                    copilot_response.metadata.get("llm_provider")
                    if copilot_response.metadata
                    else None
                ),
            )

            # Extract specific insights based on analysis type
            await self._extract_analysis_insights(
                analysis_type, copilot_response, llm_result
            )

            return llm_result

        except Exception as e:
            self.logger.error(f"Failed to execute {analysis_type.value} analysis: {e}")
            return None

    def _generate_analysis_query(
        self,
        analysis_type: LLMAnalysisType,
        request: EnhancedDiscoveryRequest,
        result: EnhancedDiscoveryResult,
    ) -> str:
        """Generate appropriate query for specific analysis type."""

        base_context = f"Protocol discovered: {result.protocol_type} (confidence: {result.confidence:.2f})"

        if analysis_type == LLMAnalysisType.PROTOCOL_IDENTIFICATION:
            return f"{base_context}. Please provide detailed analysis of this protocol discovery, including its characteristics, typical use cases, and key identifying features."

        elif analysis_type == LLMAnalysisType.SECURITY_ASSESSMENT:
            return f"{base_context}. Perform a comprehensive security assessment. Identify potential vulnerabilities, attack vectors, and security best practices for this protocol."

        elif analysis_type == LLMAnalysisType.COMPLIANCE_CHECK:
            frameworks = request.compliance_frameworks or ["GDPR", "HIPAA", "PCI-DSS"]
            return f"{base_context}. Analyze compliance implications for {', '.join(frameworks)} frameworks. Identify any regulatory considerations or requirements."

        elif analysis_type == LLMAnalysisType.ANOMALY_EXPLANATION:
            return f"{base_context}. Analyze any anomalies or unusual patterns detected in the protocol traffic. Explain potential causes and implications."

        elif analysis_type == LLMAnalysisType.FIELD_SEMANTICS:
            return f"{base_context}. Provide semantic analysis of the protocol fields, their purposes, relationships, and data types."

        elif analysis_type == LLMAnalysisType.VULNERABILITY_ANALYSIS:
            return f"{base_context}. Conduct detailed vulnerability analysis. Identify known CVEs, common implementation flaws, and mitigation strategies."

        else:
            return f"{base_context}. Provide comprehensive analysis and insights about this protocol discovery."

    def _map_analysis_to_copilot_type(self, analysis_type: LLMAnalysisType) -> str:
        """Map LLM analysis type to copilot query type."""
        mapping = {
            LLMAnalysisType.PROTOCOL_IDENTIFICATION: "protocol_analysis",
            LLMAnalysisType.SECURITY_ASSESSMENT: "security_assessment",
            LLMAnalysisType.COMPLIANCE_CHECK: "compliance_check",
            LLMAnalysisType.ANOMALY_EXPLANATION: "anomaly_detection",
            LLMAnalysisType.FIELD_SEMANTICS: "field_detection",
            LLMAnalysisType.VULNERABILITY_ANALYSIS: "security_assessment",
        }
        return mapping.get(analysis_type, "general_question")

    async def _extract_analysis_insights(
        self,
        analysis_type: LLMAnalysisType,
        copilot_response: CopilotResponse,
        llm_result: LLMAnalysisResult,
    ) -> None:
        """Extract specific insights from copilot response based on analysis type."""

        response_text = copilot_response.response.lower()

        if analysis_type == LLMAnalysisType.SECURITY_ASSESSMENT:
            # Extract security-specific insights
            if any(
                term in response_text
                for term in ["vulnerable", "exploit", "attack", "risk"]
            ):
                llm_result.security_implications = (
                    "Security concerns identified - review recommendations"
                )
            else:
                llm_result.security_implications = (
                    "No immediate security concerns detected"
                )

        elif analysis_type == LLMAnalysisType.COMPLIANCE_CHECK:
            # Extract compliance-specific insights
            if any(
                term in response_text
                for term in ["compliant", "compliance", "regulatory"]
            ):
                llm_result.compliance_notes = (
                    "Compliance considerations identified - review analysis"
                )
            else:
                llm_result.compliance_notes = "Standard compliance considerations apply"

    async def _generate_natural_language_summary(
        self, result: EnhancedDiscoveryResult
    ) -> None:
        """Generate comprehensive natural language summary of discovery results."""

        if not self.protocol_copilot:
            return

        try:
            # Compile comprehensive context
            summary_context = {
                "protocol_type": result.protocol_type,
                "confidence": result.confidence,
                "phases_completed": [phase.value for phase in result.phases_completed],
                "llm_analyses": {
                    analysis_type.value: analysis_result.response
                    for analysis_type, analysis_result in result.llm_analyses.items()
                },
            }

            summary_query = f"""
            Provide a comprehensive, executive-level summary of this protocol discovery:
            
            Protocol: {result.protocol_type}
            Confidence: {result.confidence:.2f}
            Analysis phases completed: {len(result.phases_completed)}
            LLM analyses performed: {len(result.llm_analyses)}
            
            Please synthesize all findings into a clear, actionable summary suitable for technical and business stakeholders.
            Include key insights, recommendations, and any critical considerations.
            """

            copilot_query = CopilotQuery(
                query=summary_query,
                query_type="protocol_analysis",
                user_id="system",
                session_id=f"summary_{int(time.time())}",
                context=summary_context,
                enable_learning=False,
            )

            copilot_response = await self.protocol_copilot.process_query(copilot_query)

            result.natural_language_summary = copilot_response.response
            result.recommendations.extend(copilot_response.suggestions)

            # Extract security and compliance summaries
            if any(
                analysis_type
                in [
                    LLMAnalysisType.SECURITY_ASSESSMENT,
                    LLMAnalysisType.VULNERABILITY_ANALYSIS,
                ]
                for analysis_type in result.llm_analyses
            ):
                result.security_assessment = self._extract_security_summary(
                    result.llm_analyses
                )

            if LLMAnalysisType.COMPLIANCE_CHECK in result.llm_analyses:
                result.compliance_status = self._extract_compliance_summary(
                    result.llm_analyses
                )

        except Exception as e:
            self.logger.error(f"Failed to generate natural language summary: {e}")
            result.natural_language_summary = f"Protocol {result.protocol_type} discovered with {result.confidence:.2f} confidence"

    def _extract_security_summary(
        self, analyses: Dict[LLMAnalysisType, LLMAnalysisResult]
    ) -> str:
        """Extract security summary from LLM analyses."""
        security_analyses = [
            analysis
            for analysis_type, analysis in analyses.items()
            if analysis_type
            in [
                LLMAnalysisType.SECURITY_ASSESSMENT,
                LLMAnalysisType.VULNERABILITY_ANALYSIS,
            ]
        ]

        if not security_analyses:
            return "No security analysis performed"

        # Combine security insights
        high_confidence_findings = [
            analysis.response
            for analysis in security_analyses
            if analysis.confidence > 0.8
        ]

        if high_confidence_findings:
            return f"Security analysis completed with {len(high_confidence_findings)} high-confidence findings"
        else:
            return "Security analysis completed - review detailed findings"

    def _extract_compliance_summary(
        self, analyses: Dict[LLMAnalysisType, LLMAnalysisResult]
    ) -> str:
        """Extract compliance summary from LLM analyses."""
        compliance_analysis = analyses.get(LLMAnalysisType.COMPLIANCE_CHECK)

        if not compliance_analysis:
            return "No compliance analysis performed"

        if compliance_analysis.confidence > 0.8:
            return "Compliance analysis completed with high confidence"
        else:
            return "Compliance analysis completed - review findings"

    def _generate_llm_cache_key(
        self, messages: List[bytes], analysis_type: LLMAnalysisType
    ) -> str:
        """Generate cache key for LLM analysis results."""
        import hashlib

        content = b"".join(messages[:2])  # First 2 messages
        combined = content + analysis_type.value.encode("utf-8")
        return hashlib.sha256(combined).hexdigest()[:16]

    async def get_enhanced_protocol_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get enhanced protocol profiles with LLM analysis summaries."""
        base_profiles = await super().get_protocol_profiles()

        # Enhance with LLM insights if available
        for protocol_name, profile in base_profiles.items():
            if protocol_name in self.protocol_profiles:
                protocol_profile = self.protocol_profiles[protocol_name]

                # Add LLM enhancement indicators
                profile["llm_enhanced"] = True
                profile["analysis_capabilities"] = [
                    analysis_type.value for analysis_type in LLMAnalysisType
                ]

                # Add recent analysis counts if cache exists
                recent_analyses = sum(
                    1
                    for cache_analyses in self.llm_analysis_cache.values()
                    if any(analysis.response for analysis in cache_analyses.values())
                )
                profile["recent_llm_analyses"] = recent_analyses

        return base_profiles

    async def clear_llm_analysis_cache(self) -> int:
        """Clear LLM analysis cache and return number of items cleared."""
        count = len(self.llm_analysis_cache)
        self.llm_analysis_cache.clear()
        self.logger.info(f"Cleared {count} LLM analysis cache entries")
        return count

    async def get_llm_analysis_statistics(self) -> Dict[str, Any]:
        """Get LLM analysis statistics."""
        stats = {
            "total_cached_analyses": len(self.llm_analysis_cache),
            "analysis_types_available": [
                analysis_type.value for analysis_type in LLMAnalysisType
            ],
            "copilot_available": self.protocol_copilot is not None,
            "llm_service_available": self.llm_service is not None,
        }

        if self.protocol_copilot:
            copilot_health = self.protocol_copilot.get_health_status()
            stats["copilot_health"] = copilot_health

        return stats


# Factory function
async def create_enhanced_protocol_discovery_orchestrator(
    config: Config, protocol_copilot: Optional[ProtocolIntelligenceCopilot] = None
) -> EnhancedProtocolDiscoveryOrchestrator:
    """Create and initialize enhanced protocol discovery orchestrator."""
    orchestrator = EnhancedProtocolDiscoveryOrchestrator(config, protocol_copilot)
    await orchestrator.initialize()
    return orchestrator
