"""
QBITEL - Protocol Intelligence Copilot
Main implementation of the natural language interface for protocol analysis.
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from prometheus_client import Counter, Histogram

from ..core.config import get_config
from ..core.exceptions import QbitelAIException
from ..llm.unified_llm_service import get_llm_service, LLMRequest
from ..llm.rag_engine import RAGEngine
from .context_manager import ConversationContextManager
from .parser_improvement_engine import ParserImprovementEngine, ParserError
from .protocol_behavior_explainer import (
    ProtocolBehaviorExplainer,
    ProtocolBehaviorQuery,
    BehaviorExplanation,
)
from ..discovery.protocol_discovery_orchestrator import ProtocolDiscoveryOrchestrator

# Metrics
COPILOT_QUERY_COUNTER = Counter(
    "qbitel_copilot_queries_total", "Total copilot queries", ["query_type", "status"]
)
COPILOT_QUERY_DURATION = Histogram(
    "qbitel_copilot_query_duration_seconds", "Copilot query duration", ["query_type"]
)
COPILOT_CONFIDENCE_SCORE = Histogram(
    "qbitel_copilot_confidence_score", "Copilot confidence scores"
)

logger = logging.getLogger(__name__)


class CopilotException(QbitelAIException):
    """Copilot-specific exception."""

    pass


@dataclass
class CopilotQuery:
    """Copilot query structure."""

    query: str
    user_id: str
    session_id: str
    context: Dict[str, Any] = None
    query_type: str = "general"
    timestamp: datetime = None
    packet_data: Optional[bytes] = None


@dataclass
class CopilotResponse:
    """Copilot response structure."""

    response: str
    confidence: float
    source_data: List[Dict[str, Any]] = None
    processing_time: float = 0.0
    query_type: str = "general"
    suggestions: List[str] = None
    metadata: Dict[str, Any] = None
    visualizations: List[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = None  # Backward compatibility alias

    def __post_init__(self):
        """Initialize backward compatibility fields."""
        # If sources is provided but source_data is not, use sources
        if self.sources is not None and self.source_data is None:
            self.source_data = self.sources
        # If source_data is provided but sources is not, use source_data
        elif self.source_data is not None and self.sources is None:
            self.sources = self.source_data
        # Ensure both are set to same value for backward compatibility
        elif self.source_data is None and self.sources is None:
            self.source_data = []
            self.sources = []


class ProtocolIntelligenceCopilot:
    """
    Main Protocol Intelligence Copilot integrating with existing QBITEL architecture.
    Provides natural language interface for protocol analysis, field detection, and security assessment.
    """

    def __init__(self, discovery_orchestrator: ProtocolDiscoveryOrchestrator = None):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Core components - will be initialized in initialize()
        self.llm_service = None
        self.rag_engine = RAGEngine()
        self.context_manager = ConversationContextManager()
        self.discovery_orchestrator = discovery_orchestrator

        # New enhancement components
        self.parser_improvement_engine = None
        self.behavior_explainer = None

        # Query classification patterns
        self.query_patterns = {
            "protocol_analysis": [
                r"analyze.*protocol",
                r"what.*protocol.*is",
                r"identify.*protocol",
                r"protocol.*structure",
                r"protocol.*behavior",
                r"how.*protocol.*works",
            ],
            "field_detection": [
                r"fields?.*in.*message",
                r"parse.*message",
                r"message.*structure",
                r"field.*boundaries",
                r"extract.*fields",
                r"message.*format",
            ],
            "threat_assessment": [
                r"threat.*level",
                r"suspicious.*activity",
                r"security.*risk",
                r"anomaly.*detection",
                r"malicious.*traffic",
                r"attack.*pattern",
            ],
            "compliance_check": [
                r"compliance.*with",
                r"regulatory.*requirement",
                r"audit.*trail",
                r"policy.*violation",
                r"pci.*dss",
                r"hipaa.*compliance",
            ],
            "performance_analysis": [
                r"performance.*issues",
                r"latency.*analysis",
                r"throughput.*analysis",
                r"bandwidth.*usage",
                r"network.*optimization",
            ],
        }

        # Response templates for different query types
        self.response_templates = {
            "protocol_analysis": """
Based on the protocol analysis:

{analysis_result}

**Key Findings:**
{key_findings}

**Security Considerations:**
{security_implications}

**Technical Details:**
{technical_details}
            """,
            "field_detection": """
Field analysis results:

{field_results}

**Detected Fields:**
{detected_fields}

**Field Types:**
{field_types}

**Parsing Confidence:** {confidence}%
            """,
            "threat_assessment": """
Security Assessment:

**Threat Level:** {threat_level}

**Risk Factors:**
{risk_factors}

**Recommendations:**
{recommendations}

**Immediate Actions:**
{immediate_actions}
            """,
        }

    async def initialize(self) -> None:
        """Initialize copilot components."""
        try:
            self.logger.info("Initializing Protocol Intelligence Copilot...")

            # Get LLM service - it must be initialized first
            self.llm_service = get_llm_service()
            if self.llm_service is None:
                raise CopilotException(
                    "LLM service not initialized. Call initialize_llm_service() before creating copilot."
                )

            # Initialize all components
            await self.rag_engine.initialize()
            await self.context_manager.initialize()

            # Initialize discovery orchestrator if not provided
            if not self.discovery_orchestrator:
                self.discovery_orchestrator = ProtocolDiscoveryOrchestrator(self.config)
                await self.discovery_orchestrator.initialize()

            # Initialize enhancement components
            self.parser_improvement_engine = ParserImprovementEngine(
                self.llm_service, self.rag_engine
            )

            self.behavior_explainer = ProtocolBehaviorExplainer(
                self.llm_service, self.rag_engine
            )

            self.logger.info("Protocol Intelligence Copilot initialized successfully")

        except Exception as e:
            self.logger.error(
                f"Failed to initialize Protocol Intelligence Copilot: {e}"
            )
            raise CopilotException(f"Copilot initialization failed: {e}")

    async def process_query(self, query: CopilotQuery) -> CopilotResponse:
        """
        Process natural language query with full context awareness.

        Args:
            query: Copilot query with user input and context

        Returns:
            Comprehensive copilot response with analysis and suggestions
        """
        start_time = time.time()

        try:
            # Set timestamp if not provided
            if not query.timestamp:
                query.timestamp = datetime.now()

            # Classify query type
            query_type = self._classify_query(query.query)
            query.query_type = query_type

            # Get conversation context
            conversation_context = await self.context_manager.get_context(
                query.user_id, query.session_id
            )

            # Enhance with RAG knowledge
            enhanced_context = await self.rag_engine.enhance_query_context(
                query.query, {**conversation_context, **(query.context or {})}
            )

            # Process based on query type
            if query_type == "protocol_analysis":
                response = await self._handle_protocol_analysis(query, enhanced_context)
            elif query_type == "field_detection":
                response = await self._handle_field_detection(query, enhanced_context)
            elif query_type == "threat_assessment":
                response = await self._handle_threat_assessment(query, enhanced_context)
            elif query_type == "compliance_check":
                response = await self._handle_compliance_check(query, enhanced_context)
            elif query_type == "performance_analysis":
                response = await self._handle_performance_analysis(
                    query, enhanced_context
                )
            else:
                response = await self._handle_general_query(query, enhanced_context)

            # Update conversation context
            await self.context_manager.update_context(
                query.user_id,
                query.session_id,
                query.query,
                response.response,
                query_type,
                response.confidence,
                {"source_data": response.source_data},
            )

            # Update metrics
            COPILOT_QUERY_COUNTER.labels(query_type=query_type, status="success").inc()
            COPILOT_QUERY_DURATION.labels(query_type=query_type).observe(
                time.time() - start_time
            )
            COPILOT_CONFIDENCE_SCORE.observe(response.confidence)

            response.processing_time = time.time() - start_time
            return response

        except Exception as e:
            self.logger.error(f"Copilot query processing failed: {e}")
            COPILOT_QUERY_COUNTER.labels(
                query_type=query.query_type or "unknown", status="error"
            ).inc()

            return CopilotResponse(
                response="I apologize, but I encountered an error processing your query. Please try again or rephrase your question.",
                confidence=0.0,
                source_data=[],
                processing_time=time.time() - start_time,
                query_type=query.query_type or "error",
                metadata={"error": str(e)},
                suggestions=[
                    "Try rephrasing your question",
                    "Check if you've provided valid protocol data",
                    "Contact support if the issue persists",
                ],
            )

    def _classify_query(self, query: str) -> str:
        """Classify query type using pattern matching."""
        query_lower = query.lower()

        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type

        return "general"

    async def _handle_protocol_analysis(
        self, query: CopilotQuery, context: Dict[str, Any]
    ) -> CopilotResponse:
        """Handle protocol analysis queries."""
        try:
            # Analyze packet data if provided
            protocol_analysis = None
            if query.packet_data:
                protocol_analysis = await self._analyze_packet_data(query.packet_data)

            # Prepare LLM request
            llm_request = LLMRequest(
                prompt=f"""
                User Query: {query.query}
                
                {f"Protocol Analysis Results: {protocol_analysis}" if protocol_analysis else ""}
                
                Available Context:
                {self._format_context_for_llm(context)}
                
                Please provide a comprehensive protocol analysis response including:
                1. Direct answer to the user's question
                2. Technical protocol details
                3. Security implications
                4. Practical recommendations
                
                Be specific, technical, and actionable in your response.
                """,
                feature_domain="protocol_copilot",
                context=context,
                max_tokens=1500,
                temperature=0.2,
            )

            llm_response = await self.llm_service.process_request(llm_request)

            # Extract source data
            source_data = self._extract_source_data(context)
            if protocol_analysis:
                source_data.append(
                    {
                        "type": "protocol_analysis",
                        "data": protocol_analysis,
                        "confidence": protocol_analysis.get("confidence", 0.8),
                    }
                )

            return CopilotResponse(
                response=llm_response.content,
                confidence=llm_response.confidence,
                source_data=source_data,
                processing_time=0,  # Set by caller
                query_type="protocol_analysis",
                suggestions=self._generate_followup_suggestions(
                    "protocol_analysis", query.query
                ),
                metadata={
                    "llm_provider": llm_response.provider,
                    "tokens_used": llm_response.tokens_used,
                    "protocol_analysis_performed": protocol_analysis is not None,
                },
                visualizations=self._generate_visualizations(
                    "protocol_analysis", protocol_analysis
                ),
            )

        except Exception as e:
            self.logger.error(f"Protocol analysis failed: {e}")
            raise

    async def _handle_field_detection(
        self, query: CopilotQuery, context: Dict[str, Any]
    ) -> CopilotResponse:
        """Handle field detection queries."""
        try:
            field_analysis = None

            # Perform field detection if packet data provided
            if query.packet_data:
                field_analysis = await self._detect_message_fields(query.packet_data)

            llm_request = LLMRequest(
                prompt=f"""
                User Query: {query.query}
                
                {f"Field Detection Results: {field_analysis}" if field_analysis else ""}
                
                Context:
                {self._format_context_for_llm(context)}
                
                Provide detailed guidance on message field detection and structure analysis.
                Include:
                1. Field boundary detection techniques
                2. Field type inference methods
                3. Parsing strategies
                4. Common challenges and solutions
                """,
                feature_domain="protocol_copilot",
                context=context,
                max_tokens=1200,
                temperature=0.1,
            )

            llm_response = await self.llm_service.process_request(llm_request)

            source_data = self._extract_source_data(context)
            if field_analysis:
                source_data.append({"type": "field_detection", "data": field_analysis})

            return CopilotResponse(
                response=llm_response.content,
                confidence=llm_response.confidence,
                source_data=source_data,
                processing_time=0,
                query_type="field_detection",
                suggestions=self._generate_followup_suggestions(
                    "field_detection", query.query
                ),
                metadata={
                    "llm_provider": llm_response.provider,
                    "tokens_used": llm_response.tokens_used,
                    "field_analysis_performed": field_analysis is not None,
                },
                visualizations=self._generate_visualizations(
                    "field_detection", field_analysis
                ),
            )

        except Exception as e:
            self.logger.error(f"Field detection failed: {e}")
            raise

    async def _handle_threat_assessment(
        self, query: CopilotQuery, context: Dict[str, Any]
    ) -> CopilotResponse:
        """Handle threat assessment queries."""
        try:
            threat_analysis = None

            # Perform threat assessment if data provided
            if query.packet_data or query.context:
                threat_analysis = await self._assess_security_threats(
                    query.packet_data, query.context or {}
                )

            llm_request = LLMRequest(
                prompt=f"""
                User Query: {query.query}
                
                {f"Threat Analysis: {threat_analysis}" if threat_analysis else ""}
                
                Security Context:
                {self._format_security_context(context)}
                
                Provide a comprehensive security assessment including:
                1. Threat level evaluation
                2. Risk factors identification
                3. Attack vector analysis
                4. Mitigation recommendations
                5. Immediate action items
                """,
                feature_domain="security_orchestrator",  # Use security domain
                context=context,
                max_tokens=1800,
                temperature=0.1,
            )

            llm_response = await self.llm_service.process_request(llm_request)

            source_data = self._extract_source_data(context)
            if threat_analysis:
                source_data.append(
                    {"type": "threat_assessment", "data": threat_analysis}
                )

            return CopilotResponse(
                response=llm_response.content,
                confidence=llm_response.confidence,
                source_data=source_data,
                processing_time=0,
                query_type="threat_assessment",
                suggestions=self._generate_followup_suggestions(
                    "threat_assessment", query.query
                ),
                metadata={
                    "llm_provider": llm_response.provider,
                    "tokens_used": llm_response.tokens_used,
                    "threat_level": (
                        threat_analysis.get("threat_level", "unknown")
                        if threat_analysis
                        else "unknown"
                    ),
                },
            )

        except Exception as e:
            self.logger.error(f"Threat assessment failed: {e}")
            raise

    async def _handle_compliance_check(
        self, query: CopilotQuery, context: Dict[str, Any]
    ) -> CopilotResponse:
        """Handle compliance check queries."""
        try:
            # Extract compliance framework from query
            framework = self._extract_compliance_framework(query.query)

            llm_request = LLMRequest(
                prompt=f"""
                User Query: {query.query}
                
                Compliance Framework: {framework}
                
                Compliance Context:
                {self._format_compliance_context(context)}
                
                Provide detailed compliance guidance including:
                1. Specific regulatory requirements
                2. Compliance gap analysis
                3. Remediation steps
                4. Documentation requirements
                5. Audit preparation guidance
                """,
                feature_domain="compliance_reporter",  # Use compliance domain
                context=context,
                max_tokens=2000,
                temperature=0.1,
            )

            llm_response = await self.llm_service.process_request(llm_request)

            return CopilotResponse(
                response=llm_response.content,
                confidence=llm_response.confidence,
                source_data=self._extract_source_data(context),
                processing_time=0,
                query_type="compliance_check",
                suggestions=self._generate_followup_suggestions(
                    "compliance_check", query.query
                ),
                metadata={
                    "llm_provider": llm_response.provider,
                    "tokens_used": llm_response.tokens_used,
                    "compliance_framework": framework,
                },
            )

        except Exception as e:
            self.logger.error(f"Compliance check failed: {e}")
            raise

    async def _handle_performance_analysis(
        self, query: CopilotQuery, context: Dict[str, Any]
    ) -> CopilotResponse:
        """Handle performance analysis queries."""
        try:
            llm_request = LLMRequest(
                prompt=f"""
                User Query: {query.query}
                
                Performance Context:
                {self._format_context_for_llm(context)}
                
                Provide performance analysis and optimization guidance including:
                1. Performance bottleneck identification
                2. Optimization strategies
                3. Monitoring recommendations
                4. Best practices
                """,
                feature_domain="protocol_copilot",
                context=context,
                max_tokens=1500,
                temperature=0.2,
            )

            llm_response = await self.llm_service.process_request(llm_request)

            return CopilotResponse(
                response=llm_response.content,
                confidence=llm_response.confidence,
                source_data=self._extract_source_data(context),
                processing_time=0,
                query_type="performance_analysis",
                suggestions=self._generate_followup_suggestions(
                    "performance_analysis", query.query
                ),
                metadata={
                    "llm_provider": llm_response.provider,
                    "tokens_used": llm_response.tokens_used,
                },
            )

        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            raise

    async def _handle_general_query(
        self, query: CopilotQuery, context: Dict[str, Any]
    ) -> CopilotResponse:
        """Handle general protocol-related queries."""
        try:
            llm_request = LLMRequest(
                prompt=f"""
                User Query: {query.query}
                
                Context:
                {self._format_context_for_llm(context)}
                
                Please provide a helpful response about protocol analysis, network security, 
                or cybersecurity topics. Be informative and practical.
                """,
                feature_domain="protocol_copilot",
                context=context,
                max_tokens=1200,
                temperature=0.3,
            )

            llm_response = await self.llm_service.process_request(llm_request)

            return CopilotResponse(
                response=llm_response.content,
                confidence=llm_response.confidence
                * 0.8,  # Lower confidence for general queries
                source_data=self._extract_source_data(context),
                processing_time=0,
                query_type="general",
                suggestions=self._generate_followup_suggestions("general", query.query),
                metadata={
                    "llm_provider": llm_response.provider,
                    "tokens_used": llm_response.tokens_used,
                },
            )

        except Exception as e:
            self.logger.error(f"General query failed: {e}")
            raise

    # Helper methods for data analysis and formatting

    async def _analyze_packet_data(self, packet_data: bytes) -> Dict[str, Any]:
        """Analyze packet data using existing QBITEL capabilities."""
        try:
            from ..discovery.protocol_discovery_orchestrator import DiscoveryRequest

            discovery_request = DiscoveryRequest(
                messages=[packet_data],
                confidence_threshold=0.6,
                generate_parser=True,
                validate_results=True,
            )

            result = await self.discovery_orchestrator.discover_protocol(
                discovery_request
            )

            return {
                "protocol_type": result.protocol_type,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "grammar_rules": len(result.grammar.rules) if result.grammar else 0,
                "parser_available": result.parser is not None,
                "validation_result": (
                    result.validation_result.is_valid
                    if result.validation_result
                    else None
                ),
            }

        except Exception as e:
            self.logger.error(f"Packet analysis failed: {e}")
            return {"error": str(e)}

    async def _detect_message_fields(self, message_data: bytes) -> Dict[str, Any]:
        """Detect fields in message data."""
        try:
            from ..detection.field_detector import FieldDetector

            field_detector = FieldDetector(
                vocab_size=256, embedding_dim=128, hidden_dim=256, num_tags=5
            )

            data_sequence = list(message_data[:512])
            fields = field_detector.detect_fields(data_sequence)

            return {
                "message_size": len(message_data),
                "detected_fields": len(fields),
                "field_details": fields[:10],  # First 10 fields
                "confidence": (
                    sum(f.get("confidence", 0) for f in fields) / len(fields)
                    if fields
                    else 0
                ),
            }

        except Exception as e:
            self.logger.error(f"Field detection failed: {e}")
            return {"error": str(e)}

    async def _assess_security_threats(
        self, packet_data: Optional[bytes], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess security threats in the data."""
        try:
            threat_indicators = []
            threat_level = "low"

            # Simple threat assessment logic
            if packet_data:
                # Check packet size anomalies
                if len(packet_data) > 8192:
                    threat_indicators.append("Unusually large packet size")
                    threat_level = "medium"

                # Check for suspicious patterns (simplified)
                suspicious_bytes = [0x00, 0xFF, 0x90, 0xCC]  # NOP, padding, etc.
                suspicious_count = sum(
                    1 for byte in packet_data if byte in suspicious_bytes
                )

                if suspicious_count > len(packet_data) * 0.5:
                    threat_indicators.append("High concentration of suspicious bytes")
                    threat_level = "high"

            # Check context for additional threats
            if context.get("source_ip") in ["127.0.0.1", "0.0.0.0"]:
                threat_indicators.append("Suspicious source IP address")

            return {
                "threat_level": threat_level,
                "threat_indicators": threat_indicators,
                "risk_score": len(threat_indicators) * 0.3,
                "assessment_time": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Threat assessment failed: {e}")
            return {"error": str(e)}

    def _extract_compliance_framework(self, query: str) -> str:
        """Extract compliance framework from query."""
        frameworks = {
            "pci": "PCI-DSS",
            "hipaa": "HIPAA",
            "sox": "SOX",
            "gdpr": "GDPR",
            "basel": "Basel III",
            "nerc": "NERC CIP",
            "iso": "ISO 27001",
        }

        query_lower = query.lower()
        for key, framework in frameworks.items():
            if key in query_lower:
                return framework

        return "General Compliance"

    def _format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """Format context for LLM consumption."""
        formatted = []

        # Format relevant protocols
        if context.get("relevant_protocols"):
            formatted.append("Relevant Protocol Knowledge:")
            for protocol in context["relevant_protocols"][:3]:
                formatted.append(
                    f"- {protocol.get('metadata', {}).get('protocol', 'Unknown')}: {protocol.get('content', '')[:200]}..."
                )

        # Format conversation history
        if context.get("recent_conversation"):
            formatted.append("\nRecent Conversation:")
            for turn in context["recent_conversation"][-2:]:  # Last 2 turns
                formatted.append(f"Q: {turn['user_query'][:100]}...")
                formatted.append(f"A: {turn['assistant_response'][:100]}...")

        return "\n".join(formatted) if formatted else "No specific context available."

    def _format_security_context(self, context: Dict[str, Any]) -> str:
        """Format security-specific context."""
        security_context = []

        if context.get("security_context"):
            security_context.append("Security Patterns:")
            for pattern in context["security_context"][:2]:
                security_context.append(f"- {pattern.get('content', '')[:150]}...")

        return (
            "\n".join(security_context)
            if security_context
            else "No security context available."
        )

    def _format_compliance_context(self, context: Dict[str, Any]) -> str:
        """Format compliance-specific context."""
        compliance_context = []

        if context.get("compliance_context"):
            compliance_context.append("Compliance Information:")
            for rule in context["compliance_context"][:2]:
                compliance_context.append(f"- {rule.get('content', '')[:150]}...")

        return (
            "\n".join(compliance_context)
            if compliance_context
            else "No compliance context available."
        )

    def _extract_source_data(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract source data references for transparency."""
        source_data = []

        if context.get("relevant_protocols"):
            for protocol in context["relevant_protocols"]:
                source_data.append(
                    {
                        "type": "protocol_knowledge",
                        "name": protocol.get("metadata", {}).get("protocol"),
                        "similarity": protocol.get("similarity", 0.0),
                        "source": "knowledge_base",
                    }
                )

        if context.get("rag_metadata"):
            source_data.append(
                {
                    "type": "rag_search",
                    "total_results": context["rag_metadata"].get("total_results", 0),
                    "avg_similarity": context["rag_metadata"].get(
                        "avg_similarity", 0.0
                    ),
                    "source": "vector_database",
                }
            )

        return source_data

    def _generate_followup_suggestions(
        self, query_type: str, original_query: str
    ) -> List[str]:
        """Generate contextual follow-up suggestions."""
        base_suggestions = {
            "protocol_analysis": [
                "Analyze security implications of this protocol",
                "Show me similar protocol patterns",
                "What are the compliance requirements?",
                "How can I optimize this protocol's performance?",
            ],
            "field_detection": [
                "How can I improve field detection accuracy?",
                "Show me field detection confidence scores",
                "Analyze field semantics and types",
                "Generate a parser for this message format",
            ],
            "threat_assessment": [
                "What mitigation strategies do you recommend?",
                "Show me related security events",
                "Analyze attack patterns for this threat",
                "Create an incident response plan",
            ],
            "compliance_check": [
                "Show me specific compliance gaps",
                "Generate a compliance report",
                "What are the remediation steps?",
                "Schedule automated compliance monitoring",
            ],
            "performance_analysis": [
                "Identify specific bottlenecks",
                "Recommend optimization strategies",
                "Set up performance monitoring",
                "Compare with industry benchmarks",
            ],
        }

        suggestions = base_suggestions.get(
            query_type,
            [
                "Tell me more about this topic",
                "Show me related information",
                "How can I learn more about this?",
                "What are the best practices?",
            ],
        )

        return suggestions[:3]  # Return top 3 suggestions

    def _generate_visualizations(
        self, query_type: str, analysis_data: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Generate visualization suggestions for the response."""
        visualizations = []

        if query_type == "protocol_analysis" and analysis_data:
            visualizations.append(
                {
                    "type": "protocol_structure",
                    "title": "Protocol Structure Diagram",
                    "data": {
                        "protocol_type": analysis_data.get("protocol_type"),
                        "confidence": analysis_data.get("confidence"),
                        "fields_detected": analysis_data.get("grammar_rules", 0),
                    },
                }
            )

        elif query_type == "field_detection" and analysis_data:
            visualizations.append(
                {
                    "type": "field_boundaries",
                    "title": "Message Field Boundaries",
                    "data": {
                        "message_size": analysis_data.get("message_size"),
                        "field_count": analysis_data.get("detected_fields"),
                        "confidence": analysis_data.get("confidence"),
                    },
                }
            )

        return visualizations

    async def suggest_parser_improvements(
        self, parser_code: str, errors: List[str], context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest parser improvements based on errors.

        Args:
            parser_code: Parser source code
            errors: List of error messages
            context: Additional context

        Returns:
            List of improvement suggestions
        """
        try:
            if not self.parser_improvement_engine:
                raise CopilotException("Parser improvement engine not initialized")

            improvements = (
                await self.parser_improvement_engine.suggest_parser_improvements(
                    parser_code, errors, context
                )
            )

            # Convert to dict format
            return [
                {
                    "type": imp.improvement_type,
                    "title": imp.title,
                    "description": imp.description,
                    "code_example": imp.code_example,
                    "priority": imp.priority,
                    "impact": imp.estimated_impact,
                    "steps": imp.implementation_steps,
                    "confidence": imp.confidence,
                }
                for imp in improvements
            ]

        except Exception as e:
            self.logger.error(f"Failed to suggest parser improvements: {e}")
            return []

    async def explain_protocol_behavior(
        self,
        protocol_type: str,
        messages: List[bytes],
        question: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Explain protocol behavior based on message analysis.

        Args:
            protocol_type: Type of protocol
            messages: List of protocol messages
            question: User's question about behavior
            context: Additional context

        Returns:
            Behavior explanation with analysis
        """
        try:
            if not self.behavior_explainer:
                raise CopilotException("Behavior explainer not initialized")

            query = ProtocolBehaviorQuery(
                protocol_type=protocol_type,
                messages=messages,
                question=question,
                context=context or {},
                include_examples=True,
                detail_level="standard",
            )

            explanation = await self.behavior_explainer.explain_protocol_behavior(query)

            # Convert to dict format
            return {
                "explanation": explanation.explanation,
                "key_observations": explanation.key_observations,
                "message_patterns": explanation.message_patterns,
                "sequence_analysis": explanation.sequence_analysis,
                "security_implications": explanation.security_implications,
                "performance_notes": explanation.performance_notes,
                "examples": explanation.examples,
                "confidence": explanation.confidence,
                "sources": explanation.sources,
                "metadata": explanation.metadata,
            }

        except Exception as e:
            self.logger.error(f"Failed to explain protocol behavior: {e}")
            return {
                "explanation": f"Unable to explain protocol behavior: {str(e)}",
                "confidence": 0.0,
                "error": str(e),
            }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of copilot components."""
        return {
            "copilot": "healthy",
            "llm_service": self.llm_service.get_health_status(),
            "rag_engine": self.rag_engine.get_collection_stats(),
            "discovery_orchestrator": (
                "connected" if self.discovery_orchestrator else "not_connected"
            ),
            "context_manager": "active",
            "parser_improvement_engine": (
                "active" if self.parser_improvement_engine else "not_initialized"
            ),
            "behavior_explainer": (
                "active" if self.behavior_explainer else "not_initialized"
            ),
            "query_patterns": len(self.query_patterns),
            "response_templates": len(self.response_templates),
        }

    async def shutdown(self) -> None:
        """Shutdown copilot and all components."""
        try:
            self.logger.info("Shutting down Protocol Intelligence Copilot...")

            await self.llm_service.shutdown()
            await self.context_manager.shutdown()

            if self.discovery_orchestrator:
                await self.discovery_orchestrator.shutdown()

            self.logger.info("Protocol Intelligence Copilot shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during copilot shutdown: {e}")


# Factory function for easy integration
async def create_protocol_copilot(
    discovery_orchestrator: ProtocolDiscoveryOrchestrator = None,
) -> ProtocolIntelligenceCopilot:
    """Factory function to create Protocol Intelligence Copilot."""
    copilot = ProtocolIntelligenceCopilot(discovery_orchestrator)
    await copilot.initialize()
    return copilot
