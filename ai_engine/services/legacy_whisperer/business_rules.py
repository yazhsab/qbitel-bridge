"""
QBITEL - Business Rules Extractor Service

Documentation generation, behavior analysis, and historical context for legacy systems.

Responsibilities:
- Extract business rules from protocol specifications
- Generate protocol documentation
- Assess protocol complexity
- Calculate analysis confidence scores
- Gather historical context via RAG
- Identify root causes of legacy behaviors
- Generate modernization recommendations
"""

import json
import logging
from typing import Dict, Any, List, Optional

from .models import (
    ProtocolPattern,
    ProtocolField,
    ProtocolSpecification,
    LegacyWhispererException,
)
from ...llm.unified_llm_service import get_llm_service, LLMRequest
from ...llm.rag_engine import RAGEngine

logger = logging.getLogger(__name__)


class BusinessRulesExtractor:
    """
    Service for extracting business rules and generating documentation.

    Features:
    - Comprehensive protocol documentation generation
    - Behavior analysis using LLM
    - Historical context retrieval via RAG
    - Modernization approach suggestions
    """

    def __init__(self, llm_service=None, rag_engine=None):
        """
        Initialize the Business Rules Extractor.

        Args:
            llm_service: Optional LLM service instance
            rag_engine: Optional RAG engine instance
        """
        self.llm_service = llm_service or get_llm_service()
        self.rag_engine = rag_engine or RAGEngine()
        self.logger = logging.getLogger(__name__)

    async def generate_documentation(
        self,
        samples: List[bytes],
        patterns: List[ProtocolPattern],
        fields: List[ProtocolField],
        message_types: List[Dict[str, Any]],
        characteristics: Dict[str, bool],
        system_context: str,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive protocol documentation using LLM.

        Args:
            samples: Traffic samples
            patterns: Identified patterns
            fields: Identified fields
            message_types: Identified message types
            characteristics: Protocol characteristics
            system_context: Additional context

        Returns:
            Documentation dictionary
        """
        llm_request = LLMRequest(
            prompt=f"""
            Generate comprehensive documentation for this legacy protocol:

            System Context: {system_context}

            Analyzed Samples: {len(samples)}

            Identified Patterns:
            {chr(10).join(f"- {p.pattern_type}: {p.description} (confidence: {p.confidence:.2%})" for p in patterns)}

            Protocol Fields:
            {chr(10).join(f"- {f.name} at offset {f.offset}, length {f.length}: {f.description}" for f in fields)}

            Message Types: {len(message_types)}

            Characteristics:
            - Binary Protocol: {characteristics['is_binary']}
            - Stateful: {characteristics['is_stateful']}
            - Uses Encryption: {characteristics['uses_encryption']}
            - Has Checksums: {characteristics['has_checksums']}

            Please provide:
            1. Protocol name (if identifiable)
            2. Version information
            3. Detailed description
            4. Full technical documentation
            5. Usage examples
            6. Known implementations
            7. Historical context
            8. Common issues
            9. Security concerns

            Format as JSON with these keys.
            """,
            feature_domain="legacy_whisperer",
            context={"analysis_type": "documentation_generation"},
            max_tokens=3000,
            temperature=0.2,
        )

        response = await self.llm_service.process_request(llm_request)

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())
        except:
            # Return basic documentation as fallback
            return {
                "protocol_name": "Unknown Legacy Protocol",
                "version": "1.0",
                "description": "Legacy protocol identified through traffic analysis",
                "full_documentation": response.content,
                "usage_examples": [],
                "known_implementations": [],
                "historical_context": "",
                "common_issues": [],
                "security_concerns": [],
            }

    async def analyze_behavior_with_llm(self, behavior: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze legacy behavior using LLM.

        Args:
            behavior: Description of the legacy behavior
            context: Additional context

        Returns:
            Analysis dictionary
        """
        llm_request = LLMRequest(
            prompt=f"""
            Analyze this legacy system behavior:

            Behavior: {behavior}

            Context:
            {json.dumps(context, indent=2)}

            Provide:
            1. Technical explanation of why this behavior exists
            2. Root causes (technical debt, historical decisions, etc.)
            3. Implications for the system
            4. Confidence in your analysis

            Format as JSON with keys: technical_explanation, root_causes, implications, confidence
            """,
            feature_domain="legacy_whisperer",
            context={"analysis_type": "behavior_analysis"},
            max_tokens=2000,
            temperature=0.2,
        )

        response = await self.llm_service.process_request(llm_request)

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())
        except:
            return {
                "technical_explanation": response.content,
                "root_causes": [],
                "implications": [],
                "confidence": 0.7,
            }

    async def gather_historical_context(self, behavior: str, context: Dict[str, Any]) -> str:
        """
        Gather historical context using RAG.

        Args:
            behavior: Description of the behavior
            context: Additional context

        Returns:
            Historical context string
        """
        # Search knowledge base for similar behaviors
        query = f"legacy system behavior: {behavior}"
        results = await self.rag_engine.query_similar(query, collection_name="protocol_knowledge", n_results=3)

        if results.documents:
            historical_context = "Historical Context:\n\n"
            for doc in results.documents:
                historical_context += f"- {doc.content[:200]}...\n"
            return historical_context

        return "No specific historical context found in knowledge base."

    async def suggest_modernization_approaches(
        self, behavior: str, context: Dict[str, Any], analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Suggest modernization approaches.

        Args:
            behavior: Description of the behavior
            context: Additional context
            analysis: Behavior analysis results

        Returns:
            List of modernization approaches
        """
        llm_request = LLMRequest(
            prompt=f"""
            Suggest modernization approaches for this legacy behavior:

            Behavior: {behavior}
            Analysis: {json.dumps(analysis, indent=2)}
            Context: {json.dumps(context, indent=2)}

            Provide 3-5 different modernization approaches, each with:
            1. Approach name
            2. Description
            3. Benefits
            4. Drawbacks
            5. Complexity (low/medium/high)
            6. Estimated timeline
            7. Required resources

            Format as JSON array.
            """,
            feature_domain="legacy_whisperer",
            context={"analysis_type": "modernization_approaches"},
            max_tokens=2500,
            temperature=0.3,
        )

        response = await self.llm_service.process_request(llm_request)

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())
        except:
            return [
                {
                    "name": "Gradual Modernization",
                    "description": "Incrementally replace legacy components",
                    "benefits": ["Lower risk", "Continuous operation"],
                    "drawbacks": ["Longer timeline"],
                    "complexity": "medium",
                    "timeline": "6-12 months",
                    "resources": ["2-3 developers", "1 architect"],
                }
            ]

    def calculate_completeness(
        self,
        analysis: Dict[str, Any],
        approaches: List[Dict[str, Any]],
        risks: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate explanation completeness.

        Args:
            analysis: Behavior analysis results
            approaches: Modernization approaches
            risks: Risk assessment results

        Returns:
            Completeness score (0.0 to 1.0)
        """
        completeness = 0.0

        # Check analysis completeness
        if analysis.get("technical_explanation"):
            completeness += 0.25
        if analysis.get("root_causes"):
            completeness += 0.15
        if analysis.get("implications"):
            completeness += 0.15

        # Check approaches
        if len(approaches) >= 3:
            completeness += 0.25
        elif len(approaches) > 0:
            completeness += 0.15

        # Check risks
        if len(risks) >= 3:
            completeness += 0.20
        elif len(risks) > 0:
            completeness += 0.10

        return min(1.0, completeness)
