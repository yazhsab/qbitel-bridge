"""
QBITEL - Data Flow Analyzer Service

Protocol differences analysis and risk assessment for legacy modernization.

Responsibilities:
- Analyze data flow between legacy and target protocols
- Identify transformation requirements
- Map integration points
- Assess performance implications
- Identify security concerns
- Calculate overall modernization risk level
"""

import json
import logging
from typing import Dict, Any, List, Optional

from .models import (
    ProtocolSpecification,
    ModernizationRisk,
    LegacyWhispererException,
)
from ...llm.unified_llm_service import get_llm_service, LLMRequest

logger = logging.getLogger(__name__)


class DataFlowAnalyzer:
    """
    Service for analyzing data flow and assessing modernization risks.

    Features:
    - Protocol differences analysis
    - Risk assessment for modernization
    - Integration point mapping
    - Data transformation requirements identification
    """

    def __init__(self, llm_service=None):
        """
        Initialize the Data Flow Analyzer.

        Args:
            llm_service: Optional LLM service instance
        """
        self.llm_service = llm_service or get_llm_service()
        self.logger = logging.getLogger(__name__)

    async def analyze_protocol_differences(
        self, legacy_protocol: ProtocolSpecification, target_protocol: str
    ) -> Dict[str, Any]:
        """
        Analyze differences between legacy and target protocols.

        Args:
            legacy_protocol: Legacy protocol specification
            target_protocol: Target protocol name

        Returns:
            Dictionary with protocol differences analysis
        """
        llm_request = LLMRequest(
            prompt=f"""
            Analyze the differences between this legacy protocol and {target_protocol}:

            Legacy Protocol: {legacy_protocol.protocol_name}
            Description: {legacy_protocol.description}
            Complexity: {legacy_protocol.complexity.value}

            Fields:
            {chr(10).join(f"- {f.name}: {f.field_type} ({f.description})" for f in legacy_protocol.fields)}

            Target Protocol: {target_protocol}

            Please identify:
            1. Key structural differences
            2. Data transformation requirements
            3. Integration points needed
            4. API endpoints to create
            5. Performance considerations
            6. Security implications

            Format as JSON.
            """,
            feature_domain="legacy_whisperer",
            context={"analysis_type": "protocol_differences"},
            max_tokens=2000,
            temperature=0.1,
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
                "structural_differences": [],
                "transformation_requirements": [],
                "integration_points": [],
                "api_endpoints": [],
                "performance_notes": "",
                "security_implications": [],
            }

    async def assess_modernization_risks(
        self, behavior: str, context: Dict[str, Any], approaches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Assess modernization risks.

        Args:
            behavior: Description of the behavior
            context: Additional context
            approaches: Proposed modernization approaches

        Returns:
            List of risk assessments
        """
        llm_request = LLMRequest(
            prompt=f"""
            Assess risks for modernizing this legacy behavior:

            Behavior: {behavior}
            Context: {json.dumps(context, indent=2)}
            Proposed Approaches: {json.dumps(approaches, indent=2)}

            Identify risks in these categories:
            1. Technical risks
            2. Business risks
            3. Operational risks
            4. Security risks
            5. Compliance risks

            For each risk provide:
            - Risk description
            - Severity (low/medium/high/critical)
            - Likelihood (low/medium/high)
            - Mitigation strategies

            Format as JSON array.
            """,
            feature_domain="legacy_whisperer",
            context={"analysis_type": "risk_assessment"},
            max_tokens=2500,
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
            return [
                {
                    "category": "technical",
                    "description": "Compatibility issues with existing systems",
                    "severity": "medium",
                    "likelihood": "medium",
                    "mitigation": ["Comprehensive testing", "Phased rollout"],
                }
            ]

    def determine_overall_risk(self, risks: List[Dict[str, Any]]) -> ModernizationRisk:
        """
        Determine overall risk level from individual risks.

        Args:
            risks: List of risk assessments

        Returns:
            Overall modernization risk level
        """
        if not risks:
            return ModernizationRisk.LOW

        # Count risks by severity
        critical_count = sum(1 for r in risks if r.get("severity") == "critical")
        high_count = sum(1 for r in risks if r.get("severity") == "high")

        if critical_count > 0:
            return ModernizationRisk.CRITICAL
        elif high_count >= 3:
            return ModernizationRisk.HIGH
        elif high_count > 0:
            return ModernizationRisk.MEDIUM
        else:
            return ModernizationRisk.LOW

    def select_best_approach(self, approaches: List[Dict[str, Any]], risks: List[Dict[str, Any]]) -> Optional[str]:
        """
        Select the best modernization approach based on analysis.

        Args:
            approaches: Available modernization approaches
            risks: Risk assessments

        Returns:
            Name of the recommended approach, or None
        """
        if not approaches:
            return None

        # Score each approach based on complexity and risks
        scores: Dict[str, float] = {}
        for approach in approaches:
            score = 0.0

            # Lower complexity is better
            complexity = approach.get("complexity", "medium")
            if complexity == "low":
                score += 0.4
            elif complexity == "medium":
                score += 0.2

            # Consider benefits
            benefits = approach.get("benefits", [])
            score += len(benefits) * 0.1

            # Consider drawbacks (negative)
            drawbacks = approach.get("drawbacks", [])
            score -= len(drawbacks) * 0.05

            scores[approach.get("name", "")] = score

        # Return approach with highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return approaches[0].get("name")
