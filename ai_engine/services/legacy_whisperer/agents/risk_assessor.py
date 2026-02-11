"""
QBITEL - Risk Assessor Agent

Specialized agent for evaluating modernization risks.
"""

import logging
from typing import Dict, List, Any, Optional

from .base import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    AgentMessage,
    MessageType,
)
from .tools import RiskCalculatorTool


logger = logging.getLogger(__name__)


class RiskAssessorAgent(BaseAgent):
    """
    Agent specialized in risk assessment.

    Responsibilities:
    - Evaluate technical risks
    - Assess business impact
    - Identify operational concerns
    - Recommend risk mitigation strategies
    """

    def __init__(self, llm_service: Any, config: Optional[AgentConfig] = None):
        config = config or AgentConfig(
            role=AgentRole.RISK_ASSESSOR,
            name="Risk Assessment Specialist",
            description=(
                "Expert in evaluating modernization risks for legacy systems. "
                "Assesses technical, business, and operational risks with "
                "practical mitigation strategies."
            ),
            model="gpt-4o",
            temperature=0.4,  # More deterministic for risk assessment
            max_tokens=4096,
            max_iterations=10,
            enable_reflection=True,
        )

        # Initialize tools
        tools = [
            RiskCalculatorTool(),
        ]

        super().__init__(config, llm_service, tools)

        self.assessments: Dict[str, Dict[str, Any]] = {}

    @property
    def system_prompt(self) -> str:
        """Risk assessor specific system prompt."""
        base_prompt = super().system_prompt

        risk_prompt = """

## Risk Assessment Expertise

As a Risk Assessment Specialist, you evaluate:

1. **Technical Risks**
   - Protocol complexity and undocumented behavior
   - Data integrity during migration
   - Performance degradation
   - Integration challenges
   - Technical debt accumulation

2. **Business Risks**
   - Service disruption impact
   - Compliance implications
   - Cost overruns
   - Timeline delays
   - Stakeholder resistance

3. **Operational Risks**
   - Training requirements
   - Operational procedure changes
   - Monitoring and support needs
   - Rollback complexity
   - Disaster recovery impact

4. **Security Risks**
   - Vulnerability exposure during transition
   - Authentication/authorization gaps
   - Data leakage potential
   - Audit trail continuity

## Risk Assessment Framework

Use the FAIR (Factor Analysis of Information Risk) inspired approach:

1. **Identify Assets**: What's at risk?
2. **Identify Threats**: What could go wrong?
3. **Estimate Probability**: How likely?
4. **Estimate Impact**: How severe?
5. **Calculate Risk**: Probability × Impact
6. **Prioritize**: Focus on highest risks
7. **Mitigate**: Develop countermeasures

## Risk Levels

- **Critical** (>3.5): Immediate attention required, may block modernization
- **High** (2.5-3.5): Significant risk, needs mitigation plan before proceeding
- **Medium** (1.5-2.5): Manageable with standard precautions
- **Low** (<1.5): Acceptable risk, monitor during implementation

## Output Format

```
RISK ASSESSMENT REPORT
=====================

Overall Risk Level: [Critical/High/Medium/Low]
Overall Risk Score: [0.0-4.0]

Technical Risks:
- Risk 1: [Description] - [Level] ([Score])
  Mitigation: [Strategy]
- Risk 2: ...

Business Risks:
- Risk 1: ...

Operational Risks:
- Risk 1: ...

Security Risks:
- Risk 1: ...

Recommended Actions:
1. [Priority action]
2. [Secondary action]
...

Go/No-Go Recommendation: [Proceed/Proceed with caution/Delay/Do not proceed]
```
"""
        return base_prompt + risk_prompt

    async def assess_modernization_risk(
        self,
        protocol_spec: Dict[str, Any],
        target_protocol: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess risks for modernizing a protocol.

        Args:
            protocol_spec: Current protocol specification
            target_protocol: Target modern protocol
            context: Business and operational context

        Returns:
            Comprehensive risk assessment
        """
        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Assess the risks of modernizing this legacy protocol to {target_protocol}.

Current Protocol:
{self._format_spec(protocol_spec)}

Target: {target_protocol}

Business Context:
- Criticality: {context.get('criticality', 'unknown')}
- Dependencies: {context.get('dependencies', 'unknown')}
- Timeline: {context.get('timeline', 'unknown')}

Your task:
1. Use the calculate_risk tool to get base risk scores
2. Identify specific risks in each category
3. Develop mitigation strategies
4. Provide a go/no-go recommendation

Be thorough but practical. Focus on actionable insights.
""",
            data={
                "protocol_spec": protocol_spec,
                "target_protocol": target_protocol,
                "context": context
            }
        )

        response = await self.process_message(message)

        # Store assessment
        assessment_id = f"assess_{len(self.assessments)}"
        self.assessments[assessment_id] = {
            "protocol": protocol_spec.get("protocol_name", "unknown"),
            "target": target_protocol,
            "result": response.data
        }

        return response.data

    async def evaluate_approach_risks(
        self,
        approaches: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate risks for different modernization approaches.

        Args:
            approaches: List of potential approaches
            context: Business and operational context

        Returns:
            Risk evaluation for each approach
        """
        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Evaluate the risks of these modernization approaches.

Approaches:
{self._format_approaches(approaches)}

Context:
{self._format_context(context)}

Your task:
For each approach:
1. Calculate risk scores using the tool
2. Identify specific risks
3. Compare trade-offs
4. Rank approaches by risk-adjusted value

Provide a recommendation on which approach to pursue.
""",
            data={
                "approaches": approaches,
                "context": context
            }
        )

        response = await self.process_message(message)
        return response.data.get("evaluations", {})

    async def identify_dependencies(
        self,
        protocol_spec: Dict[str, Any],
        system_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify system dependencies that may impact modernization.

        Args:
            protocol_spec: Protocol specification
            system_info: Information about connected systems

        Returns:
            List of dependencies with risk ratings
        """
        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Identify dependencies that could impact protocol modernization.

Protocol: {protocol_spec.get('protocol_name', 'Unknown')}

System Information:
{self._format_context(system_info)}

Your task:
1. Identify upstream dependencies (systems that send to this)
2. Identify downstream dependencies (systems that receive from this)
3. Assess risk of each dependency
4. Suggest coordination requirements

For each dependency, provide:
- System name/type
- Dependency type (upstream/downstream/bidirectional)
- Impact if disrupted
- Coordination needed
""",
            data={
                "protocol_spec": protocol_spec,
                "system_info": system_info
            }
        )

        response = await self.process_message(message)
        return response.data.get("dependencies", [])

    def _format_spec(self, spec: Dict[str, Any]) -> str:
        """Format protocol spec for prompt."""
        lines = [f"- {k}: {v}" for k, v in spec.items() if not isinstance(v, (dict, list))]
        return "\n".join(lines)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt."""
        return "\n".join(f"- {k}: {v}" for k, v in context.items())

    def _format_approaches(self, approaches: List[Dict[str, Any]]) -> str:
        """Format approaches for prompt."""
        lines = []
        for i, approach in enumerate(approaches, 1):
            lines.append(f"\n{i}. {approach.get('name', f'Approach {i}')}")
            for k, v in approach.items():
                if k != 'name':
                    lines.append(f"   - {k}: {v}")
        return "\n".join(lines)


# Factory function
def create_risk_assessor(llm_service: Any) -> RiskAssessorAgent:
    """Create a Risk Assessor agent."""
    return RiskAssessorAgent(llm_service)


__all__ = ["RiskAssessorAgent", "create_risk_assessor"]
