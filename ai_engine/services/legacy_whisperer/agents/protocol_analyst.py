"""
QBITEL - Protocol Analyst Agent

Specialized agent for analyzing protocol traffic and structure.
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
from .tools import TrafficAnalysisTool, PatternRecognitionTool


logger = logging.getLogger(__name__)


class ProtocolAnalystAgent(BaseAgent):
    """
    Agent specialized in protocol analysis.

    Responsibilities:
    - Analyze traffic patterns
    - Identify protocol structure
    - Detect message types
    - Infer field boundaries
    """

    def __init__(self, llm_service: Any, config: Optional[AgentConfig] = None):
        config = config or AgentConfig(
            role=AgentRole.PROTOCOL_ANALYST,
            name="Protocol Analyst",
            description=(
                "Expert in analyzing network protocols and message formats. "
                "Specializes in reverse engineering protocol structures, "
                "identifying field boundaries, detecting encoding types, "
                "and inferring message semantics from traffic samples."
            ),
            model="gpt-4o",
            temperature=0.3,  # Lower temperature for analytical tasks
            max_tokens=4096,
            max_iterations=15,
            enable_reflection=True,
        )

        # Initialize tools
        tools = [
            TrafficAnalysisTool(),
            PatternRecognitionTool(),
        ]

        super().__init__(config, llm_service, tools)

        # Protocol analyst specific state
        self.analyzed_protocols: Dict[str, Dict[str, Any]] = {}

    @property
    def system_prompt(self) -> str:
        """Protocol analyst specific system prompt."""
        base_prompt = super().system_prompt

        analyst_prompt = """

## Protocol Analysis Expertise

As a Protocol Analyst, you have deep expertise in:

1. **Traffic Pattern Analysis**
   - Statistical analysis of byte distributions
   - Identification of fixed vs variable fields
   - Detection of delimiters and boundaries
   - Recognition of common protocol signatures

2. **Encoding Detection**
   - ASCII, UTF-8, EBCDIC identification
   - Binary protocol characteristics
   - Encryption/compression detection via entropy

3. **Structure Inference**
   - Header/body/trailer identification
   - Field boundary detection
   - Message type classification
   - State machine inference

4. **Protocol Families**
   - Text-based (HTTP, FTP, SMTP)
   - Binary (ISO 8583, ASN.1)
   - Financial (FIX, SWIFT MT)
   - Legacy mainframe formats

## Analysis Methodology

Follow this systematic approach:

1. **Initial Assessment**: Analyze sample statistics (lengths, distributions)
2. **Pattern Detection**: Look for recurring patterns and signatures
3. **Structure Mapping**: Identify field boundaries and types
4. **Semantic Analysis**: Infer meaning of fields
5. **Validation**: Cross-check findings across samples

## Output Format

When reporting analysis results, structure them as:

```
PROTOCOL ANALYSIS REPORT
========================

Protocol Type: [Identified/Unknown]
Encoding: [ASCII/Binary/EBCDIC/etc.]
Structure: [Fixed-length/Variable/Mixed]
Confidence: [High/Medium/Low]

Fields Identified:
- Position 0-4: [Field name/purpose]
- Position 5-8: [Field name/purpose]
...

Message Types:
- Type 1: [Description]
- Type 2: [Description]

Recommendations:
- [Next steps for further analysis]
```
"""
        return base_prompt + analyst_prompt

    async def analyze_traffic(
        self,
        traffic_samples: List[bytes],
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze protocol traffic samples.

        Args:
            traffic_samples: List of protocol message bytes
            context: Additional context about the system

        Returns:
            Analysis results including patterns, fields, and structure
        """
        # Convert samples to base64 for tool
        import base64
        encoded_samples = [base64.b64encode(s).decode() for s in traffic_samples]

        # Create task message
        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Analyze the following {len(traffic_samples)} protocol traffic samples.

Context: {context if context else 'No additional context provided.'}

Your task:
1. Use the analyze_traffic tool to get statistical analysis
2. Use the recognize_patterns tool to identify known protocols
3. Synthesize findings into a comprehensive analysis

Focus on:
- Protocol type identification
- Encoding detection
- Field structure
- Message types
- Any security concerns
""",
            data={
                "samples": encoded_samples,
                "sample_count": len(traffic_samples)
            }
        )

        # Process message using agentic loop
        response = await self.process_message(message)

        # Store results
        analysis_id = f"analysis_{len(self.analyzed_protocols)}"
        self.analyzed_protocols[analysis_id] = {
            "samples_count": len(traffic_samples),
            "context": context,
            "result": response.data
        }

        return response.data

    async def identify_fields(
        self,
        samples: List[bytes],
        hints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify protocol fields from samples.

        Args:
            samples: Protocol message samples
            hints: Optional hints about expected fields

        Returns:
            List of identified fields with positions and types
        """
        import base64
        encoded_samples = [base64.b64encode(s).decode() for s in samples]

        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Identify the field structure in these {len(samples)} protocol samples.

Hints: {hints if hints else 'None provided.'}

Your task:
1. Analyze traffic to find field boundaries
2. Determine field types (numeric, string, binary, etc.)
3. Infer field purposes based on patterns

For each field, provide:
- Position (start-end bytes)
- Type
- Likely purpose
- Whether it's fixed or variable length
""",
            data={
                "samples": encoded_samples,
                "hints": hints or {}
            }
        )

        response = await self.process_message(message)
        return response.data.get("fields", [])

    async def classify_message_types(
        self,
        samples: List[bytes],
        field_info: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, List[int]]:
        """
        Classify samples into message types.

        Args:
            samples: Protocol message samples
            field_info: Previously identified field information

        Returns:
            Dictionary mapping message type to sample indices
        """
        import base64
        encoded_samples = [base64.b64encode(s).decode() for s in samples]

        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Classify these {len(samples)} protocol messages into distinct types.

Field information: {field_info if field_info else 'Not yet identified.'}

Your task:
1. Analyze samples to identify distinguishing characteristics
2. Group samples by message type
3. Describe each message type

Provide:
- Number of distinct message types found
- Characteristics of each type
- Which samples belong to which type
""",
            data={
                "samples": encoded_samples,
                "field_info": field_info
            }
        )

        response = await self.process_message(message)
        return response.data.get("message_types", {})


# Factory function
def create_protocol_analyst(llm_service: Any) -> ProtocolAnalystAgent:
    """Create a Protocol Analyst agent."""
    return ProtocolAnalystAgent(llm_service)


__all__ = ["ProtocolAnalystAgent", "create_protocol_analyst"]
