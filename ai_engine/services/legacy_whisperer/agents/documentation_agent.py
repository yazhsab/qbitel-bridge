"""
QBITEL - Documentation Agent

Specialized agent for generating protocol documentation and explanations.
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
from .tools import DocumentationSearchTool

logger = logging.getLogger(__name__)


class DocumentationAgent(BaseAgent):
    """
    Agent specialized in documentation generation.

    Responsibilities:
    - Generate protocol documentation
    - Search existing documentation
    - Create usage examples
    - Explain legacy behavior
    """

    def __init__(self, llm_service: Any, rag_engine: Any = None, config: Optional[AgentConfig] = None):
        config = config or AgentConfig(
            role=AgentRole.DOCUMENTATION,
            name="Documentation Specialist",
            description=(
                "Expert technical writer specializing in legacy system documentation. "
                "Creates comprehensive, accurate documentation from analysis results, "
                "explains complex behaviors clearly, and produces usage examples."
            ),
            model="gpt-4o",
            temperature=0.5,  # Balanced for creative but accurate writing
            max_tokens=8192,  # Longer for documentation
            max_iterations=10,
            enable_reflection=True,
        )

        # Initialize tools
        tools = [
            DocumentationSearchTool(rag_engine),
        ]

        super().__init__(config, llm_service, tools)

        self.rag_engine = rag_engine
        self.generated_docs: Dict[str, str] = {}

    @property
    def system_prompt(self) -> str:
        """Documentation agent specific system prompt."""
        base_prompt = super().system_prompt

        doc_prompt = """

## Documentation Expertise

As a Documentation Specialist, you excel at:

1. **Technical Writing**
   - Clear, precise explanations of complex protocols
   - Structured documentation with proper sections
   - Appropriate use of examples and diagrams (described textually)
   - Audience-appropriate language

2. **Protocol Documentation**
   - Message format specifications
   - Field definitions and constraints
   - State machine descriptions
   - Error handling documentation

3. **Historical Context**
   - Understanding legacy system constraints
   - Explaining why certain design decisions were made
   - Connecting old and new technology concepts

4. **Usage Examples**
   - Code snippets in multiple languages
   - Request/response examples
   - Edge case handling
   - Best practices

## Documentation Structure

Use this template for protocol documentation:

```markdown
# [Protocol Name] Documentation

## Overview
Brief description of the protocol and its purpose.

## Message Format
### Header
- Field definitions
- Encoding information

### Body
- Message types
- Field specifications

### Trailer
- Checksums
- Terminators

## Message Types
### Type 1: [Name]
Description and examples

### Type 2: [Name]
Description and examples

## Usage Examples
### Sending a Request
```code
example code
```

### Parsing a Response
```code
example code
```

## Error Handling
Common errors and how to handle them.

## Security Considerations
Known vulnerabilities and mitigations.

## References
Related documentation and standards.
```

## Writing Guidelines

1. **Be Precise**: Use exact byte positions and values
2. **Be Complete**: Document all fields and message types
3. **Be Practical**: Include working examples
4. **Be Aware**: Note security concerns and gotchas
"""
        return base_prompt + doc_prompt

    async def generate_documentation(
        self, analysis_results: Dict[str, Any], protocol_name: str = "Unknown Protocol", context: str = ""
    ) -> str:
        """
        Generate comprehensive protocol documentation.

        Args:
            analysis_results: Results from protocol analysis
            protocol_name: Name of the protocol
            context: Additional context

        Returns:
            Markdown documentation
        """
        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Generate comprehensive documentation for the "{protocol_name}" protocol.

Analysis Results:
{self._format_analysis(analysis_results)}

Context: {context if context else 'Enterprise legacy system.'}

Your task:
1. Search for any existing documentation about similar protocols
2. Create comprehensive documentation following the standard template
3. Include practical usage examples
4. Note any security concerns

The documentation should be:
- Complete and accurate
- Well-structured
- Suitable for both developers and architects
- Include code examples in Python and Java
""",
            data={"analysis": analysis_results, "protocol_name": protocol_name},
        )

        response = await self.process_message(message)

        # Extract documentation from response
        documentation = response.data.get("documentation", response.content)

        # Store generated docs
        self.generated_docs[protocol_name] = documentation

        return documentation

    async def explain_behavior(self, behavior: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain legacy system behavior.

        Args:
            behavior: Description of the behavior to explain
            context: Contextual information

        Returns:
            Explanation with historical context and recommendations
        """
        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Explain the following legacy system behavior:

Behavior: {behavior}

Context:
{self._format_context(context)}

Your task:
1. Search documentation for relevant information
2. Explain why this behavior exists (historical reasons)
3. Describe technical implementation details
4. Suggest modern alternatives

Provide:
- Technical explanation
- Historical context (why was it done this way?)
- Impact on modernization
- Recommended approach for handling
""",
            data={"behavior": behavior, "context": context},
        )

        response = await self.process_message(message)

        return {"explanation": response.content, "data": response.data}

    async def create_usage_examples(self, protocol_spec: Dict[str, Any], languages: List[str] = None) -> Dict[str, str]:
        """
        Create code examples for using the protocol.

        Args:
            protocol_spec: Protocol specification
            languages: Target programming languages

        Returns:
            Dictionary of language to code examples
        """
        languages = languages or ["python", "java"]

        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Create usage examples for this protocol specification.

Protocol Specification:
{self._format_spec(protocol_spec)}

Languages: {', '.join(languages)}

For each language, provide:
1. How to parse incoming messages
2. How to construct outgoing messages
3. Common operations
4. Error handling

Examples should be:
- Complete and runnable
- Well-commented
- Following language best practices
""",
            data={"spec": protocol_spec, "languages": languages},
        )

        response = await self.process_message(message)

        return response.data.get("examples", {})

    def _format_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format analysis results for prompt."""
        lines = []
        for key, value in analysis.items():
            if isinstance(value, dict):
                lines.append(f"\n{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"\n{key}:")
                for item in value[:10]:  # Limit for prompt
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt."""
        return "\n".join(f"- {k}: {v}" for k, v in context.items())

    def _format_spec(self, spec: Dict[str, Any]) -> str:
        """Format protocol spec for prompt."""
        return self._format_analysis(spec)


# Factory function
def create_documentation_agent(llm_service: Any, rag_engine: Any = None) -> DocumentationAgent:
    """Create a Documentation agent."""
    return DocumentationAgent(llm_service, rag_engine)


__all__ = ["DocumentationAgent", "create_documentation_agent"]
