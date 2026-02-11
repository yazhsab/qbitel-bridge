"""
QBITEL - Code Generator Agent

Specialized agent for generating protocol adapter code.
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
from .tools import CodeGenerationTool, TestGenerationTool


logger = logging.getLogger(__name__)


class CodeGeneratorAgent(BaseAgent):
    """
    Agent specialized in code generation.

    Responsibilities:
    - Generate protocol adapters
    - Create test suites
    - Produce integration code
    - Generate deployment configurations
    """

    def __init__(self, llm_service: Any, config: Optional[AgentConfig] = None):
        config = config or AgentConfig(
            role=AgentRole.CODE_GENERATOR,
            name="Code Generation Specialist",
            description=(
                "Expert software engineer specializing in protocol adapters "
                "and integration code. Generates production-quality code with "
                "comprehensive tests and documentation."
            ),
            model="gpt-4o",
            temperature=0.2,  # Lower for consistent code generation
            max_tokens=8192,  # Larger for code
            max_iterations=15,
            enable_reflection=True,
        )

        # Initialize tools
        tools = [
            CodeGenerationTool(llm_service),
            TestGenerationTool(),
        ]

        super().__init__(config, llm_service, tools)

        self.generated_code: Dict[str, Dict[str, str]] = {}

    @property
    def system_prompt(self) -> str:
        """Code generator specific system prompt."""
        base_prompt = super().system_prompt

        code_prompt = """

## Code Generation Expertise

As a Code Generation Specialist, you produce:

1. **Protocol Adapters**
   - Message parsing and serialization
   - Field mapping and transformation
   - Error handling and validation
   - Performance-optimized implementations

2. **Test Suites**
   - Unit tests with high coverage
   - Integration tests
   - Edge case testing
   - Performance benchmarks

3. **Supporting Code**
   - Data models and types
   - Utility functions
   - Configuration management
   - Logging and metrics

4. **Deployment Artifacts**
   - Dockerfiles
   - Kubernetes manifests
   - CI/CD configurations
   - Environment configs

## Code Quality Standards

All generated code must:

1. **Be Production-Ready**
   - No TODO comments in final code
   - Complete error handling
   - Proper logging
   - Type hints (Python) / Types (TypeScript) / Generics (Java, Go)

2. **Follow Best Practices**
   - Clean code principles
   - SOLID principles
   - Language-specific conventions
   - Security best practices

3. **Be Testable**
   - Dependency injection
   - Mockable interfaces
   - Clear separation of concerns

4. **Be Documented**
   - Docstrings/Javadoc
   - Inline comments for complex logic
   - Usage examples

## Language-Specific Guidelines

### Python
- Use dataclasses for models
- Async/await for I/O operations
- Type hints throughout
- Follow PEP 8

### Java
- Use modern Java features (streams, optionals)
- Follow Maven/Gradle conventions
- Use appropriate design patterns
- Lombok for boilerplate reduction

### Go
- Follow Go idioms (error returns, interfaces)
- Use standard library where possible
- Proper error wrapping
- Context for cancellation

### TypeScript
- Strict mode
- Interface-based typing
- Modern ES features
- Proper async handling

## Output Format

When generating code, structure output as:

```
=== FILE: [filename] ===
[complete code content]

=== FILE: [test_filename] ===
[complete test content]

=== DEPENDENCIES ===
[list of required dependencies]

=== USAGE ===
[how to use the generated code]
```
"""
        return base_prompt + code_prompt

    async def generate_adapter(
        self,
        protocol_spec: Dict[str, Any],
        target_protocol: str,
        language: str = "python"
    ) -> Dict[str, str]:
        """
        Generate a protocol adapter.

        Args:
            protocol_spec: Source protocol specification
            target_protocol: Target protocol (REST, gRPC, etc.)
            language: Programming language

        Returns:
            Dictionary of filename to code content
        """
        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Generate a complete {language} adapter from legacy protocol to {target_protocol}.

Protocol Specification:
{self._format_spec(protocol_spec)}

Target: {target_protocol}
Language: {language}

Your task:
1. Use generate_code tool to create adapter structure
2. Use generate_tests tool to create test suite
3. Add error handling and logging
4. Include configuration support

Generate:
- Main adapter class
- Data models
- Utility functions
- Comprehensive tests
- Configuration template
""",
            data={
                "protocol_spec": protocol_spec,
                "target_protocol": target_protocol,
                "language": language
            }
        )

        response = await self.process_message(message)

        # Store generated code
        code_id = f"{protocol_spec.get('protocol_name', 'unknown')}_{target_protocol}_{language}"
        self.generated_code[code_id] = response.data

        return response.data

    async def generate_tests(
        self,
        code: Dict[str, str],
        protocol_spec: Dict[str, Any],
        language: str = "python"
    ) -> Dict[str, str]:
        """
        Generate tests for adapter code.

        Args:
            code: Generated adapter code
            protocol_spec: Protocol specification
            language: Programming language

        Returns:
            Dictionary of test filename to test code
        """
        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Generate comprehensive tests for this adapter code.

Adapter Code:
{self._format_code(code)}

Protocol Specification:
{self._format_spec(protocol_spec)}

Language: {language}

Your task:
1. Use generate_tests tool with all test types
2. Create edge case tests based on protocol spec
3. Add performance tests
4. Include integration test scaffolding

Tests should cover:
- All message types
- All transformation paths
- Error conditions
- Edge cases (empty, malformed, oversized)
- Performance under load
""",
            data={
                "code": code,
                "protocol_spec": protocol_spec,
                "language": language
            }
        )

        response = await self.process_message(message)
        return response.data.get("tests", {})

    async def generate_integration_guide(
        self,
        adapter_code: Dict[str, str],
        protocol_spec: Dict[str, Any],
        target_protocol: str
    ) -> str:
        """
        Generate integration guide for the adapter.

        Args:
            adapter_code: Generated adapter code
            protocol_spec: Protocol specification
            target_protocol: Target protocol

        Returns:
            Markdown integration guide
        """
        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Generate an integration guide for this protocol adapter.

Protocol: {protocol_spec.get('protocol_name', 'Unknown')}
Target: {target_protocol}

Code Files:
{', '.join(adapter_code.keys())}

Create a guide covering:
1. Prerequisites and dependencies
2. Installation steps
3. Configuration options
4. Basic usage examples
5. Advanced usage patterns
6. Troubleshooting common issues
7. Performance tuning
8. Monitoring and observability
""",
            data={
                "adapter_code": adapter_code,
                "protocol_spec": protocol_spec,
                "target_protocol": target_protocol
            }
        )

        response = await self.process_message(message)
        return response.content

    async def generate_deployment_config(
        self,
        adapter_code: Dict[str, str],
        language: str,
        platform: str = "kubernetes"
    ) -> Dict[str, str]:
        """
        Generate deployment configurations.

        Args:
            adapter_code: Generated adapter code
            language: Programming language
            platform: Deployment platform (kubernetes, docker, aws)

        Returns:
            Dictionary of config filename to content
        """
        message = AgentMessage(
            message_type=MessageType.TASK,
            sender=AgentRole.ORCHESTRATOR,
            recipient=self.config.role,
            content=f"""Generate deployment configuration for this adapter.

Language: {language}
Platform: {platform}

Code Files:
{', '.join(adapter_code.keys())}

Generate:
- Dockerfile with multi-stage build
- {platform} deployment manifests
- ConfigMap/Secret templates
- Service definitions
- Health check configurations
- Resource limits recommendations
""",
            data={
                "adapter_code": adapter_code,
                "language": language,
                "platform": platform
            }
        )

        response = await self.process_message(message)
        return response.data.get("configs", {})

    def _format_spec(self, spec: Dict[str, Any]) -> str:
        """Format protocol spec for prompt."""
        lines = []
        for key, value in spec.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value[:5]:
                    lines.append(f"  - {item}")
            elif isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in list(value.items())[:5]:
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _format_code(self, code: Dict[str, str]) -> str:
        """Format code for prompt."""
        lines = []
        for filename, content in code.items():
            lines.append(f"\n=== {filename} ===")
            lines.append(content[:2000])  # Truncate for prompt
            if len(content) > 2000:
                lines.append("... (truncated)")
        return "\n".join(lines)


# Factory function
def create_code_generator(llm_service: Any) -> CodeGeneratorAgent:
    """Create a Code Generator agent."""
    return CodeGeneratorAgent(llm_service)


__all__ = ["CodeGeneratorAgent", "create_code_generator"]
