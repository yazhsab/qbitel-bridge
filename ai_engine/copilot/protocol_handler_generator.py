"""
QBITEL - Custom Protocol Handler Code Generation

Generates production-ready protocol handler code using LLM-powered
code generation with security best practices.
"""

import asyncio
import logging
import time
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from prometheus_client import Counter, Histogram

from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest, get_llm_service
from ..core.config import Config
from ..core.exceptions import QbitelAIException


# Prometheus metrics
CODE_GENERATION_COUNTER = Counter(
    "qbitel_code_generation_total",
    "Total code generations",
    ["language", "component_type"],
    registry=None,
)

CODE_GENERATION_DURATION = Histogram(
    "qbitel_code_generation_duration_seconds",
    "Code generation duration",
    registry=None,
)


logger = logging.getLogger(__name__)


class ProgrammingLanguage(str, Enum):
    """Supported programming languages for code generation."""

    PYTHON = "python"
    RUST = "rust"
    GO = "go"
    C = "c"
    CPP = "cpp"


class ComponentType(str, Enum):
    """Types of protocol handler components."""

    PARSER = "parser"  # Message parsing
    SERIALIZER = "serializer"  # Message serialization
    STATE_MACHINE = "state_machine"  # Protocol state machine
    VALIDATOR = "validator"  # Input validation
    HANDLER = "handler"  # Complete handler
    TEST = "test"  # Unit tests


@dataclass
class ProtocolField:
    """Protocol message field definition."""

    name: str
    field_type: str  # int, string, bytes, etc.
    length: Optional[int]  # Fixed length or None for variable
    required: bool
    default_value: Optional[Any] = None
    description: str = ""
    validation_rules: List[str] = field(default_factory=list)


@dataclass
class ProtocolMessage:
    """Protocol message definition."""

    message_name: str
    message_type: int
    fields: List[ProtocolField]
    description: str = ""


@dataclass
class ProtocolSpec:
    """Complete protocol specification."""

    protocol_name: str
    protocol_version: str
    description: str
    messages: List[ProtocolMessage]
    endianness: str = "big"  # big or little
    encoding: str = "utf-8"
    security_requirements: List[str] = field(default_factory=list)


@dataclass
class GeneratedCode:
    """Generated code artifact."""

    artifact_id: str
    component_type: ComponentType
    language: ProgrammingLanguage
    source_code: str
    file_name: str
    dependencies: List[str]
    test_code: Optional[str] = None
    documentation: str = ""
    security_notes: List[str] = field(default_factory=list)
    llm_rationale: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "component_type": self.component_type.value,
            "language": self.language.value,
            "source_code": self.source_code,
            "file_name": self.file_name,
            "dependencies": self.dependencies,
            "test_code": self.test_code,
            "documentation": self.documentation,
            "security_notes": self.security_notes,
            "llm_rationale": self.llm_rationale,
            "timestamp": self.timestamp.isoformat(),
        }


class ProtocolHandlerGenerator:
    """
    Generates custom protocol handler code using LLM.

    Produces production-ready, secure code with comprehensive error handling,
    validation, and testing.
    """

    def __init__(self, config: Config, llm_service: Optional[UnifiedLLMService] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_service = llm_service or get_llm_service(config)

    async def generate_handler(
        self,
        protocol_spec: ProtocolSpec,
        language: ProgrammingLanguage,
        component_type: ComponentType = ComponentType.HANDLER,
    ) -> GeneratedCode:
        """
        Generate protocol handler code.

        Args:
            protocol_spec: Protocol specification
            language: Target programming language
            component_type: Type of component to generate

        Returns:
            Generated code artifact
        """
        start_time = time.time()

        try:
            self.logger.info(
                f"Generating {component_type.value} in {language.value} "
                f"for protocol {protocol_spec.protocol_name}"
            )

            # Generate code using LLM
            source_code, dependencies, security_notes, rationale = (
                await self._generate_code_llm(protocol_spec, language, component_type)
            )

            # Generate tests
            test_code = await self._generate_tests(
                protocol_spec, language, component_type, source_code
            )

            # Generate documentation
            documentation = self._generate_documentation(
                protocol_spec, component_type, source_code
            )

            # Determine file name
            file_name = self._determine_file_name(
                protocol_spec.protocol_name, component_type, language
            )

            # Create artifact
            artifact_id = f"gen_{protocol_spec.protocol_name}_{component_type.value}_{int(time.time())}"
            artifact = GeneratedCode(
                artifact_id=artifact_id,
                component_type=component_type,
                language=language,
                source_code=source_code,
                file_name=file_name,
                dependencies=dependencies,
                test_code=test_code,
                documentation=documentation,
                security_notes=security_notes,
                llm_rationale=rationale,
            )

            # Metrics
            CODE_GENERATION_COUNTER.labels(
                language=language.value,
                component_type=component_type.value,
            ).inc()
            CODE_GENERATION_DURATION.observe(time.time() - start_time)

            self.logger.info(
                f"Code generation completed: {artifact_id}, "
                f"lines={len(source_code.splitlines())}"
            )

            return artifact

        except Exception as e:
            self.logger.error(f"Code generation error: {e}", exc_info=True)
            raise QbitelAIException(f"Code generation failed: {e}")

    async def _generate_code_llm(
        self,
        protocol_spec: ProtocolSpec,
        language: ProgrammingLanguage,
        component_type: ComponentType,
    ) -> Tuple[str, List[str], List[str], str]:
        """Generate code using LLM."""
        prompt = self._build_code_generation_prompt(
            protocol_spec, language, component_type
        )

        llm_request = LLMRequest(
            prompt=prompt,
            feature_domain="code_generation",
            context={"protocol": protocol_spec.protocol_name},
            max_tokens=3500,
            temperature=0.2,  # Low temperature for consistent, reliable code
        )

        response = await self.llm_service.query(llm_request)

        # Parse LLM response
        source_code, dependencies, security_notes = self._parse_code_response(
            response.content, language
        )

        return source_code, dependencies, security_notes, response.content[:500]

    def _build_code_generation_prompt(
        self,
        protocol_spec: ProtocolSpec,
        language: ProgrammingLanguage,
        component_type: ComponentType,
    ) -> str:
        """Build prompt for code generation."""
        messages_desc = "\n".join(
            [
                f"- {msg.message_name} (type={msg.message_type}): {msg.description}\n  Fields: "
                + ", ".join([f"{f.name}:{f.field_type}" for f in msg.fields])
                for msg in protocol_spec.messages
            ]
        )

        security_requirements = (
            "\n".join([f"- {req}" for req in protocol_spec.security_requirements])
            if protocol_spec.security_requirements
            else "- Input validation\n- Error handling\n- Bounds checking"
        )

        return f"""
Generate production-ready {component_type.value} code in {language.value} for the following protocol.

**Protocol Specification:**
Name: {protocol_spec.protocol_name}
Version: {protocol_spec.protocol_version}
Description: {protocol_spec.description}
Endianness: {protocol_spec.endianness}
Encoding: {protocol_spec.encoding}

**Messages:**
{messages_desc}

**Security Requirements:**
{security_requirements}

**Requirements:**
1. Write clean, idiomatic {language.value} code
2. Include comprehensive error handling
3. Implement input validation and bounds checking
4. Add inline documentation/comments
5. Follow security best practices (avoid buffer overflows, injection, etc.)
6. Make the code production-ready (no TODOs or placeholders)
7. Include logging where appropriate

**Component Type:** {component_type.value}

For PARSER: Implement message parsing from bytes to structured data
For SERIALIZER: Implement serialization from structured data to bytes
For VALIDATOR: Implement input validation logic
For HANDLER: Implement complete handler with parse, validate, and process

**Output Format:**
Provide your response as:
{{
  "source_code": "complete source code here",
  "dependencies": ["dependency1", "dependency2"],
  "security_notes": ["note1", "note2"]
}}

Ensure the source code is complete and ready to use.
"""

    def _parse_code_response(
        self, llm_response: str, language: ProgrammingLanguage
    ) -> Tuple[str, List[str], List[str]]:
        """Parse LLM response to extract code."""
        try:
            # Try to extract JSON
            start_idx = llm_response.find("{")
            end_idx = llm_response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                data = json.loads(json_str)

                source_code = data.get("source_code", "")
                dependencies = data.get("dependencies", [])
                security_notes = data.get("security_notes", [])

                # Clean up code (remove markdown code blocks if present)
                source_code = self._clean_code_block(source_code, language)

                return source_code, dependencies, security_notes

        except Exception as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")

        # Fallback: extract code blocks from markdown
        code_blocks = self._extract_code_blocks(llm_response, language)
        if code_blocks:
            return code_blocks[0], [], ["Extracted from markdown code block"]

        # Last resort: return entire response
        return llm_response, [], ["Raw LLM response"]

    def _clean_code_block(self, code: str, language: ProgrammingLanguage) -> str:
        """Clean code block by removing markdown formatting."""
        # Remove markdown code fences
        code = re.sub(r"^```\w*\n", "", code)
        code = re.sub(r"\n```$", "", code)

        return code.strip()

    def _extract_code_blocks(
        self, text: str, language: ProgrammingLanguage
    ) -> List[str]:
        """Extract code blocks from markdown."""
        lang_markers = {
            ProgrammingLanguage.PYTHON: ["python", "py"],
            ProgrammingLanguage.RUST: ["rust", "rs"],
            ProgrammingLanguage.GO: ["go", "golang"],
            ProgrammingLanguage.C: ["c"],
            ProgrammingLanguage.CPP: ["cpp", "c++"],
        }

        markers = lang_markers.get(language, [language.value])
        pattern = r"```(?:" + "|".join(markers) + r")?\n(.*?)```"

        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]

    async def _generate_tests(
        self,
        protocol_spec: ProtocolSpec,
        language: ProgrammingLanguage,
        component_type: ComponentType,
        source_code: str,
    ) -> str:
        """Generate unit tests for the generated code."""
        prompt = f"""
Generate comprehensive unit tests for the following {language.value} {component_type.value} code.

**Protocol:** {protocol_spec.protocol_name}

**Source Code:**
```{language.value}
{source_code[:1000]}
```

**Requirements:**
1. Test normal/valid cases
2. Test edge cases (empty input, max values, etc.)
3. Test error cases (invalid input, malformed data)
4. Use appropriate testing framework ({self._get_test_framework(language)})
5. Include test data fixtures
6. Aim for >80% code coverage

Provide only the test code, ready to run.
"""

        try:
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="code_generation",
                max_tokens=2000,
                temperature=0.2,
            )

            response = await self.llm_service.query(llm_request)

            # Extract test code
            code_blocks = self._extract_code_blocks(response.content, language)
            if code_blocks:
                return code_blocks[0]

            return response.content

        except Exception as e:
            self.logger.warning(f"Test generation failed: {e}")
            return f"# TODO: Add tests for {component_type.value}"

    def _get_test_framework(self, language: ProgrammingLanguage) -> str:
        """Get recommended test framework for language."""
        frameworks = {
            ProgrammingLanguage.PYTHON: "pytest",
            ProgrammingLanguage.RUST: "built-in #[test]",
            ProgrammingLanguage.GO: "testing package",
            ProgrammingLanguage.C: "CUnit or Check",
            ProgrammingLanguage.CPP: "Google Test",
        }
        return frameworks.get(language, "standard testing framework")

    def _generate_documentation(
        self,
        protocol_spec: ProtocolSpec,
        component_type: ComponentType,
        source_code: str,
    ) -> str:
        """Generate documentation for the generated code."""
        lines_of_code = len(source_code.splitlines())

        doc = f"""
# {protocol_spec.protocol_name} {component_type.value.title()}

## Overview
This {component_type.value} implements support for the {protocol_spec.protocol_name} protocol (version {protocol_spec.protocol_version}).

**Description:** {protocol_spec.description}

## Component Details
- **Type:** {component_type.value}
- **Lines of Code:** {lines_of_code}
- **Messages Supported:** {len(protocol_spec.messages)}

## Messages
{self._format_messages_doc(protocol_spec.messages)}

## Usage
See the source code comments and test cases for usage examples.

## Security Considerations
- Input validation is implemented for all fields
- Bounds checking is performed on all buffer operations
- Error handling follows best practices

## Testing
Comprehensive unit tests are provided in the accompanying test file.

## Generated
This code was automatically generated by QBITEL Protocol Handler Generator.
"""
        return doc.strip()

    def _format_messages_doc(self, messages: List[ProtocolMessage]) -> str:
        """Format message documentation."""
        doc_lines = []
        for msg in messages:
            doc_lines.append(f"### {msg.message_name} (Type: {msg.message_type})")
            doc_lines.append(f"{msg.description}\n")
            doc_lines.append("**Fields:**")
            for field in msg.fields:
                required_str = "required" if field.required else "optional"
                doc_lines.append(
                    f"- `{field.name}` ({field.field_type}, {required_str}): {field.description}"
                )
            doc_lines.append("")

        return "\n".join(doc_lines)

    def _determine_file_name(
        self,
        protocol_name: str,
        component_type: ComponentType,
        language: ProgrammingLanguage,
    ) -> str:
        """Determine appropriate file name for generated code."""
        base_name = f"{protocol_name.lower()}_{component_type.value}"

        extensions = {
            ProgrammingLanguage.PYTHON: ".py",
            ProgrammingLanguage.RUST: ".rs",
            ProgrammingLanguage.GO: ".go",
            ProgrammingLanguage.C: ".c",
            ProgrammingLanguage.CPP: ".cpp",
        }

        ext = extensions.get(language, ".txt")
        return base_name + ext


# Factory function
def get_protocol_handler_generator(
    config: Config, llm_service: Optional[UnifiedLLMService] = None
) -> ProtocolHandlerGenerator:
    """Factory function to get ProtocolHandlerGenerator instance."""
    return ProtocolHandlerGenerator(config, llm_service)
