"""
CRONOS AI - Parser Improvement Engine
Provides intelligent suggestions for parser improvements based on errors and analysis.
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter

from ..core.exceptions import CronosAIException
from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest
from ..llm.rag_engine import RAGEngine

logger = logging.getLogger(__name__)


@dataclass
class ParserError:
    """Parser error structure."""

    error_type: str
    error_message: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    context: Optional[str] = None
    severity: str = "error"  # error, warning, info
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ParserImprovement:
    """Parser improvement suggestion."""

    improvement_type: str
    title: str
    description: str
    code_example: Optional[str] = None
    priority: str = "medium"  # low, medium, high, critical
    estimated_impact: str = "moderate"
    implementation_steps: List[str] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)
    confidence: float = 0.8


@dataclass
class ParserAnalysis:
    """Complete parser analysis result."""

    parser_id: str
    analysis_timestamp: datetime
    errors: List[ParserError]
    improvements: List[ParserImprovement]
    overall_health: str  # excellent, good, fair, poor
    complexity_score: float
    maintainability_score: float
    performance_score: float
    recommendations_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParserImprovementEngine:
    """
    Intelligent parser improvement suggestion engine.

    Analyzes parser errors, code quality, and performance to provide
    actionable improvement suggestions with code examples.
    """

    def __init__(self, llm_service: UnifiedLLMService, rag_engine: RAGEngine):
        self.llm_service = llm_service
        self.rag_engine = rag_engine
        self.logger = logging.getLogger(__name__)

        # Error pattern recognition
        self.error_patterns = {
            "syntax_error": r"(syntax error|unexpected token|invalid syntax)",
            "type_error": r"(type error|type mismatch|cannot convert)",
            "boundary_error": r"(out of bounds|index error|overflow)",
            "encoding_error": r"(encoding|decode|unicode)",
            "validation_error": r"(validation failed|invalid value|constraint)",
            "performance_issue": r"(timeout|slow|performance|bottleneck)",
        }

        # Improvement templates
        self.improvement_templates = {
            "error_handling": {
                "title": "Enhance Error Handling",
                "priority": "high",
                "impact": "significant",
            },
            "validation": {
                "title": "Add Input Validation",
                "priority": "high",
                "impact": "significant",
            },
            "performance": {
                "title": "Optimize Performance",
                "priority": "medium",
                "impact": "moderate",
            },
            "maintainability": {
                "title": "Improve Code Maintainability",
                "priority": "medium",
                "impact": "moderate",
            },
            "documentation": {
                "title": "Add Documentation",
                "priority": "low",
                "impact": "minor",
            },
        }

    async def analyze_parser(
        self,
        parser_code: str,
        errors: List[ParserError],
        parser_metadata: Dict[str, Any] = None,
    ) -> ParserAnalysis:
        """
        Analyze parser and generate improvement suggestions.

        Args:
            parser_code: Parser source code
            errors: List of parser errors
            parser_metadata: Additional parser metadata

        Returns:
            Complete parser analysis with improvements
        """
        try:
            self.logger.info("Starting parser analysis...")

            # Classify errors
            error_categories = self._classify_errors(errors)

            # Analyze code quality
            quality_metrics = self._analyze_code_quality(parser_code)

            # Generate improvements using LLM
            improvements = await self._generate_improvements(
                parser_code, errors, error_categories, quality_metrics
            )

            # Calculate overall health
            overall_health = self._calculate_health_score(
                errors, quality_metrics, improvements
            )

            # Generate summary
            summary = await self._generate_recommendations_summary(
                improvements, overall_health, quality_metrics
            )

            return ParserAnalysis(
                parser_id=(
                    parser_metadata.get("parser_id", "unknown")
                    if parser_metadata
                    else "unknown"
                ),
                analysis_timestamp=datetime.now(),
                errors=errors,
                improvements=improvements,
                overall_health=overall_health,
                complexity_score=quality_metrics["complexity"],
                maintainability_score=quality_metrics["maintainability"],
                performance_score=quality_metrics["performance"],
                recommendations_summary=summary,
                metadata={
                    "error_categories": error_categories,
                    "total_errors": len(errors),
                    "total_improvements": len(improvements),
                    **(parser_metadata or {}),
                },
            )

        except Exception as e:
            self.logger.error(f"Parser analysis failed: {e}")
            raise CronosAIException(f"Parser analysis failed: {e}")

    async def suggest_parser_improvements(
        self, parser_code: str, errors: List[str], context: Dict[str, Any] = None
    ) -> List[ParserImprovement]:
        """
        Generate parser improvement suggestions based on errors.

        Args:
            parser_code: Parser source code
            errors: List of error messages
            context: Additional context

        Returns:
            List of improvement suggestions
        """
        try:
            # Convert error strings to ParserError objects
            parser_errors = [
                ParserError(
                    error_type=self._detect_error_type(error),
                    error_message=error,
                    severity="error",
                )
                for error in errors
            ]

            # Analyze and get improvements
            analysis = await self.analyze_parser(parser_code, parser_errors, context)

            return analysis.improvements

        except Exception as e:
            self.logger.error(f"Failed to suggest improvements: {e}")
            return []

    def _classify_errors(
        self, errors: List[ParserError]
    ) -> Dict[str, List[ParserError]]:
        """Classify errors by type."""
        categories = defaultdict(list)

        for error in errors:
            error_type = error.error_type or self._detect_error_type(
                error.error_message
            )
            categories[error_type].append(error)

        return dict(categories)

    def _detect_error_type(self, error_message: str) -> str:
        """Detect error type from message."""
        error_lower = error_message.lower()

        for error_type, pattern in self.error_patterns.items():
            if re.search(pattern, error_lower):
                return error_type

        return "unknown"

    def _analyze_code_quality(self, parser_code: str) -> Dict[str, float]:
        """Analyze parser code quality metrics."""
        metrics = {
            "complexity": 0.0,
            "maintainability": 0.0,
            "performance": 0.0,
            "documentation": 0.0,
        }

        lines = parser_code.split("\n")
        total_lines = len(lines)

        # Complexity analysis (simplified)
        nested_blocks = 0
        max_nesting = 0
        current_nesting = 0

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("if ", "for ", "while ", "try:", "with ")):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
                nested_blocks += 1
            elif stripped in ("else:", "elif ", "except:", "finally:"):
                nested_blocks += 1
            elif stripped.startswith(("return", "break", "continue")):
                current_nesting = max(0, current_nesting - 1)

        # Complexity score (0-1, lower is better)
        metrics["complexity"] = min(
            1.0, (max_nesting * 0.2 + nested_blocks / max(total_lines, 1))
        )

        # Maintainability (0-1, higher is better)
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        docstring_lines = parser_code.count('"""') + parser_code.count("'''")
        function_count = parser_code.count("def ")
        class_count = parser_code.count("class ")

        documentation_ratio = (comment_lines + docstring_lines * 3) / max(
            total_lines, 1
        )
        structure_score = min(
            1.0, (function_count + class_count * 2) / max(total_lines / 20, 1)
        )

        metrics["maintainability"] = min(
            1.0, (documentation_ratio * 0.4 + structure_score * 0.6)
        )

        # Performance indicators (0-1, higher is better)
        # Look for performance anti-patterns
        performance_issues = 0
        performance_issues += (
            parser_code.count("for ") * parser_code.count("for ") * 0.1
        )  # Nested loops
        performance_issues += parser_code.count("while True:") * 0.2  # Infinite loops
        performance_issues += parser_code.count("sleep(") * 0.1  # Blocking calls

        metrics["performance"] = max(0.0, 1.0 - min(1.0, performance_issues))

        # Documentation score
        metrics["documentation"] = min(1.0, documentation_ratio * 2)

        return metrics

    async def _generate_improvements(
        self,
        parser_code: str,
        errors: List[ParserError],
        error_categories: Dict[str, List[ParserError]],
        quality_metrics: Dict[str, float],
    ) -> List[ParserImprovement]:
        """Generate improvement suggestions using LLM."""
        improvements = []

        try:
            # Get relevant knowledge from RAG
            rag_context = await self.rag_engine.query_similar(
                "parser improvement best practices error handling",
                collection_name="protocol_knowledge",
                n_results=3,
                similarity_threshold=0.6,
            )

            # Build context for LLM
            context_str = self._build_improvement_context(
                parser_code, errors, error_categories, quality_metrics, rag_context
            )

            # Request LLM analysis
            llm_request = LLMRequest(
                prompt=f"""
                Analyze this protocol parser and provide specific, actionable improvement suggestions.
                
                {context_str}
                
                Provide improvements in the following format for each suggestion:
                
                IMPROVEMENT: [Type]
                TITLE: [Clear, concise title]
                PRIORITY: [low/medium/high/critical]
                DESCRIPTION: [Detailed description of the improvement]
                CODE_EXAMPLE:
                ```python
                [Concrete code example showing the improvement]
                ```
                STEPS:
                1. [Implementation step 1]
                2. [Implementation step 2]
                ...
                IMPACT: [Expected impact description]
                ---
                
                Focus on:
                1. Error handling improvements
                2. Performance optimizations
                3. Code maintainability
                4. Security enhancements
                5. Best practices
                """,
                feature_domain="protocol_copilot",
                max_tokens=2000,
                temperature=0.2,
            )

            llm_response = await self.llm_service.process_request(llm_request)

            # Parse LLM response into structured improvements
            improvements = self._parse_llm_improvements(llm_response.content, errors)

            # Add rule-based improvements
            rule_based = self._generate_rule_based_improvements(
                error_categories, quality_metrics
            )
            improvements.extend(rule_based)

            # Sort by priority
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            improvements.sort(key=lambda x: priority_order.get(x.priority, 4))

            return improvements[:10]  # Return top 10 improvements

        except Exception as e:
            self.logger.error(f"Failed to generate improvements: {e}")
            # Return rule-based improvements as fallback
            return self._generate_rule_based_improvements(
                error_categories, quality_metrics
            )

    def _build_improvement_context(
        self,
        parser_code: str,
        errors: List[ParserError],
        error_categories: Dict[str, List[ParserError]],
        quality_metrics: Dict[str, float],
        rag_context: Any,
    ) -> str:
        """Build context string for LLM."""
        context_parts = []

        # Parser code snippet
        code_lines = parser_code.split("\n")
        if len(code_lines) > 50:
            context_parts.append(
                f"Parser Code (first 30 lines, last 20 lines):\n```python\n"
            )
            context_parts.append("\n".join(code_lines[:30]))
            context_parts.append("\n...\n")
            context_parts.append("\n".join(code_lines[-20:]))
            context_parts.append("\n```\n")
        else:
            context_parts.append(f"Parser Code:\n```python\n{parser_code}\n```\n")

        # Error summary
        context_parts.append(f"\nErrors Found ({len(errors)} total):")
        for category, category_errors in error_categories.items():
            context_parts.append(f"\n{category.upper()} ({len(category_errors)}):")
            for error in category_errors[:3]:  # Show first 3 of each type
                context_parts.append(f"  - {error.error_message}")

        # Quality metrics
        context_parts.append(f"\nQuality Metrics:")
        context_parts.append(f"  - Complexity: {quality_metrics['complexity']:.2f}")
        context_parts.append(
            f"  - Maintainability: {quality_metrics['maintainability']:.2f}"
        )
        context_parts.append(f"  - Performance: {quality_metrics['performance']:.2f}")
        context_parts.append(
            f"  - Documentation: {quality_metrics['documentation']:.2f}"
        )

        # RAG context
        if rag_context and rag_context.documents:
            context_parts.append(f"\nRelevant Best Practices:")
            for doc in rag_context.documents[:2]:
                context_parts.append(f"  - {doc.content[:200]}...")

        return "\n".join(context_parts)

    def _parse_llm_improvements(
        self, llm_response: str, errors: List[ParserError]
    ) -> List[ParserImprovement]:
        """Parse LLM response into structured improvements."""
        improvements = []

        try:
            # Split by improvement separator
            sections = llm_response.split("---")

            for section in sections:
                if not section.strip():
                    continue

                improvement = self._parse_improvement_section(section, errors)
                if improvement:
                    improvements.append(improvement)

        except Exception as e:
            self.logger.error(f"Failed to parse LLM improvements: {e}")

        return improvements

    def _parse_improvement_section(
        self, section: str, errors: List[ParserError]
    ) -> Optional[ParserImprovement]:
        """Parse a single improvement section."""
        try:
            lines = section.strip().split("\n")

            improvement_data = {
                "improvement_type": "general",
                "title": "Improvement",
                "description": "",
                "code_example": None,
                "priority": "medium",
                "implementation_steps": [],
                "related_errors": [],
            }

            current_field = None
            code_block = []
            in_code_block = False

            for line in lines:
                line = line.strip()

                if line.startswith("IMPROVEMENT:"):
                    improvement_data["improvement_type"] = (
                        line.split(":", 1)[1].strip().lower()
                    )
                elif line.startswith("TITLE:"):
                    improvement_data["title"] = line.split(":", 1)[1].strip()
                elif line.startswith("PRIORITY:"):
                    priority = line.split(":", 1)[1].strip().lower()
                    if priority in ["low", "medium", "high", "critical"]:
                        improvement_data["priority"] = priority
                elif line.startswith("DESCRIPTION:"):
                    current_field = "description"
                    improvement_data["description"] = line.split(":", 1)[1].strip()
                elif line.startswith("CODE_EXAMPLE:"):
                    current_field = "code"
                elif line.startswith("```"):
                    if in_code_block:
                        improvement_data["code_example"] = "\n".join(code_block)
                        code_block = []
                        in_code_block = False
                        current_field = None
                    else:
                        in_code_block = True
                elif line.startswith("STEPS:"):
                    current_field = "steps"
                elif line.startswith("IMPACT:"):
                    improvement_data["estimated_impact"] = line.split(":", 1)[1].strip()
                    current_field = None
                elif current_field == "description" and line:
                    improvement_data["description"] += " " + line
                elif current_field == "code" and in_code_block:
                    code_block.append(line)
                elif current_field == "steps" and line.startswith(tuple("123456789")):
                    step = line.split(".", 1)[1].strip() if "." in line else line
                    improvement_data["implementation_steps"].append(step)

            # Match related errors
            for error in errors:
                if any(
                    keyword in error.error_message.lower()
                    for keyword in improvement_data["improvement_type"].split()
                ):
                    improvement_data["related_errors"].append(error.error_message)

            return ParserImprovement(**improvement_data)

        except Exception as e:
            self.logger.error(f"Failed to parse improvement section: {e}")
            return None

    def _generate_rule_based_improvements(
        self,
        error_categories: Dict[str, List[ParserError]],
        quality_metrics: Dict[str, float],
    ) -> List[ParserImprovement]:
        """Generate improvements based on rules."""
        improvements = []

        # Error handling improvements
        if "syntax_error" in error_categories or "type_error" in error_categories:
            improvements.append(
                ParserImprovement(
                    improvement_type="error_handling",
                    title="Add Comprehensive Error Handling",
                    description="Implement try-except blocks around parsing operations to handle syntax and type errors gracefully.",
                    code_example="""try:
    parsed_value = parse_field(data)
except (ValueError, TypeError) as e:
    logger.error(f"Parsing failed: {e}")
    return default_value
except Exception as e:
    logger.critical(f"Unexpected error: {e}")
    raise ParserException(f"Critical parsing error: {e}")""",
                    priority="high",
                    estimated_impact="significant",
                    implementation_steps=[
                        "Identify all parsing operations that can fail",
                        "Wrap operations in try-except blocks",
                        "Add specific exception handlers for common errors",
                        "Implement proper logging for debugging",
                        "Add fallback mechanisms for recoverable errors",
                    ],
                    related_errors=[
                        e.error_message
                        for e in error_categories.get("syntax_error", [])[:3]
                    ],
                    confidence=0.9,
                )
            )

        # Performance improvements
        if quality_metrics["performance"] < 0.7:
            improvements.append(
                ParserImprovement(
                    improvement_type="performance",
                    title="Optimize Parser Performance",
                    description="Reduce computational complexity and improve parsing speed through caching and efficient algorithms.",
                    code_example='''from functools import lru_cache

@lru_cache(maxsize=1000)
def parse_cached_field(field_data: bytes) -> Dict:
    """Cache frequently parsed fields."""
    return parse_field(field_data)

# Use generators for large datasets
def parse_messages_stream(messages):
    """Stream parsing to reduce memory usage."""
    for message in messages:
        yield parse_message(message)''',
                    priority="medium",
                    estimated_impact="moderate",
                    implementation_steps=[
                        "Profile parser to identify bottlenecks",
                        "Implement caching for repeated operations",
                        "Use generators for large data processing",
                        "Optimize regex patterns and string operations",
                        "Consider parallel processing for independent operations",
                    ],
                    confidence=0.85,
                )
            )

        # Maintainability improvements
        if quality_metrics["maintainability"] < 0.6:
            improvements.append(
                ParserImprovement(
                    improvement_type="maintainability",
                    title="Improve Code Structure and Documentation",
                    description="Enhance code organization, add comprehensive documentation, and follow best practices.",
                    code_example='''class ProtocolParser:
    """
    Parser for XYZ protocol messages.
    
    Attributes:
        field_definitions: Dictionary of field specifications
        validation_rules: List of validation rules to apply
    
    Example:
        >>> parser = ProtocolParser()
        >>> result = parser.parse(message_data)
    """
    
    def parse(self, data: bytes) -> ParseResult:
        """
        Parse protocol message from raw bytes.
        
        Args:
            data: Raw message bytes
            
        Returns:
            ParseResult with extracted fields
            
        Raises:
            ParserException: If parsing fails
        """
        # Implementation
        pass''',
                    priority="medium",
                    estimated_impact="moderate",
                    implementation_steps=[
                        "Add docstrings to all classes and functions",
                        "Break down large functions into smaller, focused ones",
                        "Use type hints for better code clarity",
                        "Add inline comments for complex logic",
                        "Create comprehensive examples and usage documentation",
                    ],
                    confidence=0.8,
                )
            )

        # Validation improvements
        if "validation_error" in error_categories:
            improvements.append(
                ParserImprovement(
                    improvement_type="validation",
                    title="Strengthen Input Validation",
                    description="Add comprehensive input validation to prevent invalid data from causing errors.",
                    code_example='''def validate_message(data: bytes) -> bool:
    """Validate message before parsing."""
    if not data:
        raise ValueError("Empty message data")
    
    if len(data) < MIN_MESSAGE_SIZE:
        raise ValueError(f"Message too short: {len(data)} bytes")
    
    if len(data) > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {len(data)} bytes")
    
    # Validate message header
    if not is_valid_header(data[:HEADER_SIZE]):
        raise ValueError("Invalid message header")
    
    return True''',
                    priority="high",
                    estimated_impact="significant",
                    implementation_steps=[
                        "Define validation rules for all input fields",
                        "Implement validation functions",
                        "Add validation before parsing operations",
                        "Provide clear error messages for validation failures",
                        "Log validation failures for monitoring",
                    ],
                    related_errors=[
                        e.error_message
                        for e in error_categories.get("validation_error", [])[:3]
                    ],
                    confidence=0.9,
                )
            )

        return improvements

    def _calculate_health_score(
        self,
        errors: List[ParserError],
        quality_metrics: Dict[str, float],
        improvements: List[ParserImprovement],
    ) -> str:
        """Calculate overall parser health."""
        # Count critical issues
        critical_errors = sum(1 for e in errors if e.severity == "error")
        critical_improvements = sum(1 for i in improvements if i.priority == "critical")

        # Calculate weighted score
        error_score = max(0, 1.0 - (critical_errors * 0.2))
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        improvement_score = max(0, 1.0 - (critical_improvements * 0.15))

        overall_score = (
            error_score * 0.4 + quality_score * 0.4 + improvement_score * 0.2
        )

        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.7:
            return "good"
        elif overall_score >= 0.5:
            return "fair"
        else:
            return "poor"

    async def _generate_recommendations_summary(
        self,
        improvements: List[ParserImprovement],
        health: str,
        quality_metrics: Dict[str, float],
    ) -> str:
        """Generate executive summary of recommendations."""
        try:
            # Count improvements by priority
            priority_counts = Counter(i.priority for i in improvements)

            summary_parts = [
                f"Parser Health: {health.upper()}",
                f"\nQuality Metrics:",
                f"  - Complexity: {quality_metrics['complexity']:.1%}",
                f"  - Maintainability: {quality_metrics['maintainability']:.1%}",
                f"  - Performance: {quality_metrics['performance']:.1%}",
                f"\nRecommendations: {len(improvements)} total",
                f"  - Critical: {priority_counts.get('critical', 0)}",
                f"  - High: {priority_counts.get('high', 0)}",
                f"  - Medium: {priority_counts.get('medium', 0)}",
                f"  - Low: {priority_counts.get('low', 0)}",
                f"\nTop Priority Actions:",
            ]

            # Add top 3 improvements
            for i, improvement in enumerate(improvements[:3], 1):
                summary_parts.append(
                    f"  {i}. {improvement.title} ({improvement.priority} priority)"
                )

            return "\n".join(summary_parts)

        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            return (
                f"Parser health: {health}. {len(improvements)} improvements suggested."
            )
