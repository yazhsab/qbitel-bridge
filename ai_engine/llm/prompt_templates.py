"""
QBITEL - Prompt Template Management
Version-controlled prompt templates with A/B testing and analytics.

Features:
- Template versioning and rollback
- Variable interpolation
- A/B testing support
- Performance tracking
- Template composition
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel, Field


class TemplateStatus(Enum):
    """Status of a prompt template."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class VariableType(Enum):
    """Types of template variables."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"
    OPTIONAL = "optional"


@dataclass
class TemplateVariable:
    """Definition of a template variable."""

    name: str
    var_type: VariableType
    description: str
    default: Optional[Any] = None
    required: bool = True
    validation_regex: Optional[str] = None
    max_length: Optional[int] = None


@dataclass
class TemplateVersion:
    """A specific version of a template."""

    version: str  # Semantic versioning (e.g., "1.0.0")
    content: str
    system_prompt: Optional[str]
    variables: List[TemplateVariable]
    created_at: datetime
    created_by: str
    changelog: str
    is_current: bool = False
    performance_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTemplate:
    """A prompt template with version history."""

    template_id: str
    name: str
    description: str
    feature_domain: str
    status: TemplateStatus
    versions: List[TemplateVersion]
    current_version: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_current(self) -> Optional[TemplateVersion]:
        """Get the current version of the template."""
        for v in self.versions:
            if v.version == self.current_version:
                return v
        return self.versions[-1] if self.versions else None


@dataclass
class RenderedPrompt:
    """A rendered prompt ready for LLM consumption."""

    prompt: str
    system_prompt: Optional[str]
    template_id: str
    version: str
    variables_used: Dict[str, Any]
    render_time: float
    validation_passed: bool
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing templates."""

    test_id: str
    template_id: str
    version_a: str
    version_b: str
    traffic_split: float  # 0.0 to 1.0, percentage going to version_b
    start_time: datetime
    end_time: Optional[datetime]
    min_samples: int = 100
    is_active: bool = True


@dataclass
class ABTestResult:
    """Results from an A/B test."""

    test_id: str
    version_a_samples: int
    version_b_samples: int
    version_a_metrics: Dict[str, float]
    version_b_metrics: Dict[str, float]
    winner: Optional[str]
    confidence: float
    is_significant: bool


class PromptTemplateManager:
    """
    Manages prompt templates with versioning, rendering, and A/B testing.

    Features:
    - Version control for prompts
    - Variable interpolation with validation
    - A/B testing framework
    - Performance tracking
    - Template composition
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        enable_ab_testing: bool = True,
        enable_analytics: bool = True,
    ):
        self.logger = logging.getLogger(__name__)
        self.storage_path = Path(storage_path) if storage_path else None
        self.enable_ab_testing = enable_ab_testing
        self.enable_analytics = enable_analytics

        # Template storage
        self._templates: Dict[str, PromptTemplate] = {}

        # A/B tests
        self._ab_tests: Dict[str, ABTestConfig] = {}
        self._ab_results: Dict[str, Dict[str, List[float]]] = {}  # test_id -> version -> scores

        # Analytics
        self._usage_stats: Dict[str, Dict[str, int]] = {}  # template_id -> version -> count
        self._performance_data: Dict[str, List[Dict[str, Any]]] = {}

        # Load built-in templates
        self._load_builtin_templates()

    def _load_builtin_templates(self) -> None:
        """Load built-in protocol analysis templates."""
        builtin_templates = [
            PromptTemplate(
                template_id="protocol_analysis",
                name="Protocol Analysis",
                description="Analyze network protocol behavior and structure",
                feature_domain="protocol_copilot",
                status=TemplateStatus.ACTIVE,
                current_version="1.0.0",
                tags=["protocol", "analysis", "network"],
                versions=[
                    TemplateVersion(
                        version="1.0.0",
                        content="""Analyze the following network protocol data:

Protocol Type: {{protocol_type}}
Capture Data:
{{capture_data}}

{% if context %}
Additional Context:
{{context}}
{% endif %}

Please provide:
1. Protocol identification and version
2. Message structure breakdown
3. Field-by-field analysis
4. Any anomalies or security concerns
5. Recommendations for handling""",
                        system_prompt="You are a network protocol expert. Analyze protocol data accurately and identify potential security issues.",
                        variables=[
                            TemplateVariable("protocol_type", VariableType.STRING, "Type of protocol", required=True),
                            TemplateVariable("capture_data", VariableType.STRING, "Raw capture data", required=True),
                            TemplateVariable("context", VariableType.STRING, "Additional context", required=False),
                        ],
                        created_at=datetime.now(),
                        created_by="system",
                        changelog="Initial version",
                        is_current=True,
                    )
                ],
            ),
            PromptTemplate(
                template_id="security_assessment",
                name="Security Assessment",
                description="Assess security posture and identify vulnerabilities",
                feature_domain="security_orchestrator",
                status=TemplateStatus.ACTIVE,
                current_version="1.0.0",
                tags=["security", "assessment", "vulnerability"],
                versions=[
                    TemplateVersion(
                        version="1.0.0",
                        content="""Perform a security assessment for:

Target: {{target}}
Assessment Type: {{assessment_type}}

{% if previous_findings %}
Previous Findings:
{{previous_findings}}
{% endif %}

Configuration:
{{configuration}}

Provide:
1. Identified vulnerabilities (severity: critical/high/medium/low)
2. Attack vectors
3. Remediation recommendations
4. Risk score (1-10)
5. Compliance implications""",
                        system_prompt="You are a cybersecurity expert. Provide thorough, accurate security assessments with actionable recommendations.",
                        variables=[
                            TemplateVariable("target", VariableType.STRING, "Assessment target", required=True),
                            TemplateVariable("assessment_type", VariableType.STRING, "Type of assessment", default="full"),
                            TemplateVariable("configuration", VariableType.STRING, "Target configuration", required=True),
                            TemplateVariable("previous_findings", VariableType.STRING, "Previous findings", required=False),
                        ],
                        created_at=datetime.now(),
                        created_by="system",
                        changelog="Initial version",
                        is_current=True,
                    )
                ],
            ),
            PromptTemplate(
                template_id="compliance_check",
                name="Compliance Check",
                description="Check configuration against compliance frameworks",
                feature_domain="compliance_reporter",
                status=TemplateStatus.ACTIVE,
                current_version="1.0.0",
                tags=["compliance", "audit", "framework"],
                versions=[
                    TemplateVersion(
                        version="1.0.0",
                        content="""Check compliance against {{framework}}:

Configuration to Check:
{{configuration}}

Specific Controls:
{% for control in controls %}
- {{control}}
{% endfor %}

Provide:
1. Compliance status for each control
2. Gaps identified
3. Evidence requirements
4. Remediation steps
5. Overall compliance score""",
                        system_prompt="You are a compliance expert. Assess configurations against regulatory frameworks accurately and provide actionable remediation guidance.",
                        variables=[
                            TemplateVariable("framework", VariableType.STRING, "Compliance framework", default="PCI-DSS"),
                            TemplateVariable("configuration", VariableType.STRING, "Configuration to check", required=True),
                            TemplateVariable(
                                "controls", VariableType.LIST, "Specific controls to check", required=False, default=[]
                            ),
                        ],
                        created_at=datetime.now(),
                        created_by="system",
                        changelog="Initial version",
                        is_current=True,
                    )
                ],
            ),
            PromptTemplate(
                template_id="code_generation",
                name="Code Generation",
                description="Generate code from specifications",
                feature_domain="translation_studio",
                status=TemplateStatus.ACTIVE,
                current_version="1.0.0",
                tags=["code", "generation", "api"],
                versions=[
                    TemplateVersion(
                        version="1.0.0",
                        content="""Generate {{language}} code for:

Specification:
{{specification}}

Requirements:
{% for req in requirements %}
- {{req}}
{% endfor %}

{% if examples %}
Examples:
{{examples}}
{% endif %}

Generate production-ready code with:
1. Proper error handling
2. Type hints/annotations
3. Documentation
4. Unit test examples""",
                        system_prompt="You are an expert software engineer. Generate clean, production-ready code following best practices.",
                        variables=[
                            TemplateVariable("language", VariableType.STRING, "Target programming language", default="python"),
                            TemplateVariable("specification", VariableType.STRING, "Code specification", required=True),
                            TemplateVariable("requirements", VariableType.LIST, "Specific requirements", default=[]),
                            TemplateVariable("examples", VariableType.STRING, "Example code/usage", required=False),
                        ],
                        created_at=datetime.now(),
                        created_by="system",
                        changelog="Initial version",
                        is_current=True,
                    )
                ],
            ),
        ]

        for template in builtin_templates:
            self._templates[template.template_id] = template

    def create_template(
        self,
        template_id: str,
        name: str,
        description: str,
        content: str,
        feature_domain: str,
        system_prompt: Optional[str] = None,
        variables: Optional[List[TemplateVariable]] = None,
        tags: Optional[List[str]] = None,
        created_by: str = "user",
    ) -> PromptTemplate:
        """
        Create a new prompt template.

        Args:
            template_id: Unique identifier
            name: Human-readable name
            description: Template description
            content: Template content with variables
            feature_domain: Associated feature domain
            system_prompt: Optional system prompt
            variables: Template variables
            tags: Template tags
            created_by: Creator identifier

        Returns:
            Created PromptTemplate
        """
        if template_id in self._templates:
            raise ValueError(f"Template '{template_id}' already exists")

        # Auto-detect variables if not provided
        if variables is None:
            variables = self._detect_variables(content)

        version = TemplateVersion(
            version="1.0.0",
            content=content,
            system_prompt=system_prompt,
            variables=variables,
            created_at=datetime.now(),
            created_by=created_by,
            changelog="Initial version",
            is_current=True,
        )

        template = PromptTemplate(
            template_id=template_id,
            name=name,
            description=description,
            feature_domain=feature_domain,
            status=TemplateStatus.ACTIVE,
            versions=[version],
            current_version="1.0.0",
            tags=tags or [],
        )

        self._templates[template_id] = template
        return template

    def update_template(
        self,
        template_id: str,
        content: str,
        system_prompt: Optional[str] = None,
        variables: Optional[List[TemplateVariable]] = None,
        changelog: str = "",
        created_by: str = "user",
        version_bump: str = "minor",  # "major", "minor", "patch"
    ) -> TemplateVersion:
        """
        Create a new version of an existing template.

        Args:
            template_id: Template to update
            content: New template content
            system_prompt: New system prompt
            variables: New variables (or auto-detect)
            changelog: Description of changes
            created_by: Updater identifier
            version_bump: Type of version bump

        Returns:
            New TemplateVersion
        """
        if template_id not in self._templates:
            raise ValueError(f"Template '{template_id}' not found")

        template = self._templates[template_id]
        current = template.get_current()

        # Calculate new version
        if current:
            parts = current.version.split(".")
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

            if version_bump == "major":
                new_version = f"{major + 1}.0.0"
            elif version_bump == "minor":
                new_version = f"{major}.{minor + 1}.0"
            else:
                new_version = f"{major}.{minor}.{patch + 1}"
        else:
            new_version = "1.0.0"

        # Auto-detect variables if not provided
        if variables is None:
            variables = self._detect_variables(content)

        # Mark old versions as not current
        for v in template.versions:
            v.is_current = False

        version = TemplateVersion(
            version=new_version,
            content=content,
            system_prompt=system_prompt or (current.system_prompt if current else None),
            variables=variables,
            created_at=datetime.now(),
            created_by=created_by,
            changelog=changelog,
            is_current=True,
        )

        template.versions.append(version)
        template.current_version = new_version
        template.updated_at = datetime.now()

        return version

    def render(
        self,
        template_id: str,
        variables: Dict[str, Any],
        version: Optional[str] = None,
        strict: bool = True,
    ) -> RenderedPrompt:
        """
        Render a template with variables.

        Args:
            template_id: Template to render
            variables: Variable values
            version: Specific version (or use current)
            strict: Raise errors on validation failures

        Returns:
            RenderedPrompt with rendered content
        """
        start_time = time.time()

        if template_id not in self._templates:
            raise ValueError(f"Template '{template_id}' not found")

        template = self._templates[template_id]

        # Check for A/B test
        if self.enable_ab_testing:
            version = self._get_ab_test_version(template_id, version)

        # Get template version
        template_version = None
        if version:
            for v in template.versions:
                if v.version == version:
                    template_version = v
                    break
        else:
            template_version = template.get_current()

        if not template_version:
            raise ValueError(f"Version '{version}' not found for template '{template_id}'")

        # Validate variables
        validation_errors = self._validate_variables(template_version, variables)
        validation_passed = len(validation_errors) == 0

        if strict and not validation_passed:
            raise ValueError(f"Variable validation failed: {validation_errors}")

        # Render template
        rendered_content = self._interpolate(template_version.content, variables)
        rendered_system = None
        if template_version.system_prompt:
            rendered_system = self._interpolate(template_version.system_prompt, variables)

        # Track usage
        if self.enable_analytics:
            self._track_usage(template_id, template_version.version)

        return RenderedPrompt(
            prompt=rendered_content,
            system_prompt=rendered_system,
            template_id=template_id,
            version=template_version.version,
            variables_used=variables,
            render_time=time.time() - start_time,
            validation_passed=validation_passed,
            validation_errors=validation_errors,
        )

    def _detect_variables(self, content: str) -> List[TemplateVariable]:
        """Auto-detect variables from template content."""
        variables = []
        seen = set()

        # Match {{variable}} and {{variable|default}}
        pattern = r"\{\{(\w+)(?:\|([^}]+))?\}\}"
        for match in re.finditer(pattern, content):
            name = match.group(1)
            default = match.group(2)

            if name in seen:
                continue
            seen.add(name)

            variables.append(
                TemplateVariable(
                    name=name,
                    var_type=VariableType.STRING,
                    description=f"Variable: {name}",
                    default=default,
                    required=default is None,
                )
            )

        # Match {% for item in list %}
        list_pattern = r"\{% for \w+ in (\w+) %\}"
        for match in re.finditer(list_pattern, content):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                variables.append(
                    TemplateVariable(
                        name=name,
                        var_type=VariableType.LIST,
                        description=f"List variable: {name}",
                        default=[],
                        required=False,
                    )
                )

        return variables

    def _validate_variables(
        self,
        version: TemplateVersion,
        variables: Dict[str, Any],
    ) -> List[str]:
        """Validate variable values against definitions."""
        errors = []

        for var_def in version.variables:
            value = variables.get(var_def.name)

            # Check required
            if var_def.required and value is None and var_def.default is None:
                errors.append(f"Required variable '{var_def.name}' is missing")
                continue

            if value is None:
                continue

            # Type validation
            if var_def.var_type == VariableType.LIST and not isinstance(value, list):
                errors.append(f"Variable '{var_def.name}' must be a list")
            elif var_def.var_type == VariableType.NUMBER and not isinstance(value, (int, float)):
                errors.append(f"Variable '{var_def.name}' must be a number")
            elif var_def.var_type == VariableType.BOOLEAN and not isinstance(value, bool):
                errors.append(f"Variable '{var_def.name}' must be a boolean")

            # Length validation
            if var_def.max_length and isinstance(value, str) and len(value) > var_def.max_length:
                errors.append(f"Variable '{var_def.name}' exceeds max length {var_def.max_length}")

            # Regex validation
            if var_def.validation_regex and isinstance(value, str):
                if not re.match(var_def.validation_regex, value):
                    errors.append(f"Variable '{var_def.name}' does not match pattern")

        return errors

    def _interpolate(self, template: str, variables: Dict[str, Any]) -> str:
        """Interpolate variables into template."""
        result = template

        # Handle conditionals {% if var %}...{% endif %}
        if_pattern = r"\{% if (\w+) %\}(.*?)\{% endif %\}"
        for match in re.finditer(if_pattern, result, re.DOTALL):
            var_name = match.group(1)
            content = match.group(2)
            if variables.get(var_name):
                # Render the content
                rendered = self._interpolate(content, variables)
                result = result.replace(match.group(0), rendered)
            else:
                result = result.replace(match.group(0), "")

        # Handle for loops {% for item in list %}...{% endfor %}
        for_pattern = r"\{% for (\w+) in (\w+) %\}(.*?)\{% endfor %\}"
        for match in re.finditer(for_pattern, result, re.DOTALL):
            item_name = match.group(1)
            list_name = match.group(2)
            content = match.group(3)

            items = variables.get(list_name, [])
            rendered_items = []
            for item in items:
                item_rendered = content.replace(f"{{{{{item_name}}}}}", str(item))
                rendered_items.append(item_rendered.strip())

            result = result.replace(match.group(0), "\n".join(rendered_items))

        # Handle simple variables {{var}} and {{var|default}}
        var_pattern = r"\{\{(\w+)(?:\|([^}]+))?\}\}"
        for match in re.finditer(var_pattern, result):
            var_name = match.group(1)
            default = match.group(2)
            value = variables.get(var_name, default)
            if value is not None:
                result = result.replace(match.group(0), str(value))

        return result.strip()

    def _get_ab_test_version(
        self,
        template_id: str,
        requested_version: Optional[str],
    ) -> Optional[str]:
        """Get version based on A/B test if active."""
        if requested_version:
            return requested_version

        for test in self._ab_tests.values():
            if test.template_id == template_id and test.is_active:
                # Simple random assignment based on hash
                import random

                if random.random() < test.traffic_split:
                    return test.version_b
                else:
                    return test.version_a

        return None

    def _track_usage(self, template_id: str, version: str) -> None:
        """Track template usage for analytics."""
        if template_id not in self._usage_stats:
            self._usage_stats[template_id] = {}
        if version not in self._usage_stats[template_id]:
            self._usage_stats[template_id][version] = 0
        self._usage_stats[template_id][version] += 1

    def start_ab_test(
        self,
        test_id: str,
        template_id: str,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5,
        min_samples: int = 100,
    ) -> ABTestConfig:
        """Start an A/B test for a template."""
        if template_id not in self._templates:
            raise ValueError(f"Template '{template_id}' not found")

        config = ABTestConfig(
            test_id=test_id,
            template_id=template_id,
            version_a=version_a,
            version_b=version_b,
            traffic_split=traffic_split,
            start_time=datetime.now(),
            end_time=None,
            min_samples=min_samples,
            is_active=True,
        )

        self._ab_tests[test_id] = config
        self._ab_results[test_id] = {"a": [], "b": []}

        return config

    def record_ab_result(
        self,
        test_id: str,
        version: str,
        score: float,
    ) -> None:
        """Record a result for an A/B test."""
        if test_id not in self._ab_results:
            return

        variant = "a" if version == self._ab_tests[test_id].version_a else "b"
        self._ab_results[test_id][variant].append(score)

    def get_ab_test_results(self, test_id: str) -> ABTestResult:
        """Get results of an A/B test."""
        if test_id not in self._ab_tests:
            raise ValueError(f"Test '{test_id}' not found")

        config = self._ab_tests[test_id]
        results = self._ab_results[test_id]

        a_scores = results["a"]
        b_scores = results["b"]

        def mean(lst):
            return sum(lst) / len(lst) if lst else 0

        def std(lst):
            if len(lst) < 2:
                return 0
            m = mean(lst)
            return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5

        a_mean = mean(a_scores)
        b_mean = mean(b_scores)

        # Simple statistical significance check
        n_a, n_b = len(a_scores), len(b_scores)
        is_significant = n_a >= config.min_samples and n_b >= config.min_samples

        winner = None
        if is_significant:
            if a_mean > b_mean * 1.05:  # 5% improvement threshold
                winner = config.version_a
            elif b_mean > a_mean * 1.05:
                winner = config.version_b

        # Calculate approximate confidence
        confidence = 0.0
        if is_significant:
            pooled_std = ((std(a_scores) ** 2 + std(b_scores) ** 2) / 2) ** 0.5
            if pooled_std > 0:
                z = abs(a_mean - b_mean) / (pooled_std * (1 / n_a + 1 / n_b) ** 0.5)
                # Approximate p-value from z-score
                confidence = min(0.99, z / 3)

        return ABTestResult(
            test_id=test_id,
            version_a_samples=n_a,
            version_b_samples=n_b,
            version_a_metrics={"mean": a_mean, "std": std(a_scores)},
            version_b_metrics={"mean": b_mean, "std": std(b_scores)},
            winner=winner,
            confidence=confidence,
            is_significant=is_significant,
        )

    def end_ab_test(self, test_id: str) -> None:
        """End an A/B test."""
        if test_id in self._ab_tests:
            self._ab_tests[test_id].is_active = False
            self._ab_tests[test_id].end_time = datetime.now()

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def list_templates(
        self,
        feature_domain: Optional[str] = None,
        status: Optional[TemplateStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[PromptTemplate]:
        """List templates with optional filters."""
        templates = list(self._templates.values())

        if feature_domain:
            templates = [t for t in templates if t.feature_domain == feature_domain]
        if status:
            templates = [t for t in templates if t.status == status]
        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]

        return templates

    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get template usage statistics."""
        return self._usage_stats

    def rollback_version(self, template_id: str, version: str) -> None:
        """Rollback to a previous version."""
        if template_id not in self._templates:
            raise ValueError(f"Template '{template_id}' not found")

        template = self._templates[template_id]

        # Find the version
        target = None
        for v in template.versions:
            if v.version == version:
                target = v
                break

        if not target:
            raise ValueError(f"Version '{version}' not found")

        # Update current markers
        for v in template.versions:
            v.is_current = v.version == version

        template.current_version = version
        template.updated_at = datetime.now()


# Global template manager instance
_global_template_manager: Optional[PromptTemplateManager] = None


def get_template_manager() -> PromptTemplateManager:
    """Get or create global template manager instance."""
    global _global_template_manager
    if _global_template_manager is None:
        _global_template_manager = PromptTemplateManager()
    return _global_template_manager


def configure_template_manager(**config) -> PromptTemplateManager:
    """Configure and return global template manager."""
    global _global_template_manager
    _global_template_manager = PromptTemplateManager(**config)
    return _global_template_manager
