"""
QBITEL Bridge - Legacy Whisperer Data Models

Shared data models, enums, and exceptions for the Legacy Whisperer services.
"""

import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional

from ...core.exceptions import QbitelAIException


class LegacyWhispererException(QbitelAIException):
    """Legacy Whisperer-specific exception."""

    pass


class ProtocolComplexity(str, Enum):
    """Protocol complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


class ModernizationRisk(str, Enum):
    """Modernization risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AdapterLanguage(str, Enum):
    """Supported adapter code languages."""

    PYTHON = "python"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    TYPESCRIPT = "typescript"
    CSHARP = "csharp"


@dataclass
class ProtocolPattern:
    """Identified protocol pattern."""

    pattern_type: str
    description: str
    frequency: int
    confidence: float
    examples: List[bytes] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolField:
    """Reverse-engineered protocol field."""

    name: str
    offset: int
    length: int
    field_type: str
    description: str
    possible_values: List[Any] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ProtocolSpecification:
    """Complete protocol specification from reverse engineering."""

    protocol_name: str
    version: str
    description: str
    complexity: ProtocolComplexity

    # Structure information
    fields: List[ProtocolField] = field(default_factory=list)
    message_types: List[Dict[str, Any]] = field(default_factory=list)
    patterns: List[ProtocolPattern] = field(default_factory=list)

    # Protocol characteristics
    is_binary: bool = True
    is_stateful: bool = False
    uses_encryption: bool = False
    has_checksums: bool = False

    # Analysis metadata
    confidence_score: float = 0.0
    analysis_time: float = 0.0
    samples_analyzed: int = 0

    # Documentation
    documentation: str = ""
    usage_examples: List[str] = field(default_factory=list)
    known_implementations: List[str] = field(default_factory=list)

    # Historical context
    historical_context: str = ""
    common_issues: List[str] = field(default_factory=list)
    security_concerns: List[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    spec_id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:16])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result["created_at"] = result["created_at"].isoformat()
        result["complexity"] = result["complexity"]
        return result


@dataclass
class AdapterCode:
    """Generated protocol adapter code."""

    source_protocol: str
    target_protocol: str
    language: AdapterLanguage

    # Generated code
    adapter_code: str
    test_code: str
    documentation: str

    # Code metadata
    dependencies: List[str] = field(default_factory=list)
    configuration_template: str = ""
    deployment_guide: str = ""

    # Integration information
    integration_points: List[str] = field(default_factory=list)
    api_endpoints: List[Dict[str, Any]] = field(default_factory=list)

    # Quality metrics
    code_quality_score: float = 0.0
    test_coverage: float = 0.0
    performance_notes: str = ""

    # Generation metadata
    generation_time: float = 0.0
    llm_provider: str = ""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    adapter_id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:16])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result["created_at"] = result["created_at"].isoformat()
        result["language"] = result["language"]
        return result


@dataclass
class Explanation:
    """Explanation of legacy system behavior."""

    behavior_description: str
    technical_explanation: str
    historical_context: str

    # Analysis
    root_causes: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)

    # Modernization guidance
    modernization_approaches: List[Dict[str, Any]] = field(default_factory=list)
    recommended_approach: Optional[str] = None

    # Risk assessment
    modernization_risks: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: ModernizationRisk = ModernizationRisk.MEDIUM

    # Implementation guidance
    implementation_steps: List[str] = field(default_factory=list)
    estimated_effort: str = ""
    required_expertise: List[str] = field(default_factory=list)

    # Quality metrics
    confidence: float = 0.0
    completeness: float = 0.0

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    explanation_id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:16])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result["created_at"] = result["created_at"].isoformat()
        result["risk_level"] = result["risk_level"]
        return result
