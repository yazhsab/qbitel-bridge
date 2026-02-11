"""
QBITEL - Protocol Translation Studio Models
Enterprise-grade data structures for protocol translation, API generation, and code generation.
"""

import asyncio
import base64
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import uuid

from ..core.config import Config
from ..core.exceptions import QbitelAIException


class TranslationException(QbitelAIException):
    """Translation-specific exceptions."""

    pass


class APIStyle(str, Enum):
    """Supported API styles."""

    REST = "rest"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    ASYNC = "async"


class CodeLanguage(str, Enum):
    """Supported code generation languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CSHARP = "csharp"
    CPP = "cpp"
    PHP = "php"
    RUBY = "ruby"
    KOTLIN = "kotlin"
    SWIFT = "swift"


class FieldType(str, Enum):
    """Protocol field data types."""

    INTEGER = "integer"
    STRING = "string"
    BINARY = "binary"
    BOOLEAN = "boolean"
    FLOAT = "float"
    TIMESTAMP = "timestamp"
    ADDRESS = "address"
    LENGTH = "length"
    CHECKSUM = "checksum"
    DELIMITER = "delimiter"
    RESERVED = "reserved"
    UNKNOWN = "unknown"


class ProtocolFormat(str, Enum):
    """Protocol data formats."""

    BINARY = "binary"
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    MESSAGEPACK = "messagepack"
    YAML = "yaml"


class SecurityLevel(str, Enum):
    """API security levels."""

    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ENTERPRISE = "enterprise"
    RESTRICTED = "restricted"


class GenerationStatus(str, Enum):
    """Generation process status."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProtocolField:
    """Represents a protocol field with semantic information."""

    name: str
    field_type: str
    offset: int
    length: int
    description: Optional[str] = None
    semantic_type: Optional[str] = None
    validation_rules: List[str] = field(default_factory=list)
    examples: List[Any] = field(default_factory=list)
    optional: bool = False
    deprecated: bool = False
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class ProtocolSchema:
    """Complete protocol schema definition."""

    name: str
    version: str
    description: Optional[str] = None
    format: ProtocolFormat = ProtocolFormat.BINARY
    fields: List[ProtocolField] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    security_requirements: List[str] = field(default_factory=list)
    compliance_tags: List[str] = field(default_factory=list)
    semantic_info: Dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def add_field(self, field: ProtocolField) -> None:
        """Add a field to the schema."""
        self.fields.append(field)
        self.updated_at = datetime.now(timezone.utc)

    def get_field_by_name(self, name: str) -> Optional[ProtocolField]:
        """Get field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class APIEndpoint:
    """API endpoint definition."""

    path: str
    method: str
    summary: str
    description: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    operation_id: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if not self.operation_id:
            self.operation_id = f"{self.method}_{self.path.replace('/', '_').replace('{', '').replace('}', '')}"


@dataclass
class APISpecification:
    """Complete API specification."""

    title: str
    version: str
    description: Optional[str] = None
    api_style: APIStyle = APIStyle.REST
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    base_url: Optional[str] = None
    endpoints: List[APIEndpoint] = field(default_factory=list)
    schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security_schemes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    servers: List[Dict[str, Any]] = field(default_factory=list)
    contact: Optional[Dict[str, Any]] = None
    license: Optional[Dict[str, Any]] = None
    extensions: Dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    spec_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def add_endpoint(self, endpoint: APIEndpoint) -> None:
        """Add an endpoint to the API specification."""
        self.endpoints.append(endpoint)

    def to_openapi_dict(self) -> Dict[str, Any]:
        """Convert to OpenAPI 3.0 specification dictionary."""
        openapi_spec = {
            "openapi": "3.0.3",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": self.description or f"Generated API for {self.title}",
            },
            "servers": self.servers
            or [{"url": self.base_url or "http://localhost:8000"}],
            "paths": {},
            "components": {
                "schemas": self.schemas,
                "securitySchemes": self.security_schemes,
            },
        }

        if self.contact:
            openapi_spec["info"]["contact"] = self.contact
        if self.license:
            openapi_spec["info"]["license"] = self.license

        # Convert endpoints to OpenAPI paths
        for endpoint in self.endpoints:
            if endpoint.path not in openapi_spec["paths"]:
                openapi_spec["paths"][endpoint.path] = {}

            openapi_spec["paths"][endpoint.path][endpoint.method.lower()] = {
                "summary": endpoint.summary,
                "description": endpoint.description or endpoint.summary,
                "operationId": endpoint.operation_id,
                "parameters": endpoint.parameters,
                "responses": endpoint.responses,
                "tags": endpoint.tags,
                "deprecated": endpoint.deprecated,
            }

            if endpoint.request_body:
                openapi_spec["paths"][endpoint.path][endpoint.method.lower()][
                    "requestBody"
                ] = endpoint.request_body

            if endpoint.security:
                openapi_spec["paths"][endpoint.path][endpoint.method.lower()][
                    "security"
                ] = endpoint.security

        if self.extensions:
            openapi_spec.setdefault("x-qbitel-extensions", self.extensions)

        return openapi_spec


@dataclass
class GeneratedCode:
    """Generated code artifact."""

    language: CodeLanguage
    code: str
    filename: str
    dependencies: List[str] = field(default_factory=list)
    documentation: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    tests: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    code_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def save_to_file(self, base_path: Path) -> Path:
        """Save generated code to file."""
        output_path = base_path / self.filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.code)

        return output_path


@dataclass
class GeneratedSDK:
    """Complete generated SDK package."""

    name: str
    version: str
    language: CodeLanguage
    description: Optional[str] = None

    # Code artifacts
    source_files: List[GeneratedCode] = field(default_factory=list)
    test_files: List[GeneratedCode] = field(default_factory=list)
    documentation_files: List[GeneratedCode] = field(default_factory=list)
    config_files: List[GeneratedCode] = field(default_factory=list)

    # Package metadata
    dependencies: List[str] = field(default_factory=list)
    dev_dependencies: List[str] = field(default_factory=list)
    build_instructions: Optional[str] = None
    installation_guide: Optional[str] = None

    # Generation metadata
    generated_from_api: str = ""  # API spec ID
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sdk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def add_source_file(self, code: GeneratedCode) -> None:
        """Add a source file to the SDK."""
        self.source_files.append(code)

    def generate_package_structure(self, base_path: Path) -> Dict[str, Path]:
        """Generate complete SDK package structure."""
        sdk_path = base_path / self.name
        paths = {}

        # Create directory structure
        (sdk_path / "src").mkdir(parents=True, exist_ok=True)
        (sdk_path / "tests").mkdir(parents=True, exist_ok=True)
        (sdk_path / "docs").mkdir(parents=True, exist_ok=True)
        (sdk_path / "examples").mkdir(parents=True, exist_ok=True)

        # Save source files
        for code in self.source_files:
            file_path = code.save_to_file(sdk_path / "src")
            paths[f"src/{code.filename}"] = file_path

        # Save test files
        for code in self.test_files:
            file_path = code.save_to_file(sdk_path / "tests")
            paths[f"tests/{code.filename}"] = file_path

        # Save documentation files
        for code in self.documentation_files:
            file_path = code.save_to_file(sdk_path / "docs")
            paths[f"docs/{code.filename}"] = file_path

        # Save config files
        for code in self.config_files:
            file_path = code.save_to_file(sdk_path)
            paths[code.filename] = file_path

        return paths


@dataclass
class ProtocolBridgeConfig:
    """Configuration for protocol bridge translation."""

    source_protocol: str
    target_protocol: str
    mapping_rules: Dict[str, str] = field(default_factory=dict)
    transformation_functions: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)
    performance_settings: Dict[str, Any] = field(default_factory=dict)
    security_settings: Dict[str, Any] = field(default_factory=dict)

    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TranslationMode(str, Enum):
    """Protocol translation modes."""

    DIRECT = "direct"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    STREAMING = "streaming"
    BATCH = "batch"


class QualityLevel(str, Enum):
    """Translation quality levels."""

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    PERFECT = "perfect"


@dataclass
class TranslationRule:
    """Rule for protocol field translation."""

    source_field: str
    target_field: str
    transformation: Optional[str] = None
    validation: Optional[str] = None
    priority: int = 5
    conditions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationRequest:
    """Request for protocol translation and API generation."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_protocol: Optional[str] = None
    target_protocol: Optional[str] = None
    data: Optional[bytes] = None
    protocol_data: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    translation_mode: TranslationMode = TranslationMode.HYBRID
    quality_level: QualityLevel = QualityLevel.BALANCED
    target_api_style: APIStyle = APIStyle.REST
    target_languages: List[CodeLanguage] = field(
        default_factory=lambda: [CodeLanguage.PYTHON]
    )
    generate_documentation: bool = True
    generate_tests: bool = True
    generate_examples: bool = True
    include_validation: bool = True
    security_level: SecurityLevel = SecurityLevel.AUTHENTICATED
    preserve_metadata: bool = True
    enable_validation: bool = True
    custom_rules: List[TranslationRule] = field(default_factory=list)
    user_context: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    priority: int = 5  # 1-10, 10 highest
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Normalise payload handling and ensure compatibility with older fields."""
        if not self.request_id:
            self.request_id = str(uuid.uuid4())

        # Allow legacy callers to pass protocol_data instead of data.
        if self.data is None and self.protocol_data is not None:
            self.data = self.protocol_data
        elif self.protocol_data is None and self.data is not None:
            self.protocol_data = self.data

        if isinstance(self.data, str):
            self.data = self.data.encode("utf-8")
        if isinstance(self.protocol_data, str):
            self.protocol_data = self.protocol_data.encode("utf-8")

        if self.data is None:
            self.data = b""
        if self.protocol_data is None:
            self.protocol_data = self.data

        # Clamp priority within expected bounds.
        self.priority = max(1, min(10, self.priority))

    def validate(self) -> List[str]:
        """Validate request contents and return a list of issues."""
        errors: List[str] = []

        if not self.source_protocol:
            errors.append("source_protocol is required")
        if not self.target_protocol:
            errors.append("target_protocol is required")
        if not self.data:
            errors.append("data payload must be provided")
        if self.data and not isinstance(self.data, (bytes, bytearray)):
            errors.append("data must be bytes")
        if not isinstance(self.priority, int) or not (1 <= self.priority <= 10):
            errors.append("priority must be an integer between 1 and 10")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Create JSON-safe representation of the request."""
        encoded_data = (
            base64.b64encode(self.data).decode("ascii") if self.data else None
        )
        return {
            "request_id": self.request_id,
            "source_protocol": self.source_protocol,
            "target_protocol": self.target_protocol,
            "data": encoded_data,
            "metadata": self.metadata,
            "translation_mode": self.translation_mode.value,
            "quality_level": self.quality_level.value,
            "priority": self.priority,
        }


@dataclass
class LLMAnalysisResult:
    """Results from LLM semantic analysis."""

    protocol_understanding: str
    field_semantics: Dict[str, str] = field(default_factory=dict)
    api_recommendations: List[str] = field(default_factory=list)
    security_considerations: List[str] = field(default_factory=list)
    compliance_notes: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0

    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class APIGenerationResult:
    """Complete result of API generation process."""

    request_id: str
    status: GenerationStatus

    # Generated artifacts
    protocol_schema: Optional[ProtocolSchema] = None
    api_specification: Optional[APISpecification] = None
    generated_sdks: List[GeneratedSDK] = field(default_factory=list)

    # Analysis results
    llm_analysis: Optional[LLMAnalysisResult] = None
    discovery_result: Optional[Dict[str, Any]] = None

    # Process metadata
    processing_time: float = 0.0
    generation_steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Quality metrics
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    security_score: float = 0.0

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def mark_completed(self) -> None:
        """Mark the generation as completed."""
        self.status = GenerationStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, error_message: str) -> None:
        """Mark the generation as failed."""
        self.status = GenerationStatus.FAILED
        self.errors.append(error_message)
        self.completed_at = datetime.now(timezone.utc)


@dataclass
class TranslationResult:
    """Result of protocol translation."""

    translation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_protocol: str = ""
    target_protocol: str = ""
    translated_data: bytes = b""
    confidence: float = 0.0
    processing_time: float = 0.0
    translation_mode: TranslationMode = TranslationMode.HYBRID
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def validate(self) -> List[str]:
        """Validate translation output and return issues."""
        errors: List[str] = []

        if not self.source_protocol:
            errors.append("source_protocol is required")
        if not self.target_protocol:
            errors.append("target_protocol is required")
        if not isinstance(self.translated_data, (bytes, bytearray)):
            errors.append("translated_data must be bytes")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append("confidence must be between 0.0 and 1.0")
        if self.processing_time < 0:
            errors.append("processing_time must be non-negative")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Create JSON-safe representation of the translation result."""
        encoded_data = (
            base64.b64encode(self.translated_data).decode("ascii")
            if self.translated_data
            else None
        )
        return {
            "translation_id": self.translation_id,
            "source_protocol": self.source_protocol,
            "target_protocol": self.target_protocol,
            "translated_data": encoded_data,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "translation_mode": self.translation_mode.value,
            "metadata": self.metadata,
            "validation_errors": self.validation_errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)

        # Convert datetime objects to ISO strings
        if result["created_at"]:
            result["created_at"] = result["created_at"].isoformat()
        if result["completed_at"]:
            result["completed_at"] = result["completed_at"].isoformat()

        return result


@dataclass
class TranslationStudioMetrics:
    """Metrics for translation studio operations."""

    total_requests: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    average_processing_time: float = 0.0

    # By language
    language_usage: Dict[str, int] = field(default_factory=dict)

    # By API style
    api_style_usage: Dict[str, int] = field(default_factory=dict)

    # Quality metrics
    average_confidence: float = 0.0
    average_completeness: float = 0.0
    average_security_score: float = 0.0

    # Error tracking
    error_types: Dict[str, int] = field(default_factory=dict)

    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update_metrics(self, result: APIGenerationResult) -> None:
        """Update metrics with a new result."""
        self.total_requests += 1

        if result.status == GenerationStatus.COMPLETED:
            self.successful_generations += 1

            # Update averages
            self.average_confidence = (
                self.average_confidence * (self.successful_generations - 1)
                + result.confidence_score
            ) / self.successful_generations
            self.average_completeness = (
                self.average_completeness * (self.successful_generations - 1)
                + result.completeness_score
            ) / self.successful_generations
            self.average_security_score = (
                self.average_security_score * (self.successful_generations - 1)
                + result.security_score
            ) / self.successful_generations

        elif result.status == GenerationStatus.FAILED:
            self.failed_generations += 1

            # Track error types
            for error in result.errors:
                error_type = error.split(":")[0] if ":" in error else "unknown"
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

        # Update processing time average
        if result.processing_time > 0:
            total_completed = self.successful_generations + self.failed_generations
            self.average_processing_time = (
                self.average_processing_time * (total_completed - 1)
                + result.processing_time
            ) / total_completed

        self.last_updated = datetime.now(timezone.utc)


# Factory functions for creating common configurations


def create_rest_api_spec(
    title: str, version: str = "1.0.0", description: Optional[str] = None
) -> APISpecification:
    """Create a basic REST API specification."""
    return APISpecification(
        title=title,
        version=version,
        description=description,
        api_style=APIStyle.REST,
        security_level=SecurityLevel.AUTHENTICATED,
        security_schemes={
            "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
            "apiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
        },
    )


def create_protocol_field(
    name: str,
    field_type: str,
    offset: int,
    length: int,
    description: Optional[str] = None,
    optional: bool = False,
) -> ProtocolField:
    """Create a protocol field with validation."""
    return ProtocolField(
        name=name,
        field_type=field_type,
        offset=offset,
        length=length,
        description=description,
        optional=optional,
        validation_rules=[f"length_equals_{length}", f"offset_at_{offset}"],
    )


def create_translation_request(
    protocol_data: bytes,
    target_languages: Optional[List[CodeLanguage]] = None,
    api_style: APIStyle = APIStyle.REST,
) -> TranslationRequest:
    """Create a translation request with sensible defaults."""
    return TranslationRequest(
        request_id=str(uuid.uuid4()),
        protocol_data=protocol_data,
        target_api_style=api_style,
        target_languages=target_languages
        or [CodeLanguage.PYTHON, CodeLanguage.TYPESCRIPT],
        generate_documentation=True,
        generate_tests=True,
        generate_examples=True,
        enable_semantic_analysis=True,
        enable_security_analysis=True,
    )


# Utility functions


def calculate_schema_hash(schema: ProtocolSchema) -> str:
    """Calculate a hash for protocol schema for caching purposes."""
    schema_str = json.dumps(schema.to_dict(), sort_keys=True)
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def validate_api_specification(spec: APISpecification) -> List[str]:
    """Validate an API specification and return list of issues."""
    issues = []

    if not spec.title:
        issues.append("API specification must have a title")

    if not spec.version:
        issues.append("API specification must have a version")

    if spec.api_style != APIStyle.GRPC and not spec.endpoints:
        issues.append("API specification must have at least one endpoint")

    # Check for duplicate endpoints
    endpoint_keys = [(ep.path, ep.method) for ep in spec.endpoints]
    if len(endpoint_keys) != len(set(endpoint_keys)):
        issues.append("API specification contains duplicate endpoints")

    # Validate endpoint responses
    for endpoint in spec.endpoints:
        if not endpoint.responses:
            issues.append(
                f"Endpoint {endpoint.method} {endpoint.path} must have response definitions"
            )
        elif "200" not in endpoint.responses and "201" not in endpoint.responses:
            issues.append(
                f"Endpoint {endpoint.method} {endpoint.path} should have a success response (200/201)"
            )

    return issues


def merge_protocol_schemas(schemas: List[ProtocolSchema]) -> ProtocolSchema:
    """Merge multiple protocol schemas into one comprehensive schema."""
    if not schemas:
        raise ValueError("Cannot merge empty list of schemas")

    if len(schemas) == 1:
        return schemas[0]

    # Use the first schema as base
    merged = ProtocolSchema(
        name=f"merged_{schemas[0].name}",
        version="1.0.0",
        description=f"Merged schema from {len(schemas)} protocol schemas",
        format=schemas[0].format,
    )

    # Merge fields from all schemas
    all_fields = []
    for schema in schemas:
        all_fields.extend(schema.fields)

    # Remove duplicate fields (by name)
    seen_names = set()
    for field in all_fields:
        if field.name not in seen_names:
            merged.add_field(field)
            seen_names.add(field.name)

    # Merge metadata
    for schema in schemas:
        merged.metadata.update(schema.metadata)

    # Merge security requirements and compliance tags
    for schema in schemas:
        merged.security_requirements.extend(schema.security_requirements)
        merged.compliance_tags.extend(schema.compliance_tags)

    # Remove duplicates
    merged.security_requirements = list(set(merged.security_requirements))
    merged.compliance_tags = list(set(merged.compliance_tags))

    return merged


def create_protocol_schema(
    name: str,
    fields: Optional[List[Dict[str, Any]]] = None,
    *,
    version: str = "1.0",
    description: Optional[str] = None,
    semantic_info: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    protocol_format: ProtocolFormat = ProtocolFormat.BINARY,
) -> ProtocolSchema:
    """Factory helper for quickly constructing protocol schemas."""
    schema = ProtocolSchema(
        name=name,
        version=version,
        description=description,
        format=protocol_format,
        semantic_info=semantic_info or {},
        metadata=metadata or {},
    )

    for index, field_def in enumerate(fields or []):
        if not validate_field_definition(field_def):
            continue

        length = field_def.get("size", field_def.get("length", 0)) or 0
        offset = field_def.get("offset", index * max(1, length))
        field = ProtocolField(
            name=field_def["name"],
            field_type=field_def.get("type", FieldType.STRING.value),
            offset=offset,
            length=length,
            description=field_def.get("description"),
            optional=field_def.get("optional", False),
            semantic_type=field_def.get("semantic_type"),
        )
        schema.add_field(field)

    return schema


def create_api_specification(
    protocol_schema: ProtocolSchema,
    *,
    api_style: APIStyle = APIStyle.REST,
    security_level: SecurityLevel = SecurityLevel.PUBLIC,
) -> APISpecification:
    """Factory helper for generating simple API specifications."""
    spec = APISpecification(
        title=f"{protocol_schema.name} API",
        version="1.0.0",
        api_style=api_style,
        security_level=security_level,
    )

    if api_style == APIStyle.REST:
        spec.add_endpoint(
            APIEndpoint(
                path="/messages",
                method="POST",
                summary="Create message",
                description="Create a new protocol message",
                request_body={
                    "required": True,
                    "content": {"application/json": {"schema": {"type": "object"}}},
                },
                responses={"201": {"description": "Created"}},
            )
        )
        spec.add_endpoint(
            APIEndpoint(
                path="/messages/{id}",
                method="GET",
                summary="Get message",
                description="Retrieve a protocol message by ID",
                parameters=[
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                responses={"200": {"description": "Success"}},
            )
        )
    elif api_style == APIStyle.GRAPHQL:
        spec.add_endpoint(
            APIEndpoint(
                path="/graphql",
                method="POST",
                summary=f"GraphQL endpoint for {protocol_schema.name}",
                description="Execute GraphQL operations",
            )
        )
    # gRPC endpoints are represented via extensions and do not expose HTTP paths.

    return spec


def validate_field_definition(field_def: Dict[str, Any]) -> bool:
    """Validate that a field definition contains the minimum required structure."""
    if not field_def.get("name"):
        return False
    field_type = field_def.get("type")
    if field_type not in {ft.value for ft in FieldType}:
        return False
    size = field_def.get("size", field_def.get("length"))
    if size is not None and size < 0:
        return False
    return True


def calculate_schema_complexity(schema: ProtocolSchema) -> float:
    """Rudimentary complexity heuristic for protocol schemas."""
    if not schema.fields:
        return 0.0

    total_fields = len(schema.fields)
    variable_fields = sum(
        1 for field in schema.fields if field.length == 0 or field.optional
    )
    unique_types = len({field.field_type for field in schema.fields})

    complexity = (
        0.4 * (variable_fields / total_fields)
        + 0.4 * (unique_types / len(FieldType))
        + 0.2 * min(1.0, total_fields / 10)
    )
    return round(min(1.0, complexity), 3)
