"""
QBITEL - Enhanced Protocol Discovery with API Generation
Extends protocol discovery with automatic API generation and semantic analysis for Translation Studio.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..discovery.protocol_discovery_orchestrator import (
    ProtocolDiscoveryOrchestrator,
    DiscoveryRequest,
    DiscoveryResult,
    DiscoveryPhase,
)
from ..discovery.enhanced_protocol_discovery_orchestrator import (
    EnhancedProtocolDiscoveryOrchestrator,
    EnhancedDiscoveryRequest,
    EnhancedDiscoveryResult,
    LLMAnalysisType,
)
from ..core.config import Config
from ..core.exceptions import ProtocolException, ModelException
from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest, LLMResponse
from ..detection.field_detector import FieldDetector, FieldBoundary, FieldType

from .models import (
    ProtocolSchema,
    ProtocolField,
    APISpecification,
    APIEndpoint,
    APIStyle,
    ProtocolFormat,
    SecurityLevel,
    TranslationRequest,
    APIGenerationResult,
    GenerationStatus,
    LLMAnalysisResult,
    TranslationException,
    create_rest_api_spec,
    create_protocol_field,
    validate_api_specification,
)
from .api_generation.schema_utils import (
    generate_examples,
    generate_graphql_assets,
    generate_grpc_assets,
)

from prometheus_client import Counter, Histogram, Gauge
import hashlib

# Metrics for API generation
API_GENERATION_COUNTER = Counter(
    "qbitel_api_generation_total",
    "Total API generation attempts",
    ["protocol_type", "api_style", "status"],
)

API_GENERATION_DURATION = Histogram(
    "qbitel_api_generation_duration_seconds",
    "API generation duration",
    ["protocol_type", "api_style"],
)

SCHEMA_EXTRACTION_DURATION = Histogram("qbitel_schema_extraction_duration_seconds", "Protocol schema extraction duration")

SEMANTIC_ANALYSIS_COUNTER = Counter(
    "qbitel_semantic_analysis_total",
    "Total semantic analyses performed",
    ["analysis_type", "status"],
)

logger = logging.getLogger(__name__)


@dataclass
class APIGenerationRequest(EnhancedDiscoveryRequest):
    """Extended discovery request for API generation."""

    target_api_style: APIStyle = APIStyle.REST
    include_authentication: bool = True
    security_level: SecurityLevel = SecurityLevel.AUTHENTICATED
    generate_openapi_spec: bool = True
    api_base_path: str = "/api/v1"
    include_examples: bool = True
    enable_field_semantics: bool = True


@dataclass
class APIGenerationDiscoveryResult(EnhancedDiscoveryResult):
    """Extended discovery result with API generation."""

    protocol_schema: Optional[ProtocolSchema] = None
    api_specification: Optional[APISpecification] = None
    field_mappings: Dict[str, str] = field(default_factory=dict)
    api_generation_metadata: Dict[str, Any] = field(default_factory=dict)
    schema_confidence: float = 0.0
    api_confidence: float = 0.0


class EnhancedProtocolDiscoveryOrchestrator(EnhancedProtocolDiscoveryOrchestrator):
    """
    Enhanced Protocol Discovery Orchestrator with API Generation capabilities.

    Extends the base enhanced orchestrator to provide automatic API generation
    from discovered protocols, including semantic field analysis and intelligent
    endpoint design.
    """

    def __init__(self, config: Config, llm_service: Optional[UnifiedLLMService] = None):
        """Initialize enhanced orchestrator with API generation capabilities."""
        super().__init__(config)
        self.llm_service = llm_service

        # API generation settings
        self.enable_api_generation = True
        self.default_api_style = APIStyle.REST
        self.api_generation_timeout = 60.0
        self.max_endpoints_per_protocol = 20
        self.enable_field_semantic_analysis = True

        # Field detector for protocol structure analysis
        self.field_detector: Optional[FieldDetector] = None

        # API generation cache
        self.api_generation_cache: Dict[str, APISpecification] = {}
        self.schema_cache: Dict[str, ProtocolSchema] = {}

        self.logger.info("Enhanced Protocol Discovery Orchestrator with API Generation initialized")

    async def initialize(self) -> None:
        """Initialize enhanced orchestrator with field detection capabilities."""
        await super().initialize()

        try:
            # Initialize field detector
            if not self.field_detector:
                self.field_detector = FieldDetector(self.config)
                await self.field_detector.initialize()
                self.logger.info("Field detector initialized for API generation")

            # Initialize LLM service if not already available
            if not self.llm_service and self.protocol_copilot:
                self.llm_service = self.protocol_copilot.llm_service

        except Exception as e:
            self.logger.error(f"Failed to initialize API generation components: {e}")
            # Continue with basic discovery capabilities
            self.enable_api_generation = False

    async def discover_and_generate_api(self, request: APIGenerationRequest) -> APIGenerationDiscoveryResult:
        """
        Main entry point for protocol discovery with automatic API generation.

        Args:
            request: Extended discovery request with API generation options

        Returns:
            Comprehensive result with protocol discovery and generated API
        """
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            self.logger.info(f"Starting discovery and API generation for {len(request.messages)} messages")

            # Phase 1: Enhanced Protocol Discovery
            discovery_result = await self.discover_protocol_enhanced(request)

            # Convert to API generation result
            api_result = APIGenerationDiscoveryResult(
                protocol_type=discovery_result.protocol_type,
                confidence=discovery_result.confidence,
                grammar=discovery_result.grammar,
                parser=discovery_result.parser,
                validation_result=discovery_result.validation_result,
                statistical_analysis=discovery_result.statistical_analysis,
                classification_details=discovery_result.classification_details,
                processing_time=discovery_result.processing_time,
                phases_completed=discovery_result.phases_completed.copy(),
                metadata=discovery_result.metadata.copy(),
                llm_analyses=discovery_result.llm_analyses.copy(),
                natural_language_summary=discovery_result.natural_language_summary,
                security_assessment=discovery_result.security_assessment,
                compliance_status=discovery_result.compliance_status,
                recommendations=discovery_result.recommendations.copy(),
            )

            # Phase 2: Protocol Schema Extraction
            if self.enable_api_generation and discovery_result.confidence >= 0.6:
                await self._extract_protocol_schema(request, api_result)

                # Phase 3: API Specification Generation
                if api_result.protocol_schema:
                    await self._generate_api_specification(request, api_result)

                    # Phase 4: Semantic Field Analysis
                    if request.enable_field_semantics:
                        await self._perform_semantic_field_analysis(request, api_result)

            # Update processing time and metadata
            api_result.processing_time = time.time() - start_time
            api_result.api_generation_metadata = {
                "generation_enabled": self.enable_api_generation,
                "target_api_style": request.target_api_style.value,
                "security_level": request.security_level.value,
                "field_semantics_enabled": request.enable_field_semantics,
                "schema_generated": api_result.protocol_schema is not None,
                "api_spec_generated": api_result.api_specification is not None,
            }

            # Update metrics
            status = "success" if api_result.api_specification else "partial"
            API_GENERATION_COUNTER.labels(
                protocol_type=api_result.protocol_type,
                api_style=request.target_api_style.value,
                status=status,
            ).inc()

            API_GENERATION_DURATION.labels(
                protocol_type=api_result.protocol_type,
                api_style=request.target_api_style.value,
            ).observe(api_result.processing_time)

            self.logger.info(
                f"API generation completed: {api_result.protocol_type} -> {request.target_api_style.value} "
                f"(schema_confidence: {api_result.schema_confidence:.2f}, "
                f"api_confidence: {api_result.api_confidence:.2f}, "
                f"time: {api_result.processing_time:.2f}s)"
            )

            return api_result

        except Exception as e:
            self.logger.error(f"API generation failed: {e}")

            # Update error metrics
            API_GENERATION_COUNTER.labels(
                protocol_type="unknown",
                api_style=request.target_api_style.value,
                status="error",
            ).inc()

            # Return partial result with error
            return APIGenerationDiscoveryResult(
                protocol_type="unknown",
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e), "phase": "api_generation"},
            )

    async def _extract_protocol_schema(self, request: APIGenerationRequest, result: APIGenerationDiscoveryResult) -> None:
        """Extract detailed protocol schema from discovery results."""
        schema_start = time.time()

        try:
            self.logger.debug("Starting protocol schema extraction")

            # Check cache first
            cache_key = self._generate_schema_cache_key(request.messages)
            if cache_key in self.schema_cache:
                result.protocol_schema = self.schema_cache[cache_key]
                result.schema_confidence = 0.9  # High confidence for cached results
                return

            # Create base protocol schema
            protocol_schema = ProtocolSchema(
                name=result.protocol_type,
                version="1.0.0",
                description=f"Auto-generated schema for {result.protocol_type} protocol",
                format=self._infer_protocol_format(request.messages[0] if request.messages else b""),
            )

            # Extract fields using field detector
            if self.field_detector and request.messages:
                detected_fields = await self._detect_protocol_fields(request.messages[0])

                for field_boundary in detected_fields:
                    protocol_field = ProtocolField(
                        name=field_boundary.field_name or f"field_{field_boundary.start_pos}",
                        field_type=field_boundary.field_type.value,
                        offset=field_boundary.start_pos,
                        length=field_boundary.end_pos - field_boundary.start_pos,
                        description=f"Auto-detected {field_boundary.field_type.value} field",
                        optional=False,
                        constraints={
                            "min_length": field_boundary.end_pos - field_boundary.start_pos,
                            "max_length": field_boundary.end_pos - field_boundary.start_pos,
                            "confidence": field_boundary.confidence,
                        },
                    )

                    if field_boundary.raw_value:
                        protocol_field.examples = [field_boundary.raw_value.hex()]

                    protocol_schema.add_field(protocol_field)

            # Enhance schema with LLM analysis if available
            if result.llm_analyses and LLMAnalysisType.FIELD_SEMANTICS in result.llm_analyses:
                semantic_analysis = result.llm_analyses[LLMAnalysisType.FIELD_SEMANTICS]
                await self._enhance_schema_with_semantics(protocol_schema, semantic_analysis)

            # Add metadata from discovery
            protocol_schema.metadata.update(
                {
                    "discovery_confidence": result.confidence,
                    "grammar_rules_count": (len(result.grammar.rules) if result.grammar else 0),
                    "statistical_features": result.statistical_analysis or {},
                    "extraction_method": "automated_discovery",
                }
            )

            # Calculate schema confidence
            field_count = len(protocol_schema.fields)
            if field_count > 0:
                result.schema_confidence = min(0.9, 0.5 + (field_count * 0.1))
            else:
                result.schema_confidence = 0.3

            # Cache and store result
            self.schema_cache[cache_key] = protocol_schema
            result.protocol_schema = protocol_schema

            schema_time = time.time() - schema_start
            SCHEMA_EXTRACTION_DURATION.observe(schema_time)

            self.logger.debug(
                f"Schema extraction completed: {field_count} fields, "
                f"confidence: {result.schema_confidence:.2f}, "
                f"time: {schema_time:.2f}s"
            )

        except Exception as e:
            self.logger.error(f"Protocol schema extraction failed: {e}")
            result.schema_confidence = 0.0
            result.protocol_schema = None

    async def _detect_protocol_fields(self, message_data: bytes) -> List[FieldBoundary]:
        """Detect protocol fields using the field detector."""
        try:
            if not self.field_detector:
                return []

            # Detect field boundaries
            field_boundaries = await self.field_detector.detect_boundaries(message_data)

            # Infer field types
            enhanced_boundaries = await self.field_detector.infer_field_types(field_boundaries)

            return enhanced_boundaries

        except Exception as e:
            self.logger.error(f"Field detection failed: {e}")
            return []

    async def _generate_api_specification(self, request: APIGenerationRequest, result: APIGenerationDiscoveryResult) -> None:
        """Generate comprehensive API specification from protocol schema."""
        try:
            self.logger.debug("Starting API specification generation")

            if not result.protocol_schema:
                self.logger.warning("No protocol schema available for API generation")
                return

            # Create base API specification
            api_spec = create_rest_api_spec(
                title=f"{result.protocol_type.title()} Protocol API",
                version="1.0.0",
                description=f"Auto-generated REST API for {result.protocol_type} protocol interaction",
            )

            api_spec.api_style = request.target_api_style
            api_spec.base_url = f"https://api.example.com{request.api_base_path}"

            # Generate endpoints based on protocol schema and style
            if request.target_api_style == APIStyle.REST:
                await self._generate_rest_endpoints(api_spec, result.protocol_schema, request)
            elif request.target_api_style == APIStyle.GRAPHQL:
                await self._generate_graphql_schema(api_spec, result.protocol_schema, request)
            elif request.target_api_style == APIStyle.GRPC:
                await self._generate_grpc_service(api_spec, result.protocol_schema, request)

            # Add security schemes based on security level
            self._add_security_schemes(api_spec, request.security_level)

            # Add protocol-specific schemas to OpenAPI components
            self._add_protocol_schemas(api_spec, result.protocol_schema)

            # Generate examples if requested
            if request.include_examples:
                await self._add_api_examples(api_spec, result.protocol_schema, request.messages)

            # Validate generated API specification
            validation_issues = validate_api_specification(api_spec)
            if validation_issues:
                self.logger.warning(f"API specification validation issues: {validation_issues}")
                result.metadata.setdefault("warnings", []).extend(validation_issues)

            # Calculate API confidence based on completeness
            endpoint_count = len(api_spec.endpoints)
            schema_count = len(api_spec.schemas)
            result.api_confidence = min(0.95, 0.6 + (endpoint_count * 0.05) + (schema_count * 0.02))

            result.api_specification = api_spec

            self.logger.debug(
                f"API specification generated: {endpoint_count} endpoints, "
                f"{schema_count} schemas, confidence: {result.api_confidence:.2f}"
            )

        except Exception as e:
            self.logger.error(f"API specification generation failed: {e}")
            result.api_confidence = 0.0
            result.api_specification = None

    async def _generate_rest_endpoints(
        self,
        api_spec: APISpecification,
        protocol_schema: ProtocolSchema,
        request: APIGenerationRequest,
    ) -> None:
        """Generate RESTful endpoints for protocol operations."""
        base_path = request.api_base_path.rstrip("/")
        resource_name = protocol_schema.name.lower().replace("_", "-")

        # Common response schemas
        error_schema = {
            "type": "object",
            "properties": {
                "error": {"type": "string"},
                "message": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"},
            },
            "required": ["error", "message"],
        }

        # Protocol message schema
        message_schema = {
            "type": "object",
            "properties": {
                "data": {"type": "string", "format": "binary"},
                "metadata": {"type": "object"},
                "timestamp": {"type": "string", "format": "date-time"},
            },
            "required": ["data"],
        }

        # 1. Parse protocol message endpoint
        parse_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/parse",
            method="POST",
            summary=f"Parse {protocol_schema.name} protocol message",
            description=f"Parse and validate a {protocol_schema.name} protocol message, extracting structured data",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string", "format": "base64"},
                                "options": {"type": "object"},
                            },
                            "required": ["message"],
                        }
                    }
                },
            },
            responses={
                "200": {
                    "description": "Successfully parsed protocol message",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "parsed_data": {"$ref": f"#/components/schemas/{protocol_schema.name}ParsedData"},
                                    "metadata": {"type": "object"},
                                    "validation_status": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "400": {
                    "description": "Invalid protocol message",
                    "content": {"application/json": {"schema": error_schema}},
                },
                "422": {
                    "description": "Message validation failed",
                    "content": {"application/json": {"schema": error_schema}},
                },
            },
            tags=[protocol_schema.name, "parsing"],
            security=[{"bearerAuth": []}] if request.include_authentication else [],
        )
        api_spec.add_endpoint(parse_endpoint)

        # 2. Generate protocol message endpoint
        generate_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/generate",
            method="POST",
            summary=f"Generate {protocol_schema.name} protocol message",
            description=f"Generate a valid {protocol_schema.name} protocol message from structured data",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "data": {"$ref": f"#/components/schemas/{protocol_schema.name}Data"},
                                "options": {"type": "object"},
                            },
                            "required": ["data"],
                        }
                    }
                },
            },
            responses={
                "201": {
                    "description": "Successfully generated protocol message",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string", "format": "base64"},
                                    "metadata": {"type": "object"},
                                },
                            }
                        }
                    },
                },
                "400": {
                    "description": "Invalid input data",
                    "content": {"application/json": {"schema": error_schema}},
                },
            },
            tags=[protocol_schema.name, "generation"],
            security=[{"bearerAuth": []}] if request.include_authentication else [],
        )
        api_spec.add_endpoint(generate_endpoint)

        # 3. Validate protocol message endpoint
        validate_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/validate",
            method="POST",
            summary=f"Validate {protocol_schema.name} protocol message",
            description=f"Validate a {protocol_schema.name} protocol message against schema rules",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string", "format": "base64"},
                                "strict": {"type": "boolean", "default": False},
                            },
                            "required": ["message"],
                        }
                    }
                },
            },
            responses={
                "200": {
                    "description": "Validation result",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "valid": {"type": "boolean"},
                                    "errors": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "warnings": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                            }
                        }
                    },
                }
            },
            tags=[protocol_schema.name, "validation"],
            security=[{"bearerAuth": []}] if request.include_authentication else [],
        )
        api_spec.add_endpoint(validate_endpoint)

        # 4. Protocol schema endpoint
        schema_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/schema",
            method="GET",
            summary=f"Get {protocol_schema.name} protocol schema",
            description=f"Retrieve the complete schema definition for {protocol_schema.name} protocol",
            responses={
                "200": {
                    "description": "Protocol schema",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "schema": {"$ref": f"#/components/schemas/{protocol_schema.name}Schema"},
                                    "version": {"type": "string"},
                                    "metadata": {"type": "object"},
                                },
                            }
                        }
                    },
                }
            },
            tags=[protocol_schema.name, "schema"],
        )
        api_spec.add_endpoint(schema_endpoint)

    async def _generate_graphql_schema(
        self,
        api_spec: APISpecification,
        protocol_schema: ProtocolSchema,
        request: APIGenerationRequest,
    ) -> None:
        """Generate GraphQL schema for protocol operations."""
        graphql_assets = generate_graphql_assets(
            protocol_schema,
            base_path=request.api_base_path,
            security_level=request.security_level,
        )

        graphql_endpoint = APIEndpoint(
            path=graphql_assets["endpoint"],
            method="POST",
            summary=f"GraphQL endpoint for {protocol_schema.name}",
            description="Execute GraphQL queries and mutations for discovered protocol",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "variables": {"type": "object"},
                                "operationName": {"type": "string"},
                            },
                            "required": ["query"],
                        },
                        "examples": {
                            "sample": {
                                "summary": "Sample GraphQL query",
                                "value": {
                                    "query": graphql_assets["operations"]["query"],
                                    "variables": {"id": "example-id"},
                                },
                            }
                        },
                    }
                },
            },
            responses={
                "200": {
                    "description": "GraphQL response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "data": {"type": "object"},
                                    "errors": {
                                        "type": "array",
                                        "items": {"type": "object"},
                                    },
                                },
                            }
                        }
                    },
                }
            },
            tags=["graphql"],
        )

        if graphql_assets.get("headers"):
            graphql_endpoint.security = [{"bearerAuth": []}]

        api_spec.add_endpoint(graphql_endpoint)
        api_spec.schemas[f"{protocol_schema.name}GraphQLSchema"] = {
            "type": "string",
            "description": f"GraphQL SDL for {protocol_schema.name}",
            "example": graphql_assets["sdl"],
        }
        api_spec.extensions.setdefault("graphql", graphql_assets)

    async def _generate_grpc_service(
        self,
        api_spec: APISpecification,
        protocol_schema: ProtocolSchema,
        request: APIGenerationRequest,
    ) -> None:
        """Generate gRPC service definition for protocol operations."""
        grpc_assets = generate_grpc_assets(protocol_schema)
        api_spec.schemas[f"{protocol_schema.name}GrpcDescriptor"] = {
            "type": "string",
            "description": f"gRPC service definition for {protocol_schema.name}",
            "example": grpc_assets["proto"],
        }
        api_spec.extensions.setdefault("grpc", grpc_assets)

    def _add_security_schemes(self, api_spec: APISpecification, security_level: SecurityLevel) -> None:
        """Add appropriate security schemes based on security level."""
        if security_level == SecurityLevel.PUBLIC:
            # No authentication required
            return

        elif security_level == SecurityLevel.AUTHENTICATED:
            api_spec.security_schemes["bearerAuth"] = {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
            }

        elif security_level in [
            SecurityLevel.AUTHORIZED,
            SecurityLevel.ENTERPRISE,
            SecurityLevel.RESTRICTED,
        ]:
            api_spec.security_schemes.update(
                {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT",
                    },
                    "apiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key",
                    },
                    "oauth2": {
                        "type": "oauth2",
                        "flows": {
                            "clientCredentials": {
                                "tokenUrl": "https://auth.example.com/oauth/token",
                                "scopes": {
                                    "protocol:read": "Read protocol data",
                                    "protocol:write": "Write protocol data",
                                    "protocol:admin": "Administrative access",
                                },
                            }
                        },
                    },
                }
            )

    def _add_protocol_schemas(self, api_spec: APISpecification, protocol_schema: ProtocolSchema) -> None:
        """Add protocol-specific schemas to OpenAPI components."""
        schema_name = protocol_schema.name

        # Data schema for the protocol
        data_properties = {}
        for field in protocol_schema.fields:
            field_schema = self._convert_field_to_openapi_schema(field)
            data_properties[field.name] = field_schema

        api_spec.schemas[f"{schema_name}Data"] = {
            "type": "object",
            "properties": data_properties,
            "description": f"Data structure for {schema_name} protocol",
        }

        # Parsed data schema (includes metadata)
        api_spec.schemas[f"{schema_name}ParsedData"] = {
            "allOf": [
                {"$ref": f"#/components/schemas/{schema_name}Data"},
                {
                    "type": "object",
                    "properties": {
                        "raw_data": {"type": "string", "format": "base64"},
                        "field_offsets": {"type": "object"},
                        "validation_results": {"type": "object"},
                    },
                },
            ]
        }

        # Schema definition
        api_spec.schemas[f"{schema_name}Schema"] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "format": {"type": "string"},
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "offset": {"type": "integer"},
                            "length": {"type": "integer"},
                            "description": {"type": "string"},
                            "optional": {"type": "boolean"},
                        },
                    },
                },
            },
        }

    def _convert_field_to_openapi_schema(self, field: ProtocolField) -> Dict[str, Any]:
        """Convert a protocol field to OpenAPI schema format."""
        field_schema = {"description": field.description or f"{field.name} field"}

        # Map field types to OpenAPI types
        if field.field_type in ["integer", "length"]:
            field_schema["type"] = "integer"
            if field.length == 1:
                field_schema.update({"minimum": 0, "maximum": 255})
            elif field.length == 2:
                field_schema.update({"minimum": 0, "maximum": 65535})
            elif field.length == 4:
                field_schema.update({"minimum": 0, "maximum": 4294967295})

        elif field.field_type == "string":
            field_schema["type"] = "string"
            if field.length > 0:
                field_schema["maxLength"] = field.length

        elif field.field_type == "binary":
            field_schema.update({"type": "string", "format": "binary"})

        elif field.field_type == "timestamp":
            field_schema.update({"type": "string", "format": "date-time"})

        elif field.field_type == "address":
            field_schema.update({"type": "string", "format": "ipv4"})

        else:
            field_schema["type"] = "string"

        # Add examples if available
        if field.examples:
            field_schema["examples"] = field.examples

        return field_schema

    async def _add_api_examples(
        self,
        api_spec: APISpecification,
        protocol_schema: ProtocolSchema,
        sample_messages: List[bytes],
    ) -> None:
        """Add realistic examples to API specification."""
        examples = generate_examples(protocol_schema, sample_messages)
        if not examples:
            return

        schema_key = f"{protocol_schema.name}Data"
        payload_schema = api_spec.schemas.get(schema_key)
        if payload_schema is not None:
            payload_schema["examples"] = [example["structured"] for example in examples]

        for endpoint in api_spec.endpoints:
            if endpoint.request_body:
                for media in endpoint.request_body.get("content", {}).values():
                    media.setdefault("examples", {})
                    media["examples"]["sample"] = {
                        "summary": examples[0]["summary"],
                        "value": examples[0]["structured"],
                    }
            for response in endpoint.responses.values():
                for media in response.get("content", {}).values():
                    media.setdefault("examples", {})
                    media["examples"]["sample"] = {
                        "summary": examples[0]["summary"],
                        "value": examples[0]["structured"],
                    }

        api_spec.extensions.setdefault("examples", examples)

    async def _perform_semantic_field_analysis(
        self, request: APIGenerationRequest, result: APIGenerationDiscoveryResult
    ) -> None:
        """Perform semantic analysis of protocol fields using LLM."""
        if not self.llm_service or not result.protocol_schema:
            return

        try:
            SEMANTIC_ANALYSIS_COUNTER.labels(analysis_type="field_semantics", status="started").inc()

            # Prepare context for LLM analysis
            fields_info = []
            for field in result.protocol_schema.fields:
                field_info = {
                    "name": field.name,
                    "type": field.field_type,
                    "offset": field.offset,
                    "length": field.length,
                    "examples": field.examples[:3],  # Limit examples
                }
                fields_info.append(field_info)

            context = {
                "protocol_name": result.protocol_type,
                "fields": fields_info,
                "protocol_format": result.protocol_schema.format.value,
                "discovery_confidence": result.confidence,
            }

            # Create LLM request for semantic analysis
            prompt = f"""
            Analyze the semantic meaning and relationships of the following protocol fields for {result.protocol_type}:
            
            Protocol: {result.protocol_type}
            Format: {result.protocol_schema.format.value}
            Fields: {json.dumps(fields_info, indent=2)}
            
            Please provide:
            1. Semantic interpretation of each field's purpose
            2. Likely data relationships between fields
            3. Suggested field naming improvements
            4. API endpoint design recommendations
            5. Security considerations for field exposure
            
            Focus on practical API design implications.
            """

            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="translation_studio",
                context=context,
                max_tokens=1500,
                temperature=0.3,
            )

            llm_response = await self.llm_service.process_request(llm_request)

            # Create semantic analysis result
            semantic_result = LLMAnalysisResult(
                protocol_understanding=llm_response.content,
                confidence_score=llm_response.confidence,
                processing_time=llm_response.processing_time,
            )

            # Parse recommendations from LLM response
            response_text = llm_response.content.lower()
            if "security" in response_text:
                semantic_result.security_considerations = ["Review field-level security based on LLM analysis"]

            if "recommend" in response_text:
                semantic_result.api_recommendations = ["Apply LLM-suggested field naming and endpoint design improvements"]

            # Store in result
            result.llm_analyses[LLMAnalysisType.FIELD_SEMANTICS] = semantic_result

            # Extract field mappings for better naming
            await self._extract_field_mappings(result, semantic_result)

            SEMANTIC_ANALYSIS_COUNTER.labels(analysis_type="field_semantics", status="success").inc()

        except Exception as e:
            self.logger.error(f"Semantic field analysis failed: {e}")
            SEMANTIC_ANALYSIS_COUNTER.labels(analysis_type="field_semantics", status="error").inc()

    async def _extract_field_mappings(self, result: APIGenerationDiscoveryResult, semantic_result: LLMAnalysisResult) -> None:
        """Extract improved field mappings from semantic analysis."""
        # Simple extraction from LLM response
        # In production, this would be more sophisticated
        result.field_mappings = {}

        if result.protocol_schema:
            for field in result.protocol_schema.fields:
                # Use existing name as default
                result.field_mappings[field.name] = field.name

    async def _enhance_schema_with_semantics(self, schema: ProtocolSchema, semantic_analysis: LLMAnalysisResult) -> None:
        """Enhance protocol schema with semantic analysis results."""
        # Add semantic insights to schema metadata
        schema.metadata.update(
            {
                "semantic_analysis": {
                    "understanding": semantic_analysis.protocol_understanding,
                    "confidence": semantic_analysis.confidence_score,
                    "recommendations": semantic_analysis.api_recommendations,
                }
            }
        )

    def _infer_protocol_format(self, sample_data: bytes) -> ProtocolFormat:
        """Infer protocol format from sample data."""
        if not sample_data:
            return ProtocolFormat.BINARY

        try:
            # Try to decode as text
            text_data = sample_data.decode("utf-8")

            if text_data.strip().startswith("{") and text_data.strip().endswith("}"):
                return ProtocolFormat.JSON
            elif text_data.strip().startswith("<") and text_data.strip().endswith(">"):
                return ProtocolFormat.XML
            elif all(32 <= b <= 126 for b in sample_data[:100]):  # Printable ASCII
                return ProtocolFormat.TEXT
            else:
                return ProtocolFormat.BINARY

        except UnicodeDecodeError:
            return ProtocolFormat.BINARY

    def _generate_schema_cache_key(self, messages: List[bytes]) -> str:
        """Generate cache key for protocol schema."""
        content = b"".join(messages[:2])  # First 2 messages
        return hashlib.sha256(content).hexdigest()[:16]

    async def get_api_generation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive API generation metrics."""
        base_metrics = await super().get_discovery_statistics()

        api_metrics = {
            "api_generation_enabled": self.enable_api_generation,
            "cached_schemas": len(self.schema_cache),
            "cached_apis": len(self.api_generation_cache),
            "field_detector_available": self.field_detector is not None,
            "llm_service_available": self.llm_service is not None,
            "supported_api_styles": [style.value for style in APIStyle],
            "max_endpoints_per_protocol": self.max_endpoints_per_protocol,
        }

        return {**base_metrics, **api_metrics}

    async def clear_api_caches(self) -> Dict[str, int]:
        """Clear API generation caches."""
        schema_count = len(self.schema_cache)
        api_count = len(self.api_generation_cache)

        self.schema_cache.clear()
        self.api_generation_cache.clear()

        return {"schemas_cleared": schema_count, "apis_cleared": api_count}
