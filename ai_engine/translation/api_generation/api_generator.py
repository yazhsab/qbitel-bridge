"""
CRONOS AI - API Generation Engine
Advanced API generation with OpenAPI 3.0 support, intelligent endpoint design, and multi-style API generation.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import yaml
import hashlib

from ...core.config import Config
from ...core.exceptions import CronosAIException
from ...llm.unified_llm_service import UnifiedLLMService, LLMRequest, LLMResponse

from ..models import (
    ProtocolSchema,
    ProtocolField,
    APISpecification,
    APIEndpoint,
    APIStyle,
    SecurityLevel,
    GeneratedCode,
    CodeLanguage,
    ProtocolFormat,
    TranslationException
)

from prometheus_client import Counter, Histogram, Gauge
import uuid

from .schema_utils import (
    generate_examples,
    generate_graphql_assets,
    generate_grpc_assets,
)

# Metrics for API generation
API_SPEC_GENERATION_COUNTER = Counter(
    'cronos_api_spec_generation_total',
    'Total API specification generations',
    ['style', 'format', 'status']
)

API_SPEC_GENERATION_DURATION = Histogram(
    'cronos_api_spec_generation_duration_seconds',
    'API specification generation duration',
    ['style']
)

OPENAPI_VALIDATION_COUNTER = Counter(
    'cronos_openapi_validation_total',
    'OpenAPI specification validations',
    ['status']
)

ENDPOINT_GENERATION_GAUGE = Gauge(
    'cronos_generated_endpoints_total',
    'Total generated API endpoints'
)

logger = logging.getLogger(__name__)


class APIGenerationException(CronosAIException):
    """API generation specific exceptions."""
    pass


@dataclass
class EndpointTemplate:
    """Template for generating API endpoints."""
    name: str
    path_pattern: str
    method: str
    summary_template: str
    description_template: str
    request_schema: Optional[Dict[str, Any]] = None
    response_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    security_required: bool = True
    operation_type: str = "data"  # data, control, metadata, diagnostic


@dataclass
class APIGenerationContext:
    """Context for API generation process."""
    protocol_schema: ProtocolSchema
    target_style: APIStyle
    security_level: SecurityLevel
    base_path: str = "/api/v1"
    include_examples: bool = True
    include_documentation: bool = True
    enable_validation: bool = True
    custom_templates: List[EndpointTemplate] = field(default_factory=list)
    llm_enhancement: bool = True
    generation_hints: Dict[str, Any] = field(default_factory=dict)


class APIGenerator:
    """
    Enterprise API Generation Engine.
    
    Generates comprehensive API specifications from protocol schemas with:
    - OpenAPI 3.0 compliance
    - Multiple API styles (REST, GraphQL, gRPC)
    - Intelligent endpoint design
    - Security integration
    - LLM-enhanced documentation
    """
    
    def __init__(self, config: Config, llm_service: Optional[UnifiedLLMService] = None):
        """Initialize API generator."""
        self.config = config
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)
        
        # Generation settings
        self.enable_llm_enhancement = True
        self.max_endpoints_per_resource = 15
        self.default_api_version = "1.0.0"
        self.include_deprecated_endpoints = False
        self.enable_batch_operations = True
        self.enable_async_operations = True
        
        # Built-in endpoint templates
        self.endpoint_templates: Dict[str, List[EndpointTemplate]] = {
            "rest": self._get_rest_templates(),
            "graphql": self._get_graphql_templates(),
            "grpc": self._get_grpc_templates()
        }
        
        # Generated API cache
        self.api_cache: Dict[str, APISpecification] = {}
        self.template_cache: Dict[str, List[EndpointTemplate]] = {}
        
        # Quality metrics
        self.generation_metrics = {
            'total_generated': 0,
            'successful_generations': 0,
            'average_endpoints_per_api': 0.0,
            'average_generation_time': 0.0
        }
        
        self.logger.info("API Generator initialized")

    async def generate_api_specification(
        self,
        context: APIGenerationContext
    ) -> APISpecification:
        """
        Generate comprehensive API specification from protocol schema.
        
        Args:
            context: API generation context with all configuration
            
        Returns:
            Complete OpenAPI specification
        """
        start_time = time.time()
        
        try:
            self.logger.info(
                f"Generating {context.target_style.value} API for protocol: {context.protocol_schema.name}"
            )
            
            # Check cache first
            cache_key = self._generate_cache_key(context)
            if cache_key in self.api_cache:
                self.logger.debug("Returning cached API specification")
                return self.api_cache[cache_key]
            
            # Create base API specification
            api_spec = APISpecification(
                title=f"{context.protocol_schema.name.title()} Protocol API",
                version=self.default_api_version,
                description=await self._generate_api_description(context),
                api_style=context.target_style,
                security_level=context.security_level,
                base_url=f"https://api.example.com{context.base_path}"
            )
            
            # Add servers configuration
            api_spec.servers = [
                {
                    "url": f"https://api.example.com{context.base_path}",
                    "description": "Production server"
                },
                {
                    "url": f"https://staging-api.example.com{context.base_path}",
                    "description": "Staging server"
                },
                {
                    "url": f"http://localhost:8000{context.base_path}",
                    "description": "Development server"
                }
            ]
            
            # Add contact and license information
            api_spec.contact = {
                "name": "API Support",
                "url": "https://example.com/support",
                "email": "api-support@example.com"
            }
            
            api_spec.license = {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
            
            # Generate security schemes
            self._add_security_schemes(api_spec, context.security_level)
            
            # Generate endpoints based on API style
            if context.target_style == APIStyle.REST:
                await self._generate_rest_endpoints(api_spec, context)
            elif context.target_style == APIStyle.GRAPHQL:
                await self._generate_graphql_schema(api_spec, context)
            elif context.target_style == APIStyle.GRPC:
                await self._generate_grpc_service(api_spec, context)
            elif context.target_style == APIStyle.WEBSOCKET:
                await self._generate_websocket_endpoints(api_spec, context)
            elif context.target_style == APIStyle.ASYNC:
                await self._generate_async_endpoints(api_spec, context)
            
            # Add protocol schemas to components
            self._add_protocol_schemas(api_spec, context.protocol_schema)
            
            # Add common response schemas
            self._add_common_schemas(api_spec)
            
            # Enhance with LLM if enabled
            if context.llm_enhancement and self.llm_service:
                await self._enhance_with_llm(api_spec, context)
            
            # Add examples if requested
            if context.include_examples:
                await self._add_comprehensive_examples(api_spec, context)
            
            # Validate generated specification
            if context.enable_validation:
                validation_result = await self._validate_openapi_spec(api_spec)
                if validation_result['errors']:
                    self.logger.warning(f"API validation issues: {validation_result['errors']}")
            
            # Cache the result
            self.api_cache[cache_key] = api_spec
            
            # Update metrics
            generation_time = time.time() - start_time
            self._update_generation_metrics(api_spec, generation_time)
            
            API_SPEC_GENERATION_COUNTER.labels(
                style=context.target_style.value,
                format=context.protocol_schema.format.value,
                status='success'
            ).inc()
            
            API_SPEC_GENERATION_DURATION.labels(
                style=context.target_style.value
            ).observe(generation_time)
            
            ENDPOINT_GENERATION_GAUGE.set(len(api_spec.endpoints))
            
            self.logger.info(
                f"API specification generated successfully: {len(api_spec.endpoints)} endpoints, "
                f"time: {generation_time:.2f}s"
            )
            
            return api_spec
            
        except Exception as e:
            self.logger.error(f"API specification generation failed: {e}")
            
            API_SPEC_GENERATION_COUNTER.labels(
                style=context.target_style.value,
                format=context.protocol_schema.format.value,
                status='error'
            ).inc()
            
            raise APIGenerationException(f"API generation failed: {e}")

    async def _generate_rest_endpoints(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext
    ) -> None:
        """Generate comprehensive REST API endpoints."""
        protocol_schema = context.protocol_schema
        resource_name = protocol_schema.name.lower().replace('_', '-')
        base_path = context.base_path.rstrip('/')
        
        # Get endpoint templates
        templates = self.endpoint_templates["rest"]
        if context.custom_templates:
            templates.extend(context.custom_templates)
        
        # Generate core CRUD endpoints
        await self._generate_crud_endpoints(api_spec, context, resource_name, base_path)
        
        # Generate protocol-specific endpoints
        await self._generate_protocol_endpoints(api_spec, context, resource_name, base_path)
        
        # Generate utility endpoints
        await self._generate_utility_endpoints(api_spec, context, resource_name, base_path)
        
        # Generate batch endpoints if enabled
        if context.enable_batch_operations:
            await self._generate_batch_endpoints(api_spec, context, resource_name, base_path)
        
        # Generate async endpoints if enabled
        if self.enable_async_operations:
            await self._generate_async_processing_endpoints(api_spec, context, resource_name, base_path)

    async def _generate_crud_endpoints(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext,
        resource_name: str,
        base_path: str
    ) -> None:
        """Generate standard CRUD endpoints."""
        protocol_name = context.protocol_schema.name
        
        # CREATE - Parse/Process protocol message
        create_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}",
            method="POST",
            summary=f"Process {protocol_name} message",
            description=f"Parse and process a {protocol_name} protocol message, returning structured data",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "string",
                                    "format": "base64",
                                    "description": f"Base64 encoded {protocol_name} message"
                                },
                                "options": {
                                    "type": "object",
                                    "properties": {
                                        "validate": {"type": "boolean", "default": True},
                                        "extract_metadata": {"type": "boolean", "default": True}
                                    }
                                }
                            },
                            "required": ["data"]
                        }
                    },
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string", "format": "binary"}
                            }
                        }
                    }
                }
            },
            responses={
                "201": {
                    "description": "Message processed successfully",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": f"#/components/schemas/{protocol_name}ProcessedMessage"}
                        }
                    }
                },
                "400": {
                    "description": "Invalid message format",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}
                },
                "422": {
                    "description": "Message validation failed",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ValidationError"}}}
                }
            },
            tags=[protocol_name, "processing"],
            security=[{"bearerAuth": []}] if context.security_level != SecurityLevel.PUBLIC else []
        )
        api_spec.add_endpoint(create_endpoint)
        
        # READ - Get processed message by ID
        read_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/{{messageId}}",
            method="GET",
            summary=f"Retrieve processed {protocol_name} message",
            description=f"Get details of a previously processed {protocol_name} message",
            parameters=[
                {
                    "name": "messageId",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string", "format": "uuid"},
                    "description": "Unique identifier of the processed message"
                },
                {
                    "name": "include",
                    "in": "query",
                    "required": False,
                    "schema": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["metadata", "raw_data", "validation_results"]}
                    },
                    "description": "Additional data to include in response"
                }
            ],
            responses={
                "200": {
                    "description": "Message details retrieved",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": f"#/components/schemas/{protocol_name}ProcessedMessage"}
                        }
                    }
                },
                "404": {
                    "description": "Message not found",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}
                }
            },
            tags=[protocol_name, "retrieval"],
            security=[{"bearerAuth": []}] if context.security_level != SecurityLevel.PUBLIC else []
        )
        api_spec.add_endpoint(read_endpoint)
        
        # LIST - Get processed messages with pagination
        list_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}",
            method="GET",
            summary=f"List processed {protocol_name} messages",
            description=f"Retrieve a paginated list of processed {protocol_name} messages",
            parameters=[
                {
                    "name": "page",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 1, "default": 1},
                    "description": "Page number for pagination"
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                    "description": "Number of items per page"
                },
                {
                    "name": "filter",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter criteria"
                },
                {
                    "name": "sort",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["created_at", "updated_at", "protocol_type"]},
                    "description": "Sort field"
                },
                {
                    "name": "order",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
                    "description": "Sort order"
                }
            ],
            responses={
                "200": {
                    "description": "List of processed messages",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "data": {
                                        "type": "array",
                                        "items": {"$ref": f"#/components/schemas/{protocol_name}ProcessedMessage"}
                                    },
                                    "pagination": {"$ref": "#/components/schemas/PaginationInfo"},
                                    "filters_applied": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            },
            tags=[protocol_name, "listing"],
            security=[{"bearerAuth": []}] if context.security_level != SecurityLevel.PUBLIC else []
        )
        api_spec.add_endpoint(list_endpoint)
        
        # DELETE - Remove processed message
        if context.security_level in [SecurityLevel.AUTHORIZED, SecurityLevel.ENTERPRISE, SecurityLevel.RESTRICTED]:
            delete_endpoint = APIEndpoint(
                path=f"{base_path}/{resource_name}/{{messageId}}",
                method="DELETE",
                summary=f"Delete processed {protocol_name} message",
                description=f"Remove a processed {protocol_name} message from the system",
                parameters=[
                    {
                        "name": "messageId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string", "format": "uuid"},
                        "description": "Unique identifier of the message to delete"
                    }
                ],
                responses={
                    "204": {"description": "Message deleted successfully"},
                    "404": {
                        "description": "Message not found",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}
                    },
                    "403": {
                        "description": "Insufficient permissions",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}
                    }
                },
                tags=[protocol_name, "management"],
                security=[{"bearerAuth": [], "oauth2": ["protocol:admin"]}]
            )
            api_spec.add_endpoint(delete_endpoint)

    async def _generate_protocol_endpoints(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext,
        resource_name: str,
        base_path: str
    ) -> None:
        """Generate protocol-specific endpoints."""
        protocol_name = context.protocol_schema.name
        
        # Validate endpoint
        validate_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/validate",
            method="POST",
            summary=f"Validate {protocol_name} message",
            description=f"Validate a {protocol_name} protocol message against schema without processing",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "data": {"type": "string", "format": "base64"},
                                "strict": {"type": "boolean", "default": False},
                                "rules": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Custom validation rules"
                                }
                            },
                            "required": ["data"]
                        }
                    }
                }
            },
            responses={
                "200": {
                    "description": "Validation result",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ValidationResult"}
                        }
                    }
                }
            },
            tags=[protocol_name, "validation"]
        )
        api_spec.add_endpoint(validate_endpoint)
        
        # Generate endpoint
        generate_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/generate",
            method="POST",
            summary=f"Generate {protocol_name} message",
            description=f"Generate a valid {protocol_name} protocol message from structured data",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "template": {"type": "string", "description": "Template to use for generation"},
                                "fields": {"$ref": f"#/components/schemas/{protocol_name}Fields"},
                                "options": {
                                    "type": "object",
                                    "properties": {
                                        "format": {"type": "string", "enum": ["binary", "hex", "base64"], "default": "base64"},
                                        "validate": {"type": "boolean", "default": True}
                                    }
                                }
                            },
                            "required": ["fields"]
                        }
                    }
                }
            },
            responses={
                "201": {
                    "description": "Message generated successfully",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"},
                                    "format": {"type": "string"},
                                    "metadata": {"type": "object"},
                                    "validation_result": {"$ref": "#/components/schemas/ValidationResult"}
                                }
                            }
                        }
                    }
                }
            },
            tags=[protocol_name, "generation"]
        )
        api_spec.add_endpoint(generate_endpoint)
        
        # Schema endpoint
        schema_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/schema",
            method="GET",
            summary=f"Get {protocol_name} protocol schema",
            description=f"Retrieve the complete schema definition for {protocol_name} protocol",
            parameters=[
                {
                    "name": "version",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Specific schema version"
                },
                {
                    "name": "format",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["json", "yaml", "openapi"], "default": "json"},
                    "description": "Response format"
                }
            ],
            responses={
                "200": {
                    "description": "Protocol schema",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": f"#/components/schemas/{protocol_name}Schema"}
                        },
                        "application/yaml": {
                            "schema": {"type": "string"}
                        }
                    }
                }
            },
            tags=[protocol_name, "schema"]
        )
        api_spec.add_endpoint(schema_endpoint)

    async def _generate_utility_endpoints(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext,
        resource_name: str,
        base_path: str
    ) -> None:
        """Generate utility and diagnostic endpoints."""
        protocol_name = context.protocol_schema.name
        
        # Statistics endpoint
        stats_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/statistics",
            method="GET",
            summary=f"Get {protocol_name} processing statistics",
            description="Retrieve processing statistics and metrics",
            parameters=[
                {
                    "name": "timeframe",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["1h", "24h", "7d", "30d"], "default": "24h"},
                    "description": "Statistics timeframe"
                }
            ],
            responses={
                "200": {
                    "description": "Processing statistics",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ProcessingStatistics"}
                        }
                    }
                }
            },
            tags=[protocol_name, "monitoring"]
        )
        api_spec.add_endpoint(stats_endpoint)
        
        # Health check endpoint
        health_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/health",
            method="GET",
            summary="Health check",
            description="Check the health status of protocol processing services",
            responses={
                "200": {
                    "description": "Service is healthy",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/HealthStatus"}
                        }
                    }
                },
                "503": {
                    "description": "Service is unhealthy",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/HealthStatus"}
                        }
                    }
                }
            },
            tags=["health", "monitoring"]
        )
        api_spec.add_endpoint(health_endpoint)

    async def _generate_batch_endpoints(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext,
        resource_name: str,
        base_path: str
    ) -> None:
        """Generate batch operation endpoints."""
        protocol_name = context.protocol_schema.name
        
        # Batch process endpoint
        batch_process_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/batch",
            method="POST",
            summary=f"Batch process {protocol_name} messages",
            description=f"Process multiple {protocol_name} protocol messages in a single request",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "messages": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "data": {"type": "string", "format": "base64"}
                                        }
                                    },
                                    "maxItems": 100
                                },
                                "options": {
                                    "type": "object",
                                    "properties": {
                                        "fail_fast": {"type": "boolean", "default": False},
                                        "parallel": {"type": "boolean", "default": True}
                                    }
                                }
                            },
                            "required": ["messages"]
                        }
                    }
                }
            },
            responses={
                "207": {
                    "description": "Multi-status response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "results": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "status": {"type": "integer"},
                                                "result": {"type": "object"},
                                                "error": {"type": "string"}
                                            }
                                        }
                                    },
                                    "summary": {
                                        "type": "object",
                                        "properties": {
                                            "total": {"type": "integer"},
                                            "successful": {"type": "integer"},
                                            "failed": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            tags=[protocol_name, "batch"],
            security=[{"bearerAuth": []}] if context.security_level != SecurityLevel.PUBLIC else []
        )
        api_spec.add_endpoint(batch_process_endpoint)

    async def _generate_async_processing_endpoints(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext,
        resource_name: str,
        base_path: str
    ) -> None:
        """Generate asynchronous processing endpoints."""
        protocol_name = context.protocol_schema.name
        
        # Submit async job endpoint
        async_submit_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/async",
            method="POST",
            summary=f"Submit {protocol_name} processing job",
            description=f"Submit a {protocol_name} processing job for asynchronous execution",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "data": {"type": "string", "format": "base64"},
                                "callback_url": {"type": "string", "format": "uri"},
                                "priority": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}
                            },
                            "required": ["data"]
                        }
                    }
                }
            },
            responses={
                "202": {
                    "description": "Job submitted successfully",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "job_id": {"type": "string", "format": "uuid"},
                                    "status": {"type": "string", "enum": ["queued"]},
                                    "estimated_completion": {"type": "string", "format": "date-time"}
                                }
                            }
                        }
                    }
                }
            },
            tags=[protocol_name, "async"],
            security=[{"bearerAuth": []}] if context.security_level != SecurityLevel.PUBLIC else []
        )
        api_spec.add_endpoint(async_submit_endpoint)
        
        # Get async job status endpoint
        async_status_endpoint = APIEndpoint(
            path=f"{base_path}/{resource_name}/async/{{jobId}}",
            method="GET",
            summary="Get async job status",
            description="Check the status of an asynchronous processing job",
            parameters=[
                {
                    "name": "jobId",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string", "format": "uuid"}
                }
            ],
            responses={
                "200": {
                    "description": "Job status",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AsyncJobStatus"}
                        }
                    }
                }
            },
            tags=[protocol_name, "async"]
        )
        api_spec.add_endpoint(async_status_endpoint)

    async def _generate_graphql_schema(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext
    ) -> None:
        """Generate GraphQL schema and resolvers."""
        graphql_assets = generate_graphql_assets(
            context.protocol_schema,
            base_path=context.base_path,
            security_level=context.security_level,
        )

        graphql_endpoint = APIEndpoint(
            path=graphql_assets["endpoint"],
            method="POST",
            summary=f"GraphQL endpoint for {context.protocol_schema.name}",
            description="Execute GraphQL queries and mutations for protocol operations",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "variables": {"type": "object"},
                                "operationName": {"type": "string"}
                            },
                            "required": ["query"]
                        },
                        "examples": {
                            "sample": {
                                "summary": "Sample GraphQL query",
                                "value": {
                                    "query": graphql_assets["operations"]["query"],
                                    "variables": {"id": "example-id"}
                                }
                            }
                        }
                    }
                }
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
                                    "errors": {"type": "array", "items": {"type": "object"}}
                                }
                            }
                        }
                    }
                }
            },
            tags=["graphql"]
        )

        if graphql_assets.get("headers"):
            graphql_endpoint.security = [
                {"bearerAuth": []}
            ]

        api_spec.add_endpoint(graphql_endpoint)
        api_spec.schemas[f"{context.protocol_schema.name}GraphQLSchema"] = {
            "type": "string",
            "description": f"GraphQL SDL for {context.protocol_schema.name}",
            "example": graphql_assets["sdl"],
        }
        api_spec.extensions.setdefault("graphql", graphql_assets)

    async def _generate_grpc_service(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext
    ) -> None:
        """Generate gRPC service definition."""
        grpc_assets = generate_grpc_assets(context.protocol_schema)
        api_spec.schemas[f"{context.protocol_schema.name}GrpcDescriptor"] = {
            "type": "string",
            "description": f"gRPC service definition for {context.protocol_schema.name}",
            "example": grpc_assets["proto"],
        }
        api_spec.extensions.setdefault("grpc", grpc_assets)

    async def _generate_websocket_endpoints(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext
    ) -> None:
        """Generate WebSocket endpoints for real-time communication."""
        protocol_name = context.protocol_schema.name
        
        # WebSocket connection endpoint
        ws_endpoint = APIEndpoint(
            path=f"{context.base_path}/{protocol_name.lower()}/ws",
            method="GET",
            summary=f"WebSocket connection for {protocol_name}",
            description=f"Establish WebSocket connection for real-time {protocol_name} protocol processing",
            parameters=[
                {
                    "name": "Connection",
                    "in": "header",
                    "required": True,
                    "schema": {"type": "string", "enum": ["Upgrade"]}
                },
                {
                    "name": "Upgrade",
                    "in": "header",
                    "required": True,
                    "schema": {"type": "string", "enum": ["websocket"]}
                }
            ],
            responses={
                "101": {"description": "Switching Protocols"},
                "400": {"description": "Bad Request"}
            },
            tags=[protocol_name, "websocket"]
        )
        api_spec.add_endpoint(ws_endpoint)

    async def _generate_async_endpoints(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext
    ) -> None:
        """Generate async/event-driven endpoints."""
        # Generate webhook endpoints for async notifications
        await self._generate_webhook_endpoints(api_spec, context)
        # Generate event subscription endpoints
        await self._generate_event_endpoints(api_spec, context)

    async def _generate_webhook_endpoints(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext
    ) -> None:
        """Generate webhook endpoints for notifications."""
        protocol_name = context.protocol_schema.name
        
        # Register webhook endpoint
        webhook_register_endpoint = APIEndpoint(
            path=f"{context.base_path}/webhooks",
            method="POST",
            summary="Register webhook",
            description="Register a webhook for protocol processing notifications",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "format": "uri"},
                                "events": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": ["processed", "validated", "error"]}
                                },
                                "secret": {"type": "string", "minLength": 16}
                            },
                            "required": ["url", "events"]
                        }
                    }
                }
            },
            responses={
                "201": {
                    "description": "Webhook registered",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "webhook_id": {"type": "string", "format": "uuid"},
                                    "url": {"type": "string"},
                                    "events": {"type": "array", "items": {"type": "string"}}
                                }
                            }
                        }
                    }
                }
            },
            tags=["webhooks"],
            security=[{"bearerAuth": []}]
        )
        api_spec.add_endpoint(webhook_register_endpoint)

    async def _generate_event_endpoints(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext
    ) -> None:
        """Generate event subscription endpoints."""
        # Server-Sent Events endpoint
        sse_endpoint = APIEndpoint(
            path=f"{context.base_path}/events",
            method="GET",
            summary="Server-Sent Events stream",
            description="Subscribe to real-time protocol processing events",
            parameters=[
                {
                    "name": "filter",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Event filter criteria"
                }
            ],
            responses={
                "200": {
                    "description": "Event stream",
                    "content": {
                        "text/event-stream": {
                            "schema": {"type": "string"}
                        }
                    }
                }
            },
            tags=["events"]
        )
        api_spec.add_endpoint(sse_endpoint)

    def _add_security_schemes(
        self,
        api_spec: APISpecification,
        security_level: SecurityLevel
    ) -> None:
        """Add comprehensive security schemes based on security level."""
        if security_level == SecurityLevel.PUBLIC:
            return
        
        base_schemes = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token authentication"
            }
        }
        
        if security_level in [SecurityLevel.AUTHENTICATED, SecurityLevel.AUTHORIZED]:
            base_schemes["apiKeyAuth"] = {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key authentication"
            }
        
        if security_level in [SecurityLevel.AUTHORIZED, SecurityLevel.ENTERPRISE, SecurityLevel.RESTRICTED]:
            base_schemes.update({
                "oauth2": {
                    "type": "oauth2",
                    "description": "OAuth 2.0 with PKCE",
                    "flows": {
                        "authorizationCode": {
                            "authorizationUrl": "https://auth.example.com/oauth/authorize",
                            "tokenUrl": "https://auth.example.com/oauth/token",
                            "scopes": {
                                "protocol:read": "Read protocol data",
                                "protocol:write": "Process protocol messages",
                                "protocol:admin": "Administrative operations"
                            }
                        },
                        "clientCredentials": {
                            "tokenUrl": "https://auth.example.com/oauth/token",
                            "scopes": {
                                "protocol:service": "Service-to-service access"
                            }
                        }
                    }
                }
            })
        
        if security_level in [SecurityLevel.ENTERPRISE, SecurityLevel.RESTRICTED]:
            base_schemes["mutualTLS"] = {
                "type": "mutualTLS",
                "description": "Mutual TLS authentication"
            }
        
        api_spec.security_schemes.update(base_schemes)

    def _add_protocol_schemas(
        self,
        api_spec: APISpecification,
        protocol_schema: ProtocolSchema
    ) -> None:
        """Add protocol-specific schemas to OpenAPI components."""
        schema_name = protocol_schema.name
        
        # Field schemas
        fields_properties = {}
        for field in protocol_schema.fields:
            field_schema = self._convert_field_to_schema(field)
            fields_properties[field.name] = field_schema
        
        # Core schemas
        api_spec.schemas.update({
            f"{schema_name}Fields": {
                "type": "object",
                "properties": fields_properties,
                "description": f"Field definitions for {schema_name} protocol"
            },
            f"{schema_name}ProcessedMessage": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "format": "uuid"},
                    "fields": {"$ref": f"#/components/schemas/{schema_name}Fields"},
                    "raw_data": {"type": "string", "format": "base64"},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "protocol_version": {"type": "string"},
                            "processed_at": {"type": "string", "format": "date-time"},
                            "processing_time_ms": {"type": "number"},
                            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    },
                    "validation_result": {"$ref": "#/components/schemas/ValidationResult"}
                },
                "required": ["id", "fields", "metadata"]
            },
            f"{schema_name}Schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "description": {"type": "string"},
                    "format": {"type": "string", "enum": [f.value for f in ProtocolFormat]},
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
                                "optional": {"type": "boolean"}
                            }
                        }
                    }
                }
            }
        })

    def _add_common_schemas(self, api_spec: APISpecification) -> None:
        """Add common schemas used across endpoints."""
        common_schemas = {
            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "message": {"type": "string"},
                    "code": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "request_id": {"type": "string"}
                },
                "required": ["error", "message"]
            },
            "ValidationError": {
                "allOf": [
                    {"$ref": "#/components/schemas/Error"},
                    {
                        "type": "object",
                        "properties": {
                            "validation_errors": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field": {"type": "string"},
                                        "error": {"type": "string"},
                                        "expected": {"type": "string"},
                                        "actual": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                ]
            },
            "ValidationResult": {
                "type": "object",
                "properties": {
                    "valid": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "errors": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "warnings": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "field_validations": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "valid": {"type": "boolean"},
                                "message": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "PaginationInfo": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "total": {"type": "integer"},
                    "pages": {"type": "integer"},
                    "has_next": {"type": "boolean"},
                    "has_prev": {"type": "boolean"}
                }
            },
            "ProcessingStatistics": {
                "type": "object",
                "properties": {
                    "total_processed": {"type": "integer"},
                    "successful": {"type": "integer"},
                    "failed": {"type": "integer"},
                    "average_processing_time_ms": {"type": "number"},
                    "throughput_per_minute": {"type": "number"},
                    "error_rate": {"type": "number"},
                    "timeframe": {"type": "string"}
                }
            },
            "HealthStatus": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["healthy", "unhealthy", "degraded"]},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "version": {"type": "string"},
                    "services": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "message": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "AsyncJobStatus": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "format": "uuid"},
                    "status": {"type": "string", "enum": ["queued", "processing", "completed", "failed"]},
                    "progress": {"type": "integer", "minimum": 0, "maximum": 100},
                    "result": {"type": "object"},
                    "error": {"type": "string"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "updated_at": {"type": "string", "format": "date-time"}
                }
            }
        }
        
        api_spec.schemas.update(common_schemas)

    def _convert_field_to_schema(self, field: ProtocolField) -> Dict[str, Any]:
        """Convert protocol field to OpenAPI schema."""
        schema = {
            "description": field.description or f"{field.name} field",
            "x-field-offset": field.offset,
            "x-field-length": field.length
        }
        
        # Type mapping
        if field.field_type in ["integer", "length"]:
            schema["type"] = "integer"
            if field.length <= 1:
                schema.update({"minimum": 0, "maximum": 255})
            elif field.length <= 2:
                schema.update({"minimum": 0, "maximum": 65535})
            elif field.length <= 4:
                schema.update({"minimum": 0, "maximum": 4294967295})
        elif field.field_type == "string":
            schema["type"] = "string"
            if field.length > 0:
                schema["maxLength"] = field.length
        elif field.field_type == "binary":
            schema.update({"type": "string", "format": "binary"})
        elif field.field_type == "timestamp":
            schema.update({"type": "string", "format": "date-time"})
        elif field.field_type == "address":
            schema.update({"type": "string", "pattern": r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"})
        else:
            schema["type"] = "string"
        
        # Add constraints
        if field.constraints:
            for key, value in field.constraints.items():
                if key in ["minimum", "maximum", "minLength", "maxLength", "pattern"]:
                    schema[key] = value
        
        # Add examples
        if field.examples:
            schema["examples"] = field.examples[:3]
        
        return schema

    async def _generate_api_description(self, context: APIGenerationContext) -> str:
        """Generate comprehensive API description."""
        if context.llm_enhancement and self.llm_service:
            try:
                prompt = f"""
                Generate a comprehensive API description for a {context.protocol_schema.name} protocol API.
                
                Protocol: {context.protocol_schema.name}
                Format: {context.protocol_schema.format.value}
                Fields: {len(context.protocol_schema.fields)} fields
                Security Level: {context.security_level.value}
                API Style: {context.target_style.value}
                
                Include:
                1. Brief overview of the protocol
                2. Key capabilities of the API
                3. Primary use cases
                4. Security considerations
                5. Integration notes
                
                Keep it professional and concise (2-3 paragraphs).
                """
                
                llm_request = LLMRequest(
                    prompt=prompt,
                    feature_domain="translation_studio",
                    max_tokens=400,
                    temperature=0.3
                )
                
                response = await self.llm_service.process_request(llm_request)
                return response.content.strip()
                
            except Exception as e:
                self.logger.warning(f"LLM description generation failed: {e}")
        
        # Fallback to template-based description
        return f"""
        This API provides comprehensive {context.protocol_schema.name} protocol processing capabilities,
        including message parsing, validation, generation, and real-time processing.
        
        The API supports {context.target_style.value.upper()} operations with {context.security_level.value} security level,
        making it suitable for enterprise integration and development workflows.
        
        Key features include batch processing, asynchronous operations, webhooks, and comprehensive
        monitoring capabilities for production deployments.
        """.strip()

    async def _enhance_with_llm(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext
    ) -> None:
        """Enhance API specification with LLM-generated improvements."""
        if not self.llm_service:
            return
        
        try:
            # Enhance endpoint descriptions
            for endpoint in api_spec.endpoints:
                if len(endpoint.description) < 100:  # Enhance short descriptions
                    enhanced_desc = await self._enhance_endpoint_description(endpoint, context)
                    if enhanced_desc:
                        endpoint.description = enhanced_desc
                        
        except Exception as e:
            self.logger.warning(f"LLM enhancement failed: {e}")

    async def _enhance_endpoint_description(
        self,
        endpoint: APIEndpoint,
        context: APIGenerationContext
    ) -> Optional[str]:
        """Enhance individual endpoint description with LLM."""
        try:
            prompt = f"""
            Enhance this API endpoint description:
            
            Method: {endpoint.method}
            Path: {endpoint.path}
            Summary: {endpoint.summary}
            Current Description: {endpoint.description}
            
            Protocol: {context.protocol_schema.name}
            
            Provide a detailed, professional description (1-2 sentences) that explains:
            1. What the endpoint does
            2. When it should be used
            3. Any important considerations
            
            Keep it concise but informative.
            """
            
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="translation_studio",
                max_tokens=150,
                temperature=0.2
            )
            
            response = await self.llm_service.process_request(llm_request)
            return response.content.strip()
            
        except Exception:
            return None

    async def _add_comprehensive_examples(
        self,
        api_spec: APISpecification,
        context: APIGenerationContext
    ) -> None:
        """Add comprehensive examples to API specification."""
        examples = generate_examples(context.protocol_schema)
        if not examples:
            return

        schema_key = f"{context.protocol_schema.name}Data"
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

    async def _validate_openapi_spec(self, api_spec: APISpecification) -> Dict[str, Any]:
        """Validate generated OpenAPI specification."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            openapi_dict = api_spec.to_openapi_dict()
            
            # Basic structure validation
            required_fields = ['openapi', 'info', 'paths']
            for field in required_fields:
                if field not in openapi_dict:
                    validation_result['errors'].append(f"Missing required field: {field}")
                    validation_result['valid'] = False
            
            # Validate info section
            info = openapi_dict.get('info', {})
            if not info.get('title'):
                validation_result['errors'].append("API title is required")
                validation_result['valid'] = False
                
            if not info.get('version'):
                validation_result['errors'].append("API version is required")
                validation_result['valid'] = False
            
            # Validate paths
            paths = openapi_dict.get('paths', {})
            if not paths:
                validation_result['warnings'].append("No API paths defined")
            
            # Update metrics
            status = 'valid' if validation_result['valid'] else 'invalid'
            OPENAPI_VALIDATION_COUNTER.labels(status=status).inc()
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            OPENAPI_VALIDATION_COUNTER.labels(status='error').inc()
        
        return validation_result

    def _generate_cache_key(self, context: APIGenerationContext) -> str:
        """Generate cache key for API specification."""
        key_data = {
            'schema_id': context.protocol_schema.schema_id,
            'target_style': context.target_style.value,
            'security_level': context.security_level.value,
            'base_path': context.base_path,
            'llm_enhancement': context.llm_enhancement
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _update_generation_metrics(
        self,
        api_spec: APISpecification,
        generation_time: float
    ) -> None:
        """Update generation metrics."""
        self.generation_metrics['total_generated'] += 1
        self.generation_metrics['successful_generations'] += 1
        
        # Update endpoint average
        endpoint_count = len(api_spec.endpoints)
        total_generated = self.generation_metrics['total_generated']
        current_avg = self.generation_metrics['average_endpoints_per_api']
        self.generation_metrics['average_endpoints_per_api'] = (
            (current_avg * (total_generated - 1) + endpoint_count) / total_generated
        )
        
        # Update time average
        current_time_avg = self.generation_metrics['average_generation_time']
        self.generation_metrics['average_generation_time'] = (
            (current_time_avg * (total_generated - 1) + generation_time) / total_generated
        )

    def _get_rest_templates(self) -> List[EndpointTemplate]:
        """Get built-in REST endpoint templates."""
        return [
            EndpointTemplate(
                name="create",
                path_pattern="/{resource}",
                method="POST",
                summary_template="Create new {resource}",
                description_template="Create a new {resource} instance",
                operation_type="data"
            ),
            EndpointTemplate(
                name="list",
                path_pattern="/{resource}",
                method="GET",
                summary_template="List {resource}",
                description_template="Retrieve a list of {resource} instances",
                operation_type="data"
            ),
            EndpointTemplate(
                name="get",
                path_pattern="/{resource}/{{id}}",
                method="GET",
                summary_template="Get {resource}",
                description_template="Retrieve a specific {resource} instance",
                operation_type="data"
            ),
            EndpointTemplate(
                name="update",
                path_pattern="/{resource}/{{id}}",
                method="PUT",
                summary_template="Update {resource}",
                description_template="Update an existing {resource} instance",
                operation_type="data"
            ),
            EndpointTemplate(
                name="delete",
                path_pattern="/{resource}/{{id}}",
                method="DELETE",
                summary_template="Delete {resource}",
                description_template="Delete a {resource} instance",
                operation_type="control",
                security_required=True
            )
        ]

    def _get_graphql_templates(self) -> List[EndpointTemplate]:
        """Get built-in GraphQL templates."""
        return [
            EndpointTemplate(
                name="graphql",
                path_pattern="/graphql",
                method="POST",
                summary_template="GraphQL endpoint",
                description_template="GraphQL endpoint for {resource} operations",
                operation_type="data"
            )
        ]

    def _get_grpc_templates(self) -> List[EndpointTemplate]:
        """Get built-in gRPC templates."""
        return [
            EndpointTemplate(
                name="grpc_service",
                path_pattern="/grpc",
                method="POST",
                summary_template="gRPC service",
                description_template="gRPC service for {resource} operations",
                operation_type="data"
            )
        ]

    async def export_openapi_spec(
        self,
        api_spec: APISpecification,
        format: str = "json",
        output_path: Optional[Path] = None
    ) -> str:
        """Export OpenAPI specification to file or string."""
        openapi_dict = api_spec.to_openapi_dict()
        
        if format.lower() == "yaml":
            content = yaml.dump(openapi_dict, default_flow_style=False, sort_keys=False)
        else:
            content = json.dumps(openapi_dict, indent=2)
        
        if output_path:
            output_path.write_text(content, encoding='utf-8')
            self.logger.info(f"OpenAPI specification exported to {output_path}")
        
        return content

    def get_generation_metrics(self) -> Dict[str, Any]:
        """Get API generation metrics."""
        return {
            **self.generation_metrics,
            'cached_specifications': len(self.api_cache),
            'supported_styles': [style.value for style in APIStyle],
            'endpoint_templates': sum(len(templates) for templates in self.endpoint_templates.values())
        }

    async def clear_cache(self) -> int:
        """Clear API generation cache."""
        count = len(self.api_cache)
        self.api_cache.clear()
        self.template_cache.clear()
        return count
