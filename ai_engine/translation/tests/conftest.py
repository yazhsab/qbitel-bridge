"""
CRONOS AI - Translation Studio Test Configuration
Enterprise-grade test fixtures and configuration for translation studio tests.
"""

import pytest
import asyncio
import tempfile
import shutil
import json
import base64
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
from pathlib import Path
import uuid
from datetime import datetime, timezone

from ai_engine.translation.models import (
    ProtocolSchema,
    APISpecification,
    APIEndpoint,
    GeneratedSDK,
    TranslationRequest,
    CodeLanguage,
    APIStyle,
    SecurityLevel,
    FieldType,
    TranslationMode,
    QualityLevel,
)
from ai_engine.translation.models import TranslationResult
from ai_engine.translation.exceptions import ErrorContext, create_error_context
from ai_engine.translation.logging import LogContext, LogComponent, LogOperation


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        'translation': {
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'supported_languages': ['python', 'typescript', 'javascript', 'go'],
            'api_styles': ['rest', 'graphql', 'grpc'],
            'cache_ttl': 3600,
            'confidence_threshold': 0.7,
            'max_concurrent_operations': 10
        },
        'llm': {
            'provider': 'openai',
            'model': 'gpt-4',
            'max_tokens': 4000,
            'temperature': 0.1
        },
        'rag': {
            'collection_name': 'translation_patterns',
            'similarity_threshold': 0.8,
            'max_results': 10
        },
        'security': {
            'auth_enabled': True,
            'rate_limit': 100,
            'max_request_size': 10 * 1024 * 1024
        }
    }


@pytest.fixture
def sample_protocol_messages():
    """Sample protocol messages for testing."""
    return [
        b'\x01\x00\x04\x00Hello',
        b'\x01\x01\x05\x00World',
        b'\x02\x00\x06\x00TestMsg',
        b'\x03\x00\x08\x00Complete'
    ]


@pytest.fixture
def sample_protocol_schema():
    """Sample protocol schema for testing."""
    return ProtocolSchema(
        name="TestProtocol",
        version="1.0",
        description="A test protocol for validation",
        fields=[
            {
                'name': 'message_type',
                'type': FieldType.INTEGER.value,
                'size': 1,
                'description': 'Message type identifier'
            },
            {
                'name': 'flags',
                'type': FieldType.INTEGER.value,
                'size': 1,
                'description': 'Message flags'
            },
            {
                'name': 'length',
                'type': FieldType.INTEGER.value,
                'size': 2,
                'description': 'Message length'
            },
            {
                'name': 'payload',
                'type': FieldType.STRING.value,
                'variable_length': True,
                'description': 'Message payload'
            }
        ],
        semantic_info={
            'domain': 'messaging',
            'purpose': 'Simple message exchange',
            'complexity': 'low'
        }
    )


@pytest.fixture
def sample_api_specification():
    """Sample API specification for testing."""
    return APISpecification(
        title="Test Protocol API",
        version="1.0.0",
        description="API generated from test protocol",
        base_url="https://api.test.com",
        api_style=APIStyle.REST,
        security_level=SecurityLevel.AUTHENTICATED,
        endpoints=[
            APIEndpoint(
                path='/messages',
                method='POST',
                summary='Send a new message',
                description='Send a message using the test protocol',
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": '#/components/schemas/Message'}
                        }
                    }
                },
                responses={
                    "201": {
                        "description": "Message created",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": '#/components/schemas/Message'}
                            }
                        }
                    }
                }
            ),
            APIEndpoint(
                path='/messages/{id}',
                method='GET',
                summary='Get message by ID',
                description='Retrieve a specific message',
                parameters=[
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                responses={
                    "200": {
                        "description": "Message details",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": '#/components/schemas/Message'}
                            }
                        }
                    }
                }
            )
        ],
        schemas={
            'Message': {
                'type': 'object',
                'properties': {
                    'message_type': {'type': 'integer'},
                    'flags': {'type': 'integer'},
                    'length': {'type': 'integer'},
                    'payload': {'type': 'string'}
                },
                'required': ['message_type', 'payload']
            }
        },
        extensions={}
    )


@pytest.fixture
def sample_generated_sdk():
    """Sample generated SDK for testing."""
    return GeneratedSDK(
        sdk_id=str(uuid.uuid4()),
        name="test-protocol-python-sdk",
        language=CodeLanguage.PYTHON,
        version="1.0.0",
        description="Python SDK for test protocol",
        source_files={
            'client.py': 'class TestProtocolClient:\n    pass',
            'models.py': 'class Message:\n    pass',
            '__init__.py': 'from .client import TestProtocolClient'
        },
        test_files={
            'test_client.py': 'def test_client():\n    pass'
        },
        config_files={
            'setup.py': 'from setuptools import setup\nsetup(name="test-protocol-sdk")',
            'requirements.txt': 'requests>=2.25.0\npydantic>=1.8.0'
        },
        documentation_files={
            'README.md': '# Test Protocol SDK\n\nUsage instructions...',
            'API.md': '# API Documentation\n\nEndpoint details...'
        }
    )


@pytest.fixture
def sample_translation_request():
    """Sample translation request for testing."""
    return TranslationRequest(
        request_id=str(uuid.uuid4()),
        source_protocol="HTTP",
        target_protocol="WebSocket",
        data=b'{"message": "hello", "type": "text"}',
        metadata={'user_id': 'test-user', 'session_id': 'test-session'}
    )


@pytest.fixture
def sample_translation_result():
    """Sample translation result for testing."""
    return TranslationResult(
        translation_id=str(uuid.uuid4()),
        source_protocol="HTTP",
        target_protocol="WebSocket",
        translated_data=b'{"op": "message", "d": {"content": "hello", "type": "text"}}',
        confidence=0.95,
        processing_time=0.15,
        validation_errors=[],
        warnings=[]
    )


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    mock_service = AsyncMock()
    
    # Mock semantic analysis response
    mock_service.analyze_semantic_fields.return_value = {
        'field_meanings': {
            'message_type': 'Identifies the type of message being sent',
            'payload': 'The actual message content'
        },
        'relationships': [
            {'field1': 'message_type', 'field2': 'payload', 'relationship': 'determines_format'}
        ],
        'confidence': 0.9
    }
    
    # Mock API generation response
    mock_service.generate_api_specification.return_value = {
        'openapi': '3.0.0',
        'info': {'title': 'Test API', 'version': '1.0.0'},
        'paths': {
            '/messages': {
                'post': {
                    'summary': 'Send message',
                    'operationId': 'sendMessage'
                }
            }
        }
    }
    
    # Mock code generation response
    mock_service.generate_code.return_value = {
        'code': 'class TestClient:\n    def __init__(self):\n        pass',
        'quality_score': 0.85,
        'documentation': 'Generated client for test protocol'
    }
    
    return mock_service


@pytest.fixture
def mock_rag_engine():
    """Mock RAG engine for testing."""
    mock_engine = AsyncMock()
    
    # Mock pattern search results
    mock_engine.query_protocol_patterns.return_value = Mock(
        documents=[
            Mock(id='pattern1', content='HTTP pattern example', metadata={'type': 'http'}),
            Mock(id='pattern2', content='WebSocket pattern example', metadata={'type': 'websocket'})
        ],
        similarity_scores=[0.9, 0.8],
        total_results=2,
        processing_time=0.05
    )
    
    # Mock code templates
    mock_engine.get_code_generation_templates.return_value = Mock(
        documents=[
            Mock(id='template1', content='Python client template', metadata={'language': 'python'}),
            Mock(id='template2', content='TypeScript client template', metadata={'language': 'typescript'})
        ],
        similarity_scores=[0.95, 0.90],
        total_results=2
    )
    
    # Mock best practices
    mock_engine.get_translation_best_practices.return_value = Mock(
        documents=[
            Mock(id='practice1', content='Always validate input', metadata={'category': 'validation'}),
            Mock(id='practice2', content='Handle errors gracefully', metadata={'category': 'error_handling'})
        ],
        similarity_scores=[0.88, 0.82],
        total_results=2
    )
    
    return mock_engine


@pytest.fixture
def mock_discovery_orchestrator():
    """Mock enhanced discovery orchestrator for testing."""
    mock_orchestrator = AsyncMock()
    
    mock_orchestrator.discover_and_generate_api.return_value = Mock(
        protocol_type="TestProtocol",
        confidence=0.9,
        api_specification=Mock(),
        generated_sdks=[],
        processing_time=1.5,
        recommendations=['Consider adding validation', 'Implement rate limiting'],
        natural_language_summary="This is a simple messaging protocol"
    )
    
    return mock_orchestrator


@pytest.fixture
def mock_api_generator():
    """Mock API generator for testing."""
    mock_generator = AsyncMock()
    
    mock_generator.generate_api_specification.return_value = Mock(
        spec_id=str(uuid.uuid4()),
        title="Generated API",
        version="1.0.0",
        endpoints=[],
        schemas={}
    )
    
    mock_generator.get_generation_metrics.return_value = {
        'total_generated': 1,
        'average_confidence': 0.9,
        'processing_time': 1.2
    }
    
    return mock_generator


@pytest.fixture
def mock_code_generator():
    """Mock code generator for testing."""
    mock_generator = AsyncMock()
    
    mock_generator.generate_sdk.return_value = Mock(
        sdk_id=str(uuid.uuid4()),
        name="test-sdk",
        language=CodeLanguage.PYTHON,
        version="1.0.0",
        source_files={'client.py': 'test code'},
        test_files={'test_client.py': 'test code'},
        config_files={'setup.py': 'setup code'}
    )
    
    mock_generator.generate_multiple_sdks.return_value = {
        CodeLanguage.PYTHON: Mock(sdk_id=str(uuid.uuid4()), name="python-sdk"),
        CodeLanguage.TYPESCRIPT: Mock(sdk_id=str(uuid.uuid4()), name="typescript-sdk")
    }
    
    mock_generator.get_generation_metrics.return_value = {
        'total_generated': 2,
        'languages': ['python', 'typescript'],
        'average_quality': 0.85
    }
    
    return mock_generator


@pytest.fixture
def mock_protocol_bridge():
    """Mock protocol bridge for testing."""
    mock_bridge = AsyncMock()
    
    mock_bridge.translate_protocol.return_value = Mock(
        translation_id=str(uuid.uuid4()),
        source_protocol="HTTP",
        target_protocol="WebSocket",
        translated_data=b'translated data',
        confidence=0.9,
        processing_time=0.5,
        validation_errors=[],
        warnings=[]
    )
    
    mock_bridge.batch_translate.return_value = [
        Mock(translation_id=str(uuid.uuid4()), confidence=0.9, validation_errors=[]),
        Mock(translation_id=str(uuid.uuid4()), confidence=0.85, validation_errors=[])
    ]
    
    mock_bridge.create_streaming_connection.return_value = str(uuid.uuid4())
    
    mock_bridge.get_connection_status.return_value = {
        'connection_id': str(uuid.uuid4()),
        'status': 'active',
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    mock_bridge.get_bridge_metrics.return_value = {
        'total_translations': 10,
        'active_connections': 2,
        'average_confidence': 0.88
    }
    
    return mock_bridge


@pytest.fixture
def log_context():
    """Sample log context for testing."""
    return LogContext(
        request_id=str(uuid.uuid4()),
        user_id='test-user',
        session_id='test-session',
        component=LogComponent.API_ENDPOINTS,
        operation=LogOperation.REQUEST_PROCESSING
    )


@pytest.fixture
def error_context():
    """Sample error context for testing."""
    return create_error_context(
        request_id=str(uuid.uuid4()),
        user_id='test-user',
        component='api_endpoints',
        operation='discover_protocol'
    )


@pytest.fixture
def mock_cache():
    """Mock cache service for testing."""
    cache_data = {}
    
    mock_cache = Mock()
    mock_cache.get.side_effect = lambda key: cache_data.get(key)
    mock_cache.set.side_effect = lambda key, value, ttl=None: cache_data.update({key: value})
    mock_cache.delete.side_effect = lambda key: cache_data.pop(key, None)
    mock_cache.exists.side_effect = lambda key: key in cache_data
    mock_cache.clear.side_effect = lambda: cache_data.clear()
    
    return mock_cache


@pytest.fixture
def mock_metrics():
    """Mock Prometheus metrics for testing."""
    mock_metrics = Mock()
    mock_metrics.inc = Mock()
    mock_metrics.dec = Mock()
    mock_metrics.observe = Mock()
    mock_metrics.set = Mock()
    return mock_metrics


@pytest.fixture
def base64_encoded_messages():
    """Base64 encoded protocol messages for API testing."""
    messages = [
        b'\x01\x00\x04\x00Hello',
        b'\x01\x01\x05\x00World',
        b'\x02\x00\x06\x00TestMsg'
    ]
    return [base64.b64encode(msg).decode('utf-8') for msg in messages]


@pytest.fixture
def api_test_client():
    """Test client for API endpoints."""
    from fastapi.testclient import TestClient
    from ai_engine.translation.api_endpoints import router
    
    # Create a test app with the translation router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    return TestClient(app)


@pytest.fixture
def mock_current_user():
    """Mock current user for authentication testing."""
    return {
        'user_id': 'test-user',
        'username': 'testuser',
        'roles': ['user', 'translator'],
        'permissions': ['read', 'write', 'translate']
    }


# Helper functions for tests

def create_test_protocol_data() -> List[bytes]:
    """Create test protocol data for discovery testing."""
    return [
        b'\x01\x00\x04\x00Test',
        b'\x01\x01\x05\x00Hello',
        b'\x02\x00\x06\x00World',
        b'\x03\x00\x07\x00Message'
    ]


def create_test_openapi_spec() -> Dict[str, Any]:
    """Create a test OpenAPI specification."""
    return {
        'openapi': '3.0.0',
        'info': {
            'title': 'Test Protocol API',
            'version': '1.0.0',
            'description': 'Generated API for test protocol'
        },
        'servers': [
            {'url': 'https://api.test.com/v1', 'description': 'Production server'}
        ],
        'paths': {
            '/messages': {
                'post': {
                    'summary': 'Send a message',
                    'operationId': 'sendMessage',
                    'requestBody': {
                        'required': True,
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/Message'}
                            }
                        }
                    },
                    'responses': {
                        '200': {
                            'description': 'Message sent successfully',
                            'content': {
                                'application/json': {
                                    'schema': {'$ref': '#/components/schemas/MessageResponse'}
                                }
                            }
                        }
                    }
                }
            }
        },
        'components': {
            'schemas': {
                'Message': {
                    'type': 'object',
                    'properties': {
                        'message_type': {'type': 'integer'},
                        'payload': {'type': 'string'}
                    },
                    'required': ['message_type', 'payload']
                },
                'MessageResponse': {
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'string'},
                        'status': {'type': 'string'}
                    }
                }
            },
            'securitySchemes': {
                'bearerAuth': {
                    'type': 'http',
                    'scheme': 'bearer',
                    'bearerFormat': 'JWT'
                }
            }
        },
        'security': [{'bearerAuth': []}]
    }


def assert_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def assert_valid_timestamp(value: str) -> bool:
    """Check if a string is a valid ISO timestamp."""
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False


# Performance testing helpers

@pytest.fixture
def performance_config():
    """Configuration for performance testing."""
    return {
        'max_response_time': 5.0,  # seconds
        'max_memory_usage': 500 * 1024 * 1024,  # 500MB
        'concurrent_requests': 10,
        'test_duration': 30,  # seconds
        'acceptable_error_rate': 0.01  # 1%
    }


def measure_performance(func):
    """Decorator to measure function performance."""
    import time
    import psutil
    import os
    
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        # Measure before
        start_time = time.time()
        start_memory = process.memory_info().rss
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Measure after
        end_time = time.time()
        end_memory = process.memory_info().rss
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        return {
            'result': result,
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'peak_memory': end_memory
        }
    
    return wrapper


# Test data generators

def generate_protocol_variations(base_protocol: bytes, count: int = 10) -> List[bytes]:
    """Generate variations of a protocol message for testing."""
    variations = [base_protocol]
    
    for i in range(1, count):
        # Create variations by changing payload
        payload = f"Message{i}".encode()
        variation = base_protocol[:-5] + len(payload).to_bytes(2, 'big') + payload
        variations.append(variation)
    
    return variations


def generate_test_languages() -> List[CodeLanguage]:
    """Generate list of test languages."""
    return [CodeLanguage.PYTHON, CodeLanguage.TYPESCRIPT, CodeLanguage.JAVASCRIPT]


def generate_test_api_styles() -> List[APIStyle]:
    """Generate list of test API styles."""
    return [APIStyle.REST, APIStyle.GRAPHQL, APIStyle.GRPC]


# Async test helpers

async def async_test_timeout(coro, timeout: float = 30.0):
    """Run an async test with timeout."""
    return await asyncio.wait_for(coro, timeout=timeout)


# Integration test setup

@pytest.fixture
def integration_test_config():
    """Configuration for integration testing."""
    return {
        'test_database_url': 'sqlite:///test_translation.db',
        'test_cache_url': 'redis://localhost:6379/1',
        'test_vector_db': 'chroma_test',
        'cleanup_after_test': True,
        'seed_test_data': True
    }


# Mock authentication

@pytest.fixture
def mock_auth():
    """Mock authentication for testing."""
    def get_current_user_override():
        return {
            'user_id': 'test-user',
            'username': 'testuser',
            'roles': ['user'],
            'permissions': ['read', 'write']
        }
    
    return get_current_user_override


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.security = pytest.mark.security
pytest.mark.api = pytest.mark.api


__all__ = [
    'mock_config',
    'sample_protocol_messages',
    'sample_protocol_schema',
    'sample_api_specification',
    'sample_generated_sdk',
    'sample_translation_request',
    'sample_translation_result',
    'mock_llm_service',
    'mock_rag_engine',
    'mock_discovery_orchestrator',
    'mock_api_generator',
    'mock_code_generator',
    'mock_protocol_bridge',
    'log_context',
    'error_context',
    'mock_cache',
    'mock_metrics',
    'base64_encoded_messages',
    'api_test_client',
    'mock_current_user',
    'create_test_protocol_data',
    'create_test_openapi_spec',
    'assert_valid_uuid',
    'assert_valid_timestamp',
    'performance_config',
    'measure_performance',
    'generate_protocol_variations',
    'generate_test_languages',
    'generate_test_api_styles',
    'async_test_timeout',
    'integration_test_config',
    'mock_auth'
]
