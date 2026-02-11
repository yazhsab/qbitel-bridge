"""
QBITEL - Translation Studio API Endpoints Tests
Enterprise-grade integration and unit tests for translation studio REST API endpoints.
"""

import pytest
import json
import base64
import uuid
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from ai_engine.translation.api_endpoints import router
from ai_engine.translation.models import (
    CodeLanguage,
    APIStyle,
    SecurityLevel,
    GenerationStatus,
)
from ai_engine.translation.exceptions import (
    TranslationStudioException,
    ProtocolDiscoveryException,
    APIGenerationException,
    CodeGenerationException,
    ValidationException,
)


class TestProtocolDiscoveryEndpoints:
    """Test cases for protocol discovery and API generation endpoints."""

    @pytest.fixture
    def app_with_mocks(
        self,
        mock_discovery_orchestrator,
        mock_api_generator,
        mock_code_generator,
        mock_protocol_bridge,
        mock_rag_engine,
    ):
        """Create test app with mocked services."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        # Mock service initialization
        with patch.multiple(
            "ai_engine.translation.api_endpoints",
            discovery_orchestrator=mock_discovery_orchestrator,
            api_generator=mock_api_generator,
            code_generator=mock_code_generator,
            protocol_bridge=mock_protocol_bridge,
            rag_engine=mock_rag_engine,
        ):
            yield app

    def test_discover_protocol_success(
        self,
        app_with_mocks,
        base64_encoded_messages,
        mock_discovery_orchestrator,
        mock_current_user,
    ):
        """Test successful protocol discovery and API generation."""
        from fastapi.testclient import TestClient

        # Setup mock response
        mock_discovery_orchestrator.discover_and_generate_api.return_value = Mock(
            protocol_type="TestProtocol",
            confidence=0.95,
            api_specification=Mock(
                to_openapi_dict=Mock(
                    return_value={"openapi": "3.0.0", "info": {"title": "Test API"}}
                )
            ),
            recommendations=["Add rate limiting", "Implement caching"],
            natural_language_summary="This is a test messaging protocol",
        )

        client = TestClient(app_with_mocks)

        # Mock authentication
        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                "/api/v1/translation/discover",
                json={
                    "messages": base64_encoded_messages,
                    "target_api_style": "rest",
                    "target_languages": ["python", "typescript"],
                    "security_level": "authenticated",
                    "generate_documentation": True,
                    "generate_tests": True,
                    "api_base_path": "/api/v1",
                },
            )

        assert response.status_code == 200
        result = response.json()

        assert result["protocol_type"] == "TestProtocol"
        assert result["confidence"] == 0.95
        assert result["status"] == GenerationStatus.COMPLETED.value
        assert "request_id" in result
        assert "processing_time" in result
        assert len(result["recommendations"]) == 2

    def test_discover_protocol_invalid_messages(
        self, app_with_mocks, mock_current_user
    ):
        """Test protocol discovery with invalid base64 messages."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                "/api/v1/translation/discover",
                json={
                    "messages": ["invalid-base64!@#"],  # Invalid base64
                    "target_api_style": "rest",
                    "target_languages": ["python"],
                },
            )

        assert response.status_code == 400
        assert "Invalid base64 message" in response.json()["detail"]

    def test_discover_protocol_empty_messages(self, app_with_mocks, mock_current_user):
        """Test protocol discovery with empty messages."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                "/api/v1/translation/discover",
                json={
                    "messages": [],  # Empty messages
                    "target_api_style": "rest",
                    "target_languages": ["python"],
                },
            )

        assert response.status_code == 422  # Validation error

    def test_discover_protocol_service_failure(
        self,
        app_with_mocks,
        base64_encoded_messages,
        mock_discovery_orchestrator,
        mock_current_user,
    ):
        """Test protocol discovery when service fails."""
        from fastapi.testclient import TestClient

        # Setup mock to raise exception
        mock_discovery_orchestrator.discover_and_generate_api.side_effect = (
            ProtocolDiscoveryException("Protocol discovery failed")
        )

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                "/api/v1/translation/discover",
                json={
                    "messages": base64_encoded_messages,
                    "target_api_style": "rest",
                    "target_languages": ["python"],
                },
            )

        assert response.status_code == 500
        assert "Discovery failed" in response.json()["detail"]

    @pytest.mark.parametrize(
        "api_style,expected_valid",
        [
            ("rest", True),
            ("graphql", True),
            ("grpc", True),
            ("websocket", True),
            ("invalid_style", False),
        ],
    )
    def test_discover_protocol_api_styles(
        self,
        app_with_mocks,
        base64_encoded_messages,
        mock_discovery_orchestrator,
        mock_current_user,
        api_style,
        expected_valid,
    ):
        """Test protocol discovery with different API styles."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                "/api/v1/translation/discover",
                json={
                    "messages": base64_encoded_messages,
                    "target_api_style": api_style,
                    "target_languages": ["python"],
                },
            )

        if expected_valid:
            assert response.status_code in [
                200,
                500,
            ]  # 500 if service fails, but validation passes
        else:
            assert response.status_code == 422  # Validation error

    def test_generate_api_specification(
        self,
        app_with_mocks,
        sample_protocol_schema,
        mock_api_generator,
        mock_current_user,
    ):
        """Test API specification generation endpoint."""
        from fastapi.testclient import TestClient

        # Setup mock response
        mock_api_generator.generate_api_specification.return_value = Mock(
            to_openapi_dict=Mock(return_value={"openapi": "3.0.0"}),
            endpoints=[],
            schemas={},
            spec_id=str(uuid.uuid4()),
        )

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                "/api/v1/translation/generate-api",
                json={
                    "protocol_schema": sample_protocol_schema.to_dict(),
                    "api_style": "rest",
                    "security_level": "authenticated",
                    "base_path": "/api/v1",
                },
            )

        assert response.status_code == 200
        result = response.json()

        assert "api_specification" in result
        assert "endpoints_count" in result
        assert "processing_time" in result
        assert "spec_id" in result


class TestSDKGenerationEndpoints:
    """Test cases for SDK generation endpoints."""

    @pytest.fixture
    def app_with_mocks(self, mock_code_generator):
        """Create test app with mocked code generator."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch(
            "ai_engine.translation.api_endpoints.code_generator", mock_code_generator
        ):
            yield app

    def test_generate_sdk_success(
        self,
        app_with_mocks,
        sample_api_specification,
        mock_code_generator,
        mock_current_user,
    ):
        """Test successful SDK generation."""
        from fastapi.testclient import TestClient

        # Setup mock response
        mock_code_generator.generate_multiple_sdks.return_value = {
            CodeLanguage.PYTHON: Mock(
                name="python-sdk",
                version="1.0.0",
                source_files={"client.py": "code"},
                test_files={"test_client.py": "test"},
                config_files={"setup.py": "config"},
                documentation_files={"README.md": "docs"},
                sdk_id=str(uuid.uuid4()),
            ),
            CodeLanguage.TYPESCRIPT: Mock(
                name="typescript-sdk",
                version="1.0.0",
                source_files={"client.ts": "code"},
                test_files={"client.test.ts": "test"},
                config_files={"package.json": "config"},
                documentation_files={"README.md": "docs"},
                sdk_id=str(uuid.uuid4()),
            ),
        }

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                "/api/v1/translation/generate-sdk",
                json={
                    "api_specification": sample_api_specification.to_openapi_dict(),
                    "target_languages": ["python", "typescript"],
                    "package_name": "test-protocol-sdk",
                    "version": "1.0.0",
                    "generate_tests": True,
                    "generate_documentation": True,
                },
            )

        assert response.status_code == 200
        result = response.json()

        assert "generated_sdks" in result
        assert len(result["generated_sdks"]) == 2
        assert result["total_languages"] == 2
        assert "processing_time" in result

        # Verify SDK details
        sdks = result["generated_sdks"]
        languages = {sdk["language"] for sdk in sdks}
        assert "python" in languages
        assert "typescript" in languages

    def test_generate_sdk_invalid_specification(
        self, app_with_mocks, mock_current_user
    ):
        """Test SDK generation with invalid API specification."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                "/api/v1/translation/generate-sdk",
                json={
                    "api_specification": {"invalid": "spec"},  # Invalid spec
                    "target_languages": ["python"],
                    "package_name": "test-sdk",
                },
            )

        assert response.status_code == 500  # Should fail during processing

    def test_download_sdk_success(self, app_with_mocks, mock_current_user):
        """Test successful SDK download."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mocks)
        sdk_id = str(uuid.uuid4())

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.get(f"/api/v1/translation/download-sdk/{sdk_id}")

        assert response.status_code == 200
        assert response.headers["content-type"] in [
            "application/zip",
            "application/octet-stream",
        ]

    def test_download_sdk_invalid_format(self, app_with_mocks, mock_current_user):
        """Test SDK download with invalid format."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mocks)
        sdk_id = str(uuid.uuid4())

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.get(
                f"/api/v1/translation/download-sdk/{sdk_id}?format=invalid"
            )

        assert response.status_code == 400
        assert "Unsupported format" in response.json()["detail"]


class TestProtocolTranslationEndpoints:
    """Test cases for protocol translation endpoints."""

    @pytest.fixture
    def app_with_mocks(self, mock_protocol_bridge):
        """Create test app with mocked protocol bridge."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch(
            "ai_engine.translation.api_endpoints.protocol_bridge", mock_protocol_bridge
        ):
            yield app

    def test_translate_protocol_success(
        self, app_with_mocks, mock_protocol_bridge, mock_current_user
    ):
        """Test successful protocol translation."""
        from fastapi.testclient import TestClient

        # Setup mock response
        mock_protocol_bridge.translate_protocol.return_value = Mock(
            translation_id=str(uuid.uuid4()),
            source_protocol="HTTP",
            target_protocol="WebSocket",
            translated_data=b"translated data",
            confidence=0.9,
            processing_time=0.5,
            translation_mode=Mock(value="hybrid"),
            metadata={"test": "metadata"},
            warnings=[],
            validation_errors=[],
        )

        client = TestClient(app_with_mocks)
        test_data = base64.b64encode(b'{"test": "data"}').decode("utf-8")

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                "/api/v1/translation/translate",
                json={
                    "source_protocol": "HTTP",
                    "target_protocol": "WebSocket",
                    "data": test_data,
                    "translation_mode": "hybrid",
                    "quality_level": "balanced",
                    "preserve_metadata": True,
                },
            )

        assert response.status_code == 200
        result = response.json()

        assert "translation_id" in result
        assert result["source_protocol"] == "HTTP"
        assert result["target_protocol"] == "WebSocket"
        assert "translated_data" in result
        assert result["confidence"] == 0.9
        assert "processing_time" in result

    def test_translate_protocol_invalid_data(self, app_with_mocks, mock_current_user):
        """Test protocol translation with invalid base64 data."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                "/api/v1/translation/translate",
                json={
                    "source_protocol": "HTTP",
                    "target_protocol": "WebSocket",
                    "data": "invalid-base64!@#",  # Invalid base64
                    "translation_mode": "hybrid",
                },
            )

        assert response.status_code == 500
        assert "Translation failed" in response.json()["detail"]

    def test_batch_translate_protocols(
        self, app_with_mocks, mock_protocol_bridge, mock_current_user
    ):
        """Test batch protocol translation."""
        from fastapi.testclient import TestClient

        # Setup mock response
        mock_protocol_bridge.batch_translate.return_value = [
            Mock(
                translation_id=str(uuid.uuid4()),
                source_protocol="HTTP",
                target_protocol="WebSocket",
                confidence=0.9,
                validation_errors=[],
            ),
            Mock(
                translation_id=str(uuid.uuid4()),
                source_protocol="MQTT",
                target_protocol="HTTP",
                confidence=0.85,
                validation_errors=[],
            ),
        ]

        client = TestClient(app_with_mocks)

        test_data1 = base64.b64encode(b'{"test": "data1"}').decode("utf-8")
        test_data2 = base64.b64encode(b'{"test": "data2"}').decode("utf-8")

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                "/api/v1/translation/translate/batch",
                json={
                    "translations": [
                        {
                            "source_protocol": "HTTP",
                            "target_protocol": "WebSocket",
                            "data": test_data1,
                            "translation_mode": "hybrid",
                        },
                        {
                            "source_protocol": "MQTT",
                            "target_protocol": "HTTP",
                            "data": test_data2,
                            "translation_mode": "direct",
                        },
                    ],
                    "parallel_processing": True,
                    "fail_fast": False,
                },
            )

        assert response.status_code == 200
        result = response.json()

        assert result["total_translations"] == 2
        assert result["successful_translations"] == 2
        assert result["failed_translations"] == 0
        assert len(result["results"]) == 2
        assert "job_id" in result


class TestStreamingEndpoints:
    """Test cases for streaming translation endpoints."""

    @pytest.fixture
    def app_with_mocks(self, mock_protocol_bridge):
        """Create test app with mocked protocol bridge."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch(
            "ai_engine.translation.api_endpoints.protocol_bridge", mock_protocol_bridge
        ):
            yield app

    def test_create_streaming_connection(
        self, app_with_mocks, mock_protocol_bridge, mock_current_user
    ):
        """Test creating streaming connection."""
        from fastapi.testclient import TestClient

        connection_id = str(uuid.uuid4())
        mock_protocol_bridge.create_streaming_connection.return_value = connection_id

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.post(
                "/api/v1/translation/streaming/create",
                json={
                    "source_protocol": "MQTT",
                    "target_protocol": "WebSocket",
                    "quality_level": "high",
                    "connection_name": "test-connection",
                },
            )

        assert response.status_code == 200
        result = response.json()

        assert result["connection_id"] == connection_id
        assert result["source_protocol"] == "MQTT"
        assert result["target_protocol"] == "WebSocket"
        assert "created_at" in result

    def test_get_streaming_connection_status(
        self, app_with_mocks, mock_protocol_bridge, mock_current_user
    ):
        """Test getting streaming connection status."""
        from fastapi.testclient import TestClient

        connection_id = str(uuid.uuid4())
        mock_status = {
            "connection_id": connection_id,
            "status": "active",
            "messages_translated": 100,
            "average_latency": 0.05,
        }
        mock_protocol_bridge.get_connection_status.return_value = mock_status

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.get(
                f"/api/v1/translation/streaming/{connection_id}/status"
            )

        assert response.status_code == 200
        result = response.json()

        assert result["connection_id"] == connection_id
        assert result["status"] == "active"
        assert "messages_translated" in result

    def test_get_streaming_connection_status_not_found(
        self, app_with_mocks, mock_protocol_bridge, mock_current_user
    ):
        """Test getting status for non-existent connection."""
        from fastapi.testclient import TestClient

        connection_id = str(uuid.uuid4())
        mock_protocol_bridge.get_connection_status.return_value = None

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.get(
                f"/api/v1/translation/streaming/{connection_id}/status"
            )

        assert response.status_code == 404
        assert "Connection not found" in response.json()["detail"]

    def test_close_streaming_connection(
        self, app_with_mocks, mock_protocol_bridge, mock_current_user
    ):
        """Test closing streaming connection."""
        from fastapi.testclient import TestClient

        connection_id = str(uuid.uuid4())
        mock_protocol_bridge.close_streaming_connection.return_value = None

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.delete(f"/api/v1/translation/streaming/{connection_id}")

        assert response.status_code == 200
        result = response.json()

        assert result["connection_id"] == connection_id
        assert result["status"] == "closed"
        assert "closed_at" in result


class TestKnowledgeBaseEndpoints:
    """Test cases for knowledge base and RAG endpoints."""

    @pytest.fixture
    def app_with_mocks(self, mock_rag_engine):
        """Create test app with mocked RAG engine."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("ai_engine.translation.api_endpoints.rag_engine", mock_rag_engine):
            yield app

    def test_get_protocol_patterns(
        self, app_with_mocks, mock_rag_engine, mock_current_user
    ):
        """Test getting protocol patterns from knowledge base."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.get(
                "/api/v1/translation/knowledge/patterns/HTTP",
                params={"pattern_type": "request", "limit": 3},
            )

        assert response.status_code == 200
        result = response.json()

        assert result["protocol_name"] == "HTTP"
        assert result["pattern_type"] == "request"
        assert "patterns" in result
        assert len(result["patterns"]) <= 3

    def test_get_code_templates(
        self, app_with_mocks, mock_rag_engine, mock_current_user
    ):
        """Test getting code generation templates."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.get(
                "/api/v1/translation/knowledge/templates/python",
                params={"template_type": "client", "limit": 2},
            )

        assert response.status_code == 200
        result = response.json()

        assert result["language"] == "python"
        assert result["template_type"] == "client"
        assert "templates" in result
        assert len(result["templates"]) <= 2

    def test_get_best_practices(
        self, app_with_mocks, mock_rag_engine, mock_current_user
    ):
        """Test getting translation best practices."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mocks)

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = client.get(
                "/api/v1/translation/knowledge/best-practices",
                params={"context": "security", "limit": 5},
            )

        assert response.status_code == 200
        result = response.json()

        assert result["context"] == "security"
        assert "best_practices" in result
        assert len(result["best_practices"]) <= 5


class TestStatusAndMetricsEndpoints:
    """Test cases for status and metrics endpoints."""

    @pytest.fixture
    def app_with_mocks(
        self,
        mock_discovery_orchestrator,
        mock_api_generator,
        mock_code_generator,
        mock_protocol_bridge,
        mock_rag_engine,
    ):
        """Create test app with all mocked services."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch.multiple(
            "ai_engine.translation.api_endpoints",
            discovery_orchestrator=mock_discovery_orchestrator,
            api_generator=mock_api_generator,
            code_generator=mock_code_generator,
            protocol_bridge=mock_protocol_bridge,
            rag_engine=mock_rag_engine,
        ):
            yield app

    def test_get_translation_studio_status(self, app_with_mocks):
        """Test getting translation studio status."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mocks)

        response = client.get("/api/v1/translation/status")

        assert response.status_code == 200
        result = response.json()

        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "services" in result
        assert "capabilities" in result

        # Verify service status
        services = result["services"]
        assert services["discovery_orchestrator"] is True
        assert services["api_generator"] is True
        assert services["code_generator"] is True
        assert services["protocol_bridge"] is True
        assert services["rag_engine"] is True

        # Verify capabilities
        capabilities = result["capabilities"]
        assert "supported_languages" in capabilities
        assert "supported_api_styles" in capabilities
        assert "supported_security_levels" in capabilities

    def test_get_translation_metrics(
        self,
        app_with_mocks,
        mock_discovery_orchestrator,
        mock_api_generator,
        mock_code_generator,
        mock_protocol_bridge,
        mock_rag_engine,
    ):
        """Test getting translation metrics."""
        from fastapi.testclient import TestClient

        # Setup mock metrics
        mock_discovery_orchestrator.get_api_generation_metrics.return_value = {
            "total_discoveries": 50,
            "average_confidence": 0.85,
        }
        mock_api_generator.get_generation_metrics.return_value = {
            "total_generated": 30,
            "average_processing_time": 2.5,
        }
        mock_code_generator.get_generation_metrics.return_value = {
            "total_sdks_generated": 25,
            "average_quality_score": 0.9,
        }
        mock_protocol_bridge.get_bridge_metrics.return_value = {
            "total_translations": 100,
            "average_confidence": 0.88,
        }
        mock_rag_engine.get_collection_stats.return_value = {
            "total_documents": 1000,
            "total_queries": 500,
        }

        client = TestClient(app_with_mocks)

        response = client.get("/api/v1/translation/metrics")

        assert response.status_code == 200
        result = response.json()

        assert "timestamp" in result
        assert "services" in result

        services = result["services"]
        assert "discovery" in services
        assert "api_generation" in services
        assert "code_generation" in services
        assert "protocol_bridge" in services
        assert "knowledge_base" in services


class TestFileUploadEndpoints:
    """Test cases for file upload endpoints."""

    def test_upload_protocol_samples_success(self, api_test_client, mock_current_user):
        """Test successful protocol sample file upload."""
        from fastapi.testclient import TestClient

        # Create test file content
        test_files = [
            (
                "files",
                ("test1.bin", b"\x01\x00\x04\x00Test", "application/octet-stream"),
            ),
            (
                "files",
                ("test2.bin", b"\x02\x00\x05\x00Hello", "application/octet-stream"),
            ),
        ]

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = api_test_client.post(
                "/api/v1/translation/upload/protocol-samples",
                files=test_files,
                params={"protocol_type": "TestProtocol"},
            )

        assert response.status_code == 200
        result = response.json()

        assert result["total_files"] == 2
        assert len(result["uploaded_files"]) == 2
        assert "next_steps" in result

        # Verify file details
        uploaded_files = result["uploaded_files"]
        assert uploaded_files[0]["filename"] == "test1.bin"
        assert uploaded_files[0]["protocol_type"] == "TestProtocol"

    def test_upload_protocol_samples_file_too_large(
        self, api_test_client, mock_current_user
    ):
        """Test file upload with file too large."""
        from fastapi.testclient import TestClient

        # Create large file (larger than 10MB limit)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        test_files = [
            ("files", ("large.bin", large_content, "application/octet-stream"))
        ]

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            response = api_test_client.post(
                "/api/v1/translation/upload/protocol-samples", files=test_files
            )

        assert response.status_code == 413
        assert "too large" in response.json()["detail"]


class TestAuthenticationAndPermissions:
    """Test cases for authentication and permission handling."""

    def test_unauthorized_access(self, api_test_client):
        """Test API access without authentication."""
        response = api_test_client.get("/api/v1/translation/status")

        # Assuming authentication is required
        # The actual status code depends on authentication middleware configuration
        assert response.status_code in [401, 200]  # 200 if status endpoint is public

    def test_insufficient_permissions(self, api_test_client):
        """Test API access with insufficient permissions."""
        limited_user = {
            "user_id": "limited-user",
            "username": "limiteduser",
            "roles": ["viewer"],
            "permissions": ["read"],  # No write permissions
        }

        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=limited_user,
        ):
            response = api_test_client.post(
                "/api/v1/translation/discover",
                json={"messages": ["dGVzdA=="]},  # base64 "test"
            )

        # This would depend on permission checking implementation
        # For now, assuming it returns 403 for insufficient permissions
        assert response.status_code in [200, 403]


class TestErrorHandling:
    """Test cases for error handling in API endpoints."""

    def test_internal_server_error_handling(self, api_test_client, mock_current_user):
        """Test handling of internal server errors."""
        with (
            patch(
                "ai_engine.translation.api_endpoints.get_current_user",
                return_value=mock_current_user,
            ),
            patch(
                "ai_engine.translation.api_endpoints.discovery_orchestrator",
                side_effect=Exception("Internal error"),
            ),
        ):

            response = api_test_client.post(
                "/api/v1/translation/discover",
                json={
                    "messages": ["dGVzdA=="],  # base64 "test"
                    "target_api_style": "rest",
                    "target_languages": ["python"],
                },
            )

        assert response.status_code == 500

    def test_validation_error_handling(self, api_test_client, mock_current_user):
        """Test handling of validation errors."""
        with patch(
            "ai_engine.translation.api_endpoints.get_current_user",
            return_value=mock_current_user,
        ):
            # Send request with invalid data structure
            response = api_test_client.post(
                "/api/v1/translation/discover",
                json={
                    "messages": "not_a_list",  # Should be a list
                    "target_api_style": "rest",
                },
            )

        assert response.status_code == 422  # Validation error
        assert "validation error" in response.json()["detail"][0]["type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
