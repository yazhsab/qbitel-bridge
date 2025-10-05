"""
CRONOS AI Engine - API Tests

This module contains tests for REST and gRPC API endpoints.
"""

import pytest
import pytest_asyncio
import asyncio
from contextlib import ExitStack
import json
import base64
from unittest.mock import Mock, patch, AsyncMock
from types import SimpleNamespace
from fastapi.testclient import TestClient
import grpc
from grpc import aio as grpc_aio

from ai_engine.api.rest import create_app
from ai_engine.api.grpc import AIEngineGRPCService, GRPCServer
from ai_engine.api.schemas import (
    ProtocolDiscoveryRequest,
    FieldDetectionRequest,
    AnomalyDetectionRequest,
    DataFormat,
    ProtocolType,
    DetectionLevel,
)
from ai_engine.api.auth import initialize_auth, get_api_key
from ai_engine.core.config import Config
from . import TestConfig


@pytest.fixture
def config():
    """Create test configuration."""
    cfg = Config()
    cfg.model_path = "test_models"
    cfg.data_path = "test_data"
    cfg.device = "cpu"
    return cfg


@pytest.fixture
def app(config):
    """Create test FastAPI app."""
    application = create_app(config)
    application.router.on_startup.clear()
    application.router.on_shutdown.clear()
    return application


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create authentication headers."""
    api_key = get_api_key()
    return {"Authorization": f"Bearer {api_key}"}


@pytest.fixture
def sample_data():
    """Create sample request data."""
    http_data = b"GET /api/v1/users HTTP/1.1\r\nHost: example.com\r\n\r\n"
    return {
        "base64_http": base64.b64encode(http_data).decode("utf-8"),
        "hex_modbus": "0103000000024C0B",
        "text_data": "Hello World",
    }


class TestRESTAPI:
    """Test cases for REST API endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "checks" in data

    def test_health_check_unhealthy_engine(self, client):
        """Test health check with unhealthy engine."""
        with patch("ai_engine.api.rest._ai_engine", None):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["unhealthy", "degraded"]

    def test_root_endpoint(self, client):
        """Root endpoint advertises features."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"].startswith("CRONOS AI Engine")
        assert "features" in data

    def test_detection_requires_auth(self, client):
        """Detection endpoints require authentication."""
        payload = {
            "packet_data": base64.b64encode(b"PING").decode(),
            "metadata": {},
            "enable_llm_analysis": False,
        }
        response = client.post("/api/v1/discover", json=payload)
        assert response.status_code == 401

    def test_protocol_discovery_success(self, client, auth_headers, sample_data):
        """Test successful protocol discovery."""
        payload = {
            "packet_data": base64.b64encode(b"GET / HTTP/1.1\r\n\r\n").decode(),
            "metadata": {"source": "unit-test"},
            "enable_llm_analysis": False,
        }

        mock_engine = SimpleNamespace(
            discover_protocol=AsyncMock(
                return_value={
                    "protocol_type": "http",
                    "confidence": 0.92,
                    "structure": {"fields": []},
                    "metadata": {"characteristics": {"method": "GET"}},
                    "processing_time": 0.02,
                }
            )
        )

        with patch("ai_engine.api.rest._ai_engine", mock_engine):
            response = client.post(
                "/api/v1/discover", json=payload, headers=auth_headers
            )

        assert response.status_code == 200
        data = response.json()
        assert data["protocol_type"] == "http"
        assert data["confidence"] == 0.92
        assert data["metadata"]["characteristics"]["method"] == "GET"

    def test_protocol_discovery_invalid_data(self, client, auth_headers):
        """Invalid base64 payload returns 422."""
        payload = {"packet_data": "invalid_base64_data!@#", "metadata": None}
        response = client.post("/api/v1/discover", json=payload, headers=auth_headers)
        assert response.status_code == 422

    def test_field_detection_success(self, client, auth_headers, sample_data):
        """Test successful field detection."""
        payload = {
            "message_data": base64.b64encode(
                b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
            ).decode(),
            "protocol_type": "http",
            "enable_llm_analysis": False,
        }

        mock_engine = SimpleNamespace(
            detect_fields=AsyncMock(
                return_value=[
                    {
                        "id": "field_1",
                        "name": "method",
                        "start": 0,
                        "end": 3,
                        "type": "method",
                        "confidence": 0.95,
                    },
                    {
                        "id": "field_2",
                        "name": "path",
                        "start": 4,
                        "end": 17,
                        "type": "path",
                        "confidence": 0.88,
                    },
                ]
            )
        )

        with patch("ai_engine.api.rest._ai_engine", mock_engine):
            response = client.post(
                "/api/v1/detect-fields", json=payload, headers=auth_headers
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_fields"] == 2
        assert data["detected_fields"][0]["field_type"] == "method"

    def test_app_startup_initializes_services(self, config):
        with ExitStack() as stack:
            mock_engine_cls = stack.enter_context(
                patch("ai_engine.api.rest.CronosAIEngine")
            )
            mock_init_llm = stack.enter_context(
                patch(
                    "ai_engine.api.rest.initialize_llm_service", new_callable=AsyncMock
                )
            )
            mock_shutdown_llm = stack.enter_context(
                patch("ai_engine.api.rest.shutdown_llm_service", new_callable=AsyncMock)
            )
            mock_create_copilot = stack.enter_context(
                patch(
                    "ai_engine.api.rest.create_protocol_copilot", new_callable=AsyncMock
                )
            )
            mock_metrics_cls = stack.enter_context(
                patch("ai_engine.api.rest.MetricsCollector")
            )
            mock_init_translation = stack.enter_context(
                patch(
                    "ai_engine.api.rest.initialize_translation_studio",
                    new_callable=AsyncMock,
                )
            )
            mock_shutdown_translation = stack.enter_context(
                patch(
                    "ai_engine.api.rest.shutdown_translation_studio",
                    new_callable=AsyncMock,
                )
            )
            mock_init_alert = stack.enter_context(
                patch(
                    "ai_engine.api.rest.initialize_alert_manager",
                    new_callable=AsyncMock,
                )
            )
            mock_shutdown_alert = stack.enter_context(
                patch(
                    "ai_engine.api.rest.shutdown_alert_manager", new_callable=AsyncMock
                )
            )
            mock_get_policy = stack.enter_context(
                patch("ai_engine.api.rest.get_policy_engine")
            )
            mock_init_security = stack.enter_context(
                patch(
                    "ai_engine.api.rest.initialize_security_orchestrator",
                    new_callable=AsyncMock,
                )
            )
            mock_shutdown_security = stack.enter_context(
                patch(
                    "ai_engine.api.rest.shutdown_security_orchestrator",
                    new_callable=AsyncMock,
                )
            )

            engine_instance = AsyncMock()
            engine_instance.initialize = AsyncMock()
            engine_instance.shutdown = AsyncMock()
            mock_engine_cls.return_value = engine_instance

            metrics_instance = Mock()
            metrics_instance.start = AsyncMock()
            metrics_instance.shutdown = AsyncMock()
            mock_metrics_cls.return_value = metrics_instance

            llm_service = Mock()
            mock_init_llm.return_value = llm_service

            copilot_instance = Mock()
            copilot_instance.get_health_status.return_value = {
                "llm_service": {"providers": {}},
                "rag_engine": "healthy",
            }
            copilot_instance.shutdown = AsyncMock()
            mock_create_copilot.return_value = copilot_instance

            translation_instance = Mock()
            mock_init_translation.return_value = translation_instance

            alert_manager = Mock()
            mock_init_alert.return_value = alert_manager

            policy_engine = Mock()
            mock_get_policy.return_value = policy_engine

            import ai_engine.api.rest as rest_module

            rest_module._ai_engine = None
            rest_module._protocol_copilot = None
            rest_module._metrics_collector = None

            with TestClient(create_app(config)) as client:
                response = client.get("/health")
                assert response.status_code == 200

            rest_module._ai_engine = None
            rest_module._protocol_copilot = None
            rest_module._metrics_collector = None

        engine_instance.initialize.assert_awaited_once()
        metrics_instance.start.assert_awaited_once()
        mock_init_llm.assert_awaited_once()
        mock_create_copilot.assert_awaited_once()
        mock_init_translation.assert_awaited_once()
        mock_init_alert.assert_awaited_once()
        mock_init_security.assert_awaited_once()

        copilot_instance.shutdown.assert_awaited_once()
        metrics_instance.shutdown.assert_awaited_once()
        engine_instance.shutdown.assert_awaited_once()
        mock_shutdown_llm.assert_awaited_once()
        mock_shutdown_translation.assert_awaited_once()
        mock_shutdown_alert.assert_awaited_once()
        mock_shutdown_security.assert_awaited_once()


class TestGRPCAPI:
    """Test cases for gRPC API."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        config.grpc_port = 50052  # Use different port for testing
        return config

    @pytest_asyncio.fixture
    async def grpc_service(self, config):
        """Create gRPC service for testing."""
        service = AIEngineGRPCService(config)
        service.ai_engine = None
        return service

    @pytest.fixture
    def sample_data(self):
        """Create sample gRPC request data."""
        return {
            "http_data": base64.b64encode(b"GET / HTTP/1.1\r\n\r\n").decode(),
            "modbus_data": "0103000000024C0B",
        }

    @pytest.mark.asyncio
    async def test_grpc_service_initialization(self, grpc_service):
        """Test gRPC service initialization."""
        assert grpc_service is not None
        assert grpc_service.config is not None
        assert grpc_service.stats["total_requests"] == 0

    @pytest.mark.asyncio
    async def test_protocol_discovery_grpc(self, grpc_service, sample_data):
        """Test gRPC protocol discovery."""
        # Create mock request
        mock_request = Mock()
        mock_request.data = sample_data["http_data"]
        mock_request.data_format = "base64"
        mock_request.confidence_threshold = 0.7
        mock_request.max_samples = 1000
        mock_request.include_grammar = False

        # Create mock context
        mock_context = Mock()

        # Mock AI Engine
        engine = SimpleNamespace(_initialized=True)
        engine.discover_protocol = AsyncMock(
            return_value={
                "protocol_type": "http",
                "confidence": 0.9,
                "structure": {"fields": []},
                "metadata": {"characteristics": {"method": "GET"}},
                "processing_time": 0.05,
            }
        )
        grpc_service.ai_engine = engine

        # Test discovery
        result = await grpc_service.DiscoverProtocol(mock_request, mock_context)

        assert result["discovered_protocol"] == "http"
        assert result["confidence_score"] == 0.9
        assert "processing_time_ms" in result
        assert grpc_service.stats["total_requests"] == 1
        assert grpc_service.stats["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_field_detection_grpc(self, grpc_service, sample_data):
        """Test gRPC field detection."""
        # Create mock request
        mock_request = Mock()
        mock_request.data = sample_data["http_data"]
        mock_request.data_format = "base64"
        mock_request.protocol_hint = "http"
        mock_request.detection_level = "medium"
        mock_request.include_boundaries = True

        # Create mock context
        mock_context = Mock()

        # Mock AI Engine
        engine = SimpleNamespace(_initialized=True)
        engine.detect_fields = AsyncMock(
            return_value=[
                {
                    "id": "field_1",
                    "start": 0,
                    "end": 3,
                    "type": "method",
                    "confidence": 0.9,
                }
            ]
        )
        grpc_service.ai_engine = engine

        # Test detection
        result = await grpc_service.DetectFields(mock_request, mock_context)

        assert result["total_fields"] == 1
        assert len(result["detected_fields"]) == 1
        assert result["detected_fields"][0]["field_type"] == "method"

    @pytest.mark.asyncio
    async def test_anomaly_detection_grpc(self, grpc_service, sample_data):
        """Test gRPC anomaly detection."""
        # Create mock request
        mock_request = Mock()
        mock_request.data = sample_data["http_data"]
        mock_request.data_format = "base64"
        mock_request.sensitivity = "medium"
        mock_request.anomaly_threshold = 0.5

        # Create mock context
        mock_context = Mock()

        # Mock AI Engine
        engine = SimpleNamespace(_initialized=True)
        engine.detect_anomaly = AsyncMock(
            return_value={
                "is_anomaly": False,
                "anomaly_score": 0.2,
                "confidence": 0.95,
                "context": {},
                "processing_time": 0.12,
            }
        )
        grpc_service.ai_engine = engine

        # Test detection
        result = await grpc_service.DetectAnomalies(mock_request, mock_context)

        assert result["is_anomalous"] is False
        assert result["anomaly_score"]["score"] == 0.2
        assert result["anomaly_score"]["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_grpc_error_handling(self, grpc_service, sample_data):
        """Test gRPC error handling."""
        # Create mock request
        mock_request = Mock()
        mock_request.data = sample_data["http_data"]
        mock_request.data_format = "base64"

        # Create mock context
        mock_context = Mock()

        # Mock AI Engine to raise exception
        engine = SimpleNamespace(_initialized=True)
        engine.discover_protocol = AsyncMock(side_effect=Exception("Test error"))
        grpc_service.ai_engine = engine

        # Test error handling
        result = await grpc_service.DiscoverProtocol(mock_request, mock_context)

        assert result == {}  # Empty response on error
        mock_context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)
        assert grpc_service.stats["failed_requests"] == 1

    @pytest.mark.asyncio
    async def test_service_status_grpc(self, grpc_service):
        """Test gRPC service status."""
        # Create mock request and context
        mock_request = Mock()
        mock_context = Mock()

        # Mock AI Engine status
        mock_engine_status = {"status": "ready", "components": {}}
        engine = SimpleNamespace(
            _initialized=True, get_model_info=lambda: mock_engine_status
        )
        grpc_service.ai_engine = engine

        # Test status
        result = await grpc_service.GetServiceStatus(mock_request, mock_context)

        assert result["service_name"] == "CRONOS AI Engine gRPC"
        assert result["version"] == "1.0.0"
        assert "uptime_seconds" in result
        assert "statistics" in result
        assert "engine_status" in result

    @pytest.mark.asyncio
    async def test_health_check_grpc(self, grpc_service):
        """Test gRPC health check."""
        # Create mock request and context
        mock_request = Mock()
        mock_context = Mock()

        # Mock AI Engine
        engine = SimpleNamespace(_initialized=True)
        grpc_service.ai_engine = engine

        # Test health check
        result = await grpc_service.HealthCheck(mock_request, mock_context)

        assert result["status"] == "healthy"
        assert result["service"] == "cronos-ai-grpc"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_batch_processing_grpc(self, grpc_service, sample_data):
        """Test gRPC batch processing."""
        # Create mock request
        mock_request = Mock()
        mock_request.operation = "discovery"
        mock_request.data_items = [sample_data["http_data"]] * 3

        # Create mock context
        mock_context = Mock()

        # Mock AI Engine
        engine = SimpleNamespace(_initialized=True)
        engine.discover_protocol = AsyncMock(
            return_value={
                "protocol_type": "http",
                "confidence": 0.88,
                "processing_time": 0.04,
                "structure": {},
            }
        )
        grpc_service.ai_engine = engine

        # Test batch processing
        result = await grpc_service.BatchProcess(mock_request, mock_context)

        assert result["total_items"] == 3
        assert "batch_id" in result
        assert "results" in result
        assert all(item["success"] for item in result["results"])


class TestAPIAuthentication:
    """Test cases for API authentication."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    def test_api_key_authentication(self, config):
        """Test API key authentication."""
        initialize_auth(config)
        api_key = get_api_key()

        assert api_key is not None
        assert (
            api_key.startswith("cronos_ai_")
            or api_key == "cronos_ai_mock_key_for_testing"
        )

    def test_invalid_api_key(self, client, sample_data):
        """Invalid bearer token should be rejected."""
        payload = {"packet_data": sample_data["base64_http"], "metadata": None}
        response = client.post(
            "/api/v1/discover",
            json=payload,
            headers={"Authorization": "Bearer invalid_key"},
        )
        assert response.status_code == 401

    def test_missing_authentication(self, client, sample_data):
        """Missing credentials should return 401."""
        payload = {"packet_data": sample_data["base64_http"], "metadata": None}
        response = client.post("/api/v1/discover", json=payload)
        assert response.status_code == 401


class TestAPIPerformance:
    """Performance tests for API endpoints."""

    @pytest.mark.asyncio
    async def test_api_response_times(self, client, auth_headers):
        """Test API response times."""
        import time

        start_time = time.time()
        response = client.get("/health", headers=auth_headers)
        end_time = time.time()

        response_time_ms = (end_time - start_time) * 1000

        assert response.status_code == 200
        assert response_time_ms < TestConfig.PERFORMANCE_THRESHOLD_MS

    def test_concurrent_requests(self, client, auth_headers):
        """Test handling of concurrent requests."""
        import threading
        import time

        results = []

        def make_request():
            start_time = time.time()
            response = client.get("/health", headers=auth_headers)
            end_time = time.time()
            results.append(
                {
                    "status_code": response.status_code,
                    "response_time": (end_time - start_time) * 1000,
                }
            )

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(results) == 10
        for result in results:
            assert result["status_code"] == 200
            assert result["response_time"] < TestConfig.PERFORMANCE_THRESHOLD_MS


# Pytest fixtures for all API tests


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.model_path = "test_models"
    config.data_path = "test_data"
    config.device = "cpu"
    return config


@pytest.fixture
def sample_data():
    """Create sample test data."""
    http_data = b"GET /api/v1/users HTTP/1.1\r\nHost: example.com\r\n\r\n"
    return {
        "base64_http": base64.b64encode(http_data).decode("utf-8"),
        "hex_modbus": "0103000000024C0B",
        "text_data": "Hello World",
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
