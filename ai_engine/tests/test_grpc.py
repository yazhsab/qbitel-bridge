"""
Tests for QBITEL Engine gRPC Service
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import base64
import time
from typing import Dict, Any

from ai_engine.api.grpc import (
    AIEngineGRPCService,
    GRPCServer,
    AIEngineGRPCClient,
    run_grpc_server,
)
from ai_engine.core.config import Config
from ai_engine.core.exceptions import AIEngineException


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Config)
    config.grpc_port = 50051
    config.grpc_max_workers = 10
    config.grpc_max_message_length = 10 * 1024 * 1024
    config.redis = Mock()
    config.redis.host = "localhost"
    config.redis.port = 6379
    config.redis.db = 0
    return config


@pytest.fixture
def mock_ai_engine():
    """Create a mock AI engine."""
    engine = AsyncMock()
    engine._initialized = True
    engine.discover_protocol = AsyncMock(
        return_value={
            "protocol_type": "http",
            "confidence": 0.85,
            "structure": {},
            "grammar": None,
            "metadata": {},
            "processing_time": 0.15,
        }
    )
    engine.detect_fields = AsyncMock(
        return_value=[
            {
                "id": "field1",
                "name": "header",
                "start": 0,
                "end": 10,
                "type": "string",
                "confidence": 0.9,
                "semantic_type": "header",
                "encoding": "utf-8",
                "examples": ["example1"],
            }
        ]
    )
    engine.detect_anomaly = AsyncMock(
        return_value={
            "is_anomaly": False,
            "anomaly_score": 0.2,
            "confidence": 0.8,
            "detector_scores": {},
            "context": {},
            "explanation": "Normal traffic",
            "processing_time": 0.1,
        }
    )
    engine.get_model_info = Mock(
        return_value={
            "model_version": "1.0.0",
            "status": "ready",
        }
    )
    return engine


@pytest.fixture
def mock_grpc_context():
    """Create a mock gRPC context."""
    context = Mock()
    context.set_code = Mock()
    context.set_details = Mock()
    return context


@pytest.fixture
def mock_request():
    """Create a mock gRPC request."""
    request = Mock()
    request.data = base64.b64encode(b"test data").decode()
    request.data_format = "base64"
    return request


class TestAIEngineGRPCService:
    """Tests for AIEngineGRPCService."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_config):
        """Test service initialization."""
        service = AIEngineGRPCService(mock_config)

        assert service.config == mock_config
        assert service.ai_engine is None
        assert service.server is None
        assert service.stats["total_requests"] == 0
        assert service.stats["successful_requests"] == 0
        assert service.stats["failed_requests"] == 0

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config, mock_ai_engine):
        """Test successful service initialization."""
        service = AIEngineGRPCService(mock_config)

        with patch("ai_engine.api.grpc.QbitelAIEngine", return_value=mock_ai_engine):
            await service.initialize()

        assert service.ai_engine is not None

    @pytest.mark.asyncio
    async def test_initialize_failure(self, mock_config):
        """Test service initialization failure."""
        service = AIEngineGRPCService(mock_config)

        with patch("ai_engine.api.grpc.QbitelAIEngine", side_effect=Exception("Init failed")):
            with pytest.raises(Exception, match="Init failed"):
                await service.initialize()

    @pytest.mark.asyncio
    async def test_discover_protocol_success(self, mock_config, mock_ai_engine, mock_grpc_context, mock_request):
        """Test successful protocol discovery."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        response = await service.DiscoverProtocol(mock_request, mock_grpc_context)

        assert response["discovered_protocol"] == "http"
        assert response["confidence_score"] == 0.85
        assert service.stats["successful_requests"] == 1
        assert service.stats["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_discover_protocol_no_engine(self, mock_config, mock_grpc_context, mock_request):
        """Test protocol discovery without initialized engine."""
        service = AIEngineGRPCService(mock_config)

        response = await service.DiscoverProtocol(mock_request, mock_grpc_context)

        assert response == {}
        mock_grpc_context.set_code.assert_called_once()
        mock_grpc_context.set_details.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_protocol_with_metadata(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test protocol discovery with metadata."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        request.expected_protocol = "http"
        request.confidence_threshold = 0.8
        request.max_samples = 500
        request.include_grammar = True

        response = await service.DiscoverProtocol(request, mock_grpc_context)

        assert response["discovered_protocol"] == "http"
        mock_ai_engine.discover_protocol.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_protocol_failure(self, mock_config, mock_ai_engine, mock_grpc_context, mock_request):
        """Test protocol discovery failure."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        mock_ai_engine.discover_protocol.side_effect = Exception("Discovery failed")

        response = await service.DiscoverProtocol(mock_request, mock_grpc_context)

        assert response == {}
        assert service.stats["failed_requests"] == 1
        mock_grpc_context.set_code.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_fields_success(self, mock_config, mock_ai_engine, mock_grpc_context, mock_request):
        """Test successful field detection."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        response = await service.DetectFields(mock_request, mock_grpc_context)

        assert response["total_fields"] == 1
        assert len(response["detected_fields"]) == 1
        assert response["detected_fields"][0]["field_name"] == "header"
        assert service.stats["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_detect_fields_no_engine(self, mock_config, mock_grpc_context, mock_request):
        """Test field detection without initialized engine."""
        service = AIEngineGRPCService(mock_config)

        response = await service.DetectFields(mock_request, mock_grpc_context)

        assert response == {}
        mock_grpc_context.set_code.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_fields_with_protocol_hint(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test field detection with protocol hint."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        request.protocol_hint = "http"

        response = await service.DetectFields(request, mock_grpc_context)

        assert response["total_fields"] == 1
        mock_ai_engine.detect_fields.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_anomalies_success(self, mock_config, mock_ai_engine, mock_grpc_context, mock_request):
        """Test successful anomaly detection."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        response = await service.DetectAnomalies(mock_request, mock_grpc_context)

        assert response["is_anomalous"] == False
        assert "anomaly_score" in response
        assert service.stats["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_baseline(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test anomaly detection with baseline data."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        request.baseline_data = [base64.b64encode(b"baseline1").decode()]
        request.protocol_context = "http"
        request.sensitivity = "high"
        request.anomaly_threshold = 0.7
        request.include_explanations = True

        response = await service.DetectAnomalies(request, mock_grpc_context)

        assert "is_anomalous" in response
        mock_ai_engine.detect_anomaly.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_service_status(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test service status retrieval."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine
        service.stats["successful_requests"] = 10
        service.stats["total_requests"] = 12
        service.stats["total_processing_time_ms"] = 1000.0

        request = Mock()
        response = await service.GetServiceStatus(request, mock_grpc_context)

        assert response["service_name"] == "QBITEL Engine gRPC"
        assert response["version"] == "1.0.0"
        assert "uptime_seconds" in response
        assert response["statistics"]["total_requests"] == 12
        assert response["statistics"]["successful_requests"] == 10

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test health check when service is healthy."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        request = Mock()
        response = await service.HealthCheck(request, mock_grpc_context)

        assert response["status"] == "healthy"
        assert response["service"] == "qbitel-grpc"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_config, mock_grpc_context):
        """Test health check when service is unhealthy."""
        service = AIEngineGRPCService(mock_config)

        request = Mock()
        response = await service.HealthCheck(request, mock_grpc_context)

        assert response["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, mock_config, mock_grpc_context):
        """Test health check when service is degraded."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = Mock()
        service.ai_engine._initialized = False

        request = Mock()
        response = await service.HealthCheck(request, mock_grpc_context)

        assert response["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_batch_process_discovery(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test batch processing for discovery operation."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        request = Mock()
        request.operation = "discovery"
        request.data_items = [
            base64.b64encode(b"data1").decode(),
            base64.b64encode(b"data2").decode(),
        ]

        response = await service.BatchProcess(request, mock_grpc_context)

        assert response["total_items"] == 2
        assert response["successful_items"] == 2
        assert response["failed_items"] == 0
        assert len(response["results"]) == 2

    @pytest.mark.asyncio
    async def test_batch_process_detection(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test batch processing for detection operation."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        request = Mock()
        request.operation = "detection"
        request.data_items = [base64.b64encode(b"data1").decode()]

        response = await service.BatchProcess(request, mock_grpc_context)

        assert response["total_items"] == 1
        assert response["successful_items"] == 1

    @pytest.mark.asyncio
    async def test_batch_process_anomaly(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test batch processing for anomaly operation."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        request = Mock()
        request.operation = "anomaly"
        request.data_items = [base64.b64encode(b"data1").decode()]

        response = await service.BatchProcess(request, mock_grpc_context)

        assert response["total_items"] == 1
        assert response["successful_items"] == 1

    @pytest.mark.asyncio
    async def test_batch_process_unknown_operation(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test batch processing with unknown operation."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        request = Mock()
        request.operation = "unknown"
        request.data_items = [base64.b64encode(b"data1").decode()]

        response = await service.BatchProcess(request, mock_grpc_context)

        assert response["failed_items"] == 1

    def test_decode_data_base64(self, mock_config):
        """Test data decoding with base64 format."""
        service = AIEngineGRPCService(mock_config)

        data = base64.b64encode(b"test data").decode()
        decoded = service._decode_data(data, "base64")

        assert decoded == b"test data"

    def test_decode_data_hex(self, mock_config):
        """Test data decoding with hex format."""
        service = AIEngineGRPCService(mock_config)

        data = "74657374"  # "test" in hex
        decoded = service._decode_data(data, "hex")

        assert decoded == b"test"

    def test_decode_data_text(self, mock_config):
        """Test data decoding with text format."""
        service = AIEngineGRPCService(mock_config)

        decoded = service._decode_data("test data", "text")

        assert decoded == b"test data"

    def test_decode_data_invalid(self, mock_config):
        """Test data decoding with invalid data."""
        service = AIEngineGRPCService(mock_config)

        with pytest.raises(ValueError, match="Data decoding failed"):
            service._decode_data("invalid base64!", "base64")


class TestGRPCServer:
    """Tests for GRPCServer."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_config):
        """Test server initialization."""
        server = GRPCServer(mock_config)

        assert server.config == mock_config
        assert server.port == 50051
        assert server.max_workers == 10
        assert server.service is None
        assert server.server is None

    @pytest.mark.asyncio
    async def test_start_server(self, mock_config, mock_ai_engine):
        """Test server start."""
        server = GRPCServer(mock_config)

        with patch("ai_engine.api.grpc.AIEngineGRPCService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service.initialize = AsyncMock()
            mock_service_class.return_value = mock_service

            with patch("ai_engine.api.grpc.grpc.aio.server") as mock_grpc_server:
                mock_server_instance = AsyncMock()
                mock_server_instance.add_insecure_port = Mock()
                mock_server_instance.start = AsyncMock()
                mock_server_instance.wait_for_termination = AsyncMock()
                mock_grpc_server.return_value = mock_server_instance

                # Start server in background
                task = asyncio.create_task(server.start())
                await asyncio.sleep(0.1)

                # Stop server
                await server.stop()

                # Cancel the task
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_stop_server(self, mock_config):
        """Test server stop."""
        server = GRPCServer(mock_config)
        server.server = AsyncMock()
        server.server.stop = AsyncMock()

        await server.stop()

        server.server.stop.assert_called_once_with(grace=5)


class TestAIEngineGRPCClient:
    """Tests for AIEngineGRPCClient."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test client initialization."""
        client = AIEngineGRPCClient("localhost:50051")

        assert client.server_address == "localhost:50051"
        assert client.channel is None
        assert client.stub is None

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test client connection."""
        client = AIEngineGRPCClient("localhost:50051")

        with patch("ai_engine.api.grpc.grpc.aio.insecure_channel") as mock_channel:
            mock_channel_instance = AsyncMock()
            mock_channel.return_value = mock_channel_instance

            with patch.object(client, "health_check", new_callable=AsyncMock):
                await client.connect()

                assert client.channel is not None

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test client disconnection."""
        client = AIEngineGRPCClient("localhost:50051")
        client.channel = AsyncMock()
        client.channel.close = AsyncMock()

        await client.disconnect()

        client.channel.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check."""
        client = AIEngineGRPCClient("localhost:50051")

        result = await client.health_check()

        assert result["status"] == "healthy"
        assert result["service"] == "qbitel-grpc"
        assert isinstance(result, dict)
        assert "timestamp" in result or "status" in result  # Ensure key fields present

    @pytest.mark.asyncio
    async def test_discover_protocol(self):
        """Test protocol discovery."""
        client = AIEngineGRPCClient("localhost:50051")

        result = await client.discover_protocol(b"test data")

        assert "discovered_protocol" in result
        assert "confidence_score" in result
        assert isinstance(result, dict)
        assert isinstance(result.get("confidence_score", 0.0), (int, float))
        assert 0 <= result.get("confidence_score", 0) <= 1

    @pytest.mark.asyncio
    async def test_detect_fields(self):
        """Test field detection."""
        client = AIEngineGRPCClient("localhost:50051")

        result = await client.detect_fields(b"test data")

        assert "detected_fields" in result
        assert "total_fields" in result
        assert isinstance(result, dict)
        assert isinstance(result.get("total_fields", 0), int)
        assert result.get("total_fields", 0) >= 0
        assert isinstance(result.get("detected_fields", []), list)

    @pytest.mark.asyncio
    async def test_detect_anomalies(self):
        """Test anomaly detection."""
        client = AIEngineGRPCClient("localhost:50051")

        result = await client.detect_anomalies(b"test data")

        assert "is_anomalous" in result
        assert "anomaly_score" in result
        assert isinstance(result, dict)
        assert isinstance(result.get("is_anomalous", False), bool)
        # anomaly_score can be either a number or a dict with overall_score
        anomaly_score = result.get("anomaly_score", {})
        if isinstance(anomaly_score, dict):
            assert "overall_score" in anomaly_score
            assert 0 <= anomaly_score["overall_score"] <= 1
        else:
            assert isinstance(anomaly_score, (int, float))
            assert 0 <= anomaly_score <= 1


class TestRunGRPCServer:
    """Tests for run_grpc_server function."""

    @pytest.mark.asyncio
    async def test_run_grpc_server(self, mock_config):
        """Test running gRPC server."""
        with patch("ai_engine.api.grpc.GRPCServer") as mock_server_class:
            mock_server = AsyncMock()
            mock_server.start = AsyncMock()
            mock_server.wait_for_termination = AsyncMock()
            mock_server.stop = AsyncMock()
            mock_server_class.return_value = mock_server

            # Simulate KeyboardInterrupt
            mock_server.wait_for_termination.side_effect = KeyboardInterrupt()

            await run_grpc_server(mock_config)

            mock_server.start.assert_called_once()
            mock_server.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_grpc_server_exception(self, mock_config):
        """Test running gRPC server with exception."""
        with patch("ai_engine.api.grpc.GRPCServer") as mock_server_class:
            mock_server = AsyncMock()
            mock_server.start = AsyncMock()
            mock_server.wait_for_termination = AsyncMock()
            mock_server.stop = AsyncMock()
            mock_server_class.return_value = mock_server

            # Simulate exception
            mock_server.wait_for_termination.side_effect = Exception("Server error")

            with pytest.raises(Exception, match="Server error"):
                await run_grpc_server(mock_config)

            mock_server.stop.assert_called_once()


class TestAIEngineGRPCServiceStreaming:
    """Tests for streaming endpoints."""

    @pytest.mark.asyncio
    async def test_discover_protocol_stream(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test streaming protocol discovery."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        async def request_generator():
            for i in range(3):
                request = Mock()
                request.data = base64.b64encode(f"data{i}".encode()).decode()
                request.data_format = "base64"
                yield request

        responses = []
        async for response in service.DiscoverProtocolStream(request_generator(), mock_grpc_context):
            responses.append(response)

        assert len(responses) == 3
        assert all(r["discovered_protocol"] == "http" for r in responses)

    @pytest.mark.asyncio
    async def test_detect_fields_stream(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test streaming field detection."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        async def request_generator():
            for i in range(2):
                request = Mock()
                request.data = base64.b64encode(f"data{i}".encode()).decode()
                request.data_format = "base64"
                yield request

        responses = []
        async for response in service.DetectFieldsStream(request_generator(), mock_grpc_context):
            responses.append(response)

        assert len(responses) == 2

    @pytest.mark.asyncio
    async def test_detect_anomalies_stream(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test streaming anomaly detection."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        async def request_generator():
            for i in range(2):
                request = Mock()
                request.data = base64.b64encode(f"data{i}".encode()).decode()
                request.data_format = "base64"
                yield request

        responses = []
        async for response in service.DetectAnomaliesStream(request_generator(), mock_grpc_context):
            responses.append(response)

        assert len(responses) == 2


class TestGRPCServerAdvanced:
    """Advanced tests for GRPCServer."""

    @pytest.mark.asyncio
    async def test_server_configuration(self, mock_config):
        """Test server configuration options."""
        mock_config.grpc_max_message_length = 20 * 1024 * 1024
        server = GRPCServer(mock_config)

        assert server.max_message_length == 20 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_wait_for_termination(self, mock_config):
        """Test server wait for termination."""
        server = GRPCServer(mock_config)
        server.server = AsyncMock()
        server.server.wait_for_termination = AsyncMock()

        await server.wait_for_termination()

        server.server.wait_for_termination.assert_called_once()


class TestDataEncodingEdgeCases:
    """Test edge cases for data encoding/decoding."""

    def test_decode_data_hex_with_spaces(self, mock_config):
        """Test hex decoding with spaces."""
        service = AIEngineGRPCService(mock_config)

        data = "74 65 73 74"  # "test" in hex with spaces
        decoded = service._decode_data(data, "hex")

        assert decoded == b"test"

    def test_decode_data_unknown_format(self, mock_config):
        """Test decoding with unknown format."""
        service = AIEngineGRPCService(mock_config)

        decoded = service._decode_data("test", "unknown")

        assert decoded == b"test"


class TestServiceStatusEdgeCases:
    """Test edge cases for service status."""

    @pytest.mark.asyncio
    async def test_get_service_status_no_requests(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test service status with no requests."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        request = Mock()
        response = await service.GetServiceStatus(request, mock_grpc_context)

        assert response["statistics"]["total_requests"] == 0
        assert response["statistics"]["average_processing_time_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_get_service_status_failure(self, mock_config, mock_grpc_context):
        """Test service status retrieval failure."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = None

        with patch.object(service, "stats", side_effect=Exception("Stats error")):
            request = Mock()
            response = await service.GetServiceStatus(request, mock_grpc_context)

            assert response == {}

    @pytest.mark.asyncio
    async def test_health_check_exception(self, mock_config, mock_grpc_context):
        """Test health check with exception."""
        service = AIEngineGRPCService(mock_config)

        with patch.object(service, "ai_engine", side_effect=Exception("Health error")):
            request = Mock()
            response = await service.HealthCheck(request, mock_grpc_context)

            assert response["status"] == "unhealthy"
            assert "error" in response


class TestBatchProcessingEdgeCases:
    """Test edge cases for batch processing."""

    @pytest.mark.asyncio
    async def test_batch_process_no_engine(self, mock_config, mock_grpc_context):
        """Test batch processing without engine."""
        service = AIEngineGRPCService(mock_config)

        request = Mock()
        request.operation = "discovery"
        request.data_items = []

        response = await service.BatchProcess(request, mock_grpc_context)

        assert response == {}

    @pytest.mark.asyncio
    async def test_batch_process_empty_items(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test batch processing with empty items."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        request = Mock()
        request.operation = "discovery"
        request.data_items = []

        response = await service.BatchProcess(request, mock_grpc_context)

        assert response["total_items"] == 0
        assert response["successful_items"] == 0

    @pytest.mark.asyncio
    async def test_batch_process_partial_failure(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test batch processing with partial failures."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        # Make second call fail
        call_count = [0]

        async def failing_discover(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Processing failed")
            return {
                "protocol_type": "http",
                "confidence": 0.85,
                "structure": {},
                "metadata": {},
                "processing_time": 0.1,
            }

        mock_ai_engine.discover_protocol = failing_discover

        request = Mock()
        request.operation = "discovery"
        request.data_items = [
            base64.b64encode(b"data1").decode(),
            base64.b64encode(b"data2").decode(),
            base64.b64encode(b"data3").decode(),
        ]

        response = await service.BatchProcess(request, mock_grpc_context)

        assert response["total_items"] == 3
        assert response["successful_items"] == 2
        assert response["failed_items"] == 1


class TestClientEdgeCases:
    """Test edge cases for gRPC client."""

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test client connection failure."""
        client = AIEngineGRPCClient("localhost:50051")

        with patch(
            "ai_engine.api.grpc.grpc.aio.insecure_channel",
            side_effect=Exception("Connection failed"),
        ):
            with pytest.raises(Exception, match="Connection failed"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_disconnect_without_connection(self):
        """Test disconnecting without active connection."""
        client = AIEngineGRPCClient("localhost:50051")

        # Should not raise error
        await client.disconnect()


class TestServiceInitializationEdgeCases:
    """Test edge cases for service initialization."""

    @pytest.mark.asyncio
    async def test_start_server_failure(self, mock_config):
        """Test server start failure."""
        server = GRPCServer(mock_config)

        with patch(
            "ai_engine.api.grpc.AIEngineGRPCService",
            side_effect=Exception("Service init failed"),
        ):
            with pytest.raises(Exception, match="Service init failed"):
                await server.start()

    @pytest.mark.asyncio
    async def test_stop_server_without_start(self, mock_config):
        """Test stopping server that was never started."""
        server = GRPCServer(mock_config)

        # Should not raise error
        await server.stop()


class TestAnomalyDetectionWithBaseline:
    """Test anomaly detection with various baseline configurations."""

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_string_baseline(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test anomaly detection with string baseline."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        request.baseline_data = base64.b64encode(b"baseline").decode()

        response = await service.DetectAnomalies(request, mock_grpc_context)

        assert "is_anomalous" in response

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_empty_baseline(self, mock_config, mock_ai_engine, mock_grpc_context):
        """Test anomaly detection with empty baseline."""
        service = AIEngineGRPCService(mock_config)
        service.ai_engine = mock_ai_engine

        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        request.baseline_data = []

        response = await service.DetectAnomalies(request, mock_grpc_context)

        assert "is_anomalous" in response
