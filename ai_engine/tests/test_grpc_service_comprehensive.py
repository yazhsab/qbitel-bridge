"""
Comprehensive Unit Tests for gRPC Service
Tests for ai_engine/api/grpc.py
"""

import pytest
import asyncio
import base64
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from ai_engine.api.grpc import (
    AIEngineGRPCService,
    GRPCServer,
    AIEngineGRPCClient,
    run_grpc_server,
)
from ai_engine.core.config import Config
from ai_engine.core.exceptions import AIEngineException


class TestAIEngineGRPCService:
    """Test AIEngineGRPCService class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.grpc_port = 50051
        config.grpc_max_workers = 10
        return config

    @pytest.fixture
    def service(self, config):
        """Create gRPC service instance."""
        return AIEngineGRPCService(config)

    def test_service_initialization(self, service):
        """Test service initialization."""
        assert service.config is not None
        assert service.ai_engine is None
        assert service.server is None
        assert isinstance(service.stats, dict)
        assert service.stats["total_requests"] == 0
        assert service.stats["successful_requests"] == 0
        assert service.stats["failed_requests"] == 0

    @pytest.mark.asyncio
    async def test_initialize_service(self, service):
        """Test service initialization."""
        with patch("ai_engine.api.grpc.CronosAIEngine") as mock_engine:
            mock_engine_instance = AsyncMock()
            mock_engine.return_value = mock_engine_instance

            await service.initialize()

            assert service.ai_engine is not None
            mock_engine_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_service_failure(self, service):
        """Test service initialization failure."""
        with patch(
            "ai_engine.api.grpc.CronosAIEngine", side_effect=Exception("Init failed")
        ):
            with pytest.raises(Exception):
                await service.initialize()

    @pytest.mark.asyncio
    async def test_discover_protocol(self, service):
        """Test protocol discovery endpoint."""
        # Setup
        service.ai_engine = AsyncMock()
        service.ai_engine.discover_protocol = AsyncMock(
            return_value={
                "protocol_type": "http",
                "confidence": 0.95,
                "structure": {"method": "GET"},
                "processing_time": 0.15,
            }
        )

        # Create mock request
        request = Mock()
        request.data = base64.b64encode(b"GET / HTTP/1.1").decode()
        request.data_format = "base64"
        request.expected_protocol = None
        request.confidence_threshold = 0.7

        context = Mock()

        # Execute
        response = await service.DiscoverProtocol(request, context)

        # Verify
        assert response["discovered_protocol"] == "http"
        assert response["confidence_score"] == 0.95
        assert service.stats["successful_requests"] == 1
        assert service.stats["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_discover_protocol_not_initialized(self, service):
        """Test protocol discovery when engine not initialized."""
        request = Mock()
        context = Mock()

        response = await service.DiscoverProtocol(request, context)

        assert response == {}
        context.set_code.assert_called()

    @pytest.mark.asyncio
    async def test_discover_protocol_failure(self, service):
        """Test protocol discovery failure handling."""
        service.ai_engine = AsyncMock()
        service.ai_engine.discover_protocol = AsyncMock(
            side_effect=Exception("Discovery failed")
        )

        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"

        context = Mock()

        response = await service.DiscoverProtocol(request, context)

        assert response == {}
        assert service.stats["failed_requests"] == 1
        context.set_code.assert_called()

    @pytest.mark.asyncio
    async def test_discover_protocol_stream(self, service):
        """Test streaming protocol discovery."""
        service.ai_engine = AsyncMock()
        service.ai_engine.discover_protocol = AsyncMock(
            return_value={"protocol_type": "http", "confidence": 0.95}
        )

        async def request_generator():
            for i in range(3):
                request = Mock()
                request.data = base64.b64encode(f"data_{i}".encode()).decode()
                request.data_format = "base64"
                yield request

        context = Mock()

        responses = []
        async for response in service.DiscoverProtocolStream(
            request_generator(), context
        ):
            responses.append(response)

        assert len(responses) == 3

    @pytest.mark.asyncio
    async def test_detect_fields(self, service):
        """Test field detection endpoint."""
        service.ai_engine = AsyncMock()
        service.ai_engine.detect_fields = AsyncMock(
            return_value=[
                {
                    "id": "field1",
                    "name": "header",
                    "start": 0,
                    "end": 10,
                    "type": "string",
                    "confidence": 0.9,
                }
            ]
        )

        request = Mock()
        request.data = base64.b64encode(b"test data").decode()
        request.data_format = "base64"
        request.protocol_hint = "http"

        context = Mock()

        response = await service.DetectFields(request, context)

        assert "detected_fields" in response
        assert len(response["detected_fields"]) == 1
        assert response["total_fields"] == 1
        assert service.stats["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_detect_fields_stream(self, service):
        """Test streaming field detection."""
        service.ai_engine = AsyncMock()
        service.ai_engine.detect_fields = AsyncMock(return_value=[])

        async def request_generator():
            for i in range(2):
                request = Mock()
                request.data = base64.b64encode(f"data_{i}".encode()).decode()
                request.data_format = "base64"
                yield request

        context = Mock()

        responses = []
        async for response in service.DetectFieldsStream(request_generator(), context):
            responses.append(response)

        assert len(responses) == 2

    @pytest.mark.asyncio
    async def test_detect_anomalies(self, service):
        """Test anomaly detection endpoint."""
        service.ai_engine = AsyncMock()
        service.ai_engine.detect_anomaly = AsyncMock(
            return_value={
                "is_anomaly": True,
                "anomaly_score": 0.85,
                "confidence": 0.9,
                "explanation": "Unusual pattern detected",
                "processing_time": 0.18,
            }
        )

        request = Mock()
        request.data = base64.b64encode(b"anomalous data").decode()
        request.data_format = "base64"
        request.protocol_context = "http"
        request.sensitivity = "high"

        context = Mock()

        response = await service.DetectAnomalies(request, context)

        assert response["is_anomalous"] is True
        assert "anomaly_score" in response
        assert service.stats["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_baseline(self, service):
        """Test anomaly detection with baseline data."""
        service.ai_engine = AsyncMock()
        service.ai_engine.detect_anomaly = AsyncMock(
            return_value={"is_anomaly": False, "anomaly_score": 0.2, "confidence": 0.95}
        )

        request = Mock()
        request.data = base64.b64encode(b"normal data").decode()
        request.data_format = "base64"
        request.baseline_data = [base64.b64encode(b"baseline1").decode()]

        context = Mock()

        response = await service.DetectAnomalies(request, context)

        assert response["is_anomalous"] is False

    @pytest.mark.asyncio
    async def test_get_service_status(self, service):
        """Test getting service status."""
        service.ai_engine = Mock()
        service.ai_engine.get_model_info = Mock(return_value={"models": ["model1"]})
        service.stats["total_requests"] = 100
        service.stats["successful_requests"] = 95
        service.stats["failed_requests"] = 5
        service.stats["total_processing_time_ms"] = 10000

        request = Mock()
        context = Mock()

        response = await service.GetServiceStatus(request, context)

        assert response["service_name"] == "CRONOS AI Engine gRPC"
        assert response["statistics"]["total_requests"] == 100
        assert response["statistics"]["successful_requests"] == 95
        assert response["statistics"]["success_rate"] == 0.95
        assert "uptime_seconds" in response

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, service):
        """Test health check when service is healthy."""
        service.ai_engine = Mock()
        service.ai_engine._initialized = True

        request = Mock()
        context = Mock()

        response = await service.HealthCheck(request, context)

        assert response["status"] == "healthy"
        assert response["service"] == "cronos-ai-grpc"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, service):
        """Test health check when service is unhealthy."""
        service.ai_engine = None

        request = Mock()
        context = Mock()

        response = await service.HealthCheck(request, context)

        assert response["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, service):
        """Test health check when service is degraded."""
        service.ai_engine = Mock()
        service.ai_engine._initialized = False

        request = Mock()
        context = Mock()

        response = await service.HealthCheck(request, context)

        assert response["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_batch_process_discovery(self, service):
        """Test batch processing for discovery."""
        service.ai_engine = AsyncMock()
        service.ai_engine.discover_protocol = AsyncMock(
            return_value={"protocol_type": "http", "confidence": 0.9}
        )

        request = Mock()
        request.operation = "discovery"
        request.data_items = [
            base64.b64encode(b"data1").decode(),
            base64.b64encode(b"data2").decode(),
        ]

        context = Mock()

        response = await service.BatchProcess(request, context)

        assert response["total_items"] == 2
        assert response["successful_items"] == 2
        assert len(response["results"]) == 2

    @pytest.mark.asyncio
    async def test_batch_process_detection(self, service):
        """Test batch processing for field detection."""
        service.ai_engine = AsyncMock()
        service.ai_engine.detect_fields = AsyncMock(return_value=[])

        request = Mock()
        request.operation = "detection"
        request.data_items = [base64.b64encode(b"data1").decode()]

        context = Mock()

        response = await service.BatchProcess(request, context)

        assert response["total_items"] == 1
        assert response["successful_items"] == 1

    @pytest.mark.asyncio
    async def test_batch_process_anomaly(self, service):
        """Test batch processing for anomaly detection."""
        service.ai_engine = AsyncMock()
        service.ai_engine.detect_anomaly = AsyncMock(return_value={"is_anomaly": False})

        request = Mock()
        request.operation = "anomaly"
        request.data_items = [base64.b64encode(b"data1").decode()]

        context = Mock()

        response = await service.BatchProcess(request, context)

        assert response["total_items"] == 1

    @pytest.mark.asyncio
    async def test_batch_process_with_failures(self, service):
        """Test batch processing with some failures."""
        service.ai_engine = AsyncMock()
        service.ai_engine.discover_protocol = AsyncMock(
            side_effect=[
                {"protocol_type": "http"},
                Exception("Failed"),
                {"protocol_type": "tcp"},
            ]
        )

        request = Mock()
        request.operation = "discovery"
        request.data_items = [
            base64.b64encode(b"data1").decode(),
            base64.b64encode(b"data2").decode(),
            base64.b64encode(b"data3").decode(),
        ]

        context = Mock()

        response = await service.BatchProcess(request, context)

        assert response["total_items"] == 3
        assert response["successful_items"] == 2
        assert response["failed_items"] == 1

    def test_decode_data_base64(self, service):
        """Test decoding base64 data."""
        encoded = base64.b64encode(b"test data").decode()

        decoded = service._decode_data(encoded, "base64")

        assert decoded == b"test data"

    def test_decode_data_hex(self, service):
        """Test decoding hex data."""
        hex_data = "74657374"

        decoded = service._decode_data(hex_data, "hex")

        assert decoded == b"test"

    def test_decode_data_text(self, service):
        """Test decoding text data."""
        text_data = "test data"

        decoded = service._decode_data(text_data, "text")

        assert decoded == b"test data"

    def test_decode_data_invalid(self, service):
        """Test decoding invalid data."""
        with pytest.raises(ValueError):
            service._decode_data("invalid base64!", "base64")


class TestGRPCServer:
    """Test GRPCServer class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.grpc_port = 50051
        config.grpc_max_workers = 10
        config.grpc_max_message_length = 10 * 1024 * 1024
        return config

    @pytest.fixture
    def grpc_server(self, config):
        """Create gRPC server instance."""
        return GRPCServer(config)

    def test_server_initialization(self, grpc_server):
        """Test server initialization."""
        assert grpc_server.config is not None
        assert grpc_server.service is None
        assert grpc_server.server is None
        assert grpc_server.port == 50051
        assert grpc_server.max_workers == 10

    @pytest.mark.asyncio
    async def test_start_server(self, grpc_server):
        """Test starting gRPC server."""
        with patch("grpc.aio.server") as mock_server:
            mock_server_instance = AsyncMock()
            mock_server_instance.add_insecure_port = Mock()
            mock_server_instance.start = AsyncMock()
            mock_server.return_value = mock_server_instance

            with patch.object(grpc_server, "service", None):
                with patch("ai_engine.api.grpc.AIEngineGRPCService") as mock_service:
                    mock_service_instance = AsyncMock()
                    mock_service.return_value = mock_service_instance

                    # Start server in background
                    task = asyncio.create_task(grpc_server.start())
                    await asyncio.sleep(0.1)
                    task.cancel()

                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                    mock_service_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_server(self, grpc_server):
        """Test stopping gRPC server."""
        mock_server = AsyncMock()
        mock_server.stop = AsyncMock()
        grpc_server.server = mock_server

        await grpc_server.stop()

        mock_server.stop.assert_called_once_with(grace=5)

    @pytest.mark.asyncio
    async def test_stop_server_with_custom_grace(self, grpc_server):
        """Test stopping server with custom grace period."""
        mock_server = AsyncMock()
        mock_server.stop = AsyncMock()
        grpc_server.server = mock_server

        await grpc_server.stop(grace_period=10)

        mock_server.stop.assert_called_once_with(grace=10)


class TestAIEngineGRPCClient:
    """Test AIEngineGRPCClient class."""

    @pytest.fixture
    def client(self):
        """Create gRPC client instance."""
        return AIEngineGRPCClient("localhost:50051")

    def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.server_address == "localhost:50051"
        assert client.channel is None
        assert client.stub is None

    @pytest.mark.asyncio
    async def test_connect_client(self, client):
        """Test connecting client to server."""
        with patch("grpc.aio.insecure_channel") as mock_channel:
            mock_channel_instance = AsyncMock()
            mock_channel.return_value = mock_channel_instance

            with patch.object(client, "health_check", new_callable=AsyncMock):
                await client.connect()

                assert client.channel is not None

    @pytest.mark.asyncio
    async def test_connect_client_failure(self, client):
        """Test client connection failure."""
        with patch(
            "grpc.aio.insecure_channel", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(Exception):
                await client.connect()

    @pytest.mark.asyncio
    async def test_disconnect_client(self, client):
        """Test disconnecting client."""
        mock_channel = AsyncMock()
        mock_channel.close = AsyncMock()
        client.channel = mock_channel

        await client.disconnect()

        mock_channel.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check."""
        result = await client.health_check()

        assert result["status"] == "healthy"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_discover_protocol(self, client):
        """Test protocol discovery call."""
        result = await client.discover_protocol(b"test data")

        assert "discovered_protocol" in result
        assert "confidence_score" in result

    @pytest.mark.asyncio
    async def test_detect_fields(self, client):
        """Test field detection call."""
        result = await client.detect_fields(b"test data")

        assert "detected_fields" in result
        assert "total_fields" in result

    @pytest.mark.asyncio
    async def test_detect_anomalies(self, client):
        """Test anomaly detection call."""
        result = await client.detect_anomalies(b"test data")

        assert "is_anomalous" in result
        assert "anomaly_score" in result


class TestRunGRPCServer:
    """Test run_grpc_server function."""

    @pytest.mark.asyncio
    async def test_run_grpc_server(self):
        """Test running gRPC server."""
        config = Mock(spec=Config)

        with patch("ai_engine.api.grpc.GRPCServer") as mock_server_class:
            mock_server = AsyncMock()
            mock_server.start = AsyncMock()
            mock_server.wait_for_termination = AsyncMock()
            mock_server_class.return_value = mock_server

            # Run server briefly
            task = asyncio.create_task(run_grpc_server(config))
            await asyncio.sleep(0.1)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            mock_server.start.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
