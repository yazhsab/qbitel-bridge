"""
CRONOS AI Engine - gRPC Service

This module implements the gRPC service for high-performance AI Engine communication.
"""

import logging
import time
import asyncio
import base64
from typing import Dict, Any, Optional
from concurrent import futures
import grpc

from ..core.engine import CronosAIEngine
from ..core.config import Config
from ..core.exceptions import AIEngineException


# gRPC service definition (would typically be generated from .proto file)
class AIEngineGRPCService:
    """
    gRPC service for CRONOS AI Engine.

    This service provides high-performance RPC endpoints for all
    AI Engine functionality with streaming capabilities.
    """

    def __init__(self, config: Config):
        """Initialize gRPC service."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize AI Engine
        self.ai_engine: Optional[CronosAIEngine] = None

        # gRPC server
        self.server: Optional[grpc.aio.Server] = None

        # Service statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time_ms": 0.0,
            "start_time": time.time(),
        }

        self.logger.info("AIEngineGRPCService initialized")

    async def initialize(self) -> None:
        """Initialize the gRPC service."""
        try:
            # Initialize AI Engine
            self.ai_engine = CronosAIEngine(self.config)
            await self.ai_engine.initialize()

            self.logger.info("gRPC service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize gRPC service: {e}")
            raise

    async def start_server(self, port: int = 50051) -> None:
        """Start the gRPC server."""
        try:
            # Create server
            self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))

            # Add service to server (would use generated code in real implementation)
            # add_AIEngineServicer_to_server(self, self.server)

            # Add insecure port
            listen_addr = f"[::]:{port}"
            self.server.add_insecure_port(listen_addr)

            # Start server
            await self.server.start()
            self.logger.info(f"gRPC server started on {listen_addr}")

            # Keep server running
            await self.server.wait_for_termination()

        except Exception as e:
            self.logger.error(f"gRPC server failed: {e}")
            raise

    async def stop_server(self) -> None:
        """Stop the gRPC server."""
        if self.server:
            await self.server.stop(grace=5)
            self.logger.info("gRPC server stopped")

    # Protocol Discovery Methods

    async def DiscoverProtocol(self, request, context) -> Dict[str, Any]:
        """
        gRPC endpoint for protocol discovery.

        Args:
            request: ProtocolDiscoveryRequest
            context: gRPC context

        Returns:
            ProtocolDiscoveryResponse
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            if not self.ai_engine:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("AI Engine not initialized")
                return {}

            # Decode input data
            data_bytes = self._decode_data(request.data, request.data_format)

            metadata = {
                "expected_protocol": getattr(request, "expected_protocol", None),
                "confidence_threshold": getattr(request, "confidence_threshold", 0.7),
                "max_samples": getattr(request, "max_samples", 1000),
                "include_grammar": getattr(request, "include_grammar", False),
            }
            # Remove nulls to avoid clutter
            metadata = {k: v for k, v in metadata.items() if v is not None}

            result = await self.ai_engine.discover_protocol(
                data_bytes, metadata=metadata if metadata else None
            )

            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.stats["successful_requests"] += 1
            self.stats["total_processing_time_ms"] += processing_time

            response = {
                "discovered_protocol": result.get("protocol_type", "unknown"),
                "confidence_score": result.get("confidence", 0.0),
                "structure": result.get("structure", {}),
                "grammar": result.get("grammar"),
                "metadata": result.get("metadata", {}),
                "processing_time_ms": result.get("processing_time", 0.0) * 1000,
            }

            self.logger.info(f"Protocol discovery completed in {processing_time:.2f}ms")
            return response

        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"Protocol discovery failed: {e}")

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Protocol discovery failed: {e}")
            return {}

    async def DiscoverProtocolStream(self, request_iterator, context):
        """
        Streaming protocol discovery for multiple samples.

        Args:
            request_iterator: Stream of ProtocolDiscoveryRequest
            context: gRPC context

        Yields:
            ProtocolDiscoveryResponse for each request
        """
        async for request in request_iterator:
            try:
                response = await self.DiscoverProtocol(request, context)
                yield response

            except Exception as e:
                self.logger.error(f"Streaming protocol discovery failed: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Streaming error: {e}")
                break

    # Field Detection Methods

    async def DetectFields(self, request, context) -> Dict[str, Any]:
        """
        gRPC endpoint for field detection.

        Args:
            request: FieldDetectionRequest
            context: gRPC context

        Returns:
            FieldDetectionResponse
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            if not self.ai_engine:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("AI Engine not initialized")
                return {}

            # Decode input data
            data_bytes = self._decode_data(request.data, request.data_format)

            protocol_hint = getattr(request, "protocol_hint", None)
            result = await self.ai_engine.detect_fields(data_bytes, protocol_hint)

            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.stats["successful_requests"] += 1
            self.stats["total_processing_time_ms"] += processing_time

            # Build response
            detected_fields = []
            for field in result:
                detected_fields.append(
                    {
                        "field_id": field.get("id"),
                        "field_name": field.get("name", ""),
                        "start_offset": field.get("start"),
                        "end_offset": field.get("end"),
                        "field_type": field.get("type", "unknown"),
                        "confidence": field.get("confidence"),
                        "semantic_type": field.get("semantic_type", ""),
                        "encoding": field.get("encoding", ""),
                        "examples": field.get("examples", []),
                    }
                )

            response = {
                "detected_fields": detected_fields,
                "total_fields": len(detected_fields),
                "processing_time_ms": processing_time,
            }

            self.logger.info(f"Field detection completed in {processing_time:.2f}ms")
            return response

        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"Field detection failed: {e}")

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Field detection failed: {e}")
            return {}

    async def DetectFieldsStream(self, request_iterator, context):
        """
        Streaming field detection for multiple samples.

        Args:
            request_iterator: Stream of FieldDetectionRequest
            context: gRPC context

        Yields:
            FieldDetectionResponse for each request
        """
        async for request in request_iterator:
            try:
                response = await self.DetectFields(request, context)
                yield response

            except Exception as e:
                self.logger.error(f"Streaming field detection failed: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Streaming error: {e}")
                break

    # Anomaly Detection Methods

    async def DetectAnomalies(self, request, context) -> Dict[str, Any]:
        """
        gRPC endpoint for anomaly detection.

        Args:
            request: AnomalyDetectionRequest
            context: gRPC context

        Returns:
            AnomalyDetectionResponse
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            if not self.ai_engine:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("AI Engine not initialized")
                return {}

            # Decode input data
            data_bytes = self._decode_data(request.data, request.data_format)

            # Decode baseline data if provided
            baseline_bytes = None
            if hasattr(request, "baseline_data") and request.baseline_data:
                baseline_bytes = [
                    self._decode_data(baseline, request.data_format)
                    for baseline in request.baseline_data
                ]

            context = {
                "protocol_context": getattr(request, "protocol_context", None),
                "sensitivity": getattr(request, "sensitivity", "medium"),
                "baseline_data": baseline_bytes,
                "anomaly_threshold": getattr(request, "anomaly_threshold", 0.5),
                "include_explanations": getattr(request, "include_explanations", True),
            }
            context = {k: v for k, v in context.items() if v is not None}

            result = await self.ai_engine.detect_anomaly(
                data_bytes, context=context if context else None
            )

            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.stats["successful_requests"] += 1
            self.stats["total_processing_time_ms"] += processing_time

            # Build anomaly score
            anomaly_score = {
                "score": result.get("anomaly_score", 0.0),
                "confidence": result.get("confidence", 0.0),
                "threshold": context.get("anomaly_threshold") if context else None,
            }

            explanations = []
            explanation_text = result.get("explanation")
            if explanation_text:
                explanations.append(
                    {
                        "description": explanation_text,
                        "confidence": result.get("confidence", 0.0),
                    }
                )

            response = {
                "is_anomalous": result.get("is_anomaly", False),
                "anomaly_score": anomaly_score,
                "detector_scores": result.get("detector_scores", {}),
                "context": result.get("context", {}),
                "anomaly_explanations": explanations,
                "processing_time_ms": result.get("processing_time", 0.0) * 1000,
            }

            self.logger.info(f"Anomaly detection completed in {processing_time:.2f}ms")
            return response

        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"Anomaly detection failed: {e}")

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Anomaly detection failed: {e}")
            return {}

    async def DetectAnomaliesStream(self, request_iterator, context):
        """
        Streaming anomaly detection for multiple samples.

        Args:
            request_iterator: Stream of AnomalyDetectionRequest
            context: gRPC context

        Yields:
            AnomalyDetectionResponse for each request
        """
        async for request in request_iterator:
            try:
                response = await self.DetectAnomalies(request, context)
                yield response

            except Exception as e:
                self.logger.error(f"Streaming anomaly detection failed: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Streaming error: {e}")
                break

    # Service Management Methods

    async def GetServiceStatus(self, request, context) -> Dict[str, Any]:
        """
        Get service status and statistics.

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            Service status information
        """
        try:
            uptime = time.time() - self.stats["start_time"]

            # Calculate average processing time
            avg_processing_time = 0.0
            if self.stats["successful_requests"] > 0:
                avg_processing_time = (
                    self.stats["total_processing_time_ms"]
                    / self.stats["successful_requests"]
                )

            # Get AI Engine status
            engine_status = {}
            if self.ai_engine:
                engine_status = self.ai_engine.get_model_info()

            return {
                "service_name": "CRONOS AI Engine gRPC",
                "version": "1.0.0",
                "uptime_seconds": uptime,
                "statistics": {
                    "total_requests": self.stats["total_requests"],
                    "successful_requests": self.stats["successful_requests"],
                    "failed_requests": self.stats["failed_requests"],
                    "success_rate": (
                        self.stats["successful_requests"]
                        / max(1, self.stats["total_requests"])
                    ),
                    "average_processing_time_ms": avg_processing_time,
                },
                "engine_status": engine_status,
                "timestamp": time.time(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get service status: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Status check failed: {e}")
            return {}

    async def HealthCheck(self, request, context) -> Dict[str, Any]:
        """
        Simple health check endpoint.

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            Health status
        """
        try:
            status = "healthy"

            # Check AI Engine
            if not self.ai_engine:
                status = "unhealthy"
            else:
                if not getattr(self.ai_engine, "_initialized", False):
                    status = "degraded"

            return {
                "status": status,
                "timestamp": time.time(),
                "service": "cronos-ai-grpc",
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "timestamp": time.time(),
                "service": "cronos-ai-grpc",
                "error": str(e),
            }

    # Batch Processing Methods

    async def BatchProcess(self, request, context) -> Dict[str, Any]:
        """
        Process multiple items in a single request.

        Args:
            request: BatchProcessingRequest
            context: gRPC context

        Returns:
            BatchProcessingResponse
        """
        start_time = time.time()

        try:
            if not self.ai_engine:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("AI Engine not initialized")
                return {}

            operation = getattr(request, "operation", "discovery")
            data_items = getattr(request, "data_items", [])

            results = []
            successful_count = 0
            failed_count = 0

            for i, data_item in enumerate(data_items):
                try:
                    item_start_time = time.time()

                    # Process based on operation type
                    if operation == "discovery":
                        decoded = self._decode_data(data_item, "base64")
                        result = await self.ai_engine.discover_protocol(decoded)

                    elif operation == "detection":
                        decoded = self._decode_data(data_item, "base64")
                        result = await self.ai_engine.detect_fields(decoded)

                    elif operation == "anomaly":
                        decoded = self._decode_data(data_item, "base64")
                        result = await self.ai_engine.detect_anomaly(decoded)

                    else:
                        raise ValueError(f"Unknown operation: {operation}")

                    processing_time = (time.time() - item_start_time) * 1000
                    results.append(
                        {
                            "item_index": i,
                            "success": True,
                            "result": result,
                            "processing_time_ms": processing_time,
                        }
                    )

                    successful_count += 1

                except Exception as e:
                    results.append(
                        {
                            "item_index": i,
                            "success": False,
                            "error": str(e),
                            "processing_time_ms": 0.0,
                        }
                    )

                    failed_count += 1

            total_processing_time = (time.time() - start_time) * 1000

            return {
                "batch_id": f"batch_{int(time.time())}",
                "total_items": len(data_items),
                "successful_items": successful_count,
                "failed_items": failed_count,
                "results": results,
                "total_processing_time_ms": total_processing_time,
                "average_item_time_ms": total_processing_time / max(1, len(data_items)),
            }

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Batch processing failed: {e}")
            return {}

    # Utility Methods

    def _decode_data(self, data: str, data_format: str = "base64") -> bytes:
        """Decode data based on format."""
        try:
            if data_format == "base64":
                return base64.b64decode(data)
            elif data_format == "hex":
                return bytes.fromhex(data.replace(" ", ""))
            elif data_format == "text":
                return data.encode("utf-8")
            else:
                return data.encode("latin-1")

        except Exception as e:
            raise ValueError(f"Data decoding failed: {e}")


# gRPC Server Management


class GRPCServer:
    """
    gRPC server manager for AI Engine service.

    This class manages the lifecycle of the gRPC server
    and provides utilities for server management.
    """

    def __init__(self, config: Config):
        """Initialize gRPC server manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.service: Optional[AIEngineGRPCService] = None
        self.server: Optional[grpc.aio.Server] = None

        self.port = getattr(config, "grpc_port", 50051)
        self.max_workers = getattr(config, "grpc_max_workers", 10)
        self.max_message_length = getattr(
            config, "grpc_max_message_length", 10 * 1024 * 1024
        )  # 10MB

    async def start(self) -> None:
        """Start the gRPC server."""
        try:
            self.logger.info("Starting gRPC server...")

            # Create service
            self.service = AIEngineGRPCService(self.config)
            await self.service.initialize()

            # Create server with options
            options = [
                ("grpc.max_send_message_length", self.max_message_length),
                ("grpc.max_receive_message_length", self.max_message_length),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_time_between_pings_ms", 10000),
                ("grpc.http2.min_ping_interval_without_data_ms", 300000),
            ]

            self.server = grpc.aio.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=options,
            )

            # Add service to server (would use generated code in real implementation)
            # add_AIEngineServicer_to_server(self.service, self.server)

            # Add ports
            listen_addr = f"[::]:{self.port}"
            self.server.add_insecure_port(listen_addr)

            # Enable reflection for debugging
            # reflection.enable_server_reflection(SERVICE_NAMES, self.server)

            # Start server
            await self.server.start()
            self.logger.info(f"gRPC server started on {listen_addr}")

        except Exception as e:
            self.logger.error(f"Failed to start gRPC server: {e}")
            raise

    async def stop(self, grace_period: int = 5) -> None:
        """Stop the gRPC server."""
        if self.server:
            self.logger.info("Stopping gRPC server...")
            await self.server.stop(grace=grace_period)
            self.logger.info("gRPC server stopped")

    async def wait_for_termination(self) -> None:
        """Wait for server termination."""
        if self.server:
            await self.server.wait_for_termination()


# Client utilities


class AIEngineGRPCClient:
    """
    gRPC client for AI Engine service.

    This class provides a convenient client interface
    for communicating with the AI Engine gRPC service.
    """

    def __init__(self, server_address: str = "localhost:50051"):
        """Initialize gRPC client."""
        self.server_address = server_address
        self.logger = logging.getLogger(__name__)

        self.channel: Optional[grpc.aio.Channel] = None
        self.stub = None  # Would be generated stub

    async def connect(self) -> None:
        """Connect to gRPC server."""
        try:
            # Create channel
            self.channel = grpc.aio.insecure_channel(self.server_address)

            # Create stub (would use generated code)
            # self.stub = AIEngineStub(self.channel)

            # Test connection
            await self.health_check()

            self.logger.info(f"Connected to gRPC server at {self.server_address}")

        except Exception as e:
            self.logger.error(f"Failed to connect to gRPC server: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from gRPC server."""
        if self.channel:
            await self.channel.close()
            self.logger.info("Disconnected from gRPC server")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        # Mock implementation
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "cronos-ai-grpc",
        }

    async def discover_protocol(self, data: bytes, **kwargs) -> Dict[str, Any]:
        """Call protocol discovery service."""
        # Mock implementation - would use actual gRPC call
        return {
            "discovered_protocol": "http",
            "confidence_score": 0.85,
            "processing_time_ms": 150.0,
        }

    async def detect_fields(self, data: bytes, **kwargs) -> Dict[str, Any]:
        """Call field detection service."""
        # Mock implementation - would use actual gRPC call
        return {"detected_fields": [], "total_fields": 0, "processing_time_ms": 120.0}

    async def detect_anomalies(self, data: bytes, **kwargs) -> Dict[str, Any]:
        """Call anomaly detection service."""
        # Mock implementation - would use actual gRPC call
        return {
            "is_anomalous": False,
            "anomaly_score": {"overall_score": 0.2},
            "processing_time_ms": 180.0,
        }


# Main server runner


async def run_grpc_server(config: Config) -> None:
    """
    Run the gRPC server.

    Args:
        config: Configuration object
    """
    server = GRPCServer(config)

    try:
        await server.start()
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop()
    except Exception as e:
        logging.getLogger(__name__).error(f"gRPC server error: {e}")
        await server.stop()
        raise


if __name__ == "__main__":
    # Example usage
    import asyncio
    from ..core.config import Config

    config = Config()
    asyncio.run(run_grpc_server(config))
