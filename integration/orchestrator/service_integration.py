#!/usr/bin/env python3
"""
QBITEL - Service Integration Orchestrator
Coordinates communication between AI/ML engine, protocol processing, and security components.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
import aiokafka
from concurrent.futures import ThreadPoolExecutor
import grpc
import grpc.aio
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from integration.config.unified_config import get_config, get_service_config
from ai_engine.api.grpc import ProtocolDiscoveryServicer, AIInferenceServicer
from rust.dataplane.target.debug import libdpi_engine  # FFI bindings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component status enumeration"""

    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class Message:
    """Internal message format for component communication"""

    id: str
    timestamp: float
    source: str
    destination: str
    message_type: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    priority: int = 0


@dataclass
class ComponentHealth:
    """Component health information"""

    component: str
    status: ComponentStatus
    last_heartbeat: float
    error_count: int = 0
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = None


class ServiceIntegrationOrchestrator:
    """
    Main orchestrator for integrating all QBITEL components.
    Handles message routing, health monitoring, and component coordination.
    """

    def __init__(self):
        self.config = get_config()
        self.components: Dict[str, ComponentHealth] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False

        # Initialize connections
        self.redis_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
        self.grpc_server = None

        # Thread pools for different types of work
        self.io_executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix="io_")
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="cpu_"
        )

        # Message queues for internal routing
        self.message_queue = asyncio.Queue(maxsize=10000)
        self.health_queue = asyncio.Queue(maxsize=1000)

        # Performance metrics
        self.message_count = 0
        self.error_count = 0
        self.start_time = time.time()

    async def initialize(self):
        """Initialize all connections and components"""
        logger.info("Initializing Service Integration Orchestrator...")

        try:
            # Initialize Redis connection
            await self._init_redis()

            # Initialize Kafka connections
            await self._init_kafka()

            # Initialize gRPC server
            await self._init_grpc_server()

            # Register message handlers
            self._register_message_handlers()

            # Initialize component health monitoring
            await self._init_health_monitoring()

            logger.info("Service Integration Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise

    async def _init_redis(self):
        """Initialize Redis connection"""
        if self.config.redis:
            try:
                redis_config = {
                    "host": self.config.redis.host,
                    "port": self.config.redis.port,
                    "db": self.config.redis.database,
                    "password": (
                        self.config.redis.password
                        if self.config.redis.password
                        else None
                    ),
                    "ssl": self.config.redis.ssl,
                    "retry_on_timeout": True,
                    "socket_connect_timeout": self.config.redis.connection_timeout,
                }

                self.redis_client = await aioredis.from_url(
                    f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}",
                    password=redis_config["password"],
                    ssl=redis_config["ssl"],
                    retry_on_timeout=redis_config["retry_on_timeout"],
                )

                # Test connection
                await self.redis_client.ping()
                logger.info("Redis connection established")

            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis_client = None

    async def _init_kafka(self):
        """Initialize Kafka producer and consumer"""
        if self.config.kafka:
            try:
                kafka_config = {
                    "bootstrap_servers": self.config.kafka.bootstrap_servers,
                    "security_protocol": self.config.kafka.security_protocol,
                    "sasl_mechanism": self.config.kafka.sasl_mechanism,
                    "sasl_plain_username": self.config.kafka.sasl_username,
                    "sasl_plain_password": self.config.kafka.sasl_password,
                    "ssl_cafile": self.config.kafka.ssl_ca_location,
                }

                # Initialize producer
                self.kafka_producer = aiokafka.AIOKafkaProducer(
                    **kafka_config,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    compression_type="snappy",
                    max_batch_size=32768,
                    linger_ms=10,
                )
                await self.kafka_producer.start()

                # Initialize consumer
                self.kafka_consumer = aiokafka.AIOKafkaConsumer(
                    "qbitel-internal",
                    **kafka_config,
                    group_id=self.config.kafka.group_id,
                    auto_offset_reset=self.config.kafka.auto_offset_reset,
                    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                )
                await self.kafka_consumer.start()

                logger.info("Kafka connections established")

            except Exception as e:
                logger.error(f"Failed to connect to Kafka: {e}")
                self.kafka_producer = None
                self.kafka_consumer = None

    async def _init_grpc_server(self):
        """Initialize gRPC server for internal component communication"""
        try:
            self.grpc_server = grpc.aio.server(
                ThreadPoolExecutor(max_workers=50),
                options=[
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.http2.max_pings_without_data", 0),
                    ("grpc.http2.min_time_between_pings_ms", 10000),
                    ("grpc.http2.min_ping_interval_without_data_ms", 300000),
                ],
            )

            # Add servicers
            protocol_servicer = ProtocolDiscoveryServicer(self)
            ai_servicer = AIInferenceServicer(self)

            # Register services (assuming protobuf definitions exist)
            # add_ProtocolDiscoveryServiceServicer_to_server(protocol_servicer, self.grpc_server)
            # add_AIInferenceServiceServicer_to_server(ai_servicer, self.grpc_server)

            listen_addr = f"{self.config.api_host}:9090"
            self.grpc_server.add_insecure_port(listen_addr)

            await self.grpc_server.start()
            logger.info(f"gRPC server started on {listen_addr}")

        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")
            self.grpc_server = None

    def _register_message_handlers(self):
        """Register message handlers for different message types"""
        self.message_handlers = {
            "protocol_discovery": self._handle_protocol_discovery,
            "ai_inference": self._handle_ai_inference,
            "security_analysis": self._handle_security_analysis,
            "packet_processing": self._handle_packet_processing,
            "health_check": self._handle_health_check,
            "config_update": self._handle_config_update,
            "metric_update": self._handle_metric_update,
        }

    async def _init_health_monitoring(self):
        """Initialize component health monitoring"""
        self.components = {
            "ai_engine": ComponentHealth(
                "ai_engine", ComponentStatus.STARTING, time.time()
            ),
            "protocol_processor": ComponentHealth(
                "protocol_processor", ComponentStatus.STARTING, time.time()
            ),
            "security_analyzer": ComponentHealth(
                "security_analyzer", ComponentStatus.STARTING, time.time()
            ),
            "dpi_engine": ComponentHealth(
                "dpi_engine", ComponentStatus.STARTING, time.time()
            ),
            "quantum_crypto": ComponentHealth(
                "quantum_crypto", ComponentStatus.STARTING, time.time()
            ),
        }

    async def start(self):
        """Start the orchestrator and all background tasks"""
        self.running = True
        logger.info("Starting Service Integration Orchestrator...")

        # Start background tasks
        tasks = [
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._kafka_message_consumer()),
            asyncio.create_task(self._metrics_collector()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Orchestrator tasks cancelled")
        except Exception as e:
            logger.error(f"Error in orchestrator tasks: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown orchestrator and cleanup resources"""
        logger.info("Shutting down Service Integration Orchestrator...")
        self.running = False

        # Close connections
        if self.kafka_producer:
            await self.kafka_producer.stop()
        if self.kafka_consumer:
            await self.kafka_consumer.stop()
        if self.redis_client:
            await self.redis_client.close()
        if self.grpc_server:
            await self.grpc_server.stop(grace=5)

        # Shutdown thread pools
        self.io_executor.shutdown(wait=True)
        self.cpu_executor.shutdown(wait=True)

        logger.info("Service Integration Orchestrator shutdown complete")

    async def _message_processor(self):
        """Process internal messages"""
        while self.running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)

                # Route message to appropriate handler
                handler = self.message_handlers.get(message.message_type)
                if handler:
                    await handler(message)
                else:
                    logger.warning(
                        f"No handler for message type: {message.message_type}"
                    )

                self.message_count += 1

            except asyncio.TimeoutError:
                continue  # Normal timeout, continue processing
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.error_count += 1

    async def _health_monitor(self):
        """Monitor component health"""
        while self.running:
            try:
                current_time = time.time()

                # Check component health
                for component_name, health in self.components.items():
                    time_since_heartbeat = current_time - health.last_heartbeat

                    if time_since_heartbeat > 60:  # 60 seconds timeout
                        if health.status != ComponentStatus.OFFLINE:
                            health.status = ComponentStatus.OFFLINE
                            logger.warning(f"Component {component_name} went offline")
                    elif time_since_heartbeat > 30:  # 30 seconds warning
                        if health.status == ComponentStatus.HEALTHY:
                            health.status = ComponentStatus.DEGRADED
                            logger.warning(f"Component {component_name} is degraded")

                # Send health update to monitoring system
                await self._publish_health_metrics()

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(10)

    async def _kafka_message_consumer(self):
        """Consume messages from Kafka"""
        if not self.kafka_consumer:
            return

        while self.running:
            try:
                async for kafka_message in self.kafka_consumer:
                    # Convert Kafka message to internal format
                    internal_message = Message(
                        id=f"kafka_{time.time()}",
                        timestamp=time.time(),
                        source="kafka",
                        destination="orchestrator",
                        message_type=kafka_message.value.get("type", "unknown"),
                        payload=kafka_message.value.get("payload", {}),
                        correlation_id=kafka_message.value.get("correlation_id"),
                    )

                    await self.message_queue.put(internal_message)

            except Exception as e:
                logger.error(f"Error consuming Kafka messages: {e}")
                await asyncio.sleep(5)

    async def _metrics_collector(self):
        """Collect and publish metrics"""
        while self.running:
            try:
                metrics = {
                    "timestamp": time.time(),
                    "uptime": time.time() - self.start_time,
                    "message_count": self.message_count,
                    "error_count": self.error_count,
                    "queue_size": self.message_queue.qsize(),
                    "component_count": len(
                        [
                            c
                            for c in self.components.values()
                            if c.status
                            in [ComponentStatus.HEALTHY, ComponentStatus.DEGRADED]
                        ]
                    ),
                    "memory_usage": self._get_memory_usage(),
                }

                # Publish to Redis if available
                if self.redis_client:
                    await self.redis_client.setex(
                        "qbitel:orchestrator:metrics",
                        60,  # 60 second TTL
                        json.dumps(metrics),
                    )

                # Publish to Kafka if available
                if self.kafka_producer:
                    await self.kafka_producer.send(
                        "qbitel-metrics",
                        {"type": "orchestrator_metrics", "data": metrics},
                    )

                await asyncio.sleep(30)  # Collect every 30 seconds

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(30)

    # Message Handlers
    async def _handle_protocol_discovery(self, message: Message):
        """Handle protocol discovery requests"""
        try:
            payload = message.payload
            protocol_data = payload.get("protocol_data")

            if not protocol_data:
                logger.warning("Protocol discovery message missing protocol_data")
                return

            # Forward to AI engine for analysis
            ai_message = Message(
                id=f"ai_{time.time()}",
                timestamp=time.time(),
                source="orchestrator",
                destination="ai_engine",
                message_type="analyze_protocol",
                payload={
                    "protocol_data": protocol_data,
                    "analysis_type": "discovery",
                    "confidence_threshold": 0.8,
                },
                correlation_id=message.correlation_id,
            )

            # Send via gRPC to AI engine (if available) or queue for processing
            await self._route_message(ai_message)

        except Exception as e:
            logger.error(f"Error handling protocol discovery: {e}")

    async def _handle_ai_inference(self, message: Message):
        """Handle AI inference requests"""
        try:
            payload = message.payload
            input_data = payload.get("input_data")
            model_type = payload.get("model_type", "classification")

            # Route to appropriate AI model
            if model_type == "classification":
                result = await self._run_protocol_classification(input_data)
            elif model_type == "anomaly_detection":
                result = await self._run_anomaly_detection(input_data)
            elif model_type == "field_detection":
                result = await self._run_field_detection(input_data)
            else:
                logger.warning(f"Unknown AI model type: {model_type}")
                return

            # Send result back to requester
            response_message = Message(
                id=f"ai_response_{time.time()}",
                timestamp=time.time(),
                source="orchestrator",
                destination=message.source,
                message_type="ai_inference_result",
                payload={
                    "result": result,
                    "model_type": model_type,
                },
                correlation_id=message.correlation_id,
            )

            await self._route_message(response_message)

        except Exception as e:
            logger.error(f"Error handling AI inference: {e}")

    async def _handle_security_analysis(self, message: Message):
        """Handle security analysis requests"""
        try:
            payload = message.payload
            packet_data = payload.get("packet_data")
            analysis_type = payload.get("analysis_type", "threat_detection")

            # Forward to security analyzer
            security_message = Message(
                id=f"security_{time.time()}",
                timestamp=time.time(),
                source="orchestrator",
                destination="security_analyzer",
                message_type="analyze_security",
                payload={
                    "packet_data": packet_data,
                    "analysis_type": analysis_type,
                },
                correlation_id=message.correlation_id,
            )

            await self._route_message(security_message)

        except Exception as e:
            logger.error(f"Error handling security analysis: {e}")

    async def _handle_packet_processing(self, message: Message):
        """Handle packet processing requests"""
        try:
            payload = message.payload
            packet_data = payload.get("packet_data")

            # Send to DPI engine for initial processing
            dpi_result = await self._run_dpi_analysis(packet_data)

            # If protocol is unknown, trigger discovery
            if dpi_result.get("protocol") == "unknown":
                discovery_message = Message(
                    id=f"discovery_{time.time()}",
                    timestamp=time.time(),
                    source="orchestrator",
                    destination="ai_engine",
                    message_type="protocol_discovery",
                    payload={"protocol_data": packet_data},
                    correlation_id=message.correlation_id,
                )
                await self.message_queue.put(discovery_message)

            # Always run security analysis
            security_message = Message(
                id=f"security_{time.time()}",
                timestamp=time.time(),
                source="orchestrator",
                destination="security_analyzer",
                message_type="security_analysis",
                payload={"packet_data": packet_data, "dpi_result": dpi_result},
                correlation_id=message.correlation_id,
            )
            await self.message_queue.put(security_message)

        except Exception as e:
            logger.error(f"Error handling packet processing: {e}")

    async def _handle_health_check(self, message: Message):
        """Handle health check messages"""
        component = message.payload.get("component")
        status = message.payload.get("status")
        metrics = message.payload.get("metrics", {})

        if component in self.components:
            self.components[component].last_heartbeat = time.time()
            if status:
                self.components[component].status = ComponentStatus(status)
            self.components[component].metrics = metrics

    async def _handle_config_update(self, message: Message):
        """Handle configuration updates"""
        component = message.payload.get("component")
        config_key = message.payload.get("config_key")
        config_value = message.payload.get("config_value")

        logger.info(
            f"Configuration update for {component}: {config_key} = {config_value}"
        )

        # Broadcast to relevant components
        if component == "all":
            for comp_name in self.components.keys():
                await self._send_config_update(comp_name, config_key, config_value)
        else:
            await self._send_config_update(component, config_key, config_value)

    async def _handle_metric_update(self, message: Message):
        """Handle metric updates"""
        metrics = message.payload.get("metrics", {})
        component = message.payload.get("component")

        if component in self.components:
            self.components[component].metrics = metrics

    # Helper Methods
    async def _route_message(self, message: Message):
        """Route message to appropriate destination"""
        if message.destination == "orchestrator":
            await self.message_queue.put(message)
        else:
            # Send via appropriate transport (gRPC, Kafka, etc.)
            if self.kafka_producer:
                kafka_message = {
                    "type": message.message_type,
                    "payload": message.payload,
                    "correlation_id": message.correlation_id,
                    "timestamp": message.timestamp,
                    "source": message.source,
                    "destination": message.destination,
                }
                await self.kafka_producer.send(
                    f"qbitel-{message.destination}", kafka_message
                )

    async def _run_protocol_classification(self, input_data):
        """Run protocol classification via AI engine"""
        # This would interface with the actual AI engine
        # For now, return mock result
        return {
            "protocol": "HTTP",
            "confidence": 0.95,
            "version": "1.1",
            "features": input_data.get("features", {}),
        }

    async def _run_anomaly_detection(self, input_data):
        """Run anomaly detection"""
        return {"is_anomaly": False, "anomaly_score": 0.1, "confidence": 0.88}

    async def _run_field_detection(self, input_data):
        """Run field boundary detection"""
        return {
            "fields": [
                {"start": 0, "end": 8, "type": "header"},
                {"start": 8, "end": 24, "type": "payload"},
            ],
            "confidence": 0.92,
        }

    async def _run_dpi_analysis(self, packet_data):
        """Run DPI analysis using Rust DPI engine"""
        # Interface with Rust DPI engine via FFI
        # This would call the actual Rust implementation
        return {
            "protocol": "TCP",
            "application": "HTTP",
            "classification_confidence": 0.9,
            "metadata": {},
        }

    async def _send_config_update(self, component: str, key: str, value: Any):
        """Send configuration update to component"""
        config_message = Message(
            id=f"config_{time.time()}",
            timestamp=time.time(),
            source="orchestrator",
            destination=component,
            message_type="config_update",
            payload={
                "config_key": key,
                "config_value": value,
            },
        )
        await self._route_message(config_message)

    async def _publish_health_metrics(self):
        """Publish health metrics to monitoring system"""
        health_summary = {
            "timestamp": time.time(),
            "components": {
                name: asdict(health) for name, health in self.components.items()
            },
            "overall_status": self._calculate_overall_health(),
        }

        if self.redis_client:
            await self.redis_client.setex(
                "qbitel:orchestrator:health",
                60,  # 60 second TTL
                json.dumps(health_summary, default=str),
            )

    def _calculate_overall_health(self) -> str:
        """Calculate overall system health"""
        healthy_count = len(
            [c for c in self.components.values() if c.status == ComponentStatus.HEALTHY]
        )
        total_count = len(self.components)

        if healthy_count == total_count:
            return "healthy"
        elif healthy_count >= total_count * 0.7:
            return "degraded"
        else:
            return "unhealthy"

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }

    # Public API
    async def send_message(self, message: Message):
        """Send message through orchestrator"""
        await self.message_queue.put(message)

    def get_component_health(self, component: str) -> Optional[ComponentHealth]:
        """Get health information for specific component"""
        return self.components.get(component)

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        return {
            "overall_status": self._calculate_overall_health(),
            "components": {
                name: asdict(health) for name, health in self.components.items()
            },
            "uptime": time.time() - self.start_time,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.message_count, 1),
        }


# Global orchestrator instance
_orchestrator = None


def get_orchestrator() -> ServiceIntegrationOrchestrator:
    """Get global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ServiceIntegrationOrchestrator()
    return _orchestrator


async def main():
    """Main entry point for orchestrator"""
    orchestrator = ServiceIntegrationOrchestrator()

    try:
        await orchestrator.initialize()
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
