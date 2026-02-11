#!/usr/bin/env python3
"""
QBITEL - Kafka Streaming Infrastructure
Production-ready Kafka streaming service for real-time data processing.
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
import avro.schema
import avro.io
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from integration.config.unified_config import get_config, get_service_config
from integration.orchestrator.service_integration import get_orchestrator, Message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Stream event types"""

    PACKET_CAPTURED = "packet_captured"
    PROTOCOL_DISCOVERED = "protocol_discovered"
    AI_ANALYSIS = "ai_analysis"
    SECURITY_EVENT = "security_event"
    THREAT_DETECTED = "threat_detected"
    PERFORMANCE_METRIC = "performance_metric"
    AUDIT_LOG = "audit_log"
    SYSTEM_ALERT = "system_alert"


@dataclass
class StreamEvent:
    """Stream event structure"""

    event_id: str
    event_type: StreamEventType
    timestamp: float
    source_component: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    partition_key: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class StreamTopology:
    """Stream processing topology definition"""

    name: str
    input_topics: List[str]
    output_topics: List[str]
    processing_function: Callable
    parallelism: int = 1
    error_topic: Optional[str] = None


@dataclass
class StreamMetrics:
    """Stream processing metrics"""

    messages_produced: int = 0
    messages_consumed: int = 0
    processing_errors: int = 0
    average_latency_ms: float = 0.0
    throughput_per_second: float = 0.0
    last_processed_time: float = 0.0


class KafkaStreamingService:
    """
    Production Kafka streaming service for QBITEL.
    Handles real-time data streaming between components.
    """

    def __init__(self):
        self.config = get_service_config("kafka")
        self.orchestrator = get_orchestrator()

        # Kafka clients
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}

        # Stream processing
        self.stream_processors: Dict[str, StreamTopology] = {}
        self.running = False

        # Performance tracking
        self.metrics = StreamMetrics()
        self.start_time = time.time()

        # Topic configuration
        self.topics = {
            "packet-processing": {
                "partitions": 16,
                "replication_factor": 3,
                "config": {
                    "retention.ms": 3600000,  # 1 hour
                    "compression.type": "snappy",
                },
            },
            "ai-analysis": {
                "partitions": 8,
                "replication_factor": 3,
                "config": {
                    "retention.ms": 86400000,  # 24 hours
                    "compression.type": "lz4",
                },
            },
            "security-events": {
                "partitions": 4,
                "replication_factor": 3,
                "config": {
                    "retention.ms": 604800000,  # 7 days
                    "compression.type": "gzip",
                },
            },
            "threat-intel": {
                "partitions": 2,
                "replication_factor": 3,
                "config": {
                    "retention.ms": 2592000000,  # 30 days
                    "compression.type": "gzip",
                },
            },
            "metrics": {
                "partitions": 8,
                "replication_factor": 3,
                "config": {
                    "retention.ms": 86400000,  # 24 hours
                    "compression.type": "snappy",
                },
            },
            "audit-logs": {
                "partitions": 4,
                "replication_factor": 3,
                "config": {
                    "retention.ms": 31536000000,  # 1 year
                    "compression.type": "gzip",
                },
            },
            "deadletter": {
                "partitions": 4,
                "replication_factor": 3,
                "config": {
                    "retention.ms": 604800000,  # 7 days
                    "compression.type": "gzip",
                },
            },
        }

        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}

        # Thread pool for processing
        self.processing_executor = ThreadPoolExecutor(
            max_workers=20, thread_name_prefix="stream_"
        )

        # Avro schemas for serialization
        self.avro_schemas = {}

    async def initialize(self):
        """Initialize Kafka streaming service"""
        logger.info("Initializing Kafka Streaming Service...")

        try:
            # Initialize Kafka producer
            await self._init_producer()

            # Create topics
            await self._create_topics()

            # Initialize Avro schemas
            await self._init_avro_schemas()

            # Register message handlers
            self._register_message_handlers()

            # Initialize stream processors
            await self._init_stream_processors()

            logger.info("Kafka Streaming Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Kafka streaming service: {e}")
            raise

    async def _init_producer(self):
        """Initialize Kafka producer"""
        try:
            kafka_config = {
                "bootstrap_servers": self.config.get(
                    "bootstrap_servers", ["localhost:9092"]
                ),
                "security_protocol": self.config.get("security_protocol", "PLAINTEXT"),
                "compression_type": "snappy",
                "max_batch_size": 32768,
                "linger_ms": 10,
                "acks": "all",
                "retries": 3,
                "max_in_flight_requests_per_connection": 1,
                "enable_idempotence": True,
            }

            # Add authentication if configured
            if self.config.get("sasl_mechanism"):
                kafka_config.update(
                    {
                        "sasl_mechanism": self.config.get("sasl_mechanism"),
                        "sasl_plain_username": self.config.get("sasl_username"),
                        "sasl_plain_password": self.config.get("sasl_password"),
                    }
                )

            # Add SSL if configured
            if self.config.get("ssl_ca_location"):
                kafka_config.update(
                    {
                        "ssl_cafile": self.config.get("ssl_ca_location"),
                    }
                )

            self.producer = AIOKafkaProducer(
                **kafka_config,
                value_serializer=self._serialize_message,
                key_serializer=lambda k: k.encode("utf-8") if k else None,
            )

            await self.producer.start()
            logger.info("Kafka producer initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise

    async def _create_topics(self):
        """Create Kafka topics if they don't exist"""
        try:
            from kafka.admin import KafkaAdminClient, NewTopic
            from kafka.errors import TopicAlreadyExistsError

            # Create admin client
            admin_config = {
                "bootstrap_servers": self.config.get(
                    "bootstrap_servers", ["localhost:9092"]
                ),
            }

            if self.config.get("security_protocol") != "PLAINTEXT":
                admin_config.update(
                    {
                        "security_protocol": self.config.get("security_protocol"),
                        "sasl_mechanism": self.config.get("sasl_mechanism"),
                        "sasl_plain_username": self.config.get("sasl_username"),
                        "sasl_plain_password": self.config.get("sasl_password"),
                    }
                )

            admin_client = KafkaAdminClient(**admin_config)

            # Create topics
            topics_to_create = []
            for topic_name, topic_config in self.topics.items():
                new_topic = NewTopic(
                    name=f"qbitel-{topic_name}",
                    num_partitions=topic_config["partitions"],
                    replication_factor=topic_config["replication_factor"],
                    topic_configs=topic_config["config"],
                )
                topics_to_create.append(new_topic)

            try:
                result = admin_client.create_topics(topics_to_create)
                for topic, future in result.values():
                    try:
                        future.result()  # Block until topic is created
                        logger.info(f"Created topic: {topic}")
                    except TopicAlreadyExistsError:
                        logger.info(f"Topic already exists: {topic}")

            except Exception as e:
                logger.warning(f"Error creating topics (may already exist): {e}")

            admin_client.close()
            logger.info("Kafka topics verified/created")

        except Exception as e:
            logger.error(f"Failed to create Kafka topics: {e}")

    async def _init_avro_schemas(self):
        """Initialize Avro schemas for message serialization"""
        try:
            # Define Avro schema for stream events
            stream_event_schema = {
                "type": "record",
                "name": "StreamEvent",
                "namespace": "qbitel.ai.streaming",
                "fields": [
                    {"name": "event_id", "type": "string"},
                    {"name": "event_type", "type": "string"},
                    {"name": "timestamp", "type": "double"},
                    {"name": "source_component", "type": "string"},
                    {"name": "data", "type": {"type": "map", "values": "string"}},
                    {
                        "name": "correlation_id",
                        "type": ["null", "string"],
                        "default": None,
                    },
                    {
                        "name": "partition_key",
                        "type": ["null", "string"],
                        "default": None,
                    },
                    {
                        "name": "metadata",
                        "type": ["null", {"type": "map", "values": "string"}],
                        "default": None,
                    },
                ],
            }

            self.avro_schemas["stream_event"] = avro.schema.parse(
                json.dumps(stream_event_schema)
            )
            logger.info("Avro schemas initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Avro schemas: {e}")

    def _register_message_handlers(self):
        """Register message handlers for different event types"""
        self.message_handlers = {
            StreamEventType.PACKET_CAPTURED: self._handle_packet_captured,
            StreamEventType.PROTOCOL_DISCOVERED: self._handle_protocol_discovered,
            StreamEventType.AI_ANALYSIS: self._handle_ai_analysis,
            StreamEventType.SECURITY_EVENT: self._handle_security_event,
            StreamEventType.THREAT_DETECTED: self._handle_threat_detected,
            StreamEventType.PERFORMANCE_METRIC: self._handle_performance_metric,
            StreamEventType.AUDIT_LOG: self._handle_audit_log,
            StreamEventType.SYSTEM_ALERT: self._handle_system_alert,
        }

    async def _init_stream_processors(self):
        """Initialize stream processing topologies"""
        try:
            # Packet processing topology
            self.stream_processors["packet-processing"] = StreamTopology(
                name="packet-processing",
                input_topics=["qbitel-packet-processing"],
                output_topics=["qbitel-ai-analysis"],
                processing_function=self._process_packet_stream,
                parallelism=4,
                error_topic="qbitel-deadletter",
            )

            # AI analysis topology
            self.stream_processors["ai-analysis"] = StreamTopology(
                name="ai-analysis",
                input_topics=["qbitel-ai-analysis"],
                output_topics=["qbitel-security-events"],
                processing_function=self._process_ai_analysis_stream,
                parallelism=2,
                error_topic="qbitel-deadletter",
            )

            # Security events topology
            self.stream_processors["security-events"] = StreamTopology(
                name="security-events",
                input_topics=["qbitel-security-events"],
                output_topics=["qbitel-threat-intel", "qbitel-audit-logs"],
                processing_function=self._process_security_events_stream,
                parallelism=2,
                error_topic="qbitel-deadletter",
            )

            logger.info(f"Initialized {len(self.stream_processors)} stream processors")

        except Exception as e:
            logger.error(f"Failed to initialize stream processors: {e}")
            raise

    async def start(self):
        """Start the streaming service"""
        self.running = True
        logger.info("Starting Kafka Streaming Service...")

        try:
            # Start stream processors
            tasks = []
            for processor_name, topology in self.stream_processors.items():
                for i in range(topology.parallelism):
                    task = asyncio.create_task(
                        self._run_stream_processor(processor_name, topology, i)
                    )
                    tasks.append(task)

            # Start metrics collector
            tasks.append(asyncio.create_task(self._metrics_collector()))

            # Start health monitor
            tasks.append(asyncio.create_task(self._health_monitor()))

            # Wait for all tasks
            await asyncio.gather(*tasks)

        except asyncio.CancelledError:
            logger.info("Streaming service tasks cancelled")
        except Exception as e:
            logger.error(f"Error in streaming service: {e}")
        finally:
            await self.shutdown()

    async def _run_stream_processor(
        self, processor_name: str, topology: StreamTopology, instance_id: int
    ):
        """Run a stream processor instance"""
        consumer_group = f"{topology.name}-processor-{instance_id}"
        consumer = None

        try:
            # Create consumer for this processor
            consumer = await self._create_consumer(
                consumer_group, topology.input_topics
            )

            logger.info(
                f"Started stream processor {processor_name} instance {instance_id}"
            )

            # Process messages
            async for message in consumer:
                try:
                    # Deserialize message
                    event = self._deserialize_message(message.value)

                    # Process with topology function
                    result = await topology.processing_function(event, message)

                    # Send results to output topics if any
                    if result and topology.output_topics:
                        await self._send_processed_result(
                            result, topology.output_topics
                        )

                    # Update metrics
                    self.metrics.messages_consumed += 1
                    self.metrics.last_processed_time = time.time()

                except Exception as e:
                    logger.error(f"Error processing message in {processor_name}: {e}")
                    self.metrics.processing_errors += 1

                    # Send to error topic if configured
                    if topology.error_topic:
                        await self._send_to_error_topic(
                            message, topology.error_topic, str(e)
                        )

        except Exception as e:
            logger.error(f"Error in stream processor {processor_name}: {e}")
        finally:
            if consumer:
                await consumer.stop()

    async def _create_consumer(
        self, consumer_group: str, topics: List[str]
    ) -> AIOKafkaConsumer:
        """Create Kafka consumer"""
        kafka_config = {
            "bootstrap_servers": self.config.get(
                "bootstrap_servers", ["localhost:9092"]
            ),
            "group_id": consumer_group,
            "auto_offset_reset": self.config.get("auto_offset_reset", "latest"),
            "enable_auto_commit": True,
            "auto_commit_interval_ms": 1000,
            "session_timeout_ms": 30000,
            "heartbeat_interval_ms": 10000,
            "max_poll_records": 500,
        }

        # Add authentication if configured
        if self.config.get("security_protocol") != "PLAINTEXT":
            kafka_config.update(
                {
                    "security_protocol": self.config.get("security_protocol"),
                    "sasl_mechanism": self.config.get("sasl_mechanism"),
                    "sasl_plain_username": self.config.get("sasl_username"),
                    "sasl_plain_password": self.config.get("sasl_password"),
                }
            )

        consumer = AIOKafkaConsumer(
            *topics,
            **kafka_config,
            value_deserializer=lambda m: m,  # We'll handle deserialization manually
        )

        await consumer.start()
        return consumer

    # Stream processing functions
    async def _process_packet_stream(
        self, event: StreamEvent, message
    ) -> Optional[StreamEvent]:
        """Process packet capture events"""
        try:
            if event.event_type != StreamEventType.PACKET_CAPTURED:
                return None

            packet_data = event.data.get("packet_data")
            if not packet_data:
                return None

            # Create AI analysis request
            analysis_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=StreamEventType.AI_ANALYSIS,
                timestamp=time.time(),
                source_component="packet-processor",
                data={
                    "packet_data": packet_data,
                    "analysis_type": "protocol_discovery",
                    "priority": event.data.get("priority", "normal"),
                },
                correlation_id=event.correlation_id,
                partition_key=event.partition_key,
            )

            return analysis_event

        except Exception as e:
            logger.error(f"Error processing packet stream: {e}")
            return None

    async def _process_ai_analysis_stream(
        self, event: StreamEvent, message
    ) -> Optional[StreamEvent]:
        """Process AI analysis results"""
        try:
            if event.event_type != StreamEventType.AI_ANALYSIS:
                return None

            analysis_result = event.data.get("analysis_result")
            if not analysis_result:
                return None

            # Check if security analysis is needed
            threat_level = analysis_result.get("threat_level", "low")
            if threat_level in ["medium", "high", "critical"]:
                security_event = StreamEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=StreamEventType.SECURITY_EVENT,
                    timestamp=time.time(),
                    source_component="ai-analyzer",
                    data={
                        "analysis_result": analysis_result,
                        "threat_level": threat_level,
                        "requires_action": threat_level in ["high", "critical"],
                    },
                    correlation_id=event.correlation_id,
                    partition_key=event.partition_key,
                )

                return security_event

            return None

        except Exception as e:
            logger.error(f"Error processing AI analysis stream: {e}")
            return None

    async def _process_security_events_stream(
        self, event: StreamEvent, message
    ) -> List[StreamEvent]:
        """Process security events"""
        try:
            if event.event_type != StreamEventType.SECURITY_EVENT:
                return []

            results = []

            # Create audit log entry
            audit_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=StreamEventType.AUDIT_LOG,
                timestamp=time.time(),
                source_component="security-processor",
                data={
                    "event_type": "security_event_processed",
                    "original_event": asdict(event),
                    "severity": event.data.get("threat_level", "low"),
                },
                correlation_id=event.correlation_id,
            )
            results.append(audit_event)

            # Create threat intelligence update if high severity
            threat_level = event.data.get("threat_level", "low")
            if threat_level in ["high", "critical"]:
                threat_intel_event = StreamEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=StreamEventType.THREAT_DETECTED,
                    timestamp=time.time(),
                    source_component="security-processor",
                    data={
                        "threat_indicators": event.data.get("analysis_result", {}).get(
                            "indicators", []
                        ),
                        "threat_level": threat_level,
                        "confidence": event.data.get("analysis_result", {}).get(
                            "confidence", 0.0
                        ),
                    },
                    correlation_id=event.correlation_id,
                )
                results.append(threat_intel_event)

            return results

        except Exception as e:
            logger.error(f"Error processing security events stream: {e}")
            return []

    # Message handlers
    async def _handle_packet_captured(self, event: StreamEvent):
        """Handle packet captured events"""
        await self.send_event("qbitel-packet-processing", event)

    async def _handle_protocol_discovered(self, event: StreamEvent):
        """Handle protocol discovery events"""
        await self.send_event("qbitel-ai-analysis", event)

    async def _handle_ai_analysis(self, event: StreamEvent):
        """Handle AI analysis events"""
        await self.send_event("qbitel-ai-analysis", event)

    async def _handle_security_event(self, event: StreamEvent):
        """Handle security events"""
        await self.send_event("qbitel-security-events", event)

    async def _handle_threat_detected(self, event: StreamEvent):
        """Handle threat detection events"""
        await self.send_event("qbitel-threat-intel", event)

    async def _handle_performance_metric(self, event: StreamEvent):
        """Handle performance metric events"""
        await self.send_event("qbitel-metrics", event)

    async def _handle_audit_log(self, event: StreamEvent):
        """Handle audit log events"""
        await self.send_event("qbitel-audit-logs", event)

    async def _handle_system_alert(self, event: StreamEvent):
        """Handle system alert events"""
        # Send to multiple topics for high-priority alerts
        await self.send_event("qbitel-security-events", event)
        await self.send_event("qbitel-audit-logs", event)

    # Utility methods
    def _serialize_message(self, message: Any) -> bytes:
        """Serialize message for Kafka"""
        try:
            if isinstance(message, StreamEvent):
                # Use Avro serialization for structured events
                return self._serialize_with_avro(message, "stream_event")
            else:
                # Use JSON for other messages
                return json.dumps(message, default=str).encode("utf-8")
        except Exception as e:
            logger.error(f"Error serializing message: {e}")
            return json.dumps({"error": "serialization_failed"}).encode("utf-8")

    def _deserialize_message(self, message_bytes: bytes) -> Any:
        """Deserialize message from Kafka"""
        try:
            # Try Avro first
            try:
                return self._deserialize_with_avro(message_bytes, "stream_event")
            except:
                # Fallback to JSON
                return json.loads(message_bytes.decode("utf-8"))
        except Exception as e:
            logger.error(f"Error deserializing message: {e}")
            return {"error": "deserialization_failed"}

    def _serialize_with_avro(self, obj: StreamEvent, schema_name: str) -> bytes:
        """Serialize object with Avro"""
        try:
            schema = self.avro_schemas[schema_name]

            # Convert dataclass to dict
            obj_dict = asdict(obj)
            obj_dict["event_type"] = obj.event_type.value  # Convert enum to string

            # Convert data dict values to strings for Avro map type
            if obj_dict["data"]:
                obj_dict["data"] = {
                    k: json.dumps(v) if not isinstance(v, str) else v
                    for k, v in obj_dict["data"].items()
                }

            # Serialize
            bytes_writer = io.BytesIO()
            encoder = avro.io.BinaryEncoder(bytes_writer)
            datum_writer = avro.io.DatumWriter(schema)
            datum_writer.write(obj_dict, encoder)

            return bytes_writer.getvalue()

        except Exception as e:
            logger.error(f"Error serializing with Avro: {e}")
            return json.dumps(asdict(obj), default=str).encode("utf-8")

    def _deserialize_with_avro(
        self, message_bytes: bytes, schema_name: str
    ) -> StreamEvent:
        """Deserialize object with Avro"""
        schema = self.avro_schemas[schema_name]

        bytes_reader = io.BytesIO(message_bytes)
        decoder = avro.io.BinaryDecoder(bytes_reader)
        datum_reader = avro.io.DatumReader(schema)
        obj_dict = datum_reader.read(decoder)

        # Convert back to StreamEvent
        obj_dict["event_type"] = StreamEventType(obj_dict["event_type"])

        # Convert data string values back to objects
        if obj_dict["data"]:
            data = {}
            for k, v in obj_dict["data"].items():
                try:
                    data[k] = json.loads(v)
                except:
                    data[k] = v
            obj_dict["data"] = data

        return StreamEvent(**obj_dict)

    async def _send_processed_result(self, result: Any, output_topics: List[str]):
        """Send processed result to output topics"""
        try:
            if isinstance(result, list):
                # Multiple events
                for event in result:
                    for topic in output_topics:
                        await self.send_event(topic, event)
            else:
                # Single event
                for topic in output_topics:
                    await self.send_event(topic, result)

        except Exception as e:
            logger.error(f"Error sending processed result: {e}")

    async def _send_to_error_topic(
        self, original_message, error_topic: str, error_msg: str
    ):
        """Send failed message to error topic"""
        try:
            error_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=StreamEventType.SYSTEM_ALERT,
                timestamp=time.time(),
                source_component="stream-processor",
                data={
                    "error_type": "processing_failure",
                    "error_message": error_msg,
                    "original_topic": original_message.topic,
                    "original_partition": original_message.partition,
                    "original_offset": original_message.offset,
                },
            )

            await self.send_event(error_topic, error_event)

        except Exception as e:
            logger.error(f"Error sending to error topic: {e}")

    async def _metrics_collector(self):
        """Collect and publish streaming metrics"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds

                # Calculate throughput
                current_time = time.time()
                uptime = current_time - self.start_time
                throughput = self.metrics.messages_consumed / max(uptime, 1)

                metrics_event = StreamEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=StreamEventType.PERFORMANCE_METRIC,
                    timestamp=current_time,
                    source_component="kafka-streaming-service",
                    data={
                        "messages_produced": self.metrics.messages_produced,
                        "messages_consumed": self.metrics.messages_consumed,
                        "processing_errors": self.metrics.processing_errors,
                        "throughput_per_second": throughput,
                        "error_rate": self.metrics.processing_errors
                        / max(self.metrics.messages_consumed, 1),
                        "uptime_seconds": uptime,
                        "active_processors": len(self.stream_processors),
                    },
                )

                await self.send_event("qbitel-metrics", metrics_event)

            except Exception as e:
                logger.error(f"Error collecting streaming metrics: {e}")
                await asyncio.sleep(30)

    async def _health_monitor(self):
        """Monitor streaming service health"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Check producer health
                producer_healthy = self.producer is not None

                # Check consumer health
                consumer_count = len(self.consumers)

                # Check processing health
                recent_activity = (
                    time.time() - self.metrics.last_processed_time
                ) < 300  # 5 minutes

                health_status = (
                    "healthy" if (producer_healthy and recent_activity) else "degraded"
                )

                health_event = StreamEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=StreamEventType.SYSTEM_ALERT,
                    timestamp=time.time(),
                    source_component="kafka-streaming-service",
                    data={
                        "health_status": health_status,
                        "producer_healthy": producer_healthy,
                        "consumer_count": consumer_count,
                        "recent_activity": recent_activity,
                        "last_processed_time": self.metrics.last_processed_time,
                    },
                )

                # Send health status to orchestrator
                message = Message(
                    id=f"streaming_health_{time.time()}",
                    timestamp=time.time(),
                    source="kafka_streaming_service",
                    destination="orchestrator",
                    message_type="health_check",
                    payload={
                        "component": "kafka_streaming_service",
                        "status": health_status,
                        "metrics": asdict(self.metrics),
                    },
                )

                await self.orchestrator.send_message(message)

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)

    # Public API
    async def send_event(
        self, topic: str, event: StreamEvent, partition_key: Optional[str] = None
    ):
        """Send event to Kafka topic"""
        try:
            if not self.producer:
                logger.error("Producer not initialized")
                return

            key = partition_key or event.partition_key

            await self.producer.send(topic=topic, value=event, key=key)

            self.metrics.messages_produced += 1

        except Exception as e:
            logger.error(f"Error sending event to topic {topic}: {e}")

    async def send_raw_message(
        self, topic: str, message: Dict[str, Any], key: Optional[str] = None
    ):
        """Send raw message to Kafka topic"""
        try:
            if not self.producer:
                logger.error("Producer not initialized")
                return

            await self.producer.send(
                topic=topic,
                value=json.dumps(message, default=str).encode("utf-8"),
                key=key.encode("utf-8") if key else None,
            )

            self.metrics.messages_produced += 1

        except Exception as e:
            logger.error(f"Error sending raw message to topic {topic}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current streaming metrics"""
        uptime = time.time() - self.start_time
        return {
            "messages_produced": self.metrics.messages_produced,
            "messages_consumed": self.metrics.messages_consumed,
            "processing_errors": self.metrics.processing_errors,
            "error_rate": self.metrics.processing_errors
            / max(self.metrics.messages_consumed, 1),
            "throughput_per_second": self.metrics.messages_consumed / max(uptime, 1),
            "uptime_seconds": uptime,
            "active_processors": len(self.stream_processors),
            "producer_healthy": self.producer is not None,
        }

    async def shutdown(self):
        """Shutdown streaming service"""
        logger.info("Shutting down Kafka Streaming Service...")

        self.running = False

        # Stop producer
        if self.producer:
            await self.producer.stop()

        # Stop consumers
        for consumer in self.consumers.values():
            await consumer.stop()

        # Shutdown thread pool
        self.processing_executor.shutdown(wait=True)

        logger.info("Kafka Streaming Service shutdown complete")


# Global streaming service instance
_streaming_service = None


def get_streaming_service() -> KafkaStreamingService:
    """Get global streaming service instance"""
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = KafkaStreamingService()
    return _streaming_service


async def main():
    """Main entry point for streaming service"""
    service = KafkaStreamingService()

    try:
        await service.initialize()
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Streaming service error: {e}")
    finally:
        await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
