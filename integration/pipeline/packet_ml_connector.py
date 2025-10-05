#!/usr/bin/env python3
"""
CRONOS AI - Packet Processing to AI/ML Pipeline Connector
Connects packet processing from Rust dataplane to AI/ML analysis pipeline.
"""

import asyncio
import logging
import json
import time
import struct
import socket
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import ctypes
from ctypes import Structure, c_uint8, c_uint16, c_uint32, c_uint64, c_char_p, POINTER
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
import sys
import hashlib

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from integration.config.unified_config import get_config, get_service_config
from integration.orchestrator.service_integration import get_orchestrator, Message
from integration.streaming.kafka_streaming_service import (
    get_streaming_service,
    StreamEvent,
    StreamEventType,
)
from integration.ai_pipeline.ml_protocol_bridge import get_ml_bridge, ProtocolData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PacketDirection(Enum):
    """Packet direction enumeration"""

    INGRESS = "ingress"
    EGRESS = "egress"
    INTERNAL = "internal"


class ProcessingPriority(Enum):
    """Processing priority levels"""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class PacketMetadata:
    """Packet metadata structure"""

    packet_id: str
    timestamp: float
    capture_interface: str
    packet_size: int
    direction: PacketDirection
    priority: ProcessingPriority
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    protocol: str
    vlan_id: Optional[int] = None
    mpls_labels: Optional[List[int]] = None


@dataclass
class ProcessedPacket:
    """Processed packet structure"""

    metadata: PacketMetadata
    raw_data: bytes
    extracted_payload: bytes
    protocol_hints: Dict[str, Any]
    processing_flags: Dict[str, bool]


# C structures for interfacing with Rust dataplane
class CPacketHeader(Structure):
    """C structure matching Rust packet header"""

    _fields_ = [
        ("timestamp_sec", c_uint64),
        ("timestamp_nsec", c_uint32),
        ("packet_size", c_uint32),
        ("source_ip", c_uint32),
        ("dest_ip", c_uint32),
        ("source_port", c_uint16),
        ("dest_port", c_uint16),
        ("protocol", c_uint8),
        ("flags", c_uint8),
    ]


class CPacketData(Structure):
    """C structure for packet data"""

    _fields_ = [
        ("header", CPacketHeader),
        ("data_len", c_uint32),
        ("data", c_char_p),
    ]


class PacketMLConnector:
    """
    High-performance connector between packet processing and ML pipeline.
    Handles real-time packet analysis and routing.
    """

    def __init__(self):
        self.config = get_config()
        self.orchestrator = get_orchestrator()
        self.streaming_service = get_streaming_service()
        self.ml_bridge = get_ml_bridge()

        # Performance settings
        self.max_packet_size = 65536  # 64KB max packet size
        self.batch_size = 1000  # Process packets in batches
        self.batch_timeout = 0.1  # 100ms batch timeout

        # Processing queues
        self.packet_queue = asyncio.Queue(maxsize=100000)  # High throughput queue
        self.analysis_queue = asyncio.Queue(maxsize=50000)
        self.result_queue = asyncio.Queue(maxsize=50000)

        # Connection to Rust dataplane
        self.rust_interface = None
        self.interface_thread = None

        # Processing threads
        self.processing_threads = []
        self.running = False

        # Performance metrics
        self.packets_processed = 0
        self.packets_analyzed = 0
        self.processing_errors = 0
        self.average_latency_ms = 0.0
        self.throughput_pps = 0.0  # Packets per second

        # Packet filtering and sampling
        self.packet_filters = []
        self.sampling_rate = 1.0  # Process all packets by default
        self.filter_cache = {}

        # Protocol detection cache
        self.protocol_cache = {}
        self.cache_hit_count = 0

        # Thread pools
        self.packet_processor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="packet_proc_"
        )
        self.analysis_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="packet_analysis_"
        )

        # Load Rust library if available
        self._load_rust_library()

    def _load_rust_library(self):
        """Load Rust dataplane library for FFI"""
        try:
            # Try to load the compiled Rust library
            rust_lib_path = (
                project_root / "rust/dataplane/target/release/libdpi_engine.so"
            )
            if rust_lib_path.exists():
                self.rust_lib = ctypes.CDLL(str(rust_lib_path))

                # Define function signatures
                self.rust_lib.start_packet_capture.argtypes = [c_char_p]
                self.rust_lib.start_packet_capture.restype = ctypes.c_int

                self.rust_lib.get_next_packet.argtypes = [POINTER(CPacketData)]
                self.rust_lib.get_next_packet.restype = ctypes.c_int

                self.rust_lib.stop_packet_capture.argtypes = []
                self.rust_lib.stop_packet_capture.restype = ctypes.c_int

                logger.info("Rust dataplane library loaded successfully")
            else:
                logger.warning("Rust dataplane library not found, using mock interface")
                self.rust_lib = None

        except Exception as e:
            logger.error(f"Failed to load Rust library: {e}")
            self.rust_lib = None

    async def initialize(self):
        """Initialize packet processing connector"""
        logger.info("Initializing Packet-ML Connector...")

        try:
            # Initialize packet filters
            await self._init_packet_filters()

            # Start packet capture interface
            await self._start_packet_capture()

            # Start processing tasks
            await self._start_processing_tasks()

            logger.info("Packet-ML Connector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Packet-ML Connector: {e}")
            raise

    async def _init_packet_filters(self):
        """Initialize packet filtering rules"""
        try:
            # Load packet filters from configuration
            self.packet_filters = [
                {
                    "name": "skip_icmp",
                    "condition": lambda pkt: pkt.protocol != "ICMP",
                    "enabled": True,
                },
                {
                    "name": "skip_arp",
                    "condition": lambda pkt: pkt.protocol != "ARP",
                    "enabled": True,
                },
                {
                    "name": "size_filter",
                    "condition": lambda pkt: pkt.packet_size
                    >= 64,  # Skip too small packets
                    "enabled": True,
                },
                {
                    "name": "suspicious_ports",
                    "condition": lambda pkt: self._is_suspicious_port(pkt.dest_port)
                    or self._is_suspicious_port(pkt.source_port),
                    "enabled": True,
                    "priority": ProcessingPriority.HIGH,
                },
            ]

            # Load sampling rate from config
            self.sampling_rate = (
                self.config.ai_engine.get("sampling_rate", 1.0)
                if self.config.ai_engine
                else 1.0
            )

            logger.info(f"Initialized {len(self.packet_filters)} packet filters")

        except Exception as e:
            logger.error(f"Error initializing packet filters: {e}")

    def _is_suspicious_port(self, port: int) -> bool:
        """Check if port is suspicious"""
        suspicious_ports = {1337, 4444, 5555, 6666, 31337, 8080, 8888}
        return port in suspicious_ports

    async def _start_packet_capture(self):
        """Start packet capture from Rust dataplane"""
        try:
            if self.rust_lib:
                # Start real packet capture
                interface = b"eth0"  # Default interface
                result = self.rust_lib.start_packet_capture(interface)
                if result == 0:
                    logger.info("Started packet capture from Rust dataplane")
                    # Start interface thread
                    self.interface_thread = threading.Thread(
                        target=self._packet_capture_loop, daemon=True
                    )
                    self.interface_thread.start()
                else:
                    logger.error("Failed to start packet capture")
                    await self._start_mock_capture()
            else:
                # Use mock packet capture for testing
                await self._start_mock_capture()

        except Exception as e:
            logger.error(f"Error starting packet capture: {e}")
            await self._start_mock_capture()

    def _packet_capture_loop(self):
        """Packet capture loop (runs in separate thread)"""
        logger.info("Started packet capture loop")

        while self.running:
            try:
                if self.rust_lib:
                    # Get packet from Rust
                    packet_data = CPacketData()
                    result = self.rust_lib.get_next_packet(ctypes.byref(packet_data))

                    if result == 0 and packet_data.data_len > 0:
                        # Convert C struct to Python objects
                        processed_packet = self._convert_c_packet(packet_data)

                        # Add to queue (non-blocking)
                        try:
                            asyncio.run_coroutine_threadsafe(
                                self.packet_queue.put(processed_packet),
                                asyncio.get_event_loop(),
                            ).result(
                                timeout=0.001
                            )  # 1ms timeout
                        except:
                            # Drop packet if queue is full
                            pass

                    time.sleep(0.0001)  # 100µs sleep to prevent CPU spinning
                else:
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in packet capture loop: {e}")
                time.sleep(0.1)

        logger.info("Packet capture loop stopped")

    def _convert_c_packet(self, c_packet: CPacketData) -> ProcessedPacket:
        """Convert C packet structure to Python ProcessedPacket"""
        try:
            header = c_packet.header

            # Convert IP addresses
            source_ip = socket.inet_ntoa(struct.pack("!I", header.source_ip))
            dest_ip = socket.inet_ntoa(struct.pack("!I", header.dest_ip))

            # Convert protocol number to string
            protocol_map = {1: "ICMP", 6: "TCP", 17: "UDP"}
            protocol = protocol_map.get(header.protocol, f"PROTO_{header.protocol}")

            # Create metadata
            metadata = PacketMetadata(
                packet_id=f"pkt_{header.timestamp_sec}_{header.timestamp_nsec}",
                timestamp=header.timestamp_sec + (header.timestamp_nsec / 1e9),
                capture_interface="eth0",
                packet_size=c_packet.data_len,
                direction=PacketDirection.INGRESS,
                priority=ProcessingPriority.NORMAL,
                source_ip=source_ip,
                dest_ip=dest_ip,
                source_port=header.source_port,
                dest_port=header.dest_port,
                protocol=protocol,
            )

            # Extract packet data
            raw_data = ctypes.string_at(c_packet.data, c_packet.data_len)

            # Extract payload (skip headers - simplified)
            payload_offset = self._calculate_payload_offset(protocol, raw_data)
            extracted_payload = (
                raw_data[payload_offset:] if payload_offset < len(raw_data) else b""
            )

            # Create processed packet
            processed_packet = ProcessedPacket(
                metadata=metadata,
                raw_data=raw_data,
                extracted_payload=extracted_payload,
                protocol_hints={"detected_protocol": protocol},
                processing_flags={"requires_analysis": True},
            )

            return processed_packet

        except Exception as e:
            logger.error(f"Error converting C packet: {e}")
            # Return empty packet on error
            return ProcessedPacket(
                metadata=PacketMetadata(
                    packet_id="error_packet",
                    timestamp=time.time(),
                    capture_interface="unknown",
                    packet_size=0,
                    direction=PacketDirection.INTERNAL,
                    priority=ProcessingPriority.LOW,
                    source_ip="0.0.0.0",
                    dest_ip="0.0.0.0",
                    source_port=0,
                    dest_port=0,
                    protocol="UNKNOWN",
                ),
                raw_data=b"",
                extracted_payload=b"",
                protocol_hints={},
                processing_flags={"requires_analysis": False},
            )

    def _calculate_payload_offset(self, protocol: str, data: bytes) -> int:
        """Calculate offset to payload data"""
        try:
            if protocol == "TCP":
                if len(data) >= 20:  # Minimum TCP header
                    tcp_header_len = ((data[12] >> 4) & 0xF) * 4
                    return 20 + tcp_header_len  # IP header (20) + TCP header
            elif protocol == "UDP":
                return 28  # IP header (20) + UDP header (8)
            elif protocol == "ICMP":
                return 28  # IP header (20) + ICMP header (8)

            return 20  # Default IP header size

        except Exception as e:
            logger.error(f"Error calculating payload offset: {e}")
            return 0

    async def _start_mock_capture(self):
        """Start mock packet capture for testing"""
        logger.info("Starting mock packet capture for testing")

        async def mock_packet_generator():
            """Generate mock packets for testing"""
            packet_id = 0

            while self.running:
                try:
                    # Generate mock packet
                    packet_id += 1

                    mock_metadata = PacketMetadata(
                        packet_id=f"mock_pkt_{packet_id}",
                        timestamp=time.time(),
                        capture_interface="mock0",
                        packet_size=256,
                        direction=PacketDirection.INGRESS,
                        priority=ProcessingPriority.NORMAL,
                        source_ip="192.168.1.100",
                        dest_ip="10.0.0.1",
                        source_port=12345,
                        dest_port=80,
                        protocol="TCP",
                    )

                    # Generate mock HTTP request
                    mock_payload = b"GET / HTTP/1.1\r\nHost: example.com\r\nUser-Agent: Mozilla/5.0\r\n\r\n"
                    mock_raw = (
                        b"\x45\x00" + b"\x00" * 18 + mock_payload
                    )  # Simplified packet

                    mock_packet = ProcessedPacket(
                        metadata=mock_metadata,
                        raw_data=mock_raw,
                        extracted_payload=mock_payload,
                        protocol_hints={"detected_protocol": "HTTP"},
                        processing_flags={"requires_analysis": True},
                    )

                    await self.packet_queue.put(mock_packet)
                    await asyncio.sleep(0.01)  # 100 packets per second

                except Exception as e:
                    logger.error(f"Error in mock packet generator: {e}")
                    await asyncio.sleep(1)

        # Start mock packet generator
        asyncio.create_task(mock_packet_generator())

    async def _start_processing_tasks(self):
        """Start packet processing background tasks"""
        self.running = True

        # Start packet processors
        for i in range(4):  # 4 packet processing workers
            asyncio.create_task(self._packet_processor_worker(f"processor_{i}"))

        # Start analysis processors
        for i in range(2):  # 2 analysis workers
            asyncio.create_task(self._analysis_processor_worker(f"analyzer_{i}"))

        # Start result handler
        asyncio.create_task(self._result_handler())

        # Start batch processor
        asyncio.create_task(self._batch_processor())

        # Start metrics collector
        asyncio.create_task(self._metrics_collector())

        logger.info("Started packet processing tasks")

    async def _packet_processor_worker(self, worker_id: str):
        """Process individual packets"""
        logger.info(f"Started packet processor worker: {worker_id}")

        while self.running:
            try:
                # Get packet from queue
                packet = await asyncio.wait_for(self.packet_queue.get(), timeout=1.0)

                # Apply filters
                if not await self._apply_filters(packet):
                    continue

                # Apply sampling
                if not self._should_sample():
                    continue

                # Check protocol cache
                cache_key = self._generate_packet_cache_key(packet)
                if cache_key in self.protocol_cache:
                    cached_result = self.protocol_cache[cache_key]
                    packet.protocol_hints.update(cached_result)
                    self.cache_hit_count += 1

                # Add to analysis queue
                await self.analysis_queue.put(packet)
                self.packets_processed += 1

            except asyncio.TimeoutError:
                continue  # Normal timeout
            except Exception as e:
                logger.error(f"Error in packet processor {worker_id}: {e}")
                self.processing_errors += 1
                await asyncio.sleep(0.1)

    async def _analysis_processor_worker(self, worker_id: str):
        """Process packets for AI analysis"""
        logger.info(f"Started analysis processor worker: {worker_id}")

        while self.running:
            try:
                # Get packet from analysis queue
                packet = await asyncio.wait_for(self.analysis_queue.get(), timeout=1.0)

                # Create protocol data for ML analysis
                protocol_data = ProtocolData(
                    raw_data=packet.extracted_payload,
                    timestamp=packet.metadata.timestamp,
                    source_ip=packet.metadata.source_ip,
                    dest_ip=packet.metadata.dest_ip,
                    source_port=packet.metadata.source_port,
                    dest_port=packet.metadata.dest_port,
                    protocol=packet.metadata.protocol,
                    metadata=asdict(packet.metadata),
                )

                # Queue for ML analysis
                await self.ml_bridge.queue_analysis(
                    protocol_data, correlation_id=packet.metadata.packet_id
                )

                # Create stream event for Kafka
                stream_event = StreamEvent(
                    event_id=packet.metadata.packet_id,
                    event_type=StreamEventType.PACKET_CAPTURED,
                    timestamp=packet.metadata.timestamp,
                    source_component="packet_ml_connector",
                    data={
                        "packet_metadata": asdict(packet.metadata),
                        "protocol_hints": packet.protocol_hints,
                        "payload_size": len(packet.extracted_payload),
                        "processing_priority": packet.metadata.priority.value,
                    },
                    correlation_id=packet.metadata.packet_id,
                    partition_key=f"{packet.metadata.source_ip}:{packet.metadata.dest_ip}",
                )

                # Send to Kafka
                await self.streaming_service.send_event(
                    "cronos-ai-packet-processing", stream_event
                )

                self.packets_analyzed += 1

            except asyncio.TimeoutError:
                continue  # Normal timeout
            except Exception as e:
                logger.error(f"Error in analysis processor {worker_id}: {e}")
                self.processing_errors += 1
                await asyncio.sleep(0.1)

    async def _batch_processor(self):
        """Process packets in batches for efficiency"""
        packet_batch = []
        last_batch_time = time.time()

        while self.running:
            try:
                await asyncio.sleep(0.001)  # 1ms processing cycle

                current_time = time.time()
                batch_ready = len(packet_batch) >= self.batch_size or (
                    len(packet_batch) > 0
                    and (current_time - last_batch_time) > self.batch_timeout
                )

                if batch_ready and packet_batch:
                    # Process batch
                    await self._process_packet_batch(packet_batch)
                    packet_batch.clear()
                    last_batch_time = current_time

            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(0.1)

    async def _process_packet_batch(self, packets: List[ProcessedPacket]):
        """Process a batch of packets efficiently"""
        try:
            # Group packets by characteristics for efficient processing
            protocol_groups = {}
            for packet in packets:
                protocol = packet.metadata.protocol
                if protocol not in protocol_groups:
                    protocol_groups[protocol] = []
                protocol_groups[protocol].append(packet)

            # Process each group
            for protocol, group_packets in protocol_groups.items():
                await self._process_protocol_group(protocol, group_packets)

        except Exception as e:
            logger.error(f"Error processing packet batch: {e}")

    async def _process_protocol_group(
        self, protocol: str, packets: List[ProcessedPacket]
    ):
        """Process a group of packets with the same protocol"""
        try:
            # Optimize processing for specific protocols
            if protocol == "HTTP":
                await self._process_http_packets(packets)
            elif protocol == "TCP":
                await self._process_tcp_packets(packets)
            else:
                # Generic processing
                for packet in packets:
                    await self._process_generic_packet(packet)

        except Exception as e:
            logger.error(f"Error processing {protocol} packet group: {e}")

    async def _process_http_packets(self, packets: List[ProcessedPacket]):
        """Optimized processing for HTTP packets"""
        for packet in packets:
            try:
                # Extract HTTP headers and analyze
                http_data = packet.extracted_payload.decode("utf-8", errors="ignore")
                if "HTTP" in http_data:
                    # Quick HTTP analysis
                    packet.protocol_hints["http_method"] = self._extract_http_method(
                        http_data
                    )
                    packet.protocol_hints["http_host"] = self._extract_http_host(
                        http_data
                    )
                    packet.processing_flags["high_priority"] = (
                        "admin" in http_data.lower()
                    )

            except Exception as e:
                logger.error(f"Error processing HTTP packet: {e}")

    async def _process_tcp_packets(self, packets: List[ProcessedPacket]):
        """Optimized processing for TCP packets"""
        for packet in packets:
            try:
                # TCP-specific analysis
                packet.protocol_hints["payload_entropy"] = self._calculate_entropy(
                    packet.extracted_payload
                )
                packet.processing_flags["encrypted"] = (
                    packet.protocol_hints["payload_entropy"] > 7.0
                )

            except Exception as e:
                logger.error(f"Error processing TCP packet: {e}")

    async def _process_generic_packet(self, packet: ProcessedPacket):
        """Generic packet processing"""
        try:
            # Basic analysis for unknown protocols
            packet.protocol_hints["payload_size"] = len(packet.extracted_payload)
            packet.protocol_hints["has_payload"] = len(packet.extracted_payload) > 0

        except Exception as e:
            logger.error(f"Error processing generic packet: {e}")

    async def _result_handler(self):
        """Handle analysis results"""
        while self.running:
            try:
                # This would handle results from ML analysis
                await asyncio.sleep(1)
                # Results are handled via Kafka streaming

            except Exception as e:
                logger.error(f"Error in result handler: {e}")
                await asyncio.sleep(1)

    async def _metrics_collector(self):
        """Collect performance metrics"""
        last_packet_count = 0
        last_time = time.time()

        while self.running:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds

                current_time = time.time()
                current_packets = self.packets_processed

                # Calculate throughput
                time_delta = current_time - last_time
                packet_delta = current_packets - last_packet_count

                if time_delta > 0:
                    self.throughput_pps = packet_delta / time_delta

                # Calculate error rate
                error_rate = self.processing_errors / max(self.packets_processed, 1)

                # Create metrics event
                metrics_event = StreamEvent(
                    event_id=f"connector_metrics_{int(current_time)}",
                    event_type=StreamEventType.PERFORMANCE_METRIC,
                    timestamp=current_time,
                    source_component="packet_ml_connector",
                    data={
                        "packets_processed": self.packets_processed,
                        "packets_analyzed": self.packets_analyzed,
                        "processing_errors": self.processing_errors,
                        "error_rate": error_rate,
                        "throughput_pps": self.throughput_pps,
                        "cache_hit_rate": self.cache_hit_count
                        / max(self.packets_processed, 1),
                        "queue_sizes": {
                            "packet_queue": self.packet_queue.qsize(),
                            "analysis_queue": self.analysis_queue.qsize(),
                        },
                    },
                )

                await self.streaming_service.send_event(
                    "cronos-ai-metrics", metrics_event
                )

                # Update for next iteration
                last_packet_count = current_packets
                last_time = current_time

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(30)

    # Utility methods
    async def _apply_filters(self, packet: ProcessedPacket) -> bool:
        """Apply packet filters"""
        try:
            for filter_rule in self.packet_filters:
                if filter_rule.get("enabled", True):
                    if not filter_rule["condition"](packet.metadata):
                        return False

                    # Set priority if specified
                    if "priority" in filter_rule:
                        packet.metadata.priority = filter_rule["priority"]

            return True

        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return True  # Allow packet through on error

    def _should_sample(self) -> bool:
        """Determine if packet should be sampled"""
        import random

        return random.random() <= self.sampling_rate

    def _generate_packet_cache_key(self, packet: ProcessedPacket) -> str:
        """Generate cache key for packet"""
        try:
            key_data = f"{packet.metadata.protocol}:{packet.metadata.dest_port}:{packet.extracted_payload[:32]}"
            return hashlib.md5(key_data.encode()).hexdigest()[:8]
        except:
            return "default"

    def _extract_http_method(self, http_data: str) -> str:
        """Extract HTTP method from request"""
        try:
            first_line = http_data.split("\r\n")[0]
            return first_line.split(" ")[0]
        except:
            return "UNKNOWN"

    def _extract_http_host(self, http_data: str) -> str:
        """Extract HTTP host header"""
        try:
            for line in http_data.split("\r\n"):
                if line.lower().startswith("host:"):
                    return line.split(":", 1)[1].strip()
            return "unknown"
        except:
            return "unknown"

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        try:
            if not data:
                return 0.0

            # Count byte frequencies
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1

            # Calculate entropy
            entropy = 0.0
            data_len = len(data)

            for count in byte_counts:
                if count > 0:
                    probability = count / data_len
                    entropy -= probability * (probability.bit_length() - 1)

            return entropy

        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 0.0

    # Public API
    def get_metrics(self) -> Dict[str, Any]:
        """Get current connector metrics"""
        return {
            "packets_processed": self.packets_processed,
            "packets_analyzed": self.packets_analyzed,
            "processing_errors": self.processing_errors,
            "throughput_pps": self.throughput_pps,
            "error_rate": self.processing_errors / max(self.packets_processed, 1),
            "cache_hit_rate": self.cache_hit_count / max(self.packets_processed, 1),
            "queue_utilization": {
                "packet_queue": self.packet_queue.qsize() / self.packet_queue.maxsize,
                "analysis_queue": self.analysis_queue.qsize()
                / self.analysis_queue.maxsize,
            },
        }

    async def add_packet_filter(
        self,
        name: str,
        condition: Callable,
        priority: Optional[ProcessingPriority] = None,
    ):
        """Add custom packet filter"""
        filter_rule = {
            "name": name,
            "condition": condition,
            "enabled": True,
        }

        if priority:
            filter_rule["priority"] = priority

        self.packet_filters.append(filter_rule)
        logger.info(f"Added packet filter: {name}")

    async def set_sampling_rate(self, rate: float):
        """Set packet sampling rate (0.0 to 1.0)"""
        self.sampling_rate = max(0.0, min(1.0, rate))
        logger.info(f"Set sampling rate to: {self.sampling_rate}")

    async def shutdown(self):
        """Shutdown packet connector"""
        logger.info("Shutting down Packet-ML Connector...")

        self.running = False

        # Stop Rust packet capture
        if self.rust_lib:
            self.rust_lib.stop_packet_capture()

        # Wait for interface thread
        if self.interface_thread:
            self.interface_thread.join(timeout=5)

        # Shutdown thread pools
        self.packet_processor.shutdown(wait=True)
        self.analysis_executor.shutdown(wait=True)

        logger.info("Packet-ML Connector shutdown complete")


# Global connector instance
_packet_connector = None


def get_packet_connector() -> PacketMLConnector:
    """Get global packet connector instance"""
    global _packet_connector
    if _packet_connector is None:
        _packet_connector = PacketMLConnector()
    return _packet_connector


async def main():
    """Main entry point for packet connector"""
    connector = PacketMLConnector()

    try:
        await connector.initialize()

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Packet connector error: {e}")
    finally:
        await connector.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
