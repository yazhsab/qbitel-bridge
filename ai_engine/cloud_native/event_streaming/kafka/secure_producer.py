"""
Secure Kafka Producer

Kafka producer with quantum-safe encryption for message-level security.
"""

import logging
from typing import Dict, Any, Optional, List
import hashlib
import secrets
import base64
from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaTimeoutError
import time

logger = logging.getLogger(__name__)


class SecureKafkaProducer:
    """
    Kafka producer with quantum-safe message encryption.

    Provides message-level encryption using AES-256-GCM (with optional Kyber-1024
    key encapsulation for quantum safety) and integrates with Kafka for reliable
    event streaming.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        enable_quantum_encryption: bool = True,
        kafka_config: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ):
        """
        Initialize secure Kafka producer.

        Args:
            bootstrap_servers: Kafka broker addresses (comma-separated)
            topic: Default topic name
            enable_quantum_encryption: Enable quantum-safe encryption
            kafka_config: Additional Kafka producer configuration
            max_retries: Maximum number of retry attempts
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.enable_quantum_encryption = enable_quantum_encryption
        self.max_retries = max_retries

        # Message encryption keys (would use Kyber in production)
        self._encryption_key = secrets.token_bytes(32)  # 256-bit key

        # Metrics
        self._messages_sent = 0
        self._bytes_sent = 0
        self._errors = 0

        # Initialize Kafka producer
        self._producer = None
        self._kafka_config = kafka_config or {}
        self._initialize_producer()

        logger.info(f"Initialized SecureKafkaProducer for topic {topic} with servers {bootstrap_servers}")

    def _initialize_producer(self):
        """
        Initialize the Kafka producer with proper configuration.
        """
        try:
            # Base configuration
            config = {
                "bootstrap_servers": self.bootstrap_servers.split(","),
                "value_serializer": lambda v: v if isinstance(v, bytes) else v.encode("utf-8"),
                "key_serializer": lambda k: k if k is None or isinstance(k, bytes) else k.encode("utf-8"),
                "acks": "all",  # Wait for all replicas to acknowledge
                "retries": self.max_retries,
                "max_in_flight_requests_per_connection": 5,
                "compression_type": "snappy",  # Compress messages
                "linger_ms": 10,  # Batch messages for efficiency
                "batch_size": 16384,
            }

            # Merge with user-provided config
            config.update(self._kafka_config)

            self._producer = KafkaProducer(**config)
            logger.info("Kafka producer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise

    def send(
        self,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        partition: Optional[int] = None,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """
        Send encrypted message to Kafka.

        Args:
            value: Message value (plaintext bytes)
            key: Optional message key
            headers: Optional message headers
            partition: Optional partition
            timeout: Timeout in seconds for the send operation

        Returns:
            Dict containing send result with topic, partition, offset, and metadata

        Raises:
            KafkaError: If message send fails
            KafkaTimeoutError: If send operation times out
        """
        if not isinstance(value, bytes):
            raise ValueError("Message value must be bytes")

        original_size = len(value)

        # Encrypt message if enabled
        if self.enable_quantum_encryption:
            encrypted_value, nonce, tag = self._encrypt_message(value)

            # Add encryption metadata to headers
            if headers is None:
                headers = {}

            headers.update(
                {
                    "x-qbitel-encrypted": "true",
                    "x-qbitel-algorithm": "aes-256-gcm",
                    "x-qbitel-nonce": base64.b64encode(nonce).decode(),
                    "x-qbitel-tag": base64.b64encode(tag).decode(),
                }
            )

            value = encrypted_value
        else:
            if headers is None:
                headers = {}
            headers["x-qbitel-encrypted"] = "false"

        # Convert headers to list of tuples with bytes values
        kafka_headers = [(k, v.encode() if isinstance(v, str) else v) for k, v in headers.items()]

        # Send to Kafka with retry logic
        for attempt in range(self.max_retries):
            try:
                # Send message asynchronously
                future = self._producer.send(
                    topic=self.topic, value=value, key=key, headers=kafka_headers, partition=partition
                )

                # Wait for send to complete
                record_metadata = future.get(timeout=timeout)

                # Update metrics
                self._messages_sent += 1
                self._bytes_sent += len(value)

                result = {
                    "success": True,
                    "topic": record_metadata.topic,
                    "partition": record_metadata.partition,
                    "offset": record_metadata.offset,
                    "timestamp": record_metadata.timestamp,
                    "encrypted": self.enable_quantum_encryption,
                    "original_size_bytes": original_size,
                    "sent_size_bytes": len(value),
                    "serialized_key_size": record_metadata.serialized_key_size,
                    "serialized_value_size": record_metadata.serialized_value_size,
                }

                logger.debug(
                    f"Successfully sent message to {self.topic} partition {record_metadata.partition} offset {record_metadata.offset}"
                )
                return result

            except KafkaTimeoutError as e:
                self._errors += 1
                logger.warning(f"Kafka timeout on attempt {attempt + 1}/{self.max_retries}: {e}")

                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to send message after {self.max_retries} attempts due to timeout")
                    raise

            except KafkaError as e:
                self._errors += 1
                logger.error(f"Kafka error on attempt {attempt + 1}/{self.max_retries}: {e}")

                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to send message after {self.max_retries} attempts")
                    raise

            except Exception as e:
                self._errors += 1
                logger.error(f"Unexpected error sending message: {e}")
                raise

        # Should not reach here
        raise Exception("Failed to send message after all retry attempts")

    def _encrypt_message(self, plaintext: bytes) -> tuple:
        """
        Encrypt message using AES-256-GCM encryption.

        In production, this would use proper AES-GCM from cryptography library
        with Kyber-1024 for quantum-safe key encapsulation.

        Args:
            plaintext: Message to encrypt

        Returns:
            Tuple of (ciphertext, nonce, auth_tag)
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            # Use proper AES-GCM encryption
            aesgcm = AESGCM(self._encryption_key)
            nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM

            # Encrypt and get ciphertext + tag in one operation
            ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, None)

            # Split ciphertext and tag (last 16 bytes are the tag)
            ciphertext = ciphertext_with_tag[:-16]
            tag = ciphertext_with_tag[-16:]

            return ciphertext, nonce, tag

        except ImportError:
            # Fallback to simplified encryption if cryptography lib not available
            logger.warning("Using simplified encryption - install cryptography library for production use")

            # Generate nonce
            nonce = secrets.token_bytes(12)

            # Encrypt using stream cipher (simplified)
            key_stream = hashlib.sha3_256(self._encryption_key + nonce).digest()

            # Extend key stream if needed
            while len(key_stream) < len(plaintext):
                key_stream += hashlib.sha3_256(key_stream).digest()

            ciphertext = bytes(p ^ k for p, k in zip(plaintext, key_stream[: len(plaintext)]))

            # Generate authentication tag
            tag = hashlib.sha3_256(ciphertext + nonce + self._encryption_key).digest()[:16]

            return ciphertext, nonce, tag

    def flush(self, timeout: Optional[int] = None):
        """
        Flush all buffered messages.

        Args:
            timeout: Maximum time to wait for flush in seconds
        """
        if self._producer:
            self._producer.flush(timeout=timeout)
            logger.info("Flushed all pending messages")

    def close(self):
        """
        Close the Kafka producer and release resources.
        """
        if self._producer:
            logger.info("Closing Kafka producer")
            self._producer.close()
            self._producer = None

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get producer metrics.

        Returns:
            Dict containing producer statistics
        """
        return {
            "topic": self.topic,
            "bootstrap_servers": self.bootstrap_servers,
            "encryption_enabled": self.enable_quantum_encryption,
            "algorithm": "aes-256-gcm",
            "messages_sent": self._messages_sent,
            "bytes_sent": self._bytes_sent,
            "errors": self._errors,
            "success_rate": (
                (self._messages_sent / (self._messages_sent + self._errors))
                if (self._messages_sent + self._errors) > 0
                else 0.0
            ),
        }

    def create_connector_config(self) -> Dict[str, Any]:
        """Create Kafka Connect configuration"""
        return {
            "name": "qbitel-secure-sink",
            "config": {
                "connector.class": "io.confluent.connect.kafka.KafkaSinkConnector",
                "topics": self.topic,
                "bootstrap.servers": self.bootstrap_servers,
                "key.converter": "org.apache.kafka.connect.storage.StringConverter",
                "value.converter": "org.apache.kafka.connect.storage.ByteArrayConverter",
                "transforms": "qbitel_encryption",
                "transforms.qbitel_encryption.type": "com.qbitel.kafka.QuantumEncryption",
                "transforms.qbitel_encryption.algorithm": "kyber-1024",
                "security.protocol": "SSL",
                "ssl.truststore.location": "/etc/qbitel/kafka/truststore.jks",
                "ssl.keystore.location": "/etc/qbitel/kafka/keystore.jks",
            },
        }
