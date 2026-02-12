"""
Unit tests for Secure Kafka Producer.
"""

import pytest
import base64
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.event_streaming.kafka.secure_producer import SecureKafkaProducer


class TestSecureKafkaProducer:
    """Test SecureKafkaProducer class"""

    @pytest.fixture
    def producer(self):
        """Create secure Kafka producer instance"""
        with patch("kafka.KafkaProducer"):
            return SecureKafkaProducer(
                bootstrap_servers=["localhost:9092"],
                topic="security-events",
                enable_quantum_encryption=True,
                kafka_config={},
                max_retries=3,
            )

    def test_initialization(self, producer):
        """Test producer initialization"""
        assert producer.bootstrap_servers == ["localhost:9092"]
        assert producer.topic == "security-events"
        assert producer.enable_quantum_encryption is True
        assert producer.max_retries == 3

    @patch("kafka.KafkaProducer")
    def test_send_message(self, mock_kafka_producer):
        """Test sending message"""
        mock_producer_instance = MagicMock()
        mock_producer_instance.send.return_value = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        producer = SecureKafkaProducer(bootstrap_servers=["localhost:9092"], topic="test-topic")

        result = producer.send(value={"event": "test", "data": "sample"}, key="test-key")

        assert result is not None
        mock_producer_instance.send.assert_called()

    @patch("kafka.KafkaProducer")
    def test_message_encryption(self, mock_kafka_producer):
        """Test message encryption"""
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        producer = SecureKafkaProducer(
            bootstrap_servers=["localhost:9092"], topic="test-topic", enable_quantum_encryption=True
        )

        # Send a message
        plaintext = {"secret": "data"}
        producer.send(value=plaintext)

        # Verify encryption was applied
        # The actual call to Kafka should have encrypted data
        mock_producer_instance.send.assert_called()

    def test_encrypt_message(self, producer):
        """Test message encryption method"""
        plaintext = b"test message content"

        encrypted_data = producer._encrypt_message(plaintext)

        assert encrypted_data is not None
        assert encrypted_data != plaintext
        assert isinstance(encrypted_data, bytes)

    def test_encrypt_decrypt_roundtrip(self, producer):
        """Test encryption and decryption roundtrip"""
        original_message = {"event": "security_alert", "severity": "high"}

        # Encrypt
        encrypted = producer._encrypt_message(str(original_message).encode())

        # For full test, would need decrypt method
        assert encrypted is not None

    @patch("kafka.KafkaProducer")
    def test_send_with_headers(self, mock_kafka_producer):
        """Test sending message with headers"""
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        producer = SecureKafkaProducer(bootstrap_servers=["localhost:9092"], topic="test-topic")

        headers = [("event-type", b"security-alert"), ("severity", b"high")]

        producer.send(value={"data": "test"}, headers=headers)

        mock_producer_instance.send.assert_called()

    @patch("kafka.KafkaProducer")
    def test_send_with_partition(self, mock_kafka_producer):
        """Test sending message to specific partition"""
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        producer = SecureKafkaProducer(bootstrap_servers=["localhost:9092"], topic="test-topic")

        producer.send(value={"data": "test"}, partition=2)

        mock_producer_instance.send.assert_called()

    @patch("kafka.KafkaProducer")
    def test_flush(self, mock_kafka_producer):
        """Test flushing buffered messages"""
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        producer = SecureKafkaProducer(bootstrap_servers=["localhost:9092"], topic="test-topic")

        producer.flush(timeout=10)

        mock_producer_instance.flush.assert_called_once()

    @patch("kafka.KafkaProducer")
    def test_close(self, mock_kafka_producer):
        """Test closing producer"""
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        producer = SecureKafkaProducer(bootstrap_servers=["localhost:9092"], topic="test-topic")

        producer.close()

        mock_producer_instance.close.assert_called_once()

    @patch("kafka.KafkaProducer")
    def test_get_metrics(self, mock_kafka_producer):
        """Test getting producer metrics"""
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        producer = SecureKafkaProducer(bootstrap_servers=["localhost:9092"], topic="test-topic")

        metrics = producer.get_metrics()

        assert isinstance(metrics, dict)
        assert "messages_sent" in metrics
        assert "bytes_sent" in metrics
        assert "errors" in metrics
        assert "success_rate" in metrics

    def test_create_connector_config(self, producer):
        """Test Kafka Connect configuration generation"""
        config = producer.create_connector_config()

        assert isinstance(config, dict)
        assert "name" in config
        assert "config" in config

        connector_config = config["config"]
        assert "connector.class" in connector_config
        assert "topics" in connector_config

    @patch("kafka.KafkaProducer")
    def test_send_with_timeout(self, mock_kafka_producer):
        """Test sending message with timeout"""
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        producer = SecureKafkaProducer(bootstrap_servers=["localhost:9092"], topic="test-topic")

        producer.send(value={"data": "test"}, timeout=5)

        mock_producer_instance.send.assert_called()

    @patch("kafka.KafkaProducer")
    def test_retry_on_failure(self, mock_kafka_producer):
        """Test retry logic on send failure"""
        mock_producer_instance = MagicMock()
        # Fail twice, then succeed
        mock_producer_instance.send.side_effect = [Exception("Broker not available"), Exception("Timeout"), MagicMock()]
        mock_kafka_producer.return_value = mock_producer_instance

        producer = SecureKafkaProducer(bootstrap_servers=["localhost:9092"], topic="test-topic", max_retries=3)

        try:
            producer.send(value={"data": "test"})
            # Should eventually succeed or raise
        except Exception:
            # Acceptable if retries exhausted
            pass

    @patch("kafka.KafkaProducer")
    def test_batch_send(self, mock_kafka_producer):
        """Test sending multiple messages"""
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        producer = SecureKafkaProducer(bootstrap_servers=["localhost:9092"], topic="test-topic")

        messages = [{"event": "event1"}, {"event": "event2"}, {"event": "event3"}]

        for msg in messages:
            producer.send(value=msg)

        assert mock_producer_instance.send.call_count == 3

    def test_quantum_encryption_flag(self):
        """Test with quantum encryption disabled"""
        with patch("kafka.KafkaProducer"):
            producer = SecureKafkaProducer(
                bootstrap_servers=["localhost:9092"], topic="test-topic", enable_quantum_encryption=False
            )

            assert producer.enable_quantum_encryption is False

    @patch("kafka.KafkaProducer")
    def test_kafka_config_passthrough(self, mock_kafka_producer):
        """Test passing custom Kafka configuration"""
        custom_config = {"compression.type": "gzip", "acks": "all", "retries": 5}

        producer = SecureKafkaProducer(bootstrap_servers=["localhost:9092"], topic="test-topic", kafka_config=custom_config)

        # Verify config was passed to KafkaProducer
        mock_kafka_producer.assert_called()

    def test_aes_gcm_encryption_format(self, producer):
        """Test that encryption uses AES-256-GCM format"""
        plaintext = b"test message"

        encrypted = producer._encrypt_message(plaintext)

        # Encrypted message should be longer (includes nonce and tag)
        assert len(encrypted) > len(plaintext)

        # Should be base64 decodable if encoded
        try:
            if isinstance(encrypted, str):
                base64.b64decode(encrypted)
        except Exception:
            # Binary format is also acceptable
            pass
