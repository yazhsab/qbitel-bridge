"""
Unit tests for Secure Kafka Producer
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import base64

# Mock kafka module before importing SecureKafkaProducer
import sys
from unittest.mock import MagicMock

kafka_mock = MagicMock()
sys.modules["kafka"] = kafka_mock
sys.modules["kafka.errors"] = kafka_mock.errors


# Now we can safely define these as they'll be used in the tests
class KafkaError(Exception):
    pass


class KafkaTimeoutError(Exception):
    pass


kafka_mock.errors.KafkaError = KafkaError
kafka_mock.errors.KafkaTimeoutError = KafkaTimeoutError

from ai_engine.cloud_native.event_streaming.kafka.secure_producer import SecureKafkaProducer


class TestSecureKafkaProducer:
    """Test suite for Secure Kafka Producer"""

    @pytest.fixture
    def mock_kafka_producer(self):
        """Mock KafkaProducer"""
        with patch("ai_engine.cloud_native.event_streaming.kafka.secure_producer.KafkaProducer") as mock:
            producer_instance = MagicMock()
            mock.return_value = producer_instance
            yield producer_instance

    @pytest.fixture
    def producer(self, mock_kafka_producer):
        """Create a SecureKafkaProducer instance"""
        return SecureKafkaProducer(bootstrap_servers="localhost:9092", topic="test-topic", enable_quantum_encryption=True)

    @pytest.fixture
    def producer_no_encryption(self, mock_kafka_producer):
        """Create a SecureKafkaProducer without encryption"""
        return SecureKafkaProducer(bootstrap_servers="localhost:9092", topic="test-topic", enable_quantum_encryption=False)

    def test_initialization(self, producer, mock_kafka_producer):
        """Test producer initialization"""
        assert producer.bootstrap_servers == "localhost:9092"
        assert producer.topic == "test-topic"
        assert producer.enable_quantum_encryption is True
        assert producer._messages_sent == 0
        assert producer._bytes_sent == 0
        assert producer._errors == 0
        assert producer._producer is not None

    def test_initialization_with_custom_config(self, mock_kafka_producer):
        """Test initialization with custom Kafka config"""
        custom_config = {"acks": 1, "compression_type": "gzip"}

        producer = SecureKafkaProducer(bootstrap_servers="localhost:9092", topic="test-topic", kafka_config=custom_config)

        # Verify KafkaProducer was called with merged config
        call_args = mock_kafka_producer.call_args
        assert call_args is not None

    def test_send_message_success(self, producer, mock_kafka_producer):
        """Test successful message sending"""
        # Setup mock
        future = MagicMock()
        record_metadata = MagicMock()
        record_metadata.topic = "test-topic"
        record_metadata.partition = 0
        record_metadata.offset = 123
        record_metadata.timestamp = 1234567890
        record_metadata.serialized_key_size = 0
        record_metadata.serialized_value_size = 100
        future.get.return_value = record_metadata
        mock_kafka_producer.send.return_value = future

        # Send message
        message = b"Test message"
        result = producer.send(message)

        # Verify result
        assert result["success"] is True
        assert result["topic"] == "test-topic"
        assert result["partition"] == 0
        assert result["offset"] == 123
        assert result["encrypted"] is True
        assert producer._messages_sent == 1

        # Verify Kafka send was called
        mock_kafka_producer.send.assert_called_once()

    def test_send_message_without_encryption(self, producer_no_encryption, mock_kafka_producer):
        """Test sending message without encryption"""
        # Setup mock
        future = MagicMock()
        record_metadata = MagicMock()
        record_metadata.topic = "test-topic"
        record_metadata.partition = 0
        record_metadata.offset = 123
        record_metadata.timestamp = 1234567890
        record_metadata.serialized_key_size = 0
        record_metadata.serialized_value_size = 12
        future.get.return_value = record_metadata
        mock_kafka_producer.send.return_value = future

        # Send message
        message = b"Test message"
        result = producer_no_encryption.send(message)

        # Verify result
        assert result["success"] is True
        assert result["encrypted"] is False

        # Verify headers indicate no encryption
        call_args = mock_kafka_producer.send.call_args
        headers = dict(call_args[1]["headers"])
        assert b"false" in headers[b"x-qbitel-encrypted"]

    def test_send_with_encryption_headers(self, producer, mock_kafka_producer):
        """Test that encryption headers are added"""
        # Setup mock
        future = MagicMock()
        record_metadata = MagicMock()
        record_metadata.topic = "test-topic"
        record_metadata.partition = 0
        record_metadata.offset = 123
        record_metadata.timestamp = 1234567890
        record_metadata.serialized_key_size = 0
        record_metadata.serialized_value_size = 100
        future.get.return_value = record_metadata
        mock_kafka_producer.send.return_value = future

        # Send message
        message = b"Test message"
        result = producer.send(message, headers={"custom": "value"})

        # Verify encryption headers were added
        call_args = mock_kafka_producer.send.call_args
        headers = dict(call_args[1]["headers"])

        assert b"true" in headers.get(b"x-qbitel-encrypted", b"")
        assert b"aes-256-gcm" in headers.get(b"x-qbitel-algorithm", b"")
        assert b"x-qbitel-nonce" in headers
        assert b"x-qbitel-tag" in headers
        assert b"custom" in headers

    def test_send_with_key_and_partition(self, producer, mock_kafka_producer):
        """Test sending with key and specific partition"""
        # Setup mock
        future = MagicMock()
        record_metadata = MagicMock()
        record_metadata.topic = "test-topic"
        record_metadata.partition = 5
        record_metadata.offset = 456
        record_metadata.timestamp = 1234567890
        record_metadata.serialized_key_size = 10
        record_metadata.serialized_value_size = 100
        future.get.return_value = record_metadata
        mock_kafka_producer.send.return_value = future

        # Send message
        result = producer.send(value=b"Test message", key=b"test-key", partition=5)

        # Verify
        assert result["partition"] == 5
        call_args = mock_kafka_producer.send.call_args
        assert call_args[1]["key"] == b"test-key"
        assert call_args[1]["partition"] == 5

    def test_send_invalid_value_type(self, producer):
        """Test that sending non-bytes value raises error"""
        with pytest.raises(ValueError, match="Message value must be bytes"):
            producer.send("This is a string, not bytes")

    def test_send_with_kafka_timeout_and_retry(self, producer, mock_kafka_producer):
        """Test retry logic on KafkaTimeoutError"""
        # Setup mock to fail twice, then succeed
        future_fail = MagicMock()
        future_fail.get.side_effect = KafkaTimeoutError("Timeout")

        future_success = MagicMock()
        record_metadata = MagicMock()
        record_metadata.topic = "test-topic"
        record_metadata.partition = 0
        record_metadata.offset = 123
        record_metadata.timestamp = 1234567890
        record_metadata.serialized_key_size = 0
        record_metadata.serialized_value_size = 100
        future_success.get.return_value = record_metadata

        mock_kafka_producer.send.side_effect = [future_fail, future_fail, future_success]

        # Send message - should succeed after retries
        with patch("time.sleep"):  # Speed up test
            result = producer.send(b"Test message")

        assert result["success"] is True
        assert mock_kafka_producer.send.call_count == 3
        assert producer._errors == 2  # Two failures before success

    def test_send_max_retries_exceeded(self, producer, mock_kafka_producer):
        """Test that max retries are respected"""
        # Setup mock to always fail
        future = MagicMock()
        future.get.side_effect = KafkaTimeoutError("Timeout")
        mock_kafka_producer.send.return_value = future

        # Should raise after max retries
        with patch("time.sleep"):
            with pytest.raises(KafkaTimeoutError):
                producer.send(b"Test message")

        assert mock_kafka_producer.send.call_count == 3
        assert producer._errors == 3

    def test_send_with_kafka_error(self, producer, mock_kafka_producer):
        """Test handling of KafkaError"""
        # Setup mock to fail
        future = MagicMock()
        future.get.side_effect = KafkaError("Connection failed")
        mock_kafka_producer.send.return_value = future

        # Should raise after retries
        with patch("time.sleep"):
            with pytest.raises(KafkaError):
                producer.send(b"Test message")

        assert producer._errors == 3

    def test_encrypt_message(self, producer):
        """Test message encryption"""
        plaintext = b"Secret message"

        ciphertext, nonce, tag = producer._encrypt_message(plaintext)

        # Verify outputs
        assert isinstance(ciphertext, bytes)
        assert isinstance(nonce, bytes)
        assert isinstance(tag, bytes)
        assert len(nonce) == 12  # GCM nonce size
        assert len(tag) == 16  # GCM tag size
        assert ciphertext != plaintext  # Should be encrypted

    def test_encrypt_message_with_cryptography_library(self, producer):
        """Test encryption using cryptography library"""
        with patch("ai_engine.cloud_native.event_streaming.kafka.secure_producer.AESGCM") as mock_aesgcm:
            # Setup mock
            mock_cipher = MagicMock()
            mock_cipher.encrypt.return_value = b"encrypted_data_with_tag" + b"\x00" * 16
            mock_aesgcm.return_value = mock_cipher

            plaintext = b"Secret message"
            ciphertext, nonce, tag = producer._encrypt_message(plaintext)

            # Verify AESGCM was used
            mock_aesgcm.assert_called_once()
            mock_cipher.encrypt.assert_called_once()

    def test_flush(self, producer, mock_kafka_producer):
        """Test flushing pending messages"""
        producer.flush(timeout=30)

        mock_kafka_producer.flush.assert_called_once_with(timeout=30)

    def test_close(self, producer, mock_kafka_producer):
        """Test producer cleanup"""
        assert producer._producer is not None

        producer.close()

        mock_kafka_producer.close.assert_called_once()
        assert producer._producer is None

    def test_get_metrics(self, producer):
        """Test metrics retrieval"""
        # Simulate some activity
        producer._messages_sent = 100
        producer._bytes_sent = 5000
        producer._errors = 5

        metrics = producer.get_metrics()

        assert metrics["topic"] == "test-topic"
        assert metrics["bootstrap_servers"] == "localhost:9092"
        assert metrics["encryption_enabled"] is True
        assert metrics["algorithm"] == "aes-256-gcm"
        assert metrics["messages_sent"] == 100
        assert metrics["bytes_sent"] == 5000
        assert metrics["errors"] == 5
        assert metrics["success_rate"] == pytest.approx(100 / 105, 0.001)

    def test_get_metrics_no_activity(self, producer):
        """Test metrics with no activity"""
        metrics = producer.get_metrics()

        assert metrics["messages_sent"] == 0
        assert metrics["bytes_sent"] == 0
        assert metrics["errors"] == 0
        assert metrics["success_rate"] == 0.0

    def test_create_connector_config(self, producer):
        """Test Kafka Connect configuration generation"""
        config = producer.create_connector_config()

        assert config["name"] == "qbitel-secure-sink"
        assert "config" in config
        assert config["config"]["topics"] == "test-topic"
        assert config["config"]["bootstrap.servers"] == "localhost:9092"

    def test_multiple_servers(self, mock_kafka_producer):
        """Test initialization with multiple bootstrap servers"""
        producer = SecureKafkaProducer(bootstrap_servers="server1:9092,server2:9092,server3:9092", topic="test-topic")

        # Verify servers were parsed correctly
        call_args = mock_kafka_producer.call_args[1]
        assert call_args["bootstrap_servers"] == ["server1:9092", "server2:9092", "server3:9092"]

    def test_encryption_consistency(self, producer):
        """Test that encryption produces consistent format"""
        plaintext = b"Test message for encryption"

        # Encrypt twice
        ciphertext1, nonce1, tag1 = producer._encrypt_message(plaintext)
        ciphertext2, nonce2, tag2 = producer._encrypt_message(plaintext)

        # Nonces should be different (random)
        assert nonce1 != nonce2

        # Ciphertexts should be different (due to different nonces)
        assert ciphertext1 != ciphertext2

        # But lengths should be consistent
        assert len(ciphertext1) == len(ciphertext2)
        assert len(tag1) == len(tag2) == 16

    @pytest.mark.parametrize("message_size", [10, 100, 1000, 10000])
    def test_various_message_sizes(self, producer, mock_kafka_producer, message_size):
        """Test sending messages of various sizes"""
        # Setup mock
        future = MagicMock()
        record_metadata = MagicMock()
        record_metadata.topic = "test-topic"
        record_metadata.partition = 0
        record_metadata.offset = 123
        record_metadata.timestamp = 1234567890
        record_metadata.serialized_key_size = 0
        record_metadata.serialized_value_size = message_size
        future.get.return_value = record_metadata
        mock_kafka_producer.send.return_value = future

        # Send message
        message = b"x" * message_size
        result = producer.send(message)

        assert result["success"] is True
        assert result["original_size_bytes"] == message_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
