"""
Integration tests for Kafka Streaming with Security.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.event_streaming.kafka.secure_producer import SecureKafkaProducer
from cloud_native.event_streaming.kafka.threat_detector import KafkaThreatDetector


class TestKafkaStreaming:
    """Test Kafka streaming integration"""

    @patch("kafka.KafkaProducer")
    def test_secure_streaming_with_threat_detection(self, mock_kafka):
        """Test secure message production with threat detection"""
        mock_producer = MagicMock()
        mock_kafka.return_value = mock_producer

        # Create secure producer
        producer = SecureKafkaProducer(
            bootstrap_servers=["localhost:9092"], topic="security-events", enable_quantum_encryption=True
        )

        # Send security events
        events = [
            {"event": "container_started", "severity": "low"},
            {"event": "unauthorized_access", "severity": "high"},
            {"event": "quantum_vulnerability_detected", "severity": "critical"},
        ]

        for event in events:
            producer.send(value=event)

        # Analyze stream for threats
        detector = KafkaThreatDetector()
        anomalies = detector.analyze_stream(
            topic="security-events", message_rate=15000, error_rate=0.01, consumer_lag=50  # High rate
        )

        assert mock_producer.send.call_count == 3
        assert isinstance(anomalies, list)
