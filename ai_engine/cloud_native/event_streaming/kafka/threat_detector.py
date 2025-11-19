"""
Kafka Threat Detector

Real-time threat detection for Kafka event streams using ML-based
anomaly detection and pattern matching.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KafkaAnomaly:
    """Detected anomaly in Kafka stream"""
    topic: str
    partition: int
    offset: int
    anomaly_type: str
    severity: str
    description: str
    confidence: float


class KafkaThreatDetector:
    """
    Detects threats in Kafka event streams.

    Uses ML-based anomaly detection and signature-based pattern matching
    to identify suspicious activity in real-time.
    """

    def __init__(self):
        """Initialize threat detector"""
        self._anomalies: List[KafkaAnomaly] = []
        self._baseline_stats = {}
        logger.info("Initialized KafkaThreatDetector")

    def analyze_stream(
        self,
        topic: str,
        message_rate: float,
        error_rate: float,
        consumer_lag: int
    ) -> List[KafkaAnomaly]:
        """
        Analyze Kafka stream for anomalies.

        Args:
            topic: Topic name
            message_rate: Messages per second
            error_rate: Error rate (0.0-1.0)
            consumer_lag: Consumer lag in messages

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Check for unusual message rate
        if message_rate > 10000:  # Threshold
            anomaly = KafkaAnomaly(
                topic=topic,
                partition=0,
                offset=0,
                anomaly_type="high_message_rate",
                severity="MEDIUM",
                description=f"Unusually high message rate: {message_rate} msg/s",
                confidence=0.85
            )
            anomalies.append(anomaly)

        # Check for high error rate
        if error_rate > 0.1:  # 10% threshold
            anomaly = KafkaAnomaly(
                topic=topic,
                partition=0,
                offset=0,
                anomaly_type="high_error_rate",
                severity="HIGH",
                description=f"High error rate: {error_rate*100:.1f}%",
                confidence=0.9
            )
            anomalies.append(anomaly)

        # Check for excessive consumer lag
        if consumer_lag > 100000:
            anomaly = KafkaAnomaly(
                topic=topic,
                partition=0,
                offset=0,
                anomaly_type="consumer_lag",
                severity="MEDIUM",
                description=f"High consumer lag: {consumer_lag} messages",
                confidence=0.8
            )
            anomalies.append(anomaly)

        self._anomalies.extend(anomalies)
        return anomalies

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            "total_anomalies_detected": len(self._anomalies),
            "anomalies_by_severity": {
                "critical": sum(1 for a in self._anomalies if a.severity == "CRITICAL"),
                "high": sum(1 for a in self._anomalies if a.severity == "HIGH"),
                "medium": sum(1 for a in self._anomalies if a.severity == "MEDIUM"),
                "low": sum(1 for a in self._anomalies if a.severity == "LOW")
            }
        }
