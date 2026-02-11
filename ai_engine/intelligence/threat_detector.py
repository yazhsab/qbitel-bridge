"""
Threat Detector

Real-time threat detection and anomaly analysis for banking protocols.
Uses pattern matching and statistical analysis to identify threats.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional
import hashlib
import statistics
import uuid


logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AnomalyType(Enum):
    """Types of detected anomalies."""

    VOLUME_SPIKE = "volume_spike"
    UNUSUAL_PATTERN = "unusual_pattern"
    PROTOCOL_VIOLATION = "protocol_violation"
    TIMING_ANOMALY = "timing_anomaly"
    VALUE_ANOMALY = "value_anomaly"
    SEQUENCE_ANOMALY = "sequence_anomaly"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    REPLAY_ATTACK = "replay_attack"
    INJECTION_ATTEMPT = "injection_attempt"


@dataclass
class ThreatAlert:
    """A detected threat alert."""

    alert_id: str
    threat_level: ThreatLevel
    anomaly_type: AnomalyType
    title: str
    description: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    auto_mitigated: bool = False
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "threat_level": self.threat_level.value,
            "anomaly_type": self.anomaly_type.value,
            "title": self.title,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "source_id": self.source_id,
            "target_id": self.target_id,
            "indicators": self.indicators,
            "auto_mitigated": self.auto_mitigated,
        }


@dataclass
class ThreatRule:
    """Rule for threat detection."""

    rule_id: str
    name: str
    description: str
    anomaly_type: AnomalyType
    threat_level: ThreatLevel
    condition: Callable[[Dict[str, Any]], bool]
    enabled: bool = True
    cooldown_seconds: int = 60
    last_triggered: Optional[datetime] = None


@dataclass
class BaselineMetrics:
    """Baseline metrics for anomaly detection."""

    metric_name: str
    mean: float = 0.0
    std_dev: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    sample_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ThreatDetector:
    """
    Real-time threat detection engine.

    Capabilities:
    - Pattern-based threat detection
    - Statistical anomaly detection
    - Protocol violation detection
    - Replay attack detection
    - Rate limiting and volume spike detection
    """

    def __init__(
        self,
        baseline_window_size: int = 1000,
        alert_callback: Optional[Callable[[ThreatAlert], None]] = None,
    ):
        self._rules: Dict[str, ThreatRule] = {}
        self._baselines: Dict[str, BaselineMetrics] = {}
        self._message_history: Deque[Dict[str, Any]] = deque(maxlen=baseline_window_size)
        self._seen_hashes: Dict[str, datetime] = {}
        self._rate_counters: Dict[str, Deque[datetime]] = {}
        self._alert_callback = alert_callback

        # Initialize default rules
        self._initialize_default_rules()

    def analyze(
        self,
        message: Dict[str, Any],
        protocol_type: str = "unknown",
    ) -> List[ThreatAlert]:
        """
        Analyze a message for threats.

        Args:
            message: Message data to analyze
            protocol_type: Type of protocol

        Returns:
            List of ThreatAlert for detected threats
        """
        alerts = []

        # Add protocol type to message context
        message["_protocol_type"] = protocol_type
        message["_analyzed_at"] = datetime.utcnow()

        # Check replay attack
        replay_alert = self._check_replay_attack(message)
        if replay_alert:
            alerts.append(replay_alert)

        # Check rate limiting
        rate_alert = self._check_rate_limits(message)
        if rate_alert:
            alerts.append(rate_alert)

        # Check protocol-specific rules
        for rule in self._rules.values():
            if rule.enabled:
                try:
                    if self._should_trigger_rule(rule, message):
                        alert = self._create_alert_from_rule(rule, message)
                        alerts.append(alert)
                except Exception as e:
                    logger.warning(f"Rule {rule.rule_id} evaluation failed: {e}")

        # Check statistical anomalies
        anomaly_alerts = self._check_statistical_anomalies(message)
        alerts.extend(anomaly_alerts)

        # Update baselines with this message
        self._update_baselines(message)

        # Store in history
        self._message_history.append(message)

        # Notify callback
        for alert in alerts:
            if self._alert_callback:
                try:
                    self._alert_callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

        return alerts

    def add_rule(self, rule: ThreatRule) -> None:
        """Add a detection rule."""
        self._rules[rule.rule_id] = rule
        logger.info(f"Added threat detection rule: {rule.name}")

    def remove_rule(self, rule_id: str) -> None:
        """Remove a detection rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]

    def get_alerts_summary(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        # In production, this would query stored alerts
        return {
            "since": since.isoformat() if since else None,
            "active_rules": len([r for r in self._rules.values() if r.enabled]),
            "baselines_tracked": len(self._baselines),
        }

    def set_baseline(
        self,
        metric_name: str,
        baseline: BaselineMetrics,
    ) -> None:
        """Set a baseline metric for anomaly detection."""
        self._baselines[metric_name] = baseline

    def train_baseline(
        self,
        metric_name: str,
        values: List[float],
    ) -> BaselineMetrics:
        """Train a baseline from historical data."""
        if not values:
            raise ValueError("Cannot train baseline with empty data")

        baseline = BaselineMetrics(
            metric_name=metric_name,
            mean=statistics.mean(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0,
            min_value=min(values),
            max_value=max(values),
            sample_count=len(values),
        )

        self._baselines[metric_name] = baseline
        return baseline

    # =========================================================================
    # Detection Methods
    # =========================================================================

    def _check_replay_attack(
        self,
        message: Dict[str, Any],
    ) -> Optional[ThreatAlert]:
        """Check for replay attack."""
        # Create message hash
        msg_hash = self._hash_message(message)

        # Check if seen before
        if msg_hash in self._seen_hashes:
            original_time = self._seen_hashes[msg_hash]
            time_diff = (datetime.utcnow() - original_time).total_seconds()

            # If seen within 24 hours, flag as potential replay
            if time_diff < 86400:
                return ThreatAlert(
                    alert_id=str(uuid.uuid4()),
                    threat_level=ThreatLevel.HIGH,
                    anomaly_type=AnomalyType.REPLAY_ATTACK,
                    title="Potential Replay Attack Detected",
                    description=f"Duplicate message detected, original seen {time_diff:.0f} seconds ago",
                    evidence=[
                        {"message_hash": msg_hash},
                        {"original_time": original_time.isoformat()},
                    ],
                    indicators=["duplicate_message", "timing_suspicion"],
                    recommended_actions=[
                        "Verify message authenticity",
                        "Check for compromised credentials",
                        "Review transaction logs",
                    ],
                )

        # Store hash
        self._seen_hashes[msg_hash] = datetime.utcnow()

        # Cleanup old hashes (older than 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self._seen_hashes = {
            h: t for h, t in self._seen_hashes.items() if t > cutoff
        }

        return None

    def _check_rate_limits(
        self,
        message: Dict[str, Any],
    ) -> Optional[ThreatAlert]:
        """Check rate limits for volume spike detection."""
        source_id = message.get("source_id") or message.get("sender") or "unknown"

        # Initialize counter if needed
        if source_id not in self._rate_counters:
            self._rate_counters[source_id] = deque(maxlen=1000)

        # Add current timestamp
        self._rate_counters[source_id].append(datetime.utcnow())

        # Count messages in last minute
        one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
        recent_count = sum(
            1 for t in self._rate_counters[source_id] if t > one_minute_ago
        )

        # Check against baseline or default threshold
        threshold = 100  # Default threshold

        if "message_rate" in self._baselines:
            baseline = self._baselines["message_rate"]
            # Alert if > 3 standard deviations above mean
            threshold = baseline.mean + (3 * baseline.std_dev)

        if recent_count > threshold:
            return ThreatAlert(
                alert_id=str(uuid.uuid4()),
                threat_level=ThreatLevel.MEDIUM,
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                title="Volume Spike Detected",
                description=f"Source {source_id} sent {recent_count} messages in last minute (threshold: {threshold:.0f})",
                source_id=source_id,
                evidence=[
                    {"message_count": recent_count},
                    {"threshold": threshold},
                ],
                indicators=["high_volume", "potential_dos"],
                recommended_actions=[
                    "Monitor source activity",
                    "Consider rate limiting",
                    "Verify legitimate traffic",
                ],
            )

        return None

    def _check_statistical_anomalies(
        self,
        message: Dict[str, Any],
    ) -> List[ThreatAlert]:
        """Check for statistical anomalies."""
        alerts = []

        # Check amount anomalies
        amount = message.get("amount") or message.get("value")
        if amount is not None and "amount" in self._baselines:
            try:
                amount_float = float(amount)
                baseline = self._baselines["amount"]

                # Z-score calculation
                if baseline.std_dev > 0:
                    z_score = abs(amount_float - baseline.mean) / baseline.std_dev

                    if z_score > 4:  # 4 standard deviations
                        alerts.append(ThreatAlert(
                            alert_id=str(uuid.uuid4()),
                            threat_level=ThreatLevel.MEDIUM,
                            anomaly_type=AnomalyType.VALUE_ANOMALY,
                            title="Unusual Transaction Amount",
                            description=f"Amount {amount} is {z_score:.1f} standard deviations from mean",
                            evidence=[
                                {"amount": amount_float},
                                {"baseline_mean": baseline.mean},
                                {"z_score": z_score},
                            ],
                            recommended_actions=[
                                "Verify transaction legitimacy",
                                "Check for fraud indicators",
                            ],
                        ))
            except (ValueError, TypeError):
                pass

        # Check timing anomalies
        if "timing" in self._baselines:
            # Analyze time between messages
            if len(self._message_history) > 1:
                prev_time = self._message_history[-1].get("_analyzed_at")
                curr_time = message.get("_analyzed_at")

                if prev_time and curr_time:
                    interval = (curr_time - prev_time).total_seconds()
                    baseline = self._baselines["timing"]

                    if baseline.std_dev > 0:
                        z_score = abs(interval - baseline.mean) / baseline.std_dev

                        if z_score > 4:
                            alerts.append(ThreatAlert(
                                alert_id=str(uuid.uuid4()),
                                threat_level=ThreatLevel.LOW,
                                anomaly_type=AnomalyType.TIMING_ANOMALY,
                                title="Unusual Message Timing",
                                description=f"Message interval {interval:.2f}s is unusual",
                                evidence=[{"interval": interval}],
                            ))

        return alerts

    def _should_trigger_rule(
        self,
        rule: ThreatRule,
        message: Dict[str, Any],
    ) -> bool:
        """Check if a rule should trigger."""
        # Check cooldown
        if rule.last_triggered:
            elapsed = (datetime.utcnow() - rule.last_triggered).total_seconds()
            if elapsed < rule.cooldown_seconds:
                return False

        # Evaluate condition
        if rule.condition(message):
            rule.last_triggered = datetime.utcnow()
            return True

        return False

    def _create_alert_from_rule(
        self,
        rule: ThreatRule,
        message: Dict[str, Any],
    ) -> ThreatAlert:
        """Create alert from triggered rule."""
        return ThreatAlert(
            alert_id=str(uuid.uuid4()),
            threat_level=rule.threat_level,
            anomaly_type=rule.anomaly_type,
            title=rule.name,
            description=rule.description,
            source_id=message.get("source_id"),
            target_id=message.get("target_id"),
            evidence=[{"triggered_by_rule": rule.rule_id}],
        )

    def _hash_message(self, message: Dict[str, Any]) -> str:
        """Create hash of message for deduplication."""
        # Exclude timestamp fields for comparison
        hashable = {
            k: v for k, v in message.items()
            if not k.startswith("_") and k not in ("timestamp", "created_at")
        }

        hash_input = str(sorted(hashable.items())).encode()
        return hashlib.sha256(hash_input).hexdigest()[:32]

    def _update_baselines(self, message: Dict[str, Any]) -> None:
        """Update baselines with new message data."""
        # Update amount baseline
        amount = message.get("amount") or message.get("value")
        if amount is not None:
            try:
                amount_float = float(amount)
                self._update_baseline_metric("amount", amount_float)
            except (ValueError, TypeError):
                pass

    def _update_baseline_metric(
        self,
        metric_name: str,
        value: float,
    ) -> None:
        """Update a baseline metric with a new value using online algorithm."""
        if metric_name not in self._baselines:
            self._baselines[metric_name] = BaselineMetrics(metric_name=metric_name)

        baseline = self._baselines[metric_name]
        baseline.sample_count += 1
        n = baseline.sample_count

        # Welford's online algorithm for mean and variance
        delta = value - baseline.mean
        baseline.mean += delta / n

        if n > 1:
            delta2 = value - baseline.mean
            # Update variance (M2 / (n-1) = variance)
            m2 = (baseline.std_dev ** 2) * (n - 2) if n > 2 else 0
            m2 += delta * delta2
            baseline.std_dev = (m2 / (n - 1)) ** 0.5 if n > 1 else 0

        baseline.min_value = min(baseline.min_value, value) if baseline.sample_count > 1 else value
        baseline.max_value = max(baseline.max_value, value) if baseline.sample_count > 1 else value
        baseline.last_updated = datetime.utcnow()

    # =========================================================================
    # Default Rules
    # =========================================================================

    def _initialize_default_rules(self) -> None:
        """Initialize default detection rules."""
        # Large transaction rule
        self.add_rule(ThreatRule(
            rule_id="large_transaction",
            name="Large Transaction Detected",
            description="Transaction amount exceeds threshold",
            anomaly_type=AnomalyType.VALUE_ANOMALY,
            threat_level=ThreatLevel.MEDIUM,
            condition=lambda m: float(m.get("amount", 0) or 0) > 1000000,
        ))

        # Unusual hours rule
        self.add_rule(ThreatRule(
            rule_id="unusual_hours",
            name="Off-Hours Activity",
            description="Transaction during unusual hours (11PM-5AM)",
            anomaly_type=AnomalyType.TIMING_ANOMALY,
            threat_level=ThreatLevel.LOW,
            condition=lambda m: datetime.utcnow().hour in (23, 0, 1, 2, 3, 4),
        ))

        # Protocol violation rule
        self.add_rule(ThreatRule(
            rule_id="missing_signature",
            name="Missing Digital Signature",
            description="Message lacks required digital signature",
            anomaly_type=AnomalyType.PROTOCOL_VIOLATION,
            threat_level=ThreatLevel.HIGH,
            condition=lambda m: m.get("requires_signature") and not m.get("signature"),
        ))

        # Injection attempt rule
        self.add_rule(ThreatRule(
            rule_id="injection_chars",
            name="Potential Injection Attempt",
            description="Suspicious characters detected in input",
            anomaly_type=AnomalyType.INJECTION_ATTEMPT,
            threat_level=ThreatLevel.HIGH,
            condition=lambda m: any(
                c in str(m.get("payload", ""))
                for c in ["<script>", "DROP TABLE", "'; --", "UNION SELECT"]
            ),
        ))

        # Cross-border high value
        self.add_rule(ThreatRule(
            rule_id="cross_border_high_value",
            name="High-Value Cross-Border Transaction",
            description="Large transaction crossing borders",
            anomaly_type=AnomalyType.BEHAVIORAL_ANOMALY,
            threat_level=ThreatLevel.MEDIUM,
            condition=lambda m: (
                float(m.get("amount", 0) or 0) > 100000 and
                m.get("source_country") != m.get("target_country")
            ),
        ))
