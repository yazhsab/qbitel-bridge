"""
Call Quality Monitoring with PQC Overhead Tracking

Real-time voice quality monitoring for BPO contact centers with
post-quantum cryptography (PQC) overhead awareness.

Implements:
- ITU-T G.107 E-model MOS calculation
- Real-time jitter, packet loss, latency monitoring
- PQC cryptographic overhead tracking and budgeting
- Degradation detection with root-cause analysis
- Automatic remediation (codec switching, PQC level reduction)
- Quality trend analysis with sliding window
- Comprehensive quality reporting

Voice quality is the single most important KPI for BPO operations.
PQC encryption (ML-KEM, ML-DSA) adds measurable overhead to each
RTP packet, and this module ensures that cryptographic protection
does not degrade the caller experience below acceptable thresholds.

Reference: ITU-T G.107 (E-model), ITU-T P.800 (MOS), RFC 3550 (RTP)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import logging
import re
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QualityMetricType(Enum):
    """Types of voice quality metrics collected."""

    MOS_SCORE = auto()             # Mean Opinion Score (1.0 - 4.5)
    JITTER_MS = auto()             # Inter-packet arrival variance
    PACKET_LOSS_PCT = auto()       # Packet loss percentage
    LATENCY_MS = auto()            # One-way delay in milliseconds
    R_FACTOR = auto()              # R-factor from E-model (0 - 100)
    ECHO_RETURN_LOSS = auto()      # Echo return loss in dB
    SILENCE_RATIO = auto()         # Ratio of silence to speech
    CODEC_QUALITY = auto()         # Codec impairment factor
    PQC_OVERHEAD_MS = auto()       # PQC crypto operation overhead
    DTMF_CLARITY = auto()          # DTMF tone detection clarity


class QualityDegradationType(Enum):
    """Types of quality degradation detected."""

    NETWORK_CONGESTION = auto()
    CODEC_MISMATCH = auto()
    PQC_OVERHEAD_EXCESSIVE = auto()
    JITTER_SPIKE = auto()
    PACKET_BURST_LOSS = auto()
    ECHO_FEEDBACK = auto()
    SILENCE_SUPPRESSION_FAILURE = auto()
    DTMF_CORRUPTION = auto()
    BANDWIDTH_SATURATION = auto()
    ENCRYPTION_LATENCY = auto()


class QualitySeverity(Enum):
    """Quality severity levels mapped to MOS ranges."""

    EXCELLENT = (1, "Excellent", 4.3, 5.0)
    GOOD = (2, "Good", 4.0, 4.3)
    ACCEPTABLE = (3, "Acceptable", 3.6, 4.0)
    POOR = (4, "Poor", 3.1, 3.6)
    UNACCEPTABLE = (5, "Unacceptable", 2.6, 3.1)
    CRITICAL = (6, "Critical", 0.0, 2.6)

    def __init__(
        self,
        level: int,
        display_name: str,
        min_mos: float,
        max_mos: float,
    ):
        self.level = level
        self.display_name = display_name
        self.min_mos = min_mos
        self.max_mos = max_mos


class QualityAction(Enum):
    """Actions taken in response to quality degradation."""

    LOG = auto()                   # Log event only
    ALERT = auto()                 # Send alert to operations
    REDUCE_PQC_LEVEL = auto()      # Downgrade PQC algorithm
    SWITCH_CODEC = auto()          # Switch to a better codec
    REROUTE = auto()               # Reroute call via alternate path
    INCREASE_BUFFER = auto()       # Increase jitter buffer
    ESCALATE = auto()              # Escalate to network engineering
    TERMINATE_CALL = auto()        # Terminate critically degraded call


class MonitoringMode(Enum):
    """Quality monitoring operational modes."""

    PASSIVE = auto()               # Observe RTP stream passively
    ACTIVE_PROBE = auto()          # Inject probe packets
    HYBRID = auto()                # Passive + periodic probes


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QualityThresholds:
    """Threshold configuration for quality metric violations."""

    max_jitter_ms: float = 30.0
    max_packet_loss_pct: float = 1.0
    max_latency_ms: float = 150.0
    min_mos: float = 3.5
    max_pqc_overhead_ms: float = 2.0
    min_r_factor: float = 70.0
    max_echo_return_loss_db: float = -18.0
    max_silence_ratio: float = 0.3

    def validate(self) -> List[str]:
        """Validate threshold configuration, returning list of errors."""
        errors: List[str] = []
        if self.max_jitter_ms <= 0:
            errors.append("max_jitter_ms must be positive")
        if not (0.0 <= self.max_packet_loss_pct <= 100.0):
            errors.append("max_packet_loss_pct must be between 0 and 100")
        if self.max_latency_ms <= 0:
            errors.append("max_latency_ms must be positive")
        if not (1.0 <= self.min_mos <= 4.5):
            errors.append("min_mos must be between 1.0 and 4.5")
        if self.max_pqc_overhead_ms <= 0:
            errors.append("max_pqc_overhead_ms must be positive")
        if not (0.0 <= self.min_r_factor <= 100.0):
            errors.append("min_r_factor must be between 0 and 100")
        if self.max_silence_ratio < 0 or self.max_silence_ratio > 1.0:
            errors.append("max_silence_ratio must be between 0.0 and 1.0")
        return errors


@dataclass
class QualityMetric:
    """A single quality metric measurement."""

    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_type: QualityMetricType = QualityMetricType.MOS_SCORE
    value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    call_id: str = ""
    agent_id: str = ""
    threshold: Optional[float] = None
    is_violation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metric to dictionary."""
        return {
            "metric_id": self.metric_id,
            "metric_type": self.metric_type.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "call_id": self.call_id,
            "agent_id": self.agent_id,
            "threshold": self.threshold,
            "is_violation": self.is_violation,
        }


@dataclass
class PQCOverheadSample:
    """A single PQC cryptographic operation overhead measurement."""

    sample_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    algorithm: str = ""       # e.g., "ML-KEM-512", "ML-KEM-768"
    operation: str = ""       # e.g., "encapsulate", "decapsulate", "sign", "verify"
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    call_id: str = ""
    packet_sequence: int = 0


@dataclass
class QualityEvent:
    """A detected quality degradation event."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str = ""
    agent_id: str = ""
    tenant_id: str = ""
    degradation_type: QualityDegradationType = QualityDegradationType.NETWORK_CONGESTION
    severity: QualitySeverity = QualitySeverity.ACCEPTABLE
    metrics: Dict[str, float] = field(default_factory=dict)
    root_cause: str = ""
    action_taken: QualityAction = QualityAction.LOG
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    pqc_algorithm: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "event_id": self.event_id,
            "call_id": self.call_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "degradation_type": self.degradation_type.name,
            "severity": self.severity.display_name,
            "severity_level": self.severity.level,
            "metrics": dict(self.metrics),
            "root_cause": self.root_cause,
            "action_taken": self.action_taken.name,
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "pqc_algorithm": self.pqc_algorithm,
            "evidence": dict(self.evidence),
        }


@dataclass
class QualityPolicy:
    """Quality monitoring policy configuration."""

    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default"
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    pqc_overhead_budget_ms: float = 2.0
    monitoring_mode: MonitoringMode = MonitoringMode.PASSIVE
    auto_remediate: bool = True
    alert_cooldown_seconds: int = 60
    trend_window_size: int = 100
    mos_calculation_model: str = "e_model_g107"

    def validate(self) -> List[str]:
        """Validate policy configuration, returning list of errors."""
        errors: List[str] = []
        errors.extend(self.thresholds.validate())
        if self.pqc_overhead_budget_ms <= 0:
            errors.append("pqc_overhead_budget_ms must be positive")
        if self.alert_cooldown_seconds < 0:
            errors.append("alert_cooldown_seconds must be non-negative")
        if self.trend_window_size < 2:
            errors.append("trend_window_size must be at least 2")
        valid_models = {"e_model_g107", "e_model_g107_extended", "pesq_estimate"}
        if self.mos_calculation_model not in valid_models:
            errors.append(
                f"mos_calculation_model must be one of {sorted(valid_models)}"
            )
        return errors


@dataclass
class QualityReport:
    """Aggregated quality report for a time period."""

    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    total_calls_monitored: int = 0
    average_mos: float = 0.0
    min_mos: float = 0.0
    max_mos: float = 0.0
    total_degradation_events: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_severity: Dict[str, int] = field(default_factory=dict)
    pqc_overhead_avg_ms: float = 0.0
    pqc_overhead_p95_ms: float = 0.0
    pqc_overhead_max_ms: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "report_id": self.report_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_calls_monitored": self.total_calls_monitored,
            "average_mos": round(self.average_mos, 2),
            "min_mos": round(self.min_mos, 2),
            "max_mos": round(self.max_mos, 2),
            "total_degradation_events": self.total_degradation_events,
            "events_by_type": dict(self.events_by_type),
            "events_by_severity": dict(self.events_by_severity),
            "pqc_overhead_avg_ms": round(self.pqc_overhead_avg_ms, 3),
            "pqc_overhead_p95_ms": round(self.pqc_overhead_p95_ms, 3),
            "pqc_overhead_max_ms": round(self.pqc_overhead_max_ms, 3),
            "recommendations": list(self.recommendations),
        }


# ---------------------------------------------------------------------------
# Engine classes
# ---------------------------------------------------------------------------


class MOSCalculator:
    """
    ITU-T G.107 E-model simplified MOS calculation.

    Computes the R-factor and converts to MOS using the standard formulas
    from ITU-T Recommendation G.107. This implementation includes a
    PQC overhead term that adds to the total one-way delay, allowing
    accurate quality prediction for quantum-safe encrypted voice streams.

    Key parameters:
    - Id: delay impairment factor
    - Ie_eff: equipment impairment factor (codec-dependent)
    - Is: simultaneous impairment factor (set to 0 for simplicity)
    - A: advantage factor (set to 0 for landline-grade BPO)
    """

    # Equipment impairment factors (Ie) per ITU-T G.113 Appendix I
    CODEC_IE: Dict[str, float] = {
        "G.711": 0.0,
        "G.729": 10.0,
        "G.722": 0.0,
        "G.726": 2.0,
        "OPUS": 0.0,
        "PCMU": 0.0,
        "PCMA": 0.0,
    }

    # Packet-loss robustness factor (Bpl) per codec
    CODEC_BPL: Dict[str, float] = {
        "G.711": 25.1,
        "G.729": 19.0,
        "G.722": 25.1,
        "G.726": 23.0,
        "OPUS": 30.0,
        "PCMU": 25.1,
        "PCMA": 25.1,
    }

    @staticmethod
    def _heaviside(x: float) -> float:
        """Heaviside step function: H(x) = 1 if x >= 0, else 0."""
        return 1.0 if x >= 0 else 0.0

    @classmethod
    def calculate_r_factor(
        cls,
        latency_ms: float,
        jitter_ms: float,
        packet_loss_pct: float,
        codec: str = "G.711",
        pqc_overhead_ms: float = 0.0,
    ) -> float:
        """
        Calculate the R-factor using the ITU-T G.107 E-model.

        R = R0 - Is - Id - Ie_eff + A

        Where:
        - R0 = 93.2 (base signal-to-noise ratio)
        - Is = 0 (simultaneous impairment, simplified)
        - Id = delay impairment
        - Ie_eff = effective equipment impairment (codec + packet loss)
        - A = 0 (advantage factor for BPO wired connections)

        Args:
            latency_ms: One-way network delay in milliseconds.
            jitter_ms: Inter-packet jitter in milliseconds.
            packet_loss_pct: Packet loss as a percentage (0-100).
            codec: Codec name (e.g., "G.711", "G.729").
            pqc_overhead_ms: Additional delay from PQC operations.

        Returns:
            R-factor value clamped to [0, 100].
        """
        r0 = 93.2

        # Total one-way delay includes network latency, half of jitter
        # (as an effective delay), and PQC cryptographic overhead
        total_delay = latency_ms + (jitter_ms / 2.0) + pqc_overhead_ms

        # Delay impairment factor (Id)
        # Id = 0.024 * d + 0.11 * (d - 177.3) * H(d - 177.3)
        id_factor = (
            0.024 * total_delay
            + 0.11 * (total_delay - 177.3) * cls._heaviside(total_delay - 177.3)
        )

        # Equipment impairment factor (Ie_eff)
        # Ie_eff = Ie + (95 - Ie) * Ppl / (Ppl + Bpl)
        ie = cls.CODEC_IE.get(codec, 0.0)
        bpl = cls.CODEC_BPL.get(codec, 25.1)
        ppl = packet_loss_pct  # Packet loss probability as percentage
        if ppl + bpl > 0:
            ie_eff = ie + (95.0 - ie) * ppl / (ppl + bpl)
        else:
            ie_eff = ie

        # R = R0 - Is - Id - Ie_eff + A
        # Is = 0, A = 0 for BPO wired environment
        r_factor = r0 - id_factor - ie_eff

        # Clamp to valid range
        return max(0.0, min(100.0, r_factor))

    @staticmethod
    def r_to_mos(r_factor: float) -> float:
        """
        Convert R-factor to Mean Opinion Score (MOS).

        For R <= 0:  MOS = 1.0
        For R >= 100: MOS = 4.5
        Otherwise:
            MOS = 1 + 0.035 * R + R * (R - 60) * (100 - R) * 7e-6

        Args:
            r_factor: The computed R-factor (0-100).

        Returns:
            MOS value in the range [1.0, 4.5].
        """
        if r_factor <= 0:
            return 1.0
        if r_factor >= 100:
            return 4.5
        mos = 1.0 + 0.035 * r_factor + r_factor * (r_factor - 60.0) * (100.0 - r_factor) * 7.0e-6
        return max(1.0, min(4.5, mos))

    @classmethod
    def calculate_mos(
        cls,
        latency_ms: float,
        jitter_ms: float,
        packet_loss_pct: float,
        codec: str = "G.711",
        pqc_overhead_ms: float = 0.0,
    ) -> float:
        """
        Calculate MOS from raw network metrics.

        Convenience method combining R-factor calculation and MOS conversion.

        Args:
            latency_ms: One-way network delay in milliseconds.
            jitter_ms: Inter-packet jitter in milliseconds.
            packet_loss_pct: Packet loss as a percentage (0-100).
            codec: Codec name (e.g., "G.711", "G.729").
            pqc_overhead_ms: Additional delay from PQC operations.

        Returns:
            MOS value in the range [1.0, 4.5].
        """
        r = cls.calculate_r_factor(latency_ms, jitter_ms, packet_loss_pct, codec, pqc_overhead_ms)
        return cls.r_to_mos(r)

    @classmethod
    def severity_from_mos(cls, mos: float) -> QualitySeverity:
        """
        Determine quality severity level from a MOS score.

        Args:
            mos: Mean Opinion Score value.

        Returns:
            Corresponding QualitySeverity enum member.
        """
        for severity in QualitySeverity:
            if severity.min_mos <= mos < severity.max_mos:
                return severity
        # Edge case: MOS exactly 5.0 maps to EXCELLENT
        if mos >= QualitySeverity.EXCELLENT.min_mos:
            return QualitySeverity.EXCELLENT
        return QualitySeverity.CRITICAL


class PQCOverheadTracker:
    """
    Tracks PQC cryptographic operation overhead impact on voice quality.

    Maintains a sliding window of overhead samples and provides statistics
    for budget compliance checking. When overhead exceeds the configured
    budget, the tracker recommends lighter PQC algorithms.
    """

    # Typical overhead per algorithm (ms) for recommendation engine
    ALGORITHM_BENCHMARKS: Dict[str, float] = {
        "ML-KEM-512": 0.8,
        "ML-KEM-768": 1.5,
        "ML-KEM-1024": 2.8,
        "ML-DSA-44": 1.2,
        "ML-DSA-65": 2.0,
        "ML-DSA-87": 3.5,
    }

    def __init__(self, budget_ms: float = 2.0, window_size: int = 1000):
        self._budget_ms = budget_ms
        self._samples: List[PQCOverheadSample] = []
        self._window_size = window_size
        self._algorithm_stats: Dict[str, Dict[str, float]] = {}

    def record_sample(self, sample: PQCOverheadSample) -> None:
        """
        Record a PQC operation overhead sample.

        Maintains the sliding window at max size and updates per-algorithm
        running statistics.

        Args:
            sample: The overhead measurement to record.
        """
        self._samples.append(sample)

        # Maintain sliding window
        if len(self._samples) > self._window_size:
            self._samples = self._samples[-self._window_size:]

        # Update per-algorithm statistics
        algo = sample.algorithm
        if algo:
            stats = self._algorithm_stats.get(algo, {
                "count": 0.0,
                "total_ms": 0.0,
                "max_ms": 0.0,
            })
            stats["count"] += 1.0
            stats["total_ms"] += sample.duration_ms
            if sample.duration_ms > stats["max_ms"]:
                stats["max_ms"] = sample.duration_ms
            self._algorithm_stats[algo] = stats

    def get_average_overhead(self, algorithm: Optional[str] = None) -> float:
        """
        Calculate average overhead across all samples or for a specific algorithm.

        Args:
            algorithm: Optional algorithm name to filter by.

        Returns:
            Average overhead in milliseconds, or 0.0 if no samples.
        """
        if algorithm and algorithm in self._algorithm_stats:
            stats = self._algorithm_stats[algorithm]
            if stats["count"] > 0:
                return stats["total_ms"] / stats["count"]
            return 0.0

        if not self._samples:
            return 0.0

        if algorithm:
            filtered = [s for s in self._samples if s.algorithm == algorithm]
            if not filtered:
                return 0.0
            return sum(s.duration_ms for s in filtered) / len(filtered)

        return sum(s.duration_ms for s in self._samples) / len(self._samples)

    def get_p95_overhead(self, algorithm: Optional[str] = None) -> float:
        """
        Calculate the 95th percentile overhead.

        Args:
            algorithm: Optional algorithm name to filter by.

        Returns:
            95th percentile overhead in milliseconds, or 0.0 if no samples.
        """
        samples = self._samples
        if algorithm:
            samples = [s for s in samples if s.algorithm == algorithm]

        if not samples:
            return 0.0

        durations = sorted(s.duration_ms for s in samples)
        idx = int(len(durations) * 0.95)
        idx = min(idx, len(durations) - 1)
        return durations[idx]

    def get_max_overhead(self, algorithm: Optional[str] = None) -> float:
        """
        Get the maximum recorded overhead.

        Args:
            algorithm: Optional algorithm name to filter by.

        Returns:
            Maximum overhead in milliseconds, or 0.0 if no samples.
        """
        samples = self._samples
        if algorithm:
            samples = [s for s in samples if s.algorithm == algorithm]

        if not samples:
            return 0.0

        return max(s.duration_ms for s in samples)

    def is_within_budget(self) -> bool:
        """
        Check if average overhead is within the configured budget.

        Returns:
            True if average overhead <= budget, False otherwise.
        """
        return self.get_average_overhead() <= self._budget_ms

    def get_recommendation(self) -> Optional[str]:
        """
        Generate a recommendation if overhead exceeds budget.

        Analyzes per-algorithm overhead and suggests lighter alternatives
        that would bring the system within budget.

        Returns:
            Recommendation string, or None if within budget.
        """
        avg = self.get_average_overhead()
        if avg <= self._budget_ms:
            return None

        # Find the algorithm contributing most overhead
        worst_algo = ""
        worst_avg = 0.0
        for algo, stats in self._algorithm_stats.items():
            if stats["count"] > 0:
                algo_avg = stats["total_ms"] / stats["count"]
                if algo_avg > worst_avg:
                    worst_avg = algo_avg
                    worst_algo = algo

        if not worst_algo:
            return f"PQC overhead ({avg:.1f}ms) exceeds budget ({self._budget_ms:.1f}ms)"

        # Suggest a lighter alternative
        lighter = self._find_lighter_algorithm(worst_algo)
        if lighter:
            lighter_bench = self.ALGORITHM_BENCHMARKS.get(lighter, 0.0)
            return (
                f"Switch from {worst_algo} to {lighter} to reduce overhead "
                f"from {worst_avg:.1f}ms to ~{lighter_bench:.1f}ms"
            )

        return (
            f"PQC algorithm {worst_algo} averaging {worst_avg:.1f}ms "
            f"exceeds budget of {self._budget_ms:.1f}ms; "
            f"consider reducing security level or increasing budget"
        )

    def _find_lighter_algorithm(self, current: str) -> Optional[str]:
        """Find a lighter algorithm in the same family."""
        # Algorithm families and their lighter alternatives
        downgrade_map: Dict[str, str] = {
            "ML-KEM-1024": "ML-KEM-768",
            "ML-KEM-768": "ML-KEM-512",
            "ML-DSA-87": "ML-DSA-65",
            "ML-DSA-65": "ML-DSA-44",
        }
        return downgrade_map.get(current)

    @property
    def sample_count(self) -> int:
        """Number of samples in the current window."""
        return len(self._samples)


class CallQualityMonitor:
    """
    Main call quality monitoring engine with PQC overhead awareness.

    Monitors real-time voice quality metrics (MOS, jitter, packet loss, latency),
    detects degradation events, and tracks PQC cryptographic overhead impact.
    Supports configurable policies with automatic remediation recommendations.

    Typical usage:
        monitor = CallQualityMonitor()
        event = monitor.analyze_rtp_stream(
            call_id="call-123", agent_id="agent-45", tenant_id="tenant-1",
            jitter_ms=15.0, packet_loss_pct=0.2, latency_ms=80.0,
            codec="G.711", pqc_overhead_ms=1.1,
        )
        if event:
            logger.warning("Quality degradation: %s", event.root_cause)
    """

    def __init__(self, policy: Optional[QualityPolicy] = None):
        self._policy = policy or QualityPolicy()
        self._mos_calculator = MOSCalculator()
        self._pqc_tracker = PQCOverheadTracker(
            budget_ms=self._policy.pqc_overhead_budget_ms,
            window_size=self._policy.trend_window_size * 10,
        )
        self._active_calls: Dict[str, List[QualityMetric]] = {}
        self._events: List[QualityEvent] = []
        self._alert_cooldowns: Dict[str, datetime] = {}

    @property
    def policy(self) -> QualityPolicy:
        """Return the active monitoring policy."""
        return self._policy

    @property
    def events(self) -> List[QualityEvent]:
        """Return all recorded quality events."""
        return list(self._events)

    def analyze_rtp_stream(
        self,
        call_id: str,
        agent_id: str,
        tenant_id: str,
        jitter_ms: float,
        packet_loss_pct: float,
        latency_ms: float,
        codec: str = "G.711",
        pqc_overhead_ms: float = 0.0,
        **kwargs: Any,
    ) -> Optional[QualityEvent]:
        """
        Analyze RTP stream metrics and detect quality degradation.

        Calculates MOS from the raw metrics, checks against policy thresholds,
        stores metrics for trend analysis, and generates a QualityEvent if
        any violation is detected.

        Args:
            call_id: Unique identifier for the call.
            agent_id: Identifier for the handling agent.
            tenant_id: Tenant / organization identifier.
            jitter_ms: Measured jitter in milliseconds.
            packet_loss_pct: Measured packet loss percentage.
            latency_ms: Measured one-way latency in milliseconds.
            codec: Active codec name.
            pqc_overhead_ms: Measured PQC overhead in milliseconds.
            **kwargs: Additional metric context (e.g., echo_return_loss_db).

        Returns:
            A QualityEvent if a violation was detected, or None.
        """
        # Calculate MOS and R-factor
        mos = self._mos_calculator.calculate_mos(
            latency_ms, jitter_ms, packet_loss_pct, codec, pqc_overhead_ms
        )
        r_factor = self._mos_calculator.calculate_r_factor(
            latency_ms, jitter_ms, packet_loss_pct, codec, pqc_overhead_ms
        )

        # Build metrics snapshot
        raw_metrics: Dict[str, float] = {
            "mos": mos,
            "r_factor": r_factor,
            "jitter_ms": jitter_ms,
            "packet_loss_pct": packet_loss_pct,
            "latency_ms": latency_ms,
            "pqc_overhead_ms": pqc_overhead_ms,
        }
        # Include any extra metrics from kwargs
        for key in ("echo_return_loss_db", "silence_ratio", "dtmf_clarity"):
            if key in kwargs:
                raw_metrics[key] = float(kwargs[key])

        # Store MOS metric for this call
        mos_metric = QualityMetric(
            metric_type=QualityMetricType.MOS_SCORE,
            value=mos,
            call_id=call_id,
            agent_id=agent_id,
            threshold=self._policy.thresholds.min_mos,
            is_violation=(mos < self._policy.thresholds.min_mos),
        )
        if call_id not in self._active_calls:
            self._active_calls[call_id] = []
        self._active_calls[call_id].append(mos_metric)

        # Trim to trend window size
        window = self._policy.trend_window_size
        if len(self._active_calls[call_id]) > window:
            self._active_calls[call_id] = self._active_calls[call_id][-window:]

        # Check thresholds
        violations = self._check_thresholds(call_id, raw_metrics)

        if not violations:
            return None

        # Use the most severe violation
        violations.sort(key=lambda v: v[1].level, reverse=True)
        degradation_type, severity = violations[0]

        # Determine root cause
        root_cause = self._determine_root_cause(degradation_type, raw_metrics)

        # Determine action
        action = self._determine_action(severity, degradation_type)

        # Check alert cooldown
        if not self._can_alert(call_id):
            action = QualityAction.LOG

        # Build event
        event = QualityEvent(
            call_id=call_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            degradation_type=degradation_type,
            severity=severity,
            metrics=raw_metrics,
            root_cause=root_cause,
            action_taken=action,
            pqc_algorithm=kwargs.get("pqc_algorithm"),
            evidence={
                "codec": codec,
                "violations": [
                    {"type": v[0].name, "severity": v[1].display_name}
                    for v in violations
                ],
            },
        )
        self._events.append(event)

        # Update alert cooldown
        if action != QualityAction.LOG:
            self._alert_cooldowns[call_id] = datetime.utcnow()

        logger.info(
            "Quality event [%s] call=%s severity=%s mos=%.2f cause=%s",
            degradation_type.name,
            call_id,
            severity.display_name,
            mos,
            root_cause,
        )
        return event

    def record_pqc_overhead(self, sample: PQCOverheadSample) -> Optional[QualityEvent]:
        """
        Record a PQC operation overhead sample and check budget.

        Args:
            sample: The PQC overhead measurement.

        Returns:
            A QualityEvent if the PQC overhead exceeds the budget, or None.
        """
        self._pqc_tracker.record_sample(sample)

        if self._pqc_tracker.is_within_budget():
            return None

        avg_overhead = self._pqc_tracker.get_average_overhead()
        p95_overhead = self._pqc_tracker.get_p95_overhead()
        recommendation = self._pqc_tracker.get_recommendation()

        # Determine severity based on how far over budget
        budget = self._policy.pqc_overhead_budget_ms
        ratio = avg_overhead / budget if budget > 0 else 10.0
        if ratio > 3.0:
            severity = QualitySeverity.CRITICAL
        elif ratio > 2.0:
            severity = QualitySeverity.UNACCEPTABLE
        elif ratio > 1.5:
            severity = QualitySeverity.POOR
        else:
            severity = QualitySeverity.ACCEPTABLE

        event = QualityEvent(
            call_id=sample.call_id,
            degradation_type=QualityDegradationType.PQC_OVERHEAD_EXCESSIVE,
            severity=severity,
            metrics={
                "pqc_overhead_avg_ms": avg_overhead,
                "pqc_overhead_p95_ms": p95_overhead,
                "pqc_budget_ms": budget,
            },
            root_cause=recommendation or f"PQC overhead {avg_overhead:.1f}ms exceeds {budget:.1f}ms budget",
            action_taken=QualityAction.REDUCE_PQC_LEVEL if self._policy.auto_remediate else QualityAction.ALERT,
            pqc_algorithm=sample.algorithm,
            evidence={
                "algorithm": sample.algorithm,
                "operation": sample.operation,
                "sample_duration_ms": sample.duration_ms,
                "window_size": self._pqc_tracker.sample_count,
            },
        )
        self._events.append(event)

        logger.warning(
            "PQC overhead exceeded: avg=%.1fms p95=%.1fms budget=%.1fms algo=%s",
            avg_overhead, p95_overhead, budget, sample.algorithm,
        )
        return event

    def check_pqc_budget(self, call_id: str) -> Dict[str, Any]:
        """
        Check if PQC overhead is within budget for a call.

        Args:
            call_id: The call identifier.

        Returns:
            Dictionary with budget compliance details.
        """
        avg = self._pqc_tracker.get_average_overhead()
        p95 = self._pqc_tracker.get_p95_overhead()
        max_overhead = self._pqc_tracker.get_max_overhead()
        recommendation = self._pqc_tracker.get_recommendation()

        return {
            "call_id": call_id,
            "is_within_budget": self._pqc_tracker.is_within_budget(),
            "budget_ms": self._policy.pqc_overhead_budget_ms,
            "avg_overhead_ms": round(avg, 3),
            "p95_overhead_ms": round(p95, 3),
            "max_overhead_ms": round(max_overhead, 3),
            "sample_count": self._pqc_tracker.sample_count,
            "recommendation": recommendation,
        }

    def detect_degradation_trend(self, call_id: str) -> List[QualityEvent]:
        """
        Detect quality degradation trends using sliding window analysis.

        Looks for:
        - Declining MOS over the last N samples
        - Sustained poor quality (multiple consecutive violations)
        - Sudden quality drops (large MOS decrease between windows)

        Args:
            call_id: The call identifier to analyze.

        Returns:
            List of QualityEvent objects for any detected trends.
        """
        metrics = self._active_calls.get(call_id, [])
        if len(metrics) < 4:
            return []

        events: List[QualityEvent] = []
        mos_values = [m.value for m in metrics if m.metric_type == QualityMetricType.MOS_SCORE]

        if len(mos_values) < 4:
            return []

        # Trend 1: Declining MOS — compare first half average to second half
        midpoint = len(mos_values) // 2
        first_half_avg = sum(mos_values[:midpoint]) / midpoint
        second_half_avg = sum(mos_values[midpoint:]) / (len(mos_values) - midpoint)

        if first_half_avg - second_half_avg > 0.3:
            severity = MOSCalculator.severity_from_mos(second_half_avg)
            event = QualityEvent(
                call_id=call_id,
                degradation_type=QualityDegradationType.NETWORK_CONGESTION,
                severity=severity,
                metrics={
                    "mos_first_half_avg": round(first_half_avg, 2),
                    "mos_second_half_avg": round(second_half_avg, 2),
                    "mos_decline": round(first_half_avg - second_half_avg, 2),
                    "sample_count": len(mos_values),
                },
                root_cause=(
                    f"Declining MOS trend: {first_half_avg:.2f} -> {second_half_avg:.2f} "
                    f"over {len(mos_values)} samples"
                ),
                action_taken=QualityAction.ALERT,
            )
            events.append(event)

        # Trend 2: Sustained poor quality — count consecutive violations
        consecutive_poor = 0
        max_consecutive_poor = 0
        for mos_val in mos_values:
            if mos_val < self._policy.thresholds.min_mos:
                consecutive_poor += 1
                max_consecutive_poor = max(max_consecutive_poor, consecutive_poor)
            else:
                consecutive_poor = 0

        sustained_threshold = max(3, self._policy.trend_window_size // 10)
        if max_consecutive_poor >= sustained_threshold:
            poor_mos_values = [v for v in mos_values if v < self._policy.thresholds.min_mos]
            avg_poor = sum(poor_mos_values) / len(poor_mos_values) if poor_mos_values else 0.0
            severity = MOSCalculator.severity_from_mos(avg_poor)
            event = QualityEvent(
                call_id=call_id,
                degradation_type=QualityDegradationType.BANDWIDTH_SATURATION,
                severity=severity,
                metrics={
                    "consecutive_poor_samples": max_consecutive_poor,
                    "avg_poor_mos": round(avg_poor, 2),
                    "threshold_mos": self._policy.thresholds.min_mos,
                },
                root_cause=(
                    f"Sustained poor quality: {max_consecutive_poor} consecutive "
                    f"samples below MOS {self._policy.thresholds.min_mos}"
                ),
                action_taken=QualityAction.ESCALATE,
            )
            events.append(event)

        # Trend 3: Sudden quality drop — compare last sample to recent average
        if len(mos_values) >= 5:
            recent_avg = sum(mos_values[-6:-1]) / 5
            latest = mos_values[-1]
            if recent_avg - latest > 0.5:
                severity = MOSCalculator.severity_from_mos(latest)
                event = QualityEvent(
                    call_id=call_id,
                    degradation_type=QualityDegradationType.JITTER_SPIKE,
                    severity=severity,
                    metrics={
                        "recent_avg_mos": round(recent_avg, 2),
                        "latest_mos": round(latest, 2),
                        "mos_drop": round(recent_avg - latest, 2),
                    },
                    root_cause=(
                        f"Sudden quality drop: MOS fell from {recent_avg:.2f} "
                        f"to {latest:.2f}"
                    ),
                    action_taken=QualityAction.INCREASE_BUFFER,
                )
                events.append(event)

        for event in events:
            self._events.append(event)

        return events

    def get_call_quality_summary(self, call_id: str) -> Dict[str, Any]:
        """
        Get quality summary for a specific call.

        Args:
            call_id: The call identifier.

        Returns:
            Dictionary with call quality summary including average MOS,
            violation count, and PQC overhead impact.
        """
        metrics = self._active_calls.get(call_id, [])
        mos_values = [m.value for m in metrics if m.metric_type == QualityMetricType.MOS_SCORE]

        if not mos_values:
            return {
                "call_id": call_id,
                "sample_count": 0,
                "average_mos": 0.0,
                "min_mos": 0.0,
                "max_mos": 0.0,
                "violation_count": 0,
                "quality_grade": "UNKNOWN",
                "pqc_budget_status": self.check_pqc_budget(call_id),
            }

        avg_mos = sum(mos_values) / len(mos_values)
        min_mos = min(mos_values)
        max_mos = max(mos_values)
        violations = sum(1 for v in mos_values if v < self._policy.thresholds.min_mos)
        severity = MOSCalculator.severity_from_mos(avg_mos)

        # Count events for this call
        call_events = [e for e in self._events if e.call_id == call_id]
        events_by_type: Dict[str, int] = {}
        for evt in call_events:
            key = evt.degradation_type.name
            events_by_type[key] = events_by_type.get(key, 0) + 1

        return {
            "call_id": call_id,
            "sample_count": len(mos_values),
            "average_mos": round(avg_mos, 2),
            "min_mos": round(min_mos, 2),
            "max_mos": round(max_mos, 2),
            "violation_count": violations,
            "violation_pct": round(violations / len(mos_values) * 100, 1),
            "quality_grade": severity.display_name,
            "events_by_type": events_by_type,
            "total_events": len(call_events),
            "pqc_budget_status": self.check_pqc_budget(call_id),
        }

    def generate_report(
        self, period_start: datetime, period_end: datetime
    ) -> QualityReport:
        """
        Generate a quality report for the given time period.

        Aggregates all events and metrics within the period, computes
        statistics, and generates actionable recommendations.

        Args:
            period_start: Start of the reporting period.
            period_end: End of the reporting period.

        Returns:
            A populated QualityReport instance.
        """
        # Filter events in period
        period_events = [
            e for e in self._events
            if period_start <= e.detected_at <= period_end
        ]

        # Aggregate events by type and severity
        events_by_type: Dict[str, int] = {}
        events_by_severity: Dict[str, int] = {}
        for evt in period_events:
            type_key = evt.degradation_type.name
            events_by_type[type_key] = events_by_type.get(type_key, 0) + 1
            sev_key = evt.severity.display_name
            events_by_severity[sev_key] = events_by_severity.get(sev_key, 0) + 1

        # Collect all MOS values across active calls
        all_mos: List[float] = []
        call_ids: Set[str] = set()
        for cid, metrics_list in self._active_calls.items():
            for m in metrics_list:
                if (
                    m.metric_type == QualityMetricType.MOS_SCORE
                    and period_start <= m.timestamp <= period_end
                ):
                    all_mos.append(m.value)
                    call_ids.add(cid)

        avg_mos = sum(all_mos) / len(all_mos) if all_mos else 0.0
        min_mos = min(all_mos) if all_mos else 0.0
        max_mos = max(all_mos) if all_mos else 0.0

        # PQC overhead stats
        pqc_avg = self._pqc_tracker.get_average_overhead()
        pqc_p95 = self._pqc_tracker.get_p95_overhead()
        pqc_max = self._pqc_tracker.get_max_overhead()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_mos, period_events, pqc_avg
        )

        return QualityReport(
            period_start=period_start,
            period_end=period_end,
            total_calls_monitored=len(call_ids),
            average_mos=avg_mos,
            min_mos=min_mos,
            max_mos=max_mos,
            total_degradation_events=len(period_events),
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            pqc_overhead_avg_ms=pqc_avg,
            pqc_overhead_p95_ms=pqc_p95,
            pqc_overhead_max_ms=pqc_max,
            recommendations=recommendations,
        )

    def _check_thresholds(
        self, call_id: str, metrics: Dict[str, float]
    ) -> List[Tuple[QualityDegradationType, QualitySeverity]]:
        """
        Check all metrics against policy thresholds.

        Args:
            call_id: The call identifier.
            metrics: Dictionary of metric name to measured value.

        Returns:
            List of (degradation_type, severity) tuples for each violation.
        """
        violations: List[Tuple[QualityDegradationType, QualitySeverity]] = []
        thresholds = self._policy.thresholds

        mos = metrics.get("mos", 4.5)
        severity = MOSCalculator.severity_from_mos(mos)

        # MOS below minimum
        if mos < thresholds.min_mos:
            violations.append((QualityDegradationType.NETWORK_CONGESTION, severity))

        # Jitter exceeds maximum
        jitter = metrics.get("jitter_ms", 0.0)
        if jitter > thresholds.max_jitter_ms:
            jitter_ratio = jitter / thresholds.max_jitter_ms
            if jitter_ratio > 3.0:
                jitter_sev = QualitySeverity.CRITICAL
            elif jitter_ratio > 2.0:
                jitter_sev = QualitySeverity.UNACCEPTABLE
            elif jitter_ratio > 1.5:
                jitter_sev = QualitySeverity.POOR
            else:
                jitter_sev = QualitySeverity.ACCEPTABLE
            violations.append((QualityDegradationType.JITTER_SPIKE, jitter_sev))

        # Packet loss exceeds maximum
        pkt_loss = metrics.get("packet_loss_pct", 0.0)
        if pkt_loss > thresholds.max_packet_loss_pct:
            loss_ratio = pkt_loss / thresholds.max_packet_loss_pct if thresholds.max_packet_loss_pct > 0 else 10.0
            if loss_ratio > 5.0:
                loss_sev = QualitySeverity.CRITICAL
            elif loss_ratio > 3.0:
                loss_sev = QualitySeverity.UNACCEPTABLE
            elif loss_ratio > 2.0:
                loss_sev = QualitySeverity.POOR
            else:
                loss_sev = QualitySeverity.ACCEPTABLE
            violations.append((QualityDegradationType.PACKET_BURST_LOSS, loss_sev))

        # Latency exceeds maximum
        latency = metrics.get("latency_ms", 0.0)
        if latency > thresholds.max_latency_ms:
            lat_ratio = latency / thresholds.max_latency_ms if thresholds.max_latency_ms > 0 else 10.0
            if lat_ratio > 2.0:
                lat_sev = QualitySeverity.CRITICAL
            elif lat_ratio > 1.5:
                lat_sev = QualitySeverity.POOR
            else:
                lat_sev = QualitySeverity.ACCEPTABLE
            violations.append((QualityDegradationType.BANDWIDTH_SATURATION, lat_sev))

        # PQC overhead exceeds budget
        pqc = metrics.get("pqc_overhead_ms", 0.0)
        if pqc > thresholds.max_pqc_overhead_ms:
            pqc_ratio = pqc / thresholds.max_pqc_overhead_ms if thresholds.max_pqc_overhead_ms > 0 else 10.0
            if pqc_ratio > 3.0:
                pqc_sev = QualitySeverity.CRITICAL
            elif pqc_ratio > 2.0:
                pqc_sev = QualitySeverity.POOR
            else:
                pqc_sev = QualitySeverity.ACCEPTABLE
            violations.append((QualityDegradationType.PQC_OVERHEAD_EXCESSIVE, pqc_sev))

        # R-factor below minimum
        r_factor = metrics.get("r_factor", 100.0)
        if r_factor < thresholds.min_r_factor:
            r_severity = MOSCalculator.severity_from_mos(
                MOSCalculator.r_to_mos(r_factor)
            )
            violations.append((QualityDegradationType.NETWORK_CONGESTION, r_severity))

        # Echo return loss check (optional metric)
        echo_db = metrics.get("echo_return_loss_db")
        if echo_db is not None and echo_db > thresholds.max_echo_return_loss_db:
            violations.append((QualityDegradationType.ECHO_FEEDBACK, QualitySeverity.POOR))

        # Silence ratio check (optional metric)
        silence = metrics.get("silence_ratio")
        if silence is not None and silence > thresholds.max_silence_ratio:
            violations.append((
                QualityDegradationType.SILENCE_SUPPRESSION_FAILURE,
                QualitySeverity.ACCEPTABLE,
            ))

        return violations

    def _determine_root_cause(
        self,
        degradation_type: QualityDegradationType,
        metrics: Dict[str, float],
    ) -> str:
        """
        Determine likely root cause based on degradation type and metrics.

        Args:
            degradation_type: The detected degradation type.
            metrics: The metric measurements at detection time.

        Returns:
            Human-readable root cause description.
        """
        causes: Dict[QualityDegradationType, Callable[[], str]] = {
            QualityDegradationType.NETWORK_CONGESTION: lambda: (
                f"Network congestion: latency={metrics.get('latency_ms', 0):.0f}ms "
                f"jitter={metrics.get('jitter_ms', 0):.0f}ms "
                f"loss={metrics.get('packet_loss_pct', 0):.1f}% "
                f"MOS={metrics.get('mos', 0):.2f}"
            ),
            QualityDegradationType.CODEC_MISMATCH: lambda: (
                f"Codec mismatch or transcoding degradation: "
                f"MOS={metrics.get('mos', 0):.2f}"
            ),
            QualityDegradationType.PQC_OVERHEAD_EXCESSIVE: lambda: (
                f"PQC cryptographic overhead "
                f"{metrics.get('pqc_overhead_ms', 0):.1f}ms "
                f"exceeds {self._policy.thresholds.max_pqc_overhead_ms:.1f}ms budget"
            ),
            QualityDegradationType.JITTER_SPIKE: lambda: (
                f"Jitter spike: {metrics.get('jitter_ms', 0):.1f}ms "
                f"(max allowed {self._policy.thresholds.max_jitter_ms:.0f}ms)"
            ),
            QualityDegradationType.PACKET_BURST_LOSS: lambda: (
                f"Packet burst loss: {metrics.get('packet_loss_pct', 0):.2f}% "
                f"(max allowed {self._policy.thresholds.max_packet_loss_pct:.1f}%)"
            ),
            QualityDegradationType.ECHO_FEEDBACK: lambda: (
                f"Echo feedback detected: "
                f"ERL={metrics.get('echo_return_loss_db', 0):.1f}dB"
            ),
            QualityDegradationType.SILENCE_SUPPRESSION_FAILURE: lambda: (
                f"Silence suppression failure: ratio="
                f"{metrics.get('silence_ratio', 0):.2f}"
            ),
            QualityDegradationType.DTMF_CORRUPTION: lambda: (
                f"DTMF tone corruption affecting IVR navigation"
            ),
            QualityDegradationType.BANDWIDTH_SATURATION: lambda: (
                f"Bandwidth saturation: latency={metrics.get('latency_ms', 0):.0f}ms "
                f"exceeds {self._policy.thresholds.max_latency_ms:.0f}ms limit"
            ),
            QualityDegradationType.ENCRYPTION_LATENCY: lambda: (
                f"Encryption processing latency: "
                f"{metrics.get('pqc_overhead_ms', 0):.1f}ms"
            ),
        }

        handler = causes.get(degradation_type)
        if handler:
            return handler()

        return f"Quality degradation: {degradation_type.name}"

    def _determine_action(
        self,
        severity: QualitySeverity,
        degradation_type: QualityDegradationType,
    ) -> QualityAction:
        """
        Determine the appropriate remediation action.

        Args:
            severity: The detected severity level.
            degradation_type: The degradation type.

        Returns:
            The recommended QualityAction.
        """
        if not self._policy.auto_remediate:
            if severity.level >= QualitySeverity.POOR.level:
                return QualityAction.ALERT
            return QualityAction.LOG

        # PQC-specific remediation
        if degradation_type == QualityDegradationType.PQC_OVERHEAD_EXCESSIVE:
            if severity.level >= QualitySeverity.UNACCEPTABLE.level:
                return QualityAction.REDUCE_PQC_LEVEL
            return QualityAction.ALERT

        # Jitter-specific remediation
        if degradation_type == QualityDegradationType.JITTER_SPIKE:
            return QualityAction.INCREASE_BUFFER

        # Severity-based escalation
        if severity.level >= QualitySeverity.CRITICAL.level:
            return QualityAction.TERMINATE_CALL
        if severity.level >= QualitySeverity.UNACCEPTABLE.level:
            return QualityAction.ESCALATE
        if severity.level >= QualitySeverity.POOR.level:
            return QualityAction.REROUTE
        if severity.level >= QualitySeverity.ACCEPTABLE.level:
            return QualityAction.ALERT

        return QualityAction.LOG

    def _can_alert(self, call_id: str) -> bool:
        """
        Check if alerting is allowed (respecting cooldown).

        Args:
            call_id: The call identifier.

        Returns:
            True if an alert can be sent, False if in cooldown.
        """
        last_alert = self._alert_cooldowns.get(call_id)
        if last_alert is None:
            return True
        elapsed = (datetime.utcnow() - last_alert).total_seconds()
        return elapsed >= self._policy.alert_cooldown_seconds

    def _generate_recommendations(
        self,
        avg_mos: float,
        events: List[QualityEvent],
        pqc_avg: float,
    ) -> List[str]:
        """
        Generate actionable recommendations based on report data.

        Args:
            avg_mos: Average MOS for the period.
            events: Quality events in the period.
            pqc_avg: Average PQC overhead in the period.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if avg_mos < 3.5:
            recommendations.append(
                f"Average MOS ({avg_mos:.2f}) is below acceptable threshold (3.5). "
                f"Review network capacity and codec selection."
            )
        elif avg_mos < 4.0:
            recommendations.append(
                f"Average MOS ({avg_mos:.2f}) is acceptable but could be improved. "
                f"Consider enabling wideband codecs (G.722/OPUS)."
            )

        # PQC overhead recommendations
        pqc_recommendation = self._pqc_tracker.get_recommendation()
        if pqc_recommendation:
            recommendations.append(pqc_recommendation)
        elif pqc_avg > 0 and pqc_avg <= self._policy.pqc_overhead_budget_ms:
            recommendations.append(
                f"PQC overhead ({pqc_avg:.1f}ms) is within budget "
                f"({self._policy.pqc_overhead_budget_ms:.1f}ms). "
                f"Quantum-safe protection is operating efficiently."
            )

        # Analyze event patterns
        type_counts: Dict[str, int] = {}
        for evt in events:
            key = evt.degradation_type.name
            type_counts[key] = type_counts.get(key, 0) + 1

        if type_counts.get("JITTER_SPIKE", 0) > 5:
            recommendations.append(
                "Frequent jitter spikes detected. Consider increasing "
                "jitter buffer depth or enabling adaptive jitter buffering."
            )

        if type_counts.get("PACKET_BURST_LOSS", 0) > 5:
            recommendations.append(
                "Recurring packet burst loss. Investigate network path "
                "for congestion points or faulty equipment."
            )

        if type_counts.get("PQC_OVERHEAD_EXCESSIVE", 0) > 3:
            recommendations.append(
                "Repeated PQC overhead violations. Evaluate if a lighter "
                "algorithm (e.g., ML-KEM-512) can meet security requirements."
            )

        critical_count = sum(
            1 for evt in events
            if evt.severity.level >= QualitySeverity.CRITICAL.level
        )
        if critical_count > 0:
            recommendations.append(
                f"{critical_count} critical quality events detected. "
                f"Immediate investigation recommended."
            )

        if not recommendations:
            recommendations.append(
                "Voice quality metrics are within acceptable thresholds. "
                "No action required."
            )

        return recommendations

    def end_call(self, call_id: str) -> Dict[str, Any]:
        """
        Mark a call as ended and return its final quality summary.

        Args:
            call_id: The call identifier to finalize.

        Returns:
            Final quality summary for the call.
        """
        summary = self.get_call_quality_summary(call_id)

        # Clean up active call data
        self._active_calls.pop(call_id, None)
        self._alert_cooldowns.pop(call_id, None)

        logger.info(
            "Call ended [%s] avg_mos=%.2f violations=%d",
            call_id,
            summary.get("average_mos", 0.0),
            summary.get("violation_count", 0),
        )
        return summary

    # -------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------

    @classmethod
    def create_default_policy(cls) -> "QualityPolicy":
        """Factory: default BPO voice quality policy."""
        return QualityPolicy(
            name="default",
            thresholds=QualityThresholds(),
            pqc_overhead_budget_ms=2.0,
            monitoring_mode=MonitoringMode.PASSIVE,
            auto_remediate=True,
            alert_cooldown_seconds=60,
            trend_window_size=100,
        )

    @classmethod
    def create_high_quality_policy(cls) -> "QualityPolicy":
        """Factory: premium voice quality with tighter thresholds."""
        return QualityPolicy(
            name="high_quality",
            thresholds=QualityThresholds(
                max_jitter_ms=20.0,
                max_packet_loss_pct=0.5,
                max_latency_ms=100.0,
                min_mos=4.0,
            ),
            pqc_overhead_budget_ms=1.5,
        )

    @classmethod
    def create_pqc_aware_policy(cls) -> "QualityPolicy":
        """Factory: policy with increased PQC overhead budget for high-security calls."""
        return QualityPolicy(
            name="pqc_aware",
            thresholds=QualityThresholds(
                max_latency_ms=200.0,
                min_mos=3.3,
            ),
            pqc_overhead_budget_ms=5.0,
        )
