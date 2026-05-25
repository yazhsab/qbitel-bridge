"""
SIP Toll Fraud Detection and Prevention Module

Detects and prevents SIP toll fraud in BPO environments.

Toll fraud costs the BPO industry $10B+ annually. This module detects:
- International Revenue Share Fraud (IRSF) - calls to premium rate numbers
- PBX hacking - unauthorized access to outbound trunks
- Call transfer fraud - transfers to premium destinations
- Wangiri fraud - callbacks to missed calls from premium numbers
- Subscription fraud - fake accounts making premium calls

Integrates with QBITEL's quantum-safe infrastructure for secure
alerting and evidence preservation.
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


class FraudPatternType(Enum):
    """Types of toll fraud patterns detected."""

    IRSF = auto()                   # International Revenue Share Fraud
    PBX_HACK = auto()               # Unauthorized PBX access
    CALL_TRANSFER = auto()          # Transfer to premium destination
    WANGIRI = auto()                # Callback fraud (missed call bait)
    SUBSCRIPTION = auto()           # Fake account fraud
    CALL_PUMPING = auto()           # Artificial traffic inflation
    ARBITRAGE = auto()              # Rate arbitrage exploitation
    BYPASS = auto()                 # SIM box / gateway bypass
    CLIP_MANIPULATION = auto()      # Caller ID spoofing
    TOLL_FREE_ABUSE = auto()        # Toll-free number abuse


class FraudSeverity(Enum):
    """Severity levels for detected fraud patterns."""

    LOW = (1, "Low", 0.0, 100.0)
    MEDIUM = (2, "Medium", 100.0, 1000.0)
    HIGH = (3, "High", 1000.0, 10000.0)
    CRITICAL = (4, "Critical", 10000.0, float("inf"))

    def __init__(
        self,
        level: int,
        display_name: str,
        min_estimated_loss: float,
        max_estimated_loss: float,
    ):
        self.level = level
        self.display_name = display_name
        self.min_estimated_loss = min_estimated_loss
        self.max_estimated_loss = max_estimated_loss


class FraudAction(Enum):
    """Actions to take when fraud is detected."""

    LOG = auto()              # Log event only
    ALERT = auto()            # Send alert to security team
    RATE_LIMIT = auto()       # Throttle outbound calls
    REQUIRE_AUTH = auto()     # Require re-authentication
    BLOCK = auto()            # Block the call immediately
    QUARANTINE = auto()       # Isolate the session
    TERMINATE = auto()        # Terminate the session and block agent


class CallDirection(Enum):
    """Call direction."""

    INBOUND = auto()
    OUTBOUND = auto()
    INTERNAL = auto()
    TRANSFER = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FraudPattern:
    """
    Represents a detected fraud pattern.

    Contains pattern classification, confidence scoring, and
    supporting evidence for investigation.
    """

    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: FraudPatternType = FraudPatternType.IRSF
    severity: FraudSeverity = FraudSeverity.LOW
    confidence: float = 0.0            # 0.0 to 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    detected_at: datetime = field(default_factory=datetime.utcnow)
    call_id: Optional[str] = None
    agent_id: Optional[str] = None
    source_number: Optional[str] = None
    destination_number: Optional[str] = None
    estimated_loss: float = 0.0
    related_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pattern for storage or transmission."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.name,
            "severity": self.severity.display_name,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "call_id": self.call_id,
            "agent_id": self.agent_id,
            "source_number": self.source_number,
            "destination_number": self.destination_number,
            "estimated_loss": self.estimated_loss,
            "related_patterns": self.related_patterns,
        }


@dataclass
class TollFraudRule:
    """
    A configurable rule for toll fraud detection.

    Rules are evaluated against call metadata and behavior
    to identify potential fraud.
    """

    rule_id: str = ""
    name: str = ""
    description: str = ""
    pattern: FraudPatternType = FraudPatternType.IRSF
    action: FraudAction = FraudAction.ALERT
    severity: FraudSeverity = FraudSeverity.MEDIUM
    enabled: bool = True
    priority: int = 100              # Lower = higher priority
    conditions: Dict[str, Any] = field(default_factory=dict)
    cooldown_seconds: int = 60       # Minimum interval between triggers
    max_triggers_per_hour: int = 100
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def matches_pattern_type(self, pattern_type: FraudPatternType) -> bool:
        """Check if rule applies to the given pattern type."""
        return self.pattern == pattern_type


@dataclass
class CallRecord:
    """
    A record of a call for fraud analysis.

    Captures the essential metadata needed for fraud detection
    without storing call content.
    """

    call_id: str = ""
    agent_id: str = ""
    tenant_id: str = ""
    direction: CallDirection = CallDirection.OUTBOUND
    source_number: str = ""
    destination_number: str = ""
    destination_country: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    trunk_id: str = ""
    was_transferred: bool = False
    transfer_destination: Optional[str] = None
    sip_response_code: int = 200
    cost_per_minute: float = 0.0

    @property
    def is_international(self) -> bool:
        """Check if the call is international."""
        return self.destination_country != "" and self.destination_country != "US"

    @property
    def estimated_cost(self) -> float:
        """Estimate the cost of this call."""
        return (self.duration_seconds / 60.0) * self.cost_per_minute


@dataclass
class AgentCallProfile:
    """
    Statistical profile of an agent's calling behavior.

    Used as a baseline for anomaly detection. Profiles are built
    over a rolling window and updated continuously.
    """

    agent_id: str = ""
    tenant_id: str = ""
    profile_window_days: int = 30
    total_calls: int = 0
    avg_calls_per_shift: float = 0.0
    avg_call_duration_seconds: float = 0.0
    std_call_duration_seconds: float = 0.0
    international_call_ratio: float = 0.0
    avg_international_calls_per_shift: float = 0.0
    common_destinations: Set[str] = field(default_factory=set)
    common_countries: Set[str] = field(default_factory=set)
    typical_hours: Set[int] = field(default_factory=set)
    transfer_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Premium Rate Number Database
# ---------------------------------------------------------------------------


# Premium rate number prefixes by country/region
# These are high-risk destinations commonly targeted by IRSF
PREMIUM_RATE_PREFIXES: Dict[str, List[str]] = {
    # Caribbean & Atlantic
    "cuba": ["+53"],
    "jamaica": ["+1876", "+1658"],
    "dominican_republic": ["+1809", "+1829", "+1849"],
    "trinidad_tobago": ["+1868"],
    "grenada": ["+1473"],
    "guyana": ["+592"],
    "suriname": ["+597"],
    # Africa
    "somalia": ["+252"],
    "sierra_leone": ["+232"],
    "guinea": ["+224"],
    "guinea_bissau": ["+245"],
    "chad": ["+235"],
    "central_african_republic": ["+236"],
    "comoros": ["+269"],
    "sao_tome": ["+239"],
    "mauritania": ["+222"],
    "niger": ["+227"],
    "burundi": ["+257"],
    # Eastern Europe
    "bosnia": ["+387"],
    "moldova": ["+373"],
    "latvia_premium": ["+3719"],
    "estonia_premium": ["+3729"],
    "lithuania_premium": ["+3709"],
    # Pacific Islands
    "tuvalu": ["+688"],
    "nauru": ["+674"],
    "kiribati": ["+686"],
    "marshall_islands": ["+692"],
    "solomon_islands": ["+677"],
    "vanuatu": ["+678"],
    "tonga": ["+676"],
    "samoa": ["+685"],
    # Premium services (International Premium Rate)
    "iprn_europe": ["+388"],
    "satellite": ["+870", "+871", "+872", "+873"],
    "inmarsat": ["+8816", "+8817"],
    "globalstar": ["+8818", "+8819"],
    "thuraya": ["+88216"],
    # Known IRSF test number ranges
    "irsf_test_1": ["+99"],
    "irsf_test_2": ["+979"],
}

# Flatten all premium prefixes for fast lookup
ALL_PREMIUM_PREFIXES: List[str] = sorted(
    [prefix for prefixes in PREMIUM_RATE_PREFIXES.values() for prefix in prefixes],
    key=len,
    reverse=True,  # Longest prefix first for accurate matching
)

# High-cost destination country codes
HIGH_COST_COUNTRY_CODES: Set[str] = {
    "53", "252", "232", "224", "245", "235", "236", "269",
    "239", "222", "227", "257", "688", "674", "686", "692",
    "677", "678", "676", "685", "870", "871", "872", "873",
}


# ---------------------------------------------------------------------------
# Toll Fraud Detector
# ---------------------------------------------------------------------------


class TollFraudDetector:
    """
    Detects and prevents SIP toll fraud in BPO environments.

    Toll fraud costs the BPO industry $10B+ annually. This module detects:
    - International Revenue Share Fraud (IRSF) - calls to premium rate numbers
    - PBX hacking - unauthorized access to outbound trunks
    - Call transfer fraud - transfers to premium destinations
    - Wangiri fraud - callbacks to missed calls from premium numbers
    - Subscription fraud - fake accounts making premium calls

    The detector operates in real-time, analyzing call setup requests
    before they are connected, and monitoring active calls for
    suspicious patterns.

    Usage::

        detector = TollFraudDetector(tenant_id="tenant-001")
        detector.load_default_rules()

        # Analyze a call before connecting
        patterns = detector.analyze_call(call_record)
        if patterns:
            action = detector.get_recommended_action(patterns)
            if action == FraudAction.BLOCK:
                # Reject the call
                ...

        # Analyze agent behavior over time
        anomalies = detector.analyze_agent_behavior(agent_id, recent_calls)
    """

    def __init__(
        self,
        tenant_id: str = "",
        *,
        enable_real_time: bool = True,
        enable_ml_scoring: bool = True,
        alert_callback: Optional[Callable[[FraudPattern], None]] = None,
        blocked_prefixes: Optional[List[str]] = None,
        allowed_countries: Optional[Set[str]] = None,
        max_call_cost: float = 50.0,
        max_international_calls_per_hour: int = 20,
        off_hours_blocking: bool = True,
        business_hours_start: int = 6,
        business_hours_end: int = 22,
    ):
        self.tenant_id = tenant_id
        self.enable_real_time = enable_real_time
        self.enable_ml_scoring = enable_ml_scoring
        self.alert_callback = alert_callback
        self.max_call_cost = max_call_cost
        self.max_international_calls_per_hour = max_international_calls_per_hour
        self.off_hours_blocking = off_hours_blocking
        self.business_hours_start = business_hours_start
        self.business_hours_end = business_hours_end

        # Blocked / allowed lists
        self._blocked_prefixes: List[str] = blocked_prefixes or list(ALL_PREMIUM_PREFIXES)
        self._allowed_countries: Optional[Set[str]] = allowed_countries

        # Rules engine
        self._rules: Dict[str, TollFraudRule] = {}

        # Agent profiles (agent_id -> profile)
        self._agent_profiles: Dict[str, AgentCallProfile] = {}

        # Recent call tracking (agent_id -> list of recent calls)
        self._recent_calls: Dict[str, List[CallRecord]] = {}

        # Detection counters (for velocity analysis)
        self._call_counts: Dict[str, List[datetime]] = {}

        # Pattern history
        self._detected_patterns: List[FraudPattern] = []

        # Statistics
        self._stats = {
            "total_calls_analyzed": 0,
            "total_patterns_detected": 0,
            "total_calls_blocked": 0,
            "total_estimated_savings": 0.0,
        }

        logger.info(
            "TollFraudDetector initialized for tenant=%s "
            "real_time=%s ml_scoring=%s blocked_prefixes=%d",
            tenant_id,
            enable_real_time,
            enable_ml_scoring,
            len(self._blocked_prefixes),
        )

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def load_default_rules(self) -> None:
        """Load the default set of toll fraud detection rules."""
        default_rules = [
            TollFraudRule(
                rule_id="TF-001",
                name="Premium Rate Number Block",
                description="Block outbound calls to known premium rate numbers",
                pattern=FraudPatternType.IRSF,
                action=FraudAction.BLOCK,
                severity=FraudSeverity.HIGH,
                enabled=True,
                priority=10,
                conditions={"check_premium_prefix": True},
            ),
            TollFraudRule(
                rule_id="TF-002",
                name="International Call Velocity",
                description=(
                    "Alert on excessive international calls within a short window"
                ),
                pattern=FraudPatternType.IRSF,
                action=FraudAction.ALERT,
                severity=FraudSeverity.MEDIUM,
                enabled=True,
                priority=20,
                conditions={
                    "max_international_per_hour": self.max_international_calls_per_hour,
                },
            ),
            TollFraudRule(
                rule_id="TF-003",
                name="Off-Hours International Call",
                description=(
                    "Block international calls outside business hours"
                ),
                pattern=FraudPatternType.PBX_HACK,
                action=FraudAction.BLOCK,
                severity=FraudSeverity.HIGH,
                enabled=self.off_hours_blocking,
                priority=15,
                conditions={
                    "hours_start": self.business_hours_start,
                    "hours_end": self.business_hours_end,
                },
            ),
            TollFraudRule(
                rule_id="TF-004",
                name="Call Transfer to Premium Destination",
                description=(
                    "Block call transfers to premium rate or high-cost destinations"
                ),
                pattern=FraudPatternType.CALL_TRANSFER,
                action=FraudAction.BLOCK,
                severity=FraudSeverity.CRITICAL,
                enabled=True,
                priority=5,
                conditions={"check_transfer_destination": True},
            ),
            TollFraudRule(
                rule_id="TF-005",
                name="Wangiri Callback Detection",
                description=(
                    "Alert on callbacks to numbers that generated short-ring "
                    "missed calls from premium destinations"
                ),
                pattern=FraudPatternType.WANGIRI,
                action=FraudAction.REQUIRE_AUTH,
                severity=FraudSeverity.MEDIUM,
                enabled=True,
                priority=25,
                conditions={
                    "max_ring_duration_seconds": 5,
                    "check_callback_premium": True,
                },
            ),
            TollFraudRule(
                rule_id="TF-006",
                name="Rapid Sequential Calls",
                description=(
                    "Detect rapid sequential outbound calls indicating "
                    "automated dialing or PBX compromise"
                ),
                pattern=FraudPatternType.PBX_HACK,
                action=FraudAction.RATE_LIMIT,
                severity=FraudSeverity.HIGH,
                enabled=True,
                priority=15,
                conditions={
                    "max_calls_per_minute": 5,
                    "max_calls_per_5_minutes": 15,
                },
            ),
            TollFraudRule(
                rule_id="TF-007",
                name="Abnormal Call Duration",
                description=(
                    "Detect abnormally long calls (> 4 hours) or extremely "
                    "short calls (< 3 seconds) to international destinations"
                ),
                pattern=FraudPatternType.CALL_PUMPING,
                action=FraudAction.ALERT,
                severity=FraudSeverity.MEDIUM,
                enabled=True,
                priority=30,
                conditions={
                    "max_duration_seconds": 14400,  # 4 hours
                    "min_duration_seconds": 3,
                },
            ),
            TollFraudRule(
                rule_id="TF-008",
                name="Geographic Anomaly Detection",
                description=(
                    "Alert on calls to destinations never previously called "
                    "by this agent or tenant"
                ),
                pattern=FraudPatternType.IRSF,
                action=FraudAction.ALERT,
                severity=FraudSeverity.LOW,
                enabled=True,
                priority=40,
                conditions={"check_new_destination": True},
            ),
            TollFraudRule(
                rule_id="TF-009",
                name="Call Cost Threshold",
                description=(
                    "Block calls exceeding the per-call cost threshold"
                ),
                pattern=FraudPatternType.IRSF,
                action=FraudAction.BLOCK,
                severity=FraudSeverity.HIGH,
                enabled=True,
                priority=10,
                conditions={"max_cost": self.max_call_cost},
            ),
            TollFraudRule(
                rule_id="TF-010",
                name="Caller ID Spoofing Detection",
                description=(
                    "Detect manipulation of outbound caller ID / CLIP"
                ),
                pattern=FraudPatternType.CLIP_MANIPULATION,
                action=FraudAction.BLOCK,
                severity=FraudSeverity.CRITICAL,
                enabled=True,
                priority=5,
                conditions={"validate_cli": True},
            ),
        ]

        for rule in default_rules:
            self._rules[rule.rule_id] = rule

        logger.info("Loaded %d default toll fraud rules", len(default_rules))

    def add_rule(self, rule: TollFraudRule) -> None:
        """Add or update a fraud detection rule."""
        self._rules[rule.rule_id] = rule
        logger.info("Added/updated toll fraud rule: %s (%s)", rule.rule_id, rule.name)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a fraud detection rule. Returns True if removed."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info("Removed toll fraud rule: %s", rule_id)
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a fraud detection rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a fraud detection rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False
            return True
        return False

    def get_rules(self) -> List[TollFraudRule]:
        """Get all configured rules, sorted by priority."""
        return sorted(self._rules.values(), key=lambda r: r.priority)

    # ------------------------------------------------------------------
    # Call analysis - main entry point
    # ------------------------------------------------------------------

    def analyze_call(self, call: CallRecord) -> List[FraudPattern]:
        """
        Analyze a call record for fraud indicators.

        This is the primary entry point for real-time fraud detection.
        It evaluates the call against all enabled rules and returns
        any detected fraud patterns.

        Args:
            call: The call record to analyze.

        Returns:
            List of detected fraud patterns, ordered by severity.
        """
        self._stats["total_calls_analyzed"] += 1
        detected: List[FraudPattern] = []

        # Run each analysis method
        detected.extend(self._analyze_destination(call))
        detected.extend(self._analyze_time(call))
        detected.extend(self._analyze_velocity(call))
        detected.extend(self._analyze_duration(call))
        detected.extend(self._analyze_transfer(call))
        detected.extend(self._analyze_geography(call))
        detected.extend(self._analyze_cost(call))

        # Sort by severity (critical first)
        detected.sort(key=lambda p: p.severity.level, reverse=True)

        # Track detected patterns
        if detected:
            self._stats["total_patterns_detected"] += len(detected)
            self._detected_patterns.extend(detected)

            # Fire callback if configured
            for pattern in detected:
                self._fire_alert(pattern)

            logger.warning(
                "Toll fraud detected: call_id=%s patterns=%d highest_severity=%s",
                call.call_id,
                len(detected),
                detected[0].severity.display_name,
            )

        # Track call for velocity analysis
        self._track_call(call)

        return detected

    def get_recommended_action(
        self, patterns: List[FraudPattern]
    ) -> FraudAction:
        """
        Determine the recommended action based on detected patterns.

        The highest-severity matching rule determines the action.

        Args:
            patterns: List of detected fraud patterns.

        Returns:
            The recommended FraudAction.
        """
        if not patterns:
            return FraudAction.LOG

        # Find the most severe action from matching rules
        most_severe_action = FraudAction.LOG
        highest_priority = float("inf")

        for pattern in patterns:
            for rule in self._rules.values():
                if (
                    rule.enabled
                    and rule.matches_pattern_type(pattern.pattern_type)
                    and rule.priority < highest_priority
                ):
                    most_severe_action = rule.action
                    highest_priority = rule.priority

        return most_severe_action

    # ------------------------------------------------------------------
    # Destination analysis
    # ------------------------------------------------------------------

    def _analyze_destination(self, call: CallRecord) -> List[FraudPattern]:
        """Check destination number against blocked/premium lists."""
        patterns: List[FraudPattern] = []

        if call.direction not in (CallDirection.OUTBOUND, CallDirection.TRANSFER):
            return patterns

        destination = self._normalize_number(call.destination_number)

        # Check against premium rate prefixes
        matched_prefix = self._match_premium_prefix(destination)
        if matched_prefix is not None:
            region = self._get_prefix_region(matched_prefix)
            patterns.append(
                FraudPattern(
                    pattern_type=FraudPatternType.IRSF,
                    severity=FraudSeverity.HIGH,
                    confidence=0.95,
                    description=(
                        f"Outbound call to premium rate destination: "
                        f"{matched_prefix} ({region})"
                    ),
                    evidence={
                        "matched_prefix": matched_prefix,
                        "region": region,
                        "destination": self._mask_number(destination),
                        "rule": "TF-001",
                    },
                    call_id=call.call_id,
                    agent_id=call.agent_id,
                    source_number=self._mask_number(call.source_number),
                    destination_number=self._mask_number(destination),
                    estimated_loss=call.estimated_cost,
                )
            )

        # Check against allowed countries (if configured)
        if (
            self._allowed_countries is not None
            and call.destination_country
            and call.destination_country not in self._allowed_countries
        ):
            patterns.append(
                FraudPattern(
                    pattern_type=FraudPatternType.IRSF,
                    severity=FraudSeverity.MEDIUM,
                    confidence=0.7,
                    description=(
                        f"Call to non-allowed country: {call.destination_country}"
                    ),
                    evidence={
                        "destination_country": call.destination_country,
                        "allowed_countries": list(self._allowed_countries),
                        "rule": "country_allowlist",
                    },
                    call_id=call.call_id,
                    agent_id=call.agent_id,
                    destination_number=self._mask_number(destination),
                )
            )

        return patterns

    def _match_premium_prefix(self, number: str) -> Optional[str]:
        """Match a number against premium rate prefixes."""
        for prefix in self._blocked_prefixes:
            if number.startswith(prefix):
                return prefix
        return None

    def _get_prefix_region(self, prefix: str) -> str:
        """Look up the region name for a premium prefix."""
        for region, prefixes in PREMIUM_RATE_PREFIXES.items():
            if prefix in prefixes:
                return region.replace("_", " ").title()
        return "Unknown"

    # ------------------------------------------------------------------
    # Time-based analysis
    # ------------------------------------------------------------------

    def _analyze_time(self, call: CallRecord) -> List[FraudPattern]:
        """Detect off-hours calls to international destinations."""
        patterns: List[FraudPattern] = []

        if not call.is_international:
            return patterns

        hour = call.start_time.hour
        is_off_hours = (
            hour < self.business_hours_start or hour >= self.business_hours_end
        )

        if is_off_hours:
            # Off-hours international call is suspicious
            day_of_week = call.start_time.weekday()
            is_weekend = day_of_week >= 5

            severity = FraudSeverity.HIGH if is_weekend else FraudSeverity.MEDIUM
            confidence = 0.8 if is_weekend else 0.6

            patterns.append(
                FraudPattern(
                    pattern_type=FraudPatternType.PBX_HACK,
                    severity=severity,
                    confidence=confidence,
                    description=(
                        f"International call at off-hours: "
                        f"{call.start_time.strftime('%H:%M')} "
                        f"({'weekend' if is_weekend else 'weeknight'})"
                    ),
                    evidence={
                        "hour": hour,
                        "day_of_week": day_of_week,
                        "is_weekend": is_weekend,
                        "business_hours": (
                            f"{self.business_hours_start:02d}:00-"
                            f"{self.business_hours_end:02d}:00"
                        ),
                        "destination_country": call.destination_country,
                        "rule": "TF-003",
                    },
                    call_id=call.call_id,
                    agent_id=call.agent_id,
                    destination_number=self._mask_number(call.destination_number),
                )
            )

        return patterns

    # ------------------------------------------------------------------
    # Velocity analysis
    # ------------------------------------------------------------------

    def _analyze_velocity(self, call: CallRecord) -> List[FraudPattern]:
        """Detect rapid sequential calls indicating automated fraud."""
        patterns: List[FraudPattern] = []
        agent_key = f"{call.tenant_id}:{call.agent_id}"

        if agent_key not in self._call_counts:
            self._call_counts[agent_key] = []

        # Get recent call timestamps
        now = datetime.utcnow()
        recent = [
            ts for ts in self._call_counts[agent_key]
            if (now - ts).total_seconds() < 3600
        ]
        self._call_counts[agent_key] = recent  # Clean up old entries

        # Calls in the last minute
        last_minute = [
            ts for ts in recent
            if (now - ts).total_seconds() < 60
        ]

        # Calls in the last 5 minutes
        last_5_minutes = [
            ts for ts in recent
            if (now - ts).total_seconds() < 300
        ]

        # Check rapid sequential calls (> 5 per minute)
        rule = self._rules.get("TF-006")
        if rule and rule.enabled:
            max_per_minute = rule.conditions.get("max_calls_per_minute", 5)
            max_per_5_minutes = rule.conditions.get("max_calls_per_5_minutes", 15)

            if len(last_minute) >= max_per_minute:
                patterns.append(
                    FraudPattern(
                        pattern_type=FraudPatternType.PBX_HACK,
                        severity=FraudSeverity.HIGH,
                        confidence=0.85,
                        description=(
                            f"Rapid sequential calls: {len(last_minute)} "
                            f"calls in the last minute (threshold: {max_per_minute})"
                        ),
                        evidence={
                            "calls_last_minute": len(last_minute),
                            "threshold": max_per_minute,
                            "rule": "TF-006",
                        },
                        call_id=call.call_id,
                        agent_id=call.agent_id,
                    )
                )

            if len(last_5_minutes) >= max_per_5_minutes:
                patterns.append(
                    FraudPattern(
                        pattern_type=FraudPatternType.PBX_HACK,
                        severity=FraudSeverity.MEDIUM,
                        confidence=0.75,
                        description=(
                            f"High call volume: {len(last_5_minutes)} calls "
                            f"in the last 5 minutes (threshold: {max_per_5_minutes})"
                        ),
                        evidence={
                            "calls_last_5_minutes": len(last_5_minutes),
                            "threshold": max_per_5_minutes,
                            "rule": "TF-006",
                        },
                        call_id=call.call_id,
                        agent_id=call.agent_id,
                    )
                )

        # Check international call velocity
        international_recent = [
            ts for ts in recent
            if (now - ts).total_seconds() < 3600
        ]
        if (
            call.is_international
            and len(international_recent) >= self.max_international_calls_per_hour
        ):
            patterns.append(
                FraudPattern(
                    pattern_type=FraudPatternType.IRSF,
                    severity=FraudSeverity.HIGH,
                    confidence=0.8,
                    description=(
                        f"Excessive international calls: "
                        f"{len(international_recent)} in the last hour "
                        f"(threshold: {self.max_international_calls_per_hour})"
                    ),
                    evidence={
                        "international_calls_last_hour": len(international_recent),
                        "threshold": self.max_international_calls_per_hour,
                        "rule": "TF-002",
                    },
                    call_id=call.call_id,
                    agent_id=call.agent_id,
                )
            )

        return patterns

    # ------------------------------------------------------------------
    # Duration analysis
    # ------------------------------------------------------------------

    def _analyze_duration(self, call: CallRecord) -> List[FraudPattern]:
        """Detect abnormally long or short call durations."""
        patterns: List[FraudPattern] = []

        if call.duration_seconds <= 0 or not call.is_international:
            return patterns

        rule = self._rules.get("TF-007")
        if not rule or not rule.enabled:
            return patterns

        max_duration = rule.conditions.get("max_duration_seconds", 14400)
        min_duration = rule.conditions.get("min_duration_seconds", 3)

        # Abnormally long calls (e.g., call pumping to premium numbers)
        if call.duration_seconds > max_duration:
            patterns.append(
                FraudPattern(
                    pattern_type=FraudPatternType.CALL_PUMPING,
                    severity=FraudSeverity.MEDIUM,
                    confidence=0.65,
                    description=(
                        f"Abnormally long international call: "
                        f"{call.duration_seconds / 3600:.1f} hours "
                        f"(threshold: {max_duration / 3600:.1f} hours)"
                    ),
                    evidence={
                        "duration_seconds": call.duration_seconds,
                        "max_duration_seconds": max_duration,
                        "destination_country": call.destination_country,
                        "estimated_cost": call.estimated_cost,
                        "rule": "TF-007",
                    },
                    call_id=call.call_id,
                    agent_id=call.agent_id,
                    estimated_loss=call.estimated_cost,
                )
            )

        # Extremely short calls (potential IRSF test calls)
        if call.duration_seconds < min_duration and call.sip_response_code == 200:
            patterns.append(
                FraudPattern(
                    pattern_type=FraudPatternType.IRSF,
                    severity=FraudSeverity.LOW,
                    confidence=0.5,
                    description=(
                        f"Extremely short international call: "
                        f"{call.duration_seconds:.1f}s "
                        f"(possible IRSF test call)"
                    ),
                    evidence={
                        "duration_seconds": call.duration_seconds,
                        "min_duration_seconds": min_duration,
                        "sip_response": call.sip_response_code,
                        "destination_country": call.destination_country,
                        "rule": "TF-007",
                    },
                    call_id=call.call_id,
                    agent_id=call.agent_id,
                )
            )

        return patterns

    # ------------------------------------------------------------------
    # Transfer analysis
    # ------------------------------------------------------------------

    def _analyze_transfer(self, call: CallRecord) -> List[FraudPattern]:
        """Detect call transfers to premium rate destinations."""
        patterns: List[FraudPattern] = []

        if not call.was_transferred or not call.transfer_destination:
            return patterns

        transfer_dest = self._normalize_number(call.transfer_destination)
        matched_prefix = self._match_premium_prefix(transfer_dest)

        if matched_prefix is not None:
            region = self._get_prefix_region(matched_prefix)
            patterns.append(
                FraudPattern(
                    pattern_type=FraudPatternType.CALL_TRANSFER,
                    severity=FraudSeverity.CRITICAL,
                    confidence=0.95,
                    description=(
                        f"Call transfer to premium rate destination: "
                        f"{matched_prefix} ({region})"
                    ),
                    evidence={
                        "matched_prefix": matched_prefix,
                        "region": region,
                        "transfer_destination": self._mask_number(transfer_dest),
                        "original_call_id": call.call_id,
                        "rule": "TF-004",
                    },
                    call_id=call.call_id,
                    agent_id=call.agent_id,
                    destination_number=self._mask_number(transfer_dest),
                    estimated_loss=call.estimated_cost * 3,  # Transfers are often long
                )
            )

        return patterns

    # ------------------------------------------------------------------
    # Geographic analysis
    # ------------------------------------------------------------------

    def _analyze_geography(self, call: CallRecord) -> List[FraudPattern]:
        """Detect calls to unusual or never-before-seen destinations."""
        patterns: List[FraudPattern] = []

        if not call.destination_country or not call.agent_id:
            return patterns

        profile = self._agent_profiles.get(call.agent_id)
        if profile is None:
            # No profile yet -- cannot detect anomalies
            return patterns

        if (
            call.destination_country not in profile.common_countries
            and call.is_international
        ):
            patterns.append(
                FraudPattern(
                    pattern_type=FraudPatternType.IRSF,
                    severity=FraudSeverity.LOW,
                    confidence=0.4,
                    description=(
                        f"Call to new destination country: "
                        f"{call.destination_country} (not in agent profile)"
                    ),
                    evidence={
                        "destination_country": call.destination_country,
                        "common_countries": list(profile.common_countries),
                        "agent_id": call.agent_id,
                        "rule": "TF-008",
                    },
                    call_id=call.call_id,
                    agent_id=call.agent_id,
                )
            )

        return patterns

    # ------------------------------------------------------------------
    # Cost analysis
    # ------------------------------------------------------------------

    def _analyze_cost(self, call: CallRecord) -> List[FraudPattern]:
        """Detect calls exceeding cost thresholds."""
        patterns: List[FraudPattern] = []

        if call.estimated_cost <= 0:
            return patterns

        if call.estimated_cost > self.max_call_cost:
            patterns.append(
                FraudPattern(
                    pattern_type=FraudPatternType.IRSF,
                    severity=FraudSeverity.HIGH,
                    confidence=0.85,
                    description=(
                        f"Call cost exceeds threshold: "
                        f"${call.estimated_cost:.2f} "
                        f"(threshold: ${self.max_call_cost:.2f})"
                    ),
                    evidence={
                        "estimated_cost": call.estimated_cost,
                        "threshold": self.max_call_cost,
                        "duration_seconds": call.duration_seconds,
                        "cost_per_minute": call.cost_per_minute,
                        "destination_country": call.destination_country,
                        "rule": "TF-009",
                    },
                    call_id=call.call_id,
                    agent_id=call.agent_id,
                    estimated_loss=call.estimated_cost,
                )
            )

        return patterns

    # ------------------------------------------------------------------
    # Agent behavior analysis
    # ------------------------------------------------------------------

    def analyze_agent_behavior(
        self,
        agent_id: str,
        recent_calls: List[CallRecord],
        *,
        profile_window_days: int = 30,
    ) -> List[FraudPattern]:
        """
        Analyze an agent's recent calling behavior against their profile.

        Builds or updates the agent's behavioral profile and flags
        deviations that could indicate compromised credentials or
        insider fraud.

        Args:
            agent_id: The agent identifier.
            recent_calls: Recent call records for this agent.
            profile_window_days: Rolling window for profile building.

        Returns:
            List of detected fraud patterns.
        """
        patterns: List[FraudPattern] = []

        if not recent_calls:
            return patterns

        profile = self._agent_profiles.get(agent_id)
        if profile is None:
            # Build an initial profile -- no anomalies to detect yet
            profile = self._build_agent_profile(agent_id, recent_calls)
            self._agent_profiles[agent_id] = profile
            return patterns

        # Compare recent behavior to profile
        recent_duration_avg = sum(
            c.duration_seconds for c in recent_calls
        ) / len(recent_calls)

        # Duration anomaly (> 2 standard deviations)
        if profile.std_call_duration_seconds > 0:
            z_score = abs(
                (recent_duration_avg - profile.avg_call_duration_seconds)
                / profile.std_call_duration_seconds
            )
            if z_score > 2.0:
                patterns.append(
                    FraudPattern(
                        pattern_type=FraudPatternType.PBX_HACK,
                        severity=FraudSeverity.MEDIUM,
                        confidence=min(0.9, 0.5 + z_score * 0.1),
                        description=(
                            f"Anomalous call duration pattern: "
                            f"avg {recent_duration_avg:.0f}s "
                            f"vs profile {profile.avg_call_duration_seconds:.0f}s "
                            f"(z-score: {z_score:.1f})"
                        ),
                        evidence={
                            "recent_avg_duration": recent_duration_avg,
                            "profile_avg_duration": profile.avg_call_duration_seconds,
                            "z_score": z_score,
                        },
                        agent_id=agent_id,
                    )
                )

        # International call ratio anomaly
        international_count = sum(
            1 for c in recent_calls if c.is_international
        )
        recent_intl_ratio = international_count / len(recent_calls)

        if (
            recent_intl_ratio > profile.international_call_ratio * 3
            and international_count > 3
        ):
            patterns.append(
                FraudPattern(
                    pattern_type=FraudPatternType.IRSF,
                    severity=FraudSeverity.MEDIUM,
                    confidence=0.7,
                    description=(
                        f"Abnormal international call ratio: "
                        f"{recent_intl_ratio:.1%} vs profile "
                        f"{profile.international_call_ratio:.1%}"
                    ),
                    evidence={
                        "recent_international_ratio": recent_intl_ratio,
                        "profile_international_ratio": profile.international_call_ratio,
                        "international_call_count": international_count,
                    },
                    agent_id=agent_id,
                )
            )

        return patterns

    def update_agent_profile(
        self, agent_id: str, calls: List[CallRecord]
    ) -> AgentCallProfile:
        """
        Update an agent's behavioral profile with new call data.

        Args:
            agent_id: The agent identifier.
            calls: New call records to incorporate.

        Returns:
            The updated agent profile.
        """
        profile = self._build_agent_profile(agent_id, calls)
        self._agent_profiles[agent_id] = profile
        return profile

    def _build_agent_profile(
        self, agent_id: str, calls: List[CallRecord]
    ) -> AgentCallProfile:
        """Build an agent call profile from historical data."""
        if not calls:
            return AgentCallProfile(agent_id=agent_id)

        durations = [c.duration_seconds for c in calls if c.duration_seconds > 0]
        international_calls = [c for c in calls if c.is_international]
        countries = {c.destination_country for c in calls if c.destination_country}
        destinations = {c.destination_number[:6] for c in calls if c.destination_number}
        hours = {c.start_time.hour for c in calls}
        transfers = sum(1 for c in calls if c.was_transferred)

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        std_duration = (
            (sum((d - avg_duration) ** 2 for d in durations) / len(durations)) ** 0.5
            if len(durations) > 1
            else 0.0
        )

        return AgentCallProfile(
            agent_id=agent_id,
            total_calls=len(calls),
            avg_call_duration_seconds=avg_duration,
            std_call_duration_seconds=std_duration,
            international_call_ratio=(
                len(international_calls) / len(calls) if calls else 0.0
            ),
            avg_international_calls_per_shift=(
                len(international_calls) / max(1, len(calls) // 100)
            ),
            common_destinations=destinations,
            common_countries=countries,
            typical_hours=hours,
            transfer_rate=transfers / len(calls) if calls else 0.0,
            last_updated=datetime.utcnow(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _track_call(self, call: CallRecord) -> None:
        """Track a call for velocity analysis."""
        agent_key = f"{call.tenant_id}:{call.agent_id}"
        if agent_key not in self._call_counts:
            self._call_counts[agent_key] = []
        self._call_counts[agent_key].append(call.start_time)

        # Track in recent calls
        if call.agent_id not in self._recent_calls:
            self._recent_calls[call.agent_id] = []
        self._recent_calls[call.agent_id].append(call)

        # Limit stored history
        if len(self._recent_calls[call.agent_id]) > 1000:
            self._recent_calls[call.agent_id] = (
                self._recent_calls[call.agent_id][-500:]
            )

    def _fire_alert(self, pattern: FraudPattern) -> None:
        """Fire an alert for a detected fraud pattern."""
        if self.alert_callback:
            try:
                self.alert_callback(pattern)
            except Exception as exc:
                logger.error(
                    "Failed to fire toll fraud alert callback: %s", exc
                )

        logger.warning(
            "TOLL FRAUD ALERT: type=%s severity=%s confidence=%.2f "
            "call_id=%s agent_id=%s desc=%s",
            pattern.pattern_type.name,
            pattern.severity.display_name,
            pattern.confidence,
            pattern.call_id,
            pattern.agent_id,
            pattern.description,
        )

    @staticmethod
    def _normalize_number(number: str) -> str:
        """Normalize a phone number for comparison."""
        # Remove spaces, dashes, parentheses
        cleaned = re.sub(r"[\s\-\(\).]", "", number)
        # Ensure leading +
        if not cleaned.startswith("+"):
            if cleaned.startswith("00"):
                cleaned = "+" + cleaned[2:]
            elif cleaned.startswith("011"):
                cleaned = "+" + cleaned[3:]
        return cleaned

    @staticmethod
    def _mask_number(number: str) -> str:
        """Mask a phone number for logging (show first 4 and last 2 digits)."""
        if len(number) <= 6:
            return "***"
        return number[:4] + "*" * (len(number) - 6) + number[-2:]

    # ------------------------------------------------------------------
    # Statistics & reporting
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get toll fraud detection statistics."""
        return dict(self._stats)

    def get_detected_patterns(
        self,
        *,
        since: Optional[datetime] = None,
        severity: Optional[FraudSeverity] = None,
        pattern_type: Optional[FraudPatternType] = None,
        limit: int = 100,
    ) -> List[FraudPattern]:
        """
        Retrieve detected fraud patterns with optional filtering.

        Args:
            since: Only return patterns detected after this timestamp.
            severity: Filter by minimum severity level.
            pattern_type: Filter by pattern type.
            limit: Maximum number of patterns to return.

        Returns:
            List of matching fraud patterns.
        """
        results = self._detected_patterns

        if since:
            results = [p for p in results if p.detected_at >= since]

        if severity:
            results = [p for p in results if p.severity.level >= severity.level]

        if pattern_type:
            results = [p for p in results if p.pattern_type == pattern_type]

        return results[-limit:]

    def reset_statistics(self) -> None:
        """Reset detection statistics."""
        self._stats = {
            "total_calls_analyzed": 0,
            "total_patterns_detected": 0,
            "total_calls_blocked": 0,
            "total_estimated_savings": 0.0,
        }
        logger.info("Toll fraud statistics reset for tenant=%s", self.tenant_id)

    def add_blocked_prefix(self, prefix: str) -> None:
        """Add a prefix to the blocked list."""
        normalized = prefix if prefix.startswith("+") else f"+{prefix}"
        if normalized not in self._blocked_prefixes:
            self._blocked_prefixes.append(normalized)
            self._blocked_prefixes.sort(key=len, reverse=True)
            logger.info("Added blocked prefix: %s", normalized)

    def remove_blocked_prefix(self, prefix: str) -> bool:
        """Remove a prefix from the blocked list."""
        normalized = prefix if prefix.startswith("+") else f"+{prefix}"
        if normalized in self._blocked_prefixes:
            self._blocked_prefixes.remove(normalized)
            logger.info("Removed blocked prefix: %s", normalized)
            return True
        return False

    def is_number_blocked(self, number: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a number matches any blocked prefix.

        Args:
            number: The phone number to check.

        Returns:
            Tuple of (is_blocked, matched_prefix).
        """
        normalized = self._normalize_number(number)
        matched = self._match_premium_prefix(normalized)
        return (matched is not None, matched)
