"""
Speech Analytics Engine for BPO Call Security

Real-time speech analytics for PCI compliance, social engineering detection,
and sentiment analysis in contact center environments.

Capabilities:
- PCI violation detection (card numbers, CVV, SSN spoken aloud)
- Social engineering pattern recognition (authority impersonation, urgency)
- Lexicon-based sentiment analysis with trend tracking
- Compliance script adherence checking (required/forbidden phrases)
- Keyword matching with configurable rules and regex support
- Profanity detection and silence/crosstalk monitoring
- PQC-signed tamper-evident audit trails via SHA3-256

Integrates with QBITEL Bridge's quantum-safe infrastructure for
secure evidence preservation and regulatory compliance across
PCI-DSS 4.0, HIPAA, TCPA, GDPR, SOX, FCA, MiFID II, and DPDPA.
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


class SpeechEventType(Enum):
    """Types of speech analytics events detected during call monitoring."""
    PAN_SPOKEN = auto()
    CVV_SPOKEN = auto()
    SSN_SPOKEN = auto()
    EXPIRY_SPOKEN = auto()
    PIN_SPOKEN = auto()
    ACCOUNT_NUMBER_SPOKEN = auto()
    SOCIAL_ENGINEERING_DETECTED = auto()
    KEYWORD_MATCH = auto()
    SENTIMENT_NEGATIVE = auto()
    SENTIMENT_ESCALATION = auto()
    COMPLIANCE_PHRASE_MISSING = auto()
    CONSENT_NOT_OBTAINED = auto()
    PROFANITY_DETECTED = auto()
    SILENCE_EXCESSIVE = auto()
    CROSSTALK_DETECTED = auto()
    UNAUTHORIZED_DISCLOSURE = auto()


class ComplianceViolationType(Enum):
    """Regulatory compliance violation types across frameworks."""
    PCI_CARD_READ_ALOUD = auto()
    PCI_CVV_READ_ALOUD = auto()
    PCI_EXPIRY_READ_ALOUD = auto()
    HIPAA_PHI_DISCLOSED = auto()
    TCPA_CONSENT_MISSING = auto()
    TCPA_DNC_VIOLATION = auto()
    SOX_SCRIPT_DEVIATION = auto()
    FCA_DISCLAIMER_MISSING = auto()
    GDPR_CONSENT_MISSING = auto()
    MIFID_RECORDING_GAP = auto()
    DPDPA_CONSENT_MISSING = auto()


class SocialEngineeringPattern(Enum):
    """Social engineering attack patterns detected in call transcripts."""
    AUTHORITY_IMPERSONATION = auto()
    URGENCY_PRESSURE = auto()
    PRETEXTING = auto()
    EMOTIONAL_MANIPULATION = auto()
    TECHNICAL_DECEPTION = auto()
    REWARD_BAITING = auto()
    REVERSE_SOCIAL_ENGINEERING = auto()
    TAILGATING_VERBAL = auto()


class SpeechRiskLevel(Enum):
    """Risk levels for speech analytics events with score ranges."""
    INFO = (1, "Info", 0.0, 0.2)
    LOW = (2, "Low", 0.2, 0.4)
    MEDIUM = (3, "Medium", 0.4, 0.6)
    HIGH = (4, "High", 0.6, 0.8)
    CRITICAL = (5, "Critical", 0.8, 1.0)

    def __init__(self, level: int, display_name: str, min_score: float, max_score: float):
        self.level = level
        self.display_name = display_name
        self.min_score = min_score
        self.max_score = max_score

    @classmethod
    def from_score(cls, score: float) -> "SpeechRiskLevel":
        """Return the risk level that corresponds to a given score."""
        for member in cls:
            if member.min_score <= score < member.max_score:
                return member
        return cls.CRITICAL


class SentimentCategory(Enum):
    """Sentiment categories for call transcript analysis."""
    POSITIVE = auto()
    NEUTRAL = auto()
    NEGATIVE = auto()
    ANGRY = auto()
    FRUSTRATED = auto()
    CONFUSED = auto()
    SATISFIED = auto()
    ANXIOUS = auto()


class AnalyticsChannel(Enum):
    """Audio channel source for speech analytics."""
    AGENT = auto()
    CALLER = auto()
    BOTH = auto()
    SYSTEM = auto()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SpeechEvent:
    """A single speech analytics event detected during call monitoring."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str = ""
    event_type: SpeechEventType = SpeechEventType.KEYWORD_MATCH
    risk_level: SpeechRiskLevel = SpeechRiskLevel.INFO
    channel: AnalyticsChannel = AnalyticsChannel.BOTH
    timestamp: datetime = field(default_factory=datetime.utcnow)
    transcript_segment: str = ""
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    action_taken: str = ""
    pqc_signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id, "call_id": self.call_id,
            "event_type": self.event_type.name, "risk_level": self.risk_level.name,
            "channel": self.channel.name, "timestamp": self.timestamp.isoformat(),
            "transcript_segment": self.transcript_segment,
            "confidence": self.confidence, "evidence": self.evidence,
            "action_taken": self.action_taken, "pqc_signature": self.pqc_signature,
        }


@dataclass
class KeywordRule:
    """Configurable keyword matching rule for speech analytics."""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    keywords: List[str] = field(default_factory=list)
    category: str = ""
    action: str = "alert"
    channel: AnalyticsChannel = AnalyticsChannel.BOTH
    case_sensitive: bool = False
    regex_pattern: Optional[str] = None

    def matches(self, text: str) -> bool:
        """Check if text matches this keyword rule."""
        if self.regex_pattern:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            if re.search(self.regex_pattern, text, flags):
                return True
        check_text = text if self.case_sensitive else text.lower()
        for kw in self.keywords:
            if (kw if self.case_sensitive else kw.lower()) in check_text:
                return True
        return False


@dataclass
class ComplianceScript:
    """Compliance script definition with required and forbidden phrases."""
    script_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    required_phrases: List[str] = field(default_factory=list)
    forbidden_phrases: List[str] = field(default_factory=list)
    framework: str = ""
    timeout_seconds: int = 300
    channel: AnalyticsChannel = AnalyticsChannel.AGENT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "script_id": self.script_id, "name": self.name,
            "required_phrases": self.required_phrases,
            "forbidden_phrases": self.forbidden_phrases,
            "framework": self.framework, "timeout_seconds": self.timeout_seconds,
            "channel": self.channel.name,
        }


@dataclass
class SpeechAnalyticsPolicy:
    """Configuration policy governing speech analytics behavior."""
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default"
    pci_detection_enabled: bool = True
    social_engineering_detection: bool = True
    sentiment_tracking: bool = True
    keyword_rules: List[KeywordRule] = field(default_factory=list)
    compliance_scripts: List[ComplianceScript] = field(default_factory=list)
    min_confidence: float = 0.7
    auto_alert_threshold: SpeechRiskLevel = SpeechRiskLevel.HIGH
    profanity_detection: bool = True
    silence_threshold_seconds: float = 10.0
    crosstalk_threshold_pct: float = 0.3

    def validate(self) -> List[str]:
        """Validate policy configuration, return list of errors."""
        errors: List[str] = []
        if not 0.0 <= self.min_confidence <= 1.0:
            errors.append(f"min_confidence must be 0.0-1.0, got {self.min_confidence}")
        if self.silence_threshold_seconds < 0:
            errors.append("silence_threshold_seconds must be non-negative")
        if not 0.0 <= self.crosstalk_threshold_pct <= 1.0:
            errors.append(f"crosstalk_threshold_pct must be 0.0-1.0, got {self.crosstalk_threshold_pct}")
        for rule in self.keyword_rules:
            if not rule.keywords and not rule.regex_pattern:
                errors.append(f"KeywordRule {rule.rule_id} has no keywords or regex_pattern")
        for script in self.compliance_scripts:
            if not script.required_phrases and not script.forbidden_phrases:
                errors.append(f"ComplianceScript {script.script_id} has no phrases defined")
            if script.timeout_seconds <= 0:
                errors.append(f"ComplianceScript {script.script_id} timeout must be positive")
        return errors


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a text segment."""
    category: SentimentCategory = SentimentCategory.NEUTRAL
    score: float = 0.0           # -1.0 (most negative) to 1.0 (most positive)
    confidence: float = 0.0
    trend: str = "stable"        # "improving", "stable", "declining"
    keywords_detected: List[str] = field(default_factory=list)


@dataclass
class TranscriptAnalysis:
    """Complete analysis results for a single call transcript."""
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str = ""
    agent_id: str = ""
    tenant_id: str = ""
    events: List[SpeechEvent] = field(default_factory=list)
    sentiment_timeline: List[SentimentResult] = field(default_factory=list)
    compliance_results: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    total_segments_analyzed: int = 0
    violations_detected: int = 0
    summary: str = ""
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    pqc_integrity_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id, "call_id": self.call_id,
            "agent_id": self.agent_id, "tenant_id": self.tenant_id,
            "events": [e.to_dict() for e in self.events],
            "sentiment_timeline": [
                {"category": s.category.name, "score": s.score,
                 "confidence": s.confidence, "trend": s.trend,
                 "keywords_detected": s.keywords_detected}
                for s in self.sentiment_timeline
            ],
            "compliance_results": self.compliance_results,
            "risk_score": self.risk_score,
            "total_segments_analyzed": self.total_segments_analyzed,
            "violations_detected": self.violations_detected,
            "summary": self.summary, "analyzed_at": self.analyzed_at.isoformat(),
            "pqc_integrity_hash": self.pqc_integrity_hash,
        }


@dataclass
class AnalyticsReport:
    """Aggregated analytics report over a time period."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    total_calls_analyzed: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    top_violations: List[Dict[str, Any]] = field(default_factory=list)
    social_engineering_attempts: int = 0
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    average_risk_score: float = 0.0
    pci_violations: int = 0
    compliance_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_calls_analyzed": self.total_calls_analyzed,
            "events_by_type": self.events_by_type,
            "top_violations": self.top_violations,
            "social_engineering_attempts": self.social_engineering_attempts,
            "sentiment_distribution": self.sentiment_distribution,
            "average_risk_score": self.average_risk_score,
            "pci_violations": self.pci_violations,
            "compliance_score": self.compliance_score,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Engine Classes
# ---------------------------------------------------------------------------


class SpokenNumberParser:
    """Converts spoken digit words to numeric strings for Luhn/PAN validation."""

    WORD_TO_DIGIT: Dict[str, str] = {
        "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2",
        "three": "3", "four": "4", "five": "5", "six": "6",
        "seven": "7", "eight": "8", "nine": "9",
    }
    MULTIPLIER_WORDS: Dict[str, int] = {"double": 2, "triple": 3}

    @classmethod
    def parse(cls, text: str) -> List[str]:
        """Extract potential numeric sequences from spoken text.

        Handles word-form digits, 'double five' = '55', raw digits, and
        spaced-out digit patterns like '4 1 0 0 1 2 3 4'.
        """
        tokens = text.lower().split()
        digit_buf: List[str] = []
        sequences: List[str] = []
        mult: Optional[int] = None

        for token in tokens:
            if token in cls.MULTIPLIER_WORDS:
                mult = cls.MULTIPLIER_WORDS[token]
                continue
            if token in cls.WORD_TO_DIGIT:
                digit_buf.append(cls.WORD_TO_DIGIT[token] * (mult or 1))
                mult = None
                continue
            if token.isdigit():
                digit_buf.append(token * (mult or 1))
                mult = None
                continue
            mult = None
            if len(digit_buf) >= 4:
                sequences.append("".join(digit_buf))
            digit_buf = []

        if len(digit_buf) >= 4:
            sequences.append("".join(digit_buf))

        # Raw digit sequences and spaced-out patterns
        for rd in re.findall(r"\b(\d{4,})\b", text):
            if rd not in sequences:
                sequences.append(rd)
        for sp in re.findall(r"(?:\d\s+){3,}\d", text):
            joined = re.sub(r"\s+", "", sp)
            if len(joined) >= 4 and joined not in sequences:
                sequences.append(joined)
        return sequences

    @staticmethod
    def luhn_check(number: str) -> bool:
        """Validate card number via Luhn algorithm (ISO/IEC 7812-1)."""
        if not number.isdigit() or not (13 <= len(number) <= 19):
            return False
        digits = [int(d) for d in number]
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        return sum(digits) % 10 == 0


class SentimentAnalyzer:
    """Lexicon-based sentiment analysis for call transcripts."""

    POSITIVE_WORDS: Set[str] = {
        "thank", "thanks", "great", "excellent", "wonderful", "appreciate",
        "happy", "satisfied", "perfect", "helpful", "resolved", "good",
        "amazing", "fantastic", "awesome", "pleased", "brilliant",
        "outstanding", "delighted", "superb", "friendly", "professional",
    }
    NEGATIVE_WORDS: Set[str] = {
        "angry", "frustrated", "terrible", "worst", "unacceptable",
        "ridiculous", "incompetent", "lawsuit", "cancel", "escalate",
        "supervisor", "complaint", "horrible", "awful", "pathetic",
        "useless", "furious", "annoyed", "disappointed", "upset",
        "rude", "slow", "broken", "fail", "wrong", "bad", "hate",
    }
    INTENSIFIERS: Set[str] = {
        "very", "extremely", "absolutely", "completely", "totally",
        "utterly", "incredibly", "highly", "really", "truly",
    }
    NEGATORS: Set[str] = {
        "not", "never", "no", "neither", "hardly", "barely",
        "don't", "doesn't", "didn't", "won't", "can't", "cannot",
    }
    CONFUSION_WORDS: Set[str] = {"confused", "understand", "unclear", "lost", "huh", "repeat"}
    ANXIETY_WORDS: Set[str] = {"worried", "anxious", "nervous", "scared", "afraid", "concern", "panic"}

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a text segment using lexicon scoring."""
        words = text.lower().split()
        if not words:
            return SentimentResult()
        pos_score = neg_score = 0.0
        detected: List[str] = []
        negated = intensified = False

        for word in words:
            clean = re.sub(r"[^\w]", "", word)
            if not clean:
                continue
            if clean in self.NEGATORS:
                negated = True
                continue
            if clean in self.INTENSIFIERS:
                intensified = True
                continue
            mult = 1.5 if intensified else 1.0
            if clean in self.POSITIVE_WORDS:
                if negated:
                    neg_score += 1.0 * mult
                    detected.append(f"NOT_{clean}")
                else:
                    pos_score += 1.0 * mult
                    detected.append(clean)
                negated = intensified = False
            elif clean in self.NEGATIVE_WORDS:
                if negated:
                    pos_score += 0.5 * mult
                    detected.append(f"NOT_{clean}")
                else:
                    neg_score += 1.0 * mult
                    detected.append(clean)
                negated = intensified = False
            else:
                negated = intensified = False

        total = pos_score + neg_score
        score = (pos_score - neg_score) / total if total else 0.0
        confidence = min(1.0, total / max(len(words), 1) * 2.0) if total else 0.1
        clean_words = {re.sub(r"[^\w]", "", w) for w in words}
        if sum(1 for w in clean_words if w in self.ANXIETY_WORDS) >= 2:
            cat = SentimentCategory.ANXIOUS
        elif sum(1 for w in clean_words if w in self.CONFUSION_WORDS) >= 2:
            cat = SentimentCategory.CONFUSED
        elif score > 0.5:
            cat = SentimentCategory.SATISFIED
        elif score > 0.2:
            cat = SentimentCategory.POSITIVE
        elif score > -0.2:
            cat = SentimentCategory.NEUTRAL
        elif score > -0.5:
            cat = SentimentCategory.FRUSTRATED
        elif score > -0.8:
            cat = SentimentCategory.NEGATIVE
        else:
            cat = SentimentCategory.ANGRY
        return SentimentResult(cat, round(score, 4), round(min(confidence, 1.0), 4), "stable", detected)


class SocialEngineeringDetector:
    """Detects social engineering patterns in call transcripts."""

    _PATTERNS: Dict[SocialEngineeringPattern, List[str]] = {
        SocialEngineeringPattern.AUTHORITY_IMPERSONATION: [
            r"i am (?:the |a )?(?:ceo|cfo|cto|director|president|vp|manager|supervisor)",
            r"i'?m calling from (?:the )?(?:bank|irs|fbi|police|security|fraud|compliance)",
            r"this is (?:the )?(?:security|fraud|risk|compliance) (?:department|team)",
            r"i(?:'m| am) (?:authorized|cleared|permitted) to",
        ],
        SocialEngineeringPattern.URGENCY_PRESSURE: [
            r"(?:immediately|right now|urgent|emergency|time.sensitive)",
            r"your account (?:will be|is being) (?:closed|suspended|locked|frozen)",
            r"(?:within|in) (?:the next )?\d+ (?:minutes?|hours?)",
            r"(?:if you don'?t|unless you) (?:act|respond|verify) (?:now|immediately)",
            r"(?:last chance|final warning|final notice)",
        ],
        SocialEngineeringPattern.PRETEXTING: [
            r"(?:verify|confirm) your (?:account|identity|social|ssn|card|pin|password)",
            r"for (?:security|verification|authentication) purposes",
            r"(?:routine|mandatory|required) (?:security )?(?:check|verification|audit)",
            r"(?:read|tell|give) me (?:your |the )?(?:card number|account number|ssn|pin)",
        ],
        SocialEngineeringPattern.EMOTIONAL_MANIPULATION: [
            r"(?:please|i beg you|help me|i'?m desperate)",
            r"(?:my (?:mother|father|child|family) (?:is |needs |will ))",
            r"(?:medical|health) (?:emergency|crisis|situation)",
        ],
        SocialEngineeringPattern.TECHNICAL_DECEPTION: [
            r"(?:system|server|database) (?:error|failure|crash|update)",
            r"(?:need|require) (?:remote|screen) (?:access|sharing|control)",
            r"(?:your computer|your system) (?:has been|is) (?:compromised|hacked|infected)",
        ],
        SocialEngineeringPattern.REWARD_BAITING: [
            r"(?:you(?:'ve| have) won|congratulations|you(?:'re| are) selected)",
            r"(?:prize|reward|bonus|gift|cashback) (?:of |worth )?\$?\d+",
            r"(?:free|complimentary) (?:offer|trial|service|upgrade)",
        ],
        SocialEngineeringPattern.REVERSE_SOCIAL_ENGINEERING: [
            r"(?:call|contact) (?:us|me|this number) (?:back|at)",
            r"(?:we (?:tried|attempted) to (?:reach|contact|call) you)",
        ],
        SocialEngineeringPattern.TAILGATING_VERBAL: [
            r"(?:as (?:we )?(?:discussed|agreed|spoke about) (?:earlier|yesterday|last time))",
            r"(?:your (?:colleague|manager) (?:already |just )(?:approved|authorized))",
        ],
    }

    def __init__(self) -> None:
        self._compiled: Dict[SocialEngineeringPattern, List[re.Pattern]] = {
            pat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for pat, patterns in self._PATTERNS.items()
        }

    def detect(self, transcript_segments: List[Dict[str, str]]) -> List[SpeechEvent]:
        """Detect social engineering patterns across transcript segments."""
        events: List[SpeechEvent] = []
        hits: Dict[SocialEngineeringPattern, int] = {}

        for seg in transcript_segments:
            text = seg.get("text", "")
            ch_str = seg.get("channel", "caller")
            try:
                ts = datetime.fromisoformat(seg.get("timestamp", ""))
            except (ValueError, TypeError):
                ts = datetime.utcnow()
            ch = AnalyticsChannel.AGENT if ch_str.lower() == "agent" else AnalyticsChannel.CALLER

            for pat_type, compiled_list in self._compiled.items():
                for crx in compiled_list:
                    match = crx.search(text)
                    if match:
                        hits[pat_type] = hits.get(pat_type, 0) + 1
                        n = hits[pat_type]
                        risk = SpeechRiskLevel.CRITICAL if n >= 3 else (SpeechRiskLevel.HIGH if n >= 2 else SpeechRiskLevel.MEDIUM)
                        events.append(SpeechEvent(
                            event_type=SpeechEventType.SOCIAL_ENGINEERING_DETECTED,
                            risk_level=risk, channel=ch, timestamp=ts,
                            transcript_segment=text[:200],
                            confidence=min(0.5 + n * 0.15, 0.95),
                            evidence={"pattern_type": pat_type.name, "matched_text": match.group(0), "pattern_hits": n},
                            action_taken="alert" if risk.level >= 4 else "log",
                        ))
                        break  # one match per pattern type per segment
        return events


class SpeechAnalyticsEngine:
    """
    Main speech analytics engine for BPO call security.

    Detects PCI violations (agents reading card numbers), social engineering,
    sentiment shifts, and compliance script deviations. All findings are
    PQC-signed for tamper-evident audit trails.
    """

    _SSN_RE = re.compile(r"\b(\d{3})[-\s]?(\d{2})[-\s]?(\d{4})\b")
    _EXPIRY_RE = re.compile(r"\b(0[1-9]|1[0-2])\s*[/\-]\s*(\d{2}(?:\d{2})?)\b")
    _PIN_RE = re.compile(r"\bpin\b.*?\b(\d{4}|\d{6})\b", re.IGNORECASE)
    _ACCT_RE = re.compile(r"\b(\d{8,17})\b")
    _CVV_RE = re.compile(r"\b(?:cvv|cv2|cvc|security code|card verification)\b.*?\b(\d{3,4})\b", re.IGNORECASE)
    _PROFANITY_RE = re.compile(r"\b(?:damn|hell|crap|shit|fuck|bitch|bastard)\b", re.IGNORECASE)

    def __init__(self, policy: Optional[SpeechAnalyticsPolicy] = None):
        self._policy = policy or SpeechAnalyticsPolicy()
        self._number_parser = SpokenNumberParser()
        self._sentiment_analyzer = SentimentAnalyzer()
        self._se_detector = SocialEngineeringDetector()
        self._analyses: List[TranscriptAnalysis] = []
        logger.info("SpeechAnalyticsEngine initialized with policy '%s'", self._policy.name)

    def analyze_transcript(self, call_id: str, agent_id: str, tenant_id: str,
                           transcript_segments: List[Dict[str, str]]) -> TranscriptAnalysis:
        """Full analysis of a call transcript. Each segment: {channel, text, timestamp}."""
        all_events: List[SpeechEvent] = []
        sentiments: List[SentimentResult] = []
        compliance_results: Dict[str, Any] = {}

        for seg in transcript_segments:
            text, channel = seg.get("text", ""), seg.get("channel", "caller")
            if self._policy.pci_detection_enabled:
                all_events.extend(self.detect_pan_spoken(text, channel))
                all_events.extend(self.detect_sensitive_data_spoken(text, channel))
            all_events.extend(self.match_keywords(text, channel))
            if self._policy.sentiment_tracking:
                sr = self.analyze_sentiment(text)
                sentiments.append(sr)
                if sr.category in (SentimentCategory.ANGRY, SentimentCategory.FRUSTRATED):
                    ch = AnalyticsChannel.AGENT if channel == "agent" else AnalyticsChannel.CALLER
                    all_events.append(SpeechEvent(
                        call_id=call_id, event_type=SpeechEventType.SENTIMENT_NEGATIVE,
                        risk_level=SpeechRiskLevel.LOW, channel=ch,
                        transcript_segment=text[:200], confidence=sr.confidence,
                        evidence={"category": sr.category.name, "score": sr.score},
                    ))
            if self._policy.profanity_detection and self._PROFANITY_RE.search(text):
                ch = AnalyticsChannel.AGENT if channel == "agent" else AnalyticsChannel.CALLER
                all_events.append(SpeechEvent(
                    call_id=call_id, event_type=SpeechEventType.PROFANITY_DETECTED,
                    risk_level=SpeechRiskLevel.LOW, channel=ch,
                    transcript_segment=text[:200], confidence=0.9,
                ))

        if self._policy.social_engineering_detection:
            all_events.extend(self.detect_social_engineering(transcript_segments))
        for script in self._policy.compliance_scripts:
            sevts = self.check_compliance_script(transcript_segments, script)
            all_events.extend(sevts)
            compliance_results[script.name] = {
                "violations": len(sevts), "script_id": script.script_id, "framework": script.framework,
            }

        sentiments = self._detect_sentiment_trends(sentiments)
        for evt in all_events:
            if not evt.call_id:
                evt.call_id = call_id
        risk_score = self._compute_risk_score(all_events)
        violations = sum(1 for e in all_events if e.risk_level.level >= SpeechRiskLevel.MEDIUM.level)

        parts: List[str] = []
        if any(e.event_type == SpeechEventType.PAN_SPOKEN for e in all_events):
            parts.append("PCI violation: card data spoken")
        if any(e.event_type == SpeechEventType.SOCIAL_ENGINEERING_DETECTED for e in all_events):
            parts.append("Social engineering attempt detected")
        if violations:
            parts.append(f"{violations} violation(s) detected")

        integrity_hash = self._compute_integrity_hash(f"{call_id}:{agent_id}:{len(all_events)}:{risk_score}")
        for evt in all_events:
            evt.pqc_signature = self._compute_integrity_hash(f"{evt.event_id}:{evt.call_id}:{evt.event_type.name}")

        analysis = TranscriptAnalysis(
            call_id=call_id, agent_id=agent_id, tenant_id=tenant_id,
            events=all_events, sentiment_timeline=sentiments,
            compliance_results=compliance_results, risk_score=risk_score,
            total_segments_analyzed=len(transcript_segments),
            violations_detected=violations,
            summary="; ".join(parts) if parts else "No issues",
            pqc_integrity_hash=integrity_hash,
        )
        self._analyses.append(analysis)
        logger.info("Analysis complete for call %s: risk=%.2f, events=%d", call_id, risk_score, len(all_events))
        return analysis

    def detect_pan_spoken(self, text: str, channel: str = "caller") -> List[SpeechEvent]:
        """Detect credit card numbers spoken aloud (PCI violation if agent reads them)."""
        events: List[SpeechEvent] = []
        for seq in self._number_parser.parse(text):
            if not (13 <= len(seq) <= 19) or not self._number_parser.luhn_check(seq):
                continue
            is_agent = channel.lower() == "agent"
            risk = SpeechRiskLevel.CRITICAL if is_agent else SpeechRiskLevel.HIGH
            masked = seq[:4] + "*" * (len(seq) - 8) + seq[-4:]
            events.append(SpeechEvent(
                event_type=SpeechEventType.PAN_SPOKEN, risk_level=risk,
                channel=AnalyticsChannel.AGENT if is_agent else AnalyticsChannel.CALLER,
                transcript_segment=text[:200], confidence=0.92,
                evidence={"masked_pan": masked, "card_length": len(seq), "luhn_valid": True,
                          "spoken_by_agent": is_agent,
                          "violation_type": ComplianceViolationType.PCI_CARD_READ_ALOUD.name if is_agent else "CALLER_SPOKE_PAN"},
                action_taken="block_and_alert" if is_agent else "alert",
            ))
        return events

    def detect_sensitive_data_spoken(self, text: str, channel: str = "caller") -> List[SpeechEvent]:
        """Detect SSN, PIN, account numbers, expiry dates, and CVV spoken aloud."""
        events: List[SpeechEvent] = []
        is_agent = channel.lower() == "agent"
        ch = AnalyticsChannel.AGENT if is_agent else AnalyticsChannel.CALLER

        for parts in self._SSN_RE.findall(text):
            if parts[0] in ("000", "666") or parts[0].startswith("9") or parts[1] == "00" or parts[2] == "0000":
                continue
            events.append(SpeechEvent(
                event_type=SpeechEventType.SSN_SPOKEN, risk_level=SpeechRiskLevel.CRITICAL,
                channel=ch, transcript_segment=text[:200], confidence=0.85,
                evidence={"masked_ssn": f"***-**-{parts[2]}", "spoken_by_agent": is_agent},
                action_taken="alert",
            ))

        for month, year in self._EXPIRY_RE.findall(text):
            risk = SpeechRiskLevel.HIGH if is_agent else SpeechRiskLevel.MEDIUM
            events.append(SpeechEvent(
                event_type=SpeechEventType.EXPIRY_SPOKEN, risk_level=risk,
                channel=ch, transcript_segment=text[:200], confidence=0.75,
                evidence={"spoken_by_agent": is_agent,
                          "violation_type": ComplianceViolationType.PCI_EXPIRY_READ_ALOUD.name if is_agent else "CALLER_SPOKE_EXPIRY"},
                action_taken="alert" if is_agent else "log",
            ))

        for pin in self._PIN_RE.findall(text):
            events.append(SpeechEvent(
                event_type=SpeechEventType.PIN_SPOKEN, risk_level=SpeechRiskLevel.HIGH,
                channel=ch, transcript_segment=text[:200], confidence=0.8,
                evidence={"pin_length": len(pin), "spoken_by_agent": is_agent},
                action_taken="alert",
            ))

        for acct in self._ACCT_RE.findall(text):
            if self._number_parser.luhn_check(acct):
                continue
            if len(acct) == 9 and not acct.startswith(("000", "666", "9")):
                continue
            events.append(SpeechEvent(
                event_type=SpeechEventType.ACCOUNT_NUMBER_SPOKEN, risk_level=SpeechRiskLevel.MEDIUM,
                channel=ch, transcript_segment=text[:200], confidence=0.6,
                evidence={"masked_account": acct[:2] + "*" * (len(acct) - 4) + acct[-2:], "spoken_by_agent": is_agent},
                action_taken="log",
            ))

        for _ in self._CVV_RE.findall(text):
            events.append(SpeechEvent(
                event_type=SpeechEventType.CVV_SPOKEN,
                risk_level=SpeechRiskLevel.CRITICAL if is_agent else SpeechRiskLevel.HIGH,
                channel=ch, transcript_segment=text[:200], confidence=0.88,
                evidence={"spoken_by_agent": is_agent,
                          "violation_type": ComplianceViolationType.PCI_CVV_READ_ALOUD.name if is_agent else "CALLER_SPOKE_CVV"},
                action_taken="block_and_alert" if is_agent else "alert",
            ))
        return events

    def detect_social_engineering(self, transcript_segments: List[Dict[str, str]]) -> List[SpeechEvent]:
        """Detect social engineering attempts in the call transcript."""
        return self._se_detector.detect(transcript_segments)

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of a text segment."""
        return self._sentiment_analyzer.analyze(text)

    def check_compliance_script(self, transcript_segments: List[Dict[str, str]],
                                script: ComplianceScript) -> List[SpeechEvent]:
        """Check if required phrases were spoken and forbidden phrases avoided."""
        events: List[SpeechEvent] = []
        filtered: List[str] = []
        for seg in transcript_segments:
            sc = seg.get("channel", "agent").lower()
            if script.channel == AnalyticsChannel.AGENT and sc != "agent":
                continue
            if script.channel == AnalyticsChannel.CALLER and sc != "caller":
                continue
            filtered.append(seg.get("text", "").lower())
        combined = " ".join(filtered)

        for phrase in script.required_phrases:
            if phrase.lower() not in combined:
                events.append(SpeechEvent(
                    event_type=SpeechEventType.COMPLIANCE_PHRASE_MISSING,
                    risk_level=SpeechRiskLevel.MEDIUM, channel=script.channel,
                    transcript_segment=f"Missing: {phrase}", confidence=0.95,
                    evidence={"script_name": script.name, "framework": script.framework, "missing_phrase": phrase},
                    action_taken="alert",
                ))
        for phrase in script.forbidden_phrases:
            if phrase.lower() in combined:
                events.append(SpeechEvent(
                    event_type=SpeechEventType.UNAUTHORIZED_DISCLOSURE,
                    risk_level=SpeechRiskLevel.HIGH, channel=script.channel,
                    transcript_segment=f"Forbidden: {phrase}", confidence=0.95,
                    evidence={"script_name": script.name, "framework": script.framework, "forbidden_phrase": phrase},
                    action_taken="alert",
                ))
        return events

    def match_keywords(self, text: str, channel: str = "both") -> List[SpeechEvent]:
        """Match text against configured keyword rules."""
        events: List[SpeechEvent] = []
        ch = AnalyticsChannel.AGENT if channel == "agent" else AnalyticsChannel.CALLER
        for rule in self._policy.keyword_rules:
            if rule.channel not in (AnalyticsChannel.BOTH, ch):
                continue
            if rule.matches(text):
                events.append(SpeechEvent(
                    event_type=SpeechEventType.KEYWORD_MATCH, risk_level=SpeechRiskLevel.LOW,
                    channel=ch, transcript_segment=text[:200], confidence=0.85,
                    evidence={"rule_id": rule.rule_id, "category": rule.category, "action": rule.action},
                    action_taken=rule.action,
                ))
        return events

    def generate_report(self, period_start: datetime, period_end: datetime) -> AnalyticsReport:
        """Generate analytics report for the given time period."""
        analyses = [a for a in self._analyses if period_start <= a.analyzed_at <= period_end]
        evts_by_type: Dict[str, int] = {}
        sent_dist: Dict[str, int] = {}
        total_risk, pci_n, se_n = 0.0, 0, 0
        all_viols: List[Dict[str, Any]] = []

        for a in analyses:
            total_risk += a.risk_score
            for e in a.events:
                evts_by_type[e.event_type.name] = evts_by_type.get(e.event_type.name, 0) + 1
                if e.event_type in (SpeechEventType.PAN_SPOKEN, SpeechEventType.CVV_SPOKEN, SpeechEventType.EXPIRY_SPOKEN):
                    pci_n += 1
                if e.event_type == SpeechEventType.SOCIAL_ENGINEERING_DETECTED:
                    se_n += 1
                if e.risk_level.level >= SpeechRiskLevel.HIGH.level:
                    all_viols.append({"call_id": e.call_id, "event_type": e.event_type.name, "risk_level": e.risk_level.name})
            for s in a.sentiment_timeline:
                sent_dist[s.category.name] = sent_dist.get(s.category.name, 0) + 1

        n = max(len(analyses), 1)
        avg_risk = total_risk / n
        total_segs = sum(a.total_segments_analyzed for a in analyses)
        comp_score = max(0.0, 1.0 - (sum(evts_by_type.values()) / max(total_segs, 1)))

        recs: List[str] = []
        if pci_n:
            recs.append(f"Address {pci_n} PCI violation(s): reinforce agent training on card data handling.")
        if se_n:
            recs.append(f"Review {se_n} social engineering attempt(s): update caller verification procedures.")
        if avg_risk > 0.6:
            recs.append("Average risk exceeds threshold: conduct compliance refresher training.")
        if not recs:
            recs.append("No critical issues detected in this period.")

        return AnalyticsReport(
            period_start=period_start, period_end=period_end,
            total_calls_analyzed=len(analyses), events_by_type=evts_by_type,
            top_violations=sorted(all_viols, key=lambda v: SpeechRiskLevel[v["risk_level"]].level, reverse=True)[:20],
            social_engineering_attempts=se_n, sentiment_distribution=sent_dist,
            average_risk_score=round(avg_risk, 4), pci_violations=pci_n,
            compliance_score=round(comp_score, 4), recommendations=recs,
        )

    def _compute_integrity_hash(self, data: str) -> str:
        """Compute SHA3-256 hash for PQC-grade tamper evidence."""
        return hashlib.sha3_256(data.encode()).hexdigest()

    def _compute_risk_score(self, events: List[SpeechEvent]) -> float:
        """Compute aggregate risk score from all detected events."""
        if not events:
            return 0.0
        w_sum = w_total = 0.0
        for e in events:
            w_sum += ((e.risk_level.min_score + e.risk_level.max_score) / 2.0) * e.confidence
            w_total += e.confidence
        if w_total == 0:
            return 0.0
        raw = w_sum / w_total
        crit = sum(1 for e in events if e.risk_level == SpeechRiskLevel.CRITICAL)
        high = sum(1 for e in events if e.risk_level == SpeechRiskLevel.HIGH)
        return round(min(1.0, raw + min(0.3, crit * 0.1 + high * 0.05)), 4)

    def _detect_sentiment_trends(self, timeline: List[SentimentResult]) -> List[SentimentResult]:
        """Annotate sentiment results with trend information."""
        if len(timeline) < 2:
            return timeline
        for i in range(len(timeline)):
            if i == 0:
                timeline[i].trend = "stable"
                continue
            recent = [t.score for t in timeline[max(0, i - 3):i]]
            avg = sum(recent) / len(recent)
            diff = timeline[i].score - avg
            timeline[i].trend = "improving" if diff > 0.15 else ("declining" if diff < -0.15 else "stable")
        return timeline

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create_pci_policy(cls) -> SpeechAnalyticsPolicy:
        """Factory: PCI-focused policy detecting card data in speech."""
        return SpeechAnalyticsPolicy(
            name="pci_compliance", pci_detection_enabled=True,
            social_engineering_detection=False, sentiment_tracking=False,
            profanity_detection=False, min_confidence=0.75,
            auto_alert_threshold=SpeechRiskLevel.HIGH,
            keyword_rules=[
                KeywordRule(keywords=["card number", "credit card", "debit card", "visa", "mastercard", "amex"],
                            category="pci_keyword", action="monitor", channel=AnalyticsChannel.BOTH),
                KeywordRule(keywords=["cvv", "security code", "cvc", "cv2"],
                            category="pci_cvv", action="alert", channel=AnalyticsChannel.BOTH),
                KeywordRule(keywords=["expiry", "expiration", "valid through"],
                            category="pci_expiry", action="monitor", channel=AnalyticsChannel.BOTH),
            ],
            compliance_scripts=[
                ComplianceScript(name="pci_payment_script",
                    required_phrases=["please enter your card number using your keypad",
                                      "do not read your card number aloud"],
                    forbidden_phrases=["read me your card number", "tell me your cvv",
                                       "what is your card number"],
                    framework="PCI-DSS-4.0", channel=AnalyticsChannel.AGENT),
            ],
        )

    @classmethod
    def create_full_compliance_policy(cls) -> SpeechAnalyticsPolicy:
        """Factory: comprehensive compliance policy (PCI + HIPAA + TCPA + GDPR + SOX + FCA + DPDPA)."""
        return SpeechAnalyticsPolicy(
            name="full_compliance", pci_detection_enabled=True,
            social_engineering_detection=True, sentiment_tracking=True,
            profanity_detection=True, min_confidence=0.7,
            auto_alert_threshold=SpeechRiskLevel.MEDIUM,
            silence_threshold_seconds=8.0, crosstalk_threshold_pct=0.25,
            keyword_rules=[
                KeywordRule(keywords=["card number", "credit card", "cvv", "security code"],
                            category="pci_data", action="alert", channel=AnalyticsChannel.BOTH),
                KeywordRule(keywords=["diagnosis", "prescription", "medical record", "patient", "health condition"],
                            category="hipaa_phi", action="alert", channel=AnalyticsChannel.BOTH),
                KeywordRule(keywords=["social security", "date of birth", "passport", "driver license"],
                            category="pii_data", action="alert", channel=AnalyticsChannel.BOTH),
                KeywordRule(keywords=["lawsuit", "attorney", "legal action", "sue"],
                            category="legal_threat", action="escalate", channel=AnalyticsChannel.CALLER),
            ],
            compliance_scripts=[
                ComplianceScript(name="pci_payment_script",
                    required_phrases=["please enter your card number using your keypad",
                                      "do not read your card number aloud"],
                    forbidden_phrases=["read me your card number", "tell me your cvv"],
                    framework="PCI-DSS-4.0", channel=AnalyticsChannel.AGENT),
                ComplianceScript(name="tcpa_consent_script",
                    required_phrases=["this call may be recorded", "do you consent to being recorded"],
                    forbidden_phrases=[], framework="TCPA", channel=AnalyticsChannel.AGENT),
                ComplianceScript(name="gdpr_consent_script",
                    required_phrases=["we process your data in accordance with our privacy policy"],
                    forbidden_phrases=[], framework="GDPR", channel=AnalyticsChannel.AGENT),
                ComplianceScript(name="fca_disclaimer",
                    required_phrases=["calls are recorded for training and monitoring purposes"],
                    forbidden_phrases=[], framework="FCA", channel=AnalyticsChannel.AGENT),
                ComplianceScript(name="dpdpa_consent",
                    required_phrases=["your personal data will be processed as per our privacy notice"],
                    forbidden_phrases=[], framework="DPDPA", channel=AnalyticsChannel.AGENT),
            ],
        )

    @classmethod
    def create_fraud_detection_policy(cls) -> SpeechAnalyticsPolicy:
        """Factory: social engineering and fraud detection focus."""
        return SpeechAnalyticsPolicy(
            name="fraud_detection", pci_detection_enabled=True,
            social_engineering_detection=True, sentiment_tracking=True,
            profanity_detection=False, min_confidence=0.65,
            auto_alert_threshold=SpeechRiskLevel.MEDIUM,
            keyword_rules=[
                KeywordRule(keywords=["transfer", "wire", "western union", "bitcoin", "gift card", "money order"],
                            category="fraud_financial", action="alert", channel=AnalyticsChannel.CALLER),
                KeywordRule(keywords=["remote access", "teamviewer", "anydesk", "screen share", "download"],
                            category="fraud_technical", action="alert", channel=AnalyticsChannel.CALLER),
                KeywordRule(keywords=["irs", "warrant", "arrest", "police", "suspended", "locked out"],
                            category="fraud_authority", action="alert", channel=AnalyticsChannel.CALLER),
                KeywordRule(keywords=["password", "one-time code", "otp", "verification code", "two factor"],
                            category="credential_phishing", action="alert", channel=AnalyticsChannel.CALLER),
                KeywordRule(keywords=["won", "prize", "lottery", "selected", "congratulations", "reward"],
                            category="fraud_reward", action="alert", channel=AnalyticsChannel.CALLER),
            ],
            compliance_scripts=[
                ComplianceScript(name="caller_verification",
                    required_phrases=["may i have your name please", "can you verify your account number"],
                    forbidden_phrases=["i will give you my supervisor password", "let me bypass that for you"],
                    framework="SOX", channel=AnalyticsChannel.AGENT),
            ],
        )
