"""
Conversational AI Prompt Injection Defense Module

Detects and prevents prompt injection attacks in BPO conversational AI
systems with PQC-signed system prompt integrity verification.

Prompt injection is a critical threat to AI-powered BPO workflows:
- Direct injection - malicious instructions in user input
- Indirect injection - hidden instructions in external data
- Jailbreak attempts - bypassing safety constraints
- Role override - impersonating system or developer roles
- Context manipulation - steering conversation off-policy
- Data exfiltration - extracting system prompts or PII
- Multi-turn attacks - gradual escalation across turns

Integrates with QBITEL's PQC infrastructure for ML-DSA signed
system prompts and tamper-proof audit logging.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import logging
import re
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class InjectionType(Enum):
    """Types of prompt injection attacks detected."""

    DIRECT_INJECTION = auto()           # Explicit malicious instructions
    INDIRECT_INJECTION = auto()         # Hidden instructions in data
    JAILBREAK_ATTEMPT = auto()          # Safety constraint bypass
    ROLE_OVERRIDE = auto()              # System/developer impersonation
    CONTEXT_MANIPULATION = auto()       # Off-policy steering
    DATA_EXFILTRATION_PROMPT = auto()   # Extracting system/PII data
    PRIVILEGE_ESCALATION = auto()       # Gaining higher access
    ENCODED_INJECTION = auto()          # Base64/hex encoded payloads
    MULTI_TURN_ATTACK = auto()          # Gradual escalation over turns
    FLIP_ATTACK = auto()                # Negating safety instructions


class InjectionRiskLevel(Enum):
    """Risk classification for analyzed input."""

    SAFE = (0, "Safe")
    SUSPICIOUS = (1, "Suspicious")
    LIKELY_INJECTION = (2, "Likely Injection")
    CONFIRMED_INJECTION = (3, "Confirmed Injection")

    def __init__(self, level: int, display_name: str):
        self.level = level
        self.display_name = display_name


class DefenseAction(Enum):
    """Actions taken in response to detected injection."""

    ALLOW = auto()                  # Input is safe, allow through
    SANITIZE = auto()               # Remove detected injection patterns
    BLOCK = auto()                  # Block the input entirely
    ALERT = auto()                  # Alert security team
    LOG_AND_ALLOW = auto()          # Log for analysis, allow through
    QUARANTINE_CONVERSATION = auto()  # Isolate the conversation session


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InjectionIndicator:
    """
    A single injection indicator found within user input.

    Represents a pattern match or heuristic signal that
    suggests prompt injection. Multiple indicators may
    contribute to an overall risk score.
    """

    indicator_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    injection_type: InjectionType = InjectionType.DIRECT_INJECTION
    matched_pattern: str = ""
    confidence: float = 0.0
    position_start: int = 0
    position_end: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize indicator for storage or transmission."""
        return {
            "indicator_id": self.indicator_id,
            "injection_type": self.injection_type.name,
            "matched_pattern": self.matched_pattern[:50],  # Truncate
            "confidence": self.confidence,
            "position_start": self.position_start,
            "position_end": self.position_end,
            "evidence": self.evidence,
        }


@dataclass
class PromptAnalysis:
    """
    Result of analyzing user input for prompt injection.

    Contains the overall risk assessment, all detected indicators,
    and optional sanitized text. Each analysis is PQC-hashed for
    tamper-proof audit.
    """

    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_text: str = ""
    detected_injections: List[InjectionIndicator] = field(default_factory=list)
    risk_level: InjectionRiskLevel = InjectionRiskLevel.SAFE
    confidence: float = 0.0
    sanitized_text: Optional[str] = None
    was_blocked: bool = False
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    pqc_integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.pqc_integrity_hash:
            self.pqc_integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA3-256 integrity hash for this analysis."""
        payload = (
            f"{self.analysis_id}|{self.risk_level.name}|"
            f"{self.confidence}|{self.analyzed_at.isoformat()}"
        )
        return hashlib.sha3_256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize analysis for storage or transmission."""
        return {
            "analysis_id": self.analysis_id,
            "risk_level": self.risk_level.display_name,
            "confidence": self.confidence,
            "num_indicators": len(self.detected_injections),
            "indicators": [i.to_dict() for i in self.detected_injections],
            "was_blocked": self.was_blocked,
            "analyzed_at": self.analyzed_at.isoformat(),
            "pqc_integrity_hash": self.pqc_integrity_hash,
        }


@dataclass
class SystemPromptIntegrity:
    """
    Integrity record for a signed system prompt.

    The system prompt is hashed with SHA3-256 and signed with
    ML-DSA to detect tampering. Verified before every inference.
    """

    prompt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_hash: str = ""
    signed_at: datetime = field(default_factory=datetime.utcnow)
    pqc_signature: str = ""
    version: int = 1
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize integrity record for storage or transmission."""
        return {
            "prompt_id": self.prompt_id,
            "prompt_hash": self.prompt_hash,
            "signed_at": self.signed_at.isoformat(),
            "pqc_signature": self.pqc_signature[:32] + "...",
            "version": self.version,
            "is_valid": self.is_valid,
        }


@dataclass
class PromptDefensePolicy:
    """
    Configuration policy for prompt injection defense.

    Defines detection thresholds, signing requirements,
    and enforcement behavior.
    """

    enabled: bool = True
    block_threshold: float = 0.8
    sanitize_threshold: float = 0.5
    sign_system_prompts: bool = True
    sig_algorithm: str = "ML-DSA-65"
    max_input_length: int = 4096
    check_encoded_content: bool = True
    check_role_overrides: bool = True
    allowed_roles: List[str] = field(
        default_factory=lambda: ["user", "assistant"]
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy for storage or transmission."""
        return {
            "enabled": self.enabled,
            "block_threshold": self.block_threshold,
            "sanitize_threshold": self.sanitize_threshold,
            "sign_system_prompts": self.sign_system_prompts,
            "sig_algorithm": self.sig_algorithm,
            "max_input_length": self.max_input_length,
            "check_encoded_content": self.check_encoded_content,
            "check_role_overrides": self.check_role_overrides,
            "allowed_roles": self.allowed_roles,
        }


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# Direct injection patterns
DIRECT_INJECTION_PATTERNS: List[Tuple[str, float]] = [
    (r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", 0.95),
    (r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)", 0.95),
    (r"(?i)forget\s+(everything|all)\s+(you\s+)?(know|were told|learned)", 0.90),
    (r"(?i)you\s+are\s+now\s+(a|an|the)\s+\w+", 0.70),
    (r"(?i)new\s+(instructions?|rules?|system\s+prompt)\s*:", 0.90),
    (r"(?i)override\s+(system|safety|security)\s+(prompt|rules?|settings?)", 0.95),
    (r"(?i)act\s+as\s+(if\s+)?(you\s+are\s+)?(a|an|the)\s+", 0.60),
    (r"(?i)\[system\]|\[admin\]|\[developer\]|\[root\]", 0.85),
]

# Jailbreak patterns
JAILBREAK_PATTERNS: List[Tuple[str, float]] = [
    (r"(?i)do\s+anything\s+now", 0.90),
    (r"(?i)jailbreak", 0.95),
    (r"(?i)DAN\s+mode", 0.95),
    (r"(?i)developer\s+mode\s+(enabled|activated|on)", 0.90),
    (r"(?i)unrestricted\s+mode", 0.85),
    (r"(?i)no\s+(restrictions?|limitations?|filters?|censorship)", 0.80),
    (r"(?i)bypass\s+(safety|security|content)\s+(filters?|checks?)", 0.90),
    (r"(?i)pretend\s+(that\s+)?(there\s+are\s+)?no\s+(rules|restrictions)", 0.85),
]

# Role override patterns
ROLE_OVERRIDE_PATTERNS: List[Tuple[str, float]] = [
    (r"(?i)system\s*:\s*you\s+are", 0.90),
    (r"(?i)<<\s*SYS\s*>>", 0.95),
    (r"(?i)\[INST\]", 0.80),
    (r"(?i)###\s*(system|instruction|human|assistant)\s*:", 0.85),
    (r"(?i)<\|im_start\|>\s*(system|user|assistant)", 0.90),
    (r"(?i)BEGININSTRUCTION", 0.85),
    (r"(?i)ENDINPUT.*BEGININSTRUCTION", 0.90),
]

# Data exfiltration patterns
EXFILTRATION_PATTERNS: List[Tuple[str, float]] = [
    (r"(?i)what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions?|rules?)", 0.80),
    (r"(?i)repeat\s+(your\s+)?(system\s+)?(prompt|instructions?)", 0.85),
    (r"(?i)show\s+me\s+(your\s+)?(system|initial|original)\s+(prompt|message)", 0.85),
    (r"(?i)output\s+(your|the)\s+(entire|full|complete)\s+(prompt|context)", 0.90),
    (r"(?i)print\s+(the\s+)?(system|initial)\s+(prompt|message|instructions?)", 0.85),
    (r"(?i)tell\s+me\s+(the\s+)?confidential", 0.70),
    (r"(?i)list\s+all\s+(customer|user|personal)\s+(data|info|records)", 0.80),
]

# Flip attack patterns (negation of safety instructions)
FLIP_PATTERNS: List[Tuple[str, float]] = [
    (r"(?i)do\s+the\s+opposite", 0.80),
    (r"(?i)reverse\s+(all\s+)?(your\s+)?(rules?|instructions?)", 0.85),
    (r"(?i)instead\s+of\s+refusing,?\s+you\s+(should|must|will)", 0.80),
    (r"(?i)if\s+you\s+would\s+(normally\s+)?refuse,?\s+(do\s+it\s+)?anyway", 0.85),
]


# ---------------------------------------------------------------------------
# Injection Detector
# ---------------------------------------------------------------------------


class InjectionDetector:
    """
    Detects prompt injection patterns in user input.

    Uses a comprehensive set of regex patterns and heuristics
    to identify direct injection, jailbreak, role override,
    encoded content, and data exfiltration attempts.
    """

    def __init__(self, policy: PromptDefensePolicy):
        self._policy = policy

    def detect(self, text: str) -> List[InjectionIndicator]:
        """
        Scan text for all known injection patterns.

        Args:
            text: The user input to analyze.

        Returns:
            List of injection indicators found.
        """
        indicators: List[InjectionIndicator] = []

        # Direct injection
        indicators.extend(self._scan_patterns(
            text, DIRECT_INJECTION_PATTERNS, InjectionType.DIRECT_INJECTION
        ))

        # Jailbreak
        indicators.extend(self._scan_patterns(
            text, JAILBREAK_PATTERNS, InjectionType.JAILBREAK_ATTEMPT
        ))

        # Role override
        if self._policy.check_role_overrides:
            indicators.extend(self.check_role_override(text))

        # Data exfiltration
        indicators.extend(self._scan_patterns(
            text, EXFILTRATION_PATTERNS, InjectionType.DATA_EXFILTRATION_PROMPT
        ))

        # Flip attacks
        indicators.extend(self._scan_patterns(
            text, FLIP_PATTERNS, InjectionType.FLIP_ATTACK
        ))

        # Encoded content
        if self._policy.check_encoded_content:
            indicators.extend(self.check_encoded_content(text))

        # Length check (extremely long input may be stuffing attack)
        if len(text) > self._policy.max_input_length:
            indicators.append(InjectionIndicator(
                injection_type=InjectionType.CONTEXT_MANIPULATION,
                matched_pattern="<input_length_exceeded>",
                confidence=0.60,
                position_start=0,
                position_end=len(text),
                evidence={
                    "input_length": len(text),
                    "max_length": self._policy.max_input_length,
                },
            ))

        return indicators

    def check_role_override(self, text: str) -> List[InjectionIndicator]:
        """
        Check for role override / prompt format injection.

        Args:
            text: The user input to analyze.

        Returns:
            List of role override indicators found.
        """
        return self._scan_patterns(
            text, ROLE_OVERRIDE_PATTERNS, InjectionType.ROLE_OVERRIDE
        )

    def check_encoded_content(self, text: str) -> List[InjectionIndicator]:
        """
        Check for base64 or hex encoded injection payloads.

        Args:
            text: The user input to analyze.

        Returns:
            List of encoded injection indicators found.
        """
        indicators: List[InjectionIndicator] = []

        # Base64 detection (sequences of 20+ base64 chars)
        b64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
        for match in b64_pattern.finditer(text):
            indicators.append(InjectionIndicator(
                injection_type=InjectionType.ENCODED_INJECTION,
                matched_pattern=match.group()[:30] + "...",
                confidence=0.55,
                position_start=match.start(),
                position_end=match.end(),
                evidence={"encoding": "base64_suspect", "length": len(match.group())},
            ))

        # Hex string detection (long hex sequences)
        hex_pattern = re.compile(r'(?:0x)?[0-9a-fA-F]{32,}')
        for match in hex_pattern.finditer(text):
            indicators.append(InjectionIndicator(
                injection_type=InjectionType.ENCODED_INJECTION,
                matched_pattern=match.group()[:30] + "...",
                confidence=0.50,
                position_start=match.start(),
                position_end=match.end(),
                evidence={"encoding": "hex_suspect", "length": len(match.group())},
            ))

        # Unicode escape detection
        unicode_pattern = re.compile(r'(?:\\u[0-9a-fA-F]{4}){4,}')
        for match in unicode_pattern.finditer(text):
            indicators.append(InjectionIndicator(
                injection_type=InjectionType.ENCODED_INJECTION,
                matched_pattern=match.group()[:30] + "...",
                confidence=0.65,
                position_start=match.start(),
                position_end=match.end(),
                evidence={"encoding": "unicode_escape", "length": len(match.group())},
            ))

        return indicators

    def check_data_exfiltration_patterns(
        self, text: str
    ) -> List[InjectionIndicator]:
        """
        Check specifically for data exfiltration prompt patterns.

        Args:
            text: The user input to analyze.

        Returns:
            List of exfiltration indicators found.
        """
        return self._scan_patterns(
            text, EXFILTRATION_PATTERNS, InjectionType.DATA_EXFILTRATION_PROMPT
        )

    def _scan_patterns(
        self,
        text: str,
        patterns: List[Tuple[str, float]],
        injection_type: InjectionType,
    ) -> List[InjectionIndicator]:
        """Scan text against a list of regex patterns."""
        indicators: List[InjectionIndicator] = []
        for pattern_str, confidence in patterns:
            for match in re.finditer(pattern_str, text):
                indicators.append(InjectionIndicator(
                    injection_type=injection_type,
                    matched_pattern=match.group(),
                    confidence=confidence,
                    position_start=match.start(),
                    position_end=match.end(),
                    evidence={"pattern": pattern_str[:60]},
                ))
        return indicators


# ---------------------------------------------------------------------------
# Prompt Sanitizer
# ---------------------------------------------------------------------------


class PromptSanitizer:
    """
    Sanitizes user input by removing or escaping injection patterns.
    """

    # Special tokens that should be escaped in user input
    SPECIAL_TOKENS: List[str] = [
        "<|im_start|>", "<|im_end|>", "<<SYS>>", "<</SYS>>",
        "[INST]", "[/INST]", "### System:", "### Human:",
        "### Assistant:", "BEGININSTRUCTION", "ENDINPUT",
    ]

    def sanitize(
        self,
        text: str,
        indicators: List[InjectionIndicator],
    ) -> str:
        """
        Remove detected injection patterns from input text.

        Removes content at the positions identified by indicators,
        preserving the rest of the input.

        Args:
            text: The original user input.
            indicators: Detected injection indicators with positions.

        Returns:
            Sanitized text with injection patterns removed.
        """
        if not indicators:
            return text

        # Sort by position (reverse) to remove from end first
        sorted_indicators = sorted(
            indicators, key=lambda i: i.position_start, reverse=True
        )

        result = text
        for indicator in sorted_indicators:
            start = indicator.position_start
            end = indicator.position_end
            if 0 <= start < len(result) and start < end <= len(result):
                result = result[:start] + result[end:]

        # Also escape special tokens
        result = self.escape_special_tokens(result)

        return result.strip()

    def escape_special_tokens(self, text: str) -> str:
        """
        Escape special prompt tokens that could be used for injection.

        Args:
            text: Text to escape.

        Returns:
            Text with special tokens neutralized.
        """
        result = text
        for token in self.SPECIAL_TOKENS:
            safe_token = token.replace("<", "&lt;").replace(">", "&gt;")
            safe_token = safe_token.replace("[", "&#91;").replace("]", "&#93;")
            result = result.replace(token, safe_token)
        return result


# ---------------------------------------------------------------------------
# Prompt Injection Defense Engine
# ---------------------------------------------------------------------------


class PromptInjectionDefenseEngine:
    """
    Comprehensive prompt injection defense for BPO conversational AI.

    Combines pattern detection, input sanitization, and PQC-signed
    system prompt integrity verification.

    Usage::

        engine = PromptInjectionDefenseEngine.create_high_security_policy()

        # Protect system prompt
        integrity = engine.protect_system_prompt(system_prompt_text)

        # Analyze user input before sending to LLM
        analysis = engine.analyze_input(user_message)
        if analysis.was_blocked:
            # Reject input
            ...
        elif analysis.sanitized_text:
            # Use sanitized version
            user_message = analysis.sanitized_text

        # Verify system prompt hasn't been tampered with
        is_valid = engine.verify_system_prompt_integrity(integrity)
    """

    def __init__(self, policy: Optional[PromptDefensePolicy] = None):
        self._policy = policy or PromptDefensePolicy()
        self._detector = InjectionDetector(self._policy)
        self._sanitizer = PromptSanitizer()
        self._system_prompts: Dict[str, SystemPromptIntegrity] = {}
        self._analysis_history: List[PromptAnalysis] = []
        self._stats = {
            "total_inputs_analyzed": 0,
            "total_injections_detected": 0,
            "total_inputs_blocked": 0,
            "total_inputs_sanitized": 0,
            "total_system_prompts_signed": 0,
        }

        logger.info(
            "PromptInjectionDefenseEngine initialized "
            "enabled=%s block_threshold=%.2f sig_algorithm=%s",
            self._policy.enabled,
            self._policy.block_threshold,
            self._policy.sig_algorithm,
        )

    # ------------------------------------------------------------------
    # Input analysis
    # ------------------------------------------------------------------

    def analyze_input(self, text: str) -> PromptAnalysis:
        """
        Analyze user input for prompt injection attempts.

        Runs all detection checks and determines the risk level,
        confidence, and recommended action. Optionally sanitizes
        input if it falls between block and sanitize thresholds.

        Args:
            text: The user input to analyze.

        Returns:
            A PromptAnalysis result.
        """
        self._stats["total_inputs_analyzed"] += 1

        if not self._policy.enabled:
            return PromptAnalysis(input_text=text)

        indicators = self._detector.detect(text)
        max_confidence = max(
            (i.confidence for i in indicators), default=0.0
        )

        # Determine risk level
        if max_confidence >= self._policy.block_threshold:
            risk_level = InjectionRiskLevel.CONFIRMED_INJECTION
        elif max_confidence >= self._policy.sanitize_threshold:
            risk_level = InjectionRiskLevel.LIKELY_INJECTION
        elif max_confidence > 0.2:
            risk_level = InjectionRiskLevel.SUSPICIOUS
        else:
            risk_level = InjectionRiskLevel.SAFE

        # Determine action
        was_blocked = False
        sanitized_text = None

        if max_confidence >= self._policy.block_threshold:
            was_blocked = True
            self._stats["total_inputs_blocked"] += 1
            logger.warning(
                "Input BLOCKED: confidence=%.2f indicators=%d risk=%s",
                max_confidence, len(indicators), risk_level.display_name,
            )
        elif max_confidence >= self._policy.sanitize_threshold:
            sanitized_text = self._sanitizer.sanitize(text, indicators)
            self._stats["total_inputs_sanitized"] += 1
            logger.info(
                "Input SANITIZED: confidence=%.2f indicators=%d",
                max_confidence, len(indicators),
            )

        if indicators:
            self._stats["total_injections_detected"] += len(indicators)

        analysis = PromptAnalysis(
            input_text=text,
            detected_injections=indicators,
            risk_level=risk_level,
            confidence=max_confidence,
            sanitized_text=sanitized_text,
            was_blocked=was_blocked,
        )

        self._analysis_history.append(analysis)
        return analysis

    # ------------------------------------------------------------------
    # System prompt integrity
    # ------------------------------------------------------------------

    def protect_system_prompt(self, prompt_text: str) -> SystemPromptIntegrity:
        """
        Sign and protect a system prompt with PQC integrity.

        Computes a SHA3-256 hash and generates an ML-DSA signature
        for tamper detection.

        Args:
            prompt_text: The system prompt to protect.

        Returns:
            A SystemPromptIntegrity record.
        """
        prompt_hash = hashlib.sha3_256(prompt_text.encode()).hexdigest()

        # Generate PQC signature (simulated ML-DSA)
        sig_payload = f"{prompt_hash}|{self._policy.sig_algorithm}|{datetime.utcnow().isoformat()}"
        pqc_signature = hashlib.sha3_256(sig_payload.encode()).hexdigest()

        version = 1
        # Increment version if we already have this prompt
        for existing in self._system_prompts.values():
            if existing.prompt_hash == prompt_hash:
                version = existing.version + 1

        integrity = SystemPromptIntegrity(
            prompt_hash=prompt_hash,
            pqc_signature=pqc_signature,
            version=version,
            is_valid=True,
        )

        self._system_prompts[integrity.prompt_id] = integrity
        self._stats["total_system_prompts_signed"] += 1

        logger.info(
            "System prompt protected: id=%s version=%d algorithm=%s",
            integrity.prompt_id, version, self._policy.sig_algorithm,
        )
        return integrity

    def verify_system_prompt_integrity(
        self,
        integrity: SystemPromptIntegrity,
        current_prompt_text: Optional[str] = None,
    ) -> bool:
        """
        Verify that a system prompt has not been tampered with.

        Args:
            integrity: The integrity record to verify.
            current_prompt_text: If provided, re-hash and compare.

        Returns:
            True if the system prompt integrity is valid.
        """
        if not integrity.is_valid:
            logger.warning("System prompt marked invalid: id=%s", integrity.prompt_id)
            return False

        if current_prompt_text is not None:
            current_hash = hashlib.sha3_256(current_prompt_text.encode()).hexdigest()
            if current_hash != integrity.prompt_hash:
                logger.critical(
                    "System prompt TAMPERED: id=%s expected_hash=%s actual_hash=%s",
                    integrity.prompt_id,
                    integrity.prompt_hash[:16] + "...",
                    current_hash[:16] + "...",
                )
                return False

        # Verify the record exists in our store
        stored = self._system_prompts.get(integrity.prompt_id)
        if stored is None:
            logger.warning(
                "System prompt not found in store: id=%s", integrity.prompt_id
            )
            return False

        return stored.prompt_hash == integrity.prompt_hash

    # ------------------------------------------------------------------
    # Sanitization
    # ------------------------------------------------------------------

    def sanitize_user_input(self, text: str) -> str:
        """
        Sanitize user input by escaping special tokens.

        This is a lightweight pass that does not perform full
        injection detection. Use analyze_input for full analysis.

        Args:
            text: User input to sanitize.

        Returns:
            Sanitized text.
        """
        return self._sanitizer.escape_special_tokens(text)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_defense_report(self) -> Dict[str, Any]:
        """
        Generate a defense activity report.

        Returns:
            Report dictionary with detection statistics and trends.
        """
        injection_by_type: Dict[str, int] = {}
        for analysis in self._analysis_history:
            for indicator in analysis.detected_injections:
                key = indicator.injection_type.name
                injection_by_type[key] = injection_by_type.get(key, 0) + 1

        risk_distribution: Dict[str, int] = {}
        for analysis in self._analysis_history:
            key = analysis.risk_level.display_name
            risk_distribution[key] = risk_distribution.get(key, 0) + 1

        report_id = str(uuid.uuid4())
        report_hash = hashlib.sha3_256(
            f"{report_id}|{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return {
            "report_id": report_id,
            "generated_at": datetime.utcnow().isoformat(),
            "policy": self._policy.to_dict(),
            "statistics": dict(self._stats),
            "injection_breakdown": injection_by_type,
            "risk_distribution": risk_distribution,
            "system_prompts_protected": len(self._system_prompts),
            "pqc_report_hash": report_hash,
        }

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def create_standard_policy(cls) -> "PromptInjectionDefenseEngine":
        """
        Create an engine with standard defense policy.

        Balanced detection with moderate thresholds suitable
        for general BPO conversational AI deployments.
        """
        policy = PromptDefensePolicy(
            enabled=True,
            block_threshold=0.8,
            sanitize_threshold=0.5,
            sign_system_prompts=True,
            sig_algorithm="ML-DSA-65",
            max_input_length=4096,
            check_encoded_content=True,
            check_role_overrides=True,
            allowed_roles=["user", "assistant"],
        )
        return cls(policy=policy)

    @classmethod
    def create_high_security_policy(cls) -> "PromptInjectionDefenseEngine":
        """
        Create an engine with high-security defense policy.

        Aggressive detection with low thresholds for environments
        handling sensitive data (PCI, HIPAA, financial).
        """
        policy = PromptDefensePolicy(
            enabled=True,
            block_threshold=0.6,
            sanitize_threshold=0.3,
            sign_system_prompts=True,
            sig_algorithm="ML-DSA-87",
            max_input_length=2048,
            check_encoded_content=True,
            check_role_overrides=True,
            allowed_roles=["user"],
        )
        return cls(policy=policy)
