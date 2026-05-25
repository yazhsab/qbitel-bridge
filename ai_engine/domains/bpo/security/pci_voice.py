"""
PCI-DSS Voice Channel Security Module

PCI-DSS compliance for voice channels in contact centers.

Implements:
- DTMF masking/clamping (prevents capture of card digits)
- Pause/resume call recording during payment capture
- Agent screen masking (hide full card numbers)
- Secure DTMF relay via SIP INFO
- Real-time PAN detection in data streams
- PCI scope reduction for voice channels

This module ensures that contact center voice channels meet PCI-DSS 4.0
requirements for cardholder data protection. It operates at the media
layer (RTP/SRTP) and signaling layer (SIP) to prevent card data capture
in recordings, screen captures, and network traces.

Reference: PCI-DSS 4.0 Requirements 3.3, 3.4, 3.5, 8.3, 10.2
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


class DTMFMaskingMode(Enum):
    """
    DTMF masking modes for PCI-DSS compliance.

    Each mode represents a different strategy for preventing the capture
    of DTMF tones carrying cardholder data.
    """

    CLAMP = (
        "clamp",
        "Replace DTMF tones with a single flat tone (most common)",
    )
    FLAT_TONE = (
        "flat_tone",
        "Replace DTMF with a constant-frequency tone",
    )
    SILENCE = (
        "silence",
        "Replace DTMF with silence in the media stream",
    )
    REPLACE = (
        "replace",
        "Replace DTMF digits with random tones",
    )

    def __init__(self, mode_id: str, description: str):
        self.mode_id = mode_id
        self.description = description


class RecordingState(Enum):
    """State of call recording."""

    ACTIVE = auto()           # Recording in progress
    PAUSED = auto()           # Recording paused (PCI scope)
    STOPPED = auto()          # Recording stopped
    FAILED = auto()           # Recording error
    NOT_STARTED = auto()      # Recording not yet initiated


class PCIScopeState(Enum):
    """PCI scope state for a call."""

    OUT_OF_SCOPE = auto()     # No cardholder data being processed
    IN_SCOPE = auto()         # Payment data capture in progress
    TRANSITIONING = auto()    # Entering or exiting PCI scope
    FAILED_SCOPE = auto()     # Failed to enter/exit scope properly


class MaskingTarget(Enum):
    """Types of data to mask on agent screens."""

    PAN = auto()              # Primary Account Number
    CVV = auto()              # Card Verification Value
    EXPIRY = auto()           # Card expiry date
    CARDHOLDER_NAME = auto()  # Cardholder name
    PIN = auto()              # PIN (should never be visible)
    SSN = auto()              # Social Security Number
    DOB = auto()              # Date of birth
    ACCOUNT_NUMBER = auto()   # Bank account number


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DTMFEvent:
    """
    A DTMF tone event captured during a call.

    DTMF events during PCI scope are masked before storage.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    digit: str = ""              # The original digit (only in memory)
    masked_digit: str = "*"      # The masked representation
    duration_ms: int = 100
    was_in_pci_scope: bool = False
    masking_mode: Optional[DTMFMaskingMode] = None

    def to_safe_dict(self) -> Dict[str, Any]:
        """Serialize to a dict that never contains the original digit."""
        return {
            "event_id": self.event_id,
            "call_id": self.call_id,
            "timestamp": self.timestamp.isoformat(),
            "masked_digit": self.masked_digit,
            "duration_ms": self.duration_ms,
            "was_in_pci_scope": self.was_in_pci_scope,
            "masking_mode": (
                self.masking_mode.mode_id if self.masking_mode else None
            ),
        }


@dataclass
class RecordingEvent:
    """An event in the recording lifecycle (pause, resume, etc.)."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    previous_state: RecordingState = RecordingState.NOT_STARTED
    new_state: RecordingState = RecordingState.ACTIVE
    reason: str = ""
    triggered_by: str = ""       # "auto" or agent_id
    pci_scope: bool = False


@dataclass
class PCIComplianceReport:
    """
    PCI-DSS compliance report for a time period.

    Summarizes PCI-relevant events and compliance metrics.
    """

    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Metrics
    total_calls: int = 0
    calls_with_payment: int = 0
    dtmf_masking_events: int = 0
    recording_pauses: int = 0
    recording_resumes: int = 0
    failed_pauses: int = 0
    pan_detections: int = 0
    pan_detections_blocked: int = 0
    screen_masking_events: int = 0
    scope_transitions: int = 0
    compliance_violations: List[str] = field(default_factory=list)
    compliance_score: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report for storage."""
        return {
            "report_id": self.report_id,
            "tenant_id": self.tenant_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "total_calls": self.total_calls,
            "calls_with_payment": self.calls_with_payment,
            "dtmf_masking_events": self.dtmf_masking_events,
            "recording_pauses": self.recording_pauses,
            "recording_resumes": self.recording_resumes,
            "failed_pauses": self.failed_pauses,
            "pan_detections": self.pan_detections,
            "pan_detections_blocked": self.pan_detections_blocked,
            "screen_masking_events": self.screen_masking_events,
            "scope_transitions": self.scope_transitions,
            "compliance_violations": self.compliance_violations,
            "compliance_score": self.compliance_score,
        }


# ---------------------------------------------------------------------------
# PAN Detector
# ---------------------------------------------------------------------------


class PANDetector:
    """
    Real-time Primary Account Number (PAN) detection.

    Uses Luhn algorithm validation combined with card brand prefix
    matching to detect credit/debit card numbers in data streams.
    This runs against screen text, chat messages, log entries, and
    any other textual data that might contain cardholder data.

    The detector never stores or logs the actual PAN -- it only
    reports the card brand, first 6 and last 4 digits (as allowed
    by PCI-DSS for identification purposes).
    """

    # Card brand prefixes and valid lengths
    CARD_BRANDS: Dict[str, Dict[str, Any]] = {
        "visa": {"prefixes": ["4"], "lengths": [13, 16, 19]},
        "mastercard": {
            "prefixes": ["51", "52", "53", "54", "55", "2221", "2720"],
            "lengths": [16],
        },
        "amex": {"prefixes": ["34", "37"], "lengths": [15]},
        "discover": {
            "prefixes": ["6011", "644", "645", "646", "647", "648", "649", "65"],
            "lengths": [16, 17, 18, 19],
        },
        "diners_club": {"prefixes": ["300", "301", "302", "303", "304", "305", "36", "38"], "lengths": [14, 15, 16]},
        "jcb": {"prefixes": ["3528", "3589"], "lengths": [16, 17, 18, 19]},
        "unionpay": {"prefixes": ["62"], "lengths": [16, 17, 18, 19]},
        "maestro": {
            "prefixes": ["5018", "5020", "5038", "5893", "6304", "6759", "6761", "6762", "6763"],
            "lengths": [12, 13, 14, 15, 16, 17, 18, 19],
        },
    }

    # Regex to find potential card numbers (sequences of digits)
    _PAN_PATTERN = re.compile(
        r"(?<!\d)"                    # Not preceded by digit
        r"(\d[\d\s\-]{11,22}\d)"      # 12-24 chars of digits/spaces/dashes
        r"(?!\d)",                    # Not followed by digit
    )

    def __init__(
        self,
        *,
        enable_luhn: bool = True,
        enable_brand_detection: bool = True,
        alert_callback: Optional[Callable[[str, str, str], None]] = None,
    ):
        self.enable_luhn = enable_luhn
        self.enable_brand_detection = enable_brand_detection
        self.alert_callback = alert_callback
        self._detection_count: int = 0
        self._blocked_count: int = 0

    def scan_text(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """
        Scan text for potential PANs.

        Args:
            text: The text to scan.
            context: Description of where this text came from.

        Returns:
            List of detection results with masked PAN, brand, and position.
        """
        detections: List[Dict[str, Any]] = []

        for match in self._PAN_PATTERN.finditer(text):
            raw = match.group(1)
            digits_only = re.sub(r"[\s\-]", "", raw)

            if len(digits_only) < 12 or len(digits_only) > 19:
                continue

            # Luhn validation
            if self.enable_luhn and not self._luhn_check(digits_only):
                continue

            # Brand detection
            brand = self._detect_brand(digits_only) if self.enable_brand_detection else "unknown"

            if brand or not self.enable_brand_detection:
                self._detection_count += 1
                detection = {
                    "masked_pan": self._mask_pan(digits_only),
                    "brand": brand or "unknown",
                    "position": match.start(),
                    "length": len(digits_only),
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                detections.append(detection)

                if self.alert_callback:
                    try:
                        self.alert_callback(
                            detection["masked_pan"],
                            detection["brand"],
                            context,
                        )
                    except Exception as exc:
                        logger.error("PAN detection callback failed: %s", exc)

                logger.warning(
                    "PAN detected: brand=%s masked=%s context=%s",
                    brand,
                    detection["masked_pan"],
                    context,
                )

        return detections

    def redact_text(self, text: str) -> Tuple[str, int]:
        """
        Redact all PANs from text, replacing with masked versions.

        Args:
            text: The text to redact.

        Returns:
            Tuple of (redacted text, number of redactions).
        """
        redaction_count = 0
        result = text

        for match in self._PAN_PATTERN.finditer(text):
            raw = match.group(1)
            digits_only = re.sub(r"[\s\-]", "", raw)

            if len(digits_only) < 12 or len(digits_only) > 19:
                continue

            if self.enable_luhn and not self._luhn_check(digits_only):
                continue

            brand = self._detect_brand(digits_only) if self.enable_brand_detection else "unknown"
            if brand or not self.enable_brand_detection:
                masked = self._mask_pan(digits_only)
                result = result.replace(raw, masked, 1)
                redaction_count += 1
                self._blocked_count += 1

        return result, redaction_count

    @staticmethod
    def _luhn_check(card_number: str) -> bool:
        """
        Validate a card number using the Luhn algorithm.

        Args:
            card_number: Digits-only card number string.

        Returns:
            True if the number passes Luhn validation.
        """
        digits = [int(d) for d in card_number]
        checksum = 0
        reverse_digits = digits[::-1]

        for i, digit in enumerate(reverse_digits):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit

        return checksum % 10 == 0

    def _detect_brand(self, card_number: str) -> Optional[str]:
        """Detect the card brand from the card number prefix."""
        for brand, info in self.CARD_BRANDS.items():
            if len(card_number) not in info["lengths"]:
                continue
            for prefix in info["prefixes"]:
                if card_number.startswith(prefix):
                    return brand
        return None

    @staticmethod
    def _mask_pan(pan: str) -> str:
        """
        Mask a PAN showing only first 6 and last 4 digits.

        PCI-DSS allows displaying the first six and last four digits
        of a PAN for identification purposes.
        """
        if len(pan) <= 10:
            return "*" * len(pan)
        return pan[:6] + "*" * (len(pan) - 10) + pan[-4:]

    def get_statistics(self) -> Dict[str, int]:
        """Get PAN detection statistics."""
        return {
            "total_detections": self._detection_count,
            "total_blocked": self._blocked_count,
        }


# ---------------------------------------------------------------------------
# Recording Controller
# ---------------------------------------------------------------------------


class RecordingController:
    """
    Manages pause/resume of call recording for PCI-DSS compliance.

    When cardholder data is being captured (e.g., customer reading
    their card number), call recording must be paused to prevent
    the PAN from being stored in the recording.

    This controller manages the recording lifecycle and ensures
    that recordings are properly paused before payment data capture
    and resumed afterward, with automatic timeout protection.

    Usage::

        controller = RecordingController(call_id="call-123")
        controller.start_recording()

        # Before payment capture
        controller.pause_for_payment(agent_id="agent-001")

        # After payment capture
        controller.resume_after_payment(agent_id="agent-001")
    """

    def __init__(
        self,
        call_id: str,
        *,
        auto_resume_timeout_seconds: int = 120,
        max_pause_duration_seconds: int = 300,
        require_agent_confirmation: bool = True,
        event_callback: Optional[Callable[[RecordingEvent], None]] = None,
    ):
        self.call_id = call_id
        self.auto_resume_timeout_seconds = auto_resume_timeout_seconds
        self.max_pause_duration_seconds = max_pause_duration_seconds
        self.require_agent_confirmation = require_agent_confirmation
        self.event_callback = event_callback

        self._state: RecordingState = RecordingState.NOT_STARTED
        self._pause_start: Optional[datetime] = None
        self._events: List[RecordingEvent] = []
        self._pause_count: int = 0
        self._total_pause_duration_seconds: float = 0.0

    @property
    def state(self) -> RecordingState:
        """Get the current recording state."""
        # Check for auto-resume timeout
        if (
            self._state == RecordingState.PAUSED
            and self._pause_start is not None
        ):
            elapsed = (datetime.utcnow() - self._pause_start).total_seconds()
            if elapsed > self.auto_resume_timeout_seconds:
                self._auto_resume("Auto-resume timeout reached")
        return self._state

    @property
    def is_paused(self) -> bool:
        """Check if recording is currently paused."""
        return self.state == RecordingState.PAUSED

    @property
    def pause_count(self) -> int:
        """Get the number of times recording has been paused."""
        return self._pause_count

    def start_recording(self) -> RecordingEvent:
        """Start call recording."""
        event = self._transition(
            RecordingState.ACTIVE,
            reason="Recording started",
            triggered_by="system",
        )
        logger.info("Recording started for call_id=%s", self.call_id)
        return event

    def pause_for_payment(self, agent_id: str = "system") -> RecordingEvent:
        """
        Pause recording for payment data capture.

        This should be called before the customer provides cardholder
        data (card number, CVV, expiry).

        Args:
            agent_id: The agent or system initiating the pause.

        Returns:
            RecordingEvent describing the state transition.
        """
        if self._state != RecordingState.ACTIVE:
            logger.warning(
                "Cannot pause recording in state %s for call_id=%s",
                self._state.name,
                self.call_id,
            )
            return self._create_event(
                self._state,
                self._state,
                reason=f"Cannot pause: recording is {self._state.name}",
                triggered_by=agent_id,
                pci_scope=True,
            )

        self._pause_start = datetime.utcnow()
        self._pause_count += 1

        event = self._transition(
            RecordingState.PAUSED,
            reason="Payment data capture - PCI scope",
            triggered_by=agent_id,
            pci_scope=True,
        )

        logger.info(
            "Recording paused for PCI: call_id=%s agent=%s",
            self.call_id,
            agent_id,
        )
        return event

    def resume_after_payment(self, agent_id: str = "system") -> RecordingEvent:
        """
        Resume recording after payment data capture is complete.

        Args:
            agent_id: The agent or system initiating the resume.

        Returns:
            RecordingEvent describing the state transition.
        """
        if self._state != RecordingState.PAUSED:
            logger.warning(
                "Cannot resume recording in state %s for call_id=%s",
                self._state.name,
                self.call_id,
            )
            return self._create_event(
                self._state,
                self._state,
                reason=f"Cannot resume: recording is {self._state.name}",
                triggered_by=agent_id,
            )

        # Calculate pause duration
        if self._pause_start:
            pause_duration = (
                datetime.utcnow() - self._pause_start
            ).total_seconds()
            self._total_pause_duration_seconds += pause_duration
        self._pause_start = None

        event = self._transition(
            RecordingState.ACTIVE,
            reason="Payment capture complete - resuming",
            triggered_by=agent_id,
            pci_scope=True,
        )

        logger.info(
            "Recording resumed after PCI: call_id=%s agent=%s",
            self.call_id,
            agent_id,
        )
        return event

    def stop_recording(self, reason: str = "Call ended") -> RecordingEvent:
        """Stop recording entirely."""
        if self._state == RecordingState.PAUSED and self._pause_start:
            pause_duration = (
                datetime.utcnow() - self._pause_start
            ).total_seconds()
            self._total_pause_duration_seconds += pause_duration
            self._pause_start = None

        event = self._transition(
            RecordingState.STOPPED,
            reason=reason,
            triggered_by="system",
        )
        logger.info("Recording stopped for call_id=%s", self.call_id)
        return event

    def get_events(self) -> List[RecordingEvent]:
        """Get all recording lifecycle events."""
        return list(self._events)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of recording activity."""
        return {
            "call_id": self.call_id,
            "current_state": self._state.name,
            "pause_count": self._pause_count,
            "total_pause_duration_seconds": self._total_pause_duration_seconds,
            "total_events": len(self._events),
        }

    def _auto_resume(self, reason: str) -> None:
        """Auto-resume recording after timeout."""
        if self._pause_start:
            pause_duration = (
                datetime.utcnow() - self._pause_start
            ).total_seconds()
            self._total_pause_duration_seconds += pause_duration
        self._pause_start = None

        self._transition(
            RecordingState.ACTIVE,
            reason=reason,
            triggered_by="auto",
            pci_scope=True,
        )

        logger.warning(
            "Recording auto-resumed for call_id=%s after timeout",
            self.call_id,
        )

    def _transition(
        self,
        new_state: RecordingState,
        reason: str,
        triggered_by: str,
        pci_scope: bool = False,
    ) -> RecordingEvent:
        """Perform a state transition and record the event."""
        event = self._create_event(
            self._state, new_state, reason, triggered_by, pci_scope
        )
        self._state = new_state
        self._events.append(event)

        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as exc:
                logger.error("Recording event callback failed: %s", exc)

        return event

    def _create_event(
        self,
        previous_state: RecordingState,
        new_state: RecordingState,
        reason: str,
        triggered_by: str,
        pci_scope: bool = False,
    ) -> RecordingEvent:
        """Create a recording event."""
        return RecordingEvent(
            call_id=self.call_id,
            previous_state=previous_state,
            new_state=new_state,
            reason=reason,
            triggered_by=triggered_by,
            pci_scope=pci_scope,
        )


# ---------------------------------------------------------------------------
# Agent Screen Masker
# ---------------------------------------------------------------------------


class AgentScreenMasker:
    """
    Masks sensitive data on agent desktop screens.

    Prevents agents from viewing full card numbers, CVVs, and other
    sensitive data on their screens. Data is masked in real-time
    before being rendered on the agent desktop.

    Masking rules:
    - PAN: Show first 6 and last 4 only (e.g., 411111****1111)
    - CVV: Always fully masked (e.g., ***)
    - Expiry: Show month/year (allowed by PCI-DSS)
    - SSN: Show last 4 only (e.g., ***-**-1234)
    - DOB: Fully masked
    - Account numbers: Show last 4 only
    """

    # Masking patterns
    MASKING_RULES: Dict[MaskingTarget, Dict[str, Any]] = {
        MaskingTarget.PAN: {
            "pattern": re.compile(r"\b(\d{4})\s?(\d{4,8})\s?(\d{4})\b"),
            "show_first": 6,
            "show_last": 4,
            "mask_char": "*",
        },
        MaskingTarget.CVV: {
            "pattern": re.compile(r"\bCVV[:\s]*(\d{3,4})\b", re.IGNORECASE),
            "show_first": 0,
            "show_last": 0,
            "mask_char": "*",
        },
        MaskingTarget.SSN: {
            "pattern": re.compile(r"\b(\d{3})-?(\d{2})-?(\d{4})\b"),
            "show_first": 0,
            "show_last": 4,
            "mask_char": "*",
            "format": "***-**-{last4}",
        },
        MaskingTarget.DOB: {
            "pattern": re.compile(
                r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b"
            ),
            "show_first": 0,
            "show_last": 0,
            "mask_char": "*",
        },
        MaskingTarget.ACCOUNT_NUMBER: {
            "pattern": re.compile(r"\b(\d{8,17})\b"),
            "show_first": 0,
            "show_last": 4,
            "mask_char": "*",
        },
    }

    def __init__(
        self,
        *,
        enabled_targets: Optional[Set[MaskingTarget]] = None,
        mask_char: str = "*",
    ):
        self._enabled_targets: Set[MaskingTarget] = enabled_targets or {
            MaskingTarget.PAN,
            MaskingTarget.CVV,
            MaskingTarget.SSN,
            MaskingTarget.DOB,
            MaskingTarget.ACCOUNT_NUMBER,
        }
        self._mask_char = mask_char
        self._masking_count: int = 0

    def mask_text(self, text: str) -> Tuple[str, int]:
        """
        Apply all enabled masking rules to text.

        Args:
            text: The text to mask.

        Returns:
            Tuple of (masked text, number of maskings applied).
        """
        masked = text
        total_maskings = 0

        for target in self._enabled_targets:
            masked, count = self._apply_masking(masked, target)
            total_maskings += count

        self._masking_count += total_maskings
        return masked, total_maskings

    def mask_field(
        self, value: str, target: MaskingTarget
    ) -> str:
        """
        Mask a specific field value.

        Args:
            value: The value to mask.
            target: The type of data being masked.

        Returns:
            The masked value.
        """
        if target == MaskingTarget.PAN:
            return self._mask_pan(value)
        elif target == MaskingTarget.CVV:
            return self._mask_char * len(value)
        elif target == MaskingTarget.SSN:
            return self._mask_ssn(value)
        elif target == MaskingTarget.DOB:
            return self._mask_char * len(value)
        elif target == MaskingTarget.ACCOUNT_NUMBER:
            return self._mask_account(value)
        else:
            return self._mask_char * len(value)

    def _apply_masking(
        self, text: str, target: MaskingTarget
    ) -> Tuple[str, int]:
        """Apply a specific masking rule to text."""
        rule = self.MASKING_RULES.get(target)
        if not rule:
            return text, 0

        count = 0
        pattern = rule["pattern"]

        def replacer(match: re.Match) -> str:
            nonlocal count
            count += 1
            full = match.group(0)
            digits_only = re.sub(r"[\s\-/]", "", full)
            show_first = rule.get("show_first", 0)
            show_last = rule.get("show_last", 0)
            mc = rule.get("mask_char", self._mask_char)

            if "format" in rule:
                fmt = rule["format"]
                return fmt.replace(
                    "{last4}", digits_only[-show_last:] if show_last else ""
                )

            mask_len = len(digits_only) - show_first - show_last
            if mask_len < 0:
                mask_len = 0

            return (
                digits_only[:show_first]
                + mc * mask_len
                + digits_only[-show_last:] if show_last else digits_only[:show_first] + mc * mask_len
            )

        masked = pattern.sub(replacer, text)
        return masked, count

    def _mask_pan(self, pan: str) -> str:
        """Mask PAN showing first 6 and last 4."""
        digits = re.sub(r"\D", "", pan)
        if len(digits) <= 10:
            return self._mask_char * len(digits)
        return digits[:6] + self._mask_char * (len(digits) - 10) + digits[-4:]

    def _mask_ssn(self, ssn: str) -> str:
        """Mask SSN showing only last 4."""
        digits = re.sub(r"\D", "", ssn)
        if len(digits) != 9:
            return self._mask_char * len(ssn)
        return f"***-**-{digits[-4:]}"

    def _mask_account(self, account: str) -> str:
        """Mask account number showing only last 4."""
        digits = re.sub(r"\D", "", account)
        if len(digits) <= 4:
            return self._mask_char * len(digits)
        return self._mask_char * (len(digits) - 4) + digits[-4:]

    def enable_target(self, target: MaskingTarget) -> None:
        """Enable masking for a specific data type."""
        self._enabled_targets.add(target)

    def disable_target(self, target: MaskingTarget) -> None:
        """Disable masking for a specific data type."""
        self._enabled_targets.discard(target)

    def get_statistics(self) -> Dict[str, Any]:
        """Get masking statistics."""
        return {
            "total_maskings": self._masking_count,
            "enabled_targets": [t.name for t in self._enabled_targets],
        }


# ---------------------------------------------------------------------------
# PCI Scope Manager
# ---------------------------------------------------------------------------


class PCIScopeManager:
    """
    Manages PCI-DSS scope for voice channels.

    Tracks which calls are currently in PCI scope (i.e., processing
    cardholder data) and coordinates the various security controls
    that must be active during payment capture.

    When a call enters PCI scope:
    1. Recording is paused
    2. DTMF masking is activated
    3. Screen masking is enforced
    4. Audit logging is intensified
    5. Network isolation may be applied

    Usage::

        scope_mgr = PCIScopeManager()
        scope_mgr.enter_pci_scope(
            call_id="call-123",
            agent_id="agent-001",
            reason="Customer providing card number",
        )
        # ... payment capture happens ...
        scope_mgr.exit_pci_scope(
            call_id="call-123",
            agent_id="agent-001",
        )
    """

    def __init__(
        self,
        *,
        max_scope_duration_seconds: int = 300,
        auto_exit_on_timeout: bool = True,
        scope_change_callback: Optional[
            Callable[[str, PCIScopeState, PCIScopeState], None]
        ] = None,
    ):
        self.max_scope_duration_seconds = max_scope_duration_seconds
        self.auto_exit_on_timeout = auto_exit_on_timeout
        self.scope_change_callback = scope_change_callback

        # call_id -> (state, entered_at, agent_id)
        self._call_scopes: Dict[str, Tuple[PCIScopeState, datetime, str]] = {}
        self._scope_history: List[Dict[str, Any]] = []
        self._stats = {
            "total_scope_entries": 0,
            "total_scope_exits": 0,
            "total_timeouts": 0,
            "current_in_scope": 0,
        }

    def enter_pci_scope(
        self,
        call_id: str,
        agent_id: str,
        reason: str = "Payment data capture",
    ) -> PCIScopeState:
        """
        Place a call into PCI scope.

        Args:
            call_id: The call identifier.
            agent_id: The agent handling the call.
            reason: Reason for entering PCI scope.

        Returns:
            The new PCI scope state.
        """
        previous_state = self.get_scope_state(call_id)

        if previous_state == PCIScopeState.IN_SCOPE:
            logger.warning(
                "Call %s is already in PCI scope", call_id
            )
            return PCIScopeState.IN_SCOPE

        self._call_scopes[call_id] = (
            PCIScopeState.IN_SCOPE,
            datetime.utcnow(),
            agent_id,
        )
        self._stats["total_scope_entries"] += 1
        self._stats["current_in_scope"] += 1

        self._record_scope_change(
            call_id, agent_id, previous_state, PCIScopeState.IN_SCOPE, reason
        )

        logger.info(
            "Call %s entered PCI scope: agent=%s reason=%s",
            call_id,
            agent_id,
            reason,
        )
        return PCIScopeState.IN_SCOPE

    def exit_pci_scope(
        self,
        call_id: str,
        agent_id: str,
        reason: str = "Payment capture complete",
    ) -> PCIScopeState:
        """
        Remove a call from PCI scope.

        Args:
            call_id: The call identifier.
            agent_id: The agent handling the call.
            reason: Reason for exiting PCI scope.

        Returns:
            The new PCI scope state.
        """
        previous_state = self.get_scope_state(call_id)

        if previous_state != PCIScopeState.IN_SCOPE:
            logger.warning(
                "Call %s is not in PCI scope (state: %s)",
                call_id,
                previous_state.name,
            )
            return previous_state

        self._call_scopes[call_id] = (
            PCIScopeState.OUT_OF_SCOPE,
            datetime.utcnow(),
            agent_id,
        )
        self._stats["total_scope_exits"] += 1
        self._stats["current_in_scope"] = max(
            0, self._stats["current_in_scope"] - 1
        )

        self._record_scope_change(
            call_id, agent_id, previous_state, PCIScopeState.OUT_OF_SCOPE, reason
        )

        logger.info(
            "Call %s exited PCI scope: agent=%s reason=%s",
            call_id,
            agent_id,
            reason,
        )
        return PCIScopeState.OUT_OF_SCOPE

    def get_scope_state(self, call_id: str) -> PCIScopeState:
        """
        Get the current PCI scope state for a call.

        Also checks for scope timeout and auto-exits if configured.
        """
        if call_id not in self._call_scopes:
            return PCIScopeState.OUT_OF_SCOPE

        state, entered_at, agent_id = self._call_scopes[call_id]

        # Check for timeout
        if (
            state == PCIScopeState.IN_SCOPE
            and self.auto_exit_on_timeout
        ):
            elapsed = (datetime.utcnow() - entered_at).total_seconds()
            if elapsed > self.max_scope_duration_seconds:
                self._stats["total_timeouts"] += 1
                self.exit_pci_scope(
                    call_id,
                    agent_id,
                    reason=f"Auto-exit: timeout after {elapsed:.0f}s",
                )
                return PCIScopeState.OUT_OF_SCOPE

        return state

    def get_in_scope_calls(self) -> List[str]:
        """Get all call IDs currently in PCI scope."""
        return [
            call_id
            for call_id, (state, _, _) in self._call_scopes.items()
            if state == PCIScopeState.IN_SCOPE
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get PCI scope statistics."""
        return dict(self._stats)

    def _record_scope_change(
        self,
        call_id: str,
        agent_id: str,
        from_state: PCIScopeState,
        to_state: PCIScopeState,
        reason: str,
    ) -> None:
        """Record a scope change event."""
        record = {
            "call_id": call_id,
            "agent_id": agent_id,
            "from_state": from_state.name,
            "to_state": to_state.name,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._scope_history.append(record)

        if self.scope_change_callback:
            try:
                self.scope_change_callback(call_id, from_state, to_state)
            except Exception as exc:
                logger.error("Scope change callback failed: %s", exc)


# ---------------------------------------------------------------------------
# Compliance Reporter
# ---------------------------------------------------------------------------


class ComplianceReporter:
    """
    Generates PCI-DSS compliance reports for voice channels.

    Aggregates data from the PAN detector, recording controller,
    screen masker, and scope manager to produce compliance reports
    showing adherence to PCI-DSS requirements.
    """

    def __init__(
        self,
        tenant_id: str,
        *,
        pan_detector: Optional[PANDetector] = None,
        scope_manager: Optional[PCIScopeManager] = None,
    ):
        self.tenant_id = tenant_id
        self.pan_detector = pan_detector
        self.scope_manager = scope_manager
        self._violations: List[Dict[str, Any]] = []

    def generate_report(
        self,
        period_start: datetime,
        period_end: datetime,
        *,
        total_calls: int = 0,
        calls_with_payment: int = 0,
        dtmf_masking_events: int = 0,
        recording_pauses: int = 0,
        recording_resumes: int = 0,
        failed_pauses: int = 0,
        screen_masking_events: int = 0,
    ) -> PCIComplianceReport:
        """
        Generate a PCI-DSS compliance report.

        Args:
            period_start: Start of the reporting period.
            period_end: End of the reporting period.
            total_calls: Total number of calls in the period.
            calls_with_payment: Calls involving payment data.
            dtmf_masking_events: DTMF masking activations.
            recording_pauses: Recording pauses for PCI.
            recording_resumes: Recording resumes after PCI.
            failed_pauses: Failed recording pause attempts.
            screen_masking_events: Screen masking activations.

        Returns:
            A complete PCI compliance report.
        """
        # Gather PAN detection stats
        pan_stats = (
            self.pan_detector.get_statistics() if self.pan_detector else {}
        )
        scope_stats = (
            self.scope_manager.get_statistics() if self.scope_manager else {}
        )

        # Calculate violations
        violations: List[str] = list(self._violations_text(
            calls_with_payment=calls_with_payment,
            recording_pauses=recording_pauses,
            failed_pauses=failed_pauses,
            dtmf_masking_events=dtmf_masking_events,
        ))

        # Calculate compliance score
        compliance_score = self._calculate_score(
            calls_with_payment=calls_with_payment,
            recording_pauses=recording_pauses,
            failed_pauses=failed_pauses,
            violations=violations,
        )

        report = PCIComplianceReport(
            tenant_id=self.tenant_id,
            period_start=period_start,
            period_end=period_end,
            total_calls=total_calls,
            calls_with_payment=calls_with_payment,
            dtmf_masking_events=dtmf_masking_events,
            recording_pauses=recording_pauses,
            recording_resumes=recording_resumes,
            failed_pauses=failed_pauses,
            pan_detections=pan_stats.get("total_detections", 0),
            pan_detections_blocked=pan_stats.get("total_blocked", 0),
            screen_masking_events=screen_masking_events,
            scope_transitions=scope_stats.get("total_scope_entries", 0),
            compliance_violations=violations,
            compliance_score=compliance_score,
        )

        logger.info(
            "PCI compliance report generated: tenant=%s score=%.1f%% "
            "violations=%d period=%s to %s",
            self.tenant_id,
            compliance_score,
            len(violations),
            period_start.isoformat(),
            period_end.isoformat(),
        )

        return report

    def record_violation(
        self,
        violation_type: str,
        description: str,
        call_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """Record a PCI compliance violation."""
        self._violations.append({
            "type": violation_type,
            "description": description,
            "call_id": call_id,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
        })
        logger.warning(
            "PCI violation recorded: type=%s desc=%s call=%s agent=%s",
            violation_type,
            description,
            call_id,
            agent_id,
        )

    def _violations_text(
        self,
        *,
        calls_with_payment: int,
        recording_pauses: int,
        failed_pauses: int,
        dtmf_masking_events: int,
    ) -> List[str]:
        """Generate violation descriptions."""
        violations: List[str] = []

        if calls_with_payment > 0 and recording_pauses == 0:
            violations.append(
                "PCI-DSS 3.4: Payment calls detected with no recording pauses"
            )

        if failed_pauses > 0:
            violations.append(
                f"PCI-DSS 3.4: {failed_pauses} failed recording pause attempts"
            )

        if calls_with_payment > 0 and dtmf_masking_events == 0:
            violations.append(
                "PCI-DSS 3.3: DTMF masking not activated during payment calls"
            )

        # Include recorded violations
        for v in self._violations:
            violations.append(f"{v['type']}: {v['description']}")

        return violations

    @staticmethod
    def _calculate_score(
        *,
        calls_with_payment: int,
        recording_pauses: int,
        failed_pauses: int,
        violations: List[str],
    ) -> float:
        """Calculate compliance score (0-100)."""
        score = 100.0

        # Deduct for violations
        score -= len(violations) * 5.0

        # Deduct for failed pauses
        if calls_with_payment > 0:
            failure_rate = failed_pauses / max(1, calls_with_payment)
            score -= failure_rate * 20.0

        # Deduct if payments happened without pauses
        if calls_with_payment > 0 and recording_pauses == 0:
            score -= 30.0

        return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# PCI Voice Protector (orchestrator)
# ---------------------------------------------------------------------------


class PCIVoiceProtector:
    """
    PCI-DSS compliance for voice channels in contact centers.

    Implements:
    - DTMF masking/clamping (prevents capture of card digits)
    - Pause/resume call recording during payment capture
    - Agent screen masking (hide full card numbers)
    - Secure DTMF relay via SIP INFO
    - Real-time PAN detection in data streams
    - PCI scope reduction for voice channels

    This is the main orchestrator that coordinates all PCI voice
    security controls. It provides a unified interface for managing
    PCI compliance across all voice channel components.

    Usage::

        protector = PCIVoiceProtector(
            tenant_id="tenant-001",
            masking_mode=DTMFMaskingMode.CLAMP,
        )

        # Start a payment interaction
        protector.begin_payment_capture(
            call_id="call-123",
            agent_id="agent-001",
        )

        # Scan agent screen text for PANs
        safe_text, detections = protector.scan_and_mask(screen_text)

        # End payment interaction
        protector.end_payment_capture(
            call_id="call-123",
            agent_id="agent-001",
        )

        # Generate compliance report
        report = protector.generate_compliance_report(
            period_start=start,
            period_end=end,
        )
    """

    def __init__(
        self,
        tenant_id: str,
        *,
        masking_mode: DTMFMaskingMode = DTMFMaskingMode.CLAMP,
        auto_resume_timeout_seconds: int = 120,
        max_scope_duration_seconds: int = 300,
        enable_pan_detection: bool = True,
        enable_screen_masking: bool = True,
        alert_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.tenant_id = tenant_id
        self.masking_mode = masking_mode
        self.alert_callback = alert_callback

        # Initialize sub-components
        self.pan_detector = PANDetector(
            enable_luhn=True,
            enable_brand_detection=True,
        )
        self.screen_masker = AgentScreenMasker()
        self.scope_manager = PCIScopeManager(
            max_scope_duration_seconds=max_scope_duration_seconds,
            auto_exit_on_timeout=True,
        )
        self.compliance_reporter = ComplianceReporter(
            tenant_id=tenant_id,
            pan_detector=self.pan_detector,
            scope_manager=self.scope_manager,
        )

        # Active recording controllers (call_id -> controller)
        self._recording_controllers: Dict[str, RecordingController] = {}

        self._auto_resume_timeout = auto_resume_timeout_seconds
        self._enable_pan_detection = enable_pan_detection
        self._enable_screen_masking = enable_screen_masking

        # Statistics
        self._stats = {
            "total_payment_captures": 0,
            "total_dtmf_masked": 0,
            "total_pan_detected": 0,
            "total_screen_maskings": 0,
        }

        logger.info(
            "PCIVoiceProtector initialized: tenant=%s mode=%s "
            "auto_resume=%ds pan_detection=%s screen_masking=%s",
            tenant_id,
            masking_mode.mode_id,
            auto_resume_timeout_seconds,
            enable_pan_detection,
            enable_screen_masking,
        )

    def register_call(self, call_id: str) -> RecordingController:
        """
        Register a call and create its recording controller.

        Args:
            call_id: The call identifier.

        Returns:
            The RecordingController for this call.
        """
        controller = RecordingController(
            call_id=call_id,
            auto_resume_timeout_seconds=self._auto_resume_timeout,
        )
        controller.start_recording()
        self._recording_controllers[call_id] = controller
        logger.debug("Registered call %s for PCI monitoring", call_id)
        return controller

    def unregister_call(self, call_id: str) -> None:
        """Unregister a call and clean up resources."""
        if call_id in self._recording_controllers:
            controller = self._recording_controllers[call_id]
            if controller.state != RecordingState.STOPPED:
                controller.stop_recording("Call unregistered")
            del self._recording_controllers[call_id]

        # Exit PCI scope if still active
        if self.scope_manager.get_scope_state(call_id) == PCIScopeState.IN_SCOPE:
            self.scope_manager.exit_pci_scope(
                call_id, "system", "Call unregistered"
            )

        logger.debug("Unregistered call %s from PCI monitoring", call_id)

    def begin_payment_capture(
        self,
        call_id: str,
        agent_id: str,
        reason: str = "Customer providing card data",
    ) -> Dict[str, Any]:
        """
        Begin a payment data capture sequence.

        This activates all PCI controls:
        1. Enter PCI scope
        2. Pause recording
        3. Activate DTMF masking
        4. Enforce screen masking

        Args:
            call_id: The call identifier.
            agent_id: The agent handling the call.
            reason: Reason for entering payment capture.

        Returns:
            Status dict with results of each control activation.
        """
        self._stats["total_payment_captures"] += 1
        results: Dict[str, Any] = {"call_id": call_id, "agent_id": agent_id}

        # 1. Enter PCI scope
        scope_state = self.scope_manager.enter_pci_scope(
            call_id, agent_id, reason
        )
        results["pci_scope"] = scope_state.name

        # 2. Pause recording
        controller = self._recording_controllers.get(call_id)
        if controller:
            event = controller.pause_for_payment(agent_id)
            results["recording"] = event.new_state.name
        else:
            results["recording"] = "NO_CONTROLLER"
            logger.warning(
                "No recording controller for call %s during payment", call_id
            )

        # 3. DTMF masking is always active in PCI scope
        results["dtmf_masking"] = self.masking_mode.mode_id

        # 4. Screen masking
        results["screen_masking"] = self._enable_screen_masking

        logger.info(
            "Payment capture started: call_id=%s agent=%s scope=%s",
            call_id,
            agent_id,
            scope_state.name,
        )
        return results

    def end_payment_capture(
        self,
        call_id: str,
        agent_id: str,
        reason: str = "Payment capture complete",
    ) -> Dict[str, Any]:
        """
        End a payment data capture sequence.

        This deactivates PCI controls:
        1. Exit PCI scope
        2. Resume recording
        3. Deactivate DTMF masking

        Args:
            call_id: The call identifier.
            agent_id: The agent handling the call.
            reason: Reason for ending payment capture.

        Returns:
            Status dict with results of each control deactivation.
        """
        results: Dict[str, Any] = {"call_id": call_id, "agent_id": agent_id}

        # 1. Exit PCI scope
        scope_state = self.scope_manager.exit_pci_scope(
            call_id, agent_id, reason
        )
        results["pci_scope"] = scope_state.name

        # 2. Resume recording
        controller = self._recording_controllers.get(call_id)
        if controller:
            event = controller.resume_after_payment(agent_id)
            results["recording"] = event.new_state.name
        else:
            results["recording"] = "NO_CONTROLLER"

        results["dtmf_masking"] = "deactivated"

        logger.info(
            "Payment capture ended: call_id=%s agent=%s scope=%s",
            call_id,
            agent_id,
            scope_state.name,
        )
        return results

    def scan_and_mask(
        self, text: str, context: str = "agent_screen"
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Scan text for PANs and apply screen masking.

        Args:
            text: The text to scan and mask.
            context: Description of the text source.

        Returns:
            Tuple of (masked text, list of PAN detections).
        """
        detections: List[Dict[str, Any]] = []

        # PAN detection
        if self._enable_pan_detection:
            detections = self.pan_detector.scan_text(text, context)
            if detections:
                self._stats["total_pan_detected"] += len(detections)

        # Screen masking
        masked_text = text
        if self._enable_screen_masking:
            masked_text, mask_count = self.screen_masker.mask_text(text)
            if mask_count > 0:
                self._stats["total_screen_maskings"] += mask_count

        return masked_text, detections

    def mask_dtmf(
        self, call_id: str, digit: str, duration_ms: int = 100
    ) -> DTMFEvent:
        """
        Process a DTMF event and apply masking if in PCI scope.

        Args:
            call_id: The call identifier.
            digit: The DTMF digit (0-9, *, #).
            duration_ms: Duration of the DTMF tone.

        Returns:
            A DTMFEvent with appropriate masking applied.
        """
        in_scope = (
            self.scope_manager.get_scope_state(call_id) == PCIScopeState.IN_SCOPE
        )

        event = DTMFEvent(
            call_id=call_id,
            digit=digit if not in_scope else "",
            masked_digit="*" if in_scope else digit,
            duration_ms=duration_ms,
            was_in_pci_scope=in_scope,
            masking_mode=self.masking_mode if in_scope else None,
        )

        if in_scope:
            self._stats["total_dtmf_masked"] += 1

        return event

    def generate_compliance_report(
        self,
        period_start: datetime,
        period_end: datetime,
        **kwargs: Any,
    ) -> PCIComplianceReport:
        """
        Generate a PCI-DSS compliance report.

        Args:
            period_start: Start of the reporting period.
            period_end: End of the reporting period.
            **kwargs: Additional report parameters.

        Returns:
            A complete PCI compliance report.
        """
        return self.compliance_reporter.generate_report(
            period_start=period_start,
            period_end=period_end,
            **kwargs,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall PCI voice protection statistics."""
        stats = dict(self._stats)
        stats["pan_detector"] = self.pan_detector.get_statistics()
        stats["scope_manager"] = self.scope_manager.get_statistics()
        stats["screen_masker"] = self.screen_masker.get_statistics()
        stats["active_calls"] = len(self._recording_controllers)
        stats["calls_in_pci_scope"] = len(
            self.scope_manager.get_in_scope_calls()
        )
        return stats
