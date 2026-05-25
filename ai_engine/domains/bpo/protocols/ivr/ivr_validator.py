"""
IVR Security Validator

Validates IVR interactions for security compliance including:
- PCI DSS compliance for DTMF payment capture
- Automatic DTMF masking trigger on payment pattern detection
- IVR navigation path validation (automated probing/brute force)
- DTMF input rate limiting (automated attack prevention)
- Transfer destination validation (toll fraud prevention)
- Comprehensive interaction logging for compliance audit
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
)
from ai_engine.domains.bpo.protocols.ivr.ivr_message import (
    DTMFEvent,
    IVRFlow,
    IVRMenuNode,
    IVRMessage,
    IVRMessageType,
    IVRTransferType,
    PCIPaymentCapture,
)


# Premium rate and toll fraud number prefixes
TOLL_FRAUD_PREFIXES: Dict[str, List[str]] = {
    "US_PREMIUM": ["900", "976"],
    "UK_PREMIUM": ["09"],
    "IRSF": ["882", "883"],              # International Revenue Share Fraud
    "SATELLITE": ["8816", "8817"],        # Satellite phone (high cost)
    "CUBAN": ["53"],                      # Commonly used for IRSF
    "SOMALI": ["252"],                    # Commonly used for IRSF
}

# Known safe country code prefixes
SAFE_COUNTRY_PREFIXES: Set[str] = {
    "1",    # US/Canada
    "44",   # UK
    "61",   # Australia
    "49",   # Germany
    "33",   # France
    "81",   # Japan
    "91",   # India
}


@dataclass
class IVRSessionState:
    """Tracks the state of an IVR session for security monitoring."""

    session_id: str
    call_id: str = ""
    caller_id: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # Navigation tracking
    nodes_visited: List[str] = field(default_factory=list)
    node_visit_times: List[datetime] = field(default_factory=list)
    invalid_inputs: int = 0
    total_inputs: int = 0
    timeout_count: int = 0

    # DTMF tracking
    dtmf_events: List[DTMFEvent] = field(default_factory=list)
    dtmf_buffer: str = ""                # Accumulated DTMF digits
    last_dtmf_time: Optional[datetime] = None
    dtmf_rate_violations: int = 0

    # Payment state
    active_payment: Optional[PCIPaymentCapture] = None
    payment_attempts: int = 0

    # Security state
    is_flagged: bool = False
    flag_reasons: List[str] = field(default_factory=list)
    is_terminated: bool = False


@dataclass
class IVRSecurityPolicy:
    """Configuration for IVR security policies."""

    # DTMF rate limiting
    max_dtmf_per_second: int = 8           # Human can press ~4-5/sec
    max_dtmf_per_minute: int = 120
    dtmf_rate_window_seconds: int = 5

    # Navigation limits
    max_invalid_inputs: int = 10           # Before flagging as probe
    max_node_visits_per_minute: int = 30   # Menu navigation rate limit
    max_total_nodes_per_session: int = 200
    max_retry_count: int = 5              # Per menu node

    # PCI compliance
    auto_mask_on_payment_pattern: bool = True
    card_number_min_sequential_digits: int = 6  # Digits before auto-mask triggers
    payment_capture_timeout_seconds: int = 120
    max_payment_attempts: int = 3

    # Transfer validation
    block_premium_rate_transfers: bool = True
    block_international_transfers: bool = False
    allowed_transfer_prefixes: Set[str] = field(default_factory=set)
    blocked_transfer_prefixes: Set[str] = field(default_factory=set)
    max_external_transfers_per_session: int = 3

    # Session limits
    max_session_duration_seconds: int = 3600  # 1 hour in IVR
    idle_timeout_seconds: int = 30            # IVR inactivity timeout

    # Logging
    log_all_dtmf: bool = False               # PCI: must not log card digits
    log_navigation: bool = True
    log_transfers: bool = True
    mask_caller_id_in_logs: bool = True


class IVRValidator(BaseValidator):
    """
    Security validator for IVR interactions.

    Monitors IVR sessions in real-time to enforce:
    - PCI DSS compliance for DTMF-based payment capture
    - Automated attack and probing detection
    - Toll fraud prevention on transfers
    - Rate limiting for DTMF input
    - Navigation pattern anomaly detection
    """

    # Pattern for detecting sequential digit input (potential card number)
    CARD_START_PATTERNS = [
        re.compile(r"^4\d{5}"),          # Visa
        re.compile(r"^5[1-5]\d{4}"),     # Mastercard
        re.compile(r"^3[47]\d{4}"),      # Amex
        re.compile(r"^6(?:011|5\d{2})\d{2}"),  # Discover
        re.compile(r"^35\d{4}"),         # JCB
    ]

    def __init__(
        self,
        policy: Optional[IVRSecurityPolicy] = None,
        strict: bool = True,
    ):
        """
        Initialize the IVR validator.

        Args:
            policy: Security policy configuration
            strict: If True, treat warnings as errors
        """
        super().__init__(strict)
        self.policy = policy or IVRSecurityPolicy()
        self._sessions: Dict[str, IVRSessionState] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._caller_session_count: Dict[str, int] = defaultdict(int)

    @property
    def name(self) -> str:
        return "IVRValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate IVR data.

        Args:
            data: Can be IVRMessage, DTMFEvent, PCIPaymentCapture, or dict

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, IVRMessage):
            self._validate_message(data, result)
        elif isinstance(data, DTMFEvent):
            self._validate_dtmf_event(data, result)
        elif isinstance(data, PCIPaymentCapture):
            self._validate_payment_capture(data, result)
        elif isinstance(data, dict):
            self._validate_dict(data, result)
        else:
            result.add_error(
                "IVR_INVALID_INPUT",
                "Input must be an IVRMessage, DTMFEvent, PCIPaymentCapture, or dict",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def validate_dtmf_sequence(
        self,
        session_id: str,
        event: DTMFEvent,
    ) -> Tuple[ValidationResult, bool]:
        """
        Validate a DTMF input event for PCI compliance.

        Detects payment card number patterns in DTMF sequences
        and automatically triggers masking when detected.

        Args:
            session_id: IVR session identifier
            event: The DTMF event to validate

        Returns:
            Tuple of (ValidationResult, should_mask) where should_mask
            indicates whether DTMF masking should be activated
        """
        result = self._create_result()
        should_mask = False

        session = self._get_or_create_session(session_id, event.call_id)
        now = datetime.now()
        session.last_activity = now
        session.total_inputs += 1

        # Rate limiting check
        rate_result = self._check_dtmf_rate(session, now)
        result.merge(rate_result)

        # Record the event
        session.dtmf_events.append(event)
        session.last_dtmf_time = now

        # Accumulate digits in buffer
        if event.digit.isdigit():
            session.dtmf_buffer += event.digit
        elif event.digit in ("#", "*"):
            # Hash/star typically terminates input; reset buffer
            session.dtmf_buffer = ""

        # Check for payment card pattern
        if (
            self.policy.auto_mask_on_payment_pattern
            and len(session.dtmf_buffer) >= self.policy.card_number_min_sequential_digits
        ):
            if self._looks_like_card_number(session.dtmf_buffer):
                should_mask = True

                # Auto-start payment capture if not already active
                if not session.active_payment:
                    session.active_payment = PCIPaymentCapture(
                        call_id=event.call_id,
                        session_id=session_id,
                    )
                    session.active_payment.start_capture()
                    session.payment_attempts += 1

                    result.add_warning(
                        "IVR_PCI_MASK_TRIGGERED",
                        f"Payment card pattern detected after "
                        f"{len(session.dtmf_buffer)} digits. "
                        f"DTMF masking activated.",
                        field="dtmf_sequence",
                    )

                    self._audit_log_entry(
                        action="PCI_MASKING_ACTIVATED",
                        session_id=session_id,
                        call_id=event.call_id,
                        details={
                            "trigger_length": len(session.dtmf_buffer),
                            "payment_attempt": session.payment_attempts,
                        },
                    )

                # Feed digit to payment capture
                event.masked = True
                session.active_payment.receive_digit(event.digit)

        # If payment capture is active, mask all digits
        if session.active_payment and session.active_payment.masking_active:
            event.masked = True
            should_mask = True

        # Log the DTMF event (masked if payment is active)
        if self.policy.log_all_dtmf or not event.masked:
            self._audit_log_entry(
                action="DTMF_INPUT",
                session_id=session_id,
                call_id=event.call_id,
                details={
                    "digit": event.masked_digit,
                    "masked": event.masked,
                    "duration_ms": event.duration_ms,
                },
            )

        return result, should_mask

    def validate_navigation(
        self,
        session_id: str,
        node_id: str,
        digit_pressed: str,
        flow: IVRFlow,
    ) -> ValidationResult:
        """
        Validate an IVR menu navigation event.

        Detects automated probing and brute force attacks on the
        IVR menu system by monitoring navigation patterns.

        Args:
            session_id: IVR session identifier
            node_id: Current menu node ID
            digit_pressed: DTMF digit that triggered navigation
            flow: The IVR flow definition

        Returns:
            ValidationResult with any security concerns
        """
        result = self._create_result()

        session = self._get_or_create_session(session_id)
        now = datetime.now()
        session.last_activity = now
        session.nodes_visited.append(node_id)
        session.node_visit_times.append(now)

        current_node = flow.get_node(node_id)
        if not current_node:
            result.add_error(
                "IVR_UNKNOWN_NODE",
                f"Navigation to unknown node: {node_id}",
                field="node_id",
            )
            return result

        # Check if the digit is a valid option
        child_id = current_node.get_child_id(digit_pressed)
        if child_id is None:
            session.invalid_inputs += 1
            if session.invalid_inputs > self.policy.max_invalid_inputs:
                result.add_error(
                    "IVR_AUTOMATED_PROBE",
                    f"Excessive invalid inputs ({session.invalid_inputs}) "
                    f"suggests automated probing or brute force attack",
                    field="navigation",
                    severity=ValidationSeverity.CRITICAL,
                )
                session.is_flagged = True
                session.flag_reasons.append("automated_probe_detection")

                self._audit_log_entry(
                    action="SECURITY_PROBE_DETECTED",
                    session_id=session_id,
                    details={
                        "invalid_inputs": session.invalid_inputs,
                        "last_node": node_id,
                        "last_digit": digit_pressed,
                    },
                )
            else:
                result.add_warning(
                    "IVR_INVALID_INPUT",
                    f"Invalid input '{digit_pressed}' at node '{node_id}'. "
                    f"Valid options: {list(current_node.options.keys())}",
                    field="digit_pressed",
                )

        # Check navigation rate
        recent_visits = [
            t for t in session.node_visit_times
            if (now - t).total_seconds() < 60
        ]
        if len(recent_visits) > self.policy.max_node_visits_per_minute:
            result.add_error(
                "IVR_RAPID_NAVIGATION",
                f"Abnormal navigation rate: {len(recent_visits)} "
                f"nodes/minute (limit: {self.policy.max_node_visits_per_minute})",
                field="navigation_rate",
                severity=ValidationSeverity.CRITICAL,
            )
            session.is_flagged = True
            session.flag_reasons.append("rapid_navigation")

        # Check total nodes visited
        if len(session.nodes_visited) > self.policy.max_total_nodes_per_session:
            result.add_warning(
                "IVR_EXCESSIVE_NAVIGATION",
                f"Session has visited {len(session.nodes_visited)} nodes "
                f"(threshold: {self.policy.max_total_nodes_per_session})",
                field="total_navigation",
            )

        # Check authentication requirement
        if current_node.requires_authentication:
            # This is a policy check - the caller should be authenticated
            result.add_warning(
                "IVR_AUTH_REQUIRED_NODE",
                f"Node '{node_id}' requires authentication. "
                f"Ensure caller is verified before proceeding.",
                field="authentication",
            )

        if self.policy.log_navigation:
            self._audit_log_entry(
                action="IVR_NAVIGATION",
                session_id=session_id,
                details={
                    "from_node": node_id,
                    "digit": digit_pressed,
                    "to_node": child_id or "INVALID",
                    "is_valid": child_id is not None,
                },
            )

        return result

    def validate_transfer_destination(
        self,
        session_id: str,
        destination: str,
        transfer_type: IVRTransferType,
    ) -> ValidationResult:
        """
        Validate a call transfer destination to prevent toll fraud.

        Checks the destination number against known premium rate,
        IRSF, and blocked number patterns.

        Args:
            session_id: IVR session identifier
            destination: The destination phone number or queue name
            transfer_type: Type of transfer being attempted

        Returns:
            ValidationResult indicating whether the transfer is allowed
        """
        result = self._create_result()

        session = self._sessions.get(session_id)
        if not session:
            session = self._get_or_create_session(session_id)

        # Internal transfers (agent, queue) are generally safe
        if transfer_type in (IVRTransferType.AGENT, IVRTransferType.QUEUE):
            self._audit_log_entry(
                action="TRANSFER_VALIDATED",
                session_id=session_id,
                details={
                    "destination": destination,
                    "type": transfer_type.value,
                    "result": "allowed",
                },
            )
            return result

        # External transfers require validation
        clean_number = re.sub(r"[\s\-.()+]", "", destination)

        # Check premium rate numbers
        if self.policy.block_premium_rate_transfers:
            for category, prefixes in TOLL_FRAUD_PREFIXES.items():
                for prefix in prefixes:
                    if clean_number.startswith(prefix):
                        result.add_error(
                            "IVR_TOLL_FRAUD_BLOCKED",
                            f"Transfer to {category} number blocked: "
                            f"{self._mask_phone(destination)}",
                            field="destination",
                            severity=ValidationSeverity.CRITICAL,
                        )
                        self._audit_log_entry(
                            action="TOLL_FRAUD_BLOCKED",
                            session_id=session_id,
                            details={
                                "destination": self._mask_phone(destination),
                                "category": category,
                                "prefix_matched": prefix,
                            },
                        )
                        return result

        # Check blocked prefixes
        for prefix in self.policy.blocked_transfer_prefixes:
            if clean_number.startswith(prefix):
                result.add_error(
                    "IVR_BLOCKED_DESTINATION",
                    f"Transfer to blocked destination: "
                    f"{self._mask_phone(destination)}",
                    field="destination",
                    severity=ValidationSeverity.CRITICAL,
                )
                return result

        # Check allowed prefixes (if configured)
        if self.policy.allowed_transfer_prefixes:
            allowed = any(
                clean_number.startswith(p)
                for p in self.policy.allowed_transfer_prefixes
            )
            if not allowed:
                result.add_error(
                    "IVR_UNAUTHORIZED_DESTINATION",
                    f"Transfer destination not in allowed list: "
                    f"{self._mask_phone(destination)}",
                    field="destination",
                )
                return result

        # Check international transfer blocking
        if self.policy.block_international_transfers:
            is_domestic = (
                clean_number.startswith("1")
                or len(clean_number) <= 10
                or clean_number.startswith("+1")
            )
            if not is_domestic:
                result.add_error(
                    "IVR_INTERNATIONAL_BLOCKED",
                    "International transfers are not permitted",
                    field="destination",
                )
                return result

        if self.policy.log_transfers:
            self._audit_log_entry(
                action="TRANSFER_VALIDATED",
                session_id=session_id,
                details={
                    "destination": self._mask_phone(destination),
                    "type": transfer_type.value,
                    "result": "allowed",
                },
            )

        return result

    def validate_dtmf_rate(
        self,
        session_id: str,
    ) -> ValidationResult:
        """
        Check the DTMF input rate for the session.

        Excessive DTMF input rates indicate automated attacks
        rather than human interaction.

        Args:
            session_id: IVR session identifier

        Returns:
            ValidationResult with rate limit violations
        """
        result = self._create_result()
        session = self._sessions.get(session_id)

        if not session:
            return result

        now = datetime.now()
        return self._check_dtmf_rate(session, now)

    def get_audit_log(
        self,
        session_id: Optional[str] = None,
        call_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit log entries with optional filtering.

        Args:
            session_id: Filter by session ID
            call_id: Filter by call ID
            action: Filter by action type
            since: Only return entries after this time

        Returns:
            List of audit log entry dictionaries
        """
        entries = self._audit_log

        if session_id:
            entries = [e for e in entries if e.get("session_id") == session_id]
        if call_id:
            entries = [e for e in entries if e.get("call_id") == call_id]
        if action:
            entries = [e for e in entries if e.get("action") == action]
        if since:
            entries = [
                e for e in entries
                if datetime.fromisoformat(e["timestamp"]) >= since
            ]

        return entries

    def terminate_session(self, session_id: str) -> None:
        """
        Clean up IVR session state.

        Args:
            session_id: Session identifier to terminate
        """
        session = self._sessions.get(session_id)
        if session:
            # Cancel any active payment capture
            if session.active_payment and not session.active_payment.is_complete:
                session.active_payment.cancel()

            self._audit_log_entry(
                action="SESSION_TERMINATED",
                session_id=session_id,
                call_id=session.call_id,
                details={
                    "duration_seconds": (
                        datetime.now() - session.started_at
                    ).total_seconds(),
                    "total_inputs": session.total_inputs,
                    "invalid_inputs": session.invalid_inputs,
                    "nodes_visited": len(session.nodes_visited),
                    "payment_attempts": session.payment_attempts,
                    "is_flagged": session.is_flagged,
                    "flag_reasons": session.flag_reasons,
                },
            )

            if session.caller_id:
                self._caller_session_count[session.caller_id] = max(
                    0, self._caller_session_count[session.caller_id] - 1
                )
            del self._sessions[session_id]

    def _check_dtmf_rate(
        self, session: IVRSessionState, now: datetime
    ) -> ValidationResult:
        """Check DTMF input rate against limits."""
        result = self._create_result()

        # Count recent DTMF events
        window_start = now - timedelta(
            seconds=self.policy.dtmf_rate_window_seconds
        )
        recent_events = [
            e for e in session.dtmf_events
            if e.timestamp >= window_start
        ]

        rate_per_second = len(recent_events) / max(
            self.policy.dtmf_rate_window_seconds, 1
        )

        if rate_per_second > self.policy.max_dtmf_per_second:
            session.dtmf_rate_violations += 1
            result.add_error(
                "IVR_DTMF_RATE_EXCEEDED",
                f"DTMF input rate {rate_per_second:.1f}/sec exceeds "
                f"limit of {self.policy.max_dtmf_per_second}/sec. "
                f"Possible automated attack.",
                field="dtmf_rate",
                severity=ValidationSeverity.CRITICAL,
            )

            if session.dtmf_rate_violations >= 3:
                session.is_flagged = True
                session.flag_reasons.append("excessive_dtmf_rate")

                self._audit_log_entry(
                    action="AUTOMATED_ATTACK_DETECTED",
                    session_id=session.session_id,
                    call_id=session.call_id,
                    details={
                        "rate_per_second": rate_per_second,
                        "violations": session.dtmf_rate_violations,
                    },
                )

        # Check per-minute rate
        minute_start = now - timedelta(seconds=60)
        minute_events = [
            e for e in session.dtmf_events
            if e.timestamp >= minute_start
        ]
        if len(minute_events) > self.policy.max_dtmf_per_minute:
            result.add_warning(
                "IVR_DTMF_MINUTE_RATE",
                f"DTMF input count {len(minute_events)}/min exceeds "
                f"limit of {self.policy.max_dtmf_per_minute}/min",
                field="dtmf_rate",
            )

        return result

    def _looks_like_card_number(self, digits: str) -> bool:
        """
        Check if an accumulated digit sequence looks like a card number.

        Uses BIN (Bank Identification Number) prefix patterns to detect
        probable credit/debit card number entry.

        Args:
            digits: Accumulated DTMF digit string

        Returns:
            True if the digit sequence matches a card number pattern
        """
        for pattern in self.CARD_START_PATTERNS:
            if pattern.match(digits):
                return True
        return False

    def _validate_message(
        self, msg: IVRMessage, result: ValidationResult
    ) -> None:
        """Validate an IVRMessage object."""
        if not msg.call_id:
            result.add_error(
                "IVR_MISSING_CALL_ID",
                "Call ID is required for IVR messages",
                field="call_id",
            )

        if not msg.session_id:
            result.add_warning(
                "IVR_MISSING_SESSION_ID",
                "Session ID should be provided for tracking",
                field="session_id",
            )

        if msg.message_type == IVRMessageType.TRANSFER:
            destination = msg.payload.get("destination", "")
            if not destination:
                result.add_error(
                    "IVR_MISSING_TRANSFER_DEST",
                    "Transfer destination is required",
                    field="payload/destination",
                )

        if msg.message_type == IVRMessageType.DTMF_INPUT:
            digit = msg.payload.get("digit", "")
            valid_digits = set("0123456789*#ABCD")
            if digit and digit not in valid_digits:
                result.add_error(
                    "IVR_INVALID_DTMF",
                    f"Invalid DTMF digit: {digit}",
                    field="payload/digit",
                )

    def _validate_dtmf_event(
        self, event: DTMFEvent, result: ValidationResult
    ) -> None:
        """Validate a DTMFEvent object."""
        valid_digits = set("0123456789*#ABCD")
        if event.digit not in valid_digits:
            result.add_error(
                "IVR_INVALID_DTMF_DIGIT",
                f"Invalid DTMF digit: {event.digit}",
                field="digit",
            )

        if event.duration_ms < 40:
            result.add_warning(
                "IVR_SHORT_DTMF",
                f"DTMF duration {event.duration_ms}ms is below minimum "
                f"(40ms). May indicate automated input.",
                field="duration_ms",
            )

        if event.duration_ms > 10000:
            result.add_warning(
                "IVR_LONG_DTMF",
                f"DTMF duration {event.duration_ms}ms is unusually long",
                field="duration_ms",
            )

    def _validate_payment_capture(
        self, capture: PCIPaymentCapture, result: ValidationResult
    ) -> None:
        """Validate a PCIPaymentCapture state."""
        if capture.is_timed_out:
            result.add_error(
                "IVR_PAYMENT_TIMEOUT",
                "Payment capture has timed out",
                field="payment",
                severity=ValidationSeverity.ERROR,
            )

        if capture.masking_active and not capture.capture_started_at:
            result.add_error(
                "IVR_PCI_STATE_ERROR",
                "DTMF masking is active but capture was not started",
                field="masking_active",
            )

        if capture.card_digits_received > capture.max_card_digits:
            result.add_error(
                "IVR_PCI_OVERFLOW",
                f"Card digits ({capture.card_digits_received}) exceed "
                f"maximum ({capture.max_card_digits})",
                field="card_digits",
            )

    def _validate_dict(
        self, data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate IVR data from a dictionary."""
        if "call_id" not in data:
            result.add_error(
                "IVR_MISSING_CALL_ID",
                "Call ID is required",
                field="call_id",
            )

        if "message_type" in data:
            valid_types = {t.msg_type for t in IVRMessageType}
            if data["message_type"] not in valid_types:
                result.add_error(
                    "IVR_INVALID_MESSAGE_TYPE",
                    f"Unknown message type: {data['message_type']}",
                    field="message_type",
                )

    def _get_or_create_session(
        self,
        session_id: str,
        call_id: str = "",
        caller_id: str = "",
    ) -> IVRSessionState:
        """Get an existing session or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = IVRSessionState(
                session_id=session_id,
                call_id=call_id,
                caller_id=caller_id,
            )
            if caller_id:
                self._caller_session_count[caller_id] += 1
        return self._sessions[session_id]

    def _mask_phone(self, phone: str) -> str:
        """Mask a phone number for logging, keeping last 4 digits."""
        clean = re.sub(r"[^\d+]", "", phone)
        if len(clean) > 4:
            return "*" * (len(clean) - 4) + clean[-4:]
        return clean

    def _audit_log_entry(
        self,
        action: str,
        session_id: str = "",
        call_id: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an entry in the audit log."""
        session = self._sessions.get(session_id)
        caller_id = ""
        if session and session.caller_id:
            if self.policy.mask_caller_id_in_logs:
                caller_id = self._mask_phone(session.caller_id)
            else:
                caller_id = session.caller_id

        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "session_id": session_id,
            "call_id": call_id or (session.call_id if session else ""),
            "caller_id": caller_id,
            "details": details or {},
        }
        self._audit_log.append(entry)
