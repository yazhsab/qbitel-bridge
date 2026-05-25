"""
TN3270e Terminal Session Security Validator

Validates terminal session security including:
- Session establishment authorization
- Data exfiltration detection (abnormal screen scraping)
- Credential stuffing detection (rapid login attempts)
- Sensitive screen access monitoring
- Protected field tampering detection
- Session timeout enforcement
- Comprehensive audit logging of terminal actions
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
)
from ai_engine.domains.bpo.protocols.terminal.tn3270e_message import (
    AIDCode,
    ScreenBuffer,
    ScreenField,
    SecurityEvent,
    SecurityEventType,
    TN3270eMessage,
)
from ai_engine.domains.bpo.protocols.terminal.tn3270e_parser import (
    TN3270eParser,
    SensitiveDataMatch,
)


@dataclass
class SessionState:
    """Tracks the state of a terminal session for security monitoring."""

    session_id: str
    agent_id: str = ""
    source_ip: str = ""
    established_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    is_authenticated: bool = False
    login_attempts: int = 0
    last_login_attempt: Optional[datetime] = None
    screens_accessed: List[str] = field(default_factory=list)
    screen_access_times: List[datetime] = field(default_factory=list)
    sensitive_screens_accessed: int = 0
    total_keystrokes: int = 0
    aid_history: List[int] = field(default_factory=list)
    field_modifications: int = 0
    protected_field_violations: int = 0
    data_read_count: int = 0
    security_events: List[SecurityEvent] = field(default_factory=list)
    current_screen_name: str = ""
    previous_screen_name: str = ""
    is_timed_out: bool = False


@dataclass
class TN3270eSecurityPolicy:
    """Configuration for terminal session security policies."""

    # Session limits
    session_timeout_seconds: int = 900            # 15 minutes idle timeout
    max_session_duration_seconds: int = 28800     # 8 hours max session
    max_concurrent_sessions_per_agent: int = 1

    # Login security
    max_login_attempts: int = 3
    login_lockout_duration_seconds: int = 300     # 5-minute lockout
    login_attempt_window_seconds: int = 60        # Window for counting attempts

    # Data exfiltration thresholds
    max_screens_per_minute: int = 30              # Screen navigation rate limit
    max_sensitive_screens_per_session: int = 100
    max_data_reads_per_minute: int = 60
    screen_scraping_threshold: int = 50           # Reads in rapid succession

    # Protected field enforcement
    reject_protected_field_modifications: bool = True
    max_protected_field_violations: int = 3       # Before session termination

    # Monitoring
    log_all_keystrokes: bool = False              # PCI: only log non-sensitive
    log_screen_transitions: bool = True
    log_sensitive_data_access: bool = True
    alert_on_sensitive_data: bool = True

    # Allowed operations
    allowed_transaction_screens: Set[str] = field(default_factory=set)
    restricted_screens: Set[str] = field(default_factory=set)


class TN3270eValidator(BaseValidator):
    """
    Security validator for TN3270e terminal sessions.

    Monitors terminal sessions in real-time to detect:
    - Unauthorized access attempts
    - Screen scraping and data exfiltration
    - Credential stuffing attacks
    - Protected field tampering
    - Abnormal navigation patterns
    - Session timeout violations
    """

    def __init__(
        self,
        policy: Optional[TN3270eSecurityPolicy] = None,
        strict: bool = True,
    ):
        """
        Initialize the TN3270e validator.

        Args:
            policy: Security policy configuration
            strict: If True, treat warnings as errors
        """
        super().__init__(strict)
        self.policy = policy or TN3270eSecurityPolicy()
        self.parser = TN3270eParser(strict=False)
        self._sessions: Dict[str, SessionState] = {}
        self._locked_agents: Dict[str, datetime] = {}
        self._agent_session_count: Dict[str, int] = defaultdict(int)
        self._audit_log: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "TN3270eValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate terminal session data.

        Args:
            data: Can be a TN3270eMessage, ScreenBuffer, SessionState, or dict

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, TN3270eMessage):
            self._validate_message(data, result)
        elif isinstance(data, ScreenBuffer):
            self._validate_screen(data, result)
        elif isinstance(data, SessionState):
            self._validate_session_state(data, result)
        elif isinstance(data, dict):
            self._validate_dict(data, result)
        else:
            result.add_error(
                "TN3270E_INVALID_INPUT",
                "Input must be a TN3270eMessage, ScreenBuffer, SessionState, or dict",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def validate_session_establishment(
        self,
        session_id: str,
        agent_id: str,
        source_ip: str,
    ) -> ValidationResult:
        """
        Validate a new terminal session establishment.

        Checks for:
        - Agent lockout status
        - Concurrent session limits
        - Source IP authorization

        Args:
            session_id: Unique session identifier
            agent_id: Agent attempting to establish session
            source_ip: Source IP address of the connection

        Returns:
            ValidationResult indicating whether session is allowed
        """
        result = self._create_result()

        # Check if agent is locked out
        if agent_id in self._locked_agents:
            lockout_expires = self._locked_agents[agent_id]
            if datetime.now() < lockout_expires:
                remaining = (lockout_expires - datetime.now()).seconds
                result.add_error(
                    "TN3270E_AGENT_LOCKED_OUT",
                    f"Agent {agent_id} is locked out for {remaining} more seconds "
                    f"due to excessive login failures",
                    field="agent_id",
                    severity=ValidationSeverity.CRITICAL,
                )
                self._record_security_event(
                    session_id=session_id,
                    event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
                    severity="CRITICAL",
                    description=f"Locked-out agent {agent_id} attempted session from {source_ip}",
                    agent_id=agent_id,
                    source_ip=source_ip,
                )
                return result
            else:
                del self._locked_agents[agent_id]

        # Check concurrent session limit
        if self._agent_session_count[agent_id] >= self.policy.max_concurrent_sessions_per_agent:
            result.add_error(
                "TN3270E_CONCURRENT_SESSION_LIMIT",
                f"Agent {agent_id} has reached the maximum of "
                f"{self.policy.max_concurrent_sessions_per_agent} concurrent sessions",
                field="agent_id",
                severity=ValidationSeverity.CRITICAL,
            )
            self._record_security_event(
                session_id=session_id,
                event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
                severity="WARNING",
                description=f"Agent {agent_id} exceeded concurrent session limit",
                agent_id=agent_id,
                source_ip=source_ip,
            )
            return result

        # Create session state
        session = SessionState(
            session_id=session_id,
            agent_id=agent_id,
            source_ip=source_ip,
        )
        self._sessions[session_id] = session
        self._agent_session_count[agent_id] += 1

        self._audit_log_entry(
            action="SESSION_ESTABLISHED",
            session_id=session_id,
            agent_id=agent_id,
            source_ip=source_ip,
        )

        return result

    def validate_login_attempt(
        self,
        session_id: str,
        success: bool,
    ) -> ValidationResult:
        """
        Validate and track a login attempt.

        Detects credential stuffing by monitoring rapid failed logins.

        Args:
            session_id: Session identifier
            success: Whether the login attempt succeeded

        Returns:
            ValidationResult with credential stuffing warnings
        """
        result = self._create_result()
        session = self._sessions.get(session_id)

        if not session:
            result.add_error(
                "TN3270E_UNKNOWN_SESSION",
                f"No session found for ID {session_id}",
                severity=ValidationSeverity.CRITICAL,
            )
            return result

        now = datetime.now()
        session.last_activity = now

        if success:
            session.is_authenticated = True
            session.login_attempts = 0
            self._audit_log_entry(
                action="LOGIN_SUCCESS",
                session_id=session_id,
                agent_id=session.agent_id,
            )
            return result

        # Failed login
        session.login_attempts += 1
        session.last_login_attempt = now

        self._audit_log_entry(
            action="LOGIN_FAILED",
            session_id=session_id,
            agent_id=session.agent_id,
            details={"attempt_number": session.login_attempts},
        )

        # Check for credential stuffing
        if session.login_attempts >= self.policy.max_login_attempts:
            # Lock out the agent
            lockout_until = now + timedelta(
                seconds=self.policy.login_lockout_duration_seconds
            )
            self._locked_agents[session.agent_id] = lockout_until

            result.add_error(
                "TN3270E_CREDENTIAL_STUFFING",
                f"Agent {session.agent_id} locked out after "
                f"{session.login_attempts} failed login attempts",
                field="login",
                severity=ValidationSeverity.CRITICAL,
            )

            self._record_security_event(
                session_id=session_id,
                event_type=SecurityEventType.CREDENTIAL_STUFFING,
                severity="CRITICAL",
                description=(
                    f"Credential stuffing detected: {session.login_attempts} "
                    f"failed attempts from {session.source_ip}"
                ),
                agent_id=session.agent_id,
                source_ip=session.source_ip,
            )
        elif session.login_attempts >= 2:
            result.add_warning(
                "TN3270E_LOGIN_FAILURES",
                f"Agent {session.agent_id} has {session.login_attempts} "
                f"consecutive failed login attempts",
                field="login",
            )

        return result

    def validate_screen_access(
        self,
        session_id: str,
        screen: ScreenBuffer,
        screen_name: str = "",
    ) -> ValidationResult:
        """
        Validate access to a terminal screen.

        Monitors for:
        - Rapid screen navigation (scraping detection)
        - Access to sensitive screens
        - Restricted screen access
        - Sensitive data exposure

        Args:
            session_id: Session identifier
            screen: Current screen buffer
            screen_name: Application-level screen name

        Returns:
            ValidationResult with any security concerns
        """
        result = self._create_result()
        session = self._sessions.get(session_id)

        if not session:
            result.add_error(
                "TN3270E_UNKNOWN_SESSION",
                f"No session found for ID {session_id}",
                severity=ValidationSeverity.CRITICAL,
            )
            return result

        now = datetime.now()
        session.last_activity = now
        session.previous_screen_name = session.current_screen_name
        session.current_screen_name = screen_name
        session.screens_accessed.append(screen_name)
        session.screen_access_times.append(now)

        # Check session timeout
        timeout_result = self.check_session_timeout(session_id)
        if not timeout_result.is_valid:
            result.merge(timeout_result)
            return result

        # Check restricted screens
        if screen_name and screen_name.upper() in self.policy.restricted_screens:
            result.add_error(
                "TN3270E_RESTRICTED_SCREEN",
                f"Access to restricted screen '{screen_name}' is not permitted",
                field="screen_name",
                severity=ValidationSeverity.CRITICAL,
            )
            self._record_security_event(
                session_id=session_id,
                event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
                severity="CRITICAL",
                description=f"Agent attempted to access restricted screen: {screen_name}",
                agent_id=session.agent_id,
                screen_name=screen_name,
            )
            return result

        # Check screen navigation rate (scraping detection)
        recent_accesses = [
            t for t in session.screen_access_times
            if (now - t).total_seconds() < 60
        ]
        if len(recent_accesses) > self.policy.max_screens_per_minute:
            result.add_error(
                "TN3270E_SCREEN_SCRAPING",
                f"Abnormal screen navigation rate detected: "
                f"{len(recent_accesses)} screens in the last minute "
                f"(limit: {self.policy.max_screens_per_minute})",
                field="navigation",
                severity=ValidationSeverity.CRITICAL,
            )
            self._record_security_event(
                session_id=session_id,
                event_type=SecurityEventType.DATA_EXFILTRATION,
                severity="CRITICAL",
                description=(
                    f"Screen scraping detected: {len(recent_accesses)} "
                    f"screens/minute from {session.source_ip}"
                ),
                agent_id=session.agent_id,
                screen_name=screen_name,
                details={"rate": len(recent_accesses)},
            )

        # Check for sensitive screen access
        is_sensitive = self.parser.detect_sensitive_screen(screen)
        if is_sensitive:
            session.sensitive_screens_accessed += 1

            if self.policy.log_sensitive_data_access:
                self._audit_log_entry(
                    action="SENSITIVE_SCREEN_ACCESS",
                    session_id=session_id,
                    agent_id=session.agent_id,
                    details={
                        "screen_name": screen_name,
                        "sensitive_count": session.sensitive_screens_accessed,
                    },
                )

            if session.sensitive_screens_accessed > self.policy.max_sensitive_screens_per_session:
                result.add_warning(
                    "TN3270E_EXCESSIVE_SENSITIVE_ACCESS",
                    f"Agent has accessed {session.sensitive_screens_accessed} "
                    f"sensitive screens this session "
                    f"(threshold: {self.policy.max_sensitive_screens_per_session})",
                    field="sensitive_access",
                )
                self._record_security_event(
                    session_id=session_id,
                    event_type=SecurityEventType.SENSITIVE_SCREEN_ACCESS,
                    severity="WARNING",
                    description=(
                        f"Excessive sensitive screen access: "
                        f"{session.sensitive_screens_accessed} screens"
                    ),
                    agent_id=session.agent_id,
                    screen_name=screen_name,
                )

        # Detect sensitive data patterns on screen
        if self.policy.alert_on_sensitive_data:
            sensitive_matches = self.parser.detect_sensitive_data(screen)
            if sensitive_matches:
                session.data_read_count += len(sensitive_matches)
                for match in sensitive_matches:
                    self._record_security_event(
                        session_id=session_id,
                        event_type=SecurityEventType.SENSITIVE_SCREEN_ACCESS,
                        severity="INFO",
                        description=(
                            f"Sensitive data displayed: {match.pattern_type} "
                            f"at row {match.row}, col {match.column}"
                        ),
                        agent_id=session.agent_id,
                        screen_name=screen_name,
                        field_position=match.position,
                        details={
                            "pattern_type": match.pattern_type,
                            "masked_value": match.masked_value,
                            "confidence": match.confidence,
                        },
                    )

        if self.policy.log_screen_transitions:
            self._audit_log_entry(
                action="SCREEN_ACCESS",
                session_id=session_id,
                agent_id=session.agent_id,
                details={
                    "screen_name": screen_name,
                    "previous_screen": session.previous_screen_name,
                    "is_sensitive": is_sensitive,
                    "field_count": len(screen.fields),
                },
            )

        return result

    def validate_field_modification(
        self,
        session_id: str,
        screen: ScreenBuffer,
        field_position: int,
        new_value: str,
    ) -> ValidationResult:
        """
        Validate a field modification on the terminal screen.

        Ensures protected fields are not being tampered with and
        logs all field modifications for audit purposes.

        Args:
            session_id: Session identifier
            screen: Current screen buffer
            field_position: Buffer position of the field being modified
            new_value: The new value being entered

        Returns:
            ValidationResult indicating whether the modification is allowed
        """
        result = self._create_result()
        session = self._sessions.get(session_id)

        if not session:
            result.add_error(
                "TN3270E_UNKNOWN_SESSION",
                f"No session found for ID {session_id}",
                severity=ValidationSeverity.CRITICAL,
            )
            return result

        session.last_activity = datetime.now()
        session.field_modifications += 1

        # Find the field at the given position
        target_field = screen.find_field_at(field_position)

        if target_field is None:
            result.add_warning(
                "TN3270E_UNKNOWN_FIELD",
                f"No field found at buffer position {field_position}",
                field="field_position",
            )
            return result

        # Check if modifying a protected field
        if target_field.is_protected:
            session.protected_field_violations += 1

            if self.policy.reject_protected_field_modifications:
                result.add_error(
                    "TN3270E_PROTECTED_FIELD_TAMPER",
                    f"Attempt to modify protected field at position {field_position} "
                    f"(row {target_field.row}, col {target_field.column})",
                    field="protected_field",
                    severity=ValidationSeverity.CRITICAL,
                )

            self._record_security_event(
                session_id=session_id,
                event_type=SecurityEventType.PROTECTED_FIELD_TAMPER,
                severity="CRITICAL",
                description=(
                    f"Protected field modification attempt at "
                    f"row {target_field.row}, col {target_field.column}"
                ),
                agent_id=session.agent_id,
                screen_name=session.current_screen_name,
                field_position=field_position,
                details={
                    "field_content": target_field.content[:50],
                    "attempted_value": new_value[:50],
                    "violation_count": session.protected_field_violations,
                },
            )

            # Check if violations exceed threshold
            if session.protected_field_violations >= self.policy.max_protected_field_violations:
                result.add_error(
                    "TN3270E_SESSION_TERMINATED",
                    f"Session terminated due to {session.protected_field_violations} "
                    f"protected field violations",
                    field="session",
                    severity=ValidationSeverity.CRITICAL,
                )
        else:
            # Log normal field modification
            self._audit_log_entry(
                action="FIELD_MODIFIED",
                session_id=session_id,
                agent_id=session.agent_id,
                details={
                    "field_position": field_position,
                    "field_row": target_field.row,
                    "field_column": target_field.column,
                    "is_hidden": target_field.is_hidden,
                    # Do not log actual values for hidden fields (passwords)
                    "value_logged": not target_field.is_hidden,
                },
            )

        return result

    def check_session_timeout(self, session_id: str) -> ValidationResult:
        """
        Check if a session has exceeded its timeout limits.

        Enforces both idle timeout and maximum session duration.

        Args:
            session_id: Session identifier

        Returns:
            ValidationResult indicating timeout status
        """
        result = self._create_result()
        session = self._sessions.get(session_id)

        if not session:
            result.add_error(
                "TN3270E_UNKNOWN_SESSION",
                f"No session found for ID {session_id}",
                severity=ValidationSeverity.CRITICAL,
            )
            return result

        now = datetime.now()

        # Check idle timeout
        idle_seconds = (now - session.last_activity).total_seconds()
        if idle_seconds > self.policy.session_timeout_seconds:
            session.is_timed_out = True
            result.add_error(
                "TN3270E_SESSION_IDLE_TIMEOUT",
                f"Session idle for {int(idle_seconds)} seconds "
                f"(limit: {self.policy.session_timeout_seconds}s)",
                field="session_timeout",
                severity=ValidationSeverity.CRITICAL,
            )
            self._record_security_event(
                session_id=session_id,
                event_type=SecurityEventType.SESSION_ANOMALY,
                severity="WARNING",
                description=f"Session idle timeout after {int(idle_seconds)} seconds",
                agent_id=session.agent_id,
            )

        # Check maximum session duration
        session_duration = (now - session.established_at).total_seconds()
        if session_duration > self.policy.max_session_duration_seconds:
            session.is_timed_out = True
            result.add_error(
                "TN3270E_SESSION_MAX_DURATION",
                f"Session duration {int(session_duration)}s exceeds maximum "
                f"of {self.policy.max_session_duration_seconds}s",
                field="session_duration",
                severity=ValidationSeverity.CRITICAL,
            )
            self._record_security_event(
                session_id=session_id,
                event_type=SecurityEventType.SESSION_ANOMALY,
                severity="WARNING",
                description=(
                    f"Session exceeded max duration: "
                    f"{int(session_duration)} seconds"
                ),
                agent_id=session.agent_id,
            )

        return result

    def terminate_session(self, session_id: str) -> None:
        """
        Clean up session state when a session is terminated.

        Args:
            session_id: Session identifier to terminate
        """
        session = self._sessions.get(session_id)
        if session:
            self._agent_session_count[session.agent_id] = max(
                0, self._agent_session_count[session.agent_id] - 1
            )
            self._audit_log_entry(
                action="SESSION_TERMINATED",
                session_id=session_id,
                agent_id=session.agent_id,
                details={
                    "duration_seconds": (
                        datetime.now() - session.established_at
                    ).total_seconds(),
                    "screens_accessed": len(session.screens_accessed),
                    "sensitive_screens": session.sensitive_screens_accessed,
                    "field_modifications": session.field_modifications,
                    "protected_violations": session.protected_field_violations,
                    "security_events": len(session.security_events),
                },
            )
            del self._sessions[session_id]

    def get_session_security_events(
        self, session_id: str
    ) -> List[SecurityEvent]:
        """
        Get all security events for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of SecurityEvent objects
        """
        session = self._sessions.get(session_id)
        if session:
            return session.security_events
        return []

    def get_audit_log(
        self,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit log entries with optional filtering.

        Args:
            session_id: Filter by session ID
            agent_id: Filter by agent ID
            action: Filter by action type
            since: Only return entries after this time

        Returns:
            List of audit log entry dictionaries
        """
        entries = self._audit_log

        if session_id:
            entries = [e for e in entries if e.get("session_id") == session_id]
        if agent_id:
            entries = [e for e in entries if e.get("agent_id") == agent_id]
        if action:
            entries = [e for e in entries if e.get("action") == action]
        if since:
            entries = [
                e for e in entries
                if datetime.fromisoformat(e["timestamp"]) >= since
            ]

        return entries

    def _validate_message(
        self, msg: TN3270eMessage, result: ValidationResult
    ) -> None:
        """Validate a TN3270eMessage object."""
        if msg.parse_errors:
            for error in msg.parse_errors:
                result.add_error(
                    "TN3270E_PARSE_ERROR",
                    f"Message parse error: {error}",
                    severity=ValidationSeverity.ERROR,
                )

        # Validate header
        if msg.header.sequence_number > 65535:
            result.add_error(
                "TN3270E_INVALID_SEQUENCE",
                f"Sequence number {msg.header.sequence_number} exceeds maximum",
                field="header/sequence_number",
            )

        # Validate data type
        try:
            TN3270eDataType(msg.header.data_type)
        except ValueError:
            result.add_error(
                "TN3270E_INVALID_DATA_TYPE",
                f"Unknown data type: {msg.header.data_type}",
                field="header/data_type",
            )

    def _validate_screen(
        self, screen: ScreenBuffer, result: ValidationResult
    ) -> None:
        """Validate a screen buffer for security concerns."""
        sensitive_matches = self.parser.detect_sensitive_data(screen)
        if sensitive_matches:
            for match in sensitive_matches:
                result.add_warning(
                    "TN3270E_SENSITIVE_DATA_EXPOSED",
                    f"Sensitive data ({match.pattern_type}) detected at "
                    f"row {match.row}, col {match.column}",
                    field=f"screen_{match.row}_{match.column}",
                )

    def _validate_session_state(
        self, session: SessionState, result: ValidationResult
    ) -> None:
        """Validate a session state object."""
        if session.is_timed_out:
            result.add_error(
                "TN3270E_SESSION_TIMED_OUT",
                "Session has been timed out",
                severity=ValidationSeverity.CRITICAL,
            )

        if not session.is_authenticated:
            result.add_warning(
                "TN3270E_UNAUTHENTICATED",
                "Session is not authenticated",
                field="authentication",
            )

        if session.protected_field_violations > 0:
            result.add_warning(
                "TN3270E_FIELD_VIOLATIONS",
                f"Session has {session.protected_field_violations} "
                f"protected field violations",
                field="protected_fields",
            )

    def _validate_dict(
        self, data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate terminal data from a dictionary."""
        if "session_id" not in data:
            result.add_error(
                "TN3270E_MISSING_SESSION_ID",
                "Session ID is required",
                field="session_id",
            )

        if "agent_id" not in data:
            result.add_warning(
                "TN3270E_MISSING_AGENT_ID",
                "Agent ID should be provided for audit purposes",
                field="agent_id",
            )

    def _record_security_event(
        self,
        session_id: str,
        event_type: SecurityEventType,
        severity: str,
        description: str,
        agent_id: str = "",
        source_ip: str = "",
        screen_name: str = "",
        field_position: int = -1,
        details: Optional[Dict[str, Any]] = None,
    ) -> SecurityEvent:
        """Create and record a security event."""
        event = SecurityEvent(
            event_type=event_type,
            session_id=session_id,
            agent_id=agent_id,
            severity=severity,
            description=description,
            screen_name=screen_name,
            field_position=field_position,
            source_ip=source_ip,
            details=details or {},
        )

        session = self._sessions.get(session_id)
        if session:
            session.security_events.append(event)

        self._audit_log_entry(
            action=f"SECURITY_EVENT_{event_type.value.upper()}",
            session_id=session_id,
            agent_id=agent_id,
            details={
                "severity": severity,
                "description": description,
                "screen_name": screen_name,
                **(details or {}),
            },
        )

        return event

    def _audit_log_entry(
        self,
        action: str,
        session_id: str = "",
        agent_id: str = "",
        source_ip: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an entry in the audit log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "session_id": session_id,
            "agent_id": agent_id,
            "source_ip": source_ip,
            "details": details or {},
        }
        self._audit_log.append(entry)
