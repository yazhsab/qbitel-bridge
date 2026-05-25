"""
CTI Security Validator

Validates CTI (Computer Telephony Integration) operations for security:
- Agent state transition validation (detect impossible state changes)
- Call routing integrity (prevent unauthorized queue jumping)
- Agent login pattern monitoring (detect shared credential usage)
- Transfer and conference request validation (prevent toll fraud)
- Agent permission level enforcement
- Abnormal call pattern detection (unusual hold times, rapid transfers)
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
from ai_engine.domains.bpo.protocols.cti.cti_message import (
    AgentPermissions,
    AgentState,
    CallRoute,
    CTICallInfo,
    CTICallState,
    CTIMessage,
    CTIMessageType,
    VALID_AGENT_TRANSITIONS,
    VALID_CALL_TRANSITIONS,
)


@dataclass
class AgentSessionInfo:
    """Tracks an agent's session state for security monitoring."""

    agent_id: str
    current_state: AgentState = AgentState.LOGGED_OUT
    login_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)
    source_ip: str = ""
    workstation_id: str = ""
    permissions: AgentPermissions = field(default_factory=AgentPermissions)

    # Login tracking
    login_history: List[Dict[str, Any]] = field(default_factory=list)
    concurrent_login_locations: Set[str] = field(default_factory=set)

    # Call tracking
    active_calls: Dict[str, CTICallInfo] = field(default_factory=dict)
    calls_handled_today: int = 0
    total_talk_seconds: float = 0.0
    total_hold_seconds: float = 0.0
    total_wrap_seconds: float = 0.0

    # Transfer tracking
    transfers_today: int = 0
    external_transfers_today: int = 0

    # Pattern tracking
    state_changes: List[Tuple[AgentState, AgentState, datetime]] = field(
        default_factory=list
    )
    hold_events: List[Tuple[str, float]] = field(default_factory=list)  # (call_id, duration)
    short_calls: int = 0          # Calls under 10 seconds
    rapid_transfers: int = 0       # Transfers within 30 seconds of answer

    # Security
    invalid_transitions: int = 0
    security_violations: int = 0


@dataclass
class CTISecurityPolicy:
    """Configuration for CTI security policies."""

    # Agent state validation
    enforce_state_machine: bool = True
    max_invalid_transitions: int = 5        # Before flagging

    # Login security
    detect_shared_credentials: bool = True
    max_concurrent_logins: int = 1          # Per agent ID
    login_cooldown_seconds: int = 60        # Minimum time between logins
    max_logins_per_hour: int = 5

    # Call pattern monitoring
    max_hold_duration_seconds: int = 300    # 5-minute hold limit
    max_wrap_up_seconds: int = 120          # 2-minute wrap-up limit
    short_call_threshold_seconds: int = 10  # Calls shorter than this are flagged
    max_short_calls_per_hour: int = 20
    rapid_transfer_threshold_seconds: int = 30

    # Transfer and conference validation
    max_transfers_per_call: int = 5
    max_external_transfers_per_shift: int = 20
    validate_transfer_destinations: bool = True
    block_premium_rate_transfers: bool = True

    # Queue integrity
    enforce_queue_authorization: bool = True
    enforce_skill_group_authorization: bool = True
    max_queue_priority_override: int = 3    # Max allowed priority bump

    # Fraud number patterns (toll fraud prevention)
    blocked_transfer_patterns: List[str] = field(default_factory=lambda: [
        r"^900",        # US premium rate
        r"^976",        # US premium rate
        r"^882\d",      # IRSF satellite
        r"^883\d",      # IRSF satellite
    ])

    # Logging
    log_state_changes: bool = True
    log_call_events: bool = True
    log_transfers: bool = True
    log_security_violations: bool = True


class CTIValidator(BaseValidator):
    """
    Security validator for CTI operations.

    Monitors CTI events in real-time to enforce:
    - Agent state machine integrity
    - Call routing authorization
    - Shared credential detection
    - Toll fraud prevention on transfers
    - Abnormal call pattern detection
    - Permission-level enforcement
    """

    def __init__(
        self,
        policy: Optional[CTISecurityPolicy] = None,
        strict: bool = True,
    ):
        """
        Initialize the CTI validator.

        Args:
            policy: Security policy configuration
            strict: If True, treat warnings as errors
        """
        super().__init__(strict)
        self.policy = policy or CTISecurityPolicy()
        self._agents: Dict[str, AgentSessionInfo] = {}
        self._agent_login_ips: Dict[str, Set[str]] = defaultdict(set)
        self._agent_login_times: Dict[str, List[datetime]] = defaultdict(list)
        self._audit_log: List[Dict[str, Any]] = []
        self._blocked_patterns = [
            re.compile(p) for p in self.policy.blocked_transfer_patterns
        ]

    @property
    def name(self) -> str:
        return "CTIValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate CTI data.

        Args:
            data: Can be CTIMessage, CTICallInfo, AgentSessionInfo, or dict

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, CTIMessage):
            self._validate_message(data, result)
        elif isinstance(data, CTICallInfo):
            self._validate_call_info(data, result)
        elif isinstance(data, AgentSessionInfo):
            self._validate_agent_session(data, result)
        elif isinstance(data, dict):
            self._validate_dict(data, result)
        else:
            result.add_error(
                "CTI_INVALID_INPUT",
                "Input must be a CTIMessage, CTICallInfo, AgentSessionInfo, or dict",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def validate_agent_state_transition(
        self,
        agent_id: str,
        new_state: AgentState,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate an agent state transition.

        Checks the transition against the defined state machine to
        detect impossible or suspicious state changes.

        Args:
            agent_id: Agent identifier
            new_state: The target agent state
            context: Optional context data (call_id, reason, etc.)

        Returns:
            ValidationResult indicating whether the transition is valid
        """
        result = self._create_result()
        context = context or {}
        agent = self._get_or_create_agent(agent_id)
        current_state = agent.current_state
        now = datetime.now()

        if not self.policy.enforce_state_machine:
            agent.current_state = new_state
            agent.last_state_change = now
            return result

        # Check if transition is valid
        valid_targets = VALID_AGENT_TRANSITIONS.get(current_state, set())

        if new_state not in valid_targets:
            agent.invalid_transitions += 1
            result.add_error(
                "CTI_INVALID_STATE_TRANSITION",
                f"Invalid agent state transition: {current_state._state} -> "
                f"{new_state._state}. Valid transitions from "
                f"{current_state._state}: "
                f"{[s._state for s in valid_targets]}",
                field="agent_state",
                severity=ValidationSeverity.CRITICAL,
            )

            self._audit_log_entry(
                action="INVALID_STATE_TRANSITION",
                agent_id=agent_id,
                details={
                    "from_state": current_state._state,
                    "to_state": new_state._state,
                    "valid_targets": [s._state for s in valid_targets],
                    "violation_count": agent.invalid_transitions,
                    **context,
                },
            )

            if agent.invalid_transitions >= self.policy.max_invalid_transitions:
                result.add_error(
                    "CTI_AGENT_STATE_ANOMALY",
                    f"Agent {agent_id} has {agent.invalid_transitions} "
                    f"invalid state transitions. Session may be compromised.",
                    field="agent_state",
                    severity=ValidationSeverity.CRITICAL,
                )
                agent.security_violations += 1

            return result

        # Record valid transition
        agent.state_changes.append((current_state, new_state, now))
        agent.current_state = new_state
        agent.last_state_change = now

        if self.policy.log_state_changes:
            self._audit_log_entry(
                action="AGENT_STATE_CHANGE",
                agent_id=agent_id,
                details={
                    "from_state": current_state._state,
                    "to_state": new_state._state,
                    **context,
                },
            )

        return result

    def validate_call_state_transition(
        self,
        call_id: str,
        agent_id: str,
        new_state: CTICallState,
    ) -> ValidationResult:
        """
        Validate a call state transition.

        Checks the transition against the call state machine.

        Args:
            call_id: Call identifier
            agent_id: Agent handling the call
            new_state: The target call state

        Returns:
            ValidationResult indicating whether the transition is valid
        """
        result = self._create_result()
        agent = self._get_or_create_agent(agent_id)
        now = datetime.now()

        call_info = agent.active_calls.get(call_id)
        if not call_info:
            # New call
            call_info = CTICallInfo(call_id=call_id, agent_id=agent_id)
            agent.active_calls[call_id] = call_info

        current_state = call_info.state
        valid_targets = VALID_CALL_TRANSITIONS.get(current_state, set())

        if new_state not in valid_targets:
            result.add_error(
                "CTI_INVALID_CALL_TRANSITION",
                f"Invalid call state transition: {current_state._state} -> "
                f"{new_state._state} for call {call_id}",
                field="call_state",
                severity=ValidationSeverity.CRITICAL,
            )
            return result

        # Update call state and timing
        if new_state == CTICallState.RINGING:
            call_info.offered_at = now
        elif new_state == CTICallState.CONNECTED:
            if current_state == CTICallState.RINGING:
                call_info.answered_at = now
            elif current_state == CTICallState.HELD and call_info.held_at:
                # Calculate hold duration
                hold_duration = (now - call_info.held_at).total_seconds()
                call_info.total_hold_duration_seconds += hold_duration
                agent.hold_events.append((call_id, hold_duration))
                call_info.held_at = None
        elif new_state == CTICallState.HELD:
            call_info.held_at = now
            call_info.hold_count += 1
        elif new_state == CTICallState.WRAPPING_UP:
            call_info.wrap_up_started_at = now
        elif new_state == CTICallState.DISCONNECTED:
            call_info.ended_at = now
            self._analyze_completed_call(agent, call_info, result)

        call_info.state = new_state

        if self.policy.log_call_events:
            self._audit_log_entry(
                action="CALL_STATE_CHANGE",
                agent_id=agent_id,
                call_id=call_id,
                details={
                    "from_state": current_state._state,
                    "to_state": new_state._state,
                },
            )

        return result

    def validate_agent_login(
        self,
        agent_id: str,
        source_ip: str,
        workstation_id: str = "",
    ) -> ValidationResult:
        """
        Validate an agent login attempt.

        Detects shared credential usage by monitoring login patterns
        across multiple locations and rapid re-logins.

        Args:
            agent_id: Agent identifier
            source_ip: Source IP address of the login
            workstation_id: Workstation identifier

        Returns:
            ValidationResult with any security concerns
        """
        result = self._create_result()
        now = datetime.now()

        # Track login IP addresses
        self._agent_login_ips[agent_id].add(source_ip)
        self._agent_login_times[agent_id].append(now)

        # Check for shared credential usage (multiple IPs)
        if self.policy.detect_shared_credentials:
            # Check recent unique IPs (last hour)
            recent_ips = set()
            recent_cutoff = now - timedelta(hours=1)
            agent = self._agents.get(agent_id)
            if agent:
                recent_logins = [
                    entry for entry in agent.login_history
                    if datetime.fromisoformat(entry["timestamp"]) >= recent_cutoff
                ]
                recent_ips = {entry["source_ip"] for entry in recent_logins}
            recent_ips.add(source_ip)

            if len(recent_ips) > self.policy.max_concurrent_logins:
                result.add_error(
                    "CTI_SHARED_CREDENTIALS",
                    f"Agent {agent_id} has logged in from {len(recent_ips)} "
                    f"different IPs in the last hour: {recent_ips}. "
                    f"Possible shared credential usage.",
                    field="agent_login",
                    severity=ValidationSeverity.CRITICAL,
                )

                self._audit_log_entry(
                    action="SHARED_CREDENTIAL_DETECTED",
                    agent_id=agent_id,
                    details={
                        "ip_addresses": list(recent_ips),
                        "current_ip": source_ip,
                    },
                )

        # Check login rate
        recent_logins = [
            t for t in self._agent_login_times[agent_id]
            if (now - t).total_seconds() < 3600
        ]
        if len(recent_logins) > self.policy.max_logins_per_hour:
            result.add_warning(
                "CTI_EXCESSIVE_LOGINS",
                f"Agent {agent_id} has logged in {len(recent_logins)} times "
                f"in the last hour (limit: {self.policy.max_logins_per_hour})",
                field="agent_login",
            )

        # Check login cooldown
        agent = self._get_or_create_agent(agent_id)
        if agent.login_time:
            time_since_last = (now - agent.login_time).total_seconds()
            if time_since_last < self.policy.login_cooldown_seconds:
                result.add_warning(
                    "CTI_RAPID_RELOGIN",
                    f"Agent {agent_id} logged in again after only "
                    f"{int(time_since_last)} seconds "
                    f"(cooldown: {self.policy.login_cooldown_seconds}s)",
                    field="agent_login",
                )

        # Record login
        agent.login_time = now
        agent.source_ip = source_ip
        agent.workstation_id = workstation_id
        agent.concurrent_login_locations.add(source_ip)
        agent.login_history.append({
            "timestamp": now.isoformat(),
            "source_ip": source_ip,
            "workstation_id": workstation_id,
        })

        self._audit_log_entry(
            action="AGENT_LOGIN",
            agent_id=agent_id,
            details={
                "source_ip": source_ip,
                "workstation_id": workstation_id,
            },
        )

        return result

    def validate_call_routing(
        self,
        call_id: str,
        route: CallRoute,
        agent_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate call routing integrity.

        Prevents unauthorized queue jumping and ensures routing
        adheres to skill group and priority rules.

        Args:
            call_id: Call identifier
            route: Proposed call route
            agent_id: Agent to route to (if direct routing)

        Returns:
            ValidationResult indicating whether the routing is authorized
        """
        result = self._create_result()

        # Check queue availability
        if not route.is_available:
            result.add_error(
                "CTI_QUEUE_FULL",
                f"Queue '{route.queue_name}' has reached maximum depth "
                f"({route.queue_depth}/{route.max_queue_depth})",
                field="queue_depth",
            )

        # Check agent queue authorization
        if agent_id and self.policy.enforce_queue_authorization:
            agent = self._agents.get(agent_id)
            if agent and not agent.permissions.has_queue_access(route.queue_name):
                result.add_error(
                    "CTI_UNAUTHORIZED_QUEUE",
                    f"Agent {agent_id} is not authorized for queue "
                    f"'{route.queue_name}'",
                    field="queue_name",
                    severity=ValidationSeverity.CRITICAL,
                )

                self._audit_log_entry(
                    action="UNAUTHORIZED_QUEUE_ACCESS",
                    agent_id=agent_id,
                    call_id=call_id,
                    details={
                        "queue_name": route.queue_name,
                        "skill_group": route.skill_group,
                    },
                )

        # Check skill group authorization
        if agent_id and self.policy.enforce_skill_group_authorization:
            agent = self._agents.get(agent_id)
            if agent and route.skill_group:
                if not agent.permissions.has_skill_group_access(route.skill_group):
                    result.add_error(
                        "CTI_UNAUTHORIZED_SKILL_GROUP",
                        f"Agent {agent_id} is not authorized for skill group "
                        f"'{route.skill_group}'",
                        field="skill_group",
                        severity=ValidationSeverity.CRITICAL,
                    )

        # Validate priority (detect queue jumping)
        if route.priority < self.policy.max_queue_priority_override:
            result.add_warning(
                "CTI_HIGH_PRIORITY_ROUTE",
                f"Call routed with high priority {route.priority} "
                f"(normal minimum: {self.policy.max_queue_priority_override}). "
                f"Verify this is authorized.",
                field="priority",
            )

        self._audit_log_entry(
            action="CALL_ROUTED",
            call_id=call_id,
            agent_id=agent_id or "",
            details={
                "queue_name": route.queue_name,
                "skill_group": route.skill_group,
                "priority": route.priority,
                "estimated_wait": route.estimated_wait,
                "queue_depth": route.queue_depth,
            },
        )

        return result

    def validate_transfer_request(
        self,
        call_id: str,
        agent_id: str,
        destination: str,
        is_external: bool = False,
    ) -> ValidationResult:
        """
        Validate a call transfer or conference request.

        Checks agent permissions and prevents toll fraud by
        validating transfer destinations.

        Args:
            call_id: Call identifier
            agent_id: Agent requesting the transfer
            destination: Transfer destination (agent ID, queue, or phone number)
            is_external: Whether this is an external transfer

        Returns:
            ValidationResult indicating whether the transfer is allowed
        """
        result = self._create_result()
        agent = self._get_or_create_agent(agent_id)
        now = datetime.now()

        # Check permission for transfer type
        if is_external:
            if not agent.permissions.can_transfer_external:
                result.add_error(
                    "CTI_TRANSFER_NOT_PERMITTED",
                    f"Agent {agent_id} does not have permission for "
                    f"external transfers",
                    field="transfer",
                    severity=ValidationSeverity.CRITICAL,
                )
                agent.security_violations += 1

                self._audit_log_entry(
                    action="UNAUTHORIZED_TRANSFER",
                    agent_id=agent_id,
                    call_id=call_id,
                    details={
                        "destination": self._mask_destination(destination),
                        "type": "external",
                    },
                )
                return result

            # Check external transfer limits
            agent.external_transfers_today += 1
            if agent.external_transfers_today > self.policy.max_external_transfers_per_shift:
                result.add_error(
                    "CTI_EXCESSIVE_EXTERNAL_TRANSFERS",
                    f"Agent {agent_id} has exceeded the daily external "
                    f"transfer limit ({agent.external_transfers_today}/"
                    f"{self.policy.max_external_transfers_per_shift})",
                    field="transfer_count",
                    severity=ValidationSeverity.CRITICAL,
                )

            # Validate destination number for toll fraud
            if self.policy.block_premium_rate_transfers:
                clean_number = re.sub(r"[\s\-.()+]", "", destination)
                for pattern in self._blocked_patterns:
                    if pattern.match(clean_number):
                        result.add_error(
                            "CTI_TOLL_FRAUD_BLOCKED",
                            f"Transfer to potentially fraudulent number blocked: "
                            f"{self._mask_destination(destination)}",
                            field="destination",
                            severity=ValidationSeverity.CRITICAL,
                        )
                        agent.security_violations += 1

                        self._audit_log_entry(
                            action="TOLL_FRAUD_BLOCKED",
                            agent_id=agent_id,
                            call_id=call_id,
                            details={
                                "destination": self._mask_destination(destination),
                                "pattern_matched": True,
                            },
                        )
                        return result
        else:
            if not agent.permissions.can_transfer_internal:
                result.add_error(
                    "CTI_TRANSFER_NOT_PERMITTED",
                    f"Agent {agent_id} does not have permission for "
                    f"internal transfers",
                    field="transfer",
                    severity=ValidationSeverity.CRITICAL,
                )
                return result

        # Check call transfer count
        call_info = agent.active_calls.get(call_id)
        if call_info:
            call_info.transfer_count += 1
            call_info.transfer_destinations.append(destination)

            if call_info.transfer_count > self.policy.max_transfers_per_call:
                result.add_warning(
                    "CTI_EXCESSIVE_TRANSFERS",
                    f"Call {call_id} has been transferred "
                    f"{call_info.transfer_count} times "
                    f"(limit: {self.policy.max_transfers_per_call})",
                    field="transfer_count",
                )

            # Check for rapid transfer (answered and immediately transferred)
            if call_info.answered_at:
                talk_time = (now - call_info.answered_at).total_seconds()
                if talk_time < self.policy.rapid_transfer_threshold_seconds:
                    agent.rapid_transfers += 1
                    result.add_warning(
                        "CTI_RAPID_TRANSFER",
                        f"Call transferred after only {int(talk_time)} seconds "
                        f"(threshold: {self.policy.rapid_transfer_threshold_seconds}s)",
                        field="transfer_timing",
                    )

        agent.transfers_today += 1

        if self.policy.log_transfers:
            self._audit_log_entry(
                action="CALL_TRANSFER",
                agent_id=agent_id,
                call_id=call_id,
                details={
                    "destination": self._mask_destination(destination)
                    if is_external
                    else destination,
                    "is_external": is_external,
                    "transfers_today": agent.transfers_today,
                },
            )

        return result

    def validate_agent_permissions(
        self,
        agent_id: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Enforce agent permission levels for a specific operation.

        Args:
            agent_id: Agent identifier
            operation: The operation being attempted
            context: Additional context for the permission check

        Returns:
            ValidationResult indicating whether the operation is permitted
        """
        result = self._create_result()
        context = context or {}
        agent = self._get_or_create_agent(agent_id)
        permissions = agent.permissions

        operation_checks = {
            "transfer_internal": permissions.can_transfer_internal,
            "transfer_external": permissions.can_transfer_external,
            "conference": permissions.can_conference,
            "barge_in": permissions.can_barge_in,
            "silent_monitor": permissions.can_silent_monitor,
            "force_logout": permissions.can_force_logout,
            "change_queue": permissions.can_change_queue,
            "access_recording": permissions.can_access_recording,
            "modify_call_data": permissions.can_modify_call_data,
        }

        if operation in operation_checks:
            if not operation_checks[operation]:
                result.add_error(
                    "CTI_PERMISSION_DENIED",
                    f"Agent {agent_id} (role: {permissions.role}) does not "
                    f"have permission for operation: {operation}",
                    field="permissions",
                    severity=ValidationSeverity.CRITICAL,
                )

                self._audit_log_entry(
                    action="PERMISSION_DENIED",
                    agent_id=agent_id,
                    details={
                        "operation": operation,
                        "role": permissions.role,
                        **context,
                    },
                )
        else:
            result.add_warning(
                "CTI_UNKNOWN_OPERATION",
                f"Unknown operation for permission check: {operation}",
                field="operation",
            )

        return result

    def detect_abnormal_patterns(
        self,
        agent_id: str,
    ) -> ValidationResult:
        """
        Detect abnormal call patterns for an agent.

        Analyzes hold times, transfer rates, short calls, and
        other patterns that may indicate fraud or misuse.

        Args:
            agent_id: Agent identifier

        Returns:
            ValidationResult with any detected anomalies
        """
        result = self._create_result()
        agent = self._agents.get(agent_id)

        if not agent:
            return result

        # Check for excessive short calls
        if agent.short_calls > self.policy.max_short_calls_per_hour:
            result.add_warning(
                "CTI_EXCESSIVE_SHORT_CALLS",
                f"Agent {agent_id} has {agent.short_calls} short calls "
                f"(< {self.policy.short_call_threshold_seconds}s) in the "
                f"current period (limit: {self.policy.max_short_calls_per_hour})",
                field="call_pattern",
            )

        # Check for excessive rapid transfers
        if agent.rapid_transfers > 5:
            result.add_warning(
                "CTI_EXCESSIVE_RAPID_TRANSFERS",
                f"Agent {agent_id} has {agent.rapid_transfers} rapid transfers "
                f"(within {self.policy.rapid_transfer_threshold_seconds}s "
                f"of answering)",
                field="transfer_pattern",
            )

        # Check for unusual hold patterns
        if agent.hold_events:
            avg_hold = sum(d for _, d in agent.hold_events) / len(agent.hold_events)
            long_holds = [
                d for _, d in agent.hold_events
                if d > self.policy.max_hold_duration_seconds
            ]
            if long_holds:
                result.add_warning(
                    "CTI_EXCESSIVE_HOLD_TIME",
                    f"Agent {agent_id} has {len(long_holds)} holds exceeding "
                    f"{self.policy.max_hold_duration_seconds}s "
                    f"(avg hold: {avg_hold:.0f}s)",
                    field="hold_pattern",
                )

        # Check for excessive external transfers (toll fraud indicator)
        if agent.external_transfers_today > self.policy.max_external_transfers_per_shift:
            result.add_error(
                "CTI_TOLL_FRAUD_RISK",
                f"Agent {agent_id} has {agent.external_transfers_today} external "
                f"transfers today "
                f"(limit: {self.policy.max_external_transfers_per_shift}). "
                f"Possible toll fraud.",
                field="transfer_pattern",
                severity=ValidationSeverity.CRITICAL,
            )

        # Check state change frequency
        if len(agent.state_changes) > 0:
            recent_changes = [
                t for _, _, t in agent.state_changes
                if (datetime.now() - t).total_seconds() < 300
            ]
            if len(recent_changes) > 20:
                result.add_warning(
                    "CTI_RAPID_STATE_CHANGES",
                    f"Agent {agent_id} has {len(recent_changes)} state changes "
                    f"in the last 5 minutes. May indicate automation or glitch.",
                    field="state_pattern",
                )

        return result

    def get_audit_log(
        self,
        agent_id: Optional[str] = None,
        call_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit log entries with optional filtering.

        Args:
            agent_id: Filter by agent ID
            call_id: Filter by call ID
            action: Filter by action type
            since: Only return entries after this time

        Returns:
            List of audit log entry dictionaries
        """
        entries = self._audit_log

        if agent_id:
            entries = [e for e in entries if e.get("agent_id") == agent_id]
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

    def set_agent_permissions(
        self,
        agent_id: str,
        permissions: AgentPermissions,
    ) -> None:
        """
        Set agent permissions for authorization checks.

        Args:
            agent_id: Agent identifier
            permissions: Permission configuration to apply
        """
        agent = self._get_or_create_agent(agent_id)
        agent.permissions = permissions
        agent.permissions.agent_id = agent_id

    def _analyze_completed_call(
        self,
        agent: AgentSessionInfo,
        call: CTICallInfo,
        result: ValidationResult,
    ) -> None:
        """Analyze a completed call for abnormal patterns."""
        agent.calls_handled_today += 1
        agent.total_talk_seconds += call.talk_duration_seconds
        agent.total_hold_seconds += call.total_hold_duration_seconds

        # Check for short call
        if call.talk_duration_seconds < self.policy.short_call_threshold_seconds:
            agent.short_calls += 1
            if agent.short_calls > self.policy.max_short_calls_per_hour:
                result.add_warning(
                    "CTI_SHORT_CALL_PATTERN",
                    f"Call {call.call_id} lasted only "
                    f"{call.talk_duration_seconds:.0f}s. Agent has "
                    f"{agent.short_calls} short calls this period.",
                    field="call_duration",
                )

        # Check for excessive hold time on this call
        if call.total_hold_duration_seconds > self.policy.max_hold_duration_seconds:
            result.add_warning(
                "CTI_LONG_HOLD",
                f"Call {call.call_id} had {call.total_hold_duration_seconds:.0f}s "
                f"total hold time "
                f"(limit: {self.policy.max_hold_duration_seconds}s)",
                field="hold_duration",
            )

    def _validate_message(
        self, msg: CTIMessage, result: ValidationResult
    ) -> None:
        """Validate a CTIMessage object."""
        if not msg.call_id and msg.message_type in (
            CTIMessageType.CALL_OFFERED,
            CTIMessageType.CALL_ANSWERED,
            CTIMessageType.CALL_HELD,
            CTIMessageType.CALL_RETRIEVED,
            CTIMessageType.CALL_TRANSFERRED,
            CTIMessageType.CALL_CONFERENCED,
            CTIMessageType.CALL_ENDED,
        ):
            result.add_error(
                "CTI_MISSING_CALL_ID",
                "Call ID is required for call events",
                field="call_id",
            )

        if not msg.agent_id:
            result.add_warning(
                "CTI_MISSING_AGENT_ID",
                "Agent ID should be provided for CTI events",
                field="agent_id",
            )

    def _validate_call_info(
        self, call: CTICallInfo, result: ValidationResult
    ) -> None:
        """Validate a CTICallInfo object."""
        if not call.call_id:
            result.add_error(
                "CTI_MISSING_CALL_ID",
                "Call ID is required",
                field="call_id",
            )

        if call.hold_count > 10:
            result.add_warning(
                "CTI_EXCESSIVE_HOLDS",
                f"Call has been placed on hold {call.hold_count} times",
                field="hold_count",
            )

        if call.transfer_count > self.policy.max_transfers_per_call:
            result.add_warning(
                "CTI_EXCESSIVE_TRANSFERS",
                f"Call has been transferred {call.transfer_count} times",
                field="transfer_count",
            )

    def _validate_agent_session(
        self, agent: AgentSessionInfo, result: ValidationResult
    ) -> None:
        """Validate an AgentSessionInfo object."""
        if agent.invalid_transitions > 0:
            result.add_warning(
                "CTI_STATE_VIOLATIONS",
                f"Agent has {agent.invalid_transitions} invalid state transitions",
                field="state_transitions",
            )

        if agent.security_violations > 0:
            result.add_error(
                "CTI_SECURITY_VIOLATIONS",
                f"Agent has {agent.security_violations} security violations",
                field="security",
                severity=ValidationSeverity.CRITICAL,
            )

    def _validate_dict(
        self, data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate CTI data from a dictionary."""
        if "message_type" in data:
            valid_types = {t.msg_type for t in CTIMessageType}
            if data["message_type"] not in valid_types:
                result.add_error(
                    "CTI_INVALID_MESSAGE_TYPE",
                    f"Unknown message type: {data['message_type']}",
                    field="message_type",
                )

        if "agent_state" in data:
            valid_states = {s._state for s in AgentState}
            if data["agent_state"] not in valid_states:
                result.add_error(
                    "CTI_INVALID_AGENT_STATE",
                    f"Unknown agent state: {data['agent_state']}",
                    field="agent_state",
                )

    def _get_or_create_agent(self, agent_id: str) -> AgentSessionInfo:
        """Get an existing agent session or create a new one."""
        if agent_id not in self._agents:
            self._agents[agent_id] = AgentSessionInfo(
                agent_id=agent_id,
                permissions=AgentPermissions(agent_id=agent_id),
            )
        return self._agents[agent_id]

    def _mask_destination(self, destination: str) -> str:
        """Mask a phone number for logging, keeping last 4 digits."""
        clean = re.sub(r"[^\d+]", "", destination)
        if len(clean) > 4:
            return "*" * (len(clean) - 4) + clean[-4:]
        return clean

    def _audit_log_entry(
        self,
        action: str,
        agent_id: str = "",
        call_id: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an entry in the audit log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "agent_id": agent_id,
            "call_id": call_id,
            "details": details or {},
        }
        self._audit_log.append(entry)
