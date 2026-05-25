"""
CTI (Computer Telephony Integration) Data Structures

Data structures for CTI bridge protocol handling compatible with
CSTA and TAPI standards:
- CTI message envelope for telephony events
- Call state machine with defined transitions
- Agent state machine with defined transitions
- Call routing definitions with skill-based queuing
- Message type classification for CTI events
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class CTIMessageType(Enum):
    """Types of CTI protocol messages."""

    # Call events
    CALL_OFFERED = ("CALL_OFFERED", "Incoming call offered to agent")
    CALL_ANSWERED = ("CALL_ANSWERED", "Agent answered the call")
    CALL_HELD = ("CALL_HELD", "Call placed on hold")
    CALL_RETRIEVED = ("CALL_RETRIEVED", "Call retrieved from hold")
    CALL_TRANSFERRED = ("CALL_TRANSFERRED", "Call transferred to another party")
    CALL_CONFERENCED = ("CALL_CONFERENCED", "Call joined into conference")
    CALL_ENDED = ("CALL_ENDED", "Call disconnected")

    # Agent events
    AGENT_LOGIN = ("AGENT_LOGIN", "Agent logged into the CTI system")
    AGENT_LOGOUT = ("AGENT_LOGOUT", "Agent logged out of the CTI system")
    AGENT_READY = ("AGENT_READY", "Agent set to ready state")
    AGENT_NOT_READY = ("AGENT_NOT_READY", "Agent set to not-ready state")
    AGENT_WRAP_UP = ("AGENT_WRAP_UP", "Agent entered wrap-up/after-call work")

    def __init__(self, msg_type: str, description: str):
        self.msg_type = msg_type
        self.description = description


class CTICallState(Enum):
    """
    Call state machine states.

    Valid transitions:
        IDLE -> RINGING -> CONNECTED -> HELD -> CONNECTED
        CONNECTED -> TRANSFERRING -> DISCONNECTED
        CONNECTED -> CONFERENCING -> CONNECTED
        CONNECTED -> WRAPPING_UP -> DISCONNECTED
        * -> DISCONNECTED
    """

    IDLE = ("IDLE", "No active call")
    RINGING = ("RINGING", "Call is ringing/alerting")
    CONNECTED = ("CONNECTED", "Call is connected and active")
    HELD = ("HELD", "Call is on hold")
    TRANSFERRING = ("TRANSFERRING", "Call is being transferred")
    CONFERENCING = ("CONFERENCING", "Call is being conferenced")
    WRAPPING_UP = ("WRAPPING_UP", "Post-call wrap-up in progress")
    DISCONNECTED = ("DISCONNECTED", "Call has been disconnected")

    def __init__(self, state: str, description: str):
        self._state = state
        self.description = description


class AgentState(Enum):
    """
    Agent state machine states.

    Valid transitions:
        LOGGED_OUT -> READY (via login)
        LOGGED_OUT -> NOT_READY (via login to not-ready)
        READY -> ON_CALL (call offered and answered)
        READY -> NOT_READY (agent goes unavailable)
        ON_CALL -> WRAPPING (call ends, wrap-up starts)
        ON_CALL -> READY (call ends, no wrap-up required)
        WRAPPING -> READY (wrap-up completed)
        WRAPPING -> NOT_READY (agent goes unavailable after wrap)
        NOT_READY -> READY (agent returns to available)
        NOT_READY -> BREAK (agent goes on break)
        BREAK -> NOT_READY (agent returns from break)
        BREAK -> READY (agent returns from break to ready)
        * -> LOGGED_OUT (via logout)
    """

    LOGGED_OUT = ("LOGGED_OUT", "Agent is not logged in")
    READY = ("READY", "Agent is available for calls")
    NOT_READY = ("NOT_READY", "Agent is logged in but unavailable")
    ON_CALL = ("ON_CALL", "Agent is handling a call")
    WRAPPING = ("WRAPPING", "Agent is in after-call work")
    BREAK = ("BREAK", "Agent is on scheduled break")

    def __init__(self, state: str, description: str):
        self._state = state
        self.description = description


# Define valid state transitions
VALID_CALL_TRANSITIONS: Dict[CTICallState, Set[CTICallState]] = {
    CTICallState.IDLE: {CTICallState.RINGING},
    CTICallState.RINGING: {CTICallState.CONNECTED, CTICallState.DISCONNECTED},
    CTICallState.CONNECTED: {
        CTICallState.HELD,
        CTICallState.TRANSFERRING,
        CTICallState.CONFERENCING,
        CTICallState.WRAPPING_UP,
        CTICallState.DISCONNECTED,
    },
    CTICallState.HELD: {CTICallState.CONNECTED, CTICallState.DISCONNECTED},
    CTICallState.TRANSFERRING: {
        CTICallState.CONNECTED,  # Transfer cancelled
        CTICallState.DISCONNECTED,
    },
    CTICallState.CONFERENCING: {
        CTICallState.CONNECTED,
        CTICallState.DISCONNECTED,
    },
    CTICallState.WRAPPING_UP: {CTICallState.DISCONNECTED, CTICallState.IDLE},
    CTICallState.DISCONNECTED: {CTICallState.IDLE},
}

VALID_AGENT_TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
    AgentState.LOGGED_OUT: {AgentState.READY, AgentState.NOT_READY},
    AgentState.READY: {
        AgentState.ON_CALL,
        AgentState.NOT_READY,
        AgentState.LOGGED_OUT,
    },
    AgentState.NOT_READY: {
        AgentState.READY,
        AgentState.BREAK,
        AgentState.LOGGED_OUT,
    },
    AgentState.ON_CALL: {
        AgentState.WRAPPING,
        AgentState.READY,
        AgentState.NOT_READY,
        AgentState.LOGGED_OUT,
    },
    AgentState.WRAPPING: {
        AgentState.READY,
        AgentState.NOT_READY,
        AgentState.LOGGED_OUT,
    },
    AgentState.BREAK: {
        AgentState.READY,
        AgentState.NOT_READY,
        AgentState.LOGGED_OUT,
    },
}


@dataclass
class CallRoute:
    """
    Defines a call routing configuration.

    Used for skill-based routing to direct calls to the appropriate
    agent queue based on skill requirements and priority.
    """

    route_id: str = ""                  # Unique route identifier
    queue_name: str = ""                # Target queue name
    skill_group: str = ""               # Required agent skill group
    priority: int = 5                   # Routing priority (1=highest, 10=lowest)
    estimated_wait: int = 0             # Estimated wait time in seconds
    agents_available: int = 0           # Number of available agents in queue
    agents_on_call: int = 0             # Number of agents currently on calls
    queue_depth: int = 0                # Number of calls waiting in queue
    service_level_target: int = 20      # Target answer time in seconds
    overflow_route_id: str = ""         # Route for overflow if queue is full
    max_queue_depth: int = 100          # Maximum calls in queue before overflow

    @property
    def is_available(self) -> bool:
        """Check if the queue has capacity to accept calls."""
        return self.queue_depth < self.max_queue_depth

    @property
    def utilization(self) -> float:
        """Calculate queue utilization as a percentage."""
        total_agents = self.agents_available + self.agents_on_call
        if total_agents == 0:
            return 100.0
        return (self.agents_on_call / total_agents) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "route_id": self.route_id,
            "queue_name": self.queue_name,
            "skill_group": self.skill_group,
            "priority": self.priority,
            "estimated_wait": self.estimated_wait,
            "agents_available": self.agents_available,
            "agents_on_call": self.agents_on_call,
            "queue_depth": self.queue_depth,
            "service_level_target": self.service_level_target,
            "is_available": self.is_available,
            "utilization": round(self.utilization, 1),
        }

    def __str__(self) -> str:
        return (
            f"CallRoute({self.route_id}, queue='{self.queue_name}', "
            f"skill='{self.skill_group}', pri={self.priority}, "
            f"wait={self.estimated_wait}s)"
        )


@dataclass
class AgentPermissions:
    """
    Permission levels for agent operations.

    Controls what CTI operations an agent is authorized to perform.
    """

    agent_id: str = ""
    role: str = "agent"                           # agent, supervisor, admin
    can_transfer_internal: bool = True
    can_transfer_external: bool = False
    can_conference: bool = True
    can_barge_in: bool = False                    # Supervisor can join calls
    can_silent_monitor: bool = False              # Supervisor can listen
    can_force_logout: bool = False                # Admin can force logout
    can_change_queue: bool = False                # Can change own queue assignment
    allowed_queues: Set[str] = field(default_factory=set)
    allowed_skill_groups: Set[str] = field(default_factory=set)
    max_concurrent_calls: int = 1
    max_hold_duration_seconds: int = 300          # 5-minute hold limit
    max_wrap_up_seconds: int = 120                # 2-minute wrap-up limit
    can_access_recording: bool = False
    can_modify_call_data: bool = True

    def has_queue_access(self, queue_name: str) -> bool:
        """Check if agent has access to a specific queue."""
        if not self.allowed_queues:
            return True  # No restrictions = access all
        return queue_name in self.allowed_queues

    def has_skill_group_access(self, skill_group: str) -> bool:
        """Check if agent has access to a specific skill group."""
        if not self.allowed_skill_groups:
            return True
        return skill_group in self.allowed_skill_groups

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "can_transfer_internal": self.can_transfer_internal,
            "can_transfer_external": self.can_transfer_external,
            "can_conference": self.can_conference,
            "can_barge_in": self.can_barge_in,
            "can_silent_monitor": self.can_silent_monitor,
            "can_force_logout": self.can_force_logout,
            "max_concurrent_calls": self.max_concurrent_calls,
            "max_hold_duration_seconds": self.max_hold_duration_seconds,
            "max_wrap_up_seconds": self.max_wrap_up_seconds,
            "allowed_queues": list(self.allowed_queues),
            "allowed_skill_groups": list(self.allowed_skill_groups),
        }


@dataclass
class CTICallInfo:
    """
    Detailed information about an active call.

    Tracks call state, timing, and routing metadata.
    """

    call_id: str = ""
    state: CTICallState = CTICallState.IDLE
    agent_id: str = ""
    queue_name: str = ""
    caller_id: str = ""
    called_number: str = ""
    direction: str = "inbound"              # inbound, outbound, internal

    # Timing
    offered_at: Optional[datetime] = None
    answered_at: Optional[datetime] = None
    held_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Hold tracking
    hold_count: int = 0
    total_hold_duration_seconds: float = 0.0

    # Transfer/conference tracking
    transfer_count: int = 0
    original_agent_id: str = ""
    transfer_destinations: List[str] = field(default_factory=list)

    # Wrap-up
    wrap_up_started_at: Optional[datetime] = None
    wrap_up_code: str = ""
    wrap_up_notes: str = ""

    @property
    def talk_duration_seconds(self) -> float:
        """Calculate the talk duration (excluding hold time)."""
        if not self.answered_at:
            return 0.0
        end = self.ended_at or datetime.now()
        total = (end - self.answered_at).total_seconds()
        return max(0.0, total - self.total_hold_duration_seconds)

    @property
    def ring_duration_seconds(self) -> float:
        """Calculate how long the call rang before being answered."""
        if not self.offered_at:
            return 0.0
        answer_time = self.answered_at or self.ended_at or datetime.now()
        return (answer_time - self.offered_at).total_seconds()

    @property
    def current_hold_duration_seconds(self) -> float:
        """Calculate current hold duration if call is on hold."""
        if self.state != CTICallState.HELD or not self.held_at:
            return 0.0
        return (datetime.now() - self.held_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "call_id": self.call_id,
            "state": self.state._state,
            "agent_id": self.agent_id,
            "queue_name": self.queue_name,
            "caller_id": self.caller_id,
            "direction": self.direction,
            "talk_duration_seconds": round(self.talk_duration_seconds, 1),
            "ring_duration_seconds": round(self.ring_duration_seconds, 1),
            "hold_count": self.hold_count,
            "total_hold_duration_seconds": round(
                self.total_hold_duration_seconds, 1
            ),
            "transfer_count": self.transfer_count,
            "wrap_up_code": self.wrap_up_code,
            "offered_at": (
                self.offered_at.isoformat() if self.offered_at else None
            ),
            "answered_at": (
                self.answered_at.isoformat() if self.answered_at else None
            ),
            "ended_at": (
                self.ended_at.isoformat() if self.ended_at else None
            ),
        }

    def __str__(self) -> str:
        return (
            f"CTICallInfo({self.call_id}, state={self.state._state}, "
            f"agent={self.agent_id}, queue={self.queue_name})"
        )


@dataclass
class CTIMessage:
    """
    Complete CTI protocol message.

    Represents a telephony event or action in the CTI system,
    combining message metadata with the event payload.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: CTIMessageType = CTIMessageType.CALL_OFFERED
    call_id: str = ""
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    payload: Dict[str, Any] = field(default_factory=dict)

    # Context
    queue_name: str = ""
    skill_group: str = ""
    caller_id: str = ""
    direction: str = "inbound"

    # State change info
    previous_state: str = ""
    new_state: str = ""

    # Metadata
    source_system: str = ""    # CTI server identifier
    sequence_number: int = 0
    correlation_id: str = ""   # For correlating related events
    parse_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.msg_type,
            "call_id": self.call_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "queue_name": self.queue_name,
            "skill_group": self.skill_group,
            "caller_id": self.caller_id,
            "direction": self.direction,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "source_system": self.source_system,
            "sequence_number": self.sequence_number,
            "correlation_id": self.correlation_id,
        }

    def __str__(self) -> str:
        return (
            f"CTIMessage({self.message_type.msg_type}, "
            f"call={self.call_id}, "
            f"agent={self.agent_id})"
        )
