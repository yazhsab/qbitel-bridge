"""
IVR Protocol Data Structures

Data structures for Interactive Voice Response (IVR) protocol handling:
- IVR message envelope for call events
- DTMF event representation with masking support
- IVR menu tree and flow definitions
- PCI-compliant payment capture flow
- Message type classification for IVR interactions
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class IVRMessageType(Enum):
    """Types of IVR protocol messages."""

    DTMF_INPUT = ("DTMF_INPUT", "DTMF digit input from caller")
    SPEECH_INPUT = ("SPEECH_INPUT", "Speech recognition input from caller")
    PROMPT_PLAY = ("PROMPT_PLAY", "Audio prompt playback to caller")
    MENU_NAVIGATION = ("MENU_NAVIGATION", "Caller navigated an IVR menu")
    TRANSFER = ("TRANSFER", "Call transfer to agent or queue")
    HANGUP = ("HANGUP", "Call disconnected")
    TIMEOUT = ("TIMEOUT", "Input timeout - no response from caller")
    ERROR = ("ERROR", "IVR system error")

    def __init__(self, msg_type: str, description: str):
        self.msg_type = msg_type
        self.description = description


class IVRTransferType(Enum):
    """Types of IVR call transfers."""

    AGENT = "agent"            # Transfer to a specific agent
    QUEUE = "queue"            # Transfer to a skill queue
    EXTERNAL = "external"      # Transfer to external number
    CALLBACK = "callback"      # Schedule callback
    VOICEMAIL = "voicemail"    # Transfer to voicemail


class DTMFDigit(Enum):
    """DTMF digit values with their frequency pairs."""

    D0 = ("0", 941, 1336)
    D1 = ("1", 697, 1209)
    D2 = ("2", 697, 1336)
    D3 = ("3", 697, 1477)
    D4 = ("4", 770, 1209)
    D5 = ("5", 770, 1336)
    D6 = ("6", 770, 1477)
    D7 = ("7", 852, 1209)
    D8 = ("8", 852, 1336)
    D9 = ("9", 852, 1477)
    STAR = ("*", 941, 1209)
    HASH = ("#", 941, 1477)
    A = ("A", 697, 1633)
    B = ("B", 770, 1633)
    C = ("C", 852, 1633)
    D = ("D", 941, 1633)

    def __init__(self, digit: str, low_freq: int, high_freq: int):
        self.digit = digit
        self.low_freq = low_freq
        self.high_freq = high_freq


@dataclass
class DTMFEvent:
    """
    Represents a single DTMF digit input event.

    Tracks individual DTMF tones with timing information and
    supports masking for PCI compliance during payment capture.
    """

    digit: str                          # The DTMF digit (0-9, *, #, A-D)
    duration_ms: int = 100              # Duration of the tone in milliseconds
    timestamp: datetime = field(default_factory=datetime.now)
    call_id: str = ""                   # Associated call identifier
    session_id: str = ""                # IVR session identifier
    masked: bool = False                # Whether the digit has been masked for PCI
    inter_digit_time_ms: int = 0        # Time since previous digit

    @property
    def masked_digit(self) -> str:
        """Get the digit value, masked if PCI masking is active."""
        if self.masked:
            return "*"
        return self.digit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "digit": self.masked_digit,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "call_id": self.call_id,
            "session_id": self.session_id,
            "masked": self.masked,
            "inter_digit_time_ms": self.inter_digit_time_ms,
        }

    def __str__(self) -> str:
        display = self.masked_digit
        return f"DTMFEvent('{display}', {self.duration_ms}ms, call={self.call_id})"


@dataclass
class IVRMenuNode:
    """
    Represents a node in the IVR menu tree.

    Each node corresponds to a menu level in the IVR flow with
    options mapped to DTMF digits that navigate to child nodes
    or trigger actions.
    """

    node_id: str = ""                       # Unique identifier for this menu node
    name: str = ""                          # Human-readable menu name
    prompt_text: str = ""                   # Text of the audio prompt played
    prompt_audio_file: str = ""             # Path to the audio file
    options: Dict[str, str] = field(default_factory=dict)  # digit -> child_node_id
    action: str = ""                        # Action to execute at this node
    timeout_seconds: int = 5                # Input timeout duration
    max_retries: int = 3                    # Maximum retry attempts
    parent_id: str = ""                     # Parent node ID
    is_root: bool = False                   # Whether this is the root menu
    is_payment_node: bool = False           # Whether this node captures payment data
    requires_authentication: bool = False   # Whether auth is required to reach this node

    def get_child_id(self, digit: str) -> Optional[str]:
        """Get the child node ID for a given DTMF digit."""
        return self.options.get(digit)

    @property
    def option_count(self) -> int:
        """Get the number of menu options."""
        return len(self.options)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "prompt_text": self.prompt_text,
            "options": self.options,
            "action": self.action,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "parent_id": self.parent_id,
            "is_root": self.is_root,
            "is_payment_node": self.is_payment_node,
            "requires_authentication": self.requires_authentication,
        }

    def __str__(self) -> str:
        return f"IVRMenuNode({self.node_id}, '{self.name}', options={list(self.options.keys())})"


@dataclass
class IVRFlow:
    """
    Defines a complete IVR call flow.

    Contains the menu tree structure, entry point, and flow metadata
    for routing calls through the IVR system.
    """

    flow_id: str = ""                        # Unique flow identifier
    name: str = ""                           # Human-readable flow name
    version: str = "1.0"                     # Flow version
    description: str = ""                    # Flow description
    entry_node_id: str = ""                  # Root node of the menu tree
    nodes: Dict[str, IVRMenuNode] = field(default_factory=dict)  # node_id -> node
    language: str = "en-US"                  # Default language
    max_call_duration_seconds: int = 3600    # Maximum call duration in IVR
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

    def add_node(self, node: IVRMenuNode) -> None:
        """Add a menu node to the flow."""
        self.nodes[node.node_id] = node
        if node.is_root:
            self.entry_node_id = node.node_id
        self.updated_at = datetime.now()

    def get_node(self, node_id: str) -> Optional[IVRMenuNode]:
        """Get a menu node by ID."""
        return self.nodes.get(node_id)

    def get_path(self, from_node_id: str, to_node_id: str) -> List[str]:
        """
        Find the navigation path between two nodes.

        Returns list of node IDs from source to destination,
        or empty list if no path exists.
        """
        if from_node_id == to_node_id:
            return [from_node_id]

        visited = set()
        queue = [[from_node_id]]

        while queue:
            path = queue.pop(0)
            current = path[-1]

            if current in visited:
                continue
            visited.add(current)

            node = self.nodes.get(current)
            if not node:
                continue

            for digit, child_id in node.options.items():
                if child_id == to_node_id:
                    return path + [child_id]
                if child_id not in visited and child_id in self.nodes:
                    queue.append(path + [child_id])

        return []

    def get_payment_nodes(self) -> List[IVRMenuNode]:
        """Get all nodes marked as payment capture nodes."""
        return [n for n in self.nodes.values() if n.is_payment_node]

    def validate_structure(self) -> List[str]:
        """
        Validate the flow structure for common issues.

        Returns:
            List of validation error messages
        """
        errors = []

        if not self.entry_node_id:
            errors.append("Flow has no entry node defined")
        elif self.entry_node_id not in self.nodes:
            errors.append(
                f"Entry node '{self.entry_node_id}' not found in flow"
            )

        # Check for orphan nodes (unreachable from entry)
        if self.entry_node_id:
            reachable = set()
            self._find_reachable(self.entry_node_id, reachable)
            for node_id in self.nodes:
                if node_id not in reachable:
                    errors.append(
                        f"Node '{node_id}' is unreachable from entry point"
                    )

        # Check for dangling references
        for node_id, node in self.nodes.items():
            for digit, child_id in node.options.items():
                if child_id not in self.nodes:
                    errors.append(
                        f"Node '{node_id}' option '{digit}' references "
                        f"non-existent node '{child_id}'"
                    )

        return errors

    def _find_reachable(self, node_id: str, visited: set) -> None:
        """Recursively find all reachable nodes from a starting point."""
        if node_id in visited:
            return
        visited.add(node_id)

        node = self.nodes.get(node_id)
        if node:
            for child_id in node.options.values():
                self._find_reachable(child_id, visited)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "flow_id": self.flow_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "entry_node_id": self.entry_node_id,
            "node_count": len(self.nodes),
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "language": self.language,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __str__(self) -> str:
        return (
            f"IVRFlow({self.flow_id}, '{self.name}', "
            f"nodes={len(self.nodes)}, active={self.is_active})"
        )


@dataclass
class PCIPaymentCapture:
    """
    PCI-compliant payment capture flow for IVR.

    Manages the state of a DTMF-based payment card capture,
    enforcing PCI DSS requirements for masking and data handling.
    """

    capture_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str = ""
    session_id: str = ""
    card_present: bool = False             # Whether physical card is present
    masking_active: bool = True            # Whether DTMF masking is engaged
    capture_started_at: Optional[datetime] = None
    capture_completed_at: Optional[datetime] = None

    # Card data state (only last 4 stored, rest masked)
    card_digits_received: int = 0
    card_last_four: str = ""               # Only the last 4 digits of card
    card_type: str = ""                    # Visa, Mastercard, Amex, etc.
    expiry_received: bool = False
    cvv_received: bool = False

    # Capture limits
    max_card_digits: int = 19              # Maximum PAN length
    min_card_digits: int = 13              # Minimum PAN length
    max_capture_duration_seconds: int = 120  # Timeout for payment capture

    # Status
    is_complete: bool = False
    is_timed_out: bool = False
    is_cancelled: bool = False
    error_message: str = ""

    def start_capture(self) -> None:
        """Begin the payment capture process with DTMF masking."""
        self.capture_started_at = datetime.now()
        self.masking_active = True
        self.card_digits_received = 0
        self.card_last_four = ""
        self.is_complete = False

    def receive_digit(self, digit: str) -> bool:
        """
        Process a received DTMF digit during payment capture.

        Only stores the last 4 digits per PCI DSS requirements.
        All previous digits are counted but not stored.

        Args:
            digit: The DTMF digit received

        Returns:
            True if digit was accepted, False if capture is full
        """
        if not digit.isdigit():
            return False

        if self.card_digits_received >= self.max_card_digits:
            return False

        self.card_digits_received += 1

        # Rolling last-four buffer
        self.card_last_four = (self.card_last_four + digit)[-4:]

        # Detect card type from initial digits
        if self.card_digits_received <= 2:
            self._detect_card_type()

        return True

    def complete_card_capture(self) -> bool:
        """
        Mark card number capture as complete.

        Returns:
            True if the captured card number meets minimum length
        """
        if self.card_digits_received < self.min_card_digits:
            self.error_message = (
                f"Card number too short: {self.card_digits_received} digits "
                f"(minimum {self.min_card_digits})"
            )
            return False

        return True

    def receive_expiry(self, mmyy: str) -> bool:
        """
        Process expiry date input.

        Args:
            mmyy: Four-digit expiry in MMYY format

        Returns:
            True if expiry format is valid
        """
        if len(mmyy) != 4 or not mmyy.isdigit():
            return False

        month = int(mmyy[:2])
        if month < 1 or month > 12:
            return False

        self.expiry_received = True
        return True

    def receive_cvv(self, cvv: str) -> bool:
        """
        Process CVV/CVC input.

        Args:
            cvv: 3 or 4 digit security code

        Returns:
            True if CVV format is valid
        """
        if not cvv.isdigit():
            return False

        expected_length = 4 if self.card_type == "AMEX" else 3
        if len(cvv) != expected_length:
            return False

        self.cvv_received = True
        return True

    def finalize(self) -> bool:
        """
        Finalize the payment capture.

        Returns:
            True if all required data has been captured
        """
        if (
            self.card_digits_received >= self.min_card_digits
            and self.expiry_received
            and self.cvv_received
        ):
            self.is_complete = True
            self.capture_completed_at = datetime.now()
            self.masking_active = False
            return True

        return False

    def check_timeout(self) -> bool:
        """
        Check if the capture has exceeded its time limit.

        Returns:
            True if timed out
        """
        if self.capture_started_at:
            elapsed = (datetime.now() - self.capture_started_at).total_seconds()
            if elapsed > self.max_capture_duration_seconds:
                self.is_timed_out = True
                self.masking_active = False
                return True
        return False

    def cancel(self) -> None:
        """Cancel the payment capture and clear sensitive data."""
        self.is_cancelled = True
        self.masking_active = False
        self.card_last_four = ""
        self.card_digits_received = 0
        self.expiry_received = False
        self.cvv_received = False

    def _detect_card_type(self) -> None:
        """Detect the card type from the initial digits."""
        prefix = self.card_last_four[:self.card_digits_received]
        if prefix.startswith("4"):
            self.card_type = "VISA"
        elif prefix.startswith("5") or prefix.startswith("2"):
            self.card_type = "MASTERCARD"
        elif prefix.startswith("3"):
            if len(prefix) >= 2 and prefix[1] in ("4", "7"):
                self.card_type = "AMEX"
            else:
                self.card_type = "OTHER"
        elif prefix.startswith("6"):
            self.card_type = "DISCOVER"
        else:
            self.card_type = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (PCI-safe)."""
        return {
            "capture_id": self.capture_id,
            "call_id": self.call_id,
            "session_id": self.session_id,
            "card_present": self.card_present,
            "masking_active": self.masking_active,
            "card_digits_received": self.card_digits_received,
            "card_last_four": f"****{self.card_last_four}" if self.card_last_four else "",
            "card_type": self.card_type,
            "expiry_received": self.expiry_received,
            "cvv_received": self.cvv_received,
            "is_complete": self.is_complete,
            "is_timed_out": self.is_timed_out,
            "is_cancelled": self.is_cancelled,
            "capture_started_at": (
                self.capture_started_at.isoformat() if self.capture_started_at else None
            ),
            "capture_completed_at": (
                self.capture_completed_at.isoformat()
                if self.capture_completed_at
                else None
            ),
        }

    def __str__(self) -> str:
        status = "complete" if self.is_complete else "in_progress"
        if self.is_timed_out:
            status = "timed_out"
        if self.is_cancelled:
            status = "cancelled"
        return (
            f"PCIPaymentCapture({self.capture_id[:8]}..., "
            f"type={self.card_type}, "
            f"digits={self.card_digits_received}, "
            f"status={status})"
        )


@dataclass
class IVRMessage:
    """
    Complete IVR protocol message.

    Represents an event or action in the IVR system, combining
    message metadata with the event payload.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: IVRMessageType = IVRMessageType.DTMF_INPUT
    call_id: str = ""
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    payload: Dict[str, Any] = field(default_factory=dict)

    # Context
    current_node_id: str = ""     # Current IVR menu node
    caller_id: str = ""           # Caller's phone number
    called_number: str = ""       # Dialed number
    language: str = "en-US"       # Caller's language preference

    # Metadata
    parse_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.msg_type,
            "call_id": self.call_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "current_node_id": self.current_node_id,
            "caller_id": self.caller_id,
            "called_number": self.called_number,
            "language": self.language,
        }

    def __str__(self) -> str:
        return (
            f"IVRMessage({self.message_type.msg_type}, "
            f"call={self.call_id}, "
            f"node={self.current_node_id})"
        )
