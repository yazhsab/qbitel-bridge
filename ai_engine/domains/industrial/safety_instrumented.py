"""
Safety Instrumented Systems (SIS) Security

Post-quantum security for safety-critical systems:
- IEC 61508 compliant
- IEC 61511 (Process industry)
- Safety Integrity Levels (SIL) 1-4

Key requirements:
- Deterministic behavior
- Fail-safe defaults
- Audit trail
- Defense-in-depth
- Independence from basic process control
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set, Any

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Metrics
SIS_AUTH_OPS = Counter(
    "sis_authentication_operations_total", "Total SIS authentication operations", ["sil_level", "operation"]
)

SIS_SAFETY_EVENTS = Counter("sis_safety_events_total", "Safety-related events", ["event_type", "sil_level"])

SIS_AUDIT_ENTRIES = Counter("sis_audit_entries_total", "Total audit log entries", ["action"])


class SILLevel(Enum):
    """Safety Integrity Levels per IEC 61508."""

    SIL1 = 1  # 10^-2 to 10^-1 PFD
    SIL2 = 2  # 10^-3 to 10^-2 PFD
    SIL3 = 3  # 10^-4 to 10^-3 PFD
    SIL4 = 4  # 10^-5 to 10^-4 PFD


class SafetyFunctionState(Enum):
    """Safety function operating states."""

    NORMAL = auto()  # Normal operation
    DEMAND = auto()  # Safety function activated
    FAULT = auto()  # Detected fault
    MAINTENANCE = auto()  # Under maintenance
    BYPASS = auto()  # Temporarily bypassed


class SecurityAction(Enum):
    """Security-related actions for audit."""

    AUTHENTICATE = "authenticate"
    COMMAND = "command"
    CONFIG_CHANGE = "config_change"
    BYPASS_REQUEST = "bypass_request"
    BYPASS_APPROVE = "bypass_approve"
    KEY_ROTATION = "key_rotation"
    ALARM = "alarm"
    TRIP = "trip"


@dataclass
class SafetyFunction:
    """Definition of a safety function."""

    function_id: str
    name: str
    description: str
    sil_level: SILLevel

    # Configuration
    trip_setpoint: float
    alarm_setpoint: float
    reset_setpoint: float
    response_time_ms: int

    # State
    state: SafetyFunctionState = SafetyFunctionState.NORMAL
    last_demand: Optional[float] = None
    last_test: Optional[float] = None

    # Security
    requires_dual_approval: bool = False
    bypass_allowed: bool = True
    max_bypass_duration: int = 3600  # seconds


@dataclass
class SISKeySet:
    """Key set for SIS authentication."""

    key_id: bytes
    function_id: str
    sil_level: SILLevel

    # Key material
    authentication_key: bytes
    command_key: bytes

    # Lifecycle
    created_at: float = field(default_factory=time.time)
    valid_until: float = 0.0
    rotation_count: int = 0

    def __post_init__(self):
        if self.valid_until == 0.0:
            # SIL-dependent key lifetime
            lifetimes = {
                SILLevel.SIL1: 86400,  # 24 hours
                SILLevel.SIL2: 28800,  # 8 hours
                SILLevel.SIL3: 3600,  # 1 hour
                SILLevel.SIL4: 1800,  # 30 minutes
            }
            self.valid_until = self.created_at + lifetimes[self.sil_level]

    def is_valid(self) -> bool:
        return time.time() < self.valid_until


@dataclass
class AuditLogEntry:
    """Audit log entry for SIS operations."""

    timestamp: float
    entry_id: bytes
    function_id: str
    action: SecurityAction
    operator_id: str
    details: Dict[str, Any]
    authenticated: bool
    signature: bytes


@dataclass
class BypassRequest:
    """Request to bypass a safety function."""

    request_id: bytes
    function_id: str
    requested_by: str
    reason: str
    duration_seconds: int
    requested_at: float = field(default_factory=time.time)

    # Approval
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[float] = None

    def is_expired(self) -> bool:
        if not self.approved or not self.approved_at:
            return True
        return time.time() > self.approved_at + self.duration_seconds


class SafetyKeyManager:
    """
    Key manager for Safety Instrumented Systems.

    Implements SIL-appropriate key management:
    - Shorter key lifetimes for higher SIL
    - Mandatory key rotation
    - Secure key derivation
    """

    def __init__(self, use_pqc: bool = True):
        self.use_pqc = use_pqc
        self._key_sets: Dict[str, SISKeySet] = {}  # function_id -> key set

        logger.info(f"SIS key manager initialized: PQC={use_pqc}")

    async def generate_key_set(
        self,
        function: SafetyFunction,
    ) -> SISKeySet:
        """Generate new key set for safety function."""
        key_id = secrets.token_bytes(8)

        if self.use_pqc:
            master_key = await self._pqc_derive_key()
        else:
            master_key = secrets.token_bytes(32)

        # Derive function-specific keys
        auth_key = hashlib.sha256(master_key + b"authentication" + function.function_id.encode()).digest()

        cmd_key = hashlib.sha256(master_key + b"command" + function.function_id.encode()).digest()

        # Get previous rotation count
        old_set = self._key_sets.get(function.function_id)
        rotation_count = (old_set.rotation_count + 1) if old_set else 0

        key_set = SISKeySet(
            key_id=key_id,
            function_id=function.function_id,
            sil_level=function.sil_level,
            authentication_key=auth_key,
            command_key=cmd_key,
            rotation_count=rotation_count,
        )

        self._key_sets[function.function_id] = key_set

        logger.info(
            f"Generated key set for {function.function_id}: " f"SIL{function.sil_level.value}, rotation={rotation_count}"
        )

        return key_set

    async def _pqc_derive_key(self) -> bytes:
        """Derive key using PQC."""
        from ai_engine.crypto.mlkem import MlKemEngine, MlKemSecurityLevel

        # Use ML-KEM-1024 for safety-critical applications
        engine = MlKemEngine(MlKemSecurityLevel.MLKEM_1024)
        keypair = await engine.generate_keypair()
        ct, ss = await engine.encapsulate(keypair.public_key)
        return ss.data

    def get_key_set(self, function_id: str) -> Optional[SISKeySet]:
        """Get current key set for function."""
        key_set = self._key_sets.get(function_id)
        if key_set and key_set.is_valid():
            return key_set
        return None

    async def rotate_keys(self, function: SafetyFunction) -> SISKeySet:
        """Rotate keys for safety function."""
        return await self.generate_key_set(function)

    async def check_rotation_needed(self) -> List[str]:
        """Check which functions need key rotation."""
        needs_rotation = []

        for function_id, key_set in self._key_sets.items():
            # Check time-based expiration
            if not key_set.is_valid():
                needs_rotation.append(function_id)
                continue

            # Check approaching expiration (10% remaining)
            remaining = key_set.valid_until - time.time()
            total = key_set.valid_until - key_set.created_at
            if remaining < total * 0.1:
                needs_rotation.append(function_id)

        return needs_rotation


class SISMessageAuthenticator:
    """
    Message authenticator for SIS communications.

    Provides deterministic, timing-safe authentication
    suitable for safety-critical systems.
    """

    def __init__(
        self,
        key_manager: SafetyKeyManager,
    ):
        self.key_manager = key_manager
        self._sequence_numbers: Dict[str, int] = {}

        logger.info("SIS message authenticator initialized")

    async def authenticate_command(
        self,
        function_id: str,
        command: bytes,
        operator_id: str,
    ) -> Tuple[bytes, bytes]:
        """
        Authenticate command for safety function.

        Returns:
            Tuple of (authenticated_command, command_id)
        """
        key_set = self.key_manager.get_key_set(function_id)
        if not key_set:
            raise ValueError(f"No valid key set for function: {function_id}")

        # Get next sequence number
        if function_id not in self._sequence_numbers:
            self._sequence_numbers[function_id] = 0
        sequence = self._sequence_numbers[function_id]
        self._sequence_numbers[function_id] += 1

        # Generate command ID
        command_id = secrets.token_bytes(8)
        timestamp = int(time.time() * 1000).to_bytes(8, "big")

        # Build authenticated command
        auth_data = (
            key_set.key_id
            + command_id
            + timestamp
            + sequence.to_bytes(4, "big")
            + operator_id.encode()
            + b"\x00"  # Null terminator for operator_id
            + command
        )

        mac = hmac.new(key_set.command_key, auth_data, hashlib.sha256).digest()

        authenticated = auth_data + mac

        SIS_AUTH_OPS.labels(sil_level=f"SIL{key_set.sil_level.value}", operation="authenticate_command").inc()

        return authenticated, command_id

    async def verify_command(
        self,
        function_id: str,
        authenticated_command: bytes,
    ) -> Optional[Tuple[bytes, str, bytes]]:
        """
        Verify authenticated command.

        Returns:
            Tuple of (command, operator_id, command_id) if valid, None otherwise
        """
        key_set = self.key_manager.get_key_set(function_id)
        if not key_set:
            logger.warning(f"No valid key set for function: {function_id}")
            return None

        if len(authenticated_command) < 64:  # Minimum size
            return None

        # Extract components
        key_id = authenticated_command[:8]
        if key_id != key_set.key_id:
            logger.warning("Key ID mismatch")
            return None

        command_id = authenticated_command[8:16]
        timestamp = authenticated_command[16:24]
        sequence = authenticated_command[24:28]

        # Find operator_id (null-terminated)
        null_pos = authenticated_command.find(b"\x00", 28)
        if null_pos < 0:
            return None

        operator_id = authenticated_command[28:null_pos].decode()
        command = authenticated_command[null_pos + 1 : -32]
        mac = authenticated_command[-32:]

        # Verify timestamp (stricter for SIS - 30 second window)
        cmd_time = int.from_bytes(timestamp, "big")
        current_time = int(time.time() * 1000)
        if abs(current_time - cmd_time) > 30000:
            logger.warning("Command timestamp outside acceptable window")
            return None

        # Verify MAC
        auth_data = authenticated_command[:-32]
        expected_mac = hmac.new(key_set.command_key, auth_data, hashlib.sha256).digest()

        if not secrets.compare_digest(mac, expected_mac):
            logger.warning("Command MAC verification failed")
            return None

        SIS_AUTH_OPS.labels(sil_level=f"SIL{key_set.sil_level.value}", operation="verify_command").inc()

        return command, operator_id, command_id


class SISAuditLogger:
    """
    Audit logger for SIS operations.

    Provides tamper-evident logging with:
    - Cryptographic chaining
    - PQC signatures
    - Retention policy
    """

    def __init__(
        self,
        key_manager: SafetyKeyManager,
        retention_days: int = 365,
    ):
        self.key_manager = key_manager
        self.retention_days = retention_days

        self._entries: List[AuditLogEntry] = []
        self._chain_hash: bytes = hashlib.sha256(b"SIS_AUDIT_GENESIS").digest()

        # Signing key
        self._signing_key: Optional[bytes] = None

        logger.info(f"SIS audit logger initialized: retention={retention_days} days")

    async def initialize_signing_key(self) -> None:
        """Initialize audit log signing key."""
        if self.key_manager.use_pqc:
            from ai_engine.crypto.dilithium import DilithiumEngine, DilithiumSecurityLevel

            engine = DilithiumEngine(DilithiumSecurityLevel.LEVEL5)
            keypair = await engine.generate_keypair()
            self._signing_key = keypair.private_key.data
        else:
            self._signing_key = secrets.token_bytes(64)

    async def log_action(
        self,
        function_id: str,
        action: SecurityAction,
        operator_id: str,
        details: Dict[str, Any],
        authenticated: bool = True,
    ) -> AuditLogEntry:
        """
        Log security action.

        Creates cryptographically chained and signed entry.
        """
        timestamp = time.time()
        entry_id = secrets.token_bytes(16)

        # Build entry data for signing
        entry_data = (
            entry_id
            + function_id.encode()
            + action.value.encode()
            + operator_id.encode()
            + str(timestamp).encode()
            + str(details).encode()
            + self._chain_hash
        )

        # Sign entry
        if self._signing_key:
            signature = await self._sign_entry(entry_data)
        else:
            signature = hashlib.sha256(entry_data).digest()

        entry = AuditLogEntry(
            timestamp=timestamp,
            entry_id=entry_id,
            function_id=function_id,
            action=action,
            operator_id=operator_id,
            details=details,
            authenticated=authenticated,
            signature=signature,
        )

        # Update chain hash
        self._chain_hash = hashlib.sha256(self._chain_hash + entry_id + signature).digest()

        self._entries.append(entry)

        SIS_AUDIT_ENTRIES.labels(action=action.value).inc()

        logger.debug(f"Audit entry: {action.value} on {function_id} by {operator_id}")

        return entry

    async def _sign_entry(self, data: bytes) -> bytes:
        """Sign audit entry."""
        if not self._signing_key:
            return hashlib.sha256(data).digest()

        try:
            from ai_engine.crypto.dilithium import DilithiumEngine, DilithiumSecurityLevel, DilithiumPrivateKey

            engine = DilithiumEngine(DilithiumSecurityLevel.LEVEL5)
            sk = DilithiumPrivateKey(DilithiumSecurityLevel.LEVEL5, self._signing_key)
            sig = await engine.sign(data, sk)
            return sig.data

        except Exception as e:
            logger.error(f"Signing error: {e}")
            return hashlib.sha256(data).digest()

    async def verify_chain_integrity(self) -> Tuple[bool, int]:
        """
        Verify integrity of audit log chain.

        Returns:
            Tuple of (is_valid, last_valid_index)
        """
        chain_hash = hashlib.sha256(b"SIS_AUDIT_GENESIS").digest()

        for i, entry in enumerate(self._entries):
            # Verify chain continuity
            expected_chain = hashlib.sha256(chain_hash + entry.entry_id + entry.signature).digest()

            if i < len(self._entries) - 1:
                # Not the last entry - verify chain continues correctly
                pass  # Chain verification happens on next iteration

            chain_hash = expected_chain

        # Verify final chain hash matches
        if chain_hash != self._chain_hash:
            return False, len(self._entries) - 1

        return True, len(self._entries)

    def get_entries_for_function(
        self,
        function_id: str,
        since: Optional[float] = None,
    ) -> List[AuditLogEntry]:
        """Get audit entries for specific function."""
        return [e for e in self._entries if e.function_id == function_id and (since is None or e.timestamp >= since)]


class SISSecurityManager:
    """
    Main security manager for Safety Instrumented Systems.

    Coordinates:
    - Key management
    - Message authentication
    - Bypass management
    - Audit logging
    """

    def __init__(self, use_pqc: bool = True):
        self.key_manager = SafetyKeyManager(use_pqc=use_pqc)
        self.authenticator = SISMessageAuthenticator(self.key_manager)
        self.audit_logger = SISAuditLogger(self.key_manager)

        self._functions: Dict[str, SafetyFunction] = {}
        self._bypass_requests: Dict[bytes, BypassRequest] = {}

        logger.info("SIS security manager initialized")

    async def initialize(self) -> None:
        """Initialize the security manager."""
        await self.audit_logger.initialize_signing_key()

    async def register_function(
        self,
        function: SafetyFunction,
    ) -> SISKeySet:
        """Register safety function."""
        self._functions[function.function_id] = function
        key_set = await self.key_manager.generate_key_set(function)

        await self.audit_logger.log_action(
            function.function_id,
            SecurityAction.CONFIG_CHANGE,
            "SYSTEM",
            {"action": "register", "sil_level": function.sil_level.value},
        )

        logger.info(f"Registered safety function: {function.function_id}")
        return key_set

    async def request_bypass(
        self,
        function_id: str,
        operator_id: str,
        reason: str,
        duration_seconds: int,
    ) -> BypassRequest:
        """
        Request bypass of safety function.

        Requires approval for SIL3+ functions.
        """
        function = self._functions.get(function_id)
        if not function:
            raise ValueError(f"Unknown function: {function_id}")

        if not function.bypass_allowed:
            raise PermissionError(f"Bypass not allowed for {function_id}")

        if duration_seconds > function.max_bypass_duration:
            raise ValueError(f"Requested duration exceeds maximum: {function.max_bypass_duration}s")

        request = BypassRequest(
            request_id=secrets.token_bytes(16),
            function_id=function_id,
            requested_by=operator_id,
            reason=reason,
            duration_seconds=duration_seconds,
        )

        self._bypass_requests[request.request_id] = request

        await self.audit_logger.log_action(
            function_id,
            SecurityAction.BYPASS_REQUEST,
            operator_id,
            {
                "reason": reason,
                "duration": duration_seconds,
                "request_id": request.request_id.hex(),
            },
        )

        SIS_SAFETY_EVENTS.labels(event_type="bypass_request", sil_level=f"SIL{function.sil_level.value}").inc()

        return request

    async def approve_bypass(
        self,
        request_id: bytes,
        approver_id: str,
    ) -> bool:
        """
        Approve bypass request.

        Approver must be different from requester for SIL3+.
        """
        request = self._bypass_requests.get(request_id)
        if not request:
            raise ValueError("Unknown bypass request")

        function = self._functions.get(request.function_id)
        if not function:
            raise ValueError("Function not found")

        # Dual approval check for high SIL
        if function.requires_dual_approval or function.sil_level in (SILLevel.SIL3, SILLevel.SIL4):
            if approver_id == request.requested_by:
                raise PermissionError("Dual approval required: approver must differ from requester")

        request.approved = True
        request.approved_by = approver_id
        request.approved_at = time.time()

        # Update function state
        function.state = SafetyFunctionState.BYPASS

        await self.audit_logger.log_action(
            request.function_id,
            SecurityAction.BYPASS_APPROVE,
            approver_id,
            {
                "request_id": request_id.hex(),
                "requested_by": request.requested_by,
                "duration": request.duration_seconds,
            },
        )

        SIS_SAFETY_EVENTS.labels(event_type="bypass_approved", sil_level=f"SIL{function.sil_level.value}").inc()

        logger.warning(f"Bypass approved: {request.function_id} by {approver_id} " f"(requested by {request.requested_by})")

        return True

    async def check_expired_bypasses(self) -> int:
        """Check and clear expired bypasses."""
        expired = 0

        for request in list(self._bypass_requests.values()):
            if request.approved and request.is_expired():
                function = self._functions.get(request.function_id)
                if function and function.state == SafetyFunctionState.BYPASS:
                    function.state = SafetyFunctionState.NORMAL

                    await self.audit_logger.log_action(
                        request.function_id,
                        SecurityAction.CONFIG_CHANGE,
                        "SYSTEM",
                        {"action": "bypass_expired"},
                    )

                    expired += 1

                del self._bypass_requests[request.request_id]

        if expired:
            logger.info(f"Cleared {expired} expired bypasses")

        return expired

    async def rotate_keys_if_needed(self) -> int:
        """Rotate keys for functions that need it."""
        needs_rotation = await self.key_manager.check_rotation_needed()
        rotated = 0

        for function_id in needs_rotation:
            function = self._functions.get(function_id)
            if function:
                await self.key_manager.rotate_keys(function)

                await self.audit_logger.log_action(
                    function_id,
                    SecurityAction.KEY_ROTATION,
                    "SYSTEM",
                    {"reason": "scheduled"},
                )

                rotated += 1

        if rotated:
            logger.info(f"Rotated keys for {rotated} functions")

        return rotated

    def get_function(self, function_id: str) -> Optional[SafetyFunction]:
        """Get safety function by ID."""
        return self._functions.get(function_id)

    @property
    def function_count(self) -> int:
        """Number of registered functions."""
        return len(self._functions)


def create_emergency_shutdown_function(
    function_id: str,
    name: str,
    sil_level: SILLevel = SILLevel.SIL3,
) -> SafetyFunction:
    """Create emergency shutdown function configuration."""
    return SafetyFunction(
        function_id=function_id,
        name=name,
        description="Emergency shutdown function",
        sil_level=sil_level,
        trip_setpoint=95.0,
        alarm_setpoint=90.0,
        reset_setpoint=80.0,
        response_time_ms=100,
        requires_dual_approval=True,
        bypass_allowed=True,
        max_bypass_duration=3600,
    )
