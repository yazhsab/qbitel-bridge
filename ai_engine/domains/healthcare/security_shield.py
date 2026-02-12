"""
Medical Device Security Shield

External proxy providing PQC protection for legacy medical devices
that cannot be modified or updated. Based on MIT's IMDShield research.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    Network                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                 Security Shield                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ PQC Engine  │  │ Session     │  │ Protocol            │  │
│  │ (ML-KEM-768)│  │ Manager     │  │ Translator          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │ (Legacy Protocol)
┌───────────────────────▼─────────────────────────────────────┐
│              Legacy Medical Device                           │
│  (Pacemaker, ICD, Insulin Pump, etc.)                       │
└─────────────────────────────────────────────────────────────┘

The shield handles all quantum-safe cryptographic operations,
allowing legacy devices to benefit from PQC without modification.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


class ShieldMode(Enum):
    """Operating modes for the security shield."""

    TRANSPARENT = auto()  # Pass-through with encryption
    FILTERING = auto()  # Filter unauthorized commands
    MONITORING = auto()  # Log and alert only
    EMERGENCY = auto()  # Allow all (for medical emergencies)


class DeviceProtocol(Enum):
    """Legacy device communication protocols."""

    MICS = auto()  # Medical Implant Communication Service (402-405 MHz)
    MEDRADIO = auto()  # Medical Device Radio (401-406 MHz)
    BLUETOOTH_LE = auto()  # Bluetooth Low Energy
    ZIGBEE = auto()  # ZigBee (medical profile)
    PROPRIETARY = auto()  # Vendor-specific


@dataclass
class ShieldConfiguration:
    """Configuration for a security shield instance."""

    shield_id: str
    device_id: str
    device_protocol: DeviceProtocol
    mode: ShieldMode = ShieldMode.TRANSPARENT
    allowed_commands: List[bytes] = field(default_factory=list)
    blocked_commands: List[bytes] = field(default_factory=list)
    session_timeout_seconds: int = 3600
    max_message_size: int = 1024
    enable_audit_log: bool = True
    emergency_override_enabled: bool = True

    # PQC settings
    pqc_algorithm: str = "ml-kem-768"
    signature_algorithm: str = "falcon-512"


@dataclass
class AuthorizedPeer:
    """Authorized peer that can communicate with the protected device."""

    peer_id: str
    public_key: bytes
    roles: List[str]  # e.g., ["clinician", "programmer"]
    authorized_commands: List[bytes]
    last_authenticated: float = 0
    session_key: Optional[bytes] = None


@dataclass
class AuditLogEntry:
    """Audit log entry for shield operations."""

    timestamp: float
    event_type: str
    peer_id: Optional[str]
    command: Optional[bytes]
    allowed: bool
    details: str


# Prometheus metrics
SHIELD_MESSAGES = Counter(
    "healthcare_shield_messages_total", "Messages processed by security shield", ["direction", "device_id", "allowed"]
)

SHIELD_LATENCY = Histogram(
    "healthcare_shield_latency_seconds", "Security shield processing latency", ["operation", "device_id"]
)

SHIELD_ATTACKS_BLOCKED = Counter(
    "healthcare_shield_attacks_blocked_total", "Attacks blocked by security shield", ["attack_type", "device_id"]
)


class LegacyDeviceProxy:
    """
    Proxy for communicating with a legacy medical device.

    Handles protocol translation and message forwarding.
    """

    def __init__(
        self,
        device_id: str,
        protocol: DeviceProtocol,
        device_address: str,
    ):
        self.device_id = device_id
        self.protocol = protocol
        self.device_address = device_address
        self._connected = False

    async def connect(self) -> bool:
        """Connect to the legacy device."""
        # Implementation depends on device protocol
        logger.info(f"Connecting to device {self.device_id} via {self.protocol.name}")
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Disconnect from the legacy device."""
        logger.info(f"Disconnecting from device {self.device_id}")
        self._connected = False

    async def send(self, data: bytes) -> bool:
        """Send data to the device."""
        if not self._connected:
            raise RuntimeError("Not connected to device")
        # Actual implementation would send via appropriate protocol
        logger.debug(f"Sending {len(data)} bytes to {self.device_id}")
        return True

    async def receive(self, timeout: float = 5.0) -> Optional[bytes]:
        """Receive data from the device."""
        if not self._connected:
            raise RuntimeError("Not connected to device")
        # Actual implementation would receive via appropriate protocol
        return None

    @property
    def is_connected(self) -> bool:
        return self._connected


class MedicalDeviceSecurityShield:
    """
    Security shield providing PQC protection for legacy medical devices.

    The shield sits between the network and the legacy device, handling
    all cryptographic operations and access control.
    """

    def __init__(
        self,
        config: ShieldConfiguration,
        device_proxy: LegacyDeviceProxy,
    ):
        self.config = config
        self.proxy = device_proxy

        # Authorized peers
        self._authorized_peers: Dict[str, AuthorizedPeer] = {}

        # Audit log
        self._audit_log: List[AuditLogEntry] = []
        self._max_audit_entries = 10000

        # Initialize PQC engines
        self._initialize_pqc()

        # Shield state
        self._active = False
        self._emergency_mode = False

        logger.info(
            f"Security shield initialized: device={config.device_id}, "
            f"mode={config.mode.name}, protocol={config.device_protocol.name}"
        )

    def _initialize_pqc(self) -> None:
        """Initialize PQC cryptographic engines."""
        from ai_engine.crypto.mlkem import MlKemEngine, MlKemSecurityLevel
        from ai_engine.crypto.falcon import FalconEngine, FalconSecurityLevel

        # Select ML-KEM level
        level_map = {
            "ml-kem-512": MlKemSecurityLevel.MLKEM_512,
            "ml-kem-768": MlKemSecurityLevel.MLKEM_768,
            "ml-kem-1024": MlKemSecurityLevel.MLKEM_1024,
        }
        mlkem_level = level_map.get(self.config.pqc_algorithm, MlKemSecurityLevel.MLKEM_768)

        self._kem_engine = MlKemEngine(mlkem_level)

        # Select Falcon level
        falcon_level = (
            FalconSecurityLevel.FALCON_512 if "512" in self.config.signature_algorithm else FalconSecurityLevel.FALCON_1024
        )
        self._sig_engine = FalconEngine(falcon_level)

        # Generate shield's key pair
        self._keypair = None  # Will be generated on start

    async def start(self) -> None:
        """Start the security shield."""
        logger.info(f"Starting security shield for {self.config.device_id}")

        # Generate key pair
        self._keypair = await self._kem_engine.generate_keypair()

        # Connect to legacy device
        await self.proxy.connect()

        self._active = True

        logger.info(f"Security shield active for {self.config.device_id}")

    async def stop(self) -> None:
        """Stop the security shield."""
        logger.info(f"Stopping security shield for {self.config.device_id}")

        self._active = False

        await self.proxy.disconnect()

    async def register_peer(
        self,
        peer_id: str,
        public_key: bytes,
        roles: List[str],
        authorized_commands: Optional[List[bytes]] = None,
    ) -> bool:
        """
        Register an authorized peer.

        Args:
            peer_id: Unique peer identifier
            public_key: Peer's ML-KEM public key
            roles: Peer's roles (clinician, programmer, etc.)
            authorized_commands: Commands this peer can send

        Returns:
            True if registration successful
        """
        if authorized_commands is None:
            authorized_commands = []

        peer = AuthorizedPeer(
            peer_id=peer_id,
            public_key=public_key,
            roles=roles,
            authorized_commands=authorized_commands,
        )

        self._authorized_peers[peer_id] = peer

        self._log_event("peer_registered", peer_id, None, True, f"Roles: {roles}")

        logger.info(f"Registered peer {peer_id} with roles {roles}")

        return True

    async def authenticate_peer(
        self,
        peer_id: str,
        peer_public_key: bytes,
    ) -> Optional[bytes]:
        """
        Authenticate a peer and establish session key.

        Args:
            peer_id: Peer identifier
            peer_public_key: Peer's ephemeral public key

        Returns:
            Session key if authentication successful, None otherwise
        """
        if peer_id not in self._authorized_peers:
            self._log_event("auth_failed", peer_id, None, False, "Unknown peer")
            SHIELD_ATTACKS_BLOCKED.labels(attack_type="unauthorized_peer", device_id=self.config.device_id).inc()
            return None

        peer = self._authorized_peers[peer_id]

        start = time.time()

        # Perform ML-KEM key exchange
        from ai_engine.crypto.mlkem import MlKemPublicKey

        mlkem_pk = MlKemPublicKey(self._kem_engine.level, peer_public_key)
        ciphertext, shared_secret = await self._kem_engine.encapsulate(mlkem_pk)

        session_key = shared_secret.data

        # Store session key
        peer.session_key = session_key
        peer.last_authenticated = time.time()

        elapsed = time.time() - start
        SHIELD_LATENCY.labels(operation="authenticate", device_id=self.config.device_id).observe(elapsed)

        self._log_event("auth_success", peer_id, None, True, f"Session established in {elapsed:.3f}s")

        logger.info(f"Authenticated peer {peer_id}")

        return session_key

    async def process_inbound(
        self,
        peer_id: str,
        encrypted_message: bytes,
    ) -> Tuple[bool, Optional[bytes]]:
        """
        Process an inbound message from a peer to the device.

        Args:
            peer_id: Sender's peer ID
            encrypted_message: Encrypted message from peer

        Returns:
            Tuple of (allowed, response)
        """
        if not self._active:
            return False, None

        # Emergency mode bypasses all checks
        if self._emergency_mode:
            logger.warning(f"Emergency mode: allowing message from {peer_id}")
            return await self._forward_to_device(encrypted_message)

        # Check if peer is authenticated
        if peer_id not in self._authorized_peers:
            SHIELD_MESSAGES.labels(direction="inbound", device_id=self.config.device_id, allowed="false").inc()
            self._log_event("message_blocked", peer_id, None, False, "Unknown peer")
            return False, None

        peer = self._authorized_peers[peer_id]

        if peer.session_key is None:
            self._log_event("message_blocked", peer_id, None, False, "No session established")
            return False, None

        start = time.time()

        # Decrypt message
        try:
            plaintext = await self._decrypt_message(encrypted_message, peer.session_key)
        except Exception as e:
            SHIELD_ATTACKS_BLOCKED.labels(attack_type="decrypt_failure", device_id=self.config.device_id).inc()
            self._log_event("decrypt_failed", peer_id, None, False, str(e))
            return False, None

        # Check if command is allowed
        allowed = self._check_command_allowed(plaintext, peer)

        if not allowed:
            SHIELD_MESSAGES.labels(direction="inbound", device_id=self.config.device_id, allowed="false").inc()
            SHIELD_ATTACKS_BLOCKED.labels(attack_type="unauthorized_command", device_id=self.config.device_id).inc()
            self._log_event("command_blocked", peer_id, plaintext[:32], False, "Command not authorized for peer")
            return False, None

        # Forward to device
        success, response = await self._forward_to_device(plaintext)

        elapsed = time.time() - start
        SHIELD_LATENCY.labels(operation="process_inbound", device_id=self.config.device_id).observe(elapsed)
        SHIELD_MESSAGES.labels(direction="inbound", device_id=self.config.device_id, allowed="true").inc()

        self._log_event("message_forwarded", peer_id, plaintext[:32], True, f"Processed in {elapsed:.3f}s")

        # Encrypt response if any
        if response and peer.session_key:
            response = await self._encrypt_message(response, peer.session_key)

        return True, response

    async def process_outbound(
        self,
        data: bytes,
        target_peer_id: str,
    ) -> Optional[bytes]:
        """
        Process an outbound message from the device to a peer.

        Args:
            data: Plaintext data from device
            target_peer_id: Target peer ID

        Returns:
            Encrypted message for peer, or None if peer not found
        """
        if target_peer_id not in self._authorized_peers:
            return None

        peer = self._authorized_peers[target_peer_id]

        if peer.session_key is None:
            return None

        start = time.time()

        encrypted = await self._encrypt_message(data, peer.session_key)

        elapsed = time.time() - start
        SHIELD_LATENCY.labels(operation="process_outbound", device_id=self.config.device_id).observe(elapsed)
        SHIELD_MESSAGES.labels(direction="outbound", device_id=self.config.device_id, allowed="true").inc()

        return encrypted

    def _check_command_allowed(
        self,
        command: bytes,
        peer: AuthorizedPeer,
    ) -> bool:
        """Check if a command is allowed for a peer."""
        if self.config.mode == ShieldMode.MONITORING:
            return True  # Monitor only, don't block

        # Check explicit blocklist
        for blocked in self.config.blocked_commands:
            if command.startswith(blocked):
                return False

        # Check if command is in peer's authorized list
        if peer.authorized_commands:
            for allowed in peer.authorized_commands:
                if command.startswith(allowed):
                    return True
            return False

        # Check global allowlist
        if self.config.allowed_commands:
            for allowed in self.config.allowed_commands:
                if command.startswith(allowed):
                    return True
            return False

        # Default: allow in transparent mode
        return self.config.mode == ShieldMode.TRANSPARENT

    async def _forward_to_device(
        self,
        data: bytes,
    ) -> Tuple[bool, Optional[bytes]]:
        """Forward data to the legacy device."""
        if not self.proxy.is_connected:
            return False, None

        success = await self.proxy.send(data)
        if not success:
            return False, None

        # Wait for response
        response = await self.proxy.receive(timeout=5.0)

        return True, response

    async def _encrypt_message(
        self,
        plaintext: bytes,
        session_key: bytes,
    ) -> bytes:
        """Encrypt a message using AES-256-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import secrets

        aesgcm = AESGCM(session_key)
        nonce = secrets.token_bytes(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        return nonce + ciphertext

    async def _decrypt_message(
        self,
        ciphertext: bytes,
        session_key: bytes,
    ) -> bytes:
        """Decrypt a message using AES-256-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = ciphertext[:12]
        actual_ciphertext = ciphertext[12:]

        aesgcm = AESGCM(session_key)
        return aesgcm.decrypt(nonce, actual_ciphertext, None)

    def enable_emergency_mode(self) -> None:
        """
        Enable emergency mode.

        In emergency mode, all access control is bypassed to ensure
        medical professionals can always access the device.
        """
        if not self.config.emergency_override_enabled:
            logger.warning("Emergency mode not enabled in configuration")
            return

        logger.critical(f"EMERGENCY MODE ENABLED for {self.config.device_id}")
        self._emergency_mode = True
        self._log_event("emergency_mode_enabled", None, None, True, "All access control bypassed")

    def disable_emergency_mode(self) -> None:
        """Disable emergency mode."""
        logger.info(f"Emergency mode disabled for {self.config.device_id}")
        self._emergency_mode = False
        self._log_event("emergency_mode_disabled", None, None, True, "Normal access control resumed")

    def _log_event(
        self,
        event_type: str,
        peer_id: Optional[str],
        command: Optional[bytes],
        allowed: bool,
        details: str,
    ) -> None:
        """Add entry to audit log."""
        if not self.config.enable_audit_log:
            return

        entry = AuditLogEntry(
            timestamp=time.time(),
            event_type=event_type,
            peer_id=peer_id,
            command=command,
            allowed=allowed,
            details=details,
        )

        self._audit_log.append(entry)

        # Trim log if too large
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries :]

    def get_audit_log(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        entries = self._audit_log[-limit:]

        if event_type:
            entries = [e for e in entries if e.event_type == event_type]

        return [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "peer_id": e.peer_id,
                "allowed": e.allowed,
                "details": e.details,
            }
            for e in entries
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get shield status."""
        return {
            "shield_id": self.config.shield_id,
            "device_id": self.config.device_id,
            "active": self._active,
            "emergency_mode": self._emergency_mode,
            "mode": self.config.mode.name,
            "protocol": self.config.device_protocol.name,
            "authorized_peers": len(self._authorized_peers),
            "device_connected": self.proxy.is_connected,
            "audit_log_entries": len(self._audit_log),
        }
