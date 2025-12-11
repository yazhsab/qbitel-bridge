"""
PLC (Programmable Logic Controller) Authentication

Post-quantum authentication for PLCs:
- Firmware validation
- Program integrity verification
- Secure boot support
- Vendor-specific protocol support

Key challenges:
- Proprietary protocols (Siemens S7, Allen-Bradley, Mitsubishi)
- Limited CPU/memory resources
- Real-time scan cycle requirements
- 10-20 year lifecycles
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Metrics
PLC_AUTH_OPS = Counter(
    'plc_authentication_operations_total',
    'Total PLC authentication operations',
    ['vendor', 'operation']
)

PLC_FIRMWARE_VALIDATIONS = Counter(
    'plc_firmware_validations_total',
    'Total firmware validations',
    ['result']
)

PLC_ACTIVE_SESSIONS = Gauge(
    'plc_active_authenticated_sessions',
    'Number of active authenticated PLC sessions'
)


class PLCVendor(Enum):
    """PLC vendors with distinct protocol requirements."""

    SIEMENS = "siemens"             # S7 protocol
    ALLEN_BRADLEY = "allen-bradley"  # EtherNet/IP, CIP
    MITSUBISHI = "mitsubishi"       # MELSEC
    SCHNEIDER = "schneider"         # Modbus, Unity
    ABB = "abb"                     # ABB protocols
    GENERIC = "generic"             # Generic Modbus/OPC-UA


class PLCSecurityMode(Enum):
    """PLC security operating modes."""

    DISABLED = auto()       # No security (legacy mode)
    MONITORING = auto()     # Passive monitoring only
    AUTHENTICATION = auto()  # Active authentication
    FULL = auto()           # Full security with encryption


class PLCCapability(Enum):
    """PLC security capabilities."""

    BASIC_AUTH = auto()         # Username/password
    CERTIFICATE = auto()        # X.509 certificates
    SECURE_BOOT = auto()        # Secure boot chain
    FIRMWARE_SIGN = auto()      # Signed firmware updates
    PQC_CAPABLE = auto()        # Post-quantum support
    HSM_SUPPORT = auto()        # Hardware security module


@dataclass
class PLCProfile:
    """Profile of a PLC's capabilities and requirements."""

    device_id: str
    vendor: PLCVendor
    model: str
    firmware_version: str

    # Resource constraints
    cpu_mhz: int
    memory_kb: int
    scan_cycle_ms: float

    # Security capabilities
    capabilities: Set[PLCCapability] = field(default_factory=set)
    security_mode: PLCSecurityMode = PLCSecurityMode.AUTHENTICATION


@dataclass
class PLCSecurityContext:
    """Security context for PLC session."""

    session_id: bytes
    plc_id: str
    client_id: str
    vendor: PLCVendor

    # Key material
    authentication_key: bytes
    program_validation_key: bytes

    # Session state
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    authenticated_commands: int = 0

    # Limits
    max_age_seconds: int = 3600
    max_commands: int = 100000

    def is_valid(self) -> bool:
        """Check session validity."""
        age = time.time() - self.created_at
        return (
            age < self.max_age_seconds and
            self.authenticated_commands < self.max_commands
        )


@dataclass
class FirmwareManifest:
    """Firmware update manifest."""

    version: str
    vendor: PLCVendor
    model: str
    size_bytes: int
    sha256_hash: bytes
    signature: bytes  # PQC signature
    timestamp: float
    release_notes: str = ""


@dataclass
class ProgramBlock:
    """PLC program block for validation."""

    block_type: str  # OB, FC, FB, DB, etc.
    block_number: int
    name: str
    content_hash: bytes
    signature: Optional[bytes] = None
    timestamp: float = field(default_factory=time.time)


class PLCKeyManager:
    """
    Key manager for PLC authentication.

    Handles:
    - Session key derivation
    - Firmware signing keys
    - Program validation keys
    """

    def __init__(self, use_pqc: bool = True):
        self.use_pqc = use_pqc

        self._plc_keys: Dict[str, bytes] = {}  # PLC ID -> master key
        self._signing_keys: Dict[PLCVendor, Tuple[bytes, bytes]] = {}  # vendor -> (pub, priv)

        logger.info(f"PLC key manager initialized: PQC={use_pqc}")

    async def initialize_plc(
        self,
        plc_id: str,
        vendor: PLCVendor,
    ) -> bytes:
        """
        Initialize key material for a PLC.

        Returns master key (should be securely provisioned).
        """
        if self.use_pqc:
            master_key = await self._generate_pqc_key()
        else:
            master_key = secrets.token_bytes(32)

        self._plc_keys[plc_id] = master_key

        logger.debug(f"Initialized keys for PLC: {plc_id}")
        return master_key

    async def _generate_pqc_key(self) -> bytes:
        """Generate PQC-derived key."""
        from ai_engine.crypto.mlkem import MlKemEngine, MlKemSecurityLevel

        engine = MlKemEngine(MlKemSecurityLevel.MLKEM_768)
        keypair = await engine.generate_keypair()
        ct, ss = await engine.encapsulate(keypair.public_key)
        return ss.data

    def derive_session_key(
        self,
        plc_id: str,
        session_id: bytes,
        purpose: str,
    ) -> bytes:
        """Derive session-specific key."""
        master_key = self._plc_keys.get(plc_id)
        if not master_key:
            raise ValueError(f"Unknown PLC: {plc_id}")

        return hashlib.sha256(
            master_key + session_id + purpose.encode()
        ).digest()

    async def generate_signing_keypair(
        self,
        vendor: PLCVendor,
    ) -> Tuple[bytes, bytes]:
        """Generate firmware signing keypair."""
        if self.use_pqc:
            from ai_engine.crypto.dilithium import DilithiumEngine, DilithiumSecurityLevel

            engine = DilithiumEngine(DilithiumSecurityLevel.LEVEL3)
            keypair = await engine.generate_keypair()
            public_key = keypair.public_key.data
            private_key = keypair.private_key.data
        else:
            # Classical fallback
            public_key = secrets.token_bytes(32)
            private_key = secrets.token_bytes(64)

        self._signing_keys[vendor] = (public_key, private_key)
        return public_key, private_key

    def get_verification_key(self, vendor: PLCVendor) -> Optional[bytes]:
        """Get firmware verification public key."""
        keys = self._signing_keys.get(vendor)
        return keys[0] if keys else None


class FirmwareValidator:
    """
    Firmware validation for PLCs.

    Validates:
    - Firmware authenticity (signature)
    - Firmware integrity (hash)
    - Version compatibility
    """

    def __init__(self, key_manager: PLCKeyManager):
        self.key_manager = key_manager
        self._approved_versions: Dict[str, List[str]] = {}  # model -> approved versions

        logger.info("Firmware validator initialized")

    async def validate_firmware(
        self,
        manifest: FirmwareManifest,
        firmware_data: bytes,
    ) -> Tuple[bool, str]:
        """
        Validate firmware update.

        Returns:
            Tuple of (is_valid, reason)
        """
        # Verify hash
        computed_hash = hashlib.sha256(firmware_data).digest()
        if not secrets.compare_digest(computed_hash, manifest.sha256_hash):
            PLC_FIRMWARE_VALIDATIONS.labels(result="hash_mismatch").inc()
            return False, "Firmware hash mismatch"

        # Verify size
        if len(firmware_data) != manifest.size_bytes:
            PLC_FIRMWARE_VALIDATIONS.labels(result="size_mismatch").inc()
            return False, f"Size mismatch: expected {manifest.size_bytes}, got {len(firmware_data)}"

        # Verify signature
        if not await self._verify_signature(manifest):
            PLC_FIRMWARE_VALIDATIONS.labels(result="signature_invalid").inc()
            return False, "Invalid firmware signature"

        # Check approved versions
        model_key = f"{manifest.vendor.value}:{manifest.model}"
        if model_key in self._approved_versions:
            if manifest.version not in self._approved_versions[model_key]:
                PLC_FIRMWARE_VALIDATIONS.labels(result="version_not_approved").inc()
                return False, f"Version {manifest.version} not approved"

        PLC_FIRMWARE_VALIDATIONS.labels(result="valid").inc()
        logger.info(
            f"Firmware validated: {manifest.vendor.value}/{manifest.model} "
            f"v{manifest.version}"
        )

        return True, "Firmware valid"

    async def _verify_signature(self, manifest: FirmwareManifest) -> bool:
        """Verify firmware signature."""
        public_key = self.key_manager.get_verification_key(manifest.vendor)
        if not public_key:
            logger.warning(f"No verification key for vendor: {manifest.vendor.value}")
            return False

        # Build signed data
        signed_data = (
            manifest.version.encode() +
            manifest.model.encode() +
            manifest.size_bytes.to_bytes(8, 'big') +
            manifest.sha256_hash
        )

        try:
            from ai_engine.crypto.dilithium import (
                DilithiumEngine, DilithiumSecurityLevel,
                DilithiumPublicKey, DilithiumSignature
            )

            engine = DilithiumEngine(DilithiumSecurityLevel.LEVEL3)
            pk = DilithiumPublicKey(DilithiumSecurityLevel.LEVEL3, public_key)
            sig = DilithiumSignature(DilithiumSecurityLevel.LEVEL3, manifest.signature)

            return await engine.verify(signed_data, sig, pk)

        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False

    async def sign_firmware(
        self,
        manifest: FirmwareManifest,
    ) -> FirmwareManifest:
        """Sign firmware manifest."""
        keys = self.key_manager._signing_keys.get(manifest.vendor)
        if not keys:
            raise ValueError(f"No signing key for vendor: {manifest.vendor.value}")

        _, private_key = keys

        signed_data = (
            manifest.version.encode() +
            manifest.model.encode() +
            manifest.size_bytes.to_bytes(8, 'big') +
            manifest.sha256_hash
        )

        try:
            from ai_engine.crypto.dilithium import (
                DilithiumEngine, DilithiumSecurityLevel, DilithiumPrivateKey
            )

            engine = DilithiumEngine(DilithiumSecurityLevel.LEVEL3)
            sk = DilithiumPrivateKey(DilithiumSecurityLevel.LEVEL3, private_key)
            signature = await engine.sign(signed_data, sk)
            manifest.signature = signature.data

        except Exception as e:
            logger.error(f"Signing error: {e}")
            raise

        return manifest

    def approve_version(
        self,
        vendor: PLCVendor,
        model: str,
        version: str,
    ) -> None:
        """Add version to approved list."""
        key = f"{vendor.value}:{model}"
        if key not in self._approved_versions:
            self._approved_versions[key] = []
        self._approved_versions[key].append(version)


class PLCAuthenticator:
    """
    Main PLC authentication engine.

    Provides:
    - Session establishment
    - Command authentication
    - Program validation
    """

    def __init__(self, use_pqc: bool = True):
        self.key_manager = PLCKeyManager(use_pqc=use_pqc)
        self.firmware_validator = FirmwareValidator(self.key_manager)

        self._contexts: Dict[str, PLCSecurityContext] = {}
        self._plc_profiles: Dict[str, PLCProfile] = {}

        logger.info(f"PLC authenticator initialized: PQC={use_pqc}")

    async def register_plc(
        self,
        profile: PLCProfile,
    ) -> bytes:
        """
        Register PLC and initialize security.

        Returns master key for secure provisioning.
        """
        master_key = await self.key_manager.initialize_plc(
            profile.device_id,
            profile.vendor,
        )

        self._plc_profiles[profile.device_id] = profile

        PLC_AUTH_OPS.labels(
            vendor=profile.vendor.value,
            operation="register"
        ).inc()

        logger.info(f"Registered PLC: {profile.device_id} ({profile.vendor.value})")
        return master_key

    async def establish_session(
        self,
        plc_id: str,
        client_id: str,
    ) -> PLCSecurityContext:
        """
        Establish authenticated session with PLC.

        In production, would involve challenge-response.
        """
        profile = self._plc_profiles.get(plc_id)
        if not profile:
            raise ValueError(f"Unknown PLC: {plc_id}")

        session_id = secrets.token_bytes(16)

        auth_key = self.key_manager.derive_session_key(
            plc_id,
            session_id,
            "authentication",
        )

        program_key = self.key_manager.derive_session_key(
            plc_id,
            session_id,
            "program_validation",
        )

        context = PLCSecurityContext(
            session_id=session_id,
            plc_id=plc_id,
            client_id=client_id,
            vendor=profile.vendor,
            authentication_key=auth_key,
            program_validation_key=program_key,
        )

        self._contexts[session_id.hex()] = context
        PLC_ACTIVE_SESSIONS.set(len(self._contexts))

        PLC_AUTH_OPS.labels(
            vendor=profile.vendor.value,
            operation="establish_session"
        ).inc()

        logger.info(f"Session established: {client_id} -> {plc_id}")
        return context

    async def authenticate_command(
        self,
        context: PLCSecurityContext,
        command: bytes,
    ) -> bytes:
        """
        Authenticate command for PLC.

        Returns command with authentication data appended.
        """
        if not context.is_valid():
            raise ValueError("Session expired")

        context.authenticated_commands += 1
        context.last_activity = time.time()

        # Build authenticated command
        timestamp = int(time.time() * 1000).to_bytes(8, 'big')
        sequence = context.authenticated_commands.to_bytes(4, 'big')

        mac_input = context.session_id + timestamp + sequence + command
        mac = hmac.new(
            context.authentication_key,
            mac_input,
            hashlib.sha256
        ).digest()[:16]

        PLC_AUTH_OPS.labels(
            vendor=context.vendor.value,
            operation="authenticate_command"
        ).inc()

        return timestamp + sequence + command + mac

    async def verify_command(
        self,
        context: PLCSecurityContext,
        authenticated_command: bytes,
    ) -> Optional[bytes]:
        """
        Verify authenticated command.

        Returns original command if valid, None otherwise.
        """
        if len(authenticated_command) < 28:  # 8 + 4 + min 0 + 16
            return None

        timestamp = authenticated_command[:8]
        sequence = authenticated_command[8:12]
        command = authenticated_command[12:-16]
        mac = authenticated_command[-16:]

        # Verify timestamp freshness (5 minute window)
        cmd_time = int.from_bytes(timestamp, 'big')
        current_time = int(time.time() * 1000)

        if abs(current_time - cmd_time) > 300000:
            logger.warning("Command timestamp outside acceptable window")
            return None

        # Verify MAC
        mac_input = context.session_id + timestamp + sequence + command
        expected_mac = hmac.new(
            context.authentication_key,
            mac_input,
            hashlib.sha256
        ).digest()[:16]

        if not secrets.compare_digest(mac, expected_mac):
            logger.warning("Command MAC verification failed")
            return None

        return command

    async def validate_program_block(
        self,
        context: PLCSecurityContext,
        block: ProgramBlock,
        block_content: bytes,
    ) -> Tuple[bool, str]:
        """
        Validate PLC program block.

        Returns (is_valid, reason).
        """
        # Verify content hash
        computed_hash = hashlib.sha256(block_content).digest()
        if not secrets.compare_digest(computed_hash, block.content_hash):
            return False, "Content hash mismatch"

        # Verify signature if present
        if block.signature:
            signed_data = (
                block.block_type.encode() +
                block.block_number.to_bytes(4, 'big') +
                block.content_hash
            )

            expected_sig = hmac.new(
                context.program_validation_key,
                signed_data,
                hashlib.sha256
            ).digest()

            if not secrets.compare_digest(block.signature, expected_sig):
                return False, "Invalid block signature"

        return True, "Block valid"

    def get_context(self, session_id: str) -> Optional[PLCSecurityContext]:
        """Get session context."""
        return self._contexts.get(session_id)

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        expired = [
            sid for sid, ctx in self._contexts.items()
            if not ctx.is_valid()
        ]

        for sid in expired:
            del self._contexts[sid]

        if expired:
            PLC_ACTIVE_SESSIONS.set(len(self._contexts))
            logger.info(f"Cleaned up {len(expired)} expired PLC sessions")

        return len(expired)


def create_siemens_profile(
    device_id: str,
    model: str = "S7-1500",
    firmware_version: str = "2.9",
) -> PLCProfile:
    """Create profile for Siemens S7 PLC."""
    return PLCProfile(
        device_id=device_id,
        vendor=PLCVendor.SIEMENS,
        model=model,
        firmware_version=firmware_version,
        cpu_mhz=400,
        memory_kb=4096,
        scan_cycle_ms=1.0,
        capabilities={
            PLCCapability.BASIC_AUTH,
            PLCCapability.CERTIFICATE,
            PLCCapability.SECURE_BOOT,
            PLCCapability.FIRMWARE_SIGN,
            PLCCapability.PQC_CAPABLE,
        },
        security_mode=PLCSecurityMode.FULL,
    )


def create_allen_bradley_profile(
    device_id: str,
    model: str = "ControlLogix",
    firmware_version: str = "33",
) -> PLCProfile:
    """Create profile for Allen-Bradley PLC."""
    return PLCProfile(
        device_id=device_id,
        vendor=PLCVendor.ALLEN_BRADLEY,
        model=model,
        firmware_version=firmware_version,
        cpu_mhz=300,
        memory_kb=2048,
        scan_cycle_ms=2.0,
        capabilities={
            PLCCapability.BASIC_AUTH,
            PLCCapability.CERTIFICATE,
            PLCCapability.FIRMWARE_SIGN,
        },
        security_mode=PLCSecurityMode.AUTHENTICATION,
    )
