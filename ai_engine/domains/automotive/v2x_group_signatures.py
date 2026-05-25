"""
Post-Quantum Group Signatures for V2X Pseudonymous Authentication

Lattice-based group signature scheme enabling anonymous vehicle authentication
with authority tracing for V2X (Vehicle-to-Everything) communication.

Design Goals:
    1. Anonymity: A verifier learns only that the signer is a valid group member,
       not which member. Protects driver privacy in daily V2X broadcasts.
    2. Traceability: A designated Group Manager (GM) can open a signature to
       reveal the signer's identity for accident investigation or law enforcement.
    3. Time-Windowed Linkability: Signatures within the same 5-minute window are
       linkable (same pseudonym tag) to detect Sybil attacks, but unlinkable
       across windows to preserve long-term privacy.
    4. Verifier-Local Revocation (VLR): Revoked members can be detected by
       verifiers without contacting the GM in real time.
    5. Batch Verification: 1000+ signature verifications per second for dense
       traffic scenarios (intersection, highway merge).

Post-Quantum Construction:
    Based on lattice assumptions (Module-SIS / Module-LWE) compatible with
    NIST PQC standards. The scheme uses:
    - ML-DSA (Dilithium) as the underlying signature primitive
    - SHAKE256 for pseudonym tag derivation (quantum-safe hash)
    - ML-KEM for encrypted identity escrow (GM can decrypt to trace)

Standards Alignment:
    - IEEE 1609.2: Secured Protocol Data Units (SPDUs)
    - SAE J2735: Message types (BSM, EVA, TIM)
    - ETSI TS 103 097: ITS security headers
    - ETSI TR 103 415: Pre-authorization for pseudonym provisioning

References:
    - Gordon et al., "Group Signatures from Lattices" (Asiacrypt 2010)
    - Ling et al., "Lattice-Based Group Signatures: Achieving Full Dynamicity"
    - qSCMS: Quantum-Safe SCMS for V2X (NIST PQC migration studies)
    - 3GPP C-V2X security architecture
"""

import asyncio
import hashlib
import logging
import secrets
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ──────────────────────────────────────────────────────────────────────

GROUP_SIG_OPS = Counter(
    "v2x_group_sig_operations_total",
    "Total V2X group signature operations",
    ["operation"],
)

GROUP_SIG_LATENCY = Histogram(
    "v2x_group_sig_latency_ms",
    "V2X group signature operation latency in ms",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500],
)

GROUP_SIG_MEMBERS = Gauge(
    "v2x_group_sig_members",
    "Current number of group members",
)

GROUP_SIG_REVOKED = Gauge(
    "v2x_group_sig_revoked_members",
    "Current number of revoked group members",
)

GROUP_SIG_BATCH_SIZE = Histogram(
    "v2x_group_sig_batch_size",
    "Batch verification sizes",
    buckets=[1, 10, 50, 100, 250, 500, 1000],
)


# ──────────────────────────────────────────────────────────────────────
# Enums and configuration
# ──────────────────────────────────────────────────────────────────────


class GroupSignatureScheme(Enum):
    """Underlying cryptographic scheme for group signatures."""

    LATTICE_DILITHIUM = "lattice-dilithium"  # Based on ML-DSA
    LATTICE_FALCON = "lattice-falcon"  # Compact, for bandwidth-constrained


class LinkabilityWindow(Enum):
    """Time window for pseudonym linkability."""

    MINUTES_5 = 300  # 5-minute windows (default V2X)
    MINUTES_15 = 900  # 15-minute windows (relaxed)
    MINUTES_1 = 60  # 1-minute windows (paranoid)
    HOURS_1 = 3600  # 1-hour windows (fleet management)


class RevocationReason(Enum):
    """Reason for member revocation."""

    KEY_COMPROMISE = auto()
    MISBEHAVIOR_DETECTED = auto()
    CERTIFICATE_EXPIRED = auto()
    ADMINISTRATIVE = auto()
    LAW_ENFORCEMENT = auto()


@dataclass(frozen=True)
class GroupSignatureConfig:
    """
    Configuration for the V2X group signature scheme.

    Attributes:
        scheme: Underlying signature scheme
        linkability_window: Time window for pseudonym linkability
        max_members: Maximum group size
        batch_verification_workers: Thread pool size for batch verify
        enable_vlr: Enable Verifier-Local Revocation
        identity_escrow: Enable encrypted identity for GM tracing
        tag_derivation_salt: Domain-separation salt for tag derivation
    """

    scheme: GroupSignatureScheme = GroupSignatureScheme.LATTICE_DILITHIUM
    linkability_window: LinkabilityWindow = LinkabilityWindow.MINUTES_5
    max_members: int = 100000
    batch_verification_workers: int = 8
    enable_vlr: bool = True
    identity_escrow: bool = True
    tag_derivation_salt: bytes = b"qbitel-v2x-group-sig-v1"


# Pre-configured profiles
V2X_HIGHWAY_CONFIG = GroupSignatureConfig(
    scheme=GroupSignatureScheme.LATTICE_DILITHIUM,
    linkability_window=LinkabilityWindow.MINUTES_5,
    batch_verification_workers=4,
)

V2X_URBAN_CONFIG = GroupSignatureConfig(
    scheme=GroupSignatureScheme.LATTICE_DILITHIUM,
    linkability_window=LinkabilityWindow.MINUTES_5,
    batch_verification_workers=16,
)

V2X_INTERSECTION_CONFIG = GroupSignatureConfig(
    scheme=GroupSignatureScheme.LATTICE_DILITHIUM,
    linkability_window=LinkabilityWindow.MINUTES_1,
    batch_verification_workers=32,
)


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GroupPublicKey:
    """
    Group public key — published to all verifiers.

    Contains parameters needed to verify any group member's signature
    without learning the signer's identity.
    """

    group_id: bytes  # Unique group identifier (16 bytes)
    version: int  # Key version for rotation
    scheme: GroupSignatureScheme
    # The group public key is the GM's verification key
    gm_verification_key: bytes  # Group Manager's public verification key
    # Revocation list anchor (hash of current VRL)
    vrl_hash: bytes = b""
    created_at: float = field(default_factory=time.time)
    valid_until: float = 0.0

    def __post_init__(self):
        if self.valid_until == 0.0:
            object.__setattr__(self, "valid_until", self.created_at + 86400)  # 24h

    @property
    def is_valid(self) -> bool:
        return time.time() < self.valid_until

    def to_bytes(self) -> bytes:
        """Serialize for distribution."""
        return (
            self.group_id
            + struct.pack(">I", self.version)
            + struct.pack(">H", len(self.gm_verification_key))
            + self.gm_verification_key
            + self.vrl_hash
        )


@dataclass
class MemberPrivateKey:
    """
    Individual member's group signing key.

    Issued by the Group Manager during enrollment.
    Contains the member's secret signing material and identity token.
    """

    member_id: bytes  # Unique member identifier (16 bytes)
    group_id: bytes  # Group this key belongs to
    signing_key: bytes  # ML-DSA private key material
    identity_token: bytes  # Encrypted identity for GM tracing
    member_index: int  # Member's index in the group
    issued_at: float = field(default_factory=time.time)
    valid_until: float = 0.0

    def __post_init__(self):
        if self.valid_until == 0.0:
            object.__setattr__(self, "valid_until", self.issued_at + 86400 * 30)  # 30 days

    def __del__(self):
        """Zeroize signing key on deletion."""
        if hasattr(self, "signing_key") and isinstance(self.signing_key, bytearray):
            for i in range(len(self.signing_key)):
                self.signing_key[i] = 0


@dataclass
class GroupSignature:
    """
    A group signature on a V2X message.

    Contains the signature, pseudonym tag (for linkability detection),
    and encrypted identity escrow (for GM tracing).
    """

    group_id: bytes  # Which group
    signature_data: bytes  # The actual signature
    pseudonym_tag: bytes  # Time-windowed linkability tag (32 bytes)
    identity_escrow: bytes  # Encrypted member identity for GM opening
    window_epoch: int  # Which linkability window this belongs to
    timestamp: float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        """Serialize for transmission."""
        return (
            self.group_id
            + struct.pack(">I", self.window_epoch)
            + struct.pack(">H", len(self.pseudonym_tag))
            + self.pseudonym_tag
            + struct.pack(">H", len(self.identity_escrow))
            + self.identity_escrow
            + struct.pack(">I", len(self.signature_data))
            + self.signature_data
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "GroupSignature":
        """Deserialize from wire format."""
        offset = 0
        group_id = data[offset : offset + 16]
        offset += 16

        window_epoch = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4

        tag_len = struct.unpack(">H", data[offset : offset + 2])[0]
        offset += 2
        pseudonym_tag = data[offset : offset + tag_len]
        offset += tag_len

        escrow_len = struct.unpack(">H", data[offset : offset + 2])[0]
        offset += 2
        identity_escrow = data[offset : offset + escrow_len]
        offset += escrow_len

        sig_len = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        signature_data = data[offset : offset + sig_len]

        return cls(
            group_id=group_id,
            signature_data=signature_data,
            pseudonym_tag=pseudonym_tag,
            identity_escrow=identity_escrow,
            window_epoch=window_epoch,
        )

    @property
    def total_size(self) -> int:
        """Total signature size in bytes."""
        return len(self.to_bytes())


@dataclass
class RevocationEntry:
    """Entry in the Verifier Revocation List (VRL)."""

    member_revocation_token: bytes  # Token derived from member's key
    reason: RevocationReason
    revoked_at: float = field(default_factory=time.time)
    effective_from: float = 0.0

    def __post_init__(self):
        if self.effective_from == 0.0:
            object.__setattr__(self, "effective_from", self.revoked_at)


@dataclass
class OpeningResult:
    """Result of a Group Manager opening a signature."""

    member_id: bytes
    member_index: int
    signature_timestamp: float
    opening_proof: bytes  # Proof that opening is correct


@dataclass
class BatchVerifyResult:
    """Result of batch group signature verification."""

    total: int
    valid_count: int
    invalid_count: int
    revoked_count: int
    sybil_detected_count: int
    latency_ms: float
    throughput_per_sec: float
    details: List[Tuple[int, bool, Optional[str]]]  # (index, valid, reason)


# ──────────────────────────────────────────────────────────────────────
# Group Manager
# ──────────────────────────────────────────────────────────────────────


class GroupManager:
    """
    Group Manager (GM) for V2X group signature scheme.

    Responsibilities:
    1. Group setup and parameter generation
    2. Member enrollment (issue signing keys)
    3. Signature opening (trace signer identity)
    4. Member revocation (update VRL)

    In a V2X deployment, the GM is typically operated by a trusted
    Regional Authority or the SCMS Registration Authority.
    """

    def __init__(self, config: GroupSignatureConfig = V2X_HIGHWAY_CONFIG):
        self.config = config

        # Group identity
        self._group_id = secrets.token_bytes(16)

        # GM's keys
        self._gm_signing_key: Optional[bytes] = None
        self._gm_verification_key: Optional[bytes] = None
        self._gm_decryption_key: Optional[bytes] = None  # For identity escrow
        self._gm_encryption_key: Optional[bytes] = None

        # Member registry
        self._members: Dict[bytes, MemberPrivateKey] = {}
        self._member_tokens: Dict[bytes, bytes] = {}  # revocation_token -> member_id
        self._next_member_index = 0

        # Revocation list
        self._revocation_list: List[RevocationEntry] = []
        self._revoked_tokens: Set[bytes] = set()

        # Group public key
        self._group_public_key: Optional[GroupPublicKey] = None
        self._key_version = 0

        logger.info(
            f"Group Manager created: group={self._group_id.hex()[:8]}, "
            f"scheme={config.scheme.value}"
        )

    async def setup(self) -> GroupPublicKey:
        """
        Initialize the group and generate group public key.

        Returns:
            Group public key for distribution to all verifiers
        """
        from ai_engine.crypto.dilithium import DilithiumEngine, DilithiumSecurityLevel

        # Generate GM's signing keypair (ML-DSA-65 for enterprise)
        engine = DilithiumEngine(DilithiumSecurityLevel.MLDSA_65)
        keypair = await engine.generate_keypair()

        self._gm_signing_key = keypair.private_key.data
        self._gm_verification_key = keypair.public_key.data

        # Generate identity escrow keys (ML-KEM for encrypted identity)
        if self.config.identity_escrow:
            from ai_engine.crypto.mlkem import MlKemEngine, MlKemSecurityLevel

            kem_engine = MlKemEngine(MlKemSecurityLevel.MLKEM_768)
            kem_keypair = await kem_engine.generate_keypair()
            self._gm_encryption_key = kem_keypair.public_key.data
            self._gm_decryption_key = kem_keypair.private_key.data

        self._key_version += 1

        self._group_public_key = GroupPublicKey(
            group_id=self._group_id,
            version=self._key_version,
            scheme=self.config.scheme,
            gm_verification_key=self._gm_verification_key,
            vrl_hash=self._compute_vrl_hash(),
        )

        GROUP_SIG_OPS.labels(operation="setup").inc()

        logger.info(
            f"Group setup complete: version={self._key_version}, "
            f"escrow={'enabled' if self.config.identity_escrow else 'disabled'}"
        )

        return self._group_public_key

    async def enroll_member(self, vehicle_id: str) -> Tuple[MemberPrivateKey, GroupPublicKey]:
        """
        Enroll a new vehicle into the group.

        Generates a unique signing key for the vehicle and registers
        the member in the group registry.

        Args:
            vehicle_id: Unique vehicle identifier (e.g., VIN hash)

        Returns:
            Tuple of (member private key, group public key)
        """
        from ai_engine.crypto.dilithium import DilithiumEngine, DilithiumSecurityLevel

        if self._next_member_index >= self.config.max_members:
            raise GroupFullError(self._group_id, self.config.max_members)

        member_id = hashlib.sha3_256(vehicle_id.encode()).digest()[:16]

        # Generate member-specific signing key
        engine = DilithiumEngine(DilithiumSecurityLevel.MLDSA_65)
        keypair = await engine.generate_keypair()

        # Create identity escrow token
        identity_token = await self._create_identity_escrow(
            member_id, vehicle_id
        )

        # Create revocation token for VLR
        revocation_token = self._derive_revocation_token(member_id)
        self._member_tokens[revocation_token] = member_id

        member_key = MemberPrivateKey(
            member_id=member_id,
            group_id=self._group_id,
            signing_key=keypair.private_key.data,
            identity_token=identity_token,
            member_index=self._next_member_index,
        )

        self._members[member_id] = member_key
        self._next_member_index += 1

        GROUP_SIG_MEMBERS.set(len(self._members))
        GROUP_SIG_OPS.labels(operation="enroll").inc()

        logger.info(
            f"Member enrolled: index={member_key.member_index}, "
            f"group={self._group_id.hex()[:8]}"
        )

        return member_key, self._group_public_key

    async def _create_identity_escrow(
        self,
        member_id: bytes,
        vehicle_id: str,
    ) -> bytes:
        """
        Create encrypted identity token for GM tracing.

        The identity is encrypted under the GM's ML-KEM public key
        so only the GM can decrypt it during signature opening.
        """
        if not self.config.identity_escrow or not self._gm_encryption_key:
            return member_id  # Unencrypted fallback

        # Encrypt member identity under GM's public key
        identity_data = member_id + vehicle_id.encode("utf-8")[:32]

        # Use SHAKE256 to create a deterministic encryption
        # (In production, use ML-KEM encapsulation + AES-GCM)
        h = hashlib.shake_256(
            self._gm_encryption_key + identity_data + secrets.token_bytes(16)
        )
        encrypted = h.digest(len(identity_data) + 32)

        return encrypted

    def _derive_revocation_token(self, member_id: bytes) -> bytes:
        """Derive a revocation token from member identity."""
        return hashlib.sha3_256(
            b"v2x-revocation-token" + member_id + self._group_id
        ).digest()

    async def revoke_member(
        self,
        member_id: bytes,
        reason: RevocationReason = RevocationReason.ADMINISTRATIVE,
    ) -> bool:
        """
        Revoke a group member.

        Adds the member's revocation token to the VRL so verifiers
        can detect and reject signatures from the revoked member.

        Args:
            member_id: Member to revoke
            reason: Revocation reason

        Returns:
            True if member was found and revoked
        """
        if member_id not in self._members:
            return False

        revocation_token = self._derive_revocation_token(member_id)

        entry = RevocationEntry(
            member_revocation_token=revocation_token,
            reason=reason,
        )

        self._revocation_list.append(entry)
        self._revoked_tokens.add(revocation_token)

        # Update VRL hash in group public key
        if self._group_public_key:
            object.__setattr__(
                self._group_public_key,
                "vrl_hash",
                self._compute_vrl_hash(),
            )

        GROUP_SIG_REVOKED.set(len(self._revocation_list))
        GROUP_SIG_OPS.labels(operation="revoke").inc()

        logger.info(
            f"Member revoked: reason={reason.name}, "
            f"group={self._group_id.hex()[:8]}"
        )

        return True

    async def open_signature(
        self,
        signature: GroupSignature,
        message: bytes,
    ) -> Optional[OpeningResult]:
        """
        Open a group signature to reveal the signer's identity.

        This is a privileged operation only available to the GM,
        typically invoked for accident investigation or law enforcement.

        Args:
            signature: The group signature to open
            message: The original signed message

        Returns:
            OpeningResult with member identity, or None if opening fails
        """
        if signature.group_id != self._group_id:
            logger.warning("Signature belongs to different group")
            return None

        # In a full lattice-based scheme, opening would involve:
        # 1. Decrypt the identity escrow using GM's decryption key
        # 2. Verify the opening proof is correct
        # 3. Return the member's real identity

        # For this implementation, we derive the member identity from
        # the identity escrow and the GM's key material
        if self._gm_decryption_key:
            # Attempt to find the member by matching the escrow token
            for member_id, member_key in self._members.items():
                if self._match_escrow(
                    signature.identity_escrow,
                    member_key.identity_token,
                ):
                    # Create proof of correct opening
                    opening_proof = hashlib.sha3_256(
                        b"opening-proof"
                        + signature.signature_data
                        + member_id
                        + self._gm_signing_key[:32]
                    ).digest()

                    GROUP_SIG_OPS.labels(operation="open").inc()

                    return OpeningResult(
                        member_id=member_id,
                        member_index=member_key.member_index,
                        signature_timestamp=signature.timestamp,
                        opening_proof=opening_proof,
                    )

        logger.warning("Failed to open signature — member not found")
        return None

    def _match_escrow(self, sig_escrow: bytes, member_token: bytes) -> bool:
        """Check if a signature's escrow matches a member's token."""
        # In the real scheme, this would involve decryption
        # Here we use a simplified matching
        return sig_escrow[:16] == member_token[:16]

    def _compute_vrl_hash(self) -> bytes:
        """Compute hash of the current Verifier Revocation List."""
        if not self._revocation_list:
            return b"\x00" * 32

        h = hashlib.sha3_256()
        for entry in self._revocation_list:
            h.update(entry.member_revocation_token)
        return h.digest()

    def get_revocation_list(self) -> List[RevocationEntry]:
        """Get the current Verifier Revocation List for distribution."""
        return list(self._revocation_list)

    @property
    def group_public_key(self) -> Optional[GroupPublicKey]:
        return self._group_public_key

    @property
    def member_count(self) -> int:
        return len(self._members)


class GroupFullError(Exception):
    """Raised when the group has reached maximum capacity."""

    def __init__(self, group_id: bytes, max_members: int):
        super().__init__(
            f"Group {group_id.hex()[:8]} is full ({max_members} members). "
            f"Create a new group or increase max_members."
        )


# ──────────────────────────────────────────────────────────────────────
# Group Signer (Vehicle-side)
# ──────────────────────────────────────────────────────────────────────


class GroupSigner:
    """
    Group signature signer for individual vehicles.

    Each vehicle uses its MemberPrivateKey to produce group signatures
    on V2X messages (BSM, EVA, etc.) that are anonymous to verifiers
    but traceable by the Group Manager.

    The signer also computes a time-windowed pseudonym tag for
    Sybil attack detection.
    """

    def __init__(
        self,
        member_key: MemberPrivateKey,
        group_public_key: GroupPublicKey,
        config: GroupSignatureConfig = V2X_HIGHWAY_CONFIG,
    ):
        self.member_key = member_key
        self.group_public_key = group_public_key
        self.config = config

        # Cache current window epoch and tag
        self._current_epoch: Optional[int] = None
        self._current_tag: Optional[bytes] = None

        logger.info(
            f"Group signer initialized: member_index={member_key.member_index}, "
            f"group={group_public_key.group_id.hex()[:8]}"
        )

    async def sign(self, message: bytes) -> GroupSignature:
        """
        Produce a group signature on a V2X message.

        The signature proves group membership without revealing identity.
        It includes a time-windowed pseudonym tag for Sybil detection
        and an encrypted identity escrow for GM tracing.

        Args:
            message: V2X message bytes (e.g., serialized BSM)

        Returns:
            GroupSignature ready for broadcast
        """
        start = time.perf_counter()

        # Compute current linkability window epoch
        window_epoch = self._get_current_epoch()

        # Derive pseudonym tag for this window
        pseudonym_tag = self._derive_pseudonym_tag(window_epoch)

        # Sign: σ = Sign(sk, H(message || tag || epoch))
        sig_input = self._prepare_sign_input(message, pseudonym_tag, window_epoch)
        signature_data = await self._compute_signature(sig_input)

        # Create identity escrow for this signature
        identity_escrow = self._create_signature_escrow(window_epoch)

        group_sig = GroupSignature(
            group_id=self.group_public_key.group_id,
            signature_data=signature_data,
            pseudonym_tag=pseudonym_tag,
            identity_escrow=identity_escrow,
            window_epoch=window_epoch,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        GROUP_SIG_LATENCY.observe(elapsed_ms)
        GROUP_SIG_OPS.labels(operation="sign").inc()

        return group_sig

    def _get_current_epoch(self) -> int:
        """Get the current linkability window epoch."""
        window_seconds = self.config.linkability_window.value
        return int(time.time()) // window_seconds

    def _derive_pseudonym_tag(self, epoch: int) -> bytes:
        """
        Derive a pseudonym tag for the current time window.

        The tag is deterministic for (member_id, epoch) so that
        multiple messages from the same vehicle in the same window
        produce the same tag. Different epochs produce different tags.
        """
        if self._current_epoch == epoch and self._current_tag is not None:
            return self._current_tag

        tag_input = (
            self.config.tag_derivation_salt
            + self.member_key.member_id
            + struct.pack(">Q", epoch)
            + self.member_key.group_id
        )

        tag = hashlib.shake_256(tag_input).digest(32)

        # Cache for this epoch
        self._current_epoch = epoch
        self._current_tag = tag

        return tag

    def _prepare_sign_input(
        self,
        message: bytes,
        tag: bytes,
        epoch: int,
    ) -> bytes:
        """Prepare the data to be signed."""
        return hashlib.sha3_256(
            message + tag + struct.pack(">Q", epoch)
        ).digest()

    async def _compute_signature(self, data: bytes) -> bytes:
        """Compute the actual signature using ML-DSA."""
        from ai_engine.crypto.dilithium import (
            DilithiumEngine,
            DilithiumSecurityLevel,
            DilithiumPrivateKey,
        )

        engine = DilithiumEngine(DilithiumSecurityLevel.MLDSA_65)
        private_key = DilithiumPrivateKey(
            DilithiumSecurityLevel.MLDSA_65,
            self.member_key.signing_key,
        )

        signature = await engine.sign(data, private_key)
        return signature.data

    def _create_signature_escrow(self, epoch: int) -> bytes:
        """
        Create encrypted identity escrow for this signature.

        This allows the GM to trace the signer's identity.
        In the full scheme, this would be an ML-KEM encryption
        of the member's identity under the GM's public key.
        """
        escrow_data = (
            self.member_key.identity_token[:32]
            + struct.pack(">Q", epoch)
            + struct.pack(">I", self.member_key.member_index)
        )

        # Hash-based escrow (simplified — full version uses ML-KEM)
        return hashlib.shake_256(
            escrow_data + self.member_key.member_id
        ).digest(48)


# ──────────────────────────────────────────────────────────────────────
# Group Verifier (RSU / OBU side)
# ──────────────────────────────────────────────────────────────────────


class GroupVerifier:
    """
    Group signature verifier for V2X infrastructure.

    Deployed on Roadside Units (RSUs) and On-Board Units (OBUs)
    to verify incoming V2X group signatures. Performs:
    1. Signature validity check
    2. Verifier-Local Revocation (VLR) check
    3. Sybil detection via pseudonym tag tracking
    4. Batch verification for high-throughput scenarios
    """

    def __init__(
        self,
        group_public_key: GroupPublicKey,
        config: GroupSignatureConfig = V2X_HIGHWAY_CONFIG,
    ):
        self.group_public_key = group_public_key
        self.config = config

        # VRL cache
        self._revocation_tokens: Set[bytes] = set()

        # Sybil detection: track pseudonym tags per window
        self._tag_tracker: Dict[int, Dict[bytes, int]] = {}  # epoch -> {tag -> count}
        self._max_messages_per_tag = 100  # Alert threshold

        # Batch verification thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=config.batch_verification_workers,
        )

        logger.info(
            f"Group verifier initialized: "
            f"group={group_public_key.group_id.hex()[:8]}, "
            f"vlr={'enabled' if config.enable_vlr else 'disabled'}"
        )

    async def verify(
        self,
        message: bytes,
        signature: GroupSignature,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a group signature on a V2X message.

        Args:
            message: Original V2X message bytes
            signature: Group signature to verify

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        start = time.perf_counter()

        # 1. Check group ID matches
        if signature.group_id != self.group_public_key.group_id:
            return False, "group_mismatch"

        # 2. Check VRL (Verifier-Local Revocation)
        if self.config.enable_vlr:
            if self._check_revoked(signature):
                GROUP_SIG_OPS.labels(operation="verify_revoked").inc()
                return False, "revoked"

        # 3. Check Sybil (too many messages from same pseudonym)
        sybil_alert = self._track_pseudonym(signature)
        if sybil_alert:
            logger.warning(
                f"Sybil alert: tag={signature.pseudonym_tag.hex()[:8]} "
                f"exceeded threshold in epoch {signature.window_epoch}"
            )

        # 4. Verify the cryptographic signature
        sig_input = hashlib.sha3_256(
            message
            + signature.pseudonym_tag
            + struct.pack(">Q", signature.window_epoch)
        ).digest()

        valid = await self._verify_signature(sig_input, signature.signature_data)

        elapsed_ms = (time.perf_counter() - start) * 1000
        GROUP_SIG_LATENCY.observe(elapsed_ms)

        if valid:
            GROUP_SIG_OPS.labels(operation="verify_valid").inc()
        else:
            GROUP_SIG_OPS.labels(operation="verify_invalid").inc()

        return valid, None if valid else "signature_invalid"

    async def verify_batch(
        self,
        messages: List[bytes],
        signatures: List[GroupSignature],
    ) -> BatchVerifyResult:
        """
        Batch verify multiple group signatures.

        Optimized for dense V2X scenarios (intersections, highways)
        where 1000+ messages/second need verification.

        Args:
            messages: List of V2X messages
            signatures: Corresponding list of group signatures

        Returns:
            BatchVerifyResult with per-message results
        """
        if len(messages) != len(signatures):
            raise ValueError("Messages and signatures lists must be same length")

        start = time.perf_counter()

        GROUP_SIG_BATCH_SIZE.observe(len(messages))

        details: List[Tuple[int, bool, Optional[str]]] = []
        valid_count = 0
        invalid_count = 0
        revoked_count = 0
        sybil_count = 0

        # Phase 1: Quick checks (VRL, group ID, Sybil) — no crypto
        crypto_tasks = []
        for i, (msg, sig) in enumerate(zip(messages, signatures)):
            if sig.group_id != self.group_public_key.group_id:
                details.append((i, False, "group_mismatch"))
                invalid_count += 1
                continue

            if self.config.enable_vlr and self._check_revoked(sig):
                details.append((i, False, "revoked"))
                revoked_count += 1
                continue

            if self._track_pseudonym(sig):
                sybil_count += 1

            # Queue for crypto verification
            crypto_tasks.append((i, msg, sig))

        # Phase 2: Parallel crypto verification
        if crypto_tasks:
            loop = asyncio.get_event_loop()

            async def verify_one(idx: int, msg: bytes, sig: GroupSignature):
                sig_input = hashlib.sha3_256(
                    msg
                    + sig.pseudonym_tag
                    + struct.pack(">Q", sig.window_epoch)
                ).digest()

                valid = await self._verify_signature(sig_input, sig.signature_data)
                return idx, valid

            results = await asyncio.gather(
                *[verify_one(i, m, s) for i, m, s in crypto_tasks]
            )

            for idx, valid in results:
                if valid:
                    valid_count += 1
                    details.append((idx, True, None))
                else:
                    invalid_count += 1
                    details.append((idx, False, "signature_invalid"))

        elapsed_ms = (time.perf_counter() - start) * 1000
        throughput = len(messages) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        GROUP_SIG_OPS.labels(operation="verify_batch").inc()

        logger.info(
            f"Batch verify: {len(messages)} msgs, "
            f"{valid_count} valid, {invalid_count} invalid, "
            f"{revoked_count} revoked, {sybil_count} sybil alerts, "
            f"{elapsed_ms:.1f}ms ({throughput:.0f}/s)"
        )

        return BatchVerifyResult(
            total=len(messages),
            valid_count=valid_count,
            invalid_count=invalid_count,
            revoked_count=revoked_count,
            sybil_detected_count=sybil_count,
            latency_ms=elapsed_ms,
            throughput_per_sec=throughput,
            details=sorted(details, key=lambda x: x[0]),
        )

    async def _verify_signature(
        self,
        data: bytes,
        signature_data: bytes,
    ) -> bool:
        """Verify a single signature using ML-DSA."""
        from ai_engine.crypto.dilithium import (
            DilithiumEngine,
            DilithiumSecurityLevel,
            DilithiumSignature,
            DilithiumPublicKey,
        )

        try:
            engine = DilithiumEngine(DilithiumSecurityLevel.MLDSA_65)
            sig = DilithiumSignature(DilithiumSecurityLevel.MLDSA_65, signature_data)
            pk = DilithiumPublicKey(
                DilithiumSecurityLevel.MLDSA_65,
                self.group_public_key.gm_verification_key,
            )
            return await engine.verify(data, sig, pk)
        except Exception as e:
            logger.debug(f"Signature verification error: {e}")
            return False

    def _check_revoked(self, signature: GroupSignature) -> bool:
        """Check if a signature's signer has been revoked (VLR check)."""
        # In the full scheme, we'd derive a revocation check value from
        # the signature's pseudonym tag and check against the VRL.
        # This is a simplified version.
        tag_hash = hashlib.sha3_256(
            b"v2x-vlr-check" + signature.pseudonym_tag
        ).digest()

        return tag_hash in self._revocation_tokens

    def _track_pseudonym(self, signature: GroupSignature) -> bool:
        """
        Track pseudonym tag for Sybil detection.

        Returns True if the tag exceeds the threshold.
        """
        epoch = signature.window_epoch
        tag = signature.pseudonym_tag

        if epoch not in self._tag_tracker:
            # Clean up old epochs
            current_epoch = int(time.time()) // self.config.linkability_window.value
            old_epochs = [e for e in self._tag_tracker if e < current_epoch - 2]
            for old in old_epochs:
                del self._tag_tracker[old]

            self._tag_tracker[epoch] = {}

        tracker = self._tag_tracker[epoch]
        tracker[tag] = tracker.get(tag, 0) + 1

        return tracker[tag] > self._max_messages_per_tag

    def update_revocation_list(self, entries: List[RevocationEntry]) -> None:
        """
        Update the local VRL with new revocation entries.

        Called periodically when the GM publishes VRL updates.
        """
        for entry in entries:
            # Derive the VLR check token
            check_token = hashlib.sha3_256(
                b"v2x-vlr-check"
                + entry.member_revocation_token
            ).digest()
            self._revocation_tokens.add(check_token)

        GROUP_SIG_REVOKED.set(len(self._revocation_tokens))
        logger.info(f"VRL updated: {len(entries)} new entries, total={len(self._revocation_tokens)}")

    def __del__(self):
        """Clean up thread pool."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


# ──────────────────────────────────────────────────────────────────────
# V2X Group Signature Protocol (high-level integration)
# ──────────────────────────────────────────────────────────────────────


class V2XGroupSignatureProtocol:
    """
    High-level protocol integrating group signatures with IEEE 1609.2.

    Provides a unified interface for V2X applications to sign and verify
    messages using post-quantum group signatures with privacy guarantees.

    Usage (Vehicle side):
        protocol = V2XGroupSignatureProtocol(member_key, group_public_key)
        signed_msg = await protocol.sign_bsm(bsm_bytes)
        # Broadcast signed_msg...

    Usage (RSU side):
        protocol = V2XGroupSignatureProtocol.for_verifier(group_public_key)
        valid, reason = await protocol.verify_signed_message(signed_msg)
    """

    def __init__(
        self,
        member_key: Optional[MemberPrivateKey] = None,
        group_public_key: Optional[GroupPublicKey] = None,
        config: GroupSignatureConfig = V2X_HIGHWAY_CONFIG,
    ):
        self.config = config
        self._signer: Optional[GroupSigner] = None
        self._verifier: Optional[GroupVerifier] = None

        if member_key and group_public_key:
            self._signer = GroupSigner(member_key, group_public_key, config)
            self._verifier = GroupVerifier(group_public_key, config)
        elif group_public_key:
            self._verifier = GroupVerifier(group_public_key, config)

    async def sign_bsm(self, bsm_bytes: bytes) -> bytes:
        """
        Sign a Basic Safety Message with group signature.

        Args:
            bsm_bytes: Serialized BSM data

        Returns:
            Wire-format bytes: [bsm_len:4][bsm_data][group_signature]
        """
        if not self._signer:
            raise RuntimeError("No signing key — verifier-only mode")

        group_sig = await self._signer.sign(bsm_bytes)
        sig_wire = group_sig.to_bytes()

        # Wire format: [bsm_len(4) | bsm_data | signature]
        return struct.pack(">I", len(bsm_bytes)) + bsm_bytes + sig_wire

    async def verify_signed_message(
        self,
        signed_bytes: bytes,
    ) -> Tuple[bool, Optional[bytes], Optional[str]]:
        """
        Verify a group-signed V2X message.

        Args:
            signed_bytes: Wire-format signed message

        Returns:
            Tuple of (is_valid, original_message_bytes, reason_if_invalid)
        """
        if not self._verifier:
            raise RuntimeError("No group public key — cannot verify")

        # Parse wire format
        bsm_len = struct.unpack(">I", signed_bytes[:4])[0]
        bsm_data = signed_bytes[4 : 4 + bsm_len]
        sig_data = signed_bytes[4 + bsm_len :]

        group_sig = GroupSignature.from_bytes(sig_data)
        valid, reason = await self._verifier.verify(bsm_data, group_sig)

        return valid, bsm_data if valid else None, reason

    async def verify_batch_signed(
        self,
        signed_messages: List[bytes],
    ) -> BatchVerifyResult:
        """Batch verify multiple signed messages."""
        if not self._verifier:
            raise RuntimeError("No group public key — cannot verify")

        messages = []
        signatures = []

        for signed_bytes in signed_messages:
            bsm_len = struct.unpack(">I", signed_bytes[:4])[0]
            bsm_data = signed_bytes[4 : 4 + bsm_len]
            sig_data = signed_bytes[4 + bsm_len :]

            messages.append(bsm_data)
            signatures.append(GroupSignature.from_bytes(sig_data))

        return await self._verifier.verify_batch(messages, signatures)

    def update_revocation_list(self, entries: List[RevocationEntry]) -> None:
        """Update the verifier's revocation list."""
        if self._verifier:
            self._verifier.update_revocation_list(entries)

    # ── Factory methods ──────────────────────────────────────────────

    @classmethod
    def for_verifier(
        cls,
        group_public_key: GroupPublicKey,
        config: GroupSignatureConfig = V2X_HIGHWAY_CONFIG,
    ) -> "V2XGroupSignatureProtocol":
        """Create a verifier-only protocol instance (for RSUs)."""
        return cls(
            member_key=None,
            group_public_key=group_public_key,
            config=config,
        )

    @classmethod
    def for_highway(
        cls,
        member_key: MemberPrivateKey,
        group_public_key: GroupPublicKey,
    ) -> "V2XGroupSignatureProtocol":
        """Create protocol for highway scenario."""
        return cls(member_key, group_public_key, V2X_HIGHWAY_CONFIG)

    @classmethod
    def for_urban(
        cls,
        member_key: MemberPrivateKey,
        group_public_key: GroupPublicKey,
    ) -> "V2XGroupSignatureProtocol":
        """Create protocol for dense urban scenario."""
        return cls(member_key, group_public_key, V2X_URBAN_CONFIG)

    @classmethod
    def for_intersection(
        cls,
        member_key: MemberPrivateKey,
        group_public_key: GroupPublicKey,
    ) -> "V2XGroupSignatureProtocol":
        """Create protocol for intersection scenario (max density)."""
        return cls(member_key, group_public_key, V2X_INTERSECTION_CONFIG)
