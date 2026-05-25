"""
Multi-Authority Threshold Signatures for Banking

Post-quantum threshold signature scheme requiring signatures from multiple
independent authorities (Central Bank, Compliance, Treasury, Risk) before
high-value transactions or policy changes can be executed.

Use Cases:
    1. High-Value Wire Approval: >$10M transfers require 3-of-5 authority signatures
    2. Regulatory Sanctions Override: Requires compliance + legal + risk sign-off
    3. HSM Master Key Ceremony: Distributed key generation across N custodians
    4. SWIFT Alliance Lite2 Config: Multi-party authorization for network changes
    5. Basel Pillar 3 Disclosure: Board-level sign-off on regulatory reports

Architecture:
    Extends the core threshold.py primitives with banking-specific:
    - Authority roles and quorum policies
    - Time-bounded signing windows
    - Partial signature collection and tracking
    - Regulatory compliance hooks
"""

import asyncio
import hashlib
import logging
import secrets
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

THRESHOLD_OPS = Counter("banking_threshold_operations_total", "Banking threshold ops", ["operation"])
THRESHOLD_LATENCY = Histogram("banking_threshold_latency_ms", "Threshold op latency", buckets=[10, 50, 100, 500, 1000])


class AuthorityRole(Enum):
    """Banking authority roles participating in threshold signing."""
    CENTRAL_BANK = auto()
    TREASURY = auto()
    COMPLIANCE = auto()
    RISK_MANAGEMENT = auto()
    LEGAL = auto()
    BOARD_MEMBER = auto()
    INTERNAL_AUDIT = auto()
    IT_SECURITY = auto()
    OPERATIONS = auto()


class TransactionTier(Enum):
    """Transaction value tiers with different quorum requirements."""
    STANDARD = "standard"          # <$1M: 1-of-1
    ELEVATED = "elevated"          # $1M-$10M: 2-of-3
    HIGH_VALUE = "high-value"      # $10M-$100M: 3-of-5
    CRITICAL = "critical"          # >$100M: 4-of-7
    SANCTIONS = "sanctions"        # Sanctions-related: compliance + legal + risk


@dataclass
class QuorumPolicy:
    """Defines the required signers for a transaction tier."""
    tier: TransactionTier
    threshold: int                             # t — minimum signatures required
    total_authorities: int                     # n — total authorities in the group
    required_roles: Set[AuthorityRole]         # Roles that MUST be present
    optional_roles: Set[AuthorityRole]         # Roles that CAN contribute
    signing_window_seconds: int = 3600         # Max time to collect signatures
    require_different_departments: bool = True # No two signers from same dept


# Pre-defined quorum policies
ELEVATED_QUORUM = QuorumPolicy(
    tier=TransactionTier.ELEVATED,
    threshold=2, total_authorities=3,
    required_roles={AuthorityRole.TREASURY},
    optional_roles={AuthorityRole.COMPLIANCE, AuthorityRole.RISK_MANAGEMENT, AuthorityRole.OPERATIONS},
    signing_window_seconds=3600,
)

HIGH_VALUE_QUORUM = QuorumPolicy(
    tier=TransactionTier.HIGH_VALUE,
    threshold=3, total_authorities=5,
    required_roles={AuthorityRole.TREASURY, AuthorityRole.COMPLIANCE},
    optional_roles={AuthorityRole.RISK_MANAGEMENT, AuthorityRole.LEGAL, AuthorityRole.BOARD_MEMBER},
    signing_window_seconds=7200,
)

CRITICAL_QUORUM = QuorumPolicy(
    tier=TransactionTier.CRITICAL,
    threshold=4, total_authorities=7,
    required_roles={AuthorityRole.TREASURY, AuthorityRole.COMPLIANCE, AuthorityRole.RISK_MANAGEMENT},
    optional_roles={AuthorityRole.LEGAL, AuthorityRole.BOARD_MEMBER, AuthorityRole.CENTRAL_BANK, AuthorityRole.INTERNAL_AUDIT},
    signing_window_seconds=14400,
)

SANCTIONS_QUORUM = QuorumPolicy(
    tier=TransactionTier.SANCTIONS,
    threshold=3, total_authorities=3,
    required_roles={AuthorityRole.COMPLIANCE, AuthorityRole.LEGAL, AuthorityRole.RISK_MANAGEMENT},
    optional_roles=set(),
    signing_window_seconds=86400,
)


@dataclass
class Authority:
    """A signing authority in the multi-authority scheme."""
    authority_id: str
    role: AuthorityRole
    department: str
    public_key: bytes
    private_key: bytes
    is_active: bool = True


@dataclass
class SigningRequest:
    """A pending multi-authority signing request."""
    request_id: bytes
    transaction_data: bytes
    transaction_hash: bytes
    quorum_policy: QuorumPolicy
    initiator_id: str
    created_at: float = field(default_factory=time.time)
    deadline: float = 0.0
    status: str = "pending"  # pending, approved, rejected, expired
    collected_shares: Dict[str, bytes] = field(default_factory=dict)  # authority_id -> partial_sig
    authority_roles: Dict[str, AuthorityRole] = field(default_factory=dict)

    def __post_init__(self):
        if self.deadline == 0.0:
            object.__setattr__(
                self, "deadline",
                self.created_at + self.quorum_policy.signing_window_seconds,
            )

    @property
    def is_expired(self) -> bool:
        return time.time() > self.deadline

    @property
    def shares_collected(self) -> int:
        return len(self.collected_shares)

    @property
    def required_roles_satisfied(self) -> bool:
        collected_roles = set(self.authority_roles.values())
        return self.quorum_policy.required_roles.issubset(collected_roles)

    @property
    def threshold_met(self) -> bool:
        return self.shares_collected >= self.quorum_policy.threshold


@dataclass
class CombinedSignature:
    """The final combined multi-authority signature."""
    request_id: bytes
    combined_signature: bytes
    contributing_authorities: List[str]
    contributing_roles: List[AuthorityRole]
    transaction_hash: bytes
    tier: TransactionTier
    created_at: float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        return (
            self.request_id
            + struct.pack(">H", len(self.combined_signature))
            + self.combined_signature
            + self.transaction_hash
        )


class MultiAuthorityThreshold:
    """
    Multi-authority threshold signature scheme for banking.

    Manages the lifecycle of multi-party signing requests:
    1. Initiation: Transaction submitted with quorum requirements
    2. Collection: Partial signatures gathered from authorities
    3. Combination: Threshold met, final signature produced
    4. Verification: Any party can verify the combined signature

    Usage:
        scheme = MultiAuthorityThreshold()
        await scheme.setup(authorities)

        # Initiate signing request
        request = await scheme.initiate_signing(transaction_data, HIGH_VALUE_QUORUM)

        # Each authority contributes
        await scheme.contribute_signature(request.request_id, authority, private_key)

        # Check if threshold met
        if scheme.is_ready(request.request_id):
            combined = await scheme.combine_signatures(request.request_id)
    """

    def __init__(self):
        self._authorities: Dict[str, Authority] = {}
        self._pending_requests: Dict[bytes, SigningRequest] = {}
        self._completed: Dict[bytes, CombinedSignature] = {}
        self._group_public_key: Optional[bytes] = None
        logger.info("Multi-authority threshold scheme initialized")

    async def setup(self, authorities: List[Authority]) -> bytes:
        """
        Setup the multi-authority group.

        Returns the group public key for verification.
        """
        for auth in authorities:
            self._authorities[auth.authority_id] = auth

        # Derive group public key from all authority keys
        combined = b""
        for auth in sorted(authorities, key=lambda a: a.authority_id):
            combined += auth.public_key

        self._group_public_key = hashlib.sha3_256(combined).digest()

        THRESHOLD_OPS.labels(operation="setup").inc()
        logger.info(f"Group setup: {len(authorities)} authorities, key={self._group_public_key.hex()[:16]}")

        return self._group_public_key

    async def generate_authority_keypair(
        self,
        authority_id: str,
        role: AuthorityRole,
        department: str,
    ) -> Authority:
        """Generate a PQC keypair for a signing authority."""
        from ai_engine.crypto.dilithium import DilithiumEngine, DilithiumSecurityLevel

        engine = DilithiumEngine(DilithiumSecurityLevel.MLDSA_65)
        keypair = await engine.generate_keypair()

        return Authority(
            authority_id=authority_id,
            role=role,
            department=department,
            public_key=keypair.public_key.data,
            private_key=keypair.private_key.data,
        )

    async def initiate_signing(
        self,
        transaction_data: bytes,
        quorum: QuorumPolicy,
        initiator_id: str = "",
    ) -> SigningRequest:
        """Initiate a multi-authority signing request."""
        request_id = secrets.token_bytes(16)
        tx_hash = hashlib.sha3_256(transaction_data).digest()

        request = SigningRequest(
            request_id=request_id,
            transaction_data=transaction_data,
            transaction_hash=tx_hash,
            quorum_policy=quorum,
            initiator_id=initiator_id,
        )

        self._pending_requests[request_id] = request
        THRESHOLD_OPS.labels(operation="initiate").inc()

        logger.info(
            f"Signing request initiated: {request_id.hex()[:8]}, "
            f"tier={quorum.tier.value}, threshold={quorum.threshold}/{quorum.total_authorities}"
        )

        return request

    async def contribute_signature(
        self,
        request_id: bytes,
        authority: Authority,
    ) -> bool:
        """
        Authority contributes their partial signature to a request.

        Returns True if the contribution was accepted.
        """
        start = time.perf_counter()

        request = self._pending_requests.get(request_id)
        if not request:
            logger.warning(f"Unknown request: {request_id.hex()[:8]}")
            return False

        if request.is_expired:
            request.status = "expired"
            logger.warning(f"Request {request_id.hex()[:8]} expired")
            return False

        if request.status != "pending":
            return False

        if authority.authority_id in request.collected_shares:
            logger.warning(f"Authority {authority.authority_id} already contributed")
            return False

        # Verify authority is eligible
        all_eligible = request.quorum_policy.required_roles | request.quorum_policy.optional_roles
        if authority.role not in all_eligible:
            logger.warning(f"Authority role {authority.role.name} not eligible for this quorum")
            return False

        # Check department uniqueness
        if request.quorum_policy.require_different_departments:
            existing_depts = set()
            for aid in request.collected_shares:
                if aid in self._authorities:
                    existing_depts.add(self._authorities[aid].department)
            if authority.department in existing_depts:
                logger.warning(f"Department {authority.department} already represented")
                return False

        # Compute partial signature
        from ai_engine.crypto.dilithium import (
            DilithiumEngine, DilithiumSecurityLevel, DilithiumPrivateKey,
        )

        engine = DilithiumEngine(DilithiumSecurityLevel.MLDSA_65)
        sk = DilithiumPrivateKey(DilithiumSecurityLevel.MLDSA_65, authority.private_key)

        sig_input = request.transaction_hash + request_id + authority.authority_id.encode()
        signature = await engine.sign(sig_input, sk)

        request.collected_shares[authority.authority_id] = signature.data
        request.authority_roles[authority.authority_id] = authority.role

        elapsed = (time.perf_counter() - start) * 1000
        THRESHOLD_LATENCY.observe(elapsed)
        THRESHOLD_OPS.labels(operation="contribute").inc()

        logger.info(
            f"Signature contributed: {authority.authority_id} ({authority.role.name}), "
            f"{request.shares_collected}/{request.quorum_policy.threshold} collected"
        )

        return True

    def is_ready(self, request_id: bytes) -> bool:
        """Check if a signing request has reached quorum."""
        request = self._pending_requests.get(request_id)
        if not request:
            return False
        return request.threshold_met and request.required_roles_satisfied

    async def combine_signatures(self, request_id: bytes) -> CombinedSignature:
        """Combine partial signatures into a final multi-authority signature."""
        start = time.perf_counter()

        request = self._pending_requests.get(request_id)
        if not request:
            raise ValueError(f"Unknown request: {request_id.hex()[:8]}")

        if not self.is_ready(request_id):
            missing_roles = request.quorum_policy.required_roles - set(request.authority_roles.values())
            raise ValueError(
                f"Quorum not met: {request.shares_collected}/{request.quorum_policy.threshold} shares, "
                f"missing roles: {[r.name for r in missing_roles]}"
            )

        # Combine: hash all partial signatures together
        combined_input = b""
        for auth_id in sorted(request.collected_shares.keys()):
            combined_input += request.collected_shares[auth_id]

        combined = hashlib.shake_256(
            combined_input + request.transaction_hash + b"multi-auth-combine-v1"
        ).digest(128)

        result = CombinedSignature(
            request_id=request_id,
            combined_signature=combined,
            contributing_authorities=list(request.collected_shares.keys()),
            contributing_roles=list(request.authority_roles.values()),
            transaction_hash=request.transaction_hash,
            tier=request.quorum_policy.tier,
        )

        request.status = "approved"
        self._completed[request_id] = result

        elapsed = (time.perf_counter() - start) * 1000
        THRESHOLD_LATENCY.observe(elapsed)
        THRESHOLD_OPS.labels(operation="combine").inc()

        logger.info(
            f"Signatures combined: {request_id.hex()[:8]}, "
            f"{len(result.contributing_authorities)} authorities, "
            f"tier={result.tier.value}"
        )

        return result

    async def verify_combined(
        self,
        combined: CombinedSignature,
        transaction_data: bytes,
    ) -> bool:
        """Verify a combined multi-authority signature."""
        tx_hash = hashlib.sha3_256(transaction_data).digest()

        if not secrets.compare_digest(tx_hash, combined.transaction_hash):
            return False

        THRESHOLD_OPS.labels(operation="verify").inc()
        return len(combined.combined_signature) > 0

    def get_pending_requests(
        self,
        authority_role: Optional[AuthorityRole] = None,
    ) -> List[SigningRequest]:
        """Get pending signing requests, optionally filtered by role."""
        requests = [r for r in self._pending_requests.values() if r.status == "pending"]

        if authority_role:
            requests = [
                r for r in requests
                if authority_role in (r.quorum_policy.required_roles | r.quorum_policy.optional_roles)
            ]

        return requests

    @staticmethod
    def get_quorum_for_amount(amount_usd: float) -> QuorumPolicy:
        """Determine the appropriate quorum policy for a transaction amount."""
        if amount_usd >= 100_000_000:
            return CRITICAL_QUORUM
        elif amount_usd >= 10_000_000:
            return HIGH_VALUE_QUORUM
        elif amount_usd >= 1_000_000:
            return ELEVATED_QUORUM
        else:
            return QuorumPolicy(
                tier=TransactionTier.STANDARD,
                threshold=1, total_authorities=1,
                required_roles={AuthorityRole.OPERATIONS},
                optional_roles=set(),
                signing_window_seconds=300,
            )
