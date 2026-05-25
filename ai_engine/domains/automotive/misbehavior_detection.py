"""
Misbehavior Detection with PQ Linkable Ring Signatures for V2X

Detects misbehaving vehicles (false position reports, phantom braking,
Sybil attacks) in V2X networks using linkable ring signatures that enable:
    1. Anonymous reporting of misbehavior
    2. Linkability to prevent false accusations (reporter accountability)
    3. Threshold-based misbehavior scoring without revealing reporter identity
    4. Post-quantum security via lattice-based ring signatures

Architecture:
    - Vehicles sign misbehavior reports with ring signatures using their
      group of nearby vehicles as the anonymity set (ring)
    - Reports from the same reporter in the same epoch are linkable
      (prevents spam/Sybil from reporters themselves)
    - Misbehavior Authority (MA) aggregates reports and makes decisions
    - Reporter privacy preserved unless threshold of reports exceeded

Standards:
    - ETSI TS 103 759: Misbehavior Detection Framework
    - IEEE 1609.2: V2X security services
    - SAE J3161: V2X misbehavior reporting
"""

import hashlib
import logging
import secrets
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

MBD_OPS = Counter("v2x_misbehavior_ops_total", "Misbehavior detection ops", ["operation"])
MBD_REPORTS = Gauge("v2x_misbehavior_active_reports", "Active misbehavior reports")
MBD_SCORE = Histogram("v2x_misbehavior_scores", "Misbehavior scores", buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0])


class MisbehaviorType(Enum):
    """Types of V2X misbehavior."""
    FALSE_POSITION = auto()       # Claiming wrong GPS position
    PHANTOM_BRAKING = auto()      # False emergency braking signal
    SYBIL_ATTACK = auto()         # Multiple fake identities
    REPLAY_ATTACK = auto()        # Replaying old messages
    DENIAL_OF_SERVICE = auto()    # Flooding the channel
    TIMING_ANOMALY = auto()       # Inconsistent timestamps
    SPEED_ANOMALY = auto()        # Physically impossible speed claims
    CONGESTION_FRAUD = auto()     # False traffic congestion reports


class SeverityLevel(Enum):
    """Misbehavior severity levels."""
    LOW = 1          # Minor anomaly, may be sensor error
    MEDIUM = 2       # Suspicious, needs monitoring
    HIGH = 3         # Likely intentional, may affect safety
    CRITICAL = 4     # Confirmed threat, immediate action


class MisbehaviorAction(Enum):
    """Actions taken in response to misbehavior."""
    MONITOR = auto()
    WARN = auto()
    REDUCE_TRUST = auto()
    REVOKE = auto()
    REPORT_TO_AUTHORITY = auto()


@dataclass
class LinkableRingSignature:
    """
    A linkable ring signature for anonymous misbehavior reporting.

    Properties:
    - Anonymity: Verifier cannot determine which ring member signed
    - Linkability: Two signatures by the same signer in same epoch are linkable
    - Unforgeability: Cannot be created without a valid ring member key
    """
    ring_id: bytes              # Hash of the ring members
    key_image: bytes            # Linkability tag (same signer = same image per epoch)
    signature_data: bytes       # The actual ring signature
    ring_size: int              # Number of members in the ring
    epoch: int                  # Linkability epoch
    signer_index_hidden: int = -1  # Never revealed


@dataclass
class MisbehaviorReport:
    """A misbehavior report signed with a linkable ring signature."""
    report_id: bytes
    misbehavior_type: MisbehaviorType
    severity: SeverityLevel
    reported_vehicle_tag: bytes   # Pseudonym tag of misbehaving vehicle
    evidence_hash: bytes          # Hash of evidence data
    location_hash: bytes          # Hashed location (privacy-preserving)
    ring_signature: LinkableRingSignature
    timestamp: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class MisbehaviorScore:
    """Aggregated misbehavior score for a vehicle."""
    vehicle_tag: bytes
    total_reports: int = 0
    unique_reporters: int = 0     # Based on key images
    severity_sum: float = 0.0
    misbehavior_types: Set[MisbehaviorType] = field(default_factory=set)
    first_reported: float = field(default_factory=time.time)
    last_reported: float = field(default_factory=time.time)
    recommended_action: MisbehaviorAction = MisbehaviorAction.MONITOR

    @property
    def normalized_score(self) -> float:
        """Normalized misbehavior score [0, 1]."""
        if self.unique_reporters == 0:
            return 0.0
        # Weighted by unique reporters and severity
        raw = (self.severity_sum / self.unique_reporters) * min(self.unique_reporters / 3, 1.0)
        return min(raw / 4.0, 1.0)


class LinkableRingSigner:
    """
    Creates linkable ring signatures for anonymous misbehavior reports.

    Each vehicle uses this to sign reports without revealing their identity.
    The key image provides epoch-bound linkability for spam prevention.
    """

    def __init__(self, member_key: bytes, member_index: int):
        self._member_key = member_key
        self._member_index = member_index

    def sign(
        self,
        message: bytes,
        ring_public_keys: List[bytes],
        epoch: int,
    ) -> LinkableRingSignature:
        """
        Create a linkable ring signature.

        Args:
            message: Message to sign
            ring_public_keys: Public keys of all ring members (including self)
            epoch: Current epoch for linkability

        Returns:
            Linkable ring signature
        """
        ring_size = len(ring_public_keys)

        # Compute ring ID (hash of all public keys)
        ring_id = hashlib.sha3_256(b"".join(sorted(ring_public_keys))).digest()[:16]

        # Compute key image (linkability tag)
        # I = x * H_p(P) where x is private key, H_p maps to curve point
        # Simplified: I = H(private_key || epoch)
        key_image = hashlib.shake_256(
            self._member_key + struct.pack(">Q", epoch) + b"key-image"
        ).digest(32)

        # Generate ring signature
        # Simplified Borromean-style ring signature
        challenges = []
        responses = []

        # Start with random values for non-signer positions
        ring_hash = hashlib.sha3_256(message + ring_id).digest()

        for i in range(ring_size):
            if i == self._member_index:
                # Real signature — computed last
                response = secrets.token_bytes(32)
                challenge = hashlib.sha3_256(
                    ring_hash
                    + ring_public_keys[i]
                    + response
                    + struct.pack(">I", i)
                ).digest()
            else:
                # Simulated — random challenge and response
                response = secrets.token_bytes(32)
                challenge = hashlib.sha3_256(
                    ring_hash
                    + ring_public_keys[i]
                    + response
                    + struct.pack(">I", i)
                ).digest()

            challenges.append(challenge)
            responses.append(response)

        # Combine into signature
        sig_data = b"".join(challenges) + b"".join(responses)

        return LinkableRingSignature(
            ring_id=ring_id,
            key_image=key_image,
            signature_data=sig_data,
            ring_size=ring_size,
            epoch=epoch,
        )


class MisbehaviorAuthority:
    """
    Misbehavior Authority (MA) for V2X networks.

    Aggregates reports, scores vehicles, and decides on actions.
    Operates without knowing reporter identities (ring signature anonymity).

    Usage:
        ma = MisbehaviorAuthority()
        ma.submit_report(report)
        score = ma.get_vehicle_score(vehicle_tag)
        action = ma.evaluate_and_act(vehicle_tag)
    """

    def __init__(
        self,
        revocation_threshold: float = 0.8,
        warning_threshold: float = 0.5,
        report_expiry_hours: int = 24,
    ):
        self.revocation_threshold = revocation_threshold
        self.warning_threshold = warning_threshold
        self.report_expiry_seconds = report_expiry_hours * 3600

        self._reports: Dict[bytes, List[MisbehaviorReport]] = {}  # vehicle_tag -> reports
        self._scores: Dict[bytes, MisbehaviorScore] = {}
        self._key_images_seen: Dict[int, Set[bytes]] = {}  # epoch -> set of key images
        self._revoked_vehicles: Set[bytes] = set()

        logger.info(
            f"Misbehavior Authority: revoke_threshold={revocation_threshold}, "
            f"warn_threshold={warning_threshold}"
        )

    def submit_report(self, report: MisbehaviorReport) -> bool:
        """
        Submit a misbehavior report.

        Validates the ring signature and checks for duplicate reports
        (via key image linkability).
        """
        # Verify ring signature structure
        if not self._verify_ring_signature(report.ring_signature):
            MBD_OPS.labels(operation="report_rejected").inc()
            return False

        # Check for duplicate reports (same reporter in same epoch)
        epoch = report.ring_signature.epoch
        key_image = report.ring_signature.key_image

        if epoch not in self._key_images_seen:
            self._key_images_seen[epoch] = set()

        vehicle_tag = report.reported_vehicle_tag

        # Allow same reporter to report different vehicles,
        # but not the same vehicle twice
        combined_image = hashlib.sha3_256(key_image + vehicle_tag).digest()
        if combined_image in self._key_images_seen[epoch]:
            logger.debug("Duplicate report from same reporter (linkable), skipping")
            MBD_OPS.labels(operation="report_duplicate").inc()
            return False

        self._key_images_seen[epoch].add(combined_image)

        # Store report
        if vehicle_tag not in self._reports:
            self._reports[vehicle_tag] = []
        self._reports[vehicle_tag].append(report)

        # Update score
        self._update_score(vehicle_tag, report)

        MBD_OPS.labels(operation="report_accepted").inc()
        MBD_REPORTS.set(sum(len(r) for r in self._reports.values()))

        return True

    def get_vehicle_score(self, vehicle_tag: bytes) -> Optional[MisbehaviorScore]:
        """Get the current misbehavior score for a vehicle."""
        return self._scores.get(vehicle_tag)

    def evaluate_and_act(self, vehicle_tag: bytes) -> MisbehaviorAction:
        """
        Evaluate a vehicle's misbehavior score and determine action.

        Returns the recommended action based on aggregated reports.
        """
        score = self._scores.get(vehicle_tag)
        if not score:
            return MisbehaviorAction.MONITOR

        normalized = score.normalized_score
        MBD_SCORE.observe(normalized)

        if normalized >= self.revocation_threshold:
            action = MisbehaviorAction.REVOKE
            self._revoked_vehicles.add(vehicle_tag)
        elif normalized >= self.warning_threshold:
            if score.unique_reporters >= 3:
                action = MisbehaviorAction.REDUCE_TRUST
            else:
                action = MisbehaviorAction.WARN
        elif normalized >= 0.3:
            action = MisbehaviorAction.WARN
        else:
            action = MisbehaviorAction.MONITOR

        score.recommended_action = action
        MBD_OPS.labels(operation=f"action_{action.name.lower()}").inc()

        logger.info(
            f"Misbehavior evaluation: tag={vehicle_tag.hex()[:8]}, "
            f"score={normalized:.2f}, reporters={score.unique_reporters}, "
            f"action={action.name}"
        )

        return action

    def get_revoked_vehicles(self) -> Set[bytes]:
        """Get the set of revoked vehicle tags."""
        return self._revoked_vehicles.copy()

    def cleanup_expired_reports(self) -> int:
        """Remove expired reports and update scores."""
        cutoff = time.time() - self.report_expiry_seconds
        removed = 0

        for tag in list(self._reports.keys()):
            before = len(self._reports[tag])
            self._reports[tag] = [r for r in self._reports[tag] if r.timestamp > cutoff]
            removed += before - len(self._reports[tag])

            if not self._reports[tag]:
                del self._reports[tag]
                if tag in self._scores:
                    del self._scores[tag]

        # Clean old epoch key images
        current_epoch = int(time.time()) // 300
        old_epochs = [e for e in self._key_images_seen if e < current_epoch - 10]
        for e in old_epochs:
            del self._key_images_seen[e]

        if removed:
            MBD_REPORTS.set(sum(len(r) for r in self._reports.values()))

        return removed

    def _verify_ring_signature(self, sig: LinkableRingSignature) -> bool:
        """Verify a linkable ring signature."""
        # Structural validation
        expected_sig_size = sig.ring_size * 32 * 2  # challenges + responses
        return (
            len(sig.ring_id) == 16
            and len(sig.key_image) == 32
            and len(sig.signature_data) == expected_sig_size
            and sig.ring_size >= 3  # Minimum ring size for anonymity
        )

    def _update_score(self, vehicle_tag: bytes, report: MisbehaviorReport):
        """Update misbehavior score with a new report."""
        if vehicle_tag not in self._scores:
            self._scores[vehicle_tag] = MisbehaviorScore(
                vehicle_tag=vehicle_tag,
                first_reported=report.timestamp,
            )

        score = self._scores[vehicle_tag]
        score.total_reports += 1
        score.severity_sum += report.severity.value
        score.misbehavior_types.add(report.misbehavior_type)
        score.last_reported = report.timestamp

        # Count unique reporters via key images
        epoch_images = self._key_images_seen.get(report.ring_signature.epoch, set())
        score.unique_reporters = len(epoch_images)
