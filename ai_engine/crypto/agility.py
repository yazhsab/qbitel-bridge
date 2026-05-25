"""
Post-Quantum Crypto Agility Negotiation Protocol

Enables runtime algorithm negotiation between peers during the classical-to-PQC
transition period. Supports capability advertisement, preference ordering,
hybrid fallback, and automatic algorithm rotation when vulnerabilities are
discovered.

Design Goals:
    1. Forward Compatibility: Negotiate unknown-future algorithms via OID/name
    2. Hybrid Negotiation: Classical + PQC simultaneous operation
    3. Policy Enforcement: Minimum security levels, banned algorithms
    4. Zero-Downtime Rotation: Hot-swap algorithms without session interruption
    5. Audit Trail: Full negotiation history for compliance

Protocol Flow:
    1. Initiator sends CryptoCapability advertisement
    2. Responder replies with CryptoCapability + selection
    3. Both derive NegotiatedSuite from intersection of capabilities
    4. Session proceeds with agreed algorithms
    5. Either side can trigger renegotiation at any time

Standards Alignment:
    - NIST SP 800-227: Recommendations for Transition to PQC
    - CNSA 2.0: NSA algorithm suite timeline
    - ETSI TS 103 744: Migration strategies for PQC
    - RFC 8446: TLS 1.3 cipher suite negotiation model
"""

import hashlib
import logging
import secrets
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)

# Metrics
AGILITY_NEGOTIATIONS = Counter("crypto_agility_negotiations_total", "Negotiations", ["result"])
AGILITY_ACTIVE_SUITES = Gauge("crypto_agility_active_suites", "Active negotiated suites")


class AlgorithmFamily(Enum):
    """Cryptographic algorithm families."""
    KEM = "kem"
    SIGNATURE = "signature"
    SYMMETRIC = "symmetric"
    HASH = "hash"
    MAC = "mac"
    KDF = "kdf"


class SecurityEra(Enum):
    """Cryptographic security era classification."""
    CLASSICAL = "classical"
    HYBRID = "hybrid"
    POST_QUANTUM = "post-quantum"
    CNSA2 = "cnsa-2.0"


class AlgorithmStatus(Enum):
    """Algorithm lifecycle status."""
    RECOMMENDED = auto()
    ACCEPTABLE = auto()
    LEGACY = auto()
    DEPRECATED = auto()
    BANNED = auto()


@dataclass(frozen=True)
class AlgorithmDescriptor:
    """
    Describes a cryptographic algorithm with its properties.

    Used in capability advertisements and negotiation.
    """
    name: str                        # Canonical name (e.g., "ML-KEM-768")
    family: AlgorithmFamily          # KEM, signature, etc.
    security_bits: int               # Estimated security strength
    era: SecurityEra                 # Classical, hybrid, or PQ
    nist_level: int = 0              # NIST security level (0 if N/A)
    key_size: int = 0                # Public key size in bytes
    output_size: int = 0             # Ciphertext/signature size
    status: AlgorithmStatus = AlgorithmStatus.RECOMMENDED
    oid: str = ""                    # ASN.1 OID if assigned

    @property
    def is_post_quantum(self) -> bool:
        return self.era in (SecurityEra.POST_QUANTUM, SecurityEra.CNSA2, SecurityEra.HYBRID)


# ── Algorithm Registry ────────────────────────────────────────────

# Well-known algorithms with their properties
ALGORITHM_REGISTRY: Dict[str, AlgorithmDescriptor] = {
    # KEMs
    "ML-KEM-512": AlgorithmDescriptor("ML-KEM-512", AlgorithmFamily.KEM, 128, SecurityEra.POST_QUANTUM, 1, 800, 768),
    "ML-KEM-768": AlgorithmDescriptor("ML-KEM-768", AlgorithmFamily.KEM, 192, SecurityEra.POST_QUANTUM, 3, 1184, 1088),
    "ML-KEM-1024": AlgorithmDescriptor("ML-KEM-1024", AlgorithmFamily.KEM, 256, SecurityEra.POST_QUANTUM, 5, 1568, 1568),
    "X25519": AlgorithmDescriptor("X25519", AlgorithmFamily.KEM, 128, SecurityEra.CLASSICAL, 0, 32, 32),
    "P-384": AlgorithmDescriptor("P-384", AlgorithmFamily.KEM, 192, SecurityEra.CLASSICAL, 0, 97, 97),
    "X25519-ML-KEM-768": AlgorithmDescriptor("X25519-ML-KEM-768", AlgorithmFamily.KEM, 192, SecurityEra.HYBRID, 3, 1216, 1120),
    "P384-ML-KEM-1024": AlgorithmDescriptor("P384-ML-KEM-1024", AlgorithmFamily.KEM, 256, SecurityEra.HYBRID, 5, 1665, 1665),
    # Signatures
    "ML-DSA-44": AlgorithmDescriptor("ML-DSA-44", AlgorithmFamily.SIGNATURE, 128, SecurityEra.POST_QUANTUM, 2, 1312, 2420),
    "ML-DSA-65": AlgorithmDescriptor("ML-DSA-65", AlgorithmFamily.SIGNATURE, 192, SecurityEra.POST_QUANTUM, 3, 1952, 3293),
    "ML-DSA-87": AlgorithmDescriptor("ML-DSA-87", AlgorithmFamily.SIGNATURE, 256, SecurityEra.POST_QUANTUM, 5, 2592, 4595),
    "Falcon-512": AlgorithmDescriptor("Falcon-512", AlgorithmFamily.SIGNATURE, 128, SecurityEra.POST_QUANTUM, 1, 897, 666),
    "Falcon-1024": AlgorithmDescriptor("Falcon-1024", AlgorithmFamily.SIGNATURE, 256, SecurityEra.POST_QUANTUM, 5, 1793, 1330),
    "SLH-DSA-128f": AlgorithmDescriptor("SLH-DSA-128f", AlgorithmFamily.SIGNATURE, 128, SecurityEra.POST_QUANTUM, 1, 32, 17088),
    "ECDSA-P256": AlgorithmDescriptor("ECDSA-P256", AlgorithmFamily.SIGNATURE, 128, SecurityEra.CLASSICAL, 0, 64, 72),
    "Ed25519": AlgorithmDescriptor("Ed25519", AlgorithmFamily.SIGNATURE, 128, SecurityEra.CLASSICAL, 0, 32, 64),
    # LMS/XMSS (CNSA 2.0 stateful)
    "LMS-SHA256-H20": AlgorithmDescriptor("LMS-SHA256-H20", AlgorithmFamily.SIGNATURE, 256, SecurityEra.CNSA2, 5, 56, 4784),
    "XMSS-SHA2-20": AlgorithmDescriptor("XMSS-SHA2-20", AlgorithmFamily.SIGNATURE, 256, SecurityEra.CNSA2, 5, 64, 2500),
    # Symmetric
    "AES-256-GCM": AlgorithmDescriptor("AES-256-GCM", AlgorithmFamily.SYMMETRIC, 256, SecurityEra.POST_QUANTUM, 0, 32, 0),
    "AES-128-GCM": AlgorithmDescriptor("AES-128-GCM", AlgorithmFamily.SYMMETRIC, 128, SecurityEra.CLASSICAL, 0, 16, 0),
    # Hash
    "SHA3-256": AlgorithmDescriptor("SHA3-256", AlgorithmFamily.HASH, 256, SecurityEra.POST_QUANTUM, 0, 0, 32),
    "SHAKE256": AlgorithmDescriptor("SHAKE256", AlgorithmFamily.HASH, 256, SecurityEra.POST_QUANTUM, 0, 0, 0),
    "SHA-256": AlgorithmDescriptor("SHA-256", AlgorithmFamily.HASH, 128, SecurityEra.CLASSICAL, 0, 0, 32),
    # KDF
    "HKDF-SHA256": AlgorithmDescriptor("HKDF-SHA256", AlgorithmFamily.KDF, 128, SecurityEra.CLASSICAL, 0, 0, 0),
    "HKDF-SHA3-256": AlgorithmDescriptor("HKDF-SHA3-256", AlgorithmFamily.KDF, 256, SecurityEra.POST_QUANTUM, 0, 0, 0),
}


@dataclass
class CryptoPolicy:
    """
    Organizational crypto policy constraining negotiation.

    Defines which algorithms are allowed, minimum security levels,
    and migration timelines.
    """
    minimum_security_bits: int = 128
    minimum_nist_level: int = 1
    required_era: Optional[SecurityEra] = None
    banned_algorithms: FrozenSet[str] = frozenset()
    preferred_kem: str = "ML-KEM-768"
    preferred_sig: str = "ML-DSA-65"
    preferred_symmetric: str = "AES-256-GCM"
    preferred_hash: str = "SHA3-256"
    require_hybrid: bool = False         # Force hybrid mode during transition
    allow_classical_only: bool = True    # Allow pure classical (pre-migration)
    cnsa2_required: bool = False         # Require CNSA 2.0 compliance


# Pre-defined policies
TRANSITIONAL_POLICY = CryptoPolicy(
    minimum_security_bits=128,
    require_hybrid=True,
    allow_classical_only=False,
)

POST_QUANTUM_POLICY = CryptoPolicy(
    minimum_security_bits=192,
    minimum_nist_level=3,
    required_era=SecurityEra.POST_QUANTUM,
    allow_classical_only=False,
    banned_algorithms=frozenset({"ECDSA-P256", "Ed25519", "X25519", "AES-128-GCM", "SHA-256"}),
)

CNSA2_POLICY = CryptoPolicy(
    minimum_security_bits=256,
    minimum_nist_level=5,
    required_era=SecurityEra.CNSA2,
    preferred_kem="ML-KEM-1024",
    preferred_sig="ML-DSA-87",
    cnsa2_required=True,
    allow_classical_only=False,
    require_hybrid=False,
)

LEGACY_COMPATIBLE_POLICY = CryptoPolicy(
    minimum_security_bits=128,
    allow_classical_only=True,
    require_hybrid=False,
)


@dataclass
class CryptoCapability:
    """
    Advertised cryptographic capabilities of a peer.

    Sent during negotiation to declare supported algorithms
    and preferences.
    """
    peer_id: str
    supported_kems: List[str]        # Ordered by preference
    supported_sigs: List[str]
    supported_symmetric: List[str]
    supported_hashes: List[str]
    supported_kdfs: List[str]
    policy_era: SecurityEra = SecurityEra.HYBRID
    max_key_size: int = 0            # 0 = no constraint
    max_signature_size: int = 0      # 0 = no constraint
    timestamp: float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        """Serialize for wire transmission."""
        parts = [
            self.peer_id.encode(),
            b"|",
            ",".join(self.supported_kems).encode(),
            b"|",
            ",".join(self.supported_sigs).encode(),
            b"|",
            ",".join(self.supported_symmetric).encode(),
            b"|",
            ",".join(self.supported_hashes).encode(),
        ]
        return b"".join(parts)


@dataclass
class NegotiatedSuite:
    """
    Result of crypto agility negotiation.

    Contains the agreed-upon algorithms for each operation type.
    """
    suite_id: bytes
    kem: AlgorithmDescriptor
    signature: AlgorithmDescriptor
    symmetric: AlgorithmDescriptor
    hash_algo: AlgorithmDescriptor
    kdf: AlgorithmDescriptor
    era: SecurityEra
    negotiated_at: float = field(default_factory=time.time)
    valid_until: float = 0.0
    initiator_id: str = ""
    responder_id: str = ""

    def __post_init__(self):
        if self.valid_until == 0.0:
            object.__setattr__(self, "valid_until", self.negotiated_at + 86400)

    @property
    def security_level(self) -> int:
        """Minimum security bits across all algorithms in suite."""
        return min(
            self.kem.security_bits,
            self.signature.security_bits,
            self.symmetric.security_bits,
            self.hash_algo.security_bits,
        )

    @property
    def is_fully_post_quantum(self) -> bool:
        return all(
            a.is_post_quantum
            for a in [self.kem, self.signature, self.symmetric, self.hash_algo]
        )


class NegotiationFailedError(Exception):
    """Raised when peers cannot agree on a crypto suite."""
    pass


class CryptoAgilityNegotiator:
    """
    Crypto agility negotiation engine.

    Negotiates algorithm suites between peers based on capabilities,
    preferences, and organizational policies.

    Usage:
        negotiator = CryptoAgilityNegotiator("peer-a", policy=TRANSITIONAL_POLICY)
        my_caps = negotiator.advertise_capabilities()

        # Exchange capabilities with peer...

        suite = negotiator.negotiate(my_caps, peer_caps)
        # Use suite.kem, suite.signature, etc.
    """

    def __init__(
        self,
        peer_id: str,
        policy: CryptoPolicy = TRANSITIONAL_POLICY,
        registry: Optional[Dict[str, AlgorithmDescriptor]] = None,
    ):
        self.peer_id = peer_id
        self.policy = policy
        self.registry = registry or ALGORITHM_REGISTRY

        self._negotiation_history: List[NegotiatedSuite] = []

        logger.info(f"Crypto agility negotiator: peer={peer_id}, era={policy.required_era}")

    def advertise_capabilities(self) -> CryptoCapability:
        """
        Generate capability advertisement based on local policy.

        Filters the algorithm registry against the local policy
        and returns supported algorithms ordered by preference.
        """
        kems = self._filter_algorithms(AlgorithmFamily.KEM)
        sigs = self._filter_algorithms(AlgorithmFamily.SIGNATURE)
        symmetric = self._filter_algorithms(AlgorithmFamily.SYMMETRIC)
        hashes = self._filter_algorithms(AlgorithmFamily.HASH)
        kdfs = self._filter_algorithms(AlgorithmFamily.KDF)

        cap = CryptoCapability(
            peer_id=self.peer_id,
            supported_kems=[a.name for a in kems],
            supported_sigs=[a.name for a in sigs],
            supported_symmetric=[a.name for a in symmetric],
            supported_hashes=[a.name for a in hashes],
            supported_kdfs=[a.name for a in kdfs],
            policy_era=self.policy.required_era or SecurityEra.HYBRID,
        )

        return cap

    def negotiate(
        self,
        local_caps: CryptoCapability,
        remote_caps: CryptoCapability,
    ) -> NegotiatedSuite:
        """
        Negotiate a crypto suite from two capability advertisements.

        Finds the best mutually supported algorithm for each family,
        preferring the initiator's ordering.
        """
        # Find intersection for each family
        kem = self._select_best(
            local_caps.supported_kems,
            remote_caps.supported_kems,
            AlgorithmFamily.KEM,
        )
        sig = self._select_best(
            local_caps.supported_sigs,
            remote_caps.supported_sigs,
            AlgorithmFamily.SIGNATURE,
        )
        sym = self._select_best(
            local_caps.supported_symmetric,
            remote_caps.supported_symmetric,
            AlgorithmFamily.SYMMETRIC,
        )
        hash_algo = self._select_best(
            local_caps.supported_hashes,
            remote_caps.supported_hashes,
            AlgorithmFamily.HASH,
        )
        kdf = self._select_best(
            local_caps.supported_kdfs,
            remote_caps.supported_kdfs,
            AlgorithmFamily.KDF,
        )

        if not all([kem, sig, sym, hash_algo]):
            AGILITY_NEGOTIATIONS.labels(result="failed").inc()
            raise NegotiationFailedError(
                f"No common algorithms between {local_caps.peer_id} and {remote_caps.peer_id}"
            )

        # If KDF not found, use default
        if not kdf:
            kdf = self.registry.get("HKDF-SHA3-256") or self.registry.get("HKDF-SHA256")

        # Determine negotiated era
        all_algos = [kem, sig, sym, hash_algo]
        if all(a.era in (SecurityEra.POST_QUANTUM, SecurityEra.CNSA2) for a in all_algos):
            era = SecurityEra.POST_QUANTUM
        elif any(a.is_post_quantum for a in all_algos):
            era = SecurityEra.HYBRID
        else:
            era = SecurityEra.CLASSICAL

        suite_id = hashlib.sha3_256(
            kem.name.encode() + sig.name.encode()
            + sym.name.encode() + hash_algo.name.encode()
            + secrets.token_bytes(8)
        ).digest()[:16]

        suite = NegotiatedSuite(
            suite_id=suite_id,
            kem=kem,
            signature=sig,
            symmetric=sym,
            hash_algo=hash_algo,
            kdf=kdf,
            era=era,
            initiator_id=local_caps.peer_id,
            responder_id=remote_caps.peer_id,
        )

        self._negotiation_history.append(suite)
        AGILITY_NEGOTIATIONS.labels(result="success").inc()
        AGILITY_ACTIVE_SUITES.set(len(self._negotiation_history))

        logger.info(
            f"Negotiated suite: KEM={kem.name}, SIG={sig.name}, "
            f"SYM={sym.name}, era={era.value}, "
            f"security={suite.security_level}bit"
        )

        return suite

    def renegotiate(
        self,
        current_suite: NegotiatedSuite,
        ban_algorithm: Optional[str] = None,
        upgrade_era: Optional[SecurityEra] = None,
    ) -> CryptoPolicy:
        """
        Trigger renegotiation by updating the local policy.

        Call this when a vulnerability is discovered or when
        migrating to a higher security era.

        Returns updated policy for use in new negotiation.
        """
        new_banned = set(self.policy.banned_algorithms)
        if ban_algorithm:
            new_banned.add(ban_algorithm)
            logger.warning(f"Algorithm {ban_algorithm} banned — renegotiation required")

        new_era = upgrade_era or self.policy.required_era

        self.policy = CryptoPolicy(
            minimum_security_bits=self.policy.minimum_security_bits,
            minimum_nist_level=self.policy.minimum_nist_level,
            required_era=new_era,
            banned_algorithms=frozenset(new_banned),
            preferred_kem=self.policy.preferred_kem,
            preferred_sig=self.policy.preferred_sig,
            preferred_symmetric=self.policy.preferred_symmetric,
            preferred_hash=self.policy.preferred_hash,
            require_hybrid=self.policy.require_hybrid,
            allow_classical_only=self.policy.allow_classical_only,
            cnsa2_required=self.policy.cnsa2_required,
        )

        AGILITY_NEGOTIATIONS.labels(result="renegotiation").inc()

        return self.policy

    def _filter_algorithms(self, family: AlgorithmFamily) -> List[AlgorithmDescriptor]:
        """Filter and sort algorithms for a family based on policy."""
        candidates = []

        for name, algo in self.registry.items():
            if algo.family != family:
                continue
            if name in self.policy.banned_algorithms:
                continue
            if algo.security_bits < self.policy.minimum_security_bits:
                continue
            if algo.status == AlgorithmStatus.BANNED:
                continue
            if self.policy.required_era and algo.era != self.policy.required_era:
                # Allow if it meets minimum era
                if not algo.is_post_quantum and self.policy.required_era != SecurityEra.CLASSICAL:
                    if not self.policy.allow_classical_only:
                        continue
            if self.policy.cnsa2_required and algo.era != SecurityEra.CNSA2:
                if algo.nist_level < 5:
                    continue

            candidates.append(algo)

        # Sort: preferred first, then by security bits descending
        preferred = self._get_preferred(family)

        def sort_key(a):
            pref_bonus = 0 if a.name == preferred else 1
            return (pref_bonus, -a.security_bits, a.output_size)

        candidates.sort(key=sort_key)
        return candidates

    def _get_preferred(self, family: AlgorithmFamily) -> str:
        prefs = {
            AlgorithmFamily.KEM: self.policy.preferred_kem,
            AlgorithmFamily.SIGNATURE: self.policy.preferred_sig,
            AlgorithmFamily.SYMMETRIC: self.policy.preferred_symmetric,
            AlgorithmFamily.HASH: self.policy.preferred_hash,
            AlgorithmFamily.KDF: "HKDF-SHA3-256",
            AlgorithmFamily.MAC: "HMAC-SHA3-256",
        }
        return prefs.get(family, "")

    def _select_best(
        self,
        local_prefs: List[str],
        remote_prefs: List[str],
        family: AlgorithmFamily,
    ) -> Optional[AlgorithmDescriptor]:
        """Select the best mutually supported algorithm."""
        remote_set = set(remote_prefs)

        for name in local_prefs:
            if name in remote_set and name in self.registry:
                algo = self.registry[name]
                if name not in self.policy.banned_algorithms:
                    return algo

        return None

    @property
    def negotiation_history(self) -> List[NegotiatedSuite]:
        return list(self._negotiation_history)
