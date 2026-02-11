"""
Banking Domain PQC Profiles

Post-quantum cryptography profiles optimized for different banking
subdomains with specific latency, throughput, and security requirements.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class BankingSubdomain(Enum):
    """Banking system subdomains with specific requirements."""

    # Payment Systems
    CARD_PAYMENTS = auto()          # ISO 8583, EMV, tokenization
    WIRE_TRANSFERS = auto()         # SWIFT, ISO 20022
    DOMESTIC_CLEARING = auto()      # ACH, FedWire, CHIPS
    REAL_TIME_PAYMENTS = auto()     # FedNow, RTP, SCT Inst
    CROSS_BORDER = auto()           # Correspondent banking

    # Capital Markets
    TRADING = auto()                # FIX protocol, order execution
    MARKET_DATA = auto()            # FAST, ITCH, real-time feeds
    POST_TRADE = auto()             # Settlement, clearing

    # Core Banking
    ACCOUNT_MANAGEMENT = auto()     # Core banking APIs
    LENDING = auto()                # Loan origination, servicing
    TREASURY = auto()               # Cash management, FX

    # Legacy Systems
    LEGACY_MAINFRAME = auto()       # z/OS, CICS, IMS
    LEGACY_MIDRANGE = auto()        # AS/400, iSeries

    # Security Infrastructure
    HSM_OPERATIONS = auto()         # Hardware security modules
    PKI_INFRASTRUCTURE = auto()     # Certificate management
    KEY_MANAGEMENT = auto()         # Key lifecycle

    # Customer Channels
    DIGITAL_BANKING = auto()        # Web, Mobile APIs
    ATM_NETWORK = auto()            # ATM encryption, PIN
    OPEN_BANKING = auto()           # PSD2, API banking


class PQCAlgorithm(Enum):
    """Post-quantum cryptographic algorithms."""

    # Key Encapsulation Mechanisms (NIST FIPS 203)
    ML_KEM_512 = ("ML-KEM-512", "kem", 128, 800, 768)      # name, type, security_bits, pk_size, ct_size
    ML_KEM_768 = ("ML-KEM-768", "kem", 192, 1184, 1088)
    ML_KEM_1024 = ("ML-KEM-1024", "kem", 256, 1568, 1568)

    # Digital Signatures (NIST FIPS 204 - Dilithium)
    ML_DSA_44 = ("ML-DSA-44", "signature", 128, 1312, 2420)   # pk_size, sig_size
    ML_DSA_65 = ("ML-DSA-65", "signature", 192, 1952, 3293)
    ML_DSA_87 = ("ML-DSA-87", "signature", 256, 2592, 4595)

    # Falcon (NIST selected, compact signatures)
    FALCON_512 = ("Falcon-512", "signature", 128, 897, 690)
    FALCON_1024 = ("Falcon-1024", "signature", 256, 1793, 1330)

    # SLH-DSA (SPHINCS+, hash-based)
    SLH_DSA_128F = ("SLH-DSA-SHA2-128f", "signature", 128, 32, 17088)
    SLH_DSA_256F = ("SLH-DSA-SHA2-256f", "signature", 256, 64, 49856)

    # Hybrid schemes
    X25519_ML_KEM_768 = ("X25519-ML-KEM-768", "hybrid_kem", 192, 1216, 1120)
    P384_ML_KEM_1024 = ("P384-ML-KEM-1024", "hybrid_kem", 256, 1665, 1665)
    ED25519_ML_DSA_65 = ("Ed25519-ML-DSA-65", "hybrid_sig", 192, 1984, 3357)

    def __init__(self, name: str, algo_type: str, security_bits: int,
                 pk_size: int, secondary_size: int):
        self.algo_name = name
        self.algo_type = algo_type
        self.security_bits = security_bits
        self.public_key_size = pk_size
        self.secondary_size = secondary_size  # ct_size for KEM, sig_size for signatures

    @property
    def is_hybrid(self) -> bool:
        return self.algo_type.startswith("hybrid")

    @property
    def is_kem(self) -> bool:
        return "kem" in self.algo_type

    @property
    def is_signature(self) -> bool:
        return "sig" in self.algo_type


@dataclass
class BankingSecurityConstraints:
    """Security constraints for banking subdomain."""

    # Performance constraints
    max_latency_ms: float = 100.0           # Maximum acceptable latency
    target_latency_ms: float = 10.0         # Target latency for optimization
    throughput_tps: int = 1000              # Target transactions per second
    max_message_size_kb: int = 64           # Maximum message size

    # Cryptographic constraints
    require_fips: bool = True               # FIPS 140-3 required
    require_pqc: bool = True                # PQC mandatory
    allow_hybrid: bool = True               # Classical + PQC hybrid allowed
    min_security_bits: int = 128            # Minimum security level
    allow_software_crypto: bool = False     # Allow software-only crypto

    # Compliance constraints
    pci_dss_level: int = 1                  # PCI-DSS SAQ level (1-4)
    require_hsm: bool = True                # HSM mandatory for key operations
    require_audit: bool = True              # Audit trail required
    data_residency: Optional[str] = None    # Data sovereignty requirements
    retention_years: int = 7                # Data retention period

    # Availability constraints
    availability_target: float = 0.9999     # Four nines
    max_recovery_time_minutes: int = 15     # RTO
    max_data_loss_seconds: int = 0          # RPO (zero for payment systems)

    def validate(self) -> List[str]:
        """Validate constraints consistency."""
        errors = []

        if self.min_security_bits < 128:
            errors.append("Minimum security bits must be at least 128 for banking")

        if self.require_fips and self.allow_software_crypto:
            errors.append("FIPS mode typically requires hardware crypto")

        if self.pci_dss_level == 1 and not self.require_hsm:
            errors.append("PCI-DSS Level 1 requires HSM for key management")

        if self.max_latency_ms < self.target_latency_ms:
            errors.append("Target latency cannot exceed maximum latency")

        return errors


@dataclass
class BankingPQCProfile:
    """PQC algorithm selection for banking subdomain."""

    subdomain: BankingSubdomain
    constraints: BankingSecurityConstraints

    # Key encapsulation
    kem_algorithm: PQCAlgorithm = PQCAlgorithm.ML_KEM_768
    kem_fallback: Optional[PQCAlgorithm] = PQCAlgorithm.ML_KEM_1024

    # Digital signatures
    sig_algorithm: PQCAlgorithm = PQCAlgorithm.ML_DSA_65
    sig_fallback: Optional[PQCAlgorithm] = PQCAlgorithm.ML_DSA_87

    # Hybrid mode algorithms
    hybrid_kem: Optional[PQCAlgorithm] = PQCAlgorithm.X25519_ML_KEM_768
    hybrid_sig: Optional[PQCAlgorithm] = PQCAlgorithm.ED25519_ML_DSA_65

    # Hash functions
    hash_algorithm: str = "SHA3-256"
    mac_algorithm: str = "HMAC-SHA3-256"

    # Session settings
    session_key_bits: int = 256
    key_rotation_hours: int = 24

    # Additional settings
    use_hybrid_mode: bool = True
    allow_algorithm_negotiation: bool = True
    strict_mode: bool = True  # Fail if PQC not available

    def __post_init__(self):
        """Validate profile configuration."""
        errors = self.constraints.validate()
        if errors:
            logger.warning(f"Profile validation warnings: {errors}")

    @classmethod
    def for_subdomain(cls, subdomain: BankingSubdomain) -> "BankingPQCProfile":
        """Get optimized profile for subdomain."""

        profiles = {
            # Real-time payments - ultra-low latency
            BankingSubdomain.REAL_TIME_PAYMENTS: cls(
                subdomain=subdomain,
                constraints=BankingSecurityConstraints(
                    max_latency_ms=50.0,
                    target_latency_ms=5.0,
                    throughput_tps=10000,
                    max_recovery_time_minutes=1,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_512,  # Faster
                sig_algorithm=PQCAlgorithm.FALCON_512,   # Compact, fast verify
                use_hybrid_mode=True,
                key_rotation_hours=1,  # Frequent rotation
            ),

            # Wire transfers - high security, batch tolerant
            BankingSubdomain.WIRE_TRANSFERS: cls(
                subdomain=subdomain,
                constraints=BankingSecurityConstraints(
                    max_latency_ms=500.0,
                    target_latency_ms=100.0,
                    throughput_tps=1000,
                    require_hsm=True,
                    pci_dss_level=1,
                    retention_years=10,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_1024,  # Maximum security
                sig_algorithm=PQCAlgorithm.ML_DSA_87,
                hybrid_kem=PQCAlgorithm.P384_ML_KEM_1024,
                use_hybrid_mode=True,
                strict_mode=True,
            ),

            # Card payments - balanced latency/security
            BankingSubdomain.CARD_PAYMENTS: cls(
                subdomain=subdomain,
                constraints=BankingSecurityConstraints(
                    max_latency_ms=100.0,
                    target_latency_ms=20.0,
                    throughput_tps=5000,
                    require_hsm=True,
                    pci_dss_level=1,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                use_hybrid_mode=True,
            ),

            # Trading - extreme low latency
            BankingSubdomain.TRADING: cls(
                subdomain=subdomain,
                constraints=BankingSecurityConstraints(
                    max_latency_ms=1.0,
                    target_latency_ms=0.1,
                    throughput_tps=100000,
                    allow_software_crypto=True,  # Hardware too slow
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_512,
                sig_algorithm=PQCAlgorithm.FALCON_512,  # Fastest verification
                use_hybrid_mode=False,  # Pure PQC for speed
                key_rotation_hours=1,
                allow_algorithm_negotiation=False,  # Pre-configured
            ),

            # HSM operations - maximum security
            BankingSubdomain.HSM_OPERATIONS: cls(
                subdomain=subdomain,
                constraints=BankingSecurityConstraints(
                    max_latency_ms=1000.0,  # HSM operations are slow
                    require_fips=True,
                    require_hsm=True,
                    min_security_bits=256,
                    allow_software_crypto=False,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_1024,
                sig_algorithm=PQCAlgorithm.ML_DSA_87,
                hybrid_kem=PQCAlgorithm.P384_ML_KEM_1024,
                use_hybrid_mode=True,
                strict_mode=True,
                key_rotation_hours=168,  # Weekly for HSM keys
            ),

            # ATM network - HSM-backed PIN security
            BankingSubdomain.ATM_NETWORK: cls(
                subdomain=subdomain,
                constraints=BankingSecurityConstraints(
                    max_latency_ms=200.0,
                    target_latency_ms=50.0,
                    throughput_tps=500,
                    require_hsm=True,
                    pci_dss_level=1,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                use_hybrid_mode=True,
            ),

            # Domestic clearing - batch processing
            BankingSubdomain.DOMESTIC_CLEARING: cls(
                subdomain=subdomain,
                constraints=BankingSecurityConstraints(
                    max_latency_ms=1000.0,
                    target_latency_ms=200.0,
                    throughput_tps=10000,  # Batch throughput
                    require_hsm=True,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                use_hybrid_mode=True,
            ),

            # Open Banking - API security
            BankingSubdomain.OPEN_BANKING: cls(
                subdomain=subdomain,
                constraints=BankingSecurityConstraints(
                    max_latency_ms=200.0,
                    target_latency_ms=50.0,
                    throughput_tps=2000,
                    require_hsm=False,  # Can use software for API
                    allow_software_crypto=True,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                hybrid_kem=PQCAlgorithm.X25519_ML_KEM_768,
                use_hybrid_mode=True,
            ),

            # Digital banking - web/mobile
            BankingSubdomain.DIGITAL_BANKING: cls(
                subdomain=subdomain,
                constraints=BankingSecurityConstraints(
                    max_latency_ms=300.0,
                    target_latency_ms=100.0,
                    throughput_tps=5000,
                    allow_software_crypto=True,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                use_hybrid_mode=True,
            ),

            # Legacy mainframe
            BankingSubdomain.LEGACY_MAINFRAME: cls(
                subdomain=subdomain,
                constraints=BankingSecurityConstraints(
                    max_latency_ms=500.0,
                    require_hsm=True,  # ICSF
                    require_fips=True,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                use_hybrid_mode=True,  # Bridge to modern systems
            ),

            # Market data - high throughput, integrity focus
            BankingSubdomain.MARKET_DATA: cls(
                subdomain=subdomain,
                constraints=BankingSecurityConstraints(
                    max_latency_ms=10.0,
                    target_latency_ms=1.0,
                    throughput_tps=500000,  # Very high throughput
                    allow_software_crypto=True,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_512,
                sig_algorithm=PQCAlgorithm.FALCON_512,  # Fast batch verify
                use_hybrid_mode=False,
                key_rotation_hours=1,
            ),
        }

        # Return specific profile or default
        return profiles.get(subdomain, cls(
            subdomain=subdomain,
            constraints=BankingSecurityConstraints(),
        ))

    def get_kem_for_security_level(self, min_bits: int) -> PQCAlgorithm:
        """Get appropriate KEM for security level."""
        if min_bits >= 256:
            return PQCAlgorithm.ML_KEM_1024
        elif min_bits >= 192:
            return PQCAlgorithm.ML_KEM_768
        else:
            return PQCAlgorithm.ML_KEM_512

    def get_sig_for_security_level(self, min_bits: int) -> PQCAlgorithm:
        """Get appropriate signature algorithm for security level."""
        if min_bits >= 256:
            return PQCAlgorithm.ML_DSA_87
        elif min_bits >= 192:
            return PQCAlgorithm.ML_DSA_65
        else:
            return PQCAlgorithm.ML_DSA_44

    def to_dict(self) -> Dict:
        """Convert profile to dictionary."""
        return {
            "subdomain": self.subdomain.name,
            "constraints": {
                "max_latency_ms": self.constraints.max_latency_ms,
                "target_latency_ms": self.constraints.target_latency_ms,
                "throughput_tps": self.constraints.throughput_tps,
                "require_fips": self.constraints.require_fips,
                "require_pqc": self.constraints.require_pqc,
                "require_hsm": self.constraints.require_hsm,
                "min_security_bits": self.constraints.min_security_bits,
            },
            "algorithms": {
                "kem": self.kem_algorithm.algo_name,
                "signature": self.sig_algorithm.algo_name,
                "hybrid_kem": self.hybrid_kem.algo_name if self.hybrid_kem else None,
                "hybrid_sig": self.hybrid_sig.algo_name if self.hybrid_sig else None,
                "hash": self.hash_algorithm,
            },
            "settings": {
                "use_hybrid_mode": self.use_hybrid_mode,
                "strict_mode": self.strict_mode,
                "key_rotation_hours": self.key_rotation_hours,
            },
        }


# Pre-configured profiles for common use cases
BANKING_PROFILES: Dict[str, BankingPQCProfile] = {
    "swift_messaging": BankingPQCProfile.for_subdomain(BankingSubdomain.WIRE_TRANSFERS),
    "real_time_payments": BankingPQCProfile.for_subdomain(BankingSubdomain.REAL_TIME_PAYMENTS),
    "card_processing": BankingPQCProfile.for_subdomain(BankingSubdomain.CARD_PAYMENTS),
    "trading_systems": BankingPQCProfile.for_subdomain(BankingSubdomain.TRADING),
    "hsm_operations": BankingPQCProfile.for_subdomain(BankingSubdomain.HSM_OPERATIONS),
    "domestic_clearing": BankingPQCProfile.for_subdomain(BankingSubdomain.DOMESTIC_CLEARING),
    "atm_network": BankingPQCProfile.for_subdomain(BankingSubdomain.ATM_NETWORK),
    "open_banking": BankingPQCProfile.for_subdomain(BankingSubdomain.OPEN_BANKING),
    "digital_banking": BankingPQCProfile.for_subdomain(BankingSubdomain.DIGITAL_BANKING),
    "market_data": BankingPQCProfile.for_subdomain(BankingSubdomain.MARKET_DATA),
    "mainframe": BankingPQCProfile.for_subdomain(BankingSubdomain.LEGACY_MAINFRAME),
}


def get_profile(name: str) -> Optional[BankingPQCProfile]:
    """Get a pre-configured banking profile by name."""
    return BANKING_PROFILES.get(name)


def list_profiles() -> List[str]:
    """List available pre-configured profiles."""
    return list(BANKING_PROFILES.keys())
