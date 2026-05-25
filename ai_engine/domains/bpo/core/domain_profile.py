"""
BPO & Call Center Domain PQC Profiles

Post-quantum cryptography profiles optimized for different BPO/call center
subdomains with specific latency, throughput, and security requirements.

Key considerations for BPO/Call Center:
- Voice latency: <150ms end-to-end for real-time conversations
- Terminal sessions: Sub-second response for agent productivity
- High concurrency: 1000s of simultaneous agent sessions
- PCI-DSS: DTMF masking and call recording encryption
- Remote agents: VPN-less quantum-safe tunnels
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class BPOSubdomain(Enum):
    """BPO/Call Center system subdomains with specific requirements."""

    # Voice & Telephony
    VOICE_SIGNALING = auto()       # SIP/SDP signaling security
    VOICE_MEDIA = auto()           # RTP/SRTP media stream encryption
    IVR_SYSTEMS = auto()           # Interactive Voice Response
    CALL_RECORDING = auto()        # Call recording encryption & storage

    # Agent Operations
    AGENT_DESKTOP = auto()         # Agent desktop application security
    TERMINAL_EMULATION = auto()    # TN3270e/TN5250 mainframe access
    SCREEN_RECORDING = auto()      # Screen capture for QA compliance

    # Contact Center Infrastructure
    PBX_INTEGRATION = auto()       # PBX/ACD system connectivity
    CTI_MIDDLEWARE = auto()         # Computer Telephony Integration
    WFM_SYSTEMS = auto()           # Workforce Management

    # Customer Data
    CRM_INTEGRATION = auto()       # CRM system connectivity
    PAYMENT_PROCESSING = auto()    # PCI-DSS payment in voice channels
    CUSTOMER_DATA = auto()         # PII/PHI data handling

    # Remote Operations
    REMOTE_AGENT = auto()          # Work-from-home agent security
    VPN_LESS_ACCESS = auto()       # Quantum-safe direct tunnels

    # Analytics & Compliance
    QUALITY_MONITORING = auto()    # Quality assurance and monitoring
    SPEECH_ANALYTICS = auto()      # Real-time speech analytics
    COMPLIANCE_RECORDING = auto()  # Regulatory recording requirements


class PQCAlgorithm(Enum):
    """Post-quantum cryptographic algorithms for BPO domain."""

    # Key Encapsulation Mechanisms (NIST FIPS 203)
    ML_KEM_512 = ("ML-KEM-512", "kem", 128, 800, 768)
    ML_KEM_768 = ("ML-KEM-768", "kem", 192, 1184, 1088)
    ML_KEM_1024 = ("ML-KEM-1024", "kem", 256, 1568, 1568)

    # Digital Signatures (NIST FIPS 204 - Dilithium)
    ML_DSA_44 = ("ML-DSA-44", "signature", 128, 1312, 2420)
    ML_DSA_65 = ("ML-DSA-65", "signature", 192, 1952, 3293)
    ML_DSA_87 = ("ML-DSA-87", "signature", 256, 2592, 4595)

    # Falcon (compact signatures - good for bandwidth-constrained voice)
    FALCON_512 = ("Falcon-512", "signature", 128, 897, 690)
    FALCON_1024 = ("Falcon-1024", "signature", 256, 1793, 1330)

    # Hybrid schemes
    X25519_ML_KEM_768 = ("X25519-ML-KEM-768", "hybrid_kem", 192, 1216, 1120)
    P384_ML_KEM_1024 = ("P384-ML-KEM-1024", "hybrid_kem", 256, 1665, 1665)
    ED25519_ML_DSA_65 = ("Ed25519-ML-DSA-65", "hybrid_sig", 192, 1984, 3357)

    def __init__(self, name: str, algo_type: str, security_bits: int, pk_size: int, secondary_size: int):
        self.algo_name = name
        self.algo_type = algo_type
        self.security_bits = security_bits
        self.public_key_size = pk_size
        self.secondary_size = secondary_size

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
class BPOSecurityConstraints:
    """Security constraints for BPO/call center subdomain."""

    # Performance constraints
    max_latency_ms: float = 150.0          # Voice latency budget
    target_latency_ms: float = 50.0        # Target for agent experience
    throughput_sessions: int = 5000        # Concurrent agent sessions
    max_message_size_kb: int = 32          # SIP/CTI message size limit

    # Voice-specific constraints
    voice_codec_overhead_ms: float = 20.0  # Max PQC overhead on voice path
    rtp_packet_interval_ms: float = 20.0   # RTP packet interval (20ms typical)
    jitter_buffer_ms: float = 60.0         # Acceptable jitter buffer
    max_voice_latency_ms: float = 150.0    # ITU-T G.114 recommendation

    # Cryptographic constraints
    require_fips: bool = True              # FIPS 140-3 required
    require_pqc: bool = True               # PQC mandatory
    allow_hybrid: bool = True              # Classical + PQC hybrid allowed
    min_security_bits: int = 128           # Minimum security level
    allow_software_crypto: bool = True     # Software crypto OK for most BPO ops

    # Compliance constraints
    pci_dss_level: int = 1                 # PCI-DSS SAQ level
    require_call_recording: bool = True    # Mandatory call recording
    require_dtmf_masking: bool = True      # DTMF masking for PCI
    require_audit: bool = True             # Audit trail required
    data_residency: Optional[str] = None   # Data sovereignty
    retention_years: int = 7               # Call recording retention

    # Availability constraints
    availability_target: float = 0.9999    # Four nines
    max_recovery_time_minutes: int = 5     # RTO for voice services
    max_data_loss_seconds: int = 0         # RPO for recordings

    def validate(self) -> List[str]:
        """Validate constraints consistency."""
        errors = []

        if self.min_security_bits < 128:
            errors.append("Minimum security bits must be at least 128 for BPO")

        if self.voice_codec_overhead_ms > self.max_voice_latency_ms:
            errors.append("PQC overhead cannot exceed max voice latency")

        if self.max_latency_ms < self.target_latency_ms:
            errors.append("Target latency cannot exceed maximum latency")

        if self.pci_dss_level == 1 and not self.require_dtmf_masking:
            errors.append("PCI-DSS Level 1 requires DTMF masking for payment calls")

        if self.require_call_recording and self.retention_years < 1:
            errors.append("Call recording retention must be at least 1 year")

        return errors


@dataclass
class BPOPQCProfile:
    """PQC algorithm selection for BPO/call center subdomain."""

    subdomain: BPOSubdomain
    constraints: BPOSecurityConstraints

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
    key_rotation_hours: int = 8           # Per shift rotation for BPO

    # Additional settings
    use_hybrid_mode: bool = True
    allow_algorithm_negotiation: bool = True
    strict_mode: bool = True

    def __post_init__(self):
        """Validate profile configuration."""
        errors = self.constraints.validate()
        if errors:
            logger.warning(f"BPO profile validation warnings: {errors}")

    @classmethod
    def for_subdomain(cls, subdomain: BPOSubdomain) -> "BPOPQCProfile":
        """Get optimized profile for BPO subdomain."""

        profiles = {
            # Voice signaling - low latency, moderate security
            BPOSubdomain.VOICE_SIGNALING: cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(
                    max_latency_ms=50.0,
                    target_latency_ms=10.0,
                    throughput_sessions=10000,
                    voice_codec_overhead_ms=5.0,
                    max_voice_latency_ms=150.0,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_512,    # Fast for signaling
                sig_algorithm=PQCAlgorithm.FALCON_512,     # Compact signatures
                use_hybrid_mode=True,
                key_rotation_hours=1,                       # Frequent for voice
            ),

            # Voice media - ultra-low latency, per-packet encryption
            BPOSubdomain.VOICE_MEDIA: cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(
                    max_latency_ms=20.0,
                    target_latency_ms=5.0,
                    throughput_sessions=10000,
                    voice_codec_overhead_ms=2.0,
                    rtp_packet_interval_ms=20.0,
                    allow_software_crypto=True,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_512,    # Fastest KEM
                sig_algorithm=PQCAlgorithm.FALCON_512,     # Fastest verify
                use_hybrid_mode=False,                     # Pure PQC for speed
                key_rotation_hours=1,
                allow_algorithm_negotiation=False,
            ),

            # IVR systems - moderate latency, automated processing
            BPOSubdomain.IVR_SYSTEMS: cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(
                    max_latency_ms=200.0,
                    target_latency_ms=50.0,
                    throughput_sessions=5000,
                    require_dtmf_masking=True,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                use_hybrid_mode=True,
            ),

            # Call recording - high security, batch tolerant
            BPOSubdomain.CALL_RECORDING: cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(
                    max_latency_ms=1000.0,
                    target_latency_ms=200.0,
                    throughput_sessions=5000,
                    require_call_recording=True,
                    retention_years=7,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_1024,   # Max security for recordings
                sig_algorithm=PQCAlgorithm.ML_DSA_87,
                hybrid_kem=PQCAlgorithm.P384_ML_KEM_1024,
                use_hybrid_mode=True,
                strict_mode=True,
                key_rotation_hours=24,
            ),

            # Agent desktop - balanced latency/security
            BPOSubdomain.AGENT_DESKTOP: cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(
                    max_latency_ms=200.0,
                    target_latency_ms=50.0,
                    throughput_sessions=10000,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                use_hybrid_mode=True,
                key_rotation_hours=8,                      # Per shift
            ),

            # Terminal emulation - mainframe access security
            BPOSubdomain.TERMINAL_EMULATION: cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(
                    max_latency_ms=300.0,
                    target_latency_ms=100.0,
                    throughput_sessions=2000,
                    require_fips=True,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                use_hybrid_mode=True,
                strict_mode=True,
            ),

            # Payment processing - maximum security for PCI
            BPOSubdomain.PAYMENT_PROCESSING: cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(
                    max_latency_ms=100.0,
                    target_latency_ms=20.0,
                    throughput_sessions=3000,
                    pci_dss_level=1,
                    require_dtmf_masking=True,
                    require_fips=True,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_1024,
                sig_algorithm=PQCAlgorithm.ML_DSA_87,
                hybrid_kem=PQCAlgorithm.P384_ML_KEM_1024,
                use_hybrid_mode=True,
                strict_mode=True,
            ),

            # Remote agent - VPN-less quantum-safe tunnels
            BPOSubdomain.REMOTE_AGENT: cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(
                    max_latency_ms=200.0,
                    target_latency_ms=50.0,
                    throughput_sessions=20000,
                    allow_software_crypto=True,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                hybrid_kem=PQCAlgorithm.X25519_ML_KEM_768,
                use_hybrid_mode=True,
                key_rotation_hours=4,                      # More frequent for remote
            ),

            # Quality monitoring - recording and analytics
            BPOSubdomain.QUALITY_MONITORING: cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(
                    max_latency_ms=500.0,
                    target_latency_ms=200.0,
                    throughput_sessions=1000,
                    require_call_recording=True,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                use_hybrid_mode=True,
            ),

            # CRM integration - customer data protection
            BPOSubdomain.CRM_INTEGRATION: cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(
                    max_latency_ms=300.0,
                    target_latency_ms=100.0,
                    throughput_sessions=5000,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                use_hybrid_mode=True,
            ),

            # CTI middleware - telephony-computer bridge
            BPOSubdomain.CTI_MIDDLEWARE: cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(
                    max_latency_ms=100.0,
                    target_latency_ms=20.0,
                    throughput_sessions=10000,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_512,
                sig_algorithm=PQCAlgorithm.FALCON_512,
                use_hybrid_mode=True,
                key_rotation_hours=4,
            ),

            # Speech analytics - real-time processing
            BPOSubdomain.SPEECH_ANALYTICS: cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(
                    max_latency_ms=500.0,
                    target_latency_ms=200.0,
                    throughput_sessions=2000,
                    allow_software_crypto=True,
                ),
                kem_algorithm=PQCAlgorithm.ML_KEM_768,
                sig_algorithm=PQCAlgorithm.ML_DSA_65,
                use_hybrid_mode=True,
            ),
        }

        # Return specific profile or default
        return profiles.get(
            subdomain,
            cls(
                subdomain=subdomain,
                constraints=BPOSecurityConstraints(),
            ),
        )

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
                "throughput_sessions": self.constraints.throughput_sessions,
                "max_voice_latency_ms": self.constraints.max_voice_latency_ms,
                "require_fips": self.constraints.require_fips,
                "require_pqc": self.constraints.require_pqc,
                "require_dtmf_masking": self.constraints.require_dtmf_masking,
                "require_call_recording": self.constraints.require_call_recording,
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


# Pre-configured profiles for common BPO use cases
BPO_PROFILES: Dict[str, BPOPQCProfile] = {
    "voice_signaling": BPOPQCProfile.for_subdomain(BPOSubdomain.VOICE_SIGNALING),
    "voice_media": BPOPQCProfile.for_subdomain(BPOSubdomain.VOICE_MEDIA),
    "ivr_systems": BPOPQCProfile.for_subdomain(BPOSubdomain.IVR_SYSTEMS),
    "call_recording": BPOPQCProfile.for_subdomain(BPOSubdomain.CALL_RECORDING),
    "agent_desktop": BPOPQCProfile.for_subdomain(BPOSubdomain.AGENT_DESKTOP),
    "terminal_emulation": BPOPQCProfile.for_subdomain(BPOSubdomain.TERMINAL_EMULATION),
    "payment_processing": BPOPQCProfile.for_subdomain(BPOSubdomain.PAYMENT_PROCESSING),
    "remote_agent": BPOPQCProfile.for_subdomain(BPOSubdomain.REMOTE_AGENT),
    "quality_monitoring": BPOPQCProfile.for_subdomain(BPOSubdomain.QUALITY_MONITORING),
    "crm_integration": BPOPQCProfile.for_subdomain(BPOSubdomain.CRM_INTEGRATION),
    "cti_middleware": BPOPQCProfile.for_subdomain(BPOSubdomain.CTI_MIDDLEWARE),
    "speech_analytics": BPOPQCProfile.for_subdomain(BPOSubdomain.SPEECH_ANALYTICS),
}


def get_profile(name: str) -> Optional[BPOPQCProfile]:
    """Get a pre-configured BPO profile by name."""
    return BPO_PROFILES.get(name)


def list_profiles() -> List[str]:
    """List available pre-configured BPO profiles."""
    return list(BPO_PROFILES.keys())
