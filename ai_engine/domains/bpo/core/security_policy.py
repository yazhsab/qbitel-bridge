"""
BPO & Call Center Security Policy Module

Defines security policies, data classification, and compliance
requirements for BPO and contact center operations.

Key BPO-specific considerations:
- Voice channel PCI-DSS compliance (DTMF masking, pause/resume recording)
- Agent session security (screen lock, idle timeout, clipboard blocking)
- Remote agent endpoint security (VPN-less, eBPF monitoring)
- Customer PII/PHI data protection during calls
- Toll fraud prevention and detection
- Multi-tenant isolation for BPO clients
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for BPO operations."""

    STANDARD = (1, "Standard", 128)       # Internal operations, non-customer data
    ENHANCED = (2, "Enhanced", 192)       # Customer interactions, CRM data
    CRITICAL = (3, "Critical", 256)       # Payment processing, financial data
    RESTRICTED = (4, "Restricted", 256)   # Healthcare PHI, government contracts

    def __init__(self, level: int, display_name: str, min_key_bits: int):
        self.level = level
        self.display_name = display_name
        self.min_key_bits = min_key_bits


class DataClassification(Enum):
    """Data sensitivity classification for BPO data."""

    PUBLIC = (0, "Public", False, False, 0)
    INTERNAL = (1, "Internal", False, False, 1)
    CONFIDENTIAL = (2, "Confidential", True, False, 3)
    RESTRICTED = (3, "Restricted", True, True, 7)
    TOP_SECRET = (4, "Top Secret", True, True, 10)

    def __init__(self, level: int, display_name: str, requires_encryption: bool,
                 requires_hsm: bool, min_retention_years: int):
        self.level = level
        self.display_name = display_name
        self.requires_encryption = requires_encryption
        self.requires_hsm = requires_hsm
        self.min_retention_years = min_retention_years


class BPOComplianceRequirement(Enum):
    """Compliance framework requirements specific to BPO/Call Centers."""

    # Payment
    PCI_DSS_4_0 = ("PCI-DSS 4.0", "Payment Card Industry Data Security Standard")

    # Telephony
    TCPA = ("TCPA", "Telephone Consumer Protection Act")
    TSR = ("TSR", "Telemarketing Sales Rule")

    # Data Protection
    GDPR = ("GDPR", "General Data Protection Regulation")
    CCPA = ("CCPA", "California Consumer Privacy Act")

    # Healthcare BPO
    HIPAA = ("HIPAA", "Health Insurance Portability and Accountability Act")
    HITECH = ("HITECH", "Health Information Technology for Economic and Clinical Health Act")

    # Financial Services BPO
    SOX = ("SOX", "Sarbanes-Oxley Act")
    GLBA = ("GLBA", "Gramm-Leach-Bliley Act")
    DORA = ("DORA", "Digital Operational Resilience Act")
    FCA = ("FCA", "Financial Conduct Authority Regulations")
    MIFID_II = ("MiFID II", "Markets in Financial Instruments Directive")

    # General Security
    SOC_2 = ("SOC 2", "Service Organization Control 2")
    ISO_27001 = ("ISO 27001", "Information Security Management System")
    NIST_800_53 = ("NIST 800-53", "Security and Privacy Controls")
    NIST_PQC = ("NIST PQC", "Post-Quantum Cryptography Standards")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


@dataclass
class VoiceSecurityPolicy:
    """Voice channel specific security settings."""

    # Call recording
    record_all_calls: bool = True
    encrypt_recordings: bool = True
    recording_encryption_algorithm: str = "AES-256-GCM"
    recording_retention_days: int = 2555              # 7 years default

    # DTMF masking (PCI-DSS)
    enable_dtmf_masking: bool = True
    dtmf_masking_mode: str = "CLAMP"                 # CLAMP, FLAT, REPLACE
    mask_dtmf_in_recording: bool = True
    mask_dtmf_in_screen: bool = True

    # Pause/Resume for PCI
    enable_pause_resume: bool = True
    auto_pause_on_payment: bool = True
    auto_resume_timeout_seconds: int = 120

    # Voice encryption
    encrypt_voice_media: bool = True
    srtp_profile: str = "SRTP-PQC-AES256-GCM"
    require_srtp: bool = True

    # Toll fraud prevention
    enable_toll_fraud_detection: bool = True
    max_call_duration_hours: int = 4
    international_call_restrictions: List[str] = field(default_factory=list)
    premium_rate_blocking: bool = True

    # Voice biometrics
    enable_voice_biometrics: bool = False
    biometric_enrollment_required: bool = False


@dataclass
class AgentSecurityPolicy:
    """Agent desktop and session security settings."""

    # Authentication
    require_mfa: bool = True
    mfa_methods: Set[str] = field(default_factory=lambda: {"totp", "hardware_token", "biometric"})

    # Session management
    max_session_duration_hours: int = 10            # Typical BPO shift
    idle_timeout_minutes: int = 5                    # Aggressive for PCI
    concurrent_sessions_allowed: int = 1
    session_lock_on_idle: bool = True

    # Desktop security
    block_clipboard: bool = True                     # Prevent copy/paste of PII
    block_screen_capture: bool = True                # Prevent screenshots
    block_usb_storage: bool = True                   # Prevent data exfiltration
    block_personal_email: bool = True                # Prevent email exfiltration
    block_cloud_storage: bool = True                 # Prevent cloud uploads
    watermark_screen: bool = True                    # Forensic watermarking

    # Data handling
    mask_pii_on_screen: bool = True                  # Show only last 4 digits
    mask_credit_card: bool = True
    mask_ssn: bool = True
    auto_clear_after_call: bool = True               # Clear screen after each call

    # Remote agent specific
    require_endpoint_compliance: bool = True
    endpoint_os_requirements: List[str] = field(
        default_factory=lambda: ["Windows 10+", "macOS 12+"]
    )
    require_antivirus: bool = True
    require_disk_encryption: bool = True
    home_network_requirements: str = "WPA3"


@dataclass
class EncryptionPolicy:
    """Encryption policy for BPO operations."""

    # Data at rest
    encrypt_at_rest: bool = True
    at_rest_algorithm: str = "AES-256-GCM"
    at_rest_key_rotation_days: int = 90

    # Data in transit
    encrypt_in_transit: bool = True
    min_tls_version: str = "TLS 1.3"
    allowed_cipher_suites: List[str] = field(
        default_factory=lambda: [
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
        ]
    )

    # PQC settings
    require_pqc: bool = True
    pqc_kem_algorithm: str = "ML-KEM-768"
    pqc_sig_algorithm: str = "ML-DSA-65"
    hybrid_mode: bool = True

    # Voice-specific encryption
    voice_encryption: str = "SRTP-PQC"
    signaling_encryption: str = "SIP-TLS-PQC"

    # Key management
    key_derivation_function: str = "HKDF-SHA3-256"
    min_key_length_bits: int = 256


@dataclass
class AuditPolicy:
    """Audit logging policy for BPO operations."""

    # What to log
    log_authentication: bool = True
    log_authorization: bool = True
    log_data_access: bool = True
    log_data_modification: bool = True
    log_call_events: bool = True                     # Call start/end/transfer
    log_agent_actions: bool = True                   # Agent screen actions
    log_dtmf_events: bool = True                     # DTMF masking events (masked)
    log_recording_events: bool = True                # Pause/resume events

    # Sensitive data handling in logs
    mask_pii: bool = True
    mask_pan: bool = True
    mask_phone_numbers: bool = True
    mask_account_numbers: bool = True
    hash_sensitive_ids: bool = True

    # Retention
    retention_days: int = 2555                       # 7 years
    archive_after_days: int = 90

    # Integrity
    tamper_evident: bool = True
    cryptographic_binding: bool = True
    real_time_alerting: bool = True


@dataclass
class MultiTenantPolicy:
    """Multi-tenant isolation for BPO clients."""

    # Isolation
    require_tenant_isolation: bool = True
    data_isolation_level: str = "STRICT"             # STRICT, LOGICAL, SHARED
    network_isolation: bool = True
    separate_encryption_keys: bool = True

    # Tenant configuration
    tenant_id: str = ""
    tenant_name: str = ""
    tenant_security_level: SecurityLevel = SecurityLevel.ENHANCED
    tenant_compliance: Set[BPOComplianceRequirement] = field(default_factory=set)

    # SLA settings
    max_agents: int = 0                              # 0 = unlimited
    max_concurrent_calls: int = 0
    recording_storage_gb: int = 0


@dataclass
class BPOSecurityPolicy:
    """
    Comprehensive security policy for BPO/call center operations.

    Combines voice security, agent security, encryption, audit,
    multi-tenancy, and compliance requirements.
    """

    # Policy identification
    policy_id: str = ""
    policy_name: str = ""
    version: str = "1.0"
    effective_date: datetime = field(default_factory=datetime.now)
    review_date: Optional[datetime] = None

    # Security configuration
    security_level: SecurityLevel = SecurityLevel.ENHANCED
    data_classification: DataClassification = DataClassification.CONFIDENTIAL

    # Compliance requirements
    compliance_frameworks: Set[BPOComplianceRequirement] = field(
        default_factory=lambda: {
            BPOComplianceRequirement.PCI_DSS_4_0,
            BPOComplianceRequirement.GDPR,
            BPOComplianceRequirement.SOC_2,
        }
    )

    # Sub-policies
    voice_security: VoiceSecurityPolicy = field(default_factory=VoiceSecurityPolicy)
    agent_security: AgentSecurityPolicy = field(default_factory=AgentSecurityPolicy)
    encryption: EncryptionPolicy = field(default_factory=EncryptionPolicy)
    audit: AuditPolicy = field(default_factory=AuditPolicy)
    multi_tenant: MultiTenantPolicy = field(default_factory=MultiTenantPolicy)

    # Risk settings
    risk_tolerance: str = "LOW"
    incident_response_sla_minutes: int = 15
    escalation_contacts: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Set review date if not specified."""
        if self.review_date is None:
            self.review_date = self.effective_date + timedelta(days=365)

    def validate(self) -> List[str]:
        """Validate policy configuration."""
        errors = []

        # Check security level matches data classification
        if (
            self.data_classification == DataClassification.TOP_SECRET
            and self.security_level.level < SecurityLevel.CRITICAL.level
        ):
            errors.append("Top Secret data requires Critical security level")

        # Check encryption requirements
        if self.data_classification.requires_encryption and not self.encryption.encrypt_at_rest:
            errors.append(f"{self.data_classification.display_name} data requires encryption at rest")

        # PCI-DSS voice requirements
        if BPOComplianceRequirement.PCI_DSS_4_0 in self.compliance_frameworks:
            if not self.voice_security.enable_dtmf_masking:
                errors.append("PCI-DSS requires DTMF masking for payment calls")
            if not self.voice_security.encrypt_recordings:
                errors.append("PCI-DSS requires encrypted call recordings")
            if not self.encryption.encrypt_in_transit:
                errors.append("PCI-DSS requires encryption in transit")
            if not self.agent_security.mask_credit_card:
                errors.append("PCI-DSS requires credit card masking on agent screens")
            if not self.audit.log_data_access:
                errors.append("PCI-DSS requires logging of data access")

        # TCPA requirements
        if BPOComplianceRequirement.TCPA in self.compliance_frameworks:
            if not self.audit.log_call_events:
                errors.append("TCPA requires call event logging for consent tracking")

        # HIPAA requirements
        if BPOComplianceRequirement.HIPAA in self.compliance_frameworks:
            if not self.encryption.encrypt_at_rest:
                errors.append("HIPAA requires encryption of PHI at rest")
            if not self.agent_security.block_clipboard:
                errors.append("HIPAA recommends clipboard blocking for PHI protection")
            if self.audit.retention_days < 2190:  # 6 years
                errors.append("HIPAA requires minimum 6 year audit retention")

        # SOC 2 requirements
        if BPOComplianceRequirement.SOC_2 in self.compliance_frameworks:
            if not self.audit.real_time_alerting:
                errors.append("SOC 2 requires real-time security alerting")

        return errors

    @classmethod
    def for_financial_services_bpo(cls) -> "BPOSecurityPolicy":
        """Create policy for financial services BPO."""
        return cls(
            policy_id="BPO-FIN-001",
            policy_name="Financial Services BPO Security Policy",
            security_level=SecurityLevel.CRITICAL,
            data_classification=DataClassification.RESTRICTED,
            compliance_frameworks={
                BPOComplianceRequirement.PCI_DSS_4_0,
                BPOComplianceRequirement.SOX,
                BPOComplianceRequirement.GLBA,
                BPOComplianceRequirement.GDPR,
                BPOComplianceRequirement.SOC_2,
                BPOComplianceRequirement.NIST_PQC,
            },
            voice_security=VoiceSecurityPolicy(
                record_all_calls=True,
                enable_dtmf_masking=True,
                enable_pause_resume=True,
                enable_toll_fraud_detection=True,
                premium_rate_blocking=True,
            ),
            agent_security=AgentSecurityPolicy(
                require_mfa=True,
                idle_timeout_minutes=3,
                block_clipboard=True,
                block_screen_capture=True,
                mask_credit_card=True,
                auto_clear_after_call=True,
            ),
            encryption=EncryptionPolicy(
                require_pqc=True,
                hybrid_mode=True,
                pqc_kem_algorithm="ML-KEM-1024",
                pqc_sig_algorithm="ML-DSA-87",
            ),
            audit=AuditPolicy(
                mask_pan=True,
                tamper_evident=True,
                retention_days=3650,  # 10 years for financial
            ),
            risk_tolerance="LOW",
            incident_response_sla_minutes=10,
        )

    @classmethod
    def for_healthcare_bpo(cls) -> "BPOSecurityPolicy":
        """Create policy for healthcare BPO."""
        return cls(
            policy_id="BPO-HC-001",
            policy_name="Healthcare BPO Security Policy",
            security_level=SecurityLevel.CRITICAL,
            data_classification=DataClassification.RESTRICTED,
            compliance_frameworks={
                BPOComplianceRequirement.HIPAA,
                BPOComplianceRequirement.HITECH,
                BPOComplianceRequirement.GDPR,
                BPOComplianceRequirement.SOC_2,
                BPOComplianceRequirement.NIST_PQC,
            },
            voice_security=VoiceSecurityPolicy(
                record_all_calls=True,
                encrypt_recordings=True,
                enable_voice_biometrics=True,
            ),
            agent_security=AgentSecurityPolicy(
                require_mfa=True,
                idle_timeout_minutes=5,
                block_clipboard=True,
                block_screen_capture=True,
                mask_pii_on_screen=True,
                mask_ssn=True,
                require_endpoint_compliance=True,
            ),
            encryption=EncryptionPolicy(
                require_pqc=True,
                hybrid_mode=True,
                pqc_kem_algorithm="ML-KEM-1024",
                pqc_sig_algorithm="ML-DSA-87",
            ),
            audit=AuditPolicy(
                retention_days=2190,  # 6 years for HIPAA
                tamper_evident=True,
                real_time_alerting=True,
            ),
            risk_tolerance="LOW",
            incident_response_sla_minutes=15,
        )

    @classmethod
    def for_general_customer_service(cls) -> "BPOSecurityPolicy":
        """Create policy for general customer service BPO."""
        return cls(
            policy_id="BPO-CS-001",
            policy_name="General Customer Service BPO Security Policy",
            security_level=SecurityLevel.ENHANCED,
            data_classification=DataClassification.CONFIDENTIAL,
            compliance_frameworks={
                BPOComplianceRequirement.GDPR,
                BPOComplianceRequirement.SOC_2,
                BPOComplianceRequirement.NIST_PQC,
            },
            voice_security=VoiceSecurityPolicy(
                record_all_calls=True,
                enable_toll_fraud_detection=True,
            ),
            agent_security=AgentSecurityPolicy(
                require_mfa=True,
                idle_timeout_minutes=10,
                mask_pii_on_screen=True,
            ),
            encryption=EncryptionPolicy(
                require_pqc=True,
                hybrid_mode=True,
            ),
            risk_tolerance="MEDIUM",
            incident_response_sla_minutes=30,
        )

    @classmethod
    def for_remote_workforce(cls) -> "BPOSecurityPolicy":
        """Create policy for remote/work-from-home BPO agents."""
        return cls(
            policy_id="BPO-REMOTE-001",
            policy_name="Remote Workforce BPO Security Policy",
            security_level=SecurityLevel.ENHANCED,
            data_classification=DataClassification.CONFIDENTIAL,
            compliance_frameworks={
                BPOComplianceRequirement.GDPR,
                BPOComplianceRequirement.SOC_2,
                BPOComplianceRequirement.NIST_PQC,
            },
            agent_security=AgentSecurityPolicy(
                require_mfa=True,
                mfa_methods={"totp", "biometric"},
                idle_timeout_minutes=3,
                block_clipboard=True,
                block_usb_storage=True,
                block_personal_email=True,
                block_cloud_storage=True,
                watermark_screen=True,
                require_endpoint_compliance=True,
                require_antivirus=True,
                require_disk_encryption=True,
                home_network_requirements="WPA3",
            ),
            encryption=EncryptionPolicy(
                require_pqc=True,
                hybrid_mode=True,
                voice_encryption="SRTP-PQC",
                signaling_encryption="SIP-TLS-PQC",
            ),
            risk_tolerance="LOW",
            incident_response_sla_minutes=15,
        )

    def to_dict(self) -> Dict:
        """Convert policy to dictionary."""
        return {
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "version": self.version,
            "effective_date": self.effective_date.isoformat(),
            "security_level": self.security_level.display_name,
            "data_classification": self.data_classification.display_name,
            "compliance_frameworks": [f.code for f in self.compliance_frameworks],
            "voice_security": {
                "dtmf_masking": self.voice_security.enable_dtmf_masking,
                "recording_encrypted": self.voice_security.encrypt_recordings,
                "toll_fraud_detection": self.voice_security.enable_toll_fraud_detection,
                "srtp_profile": self.voice_security.srtp_profile,
            },
            "agent_security": {
                "require_mfa": self.agent_security.require_mfa,
                "clipboard_blocked": self.agent_security.block_clipboard,
                "screen_capture_blocked": self.agent_security.block_screen_capture,
                "pii_masked": self.agent_security.mask_pii_on_screen,
            },
            "encryption": {
                "require_pqc": self.encryption.require_pqc,
                "hybrid_mode": self.encryption.hybrid_mode,
                "kem_algorithm": self.encryption.pqc_kem_algorithm,
                "sig_algorithm": self.encryption.pqc_sig_algorithm,
            },
            "risk_tolerance": self.risk_tolerance,
        }


# Pre-defined BPO policies
BPO_POLICIES: Dict[str, BPOSecurityPolicy] = {
    "financial_services": BPOSecurityPolicy.for_financial_services_bpo(),
    "healthcare": BPOSecurityPolicy.for_healthcare_bpo(),
    "general_customer_service": BPOSecurityPolicy.for_general_customer_service(),
    "remote_workforce": BPOSecurityPolicy.for_remote_workforce(),
}


def get_policy(name: str) -> Optional[BPOSecurityPolicy]:
    """Get a pre-defined BPO security policy."""
    return BPO_POLICIES.get(name)


def list_policies() -> List[str]:
    """List available pre-defined BPO policies."""
    return list(BPO_POLICIES.keys())
