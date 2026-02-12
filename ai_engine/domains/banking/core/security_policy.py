"""
Banking Security Policy Module

Defines security policies, data classification, and compliance
requirements for banking operations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for banking operations."""

    # Standard security for internal operations
    STANDARD = (1, "Standard", 128)

    # Enhanced security for sensitive operations
    ENHANCED = (2, "Enhanced", 192)

    # Maximum security for critical operations
    CRITICAL = (3, "Critical", 256)

    # Restricted - highest security, limited access
    RESTRICTED = (4, "Restricted", 256)

    def __init__(self, level: int, display_name: str, min_key_bits: int):
        self.level = level
        self.display_name = display_name
        self.min_key_bits = min_key_bits


class DataClassification(Enum):
    """Data sensitivity classification for banking data."""

    # Public data - no restrictions
    PUBLIC = (0, "Public", False, False, 0)

    # Internal use only
    INTERNAL = (1, "Internal", False, False, 1)

    # Confidential business data
    CONFIDENTIAL = (2, "Confidential", True, False, 3)

    # Restricted - PII, financial data
    RESTRICTED = (3, "Restricted", True, True, 7)

    # Top Secret - HSM keys, master secrets
    TOP_SECRET = (4, "Top Secret", True, True, 10)

    def __init__(self, level: int, display_name: str, requires_encryption: bool, requires_hsm: bool, min_retention_years: int):
        self.level = level
        self.display_name = display_name
        self.requires_encryption = requires_encryption
        self.requires_hsm = requires_hsm
        self.min_retention_years = min_retention_years


class ComplianceRequirement(Enum):
    """Compliance framework requirements."""

    # Payment Card Industry
    PCI_DSS_4_0 = ("PCI-DSS 4.0", "Payment Card Industry Data Security Standard")

    # EU Digital Operational Resilience
    DORA = ("DORA", "Digital Operational Resilience Act")

    # Basel Banking Regulations
    BASEL_III = ("Basel III", "Basel III Capital Framework")
    BASEL_IV = ("Basel IV", "Basel IV Final Reforms")
    BCBS_239 = ("BCBS 239", "Risk Data Aggregation and Reporting")

    # Data Protection
    GDPR = ("GDPR", "General Data Protection Regulation")
    CCPA = ("CCPA", "California Consumer Privacy Act")

    # Financial Reporting
    SOX = ("SOX", "Sarbanes-Oxley Act")
    EMIR = ("EMIR", "European Market Infrastructure Regulation")
    MIFID_II = ("MiFID II", "Markets in Financial Instruments Directive")

    # US Banking
    GLBA = ("GLBA", "Gramm-Leach-Bliley Act")
    FFIEC = ("FFIEC", "Federal Financial Institutions Examination Council")

    # Quantum Readiness
    NIST_PQC = ("NIST PQC", "NIST Post-Quantum Cryptography Standards")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


@dataclass
class AccessControl:
    """Access control configuration."""

    # Authentication requirements
    require_mfa: bool = True
    mfa_methods: Set[str] = field(default_factory=lambda: {"totp", "hardware_token"})

    # Session management
    max_session_duration_hours: int = 8
    idle_timeout_minutes: int = 15
    concurrent_sessions_allowed: int = 1

    # Authorization
    require_approval_for_changes: bool = True
    dual_control_required: bool = False
    four_eyes_principle: bool = False

    # IP restrictions
    allowed_ip_ranges: List[str] = field(default_factory=list)
    deny_ip_ranges: List[str] = field(default_factory=list)

    # Time-based access
    allowed_hours_start: int = 6  # 6 AM
    allowed_hours_end: int = 22  # 10 PM
    allowed_days: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})  # Mon-Fri


@dataclass
class EncryptionPolicy:
    """Encryption policy configuration."""

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

    # Key management
    key_derivation_function: str = "HKDF-SHA3-256"
    min_key_length_bits: int = 256


@dataclass
class AuditPolicy:
    """Audit logging policy configuration."""

    # What to log
    log_authentication: bool = True
    log_authorization: bool = True
    log_data_access: bool = True
    log_data_modification: bool = True
    log_admin_actions: bool = True
    log_system_events: bool = True

    # Sensitive data handling in logs
    mask_pii: bool = True
    mask_pan: bool = True
    mask_account_numbers: bool = True
    hash_sensitive_ids: bool = True

    # Retention
    retention_days: int = 2555  # 7 years
    archive_after_days: int = 90

    # Integrity
    tamper_evident: bool = True
    cryptographic_binding: bool = True
    real_time_alerting: bool = True


@dataclass
class BankingSecurityPolicy:
    """
    Comprehensive security policy for banking operations.

    Combines security levels, data classification, access control,
    encryption, and audit requirements.
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
    compliance_frameworks: Set[ComplianceRequirement] = field(
        default_factory=lambda: {
            ComplianceRequirement.PCI_DSS_4_0,
            ComplianceRequirement.GDPR,
        }
    )

    # Sub-policies
    access_control: AccessControl = field(default_factory=AccessControl)
    encryption: EncryptionPolicy = field(default_factory=EncryptionPolicy)
    audit: AuditPolicy = field(default_factory=AuditPolicy)

    # Risk settings
    risk_tolerance: str = "LOW"  # LOW, MEDIUM, HIGH
    max_transaction_amount: Optional[float] = None
    velocity_limits: Dict[str, int] = field(default_factory=dict)

    # Incident response
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

        # Check HSM requirements
        if self.data_classification.requires_hsm and not self.encryption.require_pqc:
            errors.append(f"{self.data_classification.display_name} data requires PQC with HSM")

        # Check PCI-DSS requirements
        if ComplianceRequirement.PCI_DSS_4_0 in self.compliance_frameworks:
            if not self.encryption.encrypt_in_transit:
                errors.append("PCI-DSS requires encryption in transit")
            if self.encryption.min_tls_version < "TLS 1.2":
                errors.append("PCI-DSS requires TLS 1.2 or higher")
            if not self.audit.log_data_access:
                errors.append("PCI-DSS requires logging of data access")

        # Check DORA requirements
        if ComplianceRequirement.DORA in self.compliance_frameworks:
            if self.incident_response_sla_minutes > 60:
                errors.append("DORA requires incident response within 60 minutes")
            if not self.audit.real_time_alerting:
                errors.append("DORA requires real-time alerting")

        return errors

    @classmethod
    def for_payment_processing(cls) -> "BankingSecurityPolicy":
        """Create policy for payment processing."""
        return cls(
            policy_id="PAYMENT-001",
            policy_name="Payment Processing Security Policy",
            security_level=SecurityLevel.CRITICAL,
            data_classification=DataClassification.RESTRICTED,
            compliance_frameworks={
                ComplianceRequirement.PCI_DSS_4_0,
                ComplianceRequirement.GDPR,
                ComplianceRequirement.NIST_PQC,
            },
            access_control=AccessControl(
                require_mfa=True,
                dual_control_required=True,
                max_session_duration_hours=4,
            ),
            encryption=EncryptionPolicy(
                require_pqc=True,
                hybrid_mode=True,
                min_key_length_bits=256,
            ),
            audit=AuditPolicy(
                mask_pan=True,
                tamper_evident=True,
                retention_days=2555,
            ),
            risk_tolerance="LOW",
            incident_response_sla_minutes=15,
        )

    @classmethod
    def for_wire_transfer(cls) -> "BankingSecurityPolicy":
        """Create policy for wire transfers."""
        return cls(
            policy_id="WIRE-001",
            policy_name="Wire Transfer Security Policy",
            security_level=SecurityLevel.CRITICAL,
            data_classification=DataClassification.RESTRICTED,
            compliance_frameworks={
                ComplianceRequirement.PCI_DSS_4_0,
                ComplianceRequirement.GDPR,
                ComplianceRequirement.DORA,
                ComplianceRequirement.NIST_PQC,
            },
            access_control=AccessControl(
                require_mfa=True,
                dual_control_required=True,
                four_eyes_principle=True,
                concurrent_sessions_allowed=1,
            ),
            encryption=EncryptionPolicy(
                require_pqc=True,
                hybrid_mode=True,
                pqc_kem_algorithm="ML-KEM-1024",
                pqc_sig_algorithm="ML-DSA-87",
            ),
            audit=AuditPolicy(
                mask_account_numbers=True,
                cryptographic_binding=True,
                retention_days=3650,  # 10 years
            ),
            risk_tolerance="LOW",
            max_transaction_amount=1000000.0,
            velocity_limits={
                "daily_count": 100,
                "daily_amount": 10000000.0,
            },
        )

    @classmethod
    def for_trading(cls) -> "BankingSecurityPolicy":
        """Create policy for trading systems."""
        return cls(
            policy_id="TRADE-001",
            policy_name="Trading Systems Security Policy",
            security_level=SecurityLevel.CRITICAL,
            data_classification=DataClassification.CONFIDENTIAL,
            compliance_frameworks={
                ComplianceRequirement.MIFID_II,
                ComplianceRequirement.EMIR,
                ComplianceRequirement.DORA,
                ComplianceRequirement.NIST_PQC,
            },
            access_control=AccessControl(
                require_mfa=True,
                max_session_duration_hours=12,
                idle_timeout_minutes=30,
            ),
            encryption=EncryptionPolicy(
                require_pqc=True,
                hybrid_mode=False,  # Pure PQC for speed
                pqc_kem_algorithm="ML-KEM-512",
                pqc_sig_algorithm="Falcon-512",
            ),
            audit=AuditPolicy(
                log_data_access=True,
                retention_days=2555,  # 7 years for MiFID II
            ),
            risk_tolerance="MEDIUM",
        )

    @classmethod
    def for_hsm_operations(cls) -> "BankingSecurityPolicy":
        """Create policy for HSM operations."""
        return cls(
            policy_id="HSM-001",
            policy_name="HSM Operations Security Policy",
            security_level=SecurityLevel.RESTRICTED,
            data_classification=DataClassification.TOP_SECRET,
            compliance_frameworks={
                ComplianceRequirement.PCI_DSS_4_0,
                ComplianceRequirement.NIST_PQC,
            },
            access_control=AccessControl(
                require_mfa=True,
                mfa_methods={"hardware_token"},
                dual_control_required=True,
                four_eyes_principle=True,
                concurrent_sessions_allowed=1,
                max_session_duration_hours=1,
            ),
            encryption=EncryptionPolicy(
                require_pqc=True,
                hybrid_mode=True,
                pqc_kem_algorithm="ML-KEM-1024",
                pqc_sig_algorithm="ML-DSA-87",
                min_key_length_bits=256,
            ),
            audit=AuditPolicy(
                log_admin_actions=True,
                cryptographic_binding=True,
                tamper_evident=True,
                real_time_alerting=True,
                retention_days=3650,
            ),
            risk_tolerance="LOW",
            incident_response_sla_minutes=5,
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
            "encryption": {
                "require_pqc": self.encryption.require_pqc,
                "hybrid_mode": self.encryption.hybrid_mode,
                "kem_algorithm": self.encryption.pqc_kem_algorithm,
                "sig_algorithm": self.encryption.pqc_sig_algorithm,
            },
            "access_control": {
                "require_mfa": self.access_control.require_mfa,
                "dual_control": self.access_control.dual_control_required,
            },
            "risk_tolerance": self.risk_tolerance,
        }


# Pre-defined policies
BANKING_POLICIES: Dict[str, BankingSecurityPolicy] = {
    "payment_processing": BankingSecurityPolicy.for_payment_processing(),
    "wire_transfer": BankingSecurityPolicy.for_wire_transfer(),
    "trading": BankingSecurityPolicy.for_trading(),
    "hsm_operations": BankingSecurityPolicy.for_hsm_operations(),
}


def get_policy(name: str) -> Optional[BankingSecurityPolicy]:
    """Get a pre-defined security policy."""
    return BANKING_POLICIES.get(name)


def list_policies() -> List[str]:
    """List available pre-defined policies."""
    return list(BANKING_POLICIES.keys())
