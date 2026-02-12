"""
Certificate Type Definitions

Defines certificate types, status, and metadata structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class CertificateType(Enum):
    """Types of certificates used in banking."""

    ROOT_CA = "root_ca"  # Root Certificate Authority
    INTERMEDIATE_CA = "intermediate_ca"  # Intermediate CA
    SERVER = "server"  # Server/TLS certificate
    CLIENT = "client"  # Client authentication
    CODE_SIGNING = "code_signing"  # Code signing
    EMAIL = "email"  # S/MIME email
    DOCUMENT_SIGNING = "document_signing"  # Document signing

    # Banking specific
    SWIFT_PKI = "swift_pki"  # SWIFT PKI certificate
    PSD2_QWAC = "psd2_qwac"  # PSD2 Qualified Website Authentication
    PSD2_QSEAL = "psd2_qseal"  # PSD2 Qualified Electronic Seal
    EMV_CA = "emv_ca"  # EMV Certificate Authority
    EMV_ISSUER = "emv_issuer"  # EMV Issuer certificate
    EMV_ICC = "emv_icc"  # EMV ICC certificate

    # PQC certificates
    PQC_HYBRID = "pqc_hybrid"  # Hybrid classical + PQC


class CertificateStatus(Enum):
    """Certificate status."""

    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    NOT_YET_VALID = "not_yet_valid"
    UNKNOWN = "unknown"


class KeyUsage(Enum):
    """X.509 Key Usage flags."""

    DIGITAL_SIGNATURE = "digitalSignature"
    NON_REPUDIATION = "nonRepudiation"
    KEY_ENCIPHERMENT = "keyEncipherment"
    DATA_ENCIPHERMENT = "dataEncipherment"
    KEY_AGREEMENT = "keyAgreement"
    KEY_CERT_SIGN = "keyCertSign"
    CRL_SIGN = "cRLSign"
    ENCIPHER_ONLY = "encipherOnly"
    DECIPHER_ONLY = "decipherOnly"


class ExtendedKeyUsage(Enum):
    """X.509 Extended Key Usage OIDs."""

    SERVER_AUTH = "1.3.6.1.5.5.7.3.1"
    CLIENT_AUTH = "1.3.6.1.5.5.7.3.2"
    CODE_SIGNING = "1.3.6.1.5.5.7.3.3"
    EMAIL_PROTECTION = "1.3.6.1.5.5.7.3.4"
    TIME_STAMPING = "1.3.6.1.5.5.7.3.8"
    OCSP_SIGNING = "1.3.6.1.5.5.7.3.9"

    # PSD2 specific
    PSD2_PSP_AS = "0.4.0.19495.1.1"  # Account Servicing
    PSD2_PSP_PI = "0.4.0.19495.1.2"  # Payment Initiation
    PSD2_PSP_AI = "0.4.0.19495.1.3"  # Account Information
    PSD2_PSP_IC = "0.4.0.19495.1.4"  # Issuing of Card-Based Payment


@dataclass
class SubjectInfo:
    """Certificate subject information."""

    common_name: str
    organization: Optional[str] = None
    organizational_unit: Optional[str] = None
    country: Optional[str] = None
    state: Optional[str] = None
    locality: Optional[str] = None
    email: Optional[str] = None

    # PSD2 specific fields
    organization_id: Optional[str] = None  # PSP authorization number
    nca_name: Optional[str] = None  # National Competent Authority
    nca_id: Optional[str] = None  # NCA identifier

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for certificate generation."""
        result = {"CN": self.common_name}

        if self.organization:
            result["O"] = self.organization
        if self.organizational_unit:
            result["OU"] = self.organizational_unit
        if self.country:
            result["C"] = self.country
        if self.state:
            result["ST"] = self.state
        if self.locality:
            result["L"] = self.locality
        if self.email:
            result["emailAddress"] = self.email

        return result


@dataclass
class CertificateInfo:
    """Complete certificate information and metadata."""

    # Identity
    certificate_id: str
    serial_number: str
    subject: SubjectInfo
    issuer: SubjectInfo

    # Type
    cert_type: CertificateType
    is_ca: bool = False
    path_length: Optional[int] = None  # For CA certs

    # Validity
    not_before: datetime = field(default_factory=datetime.utcnow)
    not_after: Optional[datetime] = None
    status: CertificateStatus = CertificateStatus.VALID

    # Key info
    public_key_algorithm: str = "RSA"
    key_size: int = 2048
    signature_algorithm: str = "SHA256withRSA"

    # Extensions
    key_usage: List[KeyUsage] = field(default_factory=list)
    extended_key_usage: List[ExtendedKeyUsage] = field(default_factory=list)
    san_dns_names: List[str] = field(default_factory=list)
    san_ip_addresses: List[str] = field(default_factory=list)
    san_emails: List[str] = field(default_factory=list)
    san_uris: List[str] = field(default_factory=list)

    # Revocation info
    crl_distribution_points: List[str] = field(default_factory=list)
    ocsp_responders: List[str] = field(default_factory=list)

    # Authority info
    ca_issuers: List[str] = field(default_factory=list)

    # Storage
    certificate_pem: Optional[str] = None
    certificate_der: Optional[bytes] = None
    private_key_handle: Optional[str] = None  # HSM key handle

    # Chain
    issuer_certificate_id: Optional[str] = None
    chain_certificate_ids: List[str] = field(default_factory=list)

    # Metadata
    fingerprint_sha256: Optional[str] = None
    fingerprint_sha1: Optional[str] = None
    created_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    revocation_reason: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if certificate is currently valid."""
        now = datetime.utcnow()

        if self.status != CertificateStatus.VALID:
            return False

        if now < self.not_before:
            return False

        if self.not_after and now > self.not_after:
            return False

        return True

    @property
    def is_expired(self) -> bool:
        """Check if certificate has expired."""
        if not self.not_after:
            return False
        return datetime.utcnow() > self.not_after

    @property
    def days_until_expiry(self) -> Optional[int]:
        """Get days until certificate expires."""
        if not self.not_after:
            return None
        delta = self.not_after - datetime.utcnow()
        return max(0, delta.days)

    @property
    def is_self_signed(self) -> bool:
        """Check if certificate is self-signed."""
        return self.subject.common_name == self.issuer.common_name and self.subject.organization == self.issuer.organization

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "certificate_id": self.certificate_id,
            "serial_number": self.serial_number,
            "subject": self.subject.to_dict(),
            "issuer": self.issuer.to_dict(),
            "cert_type": self.cert_type.value,
            "is_ca": self.is_ca,
            "not_before": self.not_before.isoformat(),
            "not_after": self.not_after.isoformat() if self.not_after else None,
            "status": self.status.value,
            "public_key_algorithm": self.public_key_algorithm,
            "key_size": self.key_size,
            "signature_algorithm": self.signature_algorithm,
            "is_valid": self.is_valid,
            "days_until_expiry": self.days_until_expiry,
            "fingerprint_sha256": self.fingerprint_sha256,
        }
