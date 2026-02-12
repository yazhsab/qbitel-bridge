"""
Certificate Validator

Validates certificates for banking applications.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ai_engine.domains.banking.security.pki.certificate_types import (
    CertificateInfo,
    CertificateStatus,
    CertificateType,
    KeyUsage,
    ExtendedKeyUsage,
)


class ValidationSeverity(Enum):
    """Severity of validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    code: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    field: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of certificate validation."""

    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)
    certificate_id: Optional[str] = None

    def add_error(self, code: str, message: str, field: Optional[str] = None) -> None:
        """Add an error."""
        self.issues.append(
            ValidationIssue(
                code=code,
                message=message,
                severity=ValidationSeverity.ERROR,
                field=field,
            )
        )
        self.is_valid = False

    def add_warning(self, code: str, message: str, field: Optional[str] = None) -> None:
        """Add a warning."""
        self.issues.append(
            ValidationIssue(
                code=code,
                message=message,
                severity=ValidationSeverity.WARNING,
                field=field,
            )
        )

    def add_info(self, code: str, message: str, field: Optional[str] = None) -> None:
        """Add an info message."""
        self.issues.append(
            ValidationIssue(
                code=code,
                message=message,
                severity=ValidationSeverity.INFO,
                field=field,
            )
        )

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get errors only."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get warnings only."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "checked_at": self.checked_at.isoformat(),
            "certificate_id": self.certificate_id,
            "errors": [{"code": i.code, "message": i.message, "field": i.field} for i in self.errors],
            "warnings": [{"code": i.code, "message": i.message, "field": i.field} for i in self.warnings],
        }


class CertificateValidator:
    """
    Validator for certificates.

    Validates:
    - Certificate validity period
    - Certificate chain
    - Key usage and extended key usage
    - Revocation status
    - PSD2 specific requirements
    - Banking-specific requirements
    """

    def __init__(
        self,
        certificate_store: Optional[Dict[str, CertificateInfo]] = None,
        revoked_certificates: Optional[Set[str]] = None,
        check_revocation: bool = True,
    ):
        """
        Initialize validator.

        Args:
            certificate_store: Store of known certificates
            revoked_certificates: Set of revoked certificate IDs
            check_revocation: Whether to check revocation status
        """
        self._cert_store = certificate_store or {}
        self._revoked = revoked_certificates or set()
        self._check_revocation = check_revocation

    def validate(self, certificate: CertificateInfo) -> ValidationResult:
        """
        Validate a certificate.

        Args:
            certificate: Certificate to validate

        Returns:
            Validation result
        """
        result = ValidationResult(certificate_id=certificate.certificate_id)

        # Basic validity checks
        self._validate_validity_period(certificate, result)
        self._validate_key_usage(certificate, result)
        self._validate_chain(certificate, result)

        # Revocation check
        if self._check_revocation:
            self._validate_revocation(certificate, result)

        # Type-specific validation
        if certificate.cert_type in (CertificateType.PSD2_QWAC, CertificateType.PSD2_QSEAL):
            self._validate_psd2(certificate, result)

        if certificate.cert_type in (CertificateType.EMV_CA, CertificateType.EMV_ISSUER):
            self._validate_emv(certificate, result)

        return result

    def validate_chain(
        self,
        leaf_certificate: CertificateInfo,
        chain: List[CertificateInfo],
    ) -> ValidationResult:
        """
        Validate a certificate chain.

        Args:
            leaf_certificate: End-entity certificate
            chain: Chain of certificates (intermediate -> root)

        Returns:
            Validation result
        """
        result = ValidationResult(certificate_id=leaf_certificate.certificate_id)

        # Validate leaf certificate
        leaf_result = self.validate(leaf_certificate)
        result.issues.extend(leaf_result.issues)

        if not chain:
            result.add_error("CHAIN_EMPTY", "Certificate chain is empty")
            return result

        # Validate chain linkage
        current = leaf_certificate
        for i, issuer_cert in enumerate(chain):
            # Check issuer matches
            if current.issuer.common_name != issuer_cert.subject.common_name:
                result.add_error(
                    "CHAIN_ISSUER_MISMATCH",
                    f"Issuer mismatch at position {i}",
                )

            # Check issuer is CA
            if not issuer_cert.is_ca and i < len(chain) - 1:
                result.add_error(
                    "CHAIN_NOT_CA",
                    f"Certificate at position {i} is not a CA",
                )

            # Validate each certificate in chain
            chain_result = self.validate(issuer_cert)
            result.issues.extend(chain_result.issues)

            current = issuer_cert

        # Check chain ends with self-signed (root)
        if not chain[-1].is_self_signed:
            result.add_warning(
                "CHAIN_NO_ROOT",
                "Chain does not end with self-signed root",
            )

        result.is_valid = len(result.errors) == 0
        return result

    def validate_for_purpose(
        self,
        certificate: CertificateInfo,
        purpose: str,
    ) -> ValidationResult:
        """
        Validate certificate for specific purpose.

        Args:
            certificate: Certificate to validate
            purpose: Intended purpose (e.g., "tls_server", "code_signing")

        Returns:
            Validation result
        """
        result = self.validate(certificate)

        purpose_requirements = {
            "tls_server": {
                "key_usage": [KeyUsage.DIGITAL_SIGNATURE, KeyUsage.KEY_ENCIPHERMENT],
                "extended_key_usage": [ExtendedKeyUsage.SERVER_AUTH],
            },
            "tls_client": {
                "key_usage": [KeyUsage.DIGITAL_SIGNATURE],
                "extended_key_usage": [ExtendedKeyUsage.CLIENT_AUTH],
            },
            "code_signing": {
                "key_usage": [KeyUsage.DIGITAL_SIGNATURE],
                "extended_key_usage": [ExtendedKeyUsage.CODE_SIGNING],
            },
            "psd2_payment": {
                "extended_key_usage": [ExtendedKeyUsage.PSD2_PSP_PI],
            },
            "psd2_account_info": {
                "extended_key_usage": [ExtendedKeyUsage.PSD2_PSP_AI],
            },
        }

        requirements = purpose_requirements.get(purpose, {})

        # Check required key usage
        required_key_usage = requirements.get("key_usage", [])
        for ku in required_key_usage:
            if ku not in certificate.key_usage:
                result.add_error(
                    "MISSING_KEY_USAGE",
                    f"Missing required key usage: {ku.value}",
                    field="keyUsage",
                )

        # Check required extended key usage
        required_eku = requirements.get("extended_key_usage", [])
        for eku in required_eku:
            if eku not in certificate.extended_key_usage:
                result.add_error(
                    "MISSING_EXTENDED_KEY_USAGE",
                    f"Missing required extended key usage: {eku.value}",
                    field="extendedKeyUsage",
                )

        return result

    def _validate_validity_period(
        self,
        cert: CertificateInfo,
        result: ValidationResult,
    ) -> None:
        """Validate certificate validity period."""
        now = datetime.utcnow()

        # Check not yet valid
        if now < cert.not_before:
            result.add_error(
                "NOT_YET_VALID",
                f"Certificate not valid until {cert.not_before}",
                field="notBefore",
            )

        # Check expired
        if cert.not_after and now > cert.not_after:
            result.add_error(
                "EXPIRED",
                f"Certificate expired on {cert.not_after}",
                field="notAfter",
            )

        # Warning for soon-to-expire
        if cert.days_until_expiry is not None:
            if cert.days_until_expiry <= 30:
                result.add_warning(
                    "EXPIRING_SOON",
                    f"Certificate expires in {cert.days_until_expiry} days",
                    field="notAfter",
                )

    def _validate_key_usage(
        self,
        cert: CertificateInfo,
        result: ValidationResult,
    ) -> None:
        """Validate key usage."""
        # CA certificates must have keyCertSign
        if cert.is_ca:
            if KeyUsage.KEY_CERT_SIGN not in cert.key_usage:
                result.add_error(
                    "CA_MISSING_KEY_CERT_SIGN",
                    "CA certificate must have keyCertSign usage",
                    field="keyUsage",
                )

        # Check key size
        if cert.public_key_algorithm == "RSA":
            if cert.key_size < 2048:
                result.add_error(
                    "KEY_SIZE_TOO_SMALL",
                    f"RSA key size {cert.key_size} is below minimum 2048",
                    field="keySize",
                )
            elif cert.key_size < 3072:
                result.add_warning(
                    "KEY_SIZE_WEAK",
                    f"RSA key size {cert.key_size} may be weak, consider 3072+",
                    field="keySize",
                )

    def _validate_chain(
        self,
        cert: CertificateInfo,
        result: ValidationResult,
    ) -> None:
        """Validate certificate chain linkage."""
        if cert.is_self_signed:
            return  # Self-signed doesn't need chain validation

        # Check issuer certificate exists
        if cert.issuer_certificate_id:
            issuer = self._cert_store.get(cert.issuer_certificate_id)
            if not issuer:
                result.add_warning(
                    "ISSUER_NOT_FOUND",
                    "Issuer certificate not found in store",
                    field="issuer",
                )

    def _validate_revocation(
        self,
        cert: CertificateInfo,
        result: ValidationResult,
    ) -> None:
        """Validate revocation status."""
        if cert.certificate_id in self._revoked:
            result.add_error(
                "REVOKED",
                "Certificate has been revoked",
                field="status",
            )
            return

        if cert.status == CertificateStatus.REVOKED:
            result.add_error(
                "REVOKED",
                f"Certificate revoked: {cert.revocation_reason}",
                field="status",
            )

    def _validate_psd2(
        self,
        cert: CertificateInfo,
        result: ValidationResult,
    ) -> None:
        """Validate PSD2 specific requirements."""
        # Check organization ID (PSP authorization number)
        if not cert.subject.organization_id:
            result.add_error(
                "PSD2_MISSING_ORG_ID",
                "PSD2 certificate must have organization identifier",
                field="subject.organizationId",
            )

        # Check NCA information
        if not cert.subject.nca_name:
            result.add_warning(
                "PSD2_MISSING_NCA",
                "PSD2 certificate should have NCA information",
                field="subject.ncaName",
            )

        # Check for at least one PSP role
        psd2_roles = [
            ExtendedKeyUsage.PSD2_PSP_AS,
            ExtendedKeyUsage.PSD2_PSP_PI,
            ExtendedKeyUsage.PSD2_PSP_AI,
            ExtendedKeyUsage.PSD2_PSP_IC,
        ]

        has_role = any(r in cert.extended_key_usage for r in psd2_roles)
        if not has_role:
            result.add_error(
                "PSD2_NO_ROLE",
                "PSD2 certificate must have at least one PSP role",
                field="extendedKeyUsage",
            )

        # QWAC specific
        if cert.cert_type == CertificateType.PSD2_QWAC:
            if ExtendedKeyUsage.SERVER_AUTH not in cert.extended_key_usage:
                result.add_error(
                    "PSD2_QWAC_NO_SERVER_AUTH",
                    "QWAC must have serverAuth extended key usage",
                    field="extendedKeyUsage",
                )

            if not cert.san_dns_names:
                result.add_error(
                    "PSD2_QWAC_NO_SAN",
                    "QWAC must have Subject Alternative Names",
                    field="subjectAltName",
                )

    def _validate_emv(
        self,
        cert: CertificateInfo,
        result: ValidationResult,
    ) -> None:
        """Validate EMV certificate requirements."""
        # EMV certificates typically have specific length requirements
        if cert.cert_type == CertificateType.EMV_CA:
            if cert.key_size < 1984:
                result.add_error(
                    "EMV_CA_KEY_SIZE",
                    "EMV CA certificate key size must be at least 1984 bits",
                    field="keySize",
                )

        # Check signature algorithm
        allowed_algorithms = ["SHA256withRSA", "SHA384withRSA", "SHA512withRSA"]
        if cert.signature_algorithm not in allowed_algorithms:
            result.add_warning(
                "EMV_SIGNATURE_ALGORITHM",
                f"EMV recommends using {allowed_algorithms}",
                field="signatureAlgorithm",
            )
