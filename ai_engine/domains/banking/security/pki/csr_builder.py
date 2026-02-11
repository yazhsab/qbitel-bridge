"""
CSR (Certificate Signing Request) Builder

Builds CSRs for certificate issuance.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ai_engine.domains.banking.security.pki.certificate_types import (
    SubjectInfo,
    KeyUsage,
    ExtendedKeyUsage,
)


@dataclass
class CSRInfo:
    """CSR information."""

    subject: SubjectInfo
    public_key_pem: Optional[str] = None
    public_key_der: Optional[bytes] = None
    signature: Optional[bytes] = None
    csr_pem: Optional[str] = None
    csr_der: Optional[bytes] = None

    # Extensions
    key_usage: List[KeyUsage] = field(default_factory=list)
    extended_key_usage: List[ExtendedKeyUsage] = field(default_factory=list)
    san_dns_names: List[str] = field(default_factory=list)
    san_ip_addresses: List[str] = field(default_factory=list)
    san_emails: List[str] = field(default_factory=list)

    # PSD2 specific
    psd2_roles: List[str] = field(default_factory=list)


class CSRBuilder:
    """
    Builder for Certificate Signing Requests.

    Supports:
    - Standard X.509 CSRs
    - PSD2 CSRs with required extensions
    - Subject Alternative Names
    - Custom extensions
    """

    def __init__(self):
        """Initialize CSR builder."""
        self._subject: Optional[SubjectInfo] = None
        self._key_usage: List[KeyUsage] = []
        self._extended_key_usage: List[ExtendedKeyUsage] = []
        self._san_dns: List[str] = []
        self._san_ip: List[str] = []
        self._san_email: List[str] = []
        self._psd2_roles: List[str] = []
        self._private_key = None

    def set_subject(
        self,
        common_name: str,
        organization: Optional[str] = None,
        organizational_unit: Optional[str] = None,
        country: Optional[str] = None,
        state: Optional[str] = None,
        locality: Optional[str] = None,
        email: Optional[str] = None,
    ) -> "CSRBuilder":
        """Set the CSR subject."""
        self._subject = SubjectInfo(
            common_name=common_name,
            organization=organization,
            organizational_unit=organizational_unit,
            country=country,
            state=state,
            locality=locality,
            email=email,
        )
        return self

    def set_subject_info(self, subject: SubjectInfo) -> "CSRBuilder":
        """Set subject from SubjectInfo object."""
        self._subject = subject
        return self

    def add_key_usage(self, *usages: KeyUsage) -> "CSRBuilder":
        """Add key usage flags."""
        self._key_usage.extend(usages)
        return self

    def add_extended_key_usage(self, *usages: ExtendedKeyUsage) -> "CSRBuilder":
        """Add extended key usage."""
        self._extended_key_usage.extend(usages)
        return self

    def add_san_dns(self, *dns_names: str) -> "CSRBuilder":
        """Add DNS Subject Alternative Names."""
        self._san_dns.extend(dns_names)
        return self

    def add_san_ip(self, *ip_addresses: str) -> "CSRBuilder":
        """Add IP Subject Alternative Names."""
        self._san_ip.extend(ip_addresses)
        return self

    def add_san_email(self, *emails: str) -> "CSRBuilder":
        """Add email Subject Alternative Names."""
        self._san_email.extend(emails)
        return self

    def set_psd2_roles(
        self,
        account_servicing: bool = False,
        payment_initiation: bool = False,
        account_information: bool = False,
        card_issuing: bool = False,
    ) -> "CSRBuilder":
        """Set PSD2 PSP roles."""
        self._psd2_roles = []

        if account_servicing:
            self._psd2_roles.append("PSP_AS")
            self._extended_key_usage.append(ExtendedKeyUsage.PSD2_PSP_AS)

        if payment_initiation:
            self._psd2_roles.append("PSP_PI")
            self._extended_key_usage.append(ExtendedKeyUsage.PSD2_PSP_PI)

        if account_information:
            self._psd2_roles.append("PSP_AI")
            self._extended_key_usage.append(ExtendedKeyUsage.PSD2_PSP_AI)

        if card_issuing:
            self._psd2_roles.append("PSP_IC")
            self._extended_key_usage.append(ExtendedKeyUsage.PSD2_PSP_IC)

        return self

    def for_server_certificate(self) -> "CSRBuilder":
        """Configure for server/TLS certificate."""
        self._key_usage = [
            KeyUsage.DIGITAL_SIGNATURE,
            KeyUsage.KEY_ENCIPHERMENT,
        ]
        self._extended_key_usage = [
            ExtendedKeyUsage.SERVER_AUTH,
        ]
        return self

    def for_client_certificate(self) -> "CSRBuilder":
        """Configure for client authentication certificate."""
        self._key_usage = [
            KeyUsage.DIGITAL_SIGNATURE,
            KeyUsage.KEY_ENCIPHERMENT,
        ]
        self._extended_key_usage = [
            ExtendedKeyUsage.CLIENT_AUTH,
        ]
        return self

    def for_code_signing(self) -> "CSRBuilder":
        """Configure for code signing certificate."""
        self._key_usage = [
            KeyUsage.DIGITAL_SIGNATURE,
        ]
        self._extended_key_usage = [
            ExtendedKeyUsage.CODE_SIGNING,
        ]
        return self

    def for_psd2_qwac(self) -> "CSRBuilder":
        """Configure for PSD2 QWAC certificate."""
        self._key_usage = [
            KeyUsage.DIGITAL_SIGNATURE,
            KeyUsage.KEY_ENCIPHERMENT,
        ]
        self._extended_key_usage.extend([
            ExtendedKeyUsage.SERVER_AUTH,
            ExtendedKeyUsage.CLIENT_AUTH,
        ])
        return self

    def for_psd2_qseal(self) -> "CSRBuilder":
        """Configure for PSD2 QSEAL certificate."""
        self._key_usage = [
            KeyUsage.DIGITAL_SIGNATURE,
            KeyUsage.NON_REPUDIATION,
        ]
        return self

    def build(self) -> CSRInfo:
        """Build the CSR info."""
        if not self._subject:
            raise ValueError("Subject is required")

        return CSRInfo(
            subject=self._subject,
            key_usage=list(set(self._key_usage)),
            extended_key_usage=list(set(self._extended_key_usage)),
            san_dns_names=list(set(self._san_dns)),
            san_ip_addresses=list(set(self._san_ip)),
            san_emails=list(set(self._san_email)),
            psd2_roles=self._psd2_roles,
        )

    def build_pem(self, private_key: bytes) -> str:
        """
        Build CSR and return as PEM string.

        In production, this would use cryptography library to
        actually generate and sign the CSR.
        """
        csr_info = self.build()

        # Placeholder - in production would generate actual CSR
        pem = "-----BEGIN CERTIFICATE REQUEST-----\n"
        pem += "MIICpDCCAYwCAQAwXzELMAkGA1UEBhMCVVMxCzAJBgNVBAgMAldBMRAwDgYDVQQH\n"
        pem += "DAdTZWF0dGxlMRMwEQYDVQQKDApFeGFtcGxlIENvMRwwGgYDVQQDDBN3d3cuZXhh\n"
        pem += "bXBsZS5jb20uY29tMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...\n"
        pem += "-----END CERTIFICATE REQUEST-----\n"

        csr_info.csr_pem = pem
        return pem

    def reset(self) -> "CSRBuilder":
        """Reset builder to initial state."""
        self._subject = None
        self._key_usage = []
        self._extended_key_usage = []
        self._san_dns = []
        self._san_ip = []
        self._san_email = []
        self._psd2_roles = []
        self._private_key = None
        return self
