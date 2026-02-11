"""
Certificate Manager

Manages certificates throughout their lifecycle.
"""

import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from ai_engine.domains.banking.security.hsm import HSMProvider, HSMKeyType
from ai_engine.domains.banking.security.pki.certificate_types import (
    CertificateType,
    CertificateStatus,
    CertificateInfo,
    SubjectInfo,
    KeyUsage,
    ExtendedKeyUsage,
)


class CertificateManagerError(Exception):
    """Exception for certificate management operations."""
    pass


class CertificateManager:
    """
    Certificate Manager for PKI operations.

    Features:
    - Certificate generation and signing
    - Certificate storage and retrieval
    - Certificate chain building
    - Certificate revocation
    - OCSP and CRL support
    """

    def __init__(
        self,
        hsm_provider: Optional[HSMProvider] = None,
        audit_callback: Optional[Callable[[str, Dict], None]] = None,
    ):
        """
        Initialize certificate manager.

        Args:
            hsm_provider: HSM for key storage
            audit_callback: Optional audit logging callback
        """
        self._hsm = hsm_provider
        self._audit_callback = audit_callback
        self._certificates: Dict[str, CertificateInfo] = {}
        self._revoked: Dict[str, Dict[str, Any]] = {}

    def generate_self_signed_ca(
        self,
        subject: SubjectInfo,
        key_type: HSMKeyType = HSMKeyType.RSA_4096,
        validity_years: int = 10,
        path_length: int = 0,
        **kwargs,
    ) -> CertificateInfo:
        """
        Generate a self-signed CA certificate.

        Args:
            subject: Certificate subject info
            key_type: Type of key to generate
            validity_years: Certificate validity in years
            path_length: CA path length constraint

        Returns:
            Generated certificate info
        """
        cert_id = str(uuid.uuid4())
        serial = self._generate_serial_number()

        # Generate key pair
        hsm_handle = None
        if self._hsm:
            pub_handle, priv_handle = self._hsm.generate_key_pair(
                key_type=key_type,
                label=f"CA_{subject.common_name}_{cert_id[:8]}",
            )
            hsm_handle = priv_handle.key_id

        # Create certificate info
        now = datetime.utcnow()
        cert_info = CertificateInfo(
            certificate_id=cert_id,
            serial_number=serial,
            subject=subject,
            issuer=subject,  # Self-signed
            cert_type=CertificateType.ROOT_CA,
            is_ca=True,
            path_length=path_length,
            not_before=now,
            not_after=now + timedelta(days=validity_years * 365),
            public_key_algorithm=self._get_algorithm_name(key_type),
            key_size=key_type.key_size,
            signature_algorithm=self._get_signature_algorithm(key_type),
            key_usage=[
                KeyUsage.KEY_CERT_SIGN,
                KeyUsage.CRL_SIGN,
                KeyUsage.DIGITAL_SIGNATURE,
            ],
            private_key_handle=hsm_handle,
            created_at=now,
        )

        # Generate fingerprints (placeholder - would use actual cert)
        cert_info.fingerprint_sha256 = hashlib.sha256(
            f"{cert_id}{serial}".encode()
        ).hexdigest()

        self._certificates[cert_id] = cert_info

        self._audit("ca_certificate_generated", {
            "certificate_id": cert_id,
            "subject": subject.common_name,
            "validity_years": validity_years,
        })

        return cert_info

    def generate_certificate(
        self,
        subject: SubjectInfo,
        issuer_cert_id: str,
        cert_type: CertificateType,
        key_type: HSMKeyType = HSMKeyType.RSA_2048,
        validity_days: int = 365,
        key_usage: Optional[List[KeyUsage]] = None,
        extended_key_usage: Optional[List[ExtendedKeyUsage]] = None,
        san_dns_names: Optional[List[str]] = None,
        san_ip_addresses: Optional[List[str]] = None,
        **kwargs,
    ) -> CertificateInfo:
        """
        Generate a certificate signed by an issuer.

        Args:
            subject: Certificate subject
            issuer_cert_id: ID of issuing CA certificate
            cert_type: Type of certificate
            key_type: Key type to generate
            validity_days: Validity period in days
            key_usage: Key usage flags
            extended_key_usage: Extended key usage
            san_dns_names: Subject Alternative Names (DNS)
            san_ip_addresses: Subject Alternative Names (IP)

        Returns:
            Generated certificate info
        """
        # Get issuer certificate
        issuer_cert = self._certificates.get(issuer_cert_id)
        if not issuer_cert:
            raise CertificateManagerError(f"Issuer certificate not found: {issuer_cert_id}")

        if not issuer_cert.is_ca:
            raise CertificateManagerError("Issuer certificate is not a CA")

        cert_id = str(uuid.uuid4())
        serial = self._generate_serial_number()

        # Generate key pair
        hsm_handle = None
        if self._hsm:
            pub_handle, priv_handle = self._hsm.generate_key_pair(
                key_type=key_type,
                label=f"{cert_type.value}_{subject.common_name}_{cert_id[:8]}",
            )
            hsm_handle = priv_handle.key_id

        # Default key usage based on certificate type
        if key_usage is None:
            key_usage = self._default_key_usage(cert_type)

        if extended_key_usage is None:
            extended_key_usage = self._default_extended_key_usage(cert_type)

        now = datetime.utcnow()
        cert_info = CertificateInfo(
            certificate_id=cert_id,
            serial_number=serial,
            subject=subject,
            issuer=issuer_cert.subject,
            cert_type=cert_type,
            is_ca=cert_type in (CertificateType.INTERMEDIATE_CA,),
            not_before=now,
            not_after=now + timedelta(days=validity_days),
            public_key_algorithm=self._get_algorithm_name(key_type),
            key_size=key_type.key_size,
            signature_algorithm=self._get_signature_algorithm(
                self._get_key_type_from_size(issuer_cert.key_size)
            ),
            key_usage=key_usage,
            extended_key_usage=extended_key_usage,
            san_dns_names=san_dns_names or [],
            san_ip_addresses=san_ip_addresses or [],
            private_key_handle=hsm_handle,
            issuer_certificate_id=issuer_cert_id,
            created_at=now,
        )

        cert_info.fingerprint_sha256 = hashlib.sha256(
            f"{cert_id}{serial}".encode()
        ).hexdigest()

        self._certificates[cert_id] = cert_info

        self._audit("certificate_generated", {
            "certificate_id": cert_id,
            "subject": subject.common_name,
            "cert_type": cert_type.value,
            "issuer": issuer_cert.subject.common_name,
        })

        return cert_info

    def generate_psd2_certificate(
        self,
        subject: SubjectInfo,
        issuer_cert_id: str,
        psp_roles: List[ExtendedKeyUsage],
        validity_days: int = 730,
        **kwargs,
    ) -> CertificateInfo:
        """
        Generate a PSD2 QWAC or QSEAL certificate.

        Args:
            subject: Certificate subject with PSD2 fields
            issuer_cert_id: ID of QTSP CA
            psp_roles: PSD2 PSP roles
            validity_days: Validity period

        Returns:
            Generated certificate info
        """
        # Validate PSD2 required fields
        if not subject.organization_id:
            raise CertificateManagerError("organization_id required for PSD2 certificate")

        # Determine certificate type
        cert_type = CertificateType.PSD2_QWAC
        if ExtendedKeyUsage.SERVER_AUTH not in psp_roles:
            cert_type = CertificateType.PSD2_QSEAL

        key_usage = [
            KeyUsage.DIGITAL_SIGNATURE,
            KeyUsage.KEY_ENCIPHERMENT,
        ]

        extended_key_usage = list(psp_roles)
        if cert_type == CertificateType.PSD2_QWAC:
            extended_key_usage.extend([
                ExtendedKeyUsage.SERVER_AUTH,
                ExtendedKeyUsage.CLIENT_AUTH,
            ])

        return self.generate_certificate(
            subject=subject,
            issuer_cert_id=issuer_cert_id,
            cert_type=cert_type,
            validity_days=validity_days,
            key_usage=key_usage,
            extended_key_usage=extended_key_usage,
            **kwargs,
        )

    def get_certificate(self, cert_id: str) -> Optional[CertificateInfo]:
        """Get certificate by ID."""
        return self._certificates.get(cert_id)

    def get_certificate_by_subject(
        self,
        common_name: str,
        organization: Optional[str] = None,
    ) -> Optional[CertificateInfo]:
        """Get certificate by subject."""
        for cert in self._certificates.values():
            if cert.subject.common_name == common_name:
                if organization is None or cert.subject.organization == organization:
                    return cert
        return None

    def list_certificates(
        self,
        cert_type: Optional[CertificateType] = None,
        status: Optional[CertificateStatus] = None,
        include_expired: bool = False,
    ) -> List[CertificateInfo]:
        """List certificates matching criteria."""
        result = []

        for cert in self._certificates.values():
            if cert_type and cert.cert_type != cert_type:
                continue

            if status and cert.status != status:
                continue

            if not include_expired and cert.is_expired:
                continue

            result.append(cert)

        return result

    def get_certificate_chain(self, cert_id: str) -> List[CertificateInfo]:
        """Build certificate chain from leaf to root."""
        chain = []
        current_id = cert_id

        while current_id:
            cert = self._certificates.get(current_id)
            if not cert:
                break

            chain.append(cert)

            if cert.is_self_signed:
                break

            current_id = cert.issuer_certificate_id

        return chain

    def revoke_certificate(
        self,
        cert_id: str,
        reason: str = "unspecified",
    ) -> None:
        """Revoke a certificate."""
        cert = self._certificates.get(cert_id)
        if not cert:
            raise CertificateManagerError(f"Certificate not found: {cert_id}")

        cert.status = CertificateStatus.REVOKED
        cert.revoked_at = datetime.utcnow()
        cert.revocation_reason = reason

        self._revoked[cert_id] = {
            "revoked_at": cert.revoked_at.isoformat(),
            "reason": reason,
            "serial_number": cert.serial_number,
        }

        self._audit("certificate_revoked", {
            "certificate_id": cert_id,
            "subject": cert.subject.common_name,
            "reason": reason,
        })

    def is_revoked(self, cert_id: str) -> bool:
        """Check if certificate is revoked."""
        return cert_id in self._revoked

    def get_expiring_certificates(
        self,
        days: int = 30,
    ) -> List[CertificateInfo]:
        """Get certificates expiring within specified days."""
        result = []
        cutoff = datetime.utcnow() + timedelta(days=days)

        for cert in self._certificates.values():
            if cert.status != CertificateStatus.VALID:
                continue
            if cert.not_after and cert.not_after <= cutoff:
                result.append(cert)

        return sorted(result, key=lambda c: c.not_after or datetime.max)

    def export_certificate_inventory(self) -> List[Dict[str, Any]]:
        """Export certificate inventory."""
        return [cert.to_dict() for cert in self._certificates.values()]

    def _generate_serial_number(self) -> str:
        """Generate unique serial number."""
        import secrets
        return secrets.token_hex(20)

    def _get_algorithm_name(self, key_type: HSMKeyType) -> str:
        """Get algorithm name from key type."""
        if "RSA" in key_type.algorithm_name:
            return "RSA"
        elif "EC" in key_type.algorithm_name:
            return "EC"
        elif "ML-KEM" in key_type.algorithm_name:
            return "ML-KEM"
        elif "ML-DSA" in key_type.algorithm_name:
            return "ML-DSA"
        return key_type.algorithm_name

    def _get_signature_algorithm(self, key_type: HSMKeyType) -> str:
        """Get signature algorithm for key type."""
        if "RSA" in key_type.algorithm_name:
            return "SHA256withRSA"
        elif "EC" in key_type.algorithm_name:
            return "SHA256withECDSA"
        elif "ML-DSA" in key_type.algorithm_name:
            return "ML-DSA"
        return "SHA256withRSA"

    def _get_key_type_from_size(self, size: int) -> HSMKeyType:
        """Get key type from size."""
        mapping = {
            2048: HSMKeyType.RSA_2048,
            3072: HSMKeyType.RSA_3072,
            4096: HSMKeyType.RSA_4096,
            256: HSMKeyType.EC_P256,
            384: HSMKeyType.EC_P384,
        }
        return mapping.get(size, HSMKeyType.RSA_2048)

    def _default_key_usage(self, cert_type: CertificateType) -> List[KeyUsage]:
        """Get default key usage for certificate type."""
        defaults = {
            CertificateType.ROOT_CA: [KeyUsage.KEY_CERT_SIGN, KeyUsage.CRL_SIGN],
            CertificateType.INTERMEDIATE_CA: [KeyUsage.KEY_CERT_SIGN, KeyUsage.CRL_SIGN],
            CertificateType.SERVER: [KeyUsage.DIGITAL_SIGNATURE, KeyUsage.KEY_ENCIPHERMENT],
            CertificateType.CLIENT: [KeyUsage.DIGITAL_SIGNATURE, KeyUsage.KEY_ENCIPHERMENT],
            CertificateType.CODE_SIGNING: [KeyUsage.DIGITAL_SIGNATURE],
            CertificateType.EMAIL: [KeyUsage.DIGITAL_SIGNATURE, KeyUsage.KEY_ENCIPHERMENT],
        }
        return defaults.get(cert_type, [KeyUsage.DIGITAL_SIGNATURE])

    def _default_extended_key_usage(
        self,
        cert_type: CertificateType,
    ) -> List[ExtendedKeyUsage]:
        """Get default extended key usage for certificate type."""
        defaults = {
            CertificateType.SERVER: [ExtendedKeyUsage.SERVER_AUTH],
            CertificateType.CLIENT: [ExtendedKeyUsage.CLIENT_AUTH],
            CertificateType.CODE_SIGNING: [ExtendedKeyUsage.CODE_SIGNING],
            CertificateType.EMAIL: [ExtendedKeyUsage.EMAIL_PROTECTION],
        }
        return defaults.get(cert_type, [])

    def _audit(self, event: str, data: Dict[str, Any]) -> None:
        """Log audit event."""
        if self._audit_callback:
            audit_record = {
                "event": event,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
            }
            self._audit_callback(event, audit_record)
