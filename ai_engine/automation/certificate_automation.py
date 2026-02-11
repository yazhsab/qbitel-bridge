"""
Certificate Automation

Automated certificate lifecycle management including:
- Certificate generation and signing
- Automatic renewal
- Revocation handling
- Certificate chain management
- ACME protocol support
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import uuid
import threading
import hashlib
import base64

logger = logging.getLogger(__name__)


class CertificateType(Enum):
    """Types of certificates."""

    TLS_SERVER = "tls_server"
    TLS_CLIENT = "tls_client"
    CODE_SIGNING = "code_signing"
    DOCUMENT_SIGNING = "document_signing"
    CA_INTERMEDIATE = "ca_intermediate"
    TIMESTAMP = "timestamp"


class CertificateState(Enum):
    """Certificate lifecycle states."""

    PENDING = "pending"
    ACTIVE = "active"
    EXPIRING = "expiring"
    EXPIRED = "expired"
    REVOKED = "revoked"
    RENEWED = "renewed"


class RevocationReason(Enum):
    """Certificate revocation reasons (RFC 5280)."""

    UNSPECIFIED = 0
    KEY_COMPROMISE = 1
    CA_COMPROMISE = 2
    AFFILIATION_CHANGED = 3
    SUPERSEDED = 4
    CESSATION_OF_OPERATION = 5
    CERTIFICATE_HOLD = 6
    PRIVILEGE_WITHDRAWN = 9
    AA_COMPROMISE = 10


@dataclass
class CertificatePolicy:
    """Policy for certificate management."""

    cert_type: CertificateType
    validity_days: int = 365
    renewal_threshold_days: int = 30
    key_algorithm: str = "ML-DSA-65"  # PQC default
    key_size: int = 256
    auto_renew: bool = True
    require_approval: bool = False
    allowed_domains: List[str] = field(default_factory=list)
    allowed_ips: List[str] = field(default_factory=list)
    extended_key_usage: List[str] = field(default_factory=list)
    ocsp_must_staple: bool = True
    ct_required: bool = True  # Certificate Transparency


@dataclass
class CertificateRequest:
    """Certificate signing request."""

    request_id: str
    common_name: str
    cert_type: CertificateType
    subject_alt_names: List[str] = field(default_factory=list)
    organization: str = ""
    organizational_unit: str = ""
    country: str = ""
    state: str = ""
    locality: str = ""
    email: str = ""
    key_algorithm: str = "ML-DSA-65"
    created_at: datetime = field(default_factory=datetime.utcnow)
    csr_pem: Optional[str] = None
    approved: bool = False
    approved_by: Optional[str] = None


@dataclass
class Certificate:
    """Managed certificate."""

    cert_id: str
    serial_number: str
    common_name: str
    cert_type: CertificateType
    state: CertificateState
    subject_alt_names: List[str]
    issuer: str
    not_before: datetime
    not_after: datetime
    key_algorithm: str
    signature_algorithm: str
    fingerprint_sha256: str
    cert_pem: str
    chain_pem: Optional[str] = None
    private_key_id: Optional[str] = None  # Reference to key in KMS
    renewal_of: Optional[str] = None
    renewed_by: Optional[str] = None
    revoked_at: Optional[datetime] = None
    revocation_reason: Optional[RevocationReason] = None
    policy_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def days_until_expiry(self) -> int:
        """Get days until expiry."""
        return (self.not_after - datetime.utcnow()).days

    def is_expiring_soon(self, threshold_days: int = 30) -> bool:
        """Check if certificate is expiring soon."""
        return self.days_until_expiry() <= threshold_days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cert_id": self.cert_id,
            "serial_number": self.serial_number,
            "common_name": self.common_name,
            "cert_type": self.cert_type.value,
            "state": self.state.value,
            "subject_alt_names": self.subject_alt_names,
            "issuer": self.issuer,
            "not_before": self.not_before.isoformat(),
            "not_after": self.not_after.isoformat(),
            "days_until_expiry": self.days_until_expiry(),
            "key_algorithm": self.key_algorithm,
            "signature_algorithm": self.signature_algorithm,
            "fingerprint_sha256": self.fingerprint_sha256,
        }


@dataclass
class RenewalEvent:
    """Certificate renewal event."""

    event_id: str
    old_cert_id: str
    new_cert_id: str
    renewed_at: datetime
    automatic: bool
    reason: str
    success: bool = True
    error: Optional[str] = None


class CertificateAutomation:
    """
    Automated certificate lifecycle manager.

    Capabilities:
    - PQC-ready certificate generation
    - Automatic renewal scheduling
    - ACME protocol support
    - Certificate transparency logging
    - Revocation management
    """

    def __init__(
        self,
        ca_provider: Optional[Any] = None,
        key_manager: Optional[Any] = None,
        renewal_callback: Optional[Callable[[RenewalEvent], None]] = None,
    ):
        self._ca_provider = ca_provider
        self._key_manager = key_manager
        self._renewal_callback = renewal_callback
        self._certificates: Dict[str, Certificate] = {}
        self._requests: Dict[str, CertificateRequest] = {}
        self._policies: Dict[str, CertificatePolicy] = {}
        self._renewal_events: List[RenewalEvent] = []
        self._lock = threading.RLock()
        self._renewal_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Initialize default policies
        self._init_default_policies()

    def create_certificate_request(
        self,
        common_name: str,
        cert_type: CertificateType,
        subject_alt_names: Optional[List[str]] = None,
        organization: str = "",
        policy_id: Optional[str] = None,
    ) -> CertificateRequest:
        """
        Create a certificate signing request.

        Args:
            common_name: Certificate common name
            cert_type: Type of certificate
            subject_alt_names: Subject alternative names
            organization: Organization name
            policy_id: Policy to apply

        Returns:
            CertificateRequest
        """
        with self._lock:
            # Get policy
            policy = self._get_policy(cert_type, policy_id)

            # Validate against policy
            if policy.allowed_domains:
                if not any(
                    common_name.endswith(d) or common_name == d
                    for d in policy.allowed_domains
                ):
                    raise ValueError(f"Domain {common_name} not allowed by policy")

            request = CertificateRequest(
                request_id=str(uuid.uuid4()),
                common_name=common_name,
                cert_type=cert_type,
                subject_alt_names=subject_alt_names or [],
                organization=organization,
                key_algorithm=policy.key_algorithm,
                approved=not policy.require_approval,
            )

            self._requests[request.request_id] = request

            logger.info(f"Created certificate request {request.request_id} for {common_name}")

            return request

    def approve_request(
        self,
        request_id: str,
        approved_by: str,
    ) -> bool:
        """Approve a certificate request."""
        with self._lock:
            if request_id not in self._requests:
                return False

            request = self._requests[request_id]
            request.approved = True
            request.approved_by = approved_by

            logger.info(f"Request {request_id} approved by {approved_by}")
            return True

    def issue_certificate(
        self,
        request_id: str,
        policy_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Certificate:
        """
        Issue a certificate from an approved request.

        Args:
            request_id: Request ID
            policy_id: Policy to apply
            tags: Optional tags

        Returns:
            Issued Certificate
        """
        with self._lock:
            if request_id not in self._requests:
                raise ValueError(f"Request not found: {request_id}")

            request = self._requests[request_id]

            if not request.approved:
                raise ValueError("Request not approved")

            # Get policy
            policy = self._get_policy(request.cert_type, policy_id)

            # Generate serial number
            serial_number = self._generate_serial_number()

            # Calculate validity
            not_before = datetime.utcnow()
            not_after = not_before + timedelta(days=policy.validity_days)

            # Generate key pair (or use existing from KMS)
            private_key_id = None
            if self._key_manager:
                try:
                    key_metadata = self._key_manager.generate_key(
                        key_type="signing",
                        tags={"cert_cn": request.common_name},
                    )
                    private_key_id = key_metadata.key_id
                except Exception as e:
                    logger.warning(f"Key generation via KMS failed: {e}")

            # Generate certificate (simulated - in production use CA provider)
            cert_pem = self._generate_certificate_pem(
                request, serial_number, not_before, not_after, policy
            )

            # Calculate fingerprint
            fingerprint = self._calculate_fingerprint(cert_pem)

            # Determine signature algorithm
            sig_alg = self._get_signature_algorithm(policy.key_algorithm)

            # Create certificate
            certificate = Certificate(
                cert_id=str(uuid.uuid4()),
                serial_number=serial_number,
                common_name=request.common_name,
                cert_type=request.cert_type,
                state=CertificateState.ACTIVE,
                subject_alt_names=request.subject_alt_names,
                issuer=self._get_issuer_dn(),
                not_before=not_before,
                not_after=not_after,
                key_algorithm=policy.key_algorithm,
                signature_algorithm=sig_alg,
                fingerprint_sha256=fingerprint,
                cert_pem=cert_pem,
                private_key_id=private_key_id,
                policy_id=policy_id,
                tags=tags or {},
            )

            self._certificates[certificate.cert_id] = certificate

            # Clean up request
            del self._requests[request_id]

            logger.info(
                f"Issued certificate {certificate.cert_id} for {request.common_name}"
            )

            return certificate

    def renew_certificate(
        self,
        cert_id: str,
        reason: str = "Scheduled renewal",
        automatic: bool = False,
    ) -> RenewalEvent:
        """
        Renew a certificate.

        Args:
            cert_id: Certificate to renew
            reason: Reason for renewal
            automatic: Whether this is automatic renewal

        Returns:
            RenewalEvent
        """
        with self._lock:
            if cert_id not in self._certificates:
                raise ValueError(f"Certificate not found: {cert_id}")

            old_cert = self._certificates[cert_id]

            # Create renewal request
            request = self.create_certificate_request(
                common_name=old_cert.common_name,
                cert_type=old_cert.cert_type,
                subject_alt_names=old_cert.subject_alt_names,
                policy_id=old_cert.policy_id,
            )

            # Auto-approve renewals
            request.approved = True
            request.approved_by = "auto-renewal" if automatic else "manual-renewal"

            # Issue new certificate
            new_cert = self.issue_certificate(
                request_id=request.request_id,
                policy_id=old_cert.policy_id,
                tags=old_cert.tags,
            )

            # Link certificates
            new_cert.renewal_of = cert_id
            old_cert.renewed_by = new_cert.cert_id
            old_cert.state = CertificateState.RENEWED

            # Create renewal event
            event = RenewalEvent(
                event_id=str(uuid.uuid4()),
                old_cert_id=cert_id,
                new_cert_id=new_cert.cert_id,
                renewed_at=datetime.utcnow(),
                automatic=automatic,
                reason=reason,
                success=True,
            )

            self._renewal_events.append(event)

            # Notify callback
            if self._renewal_callback:
                try:
                    self._renewal_callback(event)
                except Exception as e:
                    logger.error(f"Renewal callback failed: {e}")

            logger.info(f"Renewed certificate {cert_id} -> {new_cert.cert_id}")

            return event

    def revoke_certificate(
        self,
        cert_id: str,
        reason: RevocationReason = RevocationReason.UNSPECIFIED,
        revoked_by: str = "",
    ) -> bool:
        """
        Revoke a certificate.

        Args:
            cert_id: Certificate to revoke
            reason: Revocation reason
            revoked_by: Who revoked it

        Returns:
            True if successful
        """
        with self._lock:
            if cert_id not in self._certificates:
                return False

            cert = self._certificates[cert_id]
            cert.state = CertificateState.REVOKED
            cert.revoked_at = datetime.utcnow()
            cert.revocation_reason = reason

            # Destroy private key if managed
            if cert.private_key_id and self._key_manager:
                try:
                    self._key_manager.destroy_key(
                        cert.private_key_id,
                        reason=f"Certificate revoked: {reason.name}",
                    )
                except Exception as e:
                    logger.error(f"Key destruction failed: {e}")

            logger.warning(
                f"Revoked certificate {cert_id}: {reason.name} by {revoked_by}"
            )

            return True

    def get_certificate(self, cert_id: str) -> Optional[Certificate]:
        """Get certificate by ID."""
        return self._certificates.get(cert_id)

    def find_certificate_by_cn(self, common_name: str) -> List[Certificate]:
        """Find certificates by common name."""
        return [
            c for c in self._certificates.values()
            if c.common_name == common_name and c.state == CertificateState.ACTIVE
        ]

    def list_certificates(
        self,
        cert_type: Optional[CertificateType] = None,
        state: Optional[CertificateState] = None,
    ) -> List[Certificate]:
        """List certificates with optional filtering."""
        certs = list(self._certificates.values())

        if cert_type:
            certs = [c for c in certs if c.cert_type == cert_type]

        if state:
            certs = [c for c in certs if c.state == state]

        return certs

    def get_expiring_certificates(
        self,
        days: int = 30,
    ) -> List[Certificate]:
        """Get certificates expiring within specified days."""
        return [
            c for c in self._certificates.values()
            if c.state == CertificateState.ACTIVE and c.is_expiring_soon(days)
        ]

    def start_renewal_scheduler(
        self,
        check_interval_seconds: int = 86400,  # Daily
    ) -> None:
        """Start background renewal scheduler."""
        if self._renewal_thread and self._renewal_thread.is_alive():
            return

        self._shutdown_event.clear()

        def renewal_loop():
            while not self._shutdown_event.is_set():
                try:
                    self._check_and_renew()
                except Exception as e:
                    logger.error(f"Renewal check failed: {e}")

                self._shutdown_event.wait(check_interval_seconds)

        self._renewal_thread = threading.Thread(
            target=renewal_loop,
            daemon=True,
            name="cert-renewal-scheduler",
        )
        self._renewal_thread.start()
        logger.info("Certificate renewal scheduler started")

    def stop_renewal_scheduler(self) -> None:
        """Stop the renewal scheduler."""
        self._shutdown_event.set()
        if self._renewal_thread:
            self._renewal_thread.join(timeout=5)
        logger.info("Certificate renewal scheduler stopped")

    def _check_and_renew(self) -> None:
        """Check for certificates needing renewal and renew them."""
        for cert_id, cert in list(self._certificates.items()):
            if cert.state != CertificateState.ACTIVE:
                continue

            # Get policy
            policy = self._get_policy(cert.cert_type, cert.policy_id)

            if not policy.auto_renew:
                continue

            # Check if renewal is due
            if cert.is_expiring_soon(policy.renewal_threshold_days):
                try:
                    # Update state to expiring
                    cert.state = CertificateState.EXPIRING

                    # Renew
                    self.renew_certificate(
                        cert_id,
                        "Automatic scheduled renewal",
                        automatic=True,
                    )
                except Exception as e:
                    logger.error(f"Auto-renewal failed for {cert_id}: {e}")

    def _init_default_policies(self) -> None:
        """Initialize default certificate policies."""
        self._policies["default_tls"] = CertificatePolicy(
            cert_type=CertificateType.TLS_SERVER,
            validity_days=90,  # Short-lived for security
            renewal_threshold_days=30,
            key_algorithm="ML-DSA-65",
            auto_renew=True,
        )

        self._policies["default_client"] = CertificatePolicy(
            cert_type=CertificateType.TLS_CLIENT,
            validity_days=365,
            renewal_threshold_days=30,
            key_algorithm="ML-DSA-65",
            auto_renew=True,
        )

        self._policies["default_signing"] = CertificatePolicy(
            cert_type=CertificateType.CODE_SIGNING,
            validity_days=365,
            renewal_threshold_days=60,
            key_algorithm="ML-DSA-65",
            auto_renew=True,
            require_approval=True,  # Code signing needs approval
        )

    def _get_policy(
        self,
        cert_type: CertificateType,
        policy_id: Optional[str],
    ) -> CertificatePolicy:
        """Get policy for certificate type."""
        if policy_id and policy_id in self._policies:
            return self._policies[policy_id]

        # Return type-specific default
        default_map = {
            CertificateType.TLS_SERVER: "default_tls",
            CertificateType.TLS_CLIENT: "default_client",
            CertificateType.CODE_SIGNING: "default_signing",
        }

        policy_key = default_map.get(cert_type, "default_tls")
        return self._policies.get(policy_key, CertificatePolicy(cert_type=cert_type))

    def _generate_serial_number(self) -> str:
        """Generate unique serial number."""
        return hashlib.sha256(
            f"{uuid.uuid4()}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:40]

    def _generate_certificate_pem(
        self,
        request: CertificateRequest,
        serial_number: str,
        not_before: datetime,
        not_after: datetime,
        policy: CertificatePolicy,
    ) -> str:
        """Generate certificate PEM (simulated)."""
        # In production, this would use the CA provider to sign
        # For now, return a placeholder
        cert_data = {
            "version": 3,
            "serial": serial_number,
            "subject": f"CN={request.common_name},O={request.organization}",
            "issuer": self._get_issuer_dn(),
            "not_before": not_before.isoformat(),
            "not_after": not_after.isoformat(),
            "key_algorithm": policy.key_algorithm,
            "extensions": {
                "subject_alt_names": request.subject_alt_names,
                "key_usage": ["digitalSignature", "keyEncipherment"],
                "extended_key_usage": policy.extended_key_usage or ["serverAuth"],
            },
        }

        # Encode as base64 (simulated PEM)
        import json
        encoded = base64.b64encode(json.dumps(cert_data).encode()).decode()

        return f"""-----BEGIN CERTIFICATE-----
{encoded}
-----END CERTIFICATE-----"""

    def _calculate_fingerprint(self, cert_pem: str) -> str:
        """Calculate SHA-256 fingerprint."""
        return hashlib.sha256(cert_pem.encode()).hexdigest()

    def _get_signature_algorithm(self, key_algorithm: str) -> str:
        """Get signature algorithm for key algorithm."""
        algo_map = {
            "ML-DSA-65": "ML-DSA-65",
            "ML-DSA-44": "ML-DSA-44",
            "ML-DSA-87": "ML-DSA-87",
            "ECDSA-P256": "SHA256withECDSA",
            "ECDSA-P384": "SHA384withECDSA",
            "RSA-2048": "SHA256withRSA",
            "RSA-4096": "SHA384withRSA",
        }
        return algo_map.get(key_algorithm, key_algorithm)

    def _get_issuer_dn(self) -> str:
        """Get issuer distinguished name."""
        return "CN=Qbitel AI Root CA,O=Qbitel AI,C=US"

    def get_renewal_history(
        self,
        cert_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[RenewalEvent]:
        """Get renewal history."""
        events = self._renewal_events

        if cert_id:
            events = [
                e for e in events
                if e.old_cert_id == cert_id or e.new_cert_id == cert_id
            ]

        return events[-limit:]

    def get_certificate_chain(self, cert_id: str) -> List[Certificate]:
        """Get certificate chain (cert + renewals)."""
        chain = []
        current_id = cert_id

        while current_id:
            cert = self._certificates.get(current_id)
            if not cert:
                break
            chain.append(cert)
            current_id = cert.renewal_of

        return chain

    def validate_certificate(
        self,
        cert_id: str,
    ) -> Dict[str, Any]:
        """Validate a certificate."""
        cert = self._certificates.get(cert_id)
        if not cert:
            return {"valid": False, "error": "Certificate not found"}

        issues = []

        # Check state
        if cert.state == CertificateState.REVOKED:
            issues.append("Certificate is revoked")
        elif cert.state == CertificateState.EXPIRED:
            issues.append("Certificate is expired")

        # Check expiry
        now = datetime.utcnow()
        if now < cert.not_before:
            issues.append("Certificate not yet valid")
        elif now > cert.not_after:
            issues.append("Certificate has expired")
            cert.state = CertificateState.EXPIRED

        # Check PQC readiness
        pqc_algorithms = {"ML-DSA-44", "ML-DSA-65", "ML-DSA-87"}
        is_pqc = cert.key_algorithm in pqc_algorithms

        return {
            "valid": len(issues) == 0,
            "cert_id": cert_id,
            "state": cert.state.value,
            "days_until_expiry": cert.days_until_expiry(),
            "is_pqc_ready": is_pqc,
            "issues": issues,
        }
