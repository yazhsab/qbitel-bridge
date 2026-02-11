"""
Quantum Certificate Manager

Manages quantum-safe certificates for Istio service mesh using post-quantum
cryptography (Kyber for key exchange, Dilithium for signatures).
"""

import base64
import datetime
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

# Post-quantum cryptography imports
from kyber_py.kyber import Kyber512, Kyber768, Kyber1024
from dilithium_py.dilithium import Dilithium2, Dilithium3, Dilithium5

logger = logging.getLogger(__name__)


class CertificateAlgorithm(Enum):
    """Supported quantum-safe algorithms"""
    KYBER_512 = "kyber-512"
    KYBER_768 = "kyber-768"
    KYBER_1024 = "kyber-1024"
    DILITHIUM_2 = "dilithium-2"
    DILITHIUM_3 = "dilithium-3"
    DILITHIUM_5 = "dilithium-5"


@dataclass
class CertificateMetadata:
    """Metadata for quantum-safe certificates"""
    service_name: str
    namespace: str
    valid_from: datetime.datetime
    valid_until: datetime.datetime
    key_algorithm: CertificateAlgorithm
    signature_algorithm: CertificateAlgorithm
    subject: str
    issuer: str
    serial_number: str
    fingerprint: str


class QuantumCertificateManager:
    """
    Manages quantum-safe certificates for Istio service mesh.

    Provides automatic certificate generation, rotation, and distribution
    using post-quantum cryptographic algorithms (Kyber + Dilithium).
    """

    def __init__(
        self,
        key_algorithm: CertificateAlgorithm = CertificateAlgorithm.KYBER_1024,
        signature_algorithm: CertificateAlgorithm = CertificateAlgorithm.DILITHIUM_5,
        cert_validity_days: int = 90,
        rotation_threshold_days: int = 30,
        auto_rotation: bool = True
    ):
        """
        Initialize the Quantum Certificate Manager.

        Args:
            key_algorithm: Post-quantum key exchange algorithm
            signature_algorithm: Post-quantum signature algorithm
            cert_validity_days: Certificate validity period in days
            rotation_threshold_days: Days before expiry to trigger rotation
            auto_rotation: Enable automatic certificate rotation
        """
        self.key_algorithm = key_algorithm
        self.signature_algorithm = signature_algorithm
        self.cert_validity_days = cert_validity_days
        self.rotation_threshold_days = rotation_threshold_days
        self.auto_rotation = auto_rotation
        self._certificate_cache: Dict[str, Dict[str, Any]] = {}

        # Map signature algorithms to their implementations
        self._signature_algorithms = {
            CertificateAlgorithm.DILITHIUM_2: Dilithium2,
            CertificateAlgorithm.DILITHIUM_3: Dilithium3,
            CertificateAlgorithm.DILITHIUM_5: Dilithium5
        }

        logger.info(
            f"Initialized QuantumCertificateManager with "
            f"key_algo={key_algorithm.value}, "
            f"sig_algo={signature_algorithm.value}"
        )

    def generate_root_ca(
        self,
        subject: str = "CN=QBITEL Root CA,O=QBITEL,C=US",
        validity_years: int = 10
    ) -> Dict[str, Any]:
        """
        Generate quantum-safe root CA certificate.

        Args:
            subject: Certificate subject DN
            validity_years: Validity period in years

        Returns:
            Dict containing CA certificate and private keys
        """
        logger.info(f"Generating quantum-safe Root CA: {subject}")

        # Generate quantum-safe key pair (using Kyber KEM for key exchange)
        kem_private_key, kem_public_key = self._generate_key_pair(self.key_algorithm)

        # Generate signature key pair (using Dilithium for signing)
        sig_private_key, sig_public_key = self._generate_signature_key_pair(self.signature_algorithm)

        # Calculate validity period
        valid_from = datetime.datetime.utcnow()
        valid_until = valid_from + datetime.timedelta(days=validity_years * 365)

        # Generate serial number
        serial_number = self._generate_serial_number()

        # Create certificate structure
        cert_data = {
            "version": "v3",
            "serialNumber": serial_number,
            "issuer": subject,
            "subject": subject,
            "notBefore": valid_from.isoformat(),
            "notAfter": valid_until.isoformat(),
            "publicKey": {
                "algorithm": self.key_algorithm.value,
                "keyData": base64.b64encode(kem_public_key).decode()
            },
            "signaturePublicKey": {
                "algorithm": self.signature_algorithm.value,
                "keyData": base64.b64encode(sig_public_key).decode()
            },
            "extensions": {
                "basicConstraints": {
                    "critical": True,
                    "ca": True,
                    "pathLenConstraint": None
                },
                "keyUsage": {
                    "critical": True,
                    "keyCertSign": True,
                    "crlSign": True
                }
            }
        }

        # Sign certificate with Dilithium (self-signed for root CA)
        signature = self._sign_certificate(cert_data, sig_private_key)

        certificate = {
            "certificate": cert_data,
            "signature": {
                "algorithm": self.signature_algorithm.value,
                "value": base64.b64encode(signature).decode()
            }
        }

        # Create PEM-encoded certificate and keys
        pem_cert = self._encode_pem(certificate, "QUANTUM CERTIFICATE")
        pem_kem_key = self._encode_pem(
            {"algorithm": self.key_algorithm.value, "key": base64.b64encode(kem_private_key).decode()},
            "QUANTUM KEM PRIVATE KEY"
        )
        pem_sig_key = self._encode_pem(
            {"algorithm": self.signature_algorithm.value, "key": base64.b64encode(sig_private_key).decode()},
            "QUANTUM SIGNATURE PRIVATE KEY"
        )

        result = {
            "certificate": cert_data,
            "certificate_pem": pem_cert,
            "kem_private_key_pem": pem_kem_key,
            "signature_private_key_pem": pem_sig_key,
            "kem_public_key": base64.b64encode(kem_public_key).decode(),
            "signature_public_key": base64.b64encode(sig_public_key).decode(),
            "metadata": CertificateMetadata(
                service_name="root-ca",
                namespace="qbitel-system",
                valid_from=valid_from,
                valid_until=valid_until,
                key_algorithm=self.key_algorithm,
                signature_algorithm=self.signature_algorithm,
                subject=subject,
                issuer=subject,
                serial_number=serial_number,
                fingerprint=self._calculate_fingerprint(certificate)
            ).__dict__
        }

        logger.info(
            f"Generated Root CA with serial: {serial_number}, "
            f"KEM: {self.key_algorithm.value}, Signature: {self.signature_algorithm.value}"
        )
        return result

    def generate_service_certificate(
        self,
        service_name: str,
        namespace: str,
        ca_cert: Dict[str, Any],
        ca_signature_private_key: str,
        sans: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate quantum-safe certificate for a service in the mesh.

        Args:
            service_name: Name of the Kubernetes service
            namespace: Namespace of the service
            ca_cert: Root CA certificate
            ca_signature_private_key: Root CA Dilithium private key for signing
            sans: Subject Alternative Names (DNS names, IPs)

        Returns:
            Dict containing service certificate and private keys
        """
        subject = f"CN={service_name}.{namespace}.svc.cluster.local,O=QBITEL"
        logger.info(f"Generating service certificate: {subject}")

        # Generate service Kyber key pair for key exchange
        kem_private_key, kem_public_key = self._generate_key_pair(self.key_algorithm)

        # Generate service Dilithium key pair for signing/verification
        sig_private_key, sig_public_key = self._generate_signature_key_pair(self.signature_algorithm)

        # Calculate validity period
        valid_from = datetime.datetime.utcnow()
        valid_until = valid_from + datetime.timedelta(days=self.cert_validity_days)

        # Generate serial number
        serial_number = self._generate_serial_number()

        # Default SANs
        if sans is None:
            sans = [
                f"{service_name}.{namespace}.svc.cluster.local",
                f"{service_name}.{namespace}.svc",
                f"{service_name}.{namespace}",
                service_name
            ]

        # Create certificate structure
        cert_data = {
            "version": "v3",
            "serialNumber": serial_number,
            "issuer": ca_cert["certificate"]["subject"],
            "subject": subject,
            "notBefore": valid_from.isoformat(),
            "notAfter": valid_until.isoformat(),
            "publicKey": {
                "algorithm": self.key_algorithm.value,
                "keyData": base64.b64encode(kem_public_key).decode()
            },
            "signaturePublicKey": {
                "algorithm": self.signature_algorithm.value,
                "keyData": base64.b64encode(sig_public_key).decode()
            },
            "extensions": {
                "basicConstraints": {
                    "critical": True,
                    "ca": False
                },
                "keyUsage": {
                    "critical": True,
                    "digitalSignature": True,
                    "keyEncipherment": True
                },
                "extendedKeyUsage": {
                    "serverAuth": True,
                    "clientAuth": True
                },
                "subjectAltName": {
                    "dnsNames": sans
                }
            }
        }

        # Sign certificate with CA Dilithium private key
        ca_sig_key_dict = self._decode_pem(ca_signature_private_key)
        ca_sig_key_data = base64.b64decode(ca_sig_key_dict["key"])
        signature = self._sign_certificate(cert_data, ca_sig_key_data)

        certificate = {
            "certificate": cert_data,
            "signature": {
                "algorithm": self.signature_algorithm.value,
                "value": base64.b64encode(signature).decode()
            }
        }

        # Create PEM-encoded certificate and keys
        pem_cert = self._encode_pem(certificate, "QUANTUM CERTIFICATE")
        pem_kem_key = self._encode_pem(
            {"algorithm": self.key_algorithm.value, "key": base64.b64encode(kem_private_key).decode()},
            "QUANTUM KEM PRIVATE KEY"
        )
        pem_sig_key = self._encode_pem(
            {"algorithm": self.signature_algorithm.value, "key": base64.b64encode(sig_private_key).decode()},
            "QUANTUM SIGNATURE PRIVATE KEY"
        )

        result = {
            "certificate_pem": pem_cert,
            "kem_private_key_pem": pem_kem_key,
            "signature_private_key_pem": pem_sig_key,
            "kem_public_key": base64.b64encode(kem_public_key).decode(),
            "signature_public_key": base64.b64encode(sig_public_key).decode(),
            "ca_certificate_pem": ca_cert.get("certificate_pem", ""),
            "metadata": CertificateMetadata(
                service_name=service_name,
                namespace=namespace,
                valid_from=valid_from,
                valid_until=valid_until,
                key_algorithm=self.key_algorithm,
                signature_algorithm=self.signature_algorithm,
                subject=subject,
                issuer=ca_cert["certificate"]["subject"],
                serial_number=serial_number,
                fingerprint=self._calculate_fingerprint(certificate)
            ).__dict__
        }

        # Cache certificate
        cache_key = f"{namespace}/{service_name}"
        self._certificate_cache[cache_key] = result

        logger.info(
            f"Generated service certificate with serial: {serial_number}, "
            f"KEM: {self.key_algorithm.value}, Signature: {self.signature_algorithm.value}"
        )
        return result

    def create_kubernetes_secret(
        self,
        cert_data: Dict[str, Any],
        secret_name: str,
        namespace: str
    ) -> Dict[str, Any]:
        """
        Create Kubernetes Secret manifest for certificate storage.

        Args:
            cert_data: Certificate data from generate_service_certificate
            secret_name: Name of the Kubernetes secret
            namespace: Target namespace

        Returns:
            Dict containing Kubernetes Secret manifest
        """
        secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": secret_name,
                "namespace": namespace,
                "labels": {
                    "app": "qbitel",
                    "component": "quantum-certs"
                },
                "annotations": {
                    "qbitel.ai/cert-serial": cert_data["metadata"]["serial_number"],
                    "qbitel.ai/valid-until": cert_data["metadata"]["valid_until"],
                    "qbitel.ai/key-algorithm": cert_data["metadata"]["key_algorithm"],
                    "qbitel.ai/signature-algorithm": cert_data["metadata"]["signature_algorithm"]
                }
            },
            "type": "kubernetes.io/tls",
            "data": {
                "tls.crt": base64.b64encode(cert_data["certificate_pem"].encode()).decode(),
                "tls.key": base64.b64encode(cert_data["private_key_pem"].encode()).decode(),
                "ca.crt": base64.b64encode(cert_data["ca_certificate_pem"].encode()).decode()
            }
        }

        return secret

    def check_rotation_needed(self, cert_data: Dict[str, Any]) -> bool:
        """
        Check if certificate rotation is needed.

        Args:
            cert_data: Certificate data

        Returns:
            bool: True if rotation is needed
        """
        valid_until_str = cert_data["metadata"]["valid_until"]
        valid_until = datetime.datetime.fromisoformat(valid_until_str)

        days_until_expiry = (valid_until - datetime.datetime.utcnow()).days

        if days_until_expiry <= self.rotation_threshold_days:
            logger.warning(
                f"Certificate rotation needed: "
                f"{days_until_expiry} days until expiry"
            )
            return True

        return False

    def _generate_key_pair(
        self,
        algorithm: CertificateAlgorithm
    ) -> Tuple[bytes, bytes]:
        """
        Generate quantum-safe key pair using real Kyber implementation.

        Args:
            algorithm: Key generation algorithm

        Returns:
            Tuple of (private_key, public_key) as bytes
        """
        logger.info(f"Generating quantum-safe key pair with {algorithm.value}")

        if algorithm == CertificateAlgorithm.KYBER_512:
            # Use real Kyber-512 implementation
            public_key, private_key = Kyber512.keygen()
            logger.info(f"Generated Kyber-512 keys: pk={len(public_key)} bytes, sk={len(private_key)} bytes")

        elif algorithm == CertificateAlgorithm.KYBER_768:
            # Use real Kyber-768 implementation
            public_key, private_key = Kyber768.keygen()
            logger.info(f"Generated Kyber-768 keys: pk={len(public_key)} bytes, sk={len(private_key)} bytes")

        elif algorithm == CertificateAlgorithm.KYBER_1024:
            # Use real Kyber-1024 implementation (highest security)
            public_key, private_key = Kyber1024.keygen()
            logger.info(f"Generated Kyber-1024 keys: pk={len(public_key)} bytes, sk={len(private_key)} bytes")

        else:
            # Fallback to Kyber-1024 for maximum security
            logger.warning(f"Unknown algorithm {algorithm.value}, defaulting to Kyber-1024")
            public_key, private_key = Kyber1024.keygen()

        return private_key, public_key

    def _generate_signature_key_pair(
        self,
        algorithm: CertificateAlgorithm
    ) -> Tuple[bytes, bytes]:
        """
        Generate quantum-safe signature key pair using real Dilithium implementation.

        Args:
            algorithm: Signature algorithm

        Returns:
            Tuple of (private_key, public_key) as bytes
        """
        logger.info(f"Generating quantum-safe signature key pair with {algorithm.value}")

        if algorithm == CertificateAlgorithm.DILITHIUM_2:
            public_key, private_key = Dilithium2.keygen()
            logger.info(f"Generated Dilithium-2 keys: pk={len(public_key)} bytes, sk={len(private_key)} bytes")

        elif algorithm == CertificateAlgorithm.DILITHIUM_3:
            public_key, private_key = Dilithium3.keygen()
            logger.info(f"Generated Dilithium-3 keys: pk={len(public_key)} bytes, sk={len(private_key)} bytes")

        elif algorithm == CertificateAlgorithm.DILITHIUM_5:
            public_key, private_key = Dilithium5.keygen()
            logger.info(f"Generated Dilithium-5 keys: pk={len(public_key)} bytes, sk={len(private_key)} bytes")

        else:
            # Fallback to Dilithium-5 for maximum security
            logger.warning(f"Unknown signature algorithm {algorithm.value}, defaulting to Dilithium-5")
            public_key, private_key = Dilithium5.keygen()

        return private_key, public_key

    def _sign_certificate(
        self,
        cert_data: Dict[str, Any],
        private_key: bytes,
        algorithm: Optional[CertificateAlgorithm] = None
    ) -> bytes:
        """
        Sign certificate using quantum-safe Dilithium signature algorithm.

        Args:
            cert_data: Certificate data to sign
            private_key: Dilithium private key for signing
            algorithm: Signature algorithm (defaults to self.signature_algorithm)

        Returns:
            Signature bytes
        """
        if algorithm is None:
            algorithm = self.signature_algorithm

        # Serialize certificate data canonically
        cert_json = json.dumps(cert_data, sort_keys=True)
        message = cert_json.encode('utf-8')

        # Get the Dilithium implementation
        dilithium_impl = self._signature_algorithms.get(algorithm, Dilithium5)

        # Sign with real Dilithium implementation
        signature = dilithium_impl.sign(private_key, message)

        logger.info(
            f"Signed certificate with {algorithm.value}: "
            f"message={len(message)} bytes, signature={len(signature)} bytes"
        )

        return signature

    def _verify_certificate_signature(
        self,
        cert_data: Dict[str, Any],
        signature: bytes,
        public_key: bytes,
        algorithm: Optional[CertificateAlgorithm] = None
    ) -> bool:
        """
        Verify certificate signature using Dilithium.

        Args:
            cert_data: Certificate data that was signed
            signature: Signature to verify
            public_key: Dilithium public key
            algorithm: Signature algorithm (defaults to self.signature_algorithm)

        Returns:
            True if signature is valid, False otherwise
        """
        if algorithm is None:
            algorithm = self.signature_algorithm

        # Serialize certificate data canonically (same as signing)
        cert_json = json.dumps(cert_data, sort_keys=True)
        message = cert_json.encode('utf-8')

        # Get the Dilithium implementation
        dilithium_impl = self._signature_algorithms.get(algorithm, Dilithium5)

        try:
            # Verify with real Dilithium implementation
            is_valid = dilithium_impl.verify(public_key, message, signature)
            logger.info(f"Signature verification result: {is_valid}")
            return is_valid
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    def _generate_serial_number(self) -> str:
        """Generate unique serial number for certificate"""
        import secrets
        return secrets.token_hex(16)

    def _calculate_fingerprint(self, certificate: Dict[str, Any]) -> str:
        """Calculate SHA-256 fingerprint of certificate"""
        cert_json = json.dumps(certificate, sort_keys=True)
        fingerprint = hashlib.sha256(cert_json.encode()).hexdigest()
        return fingerprint

    def _encode_pem(self, data: Dict[str, Any], label: str) -> str:
        """
        Encode data in PEM format.

        Args:
            data: Data to encode
            label: PEM label (e.g., "CERTIFICATE", "PRIVATE KEY")

        Returns:
            PEM-encoded string
        """
        json_data = json.dumps(data, indent=2)
        b64_data = base64.b64encode(json_data.encode()).decode()

        # Format as PEM with 64-character lines
        lines = [b64_data[i:i+64] for i in range(0, len(b64_data), 64)]

        pem = f"-----BEGIN {label}-----\n"
        pem += "\n".join(lines)
        pem += f"\n-----END {label}-----\n"

        return pem

    def _decode_pem(self, pem_data: str) -> Dict[str, Any]:
        """
        Decode PEM format back to data.

        Args:
            pem_data: PEM-encoded string

        Returns:
            Decoded data dictionary
        """
        # Extract base64 data between BEGIN and END markers
        lines = pem_data.strip().split('\n')
        b64_lines = [line for line in lines if not line.startswith('-----')]
        b64_data = ''.join(b64_lines)

        # Decode base64 and parse JSON
        json_data = base64.b64decode(b64_data).decode('utf-8')
        return json.loads(json_data)

    def get_certificate(self, service_name: str, namespace: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached certificate for a service.

        Args:
            service_name: Service name
            namespace: Namespace

        Returns:
            Certificate data or None if not found
        """
        cache_key = f"{namespace}/{service_name}"
        return self._certificate_cache.get(cache_key)

    def rotate_certificate(
        self,
        service_name: str,
        namespace: str,
        ca_cert: Dict[str, Any],
        ca_signature_private_key: str
    ) -> Dict[str, Any]:
        """
        Rotate an existing service certificate.

        Args:
            service_name: Service name
            namespace: Namespace
            ca_cert: Root CA certificate
            ca_signature_private_key: Root CA Dilithium signature private key

        Returns:
            New certificate data
        """
        logger.info(f"Rotating certificate for {service_name}.{namespace}")

        # Generate new certificate
        new_cert = self.generate_service_certificate(
            service_name=service_name,
            namespace=namespace,
            ca_cert=ca_cert,
            ca_signature_private_key=ca_signature_private_key
        )

        return new_cert
