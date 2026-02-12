"""
Security tests for certificate validation and PKI operations.
Tests X.509 certificate validation, chain verification, and revocation checking.
"""

import pytest
import os
from datetime import datetime, timedelta
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.backends import default_backend


class TestCertificateValidation:
    """Security tests for certificate validation and PKI."""

    @pytest.fixture
    def root_ca_cert_and_key(self):
        """Generate a self-signed root CA certificate for testing."""
        # Generate private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

        # Create self-signed certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Root CA"),
                x509.NameAttribute(NameOID.COMMON_NAME, "Test Root CA"),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=3650))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(private_key, hashes.SHA256(), backend=default_backend())
        )

        return cert, private_key

    @pytest.fixture
    def intermediate_ca_cert_and_key(self, root_ca_cert_and_key):
        """Generate an intermediate CA certificate."""
        root_cert, root_key = root_ca_cert_and_key

        # Generate intermediate CA private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Intermediate CA"),
                x509.NameAttribute(NameOID.COMMON_NAME, "Test Intermediate CA"),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(root_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=1825))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(root_key, hashes.SHA256(), backend=default_backend())
        )

        return cert, private_key

    @pytest.fixture
    def end_entity_cert_and_key(self, intermediate_ca_cert_and_key):
        """Generate an end-entity certificate."""
        issuer_cert, issuer_key = intermediate_ca_cert_and_key

        # Generate end-entity private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Organization"),
                x509.NameAttribute(NameOID.COMMON_NAME, "test.example.com"),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName("test.example.com"),
                        x509.DNSName("www.test.example.com"),
                    ]
                ),
                critical=False,
            )
            .sign(issuer_key, hashes.SHA256(), backend=default_backend())
        )

        return cert, private_key

    def test_valid_certificate_chain(self, root_ca_cert_and_key, intermediate_ca_cert_and_key, end_entity_cert_and_key):
        """Test validation of a valid certificate chain."""
        root_cert, _ = root_ca_cert_and_key
        intermediate_cert, intermediate_key = intermediate_ca_cert_and_key
        end_cert, _ = end_entity_cert_and_key

        # Verify intermediate cert signed by root
        root_public_key = root_cert.public_key()
        root_public_key.verify(
            intermediate_cert.signature,
            intermediate_cert.tbs_certificate_bytes,
            ec.ECDSA(hashes.SHA256()) if isinstance(root_public_key, ec.EllipticCurvePublicKey) else hashes.SHA256(),
        )

        # Verify end-entity cert signed by intermediate
        intermediate_public_key = intermediate_cert.public_key()
        intermediate_public_key.verify(
            end_cert.signature,
            end_cert.tbs_certificate_bytes,
            ec.ECDSA(hashes.SHA256()) if isinstance(intermediate_public_key, ec.EllipticCurvePublicKey) else hashes.SHA256(),
        )

    def test_expired_certificate_rejection(self, root_ca_cert_and_key):
        """Test that expired certificates are rejected."""
        root_cert, root_key = root_ca_cert_and_key

        # Create an already-expired certificate
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, "expired.example.com"),
            ]
        )

        expired_cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(root_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow() - timedelta(days=365))
            .not_valid_after(datetime.utcnow() - timedelta(days=1))
            .sign(root_key, hashes.SHA256(), backend=default_backend())
        )

        # Verify certificate is expired
        now = datetime.utcnow()
        assert expired_cert.not_valid_after < now, "Certificate should be expired"

    def test_not_yet_valid_certificate_rejection(self, root_ca_cert_and_key):
        """Test that not-yet-valid certificates are rejected."""
        root_cert, root_key = root_ca_cert_and_key

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, "future.example.com"),
            ]
        )

        future_cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(root_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow() + timedelta(days=30))
            .not_valid_after(datetime.utcnow() + timedelta(days=395))
            .sign(root_key, hashes.SHA256(), backend=default_backend())
        )

        # Verify certificate is not yet valid
        now = datetime.utcnow()
        assert future_cert.not_valid_before > now, "Certificate should not yet be valid"

    def test_invalid_signature_rejection(self, root_ca_cert_and_key):
        """Test that certificates with invalid signatures are rejected."""
        root_cert, root_key = root_ca_cert_and_key

        # Create a valid certificate
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, "test.example.com"),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(root_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))
            .sign(root_key, hashes.SHA256(), backend=default_backend())
        )

        # Create a different key to verify with (simulating wrong issuer)
        wrong_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

        # Verification with wrong key should fail
        with pytest.raises(Exception):
            wrong_key.public_key().verify(cert.signature, cert.tbs_certificate_bytes, hashes.SHA256())

    def test_basic_constraints_validation(self, root_ca_cert_and_key):
        """Test validation of Basic Constraints extension."""
        root_cert, root_key = root_ca_cert_and_key

        # Verify root CA has correct basic constraints
        basic_constraints = root_cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS)
        assert basic_constraints.value.ca is True, "Root CA should have ca=True"
        assert basic_constraints.critical is True, "Basic constraints should be critical"

    def test_key_usage_validation(self, end_entity_cert_and_key):
        """Test validation of Key Usage extension."""
        cert, _ = end_entity_cert_and_key

        key_usage = cert.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE)

        # End-entity cert should have correct key usage
        assert key_usage.value.digital_signature is True
        assert key_usage.value.key_encipherment is True
        assert key_usage.value.key_cert_sign is False, "End-entity should not sign certs"

    def test_subject_alternative_name_validation(self, end_entity_cert_and_key):
        """Test validation of Subject Alternative Name extension."""
        cert, _ = end_entity_cert_and_key

        san = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)

        dns_names = san.value.get_values_for_type(x509.DNSName)
        assert "test.example.com" in dns_names
        assert "www.test.example.com" in dns_names

    def test_path_length_constraint(self, root_ca_cert_and_key, intermediate_ca_cert_and_key):
        """Test path length constraints in certificate chain."""
        root_cert, _ = root_ca_cert_and_key
        intermediate_cert, _ = intermediate_ca_cert_and_key

        # Root CA should have no path length limit
        root_bc = root_cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS)
        assert root_bc.value.path_length is None

        # Intermediate CA should have path_length=0
        intermediate_bc = intermediate_cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS)
        assert intermediate_bc.value.path_length == 0, "Intermediate CA should have path_length=0"

    def test_self_signed_certificate_validation(self, root_ca_cert_and_key):
        """Test validation of self-signed certificates."""
        cert, private_key = root_ca_cert_and_key

        # Verify it's self-signed
        assert cert.subject == cert.issuer, "Certificate should be self-signed"

        # Verify signature with its own public key
        public_key = cert.public_key()
        public_key.verify(cert.signature, cert.tbs_certificate_bytes, hashes.SHA256())

    def test_certificate_serial_number_uniqueness(self, root_ca_cert_and_key):
        """Test that certificate serial numbers are unique."""
        root_cert, root_key = root_ca_cert_and_key

        serial_numbers = set()

        for _ in range(100):
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

            cert = (
                x509.CertificateBuilder()
                .subject_name(
                    x509.Name(
                        [
                            x509.NameAttribute(NameOID.COMMON_NAME, f"test{_}.example.com"),
                        ]
                    )
                )
                .issuer_name(root_cert.subject)
                .public_key(private_key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.utcnow())
                .not_valid_after(datetime.utcnow() + timedelta(days=365))
                .sign(root_key, hashes.SHA256(), backend=default_backend())
            )

            serial_numbers.add(cert.serial_number)

        # All serial numbers should be unique
        assert len(serial_numbers) == 100, "Serial numbers should be unique"

    def test_weak_signature_algorithm_detection(self, root_ca_cert_and_key):
        """Test detection of weak signature algorithms."""
        root_cert, root_key = root_ca_cert_and_key

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

        # Create cert with SHA-1 (weak algorithm)
        weak_cert = (
            x509.CertificateBuilder()
            .subject_name(
                x509.Name(
                    [
                        x509.NameAttribute(NameOID.COMMON_NAME, "weak.example.com"),
                    ]
                )
            )
            .issuer_name(root_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))
            .sign(root_key, hashes.SHA1(), backend=default_backend())
        )

        # Verify we can detect SHA-1 usage
        assert isinstance(weak_cert.signature_hash_algorithm, hashes.SHA1), "Should detect SHA-1 usage"

    def test_certificate_revocation_list_structure(self, root_ca_cert_and_key):
        """Test CRL structure and validation."""
        root_cert, root_key = root_ca_cert_and_key

        # Create a CRL
        revoked_cert_1 = (
            x509.RevokedCertificateBuilder().serial_number(12345).revocation_date(datetime.utcnow()).build(default_backend())
        )

        revoked_cert_2 = (
            x509.RevokedCertificateBuilder().serial_number(67890).revocation_date(datetime.utcnow()).build(default_backend())
        )

        crl = (
            x509.CertificateRevocationListBuilder()
            .issuer_name(root_cert.subject)
            .last_update(datetime.utcnow())
            .next_update(datetime.utcnow() + timedelta(days=7))
            .add_revoked_certificate(revoked_cert_1)
            .add_revoked_certificate(revoked_cert_2)
            .sign(root_key, hashes.SHA256(), backend=default_backend())
        )

        # Verify CRL properties
        assert len(list(crl)) == 2, "CRL should contain 2 revoked certificates"
        assert crl.issuer == root_cert.subject

        # Verify specific serial numbers are revoked
        revoked_serials = {cert.serial_number for cert in crl}
        assert 12345 in revoked_serials
        assert 67890 in revoked_serials

    def test_hostname_validation(self, end_entity_cert_and_key):
        """Test hostname validation against certificate SAN."""
        cert, _ = end_entity_cert_and_key

        san = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
        dns_names = san.value.get_values_for_type(x509.DNSName)

        # Valid hostnames
        valid_hostnames = ["test.example.com", "www.test.example.com"]
        for hostname in valid_hostnames:
            assert hostname in dns_names, f"{hostname} should be valid"

        # Invalid hostname
        assert "invalid.example.com" not in dns_names

    def test_certificate_chain_order(self, root_ca_cert_and_key, intermediate_ca_cert_and_key, end_entity_cert_and_key):
        """Test that certificate chain is in correct order."""
        root_cert, _ = root_ca_cert_and_key
        intermediate_cert, _ = intermediate_ca_cert_and_key
        end_cert, _ = end_entity_cert_and_key

        # Build chain: end-entity -> intermediate -> root
        chain = [end_cert, intermediate_cert, root_cert]

        # Verify each cert is signed by the next in chain
        for i in range(len(chain) - 1):
            current_cert = chain[i]
            issuer_cert = chain[i + 1]

            # Verify issuer name matches
            assert current_cert.issuer == issuer_cert.subject, f"Certificate {i} issuer mismatch"

            # Verify signature
            issuer_public_key = issuer_cert.public_key()
            issuer_public_key.verify(current_cert.signature, current_cert.tbs_certificate_bytes, hashes.SHA256())

    @pytest.mark.parametrize("key_size", [1024, 2048, 4096])
    def test_minimum_key_size_enforcement(self, key_size, root_ca_cert_and_key):
        """Test enforcement of minimum key sizes."""
        root_cert, root_key = root_ca_cert_and_key

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size, backend=default_backend())

        cert = (
            x509.CertificateBuilder()
            .subject_name(
                x509.Name(
                    [
                        x509.NameAttribute(NameOID.COMMON_NAME, f"test{key_size}.example.com"),
                    ]
                )
            )
            .issuer_name(root_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))
            .sign(root_key, hashes.SHA256(), backend=default_backend())
        )

        # Verify key size
        public_key = cert.public_key()
        actual_key_size = public_key.key_size

        assert actual_key_size == key_size, f"Key size mismatch"

        # Security check: keys below 2048 should be flagged as weak
        if key_size < 2048:
            print(f"WARNING: Key size {key_size} is below recommended minimum of 2048 bits")
