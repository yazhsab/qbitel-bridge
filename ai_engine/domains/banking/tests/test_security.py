"""
Tests for Security Module (HSM, Key Management, PKI, PIN Security)

Tests cover:
- HSM provider operations
- Key management lifecycle
- Key derivation and wrapping
- PKI certificate operations
- PIN block handling
- PIN verification
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from ai_engine.domains.banking.security.hsm import (
    HSMConfig,
    HSMKeyType,
    HSMAlgorithm,
    HSMKeyHandle,
    SoftHSM,
    create_hsm_provider,
    HSMError,
    HSMKeyNotFoundError,
)
from ai_engine.domains.banking.security.key_management import (
    KeyManager,
    KeyInfo,
    KeyState,
    KeyPurpose,
    KeyRotationPolicy,
    KeyDerivation,
    KeyWrapping,
)
from ai_engine.domains.banking.security.pki import (
    CertificateManager,
    CertificateInfo,
    CertificateType,
    CertificateStatus,
    CSRBuilder,
    CertificateValidator,
)
from ai_engine.domains.banking.security.pin_security import (
    PINBlock,
    PINBlockFormat,
    PINTranslator,
    PINValidator,
    PVV,
    CVV,
)


class TestSoftHSM:
    """Tests for SoftHSM implementation."""

    @pytest.fixture
    def hsm(self):
        """Create SoftHSM instance."""
        config = HSMConfig(provider_type="softhsm", enable_pqc=True)
        hsm = SoftHSM(config)
        hsm.connect()
        return hsm

    def test_connect_disconnect(self, hsm):
        """Test HSM connection."""
        assert hsm.is_connected is True

        hsm.disconnect()
        assert hsm.is_connected is False

    def test_generate_symmetric_key(self, hsm):
        """Test symmetric key generation."""
        key_handle = hsm.generate_key(
            key_type=HSMKeyType.AES_256,
            label="test_aes_key",
        )

        assert key_handle.key_id is not None
        assert key_handle.key_type == HSMKeyType.AES_256
        assert key_handle.can_encrypt is True
        assert key_handle.can_decrypt is True

    def test_generate_key_pair(self, hsm):
        """Test asymmetric key pair generation."""
        pub_handle, priv_handle = hsm.generate_key_pair(
            key_type=HSMKeyType.RSA_2048,
            label="test_rsa_key",
        )

        assert pub_handle.key_id is not None
        assert priv_handle.key_id is not None
        assert pub_handle.can_encrypt is True
        assert priv_handle.can_decrypt is True
        assert priv_handle.can_sign is True

    def test_encrypt_decrypt_aes(self, hsm):
        """Test AES encryption and decryption."""
        key_handle = hsm.generate_key(
            key_type=HSMKeyType.AES_256,
            label="test_encryption_key",
        )

        plaintext = b"This is a secret message"

        # Encrypt
        result = hsm.encrypt(
            key_handle=key_handle,
            plaintext=plaintext,
            algorithm=HSMAlgorithm.AES_GCM,
        )

        assert result.ciphertext != plaintext
        assert result.iv is not None
        assert result.tag is not None

        # Decrypt
        decrypted = hsm.decrypt(
            key_handle=key_handle,
            ciphertext=result.ciphertext,
            algorithm=HSMAlgorithm.AES_GCM,
            iv=result.iv,
            tag=result.tag,
        )

        assert decrypted.plaintext == plaintext

    def test_sign_verify(self, hsm):
        """Test signing and verification."""
        pub_handle, priv_handle = hsm.generate_key_pair(
            key_type=HSMKeyType.RSA_2048,
            label="test_signing_key",
        )

        data = b"Data to sign"

        # Sign
        signature = hsm.sign(
            key_handle=priv_handle,
            data=data,
            algorithm=HSMAlgorithm.RSA_PSS,
        )

        assert signature.signature is not None
        assert len(signature.signature) > 0

        # Verify
        verification = hsm.verify(
            key_handle=pub_handle,
            data=data,
            signature=signature.signature,
            algorithm=HSMAlgorithm.RSA_PSS,
        )

        assert verification.valid is True

    def test_generate_random(self, hsm):
        """Test random number generation."""
        random1 = hsm.generate_random(32)
        random2 = hsm.generate_random(32)

        assert len(random1) == 32
        assert len(random2) == 32
        assert random1 != random2

    def test_hash(self, hsm):
        """Test hash computation."""
        data = b"Hello, World!"

        hash_result = hsm.hash(data, HSMAlgorithm.SHA256)

        assert len(hash_result) == 32  # SHA-256 produces 32 bytes


class TestHSMFactory:
    """Tests for HSM factory."""

    def test_create_softhsm(self):
        """Test creating SoftHSM via factory."""
        config = HSMConfig(provider_type="softhsm")
        hsm = create_hsm_provider(config)

        assert hsm.provider_name == "SoftHSM"

    def test_invalid_provider(self):
        """Test error for invalid provider."""
        config = HSMConfig(provider_type="invalid_provider")

        with pytest.raises(HSMError):
            create_hsm_provider(config)


class TestKeyManager:
    """Tests for key management."""

    @pytest.fixture
    def key_manager(self):
        """Create KeyManager with SoftHSM."""
        config = HSMConfig(provider_type="softhsm")
        hsm = SoftHSM(config)
        hsm.connect()
        return KeyManager(hsm)

    def test_generate_key(self, key_manager):
        """Test key generation."""
        key_info = key_manager.generate_key(
            alias="test_key",
            key_type=HSMKeyType.AES_256,
            purpose=KeyPurpose.DATA_ENCRYPTION,
            owner="test_user",
        )

        assert key_info.key_id is not None
        assert key_info.alias == "test_key"
        assert key_info.state == KeyState.ACTIVE

    def test_key_lifecycle(self, key_manager):
        """Test key lifecycle transitions."""
        key_info = key_manager.generate_key(
            alias="lifecycle_test",
            key_type=HSMKeyType.AES_256,
            purpose=KeyPurpose.DATA_ENCRYPTION,
            auto_activate=False,
        )

        assert key_info.state == KeyState.PRE_ACTIVATION

        # Activate
        key_manager.activate_key("lifecycle_test")
        key_info = key_manager.get_key("lifecycle_test")
        assert key_info.state == KeyState.ACTIVE

        # Deactivate
        key_manager.deactivate_key("lifecycle_test")
        key_info = key_manager.get_key("lifecycle_test")
        assert key_info.state == KeyState.DEACTIVATED

    def test_key_rotation(self, key_manager):
        """Test key rotation."""
        # Create initial key
        original = key_manager.generate_key(
            alias="rotate_test",
            key_type=HSMKeyType.AES_256,
            purpose=KeyPurpose.DATA_ENCRYPTION,
        )

        # Rotate
        new_key = key_manager.rotate_key("rotate_test", reason="scheduled")

        assert new_key.key_id != original.key_id
        assert new_key.version == original.version + 1
        assert new_key.previous_key_id == original.key_id

    def test_get_key_by_alias(self, key_manager):
        """Test retrieving key by alias."""
        key_manager.generate_key(
            alias="lookup_test",
            key_type=HSMKeyType.AES_256,
            purpose=KeyPurpose.DATA_ENCRYPTION,
        )

        found = key_manager.get_key("lookup_test")

        assert found is not None
        assert found.alias == "lookup_test"


class TestKeyDerivation:
    """Tests for key derivation."""

    def test_hkdf(self):
        """Test HKDF derivation."""
        kdf = KeyDerivation()

        ikm = b"input keying material"
        salt = b"salt value"
        info = b"context info"

        derived = kdf.derive_hkdf(ikm, salt, info, length=32)

        assert len(derived) == 32

        # Same inputs should produce same output
        derived2 = kdf.derive_hkdf(ikm, salt, info, length=32)
        assert derived == derived2

    def test_sp800_108_counter(self):
        """Test SP 800-108 counter mode derivation."""
        kdf = KeyDerivation()

        key = b"base key material!"  # 18 bytes
        label = b"encryption"
        context = b"application context"

        derived = kdf.derive_sp800_108_counter(
            key=key,
            label=label,
            context=context,
            length=32,
        )

        assert len(derived) == 32

    def test_emv_icc_master_key(self):
        """Test EMV ICC Master Key derivation."""
        kdf = KeyDerivation()

        imk = bytes.fromhex("0123456789ABCDEFFEDCBA9876543210")
        pan = "4761111111111111"
        psn = "00"

        icc_mk = kdf.derive_emv_icc_master_key(imk, pan, psn)

        assert len(icc_mk) == 16


class TestKeyWrapping:
    """Tests for key wrapping."""

    def test_aes_key_wrap(self):
        """Test AES key wrap/unwrap."""
        wrapper = KeyWrapping()

        kek = bytes(32)  # 256-bit KEK
        key_to_wrap = bytes(16)  # 128-bit key

        wrapped = wrapper.wrap_aes_key_wrap(kek, key_to_wrap)

        # Wrapped key is 8 bytes longer
        assert len(wrapped) == len(key_to_wrap) + 8

        # Unwrap
        unwrapped = wrapper.unwrap_aes_key_wrap(kek, wrapped)
        assert unwrapped == key_to_wrap


class TestCertificateManager:
    """Tests for certificate management."""

    @pytest.fixture
    def cert_manager(self):
        """Create CertificateManager."""
        return CertificateManager()

    def test_generate_self_signed_ca(self, cert_manager):
        """Test self-signed CA generation."""
        from ai_engine.domains.banking.security.pki.certificate_types import SubjectInfo

        subject = SubjectInfo(
            common_name="Test Root CA",
            organization="Test Organization",
            country="US",
        )

        cert = cert_manager.generate_self_signed_ca(
            subject=subject,
            validity_years=10,
        )

        assert cert.certificate_id is not None
        assert cert.is_ca is True
        assert cert.is_self_signed is True
        assert cert.cert_type == CertificateType.ROOT_CA

    def test_generate_server_certificate(self, cert_manager):
        """Test server certificate generation."""
        from ai_engine.domains.banking.security.pki.certificate_types import SubjectInfo

        # First create CA
        ca_subject = SubjectInfo(
            common_name="Test CA",
            organization="Test Org",
            country="US",
        )
        ca_cert = cert_manager.generate_self_signed_ca(ca_subject)

        # Generate server cert
        server_subject = SubjectInfo(
            common_name="server.example.com",
            organization="Test Org",
            country="US",
        )
        server_cert = cert_manager.generate_certificate(
            subject=server_subject,
            issuer_cert_id=ca_cert.certificate_id,
            cert_type=CertificateType.SERVER,
            san_dns_names=["server.example.com", "www.example.com"],
        )

        assert server_cert.certificate_id is not None
        assert server_cert.is_ca is False
        assert server_cert.cert_type == CertificateType.SERVER
        assert "server.example.com" in server_cert.san_dns_names

    def test_certificate_expiry(self, cert_manager):
        """Test certificate expiry checking."""
        from ai_engine.domains.banking.security.pki.certificate_types import SubjectInfo

        subject = SubjectInfo(common_name="Test Cert", organization="Test")
        ca_cert = cert_manager.generate_self_signed_ca(subject, validity_years=1)

        assert ca_cert.days_until_expiry is not None
        assert ca_cert.days_until_expiry > 360


class TestCSRBuilder:
    """Tests for CSR building."""

    def test_build_server_csr(self):
        """Test building server CSR."""
        builder = CSRBuilder()

        csr = (
            builder
            .set_subject(
                common_name="server.example.com",
                organization="Example Corp",
                country="US",
            )
            .for_server_certificate()
            .add_san_dns("server.example.com", "www.example.com")
            .build()
        )

        assert csr.subject.common_name == "server.example.com"
        assert len(csr.san_dns_names) == 2

    def test_build_psd2_csr(self):
        """Test building PSD2 CSR."""
        from ai_engine.domains.banking.security.pki.certificate_types import SubjectInfo

        subject = SubjectInfo(
            common_name="api.bank.com",
            organization="Test Bank",
            country="GB",
            organization_id="PSDGB-FCA-123456",
            nca_name="FCA",
        )

        builder = CSRBuilder()
        csr = (
            builder
            .set_subject_info(subject)
            .for_psd2_qwac()
            .set_psd2_roles(
                payment_initiation=True,
                account_information=True,
            )
            .add_san_dns("api.bank.com")
            .build()
        )

        assert "PSP_PI" in csr.psd2_roles
        assert "PSP_AI" in csr.psd2_roles


class TestCertificateValidator:
    """Tests for certificate validation."""

    def test_validate_valid_certificate(self):
        """Test validation of valid certificate."""
        from ai_engine.domains.banking.security.pki.certificate_types import SubjectInfo, KeyUsage

        cert = CertificateInfo(
            certificate_id="test-cert-001",
            serial_number="1234567890",
            subject=SubjectInfo(common_name="test.example.com", organization="Test"),
            issuer=SubjectInfo(common_name="Test CA", organization="Test"),
            cert_type=CertificateType.SERVER,
            not_before=datetime.utcnow() - timedelta(days=1),
            not_after=datetime.utcnow() + timedelta(days=365),
            key_usage=[KeyUsage.DIGITAL_SIGNATURE, KeyUsage.KEY_ENCIPHERMENT],
            status=CertificateStatus.VALID,
        )

        validator = CertificateValidator()
        result = validator.validate(cert)

        assert result.is_valid

    def test_validate_expired_certificate(self):
        """Test validation catches expired certificate."""
        from ai_engine.domains.banking.security.pki.certificate_types import SubjectInfo

        cert = CertificateInfo(
            certificate_id="expired-cert",
            serial_number="999",
            subject=SubjectInfo(common_name="expired.example.com", organization="Test"),
            issuer=SubjectInfo(common_name="Test CA", organization="Test"),
            cert_type=CertificateType.SERVER,
            not_before=datetime.utcnow() - timedelta(days=365),
            not_after=datetime.utcnow() - timedelta(days=1),  # Expired
        )

        validator = CertificateValidator()
        result = validator.validate(cert)

        assert not result.is_valid
        assert any("expir" in e.message.lower() for e in result.errors)


class TestPINBlock:
    """Tests for PIN block handling."""

    def test_create_iso_format_0(self):
        """Test ISO Format 0 PIN block creation."""
        pin_block = PINBlock.create_iso_format_0(
            pin="1234",
            pan="4761111111111111",
        )

        assert pin_block.format == PINBlockFormat.ISO_0
        assert len(pin_block.block_data) == 8
        assert pin_block.is_encrypted is False

    def test_create_iso_format_1(self):
        """Test ISO Format 1 PIN block creation."""
        pin_block = PINBlock.create_iso_format_1(
            pin="5678",
            transaction_number=12345,
        )

        assert pin_block.format == PINBlockFormat.ISO_1
        assert len(pin_block.block_data) == 8

    def test_create_iso_format_2(self):
        """Test ISO Format 2 PIN block creation."""
        pin_block = PINBlock.create_iso_format_2(pin="9012")

        assert pin_block.format == PINBlockFormat.ISO_2
        assert len(pin_block.block_data) == 8

    def test_extract_pin_format_0(self):
        """Test PIN extraction from Format 0."""
        original_pin = "4567"
        pan = "4761111111111111"

        pin_block = PINBlock.create_iso_format_0(original_pin, pan)
        extracted = pin_block.extract_pin(pan)

        assert extracted == original_pin

    def test_extract_pin_format_2(self):
        """Test PIN extraction from Format 2."""
        original_pin = "7890"

        pin_block = PINBlock.create_iso_format_2(original_pin)
        extracted = pin_block.extract_pin()

        assert extracted == original_pin

    def test_create_iso_format_4(self):
        """Test ISO Format 4 (AES) PIN block creation."""
        pin_block = PINBlock.create_iso_format_4(
            pin="1234",
            pan="4761111111111111",
        )

        assert pin_block.format == PINBlockFormat.ISO_4
        assert len(pin_block.block_data) == 16  # AES block size


class TestPINValidator:
    """Tests for PIN validation."""

    def test_validate_valid_pin(self):
        """Test validation of valid PIN."""
        validator = PINValidator()

        result = validator.validate("3847")

        assert result.is_valid

    def test_validate_too_short(self):
        """Test validation catches too short PIN."""
        validator = PINValidator(min_length=4)

        result = validator.validate("123")

        assert not result.is_valid
        assert any("short" in e.message.lower() for e in result.errors)

    def test_validate_all_same_digits(self):
        """Test validation catches all same digits."""
        validator = PINValidator()

        result = validator.validate("1111")

        assert not result.is_valid
        assert any("same" in e.message.lower() for e in result.errors)

    def test_validate_ascending_sequence(self):
        """Test validation catches ascending sequence."""
        validator = PINValidator()

        result = validator.validate("1234")

        assert not result.is_valid
        assert any("ascend" in e.message.lower() or "blacklist" in e.message.lower()
                   for e in result.errors)

    def test_validate_pan_match(self):
        """Test validation catches PIN matching PAN digits."""
        validator = PINValidator()
        pan = "4761111111111111"

        result = validator.validate("7611", pan=pan)

        assert not result.is_valid

    def test_validate_blacklisted(self):
        """Test validation catches blacklisted PIN."""
        validator = PINValidator()

        result = validator.validate("0000")

        assert not result.is_valid


class TestPINVerification:
    """Tests for PIN verification values."""

    def test_pvv_generation_software(self):
        """Test PVV generation (software mode)."""
        pvv = PVV()

        result = pvv._generate_visa_pvv_software(
            pin="1234",
            pan="4761111111111111",
            pvki=0,
        )

        assert len(result) == 4
        assert result.isdigit()

    def test_pvv_deterministic(self):
        """Test PVV is deterministic."""
        pvv = PVV()

        result1 = pvv._generate_visa_pvv_software("1234", "4761111111111111", 0)
        result2 = pvv._generate_visa_pvv_software("1234", "4761111111111111", 0)

        assert result1 == result2

    def test_cvv_generation_software(self):
        """Test CVV generation (software mode)."""
        cvv = CVV()

        result = cvv._generate_cvv_software(
            pan="4761111111111111",
            expiry="2312",
            service_code="101",
        )

        assert len(result) == 3
        assert result.isdigit()

    def test_cvv2_generation(self):
        """Test CVV2 generation."""
        cvv = CVV()

        # CVV2 uses service code "000"
        result = cvv._generate_cvv_software(
            pan="4761111111111111",
            expiry="1225",  # MMYY for CVV2
            service_code="000",
        )

        assert len(result) == 3
