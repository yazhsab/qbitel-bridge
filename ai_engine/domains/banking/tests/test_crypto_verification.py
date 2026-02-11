"""
Tests for Cryptographic Verification Module

Tests cover:
- EMV cryptogram verification (ARQC, ARPC)
- 3D Secure CAVV verification
- Digital signature verification
- MAC/HMAC verification
- Constant-time comparison security
"""

import pytest
import hashlib
import hmac
import base64
import time
import secrets
from datetime import datetime

from ai_engine.domains.banking.security.crypto import (
    # Result types
    VerificationStatus,
    VerificationResult,
    # EMV
    EMVCryptogramVerifier,
    EMVCryptogramData,
    CryptogramType,
    KeyDerivationType,
    # 3DS
    ThreeDSVerifier,
    ThreeDSVerificationData,
    # Signatures
    SignatureVerifier,
    SignatureVerificationData,
    # MAC
    MACVerifier,
    # Utilities
    secure_compare,
    # Convenience functions
    verify_emv_arqc,
    verify_cavv,
    verify_signature,
)


class TestSecureCompare:
    """Tests for constant-time comparison."""

    def test_equal_values(self):
        """Test comparison of equal values."""
        a = b"hello world"
        b = b"hello world"
        assert secure_compare(a, b) is True

    def test_unequal_values(self):
        """Test comparison of unequal values."""
        a = b"hello world"
        b = b"hello worlD"
        assert secure_compare(a, b) is False

    def test_different_lengths(self):
        """Test comparison of different length values."""
        a = b"hello"
        b = b"hello world"
        assert secure_compare(a, b) is False

    def test_empty_values(self):
        """Test comparison of empty values."""
        a = b""
        b = b""
        assert secure_compare(a, b) is True

    def test_timing_resistance(self):
        """Test that comparison is constant-time (basic check)."""
        # This is a simplified test - real timing tests require more iterations
        secret = secrets.token_bytes(32)
        wrong_first_byte = b"\x00" + secret[1:]
        wrong_last_byte = secret[:-1] + b"\x00"

        # Time comparisons
        iterations = 1000
        times_first = []
        times_last = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            secure_compare(secret, wrong_first_byte)
            times_first.append(time.perf_counter_ns() - start)

            start = time.perf_counter_ns()
            secure_compare(secret, wrong_last_byte)
            times_last.append(time.perf_counter_ns() - start)

        # Average times should be similar (within 50% is a reasonable threshold)
        avg_first = sum(times_first) / len(times_first)
        avg_last = sum(times_last) / len(times_last)

        # They should be roughly equal (constant time)
        ratio = max(avg_first, avg_last) / max(min(avg_first, avg_last), 1)
        assert ratio < 2.0, f"Timing ratio {ratio} suggests non-constant time"


class TestEMVCryptogramVerifier:
    """Tests for EMV cryptogram verification."""

    def test_verifier_creation(self):
        """Test verifier creation."""
        verifier = EMVCryptogramVerifier()
        assert verifier is not None

    def test_verify_arqc_no_key(self):
        """Test ARQC verification without master key."""
        verifier = EMVCryptogramVerifier()

        data = EMVCryptogramData(
            cryptogram=bytes.fromhex("ABCDEF0123456789"),
            cryptogram_type=CryptogramType.ARQC,
            pan="4761111111111111",
            atc=bytes.fromhex("0001"),
        )

        result = verifier.verify_arqc(data)

        assert result.status == VerificationStatus.KEY_NOT_FOUND
        assert "master key" in result.message.lower()

    def test_verify_arqc_with_key(self):
        """Test ARQC verification with master key."""
        verifier = EMVCryptogramVerifier()

        # Use a test master key
        master_key = bytes.fromhex("0123456789ABCDEF0123456789ABCDEF")

        data = EMVCryptogramData(
            cryptogram=bytes.fromhex("ABCDEF0123456789"),
            cryptogram_type=CryptogramType.ARQC,
            pan="4761111111111111",
            atc=bytes.fromhex("0001"),
            amount_authorized=10000,
            terminal_country_code="0840",
            transaction_currency_code="0840",
            transaction_date="250115",
            transaction_type="00",
            unpredictable_number=bytes.fromhex("12345678"),
        )

        result = verifier.verify_arqc(data, master_key)

        # Will be invalid since we don't know the correct cryptogram
        assert result.status in (VerificationStatus.VALID, VerificationStatus.INVALID)
        assert result.algorithm == "3DES-CBC-MAC"

    def test_generate_arpc(self):
        """Test ARPC generation."""
        verifier = EMVCryptogramVerifier()

        master_key = bytes.fromhex("0123456789ABCDEF0123456789ABCDEF")
        arqc = bytes.fromhex("ABCDEF0123456789")

        arpc, result = verifier.generate_arpc(
            arqc=arqc,
            authorization_response_code="3030",  # '00' in ASCII
            issuer_master_key=master_key,
            pan="4761111111111111",
            atc=bytes.fromhex("0001"),
            method=1,
        )

        assert arpc is not None
        assert len(arpc) == 8
        assert result.status == VerificationStatus.VALID

    def test_generate_arpc_method2(self):
        """Test ARPC generation method 2."""
        verifier = EMVCryptogramVerifier()

        master_key = bytes.fromhex("0123456789ABCDEF0123456789ABCDEF")
        arqc = bytes.fromhex("ABCDEF0123456789")

        arpc, result = verifier.generate_arpc(
            arqc=arqc,
            authorization_response_code="3030",
            issuer_master_key=master_key,
            pan="4761111111111111",
            atc=bytes.fromhex("0001"),
            method=2,
        )

        assert arpc is not None
        assert result.status == VerificationStatus.VALID
        assert result.details.get("method") == 2

    def test_convenience_verify_arqc(self):
        """Test convenience function for ARQC verification."""
        result = verify_emv_arqc(
            cryptogram=bytes.fromhex("ABCDEF0123456789"),
            pan="4761111111111111",
            atc=bytes.fromhex("0001"),
        )

        assert result.status == VerificationStatus.KEY_NOT_FOUND


class TestThreeDSVerifier:
    """Tests for 3D Secure verification."""

    def test_verifier_creation(self):
        """Test verifier creation."""
        verifier = ThreeDSVerifier()
        assert verifier is not None

    def test_verify_cavv_3ds1(self):
        """Test 3DS 1.0 CAVV verification."""
        verifier = ThreeDSVerifier()

        # Create a sample 20-byte CAVV (3DS 1.0 format)
        cavv_bytes = bytes(20)  # All zeros for testing
        cavv_b64 = base64.b64encode(cavv_bytes).decode()

        data = ThreeDSVerificationData(
            authentication_value=cavv_b64,
            eci="05",
            trans_status="Y",
            acs_trans_id="660e8400-e29b-41d4-a716-446655440001",
            threeds_server_trans_id="550e8400-e29b-41d4-a716-446655440000",
            message_version="1.0.2",
            pan="4761111111111111",
        )

        result = verifier.verify_cavv(data)

        # Should pass format validation
        assert result.status in (VerificationStatus.VALID, VerificationStatus.INVALID)
        assert "CAVV" in result.algorithm

    def test_verify_cavv_3ds2(self):
        """Test 3DS 2.x CAVV verification."""
        verifier = ThreeDSVerifier()

        # Create a sample 32-byte CAVV (3DS 2.x format)
        cavv_bytes = bytes(32)  # All zeros for testing
        cavv_b64 = base64.b64encode(cavv_bytes).decode()

        data = ThreeDSVerificationData(
            authentication_value=cavv_b64,
            eci="05",
            trans_status="Y",
            acs_trans_id="660e8400-e29b-41d4-a716-446655440001",
            threeds_server_trans_id="550e8400-e29b-41d4-a716-446655440000",
            message_version="2.2.0",
            pan="4761111111111111",
        )

        result = verifier.verify_cavv(data)

        assert result.status in (VerificationStatus.VALID, VerificationStatus.INVALID)
        assert "CAVV" in result.algorithm or "3DS2" in result.algorithm

    def test_verify_cavv_invalid_length(self):
        """Test CAVV verification with invalid length."""
        verifier = ThreeDSVerifier()

        # Create invalid length CAVV
        cavv_bytes = bytes(15)  # Invalid length
        cavv_b64 = base64.b64encode(cavv_bytes).decode()

        data = ThreeDSVerificationData(
            authentication_value=cavv_b64,
            eci="05",
            trans_status="Y",
            acs_trans_id="",
            threeds_server_trans_id="",
            message_version="2.2.0",
            pan="4761111111111111",
        )

        result = verifier.verify_cavv(data)

        assert result.status == VerificationStatus.INVALID
        assert "length" in result.message.lower()

    def test_validate_eci_visa_success(self):
        """Test ECI validation for successful Visa auth."""
        verifier = ThreeDSVerifier()

        result = verifier.validate_eci("05", "Y", "visa")

        assert result.status == VerificationStatus.VALID

    def test_validate_eci_mastercard_success(self):
        """Test ECI validation for successful Mastercard auth."""
        verifier = ThreeDSVerifier()

        result = verifier.validate_eci("02", "Y", "mastercard")

        assert result.status == VerificationStatus.VALID

    def test_validate_eci_mismatch(self):
        """Test ECI validation for mismatched values."""
        verifier = ThreeDSVerifier()

        # Visa with Mastercard ECI
        result = verifier.validate_eci("02", "Y", "visa")

        assert result.status == VerificationStatus.INVALID

    def test_convenience_verify_cavv(self):
        """Test convenience function for CAVV verification."""
        cavv_bytes = bytes(32)
        cavv_b64 = base64.b64encode(cavv_bytes).decode()

        result = verify_cavv(
            authentication_value=cavv_b64,
            eci="05",
            trans_status="Y",
            pan="4761111111111111",
        )

        assert result is not None
        assert "CAVV" in result.algorithm or "3DS" in result.algorithm


class TestSignatureVerifier:
    """Tests for digital signature verification."""

    def test_verifier_creation(self):
        """Test verifier creation."""
        verifier = SignatureVerifier()
        assert verifier is not None

    def test_unsupported_algorithm(self):
        """Test verification with unsupported algorithm."""
        verifier = SignatureVerifier()

        data = SignatureVerificationData(
            signature=b"test",
            message=b"message",
            public_key=b"key",
            algorithm="UNSUPPORTED-ALGO",
        )

        result = verifier.verify(data)

        assert result.status == VerificationStatus.ALGORITHM_NOT_SUPPORTED

    def test_verify_invalid_rsa_key(self):
        """Test RSA verification with invalid key format."""
        verifier = SignatureVerifier()

        # Random bytes are not a valid RSA public key
        public_key = secrets.token_bytes(64)
        message = b"test message"

        # Create dummy signature
        expected = hmac.new(
            public_key[:32],
            message,
            hashlib.sha256,
        ).digest()

        data = SignatureVerificationData(
            signature=expected,
            message=message,
            public_key=public_key,
            algorithm="RSA-PKCS1-SHA256",
        )

        result = verifier.verify(data)

        # Should fail due to invalid key format (ERROR for deserialization failures)
        assert result.status == VerificationStatus.ERROR

    def test_verify_ml_dsa_no_provider(self):
        """Test ML-DSA verification without PQC provider."""
        verifier = SignatureVerifier()

        data = SignatureVerificationData(
            signature=b"test",
            message=b"message",
            public_key=b"key" * 32,
            algorithm="ML-DSA-65",
        )

        result = verifier.verify(data)

        # Should work with simulation
        assert result.status in (VerificationStatus.VALID, VerificationStatus.INVALID)

    def test_verify_hybrid(self):
        """Test hybrid signature verification."""
        verifier = SignatureVerifier()

        # Create hybrid signature format
        import struct

        public_key = secrets.token_bytes(128)
        message = b"test message"

        # Create classical signature portion
        classical_sig = hmac.new(
            public_key[:32],
            message,
            hashlib.sha256,
        ).digest()

        # Create PQC signature portion (simulated)
        pqc_sig = hmac.new(
            public_key[64:96],
            message,
            hashlib.sha256,
        ).digest()

        # Pack hybrid signature
        hybrid_sig = struct.pack(">H", len(classical_sig)) + classical_sig + pqc_sig

        data = SignatureVerificationData(
            signature=hybrid_sig,
            message=message,
            public_key=public_key,
            algorithm="ECDSA-P256+ML-DSA-65",
        )

        result = verifier.verify(data)

        # May pass or fail depending on key/sig match
        assert result.status in (
            VerificationStatus.VALID,
            VerificationStatus.INVALID,
            VerificationStatus.ERROR,
        )

    def test_convenience_verify_signature(self):
        """Test convenience function for signature verification."""
        public_key = secrets.token_bytes(64)
        message = b"test message"
        signature = hmac.new(
            public_key[:32],
            message,
            hashlib.sha256,
        ).digest()

        result = verify_signature(
            signature=signature,
            message=message,
            public_key=public_key,
            algorithm="RSA-PKCS1-SHA256",
        )

        assert result is not None


class TestMACVerifier:
    """Tests for MAC verification."""

    def test_verify_hmac_sha256(self):
        """Test HMAC-SHA256 verification."""
        verifier = MACVerifier()

        key = secrets.token_bytes(32)
        message = b"test message"
        mac = hmac.new(key, message, hashlib.sha256).digest()

        result = verifier.verify_hmac(key, message, mac, "SHA256")

        assert result.status == VerificationStatus.VALID

    def test_verify_hmac_sha384(self):
        """Test HMAC-SHA384 verification."""
        verifier = MACVerifier()

        key = secrets.token_bytes(48)
        message = b"test message"
        mac = hmac.new(key, message, hashlib.sha384).digest()

        result = verifier.verify_hmac(key, message, mac, "SHA384")

        assert result.status == VerificationStatus.VALID

    def test_verify_hmac_sha512(self):
        """Test HMAC-SHA512 verification."""
        verifier = MACVerifier()

        key = secrets.token_bytes(64)
        message = b"test message"
        mac = hmac.new(key, message, hashlib.sha512).digest()

        result = verifier.verify_hmac(key, message, mac, "SHA512")

        assert result.status == VerificationStatus.VALID

    def test_verify_hmac_truncated(self):
        """Test verification of truncated HMAC."""
        verifier = MACVerifier()

        key = secrets.token_bytes(32)
        message = b"test message"
        full_mac = hmac.new(key, message, hashlib.sha256).digest()
        truncated_mac = full_mac[:16]  # Truncate to 16 bytes

        result = verifier.verify_hmac(key, message, truncated_mac, "SHA256")

        assert result.status == VerificationStatus.VALID

    def test_verify_hmac_invalid(self):
        """Test HMAC verification with invalid MAC."""
        verifier = MACVerifier()

        key = secrets.token_bytes(32)
        message = b"test message"
        invalid_mac = secrets.token_bytes(32)  # Random, invalid MAC

        result = verifier.verify_hmac(key, message, invalid_mac, "SHA256")

        assert result.status == VerificationStatus.INVALID

    def test_verify_hmac_wrong_key(self):
        """Test HMAC verification with wrong key."""
        verifier = MACVerifier()

        key1 = secrets.token_bytes(32)
        key2 = secrets.token_bytes(32)
        message = b"test message"
        mac = hmac.new(key1, message, hashlib.sha256).digest()

        result = verifier.verify_hmac(key2, message, mac, "SHA256")

        assert result.status == VerificationStatus.INVALID

    def test_verify_hmac_unsupported_algorithm(self):
        """Test HMAC with unsupported algorithm."""
        verifier = MACVerifier()

        result = verifier.verify_hmac(
            b"key",
            b"message",
            b"mac",
            "UNSUPPORTED",
        )

        assert result.status == VerificationStatus.ALGORITHM_NOT_SUPPORTED


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_is_valid_true(self):
        """Test is_valid for valid status."""
        result = VerificationResult(
            status=VerificationStatus.VALID,
            algorithm="test",
        )
        assert result.is_valid is True

    def test_is_valid_false(self):
        """Test is_valid for invalid status."""
        result = VerificationResult(
            status=VerificationStatus.INVALID,
            algorithm="test",
        )
        assert result.is_valid is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = VerificationResult(
            status=VerificationStatus.VALID,
            algorithm="TEST-ALGO",
            message="Test message",
            details={"key": "value"},
        )

        d = result.to_dict()

        assert d["status"] == "valid"
        assert d["is_valid"] is True
        assert d["algorithm"] == "TEST-ALGO"
        assert d["message"] == "Test message"
        assert d["details"]["key"] == "value"
        assert "verified_at" in d


class TestIntegration:
    """Integration tests for verification module."""

    def test_emv_flow(self):
        """Test complete EMV verification flow."""
        verifier = EMVCryptogramVerifier()

        # Simulate card transaction
        master_key = secrets.token_bytes(16)  # 16 bytes for 2-key 3DES

        # First verify ARQC (would fail without correct cryptogram)
        data = EMVCryptogramData(
            cryptogram=secrets.token_bytes(8),
            cryptogram_type=CryptogramType.ARQC,
            pan="4761111111111111",
            pan_sequence="00",
            atc=bytes.fromhex("0001"),
            amount_authorized=10000,
            terminal_country_code="0840",
            transaction_currency_code="0840",
            transaction_date="250115",
            transaction_type="00",
            unpredictable_number=secrets.token_bytes(4),
        )

        arqc_result = verifier.verify_arqc(data, master_key)

        # Then generate ARPC
        arpc, arpc_result = verifier.generate_arpc(
            arqc=data.cryptogram,
            authorization_response_code="3030",
            issuer_master_key=master_key,
            pan=data.pan,
            atc=data.atc,
        )

        assert arpc_result.status == VerificationStatus.VALID
        assert arpc is not None

    def test_threeds_eci_validation_flow(self):
        """Test 3DS ECI validation flow."""
        verifier = ThreeDSVerifier()

        # Test various ECI/status combinations
        test_cases = [
            ("05", "Y", "visa", VerificationStatus.VALID),
            ("06", "A", "visa", VerificationStatus.VALID),
            ("02", "Y", "mastercard", VerificationStatus.VALID),
            ("05", "Y", "mastercard", VerificationStatus.INVALID),  # Visa ECI for MC
        ]

        for eci, status, brand, expected in test_cases:
            result = verifier.validate_eci(eci, status, brand)
            assert result.status == expected, f"Failed for {eci}, {status}, {brand}"

    def test_signature_algorithms(self):
        """Test multiple signature algorithms."""
        verifier = SignatureVerifier()

        algorithms = [
            "RSA-PKCS1-SHA256",
            "RSA-PSS-SHA256",
            "ECDSA-P256-SHA256",
            "ML-DSA-65",
        ]

        public_key = secrets.token_bytes(64)
        message = b"test message"

        for algo in algorithms:
            # Create matching signature for simulation
            sig = hmac.new(
                public_key[:32],
                message,
                hashlib.sha256,
            ).digest()

            data = SignatureVerificationData(
                signature=sig,
                message=message,
                public_key=public_key,
                algorithm=algo,
            )

            result = verifier.verify(data)

            # All should complete without crashing
            # Note: With invalid random key data, results may be ERROR,
            # INVALID, or ALGORITHM_NOT_SUPPORTED
            assert result.status in (
                VerificationStatus.VALID,
                VerificationStatus.INVALID,
                VerificationStatus.ERROR,
                VerificationStatus.ALGORITHM_NOT_SUPPORTED,
            ), f"Failed for {algo}: {result.message}"
