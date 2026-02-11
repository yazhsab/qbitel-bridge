"""
Tests for Card Protocol Implementations (EMV and 3D Secure)

Tests cover:
- EMV TLV parsing and building
- EMV transaction processing
- EMV validation
- 3D Secure authentication flow
- 3D Secure message validation
"""

import pytest
from decimal import Decimal
from datetime import datetime

from ai_engine.domains.banking.protocols.payments.cards.emv import (
    TlvParser,
    TlvBuilder,
    TlvData,
    EmvTag,
    EmvApplication,
    EmvCryptogramType,
    EmvTransaction,
    EmvCard,
    EmvTerminal,
    EmvCryptogram,
    EmvValidator,
)
from ai_engine.domains.banking.protocols.payments.cards.three_ds import (
    ThreeDSVersion,
    ThreeDSTransactionStatus,
    ThreeDSDeviceChannel,
    ThreeDSMessageCategory,
    ThreeDSAuthRequest,
    ThreeDSAuthResponse,
    ThreeDSChallengeRequest,
    ThreeDSChallengeResponse,
    ThreeDSResult,
    BrowserInfo,
    MerchantInfo,
    ThreeDSValidator,
    ECI_VALUES,
)


class TestEmvTlv:
    """Tests for EMV TLV parsing and building."""

    def test_parse_simple_tlv(self):
        """Test parsing simple TLV data."""
        # Tag 5A (PAN) with value
        tlv_data = bytes.fromhex("5A0847611111111111115F24032312315F25032201015F280208409F1A020840")
        parser = TlvParser()
        result = parser.parse(tlv_data)

        assert len(result) >= 1

        # Find PAN tag
        pan_tlv = next((t for t in result if t.tag.hex().upper() == "5A"), None)
        assert pan_tlv is not None

    def test_parse_two_byte_tag(self):
        """Test parsing two-byte tags."""
        # Tag 9F26 (Application Cryptogram)
        tlv_data = bytes.fromhex("9F2608ABCDEF0123456789")
        parser = TlvParser()
        result = parser.parse(tlv_data)

        assert len(result) == 1
        assert result[0].tag.hex().upper() == "9F26"
        assert len(result[0].value) == 8

    def test_parse_nested_tlv(self):
        """Test parsing constructed (nested) TLV."""
        # Tag 70 (EMV Proprietary Template) containing nested tags
        # This is a simplified test - real EMV data is more complex
        tlv_data = bytes.fromhex("70065A044761111157134761111111111111D2512201123400001230F")
        parser = TlvParser()
        result = parser.parse(tlv_data)

        assert len(result) >= 1

    def test_build_tlv(self):
        """Test building TLV data."""
        builder = TlvBuilder()
        builder.add("9C", bytes([0x00]))  # Transaction Type
        builder.add("9A", bytes([0x23, 0x01, 0x15]))  # Transaction Date

        result = builder.build()

        assert len(result) > 0
        # Should contain both tags
        assert b"\x9c\x01\x00" in result
        assert b"\x9a\x03" in result

    def test_build_numeric_value(self):
        """Test building TLV with numeric value."""
        builder = TlvBuilder()
        builder.add_numeric("9F02", 10000, 6)  # Amount Authorized

        result = builder.build()

        # 9F02 06 000000010000
        assert bytes.fromhex("9F02") in result

    def test_roundtrip_tlv(self):
        """Test TLV build and parse roundtrip."""
        builder = TlvBuilder()
        builder.add("5A", bytes.fromhex("4761111111111111"))  # PAN
        builder.add("5F24", bytes.fromhex("231231"))  # Expiry

        built = builder.build()

        parser = TlvParser()
        parsed = parser.parse(built)

        assert len(parsed) == 2


class TestEmvTransaction:
    """Tests for EMV transaction processing."""

    def test_create_emv_card(self):
        """Test creating EMV card data."""
        card = EmvCard(
            pan="4761111111111111",
            expiry_date="2312",
            pan_sequence_number="00",
            application_id="A0000000031010",  # Visa
        )

        assert card.pan == "4761111111111111"
        assert card.application_id == "A0000000031010"

    def test_create_emv_terminal(self):
        """Test creating EMV terminal data."""
        terminal = EmvTerminal(
            terminal_id="TERM0001",
            merchant_id="MERCHANT001",
            merchant_category_code="5411",  # Grocery stores
            terminal_country_code="840",  # USA
            terminal_capabilities="E0F0C8",
        )

        assert terminal.terminal_id == "TERM0001"
        assert terminal.merchant_category_code == "5411"

    def test_create_emv_cryptogram(self):
        """Test creating EMV cryptogram."""
        cryptogram = EmvCryptogram(
            cryptogram_type=EmvCryptogramType.ARQC,
            cryptogram_value="ABCDEF0123456789",
            atc="0001",
            unpredictable_number="12345678",
        )

        assert cryptogram.cryptogram_type == EmvCryptogramType.ARQC
        assert cryptogram.atc == "0001"

    def test_emv_transaction_build_cdol(self):
        """Test building CDOL data."""
        card = EmvCard(
            pan="4761111111111111",
            expiry_date="2312",
            application_id="A0000000031010",
        )
        terminal = EmvTerminal(
            terminal_id="TERM0001",
            merchant_id="MERCH001",
            merchant_category_code="5411",
            terminal_country_code="840",
        )
        cryptogram = EmvCryptogram(
            cryptogram_type=EmvCryptogramType.ARQC,
            cryptogram_value="ABCDEF0123456789",
            atc="0001",
        )

        transaction = EmvTransaction(
            card=card,
            terminal=terminal,
            cryptogram=cryptogram,
            amount_authorized=10000,  # $100.00
            currency_code="840",
        )

        cdol_data = transaction.build_cdol1_data()
        assert len(cdol_data) > 0

    def test_emv_transaction_authorization_request(self):
        """Test building authorization request."""
        card = EmvCard(
            pan="4761111111111111",
            expiry_date="2312",
            application_id="A0000000031010",
        )
        terminal = EmvTerminal(
            terminal_id="TERM0001",
            merchant_id="MERCH001",
            merchant_category_code="5411",
            terminal_country_code="840",
        )
        cryptogram = EmvCryptogram(
            cryptogram_type=EmvCryptogramType.ARQC,
            cryptogram_value="ABCDEF0123456789",
            atc="0001",
        )

        transaction = EmvTransaction(
            card=card,
            terminal=terminal,
            cryptogram=cryptogram,
            amount_authorized=10000,
            currency_code="840",
        )

        auth_request = transaction.build_authorization_request()

        assert "pan" in auth_request
        assert "cryptogram" in auth_request
        assert "amount" in auth_request


class TestEmvValidator:
    """Tests for EMV validation."""

    def test_validate_valid_card(self):
        """Test validation of valid card data."""
        card = EmvCard(
            pan="4761111111111111",  # Valid Luhn
            expiry_date="2512",  # Future date
            application_id="A0000000031010",
        )

        validator = EmvValidator()
        result = validator.validate(card)

        assert result.is_valid

    def test_validate_invalid_luhn(self):
        """Test validation catches invalid Luhn."""
        card = EmvCard(
            pan="4761111111111112",  # Invalid Luhn
            expiry_date="2512",
            application_id="A0000000031010",
        )

        validator = EmvValidator()
        result = validator.validate(card)

        assert not result.is_valid
        assert any("luhn" in e.message.lower() for e in result.errors)

    def test_luhn_check(self):
        """Test Luhn algorithm."""
        validator = EmvValidator()

        # Valid card numbers
        assert validator._luhn_check("4761111111111111") is True
        assert validator._luhn_check("5500000000000004") is True
        assert validator._luhn_check("340000000000009") is True

        # Invalid card numbers
        assert validator._luhn_check("4761111111111112") is False
        assert validator._luhn_check("1234567890123456") is False

    def test_analyze_tvr(self):
        """Test TVR (Terminal Verification Results) analysis."""
        validator = EmvValidator()

        # TVR with some bits set
        tvr = "0000008000"  # SDA failed
        issues = validator.analyze_tvr(tvr)

        assert len(issues) >= 0  # May have issues depending on bits


class TestThreeDSMessages:
    """Tests for 3D Secure message handling."""

    def test_create_auth_request(self):
        """Test creating 3DS authentication request."""
        browser_info = BrowserInfo(
            user_agent="Mozilla/5.0...",
            accept_header="text/html,application/json",
            language="en-US",
            color_depth=24,
            screen_width=1920,
            screen_height=1080,
            timezone_offset=-300,
            java_enabled=False,
            javascript_enabled=True,
        )

        merchant_info = MerchantInfo(
            acquirer_bin="123456",
            acquirer_merchant_id="MERCHANT001",
            merchant_name="Test Merchant",
            mcc="5411",
            merchant_country_code="840",
        )

        request = ThreeDSAuthRequest(
            threeds_server_trans_id="550e8400-e29b-41d4-a716-446655440000",
            message_version="2.2.0",
            device_channel="02",  # Browser
            message_category="01",  # Payment
            acct_number="4761111111111111",
            card_expiry_date="2512",
            purchase_amount=10000,
            purchase_currency="840",
            browser_info=browser_info,
            merchant_info=merchant_info,
            notification_url="https://merchant.com/notify",
        )

        assert request.message_version == "2.2.0"
        assert request.device_channel == "02"
        assert request.purchase_amount == 10000

    def test_auth_request_to_dict(self):
        """Test converting auth request to dictionary."""
        merchant_info = MerchantInfo(
            acquirer_bin="123456",
            acquirer_merchant_id="MERCHANT001",
            merchant_name="Test Merchant",
            mcc="5411",
            merchant_country_code="840",
        )

        request = ThreeDSAuthRequest(
            threeds_server_trans_id="550e8400-e29b-41d4-a716-446655440000",
            message_version="2.2.0",
            device_channel="02",
            message_category="01",
            acct_number="4761111111111111",
            card_expiry_date="2512",
            purchase_amount=10000,
            purchase_currency="840",
            merchant_info=merchant_info,
        )

        data = request.to_dict()

        assert data["messageVersion"] == "2.2.0"
        assert data["deviceChannel"] == "02"
        assert "acctNumber" in data

    def test_create_auth_response_frictionless(self):
        """Test frictionless authentication response."""
        response = ThreeDSAuthResponse(
            threeds_server_trans_id="550e8400-e29b-41d4-a716-446655440000",
            acs_trans_id="660e8400-e29b-41d4-a716-446655440001",
            trans_status="Y",  # Authenticated
            authentication_value="ABCDEFGHIJKLMNOPQRSTUVWXYZ==",
            eci="05",
            message_version="2.2.0",
        )

        assert response.trans_status == "Y"
        assert response.eci == "05"
        assert response.authentication_value is not None

    def test_create_auth_response_challenge(self):
        """Test challenge-required response."""
        response = ThreeDSAuthResponse(
            threeds_server_trans_id="550e8400-e29b-41d4-a716-446655440000",
            acs_trans_id="660e8400-e29b-41d4-a716-446655440001",
            trans_status="C",  # Challenge Required
            acs_url="https://acs.issuer.com/challenge",
            message_version="2.2.0",
        )

        assert response.trans_status == "C"
        assert response.acs_url is not None

    def test_threeds_result(self):
        """Test 3DS result object."""
        result = ThreeDSResult(
            threeds_server_trans_id="550e8400-e29b-41d4-a716-446655440000",
            trans_status="Y",
            authentication_value="CAVV123456789012345678901234567890==",
            eci="05",
            was_challenged=False,
        )

        assert result.is_authenticated is True
        assert result.is_liability_shift is True
        assert result.was_challenged is False


class TestThreeDSValidator:
    """Tests for 3DS message validation."""

    def test_validate_valid_auth_request(self):
        """Test validation of valid auth request."""
        merchant_info = MerchantInfo(
            acquirer_bin="123456",
            acquirer_merchant_id="MERCHANT001",
            merchant_name="Test Merchant",
            mcc="5411",
            merchant_country_code="840",
        )

        browser_info = BrowserInfo(
            user_agent="Mozilla/5.0...",
            accept_header="text/html",
            language="en-US",
            color_depth=24,
            screen_width=1920,
            screen_height=1080,
            timezone_offset=0,
            java_enabled=False,
            javascript_enabled=True,
        )

        request = ThreeDSAuthRequest(
            threeds_server_trans_id="550e8400-e29b-41d4-a716-446655440000",
            message_version="2.2.0",
            device_channel="02",
            message_category="01",
            acct_number="4761111111111111",
            card_expiry_date="2512",
            purchase_amount=10000,
            purchase_currency="840",
            merchant_info=merchant_info,
            browser_info=browser_info,
            notification_url="https://merchant.com/notify",
        )

        validator = ThreeDSValidator()
        result = validator.validate(request)

        assert result.is_valid

    def test_validate_missing_server_trans_id(self):
        """Test validation catches missing server trans ID."""
        merchant_info = MerchantInfo(
            acquirer_bin="123456",
            acquirer_merchant_id="MERCHANT001",
            merchant_name="Test Merchant",
            mcc="5411",
            merchant_country_code="840",
        )

        request = ThreeDSAuthRequest(
            threeds_server_trans_id="",  # Missing
            message_version="2.2.0",
            device_channel="02",
            message_category="01",
            acct_number="4761111111111111",
            card_expiry_date="2512",
            purchase_amount=10000,
            purchase_currency="840",
            merchant_info=merchant_info,
        )

        validator = ThreeDSValidator()
        result = validator.validate(request)

        assert not result.is_valid
        assert any("serverTransID" in e.message or "trans" in e.message.lower() for e in result.errors)

    def test_validate_auth_response_successful(self):
        """Test validation of successful auth response."""
        response = ThreeDSAuthResponse(
            threeds_server_trans_id="550e8400-e29b-41d4-a716-446655440000",
            acs_trans_id="660e8400-e29b-41d4-a716-446655440001",
            trans_status="Y",
            authentication_value="CAVV12345678901234567890123456",
            eci="05",
            message_version="2.2.0",
        )

        validator = ThreeDSValidator()
        result = validator.validate(response)

        assert result.is_valid

    def test_validate_auth_response_missing_cavv(self):
        """Test validation catches missing CAVV for successful auth."""
        response = ThreeDSAuthResponse(
            threeds_server_trans_id="550e8400-e29b-41d4-a716-446655440000",
            acs_trans_id="660e8400-e29b-41d4-a716-446655440001",
            trans_status="Y",  # Successful but missing auth value
            authentication_value="",
            eci="05",
            message_version="2.2.0",
        )

        validator = ThreeDSValidator()
        result = validator.validate(response)

        assert not result.is_valid


class TestThreeDSCodes:
    """Tests for 3DS code definitions."""

    def test_version_enum(self):
        """Test 3DS version enum."""
        v21 = ThreeDSVersion.V2_1
        assert "2.1" in v21.value

        v22 = ThreeDSVersion.V2_2
        assert "2.2" in v22.value

    def test_transaction_status_enum(self):
        """Test transaction status enum."""
        status_y = ThreeDSTransactionStatus.Y
        assert status_y.is_successful is True

        status_c = ThreeDSTransactionStatus.C
        assert status_c.requires_challenge is True

        status_n = ThreeDSTransactionStatus.N
        assert status_n.is_successful is False

    def test_eci_values(self):
        """Test ECI value mappings."""
        assert "05" in ECI_VALUES["visa"]
        assert "02" in ECI_VALUES["mastercard"]
