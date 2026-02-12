"""
Tests for SWIFT MT Message Support

Tests cover:
- MT103 Single Customer Credit Transfer
- MT202 General Financial Institution Transfer
- SWIFT message parsing
- SWIFT message validation
"""

import pytest
from decimal import Decimal
from datetime import date

from ai_engine.domains.banking.protocols.messaging.swift import (
    SwiftParser,
    SwiftParseError,
    SwiftValidator,
    SwiftMessageType,
    SwiftFieldTag,
    MT103Message,
    MT103Party,
    MT103Charges,
    MT103Builder,
    MT202Message,
    MT202Agent,
    MT202Builder,
)


class TestSwiftParser:
    """Tests for SWIFT message parsing."""

    def test_parse_mt103_basic(self):
        """Test parsing a basic MT103 message."""
        raw_message = """{1:F01BANKUSNYAXXX0000000000}
{2:O1031234200115BANKGB2LAXXX00000000002001151234N}
{4:
:20:REFERENCE123456
:23B:CRED
:32A:230115USD1000,00
:50K:/12345678
ORDERING CUSTOMER
123 MAIN ST
NEW YORK NY
:59:/87654321
BENEFICIARY
456 HIGH ST
LONDON
:71A:SHA
-}"""
        parser = SwiftParser(strict=False)
        message = parser.parse(raw_message)

        assert message.basic_header.content is not None
        assert message.application_header.content is not None
        assert len(message.text_block.fields) >= 5

    def test_parse_field_extraction(self):
        """Test field extraction from text block."""
        raw_message = """{1:F01TESTUSNYAXXX0000000000}
{2:O103}
{4:
:20:REF123
:23B:CRED
:32A:230115EUR500,00
:71A:SHA
-}"""
        parser = SwiftParser(strict=False)
        message = parser.parse(raw_message)

        # Check field 20 (reference)
        field_20 = next((f for f in message.text_block.fields if f.tag == "20"), None)
        assert field_20 is not None
        assert field_20.value == "REF123"

        # Check field 32A (value date/amount)
        field_32a = next((f for f in message.text_block.fields if f.tag == "32A"), None)
        assert field_32a is not None
        assert "EUR" in field_32a.value

    def test_parse_blocks(self):
        """Test block extraction."""
        raw_message = """{1:F01BANKUSNYAXXX0000000000}
{2:O1030000000000}
{3:{108:MSGREF123}}
{4:
:20:REF
-}
{5:{CHK:123ABC}}"""
        parser = SwiftParser(strict=False)
        message = parser.parse(raw_message)

        assert message.basic_header.content is not None
        assert message.application_header.content is not None
        assert message.user_header is not None
        assert message.trailer is not None

    def test_parse_error_missing_blocks(self):
        """Test error handling for missing required blocks."""
        raw_message = "{4:\n:20:REF\n-}"
        parser = SwiftParser(strict=False)
        message = parser.parse(raw_message)

        # Should have parse errors
        assert len(message.parse_errors) > 0

    def test_parse_strict_mode(self):
        """Test strict mode raises exceptions."""
        raw_message = "{4:\n:20:REF\n-}"  # Missing required blocks
        parser = SwiftParser(strict=True)

        with pytest.raises(SwiftParseError):
            parser.parse(raw_message)


class TestMT103:
    """Tests for MT103 Single Customer Credit Transfer."""

    def test_mt103_create_payment(self):
        """Test creating an MT103 payment message."""
        message = MT103Builder.create_payment(
            sender_bic="BANKUSNY",
            receiver_bic="BANKGB2L",
            reference="PAY20230115001",
            value_date=date(2023, 1, 15),
            currency="USD",
            amount=Decimal("10000.00"),
            ordering_customer_name="ACME CORPORATION",
            ordering_customer_account="123456789",
            ordering_customer_address="123 MAIN ST, NEW YORK, NY",
            beneficiary_name="WIDGETS LTD",
            beneficiary_account="987654321",
            beneficiary_address="456 HIGH ST, LONDON",
        )

        assert message.sender_reference == "PAY20230115001"
        assert message.currency == "USD"
        assert message.amount == Decimal("10000.00")
        assert message.ordering_customer.name == "ACME CORPORATION"
        assert message.beneficiary.name == "WIDGETS LTD"

    def test_mt103_party_creation(self):
        """Test MT103 party data structure."""
        party = MT103Party(
            name="JOHN DOE",
            account="12345678",
            address=["123 MAIN ST", "NEW YORK NY 10001"],
        )

        assert party.name == "JOHN DOE"
        assert party.account == "12345678"
        assert len(party.address) == 2

    def test_mt103_charges(self):
        """Test MT103 charges handling."""
        charges = MT103Charges(
            details_of_charges="SHA",
            sender_charges_currency="USD",
            sender_charges_amount=Decimal("25.00"),
        )

        assert charges.details_of_charges == "SHA"
        assert charges.sender_charges_amount == Decimal("25.00")

    def test_mt103_to_swift_format(self):
        """Test MT103 conversion to SWIFT format."""
        message = MT103Builder.create_payment(
            sender_bic="BANKUSNY",
            receiver_bic="BANKGB2L",
            reference="TEST001",
            value_date=date(2023, 1, 15),
            currency="EUR",
            amount=Decimal("5000.00"),
            ordering_customer_name="TEST CUSTOMER",
            ordering_customer_account="11111111",
            beneficiary_name="BENEFICIARY",
            beneficiary_account="22222222",
        )

        swift_text = message.to_swift_text()

        assert ":20:TEST001" in swift_text
        assert ":32A:" in swift_text
        assert "EUR" in swift_text


class TestMT202:
    """Tests for MT202 General Financial Institution Transfer."""

    def test_mt202_create(self):
        """Test creating an MT202 message."""
        message = MT202Builder.create_transfer(
            sender_bic="BANKUSNY",
            receiver_bic="BANKGB2L",
            transaction_reference="TRN20230115001",
            related_reference="REL001",
            value_date=date(2023, 1, 15),
            currency="USD",
            amount=Decimal("1000000.00"),
            beneficiary_institution_bic="BENEFBIC",
        )

        assert message.transaction_reference == "TRN20230115001"
        assert message.related_reference == "REL001"
        assert message.currency == "USD"
        assert message.amount == Decimal("1000000.00")

    def test_mt202_cov(self):
        """Test MT202 COV (cover payment)."""
        message = MT202Builder.create_cov_transfer(
            sender_bic="BANKUSNY",
            receiver_bic="BANKGB2L",
            transaction_reference="COV20230115001",
            related_reference="REL002",
            value_date=date(2023, 1, 15),
            currency="USD",
            amount=Decimal("50000.00"),
            beneficiary_institution_bic="BENEFBIC",
            underlying_customer_reference="CUSTREF001",
            ordering_customer_name="ORDERING CORP",
            beneficiary_customer_name="BENEFICIARY CORP",
        )

        assert message.is_cov is True
        assert message.cover_info is not None
        assert message.cover_info.underlying_customer_reference == "CUSTREF001"


class TestSwiftValidator:
    """Tests for SWIFT message validation."""

    def test_validate_valid_mt103(self):
        """Test validation of valid MT103."""
        message = MT103Builder.create_payment(
            sender_bic="BANKUSNY",
            receiver_bic="BANKGB2L",
            reference="VALID001",
            value_date=date(2023, 1, 15),
            currency="USD",
            amount=Decimal("1000.00"),
            ordering_customer_name="CUSTOMER",
            ordering_customer_account="12345678",
            beneficiary_name="BENEFICIARY",
            beneficiary_account="87654321",
        )

        validator = SwiftValidator()
        result = validator.validate_mt103(message)

        assert result.is_valid

    def test_validate_missing_reference(self):
        """Test validation catches missing reference."""
        message = MT103Message(
            sender_reference="",  # Missing
            bank_operation_code="CRED",
            value_date=date(2023, 1, 15),
            currency="USD",
            amount=Decimal("1000.00"),
            ordering_customer=MT103Party(name="CUSTOMER", account="123"),
            beneficiary=MT103Party(name="BENEFICIARY", account="456"),
            charges=MT103Charges(details_of_charges="SHA"),
        )

        validator = SwiftValidator()
        result = validator.validate_mt103(message)

        assert not result.is_valid
        assert any("reference" in e.message.lower() for e in result.errors)

    def test_validate_invalid_bic(self):
        """Test validation of BIC format."""
        validator = SwiftValidator()

        # Valid BICs
        assert validator._validate_bic("BANKUSNY") is True
        assert validator._validate_bic("BANKUSNYXXX") is True

        # Invalid BICs
        assert validator._validate_bic("BANK") is False  # Too short
        assert validator._validate_bic("123456789012") is False  # Numeric


class TestSwiftCodes:
    """Tests for SWIFT code definitions."""

    def test_message_type_lookup(self):
        """Test message type lookup."""
        mt103 = SwiftMessageType.MT103
        assert mt103.code == "103"
        assert "credit" in mt103.description.lower()

        mt202 = SwiftMessageType.MT202
        assert mt202.code == "202"

    def test_field_tag_properties(self):
        """Test field tag properties."""
        f20 = SwiftFieldTag.F20
        assert f20.tag == "20"
        assert f20.mandatory is True

        f32a = SwiftFieldTag.F32A
        assert f32a.tag == "32A"
