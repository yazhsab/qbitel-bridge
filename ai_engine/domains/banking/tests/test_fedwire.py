"""
Tests for FedWire Protocol Implementation
"""

import pytest
from datetime import date, datetime
from decimal import Decimal

from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_codes import (
    TypeCode,
    TypeSubCode,
    BusinessFunctionCode,
    FedWireTag,
    IDCode,
    DrawdownDebitAccountAdviceCode,
    MANDATORY_TAGS,
    BFC_REQUIREMENTS,
)
from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_message import (
    FedWireMessage,
    SenderInfo,
    ReceiverInfo,
    BeneficiaryInfo,
    OriginatorInfo,
    IntermediaryInfo,
    FIInfo,
    Address,
    IMAD,
    SenderSuppliedInfo,
    OriginatorToBeneficiaryInfo,
    FIToFIInfo,
)
from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_parser import (
    FedWireParser,
    FedWireParseError,
)
from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_builder import (
    FedWireBuilder,
    FedWireBuildError,
    create_bank_transfer,
    create_customer_transfer,
    create_cover_payment,
)


class TestTypeCode:
    """Tests for FedWire Type Codes."""

    def test_basic_transfer(self):
        """Test basic funds transfer type code."""
        tc = TypeCode.BASIC_FUNDS_TRANSFER
        assert tc.code == "10"
        assert "Basic" in tc.description

    def test_reversal_codes(self):
        """Test reversal type codes."""
        assert TypeCode.REQUEST_FOR_REVERSAL.code == "15"
        assert TypeCode.REVERSAL_OF_TRANSFER.code == "16"

    def test_from_code(self):
        """Test code lookup."""
        tc = TypeCode.from_code("10")
        assert tc == TypeCode.BASIC_FUNDS_TRANSFER

        tc = TypeCode.from_code("99")  # Invalid
        assert tc is None


class TestBusinessFunctionCode:
    """Tests for Business Function Codes."""

    def test_bank_transfer(self):
        """Test bank transfer code."""
        bfc = BusinessFunctionCode.BANK_TRANSFER
        assert bfc.code == "BTR"
        assert "Bank" in bfc.description

    def test_customer_transfer(self):
        """Test customer transfer code."""
        bfc = BusinessFunctionCode.CUSTOMER_TRANSFER
        assert bfc.code == "CTR"

    def test_customer_transfer_plus(self):
        """Test customer transfer plus code."""
        bfc = BusinessFunctionCode.CUSTOMER_TRANSFER_PLUS
        assert bfc.code == "CTP"

    def test_cover_payment(self):
        """Test cover payment code."""
        bfc = BusinessFunctionCode.COVER_PAYMENT
        assert bfc.code == "COV"

    def test_from_code(self):
        """Test code lookup."""
        bfc = BusinessFunctionCode.from_code("BTR")
        assert bfc == BusinessFunctionCode.BANK_TRANSFER


class TestIDCode:
    """Tests for ID Codes."""

    def test_aba_routing(self):
        """Test ABA routing ID code."""
        ic = IDCode.ABA_ROUTING
        assert ic.code == "D"

    def test_swift_bic(self):
        """Test SWIFT BIC ID code."""
        ic = IDCode.SWIFT_BIC
        assert ic.code == "B"

    def test_from_code(self):
        """Test code lookup."""
        ic = IDCode.from_code("D")
        assert ic == IDCode.ABA_ROUTING


class TestIMAD:
    """Tests for IMAD (Input Message Accountability Data)."""

    def test_imad_creation(self):
        """Test IMAD creation."""
        imad = IMAD(
            input_date=date(2023, 12, 15),
            source_id="BKRT1234",
            cycle_code="01",
            sequence_number="000001",
        )

        assert imad.source_id == "BKRT1234"
        assert imad.cycle_code == "01"

    def test_imad_to_string(self):
        """Test IMAD string conversion."""
        imad = IMAD(
            input_date=date(2023, 12, 15),
            source_id="BKRT1234",
            cycle_code="01",
            sequence_number="000001",
        )

        imad_str = imad.to_string()
        assert len(imad_str) == 22
        assert imad_str.startswith("20231215")

    def test_imad_from_string(self):
        """Test IMAD parsing from string."""
        imad_str = "20231215BKRT123401000001"
        imad = IMAD.from_string(imad_str)

        assert imad.input_date == date(2023, 12, 15)
        assert imad.source_id == "BKRT1234"
        assert imad.cycle_code == "01"
        assert imad.sequence_number == "0001"

    def test_imad_invalid_length(self):
        """Test IMAD parsing with invalid length."""
        with pytest.raises(ValueError):
            IMAD.from_string("12345")  # Too short


class TestAddress:
    """Tests for Address data class."""

    def test_address_creation(self):
        """Test address creation."""
        addr = Address(
            line1="123 Main St",
            city="New York",
            state="NY",
            postal_code="10001",
            country="US",
        )

        assert addr.line1 == "123 Main St"
        assert addr.city == "New York"

    def test_address_to_lines(self):
        """Test address conversion to lines."""
        addr = Address(
            line1="123 Main St",
            city="New York",
            state="NY",
            postal_code="10001",
        )

        lines = addr.to_lines()
        assert len(lines) >= 2
        assert "123 Main St" in lines


class TestSenderInfo:
    """Tests for SenderInfo data class."""

    def test_sender_creation(self):
        """Test sender info creation."""
        sender = SenderInfo(
            routing_number="021000089",
            short_name="CITIBANK NA",
        )

        assert sender.routing_number == "021000089"
        assert sender.short_name == "CITIBANK NA"

    def test_sender_validation_valid(self):
        """Test validation of valid sender."""
        sender = SenderInfo(
            routing_number="021000089",
            short_name="CITIBANK NA",
        )

        errors = sender.validate()
        assert len(errors) == 0

    def test_sender_validation_missing_routing(self):
        """Test validation catches missing routing."""
        sender = SenderInfo(
            routing_number="",
            short_name="CITIBANK NA",
        )

        errors = sender.validate()
        assert len(errors) > 0
        assert any("routing" in e.lower() for e in errors)

    def test_sender_validation_invalid_routing(self):
        """Test validation catches invalid routing."""
        sender = SenderInfo(
            routing_number="12345",  # Too short
            short_name="CITIBANK NA",
        )

        errors = sender.validate()
        assert len(errors) > 0


class TestFedWireMessage:
    """Tests for FedWireMessage data class."""

    def test_message_creation(self):
        """Test message creation."""
        msg = FedWireMessage(
            type_code=TypeCode.BASIC_FUNDS_TRANSFER,
            type_subcode=TypeSubCode.BASIC_TRANSFER,
            business_function_code=BusinessFunctionCode.BANK_TRANSFER,
            amount=Decimal("10000.00"),
        )

        assert msg.type_code == TypeCode.BASIC_FUNDS_TRANSFER
        assert msg.amount == Decimal("10000.00")

    def test_message_validation_valid(self):
        """Test validation of valid message."""
        msg = FedWireMessage(
            type_code=TypeCode.BASIC_FUNDS_TRANSFER,
            type_subcode=TypeSubCode.BASIC_TRANSFER,
            business_function_code=BusinessFunctionCode.BANK_TRANSFER,
            amount=Decimal("10000.00"),
            sender=SenderInfo(
                routing_number="021000089",
                short_name="CITIBANK NA",
            ),
            receiver=ReceiverInfo(
                routing_number="021000021",
                short_name="JPMORGAN CHASE",
            ),
            imad=IMAD(
                source_id="CITI1234",
                cycle_code="01",
                sequence_number="000001",
            ),
        )

        errors = msg.validate()
        assert len(errors) == 0

    def test_message_validation_ctr_requires_beneficiary(self):
        """Test CTR requires beneficiary."""
        msg = FedWireMessage(
            type_code=TypeCode.BASIC_FUNDS_TRANSFER,
            business_function_code=BusinessFunctionCode.CUSTOMER_TRANSFER,
            amount=Decimal("10000.00"),
            sender=SenderInfo(routing_number="021000089", short_name="CITI"),
            receiver=ReceiverInfo(routing_number="021000021", short_name="JPM"),
            imad=IMAD(source_id="CITI1234"),
            # Missing beneficiary
        )

        errors = msg.validate()
        assert len(errors) > 0
        assert any("beneficiary" in e.lower() for e in errors)

    def test_message_to_dict(self):
        """Test message serialization."""
        msg = FedWireMessage(
            type_code=TypeCode.BASIC_FUNDS_TRANSFER,
            business_function_code=BusinessFunctionCode.BANK_TRANSFER,
            amount=Decimal("10000.00"),
            sender=SenderInfo(routing_number="021000089", short_name="CITI"),
            receiver=ReceiverInfo(routing_number="021000021", short_name="JPM"),
        )

        data = msg.to_dict()
        assert "type_code" in data
        assert "amount" in data
        assert "sender" in data


class TestFedWireParser:
    """Tests for FedWire message parser."""

    @pytest.fixture
    def sample_wire_message(self):
        """Create sample wire format message."""
        return (
            "{1500}30    P {1510}1000"
            "{1520}20231215CITI123401000001"
            "{2000}000001000000"
            "{3100}021000089CITIBANK NA"
            "{3400}021000021JPMORGAN CHASE"
            "{3600}BTR"
        )

    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = FedWireParser()
        assert parser.strict is True

        parser = FedWireParser(strict=False)
        assert parser.strict is False

    def test_parse_basic_message(self, sample_wire_message):
        """Test parsing basic wire message."""
        parser = FedWireParser(strict=False)
        msg = parser.parse(sample_wire_message)

        assert msg is not None
        assert msg.type_code == TypeCode.BASIC_FUNDS_TRANSFER
        assert msg.business_function_code == BusinessFunctionCode.BANK_TRANSFER

    def test_parse_amount(self, sample_wire_message):
        """Test amount parsing."""
        parser = FedWireParser(strict=False)
        msg = parser.parse(sample_wire_message)

        # Amount is in cents: 000001000000 = $10,000.00
        assert msg.amount == Decimal("10000.00")

    def test_parse_sender_info(self, sample_wire_message):
        """Test sender parsing."""
        parser = FedWireParser(strict=False)
        msg = parser.parse(sample_wire_message)

        assert msg.sender.routing_number == "021000089"
        assert "CITIBANK" in msg.sender.short_name

    def test_parse_missing_mandatory_tag(self):
        """Test parsing message with missing mandatory tag."""
        incomplete = "{1500}30    P {1510}1000"  # Missing most tags

        parser = FedWireParser(strict=True)
        with pytest.raises(FedWireParseError):
            parser.parse(incomplete)

    def test_parse_optional_tags(self):
        """Test parsing optional tags."""
        msg_content = (
            "{1500}30    P {1510}1000"
            "{1520}20231215CITI123401000001"
            "{2000}000001000000"
            "{3100}021000089CITIBANK NA"
            "{3400}021000021JPMORGAN CHASE"
            "{3600}CTR"
            "{4200}D123456789*ACME CORPORATION*123 MAIN ST*NEW YORK NY"
            "{6000}PAYMENT FOR INVOICE 12345"
        )

        parser = FedWireParser(strict=False)
        msg = parser.parse(msg_content)

        assert msg.beneficiary is not None
        assert msg.beneficiary.name == "ACME CORPORATION"
        assert msg.originator_to_beneficiary is not None


class TestFedWireBuilder:
    """Tests for FedWire message builder."""

    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = FedWireBuilder()
        assert builder is not None

    def test_build_bank_transfer(self):
        """Test building bank transfer message."""
        builder = FedWireBuilder()

        content = (builder
            .set_type(TypeCode.BASIC_FUNDS_TRANSFER, TypeSubCode.BASIC_TRANSFER)
            .set_imad("CITI1234")
            .set_amount(Decimal("10000.00"))
            .set_sender("021000089", "CITIBANK NA")
            .set_receiver("021000021", "JPMORGAN CHASE")
            .set_business_function(BusinessFunctionCode.BANK_TRANSFER)
            .build()
        )

        assert content is not None
        assert "{1500}" in content  # Sender supplied
        assert "{2000}" in content  # Amount
        assert "{3100}" in content  # Sender
        assert "{3600}" in content  # Business function

    def test_build_customer_transfer(self):
        """Test building customer transfer message."""
        builder = FedWireBuilder()

        content = (builder
            .set_type(TypeCode.BASIC_FUNDS_TRANSFER)
            .set_imad("CITI1234")
            .set_amount(Decimal("25000.00"))
            .set_sender("021000089", "CITIBANK NA")
            .set_receiver("021000021", "JPMORGAN CHASE")
            .set_business_function(BusinessFunctionCode.CUSTOMER_TRANSFER)
            .set_beneficiary("ACME CORPORATION", "123456789", IDCode.DDA_NUMBER)
            .set_originator_to_beneficiary_info("INVOICE 12345", "PAYMENT DUE")
            .build()
        )

        assert content is not None
        assert "{4200}" in content  # Beneficiary
        assert "{6000}" in content  # Originator to beneficiary

    def test_build_validation_failure(self):
        """Test build fails with invalid data."""
        builder = FedWireBuilder()

        builder.set_type(TypeCode.BASIC_FUNDS_TRANSFER)
        builder.set_imad("CITI1234")
        builder.set_amount(Decimal("10000.00"))
        # Missing sender, receiver, business function

        with pytest.raises(FedWireBuildError):
            builder.build()

    def test_build_message_object(self):
        """Test building FedWireMessage object."""
        builder = FedWireBuilder()

        msg = (builder
            .set_type(TypeCode.BASIC_FUNDS_TRANSFER)
            .set_imad("CITI1234")
            .set_amount(Decimal("10000.00"))
            .set_sender("021000089", "CITIBANK NA")
            .set_receiver("021000021", "JPMORGAN CHASE")
            .set_business_function(BusinessFunctionCode.BANK_TRANSFER)
            .build_message()
        )

        assert isinstance(msg, FedWireMessage)
        assert msg.amount == Decimal("10000.00")


class TestHelperFunctions:
    """Tests for FedWire helper functions."""

    def test_create_bank_transfer(self):
        """Test create_bank_transfer helper."""
        content = create_bank_transfer(
            sender_routing="021000089",
            sender_name="CITIBANK NA",
            receiver_routing="021000021",
            receiver_name="JPMORGAN CHASE",
            amount=Decimal("50000.00"),
            source_id="CITI1234",
            sender_reference="REF123",
        )

        assert content is not None
        assert "{3600}BTR" in content  # Bank transfer

    def test_create_customer_transfer(self):
        """Test create_customer_transfer helper."""
        content = create_customer_transfer(
            sender_routing="021000089",
            sender_name="CITIBANK NA",
            receiver_routing="021000021",
            receiver_name="JPMORGAN CHASE",
            amount=Decimal("25000.00"),
            source_id="CITI1234",
            beneficiary_name="ACME CORP",
            beneficiary_account="123456789",
            payment_details="INVOICE 12345",
        )

        assert content is not None
        assert "{3600}CTR" in content  # Customer transfer
        assert "{4200}" in content  # Beneficiary

    def test_create_customer_transfer_plus(self):
        """Test CTP creation with originator."""
        content = create_customer_transfer(
            sender_routing="021000089",
            sender_name="CITIBANK NA",
            receiver_routing="021000021",
            receiver_name="JPMORGAN CHASE",
            amount=Decimal("100000.00"),
            source_id="CITI1234",
            beneficiary_name="ACME CORP",
            beneficiary_account="123456789",
            originator_name="SMITH INDUSTRIES",
            originator_account="987654321",
        )

        assert content is not None
        assert "{3600}CTP" in content  # Customer transfer plus
        assert "{5000}" in content  # Originator

    def test_create_cover_payment(self):
        """Test create_cover_payment helper."""
        content = create_cover_payment(
            sender_routing="021000089",
            sender_name="CITIBANK NA",
            receiver_routing="021000021",
            receiver_name="JPMORGAN CHASE",
            amount=Decimal("1000000.00"),
            source_id="CITI1234",
            beneficiary_name="OVERSEAS CORP",
            beneficiary_fi_bic="DEUTDEFF",
            originator_name="US COMPANY",
            originator_fi_bic="CITIUS33",
            underlying_reference="MT103-REF-12345",
        )

        assert content is not None
        assert "{3600}COV" in content  # Cover payment
        assert "{4000}" in content  # Beneficiary FI
        assert "{5100}" in content  # Originator FI
