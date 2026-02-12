"""
Tests for FIX Protocol Implementation

Tests cover:
- FIX message parsing
- FIX message building
- Order messages (New Order Single, Execution Report)
- Message validation
"""

import pytest
from decimal import Decimal
from datetime import datetime

from ai_engine.domains.banking.protocols.trading.fix import (
    FixParser,
    FixParseError,
    FixBuilder,
    FixMessageFactory,
    FixValidator,
    FixVersion,
    FixMsgType,
    FixOrdType,
    FixSide,
    FixTimeInForce,
    FixExecType,
    FixOrdStatus,
    FixMessage,
    FixField,
    FIX_TAG_NAMES,
    get_tag_name,
)

# FIX message delimiter
SOH = "\x01"


class TestFixParser:
    """Tests for FIX message parsing."""

    def test_parse_simple_message(self):
        """Test parsing a simple FIX message."""
        raw = f"8=FIX.4.4{SOH}9=100{SOH}35=D{SOH}49=SENDER{SOH}56=TARGET{SOH}34=1{SOH}52=20230115-10:30:00{SOH}11=ORDER001{SOH}54=1{SOH}55=AAPL{SOH}38=100{SOH}40=2{SOH}44=150.00{SOH}10=123{SOH}"

        parser = FixParser()
        message = parser.parse(raw)

        assert message.header.begin_string == "FIX.4.4"
        assert message.header.msg_type == "D"  # New Order Single
        assert message.header.sender_comp_id == "SENDER"
        assert message.header.target_comp_id == "TARGET"

    def test_parse_extract_fields(self):
        """Test extracting fields from parsed message."""
        raw = f"8=FIX.4.4{SOH}9=50{SOH}35=D{SOH}49=SENDER{SOH}56=TARGET{SOH}34=1{SOH}11=ORD123{SOH}55=MSFT{SOH}54=2{SOH}38=500{SOH}40=1{SOH}10=000{SOH}"

        parser = FixParser()
        message = parser.parse(raw)

        # Check specific fields
        cl_ord_id = message.get_field(11)
        assert cl_ord_id == "ORD123"

        symbol = message.get_field(55)
        assert symbol == "MSFT"

        side = message.get_field(54)
        assert side == "2"  # Sell

        order_qty = message.get_field(38)
        assert order_qty == "500"

    def test_parse_execution_report(self):
        """Test parsing execution report."""
        raw = f"8=FIX.4.4{SOH}9=150{SOH}35=8{SOH}49=EXCHANGE{SOH}56=BROKER{SOH}34=5{SOH}37=EXECID001{SOH}11=ORDER001{SOH}17=EXEC001{SOH}150=0{SOH}39=0{SOH}55=AAPL{SOH}54=1{SOH}38=100{SOH}14=0{SOH}151=100{SOH}6=0{SOH}10=999{SOH}"

        parser = FixParser()
        message = parser.parse(raw)

        assert message.header.msg_type == "8"  # Execution Report

        exec_type = message.get_field(150)
        assert exec_type == "0"  # New

        ord_status = message.get_field(39)
        assert ord_status == "0"  # New

    def test_parse_checksum_validation(self):
        """Test checksum calculation."""
        raw = f"8=FIX.4.4{SOH}9=20{SOH}35=0{SOH}49=A{SOH}56=B{SOH}34=1{SOH}10=123{SOH}"

        parser = FixParser(validate_checksum=False)
        message = parser.parse(raw)

        # Calculate expected checksum
        calculated = message.calculate_checksum()
        assert len(calculated) == 3

    def test_parse_error_invalid_format(self):
        """Test parsing invalid message format."""
        raw = "NOT A FIX MESSAGE"

        parser = FixParser()
        with pytest.raises(FixParseError):
            parser.parse(raw)

    def test_parse_error_missing_header(self):
        """Test parsing message missing header fields."""
        raw = f"35=D{SOH}11=ORDER001{SOH}10=000{SOH}"

        parser = FixParser()
        with pytest.raises(FixParseError):
            parser.parse(raw)


class TestFixBuilder:
    """Tests for FIX message building."""

    def test_build_new_order_single(self):
        """Test building New Order Single."""
        builder = FixBuilder()
        message = (
            builder.set_version(FixVersion.FIX_44)
            .set_msg_type(FixMsgType.NEW_ORDER_SINGLE)
            .set_sender("TRADER")
            .set_target("EXCHANGE")
            .set_seq_num(1)
            .set_field(11, "ORDER001")  # ClOrdID
            .set_symbol("AAPL")
            .set_side(FixSide.BUY)
            .set_order_qty(100)
            .set_ord_type(FixOrdType.LIMIT)
            .set_price(Decimal("150.00"))
            .set_time_in_force(FixTimeInForce.DAY)
            .build()
        )

        assert message.header.msg_type == "D"
        assert message.get_field(55) == "AAPL"
        assert message.get_field(54) == "1"  # Buy
        assert message.get_field(38) == "100"

    def test_build_order_cancel_request(self):
        """Test building Order Cancel Request."""
        builder = FixBuilder()
        message = (
            builder.set_version(FixVersion.FIX_44)
            .set_msg_type(FixMsgType.ORDER_CANCEL_REQUEST)
            .set_sender("TRADER")
            .set_target("EXCHANGE")
            .set_seq_num(2)
            .set_field(11, "CANCEL001")  # ClOrdID
            .set_field(41, "ORDER001")  # OrigClOrdID
            .set_symbol("AAPL")
            .set_side(FixSide.BUY)
            .build()
        )

        assert message.header.msg_type == "F"
        assert message.get_field(41) == "ORDER001"

    def test_build_to_fix_string(self):
        """Test converting message to FIX string."""
        builder = FixBuilder()
        message = (
            builder.set_version(FixVersion.FIX_44)
            .set_msg_type(FixMsgType.HEARTBEAT)
            .set_sender("A")
            .set_target("B")
            .set_seq_num(1)
            .build()
        )

        fix_string = message.to_fix()

        assert "8=FIX.4.4" in fix_string
        assert "35=0" in fix_string
        assert "49=A" in fix_string
        assert "56=B" in fix_string
        assert "10=" in fix_string  # Checksum


class TestFixMessageFactory:
    """Tests for FIX message factory."""

    def test_create_new_order_single(self):
        """Test factory method for New Order Single."""
        message = FixMessageFactory.create_new_order_single(
            sender="TRADER",
            target="EXCHANGE",
            seq_num=1,
            cl_ord_id="ORD20230115001",
            symbol="MSFT",
            side=FixSide.BUY,
            order_qty=1000,
            ord_type=FixOrdType.MARKET,
        )

        assert message.header.msg_type == "D"
        assert message.get_field(11) == "ORD20230115001"
        assert message.get_field(55) == "MSFT"
        assert message.get_field(40) == "1"  # Market

    def test_create_execution_report(self):
        """Test factory method for Execution Report."""
        message = FixMessageFactory.create_execution_report(
            sender="EXCHANGE",
            target="TRADER",
            seq_num=5,
            order_id="EXCHORD001",
            cl_ord_id="ORD001",
            exec_id="EXEC001",
            exec_type=FixExecType.FILL,
            ord_status=FixOrdStatus.FILLED,
            symbol="AAPL",
            side=FixSide.BUY,
            order_qty=100,
            last_qty=100,
            last_px=Decimal("152.50"),
            leaves_qty=0,
            cum_qty=100,
            avg_px=Decimal("152.50"),
        )

        assert message.header.msg_type == "8"
        assert message.get_field(150) == "F"  # Fill
        assert message.get_field(39) == "2"  # Filled
        assert message.get_field(32) == "100"  # LastQty

    def test_create_market_data_request(self):
        """Test factory method for Market Data Request."""
        message = FixMessageFactory.create_market_data_request(
            sender="TRADER",
            target="EXCHANGE",
            seq_num=3,
            md_req_id="MDR001",
            subscription_type="1",  # Snapshot + Updates
            market_depth=5,
            symbols=["AAPL", "MSFT", "GOOGL"],
        )

        assert message.header.msg_type == "V"
        assert message.get_field(262) == "MDR001"


class TestFixValidator:
    """Tests for FIX message validation."""

    def test_validate_new_order_single(self):
        """Test validation of New Order Single."""
        message = FixMessageFactory.create_new_order_single(
            sender="TRADER",
            target="EXCHANGE",
            seq_num=1,
            cl_ord_id="ORD001",
            symbol="AAPL",
            side=FixSide.BUY,
            order_qty=100,
            ord_type=FixOrdType.LIMIT,
            price=Decimal("150.00"),
        )

        validator = FixValidator()
        result = validator.validate(message)

        assert result.is_valid

    def test_validate_missing_required_field(self):
        """Test validation catches missing required fields."""
        # Create message missing ClOrdID
        builder = FixBuilder()
        message = (
            builder.set_version(FixVersion.FIX_44)
            .set_msg_type(FixMsgType.NEW_ORDER_SINGLE)
            .set_sender("TRADER")
            .set_target("EXCHANGE")
            .set_seq_num(1)
            .set_symbol("AAPL")
            .set_side(FixSide.BUY)
            .build()
        )

        validator = FixValidator()
        result = validator.validate(message)

        # Should have validation errors
        assert not result.is_valid
        assert any("ClOrdID" in e.message or "11" in e.message for e in result.errors)

    def test_validate_limit_order_missing_price(self):
        """Test validation catches limit order without price."""
        message = FixMessageFactory.create_new_order_single(
            sender="TRADER",
            target="EXCHANGE",
            seq_num=1,
            cl_ord_id="ORD001",
            symbol="AAPL",
            side=FixSide.BUY,
            order_qty=100,
            ord_type=FixOrdType.LIMIT,
            # Missing price for limit order
        )

        validator = FixValidator()
        result = validator.validate(message)

        assert not result.is_valid
        assert any("price" in e.message.lower() for e in result.errors)

    def test_validate_execution_report(self):
        """Test validation of Execution Report."""
        message = FixMessageFactory.create_execution_report(
            sender="EXCHANGE",
            target="TRADER",
            seq_num=1,
            order_id="ORD001",
            cl_ord_id="CLORD001",
            exec_id="EXEC001",
            exec_type=FixExecType.NEW,
            ord_status=FixOrdStatus.NEW,
            symbol="AAPL",
            side=FixSide.BUY,
            order_qty=100,
            leaves_qty=100,
            cum_qty=0,
            avg_px=Decimal("0"),
        )

        validator = FixValidator()
        result = validator.validate(message)

        assert result.is_valid


class TestFixCodes:
    """Tests for FIX code definitions."""

    def test_version_enum(self):
        """Test FIX version enum."""
        assert FixVersion.FIX_42.value == "FIX.4.2"
        assert FixVersion.FIX_44.value == "FIX.4.4"
        assert FixVersion.FIX_50.value == "FIX.5.0"

    def test_msg_type_enum(self):
        """Test message type enum."""
        assert FixMsgType.NEW_ORDER_SINGLE.code == "D"
        assert FixMsgType.EXECUTION_REPORT.code == "8"
        assert FixMsgType.ORDER_CANCEL_REQUEST.code == "F"
        assert FixMsgType.HEARTBEAT.code == "0"

    def test_side_enum(self):
        """Test side enum."""
        assert FixSide.BUY.code == "1"
        assert FixSide.SELL.code == "2"
        assert FixSide.SHORT_SELL.code == "5"

    def test_ord_type_enum(self):
        """Test order type enum."""
        assert FixOrdType.MARKET.code == "1"
        assert FixOrdType.LIMIT.code == "2"
        assert FixOrdType.STOP.code == "3"
        assert FixOrdType.STOP_LIMIT.code == "4"

    def test_tag_name_lookup(self):
        """Test tag name lookup."""
        assert get_tag_name(11) == "ClOrdID"
        assert get_tag_name(38) == "OrderQty"
        assert get_tag_name(44) == "Price"
        assert get_tag_name(54) == "Side"
        assert get_tag_name(55) == "Symbol"
