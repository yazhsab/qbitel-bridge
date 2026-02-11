"""
FIX Message Builder

Builder classes for creating FIX messages programmatically.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
import uuid

from ai_engine.domains.banking.protocols.trading.fix.fix_message import (
    FixMessage,
    FixHeader,
    FixTrailer,
    FixField,
    FixGroup,
    SOH,
)
from ai_engine.domains.banking.protocols.trading.fix.fix_codes import (
    FixVersion,
    FixMsgType,
    FixOrdType,
    FixSide,
    FixTimeInForce,
    FixExecType,
    FixOrdStatus,
    FixHandlInst,
    FixSecurityType,
    FixSettlType,
)


class FixBuilder:
    """
    Builder for FIX messages.

    Provides a fluent interface for constructing FIX messages.
    """

    def __init__(self, version: FixVersion = FixVersion.FIX_4_4):
        """
        Initialize builder.

        Args:
            version: FIX protocol version
        """
        self._message = FixMessage()
        self._message.header.begin_string = version.begin_string
        self._message.header.sending_time = datetime.utcnow()

    def reset(self, version: Optional[FixVersion] = None) -> "FixBuilder":
        """Reset the builder."""
        current_version = self._message.header.begin_string
        self._message = FixMessage()
        self._message.header.begin_string = (
            version.begin_string if version else current_version
        )
        self._message.header.sending_time = datetime.utcnow()
        return self

    # Header methods
    def set_msg_type(self, msg_type: Union[FixMsgType, str]) -> "FixBuilder":
        """Set message type."""
        if isinstance(msg_type, FixMsgType):
            self._message.header.msg_type = msg_type.code
        else:
            self._message.header.msg_type = msg_type
        return self

    def set_sender(self, sender_comp_id: str, sub_id: str = None) -> "FixBuilder":
        """Set sender CompID."""
        self._message.header.sender_comp_id = sender_comp_id
        if sub_id:
            self._message.header.sender_sub_id = sub_id
        return self

    def set_target(self, target_comp_id: str, sub_id: str = None) -> "FixBuilder":
        """Set target CompID."""
        self._message.header.target_comp_id = target_comp_id
        if sub_id:
            self._message.header.target_sub_id = sub_id
        return self

    def set_seq_num(self, seq_num: int) -> "FixBuilder":
        """Set message sequence number."""
        self._message.header.msg_seq_num = seq_num
        return self

    def set_sending_time(self, sending_time: datetime) -> "FixBuilder":
        """Set sending time."""
        self._message.header.sending_time = sending_time
        return self

    def set_poss_dup(self, poss_dup: bool = True) -> "FixBuilder":
        """Set possible duplicate flag."""
        self._message.header.poss_dup_flag = poss_dup
        return self

    def set_on_behalf_of(
        self, comp_id: str, sub_id: str = None
    ) -> "FixBuilder":
        """Set on behalf of CompID."""
        self._message.header.on_behalf_of_comp_id = comp_id
        if sub_id:
            self._message.header.on_behalf_of_sub_id = sub_id
        return self

    # Body field methods
    def set_field(self, tag: int, value: Any) -> "FixBuilder":
        """Set a body field."""
        self._message.set_field(tag, value)
        return self

    def add_field(self, tag: int, value: Any) -> "FixBuilder":
        """Add a body field (allows duplicates)."""
        self._message.add_field(tag, value)
        return self

    # Order fields
    def set_cl_ord_id(self, cl_ord_id: str) -> "FixBuilder":
        """Set ClOrdID (tag 11)."""
        return self.set_field(11, cl_ord_id)

    def set_orig_cl_ord_id(self, orig_cl_ord_id: str) -> "FixBuilder":
        """Set OrigClOrdID (tag 41)."""
        return self.set_field(41, orig_cl_ord_id)

    def set_order_id(self, order_id: str) -> "FixBuilder":
        """Set OrderID (tag 37)."""
        return self.set_field(37, order_id)

    def set_exec_id(self, exec_id: str) -> "FixBuilder":
        """Set ExecID (tag 17)."""
        return self.set_field(17, exec_id)

    def set_symbol(self, symbol: str) -> "FixBuilder":
        """Set Symbol (tag 55)."""
        return self.set_field(55, symbol)

    def set_security_id(
        self, security_id: str, id_source: str = "4"
    ) -> "FixBuilder":
        """Set SecurityID (tag 48) and SecurityIDSource (tag 22)."""
        self.set_field(48, security_id)
        return self.set_field(22, id_source)

    def set_security_type(
        self, security_type: Union[FixSecurityType, str]
    ) -> "FixBuilder":
        """Set SecurityType (tag 167)."""
        if isinstance(security_type, FixSecurityType):
            return self.set_field(167, security_type.code)
        return self.set_field(167, security_type)

    def set_side(self, side: Union[FixSide, str]) -> "FixBuilder":
        """Set Side (tag 54)."""
        if isinstance(side, FixSide):
            return self.set_field(54, side.code)
        return self.set_field(54, side)

    def set_order_qty(self, qty: Union[int, float, Decimal]) -> "FixBuilder":
        """Set OrderQty (tag 38)."""
        return self.set_field(38, qty)

    def set_order_type(self, order_type: Union[FixOrdType, str]) -> "FixBuilder":
        """Set OrdType (tag 40)."""
        if isinstance(order_type, FixOrdType):
            return self.set_field(40, order_type.code)
        return self.set_field(40, order_type)

    def set_price(self, price: Union[float, Decimal]) -> "FixBuilder":
        """Set Price (tag 44)."""
        return self.set_field(44, price)

    def set_stop_price(self, stop_price: Union[float, Decimal]) -> "FixBuilder":
        """Set StopPx (tag 99)."""
        return self.set_field(99, stop_price)

    def set_time_in_force(
        self, tif: Union[FixTimeInForce, str]
    ) -> "FixBuilder":
        """Set TimeInForce (tag 59)."""
        if isinstance(tif, FixTimeInForce):
            return self.set_field(59, tif.code)
        return self.set_field(59, tif)

    def set_handl_inst(
        self, handl_inst: Union[FixHandlInst, str]
    ) -> "FixBuilder":
        """Set HandlInst (tag 21)."""
        if isinstance(handl_inst, FixHandlInst):
            return self.set_field(21, handl_inst.code)
        return self.set_field(21, handl_inst)

    def set_transact_time(self, transact_time: datetime = None) -> "FixBuilder":
        """Set TransactTime (tag 60)."""
        if transact_time is None:
            transact_time = datetime.utcnow()
        return self.set_field(
            60, transact_time.strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
        )

    def set_currency(self, currency: str) -> "FixBuilder":
        """Set Currency (tag 15)."""
        return self.set_field(15, currency)

    def set_account(self, account: str) -> "FixBuilder":
        """Set Account (tag 1)."""
        return self.set_field(1, account)

    # Execution report fields
    def set_exec_type(self, exec_type: Union[FixExecType, str]) -> "FixBuilder":
        """Set ExecType (tag 150)."""
        if isinstance(exec_type, FixExecType):
            return self.set_field(150, exec_type.code)
        return self.set_field(150, exec_type)

    def set_ord_status(self, ord_status: Union[FixOrdStatus, str]) -> "FixBuilder":
        """Set OrdStatus (tag 39)."""
        if isinstance(ord_status, FixOrdStatus):
            return self.set_field(39, ord_status.code)
        return self.set_field(39, ord_status)

    def set_leaves_qty(self, leaves_qty: Union[int, float, Decimal]) -> "FixBuilder":
        """Set LeavesQty (tag 151)."""
        return self.set_field(151, leaves_qty)

    def set_cum_qty(self, cum_qty: Union[int, float, Decimal]) -> "FixBuilder":
        """Set CumQty (tag 14)."""
        return self.set_field(14, cum_qty)

    def set_avg_px(self, avg_px: Union[float, Decimal]) -> "FixBuilder":
        """Set AvgPx (tag 6)."""
        return self.set_field(6, avg_px)

    def set_last_qty(self, last_qty: Union[int, float, Decimal]) -> "FixBuilder":
        """Set LastQty (tag 32)."""
        return self.set_field(32, last_qty)

    def set_last_px(self, last_px: Union[float, Decimal]) -> "FixBuilder":
        """Set LastPx (tag 31)."""
        return self.set_field(31, last_px)

    # Settlement fields
    def set_settl_type(self, settl_type: Union[FixSettlType, str]) -> "FixBuilder":
        """Set SettlType (tag 63)."""
        if isinstance(settl_type, FixSettlType):
            return self.set_field(63, settl_type.code)
        return self.set_field(63, settl_type)

    def set_settl_date(self, settl_date: date) -> "FixBuilder":
        """Set SettlDate (tag 64)."""
        return self.set_field(64, settl_date.strftime("%Y%m%d"))

    def set_trade_date(self, trade_date: date) -> "FixBuilder":
        """Set TradeDate (tag 75)."""
        return self.set_field(75, trade_date.strftime("%Y%m%d"))

    # Text and reference fields
    def set_text(self, text: str) -> "FixBuilder":
        """Set Text (tag 58)."""
        return self.set_field(58, text)

    def set_exec_broker(self, broker: str) -> "FixBuilder":
        """Set ExecBroker (tag 76)."""
        return self.set_field(76, broker)

    # Group methods
    def add_group(self, group: FixGroup) -> "FixBuilder":
        """Add a repeating group."""
        self._message.add_group(group)
        return self

    def add_party(
        self,
        party_id: str,
        party_id_source: str,
        party_role: int,
    ) -> "FixBuilder":
        """Add a party to NoPartyIDs group (tag 453)."""
        group = self._message.get_group(453)
        if not group:
            group = FixGroup(count_tag=453)
            self._message.add_group(group)

        group.add_entry({
            448: party_id,
            447: party_id_source,
            452: str(party_role),
        })
        return self

    # Build method
    def build(self) -> FixMessage:
        """Build and return the FIX message."""
        return self._message


class FixMessageFactory:
    """
    Factory for creating common FIX messages.
    """

    @staticmethod
    def create_logon(
        sender: str,
        target: str,
        seq_num: int,
        heartbeat_interval: int = 30,
        encrypt_method: int = 0,
        version: FixVersion = FixVersion.FIX_4_4,
    ) -> FixMessage:
        """Create a Logon message (MsgType=A)."""
        builder = FixBuilder(version)
        builder.set_msg_type(FixMsgType.LOGON)
        builder.set_sender(sender)
        builder.set_target(target)
        builder.set_seq_num(seq_num)
        builder.set_field(98, encrypt_method)  # EncryptMethod
        builder.set_field(108, heartbeat_interval)  # HeartBtInt
        return builder.build()

    @staticmethod
    def create_logout(
        sender: str,
        target: str,
        seq_num: int,
        text: str = None,
        version: FixVersion = FixVersion.FIX_4_4,
    ) -> FixMessage:
        """Create a Logout message (MsgType=5)."""
        builder = FixBuilder(version)
        builder.set_msg_type(FixMsgType.LOGOUT)
        builder.set_sender(sender)
        builder.set_target(target)
        builder.set_seq_num(seq_num)
        if text:
            builder.set_text(text)
        return builder.build()

    @staticmethod
    def create_heartbeat(
        sender: str,
        target: str,
        seq_num: int,
        test_req_id: str = None,
        version: FixVersion = FixVersion.FIX_4_4,
    ) -> FixMessage:
        """Create a Heartbeat message (MsgType=0)."""
        builder = FixBuilder(version)
        builder.set_msg_type(FixMsgType.HEARTBEAT)
        builder.set_sender(sender)
        builder.set_target(target)
        builder.set_seq_num(seq_num)
        if test_req_id:
            builder.set_field(112, test_req_id)  # TestReqID
        return builder.build()

    @staticmethod
    def create_test_request(
        sender: str,
        target: str,
        seq_num: int,
        test_req_id: str = None,
        version: FixVersion = FixVersion.FIX_4_4,
    ) -> FixMessage:
        """Create a Test Request message (MsgType=1)."""
        builder = FixBuilder(version)
        builder.set_msg_type(FixMsgType.TEST_REQUEST)
        builder.set_sender(sender)
        builder.set_target(target)
        builder.set_seq_num(seq_num)
        builder.set_field(112, test_req_id or str(uuid.uuid4())[:8])
        return builder.build()

    @staticmethod
    def create_new_order_single(
        sender: str,
        target: str,
        seq_num: int,
        cl_ord_id: str,
        symbol: str,
        side: FixSide,
        order_qty: Union[int, float, Decimal],
        order_type: FixOrdType,
        price: Union[float, Decimal] = None,
        time_in_force: FixTimeInForce = FixTimeInForce.DAY,
        account: str = None,
        currency: str = None,
        handl_inst: FixHandlInst = FixHandlInst.AUTOMATED_NO_INTERVENTION,
        version: FixVersion = FixVersion.FIX_4_4,
    ) -> FixMessage:
        """Create a New Order Single message (MsgType=D)."""
        builder = FixBuilder(version)
        builder.set_msg_type(FixMsgType.NEW_ORDER_SINGLE)
        builder.set_sender(sender)
        builder.set_target(target)
        builder.set_seq_num(seq_num)
        builder.set_cl_ord_id(cl_ord_id)
        builder.set_symbol(symbol)
        builder.set_side(side)
        builder.set_order_qty(order_qty)
        builder.set_order_type(order_type)
        builder.set_time_in_force(time_in_force)
        builder.set_handl_inst(handl_inst)
        builder.set_transact_time()

        if price is not None:
            builder.set_price(price)
        if account:
            builder.set_account(account)
        if currency:
            builder.set_currency(currency)

        return builder.build()

    @staticmethod
    def create_execution_report(
        sender: str,
        target: str,
        seq_num: int,
        order_id: str,
        exec_id: str,
        exec_type: FixExecType,
        ord_status: FixOrdStatus,
        symbol: str,
        side: FixSide,
        leaves_qty: Union[int, float, Decimal],
        cum_qty: Union[int, float, Decimal],
        avg_px: Union[float, Decimal],
        cl_ord_id: str = None,
        last_qty: Union[int, float, Decimal] = None,
        last_px: Union[float, Decimal] = None,
        text: str = None,
        version: FixVersion = FixVersion.FIX_4_4,
    ) -> FixMessage:
        """Create an Execution Report message (MsgType=8)."""
        builder = FixBuilder(version)
        builder.set_msg_type(FixMsgType.EXECUTION_REPORT)
        builder.set_sender(sender)
        builder.set_target(target)
        builder.set_seq_num(seq_num)
        builder.set_order_id(order_id)
        builder.set_exec_id(exec_id)
        builder.set_exec_type(exec_type)
        builder.set_ord_status(ord_status)
        builder.set_symbol(symbol)
        builder.set_side(side)
        builder.set_leaves_qty(leaves_qty)
        builder.set_cum_qty(cum_qty)
        builder.set_avg_px(avg_px)
        builder.set_transact_time()

        if cl_ord_id:
            builder.set_cl_ord_id(cl_ord_id)
        if last_qty is not None:
            builder.set_last_qty(last_qty)
        if last_px is not None:
            builder.set_last_px(last_px)
        if text:
            builder.set_text(text)

        # For FIX 4.2, add ExecTransType
        if version in (FixVersion.FIX_4_0, FixVersion.FIX_4_1, FixVersion.FIX_4_2):
            builder.set_field(20, "0")  # ExecTransType = New

        return builder.build()

    @staticmethod
    def create_order_cancel_request(
        sender: str,
        target: str,
        seq_num: int,
        orig_cl_ord_id: str,
        cl_ord_id: str,
        symbol: str,
        side: FixSide,
        order_id: str = None,
        version: FixVersion = FixVersion.FIX_4_4,
    ) -> FixMessage:
        """Create an Order Cancel Request message (MsgType=F)."""
        builder = FixBuilder(version)
        builder.set_msg_type(FixMsgType.ORDER_CANCEL_REQUEST)
        builder.set_sender(sender)
        builder.set_target(target)
        builder.set_seq_num(seq_num)
        builder.set_orig_cl_ord_id(orig_cl_ord_id)
        builder.set_cl_ord_id(cl_ord_id)
        builder.set_symbol(symbol)
        builder.set_side(side)
        builder.set_transact_time()

        if order_id:
            builder.set_order_id(order_id)

        return builder.build()

    @staticmethod
    def create_order_cancel_replace_request(
        sender: str,
        target: str,
        seq_num: int,
        orig_cl_ord_id: str,
        cl_ord_id: str,
        symbol: str,
        side: FixSide,
        order_qty: Union[int, float, Decimal],
        order_type: FixOrdType,
        price: Union[float, Decimal] = None,
        order_id: str = None,
        version: FixVersion = FixVersion.FIX_4_4,
    ) -> FixMessage:
        """Create an Order Cancel/Replace Request message (MsgType=G)."""
        builder = FixBuilder(version)
        builder.set_msg_type(FixMsgType.ORDER_CANCEL_REPLACE_REQUEST)
        builder.set_sender(sender)
        builder.set_target(target)
        builder.set_seq_num(seq_num)
        builder.set_orig_cl_ord_id(orig_cl_ord_id)
        builder.set_cl_ord_id(cl_ord_id)
        builder.set_symbol(symbol)
        builder.set_side(side)
        builder.set_order_qty(order_qty)
        builder.set_order_type(order_type)
        builder.set_transact_time()

        if price is not None:
            builder.set_price(price)
        if order_id:
            builder.set_order_id(order_id)

        return builder.build()

    @staticmethod
    def create_market_data_request(
        sender: str,
        target: str,
        seq_num: int,
        md_req_id: str,
        subscription_type: int,  # 0=Snapshot, 1=Snapshot+Updates, 2=Unsubscribe
        market_depth: int,
        symbols: List[str],
        md_entry_types: List[str] = None,  # 0=Bid, 1=Offer, 2=Trade
        version: FixVersion = FixVersion.FIX_4_4,
    ) -> FixMessage:
        """Create a Market Data Request message (MsgType=V)."""
        builder = FixBuilder(version)
        builder.set_msg_type(FixMsgType.MARKET_DATA_REQUEST)
        builder.set_sender(sender)
        builder.set_target(target)
        builder.set_seq_num(seq_num)
        builder.set_field(262, md_req_id)  # MDReqID
        builder.set_field(263, subscription_type)  # SubscriptionRequestType
        builder.set_field(264, market_depth)  # MarketDepth

        # MD Entry Types group
        if md_entry_types is None:
            md_entry_types = ["0", "1"]  # Bid and Offer

        entry_types_group = FixGroup(count_tag=267)
        for entry_type in md_entry_types:
            entry_types_group.add_entry({269: entry_type})
        builder.add_group(entry_types_group)

        # Symbols group
        symbols_group = FixGroup(count_tag=146)
        for symbol in symbols:
            symbols_group.add_entry({55: symbol})
        builder.add_group(symbols_group)

        return builder.build()
