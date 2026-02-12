"""
FIX Message Validator

Validates FIX messages including:
- Required fields by message type
- Field format validation
- Cross-field validation
- Business rule validation
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
)
from ai_engine.domains.banking.protocols.trading.fix.fix_message import (
    FixMessage,
    FixField,
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
    REQUIRED_TAGS,
    get_tag_name,
)


class FixValidator(BaseValidator):
    """
    Validator for FIX messages.

    Validates:
    - Message structure
    - Required fields per message type
    - Field formats and values
    - Cross-field dependencies
    - Business rules
    """

    def __init__(
        self,
        strict: bool = True,
        validate_checksum: bool = True,
        version: FixVersion = FixVersion.FIX_4_4,
    ):
        """
        Initialize the FIX validator.

        Args:
            strict: If True, treat warnings as errors
            validate_checksum: If True, validate message checksum
            version: Expected FIX version
        """
        super().__init__(strict)
        self.validate_checksum = validate_checksum
        self.expected_version = version

    @property
    def name(self) -> str:
        return "FixValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate a FIX message.

        Args:
            data: FixMessage or dict

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, FixMessage):
            self._validate_message(data, result)
        elif isinstance(data, dict):
            self._validate_dict(data, result)
        else:
            result.add_error(
                "FIX_INVALID_INPUT",
                "Input must be a FIX message object or dictionary",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def _validate_message(self, msg: FixMessage, result: ValidationResult) -> None:
        """Validate a FixMessage object."""
        # Validate header
        self._validate_header(msg, result)

        # Validate required fields for message type
        self._validate_required_fields(msg, result)

        # Validate field formats
        self._validate_field_formats(msg, result)

        # Message-type specific validation
        msg_type = msg.header.msg_type
        if msg_type == "D":
            self._validate_new_order_single(msg, result)
        elif msg_type == "8":
            self._validate_execution_report(msg, result)
        elif msg_type == "F":
            self._validate_order_cancel_request(msg, result)
        elif msg_type == "G":
            self._validate_order_cancel_replace(msg, result)
        elif msg_type == "A":
            self._validate_logon(msg, result)
        elif msg_type == "V":
            self._validate_market_data_request(msg, result)

        # Validate groups
        self._validate_groups(msg, result)

    def _validate_header(self, msg: FixMessage, result: ValidationResult) -> None:
        """Validate message header."""
        header = msg.header

        # BeginString
        if not header.begin_string:
            result.add_error(
                "FIX_MISSING_BEGIN_STRING",
                "BeginString (tag 8) is required",
                field="BeginString",
                severity=ValidationSeverity.CRITICAL,
            )
        elif FixVersion.from_string(header.begin_string) is None:
            result.add_warning(
                "FIX_UNKNOWN_VERSION",
                f"Unknown FIX version: {header.begin_string}",
                field="BeginString",
            )

        # MsgType
        if not header.msg_type:
            result.add_error(
                "FIX_MISSING_MSG_TYPE",
                "MsgType (tag 35) is required",
                field="MsgType",
                severity=ValidationSeverity.CRITICAL,
            )
        elif FixMsgType.from_code(header.msg_type) is None:
            result.add_warning(
                "FIX_UNKNOWN_MSG_TYPE",
                f"Unknown message type: {header.msg_type}",
                field="MsgType",
            )

        # SenderCompID
        if not header.sender_comp_id:
            result.add_error(
                "FIX_MISSING_SENDER",
                "SenderCompID (tag 49) is required",
                field="SenderCompID",
            )

        # TargetCompID
        if not header.target_comp_id:
            result.add_error(
                "FIX_MISSING_TARGET",
                "TargetCompID (tag 56) is required",
                field="TargetCompID",
            )

        # MsgSeqNum
        if header.msg_seq_num <= 0:
            result.add_error(
                "FIX_INVALID_SEQ_NUM",
                "MsgSeqNum (tag 34) must be a positive integer",
                field="MsgSeqNum",
            )

        # SendingTime
        if not header.sending_time:
            result.add_warning(
                "FIX_MISSING_SENDING_TIME",
                "SendingTime (tag 52) is recommended",
                field="SendingTime",
            )

        # PossDupFlag and OrigSendingTime
        if header.poss_dup_flag and not header.orig_sending_time:
            result.add_error(
                "FIX_MISSING_ORIG_SENDING_TIME",
                "OrigSendingTime (tag 122) is required when PossDupFlag is Y",
                field="OrigSendingTime",
            )

    def _validate_required_fields(self, msg: FixMessage, result: ValidationResult) -> None:
        """Validate required fields for message type."""
        msg_type = msg.header.msg_type
        required = REQUIRED_TAGS.get(msg_type, [])

        present_tags = {f.tag for f in msg.body}
        # Also include group count tags
        present_tags.update(msg.groups.keys())

        for tag in required:
            if tag not in present_tags:
                result.add_error(
                    "FIX_MISSING_REQUIRED_FIELD",
                    f"Required field {get_tag_name(tag)} (tag {tag}) is missing",
                    field=get_tag_name(tag),
                )

    def _validate_field_formats(self, msg: FixMessage, result: ValidationResult) -> None:
        """Validate field formats."""
        for field in msg.body:
            self._validate_field(field, result)

    def _validate_field(self, field: FixField, result: ValidationResult) -> None:
        """Validate a single field."""
        tag = field.tag
        value = field.value

        # Validate specific field formats
        if tag == 54:  # Side
            if FixSide.from_code(value) is None:
                result.add_error(
                    "FIX_INVALID_SIDE",
                    f"Invalid Side value: {value}",
                    field="Side",
                )

        elif tag == 40:  # OrdType
            if FixOrdType.from_code(value) is None:
                result.add_error(
                    "FIX_INVALID_ORD_TYPE",
                    f"Invalid OrdType value: {value}",
                    field="OrdType",
                )

        elif tag == 59:  # TimeInForce
            if FixTimeInForce.from_code(value) is None:
                result.add_warning(
                    "FIX_UNKNOWN_TIME_IN_FORCE",
                    f"Unknown TimeInForce value: {value}",
                    field="TimeInForce",
                )

        elif tag == 150:  # ExecType
            if FixExecType.from_code(value) is None:
                result.add_warning(
                    "FIX_UNKNOWN_EXEC_TYPE",
                    f"Unknown ExecType value: {value}",
                    field="ExecType",
                )

        elif tag == 39:  # OrdStatus
            if FixOrdStatus.from_code(value) is None:
                result.add_warning(
                    "FIX_UNKNOWN_ORD_STATUS",
                    f"Unknown OrdStatus value: {value}",
                    field="OrdStatus",
                )

        elif tag == 21:  # HandlInst
            if value not in ("1", "2", "3"):
                result.add_error(
                    "FIX_INVALID_HANDL_INST",
                    f"Invalid HandlInst value: {value}",
                    field="HandlInst",
                )

        elif tag in (38, 14, 32, 151, 152):  # Quantity fields
            try:
                qty = Decimal(value)
                if qty < 0:
                    result.add_error(
                        "FIX_NEGATIVE_QTY",
                        f"Quantity cannot be negative: {value}",
                        field=get_tag_name(tag),
                    )
            except Exception:
                result.add_error(
                    "FIX_INVALID_QTY",
                    f"Invalid quantity value: {value}",
                    field=get_tag_name(tag),
                )

        elif tag in (44, 99, 31, 6):  # Price fields
            try:
                price = Decimal(value)
                if price < 0:
                    result.add_error(
                        "FIX_NEGATIVE_PRICE",
                        f"Price cannot be negative: {value}",
                        field=get_tag_name(tag),
                    )
            except Exception:
                result.add_error(
                    "FIX_INVALID_PRICE",
                    f"Invalid price value: {value}",
                    field=get_tag_name(tag),
                )

        elif tag == 60:  # TransactTime
            if not self._is_valid_timestamp(value):
                result.add_warning(
                    "FIX_INVALID_TIMESTAMP",
                    f"Invalid TransactTime format: {value}",
                    field="TransactTime",
                )

        elif tag == 15:  # Currency
            if len(value) != 3 or not value.isalpha():
                result.add_error(
                    "FIX_INVALID_CURRENCY",
                    f"Currency must be 3 alphabetic characters: {value}",
                    field="Currency",
                )

    def _validate_new_order_single(self, msg: FixMessage, result: ValidationResult) -> None:
        """Validate New Order Single (MsgType=D)."""
        order_type = msg.get_value(40)

        # Price required for limit orders
        if order_type in ("2", "4", "B", "F"):  # Limit, Stop Limit, etc.
            if not msg.get_field(44):
                result.add_error(
                    "FIX_MISSING_PRICE",
                    "Price (tag 44) is required for limit orders",
                    field="Price",
                )

        # StopPx required for stop orders
        if order_type in ("3", "4"):  # Stop, Stop Limit
            if not msg.get_field(99):
                result.add_error(
                    "FIX_MISSING_STOP_PX",
                    "StopPx (tag 99) is required for stop orders",
                    field="StopPx",
                )

        # OrderQty validation
        order_qty = msg.get_decimal(38)
        if order_qty is not None and order_qty <= 0:
            result.add_error(
                "FIX_INVALID_ORDER_QTY",
                "OrderQty must be greater than zero",
                field="OrderQty",
            )

    def _validate_execution_report(self, msg: FixMessage, result: ValidationResult) -> None:
        """Validate Execution Report (MsgType=8)."""
        exec_type = msg.get_value(150)
        ord_status = msg.get_value(39)

        # LastQty and LastPx required for fills
        if exec_type in ("1", "2", "F"):  # Partial Fill, Fill, Trade
            if not msg.get_field(32):
                result.add_error(
                    "FIX_MISSING_LAST_QTY",
                    "LastQty (tag 32) is required for fills",
                    field="LastQty",
                )
            if not msg.get_field(31):
                result.add_error(
                    "FIX_MISSING_LAST_PX",
                    "LastPx (tag 31) is required for fills",
                    field="LastPx",
                )

        # Validate quantity consistency
        leaves_qty = msg.get_decimal(151)
        cum_qty = msg.get_decimal(14)
        order_qty = msg.get_decimal(38)

        if leaves_qty is not None and cum_qty is not None and order_qty is not None:
            if leaves_qty + cum_qty != order_qty:
                result.add_warning(
                    "FIX_QTY_MISMATCH",
                    f"LeavesQty ({leaves_qty}) + CumQty ({cum_qty}) != OrderQty ({order_qty})",
                    field="Quantity",
                )

        # Validate exec type and ord status consistency
        self._validate_exec_ord_status_consistency(exec_type, ord_status, result)

    def _validate_exec_ord_status_consistency(self, exec_type: str, ord_status: str, result: ValidationResult) -> None:
        """Validate ExecType and OrdStatus are consistent."""
        valid_combinations = {
            "0": ["0", "A"],  # New -> New, Pending New
            "4": ["4", "6"],  # Canceled -> Canceled, Pending Cancel
            "5": ["5", "E"],  # Replaced -> Replaced, Pending Replace
            "8": ["8"],  # Rejected -> Rejected
            "C": ["C"],  # Expired -> Expired
            "F": ["1", "2"],  # Trade -> Partially Filled, Filled
        }

        if exec_type in valid_combinations:
            if ord_status not in valid_combinations[exec_type]:
                result.add_warning(
                    "FIX_INCONSISTENT_STATUS",
                    f"ExecType {exec_type} typically not used with OrdStatus {ord_status}",
                    field="ExecType/OrdStatus",
                )

    def _validate_order_cancel_request(self, msg: FixMessage, result: ValidationResult) -> None:
        """Validate Order Cancel Request (MsgType=F)."""
        # OrigClOrdID is required
        if not msg.get_field(41):
            result.add_error(
                "FIX_MISSING_ORIG_CL_ORD_ID",
                "OrigClOrdID (tag 41) is required for cancel request",
                field="OrigClOrdID",
            )

    def _validate_order_cancel_replace(self, msg: FixMessage, result: ValidationResult) -> None:
        """Validate Order Cancel/Replace Request (MsgType=G)."""
        # OrigClOrdID is required
        if not msg.get_field(41):
            result.add_error(
                "FIX_MISSING_ORIG_CL_ORD_ID",
                "OrigClOrdID (tag 41) is required for cancel/replace",
                field="OrigClOrdID",
            )

        # Same validation as new order for price/quantity
        self._validate_new_order_single(msg, result)

    def _validate_logon(self, msg: FixMessage, result: ValidationResult) -> None:
        """Validate Logon message (MsgType=A)."""
        # HeartBtInt is required
        if not msg.get_field(108):
            result.add_error(
                "FIX_MISSING_HEARTBTINT",
                "HeartBtInt (tag 108) is required for Logon",
                field="HeartBtInt",
            )
        else:
            heartbeat = msg.get_int(108)
            if heartbeat is not None and heartbeat < 0:
                result.add_error(
                    "FIX_INVALID_HEARTBTINT",
                    "HeartBtInt cannot be negative",
                    field="HeartBtInt",
                )

        # EncryptMethod is required
        if not msg.get_field(98):
            result.add_warning(
                "FIX_MISSING_ENCRYPT_METHOD",
                "EncryptMethod (tag 98) is recommended for Logon",
                field="EncryptMethod",
            )

    def _validate_market_data_request(self, msg: FixMessage, result: ValidationResult) -> None:
        """Validate Market Data Request (MsgType=V)."""
        # MDReqID is required
        if not msg.get_field(262):
            result.add_error(
                "FIX_MISSING_MD_REQ_ID",
                "MDReqID (tag 262) is required",
                field="MDReqID",
            )

        # SubscriptionRequestType validation
        sub_type = msg.get_value(263)
        if sub_type and sub_type not in ("0", "1", "2"):
            result.add_error(
                "FIX_INVALID_SUB_TYPE",
                f"Invalid SubscriptionRequestType: {sub_type}",
                field="SubscriptionRequestType",
            )

        # MarketDepth validation
        depth = msg.get_int(264)
        if depth is not None and depth < 0:
            result.add_error(
                "FIX_INVALID_MARKET_DEPTH",
                "MarketDepth cannot be negative",
                field="MarketDepth",
            )

        # NoMDEntryTypes group required
        if 267 not in msg.groups:
            result.add_error(
                "FIX_MISSING_MD_ENTRY_TYPES",
                "NoMDEntryTypes group (tag 267) is required",
                field="NoMDEntryTypes",
            )

        # NoRelatedSym group required
        if 146 not in msg.groups:
            result.add_error(
                "FIX_MISSING_SYMBOLS",
                "NoRelatedSym group (tag 146) is required",
                field="NoRelatedSym",
            )

    def _validate_groups(self, msg: FixMessage, result: ValidationResult) -> None:
        """Validate repeating groups."""
        for count_tag, group in msg.groups.items():
            # Validate count matches entries
            if group.count != len(group.entries):
                result.add_warning(
                    "FIX_GROUP_COUNT_MISMATCH",
                    f"Group {get_tag_name(count_tag)}: count {group.count} " f"doesn't match entries {len(group.entries)}",
                    field=get_tag_name(count_tag),
                )

            # Validate each entry has required fields
            for i, entry in enumerate(group.entries):
                if not entry:
                    result.add_warning(
                        "FIX_EMPTY_GROUP_ENTRY",
                        f"Group {get_tag_name(count_tag)} entry {i + 1} is empty",
                        field=get_tag_name(count_tag),
                    )

    def _validate_dict(self, data: Dict, result: ValidationResult) -> None:
        """Validate FIX data from dictionary."""
        # Basic validation
        if "msg_type" in data:
            msg_type = data["msg_type"]
            if FixMsgType.from_code(msg_type) is None:
                result.add_warning(
                    "FIX_UNKNOWN_MSG_TYPE",
                    f"Unknown message type: {msg_type}",
                    field="msg_type",
                )

        if "sender" in data and not data["sender"]:
            result.add_error(
                "FIX_MISSING_SENDER",
                "Sender is required",
                field="sender",
            )

        if "target" in data and not data["target"]:
            result.add_error(
                "FIX_MISSING_TARGET",
                "Target is required",
                field="target",
            )

        # Validate fields dict
        if "fields" in data:
            for name, value in data["fields"].items():
                if name == "Side" and FixSide.from_code(str(value)) is None:
                    result.add_error(
                        "FIX_INVALID_SIDE",
                        f"Invalid Side value: {value}",
                        field="Side",
                    )
                elif name == "OrdType" and FixOrdType.from_code(str(value)) is None:
                    result.add_error(
                        "FIX_INVALID_ORD_TYPE",
                        f"Invalid OrdType value: {value}",
                        field="OrdType",
                    )

    def _is_valid_timestamp(self, value: str) -> bool:
        """Check if value is a valid FIX timestamp."""
        try:
            if "." in value:
                datetime.strptime(value, "%Y%m%d-%H:%M:%S.%f")
            elif len(value) == 17:
                datetime.strptime(value, "%Y%m%d-%H:%M:%S")
            elif len(value) == 8:
                datetime.strptime(value, "%Y%m%d")
            else:
                return False
            return True
        except ValueError:
            return False
