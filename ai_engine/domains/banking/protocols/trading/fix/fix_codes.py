"""
FIX Protocol Codes and Constants

Defines:
- FIX versions
- Message types
- Order types and sides
- Execution and order status codes
- Tag definitions
"""

from enum import Enum
from typing import Dict, Optional


class FixVersion(Enum):
    """FIX Protocol versions."""

    FIX_4_0 = ("FIX.4.0", "4.0")
    FIX_4_1 = ("FIX.4.1", "4.1")
    FIX_4_2 = ("FIX.4.2", "4.2")
    FIX_4_3 = ("FIX.4.3", "4.3")
    FIX_4_4 = ("FIX.4.4", "4.4")
    FIX_5_0 = ("FIX.5.0", "5.0")
    FIX_5_0_SP1 = ("FIX.5.0SP1", "5.0 SP1")
    FIX_5_0_SP2 = ("FIX.5.0SP2", "5.0 SP2")
    FIXT_1_1 = ("FIXT.1.1", "FIXT 1.1")

    def __init__(self, begin_string: str, description: str):
        self._begin_string = begin_string
        self._description = description

    @property
    def begin_string(self) -> str:
        return self._begin_string

    @property
    def description(self) -> str:
        return self._description

    @classmethod
    def from_string(cls, value: str) -> Optional["FixVersion"]:
        """Get version from BeginString value."""
        for version in cls:
            if version.begin_string == value:
                return version
        return None


class FixMsgType(Enum):
    """FIX Message Types (tag 35)."""

    # Session messages
    HEARTBEAT = ("0", "Heartbeat")
    TEST_REQUEST = ("1", "Test Request")
    RESEND_REQUEST = ("2", "Resend Request")
    REJECT = ("3", "Reject")
    SEQUENCE_RESET = ("4", "Sequence Reset")
    LOGOUT = ("5", "Logout")
    LOGON = ("A", "Logon")

    # Application messages - Orders
    NEW_ORDER_SINGLE = ("D", "New Order Single")
    ORDER_CANCEL_REQUEST = ("F", "Order Cancel Request")
    ORDER_CANCEL_REPLACE_REQUEST = ("G", "Order Cancel/Replace Request")
    ORDER_STATUS_REQUEST = ("H", "Order Status Request")
    EXECUTION_REPORT = ("8", "Execution Report")
    ORDER_CANCEL_REJECT = ("9", "Order Cancel Reject")

    # Application messages - Market Data
    MARKET_DATA_REQUEST = ("V", "Market Data Request")
    MARKET_DATA_SNAPSHOT = ("W", "Market Data Snapshot/Full Refresh")
    MARKET_DATA_INCREMENTAL = ("X", "Market Data Incremental Refresh")
    MARKET_DATA_REQUEST_REJECT = ("Y", "Market Data Request Reject")

    # Application messages - Quotes
    QUOTE_REQUEST = ("R", "Quote Request")
    QUOTE = ("S", "Quote")
    QUOTE_CANCEL = ("Z", "Quote Cancel")
    QUOTE_STATUS_REQUEST = ("a", "Quote Status Request")
    QUOTE_ACKNOWLEDGEMENT = ("b", "Quote Acknowledgement")
    QUOTE_STATUS_REPORT = ("AI", "Quote Status Report")

    # Application messages - Trade Capture
    TRADE_CAPTURE_REPORT_REQUEST = ("AD", "Trade Capture Report Request")
    TRADE_CAPTURE_REPORT = ("AE", "Trade Capture Report")
    TRADE_CAPTURE_REPORT_ACK = ("AR", "Trade Capture Report Ack")

    # Application messages - Allocation
    ALLOCATION_INSTRUCTION = ("J", "Allocation Instruction")
    ALLOCATION_INSTRUCTION_ACK = ("P", "Allocation Instruction Ack")
    ALLOCATION_REPORT = ("AS", "Allocation Report")
    ALLOCATION_REPORT_ACK = ("AT", "Allocation Report Ack")

    # Application messages - Confirmations
    CONFIRMATION = ("AK", "Confirmation")
    CONFIRMATION_ACK = ("AU", "Confirmation Ack")
    CONFIRMATION_REQUEST = ("BH", "Confirmation Request")

    # Application messages - Positions
    POSITION_REPORT = ("AP", "Position Report")
    POSITION_MAINTENANCE_REQUEST = ("AL", "Position Maintenance Request")
    POSITION_MAINTENANCE_REPORT = ("AM", "Position Maintenance Report")
    REQUEST_FOR_POSITIONS = ("AN", "Request For Positions")
    REQUEST_FOR_POSITIONS_ACK = ("AO", "Request For Positions Ack")

    # Application messages - Security
    SECURITY_LIST_REQUEST = ("x", "Security List Request")
    SECURITY_LIST = ("y", "Security List")
    SECURITY_DEFINITION_REQUEST = ("c", "Security Definition Request")
    SECURITY_DEFINITION = ("d", "Security Definition")
    SECURITY_STATUS_REQUEST = ("e", "Security Status Request")
    SECURITY_STATUS = ("f", "Security Status")

    # Application messages - Business
    BUSINESS_MESSAGE_REJECT = ("j", "Business Message Reject")
    NEWS = ("B", "News")
    EMAIL = ("C", "Email")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description

    @classmethod
    def from_code(cls, code: str) -> Optional["FixMsgType"]:
        """Get message type from code."""
        for msg_type in cls:
            if msg_type.code == code:
                return msg_type
        return None


class FixOrdType(Enum):
    """Order Types (tag 40)."""

    MARKET = ("1", "Market")
    LIMIT = ("2", "Limit")
    STOP = ("3", "Stop")
    STOP_LIMIT = ("4", "Stop Limit")
    MARKET_ON_CLOSE = ("5", "Market On Close")
    WITH_OR_WITHOUT = ("6", "With Or Without")
    LIMIT_OR_BETTER = ("7", "Limit Or Better")
    LIMIT_WITH_OR_WITHOUT = ("8", "Limit With Or Without")
    ON_BASIS = ("9", "On Basis")
    ON_CLOSE = ("A", "On Close")
    LIMIT_ON_CLOSE = ("B", "Limit On Close")
    FOREX_MARKET = ("C", "Forex Market")
    PREVIOUSLY_QUOTED = ("D", "Previously Quoted")
    PREVIOUSLY_INDICATED = ("E", "Previously Indicated")
    FOREX_LIMIT = ("F", "Forex Limit")
    FOREX_SWAP = ("G", "Forex Swap")
    FOREX_PREVIOUSLY_QUOTED = ("H", "Forex Previously Quoted")
    FUNARI = ("I", "Funari")
    MARKET_IF_TOUCHED = ("J", "Market If Touched")
    MARKET_WITH_LEFTOVER_AS_LIMIT = ("K", "Market With Leftover As Limit")
    PREVIOUS_FUND_VALUATION_POINT = ("L", "Previous Fund Valuation Point")
    NEXT_FUND_VALUATION_POINT = ("M", "Next Fund Valuation Point")
    PEGGED = ("P", "Pegged")
    COUNTER_ORDER_SELECTION = ("Q", "Counter Order Selection")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description

    @classmethod
    def from_code(cls, code: str) -> Optional["FixOrdType"]:
        """Get order type from code."""
        for ord_type in cls:
            if ord_type.code == code:
                return ord_type
        return None


class FixSide(Enum):
    """Order Sides (tag 54)."""

    BUY = ("1", "Buy")
    SELL = ("2", "Sell")
    BUY_MINUS = ("3", "Buy Minus")
    SELL_PLUS = ("4", "Sell Plus")
    SELL_SHORT = ("5", "Sell Short")
    SELL_SHORT_EXEMPT = ("6", "Sell Short Exempt")
    UNDISCLOSED = ("7", "Undisclosed")
    CROSS = ("8", "Cross")
    CROSS_SHORT = ("9", "Cross Short")
    CROSS_SHORT_EXEMPT = ("A", "Cross Short Exempt")
    AS_DEFINED = ("B", "As Defined")
    OPPOSITE = ("C", "Opposite")
    SUBSCRIBE = ("D", "Subscribe")
    REDEEM = ("E", "Redeem")
    LEND = ("F", "Lend")
    BORROW = ("G", "Borrow")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description

    @classmethod
    def from_code(cls, code: str) -> Optional["FixSide"]:
        """Get side from code."""
        for side in cls:
            if side.code == code:
                return side
        return None


class FixTimeInForce(Enum):
    """Time In Force (tag 59)."""

    DAY = ("0", "Day")
    GOOD_TILL_CANCEL = ("1", "Good Till Cancel (GTC)")
    AT_THE_OPENING = ("2", "At The Opening (OPG)")
    IMMEDIATE_OR_CANCEL = ("3", "Immediate Or Cancel (IOC)")
    FILL_OR_KILL = ("4", "Fill Or Kill (FOK)")
    GOOD_TILL_CROSSING = ("5", "Good Till Crossing (GTX)")
    GOOD_TILL_DATE = ("6", "Good Till Date (GTD)")
    AT_THE_CLOSE = ("7", "At The Close")
    GOOD_THROUGH_CROSSING = ("8", "Good Through Crossing")
    AT_CROSSING = ("9", "At Crossing")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description

    @classmethod
    def from_code(cls, code: str) -> Optional["FixTimeInForce"]:
        """Get time in force from code."""
        for tif in cls:
            if tif.code == code:
                return tif
        return None


class FixExecType(Enum):
    """Execution Types (tag 150)."""

    NEW = ("0", "New")
    PARTIAL_FILL = ("1", "Partial Fill")  # Deprecated in FIX 4.3+
    FILL = ("2", "Fill")  # Deprecated in FIX 4.3+
    DONE_FOR_DAY = ("3", "Done For Day")
    CANCELED = ("4", "Canceled")
    REPLACED = ("5", "Replaced")
    PENDING_CANCEL = ("6", "Pending Cancel")
    STOPPED = ("7", "Stopped")
    REJECTED = ("8", "Rejected")
    SUSPENDED = ("9", "Suspended")
    PENDING_NEW = ("A", "Pending New")
    CALCULATED = ("B", "Calculated")
    EXPIRED = ("C", "Expired")
    RESTATED = ("D", "Restated")
    PENDING_REPLACE = ("E", "Pending Replace")
    TRADE = ("F", "Trade (Partial Fill or Fill)")
    TRADE_CORRECT = ("G", "Trade Correct")
    TRADE_CANCEL = ("H", "Trade Cancel")
    ORDER_STATUS = ("I", "Order Status")
    TRADE_IN_A_CLEARING_HOLD = ("J", "Trade In A Clearing Hold")
    TRADE_HAS_BEEN_RELEASED_TO_CLEARING = ("K", "Trade Has Been Released To Clearing")
    TRIGGERED_OR_ACTIVATED_BY_SYSTEM = ("L", "Triggered Or Activated By System")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description

    @classmethod
    def from_code(cls, code: str) -> Optional["FixExecType"]:
        """Get execution type from code."""
        for exec_type in cls:
            if exec_type.code == code:
                return exec_type
        return None


class FixOrdStatus(Enum):
    """Order Status (tag 39)."""

    NEW = ("0", "New")
    PARTIALLY_FILLED = ("1", "Partially Filled")
    FILLED = ("2", "Filled")
    DONE_FOR_DAY = ("3", "Done For Day")
    CANCELED = ("4", "Canceled")
    REPLACED = ("5", "Replaced")  # Deprecated
    PENDING_CANCEL = ("6", "Pending Cancel")
    STOPPED = ("7", "Stopped")
    REJECTED = ("8", "Rejected")
    SUSPENDED = ("9", "Suspended")
    PENDING_NEW = ("A", "Pending New")
    CALCULATED = ("B", "Calculated")
    EXPIRED = ("C", "Expired")
    ACCEPTED_FOR_BIDDING = ("D", "Accepted For Bidding")
    PENDING_REPLACE = ("E", "Pending Replace")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description

    @classmethod
    def from_code(cls, code: str) -> Optional["FixOrdStatus"]:
        """Get order status from code."""
        for status in cls:
            if status.code == code:
                return status
        return None


class FixExecTransType(Enum):
    """Execution Transaction Type (tag 20, deprecated in FIX 4.3+)."""

    NEW = ("0", "New")
    CANCEL = ("1", "Cancel")
    CORRECT = ("2", "Correct")
    STATUS = ("3", "Status")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description


class FixHandlInst(Enum):
    """Handling Instructions (tag 21)."""

    AUTOMATED_NO_INTERVENTION = ("1", "Automated execution order, private")
    AUTOMATED_WITH_INTERVENTION = ("2", "Automated execution order, public")
    MANUAL = ("3", "Manual order, best execution")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description


class FixSecurityType(Enum):
    """Security Types (tag 167)."""

    # Equity
    COMMON_STOCK = ("CS", "Common Stock")
    PREFERRED_STOCK = ("PS", "Preferred Stock")
    DEPOSITORY_RECEIPT = ("DR", "Depository Receipt")
    WARRANT = ("WAR", "Warrant")
    RIGHTS = ("RIGHTS", "Rights")
    UNIT = ("UNIT", "Unit")
    MUTUAL_FUND = ("MF", "Mutual Fund")
    ETF = ("ETF", "Exchange Traded Fund")

    # Fixed Income
    CORPORATE_BOND = ("CORP", "Corporate Bond")
    CONVERTIBLE_BOND = ("CB", "Convertible Bond")
    MUNICIPAL_BOND = ("MUNI", "Municipal Bond")
    GOVERNMENT_BOND = ("GOVT", "Government Bond")
    TREASURY_BILL = ("TB", "Treasury Bill")
    TREASURY_NOTE = ("TNOTE", "Treasury Note")
    TREASURY_BOND = ("TBOND", "Treasury Bond")
    AGENCY_BOND = ("AGNCY", "Agency Bond")
    MORTGAGE_BACKED = ("MBS", "Mortgage Backed Security")
    ASSET_BACKED = ("ABS", "Asset Backed Security")
    COMMERCIAL_PAPER = ("CP", "Commercial Paper")

    # Derivatives
    FUTURE = ("FUT", "Future")
    OPTION = ("OPT", "Option")
    INDEX_OPTION = ("IOPT", "Index Option")
    CURRENCY_OPTION = ("CUROPT", "Currency Option")
    FORWARD = ("FOR", "Forward")
    SWAP = ("SWAP", "Swap")

    # FX
    SPOT = ("SPOT", "FX Spot")
    FX_FORWARD = ("FXFWD", "FX Forward")
    FX_SWAP = ("FXSWAP", "FX Swap")

    # Other
    REPO = ("REPO", "Repurchase Agreement")
    INDEX = ("INDEX", "Index")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description

    @classmethod
    def from_code(cls, code: str) -> Optional["FixSecurityType"]:
        """Get security type from code."""
        for sec_type in cls:
            if sec_type.code == code:
                return sec_type
        return None


class FixSettlType(Enum):
    """Settlement Types (tag 63)."""

    REGULAR = ("0", "Regular / FX Spot settlement (T+2)")
    CASH = ("1", "Cash (T+0)")
    NEXT_DAY = ("2", "Next Day (T+1)")
    T_PLUS_2 = ("3", "T+2")
    T_PLUS_3 = ("4", "T+3")
    T_PLUS_4 = ("5", "T+4")
    FUTURE = ("6", "Future")
    WHEN_AND_IF_ISSUED = ("7", "When And If Issued")
    SELLERS_OPTION = ("8", "Sellers Option")
    T_PLUS_5 = ("9", "T+5")
    BROKEN_DATE = ("B", "Broken Date")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description


# FIX Tag definitions
FIX_TAG_NAMES: Dict[int, str] = {
    # Standard Header
    8: "BeginString",
    9: "BodyLength",
    35: "MsgType",
    49: "SenderCompID",
    56: "TargetCompID",
    34: "MsgSeqNum",
    52: "SendingTime",
    43: "PossDupFlag",
    97: "PossResend",
    122: "OrigSendingTime",
    # Standard Trailer
    10: "CheckSum",
    # Order fields
    11: "ClOrdID",
    14: "CumQty",
    15: "Currency",
    17: "ExecID",
    18: "ExecInst",
    19: "ExecRefID",
    20: "ExecTransType",
    21: "HandlInst",
    22: "SecurityIDSource",
    23: "IOIID",
    31: "LastPx",
    32: "LastQty",
    37: "OrderID",
    38: "OrderQty",
    39: "OrdStatus",
    40: "OrdType",
    41: "OrigClOrdID",
    44: "Price",
    45: "RefSeqNum",
    48: "SecurityID",
    54: "Side",
    55: "Symbol",
    58: "Text",
    59: "TimeInForce",
    60: "TransactTime",
    63: "SettlType",
    64: "SettlDate",
    65: "SymbolSfx",
    75: "TradeDate",
    76: "ExecBroker",
    # Execution fields
    6: "AvgPx",
    99: "StopPx",
    103: "OrdRejReason",
    150: "ExecType",
    151: "LeavesQty",
    # Party fields
    109: "ClientID",
    115: "OnBehalfOfCompID",
    116: "OnBehalfOfSubID",
    128: "DeliverToCompID",
    129: "DeliverToSubID",
    142: "SenderLocationID",
    143: "TargetLocationID",
    144: "OnBehalfOfLocationID",
    145: "DeliverToLocationID",
    # Security fields
    106: "Issuer",
    107: "SecurityDesc",
    167: "SecurityType",
    200: "MaturityMonthYear",
    201: "PutOrCall",
    202: "StrikePrice",
    205: "MaturityDay",
    206: "OptAttribute",
    207: "SecurityExchange",
    223: "CouponRate",
    225: "IssueDate",
    226: "RepurchaseTerm",
    227: "RepurchaseRate",
    228: "Factor",
    231: "ContractMultiplier",
    255: "CreditRating",
    541: "MaturityDate",
    # Quantity fields
    53: "Quantity",
    110: "MinQty",
    111: "MaxFloor",
    152: "CashOrderQty",
    # Price fields
    140: "PrevClosePx",
    423: "PriceType",
    # Commission
    12: "Commission",
    13: "CommType",
    # Allocation
    70: "AllocID",
    71: "AllocTransType",
    72: "RefAllocID",
    73: "NoOrders",
    78: "NoAllocs",
    79: "AllocAccount",
    80: "AllocQty",
    81: "ProcessCode",
    366: "AllocPrice",
    # Trade Capture
    568: "TradeRequestID",
    569: "TradeRequestType",
    570: "PreviouslyReported",
    571: "TradeReportID",
    572: "TradeReportRefID",
    573: "MatchStatus",
    574: "MatchType",
    575: "OddLot",
    576: "NoClearingInstructions",
    577: "ClearingInstruction",
    578: "TradeInputSource",
    579: "TradeInputDevice",
    580: "NoDates",
    581: "AccountType",
    635: "ClearingFeeIndicator",
    # Parties
    448: "PartyID",
    447: "PartyIDSource",
    452: "PartyRole",
    453: "NoPartyIDs",
    802: "NoPartySubIDs",
    523: "PartySubID",
    803: "PartySubIDType",
    # Position
    702: "NoPositions",
    703: "PosType",
    704: "LongQty",
    705: "ShortQty",
    706: "PosQtyStatus",
    # Market Data
    262: "MDReqID",
    263: "SubscriptionRequestType",
    264: "MarketDepth",
    265: "MDUpdateType",
    266: "AggregatedBook",
    267: "NoMDEntryTypes",
    268: "NoMDEntries",
    269: "MDEntryType",
    270: "MDEntryPx",
    271: "MDEntrySize",
    272: "MDEntryDate",
    273: "MDEntryTime",
    274: "TickDirection",
    275: "MDMkt",
    276: "QuoteCondition",
    277: "TradeCondition",
    278: "MDEntryID",
    279: "MDUpdateAction",
    280: "MDEntryRefID",
    281: "MDReqRejReason",
    282: "MDEntryOriginator",
    283: "LocationID",
    284: "DeskID",
    285: "DeleteReason",
    286: "OpenCloseSettlFlag",
    287: "SellerDays",
    288: "MDEntryBuyer",
    289: "MDEntrySeller",
    290: "MDEntryPositionNo",
    291: "FinancialStatus",
    292: "CorporateAction",
    293: "DefBidSize",
    294: "DefOfferSize",
    295: "NoQuoteEntries",
    296: "NoQuoteSets",
}


def get_tag_name(tag: int) -> str:
    """Get the name of a FIX tag."""
    return FIX_TAG_NAMES.get(tag, f"Tag{tag}")


# Required tags by message type
REQUIRED_TAGS: Dict[str, list] = {
    "D": [11, 21, 55, 54, 60, 38, 40],  # New Order Single
    "8": [37, 17, 20, 150, 39, 55, 54, 151, 14, 6],  # Execution Report
    "F": [11, 41, 55, 54, 60],  # Order Cancel Request
    "G": [11, 41, 55, 54, 60, 38, 40],  # Order Cancel/Replace
    "9": [37, 11, 41, 39, 434, 102],  # Order Cancel Reject
    "V": [262, 263, 264, 267, 146],  # Market Data Request
    "W": [262],  # Market Data Snapshot
    "X": [262],  # Market Data Incremental
    "Y": [262, 281],  # Market Data Request Reject
}
