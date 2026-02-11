"""
FIX Protocol Implementation

Financial Information eXchange (FIX) Protocol support for:
- FIX 4.2, 4.4, 5.0, and FIXT 1.1
- Order management messages (New Order Single, Execution Report, etc.)
- Market data messages
- Trade capture and allocation
- Session management

This implementation focuses on the most commonly used FIX messages
in equity and fixed income trading.
"""

from ai_engine.domains.banking.protocols.trading.fix.fix_codes import (
    FixVersion,
    FixMsgType,
    FixOrdType,
    FixSide,
    FixTimeInForce,
    FixExecType,
    FixOrdStatus,
    FixExecTransType,
    FixHandlInst,
    FixSecurityType,
    FixSettlType,
    FIX_TAG_NAMES,
    get_tag_name,
)
from ai_engine.domains.banking.protocols.trading.fix.fix_message import (
    FixField,
    FixGroup,
    FixMessage,
    FixHeader,
    FixTrailer,
)
from ai_engine.domains.banking.protocols.trading.fix.fix_parser import (
    FixParser,
    FixParseError,
    parse_fix_message,
)
from ai_engine.domains.banking.protocols.trading.fix.fix_builder import (
    FixBuilder,
    FixMessageFactory,
)
from ai_engine.domains.banking.protocols.trading.fix.fix_validator import (
    FixValidator,
)

__all__ = [
    # Codes and enums
    "FixVersion",
    "FixMsgType",
    "FixOrdType",
    "FixSide",
    "FixTimeInForce",
    "FixExecType",
    "FixOrdStatus",
    "FixExecTransType",
    "FixHandlInst",
    "FixSecurityType",
    "FixSettlType",
    "FIX_TAG_NAMES",
    "get_tag_name",
    # Message structures
    "FixField",
    "FixGroup",
    "FixMessage",
    "FixHeader",
    "FixTrailer",
    # Parser
    "FixParser",
    "FixParseError",
    "parse_fix_message",
    # Builder
    "FixBuilder",
    "FixMessageFactory",
    # Validator
    "FixValidator",
]
