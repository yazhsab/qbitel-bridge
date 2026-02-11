"""
SWIFT MT Message Protocol Support

Implementation of SWIFT MT (Message Type) messages for:
- MT103: Single Customer Credit Transfer
- MT202: General Financial Institution Transfer
- MT199/299: Free Format Messages
- MT900/910: Confirmation of Debit/Credit
- MT940/942: Customer Statement Messages

Supports:
- Message parsing and validation
- Field extraction and formatting
- Character set validation (SWIFT-X)
- Block structure handling (1-5)
"""

from ai_engine.domains.banking.protocols.messaging.swift.swift_codes import (
    SwiftMessageType,
    SwiftBlockType,
    SwiftFieldTag,
    SwiftCharacterSet,
    SWIFT_MESSAGE_CATEGORIES,
    get_message_category,
    get_field_definition,
)
from ai_engine.domains.banking.protocols.messaging.swift.swift_message import (
    SwiftBlock,
    SwiftBasicHeader,
    SwiftApplicationHeader,
    SwiftUserHeader,
    SwiftTextBlock,
    SwiftTrailerBlock,
    SwiftMessage,
    SwiftField,
)
from ai_engine.domains.banking.protocols.messaging.swift.mt103 import (
    MT103Message,
    MT103Field,
    MT103Parser,
    MT103Builder,
    MT103Party,
    MT103Charges,
)
from ai_engine.domains.banking.protocols.messaging.swift.mt202 import (
    MT202Message,
    MT202Field,
    MT202Parser,
    MT202Builder,
    MT202Agent,
)
from ai_engine.domains.banking.protocols.messaging.swift.swift_parser import (
    SwiftParser,
    SwiftParseError,
)
from ai_engine.domains.banking.protocols.messaging.swift.swift_validator import (
    SwiftValidator,
)

__all__ = [
    # Codes and enums
    "SwiftMessageType",
    "SwiftBlockType",
    "SwiftFieldTag",
    "SwiftCharacterSet",
    "SWIFT_MESSAGE_CATEGORIES",
    "get_message_category",
    "get_field_definition",
    # Base message structures
    "SwiftBlock",
    "SwiftBasicHeader",
    "SwiftApplicationHeader",
    "SwiftUserHeader",
    "SwiftTextBlock",
    "SwiftTrailerBlock",
    "SwiftMessage",
    "SwiftField",
    # MT103
    "MT103Message",
    "MT103Field",
    "MT103Parser",
    "MT103Builder",
    "MT103Party",
    "MT103Charges",
    # MT202
    "MT202Message",
    "MT202Field",
    "MT202Parser",
    "MT202Builder",
    "MT202Agent",
    # Parser and validator
    "SwiftParser",
    "SwiftParseError",
    "SwiftValidator",
]
