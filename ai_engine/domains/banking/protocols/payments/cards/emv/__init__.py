"""
EMV Protocol Implementation

EMV (Europay, Mastercard, Visa) chip card protocol support for:
- Application Selection (PSE, PPSE)
- Application Initialization
- Transaction Processing
- Risk Management
- Cardholder Verification
- Cryptogram Generation (ARQC, TC, AAC)

This implementation focuses on contact and contactless EMV transactions.
"""

from ai_engine.domains.banking.protocols.payments.cards.emv.emv_codes import (
    EmvTag,
    EmvApplication,
    EmvTransactionType,
    EmvCryptogramType,
    EmvCvmResult,
    EmvTerminalType,
    EmvPosEntryMode,
    TVR_BITS,
    TSI_BITS,
    AIP_BITS,
    get_tag_name,
    get_tag_format,
)
from ai_engine.domains.banking.protocols.payments.cards.emv.emv_tlv import (
    TlvParser,
    TlvBuilder,
    TlvTag,
    TlvData,
    parse_tlv,
    build_tlv,
)
from ai_engine.domains.banking.protocols.payments.cards.emv.emv_message import (
    EmvTransaction,
    EmvCard,
    EmvTerminal,
    EmvCryptogram,
    EmvResponse,
)
from ai_engine.domains.banking.protocols.payments.cards.emv.emv_validator import (
    EmvValidator,
)

__all__ = [
    # Codes and enums
    "EmvTag",
    "EmvApplication",
    "EmvTransactionType",
    "EmvCryptogramType",
    "EmvCvmResult",
    "EmvTerminalType",
    "EmvPosEntryMode",
    "TVR_BITS",
    "TSI_BITS",
    "AIP_BITS",
    "get_tag_name",
    "get_tag_format",
    # TLV
    "TlvParser",
    "TlvBuilder",
    "TlvTag",
    "TlvData",
    "parse_tlv",
    "build_tlv",
    # Message structures
    "EmvTransaction",
    "EmvCard",
    "EmvTerminal",
    "EmvCryptogram",
    "EmvResponse",
    # Validator
    "EmvValidator",
]
