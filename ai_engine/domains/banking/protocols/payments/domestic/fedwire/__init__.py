"""
FedWire Protocol Implementation

Federal Reserve Wire Network (Fedwire) Funds Service implementation.
Handles high-value, real-time gross settlement (RTGS) wire transfers.

Message Format:
- Fedwire uses a proprietary fixed-format message structure
- Messages are organized into tags (curly brace delimited)
- Supports both domestic and international transfers

Key Components:
- fedwire_codes: Tag definitions, type codes, and business function codes
- fedwire_message: Core message data structures
- fedwire_parser: Parse incoming Fedwire messages
- fedwire_builder: Build outgoing Fedwire messages
"""

from typing import List

from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_codes import (
    TypeCode,
    TypeSubCode,
    BusinessFunctionCode,
    FedWireTag,
    IDCode,
    DrawdownDebitAccountAdviceCode,
)
from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_message import (
    FedWireMessage,
    SenderInfo,
    ReceiverInfo,
    BeneficiaryInfo,
    OriginatorInfo,
    IntermediaryInfo,
    FIInfo,
    RemittanceInfo,
)
from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_parser import (
    FedWireParser,
)
from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_builder import (
    FedWireBuilder,
)

__all__: List[str] = [
    # Codes
    "TypeCode",
    "TypeSubCode",
    "BusinessFunctionCode",
    "FedWireTag",
    "IDCode",
    "DrawdownDebitAccountAdviceCode",
    # Message structures
    "FedWireMessage",
    "SenderInfo",
    "ReceiverInfo",
    "BeneficiaryInfo",
    "OriginatorInfo",
    "IntermediaryInfo",
    "FIInfo",
    "RemittanceInfo",
    # Parser and Builder
    "FedWireParser",
    "FedWireBuilder",
]
