"""
FedNow Instant Payment Service Implementation

The FedNow Service is the Federal Reserve's instant payment infrastructure
that enables real-time, 24x7x365 payment processing.

Key Features:
- Real-time gross settlement (RTGS)
- 24/7/365 availability
- ISO 20022 message format (pacs.008, pacs.002, camt.056)
- Request for Payment (RfP) support
- Liquidity management tools

Message Types:
- Credit Transfer (pacs.008) - Standard payment
- Payment Status Report (pacs.002) - Payment status/acknowledgment
- Payment Return (pacs.004) - Return of funds
- Request for Payment (pain.013) - Bill presentment
- Request for Payment Response (pain.014)
- Payment Cancellation Request (camt.056)
"""

from typing import List

from ai_engine.domains.banking.protocols.payments.domestic.fednow.fednow_codes import (
    FedNowMessageType,
    FedNowRejectCode,
    FedNowReturnCode,
    FedNowParticipantType,
    FedNowServiceType,
    TransactionStatus,
)
from ai_engine.domains.banking.protocols.payments.domestic.fednow.fednow_message import (
    FedNowMessage,
    FedNowCreditTransfer,
    FedNowPaymentStatus,
    FedNowPaymentReturn,
    FedNowRequestForPayment,
    FedNowParticipant,
    FedNowAccount,
)
from ai_engine.domains.banking.protocols.payments.domestic.fednow.fednow_builder import (
    FedNowBuilder,
    create_instant_payment,
    create_request_for_payment,
)
from ai_engine.domains.banking.protocols.payments.domestic.fednow.fednow_validator import (
    FedNowValidator,
)

__all__: List[str] = [
    # Codes
    "FedNowMessageType",
    "FedNowRejectCode",
    "FedNowReturnCode",
    "FedNowParticipantType",
    "FedNowServiceType",
    "TransactionStatus",
    # Message structures
    "FedNowMessage",
    "FedNowCreditTransfer",
    "FedNowPaymentStatus",
    "FedNowPaymentReturn",
    "FedNowRequestForPayment",
    "FedNowParticipant",
    "FedNowAccount",
    # Builder
    "FedNowBuilder",
    "create_instant_payment",
    "create_request_for_payment",
    # Validator
    "FedNowValidator",
]
