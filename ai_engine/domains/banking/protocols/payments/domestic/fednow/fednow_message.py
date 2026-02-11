"""
FedNow Message Data Structures

Core data structures for FedNow Instant Payment messages.
Based on ISO 20022 message formats adapted for FedNow.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional
import uuid

from ai_engine.domains.banking.protocols.payments.domestic.fednow.fednow_codes import (
    FedNowMessageType,
    FedNowRejectCode,
    FedNowReturnCode,
    TransactionStatus,
    FEDNOW_MAX_AMOUNT,
    FEDNOW_MIN_AMOUNT,
)


@dataclass
class FedNowAccount:
    """Account identification for FedNow."""

    # Account number (DDA number)
    account_number: str = ""

    # Account type
    account_type: str = "CACC"  # CACC=Checking, SVGS=Savings

    # Account name
    account_name: str = ""

    # Routing number of the account-holding institution
    routing_number: str = ""

    def validate(self) -> List[str]:
        """Validate account information."""
        errors = []
        if not self.account_number:
            errors.append("Account number is required")
        if not self.routing_number:
            errors.append("Routing number is required")
        elif len(self.routing_number) != 9 or not self.routing_number.isdigit():
            errors.append("Routing number must be 9 digits")
        return errors


@dataclass
class FedNowParticipant:
    """FedNow participant (financial institution) information."""

    # ABA Routing Number (9 digits)
    routing_number: str = ""

    # Institution name
    name: str = ""

    # LEI (Legal Entity Identifier) - optional
    lei: str = ""

    # Participant type
    participant_type: str = "DSR"  # Direct Send and Receive

    # Address
    address_line1: str = ""
    address_line2: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = "US"

    def validate(self) -> List[str]:
        """Validate participant information."""
        errors = []
        if not self.routing_number:
            errors.append("Routing number is required")
        elif len(self.routing_number) != 9:
            errors.append("Routing number must be 9 digits")
        if not self.name:
            errors.append("Institution name is required")
        return errors


@dataclass
class FedNowParty:
    """Party information (originator or beneficiary)."""

    # Name
    name: str = ""

    # Account
    account: FedNowAccount = field(default_factory=FedNowAccount)

    # Contact information
    email: str = ""
    phone: str = ""

    # Address
    address_line1: str = ""
    address_line2: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = "US"

    # Identification
    id_type: str = ""  # TXID, NIDN, CCPT, etc.
    id_value: str = ""

    def validate(self) -> List[str]:
        """Validate party information."""
        errors = []
        if not self.name:
            errors.append("Party name is required")
        errors.extend(self.account.validate())
        return errors


@dataclass
class FedNowRemittanceInfo:
    """Remittance information for FedNow payments."""

    # Unstructured remittance
    unstructured: str = ""

    # Structured remittance
    reference_type: str = ""  # SCOR, CINV, etc.
    reference_number: str = ""
    reference_issuer: str = ""

    # Invoice details
    invoice_number: str = ""
    invoice_date: Optional[date] = None
    invoice_amount: Optional[Decimal] = None

    # Additional information
    additional_info: List[str] = field(default_factory=list)


@dataclass
class FedNowMessage:
    """
    Base class for all FedNow messages.

    Contains common fields used across message types.
    """

    # Message identification
    message_id: str = ""
    creation_datetime: datetime = field(default_factory=datetime.now)

    # Message type
    message_type: FedNowMessageType = FedNowMessageType.CREDIT_TRANSFER

    # Participants
    instructing_agent: FedNowParticipant = field(default_factory=FedNowParticipant)
    instructed_agent: FedNowParticipant = field(default_factory=FedNowParticipant)

    # Processing metadata
    business_message_id: str = ""
    message_definition_id: str = ""

    # Status
    status: TransactionStatus = TransactionStatus.PDNG

    # Raw message for reference
    raw_message: str = ""

    def __post_init__(self):
        """Generate message ID if not provided."""
        if not self.message_id:
            self.message_id = self._generate_message_id()
        if not self.business_message_id:
            self.business_message_id = self.message_id

    def _generate_message_id(self) -> str:
        """Generate a unique message ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique = uuid.uuid4().hex[:8].upper()
        return f"FEDNOW{timestamp}{unique}"[:35]

    def validate(self) -> List[str]:
        """Validate the message."""
        errors = []

        if not self.message_id:
            errors.append("Message ID is required")
        elif len(self.message_id) > 35:
            errors.append("Message ID must not exceed 35 characters")

        errors.extend(self.instructing_agent.validate())
        errors.extend(self.instructed_agent.validate())

        return errors

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.iso_code,
            "creation_datetime": self.creation_datetime.isoformat(),
            "status": self.status.code,
            "instructing_agent": self.instructing_agent.routing_number,
            "instructed_agent": self.instructed_agent.routing_number,
        }


@dataclass
class FedNowCreditTransfer(FedNowMessage):
    """
    FedNow Credit Transfer (pacs.008).

    Instant payment from one account to another.
    """

    # Transaction identification
    end_to_end_id: str = ""
    transaction_id: str = ""
    uetr: str = ""  # Unique End-to-end Transaction Reference

    # Amount
    amount: Decimal = Decimal("0")
    currency: str = "USD"

    # Parties
    debtor: FedNowParty = field(default_factory=FedNowParty)
    creditor: FedNowParty = field(default_factory=FedNowParty)

    # Settlement
    settlement_method: str = "CLRG"  # Clearing system
    settlement_date: Optional[date] = None

    # Remittance
    remittance_info: FedNowRemittanceInfo = field(default_factory=FedNowRemittanceInfo)

    # Charges
    charge_bearer: str = "SLEV"  # Service level

    # Purpose
    purpose_code: str = ""
    category_purpose: str = ""

    def __post_init__(self):
        """Initialize message type and IDs."""
        super().__post_init__()
        self.message_type = FedNowMessageType.CREDIT_TRANSFER

        if not self.end_to_end_id:
            self.end_to_end_id = self._generate_message_id()
        if not self.transaction_id:
            self.transaction_id = self._generate_message_id()
        if not self.uetr:
            self.uetr = str(uuid.uuid4())

    def validate(self) -> List[str]:
        """Validate the credit transfer."""
        errors = super().validate()

        # Amount validation
        if self.amount <= 0:
            errors.append("Amount must be greater than zero")
        elif self.amount * 100 > FEDNOW_MAX_AMOUNT:
            errors.append(f"Amount exceeds FedNow maximum (${FEDNOW_MAX_AMOUNT / 100:,.2f})")
        elif self.amount * 100 < FEDNOW_MIN_AMOUNT:
            errors.append(f"Amount below FedNow minimum (${FEDNOW_MIN_AMOUNT / 100:.2f})")

        # Currency validation
        if self.currency != "USD":
            errors.append("FedNow only supports USD")

        # End-to-end ID validation
        if not self.end_to_end_id:
            errors.append("End-to-end ID is required")
        elif len(self.end_to_end_id) > 35:
            errors.append("End-to-end ID must not exceed 35 characters")

        # Party validation
        errors.extend([f"Debtor: {e}" for e in self.debtor.validate()])
        errors.extend([f"Creditor: {e}" for e in self.creditor.validate()])

        return errors

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "end_to_end_id": self.end_to_end_id,
            "transaction_id": self.transaction_id,
            "uetr": self.uetr,
            "amount": str(self.amount),
            "currency": self.currency,
            "debtor": {
                "name": self.debtor.name,
                "account": self.debtor.account.account_number,
                "routing": self.debtor.account.routing_number,
            },
            "creditor": {
                "name": self.creditor.name,
                "account": self.creditor.account.account_number,
                "routing": self.creditor.account.routing_number,
            },
            "settlement_date": self.settlement_date.isoformat() if self.settlement_date else None,
        })
        return base


@dataclass
class FedNowPaymentStatus(FedNowMessage):
    """
    FedNow Payment Status Report (pacs.002).

    Response to a credit transfer indicating acceptance or rejection.
    """

    # Original message reference
    original_message_id: str = ""
    original_end_to_end_id: str = ""
    original_transaction_id: str = ""
    original_uetr: str = ""

    # Status
    transaction_status: TransactionStatus = TransactionStatus.ACSC

    # Rejection details (if rejected)
    reject_code: Optional[FedNowRejectCode] = None
    reject_reason: str = ""
    additional_info: str = ""

    # Acceptance details (if accepted)
    accepted_datetime: Optional[datetime] = None
    settlement_datetime: Optional[datetime] = None

    # Original amount (for reference)
    original_amount: Optional[Decimal] = None

    def __post_init__(self):
        """Initialize message type."""
        super().__post_init__()
        self.message_type = FedNowMessageType.PAYMENT_STATUS_REPORT

    def validate(self) -> List[str]:
        """Validate the payment status."""
        errors = super().validate()

        if not self.original_message_id:
            errors.append("Original message ID is required")

        if not self.original_end_to_end_id:
            errors.append("Original end-to-end ID is required")

        if self.transaction_status == TransactionStatus.RJCT:
            if not self.reject_code:
                errors.append("Reject code is required for rejected status")

        return errors

    @property
    def is_accepted(self) -> bool:
        """Check if the original message was accepted."""
        return self.transaction_status.is_positive

    @property
    def is_rejected(self) -> bool:
        """Check if the original message was rejected."""
        return self.transaction_status == TransactionStatus.RJCT

    @property
    def is_final(self) -> bool:
        """Check if this is a final status."""
        return self.transaction_status.is_final

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "original_message_id": self.original_message_id,
            "original_end_to_end_id": self.original_end_to_end_id,
            "transaction_status": self.transaction_status.code,
            "is_accepted": self.is_accepted,
            "is_rejected": self.is_rejected,
            "reject_code": self.reject_code.code if self.reject_code else None,
            "reject_reason": self.reject_reason,
        })
        return base


@dataclass
class FedNowPaymentReturn(FedNowMessage):
    """
    FedNow Payment Return (pacs.004).

    Return of a previously settled payment.
    """

    # Original message reference
    original_message_id: str = ""
    original_end_to_end_id: str = ""
    original_transaction_id: str = ""
    original_uetr: str = ""

    # Return details
    return_id: str = ""
    return_code: Optional[FedNowReturnCode] = None
    return_reason: str = ""
    return_additional_info: str = ""

    # Amount being returned
    returned_amount: Decimal = Decimal("0")
    original_amount: Decimal = Decimal("0")

    # Parties (reversed from original)
    return_debtor: FedNowParty = field(default_factory=FedNowParty)  # Original creditor
    return_creditor: FedNowParty = field(default_factory=FedNowParty)  # Original debtor

    # Return timing
    original_settlement_date: Optional[date] = None
    return_request_datetime: Optional[datetime] = None

    def __post_init__(self):
        """Initialize message type."""
        super().__post_init__()
        self.message_type = FedNowMessageType.PAYMENT_RETURN

        if not self.return_id:
            self.return_id = self._generate_message_id()

    def validate(self) -> List[str]:
        """Validate the payment return."""
        errors = super().validate()

        if not self.original_message_id:
            errors.append("Original message ID is required")

        if not self.original_end_to_end_id:
            errors.append("Original end-to-end ID is required")

        if not self.return_code:
            errors.append("Return code is required")

        if self.returned_amount <= 0:
            errors.append("Returned amount must be greater than zero")

        if self.returned_amount > self.original_amount:
            errors.append("Returned amount cannot exceed original amount")

        return errors

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "original_message_id": self.original_message_id,
            "original_end_to_end_id": self.original_end_to_end_id,
            "return_id": self.return_id,
            "return_code": self.return_code.code if self.return_code else None,
            "return_reason": self.return_reason,
            "returned_amount": str(self.returned_amount),
            "original_amount": str(self.original_amount),
        })
        return base


@dataclass
class FedNowRequestForPayment(FedNowMessage):
    """
    FedNow Request for Payment (pain.013).

    Allows a creditor to request payment from a debtor.
    """

    # Request identification
    request_id: str = ""
    end_to_end_id: str = ""

    # Requested amount
    requested_amount: Decimal = Decimal("0")
    currency: str = "USD"

    # Parties
    creditor: FedNowParty = field(default_factory=FedNowParty)  # Requester
    debtor: FedNowParty = field(default_factory=FedNowParty)  # Payer

    # Request details
    expiry_datetime: Optional[datetime] = None
    purpose_code: str = ""
    category_purpose: str = ""

    # Remittance
    remittance_info: FedNowRemittanceInfo = field(default_factory=FedNowRemittanceInfo)

    # Response tracking
    response_received: bool = False
    response_status: Optional[str] = None

    def __post_init__(self):
        """Initialize message type."""
        super().__post_init__()
        self.message_type = FedNowMessageType.REQUEST_FOR_PAYMENT

        if not self.request_id:
            self.request_id = self._generate_message_id()
        if not self.end_to_end_id:
            self.end_to_end_id = self.request_id

    def validate(self) -> List[str]:
        """Validate the request for payment."""
        errors = super().validate()

        if self.requested_amount <= 0:
            errors.append("Requested amount must be greater than zero")

        if self.currency != "USD":
            errors.append("FedNow only supports USD")

        if self.expiry_datetime and self.expiry_datetime <= datetime.now():
            errors.append("Expiry datetime must be in the future")

        errors.extend([f"Creditor: {e}" for e in self.creditor.validate()])
        errors.extend([f"Debtor: {e}" for e in self.debtor.validate()])

        return errors

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "request_id": self.request_id,
            "end_to_end_id": self.end_to_end_id,
            "requested_amount": str(self.requested_amount),
            "currency": self.currency,
            "creditor": {
                "name": self.creditor.name,
                "account": self.creditor.account.account_number,
            },
            "debtor": {
                "name": self.debtor.name,
                "account": self.debtor.account.account_number,
            },
            "expiry_datetime": self.expiry_datetime.isoformat() if self.expiry_datetime else None,
        })
        return base
