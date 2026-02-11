"""
FedWire Message Builder

Builds FedWire Funds Transfer messages in wire format.
"""

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_codes import (
    TypeCode,
    TypeSubCode,
    BusinessFunctionCode,
    FedWireTag,
    IDCode,
)
from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_message import (
    FedWireMessage,
    SenderInfo,
    ReceiverInfo,
    BeneficiaryInfo,
    OriginatorInfo,
    IntermediaryInfo,
    FIInfo,
    Address,
    IMAD,
    SenderSuppliedInfo,
    OriginatorToBeneficiaryInfo,
    FIToFIInfo,
)

logger = logging.getLogger(__name__)


class FedWireBuildError(Exception):
    """Exception raised when FedWire message building fails."""
    pass


class FedWireBuilder:
    """
    Builder for FedWire Funds Transfer messages.

    Uses a fluent interface to construct FedWire messages with proper
    formatting and validation.

    Example usage:
        builder = FedWireBuilder()
        message = (builder
            .set_type(TypeCode.BASIC_FUNDS_TRANSFER, TypeSubCode.BASIC_TRANSFER)
            .set_amount(Decimal("10000.00"))
            .set_business_function(BusinessFunctionCode.CUSTOMER_TRANSFER)
            .set_sender("021000089", "CITIBANK NA")
            .set_receiver("021000021", "JPMORGAN CHASE")
            .set_beneficiary("ACME CORP", "1234567890")
            .build())
    """

    def __init__(self):
        """Initialize the builder with default values."""
        self._message = FedWireMessage()
        self._sequence_number = 1

    def set_sender_supplied(
        self,
        format_version: str = "30",
        user_request_correlation: str = "",
        test_production: str = "P",
        message_dup_code: str = " ",
    ) -> "FedWireBuilder":
        """Set sender supplied information (tag 1500)."""
        self._message.sender_supplied = SenderSuppliedInfo(
            format_version=format_version,
            user_request_correlation=user_request_correlation,
            test_production=test_production,
            message_dup_code=message_dup_code,
        )
        return self

    def set_type(
        self,
        type_code: TypeCode,
        subcode: TypeSubCode = TypeSubCode.BASIC_TRANSFER,
    ) -> "FedWireBuilder":
        """Set message type and subtype (tag 1510)."""
        self._message.type_code = type_code
        self._message.type_subcode = subcode
        return self

    def set_imad(
        self,
        source_id: str,
        cycle_code: str = "01",
        sequence_number: Optional[str] = None,
        input_date: Optional[date] = None,
    ) -> "FedWireBuilder":
        """Set Input Message Accountability Data (tag 1520)."""
        if sequence_number is None:
            sequence_number = f"{self._sequence_number:06d}"
            self._sequence_number += 1

        self._message.imad = IMAD(
            input_date=input_date or date.today(),
            source_id=source_id[:8].ljust(8),
            cycle_code=cycle_code[:2].ljust(2),
            sequence_number=sequence_number[:6],
        )
        return self

    def set_amount(
        self,
        amount: Decimal,
        currency: str = "USD",
    ) -> "FedWireBuilder":
        """Set transfer amount (tag 2000)."""
        self._message.amount = amount
        self._message.currency = currency
        return self

    def set_sender(
        self,
        routing_number: str,
        short_name: str,
        address_line1: str = "",
        address_line2: str = "",
        address_line3: str = "",
    ) -> "FedWireBuilder":
        """Set sender depository institution (tag 3100)."""
        self._message.sender = SenderInfo(
            routing_number=routing_number,
            short_name=short_name,
            address=Address(
                line1=address_line1,
                line2=address_line2,
                line3=address_line3,
            ),
        )
        return self

    def set_receiver(
        self,
        routing_number: str,
        short_name: str,
        address_line1: str = "",
        address_line2: str = "",
        address_line3: str = "",
    ) -> "FedWireBuilder":
        """Set receiver depository institution (tag 3400)."""
        self._message.receiver = ReceiverInfo(
            routing_number=routing_number,
            short_name=short_name,
            address=Address(
                line1=address_line1,
                line2=address_line2,
                line3=address_line3,
            ),
        )
        return self

    def set_business_function(
        self,
        code: BusinessFunctionCode,
    ) -> "FedWireBuilder":
        """Set business function code (tag 3600)."""
        self._message.business_function_code = code
        return self

    def set_sender_reference(self, reference: str) -> "FedWireBuilder":
        """Set sender reference (tag 3320)."""
        self._message.sender_reference = reference[:16]
        return self

    def set_previous_message_id(self, message_id: str) -> "FedWireBuilder":
        """Set previous message identifier (tag 3500)."""
        self._message.previous_message_id = message_id[:22]
        return self

    def set_beneficiary(
        self,
        name: str,
        identifier: str = "",
        id_code: Optional[IDCode] = None,
        address_line1: str = "",
        address_line2: str = "",
    ) -> "FedWireBuilder":
        """Set beneficiary information (tag 4200)."""
        self._message.beneficiary = BeneficiaryInfo(
            id_code=id_code,
            identifier=identifier,
            name=name,
            address=Address(
                line1=address_line1,
                line2=address_line2,
            ),
        )
        return self

    def set_beneficiary_fi(
        self,
        identifier: str,
        name: str = "",
        id_code: Optional[IDCode] = None,
        address_line1: str = "",
    ) -> "FedWireBuilder":
        """Set beneficiary financial institution (tag 4000)."""
        self._message.beneficiary_fi = FIInfo(
            id_code=id_code,
            identifier=identifier,
            name=name,
            address=Address(line1=address_line1),
        )
        return self

    def set_intermediary_fi(
        self,
        identifier: str,
        name: str = "",
        id_code: Optional[IDCode] = None,
        address_line1: str = "",
    ) -> "FedWireBuilder":
        """Set intermediary financial institution (tag 4100)."""
        self._message.intermediary_fi = IntermediaryInfo(
            id_code=id_code,
            identifier=identifier,
            name=name,
            address=Address(line1=address_line1),
        )
        return self

    def set_originator(
        self,
        name: str,
        identifier: str = "",
        id_code: Optional[IDCode] = None,
        address_line1: str = "",
        address_line2: str = "",
    ) -> "FedWireBuilder":
        """Set originator information (tag 5000)."""
        self._message.originator = OriginatorInfo(
            id_code=id_code,
            identifier=identifier,
            name=name,
            address=Address(
                line1=address_line1,
                line2=address_line2,
            ),
        )
        return self

    def set_originator_fi(
        self,
        identifier: str,
        name: str = "",
        id_code: Optional[IDCode] = None,
        address_line1: str = "",
    ) -> "FedWireBuilder":
        """Set originator financial institution (tag 5100)."""
        self._message.originator_fi = FIInfo(
            id_code=id_code,
            identifier=identifier,
            name=name,
            address=Address(line1=address_line1),
        )
        return self

    def set_instructing_fi(
        self,
        identifier: str,
        name: str = "",
        id_code: Optional[IDCode] = None,
    ) -> "FedWireBuilder":
        """Set instructing financial institution (tag 5200)."""
        self._message.instructing_fi = FIInfo(
            id_code=id_code,
            identifier=identifier,
            name=name,
        )
        return self

    def set_account_debited(self, account: str) -> "FedWireBuilder":
        """Set account debited in drawdown (tag 5400)."""
        self._message.account_debited = account[:34]
        return self

    def set_account_credited(self, account: str) -> "FedWireBuilder":
        """Set account credited in drawdown (tag 5500)."""
        self._message.account_credited = account[:34]
        return self

    def set_charges(
        self,
        charge_details: str,
        send_amount: Optional[Decimal] = None,
        currency: str = "USD",
    ) -> "FedWireBuilder":
        """Set charges information (tag 3700)."""
        from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_message import Charges
        self._message.charges = Charges(
            charge_details=charge_details,
            send_amount=send_amount,
            currency=currency,
        )
        return self

    def set_instructed_amount(
        self,
        amount: Decimal,
        currency: str = "USD",
    ) -> "FedWireBuilder":
        """Set instructed amount (tag 3710)."""
        self._message.instructed_amount = amount
        self._message.instructed_currency = currency
        return self

    def set_exchange_rate(self, rate: Decimal) -> "FedWireBuilder":
        """Set exchange rate (tag 3720)."""
        self._message.exchange_rate = rate
        return self

    def set_originator_to_beneficiary_info(
        self,
        line1: str = "",
        line2: str = "",
        line3: str = "",
        line4: str = "",
    ) -> "FedWireBuilder":
        """Set originator to beneficiary information (tag 6000)."""
        self._message.originator_to_beneficiary = OriginatorToBeneficiaryInfo(
            line1=line1[:35] if line1 else "",
            line2=line2[:35] if line2 else "",
            line3=line3[:35] if line3 else "",
            line4=line4[:35] if line4 else "",
        )
        return self

    def set_fi_to_fi_info(
        self,
        line1: str = "",
        line2: str = "",
        line3: str = "",
        line4: str = "",
        line5: str = "",
        line6: str = "",
    ) -> "FedWireBuilder":
        """Set FI to FI information (tag 6100)."""
        self._message.fi_to_fi_info = FIToFIInfo(
            line1=line1[:35] if line1 else "",
            line2=line2[:35] if line2 else "",
            line3=line3[:35] if line3 else "",
            line4=line4[:35] if line4 else "",
            line5=line5[:35] if line5 else "",
            line6=line6[:35] if line6 else "",
        )
        return self

    def set_unstructured_addenda(self, addenda: str) -> "FedWireBuilder":
        """Set unstructured addenda (tag 7500)."""
        self._message.unstructured_addenda = addenda[:9000]
        return self

    def set_service_message(self, message: str) -> "FedWireBuilder":
        """Set service message (tag 9000)."""
        self._message.service_message = message[:200]
        return self

    def validate(self) -> list:
        """Validate the message before building."""
        return self._message.validate()

    def build(self, validate: bool = True) -> str:
        """
        Build the FedWire message in wire format.

        Args:
            validate: If True, validate message before building

        Returns:
            FedWire message string in wire format

        Raises:
            FedWireBuildError: If validation fails
        """
        if validate:
            errors = self.validate()
            if errors:
                raise FedWireBuildError(
                    f"Message validation failed: {'; '.join(errors)}"
                )

        # Build message parts
        parts = []

        # Mandatory fields
        parts.append(self._build_tag("1500", self._message.sender_supplied.to_string()))
        parts.append(self._build_tag(
            "1510",
            f"{self._message.type_code.code}{self._message.type_subcode.code}"
        ))
        parts.append(self._build_tag("1520", self._message.imad.to_string()))
        parts.append(self._build_tag("2000", self._format_amount(self._message.amount)))
        parts.append(self._build_tag("3100", self._format_di(self._message.sender)))
        parts.append(self._build_tag("3400", self._format_di(self._message.receiver)))
        parts.append(self._build_tag("3600", self._message.business_function_code.code))

        # Optional fields
        if self._message.sender_reference:
            parts.append(self._build_tag("3320", self._message.sender_reference))

        if self._message.previous_message_id:
            parts.append(self._build_tag("3500", self._message.previous_message_id))

        if self._message.charges:
            parts.append(self._build_tag("3700", self._format_charges(self._message.charges)))

        if self._message.instructed_amount:
            amount_str = self._format_amount(self._message.instructed_amount)
            parts.append(self._build_tag(
                "3710",
                f"{self._message.instructed_currency}{amount_str}"
            ))

        if self._message.exchange_rate:
            parts.append(self._build_tag("3720", str(self._message.exchange_rate)))

        if self._message.beneficiary_fi:
            parts.append(self._build_tag("4000", self._message.beneficiary_fi.to_tag_value()))

        if self._message.intermediary_fi:
            parts.append(self._build_tag("4100", self._message.intermediary_fi.to_tag_value()))

        if self._message.beneficiary:
            parts.append(self._build_tag("4200", self._message.beneficiary.to_tag_value()))

        if self._message.originator:
            parts.append(self._build_tag("5000", self._message.originator.to_tag_value()))

        if self._message.originator_fi:
            parts.append(self._build_tag("5100", self._message.originator_fi.to_tag_value()))

        if self._message.instructing_fi:
            parts.append(self._build_tag("5200", self._message.instructing_fi.to_tag_value()))

        if self._message.account_debited:
            parts.append(self._build_tag("5400", self._message.account_debited))

        if self._message.account_credited:
            parts.append(self._build_tag("5500", self._message.account_credited))

        if self._message.originator_to_beneficiary:
            parts.append(self._build_tag(
                "6000",
                self._message.originator_to_beneficiary.to_tag_value()
            ))

        if self._message.fi_to_fi_info:
            parts.append(self._build_tag("6100", self._message.fi_to_fi_info.to_tag_value()))

        if self._message.drawdown_advice_code:
            advice_value = self._message.drawdown_advice_code
            if self._message.drawdown_advice_info:
                advice_value += "*" + "*".join(self._message.drawdown_advice_info)
            parts.append(self._build_tag("6200", advice_value))

        if self._message.unstructured_addenda:
            parts.append(self._build_tag("7500", self._message.unstructured_addenda))

        if self._message.service_message:
            parts.append(self._build_tag("9000", self._message.service_message))

        return "".join(parts)

    def build_message(self, validate: bool = True) -> FedWireMessage:
        """
        Build and return the FedWireMessage object.

        Args:
            validate: If True, validate message before returning

        Returns:
            FedWireMessage object

        Raises:
            FedWireBuildError: If validation fails
        """
        if validate:
            errors = self.validate()
            if errors:
                raise FedWireBuildError(
                    f"Message validation failed: {'; '.join(errors)}"
                )

        # Set the raw message
        self._message.raw_message = self.build(validate=False)
        return self._message

    def _build_tag(self, tag: str, value: str) -> str:
        """Build a tag-value pair."""
        return f"{{{tag}}}{value}"

    def _format_amount(self, amount: Decimal) -> str:
        """Format amount as 12-digit string (in cents)."""
        cents = int(amount * 100)
        return f"{cents:012d}"

    def _format_di(self, di) -> str:
        """Format depository institution information."""
        parts = [di.routing_number, di.short_name]

        if di.address.line1:
            parts.append(di.address.line1)
        if di.address.line2:
            parts.append(di.address.line2)
        if di.address.line3:
            parts.append(di.address.line3)

        # Join with asterisks after routing number
        if len(parts) > 1:
            return parts[0] + "*".join(parts[1:])
        return parts[0]

    def _format_charges(self, charges) -> str:
        """Format charges information."""
        result = charges.charge_details
        if charges.currency and charges.currency != "USD":
            result += charges.currency
        if charges.send_amount:
            result += self._format_amount(charges.send_amount)
        return result


def create_bank_transfer(
    sender_routing: str,
    sender_name: str,
    receiver_routing: str,
    receiver_name: str,
    amount: Decimal,
    source_id: str,
    sender_reference: str = "",
) -> str:
    """
    Create a simple bank-to-bank transfer (BTR).

    Args:
        sender_routing: Sender's ABA routing number
        sender_name: Sender bank name
        receiver_routing: Receiver's ABA routing number
        receiver_name: Receiver bank name
        amount: Transfer amount in dollars
        source_id: FedWire source ID (8 characters)
        sender_reference: Optional sender reference

    Returns:
        FedWire message in wire format
    """
    builder = FedWireBuilder()

    builder.set_type(TypeCode.BASIC_FUNDS_TRANSFER, TypeSubCode.BASIC_TRANSFER)
    builder.set_imad(source_id)
    builder.set_amount(amount)
    builder.set_sender(sender_routing, sender_name)
    builder.set_receiver(receiver_routing, receiver_name)
    builder.set_business_function(BusinessFunctionCode.BANK_TRANSFER)

    if sender_reference:
        builder.set_sender_reference(sender_reference)

    return builder.build()


def create_customer_transfer(
    sender_routing: str,
    sender_name: str,
    receiver_routing: str,
    receiver_name: str,
    amount: Decimal,
    source_id: str,
    beneficiary_name: str,
    beneficiary_account: str = "",
    originator_name: str = "",
    originator_account: str = "",
    payment_details: str = "",
) -> str:
    """
    Create a customer credit transfer (CTR or CTP).

    Args:
        sender_routing: Sender's ABA routing number
        sender_name: Sender bank name
        receiver_routing: Receiver's ABA routing number
        receiver_name: Receiver bank name
        amount: Transfer amount in dollars
        source_id: FedWire source ID (8 characters)
        beneficiary_name: Name of the beneficiary
        beneficiary_account: Beneficiary account number
        originator_name: Name of the originator (for CTP)
        originator_account: Originator account number
        payment_details: Payment reference or details

    Returns:
        FedWire message in wire format
    """
    builder = FedWireBuilder()

    # Determine if CTR or CTP based on originator presence
    if originator_name:
        bfc = BusinessFunctionCode.CUSTOMER_TRANSFER_PLUS
    else:
        bfc = BusinessFunctionCode.CUSTOMER_TRANSFER

    builder.set_type(TypeCode.BASIC_FUNDS_TRANSFER, TypeSubCode.BASIC_TRANSFER)
    builder.set_imad(source_id)
    builder.set_amount(amount)
    builder.set_sender(sender_routing, sender_name)
    builder.set_receiver(receiver_routing, receiver_name)
    builder.set_business_function(bfc)

    # Set beneficiary
    builder.set_beneficiary(
        name=beneficiary_name,
        identifier=beneficiary_account,
        id_code=IDCode.DDA_NUMBER if beneficiary_account else None,
    )

    # Set originator if provided
    if originator_name:
        builder.set_originator(
            name=originator_name,
            identifier=originator_account,
            id_code=IDCode.DDA_NUMBER if originator_account else None,
        )

    # Set payment details
    if payment_details:
        builder.set_originator_to_beneficiary_info(line1=payment_details)

    return builder.build()


def create_cover_payment(
    sender_routing: str,
    sender_name: str,
    receiver_routing: str,
    receiver_name: str,
    amount: Decimal,
    source_id: str,
    beneficiary_name: str,
    beneficiary_fi_bic: str,
    originator_name: str,
    originator_fi_bic: str,
    underlying_reference: str = "",
) -> str:
    """
    Create a cover payment (COV) for an underlying transaction.

    Cover payments are used to fund correspondent banking transactions
    where the actual payment instruction travels separately.

    Args:
        sender_routing: Sender's ABA routing number
        sender_name: Sender bank name
        receiver_routing: Receiver's ABA routing number
        receiver_name: Receiver bank name
        amount: Transfer amount in dollars
        source_id: FedWire source ID (8 characters)
        beneficiary_name: Name of the beneficiary
        beneficiary_fi_bic: Beneficiary FI's BIC/SWIFT code
        originator_name: Name of the originator
        originator_fi_bic: Originator FI's BIC/SWIFT code
        underlying_reference: Reference to underlying transaction

    Returns:
        FedWire message in wire format
    """
    builder = FedWireBuilder()

    builder.set_type(TypeCode.BASIC_FUNDS_TRANSFER, TypeSubCode.BASIC_TRANSFER)
    builder.set_imad(source_id)
    builder.set_amount(amount)
    builder.set_sender(sender_routing, sender_name)
    builder.set_receiver(receiver_routing, receiver_name)
    builder.set_business_function(BusinessFunctionCode.COVER_PAYMENT)

    # Set beneficiary
    builder.set_beneficiary(name=beneficiary_name)

    # Set beneficiary FI
    builder.set_beneficiary_fi(
        identifier=beneficiary_fi_bic,
        id_code=IDCode.SWIFT_BIC,
    )

    # Set originator
    builder.set_originator(name=originator_name)

    # Set originator FI
    builder.set_originator_fi(
        identifier=originator_fi_bic,
        id_code=IDCode.SWIFT_BIC,
    )

    # Set underlying reference
    if underlying_reference:
        builder.set_fi_to_fi_info(line1=f"UNDERLYING REF: {underlying_reference}")

    return builder.build()
