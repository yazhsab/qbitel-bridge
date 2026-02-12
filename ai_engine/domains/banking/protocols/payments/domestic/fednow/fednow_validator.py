"""
FedNow Message Validator

Validation for FedNow Instant Payment messages including:
- Message structure validation
- Amount limits
- Participant validation
- Real-time payment rules
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
    validate_routing_number,
)
from ai_engine.domains.banking.protocols.payments.domestic.fednow.fednow_codes import (
    FedNowMessageType,
    TransactionStatus,
    FEDNOW_MAX_AMOUNT,
    FEDNOW_MIN_AMOUNT,
    FEDNOW_MESSAGE_ID_MAX_LENGTH,
    FEDNOW_END_TO_END_ID_MAX_LENGTH,
)
from ai_engine.domains.banking.protocols.payments.domestic.fednow.fednow_message import (
    FedNowMessage,
    FedNowCreditTransfer,
    FedNowPaymentStatus,
    FedNowPaymentReturn,
    FedNowRequestForPayment,
)


class FedNowValidator(BaseValidator):
    """
    Validator for FedNow Instant Payment messages.

    Validates:
    - Message structure and required fields
    - Amount limits ($500,000 default max)
    - Routing number validity
    - End-to-end ID uniqueness requirements
    - Timing constraints for real-time payments
    """

    def __init__(
        self,
        strict: bool = True,
        max_amount: Optional[Decimal] = None,
        check_routing_numbers: bool = True,
    ):
        """
        Initialize the FedNow validator.

        Args:
            strict: If True, treat warnings as errors
            max_amount: Override default maximum amount
            check_routing_numbers: If True, validate routing number checksums
        """
        super().__init__(strict)
        self.max_amount = max_amount or Decimal(FEDNOW_MAX_AMOUNT) / 100
        self.min_amount = Decimal(FEDNOW_MIN_AMOUNT) / 100
        self.check_routing_numbers = check_routing_numbers

    @property
    def name(self) -> str:
        return "FedNowValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate a FedNow message.

        Args:
            data: FedNow message object or dictionary

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, FedNowCreditTransfer):
            self._validate_credit_transfer(data, result)
        elif isinstance(data, FedNowPaymentStatus):
            self._validate_payment_status(data, result)
        elif isinstance(data, FedNowPaymentReturn):
            self._validate_payment_return(data, result)
        elif isinstance(data, FedNowRequestForPayment):
            self._validate_request_for_payment(data, result)
        elif isinstance(data, FedNowMessage):
            self._validate_base_message(data, result)
        elif isinstance(data, dict):
            self._validate_dict(data, result)
        else:
            result.add_error(
                "FEDNOW_INVALID_INPUT",
                "Input must be a FedNow message object or dictionary",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def _validate_base_message(self, msg: FedNowMessage, result: ValidationResult) -> None:
        """Validate base message fields."""
        # Message ID
        if not msg.message_id:
            result.add_error(
                "FEDNOW_MISSING_MSG_ID",
                "Message ID is required",
                field="message_id",
            )
        elif len(msg.message_id) > FEDNOW_MESSAGE_ID_MAX_LENGTH:
            result.add_error(
                "FEDNOW_INVALID_MSG_ID",
                f"Message ID must not exceed {FEDNOW_MESSAGE_ID_MAX_LENGTH} characters",
                field="message_id",
            )

        # Instructing Agent
        if msg.instructing_agent:
            self._validate_participant(msg.instructing_agent, "instructing_agent", result)
        else:
            result.add_error(
                "FEDNOW_MISSING_INSTG_AGT",
                "Instructing agent is required",
                field="instructing_agent",
            )

        # Instructed Agent
        if msg.instructed_agent:
            self._validate_participant(msg.instructed_agent, "instructed_agent", result)
        else:
            result.add_error(
                "FEDNOW_MISSING_INSTD_AGT",
                "Instructed agent is required",
                field="instructed_agent",
            )

    def _validate_participant(self, participant, field_prefix: str, result: ValidationResult) -> None:
        """Validate participant (financial institution) information."""
        if not participant.routing_number:
            result.add_error(
                "FEDNOW_MISSING_ROUTING",
                f"{field_prefix}: Routing number is required",
                field=f"{field_prefix}/routing_number",
            )
        elif self.check_routing_numbers:
            error = validate_routing_number(participant.routing_number)
            if error:
                result.add_error(
                    "FEDNOW_INVALID_ROUTING",
                    f"{field_prefix}: {error}",
                    field=f"{field_prefix}/routing_number",
                )

    def _validate_credit_transfer(self, msg: FedNowCreditTransfer, result: ValidationResult) -> None:
        """Validate FedNow Credit Transfer (pacs.008)."""
        # Base validation
        self._validate_base_message(msg, result)

        # End-to-End ID
        if not msg.end_to_end_id:
            result.add_error(
                "FEDNOW_MISSING_E2E_ID",
                "End-to-end ID is required",
                field="end_to_end_id",
            )
        elif len(msg.end_to_end_id) > FEDNOW_END_TO_END_ID_MAX_LENGTH:
            result.add_error(
                "FEDNOW_INVALID_E2E_ID",
                f"End-to-end ID must not exceed {FEDNOW_END_TO_END_ID_MAX_LENGTH} characters",
                field="end_to_end_id",
            )

        # UETR
        if not msg.uetr:
            result.add_warning(
                "FEDNOW_MISSING_UETR",
                "UETR is recommended for tracking",
                field="uetr",
            )

        # Amount validation
        self._validate_amount(msg.amount, result)

        # Currency validation
        if msg.currency != "USD":
            result.add_error(
                "FEDNOW_INVALID_CURRENCY",
                "FedNow only supports USD currency",
                field="currency",
            )

        # Debtor validation
        if msg.debtor:
            self._validate_party(msg.debtor, "debtor", result)
        else:
            result.add_error(
                "FEDNOW_MISSING_DEBTOR",
                "Debtor information is required",
                field="debtor",
            )

        # Creditor validation
        if msg.creditor:
            self._validate_party(msg.creditor, "creditor", result)
        else:
            result.add_error(
                "FEDNOW_MISSING_CREDITOR",
                "Creditor information is required",
                field="creditor",
            )

        # Check debtor and creditor are different
        if msg.debtor and msg.creditor:
            if (
                msg.debtor.account.account_number == msg.creditor.account.account_number
                and msg.debtor.account.routing_number == msg.creditor.account.routing_number
            ):
                result.add_error(
                    "FEDNOW_SAME_ACCOUNT",
                    "Debtor and creditor accounts must be different",
                    field="debtor/creditor",
                )

    def _validate_payment_status(self, msg: FedNowPaymentStatus, result: ValidationResult) -> None:
        """Validate FedNow Payment Status (pacs.002)."""
        # Base validation
        self._validate_base_message(msg, result)

        # Original message reference
        if not msg.original_message_id:
            result.add_error(
                "FEDNOW_MISSING_ORIG_MSG_ID",
                "Original message ID is required",
                field="original_message_id",
            )

        if not msg.original_end_to_end_id:
            result.add_error(
                "FEDNOW_MISSING_ORIG_E2E_ID",
                "Original end-to-end ID is required",
                field="original_end_to_end_id",
            )

        # Status validation
        if not msg.transaction_status:
            result.add_error(
                "FEDNOW_MISSING_STATUS",
                "Transaction status is required",
                field="transaction_status",
            )

        # Reject code required for rejected status
        if msg.transaction_status == TransactionStatus.RJCT:
            if not msg.reject_code:
                result.add_error(
                    "FEDNOW_MISSING_REJECT_CODE",
                    "Reject code is required when status is RJCT",
                    field="reject_code",
                )

    def _validate_payment_return(self, msg: FedNowPaymentReturn, result: ValidationResult) -> None:
        """Validate FedNow Payment Return (pacs.004)."""
        # Base validation
        self._validate_base_message(msg, result)

        # Original message reference
        if not msg.original_message_id:
            result.add_error(
                "FEDNOW_MISSING_ORIG_MSG_ID",
                "Original message ID is required",
                field="original_message_id",
            )

        if not msg.original_end_to_end_id:
            result.add_error(
                "FEDNOW_MISSING_ORIG_E2E_ID",
                "Original end-to-end ID is required",
                field="original_end_to_end_id",
            )

        # Return code required
        if not msg.return_code:
            result.add_error(
                "FEDNOW_MISSING_RETURN_CODE",
                "Return code is required",
                field="return_code",
            )

        # Amount validation
        if msg.returned_amount <= 0:
            result.add_error(
                "FEDNOW_INVALID_RETURN_AMOUNT",
                "Returned amount must be greater than zero",
                field="returned_amount",
            )

        if msg.original_amount > 0 and msg.returned_amount > msg.original_amount:
            result.add_error(
                "FEDNOW_RETURN_EXCEEDS_ORIGINAL",
                "Returned amount cannot exceed original amount",
                field="returned_amount",
            )

    def _validate_request_for_payment(self, msg: FedNowRequestForPayment, result: ValidationResult) -> None:
        """Validate FedNow Request for Payment (pain.013)."""
        # Base validation
        self._validate_base_message(msg, result)

        # Amount validation
        self._validate_amount(msg.requested_amount, result, field_prefix="requested_")

        # Currency validation
        if msg.currency != "USD":
            result.add_error(
                "FEDNOW_INVALID_CURRENCY",
                "FedNow only supports USD currency",
                field="currency",
            )

        # Creditor (requester) validation
        if msg.creditor:
            self._validate_party(msg.creditor, "creditor", result)
        else:
            result.add_error(
                "FEDNOW_MISSING_CREDITOR",
                "Creditor (requester) information is required",
                field="creditor",
            )

        # Debtor (payer) validation
        if msg.debtor:
            self._validate_party(msg.debtor, "debtor", result)
        else:
            result.add_error(
                "FEDNOW_MISSING_DEBTOR",
                "Debtor (payer) information is required",
                field="debtor",
            )

        # Expiry validation
        if msg.expiry_datetime:
            if msg.expiry_datetime <= datetime.now():
                result.add_error(
                    "FEDNOW_EXPIRED_RFP",
                    "Request for payment has expired or expiry is in the past",
                    field="expiry_datetime",
                )
            elif msg.expiry_datetime > datetime.now() + timedelta(days=30):
                result.add_warning(
                    "FEDNOW_LONG_EXPIRY",
                    "Request for payment expiry is more than 30 days in the future",
                    field="expiry_datetime",
                )

    def _validate_party(self, party, field_prefix: str, result: ValidationResult) -> None:
        """Validate party (debtor/creditor) information."""
        if not party.name:
            result.add_error(
                "FEDNOW_MISSING_PARTY_NAME",
                f"{field_prefix}: Name is required",
                field=f"{field_prefix}/name",
            )

        if party.account:
            if not party.account.account_number:
                result.add_error(
                    "FEDNOW_MISSING_ACCOUNT",
                    f"{field_prefix}: Account number is required",
                    field=f"{field_prefix}/account/account_number",
                )

            if not party.account.routing_number:
                result.add_error(
                    "FEDNOW_MISSING_ROUTING",
                    f"{field_prefix}: Routing number is required",
                    field=f"{field_prefix}/account/routing_number",
                )
            elif self.check_routing_numbers:
                error = validate_routing_number(party.account.routing_number)
                if error:
                    result.add_error(
                        "FEDNOW_INVALID_ROUTING",
                        f"{field_prefix}: {error}",
                        field=f"{field_prefix}/account/routing_number",
                    )

    def _validate_amount(
        self,
        amount: Decimal,
        result: ValidationResult,
        field_prefix: str = "",
    ) -> None:
        """Validate payment amount."""
        field_name = f"{field_prefix}amount" if field_prefix else "amount"

        if amount <= 0:
            result.add_error(
                "FEDNOW_INVALID_AMOUNT",
                "Amount must be greater than zero",
                field=field_name,
            )
        elif amount < self.min_amount:
            result.add_error(
                "FEDNOW_AMOUNT_TOO_LOW",
                f"Amount ${amount} is below FedNow minimum (${self.min_amount})",
                field=field_name,
            )
        elif amount > self.max_amount:
            result.add_error(
                "FEDNOW_AMOUNT_TOO_HIGH",
                f"Amount ${amount:,.2f} exceeds FedNow maximum (${self.max_amount:,.2f})",
                field=field_name,
            )

    def _validate_dict(self, data: Dict, result: ValidationResult) -> None:
        """Validate FedNow data from dictionary."""
        # Amount
        if "amount" in data:
            try:
                amount = Decimal(str(data["amount"]))
                self._validate_amount(amount, result)
            except Exception:
                result.add_error(
                    "FEDNOW_INVALID_AMOUNT",
                    "Amount must be a valid number",
                    field="amount",
                )

        # Routing numbers
        for field in ["debtor_routing", "creditor_routing", "instructing_agent_routing", "instructed_agent_routing"]:
            if field in data and self.check_routing_numbers:
                error = validate_routing_number(data[field])
                if error:
                    result.add_error(
                        "FEDNOW_INVALID_ROUTING",
                        error,
                        field=field,
                    )

        # Currency
        if "currency" in data and data["currency"] != "USD":
            result.add_error(
                "FEDNOW_INVALID_CURRENCY",
                "FedNow only supports USD currency",
                field="currency",
            )
