"""
SEPA Message Validator

Validation for SEPA payment messages including:
- Credit Transfers (SCT, SCT Inst)
- Direct Debits (SDD Core, SDD B2B)
- IBAN and BIC validation
- Scheme-specific rules
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
    validate_iban,
    validate_bic,
)
from ai_engine.domains.banking.protocols.payments.domestic.sepa.sepa_codes import (
    SEPAScheme,
    SEPASequenceType,
    SEPA_CURRENCY,
    SEPA_MAX_AMOUNT,
    SCT_INST_MAX_AMOUNT,
    SEPA_COUNTRIES,
    SDD_CORE_FIRST_D_DAYS,
    SDD_CORE_RECUR_D_DAYS,
    SDD_B2B_D_DAYS,
)
from ai_engine.domains.banking.protocols.payments.domestic.sepa.sepa_message import (
    SEPACreditTransfer,
    SEPADirectDebit,
)


class SEPAValidator(BaseValidator):
    """
    Validator for SEPA payment messages.

    Validates:
    - Message structure and required fields
    - IBAN and BIC validity
    - Amount limits (standard and instant)
    - Currency (EUR only)
    - SEPA country membership
    - Direct debit mandate requirements
    - Collection date timelines
    """

    def __init__(
        self,
        strict: bool = True,
        max_amount: Optional[Decimal] = None,
        allow_non_sepa_countries: bool = False,
    ):
        """
        Initialize the SEPA validator.

        Args:
            strict: If True, treat warnings as errors
            max_amount: Override default maximum amount
            allow_non_sepa_countries: If True, allow non-SEPA country IBANs
        """
        super().__init__(strict)
        self.max_amount = max_amount or Decimal(str(SEPA_MAX_AMOUNT))
        self.allow_non_sepa_countries = allow_non_sepa_countries

    @property
    def name(self) -> str:
        return "SEPAValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate a SEPA message.

        Args:
            data: SEPA message object or dictionary

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, SEPACreditTransfer):
            self._validate_credit_transfer(data, result)
        elif isinstance(data, SEPADirectDebit):
            self._validate_direct_debit(data, result)
        elif isinstance(data, dict):
            self._validate_dict(data, result)
        else:
            result.add_error(
                "SEPA_INVALID_INPUT",
                "Input must be a SEPA message object or dictionary",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def _validate_credit_transfer(self, msg: SEPACreditTransfer, result: ValidationResult) -> None:
        """Validate SEPA Credit Transfer."""
        # Message ID
        if not msg.message_id:
            result.add_error(
                "SEPA_MISSING_MSG_ID",
                "Message ID is required",
                field="message_id",
            )
        elif len(msg.message_id) > 35:
            result.add_error(
                "SEPA_INVALID_MSG_ID",
                "Message ID must not exceed 35 characters",
                field="message_id",
            )

        # End-to-end ID
        if not msg.end_to_end_id:
            result.add_error(
                "SEPA_MISSING_E2E_ID",
                "End-to-end ID is required",
                field="end_to_end_id",
            )
        elif len(msg.end_to_end_id) > 35:
            result.add_error(
                "SEPA_INVALID_E2E_ID",
                "End-to-end ID must not exceed 35 characters",
                field="end_to_end_id",
            )

        # Amount validation
        self._validate_amount(msg.amount, msg.scheme, result)

        # Currency validation
        if msg.currency != SEPA_CURRENCY:
            result.add_error(
                "SEPA_INVALID_CURRENCY",
                f"SEPA only supports {SEPA_CURRENCY} currency",
                field="currency",
            )

        # Debtor validation
        if msg.debtor:
            self._validate_party(msg.debtor, "debtor", result)
        else:
            result.add_error(
                "SEPA_MISSING_DEBTOR",
                "Debtor information is required",
                field="debtor",
            )

        # Creditor validation
        if msg.creditor:
            self._validate_party(msg.creditor, "creditor", result)
        else:
            result.add_error(
                "SEPA_MISSING_CREDITOR",
                "Creditor information is required",
                field="creditor",
            )

        # Execution date validation
        if msg.requested_execution_date:
            if msg.requested_execution_date < date.today():
                result.add_warning(
                    "SEPA_PAST_EXEC_DATE",
                    "Requested execution date is in the past",
                    field="requested_execution_date",
                )

        # SCT Inst specific validations
        if msg.scheme == SEPAScheme.SCT_INST:
            self._validate_instant_payment(msg, result)

        # Remittance validation
        if msg.remittance_info:
            if msg.remittance_info.unstructured:
                if len(msg.remittance_info.unstructured) > 140:
                    result.add_error(
                        "SEPA_INVALID_REMITTANCE",
                        "Unstructured remittance must not exceed 140 characters",
                        field="remittance_info/unstructured",
                    )

    def _validate_direct_debit(self, msg: SEPADirectDebit, result: ValidationResult) -> None:
        """Validate SEPA Direct Debit."""
        # Message ID
        if not msg.message_id:
            result.add_error(
                "SEPA_MISSING_MSG_ID",
                "Message ID is required",
                field="message_id",
            )
        elif len(msg.message_id) > 35:
            result.add_error(
                "SEPA_INVALID_MSG_ID",
                "Message ID must not exceed 35 characters",
                field="message_id",
            )

        # End-to-end ID
        if not msg.end_to_end_id:
            result.add_error(
                "SEPA_MISSING_E2E_ID",
                "End-to-end ID is required",
                field="end_to_end_id",
            )

        # Creditor Scheme ID (Creditor Identifier)
        if not msg.creditor_scheme_id:
            result.add_error(
                "SEPA_MISSING_CREDITOR_ID",
                "Creditor Scheme ID (Creditor Identifier) is required for direct debits",
                field="creditor_scheme_id",
            )
        elif len(msg.creditor_scheme_id) > 35:
            result.add_error(
                "SEPA_INVALID_CREDITOR_ID",
                "Creditor Scheme ID must not exceed 35 characters",
                field="creditor_scheme_id",
            )

        # Amount validation
        self._validate_amount(msg.amount, msg.scheme, result)

        # Currency validation
        if msg.currency != SEPA_CURRENCY:
            result.add_error(
                "SEPA_INVALID_CURRENCY",
                f"SEPA only supports {SEPA_CURRENCY} currency",
                field="currency",
            )

        # Creditor validation
        if msg.creditor:
            self._validate_party(msg.creditor, "creditor", result)
        else:
            result.add_error(
                "SEPA_MISSING_CREDITOR",
                "Creditor information is required",
                field="creditor",
            )

        # Debtor validation
        if msg.debtor:
            self._validate_party(msg.debtor, "debtor", result)
        else:
            result.add_error(
                "SEPA_MISSING_DEBTOR",
                "Debtor information is required",
                field="debtor",
            )

        # Mandate validation
        self._validate_mandate(msg, result)

        # Collection date validation
        self._validate_collection_date(msg, result)

    def _validate_amount(
        self,
        amount: Decimal,
        scheme: SEPAScheme,
        result: ValidationResult,
    ) -> None:
        """Validate payment amount."""
        if amount <= 0:
            result.add_error(
                "SEPA_INVALID_AMOUNT",
                "Amount must be greater than zero",
                field="amount",
            )
        elif amount > self.max_amount:
            result.add_error(
                "SEPA_AMOUNT_TOO_HIGH",
                f"Amount exceeds SEPA maximum ({self.max_amount})",
                field="amount",
            )

        # SCT Inst specific limit
        if scheme == SEPAScheme.SCT_INST:
            if amount > Decimal(str(SCT_INST_MAX_AMOUNT)):
                result.add_error(
                    "SEPA_INST_AMOUNT_TOO_HIGH",
                    f"Amount exceeds SCT Inst maximum ({SCT_INST_MAX_AMOUNT})",
                    field="amount",
                )

    def _validate_party(self, party, field_prefix: str, result: ValidationResult) -> None:
        """Validate party (debtor/creditor) information."""
        # Name
        if not party.name:
            result.add_error(
                "SEPA_MISSING_PARTY_NAME",
                f"{field_prefix}: Name is required",
                field=f"{field_prefix}/name",
            )
        elif len(party.name) > 70:
            result.add_error(
                "SEPA_INVALID_PARTY_NAME",
                f"{field_prefix}: Name must not exceed 70 characters",
                field=f"{field_prefix}/name",
            )

        # IBAN validation
        if party.account:
            if not party.account.iban:
                result.add_error(
                    "SEPA_MISSING_IBAN",
                    f"{field_prefix}: IBAN is required",
                    field=f"{field_prefix}/account/iban",
                )
            else:
                error = validate_iban(party.account.iban)
                if error:
                    result.add_error(
                        "SEPA_INVALID_IBAN",
                        f"{field_prefix}: {error}",
                        field=f"{field_prefix}/account/iban",
                    )
                else:
                    # Check SEPA country
                    country = party.account.iban[:2].upper()
                    if country not in SEPA_COUNTRIES and not self.allow_non_sepa_countries:
                        result.add_error(
                            "SEPA_NON_SEPA_COUNTRY",
                            f"{field_prefix}: IBAN country {country} is not a SEPA country",
                            field=f"{field_prefix}/account/iban",
                        )

        # BIC validation (optional but recommended)
        if party.agent and party.agent.bic:
            error = validate_bic(party.agent.bic)
            if error:
                result.add_error(
                    "SEPA_INVALID_BIC",
                    f"{field_prefix}: {error}",
                    field=f"{field_prefix}/agent/bic",
                )

    def _validate_mandate(self, msg: SEPADirectDebit, result: ValidationResult) -> None:
        """Validate direct debit mandate."""
        if not msg.mandate:
            result.add_error(
                "SEPA_MISSING_MANDATE",
                "Mandate information is required for direct debits",
                field="mandate",
            )
            return

        # Mandate ID
        if not msg.mandate.mandate_id:
            result.add_error(
                "SEPA_MISSING_MANDATE_ID",
                "Mandate ID is required",
                field="mandate/mandate_id",
            )
        elif len(msg.mandate.mandate_id) > 35:
            result.add_error(
                "SEPA_INVALID_MANDATE_ID",
                "Mandate ID must not exceed 35 characters",
                field="mandate/mandate_id",
            )

        # Date of signature
        if not msg.mandate.date_of_signature:
            result.add_error(
                "SEPA_MISSING_MANDATE_DATE",
                "Mandate date of signature is required",
                field="mandate/date_of_signature",
            )
        elif msg.mandate.date_of_signature > date.today():
            result.add_error(
                "SEPA_FUTURE_MANDATE_DATE",
                "Mandate date of signature cannot be in the future",
                field="mandate/date_of_signature",
            )

        # Amendment validation
        if msg.mandate.amendment_indicator:
            has_amendment_detail = any(
                [
                    msg.mandate.original_mandate_id,
                    msg.mandate.original_creditor_scheme_id,
                    msg.mandate.original_debtor_account,
                    msg.mandate.original_debtor_agent,
                ]
            )
            if not has_amendment_detail:
                result.add_error(
                    "SEPA_MISSING_AMENDMENT_DETAIL",
                    "Amendment details required when amendment indicator is true",
                    field="mandate/amendment_indicator",
                )

    def _validate_collection_date(self, msg: SEPADirectDebit, result: ValidationResult) -> None:
        """Validate direct debit collection date."""
        if not msg.requested_collection_date:
            result.add_warning(
                "SEPA_MISSING_COLLECTION_DATE",
                "Requested collection date not specified",
                field="requested_collection_date",
            )
            return

        today = date.today()
        collection_date = msg.requested_collection_date

        # Cannot be in the past
        if collection_date < today:
            result.add_error(
                "SEPA_PAST_COLLECTION_DATE",
                "Requested collection date cannot be in the past",
                field="requested_collection_date",
            )
            return

        # Calculate minimum lead time based on scheme and sequence type
        if msg.scheme == SEPAScheme.SDD_CORE:
            if msg.sequence_type in [SEPASequenceType.FRST, SEPASequenceType.OOFF]:
                min_days = SDD_CORE_FIRST_D_DAYS
            else:
                min_days = SDD_CORE_RECUR_D_DAYS
        elif msg.scheme == SEPAScheme.SDD_B2B:
            min_days = SDD_B2B_D_DAYS
        else:
            min_days = SDD_CORE_RECUR_D_DAYS

        min_date = today + timedelta(days=min_days)

        if collection_date < min_date:
            result.add_warning(
                "SEPA_INSUFFICIENT_LEAD_TIME",
                f"Collection date may be too soon. {msg.scheme.code} {msg.sequence_type.code} "
                f"requires D-{min_days} lead time",
                field="requested_collection_date",
                min_days=min_days,
            )

    def _validate_instant_payment(self, msg: SEPACreditTransfer, result: ValidationResult) -> None:
        """Validate SCT Inst specific requirements."""
        # BIC is required for instant payments
        if msg.debtor and msg.debtor.agent:
            if not msg.debtor.agent.bic:
                result.add_warning(
                    "SEPA_INST_MISSING_DEBTOR_BIC",
                    "Debtor BIC is recommended for instant payments",
                    field="debtor/agent/bic",
                )

        if msg.creditor and msg.creditor.agent:
            if not msg.creditor.agent.bic:
                result.add_warning(
                    "SEPA_INST_MISSING_CREDITOR_BIC",
                    "Creditor BIC is recommended for instant payments",
                    field="creditor/agent/bic",
                )

    def _validate_dict(self, data: Dict, result: ValidationResult) -> None:
        """Validate SEPA data from dictionary."""
        # Amount
        if "amount" in data:
            try:
                amount = Decimal(str(data["amount"]))
                if amount <= 0:
                    result.add_error(
                        "SEPA_INVALID_AMOUNT",
                        "Amount must be greater than zero",
                        field="amount",
                    )
                elif amount > self.max_amount:
                    result.add_error(
                        "SEPA_AMOUNT_TOO_HIGH",
                        f"Amount exceeds SEPA maximum ({self.max_amount})",
                        field="amount",
                    )
            except Exception:
                result.add_error(
                    "SEPA_INVALID_AMOUNT",
                    "Amount must be a valid number",
                    field="amount",
                )

        # Currency
        if "currency" in data and data["currency"] != SEPA_CURRENCY:
            result.add_error(
                "SEPA_INVALID_CURRENCY",
                f"SEPA only supports {SEPA_CURRENCY} currency",
                field="currency",
            )

        # IBANs
        for field in ["debtor_iban", "creditor_iban"]:
            if field in data:
                error = validate_iban(data[field])
                if error:
                    result.add_error(
                        "SEPA_INVALID_IBAN",
                        error,
                        field=field,
                    )

        # BICs
        for field in ["debtor_bic", "creditor_bic"]:
            if field in data:
                error = validate_bic(data[field])
                if error:
                    result.add_error(
                        "SEPA_INVALID_BIC",
                        error,
                        field=field,
                    )
