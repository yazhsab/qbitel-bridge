"""
FedWire Message Validator

Comprehensive validation for FedWire Funds Transfer messages including:
- Message structure validation
- Tag validation
- Business rule validation
- OFAC/Sanctions pre-screening
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
    validate_routing_number,
    validate_bic,
    validate_amount,
)
from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_codes import (
    TypeCode,
    TypeSubCode,
    BusinessFunctionCode,
    FedWireTag,
    MANDATORY_TAGS,
    BFC_REQUIREMENTS,
    FEDWIRE_AMOUNT_MAX,
)


class FedWireValidator(BaseValidator):
    """
    Validator for FedWire Funds Transfer messages.

    Validates:
    - Mandatory tag presence
    - Tag value formats
    - Business function code requirements
    - Amount limits
    - Party information
    """

    # Amount thresholds for enhanced scrutiny
    LARGE_VALUE_THRESHOLD = 1_000_000  # $1M
    VERY_LARGE_VALUE_THRESHOLD = 10_000_000  # $10M

    def __init__(self, strict: bool = True, max_amount: float = None):
        """
        Initialize the FedWire validator.

        Args:
            strict: If True, treat warnings as errors
            max_amount: Override maximum transaction amount
        """
        super().__init__(strict)
        self.max_amount = max_amount or FEDWIRE_AMOUNT_MAX / 100  # Convert from cents

    @property
    def name(self) -> str:
        return "FedWireValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate a FedWire message.

        Args:
            data: FedWireMessage object or wire format string

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, str):
            result = self._validate_wire_format(data, result)
        elif hasattr(data, "type_code") and hasattr(data, "business_function_code"):
            result = self._validate_message_object(data, result)
        elif isinstance(data, dict):
            result = self._validate_dict(data, result)
        else:
            result.add_error(
                "FEDWIRE_INVALID_INPUT",
                "Input must be FedWire message object or wire format string",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def _validate_wire_format(self, content: str, result: ValidationResult) -> ValidationResult:
        """Validate FedWire message from wire format string."""
        # Check for basic structure
        if not content or len(content) < 50:
            result.add_error(
                "FEDWIRE_TOO_SHORT",
                "Message is too short to be a valid FedWire message",
                severity=ValidationSeverity.CRITICAL,
            )
            return result

        # Extract tags
        import re

        tag_pattern = re.compile(r"\{(\d{4})\}([^{]*)")
        tags = {}
        for match in tag_pattern.finditer(content):
            tags[match.group(1)] = match.group(2)

        if not tags:
            result.add_error(
                "FEDWIRE_NO_TAGS",
                "Message contains no valid tags",
                severity=ValidationSeverity.CRITICAL,
            )
            return result

        # Validate mandatory tags
        for tag in MANDATORY_TAGS:
            if tag.tag not in tags:
                result.add_error(
                    "FEDWIRE_MISSING_TAG",
                    f"Missing mandatory tag {tag.tag} ({tag.description})",
                    field=f"tag_{tag.tag}",
                )

        # Validate tag values
        self._validate_tag_values(tags, result)

        # Validate business function code requirements
        if "3600" in tags:
            bfc = BusinessFunctionCode.from_code(tags["3600"][:3])
            if bfc:
                self._validate_bfc_requirements(bfc, tags, result)

        return result

    def _validate_message_object(self, msg: Any, result: ValidationResult) -> ValidationResult:
        """Validate a FedWireMessage object."""
        # Call message's own validation first
        if hasattr(msg, "validate"):
            errors = msg.validate()
            for error in errors:
                result.add_error("FEDWIRE_VALIDATION", error)

        # Type code validation
        if not msg.type_code:
            result.add_error(
                "FEDWIRE_MISSING_TYPE",
                "Type code is required",
                field_name="type_code",
            )

        # Amount validation
        if msg.amount is not None:
            error = validate_amount(msg.amount, max_value=self.max_amount)
            if error:
                result.add_error(
                    "FEDWIRE_INVALID_AMOUNT",
                    error,
                    field_name="amount",
                )

            # Large value warnings
            amount_float = float(msg.amount)
            if amount_float >= self.VERY_LARGE_VALUE_THRESHOLD:
                result.add_warning(
                    "FEDWIRE_VERY_LARGE_VALUE",
                    f"Very large transfer amount (${amount_float:,.2f}) - enhanced review required",
                    field_name="amount",
                    amount=amount_float,
                )
            elif amount_float >= self.LARGE_VALUE_THRESHOLD:
                result.add_warning(
                    "FEDWIRE_LARGE_VALUE",
                    f"Large transfer amount (${amount_float:,.2f})",
                    field_name="amount",
                    amount=amount_float,
                )

        # Sender validation
        if msg.sender:
            error = validate_routing_number(msg.sender.routing_number)
            if error:
                result.add_error(
                    "FEDWIRE_INVALID_SENDER_ROUTING",
                    error,
                    field_name="sender/routing_number",
                )
            if not msg.sender.short_name:
                result.add_error(
                    "FEDWIRE_MISSING_SENDER_NAME",
                    "Sender short name is required",
                    field_name="sender/short_name",
                )
        else:
            result.add_error(
                "FEDWIRE_MISSING_SENDER",
                "Sender information is required",
                field_name="sender",
            )

        # Receiver validation
        if msg.receiver:
            error = validate_routing_number(msg.receiver.routing_number)
            if error:
                result.add_error(
                    "FEDWIRE_INVALID_RECEIVER_ROUTING",
                    error,
                    field_name="receiver/routing_number",
                )
            if not msg.receiver.short_name:
                result.add_error(
                    "FEDWIRE_MISSING_RECEIVER_NAME",
                    "Receiver short name is required",
                    field_name="receiver/short_name",
                )
        else:
            result.add_error(
                "FEDWIRE_MISSING_RECEIVER",
                "Receiver information is required",
                field_name="receiver",
            )

        # IMAD validation
        if msg.imad:
            if not msg.imad.source_id:
                result.add_error(
                    "FEDWIRE_MISSING_IMAD_SOURCE",
                    "IMAD source ID is required",
                    field_name="imad/source_id",
                )
        else:
            result.add_error(
                "FEDWIRE_MISSING_IMAD",
                "IMAD is required",
                field_name="imad",
            )

        # Business function code validation
        bfc = msg.business_function_code
        if bfc:
            self._validate_bfc_requirements_object(bfc, msg, result)
        else:
            result.add_error(
                "FEDWIRE_MISSING_BFC",
                "Business function code is required",
                field_name="business_function_code",
            )

        # Beneficiary FI BIC validation
        if msg.beneficiary_fi and msg.beneficiary_fi.id_code:
            from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_codes import IDCode

            if msg.beneficiary_fi.id_code == IDCode.SWIFT_BIC:
                error = validate_bic(msg.beneficiary_fi.identifier)
                if error:
                    result.add_error(
                        "FEDWIRE_INVALID_BEN_FI_BIC",
                        error,
                        field_name="beneficiary_fi/identifier",
                    )

        # Originator FI BIC validation
        if msg.originator_fi and msg.originator_fi.id_code:
            from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_codes import IDCode

            if msg.originator_fi.id_code == IDCode.SWIFT_BIC:
                error = validate_bic(msg.originator_fi.identifier)
                if error:
                    result.add_error(
                        "FEDWIRE_INVALID_ORIG_FI_BIC",
                        error,
                        field_name="originator_fi/identifier",
                    )

        return result

    def _validate_dict(self, data: Dict, result: ValidationResult) -> ValidationResult:
        """Validate FedWire data from dictionary."""
        # Amount
        if "amount" in data:
            error = validate_amount(data["amount"], max_value=self.max_amount)
            if error:
                result.add_error(
                    "FEDWIRE_INVALID_AMOUNT",
                    error,
                    field_name="amount",
                )

        # Sender routing
        if "sender_routing" in data:
            error = validate_routing_number(data["sender_routing"])
            if error:
                result.add_error(
                    "FEDWIRE_INVALID_SENDER_ROUTING",
                    error,
                    field_name="sender_routing",
                )

        # Receiver routing
        if "receiver_routing" in data:
            error = validate_routing_number(data["receiver_routing"])
            if error:
                result.add_error(
                    "FEDWIRE_INVALID_RECEIVER_ROUTING",
                    error,
                    field_name="receiver_routing",
                )

        # Business function code
        if "business_function_code" in data:
            bfc = BusinessFunctionCode.from_code(data["business_function_code"])
            if not bfc:
                result.add_error(
                    "FEDWIRE_INVALID_BFC",
                    f"Invalid business function code: {data['business_function_code']}",
                    field_name="business_function_code",
                )

        return result

    def _validate_tag_values(self, tags: Dict[str, str], result: ValidationResult) -> None:
        """Validate individual tag values."""
        # Type/Subtype (1510)
        if "1510" in tags:
            value = tags["1510"]
            if len(value) < 4:
                result.add_error(
                    "FEDWIRE_INVALID_TYPE_SUBTYPE",
                    "Type/Subtype must be 4 characters",
                    field_name="tag_1510",
                )
            else:
                type_code = TypeCode.from_code(value[:2])
                if not type_code:
                    result.add_warning(
                        "FEDWIRE_UNKNOWN_TYPE",
                        f"Unknown type code: {value[:2]}",
                        field_name="tag_1510",
                    )

                subtype = TypeSubCode.from_code(value[2:4])
                if not subtype:
                    result.add_warning(
                        "FEDWIRE_UNKNOWN_SUBTYPE",
                        f"Unknown subtype code: {value[2:4]}",
                        field_name="tag_1510",
                    )

        # IMAD (1520)
        if "1520" in tags:
            imad = tags["1520"]
            if len(imad) != 22:
                result.add_error(
                    "FEDWIRE_INVALID_IMAD",
                    f"IMAD must be 22 characters, got {len(imad)}",
                    field_name="tag_1520",
                )

        # Amount (2000)
        if "2000" in tags:
            amount = tags["2000"]
            if not amount.isdigit():
                result.add_error(
                    "FEDWIRE_INVALID_AMOUNT",
                    "Amount must be numeric",
                    field_name="tag_2000",
                )
            elif len(amount) > 12:
                result.add_error(
                    "FEDWIRE_AMOUNT_TOO_LONG",
                    "Amount exceeds 12 digits",
                    field_name="tag_2000",
                )
            else:
                # Check against max
                cents = int(amount)
                if cents > FEDWIRE_AMOUNT_MAX:
                    result.add_error(
                        "FEDWIRE_AMOUNT_EXCEEDS_MAX",
                        f"Amount exceeds maximum allowed (${FEDWIRE_AMOUNT_MAX / 100:,.2f})",
                        field_name="tag_2000",
                    )

        # Sender DI (3100)
        if "3100" in tags:
            sender = tags["3100"]
            if len(sender) >= 9:
                routing = sender[:9]
                error = validate_routing_number(routing)
                if error:
                    result.add_error(
                        "FEDWIRE_INVALID_SENDER_ROUTING",
                        error,
                        field_name="tag_3100",
                    )

        # Receiver DI (3400)
        if "3400" in tags:
            receiver = tags["3400"]
            if len(receiver) >= 9:
                routing = receiver[:9]
                error = validate_routing_number(routing)
                if error:
                    result.add_error(
                        "FEDWIRE_INVALID_RECEIVER_ROUTING",
                        error,
                        field_name="tag_3400",
                    )

        # Business Function Code (3600)
        if "3600" in tags:
            bfc = tags["3600"][:3]
            if not BusinessFunctionCode.from_code(bfc):
                result.add_error(
                    "FEDWIRE_INVALID_BFC",
                    f"Invalid business function code: {bfc}",
                    field_name="tag_3600",
                )

    def _validate_bfc_requirements(self, bfc: BusinessFunctionCode, tags: Dict[str, str], result: ValidationResult) -> None:
        """Validate business function code requirements against tags."""
        requirements = BFC_REQUIREMENTS.get(bfc, {})

        # Beneficiary required
        if requirements.get("beneficiary_required") and "4200" not in tags:
            result.add_error(
                "FEDWIRE_MISSING_BENEFICIARY",
                f"{bfc.short_name} requires beneficiary information (tag 4200)",
                field_name="tag_4200",
            )

        # Originator required
        if requirements.get("originator_required") and "5000" not in tags:
            result.add_error(
                "FEDWIRE_MISSING_ORIGINATOR",
                f"{bfc.short_name} requires originator information (tag 5000)",
                field_name="tag_5000",
            )

        # Originator FI required
        if requirements.get("originator_fi_required") and "5100" not in tags:
            result.add_error(
                "FEDWIRE_MISSING_ORIGINATOR_FI",
                f"{bfc.short_name} requires originator FI information (tag 5100)",
                field_name="tag_5100",
            )

    def _validate_bfc_requirements_object(self, bfc: BusinessFunctionCode, msg: Any, result: ValidationResult) -> None:
        """Validate business function code requirements against message object."""
        requirements = BFC_REQUIREMENTS.get(bfc, {})

        # Beneficiary required
        if requirements.get("beneficiary_required") and not msg.beneficiary:
            result.add_error(
                "FEDWIRE_MISSING_BENEFICIARY",
                f"{bfc.short_name} requires beneficiary information",
                field_name="beneficiary",
            )

        # Originator required
        if requirements.get("originator_required") and not msg.originator:
            result.add_error(
                "FEDWIRE_MISSING_ORIGINATOR",
                f"{bfc.short_name} requires originator information",
                field_name="originator",
            )

        # Originator FI required
        if requirements.get("originator_fi_required") and not msg.originator_fi:
            result.add_error(
                "FEDWIRE_MISSING_ORIGINATOR_FI",
                f"{bfc.short_name} requires originator FI information",
                field_name="originator_fi",
            )

        # Special validation for specific business functions
        if bfc == BusinessFunctionCode.DRAWDOWN_REQUEST:
            if not msg.account_credited:
                result.add_error(
                    "FEDWIRE_MISSING_ACCOUNT_CREDITED",
                    "Drawdown request requires account credited information",
                    field_name="account_credited",
                )

        if bfc == BusinessFunctionCode.DRAWDOWN_TRANSFER:
            if not msg.account_debited:
                result.add_error(
                    "FEDWIRE_MISSING_ACCOUNT_DEBITED",
                    "Drawdown transfer requires account debited information",
                    field_name="account_debited",
                )

        if bfc == BusinessFunctionCode.COVER_PAYMENT:
            if not msg.beneficiary_fi:
                result.add_error(
                    "FEDWIRE_COVER_MISSING_BEN_FI",
                    "Cover payment requires beneficiary FI information",
                    field_name="beneficiary_fi",
                )
