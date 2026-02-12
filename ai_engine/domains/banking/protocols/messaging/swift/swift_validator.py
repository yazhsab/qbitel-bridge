"""
SWIFT Message Validator

Validates SWIFT MT messages including:
- Structure validation
- Field format validation
- Character set validation
- Mandatory field checks
- Cross-field validation
"""

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
    validate_bic,
)
from ai_engine.domains.banking.protocols.messaging.swift.swift_message import (
    SwiftMessage,
    SwiftField,
    MessageDirection,
)
from ai_engine.domains.banking.protocols.messaging.swift.swift_codes import (
    SwiftMessageType,
    SwiftCharacterSet,
    MT103_FIELDS,
    MT202_FIELDS,
    BANK_OPERATION_CODES,
    CHARGE_CODES,
    INSTRUCTION_CODES,
    get_field_definition,
)
from ai_engine.domains.banking.protocols.messaging.swift.mt103 import MT103Message
from ai_engine.domains.banking.protocols.messaging.swift.mt202 import MT202Message


class SwiftValidator(BaseValidator):
    """
    Validator for SWIFT MT messages.

    Validates:
    - Message structure (blocks 1-5)
    - Field presence and format
    - Character set compliance
    - BIC validity
    - Amount formats
    - Date formats
    - Cross-field dependencies
    """

    # SWIFT character sets
    SWIFT_X_PATTERN = re.compile(r"^[A-Za-z0-9/\-?:().,'+ \n]*$")
    SWIFT_Z_PATTERN = re.compile(r"^[A-Za-z0-9/\-?:().,'+ \n\r]*$")
    NUMERIC_PATTERN = re.compile(r"^\d+$")
    AMOUNT_PATTERN = re.compile(r"^\d+,\d{0,2}$")
    DATE_PATTERN = re.compile(r"^\d{6}$")
    BIC_PATTERN = re.compile(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$")

    def __init__(
        self,
        strict: bool = True,
        validate_bics: bool = True,
        validate_amounts: bool = True,
    ):
        """
        Initialize the SWIFT validator.

        Args:
            strict: If True, treat warnings as errors
            validate_bics: If True, validate BIC checksums
            validate_amounts: If True, validate amount formats
        """
        super().__init__(strict)
        self.validate_bics = validate_bics
        self.validate_amounts = validate_amounts

    @property
    def name(self) -> str:
        return "SwiftValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate a SWIFT message.

        Args:
            data: SwiftMessage, MT103Message, MT202Message, or dict

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, MT103Message):
            self._validate_mt103(data, result)
        elif isinstance(data, MT202Message):
            self._validate_mt202(data, result)
        elif isinstance(data, SwiftMessage):
            self._validate_swift_message(data, result)
        elif isinstance(data, dict):
            self._validate_dict(data, result)
        else:
            result.add_error(
                "SWIFT_INVALID_INPUT",
                "Input must be a SWIFT message object or dictionary",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def _validate_swift_message(self, msg: SwiftMessage, result: ValidationResult) -> None:
        """Validate a SwiftMessage object."""
        # Validate block 1 (Basic Header)
        if not msg.basic_header.content:
            result.add_error(
                "SWIFT_MISSING_BLOCK1",
                "Basic header (block 1) is required",
                severity=ValidationSeverity.CRITICAL,
            )
        else:
            self._validate_basic_header(msg, result)

        # Validate block 2 (Application Header)
        if not msg.application_header.content:
            result.add_error(
                "SWIFT_MISSING_BLOCK2",
                "Application header (block 2) is required",
                severity=ValidationSeverity.CRITICAL,
            )
        else:
            self._validate_application_header(msg, result)

        # Validate block 4 (Text Block)
        if not msg.text_block.fields:
            result.add_error(
                "SWIFT_MISSING_BLOCK4",
                "Text block (block 4) is required",
                severity=ValidationSeverity.CRITICAL,
            )
        else:
            self._validate_text_block(msg, result)

        # Message-type specific validation
        mt = msg.message_type
        if mt == "103" or mt.startswith("103"):
            self._validate_mt103_fields(msg, result)
        elif mt == "202" or mt.startswith("202"):
            self._validate_mt202_fields(msg, result)

    def _validate_basic_header(self, msg: SwiftMessage, result: ValidationResult) -> None:
        """Validate basic header (block 1)."""
        header = msg.basic_header

        # Application ID
        if header.application_id not in ("F", "A", "L"):
            result.add_error(
                "SWIFT_INVALID_APP_ID",
                f"Invalid application ID: {header.application_id}",
                field="block1/application_id",
            )

        # Service ID
        if header.service_id not in ("01", "21"):
            result.add_error(
                "SWIFT_INVALID_SERVICE_ID",
                f"Invalid service ID: {header.service_id}",
                field="block1/service_id",
            )

        # LT Address (BIC + terminal)
        if header.lt_address:
            bic = header.lt_address[:8] if len(header.lt_address) >= 8 else header.lt_address
            if self.validate_bics:
                error = validate_bic(bic)
                if error:
                    result.add_error(
                        "SWIFT_INVALID_SENDER_BIC",
                        f"Invalid sender BIC: {error}",
                        field="block1/lt_address",
                    )

    def _validate_application_header(self, msg: SwiftMessage, result: ValidationResult) -> None:
        """Validate application header (block 2)."""
        header = msg.application_header

        # Message type
        if not header.message_type:
            result.add_error(
                "SWIFT_MISSING_MSG_TYPE",
                "Message type is required",
                field="block2/message_type",
            )
        elif not header.message_type.isdigit() or len(header.message_type) != 3:
            result.add_error(
                "SWIFT_INVALID_MSG_TYPE",
                f"Invalid message type format: {header.message_type}",
                field="block2/message_type",
            )

        # Priority
        if header.priority not in ("N", "S", "U"):
            result.add_error(
                "SWIFT_INVALID_PRIORITY",
                f"Invalid priority: {header.priority}",
                field="block2/priority",
            )

        # For input messages, validate destination BIC
        if header.direction == MessageDirection.INPUT:
            if header.destination_bic and self.validate_bics:
                bic = header.destination_bic[:8] if len(header.destination_bic) >= 8 else header.destination_bic
                error = validate_bic(bic)
                if error:
                    result.add_error(
                        "SWIFT_INVALID_RECEIVER_BIC",
                        f"Invalid receiver BIC: {error}",
                        field="block2/destination_bic",
                    )

    def _validate_text_block(self, msg: SwiftMessage, result: ValidationResult) -> None:
        """Validate text block (block 4)."""
        for field in msg.text_block.fields:
            self._validate_field(field, msg.message_type, result)

    def _validate_field(self, field: SwiftField, message_type: str, result: ValidationResult) -> None:
        """Validate a single field."""
        tag = field.full_tag
        value = field.value

        # Check character set
        if not self.SWIFT_Z_PATTERN.match(value):
            # Find invalid characters
            invalid_chars = set()
            for char in value:
                if not self.SWIFT_Z_PATTERN.match(char):
                    invalid_chars.add(char)
            result.add_error(
                "SWIFT_INVALID_CHARS",
                f"Field {tag} contains invalid characters: {invalid_chars}",
                field=f"field_{tag}",
            )

        # Check field length
        field_def = get_field_definition(message_type, field.tag)
        if field_def:
            # Parse format string for max length
            max_length = self._get_max_field_length(field_def.format)
            if max_length and len(value.replace("\n", "")) > max_length:
                result.add_warning(
                    "SWIFT_FIELD_TOO_LONG",
                    f"Field {tag} exceeds maximum length ({len(value)} > {max_length})",
                    field=f"field_{tag}",
                )

        # Validate specific field formats
        if field.tag == "32A":
            self._validate_field_32a(value, result)
        elif field.tag == "33B":
            self._validate_field_33b(value, result)
        elif field.tag in ("50", "59") and self.validate_bics:
            self._validate_party_field(field, result)
        elif field.tag in ("52", "53", "54", "55", "56", "57", "58"):
            self._validate_agent_field(field, result)

    def _validate_field_32a(self, value: str, result: ValidationResult) -> None:
        """Validate field 32A (Value Date/Currency/Amount)."""
        if len(value) < 12:
            result.add_error(
                "SWIFT_INVALID_32A",
                "Field 32A must be at least 12 characters",
                field="field_32A",
            )
            return

        date_str = value[:6]
        currency = value[6:9]
        amount_str = value[9:]

        # Validate date
        if not self.DATE_PATTERN.match(date_str):
            result.add_error(
                "SWIFT_INVALID_DATE",
                f"Invalid date format in 32A: {date_str}",
                field="field_32A",
            )
        else:
            try:
                year = int(date_str[:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                year = 2000 + year if year < 50 else 1900 + year
                date(year, month, day)
            except ValueError as e:
                result.add_error(
                    "SWIFT_INVALID_DATE",
                    f"Invalid date in 32A: {str(e)}",
                    field="field_32A",
                )

        # Validate currency
        if not currency.isalpha() or len(currency) != 3:
            result.add_error(
                "SWIFT_INVALID_CURRENCY",
                f"Invalid currency in 32A: {currency}",
                field="field_32A",
            )

        # Validate amount
        if self.validate_amounts:
            if not self.AMOUNT_PATTERN.match(amount_str) and not re.match(r"^\d+$", amount_str):
                result.add_error(
                    "SWIFT_INVALID_AMOUNT",
                    f"Invalid amount format in 32A: {amount_str}",
                    field="field_32A",
                )

    def _validate_field_33b(self, value: str, result: ValidationResult) -> None:
        """Validate field 33B (Currency/Instructed Amount)."""
        if len(value) < 4:
            result.add_error(
                "SWIFT_INVALID_33B",
                "Field 33B must be at least 4 characters",
                field="field_33B",
            )
            return

        currency = value[:3]
        amount_str = value[3:]

        # Validate currency
        if not currency.isalpha() or len(currency) != 3:
            result.add_error(
                "SWIFT_INVALID_CURRENCY",
                f"Invalid currency in 33B: {currency}",
                field="field_33B",
            )

        # Validate amount
        if self.validate_amounts:
            if not self.AMOUNT_PATTERN.match(amount_str) and not re.match(r"^\d+$", amount_str):
                result.add_error(
                    "SWIFT_INVALID_AMOUNT",
                    f"Invalid amount format in 33B: {amount_str}",
                    field="field_33B",
                )

    def _validate_party_field(self, field: SwiftField, result: ValidationResult) -> None:
        """Validate party fields (50, 59)."""
        lines = field.value.split("\n")

        for line in lines:
            # Check for BIC
            if len(line) == 8 or len(line) == 11:
                if self.BIC_PATTERN.match(line) and self.validate_bics:
                    error = validate_bic(line)
                    if error:
                        result.add_error(
                            "SWIFT_INVALID_BIC",
                            f"Invalid BIC in field {field.full_tag}: {error}",
                            field=f"field_{field.full_tag}",
                        )

    def _validate_agent_field(self, field: SwiftField, result: ValidationResult) -> None:
        """Validate agent fields (52-58)."""
        lines = field.value.split("\n")

        for line in lines:
            # Skip account lines
            if line.startswith("/"):
                continue

            # Check for BIC
            if len(line) == 8 or len(line) == 11:
                if self.BIC_PATTERN.match(line) and self.validate_bics:
                    error = validate_bic(line)
                    if error:
                        result.add_error(
                            "SWIFT_INVALID_BIC",
                            f"Invalid BIC in field {field.full_tag}: {error}",
                            field=f"field_{field.full_tag}",
                        )

    def _validate_mt103_fields(self, msg: SwiftMessage, result: ValidationResult) -> None:
        """Validate MT103-specific fields."""
        mandatory_fields = {"20", "23B", "32A", "59", "71A"}
        present_fields = {f.tag for f in msg.text_block.fields}

        # Check mandatory fields
        for tag in mandatory_fields:
            if tag not in present_fields:
                # Check with qualifier (e.g., 59A, 59F)
                has_variant = any(f.tag == tag for f in msg.text_block.fields)
                if not has_variant:
                    result.add_error(
                        "SWIFT_MISSING_FIELD",
                        f"Mandatory field {tag} is missing",
                        field=f"field_{tag}",
                    )

        # Check ordering customer (50A/50F/50K)
        has_50 = any(f.tag == "50" for f in msg.text_block.fields)
        if not has_50:
            result.add_error(
                "SWIFT_MISSING_FIELD",
                "Ordering customer (field 50) is required",
                field="field_50",
            )

        # Validate bank operation code
        boc_field = msg.text_block.get_field("23B")
        if boc_field and boc_field.value not in BANK_OPERATION_CODES:
            result.add_warning(
                "SWIFT_INVALID_BOC",
                f"Unknown bank operation code: {boc_field.value}",
                field="field_23B",
            )

        # Validate charge details
        charge_field = msg.text_block.get_field("71A")
        if charge_field and charge_field.value not in CHARGE_CODES:
            result.add_error(
                "SWIFT_INVALID_CHARGES",
                f"Invalid charge details: {charge_field.value}",
                field="field_71A",
            )

        # Validate instruction codes
        for field in msg.text_block.get_fields("23E"):
            code = field.value.split("/")[0] if "/" in field.value else field.value
            if code not in INSTRUCTION_CODES:
                result.add_warning(
                    "SWIFT_UNKNOWN_INSTRUCTION",
                    f"Unknown instruction code: {code}",
                    field="field_23E",
                )

    def _validate_mt202_fields(self, msg: SwiftMessage, result: ValidationResult) -> None:
        """Validate MT202-specific fields."""
        mandatory_fields = {"20", "21", "32A", "58"}
        present_fields = {f.tag for f in msg.text_block.fields}

        # Check mandatory fields
        for tag in mandatory_fields:
            if tag not in present_fields:
                # Check with qualifier
                has_variant = any(f.tag == tag for f in msg.text_block.fields)
                if not has_variant:
                    result.add_error(
                        "SWIFT_MISSING_FIELD",
                        f"Mandatory field {tag} is missing",
                        field=f"field_{tag}",
                    )

        # For MT202 COV, validate sequence B
        if msg.user_header and msg.user_header.fields.get("119") == "COV":
            self._validate_mt202_cov(msg, result)

    def _validate_mt202_cov(self, msg: SwiftMessage, result: ValidationResult) -> None:
        """Validate MT202 COV specific requirements."""
        # Sequence B mandatory fields for COV
        fields = msg.text_block.fields
        field_tags = [f.full_tag for f in fields]

        # Count occurrences of key fields
        field_50_count = sum(1 for f in fields if f.tag == "50")
        field_59_count = sum(1 for f in fields if f.tag == "59")

        # COV requires duplicate 50 and 59 fields (sequence A and B)
        if field_50_count < 1:
            result.add_warning(
                "SWIFT_COV_MISSING_ORDERING",
                "MT202 COV should have ordering customer in sequence B",
                field="field_50",
            )

        if field_59_count < 1:
            result.add_warning(
                "SWIFT_COV_MISSING_BENEFICIARY",
                "MT202 COV should have beneficiary in sequence B",
                field="field_59",
            )

    def _validate_mt103(self, msg: MT103Message, result: ValidationResult) -> None:
        """Validate MT103Message object."""
        # Required fields
        if not msg.sender_reference:
            result.add_error(
                "SWIFT_MISSING_REFERENCE",
                "Sender's reference (field 20) is required",
                field="sender_reference",
            )
        elif len(msg.sender_reference) > 16:
            result.add_error(
                "SWIFT_INVALID_REFERENCE",
                "Sender's reference must not exceed 16 characters",
                field="sender_reference",
            )

        if not msg.bank_operation_code:
            result.add_error(
                "SWIFT_MISSING_BOC",
                "Bank operation code (field 23B) is required",
                field="bank_operation_code",
            )
        elif msg.bank_operation_code not in BANK_OPERATION_CODES:
            result.add_warning(
                "SWIFT_INVALID_BOC",
                f"Unknown bank operation code: {msg.bank_operation_code}",
                field="bank_operation_code",
            )

        if not msg.value_date:
            result.add_error(
                "SWIFT_MISSING_VALUE_DATE",
                "Value date (field 32A) is required",
                field="value_date",
            )

        if msg.amount <= 0:
            result.add_error(
                "SWIFT_INVALID_AMOUNT",
                "Amount must be greater than zero",
                field="amount",
            )

        if not msg.currency or len(msg.currency) != 3:
            result.add_error(
                "SWIFT_INVALID_CURRENCY",
                "Valid 3-character currency code is required",
                field="currency",
            )

        # Ordering customer
        if not msg.ordering_customer.name and not msg.ordering_customer.bic:
            result.add_error(
                "SWIFT_MISSING_ORDERING_CUSTOMER",
                "Ordering customer (field 50) is required",
                field="ordering_customer",
            )

        # Beneficiary
        if not msg.beneficiary.name and not msg.beneficiary.bic:
            result.add_error(
                "SWIFT_MISSING_BENEFICIARY",
                "Beneficiary (field 59) is required",
                field="beneficiary",
            )

        # Charges
        if msg.charges.details not in CHARGE_CODES:
            result.add_error(
                "SWIFT_INVALID_CHARGES",
                f"Invalid charge details: {msg.charges.details}",
                field="charges/details",
            )

        # BIC validation
        if self.validate_bics:
            if msg.sender_bic:
                error = validate_bic(msg.sender_bic)
                if error:
                    result.add_error(
                        "SWIFT_INVALID_SENDER_BIC",
                        f"Invalid sender BIC: {error}",
                        field="sender_bic",
                    )

            if msg.receiver_bic:
                error = validate_bic(msg.receiver_bic)
                if error:
                    result.add_error(
                        "SWIFT_INVALID_RECEIVER_BIC",
                        f"Invalid receiver BIC: {error}",
                        field="receiver_bic",
                    )

    def _validate_mt202(self, msg: MT202Message, result: ValidationResult) -> None:
        """Validate MT202Message object."""
        # Required fields
        if not msg.transaction_reference:
            result.add_error(
                "SWIFT_MISSING_REFERENCE",
                "Transaction reference (field 20) is required",
                field="transaction_reference",
            )
        elif len(msg.transaction_reference) > 16:
            result.add_error(
                "SWIFT_INVALID_REFERENCE",
                "Transaction reference must not exceed 16 characters",
                field="transaction_reference",
            )

        if not msg.related_reference:
            result.add_error(
                "SWIFT_MISSING_RELATED_REF",
                "Related reference (field 21) is required",
                field="related_reference",
            )
        elif len(msg.related_reference) > 16:
            result.add_error(
                "SWIFT_INVALID_RELATED_REF",
                "Related reference must not exceed 16 characters",
                field="related_reference",
            )

        if not msg.value_date:
            result.add_error(
                "SWIFT_MISSING_VALUE_DATE",
                "Value date (field 32A) is required",
                field="value_date",
            )

        if msg.amount <= 0:
            result.add_error(
                "SWIFT_INVALID_AMOUNT",
                "Amount must be greater than zero",
                field="amount",
            )

        if not msg.currency or len(msg.currency) != 3:
            result.add_error(
                "SWIFT_INVALID_CURRENCY",
                "Valid 3-character currency code is required",
                field="currency",
            )

        # Beneficiary institution
        if not msg.beneficiary_institution.bic and not msg.beneficiary_institution.name:
            result.add_error(
                "SWIFT_MISSING_BENEFICIARY_INST",
                "Beneficiary institution (field 58) is required",
                field="beneficiary_institution",
            )

        # BIC validation
        if self.validate_bics:
            if msg.sender_bic:
                error = validate_bic(msg.sender_bic)
                if error:
                    result.add_error(
                        "SWIFT_INVALID_SENDER_BIC",
                        f"Invalid sender BIC: {error}",
                        field="sender_bic",
                    )

            if msg.receiver_bic:
                error = validate_bic(msg.receiver_bic)
                if error:
                    result.add_error(
                        "SWIFT_INVALID_RECEIVER_BIC",
                        f"Invalid receiver BIC: {error}",
                        field="receiver_bic",
                    )

            if msg.beneficiary_institution.bic:
                error = validate_bic(msg.beneficiary_institution.bic)
                if error:
                    result.add_error(
                        "SWIFT_INVALID_BENEFICIARY_BIC",
                        f"Invalid beneficiary institution BIC: {error}",
                        field="beneficiary_institution/bic",
                    )

        # COV-specific validation
        if msg.is_cov:
            if not msg.cover_info:
                result.add_error(
                    "SWIFT_MISSING_COVER_INFO",
                    "Cover information is required for MT202 COV",
                    field="cover_info",
                )
            else:
                if not msg.cover_info.ordering_customer_name:
                    result.add_warning(
                        "SWIFT_COV_MISSING_ORDERING",
                        "Ordering customer name is recommended in cover info",
                        field="cover_info/ordering_customer_name",
                    )
                if not msg.cover_info.beneficiary_name:
                    result.add_warning(
                        "SWIFT_COV_MISSING_BENEFICIARY",
                        "Beneficiary name is recommended in cover info",
                        field="cover_info/beneficiary_name",
                    )

    def _validate_dict(self, data: Dict, result: ValidationResult) -> None:
        """Validate SWIFT data from dictionary."""
        message_type = data.get("message_type", "")

        # Basic validation
        if "reference" in data or "sender_reference" in data:
            ref = data.get("reference") or data.get("sender_reference")
            if ref and len(ref) > 16:
                result.add_error(
                    "SWIFT_INVALID_REFERENCE",
                    "Reference must not exceed 16 characters",
                    field="reference",
                )

        if "amount" in data:
            try:
                amount = Decimal(str(data["amount"]))
                if amount <= 0:
                    result.add_error(
                        "SWIFT_INVALID_AMOUNT",
                        "Amount must be greater than zero",
                        field="amount",
                    )
            except Exception:
                result.add_error(
                    "SWIFT_INVALID_AMOUNT",
                    "Amount must be a valid number",
                    field="amount",
                )

        if "currency" in data:
            currency = data["currency"]
            if not currency or len(currency) != 3 or not currency.isalpha():
                result.add_error(
                    "SWIFT_INVALID_CURRENCY",
                    "Currency must be a 3-character alphabetic code",
                    field="currency",
                )

        # BIC validation
        if self.validate_bics:
            for field in ["sender_bic", "receiver_bic", "beneficiary_bic"]:
                if field in data and data[field]:
                    error = validate_bic(data[field])
                    if error:
                        result.add_error(
                            "SWIFT_INVALID_BIC",
                            f"Invalid {field}: {error}",
                            field=field,
                        )

    def _get_max_field_length(self, format_str: str) -> Optional[int]:
        """Extract maximum field length from format string."""
        # Parse format strings like "16x", "4*35x", "6!n3!a15d"
        import re

        # Find numeric patterns
        matches = re.findall(r"(\d+)(?:[*!])?[a-z]", format_str)
        if matches:
            # For patterns like "4*35x", multiply
            if "*" in format_str:
                parts = format_str.split("*")
                if len(parts) == 2:
                    try:
                        lines = int(re.search(r"(\d+)", parts[0]).group(1))
                        chars = int(re.search(r"(\d+)", parts[1]).group(1))
                        return lines * chars
                    except (AttributeError, ValueError):
                        pass

            # Sum all numeric parts
            return sum(int(m) for m in matches)

        return None
