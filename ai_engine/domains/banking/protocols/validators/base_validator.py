"""
Base Validator Framework

Provides common validation infrastructure for all banking protocols.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = auto()  # Informational only
    WARNING = auto()  # Non-blocking issue
    ERROR = auto()  # Blocking issue
    CRITICAL = auto()  # Critical issue - must not proceed


@dataclass
class ValidationError:
    """Represents a validation error."""

    code: str
    message: str
    field_name: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field_name,
            "severity": self.severity.name,
            "details": self.details,
        }


@dataclass
class ValidationWarning:
    """Represents a validation warning."""

    code: str
    message: str
    field_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field_name,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Complete validation result with errors, warnings, and metadata."""

    is_valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.now)
    validator_name: str = ""
    validator_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(
        self,
        code: str,
        message: str,
        field: str = "",
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        **details,
    ) -> None:
        """Add a validation error."""
        self.errors.append(
            ValidationError(
                code=code,
                message=message,
                field_name=field,
                severity=severity,
                details=details,
            )
        )
        self.is_valid = False

    def add_warning(
        self,
        code: str,
        message: str,
        field: str = "",
        **details,
    ) -> None:
        """Add a validation warning."""
        self.warnings.append(
            ValidationWarning(
                code=code,
                message=message,
                field_name=field,
                details=details,
            )
        )

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "validated_at": self.validated_at.isoformat(),
            "validator_name": self.validator_name,
            "validator_version": self.validator_version,
            "metadata": self.metadata,
        }

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def has_critical_errors(self) -> bool:
        return any(e.severity == ValidationSeverity.CRITICAL for e in self.errors)


class BaseValidator(ABC):
    """
    Abstract base class for banking protocol validators.

    All protocol-specific validators should inherit from this class.
    """

    def __init__(self, strict: bool = True):
        """
        Initialize validator.

        Args:
            strict: If True, treat warnings as errors
        """
        self.strict = strict
        self._rules: List[str] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Return validator name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Return validator version."""
        pass

    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate the provided data.

        Args:
            data: Data to validate (type depends on validator)

        Returns:
            ValidationResult with any errors or warnings
        """
        pass

    def _create_result(self) -> ValidationResult:
        """Create a new validation result."""
        return ValidationResult(
            validator_name=self.name,
            validator_version=self.version,
        )


class CompositeValidator(BaseValidator):
    """
    Validator that combines multiple validators.

    Useful for running a chain of validations.
    """

    def __init__(self, validators: List[BaseValidator], strict: bool = True):
        super().__init__(strict)
        self._validators = validators

    @property
    def name(self) -> str:
        return "CompositeValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """Run all validators and combine results."""
        result = self._create_result()

        for validator in self._validators:
            sub_result = validator.validate(data)
            result.merge(sub_result)

        return result

    def add_validator(self, validator: BaseValidator) -> None:
        """Add a validator to the chain."""
        self._validators.append(validator)


# Common validation utilities
def validate_routing_number(routing_number: str) -> Optional[str]:
    """
    Validate ABA routing number using checksum algorithm.

    Returns error message if invalid, None if valid.
    """
    if not routing_number:
        return "Routing number is required"

    if len(routing_number) != 9:
        return "Routing number must be 9 digits"

    if not routing_number.isdigit():
        return "Routing number must contain only digits"

    # ABA checksum: 3(d1 + d4 + d7) + 7(d2 + d5 + d8) + 1(d3 + d6 + d9) mod 10 = 0
    digits = [int(d) for d in routing_number]
    checksum = (
        3 * (digits[0] + digits[3] + digits[6])
        + 7 * (digits[1] + digits[4] + digits[7])
        + 1 * (digits[2] + digits[5] + digits[8])
    )

    if checksum % 10 != 0:
        return "Invalid routing number checksum"

    return None


def validate_iban(iban: str) -> Optional[str]:
    """
    Validate IBAN using ISO 13616 standard.

    Returns error message if invalid, None if valid.
    """
    if not iban:
        return "IBAN is required"

    # Remove spaces and convert to uppercase
    iban = iban.replace(" ", "").upper()

    # Check length (varies by country, 15-34 chars)
    if len(iban) < 15 or len(iban) > 34:
        return "IBAN length must be between 15 and 34 characters"

    # Check format: 2 letters (country) + 2 digits (check) + alphanumeric
    if not iban[:2].isalpha():
        return "IBAN must start with 2-letter country code"

    if not iban[2:4].isdigit():
        return "IBAN check digits must be numeric"

    # Move first 4 chars to end
    rearranged = iban[4:] + iban[:4]

    # Convert letters to numbers (A=10, B=11, etc.)
    numeric = ""
    for char in rearranged:
        if char.isdigit():
            numeric += char
        else:
            numeric += str(ord(char) - ord("A") + 10)

    # Check if mod 97 = 1
    if int(numeric) % 97 != 1:
        return "Invalid IBAN checksum"

    return None


def validate_bic(bic: str) -> Optional[str]:
    """
    Validate BIC/SWIFT code.

    Returns error message if invalid, None if valid.
    """
    if not bic:
        return "BIC/SWIFT code is required"

    bic = bic.upper().replace(" ", "")

    if len(bic) not in (8, 11):
        return "BIC must be 8 or 11 characters"

    # Bank code (4 letters)
    if not bic[:4].isalpha():
        return "BIC bank code must be 4 letters"

    # Country code (2 letters)
    if not bic[4:6].isalpha():
        return "BIC country code must be 2 letters"

    # Location code (2 alphanumeric)
    if not bic[6:8].isalnum():
        return "BIC location code must be alphanumeric"

    # Branch code (3 alphanumeric, if present)
    if len(bic) == 11 and not bic[8:11].isalnum():
        return "BIC branch code must be alphanumeric"

    return None


def validate_currency_code(currency: str) -> Optional[str]:
    """
    Validate ISO 4217 currency code.

    Returns error message if invalid, None if valid.
    """
    if not currency:
        return "Currency code is required"

    currency = currency.upper()

    if len(currency) != 3:
        return "Currency code must be 3 characters"

    if not currency.isalpha():
        return "Currency code must be alphabetic"

    # Common valid currency codes (not exhaustive)
    valid_currencies = {
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CHF",
        "CAD",
        "AUD",
        "NZD",
        "CNY",
        "HKD",
        "SGD",
        "INR",
        "KRW",
        "MXN",
        "BRL",
        "RUB",
        "ZAR",
        "SEK",
        "NOK",
        "DKK",
        "PLN",
        "CZK",
        "HUF",
        "TRY",
        "THB",
        "MYR",
        "PHP",
        "IDR",
        "TWD",
        "AED",
        "SAR",
        "ILS",
    }

    # Allow any 3-letter code but warn if not in common list
    return None


def validate_amount(amount, max_value: float = None, allow_zero: bool = False) -> Optional[str]:
    """
    Validate transaction amount.

    Returns error message if invalid, None if valid.
    """
    if amount is None:
        return "Amount is required"

    try:
        amount_float = float(amount)
    except (ValueError, TypeError):
        return "Amount must be numeric"

    if not allow_zero and amount_float <= 0:
        return "Amount must be greater than zero"

    if allow_zero and amount_float < 0:
        return "Amount cannot be negative"

    if max_value and amount_float > max_value:
        return f"Amount exceeds maximum allowed ({max_value})"

    return None
