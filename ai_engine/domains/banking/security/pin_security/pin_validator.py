"""
PIN Validator

Validates PINs according to various security standards and policies.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set


class PINValidationSeverity(Enum):
    """Severity of validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class PINValidationIssue:
    """A single validation issue."""

    code: str
    message: str
    severity: PINValidationSeverity = PINValidationSeverity.ERROR


@dataclass
class PINValidationResult:
    """Result of PIN validation."""

    is_valid: bool = True
    issues: List[PINValidationIssue] = field(default_factory=list)

    def add_error(self, code: str, message: str) -> None:
        """Add an error."""
        self.issues.append(
            PINValidationIssue(
                code=code,
                message=message,
                severity=PINValidationSeverity.ERROR,
            )
        )
        self.is_valid = False

    def add_warning(self, code: str, message: str) -> None:
        """Add a warning."""
        self.issues.append(
            PINValidationIssue(
                code=code,
                message=message,
                severity=PINValidationSeverity.WARNING,
            )
        )

    @property
    def errors(self) -> List[PINValidationIssue]:
        """Get errors only."""
        return [i for i in self.issues if i.severity == PINValidationSeverity.ERROR]


class PINValidator:
    """
    PIN Validator.

    Validates PINs against:
    - Length requirements
    - Character requirements
    - Weak PIN patterns
    - PAN-related patterns
    - Sequential/repeated digit patterns
    - Industry blacklists
    """

    # Common weak PINs to reject
    DEFAULT_BLACKLIST: Set[str] = {
        "0000",
        "1111",
        "2222",
        "3333",
        "4444",
        "5555",
        "6666",
        "7777",
        "8888",
        "9999",
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
        "6789",
        "0123",
        "9876",
        "8765",
        "7654",
        "6543",
        "5432",
        "4321",
        "3210",
        "1212",
        "2121",
        "1010",
        "2020",
        "0852",
        "2580",
        "1470",
        "0741",  # Keypad patterns
        "1379",
        "9731",
        "7913",
        "3197",  # More keypad patterns
        "1478",
        "8741",
        "2569",
        "9652",
        "1357",
        "7531",
        "2468",
        "8642",
    }

    def __init__(
        self,
        min_length: int = 4,
        max_length: int = 12,
        require_numeric: bool = True,
        check_weak_patterns: bool = True,
        check_pan_patterns: bool = True,
        custom_blacklist: Optional[Set[str]] = None,
    ):
        """
        Initialize PIN validator.

        Args:
            min_length: Minimum PIN length (default 4)
            max_length: Maximum PIN length (default 12)
            require_numeric: Require numeric-only PINs
            check_weak_patterns: Check for weak patterns
            check_pan_patterns: Check for PAN-related patterns
            custom_blacklist: Additional PINs to reject
        """
        self.min_length = min_length
        self.max_length = max_length
        self.require_numeric = require_numeric
        self.check_weak_patterns = check_weak_patterns
        self.check_pan_patterns = check_pan_patterns

        self.blacklist = self.DEFAULT_BLACKLIST.copy()
        if custom_blacklist:
            self.blacklist.update(custom_blacklist)

    def validate(
        self,
        pin: str,
        pan: Optional[str] = None,
        date_of_birth: Optional[str] = None,
    ) -> PINValidationResult:
        """
        Validate a PIN.

        Args:
            pin: PIN to validate
            pan: Optional PAN for pattern checking
            date_of_birth: Optional DOB for pattern checking (YYYYMMDD)

        Returns:
            Validation result
        """
        result = PINValidationResult()

        # Length check
        if len(pin) < self.min_length:
            result.add_error(
                "PIN_TOO_SHORT",
                f"PIN must be at least {self.min_length} digits",
            )

        if len(pin) > self.max_length:
            result.add_error(
                "PIN_TOO_LONG",
                f"PIN must be at most {self.max_length} digits",
            )

        # Numeric check
        if self.require_numeric and not pin.isdigit():
            result.add_error(
                "PIN_NOT_NUMERIC",
                "PIN must contain only digits",
            )

        # Weak pattern checks
        if self.check_weak_patterns:
            self._check_weak_patterns(pin, result)

        # PAN-related pattern checks
        if self.check_pan_patterns and pan:
            self._check_pan_patterns(pin, pan, result)

        # DOB pattern check
        if date_of_birth:
            self._check_dob_patterns(pin, date_of_birth, result)

        # Blacklist check
        if pin in self.blacklist:
            result.add_error(
                "PIN_BLACKLISTED",
                "PIN is on the weak PIN blacklist",
            )

        return result

    def _check_weak_patterns(self, pin: str, result: PINValidationResult) -> None:
        """Check for weak PIN patterns."""
        if not pin.isdigit():
            return

        # All same digits
        if len(set(pin)) == 1:
            result.add_error(
                "PIN_ALL_SAME",
                "PIN cannot be all same digits",
            )
            return

        # Ascending sequence
        digits = [int(d) for d in pin]
        if all(digits[i] == digits[i - 1] + 1 for i in range(1, len(digits))):
            result.add_error(
                "PIN_ASCENDING",
                "PIN cannot be ascending sequence",
            )
            return

        # Descending sequence
        if all(digits[i] == digits[i - 1] - 1 for i in range(1, len(digits))):
            result.add_error(
                "PIN_DESCENDING",
                "PIN cannot be descending sequence",
            )
            return

        # Two-digit repeating pattern
        if len(pin) >= 4 and len(pin) % 2 == 0:
            pair = pin[:2]
            if pin == pair * (len(pin) // 2):
                result.add_error(
                    "PIN_REPEATING_PAIR",
                    "PIN cannot be repeating pair pattern",
                )
                return

        # High repetition (more than half are same digit)
        digit_counts = {}
        for d in pin:
            digit_counts[d] = digit_counts.get(d, 0) + 1

        max_count = max(digit_counts.values())
        if max_count > len(pin) // 2:
            result.add_warning(
                "PIN_HIGH_REPETITION",
                "PIN has high digit repetition",
            )

    def _check_pan_patterns(self, pin: str, pan: str, result: PINValidationResult) -> None:
        """Check for PAN-related patterns."""
        # Remove spaces and dashes from PAN
        clean_pan = pan.replace(" ", "").replace("-", "")

        if not clean_pan.isdigit():
            return

        # PIN matches any 4 consecutive digits in PAN
        for i in range(len(clean_pan) - len(pin) + 1):
            if clean_pan[i : i + len(pin)] == pin:
                result.add_error(
                    "PIN_MATCHES_PAN",
                    "PIN cannot match digits from card number",
                )
                return

        # PIN is reverse of any 4 consecutive digits in PAN
        reversed_pin = pin[::-1]
        for i in range(len(clean_pan) - len(pin) + 1):
            if clean_pan[i : i + len(pin)] == reversed_pin:
                result.add_error(
                    "PIN_MATCHES_PAN_REVERSE",
                    "PIN cannot be reverse of card number digits",
                )
                return

        # PIN matches last 4 digits
        if len(clean_pan) >= 4:
            if pin == clean_pan[-4:]:
                result.add_error(
                    "PIN_MATCHES_PAN_LAST4",
                    "PIN cannot match last 4 digits of card",
                )
                return

        # PIN matches first 4 digits
        if len(clean_pan) >= 4:
            if pin == clean_pan[:4]:
                result.add_error(
                    "PIN_MATCHES_PAN_FIRST4",
                    "PIN cannot match first 4 digits of card",
                )
                return

    def _check_dob_patterns(
        self,
        pin: str,
        dob: str,
        result: PINValidationResult,
    ) -> None:
        """Check for date-of-birth related patterns."""
        # Expected format: YYYYMMDD
        if len(dob) < 8:
            return

        # Extract common patterns
        year = dob[:4]
        month = dob[4:6]
        day = dob[6:8]
        year_short = dob[2:4]

        patterns_to_check = [
            year,  # Full year
            year_short + month + day,  # YYMMDD -> 6 digits
            month + day + year_short,  # MMDDYY
            day + month + year_short,  # DDMMYY
            month + day,  # MMDD
            day + month,  # DDMM
            year_short + month,  # YYMM
            month + year_short,  # MMYY
        ]

        for pattern in patterns_to_check:
            if len(pattern) == len(pin) and pattern == pin:
                result.add_error(
                    "PIN_MATCHES_DOB",
                    "PIN cannot be derived from date of birth",
                )
                return

    def is_weak(self, pin: str) -> bool:
        """Quick check if PIN is weak."""
        result = self.validate(pin)
        return not result.is_valid
