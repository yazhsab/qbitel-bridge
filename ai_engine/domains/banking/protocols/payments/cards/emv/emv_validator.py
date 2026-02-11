"""
EMV Validator

Validates EMV transactions including:
- Card data validation
- Terminal data validation
- Cryptogram validation
- TVR/TSI analysis
"""

from datetime import date
from typing import Any, Dict, List, Optional

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
)
from ai_engine.domains.banking.protocols.payments.cards.emv.emv_message import (
    EmvTransaction,
    EmvCard,
    EmvTerminal,
    EmvCryptogram,
)
from ai_engine.domains.banking.protocols.payments.cards.emv.emv_codes import (
    EmvApplication,
    EmvTransactionType,
    EmvCryptogramType,
    TVR_BITS,
    TSI_BITS,
    AIP_BITS,
)
from ai_engine.domains.banking.protocols.payments.cards.emv.emv_tlv import (
    parse_tlv,
)


class EmvValidator(BaseValidator):
    """
    Validator for EMV transactions.

    Validates:
    - Card data (PAN, expiration, AID)
    - Terminal configuration
    - Cryptogram data
    - TVR/TSI flags
    - Business rules
    """

    def __init__(
        self,
        strict: bool = True,
        validate_expiry: bool = True,
        check_luhn: bool = True,
    ):
        """
        Initialize the EMV validator.

        Args:
            strict: If True, treat warnings as errors
            validate_expiry: If True, validate card expiration
            check_luhn: If True, validate PAN with Luhn check
        """
        super().__init__(strict)
        self.validate_expiry = validate_expiry
        self.check_luhn = check_luhn

    @property
    def name(self) -> str:
        return "EmvValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate EMV data.

        Args:
            data: EmvTransaction, EmvCard, or dict

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, EmvTransaction):
            self._validate_transaction(data, result)
        elif isinstance(data, EmvCard):
            self._validate_card(data, result)
        elif isinstance(data, dict):
            self._validate_dict(data, result)
        else:
            result.add_error(
                "EMV_INVALID_INPUT",
                "Input must be an EMV transaction, card, or dictionary",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def _validate_transaction(
        self, txn: EmvTransaction, result: ValidationResult
    ) -> None:
        """Validate complete EMV transaction."""
        # Validate card
        self._validate_card(txn.card, result)

        # Validate terminal
        self._validate_terminal(txn.terminal, result)

        # Validate cryptogram
        self._validate_cryptogram(txn.cryptogram, result)

        # Validate transaction data
        self._validate_transaction_data(txn, result)

        # Validate TVR
        self._validate_tvr(txn.terminal.tvr, result)

        # Cross-field validations
        self._validate_cross_fields(txn, result)

    def _validate_card(self, card: EmvCard, result: ValidationResult) -> None:
        """Validate card data."""
        # PAN validation
        if not card.pan:
            result.add_error(
                "EMV_MISSING_PAN",
                "PAN (tag 5A) is required",
                field="pan",
            )
        else:
            # Check PAN length
            if len(card.pan) < 13 or len(card.pan) > 19:
                result.add_error(
                    "EMV_INVALID_PAN_LENGTH",
                    f"PAN length {len(card.pan)} is invalid (expected 13-19)",
                    field="pan",
                )
            # Check PAN format (numeric only)
            elif not card.pan.isdigit():
                result.add_error(
                    "EMV_INVALID_PAN_FORMAT",
                    "PAN must contain only digits",
                    field="pan",
                )
            # Luhn check
            elif self.check_luhn and not self._luhn_check(card.pan):
                result.add_error(
                    "EMV_INVALID_PAN_CHECKSUM",
                    "PAN failed Luhn checksum validation",
                    field="pan",
                )

        # Expiration date validation
        if not card.expiration_date:
            result.add_error(
                "EMV_MISSING_EXPIRY",
                "Expiration date (tag 5F24) is required",
                field="expiration_date",
            )
        elif self.validate_expiry:
            expiry = card.expiry_date
            if expiry:
                today = date.today()
                # Check if expired (comparing first of expiry month to today)
                if expiry.replace(day=1) < today.replace(day=1):
                    result.add_error(
                        "EMV_CARD_EXPIRED",
                        f"Card expired on {card.expiration_date}",
                        field="expiration_date",
                    )
            else:
                result.add_error(
                    "EMV_INVALID_EXPIRY",
                    f"Invalid expiration date format: {card.expiration_date}",
                    field="expiration_date",
                )

        # AID validation
        if not card.aid:
            result.add_warning(
                "EMV_MISSING_AID",
                "AID (tag 9F06/84) is recommended",
                field="aid",
            )
        else:
            app = EmvApplication.from_aid(card.aid)
            if not app:
                result.add_warning(
                    "EMV_UNKNOWN_AID",
                    f"Unknown AID: {card.aid}",
                    field="aid",
                )

        # AIP validation
        if not card.aip:
            result.add_warning(
                "EMV_MISSING_AIP",
                "AIP (tag 82) is recommended",
                field="aip",
            )
        elif len(card.aip) != 4:  # 2 bytes = 4 hex chars
            result.add_error(
                "EMV_INVALID_AIP_LENGTH",
                f"AIP must be 2 bytes (4 hex chars), got {len(card.aip)}",
                field="aip",
            )

    def _validate_terminal(
        self, terminal: EmvTerminal, result: ValidationResult
    ) -> None:
        """Validate terminal data."""
        # Terminal ID
        if not terminal.terminal_id:
            result.add_warning(
                "EMV_MISSING_TERMINAL_ID",
                "Terminal ID (tag 9F1C) is recommended",
                field="terminal_id",
            )

        # Terminal capabilities
        if not terminal.terminal_capabilities:
            result.add_error(
                "EMV_MISSING_TERMINAL_CAP",
                "Terminal Capabilities (tag 9F33) is required",
                field="terminal_capabilities",
            )
        elif len(terminal.terminal_capabilities) != 6:  # 3 bytes = 6 hex chars
            result.add_error(
                "EMV_INVALID_TERMINAL_CAP",
                f"Terminal Capabilities must be 3 bytes, got {len(terminal.terminal_capabilities) // 2}",
                field="terminal_capabilities",
            )

        # Terminal country code
        if not terminal.terminal_country_code:
            result.add_error(
                "EMV_MISSING_COUNTRY_CODE",
                "Terminal Country Code (tag 9F1A) is required",
                field="terminal_country_code",
            )

        # TVR validation
        if len(terminal.tvr) != 10:  # 5 bytes = 10 hex chars
            result.add_error(
                "EMV_INVALID_TVR_LENGTH",
                f"TVR must be 5 bytes (10 hex chars), got {len(terminal.tvr)}",
                field="tvr",
            )

        # TSI validation
        if len(terminal.tsi) != 4:  # 2 bytes = 4 hex chars
            result.add_error(
                "EMV_INVALID_TSI_LENGTH",
                f"TSI must be 2 bytes (4 hex chars), got {len(terminal.tsi)}",
                field="tsi",
            )

        # CVM Results validation
        if len(terminal.cvm_results) != 6:  # 3 bytes = 6 hex chars
            result.add_error(
                "EMV_INVALID_CVM_RESULTS",
                f"CVM Results must be 3 bytes, got {len(terminal.cvm_results) // 2}",
                field="cvm_results",
            )

    def _validate_cryptogram(
        self, cryptogram: EmvCryptogram, result: ValidationResult
    ) -> None:
        """Validate cryptogram data."""
        # Application cryptogram
        if not cryptogram.application_cryptogram:
            result.add_warning(
                "EMV_MISSING_AC",
                "Application Cryptogram (tag 9F26) is required for online transactions",
                field="application_cryptogram",
            )
        elif len(cryptogram.application_cryptogram) != 16:  # 8 bytes = 16 hex chars
            result.add_error(
                "EMV_INVALID_AC_LENGTH",
                f"Application Cryptogram must be 8 bytes, got {len(cryptogram.application_cryptogram) // 2}",
                field="application_cryptogram",
            )

        # Cryptogram Information Data
        if not cryptogram.cryptogram_information_data:
            result.add_warning(
                "EMV_MISSING_CID",
                "Cryptogram Information Data (tag 9F27) is required",
                field="cryptogram_information_data",
            )
        else:
            crypt_type = cryptogram.cryptogram_type
            if not crypt_type:
                result.add_warning(
                    "EMV_UNKNOWN_CRYPT_TYPE",
                    f"Unknown cryptogram type in CID: {cryptogram.cryptogram_information_data}",
                    field="cryptogram_information_data",
                )

        # ATC
        if not cryptogram.atc:
            result.add_warning(
                "EMV_MISSING_ATC",
                "ATC (tag 9F36) is required",
                field="atc",
            )
        elif len(cryptogram.atc) != 4:  # 2 bytes = 4 hex chars
            result.add_error(
                "EMV_INVALID_ATC_LENGTH",
                f"ATC must be 2 bytes, got {len(cryptogram.atc) // 2}",
                field="atc",
            )

        # Unpredictable Number
        if not cryptogram.unpredictable_number:
            result.add_warning(
                "EMV_MISSING_UN",
                "Unpredictable Number (tag 9F37) is required",
                field="unpredictable_number",
            )
        elif len(cryptogram.unpredictable_number) != 8:  # 4 bytes = 8 hex chars
            result.add_error(
                "EMV_INVALID_UN_LENGTH",
                f"Unpredictable Number must be 4 bytes, got {len(cryptogram.unpredictable_number) // 2}",
                field="unpredictable_number",
            )

    def _validate_transaction_data(
        self, txn: EmvTransaction, result: ValidationResult
    ) -> None:
        """Validate transaction-level data."""
        # Transaction type
        if not txn.transaction_type:
            result.add_error(
                "EMV_MISSING_TXN_TYPE",
                "Transaction Type (tag 9C) is required",
                field="transaction_type",
            )
        else:
            txn_type = EmvTransactionType.from_code(txn.transaction_type)
            if not txn_type:
                result.add_warning(
                    "EMV_UNKNOWN_TXN_TYPE",
                    f"Unknown transaction type: {txn.transaction_type}",
                    field="transaction_type",
                )

        # Amount
        if txn.amount_authorized < 0:
            result.add_error(
                "EMV_NEGATIVE_AMOUNT",
                "Amount cannot be negative",
                field="amount_authorized",
            )

        # For purchase, amount should be positive
        if txn.transaction_type == "00" and txn.amount_authorized == 0:
            result.add_warning(
                "EMV_ZERO_AMOUNT",
                "Purchase amount is zero",
                field="amount_authorized",
            )

        # Currency code
        if not txn.currency_code:
            result.add_error(
                "EMV_MISSING_CURRENCY",
                "Currency Code (tag 5F2A) is required",
                field="currency_code",
            )

        # Transaction date
        if not txn.transaction_date:
            result.add_warning(
                "EMV_MISSING_DATE",
                "Transaction Date (tag 9A) is recommended",
                field="transaction_date",
            )

    def _validate_tvr(self, tvr: str, result: ValidationResult) -> None:
        """Validate and analyze TVR bits."""
        if len(tvr) != 10:
            return

        try:
            tvr_bytes = bytes.fromhex(tvr)
        except ValueError:
            result.add_error(
                "EMV_INVALID_TVR_FORMAT",
                "TVR must be valid hex",
                field="tvr",
            )
            return

        # Check critical TVR bits
        critical_bits = [
            ((1, 6), "SDA failed"),
            ((1, 3), "DDA failed"),
            ((1, 2), "CDA failed"),
            ((2, 6), "Expired application"),
            ((3, 7), "Cardholder verification was not successful"),
            ((3, 5), "PIN Try Limit exceeded"),
            ((5, 6), "Issuer authentication failed"),
        ]

        for (byte_num, bit_num), description in critical_bits:
            if self._check_bit(tvr_bytes, byte_num, bit_num):
                result.add_warning(
                    "EMV_TVR_FLAG",
                    f"TVR: {description}",
                    field="tvr",
                    byte=byte_num,
                    bit=bit_num,
                )

    def _validate_cross_fields(
        self, txn: EmvTransaction, result: ValidationResult
    ) -> None:
        """Validate cross-field dependencies."""
        # If cryptogram is ARQC, transaction should be online
        if txn.cryptogram.cryptogram_type == EmvCryptogramType.ARQC:
            if not txn.is_online:
                result.add_warning(
                    "EMV_ARQC_OFFLINE",
                    "ARQC cryptogram indicates online request but is_online is False",
                    field="cryptogram",
                )

        # If cryptogram is AAC, transaction should be declined
        if txn.cryptogram.cryptogram_type == EmvCryptogramType.AAC:
            result.add_warning(
                "EMV_OFFLINE_DECLINE",
                "Card returned AAC (offline decline)",
                field="cryptogram",
            )

        # AID should match card brand
        if txn.card.aid and txn.card.pan:
            app = EmvApplication.from_aid(txn.card.aid)
            if app:
                # Check PAN prefix matches AID brand
                pan_prefix = txn.card.pan[:1]
                aid_brand = app.name.upper()

                expected_prefixes = {
                    "VISA": ["4"],
                    "MASTERCARD": ["5", "2"],
                    "AMEX": ["3"],
                    "DISCOVER": ["6"],
                    "JCB": ["3"],
                }

                for brand, prefixes in expected_prefixes.items():
                    if brand in aid_brand and pan_prefix not in prefixes:
                        result.add_warning(
                            "EMV_AID_PAN_MISMATCH",
                            f"AID indicates {brand} but PAN prefix is {pan_prefix}",
                            field="aid",
                        )

    def _validate_dict(self, data: Dict, result: ValidationResult) -> None:
        """Validate EMV data from dictionary (TLV tag map)."""
        # PAN validation
        pan = data.get("5A", "")
        if not pan:
            result.add_error(
                "EMV_MISSING_PAN",
                "PAN (tag 5A) is required",
                field="5A",
            )
        elif self.check_luhn and not self._luhn_check(pan):
            result.add_error(
                "EMV_INVALID_PAN_CHECKSUM",
                "PAN failed Luhn checksum validation",
                field="5A",
            )

        # Expiration validation
        expiry = data.get("5F24", "")
        if not expiry:
            result.add_error(
                "EMV_MISSING_EXPIRY",
                "Expiration date (tag 5F24) is required",
                field="5F24",
            )

        # Amount validation
        amount = data.get("9F02", "")
        if amount:
            try:
                amount_int = int(amount)
                if amount_int < 0:
                    result.add_error(
                        "EMV_NEGATIVE_AMOUNT",
                        "Amount cannot be negative",
                        field="9F02",
                    )
            except ValueError:
                result.add_error(
                    "EMV_INVALID_AMOUNT",
                    "Amount must be numeric",
                    field="9F02",
                )

    def _luhn_check(self, card_number: str) -> bool:
        """Validate card number using Luhn algorithm."""
        try:
            digits = [int(d) for d in card_number]
            checksum = 0

            # Process from right to left
            for i, digit in enumerate(reversed(digits)):
                if i % 2 == 1:
                    digit *= 2
                    if digit > 9:
                        digit -= 9
                checksum += digit

            return checksum % 10 == 0
        except (ValueError, TypeError):
            return False

    def _check_bit(self, data: bytes, byte_num: int, bit_num: int) -> bool:
        """Check if a specific bit is set."""
        if byte_num < 1 or byte_num > len(data):
            return False
        byte_val = data[byte_num - 1]
        return bool(byte_val & (1 << bit_num))

    def analyze_tvr(self, tvr: str) -> List[str]:
        """
        Analyze TVR and return list of set flags.

        Args:
            tvr: TVR as hex string (10 chars / 5 bytes)

        Returns:
            List of flag descriptions
        """
        flags = []
        if len(tvr) != 10:
            return flags

        try:
            tvr_bytes = bytes.fromhex(tvr)
        except ValueError:
            return flags

        for (byte_num, bit_num), description in TVR_BITS.items():
            if "RFU" in description:
                continue
            if self._check_bit(tvr_bytes, byte_num, bit_num):
                flags.append(description)

        return flags

    def analyze_tsi(self, tsi: str) -> List[str]:
        """
        Analyze TSI and return list of set flags.

        Args:
            tsi: TSI as hex string (4 chars / 2 bytes)

        Returns:
            List of flag descriptions
        """
        flags = []
        if len(tsi) != 4:
            return flags

        try:
            tsi_bytes = bytes.fromhex(tsi)
        except ValueError:
            return flags

        for (byte_num, bit_num), description in TSI_BITS.items():
            if "RFU" in description:
                continue
            if self._check_bit(tsi_bytes, byte_num, bit_num):
                flags.append(description)

        return flags

    def analyze_aip(self, aip: str) -> List[str]:
        """
        Analyze AIP and return list of supported features.

        Args:
            aip: AIP as hex string (4 chars / 2 bytes)

        Returns:
            List of feature descriptions
        """
        features = []
        if len(aip) != 4:
            return features

        try:
            aip_bytes = bytes.fromhex(aip)
        except ValueError:
            return features

        for (byte_num, bit_num), description in AIP_BITS.items():
            if "RFU" in description:
                continue
            if self._check_bit(aip_bytes, byte_num, bit_num):
                features.append(description)

        return features
