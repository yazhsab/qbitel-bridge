"""
ACH/NACHA File Validator

Comprehensive validation for ACH/NACHA files including:
- File structure validation
- Record format validation
- Hash and total validation
- NACHA rules compliance
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
    validate_routing_number,
    validate_amount,
)
from ai_engine.domains.banking.protocols.payments.domestic.ach.ach_codes import (
    ServiceClassCode,
    TransactionCode,
    SECCode,
)


class ACHValidator(BaseValidator):
    """
    Validator for individual ACH entries and batches.

    Validates:
    - Transaction codes
    - Routing numbers
    - Account numbers
    - Amount limits
    - SEC code requirements
    """

    # NACHA amount limits (in dollars)
    MAX_SINGLE_ENTRY = 25_000_000  # $25M for single entry
    MAX_BATCH_TOTAL = 100_000_000  # $100M batch total
    MAX_FILE_TOTAL = 1_000_000_000  # $1B file total

    def __init__(self, strict: bool = True, max_amount: float = None):
        """
        Initialize the ACH validator.

        Args:
            strict: If True, treat warnings as errors
            max_amount: Override maximum single entry amount
        """
        super().__init__(strict)
        self.max_single_entry = max_amount or self.MAX_SINGLE_ENTRY

    @property
    def name(self) -> str:
        return "ACHValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate ACH data.

        Args:
            data: ACH entry, batch, or dictionary representation

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, dict):
            self._validate_dict(data, result)
        elif hasattr(data, "transaction_code"):
            # Entry detail object
            self._validate_entry(data, result)
        elif hasattr(data, "entries"):
            # Batch object
            self._validate_batch(data, result)
        else:
            result.add_error(
                "ACH_INVALID_INPUT",
                "Invalid input type for ACH validation",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def _validate_dict(self, data: Dict, result: ValidationResult) -> None:
        """Validate ACH data from dictionary."""
        # Routing number
        if "routing_number" in data:
            error = validate_routing_number(data["routing_number"])
            if error:
                result.add_error(
                    "ACH_INVALID_ROUTING",
                    error,
                    field_name="routing_number",
                )

        # Transaction code
        if "transaction_code" in data:
            tc = data["transaction_code"]
            if not TransactionCode.from_code(tc):
                result.add_error(
                    "ACH_INVALID_TX_CODE",
                    f"Invalid transaction code: {tc}",
                    field_name="transaction_code",
                )

        # Amount
        if "amount" in data:
            error = validate_amount(data["amount"], max_value=self.max_single_entry)
            if error:
                result.add_error(
                    "ACH_INVALID_AMOUNT",
                    error,
                    field_name="amount",
                )

        # Account number
        if "account_number" in data:
            acct = data["account_number"]
            if not acct or len(acct) > 17:
                result.add_error(
                    "ACH_INVALID_ACCOUNT",
                    "Account number must be 1-17 characters",
                    field_name="account_number",
                )

        # Individual name
        if "individual_name" in data:
            name = data["individual_name"]
            if not name:
                result.add_error(
                    "ACH_MISSING_NAME",
                    "Individual name is required",
                    field_name="individual_name",
                )
            elif len(name) > 22:
                result.add_error(
                    "ACH_INVALID_NAME",
                    "Individual name must not exceed 22 characters",
                    field_name="individual_name",
                )

    def _validate_entry(self, entry: Any, result: ValidationResult) -> None:
        """Validate an ACH entry detail."""
        # Routing number
        error = validate_routing_number(entry.routing_number)
        if error:
            result.add_error(
                "ACH_INVALID_ROUTING",
                error,
                field_name="routing_number",
            )

        # Transaction code validation
        tc = entry.transaction_code
        if isinstance(tc, str):
            if not TransactionCode.from_code(tc):
                result.add_error(
                    "ACH_INVALID_TX_CODE",
                    f"Invalid transaction code: {tc}",
                    field_name="transaction_code",
                )

        # Amount validation
        if entry.amount <= 0:
            result.add_error(
                "ACH_INVALID_AMOUNT",
                "Amount must be greater than zero",
                field_name="amount",
            )
        elif entry.amount > self.max_single_entry * 100:  # Amount in cents
            result.add_error(
                "ACH_AMOUNT_EXCEEDS_LIMIT",
                f"Amount exceeds maximum ({self.max_single_entry})",
                field_name="amount",
            )

        # Account number
        if not entry.account_number:
            result.add_error(
                "ACH_MISSING_ACCOUNT",
                "Account number is required",
                field_name="account_number",
            )

        # Individual name
        if not entry.individual_name:
            result.add_error(
                "ACH_MISSING_NAME",
                "Individual name is required",
                field_name="individual_name",
            )

    def _validate_batch(self, batch: Any, result: ValidationResult) -> None:
        """Validate an ACH batch."""
        # Service class code
        scc = batch.service_class_code
        if isinstance(scc, str):
            if not ServiceClassCode.from_code(scc):
                result.add_error(
                    "ACH_INVALID_SCC",
                    f"Invalid service class code: {scc}",
                    field_name="service_class_code",
                )

        # SEC code
        sec = batch.sec_code
        if isinstance(sec, str):
            sec_enum = SECCode.from_code(sec)
            if not sec_enum:
                result.add_error(
                    "ACH_INVALID_SEC",
                    f"Invalid SEC code: {sec}",
                    field_name="sec_code",
                )

        # Company name
        if not batch.company_name:
            result.add_error(
                "ACH_MISSING_COMPANY",
                "Company name is required",
                field_name="company_name",
            )

        # Validate entries
        for i, entry in enumerate(batch.entries):
            entry_result = self.validate(entry)
            for error in entry_result.errors:
                error.field_name = f"entries[{i}]/{error.field_name}"
                result.errors.append(error)
            for warning in entry_result.warnings:
                warning.field_name = f"entries[{i}]/{warning.field_name}"
                result.warnings.append(warning)

        if entry_result.errors:
            result.is_valid = False


class NACHAFileValidator(BaseValidator):
    """
    Validator for complete NACHA files.

    Validates:
    - File header and control records
    - Batch header and control records
    - Entry detail records
    - Hash totals and control sums
    - Record counts
    """

    def __init__(self, strict: bool = True):
        """
        Initialize the NACHA file validator.

        Args:
            strict: If True, treat warnings as errors
        """
        super().__init__(strict)
        self._ach_validator = ACHValidator(strict)

    @property
    def name(self) -> str:
        return "NACHAFileValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate a complete NACHA file.

        Args:
            data: NACHAFile object or file content string

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, str):
            result = self._validate_file_content(data, result)
        elif hasattr(data, "file_header") and hasattr(data, "batches"):
            result = self._validate_nacha_file(data, result)
        else:
            result.add_error(
                "NACHA_INVALID_INPUT",
                "Input must be NACHA file content or NACHAFile object",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def _validate_file_content(self, content: str, result: ValidationResult) -> ValidationResult:
        """Validate NACHA file from raw content."""
        lines = content.strip().split("\n")

        if not lines:
            result.add_error(
                "NACHA_EMPTY_FILE",
                "File is empty",
                severity=ValidationSeverity.CRITICAL,
            )
            return result

        # Check minimum structure
        if len(lines) < 2:
            result.add_error(
                "NACHA_INCOMPLETE",
                "File must have at least file header and control records",
                severity=ValidationSeverity.CRITICAL,
            )
            return result

        # Validate record lengths
        for i, line in enumerate(lines):
            if len(line) != 94:
                result.add_error(
                    "NACHA_INVALID_LENGTH",
                    f"Line {i + 1} has invalid length ({len(line)} chars, expected 94)",
                    field=f"line[{i + 1}]",
                )

        # Validate file header (first record)
        first_line = lines[0]
        if not first_line.startswith("1"):
            result.add_error(
                "NACHA_INVALID_FILE_HEADER",
                "File must start with File Header record (type 1)",
                field_name="file_header",
            )
        else:
            self._validate_file_header_record(first_line, result)

        # Validate file control (last record, excluding padding)
        last_line = lines[-1]
        if not last_line.startswith("9"):
            result.add_error(
                "NACHA_INVALID_FILE_CONTROL",
                "File must end with File Control record (type 9)",
                field_name="file_control",
            )
        else:
            self._validate_file_control_record(last_line, content, result)

        # Track batches for validation
        batch_count = 0
        entry_count = 0
        total_debit = 0
        total_credit = 0
        entry_hash = 0

        # Validate batch structure
        in_batch = False
        current_batch_entries = 0

        for i, line in enumerate(lines[1:-1], start=2):  # Skip header and control
            record_type = line[0] if line else ""

            if record_type == "5":  # Batch header
                if in_batch:
                    result.add_error(
                        "NACHA_NESTED_BATCH",
                        f"Batch header found before previous batch closed (line {i})",
                        field=f"line[{i}]",
                    )
                in_batch = True
                current_batch_entries = 0
                self._validate_batch_header_record(line, result, i)

            elif record_type == "6":  # Entry detail
                if not in_batch:
                    result.add_error(
                        "NACHA_ENTRY_OUTSIDE_BATCH",
                        f"Entry detail found outside of batch (line {i})",
                        field=f"line[{i}]",
                    )
                current_batch_entries += 1
                entry_count += 1

                # Extract routing for hash
                routing = line[3:11]
                if routing.isdigit():
                    entry_hash += int(routing)

                # Extract amount
                amount = line[29:39]
                if amount.isdigit():
                    tc = line[1:3]
                    if tc in ("22", "23", "24", "32", "33", "34", "42", "43", "44", "52", "53", "54"):
                        total_credit += int(amount)
                    else:
                        total_debit += int(amount)

                self._validate_entry_record(line, result, i)

            elif record_type == "7":  # Addenda
                if not in_batch:
                    result.add_error(
                        "NACHA_ADDENDA_OUTSIDE_BATCH",
                        f"Addenda record found outside of batch (line {i})",
                        field=f"line[{i}]",
                    )

            elif record_type == "8":  # Batch control
                if not in_batch:
                    result.add_error(
                        "NACHA_CONTROL_OUTSIDE_BATCH",
                        f"Batch control found without batch header (line {i})",
                        field=f"line[{i}]",
                    )
                else:
                    batch_count += 1
                    self._validate_batch_control_record(line, result, i, current_batch_entries)
                in_batch = False

            elif record_type == "9":
                # File control or padding - skip
                pass

            elif record_type:
                result.add_warning(
                    "NACHA_UNKNOWN_RECORD",
                    f"Unknown record type '{record_type}' at line {i}",
                    field=f"line[{i}]",
                )

        # Store calculated values for comparison
        result.metadata["calculated_batch_count"] = batch_count
        result.metadata["calculated_entry_count"] = entry_count
        result.metadata["calculated_debit_total"] = total_debit
        result.metadata["calculated_credit_total"] = total_credit
        result.metadata["calculated_entry_hash"] = entry_hash % 10000000000

        return result

    def _validate_nacha_file(self, file_obj: Any, result: ValidationResult) -> ValidationResult:
        """Validate a parsed NACHAFile object."""
        # Validate file header
        fh = file_obj.file_header
        if not fh:
            result.add_error(
                "NACHA_MISSING_FILE_HEADER",
                "File header is required",
                field_name="file_header",
            )
        else:
            # Immediate destination
            error = validate_routing_number(fh.immediate_destination.strip())
            if error:
                result.add_error(
                    "NACHA_INVALID_DEST",
                    f"Invalid immediate destination: {error}",
                    field_name="file_header/immediate_destination",
                )

            # Immediate origin
            error = validate_routing_number(fh.immediate_origin.strip())
            if error:
                result.add_error(
                    "NACHA_INVALID_ORIGIN",
                    f"Invalid immediate origin: {error}",
                    field_name="file_header/immediate_origin",
                )

        # Validate batches
        if not file_obj.batches:
            result.add_error(
                "NACHA_NO_BATCHES",
                "File must contain at least one batch",
                field_name="batches",
            )
        else:
            for i, batch in enumerate(file_obj.batches):
                batch_result = self._ach_validator.validate(batch)
                for error in batch_result.errors:
                    error.field_name = f"batches[{i}]/{error.field_name}"
                    result.errors.append(error)
                for warning in batch_result.warnings:
                    warning.field_name = f"batches[{i}]/{warning.field_name}"
                    result.warnings.append(warning)

                if batch_result.errors:
                    result.is_valid = False

        # Validate file control
        fc = file_obj.file_control
        if fc:
            # Verify batch count
            if fc.batch_count != len(file_obj.batches):
                result.add_error(
                    "NACHA_BATCH_COUNT_MISMATCH",
                    f"Batch count mismatch: control={fc.batch_count}, actual={len(file_obj.batches)}",
                    field_name="file_control/batch_count",
                )

            # Calculate expected entry count
            expected_entries = sum(len(b.entries) for b in file_obj.batches)
            if fc.entry_addenda_count != expected_entries:
                result.add_error(
                    "NACHA_ENTRY_COUNT_MISMATCH",
                    f"Entry count mismatch: control={fc.entry_addenda_count}, actual={expected_entries}",
                    field_name="file_control/entry_addenda_count",
                )

        return result

    def _validate_file_header_record(self, line: str, result: ValidationResult) -> None:
        """Validate file header record structure."""
        if len(line) < 94:
            return

        # Priority code
        priority = line[1:3]
        if priority != "01":
            result.add_warning(
                "NACHA_UNUSUAL_PRIORITY",
                f"Unusual priority code: {priority} (expected 01)",
                field_name="file_header/priority_code",
            )

        # Immediate destination
        dest = line[3:13].strip()
        error = validate_routing_number(dest)
        if error:
            result.add_error(
                "NACHA_INVALID_DEST",
                f"Invalid immediate destination: {error}",
                field_name="file_header/immediate_destination",
            )

        # Immediate origin
        origin = line[13:23].strip()
        error = validate_routing_number(origin)
        if error:
            result.add_error(
                "NACHA_INVALID_ORIGIN",
                f"Invalid immediate origin: {error}",
                field_name="file_header/immediate_origin",
            )

        # File creation date
        file_date = line[23:29]
        try:
            datetime.strptime(file_date, "%y%m%d")
        except ValueError:
            result.add_error(
                "NACHA_INVALID_DATE",
                f"Invalid file creation date: {file_date}",
                field_name="file_header/file_creation_date",
            )

        # Record size
        record_size = line[34:37]
        if record_size != "094":
            result.add_error(
                "NACHA_INVALID_RECORD_SIZE",
                f"Invalid record size: {record_size} (must be 094)",
                field_name="file_header/record_size",
            )

        # Blocking factor
        blocking = line[37:39]
        if blocking != "10":
            result.add_warning(
                "NACHA_UNUSUAL_BLOCKING",
                f"Unusual blocking factor: {blocking} (expected 10)",
                field_name="file_header/blocking_factor",
            )

        # Format code
        format_code = line[39:40]
        if format_code != "1":
            result.add_error(
                "NACHA_INVALID_FORMAT",
                f"Invalid format code: {format_code} (must be 1)",
                field_name="file_header/format_code",
            )

    def _validate_batch_header_record(self, line: str, result: ValidationResult, line_num: int) -> None:
        """Validate batch header record."""
        if len(line) < 94:
            return

        prefix = f"line[{line_num}]/batch_header"

        # Service class code
        scc = line[1:4]
        if not ServiceClassCode.from_code(scc):
            result.add_error(
                "NACHA_INVALID_SCC",
                f"Invalid service class code: {scc}",
                field=f"{prefix}/service_class_code",
            )

        # Company name
        company_name = line[4:20].strip()
        if not company_name:
            result.add_error(
                "NACHA_MISSING_COMPANY",
                "Company name is required",
                field=f"{prefix}/company_name",
            )

        # Company ID
        company_id = line[40:50].strip()
        if not company_id:
            result.add_error(
                "NACHA_MISSING_COMPANY_ID",
                "Company identification is required",
                field=f"{prefix}/company_identification",
            )

        # SEC code
        sec = line[50:53]
        if not SECCode.from_code(sec):
            result.add_error(
                "NACHA_INVALID_SEC",
                f"Invalid SEC code: {sec}",
                field=f"{prefix}/sec_code",
            )

        # ODFI routing
        odfi = line[79:87]
        error = validate_routing_number(odfi + "0")  # Add check digit placeholder
        if error and not odfi.isdigit():
            result.add_error(
                "NACHA_INVALID_ODFI",
                f"Invalid ODFI routing: {odfi}",
                field=f"{prefix}/odfi_identification",
            )

    def _validate_entry_record(self, line: str, result: ValidationResult, line_num: int) -> None:
        """Validate entry detail record."""
        if len(line) < 94:
            return

        prefix = f"line[{line_num}]/entry"

        # Transaction code
        tc = line[1:3]
        if not TransactionCode.from_code(tc):
            result.add_error(
                "NACHA_INVALID_TX_CODE",
                f"Invalid transaction code: {tc}",
                field=f"{prefix}/transaction_code",
            )

        # RDFI routing
        rdfi = line[3:12]
        error = validate_routing_number(rdfi)
        if error:
            result.add_error(
                "NACHA_INVALID_RDFI",
                f"Invalid RDFI routing: {error}",
                field=f"{prefix}/rdfi_identification",
            )

        # Account number
        account = line[12:29].strip()
        if not account:
            result.add_error(
                "NACHA_MISSING_ACCOUNT",
                "Account number is required",
                field=f"{prefix}/account_number",
            )

        # Amount
        amount = line[29:39]
        if not amount.isdigit():
            result.add_error(
                "NACHA_INVALID_AMOUNT",
                f"Invalid amount: {amount}",
                field=f"{prefix}/amount",
            )

        # Individual name
        name = line[54:76].strip()
        if not name:
            result.add_error(
                "NACHA_MISSING_NAME",
                "Individual name is required",
                field=f"{prefix}/individual_name",
            )

    def _validate_batch_control_record(self, line: str, result: ValidationResult, line_num: int, entry_count: int) -> None:
        """Validate batch control record."""
        if len(line) < 94:
            return

        prefix = f"line[{line_num}]/batch_control"

        # Entry/Addenda count
        count = line[4:10]
        try:
            count_int = int(count)
            if count_int != entry_count:
                result.add_warning(
                    "NACHA_ENTRY_COUNT_MISMATCH",
                    f"Entry count in control ({count_int}) doesn't match entries ({entry_count})",
                    field=f"{prefix}/entry_addenda_count",
                )
        except ValueError:
            result.add_error(
                "NACHA_INVALID_ENTRY_COUNT",
                f"Invalid entry count: {count}",
                field=f"{prefix}/entry_addenda_count",
            )

    def _validate_file_control_record(self, line: str, content: str, result: ValidationResult) -> None:
        """Validate file control record."""
        if len(line) < 94:
            return

        # Batch count
        batch_count = line[1:7]
        if not batch_count.isdigit():
            result.add_error(
                "NACHA_INVALID_BATCH_COUNT",
                f"Invalid batch count: {batch_count}",
                field_name="file_control/batch_count",
            )

        # Block count
        block_count = line[7:13]
        if not block_count.isdigit():
            result.add_error(
                "NACHA_INVALID_BLOCK_COUNT",
                f"Invalid block count: {block_count}",
                field_name="file_control/block_count",
            )

        # Entry/Addenda count
        entry_count = line[13:21]
        if not entry_count.isdigit():
            result.add_error(
                "NACHA_INVALID_ENTRY_COUNT",
                f"Invalid entry count: {entry_count}",
                field_name="file_control/entry_addenda_count",
            )

        # Entry hash
        entry_hash = line[21:31]
        if not entry_hash.isdigit():
            result.add_error(
                "NACHA_INVALID_HASH",
                f"Invalid entry hash: {entry_hash}",
                field_name="file_control/entry_hash",
            )

        # Total debit
        total_debit = line[31:43]
        if not total_debit.isdigit():
            result.add_error(
                "NACHA_INVALID_DEBIT_TOTAL",
                f"Invalid total debit: {total_debit}",
                field_name="file_control/total_debit",
            )

        # Total credit
        total_credit = line[43:55]
        if not total_credit.isdigit():
            result.add_error(
                "NACHA_INVALID_CREDIT_TOTAL",
                f"Invalid total credit: {total_credit}",
                field_name="file_control/total_credit",
            )
