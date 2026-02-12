"""
NACHA ACH File Parser

Parser for NACHA (National Automated Clearing House Association)
ACH file format. Supports all standard record types.

File Structure:
- File Header Record (1)
- Batch Header Record (5) [1..n]
- Entry Detail Record (6) [1..n per batch]
- Addenda Record (7) [0..n per entry]
- Batch Control Record (8)
- File Control Record (9)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
import logging
import re

from ai_engine.domains.banking.protocols.payments.domestic.ach.ach_codes import (
    ServiceClassCode,
    TransactionCode,
    SECCode,
    AddendaTypeCode,
)

logger = logging.getLogger(__name__)


class NACHAParseError(Exception):
    """Exception raised when NACHA file parsing fails."""

    def __init__(self, message: str, line_number: Optional[int] = None, record_type: Optional[str] = None):
        self.line_number = line_number
        self.record_type = record_type
        super().__init__(message)


# NACHA record length
RECORD_LENGTH = 94


@dataclass
class FileHeader:
    """
    NACHA File Header Record (Record Type 1).

    Identifies the file, immediate destination, and origin.
    """

    record_type: str = "1"
    priority_code: str = "01"
    immediate_destination: str = ""  # 10 chars, space + 9-digit routing
    immediate_origin: str = ""  # 10 chars, space + 9-digit ID
    file_creation_date: Optional[date] = None
    file_creation_time: str = ""  # HHMM
    file_id_modifier: str = "A"  # A-Z, 0-9
    record_size: str = "094"
    blocking_factor: str = "10"
    format_code: str = "1"
    immediate_destination_name: str = ""  # 23 chars
    immediate_origin_name: str = ""  # 23 chars
    reference_code: str = ""  # 8 chars

    def validate(self) -> List[str]:
        """Validate file header."""
        errors = []

        if self.record_type != "1":
            errors.append("File header record type must be '1'")

        if not self.immediate_destination or len(self.immediate_destination.strip()) < 9:
            errors.append("Invalid immediate destination routing number")

        if not self.immediate_origin or len(self.immediate_origin.strip()) < 9:
            errors.append("Invalid immediate origin ID")

        if not self.file_creation_date:
            errors.append("File creation date is required")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": self.record_type,
            "priority_code": self.priority_code,
            "immediate_destination": self.immediate_destination.strip(),
            "immediate_origin": self.immediate_origin.strip(),
            "file_creation_date": self.file_creation_date.isoformat() if self.file_creation_date else None,
            "file_creation_time": self.file_creation_time,
            "file_id_modifier": self.file_id_modifier,
            "immediate_destination_name": self.immediate_destination_name.strip(),
            "immediate_origin_name": self.immediate_origin_name.strip(),
        }


@dataclass
class BatchHeader:
    """
    NACHA Batch Header Record (Record Type 5).

    Identifies the batch and its characteristics.
    """

    record_type: str = "5"
    service_class_code: ServiceClassCode = ServiceClassCode.MIXED
    company_name: str = ""  # 16 chars
    company_discretionary_data: str = ""  # 20 chars
    company_identification: str = ""  # 10 chars (1+9 digit tax ID)
    standard_entry_class: SECCode = SECCode.PPD
    company_entry_description: str = ""  # 10 chars
    company_descriptive_date: str = ""  # 6 chars
    effective_entry_date: Optional[date] = None
    settlement_date: str = ""  # 3 chars julian date (set by ACH operator)
    originator_status_code: str = "1"  # 1 = ACH operator
    originating_dfi: str = ""  # 8 chars (first 8 of routing)
    batch_number: int = 1

    def validate(self) -> List[str]:
        """Validate batch header."""
        errors = []

        if self.record_type != "5":
            errors.append("Batch header record type must be '5'")

        if not self.company_name or len(self.company_name.strip()) == 0:
            errors.append("Company name is required")

        if not self.company_identification or len(self.company_identification.strip()) < 10:
            errors.append("Invalid company identification (must be 10 chars)")

        if not self.originating_dfi or len(self.originating_dfi.strip()) != 8:
            errors.append("Originating DFI must be 8 characters")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": self.record_type,
            "service_class_code": self.service_class_code.code,
            "company_name": self.company_name.strip(),
            "company_identification": self.company_identification.strip(),
            "standard_entry_class": self.standard_entry_class.code,
            "company_entry_description": self.company_entry_description.strip(),
            "effective_entry_date": self.effective_entry_date.isoformat() if self.effective_entry_date else None,
            "originating_dfi": self.originating_dfi.strip(),
            "batch_number": self.batch_number,
        }


@dataclass
class EntryDetail:
    """
    NACHA Entry Detail Record (Record Type 6).

    Individual transaction within a batch.
    """

    record_type: str = "6"
    transaction_code: TransactionCode = TransactionCode.CHECKING_CREDIT
    receiving_dfi: str = ""  # 8 chars routing number
    check_digit: str = ""  # 1 char
    dfi_account_number: str = ""  # 17 chars
    amount: Decimal = Decimal("0")  # 10 chars, right justified, no decimal
    individual_identification: str = ""  # 15 chars
    individual_name: str = ""  # 22 chars
    discretionary_data: str = ""  # 2 chars
    addenda_record_indicator: str = "0"  # 0 or 1
    trace_number: str = ""  # 15 chars

    # Addenda records associated with this entry
    addenda: List["Addenda"] = field(default_factory=list)

    @property
    def has_addenda(self) -> bool:
        return self.addenda_record_indicator == "1" or len(self.addenda) > 0

    @property
    def full_routing_number(self) -> str:
        """Get full 9-digit routing number."""
        return self.receiving_dfi + self.check_digit

    def validate(self) -> List[str]:
        """Validate entry detail."""
        errors = []

        if self.record_type != "6":
            errors.append("Entry detail record type must be '6'")

        if not self.receiving_dfi or len(self.receiving_dfi.strip()) != 8:
            errors.append("Receiving DFI must be 8 characters")

        if not self.dfi_account_number or len(self.dfi_account_number.strip()) == 0:
            errors.append("DFI account number is required")

        if self.amount < 0:
            errors.append("Amount cannot be negative")

        if not self.individual_name or len(self.individual_name.strip()) == 0:
            errors.append("Individual name is required")

        # Validate check digit using Luhn-like algorithm
        if self.receiving_dfi and self.check_digit:
            expected_check = self._calculate_check_digit(self.receiving_dfi)
            if self.check_digit != expected_check:
                errors.append(f"Check digit mismatch: expected {expected_check}, got {self.check_digit}")

        return errors

    def _calculate_check_digit(self, routing: str) -> str:
        """Calculate ABA routing number check digit."""
        if len(routing) < 8:
            return "0"

        weights = [3, 7, 1, 3, 7, 1, 3, 7]
        total = sum(int(routing[i]) * weights[i] for i in range(8))
        check = (10 - (total % 10)) % 10
        return str(check)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": self.record_type,
            "transaction_code": self.transaction_code.code,
            "receiving_dfi": self.receiving_dfi.strip(),
            "check_digit": self.check_digit,
            "dfi_account_number": self.dfi_account_number.strip(),
            "amount": str(self.amount),
            "individual_identification": self.individual_identification.strip(),
            "individual_name": self.individual_name.strip(),
            "addenda_record_indicator": self.addenda_record_indicator,
            "trace_number": self.trace_number.strip(),
            "addenda": [a.to_dict() for a in self.addenda],
        }


@dataclass
class Addenda:
    """
    NACHA Addenda Record (Record Type 7).

    Additional payment-related information.
    """

    record_type: str = "7"
    addenda_type_code: AddendaTypeCode = AddendaTypeCode.STANDARD
    payment_related_info: str = ""  # 80 chars
    addenda_sequence_number: int = 1  # 4 chars
    entry_detail_sequence_number: str = ""  # 7 chars

    # For return addenda (type 99)
    return_reason_code: str = ""
    original_entry_trace_number: str = ""
    date_of_death: str = ""
    original_dfi: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": self.record_type,
            "addenda_type_code": self.addenda_type_code.code,
            "payment_related_info": self.payment_related_info.strip(),
            "addenda_sequence_number": self.addenda_sequence_number,
            "entry_detail_sequence_number": self.entry_detail_sequence_number.strip(),
            "return_reason_code": self.return_reason_code,
        }


@dataclass
class BatchControl:
    """
    NACHA Batch Control Record (Record Type 8).

    Totals and hash for the batch.
    """

    record_type: str = "8"
    service_class_code: ServiceClassCode = ServiceClassCode.MIXED
    entry_addenda_count: int = 0  # 6 chars
    entry_hash: str = ""  # 10 chars
    total_debit_amount: Decimal = Decimal("0")  # 12 chars
    total_credit_amount: Decimal = Decimal("0")  # 12 chars
    company_identification: str = ""  # 10 chars
    message_authentication_code: str = ""  # 19 chars (optional)
    reserved: str = ""  # 6 chars
    originating_dfi: str = ""  # 8 chars
    batch_number: int = 1  # 7 chars

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": self.record_type,
            "service_class_code": self.service_class_code.code,
            "entry_addenda_count": self.entry_addenda_count,
            "entry_hash": self.entry_hash.strip(),
            "total_debit_amount": str(self.total_debit_amount),
            "total_credit_amount": str(self.total_credit_amount),
            "company_identification": self.company_identification.strip(),
            "originating_dfi": self.originating_dfi.strip(),
            "batch_number": self.batch_number,
        }


@dataclass
class FileControl:
    """
    NACHA File Control Record (Record Type 9).

    File-level totals and summary.
    """

    record_type: str = "9"
    batch_count: int = 0  # 6 chars
    block_count: int = 0  # 6 chars
    entry_addenda_count: int = 0  # 8 chars
    entry_hash: str = ""  # 10 chars
    total_debit_amount: Decimal = Decimal("0")  # 12 chars
    total_credit_amount: Decimal = Decimal("0")  # 12 chars
    reserved: str = ""  # 39 chars

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": self.record_type,
            "batch_count": self.batch_count,
            "block_count": self.block_count,
            "entry_addenda_count": self.entry_addenda_count,
            "entry_hash": self.entry_hash.strip(),
            "total_debit_amount": str(self.total_debit_amount),
            "total_credit_amount": str(self.total_credit_amount),
        }


@dataclass
class Batch:
    """A batch of ACH entries."""

    header: BatchHeader
    entries: List[EntryDetail] = field(default_factory=list)
    control: Optional[BatchControl] = None

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def addenda_count(self) -> int:
        return sum(len(e.addenda) for e in self.entries)

    @property
    def total_entries_and_addenda(self) -> int:
        return self.entry_count + self.addenda_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "entries": [e.to_dict() for e in self.entries],
            "control": self.control.to_dict() if self.control else None,
            "entry_count": self.entry_count,
            "addenda_count": self.addenda_count,
        }


@dataclass
class NACHAFile:
    """
    Complete NACHA ACH File.

    Contains file header, batches, and file control.
    """

    file_header: Optional[FileHeader] = None
    batches: List[Batch] = field(default_factory=list)
    file_control: Optional[FileControl] = None

    # Parser metadata
    raw_content: str = ""
    parse_errors: List[str] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)

    @property
    def batch_count(self) -> int:
        return len(self.batches)

    @property
    def total_entry_count(self) -> int:
        return sum(b.entry_count for b in self.batches)

    @property
    def total_credit_amount(self) -> Decimal:
        total = Decimal("0")
        for batch in self.batches:
            for entry in batch.entries:
                if entry.transaction_code.is_credit:
                    total += entry.amount
        return total

    @property
    def total_debit_amount(self) -> Decimal:
        total = Decimal("0")
        for batch in self.batches:
            for entry in batch.entries:
                if entry.transaction_code.is_debit:
                    total += entry.amount
        return total

    def validate(self) -> List[str]:
        """Validate entire file."""
        errors = []

        if self.file_header:
            errors.extend(self.file_header.validate())

        for batch in self.batches:
            errors.extend(batch.header.validate())
            for entry in batch.entries:
                errors.extend(entry.validate())

        # Cross-validate totals
        if self.file_control:
            if self.file_control.batch_count != self.batch_count:
                errors.append(f"Batch count mismatch: control={self.file_control.batch_count}, " f"actual={self.batch_count}")

            actual_entry_count = sum(b.total_entries_and_addenda for b in self.batches)
            if self.file_control.entry_addenda_count != actual_entry_count:
                errors.append(
                    f"Entry/addenda count mismatch: control={self.file_control.entry_addenda_count}, "
                    f"actual={actual_entry_count}"
                )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_header": self.file_header.to_dict() if self.file_header else None,
            "batches": [b.to_dict() for b in self.batches],
            "file_control": self.file_control.to_dict() if self.file_control else None,
            "summary": {
                "batch_count": self.batch_count,
                "total_entry_count": self.total_entry_count,
                "total_credit_amount": str(self.total_credit_amount),
                "total_debit_amount": str(self.total_debit_amount),
            },
            "parse_errors": self.parse_errors,
            "parse_warnings": self.parse_warnings,
        }


class NACHAParser:
    """
    NACHA ACH File Parser.

    Parses fixed-width NACHA format files.
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def parse(self, content: str) -> NACHAFile:
        """
        Parse NACHA file content.

        Args:
            content: Raw file content (string)

        Returns:
            NACHAFile object
        """
        self.errors = []
        self.warnings = []

        nacha_file = NACHAFile(raw_content=content)

        # Split into records
        records = self._split_records(content)

        if not records:
            self.errors.append("No records found in file")
            nacha_file.parse_errors = self.errors
            return nacha_file

        # Track current batch
        current_batch: Optional[Batch] = None
        current_entry: Optional[EntryDetail] = None

        for line_num, record in enumerate(records, 1):
            if len(record) < 1:
                continue

            record_type = record[0]

            try:
                if record_type == "1":
                    # File Header
                    nacha_file.file_header = self._parse_file_header(record)

                elif record_type == "5":
                    # Batch Header - start new batch
                    if current_batch is not None:
                        self.warnings.append(f"Line {line_num}: Batch started without control record")
                    current_batch = Batch(header=self._parse_batch_header(record))

                elif record_type == "6":
                    # Entry Detail
                    if current_batch is None:
                        self.errors.append(f"Line {line_num}: Entry without batch header")
                        continue
                    current_entry = self._parse_entry_detail(record)
                    current_batch.entries.append(current_entry)

                elif record_type == "7":
                    # Addenda
                    if current_entry is None:
                        self.errors.append(f"Line {line_num}: Addenda without entry")
                        continue
                    addenda = self._parse_addenda(record)
                    current_entry.addenda.append(addenda)

                elif record_type == "8":
                    # Batch Control
                    if current_batch is None:
                        self.errors.append(f"Line {line_num}: Batch control without header")
                        continue
                    current_batch.control = self._parse_batch_control(record)
                    nacha_file.batches.append(current_batch)
                    current_batch = None
                    current_entry = None

                elif record_type == "9":
                    # File Control
                    nacha_file.file_control = self._parse_file_control(record)

                else:
                    self.warnings.append(f"Line {line_num}: Unknown record type '{record_type}'")

            except Exception as e:
                self.errors.append(f"Line {line_num}: Parse error - {str(e)}")

        nacha_file.parse_errors = self.errors
        nacha_file.parse_warnings = self.warnings

        return nacha_file

    def parse_file(self, file_path: str) -> NACHAFile:
        """Parse NACHA file from path."""
        with open(file_path, "r") as f:
            content = f.read()
        return self.parse(content)

    def _split_records(self, content: str) -> List[str]:
        """Split content into fixed-length records."""
        # Remove line breaks and split by record length
        content = content.replace("\r\n", "").replace("\n", "").replace("\r", "")

        records = []
        for i in range(0, len(content), RECORD_LENGTH):
            record = content[i : i + RECORD_LENGTH]
            if record.strip():  # Skip empty records
                records.append(record)

        return records

    def _parse_file_header(self, record: str) -> FileHeader:
        """Parse File Header Record (Type 1)."""
        return FileHeader(
            record_type=record[0],
            priority_code=record[1:3],
            immediate_destination=record[3:13],
            immediate_origin=record[13:23],
            file_creation_date=self._parse_date(record[23:29]),
            file_creation_time=record[29:33],
            file_id_modifier=record[33],
            record_size=record[34:37],
            blocking_factor=record[37:39],
            format_code=record[39],
            immediate_destination_name=record[40:63],
            immediate_origin_name=record[63:86],
            reference_code=record[86:94],
        )

    def _parse_batch_header(self, record: str) -> BatchHeader:
        """Parse Batch Header Record (Type 5)."""
        scc = ServiceClassCode.from_code(record[1:4]) or ServiceClassCode.MIXED
        sec = SECCode.from_code(record[50:53]) or SECCode.PPD

        return BatchHeader(
            record_type=record[0],
            service_class_code=scc,
            company_name=record[4:20],
            company_discretionary_data=record[20:40],
            company_identification=record[40:50],
            standard_entry_class=sec,
            company_entry_description=record[53:63],
            company_descriptive_date=record[63:69],
            effective_entry_date=self._parse_date(record[69:75]),
            settlement_date=record[75:78],
            originator_status_code=record[78],
            originating_dfi=record[79:87],
            batch_number=int(record[87:94] or "1"),
        )

    def _parse_entry_detail(self, record: str) -> EntryDetail:
        """Parse Entry Detail Record (Type 6)."""
        tc = TransactionCode.from_code(record[1:3]) or TransactionCode.CHECKING_CREDIT

        amount_str = record[29:39].strip()
        amount = Decimal(amount_str) / 100 if amount_str else Decimal("0")

        return EntryDetail(
            record_type=record[0],
            transaction_code=tc,
            receiving_dfi=record[3:11],
            check_digit=record[11],
            dfi_account_number=record[12:29],
            amount=amount,
            individual_identification=record[39:54],
            individual_name=record[54:76],
            discretionary_data=record[76:78],
            addenda_record_indicator=record[78],
            trace_number=record[79:94],
        )

    def _parse_addenda(self, record: str) -> Addenda:
        """Parse Addenda Record (Type 7)."""
        atc = AddendaTypeCode.from_code(record[1:3]) or AddendaTypeCode.STANDARD

        return Addenda(
            record_type=record[0],
            addenda_type_code=atc,
            payment_related_info=record[3:83],
            addenda_sequence_number=int(record[83:87] or "1"),
            entry_detail_sequence_number=record[87:94],
        )

    def _parse_batch_control(self, record: str) -> BatchControl:
        """Parse Batch Control Record (Type 8)."""
        scc = ServiceClassCode.from_code(record[1:4]) or ServiceClassCode.MIXED

        debit_str = record[20:32].strip()
        credit_str = record[32:44].strip()

        return BatchControl(
            record_type=record[0],
            service_class_code=scc,
            entry_addenda_count=int(record[4:10] or "0"),
            entry_hash=record[10:20],
            total_debit_amount=Decimal(debit_str) / 100 if debit_str else Decimal("0"),
            total_credit_amount=Decimal(credit_str) / 100 if credit_str else Decimal("0"),
            company_identification=record[44:54],
            message_authentication_code=record[54:73],
            reserved=record[73:79],
            originating_dfi=record[79:87],
            batch_number=int(record[87:94] or "1"),
        )

    def _parse_file_control(self, record: str) -> FileControl:
        """Parse File Control Record (Type 9)."""
        debit_str = record[31:43].strip()
        credit_str = record[43:55].strip()

        return FileControl(
            record_type=record[0],
            batch_count=int(record[1:7] or "0"),
            block_count=int(record[7:13] or "0"),
            entry_addenda_count=int(record[13:21] or "0"),
            entry_hash=record[21:31],
            total_debit_amount=Decimal(debit_str) / 100 if debit_str else Decimal("0"),
            total_credit_amount=Decimal(credit_str) / 100 if credit_str else Decimal("0"),
            reserved=record[55:94],
        )

    def _parse_date(self, yymmdd: str) -> Optional[date]:
        """Parse YYMMDD date format."""
        if not yymmdd or not yymmdd.strip() or yymmdd.strip() == "000000":
            return None

        try:
            yymmdd = yymmdd.strip()
            year = 2000 + int(yymmdd[0:2])
            month = int(yymmdd[2:4])
            day = int(yymmdd[4:6])
            return date(year, month, day)
        except (ValueError, IndexError):
            return None
