"""
NACHA ACH File Builder

Builder for creating NACHA format ACH files.
Supports all standard record types and validates output.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
import logging

from ai_engine.domains.banking.protocols.payments.domestic.ach.ach_codes import (
    ServiceClassCode,
    TransactionCode,
    SECCode,
)
from ai_engine.domains.banking.protocols.payments.domestic.ach.nacha_parser import (
    NACHAFile,
    FileHeader,
    FileControl,
    BatchHeader,
    BatchControl,
    EntryDetail,
    Addenda,
    Batch,
    RECORD_LENGTH,
)

logger = logging.getLogger(__name__)


@dataclass
class ACHEntry:
    """Simplified entry data for building ACH files."""

    # Required fields
    transaction_code: TransactionCode
    routing_number: str  # 9 digits with check digit
    account_number: str  # Up to 17 chars
    amount: Decimal  # In dollars
    individual_name: str  # Up to 22 chars

    # Optional fields
    individual_id: str = ""  # Up to 15 chars
    discretionary_data: str = ""  # 2 chars
    addenda_text: str = ""  # Up to 80 chars per addenda


@dataclass
class ACHBatch:
    """Simplified batch data for building ACH files."""

    # Required fields
    company_name: str  # Up to 16 chars
    company_id: str  # 10 chars (1 + 9 digit tax ID)
    originating_dfi: str  # 8 chars (first 8 of routing)
    entries: List[ACHEntry]

    # Optional fields
    sec_code: SECCode = SECCode.PPD
    entry_description: str = "PAYMENT"  # Up to 10 chars
    effective_date: Optional[date] = None
    company_discretionary: str = ""  # Up to 20 chars
    descriptive_date: str = ""  # 6 chars


class NACHABuilder:
    """
    Builder for NACHA ACH files.

    Creates properly formatted NACHA files with automatic
    calculation of hash, totals, and sequence numbers.
    """

    def __init__(
        self,
        immediate_destination: str,
        immediate_origin: str,
        destination_name: str = "",
        origin_name: str = "",
    ):
        """
        Initialize builder.

        Args:
            immediate_destination: Destination routing number (9 digits)
            immediate_origin: Origin company ID (9 digits or tax ID)
            destination_name: Destination bank name (up to 23 chars)
            origin_name: Origin company name (up to 23 chars)
        """
        self.immediate_destination = self._format_routing(immediate_destination)
        self.immediate_origin = self._format_routing(immediate_origin)
        self.destination_name = destination_name[:23]
        self.origin_name = origin_name[:23]

        self._batches: List[ACHBatch] = []
        self._file_id_modifier = "A"

    def add_batch(self, batch: ACHBatch) -> "NACHABuilder":
        """Add a batch to the file."""
        self._batches.append(batch)
        return self

    def add_credit_entry(
        self,
        batch_index: int,
        routing_number: str,
        account_number: str,
        amount: Decimal,
        name: str,
        account_type: str = "checking",
        individual_id: str = "",
        addenda: str = "",
    ) -> "NACHABuilder":
        """Add a credit entry to an existing batch."""
        tc = TransactionCode.CHECKING_CREDIT if account_type == "checking" else TransactionCode.SAVINGS_CREDIT

        entry = ACHEntry(
            transaction_code=tc,
            routing_number=routing_number,
            account_number=account_number,
            amount=amount,
            individual_name=name,
            individual_id=individual_id,
            addenda_text=addenda,
        )

        self._batches[batch_index].entries.append(entry)
        return self

    def add_debit_entry(
        self,
        batch_index: int,
        routing_number: str,
        account_number: str,
        amount: Decimal,
        name: str,
        account_type: str = "checking",
        individual_id: str = "",
        addenda: str = "",
    ) -> "NACHABuilder":
        """Add a debit entry to an existing batch."""
        tc = TransactionCode.CHECKING_DEBIT if account_type == "checking" else TransactionCode.SAVINGS_DEBIT

        entry = ACHEntry(
            transaction_code=tc,
            routing_number=routing_number,
            account_number=account_number,
            amount=amount,
            individual_name=name,
            individual_id=individual_id,
            addenda_text=addenda,
        )

        self._batches[batch_index].entries.append(entry)
        return self

    def build(self) -> str:
        """
        Build NACHA file content.

        Returns:
            Complete NACHA file as string
        """
        lines = []

        # File Header
        file_header = self._build_file_header()
        lines.append(file_header)

        # Track totals
        total_debit = Decimal("0")
        total_credit = Decimal("0")
        total_entry_count = 0
        entry_hash_sum = 0
        batch_count = 0

        # Process batches
        for batch_num, batch in enumerate(self._batches, 1):
            batch_count += 1
            batch_debit = Decimal("0")
            batch_credit = Decimal("0")
            batch_entry_count = 0
            batch_hash = 0

            # Determine service class based on entries
            has_credits = any(e.transaction_code.is_credit for e in batch.entries)
            has_debits = any(e.transaction_code.is_debit for e in batch.entries)

            if has_credits and has_debits:
                scc = ServiceClassCode.MIXED
            elif has_credits:
                scc = ServiceClassCode.CREDITS_ONLY
            else:
                scc = ServiceClassCode.DEBITS_ONLY

            # Batch Header
            batch_header = self._build_batch_header(batch, batch_num, scc)
            lines.append(batch_header)

            # Entry Details
            trace_seq = 0
            for entry in batch.entries:
                trace_seq += 1
                batch_entry_count += 1
                total_entry_count += 1

                # Add to hash (first 8 of routing)
                routing_8 = entry.routing_number[:8]
                batch_hash += int(routing_8)
                entry_hash_sum += int(routing_8)

                # Track amounts
                if entry.transaction_code.is_credit:
                    batch_credit += entry.amount
                    total_credit += entry.amount
                else:
                    batch_debit += entry.amount
                    total_debit += entry.amount

                # Build entry
                has_addenda = bool(entry.addenda_text)
                trace_number = f"{batch.originating_dfi}{trace_seq:07d}"

                entry_line = self._build_entry_detail(entry, trace_number, has_addenda)
                lines.append(entry_line)

                # Addenda
                if has_addenda:
                    batch_entry_count += 1
                    total_entry_count += 1
                    addenda_line = self._build_addenda(entry.addenda_text, 1, trace_seq)
                    lines.append(addenda_line)

            # Batch Control
            batch_control = self._build_batch_control(
                scc=scc,
                entry_count=batch_entry_count,
                entry_hash=batch_hash,
                total_debit=batch_debit,
                total_credit=batch_credit,
                company_id=batch.company_id,
                originating_dfi=batch.originating_dfi,
                batch_number=batch_num,
            )
            lines.append(batch_control)

        # File Control
        block_count = (len(lines) + 1 + 9) // 10  # +1 for file control, round up
        file_control = self._build_file_control(
            batch_count=batch_count,
            block_count=block_count,
            entry_count=total_entry_count,
            entry_hash=entry_hash_sum,
            total_debit=total_debit,
            total_credit=total_credit,
        )
        lines.append(file_control)

        # Pad to complete block (10 records per block)
        while len(lines) % 10 != 0:
            lines.append("9" * RECORD_LENGTH)

        return "\n".join(lines)

    def build_file(self) -> NACHAFile:
        """Build and return as NACHAFile object."""
        content = self.build()

        from ai_engine.domains.banking.protocols.payments.domestic.ach.nacha_parser import NACHAParser

        parser = NACHAParser()
        return parser.parse(content)

    def _format_routing(self, routing: str) -> str:
        """Format routing number with leading space."""
        routing = routing.replace("-", "").replace(" ", "")
        return f" {routing[:9]:>9}"

    def _build_file_header(self) -> str:
        """Build File Header Record (Type 1)."""
        now = datetime.now()

        record = (
            "1"  # Record Type (1)
            "01"  # Priority Code (2)
            f"{self.immediate_destination:10}"  # Immediate Destination (10)
            f"{self.immediate_origin:10}"  # Immediate Origin (10)
            f"{now.strftime('%y%m%d')}"  # File Creation Date (6)
            f"{now.strftime('%H%M')}"  # File Creation Time (4)
            f"{self._file_id_modifier}"  # File ID Modifier (1)
            "094"  # Record Size (3)
            "10"  # Blocking Factor (2)
            "1"  # Format Code (1)
            f"{self.destination_name:23}"  # Destination Name (23)
            f"{self.origin_name:23}"  # Origin Name (23)
            f"{'':8}"  # Reference Code (8)
        )

        return record[:RECORD_LENGTH]

    def _build_batch_header(
        self,
        batch: ACHBatch,
        batch_number: int,
        scc: ServiceClassCode,
    ) -> str:
        """Build Batch Header Record (Type 5)."""
        effective = batch.effective_date or date.today()

        record = (
            "5"  # Record Type (1)
            f"{scc.code}"  # Service Class Code (3)
            f"{batch.company_name:16}"  # Company Name (16)
            f"{batch.company_discretionary:20}"  # Company Discretionary Data (20)
            f"{batch.company_id:10}"  # Company Identification (10)
            f"{batch.sec_code.code}"  # Standard Entry Class (3)
            f"{batch.entry_description:10}"  # Company Entry Description (10)
            f"{batch.descriptive_date:6}"  # Company Descriptive Date (6)
            f"{effective.strftime('%y%m%d')}"  # Effective Entry Date (6)
            f"{'':3}"  # Settlement Date (3) - ACH operator
            "1"  # Originator Status Code (1)
            f"{batch.originating_dfi:8}"  # Originating DFI (8)
            f"{batch_number:07d}"  # Batch Number (7)
        )

        return record[:RECORD_LENGTH]

    def _build_entry_detail(
        self,
        entry: ACHEntry,
        trace_number: str,
        has_addenda: bool,
    ) -> str:
        """Build Entry Detail Record (Type 6)."""
        routing = entry.routing_number.replace("-", "").replace(" ", "")
        routing_8 = routing[:8]
        check_digit = routing[8] if len(routing) > 8 else self._calculate_check_digit(routing_8)

        # Amount in cents, right justified
        amount_cents = int(entry.amount * 100)

        record = (
            "6"  # Record Type (1)
            f"{entry.transaction_code.code}"  # Transaction Code (2)
            f"{routing_8:8}"  # Receiving DFI (8)
            f"{check_digit}"  # Check Digit (1)
            f"{entry.account_number:17}"  # DFI Account Number (17)
            f"{amount_cents:010d}"  # Amount (10)
            f"{entry.individual_id:15}"  # Individual Identification (15)
            f"{entry.individual_name:22}"  # Individual Name (22)
            f"{entry.discretionary_data:2}"  # Discretionary Data (2)
            f"{'1' if has_addenda else '0'}"  # Addenda Record Indicator (1)
            f"{trace_number:15}"  # Trace Number (15)
        )

        return record[:RECORD_LENGTH]

    def _build_addenda(
        self,
        payment_info: str,
        sequence: int,
        entry_sequence: int,
    ) -> str:
        """Build Addenda Record (Type 7)."""
        record = (
            "7"  # Record Type (1)
            "05"  # Addenda Type Code (2)
            f"{payment_info:80}"  # Payment Related Information (80)
            f"{sequence:04d}"  # Addenda Sequence Number (4)
            f"{entry_sequence:07d}"  # Entry Detail Sequence Number (7)
        )

        return record[:RECORD_LENGTH]

    def _build_batch_control(
        self,
        scc: ServiceClassCode,
        entry_count: int,
        entry_hash: int,
        total_debit: Decimal,
        total_credit: Decimal,
        company_id: str,
        originating_dfi: str,
        batch_number: int,
    ) -> str:
        """Build Batch Control Record (Type 8)."""
        # Hash is last 10 digits
        hash_10 = str(entry_hash)[-10:].zfill(10)

        # Amounts in cents
        debit_cents = int(total_debit * 100)
        credit_cents = int(total_credit * 100)

        record = (
            "8"  # Record Type (1)
            f"{scc.code}"  # Service Class Code (3)
            f"{entry_count:06d}"  # Entry/Addenda Count (6)
            f"{hash_10}"  # Entry Hash (10)
            f"{debit_cents:012d}"  # Total Debit Amount (12)
            f"{credit_cents:012d}"  # Total Credit Amount (12)
            f"{company_id:10}"  # Company Identification (10)
            f"{'':19}"  # Message Authentication Code (19)
            f"{'':6}"  # Reserved (6)
            f"{originating_dfi:8}"  # Originating DFI (8)
            f"{batch_number:07d}"  # Batch Number (7)
        )

        return record[:RECORD_LENGTH]

    def _build_file_control(
        self,
        batch_count: int,
        block_count: int,
        entry_count: int,
        entry_hash: int,
        total_debit: Decimal,
        total_credit: Decimal,
    ) -> str:
        """Build File Control Record (Type 9)."""
        # Hash is last 10 digits
        hash_10 = str(entry_hash)[-10:].zfill(10)

        # Amounts in cents
        debit_cents = int(total_debit * 100)
        credit_cents = int(total_credit * 100)

        record = (
            "9"  # Record Type (1)
            f"{batch_count:06d}"  # Batch Count (6)
            f"{block_count:06d}"  # Block Count (6)
            f"{entry_count:08d}"  # Entry/Addenda Count (8)
            f"{hash_10}"  # Entry Hash (10)
            f"{debit_cents:012d}"  # Total Debit Amount (12)
            f"{credit_cents:012d}"  # Total Credit Amount (12)
            f"{'':39}"  # Reserved (39)
        )

        return record[:RECORD_LENGTH]

    def _calculate_check_digit(self, routing_8: str) -> str:
        """Calculate ABA routing number check digit."""
        if len(routing_8) < 8:
            routing_8 = routing_8.zfill(8)

        weights = [3, 7, 1, 3, 7, 1, 3, 7]
        total = sum(int(routing_8[i]) * weights[i] for i in range(8))
        check = (10 - (total % 10)) % 10
        return str(check)


def create_payroll_file(
    company_name: str,
    company_id: str,
    originating_bank_routing: str,
    destination_bank_routing: str,
    employees: List[Dict[str, Any]],
) -> str:
    """
    Helper function to create a payroll direct deposit file.

    Args:
        company_name: Employer name
        company_id: Employer tax ID (9 digits)
        originating_bank_routing: Originator's bank routing
        destination_bank_routing: Destination bank (ACH operator)
        employees: List of dicts with keys:
            - name: str
            - routing_number: str
            - account_number: str
            - amount: Decimal
            - account_type: str ("checking" or "savings")

    Returns:
        NACHA file content
    """
    builder = NACHABuilder(
        immediate_destination=destination_bank_routing,
        immediate_origin=company_id,
        destination_name="",
        origin_name=company_name,
    )

    entries = []
    for emp in employees:
        tc = (
            TransactionCode.CHECKING_CREDIT
            if emp.get("account_type", "checking") == "checking"
            else TransactionCode.SAVINGS_CREDIT
        )

        entries.append(
            ACHEntry(
                transaction_code=tc,
                routing_number=emp["routing_number"],
                account_number=emp["account_number"],
                amount=Decimal(str(emp["amount"])),
                individual_name=emp["name"],
                individual_id=emp.get("employee_id", ""),
            )
        )

    batch = ACHBatch(
        company_name=company_name,
        company_id=f"1{company_id[:9]}",  # 1 + tax ID
        originating_dfi=originating_bank_routing[:8],
        entries=entries,
        sec_code=SECCode.PPD,
        entry_description="PAYROLL",
    )

    builder.add_batch(batch)

    return builder.build()
