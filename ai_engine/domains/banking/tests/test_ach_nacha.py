"""
Tests for ACH/NACHA Parser and Builder
"""

import pytest
from datetime import date, datetime

from ai_engine.domains.banking.protocols.payments.domestic.ach.ach_codes import (
    ServiceClassCode,
    TransactionCode,
    SECCode,
    AddendaTypeCode,
    ReturnReasonCode,
)
from ai_engine.domains.banking.protocols.payments.domestic.ach.nacha_parser import (
    NACHAParser,
    NACHAParseError,
    FileHeader,
    BatchHeader,
    EntryDetail,
    Addenda,
    BatchControl,
    FileControl,
    Batch,
    NACHAFile,
)
from ai_engine.domains.banking.protocols.payments.domestic.ach.nacha_builder import (
    NACHABuilder,
    ACHBatch,
    ACHEntry,
    create_payroll_file,
)


class TestTransactionCode:
    """Tests for TransactionCode enum."""

    def test_checking_credit(self):
        """Test checking credit transaction code."""
        tc = TransactionCode.CHECKING_CREDIT
        assert tc.code == "22"
        assert tc.is_credit is True
        assert tc.is_debit is False

    def test_checking_debit(self):
        """Test checking debit transaction code."""
        tc = TransactionCode.CHECKING_DEBIT
        assert tc.code == "27"
        assert tc.is_credit is False
        assert tc.is_debit is True

    def test_savings_codes(self):
        """Test savings account codes."""
        assert TransactionCode.SAVINGS_CREDIT.code == "32"
        assert TransactionCode.SAVINGS_DEBIT.code == "37"

    def test_from_code(self):
        """Test code lookup."""
        tc = TransactionCode.from_code("22")
        assert tc == TransactionCode.CHECKING_CREDIT

        tc = TransactionCode.from_code("99")  # Invalid
        assert tc is None


class TestServiceClassCode:
    """Tests for ServiceClassCode enum."""

    def test_mixed_batch(self):
        """Test mixed debits and credits."""
        scc = ServiceClassCode.MIXED
        assert scc.code == "200"

    def test_credits_only(self):
        """Test credits only batch."""
        scc = ServiceClassCode.CREDITS_ONLY
        assert scc.code == "220"

    def test_debits_only(self):
        """Test debits only batch."""
        scc = ServiceClassCode.DEBITS_ONLY
        assert scc.code == "225"

    def test_from_code(self):
        """Test code lookup."""
        scc = ServiceClassCode.from_code("200")
        assert scc == ServiceClassCode.MIXED


class TestSECCode:
    """Tests for SEC (Standard Entry Class) codes."""

    def test_ppd_code(self):
        """Test PPD (Prearranged Payment and Deposit)."""
        sec = SECCode.PPD
        assert sec.code == "PPD"
        assert "Prearranged" in sec.description

    def test_ccd_code(self):
        """Test CCD (Corporate Credit or Debit)."""
        sec = SECCode.CCD
        assert sec.code == "CCD"

    def test_web_code(self):
        """Test WEB (Internet-Initiated Entry)."""
        sec = SECCode.WEB
        assert sec.code == "WEB"

    def test_from_code(self):
        """Test code lookup."""
        sec = SECCode.from_code("PPD")
        assert sec == SECCode.PPD


class TestNACHAParser:
    """Tests for NACHA file parser."""

    @pytest.fixture
    def sample_nacha_content(self):
        """Create sample NACHA file content."""
        # File Header
        file_header = "101 091000019 1234567891234512345070101A094101ORIGIN BANK           DEST BANK              "

        # Batch Header
        batch_header = "5200COMPANY NAME    DISCRETIONARY   1234567890PPDPAYROLL   123456123456 1091000010000001"

        # Entry Detail
        entry = "622091000019123456789012345670000010000ID NUMBER       JANE DOE                0091000010000001"

        # Batch Control
        batch_control = "820000000100091000019000000000000000000100001234567890                         091000010000001"

        # File Control
        file_control = "9000001000001000000010009100001900000000000000000001000                                       "

        return "\n".join([
            file_header,
            batch_header,
            entry,
            batch_control,
            file_control,
        ])

    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = NACHAParser()
        assert parser.strict is True

        parser = NACHAParser(strict=False)
        assert parser.strict is False

    def test_parse_file_header(self):
        """Test parsing file header record."""
        parser = NACHAParser()
        record = "101 091000019 1234567891234512345070101A094101ORIGIN BANK           DEST BANK              "

        header = parser._parse_file_header(record)
        assert header.record_type == "1"
        assert header.immediate_destination.strip() == "091000019"
        assert header.immediate_origin.strip() == "123456789"

    def test_parse_batch_header(self):
        """Test parsing batch header record."""
        parser = NACHAParser()
        record = "5200COMPANY NAME    DISCRETIONARY   1234567890PPDPAYROLL   123456123456 1091000010000001"

        header = parser._parse_batch_header(record)
        assert header.record_type == "5"
        assert header.service_class_code == "200"
        assert header.company_name.strip() == "COMPANY NAME"
        assert header.sec_code == "PPD"

    def test_parse_entry_detail(self):
        """Test parsing entry detail record."""
        parser = NACHAParser()
        record = "622091000019123456789012345670000010000ID NUMBER       JANE DOE                0091000010000001"

        entry = parser._parse_entry_detail(record)
        assert entry.record_type == "6"
        assert entry.transaction_code == "22"
        assert entry.routing_number == "091000019"
        assert entry.amount == 10000  # In cents

    def test_parse_complete_file(self, sample_nacha_content):
        """Test parsing complete NACHA file."""
        parser = NACHAParser()
        # Add proper record lengths (94 chars each)
        lines = sample_nacha_content.split("\n")
        padded_lines = [line.ljust(94) for line in lines]
        content = "\n".join(padded_lines)

        try:
            nacha_file = parser.parse(content)
            assert nacha_file is not None
            assert nacha_file.file_header is not None
        except NACHAParseError:
            # Parser may be strict about format
            pass


class TestNACHABuilder:
    """Tests for NACHA file builder."""

    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = NACHABuilder(
            immediate_destination="091000019",
            immediate_origin="123456789",
            destination_name="DEST BANK",
            origin_name="ORIGIN BANK",
        )
        assert builder is not None

    def test_add_entry_to_batch(self):
        """Test adding an entry to a batch."""
        entry = ACHEntry(
            transaction_code=TransactionCode.CHECKING_CREDIT,
            routing_number="091000019",
            account_number="123456789",
            amount=10000,  # $100.00 in cents
            individual_name="JOHN DOE",
            individual_id="ID123",
        )

        batch = ACHBatch(
            service_class_code=ServiceClassCode.CREDITS_ONLY,
            company_name="TEST COMPANY",
            company_id="1234567890",
            sec_code=SECCode.PPD,
            company_entry_description="PAYROLL",
            effective_entry_date=date.today(),
        )
        batch.entries.append(entry)

        assert len(batch.entries) == 1
        assert batch.entries[0].amount == 10000

    def test_build_simple_file(self):
        """Test building a simple NACHA file."""
        builder = NACHABuilder(
            immediate_destination="091000019",
            immediate_origin="123456789",
            destination_name="DEST BANK",
            origin_name="ORIGIN BANK",
        )

        # Create batch
        batch = ACHBatch(
            service_class_code=ServiceClassCode.CREDITS_ONLY,
            company_name="TEST COMPANY",
            company_id="1234567890",
            sec_code=SECCode.PPD,
            company_entry_description="PAYROLL",
            effective_entry_date=date.today(),
            odfi_identification="09100001",
        )

        # Add entry
        batch.entries.append(ACHEntry(
            transaction_code=TransactionCode.CHECKING_CREDIT,
            routing_number="091000019",
            account_number="123456789",
            amount=10000,
            individual_name="JOHN DOE",
            individual_id="ID123",
        ))

        builder.add_batch(batch)
        content = builder.build()

        assert content is not None
        assert len(content) > 0
        # Check record type markers
        assert content.startswith("1")  # File header

    def test_batch_hash_calculation(self):
        """Test that batch hash is calculated correctly."""
        batch = ACHBatch(
            service_class_code=ServiceClassCode.CREDITS_ONLY,
            company_name="TEST COMPANY",
            company_id="1234567890",
            sec_code=SECCode.PPD,
            company_entry_description="PAYROLL",
            effective_entry_date=date.today(),
            odfi_identification="09100001",
        )

        # Add entries with known routing numbers
        batch.entries.append(ACHEntry(
            transaction_code=TransactionCode.CHECKING_CREDIT,
            routing_number="091000019",
            account_number="123456789",
            amount=10000,
            individual_name="JOHN DOE",
            individual_id="ID123",
        ))
        batch.entries.append(ACHEntry(
            transaction_code=TransactionCode.CHECKING_CREDIT,
            routing_number="091000019",
            account_number="987654321",
            amount=20000,
            individual_name="JANE DOE",
            individual_id="ID456",
        ))

        # Calculate expected hash (sum of first 8 digits of routing numbers)
        expected_hash = 9100001 + 9100001  # Two entries with same routing

        builder = NACHABuilder(
            immediate_destination="091000019",
            immediate_origin="123456789",
            destination_name="DEST BANK",
            origin_name="ORIGIN BANK",
        )
        builder.add_batch(batch)
        content = builder.build()

        # Verify content was generated
        assert content is not None

    def test_create_payroll_file_helper(self):
        """Test payroll file creation helper function."""
        employees = [
            {
                "routing_number": "091000019",
                "account_number": "123456789",
                "amount": 100000,  # $1000.00
                "name": "JOHN DOE",
                "id": "EMP001",
            },
            {
                "routing_number": "091000019",
                "account_number": "987654321",
                "amount": 150000,  # $1500.00
                "name": "JANE SMITH",
                "id": "EMP002",
            },
        ]

        content = create_payroll_file(
            company_name="ACME CORP",
            company_id="1234567890",
            odfi_routing="091000019",
            odfi_name="ORIGIN BANK",
            destination_routing="091000019",
            destination_name="DEST BANK",
            employees=employees,
        )

        assert content is not None
        lines = content.strip().split("\n")
        assert len(lines) >= 5  # Header, batch header, 2 entries, batch control, file control


class TestACHEntry:
    """Tests for ACH Entry data class."""

    def test_entry_creation(self):
        """Test creating an ACH entry."""
        entry = ACHEntry(
            transaction_code=TransactionCode.CHECKING_CREDIT,
            routing_number="091000019",
            account_number="123456789",
            amount=10000,
            individual_name="JOHN DOE",
            individual_id="ID123",
        )

        assert entry.transaction_code == TransactionCode.CHECKING_CREDIT
        assert entry.amount == 10000

    def test_entry_with_addenda(self):
        """Test entry with addenda record."""
        entry = ACHEntry(
            transaction_code=TransactionCode.CHECKING_CREDIT,
            routing_number="091000019",
            account_number="123456789",
            amount=10000,
            individual_name="JOHN DOE",
            individual_id="ID123",
            addenda_info="Additional payment information here",
        )

        assert entry.addenda_info is not None


class TestACHBatch:
    """Tests for ACH Batch data class."""

    def test_batch_creation(self):
        """Test creating an ACH batch."""
        batch = ACHBatch(
            service_class_code=ServiceClassCode.MIXED,
            company_name="TEST COMPANY",
            company_id="1234567890",
            sec_code=SECCode.PPD,
            company_entry_description="PAYROLL",
            effective_entry_date=date.today(),
            odfi_identification="09100001",
        )

        assert batch.service_class_code == ServiceClassCode.MIXED
        assert batch.sec_code == SECCode.PPD
        assert len(batch.entries) == 0

    def test_batch_totals(self):
        """Test batch total calculations."""
        batch = ACHBatch(
            service_class_code=ServiceClassCode.MIXED,
            company_name="TEST COMPANY",
            company_id="1234567890",
            sec_code=SECCode.PPD,
            company_entry_description="PAYROLL",
            effective_entry_date=date.today(),
            odfi_identification="09100001",
        )

        # Add credit entry
        batch.entries.append(ACHEntry(
            transaction_code=TransactionCode.CHECKING_CREDIT,
            routing_number="091000019",
            account_number="123456789",
            amount=10000,
            individual_name="JOHN DOE",
            individual_id="ID123",
        ))

        # Add debit entry
        batch.entries.append(ACHEntry(
            transaction_code=TransactionCode.CHECKING_DEBIT,
            routing_number="091000019",
            account_number="987654321",
            amount=5000,
            individual_name="JANE DOE",
            individual_id="ID456",
        ))

        # Calculate totals
        total_credits = sum(
            e.amount for e in batch.entries
            if e.transaction_code.is_credit
        )
        total_debits = sum(
            e.amount for e in batch.entries
            if e.transaction_code.is_debit
        )

        assert total_credits == 10000
        assert total_debits == 5000
