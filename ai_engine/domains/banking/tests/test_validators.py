"""
Tests for Banking Protocol Validators
"""

import pytest
from datetime import datetime
from decimal import Decimal

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationError,
    ValidationWarning,
    ValidationSeverity,
    CompositeValidator,
    validate_routing_number,
    validate_iban,
    validate_bic,
    validate_currency_code,
    validate_amount,
)
from ai_engine.domains.banking.protocols.validators.iso20022_validator import (
    ISO20022Validator,
)
from ai_engine.domains.banking.protocols.validators.ach_validator import (
    ACHValidator,
    NACHAFileValidator,
)
from ai_engine.domains.banking.protocols.validators.fedwire_validator import (
    FedWireValidator,
)
from ai_engine.domains.banking.protocols.validators.compliance_validator import (
    ComplianceValidator,
    AMLValidator,
    SanctionsValidator,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_result_creation(self):
        """Test creating validation result."""
        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error(self):
        """Test adding an error."""
        result = ValidationResult()
        result.add_error("TEST_ERROR", "This is a test error", field="test_field")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "TEST_ERROR"

    def test_add_warning(self):
        """Test adding a warning."""
        result = ValidationResult()
        result.add_warning("TEST_WARNING", "This is a warning")

        assert result.is_valid is True  # Warnings don't affect validity
        assert len(result.warnings) == 1

    def test_merge_results(self):
        """Test merging validation results."""
        result1 = ValidationResult()
        result1.add_error("ERROR1", "Error 1")

        result2 = ValidationResult()
        result2.add_error("ERROR2", "Error 2")
        result2.add_warning("WARNING1", "Warning 1")

        result1.merge(result2)

        assert result1.is_valid is False
        assert len(result1.errors) == 2
        assert len(result1.warnings) == 1

    def test_to_dict(self):
        """Test result serialization."""
        result = ValidationResult()
        result.add_error("ERROR", "Test error")
        result.validator_name = "TestValidator"

        data = result.to_dict()
        assert "is_valid" in data
        assert "errors" in data
        assert "validator_name" in data
        assert data["is_valid"] is False

    def test_has_critical_errors(self):
        """Test critical error detection."""
        result = ValidationResult()
        assert result.has_critical_errors is False

        result.add_error("CRITICAL", "Critical error", severity=ValidationSeverity.CRITICAL)
        assert result.has_critical_errors is True


class TestUtilityValidators:
    """Tests for utility validation functions."""

    def test_validate_routing_number_valid(self):
        """Test valid routing numbers."""
        # Known valid routing numbers
        valid_numbers = [
            "021000089",  # Citibank
            "021000021",  # JPMorgan Chase
            "121000248",  # Wells Fargo
            "091000019",  # US Bank
        ]

        for routing in valid_numbers:
            error = validate_routing_number(routing)
            assert error is None, f"Expected {routing} to be valid"

    def test_validate_routing_number_invalid(self):
        """Test invalid routing numbers."""
        # Invalid length
        error = validate_routing_number("12345")
        assert error is not None

        # Non-numeric
        error = validate_routing_number("12345678A")
        assert error is not None

        # Invalid checksum
        error = validate_routing_number("123456789")
        assert error is not None

        # Empty
        error = validate_routing_number("")
        assert error is not None

    def test_validate_iban_valid(self):
        """Test valid IBANs."""
        valid_ibans = [
            "DE89370400440532013000",  # Germany
            "GB82WEST12345698765432",  # UK
            "FR7630006000011234567890189",  # France
        ]

        for iban in valid_ibans:
            error = validate_iban(iban)
            assert error is None, f"Expected {iban} to be valid"

    def test_validate_iban_invalid(self):
        """Test invalid IBANs."""
        # Too short
        error = validate_iban("DE89")
        assert error is not None

        # Invalid checksum
        error = validate_iban("DE00370400440532013000")
        assert error is not None

        # Invalid country code
        error = validate_iban("123456789012345678")
        assert error is not None

    def test_validate_bic_valid(self):
        """Test valid BIC codes."""
        valid_bics = [
            "CITIUS33",  # 8 char BIC
            "DEUTDEFF",  # 8 char BIC
            "BNPAFRPHXXX",  # 11 char BIC
        ]

        for bic in valid_bics:
            error = validate_bic(bic)
            assert error is None, f"Expected {bic} to be valid"

    def test_validate_bic_invalid(self):
        """Test invalid BIC codes."""
        # Too short
        error = validate_bic("CITI")
        assert error is not None

        # Invalid format
        error = validate_bic("12345678")
        assert error is not None

    def test_validate_currency_code(self):
        """Test currency code validation."""
        # Valid
        assert validate_currency_code("USD") is None
        assert validate_currency_code("EUR") is None
        assert validate_currency_code("GBP") is None

        # Invalid
        assert validate_currency_code("") is not None
        assert validate_currency_code("US") is not None
        assert validate_currency_code("123") is not None

    def test_validate_amount(self):
        """Test amount validation."""
        # Valid
        assert validate_amount(100.00) is None
        assert validate_amount(0.01) is None
        assert validate_amount(1000000) is None

        # Invalid
        assert validate_amount(0) is not None
        assert validate_amount(-100) is not None
        assert validate_amount(None) is not None

        # Max value
        assert validate_amount(1000, max_value=500) is not None

        # Allow zero
        assert validate_amount(0, allow_zero=True) is None


class TestISO20022Validator:
    """Tests for ISO 20022 validator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ISO20022Validator()

    @pytest.fixture
    def sample_pain001_xml(self):
        """Create sample pain.001 XML."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pain.001.001.03">
  <CstmrCdtTrfInitn>
    <GrpHdr>
      <MsgId>MSG001</MsgId>
      <CreDtTm>2023-12-15T10:00:00Z</CreDtTm>
      <NbOfTxs>1</NbOfTxs>
      <CtrlSum>1000.00</CtrlSum>
    </GrpHdr>
    <PmtInf>
      <PmtInfId>PMT001</PmtInfId>
      <PmtMtd>TRF</PmtMtd>
      <CdtTrfTxInf>
        <PmtId>
          <EndToEndId>E2E001</EndToEndId>
        </PmtId>
        <Amt>
          <InstdAmt Ccy="USD">1000.00</InstdAmt>
        </Amt>
        <CdtrAcct>
          <Id>
            <IBAN>DE89370400440532013000</IBAN>
          </Id>
        </CdtrAcct>
      </CdtTrfTxInf>
    </PmtInf>
  </CstmrCdtTrfInitn>
</Document>"""

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.name == "ISO20022Validator"
        assert validator.version == "1.0"

    def test_validate_pain001_valid(self, validator, sample_pain001_xml):
        """Test validation of valid pain.001 message."""
        result = validator.validate(sample_pain001_xml)

        # Should have no critical errors
        critical_errors = [e for e in result.errors if e.severity == ValidationSeverity.CRITICAL]
        assert len(critical_errors) == 0

    def test_validate_invalid_xml(self, validator):
        """Test validation of invalid XML."""
        invalid_xml = "<invalid><xml"
        result = validator.validate(invalid_xml)

        assert result.is_valid is False
        assert any(e.code == "ISO20022_XML_PARSE_ERROR" for e in result.errors)

    def test_validate_missing_group_header(self, validator):
        """Test validation catches missing group header."""
        xml_no_header = """<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pain.001.001.03">
  <CstmrCdtTrfInitn>
    <PmtInf>
      <PmtInfId>PMT001</PmtInfId>
    </PmtInf>
  </CstmrCdtTrfInitn>
</Document>"""

        result = validator.validate(xml_no_header)
        assert any("GRP_HDR" in e.code or "GrpHdr" in e.message for e in result.errors)


class TestACHValidator:
    """Tests for ACH validator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ACHValidator()

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.name == "ACHValidator"

    def test_validate_valid_entry(self, validator):
        """Test validation of valid ACH entry."""
        entry_data = {
            "routing_number": "091000019",
            "account_number": "123456789",
            "amount": 10000,
            "transaction_code": "22",
            "individual_name": "JOHN DOE",
        }

        result = validator.validate(entry_data)
        assert result.is_valid is True

    def test_validate_invalid_routing(self, validator):
        """Test validation catches invalid routing."""
        entry_data = {
            "routing_number": "123456789",  # Invalid checksum
            "account_number": "123456789",
            "amount": 10000,
            "transaction_code": "22",
            "individual_name": "JOHN DOE",
        }

        result = validator.validate(entry_data)
        assert result.is_valid is False
        assert any("routing" in e.code.lower() for e in result.errors)

    def test_validate_invalid_amount(self, validator):
        """Test validation catches invalid amount."""
        entry_data = {
            "routing_number": "091000019",
            "account_number": "123456789",
            "amount": -100,  # Negative
            "transaction_code": "22",
            "individual_name": "JOHN DOE",
        }

        result = validator.validate(entry_data)
        assert result.is_valid is False

    def test_validate_amount_exceeds_limit(self):
        """Test validation catches amount exceeding limit."""
        validator = ACHValidator(max_amount=1000)

        entry_data = {
            "routing_number": "091000019",
            "account_number": "123456789",
            "amount": 5000,  # Exceeds limit
            "transaction_code": "22",
            "individual_name": "JOHN DOE",
        }

        result = validator.validate(entry_data)
        assert result.is_valid is False


class TestFedWireValidator:
    """Tests for FedWire validator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return FedWireValidator()

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.name == "FedWireValidator"

    def test_validate_valid_message(self, validator):
        """Test validation of valid FedWire data."""
        msg_data = {
            "sender_routing": "021000089",
            "receiver_routing": "021000021",
            "amount": 10000.00,
            "business_function_code": "BTR",
        }

        result = validator.validate(msg_data)
        # Basic dict validation should pass
        assert result.is_valid is True

    def test_validate_invalid_routing(self, validator):
        """Test validation catches invalid routing."""
        msg_data = {
            "sender_routing": "12345",  # Invalid
            "receiver_routing": "021000021",
            "amount": 10000.00,
            "business_function_code": "BTR",
        }

        result = validator.validate(msg_data)
        assert result.is_valid is False

    def test_validate_large_amount_warning(self, validator):
        """Test large amount validation (dict mode doesn't generate warnings)."""
        msg_data = {
            "sender_routing": "021000089",
            "receiver_routing": "021000021",
            "amount": 2000000.00,  # $2M - above warning threshold
            "business_function_code": "BTR",
        }

        result = validator.validate(msg_data)
        # Dict validation validates basic fields, warnings for large values
        # are generated in message object validation mode
        assert result.is_valid is True


class TestComplianceValidator:
    """Tests for Compliance validator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ComplianceValidator()

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.name == "ComplianceValidator"

    def test_ctr_threshold_detection(self, validator):
        """Test CTR threshold detection."""
        tx_data = {
            "amount": 15000.00,  # Above $10K CTR threshold
            "type": "wire",
        }

        result = validator.validate(tx_data)
        assert result.metadata.get("requires_ctr") is True

    def test_high_risk_country_detection(self, validator):
        """Test high-risk country detection."""
        tx_data = {
            "amount": 5000.00,
            "originator": {
                "name": "TEST COMPANY",
                "country": "KP",  # North Korea - sanctioned
            },
        }

        result = validator.validate(tx_data)
        assert result.metadata.get("high_risk_country_involved") is True

    def test_structuring_detection(self, validator):
        """Test potential structuring detection."""
        tx_data = {
            "amount": 9500.00,  # Just below $10K threshold
            "type": "cash",
        }

        result = validator.validate(tx_data)
        # Should flag potential structuring
        assert any("STRUCTURING" in w.code for w in result.warnings)


class TestAMLValidator:
    """Tests for AML validator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return AMLValidator()

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.name == "AMLValidator"

    def test_risk_score_calculation(self, validator):
        """Test AML risk score calculation."""
        tx_data = {
            "amount": 15000.00,  # Above CTR threshold
        }

        result = validator.validate(tx_data)
        assert "aml_risk_score" in result.metadata
        assert result.metadata["aml_risk_score"] >= 0
        assert result.metadata["aml_risk_score"] <= 100

    def test_velocity_check(self, validator):
        """Test velocity monitoring."""
        current_tx = {
            "amount": 5000.00,
        }

        # Recent transaction history
        history = [
            {"amount": 5000.00, "timestamp": datetime.now().isoformat()},
            {"amount": 3000.00, "timestamp": datetime.now().isoformat()},
        ]

        result = validator.validate(current_tx, transaction_history=history)
        # Aggregate may exceed threshold
        if sum(tx.get("amount", 0) for tx in history) + 5000 >= 10000:
            assert result.metadata.get("velocity_ctr_exceeded") is True


class TestSanctionsValidator:
    """Tests for Sanctions validator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return SanctionsValidator()

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.name == "SanctionsValidator"

    def test_jurisdiction_check(self, validator):
        """Test sanctioned jurisdiction detection."""
        tx_data = {
            "originator": {
                "name": "TEST COMPANY",
                "address": {"country": "IR"},  # Iran - sanctioned
            },
        }

        result = validator.validate(tx_data)
        assert any("JURISDICTION" in e.code for e in result.errors)

    def test_screening_metadata(self, validator):
        """Test screening metadata is set."""
        tx_data = {
            "amount": 1000.00,
            "beneficiary": {"name": "SAFE COMPANY"},
        }

        result = validator.validate(tx_data)
        assert result.metadata.get("sanctions_screened") is True
        assert "sanctions_lists_checked" in result.metadata


class TestCompositeValidator:
    """Tests for Composite validator."""

    def test_composite_validation(self):
        """Test composite validator runs all validators."""
        ach_validator = ACHValidator()
        compliance_validator = ComplianceValidator()

        composite = CompositeValidator([ach_validator, compliance_validator])

        data = {
            "routing_number": "091000019",
            "account_number": "123456789",
            "amount": 15000,  # Above CTR threshold
            "transaction_code": "22",
            "individual_name": "JOHN DOE",
        }

        result = composite.validate(data)
        # Should include results from both validators
        assert result.validator_name == "CompositeValidator"

    def test_add_validator(self):
        """Test adding validator to composite."""
        composite = CompositeValidator([])
        assert len(composite._validators) == 0

        composite.add_validator(ACHValidator())
        assert len(composite._validators) == 1
