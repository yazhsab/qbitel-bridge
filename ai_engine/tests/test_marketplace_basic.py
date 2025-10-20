"""
CRONOS AI - Basic Marketplace Tests

Unit tests for Protocol Marketplace core functionality without database dependencies.
"""

import pytest
import uuid
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from ai_engine.marketplace.protocol_validator import (
    ProtocolValidator,
    ValidationResult,
)


class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            validation_type="syntax_validation",
            status="passed",
            score=Decimal("95.0"),
            errors=[],
            warnings=["Minor warning"],
            test_results={"yaml_valid": True},
            metrics={},
        )

        assert result.validation_type == "syntax_validation"
        assert result.status == "passed"
        assert result.score == Decimal("95.0")
        assert len(result.warnings) == 1
        assert result.test_results["yaml_valid"] is True

    def test_validation_result_with_errors(self):
        """Test validation result with errors."""
        result = ValidationResult(
            validation_type="parser_testing",
            status="failed",
            score=Decimal("0.0"),
            errors=["Parser execution failed", "Missing dependencies"],
            warnings=[],
            test_results={},
            metrics={},
        )

        assert result.status == "failed"
        assert len(result.errors) == 2
        assert result.score == Decimal("0.0")


class TestProtocolValidator:
    """Test protocol validation logic."""

    @pytest.mark.asyncio
    async def test_validate_syntax_yaml_success(self):
        """Test successful YAML syntax validation."""
        validator = ProtocolValidator()

        # Mock protocol object
        mock_protocol = Mock()
        mock_protocol.spec_format = Mock(value="yaml")
        mock_protocol.spec_file_url = "s3://test/spec.yaml"

        valid_yaml = """
protocol_metadata:
  name: test-protocol
  version: 1.0.0
  category: finance

protocol_spec:
  message_format: binary
  fields:
    - id: 1
      name: field1
      type: string
"""

        with patch.object(validator, '_download_spec_file', return_value=valid_yaml):
            result = await validator.validate_syntax(mock_protocol)

        assert result.status == "passed"
        assert result.validation_type == "syntax_validation"
        assert len(result.errors) == 0
        assert result.score > Decimal("0")
        assert result.test_results.get("yaml_valid") is True

    @pytest.mark.asyncio
    async def test_validate_syntax_invalid_yaml(self):
        """Test YAML syntax validation with invalid YAML."""
        validator = ProtocolValidator()

        mock_protocol = Mock()
        mock_protocol.spec_format = Mock(value="yaml")
        mock_protocol.spec_file_url = "s3://test/spec.yaml"

        invalid_yaml = """
invalid: yaml: content:
  - missing
    indent
"""

        with patch.object(validator, '_download_spec_file', return_value=invalid_yaml):
            result = await validator.validate_syntax(mock_protocol)

        assert result.status == "failed"
        assert len(result.errors) > 0
        assert result.score == Decimal("0.0")

    @pytest.mark.asyncio
    async def test_validate_syntax_missing_required_fields(self):
        """Test syntax validation with missing required fields."""
        validator = ProtocolValidator()

        mock_protocol = Mock()
        mock_protocol.spec_format = Mock(value="yaml")
        mock_protocol.spec_file_url = "s3://test/spec.yaml"

        incomplete_yaml = """
protocol_metadata:
  name: test-protocol
  # Missing version and category
"""

        with patch.object(validator, '_download_spec_file', return_value=incomplete_yaml):
            result = await validator.validate_syntax(mock_protocol)

        # Should have errors for missing fields
        assert len(result.errors) > 0
        assert any("version" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_parser_testing_no_parser(self):
        """Test parser testing when no parser is provided."""
        validator = ProtocolValidator()

        mock_protocol = Mock()
        mock_protocol.parser_code_url = None
        mock_protocol.test_data_url = None

        result = await validator.test_parser(mock_protocol)

        assert result.status == "skipped"
        assert result.validation_type == "parser_testing"
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_security_scan_safe_code(self):
        """Test security scan with safe code."""
        validator = ProtocolValidator()

        mock_protocol = Mock()
        mock_protocol.parser_code_url = "s3://test/parser.py"

        safe_code = """
def parse_packet(data: bytes) -> dict:
    '''Safe parser implementation'''
    return {
        'field1': data[:4].decode('utf-8'),
        'field2': data[4:8]
    }
"""

        with patch.object(validator, '_download_parser_code', return_value=safe_code):
            result = await validator.security_scan(mock_protocol)

        assert result.validation_type == "security_scan"
        # Safe code should not have critical/high vulnerabilities
        vulns = result.test_results.get("vulnerabilities", {})
        assert vulns.get("critical", 0) == 0
        assert vulns.get("high", 0) == 0

    @pytest.mark.asyncio
    async def test_security_scan_dangerous_code(self):
        """Test security scan detects dangerous code patterns."""
        validator = ProtocolValidator()

        mock_protocol = Mock()
        mock_protocol.parser_code_url = "s3://test/parser.py"

        dangerous_code = """
def parse_packet(data: bytes) -> dict:
    # Dangerous: using eval
    result = eval(data.decode())
    # Dangerous: using exec
    exec("import os; os.system('ls')")
    return result
"""

        with patch.object(validator, '_download_parser_code', return_value=dangerous_code):
            result = await validator.security_scan(mock_protocol)

        assert result.validation_type == "security_scan"
        # Should detect dangerous patterns
        assert len(result.errors) > 0
        assert any("eval" in error.lower() or "exec" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_performance_benchmark(self):
        """Test performance benchmarking."""
        validator = ProtocolValidator()

        mock_protocol = Mock()

        result = await validator.performance_benchmark(mock_protocol)

        assert result.validation_type == "performance_benchmark"
        assert "throughput" in result.metrics
        assert "memory_usage" in result.metrics
        assert "latency_p50" in result.metrics
        assert "latency_p95" in result.metrics
        assert "latency_p99" in result.metrics

        # Check metrics are reasonable values
        assert result.metrics["throughput"] > 0
        assert result.metrics["memory_usage"] > 0
        assert result.metrics["latency_p50"] > 0

    def test_save_validation_result(self):
        """Test saving validation result to database."""
        validator = ProtocolValidator()

        mock_session = Mock()
        protocol_id = str(uuid.uuid4())

        result = ValidationResult(
            validation_type="syntax_validation",
            status="passed",
            score=Decimal("100.0"),
            errors=[],
            warnings=[],
            test_results={"test": "passed"},
            metrics={"throughput": 5000},
        )

        # Should not raise exception
        validator._save_validation_result(mock_session, protocol_id, result)

        # Verify session.add was called
        assert mock_session.add.called

    def test_determine_overall_status_all_passed(self):
        """Test determining overall status when all validations pass."""
        validator = ProtocolValidator()

        results = {
            "syntax_validation": ValidationResult("syntax_validation", "passed"),
            "parser_testing": ValidationResult("parser_testing", "passed"),
            "security_scan": ValidationResult("security_scan", "passed"),
            "performance_benchmark": ValidationResult("performance_benchmark", "passed"),
        }

        status = validator._determine_overall_status(results)
        assert status == "passed"

    def test_determine_overall_status_syntax_failed(self):
        """Test determining overall status when syntax validation fails."""
        validator = ProtocolValidator()

        results = {
            "syntax_validation": ValidationResult("syntax_validation", "failed"),
            "parser_testing": ValidationResult("parser_testing", "skipped"),
            "security_scan": ValidationResult("security_scan", "passed"),
        }

        status = validator._determine_overall_status(results)
        assert status == "failed"

    def test_determine_overall_status_security_failed(self):
        """Test determining overall status when security scan fails."""
        validator = ProtocolValidator()

        results = {
            "syntax_validation": ValidationResult("syntax_validation", "passed"),
            "parser_testing": ValidationResult("parser_testing", "passed"),
            "security_scan": ValidationResult("security_scan", "failed"),
        }

        status = validator._determine_overall_status(results)
        assert status == "failed"


class TestMarketplaceSchemas:
    """Test marketplace API schemas."""

    def test_import_schemas(self):
        """Test that marketplace schemas can be imported."""
        from ai_engine.api import marketplace_schemas

        assert marketplace_schemas is not None
        assert hasattr(marketplace_schemas, 'ProtocolSearchRequest')
        assert hasattr(marketplace_schemas, 'ProtocolSubmitRequest')
        assert hasattr(marketplace_schemas, 'ProtocolPurchaseRequest')

    def test_protocol_category_enum(self):
        """Test ProtocolCategory enum."""
        from ai_engine.api.marketplace_schemas import ProtocolCategoryEnum

        assert ProtocolCategoryEnum.FINANCE == "finance"
        assert ProtocolCategoryEnum.HEALTHCARE == "healthcare"
        assert ProtocolCategoryEnum.IOT == "iot"

    def test_license_type_enum(self):
        """Test LicenseType enum."""
        from ai_engine.api.marketplace_schemas import LicenseTypeEnum

        assert LicenseTypeEnum.FREE == "free"
        assert LicenseTypeEnum.PAID == "paid"
        assert LicenseTypeEnum.ENTERPRISE == "enterprise"


class TestMarketplaceIntegration:
    """Test marketplace integration components."""

    def test_import_integration_modules(self):
        """Test that integration modules can be imported."""
        from ai_engine.marketplace import knowledge_base_integration

        assert knowledge_base_integration is not None
        assert hasattr(knowledge_base_integration, 'MarketplaceKnowledgeBaseIntegration')
        assert hasattr(knowledge_base_integration, 'MarketplaceProtocolDeployer')

    @pytest.mark.asyncio
    async def test_parse_specification_yaml(self):
        """Test parsing YAML specification."""
        from ai_engine.marketplace.knowledge_base_integration import MarketplaceKnowledgeBaseIntegration

        integration = MarketplaceKnowledgeBaseIntegration()

        spec_content = """
protocol_metadata:
  name: test
  version: 1.0.0
  category: finance

protocol_spec:
  fields:
    - id: 1
      name: field1
      type: string
"""
        result = integration._parse_specification(spec_content, "yaml")

        assert "protocol_metadata" in result
        assert result["protocol_metadata"]["name"] == "test"
        assert result["protocol_metadata"]["version"] == "1.0.0"
        assert "protocol_spec" in result

    @pytest.mark.asyncio
    async def test_parse_specification_json(self):
        """Test parsing JSON specification."""
        from ai_engine.marketplace.knowledge_base_integration import MarketplaceKnowledgeBaseIntegration

        integration = MarketplaceKnowledgeBaseIntegration()

        spec_content = '{"protocol_metadata": {"name": "test", "version": "1.0.0"}, "protocol_spec": {}}'
        result = integration._parse_specification(spec_content, "json")

        assert "protocol_metadata" in result
        assert result["protocol_metadata"]["name"] == "test"

    @pytest.mark.asyncio
    async def test_parse_specification_invalid_format(self):
        """Test parsing with unsupported format."""
        from ai_engine.marketplace.knowledge_base_integration import MarketplaceKnowledgeBaseIntegration

        integration = MarketplaceKnowledgeBaseIntegration()

        with pytest.raises(ValueError, match="Unsupported spec format"):
            integration._parse_specification("content", "xml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
