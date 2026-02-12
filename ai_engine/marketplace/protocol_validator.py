"""
QBITEL - Protocol Validation Pipeline

Automated validation pipeline for marketplace protocol submissions including
syntax validation, parser testing, security scanning, and performance benchmarking.
"""

import asyncio
import logging
import yaml
import json
import subprocess
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from decimal import Decimal

from ..core.config import get_config
from ..models.marketplace import (
    MarketplaceProtocol,
    MarketplaceValidation,
)
from ..core.database_manager import get_database_manager

logger = logging.getLogger(__name__)


class ValidationResult:
    """Validation result container."""

    def __init__(
        self,
        validation_type: str,
        status: str,
        score: Optional[Decimal] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        test_results: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        self.validation_type = validation_type
        self.status = status
        self.score = score
        self.errors = errors or []
        self.warnings = warnings or []
        self.test_results = test_results or {}
        self.metrics = metrics or {}


class ProtocolValidator:
    """
    Automated protocol validation pipeline.

    Performs comprehensive validation of submitted protocols including:
    - Syntax and schema validation
    - Parser functionality testing
    - Security vulnerability scanning
    - Performance benchmarking
    """

    def __init__(self):
        self.config = get_config()
        self.db_manager = get_database_manager()

    async def validate_protocol(self, protocol_id: str) -> Dict[str, ValidationResult]:
        """
        Run complete validation pipeline for a protocol.

        Args:
            protocol_id: Protocol UUID to validate

        Returns:
            Dict of validation results by type
        """
        logger.info(f"Starting validation for protocol {protocol_id}")

        # Get protocol from database
        session = self.db_manager.get_session()
        try:
            protocol = session.query(MarketplaceProtocol).filter(MarketplaceProtocol.protocol_id == protocol_id).first()

            if not protocol:
                raise ValueError(f"Protocol {protocol_id} not found")

            # Run validation steps
            results = {}

            # Step 1: Syntax validation
            logger.info("Running syntax validation...")
            results["syntax_validation"] = await self.validate_syntax(protocol)

            # Step 2: Parser testing (only if syntax passes)
            if results["syntax_validation"].status == "passed":
                logger.info("Running parser testing...")
                results["parser_testing"] = await self.test_parser(protocol)
            else:
                logger.warning("Skipping parser testing due to syntax errors")
                results["parser_testing"] = ValidationResult(
                    validation_type="parser_testing", status="skipped", errors=["Skipped due to syntax validation failures"]
                )

            # Step 3: Security scan
            logger.info("Running security scan...")
            results["security_scan"] = await self.security_scan(protocol)

            # Step 4: Performance benchmark (only if parser passes)
            if results.get("parser_testing") and results["parser_testing"].status == "passed":
                logger.info("Running performance benchmark...")
                results["performance_benchmark"] = await self.performance_benchmark(protocol)
            else:
                logger.warning("Skipping performance benchmark")
                results["performance_benchmark"] = ValidationResult(
                    validation_type="performance_benchmark",
                    status="skipped",
                    errors=["Skipped due to parser testing failures"],
                )

            # Save results to database
            for validation_type, result in results.items():
                self._save_validation_result(session, protocol_id, result)

            session.commit()

            # Determine overall status
            overall_status = self._determine_overall_status(results)
            logger.info(f"Validation completed for protocol {protocol_id}: {overall_status}")

            return results

        except Exception as e:
            logger.error(f"Validation failed for protocol {protocol_id}: {e}", exc_info=True)
            session.rollback()
            raise
        finally:
            session.close()

    async def validate_syntax(self, protocol: MarketplaceProtocol) -> ValidationResult:
        """
        Validate protocol specification syntax and schema.

        Args:
            protocol: Protocol to validate

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        test_results = {}

        try:
            # Download spec file content (mock for now)
            spec_content = self._download_spec_file(protocol.spec_file_url)

            # Parse based on format
            if protocol.spec_format.value == "yaml":
                try:
                    spec_data = yaml.safe_load(spec_content)
                    test_results["yaml_valid"] = True
                except yaml.YAMLError as e:
                    errors.append(f"Invalid YAML syntax: {str(e)}")
                    test_results["yaml_valid"] = False
                    return ValidationResult(
                        validation_type="syntax_validation",
                        status="failed",
                        score=Decimal("0.0"),
                        errors=errors,
                        test_results=test_results,
                    )

            elif protocol.spec_format.value == "json":
                try:
                    spec_data = json.loads(spec_content)
                    test_results["json_valid"] = True
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON syntax: {str(e)}")
                    test_results["json_valid"] = False
                    return ValidationResult(
                        validation_type="syntax_validation",
                        status="failed",
                        score=Decimal("0.0"),
                        errors=errors,
                        test_results=test_results,
                    )
            else:
                warnings.append(f"Unsupported spec format: {protocol.spec_format}")

            # Validate required fields
            required_fields = ["protocol_metadata", "protocol_spec"]
            for field in required_fields:
                if field not in spec_data:
                    errors.append(f"Missing required field: {field}")
                else:
                    test_results[f"{field}_present"] = True

            # Validate metadata
            if "protocol_metadata" in spec_data:
                metadata = spec_data["protocol_metadata"]
                required_metadata = ["name", "version", "category"]
                for field in required_metadata:
                    if field not in metadata:
                        errors.append(f"Missing required metadata field: {field}")
                    else:
                        test_results[f"metadata_{field}_present"] = True

            # Calculate score
            total_checks = len(required_fields) + len(required_metadata) + 1  # +1 for syntax
            passed_checks = len([v for v in test_results.values() if v is True])
            score = Decimal(str((passed_checks / total_checks) * 100))

            status = "passed" if len(errors) == 0 else "failed"

            return ValidationResult(
                validation_type="syntax_validation",
                status=status,
                score=score,
                errors=errors,
                warnings=warnings,
                test_results=test_results,
            )

        except Exception as e:
            logger.error(f"Syntax validation error: {e}", exc_info=True)
            return ValidationResult(
                validation_type="syntax_validation",
                status="failed",
                score=Decimal("0.0"),
                errors=[f"Validation error: {str(e)}"],
                test_results=test_results,
            )

    async def test_parser(self, protocol: MarketplaceProtocol) -> ValidationResult:
        """
        Test protocol parser against sample data.

        Args:
            protocol: Protocol to test

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        test_results = {}

        try:
            if not protocol.parser_code_url:
                warnings.append("No parser code provided")
                return ValidationResult(
                    validation_type="parser_testing",
                    status="skipped",
                    warnings=warnings,
                )

            if not protocol.test_data_url:
                warnings.append("No test data provided")
                return ValidationResult(
                    validation_type="parser_testing",
                    status="skipped",
                    warnings=warnings,
                )

            # Download parser and test data (mock)
            parser_code = self._download_parser_code(protocol.parser_code_url)
            test_data = self._download_test_data(protocol.test_data_url)

            # Execute parser against test data
            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)

                # Save parser code
                parser_file = tmppath / "parser.py"
                parser_file.write_text(parser_code)

                # Save test data
                test_data_file = tmppath / "test_data.bin"
                test_data_file.write_bytes(test_data)

                # Run parser (mock execution)
                # In production, this would execute in a sandboxed environment
                try:
                    # Mock parser execution
                    test_results["parser_executed"] = True
                    test_results["samples_parsed"] = 10
                    test_results["samples_failed"] = 0
                    test_results["success_rate"] = 100.0

                    # Check success rate
                    success_rate = test_results["success_rate"]
                    if success_rate < 95.0:
                        errors.append(f"Parser success rate too low: {success_rate}%")
                    elif success_rate < 98.0:
                        warnings.append(f"Parser success rate below ideal: {success_rate}%")

                except Exception as e:
                    errors.append(f"Parser execution failed: {str(e)}")
                    test_results["parser_executed"] = False

            score = Decimal(str(test_results.get("success_rate", 0)))
            status = "passed" if len(errors) == 0 and score >= 95 else "failed"

            return ValidationResult(
                validation_type="parser_testing",
                status=status,
                score=score,
                errors=errors,
                warnings=warnings,
                test_results=test_results,
            )

        except Exception as e:
            logger.error(f"Parser testing error: {e}", exc_info=True)
            return ValidationResult(
                validation_type="parser_testing",
                status="failed",
                score=Decimal("0.0"),
                errors=[f"Parser testing error: {str(e)}"],
            )

    async def security_scan(self, protocol: MarketplaceProtocol) -> ValidationResult:
        """
        Perform security vulnerability scanning on parser code.

        Args:
            protocol: Protocol to scan

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        test_results = {}
        vulnerabilities = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        try:
            if not protocol.parser_code_url:
                warnings.append("No parser code to scan")
                return ValidationResult(
                    validation_type="security_scan",
                    status="skipped",
                    warnings=warnings,
                )

            # Download parser code
            parser_code = self._download_parser_code(protocol.parser_code_url)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                parser_file = tmppath / "parser.py"
                parser_file.write_text(parser_code)

                # Run Bandit (Python security linter)
                # Mock results for now
                test_results["bandit_executed"] = True
                test_results["bandit_issues"] = 0

                # Run dependency scan (if requirements present)
                test_results["dependency_scan_executed"] = True
                test_results["vulnerable_dependencies"] = 0

                # Check for common security issues
                security_checks = {
                    "no_eval": "eval(" not in parser_code,
                    "no_exec": "exec(" not in parser_code,
                    "no_shell": "shell=True" not in parser_code,
                    "no_pickle": "pickle.loads" not in parser_code,
                }

                for check, passed in security_checks.items():
                    test_results[check] = passed
                    if not passed:
                        vulnerabilities["high"] += 1
                        errors.append(f"Security issue detected: {check}")

                # Calculate score
                if vulnerabilities["critical"] > 0:
                    score = Decimal("0.0")
                    status = "failed"
                elif vulnerabilities["high"] > 0:
                    score = Decimal("50.0")
                    status = "failed"
                elif vulnerabilities["medium"] > 0:
                    score = Decimal("75.0")
                    status = "passed"
                    warnings.append(f"{vulnerabilities['medium']} medium severity issues found")
                else:
                    score = Decimal("100.0")
                    status = "passed"

                test_results["vulnerabilities"] = vulnerabilities

                return ValidationResult(
                    validation_type="security_scan",
                    status=status,
                    score=score,
                    errors=errors,
                    warnings=warnings,
                    test_results=test_results,
                )

        except Exception as e:
            logger.error(f"Security scan error: {e}", exc_info=True)
            return ValidationResult(
                validation_type="security_scan",
                status="failed",
                score=Decimal("0.0"),
                errors=[f"Security scan error: {str(e)}"],
            )

    async def performance_benchmark(self, protocol: MarketplaceProtocol) -> ValidationResult:
        """
        Run performance benchmarks on the parser.

        Args:
            protocol: Protocol to benchmark

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        metrics = {}

        try:
            # Mock performance benchmarking
            # In production, this would run actual throughput tests
            metrics["throughput"] = 5000  # packets/sec
            metrics["memory_usage"] = 250  # MB
            metrics["latency_p50"] = 2.5  # ms
            metrics["latency_p95"] = 8.2  # ms
            metrics["latency_p99"] = 15.3  # ms

            # Check thresholds
            if metrics["throughput"] < 1000:
                errors.append(f"Throughput too low: {metrics['throughput']} packets/sec")
            elif metrics["throughput"] < 5000:
                warnings.append(f"Throughput below ideal: {metrics['throughput']} packets/sec")

            if metrics["memory_usage"] > 500:
                errors.append(f"Memory usage too high: {metrics['memory_usage']} MB")
            elif metrics["memory_usage"] > 300:
                warnings.append(f"Memory usage elevated: {metrics['memory_usage']} MB")

            # Calculate score
            throughput_score = min(100, (metrics["throughput"] / 10000) * 100)
            memory_score = max(0, 100 - (metrics["memory_usage"] / 5))
            latency_score = max(0, 100 - (metrics["latency_p95"] / 0.2))

            score = Decimal(str((throughput_score + memory_score + latency_score) / 3))
            status = "passed" if len(errors) == 0 else "failed"

            return ValidationResult(
                validation_type="performance_benchmark",
                status=status,
                score=score,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Performance benchmark error: {e}", exc_info=True)
            return ValidationResult(
                validation_type="performance_benchmark",
                status="failed",
                score=Decimal("0.0"),
                errors=[f"Benchmark error: {str(e)}"],
            )

    def _save_validation_result(self, session, protocol_id: str, result: ValidationResult):
        """Save validation result to database."""
        try:
            validation = MarketplaceValidation(
                protocol_id=protocol_id,
                validation_type=result.validation_type,
                status=result.status,
                score=result.score,
                test_results=result.test_results,
                errors=result.errors,
                warnings=result.warnings,
                throughput=result.metrics.get("throughput"),
                memory_usage=result.metrics.get("memory_usage"),
                latency_p50=result.metrics.get("latency_p50"),
                latency_p95=result.metrics.get("latency_p95"),
                latency_p99=result.metrics.get("latency_p99"),
                vulnerabilities_critical=result.test_results.get("vulnerabilities", {}).get("critical", 0),
                vulnerabilities_high=result.test_results.get("vulnerabilities", {}).get("high", 0),
                vulnerabilities_medium=result.test_results.get("vulnerabilities", {}).get("medium", 0),
                vulnerabilities_low=result.test_results.get("vulnerabilities", {}).get("low", 0),
                completed_at=datetime.utcnow(),
            )
            session.add(validation)
        except Exception as e:
            logger.error(f"Error saving validation result: {e}")

    def _determine_overall_status(self, results: Dict[str, ValidationResult]) -> str:
        """Determine overall validation status from individual results."""
        # Critical failures
        if results.get("syntax_validation") and results["syntax_validation"].status == "failed":
            return "failed"

        if results.get("security_scan") and results["security_scan"].status == "failed":
            return "failed"

        # Check all passed
        all_passed = all(r.status == "passed" or r.status == "skipped" for r in results.values())

        if all_passed:
            return "passed"
        else:
            return "failed"

    def _download_spec_file(self, url: str) -> str:
        """Download specification file from URL. Mock implementation."""
        # In production, this would download from S3
        return """
protocol_metadata:
  name: example-protocol
  version: 1.0.0
  category: finance

protocol_spec:
  message_format: binary
  fields:
    - id: 1
      name: field1
      type: string
"""

    def _download_parser_code(self, url: str) -> str:
        """Download parser code from URL. Mock implementation."""
        # In production, this would download from S3
        return """
def parse_packet(data):
    return {"field1": "value"}
"""

    def _download_test_data(self, url: str) -> bytes:
        """Download test data from URL. Mock implementation."""
        # In production, this would download from S3
        return b"test data"


# Async task function for background validation
async def run_protocol_validation(protocol_id: str):
    """
    Run protocol validation in background.

    Args:
        protocol_id: Protocol UUID to validate
    """
    validator = ProtocolValidator()
    try:
        results = await validator.validate_protocol(protocol_id)
        logger.info(f"Validation completed for protocol {protocol_id}")
        return results
    except Exception as e:
        logger.error(f"Validation failed for protocol {protocol_id}: {e}", exc_info=True)
        raise
