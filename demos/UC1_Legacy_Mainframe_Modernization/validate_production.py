#!/usr/bin/env python3
"""
UC1 Legacy Mainframe Modernization - Production Readiness Validation

This script validates that all production-ready components are properly configured
and integrated.
"""

import os
import sys
import asyncio
import importlib
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEMO_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DEMO_ROOT / "backend"))


@dataclass
class ValidationResult:
    """Result of a validation check."""
    category: str
    check: str
    passed: bool
    message: str
    severity: str = "info"  # info, warning, error


class ProductionValidator:
    """Validates production readiness of UC1 demo."""

    def __init__(self):
        self.results: List[ValidationResult] = []

    def add_result(self, category: str, check: str, passed: bool, message: str, severity: str = "info"):
        """Add validation result."""
        self.results.append(ValidationResult(
            category=category,
            check=check,
            passed=passed,
            message=message,
            severity=severity,
        ))

    def validate_file_exists(self, filepath: Path, description: str) -> bool:
        """Validate that a file exists."""
        exists = filepath.exists()
        self.add_result(
            category="Files",
            check=description,
            passed=exists,
            message=f"{'Found' if exists else 'Missing'}: {filepath.name}",
            severity="error" if not exists else "info",
        )
        return exists

    def validate_module_import(self, module_name: str, description: str) -> bool:
        """Validate that a module can be imported."""
        try:
            importlib.import_module(module_name)
            self.add_result(
                category="Imports",
                check=description,
                passed=True,
                message=f"Successfully imported {module_name}",
            )
            return True
        except ImportError as e:
            self.add_result(
                category="Imports",
                check=description,
                passed=False,
                message=f"Failed to import {module_name}: {e}",
                severity="error",
            )
            return False

    def validate_production_integration(self) -> bool:
        """Validate production module integration."""
        try:
            # Check if production imports work
            from production_app import PRODUCTION_IMPORTS_AVAILABLE

            self.add_result(
                category="Integration",
                check="Production imports",
                passed=PRODUCTION_IMPORTS_AVAILABLE,
                message="Production QBITEL modules available" if PRODUCTION_IMPORTS_AVAILABLE else "Running in standalone mode",
                severity="warning" if not PRODUCTION_IMPORTS_AVAILABLE else "info",
            )
            return PRODUCTION_IMPORTS_AVAILABLE
        except Exception as e:
            self.add_result(
                category="Integration",
                check="Production imports",
                passed=False,
                message=f"Error checking production imports: {e}",
                severity="error",
            )
            return False

    def validate_api_endpoints(self) -> bool:
        """Validate API endpoint definitions."""
        try:
            from production_app import app

            routes = [route.path for route in app.routes]
            required_routes = [
                "/health",
                "/health/ready",
                "/health/live",
                "/metrics",
                "/api/v1/systems",
                "/api/v1/analyze/cobol",
                "/api/v1/analyze/protocol",
                "/api/v1/modernize",
                "/api/v1/generate/code",
                "/api/v1/knowledge/capture",
                "/api/v1/status",
            ]

            all_present = True
            for route in required_routes:
                present = any(r.startswith(route.split("{")[0]) for r in routes)
                if not present:
                    self.add_result(
                        category="API",
                        check=f"Endpoint {route}",
                        passed=False,
                        message=f"Missing endpoint: {route}",
                        severity="error",
                    )
                    all_present = False

            if all_present:
                self.add_result(
                    category="API",
                    check="All required endpoints",
                    passed=True,
                    message=f"All {len(required_routes)} required endpoints present",
                )

            return all_present
        except Exception as e:
            self.add_result(
                category="API",
                check="Endpoint validation",
                passed=False,
                message=f"Error validating endpoints: {e}",
                severity="error",
            )
            return False

    def validate_authentication(self) -> bool:
        """Validate authentication module."""
        try:
            from auth import AuthManager, AuthConfig, Role

            manager = AuthManager()

            # Check default API key exists
            has_demo_key = any(
                user.username == "demo"
                for user in manager._api_keys.values()
            )

            self.add_result(
                category="Security",
                check="Authentication module",
                passed=True,
                message="Authentication manager initialized",
            )

            self.add_result(
                category="Security",
                check="Demo API key",
                passed=has_demo_key,
                message="Demo API key configured" if has_demo_key else "No demo API key",
                severity="warning" if not has_demo_key else "info",
            )

            return True
        except Exception as e:
            self.add_result(
                category="Security",
                check="Authentication module",
                passed=False,
                message=f"Authentication module error: {e}",
                severity="error",
            )
            return False

    def validate_database(self) -> bool:
        """Validate database module."""
        try:
            from database import DatabaseManager, DatabaseConfig, SQLALCHEMY_AVAILABLE

            self.add_result(
                category="Database",
                check="SQLAlchemy available",
                passed=SQLALCHEMY_AVAILABLE,
                message="SQLAlchemy ORM available" if SQLALCHEMY_AVAILABLE else "Using in-memory storage",
                severity="warning" if not SQLALCHEMY_AVAILABLE else "info",
            )

            if SQLALCHEMY_AVAILABLE:
                config = DatabaseConfig.from_env()
                self.add_result(
                    category="Database",
                    check="Configuration",
                    passed=True,
                    message=f"Database driver: {config.driver}",
                )

            return True
        except Exception as e:
            self.add_result(
                category="Database",
                check="Database module",
                passed=False,
                message=f"Database module error: {e}",
                severity="error",
            )
            return False

    def validate_configuration(self) -> bool:
        """Validate configuration files."""
        config_dir = DEMO_ROOT / "config"

        prod_config = config_dir / "production.yaml"
        dev_config = config_dir / "development.yaml"

        prod_exists = self.validate_file_exists(prod_config, "Production config")
        dev_exists = self.validate_file_exists(dev_config, "Development config")

        return prod_exists and dev_exists

    def validate_tests(self) -> bool:
        """Validate test suite."""
        tests_dir = DEMO_ROOT / "tests"

        test_files = [
            tests_dir / "__init__.py",
            tests_dir / "conftest.py",
            tests_dir / "test_api.py",
            tests_dir / "test_cobol_parser.py",
            tests_dir / "test_protocol_analyzer.py",
            tests_dir / "test_integration.py",
        ]

        all_exist = True
        for test_file in test_files:
            if not self.validate_file_exists(test_file, f"Test file {test_file.name}"):
                all_exist = False

        return all_exist

    def validate_cobol_samples(self) -> bool:
        """Validate COBOL sample files."""
        samples_dir = DEMO_ROOT / "cobol_samples"

        if not samples_dir.exists():
            self.add_result(
                category="Samples",
                check="COBOL samples directory",
                passed=False,
                message="cobol_samples directory not found",
                severity="warning",
            )
            return False

        cobol_files = list(samples_dir.glob("*.cbl")) + list(samples_dir.glob("*.CBL"))

        self.add_result(
            category="Samples",
            check="COBOL sample files",
            passed=len(cobol_files) > 0,
            message=f"Found {len(cobol_files)} COBOL sample files",
            severity="warning" if len(cobol_files) == 0 else "info",
        )

        return len(cobol_files) > 0

    def run_all_validations(self) -> Tuple[int, int, int]:
        """Run all validations and return (passed, warnings, errors)."""
        print("=" * 60)
        print("UC1 Legacy Mainframe Modernization")
        print("Production Readiness Validation")
        print("=" * 60)
        print()

        # Validate files
        print("Checking required files...")
        backend_dir = DEMO_ROOT / "backend"
        self.validate_file_exists(backend_dir / "production_app.py", "Production API")
        self.validate_file_exists(backend_dir / "auth.py", "Authentication module")
        self.validate_file_exists(backend_dir / "database.py", "Database module")
        self.validate_file_exists(DEMO_ROOT / "requirements.txt", "Requirements file")

        # Validate configurations
        print("Checking configuration files...")
        self.validate_configuration()

        # Validate tests
        print("Checking test suite...")
        self.validate_tests()

        # Validate samples
        print("Checking sample files...")
        self.validate_cobol_samples()

        # Validate modules
        print("Checking module imports...")
        self.validate_module_import("fastapi", "FastAPI framework")
        self.validate_module_import("pydantic", "Pydantic validation")
        self.validate_module_import("prometheus_client", "Prometheus metrics")

        # Validate production integration
        print("Checking production integration...")
        self.validate_production_integration()

        # Validate API
        print("Checking API endpoints...")
        self.validate_api_endpoints()

        # Validate security
        print("Checking security modules...")
        self.validate_authentication()

        # Validate database
        print("Checking database layer...")
        self.validate_database()

        # Calculate results
        passed = sum(1 for r in self.results if r.passed)
        warnings = sum(1 for r in self.results if not r.passed and r.severity == "warning")
        errors = sum(1 for r in self.results if not r.passed and r.severity == "error")

        return passed, warnings, errors

    def print_report(self):
        """Print validation report."""
        passed, warnings, errors = self.run_all_validations()
        total = len(self.results)

        print()
        print("=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print()

        # Group by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        for category, results in categories.items():
            print(f"\n{category}:")
            print("-" * 40)
            for result in results:
                status = "✓" if result.passed else ("⚠" if result.severity == "warning" else "✗")
                print(f"  {status} {result.check}: {result.message}")

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Total checks: {total}")
        print(f"  Passed:       {passed} ({passed/total*100:.1f}%)")
        print(f"  Warnings:     {warnings}")
        print(f"  Errors:       {errors}")
        print()

        if errors == 0:
            print("✓ PRODUCTION READY - All critical checks passed!")
            if warnings > 0:
                print(f"  (Note: {warnings} warnings to review)")
        else:
            print(f"✗ NOT PRODUCTION READY - {errors} critical error(s) found")

        print()
        return errors == 0


def main():
    """Run production validation."""
    validator = ProductionValidator()
    success = validator.print_report()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
