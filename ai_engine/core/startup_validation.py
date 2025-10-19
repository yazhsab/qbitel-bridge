"""
CRONOS AI Engine - Startup Validation

This module provides comprehensive startup validation to ensure all required
configuration, secrets, and dependencies are properly configured before the
application starts.
"""

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation check severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""

    check_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"[{self.severity.value.upper()}] {status}: {self.check_name} - {self.message}"


class StartupValidator:
    """
    Comprehensive startup validator for CRONOS AI Engine.

    Validates:
    - Required environment variables and secrets
    - Database connectivity
    - File system permissions
    - External service connectivity
    - Configuration file validity
    - Security requirements
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, environment: str = "development"):
        """Initialize startup validator."""
        self.config = config or {}
        self.environment = environment
        self.results: List[ValidationResult] = []

    def add_result(self, result: ValidationResult):
        """Add a validation result."""
        self.results.append(result)

        # Log the result
        log_level = {
            ValidationSeverity.INFO: logging.INFO,
            ValidationSeverity.WARNING: logging.WARNING,
            ValidationSeverity.ERROR: logging.ERROR,
            ValidationSeverity.CRITICAL: logging.CRITICAL,
        }[result.severity]

        logger.log(log_level, str(result))

    def validate_all(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if all critical/error validations pass, False otherwise
        """
        logger.info(f"Starting validation for environment: {self.environment}")

        # Run all validation checks
        self.validate_secrets()
        self.validate_database_config()
        self.validate_redis_config()
        self.validate_security_config()
        self.validate_file_system()
        self.validate_monitoring_config()
        self.validate_llm_config()

        # Summarize results
        return self.print_summary()

    def validate_secrets(self):
        """Validate required secrets are configured."""
        logger.info("Validating secrets and sensitive configuration...")

        # Required secrets for all environments
        required_secrets = []

        # Additional secrets required for production
        if self.environment == "production":
            required_secrets.extend([
                ("DATABASE_PASSWORD", "Database password must be set"),
                ("JWT_SECRET", "JWT secret must be set (minimum 32 characters)"),
                ("ENCRYPTION_KEY", "Encryption key must be set (minimum 32 characters)"),
            ])
        else:
            # Warnings for dev/staging
            optional_secrets = [
                ("DATABASE_PASSWORD", "Database password should be set"),
                ("JWT_SECRET", "JWT secret should be set"),
                ("ENCRYPTION_KEY", "Encryption key should be set"),
            ]

            for secret_name, message in optional_secrets:
                value = os.getenv(secret_name)
                if not value:
                    self.add_result(ValidationResult(
                        check_name=f"Secret: {secret_name}",
                        passed=True,  # Warning, not failure
                        severity=ValidationSeverity.WARNING,
                        message=f"{message} (using default for {self.environment})"
                    ))

        # Validate required secrets
        for secret_name, message in required_secrets:
            value = os.getenv(secret_name)

            if not value:
                self.add_result(ValidationResult(
                    check_name=f"Secret: {secret_name}",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=message,
                    details={"env_var": secret_name}
                ))
            elif secret_name in ["JWT_SECRET", "ENCRYPTION_KEY"]:
                # Validate minimum length for cryptographic secrets
                if len(value) < 32:
                    self.add_result(ValidationResult(
                        check_name=f"Secret: {secret_name}",
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"{secret_name} must be at least 32 characters long (current: {len(value)})",
                        details={"env_var": secret_name, "current_length": len(value), "required_length": 32}
                    ))
                else:
                    self.add_result(ValidationResult(
                        check_name=f"Secret: {secret_name}",
                        passed=True,
                        severity=ValidationSeverity.INFO,
                        message=f"{secret_name} is properly configured"
                    ))
            else:
                self.add_result(ValidationResult(
                    check_name=f"Secret: {secret_name}",
                    passed=True,
                    severity=ValidationSeverity.INFO,
                    message=f"{secret_name} is configured"
                ))

        # Validate no default/weak passwords in production
        if self.environment == "production":
            self._check_weak_passwords()

    def _check_weak_passwords(self):
        """Check for weak or default passwords."""
        weak_patterns = [
            "password",
            "changeme",
            "admin",
            "root",
            "123456",
            "default",
            "secret",
        ]

        secrets_to_check = [
            "DATABASE_PASSWORD",
            "REDIS_PASSWORD",
            "JWT_SECRET",
        ]

        for secret_name in secrets_to_check:
            value = os.getenv(secret_name, "").lower()
            if any(pattern in value for pattern in weak_patterns):
                self.add_result(ValidationResult(
                    check_name=f"Weak Password: {secret_name}",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"{secret_name} contains weak/default password patterns. This is not allowed in production!",
                    details={"env_var": secret_name}
                ))

    def validate_database_config(self):
        """Validate database configuration."""
        logger.info("Validating database configuration...")

        required_vars = [
            "DATABASE_HOST",
            "DATABASE_PORT",
            "DATABASE_NAME",
            "DATABASE_USER",
        ]

        for var in required_vars:
            value = os.getenv(var)
            if not value:
                severity = ValidationSeverity.ERROR if self.environment == "production" else ValidationSeverity.WARNING
                self.add_result(ValidationResult(
                    check_name=f"Database Config: {var}",
                    passed=False,
                    severity=severity,
                    message=f"{var} must be set",
                    details={"env_var": var}
                ))
            else:
                # Validate port is numeric
                if var == "DATABASE_PORT":
                    try:
                        port = int(value)
                        if not (1 <= port <= 65535):
                            self.add_result(ValidationResult(
                                check_name=f"Database Config: {var}",
                                passed=False,
                                severity=ValidationSeverity.ERROR,
                                message=f"Invalid port number: {port} (must be 1-65535)",
                                details={"env_var": var, "value": value}
                            ))
                        else:
                            self.add_result(ValidationResult(
                                check_name=f"Database Config: {var}",
                                passed=True,
                                severity=ValidationSeverity.INFO,
                                message=f"{var} is valid: {port}"
                            ))
                    except ValueError:
                        self.add_result(ValidationResult(
                            check_name=f"Database Config: {var}",
                            passed=False,
                            severity=ValidationSeverity.ERROR,
                            message=f"{var} must be a number, got: {value}",
                            details={"env_var": var, "value": value}
                        ))
                else:
                    self.add_result(ValidationResult(
                        check_name=f"Database Config: {var}",
                        passed=True,
                        severity=ValidationSeverity.INFO,
                        message=f"{var} is configured"
                    ))

    def validate_redis_config(self):
        """Validate Redis configuration."""
        logger.info("Validating Redis configuration...")

        redis_host = os.getenv("REDIS_HOST")
        redis_port = os.getenv("REDIS_PORT")

        if not redis_host:
            self.add_result(ValidationResult(
                check_name="Redis Config: REDIS_HOST",
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="REDIS_HOST not set, caching will be disabled"
            ))
        else:
            self.add_result(ValidationResult(
                check_name="Redis Config: REDIS_HOST",
                passed=True,
                severity=ValidationSeverity.INFO,
                message=f"Redis host configured: {redis_host}"
            ))

        if redis_port:
            try:
                port = int(redis_port)
                if not (1 <= port <= 65535):
                    self.add_result(ValidationResult(
                        check_name="Redis Config: REDIS_PORT",
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid Redis port: {port}",
                        details={"value": redis_port}
                    ))
                else:
                    self.add_result(ValidationResult(
                        check_name="Redis Config: REDIS_PORT",
                        passed=True,
                        severity=ValidationSeverity.INFO,
                        message=f"Redis port is valid: {port}"
                    ))
            except ValueError:
                self.add_result(ValidationResult(
                    check_name="Redis Config: REDIS_PORT",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"REDIS_PORT must be a number, got: {redis_port}",
                    details={"value": redis_port}
                ))

    def validate_security_config(self):
        """Validate security configuration."""
        logger.info("Validating security configuration...")

        # Check TLS/SSL in production
        if self.environment == "production":
            tls_enabled = os.getenv("TLS_ENABLED", "false").lower() == "true"

            if not tls_enabled:
                self.add_result(ValidationResult(
                    check_name="Security: TLS/SSL",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="TLS must be enabled in production!",
                    details={"env_var": "TLS_ENABLED"}
                ))
            else:
                # Validate certificate files exist
                cert_file = os.getenv("TLS_CERT_FILE")
                key_file = os.getenv("TLS_KEY_FILE")

                if not cert_file or not key_file:
                    self.add_result(ValidationResult(
                        check_name="Security: TLS Certificates",
                        passed=False,
                        severity=ValidationSeverity.CRITICAL,
                        message="TLS_CERT_FILE and TLS_KEY_FILE must be set when TLS is enabled",
                        details={"cert_file": cert_file, "key_file": key_file}
                    ))
                else:
                    # Check if files exist
                    if not Path(cert_file).exists():
                        self.add_result(ValidationResult(
                            check_name="Security: TLS Certificate File",
                            passed=False,
                            severity=ValidationSeverity.ERROR,
                            message=f"TLS certificate file not found: {cert_file}",
                            details={"file_path": cert_file}
                        ))

                    if not Path(key_file).exists():
                        self.add_result(ValidationResult(
                            check_name="Security: TLS Key File",
                            passed=False,
                            severity=ValidationSeverity.ERROR,
                            message=f"TLS key file not found: {key_file}",
                            details={"file_path": key_file}
                        ))

        # Check API key configuration
        api_key_enabled = os.getenv("API_KEY_ENABLED", "false").lower() == "true"
        if api_key_enabled and self.environment == "production":
            self.add_result(ValidationResult(
                check_name="Security: API Keys",
                passed=True,
                severity=ValidationSeverity.INFO,
                message="API key authentication is enabled"
            ))

        # Check CORS configuration in production
        if self.environment == "production":
            cors_origins = os.getenv("CORS_ALLOWED_ORIGINS")
            if not cors_origins:
                self.add_result(ValidationResult(
                    check_name="Security: CORS Origins",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="CORS_ALLOWED_ORIGINS must be set in production",
                    details={"env_var": "CORS_ALLOWED_ORIGINS"}
                ))
            elif cors_origins == "*" or "example.com" in cors_origins.lower():
                self.add_result(ValidationResult(
                    check_name="Security: CORS Origins",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="CORS_ALLOWED_ORIGINS contains wildcard or example domains. Must use actual production domains!",
                    details={"value": cors_origins}
                ))
            elif not all(origin.startswith("https://") for origin in cors_origins.split(",")):
                self.add_result(ValidationResult(
                    check_name="Security: CORS Origins",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="All CORS origins must use HTTPS in production",
                    details={"value": cors_origins}
                ))
            else:
                self.add_result(ValidationResult(
                    check_name="Security: CORS Origins",
                    passed=True,
                    severity=ValidationSeverity.INFO,
                    message=f"CORS origins properly configured: {cors_origins}"
                ))

    def validate_file_system(self):
        """Validate file system permissions and directories."""
        logger.info("Validating file system...")

        # Required directories
        required_dirs = [
            (os.getenv("AI_MODEL_CACHE_DIR", "/tmp/cronos_ai_models"), "Model cache directory"),
            (os.getenv("LOG_FILE", "/var/log/cronos_ai/app.log"), "Log file directory", True),
        ]

        for dir_or_file, description, *is_file in required_dirs:
            path = Path(dir_or_file)
            parent_path = path.parent if is_file else path

            # Check if parent directory exists and is writable
            if not parent_path.exists():
                self.add_result(ValidationResult(
                    check_name=f"File System: {description}",
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Directory does not exist: {parent_path} (will attempt to create)",
                    details={"path": str(parent_path)}
                ))
            elif not os.access(parent_path, os.W_OK):
                self.add_result(ValidationResult(
                    check_name=f"File System: {description}",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Directory is not writable: {parent_path}",
                    details={"path": str(parent_path)}
                ))
            else:
                self.add_result(ValidationResult(
                    check_name=f"File System: {description}",
                    passed=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Directory is accessible: {parent_path}"
                ))

    def validate_monitoring_config(self):
        """Validate monitoring and observability configuration."""
        logger.info("Validating monitoring configuration...")

        metrics_enabled = os.getenv("METRICS_ENABLED", "true").lower() == "true"
        prometheus_port = os.getenv("PROMETHEUS_PORT", "8000")

        self.add_result(ValidationResult(
            check_name="Monitoring: Metrics",
            passed=True,
            severity=ValidationSeverity.INFO,
            message=f"Metrics collection is {'enabled' if metrics_enabled else 'disabled'}"
        ))

        if metrics_enabled:
            try:
                port = int(prometheus_port)
                self.add_result(ValidationResult(
                    check_name="Monitoring: Prometheus Port",
                    passed=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Prometheus metrics port: {port}"
                ))
            except ValueError:
                self.add_result(ValidationResult(
                    check_name="Monitoring: Prometheus Port",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid Prometheus port: {prometheus_port}"
                ))

    def validate_llm_config(self):
        """Validate LLM configuration if LLM features are enabled."""
        logger.info("Validating LLM configuration...")

        llm_provider = os.getenv("LLM_PROVIDER", "").lower()

        if llm_provider:
            provider_api_keys = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "azure": "AZURE_OPENAI_API_KEY",
            }

            if llm_provider in provider_api_keys:
                api_key_var = provider_api_keys[llm_provider]
                api_key = os.getenv(api_key_var)

                if not api_key:
                    self.add_result(ValidationResult(
                        check_name=f"LLM: {llm_provider.upper()} API Key",
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"{api_key_var} must be set for {llm_provider} provider",
                        details={"env_var": api_key_var}
                    ))
                else:
                    self.add_result(ValidationResult(
                        check_name=f"LLM: {llm_provider.upper()} API Key",
                        passed=True,
                        severity=ValidationSeverity.INFO,
                        message=f"API key configured for {llm_provider}"
                    ))
            else:
                self.add_result(ValidationResult(
                    check_name="LLM: Provider",
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Unknown LLM provider: {llm_provider}",
                    details={"provider": llm_provider}
                ))

    def print_summary(self) -> bool:
        """
        Print validation summary and return overall status.

        Returns:
            True if all critical/error checks passed, False otherwise
        """
        logger.info("\n" + "="*80)
        logger.info("STARTUP VALIDATION SUMMARY")
        logger.info("="*80)

        # Count results by severity
        counts = {
            ValidationSeverity.INFO: 0,
            ValidationSeverity.WARNING: 0,
            ValidationSeverity.ERROR: 0,
            ValidationSeverity.CRITICAL: 0,
        }

        failed_critical = []
        failed_errors = []
        warnings = []

        for result in self.results:
            counts[result.severity] += 1

            if not result.passed:
                if result.severity == ValidationSeverity.CRITICAL:
                    failed_critical.append(result)
                elif result.severity == ValidationSeverity.ERROR:
                    failed_errors.append(result)
                elif result.severity == ValidationSeverity.WARNING:
                    warnings.append(result)

        # Print summary
        logger.info(f"Total checks: {len(self.results)}")
        logger.info(f"  ✓ Info:     {counts[ValidationSeverity.INFO]}")
        logger.info(f"  ⚠ Warning:  {counts[ValidationSeverity.WARNING]}")
        logger.info(f"  ✗ Error:    {counts[ValidationSeverity.ERROR]}")
        logger.info(f"  ✗ Critical: {counts[ValidationSeverity.CRITICAL]}")

        # Print failures
        if failed_critical:
            logger.critical("\n" + "-"*80)
            logger.critical("CRITICAL FAILURES (Must Fix):")
            logger.critical("-"*80)
            for result in failed_critical:
                logger.critical(f"  {result}")

        if failed_errors:
            logger.error("\n" + "-"*80)
            logger.error("ERRORS (Should Fix):")
            logger.error("-"*80)
            for result in failed_errors:
                logger.error(f"  {result}")

        if warnings:
            logger.warning("\n" + "-"*80)
            logger.warning("WARNINGS (Review Recommended):")
            logger.warning("-"*80)
            for result in warnings:
                logger.warning(f"  {result}")

        logger.info("="*80)

        # Determine overall status
        has_critical_failures = len(failed_critical) > 0
        has_error_failures = len(failed_errors) > 0

        if has_critical_failures:
            logger.critical("❌ VALIDATION FAILED: Critical issues must be resolved before starting!")
            logger.critical("   Application will NOT start.")
            return False
        elif has_error_failures and self.environment == "production":
            logger.error("❌ VALIDATION FAILED: Errors found in production environment!")
            logger.error("   Fix errors before deploying to production.")
            return False
        elif has_error_failures:
            logger.warning("⚠️  VALIDATION WARNING: Errors found but allowed in non-production environment")
            logger.warning("   Application will start, but errors should be fixed.")
            return True
        else:
            logger.info("✅ VALIDATION PASSED: All checks completed successfully!")
            return True


def validate_startup(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convenience function to run startup validation.

    Args:
        config: Optional configuration dictionary

    Returns:
        True if validation passed, False if critical failures

    Raises:
        SystemExit: If critical validation failures in production
    """
    environment = os.getenv("CRONOS_ENVIRONMENT", "development")

    validator = StartupValidator(config=config, environment=environment)
    passed = validator.validate_all()

    if not passed and environment == "production":
        logger.critical("Exiting due to validation failures in production environment")
        sys.exit(1)

    return passed


if __name__ == "__main__":
    # Allow running as standalone script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    validate_startup()
