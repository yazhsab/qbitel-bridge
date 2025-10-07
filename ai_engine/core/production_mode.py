"""
CRONOS AI - Production Mode Detection Utility

Centralized production mode detection and validation utilities.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class EnvironmentMode(str, Enum):
    """Environment mode enumeration."""

    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TESTING = "testing"


class ProductionModeDetector:
    """
    Centralized production mode detection.

    This class provides consistent production mode detection across
    the entire application.
    """

    # Production mode aliases
    PRODUCTION_ALIASES = {"production", "prod"}

    @staticmethod
    def is_production_mode() -> bool:
        """
        Check if running in production mode.

        Returns:
            bool: True if in production mode, False otherwise
        """
        env = os.getenv(
            "CRONOS_AI_ENVIRONMENT", os.getenv("ENVIRONMENT", "development")
        ).lower()
        return env in ProductionModeDetector.PRODUCTION_ALIASES

    @staticmethod
    def get_environment_mode() -> EnvironmentMode:
        """
        Get the current environment mode.

        Returns:
            EnvironmentMode: The current environment mode
        """
        env = os.getenv(
            "CRONOS_AI_ENVIRONMENT", os.getenv("ENVIRONMENT", "development")
        ).lower()

        if env in ProductionModeDetector.PRODUCTION_ALIASES:
            return EnvironmentMode.PRODUCTION
        elif env == "staging":
            return EnvironmentMode.STAGING
        elif env == "testing":
            return EnvironmentMode.TESTING
        else:
            return EnvironmentMode.DEVELOPMENT

    @staticmethod
    def validate_production_environment() -> Tuple[bool, List[str]]:
        """
        Validate that all production requirements are met.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list of error messages)
        """
        if not ProductionModeDetector.is_production_mode():
            return True, []

        errors = []

        # Check required environment variables
        required_vars = {
            "CRONOS_AI_DB_PASSWORD": "Database password",
            "CRONOS_AI_REDIS_PASSWORD": "Redis password",
            "CRONOS_AI_JWT_SECRET": "JWT secret",
            "CRONOS_AI_ENCRYPTION_KEY": "Encryption key",
        }

        for var_name, description in required_vars.items():
            if not os.getenv(var_name):
                errors.append(
                    f"{description} is REQUIRED in production mode. "
                    f"Set {var_name} environment variable."
                )

        return len(errors) == 0, errors

    @staticmethod
    def get_production_checklist() -> Dict[str, bool]:
        """
        Get a checklist of production requirements.

        Returns:
            Dict[str, bool]: Dictionary of requirement name to status
        """
        checklist = {
            "production_mode_enabled": ProductionModeDetector.is_production_mode(),
            "database_password_set": bool(os.getenv("CRONOS_AI_DB_PASSWORD")),
            "redis_password_set": bool(os.getenv("CRONOS_AI_REDIS_PASSWORD")),
            "jwt_secret_set": bool(os.getenv("CRONOS_AI_JWT_SECRET")),
            "encryption_key_set": bool(os.getenv("CRONOS_AI_ENCRYPTION_KEY")),
            "api_key_set": bool(os.getenv("CRONOS_AI_API_KEY")),
        }

        return checklist

    @staticmethod
    def log_environment_info() -> None:
        """Log current environment information."""
        mode = ProductionModeDetector.get_environment_mode()
        is_prod = ProductionModeDetector.is_production_mode()

        logger.info(f"Environment Mode: {mode.value}")
        logger.info(f"Production Mode: {is_prod}")

        if is_prod:
            is_valid, errors = ProductionModeDetector.validate_production_environment()
            if is_valid:
                logger.info("✓ All production requirements met")
            else:
                logger.error("✗ Production requirements not met:")
                for error in errors:
                    logger.error(f"  - {error}")


class ProductionValidator:
    """
    Validates production-specific requirements.
    """

    MIN_PASSWORD_LENGTH = 16
    MIN_SECRET_LENGTH = 32
    MIN_API_KEY_LENGTH = 32

    WEAK_PATTERNS = [
        "password",
        "admin",
        "test",
        "demo",
        "123456",
        "qwerty",
        "letmein",
        "welcome",
        "monkey",
        "dragon",
        "secret",
        "key",
        "token",
        "changeme",
        "redis",
        "cache",
    ]

    @staticmethod
    def validate_password_strength(
        password: str, min_length: int = MIN_PASSWORD_LENGTH
    ) -> Tuple[bool, List[str]]:
        """
        Validate password strength.

        Args:
            password: Password to validate
            min_length: Minimum required length

        Returns:
            Tuple[bool, List[str]]: (is_valid, list of error messages)
        """
        errors = []

        # Check minimum length
        if len(password) < min_length:
            errors.append(
                f"Password too short: {len(password)} characters "
                f"(minimum: {min_length})"
            )

        # Check for weak patterns
        password_lower = password.lower()
        found_patterns = [
            p for p in ProductionValidator.WEAK_PATTERNS if p in password_lower
        ]
        if found_patterns:
            errors.append(
                f"Password contains weak patterns: {', '.join(found_patterns)}"
            )

        # Check for sequential characters
        if ProductionValidator._has_sequential_chars(password):
            errors.append("Password contains sequential characters")

        # Check for repeated characters
        if ProductionValidator._has_repeated_chars(password):
            errors.append("Password contains too many repeated characters")

        return len(errors) == 0, errors

    @staticmethod
    def validate_secret_strength(
        secret: str, min_length: int = MIN_SECRET_LENGTH
    ) -> Tuple[bool, List[str]]:
        """
        Validate secret strength.

        Args:
            secret: Secret to validate
            min_length: Minimum required length

        Returns:
            Tuple[bool, List[str]]: (is_valid, list of error messages)
        """
        errors = []

        # Check minimum length
        if len(secret) < min_length:
            errors.append(
                f"Secret too short: {len(secret)} characters "
                f"(minimum: {min_length})"
            )

        # Check for weak patterns
        secret_lower = secret.lower()
        found_patterns = [
            p for p in ProductionValidator.WEAK_PATTERNS if p in secret_lower
        ]
        if found_patterns:
            errors.append(f"Secret contains weak patterns: {', '.join(found_patterns)}")

        # Check entropy
        if len(set(secret)) < len(secret) * 0.5:
            errors.append("Secret has low entropy (too many repeated characters)")

        # Check for sequential characters
        if ProductionValidator._has_sequential_chars(secret):
            errors.append("Secret contains sequential characters")

        return len(errors) == 0, errors

    @staticmethod
    def validate_cors_origins(
        origins: List[str], allow_wildcard: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate CORS origins configuration.

        Args:
            origins: List of CORS origins
            allow_wildcard: Whether to allow wildcard (*)

        Returns:
            Tuple[bool, List[str]]: (is_valid, list of error messages)
        """
        errors = []

        if "*" in origins and not allow_wildcard:
            errors.append(
                "CORS wildcard (*) is not allowed in production mode. "
                "Specify explicit allowed origins."
            )

        # Validate HTTPS in production
        if ProductionModeDetector.is_production_mode():
            for origin in origins:
                if origin != "*" and not origin.startswith("https://"):
                    errors.append(
                        f"Origin '{origin}' must use HTTPS in production mode"
                    )

        return len(errors) == 0, errors

    @staticmethod
    def _has_sequential_chars(value: str, min_length: int = 3) -> bool:
        """Check for sequential characters."""
        for i in range(len(value) - min_length + 1):
            substr = value[i : i + min_length]
            # Check numeric sequences
            if substr.isdigit():
                nums = [int(c) for c in substr]
                if all(nums[j] + 1 == nums[j + 1] for j in range(len(nums) - 1)):
                    return True
            # Check alphabetic sequences
            if substr.isalpha():
                chars = [ord(c.lower()) for c in substr]
                if all(chars[j] + 1 == chars[j + 1] for j in range(len(chars) - 1)):
                    return True
        return False

    @staticmethod
    def _has_repeated_chars(value: str, max_repeats: int = 3) -> bool:
        """Check for repeated characters."""
        for i in range(len(value) - max_repeats + 1):
            if len(set(value[i : i + max_repeats])) == 1:
                return True
        return False


def require_production_mode(func):
    """
    Decorator to require production mode for a function.

    Usage:
        @require_production_mode
        def production_only_function():
            pass
    """

    def wrapper(*args, **kwargs):
        if not ProductionModeDetector.is_production_mode():
            raise RuntimeError(
                f"Function {func.__name__} can only be called in production mode. "
                f"Current mode: {ProductionModeDetector.get_environment_mode().value}"
            )
        return func(*args, **kwargs)

    return wrapper


def skip_in_production(func):
    """
    Decorator to skip a function in production mode.

    Usage:
        @skip_in_production
        def development_only_function():
            pass
    """

    def wrapper(*args, **kwargs):
        if ProductionModeDetector.is_production_mode():
            logger.warning(f"Skipping {func.__name__} in production mode")
            return None
        return func(*args, **kwargs)

    return wrapper


# Convenience functions
def is_production() -> bool:
    """Check if running in production mode."""
    return ProductionModeDetector.is_production_mode()


def get_environment() -> EnvironmentMode:
    """Get current environment mode."""
    return ProductionModeDetector.get_environment_mode()


def validate_production_ready() -> Tuple[bool, List[str]]:
    """Validate production readiness."""
    return ProductionModeDetector.validate_production_environment()


def get_production_status() -> Dict[str, bool]:
    """Get production requirements status."""
    return ProductionModeDetector.get_production_checklist()


# Exception classes for production mode
class ProductionModeError(Exception):
    """Raised when production mode validation fails."""
    pass


class ProductionModeWarning(Warning):
    """Warning for production mode issues."""
    pass


# Additional classes for backward compatibility
class ProductionModeValidator:
    """Validates production mode configurations."""
    
    def __init__(self):
        self._validation_rules = []
    
    def add_rule(self, rule_func: callable):
        """Add a validation rule."""
        self._validation_rules.append(rule_func)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate production mode configuration."""
        errors = []
        for rule_func in self._validation_rules:
            try:
                if not rule_func():
                    errors.append(f"Validation rule failed: {rule_func.__name__}")
            except Exception as e:
                errors.append(f"Validation rule error: {e}")
        return len(errors) == 0, errors


class ProductionModeMonitor:
    """Monitors production mode health."""
    
    def __init__(self):
        self._metrics = {}
        self._alerts = []
    
    def record_metric(self, name: str, value: float):
        """Record a metric."""
        self._metrics[name] = value
    
    def add_alert(self, message: str):
        """Add an alert."""
        self._alerts.append(message)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            "metrics": self._metrics,
            "alerts": self._alerts,
            "status": "healthy" if not self._alerts else "degraded"
        }


class ProductionModeHealthCheck:
    """Health check for production mode."""
    
    def __init__(self, name: str):
        self.name = name
        self.last_check = None
        self.last_result = None
    
    def run_check(self) -> Dict[str, Any]:
        """Run health check."""
        try:
            # Basic production mode check
            is_prod = ProductionModeDetector.is_production_mode()
            is_valid, errors = ProductionModeDetector.validate_production_environment()
            
            self.last_result = {
                "status": "healthy" if is_valid else "unhealthy",
                "is_production": is_prod,
                "errors": errors
            }
        except Exception as e:
            self.last_result = {
                "status": "error",
                "error": str(e)
            }
        
        self.last_check = time.time()
        return self.last_result


class ProductionModeMetrics:
    """Metrics collection for production mode."""
    
    def __init__(self):
        self._counters = defaultdict(int)
        self._gauges = {}
        self._histograms = defaultdict(list)
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter."""
        self._counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        self._gauges[name] = value
    
    def record_histogram(self, name: str, value: float):
        """Record a histogram value."""
        self._histograms[name].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": self._gauges,
            "histograms": {k: len(v) for k, v in self._histograms.items()}
        }


class ProductionModeLogger:
    """Logger for production mode."""
    
    def __init__(self, name: str = __name__):
        self.logger = logging.getLogger(name)
    
    def log_production_event(self, event: str, level: str = "info"):
        """Log a production event."""
        getattr(self.logger, level)(f"Production event: {event}")
    
    def log_security_event(self, event: str):
        """Log a security event."""
        self.logger.warning(f"Security event: {event}")
    
    def log_performance_event(self, event: str, duration: float):
        """Log a performance event."""
        self.logger.info(f"Performance event: {event} (duration: {duration:.3f}s)")


class ProductionModeSecurity:
    """Security utilities for production mode."""
    
    def __init__(self):
        self._security_checks = []
    
    def add_security_check(self, check_func: callable):
        """Add a security check."""
        self._security_checks.append(check_func)
    
    def run_security_audit(self) -> Dict[str, Any]:
        """Run security audit."""
        results = []
        for check_func in self._security_checks:
            try:
                result = check_func()
                results.append({"check": check_func.__name__, "result": result})
            except Exception as e:
                results.append({"check": check_func.__name__, "error": str(e)})
        
        return {"audit_results": results}


class ProductionModePerformance:
    """Performance monitoring for production mode."""
    
    def __init__(self):
        self._performance_metrics = {}
    
    def record_performance(self, operation: str, duration: float):
        """Record performance metric."""
        if operation not in self._performance_metrics:
            self._performance_metrics[operation] = []
        self._performance_metrics[operation].append(duration)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        for operation, durations in self._performance_metrics.items():
            if durations:
                summary[operation] = {
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "count": len(durations)
                }
        return summary


class ProductionModeCompliance:
    """Compliance checking for production mode."""
    
    def __init__(self):
        self._compliance_rules = []
    
    def add_compliance_rule(self, rule_func: callable):
        """Add a compliance rule."""
        self._compliance_rules.append(rule_func)
    
    def check_compliance(self) -> Dict[str, Any]:
        """Check compliance."""
        results = []
        for rule_func in self._compliance_rules:
            try:
                result = rule_func()
                results.append({"rule": rule_func.__name__, "compliant": result})
            except Exception as e:
                results.append({"rule": rule_func.__name__, "error": str(e)})
        
        compliant_count = sum(1 for r in results if r.get("compliant", False))
        return {
            "compliance_rate": compliant_count / len(results) if results else 0,
            "results": results
        }
