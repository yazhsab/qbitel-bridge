"""
CRONOS AI Engine - Configuration Validation
Comprehensive validation for production configurations.
"""

import os
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity."""
    ERROR = "error"  # Blocks deployment
    WARNING = "warning"  # Should be fixed but not blocking
    INFO = "info"  # Informational


@dataclass
class ValidationIssue:
    """Configuration validation issue."""
    severity: ValidationSeverity
    field: str
    message: str
    suggestion: Optional[str] = None


class ConfigValidator:
    """
    Comprehensive configuration validator for production readiness.
    """
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        self.issues: List[ValidationIssue] = []
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate entire configuration.
        
        Returns:
            (is_valid, issues)
        """
        self.issues = []
        
        # Run all validation checks
        self._validate_environment(config)
        self._validate_database(config)
        self._validate_redis(config)
        self._validate_security(config)
        self._validate_monitoring(config)
        self._validate_performance(config)
        self._validate_logging(config)
        self._validate_secrets(config)
        self._validate_rate_limiting(config)
        self._validate_tls(config)
        
        # Check if any errors exist
        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
        is_valid = not has_errors
        
        return is_valid, self.issues
    
    def _validate_environment(self, config: Dict[str, Any]):
        """Validate environment configuration."""
        env = config.get("environment", "")
        
        if not env:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="environment",
                message="Environment not specified",
                suggestion="Set environment to 'production', 'staging', or 'development'"
            ))
        
        if self.environment == "production":
            if config.get("debug", False):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="debug",
                    message="Debug mode enabled in production",
                    suggestion="Set debug: false"
                ))
    
    def _validate_database(self, config: Dict[str, Any]):
        """Validate database configuration."""
        db_config = config.get("database", {})
        
        # Check required fields
        required_fields = ["host", "port", "database", "username"]
        for field in required_fields:
            if not db_config.get(field):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field=f"database.{field}",
                    message=f"Database {field} not configured"
                ))
        
        # Check password
        password = db_config.get("password", "")
        if not password and not os.getenv("DATABASE_PASSWORD"):
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="database.password",
                message="Database password not set",
                suggestion="Set DATABASE_PASSWORD environment variable"
            ))
        
        # Production-specific checks
        if self.environment == "production":
            # SSL mode
            ssl_mode = db_config.get("ssl_mode", "")
            if ssl_mode not in ["require", "verify-ca", "verify-full"]:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="database.ssl_mode",
                    message="SSL not enforced for database connection",
                    suggestion="Set ssl_mode to 'require' or higher"
                ))
            
            # Connection pool
            pool_size = db_config.get("connection_pool_size", 0)
            if pool_size < 10:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="database.connection_pool_size",
                    message="Connection pool size too small for production",
                    suggestion="Set connection_pool_size to at least 20"
                ))
    
    def _validate_redis(self, config: Dict[str, Any]):
        """Validate Redis configuration."""
        redis_config = config.get("redis", {})
        
        if not redis_config.get("host"):
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="redis.host",
                message="Redis host not configured"
            ))
        
        if self.environment == "production":
            # SSL for Redis
            if not redis_config.get("ssl", False):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="redis.ssl",
                    message="SSL not enabled for Redis",
                    suggestion="Enable SSL for Redis in production"
                ))
            
            # Password
            if not redis_config.get("password") and not os.getenv("REDIS_PASSWORD"):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="redis.password",
                    message="Redis password not set",
                    suggestion="Set REDIS_PASSWORD environment variable"
                ))
    
    def _validate_security(self, config: Dict[str, Any]):
        """Validate security configuration."""
        security_config = config.get("security", {})
        
        if self.environment == "production":
            # API key
            if not security_config.get("api_key_enabled", False):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="security.api_key_enabled",
                    message="API key authentication not enabled",
                    suggestion="Enable API key authentication in production"
                ))
            
            # JWT
            if not security_config.get("jwt_enabled", False):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="security.jwt_enabled",
                    message="JWT authentication not enabled",
                    suggestion="Enable JWT authentication in production"
                ))
            
            # JWT secret
            jwt_secret = security_config.get("jwt_secret", "")
            if not jwt_secret and not os.getenv("JWT_SECRET"):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="security.jwt_secret",
                    message="JWT secret not set",
                    suggestion="Set JWT_SECRET environment variable (min 32 chars)"
                ))
            elif jwt_secret and len(jwt_secret) < 32:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="security.jwt_secret",
                    message="JWT secret too short",
                    suggestion="Use at least 32 characters for JWT secret"
                ))
            
            # Encryption key
            encryption_key = security_config.get("encryption_key", "")
            if not encryption_key and not os.getenv("ENCRYPTION_KEY"):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="security.encryption_key",
                    message="Encryption key not set",
                    suggestion="Set ENCRYPTION_KEY environment variable (min 32 chars)"
                ))
            
            # Audit logging
            if not security_config.get("audit_logging_enabled", False):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="security.audit_logging_enabled",
                    message="Audit logging not enabled",
                    suggestion="Enable audit logging for compliance"
                ))
    
    def _validate_monitoring(self, config: Dict[str, Any]):
        """Validate monitoring configuration."""
        monitoring_config = config.get("monitoring", {})
        
        if not monitoring_config.get("metrics_enabled", False):
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="monitoring.metrics_enabled",
                message="Metrics collection not enabled"
            ))
        
        if not monitoring_config.get("prometheus_enabled", False):
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="monitoring.prometheus_enabled",
                message="Prometheus metrics not enabled"
            ))
        
        if self.environment == "production":
            if not monitoring_config.get("alerting_enabled", False):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="monitoring.alerting_enabled",
                    message="Alerting not enabled",
                    suggestion="Enable alerting for production monitoring"
                ))
    
    def _validate_performance(self, config: Dict[str, Any]):
        """Validate performance configuration."""
        perf_config = config.get("performance", {})
        
        if self.environment == "production":
            # Cache configuration
            if not perf_config.get("use_redis_cache", False):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="performance.use_redis_cache",
                    message="Redis cache not enabled",
                    suggestion="Enable Redis cache for better performance"
                ))
            
            # Memory limits
            max_memory = perf_config.get("max_memory_usage_mb", 0)
            if max_memory < 4096:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="performance.max_memory_usage_mb",
                    message="Memory limit too low for production",
                    suggestion="Set max_memory_usage_mb to at least 8192"
                ))
    
    def _validate_logging(self, config: Dict[str, Any]):
        """Validate logging configuration."""
        logging_config = config.get("logging", {})
        
        if self.environment == "production":
            # Log level
            log_level = logging_config.get("level", "").upper()
            if log_level == "DEBUG":
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="logging.level",
                    message="Debug logging enabled in production",
                    suggestion="Set log level to INFO or WARNING"
                ))
            
            # Structured logging
            if not logging_config.get("structured_logging", False):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="logging.structured_logging",
                    message="Structured logging not enabled",
                    suggestion="Enable structured logging for better log analysis"
                ))
            
            # Log format
            if logging_config.get("format") != "json":
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    field="logging.format",
                    message="JSON logging not enabled",
                    suggestion="Use JSON format for production logs"
                ))
    
    def _validate_secrets(self, config: Dict[str, Any]):
        """Validate secrets management."""
        secrets_config = config.get("secrets_manager", {})
        
        if self.environment == "production":
            if not secrets_config.get("enabled", False):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="secrets_manager.enabled",
                    message="Secrets manager not enabled",
                    suggestion="Use HashiCorp Vault or cloud secrets manager"
                ))
        
        # Check for hardcoded secrets
        self._check_hardcoded_secrets(config)
    
    def _validate_rate_limiting(self, config: Dict[str, Any]):
        """Validate rate limiting configuration."""
        rate_limit_config = config.get("rate_limiting", {})
        
        if self.environment == "production":
            if not rate_limit_config.get("enabled", False):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="rate_limiting.enabled",
                    message="Rate limiting not enabled",
                    suggestion="Enable rate limiting to prevent DoS attacks"
                ))
            
            # Check limits
            default_limit = rate_limit_config.get("default_limit", 0)
            if default_limit == 0 or default_limit > 1000:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="rate_limiting.default_limit",
                    message="Rate limit not properly configured",
                    suggestion="Set reasonable rate limits (e.g., 100 req/min)"
                ))
    
    def _validate_tls(self, config: Dict[str, Any]):
        """Validate TLS/SSL configuration."""
        security_config = config.get("security", {})
        
        if self.environment == "production":
            if not security_config.get("tls_enabled", False):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="security.tls_enabled",
                    message="TLS not enabled",
                    suggestion="Enable TLS for all production traffic"
                ))
            
            # Check certificate files
            if security_config.get("tls_enabled"):
                cert_file = security_config.get("tls_cert_file", "")
                key_file = security_config.get("tls_key_file", "")
                
                if not cert_file:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field="security.tls_cert_file",
                        message="TLS certificate file not specified"
                    ))
                
                if not key_file:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field="security.tls_key_file",
                        message="TLS key file not specified"
                    ))
    
    def _check_hardcoded_secrets(self, config: Dict[str, Any], path: str = ""):
        """Recursively check for hardcoded secrets."""
        sensitive_patterns = [
            r'password',
            r'secret',
            r'key',
            r'token',
            r'credential'
        ]
        
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                self._check_hardcoded_secrets(value, current_path)
            elif isinstance(value, str):
                # Check if field name suggests it's sensitive
                is_sensitive = any(
                    re.search(pattern, key, re.IGNORECASE)
                    for pattern in sensitive_patterns
                )
                
                if is_sensitive and value and not value.startswith("${"):
                    # Value is not empty and not an environment variable reference
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field=current_path,
                        message="Hardcoded secret detected",
                        suggestion="Use environment variables or secrets manager"
                    ))
    
    def print_report(self):
        """Print validation report."""
        if not self.issues:
            print("✅ Configuration validation passed!")
            return
        
        print(f"\n{'='*80}")
        print(f"Configuration Validation Report - Environment: {self.environment}")
        print(f"{'='*80}\n")
        
        # Group by severity
        errors = [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
        warnings = [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
        infos = [i for i in self.issues if i.severity == ValidationSeverity.INFO]
        
        if errors:
            print(f"🔴 ERRORS ({len(errors)}):")
            for issue in errors:
                print(f"  - {issue.field}: {issue.message}")
                if issue.suggestion:
                    print(f"    💡 {issue.suggestion}")
            print()
        
        if warnings:
            print(f"⚠️  WARNINGS ({len(warnings)}):")
            for issue in warnings:
                print(f"  - {issue.field}: {issue.message}")
                if issue.suggestion:
                    print(f"    💡 {issue.suggestion}")
            print()
        
        if infos:
            print(f"ℹ️  INFO ({len(infos)}):")
            for issue in infos:
                print(f"  - {issue.field}: {issue.message}")
                if issue.suggestion:
                    print(f"    💡 {issue.suggestion}")
            print()
        
        print(f"{'='*80}\n")


def validate_config_file(config_path: str, environment: str = "production") -> bool:
    """
    Validate configuration file.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name
        
    Returns:
        True if valid
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate
        validator = ConfigValidator(environment)
        is_valid, issues = validator.validate_config(config)
        
        # Print report
        validator.print_report()
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Failed to validate config: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python config_validator.py <config_file> [environment]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    env = sys.argv[2] if len(sys.argv) > 2 else "production"
    
    is_valid = validate_config_file(config_file, env)
    sys.exit(0 if is_valid else 1)