"""
CRONOS AI Engine - Configuration Management

This module provides comprehensive configuration management for the AI Engine,
supporting multiple environments and configuration sources.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum
import logging

from .exceptions import ConfigurationException

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Environment types for configuration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration with production-ready validation."""

    host: str = "localhost"
    port: int = 5432
    database: str = "cronos_ai"
    username: str = "cronos"
    password: str = ""  # MUST be set via environment variable or secrets manager
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False

    # Validation constants
    MIN_PASSWORD_LENGTH: int = 16
    WEAK_PATTERNS: List[str] = field(
        default_factory=lambda: [
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
        ]
    )

    def __post_init__(self):
        """Load password from environment with comprehensive validation."""
        if not self.password:
            # Try environment variables in priority order
            self.password = (
                os.getenv("CRONOS_AI_DB_PASSWORD")
                or os.getenv("DATABASE_PASSWORD")
                or ""
            )

            if not self.password:
                error_msg = (
                    "Database password not configured!\n"
                    "REQUIRED: Set one of the following environment variables:\n"
                    "  - CRONOS_AI_DB_PASSWORD (recommended)\n"
                    "  - DATABASE_PASSWORD\n\n"
                    "Security Requirements:\n"
                    f"  - Minimum length: {self.MIN_PASSWORD_LENGTH} characters\n"
                    "  - No common weak patterns\n"
                    "  - Use strong, randomly generated passwords\n\n"
                    "Generate a secure password:\n"
                    '  python -c "import secrets; print(secrets.token_urlsafe(32))"'
                )

                # In production, this should be a hard error
                if self._is_production_mode():
                    raise ConfigurationException(error_msg)
                else:
                    logger.warning(error_msg)

        # Validate password if set
        if self.password:
            self._validate_password()

    def _is_production_mode(self) -> bool:
        """Check if running in production mode."""
        env = os.getenv(
            "CRONOS_AI_ENVIRONMENT", os.getenv("ENVIRONMENT", "development")
        ).lower()
        return env in ("production", "prod")

    def _validate_password(self) -> None:
        """Validate password meets security requirements."""
        errors = []

        # Check minimum length
        if len(self.password) < self.MIN_PASSWORD_LENGTH:
            errors.append(
                f"Password too short: {len(self.password)} characters "
                f"(minimum: {self.MIN_PASSWORD_LENGTH})"
            )

        # Check for weak patterns
        password_lower = self.password.lower()
        found_patterns = [p for p in self.WEAK_PATTERNS if p in password_lower]
        if found_patterns:
            errors.append(
                f"Password contains weak patterns: {', '.join(found_patterns)}"
            )

        # Check for sequential characters
        if self._has_sequential_chars(self.password):
            errors.append(
                "Password contains sequential characters (e.g., '123', 'abc')"
            )

        # Check for repeated characters
        if self._has_repeated_chars(self.password):
            errors.append("Password contains too many repeated characters")

        if errors:
            error_msg = (
                "Database password validation failed:\n"
                + "\n".join(f"  - {error}" for error in errors)
                + "\n\nRemediation:\n"
                "  1. Generate a strong password:\n"
                '     python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
                "  2. Set the environment variable:\n"
                "     export CRONOS_AI_DB_PASSWORD='<generated_password>'\n"
                "  3. Store securely in your secrets manager"
            )

            if self._is_production_mode():
                raise ConfigurationException(error_msg)
            else:
                logger.warning(error_msg)

    def _has_sequential_chars(self, password: str, min_length: int = 3) -> bool:
        """Check for sequential characters."""
        for i in range(len(password) - min_length + 1):
            substr = password[i : i + min_length]
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

    def _has_repeated_chars(self, password: str, max_repeats: int = 3) -> bool:
        """Check for repeated characters."""
        for i in range(len(password) - max_repeats + 1):
            if len(set(password[i : i + max_repeats])) == 1:
                return True
        return False

    @property
    def url(self) -> str:
        """Get database URL."""
        if not self.password:
            raise ConfigurationException(
                "Database password not configured. Set CRONOS_AI_DB_PASSWORD environment variable."
            )
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis configuration with production-ready validation."""

    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 100
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    decode_responses: bool = True

    # Validation constants
    MIN_PASSWORD_LENGTH: int = 16
    WEAK_PATTERNS: List[str] = field(
        default_factory=lambda: [
            "password",
            "admin",
            "test",
            "demo",
            "123456",
            "redis",
            "cache",
            "letmein",
            "welcome",
        ]
    )

    def __post_init__(self):
        """Load password from environment with comprehensive validation."""
        if not self.password:
            # Try environment variables in priority order
            self.password = os.getenv("CRONOS_AI_REDIS_PASSWORD") or os.getenv(
                "REDIS_PASSWORD"
            )

            # In production, Redis MUST have authentication
            if not self.password and self._is_production_mode():
                error_msg = (
                    "Redis password not configured in PRODUCTION mode!\n"
                    "REQUIRED: Set one of the following environment variables:\n"
                    "  - CRONOS_AI_REDIS_PASSWORD (recommended)\n"
                    "  - REDIS_PASSWORD\n\n"
                    "Security Requirements:\n"
                    f"  - Minimum length: {self.MIN_PASSWORD_LENGTH} characters\n"
                    "  - No common weak patterns\n"
                    "  - Use strong, randomly generated passwords\n\n"
                    "Generate a secure password:\n"
                    '  python -c "import secrets; print(secrets.token_urlsafe(32))"\n\n'
                    "WARNING: Running Redis without authentication in production is a CRITICAL security risk!"
                )
                raise ConfigurationException(error_msg)
            elif not self.password:
                logger.warning(
                    "Redis password not set. This is acceptable for development but "
                    "REQUIRED for production. Set CRONOS_AI_REDIS_PASSWORD environment variable."
                )

        # Validate password if set
        if self.password:
            self._validate_password()

    def _is_production_mode(self) -> bool:
        """Check if running in production mode."""
        env = os.getenv(
            "CRONOS_AI_ENVIRONMENT", os.getenv("ENVIRONMENT", "development")
        ).lower()
        return env in ("production", "prod")

    def _validate_password(self) -> None:
        """Validate password meets security requirements."""
        errors = []

        # Check minimum length
        if len(self.password) < self.MIN_PASSWORD_LENGTH:
            errors.append(
                f"Password too short: {len(self.password)} characters "
                f"(minimum: {self.MIN_PASSWORD_LENGTH})"
            )

        # Check for weak patterns
        password_lower = self.password.lower()
        found_patterns = [p for p in self.WEAK_PATTERNS if p in password_lower]
        if found_patterns:
            errors.append(
                f"Password contains weak patterns: {', '.join(found_patterns)}"
            )

        if errors:
            error_msg = (
                "Redis password validation failed:\n"
                + "\n".join(f"  - {error}" for error in errors)
                + "\n\nRemediation:\n"
                "  1. Generate a strong password:\n"
                '     python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
                "  2. Set the environment variable:\n"
                "     export CRONOS_AI_REDIS_PASSWORD='<generated_password>'\n"
                "  3. Update Redis configuration to require authentication:\n"
                "     requirepass <generated_password>"
            )

            if self._is_production_mode():
                raise ConfigurationException(error_msg)
            else:
                logger.warning(error_msg)

    @property
    def url(self) -> str:
        """Get Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class KafkaConfig:
    """Kafka configuration."""

    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    batch_size: int = 16384
    linger_ms: int = 10
    buffer_memory: int = 33554432
    max_request_size: int = 1048576


@dataclass
class MLflowConfig:
    """MLflow configuration."""

    tracking_uri: str = "http://localhost:5000"
    artifact_location: Optional[str] = None
    experiment_name: str = "cronos-ai-experiments"
    registry_uri: Optional[str] = None
    default_artifact_root: Optional[str] = None
    backend_store_uri: Optional[str] = None


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str
    version: str = "1.0.0"
    type: str = "pytorch"  # pytorch, tensorflow, onnx
    device: str = "auto"  # auto, cpu, cuda, cuda:0, etc.
    batch_size: int = 32
    max_sequence_length: int = 512
    num_workers: int = 4
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None

    # Model-specific parameters
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    dropout_prob: float = 0.1
    activation: str = "gelu"


@dataclass
class TrainingConfig:
    """Training configuration."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100

    # Optimizer configuration
    optimizer: str = "adamw"  # adamw, adam, sgd
    lr_scheduler: str = "linear"  # linear, cosine, polynomial

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class InferenceConfig:
    """Inference configuration."""

    batch_size: int = 1
    max_batch_delay: float = 0.1  # seconds
    max_queue_size: int = 1000
    timeout: float = 30.0
    device: str = "auto"
    num_workers: int = 2

    # Model optimization
    use_dynamic_batching: bool = True
    use_tensorrt: bool = False
    use_onnx: bool = False
    optimize_for_latency: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True

    # Prometheus metrics
    metrics_port: int = 8080
    metrics_path: str = "/metrics"

    # Jaeger tracing
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    service_name: str = "cronos-ai-engine"

    # Performance monitoring
    collect_model_metrics: bool = True
    collect_data_metrics: bool = True
    collect_system_metrics: bool = True

    # Alerting
    alert_on_high_latency: bool = True
    alert_on_low_accuracy: bool = True
    alert_on_data_drift: bool = True
    latency_threshold_ms: float = 100.0
    accuracy_threshold: float = 0.85
    drift_threshold: float = 0.1


@dataclass
class SecurityConfig:
    """Security configuration with production-ready validation."""

    enable_encryption: bool = True
    encryption_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_expiry_hours: int = 24
    api_key: Optional[str] = None

    # API Security
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 100
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Model security
    sign_models: bool = True
    verify_model_signatures: bool = True
    model_encryption: bool = False

    # Validation constants
    MIN_SECRET_LENGTH: int = 32
    MIN_API_KEY_LENGTH: int = 32
    WEAK_PATTERNS: List[str] = field(
        default_factory=lambda: [
            "secret",
            "password",
            "admin",
            "test",
            "demo",
            "123456",
            "key",
            "token",
            "letmein",
            "welcome",
            "changeme",
        ]
    )

    def __post_init__(self):
        """Load secrets from secrets manager or environment with comprehensive validation."""
        # Try to load from secrets manager first
        self._load_from_secrets_manager()

        # Load JWT secret
        if not self.jwt_secret:
            self.jwt_secret = os.getenv("CRONOS_AI_JWT_SECRET") or os.getenv(
                "JWT_SECRET"
            )

            if not self.jwt_secret:
                error_msg = (
                    "JWT secret not configured!\n"
                    "REQUIRED: Configure in secrets manager (Vault/AWS/Azure) or set environment variable:\n"
                    "  - CRONOS_AI_JWT_SECRET (recommended)\n"
                    "  - JWT_SECRET\n\n"
                    "Security Requirements:\n"
                    f"  - Minimum length: {self.MIN_SECRET_LENGTH} characters\n"
                    "  - No common weak patterns\n"
                    "  - Use cryptographically secure random generation\n\n"
                    "Generate a secure JWT secret:\n"
                    '  python -c "import secrets; print(secrets.token_urlsafe(48))"\n\n'
                    "WARNING: Without a JWT secret, authentication tokens will not persist across restarts!"
                )

                if self._is_production_mode():
                    raise ConfigurationException(error_msg)
                else:
                    logger.warning(error_msg)

        # Load encryption key
        if not self.encryption_key:
            self.encryption_key = os.getenv("CRONOS_AI_ENCRYPTION_KEY") or os.getenv(
                "ENCRYPTION_KEY"
            )

            if not self.encryption_key and self.enable_encryption:
                error_msg = (
                    "Encryption key not configured but encryption is enabled!\n"
                    "REQUIRED: Configure in secrets manager (Vault/AWS/Azure) or set environment variable:\n"
                    "  - CRONOS_AI_ENCRYPTION_KEY (recommended)\n"
                    "  - ENCRYPTION_KEY\n\n"
                    "Security Requirements:\n"
                    f"  - Minimum length: {self.MIN_SECRET_LENGTH} characters\n"
                    "  - Must be base64-encoded 256-bit key\n"
                    "  - Use cryptographically secure random generation\n\n"
                    "Generate a secure encryption key:\n"
                    '  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"\n\n'
                    "WARNING: Data encryption will fail without a valid encryption key!"
                )

                if self._is_production_mode():
                    raise ConfigurationException(error_msg)
                else:
                    logger.warning(error_msg)

        # Load API key
        if not self.api_key:
            self.api_key = os.getenv("CRONOS_AI_API_KEY") or os.getenv("API_KEY")

            if not self.api_key and self._is_production_mode():
                logger.warning(
                    "API key not configured. Set CRONOS_AI_API_KEY environment variable "
                    "or configure in secrets manager for API authentication."
                )

        # Validate all secrets
        if self.jwt_secret:
            self._validate_secret("JWT secret", self.jwt_secret)
        if self.encryption_key:
            self._validate_secret("Encryption key", self.encryption_key)
        if self.api_key:
            self._validate_api_key()

        # Validate CORS configuration in production
        if self._is_production_mode() and "*" in self.cors_origins:
            raise ConfigurationException(
                "CORS wildcard (*) is FORBIDDEN in production mode!\n"
                "Security Requirement:\n"
                "  - Specify explicit allowed origins\n"
                "  - Example: ['https://app.example.com', 'https://dashboard.example.com']\n\n"
                "Remediation:\n"
                "  1. Update cors_origins in your configuration\n"
                "  2. Remove '*' and add specific domain names\n"
                "  3. Use HTTPS URLs only in production\n\n"
                "WARNING: CORS wildcard allows any origin to access your API, "
                "creating a critical security vulnerability!"
            )

    def _load_from_secrets_manager(self):
        """Load secrets from secrets manager if available."""
        try:
            # Avoid circular import
            from ..security.secrets_manager import get_secrets_manager

            secrets_mgr = get_secrets_manager()

            # Try to load JWT secret
            if not self.jwt_secret:
                jwt_secret = secrets_mgr.get_secret("jwt_secret")
                if jwt_secret:
                    self.jwt_secret = jwt_secret
                    logger.info("JWT secret loaded from secrets manager")

            # Try to load encryption key
            if not self.encryption_key:
                encryption_key = secrets_mgr.get_secret("encryption_key")
                if encryption_key:
                    self.encryption_key = encryption_key
                    logger.info("Encryption key loaded from secrets manager")

            # Try to load API key
            if not self.api_key:
                api_key = secrets_mgr.get_secret("api_key")
                if api_key:
                    self.api_key = api_key
                    logger.info("API key loaded from secrets manager")

        except Exception as e:
            logger.debug(f"Could not load secrets from secrets manager: {e}")

    def _is_production_mode(self) -> bool:
        """Check if running in production mode."""
        env = os.getenv(
            "CRONOS_AI_ENVIRONMENT", os.getenv("ENVIRONMENT", "development")
        ).lower()
        return env in ("production", "prod")

    def _validate_secret(self, secret_name: str, secret_value: str) -> None:
        """Validate secret meets security requirements."""
        errors = []

        # Check minimum length
        if len(secret_value) < self.MIN_SECRET_LENGTH:
            errors.append(
                f"{secret_name} too short: {len(secret_value)} characters "
                f"(minimum: {self.MIN_SECRET_LENGTH})"
            )

        # Check for weak patterns
        secret_lower = secret_value.lower()
        found_patterns = [p for p in self.WEAK_PATTERNS if p in secret_lower]
        if found_patterns:
            errors.append(
                f"{secret_name} contains weak patterns: {', '.join(found_patterns)}"
            )

        # Check entropy (basic check)
        if len(set(secret_value)) < len(secret_value) * 0.5:
            errors.append(
                f"{secret_name} has low entropy (too many repeated characters)"
            )

        # Check for sequential characters
        if self._has_sequential_chars(secret_value):
            errors.append(f"{secret_name} contains sequential characters")

        if errors:
            error_msg = (
                f"{secret_name} validation failed:\n"
                + "\n".join(f"  - {error}" for error in errors)
                + "\n\nRemediation:\n"
                "  1. Generate a cryptographically secure secret:\n"
                '     python -c "import secrets; print(secrets.token_urlsafe(48))"\n'
                "  2. Set the appropriate environment variable:\n"
                f"     export CRONOS_AI_{secret_name.upper().replace(' ', '_')}='<generated_secret>'\n"
                "  3. Store securely in your secrets manager\n"
                "  4. Never commit secrets to version control"
            )

            if self._is_production_mode():
                raise ConfigurationException(error_msg)
            else:
                logger.warning(error_msg)

    def _validate_api_key(self) -> None:
        """Validate API key meets security requirements."""
        errors = []

        # Check minimum length
        if len(self.api_key) < self.MIN_API_KEY_LENGTH:
            errors.append(
                f"API key too short: {len(self.api_key)} characters "
                f"(minimum: {self.MIN_API_KEY_LENGTH})"
            )

        # Check format (should be alphanumeric with special chars)
        if not any(c.isdigit() for c in self.api_key):
            errors.append("API key should contain digits")
        if not any(c.isalpha() for c in self.api_key):
            errors.append("API key should contain letters")

        # Check for weak patterns
        api_key_lower = self.api_key.lower()
        found_patterns = [p for p in self.WEAK_PATTERNS if p in api_key_lower]
        if found_patterns:
            errors.append(
                f"API key contains weak patterns: {', '.join(found_patterns)}"
            )

        if errors:
            error_msg = (
                "API key validation failed:\n"
                + "\n".join(f"  - {error}" for error in errors)
                + "\n\nRemediation:\n"
                "  1. Generate a secure API key:\n"
                "     python -c \"import secrets; print('cronos_' + secrets.token_urlsafe(32))\"\n"
                "  2. Set the environment variable:\n"
                "     export CRONOS_AI_API_KEY='<generated_key>'\n"
                "  3. Implement API key rotation policy\n"
                "  4. Monitor API key usage for anomalies"
            )

            if self._is_production_mode():
                raise ConfigurationException(error_msg)
            else:
                logger.warning(error_msg)

    def _has_sequential_chars(self, value: str, min_length: int = 4) -> bool:
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

    def validate(self) -> None:
        """Validate security configuration."""
        errors = []

        # Validate JWT secret strength
        if self.jwt_secret and len(self.jwt_secret) < self.MIN_SECRET_LENGTH:
            errors.append(
                f"JWT secret must be at least {self.MIN_SECRET_LENGTH} characters long"
            )

        # Validate encryption key
        if self.encryption_key and len(self.encryption_key) < self.MIN_SECRET_LENGTH:
            errors.append(
                f"Encryption key must be at least {self.MIN_SECRET_LENGTH} characters long"
            )

        # Validate API key
        if self.api_key and len(self.api_key) < self.MIN_API_KEY_LENGTH:
            errors.append(
                f"API key must be at least {self.MIN_API_KEY_LENGTH} characters long"
            )

        # Production-specific validations
        if self._is_production_mode():
            if not self.jwt_secret:
                errors.append("JWT secret is REQUIRED in production mode")
            if self.enable_encryption and not self.encryption_key:
                errors.append(
                    "Encryption key is REQUIRED when encryption is enabled in production"
                )
            if "*" in self.cors_origins:
                errors.append(
                    "CORS wildcard (*) is FORBIDDEN in production mode. "
                    "Specify explicit allowed origins for security."
                )

        if errors:
            raise ConfigurationException(
                f"Security configuration validation failed:\n"
                + "\n".join(f"  - {error}" for error in errors)
            )


@dataclass
class Config:
    """Main configuration class."""

    # Environment and basic settings
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    service_name: str = "cronos-ai-engine"
    version: str = "1.0.0"

    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Protocol Discovery specific
    discovery_enabled: bool = True
    discovery_batch_size: int = 1000
    discovery_learning_rate: float = 1e-4
    discovery_max_protocols: int = 1000

    # Field Detection specific
    field_detection_enabled: bool = True
    field_detection_max_fields: int = 100
    field_detection_confidence_threshold: float = 0.8
    # Feature flag for ML-based field type classifier (scaffold code)
    field_detection_ml_classifier_enabled: bool = False

    # Anomaly Detection specific
    anomaly_detection_enabled: bool = True
    anomaly_threshold: float = 0.95
    anomaly_window_size: int = 100

    # Compliance Reporter specific
    compliance_enabled: bool = True
    compliance_assessment_interval_hours: int = 24
    compliance_cache_ttl_hours: int = 6
    compliance_max_concurrent_assessments: int = 3
    compliance_report_retention_days: int = 365
    compliance_audit_trail_enabled: bool = True
    compliance_blockchain_enabled: bool = True
    compliance_timescaledb_integration: bool = True
    compliance_redis_integration: bool = True
    compliance_security_integration: bool = True

    # Translation Studio specific
    translation_studio_enabled: bool = True
    # Feature flag for processing step implementations (scaffold code)
    translation_studio_processing_steps_enabled: bool = False

    # Security Orchestrator specific
    security_orchestrator_enabled: bool = True
    # Feature flag for action implementations (scaffold code)
    security_orchestrator_actions_enabled: bool = False

    @classmethod
    def load_from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigurationException(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            return cls._from_dict(config_data)

        except yaml.YAMLError as e:
            raise ConfigurationException(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationException(f"Error loading configuration file: {e}")

    @classmethod
    def load_from_env(cls, prefix: str = "CRONOS_AI_") -> "Config":
        """Load configuration from environment variables."""
        config = cls()

        # Load basic settings
        config.environment = Environment(
            os.getenv(f"{prefix}ENVIRONMENT", config.environment.value)
        )
        config.debug = os.getenv(f"{prefix}DEBUG", str(config.debug)).lower() == "true"
        config.log_level = LogLevel(
            os.getenv(f"{prefix}LOG_LEVEL", config.log_level.value)
        )

        # Load database configuration
        if os.getenv(f"{prefix}DB_HOST"):
            config.database.host = os.getenv(f"{prefix}DB_HOST", config.database.host)
            config.database.port = int(
                os.getenv(f"{prefix}DB_PORT", str(config.database.port))
            )
            config.database.database = os.getenv(
                f"{prefix}DB_NAME", config.database.database
            )
            config.database.username = os.getenv(
                f"{prefix}DB_USER", config.database.username
            )
            config.database.password = os.getenv(
                f"{prefix}DB_PASSWORD", config.database.password
            )

        # Load Redis configuration
        if os.getenv(f"{prefix}REDIS_HOST"):
            config.redis.host = os.getenv(f"{prefix}REDIS_HOST", config.redis.host)
            config.redis.port = int(
                os.getenv(f"{prefix}REDIS_PORT", str(config.redis.port))
            )
            config.redis.password = os.getenv(
                f"{prefix}REDIS_PASSWORD", config.redis.password
            )

        # Load MLflow configuration
        if os.getenv(f"{prefix}MLFLOW_TRACKING_URI"):
            config.mlflow.tracking_uri = os.getenv(
                f"{prefix}MLFLOW_TRACKING_URI", config.mlflow.tracking_uri
            )
            config.mlflow.experiment_name = os.getenv(
                f"{prefix}MLFLOW_EXPERIMENT", config.mlflow.experiment_name
            )

        return config

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        config = cls()

        # Update basic settings
        if "environment" in data:
            config.environment = Environment(data["environment"])
        if "debug" in data:
            config.debug = data["debug"]
        if "log_level" in data:
            config.log_level = LogLevel(data["log_level"])

        # Update component configurations
        if "database" in data:
            config.database = DatabaseConfig(**data["database"])
        if "redis" in data:
            config.redis = RedisConfig(**data["redis"])
        if "kafka" in data:
            config.kafka = KafkaConfig(**data["kafka"])
        if "mlflow" in data:
            config.mlflow = MLflowConfig(**data["mlflow"])
        if "monitoring" in data:
            config.monitoring = MonitoringConfig(**data["monitoring"])
        if "security" in data:
            config.security = SecurityConfig(**data["security"])

        # Update model configurations
        if "models" in data:
            config.models = {
                name: ModelConfig(name=name, **model_data)
                for name, model_data in data["models"].items()
            }
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "inference" in data:
            config.inference = InferenceConfig(**data["inference"])

        # Update feature-specific settings
        for key in [
            "discovery_enabled",
            "discovery_batch_size",
            "discovery_learning_rate",
            "field_detection_enabled",
            "field_detection_max_fields",
            "anomaly_detection_enabled",
            "anomaly_threshold",
        ]:
            if key in data:
                setattr(config, key, data[key])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "log_level": self.log_level.value,
            "service_name": self.service_name,
            "version": self.version,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                # Don't include password in serialization
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
            },
            "models": {
                name: {
                    "version": model.version,
                    "type": model.type,
                    "device": model.device,
                    "batch_size": model.batch_size,
                }
                for name, model in self.models.items()
            },
            "discovery_enabled": self.discovery_enabled,
            "field_detection_enabled": self.field_detection_enabled,
            "anomaly_detection_enabled": self.anomaly_detection_enabled,
        }

    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []

        # Validate training configuration
        if self.training.learning_rate <= 0:
            errors.append("Training learning rate must be positive")
        if self.training.batch_size <= 0:
            errors.append("Training batch size must be positive")
        if self.training.epochs <= 0:
            errors.append("Training epochs must be positive")

        # Validate inference configuration
        if self.inference.batch_size <= 0:
            errors.append("Inference batch size must be positive")
        if self.inference.timeout <= 0:
            errors.append("Inference timeout must be positive")

        # Validate thresholds
        if not 0 <= self.anomaly_threshold <= 1:
            errors.append("Anomaly threshold must be between 0 and 1")
        if not 0 <= self.field_detection_confidence_threshold <= 1:
            errors.append(
                "Field detection confidence threshold must be between 0 and 1"
            )

        if errors:
            raise ConfigurationException(
                f"Configuration validation failed: {'; '.join(errors)}"
            )


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load_from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    config.validate()
    _config = config


def load_config(config_path: Union[str, Path]) -> Config:
    """Load and set configuration from file."""
    config = Config.load_from_file(config_path)
    set_config(config)
    return config
