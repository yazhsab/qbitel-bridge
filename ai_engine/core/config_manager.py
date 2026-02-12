"""
Enterprise configuration management for QBITEL Bridge.
Provides centralized, validated, and environment-aware configuration management.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List, Type, get_type_hints
from dataclasses import dataclass, field, fields
from pathlib import Path
from enum import Enum
import hashlib
import threading
from contextlib import contextmanager
import tempfile
import secrets

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigValidationError(Exception):
    """Configuration validation error."""

    pass


class ConfigLoadError(Exception):
    """Configuration loading error."""

    pass


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "qbitel"
    username: str = "qbitel_user"
    password: str = ""
    ssl_mode: str = "prefer"
    connection_pool_size: int = 10
    connection_timeout: int = 30

    def __post_init__(self):
        if not self.password and os.getenv("DATABASE_PASSWORD"):
            self.password = os.getenv("DATABASE_PASSWORD")


@dataclass
class RedisConfig:
    """Redis configuration."""

    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: str = ""
    ssl: bool = False
    connection_pool_size: int = 10
    connection_timeout: int = 5

    def __post_init__(self):
        if not self.password and os.getenv("REDIS_PASSWORD"):
            self.password = os.getenv("REDIS_PASSWORD")


@dataclass
class AIEngineConfig:
    """AI Engine specific configuration."""

    model_cache_dir: str = "/tmp/qbitel_models"
    use_gpu: bool = True
    gpu_memory_limit: float = 0.8
    batch_size: int = 32
    max_sequence_length: int = 512
    ensemble_size: int = 5
    learning_rate: float = 0.001
    training_epochs: int = 100
    early_stopping_patience: int = 10
    model_save_frequency: int = 10

    # Statistical Analysis
    entropy_window_size: int = 1000
    pattern_min_frequency: int = 3
    field_boundary_confidence_threshold: float = 0.7

    # Grammar Learning
    pcfg_max_depth: int = 10
    pcfg_min_rule_frequency: int = 2
    em_max_iterations: int = 50
    em_convergence_threshold: float = 1e-6

    # Classification
    cnn_filter_sizes: List[int] = field(default_factory=lambda: [3, 4, 5])
    cnn_num_filters: int = 100
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    dropout_rate: float = 0.2

    # Validation
    max_validation_rules: int = 100
    validation_timeout: float = 5.0

    def __post_init__(self):
        # Ensure model cache directory exists
        os.makedirs(self.model_cache_dir, exist_ok=True)


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""

    # Caching
    cache_size: int = 10000
    cache_ttl: int = 3600
    use_redis_cache: bool = False
    redis_cache_prefix: str = "qbitel:"

    # Memory Management
    buffer_size: int = 8192
    memory_pool_size: int = 1000
    max_memory_usage_mb: int = 4096
    gc_threshold: float = 0.8

    # Threading
    max_threads: int = 0  # 0 = auto-detect
    io_thread_pool_size: int = 20
    cpu_thread_pool_size: int = 0  # 0 = auto-detect

    # Batching
    batch_size: int = 1000
    batch_flush_interval: float = 1.0
    max_batch_delay: float = 5.0

    def __post_init__(self):
        if self.max_threads == 0:
            import psutil

            self.max_threads = min(32, (psutil.cpu_count() or 1) * 4)

        if self.cpu_thread_pool_size == 0:
            import psutil

            self.cpu_thread_pool_size = psutil.cpu_count() or 1


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""

    # Metrics Collection
    metrics_enabled: bool = True
    metrics_flush_interval: float = 30.0
    max_metric_points: int = 10000

    # Prometheus
    prometheus_enabled: bool = True
    prometheus_host: str = "0.0.0.0"
    prometheus_port: int = 8000

    # Health Checking
    health_check_enabled: bool = True
    health_check_interval: float = 60.0
    health_check_timeout: float = 10.0

    # Alerting
    alerting_enabled: bool = True
    max_alerts: int = 1000
    alert_retention_days: int = 30

    # Alert Thresholds
    cpu_alert_threshold: float = 90.0
    memory_alert_threshold: float = 85.0
    disk_alert_threshold: float = 90.0
    error_rate_alert_threshold: float = 0.05
    response_time_alert_threshold: float = 5.0


@dataclass
class SecurityConfig:
    """Security configuration."""

    # API Security
    api_key_enabled: bool = True
    api_key_header: str = "X-API-Key"
    jwt_enabled: bool = False
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # TLS/SSL
    tls_enabled: bool = True
    tls_cert_file: str = ""
    tls_key_file: str = ""
    tls_ca_file: str = ""
    tls_verify_client: bool = False

    # Encryption
    encryption_key: str = ""
    encryption_algorithm: str = "AES-256-GCM"

    # Audit Logging
    audit_logging_enabled: bool = True
    audit_log_retention_days: int = 90
    sensitive_field_masking: bool = True

    def __post_init__(self):
        # Generate secrets if not provided
        if self.jwt_enabled and not self.jwt_secret:
            self.jwt_secret = secrets.token_urlsafe(32)
            logger.warning("Generated JWT secret - ensure this is persisted in production")

        if not self.encryption_key:
            self.encryption_key = secrets.token_urlsafe(32)
            logger.warning("Generated encryption key - ensure this is persisted in production")


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    output: str = "stdout"
    file_path: str = "/var/log/qbitel/app.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    structured_logging: bool = True
    correlation_id_enabled: bool = True
    performance_logging: bool = True
    security_logging: bool = True


@dataclass
class MainConfig:
    """Main configuration container."""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    ai_engine: AIEngineConfig = field(default_factory=AIEngineConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Application settings
    app_name: str = "QBITEL Bridge"
    app_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    worker_processes: int = 1

    def __post_init__(self):
        # Environment-specific adjustments
        if self.environment == Environment.PRODUCTION:
            self.debug = False
            self.security.api_key_enabled = True
            self.security.tls_enabled = True
            self.monitoring.alerting_enabled = True
        elif self.environment == Environment.DEVELOPMENT:
            self.debug = True
            self.security.api_key_enabled = False
            self.security.tls_enabled = False


class ConfigManager:
    """Centralized configuration management."""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self._config: Optional[MainConfig] = None
        self._config_lock = threading.RLock()
        self._watchers: List[callable] = []
        self._config_hash: Optional[str] = None

        # Default configuration files to try
        self.config_files = [
            "qbitel.yaml",
            "qbitel.yml",
            "qbitel.json",
            "config.yaml",
            "config.yml",
            "config.json",
        ]

    def load_config(
        self,
        config_file: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> MainConfig:
        """Load configuration from file or dictionary."""
        with self._config_lock:
            if config_dict:
                # Load from dictionary
                self._config = self._load_from_dict(config_dict)
            elif config_file:
                # Load from specific file
                self._config = self._load_from_file(config_file)
            else:
                # Auto-discover configuration file
                self._config = self._auto_load_config()

            # Apply environment variable overrides
            self._apply_env_overrides()

            # Validate configuration
            self._validate_config()

            # Calculate config hash for change detection
            self._config_hash = self._calculate_config_hash()

            # Notify watchers
            self._notify_watchers()

            logger.info(f"Configuration loaded for environment: {self._config.environment.value}")
            return self._config

    def _auto_load_config(self) -> MainConfig:
        """Auto-discover and load configuration file."""
        # Try environment-specific files first
        env = os.getenv("QBITEL_ENVIRONMENT", "development").lower()
        env_files = [
            f"qbitel.{env}.yaml",
            f"qbitel.{env}.yml",
            f"qbitel.{env}.json",
        ]

        # Check environment-specific files
        for filename in env_files:
            config_path = self.config_dir / filename
            if config_path.exists():
                logger.info(f"Loading environment-specific config: {config_path}")
                return self._load_from_file(str(config_path))

        # Check default files
        for filename in self.config_files:
            config_path = self.config_dir / filename
            if config_path.exists():
                logger.info(f"Loading default config: {config_path}")
                return self._load_from_file(str(config_path))

        # No config file found, use defaults
        logger.warning("No configuration file found, using defaults")
        return MainConfig()

    def _load_from_file(self, config_file: str) -> MainConfig:
        """Load configuration from file."""
        config_path = Path(config_file)

        if not config_path.exists():
            raise ConfigLoadError(f"Configuration file not found: {config_file}")

        try:
            with open(config_path, "r") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                elif config_path.suffix.lower() == ".json":
                    data = json.load(f)
                else:
                    raise ConfigLoadError(f"Unsupported configuration file format: {config_path.suffix}")

            return self._load_from_dict(data or {})

        except Exception as e:
            raise ConfigLoadError(f"Error loading configuration from {config_file}: {e}")

    def _load_from_dict(self, config_dict: Dict[str, Any]) -> MainConfig:
        """Load configuration from dictionary."""
        try:
            # Handle nested configuration
            config_data = {}

            # Map top-level keys
            for key, value in config_dict.items():
                if key in ["environment"]:
                    if isinstance(value, str):
                        config_data[key] = Environment(value.lower())
                    else:
                        config_data[key] = value
                else:
                    config_data[key] = value

            # Create configuration objects for nested sections
            nested_configs = {
                "database": DatabaseConfig,
                "redis": RedisConfig,
                "ai_engine": AIEngineConfig,
                "performance": PerformanceConfig,
                "monitoring": MonitoringConfig,
                "security": SecurityConfig,
                "logging": LoggingConfig,
            }

            for section, config_class in nested_configs.items():
                if section in config_data and isinstance(config_data[section], dict):
                    try:
                        config_data[section] = config_class(**config_data[section])
                    except TypeError as e:
                        logger.warning(f"Error creating {section} config: {e}")
                        config_data[section] = config_class()

            return MainConfig(**config_data)

        except Exception as e:
            raise ConfigLoadError(f"Error parsing configuration: {e}")

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        if not self._config:
            return

        # Define environment variable mappings
        env_mappings = {
            # Database
            "DATABASE_HOST": ("database", "host"),
            "DATABASE_PORT": ("database", "port"),
            "DATABASE_NAME": ("database", "database"),
            "DATABASE_USER": ("database", "username"),
            "DATABASE_PASSWORD": ("database", "password"),
            # Redis
            "REDIS_HOST": ("redis", "host"),
            "REDIS_PORT": ("redis", "port"),
            "REDIS_PASSWORD": ("redis", "password"),
            # API
            "API_HOST": (None, "api_host"),
            "API_PORT": (None, "api_port"),
            # Environment
            "QBITEL_ENVIRONMENT": (None, "environment"),
            "DEBUG": (None, "debug"),
            # Security
            "JWT_SECRET": ("security", "jwt_secret"),
            "API_KEY_ENABLED": ("security", "api_key_enabled"),
            "TLS_ENABLED": ("security", "tls_enabled"),
            # Monitoring
            "PROMETHEUS_PORT": ("monitoring", "prometheus_port"),
            "METRICS_ENABLED": ("monitoring", "metrics_enabled"),
        }

        for env_var, (section, field) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Type conversion
                    if section:
                        section_obj = getattr(self._config, section)
                        current_value = getattr(section_obj, field)
                    else:
                        current_value = getattr(self._config, field)

                    # Convert to appropriate type
                    if isinstance(current_value, bool):
                        converted_value = value.lower() in ("true", "1", "yes", "on")
                    elif isinstance(current_value, int):
                        converted_value = int(value)
                    elif isinstance(current_value, float):
                        converted_value = float(value)
                    elif isinstance(current_value, Environment):
                        converted_value = Environment(value.lower())
                    else:
                        converted_value = value

                    # Apply override
                    if section:
                        setattr(getattr(self._config, section), field, converted_value)
                    else:
                        setattr(self._config, field, converted_value)

                    logger.debug(f"Applied environment override: {env_var}={value}")

                except (ValueError, AttributeError) as e:
                    logger.warning(f"Could not apply environment override {env_var}={value}: {e}")

    def _validate_config(self):
        """Validate configuration."""
        if not self._config:
            raise ConfigValidationError("No configuration loaded")

        # Validate required fields
        if self._config.environment == Environment.PRODUCTION:
            # Production-specific validations
            if not self._config.security.api_key_enabled and not self._config.security.jwt_enabled:
                raise ConfigValidationError("Authentication must be enabled in production")

            if self._config.debug:
                logger.warning("Debug mode enabled in production - this is not recommended")

        # Validate port ranges
        if not (1 <= self._config.api_port <= 65535):
            raise ConfigValidationError(f"Invalid API port: {self._config.api_port}")

        if not (1 <= self._config.database.port <= 65535):
            raise ConfigValidationError(f"Invalid database port: {self._config.database.port}")

        # Validate directories
        try:
            os.makedirs(self._config.ai_engine.model_cache_dir, exist_ok=True)
        except OSError as e:
            raise ConfigValidationError(f"Cannot create model cache directory: {e}")

        logger.info("Configuration validation passed")

    def _calculate_config_hash(self) -> str:
        """Calculate hash of current configuration for change detection."""
        if not self._config:
            return ""

        # Convert config to dict and hash
        config_str = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def get_config(self) -> MainConfig:
        """Get current configuration."""
        if self._config is None:
            self.load_config()
        return self._config

    def reload_config(self) -> bool:
        """Reload configuration and check for changes."""
        old_hash = self._config_hash

        try:
            self.load_config()
            return old_hash != self._config_hash
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False

    def add_config_watcher(self, callback: callable):
        """Add configuration change watcher."""
        self._watchers.append(callback)

    def _notify_watchers(self):
        """Notify configuration change watchers."""
        for watcher in self._watchers:
            try:
                watcher(self._config)
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        if not self._config:
            return {}

        def convert_dataclass(obj):
            if hasattr(obj, "__dataclass_fields__"):
                result = {}
                for field_info in fields(obj):
                    value = getattr(obj, field_info.name)
                    if hasattr(value, "__dataclass_fields__"):
                        result[field_info.name] = convert_dataclass(value)
                    elif isinstance(value, Enum):
                        result[field_info.name] = value.value
                    else:
                        result[field_info.name] = value
                return result
            else:
                return obj

        return convert_dataclass(self._config)

    def save_config(self, output_file: str, format: str = "yaml"):
        """Save current configuration to file."""
        if not self._config:
            raise ConfigValidationError("No configuration loaded")

        config_dict = self.to_dict()

        try:
            with open(output_file, "w") as f:
                if format.lower() in ["yaml", "yml"]:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == "json":
                    json.dump(config_dict, f, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Configuration saved to {output_file}")

        except Exception as e:
            raise ConfigLoadError(f"Error saving configuration: {e}")

    @contextmanager
    def temporary_config(self, overrides: Dict[str, Any]):
        """Temporarily override configuration values."""
        if not self._config:
            raise ConfigValidationError("No configuration loaded")

        # Save current config
        original_dict = self.to_dict()

        try:
            # Apply overrides
            modified_dict = original_dict.copy()
            modified_dict.update(overrides)
            temp_config = self._load_from_dict(modified_dict)

            # Replace current config
            old_config = self._config
            self._config = temp_config

            yield self._config

        finally:
            # Restore original config
            self._config = old_config


# Global configuration manager instance
_config_manager = None


def get_config_manager(config_dir: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager


def get_config() -> MainConfig:
    """Get current configuration."""
    return get_config_manager().get_config()


def load_config(
    config_file: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    config_dir: Optional[str] = None,
) -> MainConfig:
    """Load configuration."""
    manager = get_config_manager(config_dir)
    return manager.load_config(config_file, config_dict)


# Configuration validation decorators
def requires_config(func):
    """Decorator that ensures configuration is loaded."""

    def wrapper(*args, **kwargs):
        config = get_config()  # This will load config if not already loaded
        return func(*args, **kwargs)

    return wrapper


def config_section(section_name: str):
    """Decorator to inject specific configuration section."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            config = get_config()
            section_config = getattr(config, section_name, None)
            if section_config is None:
                raise ConfigValidationError(f"Configuration section '{section_name}' not found")
            return func(section_config, *args, **kwargs)

        return wrapper

    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    example_config = {
        "environment": "development",
        "debug": True,
        "api_host": "0.0.0.0",
        "api_port": 8080,
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "qbitel_dev",
            "username": "dev_user",
        },
        "ai_engine": {
            "use_gpu": False,
            "batch_size": 16,
            "model_cache_dir": "/tmp/qbitel_dev_models",
        },
        "security": {"api_key_enabled": False, "tls_enabled": False},
    }

    # Test configuration loading
    config_manager = ConfigManager()
    config = config_manager.load_config(config_dict=example_config)

    print(f"Loaded configuration for environment: {config.environment.value}")
    print(f"API will run on {config.api_host}:{config.api_port}")
    print(f"Database: {config.database.host}:{config.database.port}/{config.database.database}")
    print(f"AI Engine GPU: {config.ai_engine.use_gpu}")
