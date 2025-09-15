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

from .exceptions import ConfigurationException


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
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "cronos_ai"
    username: str = "cronos"
    password: str = "cronos123"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    
    @property
    def url(self) -> str:
        """Get database URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 100
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    decode_responses: bool = True
    
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
    """Security configuration."""
    enable_encryption: bool = True
    encryption_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_expiry_hours: int = 24
    
    # API Security
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 100
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Model security
    sign_models: bool = True
    verify_model_signatures: bool = True
    model_encryption: bool = False


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
    
    # Anomaly Detection specific
    anomaly_detection_enabled: bool = True
    anomaly_threshold: float = 0.95
    anomaly_window_size: int = 100
    
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
        config.environment = Environment(os.getenv(f"{prefix}ENVIRONMENT", config.environment.value))
        config.debug = os.getenv(f"{prefix}DEBUG", str(config.debug)).lower() == "true"
        config.log_level = LogLevel(os.getenv(f"{prefix}LOG_LEVEL", config.log_level.value))
        
        # Load database configuration
        if os.getenv(f"{prefix}DB_HOST"):
            config.database.host = os.getenv(f"{prefix}DB_HOST", config.database.host)
            config.database.port = int(os.getenv(f"{prefix}DB_PORT", str(config.database.port)))
            config.database.database = os.getenv(f"{prefix}DB_NAME", config.database.database)
            config.database.username = os.getenv(f"{prefix}DB_USER", config.database.username)
            config.database.password = os.getenv(f"{prefix}DB_PASSWORD", config.database.password)
        
        # Load Redis configuration
        if os.getenv(f"{prefix}REDIS_HOST"):
            config.redis.host = os.getenv(f"{prefix}REDIS_HOST", config.redis.host)
            config.redis.port = int(os.getenv(f"{prefix}REDIS_PORT", str(config.redis.port)))
            config.redis.password = os.getenv(f"{prefix}REDIS_PASSWORD", config.redis.password)
        
        # Load MLflow configuration
        if os.getenv(f"{prefix}MLFLOW_TRACKING_URI"):
            config.mlflow.tracking_uri = os.getenv(f"{prefix}MLFLOW_TRACKING_URI", config.mlflow.tracking_uri)
            config.mlflow.experiment_name = os.getenv(f"{prefix}MLFLOW_EXPERIMENT", config.mlflow.experiment_name)
        
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
            "discovery_enabled", "discovery_batch_size", "discovery_learning_rate",
            "field_detection_enabled", "field_detection_max_fields", 
            "anomaly_detection_enabled", "anomaly_threshold"
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
            errors.append("Field detection confidence threshold must be between 0 and 1")
        
        if errors:
            raise ConfigurationException(f"Configuration validation failed: {'; '.join(errors)}")


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