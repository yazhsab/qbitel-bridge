#!/usr/bin/env python3
"""
CRONOS AI - Unified Configuration Management System
Production-ready configuration management for all system components.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum
import consul
import etcd3
from functools import lru_cache
import threading
from datetime import datetime, timedelta
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigBackend(Enum):
    """Configuration backend types"""
    LOCAL_FILE = "local_file"
    CONSUL = "consul"
    ETCD = "etcd"
    KUBERNETES = "kubernetes"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = "require"
    connection_pool_size: int = 20
    connection_timeout: int = 30

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str
    port: int = 6379
    database: int = 0
    password: str = ""
    ssl: bool = True
    connection_pool_size: int = 20
    connection_timeout: int = 5

@dataclass
class KafkaConfig:
    """Kafka configuration"""
    bootstrap_servers: List[str]
    security_protocol: str = "SASL_SSL"
    sasl_mechanism: str = "PLAIN"
    sasl_username: str = ""
    sasl_password: str = ""
    ssl_ca_location: str = ""
    auto_offset_reset: str = "latest"
    group_id: str = "cronos-ai"

@dataclass
class AIEngineConfig:
    """AI Engine configuration"""
    model_cache_dir: str
    use_gpu: bool = True
    gpu_memory_limit: float = 0.9
    batch_size: int = 64
    max_sequence_length: int = 1024
    ensemble_size: int = 7
    learning_rate: float = 0.0005
    training_epochs: int = 200
    early_stopping_patience: int = 15
    
@dataclass
class SecurityConfig:
    """Security configuration"""
    tls_enabled: bool = True
    tls_cert_file: str = ""
    tls_key_file: str = ""
    tls_ca_file: str = ""
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 12
    encryption_key: str = ""
    encryption_algorithm: str = "AES-256-GCM"

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    metrics_enabled: bool = True
    prometheus_enabled: bool = True
    prometheus_host: str = "0.0.0.0"
    prometheus_port: int = 8000
    health_check_enabled: bool = True
    health_check_interval: float = 30.0
    alerting_enabled: bool = True

@dataclass
class CronosAIConfig:
    """Main CRONOS AI configuration"""
    environment: str
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    worker_processes: int = 4
    database: DatabaseConfig = None
    redis: RedisConfig = None
    kafka: KafkaConfig = None
    ai_engine: AIEngineConfig = None
    security: SecurityConfig = None
    monitoring: MonitoringConfig = None

class ConfigurationManager:
    """
    Unified configuration management system for CRONOS AI.
    Supports multiple backends and real-time configuration updates.
    """
    
    def __init__(self, backend: ConfigBackend = ConfigBackend.LOCAL_FILE,
                 config_path: Optional[str] = None,
                 consul_host: str = "localhost",
                 consul_port: int = 8500,
                 etcd_host: str = "localhost",
                 etcd_port: int = 2379):
        
        self.backend = backend
        self.config_path = config_path or self._get_default_config_path()
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.etcd_host = etcd_host
        self.etcd_port = etcd_port
        
        self._config_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)  # 5 minute cache TTL
        self._lock = threading.RLock()
        
        # Initialize backend connections
        self._consul_client = None
        self._etcd_client = None
        self._init_backend()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration path based on environment"""
        env = os.getenv('CRONOS_ENV', 'development')
        base_path = Path(__file__).parent.parent.parent / 'config'
        
        if env == 'production':
            return str(base_path / 'cronos_ai.production.yaml')
        else:
            return str(base_path / 'cronos_ai.yaml')
    
    def _init_backend(self):
        """Initialize configuration backend"""
        try:
            if self.backend == ConfigBackend.CONSUL:
                self._consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
                logger.info(f"Connected to Consul at {self.consul_host}:{self.consul_port}")
            elif self.backend == ConfigBackend.ETCD:
                self._etcd_client = etcd3.client(host=self.etcd_host, port=self.etcd_port)
                logger.info(f"Connected to etcd at {self.etcd_host}:{self.etcd_port}")
        except Exception as e:
            logger.warning(f"Failed to connect to {self.backend.value}: {e}")
            logger.warning("Falling back to local file configuration")
            self.backend = ConfigBackend.LOCAL_FILE
    
    @lru_cache(maxsize=1)
    def load_config(self, force_reload: bool = False) -> CronosAIConfig:
        """Load configuration from backend"""
        cache_key = "main_config"
        
        with self._lock:
            # Check cache first
            if not force_reload and cache_key in self._config_cache:
                cache_time = self._cache_timestamps.get(cache_key)
                if cache_time and datetime.now() - cache_time < self._cache_ttl:
                    return self._config_cache[cache_key]
            
            # Load from backend
            config_data = self._load_from_backend()
            config = self._parse_config(config_data)
            
            # Update cache
            self._config_cache[cache_key] = config
            self._cache_timestamps[cache_key] = datetime.now()
            
            return config
    
    def _load_from_backend(self) -> Dict[str, Any]:
        """Load configuration data from the configured backend"""
        if self.backend == ConfigBackend.LOCAL_FILE:
            return self._load_from_file()
        elif self.backend == ConfigBackend.CONSUL:
            return self._load_from_consul()
        elif self.backend == ConfigBackend.ETCD:
            return self._load_from_etcd()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _load_from_file(self) -> Dict[str, Any]:
        """Load configuration from local file"""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            return {}
    
    def _load_from_consul(self) -> Dict[str, Any]:
        """Load configuration from Consul"""
        try:
            index, data = self._consul_client.kv.get('cronos-ai/config', recurse=True)
            if not data:
                return {}
            
            config = {}
            for item in data:
                key = item['Key'].replace('cronos-ai/config/', '')
                try:
                    config[key] = json.loads(item['Value'].decode('utf-8'))
                except:
                    config[key] = item['Value'].decode('utf-8')
            
            return config
        except Exception as e:
            logger.error(f"Error loading from Consul: {e}")
            return {}
    
    def _load_from_etcd(self) -> Dict[str, Any]:
        """Load configuration from etcd"""
        try:
            config = {}
            for value, metadata in self._etcd_client.get_prefix('/cronos-ai/config/'):
                key = metadata.key.decode('utf-8').replace('/cronos-ai/config/', '')
                try:
                    config[key] = json.loads(value.decode('utf-8'))
                except:
                    config[key] = value.decode('utf-8')
            
            return config
        except Exception as e:
            logger.error(f"Error loading from etcd: {e}")
            return {}
    
    def _parse_config(self, config_data: Dict[str, Any]) -> CronosAIConfig:
        """Parse configuration data into structured format"""
        # Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)
        
        # Parse individual sections
        database_config = None
        if 'database' in config_data:
            db_data = config_data['database']
            database_config = DatabaseConfig(**db_data)
        
        redis_config = None
        if 'redis' in config_data:
            redis_data = config_data['redis']
            redis_config = RedisConfig(**redis_data)
            
        kafka_config = None
        if 'kafka' in config_data:
            kafka_data = config_data['kafka']
            kafka_config = KafkaConfig(**kafka_data)
        
        ai_engine_config = None
        if 'ai_engine' in config_data:
            ai_data = config_data['ai_engine']
            ai_engine_config = AIEngineConfig(**ai_data)
        
        security_config = None
        if 'security' in config_data:
            sec_data = config_data['security']
            security_config = SecurityConfig(**sec_data)
        
        monitoring_config = None
        if 'monitoring' in config_data:
            mon_data = config_data['monitoring']
            monitoring_config = MonitoringConfig(**mon_data)
        
        # Create main config
        main_config = CronosAIConfig(
            environment=config_data.get('environment', 'development'),
            debug=config_data.get('debug', False),
            api_host=config_data.get('api_host', '0.0.0.0'),
            api_port=config_data.get('api_port', 8080),
            worker_processes=config_data.get('worker_processes', 4),
            database=database_config,
            redis=redis_config,
            kafka=kafka_config,
            ai_engine=ai_engine_config,
            security=security_config,
            monitoring=monitoring_config
        )
        
        return main_config
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration"""
        env_mappings = {
            'DATABASE_PASSWORD': ['database', 'password'],
            'REDIS_PASSWORD': ['redis', 'password'],
            'KAFKA_USERNAME': ['kafka', 'sasl_username'],
            'KAFKA_PASSWORD': ['kafka', 'sasl_password'],
            'JWT_SECRET': ['security', 'jwt_secret'],
            'ENCRYPTION_KEY': ['security', 'encryption_key'],
            'TLS_CERT_FILE': ['security', 'tls_cert_file'],
            'TLS_KEY_FILE': ['security', 'tls_key_file'],
            'AI_MODEL_CACHE_DIR': ['ai_engine', 'model_cache_dir'],
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Navigate to nested dict
                current = config_data
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = env_value
        
        return config_data
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a specific service"""
        config = self.load_config()
        
        service_configs = {
            'ai_engine': asdict(config.ai_engine) if config.ai_engine else {},
            'database': asdict(config.database) if config.database else {},
            'redis': asdict(config.redis) if config.redis else {},
            'kafka': asdict(config.kafka) if config.kafka else {},
            'security': asdict(config.security) if config.security else {},
            'monitoring': asdict(config.monitoring) if config.monitoring else {},
        }
        
        # Add common config to all services
        common_config = {
            'environment': config.environment,
            'debug': config.debug,
            'api_host': config.api_host,
            'api_port': config.api_port,
        }
        
        if service_name in service_configs:
            service_configs[service_name].update(common_config)
            return service_configs[service_name]
        
        return common_config
    
    def update_config(self, key: str, value: Any, service: Optional[str] = None):
        """Update configuration value"""
        if self.backend == ConfigBackend.LOCAL_FILE:
            logger.warning("Configuration updates not supported for local file backend")
            return
        
        config_key = f"cronos-ai/config/{service}/{key}" if service else f"cronos-ai/config/{key}"
        
        try:
            if self.backend == ConfigBackend.CONSUL:
                self._consul_client.kv.put(config_key, json.dumps(value))
            elif self.backend == ConfigBackend.ETCD:
                self._etcd_client.put(config_key, json.dumps(value))
            
            # Clear cache to force reload
            with self._lock:
                self._config_cache.clear()
                self._cache_timestamps.clear()
            
            logger.info(f"Updated configuration: {config_key}")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
    
    def watch_config_changes(self, callback):
        """Watch for configuration changes"""
        if self.backend == ConfigBackend.CONSUL:
            self._watch_consul_changes(callback)
        elif self.backend == ConfigBackend.ETCD:
            self._watch_etcd_changes(callback)
        else:
            logger.warning("Configuration watching not supported for current backend")
    
    def _watch_consul_changes(self, callback):
        """Watch Consul for configuration changes"""
        def watch_loop():
            index = None
            while True:
                try:
                    index, data = self._consul_client.kv.get('cronos-ai/config', 
                                                           recurse=True, 
                                                           index=index, 
                                                           wait='10s')
                    if data:
                        # Clear cache and notify callback
                        with self._lock:
                            self._config_cache.clear()
                            self._cache_timestamps.clear()
                        callback('consul_change')
                except Exception as e:
                    logger.error(f"Error watching Consul changes: {e}")
        
        watch_thread = threading.Thread(target=watch_loop, daemon=True)
        watch_thread.start()
    
    def _watch_etcd_changes(self, callback):
        """Watch etcd for configuration changes"""
        def watch_loop():
            try:
                events_iterator, cancel = self._etcd_client.watch_prefix('/cronos-ai/config/')
                for event in events_iterator:
                    # Clear cache and notify callback
                    with self._lock:
                        self._config_cache.clear()
                        self._cache_timestamps.clear()
                    callback('etcd_change')
            except Exception as e:
                logger.error(f"Error watching etcd changes: {e}")
        
        watch_thread = threading.Thread(target=watch_loop, daemon=True)
        watch_thread.start()
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        config = self.load_config()
        
        if config.environment == 'production':
            # Production-specific validations
            if config.security and not config.security.jwt_secret:
                issues.append("JWT secret is required in production")
            
            if config.security and not config.security.encryption_key:
                issues.append("Encryption key is required in production")
            
            if config.security and not config.security.tls_enabled:
                issues.append("TLS must be enabled in production")
            
            if config.database and not config.database.password:
                issues.append("Database password is required in production")
        
        return issues
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for configuration system"""
        health = {
            'status': 'healthy',
            'backend': self.backend.value,
            'config_loaded': len(self._config_cache) > 0,
            'last_reload': self._cache_timestamps.get('main_config'),
            'issues': []
        }
        
        try:
            # Test configuration loading
            config = self.load_config()
            health['config_valid'] = True
            
            # Validate configuration
            validation_issues = self.validate_config()
            if validation_issues:
                health['issues'].extend(validation_issues)
                health['status'] = 'warning'
        
        except Exception as e:
            health['status'] = 'unhealthy'
            health['config_valid'] = False
            health['error'] = str(e)
        
        return health

# Global configuration manager instance
_config_manager = None

def get_config_manager(**kwargs) -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(**kwargs)
    return _config_manager

def get_config() -> CronosAIConfig:
    """Get current configuration"""
    return get_config_manager().load_config()

def get_service_config(service_name: str) -> Dict[str, Any]:
    """Get configuration for specific service"""
    return get_config_manager().get_service_config(service_name)

if __name__ == "__main__":
    # Example usage
    config_mgr = ConfigurationManager()
    config = config_mgr.load_config()
    
    print(f"Environment: {config.environment}")
    print(f"API Port: {config.api_port}")
    
    # Get service-specific config
    ai_config = config_mgr.get_service_config('ai_engine')
    print(f"AI Engine GPU enabled: {ai_config.get('use_gpu')}")
    
    # Health check
    health = config_mgr.health_check()
    print(f"Config Health: {health}")