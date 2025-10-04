"""
CRONOS AI Engine - Centralized Configuration Service

This module provides enterprise-grade configuration management with
dynamic configuration updates, environment-specific settings, versioning,
and distributed configuration synchronization.
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
import weakref
import copy
from contextlib import asynccontextmanager

import yaml
import aiofiles
import etcd3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .exceptions import ConfigException, ValidationException
from .structured_logging import get_logger


class ConfigEnvironment(str, Enum):
    """Configuration environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class ConfigFormat(str, Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"


class ConfigSource(str, Enum):
    """Configuration sources."""
    FILE = "file"
    ETCD = "etcd"
    CONSUL = "consul"
    DATABASE = "database"
    ENVIRONMENT = "environment"
    REMOTE_HTTP = "remote_http"


@dataclass
class ConfigMetadata:
    """Configuration metadata."""
    key: str
    version: str
    environment: ConfigEnvironment
    source: ConfigSource
    created_at: datetime
    updated_at: datetime
    checksum: str
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    encrypted: bool = False


@dataclass
class ConfigChange:
    """Configuration change record."""
    change_id: str
    key: str
    old_value: Any
    new_value: Any
    changed_by: str
    timestamp: datetime
    change_type: str  # "create", "update", "delete"
    reason: str = ""


class ConfigValidator:
    """Configuration value validator."""
    
    def __init__(self):
        self.validators: Dict[str, Callable] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}
    
    def register_validator(self, key_pattern: str, validator_func: Callable[[Any], bool]):
        """Register a validator function for a key pattern."""
        self.validators[key_pattern] = validator_func
    
    def register_schema(self, key_pattern: str, schema: Dict[str, Any]):
        """Register a JSON schema for a key pattern."""
        self.schemas[key_pattern] = schema
    
    def validate(self, key: str, value: Any) -> bool:
        """Validate a configuration value."""
        import fnmatch
        
        # Check custom validators
        for pattern, validator in self.validators.items():
            if fnmatch.fnmatch(key, pattern):
                try:
                    if not validator(value):
                        return False
                except Exception:
                    return False
        
        # Check schemas
        for pattern, schema in self.schemas.items():
            if fnmatch.fnmatch(key, pattern):
                try:
                    import jsonschema
                    jsonschema.validate(value, schema)
                except Exception:
                    return False
        
        return True


class ConfigWatcher:
    """File system watcher for configuration changes."""
    
    def __init__(self, config_service: 'ConfigurationService'):
        self.config_service = config_service
        self.logger = get_logger(__name__)
        self.observer = Observer()
        self.watching = False
    
    def start_watching(self, paths: List[Path]):
        """Start watching configuration directories."""
        if self.watching:
            return
        
        for path in paths:
            if path.exists():
                event_handler = ConfigFileHandler(self.config_service)
                self.observer.schedule(event_handler, str(path), recursive=True)
        
        self.observer.start()
        self.watching = True
        self.logger.info(f"Started watching {len(paths)} configuration paths")
    
    def stop_watching(self):
        """Stop watching configuration changes."""
        if self.watching:
            self.observer.stop()
            self.observer.join()
            self.watching = False
            self.logger.info("Stopped configuration watching")


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration changes."""
    
    def __init__(self, config_service: 'ConfigurationService'):
        self.config_service = config_service
        self.logger = get_logger(__name__)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix in ['.json', '.yaml', '.yml', '.toml']:
            asyncio.create_task(self.config_service.reload_file(file_path))
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix in ['.json', '.yaml', '.yml', '.toml']:
            asyncio.create_task(self.config_service.load_file(file_path))


class ConfigurationStore:
    """Base class for configuration storage backends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def get(self, key: str, environment: ConfigEnvironment) -> Optional[Any]:
        """Get configuration value."""
        raise NotImplementedError
    
    async def set(self, key: str, value: Any, environment: ConfigEnvironment, metadata: ConfigMetadata) -> bool:
        """Set configuration value."""
        raise NotImplementedError
    
    async def delete(self, key: str, environment: ConfigEnvironment) -> bool:
        """Delete configuration key."""
        raise NotImplementedError
    
    async def list_keys(self, prefix: str = "", environment: Optional[ConfigEnvironment] = None) -> List[str]:
        """List configuration keys."""
        raise NotImplementedError
    
    async def get_metadata(self, key: str, environment: ConfigEnvironment) -> Optional[ConfigMetadata]:
        """Get configuration metadata."""
        raise NotImplementedError


class FileConfigurationStore(ConfigurationStore):
    """File-based configuration store."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_path = Path(config.get('base_path', './config'))
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.metadata_cache: Dict[str, ConfigMetadata] = {}
        self.cache_lock = threading.RLock()
    
    def _get_file_path(self, environment: ConfigEnvironment, format_type: ConfigFormat = ConfigFormat.YAML) -> Path:
        """Get file path for environment."""
        filename = f"{environment.value}.{format_type.value}"
        return self.base_path / filename
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum."""
        return hashlib.sha256(data).hexdigest()
    
    async def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
            
            if file_path.suffix == '.json':
                return json.loads(content)
            elif file_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(content) or {}
            elif file_path.suffix == '.toml':
                import toml
                return toml.loads(content)
            else:
                raise ConfigException(f"Unsupported file format: {file_path.suffix}")
        
        except FileNotFoundError:
            return {}
        except Exception as e:
            raise ConfigException(f"Failed to load config file {file_path}: {e}")
    
    async def _save_file(self, file_path: Path, data: Dict[str, Any]):
        """Save configuration to file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.suffix == '.json':
                content = json.dumps(data, indent=2, sort_keys=True)
            elif file_path.suffix in ['.yaml', '.yml']:
                content = yaml.dump(data, default_flow_style=False, sort_keys=True)
            elif file_path.suffix == '.toml':
                import toml
                content = toml.dumps(data)
            else:
                raise ConfigException(f"Unsupported file format: {file_path.suffix}")
            
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(content)
        
        except Exception as e:
            raise ConfigException(f"Failed to save config file {file_path}: {e}")
    
    async def get(self, key: str, environment: ConfigEnvironment) -> Optional[Any]:
        """Get configuration value."""
        env_key = f"{environment.value}:{key}"
        
        with self.cache_lock:
            if env_key in self.cache:
                return self.cache[env_key]
        
        # Load from file
        file_path = self._get_file_path(environment)
        data = await self._load_file(file_path)
        
        # Navigate nested keys
        keys = key.split('.')
        value = data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        # Cache the result
        with self.cache_lock:
            self.cache[env_key] = value
        
        return value
    
    async def set(self, key: str, value: Any, environment: ConfigEnvironment, metadata: ConfigMetadata) -> bool:
        """Set configuration value."""
        try:
            file_path = self._get_file_path(environment)
            data = await self._load_file(file_path)
            
            # Navigate to the correct nested location
            keys = key.split('.')
            current = data
            
            # Create nested structure if needed
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = value
            
            # Save to file
            await self._save_file(file_path, data)
            
            # Update cache
            env_key = f"{environment.value}:{key}"
            with self.cache_lock:
                self.cache[env_key] = value
                self.metadata_cache[env_key] = metadata
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set config {key}: {e}")
            return False
    
    async def delete(self, key: str, environment: ConfigEnvironment) -> bool:
        """Delete configuration key."""
        try:
            file_path = self._get_file_path(environment)
            data = await self._load_file(file_path)
            
            # Navigate to the correct nested location
            keys = key.split('.')
            current = data
            
            # Navigate to parent
            for k in keys[:-1]:
                if k not in current:
                    return False
                current = current[k]
            
            # Delete the key
            if keys[-1] in current:
                del current[keys[-1]]
                
                # Save to file
                await self._save_file(file_path, data)
                
                # Remove from cache
                env_key = f"{environment.value}:{key}"
                with self.cache_lock:
                    self.cache.pop(env_key, None)
                    self.metadata_cache.pop(env_key, None)
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete config {key}: {e}")
            return False
    
    async def list_keys(self, prefix: str = "", environment: Optional[ConfigEnvironment] = None) -> List[str]:
        """List configuration keys."""
        keys = []
        
        environments = [environment] if environment else list(ConfigEnvironment)
        
        for env in environments:
            try:
                file_path = self._get_file_path(env)
                data = await self._load_file(file_path)
                
                # Flatten nested keys
                flat_keys = self._flatten_keys(data, prefix)
                keys.extend([f"{env.value}:{k}" for k in flat_keys])
                
            except Exception as e:
                self.logger.warning(f"Failed to list keys for {env}: {e}")
        
        return keys
    
    def _flatten_keys(self, data: Dict[str, Any], prefix: str = "") -> List[str]:
        """Flatten nested dictionary keys."""
        keys = []
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                keys.extend(self._flatten_keys(value, full_key))
            else:
                keys.append(full_key)
        
        return keys
    
    async def get_metadata(self, key: str, environment: ConfigEnvironment) -> Optional[ConfigMetadata]:
        """Get configuration metadata."""
        env_key = f"{environment.value}:{key}"
        
        with self.cache_lock:
            if env_key in self.metadata_cache:
                return self.metadata_cache[env_key]
        
        # For file-based storage, generate basic metadata
        file_path = self._get_file_path(environment)
        if file_path.exists():
            stat = file_path.stat()
            
            # Calculate checksum
            with open(file_path, 'rb') as f:
                checksum = self._calculate_checksum(f.read())
            
            metadata = ConfigMetadata(
                key=key,
                version="1.0",
                environment=environment,
                source=ConfigSource.FILE,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                updated_at=datetime.fromtimestamp(stat.st_mtime),
                checksum=checksum
            )
            
            with self.cache_lock:
                self.metadata_cache[env_key] = metadata
            
            return metadata
        
        return None


class EtcdConfigurationStore(ConfigurationStore):
    """etcd-based configuration store."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.etcd_client = etcd3.client(
            host=config.get('host', 'localhost'),
            port=config.get('port', 2379),
            user=config.get('user'),
            password=config.get('password')
        )
        self.prefix = config.get('prefix', '/cronos-ai/config/')
    
    def _get_key(self, key: str, environment: ConfigEnvironment) -> str:
        """Get etcd key with prefix."""
        return f"{self.prefix}{environment.value}/{key}"
    
    async def get(self, key: str, environment: ConfigEnvironment) -> Optional[Any]:
        """Get configuration value from etcd."""
        try:
            etcd_key = self._get_key(key, environment)
            value, metadata = await asyncio.get_event_loop().run_in_executor(
                None, self.etcd_client.get, etcd_key
            )
            
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value.decode('utf-8'))
            except json.JSONDecodeError:
                return value.decode('utf-8')
                
        except Exception as e:
            self.logger.error(f"Failed to get config from etcd {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, environment: ConfigEnvironment, metadata: ConfigMetadata) -> bool:
        """Set configuration value in etcd."""
        try:
            etcd_key = self._get_key(key, environment)
            
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            # Set in etcd
            success = await asyncio.get_event_loop().run_in_executor(
                None, self.etcd_client.put, etcd_key, serialized_value
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to set config in etcd {key}: {e}")
            return False
    
    async def delete(self, key: str, environment: ConfigEnvironment) -> bool:
        """Delete configuration key from etcd."""
        try:
            etcd_key = self._get_key(key, environment)
            deleted = await asyncio.get_event_loop().run_in_executor(
                None, self.etcd_client.delete, etcd_key
            )
            return deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete config from etcd {key}: {e}")
            return False
    
    async def list_keys(self, prefix: str = "", environment: Optional[ConfigEnvironment] = None) -> List[str]:
        """List configuration keys from etcd."""
        try:
            if environment:
                search_prefix = self._get_key(prefix, environment)
            else:
                search_prefix = f"{self.prefix}{prefix}"
            
            keys = await asyncio.get_event_loop().run_in_executor(
                None, lambda: [key.decode('utf-8') for key, _ in self.etcd_client.get_prefix(search_prefix)]
            )
            
            # Remove prefix from keys
            return [key[len(self.prefix):] for key in keys]
            
        except Exception as e:
            self.logger.error(f"Failed to list keys from etcd: {e}")
            return []


class ConfigurationService:
    """
    Centralized configuration management service.
    
    Provides enterprise-grade configuration management with dynamic updates,
    versioning, validation, and multiple storage backends.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize configuration service."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Environment and defaults
        self.current_environment = ConfigEnvironment(config.get('environment', 'development'))
        self.default_values: Dict[str, Any] = config.get('defaults', {})
        
        # Storage backend
        storage_config = config.get('storage', {'type': 'file'})
        self.store = self._create_store(storage_config)
        
        # Validation
        self.validator = ConfigValidator()
        self._setup_default_validators()
        
        # Change tracking
        self.change_history: List[ConfigChange] = []
        self.change_lock = threading.RLock()
        
        # Watchers and callbacks
        self.watchers: Dict[str, List[Callable]] = {}
        self.watcher_lock = threading.RLock()
        
        # File watching
        self.file_watcher = ConfigWatcher(self)
        
        # Cache for frequently accessed values
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(seconds=config.get('cache_ttl_seconds', 300))
        self.cache_lock = threading.RLock()
        
        self.logger.info(f"ConfigurationService initialized for {self.current_environment.value} environment")
    
    def _create_store(self, storage_config: Dict[str, Any]) -> ConfigurationStore:
        """Create configuration store based on config."""
        store_type = storage_config.get('type', 'file')
        
        if store_type == 'file':
            return FileConfigurationStore(storage_config)
        elif store_type == 'etcd':
            return EtcdConfigurationStore(storage_config)
        else:
            raise ConfigException(f"Unsupported storage type: {store_type}")
    
    def _setup_default_validators(self):
        """Setup default validators for common configuration patterns."""
        
        # URL validator
        def validate_url(value):
            import re
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            return bool(url_pattern.match(str(value)))
        
        self.validator.register_validator("*_url", validate_url)
        self.validator.register_validator("*_endpoint", validate_url)
        
        # Email validator
        def validate_email(value):
            import re
            email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
            return bool(email_pattern.match(str(value)))
        
        self.validator.register_validator("*_email", validate_email)
        
        # Port validator
        def validate_port(value):
            try:
                port = int(value)
                return 1 <= port <= 65535
            except (ValueError, TypeError):
                return False
        
        self.validator.register_validator("*_port", validate_port)
        
        # Positive number validator
        def validate_positive_number(value):
            try:
                return float(value) > 0
            except (ValueError, TypeError):
                return False
        
        self.validator.register_validator("*_timeout", validate_positive_number)
        self.validator.register_validator("*_interval", validate_positive_number)
    
    async def get(
        self,
        key: str,
        default: Any = None,
        environment: Optional[ConfigEnvironment] = None,
        use_cache: bool = True
    ) -> Any:
        """Get configuration value."""
        env = environment or self.current_environment
        
        # Check cache first
        if use_cache:
            cache_key = f"{env.value}:{key}"
            with self.cache_lock:
                if cache_key in self.cache:
                    value, cached_at = self.cache[cache_key]
                    if datetime.utcnow() - cached_at < self.cache_ttl:
                        return value
                    else:
                        # Remove expired cache entry
                        del self.cache[cache_key]
        
        # Get from store
        try:
            value = await self.store.get(key, env)
            
            if value is None:
                # Try default values
                if key in self.default_values:
                    value = self.default_values[key]
                else:
                    value = default
            
            # Cache the result
            if use_cache and value is not None:
                cache_key = f"{env.value}:{key}"
                with self.cache_lock:
                    self.cache[cache_key] = (value, datetime.utcnow())
            
            return value
            
        except Exception as e:
            self.logger.error(f"Failed to get config {key}: {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        environment: Optional[ConfigEnvironment] = None,
        changed_by: str = "system",
        reason: str = ""
    ) -> bool:
        """Set configuration value."""
        env = environment or self.current_environment
        
        try:
            # Validate the value
            if not self.validator.validate(key, value):
                raise ValidationException(f"Invalid value for key {key}")
            
            # Get old value for change tracking
            old_value = await self.get(key, environment=env, use_cache=False)
            
            # Create metadata
            metadata = ConfigMetadata(
                key=key,
                version="1.0",
                environment=env,
                source=ConfigSource.FILE,  # Will be updated based on store type
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                checksum=""
            )
            
            # Set in store
            success = await self.store.set(key, value, env, metadata)
            
            if success:
                # Record change
                await self._record_change(key, old_value, value, changed_by, reason)
                
                # Invalidate cache
                cache_key = f"{env.value}:{key}"
                with self.cache_lock:
                    self.cache.pop(cache_key, None)
                
                # Notify watchers
                await self._notify_watchers(key, value, old_value)
                
                self.logger.info(f"Configuration updated: {key} = {value}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to set config {key}: {e}")
            return False
    
    async def delete(
        self,
        key: str,
        environment: Optional[ConfigEnvironment] = None,
        changed_by: str = "system",
        reason: str = ""
    ) -> bool:
        """Delete configuration key."""
        env = environment or self.current_environment
        
        try:
            # Get old value for change tracking
            old_value = await self.get(key, environment=env, use_cache=False)
            
            if old_value is None:
                return False
            
            # Delete from store
            success = await self.store.delete(key, env)
            
            if success:
                # Record change
                await self._record_change(key, old_value, None, changed_by, reason, "delete")
                
                # Invalidate cache
                cache_key = f"{env.value}:{key}"
                with self.cache_lock:
                    self.cache.pop(cache_key, None)
                
                # Notify watchers
                await self._notify_watchers(key, None, old_value)
                
                self.logger.info(f"Configuration deleted: {key}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete config {key}: {e}")
            return False
    
    async def list_keys(
        self,
        prefix: str = "",
        environment: Optional[ConfigEnvironment] = None
    ) -> List[str]:
        """List configuration keys."""
        try:
            return await self.store.list_keys(prefix, environment)
        except Exception as e:
            self.logger.error(f"Failed to list keys: {e}")
            return []
    
    async def get_all(self, environment: Optional[ConfigEnvironment] = None) -> Dict[str, Any]:
        """Get all configuration values."""
        env = environment or self.current_environment
        result = {}
        
        try:
            keys = await self.list_keys(environment=env)
            
            for key in keys:
                # Remove environment prefix if present
                clean_key = key.split(':', 1)[-1]
                value = await self.get(clean_key, environment=env)
                result[clean_key] = value
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get all configs: {e}")
            return {}
    
    def watch(self, key_pattern: str, callback: Callable[[str, Any, Any], None]):
        """Watch for configuration changes."""
        with self.watcher_lock:
            if key_pattern not in self.watchers:
                self.watchers[key_pattern] = []
            self.watchers[key_pattern].append(callback)
        
        self.logger.debug(f"Added watcher for pattern: {key_pattern}")
    
    def unwatch(self, key_pattern: str, callback: Callable[[str, Any, Any], None]):
        """Remove configuration watcher."""
        with self.watcher_lock:
            if key_pattern in self.watchers:
                try:
                    self.watchers[key_pattern].remove(callback)
                    if not self.watchers[key_pattern]:
                        del self.watchers[key_pattern]
                except ValueError:
                    pass
    
    async def _notify_watchers(self, key: str, new_value: Any, old_value: Any):
        """Notify registered watchers of configuration changes."""
        import fnmatch
        
        with self.watcher_lock:
            for pattern, callbacks in self.watchers.items():
                if fnmatch.fnmatch(key, pattern):
                    for callback in callbacks.copy():  # Copy to avoid modification during iteration
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(key, new_value, old_value)
                            else:
                                callback(key, new_value, old_value)
                        except Exception as e:
                            self.logger.error(f"Watcher callback failed: {e}")
    
    async def _record_change(
        self,
        key: str,
        old_value: Any,
        new_value: Any,
        changed_by: str,
        reason: str,
        change_type: str = "update"
    ):
        """Record configuration change."""
        change = ConfigChange(
            change_id=str(time.time()),
            key=key,
            old_value=old_value,
            new_value=new_value,
            changed_by=changed_by,
            timestamp=datetime.utcnow(),
            change_type=change_type,
            reason=reason
        )
        
        with self.change_lock:
            self.change_history.append(change)
            
            # Keep only last 1000 changes
            if len(self.change_history) > 1000:
                self.change_history = self.change_history[-1000:]
    
    def get_change_history(
        self,
        key: Optional[str] = None,
        limit: int = 100
    ) -> List[ConfigChange]:
        """Get configuration change history."""
        with self.change_lock:
            changes = self.change_history.copy()
        
        # Filter by key if specified
        if key:
            changes = [c for c in changes if c.key == key]
        
        # Sort by timestamp (newest first) and limit
        changes.sort(key=lambda c: c.timestamp, reverse=True)
        return changes[:limit]
    
    async def reload_file(self, file_path: Path):
        """Reload configuration from a specific file."""
        if isinstance(self.store, FileConfigurationStore):
            # Clear cache for this environment
            env_str = file_path.stem  # filename without extension
            try:
                env = ConfigEnvironment(env_str)
                
                with self.cache_lock:
                    # Remove all cached entries for this environment
                    keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"{env_str}:")]
                    for key in keys_to_remove:
                        del self.cache[key]
                
                self.logger.info(f"Reloaded configuration from {file_path}")
                
            except ValueError:
                self.logger.warning(f"Cannot determine environment from file: {file_path}")
    
    async def load_file(self, file_path: Path):
        """Load configuration from a new file."""
        await self.reload_file(file_path)
    
    async def start_file_watching(self, paths: List[Path]):
        """Start watching configuration files for changes."""
        self.file_watcher.start_watching(paths)
    
    async def stop_file_watching(self):
        """Stop watching configuration files."""
        self.file_watcher.stop_watching()
    
    async def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configuration values."""
        errors = {}
        
        try:
            all_configs = await self.get_all()
            
            for key, value in all_configs.items():
                if not self.validator.validate(key, value):
                    if key not in errors:
                        errors[key] = []
                    errors[key].append(f"Validation failed for value: {value}")
            
        except Exception as e:
            self.logger.error(f"Failed to validate all configs: {e}")
        
        return errors
    
    async def get_metadata(self, key: str, environment: Optional[ConfigEnvironment] = None) -> Optional[ConfigMetadata]:
        """Get configuration metadata."""
        env = environment or self.current_environment
        return await self.store.get_metadata(key, env)
    
    async def export_config(
        self,
        format_type: ConfigFormat = ConfigFormat.YAML,
        environment: Optional[ConfigEnvironment] = None,
        include_metadata: bool = False
    ) -> str:
        """Export configuration in specified format."""
        env = environment or self.current_environment
        config_data = await self.get_all(env)
        
        if include_metadata:
            # Add metadata for each key
            for key in config_data.keys():
                metadata = await self.get_metadata(key, env)
                if metadata:
                    config_data[f"_metadata_{key}"] = asdict(metadata)
        
        if format_type == ConfigFormat.JSON:
            return json.dumps(config_data, indent=2, sort_keys=True, default=str)
        elif format_type == ConfigFormat.YAML:
            return yaml.dump(config_data, default_flow_style=False, sort_keys=True)
        elif format_type == ConfigFormat.TOML:
            import toml
            return toml.dumps(config_data)
        else:
            raise ConfigException(f"Unsupported export format: {format_type}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration service statistics."""
        with self.cache_lock:
            cache_size = len(self.cache)
        
        with self.change_lock:
            total_changes = len(self.change_history)
        
        with self.watcher_lock:
            total_watchers = sum(len(callbacks) for callbacks in self.watchers.values())
        
        return {
            'current_environment': self.current_environment.value,
            'cache_size': cache_size,
            'total_changes': total_changes,
            'total_watchers': total_watchers,
            'store_type': type(self.store).__name__,
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
            'last_reload': getattr(self, '_last_reload', None)
        }


# Global configuration service instance
_config_service: Optional[ConfigurationService] = None


async def initialize_config_service(config: Dict[str, Any]) -> ConfigurationService:
    """Initialize global configuration service."""
    global _config_service
    
    _config_service = ConfigurationService(config)
    _config_service._start_time = time.time()
    
    # Start file watching if configured
    watch_paths = config.get('watch_paths', [])
    if watch_paths:
        paths = [Path(p) for p in watch_paths]
        await _config_service.start_file_watching(paths)
    
    return _config_service


def get_config_service() -> Optional[ConfigurationService]:
    """Get global configuration service instance."""
    return _config_service


async def shutdown_config_service():
    """Shutdown global configuration service."""
    global _config_service
    if _config_service:
        await _config_service.stop_file_watching()
        _config_service = None


# Convenience functions for accessing configuration
async def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value using global service."""
    service = get_config_service()
    if service:
        return await service.get(key, default)
    return default


async def set_config(key: str, value: Any) -> bool:
    """Set configuration value using global service."""
    service = get_config_service()
    if service:
        return await service.set(key, value)
    return False


def watch_config(key_pattern: str, callback: Callable[[str, Any, Any], None]):
    """Watch configuration changes using global service."""
    service = get_config_service()
    if service:
        service.watch(key_pattern, callback)