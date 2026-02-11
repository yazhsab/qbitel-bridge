"""
Configuration Manager

Self-configuring security infrastructure management.
Automatically configures and maintains security settings.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import uuid
import json
import os


logger = logging.getLogger(__name__)


class ConfigurationType(Enum):
    """Types of configuration."""

    SECURITY = "security"
    HSM = "hsm"
    PQC = "pqc"
    NETWORK = "network"
    COMPLIANCE = "compliance"
    MONITORING = "monitoring"


class ConfigurationSource(Enum):
    """Source of configuration."""

    FILE = "file"
    ENVIRONMENT = "environment"
    VAULT = "vault"
    HSM = "hsm"
    API = "api"
    AUTO_DISCOVERED = "auto_discovered"


@dataclass
class ConfigurationChange:
    """Record of a configuration change."""

    change_id: str
    config_key: str
    old_value: Any
    new_value: Any
    changed_at: datetime
    changed_by: str
    reason: str
    rollback_value: Any = None


@dataclass
class SecurityConfiguration:
    """Security configuration settings."""

    # PQC Settings
    pqc_enabled: bool = True
    pqc_algorithms: List[str] = field(default_factory=lambda: ["ML-KEM-768", "ML-DSA-65"])
    hybrid_mode: bool = True  # Classical + PQC

    # HSM Settings
    hsm_provider: str = "auto"  # auto, aws, azure, gcp, softhsm
    hsm_failover_enabled: bool = True
    hsm_connection_pool_size: int = 5

    # Encryption
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90

    # TLS
    min_tls_version: str = "1.3"
    cipher_suites: List[str] = field(default_factory=lambda: [
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
    ])

    # Compliance
    compliance_frameworks: List[str] = field(default_factory=lambda: [
        "PCI-DSS", "SWIFT-CSP"
    ])
    audit_logging: bool = True

    # Monitoring
    alert_thresholds: Dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """
    Self-configuring security infrastructure manager.

    Capabilities:
    - Auto-discovery of optimal settings
    - Configuration versioning
    - Secure storage integration
    - Environment-aware configuration
    - Compliance validation
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        environment: str = "production",
    ):
        self._config_path = config_path
        self._environment = environment
        self._current_config = SecurityConfiguration()
        self._change_history: List[ConfigurationChange] = []
        self._config_sources: Dict[str, ConfigurationSource] = {}
        self._validators: Dict[str, Callable] = {}

        # Load configuration
        self._load_configuration()

    def get_config(self) -> SecurityConfiguration:
        """Get current security configuration."""
        return self._current_config

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return getattr(self._current_config, key, default)

    def set_value(
        self,
        key: str,
        value: Any,
        reason: str = "Manual update",
        changed_by: str = "system",
    ) -> bool:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: New value
            reason: Reason for change
            changed_by: Who made the change

        Returns:
            True if successful
        """
        if not hasattr(self._current_config, key):
            logger.error(f"Unknown configuration key: {key}")
            return False

        # Validate if validator exists
        if key in self._validators:
            if not self._validators[key](value):
                logger.error(f"Validation failed for {key}")
                return False

        # Record change
        old_value = getattr(self._current_config, key)
        change = ConfigurationChange(
            change_id=str(uuid.uuid4()),
            config_key=key,
            old_value=old_value,
            new_value=value,
            changed_at=datetime.utcnow(),
            changed_by=changed_by,
            reason=reason,
            rollback_value=old_value,
        )
        self._change_history.append(change)

        # Apply change
        setattr(self._current_config, key, value)

        # Persist
        self._save_configuration()

        logger.info(f"Configuration updated: {key} = {value}")
        return True

    def auto_configure(
        self,
        system_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Auto-configure based on system analysis.

        Args:
            system_analysis: Analysis from Intelligence Engine

        Returns:
            Applied configuration changes
        """
        changes = {}

        # PQC configuration based on security assessment
        if system_analysis.get("pqc_ready") is False:
            self.set_value("pqc_enabled", True, "Auto-enabled for PQC migration")
            self.set_value("hybrid_mode", True, "Hybrid mode for transition")
            changes["pqc"] = {"enabled": True, "hybrid": True}

        # HSM provider selection based on environment
        env_provider = self._detect_cloud_provider()
        if env_provider and env_provider != "unknown":
            self.set_value("hsm_provider", env_provider, f"Auto-detected {env_provider}")
            changes["hsm_provider"] = env_provider

        # Compliance frameworks based on protocol
        protocol_type = system_analysis.get("protocol_type", "")
        if "swift" in protocol_type.lower():
            frameworks = self._current_config.compliance_frameworks
            if "SWIFT-CSP" not in frameworks:
                frameworks.append("SWIFT-CSP")
                self.set_value("compliance_frameworks", frameworks, "Auto-added for SWIFT")
                changes["compliance_frameworks"] = frameworks

        if "emv" in protocol_type.lower() or "3ds" in protocol_type.lower():
            frameworks = self._current_config.compliance_frameworks
            if "PCI-DSS" not in frameworks:
                frameworks.append("PCI-DSS")
                self.set_value("compliance_frameworks", frameworks, "Auto-added for card protocols")
                changes["compliance_frameworks"] = frameworks

        # Key rotation based on security score
        security_score = system_analysis.get("security_score", 100)
        if security_score < 70:
            self.set_value("key_rotation_days", 30, "Shortened due to low security score")
            changes["key_rotation_days"] = 30

        return changes

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate current configuration.

        Returns:
            Validation result with any issues
        """
        issues = []
        warnings = []

        # Validate PQC settings
        if self._current_config.pqc_enabled:
            valid_algos = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024",
                          "ML-DSA-44", "ML-DSA-65", "ML-DSA-87"]
            for algo in self._current_config.pqc_algorithms:
                if algo not in valid_algos:
                    issues.append(f"Invalid PQC algorithm: {algo}")

        # Validate TLS
        if self._current_config.min_tls_version not in ["1.2", "1.3"]:
            issues.append(f"Invalid TLS version: {self._current_config.min_tls_version}")

        if self._current_config.min_tls_version == "1.2":
            warnings.append("TLS 1.2 is acceptable but TLS 1.3 is recommended")

        # Validate key rotation
        if self._current_config.key_rotation_days > 365:
            warnings.append("Key rotation period exceeds recommended maximum (365 days)")

        if self._current_config.key_rotation_days < 7:
            warnings.append("Key rotation period very short, may impact performance")

        # Validate HSM settings
        if self._current_config.hsm_connection_pool_size < 1:
            issues.append("HSM connection pool size must be at least 1")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }

    def export_config(self, format: str = "json") -> str:
        """Export configuration."""
        config_dict = {
            "pqc_enabled": self._current_config.pqc_enabled,
            "pqc_algorithms": self._current_config.pqc_algorithms,
            "hybrid_mode": self._current_config.hybrid_mode,
            "hsm_provider": self._current_config.hsm_provider,
            "hsm_failover_enabled": self._current_config.hsm_failover_enabled,
            "encryption_algorithm": self._current_config.encryption_algorithm,
            "key_rotation_days": self._current_config.key_rotation_days,
            "min_tls_version": self._current_config.min_tls_version,
            "cipher_suites": self._current_config.cipher_suites,
            "compliance_frameworks": self._current_config.compliance_frameworks,
            "audit_logging": self._current_config.audit_logging,
            "environment": self._environment,
            "exported_at": datetime.utcnow().isoformat(),
        }

        if format == "json":
            return json.dumps(config_dict, indent=2)
        else:
            return str(config_dict)

    def import_config(self, config_data: str, format: str = "json") -> bool:
        """Import configuration."""
        try:
            if format == "json":
                data = json.loads(config_data)
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Apply values
            for key, value in data.items():
                if hasattr(self._current_config, key):
                    self.set_value(key, value, "Imported configuration")

            return True

        except Exception as e:
            logger.error(f"Configuration import failed: {e}")
            return False

    def rollback_change(self, change_id: str) -> bool:
        """Rollback a specific configuration change."""
        for change in reversed(self._change_history):
            if change.change_id == change_id:
                if change.rollback_value is not None:
                    setattr(self._current_config, change.config_key, change.rollback_value)
                    self._save_configuration()
                    logger.info(f"Rolled back change {change_id}")
                    return True

        return False

    def get_change_history(
        self,
        limit: int = 100,
    ) -> List[ConfigurationChange]:
        """Get configuration change history."""
        return self._change_history[-limit:]

    def _load_configuration(self) -> None:
        """Load configuration from sources."""
        # Load from environment variables first
        self._load_from_environment()

        # Load from file if specified
        if self._config_path and os.path.exists(self._config_path):
            self._load_from_file()

        logger.info("Configuration loaded")

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            "QBITEL_PQC_ENABLED": ("pqc_enabled", lambda x: x.lower() == "true"),
            "QBITEL_HSM_PROVIDER": ("hsm_provider", str),
            "QBITEL_MIN_TLS_VERSION": ("min_tls_version", str),
            "QBITEL_KEY_ROTATION_DAYS": ("key_rotation_days", int),
            "QBITEL_AUDIT_LOGGING": ("audit_logging", lambda x: x.lower() == "true"),
        }

        for env_var, (config_key, converter) in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    setattr(self._current_config, config_key, converter(value))
                    self._config_sources[config_key] = ConfigurationSource.ENVIRONMENT
                except Exception as e:
                    logger.warning(f"Failed to load {env_var}: {e}")

    def _load_from_file(self) -> None:
        """Load configuration from file."""
        try:
            with open(self._config_path, 'r') as f:
                data = json.load(f)

            for key, value in data.items():
                if hasattr(self._current_config, key):
                    setattr(self._current_config, key, value)
                    self._config_sources[key] = ConfigurationSource.FILE

        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")

    def _save_configuration(self) -> None:
        """Save configuration to file."""
        if not self._config_path:
            return

        try:
            config_data = json.loads(self.export_config())

            with open(self._config_path, 'w') as f:
                json.dump(config_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def _detect_cloud_provider(self) -> str:
        """Detect cloud provider from environment."""
        # AWS
        if os.environ.get("AWS_REGION") or os.environ.get("AWS_EXECUTION_ENV"):
            return "aws"

        # Azure
        if os.environ.get("AZURE_TENANT_ID") or os.environ.get("WEBSITE_INSTANCE_ID"):
            return "azure"

        # GCP
        if os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT"):
            return "gcp"

        return "unknown"
