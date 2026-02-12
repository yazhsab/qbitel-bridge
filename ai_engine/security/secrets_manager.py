"""
QBITEL - Enterprise Secrets Management
Provides secure secrets management with support for multiple backends.
"""

import os
import json
import logging
import base64
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import hashlib
import secrets as py_secrets
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SecretBackend(str, Enum):
    """Supported secret backends."""

    ENVIRONMENT = "environment"
    VAULT = "vault"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    FILE = "file"  # Encrypted file storage (for development only)


@dataclass
class SecretMetadata:
    """Metadata for a secret."""

    key: str
    backend: SecretBackend
    created_at: datetime
    last_rotated: Optional[datetime] = None
    rotation_interval_days: Optional[int] = None
    encrypted: bool = True


class SecretValidationError(Exception):
    """Secret validation error."""

    pass


class SecretsManager:
    """
    Enterprise secrets management with multiple backend support.

    Supports:
    - Environment variables (production)
    - HashiCorp Vault
    - AWS Secrets Manager
    - Azure Key Vault
    - Encrypted file storage (development only)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize secrets manager."""
        self.config = config or {}
        self.backend = SecretBackend(self.config.get("backend", "environment"))
        self.cache: Dict[str, Any] = {}
        self.metadata: Dict[str, SecretMetadata] = {}

        # Initialize backend
        self._init_backend()

        logger.info(f"SecretsManager initialized with backend: {self.backend.value}")

    def _init_backend(self):
        """Initialize the configured backend."""
        if self.backend == SecretBackend.VAULT:
            self._init_vault()
        elif self.backend == SecretBackend.AWS_SECRETS_MANAGER:
            self._init_aws()
        elif self.backend == SecretBackend.AZURE_KEY_VAULT:
            self._init_azure()
        elif self.backend == SecretBackend.FILE:
            self._init_file_backend()

    def _init_vault(self):
        """Initialize HashiCorp Vault client."""
        try:
            import hvac

            vault_addr = self.config.get("vault_addr", os.getenv("VAULT_ADDR"))
            vault_token = self.config.get("vault_token", os.getenv("VAULT_TOKEN"))

            if not vault_addr:
                raise SecretValidationError("VAULT_ADDR not configured")

            self.vault_client = hvac.Client(url=vault_addr, token=vault_token)

            if not self.vault_client.is_authenticated():
                raise SecretValidationError("Vault authentication failed")

            logger.info("HashiCorp Vault client initialized")

        except ImportError:
            raise SecretValidationError("hvac package not installed. Install with: pip install hvac")
        except Exception as e:
            raise SecretValidationError(f"Failed to initialize Vault: {e}")

    def _init_aws(self):
        """Initialize AWS Secrets Manager client."""
        try:
            import boto3

            region = self.config.get("aws_region", os.getenv("AWS_REGION", "us-east-1"))
            self.aws_client = boto3.client("secretsmanager", region_name=region)

            logger.info("AWS Secrets Manager client initialized")

        except ImportError:
            raise SecretValidationError("boto3 package not installed. Install with: pip install boto3")
        except Exception as e:
            raise SecretValidationError(f"Failed to initialize AWS Secrets Manager: {e}")

    def _init_azure(self):
        """Initialize Azure Key Vault client."""
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential

            vault_url = self.config.get("azure_vault_url", os.getenv("AZURE_VAULT_URL"))
            if not vault_url:
                raise SecretValidationError("AZURE_VAULT_URL not configured")

            credential = DefaultAzureCredential()
            self.azure_client = SecretClient(vault_url=vault_url, credential=credential)

            logger.info("Azure Key Vault client initialized")

        except ImportError:
            raise SecretValidationError(
                "Azure SDK not installed. Install with: pip install azure-keyvault-secrets azure-identity"
            )
        except Exception as e:
            raise SecretValidationError(f"Failed to initialize Azure Key Vault: {e}")

    def _init_file_backend(self):
        """Initialize encrypted file backend (development only)."""
        logger.warning(
            "File-based secrets storage is for DEVELOPMENT ONLY. "
            "Use Vault, AWS Secrets Manager, or Azure Key Vault in production."
        )

        self.secrets_file = Path(self.config.get("secrets_file", ".secrets.enc"))
        self.encryption_key = self._get_encryption_key()

    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key for file backend."""
        key_file = Path(".secrets.key")

        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()

        # Generate new key
        key = py_secrets.token_bytes(32)
        with open(key_file, "wb") as f:
            f.write(key)

        key_file.chmod(0o600)  # Read/write for owner only
        logger.warning(f"Generated new encryption key: {key_file}")

        return key

    def get_secret(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Get secret value.

        Args:
            key: Secret key
            default: Default value if secret not found

        Returns:
            Secret value or default
        """
        # Check cache first
        if key in self.cache:
            return self.cache[key]

        try:
            value = None

            if self.backend == SecretBackend.ENVIRONMENT:
                value = self._get_from_env(key)
            elif self.backend == SecretBackend.VAULT:
                value = self._get_from_vault(key)
            elif self.backend == SecretBackend.AWS_SECRETS_MANAGER:
                value = self._get_from_aws(key)
            elif self.backend == SecretBackend.AZURE_KEY_VAULT:
                value = self._get_from_azure(key)
            elif self.backend == SecretBackend.FILE:
                value = self._get_from_file(key)

            if value is not None:
                self.cache[key] = value
                return value

            return default

        except Exception as e:
            logger.error(f"Failed to get secret '{key}': {e}")
            return default

    def _get_from_env(self, key: str) -> Optional[str]:
        """Get secret from environment variable."""
        # Try exact key first
        value = os.getenv(key)
        if value:
            return value

        # Try uppercase version
        value = os.getenv(key.upper())
        if value:
            return value

        # Try with QBITEL_AI_ prefix
        value = os.getenv(f"QBITEL_AI_{key.upper()}")
        return value

    def _get_from_vault(self, key: str) -> Optional[str]:
        """Get secret from HashiCorp Vault."""
        try:
            mount_point = self.config.get("vault_mount_point", "secret")
            path = self.config.get("vault_path", "qbitel")

            secret_path = f"{path}/{key}"
            response = self.vault_client.secrets.kv.v2.read_secret_version(path=secret_path, mount_point=mount_point)

            return response["data"]["data"].get("value")

        except Exception as e:
            logger.debug(f"Secret '{key}' not found in Vault: {e}")
            return None

    def _get_from_aws(self, key: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            secret_name = f"qbitel/{key}"
            response = self.aws_client.get_secret_value(SecretId=secret_name)

            if "SecretString" in response:
                return response["SecretString"]
            else:
                return base64.b64decode(response["SecretBinary"]).decode("utf-8")

        except self.aws_client.exceptions.ResourceNotFoundException:
            logger.debug(f"Secret '{key}' not found in AWS Secrets Manager")
            return None
        except Exception as e:
            logger.error(f"Error getting secret from AWS: {e}")
            return None

    def _get_from_azure(self, key: str) -> Optional[str]:
        """Get secret from Azure Key Vault."""
        try:
            secret = self.azure_client.get_secret(key)
            return secret.value

        except Exception as e:
            logger.debug(f"Secret '{key}' not found in Azure Key Vault: {e}")
            return None

    def _get_from_file(self, key: str) -> Optional[str]:
        """Get secret from encrypted file."""
        if not self.secrets_file.exists():
            return None

        try:
            from cryptography.fernet import Fernet

            fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key))

            with open(self.secrets_file, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = fernet.decrypt(encrypted_data)
            secrets_dict = json.loads(decrypted_data)

            return secrets_dict.get(key)

        except Exception as e:
            logger.error(f"Failed to read encrypted secrets file: {e}")
            return None

    def set_secret(self, key: str, value: str, metadata: Optional[SecretMetadata] = None):
        """
        Set secret value.

        Args:
            key: Secret key
            value: Secret value
            metadata: Optional metadata
        """
        try:
            if self.backend == SecretBackend.VAULT:
                self._set_in_vault(key, value)
            elif self.backend == SecretBackend.AWS_SECRETS_MANAGER:
                self._set_in_aws(key, value)
            elif self.backend == SecretBackend.AZURE_KEY_VAULT:
                self._set_in_azure(key, value)
            elif self.backend == SecretBackend.FILE:
                self._set_in_file(key, value)
            else:
                raise SecretValidationError(f"Setting secrets not supported for backend: {self.backend.value}")

            # Update cache
            self.cache[key] = value

            # Store metadata
            if metadata:
                self.metadata[key] = metadata

            logger.info(f"Secret '{key}' set successfully")

        except Exception as e:
            logger.error(f"Failed to set secret '{key}': {e}")
            raise

    def _set_in_vault(self, key: str, value: str):
        """Set secret in HashiCorp Vault."""
        mount_point = self.config.get("vault_mount_point", "secret")
        path = self.config.get("vault_path", "qbitel")

        secret_path = f"{path}/{key}"
        self.vault_client.secrets.kv.v2.create_or_update_secret(
            path=secret_path, secret={"value": value}, mount_point=mount_point
        )

    def _set_in_aws(self, key: str, value: str):
        """Set secret in AWS Secrets Manager."""
        secret_name = f"qbitel/{key}"

        try:
            self.aws_client.update_secret(SecretId=secret_name, SecretString=value)
        except self.aws_client.exceptions.ResourceNotFoundException:
            self.aws_client.create_secret(Name=secret_name, SecretString=value)

    def _set_in_azure(self, key: str, value: str):
        """Set secret in Azure Key Vault."""
        self.azure_client.set_secret(key, value)

    def _set_in_file(self, key: str, value: str):
        """Set secret in encrypted file."""
        from cryptography.fernet import Fernet

        # Load existing secrets
        secrets_dict = {}
        if self.secrets_file.exists():
            try:
                fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key))
                with open(self.secrets_file, "rb") as f:
                    encrypted_data = f.read()
                decrypted_data = fernet.decrypt(encrypted_data)
                secrets_dict = json.loads(decrypted_data)
            except Exception as e:
                logger.warning(f"Could not load existing secrets: {e}")

        # Update secret
        secrets_dict[key] = value

        # Encrypt and save
        fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key))
        encrypted_data = fernet.encrypt(json.dumps(secrets_dict).encode())

        with open(self.secrets_file, "wb") as f:
            f.write(encrypted_data)

        self.secrets_file.chmod(0o600)

    def delete_secret(self, key: str):
        """Delete secret."""
        try:
            if self.backend == SecretBackend.VAULT:
                self._delete_from_vault(key)
            elif self.backend == SecretBackend.AWS_SECRETS_MANAGER:
                self._delete_from_aws(key)
            elif self.backend == SecretBackend.AZURE_KEY_VAULT:
                self._delete_from_azure(key)
            elif self.backend == SecretBackend.FILE:
                self._delete_from_file(key)

            # Remove from cache
            self.cache.pop(key, None)
            self.metadata.pop(key, None)

            logger.info(f"Secret '{key}' deleted successfully")

        except Exception as e:
            logger.error(f"Failed to delete secret '{key}': {e}")
            raise

    def _delete_from_vault(self, key: str):
        """Delete secret from Vault."""
        mount_point = self.config.get("vault_mount_point", "secret")
        path = self.config.get("vault_path", "qbitel")
        secret_path = f"{path}/{key}"

        self.vault_client.secrets.kv.v2.delete_metadata_and_all_versions(path=secret_path, mount_point=mount_point)

    def _delete_from_aws(self, key: str):
        """Delete secret from AWS."""
        secret_name = f"qbitel/{key}"
        self.aws_client.delete_secret(SecretId=secret_name, ForceDeleteWithoutRecovery=True)

    def _delete_from_azure(self, key: str):
        """Delete secret from Azure."""
        self.azure_client.begin_delete_secret(key).wait()

    def _delete_from_file(self, key: str):
        """Delete secret from file."""
        if not self.secrets_file.exists():
            return

        from cryptography.fernet import Fernet

        fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key))

        with open(self.secrets_file, "rb") as f:
            encrypted_data = f.read()

        decrypted_data = fernet.decrypt(encrypted_data)
        secrets_dict = json.loads(decrypted_data)

        secrets_dict.pop(key, None)

        encrypted_data = fernet.encrypt(json.dumps(secrets_dict).encode())

        with open(self.secrets_file, "wb") as f:
            f.write(encrypted_data)

    def rotate_secret(self, key: str, new_value: str):
        """
        Rotate a secret value.

        Args:
            key: Secret key
            new_value: New secret value
        """
        metadata = self.metadata.get(key)
        if metadata:
            metadata.last_rotated = datetime.utcnow()
        else:
            metadata = SecretMetadata(
                key=key,
                backend=self.backend,
                created_at=datetime.utcnow(),
                last_rotated=datetime.utcnow(),
            )

        self.set_secret(key, new_value, metadata)
        logger.info(f"Secret '{key}' rotated successfully")

    def validate_secret(self, key: str, value: str) -> bool:
        """
        Validate secret value meets security requirements.

        Args:
            key: Secret key
            value: Secret value to validate

        Returns:
            True if valid, False otherwise
        """
        if not value or len(value) < 8:
            logger.error(f"Secret '{key}' too short (minimum 8 characters)")
            return False

        # Check for common weak patterns
        weak_patterns = ["password", "123456", "admin", "test", "demo"]
        if any(pattern in value.lower() for pattern in weak_patterns):
            logger.error(f"Secret '{key}' contains weak pattern")
            return False

        return True

    def list_secrets(self) -> List[str]:
        """List all secret keys."""
        if self.backend == SecretBackend.FILE:
            if not self.secrets_file.exists():
                return []

            from cryptography.fernet import Fernet

            fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key))

            with open(self.secrets_file, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = fernet.decrypt(encrypted_data)
            secrets_dict = json.loads(decrypted_data)

            return list(secrets_dict.keys())

        # For other backends, return cached keys
        return list(self.cache.keys())


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager(config: Optional[Dict[str, Any]] = None) -> SecretsManager:
    """Get global secrets manager instance."""
    global _secrets_manager

    if _secrets_manager is None:
        _secrets_manager = SecretsManager(config)

    return _secrets_manager


def get_secret(key: str, default: Optional[Any] = None) -> Optional[Any]:
    """Convenience function to get a secret."""
    manager = get_secrets_manager()
    return manager.get_secret(key, default)


def set_secret(key: str, value: str):
    """Convenience function to set a secret."""
    manager = get_secrets_manager()
    manager.set_secret(key, value)
