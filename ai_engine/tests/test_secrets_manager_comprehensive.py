"""
Comprehensive Unit Tests for security/secrets_manager.py
Ensures 100% code coverage for secrets management.
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from ai_engine.security.secrets_manager import (
    SecretBackend,
    SecretMetadata,
    SecretValidationError,
    SecretsManager,
    get_secrets_manager,
    get_secret,
    set_secret,
)


class TestSecretBackend:
    """Test SecretBackend enumeration."""

    def test_backend_values(self):
        """Test backend enum values."""
        assert SecretBackend.ENVIRONMENT.value == "environment"
        assert SecretBackend.VAULT.value == "vault"
        assert SecretBackend.AWS_SECRETS_MANAGER.value == "aws_secrets_manager"
        assert SecretBackend.AZURE_KEY_VAULT.value == "azure_key_vault"
        assert SecretBackend.FILE.value == "file"


class TestSecretMetadata:
    """Test SecretMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating secret metadata."""
        now = datetime.utcnow()
        metadata = SecretMetadata(
            key="test_key",
            backend=SecretBackend.ENVIRONMENT,
            created_at=now,
            last_rotated=now,
            rotation_interval_days=90,
            encrypted=True,
        )
        assert metadata.key == "test_key"
        assert metadata.backend == SecretBackend.ENVIRONMENT
        assert metadata.created_at == now
        assert metadata.last_rotated == now
        assert metadata.rotation_interval_days == 90
        assert metadata.encrypted is True

    def test_metadata_defaults(self):
        """Test metadata default values."""
        now = datetime.utcnow()
        metadata = SecretMetadata(
            key="test_key", backend=SecretBackend.ENVIRONMENT, created_at=now
        )
        assert metadata.last_rotated is None
        assert metadata.rotation_interval_days is None
        assert metadata.encrypted is True


class TestSecretValidationError:
    """Test SecretValidationError exception."""

    def test_exception_creation(self):
        """Test creating SecretValidationError."""
        exc = SecretValidationError("Test error")
        assert str(exc) == "Test error"


class TestSecretsManagerEnvironment:
    """Test SecretsManager with environment backend."""

    @pytest.fixture
    def manager(self):
        """Create SecretsManager with environment backend."""
        return SecretsManager({"backend": "environment"})

    def test_init_environment_backend(self, manager):
        """Test initialization with environment backend."""
        assert manager.backend == SecretBackend.ENVIRONMENT
        assert isinstance(manager.cache, dict)
        assert isinstance(manager.metadata, dict)

    def test_get_secret_from_env_exact_key(self, manager):
        """Test getting secret with exact key match."""
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            value = manager.get_secret("TEST_KEY")
            assert value == "test_value"

    def test_get_secret_from_env_uppercase(self, manager):
        """Test getting secret with uppercase conversion."""
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            value = manager.get_secret("test_key")
            assert value == "test_value"

    def test_get_secret_from_env_with_prefix(self, manager):
        """Test getting secret with CRONOS_AI prefix."""
        with patch.dict(os.environ, {"CRONOS_AI_TEST_KEY": "test_value"}):
            value = manager.get_secret("test_key")
            assert value == "test_value"

    def test_get_secret_not_found_returns_default(self, manager):
        """Test getting non-existent secret returns default."""
        value = manager.get_secret("NONEXISTENT_KEY", default="default_value")
        assert value == "default_value"

    def test_get_secret_caching(self, manager):
        """Test secret caching."""
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            value1 = manager.get_secret("TEST_KEY")
            # Remove from env to test cache
            os.environ.pop("TEST_KEY", None)
            value2 = manager.get_secret("TEST_KEY")
            assert value1 == value2 == "test_value"

    def test_set_secret_not_supported_for_environment(self, manager):
        """Test that setting secrets is not supported for environment backend."""
        with pytest.raises(
            SecretValidationError, match="Setting secrets not supported"
        ):
            manager.set_secret("test_key", "test_value")


class TestSecretsManagerVault:
    """Test SecretsManager with Vault backend."""

    @pytest.fixture
    def mock_vault_client(self):
        """Create mock Vault client."""
        mock_client = Mock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets = Mock()
        mock_client.secrets.kv = Mock()
        mock_client.secrets.kv.v2 = Mock()
        return mock_client

    def test_init_vault_backend_success(self, mock_vault_client):
        """Test successful Vault initialization."""
        with patch.dict("sys.modules", {"hvac": MagicMock()}):
            import sys

            sys.modules["hvac"].Client = Mock(return_value=mock_vault_client)
            with patch.dict(
                os.environ,
                {"VAULT_ADDR": "http://localhost:8200", "VAULT_TOKEN": "test_token"},
            ):
                manager = SecretsManager({"backend": "vault"})
                assert manager.backend == SecretBackend.VAULT
                assert manager.vault_client is not None

    def test_init_vault_missing_addr(self):
        """Test Vault initialization without VAULT_ADDR."""
        with patch.dict("sys.modules", {"hvac": MagicMock()}):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(
                    SecretValidationError, match="VAULT_ADDR not configured"
                ):
                    SecretsManager({"backend": "vault"})

    def test_init_vault_not_authenticated(self):
        """Test Vault initialization with failed authentication."""
        mock_client = Mock()
        mock_client.is_authenticated.return_value = False
        with patch.dict("sys.modules", {"hvac": MagicMock()}):
            import sys

            sys.modules["hvac"].Client = Mock(return_value=mock_client)
            with patch.dict(os.environ, {"VAULT_ADDR": "http://localhost:8200"}):
                with pytest.raises(
                    SecretValidationError, match="Vault authentication failed"
                ):
                    SecretsManager({"backend": "vault"})

    def test_init_vault_import_error(self):
        """Test Vault initialization with missing hvac package."""
        with patch.dict("sys.modules", {"hvac": None}):
            with pytest.raises(
                SecretValidationError, match="hvac package not installed"
            ):
                SecretsManager({"backend": "vault"})

    def test_get_secret_from_vault(self, mock_vault_client):
        """Test getting secret from Vault."""
        mock_vault_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"value": "vault_secret"}}
        }
        with patch(
            "ai_engine.security.secrets_manager.hvac.Client",
            return_value=mock_vault_client,
        ):
            with patch.dict(
                os.environ,
                {"VAULT_ADDR": "http://localhost:8200", "VAULT_TOKEN": "test_token"},
            ):
                manager = SecretsManager({"backend": "vault"})
                value = manager.get_secret("test_key")
                assert value == "vault_secret"

    def test_get_secret_from_vault_not_found(self, mock_vault_client):
        """Test getting non-existent secret from Vault."""
        mock_vault_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
            "Not found"
        )
        with patch(
            "ai_engine.security.secrets_manager.hvac.Client",
            return_value=mock_vault_client,
        ):
            with patch.dict(
                os.environ,
                {"VAULT_ADDR": "http://localhost:8200", "VAULT_TOKEN": "test_token"},
            ):
                manager = SecretsManager({"backend": "vault"})
                value = manager.get_secret("test_key", default="default")
                assert value == "default"

    def test_set_secret_in_vault(self, mock_vault_client):
        """Test setting secret in Vault."""
        with patch(
            "ai_engine.security.secrets_manager.hvac.Client",
            return_value=mock_vault_client,
        ):
            with patch.dict(
                os.environ,
                {"VAULT_ADDR": "http://localhost:8200", "VAULT_TOKEN": "test_token"},
            ):
                manager = SecretsManager({"backend": "vault"})
                manager.set_secret("test_key", "test_value")
                mock_vault_client.secrets.kv.v2.create_or_update_secret.assert_called_once()

    def test_delete_secret_from_vault(self, mock_vault_client):
        """Test deleting secret from Vault."""
        with patch(
            "ai_engine.security.secrets_manager.hvac.Client",
            return_value=mock_vault_client,
        ):
            with patch.dict(
                os.environ,
                {"VAULT_ADDR": "http://localhost:8200", "VAULT_TOKEN": "test_token"},
            ):
                manager = SecretsManager({"backend": "vault"})
                manager.delete_secret("test_key")
                mock_vault_client.secrets.kv.v2.delete_metadata_and_all_versions.assert_called_once()


class TestSecretsManagerAWS:
    """Test SecretsManager with AWS Secrets Manager backend."""

    @pytest.fixture
    def mock_aws_client(self):
        """Create mock AWS client."""
        mock_client = Mock()
        mock_client.exceptions = Mock()
        mock_client.exceptions.ResourceNotFoundException = type(
            "ResourceNotFoundException", (Exception,), {}
        )
        return mock_client

    def test_init_aws_backend_success(self, mock_aws_client):
        """Test successful AWS Secrets Manager initialization."""
        with patch(
            "ai_engine.security.secrets_manager.boto3.client",
            return_value=mock_aws_client,
        ):
            manager = SecretsManager(
                {"backend": "aws_secrets_manager", "aws_region": "us-east-1"}
            )
            assert manager.backend == SecretBackend.AWS_SECRETS_MANAGER
            assert manager.aws_client is not None

    def test_init_aws_import_error(self):
        """Test AWS initialization with missing boto3 package."""
        with patch("ai_engine.security.secrets_manager.boto3", side_effect=ImportError):
            with pytest.raises(
                SecretValidationError, match="boto3 package not installed"
            ):
                SecretsManager({"backend": "aws_secrets_manager"})

    def test_get_secret_from_aws_string(self, mock_aws_client):
        """Test getting secret from AWS (SecretString)."""
        mock_aws_client.get_secret_value.return_value = {"SecretString": "aws_secret"}
        with patch(
            "ai_engine.security.secrets_manager.boto3.client",
            return_value=mock_aws_client,
        ):
            manager = SecretsManager({"backend": "aws_secrets_manager"})
            value = manager.get_secret("test_key")
            assert value == "aws_secret"

    def test_get_secret_from_aws_binary(self, mock_aws_client):
        """Test getting secret from AWS (SecretBinary)."""
        import base64

        mock_aws_client.get_secret_value.return_value = {
            "SecretBinary": base64.b64encode(b"aws_secret")
        }
        with patch(
            "ai_engine.security.secrets_manager.boto3.client",
            return_value=mock_aws_client,
        ):
            manager = SecretsManager({"backend": "aws_secrets_manager"})
            value = manager.get_secret("test_key")
            assert value == "aws_secret"

    def test_get_secret_from_aws_not_found(self, mock_aws_client):
        """Test getting non-existent secret from AWS."""
        mock_aws_client.get_secret_value.side_effect = (
            mock_aws_client.exceptions.ResourceNotFoundException()
        )
        with patch(
            "ai_engine.security.secrets_manager.boto3.client",
            return_value=mock_aws_client,
        ):
            manager = SecretsManager({"backend": "aws_secrets_manager"})
            value = manager.get_secret("test_key", default="default")
            assert value == "default"

    def test_set_secret_in_aws_update(self, mock_aws_client):
        """Test updating existing secret in AWS."""
        with patch(
            "ai_engine.security.secrets_manager.boto3.client",
            return_value=mock_aws_client,
        ):
            manager = SecretsManager({"backend": "aws_secrets_manager"})
            manager.set_secret("test_key", "test_value")
            mock_aws_client.update_secret.assert_called_once()

    def test_set_secret_in_aws_create(self, mock_aws_client):
        """Test creating new secret in AWS."""
        mock_aws_client.update_secret.side_effect = (
            mock_aws_client.exceptions.ResourceNotFoundException()
        )
        with patch(
            "ai_engine.security.secrets_manager.boto3.client",
            return_value=mock_aws_client,
        ):
            manager = SecretsManager({"backend": "aws_secrets_manager"})
            manager.set_secret("test_key", "test_value")
            mock_aws_client.create_secret.assert_called_once()

    def test_delete_secret_from_aws(self, mock_aws_client):
        """Test deleting secret from AWS."""
        with patch(
            "ai_engine.security.secrets_manager.boto3.client",
            return_value=mock_aws_client,
        ):
            manager = SecretsManager({"backend": "aws_secrets_manager"})
            manager.delete_secret("test_key")
            mock_aws_client.delete_secret.assert_called_once()


class TestSecretsManagerAzure:
    """Test SecretsManager with Azure Key Vault backend."""

    @pytest.fixture
    def mock_azure_client(self):
        """Create mock Azure client."""
        return Mock()

    def test_init_azure_backend_success(self, mock_azure_client):
        """Test successful Azure Key Vault initialization."""
        with patch(
            "ai_engine.security.secrets_manager.SecretClient",
            return_value=mock_azure_client,
        ):
            with patch("ai_engine.security.secrets_manager.DefaultAzureCredential"):
                with patch.dict(
                    os.environ, {"AZURE_VAULT_URL": "https://test.vault.azure.net"}
                ):
                    manager = SecretsManager({"backend": "azure_key_vault"})
                    assert manager.backend == SecretBackend.AZURE_KEY_VAULT
                    assert manager.azure_client is not None

    def test_init_azure_missing_url(self):
        """Test Azure initialization without AZURE_VAULT_URL."""
        with patch("ai_engine.security.secrets_manager.SecretClient"):
            with patch("ai_engine.security.secrets_manager.DefaultAzureCredential"):
                with patch.dict(os.environ, {}, clear=True):
                    with pytest.raises(
                        SecretValidationError, match="AZURE_VAULT_URL not configured"
                    ):
                        SecretsManager({"backend": "azure_key_vault"})

    def test_init_azure_import_error(self):
        """Test Azure initialization with missing SDK."""
        with patch(
            "ai_engine.security.secrets_manager.SecretClient", side_effect=ImportError
        ):
            with pytest.raises(SecretValidationError, match="Azure SDK not installed"):
                SecretsManager({"backend": "azure_key_vault"})

    def test_get_secret_from_azure(self, mock_azure_client):
        """Test getting secret from Azure."""
        mock_secret = Mock()
        mock_secret.value = "azure_secret"
        mock_azure_client.get_secret.return_value = mock_secret
        with patch(
            "ai_engine.security.secrets_manager.SecretClient",
            return_value=mock_azure_client,
        ):
            with patch("ai_engine.security.secrets_manager.DefaultAzureCredential"):
                with patch.dict(
                    os.environ, {"AZURE_VAULT_URL": "https://test.vault.azure.net"}
                ):
                    manager = SecretsManager({"backend": "azure_key_vault"})
                    value = manager.get_secret("test_key")
                    assert value == "azure_secret"

    def test_get_secret_from_azure_not_found(self, mock_azure_client):
        """Test getting non-existent secret from Azure."""
        mock_azure_client.get_secret.side_effect = Exception("Not found")
        with patch(
            "ai_engine.security.secrets_manager.SecretClient",
            return_value=mock_azure_client,
        ):
            with patch("ai_engine.security.secrets_manager.DefaultAzureCredential"):
                with patch.dict(
                    os.environ, {"AZURE_VAULT_URL": "https://test.vault.azure.net"}
                ):
                    manager = SecretsManager({"backend": "azure_key_vault"})
                    value = manager.get_secret("test_key", default="default")
                    assert value == "default"

    def test_set_secret_in_azure(self, mock_azure_client):
        """Test setting secret in Azure."""
        with patch(
            "ai_engine.security.secrets_manager.SecretClient",
            return_value=mock_azure_client,
        ):
            with patch("ai_engine.security.secrets_manager.DefaultAzureCredential"):
                with patch.dict(
                    os.environ, {"AZURE_VAULT_URL": "https://test.vault.azure.net"}
                ):
                    manager = SecretsManager({"backend": "azure_key_vault"})
                    manager.set_secret("test_key", "test_value")
                    mock_azure_client.set_secret.assert_called_once()

    def test_delete_secret_from_azure(self, mock_azure_client):
        """Test deleting secret from Azure."""
        mock_poller = Mock()
        mock_poller.wait = Mock()
        mock_azure_client.begin_delete_secret.return_value = mock_poller
        with patch(
            "ai_engine.security.secrets_manager.SecretClient",
            return_value=mock_azure_client,
        ):
            with patch("ai_engine.security.secrets_manager.DefaultAzureCredential"):
                with patch.dict(
                    os.environ, {"AZURE_VAULT_URL": "https://test.vault.azure.net"}
                ):
                    manager = SecretsManager({"backend": "azure_key_vault"})
                    manager.delete_secret("test_key")
                    mock_azure_client.begin_delete_secret.assert_called_once()


class TestSecretsManagerFile:
    """Test SecretsManager with file backend."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_init_file_backend(self, temp_dir):
        """Test file backend initialization."""
        secrets_file = Path(temp_dir) / ".secrets.enc"
        manager = SecretsManager({"backend": "file", "secrets_file": str(secrets_file)})
        assert manager.backend == SecretBackend.FILE
        assert manager.secrets_file == secrets_file

    def test_get_encryption_key_existing(self, temp_dir):
        """Test getting existing encryption key."""
        key_file = Path(temp_dir) / ".secrets.key"
        test_key = b"test_encryption_key_32_bytes_!!"
        key_file.write_bytes(test_key)

        manager = SecretsManager(
            {"backend": "file", "secrets_file": str(Path(temp_dir) / ".secrets.enc")}
        )
        # Change to temp_dir to use the key file
        with patch(
            "ai_engine.security.secrets_manager.Path", return_value=key_file.parent
        ):
            key = manager._get_encryption_key()
            assert key == test_key

    def test_get_encryption_key_generate_new(self, temp_dir):
        """Test generating new encryption key."""
        with patch(
            "ai_engine.security.secrets_manager.Path.cwd", return_value=Path(temp_dir)
        ):
            manager = SecretsManager({"backend": "file"})
            key_file = Path(temp_dir) / ".secrets.key"
            assert key_file.exists()
            assert len(key_file.read_bytes()) == 32

    def test_set_and_get_secret_from_file(self, temp_dir):
        """Test setting and getting secret from file."""
        with patch(
            "ai_engine.security.secrets_manager.Path.cwd", return_value=Path(temp_dir)
        ):
            manager = SecretsManager({"backend": "file"})
            manager.set_secret("test_key", "test_value")
            value = manager.get_secret("test_key")
            assert value == "test_value"

    def test_get_secret_from_file_not_found(self, temp_dir):
        """Test getting non-existent secret from file."""
        with patch(
            "ai_engine.security.secrets_manager.Path.cwd", return_value=Path(temp_dir)
        ):
            manager = SecretsManager({"backend": "file"})
            value = manager.get_secret("nonexistent", default="default")
            assert value == "default"

    def test_get_secret_from_file_no_file(self, temp_dir):
        """Test getting secret when file doesn't exist."""
        secrets_file = Path(temp_dir) / ".secrets.enc"
        manager = SecretsManager({"backend": "file", "secrets_file": str(secrets_file)})
        value = manager.get_secret("test_key")
        assert value is None

    def test_delete_secret_from_file(self, temp_dir):
        """Test deleting secret from file."""
        with patch(
            "ai_engine.security.secrets_manager.Path.cwd", return_value=Path(temp_dir)
        ):
            manager = SecretsManager({"backend": "file"})
            manager.set_secret("test_key", "test_value")
            manager.delete_secret("test_key")
            value = manager.get_secret("test_key")
            assert value is None

    def test_delete_secret_from_file_no_file(self, temp_dir):
        """Test deleting secret when file doesn't exist."""
        secrets_file = Path(temp_dir) / ".secrets.enc"
        manager = SecretsManager({"backend": "file", "secrets_file": str(secrets_file)})
        # Should not raise error
        manager.delete_secret("test_key")

    def test_list_secrets_from_file(self, temp_dir):
        """Test listing secrets from file."""
        with patch(
            "ai_engine.security.secrets_manager.Path.cwd", return_value=Path(temp_dir)
        ):
            manager = SecretsManager({"backend": "file"})
            manager.set_secret("key1", "value1")
            manager.set_secret("key2", "value2")
            secrets = manager.list_secrets()
            assert "key1" in secrets
            assert "key2" in secrets

    def test_list_secrets_no_file(self, temp_dir):
        """Test listing secrets when file doesn't exist."""
        secrets_file = Path(temp_dir) / ".secrets.enc"
        manager = SecretsManager({"backend": "file", "secrets_file": str(secrets_file)})
        secrets = manager.list_secrets()
        assert secrets == []


class TestSecretsManagerRotation:
    """Test secret rotation functionality."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create SecretsManager with file backend."""
        with patch(
            "ai_engine.security.secrets_manager.Path.cwd", return_value=Path(temp_dir)
        ):
            return SecretsManager({"backend": "file"})

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_rotate_secret_new_metadata(self, manager):
        """Test rotating secret with new metadata."""
        manager.set_secret("test_key", "old_value")
        manager.rotate_secret("test_key", "new_value")

        value = manager.get_secret("test_key")
        assert value == "new_value"
        assert "test_key" in manager.metadata
        assert manager.metadata["test_key"].last_rotated is not None

    def test_rotate_secret_existing_metadata(self, manager):
        """Test rotating secret with existing metadata."""
        metadata = SecretMetadata(
            key="test_key",
            backend=SecretBackend.FILE,
            created_at=datetime.utcnow(),
        )
        manager.set_secret("test_key", "old_value", metadata)

        old_rotated = manager.metadata["test_key"].last_rotated
        manager.rotate_secret("test_key", "new_value")

        assert manager.get_secret("test_key") == "new_value"
        assert manager.metadata["test_key"].last_rotated != old_rotated


class TestSecretsManagerValidation:
    """Test secret validation functionality."""

    @pytest.fixture
    def manager(self):
        """Create SecretsManager."""
        return SecretsManager({"backend": "environment"})

    def test_validate_secret_too_short(self, manager):
        """Test validation fails for short secrets."""
        assert manager.validate_secret("key", "short") is False

    def test_validate_secret_weak_pattern_password(self, manager):
        """Test validation fails for weak pattern 'password'."""
        assert manager.validate_secret("key", "password123") is False

    def test_validate_secret_weak_pattern_123456(self, manager):
        """Test validation fails for weak pattern '123456'."""
        assert manager.validate_secret("key", "123456789") is False

    def test_validate_secret_weak_pattern_admin(self, manager):
        """Test validation fails for weak pattern 'admin'."""
        assert manager.validate_secret("key", "admin12345") is False

    def test_validate_secret_weak_pattern_test(self, manager):
        """Test validation fails for weak pattern 'test'."""
        assert manager.validate_secret("key", "test123456") is False

    def test_validate_secret_weak_pattern_demo(self, manager):
        """Test validation fails for weak pattern 'demo'."""
        assert manager.validate_secret("key", "demo123456") is False

    def test_validate_secret_strong(self, manager):
        """Test validation passes for strong secret."""
        assert manager.validate_secret("key", "Str0ng!P@ssw0rd") is True

    def test_validate_secret_empty(self, manager):
        """Test validation fails for empty secret."""
        assert manager.validate_secret("key", "") is False


class TestSecretsManagerListSecrets:
    """Test listing secrets functionality."""

    def test_list_secrets_environment_backend(self):
        """Test listing secrets returns cached keys for environment backend."""
        manager = SecretsManager({"backend": "environment"})
        with patch.dict(os.environ, {"KEY1": "value1", "KEY2": "value2"}):
            manager.get_secret("KEY1")
            manager.get_secret("KEY2")
            secrets = manager.list_secrets()
            assert "KEY1" in secrets
            assert "KEY2" in secrets


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_secrets_manager_singleton(self):
        """Test get_secrets_manager returns singleton."""
        with patch("ai_engine.security.secrets_manager._secrets_manager", None):
            manager1 = get_secrets_manager()
            manager2 = get_secrets_manager()
            assert manager1 is manager2

    def test_get_secrets_manager_with_config(self):
        """Test get_secrets_manager with custom config."""
        with patch("ai_engine.security.secrets_manager._secrets_manager", None):
            config = {"backend": "environment"}
            manager = get_secrets_manager(config)
            assert manager.backend == SecretBackend.ENVIRONMENT

    def test_get_secret_convenience_function(self):
        """Test get_secret convenience function."""
        with patch(
            "ai_engine.security.secrets_manager.get_secrets_manager"
        ) as mock_get:
            mock_manager = Mock()
            mock_manager.get_secret.return_value = "test_value"
            mock_get.return_value = mock_manager

            value = get_secret("test_key", default="default")
            assert value == "test_value"
            mock_manager.get_secret.assert_called_once_with("test_key", "default")

    def test_set_secret_convenience_function(self):
        """Test set_secret convenience function."""
        with patch(
            "ai_engine.security.secrets_manager.get_secrets_manager"
        ) as mock_get:
            mock_manager = Mock()
            mock_get.return_value = mock_manager

            set_secret("test_key", "test_value")
            mock_manager.set_secret.assert_called_once_with("test_key", "test_value")


class TestSecretsManagerErrorHandling:
    """Test error handling in SecretsManager."""

    def test_get_secret_exception_returns_default(self):
        """Test that exceptions during get_secret return default."""
        manager = SecretsManager({"backend": "environment"})
        with patch.object(
            manager, "_get_from_env", side_effect=Exception("Test error")
        ):
            value = manager.get_secret("test_key", default="default")
            assert value == "default"

    def test_set_secret_exception_raises(self):
        """Test that exceptions during set_secret are raised."""
        mock_client = Mock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.kv.v2.create_or_update_secret.side_effect = Exception(
            "Test error"
        )

        with patch(
            "ai_engine.security.secrets_manager.hvac.Client", return_value=mock_client
        ):
            with patch.dict(
                os.environ,
                {"VAULT_ADDR": "http://localhost:8200", "VAULT_TOKEN": "test"},
            ):
                manager = SecretsManager({"backend": "vault"})
                with pytest.raises(Exception):
                    manager.set_secret("test_key", "test_value")

    def test_delete_secret_exception_raises(self):
        """Test that exceptions during delete_secret are raised."""
        mock_client = Mock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.kv.v2.delete_metadata_and_all_versions.side_effect = (
            Exception("Test error")
        )

        with patch(
            "ai_engine.security.secrets_manager.hvac.Client", return_value=mock_client
        ):
            with patch.dict(
                os.environ,
                {"VAULT_ADDR": "http://localhost:8200", "VAULT_TOKEN": "test"},
            ):
                manager = SecretsManager({"backend": "vault"})
                with pytest.raises(Exception):
                    manager.delete_secret("test_key")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
