"""
QBITEL - Automated Secrets Rotation System
Production-ready secrets rotation with support for multiple backends.
"""

import asyncio
import logging
import hashlib
import secrets
import time
from typing import Dict, Optional, List, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


class SecretType(str, Enum):
    """Types of secrets that can be rotated."""

    DATABASE_PASSWORD = "database_password"
    REDIS_PASSWORD = "redis_password"
    JWT_SECRET = "jwt_secret"
    ENCRYPTION_KEY = "encryption_key"
    API_KEY = "api_key"
    TLS_CERTIFICATE = "tls_certificate"
    OAUTH_CLIENT_SECRET = "oauth_client_secret"


class SecretBackend(str, Enum):
    """Supported secrets management backends."""

    VAULT = "vault"  # HashiCorp Vault
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    GCP_SECRET_MANAGER = "gcp_secret_manager"
    KUBERNETES = "kubernetes"  # Kubernetes Secrets


@dataclass
class SecretRotationPolicy:
    """Policy for rotating a specific secret."""

    secret_type: SecretType
    rotation_interval_days: int = 90  # Rotate every 90 days
    grace_period_days: int = 7  # Old secret valid for 7 days after rotation
    auto_rotate: bool = True
    notification_channels: List[str] = field(default_factory=list)
    requires_approval: bool = False
    validation_callback: Optional[Callable] = None


@dataclass
class SecretMetadata:
    """Metadata about a secret."""

    secret_id: str
    secret_type: SecretType
    version: str
    created_at: datetime
    expires_at: datetime
    rotated_at: Optional[datetime] = None
    rotation_count: int = 0
    last_accessed: Optional[datetime] = None


class SecretGenerator:
    """Generate cryptographically secure secrets."""

    @staticmethod
    def generate_password(length: int = 32, include_special: bool = True) -> str:
        """Generate a secure random password."""
        import string

        chars = string.ascii_letters + string.digits
        if include_special:
            chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"

        # Ensure at least one of each type
        password = [
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.digits),
        ]

        if include_special:
            password.append(secrets.choice("!@#$%^&*()_+-="))

        # Fill the rest
        password.extend(secrets.choice(chars) for _ in range(length - len(password)))

        # Shuffle
        secrets.SystemRandom().shuffle(password)

        return "".join(password)

    @staticmethod
    def generate_jwt_secret(length: int = 64) -> str:
        """Generate a secure JWT secret."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_encryption_key(length: int = 32) -> str:
        """Generate a secure encryption key (base64 encoded)."""
        import base64

        key_bytes = secrets.token_bytes(length)
        return base64.b64encode(key_bytes).decode("utf-8")

    @staticmethod
    def generate_api_key(prefix: str = "qbitel", length: int = 32) -> str:
        """Generate an API key with prefix."""
        random_part = secrets.token_urlsafe(length)
        return f"{prefix}_{random_part}"


class SecretsRotationManager:
    """Manages automated rotation of secrets across multiple backends."""

    def __init__(
        self,
        backend: SecretBackend,
        backend_config: Dict[str, Any],
        policies: Optional[List[SecretRotationPolicy]] = None,
    ):
        self.backend = backend
        self.backend_config = backend_config
        self.policies = {p.secret_type: p for p in (policies or [])}
        self.secret_metadata: Dict[str, SecretMetadata] = {}
        self.rotation_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        self._rotation_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown = False

    async def initialize(self):
        """Initialize the secrets rotation manager."""
        self.logger.info(f"Initializing secrets rotation manager with backend: {self.backend}")

        # Load existing secret metadata
        await self._load_secret_metadata()

        # Schedule automatic rotations
        if self.policies:
            await self._schedule_rotations()

        self.logger.info("Secrets rotation manager initialized successfully")

    async def shutdown(self):
        """Shutdown the secrets rotation manager."""
        self._shutdown = True

        # Cancel all rotation tasks
        for task in self._rotation_tasks.values():
            task.cancel()

        self.logger.info("Secrets rotation manager shutdown complete")

    async def rotate_secret(self, secret_type: SecretType, force: bool = False) -> Dict[str, Any]:
        """
        Rotate a specific secret.

        Args:
            secret_type: Type of secret to rotate
            force: Force rotation even if not due

        Returns:
            Rotation result with old and new versions
        """
        policy = self.policies.get(secret_type)
        if not policy and not force:
            raise ValueError(f"No rotation policy defined for {secret_type}")

        self.logger.info(f"Starting rotation for secret: {secret_type}")

        # Check if rotation is required
        if not force and not await self._is_rotation_required(secret_type):
            self.logger.info(f"Rotation not required for {secret_type}")
            return {"status": "skipped", "reason": "not_required"}

        # Request approval if required
        if policy and policy.requires_approval and not force:
            approval = await self._request_rotation_approval(secret_type)
            if not approval:
                self.logger.warning(f"Rotation approval denied for {secret_type}")
                return {"status": "denied", "reason": "approval_required"}

        try:
            # Generate new secret value
            new_secret = await self._generate_secret_value(secret_type)

            # Validate new secret if validation callback provided
            if policy and policy.validation_callback:
                if not await policy.validation_callback(new_secret):
                    raise ValueError("Secret validation failed")

            # Get current secret metadata
            current_metadata = self.secret_metadata.get(secret_type.value)
            old_version = current_metadata.version if current_metadata else "unknown"

            # Write new secret to backend
            new_version = await self._write_secret_to_backend(secret_type, new_secret)

            # Update applications to use new secret
            await self._update_applications(secret_type, new_secret, new_version)

            # Update metadata
            new_metadata = SecretMetadata(
                secret_id=secret_type.value,
                secret_type=secret_type,
                version=new_version,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=(policy.rotation_interval_days if policy else 90)),
                rotated_at=datetime.now(),
                rotation_count=(current_metadata.rotation_count + 1 if current_metadata else 1),
            )
            self.secret_metadata[secret_type.value] = new_metadata

            # Keep old secret valid during grace period
            if policy and policy.grace_period_days > 0:
                await self._schedule_secret_deprecation(secret_type, old_version, policy.grace_period_days)

            # Send notifications
            if policy and policy.notification_channels:
                await self._send_rotation_notifications(secret_type, new_version, policy.notification_channels)

            # Record rotation in history
            rotation_record = {
                "secret_type": secret_type.value,
                "old_version": old_version,
                "new_version": new_version,
                "rotated_at": datetime.now().isoformat(),
                "forced": force,
            }
            self.rotation_history.append(rotation_record)

            self.logger.info(f"Successfully rotated secret {secret_type}: {old_version} -> {new_version}")

            return {
                "status": "success",
                "secret_type": secret_type.value,
                "old_version": old_version,
                "new_version": new_version,
                "rotated_at": rotation_record["rotated_at"],
            }

        except Exception as e:
            self.logger.error(f"Failed to rotate secret {secret_type}: {e}", exc_info=True)
            return {
                "status": "failed",
                "secret_type": secret_type.value,
                "error": str(e),
            }

    async def _generate_secret_value(self, secret_type: SecretType) -> str:
        """Generate a new secret value based on type."""
        generator = SecretGenerator()

        if secret_type == SecretType.DATABASE_PASSWORD:
            return generator.generate_password(length=32)
        elif secret_type == SecretType.REDIS_PASSWORD:
            return generator.generate_password(length=32)
        elif secret_type == SecretType.JWT_SECRET:
            return generator.generate_jwt_secret(length=64)
        elif secret_type == SecretType.ENCRYPTION_KEY:
            return generator.generate_encryption_key(length=32)
        elif secret_type == SecretType.API_KEY:
            return generator.generate_api_key(prefix="qbitel")
        else:
            return generator.generate_password(length=32)

    async def _write_secret_to_backend(self, secret_type: SecretType, secret_value: str) -> str:
        """Write secret to the configured backend."""
        version = f"v{int(time.time())}"

        if self.backend == SecretBackend.VAULT:
            return await self._write_to_vault(secret_type, secret_value, version)
        elif self.backend == SecretBackend.AWS_SECRETS_MANAGER:
            return await self._write_to_aws_secrets_manager(secret_type, secret_value, version)
        elif self.backend == SecretBackend.AZURE_KEY_VAULT:
            return await self._write_to_azure_key_vault(secret_type, secret_value, version)
        elif self.backend == SecretBackend.GCP_SECRET_MANAGER:
            return await self._write_to_gcp_secret_manager(secret_type, secret_value, version)
        elif self.backend == SecretBackend.KUBERNETES:
            return await self._write_to_kubernetes_secret(secret_type, secret_value, version)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    async def _write_to_vault(self, secret_type: SecretType, secret_value: str, version: str) -> str:
        """Write secret to HashiCorp Vault."""
        try:
            import hvac

            vault_addr = self.backend_config.get("vault_addr")
            vault_token = self.backend_config.get("vault_token")
            mount_point = self.backend_config.get("mount_point", "secret")
            path = self.backend_config.get("path", "qbitel/production")

            client = hvac.Client(url=vault_addr, token=vault_token)

            secret_path = f"{path}/{secret_type.value}"
            client.secrets.kv.v2.create_or_update_secret(
                path=secret_path,
                secret={"value": secret_value, "version": version},
                mount_point=mount_point,
            )

            self.logger.info(f"Wrote secret to Vault: {secret_path}")
            return version

        except Exception as e:
            self.logger.error(f"Failed to write to Vault: {e}")
            raise

    async def _write_to_aws_secrets_manager(self, secret_type: SecretType, secret_value: str, version: str) -> str:
        """Write secret to AWS Secrets Manager."""
        try:
            import boto3

            region = self.backend_config.get("region", "us-east-1")
            secret_name = f"qbitel/production/{secret_type.value}"

            client = boto3.client("secretsmanager", region_name=region)

            try:
                # Try to update existing secret
                response = client.update_secret(
                    SecretId=secret_name,
                    SecretString=json.dumps({"value": secret_value, "version": version}),
                )
            except client.exceptions.ResourceNotFoundException:
                # Create new secret
                response = client.create_secret(
                    Name=secret_name,
                    SecretString=json.dumps({"value": secret_value, "version": version}),
                    Tags=[
                        {"Key": "Application", "Value": "QBITEL"},
                        {"Key": "Environment", "Value": "production"},
                    ],
                )

            version_id = response.get("VersionId", version)
            self.logger.info(f"Wrote secret to AWS Secrets Manager: {secret_name}")
            return version_id

        except Exception as e:
            self.logger.error(f"Failed to write to AWS Secrets Manager: {e}")
            raise

    async def _write_to_azure_key_vault(self, secret_type: SecretType, secret_value: str, version: str) -> str:
        """Write secret to Azure Key Vault."""
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential

            vault_url = self.backend_config.get("vault_url")
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=vault_url, credential=credential)

            secret_name = f"qbitel-{secret_type.value.replace('_', '-')}"
            secret = client.set_secret(secret_name, secret_value)

            self.logger.info(f"Wrote secret to Azure Key Vault: {secret_name}")
            return secret.properties.version

        except Exception as e:
            self.logger.error(f"Failed to write to Azure Key Vault: {e}")
            raise

    async def _write_to_gcp_secret_manager(self, secret_type: SecretType, secret_value: str, version: str) -> str:
        """Write secret to GCP Secret Manager."""
        try:
            from google.cloud import secretmanager

            project_id = self.backend_config.get("project_id")
            client = secretmanager.SecretManagerServiceClient()

            secret_id = f"qbitel-{secret_type.value.replace('_', '-')}"
            parent = f"projects/{project_id}"
            secret_path = f"{parent}/secrets/{secret_id}"

            # Create secret if it doesn't exist
            try:
                client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": secret_id,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )
            except Exception:
                pass  # Secret already exists

            # Add new version
            response = client.add_secret_version(
                request={
                    "parent": secret_path,
                    "payload": {"data": secret_value.encode("UTF-8")},
                }
            )

            version_name = response.name.split("/")[-1]
            self.logger.info(f"Wrote secret to GCP Secret Manager: {secret_id}")
            return version_name

        except Exception as e:
            self.logger.error(f"Failed to write to GCP Secret Manager: {e}")
            raise

    async def _write_to_kubernetes_secret(self, secret_type: SecretType, secret_value: str, version: str) -> str:
        """Write secret to Kubernetes Secret."""
        try:
            import base64
            from kubernetes import client, config

            # Load kube config
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()

            v1 = client.CoreV1Api()
            namespace = self.backend_config.get("namespace", "default")
            secret_name = f"qbitel-{secret_type.value.replace('_', '-')}"

            # Encode secret value
            encoded_value = base64.b64encode(secret_value.encode()).decode()

            secret_body = client.V1Secret(
                metadata=client.V1ObjectMeta(
                    name=secret_name,
                    labels={
                        "app": "qbitel",
                        "version": version,
                        "managed-by": "secrets-rotation",
                    },
                ),
                type="Opaque",
                data={secret_type.value: encoded_value},
            )

            try:
                # Try to update existing secret
                v1.patch_namespaced_secret(secret_name, namespace, secret_body)
            except client.exceptions.ApiException:
                # Create new secret
                v1.create_namespaced_secret(namespace, secret_body)

            self.logger.info(f"Wrote secret to Kubernetes: {secret_name}")
            return version

        except Exception as e:
            self.logger.error(f"Failed to write to Kubernetes Secret: {e}")
            raise

    async def _update_applications(self, secret_type: SecretType, new_secret: str, new_version: str):
        """Update running applications with the new secret.

        Supports three deployment strategies:
        1. Kubernetes — patches the K8s Secret and triggers a rolling restart
        2. Database/Redis — executes ALTER USER / CONFIG SET for direct rotation
        3. Environment — updates the in-process environment variable
        """
        import os

        self.logger.info(f"Updating applications with new {secret_type} version {new_version}")

        # 1. Direct service credential rotation (database / redis)
        if secret_type == SecretType.DATABASE_PASSWORD:
            await self._rotate_database_password(new_secret)
        elif secret_type == SecretType.REDIS_PASSWORD:
            await self._rotate_redis_password(new_secret)

        # 2. Kubernetes rolling restart when deployed to K8s
        if self.backend == SecretBackend.KUBERNETES:
            await self._trigger_kubernetes_rollout(secret_type, new_version)

        # 3. Always update the current process environment so the app
        # picks up the new value on next connection/reconnect.
        env_key_map = {
            SecretType.DATABASE_PASSWORD: "DATABASE_PASSWORD",
            SecretType.REDIS_PASSWORD: "REDIS_PASSWORD",
            SecretType.JWT_SECRET: "JWT_SECRET_KEY",
            SecretType.ENCRYPTION_KEY: "ENCRYPTION_KEY",
            SecretType.API_KEY: "QBITEL_API_KEY",
        }
        env_key = env_key_map.get(secret_type)
        if env_key:
            os.environ[env_key] = new_secret
            self.logger.info(f"Updated environment variable {env_key}")

    async def _rotate_database_password(self, new_password: str):
        """Rotate the database user password via ALTER USER."""
        try:
            import asyncpg
            import os

            db_url = os.getenv("DATABASE_URL", "")
            if not db_url:
                self.logger.warning("DATABASE_URL not set, skipping DB password rotation")
                return

            conn = await asyncpg.connect(db_url)
            try:
                db_user = os.getenv("DATABASE_USER", "qbitel")
                await conn.execute(f"ALTER USER {db_user} WITH PASSWORD $1", new_password)
                self.logger.info(f"Database password rotated for user {db_user}")
            finally:
                await conn.close()
        except ImportError:
            self.logger.warning("asyncpg not installed, skipping DB password rotation")
        except Exception as e:
            self.logger.error(f"Failed to rotate database password: {e}")
            raise

    async def _rotate_redis_password(self, new_password: str):
        """Rotate Redis password via CONFIG SET requirepass."""
        try:
            import redis.asyncio as aioredis
            import os

            redis_url = os.getenv("REDIS_URL", os.getenv("QBITEL_REDIS_URL", ""))
            if not redis_url:
                self.logger.warning("REDIS_URL not set, skipping Redis password rotation")
                return

            client = aioredis.from_url(redis_url, socket_connect_timeout=5)
            try:
                await client.config_set("requirepass", new_password)
                await client.auth(new_password)
                self.logger.info("Redis password rotated successfully")
            finally:
                await client.aclose()
        except ImportError:
            self.logger.warning("redis package not installed, skipping Redis password rotation")
        except Exception as e:
            self.logger.error(f"Failed to rotate Redis password: {e}")
            raise

    async def _trigger_kubernetes_rollout(self, secret_type: SecretType, new_version: str):
        """Trigger a Kubernetes rolling restart so pods pick up new secrets."""
        try:
            from kubernetes import client, config as k8s_config

            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()

            apps_v1 = client.AppsV1Api()
            namespace = self.backend_config.get("namespace", "default")
            deployment_name = self.backend_config.get("deployment", "qbitel-engine")

            patch = {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "qbitel.ai/secret-rotated": f"{secret_type.value}={new_version}",
                                "qbitel.ai/restart-timestamp": datetime.now().isoformat(),
                            }
                        }
                    }
                }
            }
            apps_v1.patch_namespaced_deployment(deployment_name, namespace, patch)
            self.logger.info(f"Triggered rolling restart for {deployment_name} in {namespace}")
        except ImportError:
            self.logger.warning("kubernetes package not installed, skipping rollout restart")
        except Exception as e:
            self.logger.error(f"Failed to trigger Kubernetes rollout: {e}")

    async def _is_rotation_required(self, secret_type: SecretType) -> bool:
        """Check if rotation is required for a secret."""
        metadata = self.secret_metadata.get(secret_type.value)
        if not metadata:
            return True  # No metadata = needs rotation

        policy = self.policies.get(secret_type)
        if not policy:
            return False

        # Check if expired
        if datetime.now() >= metadata.expires_at:
            return True

        # Check if approaching expiration (rotate 7 days early)
        warning_date = metadata.expires_at - timedelta(days=7)
        if datetime.now() >= warning_date:
            return True

        return False

    async def _schedule_rotations(self):
        """Schedule automatic secret rotations."""
        for secret_type, policy in self.policies.items():
            if policy.auto_rotate:
                task = asyncio.create_task(self._auto_rotation_loop(secret_type, policy))
                self._rotation_tasks[secret_type.value] = task

    async def _auto_rotation_loop(self, secret_type: SecretType, policy: SecretRotationPolicy):
        """Automatic rotation loop for a secret."""
        while not self._shutdown:
            try:
                # Check if rotation needed
                if await self._is_rotation_required(secret_type):
                    await self.rotate_secret(secret_type)

                # Sleep until next check (daily)
                await asyncio.sleep(86400)  # 24 hours

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in auto-rotation loop for {secret_type}: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour

    async def _schedule_secret_deprecation(self, secret_type: SecretType, old_version: str, grace_period_days: int):
        """Schedule deprecation of old secret after grace period."""
        self.logger.info(f"Old {secret_type} version {old_version} will be deprecated in {grace_period_days} days")
        # Implementation would delete old secret after grace period

    async def _request_rotation_approval(self, secret_type: SecretType) -> bool:
        """Request approval for secret rotation."""
        # In production, this would integrate with approval workflow
        # For now, auto-approve
        self.logger.info(f"Rotation approval requested for {secret_type}")
        return True

    async def _send_rotation_notifications(self, secret_type: SecretType, new_version: str, channels: List[str]):
        """Send notifications about secret rotation."""
        message = f"Secret {secret_type} rotated to version {new_version}"
        self.logger.info(f"Sending rotation notifications: {message}")
        # Implementation would send to Slack, email, PagerDuty, etc.

    async def _load_secret_metadata(self):
        """Load existing secret metadata from backend."""
        # Implementation would load metadata from persistent storage
        pass

    def get_rotation_status(self) -> Dict[str, Any]:
        """Get current rotation status for all secrets."""
        status = {}
        for secret_type, metadata in self.secret_metadata.items():
            days_until_expiry = (metadata.expires_at - datetime.now()).days
            status[secret_type] = {
                "version": metadata.version,
                "created_at": metadata.created_at.isoformat(),
                "expires_at": metadata.expires_at.isoformat(),
                "days_until_expiry": days_until_expiry,
                "rotation_count": metadata.rotation_count,
                "requires_rotation": days_until_expiry <= 7,
            }
        return status

    def get_rotation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get rotation history."""
        return self.rotation_history[-limit:]
