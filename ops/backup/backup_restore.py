#!/usr/bin/env python3
"""
QBITEL — Backup and Restore Procedures
==========================================

This module provides comprehensive backup and restore functionality for:
- PostgreSQL/TimescaleDB databases
- Redis cache
- Configuration files
- Model artifacts
- Compliance evidence

Requirements:
    pip install boto3 google-cloud-storage azure-storage-blob psycopg2-binary redis

Usage:
    # Create backup
    python backup_restore.py backup --type full --destination s3://qbitel-backups/

    # Restore from backup
    python backup_restore.py restore --backup-id 20250203-120000 --source s3://qbitel-backups/

    # List backups
    python backup_restore.py list --source s3://qbitel-backups/

    # Verify backup integrity
    python backup_restore.py verify --backup-id 20250203-120000 --source s3://qbitel-backups/
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('qbitel-backup')


class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupDestination(Enum):
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    LOCAL = "local"


class BackupComponent(Enum):
    DATABASE = "database"
    REDIS = "redis"
    CONFIG = "config"
    MODELS = "models"
    EVIDENCE = "evidence"
    ALL = "all"


@dataclass
class BackupConfig:
    """Configuration for backup operations."""
    # Database settings
    db_host: str = os.environ.get("QBITEL_DB_HOST", "localhost")
    db_port: int = int(os.environ.get("QBITEL_DB_PORT", "5432"))
    db_name: str = os.environ.get("QBITEL_DB_NAME", "qbitel")
    db_user: str = os.environ.get("QBITEL_DB_USER", "qbitel")
    db_password: str = os.environ.get("QBITEL_DB_PASSWORD", "")

    # Redis settings
    redis_host: str = os.environ.get("QBITEL_REDIS_HOST", "localhost")
    redis_port: int = int(os.environ.get("QBITEL_REDIS_PORT", "6379"))
    redis_password: str = os.environ.get("QBITEL_REDIS_PASSWORD", "")

    # Storage settings
    s3_bucket: str = os.environ.get("QBITEL_BACKUP_S3_BUCKET", "")
    s3_region: str = os.environ.get("QBITEL_BACKUP_S3_REGION", "us-east-1")
    gcs_bucket: str = os.environ.get("QBITEL_BACKUP_GCS_BUCKET", "")
    azure_container: str = os.environ.get("QBITEL_BACKUP_AZURE_CONTAINER", "")
    azure_connection_string: str = os.environ.get("QBITEL_BACKUP_AZURE_CONNECTION", "")
    local_path: str = os.environ.get("QBITEL_BACKUP_LOCAL_PATH", "/var/backups/qbitel")

    # Paths
    config_path: str = os.environ.get("QBITEL_CONFIG_PATH", "/etc/qbitel")
    models_path: str = os.environ.get("QBITEL_MODELS_PATH", "/var/lib/qbitel/models")
    evidence_path: str = os.environ.get("QBITEL_EVIDENCE_PATH", "/var/lib/qbitel/evidence")

    # Retention
    retention_days: int = int(os.environ.get("QBITEL_BACKUP_RETENTION_DAYS", "90"))
    retention_count: int = int(os.environ.get("QBITEL_BACKUP_RETENTION_COUNT", "30"))

    # Encryption
    encryption_key: str = os.environ.get("QBITEL_BACKUP_ENCRYPTION_KEY", "")
    encrypt_backups: bool = os.environ.get("QBITEL_BACKUP_ENCRYPT", "true").lower() == "true"

    # Parallelism
    parallel_uploads: int = int(os.environ.get("QBITEL_BACKUP_PARALLEL_UPLOADS", "4"))

    # RTO/RPO targets
    rto_hours: int = 4  # Recovery Time Objective: 4 hours
    rpo_hours: int = 1  # Recovery Point Objective: 1 hour (backup frequency)


@dataclass
class BackupManifest:
    """Manifest describing a backup."""
    backup_id: str
    backup_type: BackupType
    created_at: datetime
    completed_at: Optional[datetime] = None
    components: List[str] = field(default_factory=list)
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    database_version: Optional[str] = None
    app_version: Optional[str] = None
    total_size_bytes: int = 0
    checksum: Optional[str] = None
    encrypted: bool = False
    compression: str = "gzip"
    status: str = "in_progress"
    error_message: Optional[str] = None
    parent_backup_id: Optional[str] = None  # For incremental backups

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "components": self.components,
            "files": self.files,
            "database_version": self.database_version,
            "app_version": self.app_version,
            "total_size_bytes": self.total_size_bytes,
            "checksum": self.checksum,
            "encrypted": self.encrypted,
            "compression": self.compression,
            "status": self.status,
            "error_message": self.error_message,
            "parent_backup_id": self.parent_backup_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupManifest":
        return cls(
            backup_id=data["backup_id"],
            backup_type=BackupType(data["backup_type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            components=data.get("components", []),
            files=data.get("files", {}),
            database_version=data.get("database_version"),
            app_version=data.get("app_version"),
            total_size_bytes=data.get("total_size_bytes", 0),
            checksum=data.get("checksum"),
            encrypted=data.get("encrypted", False),
            compression=data.get("compression", "gzip"),
            status=data.get("status", "unknown"),
            error_message=data.get("error_message"),
            parent_backup_id=data.get("parent_backup_id"),
        )


class StorageBackend:
    """Base class for storage backends."""

    def upload(self, local_path: str, remote_path: str) -> bool:
        raise NotImplementedError

    def download(self, remote_path: str, local_path: str) -> bool:
        raise NotImplementedError

    def list_files(self, prefix: str) -> List[str]:
        raise NotImplementedError

    def delete(self, remote_path: str) -> bool:
        raise NotImplementedError

    def exists(self, remote_path: str) -> bool:
        raise NotImplementedError


class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(self, bucket: str, region: str = "us-east-1"):
        import boto3
        self.bucket = bucket
        self.s3 = boto3.client('s3', region_name=region)

    def upload(self, local_path: str, remote_path: str) -> bool:
        try:
            self.s3.upload_file(local_path, self.bucket, remote_path)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False

    def download(self, remote_path: str, local_path: str) -> bool:
        try:
            self.s3.download_file(self.bucket, remote_path, local_path)
            logger.info(f"Downloaded s3://{self.bucket}/{remote_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return False

    def list_files(self, prefix: str) -> List[str]:
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return []

    def delete(self, remote_path: str) -> bool:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=remote_path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete from S3: {e}")
            return False

    def exists(self, remote_path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except Exception:
            return False


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def upload(self, local_path: str, remote_path: str) -> bool:
        try:
            dest = self.base_path / remote_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(local_path, dest)
            logger.info(f"Copied {local_path} to {dest}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            return False

    def download(self, remote_path: str, local_path: str) -> bool:
        try:
            src = self.base_path / remote_path
            import shutil
            shutil.copy2(src, local_path)
            logger.info(f"Copied {src} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            return False

    def list_files(self, prefix: str) -> List[str]:
        try:
            search_path = self.base_path / prefix
            if search_path.is_dir():
                return [str(p.relative_to(self.base_path)) for p in search_path.rglob('*') if p.is_file()]
            return []
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def delete(self, remote_path: str) -> bool:
        try:
            path = self.base_path / remote_path
            if path.is_file():
                path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False

    def exists(self, remote_path: str) -> bool:
        return (self.base_path / remote_path).exists()


class BackupManager:
    """Main backup and restore manager."""

    def __init__(self, config: BackupConfig, storage: StorageBackend):
        self.config = config
        self.storage = storage

    def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        components: List[BackupComponent] = None
    ) -> BackupManifest:
        """Create a new backup."""
        if components is None:
            components = [BackupComponent.ALL]

        backup_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type=backup_type,
            created_at=datetime.utcnow(),
            encrypted=self.config.encrypt_backups,
        )

        logger.info(f"Starting {backup_type.value} backup: {backup_id}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Determine which components to backup
                if BackupComponent.ALL in components:
                    components = [
                        BackupComponent.DATABASE,
                        BackupComponent.REDIS,
                        BackupComponent.CONFIG,
                        BackupComponent.MODELS,
                        BackupComponent.EVIDENCE,
                    ]

                # Backup each component
                for component in components:
                    logger.info(f"Backing up component: {component.value}")
                    manifest.components.append(component.value)

                    if component == BackupComponent.DATABASE:
                        self._backup_database(temp_path, manifest)
                    elif component == BackupComponent.REDIS:
                        self._backup_redis(temp_path, manifest)
                    elif component == BackupComponent.CONFIG:
                        self._backup_config(temp_path, manifest)
                    elif component == BackupComponent.MODELS:
                        self._backup_models(temp_path, manifest)
                    elif component == BackupComponent.EVIDENCE:
                        self._backup_evidence(temp_path, manifest)

                # Create final archive
                archive_name = f"qbitel-backup-{backup_id}.tar.gz"
                archive_path = temp_path / archive_name

                logger.info(f"Creating archive: {archive_path}")
                with tarfile.open(archive_path, "w:gz") as tar:
                    for item in temp_path.iterdir():
                        if item.name != archive_name:
                            tar.add(item, arcname=item.name)

                # Calculate checksum
                manifest.checksum = self._calculate_checksum(archive_path)
                manifest.total_size_bytes = archive_path.stat().st_size

                # Save manifest
                manifest_path = temp_path / "manifest.json"
                with open(manifest_path, 'w') as f:
                    json.dump(manifest.to_dict(), f, indent=2)

                # Upload to storage
                remote_prefix = f"backups/{backup_id}"
                self.storage.upload(str(archive_path), f"{remote_prefix}/{archive_name}")
                self.storage.upload(str(manifest_path), f"{remote_prefix}/manifest.json")

                manifest.completed_at = datetime.utcnow()
                manifest.status = "completed"

                logger.info(f"Backup completed: {backup_id} ({manifest.total_size_bytes / 1024 / 1024:.2f} MB)")

            except Exception as e:
                manifest.status = "failed"
                manifest.error_message = str(e)
                logger.error(f"Backup failed: {e}")
                raise

        return manifest

    def _backup_database(self, temp_path: Path, manifest: BackupManifest):
        """Backup PostgreSQL/TimescaleDB database."""
        dump_file = temp_path / "database.sql"

        # Use pg_dump for PostgreSQL
        env = os.environ.copy()
        env['PGPASSWORD'] = self.config.db_password

        cmd = [
            "pg_dump",
            "-h", self.config.db_host,
            "-p", str(self.config.db_port),
            "-U", self.config.db_user,
            "-d", self.config.db_name,
            "-F", "c",  # Custom format for parallel restore
            "-f", str(dump_file),
            "--no-owner",
            "--no-acl",
        ]

        logger.info(f"Running pg_dump: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"pg_dump failed: {result.stderr}")

        # Get database version
        version_cmd = ["psql", "-h", self.config.db_host, "-p", str(self.config.db_port),
                       "-U", self.config.db_user, "-d", self.config.db_name,
                       "-t", "-c", "SELECT version();"]
        version_result = subprocess.run(version_cmd, env=env, capture_output=True, text=True)
        if version_result.returncode == 0:
            manifest.database_version = version_result.stdout.strip()

        manifest.files["database.sql"] = {
            "size": dump_file.stat().st_size,
            "checksum": self._calculate_checksum(dump_file),
        }

    def _backup_redis(self, temp_path: Path, manifest: BackupManifest):
        """Backup Redis data."""
        import redis

        rdb_file = temp_path / "redis-dump.rdb"

        try:
            r = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password or None,
            )

            # Trigger BGSAVE and wait for completion
            r.bgsave()

            # Wait for save to complete
            import time
            while r.lastsave() == r.lastsave():
                time.sleep(0.1)

            # Get RDB file location from Redis config
            redis_dir = r.config_get('dir')['dir']
            redis_dbfile = r.config_get('dbfilename')['dbfilename']
            source_rdb = Path(redis_dir) / redis_dbfile

            if source_rdb.exists():
                import shutil
                shutil.copy2(source_rdb, rdb_file)
                manifest.files["redis-dump.rdb"] = {
                    "size": rdb_file.stat().st_size,
                    "checksum": self._calculate_checksum(rdb_file),
                }
            else:
                logger.warning("Redis RDB file not found, skipping Redis backup")

        except Exception as e:
            logger.warning(f"Redis backup failed (non-critical): {e}")

    def _backup_config(self, temp_path: Path, manifest: BackupManifest):
        """Backup configuration files."""
        config_path = Path(self.config.config_path)
        if not config_path.exists():
            logger.warning(f"Config path {config_path} does not exist, skipping")
            return

        config_archive = temp_path / "config.tar.gz"
        with tarfile.open(config_archive, "w:gz") as tar:
            tar.add(config_path, arcname="config")

        manifest.files["config.tar.gz"] = {
            "size": config_archive.stat().st_size,
            "checksum": self._calculate_checksum(config_archive),
        }

    def _backup_models(self, temp_path: Path, manifest: BackupManifest):
        """Backup ML model artifacts."""
        models_path = Path(self.config.models_path)
        if not models_path.exists():
            logger.warning(f"Models path {models_path} does not exist, skipping")
            return

        models_archive = temp_path / "models.tar.gz"
        with tarfile.open(models_archive, "w:gz") as tar:
            tar.add(models_path, arcname="models")

        manifest.files["models.tar.gz"] = {
            "size": models_archive.stat().st_size,
            "checksum": self._calculate_checksum(models_archive),
        }

    def _backup_evidence(self, temp_path: Path, manifest: BackupManifest):
        """Backup compliance evidence files."""
        evidence_path = Path(self.config.evidence_path)
        if not evidence_path.exists():
            logger.warning(f"Evidence path {evidence_path} does not exist, skipping")
            return

        evidence_archive = temp_path / "evidence.tar.gz"
        with tarfile.open(evidence_archive, "w:gz") as tar:
            tar.add(evidence_path, arcname="evidence")

        manifest.files["evidence.tar.gz"] = {
            "size": evidence_archive.stat().st_size,
            "checksum": self._calculate_checksum(evidence_archive),
        }

    def restore_backup(self, backup_id: str, components: List[BackupComponent] = None) -> bool:
        """Restore from a backup."""
        if components is None:
            components = [BackupComponent.ALL]

        logger.info(f"Starting restore from backup: {backup_id}")

        # Download manifest
        remote_prefix = f"backups/{backup_id}"
        manifest_content = None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "manifest.json"

            if not self.storage.download(f"{remote_prefix}/manifest.json", str(manifest_path)):
                raise Exception(f"Failed to download manifest for backup {backup_id}")

            with open(manifest_path) as f:
                manifest = BackupManifest.from_dict(json.load(f))

            # Download and extract archive
            archive_name = f"qbitel-backup-{backup_id}.tar.gz"
            archive_path = temp_path / archive_name

            if not self.storage.download(f"{remote_prefix}/{archive_name}", str(archive_path)):
                raise Exception(f"Failed to download backup archive {backup_id}")

            # Verify checksum
            actual_checksum = self._calculate_checksum(archive_path)
            if actual_checksum != manifest.checksum:
                raise Exception(f"Checksum mismatch! Expected {manifest.checksum}, got {actual_checksum}")

            logger.info("Checksum verified, extracting archive")

            # Extract archive
            extract_path = temp_path / "extracted"
            extract_path.mkdir()
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_path)

            # Determine which components to restore
            if BackupComponent.ALL in components:
                components = [BackupComponent(c) for c in manifest.components]

            # Restore each component
            for component in components:
                logger.info(f"Restoring component: {component.value}")

                if component == BackupComponent.DATABASE:
                    self._restore_database(extract_path)
                elif component == BackupComponent.REDIS:
                    self._restore_redis(extract_path)
                elif component == BackupComponent.CONFIG:
                    self._restore_config(extract_path)
                elif component == BackupComponent.MODELS:
                    self._restore_models(extract_path)
                elif component == BackupComponent.EVIDENCE:
                    self._restore_evidence(extract_path)

        logger.info(f"Restore completed successfully from backup: {backup_id}")
        return True

    def _restore_database(self, extract_path: Path):
        """Restore PostgreSQL database."""
        dump_file = extract_path / "database.sql"
        if not dump_file.exists():
            logger.warning("Database dump not found in backup")
            return

        env = os.environ.copy()
        env['PGPASSWORD'] = self.config.db_password

        # Drop and recreate database
        logger.info("Dropping existing database...")
        drop_cmd = [
            "psql", "-h", self.config.db_host, "-p", str(self.config.db_port),
            "-U", self.config.db_user, "-d", "postgres",
            "-c", f"DROP DATABASE IF EXISTS {self.config.db_name};"
        ]
        subprocess.run(drop_cmd, env=env, capture_output=True)

        create_cmd = [
            "psql", "-h", self.config.db_host, "-p", str(self.config.db_port),
            "-U", self.config.db_user, "-d", "postgres",
            "-c", f"CREATE DATABASE {self.config.db_name};"
        ]
        subprocess.run(create_cmd, env=env, capture_output=True)

        # Restore using pg_restore
        logger.info("Restoring database...")
        restore_cmd = [
            "pg_restore",
            "-h", self.config.db_host,
            "-p", str(self.config.db_port),
            "-U", self.config.db_user,
            "-d", self.config.db_name,
            "-j", "4",  # Parallel restore
            str(dump_file),
        ]
        result = subprocess.run(restore_cmd, env=env, capture_output=True, text=True)

        if result.returncode != 0 and "ERROR" in result.stderr:
            logger.warning(f"pg_restore warnings: {result.stderr}")

        logger.info("Database restored successfully")

    def _restore_redis(self, extract_path: Path):
        """Restore Redis data."""
        rdb_file = extract_path / "redis-dump.rdb"
        if not rdb_file.exists():
            logger.warning("Redis dump not found in backup")
            return

        import redis
        r = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            password=self.config.redis_password or None,
        )

        # Get Redis data directory
        redis_dir = r.config_get('dir')['dir']
        redis_dbfile = r.config_get('dbfilename')['dbfilename']
        target_rdb = Path(redis_dir) / redis_dbfile

        # Stop Redis, copy file, restart
        logger.warning("Redis restore requires manual intervention: stop Redis, copy RDB file, restart Redis")
        logger.info(f"Copy {rdb_file} to {target_rdb}")

    def _restore_config(self, extract_path: Path):
        """Restore configuration files."""
        config_archive = extract_path / "config.tar.gz"
        if not config_archive.exists():
            logger.warning("Config archive not found in backup")
            return

        with tarfile.open(config_archive, "r:gz") as tar:
            tar.extractall(Path(self.config.config_path).parent)
        logger.info("Configuration restored")

    def _restore_models(self, extract_path: Path):
        """Restore ML model artifacts."""
        models_archive = extract_path / "models.tar.gz"
        if not models_archive.exists():
            logger.warning("Models archive not found in backup")
            return

        with tarfile.open(models_archive, "r:gz") as tar:
            tar.extractall(Path(self.config.models_path).parent)
        logger.info("Models restored")

    def _restore_evidence(self, extract_path: Path):
        """Restore compliance evidence."""
        evidence_archive = extract_path / "evidence.tar.gz"
        if not evidence_archive.exists():
            logger.warning("Evidence archive not found in backup")
            return

        with tarfile.open(evidence_archive, "r:gz") as tar:
            tar.extractall(Path(self.config.evidence_path).parent)
        logger.info("Evidence restored")

    def list_backups(self) -> List[BackupManifest]:
        """List all available backups."""
        backups = []
        files = self.storage.list_files("backups/")

        manifest_files = [f for f in files if f.endswith("manifest.json")]

        for manifest_file in manifest_files:
            with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
                if self.storage.download(manifest_file, tmp.name):
                    with open(tmp.name) as f:
                        try:
                            manifest = BackupManifest.from_dict(json.load(f))
                            backups.append(manifest)
                        except Exception as e:
                            logger.warning(f"Failed to parse manifest {manifest_file}: {e}")

        return sorted(backups, key=lambda b: b.created_at, reverse=True)

    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity."""
        logger.info(f"Verifying backup: {backup_id}")

        remote_prefix = f"backups/{backup_id}"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download manifest
            manifest_path = temp_path / "manifest.json"
            if not self.storage.download(f"{remote_prefix}/manifest.json", str(manifest_path)):
                logger.error("Failed to download manifest")
                return False

            with open(manifest_path) as f:
                manifest = BackupManifest.from_dict(json.load(f))

            # Download and verify archive
            archive_name = f"qbitel-backup-{backup_id}.tar.gz"
            archive_path = temp_path / archive_name

            if not self.storage.download(f"{remote_prefix}/{archive_name}", str(archive_path)):
                logger.error("Failed to download archive")
                return False

            # Verify checksum
            actual_checksum = self._calculate_checksum(archive_path)
            if actual_checksum != manifest.checksum:
                logger.error(f"Checksum mismatch! Expected {manifest.checksum}, got {actual_checksum}")
                return False

            # Verify archive can be extracted
            try:
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.getmembers()
            except Exception as e:
                logger.error(f"Archive is corrupted: {e}")
                return False

            logger.info(f"Backup {backup_id} verified successfully")
            return True

    def cleanup_old_backups(self) -> int:
        """Remove backups older than retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
        deleted_count = 0

        backups = self.list_backups()

        for backup in backups:
            if backup.created_at < cutoff_date:
                logger.info(f"Deleting old backup: {backup.backup_id}")
                remote_prefix = f"backups/{backup.backup_id}"

                files = self.storage.list_files(remote_prefix)
                for f in files:
                    self.storage.delete(f)

                deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count

    @staticmethod
    def _calculate_checksum(file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="QBITEL Backup and Restore")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a backup")
    backup_parser.add_argument("--type", choices=["full", "incremental"], default="full")
    backup_parser.add_argument("--components", nargs="+",
                               choices=["database", "redis", "config", "models", "evidence", "all"],
                               default=["all"])
    backup_parser.add_argument("--destination", required=True, help="Backup destination (s3://bucket or /local/path)")

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("--backup-id", required=True, help="Backup ID to restore")
    restore_parser.add_argument("--source", required=True, help="Backup source location")
    restore_parser.add_argument("--components", nargs="+",
                                choices=["database", "redis", "config", "models", "evidence", "all"],
                                default=["all"])

    # List command
    list_parser = subparsers.add_parser("list", help="List available backups")
    list_parser.add_argument("--source", required=True, help="Backup source location")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify backup integrity")
    verify_parser.add_argument("--backup-id", required=True, help="Backup ID to verify")
    verify_parser.add_argument("--source", required=True, help="Backup source location")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove old backups")
    cleanup_parser.add_argument("--source", required=True, help="Backup source location")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Determine storage backend
    config = BackupConfig()

    def get_storage(location: str) -> StorageBackend:
        if location.startswith("s3://"):
            bucket = location.replace("s3://", "").split("/")[0]
            return S3StorageBackend(bucket, config.s3_region)
        else:
            return LocalStorageBackend(location)

    if args.command == "backup":
        storage = get_storage(args.destination)
        manager = BackupManager(config, storage)
        backup_type = BackupType.FULL if args.type == "full" else BackupType.INCREMENTAL
        components = [BackupComponent(c) for c in args.components]
        manifest = manager.create_backup(backup_type, components)
        print(f"Backup created: {manifest.backup_id}")
        print(f"Status: {manifest.status}")
        print(f"Size: {manifest.total_size_bytes / 1024 / 1024:.2f} MB")

    elif args.command == "restore":
        storage = get_storage(args.source)
        manager = BackupManager(config, storage)
        components = [BackupComponent(c) for c in args.components]
        success = manager.restore_backup(args.backup_id, components)
        sys.exit(0 if success else 1)

    elif args.command == "list":
        storage = get_storage(args.source)
        manager = BackupManager(config, storage)
        backups = manager.list_backups()
        print(f"{'ID':<20} {'Type':<12} {'Status':<12} {'Size':<12} {'Created':<25}")
        print("-" * 80)
        for b in backups:
            size = f"{b.total_size_bytes / 1024 / 1024:.1f} MB" if b.total_size_bytes else "N/A"
            print(f"{b.backup_id:<20} {b.backup_type.value:<12} {b.status:<12} {size:<12} {b.created_at.isoformat()}")

    elif args.command == "verify":
        storage = get_storage(args.source)
        manager = BackupManager(config, storage)
        success = manager.verify_backup(args.backup_id)
        print(f"Verification: {'PASSED' if success else 'FAILED'}")
        sys.exit(0 if success else 1)

    elif args.command == "cleanup":
        storage = get_storage(args.source)
        manager = BackupManager(config, storage)
        deleted = manager.cleanup_old_backups()
        print(f"Deleted {deleted} old backups")


if __name__ == "__main__":
    main()
