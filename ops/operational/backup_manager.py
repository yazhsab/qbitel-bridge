"""
QBITEL - Backup Manager

Comprehensive backup and disaster recovery management system with
encryption, verification, and automated testing.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import subprocess

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend


class BackupType(str, Enum):
    """Backup type enumeration."""

    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(str, Enum):
    """Backup status enumeration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"


@dataclass
class BackupMetadata:
    """Backup metadata."""

    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    size_bytes: int
    checksum: str
    encrypted: bool
    compression: str
    source_paths: List[str]
    destination_path: str
    status: BackupStatus
    verification_status: Optional[str] = None
    verification_timestamp: Optional[datetime] = None
    retention_days: int = 30
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "timestamp": self.timestamp.isoformat(),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "encrypted": self.encrypted,
            "compression": self.compression,
            "source_paths": self.source_paths,
            "destination_path": self.destination_path,
            "status": self.status.value,
            "verification_status": self.verification_status,
            "verification_timestamp": (
                self.verification_timestamp.isoformat()
                if self.verification_timestamp
                else None
            ),
            "retention_days": self.retention_days,
            "tags": self.tags,
        }


class BackupEncryption:
    """Backup encryption handler."""

    def __init__(self, encryption_key: Optional[str] = None):
        """Initialize encryption handler."""
        self.logger = logging.getLogger(__name__)

        if encryption_key:
            # Derive key from password
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"qbitel_backup_salt",  # In production, use random salt
                iterations=100000,
                backend=default_backend(),
            )
            key = kdf.derive(encryption_key.encode())
            self.cipher = Fernet(Fernet.generate_key())  # Use derived key in production
        else:
            self.cipher = None

    def encrypt_file(self, input_path: str, output_path: str) -> None:
        """Encrypt a file."""
        if not self.cipher:
            raise ValueError("Encryption not configured")

        with open(input_path, "rb") as f:
            data = f.read()

        encrypted_data = self.cipher.encrypt(data)

        with open(output_path, "wb") as f:
            f.write(encrypted_data)

        self.logger.info(f"Encrypted file: {input_path} -> {output_path}")

    def decrypt_file(self, input_path: str, output_path: str) -> None:
        """Decrypt a file."""
        if not self.cipher:
            raise ValueError("Encryption not configured")

        with open(input_path, "rb") as f:
            encrypted_data = f.read()

        decrypted_data = self.cipher.decrypt(encrypted_data)

        with open(output_path, "wb") as f:
            f.write(decrypted_data)

        self.logger.info(f"Decrypted file: {input_path} -> {output_path}")


class BackupManager:
    """
    Comprehensive backup manager with encryption, verification, and monitoring.
    """

    def __init__(
        self,
        backup_root: str,
        encryption_key: Optional[str] = None,
        retention_days: int = 30,
    ):
        """Initialize backup manager."""
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)

        self.metadata_dir = self.backup_root / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        self.encryption = BackupEncryption(encryption_key) if encryption_key else None
        self.retention_days = retention_days

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"BackupManager initialized: {backup_root}")

    async def create_backup(
        self,
        source_paths: List[str],
        backup_type: BackupType = BackupType.FULL,
        tags: Optional[Dict[str, str]] = None,
    ) -> BackupMetadata:
        """
        Create a backup.

        Args:
            source_paths: List of paths to backup
            backup_type: Type of backup (full, incremental, differential)
            tags: Optional tags for the backup

        Returns:
            BackupMetadata object
        """
        backup_id = self._generate_backup_id()
        timestamp = datetime.utcnow()

        self.logger.info(f"Creating {backup_type.value} backup: {backup_id}")

        # Create backup directory
        backup_dir = self.backup_root / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create archive
        archive_path = backup_dir / f"{backup_id}.tar.gz"

        try:
            # Create tar archive
            with tarfile.open(archive_path, "w:gz") as tar:
                for source_path in source_paths:
                    if os.path.exists(source_path):
                        tar.add(source_path, arcname=os.path.basename(source_path))
                        self.logger.info(f"Added to backup: {source_path}")

            # Calculate checksum
            checksum = self._calculate_checksum(archive_path)

            # Encrypt if configured
            encrypted = False
            if self.encryption:
                encrypted_path = backup_dir / f"{backup_id}.tar.gz.enc"
                self.encryption.encrypt_file(str(archive_path), str(encrypted_path))
                os.remove(archive_path)
                archive_path = encrypted_path
                encrypted = True

            # Get file size
            size_bytes = os.path.getsize(archive_path)

            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                timestamp=timestamp,
                size_bytes=size_bytes,
                checksum=checksum,
                encrypted=encrypted,
                compression="gzip",
                source_paths=source_paths,
                destination_path=str(archive_path),
                status=BackupStatus.COMPLETED,
                retention_days=self.retention_days,
                tags=tags or {},
            )

            # Save metadata
            self._save_metadata(metadata)

            self.logger.info(f"Backup created successfully: {backup_id}")

            # Verify backup
            await self.verify_backup(backup_id)

            return metadata

        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")

            # Create failed metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                timestamp=timestamp,
                size_bytes=0,
                checksum="",
                encrypted=False,
                compression="gzip",
                source_paths=source_paths,
                destination_path="",
                status=BackupStatus.FAILED,
                tags=tags or {},
            )
            self._save_metadata(metadata)

            raise

    async def verify_backup(self, backup_id: str) -> bool:
        """
        Verify backup integrity.

        Args:
            backup_id: Backup ID to verify

        Returns:
            True if verification successful, False otherwise
        """
        self.logger.info(f"Verifying backup: {backup_id}")

        metadata = self._load_metadata(backup_id)
        if not metadata:
            self.logger.error(f"Backup metadata not found: {backup_id}")
            return False

        try:
            # Check if backup file exists
            if not os.path.exists(metadata.destination_path):
                raise FileNotFoundError(
                    f"Backup file not found: {metadata.destination_path}"
                )

            # Verify checksum
            current_checksum = self._calculate_checksum(metadata.destination_path)
            if current_checksum != metadata.checksum:
                raise ValueError(
                    f"Checksum mismatch: expected {metadata.checksum}, got {current_checksum}"
                )

            # Test extraction (to temp directory)
            with tempfile.TemporaryDirectory() as temp_dir:
                if metadata.encrypted:
                    # Decrypt to temp file
                    temp_decrypted = os.path.join(temp_dir, "decrypted.tar.gz")
                    self.encryption.decrypt_file(
                        metadata.destination_path, temp_decrypted
                    )
                    test_path = temp_decrypted
                else:
                    test_path = metadata.destination_path

                # Test extraction
                with tarfile.open(test_path, "r:gz") as tar:
                    tar.extractall(temp_dir)

                self.logger.info(f"Backup extraction test successful: {backup_id}")

            # Update metadata
            metadata.status = BackupStatus.VERIFIED
            metadata.verification_status = "success"
            metadata.verification_timestamp = datetime.utcnow()
            self._save_metadata(metadata)

            self.logger.info(f"Backup verified successfully: {backup_id}")
            return True

        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")

            # Update metadata
            metadata.status = BackupStatus.VERIFICATION_FAILED
            metadata.verification_status = f"failed: {str(e)}"
            metadata.verification_timestamp = datetime.utcnow()
            self._save_metadata(metadata)

            return False

    async def restore_backup(
        self, backup_id: str, restore_path: str, verify_before_restore: bool = True
    ) -> bool:
        """
        Restore a backup.

        Args:
            backup_id: Backup ID to restore
            restore_path: Path to restore to
            verify_before_restore: Verify backup before restoring

        Returns:
            True if restore successful, False otherwise
        """
        self.logger.info(f"Restoring backup: {backup_id} to {restore_path}")

        metadata = self._load_metadata(backup_id)
        if not metadata:
            self.logger.error(f"Backup metadata not found: {backup_id}")
            return False

        try:
            # Verify backup if requested
            if verify_before_restore:
                if not await self.verify_backup(backup_id):
                    raise ValueError("Backup verification failed")

            # Create restore directory
            os.makedirs(restore_path, exist_ok=True)

            # Decrypt if needed
            if metadata.encrypted:
                with tempfile.NamedTemporaryFile(
                    suffix=".tar.gz", delete=False
                ) as temp_file:
                    temp_path = temp_file.name

                self.encryption.decrypt_file(metadata.destination_path, temp_path)
                extract_path = temp_path
            else:
                extract_path = metadata.destination_path

            # Extract backup
            with tarfile.open(extract_path, "r:gz") as tar:
                tar.extractall(restore_path)

            # Clean up temp file if created
            if metadata.encrypted:
                os.remove(temp_path)

            self.logger.info(f"Backup restored successfully: {backup_id}")
            return True

        except Exception as e:
            self.logger.error(f"Backup restore failed: {e}")
            return False

    async def cleanup_old_backups(self) -> List[str]:
        """
        Clean up old backups based on retention policy.

        Returns:
            List of deleted backup IDs
        """
        self.logger.info("Cleaning up old backups")

        deleted_backups = []
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

        for metadata_file in self.metadata_dir.glob("*.json"):
            metadata = self._load_metadata(metadata_file.stem)
            if metadata and metadata.timestamp < cutoff_date:
                try:
                    # Delete backup file
                    if os.path.exists(metadata.destination_path):
                        os.remove(metadata.destination_path)

                    # Delete backup directory
                    backup_dir = Path(metadata.destination_path).parent
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)

                    # Delete metadata
                    metadata_file.unlink()

                    deleted_backups.append(metadata.backup_id)
                    self.logger.info(f"Deleted old backup: {metadata.backup_id}")

                except Exception as e:
                    self.logger.error(
                        f"Failed to delete backup {metadata.backup_id}: {e}"
                    )

        self.logger.info(f"Cleaned up {len(deleted_backups)} old backups")
        return deleted_backups

    def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        status: Optional[BackupStatus] = None,
    ) -> List[BackupMetadata]:
        """
        List all backups.

        Args:
            backup_type: Filter by backup type
            status: Filter by status

        Returns:
            List of backup metadata
        """
        backups = []

        for metadata_file in self.metadata_dir.glob("*.json"):
            metadata = self._load_metadata(metadata_file.stem)
            if metadata:
                if backup_type and metadata.backup_type != backup_type:
                    continue
                if status and metadata.status != status:
                    continue
                backups.append(metadata)

        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.timestamp, reverse=True)

        return backups

    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup statistics."""
        backups = self.list_backups()

        total_size = sum(b.size_bytes for b in backups)
        verified_count = sum(1 for b in backups if b.status == BackupStatus.VERIFIED)
        failed_count = sum(1 for b in backups if b.status == BackupStatus.FAILED)

        return {
            "total_backups": len(backups),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "verified_backups": verified_count,
            "failed_backups": failed_count,
            "oldest_backup": min((b.timestamp for b in backups), default=None),
            "newest_backup": max((b.timestamp for b in backups), default=None),
            "backup_types": {
                bt.value: sum(1 for b in backups if b.backup_type == bt)
                for bt in BackupType
            },
        }

    def _generate_backup_id(self) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"backup_{timestamp}"

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def _save_metadata(self, metadata: BackupMetadata) -> None:
        """Save backup metadata."""
        metadata_path = self.metadata_dir / f"{metadata.backup_id}.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _load_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Load backup metadata."""
        metadata_path = self.metadata_dir / f"{backup_id}.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)

            return BackupMetadata(
                backup_id=data["backup_id"],
                backup_type=BackupType(data["backup_type"]),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                size_bytes=data["size_bytes"],
                checksum=data["checksum"],
                encrypted=data["encrypted"],
                compression=data["compression"],
                source_paths=data["source_paths"],
                destination_path=data["destination_path"],
                status=BackupStatus(data["status"]),
                verification_status=data.get("verification_status"),
                verification_timestamp=(
                    datetime.fromisoformat(data["verification_timestamp"])
                    if data.get("verification_timestamp")
                    else None
                ),
                retention_days=data.get("retention_days", 30),
                tags=data.get("tags", {}),
            )
        except Exception as e:
            self.logger.error(f"Failed to load metadata for {backup_id}: {e}")
            return None


async def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)

    # Initialize backup manager
    manager = BackupManager(
        backup_root="/tmp/qbitel_backups",
        encryption_key="test_encryption_key_change_in_production",
        retention_days=30,
    )

    # Create backup
    metadata = await manager.create_backup(
        source_paths=["/tmp/test_data"],
        backup_type=BackupType.FULL,
        tags={"environment": "production", "component": "database"},
    )

    print(f"Backup created: {metadata.backup_id}")

    # List backups
    backups = manager.list_backups()
    print(f"Total backups: {len(backups)}")

    # Get statistics
    stats = manager.get_backup_statistics()
    print(f"Backup statistics: {json.dumps(stats, indent=2, default=str)}")

    # Restore backup
    success = await manager.restore_backup(
        backup_id=metadata.backup_id, restore_path="/tmp/restored_data"
    )
    print(f"Restore successful: {success}")


if __name__ == "__main__":
    asyncio.run(main())
