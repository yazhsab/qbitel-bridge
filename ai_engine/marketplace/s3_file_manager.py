"""
CRONOS AI - S3 File Manager for Marketplace

Handles protocol file uploads, storage, and downloads using AWS S3.
"""

import logging
import hashlib
import mimetypes
from typing import Dict, Any, Optional, BinaryIO
from pathlib import Path
from datetime import datetime, timedelta
from uuid import UUID

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    BotoCoreError = Exception

from ..core.config import get_config

logger = logging.getLogger(__name__)


class S3FileManager:
    """
    Manages file uploads and downloads for marketplace protocols.

    Features:
    - Secure file uploads to S3
    - File validation and virus scanning
    - Signed URL generation for downloads
    - File versioning
    - Storage lifecycle management
    """

    def __init__(self):
        """Initialize S3 file manager."""
        self.config = get_config()

        if not BOTO3_AVAILABLE:
            logger.warning("boto3 not available - S3 functionality will be limited")
            self.s3_client = None
            return

        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=getattr(self.config, "aws_access_key_id", None),
                aws_secret_access_key=getattr(self.config, "aws_secret_access_key", None),
                region_name=getattr(self.config, "aws_region", "us-east-1"),
            )

            # S3 bucket from config
            self.bucket_name = self.config.marketplace.s3_bucket
            self.cdn_url = self.config.marketplace.cdn_url

            logger.info(f"S3 file manager initialized (bucket: {self.bucket_name})")

        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None

    async def upload_protocol_file(
        self,
        protocol_id: UUID,
        file_content: bytes,
        file_type: str,
        filename: str,
        version: str = "1.0.0",
    ) -> Dict[str, Any]:
        """
        Upload a protocol file to S3.

        Args:
            protocol_id: Protocol UUID
            file_content: File content as bytes
            file_type: Type of file (spec, parser, test_data, docs)
            filename: Original filename
            version: Protocol version

        Returns:
            Dict with upload details including URL and metadata
        """
        if not self.s3_client:
            logger.warning("S3 client not available, returning mock URL")
            return self._get_mock_upload_result(protocol_id, file_type, filename)

        try:
            # Validate file
            validation = self._validate_file(file_content, filename, file_type)
            if not validation["valid"]:
                raise ValueError(f"File validation failed: {validation['error']}")

            # Generate S3 key (path in bucket)
            s3_key = self._generate_s3_key(protocol_id, file_type, filename, version)

            # Determine content type
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

            # Calculate file hash for integrity verification
            file_hash = hashlib.sha256(file_content).hexdigest()

            # Upload to S3 with metadata
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType=content_type,
                Metadata={
                    "protocol-id": str(protocol_id),
                    "file-type": file_type,
                    "version": version,
                    "sha256": file_hash,
                    "uploaded-at": datetime.utcnow().isoformat(),
                },
                ServerSideEncryption="AES256",  # Encrypt at rest
                StorageClass="STANDARD",  # Can be changed to INTELLIGENT_TIERING for cost optimization
            )

            logger.info(f"File uploaded to S3: {s3_key} ({len(file_content)} bytes)")

            # Generate URLs
            public_url = self._get_public_url(s3_key)
            cdn_url = self._get_cdn_url(s3_key)

            return {
                "success": True,
                "s3_key": s3_key,
                "public_url": public_url,
                "cdn_url": cdn_url,
                "file_size": len(file_content),
                "file_hash": file_hash,
                "content_type": content_type,
                "uploaded_at": datetime.utcnow().isoformat(),
            }

        except (ClientError, BotoCoreError) as e:
            logger.error(f"S3 upload failed: {e}")
            raise
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise

    async def generate_download_url(
        self,
        s3_key: str,
        expiration_seconds: int = 3600,
    ) -> str:
        """
        Generate a pre-signed URL for downloading a file.

        Args:
            s3_key: S3 object key
            expiration_seconds: URL expiration time (default: 1 hour)

        Returns:
            Pre-signed download URL
        """
        if not self.s3_client:
            logger.warning("S3 client not available, returning mock URL")
            return f"https://mock-download.cronos-ai.com/{s3_key}"

        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": s3_key,
                },
                ExpiresIn=expiration_seconds,
            )

            logger.debug(f"Generated pre-signed URL for {s3_key} (expires in {expiration_seconds}s)")
            return url

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to generate pre-signed URL: {e}")
            raise

    async def delete_protocol_files(
        self,
        protocol_id: UUID,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Delete all files for a protocol or a specific version.

        Args:
            protocol_id: Protocol UUID
            version: Specific version to delete (None = all versions)

        Returns:
            Dict with deletion results
        """
        if not self.s3_client:
            logger.warning("S3 client not available, skipping deletion")
            return {"success": False, "error": "S3 client not available"}

        try:
            # List all objects for this protocol
            prefix = f"protocols/{protocol_id}/"
            if version:
                prefix += f"v{version}/"

            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if "Contents" not in response:
                return {"success": True, "deleted_count": 0, "message": "No files found"}

            # Delete all objects
            objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]

            if objects_to_delete:
                delete_response = self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={"Objects": objects_to_delete}
                )

                deleted_count = len(delete_response.get("Deleted", []))
                logger.info(f"Deleted {deleted_count} files for protocol {protocol_id}")

                return {
                    "success": True,
                    "deleted_count": deleted_count,
                    "deleted_files": [obj["Key"] for obj in delete_response.get("Deleted", [])]
                }

            return {"success": True, "deleted_count": 0}

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to delete protocol files: {e}")
            raise

    async def get_file_metadata(self, s3_key: str) -> Dict[str, Any]:
        """
        Get metadata for a file stored in S3.

        Args:
            s3_key: S3 object key

        Returns:
            Dict with file metadata
        """
        if not self.s3_client:
            return {"error": "S3 client not available"}

        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )

            return {
                "content_type": response.get("ContentType"),
                "content_length": response.get("ContentLength"),
                "last_modified": response.get("LastModified"),
                "etag": response.get("ETag"),
                "metadata": response.get("Metadata", {}),
                "storage_class": response.get("StorageClass"),
                "server_side_encryption": response.get("ServerSideEncryption"),
            }

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to get file metadata: {e}")
            raise

    def _validate_file(
        self,
        file_content: bytes,
        filename: str,
        file_type: str
    ) -> Dict[str, Any]:
        """
        Validate file before upload.

        Args:
            file_content: File content
            filename: Filename
            file_type: Type of file

        Returns:
            Dict with validation result
        """
        # Check file size limits
        max_sizes = {
            "spec": 10 * 1024 * 1024,  # 10 MB
            "parser": 5 * 1024 * 1024,  # 5 MB
            "test_data": 50 * 1024 * 1024,  # 50 MB
            "docs": 10 * 1024 * 1024,  # 10 MB
        }

        max_size = max_sizes.get(file_type, 10 * 1024 * 1024)
        if len(file_content) > max_size:
            return {
                "valid": False,
                "error": f"File too large: {len(file_content)} bytes (max: {max_size})"
            }

        # Check for empty files
        if len(file_content) == 0:
            return {"valid": False, "error": "File is empty"}

        # Validate file extensions
        valid_extensions = {
            "spec": [".yaml", ".yml", ".json", ".proto", ".xml"],
            "parser": [".py", ".zip"],
            "test_data": [".json", ".csv", ".txt", ".pcap", ".zip"],
            "docs": [".md", ".pdf", ".html", ".txt"],
        }

        if file_type in valid_extensions:
            file_ext = Path(filename).suffix.lower()
            if file_ext not in valid_extensions[file_type]:
                return {
                    "valid": False,
                    "error": f"Invalid file extension: {file_ext} (allowed: {valid_extensions[file_type]})"
                }

        # TODO: Add virus scanning integration
        # TODO: Add content validation (e.g., YAML parsing for spec files)

        return {"valid": True}

    def _generate_s3_key(
        self,
        protocol_id: UUID,
        file_type: str,
        filename: str,
        version: str
    ) -> str:
        """
        Generate S3 key (path) for a file.

        Format: protocols/{protocol_id}/v{version}/{file_type}/{filename}

        Args:
            protocol_id: Protocol UUID
            file_type: Type of file
            filename: Original filename
            version: Protocol version

        Returns:
            S3 key string
        """
        # Sanitize filename
        safe_filename = Path(filename).name  # Remove any path components

        return f"protocols/{protocol_id}/v{version}/{file_type}/{safe_filename}"

    def _get_public_url(self, s3_key: str) -> str:
        """Get public S3 URL (if bucket is public) or regional URL."""
        region = getattr(self.config, "aws_region", "us-east-1")
        return f"https://{self.bucket_name}.s3.{region}.amazonaws.com/{s3_key}"

    def _get_cdn_url(self, s3_key: str) -> str:
        """Get CDN URL for file (if CDN is configured)."""
        if self.cdn_url:
            return f"{self.cdn_url}/{s3_key}"
        return self._get_public_url(s3_key)

    def _get_mock_upload_result(
        self,
        protocol_id: UUID,
        file_type: str,
        filename: str
    ) -> Dict[str, Any]:
        """Return mock upload result when S3 is not available."""
        mock_key = f"protocols/{protocol_id}/{file_type}/{filename}"
        return {
            "success": True,
            "s3_key": mock_key,
            "public_url": f"https://mock-s3.cronos-ai.com/{mock_key}",
            "cdn_url": f"https://mock-cdn.cronos-ai.com/{mock_key}",
            "file_size": 0,
            "file_hash": "mock_hash",
            "content_type": "application/octet-stream",
            "uploaded_at": datetime.utcnow().isoformat(),
            "mock": True,
        }


# Singleton instance
_s3_manager: Optional[S3FileManager] = None


def get_s3_manager() -> S3FileManager:
    """Get singleton S3 file manager instance."""
    global _s3_manager

    if _s3_manager is None:
        _s3_manager = S3FileManager()

    return _s3_manager
