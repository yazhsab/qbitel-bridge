#!/usr/bin/env python3
"""
QBITEL - S3 Bucket Setup Script

Creates and configures the S3 bucket for marketplace protocol storage.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path to import ai_engine
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Run: pip3 install boto3")
    sys.exit(1)

from ai_engine.core.config import get_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_s3_bucket(bucket_name: str, region: str) -> bool:
    """
    Create S3 bucket with proper configuration.

    Args:
        bucket_name: Name of the S3 bucket
        region: AWS region

    Returns:
        True if successful
    """
    try:
        s3_client = boto3.client('s3', region_name=region)

        # Check if bucket already exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"✅ Bucket '{bucket_name}' already exists")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code != '404':
                logger.error(f"Error checking bucket: {e}")
                return False

        # Create bucket
        logger.info(f"Creating bucket '{bucket_name}' in region '{region}'...")

        if region == 'us-east-1':
            # us-east-1 doesn't need LocationConstraint
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )

        logger.info(f"✅ Bucket '{bucket_name}' created successfully")
        return True

    except ClientError as e:
        logger.error(f"Failed to create bucket: {e}")
        return False


def configure_bucket_encryption(bucket_name: str) -> bool:
    """
    Enable server-side encryption for the bucket.

    Args:
        bucket_name: Name of the S3 bucket

    Returns:
        True if successful
    """
    try:
        s3_client = boto3.client('s3')

        logger.info("Enabling server-side encryption (AES-256)...")

        s3_client.put_bucket_encryption(
            Bucket=bucket_name,
            ServerSideEncryptionConfiguration={
                'Rules': [
                    {
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'AES256'
                        },
                        'BucketKeyEnabled': True
                    }
                ]
            }
        )

        logger.info("✅ Encryption enabled")
        return True

    except ClientError as e:
        logger.error(f"Failed to enable encryption: {e}")
        return False


def configure_bucket_versioning(bucket_name: str) -> bool:
    """
    Enable versioning for the bucket.

    Args:
        bucket_name: Name of the S3 bucket

    Returns:
        True if successful
    """
    try:
        s3_client = boto3.client('s3')

        logger.info("Enabling versioning...")

        s3_client.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )

        logger.info("✅ Versioning enabled")
        return True

    except ClientError as e:
        logger.error(f"Failed to enable versioning: {e}")
        return False


def configure_bucket_cors(bucket_name: str, allowed_origins: list[str] | None = None) -> bool:
    """
    Configure CORS for the bucket to allow web uploads.

    SECURITY: Uses environment variable for allowed origins in production.
    Set CORS_ALLOWED_ORIGINS environment variable with comma-separated list of domains.
    Example: CORS_ALLOWED_ORIGINS=https://qbitel.example.com,https://admin.example.com

    Args:
        bucket_name: Name of the S3 bucket
        allowed_origins: Optional list of allowed origins. If not provided,
                        reads from CORS_ALLOWED_ORIGINS env var.
                        Falls back to restrictive default if not set.

    Returns:
        True if successful
    """
    import os

    try:
        s3_client = boto3.client('s3')

        logger.info("Configuring CORS...")

        # SECURITY: Get allowed origins from environment or parameter
        if allowed_origins is None:
            env_origins = os.getenv('CORS_ALLOWED_ORIGINS', '')
            if env_origins:
                allowed_origins = [origin.strip() for origin in env_origins.split(',') if origin.strip()]
                logger.info(f"Using CORS origins from environment: {allowed_origins}")
            else:
                # SECURITY: Default to restrictive origin instead of wildcard
                # In development, set CORS_ALLOWED_ORIGINS=http://localhost:3000
                allowed_origins = ['https://qbitel.example.com']
                logger.warning("⚠️  CORS_ALLOWED_ORIGINS not set - using restrictive default")
                logger.warning("    Set CORS_ALLOWED_ORIGINS env var for your domains")

        # Validate origins - must be proper URLs, not wildcards
        validated_origins = []
        for origin in allowed_origins:
            if origin == '*':
                logger.warning("⚠️  Wildcard '*' origin rejected for security - skipping")
                continue
            if not origin.startswith(('http://', 'https://')):
                logger.warning(f"⚠️  Invalid origin format '{origin}' - skipping")
                continue
            validated_origins.append(origin)

        if not validated_origins:
            logger.error("❌ No valid origins configured - CORS will not be set")
            logger.error("   Set CORS_ALLOWED_ORIGINS=https://your-domain.com")
            return False

        cors_configuration = {
            'CORSRules': [
                {
                    'AllowedHeaders': [
                        'Content-Type',
                        'Content-Length',
                        'Authorization',
                        'X-Amz-Date',
                        'X-Amz-Security-Token',
                        'X-Amz-Content-Sha256',
                    ],
                    'AllowedMethods': ['GET', 'PUT', 'POST', 'HEAD'],
                    'AllowedOrigins': validated_origins,
                    'ExposeHeaders': ['ETag', 'Content-Length'],
                    'MaxAgeSeconds': 3600
                }
            ]
        }

        s3_client.put_bucket_cors(
            Bucket=bucket_name,
            CORSConfiguration=cors_configuration
        )

        logger.info(f"✅ CORS configured for origins: {validated_origins}")
        return True

    except ClientError as e:
        logger.error(f"Failed to configure CORS: {e}")
        return False


def configure_bucket_lifecycle(bucket_name: str) -> bool:
    """
    Configure lifecycle policies to archive old files.

    Args:
        bucket_name: Name of the S3 bucket

    Returns:
        True if successful
    """
    try:
        s3_client = boto3.client('s3')

        logger.info("Configuring lifecycle policies...")

        lifecycle_configuration = {
            'Rules': [
                {
                    'Id': 'archive-old-protocols',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'protocols/'},
                    'Transitions': [
                        {
                            'Days': 90,
                            'StorageClass': 'INTELLIGENT_TIERING'
                        },
                        {
                            'Days': 365,
                            'StorageClass': 'GLACIER'
                        }
                    ],
                    'NoncurrentVersionTransitions': [
                        {
                            'NoncurrentDays': 30,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                }
            ]
        }

        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_configuration
        )

        logger.info("✅ Lifecycle policies configured")
        logger.info("   - Files move to Intelligent Tiering after 90 days")
        logger.info("   - Files archive to Glacier after 365 days")
        return True

    except ClientError as e:
        logger.error(f"Failed to configure lifecycle: {e}")
        return False


def configure_bucket_public_access(bucket_name: str, public: bool = False) -> bool:
    """
    Configure public access settings for the bucket.

    Args:
        bucket_name: Name of the S3 bucket
        public: Whether to allow public access

    Returns:
        True if successful
    """
    try:
        s3_client = boto3.client('s3')

        logger.info(f"Configuring public access (public={public})...")

        # Block all public access by default (use pre-signed URLs)
        s3_client.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                'BlockPublicAcls': not public,
                'IgnorePublicAcls': not public,
                'BlockPublicPolicy': not public,
                'RestrictPublicBuckets': not public
            }
        )

        if public:
            logger.warning("⚠️  Public access enabled - use with caution!")
        else:
            logger.info("✅ Public access blocked (using pre-signed URLs)")

        return True

    except ClientError as e:
        logger.error(f"Failed to configure public access: {e}")
        return False


def create_bucket_folders(bucket_name: str) -> bool:
    """
    Create the initial folder structure in the bucket.

    Args:
        bucket_name: Name of the S3 bucket

    Returns:
        True if successful
    """
    try:
        s3_client = boto3.client('s3')

        logger.info("Creating folder structure...")

        folders = [
            'protocols/',
            'temp/',
            'backups/'
        ]

        for folder in folders:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=folder,
                Body=b''
            )

        logger.info("✅ Folder structure created")
        return True

    except ClientError as e:
        logger.error(f"Failed to create folders: {e}")
        return False


def verify_bucket_access(bucket_name: str) -> bool:
    """
    Verify that we can read/write to the bucket.

    Args:
        bucket_name: Name of the S3 bucket

    Returns:
        True if successful
    """
    try:
        s3_client = boto3.client('s3')

        logger.info("Verifying bucket access...")

        # Test write
        test_key = 'test/access-test.txt'
        s3_client.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=b'QBITEL S3 access test'
        )

        # Test read
        response = s3_client.get_object(Bucket=bucket_name, Key=test_key)
        content = response['Body'].read()

        # Clean up
        s3_client.delete_object(Bucket=bucket_name, Key=test_key)

        logger.info("✅ Bucket access verified")
        return True

    except ClientError as e:
        logger.error(f"Failed to verify bucket access: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 70)
    print("QBITEL - S3 Bucket Setup for Protocol Marketplace")
    print("=" * 70)
    print()

    # Load configuration
    try:
        config = get_config()
        bucket_name = config.marketplace.s3_bucket
        region = config.aws_region
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.info("\nPlease ensure environment variables are set:")
        logger.info("  - MARKETPLACE_S3_BUCKET")
        logger.info("  - AWS_REGION (optional, defaults to us-east-1)")
        logger.info("  - AWS_ACCESS_KEY_ID")
        logger.info("  - AWS_SECRET_ACCESS_KEY")
        return False

    logger.info(f"Bucket Name: {bucket_name}")
    logger.info(f"Region: {region}")
    print()

    # Create bucket
    if not create_s3_bucket(bucket_name, region):
        return False

    # Configure encryption
    if not configure_bucket_encryption(bucket_name):
        logger.warning("⚠️  Encryption configuration failed - continuing anyway")

    # Configure versioning
    if not configure_bucket_versioning(bucket_name):
        logger.warning("⚠️  Versioning configuration failed - continuing anyway")

    # Configure CORS
    if not configure_bucket_cors(bucket_name):
        logger.warning("⚠️  CORS configuration failed - continuing anyway")

    # Configure lifecycle
    if not configure_bucket_lifecycle(bucket_name):
        logger.warning("⚠️  Lifecycle configuration failed - continuing anyway")

    # Configure public access (block by default)
    if not configure_bucket_public_access(bucket_name, public=False):
        logger.warning("⚠️  Public access configuration failed - continuing anyway")

    # Create folders
    if not create_bucket_folders(bucket_name):
        logger.warning("⚠️  Folder creation failed - continuing anyway")

    # Verify access
    if not verify_bucket_access(bucket_name):
        logger.error("❌ Bucket access verification failed")
        return False

    print()
    print("=" * 70)
    print("✅ S3 bucket setup complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Configure CDN (CloudFront) for faster downloads (optional)")
    print("2. Set MARKETPLACE_CDN_URL environment variable")
    print("3. Test file upload via marketplace API")
    print()
    print("Bucket URL:")
    print(f"  https://{bucket_name}.s3.{region}.amazonaws.com/")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
