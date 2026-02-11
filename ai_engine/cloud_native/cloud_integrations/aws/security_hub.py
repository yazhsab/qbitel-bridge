"""
AWS Security Hub Integration

Integrates QBITEL findings with AWS Security Hub for centralized
security monitoring and compliance management.
"""

import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timezone
from botocore.exceptions import ClientError, BotoCoreError
import time
import boto3

logger = logging.getLogger(__name__)


class FindingImportError(Exception):
    """Raised when Security Hub fails to import a finding"""
    pass


class AWSSecurityHubIntegration:
    """
    Integrates with AWS Security Hub to publish security findings.
    """

    def __init__(self, region: str = "us-east-1", account_id: Optional[str] = None, max_retries: int = 3):
        """
        Initialize AWS Security Hub integration.

        Args:
            region: AWS region
            account_id: AWS account ID
            max_retries: Maximum number of retry attempts for AWS API calls
        """
        self.region = region
        self.account_id = account_id
        self.max_retries = max_retries
        self._client = None
        logger.info(f"Initialized AWS Security Hub integration for {region}")

    def _get_client(self):
        """
        Lazy initialization of boto3 client.

        Returns:
            boto3 SecurityHub client
        """
        if self._client is None:
            try:
                self._client = boto3.client('securityhub', region_name=self.region)
                logger.info(f"Created boto3 SecurityHub client for region {self.region}")
            except Exception as e:
                logger.error(f"Failed to create boto3 client: {e}")
                raise
        return self._client

    def publish_finding(
        self,
        title: str,
        description: str,
        severity: str,
        resource_id: str,
        finding_type: str = "Software and Configuration Checks",
        resource_type: str = "Container"
    ) -> Dict[str, Any]:
        """
        Publish a finding to AWS Security Hub.

        Args:
            title: Finding title
            description: Finding description
            severity: Severity (CRITICAL, HIGH, MEDIUM, LOW, INFORMATIONAL)
            resource_id: AWS resource identifier
            finding_type: Type of finding
            resource_type: Type of AWS resource (default: Container)

        Returns:
            Dict containing AWS API response

        Raises:
            ClientError: If AWS API call fails
            ValueError: If invalid parameters provided
        """
        # Validate severity
        valid_severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"]
        if severity not in valid_severities:
            raise ValueError(f"Invalid severity: {severity}. Must be one of {valid_severities}")

        # Get current timestamp in ISO 8601 format
        timestamp = datetime.now(timezone.utc).isoformat()

        # Build finding structure
        finding = {
            "SchemaVersion": "2018-10-08",
            "Id": f"qbitel/{resource_id}/{int(time.time())}",
            "ProductArn": f"arn:aws:securityhub:{self.region}:{self.account_id}:product/qbitel/quantum-security",
            "GeneratorId": "qbitel-quantum-scanner",
            "AwsAccountId": self.account_id,
            "Types": [finding_type],
            "CreatedAt": timestamp,
            "UpdatedAt": timestamp,
            "Severity": {
                "Label": severity
            },
            "Title": title,
            "Description": description,
            "Resources": [{
                "Type": resource_type,
                "Id": resource_id
            }],
            "Compliance": {
                "Status": "FAILED" if severity in ["CRITICAL", "HIGH"] else "WARNING"
            }
        }

        # Publish to AWS Security Hub with retry logic
        for attempt in range(self.max_retries):
            try:
                client = self._get_client()
                response = client.batch_import_findings(Findings=[finding])

                # Check for failures
                if response.get('FailedCount', 0) > 0:
                    failed_findings = response.get('FailedFindings', [])
                    error_msg = f"Failed to import finding: {failed_findings}"
                    logger.error(error_msg)
                    raise FindingImportError(error_msg)

                logger.info(f"Successfully published finding to Security Hub: {title} (Severity: {severity})")
                return {
                    "success": True,
                    "finding_id": finding["Id"],
                    "response": response,
                    "timestamp": timestamp
                }

            except FindingImportError:
                # Finding import failed - don't retry, re-raise immediately
                raise

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))

                logger.warning(f"AWS API error on attempt {attempt + 1}/{self.max_retries}: {error_code} - {error_message}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to publish finding after {self.max_retries} attempts")
                    raise

            except BotoCoreError as e:
                logger.error(f"BotoCore error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise

            except Exception as e:
                # For unexpected exceptions, don't retry
                logger.error(f"Unexpected error publishing finding: {e}")
                raise

        # Should not reach here, but just in case
        raise Exception("Failed to publish finding after all retry attempts")

    def batch_publish_findings(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Publish multiple findings to AWS Security Hub in batches.

        AWS Security Hub accepts up to 100 findings per batch_import_findings call.

        Args:
            findings: List of finding dictionaries

        Returns:
            Dict containing batch import results

        Raises:
            ClientError: If AWS API call fails
        """
        if not findings:
            logger.warning("No findings to publish")
            return {"success": True, "processed": 0, "failed": 0}

        batch_size = 100  # AWS Security Hub limit
        total_processed = 0
        total_failed = 0
        results = []

        # Process in batches of 100
        for i in range(0, len(findings), batch_size):
            batch = findings[i:i + batch_size]

            try:
                client = self._get_client()
                response = client.batch_import_findings(Findings=batch)

                success_count = response.get('SuccessCount', 0)
                failed_count = response.get('FailedCount', 0)

                total_processed += success_count
                total_failed += failed_count

                results.append({
                    "batch_index": i // batch_size,
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "failed_findings": response.get('FailedFindings', [])
                })

                logger.info(f"Batch {i // batch_size + 1}: {success_count} succeeded, {failed_count} failed")

            except Exception as e:
                logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                total_failed += len(batch)
                results.append({
                    "batch_index": i // batch_size,
                    "error": str(e),
                    "failed_count": len(batch)
                })

        return {
            "success": total_failed == 0,
            "total_processed": total_processed,
            "total_failed": total_failed,
            "batches": results
        }

    def close(self):
        """Close the AWS client connection."""
        if self._client is not None:
            logger.info("Closing AWS SecurityHub client")
            self._client = None

    def create_cloudformation_template(self) -> Dict[str, Any]:
        """Create CloudFormation template for deployment"""
        return {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "QBITEL Security Hub Integration",
            "Resources": {
                "QbitelSecurityHubIntegration": {
                    "Type": "AWS::SecurityHub::ProductSubscription",
                    "Properties": {
                        "ProductArn": f"arn:aws:securityhub:{self.region}::product/qbitel/quantum-security"
                    }
                }
            }
        }
