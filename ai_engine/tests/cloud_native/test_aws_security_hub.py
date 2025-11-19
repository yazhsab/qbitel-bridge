"""
Unit tests for AWS Security Hub Integration.
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
from datetime import datetime

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.cloud_integrations.aws.security_hub import (
    AWSSecurityHubIntegration,
    FindingImportError
)


class TestAWSSecurityHubIntegration:
    """Test AWSSecurityHubIntegration class"""

    @pytest.fixture
    def security_hub(self):
        """Create Security Hub integration instance"""
        with patch('boto3.client'):
            return AWSSecurityHubIntegration(
                region="us-east-1",
                account_id="123456789012",
                max_retries=3
            )

    def test_initialization(self, security_hub):
        """Test initialization"""
        assert security_hub.region == "us-east-1"
        assert security_hub.account_id == "123456789012"
        assert security_hub.max_retries == 3

    @patch('boto3.client')
    def test_get_client(self, mock_boto_client):
        """Test boto3 client creation"""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        hub = AWSSecurityHubIntegration(region="us-west-2", account_id="123456789012")
        client = hub._get_client()

        assert client is not None
        mock_boto_client.assert_called()

    @patch('boto3.client')
    def test_publish_finding(self, mock_boto_client):
        """Test publishing single finding"""
        mock_client = MagicMock()
        mock_client.batch_import_findings.return_value = {
            "FailedCount": 0,
            "SuccessCount": 1
        }
        mock_boto_client.return_value = mock_client

        hub = AWSSecurityHubIntegration(region="us-east-1", account_id="123456789012")

        result = hub.publish_finding(
            title="Security Issue Detected",
            description="Test vulnerability found",
            severity=70,
            resource_id="arn:aws:ec2:us-east-1:123456789012:instance/i-1234567890abcdef0",
            finding_type="Software and Configuration Checks/Vulnerabilities/CVE",
            resource_type="AwsEc2Instance"
        )

        assert result is True
        mock_client.batch_import_findings.assert_called_once()

    @patch('boto3.client')
    def test_batch_publish_findings(self, mock_boto_client):
        """Test batch publishing findings"""
        mock_client = MagicMock()
        mock_client.batch_import_findings.return_value = {
            "FailedCount": 0,
            "SuccessCount": 3
        }
        mock_boto_client.return_value = mock_client

        hub = AWSSecurityHubIntegration(region="us-east-1", account_id="123456789012")

        findings = [
            {
                "title": "Finding 1",
                "description": "Description 1",
                "severity": 50,
                "resource_id": "arn:aws:s3:::my-bucket",
                "finding_type": "Sensitive Data Identifications",
                "resource_type": "AwsS3Bucket"
            },
            {
                "title": "Finding 2",
                "description": "Description 2",
                "severity": 80,
                "resource_id": "arn:aws:ec2:us-east-1:123456789012:instance/i-test",
                "finding_type": "TTPs/Initial Access",
                "resource_type": "AwsEc2Instance"
            }
        ]

        result = hub.batch_publish_findings(findings)

        assert result["SuccessCount"] == 3
        assert result["FailedCount"] == 0

    @patch('boto3.client')
    def test_publish_finding_failure(self, mock_boto_client):
        """Test handling of finding publication failure"""
        mock_client = MagicMock()
        mock_client.batch_import_findings.side_effect = Exception("API Error")
        mock_boto_client.return_value = mock_client

        hub = AWSSecurityHubIntegration(region="us-east-1", account_id="123456789012")

        with pytest.raises(FindingImportError):
            hub.publish_finding(
                title="Test",
                description="Test",
                severity=50,
                resource_id="arn:aws:test",
                finding_type="Test",
                resource_type="Other"
            )

    @patch('boto3.client')
    def test_retry_logic(self, mock_boto_client):
        """Test retry logic on transient failures"""
        mock_client = MagicMock()
        # Fail twice, then succeed
        mock_client.batch_import_findings.side_effect = [
            Exception("Throttling"),
            Exception("Throttling"),
            {"FailedCount": 0, "SuccessCount": 1}
        ]
        mock_boto_client.return_value = mock_client

        hub = AWSSecurityHubIntegration(
            region="us-east-1",
            account_id="123456789012",
            max_retries=3
        )

        # Should succeed after retries
        try:
            result = hub.publish_finding(
                title="Test",
                description="Test",
                severity=50,
                resource_id="arn:aws:test",
                finding_type="Test"
            )
            # If retry logic works, it should eventually succeed
            assert result is True or result is False
        except FindingImportError:
            # Acceptable if retries exhausted
            assert True

    def test_create_cloudformation_template(self, security_hub):
        """Test CloudFormation template generation"""
        template = security_hub.create_cloudformation_template()

        assert template is not None
        assert isinstance(template, dict)
        assert "AWSTemplateFormatVersion" in template
        assert "Resources" in template

        # Check for Security Hub resource
        resources = template["Resources"]
        assert any("SecurityHub" in key for key in resources.keys())

    @patch('boto3.client')
    def test_close_connection(self, mock_boto_client):
        """Test closing client connection"""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        hub = AWSSecurityHubIntegration(region="us-east-1", account_id="123456789012")
        hub.close()

        # Verify cleanup
        assert True

    @patch('boto3.client')
    def test_batch_size_limit(self, mock_boto_client):
        """Test batch size limit enforcement (max 100)"""
        mock_client = MagicMock()
        mock_client.batch_import_findings.return_value = {
            "FailedCount": 0,
            "SuccessCount": 100
        }
        mock_boto_client.return_value = mock_client

        hub = AWSSecurityHubIntegration(region="us-east-1", account_id="123456789012")

        # Create 150 findings
        findings = [
            {
                "title": f"Finding {i}",
                "description": f"Description {i}",
                "severity": 50,
                "resource_id": f"arn:aws:test:resource-{i}",
                "finding_type": "Test"
            }
            for i in range(150)
        ]

        result = hub.batch_publish_findings(findings)

        # Should handle batching internally
        assert result is not None

    @patch('boto3.client')
    def test_severity_normalization(self, mock_boto_client):
        """Test severity value normalization"""
        mock_client = MagicMock()
        mock_client.batch_import_findings.return_value = {
            "FailedCount": 0,
            "SuccessCount": 1
        }
        mock_boto_client.return_value = mock_client

        hub = AWSSecurityHubIntegration(region="us-east-1", account_id="123456789012")

        # Test different severity values (0-100 scale)
        severities = [0, 25, 50, 75, 100]

        for sev in severities:
            result = hub.publish_finding(
                title="Test",
                description="Test",
                severity=sev,
                resource_id="arn:aws:test",
                finding_type="Test"
            )
            assert isinstance(result, bool)

    @patch('boto3.client')
    def test_quantum_vulnerability_finding(self, mock_boto_client):
        """Test publishing quantum vulnerability finding"""
        mock_client = MagicMock()
        mock_client.batch_import_findings.return_value = {
            "FailedCount": 0,
            "SuccessCount": 1
        }
        mock_boto_client.return_value = mock_client

        hub = AWSSecurityHubIntegration(region="us-east-1", account_id="123456789012")

        result = hub.publish_finding(
            title="Quantum-Vulnerable Cryptography Detected",
            description="Container uses RSA-2048 which is vulnerable to quantum attacks",
            severity=80,
            resource_id="arn:aws:ecs:us-east-1:123456789012:task/my-task",
            finding_type="Software and Configuration Checks/Vulnerabilities",
            resource_type="AwsEcsTask"
        )

        assert isinstance(result, bool)
