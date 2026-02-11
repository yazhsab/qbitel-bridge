"""
Unit tests for AWS Security Hub Integration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from botocore.exceptions import ClientError, BotoCoreError

from ai_engine.cloud_native.cloud_integrations.aws.security_hub import (
    AWSSecurityHubIntegration,
    FindingImportError
)


class TestAWSSecurityHubIntegration:
    """Test suite for AWS Security Hub integration"""

    @pytest.fixture
    def security_hub(self):
        """Create a SecurityHub integration instance"""
        return AWSSecurityHubIntegration(
            region="us-east-1",
            account_id="123456789012",
            max_retries=3
        )

    @pytest.fixture
    def mock_boto_client(self):
        """Create a mock boto3 client"""
        with patch('boto3.client') as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    def test_initialization(self, security_hub):
        """Test SecurityHub integration initialization"""
        assert security_hub.region == "us-east-1"
        assert security_hub.account_id == "123456789012"
        assert security_hub.max_retries == 3
        assert security_hub._client is None

    def test_get_client_creates_client(self, security_hub, mock_boto_client):
        """Test lazy client initialization"""
        client = security_hub._get_client()

        assert client is not None
        assert security_hub._client is not None

    def test_publish_finding_success(self, security_hub, mock_boto_client):
        """Test successful finding publication"""
        # Setup mock response
        mock_boto_client.batch_import_findings.return_value = {
            'FailedCount': 0,
            'SuccessCount': 1
        }

        # Publish finding
        result = security_hub.publish_finding(
            title="Test Finding",
            description="This is a test finding",
            severity="HIGH",
            resource_id="test-resource-123"
        )

        # Verify result
        assert result["success"] is True
        assert "finding_id" in result
        assert "timestamp" in result
        assert result["response"]["SuccessCount"] == 1

        # Verify boto3 was called
        mock_boto_client.batch_import_findings.assert_called_once()

    def test_publish_finding_invalid_severity(self, security_hub):
        """Test that invalid severity raises ValueError"""
        with pytest.raises(ValueError, match="Invalid severity"):
            security_hub.publish_finding(
                title="Test Finding",
                description="Test",
                severity="INVALID",
                resource_id="test-123"
            )

    def test_publish_finding_with_failed_count(self, security_hub, mock_boto_client):
        """Test handling of failed findings"""
        # Setup mock response with failures
        mock_boto_client.batch_import_findings.return_value = {
            'FailedCount': 1,
            'SuccessCount': 0,
            'FailedFindings': [{'Id': 'test-id', 'ErrorCode': 'InvalidInput'}]
        }

        # Should raise FindingImportError due to failed import
        with pytest.raises(FindingImportError, match="Failed to import finding"):
            security_hub.publish_finding(
                title="Test Finding",
                description="Test",
                severity="HIGH",
                resource_id="test-123"
            )

    def test_publish_finding_retry_on_client_error(self, security_hub):
        """Test retry logic on ClientError"""
        with patch('ai_engine.cloud_native.cloud_integrations.aws.security_hub.boto3') as mock_boto:
            mock_client = MagicMock()
            mock_boto.client.return_value = mock_client

            # Setup mock to fail twice, then succeed
            mock_client.batch_import_findings.side_effect = [
                ClientError(
                    {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
                    'batch_import_findings'
                ),
                ClientError(
                    {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
                    'batch_import_findings'
                ),
                {'FailedCount': 0, 'SuccessCount': 1}
            ]

            # Should succeed after retries
            with patch('time.sleep'):  # Speed up test by mocking sleep
                result = security_hub.publish_finding(
                    title="Test Finding",
                    description="Test",
                    severity="HIGH",
                    resource_id="test-123"
                )

            assert result["success"] is True
            assert mock_client.batch_import_findings.call_count == 3

    def test_publish_finding_max_retries_exceeded(self, security_hub):
        """Test that max retries are respected"""
        with patch('ai_engine.cloud_native.cloud_integrations.aws.security_hub.boto3') as mock_boto:
            mock_client = MagicMock()
            mock_boto.client.return_value = mock_client

            # Setup mock to always fail
            mock_client.batch_import_findings.side_effect = ClientError(
                {'Error': {'Code': 'ServiceError', 'Message': 'Service unavailable'}},
                'batch_import_findings'
            )

            # Should raise after max retries
            with patch('time.sleep'):
                with pytest.raises(ClientError):
                    security_hub.publish_finding(
                        title="Test Finding",
                        description="Test",
                        severity="HIGH",
                        resource_id="test-123"
                    )

            assert mock_client.batch_import_findings.call_count == 3

    def test_batch_publish_findings(self, security_hub, mock_boto_client):
        """Test batch publishing of findings"""
        # Setup mock response
        mock_boto_client.batch_import_findings.return_value = {
            'FailedCount': 0,
            'SuccessCount': 5
        }

        # Create test findings
        findings = [
            {
                "SchemaVersion": "2018-10-08",
                "Id": f"test-{i}",
                "ProductArn": "arn:aws:securityhub:us-east-1:123456789012:product/test",
                "GeneratorId": "test-generator",
                "AwsAccountId": "123456789012",
                "Types": ["Software and Configuration Checks"],
                "CreatedAt": datetime.now(timezone.utc).isoformat(),
                "UpdatedAt": datetime.now(timezone.utc).isoformat(),
                "Severity": {"Label": "HIGH"},
                "Title": f"Finding {i}",
                "Description": f"Test finding {i}",
                "Resources": [{"Type": "Container", "Id": f"resource-{i}"}]
            }
            for i in range(5)
        ]

        # Batch publish
        result = security_hub.batch_publish_findings(findings)

        # Verify result
        assert result["success"] is True
        assert result["total_processed"] == 5
        assert result["total_failed"] == 0
        assert mock_boto_client.batch_import_findings.call_count == 1

    def test_batch_publish_findings_multiple_batches(self, security_hub, mock_boto_client):
        """Test batch publishing with more than 100 findings"""
        # Setup mock response
        mock_boto_client.batch_import_findings.return_value = {
            'FailedCount': 0,
            'SuccessCount': 100
        }

        # Create 250 findings (should be split into 3 batches)
        findings = [
            {
                "SchemaVersion": "2018-10-08",
                "Id": f"test-{i}",
                "ProductArn": "arn:aws:securityhub:us-east-1:123456789012:product/test",
                "GeneratorId": "test-generator",
                "AwsAccountId": "123456789012",
                "Types": ["Software and Configuration Checks"],
                "CreatedAt": datetime.now(timezone.utc).isoformat(),
                "UpdatedAt": datetime.now(timezone.utc).isoformat(),
                "Severity": {"Label": "HIGH"},
                "Title": f"Finding {i}",
                "Description": f"Test finding {i}",
                "Resources": [{"Type": "Container", "Id": f"resource-{i}"}]
            }
            for i in range(250)
        ]

        # Batch publish
        result = security_hub.batch_publish_findings(findings)

        # Verify 3 batches were processed
        assert mock_boto_client.batch_import_findings.call_count == 3

    def test_batch_publish_empty_findings(self, security_hub):
        """Test batch publishing with empty findings list"""
        result = security_hub.batch_publish_findings([])

        assert result["success"] is True
        assert result["processed"] == 0
        assert result["failed"] == 0

    def test_close(self, security_hub, mock_boto_client):
        """Test client cleanup"""
        # Initialize client
        security_hub._get_client()
        assert security_hub._client is not None

        # Close
        security_hub.close()
        assert security_hub._client is None

    def test_create_cloudformation_template(self, security_hub):
        """Test CloudFormation template generation"""
        template = security_hub.create_cloudformation_template()

        assert "AWSTemplateFormatVersion" in template
        assert "Resources" in template
        assert "QbitelSecurityHubIntegration" in template["Resources"]

    @pytest.mark.parametrize("severity,expected_compliance", [
        ("CRITICAL", "FAILED"),
        ("HIGH", "FAILED"),
        ("MEDIUM", "WARNING"),
        ("LOW", "WARNING"),
        ("INFORMATIONAL", "WARNING")
    ])
    def test_compliance_status_mapping(self, security_hub, mock_boto_client, severity, expected_compliance):
        """Test that compliance status is correctly set based on severity"""
        mock_boto_client.batch_import_findings.return_value = {
            'FailedCount': 0,
            'SuccessCount': 1
        }

        security_hub.publish_finding(
            title="Test",
            description="Test",
            severity=severity,
            resource_id="test-123"
        )

        # Get the call arguments
        call_args = mock_boto_client.batch_import_findings.call_args
        finding = call_args[1]['Findings'][0]

        assert finding['Compliance']['Status'] == expected_compliance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
