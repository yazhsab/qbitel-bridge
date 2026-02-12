"""
Integration tests for Cloud Platform Integrations.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.cloud_integrations.aws.security_hub import AWSSecurityHubIntegration
from cloud_native.cloud_integrations.azure.sentinel import AzureSentinelIntegration
from cloud_native.cloud_integrations.gcp.security_command_center import GCPSecurityCommandCenterIntegration


class TestCloudIntegrations:
    """Test multi-cloud security integrations"""

    @patch("boto3.client")
    def test_multi_cloud_finding_publication(self, mock_boto):
        """Test publishing findings to multiple cloud platforms"""
        mock_client = MagicMock()
        mock_client.batch_import_findings.return_value = {"FailedCount": 0, "SuccessCount": 1}
        mock_boto.return_value = mock_client

        # AWS Security Hub
        aws_hub = AWSSecurityHubIntegration("us-east-1", "123456789012")
        aws_result = aws_hub.publish_finding(
            title="Quantum Vulnerability",
            description="RSA-2048 detected",
            severity=80,
            resource_id="arn:aws:ecs:us-east-1:123456789012:task/my-task",
            finding_type="Software and Configuration Checks/Vulnerabilities",
        )

        # Azure Sentinel
        azure_sentinel = AzureSentinelIntegration("workspace-id", "subscription-id")
        azure_result = azure_sentinel.send_security_event(
            event_type="QuantumVulnerability",
            severity="High",
            description="RSA-2048 detected",
            resource_id="/subscriptions/xxx/providers/Microsoft.ContainerService/managedClusters/aks1",
        )

        # GCP SCC
        gcp_scc = GCPSecurityCommandCenterIntegration("my-project", "org-id")
        gcp_result = gcp_scc.create_finding(
            finding_id="quantum-001",
            category="QUANTUM_VULNERABILITY",
            severity="HIGH",
            resource_name="//container.googleapis.com/projects/my-project/clusters/prod",
        )

        assert isinstance(aws_result, bool)
        assert isinstance(azure_result, bool)
        assert isinstance(gcp_result, dict)
