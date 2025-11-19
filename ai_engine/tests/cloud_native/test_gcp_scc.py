"""
Unit tests for GCP Security Command Center Integration.
"""
import pytest
import json
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.cloud_integrations.gcp.security_command_center import (
    GCPSecurityCommandCenterIntegration
)


class TestGCPSecurityCommandCenterIntegration:
    """Test GCPSecurityCommandCenterIntegration class"""

    @pytest.fixture
    def scc(self):
        """Create SCC integration instance"""
        return GCPSecurityCommandCenterIntegration(
            project_id="my-gcp-project",
            organization_id="123456789012"
        )

    def test_initialization(self, scc):
        """Test initialization"""
        assert scc.project_id == "my-gcp-project"
        assert scc.organization_id == "123456789012"

    def test_create_finding(self, scc):
        """Test creating security finding"""
        result = scc.create_finding(
            finding_id="test-finding-001",
            category="VULNERABILITY",
            severity="HIGH",
            resource_name="//container.googleapis.com/projects/my-project/zones/us-central1-a/clusters/my-cluster"
        )

        assert isinstance(result, dict)

    def test_create_finding_different_severities(self, scc):
        """Test creating findings with different severities"""
        severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

        for severity in severities:
            result = scc.create_finding(
                finding_id=f"test-{severity.lower()}",
                category="VULNERABILITY",
                severity=severity,
                resource_name="//compute.googleapis.com/projects/test/zones/us-central1-a/instances/vm1"
            )

            assert isinstance(result, dict)
            if "severity" in result:
                assert result["severity"] == severity

    def test_create_quantum_vulnerability_finding(self, scc):
        """Test creating quantum vulnerability finding"""
        result = scc.create_finding(
            finding_id="quantum-vuln-001",
            category="QUANTUM_VULNERABILITY",
            severity="HIGH",
            resource_name="//container.googleapis.com/projects/my-project/clusters/prod-cluster",
            additional_properties={
                "vulnerable_algorithm": "RSA-2048",
                "recommended_alternative": "Kyber-1024",
                "affected_containers": ["app:v1.0", "api:v2.0"]
            }
        )

        assert isinstance(result, dict)

    def test_create_container_security_finding(self, scc):
        """Test creating container security finding"""
        result = scc.create_finding(
            finding_id="container-sec-001",
            category="CONTAINER_SECURITY",
            severity="MEDIUM",
            resource_name="//container.googleapis.com/projects/my-project/zones/us-central1-a/clusters/my-cluster/pods/nginx-pod",
            additional_properties={
                "violation_type": "privileged_container",
                "container_name": "nginx",
                "namespace": "production",
                "image": "nginx:1.21"
            }
        )

        assert isinstance(result, dict)

    def test_create_service_mesh_finding(self, scc):
        """Test creating service mesh security finding"""
        result = scc.create_finding(
            finding_id="mesh-sec-001",
            category="SERVICE_MESH_SECURITY",
            severity="LOW",
            resource_name="//container.googleapis.com/projects/my-project/clusters/prod",
            additional_properties={
                "mesh_type": "Istio",
                "issue": "mTLS not enforced",
                "affected_services": ["api", "backend", "frontend"]
            }
        )

        assert isinstance(result, dict)

    def test_create_deployment_manager_template(self, scc):
        """Test Deployment Manager template generation"""
        template = scc.create_deployment_manager_template()

        assert template is not None
        assert isinstance(template, dict)
        assert "resources" in template

        # Check for SCC resources
        resources = template["resources"]
        assert len(resources) > 0

    def test_create_finding_with_source_properties(self, scc):
        """Test creating finding with source properties"""
        result = scc.create_finding(
            finding_id="detailed-finding-001",
            category="VULNERABILITY",
            severity="HIGH",
            resource_name="//compute.googleapis.com/test",
            additional_properties={
                "cve": "CVE-2024-1234",
                "cvss_score": 8.5,
                "affected_package": "openssl",
                "fixed_version": "1.1.1w"
            }
        )

        assert isinstance(result, dict)

    def test_create_multiple_findings(self, scc):
        """Test creating multiple findings"""
        finding_ids = ["finding-001", "finding-002", "finding-003"]

        results = []
        for fid in finding_ids:
            result = scc.create_finding(
                finding_id=fid,
                category="VULNERABILITY",
                severity="MEDIUM",
                resource_name="//test"
            )
            results.append(result)

        assert len(results) == 3

    def test_finding_categories(self, scc):
        """Test different finding categories"""
        categories = [
            "VULNERABILITY",
            "MISCONFIGURATION",
            "THREAT",
            "COMPLIANCE",
            "ANOMALY"
        ]

        for category in categories:
            result = scc.create_finding(
                finding_id=f"test-{category.lower()}",
                category=category,
                severity="MEDIUM",
                resource_name="//test"
            )

            assert isinstance(result, dict)

    def test_resource_name_formats(self, scc):
        """Test different GCP resource name formats"""
        resource_names = [
            "//compute.googleapis.com/projects/test/zones/us-central1-a/instances/vm1",
            "//container.googleapis.com/projects/test/zones/us-central1-a/clusters/cluster1",
            "//storage.googleapis.com/projects/test/buckets/my-bucket",
            "//cloudkms.googleapis.com/projects/test/locations/global/keyRings/ring1"
        ]

        for resource_name in resource_names:
            result = scc.create_finding(
                finding_id=f"test-{len(resource_name)}",
                category="VULNERABILITY",
                severity="LOW",
                resource_name=resource_name
            )

            assert isinstance(result, dict)

    def test_deployment_manager_template_structure(self, scc):
        """Test Deployment Manager template structure"""
        template = scc.create_deployment_manager_template()

        # Should have standard Deployment Manager structure
        assert "resources" in template

        if "imports" in template:
            assert isinstance(template["imports"], list)

    def test_finding_with_event_time(self, scc):
        """Test creating finding with event time"""
        result = scc.create_finding(
            finding_id="timed-finding-001",
            category="THREAT",
            severity="HIGH",
            resource_name="//test",
            additional_properties={
                "event_time": "2024-01-01T00:00:00Z",
                "create_time": "2024-01-01T00:01:00Z"
            }
        )

        assert isinstance(result, dict)
