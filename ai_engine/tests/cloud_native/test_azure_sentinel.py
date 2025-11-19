"""
Unit tests for Azure Sentinel Integration.
"""
import pytest
import json
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.cloud_integrations.azure.sentinel import (
    AzureSentinelIntegration
)


class TestAzureSentinelIntegration:
    """Test AzureSentinelIntegration class"""

    @pytest.fixture
    def sentinel(self):
        """Create Sentinel integration instance"""
        return AzureSentinelIntegration(
            workspace_id="12345678-1234-1234-1234-123456789012",
            subscription_id="87654321-4321-4321-4321-210987654321"
        )

    def test_initialization(self, sentinel):
        """Test initialization"""
        assert sentinel.workspace_id == "12345678-1234-1234-1234-123456789012"
        assert sentinel.subscription_id == "87654321-4321-4321-4321-210987654321"

    def test_send_security_event(self, sentinel):
        """Test sending security event"""
        result = sentinel.send_security_event(
            event_type="ThreatDetection",
            severity="High",
            description="Suspicious activity detected in container",
            resource_id="/subscriptions/xxx/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/vm1"
        )

        assert isinstance(result, bool)

    def test_send_event_different_severities(self, sentinel):
        """Test sending events with different severities"""
        severities = ["Informational", "Low", "Medium", "High", "Critical"]

        for severity in severities:
            result = sentinel.send_security_event(
                event_type="SecurityAlert",
                severity=severity,
                description=f"Test event with {severity} severity",
                resource_id="/subscriptions/test"
            )

            assert isinstance(result, bool)

    def test_create_arm_template(self, sentinel):
        """Test ARM template generation"""
        template = sentinel.create_arm_template()

        assert template is not None
        assert isinstance(template, dict)
        assert "$schema" in template
        assert "resources" in template

        # Check for Sentinel workspace resource
        resources = template["resources"]
        assert len(resources) > 0

    def test_send_quantum_vulnerability_event(self, sentinel):
        """Test sending quantum vulnerability event"""
        result = sentinel.send_security_event(
            event_type="QuantumVulnerability",
            severity="High",
            description="Quantum-vulnerable cryptography detected: RSA-2048",
            resource_id="/subscriptions/xxx/resourceGroups/rg/providers/Microsoft.ContainerService/managedClusters/aks1",
            additional_data={
                "vulnerable_algorithm": "RSA-2048",
                "recommended_alternative": "Kyber-1024",
                "container_image": "myapp:v1.0.0"
            }
        )

        assert isinstance(result, bool)

    def test_send_container_security_event(self, sentinel):
        """Test sending container security event"""
        result = sentinel.send_security_event(
            event_type="ContainerSecurityViolation",
            severity="Medium",
            description="Container running with privileged mode",
            resource_id="/subscriptions/xxx/providers/Microsoft.ContainerInstance/containerGroups/cg1",
            additional_data={
                "container_name": "nginx",
                "violation_type": "privileged_mode",
                "namespace": "production"
            }
        )

        assert isinstance(result, bool)

    def test_send_service_mesh_event(self, sentinel):
        """Test sending service mesh event"""
        result = sentinel.send_security_event(
            event_type="ServiceMeshSecurity",
            severity="Low",
            description="mTLS configuration updated",
            resource_id="/subscriptions/xxx/resourceGroups/rg",
            additional_data={
                "mesh_type": "Istio",
                "configuration": "strict_mtls",
                "affected_services": ["api", "backend"]
            }
        )

        assert isinstance(result, bool)

    def test_batch_send_events(self, sentinel):
        """Test sending multiple events"""
        events = [
            {
                "event_type": "ThreatDetection",
                "severity": "High",
                "description": "Event 1",
                "resource_id": "/subscriptions/test/vm1"
            },
            {
                "event_type": "Compliance",
                "severity": "Medium",
                "description": "Event 2",
                "resource_id": "/subscriptions/test/vm2"
            }
        ]

        results = []
        for event in events:
            result = sentinel.send_security_event(**event)
            results.append(result)

        assert len(results) == 2

    def test_event_with_custom_fields(self, sentinel):
        """Test event with custom fields"""
        result = sentinel.send_security_event(
            event_type="CustomAlert",
            severity="Medium",
            description="Custom security alert",
            resource_id="/subscriptions/test",
            additional_data={
                "custom_field_1": "value1",
                "custom_field_2": 123,
                "custom_field_3": ["item1", "item2"]
            }
        )

        assert isinstance(result, bool)

    def test_arm_template_parameters(self, sentinel):
        """Test ARM template with parameters"""
        template = sentinel.create_arm_template()

        # Check for parameters section
        if "parameters" in template:
            assert isinstance(template["parameters"], dict)

    def test_invalid_workspace_id(self):
        """Test initialization with invalid workspace ID"""
        # Should still initialize but might fail on actual API calls
        sentinel = AzureSentinelIntegration(
            workspace_id="invalid-id",
            subscription_id="test-sub"
        )

        assert sentinel.workspace_id == "invalid-id"
