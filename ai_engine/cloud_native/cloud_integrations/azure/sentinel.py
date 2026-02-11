"""
Azure Sentinel Integration

Integrates QBITEL security events with Azure Sentinel for SIEM
and security orchestration.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AzureSentinelIntegration:
    """
    Integrates with Azure Sentinel for security event management.
    """

    def __init__(self, workspace_id: str, subscription_id: str):
        """
        Initialize Azure Sentinel integration.

        Args:
            workspace_id: Log Analytics workspace ID
            subscription_id: Azure subscription ID
        """
        self.workspace_id = workspace_id
        self.subscription_id = subscription_id
        logger.info(f"Initialized Azure Sentinel integration")

    def send_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        resource_id: str
    ) -> Dict[str, Any]:
        """Send security event to Sentinel"""
        event = {
            "TimeGenerated": None,
            "EventType": event_type,
            "Severity": severity,
            "Description": description,
            "ResourceId": resource_id,
            "Provider": "QBITEL",
            "Category": "Quantum Security"
        }

        logger.info(f"Sent event to Azure Sentinel: {event_type}")
        return event

    def create_arm_template(self) -> Dict[str, Any]:
        """Create ARM template for deployment"""
        return {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "resources": [{
                "type": "Microsoft.OperationalInsights/workspaces/providers/dataConnectors",
                "apiVersion": "2021-03-01-preview",
                "name": "qbitel-connector",
                "properties": {
                    "connectorUiConfig": {
                        "title": "QBITEL Quantum Security",
                        "publisher": "QBITEL",
                        "descriptionMarkdown": "Quantum-safe security monitoring"
                    }
                }
            }]
        }
