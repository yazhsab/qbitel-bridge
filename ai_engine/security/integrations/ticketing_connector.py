"""
CRONOS AI Engine - Ticketing System Integration Connectors

Specialized connectors for ticketing systems like ServiceNow and Jira.
"""

import asyncio
import json
import aiohttp
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod

from .base_connector import BaseIntegrationConnector, IntegrationResult, IntegrationType
from ..models import SecurityEvent, ThreatAnalysis, AutomatedResponse
from ..logging import get_security_logger, SecurityLogType, LogLevel


class TicketingConnector(BaseIntegrationConnector, ABC):
    """Base class for ticketing system connectors."""
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = get_security_logger("cronos.security.integrations.ticketing")
        
    @abstractmethod
    async def create_ticket(self, security_event: SecurityEvent) -> IntegrationResult:
        """Create a ticket in the ticketing system."""
        pass
    
    @abstractmethod
    async def update_ticket(self, ticket_id: str, update_data: Dict[str, Any]) -> IntegrationResult:
        """Update an existing ticket in the ticketing system."""
        pass
    
    @abstractmethod
    async def close_ticket(self, ticket_id: str, resolution_notes: str) -> IntegrationResult:
        """Close a ticket in the ticketing system."""
        pass
    
    @abstractmethod
    async def get_ticket_status(self, ticket_id: str) -> IntegrationResult:
        """Get the status of a ticket."""
        pass


class ServiceNowConnector(TicketingConnector):
    """ServiceNow ITSM integration connector."""
    
    def __init__(self, config):
        super().__init__(config)
        self.auth_header = None
        self.table_name = config.custom_config.get('table_name', 'incident')
        
    async def initialize(self) -> bool:
        """Initialize ServiceNow connection."""
        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Initializing ServiceNow connector",
                level=LogLevel.INFO,
                component="servicenow_connector"
            )
            
            # Set up basic authentication
            username = self.config.credentials.get('username')
            password = self.config.credentials.get('password')
            
            if username and password:
                auth_string = f"{username}:{password}"
                auth_bytes = auth_string.encode('ascii')
                auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
                self.auth_header = f"Basic {auth_b64}"
                
                # Test connection
                test_result = await self._test_connection()
                
                if test_result:
                    self.connection_status = "connected"
                    self.last_health_check = datetime.utcnow()
                    
                    self.logger.log_security_event(
                        SecurityLogType.CONFIGURATION_CHANGE,
                        "ServiceNow connector initialized successfully",
                        level=LogLevel.INFO,
                        component="servicenow_connector"
                    )
                    return True
                else:
                    self.connection_status = "failed"
                    return False
            else:
                self.connection_status = "failed"
                self.logger.log_security_event(
                    SecurityLogType.CONFIGURATION_CHANGE,
                    "ServiceNow credentials not provided",
                    level=LogLevel.ERROR,
                    component="servicenow_connector"
                )
                return False
                
        except Exception as e:
            self.connection_status = "failed"
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"ServiceNow connector initialization failed: {str(e)}",
                level=LogLevel.ERROR,
                component="servicenow_connector",
                error_code="SERVICENOW_INIT_ERROR"
            )
            return False
    
    async def _test_connection(self) -> bool:
        """Test ServiceNow connection."""
        
        test_url = f"{self.config.endpoint}/api/now/table/{self.table_name}?sysparm_limit=1"
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    test_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
                    
        except Exception:
            return False
    
    async def send_security_event(self, security_event: SecurityEvent) -> IntegrationResult:
        """Send security event to ServiceNow as a new incident."""
        
        try:
            # Create incident for high priority events
            if security_event.threat_level.value in ['critical', 'high']:
                return await self.create_ticket(security_event)
            else:
                return IntegrationResult(
                    success=True,
                    message="Event threat level too low for ticket creation",
                    response_data={'skipped': True}
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"ServiceNow event send failed: {str(e)}",
                error_code="SERVICENOW_SEND_ERROR"
            )
    
    async def send_threat_analysis(self, threat_analysis: ThreatAnalysis) -> IntegrationResult:
        """Send threat analysis to ServiceNow."""
        
        try:
            # Create or update existing incident with threat analysis
            incident_data = self._convert_threat_analysis_to_incident(threat_analysis)
            
            # Try to find existing incident for this threat
            existing_ticket = await self._find_existing_ticket(threat_analysis.event_id)
            
            if existing_ticket:
                # Update existing ticket
                return await self.update_ticket(
                    existing_ticket['sys_id'],
                    {
                        'work_notes': f"Threat Analysis Update: {incident_data['description']}",
                        'priority': incident_data['priority'],
                        'urgency': incident_data['urgency']
                    }
                )
            else:
                # Create new incident
                return await self._create_incident(incident_data)
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"ServiceNow threat analysis send failed: {str(e)}",
                error_code="SERVICENOW_THREAT_SEND_ERROR"
            )
    
    async def send_response_execution(self, automated_response: AutomatedResponse) -> IntegrationResult:
        """Send automated response execution to ServiceNow."""
        
        try:
            # Find existing ticket for this event
            existing_ticket = await self._find_existing_ticket(automated_response.event_id)
            
            if existing_ticket:
                # Update with response execution details
                work_notes = self._format_response_notes(automated_response)
                
                update_data = {
                    'work_notes': work_notes,
                    'state': '6' if automated_response.status.value == 'completed' else '2'  # Resolved or In Progress
                }
                
                if automated_response.status.value == 'completed':
                    update_data['resolution_notes'] = f"Automatically resolved by CRONOS AI Security Orchestrator. Response ID: {automated_response.response_id}"
                
                return await self.update_ticket(existing_ticket['sys_id'], update_data)
            else:
                return IntegrationResult(
                    success=True,
                    message="No existing ticket found for response execution",
                    response_data={'skipped': True}
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"ServiceNow response send failed: {str(e)}",
                error_code="SERVICENOW_RESPONSE_SEND_ERROR"
            )
    
    async def create_ticket(self, security_event: SecurityEvent) -> IntegrationResult:
        """Create a new incident ticket in ServiceNow."""
        
        incident_data = self._convert_to_incident(security_event)
        return await self._create_incident(incident_data)
    
    async def _create_incident(self, incident_data: Dict[str, Any]) -> IntegrationResult:
        """Create incident in ServiceNow."""
        
        create_url = f"{self.config.endpoint}/api/now/table/{self.table_name}"
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    create_url,
                    json=incident_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    
                    if response.status == 201:
                        response_data = await response.json()
                        incident = response_data.get('result', {})
                        
                        return IntegrationResult(
                            success=True,
                            message=f"Incident created in ServiceNow: {incident.get('number')}",
                            response_data={
                                'ticket_id': incident.get('sys_id'),
                                'ticket_number': incident.get('number'),
                                'state': incident.get('state')
                            }
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"ServiceNow incident creation failed: {response.status} - {error_text}",
                            error_code="SERVICENOW_CREATE_ERROR"
                        )
                        
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"ServiceNow incident creation error: {str(e)}",
                error_code="SERVICENOW_CREATE_ERROR"
            )
    
    async def update_ticket(self, ticket_id: str, update_data: Dict[str, Any]) -> IntegrationResult:
        """Update an existing incident in ServiceNow."""
        
        update_url = f"{self.config.endpoint}/api/now/table/{self.table_name}/{ticket_id}"
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    update_url,
                    json=update_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        incident = response_data.get('result', {})
                        
                        return IntegrationResult(
                            success=True,
                            message=f"Incident updated in ServiceNow: {incident.get('number')}",
                            response_data={
                                'ticket_id': incident.get('sys_id'),
                                'ticket_number': incident.get('number'),
                                'state': incident.get('state')
                            }
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"ServiceNow incident update failed: {response.status} - {error_text}",
                            error_code="SERVICENOW_UPDATE_ERROR"
                        )
                        
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"ServiceNow incident update error: {str(e)}",
                error_code="SERVICENOW_UPDATE_ERROR"
            )
    
    async def close_ticket(self, ticket_id: str, resolution_notes: str) -> IntegrationResult:
        """Close an incident in ServiceNow."""
        
        close_data = {
            'state': '6',  # Resolved
            'resolution_code': 'Solved (Permanently)',
            'resolution_notes': resolution_notes
        }
        
        return await self.update_ticket(ticket_id, close_data)
    
    async def get_ticket_status(self, ticket_id: str) -> IntegrationResult:
        """Get the status of an incident in ServiceNow."""
        
        status_url = f"{self.config.endpoint}/api/now/table/{self.table_name}/{ticket_id}"
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    status_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        incident = response_data.get('result', {})
                        
                        return IntegrationResult(
                            success=True,
                            message="Incident status retrieved",
                            response_data={
                                'ticket_id': incident.get('sys_id'),
                                'ticket_number': incident.get('number'),
                                'state': incident.get('state'),
                                'state_name': incident.get('state.display_value'),
                                'priority': incident.get('priority'),
                                'assigned_to': incident.get('assigned_to.display_value'),
                                'created_on': incident.get('opened_at'),
                                'updated_on': incident.get('sys_updated_on')
                            }
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"ServiceNow status retrieval failed: {response.status} - {error_text}",
                            error_code="SERVICENOW_STATUS_ERROR"
                        )
                        
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"ServiceNow status retrieval error: {str(e)}",
                error_code="SERVICENOW_STATUS_ERROR"
            )
    
    async def _find_existing_ticket(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Find existing ticket for an event ID."""
        
        search_url = f"{self.config.endpoint}/api/now/table/{self.table_name}"
        params = {
            'sysparm_query': f'correlation_id={event_id}',
            'sysparm_limit': 1
        }
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    search_url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        results = response_data.get('result', [])
                        return results[0] if results else None
                    else:
                        return None
                        
        except Exception:
            return None
    
    def _convert_to_incident(self, security_event: SecurityEvent) -> Dict[str, Any]:
        """Convert SecurityEvent to ServiceNow incident format."""
        
        priority, urgency = self._map_threat_level_to_priority(security_event.threat_level.value)
        
        return {
            'short_description': f"CRONOS AI Security Alert: {security_event.event_type.value}",
            'description': security_event.description,
            'priority': priority,
            'urgency': urgency,
            'category': 'Security',
            'subcategory': 'Security Incident',
            'caller_id': self.config.custom_config.get('default_caller', 'admin'),
            'assignment_group': self.config.custom_config.get('security_group', 'Security Operations'),
            'business_service': security_event.target_system,
            'correlation_id': security_event.event_id,
            'u_source_ip': security_event.source_ip,
            'u_destination_ip': security_event.destination_ip,
            'u_protocol': security_event.protocol,
            'u_port': security_event.port,
            'u_user': security_event.user,
            'u_threat_level': security_event.threat_level.value,
            'u_source_system': security_event.source_system,
            'work_notes': f"Automated incident created by CRONOS AI Security Orchestrator\nEvent ID: {security_event.event_id}\nTimestamp: {security_event.timestamp.isoformat()}"
        }
    
    def _convert_threat_analysis_to_incident(self, threat_analysis: ThreatAnalysis) -> Dict[str, Any]:
        """Convert ThreatAnalysis to ServiceNow incident format."""
        
        priority, urgency = self._map_threat_level_to_priority(threat_analysis.threat_level.value)
        
        iocs_text = "\n".join([
            f"- {ioc.ioc_type.value}: {ioc.value} (confidence: {ioc.confidence})"
            for ioc in threat_analysis.iocs[:5]  # Limit to first 5 IOCs
        ])
        
        description = f"""Threat Analysis Result
        
Threat Category: {threat_analysis.threat_category.value}
Confidence Score: {threat_analysis.confidence_score}
Attack Vectors: {', '.join([av.value for av in threat_analysis.attack_vectors])}
Affected Systems: {', '.join(threat_analysis.affected_systems)}

Business Impact:
- Criticality: {threat_analysis.business_impact.criticality.value}
- Affected Services: {', '.join(threat_analysis.business_impact.affected_services)}
- Estimated Cost: ${threat_analysis.business_impact.estimated_cost:,.2f}

Indicators of Compromise:
{iocs_text}

Recommended Actions: {', '.join([action.value for action in threat_analysis.recommended_actions])}
        """
        
        return {
            'short_description': f"CRONOS AI Threat Analysis: {threat_analysis.threat_category.value}",
            'description': description,
            'priority': priority,
            'urgency': urgency,
            'category': 'Security',
            'subcategory': 'Threat Analysis',
            'caller_id': self.config.custom_config.get('default_caller', 'admin'),
            'assignment_group': self.config.custom_config.get('security_group', 'Security Operations'),
            'correlation_id': threat_analysis.event_id,
            'u_threat_id': threat_analysis.threat_id,
            'u_threat_level': threat_analysis.threat_level.value,
            'u_confidence_score': threat_analysis.confidence_score
        }
    
    def _format_response_notes(self, automated_response: AutomatedResponse) -> str:
        """Format automated response for work notes."""
        
        actions_summary = []
        for action in automated_response.actions:
            status_emoji = "✅" if action.status.value == "completed" else "⏳" if action.status.value == "in_progress" else "❌"
            actions_summary.append(f"{status_emoji} {action.action_type.value} on {action.target_system}")
        
        return f"""Automated Response Executed - {automated_response.status.value.title()}

Response ID: {automated_response.response_id}
Confidence Score: {automated_response.confidence_score}
Execution Time: {automated_response.execution_timestamp.isoformat()}

Actions Taken:
{chr(10).join(actions_summary)}

{f'Rollback Plan Available: Yes (ID: {automated_response.rollback_plan.rollback_id})' if automated_response.rollback_plan else 'Rollback Plan Available: No'}
"""
    
    def _map_threat_level_to_priority(self, threat_level: str) -> tuple[str, str]:
        """Map CRONOS threat level to ServiceNow priority and urgency."""
        
        mapping = {
            'critical': ('1', '1'),  # Critical priority, High urgency
            'high': ('2', '2'),      # High priority, Medium urgency
            'medium': ('3', '2'),    # Moderate priority, Medium urgency
            'low': ('4', '3'),       # Low priority, Low urgency
            'informational': ('5', '3')  # Planning priority, Low urgency
        }
        
        return mapping.get(threat_level, ('3', '2'))
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for ServiceNow connector."""
        
        try:
            # Test basic connectivity
            test_result = await self._test_connection()
            
            if test_result:
                self.connection_status = "healthy"
                self.last_health_check = datetime.utcnow()
                
                return {
                    'status': 'healthy',
                    'message': 'ServiceNow connection is healthy',
                    'last_check': self.last_health_check.isoformat()
                }
            else:
                self.connection_status = "unhealthy"
                return {
                    'status': 'unhealthy',
                    'message': 'ServiceNow health check failed',
                    'last_check': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.connection_status = "unhealthy"
            return {
                'status': 'unhealthy',
                'message': f'ServiceNow health check error: {str(e)}',
                'last_check': datetime.utcnow().isoformat()
            }


class JiraConnector(TicketingConnector):
    """Jira/Jira Service Management integration connector."""
    
    def __init__(self, config):
        super().__init__(config)
        self.auth_header = None
        self.project_key = config.custom_config.get('project_key', 'SEC')
        self.issue_type = config.custom_config.get('issue_type', 'Incident')
        
    async def initialize(self) -> bool:
        """Initialize Jira connection."""
        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Initializing Jira connector",
                level=LogLevel.INFO,
                component="jira_connector"
            )
            
            # Set up authentication (basic auth or API token)
            username = self.config.credentials.get('username')
            api_token = self.config.credentials.get('api_token')
            
            if username and api_token:
                auth_string = f"{username}:{api_token}"
                auth_bytes = auth_string.encode('ascii')
                auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
                self.auth_header = f"Basic {auth_b64}"
                
                # Test connection
                test_result = await self._test_connection()
                
                if test_result:
                    self.connection_status = "connected"
                    self.last_health_check = datetime.utcnow()
                    
                    self.logger.log_security_event(
                        SecurityLogType.CONFIGURATION_CHANGE,
                        "Jira connector initialized successfully",
                        level=LogLevel.INFO,
                        component="jira_connector"
                    )
                    return True
                else:
                    self.connection_status = "failed"
                    return False
            else:
                self.connection_status = "failed"
                self.logger.log_security_event(
                    SecurityLogType.CONFIGURATION_CHANGE,
                    "Jira credentials not provided",
                    level=LogLevel.ERROR,
                    component="jira_connector"
                )
                return False
                
        except Exception as e:
            self.connection_status = "failed"
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Jira connector initialization failed: {str(e)}",
                level=LogLevel.ERROR,
                component="jira_connector",
                error_code="JIRA_INIT_ERROR"
            )
            return False
    
    async def _test_connection(self) -> bool:
        """Test Jira connection."""
        
        test_url = f"{self.config.endpoint}/rest/api/2/myself"
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    test_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
                    
        except Exception:
            return False
    
    async def send_security_event(self, security_event: SecurityEvent) -> IntegrationResult:
        """Send security event to Jira as a new issue."""
        
        try:
            # Create issue for high priority events
            if security_event.threat_level.value in ['critical', 'high']:
                return await self.create_ticket(security_event)
            else:
                return IntegrationResult(
                    success=True,
                    message="Event threat level too low for ticket creation",
                    response_data={'skipped': True}
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Jira event send failed: {str(e)}",
                error_code="JIRA_SEND_ERROR"
            )
    
    async def send_threat_analysis(self, threat_analysis: ThreatAnalysis) -> IntegrationResult:
        """Send threat analysis to Jira."""
        
        try:
            # Create or update existing issue with threat analysis
            issue_data = self._convert_threat_analysis_to_issue(threat_analysis)
            
            # Try to find existing issue for this threat
            existing_ticket = await self._find_existing_ticket(threat_analysis.event_id)
            
            if existing_ticket:
                # Add comment to existing issue
                return await self._add_comment(
                    existing_ticket['key'],
                    f"Threat Analysis Update:\n{issue_data['fields']['description']}"
                )
            else:
                # Create new issue
                return await self._create_issue(issue_data)
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Jira threat analysis send failed: {str(e)}",
                error_code="JIRA_THREAT_SEND_ERROR"
            )
    
    async def send_response_execution(self, automated_response: AutomatedResponse) -> IntegrationResult:
        """Send automated response execution to Jira."""
        
        try:
            # Find existing issue for this event
            existing_ticket = await self._find_existing_ticket(automated_response.event_id)
            
            if existing_ticket:
                # Add comment with response execution details
                comment_text = self._format_response_comment(automated_response)
                
                result = await self._add_comment(existing_ticket['key'], comment_text)
                
                # Close issue if response completed successfully
                if automated_response.status.value == 'completed':
                    await self._transition_issue(
                        existing_ticket['key'],
                        'Done',
                        f"Automatically resolved by CRONOS AI Security Orchestrator. Response ID: {automated_response.response_id}"
                    )
                
                return result
            else:
                return IntegrationResult(
                    success=True,
                    message="No existing ticket found for response execution",
                    response_data={'skipped': True}
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Jira response send failed: {str(e)}",
                error_code="JIRA_RESPONSE_SEND_ERROR"
            )
    
    async def create_ticket(self, security_event: SecurityEvent) -> IntegrationResult:
        """Create a new issue in Jira."""
        
        issue_data = self._convert_to_issue(security_event)
        return await self._create_issue(issue_data)
    
    async def _create_issue(self, issue_data: Dict[str, Any]) -> IntegrationResult:
        """Create issue in Jira."""
        
        create_url = f"{self.config.endpoint}/rest/api/2/issue"
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    create_url,
                    json=issue_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    
                    if response.status == 201:
                        response_data = await response.json()
                        
                        return IntegrationResult(
                            success=True,
                            message=f"Issue created in Jira: {response_data.get('key')}",
                            response_data={
                                'ticket_id': response_data.get('id'),
                                'ticket_key': response_data.get('key'),
                                'self_url': response_data.get('self')
                            }
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"Jira issue creation failed: {response.status} - {error_text}",
                            error_code="JIRA_CREATE_ERROR"
                        )
                        
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Jira issue creation error: {str(e)}",
                error_code="JIRA_CREATE_ERROR"
            )
    
    async def update_ticket(self, ticket_id: str, update_data: Dict[str, Any]) -> IntegrationResult:
        """Update an existing issue in Jira."""
        
        update_url = f"{self.config.endpoint}/rest/api/2/issue/{ticket_id}"
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    update_url,
                    json={'fields': update_data},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    
                    if response.status == 204:
                        return IntegrationResult(
                            success=True,
                            message=f"Issue updated in Jira: {ticket_id}",
                            response_data={'ticket_id': ticket_id}
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"Jira issue update failed: {response.status} - {error_text}",
                            error_code="JIRA_UPDATE_ERROR"
                        )
                        
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Jira issue update error: {str(e)}",
                error_code="JIRA_UPDATE_ERROR"
            )
    
    async def close_ticket(self, ticket_id: str, resolution_notes: str) -> IntegrationResult:
        """Close an issue in Jira."""
        
        return await self._transition_issue(ticket_id, 'Done', resolution_notes)
    
    async def _transition_issue(self, ticket_id: str, transition_name: str, comment: str) -> IntegrationResult:
        """Transition an issue to a new status."""
        
        # First, get available transitions
        transitions_url = f"{self.config.endpoint}/rest/api/2/issue/{ticket_id}/transitions"
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get available transitions
                async with session.get(
                    transitions_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        transitions_data = await response.json()
                        transitions = transitions_data.get('transitions', [])
                        
                        # Find the target transition
                        target_transition = None
                        for transition in transitions:
                            if transition['name'].lower() == transition_name.lower():
                                target_transition = transition
                                break
                        
                        if not target_transition:
                            return IntegrationResult(
                                success=False,
                                message=f"Transition '{transition_name}' not available for issue {ticket_id}",
                                error_code="JIRA_TRANSITION_NOT_AVAILABLE"
                            )
                        
                        # Execute transition
                        transition_data = {
                            'transition': {'id': target_transition['id']},
                            'update': {
                                'comment': [
                                    {
                                        'add': {
                                            'body': comment
                                        }
                                    }
                                ]
                            }
                        }
                        
                        headers['Content-Type'] = 'application/json'
                        
                        async with session.post(
                            transitions_url,
                            json=transition_data,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                        ) as transition_response:
                            
                            if transition_response.status == 204:
                                return IntegrationResult(
                                    success=True,
                                    message=f"Issue {ticket_id} transitioned to {transition_name}",
                                    response_data={'ticket_id': ticket_id, 'transition': transition_name}
                                )
                            else:
                                error_text = await transition_response.text()
                                return IntegrationResult(
                                    success=False,
                                    message=f"Jira transition failed: {transition_response.status} - {error_text}",
                                    error_code="JIRA_TRANSITION_ERROR"
                                )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"Failed to get transitions: {response.status} - {error_text}",
                            error_code="JIRA_GET_TRANSITIONS_ERROR"
                        )
                        
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Jira transition error: {str(e)}",
                error_code="JIRA_TRANSITION_ERROR"
            )
    
    async def get_ticket_status(self, ticket_id: str) -> IntegrationResult:
        """Get the status of an issue in Jira."""
        
        status_url = f"{self.config.endpoint}/rest/api/2/issue/{ticket_id}"
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    status_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        fields = response_data.get('fields', {})
                        
                        return IntegrationResult(
                            success=True,
                            message="Issue status retrieved",
                            response_data={
                                'ticket_id': response_data.get('id'),
                                'ticket_key': response_data.get('key'),
                                'status': fields.get('status', {}).get('name'),
                                'priority': fields.get('priority', {}).get('name'),
                                'assignee': fields.get('assignee', {}).get('displayName'),
                                'created': fields.get('created'),
                                'updated': fields.get('updated'),
                                'summary': fields.get('summary')
                            }
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"Jira status retrieval failed: {response.status} - {error_text}",
                            error_code="JIRA_STATUS_ERROR"
                        )
                        
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Jira status retrieval error: {str(e)}",
                error_code="JIRA_STATUS_ERROR"
            )
    
    async def _add_comment(self, ticket_key: str, comment_text: str) -> IntegrationResult:
        """Add a comment to a Jira issue."""
        
        comment_url = f"{self.config.endpoint}/rest/api/2/issue/{ticket_key}/comment"
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        comment_data = {
            'body': comment_text
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    comment_url,
                    json=comment_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    
                    if response.status == 201:
                        response_data = await response.json()
                        
                        return IntegrationResult(
                            success=True,
                            message=f"Comment added to issue {ticket_key}",
                            response_data={
                                'comment_id': response_data.get('id'),
                                'ticket_key': ticket_key
                            }
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"Jira comment addition failed: {response.status} - {error_text}",
                            error_code="JIRA_COMMENT_ERROR"
                        )
                        
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Jira comment error: {str(e)}",
                error_code="JIRA_COMMENT_ERROR"
            )
    
    async def _find_existing_ticket(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Find existing issue for an event ID."""
        
        search_url = f"{self.config.endpoint}/rest/api/2/search"
        params = {
            'jql': f'project = {self.project_key} AND labels = "cronos-event-{event_id}"',
            'maxResults': 1
        }
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    search_url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        issues = response_data.get('issues', [])
                        return issues[0] if issues else None
                    else:
                        return None
                        
        except Exception:
            return None
    
    def _convert_to_issue(self, security_event: SecurityEvent) -> Dict[str, Any]:
        """Convert SecurityEvent to Jira issue format."""
        
        priority = self._map_threat_level_to_jira_priority(security_event.threat_level.value)
        
        return {
            'fields': {
                'project': {'key': self.project_key},
                'summary': f"CRONOS AI Security Alert: {security_event.event_type.value}",
                'description': security_event.description,
                'issuetype': {'name': self.issue_type},
                'priority': {'name': priority},
                'labels': [
                    'cronos-ai',
                    f'cronos-event-{security_event.event_id}',
                    f'threat-{security_event.threat_level.value}',
                    'security-incident'
                ],
                'components': [{'name': 'Security'}] if self.config.custom_config.get('security_component') else [],
                'customfield_10000': security_event.event_id,  # Custom field for event ID
            }
        }
    
    def _convert_threat_analysis_to_issue(self, threat_analysis: ThreatAnalysis) -> Dict[str, Any]:
        """Convert ThreatAnalysis to Jira issue format."""
        
        priority = self._map_threat_level_to_jira_priority(threat_analysis.threat_level.value)
        
        iocs_text = "\n".join([
            f"• {ioc.ioc_type.value}: {ioc.value} (confidence: {ioc.confidence})"
            for ioc in threat_analysis.iocs[:5]  # Limit to first 5 IOCs
        ])
        
        description = f"""h3. Threat Analysis Result
        
*Threat Category:* {threat_analysis.threat_category.value}
*Confidence Score:* {threat_analysis.confidence_score}
*Attack Vectors:* {', '.join([av.value for av in threat_analysis.attack_vectors])}
*Affected Systems:* {', '.join(threat_analysis.affected_systems)}

h4. Business Impact
* *Criticality:* {threat_analysis.business_impact.criticality.value}
* *Affected Services:* {', '.join(threat_analysis.business_impact.affected_services)}
* *Estimated Cost:* ${threat_analysis.business_impact.estimated_cost:,.2f}

h4. Indicators of Compromise
{iocs_text}

*Recommended Actions:* {', '.join([action.value for action in threat_analysis.recommended_actions])}
        """
        
        return {
            'fields': {
                'project': {'key': self.project_key},
                'summary': f"CRONOS AI Threat Analysis: {threat_analysis.threat_category.value}",
                'description': description,
                'issuetype': {'name': self.issue_type},
                'priority': {'name': priority},
                'labels': [
                    'cronos-ai',
                    f'cronos-event-{threat_analysis.event_id}',
                    f'threat-{threat_analysis.threat_level.value}',
                    'threat-analysis'
                ],
                'components': [{'name': 'Security'}] if self.config.custom_config.get('security_component') else [],
                'customfield_10000': threat_analysis.event_id,  # Custom field for event ID
                'customfield_10001': threat_analysis.threat_id,  # Custom field for threat ID
            }
        }
    
    def _format_response_comment(self, automated_response: AutomatedResponse) -> str:
        """Format automated response for Jira comment."""
        
        actions_summary = []
        for action in automated_response.actions:
            status_icon = "(/) " if action.status.value == "completed" else "(i) " if action.status.value == "in_progress" else "(x) "
            actions_summary.append(f"{status_icon}{action.action_type.value} on {action.target_system}")
        
        return f"""h4. Automated Response Executed - {automated_response.status.value.title()}

*Response ID:* {automated_response.response_id}
*Confidence Score:* {automated_response.confidence_score}
*Execution Time:* {automated_response.execution_timestamp.isoformat()}

h5. Actions Taken:
{chr(10).join(actions_summary)}

*Rollback Plan Available:* {"Yes" if automated_response.rollback_plan else "No"}{f" (ID: {automated_response.rollback_plan.rollback_id})" if automated_response.rollback_plan else ""}
"""
    
    def _map_threat_level_to_jira_priority(self, threat_level: str) -> str:
        """Map CRONOS threat level to Jira priority."""
        
        mapping = {
            'critical': 'Highest',
            'high': 'High',
            'medium': 'Medium',
            'low': 'Low',
            'informational': 'Lowest'
        }
        
        return mapping.get(threat_level, 'Medium')
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for Jira connector."""
        
        try:
            # Test basic connectivity
            test_result = await self._test_connection()
            
            if test_result:
                self.connection_status = "healthy"
                self.last_health_check = datetime.utcnow()
                
                return {
                    'status': 'healthy',
                    'message': 'Jira connection is healthy',
                    'last_check': self.last_health_check.isoformat()
                }
            else:
                self.connection_status = "unhealthy"
                return {
                    'status': 'unhealthy',
                    'message': 'Jira health check failed',
                    'last_check': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.connection_status = "unhealthy"
            return {
                'status': 'unhealthy',
                'message': f'Jira health check error: {str(e)}',
                'last_check': datetime.utcnow().isoformat()
            }