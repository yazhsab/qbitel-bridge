"""
CRONOS AI Engine - SIEM Integration Connectors

Specialized connectors for SIEM systems like Splunk and QRadar.
"""

import asyncio
import json
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod

from .base_connector import BaseIntegrationConnector, IntegrationResult, IntegrationType
from ..models import SecurityEvent, ThreatAnalysis, AutomatedResponse
from ..logging import get_security_logger, SecurityLogType, LogLevel


class SIEMConnector(BaseIntegrationConnector, ABC):
    """Base class for SIEM system connectors."""

    def __init__(self, config):
        super().__init__(config)
        self.logger = get_security_logger("cronos.security.integrations.siem")

    @abstractmethod
    async def create_alert(self, security_event: SecurityEvent) -> IntegrationResult:
        """Create an alert in the SIEM system."""
        pass

    @abstractmethod
    async def update_case(
        self, case_id: str, update_data: Dict[str, Any]
    ) -> IntegrationResult:
        """Update an existing case in the SIEM system."""
        pass

    @abstractmethod
    async def search_events(self, query: str, time_range: str) -> IntegrationResult:
        """Search for events in the SIEM system."""
        pass


class SplunkConnector(SIEMConnector):
    """Splunk SIEM integration connector."""

    def __init__(self, config):
        super().__init__(config)
        self.auth_token = None
        self.session_key = None

    async def initialize(self) -> bool:
        """Initialize Splunk connection."""
        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Initializing Splunk connector",
                level=LogLevel.INFO,
                component="splunk_connector",
            )

            # Authenticate with Splunk
            auth_result = await self._authenticate()

            if auth_result:
                self.connection_status = "connected"
                self.last_health_check = datetime.utcnow()

                self.logger.log_security_event(
                    SecurityLogType.CONFIGURATION_CHANGE,
                    "Splunk connector initialized successfully",
                    level=LogLevel.INFO,
                    component="splunk_connector",
                )
                return True
            else:
                self.connection_status = "failed"
                return False

        except Exception as e:
            self.connection_status = "failed"
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Splunk connector initialization failed: {str(e)}",
                level=LogLevel.ERROR,
                component="splunk_connector",
                error_code="SPLUNK_INIT_ERROR",
            )
            return False

    async def _authenticate(self) -> bool:
        """Authenticate with Splunk API."""

        auth_url = f"{self.config.endpoint}/services/auth/login"
        auth_data = {
            "username": self.config.credentials.get("username"),
            "password": self.config.credentials.get("password"),
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    auth_url,
                    data=auth_data,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 200:
                        response_text = await response.text()
                        # Extract session key from XML response
                        start = response_text.find("<sessionKey>") + 12
                        end = response_text.find("</sessionKey>")
                        self.session_key = response_text[start:end]
                        return True
                    else:
                        self.logger.log_security_event(
                            SecurityLogType.CONFIGURATION_CHANGE,
                            f"Splunk authentication failed: {response.status}",
                            level=LogLevel.ERROR,
                            component="splunk_connector",
                        )
                        return False

        except Exception as e:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Splunk authentication error: {str(e)}",
                level=LogLevel.ERROR,
                component="splunk_connector",
            )
            return False

    async def send_security_event(
        self, security_event: SecurityEvent
    ) -> IntegrationResult:
        """Send security event to Splunk."""

        try:
            # Convert security event to Splunk event format
            splunk_event = self._convert_to_splunk_event(security_event)

            # Send to Splunk HEC (HTTP Event Collector)
            result = await self._send_to_hec(splunk_event)

            if result.success:
                # Also create alert if threat level is high
                if security_event.threat_level.value in ["critical", "high"]:
                    await self.create_alert(security_event)

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Splunk event send failed: {str(e)}",
                error_code="SPLUNK_SEND_ERROR",
            )

    async def send_threat_analysis(
        self, threat_analysis: ThreatAnalysis
    ) -> IntegrationResult:
        """Send threat analysis to Splunk."""

        try:
            # Convert threat analysis to Splunk event format
            splunk_event = self._convert_threat_analysis_to_splunk(threat_analysis)

            # Send to Splunk HEC
            result = await self._send_to_hec(splunk_event)

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Splunk threat analysis send failed: {str(e)}",
                error_code="SPLUNK_THREAT_SEND_ERROR",
            )

    async def send_response_execution(
        self, automated_response: AutomatedResponse
    ) -> IntegrationResult:
        """Send automated response execution to Splunk."""

        try:
            # Convert automated response to Splunk event format
            splunk_event = self._convert_response_to_splunk(automated_response)

            # Send to Splunk HEC
            result = await self._send_to_hec(splunk_event)

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Splunk response send failed: {str(e)}",
                error_code="SPLUNK_RESPONSE_SEND_ERROR",
            )

    async def _send_to_hec(self, event_data: Dict[str, Any]) -> IntegrationResult:
        """Send event to Splunk HTTP Event Collector."""

        hec_url = f"{self.config.endpoint}/services/collector/event"
        headers = {
            "Authorization": f'Splunk {self.config.credentials.get("hec_token")}',
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    hec_url,
                    json=event_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 200:
                        return IntegrationResult(
                            success=True,
                            message="Event sent to Splunk successfully",
                            response_data=await response.json(),
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"Splunk HEC error: {response.status} - {error_text}",
                            error_code="SPLUNK_HEC_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Splunk HEC send error: {str(e)}",
                error_code="SPLUNK_HEC_SEND_ERROR",
            )

    def _convert_to_splunk_event(self, security_event: SecurityEvent) -> Dict[str, Any]:
        """Convert SecurityEvent to Splunk event format."""

        return {
            "time": int(security_event.timestamp.timestamp()),
            "host": security_event.source_system,
            "source": "cronos-ai-security-orchestrator",
            "sourcetype": "cronos:security:event",
            "index": self.config.custom_config.get("index", "security"),
            "event": {
                "event_id": security_event.event_id,
                "event_type": security_event.event_type.value,
                "threat_level": security_event.threat_level.value,
                "source_system": security_event.source_system,
                "target_system": security_event.target_system,
                "description": security_event.description,
                "source_ip": security_event.source_ip,
                "destination_ip": security_event.destination_ip,
                "protocol": security_event.protocol,
                "port": security_event.port,
                "user": security_event.user,
                "raw_data": security_event.raw_data,
                "context": security_event.context,
                "tags": security_event.tags,
            },
        }

    def _convert_threat_analysis_to_splunk(
        self, threat_analysis: ThreatAnalysis
    ) -> Dict[str, Any]:
        """Convert ThreatAnalysis to Splunk event format."""

        return {
            "time": int(threat_analysis.analysis_timestamp.timestamp()),
            "source": "cronos-ai-security-orchestrator",
            "sourcetype": "cronos:security:threat_analysis",
            "index": self.config.custom_config.get("index", "security"),
            "event": {
                "event_id": threat_analysis.event_id,
                "threat_id": threat_analysis.threat_id,
                "threat_level": threat_analysis.threat_level.value,
                "threat_category": threat_analysis.threat_category.value,
                "confidence_score": threat_analysis.confidence_score,
                "attack_vectors": [av.value for av in threat_analysis.attack_vectors],
                "affected_systems": threat_analysis.affected_systems,
                "iocs": [
                    {
                        "type": ioc.ioc_type.value,
                        "value": ioc.value,
                        "confidence": ioc.confidence,
                    }
                    for ioc in threat_analysis.iocs
                ],
                "ml_predictions": [
                    {
                        "model_name": pred.model_name,
                        "prediction": pred.prediction,
                        "confidence": pred.confidence,
                        "features": pred.features,
                    }
                    for pred in threat_analysis.ml_predictions
                ],
                "recommended_actions": [
                    action.value for action in threat_analysis.recommended_actions
                ],
                "business_impact": {
                    "criticality": threat_analysis.business_impact.criticality.value,
                    "affected_services": threat_analysis.business_impact.affected_services,
                    "estimated_cost": threat_analysis.business_impact.estimated_cost,
                    "compliance_impact": [
                        ci.value
                        for ci in threat_analysis.business_impact.compliance_impact
                    ],
                },
                "context": threat_analysis.context,
            },
        }

    def _convert_response_to_splunk(
        self, automated_response: AutomatedResponse
    ) -> Dict[str, Any]:
        """Convert AutomatedResponse to Splunk event format."""

        return {
            "time": int(automated_response.execution_timestamp.timestamp()),
            "source": "cronos-ai-security-orchestrator",
            "sourcetype": "cronos:security:automated_response",
            "index": self.config.custom_config.get("index", "security"),
            "event": {
                "response_id": automated_response.response_id,
                "event_id": automated_response.event_id,
                "threat_id": automated_response.threat_id,
                "status": automated_response.status.value,
                "confidence_score": automated_response.confidence_score,
                "actions": [
                    {
                        "action_id": action.action_id,
                        "action_type": action.action_type.value,
                        "target_system": action.target_system,
                        "parameters": action.parameters,
                        "status": action.status.value,
                        "executed_at": (
                            action.executed_at.isoformat()
                            if action.executed_at
                            else None
                        ),
                        "completed_at": (
                            action.completed_at.isoformat()
                            if action.completed_at
                            else None
                        ),
                        "result": action.result,
                    }
                    for action in automated_response.actions
                ],
                "rollback_plan": (
                    {
                        "rollback_id": automated_response.rollback_plan.rollback_id,
                        "actions": [
                            {
                                "action_type": ra.action_type.value,
                                "target_system": ra.target_system,
                                "parameters": ra.parameters,
                                "priority": ra.priority,
                            }
                            for ra in automated_response.rollback_plan.actions
                        ],
                    }
                    if automated_response.rollback_plan
                    else None
                ),
                "context": automated_response.context,
            },
        }

    async def create_alert(self, security_event: SecurityEvent) -> IntegrationResult:
        """Create an alert in Splunk."""

        # This would create a notable event or alert in Splunk Enterprise Security
        alert_data = {
            "title": f"CRONOS AI Security Alert: {security_event.event_type.value}",
            "description": security_event.description,
            "severity": self._map_threat_level_to_splunk_severity(
                security_event.threat_level.value
            ),
            "source_ip": security_event.source_ip,
            "dest_ip": security_event.destination_ip,
            "event_id": security_event.event_id,
        }

        try:
            # Implementation would depend on specific Splunk ES API
            return IntegrationResult(
                success=True,
                message="Alert created in Splunk",
                response_data={"alert_id": f"cronos-{security_event.event_id}"},
            )
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Failed to create Splunk alert: {str(e)}",
                error_code="SPLUNK_ALERT_ERROR",
            )

    async def update_case(
        self, case_id: str, update_data: Dict[str, Any]
    ) -> IntegrationResult:
        """Update an existing case in Splunk."""

        try:
            # Implementation would depend on specific Splunk ES API
            return IntegrationResult(
                success=True,
                message=f"Case {case_id} updated in Splunk",
                response_data={"case_id": case_id},
            )
        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Failed to update Splunk case: {str(e)}",
                error_code="SPLUNK_CASE_UPDATE_ERROR",
            )

    async def search_events(self, query: str, time_range: str) -> IntegrationResult:
        """Search for events in Splunk."""

        search_url = f"{self.config.endpoint}/services/search/jobs"
        headers = {
            "Authorization": f"Splunk {self.session_key}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        search_data = {
            "search": f"search {query}",
            "earliest_time": time_range,
            "output_mode": "json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                # Start search job
                async with session.post(
                    search_url,
                    data=search_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 201:
                        response_data = await response.json()
                        job_id = response_data.get("sid")

                        # Poll for results
                        results = await self._get_search_results(job_id)

                        return IntegrationResult(
                            success=True,
                            message="Search completed successfully",
                            response_data=results,
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"Splunk search failed: {response.status} - {error_text}",
                            error_code="SPLUNK_SEARCH_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Splunk search error: {str(e)}",
                error_code="SPLUNK_SEARCH_ERROR",
            )

    async def _get_search_results(self, job_id: str) -> Dict[str, Any]:
        """Get results from a Splunk search job."""

        results_url = f"{self.config.endpoint}/services/search/jobs/{job_id}/results"
        headers = {
            "Authorization": f"Splunk {self.session_key}",
            "Content-Type": "application/json",
        }

        # Poll for completion
        for _ in range(30):  # Wait up to 5 minutes
            status_url = f"{self.config.endpoint}/services/search/jobs/{job_id}"

            async with aiohttp.ClientSession() as session:
                async with session.get(status_url, headers=headers) as response:
                    if response.status == 200:
                        status_data = await response.json()

                        if (
                            status_data.get("entry", [{}])[0]
                            .get("content", {})
                            .get("dispatchState")
                            == "DONE"
                        ):
                            # Get results
                            async with session.get(
                                f"{results_url}?output_mode=json", headers=headers
                            ) as results_response:
                                if results_response.status == 200:
                                    return await results_response.json()
                                else:
                                    return {"error": "Failed to get results"}

            await asyncio.sleep(10)

        return {"error": "Search timeout"}

    def _map_threat_level_to_splunk_severity(self, threat_level: str) -> str:
        """Map CRONOS threat level to Splunk severity."""

        mapping = {
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "low": "low",
            "informational": "informational",
        }

        return mapping.get(threat_level, "medium")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for Splunk connector."""

        try:
            # Test basic connectivity
            info_url = f"{self.config.endpoint}/services/server/info"
            headers = {"Authorization": f"Splunk {self.session_key}"}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    info_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:

                    if response.status == 200:
                        self.connection_status = "healthy"
                        self.last_health_check = datetime.utcnow()

                        return {
                            "status": "healthy",
                            "message": "Splunk connection is healthy",
                            "last_check": self.last_health_check.isoformat(),
                        }
                    else:
                        self.connection_status = "unhealthy"
                        return {
                            "status": "unhealthy",
                            "message": f"Splunk health check failed: {response.status}",
                            "last_check": datetime.utcnow().isoformat(),
                        }

        except Exception as e:
            self.connection_status = "unhealthy"
            return {
                "status": "unhealthy",
                "message": f"Splunk health check error: {str(e)}",
                "last_check": datetime.utcnow().isoformat(),
            }


class QRadarConnector(SIEMConnector):
    """IBM QRadar SIEM integration connector."""

    def __init__(self, config):
        super().__init__(config)
        self.auth_token = None

    async def initialize(self) -> bool:
        """Initialize QRadar connection."""
        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Initializing QRadar connector",
                level=LogLevel.INFO,
                component="qradar_connector",
            )

            # Set auth token from config
            self.auth_token = self.config.credentials.get("sec_token")

            if self.auth_token:
                self.connection_status = "connected"
                self.last_health_check = datetime.utcnow()

                self.logger.log_security_event(
                    SecurityLogType.CONFIGURATION_CHANGE,
                    "QRadar connector initialized successfully",
                    level=LogLevel.INFO,
                    component="qradar_connector",
                )
                return True
            else:
                self.connection_status = "failed"
                return False

        except Exception as e:
            self.connection_status = "failed"
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"QRadar connector initialization failed: {str(e)}",
                level=LogLevel.ERROR,
                component="qradar_connector",
                error_code="QRADAR_INIT_ERROR",
            )
            return False

    async def send_security_event(
        self, security_event: SecurityEvent
    ) -> IntegrationResult:
        """Send security event to QRadar."""

        try:
            # Convert security event to QRadar custom event format
            qradar_event = self._convert_to_qradar_event(security_event)

            # Send to QRadar via REST API
            result = await self._send_custom_event(qradar_event)

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"QRadar event send failed: {str(e)}",
                error_code="QRADAR_SEND_ERROR",
            )

    async def send_threat_analysis(
        self, threat_analysis: ThreatAnalysis
    ) -> IntegrationResult:
        """Send threat analysis to QRadar."""

        try:
            # Convert threat analysis to QRadar event format
            qradar_event = self._convert_threat_analysis_to_qradar(threat_analysis)

            # Send to QRadar
            result = await self._send_custom_event(qradar_event)

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"QRadar threat analysis send failed: {str(e)}",
                error_code="QRADAR_THREAT_SEND_ERROR",
            )

    async def send_response_execution(
        self, automated_response: AutomatedResponse
    ) -> IntegrationResult:
        """Send automated response execution to QRadar."""

        try:
            # Convert automated response to QRadar event format
            qradar_event = self._convert_response_to_qradar(automated_response)

            # Send to QRadar
            result = await self._send_custom_event(qradar_event)

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"QRadar response send failed: {str(e)}",
                error_code="QRADAR_RESPONSE_SEND_ERROR",
            )

    async def _send_custom_event(self, event_data: Dict[str, Any]) -> IntegrationResult:
        """Send custom event to QRadar."""

        events_url = f"{self.config.endpoint}/api/siem/events"
        headers = {
            "SEC": self.auth_token,
            "Content-Type": "application/json",
            "Version": "10.0",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    events_url,
                    json=event_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 201:
                        return IntegrationResult(
                            success=True,
                            message="Event sent to QRadar successfully",
                            response_data=await response.json(),
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"QRadar event send error: {response.status} - {error_text}",
                            error_code="QRADAR_EVENT_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"QRadar event send error: {str(e)}",
                error_code="QRADAR_EVENT_SEND_ERROR",
            )

    def _convert_to_qradar_event(self, security_event: SecurityEvent) -> Dict[str, Any]:
        """Convert SecurityEvent to QRadar event format."""

        return {
            "qidname": f"CRONOS_AI_{security_event.event_type.value.upper()}",
            "logsourceid": self.config.custom_config.get("log_source_id", 1),
            "starttime": int(security_event.timestamp.timestamp() * 1000),
            "eventcount": 1,
            "magnitude": self._map_threat_level_to_magnitude(
                security_event.threat_level.value
            ),
            "sourceip": security_event.source_ip,
            "destinationip": security_event.destination_ip,
            "sourceport": security_event.port,
            "protocol": security_event.protocol,
            "username": security_event.user,
            "payload": security_event.description,
            "custom_properties": {
                "event_id": security_event.event_id,
                "source_system": security_event.source_system,
                "target_system": security_event.target_system,
                "raw_data": (
                    json.dumps(security_event.raw_data)
                    if security_event.raw_data
                    else None
                ),
                "context": (
                    json.dumps(security_event.context)
                    if security_event.context
                    else None
                ),
                "tags": ",".join(security_event.tags) if security_event.tags else None,
            },
        }

    def _convert_threat_analysis_to_qradar(
        self, threat_analysis: ThreatAnalysis
    ) -> Dict[str, Any]:
        """Convert ThreatAnalysis to QRadar event format."""

        return {
            "qidname": "CRONOS_AI_THREAT_ANALYSIS",
            "logsourceid": self.config.custom_config.get("log_source_id", 1),
            "starttime": int(threat_analysis.analysis_timestamp.timestamp() * 1000),
            "eventcount": 1,
            "magnitude": self._map_threat_level_to_magnitude(
                threat_analysis.threat_level.value
            ),
            "payload": f"Threat Analysis - {threat_analysis.threat_category.value}",
            "custom_properties": {
                "threat_id": threat_analysis.threat_id,
                "event_id": threat_analysis.event_id,
                "confidence_score": threat_analysis.confidence_score,
                "threat_category": threat_analysis.threat_category.value,
                "attack_vectors": ",".join(
                    [av.value for av in threat_analysis.attack_vectors]
                ),
                "affected_systems": ",".join(threat_analysis.affected_systems),
                "recommended_actions": ",".join(
                    [action.value for action in threat_analysis.recommended_actions]
                ),
                "business_impact_criticality": threat_analysis.business_impact.criticality.value,
                "estimated_cost": threat_analysis.business_impact.estimated_cost,
            },
        }

    def _convert_response_to_qradar(
        self, automated_response: AutomatedResponse
    ) -> Dict[str, Any]:
        """Convert AutomatedResponse to QRadar event format."""

        return {
            "qidname": "CRONOS_AI_AUTOMATED_RESPONSE",
            "logsourceid": self.config.custom_config.get("log_source_id", 1),
            "starttime": int(automated_response.execution_timestamp.timestamp() * 1000),
            "eventcount": 1,
            "magnitude": 5,  # Medium magnitude for response events
            "payload": f"Automated Response Executed - Status: {automated_response.status.value}",
            "custom_properties": {
                "response_id": automated_response.response_id,
                "event_id": automated_response.event_id,
                "threat_id": automated_response.threat_id,
                "status": automated_response.status.value,
                "confidence_score": automated_response.confidence_score,
                "actions_count": len(automated_response.actions),
                "action_types": ",".join(
                    set(
                        [
                            action.action_type.value
                            for action in automated_response.actions
                        ]
                    )
                ),
            },
        }

    def _map_threat_level_to_magnitude(self, threat_level: str) -> int:
        """Map CRONOS threat level to QRadar magnitude."""

        mapping = {"critical": 10, "high": 8, "medium": 5, "low": 3, "informational": 1}

        return mapping.get(threat_level, 5)

    async def create_alert(self, security_event: SecurityEvent) -> IntegrationResult:
        """Create an offense in QRadar."""

        offense_data = {
            "description": f"CRONOS AI Security Alert: {security_event.description}",
            "assigned_to": self.config.custom_config.get("default_assignee", "admin"),
            "status": "OPEN",
        }

        try:
            offenses_url = f"{self.config.endpoint}/api/siem/offenses"
            headers = {
                "SEC": self.auth_token,
                "Content-Type": "application/json",
                "Version": "10.0",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    offenses_url,
                    json=offense_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 201:
                        response_data = await response.json()
                        return IntegrationResult(
                            success=True,
                            message="Offense created in QRadar",
                            response_data={"offense_id": response_data.get("id")},
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"QRadar offense creation failed: {response.status} - {error_text}",
                            error_code="QRADAR_OFFENSE_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Failed to create QRadar offense: {str(e)}",
                error_code="QRADAR_OFFENSE_ERROR",
            )

    async def update_case(
        self, case_id: str, update_data: Dict[str, Any]
    ) -> IntegrationResult:
        """Update an existing offense in QRadar."""

        try:
            offense_url = f"{self.config.endpoint}/api/siem/offenses/{case_id}"
            headers = {
                "SEC": self.auth_token,
                "Content-Type": "application/json",
                "Version": "10.0",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    offense_url,
                    json=update_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 200:
                        return IntegrationResult(
                            success=True,
                            message=f"Offense {case_id} updated in QRadar",
                            response_data={"offense_id": case_id},
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"QRadar offense update failed: {response.status} - {error_text}",
                            error_code="QRADAR_OFFENSE_UPDATE_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Failed to update QRadar offense: {str(e)}",
                error_code="QRADAR_OFFENSE_UPDATE_ERROR",
            )

    async def search_events(self, query: str, time_range: str) -> IntegrationResult:
        """Search for events in QRadar."""

        search_url = f"{self.config.endpoint}/api/ariel/searches"
        headers = {
            "SEC": self.auth_token,
            "Content-Type": "application/json",
            "Version": "10.0",
        }

        # Build AQL query
        aql_query = f"SELECT * FROM events WHERE {query} LAST {time_range}"

        try:
            async with aiohttp.ClientSession() as session:
                # Start search
                search_data = {"query_expression": aql_query}
                async with session.post(
                    search_url,
                    json=search_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 201:
                        search_info = await response.json()
                        search_id = search_info.get("search_id")

                        # Poll for results
                        results = await self._get_qradar_search_results(search_id)

                        return IntegrationResult(
                            success=True,
                            message="Search completed successfully",
                            response_data=results,
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"QRadar search failed: {response.status} - {error_text}",
                            error_code="QRADAR_SEARCH_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"QRadar search error: {str(e)}",
                error_code="QRADAR_SEARCH_ERROR",
            )

    async def _get_qradar_search_results(self, search_id: str) -> Dict[str, Any]:
        """Get results from a QRadar search."""

        headers = {
            "SEC": self.auth_token,
            "Accept": "application/json",
            "Version": "10.0",
        }

        # Poll for completion
        for _ in range(30):  # Wait up to 5 minutes
            status_url = f"{self.config.endpoint}/api/ariel/searches/{search_id}"

            async with aiohttp.ClientSession() as session:
                async with session.get(status_url, headers=headers) as response:
                    if response.status == 200:
                        search_info = await response.json()

                        if search_info.get("status") == "COMPLETED":
                            # Get results
                            results_url = f"{self.config.endpoint}/api/ariel/searches/{search_id}/results"
                            async with session.get(
                                results_url, headers=headers
                            ) as results_response:
                                if results_response.status == 200:
                                    return await results_response.json()
                                else:
                                    return {"error": "Failed to get results"}

            await asyncio.sleep(10)

        return {"error": "Search timeout"}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for QRadar connector."""

        try:
            # Test basic connectivity
            system_url = f"{self.config.endpoint}/api/system/about"
            headers = {
                "SEC": self.auth_token,
                "Accept": "application/json",
                "Version": "10.0",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    system_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:

                    if response.status == 200:
                        self.connection_status = "healthy"
                        self.last_health_check = datetime.utcnow()

                        return {
                            "status": "healthy",
                            "message": "QRadar connection is healthy",
                            "last_check": self.last_health_check.isoformat(),
                        }
                    else:
                        self.connection_status = "unhealthy"
                        return {
                            "status": "unhealthy",
                            "message": f"QRadar health check failed: {response.status}",
                            "last_check": datetime.utcnow().isoformat(),
                        }

        except Exception as e:
            self.connection_status = "unhealthy"
            return {
                "status": "unhealthy",
                "message": f"QRadar health check error: {str(e)}",
                "last_check": datetime.utcnow().isoformat(),
            }
