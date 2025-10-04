"""
CRONOS AI Engine - Integration Manager

Centralized management of all external system integrations for the
Zero-Touch Security Orchestrator.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .base_connector import (
    BaseIntegrationConnector,
    IntegrationType,
    IntegrationConfig,
    IntegrationResult,
)
from .siem_connector import SIEMConnector, SplunkConnector, QRadarConnector
from .ticketing_connector import TicketingConnector, ServiceNowConnector, JiraConnector
from .communication_connector import (
    CommunicationConnector,
    SlackConnector,
    EmailConnector,
)
from .network_security_connector import (
    NetworkSecurityConnector,
    FirewallConnector,
    IDSConnector,
)

from ..config import get_security_config
from ..logging import get_security_logger, SecurityLogType, LogLevel
from ..models import SecurityEvent, ThreatAnalysis, AutomatedResponse


class IntegrationPriority(str, Enum):
    """Priority levels for integrations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class IntegrationRegistration:
    """Registration information for an integration."""

    connector: BaseIntegrationConnector
    priority: IntegrationPriority
    enabled: bool
    auto_retry: bool
    max_failures_before_disable: int
    current_failures: int = 0


class IntegrationManager:
    """
    Central manager for all external system integrations.

    Provides unified interface for sending security events, managing connections,
    and handling integration failures and retries.
    """

    def __init__(self):
        self.config = get_security_config()
        self.logger = get_security_logger("cronos.security.integrations")

        # Integration registry
        self.integrations: Dict[str, IntegrationRegistration] = {}

        # Available connector types
        self.connector_types: Dict[str, Type[BaseIntegrationConnector]] = {
            # SIEM connectors
            "splunk": SplunkConnector,
            "qradar": QRadarConnector,
            # Ticketing connectors
            "servicenow": ServiceNowConnector,
            "jira": JiraConnector,
            # Communication connectors
            "slack": SlackConnector,
            "email": EmailConnector,
            # Network security connectors
            "firewall": FirewallConnector,
            "ids": IDSConnector,
        }

        # Event routing configuration
        self.event_routing = {
            IntegrationType.SIEM: ["security_events", "threat_analysis"],
            IntegrationType.TICKETING: ["security_events", "high_severity_threats"],
            IntegrationType.COMMUNICATION: ["critical_events", "response_execution"],
            IntegrationType.NETWORK_SECURITY: [
                "response_execution",
                "quarantine_events",
            ],
        }

        # State management
        self._initialized = False
        self._running = False
        self._background_tasks: List[asyncio.Task] = []

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Integration Manager initialized",
            level=LogLevel.INFO,
        )

    async def initialize(self) -> None:
        """Initialize the integration manager and load configured integrations."""
        if self._initialized:
            return

        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Initializing Integration Manager",
                level=LogLevel.INFO,
            )

            # Load integration configurations
            await self._load_integrations()

            # Initialize all integrations
            await self._initialize_integrations()

            # Start background tasks
            await self._start_background_tasks()

            self._initialized = True
            self._running = True

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Integration Manager initialized with {len(self.integrations)} integrations",
                level=LogLevel.INFO,
                metadata={"integration_count": len(self.integrations)},
            )

        except Exception as e:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Integration Manager initialization failed: {str(e)}",
                level=LogLevel.ERROR,
                error_code="INTEGRATION_MANAGER_INIT_FAILED",
            )
            raise

    async def _load_integrations(self):
        """Load integration configurations from settings."""

        # SIEM integrations
        siem_config = self.config.integrations.siem
        if siem_config.get("enabled", False):
            await self._register_integration(
                name="siem",
                connector_type=siem_config.get("connector_type", "splunk"),
                integration_type=IntegrationType.SIEM,
                config_data=siem_config,
                priority=IntegrationPriority.CRITICAL,
            )

        # Ticketing integrations
        ticketing_config = self.config.integrations.ticketing
        if ticketing_config.get("enabled", False):
            await self._register_integration(
                name="ticketing",
                connector_type=ticketing_config.get("system_type", "servicenow"),
                integration_type=IntegrationType.TICKETING,
                config_data=ticketing_config,
                priority=IntegrationPriority.HIGH,
            )

        # Communication integrations
        communications_config = self.config.integrations.communications

        # Slack
        slack_config = communications_config.get("slack", {})
        if slack_config.get("enabled", False):
            await self._register_integration(
                name="slack",
                connector_type="slack",
                integration_type=IntegrationType.COMMUNICATION,
                config_data=slack_config,
                priority=IntegrationPriority.MEDIUM,
            )

        # Email
        email_config = communications_config.get("email", {})
        if email_config.get("enabled", False):
            await self._register_integration(
                name="email",
                connector_type="email",
                integration_type=IntegrationType.COMMUNICATION,
                config_data=email_config,
                priority=IntegrationPriority.MEDIUM,
            )

        # Network security integrations
        network_config = self.config.integrations.network_security

        # Firewall
        firewall_config = network_config.get("firewall", {})
        if firewall_config.get("enabled", False):
            await self._register_integration(
                name="firewall",
                connector_type="firewall",
                integration_type=IntegrationType.NETWORK_SECURITY,
                config_data=firewall_config,
                priority=IntegrationPriority.HIGH,
            )

    async def _register_integration(
        self,
        name: str,
        connector_type: str,
        integration_type: IntegrationType,
        config_data: Dict[str, Any],
        priority: IntegrationPriority,
    ):
        """Register a new integration."""

        if connector_type not in self.connector_types:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Unknown connector type: {connector_type}",
                level=LogLevel.ERROR,
                error_code="UNKNOWN_CONNECTOR_TYPE",
            )
            return

        # Create integration configuration
        integration_config = IntegrationConfig(
            name=name,
            integration_type=integration_type,
            enabled=config_data.get("enabled", True),
            endpoint=config_data.get("endpoint"),
            credentials=config_data.get("credentials"),
            timeout_seconds=config_data.get("timeout_seconds", 30),
            retry_attempts=config_data.get("retry_attempts", 3),
            retry_delay_seconds=config_data.get("retry_delay_seconds", 5),
            rate_limit_requests_per_minute=config_data.get(
                "rate_limit_requests_per_minute"
            ),
            custom_config=config_data.get("custom_config", {}),
        )

        # Create connector instance
        connector_class = self.connector_types[connector_type]
        connector = connector_class(integration_config)

        # Register integration
        self.integrations[name] = IntegrationRegistration(
            connector=connector,
            priority=priority,
            enabled=integration_config.enabled,
            auto_retry=config_data.get("auto_retry", True),
            max_failures_before_disable=config_data.get(
                "max_failures_before_disable", 5
            ),
        )

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Integration registered: {name} ({connector_type})",
            level=LogLevel.INFO,
            metadata={
                "connector_type": connector_type,
                "integration_type": integration_type.value,
                "priority": priority.value,
            },
        )

    async def _initialize_integrations(self):
        """Initialize all registered integrations."""

        for name, registration in self.integrations.items():
            if registration.enabled:
                try:
                    success = await registration.connector.initialize()
                    if not success:
                        self.logger.log_security_event(
                            SecurityLogType.CONFIGURATION_CHANGE,
                            f"Integration initialization failed: {name}",
                            level=LogLevel.WARNING,
                            component=name,
                        )
                except Exception as e:
                    self.logger.log_security_event(
                        SecurityLogType.CONFIGURATION_CHANGE,
                        f"Integration initialization error: {name} - {str(e)}",
                        level=LogLevel.ERROR,
                        component=name,
                        error_code="INTEGRATION_INIT_ERROR",
                    )

    async def _start_background_tasks(self):
        """Start background maintenance tasks."""

        # Health monitoring task
        health_task = asyncio.create_task(self._background_health_monitoring())
        self._background_tasks.append(health_task)

        # Connection retry task
        retry_task = asyncio.create_task(self._background_connection_retry())
        self._background_tasks.append(retry_task)

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Started {len(self._background_tasks)} background tasks",
            level=LogLevel.INFO,
        )

    async def send_security_event(
        self,
        security_event: SecurityEvent,
        target_integrations: Optional[List[str]] = None,
    ) -> Dict[str, IntegrationResult]:
        """
        Send security event to configured integrations.

        Args:
            security_event: Security event to send
            target_integrations: Specific integrations to target (if None, uses routing rules)

        Returns:
            Results from each integration
        """

        if not self._initialized:
            await self.initialize()

        results = {}

        # Determine target integrations
        if target_integrations:
            targets = [
                name for name in target_integrations if name in self.integrations
            ]
        else:
            targets = self._determine_event_targets(security_event, "security_events")

        # Send to each target integration
        tasks = []
        for integration_name in targets:
            if integration_name in self.integrations:
                registration = self.integrations[integration_name]
                if registration.enabled:
                    task = self._send_to_integration(
                        integration_name,
                        registration.connector.send_security_event,
                        security_event,
                    )
                    tasks.append((integration_name, task))

        # Wait for all sends to complete
        for integration_name, task in tasks:
            try:
                result = await task
                results[integration_name] = result
            except Exception as e:
                results[integration_name] = IntegrationResult(
                    success=False,
                    message=f"Integration send failed: {str(e)}",
                    error_code="INTEGRATION_SEND_ERROR",
                )

        # Log summary
        successful_sends = sum(1 for result in results.values() if result.success)
        self.logger.log_security_event(
            SecurityLogType.PERFORMANCE_METRIC,
            f"Security event sent to {successful_sends}/{len(results)} integrations",
            level=LogLevel.INFO,
            event_id=security_event.event_id,
            metadata={
                "total_integrations": len(results),
                "successful_sends": successful_sends,
                "results": {name: result.success for name, result in results.items()},
            },
        )

        return results

    async def send_threat_analysis(
        self,
        threat_analysis: ThreatAnalysis,
        target_integrations: Optional[List[str]] = None,
    ) -> Dict[str, IntegrationResult]:
        """
        Send threat analysis to configured integrations.

        Args:
            threat_analysis: Threat analysis to send
            target_integrations: Specific integrations to target

        Returns:
            Results from each integration
        """

        if not self._initialized:
            await self.initialize()

        results = {}

        # Determine target integrations
        if target_integrations:
            targets = [
                name for name in target_integrations if name in self.integrations
            ]
        else:
            targets = self._determine_threat_targets(threat_analysis)

        # Send to each target integration
        tasks = []
        for integration_name in targets:
            if integration_name in self.integrations:
                registration = self.integrations[integration_name]
                if registration.enabled:
                    task = self._send_to_integration(
                        integration_name,
                        registration.connector.send_threat_analysis,
                        threat_analysis,
                    )
                    tasks.append((integration_name, task))

        # Wait for all sends to complete
        for integration_name, task in tasks:
            try:
                result = await task
                results[integration_name] = result
            except Exception as e:
                results[integration_name] = IntegrationResult(
                    success=False,
                    message=f"Integration send failed: {str(e)}",
                    error_code="INTEGRATION_SEND_ERROR",
                )

        return results

    async def send_response_execution(
        self,
        automated_response: AutomatedResponse,
        target_integrations: Optional[List[str]] = None,
    ) -> Dict[str, IntegrationResult]:
        """
        Send automated response execution details to integrations.

        Args:
            automated_response: Automated response to send
            target_integrations: Specific integrations to target

        Returns:
            Results from each integration
        """

        if not self._initialized:
            await self.initialize()

        results = {}

        # Determine target integrations
        if target_integrations:
            targets = [
                name for name in target_integrations if name in self.integrations
            ]
        else:
            targets = self._determine_response_targets(automated_response)

        # Send to each target integration
        tasks = []
        for integration_name in targets:
            if integration_name in self.integrations:
                registration = self.integrations[integration_name]
                if registration.enabled:
                    task = self._send_to_integration(
                        integration_name,
                        registration.connector.send_response_execution,
                        automated_response,
                    )
                    tasks.append((integration_name, task))

        # Wait for all sends to complete
        for integration_name, task in tasks:
            try:
                result = await task
                results[integration_name] = result
            except Exception as e:
                results[integration_name] = IntegrationResult(
                    success=False,
                    message=f"Integration send failed: {str(e)}",
                    error_code="INTEGRATION_SEND_ERROR",
                )

        return results

    async def _send_to_integration(
        self, integration_name: str, send_method, data
    ) -> IntegrationResult:
        """Send data to a specific integration with error handling."""

        registration = self.integrations[integration_name]

        try:
            result = await send_method(data)

            # Reset failure count on success
            if result.success:
                registration.current_failures = 0
            else:
                registration.current_failures += 1
                await self._handle_integration_failure(integration_name, result.message)

            return result

        except Exception as e:
            registration.current_failures += 1
            await self._handle_integration_failure(integration_name, str(e))

            return IntegrationResult(
                success=False,
                message=f"Integration error: {str(e)}",
                error_code="INTEGRATION_ERROR",
            )

    async def _handle_integration_failure(
        self, integration_name: str, error_message: str
    ):
        """Handle integration failure and disable if necessary."""

        registration = self.integrations[integration_name]

        self.logger.log_security_event(
            SecurityLogType.PERFORMANCE_METRIC,
            f"Integration failure: {integration_name} - {error_message}",
            level=LogLevel.WARNING,
            component=integration_name,
            metadata={"failure_count": registration.current_failures},
        )

        # Disable integration if too many failures
        if registration.current_failures >= registration.max_failures_before_disable:
            registration.enabled = False

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Integration disabled due to repeated failures: {integration_name}",
                level=LogLevel.ERROR,
                component=integration_name,
                error_code="INTEGRATION_DISABLED",
            )

    def _determine_event_targets(
        self, security_event: SecurityEvent, event_type: str
    ) -> List[str]:
        """Determine which integrations should receive this event."""

        targets = []

        for name, registration in self.integrations.items():
            if not registration.enabled:
                continue

            integration_type = registration.connector.config.integration_type
            routing_rules = self.event_routing.get(integration_type, [])

            if event_type in routing_rules:
                targets.append(name)

            # Special routing based on event characteristics
            if integration_type == IntegrationType.COMMUNICATION:
                if security_event.threat_level.value in ["critical", "high"]:
                    targets.append(name)

        return targets

    def _determine_threat_targets(self, threat_analysis: ThreatAnalysis) -> List[str]:
        """Determine which integrations should receive threat analysis."""

        targets = []

        for name, registration in self.integrations.items():
            if not registration.enabled:
                continue

            integration_type = registration.connector.config.integration_type

            # SIEM systems get all threat analysis
            if integration_type == IntegrationType.SIEM:
                targets.append(name)

            # Ticketing for high-severity threats
            elif integration_type == IntegrationType.TICKETING:
                if threat_analysis.threat_level.value in ["critical", "high"]:
                    targets.append(name)

        return targets

    def _determine_response_targets(
        self, automated_response: AutomatedResponse
    ) -> List[str]:
        """Determine which integrations should receive response execution details."""

        targets = []

        for name, registration in self.integrations.items():
            if not registration.enabled:
                continue

            integration_type = registration.connector.config.integration_type

            # Communication systems for all responses
            if integration_type == IntegrationType.COMMUNICATION:
                targets.append(name)

            # Network security systems for network-related responses
            elif integration_type == IntegrationType.NETWORK_SECURITY:
                network_actions = {
                    "block_ip",
                    "quarantine",
                    "isolate_system",
                    "network_segmentation",
                }
                action_types = {
                    action.action_type.value for action in automated_response.actions
                }

                if any(action in network_actions for action in action_types):
                    targets.append(name)

        return targets

    async def _background_health_monitoring(self):
        """Background task for monitoring integration health."""

        while self._running:
            try:
                for name, registration in self.integrations.items():
                    if registration.enabled:
                        # Perform health check
                        health_result = await registration.connector.health_check()

                        if health_result["status"] != "healthy":
                            self.logger.log_security_event(
                                SecurityLogType.PERFORMANCE_METRIC,
                                f"Integration health check failed: {name} - {health_result['message']}",
                                level=LogLevel.WARNING,
                                component=name,
                            )

                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Health monitoring error: {str(e)}",
                    level=LogLevel.ERROR,
                    error_code="HEALTH_MONITORING_ERROR",
                )
                await asyncio.sleep(60)

    async def _background_connection_retry(self):
        """Background task for retrying failed connections."""

        while self._running:
            try:
                for name, registration in self.integrations.items():
                    if (
                        not registration.enabled
                        and registration.auto_retry
                        and registration.current_failures > 0
                    ):

                        # Try to re-enable the integration
                        try:
                            success = await registration.connector.initialize()
                            if success:
                                registration.enabled = True
                                registration.current_failures = 0

                                self.logger.log_security_event(
                                    SecurityLogType.CONFIGURATION_CHANGE,
                                    f"Integration re-enabled: {name}",
                                    level=LogLevel.INFO,
                                    component=name,
                                )
                        except Exception as e:
                            self.logger.log_security_event(
                                SecurityLogType.PERFORMANCE_METRIC,
                                f"Integration retry failed: {name} - {str(e)}",
                                level=LogLevel.DEBUG,
                                component=name,
                            )

                await asyncio.sleep(600)  # Retry every 10 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Connection retry error: {str(e)}",
                    level=LogLevel.ERROR,
                    error_code="CONNECTION_RETRY_ERROR",
                )
                await asyncio.sleep(300)

    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations."""

        status = {
            "total_integrations": len(self.integrations),
            "enabled_integrations": sum(
                1 for reg in self.integrations.values() if reg.enabled
            ),
            "integrations": {},
        }

        for name, registration in self.integrations.items():
            status["integrations"][name] = {
                "enabled": registration.enabled,
                "priority": registration.priority.value,
                "current_failures": registration.current_failures,
                "connection_status": registration.connector.get_connection_status(),
            }

        return status

    def enable_integration(self, name: str) -> bool:
        """Enable a specific integration."""
        if name in self.integrations:
            self.integrations[name].enabled = True
            self.integrations[name].current_failures = 0

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Integration enabled: {name}",
                level=LogLevel.INFO,
                component=name,
            )
            return True
        return False

    def disable_integration(self, name: str) -> bool:
        """Disable a specific integration."""
        if name in self.integrations:
            self.integrations[name].enabled = False

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Integration disabled: {name}",
                level=LogLevel.INFO,
                component=name,
            )
            return True
        return False

    async def shutdown(self):
        """Shutdown the integration manager."""

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Integration Manager shutting down",
            level=LogLevel.INFO,
        )

        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Shutdown all integrations
        for registration in self.integrations.values():
            try:
                await registration.connector.shutdown()
            except Exception as e:
                self.logger.log_security_event(
                    SecurityLogType.CONFIGURATION_CHANGE,
                    f"Integration shutdown error: {str(e)}",
                    level=LogLevel.WARNING,
                )

        self._initialized = False

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Integration Manager shutdown complete",
            level=LogLevel.INFO,
        )


# Global integration manager instance
_integration_manager: Optional[IntegrationManager] = None


def get_integration_manager() -> IntegrationManager:
    """Get global integration manager instance."""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = IntegrationManager()
    return _integration_manager
