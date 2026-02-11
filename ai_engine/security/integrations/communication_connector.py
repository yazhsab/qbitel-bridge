"""
QBITEL Engine - Communication System Integration Connectors

Specialized connectors for communication systems like Slack and Email.
"""

import asyncio
import json
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod

from .base_connector import BaseIntegrationConnector, IntegrationResult, IntegrationType
from ..models import SecurityEvent, ThreatAnalysis, AutomatedResponse
from ..logging import get_security_logger, SecurityLogType, LogLevel


class CommunicationConnector(BaseIntegrationConnector, ABC):
    """Base class for communication system connectors."""

    def __init__(self, config):
        super().__init__(config)
        self.logger = get_security_logger("qbitel.security.integrations.communication")

    @abstractmethod
    async def send_alert(
        self, message: str, severity: str, recipients: List[str] = None
    ) -> IntegrationResult:
        """Send an alert message."""
        pass

    @abstractmethod
    async def send_notification(
        self, title: str, message: str, recipients: List[str] = None
    ) -> IntegrationResult:
        """Send a notification message."""
        pass


class SlackConnector(CommunicationConnector):
    """Slack integration connector."""

    def __init__(self, config):
        super().__init__(config)
        self.bot_token = None
        self.webhook_url = None
        self.default_channel = config.custom_config.get(
            "default_channel", "#security-alerts"
        )

    async def initialize(self) -> bool:
        """Initialize Slack connection."""
        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Initializing Slack connector",
                level=LogLevel.INFO,
                component="slack_connector",
            )

            # Set up authentication (bot token or webhook)
            self.bot_token = self.config.credentials.get("bot_token")
            self.webhook_url = self.config.credentials.get("webhook_url")

            if self.bot_token or self.webhook_url:
                # Test connection
                test_result = await self._test_connection()

                if test_result:
                    self.connection_status = "connected"
                    self.last_health_check = datetime.utcnow()

                    self.logger.log_security_event(
                        SecurityLogType.CONFIGURATION_CHANGE,
                        "Slack connector initialized successfully",
                        level=LogLevel.INFO,
                        component="slack_connector",
                    )
                    return True
                else:
                    self.connection_status = "failed"
                    return False
            else:
                self.connection_status = "failed"
                self.logger.log_security_event(
                    SecurityLogType.CONFIGURATION_CHANGE,
                    "Slack credentials not provided",
                    level=LogLevel.ERROR,
                    component="slack_connector",
                )
                return False

        except Exception as e:
            self.connection_status = "failed"
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Slack connector initialization failed: {str(e)}",
                level=LogLevel.ERROR,
                component="slack_connector",
                error_code="SLACK_INIT_ERROR",
            )
            return False

    async def _test_connection(self) -> bool:
        """Test Slack connection."""

        if self.bot_token:
            # Test using bot token
            test_url = "https://slack.com/api/auth.test"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json",
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        test_url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get("ok", False)
                        return False

            except Exception:
                return False
        elif self.webhook_url:
            # Test using webhook (send a test message)
            try:
                test_payload = {
                    "text": "QBITEL Security Orchestrator - Connection Test",
                    "channel": self.default_channel,
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.webhook_url,
                        json=test_payload,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        return response.status == 200

            except Exception:
                return False

        return False

    async def send_security_event(
        self, security_event: SecurityEvent
    ) -> IntegrationResult:
        """Send security event to Slack."""

        try:
            # Create Slack message for security event
            message_blocks = self._create_security_event_blocks(security_event)

            # Send to appropriate channel based on severity
            channel = self._get_channel_for_severity(security_event.threat_level.value)

            result = await self._send_slack_message(
                channel=channel,
                blocks=message_blocks,
                text=f"Security Alert: {security_event.event_type.value}",
            )

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Slack event send failed: {str(e)}",
                error_code="SLACK_SEND_ERROR",
            )

    async def send_threat_analysis(
        self, threat_analysis: ThreatAnalysis
    ) -> IntegrationResult:
        """Send threat analysis to Slack."""

        try:
            # Create Slack message for threat analysis
            message_blocks = self._create_threat_analysis_blocks(threat_analysis)

            # Send to security team channel
            channel = self.config.custom_config.get(
                "threat_analysis_channel", self.default_channel
            )

            result = await self._send_slack_message(
                channel=channel,
                blocks=message_blocks,
                text=f"Threat Analysis: {threat_analysis.threat_category.value}",
            )

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Slack threat analysis send failed: {str(e)}",
                error_code="SLACK_THREAT_SEND_ERROR",
            )

    async def send_response_execution(
        self, automated_response: AutomatedResponse
    ) -> IntegrationResult:
        """Send automated response execution to Slack."""

        try:
            # Create Slack message for response execution
            message_blocks = self._create_response_execution_blocks(automated_response)

            # Send to operations channel
            channel = self.config.custom_config.get(
                "operations_channel", self.default_channel
            )

            result = await self._send_slack_message(
                channel=channel,
                blocks=message_blocks,
                text=f"Automated Response: {automated_response.status.value}",
            )

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Slack response send failed: {str(e)}",
                error_code="SLACK_RESPONSE_SEND_ERROR",
            )

    async def send_alert(
        self, message: str, severity: str, recipients: List[str] = None
    ) -> IntegrationResult:
        """Send an alert message to Slack."""

        try:
            channel = self._get_channel_for_severity(severity)
            color = self._get_color_for_severity(severity)

            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":warning: *QBITEL Security Alert*\n\n{message}",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Severity:* {severity.upper()} | *Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                        }
                    ],
                },
            ]

            # Add user mentions if recipients specified
            if recipients:
                mentions = " ".join([f"<@{user}>" for user in recipients])
                blocks.insert(
                    0,
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Attention: {mentions}"},
                    },
                )

            result = await self._send_slack_message(
                channel=channel,
                blocks=blocks,
                text=f"Security Alert - {severity.upper()}",
            )

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Slack alert send failed: {str(e)}",
                error_code="SLACK_ALERT_ERROR",
            )

    async def send_notification(
        self, title: str, message: str, recipients: List[str] = None
    ) -> IntegrationResult:
        """Send a notification message to Slack."""

        try:
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":information_source: *{title}*\n\n{message}",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                        }
                    ],
                },
            ]

            # Add user mentions if recipients specified
            if recipients:
                mentions = " ".join([f"<@{user}>" for user in recipients])
                blocks.insert(
                    0,
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"FYI: {mentions}"},
                    },
                )

            result = await self._send_slack_message(
                channel=self.default_channel, blocks=blocks, text=title
            )

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Slack notification send failed: {str(e)}",
                error_code="SLACK_NOTIFICATION_ERROR",
            )

    async def _send_slack_message(
        self, channel: str, blocks: List[Dict], text: str = None
    ) -> IntegrationResult:
        """Send message to Slack using bot token or webhook."""

        if self.bot_token:
            return await self._send_via_bot_token(channel, blocks, text)
        elif self.webhook_url:
            return await self._send_via_webhook(channel, blocks, text)
        else:
            return IntegrationResult(
                success=False,
                message="No Slack credentials configured",
                error_code="SLACK_NO_CREDENTIALS",
            )

    async def _send_via_bot_token(
        self, channel: str, blocks: List[Dict], text: str = None
    ) -> IntegrationResult:
        """Send message via bot token."""

        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json",
        }

        payload = {"channel": channel, "blocks": blocks}

        if text:
            payload["text"] = text

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 200:
                        response_data = await response.json()

                        if response_data.get("ok"):
                            return IntegrationResult(
                                success=True,
                                message="Message sent to Slack successfully",
                                response_data={
                                    "channel": response_data.get("channel"),
                                    "ts": response_data.get("ts"),
                                    "message": response_data.get("message"),
                                },
                            )
                        else:
                            return IntegrationResult(
                                success=False,
                                message=f"Slack API error: {response_data.get('error')}",
                                error_code="SLACK_API_ERROR",
                            )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"Slack send failed: {response.status} - {error_text}",
                            error_code="SLACK_SEND_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Slack send error: {str(e)}",
                error_code="SLACK_SEND_ERROR",
            )

    async def _send_via_webhook(
        self, channel: str, blocks: List[Dict], text: str = None
    ) -> IntegrationResult:
        """Send message via webhook."""

        payload = {"channel": channel, "blocks": blocks}

        if text:
            payload["text"] = text

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 200:
                        return IntegrationResult(
                            success=True,
                            message="Message sent to Slack via webhook successfully",
                            response_data={"webhook_response": await response.text()},
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"Slack webhook send failed: {response.status} - {error_text}",
                            error_code="SLACK_WEBHOOK_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Slack webhook error: {str(e)}",
                error_code="SLACK_WEBHOOK_ERROR",
            )

    def _create_security_event_blocks(
        self, security_event: SecurityEvent
    ) -> List[Dict]:
        """Create Slack blocks for security event."""

        severity_emoji = self._get_emoji_for_severity(security_event.threat_level.value)
        color = self._get_color_for_severity(security_event.threat_level.value)

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{severity_emoji} Security Event Detected",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Event Type:*\n{security_event.event_type.value}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Threat Level:*\n{security_event.threat_level.value.upper()}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Source System:*\n{security_event.source_system or 'Unknown'}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Target System:*\n{security_event.target_system or 'Unknown'}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Description:*\n{security_event.description}",
                },
            },
        ]

        # Add network details if available
        if security_event.source_ip or security_event.destination_ip:
            network_fields = []
            if security_event.source_ip:
                network_fields.append(
                    {
                        "type": "mrkdwn",
                        "text": f"*Source IP:*\n{security_event.source_ip}",
                    }
                )
            if security_event.destination_ip:
                network_fields.append(
                    {
                        "type": "mrkdwn",
                        "text": f"*Destination IP:*\n{security_event.destination_ip}",
                    }
                )
            if security_event.protocol:
                network_fields.append(
                    {
                        "type": "mrkdwn",
                        "text": f"*Protocol:*\n{security_event.protocol}",
                    }
                )
            if security_event.port:
                network_fields.append(
                    {"type": "mrkdwn", "text": f"*Port:*\n{security_event.port}"}
                )

            blocks.append({"type": "section", "fields": network_fields})

        # Add context
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Event ID:* {security_event.event_id} | *Time:* {security_event.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC",
                    }
                ],
            }
        )

        return blocks

    def _create_threat_analysis_blocks(
        self, threat_analysis: ThreatAnalysis
    ) -> List[Dict]:
        """Create Slack blocks for threat analysis."""

        severity_emoji = self._get_emoji_for_severity(
            threat_analysis.threat_level.value
        )

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{severity_emoji} Threat Analysis Complete",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Threat Category:*\n{threat_analysis.threat_category.value}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Threat Level:*\n{threat_analysis.threat_level.value.upper()}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Confidence Score:*\n{threat_analysis.confidence_score:.2f}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Affected Systems:*\n{', '.join(threat_analysis.affected_systems[:3])}{' ...' if len(threat_analysis.affected_systems) > 3 else ''}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Attack Vectors:* {', '.join([av.value for av in threat_analysis.attack_vectors])}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Recommended Actions:* {', '.join([action.value for action in threat_analysis.recommended_actions])}",
                },
            },
        ]

        # Add business impact if significant
        if threat_analysis.business_impact.criticality.value in ["critical", "high"]:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":warning: *Business Impact:* {threat_analysis.business_impact.criticality.value.upper()} - Estimated cost: ${threat_analysis.business_impact.estimated_cost:,.2f}",
                    },
                }
            )

        # Add context
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Threat ID:* {threat_analysis.threat_id} | *Analysis Time:* {threat_analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC",
                    }
                ],
            }
        )

        return blocks

    def _create_response_execution_blocks(
        self, automated_response: AutomatedResponse
    ) -> List[Dict]:
        """Create Slack blocks for response execution."""

        status_emoji = (
            "✅"
            if automated_response.status.value == "completed"
            else "⏳" if automated_response.status.value == "in_progress" else "❌"
        )

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} Automated Response {automated_response.status.value.title()}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Response ID:*\n{automated_response.response_id}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:*\n{automated_response.status.value.upper()}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Confidence:*\n{automated_response.confidence_score:.2f}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Actions Count:*\n{len(automated_response.actions)}",
                    },
                ],
            },
        ]

        # Add actions summary
        actions_text = []
        for action in automated_response.actions[:5]:  # Show first 5 actions
            status_icon = (
                "✅"
                if action.status.value == "completed"
                else "⏳" if action.status.value == "in_progress" else "❌"
            )
            actions_text.append(
                f"{status_icon} {action.action_type.value} on {action.target_system}"
            )

        if len(automated_response.actions) > 5:
            actions_text.append(
                f"... and {len(automated_response.actions) - 5} more actions"
            )

        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Actions Executed:*\n" + "\n".join(actions_text),
                },
            }
        )

        # Add rollback info
        rollback_text = "Yes" if automated_response.rollback_plan else "No"
        if automated_response.rollback_plan:
            rollback_text += f" (ID: {automated_response.rollback_plan.rollback_id})"

        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Rollback Available:* {rollback_text}",
                },
            }
        )

        # Add context
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Event ID:* {automated_response.event_id} | *Execution Time:* {automated_response.execution_timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC",
                    }
                ],
            }
        )

        return blocks

    def _get_channel_for_severity(self, severity: str) -> str:
        """Get appropriate Slack channel for severity level."""

        channel_mapping = self.config.custom_config.get("channel_mapping", {})

        return channel_mapping.get(
            severity, channel_mapping.get("default", self.default_channel)
        )

    def _get_emoji_for_severity(self, severity: str) -> str:
        """Get emoji for severity level."""

        mapping = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "⚡",
            "low": "ℹ️",
            "informational": "📋",
        }

        return mapping.get(severity, "⚠️")

    def _get_color_for_severity(self, severity: str) -> str:
        """Get color for severity level."""

        mapping = {
            "critical": "#d63031",  # Red
            "high": "#e17055",  # Orange
            "medium": "#fdcb6e",  # Yellow
            "low": "#00b894",  # Green
            "informational": "#6c5ce7",  # Purple
        }

        return mapping.get(severity, "#fdcb6e")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for Slack connector."""

        try:
            # Test basic connectivity
            test_result = await self._test_connection()

            if test_result:
                self.connection_status = "healthy"
                self.last_health_check = datetime.utcnow()

                return {
                    "status": "healthy",
                    "message": "Slack connection is healthy",
                    "last_check": self.last_health_check.isoformat(),
                }
            else:
                self.connection_status = "unhealthy"
                return {
                    "status": "unhealthy",
                    "message": "Slack health check failed",
                    "last_check": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            self.connection_status = "unhealthy"
            return {
                "status": "unhealthy",
                "message": f"Slack health check error: {str(e)}",
                "last_check": datetime.utcnow().isoformat(),
            }


class EmailConnector(CommunicationConnector):
    """Email integration connector."""

    def __init__(self, config):
        super().__init__(config)
        self.smtp_server = None
        self.smtp_port = None
        self.username = None
        self.password = None
        self.use_tls = True
        self.default_recipients = config.custom_config.get("default_recipients", [])

    async def initialize(self) -> bool:
        """Initialize Email connection."""
        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Initializing Email connector",
                level=LogLevel.INFO,
                component="email_connector",
            )

            # Extract SMTP configuration
            self.smtp_server = self.config.custom_config.get("smtp_server")
            self.smtp_port = self.config.custom_config.get("smtp_port", 587)
            self.use_tls = self.config.custom_config.get("use_tls", True)
            self.username = self.config.credentials.get("username")
            self.password = self.config.credentials.get("password")

            if self.smtp_server and self.username and self.password:
                # Test connection
                test_result = await self._test_connection()

                if test_result:
                    self.connection_status = "connected"
                    self.last_health_check = datetime.utcnow()

                    self.logger.log_security_event(
                        SecurityLogType.CONFIGURATION_CHANGE,
                        "Email connector initialized successfully",
                        level=LogLevel.INFO,
                        component="email_connector",
                    )
                    return True
                else:
                    self.connection_status = "failed"
                    return False
            else:
                self.connection_status = "failed"
                self.logger.log_security_event(
                    SecurityLogType.CONFIGURATION_CHANGE,
                    "Email configuration incomplete",
                    level=LogLevel.ERROR,
                    component="email_connector",
                )
                return False

        except Exception as e:
            self.connection_status = "failed"
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Email connector initialization failed: {str(e)}",
                level=LogLevel.ERROR,
                component="email_connector",
                error_code="EMAIL_INIT_ERROR",
            )
            return False

    async def _test_connection(self) -> bool:
        """Test email connection."""

        try:
            loop = asyncio.get_event_loop()

            # Run SMTP test in thread to avoid blocking
            def test_smtp():
                try:
                    if self.use_tls:
                        server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                        server.starttls()
                    else:
                        server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)

                    server.login(self.username, self.password)
                    server.quit()
                    return True
                except Exception:
                    return False

            result = await loop.run_in_executor(None, test_smtp)
            return result

        except Exception:
            return False

    async def send_security_event(
        self, security_event: SecurityEvent
    ) -> IntegrationResult:
        """Send security event via email."""

        try:
            # Create email for security event
            subject = f"QBITEL Security Alert: {security_event.event_type.value} ({security_event.threat_level.value.upper()})"
            body = self._create_security_event_email(security_event)

            # Get recipients based on severity
            recipients = self._get_recipients_for_severity(
                security_event.threat_level.value
            )

            result = await self._send_email(
                subject=subject, body=body, recipients=recipients, is_html=True
            )

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Email event send failed: {str(e)}",
                error_code="EMAIL_SEND_ERROR",
            )

    async def send_threat_analysis(
        self, threat_analysis: ThreatAnalysis
    ) -> IntegrationResult:
        """Send threat analysis via email."""

        try:
            # Create email for threat analysis
            subject = f"QBITEL Threat Analysis: {threat_analysis.threat_category.value} ({threat_analysis.threat_level.value.upper()})"
            body = self._create_threat_analysis_email(threat_analysis)

            # Send to security team
            recipients = self.config.custom_config.get(
                "security_team_recipients", self.default_recipients
            )

            result = await self._send_email(
                subject=subject, body=body, recipients=recipients, is_html=True
            )

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Email threat analysis send failed: {str(e)}",
                error_code="EMAIL_THREAT_SEND_ERROR",
            )

    async def send_response_execution(
        self, automated_response: AutomatedResponse
    ) -> IntegrationResult:
        """Send automated response execution via email."""

        try:
            # Create email for response execution
            subject = f"QBITEL Automated Response: {automated_response.status.value.title()} - {automated_response.response_id}"
            body = self._create_response_execution_email(automated_response)

            # Send to operations team
            recipients = self.config.custom_config.get(
                "operations_team_recipients", self.default_recipients
            )

            result = await self._send_email(
                subject=subject, body=body, recipients=recipients, is_html=True
            )

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Email response send failed: {str(e)}",
                error_code="EMAIL_RESPONSE_SEND_ERROR",
            )

    async def send_alert(
        self, message: str, severity: str, recipients: List[str] = None
    ) -> IntegrationResult:
        """Send an alert email."""

        try:
            subject = f"QBITEL Security Alert - {severity.upper()}"

            # Create HTML email body
            body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: {self._get_color_for_severity(severity)}; color: white; padding: 15px; border-radius: 5px; }}
                    .content {{ margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-left: 4px solid {self._get_color_for_severity(severity)}; }}
                    .footer {{ margin-top: 20px; padding: 10px; font-size: 12px; color: #666; border-top: 1px solid #eee; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>🚨 QBITEL Security Alert</h2>
                </div>
                <div class="content">
                    <h3>Alert Details</h3>
                    <p><strong>Severity:</strong> {severity.upper()}</p>
                    <p><strong>Message:</strong></p>
                    <p>{message}</p>
                    <p><strong>Time:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
                </div>
                <div class="footer">
                    <p>This alert was generated automatically by the QBITEL Security Orchestrator.</p>
                </div>
            </body>
            </html>
            """

            target_recipients = recipients or self._get_recipients_for_severity(
                severity
            )

            result = await self._send_email(
                subject=subject, body=body, recipients=target_recipients, is_html=True
            )

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Email alert send failed: {str(e)}",
                error_code="EMAIL_ALERT_ERROR",
            )

    async def send_notification(
        self, title: str, message: str, recipients: List[str] = None
    ) -> IntegrationResult:
        """Send a notification email."""

        try:
            subject = f"QBITEL Notification: {title}"

            # Create HTML email body
            body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #007bff; color: white; padding: 15px; border-radius: 5px; }}
                    .content {{ margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-left: 4px solid #007bff; }}
                    .footer {{ margin-top: 20px; padding: 10px; font-size: 12px; color: #666; border-top: 1px solid #eee; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>📋 QBITEL Notification</h2>
                </div>
                <div class="content">
                    <h3>{title}</h3>
                    <p>{message}</p>
                    <p><strong>Time:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
                </div>
                <div class="footer">
                    <p>This notification was generated by the QBITEL Security Orchestrator.</p>
                </div>
            </body>
            </html>
            """

            target_recipients = recipients or self.default_recipients

            result = await self._send_email(
                subject=subject, body=body, recipients=target_recipients, is_html=True
            )

            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Email notification send failed: {str(e)}",
                error_code="EMAIL_NOTIFICATION_ERROR",
            )

    async def _send_email(
        self,
        subject: str,
        body: str,
        recipients: List[str],
        is_html: bool = False,
        attachments: List[Dict] = None,
    ) -> IntegrationResult:
        """Send email using SMTP."""

        try:
            loop = asyncio.get_event_loop()

            # Run email sending in thread to avoid blocking
            def send_smtp_email():
                try:
                    # Create message
                    msg = MIMEMultipart()
                    msg["From"] = self.username
                    msg["Subject"] = subject

                    # Add body
                    if is_html:
                        msg.attach(MIMEText(body, "html"))
                    else:
                        msg.attach(MIMEText(body, "plain"))

                    # Add attachments if provided
                    if attachments:
                        for attachment in attachments:
                            part = MIMEApplication(
                                attachment["data"],
                                attachment.get(
                                    "content_type", "application/octet-stream"
                                ),
                            )
                            part.add_header(
                                "Content-Disposition",
                                "attachment",
                                filename=attachment.get("filename", "attachment"),
                            )
                            msg.attach(part)

                    # Connect to SMTP server
                    if self.use_tls:
                        server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                        server.starttls()
                    else:
                        server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)

                    server.login(self.username, self.password)

                    # Send to each recipient
                    failed_recipients = []
                    successful_recipients = []

                    for recipient in recipients:
                        try:
                            msg["To"] = recipient
                            server.send_message(msg)
                            successful_recipients.append(recipient)
                            del msg["To"]  # Remove for next iteration
                        except Exception as e:
                            failed_recipients.append(
                                {"email": recipient, "error": str(e)}
                            )

                    server.quit()

                    return {
                        "success": len(failed_recipients) == 0,
                        "successful_recipients": successful_recipients,
                        "failed_recipients": failed_recipients,
                    }

                except Exception as e:
                    return {"success": False, "error": str(e)}

            result = await loop.run_in_executor(None, send_smtp_email)

            if result["success"]:
                return IntegrationResult(
                    success=True,
                    message=f"Email sent successfully to {len(result['successful_recipients'])} recipients",
                    response_data={
                        "successful_recipients": result["successful_recipients"],
                        "failed_recipients": result.get("failed_recipients", []),
                    },
                )
            else:
                return IntegrationResult(
                    success=False,
                    message=f"Email send failed: {result.get('error', 'Unknown error')}",
                    error_code="EMAIL_SMTP_ERROR",
                    response_data={
                        "failed_recipients": result.get("failed_recipients", [])
                    },
                )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Email send error: {str(e)}",
                error_code="EMAIL_SEND_ERROR",
            )

    def _create_security_event_email(self, security_event: SecurityEvent) -> str:
        """Create HTML email body for security event."""

        severity_color = self._get_color_for_severity(security_event.threat_level.value)

        network_details = ""
        if security_event.source_ip or security_event.destination_ip:
            network_details = f"""
            <h4>Network Details</h4>
            <ul>
                {f'<li><strong>Source IP:</strong> {security_event.source_ip}</li>' if security_event.source_ip else ''}
                {f'<li><strong>Destination IP:</strong> {security_event.destination_ip}</li>' if security_event.destination_ip else ''}
                {f'<li><strong>Protocol:</strong> {security_event.protocol}</li>' if security_event.protocol else ''}
                {f'<li><strong>Port:</strong> {security_event.port}</li>' if security_event.port else ''}
            </ul>
            """

        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {severity_color}; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ margin: 20px 0; }}
                .section {{ margin: 15px 0; padding: 15px; background-color: #f9f9f9; border-left: 4px solid {severity_color}; }}
                .footer {{ margin-top: 20px; padding: 10px; font-size: 12px; color: #666; border-top: 1px solid #eee; }}
                .details {{ margin: 10px 0; }}
                .details strong {{ color: #333; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚨 QBITEL Security Event Detected</h2>
            </div>
            <div class="content">
                <div class="section">
                    <h3>Event Summary</h3>
                    <div class="details">
                        <strong>Event Type:</strong> {security_event.event_type.value}<br>
                        <strong>Threat Level:</strong> {security_event.threat_level.value.upper()}<br>
                        <strong>Source System:</strong> {security_event.source_system or 'Unknown'}<br>
                        <strong>Target System:</strong> {security_event.target_system or 'Unknown'}<br>
                        <strong>User:</strong> {security_event.user or 'N/A'}<br>
                        <strong>Timestamp:</strong> {security_event.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC
                    </div>
                </div>
                
                <div class="section">
                    <h3>Description</h3>
                    <p>{security_event.description}</p>
                </div>
                
                {network_details}
                
                <div class="section">
                    <h3>Event Metadata</h3>
                    <div class="details">
                        <strong>Event ID:</strong> {security_event.event_id}<br>
                        <strong>Tags:</strong> {', '.join(security_event.tags) if security_event.tags else 'None'}
                    </div>
                </div>
            </div>
            <div class="footer">
                <p>This alert was generated automatically by the QBITEL Security Orchestrator.</p>
                <p>For more information, please check your security dashboard or contact the security team.</p>
            </div>
        </body>
        </html>
        """

    def _create_threat_analysis_email(self, threat_analysis: ThreatAnalysis) -> str:
        """Create HTML email body for threat analysis."""

        severity_color = self._get_color_for_severity(
            threat_analysis.threat_level.value
        )

        iocs_html = ""
        if threat_analysis.iocs:
            iocs_list = []
            for ioc in threat_analysis.iocs[:10]:  # Show first 10 IOCs
                iocs_list.append(
                    f"<li><strong>{ioc.ioc_type.value}:</strong> {ioc.value} (confidence: {ioc.confidence})</li>"
                )

            iocs_html = f"""
            <h4>Indicators of Compromise</h4>
            <ul>
                {''.join(iocs_list)}
                {f'<li><em>... and {len(threat_analysis.iocs) - 10} more IOCs</em></li>' if len(threat_analysis.iocs) > 10 else ''}
            </ul>
            """

        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {severity_color}; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ margin: 20px 0; }}
                .section {{ margin: 15px 0; padding: 15px; background-color: #f9f9f9; border-left: 4px solid {severity_color}; }}
                .footer {{ margin-top: 20px; padding: 10px; font-size: 12px; color: #666; border-top: 1px solid #eee; }}
                .details {{ margin: 10px 0; }}
                .details strong {{ color: #333; }}
                .critical {{ background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🔍 QBITEL Threat Analysis Complete</h2>
            </div>
            <div class="content">
                <div class="section">
                    <h3>Analysis Summary</h3>
                    <div class="details">
                        <strong>Threat Category:</strong> {threat_analysis.threat_category.value}<br>
                        <strong>Threat Level:</strong> {threat_analysis.threat_level.value.upper()}<br>
                        <strong>Confidence Score:</strong> {threat_analysis.confidence_score:.2f}<br>
                        <strong>Attack Vectors:</strong> {', '.join([av.value for av in threat_analysis.attack_vectors])}<br>
                        <strong>Affected Systems:</strong> {', '.join(threat_analysis.affected_systems)}<br>
                        <strong>Analysis Time:</strong> {threat_analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC
                    </div>
                </div>
                
                <div class="section {'critical' if threat_analysis.business_impact.criticality.value in ['critical', 'high'] else ''}">
                    <h3>Business Impact Assessment</h3>
                    <div class="details">
                        <strong>Criticality:</strong> {threat_analysis.business_impact.criticality.value.upper()}<br>
                        <strong>Affected Services:</strong> {', '.join(threat_analysis.business_impact.affected_services)}<br>
                        <strong>Estimated Cost:</strong> ${threat_analysis.business_impact.estimated_cost:,.2f}<br>
                        <strong>Compliance Impact:</strong> {', '.join([ci.value for ci in threat_analysis.business_impact.compliance_impact])}
                    </div>
                </div>
                
                {iocs_html}
                
                <div class="section">
                    <h3>Recommended Actions</h3>
                    <ul>
                        {''.join([f'<li>{action.value}</li>' for action in threat_analysis.recommended_actions])}
                    </ul>
                </div>
                
                <div class="section">
                    <h3>Analysis Metadata</h3>
                    <div class="details">
                        <strong>Threat ID:</strong> {threat_analysis.threat_id}<br>
                        <strong>Related Event ID:</strong> {threat_analysis.event_id}
                    </div>
                </div>
            </div>
            <div class="footer">
                <p>This threat analysis was generated by the QBITEL Security Orchestrator.</p>
                <p>Please review the recommendations and take appropriate action as needed.</p>
            </div>
        </body>
        </html>
        """

    def _create_response_execution_email(
        self, automated_response: AutomatedResponse
    ) -> str:
        """Create HTML email body for response execution."""

        status_color = (
            "#28a745"
            if automated_response.status.value == "completed"
            else (
                "#ffc107"
                if automated_response.status.value == "in_progress"
                else "#dc3545"
            )
        )

        actions_html = ""
        if automated_response.actions:
            actions_list = []
            for action in automated_response.actions:
                status_icon = (
                    "✅"
                    if action.status.value == "completed"
                    else "⏳" if action.status.value == "in_progress" else "❌"
                )
                actions_list.append(
                    f"<li>{status_icon} <strong>{action.action_type.value}</strong> on {action.target_system} - {action.status.value}</li>"
                )

            actions_html = f"""
            <h4>Actions Executed</h4>
            <ul>
                {''.join(actions_list)}
            </ul>
            """

        rollback_html = ""
        if automated_response.rollback_plan:
            rollback_html = f"""
            <div class="section">
                <h3>Rollback Plan</h3>
                <div class="details">
                    <strong>Rollback ID:</strong> {automated_response.rollback_plan.rollback_id}<br>
                    <strong>Available Actions:</strong> {len(automated_response.rollback_plan.actions)}<br>
                    <strong>Status:</strong> Available and ready for execution if needed
                </div>
            </div>
            """

        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {status_color}; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ margin: 20px 0; }}
                .section {{ margin: 15px 0; padding: 15px; background-color: #f9f9f9; border-left: 4px solid {status_color}; }}
                .footer {{ margin-top: 20px; padding: 10px; font-size: 12px; color: #666; border-top: 1px solid #eee; }}
                .details {{ margin: 10px 0; }}
                .details strong {{ color: #333; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>⚡ QBITEL Automated Response {automated_response.status.value.title()}</h2>
            </div>
            <div class="content">
                <div class="section">
                    <h3>Response Summary</h3>
                    <div class="details">
                        <strong>Response ID:</strong> {automated_response.response_id}<br>
                        <strong>Status:</strong> {automated_response.status.value.upper()}<br>
                        <strong>Confidence Score:</strong> {automated_response.confidence_score:.2f}<br>
                        <strong>Total Actions:</strong> {len(automated_response.actions)}<br>
                        <strong>Execution Time:</strong> {automated_response.execution_timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC
                    </div>
                </div>
                
                <div class="section">
                    {actions_html}
                </div>
                
                {rollback_html}
                
                <div class="section">
                    <h3>Response Metadata</h3>
                    <div class="details">
                        <strong>Related Event ID:</strong> {automated_response.event_id}<br>
                        <strong>Related Threat ID:</strong> {automated_response.threat_id}
                    </div>
                </div>
            </div>
            <div class="footer">
                <p>This response was executed automatically by the QBITEL Security Orchestrator.</p>
                <p>Please review the actions taken and verify the system status.</p>
            </div>
        </body>
        </html>
        """

    def _get_recipients_for_severity(self, severity: str) -> List[str]:
        """Get appropriate email recipients for severity level."""

        recipient_mapping = self.config.custom_config.get("recipient_mapping", {})

        return recipient_mapping.get(
            severity, recipient_mapping.get("default", self.default_recipients)
        )

    def _get_color_for_severity(self, severity: str) -> str:
        """Get color for severity level."""

        mapping = {
            "critical": "#dc3545",  # Red
            "high": "#fd7e14",  # Orange
            "medium": "#ffc107",  # Yellow
            "low": "#28a745",  # Green
            "informational": "#17a2b8",  # Blue
        }

        return mapping.get(severity, "#ffc107")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for Email connector."""

        try:
            # Test basic connectivity
            test_result = await self._test_connection()

            if test_result:
                self.connection_status = "healthy"
                self.last_health_check = datetime.utcnow()

                return {
                    "status": "healthy",
                    "message": "Email connection is healthy",
                    "last_check": self.last_health_check.isoformat(),
                }
            else:
                self.connection_status = "unhealthy"
                return {
                    "status": "unhealthy",
                    "message": "Email health check failed",
                    "last_check": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            self.connection_status = "unhealthy"
            return {
                "status": "unhealthy",
                "message": f"Email health check error: {str(e)}",
                "last_check": datetime.utcnow().isoformat(),
            }
