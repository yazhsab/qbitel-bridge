"""
CRONOS AI Engine - Alerting System

This module provides comprehensive alerting and notification capabilities
for monitoring AI Engine health, performance, and operational issues.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from abc import ABC, abstractmethod

import requests

from ..core.config import Config
from ..core.exceptions import AlertException
from .health import HealthStatus, SystemHealth, ComponentHealth
from .logging import get_logger


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


class RuleOperator(str, Enum):
    """Rule comparison operators."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_EQUAL = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"


@dataclass
class AlertCondition:
    """Alert rule condition."""
    metric_name: str
    operator: RuleOperator
    threshold: Union[float, str, List[Any]]
    evaluation_period: int = 300  # seconds
    data_points: int = 1  # Number of consecutive data points


@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    created_at: float
    updated_at: float
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    source_data: Dict[str, Any] = field(default_factory=dict)
    
    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = time.time()
        self.updated_at = time.time()
    
    def acknowledge(self, user: str) -> None:
        """Acknowledge alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = time.time()
        self.acknowledged_by = user
        self.updated_at = time.time()
    
    def suppress(self) -> None:
        """Suppress alert."""
        self.status = AlertStatus.SUPPRESSED
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
            "labels": self.labels,
            "annotations": self.annotations,
            "source_data": self.source_data
        }


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    description: str
    conditions: List[AlertCondition]
    severity: AlertSeverity
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list)
    cooldown_period: int = 3600  # seconds before re-alerting
    max_alerts: int = 10  # Maximum active alerts for this rule
    
    def evaluate(self, metrics_data: Dict[str, Any]) -> bool:
        """Evaluate if alert conditions are met."""
        for condition in self.conditions:
            if not self._evaluate_condition(condition, metrics_data):
                return False
        return True
    
    def _evaluate_condition(self, condition: AlertCondition, metrics_data: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        metric_value = metrics_data.get(condition.metric_name)
        if metric_value is None:
            return False
        
        threshold = condition.threshold
        operator = condition.operator
        
        try:
            if operator == RuleOperator.EQUALS:
                return metric_value == threshold
            elif operator == RuleOperator.NOT_EQUALS:
                return metric_value != threshold
            elif operator == RuleOperator.GREATER_THAN:
                return float(metric_value) > float(threshold)
            elif operator == RuleOperator.GREATER_THAN_EQUAL:
                return float(metric_value) >= float(threshold)
            elif operator == RuleOperator.LESS_THAN:
                return float(metric_value) < float(threshold)
            elif operator == RuleOperator.LESS_THAN_EQUAL:
                return float(metric_value) <= float(threshold)
            elif operator == RuleOperator.CONTAINS:
                return str(threshold) in str(metric_value)
            elif operator == RuleOperator.NOT_CONTAINS:
                return str(threshold) not in str(metric_value)
            elif operator == RuleOperator.IN:
                return metric_value in threshold
            elif operator == RuleOperator.NOT_IN:
                return metric_value not in threshold
            else:
                return False
        except (ValueError, TypeError):
            return False


class NotificationProvider(ABC):
    """Base class for notification providers."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = get_logger(f"{__name__}.{name}")
    
    @abstractmethod
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test notification provider connection."""
        pass


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        self.smtp_host = config.get("smtp_host", "localhost")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.use_tls = config.get("use_tls", True)
        self.from_email = config.get("from_email", "noreply@cronos-ai.com")
        self.to_emails = config.get("to_emails", [])
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = f"[CRONOS AI] {alert.severity.upper()}: {alert.title}"
            message["From"] = self.from_email
            message["To"] = ", ".join(self.to_emails)
            
            # Create email content
            text_content = self._create_text_content(alert)
            html_content = self._create_html_content(alert)
            
            text_part = MIMEText(text_content, "plain")
            html_part = MIMEText(html_content, "html")
            
            message.attach(text_part)
            message.attach(html_part)
            
            # Send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                server.sendmail(self.from_email, self.to_emails, message.as_string())
            
            self.logger.info(f"Email alert sent for: {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test email server connection."""
        try:
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                
                if self.username and self.password:
                    server.login(self.username, self.password)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Email connection test failed: {e}")
            return False
    
    def _create_text_content(self, alert: Alert) -> str:
        """Create plain text email content."""
        return f"""
CRONOS AI Alert: {alert.title}

Severity: {alert.severity.upper()}
Status: {alert.status.upper()}
Rule: {alert.rule_name}
Created: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert.created_at))}

Description:
{alert.description}

Labels:
{json.dumps(alert.labels, indent=2)}

Source Data:
{json.dumps(alert.source_data, indent=2)}

Alert ID: {alert.alert_id}
"""
    
    def _create_html_content(self, alert: Alert) -> str:
        """Create HTML email content."""
        severity_colors = {
            AlertSeverity.INFO: "#0066cc",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.ERROR: "#cc0000",
            AlertSeverity.CRITICAL: "#990000"
        }
        
        color = severity_colors.get(alert.severity, "#666666")
        
        return f"""
<html>
<body style="font-family: Arial, sans-serif; margin: 20px;">
    <div style="border-left: 4px solid {color}; padding-left: 20px;">
        <h2 style="color: {color}; margin-top: 0;">CRONOS AI Alert: {alert.title}</h2>
        
        <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
            <tr>
                <td style="padding: 8px; background-color: #f5f5f5; font-weight: bold; width: 120px;">Severity:</td>
                <td style="padding: 8px; color: {color}; font-weight: bold;">{alert.severity.upper()}</td>
            </tr>
            <tr>
                <td style="padding: 8px; background-color: #f5f5f5; font-weight: bold;">Status:</td>
                <td style="padding: 8px;">{alert.status.upper()}</td>
            </tr>
            <tr>
                <td style="padding: 8px; background-color: #f5f5f5; font-weight: bold;">Rule:</td>
                <td style="padding: 8px;">{alert.rule_name}</td>
            </tr>
            <tr>
                <td style="padding: 8px; background-color: #f5f5f5; font-weight: bold;">Created:</td>
                <td style="padding: 8px;">{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert.created_at))}</td>
            </tr>
        </table>
        
        <h3>Description:</h3>
        <p>{alert.description}</p>
        
        <h3>Alert ID:</h3>
        <code style="background-color: #f0f0f0; padding: 4px;">{alert.alert_id}</code>
    </div>
</body>
</html>
"""


class SlackNotificationProvider(NotificationProvider):
    """Slack notification provider."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        self.webhook_url = config.get("webhook_url", "")
        self.channel = config.get("channel", "#alerts")
        self.username = config.get("username", "CRONOS AI")
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            severity_colors = {
                AlertSeverity.INFO: "#0066cc",
                AlertSeverity.WARNING: "#ff9900",
                AlertSeverity.ERROR: "#cc0000",
                AlertSeverity.CRITICAL: "#990000"
            }
            
            severity_emojis = {
                AlertSeverity.INFO: ":information_source:",
                AlertSeverity.WARNING: ":warning:",
                AlertSeverity.ERROR: ":x:",
                AlertSeverity.CRITICAL: ":rotating_light:"
            }
            
            color = severity_colors.get(alert.severity, "#666666")
            emoji = severity_emojis.get(alert.severity, ":bell:")
            
            payload = {
                "channel": self.channel,
                "username": self.username,
                "attachments": [
                    {
                        "color": color,
                        "title": f"{emoji} {alert.title}",
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.upper(),
                                "short": True
                            },
                            {
                                "title": "Status",
                                "value": alert.status.upper(),
                                "short": True
                            },
                            {
                                "title": "Rule",
                                "value": alert.rule_name,
                                "short": True
                            },
                            {
                                "title": "Alert ID",
                                "value": alert.alert_id,
                                "short": True
                            }
                        ],
                        "timestamp": int(alert.created_at)
                    }
                ]
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Slack alert sent for: {alert.alert_id}")
                return True
            else:
                self.logger.error(f"Slack notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Slack webhook."""
        try:
            payload = {
                "channel": self.channel,
                "username": self.username,
                "text": "CRONOS AI Alert System Test Message"
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Slack connection test failed: {e}")
            return False


class WebhookNotificationProvider(NotificationProvider):
    """Generic webhook notification provider."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        self.webhook_url = config.get("webhook_url", "")
        self.headers = config.get("headers", {"Content-Type": "application/json"})
        self.timeout = config.get("timeout", 10)
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        try:
            payload = {
                "alert": alert.to_dict(),
                "timestamp": time.time(),
                "source": "cronos-ai"
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code in [200, 201, 202]:
                self.logger.info(f"Webhook alert sent for: {alert.alert_id}")
                return True
            else:
                self.logger.error(f"Webhook notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test webhook endpoint."""
        try:
            test_payload = {
                "test": True,
                "message": "CRONOS AI Alert System Test",
                "timestamp": time.time()
            }
            
            response = requests.post(
                self.webhook_url,
                json=test_payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            return response.status_code in [200, 201, 202]
            
        except Exception as e:
            self.logger.error(f"Webhook connection test failed: {e}")
            return False


@dataclass
class AlertChannel:
    """Alert notification channel configuration."""
    name: str
    provider: NotificationProvider
    enabled: bool = True
    min_severity: AlertSeverity = AlertSeverity.INFO
    max_alerts_per_hour: int = 50
    
    def should_notify(self, alert: Alert) -> bool:
        """Check if channel should be notified for alert."""
        if not self.enabled:
            return False
        
        # Check severity
        severity_levels = {
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.ERROR: 3,
            AlertSeverity.CRITICAL: 4
        }
        
        if severity_levels[alert.severity] < severity_levels[self.min_severity]:
            return False
        
        return True


class AlertManager:
    """
    Central alert management system for CRONOS AI Engine.
    
    This class manages alert rules, evaluates conditions, sends notifications,
    and tracks alert lifecycle.
    """
    
    def __init__(self, config: Config):
        """Initialize alert manager."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_channels: Dict[str, AlertChannel] = {}
        
        # Alert history
        self.alert_history: List[Alert] = []
        self.max_history_size = getattr(config, 'max_alert_history', 1000)
        
        # Evaluation state
        self.last_evaluation = 0
        self.evaluation_interval = getattr(config, 'alert_evaluation_interval', 60)  # seconds
        
        # Background tasks
        self._evaluation_task: Optional[asyncio.Task] = None
        self._stop_evaluation = asyncio.Event()
        
        # Statistics
        self.stats = {
            "total_alerts": 0,
            "active_alerts": 0,
            "alerts_by_severity": {severity.value: 0 for severity in AlertSeverity},
            "notifications_sent": 0,
            "notifications_failed": 0
        }
        
        # Initialize default alert rules
        self._initialize_default_rules()
        
        self.logger.info("AlertManager initialized")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False
    
    def add_notification_channel(self, channel: AlertChannel) -> None:
        """Add notification channel."""
        self.notification_channels[channel.name] = channel
        self.logger.info(f"Added notification channel: {channel.name}")
    
    def remove_notification_channel(self, channel_name: str) -> bool:
        """Remove notification channel."""
        if channel_name in self.notification_channels:
            del self.notification_channels[channel_name]
            self.logger.info(f"Removed notification channel: {channel_name}")
            return True
        return False
    
    async def evaluate_rules(self, metrics_data: Dict[str, Any]) -> List[Alert]:
        """Evaluate all alert rules against current metrics."""
        new_alerts = []
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                if rule.evaluate(metrics_data):
                    # Check if we should create a new alert
                    if self._should_create_alert(rule, current_time):
                        alert = self._create_alert(rule, metrics_data, current_time)
                        new_alerts.append(alert)
                        
                        # Send notifications
                        await self._send_alert_notifications(alert)
                
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule_name}: {e}")
        
        return new_alerts
    
    async def resolve_alert(self, alert_id: str, user: Optional[str] = None) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolve()
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            # Update statistics
            self.stats["active_alerts"] -= 1
            
            self.logger.info(f"Alert resolved: {alert_id}")
            return True
        
        return False
    
    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledge(user)
            
            self.logger.info(f"Alert acknowledged by {user}: {alert_id}")
            return True
        
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        return {
            **self.stats,
            "active_alerts": len(self.active_alerts),
            "rules_count": len(self.alert_rules),
            "channels_count": len(self.notification_channels),
            "history_size": len(self.alert_history)
        }
    
    async def start_evaluation(self) -> None:
        """Start background alert evaluation."""
        if self._evaluation_task:
            self.logger.warning("Alert evaluation already started")
            return
        
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())
        self.logger.info("Started background alert evaluation")
    
    async def stop_evaluation(self) -> None:
        """Stop background alert evaluation."""
        if not self._evaluation_task:
            return
        
        self._stop_evaluation.set()
        
        try:
            await asyncio.wait_for(self._evaluation_task, timeout=5.0)
        except asyncio.TimeoutError:
            self._evaluation_task.cancel()
        
        self._evaluation_task = None
        self._stop_evaluation.clear()
        self.logger.info("Stopped background alert evaluation")
    
    async def test_notification_channels(self) -> Dict[str, bool]:
        """Test all notification channels."""
        results = {}
        
        for name, channel in self.notification_channels.items():
            try:
                result = await channel.provider.test_connection()
                results[name] = result
                self.logger.info(f"Channel {name} test: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                results[name] = False
                self.logger.error(f"Channel {name} test failed: {e}")
        
        return results
    
    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules."""
        
        # High CPU usage
        cpu_rule = AlertRule(
            name="high_cpu_usage",
            description="Alert when CPU usage is consistently high",
            conditions=[
                AlertCondition(
                    metric_name="system_cpu_percent",
                    operator=RuleOperator.GREATER_THAN,
                    threshold=85.0,
                    evaluation_period=300,
                    data_points=3
                )
            ],
            severity=AlertSeverity.WARNING,
            labels={"component": "system", "metric": "cpu"},
            cooldown_period=1800  # 30 minutes
        )
        
        # High memory usage
        memory_rule = AlertRule(
            name="high_memory_usage",
            description="Alert when memory usage is critically high",
            conditions=[
                AlertCondition(
                    metric_name="system_memory_percent",
                    operator=RuleOperator.GREATER_THAN,
                    threshold=90.0,
                    evaluation_period=180,
                    data_points=2
                )
            ],
            severity=AlertSeverity.CRITICAL,
            labels={"component": "system", "metric": "memory"}
        )
        
        # AI Engine component failure
        engine_failure_rule = AlertRule(
            name="ai_engine_failure",
            description="Alert when AI Engine component is unhealthy",
            conditions=[
                AlertCondition(
                    metric_name="ai_engine_status",
                    operator=RuleOperator.EQUALS,
                    threshold="unhealthy",
                    evaluation_period=120,
                    data_points=1
                )
            ],
            severity=AlertSeverity.CRITICAL,
            labels={"component": "ai_engine"},
            cooldown_period=600  # 10 minutes
        )
        
        # High error rate
        error_rate_rule = AlertRule(
            name="high_error_rate",
            description="Alert when API error rate is high",
            conditions=[
                AlertCondition(
                    metric_name="api_error_rate",
                    operator=RuleOperator.GREATER_THAN,
                    threshold=5.0,  # 5% error rate
                    evaluation_period=300,
                    data_points=2
                )
            ],
            severity=AlertSeverity.ERROR,
            labels={"component": "api", "metric": "error_rate"}
        )
        
        # Add default rules
        for rule in [cpu_rule, memory_rule, engine_failure_rule, error_rate_rule]:
            self.add_alert_rule(rule)
    
    def _should_create_alert(self, rule: AlertRule, current_time: float) -> bool:
        """Check if we should create a new alert for this rule."""
        # Check if there are already too many active alerts for this rule
        rule_alerts = [a for a in self.active_alerts.values() if a.rule_name == rule.name]
        if len(rule_alerts) >= rule.max_alerts:
            return False
        
        # Check cooldown period
        for alert in rule_alerts:
            if current_time - alert.created_at < rule.cooldown_period:
                return False
        
        return True
    
    def _create_alert(self, rule: AlertRule, metrics_data: Dict[str, Any], current_time: float) -> Alert:
        """Create a new alert from rule and metrics."""
        alert_id = f"{rule.name}_{int(current_time * 1000000)}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            title=f"Alert: {rule.name}",
            description=rule.description,
            created_at=current_time,
            updated_at=current_time,
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy(),
            source_data=metrics_data.copy()
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Update statistics
        self.stats["total_alerts"] += 1
        self.stats["active_alerts"] += 1
        self.stats["alerts_by_severity"][rule.severity.value] += 1
        
        # Limit history size
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
        
        self.logger.warning(f"New alert created: {alert_id} ({rule.severity.value})")
        
        return alert
    
    async def _send_alert_notifications(self, alert: Alert) -> None:
        """Send notifications for alert to all appropriate channels."""
        for channel_name, channel in self.notification_channels.items():
            if channel.should_notify(alert):
                try:
                    success = await channel.provider.send_notification(alert)
                    if success:
                        self.stats["notifications_sent"] += 1
                        self.logger.info(f"Notification sent to {channel_name} for alert {alert.alert_id}")
                    else:
                        self.stats["notifications_failed"] += 1
                        self.logger.error(f"Failed to send notification to {channel_name} for alert {alert.alert_id}")
                        
                except Exception as e:
                    self.stats["notifications_failed"] += 1
                    self.logger.error(f"Error sending notification to {channel_name}: {e}")
    
    async def _evaluation_loop(self) -> None:
        """Background alert evaluation loop."""
        while not self._stop_evaluation.is_set():
            try:
                # This would get metrics from the metrics collector
                # For now, we'll use placeholder metrics
                metrics_data = await self._get_current_metrics()
                
                if metrics_data:
                    await self.evaluate_rules(metrics_data)
                
                await asyncio.sleep(self.evaluation_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(self.evaluation_interval)
    
    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics data for evaluation."""
        # This would integrate with the metrics collector
        # For now, return empty dict
        return {}


# Notification provider factory
def create_notification_provider(provider_type: str, name: str, config: Dict[str, Any]) -> NotificationProvider:
    """Create notification provider by type."""
    providers = {
        "email": EmailNotificationProvider,
        "slack": SlackNotificationProvider,
        "webhook": WebhookNotificationProvider
    }
    
    if provider_type not in providers:
        raise AlertException(f"Unknown notification provider type: {provider_type}")
    
    return providers[provider_type](name, config)