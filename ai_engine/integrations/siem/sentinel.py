"""
Microsoft Sentinel SIEM Connector

Native integration with Microsoft Sentinel:
- Log Analytics Data Collector API for event ingestion
- Azure Resource Graph API for queries
- Microsoft Graph Security API for alerts
- Logic Apps/Automation Rules integration
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional

from ai_engine.integrations.siem.base import (
    BaseSIEMConnector,
    SIEMConfig,
    SIEMEvent,
    SIEMAlert,
    SIEMQuery,
    QueryResult,
    ConnectionStatus,
    SIEMCapability,
    EventSeverity,
)

logger = logging.getLogger(__name__)


@dataclass
class SentinelConfig(SIEMConfig):
    """Configuration for Microsoft Sentinel connector."""

    # Azure settings
    subscription_id: str = ""
    resource_group: str = ""
    workspace_name: str = ""
    workspace_id: str = ""

    # Log Analytics settings
    log_analytics_key: str = ""  # Primary or secondary key
    custom_log_name: str = "QbitelAI"  # Custom log table name

    # Azure AD settings
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""

    # API endpoints
    log_analytics_endpoint: str = "https://{workspace_id}.ods.opinsights.azure.com"
    graph_endpoint: str = "https://graph.microsoft.com"
    management_endpoint: str = "https://management.azure.com"

    # Sentinel-specific
    rule_action_group_id: Optional[str] = None
    automation_rule_id: Optional[str] = None

    def __post_init__(self):
        if self.workspace_id:
            self.log_analytics_endpoint = self.log_analytics_endpoint.format(workspace_id=self.workspace_id)


class SentinelConnector(BaseSIEMConnector):
    """
    Microsoft Sentinel SIEM connector.

    Supports:
    - Event ingestion via Log Analytics Data Collector API
    - KQL queries via Log Analytics
    - Alert management via Graph Security API
    - Incident management
    - Watchlist operations

    Example:
        config = SentinelConfig(
            workspace_id="your-workspace-id",
            log_analytics_key="your-primary-key",
            tenant_id="your-tenant-id",
            client_id="your-client-id",
            client_secret="your-client-secret"
        )

        connector = SentinelConnector(config)
        await connector.connect()

        # Send events
        await connector.send_event(SIEMEvent(
            event_type="security.alert",
            message="Suspicious activity detected"
        ))
    """

    def __init__(self, config: SentinelConfig):
        super().__init__(config)
        self.config: SentinelConfig = config
        self._http_client = None
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        # Set capabilities
        self._capabilities = (
            SIEMCapability.SEND_EVENTS
            | SIEMCapability.QUERY_EVENTS
            | SIEMCapability.QUERY_ALERTS
            | SIEMCapability.CREATE_ALERTS
            | SIEMCapability.CASES
        )

    async def connect(self) -> bool:
        """Connect to Microsoft Sentinel."""
        self._status = ConnectionStatus.CONNECTING
        logger.info("Connecting to Microsoft Sentinel")

        try:
            import httpx

            self._http_client = httpx.AsyncClient(
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )

            # Get Azure AD token for API access
            await self._acquire_token()

            # Verify connection
            if not await self.health_check():
                raise ConnectionError("Health check failed")

            self._status = ConnectionStatus.CONNECTED
            self._emit("connected")
            logger.info("Connected to Microsoft Sentinel successfully")
            return True

        except ImportError:
            logger.error("httpx package required for Sentinel connector")
            self._status = ConnectionStatus.ERROR
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Sentinel: {e}")
            self._status = ConnectionStatus.ERROR
            self._emit("error", e)
            return False

    async def disconnect(self) -> None:
        """Disconnect from Sentinel."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._access_token = None
        self._token_expiry = None
        self._status = ConnectionStatus.DISCONNECTED
        self._emit("disconnected")
        logger.info("Disconnected from Microsoft Sentinel")

    async def health_check(self) -> bool:
        """Check Sentinel connection health."""
        try:
            # Query Log Analytics workspace info
            url = (
                f"{self.config.management_endpoint}/subscriptions/{self.config.subscription_id}"
                f"/resourceGroups/{self.config.resource_group}"
                f"/providers/Microsoft.OperationalInsights/workspaces/{self.config.workspace_name}"
                f"?api-version=2021-12-01-preview"
            )

            response = await self._make_request("GET", url)
            return response.get("properties", {}).get("provisioningState") == "Succeeded"
        except Exception as e:
            logger.warning(f"Sentinel health check failed: {e}")
            return False

    async def send_event(self, event: SIEMEvent) -> bool:
        """Send a single event to Sentinel."""
        return await self.send_events([event]) == 1

    async def send_events(self, events: List[SIEMEvent]) -> int:
        """Send events to Sentinel via Log Analytics Data Collector API."""
        if not events:
            return 0

        try:
            # Build payload
            payload = json.dumps([self._event_to_log_analytics(e) for e in events])

            # Build signature
            timestamp = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
            signature = self._build_signature(
                self.config.workspace_id,
                self.config.log_analytics_key,
                timestamp,
                len(payload),
            )

            # Send to Log Analytics
            url = f"{self.config.log_analytics_endpoint}/api/logs?api-version=2016-04-01"
            headers = {
                "Authorization": signature,
                "Log-Type": self.config.custom_log_name,
                "x-ms-date": timestamp,
                "Content-Type": "application/json",
                "time-generated-field": "TimeGenerated",
            }

            response = await self._http_client.post(
                url,
                content=payload,
                headers=headers,
            )

            if response.status_code == 200:
                self._emit("event_sent", len(events))
                return len(events)

            logger.error(f"Log Analytics error: {response.status_code} - {response.text}")
            return 0

        except Exception as e:
            logger.error(f"Failed to send events to Sentinel: {e}")
            return 0

    async def query(self, query: SIEMQuery) -> QueryResult:
        """Execute a KQL query against Log Analytics."""
        try:
            # Build KQL query
            kql = self._build_kql_query(query)

            # Execute query
            url = f"https://api.loganalytics.io/v1/workspaces/{self.config.workspace_id}/query"

            payload = {
                "query": kql,
            }

            if query.start_time and query.end_time:
                payload["timespan"] = f"{query.start_time.isoformat()}/" f"{query.end_time.isoformat()}"

            response = await self._make_request("POST", url, json=payload)

            # Parse results
            events = self._parse_query_results(response)

            return QueryResult(
                events=events,
                total_count=len(events),
                returned_count=len(events),
                raw=response,
            )

        except Exception as e:
            logger.error(f"Sentinel query failed: {e}")
            return QueryResult()

    async def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[List[EventSeverity]] = None,
        status: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SIEMAlert]:
        """Get alerts from Sentinel via Graph Security API."""
        try:
            # Build filter
            filters = []

            if start_time:
                filters.append(f"createdDateTime ge {start_time.isoformat()}")
            if end_time:
                filters.append(f"createdDateTime le {end_time.isoformat()}")
            if severity:
                severity_values = [self._severity_to_graph(s) for s in severity]
                filters.append(f"severity in ({','.join(severity_values)})")
            if status:
                filters.append(f"status in ({','.join(status)})")

            filter_str = " and ".join(filters) if filters else None

            # Query Graph Security API
            url = f"{self.config.graph_endpoint}/v1.0/security/alerts_v2"
            params = {
                "$top": limit,
                "$orderby": "createdDateTime desc",
            }
            if filter_str:
                params["$filter"] = filter_str

            response = await self._make_request("GET", url, params=params)

            # Parse alerts
            alerts = []
            for alert_data in response.get("value", []):
                alert = self._parse_graph_alert(alert_data)
                if alert:
                    alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"Failed to get Sentinel alerts: {e}")
            return []

    async def get_incidents(
        self,
        start_time: Optional[datetime] = None,
        status: Optional[List[str]] = None,
        severity: Optional[List[EventSeverity]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get Sentinel incidents."""
        try:
            url = (
                f"{self.config.management_endpoint}/subscriptions/{self.config.subscription_id}"
                f"/resourceGroups/{self.config.resource_group}"
                f"/providers/Microsoft.OperationalInsights/workspaces/{self.config.workspace_name}"
                f"/providers/Microsoft.SecurityInsights/incidents"
                f"?api-version=2022-11-01"
            )

            params = {"$top": limit, "$orderby": "properties/createdTimeUtc desc"}

            # Build filter
            filters = []
            if status:
                filters.append(f"properties/status in ({','.join(status)})")
            if severity:
                sev_values = [self._severity_to_sentinel(s) for s in severity]
                filters.append(f"properties/severity in ({','.join(sev_values)})")

            if filters:
                params["$filter"] = " and ".join(filters)

            response = await self._make_request("GET", url, params=params)
            return response.get("value", [])

        except Exception as e:
            logger.error(f"Failed to get Sentinel incidents: {e}")
            return []

    async def create_alert(
        self,
        title: str,
        description: str,
        severity: EventSeverity,
        events: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[SIEMAlert]:
        """Create a Sentinel incident."""
        try:
            incident_id = str(uuid.uuid4())
            url = (
                f"{self.config.management_endpoint}/subscriptions/{self.config.subscription_id}"
                f"/resourceGroups/{self.config.resource_group}"
                f"/providers/Microsoft.OperationalInsights/workspaces/{self.config.workspace_name}"
                f"/providers/Microsoft.SecurityInsights/incidents/{incident_id}"
                f"?api-version=2022-11-01"
            )

            payload = {
                "properties": {
                    "title": title,
                    "description": description,
                    "severity": self._severity_to_sentinel(severity),
                    "status": "New",
                    "classification": kwargs.get("classification"),
                    "classificationComment": kwargs.get("classification_comment"),
                }
            }

            response = await self._make_request("PUT", url, json=payload)

            if response:
                return SIEMAlert(
                    alert_id=response.get("name", incident_id),
                    title=title,
                    description=description,
                    severity=severity,
                    status="New",
                    raw=response,
                )

            return None

        except Exception as e:
            logger.error(f"Failed to create Sentinel incident: {e}")
            return None

    async def update_incident(
        self,
        incident_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update a Sentinel incident."""
        try:
            url = (
                f"{self.config.management_endpoint}/subscriptions/{self.config.subscription_id}"
                f"/resourceGroups/{self.config.resource_group}"
                f"/providers/Microsoft.OperationalInsights/workspaces/{self.config.workspace_name}"
                f"/providers/Microsoft.SecurityInsights/incidents/{incident_id}"
                f"?api-version=2022-11-01"
            )

            # Get current incident
            current = await self._make_request("GET", url)

            # Merge updates
            properties = current.get("properties", {})
            properties.update(updates)

            payload = {"properties": properties}

            await self._make_request("PUT", url, json=payload)
            return True

        except Exception as e:
            logger.error(f"Failed to update incident: {e}")
            return False

    async def add_watchlist_item(
        self,
        watchlist_alias: str,
        item_data: Dict[str, Any],
    ) -> bool:
        """Add item to Sentinel watchlist."""
        try:
            item_id = str(uuid.uuid4())
            url = (
                f"{self.config.management_endpoint}/subscriptions/{self.config.subscription_id}"
                f"/resourceGroups/{self.config.resource_group}"
                f"/providers/Microsoft.OperationalInsights/workspaces/{self.config.workspace_name}"
                f"/providers/Microsoft.SecurityInsights/watchlists/{watchlist_alias}"
                f"/watchlistItems/{item_id}"
                f"?api-version=2022-11-01"
            )

            payload = {
                "properties": {
                    "itemsKeyValue": item_data,
                }
            }

            await self._make_request("PUT", url, json=payload)
            return True

        except Exception as e:
            logger.error(f"Failed to add watchlist item: {e}")
            return False

    async def _acquire_token(self) -> None:
        """Acquire Azure AD access token."""
        if self._access_token and self._token_expiry:
            if datetime.utcnow() < self._token_expiry - timedelta(minutes=5):
                return

        url = f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0/token"

        data = {
            "grant_type": "client_credentials",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "scope": "https://management.azure.com/.default",
        }

        response = await self._http_client.post(url, data=data)
        response.raise_for_status()

        result = response.json()
        self._access_token = result["access_token"]
        self._token_expiry = datetime.utcnow() + timedelta(seconds=result.get("expires_in", 3600))

    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make authenticated API request."""
        await self._acquire_token()

        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._access_token}"
        headers["Content-Type"] = "application/json"

        if method == "GET":
            response = await self._http_client.get(url, headers=headers, **kwargs)
        elif method == "POST":
            response = await self._http_client.post(url, headers=headers, **kwargs)
        elif method == "PUT":
            response = await self._http_client.put(url, headers=headers, **kwargs)
        elif method == "DELETE":
            response = await self._http_client.delete(url, headers=headers, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code == 204:
            return {}

        response.raise_for_status()
        return response.json()

    def _build_signature(
        self,
        workspace_id: str,
        shared_key: str,
        date: str,
        content_length: int,
    ) -> str:
        """Build HMAC-SHA256 signature for Log Analytics."""
        method = "POST"
        content_type = "application/json"
        resource = "/api/logs"

        string_to_sign = f"{method}\n{content_length}\n{content_type}\n" f"x-ms-date:{date}\n{resource}"

        decoded_key = base64.b64decode(shared_key)
        encoded_hash = base64.b64encode(
            hmac.new(
                decoded_key,
                string_to_sign.encode("utf-8"),
                digestmod=hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        return f"SharedKey {workspace_id}:{encoded_hash}"

    def _event_to_log_analytics(self, event: SIEMEvent) -> Dict[str, Any]:
        """Convert SIEMEvent to Log Analytics format."""
        return {
            "TimeGenerated": event.timestamp.isoformat(),
            "EventType": event.event_type,
            "Message": event.message,
            "Severity": event.severity.value,
            "Category": event.category,
            "Subcategory": event.subcategory,
            "Source": event.source,
            "SourceType": event.source_type,
            "Host": event.host,
            "EventId": event.event_id,
            "CorrelationId": event.correlation_id,
            "User": event.user,
            "UserId": event.user_id,
            "SourceIP": event.src_ip,
            "DestinationIP": event.dst_ip,
            "AssetId": event.asset_id,
            "AssetName": event.asset_name,
            "Labels": json.dumps(event.labels),
            "Tags": json.dumps(event.tags),
            "Data": json.dumps(event.data),
        }

    def _build_kql_query(self, query: SIEMQuery) -> str:
        """Build KQL query string."""
        kql = query.query

        # If no table specified, use custom log
        if not any(kql.startswith(t) for t in ["SecurityEvent", "SigninLogs", "AuditLogs"]):
            if not kql.startswith(self.config.custom_log_name):
                kql = f"{self.config.custom_log_name}_CL | {kql}"

        # Add time filter
        if query.start_time and query.end_time:
            time_filter = (
                f"| where TimeGenerated between "
                f"(datetime({query.start_time.isoformat()}) .. datetime({query.end_time.isoformat()}))"
            )
            kql = f"{kql} {time_filter}"

        # Add limit
        if query.limit:
            kql = f"{kql} | take {query.limit}"

        return kql

    def _parse_query_results(self, response: Dict[str, Any]) -> List[SIEMEvent]:
        """Parse KQL query results."""
        events = []
        tables = response.get("tables", [])

        for table in tables:
            columns = [col["name"] for col in table.get("columns", [])]
            for row in table.get("rows", []):
                row_dict = dict(zip(columns, row))
                event = self._parse_log_analytics_event(row_dict)
                if event:
                    events.append(event)

        return events

    def _parse_log_analytics_event(self, row: Dict[str, Any]) -> Optional[SIEMEvent]:
        """Parse Log Analytics row into SIEMEvent."""
        try:
            timestamp_str = row.get("TimeGenerated", "")
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")) if timestamp_str else datetime.utcnow()

            return SIEMEvent(
                event_type=row.get("EventType", "unknown"),
                message=row.get("Message", ""),
                timestamp=timestamp,
                severity=self._parse_severity(row.get("Severity")),
                category=row.get("Category", ""),
                host=row.get("Host", ""),
                data=row,
            )
        except Exception as e:
            logger.warning(f"Failed to parse Log Analytics event: {e}")
            return None

    def _parse_graph_alert(self, alert_data: Dict[str, Any]) -> Optional[SIEMAlert]:
        """Parse Graph Security API alert."""
        try:
            return SIEMAlert(
                alert_id=alert_data.get("id", ""),
                title=alert_data.get("title", ""),
                description=alert_data.get("description", ""),
                severity=self._graph_to_severity(alert_data.get("severity", "medium")),
                status=alert_data.get("status", "new"),
                created_at=datetime.fromisoformat(alert_data.get("createdDateTime", "").replace("Z", "+00:00")),
                rule_name=alert_data.get("detectionSource", ""),
                raw=alert_data,
            )
        except Exception as e:
            logger.warning(f"Failed to parse Graph alert: {e}")
            return None

    def _parse_severity(self, severity: Optional[str]) -> EventSeverity:
        """Parse severity string."""
        severity_map = {
            "critical": EventSeverity.CRITICAL,
            "high": EventSeverity.HIGH,
            "medium": EventSeverity.MEDIUM,
            "low": EventSeverity.LOW,
            "informational": EventSeverity.INFO,
        }
        return severity_map.get((severity or "").lower(), EventSeverity.INFO)

    def _severity_to_sentinel(self, severity: EventSeverity) -> str:
        """Convert severity to Sentinel format."""
        sentinel_map = {
            EventSeverity.CRITICAL: "High",
            EventSeverity.HIGH: "High",
            EventSeverity.MEDIUM: "Medium",
            EventSeverity.LOW: "Low",
            EventSeverity.INFO: "Informational",
            EventSeverity.DEBUG: "Informational",
        }
        return sentinel_map.get(severity, "Medium")

    def _severity_to_graph(self, severity: EventSeverity) -> str:
        """Convert severity to Graph Security format."""
        return f"'{self._severity_to_sentinel(severity).lower()}'"

    def _graph_to_severity(self, severity: str) -> EventSeverity:
        """Convert Graph Security severity to enum."""
        severity_map = {
            "high": EventSeverity.HIGH,
            "medium": EventSeverity.MEDIUM,
            "low": EventSeverity.LOW,
            "informational": EventSeverity.INFO,
        }
        return severity_map.get(severity.lower(), EventSeverity.MEDIUM)
