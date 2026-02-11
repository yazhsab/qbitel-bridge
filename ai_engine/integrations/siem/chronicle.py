"""
Google Chronicle SIEM Connector

Native integration with Google Chronicle:
- Ingestion API for event data
- Search API for UDM queries
- Detection API for rule management
- Entity context API for enrichment
"""

import asyncio
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
class ChronicleConfig(SIEMConfig):
    """Configuration for Google Chronicle connector."""

    # Google Cloud settings
    project_id: str = ""
    region: str = "us"  # us, europe, asia-southeast1
    instance_id: str = ""

    # Service account
    service_account_file: Optional[str] = None
    service_account_info: Optional[Dict[str, Any]] = None

    # API endpoints
    ingestion_endpoint: str = "https://malachiteingestion-pa.googleapis.com"
    backstory_endpoint: str = "https://{region}.backstory.chronicle.security"

    # Ingestion settings
    log_type: str = "QBITEL_AI"  # Chronicle log type
    namespace: str = "qbitel"
    forwarder_id: Optional[str] = None

    # Detection settings
    detection_rule_set: Optional[str] = None

    def __post_init__(self):
        self.backstory_endpoint = self.backstory_endpoint.format(region=self.region)


class ChronicleConnector(BaseSIEMConnector):
    """
    Google Chronicle SIEM connector.

    Supports:
    - Event ingestion via Ingestion API
    - UDM searches via Search API
    - Detection rule management
    - Entity enrichment
    - Case management (SOAR integration)

    Example:
        config = ChronicleConfig(
            project_id="your-project",
            region="us",
            instance_id="your-instance",
            service_account_file="/path/to/service-account.json"
        )

        connector = ChronicleConnector(config)
        await connector.connect()

        # Send events
        await connector.send_event(SIEMEvent(
            event_type="security.alert",
            message="Suspicious activity detected"
        ))
    """

    def __init__(self, config: ChronicleConfig):
        super().__init__(config)
        self.config: ChronicleConfig = config
        self._http_client = None
        self._credentials = None
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        # Set capabilities
        self._capabilities = (
            SIEMCapability.SEND_EVENTS
            | SIEMCapability.QUERY_EVENTS
            | SIEMCapability.QUERY_ALERTS
            | SIEMCapability.CREATE_ALERTS
            | SIEMCapability.STREAM_EVENTS
            | SIEMCapability.CORRELATION_RULES
        )

    async def connect(self) -> bool:
        """Connect to Google Chronicle."""
        self._status = ConnectionStatus.CONNECTING
        logger.info("Connecting to Google Chronicle")

        try:
            import httpx

            self._http_client = httpx.AsyncClient(
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )

            # Load Google credentials
            await self._load_credentials()

            # Verify connection
            if not await self.health_check():
                raise ConnectionError("Health check failed")

            self._status = ConnectionStatus.CONNECTED
            self._emit("connected")
            logger.info("Connected to Google Chronicle successfully")
            return True

        except ImportError:
            logger.error("httpx package required for Chronicle connector")
            self._status = ConnectionStatus.ERROR
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Chronicle: {e}")
            self._status = ConnectionStatus.ERROR
            self._emit("error", e)
            return False

    async def disconnect(self) -> None:
        """Disconnect from Chronicle."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._credentials = None
        self._access_token = None
        self._status = ConnectionStatus.DISCONNECTED
        self._emit("disconnected")
        logger.info("Disconnected from Google Chronicle")

    async def health_check(self) -> bool:
        """Check Chronicle connection health."""
        try:
            # Check backstory API availability
            url = f"{self.config.backstory_endpoint}/v1/tools/status"
            response = await self._make_request("GET", url)
            return response.get("status") == "OK"
        except Exception as e:
            logger.warning(f"Chronicle health check failed: {e}")
            # Try alternative check
            return self._credentials is not None

    async def send_event(self, event: SIEMEvent) -> bool:
        """Send a single event to Chronicle."""
        return await self.send_events([event]) == 1

    async def send_events(self, events: List[SIEMEvent]) -> int:
        """Send events to Chronicle via Ingestion API."""
        if not events:
            return 0

        try:
            # Convert to UDM format
            udm_events = [self._event_to_udm(e) for e in events]

            # Build request
            url = f"{self.config.ingestion_endpoint}/v2/unstructuredlogentries:batchCreate"

            entries = []
            for i, (event, udm) in enumerate(zip(events, udm_events)):
                entry = {
                    "logText": json.dumps(udm),
                    "tsNs": int(event.timestamp.timestamp() * 1e9),
                }
                entries.append(entry)

            payload = {
                "customerId": self.config.instance_id,
                "logType": self.config.log_type,
                "entries": entries,
            }

            if self.config.forwarder_id:
                payload["forwarderId"] = self.config.forwarder_id

            response = await self._make_request("POST", url, json=payload)

            if response:
                self._emit("event_sent", len(events))
                return len(events)

            return 0

        except Exception as e:
            logger.error(f"Failed to send events to Chronicle: {e}")
            return 0

    async def query(self, query: SIEMQuery) -> QueryResult:
        """Execute a UDM search query."""
        try:
            # Start search
            search_id = await self._start_search(query)

            # Wait for completion
            await self._wait_for_search(search_id)

            # Get results
            results = await self._get_search_results(search_id, query)

            return results

        except Exception as e:
            logger.error(f"Chronicle query failed: {e}")
            return QueryResult()

    async def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[List[EventSeverity]] = None,
        status: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SIEMAlert]:
        """Get alerts from Chronicle detections."""
        try:
            url = f"{self.config.backstory_endpoint}/v1/detections"

            params = {
                "pageSize": limit,
            }

            if start_time:
                params["startTime"] = start_time.isoformat() + "Z"
            if end_time:
                params["endTime"] = end_time.isoformat() + "Z"

            response = await self._make_request("GET", url, params=params)

            alerts = []
            for detection in response.get("detections", []):
                alert = self._parse_detection(detection)
                if alert:
                    # Filter by severity if specified
                    if severity and alert.severity not in severity:
                        continue
                    alerts.append(alert)

            return alerts[:limit]

        except Exception as e:
            logger.error(f"Failed to get Chronicle alerts: {e}")
            return []

    async def stream_events(
        self,
        query: Optional[SIEMQuery] = None,
    ) -> AsyncIterator[SIEMEvent]:
        """Stream events using Chronicle's streaming API."""
        if query is None:
            query = SIEMQuery(query="")

        try:
            # Use feed API for streaming
            url = f"{self.config.backstory_endpoint}/v1/feed/udmSearch:stream"

            params = {
                "query": query.query or "*",
            }

            if query.start_time:
                params["startTime"] = query.start_time.isoformat() + "Z"

            async with self._http_client.stream(
                "GET",
                url,
                params=params,
                headers=await self._get_auth_headers(),
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            event = self._parse_udm_event(data)
                            if event:
                                yield event
                        except json.JSONDecodeError:
                            continue

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Chronicle stream error: {e}")

    async def create_alert(
        self,
        title: str,
        description: str,
        severity: EventSeverity,
        events: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[SIEMAlert]:
        """Create a Chronicle case (SOAR integration)."""
        try:
            # Chronicle uses Cases for alert management
            url = f"{self.config.backstory_endpoint}/v1/cases"

            payload = {
                "displayName": title,
                "description": description,
                "priority": self._severity_to_priority(severity),
                "status": "OPEN",
            }

            if events:
                payload["relatedEventIds"] = events

            response = await self._make_request("POST", url, json=payload)

            if response:
                return SIEMAlert(
                    alert_id=response.get("name", str(uuid.uuid4())),
                    title=title,
                    description=description,
                    severity=severity,
                    status="OPEN",
                    raw=response,
                )

            return None

        except Exception as e:
            logger.error(f"Failed to create Chronicle case: {e}")
            return None

    async def create_detection_rule(
        self,
        rule_name: str,
        rule_text: str,
        alerting: bool = True,
    ) -> Optional[str]:
        """Create a Chronicle detection rule (YARA-L)."""
        try:
            url = f"{self.config.backstory_endpoint}/v2/detect/rules"

            payload = {
                "ruleText": rule_text,
                "metadata": {
                    "displayName": rule_name,
                    "author": "qbitel",
                },
                "alerting": alerting,
            }

            response = await self._make_request("POST", url, json=payload)
            return response.get("ruleId")

        except Exception as e:
            logger.error(f"Failed to create detection rule: {e}")
            return None

    async def get_entity_context(
        self,
        entity_type: str,  # IP, DOMAIN, FILE_HASH, USER
        entity_value: str,
    ) -> Dict[str, Any]:
        """Get enrichment context for an entity."""
        try:
            url = f"{self.config.backstory_endpoint}/v1/entities:summarize"

            params = {
                "entityIndicator.type": entity_type,
                "entityIndicator.value": entity_value,
            }

            response = await self._make_request("GET", url, params=params)
            return response

        except Exception as e:
            logger.error(f"Failed to get entity context: {e}")
            return {}

    async def _load_credentials(self) -> None:
        """Load Google Cloud credentials."""
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request

            if self.config.service_account_file:
                self._credentials = service_account.Credentials.from_service_account_file(
                    self.config.service_account_file,
                    scopes=["https://www.googleapis.com/auth/chronicle-backstory"],
                )
            elif self.config.service_account_info:
                self._credentials = service_account.Credentials.from_service_account_info(
                    self.config.service_account_info,
                    scopes=["https://www.googleapis.com/auth/chronicle-backstory"],
                )
            else:
                # Use default credentials
                from google.auth import default
                self._credentials, _ = default(
                    scopes=["https://www.googleapis.com/auth/chronicle-backstory"]
                )

        except ImportError:
            logger.warning("google-auth package not installed, using API key auth")
            self._credentials = None

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        headers = {"Content-Type": "application/json"}

        if self._credentials:
            if not self._credentials.valid:
                from google.auth.transport.requests import Request
                self._credentials.refresh(Request())

            headers["Authorization"] = f"Bearer {self._credentials.token}"
        elif self.config.api_key:
            headers["X-API-Key"] = self.config.api_key

        return headers

    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make authenticated API request."""
        headers = await self._get_auth_headers()
        headers.update(kwargs.pop("headers", {}))

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

    async def _start_search(self, query: SIEMQuery) -> str:
        """Start a UDM search."""
        url = f"{self.config.backstory_endpoint}/v2/detect/rules:runRetrohunt"

        # For UDM searches
        url = f"{self.config.backstory_endpoint}/v1/udmSearch"

        params = {
            "query": query.query,
            "pageSize": query.limit,
        }

        if query.start_time:
            params["timeRange.startTime"] = query.start_time.isoformat() + "Z"
        if query.end_time:
            params["timeRange.endTime"] = query.end_time.isoformat() + "Z"

        response = await self._make_request("GET", url, params=params)
        return response.get("events", []), response.get("moreDataAvailable", False)

    async def _wait_for_search(self, search_id: str) -> None:
        """Wait for search to complete."""
        # Chronicle searches are synchronous for most queries
        pass

    async def _get_search_results(
        self,
        search_response: Any,
        query: SIEMQuery,
    ) -> QueryResult:
        """Get search results."""
        if isinstance(search_response, tuple):
            events_data, has_more = search_response
        else:
            events_data = search_response.get("events", [])
            has_more = search_response.get("moreDataAvailable", False)

        events = []
        for event_data in events_data:
            event = self._parse_udm_event(event_data)
            if event:
                events.append(event)

        return QueryResult(
            events=events,
            total_count=len(events),
            returned_count=len(events),
            has_more=has_more,
        )

    def _event_to_udm(self, event: SIEMEvent) -> Dict[str, Any]:
        """Convert SIEMEvent to UDM (Unified Data Model) format."""
        udm = {
            "metadata": {
                "eventTimestamp": event.timestamp.isoformat() + "Z",
                "eventType": self._get_udm_event_type(event.event_type),
                "productName": "QBITEL",
                "vendorName": "QBITEL",
                "description": event.message,
            },
            "securityResult": [{
                "severity": self._severity_to_udm(event.severity),
                "category": event.category.upper() if event.category else "UNKNOWN",
                "summary": event.message,
            }],
            "additional": {
                "fields": {
                    "event_type": {"stringValue": event.event_type},
                    "correlation_id": {"stringValue": event.correlation_id},
                    "source": {"stringValue": event.source},
                    **{k: {"stringValue": str(v)} for k, v in event.data.items()},
                }
            },
        }

        # Add principal (source entity)
        if event.src_ip or event.user:
            udm["principal"] = {}
            if event.src_ip:
                udm["principal"]["ip"] = [event.src_ip]
            if event.user:
                udm["principal"]["user"] = {"userid": event.user}

        # Add target (destination entity)
        if event.dst_ip or event.asset_name:
            udm["target"] = {}
            if event.dst_ip:
                udm["target"]["ip"] = [event.dst_ip]
            if event.asset_name:
                udm["target"]["hostname"] = event.asset_name

        return udm

    def _parse_udm_event(self, data: Dict[str, Any]) -> Optional[SIEMEvent]:
        """Parse UDM event into SIEMEvent."""
        try:
            metadata = data.get("udm", {}).get("metadata", {})
            security_result = data.get("udm", {}).get("securityResult", [{}])[0]
            principal = data.get("udm", {}).get("principal", {})
            target = data.get("udm", {}).get("target", {})

            timestamp_str = metadata.get("eventTimestamp", "")
            timestamp = (
                datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if timestamp_str
                else datetime.utcnow()
            )

            return SIEMEvent(
                event_type=metadata.get("eventType", "GENERIC_EVENT"),
                message=metadata.get("description", ""),
                timestamp=timestamp,
                severity=self._udm_to_severity(security_result.get("severity")),
                category=security_result.get("category", ""),
                src_ip=principal.get("ip", [None])[0] if principal.get("ip") else None,
                dst_ip=target.get("ip", [None])[0] if target.get("ip") else None,
                user=principal.get("user", {}).get("userid"),
                data=data,
            )
        except Exception as e:
            logger.warning(f"Failed to parse UDM event: {e}")
            return None

    def _parse_detection(self, detection: Dict[str, Any]) -> Optional[SIEMAlert]:
        """Parse Chronicle detection into SIEMAlert."""
        try:
            return SIEMAlert(
                alert_id=detection.get("id", str(uuid.uuid4())),
                title=detection.get("detection", [{}])[0].get("ruleName", ""),
                description=detection.get("detection", [{}])[0].get("description", ""),
                severity=self._udm_to_severity(
                    detection.get("detection", [{}])[0].get("severity")
                ),
                status="new",
                created_at=datetime.fromisoformat(
                    detection.get("detectionTime", "").replace("Z", "+00:00")
                ) if detection.get("detectionTime") else datetime.utcnow(),
                rule_id=detection.get("detection", [{}])[0].get("ruleId"),
                rule_name=detection.get("detection", [{}])[0].get("ruleName"),
                raw=detection,
            )
        except Exception as e:
            logger.warning(f"Failed to parse detection: {e}")
            return None

    def _get_udm_event_type(self, event_type: str) -> str:
        """Map event type to UDM event type."""
        type_map = {
            "authentication": "USER_LOGIN",
            "login": "USER_LOGIN",
            "logout": "USER_LOGOUT",
            "file_access": "FILE_READ",
            "network": "NETWORK_CONNECTION",
            "process": "PROCESS_LAUNCH",
            "alert": "STATUS_UPDATE",
            "security": "STATUS_UPDATE",
        }

        for key, udm_type in type_map.items():
            if key in event_type.lower():
                return udm_type

        return "GENERIC_EVENT"

    def _severity_to_udm(self, severity: EventSeverity) -> str:
        """Convert severity to UDM format."""
        severity_map = {
            EventSeverity.CRITICAL: "CRITICAL",
            EventSeverity.HIGH: "HIGH",
            EventSeverity.MEDIUM: "MEDIUM",
            EventSeverity.LOW: "LOW",
            EventSeverity.INFO: "INFORMATIONAL",
            EventSeverity.DEBUG: "INFORMATIONAL",
        }
        return severity_map.get(severity, "UNKNOWN_SEVERITY")

    def _udm_to_severity(self, udm_severity: Optional[str]) -> EventSeverity:
        """Convert UDM severity to enum."""
        severity_map = {
            "CRITICAL": EventSeverity.CRITICAL,
            "HIGH": EventSeverity.HIGH,
            "MEDIUM": EventSeverity.MEDIUM,
            "LOW": EventSeverity.LOW,
            "INFORMATIONAL": EventSeverity.INFO,
        }
        return severity_map.get((udm_severity or "").upper(), EventSeverity.MEDIUM)

    def _severity_to_priority(self, severity: EventSeverity) -> str:
        """Convert severity to Chronicle priority."""
        priority_map = {
            EventSeverity.CRITICAL: "P1",
            EventSeverity.HIGH: "P2",
            EventSeverity.MEDIUM: "P3",
            EventSeverity.LOW: "P4",
            EventSeverity.INFO: "P4",
            EventSeverity.DEBUG: "P4",
        }
        return priority_map.get(severity, "P3")
