"""
Splunk SIEM Connector

Native integration with Splunk Enterprise and Splunk Cloud:
- HTTP Event Collector (HEC) for event ingestion
- REST API for searches and alerts
- Modular inputs for real-time data
- KV Store for correlation data
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urljoin

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
class SplunkHECConfig:
    """Splunk HEC-specific configuration."""

    hec_token: str = ""
    hec_endpoint: str = "/services/collector/event"
    hec_raw_endpoint: str = "/services/collector/raw"
    use_raw: bool = False  # Use raw endpoint for higher throughput
    channel: Optional[str] = None  # HEC channel for indexer acknowledgment
    ack_enabled: bool = False


@dataclass
class SplunkConfig(SIEMConfig):
    """Configuration for Splunk connector."""

    # Splunk-specific settings
    app: str = "search"
    owner: str = "admin"

    # HEC settings
    hec: SplunkHECConfig = field(default_factory=SplunkHECConfig)

    # Search settings
    search_mode: str = "normal"  # normal, realtime
    max_time: int = 300  # Max search time
    earliest_time: str = "-24h"
    latest_time: str = "now"

    # Index settings
    index: str = "main"
    sourcetype: str = "qbitel:security"

    def __post_init__(self):
        if not self.source:
            self.source = "qbitel"
        if not self.source_type:
            self.source_type = self.sourcetype


class SplunkConnector(BaseSIEMConnector):
    """
    Splunk SIEM connector.

    Supports:
    - Event ingestion via HEC
    - Search queries via REST API
    - Alert management
    - Real-time search streaming
    - KV Store operations

    Example:
        config = SplunkConfig(
            host="splunk.example.com",
            port=8088,
            hec=SplunkHECConfig(hec_token="your-hec-token"),
            token="your-api-token"
        )

        connector = SplunkConnector(config)
        await connector.connect()

        # Send events
        await connector.send_event(SIEMEvent(
            event_type="security.alert",
            message="Suspicious activity detected"
        ))

        # Search
        results = await connector.query(SIEMQuery(
            query="index=main sourcetype=qbitel:security | stats count by event_type"
        ))
    """

    def __init__(self, config: SplunkConfig):
        super().__init__(config)
        self.config: SplunkConfig = config
        self._http_client = None
        self._session_key: Optional[str] = None

        # Set capabilities
        self._capabilities = (
            SIEMCapability.SEND_EVENTS
            | SIEMCapability.QUERY_EVENTS
            | SIEMCapability.QUERY_ALERTS
            | SIEMCapability.CREATE_ALERTS
            | SIEMCapability.STREAM_EVENTS
        )

    async def connect(self) -> bool:
        """Connect to Splunk."""
        self._status = ConnectionStatus.CONNECTING
        logger.info(f"Connecting to Splunk at {self.config.host}")

        try:
            import httpx

            # Create HTTP client
            self._http_client = httpx.AsyncClient(
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )

            # Authenticate
            if self.config.auth_type == "token":
                self._session_key = self.config.token
            else:
                await self._authenticate()

            # Verify connection
            if not await self.health_check():
                raise ConnectionError("Health check failed")

            self._status = ConnectionStatus.CONNECTED
            self._emit("connected")
            logger.info("Connected to Splunk successfully")
            return True

        except ImportError:
            logger.error("httpx package required for Splunk connector")
            self._status = ConnectionStatus.ERROR
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Splunk: {e}")
            self._status = ConnectionStatus.ERROR
            self._emit("error", e)
            return False

    async def disconnect(self) -> None:
        """Disconnect from Splunk."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._session_key = None
        self._status = ConnectionStatus.DISCONNECTED
        self._emit("disconnected")
        logger.info("Disconnected from Splunk")

    async def health_check(self) -> bool:
        """Check Splunk connection health."""
        try:
            response = await self._api_request(
                "GET",
                "/services/server/info",
                output_mode="json",
            )
            return response.get("entry") is not None
        except Exception as e:
            logger.warning(f"Splunk health check failed: {e}")
            return False

    async def send_event(self, event: SIEMEvent) -> bool:
        """Send a single event to Splunk via HEC."""
        return await self.send_events([event]) == 1

    async def send_events(self, events: List[SIEMEvent]) -> int:
        """Send multiple events to Splunk via HEC."""
        if not events:
            return 0

        try:
            # Build HEC payload
            if self.config.hec.use_raw:
                payload = self._build_raw_payload(events)
                endpoint = self.config.hec.hec_raw_endpoint
            else:
                payload = self._build_hec_payload(events)
                endpoint = self.config.hec.hec_endpoint

            # Send to HEC
            hec_url = self._build_hec_url(endpoint)
            headers = {
                "Authorization": f"Splunk {self.config.hec.hec_token}",
                "Content-Type": "application/json",
            }

            if self.config.hec.channel:
                headers["X-Splunk-Request-Channel"] = self.config.hec.channel

            response = await self._http_client.post(
                hec_url,
                content=payload,
                headers=headers,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    self._emit("event_sent", len(events))
                    return len(events)

            logger.error(f"HEC error: {response.status_code} - {response.text}")
            return 0

        except Exception as e:
            logger.error(f"Failed to send events to Splunk: {e}")
            return 0

    async def query(self, query: SIEMQuery) -> QueryResult:
        """Execute a search query in Splunk."""
        try:
            # Build SPL query
            spl = self._build_spl_query(query)

            # Create search job
            job_id = await self._create_search_job(spl, query)

            # Wait for completion
            await self._wait_for_job(job_id)

            # Get results
            results = await self._get_search_results(job_id, query)

            return results

        except Exception as e:
            logger.error(f"Splunk query failed: {e}")
            return QueryResult()

    async def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[List[EventSeverity]] = None,
        status: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SIEMAlert]:
        """Get alerts from Splunk."""
        try:
            # Get triggered alerts
            response = await self._api_request(
                "GET",
                "/services/alerts/fired_alerts",
                output_mode="json",
                count=limit,
            )

            alerts = []
            for entry in response.get("entry", []):
                alert = self._parse_alert(entry)
                if alert:
                    alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"Failed to get Splunk alerts: {e}")
            return []

    async def stream_events(
        self,
        query: Optional[SIEMQuery] = None,
    ) -> AsyncIterator[SIEMEvent]:
        """Stream events using real-time search."""
        if query is None:
            query = SIEMQuery(query="*")

        spl = self._build_spl_query(query)
        spl = f"search {spl}"

        try:
            # Create real-time search
            job_id = await self._create_search_job(
                spl,
                query,
                search_mode="realtime",
            )

            # Stream results
            while True:
                results = await self._get_search_results(
                    job_id,
                    query,
                    offset=0,
                    count=100,
                )

                for event in results.events:
                    yield event

                await asyncio.sleep(1)

        except asyncio.CancelledError:
            # Cleanup job
            await self._cancel_search_job(job_id)
            raise

    async def create_alert(
        self,
        title: str,
        description: str,
        severity: EventSeverity,
        events: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[SIEMAlert]:
        """Create a notable event in Splunk ES."""
        try:
            # Create notable event
            data = {
                "rule_name": title,
                "rule_description": description,
                "urgency": self._severity_to_urgency(severity),
                "security_domain": kwargs.get("security_domain", "threat"),
                "status": "1",  # New
            }

            if events:
                data["orig_event_ids"] = ",".join(events)

            response = await self._api_request(
                "POST",
                "/services/notable_update",
                **data,
            )

            if response:
                return SIEMAlert(
                    alert_id=response.get("notable_id", str(uuid.uuid4())),
                    title=title,
                    description=description,
                    severity=severity,
                    events=events or [],
                )

            return None

        except Exception as e:
            logger.error(f"Failed to create Splunk alert: {e}")
            return None

    async def _authenticate(self) -> None:
        """Authenticate with Splunk REST API."""
        auth_url = self._build_api_url("/services/auth/login")

        response = await self._http_client.post(
            auth_url,
            data={
                "username": self.config.username,
                "password": self.config.password,
                "output_mode": "json",
            },
        )

        if response.status_code != 200:
            raise ConnectionError(f"Authentication failed: {response.text}")

        result = response.json()
        self._session_key = result.get("sessionKey")

        if not self._session_key:
            raise ConnectionError("No session key in response")

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        **params,
    ) -> Dict[str, Any]:
        """Make an API request to Splunk."""
        url = self._build_api_url(endpoint)
        headers = {"Authorization": f"Bearer {self._session_key}"}

        if "output_mode" not in params:
            params["output_mode"] = "json"

        if method == "GET":
            response = await self._http_client.get(url, params=params, headers=headers)
        elif method == "POST":
            response = await self._http_client.post(url, data=params, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()

    async def _create_search_job(
        self,
        spl: str,
        query: SIEMQuery,
        search_mode: str = "normal",
    ) -> str:
        """Create a search job."""
        params = {
            "search": spl,
            "exec_mode": "normal" if search_mode == "normal" else "blocking",
            "max_time": self.config.max_time,
        }

        if query.start_time:
            params["earliest_time"] = query.start_time.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            params["earliest_time"] = self.config.earliest_time

        if query.end_time:
            params["latest_time"] = query.end_time.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            params["latest_time"] = self.config.latest_time

        response = await self._api_request("POST", "/services/search/jobs", **params)
        return response.get("sid", "")

    async def _wait_for_job(self, job_id: str) -> None:
        """Wait for search job to complete."""
        while True:
            response = await self._api_request(
                "GET",
                f"/services/search/jobs/{job_id}",
            )

            entry = response.get("entry", [{}])[0]
            content = entry.get("content", {})

            if content.get("isDone"):
                break

            if content.get("isFailed"):
                raise RuntimeError(f"Search job failed: {content.get('messages')}")

            await asyncio.sleep(0.5)

    async def _get_search_results(
        self,
        job_id: str,
        query: SIEMQuery,
        offset: int = 0,
        count: Optional[int] = None,
    ) -> QueryResult:
        """Get results from a search job."""
        params = {
            "offset": offset or query.offset,
            "count": count or query.limit,
        }

        if query.fields:
            params["field_list"] = ",".join(query.fields)

        response = await self._api_request(
            "GET",
            f"/services/search/jobs/{job_id}/results",
            **params,
        )

        # Parse results
        events = []
        for result in response.get("results", []):
            event = self._parse_event(result)
            if event:
                events.append(event)

        return QueryResult(
            events=events,
            total_count=int(response.get("post_process_count", len(events))),
            returned_count=len(events),
            has_more=len(events) == query.limit,
            raw=response,
        )

    async def _cancel_search_job(self, job_id: str) -> None:
        """Cancel a search job."""
        try:
            await self._api_request(
                "POST",
                f"/services/search/jobs/{job_id}/control",
                action="cancel",
            )
        except Exception as e:
            logger.warning(f"Failed to cancel job {job_id}: {e}")

    def _build_api_url(self, endpoint: str) -> str:
        """Build API URL."""
        scheme = "https" if self.config.use_ssl else "http"
        # API typically runs on port 8089
        port = self.config.port if self.config.port != 8088 else 8089
        return f"{scheme}://{self.config.host}:{port}{endpoint}"

    def _build_hec_url(self, endpoint: str) -> str:
        """Build HEC URL."""
        scheme = "https" if self.config.use_ssl else "http"
        return f"{scheme}://{self.config.host}:{self.config.port}{endpoint}"

    def _build_spl_query(self, query: SIEMQuery) -> str:
        """Build SPL query string."""
        spl = query.query

        # Add filters
        filters = []

        if query.severity:
            severity_values = [s.value for s in query.severity]
            filters.append(f"severity IN ({','.join(severity_values)})")

        if query.categories:
            filters.append(f"category IN ({','.join(query.categories)})")

        if query.sources:
            filters.append(f"source IN ({','.join(query.sources)})")

        if filters:
            spl = f"{spl} | where {' AND '.join(filters)}"

        # Add sorting
        if query.sort_by:
            spl = f"{spl} | sort {'-' if query.sort_order == 'desc' else ''}{query.sort_by}"

        return spl

    def _build_hec_payload(self, events: List[SIEMEvent]) -> str:
        """Build HEC JSON payload."""
        payload_lines = []

        for event in events:
            hec_event = {
                "time": event.timestamp.timestamp(),
                "host": event.host or self.config.host,
                "source": event.source or self.config.source,
                "sourcetype": event.source_type or self.config.sourcetype,
                "index": self.config.index,
                "event": {
                    "event_type": event.event_type,
                    "message": event.message,
                    "severity": event.severity.value,
                    "category": event.category,
                    **event.data,
                },
            }

            if event.labels:
                hec_event["fields"] = event.labels

            payload_lines.append(json.dumps(hec_event))

        return "\n".join(payload_lines)

    def _build_raw_payload(self, events: List[SIEMEvent]) -> str:
        """Build raw HEC payload."""
        return "\n".join(event.to_cef() for event in events)

    def _parse_event(self, result: Dict[str, Any]) -> Optional[SIEMEvent]:
        """Parse Splunk result into SIEMEvent."""
        try:
            timestamp_str = result.get("_time", "")
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")) if timestamp_str else datetime.utcnow()

            return SIEMEvent(
                event_type=result.get("event_type", "unknown"),
                message=result.get("message", result.get("_raw", "")),
                timestamp=timestamp,
                severity=self._parse_severity(result.get("severity")),
                category=result.get("category", ""),
                source=result.get("source", ""),
                source_type=result.get("sourcetype", ""),
                host=result.get("host", ""),
                data=result,
                raw=result.get("_raw"),
            )
        except Exception as e:
            logger.warning(f"Failed to parse event: {e}")
            return None

    def _parse_alert(self, entry: Dict[str, Any]) -> Optional[SIEMAlert]:
        """Parse Splunk alert entry."""
        try:
            content = entry.get("content", {})

            return SIEMAlert(
                alert_id=entry.get("name", str(uuid.uuid4())),
                title=content.get("savedsearch_name", ""),
                description=content.get("reason", ""),
                severity=self._urgency_to_severity(content.get("urgency", "medium")),
                status="new",
                created_at=datetime.utcnow(),
                rule_name=content.get("savedsearch_name"),
                raw=content,
            )
        except Exception as e:
            logger.warning(f"Failed to parse alert: {e}")
            return None

    def _parse_severity(self, severity: Optional[str]) -> EventSeverity:
        """Parse severity string to enum."""
        severity_map = {
            "critical": EventSeverity.CRITICAL,
            "high": EventSeverity.HIGH,
            "medium": EventSeverity.MEDIUM,
            "low": EventSeverity.LOW,
            "info": EventSeverity.INFO,
            "informational": EventSeverity.INFO,
            "debug": EventSeverity.DEBUG,
        }
        return severity_map.get((severity or "").lower(), EventSeverity.INFO)

    def _severity_to_urgency(self, severity: EventSeverity) -> str:
        """Convert severity to Splunk urgency."""
        urgency_map = {
            EventSeverity.CRITICAL: "critical",
            EventSeverity.HIGH: "high",
            EventSeverity.MEDIUM: "medium",
            EventSeverity.LOW: "low",
            EventSeverity.INFO: "informational",
            EventSeverity.DEBUG: "informational",
        }
        return urgency_map.get(severity, "medium")

    def _urgency_to_severity(self, urgency: str) -> EventSeverity:
        """Convert Splunk urgency to severity."""
        severity_map = {
            "critical": EventSeverity.CRITICAL,
            "high": EventSeverity.HIGH,
            "medium": EventSeverity.MEDIUM,
            "low": EventSeverity.LOW,
            "informational": EventSeverity.INFO,
        }
        return severity_map.get(urgency.lower(), EventSeverity.MEDIUM)
