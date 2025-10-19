"""
CRONOS AI - STIX/TAXII Client

Implements STIX 2.1 and TAXII 2.1 protocol support for threat intelligence
feed ingestion and sharing.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import base64

import aiohttp
from prometheus_client import Counter, Histogram, Gauge

from ..core.config import Config
from ..core.exceptions import CronosAIException


# Prometheus metrics
STIX_OBJECTS_INGESTED = Counter(
    "cronos_stix_objects_ingested_total",
    "Total STIX objects ingested",
    ["object_type", "source"],
    registry=None,
)

TAXII_REQUESTS = Counter(
    "cronos_taxii_requests_total",
    "Total TAXII requests",
    ["server", "endpoint", "status"],
    registry=None,
)

IOC_FEED_UPDATES = Counter(
    "cronos_ioc_feed_updates_total",
    "IOC feed update events",
    ["feed_name", "status"],
    registry=None,
)

THREAT_INTEL_AGE = Gauge(
    "cronos_threat_intel_age_hours",
    "Age of threat intelligence data in hours",
    ["source"],
    registry=None,
)


logger = logging.getLogger(__name__)


class STIXObjectType(str, Enum):
    """STIX 2.1 object types."""

    INDICATOR = "indicator"
    MALWARE = "malware"
    THREAT_ACTOR = "threat-actor"
    ATTACK_PATTERN = "attack-pattern"
    CAMPAIGN = "campaign"
    COURSE_OF_ACTION = "course-of-action"
    IDENTITY = "identity"
    INFRASTRUCTURE = "infrastructure"
    INTRUSION_SET = "intrusion-set"
    LOCATION = "location"
    OBSERVED_DATA = "observed-data"
    TOOL = "tool"
    VULNERABILITY = "vulnerability"
    RELATIONSHIP = "relationship"


class IndicatorType(str, Enum):
    """Types of indicators."""

    IPV4_ADDR = "ipv4-addr"
    IPV6_ADDR = "ipv6-addr"
    DOMAIN = "domain-name"
    URL = "url"
    EMAIL = "email-addr"
    FILE_HASH = "file"
    MUTEX = "mutex"
    REGISTRY_KEY = "windows-registry-key"


@dataclass
class STIXIndicator:
    """STIX Indicator Object."""

    id: str
    type: str = "indicator"
    spec_version: str = "2.1"
    created: datetime = field(default_factory=datetime.utcnow)
    modified: datetime = field(default_factory=datetime.utcnow)
    name: str = ""
    description: str = ""
    pattern: str = ""  # STIX pattern
    pattern_type: str = "stix"
    valid_from: datetime = field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None
    kill_chain_phases: List[Dict[str, str]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    confidence: int = 50  # 0-100
    external_references: List[Dict[str, str]] = field(default_factory=list)
    object_marking_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to STIX JSON format."""
        data = {
            "id": self.id,
            "type": self.type,
            "spec_version": self.spec_version,
            "created": self.created.isoformat() + "Z",
            "modified": self.modified.isoformat() + "Z",
            "name": self.name,
            "description": self.description,
            "pattern": self.pattern,
            "pattern_type": self.pattern_type,
            "valid_from": self.valid_from.isoformat() + "Z",
            "labels": self.labels,
            "confidence": self.confidence,
        }

        if self.valid_until:
            data["valid_until"] = self.valid_until.isoformat() + "Z"

        if self.kill_chain_phases:
            data["kill_chain_phases"] = self.kill_chain_phases

        if self.external_references:
            data["external_references"] = self.external_references

        if self.object_marking_refs:
            data["object_marking_refs"] = self.object_marking_refs

        return data


@dataclass
class TAXIIServer:
    """TAXII server configuration."""

    name: str
    url: str
    api_root: str
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    verify_ssl: bool = True
    timeout: int = 30
    collections: List[str] = field(default_factory=list)


@dataclass
class IOCFeed:
    """Indicator of Compromise feed."""

    feed_id: str
    name: str
    source: str
    feed_type: str  # stix, taxii, csv, json
    url: str
    update_interval_hours: int
    last_update: Optional[datetime] = None
    indicators_count: int = 0
    enabled: bool = True
    credentials: Optional[Dict[str, str]] = None


class STIXTAXIIClient:
    """
    STIX/TAXII client for threat intelligence feed ingestion.

    Supports STIX 2.1 and TAXII 2.1 protocols for consuming threat intelligence
    from various sources.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None

        # TAXII servers
        self.taxii_servers: Dict[str, TAXIIServer] = {}

        # IOC feeds
        self.ioc_feeds: Dict[str, IOCFeed] = {}

        # Ingested indicators cache
        self.indicators_cache: Dict[str, STIXIndicator] = {}
        self.cache_max_age_hours = 24

        # Background tasks
        self._update_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize STIX/TAXII client."""
        self.logger.info("Initializing STIX/TAXII client")

        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)

        # Load default TAXII servers
        await self._load_default_servers()

        # Load default IOC feeds
        await self._load_default_feeds()

        # Start background feed updates
        self._start_background_updates()

        self.logger.info(
            f"STIX/TAXII client initialized: "
            f"{len(self.taxii_servers)} servers, "
            f"{len(self.ioc_feeds)} feeds"
        )

    async def shutdown(self):
        """Shutdown STIX/TAXII client."""
        self.logger.info("Shutting down STIX/TAXII client")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        for task in self._update_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._update_tasks:
            await asyncio.gather(*self._update_tasks, return_exceptions=True)

        # Close HTTP session
        if self.session:
            await self.session.close()

        self.logger.info("STIX/TAXII client shutdown complete")

    async def _load_default_servers(self):
        """Load default TAXII servers."""
        # Example TAXII 2.1 servers (public threat intel sources)
        default_servers = [
            TAXIIServer(
                name="Anomali Limo",
                url="https://limo.anomali.com/api/v1/taxii2",
                api_root="/api/v1/taxii2/feeds/",
                collections=["107"],  # Emerging Threats
                verify_ssl=True,
            ),
            TAXIIServer(
                name="MISP Default",
                url="https://misp.local/servers",
                api_root="/servers/taxii",
                collections=[],
                verify_ssl=False,  # Often self-signed in local deployments
            ),
        ]

        for server in default_servers:
            self.taxii_servers[server.name] = server

    async def _load_default_feeds(self):
        """Load default IOC feeds."""
        default_feeds = [
            IOCFeed(
                feed_id="abuse_ch_ssl",
                name="Abuse.ch SSL Blacklist",
                source="abuse.ch",
                feed_type="csv",
                url="https://sslbl.abuse.ch/blacklist/sslipblacklist.csv",
                update_interval_hours=6,
            ),
            IOCFeed(
                feed_id="emergingthreats_compromised",
                name="Emerging Threats Compromised IPs",
                source="proofpoint",
                feed_type="csv",
                url="https://rules.emergingthreats.net/blockrules/compromised-ips.txt",
                update_interval_hours=1,
            ),
            IOCFeed(
                feed_id="alienvault_reputation",
                name="AlienVault IP Reputation",
                source="alienvault",
                feed_type="csv",
                url="https://reputation.alienvault.com/reputation.data",
                update_interval_hours=24,
            ),
        ]

        for feed in default_feeds:
            self.ioc_feeds[feed.feed_id] = feed

    def _start_background_updates(self):
        """Start background feed update tasks."""
        # Create update task for each enabled feed
        for feed_id, feed in self.ioc_feeds.items():
            if feed.enabled:
                task = asyncio.create_task(self._feed_update_loop(feed))
                self._update_tasks.append(task)

    async def _feed_update_loop(self, feed: IOCFeed):
        """Background loop to update a feed periodically."""
        while not self._shutdown_event.is_set():
            try:
                # Check if update is needed
                if self._should_update_feed(feed):
                    await self.update_ioc_feed(feed.feed_id)

                # Sleep until next check (every 5 minutes)
                await asyncio.sleep(300)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in feed update loop for {feed.name}: {e}")
                await asyncio.sleep(60)  # Wait before retry

    def _should_update_feed(self, feed: IOCFeed) -> bool:
        """Check if feed should be updated."""
        if not feed.last_update:
            return True

        elapsed = datetime.utcnow() - feed.last_update
        return elapsed.total_seconds() > (feed.update_interval_hours * 3600)

    async def update_ioc_feed(self, feed_id: str) -> int:
        """
        Update IOC feed from source.

        Returns:
            Number of indicators ingested
        """
        if feed_id not in self.ioc_feeds:
            raise CronosAIException(f"Unknown feed: {feed_id}")

        feed = self.ioc_feeds[feed_id]

        self.logger.info(f"Updating IOC feed: {feed.name}")
        start_time = time.time()

        try:
            if feed.feed_type == "stix":
                indicators = await self._fetch_stix_feed(feed)
            elif feed.feed_type == "taxii":
                indicators = await self._fetch_taxii_feed(feed)
            elif feed.feed_type == "csv":
                indicators = await self._fetch_csv_feed(feed)
            elif feed.feed_type == "json":
                indicators = await self._fetch_json_feed(feed)
            else:
                raise CronosAIException(f"Unsupported feed type: {feed.feed_type}")

            # Store indicators
            count = await self._store_indicators(indicators, feed.source)

            # Update feed metadata
            feed.last_update = datetime.utcnow()
            feed.indicators_count = count

            # Metrics
            IOC_FEED_UPDATES.labels(feed_name=feed.name, status="success").inc()
            THREAT_INTEL_AGE.labels(source=feed.source).set(0)

            self.logger.info(
                f"Feed {feed.name} updated: {count} indicators "
                f"in {time.time() - start_time:.2f}s"
            )

            return count

        except Exception as e:
            IOC_FEED_UPDATES.labels(feed_name=feed.name, status="error").inc()
            self.logger.error(f"Failed to update feed {feed.name}: {e}")
            raise CronosAIException(f"Feed update failed: {e}")

    async def _fetch_stix_feed(self, feed: IOCFeed) -> List[STIXIndicator]:
        """Fetch STIX feed."""
        headers = {"Accept": "application/stix+json"}

        if feed.credentials and feed.credentials.get("api_key"):
            headers["Authorization"] = f"Bearer {feed.credentials['api_key']}"

        async with self.session.get(feed.url, headers=headers) as response:
            if response.status != 200:
                raise CronosAIException(
                    f"Failed to fetch STIX feed: HTTP {response.status}"
                )

            data = await response.json()

            # Parse STIX bundle
            indicators = []
            objects = data.get("objects", [])

            for obj in objects:
                if obj.get("type") == "indicator":
                    indicator = self._parse_stix_indicator(obj)
                    indicators.append(indicator)

            return indicators

    async def _fetch_taxii_feed(self, feed: IOCFeed) -> List[STIXIndicator]:
        """Fetch TAXII 2.1 feed."""
        # Get TAXII server for this feed
        server_name = feed.credentials.get("server") if feed.credentials else None
        if not server_name or server_name not in self.taxii_servers:
            raise CronosAIException(f"TAXII server not configured for feed {feed.name}")

        server = self.taxii_servers[server_name]

        # Build TAXII API URL
        collection_id = feed.credentials.get("collection_id", "default")
        url = f"{server.url}{server.api_root}/collections/{collection_id}/objects/"

        # Authentication
        headers = {"Accept": "application/taxii+json;version=2.1"}
        auth = None

        if server.api_key:
            headers["Authorization"] = f"Bearer {server.api_key}"
        elif server.username and server.password:
            auth = aiohttp.BasicAuth(server.username, server.password)

        # Fetch objects
        async with self.session.get(
            url, headers=headers, auth=auth, ssl=server.verify_ssl
        ) as response:
            TAXII_REQUESTS.labels(
                server=server.name,
                endpoint="objects",
                status=response.status,
            ).inc()

            if response.status != 200:
                raise CronosAIException(
                    f"TAXII request failed: HTTP {response.status}"
                )

            data = await response.json()

            # Parse indicators
            indicators = []
            objects = data.get("objects", [])

            for obj in objects:
                if obj.get("type") == "indicator":
                    indicator = self._parse_stix_indicator(obj)
                    indicators.append(indicator)

            return indicators

    async def _fetch_csv_feed(self, feed: IOCFeed) -> List[STIXIndicator]:
        """Fetch CSV-based IOC feed."""
        async with self.session.get(feed.url) as response:
            if response.status != 200:
                raise CronosAIException(f"Failed to fetch CSV feed: HTTP {response.status}")

            text = await response.text()

            # Parse CSV (simple implementation)
            indicators = []
            lines = text.strip().split("\n")

            for line in lines:
                # Skip comments and headers
                if line.startswith("#") or line.startswith("//"):
                    continue

                # Parse IP addresses (simple heuristic)
                parts = line.strip().split(",")
                if not parts:
                    continue

                value = parts[0].strip()

                # Create indicator
                indicator_id = f"indicator--{hashlib.sha256(value.encode()).hexdigest()}"

                # Determine type
                if "." in value and all(p.isdigit() for p in value.split(".")):
                    # Likely IPv4
                    pattern = f"[ipv4-addr:value = '{value}']"
                    ioc_type = "ipv4-addr"
                else:
                    # Generic
                    pattern = f"[x-custom:value = '{value}']"
                    ioc_type = "unknown"

                indicator = STIXIndicator(
                    id=indicator_id,
                    name=f"{ioc_type.upper()}: {value}",
                    description=f"IOC from {feed.name}",
                    pattern=pattern,
                    pattern_type="stix",
                    labels=["malicious-activity", ioc_type],
                    confidence=70,
                )

                indicators.append(indicator)

            return indicators

    async def _fetch_json_feed(self, feed: IOCFeed) -> List[STIXIndicator]:
        """Fetch JSON-based IOC feed."""
        async with self.session.get(feed.url) as response:
            if response.status != 200:
                raise CronosAIException(f"Failed to fetch JSON feed: HTTP {response.status}")

            data = await response.json()

            # Parse JSON (flexible format support)
            indicators = []

            # Handle different JSON structures
            if isinstance(data, list):
                items = data
            elif "indicators" in data:
                items = data["indicators"]
            elif "iocs" in data:
                items = data["iocs"]
            else:
                items = [data]

            for item in items:
                indicator = self._parse_json_indicator(item, feed)
                if indicator:
                    indicators.append(indicator)

            return indicators

    def _parse_stix_indicator(self, stix_obj: Dict[str, Any]) -> STIXIndicator:
        """Parse STIX indicator object."""
        indicator_id = stix_obj.get("id", f"indicator--{hashlib.sha256(str(stix_obj).encode()).hexdigest()}")

        created = datetime.fromisoformat(
            stix_obj.get("created", datetime.utcnow().isoformat()).rstrip("Z")
        )
        modified = datetime.fromisoformat(
            stix_obj.get("modified", datetime.utcnow().isoformat()).rstrip("Z")
        )
        valid_from = datetime.fromisoformat(
            stix_obj.get("valid_from", datetime.utcnow().isoformat()).rstrip("Z")
        )

        valid_until = None
        if "valid_until" in stix_obj:
            valid_until = datetime.fromisoformat(
                stix_obj["valid_until"].rstrip("Z")
            )

        indicator = STIXIndicator(
            id=indicator_id,
            created=created,
            modified=modified,
            name=stix_obj.get("name", ""),
            description=stix_obj.get("description", ""),
            pattern=stix_obj.get("pattern", ""),
            pattern_type=stix_obj.get("pattern_type", "stix"),
            valid_from=valid_from,
            valid_until=valid_until,
            kill_chain_phases=stix_obj.get("kill_chain_phases", []),
            labels=stix_obj.get("labels", []),
            confidence=stix_obj.get("confidence", 50),
            external_references=stix_obj.get("external_references", []),
            object_marking_refs=stix_obj.get("object_marking_refs", []),
        )

        return indicator

    def _parse_json_indicator(
        self, json_obj: Dict[str, Any], feed: IOCFeed
    ) -> Optional[STIXIndicator]:
        """Parse generic JSON indicator."""
        # Extract value (flexible field names)
        value = (
            json_obj.get("value")
            or json_obj.get("indicator")
            or json_obj.get("ioc")
            or json_obj.get("ip")
            or json_obj.get("domain")
        )

        if not value:
            return None

        indicator_id = f"indicator--{hashlib.sha256(value.encode()).hexdigest()}"

        # Determine type
        ioc_type = json_obj.get("type", "unknown")

        # Create STIX pattern
        pattern = f"[{ioc_type}:value = '{value}']"

        indicator = STIXIndicator(
            id=indicator_id,
            name=f"{ioc_type.upper()}: {value}",
            description=json_obj.get("description", f"IOC from {feed.name}"),
            pattern=pattern,
            labels=json_obj.get("labels", ["malicious-activity"]),
            confidence=json_obj.get("confidence", 60),
        )

        return indicator

    async def _store_indicators(
        self, indicators: List[STIXIndicator], source: str
    ) -> int:
        """Store indicators in cache."""
        count = 0

        for indicator in indicators:
            # Store in cache
            self.indicators_cache[indicator.id] = indicator
            count += 1

            # Metrics
            STIX_OBJECTS_INGESTED.labels(
                object_type="indicator",
                source=source,
            ).inc()

        # Clean old indicators from cache
        await self._clean_cache()

        return count

    async def _clean_cache(self):
        """Clean old indicators from cache."""
        cutoff = datetime.utcnow() - timedelta(hours=self.cache_max_age_hours)

        to_remove = []
        for indicator_id, indicator in self.indicators_cache.items():
            if indicator.modified < cutoff:
                to_remove.append(indicator_id)

        for indicator_id in to_remove:
            del self.indicators_cache[indicator_id]

        if to_remove:
            self.logger.debug(f"Cleaned {len(to_remove)} old indicators from cache")

    async def query_indicators(
        self,
        indicator_type: Optional[str] = None,
        labels: Optional[List[str]] = None,
        min_confidence: int = 0,
        limit: int = 100,
    ) -> List[STIXIndicator]:
        """
        Query indicators from cache.

        Args:
            indicator_type: Filter by type (ipv4-addr, domain-name, etc.)
            labels: Filter by labels
            min_confidence: Minimum confidence score
            limit: Maximum number of results

        Returns:
            List of matching indicators
        """
        results = []

        for indicator in self.indicators_cache.values():
            # Filter by type
            if indicator_type and indicator_type not in indicator.labels:
                continue

            # Filter by labels
            if labels and not any(label in indicator.labels for label in labels):
                continue

            # Filter by confidence
            if indicator.confidence < min_confidence:
                continue

            results.append(indicator)

            if len(results) >= limit:
                break

        return results

    async def check_ioc(self, value: str, ioc_type: Optional[str] = None) -> Optional[STIXIndicator]:
        """
        Check if a value matches any known IOC.

        Args:
            value: Value to check (IP, domain, hash, etc.)
            ioc_type: Optional type hint

        Returns:
            Matching indicator if found, None otherwise
        """
        # Simple pattern matching (can be enhanced with regex)
        for indicator in self.indicators_cache.values():
            if value in indicator.pattern:
                return indicator

        return None


# Factory function
def get_stix_taxii_client(config: Config) -> STIXTAXIIClient:
    """Factory function to get STIXTAXIIClient instance."""
    return STIXTAXIIClient(config)
