"""
CRONOS AI - Threat Intelligence Platform Manager

Central manager integrating all TIP components: STIX/TAXII, MITRE ATT&CK,
and threat hunting.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from prometheus_client import Gauge

from ..core.config import Config
from ..core.exceptions import CronosAIException
from ..security.models import SecurityEvent
from .stix_taxii_client import STIXTAXIIClient, STIXIndicator, get_stix_taxii_client
from .mitre_attack_mapper import MITREATTACKMapper, TTPMapping, get_mitre_attack_mapper
from .threat_hunter import ThreatHunter, HuntCampaign, get_threat_hunter


# Prometheus metrics
TIP_HEALTH_STATUS = Gauge(
    "cronos_tip_health_status",
    "Threat Intelligence Platform health status",
    ["component"],
    registry=None,
)


logger = logging.getLogger(__name__)


class ThreatIntelligenceManager:
    """
    Threat Intelligence Platform Manager.

    Coordinates STIX/TAXII client, MITRE ATT&CK mapper, and threat hunter
    to provide comprehensive threat intelligence capabilities.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # TIP components
        self.stix_client: Optional[STIXTAXIIClient] = None
        self.attack_mapper: Optional[MITREATTACKMapper] = None
        self.threat_hunter: Optional[ThreatHunter] = None

        # Initialization status
        self._initialized = False

    async def initialize(self):
        """Initialize TIP manager and all components."""
        if self._initialized:
            self.logger.warning("TIP manager already initialized")
            return

        self.logger.info("Initializing Threat Intelligence Platform")

        try:
            # Initialize STIX/TAXII client
            self.stix_client = get_stix_taxii_client(self.config)
            await self.stix_client.initialize()
            TIP_HEALTH_STATUS.labels(component="stix_taxii").set(1)

            # Initialize MITRE ATT&CK mapper
            self.attack_mapper = get_mitre_attack_mapper(self.config)
            await self.attack_mapper.initialize()
            TIP_HEALTH_STATUS.labels(component="mitre_attack").set(1)

            # Initialize threat hunter
            self.threat_hunter = get_threat_hunter(
                self.config,
                stix_client=self.stix_client,
                attack_mapper=self.attack_mapper,
            )
            await self.threat_hunter.initialize()
            TIP_HEALTH_STATUS.labels(component="threat_hunter").set(1)

            self._initialized = True

            self.logger.info("Threat Intelligence Platform initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize TIP: {e}", exc_info=True)
            TIP_HEALTH_STATUS.labels(component="overall").set(0)
            raise CronosAIException(f"TIP initialization failed: {e}")

    async def shutdown(self):
        """Shutdown TIP manager and all components."""
        self.logger.info("Shutting down Threat Intelligence Platform")

        # Shutdown components in reverse order
        if self.threat_hunter:
            # Threat hunter doesn't need shutdown currently
            pass

        if self.attack_mapper:
            await self.attack_mapper.shutdown()

        if self.stix_client:
            await self.stix_client.shutdown()

        self._initialized = False

        self.logger.info("Threat Intelligence Platform shutdown complete")

    async def process_security_event(self, event: SecurityEvent) -> Dict[str, Any]:
        """
        Process security event through TIP pipeline.

        Args:
            event: Security event to process

        Returns:
            TIP enrichment data including IOC matches, TTP mappings, etc.
        """
        if not self._initialized:
            raise CronosAIException("TIP not initialized")

        enrichment = {
            "event_id": event.event_id,
            "timestamp": datetime.utcnow().isoformat(),
            "ioc_matches": [],
            "ttp_mapping": None,
            "threat_score": 0.0,
            "recommendations": [],
        }

        # Check against IOCs
        if self.stix_client:
            ioc_match = await self._check_iocs(event)
            if ioc_match:
                enrichment["ioc_matches"].append(ioc_match)
                enrichment["threat_score"] += 0.3

        # Map to MITRE ATT&CK
        if self.attack_mapper:
            ttp_mapping = await self.attack_mapper.map_event_to_techniques(event)
            enrichment["ttp_mapping"] = ttp_mapping.to_dict()

            if ttp_mapping.matched_techniques:
                enrichment["threat_score"] += 0.2 * len(ttp_mapping.matched_techniques)

                # Add TTP-specific recommendations
                for technique in ttp_mapping.matched_techniques:
                    enrichment["recommendations"].extend(technique.mitigations[:2])

        # Add to threat hunting history
        if self.threat_hunter:
            self.threat_hunter.add_event_to_history(event)

        # Normalize threat score (0-1)
        enrichment["threat_score"] = min(enrichment["threat_score"], 1.0)

        return enrichment

    async def _check_iocs(self, event: SecurityEvent) -> Optional[Dict[str, Any]]:
        """Check event against known IOCs."""
        # Check source IP
        source_ip = getattr(event, "source_ip", None)
        if source_ip and source_ip != "unknown":
            indicator = await self.stix_client.check_ioc(source_ip, "ipv4-addr")
            if indicator:
                return {
                    "type": "source_ip",
                    "value": source_ip,
                    "indicator_id": indicator.id,
                    "indicator_name": indicator.name,
                    "confidence": indicator.confidence,
                }

        # Check destination IP
        dest_ip = getattr(event, "destination_ip", None)
        if dest_ip and dest_ip != "unknown":
            indicator = await self.stix_client.check_ioc(dest_ip, "ipv4-addr")
            if indicator:
                return {
                    "type": "destination_ip",
                    "value": dest_ip,
                    "indicator_id": indicator.id,
                    "indicator_name": indicator.name,
                    "confidence": indicator.confidence,
                }

        return None

    async def execute_threat_hunt(
        self,
        hypotheses: Optional[List[str]] = None,
        time_range_hours: int = 24,
    ) -> HuntCampaign:
        """
        Execute threat hunting campaign.

        Args:
            hypotheses: Specific hypotheses to hunt (None = all)
            time_range_hours: Time range to hunt

        Returns:
            Hunt campaign with findings
        """
        if not self._initialized:
            raise CronosAIException("TIP not initialized")

        if not self.threat_hunter:
            raise CronosAIException("Threat hunter not available")

        campaign = await self.threat_hunter.execute_hunt_campaign(
            hypotheses=hypotheses,
            time_range_hours=time_range_hours,
        )

        return campaign

    async def query_threat_intelligence(
        self,
        query_type: str,
        query_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Query threat intelligence.

        Args:
            query_type: Type of query (iocs, techniques, hunts)
            query_params: Query parameters

        Returns:
            Query results
        """
        if not self._initialized:
            raise CronosAIException("TIP not initialized")

        if query_type == "iocs":
            return await self._query_iocs(query_params)
        elif query_type == "techniques":
            return await self._query_techniques(query_params)
        elif query_type == "hunts":
            return await self._query_hunts(query_params)
        else:
            raise CronosAIException(f"Unknown query type: {query_type}")

    async def _query_iocs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query IOCs from STIX/TAXII client."""
        if not self.stix_client:
            return {"indicators": [], "total": 0}

        indicator_type = params.get("indicator_type")
        labels = params.get("labels")
        min_confidence = params.get("min_confidence", 0)
        limit = params.get("limit", 100)

        indicators = await self.stix_client.query_indicators(
            indicator_type=indicator_type,
            labels=labels,
            min_confidence=min_confidence,
            limit=limit,
        )

        return {
            "indicators": [ind.to_dict() for ind in indicators],
            "total": len(indicators),
        }

    async def _query_techniques(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query MITRE ATT&CK techniques."""
        if not self.attack_mapper:
            return {"techniques": [], "total": 0}

        query = params.get("query", "")
        limit = params.get("limit", 10)

        if query:
            techniques = await self.attack_mapper.search_techniques(query, limit)
        else:
            # Return all techniques (limited)
            techniques = list(self.attack_mapper.techniques.values())[:limit]

        return {
            "techniques": [tech.to_dict() for tech in techniques],
            "total": len(techniques),
        }

    async def _query_hunts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query threat hunt campaigns."""
        if not self.threat_hunter:
            return {"campaigns": [], "total": 0}

        # Return active campaigns
        campaigns = list(self.threat_hunter.active_campaigns.values())

        return {
            "campaigns": [camp.to_dict() for camp in campaigns],
            "total": len(campaigns),
        }

    async def update_ioc_feeds(
        self, feed_ids: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Update IOC feeds.

        Args:
            feed_ids: Specific feed IDs to update (None = all)

        Returns:
            Update statistics per feed
        """
        if not self._initialized or not self.stix_client:
            raise CronosAIException("TIP not initialized")

        results = {}

        if feed_ids is None:
            feed_ids = list(self.stix_client.ioc_feeds.keys())

        for feed_id in feed_ids:
            try:
                count = await self.stix_client.update_ioc_feed(feed_id)
                results[feed_id] = count
            except Exception as e:
                self.logger.error(f"Failed to update feed {feed_id}: {e}")
                results[feed_id] = 0

        return results

    def get_health_status(self) -> Dict[str, Any]:
        """Get TIP health status."""
        return {
            "initialized": self._initialized,
            "components": {
                "stix_taxii": self.stix_client is not None,
                "mitre_attack": self.attack_mapper is not None,
                "threat_hunter": self.threat_hunter is not None,
            },
            "ioc_feeds": len(self.stix_client.ioc_feeds) if self.stix_client else 0,
            "indicators_cached": (
                len(self.stix_client.indicators_cache) if self.stix_client else 0
            ),
            "attack_techniques": (
                len(self.attack_mapper.techniques) if self.attack_mapper else 0
            ),
            "hunt_hypotheses": (
                len(self.threat_hunter.hypotheses) if self.threat_hunter else 0
            ),
        }

    def get_coverage_report(self) -> Dict[str, Any]:
        """Get comprehensive coverage report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "mitre_attack_coverage": None,
            "ioc_feed_status": [],
        }

        # MITRE ATT&CK coverage
        if self.attack_mapper:
            report["mitre_attack_coverage"] = (
                self.attack_mapper.get_detection_coverage_report()
            )

        # IOC feed status
        if self.stix_client:
            for feed_id, feed in self.stix_client.ioc_feeds.items():
                report["ioc_feed_status"].append(
                    {
                        "feed_id": feed_id,
                        "name": feed.name,
                        "enabled": feed.enabled,
                        "last_update": (
                            feed.last_update.isoformat() if feed.last_update else None
                        ),
                        "indicators_count": feed.indicators_count,
                    }
                )

        return report


# Global instance
_tip_manager_instance: Optional[ThreatIntelligenceManager] = None


def get_threat_intelligence_manager(config: Config) -> ThreatIntelligenceManager:
    """Get or create TIP manager instance."""
    global _tip_manager_instance

    if _tip_manager_instance is None:
        _tip_manager_instance = ThreatIntelligenceManager(config)

    return _tip_manager_instance


async def initialize_tip_manager(config: Config) -> ThreatIntelligenceManager:
    """Initialize TIP manager."""
    manager = get_threat_intelligence_manager(config)
    await manager.initialize()
    return manager


async def shutdown_tip_manager():
    """Shutdown TIP manager."""
    global _tip_manager_instance

    if _tip_manager_instance:
        await _tip_manager_instance.shutdown()
        _tip_manager_instance = None
