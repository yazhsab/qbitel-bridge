"""
CRONOS AI - Automated Threat Hunter

Proactive threat hunting using IOCs, behavioral analytics, and MITRE ATT&CK
techniques to identify hidden threats.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from prometheus_client import Counter, Histogram, Gauge

from ..core.config import Config
from ..core.exceptions import CronosAIException
from ..security.models import SecurityEvent, ThreatLevel
from .stix_taxii_client import STIXIndicator, STIXTAXIIClient
from .mitre_attack_mapper import MITREATTACKMapper, ATTACKTechnique


# Prometheus metrics
HUNT_CAMPAIGNS_EXECUTED = Counter(
    "cronos_threat_hunt_campaigns_total",
    "Total threat hunting campaigns executed",
    ["hunt_type"],
    registry=None,
)

HUNT_FINDINGS_DISCOVERED = Counter(
    "cronos_threat_hunt_findings_total",
    "Threat hunting findings discovered",
    ["severity", "finding_type"],
    registry=None,
)

HUNT_CAMPAIGN_DURATION = Histogram(
    "cronos_threat_hunt_campaign_duration_seconds",
    "Duration of hunting campaigns",
    registry=None,
)


logger = logging.getLogger(__name__)


class HuntType(str, Enum):
    """Types of threat hunts."""

    IOC_BASED = "ioc_based"  # Hunt using known IOCs
    BEHAVIOR_BASED = "behavior_based"  # Hunt using behavioral patterns
    TTP_BASED = "ttp_based"  # Hunt using MITRE ATT&CK TTPs
    ANOMALY_BASED = "anomaly_based"  # Hunt using anomaly detection
    HYBRID = "hybrid"  # Combination of multiple approaches


class FindingSeverity(str, Enum):
    """Severity of hunting findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class HuntHypothesis:
    """Threat hunting hypothesis."""

    hypothesis_id: str
    name: str
    description: str
    hunt_type: HuntType
    iocs: List[str] = field(default_factory=list)
    ttps: List[str] = field(default_factory=list)  # MITRE ATT&CK technique IDs
    behavioral_patterns: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    expected_findings: str = ""
    priority: int = 5  # 1-10, higher is more important


@dataclass
class HuntFinding:
    """Threat hunting finding."""

    finding_id: str
    hypothesis_id: str
    severity: FindingSeverity
    title: str
    description: str
    evidence: List[Dict[str, Any]]
    iocs_matched: List[str]
    ttps_matched: List[str]
    affected_assets: List[str]
    timeline: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_score: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "finding_id": self.finding_id,
            "hypothesis_id": self.hypothesis_id,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "iocs_matched": self.iocs_matched,
            "ttps_matched": self.ttps_matched,
            "affected_assets": self.affected_assets,
            "timeline": self.timeline,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HuntCampaign:
    """Threat hunting campaign."""

    campaign_id: str
    name: str
    description: str
    hypotheses: List[HuntHypothesis]
    start_time: datetime
    end_time: Optional[datetime] = None
    findings: List[HuntFinding] = field(default_factory=list)
    status: str = "running"  # running, completed, failed
    coverage_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "description": self.description,
            "hypotheses_count": len(self.hypotheses),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "findings": [f.to_dict() for f in self.findings],
            "status": self.status,
            "coverage_metrics": self.coverage_metrics,
        }


class ThreatHunter:
    """
    Automated threat hunting engine.

    Proactively searches for threats using IOCs, behavioral patterns,
    and MITRE ATT&CK techniques.
    """

    def __init__(
        self,
        config: Config,
        stix_client: Optional[STIXTAXIIClient] = None,
        attack_mapper: Optional[MITREATTACKMapper] = None,
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.stix_client = stix_client
        self.attack_mapper = attack_mapper

        # Pre-defined hunt hypotheses
        self.hypotheses: Dict[str, HuntHypothesis] = {}

        # Active campaigns
        self.active_campaigns: Dict[str, HuntCampaign] = {}

        # Historical events for hunting
        self.event_history: List[SecurityEvent] = []
        self.max_history_hours = 72  # 3 days

    async def initialize(self):
        """Initialize threat hunter."""
        self.logger.info("Initializing threat hunter")

        # Load pre-defined hunt hypotheses
        self._load_default_hypotheses()

        self.logger.info(
            f"Threat hunter initialized: {len(self.hypotheses)} hypotheses loaded"
        )

    def _load_default_hypotheses(self):
        """Load default hunting hypotheses."""
        # IOC-based hunts
        self.hypotheses["h001"] = HuntHypothesis(
            hypothesis_id="h001",
            name="Known Malicious IPs",
            description="Hunt for communications with known malicious IP addresses",
            hunt_type=HuntType.IOC_BASED,
            data_sources=["network_traffic", "firewall_logs", "proxy_logs"],
            expected_findings="Outbound connections to known C2 servers",
            priority=9,
        )

        self.hypotheses["h002"] = HuntHypothesis(
            hypothesis_id="h002",
            name="Suspicious Domain Lookups",
            description="Hunt for DNS queries to newly registered or suspicious domains",
            hunt_type=HuntType.IOC_BASED,
            data_sources=["dns_logs"],
            expected_findings="DNS queries to suspicious domains",
            priority=7,
        )

        # TTP-based hunts
        self.hypotheses["h003"] = HuntHypothesis(
            hypothesis_id="h003",
            name="Lateral Movement Detection",
            description="Hunt for lateral movement using MITRE ATT&CK T1021 techniques",
            hunt_type=HuntType.TTP_BASED,
            ttps=["T1021"],  # Remote Services
            data_sources=["authentication_logs", "network_traffic"],
            expected_findings="Abnormal remote service usage indicating lateral movement",
            priority=8,
        )

        self.hypotheses["h004"] = HuntHypothesis(
            hypothesis_id="h004",
            name="Data Exfiltration Patterns",
            description="Hunt for data exfiltration using T1048 techniques",
            hunt_type=HuntType.TTP_BASED,
            ttps=["T1048", "T1041"],  # Exfiltration
            data_sources=["network_traffic", "file_access_logs"],
            expected_findings="Large data transfers to external destinations",
            priority=9,
        )

        # Behavior-based hunts
        self.hypotheses["h005"] = HuntHypothesis(
            hypothesis_id="h005",
            name="Abnormal Authentication Patterns",
            description="Hunt for unusual authentication behavior",
            hunt_type=HuntType.BEHAVIOR_BASED,
            behavioral_patterns=[
                "multiple_failed_logins",
                "unusual_login_times",
                "impossible_travel",
            ],
            data_sources=["authentication_logs"],
            expected_findings="Credential stuffing or account compromise indicators",
            priority=8,
        )

        self.hypotheses["h006"] = HuntHypothesis(
            hypothesis_id="h006",
            name="Privilege Escalation Attempts",
            description="Hunt for privilege escalation attempts",
            hunt_type=HuntType.TTP_BASED,
            ttps=["T1068", "T1078"],  # Exploitation and Valid Accounts
            data_sources=["system_logs", "authentication_logs"],
            expected_findings="Unauthorized privilege escalation",
            priority=9,
        )

    async def execute_hunt_campaign(
        self,
        hypotheses: Optional[List[str]] = None,
        time_range_hours: int = 24,
    ) -> HuntCampaign:
        """
        Execute threat hunting campaign.

        Args:
            hypotheses: List of hypothesis IDs to hunt (None = all)
            time_range_hours: Time range to hunt over

        Returns:
            Hunt campaign with findings
        """
        start_time = time.time()

        # Determine which hypotheses to hunt
        if hypotheses is None:
            hunt_hypotheses = list(self.hypotheses.values())
        else:
            hunt_hypotheses = [
                self.hypotheses[h_id]
                for h_id in hypotheses
                if h_id in self.hypotheses
            ]

        # Sort by priority
        hunt_hypotheses.sort(key=lambda h: h.priority, reverse=True)

        # Create campaign
        campaign_id = f"campaign_{int(time.time())}"
        campaign = HuntCampaign(
            campaign_id=campaign_id,
            name=f"Threat Hunt {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            description=f"Automated hunt across {len(hunt_hypotheses)} hypotheses",
            hypotheses=hunt_hypotheses,
            start_time=datetime.utcnow(),
            status="running",
        )

        self.active_campaigns[campaign_id] = campaign

        self.logger.info(
            f"Starting hunt campaign {campaign_id} with {len(hunt_hypotheses)} hypotheses"
        )

        try:
            # Execute each hypothesis
            for hypothesis in hunt_hypotheses:
                findings = await self._hunt_hypothesis(hypothesis, time_range_hours)
                campaign.findings.extend(findings)

                # Metrics
                for finding in findings:
                    HUNT_FINDINGS_DISCOVERED.labels(
                        severity=finding.severity.value,
                        finding_type=hypothesis.hunt_type.value,
                    ).inc()

            # Calculate coverage metrics
            campaign.coverage_metrics = self._calculate_coverage_metrics(campaign)

            # Mark complete
            campaign.end_time = datetime.utcnow()
            campaign.status = "completed"

            # Metrics
            HUNT_CAMPAIGNS_EXECUTED.labels(hunt_type="automated").inc()
            HUNT_CAMPAIGN_DURATION.observe(time.time() - start_time)

            self.logger.info(
                f"Hunt campaign {campaign_id} completed: "
                f"{len(campaign.findings)} findings discovered"
            )

            return campaign

        except Exception as e:
            campaign.status = "failed"
            campaign.end_time = datetime.utcnow()
            self.logger.error(f"Hunt campaign {campaign_id} failed: {e}")
            raise CronosAIException(f"Hunt campaign failed: {e}")

    async def _hunt_hypothesis(
        self, hypothesis: HuntHypothesis, time_range_hours: int
    ) -> List[HuntFinding]:
        """Hunt a specific hypothesis."""
        findings = []

        if hypothesis.hunt_type == HuntType.IOC_BASED:
            findings = await self._hunt_ioc_based(hypothesis, time_range_hours)
        elif hypothesis.hunt_type == HuntType.TTP_BASED:
            findings = await self._hunt_ttp_based(hypothesis, time_range_hours)
        elif hypothesis.hunt_type == HuntType.BEHAVIOR_BASED:
            findings = await self._hunt_behavior_based(hypothesis, time_range_hours)

        return findings

    async def _hunt_ioc_based(
        self, hypothesis: HuntHypothesis, time_range_hours: int
    ) -> List[HuntFinding]:
        """Hunt using known IOCs."""
        findings = []

        if not self.stix_client:
            return findings

        # Get recent IOCs from threat intel feeds
        indicators = await self.stix_client.query_indicators(
            min_confidence=60,
            limit=1000,
        )

        # Search for IOC matches in event history
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        relevant_events = [
            e for e in self.event_history if e.timestamp >= cutoff_time
        ]

        for event in relevant_events:
            matched_iocs = []

            # Check event against IOCs
            for indicator in indicators:
                if await self._event_matches_ioc(event, indicator):
                    matched_iocs.append(indicator.name)

            # Create finding if IOCs matched
            if matched_iocs:
                finding = self._create_ioc_finding(
                    hypothesis, event, matched_iocs
                )
                findings.append(finding)

        return findings

    async def _hunt_ttp_based(
        self, hypothesis: HuntHypothesis, time_range_hours: int
    ) -> List[HuntFinding]:
        """Hunt using MITRE ATT&CK TTPs."""
        findings = []

        if not self.attack_mapper:
            return findings

        # Get events in time range
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        relevant_events = [
            e for e in self.event_history if e.timestamp >= cutoff_time
        ]

        # Map events to techniques and check for matches
        for event in relevant_events:
            ttp_mapping = await self.attack_mapper.map_event_to_techniques(event)

            # Check if any matched techniques are in hypothesis TTPs
            matched_ttps = [
                t.technique_id
                for t in ttp_mapping.matched_techniques
                if t.technique_id in hypothesis.ttps
            ]

            if matched_ttps:
                finding = self._create_ttp_finding(
                    hypothesis, event, matched_ttps, ttp_mapping
                )
                findings.append(finding)

        return findings

    async def _hunt_behavior_based(
        self, hypothesis: HuntHypothesis, time_range_hours: int
    ) -> List[HuntFinding]:
        """Hunt using behavioral patterns."""
        findings = []

        # Analyze behavioral patterns
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        relevant_events = [
            e for e in self.event_history if e.timestamp >= cutoff_time
        ]

        # Group events by source
        events_by_source: Dict[str, List[SecurityEvent]] = {}
        for event in relevant_events:
            source = getattr(event, "source_ip", "unknown")
            if source not in events_by_source:
                events_by_source[source] = []
            events_by_source[source].append(event)

        # Detect behavioral patterns
        for source, events in events_by_source.items():
            # Check for multiple failed logins
            if "multiple_failed_logins" in hypothesis.behavioral_patterns:
                failed_logins = [
                    e
                    for e in events
                    if "failed" in e.description.lower()
                    and "login" in e.description.lower()
                ]

                if len(failed_logins) >= 5:  # Threshold
                    finding = self._create_behavior_finding(
                        hypothesis,
                        f"Multiple failed logins from {source}",
                        failed_logins,
                    )
                    findings.append(finding)

        return findings

    async def _event_matches_ioc(
        self, event: SecurityEvent, indicator: STIXIndicator
    ) -> bool:
        """Check if event matches an IOC."""
        # Simple pattern matching (can be enhanced)
        event_text = f"{event.description} {event.source_ip} {event.destination_ip}"

        # Extract values from STIX pattern (simplified)
        # Pattern format: [ipv4-addr:value = '1.2.3.4']
        if "[" in indicator.pattern and "=" in indicator.pattern:
            value_start = indicator.pattern.find("'")
            value_end = indicator.pattern.rfind("'")

            if value_start > 0 and value_end > value_start:
                ioc_value = indicator.pattern[value_start + 1 : value_end]
                return ioc_value in event_text

        return False

    def _create_ioc_finding(
        self,
        hypothesis: HuntHypothesis,
        event: SecurityEvent,
        matched_iocs: List[str],
    ) -> HuntFinding:
        """Create IOC-based finding."""
        finding_id = f"finding_{hypothesis.hypothesis_id}_{event.event_id}"

        severity = FindingSeverity.HIGH
        if event.threat_level == ThreatLevel.CRITICAL:
            severity = FindingSeverity.CRITICAL

        finding = HuntFinding(
            finding_id=finding_id,
            hypothesis_id=hypothesis.hypothesis_id,
            severity=severity,
            title=f"IOC Match: {hypothesis.name}",
            description=f"Event matched {len(matched_iocs)} known IOC(s): {', '.join(matched_iocs[:3])}",
            evidence=[
                {
                    "type": "security_event",
                    "event_id": event.event_id,
                    "description": event.description,
                    "timestamp": event.timestamp.isoformat(),
                }
            ],
            iocs_matched=matched_iocs,
            ttps_matched=[],
            affected_assets=[getattr(event, "source_ip", "unknown")],
            timeline=[
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event": "IOC match detected",
                }
            ],
            recommendations=[
                "Investigate affected assets for compromise",
                "Block communication with matched IOCs",
                "Review related events for additional indicators",
            ],
            confidence_score=0.8,
        )

        return finding

    def _create_ttp_finding(
        self,
        hypothesis: HuntHypothesis,
        event: SecurityEvent,
        matched_ttps: List[str],
        ttp_mapping: Any,
    ) -> HuntFinding:
        """Create TTP-based finding."""
        finding_id = f"finding_{hypothesis.hypothesis_id}_{event.event_id}"

        severity = FindingSeverity.HIGH
        if len(matched_ttps) > 1:
            severity = FindingSeverity.CRITICAL

        finding = HuntFinding(
            finding_id=finding_id,
            hypothesis_id=hypothesis.hypothesis_id,
            severity=severity,
            title=f"TTP Match: {hypothesis.name}",
            description=f"Event matched MITRE ATT&CK technique(s): {', '.join(matched_ttps)}",
            evidence=[
                {
                    "type": "security_event",
                    "event_id": event.event_id,
                    "description": event.description,
                },
                {
                    "type": "ttp_mapping",
                    "techniques": matched_ttps,
                    "kill_chain_phase": ttp_mapping.kill_chain_phase,
                },
            ],
            iocs_matched=[],
            ttps_matched=matched_ttps,
            affected_assets=[getattr(event, "source_ip", "unknown")],
            timeline=[
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event": f"TTP detected: {', '.join(matched_ttps)}",
                }
            ],
            recommendations=[
                "Review kill chain phase context",
                "Check for related TTP activity",
                "Implement mitigations from MITRE ATT&CK",
            ],
            confidence_score=0.75,
        )

        return finding

    def _create_behavior_finding(
        self,
        hypothesis: HuntHypothesis,
        title: str,
        events: List[SecurityEvent],
    ) -> HuntFinding:
        """Create behavior-based finding."""
        finding_id = f"finding_{hypothesis.hypothesis_id}_{int(time.time())}"

        finding = HuntFinding(
            finding_id=finding_id,
            hypothesis_id=hypothesis.hypothesis_id,
            severity=FindingSeverity.MEDIUM,
            title=title,
            description=f"Detected behavioral pattern across {len(events)} events",
            evidence=[
                {
                    "type": "security_event",
                    "event_id": e.event_id,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in events[:10]  # Limit to first 10
            ],
            iocs_matched=[],
            ttps_matched=[],
            affected_assets=list(
                set([getattr(e, "source_ip", "unknown") for e in events])
            ),
            timeline=[
                {"timestamp": e.timestamp.isoformat(), "event": e.description}
                for e in events[:5]
            ],
            recommendations=[
                "Investigate source for compromise",
                "Review authentication policies",
                "Enable additional monitoring",
            ],
            confidence_score=0.6,
        )

        return finding

    def _calculate_coverage_metrics(self, campaign: HuntCampaign) -> Dict[str, Any]:
        """Calculate hunt campaign coverage metrics."""
        total_hypotheses = len(campaign.hypotheses)
        hypotheses_with_findings = len(
            set([f.hypothesis_id for f in campaign.findings])
        )

        return {
            "total_hypotheses": total_hypotheses,
            "hypotheses_with_findings": hypotheses_with_findings,
            "coverage_percentage": (
                hypotheses_with_findings / total_hypotheses * 100
                if total_hypotheses > 0
                else 0
            ),
            "total_findings": len(campaign.findings),
            "findings_by_severity": {
                "critical": len(
                    [f for f in campaign.findings if f.severity == FindingSeverity.CRITICAL]
                ),
                "high": len(
                    [f for f in campaign.findings if f.severity == FindingSeverity.HIGH]
                ),
                "medium": len(
                    [f for f in campaign.findings if f.severity == FindingSeverity.MEDIUM]
                ),
                "low": len(
                    [f for f in campaign.findings if f.severity == FindingSeverity.LOW]
                ),
            },
        }

    def add_event_to_history(self, event: SecurityEvent):
        """Add event to hunting history."""
        self.event_history.append(event)

        # Clean old events
        cutoff = datetime.utcnow() - timedelta(hours=self.max_history_hours)
        self.event_history = [e for e in self.event_history if e.timestamp >= cutoff]


# Factory function
def get_threat_hunter(
    config: Config,
    stix_client: Optional[STIXTAXIIClient] = None,
    attack_mapper: Optional[MITREATTACKMapper] = None,
) -> ThreatHunter:
    """Factory function to get ThreatHunter instance."""
    return ThreatHunter(config, stix_client, attack_mapper)
