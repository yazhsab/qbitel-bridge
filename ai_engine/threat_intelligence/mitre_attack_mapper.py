"""
CRONOS AI - MITRE ATT&CK Mapper

Maps security events and threat intelligence to MITRE ATT&CK framework
tactics, techniques, and procedures (TTPs).
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import aiohttp
from prometheus_client import Counter, Gauge

from ..core.config import Config
from ..core.exceptions import CronosAIException
from ..security.models import SecurityEvent, ThreatLevel


# Prometheus metrics
ATTACK_TECHNIQUE_DETECTIONS = Counter(
    "cronos_mitre_attack_technique_detections_total",
    "MITRE ATT&CK technique detections",
    ["technique_id", "tactic"],
    registry=None,
)

ATTACK_COVERAGE = Gauge(
    "cronos_mitre_attack_coverage_percent",
    "Percentage of MITRE ATT&CK techniques with detection coverage",
    registry=None,
)


logger = logging.getLogger(__name__)


class MITRETactic(str, Enum):
    """MITRE ATT&CK tactics (enterprise matrix)."""

    RECONNAISSANCE = "reconnaissance"
    RESOURCE_DEVELOPMENT = "resource-development"
    INITIAL_ACCESS = "initial-access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege-escalation"
    DEFENSE_EVASION = "defense-evasion"
    CREDENTIAL_ACCESS = "credential-access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral-movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command-and-control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


@dataclass
class ATTACKTechnique:
    """MITRE ATT&CK technique."""

    technique_id: str  # e.g., T1595, T1071.001
    name: str
    description: str
    tactics: List[MITRETactic]
    platforms: List[str]  # Windows, Linux, macOS, etc.
    data_sources: List[str]
    detection_methods: List[str]
    mitigations: List[str]
    references: List[Dict[str, str]] = field(default_factory=list)
    subtechniques: List[str] = field(default_factory=list)
    is_subtechnique: bool = False
    parent_technique: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "technique_id": self.technique_id,
            "name": self.name,
            "description": self.description,
            "tactics": [t.value for t in self.tactics],
            "platforms": self.platforms,
            "data_sources": self.data_sources,
            "detection_methods": self.detection_methods,
            "mitigations": self.mitigations,
            "references": self.references,
            "subtechniques": self.subtechniques,
            "is_subtechnique": self.is_subtechnique,
            "parent_technique": self.parent_technique,
        }


@dataclass
class TTPMapping:
    """Mapping of event to MITRE ATT&CK TTPs."""

    event_id: str
    timestamp: datetime
    matched_techniques: List[ATTACKTechnique]
    confidence_scores: Dict[str, float]  # technique_id -> confidence
    detection_rationale: str
    kill_chain_phase: Optional[str] = None
    threat_actor_ttps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "matched_techniques": [t.to_dict() for t in self.matched_techniques],
            "confidence_scores": self.confidence_scores,
            "detection_rationale": self.detection_rationale,
            "kill_chain_phase": self.kill_chain_phase,
            "threat_actor_ttps": self.threat_actor_ttps,
        }


class MITREATTACKMapper:
    """
    MITRE ATT&CK mapper for threat intelligence and event correlation.

    Maps security events to MITRE ATT&CK framework techniques and tactics.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # ATT&CK knowledge base
        self.techniques: Dict[str, ATTACKTechnique] = {}
        self.tactics: Dict[str, List[str]] = {}  # tactic -> [technique_ids]

        # Detection signatures (event patterns -> techniques)
        self.detection_signatures: Dict[str, List[str]] = {}

        # HTTP session for fetching ATT&CK data
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Initialize MITRE ATT&CK mapper."""
        self.logger.info("Initializing MITRE ATT&CK mapper")

        # Create HTTP session
        self.session = aiohttp.ClientSession()

        # Load ATT&CK knowledge base
        await self._load_attack_knowledge_base()

        # Load detection signatures
        self._load_detection_signatures()

        # Calculate coverage
        self._calculate_coverage()

        self.logger.info(
            f"MITRE ATT&CK mapper initialized: "
            f"{len(self.techniques)} techniques loaded"
        )

    async def shutdown(self):
        """Shutdown mapper."""
        if self.session:
            await self.session.close()

    async def _load_attack_knowledge_base(self):
        """Load MITRE ATT&CK knowledge base."""
        # Load subset of common techniques (in production, fetch from MITRE CTI repo)
        self.techniques = self._get_common_techniques()

        # Build tactic index
        for technique in self.techniques.values():
            for tactic in technique.tactics:
                if tactic.value not in self.tactics:
                    self.tactics[tactic.value] = []
                self.tactics[tactic.value].append(technique.technique_id)

    def _get_common_techniques(self) -> Dict[str, ATTACKTechnique]:
        """Get common MITRE ATT&CK techniques."""
        techniques = {
            "T1595": ATTACKTechnique(
                technique_id="T1595",
                name="Active Scanning",
                description="Adversaries may execute active reconnaissance scans.",
                tactics=[MITRETactic.RECONNAISSANCE],
                platforms=["Windows", "Linux", "macOS", "Network"],
                data_sources=["Network Traffic", "Network Traffic Content"],
                detection_methods=["Monitor for suspicious network scans"],
                mitigations=["Network Intrusion Prevention"],
                subtechniques=["T1595.001", "T1595.002"],
            ),
            "T1046": ATTACKTechnique(
                technique_id="T1046",
                name="Network Service Discovery",
                description="Adversaries may attempt to get a listing of services.",
                tactics=[MITRETactic.DISCOVERY],
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["Network Traffic", "Process"],
                detection_methods=["Monitor for port scans", "Detect network discovery tools"],
                mitigations=["Network Segmentation", "Filter Network Traffic"],
            ),
            "T1190": ATTACKTechnique(
                technique_id="T1190",
                name="Exploit Public-Facing Application",
                description="Adversaries may attempt to exploit public-facing applications.",
                tactics=[MITRETactic.INITIAL_ACCESS],
                platforms=["Windows", "Linux", "macOS", "Network"],
                data_sources=["Application Log", "Network Traffic"],
                detection_methods=["Monitor for anomalous network traffic", "WAF alerts"],
                mitigations=["Application Isolation", "Network Segmentation", "Privilege Security"],
            ),
            "T1071": ATTACKTechnique(
                technique_id="T1071",
                name="Application Layer Protocol",
                description="Adversaries may communicate using application layer protocols.",
                tactics=[MITRETactic.COMMAND_AND_CONTROL],
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["Network Traffic", "Network Traffic Content"],
                detection_methods=["Analyze network traffic patterns", "Detect C2 beacons"],
                mitigations=["Network Intrusion Prevention"],
                subtechniques=["T1071.001", "T1071.002", "T1071.003", "T1071.004"],
            ),
            "T1078": ATTACKTechnique(
                technique_id="T1078",
                name="Valid Accounts",
                description="Adversaries may obtain and abuse credentials.",
                tactics=[MITRETactic.INITIAL_ACCESS, MITRETactic.PERSISTENCE, MITRETactic.PRIVILEGE_ESCALATION],
                platforms=["Windows", "Linux", "macOS", "Azure AD", "Google Workspace"],
                data_sources=["Logon Session", "User Account"],
                detection_methods=["Monitor for anomalous logons", "Detect credential stuffing"],
                mitigations=["Multi-factor Authentication", "Password Policies"],
                subtechniques=["T1078.001", "T1078.002", "T1078.003", "T1078.004"],
            ),
            "T1110": ATTACKTechnique(
                technique_id="T1110",
                name="Brute Force",
                description="Adversaries may use brute force techniques.",
                tactics=[MITRETactic.CREDENTIAL_ACCESS],
                platforms=["Windows", "Linux", "macOS", "Azure AD"],
                data_sources=["Application Log", "User Account"],
                detection_methods=["Monitor for failed login attempts", "Detect password spraying"],
                mitigations=["Account Lockout Policy", "Multi-factor Authentication"],
                subtechniques=["T1110.001", "T1110.002", "T1110.003", "T1110.004"],
            ),
            "T1021": ATTACKTechnique(
                technique_id="T1021",
                name="Remote Services",
                description="Adversaries may use valid accounts to log into a service.",
                tactics=[MITRETactic.LATERAL_MOVEMENT],
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["Network Traffic", "Process", "Logon Session"],
                detection_methods=["Monitor for lateral movement", "Detect abnormal remote sessions"],
                mitigations=["Disable Unnecessary Services", "Multi-factor Authentication"],
                subtechniques=["T1021.001", "T1021.002", "T1021.003", "T1021.004", "T1021.005", "T1021.006"],
            ),
            "T1048": ATTACKTechnique(
                technique_id="T1048",
                name="Exfiltration Over Alternative Protocol",
                description="Adversaries may steal data using alternative protocols.",
                tactics=[MITRETactic.EXFILTRATION],
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["Network Traffic", "File"],
                detection_methods=["Monitor for unusual outbound traffic", "Detect DNS exfiltration"],
                mitigations=["Data Loss Prevention", "Network Segmentation"],
                subtechniques=["T1048.001", "T1048.002", "T1048.003"],
            ),
            "T1041": ATTACKTechnique(
                technique_id="T1041",
                name="Exfiltration Over C2 Channel",
                description="Adversaries may steal data over their C2 channel.",
                tactics=[MITRETactic.EXFILTRATION],
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["Network Traffic"],
                detection_methods=["Monitor C2 channels", "Detect large data transfers"],
                mitigations=["Data Loss Prevention", "Network Intrusion Prevention"],
            ),
            "T1498": ATTACKTechnique(
                technique_id="T1498",
                name="Network Denial of Service",
                description="Adversaries may perform DoS attacks.",
                tactics=[MITRETactic.IMPACT],
                platforms=["Windows", "Linux", "macOS", "Network"],
                data_sources=["Network Traffic", "Sensor Health"],
                detection_methods=["Monitor for traffic anomalies", "Detect volumetric attacks"],
                mitigations=["Filter Network Traffic", "DDoS Mitigation Services"],
                subtechniques=["T1498.001", "T1498.002"],
            ),
            "T1486": ATTACKTechnique(
                technique_id="T1486",
                name="Data Encrypted for Impact",
                description="Adversaries may encrypt data to impact availability.",
                tactics=[MITRETactic.IMPACT],
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["File", "Process", "Command"],
                detection_methods=["Monitor for ransomware behavior", "Detect mass file encryption"],
                mitigations=["Data Backup", "Behavior Prevention on Endpoint"],
            ),
            "T1068": ATTACKTechnique(
                technique_id="T1068",
                name="Exploitation for Privilege Escalation",
                description="Adversaries may exploit software vulnerabilities.",
                tactics=[MITRETactic.PRIVILEGE_ESCALATION],
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["Process", "Application Log"],
                detection_methods=["Monitor for exploitation attempts", "Detect privilege escalation"],
                mitigations=["Application Isolation", "Exploit Protection"],
            ),
            "T1557": ATTACKTechnique(
                technique_id="T1557",
                name="Adversary-in-the-Middle",
                description="Adversaries may intercept communications.",
                tactics=[MITRETactic.CREDENTIAL_ACCESS, MITRETactic.COLLECTION],
                platforms=["Windows", "Linux", "macOS", "Network"],
                data_sources=["Network Traffic"],
                detection_methods=["Monitor for ARP spoofing", "Detect SSL/TLS interception"],
                mitigations=["Encrypt Sensitive Information", "Filter Network Traffic"],
                subtechniques=["T1557.001", "T1557.002", "T1557.003"],
            ),
            "T1204": ATTACKTechnique(
                technique_id="T1204",
                name="User Execution",
                description="Adversaries may rely on user interaction.",
                tactics=[MITRETactic.EXECUTION],
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["Process", "File", "Command"],
                detection_methods=["Monitor for suspicious executions", "User training"],
                mitigations=["User Training", "Execution Prevention"],
                subtechniques=["T1204.001", "T1204.002", "T1204.003"],
            ),
            "T1059": ATTACKTechnique(
                technique_id="T1059",
                name="Command and Scripting Interpreter",
                description="Adversaries may abuse command interpreters.",
                tactics=[MITRETactic.EXECUTION],
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["Process", "Command"],
                detection_methods=["Monitor command-line activity", "Detect malicious scripts"],
                mitigations=["Execution Prevention", "Restrict Scripting"],
                subtechniques=["T1059.001", "T1059.002", "T1059.003", "T1059.004", "T1059.005", "T1059.006"],
            ),
        }

        return techniques

    def _load_detection_signatures(self):
        """Load detection signatures mapping events to techniques."""
        # Event pattern -> technique IDs
        self.detection_signatures = {
            "port_scan": ["T1046", "T1595"],
            "brute_force": ["T1110"],
            "unauthorized_access": ["T1078"],
            "lateral_movement": ["T1021"],
            "data_exfiltration": ["T1048", "T1041"],
            "dos_attack": ["T1498"],
            "ransomware": ["T1486"],
            "privilege_escalation": ["T1068"],
            "mitm_attack": ["T1557"],
            "malware_execution": ["T1204", "T1059"],
            "exploit_attempt": ["T1190"],
            "c2_communication": ["T1071"],
        }

    def _calculate_coverage(self):
        """Calculate detection coverage."""
        total_techniques = len(self.techniques)
        covered_techniques = len(
            set(
                tech_id
                for tech_ids in self.detection_signatures.values()
                for tech_id in tech_ids
            )
        )

        coverage = (covered_techniques / total_techniques * 100) if total_techniques > 0 else 0
        ATTACK_COVERAGE.set(coverage)

    async def map_event_to_techniques(
        self, event: SecurityEvent
    ) -> TTPMapping:
        """
        Map security event to MITRE ATT&CK techniques.

        Args:
            event: Security event to analyze

        Returns:
            TTP mapping with matched techniques and confidence scores
        """
        matched_techniques: List[ATTACKTechnique] = []
        confidence_scores: Dict[str, float] = {}

        # Analyze event description and metadata
        event_desc = event.description.lower()
        event_type = event.event_type.value.lower()

        # Pattern matching
        for pattern, technique_ids in self.detection_signatures.items():
            if pattern in event_desc or pattern in event_type:
                for tech_id in technique_ids:
                    if tech_id in self.techniques:
                        technique = self.techniques[tech_id]
                        matched_techniques.append(technique)

                        # Calculate confidence (simple heuristic)
                        base_confidence = 0.7
                        if event.threat_level == ThreatLevel.CRITICAL:
                            base_confidence = 0.9
                        elif event.threat_level == ThreatLevel.HIGH:
                            base_confidence = 0.8

                        confidence_scores[tech_id] = base_confidence

                        # Metrics
                        for tactic in technique.tactics:
                            ATTACK_TECHNIQUE_DETECTIONS.labels(
                                technique_id=tech_id,
                                tactic=tactic.value,
                            ).inc()

        # Determine kill chain phase
        kill_chain_phase = self._determine_kill_chain_phase(matched_techniques)

        # Generate rationale
        rationale = self._generate_detection_rationale(event, matched_techniques)

        mapping = TTPMapping(
            event_id=event.event_id,
            timestamp=datetime.utcnow(),
            matched_techniques=matched_techniques,
            confidence_scores=confidence_scores,
            detection_rationale=rationale,
            kill_chain_phase=kill_chain_phase,
        )

        return mapping

    def _determine_kill_chain_phase(
        self, techniques: List[ATTACKTechnique]
    ) -> Optional[str]:
        """Determine primary kill chain phase from techniques."""
        if not techniques:
            return None

        # Count tactics
        tactic_counts: Dict[str, int] = {}
        for technique in techniques:
            for tactic in technique.tactics:
                tactic_counts[tactic.value] = tactic_counts.get(tactic.value, 0) + 1

        # Return most common tactic
        if tactic_counts:
            return max(tactic_counts.items(), key=lambda x: x[1])[0]

        return None

    def _generate_detection_rationale(
        self, event: SecurityEvent, techniques: List[ATTACKTechnique]
    ) -> str:
        """Generate human-readable detection rationale."""
        if not techniques:
            return "No MITRE ATT&CK techniques matched"

        tech_names = ", ".join([f"{t.technique_id} ({t.name})" for t in techniques[:3]])

        rationale = (
            f"Event '{event.description}' matched {len(techniques)} MITRE ATT&CK technique(s): "
            f"{tech_names}"
        )

        if len(techniques) > 3:
            rationale += f" and {len(techniques) - 3} more"

        return rationale

    async def get_technique(self, technique_id: str) -> Optional[ATTACKTechnique]:
        """Get technique by ID."""
        return self.techniques.get(technique_id)

    async def get_techniques_by_tactic(
        self, tactic: MITRETactic
    ) -> List[ATTACKTechnique]:
        """Get all techniques for a tactic."""
        technique_ids = self.tactics.get(tactic.value, [])
        return [self.techniques[tid] for tid in technique_ids if tid in self.techniques]

    async def search_techniques(
        self, query: str, limit: int = 10
    ) -> List[ATTACKTechnique]:
        """Search techniques by name or description."""
        query_lower = query.lower()
        results = []

        for technique in self.techniques.values():
            if (
                query_lower in technique.name.lower()
                or query_lower in technique.description.lower()
            ):
                results.append(technique)

                if len(results) >= limit:
                    break

        return results

    def get_detection_coverage_report(self) -> Dict[str, Any]:
        """Get detection coverage report."""
        total_techniques = len(self.techniques)
        covered_techniques = set()

        for tech_ids in self.detection_signatures.values():
            covered_techniques.update(tech_ids)

        coverage_by_tactic = {}
        for tactic_name, technique_ids in self.tactics.items():
            covered = len([tid for tid in technique_ids if tid in covered_techniques])
            total = len(technique_ids)
            coverage_by_tactic[tactic_name] = {
                "covered": covered,
                "total": total,
                "percentage": (covered / total * 100) if total > 0 else 0,
            }

        return {
            "total_techniques": total_techniques,
            "covered_techniques": len(covered_techniques),
            "coverage_percentage": (
                len(covered_techniques) / total_techniques * 100
                if total_techniques > 0
                else 0
            ),
            "coverage_by_tactic": coverage_by_tactic,
        }


# Factory function
def get_mitre_attack_mapper(config: Config) -> MITREATTACKMapper:
    """Factory function to get MITREATTACKMapper instance."""
    return MITREATTACKMapper(config)
