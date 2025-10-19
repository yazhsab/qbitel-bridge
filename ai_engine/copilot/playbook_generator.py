"""
CRONOS AI - Automated Incident Playbook Generator

Generates actionable incident response playbooks using LLM analysis of
security events and best practices.
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from prometheus_client import Counter, Histogram

from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest, get_llm_service
from ..security.models import SecurityEvent, ThreatLevel, SecurityEventType
from ..core.config import Config
from ..core.exceptions import CronosAIException


# Prometheus metrics
PLAYBOOK_GENERATION_COUNTER = Counter(
    "cronos_playbook_generations_total",
    "Total playbooks generated",
    ["incident_type", "severity"],
    registry=None,
)

PLAYBOOK_GENERATION_DURATION = Histogram(
    "cronos_playbook_generation_duration_seconds",
    "Playbook generation duration",
    registry=None,
)

PLAYBOOK_EXECUTION_COUNTER = Counter(
    "cronos_playbook_executions_total",
    "Total playbook executions",
    ["playbook_type", "status"],
    registry=None,
)


logger = logging.getLogger(__name__)


class PlaybookPhase(str, Enum):
    """Phases of incident response."""

    IDENTIFICATION = "identification"
    CONTAINMENT = "containment"
    ERADICATION = "eradication"
    RECOVERY = "recovery"
    LESSONS_LEARNED = "lessons_learned"


class ActionPriority(str, Enum):
    """Priority levels for playbook actions."""

    CRITICAL = "critical"  # Execute immediately
    HIGH = "high"  # Execute within 15 minutes
    MEDIUM = "medium"  # Execute within 1 hour
    LOW = "low"  # Execute when convenient


class ActionType(str, Enum):
    """Types of playbook actions."""

    INVESTIGATION = "investigation"  # Gather information
    CONTAINMENT = "containment"  # Limit damage
    REMEDIATION = "remediation"  # Fix the issue
    COMMUNICATION = "communication"  # Notify stakeholders
    DOCUMENTATION = "documentation"  # Record actions
    VALIDATION = "validation"  # Verify fix


@dataclass
class PlaybookAction:
    """Individual action in a playbook."""

    action_id: str
    phase: PlaybookPhase
    action_type: ActionType
    priority: ActionPriority
    title: str
    description: str
    commands: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    estimated_duration_minutes: int = 5
    requires_approval: bool = False
    automation_available: bool = False
    validation_criteria: List[str] = field(default_factory=list)
    rollback_procedure: Optional[str] = None


@dataclass
class PlaybookMetadata:
    """Metadata about the incident and playbook."""

    incident_id: str
    incident_type: str
    severity: str
    affected_systems: List[str]
    detected_at: datetime
    generated_at: datetime
    ttl_hours: int = 48  # Time to live for playbook relevance


@dataclass
class IncidentPlaybook:
    """Complete incident response playbook."""

    playbook_id: str
    title: str
    description: str
    metadata: PlaybookMetadata
    actions: List[PlaybookAction]
    success_criteria: List[str]
    escalation_triggers: List[str]
    contact_information: Dict[str, str]
    references: List[str]  # Links to documentation, runbooks
    llm_rationale: str
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "playbook_id": self.playbook_id,
            "title": self.title,
            "description": self.description,
            "metadata": {
                "incident_id": self.metadata.incident_id,
                "incident_type": self.metadata.incident_type,
                "severity": self.metadata.severity,
                "affected_systems": self.metadata.affected_systems,
                "detected_at": self.metadata.detected_at.isoformat(),
                "generated_at": self.metadata.generated_at.isoformat(),
                "ttl_hours": self.metadata.ttl_hours,
            },
            "actions": [
                {
                    "action_id": action.action_id,
                    "phase": action.phase.value,
                    "action_type": action.action_type.value,
                    "priority": action.priority.value,
                    "title": action.title,
                    "description": action.description,
                    "commands": action.commands,
                    "prerequisites": action.prerequisites,
                    "estimated_duration_minutes": action.estimated_duration_minutes,
                    "requires_approval": action.requires_approval,
                    "automation_available": action.automation_available,
                    "validation_criteria": action.validation_criteria,
                    "rollback_procedure": action.rollback_procedure,
                }
                for action in self.actions
            ],
            "success_criteria": self.success_criteria,
            "escalation_triggers": self.escalation_triggers,
            "contact_information": self.contact_information,
            "references": self.references,
            "llm_rationale": self.llm_rationale,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
        }

    def get_actions_by_phase(self, phase: PlaybookPhase) -> List[PlaybookAction]:
        """Get all actions for a specific phase."""
        return [action for action in self.actions if action.phase == phase]

    def get_critical_actions(self) -> List[PlaybookAction]:
        """Get all critical priority actions."""
        return [
            action for action in self.actions if action.priority == ActionPriority.CRITICAL
        ]


class PlaybookGenerator:
    """
    Automated incident response playbook generator.

    Uses LLM to generate contextual, actionable playbooks for security incidents.
    """

    def __init__(self, config: Config, llm_service: Optional[UnifiedLLMService] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_service = llm_service or get_llm_service(config)

        # Standard contact information (configurable)
        self.contact_info = {
            "security_team": getattr(config, "security_team_email", "security@company.com"),
            "incident_commander": getattr(
                config, "incident_commander_email", "incident-response@company.com"
            ),
            "on_call": getattr(config, "on_call_phone", "+1-XXX-XXX-XXXX"),
            "escalation": getattr(config, "escalation_email", "ciso@company.com"),
        }

        # Playbook templates for common scenarios
        self.playbook_templates = self._load_playbook_templates()

    def _load_playbook_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load playbook templates for common incident types."""
        return {
            "unauthorized_access": {
                "title_template": "Unauthorized Access Response - {affected_systems}",
                "base_actions": [
                    "Identify compromised credentials",
                    "Revoke access immediately",
                    "Review access logs",
                    "Assess lateral movement",
                    "Reset affected credentials",
                    "Implement additional monitoring",
                ],
                "mitre_mapping": ["T1078", "T1110", "T1021"],
            },
            "malware_detection": {
                "title_template": "Malware Incident Response - {incident_type}",
                "base_actions": [
                    "Isolate infected systems",
                    "Capture forensic evidence",
                    "Identify malware variant",
                    "Scan all systems",
                    "Remediate infection",
                    "Restore from clean backup",
                ],
                "mitre_mapping": ["T1204", "T1059", "T1105"],
            },
            "data_exfiltration": {
                "title_template": "Data Exfiltration Response - {severity}",
                "base_actions": [
                    "Block outbound connections",
                    "Identify exfiltrated data",
                    "Assess data sensitivity",
                    "Notify legal/compliance",
                    "Preserve evidence",
                    "Implement DLP controls",
                ],
                "mitre_mapping": ["T1048", "T1041", "T1567"],
            },
            "denial_of_service": {
                "title_template": "DoS/DDoS Response - {affected_systems}",
                "base_actions": [
                    "Activate DDoS mitigation",
                    "Analyze traffic patterns",
                    "Block malicious sources",
                    "Scale infrastructure",
                    "Engage ISP/CDN",
                    "Monitor service recovery",
                ],
                "mitre_mapping": ["T1498", "T1499"],
            },
            "privilege_escalation": {
                "title_template": "Privilege Escalation Response - {affected_systems}",
                "base_actions": [
                    "Identify escalation vector",
                    "Revoke elevated privileges",
                    "Patch vulnerabilities",
                    "Review privilege assignments",
                    "Implement least privilege",
                    "Audit administrative access",
                ],
                "mitre_mapping": ["T1068", "T1134", "T1548"],
            },
        }

    async def generate_playbook(
        self,
        incident: SecurityEvent,
        context: Optional[Dict[str, Any]] = None,
    ) -> IncidentPlaybook:
        """
        Generate incident response playbook for a security event.

        Args:
            incident: Security event that triggered the incident
            context: Additional context (affected systems, network topology, etc.)

        Returns:
            Complete incident response playbook
        """
        start_time = time.time()

        try:
            self.logger.info(
                f"Generating playbook for incident: {incident.event_id}"
            )

            # Classify incident type
            incident_type = self._classify_incident(incident)

            # Get template if available
            template = self.playbook_templates.get(incident_type, {})

            # Generate playbook using LLM
            llm_response = await self._generate_llm_playbook(
                incident, incident_type, template, context
            )

            # Parse LLM response
            actions, success_criteria, escalation_triggers, references, confidence = (
                self._parse_llm_response(llm_response, incident, template)
            )

            # Build metadata
            metadata = PlaybookMetadata(
                incident_id=incident.event_id,
                incident_type=incident_type,
                severity=incident.threat_level.value,
                affected_systems=context.get("affected_systems", [])
                if context
                else [],
                detected_at=incident.timestamp,
                generated_at=datetime.utcnow(),
                ttl_hours=48,
            )

            # Create playbook
            playbook_id = f"playbook_{incident.event_id}_{int(time.time())}"
            title = template.get("title_template", "Incident Response Playbook").format(
                affected_systems=", ".join(metadata.affected_systems[:2]),
                incident_type=incident_type,
                severity=metadata.severity,
            )

            playbook = IncidentPlaybook(
                playbook_id=playbook_id,
                title=title,
                description=f"Automated response playbook for {incident_type} incident",
                metadata=metadata,
                actions=actions,
                success_criteria=success_criteria,
                escalation_triggers=escalation_triggers,
                contact_information=self.contact_info,
                references=references,
                llm_rationale=llm_response[:500],
                confidence_score=confidence,
            )

            # Metrics
            PLAYBOOK_GENERATION_COUNTER.labels(
                incident_type=incident_type,
                severity=metadata.severity,
            ).inc()
            PLAYBOOK_GENERATION_DURATION.observe(time.time() - start_time)

            self.logger.info(
                f"Playbook generated: {playbook_id}, "
                f"actions={len(actions)}, "
                f"confidence={confidence:.2f}"
            )

            return playbook

        except Exception as e:
            self.logger.error(f"Error generating playbook: {e}", exc_info=True)
            raise CronosAIException(f"Playbook generation failed: {e}")

    def _classify_incident(self, incident: SecurityEvent) -> str:
        """Classify incident into a category."""
        event_type = incident.event_type.value

        # Simple classification logic
        if "access" in event_type or "authentication" in event_type:
            return "unauthorized_access"
        elif "malware" in event_type or "virus" in event_type:
            return "malware_detection"
        elif "exfiltration" in event_type or "data_transfer" in event_type:
            return "data_exfiltration"
        elif "dos" in event_type or "ddos" in event_type:
            return "denial_of_service"
        elif "privilege" in event_type or "escalation" in event_type:
            return "privilege_escalation"
        else:
            return "generic_incident"

    async def _generate_llm_playbook(
        self,
        incident: SecurityEvent,
        incident_type: str,
        template: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Generate playbook content using LLM."""
        context_str = (
            json.dumps(context, indent=2) if context else "No additional context"
        )

        prompt = f"""
Generate a detailed incident response playbook for the following security incident.

**Incident Details:**
- Event ID: {incident.event_id}
- Type: {incident_type}
- Severity: {incident.threat_level.value}
- Description: {incident.description}
- Source: {incident.source_ip if hasattr(incident, 'source_ip') else 'Unknown'}

**Context:**
{context_str}

**Base Actions (from template):**
{json.dumps(template.get('base_actions', []), indent=2)}

**Task:**
Generate a comprehensive incident response playbook with the following structure:

1. **IDENTIFICATION Phase** - Actions to understand the incident (HIGH priority)
2. **CONTAINMENT Phase** - Actions to limit damage (CRITICAL priority)
3. **ERADICATION Phase** - Actions to remove the threat (HIGH priority)
4. **RECOVERY Phase** - Actions to restore normal operations (MEDIUM priority)
5. **LESSONS LEARNED Phase** - Actions for post-incident review (LOW priority)

For each action, provide:
- Phase (identification, containment, eradication, recovery, lessons_learned)
- Priority (critical, high, medium, low)
- Title (concise action name)
- Description (detailed steps)
- Commands (specific CLI commands if applicable)
- Validation criteria (how to verify success)
- Estimated duration in minutes

Also provide:
- Success criteria (how to know the incident is resolved)
- Escalation triggers (when to escalate to senior management)
- Relevant references (documentation, runbooks, CVEs)

Format as JSON:
{{
  "actions": [
    {{
      "phase": "containment",
      "priority": "critical",
      "title": "Isolate Affected Systems",
      "description": "Immediately disconnect affected systems from the network",
      "commands": ["iptables -A INPUT -s <source_ip> -j DROP"],
      "validation_criteria": ["Verify no traffic from affected systems"],
      "estimated_duration_minutes": 5
    }}
  ],
  "success_criteria": ["criterion1", "criterion2"],
  "escalation_triggers": ["trigger1", "trigger2"],
  "references": ["https://..."],
  "confidence": 0.85
}}
"""

        llm_request = LLMRequest(
            prompt=prompt,
            feature_domain="incident_playbook_generation",
            context={"incident_id": incident.event_id},
            max_tokens=3000,
            temperature=0.2,  # Low temperature for consistent, reliable playbooks
        )

        response = await self.llm_service.query(llm_request)
        return response.content

    def _parse_llm_response(
        self,
        llm_response: str,
        incident: SecurityEvent,
        template: Dict[str, Any],
    ) -> Tuple[List[PlaybookAction], List[str], List[str], List[str], float]:
        """Parse LLM response to extract playbook components."""
        try:
            # Extract JSON from response
            start_idx = llm_response.find("{")
            end_idx = llm_response.rfind("}") + 1
            json_str = llm_response[start_idx:end_idx]
            data = json.loads(json_str)

            # Parse actions
            actions = []
            for idx, action_data in enumerate(data.get("actions", [])):
                action_id = f"action_{incident.event_id}_{idx}"
                action = PlaybookAction(
                    action_id=action_id,
                    phase=PlaybookPhase(action_data.get("phase", "identification")),
                    action_type=self._infer_action_type(action_data.get("title", "")),
                    priority=ActionPriority(action_data.get("priority", "medium")),
                    title=action_data.get("title", "Untitled Action"),
                    description=action_data.get("description", ""),
                    commands=action_data.get("commands", []),
                    prerequisites=action_data.get("prerequisites", []),
                    estimated_duration_minutes=int(
                        action_data.get("estimated_duration_minutes", 10)
                    ),
                    requires_approval=action_data.get("requires_approval", False),
                    automation_available=bool(action_data.get("commands")),
                    validation_criteria=action_data.get("validation_criteria", []),
                    rollback_procedure=action_data.get("rollback_procedure"),
                )
                actions.append(action)

            # Sort actions by phase and priority
            phase_order = {
                PlaybookPhase.IDENTIFICATION: 0,
                PlaybookPhase.CONTAINMENT: 1,
                PlaybookPhase.ERADICATION: 2,
                PlaybookPhase.RECOVERY: 3,
                PlaybookPhase.LESSONS_LEARNED: 4,
            }
            priority_order = {
                ActionPriority.CRITICAL: 0,
                ActionPriority.HIGH: 1,
                ActionPriority.MEDIUM: 2,
                ActionPriority.LOW: 3,
            }
            actions.sort(
                key=lambda a: (phase_order[a.phase], priority_order[a.priority])
            )

            success_criteria = data.get(
                "success_criteria", ["Incident resolved", "Systems restored"]
            )
            escalation_triggers = data.get(
                "escalation_triggers", ["Incident persists >4 hours", "Critical data compromised"]
            )
            references = data.get("references", [])
            confidence = float(data.get("confidence", 0.7))

            return actions, success_criteria, escalation_triggers, references, confidence

        except Exception as e:
            self.logger.warning(f"Failed to parse LLM playbook response: {e}")
            # Return default playbook
            return self._create_default_playbook(incident, template)

    def _infer_action_type(self, title: str) -> ActionType:
        """Infer action type from title."""
        title_lower = title.lower()

        if any(word in title_lower for word in ["investigate", "analyze", "review", "check"]):
            return ActionType.INVESTIGATION
        elif any(word in title_lower for word in ["contain", "isolate", "block", "disconnect"]):
            return ActionType.CONTAINMENT
        elif any(word in title_lower for word in ["remove", "remediate", "patch", "fix"]):
            return ActionType.REMEDIATION
        elif any(word in title_lower for word in ["notify", "communicate", "inform", "alert"]):
            return ActionType.COMMUNICATION
        elif any(word in title_lower for word in ["document", "record", "log"]):
            return ActionType.DOCUMENTATION
        elif any(word in title_lower for word in ["verify", "validate", "test", "confirm"]):
            return ActionType.VALIDATION
        else:
            return ActionType.INVESTIGATION

    def _create_default_playbook(
        self, incident: SecurityEvent, template: Dict[str, Any]
    ) -> Tuple[List[PlaybookAction], List[str], List[str], List[str], float]:
        """Create a default playbook when LLM parsing fails."""
        actions = [
            PlaybookAction(
                action_id=f"action_{incident.event_id}_0",
                phase=PlaybookPhase.IDENTIFICATION,
                action_type=ActionType.INVESTIGATION,
                priority=ActionPriority.HIGH,
                title="Review Incident Details",
                description=f"Analyze incident: {incident.description}",
                commands=[],
                validation_criteria=["Incident scope understood"],
                estimated_duration_minutes=15,
            ),
            PlaybookAction(
                action_id=f"action_{incident.event_id}_1",
                phase=PlaybookPhase.CONTAINMENT,
                action_type=ActionType.CONTAINMENT,
                priority=ActionPriority.CRITICAL,
                title="Implement Immediate Containment",
                description="Apply immediate containment measures to limit incident spread",
                commands=[],
                validation_criteria=["Threat contained"],
                estimated_duration_minutes=30,
                requires_approval=True,
            ),
            PlaybookAction(
                action_id=f"action_{incident.event_id}_2",
                phase=PlaybookPhase.ERADICATION,
                action_type=ActionType.REMEDIATION,
                priority=ActionPriority.HIGH,
                title="Eradicate Threat",
                description="Remove the root cause of the incident",
                commands=[],
                validation_criteria=["Threat removed", "Systems clean"],
                estimated_duration_minutes=60,
            ),
            PlaybookAction(
                action_id=f"action_{incident.event_id}_3",
                phase=PlaybookPhase.RECOVERY,
                action_type=ActionType.REMEDIATION,
                priority=ActionPriority.MEDIUM,
                title="Restore Normal Operations",
                description="Restore affected systems to normal operation",
                commands=[],
                validation_criteria=["Services operational", "Monitoring in place"],
                estimated_duration_minutes=45,
            ),
        ]

        success_criteria = ["Incident resolved", "Systems secured", "Operations restored"]
        escalation_triggers = ["Unable to contain within 2 hours", "Critical systems affected"]
        references = []
        confidence = 0.5

        return actions, success_criteria, escalation_triggers, references, confidence


# Factory function
def get_playbook_generator(
    config: Config, llm_service: Optional[UnifiedLLMService] = None
) -> PlaybookGenerator:
    """Factory function to get PlaybookGenerator instance."""
    return PlaybookGenerator(config, llm_service)
