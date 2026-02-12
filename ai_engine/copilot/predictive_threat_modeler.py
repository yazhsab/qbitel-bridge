"""
QBITEL - Predictive Threat Modeling Module

Provides "What if?" scenario analysis for security decisions using LLM-powered
threat modeling and policy engine simulation.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from prometheus_client import Counter, Histogram, Gauge

from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest, get_llm_service
from ..security.models import (
    SecurityEvent,
    ThreatLevel,
    ConfidenceLevel,
    ResponseAction,
)
from ..security.decision_engine import ZeroTouchDecisionEngine
from ..policy.policy_engine import PolicyEngine
from ..core.config import Config
from ..core.exceptions import QbitelAIException

# Prometheus metrics
THREAT_MODEL_COUNTER = Counter(
    "qbitel_predictive_threat_models_total",
    "Total predictive threat models generated",
    ["scenario_type", "threat_level"],
    registry=None,
)

THREAT_MODEL_DURATION = Histogram(
    "qbitel_predictive_threat_model_duration_seconds",
    "Threat model generation duration",
    registry=None,
)

SIMULATION_COUNTER = Counter(
    "qbitel_policy_simulations_total",
    "Total policy simulations executed",
    ["outcome"],
    registry=None,
)


logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    """Types of threat modeling scenarios."""

    PORT_CHANGE = "port_change"  # Opening/closing ports
    PROTOCOL_CHANGE = "protocol_change"  # Adding/modifying protocols
    POLICY_CHANGE = "policy_change"  # Security policy modifications
    NETWORK_CHANGE = "network_change"  # Network topology changes
    ACCESS_CHANGE = "access_change"  # Access control changes
    ENCRYPTION_CHANGE = "encryption_change"  # Cryptographic changes


class RiskImpact(str, Enum):
    """Risk impact assessment."""

    CRITICAL = "critical"  # Severe security degradation
    HIGH = "high"  # Significant risk increase
    MEDIUM = "medium"  # Moderate risk increase
    LOW = "low"  # Minimal risk increase
    NEGLIGIBLE = "negligible"  # No material risk change
    POSITIVE = "positive"  # Security improvement


@dataclass
class ThreatScenario:
    """Represents a 'what if' threat scenario."""

    scenario_id: str
    scenario_type: ScenarioType
    description: str
    proposed_change: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatVector:
    """Individual threat vector identified in scenario."""

    vector_id: str
    name: str
    description: str
    likelihood: float  # 0.0 to 1.0
    impact: RiskImpact
    attack_techniques: List[str]  # MITRE ATT&CK techniques
    mitigation_recommendations: List[str]
    cve_references: List[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Results of policy simulation."""

    simulation_id: str
    scenario: ThreatScenario
    current_state: Dict[str, Any]
    proposed_state: Dict[str, Any]
    policy_violations: List[str]
    compliance_impact: Dict[str, Any]
    performance_impact: Dict[str, float]
    recommendation: str  # APPROVE, REJECT, MODIFY


@dataclass
class PredictiveThreatModel:
    """Complete predictive threat model for a scenario."""

    model_id: str
    scenario: ThreatScenario
    threat_vectors: List[ThreatVector]
    overall_risk_score: float  # 0.0 to 10.0
    risk_impact: RiskImpact
    attack_surface_changes: Dict[str, Any]
    simulation_results: SimulationResult
    recommendations: List[str]
    confidence_score: float  # 0.0 to 1.0
    llm_analysis: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "model_id": self.model_id,
            "scenario": {
                "scenario_id": self.scenario.scenario_id,
                "scenario_type": self.scenario.scenario_type.value,
                "description": self.scenario.description,
                "proposed_change": self.scenario.proposed_change,
            },
            "threat_vectors": [
                {
                    "vector_id": tv.vector_id,
                    "name": tv.name,
                    "description": tv.description,
                    "likelihood": tv.likelihood,
                    "impact": tv.impact.value,
                    "attack_techniques": tv.attack_techniques,
                    "mitigation_recommendations": tv.mitigation_recommendations,
                    "cve_references": tv.cve_references,
                }
                for tv in self.threat_vectors
            ],
            "overall_risk_score": self.overall_risk_score,
            "risk_impact": self.risk_impact.value,
            "attack_surface_changes": self.attack_surface_changes,
            "simulation_results": {
                "simulation_id": self.simulation_results.simulation_id,
                "policy_violations": self.simulation_results.policy_violations,
                "compliance_impact": self.simulation_results.compliance_impact,
                "performance_impact": self.simulation_results.performance_impact,
                "recommendation": self.simulation_results.recommendation,
            },
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score,
            "llm_analysis": self.llm_analysis,
            "timestamp": self.timestamp.isoformat(),
        }


class PredictiveThreatModeler:
    """
    Predictive threat modeling using LLM analysis and policy simulation.

    Provides "What if?" scenario analysis to assess security impact of
    proposed changes before implementation.
    """

    def __init__(
        self,
        config: Config,
        llm_service: Optional[UnifiedLLMService] = None,
        policy_engine: Optional[PolicyEngine] = None,
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_service = llm_service or get_llm_service(config)
        self.policy_engine = policy_engine or PolicyEngine(config)

        # MITRE ATT&CK technique database (subset for common scenarios)
        self.attack_techniques = self._load_attack_techniques()

    def _load_attack_techniques(self) -> Dict[str, List[str]]:
        """Load MITRE ATT&CK techniques for common scenarios."""
        return {
            "port_open": [
                "T1046",  # Network Service Discovery
                "T1595",  # Active Scanning
                "T1190",  # Exploit Public-Facing Application
                "T1133",  # External Remote Services
            ],
            "protocol_add": [
                "T1071",  # Application Layer Protocol
                "T1095",  # Non-Application Layer Protocol
                "T1572",  # Protocol Tunneling
            ],
            "encryption_weaken": [
                "T1557",  # Man-in-the-Middle
                "T1600",  # Weaken Encryption
                "T1040",  # Network Sniffing
            ],
            "access_expand": [
                "T1078",  # Valid Accounts
                "T1548",  # Abuse Elevation Control Mechanism
                "T1098",  # Account Manipulation
            ],
        }

    async def model_threat_scenario(
        self, scenario: ThreatScenario, current_context: Optional[Dict[str, Any]] = None
    ) -> PredictiveThreatModel:
        """
        Generate predictive threat model for a scenario.

        Args:
            scenario: The scenario to model
            current_context: Current system context (network topology, policies, etc.)

        Returns:
            Complete predictive threat model with recommendations
        """
        start_time = time.time()

        try:
            self.logger.info(f"Generating threat model for scenario: {scenario.scenario_id}")

            # Step 1: LLM-powered threat analysis
            threat_vectors = await self._analyze_threat_vectors(scenario, current_context)

            # Step 2: Attack surface analysis
            attack_surface_changes = await self._analyze_attack_surface_changes(scenario, current_context)

            # Step 3: Policy simulation
            simulation_results = await self._simulate_policy_impact(scenario, current_context)

            # Step 4: Calculate overall risk score
            overall_risk_score = self._calculate_risk_score(threat_vectors, attack_surface_changes, simulation_results)

            # Step 5: Determine risk impact
            risk_impact = self._determine_risk_impact(overall_risk_score)

            # Step 6: Generate LLM analysis and recommendations
            llm_analysis, recommendations, confidence = await self._generate_llm_analysis(
                scenario,
                threat_vectors,
                attack_surface_changes,
                simulation_results,
                overall_risk_score,
            )

            # Create model
            model_id = f"ptm_{scenario.scenario_id}_{int(time.time())}"
            model = PredictiveThreatModel(
                model_id=model_id,
                scenario=scenario,
                threat_vectors=threat_vectors,
                overall_risk_score=overall_risk_score,
                risk_impact=risk_impact,
                attack_surface_changes=attack_surface_changes,
                simulation_results=simulation_results,
                recommendations=recommendations,
                confidence_score=confidence,
                llm_analysis=llm_analysis,
                metadata={
                    "generation_time_seconds": time.time() - start_time,
                    "context_provided": current_context is not None,
                },
            )

            # Metrics
            THREAT_MODEL_COUNTER.labels(
                scenario_type=scenario.scenario_type.value,
                threat_level=risk_impact.value,
            ).inc()
            THREAT_MODEL_DURATION.observe(time.time() - start_time)

            self.logger.info(
                f"Threat model generated: {model_id}, " f"risk_score={overall_risk_score:.2f}, " f"impact={risk_impact.value}"
            )

            return model

        except Exception as e:
            self.logger.error(f"Error generating threat model: {e}", exc_info=True)
            raise QbitelAIException(f"Threat modeling failed: {e}")

    async def _analyze_threat_vectors(self, scenario: ThreatScenario, context: Optional[Dict[str, Any]]) -> List[ThreatVector]:
        """Analyze potential threat vectors using LLM."""
        prompt = self._build_threat_vector_prompt(scenario, context)

        llm_request = LLMRequest(
            prompt=prompt,
            feature_domain="predictive_threat_modeling",
            context={"scenario": asdict(scenario)},
            max_tokens=2000,
            temperature=0.2,  # Low temperature for consistent analysis
        )

        response = await self.llm_service.query(llm_request)

        # Parse LLM response to extract threat vectors
        threat_vectors = self._parse_threat_vectors(response.content, scenario)

        return threat_vectors

    def _build_threat_vector_prompt(self, scenario: ThreatScenario, context: Optional[Dict[str, Any]]) -> str:
        """Build LLM prompt for threat vector analysis."""
        context_str = json.dumps(context, indent=2) if context else "No context provided"

        return f"""
Analyze the following security scenario and identify potential threat vectors.

**Scenario:**
Type: {scenario.scenario_type.value}
Description: {scenario.description}
Proposed Change: {json.dumps(scenario.proposed_change, indent=2)}

**Current Context:**
{context_str}

**Task:**
Identify specific threat vectors that could be exploited as a result of this change.
For each threat vector, provide:
1. Name: Concise name for the threat
2. Description: Detailed explanation of the threat
3. Likelihood: Probability of exploitation (0.0 to 1.0)
4. Impact: Severity (critical, high, medium, low, negligible)
5. Attack Techniques: MITRE ATT&CK technique IDs if applicable
6. Mitigation Recommendations: Specific steps to reduce risk

Format your response as JSON:
{{
  "threat_vectors": [
    {{
      "name": "Threat Name",
      "description": "Detailed description",
      "likelihood": 0.7,
      "impact": "high",
      "attack_techniques": ["T1046", "T1190"],
      "mitigation_recommendations": ["Recommendation 1", "Recommendation 2"],
      "cve_references": ["CVE-2023-XXXXX"]
    }}
  ]
}}
"""

    def _parse_threat_vectors(self, llm_response: str, scenario: ThreatScenario) -> List[ThreatVector]:
        """Parse LLM response to extract threat vectors."""
        try:
            # Extract JSON from response
            start_idx = llm_response.find("{")
            end_idx = llm_response.rfind("}") + 1
            json_str = llm_response[start_idx:end_idx]
            data = json.loads(json_str)

            threat_vectors = []
            for idx, tv_data in enumerate(data.get("threat_vectors", [])):
                vector_id = f"tv_{scenario.scenario_id}_{idx}"
                threat_vector = ThreatVector(
                    vector_id=vector_id,
                    name=tv_data.get("name", "Unknown Threat"),
                    description=tv_data.get("description", ""),
                    likelihood=float(tv_data.get("likelihood", 0.5)),
                    impact=RiskImpact(tv_data.get("impact", "medium")),
                    attack_techniques=tv_data.get("attack_techniques", []),
                    mitigation_recommendations=tv_data.get("mitigation_recommendations", []),
                    cve_references=tv_data.get("cve_references", []),
                )
                threat_vectors.append(threat_vector)

            return threat_vectors

        except Exception as e:
            self.logger.warning(f"Failed to parse threat vectors from LLM response: {e}")
            # Return default threat vector
            return [
                ThreatVector(
                    vector_id=f"tv_{scenario.scenario_id}_default",
                    name="Generic Security Risk",
                    description=f"Potential security implications of {scenario.scenario_type.value}",
                    likelihood=0.5,
                    impact=RiskImpact.MEDIUM,
                    attack_techniques=[],
                    mitigation_recommendations=[
                        "Review change carefully",
                        "Implement monitoring",
                        "Test in staging environment",
                    ],
                )
            ]

    async def _analyze_attack_surface_changes(
        self, scenario: ThreatScenario, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze how the scenario changes the attack surface."""
        attack_surface = {
            "exposed_ports": [],
            "new_protocols": [],
            "encryption_changes": {},
            "access_control_changes": {},
            "network_exposure_change": 0.0,  # -1.0 to 1.0 (negative = improvement)
        }

        # Analyze based on scenario type
        if scenario.scenario_type == ScenarioType.PORT_CHANGE:
            proposed_ports = scenario.proposed_change.get("ports", [])
            attack_surface["exposed_ports"] = proposed_ports
            # More open ports = larger attack surface
            attack_surface["network_exposure_change"] = len(proposed_ports) * 0.1

        elif scenario.scenario_type == ScenarioType.PROTOCOL_CHANGE:
            new_protocols = scenario.proposed_change.get("protocols", [])
            attack_surface["new_protocols"] = new_protocols
            attack_surface["network_exposure_change"] = len(new_protocols) * 0.15

        elif scenario.scenario_type == ScenarioType.ENCRYPTION_CHANGE:
            encryption_data = scenario.proposed_change.get("encryption", {})
            attack_surface["encryption_changes"] = encryption_data
            # Weakening encryption increases risk
            if encryption_data.get("strength") == "weaken":
                attack_surface["network_exposure_change"] = 0.5

        elif scenario.scenario_type == ScenarioType.ACCESS_CHANGE:
            access_data = scenario.proposed_change.get("access_control", {})
            attack_surface["access_control_changes"] = access_data
            # Expanding access increases risk
            if access_data.get("direction") == "expand":
                attack_surface["network_exposure_change"] = 0.3

        return attack_surface

    async def _simulate_policy_impact(self, scenario: ThreatScenario, context: Optional[Dict[str, Any]]) -> SimulationResult:
        """Simulate the impact of the scenario on security policies."""
        simulation_id = f"sim_{scenario.scenario_id}_{int(time.time())}"

        # Get current state from context
        current_state = context.get("current_state", {}) if context else {}

        # Build proposed state
        proposed_state = {**current_state, **scenario.proposed_change}

        # Check for policy violations
        policy_violations = await self._check_policy_violations(proposed_state, scenario)

        # Assess compliance impact
        compliance_impact = self._assess_compliance_impact(scenario, policy_violations)

        # Estimate performance impact
        performance_impact = self._estimate_performance_impact(scenario)

        # Generate recommendation
        recommendation = self._generate_recommendation(policy_violations, compliance_impact, performance_impact)

        result = SimulationResult(
            simulation_id=simulation_id,
            scenario=scenario,
            current_state=current_state,
            proposed_state=proposed_state,
            policy_violations=policy_violations,
            compliance_impact=compliance_impact,
            performance_impact=performance_impact,
            recommendation=recommendation,
        )

        # Metrics
        SIMULATION_COUNTER.labels(outcome=recommendation).inc()

        return result

    async def _check_policy_violations(self, proposed_state: Dict[str, Any], scenario: ThreatScenario) -> List[str]:
        """Check for policy violations in proposed state."""
        violations = []

        # Example policy checks
        if scenario.scenario_type == ScenarioType.PORT_CHANGE:
            ports = scenario.proposed_change.get("ports", [])
            # Check for prohibited ports
            prohibited_ports = {23, 21, 69, 135, 139, 445}  # Telnet, FTP, TFTP, RPC
            for port in ports:
                if port in prohibited_ports:
                    violations.append(f"Port {port} is prohibited by security policy (insecure protocol)")

            # Check for high-risk port ranges
            for port in ports:
                if 1024 <= port <= 49151:  # Registered ports
                    violations.append(f"Port {port} requires security review (registered port range)")

        elif scenario.scenario_type == ScenarioType.ENCRYPTION_CHANGE:
            encryption = scenario.proposed_change.get("encryption", {})
            if encryption.get("algorithm") in ["DES", "3DES", "RC4"]:
                violations.append(f"Encryption algorithm {encryption.get('algorithm')} is deprecated and prohibited")

        return violations

    def _assess_compliance_impact(self, scenario: ThreatScenario, policy_violations: List[str]) -> Dict[str, Any]:
        """Assess impact on compliance frameworks."""
        impact = {
            "frameworks_affected": [],
            "severity": "none",
            "required_actions": [],
        }

        if policy_violations:
            impact["severity"] = "high" if len(policy_violations) > 2 else "medium"
            impact["frameworks_affected"] = ["PCI-DSS", "SOC2", "ISO27001"]
            impact["required_actions"] = [
                "Document exception request",
                "Obtain management approval",
                "Update risk register",
            ]

        return impact

    def _estimate_performance_impact(self, scenario: ThreatScenario) -> Dict[str, float]:
        """Estimate performance impact of the change."""
        impact = {
            "cpu_overhead_percent": 0.0,
            "memory_overhead_mb": 0.0,
            "latency_increase_ms": 0.0,
            "throughput_reduction_percent": 0.0,
        }

        # Estimates based on scenario type
        if scenario.scenario_type == ScenarioType.ENCRYPTION_CHANGE:
            impact["cpu_overhead_percent"] = 5.0
            impact["latency_increase_ms"] = 2.0

        elif scenario.scenario_type == ScenarioType.PROTOCOL_CHANGE:
            impact["cpu_overhead_percent"] = 3.0
            impact["memory_overhead_mb"] = 50.0
            impact["latency_increase_ms"] = 1.0

        return impact

    def _generate_recommendation(
        self,
        policy_violations: List[str],
        compliance_impact: Dict[str, Any],
        performance_impact: Dict[str, float],
    ) -> str:
        """Generate recommendation: APPROVE, REJECT, or MODIFY."""
        # REJECT if critical violations
        if len(policy_violations) > 3:
            return "REJECT"

        # MODIFY if some violations or high compliance impact
        if policy_violations or compliance_impact["severity"] in ["high", "critical"]:
            return "MODIFY"

        # Check performance impact
        if performance_impact.get("cpu_overhead_percent", 0) > 20:
            return "MODIFY"

        # APPROVE if no major issues
        return "APPROVE"

    def _calculate_risk_score(
        self,
        threat_vectors: List[ThreatVector],
        attack_surface_changes: Dict[str, Any],
        simulation_results: SimulationResult,
    ) -> float:
        """Calculate overall risk score (0.0 to 10.0)."""
        if not threat_vectors:
            return 0.0

        # Weight threat vectors by likelihood and impact
        threat_score = 0.0
        impact_weights = {
            RiskImpact.CRITICAL: 1.0,
            RiskImpact.HIGH: 0.8,
            RiskImpact.MEDIUM: 0.5,
            RiskImpact.LOW: 0.3,
            RiskImpact.NEGLIGIBLE: 0.1,
            RiskImpact.POSITIVE: -0.2,
        }

        for tv in threat_vectors:
            weight = impact_weights.get(tv.impact, 0.5)
            threat_score += tv.likelihood * weight * 10.0

        # Normalize by number of vectors
        threat_score = threat_score / max(len(threat_vectors), 1)

        # Factor in attack surface changes
        surface_multiplier = 1.0 + max(attack_surface_changes.get("network_exposure_change", 0), 0)
        threat_score *= surface_multiplier

        # Factor in policy violations
        violation_penalty = len(simulation_results.policy_violations) * 0.5
        threat_score += violation_penalty

        # Cap at 10.0
        return min(threat_score, 10.0)

    def _determine_risk_impact(self, risk_score: float) -> RiskImpact:
        """Determine risk impact category from score."""
        if risk_score >= 8.0:
            return RiskImpact.CRITICAL
        elif risk_score >= 6.0:
            return RiskImpact.HIGH
        elif risk_score >= 4.0:
            return RiskImpact.MEDIUM
        elif risk_score >= 2.0:
            return RiskImpact.LOW
        else:
            return RiskImpact.NEGLIGIBLE

    async def _generate_llm_analysis(
        self,
        scenario: ThreatScenario,
        threat_vectors: List[ThreatVector],
        attack_surface_changes: Dict[str, Any],
        simulation_results: SimulationResult,
        risk_score: float,
    ) -> Tuple[str, List[str], float]:
        """Generate comprehensive LLM analysis and recommendations."""
        prompt = f"""
Provide a comprehensive security analysis for the following scenario:

**Scenario:**
Type: {scenario.scenario_type.value}
Description: {scenario.description}
Proposed Change: {json.dumps(scenario.proposed_change, indent=2)}

**Threat Analysis:**
- Identified Threat Vectors: {len(threat_vectors)}
- Overall Risk Score: {risk_score:.2f}/10.0
- Attack Surface Changes: {json.dumps(attack_surface_changes, indent=2)}

**Policy Simulation:**
- Policy Violations: {len(simulation_results.policy_violations)}
- Recommendation: {simulation_results.recommendation}

**Task:**
1. Provide a concise executive summary of the security implications
2. List 3-5 specific, actionable recommendations to proceed safely
3. Assess your confidence in this analysis (0.0 to 1.0)

Format as JSON:
{{
  "executive_summary": "...",
  "recommendations": ["rec1", "rec2", "rec3"],
  "confidence": 0.85
}}
"""

        llm_request = LLMRequest(
            prompt=prompt,
            feature_domain="predictive_threat_modeling",
            max_tokens=1500,
            temperature=0.3,
        )

        response = await self.llm_service.query(llm_request)

        # Parse response
        try:
            start_idx = response.content.find("{")
            end_idx = response.content.rfind("}") + 1
            json_str = response.content[start_idx:end_idx]
            data = json.loads(json_str)

            analysis = data.get("executive_summary", response.content[:500])
            recommendations = data.get(
                "recommendations",
                [
                    "Review change in staging environment",
                    "Implement additional monitoring",
                    "Document security implications",
                ],
            )
            confidence = float(data.get("confidence", 0.7))

        except Exception as e:
            self.logger.warning(f"Failed to parse LLM analysis: {e}")
            analysis = response.content[:500]
            recommendations = ["Review carefully before implementing"]
            confidence = 0.5

        return analysis, recommendations, confidence


# Factory function
def get_predictive_threat_modeler(
    config: Config,
    llm_service: Optional[UnifiedLLMService] = None,
    policy_engine: Optional[PolicyEngine] = None,
) -> PredictiveThreatModeler:
    """Factory function to get PredictiveThreatModeler instance."""
    return PredictiveThreatModeler(config, llm_service, policy_engine)
