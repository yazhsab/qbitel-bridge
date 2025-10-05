"""
CRONOS AI Engine - Zero-Touch Decision Engine

Enterprise-grade autonomous security decision making with LLM integration.
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest, get_llm_service
from ..core.config import Config
from ..core.exceptions import CronosAIException
from ..monitoring.metrics import MetricsCollector
from .models import (
    SecurityEventType,
    SystemCriticality,
    SecurityEvent,
    ThreatAnalysis,
    AutomatedResponse,
    SecurityContext,
    LegacySystem,
    ResponseAction,
    ThreatLevel,
    ConfidenceLevel,
    ResponseType,
    SecurityException,
    ThreatAnalysisException,
    ResponseExecutionException,
    validate_confidence_score,
    calculate_risk_score,
)
from .threat_analyzer import ThreatAnalyzer

from prometheus_client import Counter, Histogram, Gauge


_METRIC_CACHE = {}


def _get_metric(metric_cls, name: str, *args, **kwargs):
    """Return cached Prometheus metric or create an unregistered instance."""

    if name in _METRIC_CACHE:
        return _METRIC_CACHE[name]

    kwargs = dict(kwargs)
    kwargs.setdefault("registry", None)
    metric = metric_cls(name, *args, **kwargs)
    _METRIC_CACHE[name] = metric
    return metric


# Prometheus metrics
DECISION_COUNTER = _get_metric(
    Counter,
    "cronos_security_decisions_total",
    "Security decisions made",
    ["decision_type", "confidence_level"],
)
DECISION_DURATION = _get_metric(
    Histogram, "cronos_security_decision_duration_seconds", "Decision making duration"
)
RESPONSE_EXECUTION_COUNTER = _get_metric(
    Counter,
    "cronos_security_responses_total",
    "Security responses executed",
    ["response_type", "status"],
)
AUTO_EXECUTE_GAUGE = _get_metric(
    Gauge, "cronos_security_auto_executions", "Autonomous executions per hour"
)
HUMAN_ESCALATIONS_GAUGE = _get_metric(
    Gauge, "cronos_security_human_escalations", "Human escalations per hour"
)

logger = logging.getLogger(__name__)


class ZeroTouchDecisionEngine:
    """
    Autonomous security decision making engine with LLM-powered analysis.

    This engine analyzes security events, assesses threats, and makes autonomous
    decisions about response actions while considering legacy system constraints.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core services
        self.llm_service: Optional[UnifiedLLMService] = None
        self.threat_analyzer: Optional[ThreatAnalyzer] = None
        self.metrics_collector: Optional[MetricsCollector] = None

        # Decision thresholds and configuration
        self.confidence_thresholds = {
            "auto_execute": 0.95,  # Very high confidence required for autonomous execution
            "auto_approve": 0.85,  # High confidence for auto-approval
            "escalate_threshold": 0.50,  # Below this, always escalate
        }

        # Response type risk scores (0.0 = low risk, 1.0 = high risk)
        self.response_risk_scores = {
            ResponseType.ALERT_SECURITY_TEAM: 0.0,
            ResponseType.ENABLE_MONITORING: 0.1,
            ResponseType.LOG_RETENTION_INCREASE: 0.1,
            ResponseType.BLOCK_IP: 0.3,
            ResponseType.DEPLOY_HONEYPOT: 0.3,
            ResponseType.REDIRECT_TRAFFIC: 0.4,
            ResponseType.VIRTUAL_PATCH: 0.5,
            ResponseType.RESET_CREDENTIALS: 0.6,
            ResponseType.NETWORK_SEGMENTATION: 0.7,
            ResponseType.QUARANTINE: 0.8,
            ResponseType.ISOLATE_SYSTEM: 0.8,
            ResponseType.SHUTDOWN_SERVICE: 0.9,
            ResponseType.PATCH_VULNERABILITY: 0.9,
        }

        # Decision history for learning
        self.decision_history: List[Dict[str, Any]] = []
        self.success_rates: Dict[str, float] = {}

        # State management
        self._initialized = False

        self.logger.info("Zero-Touch Decision Engine initialized")

    async def initialize(self) -> None:
        """Initialize the decision engine and its dependencies."""
        if self._initialized:
            return

        try:
            self.logger.info("Initializing Zero-Touch Decision Engine...")

            # Get LLM service - must be initialized first
            self.llm_service = get_llm_service()
            if self.llm_service is None:
                raise SecurityException(
                    "LLM service not initialized. Call initialize_llm_service() first."
                )

            # Initialize threat analyzer
            self.threat_analyzer = ThreatAnalyzer(self.config)
            await self.threat_analyzer.initialize()

            # Initialize metrics collector
            if hasattr(self.config, "metrics_enabled") and self.config.metrics_enabled:
                self.metrics_collector = MetricsCollector(self.config)
                await self.metrics_collector.initialize()

            self._initialized = True
            self.logger.info("Zero-Touch Decision Engine initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Decision Engine: {e}")
            raise SecurityException(f"Decision Engine initialization failed: {e}")

    async def analyze_and_respond(
        self,
        security_event: SecurityEvent,
        security_context: Optional[SecurityContext] = None,
        legacy_systems: Optional[List[LegacySystem]] = None,
    ) -> AutomatedResponse:
        """
        Analyze security event and generate autonomous response plan.

        Args:
            security_event: The security event to analyze
            security_context: Current security context
            legacy_systems: Affected legacy systems information

        Returns:
            Complete automated response plan with actions
        """
        start_time = time.time()

        try:
            if not self._initialized:
                await self.initialize()

            self.logger.info(f"Analyzing security event: {security_event.event_id}")

            # Step 1: Comprehensive threat analysis
            threat_analysis = await self._analyze_threat(
                security_event, security_context, legacy_systems
            )

            # Step 2: Assess business impact
            business_impact = await self._assess_business_impact(
                security_event, threat_analysis, legacy_systems
            )

            # Step 3: Generate response options
            response_options = await self._generate_response_options(
                security_event, threat_analysis, business_impact, legacy_systems
            )

            # Step 4: LLM-powered decision making
            decision = await self._make_llm_decision(
                security_event,
                threat_analysis,
                business_impact,
                response_options,
                security_context,
            )

            # Step 5: Create automated response plan
            automated_response = await self._create_response_plan(
                security_event, threat_analysis, decision, legacy_systems
            )

            # Step 6: Apply safety checks and constraints
            validated_response = await self._apply_safety_constraints(
                automated_response, legacy_systems
            )

            # Update metrics
            processing_time = time.time() - start_time
            DECISION_DURATION.observe(processing_time)
            DECISION_COUNTER.labels(
                decision_type=threat_analysis.threat_classification.value,
                confidence_level=decision["confidence_level"],
            ).inc()

            self.logger.info(
                f"Decision completed for event {security_event.event_id} "
                f"in {processing_time:.2f}s with confidence {decision['confidence']:.2f}"
            )

            return validated_response

        except Exception as e:
            self.logger.error(
                f"Decision making failed for event {security_event.event_id}: {e}"
            )

            # Create fallback escalation response
            fallback_response = await self._create_fallback_response(
                security_event, str(e)
            )
            return fallback_response

    async def _analyze_threat(
        self,
        security_event: SecurityEvent,
        security_context: Optional[SecurityContext],
        legacy_systems: Optional[List[LegacySystem]],
    ) -> ThreatAnalysis:
        """Perform comprehensive threat analysis."""

        if not self.threat_analyzer:
            raise ThreatAnalysisException("Threat analyzer not initialized")

        # Use the threat analyzer to get detailed analysis
        threat_analysis = await self.threat_analyzer.analyze_threat(
            security_event, security_context, legacy_systems
        )

        # Enhance with LLM analysis for context and insights
        llm_analysis = await self._get_llm_threat_analysis(
            security_event, threat_analysis
        )
        threat_analysis.expert_analysis = llm_analysis.get("analysis")
        threat_analysis.contextual_factors = llm_analysis.get("contextual_factors", [])
        threat_analysis.similar_incidents = llm_analysis.get("similar_incidents", [])

        return threat_analysis

    async def _assess_business_impact(
        self,
        security_event: SecurityEvent,
        threat_analysis: ThreatAnalysis,
        legacy_systems: Optional[List[LegacySystem]],
    ) -> Dict[str, Any]:
        """Assess potential business impact of the security event."""

        business_impact = {
            "financial_impact": 0.0,
            "operational_impact": "low",
            "reputation_impact": "low",
            "compliance_impact": "none",
            "affected_users": 0,
            "estimated_downtime": 0,
            "recovery_time": 0,
        }

        # Calculate impact based on affected systems
        if legacy_systems:
            total_criticality_score = 0.0
            max_users = 0

            for system in legacy_systems:
                if system.system_id in security_event.affected_systems:
                    # Add criticality weight
                    if system.criticality.value == "mission_critical":
                        total_criticality_score += 1.0
                        business_impact["operational_impact"] = "critical"
                    elif system.criticality.value == "business_critical":
                        total_criticality_score += 0.8
                        if business_impact["operational_impact"] != "critical":
                            business_impact["operational_impact"] = "high"
                    elif system.criticality.value == "important":
                        total_criticality_score += 0.6
                        if business_impact["operational_impact"] not in [
                            "critical",
                            "high",
                        ]:
                            business_impact["operational_impact"] = "medium"

                    # Aggregate user impact
                    max_users = max(max_users, getattr(system, "users_count", 0) or 0)

                    # Check compliance implications
                    if system.compliance_requirements:
                        business_impact["compliance_impact"] = "high"
                        business_impact["compliance_requirements"] = (
                            system.compliance_requirements
                        )

            business_impact["affected_users"] = max_users
            business_impact["criticality_score"] = total_criticality_score

            # Estimate financial impact based on system criticality and threat level
            threat_multiplier = {
                ThreatLevel.CRITICAL: 5.0,
                ThreatLevel.HIGH: 3.0,
                ThreatLevel.MEDIUM: 2.0,
                ThreatLevel.LOW: 1.0,
                ThreatLevel.INFO: 0.5,
            }.get(threat_analysis.threat_level, 1.0)

            business_impact["financial_impact"] = (
                total_criticality_score * threat_multiplier * 100000
            )  # Base cost

        # Use LLM for contextual business impact assessment
        llm_impact = await self._get_llm_business_impact(
            security_event, threat_analysis, business_impact
        )
        business_impact.update(llm_impact)

        return business_impact

    async def _generate_response_options(
        self,
        security_event: SecurityEvent,
        threat_analysis: ThreatAnalysis,
        business_impact: Dict[str, Any],
        legacy_systems: Optional[List[LegacySystem]],
    ) -> List[Dict[str, Any]]:
        """Generate possible response options based on threat analysis."""

        response_options = []

        # Base response options based on threat type and level
        base_responses = self._get_base_responses(
            security_event.event_type, threat_analysis.threat_level
        )

        for response_type in base_responses:
            option = {
                "response_type": response_type,
                "risk_score": self.response_risk_scores.get(response_type, 0.5),
                "estimated_effectiveness": self._estimate_effectiveness(
                    response_type, threat_analysis
                ),
                "business_impact": self._assess_response_business_impact(
                    response_type, business_impact
                ),
                "legacy_considerations": self._assess_legacy_considerations(
                    response_type, legacy_systems
                ),
            }
            response_options.append(option)

        # Sort by effectiveness and risk balance
        response_options.sort(
            key=lambda x: x["estimated_effectiveness"] - x["risk_score"], reverse=True
        )

        return response_options

    async def _make_llm_decision(
        self,
        security_event: SecurityEvent,
        threat_analysis: ThreatAnalysis,
        business_impact: Dict[str, Any],
        response_options: List[Dict[str, Any]],
        security_context: Optional[SecurityContext],
    ) -> Dict[str, Any]:
        """Use LLM to make the final security decision."""

        # Create comprehensive context for LLM decision
        decision_context = {
            "event_summary": {
                "type": security_event.event_type.value,
                "severity": security_event.threat_level.value,
                "affected_systems": security_event.affected_systems,
                "source_ip": security_event.source_ip,
                "description": security_event.description,
            },
            "threat_analysis": {
                "classification": threat_analysis.threat_classification.value,
                "confidence": threat_analysis.confidence_score,
                "business_impact_score": threat_analysis.business_impact_score,
                "ttps": threat_analysis.ttps,
                "mitre_techniques": threat_analysis.mitre_attack_techniques,
            },
            "business_impact": business_impact,
            "response_options": response_options[:5],  # Top 5 options
            "security_context": {
                "current_threat_level": (
                    security_context.current_threat_level.value
                    if security_context
                    else "medium"
                ),
                "active_incidents": (
                    len(security_context.active_incidents) if security_context else 0
                ),
                "business_hours": (
                    security_context.business_hours if security_context else True
                ),
            },
        }

        # Create LLM prompt for decision making
        decision_prompt = self._create_decision_prompt(decision_context)

        llm_request = LLMRequest(
            prompt=decision_prompt,
            feature_domain="security_orchestrator",
            context=decision_context,
            max_tokens=2000,
            temperature=0.1,  # Low temperature for consistent decisions
        )

        try:
            llm_response = await self.llm_service.process_request(llm_request)
            decision_data = self._parse_llm_decision(llm_response.content)

            # Validate and adjust decision based on safety constraints
            validated_decision = self._validate_llm_decision(
                decision_data, threat_analysis, business_impact
            )

            return {
                "recommended_actions": validated_decision.get("actions", []),
                "confidence": validated_decision.get("confidence", 0.5),
                "confidence_level": self._get_confidence_level(
                    validated_decision.get("confidence", 0.5)
                ),
                "reasoning": validated_decision.get("reasoning", ""),
                "risk_assessment": validated_decision.get("risk_assessment", {}),
                "auto_execute": self._should_auto_execute(validated_decision),
                "escalation_needed": validated_decision.get("escalation_needed", True),
                "llm_raw_response": llm_response.content,
            }

        except Exception as e:
            self.logger.error(f"LLM decision making failed: {e}")

            # Fallback to rule-based decision
            return self._fallback_decision(
                threat_analysis, business_impact, response_options
            )

    def _create_decision_prompt(self, context: Dict[str, Any]) -> str:
        """Create the decision-making prompt for the LLM."""

        prompt = f"""You are an expert cybersecurity analyst making autonomous security decisions. Analyze the following security event and recommend appropriate response actions.

SECURITY EVENT:
- Type: {context['event_summary']['type']}
- Severity: {context['event_summary']['severity']}
- Description: {context['event_summary']['description']}
- Affected Systems: {', '.join(context['event_summary']['affected_systems'])}
- Source IP: {context['event_summary'].get('source_ip', 'Unknown')}

THREAT ANALYSIS:
- Classification: {context['threat_analysis']['classification']}
- Confidence: {context['threat_analysis']['confidence']:.2f}
- Business Impact Score: {context['threat_analysis']['business_impact_score']:.2f}
- TTPs: {', '.join(context['threat_analysis']['ttps']) if context['threat_analysis']['ttps'] else 'None identified'}
- MITRE Techniques: {', '.join(context['threat_analysis']['mitre_techniques']) if context['threat_analysis']['mitre_techniques'] else 'None identified'}

BUSINESS IMPACT:
- Financial Impact: ${context['business_impact'].get('financial_impact', 0):,.0f}
- Operational Impact: {context['business_impact'].get('operational_impact', 'Unknown')}
- Affected Users: {context['business_impact'].get('affected_users', 0):,}
- Compliance Impact: {context['business_impact'].get('compliance_impact', 'None')}

AVAILABLE RESPONSE OPTIONS:
"""

        for i, option in enumerate(context["response_options"], 1):
            prompt += (
                f"{i}. {option['response_type'].value.replace('_', ' ').title()}\n"
            )
            prompt += f"   - Risk Score: {option['risk_score']:.2f}\n"
            prompt += f"   - Effectiveness: {option['estimated_effectiveness']:.2f}\n"
            prompt += f"   - Business Impact: {option['business_impact']}\n\n"

        prompt += f"""
CURRENT SECURITY CONTEXT:
- Current Threat Level: {context['security_context']['current_threat_level']}
- Active Incidents: {context['security_context']['active_incidents']}
- Business Hours: {context['security_context']['business_hours']}

Please analyze this situation and provide your recommendation in the following JSON format:

{
    "actions": ["list", "of", "recommended", "action", "types"],
    "confidence": 0.85,
    "reasoning": "Detailed explanation of your decision-making process",
    "risk_assessment": {
        "overall_risk": "high/medium/low",
        "key_risks": ["list", "of", "key", "risks"],
        "mitigation_factors": ["list", "of", "mitigation", "factors"]
    },
    "escalation_needed": false,
    "timeline": "immediate/within_1_hour/within_24_hours",
    "success_probability": 0.8,
    "business_justification": "Why this approach balances security and business needs"
}

Consider:
1. The severity and confidence of the threat
2. Potential business disruption vs security benefit
3. Whether immediate autonomous action is appropriate
4. Legacy system constraints and dependencies
5. Compliance and regulatory implications

Be decisive but cautious. Favor containment over disruption for uncertain threats.
"""

        return prompt

    def _parse_llm_decision(self, llm_content: str) -> Dict[str, Any]:
        """Parse LLM response into structured decision data."""
        try:
            # Try to extract JSON from the response
            start_idx = llm_content.find("{")
            end_idx = llm_content.rfind("}") + 1

            if start_idx != -1 and end_idx != -1:
                json_str = llm_content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback parsing if JSON not found
                return self._fallback_parse_decision(llm_content)

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse LLM JSON response: {e}")
            return self._fallback_parse_decision(llm_content)

    def _fallback_parse_decision(self, content: str) -> Dict[str, Any]:
        """Fallback decision parsing when JSON parsing fails."""

        # Extract basic information using text analysis
        decision = {
            "actions": ["alert_security_team"],  # Safe default
            "confidence": 0.3,  # Low confidence for fallback
            "reasoning": "LLM response parsing failed, using safe defaults",
            "risk_assessment": {"overall_risk": "medium"},
            "escalation_needed": True,
            "timeline": "immediate",
        }

        # Look for common indicators in the text
        content_lower = content.lower()

        if any(word in content_lower for word in ["quarantine", "isolate", "block"]):
            decision["actions"].extend(["quarantine", "enable_monitoring"])
            decision["confidence"] = 0.4

        if any(word in content_lower for word in ["high confidence", "very confident"]):
            decision["confidence"] = min(0.7, decision["confidence"] + 0.3)

        if "no escalation" in content_lower or "autonomous" in content_lower:
            decision["escalation_needed"] = False

        return decision

    def _validate_llm_decision(
        self,
        decision: Dict[str, Any],
        threat_analysis: ThreatAnalysis,
        business_impact: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate and potentially adjust LLM decision based on safety rules."""

        validated_decision = decision.copy()

        # Ensure confidence is valid
        confidence = validated_decision.get("confidence", 0.5)
        if not validate_confidence_score(confidence):
            validated_decision["confidence"] = max(0.0, min(1.0, confidence))

        # Safety rules based on threat level and business impact
        if threat_analysis.threat_level == ThreatLevel.CRITICAL:
            # Critical threats always need human oversight initially
            validated_decision["escalation_needed"] = True
            validated_decision["confidence"] = min(
                validated_decision["confidence"], 0.8
            )

        if business_impact.get("operational_impact") == "critical":
            # Critical business impact requires human approval
            validated_decision["escalation_needed"] = True

        # Ensure actions are valid response types
        valid_actions = []
        for action in validated_decision.get("actions", []):
            try:
                ResponseType(action.lower())
                valid_actions.append(action.lower())
            except ValueError:
                self.logger.warning(f"Invalid response type in LLM decision: {action}")

        if not valid_actions:
            valid_actions = ["alert_security_team"]  # Safe fallback

        validated_decision["actions"] = valid_actions

        return validated_decision

    def _should_auto_execute(self, decision: Dict[str, Any]) -> bool:
        """Determine if actions should be executed autonomously."""

        confidence = decision.get("confidence", 0.0)
        actions = decision.get("actions", [])
        escalation_needed = decision.get("escalation_needed", True)

        # Never auto-execute if escalation is needed
        if escalation_needed:
            return False

        # Check confidence threshold
        if confidence < self.confidence_thresholds["auto_execute"]:
            return False

        # Check action risk levels
        max_risk = 0.0
        for action in actions:
            try:
                response_type = ResponseType(action.lower())
                risk = self.response_risk_scores.get(response_type, 1.0)
                max_risk = max(max_risk, risk)
            except ValueError:
                return False  # Invalid action, don't auto-execute

        # Don't auto-execute high-risk actions
        if max_risk > 0.7:
            return False

        return True

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level enum."""
        if confidence >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    async def _create_response_plan(
        self,
        security_event: SecurityEvent,
        threat_analysis: ThreatAnalysis,
        decision: Dict[str, Any],
        legacy_systems: Optional[List[LegacySystem]],
    ) -> AutomatedResponse:
        """Create detailed automated response plan from decision."""

        response_actions = []

        # Create response actions based on LLM decision
        for action_type in decision.get("actions", []):
            try:
                response_type = ResponseType(action_type.lower())
                action = await self._create_response_action(
                    response_type, security_event, threat_analysis, legacy_systems
                )
                response_actions.append(action)
            except ValueError:
                self.logger.warning(f"Skipping invalid response type: {action_type}")

        # Always include alert action as baseline
        if not any(
            action.action_type == ResponseType.ALERT_SECURITY_TEAM
            for action in response_actions
        ):
            alert_action = await self._create_response_action(
                ResponseType.ALERT_SECURITY_TEAM,
                security_event,
                threat_analysis,
                legacy_systems,
            )
            response_actions.insert(0, alert_action)  # Add at beginning

        # Create the automated response
        automated_response = AutomatedResponse(
            event_id=security_event.event_id,
            analysis_id=threat_analysis.analysis_id,
            response_strategy=f"{threat_analysis.threat_classification.value}_response",
            confidence=self._get_confidence_level(decision.get("confidence", 0.5)),
            confidence_score=decision.get("confidence", 0.5),
            actions=response_actions,
            auto_execute=decision.get("auto_execute", False),
            requires_human_approval=decision.get("escalation_needed", True),
            overall_risk_score=decision.get("risk_assessment", {}).get(
                "overall_risk_score", 0.5
            ),
            estimated_business_impact={
                "reasoning": decision.get("business_justification", ""),
                "timeline": decision.get("timeline", "immediate"),
            },
        )

        return automated_response

    async def _create_response_action(
        self,
        response_type: ResponseType,
        security_event: SecurityEvent,
        threat_analysis: ThreatAnalysis,
        legacy_systems: Optional[List[LegacySystem]],
    ) -> ResponseAction:
        """Create a detailed response action."""

        action = ResponseAction(
            action_type=response_type,
            title=f"{response_type.value.replace('_', ' ').title()} - {security_event.event_type.value}",
            description=self._get_action_description(response_type, security_event),
            target_systems=security_event.affected_systems.copy(),
            risk_level=self._get_action_risk_level(response_type),
            requires_approval=self._action_requires_approval(response_type),
            priority=self._get_action_priority(
                response_type, threat_analysis.threat_level
            ),
        )

        # Add specific parameters based on action type
        if response_type == ResponseType.BLOCK_IP and security_event.source_ip:
            action.parameters = {
                "ip_address": security_event.source_ip,
                "duration": "24h",
                "scope": "organization",
            }
        elif response_type == ResponseType.QUARANTINE:
            action.parameters = {
                "isolation_type": "network_segmentation",
                "allow_management": True,
                "duration": "72h",
            }
        elif response_type == ResponseType.ENABLE_MONITORING:
            action.parameters = {
                "monitoring_level": "enhanced",
                "duration": "7d",
                "include_packet_capture": True,
            }

        # Add legacy system considerations
        if legacy_systems:
            action.metadata["legacy_considerations"] = self._get_legacy_considerations(
                response_type, legacy_systems
            )

        return action

    def _get_action_description(
        self, response_type: ResponseType, security_event: SecurityEvent
    ) -> str:
        """Get human-readable description for the action."""
        descriptions = {
            ResponseType.ALERT_SECURITY_TEAM: f"Alert security team about {security_event.event_type.value}",
            ResponseType.BLOCK_IP: f"Block source IP {security_event.source_ip} to prevent further attacks",
            ResponseType.QUARANTINE: f"Quarantine affected systems to contain the threat",
            ResponseType.ENABLE_MONITORING: f"Enable enhanced monitoring for affected systems",
            ResponseType.ISOLATE_SYSTEM: f"Isolate compromised systems from the network",
            ResponseType.VIRTUAL_PATCH: f"Apply virtual patching to protect against the exploit",
        }
        return descriptions.get(
            response_type, f"Execute {response_type.value} response"
        )

    def _get_action_risk_level(self, response_type: ResponseType) -> ThreatLevel:
        """Get risk level for the action."""
        risk_score = self.response_risk_scores.get(response_type, 0.5)
        if risk_score >= 0.8:
            return ThreatLevel.HIGH
        elif risk_score >= 0.5:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW

    def _action_requires_approval(self, response_type: ResponseType) -> bool:
        """Check if action requires human approval."""
        high_risk_actions = {
            ResponseType.SHUTDOWN_SERVICE,
            ResponseType.ISOLATE_SYSTEM,
            ResponseType.PATCH_VULNERABILITY,
            ResponseType.RESET_CREDENTIALS,
        }
        return response_type in high_risk_actions

    def _get_action_priority(
        self, response_type: ResponseType, threat_level: ThreatLevel
    ) -> int:
        """Get action priority (1 = highest)."""
        base_priorities = {
            ResponseType.ALERT_SECURITY_TEAM: 1,
            ResponseType.BLOCK_IP: 2,
            ResponseType.ENABLE_MONITORING: 3,
            ResponseType.QUARANTINE: 2,
            ResponseType.ISOLATE_SYSTEM: 1,
            ResponseType.VIRTUAL_PATCH: 4,
        }

        base_priority = base_priorities.get(response_type, 5)

        # Adjust based on threat level
        if threat_level == ThreatLevel.CRITICAL:
            return max(1, base_priority - 1)
        elif threat_level == ThreatLevel.LOW:
            return min(5, base_priority + 1)

        return base_priority

    async def _apply_safety_constraints(
        self,
        automated_response: AutomatedResponse,
        legacy_systems: Optional[List[LegacySystem]],
    ) -> AutomatedResponse:
        """Apply safety constraints and validate the response plan."""

        # Check for conflicting actions
        validated_actions = []
        action_types = set()

        for action in automated_response.actions:
            if action.action_type not in action_types:
                validated_actions.append(action)
                action_types.add(action.action_type)

        # Apply business hour constraints
        current_hour = datetime.now().hour
        is_business_hours = 8 <= current_hour <= 18

        if not is_business_hours:
            for action in validated_actions:
                if action.action_type in {
                    ResponseType.SHUTDOWN_SERVICE,
                    ResponseType.PATCH_VULNERABILITY,
                }:
                    action.requires_approval = True
                    action.metadata["constraint_reason"] = (
                        "After business hours - requires approval"
                    )

        # Apply legacy system constraints
        if legacy_systems:
            for action in validated_actions:
                self._apply_legacy_constraints(action, legacy_systems)

        automated_response.actions = validated_actions
        return automated_response

    def _apply_legacy_constraints(
        self, action: ResponseAction, legacy_systems: List[LegacySystem]
    ):
        """Apply constraints specific to legacy systems."""

        for system in legacy_systems:
            if system.system_id in action.target_systems:
                # High criticality systems need approval
                if system.criticality in {
                    SystemCriticality.MISSION_CRITICAL,
                    SystemCriticality.BUSINESS_CRITICAL,
                }:
                    action.requires_approval = True
                    action.metadata.setdefault("constraints", []).append(
                        f"Critical system {system.system_name}"
                    )

                # Check maintenance windows
                if system.maintenance_windows:
                    action.metadata.setdefault("maintenance_windows", []).extend(
                        system.maintenance_windows
                    )

                # Add protocol-specific considerations
                if system.protocol_types:
                    action.metadata.setdefault("protocols", []).extend(
                        [p.value for p in system.protocol_types]
                    )

    async def _create_fallback_response(
        self, security_event: SecurityEvent, error_message: str
    ) -> AutomatedResponse:
        """Create a safe fallback response when decision making fails."""

        # Create basic alert action
        alert_action = ResponseAction(
            action_type=ResponseType.ALERT_SECURITY_TEAM,
            title="Security Event - Decision Engine Error",
            description=f"Security event {security_event.event_type.value} detected. "
            f"Automated decision making failed: {error_message}",
            target_systems=security_event.affected_systems.copy(),
            requires_approval=False,
            priority=1,
        )

        # Create basic monitoring action
        monitoring_action = ResponseAction(
            action_type=ResponseType.ENABLE_MONITORING,
            title="Enable Enhanced Monitoring",
            description="Enable enhanced monitoring due to decision engine failure",
            target_systems=security_event.affected_systems.copy(),
            requires_approval=False,
            priority=2,
            parameters={"monitoring_level": "enhanced", "duration": "4h"},
        )

        return AutomatedResponse(
            event_id=security_event.event_id,
            response_strategy="fallback_safe_response",
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.3,
            actions=[alert_action, monitoring_action],
            auto_execute=False,
            requires_human_approval=True,
            overall_risk_score=0.2,
            metadata={"fallback_reason": error_message},
        )

    # Helper methods for LLM integration
    async def _get_llm_threat_analysis(
        self, security_event: SecurityEvent, threat_analysis: ThreatAnalysis
    ) -> Dict[str, Any]:
        """Get additional threat analysis from LLM."""

        prompt = f"""Provide expert cybersecurity analysis for this security event:

Event Type: {security_event.event_type.value}
Threat Level: {security_event.threat_level.value}  
Description: {security_event.description}
IOCs: {', '.join(security_event.indicators_of_compromise)}

Current Analysis:
- Classification: {threat_analysis.threat_classification.value}
- Confidence: {threat_analysis.confidence_score:.2f}
- Business Impact Score: {threat_analysis.business_impact_score:.2f}

Please provide:
1. Expert analysis and insights
2. Contextual factors to consider
3. Similar historical incidents
4. Key recommendations

Focus on actionable intelligence and context."""

        try:
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="security_orchestrator",
                max_tokens=1500,
                temperature=0.2,
            )

            response = await self.llm_service.process_request(llm_request)

            # Parse response for structured data
            return {
                "analysis": response.content,
                "contextual_factors": self._extract_contextual_factors(
                    response.content
                ),
                "similar_incidents": self._extract_similar_incidents(response.content),
            }

        except Exception as e:
            self.logger.warning(f"LLM threat analysis failed: {e}")
            return {
                "analysis": f"LLM analysis failed: {e}",
                "contextual_factors": [],
                "similar_incidents": [],
            }

    async def _get_llm_business_impact(
        self,
        security_event: SecurityEvent,
        threat_analysis: ThreatAnalysis,
        current_impact: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get business impact assessment from LLM."""

        prompt = f"""Assess the business impact of this security incident:

Security Event: {security_event.event_type.value}
Affected Systems: {', '.join(security_event.affected_systems)}
Threat Level: {threat_analysis.threat_level.value}
Current Impact Assessment: {json.dumps(current_impact, indent=2)}

Please provide enhanced business impact analysis considering:
1. Operational disruption potential
2. Financial implications
3. Regulatory/compliance impact  
4. Reputation risk
5. Recovery time estimates

Provide realistic estimates and context."""

        try:
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="security_orchestrator",
                max_tokens=1000,
                temperature=0.1,
            )

            response = await self.llm_service.process_request(llm_request)

            # Extract enhanced impact data from response
            return self._parse_business_impact_response(
                response.content, current_impact
            )

        except Exception as e:
            self.logger.warning(f"LLM business impact analysis failed: {e}")
            return current_impact

    def _extract_contextual_factors(self, analysis_text: str) -> List[str]:
        """Extract contextual factors from LLM analysis."""
        factors = []

        # Look for common contextual indicators in the text
        lines = analysis_text.split("\n")
        for line in lines:
            line = line.strip()
            if any(
                keyword in line.lower()
                for keyword in ["factor", "consider", "context", "important"]
            ):
                if line and not line.startswith("#") and len(line) > 10:
                    factors.append(line)

        return factors[:10]  # Limit to top 10

    def _extract_similar_incidents(self, analysis_text: str) -> List[str]:
        """Extract similar incidents from LLM analysis."""
        incidents = []

        # Look for references to similar incidents
        lines = analysis_text.split("\n")
        for line in lines:
            if any(
                keyword in line.lower()
                for keyword in ["similar", "previous", "historical", "past"]
            ):
                if line.strip() and len(line.strip()) > 15:
                    incidents.append(line.strip())

        return incidents[:5]  # Limit to top 5

    def _parse_business_impact_response(
        self, response_text: str, current_impact: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse business impact response from LLM."""

        enhanced_impact = current_impact.copy()

        # Look for specific impact indicators in the text
        text_lower = response_text.lower()

        # Update operational impact
        if any(
            word in text_lower
            for word in ["severe", "major disruption", "critical impact"]
        ):
            enhanced_impact["operational_impact"] = "critical"
        elif any(word in text_lower for word in ["significant", "moderate disruption"]):
            enhanced_impact["operational_impact"] = "high"

        # Look for financial estimates
        import re

        financial_patterns = [
            r"\$([0-9,]+)",
            r"([0-9,]+)\s*dollars?",
            r"cost.*?([0-9,]+)",
        ]
        for pattern in financial_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    amount = int(matches[0].replace(",", ""))
                    enhanced_impact["financial_impact"] = max(
                        enhanced_impact.get("financial_impact", 0), amount
                    )
                except (ValueError, IndexError):
                    continue

        # Add LLM insights
        enhanced_impact["llm_analysis"] = response_text[:500]  # Store excerpt

        return enhanced_impact

    # Response generation helpers
    def _get_base_responses(
        self, event_type: SecurityEventType, threat_level: ThreatLevel
    ) -> List[ResponseType]:
        """Get base response types for event and threat level."""

        base_responses = [
            ResponseType.ALERT_SECURITY_TEAM,
            ResponseType.ENABLE_MONITORING,
        ]

        # Add responses based on event type
        event_responses = {
            SecurityEventType.MALWARE_DETECTION: [
                ResponseType.QUARANTINE,
                ResponseType.BACKUP_SYSTEM,
            ],
            SecurityEventType.INTRUSION_ATTEMPT: [
                ResponseType.BLOCK_IP,
                ResponseType.NETWORK_SEGMENTATION,
            ],
            SecurityEventType.DATA_EXFILTRATION: [
                ResponseType.ISOLATE_SYSTEM,
                ResponseType.LOG_RETENTION_INCREASE,
            ],
            SecurityEventType.VULNERABILITY_EXPLOIT: [
                ResponseType.VIRTUAL_PATCH,
                ResponseType.PATCH_VULNERABILITY,
            ],
            SecurityEventType.RANSOMWARE_ACTIVITY: [
                ResponseType.ISOLATE_SYSTEM,
                ResponseType.BACKUP_SYSTEM,
            ],
            SecurityEventType.INSIDER_THREAT: [
                ResponseType.RESET_CREDENTIALS,
                ResponseType.LOG_RETENTION_INCREASE,
            ],
        }

        base_responses.extend(event_responses.get(event_type, []))

        # Add responses based on threat level
        if threat_level in {ThreatLevel.CRITICAL, ThreatLevel.HIGH}:
            base_responses.extend(
                [ResponseType.ESCALATE_TO_HUMAN, ResponseType.DEPLOY_HONEYPOT]
            )

        return list(set(base_responses))  # Remove duplicates

    def _estimate_effectiveness(
        self, response_type: ResponseType, threat_analysis: ThreatAnalysis
    ) -> float:
        """Estimate effectiveness of response type for the threat."""

        # Base effectiveness scores
        base_effectiveness = {
            ResponseType.ALERT_SECURITY_TEAM: 0.6,
            ResponseType.BLOCK_IP: 0.7,
            ResponseType.QUARANTINE: 0.8,
            ResponseType.ISOLATE_SYSTEM: 0.9,
            ResponseType.VIRTUAL_PATCH: 0.8,
            ResponseType.ENABLE_MONITORING: 0.5,
            ResponseType.RESET_CREDENTIALS: 0.7,
            ResponseType.BACKUP_SYSTEM: 0.4,
        }

        effectiveness = base_effectiveness.get(response_type, 0.5)

        # Adjust based on threat characteristics
        if threat_analysis.threat_level == ThreatLevel.CRITICAL:
            if response_type in {ResponseType.ISOLATE_SYSTEM, ResponseType.QUARANTINE}:
                effectiveness += 0.1

        if "network" in threat_analysis.ttps:
            if response_type == ResponseType.BLOCK_IP:
                effectiveness += 0.2

        return min(1.0, effectiveness)

    def _assess_response_business_impact(
        self, response_type: ResponseType, business_impact: Dict[str, Any]
    ) -> str:
        """Assess business impact of the response action."""

        impact_levels = {
            ResponseType.ALERT_SECURITY_TEAM: "minimal",
            ResponseType.ENABLE_MONITORING: "minimal",
            ResponseType.BLOCK_IP: "low",
            ResponseType.QUARANTINE: "medium",
            ResponseType.ISOLATE_SYSTEM: "high",
            ResponseType.SHUTDOWN_SERVICE: "critical",
        }

        return impact_levels.get(response_type, "medium")

    def _assess_legacy_considerations(
        self, response_type: ResponseType, legacy_systems: Optional[List[LegacySystem]]
    ) -> Dict[str, Any]:
        """Assess legacy system considerations for the response."""

        considerations = {
            "compatibility": "unknown",
            "risk_factors": [],
            "special_requirements": [],
        }

        if not legacy_systems:
            return considerations

        # Check for protocol-specific considerations
        protocols = set()
        for system in legacy_systems:
            protocols.update([p.value for p in system.protocol_types])

        if protocols:
            considerations["protocols"] = list(protocols)

            # Add protocol-specific risks
            if "modbus" in protocols and response_type == ResponseType.QUARANTINE:
                considerations["risk_factors"].append(
                    "MODBUS quarantine may disrupt industrial processes"
                )

            if "hl7_mllp" in protocols and response_type == ResponseType.ISOLATE_SYSTEM:
                considerations["risk_factors"].append(
                    "HL7 isolation may impact patient care systems"
                )

        return considerations

    def _fallback_decision(
        self,
        threat_analysis: ThreatAnalysis,
        business_impact: Dict[str, Any],
        response_options: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create fallback rule-based decision when LLM fails."""

        # Simple rule-based fallback
        actions = ["alert_security_team"]
        confidence = 0.4
        escalation_needed = True

        # Add monitoring for all cases
        actions.append("enable_monitoring")

        # Add containment for high threats
        if threat_analysis.threat_level in {ThreatLevel.CRITICAL, ThreatLevel.HIGH}:
            if business_impact.get("operational_impact") != "critical":
                actions.append("quarantine")
                confidence = 0.6

        return {
            "actions": actions,
            "confidence": confidence,
            "reasoning": "Fallback rule-based decision due to LLM failure",
            "risk_assessment": {"overall_risk": "medium"},
            "escalation_needed": escalation_needed,
            "auto_execute": False,
        }

    def _get_legacy_considerations(
        self, response_type: ResponseType, legacy_systems: List[LegacySystem]
    ) -> Dict[str, Any]:
        """Get detailed legacy system considerations."""

        considerations = {
            "affected_protocols": [],
            "criticality_levels": [],
            "dependency_risks": [],
            "maintenance_windows": [],
        }

        for system in legacy_systems:
            considerations["affected_protocols"].extend(
                [p.value for p in system.protocol_types]
            )
            considerations["criticality_levels"].append(system.criticality.value)
            considerations["dependency_risks"].extend(system.dependent_systems)
            considerations["maintenance_windows"].extend(system.maintenance_windows)

        # Remove duplicates
        for key in considerations:
            if isinstance(considerations[key], list):
                considerations[key] = list(set(considerations[key]))

        return considerations

    async def get_decision_metrics(self) -> Dict[str, Any]:
        """Get decision engine metrics and statistics."""

        return {
            "total_decisions": len(self.decision_history),
            "auto_executions": sum(
                1 for d in self.decision_history if d.get("auto_execute")
            ),
            "human_escalations": sum(
                1 for d in self.decision_history if d.get("escalation_needed")
            ),
            "average_confidence": sum(
                d.get("confidence", 0) for d in self.decision_history
            )
            / max(len(self.decision_history), 1),
            "success_rates": self.success_rates.copy(),
            "confidence_thresholds": self.confidence_thresholds.copy(),
        }

    async def update_decision_outcome(
        self, response_id: str, success: bool, effectiveness_score: float
    ):
        """Update decision outcome for learning and improvement."""

        # Find the decision in history
        for decision in self.decision_history:
            if decision.get("response_id") == response_id:
                decision["success"] = success
                decision["effectiveness_score"] = effectiveness_score
                decision["outcome_updated_at"] = time.time()

                # Update success rates
                decision_type = decision.get("threat_type", "unknown")
                if decision_type not in self.success_rates:
                    self.success_rates[decision_type] = []

                self.success_rates[decision_type].append(effectiveness_score)

                # Keep only recent success rates (last 100 decisions per type)
                if len(self.success_rates[decision_type]) > 100:
                    self.success_rates[decision_type] = self.success_rates[
                        decision_type
                    ][-100:]

                break

        self.logger.info(
            f"Updated decision outcome for {response_id}: success={success}, effectiveness={effectiveness_score}"
        )

    async def shutdown(self):
        """Shutdown the decision engine."""
        self.logger.info("Shutting down Zero-Touch Decision Engine...")
        self._initialized = False
