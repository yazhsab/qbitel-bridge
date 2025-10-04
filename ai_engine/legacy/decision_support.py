"""
CRONOS AI Engine - Legacy System Decision Support

Intelligent decision support system for legacy system management.
Provides recommendations, impact assessment, and action planning
using AI-powered analysis and domain expertise.
"""

import logging
import asyncio
import time
import uuid
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest
from ..core.config import Config
from ..core.exceptions import CronosAIException
from ..monitoring.metrics import AIEngineMetrics

from .models import (
    SystemFailurePrediction,
    MaintenanceRecommendation,
    SystemBehaviorPattern,
    LegacySystemContext,
    FormalizedKnowledge,
    FailureType,
    SeverityLevel,
    MaintenanceType,
)


class DecisionCategory(str, Enum):
    """Categories of decisions for legacy systems."""

    MAINTENANCE_PLANNING = "maintenance_planning"
    CAPACITY_MANAGEMENT = "capacity_management"
    RISK_MITIGATION = "risk_mitigation"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_ENHANCEMENT = "security_enhancement"
    TECHNOLOGY_UPGRADE = "technology_upgrade"
    BUSINESS_CONTINUITY = "business_continuity"
    COST_OPTIMIZATION = "cost_optimization"


class ActionType(str, Enum):
    """Types of recommended actions."""

    IMMEDIATE_ACTION = "immediate_action"
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"
    MONITORING_ADJUSTMENT = "monitoring_adjustment"
    PROCESS_IMPROVEMENT = "process_improvement"
    TRAINING_REQUIREMENT = "training_requirement"
    DOCUMENTATION_UPDATE = "documentation_update"
    TECHNOLOGY_EVALUATION = "technology_evaluation"
    POLICY_CHANGE = "policy_change"


class ImpactDimension(str, Enum):
    """Dimensions of business impact assessment."""

    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"
    SECURITY = "security"
    REPUTATION = "reputation"
    CUSTOMER = "customer"
    STRATEGIC = "strategic"


@dataclass
class DecisionContext:
    """Context for decision making."""

    decision_id: str
    decision_category: DecisionCategory
    system_context: Optional[LegacySystemContext]
    current_situation: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    objectives: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    timeline: Optional[str] = None
    budget_constraints: Optional[float] = None
    risk_tolerance: str = "medium"  # low, medium, high
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BusinessImpactAssessment:
    """Business impact assessment result."""

    assessment_id: str
    decision_context: DecisionContext
    impact_dimensions: Dict[ImpactDimension, Dict[str, Any]]
    overall_impact_score: float  # 0-10 scale
    risk_level: SeverityLevel
    affected_stakeholders: List[str]
    quantified_impacts: Dict[str, float]
    qualitative_impacts: Dict[str, str]
    mitigation_strategies: List[str]
    success_metrics: List[str]
    assessment_confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionRecommendation:
    """Action recommendation with detailed planning."""

    recommendation_id: str
    action_type: ActionType
    title: str
    description: str
    rationale: str
    priority: SeverityLevel
    estimated_effort: str  # "low", "medium", "high"
    estimated_duration: str
    estimated_cost: Optional[float]
    required_skills: List[str]
    prerequisites: List[str]
    expected_outcomes: List[str]
    success_criteria: List[str]
    risks: List[str]
    mitigation_steps: List[str]
    implementation_steps: List[Dict[str, Any]]
    monitoring_plan: Dict[str, Any]
    rollback_plan: Optional[str]
    decision_context: Optional[DecisionContext]
    business_impact: Optional[BusinessImpactAssessment]
    confidence_score: float
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionRecommendation:
    """Complete decision recommendation."""

    decision_id: str
    decision_context: DecisionContext
    recommended_actions: List[ActionRecommendation]
    business_impact: BusinessImpactAssessment
    implementation_roadmap: Dict[str, Any]
    alternative_options: List[Dict[str, Any]]
    decision_rationale: str
    confidence_level: float
    review_date: Optional[datetime]
    approval_required: bool
    estimated_timeline: str
    total_estimated_cost: float
    roi_projection: Optional[Dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecommendationEngine:
    """
    AI-powered recommendation engine for legacy system management.

    Generates intelligent recommendations based on system analysis,
    historical data, expert knowledge, and business context.
    """

    def __init__(self, config: Config, llm_service: Optional[UnifiedLLMService] = None):
        """Initialize recommendation engine."""
        self.config = config
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)

        # Recommendation templates and knowledge base
        self.recommendation_templates = {
            DecisionCategory.MAINTENANCE_PLANNING: {
                "context_factors": [
                    "system_age",
                    "failure_history",
                    "performance_trends",
                    "vendor_support",
                    "criticality",
                    "maintenance_costs",
                ],
                "recommendation_types": [
                    "preventive_maintenance",
                    "predictive_maintenance",
                    "condition_based_maintenance",
                    "emergency_response",
                ],
                "evaluation_criteria": [
                    "cost_effectiveness",
                    "risk_reduction",
                    "availability_impact",
                    "resource_requirements",
                    "timeline_feasibility",
                ],
            },
            DecisionCategory.RISK_MITIGATION: {
                "context_factors": [
                    "vulnerability_assessment",
                    "threat_landscape",
                    "business_impact",
                    "current_controls",
                    "regulatory_requirements",
                ],
                "recommendation_types": [
                    "security_hardening",
                    "backup_improvement",
                    "monitoring_enhancement",
                    "incident_response",
                    "business_continuity",
                ],
                "evaluation_criteria": [
                    "risk_reduction",
                    "implementation_complexity",
                    "cost_benefit",
                    "regulatory_compliance",
                    "stakeholder_acceptance",
                ],
            },
            DecisionCategory.PERFORMANCE_OPTIMIZATION: {
                "context_factors": [
                    "performance_metrics",
                    "bottleneck_analysis",
                    "capacity_trends",
                    "user_satisfaction",
                    "sla_compliance",
                ],
                "recommendation_types": [
                    "configuration_tuning",
                    "hardware_upgrade",
                    "software_optimization",
                    "architecture_changes",
                    "workload_balancing",
                ],
                "evaluation_criteria": [
                    "performance_improvement",
                    "cost_effectiveness",
                    "implementation_risk",
                    "maintenance_impact",
                    "scalability",
                ],
            },
        }

        # Decision-making prompts for LLM
        self.decision_prompts = {
            "generate_recommendations": """
            You are an expert in legacy system management providing strategic recommendations.
            
            Decision Context:
            - Category: {decision_category}
            - System: {system_name} ({system_type})
            - Current Situation: {current_situation}
            - Constraints: {constraints}
            - Objectives: {objectives}
            - Risk Tolerance: {risk_tolerance}
            
            System Analysis:
            {system_analysis}
            
            Historical Context:
            {historical_context}
            
            Expert Knowledge:
            {expert_knowledge}
            
            Generate comprehensive recommendations addressing:
            
            1. IMMEDIATE ACTIONS (0-24 hours):
               - Critical issues requiring immediate attention
               - Emergency response procedures
               - Risk mitigation steps
            
            2. SHORT-TERM ACTIONS (1-30 days):
               - Maintenance and optimization tasks
               - Performance improvements
               - Process adjustments
            
            3. MEDIUM-TERM STRATEGIES (1-6 months):
               - Planned maintenance cycles
               - Capacity planning initiatives
               - Technology evaluations
            
            4. LONG-TERM PLANNING (6+ months):
               - Strategic technology decisions
               - Legacy system evolution
               - Business alignment initiatives
            
            For each recommendation, provide:
            - Clear rationale and expected outcomes
            - Implementation complexity and timeline
            - Resource requirements and costs
            - Success criteria and monitoring approach
            - Risk assessment and mitigation strategies
            
            Focus on practical, actionable recommendations that balance risk, cost, and business value.
            """,
            "evaluate_alternatives": """
            You are evaluating alternative approaches for a legacy system decision.
            
            Decision Context:
            {decision_context}
            
            Alternative Options:
            {alternatives}
            
            Evaluation Criteria:
            {evaluation_criteria}
            
            Provide a comparative analysis of the alternatives:
            
            1. OPTION COMPARISON:
               - Strengths and weaknesses of each option
               - Suitability for the specific context
               - Risk-benefit analysis
            
            2. QUANTITATIVE ANALYSIS:
               - Cost comparison and ROI projections
               - Timeline and resource requirements
               - Performance and reliability impacts
            
            3. QUALITATIVE FACTORS:
               - Strategic alignment with business goals
               - Technical feasibility and complexity
               - Organizational readiness and change impact
            
            4. RECOMMENDATION RANKING:
               - Recommended priority order with justification
               - Conditions that would change the ranking
               - Hybrid approaches or combinations to consider
            
            Provide a clear recommendation with supporting rationale.
            """,
            "risk_benefit_analysis": """
            You are conducting a risk-benefit analysis for a legacy system decision.
            
            Proposed Action:
            {proposed_action}
            
            System Context:
            {system_context}
            
            Business Context:
            {business_context}
            
            Analyze the risks and benefits:
            
            1. BENEFIT ANALYSIS:
               - Quantifiable benefits (cost savings, performance gains, etc.)
               - Qualitative benefits (risk reduction, compliance, etc.)
               - Timeline for benefit realization
               - Sustainability of benefits
            
            2. RISK ANALYSIS:
               - Implementation risks and likelihood
               - Operational risks and impact
               - Business continuity risks
               - Long-term strategic risks
            
            3. RISK MITIGATION:
               - Strategies to reduce implementation risks
               - Contingency plans for major risks
               - Monitoring and early warning systems
               - Rollback and recovery procedures
            
            4. NET ASSESSMENT:
               - Overall risk-benefit ratio
               - Break-even analysis and ROI timeline
               - Sensitivity analysis for key assumptions
               - Recommendation with confidence level
            
            Provide actionable insights for decision makers.
            """,
        }

        # Historical decision tracking
        self.decision_history: List[DecisionRecommendation] = []
        self.success_metrics: Dict[str, float] = {}

        self.logger.info("RecommendationEngine initialized")

    async def generate_recommendations(
        self,
        decision_context: DecisionContext,
        system_analysis: Optional[Dict[str, Any]] = None,
        historical_context: Optional[Dict[str, Any]] = None,
        expert_knowledge: Optional[List[FormalizedKnowledge]] = None,
    ) -> DecisionRecommendation:
        """
        Generate comprehensive recommendations for a decision context.

        Args:
            decision_context: Context and requirements for the decision
            system_analysis: Current system analysis results
            historical_context: Historical data and trends
            expert_knowledge: Relevant expert knowledge

        Returns:
            Complete decision recommendation with actions and impact assessment
        """

        start_time = time.time()

        try:
            # Prepare analysis context
            analysis_context = self._prepare_analysis_context(
                decision_context, system_analysis, historical_context, expert_knowledge
            )

            # Generate LLM-powered recommendations
            llm_recommendations = {}
            if self.llm_service:
                llm_recommendations = await self._generate_llm_recommendations(
                    decision_context, analysis_context
                )

            # Apply rule-based logic for specific scenarios
            rule_based_recommendations = self._apply_rule_based_recommendations(
                decision_context, analysis_context
            )

            # Combine and prioritize recommendations
            combined_recommendations = self._combine_recommendations(
                llm_recommendations, rule_based_recommendations, decision_context
            )

            # Generate action recommendations
            action_recommendations = await self._generate_action_recommendations(
                combined_recommendations, decision_context, analysis_context
            )

            # Create implementation roadmap
            roadmap = self._create_implementation_roadmap(
                action_recommendations, decision_context
            )

            # Calculate costs and timeline
            total_cost, timeline = self._calculate_implementation_metrics(
                action_recommendations, roadmap
            )

            # Generate alternatives
            alternatives = await self._generate_alternatives(
                decision_context, analysis_context, action_recommendations
            )

            # Create decision recommendation
            decision_recommendation = DecisionRecommendation(
                decision_id=decision_context.decision_id,
                decision_context=decision_context,
                recommended_actions=action_recommendations,
                business_impact=BusinessImpactAssessment(
                    assessment_id=str(uuid.uuid4()),
                    decision_context=decision_context,
                    impact_dimensions={},  # Will be populated by ImpactAssessor
                    overall_impact_score=0.0,
                    risk_level=SeverityLevel.MEDIUM,
                    affected_stakeholders=decision_context.stakeholders,
                    quantified_impacts={},
                    qualitative_impacts={},
                    mitigation_strategies=[],
                    success_metrics=[],
                    assessment_confidence=0.0,
                ),
                implementation_roadmap=roadmap,
                alternative_options=alternatives,
                decision_rationale=self._generate_decision_rationale(
                    decision_context, action_recommendations, analysis_context
                ),
                confidence_level=self._calculate_recommendation_confidence(
                    llm_recommendations, rule_based_recommendations, analysis_context
                ),
                approval_required=self._requires_approval(action_recommendations),
                estimated_timeline=timeline,
                total_estimated_cost=total_cost,
                metadata={
                    "processing_time_seconds": time.time() - start_time,
                    "analysis_context": analysis_context,
                    "generation_method": "hybrid_llm_rules",
                },
            )

            # Store decision for learning
            self.decision_history.append(decision_recommendation)

            self.logger.info(
                f"Generated recommendations for decision {decision_context.decision_id}"
            )

            return decision_recommendation

        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            raise CronosAIException(f"Recommendation generation failed: {e}")

    def _prepare_analysis_context(
        self,
        decision_context: DecisionContext,
        system_analysis: Optional[Dict[str, Any]],
        historical_context: Optional[Dict[str, Any]],
        expert_knowledge: Optional[List[FormalizedKnowledge]],
    ) -> Dict[str, Any]:
        """Prepare comprehensive analysis context."""

        context = {
            "decision_category": decision_context.decision_category.value,
            "system_context": decision_context.system_context,
            "current_situation": decision_context.current_situation,
            "constraints": decision_context.constraints,
            "objectives": decision_context.objectives,
            "risk_tolerance": decision_context.risk_tolerance,
            "system_analysis": system_analysis or {},
            "historical_context": historical_context or {},
            "expert_knowledge": [],
        }

        # Format expert knowledge
        if expert_knowledge:
            for knowledge in expert_knowledge[:5]:  # Limit to top 5 relevant items
                context["expert_knowledge"].append(
                    {
                        "title": knowledge.title,
                        "description": knowledge.description,
                        "recommended_actions": knowledge.recommended_actions,
                        "troubleshooting_steps": knowledge.troubleshooting_steps,
                        "confidence": knowledge.confidence_score,
                    }
                )

        return context

    async def _generate_llm_recommendations(
        self, decision_context: DecisionContext, analysis_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommendations using LLM analysis."""

        if not self.llm_service:
            return {}

        try:
            # Format context for LLM
            system_name = "Unknown System"
            system_type = "unknown"
            if decision_context.system_context:
                system_name = decision_context.system_context.system_name
                system_type = decision_context.system_context.system_type.value

            prompt = self.decision_prompts["generate_recommendations"].format(
                decision_category=decision_context.decision_category.value,
                system_name=system_name,
                system_type=system_type,
                current_situation=json.dumps(
                    decision_context.current_situation, indent=2
                ),
                constraints=json.dumps(decision_context.constraints, indent=2),
                objectives=", ".join(decision_context.objectives),
                risk_tolerance=decision_context.risk_tolerance,
                system_analysis=json.dumps(
                    analysis_context.get("system_analysis", {}), indent=2
                ),
                historical_context=json.dumps(
                    analysis_context.get("historical_context", {}), indent=2
                ),
                expert_knowledge=json.dumps(
                    analysis_context.get("expert_knowledge", []), indent=2
                ),
            )

            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="legacy_whisperer",
                max_tokens=3000,
                temperature=0.2,
            )

            response = await self.llm_service.process_request(llm_request)

            # Parse LLM response
            parsed_recommendations = self._parse_llm_recommendations(response.content)
            parsed_recommendations["llm_confidence"] = response.confidence
            parsed_recommendations["processing_time"] = response.processing_time

            return parsed_recommendations

        except Exception as e:
            self.logger.error(f"LLM recommendation generation failed: {e}")
            return {"error": str(e)}

    def _parse_llm_recommendations(self, content: str) -> Dict[str, Any]:
        """Parse LLM recommendations into structured format."""

        recommendations = {
            "immediate_actions": [],
            "short_term_actions": [],
            "medium_term_strategies": [],
            "long_term_planning": [],
            "overall_assessment": "",
        }

        lines = content.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect sections
            if "immediate actions" in line.lower():
                current_section = "immediate_actions"
            elif "short-term actions" in line.lower():
                current_section = "short_term_actions"
            elif "medium-term strategies" in line.lower():
                current_section = "medium_term_strategies"
            elif "long-term planning" in line.lower():
                current_section = "long_term_planning"
            elif current_section and (
                line.startswith("-") or line.startswith("•") or line.startswith("*")
            ):
                # Extract recommendation item
                item = line.lstrip("-•* ").strip()
                if item and current_section in recommendations:
                    recommendations[current_section].append(item)

        # Extract overall assessment
        paragraphs = content.split("\n\n")
        if paragraphs and len(paragraphs[0]) > 100:
            recommendations["overall_assessment"] = paragraphs[0]

        return recommendations

    def _apply_rule_based_recommendations(
        self, decision_context: DecisionContext, analysis_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply rule-based logic for specific scenarios."""

        recommendations = {
            "critical_issues": [],
            "maintenance_recommendations": [],
            "optimization_opportunities": [],
            "risk_mitigation": [],
        }

        # Analyze current situation for rule triggers
        current_situation = decision_context.current_situation
        system_analysis = analysis_context.get("system_analysis", {})

        # Critical issue detection rules
        if current_situation.get("anomaly_detected", False):
            anomaly_score = current_situation.get("anomaly_score", 0)
            if anomaly_score > 0.8:
                recommendations["critical_issues"].append(
                    {
                        "type": "high_anomaly",
                        "description": "High anomaly score detected - immediate investigation required",
                        "priority": "critical",
                        "action": "Conduct immediate system health check and anomaly root cause analysis",
                    }
                )

        # Performance-based rules
        if current_situation.get("performance_score", 100) < 70:
            recommendations["optimization_opportunities"].append(
                {
                    "type": "performance_degradation",
                    "description": "System performance below acceptable threshold",
                    "priority": "high",
                    "action": "Perform performance analysis and optimization",
                }
            )

        # Maintenance rules
        system_age_years = 0
        if (
            decision_context.system_context
            and decision_context.system_context.installation_date
        ):
            system_age_years = (
                datetime.now() - decision_context.system_context.installation_date
            ).days / 365.25

        if system_age_years > 10:
            recommendations["maintenance_recommendations"].append(
                {
                    "type": "aging_system",
                    "description": f"System is {system_age_years:.1f} years old - enhanced maintenance required",
                    "priority": "medium",
                    "action": "Implement proactive maintenance and modernization planning",
                }
            )

        # Risk-based rules
        if (
            decision_context.risk_tolerance == "low"
            and current_situation.get("failure_probability", 0) > 0.3
        ):
            recommendations["risk_mitigation"].append(
                {
                    "type": "failure_risk",
                    "description": "Failure probability exceeds risk tolerance",
                    "priority": "high",
                    "action": "Implement additional monitoring and backup procedures",
                }
            )

        return recommendations

    def _combine_recommendations(
        self,
        llm_recommendations: Dict[str, Any],
        rule_based_recommendations: Dict[str, Any],
        decision_context: DecisionContext,
    ) -> Dict[str, Any]:
        """Combine LLM and rule-based recommendations."""

        combined = {
            "prioritized_recommendations": [],
            "confidence_scores": {},
            "source_attribution": {},
        }

        # Process LLM recommendations
        for category, items in llm_recommendations.items():
            if isinstance(items, list):
                for item in items:
                    combined["prioritized_recommendations"].append(
                        {
                            "content": item,
                            "source": "llm",
                            "category": category,
                            "confidence": llm_recommendations.get(
                                "llm_confidence", 0.7
                            ),
                        }
                    )

        # Process rule-based recommendations
        for category, items in rule_based_recommendations.items():
            for item in items:
                if isinstance(item, dict):
                    combined["prioritized_recommendations"].append(
                        {
                            "content": item,
                            "source": "rules",
                            "category": category,
                            "confidence": 0.9,  # High confidence for rule-based
                        }
                    )

        # Sort by priority and confidence
        def recommendation_score(rec):
            priority_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            priority = rec.get("content", {}).get("priority", "medium")
            priority_weight = priority_weights.get(priority, 2)
            confidence = rec.get("confidence", 0.5)
            return priority_weight * confidence

        combined["prioritized_recommendations"].sort(
            key=recommendation_score, reverse=True
        )

        return combined

    async def _generate_action_recommendations(
        self,
        combined_recommendations: Dict[str, Any],
        decision_context: DecisionContext,
        analysis_context: Dict[str, Any],
    ) -> List[ActionRecommendation]:
        """Generate detailed action recommendations."""

        action_recommendations = []

        for i, rec in enumerate(
            combined_recommendations["prioritized_recommendations"][:10]
        ):  # Top 10
            # Determine action type
            action_type = self._determine_action_type(rec, decision_context)

            # Extract details from recommendation
            content = rec.get("content", {})
            if isinstance(content, str):
                title = content[:100] + "..." if len(content) > 100 else content
                description = content
                rationale = f"Generated from {rec.get('source', 'analysis')}"
            else:
                title = content.get("description", f"Action {i+1}")
                description = content.get("description", "")
                rationale = f"Based on {content.get('type', 'analysis')}"

            # Create action recommendation
            action_rec = ActionRecommendation(
                recommendation_id=str(uuid.uuid4()),
                action_type=action_type,
                title=title,
                description=description,
                rationale=rationale,
                priority=self._convert_priority(content.get("priority", "medium")),
                estimated_effort=self._estimate_effort(content, action_type),
                estimated_duration=self._estimate_duration(content, action_type),
                estimated_cost=self._estimate_cost(
                    content, action_type, decision_context
                ),
                required_skills=self._determine_required_skills(content, action_type),
                prerequisites=self._determine_prerequisites(content, action_type),
                expected_outcomes=self._determine_expected_outcomes(
                    content, action_type
                ),
                success_criteria=self._determine_success_criteria(content, action_type),
                risks=self._identify_risks(content, action_type),
                mitigation_steps=self._generate_mitigation_steps(content, action_type),
                implementation_steps=self._generate_implementation_steps(
                    content, action_type
                ),
                monitoring_plan=self._generate_monitoring_plan(content, action_type),
                rollback_plan=self._generate_rollback_plan(content, action_type),
                decision_context=decision_context,
                confidence_score=rec.get("confidence", 0.5),
            )

            action_recommendations.append(action_rec)

        return action_recommendations

    def _determine_action_type(
        self, recommendation: Dict[str, Any], decision_context: DecisionContext
    ) -> ActionType:
        """Determine the type of action based on recommendation content."""

        content = recommendation.get("content", {})
        category = recommendation.get("category", "")

        if "immediate" in category.lower():
            return ActionType.IMMEDIATE_ACTION
        elif "maintenance" in str(content).lower():
            return ActionType.SCHEDULED_MAINTENANCE
        elif "monitor" in str(content).lower():
            return ActionType.MONITORING_ADJUSTMENT
        elif "process" in str(content).lower():
            return ActionType.PROCESS_IMPROVEMENT
        elif "train" in str(content).lower():
            return ActionType.TRAINING_REQUIREMENT
        elif "document" in str(content).lower():
            return ActionType.DOCUMENTATION_UPDATE
        elif "evaluat" in str(content).lower():
            return ActionType.TECHNOLOGY_EVALUATION
        elif "policy" in str(content).lower():
            return ActionType.POLICY_CHANGE
        else:
            return ActionType.PROCESS_IMPROVEMENT

    def _convert_priority(self, priority_str: str) -> SeverityLevel:
        """Convert string priority to SeverityLevel."""
        priority_map = {
            "critical": SeverityLevel.CRITICAL,
            "high": SeverityLevel.HIGH,
            "medium": SeverityLevel.MEDIUM,
            "low": SeverityLevel.LOW,
            "info": SeverityLevel.INFO,
        }
        return priority_map.get(priority_str.lower(), SeverityLevel.MEDIUM)

    def _estimate_effort(self, content: Any, action_type: ActionType) -> str:
        """Estimate effort level for an action."""

        # Simple heuristics based on action type
        effort_map = {
            ActionType.IMMEDIATE_ACTION: "low",
            ActionType.MONITORING_ADJUSTMENT: "low",
            ActionType.DOCUMENTATION_UPDATE: "low",
            ActionType.PROCESS_IMPROVEMENT: "medium",
            ActionType.TRAINING_REQUIREMENT: "medium",
            ActionType.SCHEDULED_MAINTENANCE: "medium",
            ActionType.TECHNOLOGY_EVALUATION: "high",
            ActionType.POLICY_CHANGE: "high",
        }

        return effort_map.get(action_type, "medium")

    def _estimate_duration(self, content: Any, action_type: ActionType) -> str:
        """Estimate duration for an action."""

        duration_map = {
            ActionType.IMMEDIATE_ACTION: "1-4 hours",
            ActionType.MONITORING_ADJUSTMENT: "1-2 days",
            ActionType.DOCUMENTATION_UPDATE: "3-5 days",
            ActionType.PROCESS_IMPROVEMENT: "1-2 weeks",
            ActionType.TRAINING_REQUIREMENT: "2-4 weeks",
            ActionType.SCHEDULED_MAINTENANCE: "4-8 hours",
            ActionType.TECHNOLOGY_EVALUATION: "4-8 weeks",
            ActionType.POLICY_CHANGE: "6-12 weeks",
        }

        return duration_map.get(action_type, "1-2 weeks")

    def _estimate_cost(
        self, content: Any, action_type: ActionType, decision_context: DecisionContext
    ) -> Optional[float]:
        """Estimate cost for an action."""

        # Base costs by action type (in USD)
        base_costs = {
            ActionType.IMMEDIATE_ACTION: 1000,
            ActionType.MONITORING_ADJUSTMENT: 2000,
            ActionType.DOCUMENTATION_UPDATE: 3000,
            ActionType.PROCESS_IMPROVEMENT: 10000,
            ActionType.TRAINING_REQUIREMENT: 15000,
            ActionType.SCHEDULED_MAINTENANCE: 5000,
            ActionType.TECHNOLOGY_EVALUATION: 25000,
            ActionType.POLICY_CHANGE: 20000,
        }

        base_cost = base_costs.get(action_type, 10000)

        # Adjust for system complexity
        if decision_context.system_context:
            if decision_context.system_context.criticality == SeverityLevel.CRITICAL:
                base_cost *= 1.5
            elif decision_context.system_context.criticality == SeverityLevel.HIGH:
                base_cost *= 1.2

        return base_cost

    def _determine_required_skills(
        self, content: Any, action_type: ActionType
    ) -> List[str]:
        """Determine required skills for an action."""

        skill_map = {
            ActionType.IMMEDIATE_ACTION: ["system_administration", "troubleshooting"],
            ActionType.MONITORING_ADJUSTMENT: ["monitoring_tools", "data_analysis"],
            ActionType.DOCUMENTATION_UPDATE: ["technical_writing", "system_knowledge"],
            ActionType.PROCESS_IMPROVEMENT: ["process_analysis", "change_management"],
            ActionType.TRAINING_REQUIREMENT: [
                "training_development",
                "domain_expertise",
            ],
            ActionType.SCHEDULED_MAINTENANCE: [
                "maintenance_procedures",
                "system_expertise",
            ],
            ActionType.TECHNOLOGY_EVALUATION: [
                "technology_assessment",
                "architecture_design",
            ],
            ActionType.POLICY_CHANGE: ["policy_development", "stakeholder_management"],
        }

        return skill_map.get(action_type, ["domain_expertise"])

    def _determine_prerequisites(
        self, content: Any, action_type: ActionType
    ) -> List[str]:
        """Determine prerequisites for an action."""

        prereq_map = {
            ActionType.IMMEDIATE_ACTION: ["system_access", "emergency_procedures"],
            ActionType.MONITORING_ADJUSTMENT: [
                "monitoring_system_access",
                "baseline_data",
            ],
            ActionType.DOCUMENTATION_UPDATE: [
                "current_documentation",
                "system_specifications",
            ],
            ActionType.PROCESS_IMPROVEMENT: [
                "current_process_documentation",
                "stakeholder_buy_in",
            ],
            ActionType.TRAINING_REQUIREMENT: [
                "training_budget",
                "subject_matter_experts",
            ],
            ActionType.SCHEDULED_MAINTENANCE: [
                "maintenance_window",
                "backup_procedures",
            ],
            ActionType.TECHNOLOGY_EVALUATION: [
                "evaluation_criteria",
                "vendor_information",
            ],
            ActionType.POLICY_CHANGE: ["current_policies", "legal_review"],
        }

        return prereq_map.get(action_type, ["stakeholder_approval"])

    def _determine_expected_outcomes(
        self, content: Any, action_type: ActionType
    ) -> List[str]:
        """Determine expected outcomes for an action."""

        outcome_map = {
            ActionType.IMMEDIATE_ACTION: ["issue_resolution", "system_stability"],
            ActionType.MONITORING_ADJUSTMENT: [
                "improved_visibility",
                "early_warning_capability",
            ],
            ActionType.DOCUMENTATION_UPDATE: [
                "accurate_documentation",
                "knowledge_preservation",
            ],
            ActionType.PROCESS_IMPROVEMENT: ["increased_efficiency", "reduced_errors"],
            ActionType.TRAINING_REQUIREMENT: [
                "improved_skills",
                "reduced_knowledge_gaps",
            ],
            ActionType.SCHEDULED_MAINTENANCE: [
                "improved_reliability",
                "extended_system_life",
            ],
            ActionType.TECHNOLOGY_EVALUATION: [
                "informed_decisions",
                "technology_roadmap",
            ],
            ActionType.POLICY_CHANGE: [
                "improved_compliance",
                "standardized_procedures",
            ],
        }

        return outcome_map.get(action_type, ["improved_system_performance"])

    def _determine_success_criteria(
        self, content: Any, action_type: ActionType
    ) -> List[str]:
        """Determine success criteria for an action."""

        criteria_map = {
            ActionType.IMMEDIATE_ACTION: [
                "issue_resolved_within_sla",
                "no_recurring_problems",
            ],
            ActionType.MONITORING_ADJUSTMENT: [
                "monitoring_coverage_improved",
                "alert_accuracy_increased",
            ],
            ActionType.DOCUMENTATION_UPDATE: [
                "documentation_completeness_>90%",
                "user_satisfaction_improved",
            ],
            ActionType.PROCESS_IMPROVEMENT: [
                "process_efficiency_increased_20%",
                "error_rate_reduced",
            ],
            ActionType.TRAINING_REQUIREMENT: [
                "training_completion_>95%",
                "competency_assessment_passed",
            ],
            ActionType.SCHEDULED_MAINTENANCE: [
                "maintenance_completed_on_time",
                "system_performance_maintained",
            ],
            ActionType.TECHNOLOGY_EVALUATION: [
                "evaluation_report_delivered",
                "recommendation_approved",
            ],
            ActionType.POLICY_CHANGE: ["policy_published", "compliance_audit_passed"],
        }

        return criteria_map.get(action_type, ["objectives_met"])

    def _identify_risks(self, content: Any, action_type: ActionType) -> List[str]:
        """Identify risks associated with an action."""

        risk_map = {
            ActionType.IMMEDIATE_ACTION: ["system_downtime", "incomplete_resolution"],
            ActionType.MONITORING_ADJUSTMENT: ["false_alerts", "monitoring_overhead"],
            ActionType.DOCUMENTATION_UPDATE: [
                "outdated_information",
                "inconsistent_documentation",
            ],
            ActionType.PROCESS_IMPROVEMENT: ["process_disruption", "user_resistance"],
            ActionType.TRAINING_REQUIREMENT: [
                "training_ineffectiveness",
                "resource_allocation_issues",
            ],
            ActionType.SCHEDULED_MAINTENANCE: [
                "extended_downtime",
                "maintenance_complications",
            ],
            ActionType.TECHNOLOGY_EVALUATION: ["biased_evaluation", "vendor_lock_in"],
            ActionType.POLICY_CHANGE: ["policy_conflicts", "implementation_challenges"],
        }

        return risk_map.get(action_type, ["implementation_delays"])

    def _generate_mitigation_steps(
        self, content: Any, action_type: ActionType
    ) -> List[str]:
        """Generate risk mitigation steps."""

        mitigation_map = {
            ActionType.IMMEDIATE_ACTION: [
                "prepare_rollback_plan",
                "monitor_system_health",
            ],
            ActionType.MONITORING_ADJUSTMENT: [
                "test_alerts_before_deployment",
                "gradual_rollout",
            ],
            ActionType.DOCUMENTATION_UPDATE: ["peer_review_process", "version_control"],
            ActionType.PROCESS_IMPROVEMENT: [
                "pilot_implementation",
                "change_management_plan",
            ],
            ActionType.TRAINING_REQUIREMENT: [
                "training_effectiveness_assessment",
                "feedback_collection",
            ],
            ActionType.SCHEDULED_MAINTENANCE: [
                "comprehensive_testing",
                "backup_procedures",
            ],
            ActionType.TECHNOLOGY_EVALUATION: [
                "independent_evaluation",
                "poc_implementation",
            ],
            ActionType.POLICY_CHANGE: [
                "stakeholder_consultation",
                "phased_implementation",
            ],
        }

        return mitigation_map.get(
            action_type, ["thorough_planning", "stakeholder_communication"]
        )

    def _generate_implementation_steps(
        self, content: Any, action_type: ActionType
    ) -> List[Dict[str, Any]]:
        """Generate implementation steps for an action."""

        # Generic implementation steps based on action type
        step_templates = {
            ActionType.IMMEDIATE_ACTION: [
                {
                    "step": 1,
                    "description": "Assess current system state",
                    "duration": "30 minutes",
                },
                {
                    "step": 2,
                    "description": "Implement immediate fixes",
                    "duration": "2 hours",
                },
                {
                    "step": 3,
                    "description": "Verify resolution",
                    "duration": "30 minutes",
                },
                {"step": 4, "description": "Document changes", "duration": "1 hour"},
            ],
            ActionType.SCHEDULED_MAINTENANCE: [
                {
                    "step": 1,
                    "description": "Schedule maintenance window",
                    "duration": "1 day",
                },
                {
                    "step": 2,
                    "description": "Prepare maintenance procedures",
                    "duration": "2 days",
                },
                {
                    "step": 3,
                    "description": "Execute maintenance",
                    "duration": "4 hours",
                },
                {
                    "step": 4,
                    "description": "Validate system operation",
                    "duration": "2 hours",
                },
                {"step": 5, "description": "Update documentation", "duration": "1 day"},
            ],
        }

        return step_templates.get(
            action_type,
            [
                {"step": 1, "description": "Plan implementation", "duration": "1 week"},
                {"step": 2, "description": "Execute action", "duration": "1 week"},
                {"step": 3, "description": "Validate results", "duration": "3 days"},
            ],
        )

    def _generate_monitoring_plan(
        self, content: Any, action_type: ActionType
    ) -> Dict[str, Any]:
        """Generate monitoring plan for an action."""

        return {
            "monitoring_frequency": (
                "daily" if action_type == ActionType.IMMEDIATE_ACTION else "weekly"
            ),
            "key_metrics": [
                "system_availability",
                "performance_metrics",
                "error_rates",
            ],
            "alert_thresholds": {
                "availability": 99.0,
                "response_time": 5000,
                "error_rate": 0.05,
            },
            "review_schedule": (
                "weekly" if action_type == ActionType.IMMEDIATE_ACTION else "monthly"
            ),
            "escalation_criteria": [
                "sla_breach",
                "performance_degradation",
                "error_spike",
            ],
        }

    def _generate_rollback_plan(
        self, content: Any, action_type: ActionType
    ) -> Optional[str]:
        """Generate rollback plan for an action."""

        if action_type in [
            ActionType.IMMEDIATE_ACTION,
            ActionType.SCHEDULED_MAINTENANCE,
        ]:
            return "Restore from backup configuration and restart system components as needed"
        elif action_type == ActionType.PROCESS_IMPROVEMENT:
            return "Revert to previous process procedures and notify stakeholders"
        else:
            return None

    def _create_implementation_roadmap(
        self,
        action_recommendations: List[ActionRecommendation],
        decision_context: DecisionContext,
    ) -> Dict[str, Any]:
        """Create implementation roadmap for recommendations."""

        # Group actions by priority and timeline
        roadmap = {
            "immediate": [],  # 0-24 hours
            "short_term": [],  # 1-7 days
            "medium_term": [],  # 1-4 weeks
            "long_term": [],  # 1+ months
        }

        for action in action_recommendations:
            # Determine timeline category
            if action.action_type == ActionType.IMMEDIATE_ACTION:
                category = "immediate"
            elif action.priority in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
                category = "short_term"
            elif "week" in action.estimated_duration.lower():
                category = "medium_term"
            else:
                category = "long_term"

            roadmap[category].append(
                {
                    "recommendation_id": action.recommendation_id,
                    "title": action.title,
                    "priority": action.priority.value,
                    "estimated_duration": action.estimated_duration,
                    "estimated_cost": action.estimated_cost,
                    "dependencies": action.prerequisites,
                }
            )

        # Add roadmap metadata
        roadmap["metadata"] = {
            "total_actions": len(action_recommendations),
            "critical_path": self._identify_critical_path(action_recommendations),
            "resource_peaks": self._identify_resource_peaks(action_recommendations),
            "milestone_dates": self._generate_milestone_dates(roadmap),
        }

        return roadmap

    def _identify_critical_path(self, actions: List[ActionRecommendation]) -> List[str]:
        """Identify critical path through action dependencies."""
        # Simplified critical path identification
        critical_actions = [
            action.recommendation_id
            for action in actions
            if action.priority in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
        ]
        return critical_actions[:5]  # Top 5 critical actions

    def _identify_resource_peaks(
        self, actions: List[ActionRecommendation]
    ) -> Dict[str, Any]:
        """Identify periods of high resource demand."""
        # Simplified resource peak analysis
        return {
            "peak_periods": ["week_2", "week_4"],
            "peak_resources": ["system_administrators", "maintenance_technicians"],
            "mitigation": "Stagger non-critical activities to balance resource utilization",
        }

    def _generate_milestone_dates(self, roadmap: Dict[str, Any]) -> Dict[str, str]:
        """Generate milestone dates for the roadmap."""
        now = datetime.now()
        return {
            "immediate_complete": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
            "short_term_complete": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
            "medium_term_complete": (now + timedelta(days=30)).strftime("%Y-%m-%d"),
            "long_term_complete": (now + timedelta(days=90)).strftime("%Y-%m-%d"),
        }

    def _calculate_implementation_metrics(
        self, actions: List[ActionRecommendation], roadmap: Dict[str, Any]
    ) -> Tuple[float, str]:
        """Calculate total cost and timeline for implementation."""

        total_cost = sum(action.estimated_cost or 0 for action in actions)

        # Determine overall timeline based on longest category
        if roadmap.get("long_term"):
            timeline = "3-6 months"
        elif roadmap.get("medium_term"):
            timeline = "1-3 months"
        elif roadmap.get("short_term"):
            timeline = "1-4 weeks"
        else:
            timeline = "1-7 days"

        return total_cost, timeline

    async def _generate_alternatives(
        self,
        decision_context: DecisionContext,
        analysis_context: Dict[str, Any],
        primary_actions: List[ActionRecommendation],
    ) -> List[Dict[str, Any]]:
        """Generate alternative approaches."""

        alternatives = []

        # Generate alternative based on different risk tolerances
        if decision_context.risk_tolerance != "low":
            alternatives.append(
                {
                    "name": "Conservative Approach",
                    "description": "Minimize risk with phased implementation and extensive testing",
                    "risk_level": "low",
                    "timeline": "Extended by 50%",
                    "cost_impact": "+30%",
                    "benefits": [
                        "Lower implementation risk",
                        "Better change management",
                    ],
                }
            )

        if decision_context.risk_tolerance != "high":
            alternatives.append(
                {
                    "name": "Aggressive Approach",
                    "description": "Accelerated implementation with parallel workstreams",
                    "risk_level": "high",
                    "timeline": "Reduced by 30%",
                    "cost_impact": "+20%",
                    "benefits": ["Faster time to value", "Early problem resolution"],
                }
            )

        # Generate cost-optimized alternative
        alternatives.append(
            {
                "name": "Cost-Optimized Approach",
                "description": "Focus on high-impact, low-cost actions first",
                "risk_level": "medium",
                "timeline": "Similar",
                "cost_impact": "-40%",
                "benefits": ["Better ROI", "Reduced budget requirements"],
            }
        )

        return alternatives

    def _generate_decision_rationale(
        self,
        decision_context: DecisionContext,
        actions: List[ActionRecommendation],
        analysis_context: Dict[str, Any],
    ) -> str:
        """Generate rationale for the decision."""

        rationale_parts = []

        # Context-based rationale
        rationale_parts.append(
            f"Based on analysis of {decision_context.system_context.system_name if decision_context.system_context else 'the system'}, "
            f"the recommended approach addresses {decision_context.decision_category.value} requirements."
        )

        # Priority-based rationale
        critical_actions = sum(
            1 for a in actions if a.priority == SeverityLevel.CRITICAL
        )
        if critical_actions > 0:
            rationale_parts.append(
                f"The plan includes {critical_actions} critical actions that require immediate attention."
            )

        # Risk-based rationale
        rationale_parts.append(
            f"The approach aligns with the specified risk tolerance level of {decision_context.risk_tolerance}."
        )

        return " ".join(rationale_parts)

    def _calculate_recommendation_confidence(
        self,
        llm_recommendations: Dict[str, Any],
        rule_based_recommendations: Dict[str, Any],
        analysis_context: Dict[str, Any],
    ) -> float:
        """Calculate overall confidence in recommendations."""

        confidence_factors = []

        # LLM confidence
        if llm_recommendations.get("llm_confidence"):
            confidence_factors.append(llm_recommendations["llm_confidence"])

        # Rule-based confidence (typically high)
        if rule_based_recommendations:
            confidence_factors.append(0.9)

        # Data quality factor
        system_analysis = analysis_context.get("system_analysis", {})
        data_quality = system_analysis.get("data_quality_score", 0.7)
        confidence_factors.append(data_quality)

        # Expert knowledge factor
        expert_knowledge = analysis_context.get("expert_knowledge", [])
        if expert_knowledge:
            avg_expert_confidence = np.mean(
                [k.get("confidence", 0.5) for k in expert_knowledge]
            )
            confidence_factors.append(avg_expert_confidence)

        return np.mean(confidence_factors) if confidence_factors else 0.5

    def _requires_approval(self, actions: List[ActionRecommendation]) -> bool:
        """Determine if recommendations require approval."""

        # Require approval for high-cost or high-risk actions
        for action in actions:
            if action.estimated_cost and action.estimated_cost > 50000:
                return True
            if action.priority == SeverityLevel.CRITICAL:
                return True
            if action.action_type in [
                ActionType.POLICY_CHANGE,
                ActionType.TECHNOLOGY_EVALUATION,
            ]:
                return True

        return False

    def get_recommendation_history(
        self, decision_category: Optional[DecisionCategory] = None, days_back: int = 30
    ) -> List[DecisionRecommendation]:
        """Get historical recommendations."""

        cutoff_date = datetime.now() - timedelta(days=days_back)

        filtered_history = [
            decision
            for decision in self.decision_history
            if decision.created_at >= cutoff_date
        ]

        if decision_category:
            filtered_history = [
                decision
                for decision in filtered_history
                if decision.decision_context.decision_category == decision_category
            ]

        return sorted(filtered_history, key=lambda d: d.created_at, reverse=True)

    def update_success_metrics(
        self, decision_id: str, success_score: float, outcomes: Dict[str, Any]
    ) -> None:
        """Update success metrics for a decision."""

        self.success_metrics[decision_id] = {
            "success_score": success_score,
            "outcomes": outcomes,
            "updated_at": datetime.now().isoformat(),
        }

        self.logger.info(
            f"Updated success metrics for decision {decision_id}: {success_score}"
        )

    def get_recommendation_effectiveness(self) -> Dict[str, Any]:
        """Get overall recommendation effectiveness metrics."""

        if not self.success_metrics:
            return {"message": "No success metrics available"}

        success_scores = [
            metrics["success_score"] for metrics in self.success_metrics.values()
        ]

        return {
            "total_decisions": len(self.decision_history),
            "decisions_with_metrics": len(self.success_metrics),
            "average_success_score": np.mean(success_scores),
            "success_rate": sum(1 for score in success_scores if score > 0.7)
            / len(success_scores),
            "improvement_trend": self._calculate_improvement_trend(),
            "top_performing_categories": self._identify_top_performing_categories(),
        }

    def _calculate_improvement_trend(self) -> str:
        """Calculate improvement trend over time."""

        if len(self.success_metrics) < 5:
            return "insufficient_data"

        # Simple trend calculation
        recent_decisions = list(self.decision_history)[-10:]
        recent_scores = []

        for decision in recent_decisions:
            if decision.decision_id in self.success_metrics:
                recent_scores.append(
                    self.success_metrics[decision.decision_id]["success_score"]
                )

        if len(recent_scores) < 5:
            return "insufficient_data"

        # Compare first half vs second half
        mid_point = len(recent_scores) // 2
        early_avg = np.mean(recent_scores[:mid_point])
        late_avg = np.mean(recent_scores[mid_point:])

        if late_avg > early_avg + 0.1:
            return "improving"
        elif late_avg < early_avg - 0.1:
            return "declining"
        else:
            return "stable"

    def _identify_top_performing_categories(self) -> List[Dict[str, Any]]:
        """Identify top performing decision categories."""

        category_performance = {}

        for decision in self.decision_history:
            if decision.decision_id in self.success_metrics:
                category = decision.decision_context.decision_category
                score = self.success_metrics[decision.decision_id]["success_score"]

                if category not in category_performance:
                    category_performance[category] = []
                category_performance[category].append(score)

        # Calculate average performance by category
        category_averages = []
        for category, scores in category_performance.items():
            if len(scores) >= 3:  # At least 3 decisions
                avg_score = np.mean(scores)
                category_averages.append(
                    {
                        "category": category.value,
                        "average_score": avg_score,
                        "decision_count": len(scores),
                    }
                )

        # Sort by performance
        category_averages.sort(key=lambda c: c["average_score"], reverse=True)

        return category_averages[:5]  # Top 5 categories


class ImpactAssessor:
    """
    Business impact assessor for legacy system decisions.

    Provides comprehensive impact analysis across multiple business dimensions
    including financial, operational, and strategic considerations.
    """

    def __init__(self, config: Config, llm_service: Optional[UnifiedLLMService] = None):
        """Initialize impact assessor."""
        self.config = config
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)

        # Impact assessment templates
        self.impact_templates = {
            ImpactDimension.FINANCIAL: {
                "metrics": [
                    "cost_savings",
                    "revenue_impact",
                    "capital_expenditure",
                    "operational_expense",
                ],
                "calculation_methods": [
                    "direct_cost",
                    "opportunity_cost",
                    "risk_adjusted_value",
                ],
                "time_horizons": [
                    "immediate",
                    "short_term",
                    "medium_term",
                    "long_term",
                ],
            },
            ImpactDimension.OPERATIONAL: {
                "metrics": ["availability", "performance", "efficiency", "capacity"],
                "calculation_methods": [
                    "baseline_comparison",
                    "benchmark_analysis",
                    "simulation",
                ],
                "time_horizons": ["immediate", "short_term", "medium_term"],
            },
            ImpactDimension.REGULATORY: {
                "metrics": ["compliance_score", "audit_findings", "regulatory_risk"],
                "calculation_methods": ["compliance_gap_analysis", "risk_assessment"],
                "time_horizons": ["immediate", "regulatory_cycle"],
            },
        }

        self.logger.info("ImpactAssessor initialized")

    async def assess_business_impact(
        self,
        decision_context: DecisionContext,
        action_recommendations: List[ActionRecommendation],
        system_context: Optional[LegacySystemContext] = None,
    ) -> BusinessImpactAssessment:
        """
        Assess comprehensive business impact of recommended actions.

        Args:
            decision_context: Context for the decision
            action_recommendations: List of recommended actions
            system_context: System context information

        Returns:
            Comprehensive business impact assessment
        """

        try:
            # Assess impact across all dimensions
            impact_dimensions = {}

            for dimension in ImpactDimension:
                impact_dimensions[dimension] = await self._assess_dimension_impact(
                    dimension, decision_context, action_recommendations, system_context
                )

            # Calculate overall impact score
            overall_score = self._calculate_overall_impact_score(impact_dimensions)

            # Determine risk level
            risk_level = self._determine_risk_level(impact_dimensions, overall_score)

            # Identify affected stakeholders
            affected_stakeholders = self._identify_affected_stakeholders(
                decision_context, action_recommendations, impact_dimensions
            )

            # Quantify impacts where possible
            quantified_impacts = self._quantify_impacts(impact_dimensions)

            # Extract qualitative impacts
            qualitative_impacts = self._extract_qualitative_impacts(impact_dimensions)

            # Generate mitigation strategies
            mitigation_strategies = await self._generate_mitigation_strategies(
                impact_dimensions, decision_context
            )

            # Define success metrics
            success_metrics = self._define_success_metrics(
                decision_context, action_recommendations, impact_dimensions
            )

            # Calculate assessment confidence
            confidence = self._calculate_assessment_confidence(
                impact_dimensions, decision_context
            )

            assessment = BusinessImpactAssessment(
                assessment_id=str(uuid.uuid4()),
                decision_context=decision_context,
                impact_dimensions=impact_dimensions,
                overall_impact_score=overall_score,
                risk_level=risk_level,
                affected_stakeholders=affected_stakeholders,
                quantified_impacts=quantified_impacts,
                qualitative_impacts=qualitative_impacts,
                mitigation_strategies=mitigation_strategies,
                success_metrics=success_metrics,
                assessment_confidence=confidence,
            )

            self.logger.info(
                f"Completed business impact assessment {assessment.assessment_id}"
            )

            return assessment

        except Exception as e:
            self.logger.error(f"Business impact assessment failed: {e}")
            raise CronosAIException(f"Impact assessment error: {e}")

    async def _assess_dimension_impact(
        self,
        dimension: ImpactDimension,
        decision_context: DecisionContext,
        actions: List[ActionRecommendation],
        system_context: Optional[LegacySystemContext],
    ) -> Dict[str, Any]:
        """Assess impact for a specific dimension."""

        dimension_impact = {
            "dimension": dimension.value,
            "impact_score": 0.0,  # -10 to +10 scale
            "certainty": 0.0,
            "timeline": "unknown",
            "details": {},
            "risks": [],
            "opportunities": [],
        }

        if dimension == ImpactDimension.FINANCIAL:
            dimension_impact = await self._assess_financial_impact(
                decision_context, actions, system_context
            )
        elif dimension == ImpactDimension.OPERATIONAL:
            dimension_impact = await self._assess_operational_impact(
                decision_context, actions, system_context
            )
        elif dimension == ImpactDimension.REGULATORY:
            dimension_impact = await self._assess_regulatory_impact(
                decision_context, actions, system_context
            )
        elif dimension == ImpactDimension.SECURITY:
            dimension_impact = await self._assess_security_impact(
                decision_context, actions, system_context
            )
        elif dimension == ImpactDimension.STRATEGIC:
            dimension_impact = await self._assess_strategic_impact(
                decision_context, actions, system_context
            )
        else:
            # Generic assessment for other dimensions
            dimension_impact["impact_score"] = 5.0  # Neutral
            dimension_impact["certainty"] = 0.5

        return dimension_impact

    async def _assess_financial_impact(
        self,
        decision_context: DecisionContext,
        actions: List[ActionRecommendation],
        system_context: Optional[LegacySystemContext],
    ) -> Dict[str, Any]:
        """Assess financial impact."""

        # Calculate direct costs
        total_implementation_cost = sum(
            action.estimated_cost or 0 for action in actions
        )

        # Estimate cost savings (simplified model)
        annual_maintenance_cost = 100000  # Base estimate
        if system_context:
            if system_context.criticality == SeverityLevel.CRITICAL:
                annual_maintenance_cost *= 2
            elif system_context.criticality == SeverityLevel.HIGH:
                annual_maintenance_cost *= 1.5

        # Calculate potential savings based on action types
        estimated_savings = 0
        for action in actions:
            if action.action_type == ActionType.SCHEDULED_MAINTENANCE:
                estimated_savings += annual_maintenance_cost * 0.1  # 10% savings
            elif action.action_type == ActionType.PERFORMANCE_OPTIMIZATION:
                estimated_savings += annual_maintenance_cost * 0.15  # 15% savings

        # Calculate ROI
        net_benefit = estimated_savings - total_implementation_cost
        roi = (
            (net_benefit / total_implementation_cost)
            if total_implementation_cost > 0
            else 0
        )

        # Risk-adjusted calculations
        failure_probability = decision_context.current_situation.get(
            "failure_probability", 0.1
        )
        avoided_failure_cost = (
            annual_maintenance_cost * 3 * failure_probability
        )  # 3x cost if failure

        # Calculate impact score (-10 to +10)
        if roi > 0.5:  # 50%+ ROI
            impact_score = 8.0
        elif roi > 0.2:  # 20%+ ROI
            impact_score = 6.0
        elif roi > 0:  # Positive ROI
            impact_score = 4.0
        elif roi > -0.2:  # Minor loss
            impact_score = 2.0
        else:  # Significant loss
            impact_score = -2.0

        return {
            "dimension": "financial",
            "impact_score": impact_score,
            "certainty": 0.7,
            "timeline": "1-2 years",
            "details": {
                "implementation_cost": total_implementation_cost,
                "estimated_annual_savings": estimated_savings,
                "roi": roi,
                "payback_period_months": (
                    (total_implementation_cost / (estimated_savings / 12))
                    if estimated_savings > 0
                    else float("inf")
                ),
                "avoided_failure_cost": avoided_failure_cost,
            },
            "risks": ["cost_overruns", "lower_than_expected_savings"],
            "opportunities": ["additional_efficiency_gains", "extended_system_life"],
        }

    async def _assess_operational_impact(
        self,
        decision_context: DecisionContext,
        actions: List[ActionRecommendation],
        system_context: Optional[LegacySystemContext],
    ) -> Dict[str, Any]:
        """Assess operational impact."""

        # Analyze operational improvements
        availability_improvement = 0
        performance_improvement = 0
        efficiency_improvement = 0

        for action in actions:
            if action.action_type == ActionType.SCHEDULED_MAINTENANCE:
                availability_improvement += 2  # 2% improvement
                performance_improvement += 5  # 5% improvement
            elif action.action_type == ActionType.MONITORING_ADJUSTMENT:
                availability_improvement += 1  # 1% improvement

        # Current operational metrics (assumed baseline)
        current_availability = decision_context.current_situation.get(
            "availability", 95.0
        )
        current_performance = decision_context.current_situation.get(
            "performance_score", 80.0
        )

        # Calculate new metrics
        new_availability = min(current_availability + availability_improvement, 99.9)
        new_performance = min(current_performance + performance_improvement, 100.0)

        # Calculate impact score
        availability_impact = (
            new_availability - current_availability
        ) / 5.0  # Normalize
        performance_impact = (new_performance - current_performance) / 20.0  # Normalize

        impact_score = min(availability_impact + performance_impact, 10.0)

        return {"dimension": "operational", "impact_score": impact_score}

    async def _create_contingency_plans(
        self, risks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create contingency plans for identified risks."""

        contingencies = []

        for risk in risks:
            risk_desc = risk.get("description", str(risk))
            if "downtime" in risk_desc.lower():
                contingencies.append(
                    {
                        "trigger": "extended_system_downtime",
                        "response": "Activate backup systems and notify stakeholders",
                        "owner": "operations_manager",
                        "timeline": "immediate",
                    }
                )
            elif "cost" in risk_desc.lower():
                contingencies.append(
                    {
                        "trigger": "cost_overrun_>20%",
                        "response": "Review scope and consider phased implementation",
                        "owner": "project_manager",
                        "timeline": "within_24_hours",
                    }
                )
            else:
                contingencies.append(
                    {
                        "trigger": f"{risk_desc}_occurs",
                        "response": "Implement standard risk response procedures",
                        "owner": "project_manager",
                        "timeline": "as_needed",
                    }
                )

        return contingencies

    def _define_risk_escalation_triggers(
        self, action: ActionRecommendation
    ) -> List[str]:
        """Define triggers for risk escalation."""

        return [
            "implementation_delay_>48_hours",
            "cost_overrun_>25%",
            "critical_system_failure",
            "stakeholder_objection",
            "compliance_violation_risk",
        ]

    def _create_communication_plan(
        self, action: ActionRecommendation, decision_context: DecisionContext
    ) -> Dict[str, Any]:
        """Create communication plan for action implementation."""

        stakeholders = list(
            set(decision_context.stakeholders + ["project_team", "end_users"])
        )

        return {
            "stakeholder_matrix": {
                stakeholder: {
                    "role": (
                        "participant" if stakeholder in ["project_team"] else "informed"
                    ),
                    "communication_frequency": self._determine_communication_frequency(
                        stakeholder, action
                    ),
                    "preferred_channels": ["email", "meetings", "dashboard"],
                    "key_messages": self._generate_stakeholder_messages(
                        stakeholder, action
                    ),
                }
                for stakeholder in stakeholders
            },
            "communication_schedule": self._create_communication_schedule(action),
            "escalation_communication": {
                "triggers": ["milestone_delay", "risk_materialization", "scope_change"],
                "recipients": ["project_manager", "sponsor", "operations_manager"],
                "timeline": "within_2_hours",
            },
            "status_reporting": {
                "frequency": (
                    "weekly" if action.priority != SeverityLevel.CRITICAL else "daily"
                ),
                "format": "dashboard_and_email",
                "metrics": ["progress_percentage", "risks_status", "budget_status"],
            },
        }

    def _determine_communication_frequency(
        self, stakeholder: str, action: ActionRecommendation
    ) -> str:
        """Determine communication frequency for stakeholder."""

        if stakeholder in ["project_team", "operations_team"]:
            return "daily" if action.priority == SeverityLevel.CRITICAL else "weekly"
        elif stakeholder in ["management", "sponsor"]:
            return "weekly"
        else:
            return "milestone_based"

    def _generate_stakeholder_messages(
        self, stakeholder: str, action: ActionRecommendation
    ) -> List[str]:
        """Generate key messages for stakeholder."""

        if stakeholder == "end_users":
            return [
                "System improvements being implemented",
                "Minimal service disruption expected",
                "Support available during transition",
            ]
        elif stakeholder == "management":
            return [
                "Strategic initiative progress update",
                "Business value realization on track",
                "Risk mitigation measures in place",
            ]
        else:
            return [
                "Project status and next steps",
                "Action items and deliverables",
                "Issue escalation if needed",
            ]

    def _create_communication_schedule(
        self, action: ActionRecommendation
    ) -> List[Dict[str, str]]:
        """Create detailed communication schedule."""

        return [
            {
                "event": "project_kickoff",
                "timing": "start_of_implementation",
                "audience": "all_stakeholders",
                "format": "meeting",
            },
            {
                "event": "progress_updates",
                "timing": (
                    "weekly" if action.priority != SeverityLevel.CRITICAL else "daily"
                ),
                "audience": "core_team",
                "format": "status_report",
            },
            {
                "event": "milestone_reviews",
                "timing": "at_each_milestone",
                "audience": "steering_committee",
                "format": "presentation",
            },
            {
                "event": "completion_announcement",
                "timing": "end_of_implementation",
                "audience": "all_stakeholders",
                "format": "announcement",
            },
        ]

    def _create_qa_plan(self, action: ActionRecommendation) -> Dict[str, Any]:
        """Create quality assurance plan."""

        return {
            "quality_standards": {
                "deliverable_quality": "enterprise_grade",
                "testing_requirements": self._determine_testing_requirements(action),
                "review_processes": [
                    "peer_review",
                    "technical_review",
                    "business_review",
                ],
                "acceptance_criteria": action.success_criteria,
            },
            "testing_strategy": {
                "test_types": self._identify_test_types(action),
                "test_environments": ["development", "staging", "production"],
                "test_data_requirements": self._identify_test_data_needs(action),
                "automated_testing": action.action_type
                in [ActionType.MONITORING_ADJUSTMENT, ActionType.PROCESS_IMPROVEMENT],
            },
            "quality_gates": self._define_quality_gates(action),
            "defect_management": {
                "tracking_system": "project_management_tool",
                "severity_levels": ["critical", "high", "medium", "low"],
                "resolution_timeframes": {
                    "critical": "4_hours",
                    "high": "24_hours",
                    "medium": "72_hours",
                    "low": "1_week",
                },
            },
        }

    def _determine_testing_requirements(
        self, action: ActionRecommendation
    ) -> List[str]:
        """Determine testing requirements based on action type."""

        if action.action_type == ActionType.IMMEDIATE_ACTION:
            return ["smoke_testing", "regression_testing"]
        elif action.action_type == ActionType.SCHEDULED_MAINTENANCE:
            return ["functional_testing", "performance_testing", "regression_testing"]
        elif action.action_type == ActionType.TECHNOLOGY_EVALUATION:
            return ["proof_of_concept", "pilot_testing", "user_acceptance_testing"]
        else:
            return ["unit_testing", "integration_testing", "user_acceptance_testing"]

    def _identify_test_types(self, action: ActionRecommendation) -> List[str]:
        """Identify types of testing required."""

        base_tests = ["functional_testing"]

        if action.action_type in [
            ActionType.SCHEDULED_MAINTENANCE,
            ActionType.IMMEDIATE_ACTION,
        ]:
            base_tests.extend(["performance_testing", "security_testing"])

        if action.priority in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            base_tests.append("stress_testing")

        return base_tests

    def _identify_test_data_needs(self, action: ActionRecommendation) -> List[str]:
        """Identify test data requirements."""

        if action.action_type == ActionType.MONITORING_ADJUSTMENT:
            return ["historical_metrics", "synthetic_data", "edge_cases"]
        elif action.action_type == ActionType.SCHEDULED_MAINTENANCE:
            return ["production_like_data", "backup_data"]
        else:
            return ["sample_data", "test_scenarios"]

    def _define_quality_gates(
        self, action: ActionRecommendation
    ) -> List[Dict[str, str]]:
        """Define quality gates for the implementation."""

        return [
            {
                "gate": "requirements_review",
                "criteria": "All requirements documented and approved",
                "checkpoint": "before_implementation",
            },
            {
                "gate": "design_review",
                "criteria": "Implementation approach reviewed and approved",
                "checkpoint": "before_execution",
            },
            {
                "gate": "testing_complete",
                "criteria": "All tests passed and defects resolved",
                "checkpoint": "before_deployment",
            },
            {
                "gate": "acceptance_review",
                "criteria": "Success criteria met and stakeholder sign-off",
                "checkpoint": "implementation_complete",
            },
        ]

    def _generate_detailed_success_metrics(
        self, action: ActionRecommendation
    ) -> Dict[str, Any]:
        """Generate detailed success metrics and KPIs."""

        return {
            "primary_metrics": {
                "implementation_success": {
                    "measure": "all_deliverables_completed",
                    "target": "100%",
                    "measurement_method": "checklist_completion",
                },
                "timeline_adherence": {
                    "measure": "on_time_delivery",
                    "target": "within_10%_of_planned_duration",
                    "measurement_method": "schedule_variance_analysis",
                },
                "budget_adherence": {
                    "measure": "cost_control",
                    "target": "within_15%_of_budget",
                    "measurement_method": "cost_variance_analysis",
                },
            },
            "secondary_metrics": {
                "stakeholder_satisfaction": {
                    "measure": "satisfaction_survey",
                    "target": ">=4.0_out_of_5.0",
                    "measurement_method": "survey_results",
                },
                "quality_metrics": {
                    "measure": "defect_density",
                    "target": "<=2_critical_defects",
                    "measurement_method": "defect_tracking",
                },
            },
            "business_impact_metrics": self._generate_business_impact_metrics(action),
            "measurement_timeline": {
                "immediate": "within_48_hours_of_completion",
                "short_term": "within_30_days",
                "long_term": "within_90_days",
            },
        }

    def _generate_business_impact_metrics(
        self, action: ActionRecommendation
    ) -> Dict[str, Any]:
        """Generate business impact metrics specific to action type."""

        if action.action_type == ActionType.SCHEDULED_MAINTENANCE:
            return {
                "system_availability": {
                    "measure": "uptime_percentage",
                    "target": ">=99.5%",
                    "baseline": "current_availability",
                },
                "performance_improvement": {
                    "measure": "response_time",
                    "target": "10%_improvement",
                    "baseline": "current_response_time",
                },
            }
        elif action.action_type == ActionType.MONITORING_ADJUSTMENT:
            return {
                "detection_capability": {
                    "measure": "issue_detection_time",
                    "target": "50%_reduction",
                    "baseline": "current_detection_time",
                },
                "false_positive_rate": {
                    "measure": "alert_accuracy",
                    "target": "<=5%_false_positives",
                    "baseline": "current_false_positive_rate",
                },
            }
        else:
            return {
                "process_efficiency": {
                    "measure": "task_completion_time",
                    "target": "20%_improvement",
                    "baseline": "current_completion_time",
                }
            }

    def _create_approval_workflow(
        self, action: ActionRecommendation, template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create approval workflow for action implementation."""

        required_approvals = template.get("required_approvals", ["manager"])

        workflow_steps = []
        for i, approver_role in enumerate(required_approvals):
            workflow_steps.append(
                {
                    "step": i + 1,
                    "approver_role": approver_role,
                    "approval_type": "sequential",
                    "criteria": self._get_approval_criteria(approver_role, action),
                    "escalation_days": 3,
                    "escalation_to": self._get_escalation_target(approver_role),
                }
            )

        return {
            "workflow_steps": workflow_steps,
            "parallel_approvals": action.priority == SeverityLevel.CRITICAL,
            "auto_approval_conditions": self._get_auto_approval_conditions(action),
            "approval_documentation": "required",
            "approval_timeout": "5_business_days",
        }

    def _get_approval_criteria(
        self, approver_role: str, action: ActionRecommendation
    ) -> List[str]:
        """Get approval criteria for specific approver role."""

        criteria_map = {
            "operations_manager": [
                "operational_impact_acceptable",
                "maintenance_window_approved",
                "resource_availability_confirmed",
            ],
            "it_manager": [
                "technical_approach_sound",
                "security_requirements_met",
                "compliance_verified",
            ],
            "cto": [
                "strategic_alignment_confirmed",
                "architecture_impact_acceptable",
                "long_term_viability_assured",
            ],
        }

        return criteria_map.get(approver_role, ["general_approval_criteria"])

    def _get_escalation_target(self, approver_role: str) -> str:
        """Get escalation target for approver role."""

        escalation_map = {
            "operations_manager": "it_director",
            "it_manager": "cto",
            "cto": "ceo",
            "manager": "director",
        }

        return escalation_map.get(approver_role, "senior_management")

    def _get_auto_approval_conditions(self, action: ActionRecommendation) -> List[str]:
        """Get conditions for automatic approval."""

        conditions = []

        if action.estimated_cost and action.estimated_cost < 5000:
            conditions.append("cost_under_5000_usd")

        if action.action_type == ActionType.MONITORING_ADJUSTMENT:
            conditions.append("monitoring_adjustment_low_risk")

        if action.estimated_effort == "low":
            conditions.append("low_effort_implementation")

        return conditions

    def _identify_dependencies(
        self, action: ActionRecommendation
    ) -> List[Dict[str, Any]]:
        """Identify dependencies for action implementation."""

        dependencies = []

        # Add prerequisite-based dependencies
        for prerequisite in action.prerequisites:
            dependencies.append(
                {
                    "type": "prerequisite",
                    "description": prerequisite,
                    "criticality": "blocking",
                    "owner": "project_manager",
                    "status": "pending",
                }
            )

        # Add action-type specific dependencies
        if action.action_type == ActionType.SCHEDULED_MAINTENANCE:
            dependencies.extend(
                [
                    {
                        "type": "maintenance_window",
                        "description": "Approved maintenance window scheduled",
                        "criticality": "blocking",
                        "owner": "operations_manager",
                        "status": "pending",
                    },
                    {
                        "type": "backup_completion",
                        "description": "System backup completed successfully",
                        "criticality": "blocking",
                        "owner": "backup_administrator",
                        "status": "pending",
                    },
                ]
            )
        elif action.action_type == ActionType.TECHNOLOGY_EVALUATION:
            dependencies.append(
                {
                    "type": "vendor_availability",
                    "description": "Vendor demonstration and support available",
                    "criticality": "non_blocking",
                    "owner": "procurement_team",
                    "status": "pending",
                }
            )

        return dependencies

    def _define_milestones(self, timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define key milestones from timeline."""

        milestones = []
        phases = timeline.get("phases", [])

        for phase in phases:
            if phase.get("critical", False):
                milestones.append(
                    {
                        "milestone": f"{phase['phase']}_complete",
                        "target_date": phase["end_date"],
                        "criteria": phase.get("deliverables", []),
                        "owner": "project_manager",
                        "review_required": True,
                    }
                )

        # Add overall completion milestone
        if phases:
            milestones.append(
                {
                    "milestone": "implementation_complete",
                    "target_date": phases[-1]["end_date"],
                    "criteria": [
                        "all_phases_complete",
                        "success_criteria_met",
                        "stakeholder_acceptance",
                    ],
                    "owner": "project_sponsor",
                    "review_required": True,
                }
            )

        return milestones

    def _create_contingency_plans(self, action: ActionRecommendation) -> Dict[str, Any]:
        """Create comprehensive contingency plans."""

        return {
            "technical_contingencies": {
                "implementation_failure": {
                    "trigger": "critical_implementation_error",
                    "response_plan": action.rollback_plan
                    or "Execute rollback procedures",
                    "recovery_time": "4_hours",
                    "responsible_team": "technical_team",
                },
                "performance_degradation": {
                    "trigger": "performance_below_baseline",
                    "response_plan": "Activate performance optimization procedures",
                    "recovery_time": "2_hours",
                    "responsible_team": "operations_team",
                },
            },
            "resource_contingencies": {
                "key_personnel_unavailable": {
                    "trigger": "key_team_member_unavailable",
                    "response_plan": "Activate backup personnel or external resources",
                    "recovery_time": "24_hours",
                    "responsible_team": "project_management",
                },
                "budget_overrun": {
                    "trigger": "cost_exceeds_approved_budget_by_20%",
                    "response_plan": "Request additional funding or reduce scope",
                    "recovery_time": "72_hours",
                    "responsible_team": "project_sponsor",
                },
            },
            "business_contingencies": {
                "stakeholder_objection": {
                    "trigger": "significant_stakeholder_concerns_raised",
                    "response_plan": "Conduct stakeholder workshop and adjust approach",
                    "recovery_time": "1_week",
                    "responsible_team": "project_management",
                },
                "regulatory_compliance_issue": {
                    "trigger": "compliance_violation_identified",
                    "response_plan": "Engage compliance team and legal counsel",
                    "recovery_time": "immediate",
                    "responsible_team": "compliance_team",
                },
            },
        }

    def _create_progress_tracking_plan(
        self, action: ActionRecommendation
    ) -> Dict[str, Any]:
        """Create progress tracking and monitoring plan."""

        return {
            "tracking_methods": {
                "milestone_tracking": "project_management_tool",
                "resource_tracking": "time_tracking_system",
                "budget_tracking": "financial_management_system",
                "quality_tracking": "quality_management_system",
            },
            "reporting_schedule": {
                "daily_updates": action.priority == SeverityLevel.CRITICAL,
                "weekly_reports": True,
                "milestone_reports": True,
                "exception_reports": "as_needed",
            },
            "kpi_monitoring": {
                "schedule_variance": "daily",
                "cost_variance": "weekly",
                "quality_metrics": "per_deliverable",
                "risk_status": "weekly",
            },
            "dashboard_requirements": {
                "real_time_updates": action.priority == SeverityLevel.CRITICAL,
                "stakeholder_views": "customized_by_role",
                "mobile_access": True,
                "alerts_enabled": True,
            },
        }

    def _define_review_checkpoints(
        self, timeline: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Define review checkpoints throughout implementation."""

        checkpoints = []
        phases = timeline.get("phases", [])

        for i, phase in enumerate(phases):
            if i == 0:  # First phase
                checkpoints.append(
                    {
                        "checkpoint": "implementation_readiness",
                        "timing": "before_phase_1_start",
                        "participants": [
                            "project_manager",
                            "technical_lead",
                            "business_sponsor",
                        ],
                        "criteria": [
                            "all_prerequisites_met",
                            "team_ready",
                            "approvals_complete",
                        ],
                        "outcomes": ["go_no_go_decision"],
                    }
                )

            # End of each phase
            checkpoints.append(
                {
                    "checkpoint": f"{phase['phase']}_review",
                    "timing": phase["end_date"],
                    "participants": ["project_team", "stakeholders"],
                    "criteria": phase.get("deliverables", []),
                    "outcomes": ["phase_acceptance", "next_phase_approval"],
                }
            )

        # Final review
        checkpoints.append(
            {
                "checkpoint": "project_closure",
                "timing": "after_final_phase",
                "participants": ["all_stakeholders"],
                "criteria": [
                    "objectives_achieved",
                    "lessons_documented",
                    "handover_complete",
                ],
                "outcomes": ["project_acceptance", "benefits_realization_plan"],
            }
        )

        return checkpoints

    def _create_escalation_procedures(
        self, action: ActionRecommendation
    ) -> Dict[str, Any]:
        """Create escalation procedures for issues."""

        return {
            "escalation_matrix": {
                "level_1": {
                    "trigger": "minor_issues_or_delays",
                    "escalate_to": "project_manager",
                    "timeline": "within_4_hours",
                    "resolution_authority": "adjust_resources_or_timeline",
                },
                "level_2": {
                    "trigger": "significant_risks_or_budget_issues",
                    "escalate_to": "project_sponsor",
                    "timeline": "within_24_hours",
                    "resolution_authority": "approve_scope_or_budget_changes",
                },
                "level_3": {
                    "trigger": "critical_failures_or_stakeholder_conflicts",
                    "escalate_to": "steering_committee",
                    "timeline": "within_48_hours",
                    "resolution_authority": "fundamental_approach_changes",
                },
            },
            "escalation_triggers": [
                "schedule_delay_>25%",
                "budget_overrun_>20%",
                "quality_gate_failure",
                "stakeholder_escalation",
                "compliance_violation",
                "security_incident",
            ],
            "communication_requirements": {
                "notification_method": "immediate_email_and_phone",
                "documentation_required": "incident_report_within_24_hours",
                "follow_up_meetings": "within_72_hours",
            },
        }

    def _identify_required_documentation(
        self, action: ActionRecommendation
    ) -> List[Dict[str, str]]:
        """Identify required documentation for action implementation."""

        base_docs = [
            {
                "document": "implementation_plan",
                "owner": "project_manager",
                "timing": "before_start",
            },
            {
                "document": "test_results",
                "owner": "qa_team",
                "timing": "during_testing",
            },
            {
                "document": "completion_report",
                "owner": "project_manager",
                "timing": "at_completion",
            },
        ]

        if action.action_type == ActionType.SCHEDULED_MAINTENANCE:
            base_docs.extend(
                [
                    {
                        "document": "maintenance_procedures",
                        "owner": "technical_team",
                        "timing": "before_maintenance",
                    },
                    {
                        "document": "system_validation_report",
                        "owner": "operations_team",
                        "timing": "after_maintenance",
                    },
                ]
            )
        elif action.action_type == ActionType.POLICY_CHANGE:
            base_docs.extend(
                [
                    {
                        "document": "policy_impact_analysis",
                        "owner": "compliance_team",
                        "timing": "before_implementation",
                    },
                    {
                        "document": "training_materials",
                        "owner": "training_team",
                        "timing": "before_rollout",
                    },
                ]
            )

        return base_docs

    def _identify_compliance_requirements(
        self, action: ActionRecommendation
    ) -> List[Dict[str, str]]:
        """Identify compliance requirements for action implementation."""

        requirements = []

        if action.action_type in [
            ActionType.POLICY_CHANGE,
            ActionType.PROCESS_IMPROVEMENT,
        ]:
            requirements.extend(
                [
                    {
                        "requirement": "regulatory_compliance_review",
                        "framework": "applicable_regulations",
                    },
                    {
                        "requirement": "audit_trail_maintenance",
                        "framework": "internal_audit",
                    },
                    {
                        "requirement": "change_management_approval",
                        "framework": "change_control",
                    },
                ]
            )

        if action.priority == SeverityLevel.CRITICAL:
            requirements.append(
                {
                    "requirement": "emergency_change_approval",
                    "framework": "emergency_procedures",
                }
            )

        if action.estimated_cost and action.estimated_cost > 25000:
            requirements.append(
                {"requirement": "financial_approval", "framework": "financial_controls"}
            )

        return requirements

    def _document_planning_assumptions(self, action: ActionRecommendation) -> List[str]:
        """Document key planning assumptions."""

        assumptions = [
            "Required resources will be available as scheduled",
            "System access and permissions will be granted as needed",
            "Stakeholder cooperation and support will be maintained",
            "No major external dependencies will cause delays",
        ]

        if action.action_type == ActionType.SCHEDULED_MAINTENANCE:
            assumptions.extend(
                [
                    "Maintenance window will be sufficient for all activities",
                    "System backup and recovery procedures are tested and functional",
                    "No critical incidents will occur during maintenance",
                ]
            )

        if action.estimated_cost and action.estimated_cost > 15000:
            assumptions.append(
                "Budget approval will be maintained throughout implementation"
            )

        return assumptions

    # Add completion method for ActionPlanner
    def get_planning_templates(self) -> Dict[ActionType, Dict[str, Any]]:
        """Get available planning templates."""
        return self.planning_templates.copy()

    def update_planning_template(
        self, action_type: ActionType, template_updates: Dict[str, Any]
    ) -> None:
        """Update planning template for an action type."""
        if action_type in self.planning_templates:
            self.planning_templates[action_type].update(template_updates)
            self.logger.info(f"Updated planning template for {action_type.value}")
        else:
            self.planning_templates[action_type] = template_updates
            self.logger.info(f"Created new planning template for {action_type.value}")
