"""
QBITEL - Compliance-specific LLM Prompt Templates

Enterprise-grade prompt management system for compliance reporting
with framework-specific templates and dynamic prompt generation.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import jinja2

from ..core.config import Config
from ..core.exceptions import QbitelAIException
from .regulatory_kb import (
    ComplianceAssessment,
    ComplianceGap,
    ComplianceRecommendation,
    ComplianceRequirement,
    RequirementSeverity,
)

logger = logging.getLogger(__name__)


class PromptException(QbitelAIException):
    """Prompt template specific exception."""

    pass


class PromptType(Enum):
    """Types of compliance prompts."""

    GAP_ANALYSIS = "gap_analysis"
    REQUIREMENT_ASSESSMENT = "requirement_assessment"
    RISK_EVALUATION = "risk_evaluation"
    RECOMMENDATION_GENERATION = "recommendation_generation"
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_REPORT = "technical_report"
    REGULATORY_FILING = "regulatory_filing"
    REMEDIATION_PLAN = "remediation_plan"
    COMPLIANCE_SCORING = "compliance_scoring"
    FRAMEWORK_MAPPING = "framework_mapping"


class PromptCategory(Enum):
    """Categories of prompt usage."""

    ASSESSMENT = "assessment"
    REPORTING = "reporting"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VALIDATION = "validation"


@dataclass
class PromptTemplate:
    """LLM prompt template with metadata."""

    id: str
    name: str
    description: str
    type: PromptType
    category: PromptCategory
    framework: Optional[str] = None
    template: str = ""
    variables: List[str] = field(default_factory=list)
    version: str = "1.0"
    created_date: datetime = field(default_factory=datetime.utcnow)
    updated_date: datetime = field(default_factory=datetime.utcnow)
    author: str = "system"
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


class CompliancePromptManager:
    """Manages compliance-specific LLM prompts and templates."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Template storage
        self.templates: Dict[str, PromptTemplate] = {}

        # Jinja2 environment for template rendering
        # Note: autoescape=False is intentional - these are LLM prompt templates,
        # not HTML templates. No XSS risk as output goes to LLM, not browsers.
        self.jinja_env = jinja2.Environment(
            loader=jinja2.DictLoader({}),
            autoescape=False,  # nosec B701 - LLM prompts, not HTML
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Load built-in templates
        self._load_builtin_templates()

        # Framework-specific configurations
        self.framework_configs = {
            "PCI_DSS_4_0": {
                "emphasis": "data protection and cardholder data security",
                "key_areas": [
                    "network security",
                    "access controls",
                    "encryption",
                    "monitoring",
                ],
                "regulatory_language": "formal and technical",
                "compliance_focus": "payment card industry standards",
            },
            "HIPAA": {
                "emphasis": "protected health information (PHI) safeguards",
                "key_areas": ["administrative", "physical", "technical safeguards"],
                "regulatory_language": "healthcare-focused and privacy-centric",
                "compliance_focus": "healthcare data protection",
            },
            "BASEL_III": {
                "emphasis": "capital adequacy and risk management",
                "key_areas": [
                    "capital ratios",
                    "liquidity requirements",
                    "risk assessment",
                ],
                "regulatory_language": "financial services terminology",
                "compliance_focus": "banking regulation compliance",
            },
            "NERC_CIP": {
                "emphasis": "critical infrastructure protection",
                "key_areas": [
                    "cyber security",
                    "physical security",
                    "personnel training",
                ],
                "regulatory_language": "utility and power grid terminology",
                "compliance_focus": "electrical grid cybersecurity",
            },
            "FDA_MEDICAL": {
                "emphasis": "medical device safety and quality",
                "key_areas": [
                    "design controls",
                    "quality management",
                    "risk management",
                ],
                "regulatory_language": "medical device regulatory terminology",
                "compliance_focus": "FDA medical device regulations",
            },
        }

    def _load_builtin_templates(self):
        """Load built-in prompt templates for compliance frameworks."""

        # Gap Analysis Template
        self.templates["gap_analysis_comprehensive"] = PromptTemplate(
            id="gap_analysis_comprehensive",
            name="Comprehensive Gap Analysis",
            description="Detailed compliance gap analysis with root cause identification",
            type=PromptType.GAP_ANALYSIS,
            category=PromptCategory.ANALYSIS,
            template="""
You are a compliance expert conducting a comprehensive gap analysis for {{ framework }} compliance.

FRAMEWORK CONTEXT:
- Framework: {{ framework }}
- Emphasis: {{ framework_config.emphasis }}
- Key Areas: {{ framework_config.key_areas | join(', ') }}
- Compliance Focus: {{ framework_config.compliance_focus }}

ASSESSMENT DATA:
- Overall Compliance Score: {{ assessment.overall_compliance_score }}%
- Risk Score: {{ assessment.risk_score }}%
- Total Requirements: {{ total_requirements }}
- Compliant: {{ assessment.compliant_requirements }}
- Non-Compliant: {{ assessment.non_compliant_requirements }}
- Partially Compliant: {{ assessment.partially_compliant_requirements }}

IDENTIFIED GAPS:
{% for gap in gaps %}
{{ loop.index }}. {{ gap.requirement_title }}
   - Severity: {{ gap.severity.value }}
   - Current State: {{ gap.current_state }}
   - Required State: {{ gap.required_state }}
   - Gap Description: {{ gap.gap_description }}
{% endfor %}

SYSTEM EVIDENCE:
{{ system_evidence | truncate(1000) }}

TASK: Provide a comprehensive gap analysis that includes:

1. **Executive Summary**: Overall compliance posture and critical findings
2. **Gap Prioritization**: Rank gaps by business risk and regulatory impact
3. **Root Cause Analysis**: Identify underlying causes for major gaps
4. **Compliance Impact**: Assess potential regulatory and business consequences
5. **Interdependencies**: Map relationships between gaps
6. **Quick Wins**: Identify gaps that can be resolved quickly with high impact

Use professional {{ framework_config.regulatory_language }} appropriate for {{ framework }} compliance reporting.

Return structured analysis with clear priorities and actionable insights.
""",
            variables=[
                "framework",
                "framework_config",
                "assessment",
                "total_requirements",
                "gaps",
                "system_evidence",
            ],
            tags=["gap_analysis", "comprehensive", "prioritization"],
        )

        # Requirement Assessment Template
        self.templates["requirement_assessment_detailed"] = PromptTemplate(
            id="requirement_assessment_detailed",
            name="Detailed Requirement Assessment",
            description="Individual requirement compliance assessment with evidence analysis",
            type=PromptType.REQUIREMENT_ASSESSMENT,
            category=PromptCategory.ASSESSMENT,
            template="""
You are a {{ framework }} compliance specialist assessing a specific requirement.

REQUIREMENT DETAILS:
- ID: {{ requirement.id }}
- Title: {{ requirement.title }}
- Description: {{ requirement.description }}
- Severity: {{ requirement.severity.value }}
- Control Type: {{ requirement.control_type.value }}
- Section: {{ requirement.section }}

VALIDATION CRITERIA:
{% for criterion in requirement.validation_criteria %}
- {{ criterion }}
{% endfor %}

EVIDENCE ANALYSIS:
{% for evidence in evidence_items %}
Source: {{ evidence.source }}
Type: {{ evidence.data_type }}
Confidence: {{ evidence.confidence }}
Data: {{ evidence.value | truncate(200) }}
---
{% endfor %}

FRAMEWORK CONTEXT:
{{ framework_config.compliance_focus }} with emphasis on {{ framework_config.emphasis }}.

TASK: Assess compliance with this requirement and provide:

1. **Compliance Status**: compliant | partially_compliant | non_compliant | not_applicable
2. **Compliance Score**: 0-100 percentage based on evidence
3. **Evidence Assessment**: Quality and completeness of available evidence
4. **Findings**: Specific observations from evidence analysis
5. **Gap Analysis**: What's missing or inadequate
6. **Risk Assessment**: Impact if non-compliant (very_low | low | medium | high | critical)
7. **Recommendations**: Specific, actionable steps to achieve compliance
8. **Implementation Guidance**: Practical advice for remediation

Focus on objective, evidence-based assessment using {{ framework_config.regulatory_language }}.

Return response as JSON with the specified structure.
""",
            variables=[
                "framework",
                "framework_config",
                "requirement",
                "evidence_items",
            ],
            tags=["requirement", "assessment", "evidence_based"],
        )

        # Executive Summary Template
        self.templates["executive_summary_strategic"] = PromptTemplate(
            id="executive_summary_strategic",
            name="Strategic Executive Summary",
            description="High-level executive summary for C-level stakeholders",
            type=PromptType.EXECUTIVE_SUMMARY,
            category=PromptCategory.REPORTING,
            template="""
You are preparing an executive summary for C-level leadership regarding {{ framework }} compliance status.

COMPLIANCE OVERVIEW:
- Framework: {{ framework }}
- Assessment Date: {{ assessment.assessment_date.strftime('%B %d, %Y') }}
- Overall Compliance: {{ assessment.overall_compliance_score }}%
- Risk Level: {{ 'HIGH' if assessment.risk_score > 70 else 'MEDIUM' if assessment.risk_score > 30 else 'LOW' }}
- Requirements Status: {{ assessment.compliant_requirements }} compliant, {{ assessment.non_compliant_requirements }} non-compliant

CRITICAL GAPS ({{ critical_gap_count }}):
{% for gap in critical_gaps[:3] %}
- {{ gap.requirement_title }}: {{ gap.impact_assessment }}
{% endfor %}

BUSINESS CONTEXT:
- Regulatory Framework: {{ framework_config.compliance_focus }}
- Key Focus Areas: {{ framework_config.key_areas | join(', ') }}
- Business Impact: {{ business_impact }}

TASK: Create a concise executive summary (2-3 paragraphs) that covers:

1. **Current Compliance Status**: Overall posture and key metrics
2. **Critical Risk Areas**: Most significant compliance gaps and their business impact
3. **Strategic Recommendations**: High-level actions required from leadership
4. **Resource Requirements**: Investment needed for compliance achievement
5. **Timeline**: Realistic timeline for reaching full compliance
6. **Business Benefits**: Value of achieving compliance beyond regulatory requirements

Use business-focused language appropriate for executive audiences. Focus on:
- Strategic impact and business risk
- Required leadership decisions and resource allocation  
- ROI and competitive advantages of compliance
- Regulatory relationship management

Avoid technical details. Emphasize business value and risk mitigation.
""",
            variables=[
                "framework",
                "framework_config",
                "assessment",
                "critical_gap_count",
                "critical_gaps",
                "business_impact",
            ],
            tags=["executive", "strategic", "business_focused"],
        )

        # Technical Report Template
        self.templates["technical_report_comprehensive"] = PromptTemplate(
            id="technical_report_comprehensive",
            name="Comprehensive Technical Report",
            description="Detailed technical compliance report for technical teams",
            type=PromptType.TECHNICAL_REPORT,
            category=PromptCategory.REPORTING,
            template="""
You are generating a comprehensive technical report for {{ framework }} compliance assessment.

ASSESSMENT METHODOLOGY:
- Automated Assessment Engine with LLM-powered analysis
- System State Analysis with {{ evidence_source_count }} data sources
- Framework-specific requirement validation
- Risk-based gap prioritization

TECHNICAL FINDINGS:
Framework: {{ framework }}
Assessment Date: {{ assessment.assessment_date.isoformat() }}
System Environment: {{ system_environment }}
Assessment Scope: {{ assessment_scope }}

COMPLIANCE METRICS:
- Overall Score: {{ assessment.overall_compliance_score }}%
- Risk Score: {{ assessment.risk_score }}%
- Requirements Breakdown:
  * Compliant: {{ assessment.compliant_requirements }} ({{ (assessment.compliant_requirements / total_requirements * 100) | round(1) }}%)
  * Partially Compliant: {{ assessment.partially_compliant_requirements }} ({{ (assessment.partially_compliant_requirements / total_requirements * 100) | round(1) }}%)
  * Non-Compliant: {{ assessment.non_compliant_requirements }} ({{ (assessment.non_compliant_requirements / total_requirements * 100) | round(1) }}%)

DETAILED FINDINGS:
{% for section in requirement_sections %}
{{ section.name }}:
{% for req in section.requirements %}
  - {{ req.id }}: {{ req.title }} - {{ req.status }}
{% endfor %}
{% endfor %}

TECHNICAL GAPS:
{% for gap in technical_gaps %}
{{ loop.index }}. {{ gap.requirement_title }}
   Technical Issue: {{ gap.gap_description }}
   Current Implementation: {{ gap.current_state }}
   Required Implementation: {{ gap.required_state }}
   Technical Complexity: {{ gap.remediation_effort }}
   Implementation Path: {{ gap.remediation_guidance }}
{% endfor %}

TASK: Generate a comprehensive technical report with:

1. **Executive Summary**: Technical compliance posture overview
2. **Methodology**: Assessment approach and data sources
3. **Detailed Findings**: Requirement-by-requirement analysis
4. **Technical Gaps**: Implementation-focused gap analysis
5. **Architecture Review**: System design compliance assessment
6. **Security Analysis**: Security control effectiveness
7. **Implementation Roadmap**: Technical remediation steps
8. **Validation Procedures**: How to verify compliance achievement
9. **Monitoring Strategy**: Ongoing compliance monitoring approach
10. **Technical Appendix**: Detailed evidence and configurations

Use precise technical language appropriate for {{ framework_config.compliance_focus }}.
Focus on implementation details, configuration requirements, and technical controls.
""",
            variables=[
                "framework",
                "framework_config",
                "assessment",
                "evidence_source_count",
                "system_environment",
                "assessment_scope",
                "total_requirements",
                "requirement_sections",
                "technical_gaps",
            ],
            tags=["technical", "comprehensive", "implementation_focused"],
        )

        # Risk Evaluation Template
        self.templates["risk_evaluation_quantitative"] = PromptTemplate(
            id="risk_evaluation_quantitative",
            name="Quantitative Risk Evaluation",
            description="Quantitative risk assessment with business impact analysis",
            type=PromptType.RISK_EVALUATION,
            category=PromptCategory.ANALYSIS,
            template="""
You are conducting a quantitative risk evaluation for {{ framework }} compliance gaps.

RISK FRAMEWORK:
Framework: {{ framework }}
Risk Assessment Model: Business Impact × Likelihood × Regulatory Consequence
Risk Categories: {{ framework_config.key_areas | join(', ') }}

COMPLIANCE DATA:
- Current Compliance: {{ assessment.overall_compliance_score }}%
- Risk Exposure: {{ assessment.risk_score }}%
- Critical Gaps: {{ critical_gap_count }}
- High-Priority Gaps: {{ high_priority_gap_count }}

IDENTIFIED RISKS:
{% for gap in gaps %}
Risk {{ loop.index }}: {{ gap.requirement_title }}
- Severity: {{ gap.severity.value }}
- Control Type: {{ gap.control_type if gap.control_type else 'Unknown' }}
- Current State: {{ gap.current_state }}
- Business Impact: {{ gap.impact_assessment }}
- Remediation Effort: {{ gap.remediation_effort }}
{% endfor %}

BUSINESS CONTEXT:
Industry: {{ industry_context }}
Regulatory Environment: {{ regulatory_environment }}
Business Criticality: {{ business_criticality }}

TASK: Provide quantitative risk evaluation including:

1. **Risk Scoring Matrix**: 
   - Probability of regulatory action (1-5 scale)
   - Business impact severity (1-5 scale)
   - Detection difficulty (1-5 scale)
   - Overall risk score calculation

2. **Financial Impact Assessment**:
   - Potential regulatory fines
   - Business disruption costs
   - Reputation damage estimates
   - Remediation investment required

3. **Risk Prioritization**:
   - Critical risks requiring immediate action
   - High risks for 30-day resolution
   - Medium risks for 90-day resolution
   - Long-term strategic risks

4. **Risk Mitigation Strategy**:
   - Immediate risk reduction measures
   - Compensating controls
   - Risk transfer options
   - Risk acceptance criteria

5. **Monitoring Plan**:
   - Key risk indicators (KRIs)
   - Risk threshold definitions
   - Escalation procedures
   - Reporting frequency

Use quantitative metrics where possible. Focus on business-relevant risk assessment for {{ framework_config.compliance_focus }}.
""",
            variables=[
                "framework",
                "framework_config",
                "assessment",
                "critical_gap_count",
                "high_priority_gap_count",
                "gaps",
                "industry_context",
                "regulatory_environment",
                "business_criticality",
            ],
            tags=["risk", "quantitative", "business_impact"],
        )

        # Remediation Plan Template
        self.templates["remediation_plan_actionable"] = PromptTemplate(
            id="remediation_plan_actionable",
            name="Actionable Remediation Plan",
            description="Detailed remediation plan with implementation roadmap",
            type=PromptType.REMEDIATION_PLAN,
            category=PromptCategory.GENERATION,
            template="""
You are creating a detailed remediation plan for {{ framework }} compliance gaps.

PROJECT CONTEXT:
- Framework: {{ framework }}
- Target Compliance: {{ target_compliance_score }}%
- Current Score: {{ assessment.overall_compliance_score }}%
- Timeline: {{ target_timeline }} months
- Budget Range: {{ budget_range }}

GAPS TO REMEDIATE:
{% for gap in prioritized_gaps %}
Gap {{ loop.index }}: {{ gap.requirement_title }}
- Priority: {{ gap.severity.value }}
- Effort: {{ gap.remediation_effort }}
- Impact: {{ gap.impact_assessment }}
- Current: {{ gap.current_state }}
- Target: {{ gap.required_state }}
{% endfor %}

ORGANIZATIONAL CONTEXT:
- Team Size: {{ team_size }}
- Technical Expertise: {{ technical_expertise }}
- Budget Constraints: {{ budget_constraints }}
- Regulatory Deadline: {{ regulatory_deadline }}

TASK: Create a comprehensive remediation plan with:

1. **Implementation Roadmap**:
   - Phase 1 (0-30 days): Critical risk mitigation
   - Phase 2 (1-3 months): High-priority gaps
   - Phase 3 (3-6 months): Medium-priority improvements
   - Phase 4 (6+ months): Long-term enhancements

2. **Detailed Action Items**:
   For each gap provide:
   - Specific implementation steps
   - Required resources and skills
   - Dependencies and prerequisites
   - Success criteria and validation
   - Risk mitigation during implementation

3. **Resource Planning**:
   - Personnel requirements (roles, skills, time)
   - Technology investments needed
   - External consultant requirements
   - Training and certification needs

4. **Project Management**:
   - Milestones and deliverables
   - Progress tracking mechanisms
   - Quality assurance procedures
   - Change management process

5. **Risk Management**:
   - Implementation risks and mitigations
   - Rollback procedures
   - Contingency plans
   - Communication protocols

6. **Success Metrics**:
   - Compliance score targets
   - Risk reduction measurements
   - Process improvement indicators
   - Business value realization

Focus on practical, actionable guidance for {{ framework_config.compliance_focus }}.
Prioritize based on regulatory risk and business impact.
""",
            variables=[
                "framework",
                "framework_config",
                "assessment",
                "target_compliance_score",
                "target_timeline",
                "budget_range",
                "prioritized_gaps",
                "team_size",
                "technical_expertise",
                "budget_constraints",
                "regulatory_deadline",
            ],
            tags=["remediation", "actionable", "project_management"],
        )

        # Regulatory Filing Template
        self.templates["regulatory_filing_formal"] = PromptTemplate(
            id="regulatory_filing_formal",
            name="Formal Regulatory Filing",
            description="Formal compliance report for regulatory submission",
            type=PromptType.REGULATORY_FILING,
            category=PromptCategory.REPORTING,
            framework=None,  # Can be used for any framework
            template="""
You are preparing a formal regulatory compliance filing for {{ framework }}.

FILING REQUIREMENTS:
- Regulatory Body: {{ regulatory_body }}
- Filing Type: {{ filing_type }}
- Compliance Period: {{ compliance_period }}
- Submission Deadline: {{ submission_deadline }}
- Required Attestations: {{ required_attestations | join(', ') }}

COMPLIANCE ASSESSMENT RESULTS:
- Assessment Date: {{ assessment.assessment_date.strftime('%B %d, %Y') }}
- Assessment Period: {{ assessment_period }}
- Overall Compliance Score: {{ assessment.overall_compliance_score }}%
- Methodology: Automated assessment with expert validation
- Assessor: {{ assessor_name }}

COMPLIANCE STATUS BY SECTION:
{% for section, status in section_compliance.items() %}
- {{ section }}: {{ status.compliant }}/{{ status.total }} requirements ({{ (status.compliant/status.total*100)|round(1) }}%)
{% endfor %}

MATERIAL FINDINGS:
{% for finding in material_findings %}
{{ loop.index }}. {{ finding.title }}
   Status: {{ finding.status }}
   Regulatory Impact: {{ finding.regulatory_impact }}
   Remediation: {{ finding.remediation_status }}
   Target Resolution: {{ finding.target_date }}
{% endfor %}

TASK: Generate formal regulatory filing document with:

1. **Executive Certification**:
   - Management attestation of compliance program
   - Responsibility acknowledgment
   - Material changes disclosure

2. **Compliance Summary**:
   - Framework adherence status
   - Assessment methodology explanation
   - Period covered and scope

3. **Detailed Findings**:
   - Section-by-section compliance status
   - Material deficiencies identification
   - Remediation plans and timelines

4. **Supporting Evidence**:
   - Assessment procedures performed
   - Key controls evaluated
   - Evidence examined and conclusions

5. **Management Response**:
   - Actions taken during assessment period
   - Ongoing improvement initiatives
   - Future compliance strategy

6. **Professional Attestation**:
   - Assessor qualifications
   - Independence statement
   - Opinion on compliance status

Use formal regulatory language appropriate for {{ framework }} submissions.
Ensure completeness and accuracy for regulatory review.
Include all required disclosures and attestations.
""",
            variables=[
                "framework",
                "regulatory_body",
                "filing_type",
                "compliance_period",
                "submission_deadline",
                "required_attestations",
                "assessment",
                "assessment_period",
                "assessor_name",
                "section_compliance",
                "material_findings",
            ],
            tags=["regulatory", "formal", "filing", "attestation"],
        )

        # Update Jinja2 loader with templates
        template_dict = {template_id: template.template for template_id, template in self.templates.items()}
        self.jinja_env.loader = jinja2.DictLoader(template_dict)

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get specific prompt template."""
        return self.templates.get(template_id)

    def list_templates(
        self,
        framework: Optional[str] = None,
        prompt_type: Optional[PromptType] = None,
        category: Optional[PromptCategory] = None,
    ) -> List[PromptTemplate]:
        """List templates with optional filtering."""
        templates = list(self.templates.values())

        if framework:
            templates = [t for t in templates if t.framework is None or t.framework == framework]

        if prompt_type:
            templates = [t for t in templates if t.type == prompt_type]

        if category:
            templates = [t for t in templates if t.category == category]

        return templates

    def render_prompt(
        self,
        template_id: str,
        variables: Dict[str, Any],
        framework: Optional[str] = None,
    ) -> str:
        """Render prompt template with provided variables."""
        try:
            template = self.get_template(template_id)
            if not template:
                raise PromptException(f"Template not found: {template_id}")

            # Add framework configuration if available
            if framework and framework in self.framework_configs:
                variables["framework_config"] = self.framework_configs[framework]

            # Validate required variables
            missing_vars = set(template.variables) - set(variables.keys())
            if missing_vars:
                self.logger.warning(f"Missing variables for template {template_id}: {missing_vars}")

            # Render template
            jinja_template = self.jinja_env.get_template(template_id)
            rendered_prompt = jinja_template.render(**variables)

            return rendered_prompt.strip()

        except jinja2.TemplateError as e:
            self.logger.error(f"Template rendering failed: {e}")
            raise PromptException(f"Template rendering failed: {e}")
        except Exception as e:
            self.logger.error(f"Prompt rendering error: {e}")
            raise PromptException(f"Prompt rendering error: {e}")

    def create_gap_analysis_prompt(
        self,
        framework: str,
        assessment: ComplianceAssessment,
        system_evidence: str = "",
    ) -> str:
        """Create gap analysis prompt for specific assessment."""
        try:
            total_requirements = (
                assessment.compliant_requirements
                + assessment.non_compliant_requirements
                + assessment.partially_compliant_requirements
                + assessment.not_assessed_requirements
            )

            variables = {
                "framework": framework,
                "assessment": assessment,
                "total_requirements": total_requirements,
                "gaps": assessment.gaps,
                "system_evidence": system_evidence,
            }

            return self.render_prompt("gap_analysis_comprehensive", variables, framework)

        except Exception as e:
            self.logger.error(f"Failed to create gap analysis prompt: {e}")
            raise PromptException(f"Gap analysis prompt creation failed: {e}")

    def create_requirement_assessment_prompt(
        self,
        framework: str,
        requirement: ComplianceRequirement,
        evidence_items: List[Any],
    ) -> str:
        """Create requirement assessment prompt."""
        try:
            variables = {
                "framework": framework,
                "requirement": requirement,
                "evidence_items": evidence_items,
            }

            return self.render_prompt("requirement_assessment_detailed", variables, framework)

        except Exception as e:
            self.logger.error(f"Failed to create requirement assessment prompt: {e}")
            raise PromptException(f"Requirement assessment prompt creation failed: {e}")

    def create_executive_summary_prompt(
        self,
        framework: str,
        assessment: ComplianceAssessment,
        business_impact: str = "Regulatory compliance and risk mitigation",
    ) -> str:
        """Create executive summary prompt."""
        try:
            critical_gaps = [g for g in assessment.gaps if g.severity == RequirementSeverity.CRITICAL]

            variables = {
                "framework": framework,
                "assessment": assessment,
                "critical_gap_count": len(critical_gaps),
                "critical_gaps": critical_gaps,
                "business_impact": business_impact,
            }

            return self.render_prompt("executive_summary_strategic", variables, framework)

        except Exception as e:
            self.logger.error(f"Failed to create executive summary prompt: {e}")
            raise PromptException(f"Executive summary prompt creation failed: {e}")

    def create_technical_report_prompt(
        self,
        framework: str,
        assessment: ComplianceAssessment,
        system_environment: str = "Production",
        assessment_scope: str = "Full system assessment",
    ) -> str:
        """Create technical report prompt."""
        try:
            total_requirements = (
                assessment.compliant_requirements
                + assessment.non_compliant_requirements
                + assessment.partially_compliant_requirements
                + assessment.not_assessed_requirements
            )

            # Organize gaps by technical complexity
            technical_gaps = [g for g in assessment.gaps if g.remediation_effort in ["medium", "high"]]

            variables = {
                "framework": framework,
                "assessment": assessment,
                "evidence_source_count": 10,  # Would be dynamically calculated
                "system_environment": system_environment,
                "assessment_scope": assessment_scope,
                "total_requirements": total_requirements,
                "requirement_sections": [],  # Would be populated from framework
                "technical_gaps": technical_gaps,
            }

            return self.render_prompt("technical_report_comprehensive", variables, framework)

        except Exception as e:
            self.logger.error(f"Failed to create technical report prompt: {e}")
            raise PromptException(f"Technical report prompt creation failed: {e}")

    def create_risk_evaluation_prompt(
        self,
        framework: str,
        assessment: ComplianceAssessment,
        industry_context: str = "Enterprise",
        regulatory_environment: str = "Regulated industry",
    ) -> str:
        """Create risk evaluation prompt."""
        try:
            critical_gaps = [g for g in assessment.gaps if g.severity == RequirementSeverity.CRITICAL]
            high_priority_gaps = [g for g in assessment.gaps if g.severity == RequirementSeverity.HIGH]

            variables = {
                "framework": framework,
                "assessment": assessment,
                "critical_gap_count": len(critical_gaps),
                "high_priority_gap_count": len(high_priority_gaps),
                "gaps": assessment.gaps,
                "industry_context": industry_context,
                "regulatory_environment": regulatory_environment,
                "business_criticality": "High",
            }

            return self.render_prompt("risk_evaluation_quantitative", variables, framework)

        except Exception as e:
            self.logger.error(f"Failed to create risk evaluation prompt: {e}")
            raise PromptException(f"Risk evaluation prompt creation failed: {e}")

    def create_remediation_plan_prompt(
        self,
        framework: str,
        assessment: ComplianceAssessment,
        target_compliance_score: int = 95,
        target_timeline: int = 6,
        budget_range: str = "Enterprise budget",
    ) -> str:
        """Create remediation plan prompt."""
        try:
            # Sort gaps by priority (severity and impact)
            prioritized_gaps = sorted(
                assessment.gaps,
                key=lambda g: (
                    {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(g.severity.value, 4),
                    {"high": 0, "medium": 1, "low": 2}.get(g.remediation_effort, 3),
                ),
            )

            variables = {
                "framework": framework,
                "assessment": assessment,
                "target_compliance_score": target_compliance_score,
                "target_timeline": target_timeline,
                "budget_range": budget_range,
                "prioritized_gaps": prioritized_gaps,
                "team_size": "Medium (5-10 people)",
                "technical_expertise": "High",
                "budget_constraints": "Moderate",
                "regulatory_deadline": f"{target_timeline} months",
            }

            return self.render_prompt("remediation_plan_actionable", variables, framework)

        except Exception as e:
            self.logger.error(f"Failed to create remediation plan prompt: {e}")
            raise PromptException(f"Remediation plan prompt creation failed: {e}")

    def create_regulatory_filing_prompt(
        self,
        framework: str,
        assessment: ComplianceAssessment,
        regulatory_body: str,
        filing_type: str = "Annual Compliance Report",
    ) -> str:
        """Create regulatory filing prompt."""
        try:
            # Identify material findings (critical and high severity gaps)
            material_findings = []
            for gap in assessment.gaps:
                if gap.severity in [
                    RequirementSeverity.CRITICAL,
                    RequirementSeverity.HIGH,
                ]:
                    material_findings.append(
                        {
                            "title": gap.requirement_title,
                            "status": "Identified",
                            "regulatory_impact": gap.impact_assessment,
                            "remediation_status": "In Progress",
                            "target_date": "Within 90 days",
                        }
                    )

            variables = {
                "framework": framework,
                "regulatory_body": regulatory_body,
                "filing_type": filing_type,
                "compliance_period": "Annual",
                "submission_deadline": "30 days from assessment",
                "required_attestations": [
                    "Management Certification",
                    "Independent Assessment",
                ],
                "assessment": assessment,
                "assessment_period": "Current year",
                "assessor_name": "QBITEL Compliance System",
                "section_compliance": {},  # Would be populated from framework
                "material_findings": material_findings,
            }

            return self.render_prompt("regulatory_filing_formal", variables, framework)

        except Exception as e:
            self.logger.error(f"Failed to create regulatory filing prompt: {e}")
            raise PromptException(f"Regulatory filing prompt creation failed: {e}")

    def add_custom_template(self, template: PromptTemplate) -> None:
        """Add custom prompt template."""
        try:
            # Validate template
            if not template.id or not template.template:
                raise PromptException("Template ID and template content are required")

            # Add to templates
            self.templates[template.id] = template

            # Update Jinja2 loader
            template_dict = {template_id: tmpl.template for template_id, tmpl in self.templates.items()}
            self.jinja_env.loader = jinja2.DictLoader(template_dict)

            self.logger.info(f"Custom template added: {template.id}")

        except Exception as e:
            self.logger.error(f"Failed to add custom template: {e}")
            raise PromptException(f"Custom template addition failed: {e}")

    def validate_prompt_variables(self, template_id: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Validate prompt variables against template requirements."""
        template = self.get_template(template_id)
        if not template:
            raise PromptException(f"Template not found: {template_id}")

        validation_result = {
            "valid": True,
            "missing_variables": [],
            "extra_variables": [],
            "warnings": [],
        }

        # Check for missing required variables
        provided_vars = set(variables.keys())
        required_vars = set(template.variables)

        missing = required_vars - provided_vars
        if missing:
            validation_result["valid"] = False
            validation_result["missing_variables"] = list(missing)

        # Check for extra variables (not necessarily an error)
        extra = provided_vars - required_vars
        if extra:
            validation_result["extra_variables"] = list(extra)
            validation_result["warnings"].append(f"Extra variables provided: {extra}")

        return validation_result

    def get_framework_config(self, framework: str) -> Dict[str, Any]:
        """Get framework-specific configuration."""
        return self.framework_configs.get(
            framework,
            {
                "emphasis": "general compliance",
                "key_areas": ["security", "governance", "risk management"],
                "regulatory_language": "professional and formal",
                "compliance_focus": "regulatory compliance",
            },
        )

    def export_templates(self, format: str = "json") -> str:
        """Export all templates for backup or sharing."""
        try:
            templates_data = {}
            for template_id, template in self.templates.items():
                templates_data[template_id] = {
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "type": template.type.value,
                    "category": template.category.value,
                    "framework": template.framework,
                    "template": template.template,
                    "variables": template.variables,
                    "version": template.version,
                    "tags": template.tags,
                }

            if format.lower() == "json":
                return json.dumps(templates_data, indent=2)
            else:
                raise PromptException(f"Unsupported export format: {format}")

        except Exception as e:
            self.logger.error(f"Template export failed: {e}")
            raise PromptException(f"Template export failed: {e}")
