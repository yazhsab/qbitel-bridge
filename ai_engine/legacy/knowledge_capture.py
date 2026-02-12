"""
QBITEL Engine - Tribal Knowledge Capture

System for capturing, formalizing, and leveraging tribal knowledge about legacy systems.
Converts expert insights into structured, searchable, and actionable knowledge base.
"""

import logging
import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re

from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest
from ..core.config import Config
from ..core.exceptions import QbitelAIException
from .models import FormalizedKnowledge, LegacySystemContext, SystemType


class KnowledgeSourceType(str, Enum):
    """Types of knowledge sources."""

    EXPERT_INTERVIEW = "expert_interview"
    INCIDENT_REPORT = "incident_report"
    MAINTENANCE_LOG = "maintenance_log"
    DOCUMENTATION = "documentation"
    OBSERVATION = "observation"
    VENDOR_SUPPORT = "vendor_support"


class KnowledgeValidationStatus(str, Enum):
    """Knowledge validation status."""

    PENDING = "pending"
    VALIDATED = "validated"
    DISPUTED = "disputed"
    DEPRECATED = "deprecated"
    UNDER_REVIEW = "under_review"


@dataclass
class ExpertProfile:
    """Profile of a domain expert."""

    expert_id: str
    name: str
    role: str
    experience_years: int
    specializations: List[str] = field(default_factory=list)
    systems_experience: List[str] = field(default_factory=list)
    credibility_score: float = 1.0
    contributions_count: int = 0
    validation_success_rate: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeExtractionSession:
    """Session for extracting knowledge from experts."""

    session_id: str
    expert_id: str
    system_id: Optional[str]
    session_type: str  # "structured_interview", "incident_debrief", "maintenance_review"
    start_time: datetime
    end_time: Optional[datetime] = None
    raw_input: List[str] = field(default_factory=list)
    extracted_knowledge: List[str] = field(default_factory=list)
    validation_notes: str = ""
    confidence_score: float = 0.0
    status: str = "in_progress"
    metadata: Dict[str, Any] = field(default_factory=dict)


class TribalKnowledgeCapture:
    """
    Capture and formalize expert knowledge about legacy systems.

    This class provides sophisticated knowledge extraction, validation,
    and formalization capabilities using LLM-powered analysis.
    """

    def __init__(self, config: Config, llm_service: UnifiedLLMService):
        """Initialize tribal knowledge capture system."""
        self.config = config
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)

        # Knowledge storage
        self.knowledge_base: Dict[str, FormalizedKnowledge] = {}
        self.expert_profiles: Dict[str, ExpertProfile] = {}
        self.active_sessions: Dict[str, KnowledgeExtractionSession] = {}

        # Knowledge extraction templates
        self.extraction_prompts = {
            "structured_interview": """
            You are conducting a structured interview with a legacy system expert to extract their tribal knowledge.
            
            Expert Profile:
            - Name: {expert_name}
            - Role: {expert_role}
            - Experience: {experience_years} years
            - Specializations: {specializations}
            
            System Context:
            {system_context}
            
            Expert Input:
            {expert_input}
            
            Please extract and formalize this knowledge into structured components:
            
            1. SYSTEM BEHAVIORS:
               - Normal operating patterns
               - Warning signs and early indicators
               - Failure modes and their characteristics
               - Performance patterns and cycles
            
            2. FAILURE INDICATORS:
               - Specific symptoms that indicate problems
               - Leading indicators that predict failures
               - Subtle signs that are easy to miss
               - False positives to be aware of
            
            3. MAINTENANCE PROCEDURES:
               - Preventive maintenance tasks
               - Diagnostic procedures
               - Repair and recovery procedures
               - Best practices and timing
            
            4. SYSTEM DEPENDENCIES:
               - Critical system relationships
               - Cascade failure patterns
               - External dependencies
               - Infrastructure requirements
            
            5. TROUBLESHOOTING KNOWLEDGE:
               - Step-by-step diagnostic procedures
               - Common root causes
               - Effective workarounds
               - Tools and techniques used
            
            6. HISTORICAL CONTEXT:
               - Past incidents and resolutions
               - Lessons learned from failures
               - Evolution of the system over time
               - Vendor-specific quirks and issues
            
            Format your response with clear sections and actionable information.
            Be specific about symptoms, thresholds, and procedures.
            """,
            "incident_debrief": """
            You are debriefing a recent incident with an expert to extract lessons learned.
            
            Incident Context:
            {incident_context}
            
            Expert Analysis:
            {expert_input}
            
            Extract the following knowledge:
            
            1. ROOT CAUSE ANALYSIS:
               - Primary cause of the incident
               - Contributing factors
               - Why existing monitoring missed it
            
            2. EARLY WARNING SIGNS:
               - Indicators that could have predicted this incident
               - Timeline of warning signs
               - Monitoring gaps identified
            
            3. RESOLUTION PROCEDURES:
               - Steps taken to resolve the incident
               - What worked and what didn't
               - Time-sensitive actions required
            
            4. PREVENTION STRATEGIES:
               - How to prevent similar incidents
               - Monitoring improvements needed
               - Process changes recommended
            
            5. SYSTEM INSIGHTS:
               - New understanding of system behavior
               - Hidden dependencies discovered
               - Performance characteristics revealed
            
            Focus on actionable insights that can prevent future incidents.
            """,
            "maintenance_review": """
            You are reviewing maintenance procedures and experiences with an expert.
            
            Maintenance Context:
            {maintenance_context}
            
            Expert Experience:
            {expert_input}
            
            Extract maintenance knowledge:
            
            1. MAINTENANCE PROCEDURES:
               - Detailed step-by-step procedures
               - Required tools and resources
               - Safety considerations
               - Time estimates and scheduling
            
            2. PREVENTIVE MAINTENANCE:
               - Optimal maintenance intervals
               - Key inspection points
               - Replacement schedules
               - Performance benchmarks
            
            3. DIAGNOSTIC TECHNIQUES:
               - How to assess system health
               - Performance testing procedures
               - Health check methodologies
               - Baseline establishment
            
            4. BEST PRACTICES:
               - What works best for this system type
               - Common mistakes to avoid
               - Optimization opportunities
               - Resource requirements
            
            5. VENDOR CONSIDERATIONS:
               - Vendor-specific requirements
               - Support escalation procedures
               - Warranty and service considerations
               - Parts availability and alternatives
            
            Emphasize practical, tested procedures that improve system reliability.
            """,
            "general_formalization": """
            You are formalizing general tribal knowledge about legacy systems.
            
            System Type: {system_type}
            Knowledge Source: {knowledge_source}
            
            Raw Knowledge Input:
            {raw_knowledge}
            
            Formalize this knowledge into a structured format:
            
            1. KNOWLEDGE CLASSIFICATION:
               - Type of knowledge (behavioral, procedural, diagnostic, etc.)
               - Applicability (system-specific or general)
               - Confidence level and validation status
            
            2. CONDITIONS AND TRIGGERS:
               - When this knowledge applies
               - Triggering conditions or symptoms
               - Environmental factors
            
            3. ACTIONABLE PROCEDURES:
               - Step-by-step procedures
               - Decision points and alternatives
               - Success criteria
            
            4. VALIDATION CRITERIA:
               - How to verify this knowledge
               - Success indicators
               - Failure modes of the procedure itself
            
            5. RELATIONSHIPS:
               - Related knowledge areas
               - Dependencies on other procedures
               - Conflicts with other approaches
            
            Structure the knowledge for easy search, validation, and application.
            """,
        }

        # Validation prompts
        self.validation_prompts = {
            "cross_validate": """
            You are cross-validating tribal knowledge from multiple experts.
            
            Knowledge Item:
            {knowledge_item}
            
            Expert Opinions:
            {expert_opinions}
            
            Perform cross-validation:
            
            1. CONSISTENCY ANALYSIS:
               - Areas of agreement between experts
               - Conflicting information identified
               - Confidence level of each aspect
            
            2. VALIDATION RECOMMENDATIONS:
               - Which aspects are well-validated
               - What needs further verification
               - Suggested validation procedures
            
            3. SYNTHESIS:
               - Consolidated best practices
               - Highlighted uncertainties
               - Recommended next steps
            
            Provide a confidence score (0-100%) for the overall knowledge item.
            """,
            "practical_validation": """
            You are validating tribal knowledge against practical system behavior.
            
            Knowledge Claim:
            {knowledge_claim}
            
            System Data Evidence:
            {system_data}
            
            Historical Incidents:
            {historical_incidents}
            
            Validate the knowledge:
            
            1. EVIDENCE ANALYSIS:
               - Does system data support the claim?
               - Are there counter-examples?
               - Statistical significance of evidence
            
            2. PRACTICAL APPLICABILITY:
               - Has this knowledge been successfully applied?
               - What are the success/failure rates?
               - Under what conditions does it work best?
            
            3. REFINEMENT SUGGESTIONS:
               - How can the knowledge be improved?
               - What additional conditions should be noted?
               - Are there important caveats missing?
            
            Rate the validation confidence (0-100%) and suggest improvements.
            """,
        }

        self.logger.info("TribalKnowledgeCapture initialized")

    async def start_expert_session(
        self,
        expert_id: str,
        session_type: str,
        system_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a knowledge extraction session with an expert."""

        session_id = str(uuid.uuid4())

        session = KnowledgeExtractionSession(
            session_id=session_id,
            expert_id=expert_id,
            system_id=system_id,
            session_type=session_type,
            start_time=datetime.now(),
            metadata=context or {},
        )

        self.active_sessions[session_id] = session

        self.logger.info(f"Started knowledge extraction session {session_id} with expert {expert_id}")
        return session_id

    async def capture_expert_input(
        self,
        session_id: str,
        expert_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Capture and process expert input during a session."""

        if session_id not in self.active_sessions:
            raise QbitelAIException(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        session.raw_input.append(expert_input)

        try:
            # Get expert profile
            expert = self.expert_profiles.get(session.expert_id)
            if not expert:
                expert = ExpertProfile(
                    expert_id=session.expert_id,
                    name="Unknown Expert",
                    role="Domain Expert",
                    experience_years=10,
                )

            # Extract and formalize knowledge
            formalized_knowledge = await self._extract_knowledge_from_input(
                expert_input=expert_input,
                session=session,
                expert=expert,
                context=context,
            )

            session.extracted_knowledge.append(formalized_knowledge["knowledge_id"])

            # Update session confidence
            session.confidence_score = self._calculate_session_confidence(session)

            return {
                "session_id": session_id,
                "knowledge_extracted": formalized_knowledge,
                "session_confidence": session.confidence_score,
                "status": "success",
            }

        except Exception as e:
            self.logger.error(f"Failed to process expert input in session {session_id}: {e}")
            return {"session_id": session_id, "error": str(e), "status": "error"}

    async def _extract_knowledge_from_input(
        self,
        expert_input: str,
        session: KnowledgeExtractionSession,
        expert: ExpertProfile,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract structured knowledge from expert input using LLM."""

        # Select appropriate extraction prompt
        prompt_template = self.extraction_prompts.get(session.session_type, self.extraction_prompts["general_formalization"])

        # Prepare context for the prompt
        system_context = ""
        if session.system_id:
            # In production, load system context from database
            system_context = f"System ID: {session.system_id}"

        # Format the prompt
        if session.session_type == "structured_interview":
            prompt = prompt_template.format(
                expert_name=expert.name,
                expert_role=expert.role,
                experience_years=expert.experience_years,
                specializations=", ".join(expert.specializations),
                system_context=system_context,
                expert_input=expert_input,
            )
        elif session.session_type == "incident_debrief":
            incident_context = context.get("incident_context", "No incident context provided") if context else ""
            prompt = prompt_template.format(incident_context=incident_context, expert_input=expert_input)
        elif session.session_type == "maintenance_review":
            maintenance_context = context.get("maintenance_context", "No maintenance context provided") if context else ""
            prompt = prompt_template.format(maintenance_context=maintenance_context, expert_input=expert_input)
        else:
            system_type = context.get("system_type", "unknown") if context else "unknown"
            knowledge_source = session.session_type
            prompt = prompt_template.format(
                system_type=system_type,
                knowledge_source=knowledge_source,
                raw_knowledge=expert_input,
            )

        # Create LLM request
        llm_request = LLMRequest(
            prompt=prompt,
            feature_domain="legacy_whisperer",
            max_tokens=3000,
            temperature=0.1,  # Low temperature for consistent extraction
        )

        # Process with LLM
        response = await self.llm_service.process_request(llm_request)

        # Parse the structured response
        structured_knowledge = self._parse_knowledge_extraction_response(response.content, session, expert)

        # Create formalized knowledge object
        knowledge = FormalizedKnowledge(
            knowledge_id=str(uuid.uuid4()),
            system_id=session.system_id,
            knowledge_type=self._determine_knowledge_type(structured_knowledge),
            title=structured_knowledge.get("title", f"Knowledge from {expert.name}"),
            description=structured_knowledge.get("description", ""),
            detailed_explanation=response.content,
            behaviors=structured_knowledge.get("behaviors", {}),
            failure_indicators=structured_knowledge.get("failure_indicators", []),
            maintenance_procedures=structured_knowledge.get("maintenance_procedures", []),
            dependencies=structured_knowledge.get("dependencies", []),
            conditions=structured_knowledge.get("conditions", []),
            triggers=structured_knowledge.get("triggers", []),
            symptoms=structured_knowledge.get("symptoms", []),
            recommended_actions=structured_knowledge.get("recommended_actions", []),
            troubleshooting_steps=structured_knowledge.get("troubleshooting_steps", []),
            workarounds=structured_knowledge.get("workarounds", []),
            confidence_score=response.confidence,
            source_expert=expert.expert_id,
            source_type=session.session_type,
            tags=self._extract_tags_from_knowledge(structured_knowledge),
            metadata={
                "session_id": session.session_id,
                "extraction_time": datetime.now().isoformat(),
                "llm_provider": response.provider,
                "processing_time": response.processing_time,
            },
        )

        # Store the knowledge
        self.knowledge_base[knowledge.knowledge_id] = knowledge

        # Update expert profile
        expert.contributions_count += 1

        self.logger.info(f"Extracted knowledge {knowledge.knowledge_id} from expert {expert.expert_id}")

        return {
            "knowledge_id": knowledge.knowledge_id,
            "structured_knowledge": structured_knowledge,
            "confidence": response.confidence,
        }

    def _parse_knowledge_extraction_response(
        self,
        response_content: str,
        session: KnowledgeExtractionSession,
        expert: ExpertProfile,
    ) -> Dict[str, Any]:
        """Parse LLM response into structured knowledge components."""

        # This is a simplified parser - in production, use more sophisticated NLP
        structured = {
            "title": f"Knowledge from {expert.name} - {session.session_type}",
            "description": "",
            "behaviors": {},
            "failure_indicators": [],
            "maintenance_procedures": [],
            "dependencies": [],
            "conditions": [],
            "triggers": [],
            "symptoms": [],
            "recommended_actions": [],
            "troubleshooting_steps": [],
            "workarounds": [],
        }

        # Extract sections using regex patterns
        sections = {
            "behaviors": [r"SYSTEM BEHAVIORS?:?", r"BEHAVIORS?:?"],
            "failure_indicators": [
                r"FAILURE INDICATORS?:?",
                r"WARNING SIGNS?:?",
                r"EARLY WARNING SIGNS?:?",
            ],
            "maintenance_procedures": [r"MAINTENANCE PROCEDURES?:?", r"PROCEDURES?:?"],
            "dependencies": [r"DEPENDENCIES:?", r"SYSTEM DEPENDENCIES:?"],
            "recommended_actions": [r"RECOMMENDED ACTIONS?:?", r"ACTIONS?:?"],
            "troubleshooting_steps": [
                r"TROUBLESHOOTING:?",
                r"DIAGNOSTIC:?",
                r"TROUBLESHOOTING KNOWLEDGE:?",
            ],
            "conditions": [r"CONDITIONS?:?", r"TRIGGERING CONDITIONS?:?"],
            "symptoms": [r"SYMPTOMS?:?", r"INDICATORS?:?"],
        }

        lines = response_content.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line starts a new section
            section_found = False
            for section_name, patterns in sections.items():
                for pattern in patterns:
                    if re.match(pattern, line.upper()):
                        current_section = section_name
                        section_found = True
                        break
                if section_found:
                    break

            # If we're in a section and this isn't a section header, add to current section
            if current_section and not section_found:
                # Extract list items
                if line.startswith(("-", "•", "*")) or re.match(r"^\d+\.", line):
                    item = re.sub(r"^[-•*\d\.\s]+", "", line).strip()
                    if item:
                        if current_section in structured:
                            if isinstance(structured[current_section], list):
                                structured[current_section].append(item)
                            elif isinstance(structured[current_section], dict):
                                # For behaviors, try to extract key-value pairs
                                if ":" in item:
                                    key, value = item.split(":", 1)
                                    structured[current_section][key.strip()] = value.strip()

        # Extract title and description from the beginning of the response
        first_lines = response_content.split("\n")[:5]
        for line in first_lines:
            if line.strip() and not any(keyword in line.upper() for keyword in ["SYSTEM", "FAILURE", "MAINTENANCE"]):
                if not structured["description"]:
                    structured["description"] = line.strip()
                break

        return structured

    def _determine_knowledge_type(self, structured_knowledge: Dict[str, Any]) -> str:
        """Determine the type of knowledge based on content."""

        if structured_knowledge.get("behaviors"):
            return "behavioral"
        elif structured_knowledge.get("maintenance_procedures"):
            return "maintenance"
        elif structured_knowledge.get("troubleshooting_steps"):
            return "troubleshooting"
        elif structured_knowledge.get("failure_indicators"):
            return "diagnostic"
        else:
            return "general"

    def _extract_tags_from_knowledge(self, structured_knowledge: Dict[str, Any]) -> List[str]:
        """Extract relevant tags from structured knowledge."""
        tags = set()

        # Add tags based on content
        if structured_knowledge.get("behaviors"):
            tags.add("behavioral")
        if structured_knowledge.get("failure_indicators"):
            tags.add("failure_indicators")
        if structured_knowledge.get("maintenance_procedures"):
            tags.add("maintenance")
        if structured_knowledge.get("troubleshooting_steps"):
            tags.add("troubleshooting")

        # Add tags from common keywords in content
        all_text = " ".join([str(v) for v in structured_knowledge.values() if isinstance(v, (str, list))]).lower()

        keyword_tags = {
            "performance": "performance",
            "memory": "memory",
            "cpu": "cpu",
            "disk": "storage",
            "network": "network",
            "database": "database",
            "mainframe": "mainframe",
            "cobol": "cobol",
            "backup": "backup",
            "security": "security",
        }

        for keyword, tag in keyword_tags.items():
            if keyword in all_text:
                tags.add(tag)

        return list(tags)

    def _calculate_session_confidence(self, session: KnowledgeExtractionSession) -> float:
        """Calculate overall confidence for a session."""
        if not session.extracted_knowledge:
            return 0.0

        # Get confidence scores from extracted knowledge
        confidences = []
        for knowledge_id in session.extracted_knowledge:
            knowledge = self.knowledge_base.get(knowledge_id)
            if knowledge:
                confidences.append(knowledge.confidence_score)

        if not confidences:
            return 0.0

        # Calculate weighted average (more recent knowledge gets higher weight)
        weights = [1.0 + (i * 0.1) for i in range(len(confidences))]
        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        weight_sum = sum(weights)

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    async def finalize_session(self, session_id: str, validation_notes: Optional[str] = None) -> Dict[str, Any]:
        """Finalize a knowledge extraction session."""

        if session_id not in self.active_sessions:
            raise QbitelAIException(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        session.end_time = datetime.now()
        session.status = "completed"

        if validation_notes:
            session.validation_notes = validation_notes

        # Calculate final session metrics
        session_duration = (session.end_time - session.start_time).total_seconds()
        knowledge_count = len(session.extracted_knowledge)

        # Update expert profile
        expert = self.expert_profiles.get(session.expert_id)
        if expert:
            # Update credibility based on session quality
            if session.confidence_score > 0.8:
                expert.credibility_score = min(expert.credibility_score + 0.1, 2.0)
            elif session.confidence_score < 0.5:
                expert.credibility_score = max(expert.credibility_score - 0.05, 0.1)

        # Remove from active sessions
        del self.active_sessions[session_id]

        self.logger.info(f"Finalized session {session_id}: {knowledge_count} knowledge items extracted")

        return {
            "session_id": session_id,
            "duration_seconds": session_duration,
            "knowledge_extracted": knowledge_count,
            "confidence_score": session.confidence_score,
            "status": "completed",
        }

    async def validate_knowledge(
        self,
        knowledge_id: str,
        validation_data: Optional[Dict[str, Any]] = None,
        cross_reference_experts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate existing knowledge using additional data or expert cross-reference."""

        if knowledge_id not in self.knowledge_base:
            raise QbitelAIException(f"Knowledge {knowledge_id} not found")

        knowledge = self.knowledge_base[knowledge_id]

        try:
            validation_result = {
                "validation_score": 0.0,
                "validation_notes": "",
                "status": "pending",
            }

            # Cross-validate with other experts if specified
            if cross_reference_experts:
                cross_validation = await self._cross_validate_with_experts(knowledge, cross_reference_experts)
                validation_result.update(cross_validation)

            # Validate against system data if provided
            if validation_data:
                practical_validation = await self._validate_against_system_data(knowledge, validation_data)
                validation_result.update(practical_validation)

            # Update knowledge with validation results
            knowledge.validation_count += 1
            knowledge.last_validated = datetime.now()

            if validation_result["validation_score"] > 0.8:
                knowledge.confidence_score = min(knowledge.confidence_score + 0.1, 1.0)
            elif validation_result["validation_score"] < 0.4:
                knowledge.confidence_score = max(knowledge.confidence_score - 0.1, 0.0)

            self.logger.info(f"Validated knowledge {knowledge_id} with score {validation_result['validation_score']:.2f}")

            return validation_result

        except Exception as e:
            self.logger.error(f"Knowledge validation failed for {knowledge_id}: {e}")
            return {
                "validation_score": 0.0,
                "validation_notes": f"Validation failed: {e}",
                "status": "error",
            }

    async def _cross_validate_with_experts(self, knowledge: FormalizedKnowledge, expert_ids: List[str]) -> Dict[str, Any]:
        """Cross-validate knowledge with multiple experts."""

        # Collect expert opinions (in production, this would involve actual expert consultation)
        expert_opinions = []
        for expert_id in expert_ids:
            expert = self.expert_profiles.get(expert_id)
            if expert:
                # Simulate expert opinion based on their profile and experience
                opinion = {
                    "expert_id": expert_id,
                    "credibility": expert.credibility_score,
                    "agreement_score": 0.8,  # Simplified - would be actual expert input
                    "notes": f"Expert {expert.name} review",
                }
                expert_opinions.append(opinion)

        if not expert_opinions:
            return {
                "validation_score": 0.5,
                "validation_notes": "No expert opinions available",
            }

        # Use LLM to analyze cross-validation
        prompt = self.validation_prompts["cross_validate"].format(
            knowledge_item=knowledge.detailed_explanation,
            expert_opinions=json.dumps(expert_opinions, indent=2),
        )

        llm_request = LLMRequest(
            prompt=prompt,
            feature_domain="legacy_whisperer",
            max_tokens=1500,
            temperature=0.1,
        )

        response = await self.llm_service.process_request(llm_request)

        # Parse validation result
        validation_score = self._extract_validation_score(response.content)

        return {
            "validation_score": validation_score,
            "validation_notes": response.content,
            "cross_validation_experts": expert_ids,
        }

    async def _validate_against_system_data(
        self, knowledge: FormalizedKnowledge, system_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate knowledge against actual system data."""

        prompt = self.validation_prompts["practical_validation"].format(
            knowledge_claim=knowledge.detailed_explanation,
            system_data=json.dumps(system_data, indent=2),
            historical_incidents=json.dumps(system_data.get("incidents", []), indent=2),
        )

        llm_request = LLMRequest(
            prompt=prompt,
            feature_domain="legacy_whisperer",
            max_tokens=1500,
            temperature=0.1,
        )

        response = await self.llm_service.process_request(llm_request)

        validation_score = self._extract_validation_score(response.content)

        return {
            "practical_validation_score": validation_score,
            "practical_validation_notes": response.content,
        }

    def _extract_validation_score(self, content: str) -> float:
        """Extract validation score from LLM response."""
        import re

        # Look for confidence/validation score patterns
        patterns = [
            r"confidence[:\s]+(\d+)%",
            r"validation[:\s]+(\d+)%",
            r"score[:\s]+(\d+)%",
            r"(\d+)%\s+confidence",
        ]

        for pattern in patterns:
            match = re.search(pattern, content.lower())
            if match:
                return float(match.group(1)) / 100.0

        # If no explicit score found, estimate based on keywords
        positive_keywords = [
            "validated",
            "confirmed",
            "accurate",
            "correct",
            "reliable",
        ]
        negative_keywords = [
            "disputed",
            "incorrect",
            "unreliable",
            "questionable",
            "inconsistent",
        ]

        content_lower = content.lower()
        positive_count = sum(1 for keyword in positive_keywords if keyword in content_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in content_lower)

        if positive_count > negative_count:
            return 0.7
        elif negative_count > positive_count:
            return 0.3
        else:
            return 0.5

    def search_knowledge(
        self,
        query: str,
        system_id: Optional[str] = None,
        knowledge_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[FormalizedKnowledge]:
        """Search knowledge base with filters."""

        results = []
        query_lower = query.lower()

        for knowledge in self.knowledge_base.values():
            # Apply filters
            if system_id and knowledge.system_id != system_id:
                continue

            if knowledge_type and knowledge.knowledge_type != knowledge_type:
                continue

            if knowledge.confidence_score < min_confidence:
                continue

            if tags and not any(tag in knowledge.tags for tag in tags):
                continue

            # Check if query matches content
            searchable_text = " ".join(
                [
                    knowledge.title,
                    knowledge.description,
                    knowledge.detailed_explanation,
                    " ".join(knowledge.failure_indicators),
                    " ".join(knowledge.recommended_actions),
                    " ".join(knowledge.tags),
                ]
            ).lower()

            if query_lower in searchable_text:
                results.append(knowledge)

        # Sort by relevance (confidence score and last accessed)
        results.sort(
            key=lambda k: (k.confidence_score, k.last_accessed or datetime.min),
            reverse=True,
        )

        return results

    def get_expert_statistics(self, expert_id: str) -> Dict[str, Any]:
        """Get statistics for an expert."""
        expert = self.expert_profiles.get(expert_id)
        if not expert:
            return {}

        # Count expert's contributions
        contributions = [k for k in self.knowledge_base.values() if k.source_expert == expert_id]

        avg_confidence = sum(k.confidence_score for k in contributions) / len(contributions) if contributions else 0.0

        return {
            "expert_id": expert_id,
            "name": expert.name,
            "experience_years": expert.experience_years,
            "specializations": expert.specializations,
            "credibility_score": expert.credibility_score,
            "total_contributions": len(contributions),
            "average_confidence": avg_confidence,
            "validation_success_rate": expert.validation_success_rate,
            "contribution_trend": self._calculate_contribution_trend(expert_id),
        }

    def _calculate_contribution_trend(self, expert_id: str) -> str:
        """Calculate the trend of expert contributions."""
        contributions = [k for k in self.knowledge_base.values() if k.source_expert == expert_id]

        if len(contributions) < 2:
            return "insufficient_data"

        # Sort by creation date
        contributions.sort(key=lambda k: k.created_at)

        # Compare recent vs older contributions
        mid_point = len(contributions) // 2
        older_avg = sum(k.confidence_score for k in contributions[:mid_point]) / mid_point
        recent_avg = sum(k.confidence_score for k in contributions[mid_point:]) / (len(contributions) - mid_point)

        if recent_avg > older_avg + 0.1:
            return "improving"
        elif recent_avg < older_avg - 0.1:
            return "declining"
        else:
            return "stable"

    def register_expert(self, expert: ExpertProfile) -> None:
        """Register a new expert in the system."""
        self.expert_profiles[expert.expert_id] = expert
        self.logger.info(f"Registered expert {expert.expert_id}: {expert.name}")

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the knowledge base."""
        total_knowledge = len(self.knowledge_base)

        # Knowledge by type
        type_counts = {}
        for knowledge in self.knowledge_base.values():
            type_counts[knowledge.knowledge_type] = type_counts.get(knowledge.knowledge_type, 0) + 1

        # Confidence distribution
        high_confidence = sum(1 for k in self.knowledge_base.values() if k.confidence_score > 0.8)
        medium_confidence = sum(1 for k in self.knowledge_base.values() if 0.5 < k.confidence_score <= 0.8)
        low_confidence = sum(1 for k in self.knowledge_base.values() if k.confidence_score <= 0.5)

        # Recent activity
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_additions = sum(1 for k in self.knowledge_base.values() if k.created_at > recent_cutoff)

        return {
            "total_knowledge_items": total_knowledge,
            "knowledge_by_type": type_counts,
            "confidence_distribution": {
                "high": high_confidence,
                "medium": medium_confidence,
                "low": low_confidence,
            },
            "recent_additions_30_days": recent_additions,
            "total_experts": len(self.expert_profiles),
            "active_sessions": len(self.active_sessions),
        }
