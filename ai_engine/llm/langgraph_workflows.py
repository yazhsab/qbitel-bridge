"""
QBITEL - LangGraph Workflow Orchestration
State machine-based agent workflows with checkpointing and human-in-the-loop support.

This module provides:
- Pre-built workflow templates for common QBITEL tasks
- State management with persistence
- Human-in-the-loop patterns
- Workflow visualization support
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Callable, TypedDict, Annotated, Literal, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# LangGraph imports with fallback
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"
    MemorySaver = None
    ToolNode = None

# LangChain imports with fallback
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    HumanMessage = None
    AIMessage = None
    SystemMessage = None
    BaseMessage = None
    tool = None

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
WORKFLOW_EXECUTION_COUNTER = Counter(
    "qbitel_workflow_executions_total",
    "Total workflow executions",
    ["workflow_type", "status"]
)
WORKFLOW_DURATION = Histogram(
    "qbitel_workflow_duration_seconds",
    "Workflow execution duration",
    ["workflow_type"]
)
WORKFLOW_STEP_COUNTER = Counter(
    "qbitel_workflow_steps_total",
    "Total workflow steps executed",
    ["workflow_type", "step_name"]
)


# =============================================================================
# State Definitions
# =============================================================================

class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowCheckpoint:
    """Checkpoint for workflow state persistence."""
    workflow_id: str
    step_name: str
    state: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProtocolAnalysisState(TypedDict, total=False):
    """State for protocol analysis workflow."""
    # Input
    protocol_data: str
    protocol_hint: Optional[str]

    # Processing state
    messages: List[Dict[str, Any]]
    current_step: str

    # Analysis results
    classification_result: Optional[Dict[str, Any]]
    field_analysis: Optional[Dict[str, Any]]
    security_assessment: Optional[Dict[str, Any]]

    # Output
    final_report: Optional[Dict[str, Any]]
    confidence_score: float

    # Workflow metadata
    workflow_id: str
    status: str
    error: Optional[str]
    requires_human_review: bool


class SecurityOrchestrationState(TypedDict, total=False):
    """State for security orchestration workflow."""
    # Input
    threat_data: Dict[str, Any]
    severity: str

    # Processing state
    messages: List[Dict[str, Any]]
    current_step: str

    # Analysis results
    threat_classification: Optional[Dict[str, Any]]
    impact_assessment: Optional[Dict[str, Any]]
    mitigation_plan: Optional[Dict[str, Any]]

    # Actions taken
    actions_executed: List[Dict[str, Any]]
    actions_pending: List[Dict[str, Any]]

    # Output
    incident_report: Optional[Dict[str, Any]]

    # Workflow metadata
    workflow_id: str
    status: str
    error: Optional[str]
    requires_human_approval: bool


class ComplianceCheckState(TypedDict, total=False):
    """State for compliance checking workflow."""
    # Input
    configuration: Dict[str, Any]
    frameworks: List[str]

    # Processing state
    messages: List[Dict[str, Any]]
    current_step: str

    # Results per framework
    framework_results: Dict[str, Dict[str, Any]]

    # Aggregated results
    compliance_gaps: List[Dict[str, Any]]
    remediation_steps: List[Dict[str, Any]]

    # Output
    compliance_report: Optional[Dict[str, Any]]
    overall_score: float

    # Workflow metadata
    workflow_id: str
    status: str
    error: Optional[str]


# =============================================================================
# Workflow Manager
# =============================================================================

class WorkflowManager:
    """
    Manages LangGraph workflows with checkpointing and persistence.

    Features:
    - Pre-built workflow templates
    - State persistence with checkpoints
    - Human-in-the-loop support
    - Workflow visualization
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Workflow registry
        self.workflows: Dict[str, Any] = {}
        self.workflow_states: Dict[str, Dict[str, Any]] = {}

        # Checkpointer for state persistence
        self.checkpointer = None
        if LANGGRAPH_AVAILABLE and MemorySaver:
            self.checkpointer = MemorySaver()

        # Human-in-the-loop callbacks
        self.human_approval_callbacks: Dict[str, Callable] = {}
        self.human_input_callbacks: Dict[str, Callable] = {}

        # Initialize built-in workflows
        self._initialize_workflows()

    def _initialize_workflows(self) -> None:
        """Initialize built-in workflow templates."""
        if not LANGGRAPH_AVAILABLE:
            self.logger.warning("LangGraph not available. Workflows will use fallback implementation.")
            return

        # Register built-in workflows
        self._register_protocol_analysis_workflow()
        self._register_security_orchestration_workflow()
        self._register_compliance_check_workflow()

        self.logger.info(f"Initialized {len(self.workflows)} workflow templates")

    # =========================================================================
    # Protocol Analysis Workflow
    # =========================================================================

    def _register_protocol_analysis_workflow(self) -> None:
        """Register the protocol analysis workflow."""
        if not LANGGRAPH_AVAILABLE:
            return

        # Define workflow graph
        workflow = StateGraph(ProtocolAnalysisState)

        # Add nodes
        workflow.add_node("classify_protocol", self._classify_protocol_node)
        workflow.add_node("analyze_fields", self._analyze_fields_node)
        workflow.add_node("assess_security", self._assess_security_node)
        workflow.add_node("human_review", self._human_review_node)
        workflow.add_node("generate_report", self._generate_report_node)

        # Define edges
        workflow.set_entry_point("classify_protocol")

        workflow.add_edge("classify_protocol", "analyze_fields")
        workflow.add_edge("analyze_fields", "assess_security")

        # Conditional edge for human review
        workflow.add_conditional_edges(
            "assess_security",
            self._should_require_human_review,
            {
                "human_review": "human_review",
                "generate_report": "generate_report"
            }
        )

        workflow.add_edge("human_review", "generate_report")
        workflow.add_edge("generate_report", END)

        # Compile with checkpointer
        compiled = workflow.compile(checkpointer=self.checkpointer)
        self.workflows["protocol_analysis"] = compiled

    async def _classify_protocol_node(self, state: ProtocolAnalysisState) -> Dict[str, Any]:
        """Classify the protocol type."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="protocol_analysis",
            step_name="classify_protocol"
        ).inc()

        # Simulated classification (in production, use actual LLM service)
        classification = {
            "protocol_type": "unknown",
            "confidence": 0.0,
            "possible_matches": [],
            "features_detected": []
        }

        return {
            "classification_result": classification,
            "current_step": "classify_protocol",
            "status": WorkflowStatus.RUNNING.value
        }

    async def _analyze_fields_node(self, state: ProtocolAnalysisState) -> Dict[str, Any]:
        """Analyze protocol fields."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="protocol_analysis",
            step_name="analyze_fields"
        ).inc()

        field_analysis = {
            "fields_detected": [],
            "field_boundaries": [],
            "data_types": {},
            "patterns": []
        }

        return {
            "field_analysis": field_analysis,
            "current_step": "analyze_fields"
        }

    async def _assess_security_node(self, state: ProtocolAnalysisState) -> Dict[str, Any]:
        """Assess security implications."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="protocol_analysis",
            step_name="assess_security"
        ).inc()

        security_assessment = {
            "vulnerabilities": [],
            "risk_level": "low",
            "recommendations": [],
            "encryption_detected": False
        }

        # Determine if human review is needed
        requires_review = security_assessment.get("risk_level") in ["high", "critical"]

        return {
            "security_assessment": security_assessment,
            "current_step": "assess_security",
            "requires_human_review": requires_review
        }

    def _should_require_human_review(self, state: ProtocolAnalysisState) -> str:
        """Determine if human review is required."""
        if state.get("requires_human_review", False):
            return "human_review"
        return "generate_report"

    async def _human_review_node(self, state: ProtocolAnalysisState) -> Dict[str, Any]:
        """Handle human review step."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="protocol_analysis",
            step_name="human_review"
        ).inc()

        return {
            "current_step": "human_review",
            "status": WorkflowStatus.WAITING_HUMAN.value
        }

    async def _generate_report_node(self, state: ProtocolAnalysisState) -> Dict[str, Any]:
        """Generate final analysis report."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="protocol_analysis",
            step_name="generate_report"
        ).inc()

        final_report = {
            "protocol_classification": state.get("classification_result", {}),
            "field_analysis": state.get("field_analysis", {}),
            "security_assessment": state.get("security_assessment", {}),
            "generated_at": datetime.utcnow().isoformat(),
            "workflow_id": state.get("workflow_id")
        }

        # Calculate confidence score
        classification_conf = state.get("classification_result", {}).get("confidence", 0.0)
        confidence_score = classification_conf

        return {
            "final_report": final_report,
            "confidence_score": confidence_score,
            "current_step": "generate_report",
            "status": WorkflowStatus.COMPLETED.value
        }

    # =========================================================================
    # Security Orchestration Workflow
    # =========================================================================

    def _register_security_orchestration_workflow(self) -> None:
        """Register the security orchestration workflow."""
        if not LANGGRAPH_AVAILABLE:
            return

        workflow = StateGraph(SecurityOrchestrationState)

        # Add nodes
        workflow.add_node("classify_threat", self._classify_threat_node)
        workflow.add_node("assess_impact", self._assess_impact_node)
        workflow.add_node("plan_mitigation", self._plan_mitigation_node)
        workflow.add_node("human_approval", self._human_approval_node)
        workflow.add_node("execute_actions", self._execute_actions_node)
        workflow.add_node("generate_incident_report", self._generate_incident_report_node)

        # Define edges
        workflow.set_entry_point("classify_threat")

        workflow.add_edge("classify_threat", "assess_impact")
        workflow.add_edge("assess_impact", "plan_mitigation")

        # Conditional edge for human approval
        workflow.add_conditional_edges(
            "plan_mitigation",
            self._should_require_approval,
            {
                "human_approval": "human_approval",
                "execute_actions": "execute_actions"
            }
        )

        workflow.add_edge("human_approval", "execute_actions")
        workflow.add_edge("execute_actions", "generate_incident_report")
        workflow.add_edge("generate_incident_report", END)

        compiled = workflow.compile(checkpointer=self.checkpointer)
        self.workflows["security_orchestration"] = compiled

    async def _classify_threat_node(self, state: SecurityOrchestrationState) -> Dict[str, Any]:
        """Classify the security threat."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="security_orchestration",
            step_name="classify_threat"
        ).inc()

        threat_classification = {
            "threat_type": "unknown",
            "category": "unclassified",
            "indicators": [],
            "related_threats": []
        }

        return {
            "threat_classification": threat_classification,
            "current_step": "classify_threat",
            "status": WorkflowStatus.RUNNING.value
        }

    async def _assess_impact_node(self, state: SecurityOrchestrationState) -> Dict[str, Any]:
        """Assess the impact of the threat."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="security_orchestration",
            step_name="assess_impact"
        ).inc()

        impact_assessment = {
            "affected_systems": [],
            "data_at_risk": [],
            "business_impact": "low",
            "urgency": "normal"
        }

        return {
            "impact_assessment": impact_assessment,
            "current_step": "assess_impact"
        }

    async def _plan_mitigation_node(self, state: SecurityOrchestrationState) -> Dict[str, Any]:
        """Plan mitigation actions."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="security_orchestration",
            step_name="plan_mitigation"
        ).inc()

        mitigation_plan = {
            "immediate_actions": [],
            "short_term_actions": [],
            "long_term_actions": [],
            "estimated_resolution_time": "unknown"
        }

        # Determine if approval is needed
        severity = state.get("severity", "low")
        requires_approval = severity in ["high", "critical"]

        return {
            "mitigation_plan": mitigation_plan,
            "current_step": "plan_mitigation",
            "requires_human_approval": requires_approval,
            "actions_pending": mitigation_plan.get("immediate_actions", [])
        }

    def _should_require_approval(self, state: SecurityOrchestrationState) -> str:
        """Determine if human approval is required."""
        if state.get("requires_human_approval", False):
            return "human_approval"
        return "execute_actions"

    async def _human_approval_node(self, state: SecurityOrchestrationState) -> Dict[str, Any]:
        """Handle human approval step."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="security_orchestration",
            step_name="human_approval"
        ).inc()

        return {
            "current_step": "human_approval",
            "status": WorkflowStatus.WAITING_HUMAN.value
        }

    async def _execute_actions_node(self, state: SecurityOrchestrationState) -> Dict[str, Any]:
        """Execute mitigation actions."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="security_orchestration",
            step_name="execute_actions"
        ).inc()

        # Execute pending actions (simulated)
        actions_executed = []
        for action in state.get("actions_pending", []):
            actions_executed.append({
                **action,
                "executed_at": datetime.utcnow().isoformat(),
                "status": "completed"
            })

        return {
            "actions_executed": actions_executed,
            "actions_pending": [],
            "current_step": "execute_actions"
        }

    async def _generate_incident_report_node(self, state: SecurityOrchestrationState) -> Dict[str, Any]:
        """Generate incident report."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="security_orchestration",
            step_name="generate_incident_report"
        ).inc()

        incident_report = {
            "incident_id": state.get("workflow_id"),
            "threat_classification": state.get("threat_classification", {}),
            "impact_assessment": state.get("impact_assessment", {}),
            "mitigation_plan": state.get("mitigation_plan", {}),
            "actions_taken": state.get("actions_executed", []),
            "generated_at": datetime.utcnow().isoformat()
        }

        return {
            "incident_report": incident_report,
            "current_step": "generate_incident_report",
            "status": WorkflowStatus.COMPLETED.value
        }

    # =========================================================================
    # Compliance Check Workflow
    # =========================================================================

    def _register_compliance_check_workflow(self) -> None:
        """Register the compliance check workflow."""
        if not LANGGRAPH_AVAILABLE:
            return

        workflow = StateGraph(ComplianceCheckState)

        # Add nodes
        workflow.add_node("check_frameworks", self._check_frameworks_node)
        workflow.add_node("identify_gaps", self._identify_gaps_node)
        workflow.add_node("generate_remediation", self._generate_remediation_node)
        workflow.add_node("compile_report", self._compile_compliance_report_node)

        # Define edges
        workflow.set_entry_point("check_frameworks")

        workflow.add_edge("check_frameworks", "identify_gaps")
        workflow.add_edge("identify_gaps", "generate_remediation")
        workflow.add_edge("generate_remediation", "compile_report")
        workflow.add_edge("compile_report", END)

        compiled = workflow.compile(checkpointer=self.checkpointer)
        self.workflows["compliance_check"] = compiled

    async def _check_frameworks_node(self, state: ComplianceCheckState) -> Dict[str, Any]:
        """Check compliance against specified frameworks."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="compliance_check",
            step_name="check_frameworks"
        ).inc()

        framework_results = {}
        for framework in state.get("frameworks", []):
            framework_results[framework] = {
                "checked_controls": [],
                "passed_controls": [],
                "failed_controls": [],
                "not_applicable": []
            }

        return {
            "framework_results": framework_results,
            "current_step": "check_frameworks",
            "status": WorkflowStatus.RUNNING.value
        }

    async def _identify_gaps_node(self, state: ComplianceCheckState) -> Dict[str, Any]:
        """Identify compliance gaps."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="compliance_check",
            step_name="identify_gaps"
        ).inc()

        compliance_gaps = []
        for framework, results in state.get("framework_results", {}).items():
            for control in results.get("failed_controls", []):
                compliance_gaps.append({
                    "framework": framework,
                    "control": control,
                    "severity": "medium",
                    "description": f"Failed control {control} in {framework}"
                })

        return {
            "compliance_gaps": compliance_gaps,
            "current_step": "identify_gaps"
        }

    async def _generate_remediation_node(self, state: ComplianceCheckState) -> Dict[str, Any]:
        """Generate remediation steps."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="compliance_check",
            step_name="generate_remediation"
        ).inc()

        remediation_steps = []
        for gap in state.get("compliance_gaps", []):
            remediation_steps.append({
                "gap": gap,
                "remediation": f"Address {gap.get('control')} in {gap.get('framework')}",
                "priority": gap.get("severity", "medium"),
                "estimated_effort": "medium"
            })

        return {
            "remediation_steps": remediation_steps,
            "current_step": "generate_remediation"
        }

    async def _compile_compliance_report_node(self, state: ComplianceCheckState) -> Dict[str, Any]:
        """Compile final compliance report."""
        WORKFLOW_STEP_COUNTER.labels(
            workflow_type="compliance_check",
            step_name="compile_report"
        ).inc()

        # Calculate overall score
        total_controls = 0
        passed_controls = 0
        for results in state.get("framework_results", {}).values():
            total_controls += len(results.get("checked_controls", []))
            passed_controls += len(results.get("passed_controls", []))

        overall_score = (passed_controls / total_controls * 100) if total_controls > 0 else 0.0

        compliance_report = {
            "workflow_id": state.get("workflow_id"),
            "frameworks_checked": state.get("frameworks", []),
            "framework_results": state.get("framework_results", {}),
            "compliance_gaps": state.get("compliance_gaps", []),
            "remediation_steps": state.get("remediation_steps", []),
            "overall_score": overall_score,
            "generated_at": datetime.utcnow().isoformat()
        }

        return {
            "compliance_report": compliance_report,
            "overall_score": overall_score,
            "current_step": "compile_report",
            "status": WorkflowStatus.COMPLETED.value
        }

    # =========================================================================
    # Workflow Execution API
    # =========================================================================

    async def execute_workflow(
        self,
        workflow_type: str,
        initial_state: Dict[str, Any],
        thread_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow.

        Args:
            workflow_type: Type of workflow to execute
            initial_state: Initial state for the workflow
            thread_id: Optional thread ID for checkpointing
            config: Optional execution configuration

        Returns:
            Final workflow state
        """
        import time
        start_time = time.time()

        workflow_id = str(uuid.uuid4())
        thread_id = thread_id or workflow_id

        # Add workflow metadata to initial state
        initial_state["workflow_id"] = workflow_id
        initial_state["status"] = WorkflowStatus.PENDING.value
        initial_state["messages"] = initial_state.get("messages", [])

        try:
            if not LANGGRAPH_AVAILABLE:
                # Fallback execution without LangGraph
                return await self._execute_fallback_workflow(workflow_type, initial_state)

            workflow = self.workflows.get(workflow_type)
            if not workflow:
                raise ValueError(f"Unknown workflow type: {workflow_type}")

            # Execute workflow
            execution_config = {"configurable": {"thread_id": thread_id}}
            if config:
                execution_config.update(config)

            final_state = await workflow.ainvoke(initial_state, execution_config)

            # Update metrics
            WORKFLOW_EXECUTION_COUNTER.labels(
                workflow_type=workflow_type,
                status="success"
            ).inc()
            WORKFLOW_DURATION.labels(workflow_type=workflow_type).observe(
                time.time() - start_time
            )

            return final_state

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            WORKFLOW_EXECUTION_COUNTER.labels(
                workflow_type=workflow_type,
                status="error"
            ).inc()

            return {
                **initial_state,
                "status": WorkflowStatus.FAILED.value,
                "error": str(e)
            }

    async def _execute_fallback_workflow(
        self,
        workflow_type: str,
        initial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow without LangGraph (fallback)."""
        self.logger.warning(f"Executing {workflow_type} workflow in fallback mode")

        state = initial_state.copy()
        state["status"] = WorkflowStatus.RUNNING.value

        # Simple sequential execution based on workflow type
        if workflow_type == "protocol_analysis":
            state = await self._classify_protocol_node(state)
            state = await self._analyze_fields_node(state)
            state = await self._assess_security_node(state)
            state = await self._generate_report_node(state)
        elif workflow_type == "security_orchestration":
            state = await self._classify_threat_node(state)
            state = await self._assess_impact_node(state)
            state = await self._plan_mitigation_node(state)
            state = await self._execute_actions_node(state)
            state = await self._generate_incident_report_node(state)
        elif workflow_type == "compliance_check":
            state = await self._check_frameworks_node(state)
            state = await self._identify_gaps_node(state)
            state = await self._generate_remediation_node(state)
            state = await self._compile_compliance_report_node(state)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        return state

    async def resume_workflow(
        self,
        workflow_type: str,
        thread_id: str,
        human_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resume a paused workflow (e.g., after human review).

        Args:
            workflow_type: Type of workflow
            thread_id: Thread ID of the paused workflow
            human_input: Optional input from human review

        Returns:
            Final workflow state
        """
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("LangGraph not available for workflow resumption")

        workflow = self.workflows.get(workflow_type)
        if not workflow:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        # Get current state
        config = {"configurable": {"thread_id": thread_id}}
        current_state = workflow.get_state(config)

        # Update state with human input
        if human_input:
            current_state = {**current_state, **human_input}

        # Resume execution
        final_state = await workflow.ainvoke(current_state, config)

        return final_state

    def get_workflow_state(
        self,
        workflow_type: str,
        thread_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the current state of a workflow."""
        if not LANGGRAPH_AVAILABLE:
            return self.workflow_states.get(thread_id)

        workflow = self.workflows.get(workflow_type)
        if not workflow:
            return None

        config = {"configurable": {"thread_id": thread_id}}
        return workflow.get_state(config)

    def list_available_workflows(self) -> List[str]:
        """List available workflow types."""
        return list(self.workflows.keys())

    def register_human_approval_callback(
        self,
        workflow_type: str,
        callback: Callable
    ) -> None:
        """Register a callback for human approval steps."""
        self.human_approval_callbacks[workflow_type] = callback

    def register_human_input_callback(
        self,
        workflow_type: str,
        callback: Callable
    ) -> None:
        """Register a callback for human input steps."""
        self.human_input_callbacks[workflow_type] = callback


# =============================================================================
# Global Workflow Manager
# =============================================================================

_workflow_manager: Optional[WorkflowManager] = None


def get_workflow_manager() -> WorkflowManager:
    """Get or create the global workflow manager."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager()
    return _workflow_manager


def set_workflow_manager(manager: WorkflowManager) -> None:
    """Set the global workflow manager."""
    global _workflow_manager
    _workflow_manager = manager
