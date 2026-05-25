"""
QBITEL Engine - BPO Automation Recipes

Pre-built deployment templates for common BPO/call center security patterns.

Each recipe is a self-contained, executable deployment plan with:
- Pre-flight validation (prerequisites, target readiness)
- Ordered execution steps with dependency tracking
- Rollback capability for every step
- Retry logic with configurable counts and timeouts

Built-in recipes:
    1. PCIDSSVoiceComplianceRecipe   - Full PCI-DSS voice compliance
    2. TollFraudPreventionRecipe     - Toll fraud detection and prevention
    3. RemoteWorkforceSecurityRecipe  - Secure remote/WFH agents
    4. QuantumSafeVoiceRecipe        - Upgrade voice to quantum-safe
    5. ComplianceSuiteRecipe         - Deploy full compliance monitoring
    6. FullBPOSecurityRecipe         - Everything combined

Use the RecipeRegistry to discover, manage, and execute recipes.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StepActionType(Enum):
    """Types of automation step actions."""

    CONFIGURE = ("configure", "Apply configuration to a target system")
    DEPLOY = ("deploy", "Deploy a new security control or component")
    VALIDATE = ("validate", "Validate a configuration or deployment")
    MONITOR = ("monitor", "Set up monitoring for a control")

    def __init__(self, action_id: str, description: str):
        self.action_id = action_id
        self.description = description


class StepStatus(Enum):
    """Execution status of an automation step."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    ROLLED_BACK = auto()


class RecipeStatus(Enum):
    """Execution status of a recipe."""

    NOT_STARTED = auto()
    VALIDATING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ROLLED_BACK = auto()
    PARTIALLY_COMPLETED = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AutomationStep:
    """
    A single step in an automation recipe.

    Each step represents an atomic action that can be executed, validated,
    and rolled back independently.
    """

    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    description: str = ""
    action_type: StepActionType = StepActionType.CONFIGURE
    target_system: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 120
    retry_count: int = 3
    rollback_action: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    output: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize step for reporting."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "action_type": self.action_type.action_id,
            "target_system": self.target_system,
            "status": self.status.name,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "depends_on": self.depends_on,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class RecipeExecutionResult:
    """Result of executing an automation recipe."""

    recipe_id: str = ""
    recipe_name: str = ""
    status: RecipeStatus = RecipeStatus.NOT_STARTED
    steps_completed: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    total_steps: int = 0
    total_duration_seconds: float = 0.0
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.steps_completed / self.total_steps

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result for reporting."""
        return {
            "recipe_id": self.recipe_id,
            "recipe_name": self.recipe_name,
            "status": self.status.name,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "steps_skipped": self.steps_skipped,
            "total_steps": self.total_steps,
            "success_rate": self.success_rate,
            "total_duration_seconds": self.total_duration_seconds,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Base Recipe
# ---------------------------------------------------------------------------


class BPOAutomationRecipe(ABC):
    """
    Base class for BPO automation recipes.

    A recipe is a self-contained deployment template that defines a sequence
    of automation steps to achieve a specific security objective. Each recipe
    supports pre-flight validation, ordered execution, and rollback.

    Subclasses must implement:
        _build_steps()      - Define the recipe's automation steps
        _build_rollback()   - Define rollback steps for undo capability

    Usage::

        recipe = PCIDSSVoiceComplianceRecipe(tenant_id="t-001")
        validation = await recipe.validate()
        if validation["valid"]:
            result = await recipe.execute()
            print(result.to_dict())
    """

    def __init__(
        self,
        tenant_id: str = "",
        *,
        dry_run: bool = False,
        stop_on_failure: bool = True,
        max_parallel_steps: int = 1,
    ):
        self.recipe_id: str = str(uuid.uuid4())
        self.name: str = ""
        self.description: str = ""
        self.target_environment: str = "bpo_call_center"
        self.tenant_id = tenant_id
        self.dry_run = dry_run
        self.stop_on_failure = stop_on_failure
        self.max_parallel_steps = max_parallel_steps

        # Steps
        self.prerequisites: List[str] = []
        self.steps: List[AutomationStep] = []
        self.rollback_steps: List[AutomationStep] = []

        # State
        self._status = RecipeStatus.NOT_STARTED
        self._execution_result: Optional[RecipeExecutionResult] = None

        # Build the recipe steps
        self._build_steps()
        self._build_rollback()

    @abstractmethod
    def _build_steps(self) -> None:
        """Define the automation steps for this recipe."""
        ...

    @abstractmethod
    def _build_rollback(self) -> None:
        """Define rollback steps for this recipe."""
        ...

    async def validate(self) -> Dict[str, Any]:
        """
        Pre-flight validation: check prerequisites and target readiness.

        Returns:
            Dictionary with validation results and any blockers found.
        """
        self._status = RecipeStatus.VALIDATING
        logger.info("Validating recipe: %s", self.name)

        validation = {
            "valid": True,
            "recipe_id": self.recipe_id,
            "recipe_name": self.name,
            "prerequisites_met": [],
            "prerequisites_missing": [],
            "warnings": [],
            "step_count": len(self.steps),
        }

        # Check prerequisites
        for prereq in self.prerequisites:
            met = await self._check_prerequisite(prereq)
            if met:
                validation["prerequisites_met"].append(prereq)
            else:
                validation["prerequisites_missing"].append(prereq)
                validation["valid"] = False

        # Validate step dependencies
        step_ids = {step.step_id for step in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    validation["warnings"].append(
                        f"Step '{step.name}' depends on unknown step '{dep}'"
                    )

        # Validate target systems are reachable (placeholder)
        target_systems = {step.target_system for step in self.steps}
        for target in target_systems:
            if target:
                validation.setdefault("target_systems", []).append(target)

        logger.info(
            "Validation result for %s: valid=%s, prereqs_met=%d/%d",
            self.name,
            validation["valid"],
            len(validation["prerequisites_met"]),
            len(self.prerequisites),
        )

        return validation

    async def execute(self) -> RecipeExecutionResult:
        """
        Execute the automation recipe.

        Runs all steps in order, respecting dependencies. Steps are retried
        on failure up to their retry_count. If stop_on_failure is True,
        execution halts on the first failure.

        Returns:
            RecipeExecutionResult with detailed status.
        """
        start_time = time.time()
        self._status = RecipeStatus.RUNNING

        result = RecipeExecutionResult(
            recipe_id=self.recipe_id,
            recipe_name=self.name,
            status=RecipeStatus.RUNNING,
            total_steps=len(self.steps),
            started_at=datetime.utcnow(),
        )

        logger.info(
            "Executing recipe: %s (%d steps, dry_run=%s)",
            self.name, len(self.steps), self.dry_run,
        )

        completed_steps: Set[str] = set()

        for step in self.steps:
            # Check dependencies
            unmet_deps = [
                dep for dep in step.depends_on if dep not in completed_steps
            ]
            if unmet_deps:
                step.status = StepStatus.SKIPPED
                step.error_message = f"Unmet dependencies: {unmet_deps}"
                result.steps_skipped += 1
                result.step_results.append(step.to_dict())
                logger.warning(
                    "Skipping step '%s': unmet dependencies %s",
                    step.name, unmet_deps,
                )
                continue

            # Execute the step with retries
            success = await self._execute_step_with_retry(step)

            result.step_results.append(step.to_dict())

            if success:
                completed_steps.add(step.step_id)
                result.steps_completed += 1
            else:
                result.steps_failed += 1
                result.errors.append(
                    f"Step '{step.name}' failed: {step.error_message}"
                )

                if self.stop_on_failure:
                    logger.error(
                        "Recipe halted at step '%s' (stop_on_failure=True)",
                        step.name,
                    )
                    break

        # Determine final status
        if result.steps_failed == 0:
            result.status = RecipeStatus.COMPLETED
            self._status = RecipeStatus.COMPLETED
        elif result.steps_completed > 0:
            result.status = RecipeStatus.PARTIALLY_COMPLETED
            self._status = RecipeStatus.PARTIALLY_COMPLETED
        else:
            result.status = RecipeStatus.FAILED
            self._status = RecipeStatus.FAILED

        result.completed_at = datetime.utcnow()
        result.total_duration_seconds = time.time() - start_time

        self._execution_result = result

        logger.info(
            "Recipe %s complete: status=%s completed=%d/%d failed=%d (%.2fs)",
            self.name,
            result.status.name,
            result.steps_completed,
            result.total_steps,
            result.steps_failed,
            result.total_duration_seconds,
        )

        return result

    async def rollback(self) -> RecipeExecutionResult:
        """
        Rollback recipe execution by running rollback steps in reverse order.

        Returns:
            RecipeExecutionResult for the rollback operation.
        """
        logger.warning("Rolling back recipe: %s", self.name)
        start_time = time.time()

        result = RecipeExecutionResult(
            recipe_id=self.recipe_id,
            recipe_name=f"{self.name} [ROLLBACK]",
            status=RecipeStatus.RUNNING,
            total_steps=len(self.rollback_steps),
            started_at=datetime.utcnow(),
        )

        for step in reversed(self.rollback_steps):
            success = await self._execute_step_with_retry(step)
            result.step_results.append(step.to_dict())

            if success:
                result.steps_completed += 1
            else:
                result.steps_failed += 1
                result.errors.append(
                    f"Rollback step '{step.name}' failed: {step.error_message}"
                )

        result.status = (
            RecipeStatus.COMPLETED if result.steps_failed == 0
            else RecipeStatus.PARTIALLY_COMPLETED
        )
        result.completed_at = datetime.utcnow()
        result.total_duration_seconds = time.time() - start_time
        self._status = RecipeStatus.ROLLED_BACK

        logger.info(
            "Rollback of %s: status=%s rolled_back=%d/%d",
            self.name, result.status.name,
            result.steps_completed, result.total_steps,
        )

        return result

    async def _execute_step_with_retry(self, step: AutomationStep) -> bool:
        """Execute a step with retry logic."""
        for attempt in range(1, step.retry_count + 1):
            step.started_at = datetime.utcnow()
            step.status = StepStatus.RUNNING

            try:
                if self.dry_run:
                    logger.info(
                        "[DRY RUN] Step '%s' on %s",
                        step.name, step.target_system,
                    )
                    step.status = StepStatus.COMPLETED
                    step.output = {"dry_run": True}
                else:
                    await asyncio.wait_for(
                        self._execute_step(step),
                        timeout=step.timeout_seconds,
                    )
                    step.status = StepStatus.COMPLETED

                step.completed_at = datetime.utcnow()
                step.duration_seconds = (
                    step.completed_at - step.started_at
                ).total_seconds()
                return True

            except asyncio.TimeoutError:
                step.error_message = (
                    f"Timeout after {step.timeout_seconds}s "
                    f"(attempt {attempt}/{step.retry_count})"
                )
                logger.warning(
                    "Step '%s' timed out (attempt %d/%d)",
                    step.name, attempt, step.retry_count,
                )
            except Exception as e:
                step.error_message = (
                    f"{e} (attempt {attempt}/{step.retry_count})"
                )
                logger.warning(
                    "Step '%s' failed (attempt %d/%d): %s",
                    step.name, attempt, step.retry_count, e,
                )

            if attempt < step.retry_count:
                await asyncio.sleep(min(2 ** attempt, 30))

        step.status = StepStatus.FAILED
        step.completed_at = datetime.utcnow()
        if step.started_at:
            step.duration_seconds = (
                step.completed_at - step.started_at
            ).total_seconds()
        return False

    async def _execute_step(self, step: AutomationStep) -> None:
        """Execute a single automation step. Override for custom logic."""
        # Default implementation simulates step execution
        logger.info(
            "Executing step '%s' [%s] on %s",
            step.name, step.action_type.action_id, step.target_system,
        )
        await asyncio.sleep(0.05)  # Simulate work
        step.output = {"executed": True, "parameters": step.parameters}

    async def _check_prerequisite(self, prerequisite: str) -> bool:
        """Check if a prerequisite is met. Override for custom checks."""
        # Default: assume prerequisites are met
        logger.debug("Checking prerequisite: %s", prerequisite)
        return True

    @property
    def status(self) -> RecipeStatus:
        """Current recipe status."""
        return self._status


# ---------------------------------------------------------------------------
# Recipe 1: PCI-DSS Voice Compliance
# ---------------------------------------------------------------------------


class PCIDSSVoiceComplianceRecipe(BPOAutomationRecipe):
    """
    Deploy full PCI-DSS voice compliance for contact centers.

    Configures DTMF masking on all IVR paths, enables recording
    pause/resume during payment capture, deploys PAN detection,
    sets up agent screen masking, configures PCI scope management,
    and generates a compliance report.
    """

    def __init__(self, tenant_id: str = "", **kwargs):
        super().__init__(tenant_id=tenant_id, **kwargs)
        self.name = "PCI-DSS Voice Compliance"
        self.description = (
            "Deploy full PCI-DSS 4.0 voice channel compliance including "
            "DTMF masking, recording controls, PAN detection, and "
            "agent screen masking."
        )
        self.prerequisites = [
            "PBX system accessible",
            "IVR configuration write access",
            "Recording system API available",
            "Agent desktop management access",
        ]

    def _build_steps(self) -> None:
        self.steps = [
            AutomationStep(
                step_id="pci-001",
                name="Configure DTMF masking on IVR paths",
                description="Enable DTMF clamping on all IVR payment collection paths",
                action_type=StepActionType.CONFIGURE,
                target_system="pbx_ivr",
                parameters={
                    "masking_mode": "clamp",
                    "scope": "all_payment_paths",
                    "tone_replacement": "flat_440hz",
                },
                timeout_seconds=120,
                retry_count=3,
                rollback_action="disable_dtmf_masking",
            ),
            AutomationStep(
                step_id="pci-002",
                name="Enable recording pause/resume",
                description="Configure automatic recording pause during payment entry",
                action_type=StepActionType.CONFIGURE,
                target_system="recording_system",
                parameters={
                    "trigger": "dtmf_payment_entry",
                    "auto_pause": True,
                    "auto_resume": True,
                    "max_pause_seconds": 120,
                },
                timeout_seconds=90,
                retry_count=3,
                rollback_action="disable_recording_pause_resume",
                depends_on=["pci-001"],
            ),
            AutomationStep(
                step_id="pci-003",
                name="Deploy PAN detection engine",
                description="Install real-time PAN detection on data streams",
                action_type=StepActionType.DEPLOY,
                target_system="agent_desktop",
                parameters={
                    "detection_types": ["credit_card", "debit_card"],
                    "action_on_detect": "mask_and_alert",
                    "luhn_validation": True,
                },
                timeout_seconds=180,
                retry_count=2,
                rollback_action="remove_pan_detection",
            ),
            AutomationStep(
                step_id="pci-004",
                name="Configure agent screen masking",
                description="Mask sensitive card fields on agent desktop",
                action_type=StepActionType.CONFIGURE,
                target_system="agent_desktop",
                parameters={
                    "fields_to_mask": ["card_number", "cvv", "expiry"],
                    "display_format": "****-****-****-{last4}",
                    "mask_in_crm": True,
                },
                timeout_seconds=60,
                retry_count=3,
                rollback_action="disable_screen_masking",
                depends_on=["pci-003"],
            ),
            AutomationStep(
                step_id="pci-005",
                name="Configure PCI scope management",
                description="Set up automated PCI scope reduction and tracking",
                action_type=StepActionType.CONFIGURE,
                target_system="compliance_engine",
                parameters={
                    "auto_descope": True,
                    "scope_tracking": True,
                    "cde_boundaries": ["payment_ivr", "agent_payment_screen"],
                },
                timeout_seconds=90,
                retry_count=2,
                rollback_action="remove_scope_management",
                depends_on=["pci-001", "pci-002"],
            ),
            AutomationStep(
                step_id="pci-006",
                name="Generate PCI-DSS compliance report",
                description="Generate initial PCI-DSS SAQ-D compliance report",
                action_type=StepActionType.VALIDATE,
                target_system="compliance_engine",
                parameters={
                    "report_type": "PCI-DSS-4.0-SAQ-D",
                    "include_evidence": True,
                    "output_format": "pdf",
                },
                timeout_seconds=120,
                retry_count=2,
                depends_on=["pci-005"],
            ),
        ]

    def _build_rollback(self) -> None:
        self.rollback_steps = [
            AutomationStep(
                step_id="pci-rb-001",
                name="Disable DTMF masking",
                action_type=StepActionType.CONFIGURE,
                target_system="pbx_ivr",
                parameters={"masking_mode": "disabled"},
            ),
            AutomationStep(
                step_id="pci-rb-002",
                name="Disable recording pause/resume",
                action_type=StepActionType.CONFIGURE,
                target_system="recording_system",
                parameters={"auto_pause": False},
            ),
            AutomationStep(
                step_id="pci-rb-003",
                name="Remove PAN detection",
                action_type=StepActionType.CONFIGURE,
                target_system="agent_desktop",
                parameters={"pan_detection": False},
            ),
            AutomationStep(
                step_id="pci-rb-004",
                name="Disable screen masking",
                action_type=StepActionType.CONFIGURE,
                target_system="agent_desktop",
                parameters={"screen_masking": False},
            ),
        ]


# ---------------------------------------------------------------------------
# Recipe 2: Toll Fraud Prevention
# ---------------------------------------------------------------------------


class TollFraudPreventionRecipe(BPOAutomationRecipe):
    """
    Deploy toll fraud prevention for BPO PBX systems.

    Loads premium rate number databases, configures IRSF detection rules,
    sets up velocity/volume alerting, configures automatic call blocking,
    and enables off-hours monitoring.
    """

    def __init__(self, tenant_id: str = "", **kwargs):
        super().__init__(tenant_id=tenant_id, **kwargs)
        self.name = "Toll Fraud Prevention"
        self.description = (
            "Deploy comprehensive toll fraud detection and prevention "
            "including IRSF detection, velocity monitoring, premium "
            "number blocking, and off-hours restrictions."
        )
        self.prerequisites = [
            "PBX system accessible",
            "Outbound trunk configuration access",
            "Alerting system configured",
        ]

    def _build_steps(self) -> None:
        self.steps = [
            AutomationStep(
                step_id="tf-001",
                name="Load premium rate number database",
                description="Import IRSF premium rate number prefixes from global database",
                action_type=StepActionType.DEPLOY,
                target_system="pbx",
                parameters={
                    "database": "irsf_global_premium_list",
                    "regions": ["caribbean", "africa", "pacific", "satellite"],
                    "auto_update": True,
                    "update_frequency_hours": 24,
                },
                timeout_seconds=180,
                retry_count=3,
                rollback_action="remove_premium_number_database",
            ),
            AutomationStep(
                step_id="tf-002",
                name="Configure IRSF detection rules",
                description="Deploy real-time IRSF detection with pattern matching",
                action_type=StepActionType.CONFIGURE,
                target_system="pbx",
                parameters={
                    "rules": [
                        {"rule": "premium_prefix_block", "action": "block"},
                        {"rule": "high_cost_country_alert", "action": "alert"},
                        {"rule": "call_transfer_to_premium", "action": "block"},
                        {"rule": "caller_id_spoofing", "action": "block"},
                    ],
                },
                timeout_seconds=120,
                retry_count=3,
                rollback_action="remove_irsf_rules",
                depends_on=["tf-001"],
            ),
            AutomationStep(
                step_id="tf-003",
                name="Set up velocity and volume alerting",
                description="Configure call velocity thresholds and volume-based alerts",
                action_type=StepActionType.CONFIGURE,
                target_system="pbx",
                parameters={
                    "max_international_per_hour": 20,
                    "max_calls_per_minute": 5,
                    "max_calls_per_5_minutes": 15,
                    "alert_channels": ["email", "sms", "siem"],
                },
                timeout_seconds=60,
                retry_count=3,
                rollback_action="remove_velocity_alerting",
                depends_on=["tf-002"],
            ),
            AutomationStep(
                step_id="tf-004",
                name="Configure automatic call blocking",
                description="Enable automatic blocking of calls matching fraud patterns",
                action_type=StepActionType.CONFIGURE,
                target_system="pbx",
                parameters={
                    "auto_block_premium": True,
                    "auto_block_spoofed_cli": True,
                    "cost_threshold_per_call": 50.0,
                    "cost_threshold_per_day": 500.0,
                },
                timeout_seconds=60,
                retry_count=3,
                rollback_action="disable_auto_blocking",
                depends_on=["tf-002"],
            ),
            AutomationStep(
                step_id="tf-005",
                name="Enable off-hours monitoring",
                description="Configure enhanced monitoring for after-hours international calls",
                action_type=StepActionType.CONFIGURE,
                target_system="pbx",
                parameters={
                    "business_hours_start": 6,
                    "business_hours_end": 22,
                    "off_hours_action": "block_international",
                    "weekend_action": "block_international",
                    "exception_list": [],
                },
                timeout_seconds=60,
                retry_count=3,
                rollback_action="disable_off_hours_monitoring",
                depends_on=["tf-003"],
            ),
            AutomationStep(
                step_id="tf-006",
                name="Validate toll fraud prevention deployment",
                description="Run test scenarios to validate fraud detection is working",
                action_type=StepActionType.VALIDATE,
                target_system="pbx",
                parameters={
                    "test_premium_block": True,
                    "test_velocity_alert": True,
                    "test_off_hours_block": True,
                },
                timeout_seconds=120,
                retry_count=2,
                depends_on=["tf-004", "tf-005"],
            ),
        ]

    def _build_rollback(self) -> None:
        self.rollback_steps = [
            AutomationStep(
                step_id="tf-rb-001",
                name="Remove premium number database",
                action_type=StepActionType.CONFIGURE,
                target_system="pbx",
                parameters={"remove_premium_db": True},
            ),
            AutomationStep(
                step_id="tf-rb-002",
                name="Remove IRSF rules",
                action_type=StepActionType.CONFIGURE,
                target_system="pbx",
                parameters={"remove_irsf_rules": True},
            ),
            AutomationStep(
                step_id="tf-rb-003",
                name="Remove velocity alerting",
                action_type=StepActionType.CONFIGURE,
                target_system="pbx",
                parameters={"remove_velocity_config": True},
            ),
        ]


# ---------------------------------------------------------------------------
# Recipe 3: Remote Workforce Security
# ---------------------------------------------------------------------------


class RemoteWorkforceSecurityRecipe(BPOAutomationRecipe):
    """
    Secure remote/work-from-home BPO agents.

    Generates PQC key pairs, configures VPN-less PQC tunnels, sets up
    device posture checking, configures geo-fencing rules, deploys
    screen watermarking, and performs network risk assessment.
    """

    def __init__(self, tenant_id: str = "", **kwargs):
        super().__init__(tenant_id=tenant_id, **kwargs)
        self.name = "Remote Workforce Security"
        self.description = (
            "Deploy quantum-safe security for remote/WFH agents including "
            "PQC tunnels, device posture verification, geo-fencing, "
            "and screen watermarking."
        )
        self.prerequisites = [
            "PQC cryptography library available",
            "Agent endpoint management system accessible",
            "HSM or key management service available",
            "Geo-IP database available",
        ]

    def _build_steps(self) -> None:
        self.steps = [
            AutomationStep(
                step_id="rw-001",
                name="Generate PQC key pairs for remote agents",
                description="Generate ML-KEM-768 key pairs for each remote agent endpoint",
                action_type=StepActionType.DEPLOY,
                target_system="key_management",
                parameters={
                    "algorithm": "ML-KEM-768",
                    "key_type": "agent_endpoint",
                    "store_in_hsm": True,
                    "rotation_hours": 4,
                },
                timeout_seconds=300,
                retry_count=2,
                rollback_action="revoke_pqc_keys",
            ),
            AutomationStep(
                step_id="rw-002",
                name="Configure VPN-less PQC tunnels",
                description="Set up quantum-safe WireGuard tunnels for remote agents",
                action_type=StepActionType.CONFIGURE,
                target_system="tunnel_gateway",
                parameters={
                    "protocol": "pqc-wireguard",
                    "kem": "ML-KEM-768",
                    "endpoint_authentication": "mutual_pqc_tls",
                    "split_tunnel_prevention": True,
                },
                timeout_seconds=180,
                retry_count=3,
                rollback_action="remove_pqc_tunnels",
                depends_on=["rw-001"],
            ),
            AutomationStep(
                step_id="rw-003",
                name="Set up device posture checking",
                description="Configure endpoint compliance verification for remote devices",
                action_type=StepActionType.DEPLOY,
                target_system="agent_endpoint",
                parameters={
                    "check_os_patches": True,
                    "check_antivirus": True,
                    "check_disk_encryption": True,
                    "check_firewall": True,
                    "minimum_os_version": "Windows 10 22H2",
                    "check_frequency_minutes": 30,
                },
                timeout_seconds=120,
                retry_count=3,
                rollback_action="remove_posture_checks",
                depends_on=["rw-002"],
            ),
            AutomationStep(
                step_id="rw-004",
                name="Configure geo-fencing rules",
                description="Restrict agent connections to approved geographic regions",
                action_type=StepActionType.CONFIGURE,
                target_system="tunnel_gateway",
                parameters={
                    "allowed_countries": ["US", "CA", "GB", "IN", "PH"],
                    "block_vpn_exit_nodes": True,
                    "block_tor_exit_nodes": True,
                    "alert_on_violation": True,
                },
                timeout_seconds=60,
                retry_count=3,
                rollback_action="remove_geo_fencing",
                depends_on=["rw-002"],
            ),
            AutomationStep(
                step_id="rw-005",
                name="Deploy screen watermarking",
                description="Enable forensic watermarking on remote agent screens",
                action_type=StepActionType.DEPLOY,
                target_system="agent_endpoint",
                parameters={
                    "watermark_content": ["agent_id", "timestamp", "tenant_id"],
                    "visibility": "semi_transparent",
                    "tamper_detection": True,
                },
                timeout_seconds=90,
                retry_count=2,
                rollback_action="remove_screen_watermarking",
                depends_on=["rw-003"],
            ),
            AutomationStep(
                step_id="rw-006",
                name="Configure network risk assessment",
                description="Set up continuous assessment of agent home network security",
                action_type=StepActionType.MONITOR,
                target_system="agent_endpoint",
                parameters={
                    "check_open_ports": True,
                    "check_public_wifi": True,
                    "check_router_security": True,
                    "risk_threshold": "medium",
                    "action_on_high_risk": "alert_and_restrict",
                },
                timeout_seconds=120,
                retry_count=2,
                rollback_action="remove_network_assessment",
                depends_on=["rw-003"],
            ),
        ]

    def _build_rollback(self) -> None:
        self.rollback_steps = [
            AutomationStep(
                step_id="rw-rb-001",
                name="Remove PQC tunnels",
                action_type=StepActionType.CONFIGURE,
                target_system="tunnel_gateway",
                parameters={"remove_tunnels": True},
            ),
            AutomationStep(
                step_id="rw-rb-002",
                name="Revoke PQC key pairs",
                action_type=StepActionType.CONFIGURE,
                target_system="key_management",
                parameters={"revoke_all_agent_keys": True},
            ),
            AutomationStep(
                step_id="rw-rb-003",
                name="Remove device posture checks",
                action_type=StepActionType.CONFIGURE,
                target_system="agent_endpoint",
                parameters={"remove_posture_checks": True},
            ),
        ]


# ---------------------------------------------------------------------------
# Recipe 4: Quantum-Safe Voice
# ---------------------------------------------------------------------------


class QuantumSafeVoiceRecipe(BPOAutomationRecipe):
    """
    Upgrade voice infrastructure to quantum-safe encryption.

    Generates ML-KEM-768 key pairs, configures SIP-PQC-TLS (port 5062),
    enables SRTP-PQC for media, wraps recording keys with PQC, and
    updates HSM configuration.
    """

    def __init__(self, tenant_id: str = "", **kwargs):
        super().__init__(tenant_id=tenant_id, **kwargs)
        self.name = "Quantum-Safe Voice Upgrade"
        self.description = (
            "Upgrade all voice infrastructure to quantum-safe encryption "
            "using ML-KEM-768 for key exchange and SRTP-PQC for media."
        )
        self.prerequisites = [
            "PQC-capable SIP proxy available",
            "HSM with PQC support",
            "SIP endpoints support TLS 1.3",
            "Recording system supports key wrapping API",
        ]

    def _build_steps(self) -> None:
        self.steps = [
            AutomationStep(
                step_id="qs-001",
                name="Generate ML-KEM-768 key pairs",
                description="Generate quantum-safe key pairs for voice infrastructure",
                action_type=StepActionType.DEPLOY,
                target_system="key_management",
                parameters={
                    "algorithm": "ML-KEM-768",
                    "key_count": 10,
                    "store_in_hsm": True,
                    "usage": "voice_key_exchange",
                },
                timeout_seconds=120,
                retry_count=2,
                rollback_action="revoke_voice_pqc_keys",
            ),
            AutomationStep(
                step_id="qs-002",
                name="Configure SIP-PQC-TLS on port 5062",
                description="Enable PQC-TLS for SIP signaling on dedicated port",
                action_type=StepActionType.CONFIGURE,
                target_system="sip_proxy",
                parameters={
                    "port": 5062,
                    "tls_version": "1.3",
                    "kem": "ML-KEM-768",
                    "signature": "ML-DSA-65",
                    "hybrid_mode": True,
                    "hybrid_classical": "X25519",
                },
                timeout_seconds=180,
                retry_count=3,
                rollback_action="disable_sip_pqc_tls",
                depends_on=["qs-001"],
            ),
            AutomationStep(
                step_id="qs-003",
                name="Enable SRTP-PQC for media",
                description="Configure quantum-safe SRTP for voice media encryption",
                action_type=StepActionType.CONFIGURE,
                target_system="media_gateway",
                parameters={
                    "cipher": "AES-256-GCM",
                    "key_exchange": "ML-KEM-768",
                    "key_rotation_minutes": 60,
                    "fallback_srtp": True,
                },
                timeout_seconds=180,
                retry_count=3,
                rollback_action="disable_srtp_pqc",
                depends_on=["qs-001"],
            ),
            AutomationStep(
                step_id="qs-004",
                name="Wrap recording keys with PQC",
                description="Re-wrap existing recording encryption keys with PQC KEMs",
                action_type=StepActionType.CONFIGURE,
                target_system="recording_system",
                parameters={
                    "wrapping_algorithm": "ML-KEM-1024",
                    "key_hierarchy": "three_tier",
                    "root_key_in_hsm": True,
                    "batch_size": 1000,
                },
                timeout_seconds=600,
                retry_count=2,
                rollback_action="unwrap_recording_keys",
                depends_on=["qs-001"],
            ),
            AutomationStep(
                step_id="qs-005",
                name="Update HSM configuration",
                description="Configure HSM for PQC algorithm support",
                action_type=StepActionType.CONFIGURE,
                target_system="hsm",
                parameters={
                    "enable_ml_kem": True,
                    "enable_ml_dsa": True,
                    "fips_203_mode": True,
                    "fips_204_mode": True,
                    "backup_classical_keys": True,
                },
                timeout_seconds=300,
                retry_count=2,
                rollback_action="revert_hsm_config",
                depends_on=["qs-001"],
            ),
            AutomationStep(
                step_id="qs-006",
                name="Validate quantum-safe voice",
                description="Run end-to-end test of PQC voice encryption",
                action_type=StepActionType.VALIDATE,
                target_system="sip_proxy",
                parameters={
                    "test_sip_pqc_tls": True,
                    "test_srtp_pqc": True,
                    "test_recording_wrap": True,
                    "test_key_rotation": True,
                },
                timeout_seconds=180,
                retry_count=2,
                depends_on=["qs-002", "qs-003", "qs-004", "qs-005"],
            ),
        ]

    def _build_rollback(self) -> None:
        self.rollback_steps = [
            AutomationStep(
                step_id="qs-rb-001",
                name="Revert HSM configuration",
                action_type=StepActionType.CONFIGURE,
                target_system="hsm",
                parameters={"revert_pqc_config": True},
            ),
            AutomationStep(
                step_id="qs-rb-002",
                name="Disable SIP-PQC-TLS",
                action_type=StepActionType.CONFIGURE,
                target_system="sip_proxy",
                parameters={"disable_port_5062": True},
            ),
            AutomationStep(
                step_id="qs-rb-003",
                name="Disable SRTP-PQC",
                action_type=StepActionType.CONFIGURE,
                target_system="media_gateway",
                parameters={"revert_to_classical_srtp": True},
            ),
        ]


# ---------------------------------------------------------------------------
# Recipe 5: Compliance Suite
# ---------------------------------------------------------------------------


class ComplianceSuiteRecipe(BPOAutomationRecipe):
    """
    Deploy full compliance monitoring for BPO environments.

    Covers PCI-DSS 4.0 voice controls, TCPA consent management,
    HIPAA PHI protection (healthcare BPO), SOC 2 monitoring,
    GDPR recording consent, and automated compliance reporting.
    """

    def __init__(self, tenant_id: str = "", *, is_healthcare: bool = False, **kwargs):
        self.is_healthcare = is_healthcare
        super().__init__(tenant_id=tenant_id, **kwargs)
        self.name = "Full Compliance Suite"
        self.description = (
            "Deploy comprehensive compliance monitoring covering PCI-DSS 4.0, "
            "TCPA, SOC 2, GDPR" +
            (", and HIPAA" if is_healthcare else "") +
            " with automated reporting."
        )
        self.prerequisites = [
            "Compliance engine available",
            "Recording system API available",
            "Agent desktop management access",
            "SIEM integration configured",
        ]

    def _build_steps(self) -> None:
        self.steps = [
            AutomationStep(
                step_id="cs-001",
                name="Deploy PCI-DSS 4.0 voice controls",
                description="Configure PCI-DSS controls for voice channels",
                action_type=StepActionType.DEPLOY,
                target_system="compliance_engine",
                parameters={
                    "framework": "PCI-DSS-4.0",
                    "scope": "voice_channels",
                    "requirements": ["3.3", "3.4", "3.5", "8.3", "10.2"],
                },
                timeout_seconds=180,
                retry_count=2,
                rollback_action="remove_pci_dss_controls",
            ),
            AutomationStep(
                step_id="cs-002",
                name="Configure TCPA consent management",
                description="Set up TCPA consent tracking and enforcement",
                action_type=StepActionType.CONFIGURE,
                target_system="compliance_engine",
                parameters={
                    "consent_tracking": True,
                    "opt_out_mechanism": "automated",
                    "dnc_list_integration": True,
                    "recording_consent_prompt": True,
                    "time_zone_enforcement": True,
                },
                timeout_seconds=120,
                retry_count=2,
                rollback_action="remove_tcpa_controls",
            ),
            AutomationStep(
                step_id="cs-003",
                name="Configure SOC 2 monitoring",
                description="Deploy SOC 2 Type II continuous monitoring controls",
                action_type=StepActionType.DEPLOY,
                target_system="compliance_engine",
                parameters={
                    "trust_criteria": [
                        "security", "availability",
                        "processing_integrity", "confidentiality",
                    ],
                    "evidence_collection": True,
                    "continuous_monitoring": True,
                },
                timeout_seconds=120,
                retry_count=2,
                rollback_action="remove_soc2_monitoring",
            ),
            AutomationStep(
                step_id="cs-004",
                name="Configure GDPR recording consent",
                description="Set up GDPR-compliant recording consent management",
                action_type=StepActionType.CONFIGURE,
                target_system="recording_system",
                parameters={
                    "consent_required": True,
                    "consent_prompt": "automated_ivr",
                    "right_to_deletion": True,
                    "data_portability": True,
                    "retention_policy_days": 365,
                },
                timeout_seconds=90,
                retry_count=2,
                rollback_action="remove_gdpr_consent",
            ),
            AutomationStep(
                step_id="cs-005",
                name="Deploy automated compliance reporting",
                description="Configure automated compliance report generation",
                action_type=StepActionType.DEPLOY,
                target_system="compliance_engine",
                parameters={
                    "report_types": [
                        "PCI-DSS-SAQ-D", "SOC2-Type-II",
                        "GDPR-DPIA", "TCPA-Compliance",
                    ],
                    "frequency": "monthly",
                    "auto_distribute": True,
                    "recipients": ["compliance_team"],
                },
                timeout_seconds=60,
                retry_count=2,
                rollback_action="remove_compliance_reporting",
                depends_on=["cs-001", "cs-002", "cs-003", "cs-004"],
            ),
        ]

        # Add HIPAA step if healthcare BPO
        if self.is_healthcare:
            self.steps.insert(4, AutomationStep(
                step_id="cs-hipaa",
                name="Deploy HIPAA PHI protection",
                description="Configure HIPAA-compliant PHI handling for healthcare BPO",
                action_type=StepActionType.DEPLOY,
                target_system="compliance_engine",
                parameters={
                    "phi_detection": True,
                    "phi_masking": True,
                    "minimum_necessary_rule": True,
                    "breach_notification": True,
                    "baa_tracking": True,
                    "audit_logging": True,
                },
                timeout_seconds=120,
                retry_count=2,
                rollback_action="remove_hipaa_controls",
                depends_on=["cs-001"],
            ))

    def _build_rollback(self) -> None:
        self.rollback_steps = [
            AutomationStep(
                step_id="cs-rb-001",
                name="Remove compliance reporting",
                action_type=StepActionType.CONFIGURE,
                target_system="compliance_engine",
                parameters={"remove_reports": True},
            ),
            AutomationStep(
                step_id="cs-rb-002",
                name="Remove GDPR consent controls",
                action_type=StepActionType.CONFIGURE,
                target_system="recording_system",
                parameters={"remove_gdpr": True},
            ),
            AutomationStep(
                step_id="cs-rb-003",
                name="Remove SOC 2 monitoring",
                action_type=StepActionType.CONFIGURE,
                target_system="compliance_engine",
                parameters={"remove_soc2": True},
            ),
        ]


# ---------------------------------------------------------------------------
# Recipe 6: Full BPO Security (combines all recipes)
# ---------------------------------------------------------------------------


class FullBPOSecurityRecipe(BPOAutomationRecipe):
    """
    Deploy everything: combines all BPO security recipes into a single
    unified deployment with cross-recipe dependency validation.

    Executes in order:
        1. PCI-DSS Voice Compliance
        2. Toll Fraud Prevention
        3. Remote Workforce Security
        4. Quantum-Safe Voice Upgrade
        5. Full Compliance Suite
    """

    def __init__(
        self,
        tenant_id: str = "",
        *,
        is_healthcare: bool = False,
        **kwargs,
    ):
        self.is_healthcare = is_healthcare
        self._sub_recipes: List[BPOAutomationRecipe] = []
        super().__init__(tenant_id=tenant_id, **kwargs)
        self.name = "Full BPO Security Suite"
        self.description = (
            "Deploy the complete QBITEL BPO security suite combining "
            "PCI-DSS compliance, toll fraud prevention, remote workforce "
            "security, quantum-safe voice, and full compliance monitoring."
        )
        self.prerequisites = [
            "PBX system accessible",
            "PQC cryptography library available",
            "HSM or key management service available",
            "Agent endpoint management system accessible",
            "Recording system API available",
            "Compliance engine available",
            "SIEM integration configured",
        ]

    def _build_steps(self) -> None:
        # Create sub-recipes
        pci_recipe = PCIDSSVoiceComplianceRecipe(
            tenant_id=self.tenant_id, dry_run=self.dry_run
        )
        toll_fraud_recipe = TollFraudPreventionRecipe(
            tenant_id=self.tenant_id, dry_run=self.dry_run
        )
        remote_recipe = RemoteWorkforceSecurityRecipe(
            tenant_id=self.tenant_id, dry_run=self.dry_run
        )
        quantum_recipe = QuantumSafeVoiceRecipe(
            tenant_id=self.tenant_id, dry_run=self.dry_run
        )
        compliance_recipe = ComplianceSuiteRecipe(
            tenant_id=self.tenant_id,
            is_healthcare=self.is_healthcare,
            dry_run=self.dry_run,
        )

        self._sub_recipes = [
            pci_recipe,
            toll_fraud_recipe,
            remote_recipe,
            quantum_recipe,
            compliance_recipe,
        ]

        # Combine all steps from sub-recipes, prefixed for uniqueness
        self.steps = []
        for i, recipe in enumerate(self._sub_recipes, 1):
            for step in recipe.steps:
                # Prefix step IDs to avoid collisions
                combined_step = AutomationStep(
                    step_id=f"full-{i}-{step.step_id}",
                    name=f"[{recipe.name}] {step.name}",
                    description=step.description,
                    action_type=step.action_type,
                    target_system=step.target_system,
                    parameters=step.parameters,
                    timeout_seconds=step.timeout_seconds,
                    retry_count=step.retry_count,
                    rollback_action=step.rollback_action,
                    depends_on=[
                        f"full-{i}-{dep}" for dep in step.depends_on
                    ],
                )
                self.steps.append(combined_step)

        # Add unified deployment report step
        self.steps.append(AutomationStep(
            step_id="full-report",
            name="Generate unified deployment report",
            description="Generate comprehensive security deployment report",
            action_type=StepActionType.VALIDATE,
            target_system="compliance_engine",
            parameters={
                "report_type": "unified_bpo_security_deployment",
                "include_all_recipes": True,
                "include_evidence": True,
            },
            timeout_seconds=180,
            retry_count=2,
        ))

    def _build_rollback(self) -> None:
        # Combine rollback steps from all sub-recipes
        self.rollback_steps = []
        for i, recipe in enumerate(self._sub_recipes, 1):
            for step in recipe.rollback_steps:
                combined_step = AutomationStep(
                    step_id=f"full-rb-{i}-{step.step_id}",
                    name=f"[{recipe.name} ROLLBACK] {step.name}",
                    description=step.description,
                    action_type=step.action_type,
                    target_system=step.target_system,
                    parameters=step.parameters,
                    timeout_seconds=step.timeout_seconds,
                    retry_count=step.retry_count,
                )
                self.rollback_steps.append(combined_step)

    async def execute(self) -> RecipeExecutionResult:
        """
        Execute the full BPO security suite.

        Validates cross-recipe dependencies before execution, then runs
        all recipes in sequence.
        """
        logger.info(
            "========================================\n"
            "  FULL BPO SECURITY SUITE DEPLOYMENT\n"
            "  Tenant: %s\n"
            "  Recipes: %d\n"
            "  Total Steps: %d\n"
            "  Dry Run: %s\n"
            "========================================",
            self.tenant_id,
            len(self._sub_recipes),
            len(self.steps),
            self.dry_run,
        )

        # Validate cross-recipe dependencies
        validation = await self.validate()
        if not validation["valid"]:
            logger.error(
                "Full BPO security suite validation failed: %s",
                validation.get("prerequisites_missing"),
            )
            return RecipeExecutionResult(
                recipe_id=self.recipe_id,
                recipe_name=self.name,
                status=RecipeStatus.FAILED,
                total_steps=len(self.steps),
                errors=[
                    f"Validation failed: {validation.get('prerequisites_missing')}"
                ],
            )

        # Execute using parent class logic
        result = await super().execute()

        logger.info(
            "========================================\n"
            "  FULL BPO SECURITY SUITE COMPLETE\n"
            "  Status: %s\n"
            "  Steps: %d/%d completed\n"
            "  Duration: %.2fs\n"
            "========================================",
            result.status.name,
            result.steps_completed,
            result.total_steps,
            result.total_duration_seconds,
        )

        return result


# ---------------------------------------------------------------------------
# Recipe Registry
# ---------------------------------------------------------------------------


class RecipeRegistry:
    """
    Central registry for BPO automation recipes.

    Provides discovery, management, and execution of automation recipes.
    Pre-loads all built-in recipes on initialization.

    Usage::

        registry = RecipeRegistry()
        recipes = registry.list_recipes()
        recipe = registry.get_recipe("pci-dss-voice-compliance")
        result = await recipe.execute()
    """

    def __init__(self, *, auto_load: bool = True):
        self._recipes: Dict[str, BPOAutomationRecipe] = {}
        self._recipe_classes: Dict[str, type] = {}

        if auto_load:
            self._load_builtin_recipes()

    def _load_builtin_recipes(self) -> None:
        """Load all built-in BPO automation recipes."""
        builtin_recipes = {
            "pci-dss-voice-compliance": PCIDSSVoiceComplianceRecipe,
            "toll-fraud-prevention": TollFraudPreventionRecipe,
            "remote-workforce-security": RemoteWorkforceSecurityRecipe,
            "quantum-safe-voice": QuantumSafeVoiceRecipe,
            "compliance-suite": ComplianceSuiteRecipe,
            "full-bpo-security": FullBPOSecurityRecipe,
        }

        for recipe_id, recipe_cls in builtin_recipes.items():
            self._recipe_classes[recipe_id] = recipe_cls

        logger.info(
            "Loaded %d built-in BPO automation recipes",
            len(builtin_recipes),
        )

    def register_recipe(
        self,
        recipe_id: str,
        recipe: BPOAutomationRecipe,
    ) -> None:
        """
        Register a recipe instance in the registry.

        Args:
            recipe_id: Unique identifier for the recipe.
            recipe: The recipe instance to register.
        """
        self._recipes[recipe_id] = recipe
        logger.info("Registered recipe: %s (%s)", recipe_id, recipe.name)

    def get_recipe(
        self,
        recipe_id: str,
        tenant_id: str = "",
        **kwargs,
    ) -> Optional[BPOAutomationRecipe]:
        """
        Get a recipe by ID.

        If the recipe is a built-in class, a new instance is created with
        the provided tenant_id and kwargs. If it is a registered instance,
        the existing instance is returned.

        Args:
            recipe_id: The recipe identifier.
            tenant_id: Tenant ID for the new instance.
            **kwargs: Additional keyword arguments for the recipe constructor.

        Returns:
            The recipe instance, or None if not found.
        """
        # Check for registered instances first
        if recipe_id in self._recipes:
            return self._recipes[recipe_id]

        # Check for built-in recipe classes
        if recipe_id in self._recipe_classes:
            recipe_cls = self._recipe_classes[recipe_id]
            return recipe_cls(tenant_id=tenant_id, **kwargs)

        logger.warning("Recipe not found: %s", recipe_id)
        return None

    def list_recipes(self) -> List[Dict[str, Any]]:
        """
        List all available recipes with their metadata.

        Returns:
            List of recipe information dictionaries.
        """
        recipes = []

        # Built-in recipe classes
        for recipe_id, recipe_cls in self._recipe_classes.items():
            # Create a temporary instance to get metadata
            temp = recipe_cls.__new__(recipe_cls)
            # Manually set attributes to avoid __init__ side effects
            temp.name = ""
            temp.description = ""
            temp.prerequisites = []
            temp.steps = []
            temp.rollback_steps = []
            temp._status = RecipeStatus.NOT_STARTED

            try:
                instance = recipe_cls()
                recipes.append({
                    "recipe_id": recipe_id,
                    "name": instance.name,
                    "description": instance.description,
                    "type": "builtin",
                    "step_count": len(instance.steps),
                    "prerequisites": instance.prerequisites,
                })
            except Exception:
                recipes.append({
                    "recipe_id": recipe_id,
                    "name": recipe_cls.__name__,
                    "type": "builtin",
                })

        # Registered instances
        for recipe_id, recipe in self._recipes.items():
            if recipe_id not in self._recipe_classes:
                recipes.append({
                    "recipe_id": recipe_id,
                    "name": recipe.name,
                    "description": recipe.description,
                    "type": "custom",
                    "step_count": len(recipe.steps),
                    "status": recipe.status.name,
                })

        return recipes

    def get_recipes_for_environment(
        self,
        env_type: str,
        tenant_id: str = "",
    ) -> List[BPOAutomationRecipe]:
        """
        Get recipes applicable to a specific environment type.

        Args:
            env_type: Environment type (e.g., "bpo_call_center",
                      "healthcare_bpo", "financial_bpo").
            tenant_id: Tenant ID for recipe instances.

        Returns:
            List of applicable recipe instances.
        """
        applicable = []

        # All BPO recipes are applicable for standard BPO
        standard_recipes = [
            "pci-dss-voice-compliance",
            "toll-fraud-prevention",
            "remote-workforce-security",
            "quantum-safe-voice",
        ]

        for recipe_id in standard_recipes:
            recipe = self.get_recipe(recipe_id, tenant_id=tenant_id)
            if recipe:
                applicable.append(recipe)

        # Add compliance suite with healthcare flag if applicable
        if env_type == "healthcare_bpo":
            recipe = self.get_recipe(
                "compliance-suite",
                tenant_id=tenant_id,
                is_healthcare=True,
            )
        else:
            recipe = self.get_recipe(
                "compliance-suite",
                tenant_id=tenant_id,
            )
        if recipe:
            applicable.append(recipe)

        # Full suite is always available
        if env_type == "healthcare_bpo":
            full = self.get_recipe(
                "full-bpo-security",
                tenant_id=tenant_id,
                is_healthcare=True,
            )
        else:
            full = self.get_recipe(
                "full-bpo-security",
                tenant_id=tenant_id,
            )
        if full:
            applicable.append(full)

        logger.info(
            "Found %d applicable recipes for environment '%s'",
            len(applicable), env_type,
        )

        return applicable

    def remove_recipe(self, recipe_id: str) -> bool:
        """Remove a registered recipe from the registry."""
        if recipe_id in self._recipes:
            del self._recipes[recipe_id]
            logger.info("Removed recipe: %s", recipe_id)
            return True
        return False
