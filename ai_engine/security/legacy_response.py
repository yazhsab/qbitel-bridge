"""
QBITEL Engine - Legacy-Aware Response Manager

Enterprise-grade response execution for legacy systems with safety constraints.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest, get_llm_service
from ..core.config import Config
from ..core.exceptions import QbitelAIException
from ..monitoring.metrics import MetricsCollector
from .models import (
    SecurityEvent,
    LegacySystem,
    ResponseAction,
    QuarantineResult,
    AutomatedResponse,
    SystemCriticality,
    ProtocolType,
    ResponseType,
    ThreatLevel,
    SecurityException,
    QuarantineException,
    ResponseExecutionException,
    ThreatAnalysis,
    SecurityContext,
)

from prometheus_client import Counter, Histogram, Gauge, Summary

# Prometheus metrics
QUARANTINE_COUNTER = Counter(
    "qbitel_security_quarantine_total",
    "Quarantine operations",
    ["system_type", "status"],
)
QUARANTINE_DURATION = Histogram(
    "qbitel_security_quarantine_duration_seconds", "Quarantine operation duration"
)
LEGACY_RESPONSE_COUNTER = Counter(
    "qbitel_security_legacy_responses_total",
    "Legacy responses executed",
    ["protocol", "action"],
)
SAFETY_VIOLATIONS = Counter(
    "qbitel_security_safety_violations_total",
    "Safety constraint violations",
    ["constraint_type"],
)
DEPENDENCY_CHECKS = Summary(
    "qbitel_security_dependency_check_duration_seconds", "Dependency check duration"
)

logger = logging.getLogger(__name__)


class LegacySystemCapability:
    """Represents capabilities and constraints of a legacy system."""

    def __init__(self, system: LegacySystem):
        self.system = system
        self.supported_actions: Set[ResponseType] = set()
        self.constraints: Dict[str, Any] = {}
        self.risk_factors: List[str] = []
        self.safety_requirements: List[str] = []

        self._analyze_capabilities()

    def _analyze_capabilities(self):
        """Analyze system capabilities based on type and protocols."""

        # Base capabilities for all systems
        self.supported_actions.add(ResponseType.ALERT_SECURITY_TEAM)
        self.supported_actions.add(ResponseType.ENABLE_MONITORING)
        self.supported_actions.add(ResponseType.LOG_RETENTION_INCREASE)

        # Protocol-specific capabilities
        for protocol in self.system.protocol_types:
            if protocol == ProtocolType.HL7_MLLP:
                self._add_hl7_capabilities()
            elif protocol == ProtocolType.ISO8583:
                self._add_iso8583_capabilities()
            elif protocol == ProtocolType.MODBUS:
                self._add_modbus_capabilities()
            elif protocol == ProtocolType.TN3270E:
                self._add_tn3270e_capabilities()

        # System criticality constraints
        if self.system.criticality == SystemCriticality.MISSION_CRITICAL:
            self.constraints["require_approval"] = True
            self.constraints["max_downtime_seconds"] = 30
            self.safety_requirements.append("Zero-downtime operations required")
        elif self.system.criticality == SystemCriticality.BUSINESS_CRITICAL:
            self.constraints["max_downtime_seconds"] = 300
            self.safety_requirements.append("Minimal downtime operations")

        # Dependency-based constraints
        if self.system.dependent_systems:
            self.constraints["check_dependencies"] = True
            self.risk_factors.append(
                f"Has {len(self.system.dependent_systems)} dependent systems"
            )

    def _add_hl7_capabilities(self):
        """Add HL7-specific capabilities and constraints."""
        self.supported_actions.update(
            {
                ResponseType.NETWORK_SEGMENTATION,
                ResponseType.VIRTUAL_PATCH,
                ResponseType.REDIRECT_TRAFFIC,
            }
        )

        self.constraints["hl7_aware"] = True
        self.constraints["preserve_message_integrity"] = True
        self.safety_requirements.extend(
            ["Maintain patient data integrity", "Preserve clinical workflow continuity"]
        )
        self.risk_factors.append("Patient safety impact")

    def _add_iso8583_capabilities(self):
        """Add ISO8583-specific capabilities and constraints."""
        self.supported_actions.update(
            {
                ResponseType.BLOCK_IP,
                ResponseType.VIRTUAL_PATCH,
                ResponseType.NETWORK_SEGMENTATION,
            }
        )

        self.constraints["iso8583_aware"] = True
        self.constraints["transaction_integrity"] = True
        self.safety_requirements.extend(
            ["Maintain transaction consistency", "Preserve financial data integrity"]
        )
        self.risk_factors.append("Financial transaction impact")

    def _add_modbus_capabilities(self):
        """Add Modbus-specific capabilities and constraints."""
        self.supported_actions.update(
            {ResponseType.VIRTUAL_PATCH, ResponseType.ENABLE_MONITORING}
        )

        # Modbus systems are often very sensitive to network changes
        self.constraints["modbus_sensitive"] = True
        self.constraints["no_network_disruption"] = True
        self.safety_requirements.extend(
            [
                "No disruption to control systems",
                "Maintain operational technology continuity",
            ]
        )
        self.risk_factors.extend(
            ["Industrial process impact", "Potential safety hazards"]
        )

    def _add_tn3270e_capabilities(self):
        """Add TN3270E-specific capabilities and constraints."""
        self.supported_actions.update(
            {ResponseType.VIRTUAL_PATCH, ResponseType.NETWORK_SEGMENTATION}
        )

        self.constraints["mainframe_aware"] = True
        self.constraints["session_preservation"] = True
        self.safety_requirements.extend(
            ["Preserve user sessions", "Maintain mainframe connectivity"]
        )
        self.risk_factors.append("Mainframe access disruption")


class DependencyAnalyzer:
    """Analyzes system dependencies for safe response execution."""

    def __init__(self, legacy_systems: List[LegacySystem]):
        self.systems = {sys.system_id: sys for sys in legacy_systems}
        self.dependency_graph = self._build_dependency_graph()
        self.critical_paths = self._identify_critical_paths()

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build system dependency graph."""
        graph = {}

        for system_id, system in self.systems.items():
            graph[system_id] = set(system.dependent_systems)

        return graph

    def _identify_critical_paths(self) -> List[List[str]]:
        """Identify critical dependency paths."""
        critical_paths = []

        # Find systems that are dependencies for mission-critical systems
        for system_id, system in self.systems.items():
            if system.criticality == SystemCriticality.MISSION_CRITICAL:
                path = self._trace_dependencies(system_id)
                if len(path) > 1:
                    critical_paths.append(path)

        return critical_paths

    def _trace_dependencies(
        self, system_id: str, visited: Optional[Set[str]] = None
    ) -> List[str]:
        """Trace dependency chain for a system."""
        if visited is None:
            visited = set()

        if system_id in visited:
            return []  # Circular dependency

        visited.add(system_id)
        path = [system_id]

        system = self.systems.get(system_id)
        if system:
            for dep_id in system.dependency_systems:
                if dep_id in self.systems:
                    dep_path = self._trace_dependencies(dep_id, visited.copy())
                    if dep_path:
                        path.extend(dep_path)

        return path

    def get_impact_analysis(self, target_systems: List[str]) -> Dict[str, Any]:
        """Analyze impact of affecting target systems."""

        with DEPENDENCY_CHECKS.time():
            impacted_systems = set(target_systems)
            cascade_effects = []

            # Find systems that depend on target systems
            for system_id in target_systems:
                for other_id, deps in self.dependency_graph.items():
                    if system_id in deps:
                        impacted_systems.add(other_id)
                        cascade_effects.append(
                            {
                                "system": other_id,
                                "depends_on": system_id,
                                "criticality": self.systems[other_id].criticality.value,
                            }
                        )

            # Calculate risk score
            risk_score = 0.0
            for system_id in impacted_systems:
                system = self.systems.get(system_id)
                if system:
                    if system.criticality == SystemCriticality.MISSION_CRITICAL:
                        risk_score += 1.0
                    elif system.criticality == SystemCriticality.BUSINESS_CRITICAL:
                        risk_score += 0.8
                    elif system.criticality == SystemCriticality.IMPORTANT:
                        risk_score += 0.6

            return {
                "total_impacted": len(impacted_systems),
                "cascade_effects": cascade_effects,
                "risk_score": risk_score,
                "impacted_systems": list(impacted_systems),
                "critical_path_affected": any(
                    any(sys in target_systems for sys in path)
                    for path in self.critical_paths
                ),
            }


class SafetyValidator:
    """Validates response safety for legacy systems."""

    def __init__(self, config: Config):
        self.config = config
        self.safety_rules = self._load_safety_rules()

    def _load_safety_rules(self) -> Dict[str, Any]:
        """Load safety rules and constraints."""
        return {
            "max_concurrent_actions": 3,
            "max_downtime_tolerance": {
                SystemCriticality.MISSION_CRITICAL: 30,  # 30 seconds
                SystemCriticality.BUSINESS_CRITICAL: 300,  # 5 minutes
                SystemCriticality.IMPORTANT: 1800,  # 30 minutes
                SystemCriticality.STANDARD: 3600,  # 1 hour
                SystemCriticality.LOW_PRIORITY: 14400,  # 4 hours
            },
            "prohibited_combinations": [
                # Don't quarantine and isolate same system simultaneously
                {ResponseType.QUARANTINE, ResponseType.ISOLATE_SYSTEM},
                # Don't shutdown service and patch simultaneously
                {ResponseType.SHUTDOWN_SERVICE, ResponseType.PATCH_VULNERABILITY},
            ],
            "protocol_constraints": {
                ProtocolType.MODBUS: {
                    "prohibited_actions": [
                        ResponseType.NETWORK_SEGMENTATION,
                        ResponseType.ISOLATE_SYSTEM,
                    ],
                    "max_response_time": 100,  # milliseconds
                },
                ProtocolType.HL7_MLLP: {
                    "required_graceful_shutdown": True,
                    "message_completion_required": True,
                },
            },
        }

    def validate_response_plan(
        self,
        response: AutomatedResponse,
        systems: List[LegacySystem],
        dependency_analyzer: DependencyAnalyzer,
    ) -> Tuple[bool, List[str]]:
        """Validate entire response plan for safety."""

        violations = []

        # Check concurrent action limits
        if len(response.actions) > self.safety_rules["max_concurrent_actions"]:
            violations.append(f"Too many concurrent actions: {len(response.actions)}")
            SAFETY_VIOLATIONS.labels(constraint_type="concurrent_actions").inc()

        # Check action combinations
        action_types = {action.action_type for action in response.actions}
        for prohibited in self.safety_rules["prohibited_combinations"]:
            if prohibited.issubset(action_types):
                violations.append(f"Prohibited action combination: {prohibited}")
                SAFETY_VIOLATIONS.labels(constraint_type="action_combination").inc()

        # Check system-specific constraints
        for action in response.actions:
            action_violations = self._validate_action(
                action, systems, dependency_analyzer
            )
            violations.extend(action_violations)

        # Check dependency impact
        target_systems = []
        for action in response.actions:
            target_systems.extend(action.target_systems)

        impact = dependency_analyzer.get_impact_analysis(target_systems)
        if impact["critical_path_affected"]:
            violations.append("Action affects critical dependency path")
            SAFETY_VIOLATIONS.labels(constraint_type="critical_path").inc()

        if impact["risk_score"] > 2.0:  # High risk threshold
            violations.append(f"High dependency risk score: {impact['risk_score']}")
            SAFETY_VIOLATIONS.labels(constraint_type="high_risk").inc()

        return len(violations) == 0, violations

    def _validate_action(
        self,
        action: ResponseAction,
        systems: List[LegacySystem],
        dependency_analyzer: DependencyAnalyzer,
    ) -> List[str]:
        """Validate individual action against safety constraints."""

        violations = []

        for system_id in action.target_systems:
            system = next((s for s in systems if s.system_id == system_id), None)
            if not system:
                continue

            # Check protocol constraints
            for protocol in system.protocol_types:
                constraints = self.safety_rules["protocol_constraints"].get(
                    protocol, {}
                )

                if action.action_type in constraints.get("prohibited_actions", []):
                    violations.append(
                        f"Action {action.action_type.value} prohibited for {protocol.value} protocol"
                    )

                if "max_response_time" in constraints:
                    if action.timeout_seconds * 1000 > constraints["max_response_time"]:
                        violations.append(
                            f"Action timeout too high for {protocol.value}: "
                            f"{action.timeout_seconds}s > {constraints['max_response_time']}ms"
                        )

            # Check downtime constraints
            max_downtime = self.safety_rules["max_downtime_tolerance"].get(
                system.criticality, 3600
            )
            if action.estimated_downtime and action.estimated_downtime > max_downtime:
                violations.append(
                    f"Estimated downtime {action.estimated_downtime}s exceeds limit "
                    f"{max_downtime}s for {system.criticality.value} system"
                )

        return violations


class LegacyAwareResponseManager:
    """
    Manages security responses for legacy systems with safety constraints.

    This manager ensures that security responses are executed safely on legacy
    systems while considering protocol constraints, dependencies, and business impact.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core services
        self.llm_service: Optional[UnifiedLLMService] = None
        self.metrics_collector: Optional[MetricsCollector] = None

        # Execution infrastructure
        self.executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="legacy-response-"
        )

        # Response tracking
        self.active_responses: Dict[str, Dict[str, Any]] = {}
        self.quarantine_registry: Dict[str, QuarantineResult] = {}

        # Safety components
        self.safety_validator: Optional[SafetyValidator] = None

        # State management
        self._initialized = False

        self.logger.info("Legacy-Aware Response Manager initialized")

    async def initialize(self) -> None:
        """Initialize the response manager and its dependencies."""
        if self._initialized:
            return

        try:
            self.logger.info("Initializing Legacy-Aware Response Manager...")

            # Initialize LLM service
            self.llm_service = get_llm_service()
            if (
                not hasattr(self.llm_service, "_initialized")
                or not self.llm_service._initialized
            ):
                await self.llm_service.initialize()

            # Initialize safety validator
            self.safety_validator = SafetyValidator(self.config)

            # Initialize metrics collector
            if hasattr(self.config, "metrics_enabled") and self.config.metrics_enabled:
                self.metrics_collector = MetricsCollector(self.config)
                await self.metrics_collector.initialize()

            self._initialized = True
            self.logger.info("Legacy-Aware Response Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Response Manager: {e}")
            raise SecurityException(f"Response Manager initialization failed: {e}")

    async def execute_response(
        self,
        response: AutomatedResponse,
        legacy_systems: List[LegacySystem],
        security_context: Optional[SecurityContext] = None,
    ) -> Dict[str, Any]:
        """
        Execute automated response with legacy system safety considerations.

        Args:
            response: The automated response plan to execute
            legacy_systems: Legacy systems information
            security_context: Current security context

        Returns:
            Execution results with success status and details
        """

        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        response_id = response.response_id

        try:
            self.logger.info(
                f"Executing response {response_id} with {len(response.actions)} actions"
            )

            # Step 1: Analyze system capabilities
            capabilities = self._analyze_system_capabilities(legacy_systems)

            # Step 2: Build dependency graph
            dependency_analyzer = DependencyAnalyzer(legacy_systems)

            # Step 3: Validate response safety
            is_safe, violations = self.safety_validator.validate_response_plan(
                response, legacy_systems, dependency_analyzer
            )

            if not is_safe:
                self.logger.warning(
                    f"Safety violations detected for response {response_id}: {violations}"
                )
                if not response.requires_human_approval:
                    # Force human approval for unsafe responses
                    response.requires_human_approval = True
                    return await self._create_safety_violation_result(
                        response_id, violations
                    )

            # Step 4: Check approval status
            if response.requires_human_approval and not response.approved_by:
                return await self._create_pending_approval_result(response_id)

            # Step 5: Execute actions with legacy awareness
            execution_results = await self._execute_actions_safely(
                response, capabilities, dependency_analyzer, security_context
            )

            # Update metrics
            execution_time = time.time() - start_time

            # Track active response
            self.active_responses[response_id] = {
                "response": response,
                "start_time": start_time,
                "status": "completed",
                "results": execution_results,
            }

            self.logger.info(
                f"Response {response_id} executed in {execution_time:.2f}s with "
                f"{execution_results['successful_actions']}/{len(response.actions)} successful actions"
            )

            return execution_results

        except Exception as e:
            self.logger.error(f"Response execution failed for {response_id}: {e}")

            # Track failed response
            self.active_responses[response_id] = {
                "response": response,
                "start_time": start_time,
                "status": "failed",
                "error": str(e),
            }

            raise ResponseExecutionException(f"Response execution failed: {e}")

    async def quarantine_legacy_system(
        self,
        system: LegacySystem,
        threat_level: ThreatLevel,
        security_context: Optional[SecurityContext] = None,
    ) -> QuarantineResult:
        """
        Safely quarantine a legacy system without disruption.

        Args:
            system: Legacy system to quarantine
            threat_level: Severity of the threat
            security_context: Current security context

        Returns:
            Quarantine operation result
        """

        start_time = time.time()

        try:
            self.logger.info(
                f"Quarantining legacy system {system.system_name} ({system.system_id})"
            )

            # Step 1: Analyze system dependencies
            dependencies = await self._analyze_dependencies([system])

            # Step 2: Create LLM-powered quarantine plan
            quarantine_plan = await self._create_quarantine_plan(
                system, dependencies, threat_level
            )

            # Step 3: Validate quarantine safety
            safety_check = await self._validate_quarantine_safety(
                system, quarantine_plan, dependencies
            )

            if not safety_check["is_safe"]:
                self.logger.warning(
                    f"Quarantine safety check failed: {safety_check['issues']}"
                )
                # Create partial quarantine plan
                quarantine_plan = await self._create_safe_quarantine_plan(
                    system, safety_check
                )

            # Step 4: Execute quarantine with monitoring
            quarantine_result = await self._execute_safe_quarantine(
                system, quarantine_plan
            )

            # Update metrics
            execution_time = time.time() - start_time
            QUARANTINE_DURATION.observe(execution_time)
            QUARANTINE_COUNTER.labels(
                system_type=system.system_type,
                status="success" if quarantine_result.success else "failed",
            ).inc()

            # Register quarantine
            self.quarantine_registry[system.system_id] = quarantine_result

            self.logger.info(
                f"Quarantine completed for {system.system_id} in {execution_time:.2f}s: "
                f"success={quarantine_result.success}"
            )

            return quarantine_result

        except Exception as e:
            QUARANTINE_COUNTER.labels(
                system_type=system.system_type, status="error"
            ).inc()
            self.logger.error(f"Quarantine failed for system {system.system_id}: {e}")

            # Create failed quarantine result
            return QuarantineResult(
                system_id=system.system_id,
                status="failed",
                success=False,
                error_message=str(e),
                quarantine_start=datetime.now(),
            )

    async def release_quarantine(self, system_id: str) -> bool:
        """Release system from quarantine."""

        if system_id not in self.quarantine_registry:
            self.logger.warning(f"No active quarantine found for system {system_id}")
            return False

        quarantine = self.quarantine_registry[system_id]

        try:
            self.logger.info(f"Releasing quarantine for system {system_id}")

            # Execute release procedure
            if quarantine.isolation_method == "vlan_isolation":
                await self._release_vlan_isolation(system_id, quarantine)
            elif quarantine.isolation_method == "firewall_rules":
                await self._release_firewall_isolation(system_id, quarantine)
            elif quarantine.isolation_method == "network_segmentation":
                await self._release_network_segmentation(system_id, quarantine)

            # Update quarantine status
            quarantine.status = "released"
            quarantine.quarantine_end = datetime.now()

            self.logger.info(f"Successfully released quarantine for system {system_id}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to release quarantine for system {system_id}: {e}"
            )
            quarantine.error_message = f"Release failed: {e}"
            return False

    def _analyze_system_capabilities(
        self, legacy_systems: List[LegacySystem]
    ) -> Dict[str, LegacySystemCapability]:
        """Analyze capabilities of all legacy systems."""

        capabilities = {}
        for system in legacy_systems:
            capabilities[system.system_id] = LegacySystemCapability(system)

        return capabilities

    async def _execute_actions_safely(
        self,
        response: AutomatedResponse,
        capabilities: Dict[str, LegacySystemCapability],
        dependency_analyzer: DependencyAnalyzer,
        security_context: Optional[SecurityContext],
    ) -> Dict[str, Any]:
        """Execute response actions with legacy system safety."""

        results = {
            "successful_actions": 0,
            "failed_actions": 0,
            "skipped_actions": 0,
            "action_results": [],
            "overall_success": False,
            "execution_summary": "",
        }

        # Sort actions by priority
        sorted_actions = sorted(response.actions, key=lambda a: a.priority)

        # Execute actions sequentially for safety
        for action in sorted_actions:
            try:
                action_result = await self._execute_single_action(
                    action, capabilities, dependency_analyzer, security_context
                )

                results["action_results"].append(action_result)

                if action_result["success"]:
                    results["successful_actions"] += 1
                elif action_result["skipped"]:
                    results["skipped_actions"] += 1
                else:
                    results["failed_actions"] += 1

                # Update metrics
                for system_id in action.target_systems:
                    system = next(
                        (
                            s
                            for cap in capabilities.values()
                            for s in [cap.system]
                            if s.system_id == system_id
                        ),
                        None,
                    )
                    if system:
                        protocols = [p.value for p in system.protocol_types] or [
                            "unknown"
                        ]
                        for protocol in protocols:
                            LEGACY_RESPONSE_COUNTER.labels(
                                protocol=protocol, action=action.action_type.value
                            ).inc()

                # Stop execution if critical action fails
                if not action_result["success"] and action.priority == 1:
                    self.logger.warning(
                        f"Critical action failed, stopping execution: {action.action_type}"
                    )
                    break

                # Add delay between actions for safety
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Action execution error: {e}")
                results["action_results"].append(
                    {
                        "action_id": action.action_id,
                        "action_type": action.action_type.value,
                        "success": False,
                        "skipped": False,
                        "error": str(e),
                    }
                )
                results["failed_actions"] += 1

        # Calculate overall success
        results["overall_success"] = (
            results["successful_actions"] > 0 and results["failed_actions"] == 0
        )

        results["execution_summary"] = (
            f"Executed {results['successful_actions'] + results['failed_actions']} actions: "
            f"{results['successful_actions']} successful, {results['failed_actions']} failed, "
            f"{results['skipped_actions']} skipped"
        )

        return results

    async def _execute_single_action(
        self,
        action: ResponseAction,
        capabilities: Dict[str, LegacySystemCapability],
        dependency_analyzer: DependencyAnalyzer,
        security_context: Optional[SecurityContext],
    ) -> Dict[str, Any]:
        """Execute a single response action safely."""

        action_start = time.time()
        result = {
            "action_id": action.action_id,
            "action_type": action.action_type.value,
            "success": False,
            "skipped": False,
            "error": None,
            "details": {},
            "execution_time": 0.0,
        }

        try:
            # Check if action is supported by target systems
            for system_id in action.target_systems:
                capability = capabilities.get(system_id)
                if (
                    not capability
                    or action.action_type not in capability.supported_actions
                ):
                    result["skipped"] = True
                    result["error"] = f"Action not supported by system {system_id}"
                    return result

            # Execute action based on type
            if action.action_type == ResponseType.ALERT_SECURITY_TEAM:
                result["details"] = await self._execute_alert_action(
                    action, security_context
                )
                result["success"] = True

            elif action.action_type == ResponseType.ENABLE_MONITORING:
                result["details"] = await self._execute_monitoring_action(
                    action, capabilities
                )
                result["success"] = True

            elif action.action_type == ResponseType.BLOCK_IP:
                result["details"] = await self._execute_block_ip_action(
                    action, capabilities
                )
                result["success"] = result["details"].get("blocks_applied", 0) > 0

            elif action.action_type == ResponseType.QUARANTINE:
                result["details"] = await self._execute_quarantine_action(
                    action, capabilities
                )
                result["success"] = result["details"].get("systems_quarantined", 0) > 0

            elif action.action_type == ResponseType.VIRTUAL_PATCH:
                result["details"] = await self._execute_virtual_patch_action(
                    action, capabilities
                )
                result["success"] = result["details"].get("patches_applied", 0) > 0

            elif action.action_type == ResponseType.NETWORK_SEGMENTATION:
                result["details"] = await self._execute_segmentation_action(
                    action, capabilities
                )
                result["success"] = result["details"].get("segments_created", 0) > 0

            else:
                result["skipped"] = True
                result["error"] = (
                    f"Action type {action.action_type.value} not implemented"
                )

        except Exception as e:
            result["error"] = str(e)
            result["success"] = False

        finally:
            result["execution_time"] = time.time() - action_start

        return result

    async def _execute_alert_action(
        self, action: ResponseAction, security_context: Optional[SecurityContext]
    ) -> Dict[str, Any]:
        """Execute security team alert."""

        alert_details = {
            "alert_sent": True,
            "recipients": ["security_team"],
            "channels": ["email", "slack", "siem"],
            "urgency": (
                "high"
                if action.risk_level in {ThreatLevel.CRITICAL, ThreatLevel.HIGH}
                else "medium"
            ),
        }

        # In real implementation, this would integrate with alerting systems
        self.logger.info(f"Security alert sent for action {action.action_id}")

        return alert_details

    async def _execute_monitoring_action(
        self, action: ResponseAction, capabilities: Dict[str, LegacySystemCapability]
    ) -> Dict[str, Any]:
        """Execute enhanced monitoring setup."""

        monitoring_details = {
            "monitoring_enabled": True,
            "systems_monitored": len(action.target_systems),
            "monitoring_level": action.parameters.get("monitoring_level", "standard"),
            "duration": action.parameters.get("duration", "24h"),
        }

        # Enable monitoring for each target system
        for system_id in action.target_systems:
            capability = capabilities.get(system_id)
            if capability:
                # Configure protocol-specific monitoring
                for protocol in capability.system.protocol_types:
                    monitoring_details[f"{protocol.value}_monitoring"] = True

        self.logger.info(
            f"Enhanced monitoring enabled for {len(action.target_systems)} systems"
        )

        return monitoring_details

    async def _execute_block_ip_action(
        self, action: ResponseAction, capabilities: Dict[str, LegacySystemCapability]
    ) -> Dict[str, Any]:
        """Execute IP blocking action."""

        ip_address = action.parameters.get("ip_address")
        if not ip_address:
            raise ResponseExecutionException("No IP address specified for blocking")

        block_details = {
            "ip_blocked": ip_address,
            "blocks_applied": 0,
            "scope": action.parameters.get("scope", "organization"),
            "duration": action.parameters.get("duration", "24h"),
        }

        # Apply IP block at network level (firewall, router, etc.)
        # In real implementation, this would integrate with network security tools
        block_details["blocks_applied"] = 1

        self.logger.info(f"IP {ip_address} blocked with scope {block_details['scope']}")

        return block_details

    async def _execute_quarantine_action(
        self, action: ResponseAction, capabilities: Dict[str, LegacySystemCapability]
    ) -> Dict[str, Any]:
        """Execute system quarantine action."""

        quarantine_details = {
            "systems_quarantined": 0,
            "isolation_method": action.parameters.get(
                "isolation_type", "network_segmentation"
            ),
            "quarantine_results": [],
        }

        for system_id in action.target_systems:
            capability = capabilities.get(system_id)
            if capability:
                try:
                    # Create quarantine for individual system
                    quarantine_result = await self.quarantine_legacy_system(
                        capability.system, action.risk_level
                    )

                    quarantine_details["quarantine_results"].append(
                        {
                            "system_id": system_id,
                            "success": quarantine_result.success,
                            "quarantine_id": quarantine_result.quarantine_id,
                        }
                    )

                    if quarantine_result.success:
                        quarantine_details["systems_quarantined"] += 1

                except Exception as e:
                    quarantine_details["quarantine_results"].append(
                        {"system_id": system_id, "success": False, "error": str(e)}
                    )

        return quarantine_details

    async def _execute_virtual_patch_action(
        self, action: ResponseAction, capabilities: Dict[str, LegacySystemCapability]
    ) -> Dict[str, Any]:
        """Execute virtual patching action."""

        patch_details = {
            "patches_applied": 0,
            "patch_type": "ips_rules",
            "systems_protected": [],
            "patch_rules": [],
        }

        # Create virtual patches based on system protocols and vulnerabilities
        for system_id in action.target_systems:
            capability = capabilities.get(system_id)
            if capability:
                # Generate protocol-specific virtual patches
                for protocol in capability.system.protocol_types:
                    patch_rule = await self._generate_virtual_patch_rule(
                        protocol, capability.system
                    )
                    patch_details["patch_rules"].append(patch_rule)
                    patch_details["systems_protected"].append(system_id)

                patch_details["patches_applied"] += 1

        self.logger.info(
            f"Virtual patches applied to {patch_details['patches_applied']} systems"
        )

        return patch_details

    async def _execute_segmentation_action(
        self, action: ResponseAction, capabilities: Dict[str, LegacySystemCapability]
    ) -> Dict[str, Any]:
        """Execute network segmentation action."""

        segmentation_details = {
            "segments_created": 0,
            "segmentation_type": "vlan_based",
            "isolated_systems": [],
            "allowed_communications": [],
        }

        for system_id in action.target_systems:
            capability = capabilities.get(system_id)
            if capability:
                # Create isolated network segment
                segment_config = await self._create_network_segment(capability.system)
                segmentation_details["isolated_systems"].append(
                    {
                        "system_id": system_id,
                        "segment_id": segment_config["segment_id"],
                        "isolation_level": segment_config["isolation_level"],
                    }
                )

                segmentation_details["segments_created"] += 1

        return segmentation_details

    async def _analyze_dependencies(
        self, systems: List[LegacySystem]
    ) -> Dict[str, Any]:
        """Analyze system dependencies for quarantine planning."""

        dependencies = {
            "direct_dependencies": [],
            "indirect_dependencies": [],
            "critical_dependencies": [],
            "safe_to_isolate": True,
        }

        for system in systems:
            # Collect direct dependencies
            dependencies["direct_dependencies"].extend(system.dependent_systems)

            # Check for critical systems in dependency chain
            for dep_id in system.dependent_systems:
                # In real implementation, lookup dependent system details
                dependencies["critical_dependencies"].append(dep_id)

            # Assess isolation safety
            if system.criticality in {
                SystemCriticality.MISSION_CRITICAL,
                SystemCriticality.BUSINESS_CRITICAL,
            }:
                if system.dependent_systems:
                    dependencies["safe_to_isolate"] = False

        return dependencies

    async def _create_quarantine_plan(
        self,
        system: LegacySystem,
        dependencies: Dict[str, Any],
        threat_level: ThreatLevel,
    ) -> Dict[str, Any]:
        """Create LLM-powered quarantine plan."""

        context = {
            "system": {
                "name": system.system_name,
                "type": system.system_type,
                "criticality": system.criticality.value,
                "protocols": [p.value for p in system.protocol_types],
                "dependencies": len(system.dependent_systems),
            },
            "threat_level": threat_level.value,
            "dependencies": dependencies,
            "constraints": {
                "uptime_requirement": system.uptime_requirements,
                "compliance_requirements": system.compliance_requirements,
            },
        }

        prompt = f"""Create a safe quarantine plan for this legacy system:

System Information:
- Name: {system.system_name}
- Type: {system.system_type} 
- Criticality: {system.criticality.value}
- Protocols: {', '.join([p.value for p in system.protocol_types])}
- Uptime Requirement: {system.uptime_requirements}%
- Dependencies: {len(system.dependent_systems)} systems depend on this
- Compliance: {', '.join(system.compliance_requirements)}

Threat Level: {threat_level.value}

Dependency Analysis: {json.dumps(dependencies, indent=2)}

Create a quarantine plan that:
1. Minimizes business disruption
2. Maintains critical dependencies 
3. Provides effective security isolation
4. Considers protocol-specific requirements
5. Includes rollback procedures

Provide the plan in JSON format with isolation_method, duration, safety_checks, and rollback_plan."""

        try:
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="security_orchestrator",
                context=context,
                max_tokens=1500,
                temperature=0.1,
            )

            response = await self.llm_service.process_request(llm_request)

            # Parse LLM response into quarantine plan
            plan = self._parse_quarantine_plan(response.content)

            # Add safety defaults
            plan.setdefault("isolation_method", "network_segmentation")
            plan.setdefault("duration", "72h")
            plan.setdefault("rollback_plan", "Reverse isolation configuration")

            return plan

        except Exception as e:
            self.logger.warning(f"LLM quarantine planning failed: {e}")

            # Fallback to safe default plan
            return {
                "isolation_method": "monitoring_only",
                "duration": "24h",
                "safety_checks": ["dependency_validation", "business_hours_check"],
                "rollback_plan": "Disable monitoring and restore normal operations",
                "fallback_reason": str(e),
            }

    def _parse_quarantine_plan(self, llm_content: str) -> Dict[str, Any]:
        """Parse LLM quarantine plan response."""

        try:
            # Extract JSON from response
            start_idx = llm_content.find("{")
            end_idx = llm_content.rfind("}") + 1

            if start_idx != -1 and end_idx != -1:
                json_str = llm_content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return self._fallback_parse_quarantine_plan(llm_content)

        except json.JSONDecodeError:
            return self._fallback_parse_quarantine_plan(llm_content)

    def _fallback_parse_quarantine_plan(self, content: str) -> Dict[str, Any]:
        """Fallback parsing for quarantine plan."""

        plan = {
            "isolation_method": "monitoring_only",
            "duration": "24h",
            "safety_checks": ["manual_approval"],
            "rollback_plan": "Manual restoration",
        }

        content_lower = content.lower()

        if "vlan" in content_lower:
            plan["isolation_method"] = "vlan_isolation"
        elif "firewall" in content_lower:
            plan["isolation_method"] = "firewall_rules"
        elif "segmentation" in content_lower:
            plan["isolation_method"] = "network_segmentation"

        return plan

    async def _validate_quarantine_safety(
        self, system: LegacySystem, plan: Dict[str, Any], dependencies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate quarantine plan safety."""

        safety_check = {"is_safe": True, "issues": [], "recommendations": []}

        # Check business impact
        if system.criticality == SystemCriticality.MISSION_CRITICAL:
            if plan.get("isolation_method") in [
                "complete_isolation",
                "system_shutdown",
            ]:
                safety_check["is_safe"] = False
                safety_check["issues"].append(
                    "Mission-critical system cannot be completely isolated"
                )

        # Check dependencies
        if not dependencies.get("safe_to_isolate", True):
            safety_check["is_safe"] = False
            safety_check["issues"].append(
                "System has critical dependencies that would be affected"
            )

        # Check protocol constraints
        for protocol in system.protocol_types:
            if protocol == ProtocolType.MODBUS and "isolation" in plan.get(
                "isolation_method", ""
            ):
                safety_check["issues"].append(
                    "MODBUS isolation may cause industrial safety issues"
                )
                safety_check["recommendations"].append(
                    "Use monitoring and virtual patching instead"
                )

        return safety_check

    async def _create_safe_quarantine_plan(
        self, system: LegacySystem, safety_check: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a safer quarantine plan based on safety check results."""

        safe_plan = {
            "isolation_method": "enhanced_monitoring",
            "duration": "24h",
            "safety_checks": ["continuous_health_monitoring"],
            "rollback_plan": "Disable monitoring",
            "restrictions": ["no_network_isolation", "maintain_dependencies"],
        }

        # Add protocol-specific safe measures
        for protocol in system.protocol_types:
            if protocol == ProtocolType.HL7_MLLP:
                safe_plan["restrictions"].append("preserve_patient_data_flow")
            elif protocol == ProtocolType.MODBUS:
                safe_plan["restrictions"].append("maintain_control_system_operation")
            elif protocol == ProtocolType.ISO8583:
                safe_plan["restrictions"].append("preserve_transaction_integrity")

        return safe_plan

    async def _execute_safe_quarantine(
        self, system: LegacySystem, plan: Dict[str, Any]
    ) -> QuarantineResult:
        """Execute quarantine plan safely."""

        quarantine_result = QuarantineResult(
            system_id=system.system_id,
            quarantine_type=plan.get("isolation_method", "monitoring_only"),
            isolation_method=plan.get("isolation_method", "monitoring_only"),
            quarantine_start=datetime.now(),
        )

        try:
            # Execute quarantine based on method
            isolation_method = plan.get("isolation_method", "monitoring_only")

            if isolation_method == "vlan_isolation":
                await self._execute_vlan_isolation(system, quarantine_result)
            elif isolation_method == "firewall_rules":
                await self._execute_firewall_isolation(system, quarantine_result)
            elif isolation_method == "network_segmentation":
                await self._execute_network_segmentation(system, quarantine_result)
            elif isolation_method == "enhanced_monitoring":
                await self._execute_enhanced_monitoring(system, quarantine_result)
            else:
                # Default to monitoring only
                await self._execute_monitoring_quarantine(system, quarantine_result)

            quarantine_result.success = True
            quarantine_result.status = "active"

        except Exception as e:
            quarantine_result.success = False
            quarantine_result.status = "failed"
            quarantine_result.error_message = str(e)

        return quarantine_result

    async def _execute_vlan_isolation(
        self, system: LegacySystem, quarantine_result: QuarantineResult
    ):
        """Execute VLAN-based isolation."""
        # In real implementation, this would configure network switches
        quarantine_result.affected_interfaces = ["eth0"]
        quarantine_result.isolation_method = "vlan_isolation"
        self.logger.info(f"VLAN isolation executed for system {system.system_id}")

    async def _execute_firewall_isolation(
        self, system: LegacySystem, quarantine_result: QuarantineResult
    ):
        """Execute firewall-based isolation."""
        # In real implementation, this would configure firewall rules
        quarantine_result.isolation_method = "firewall_rules"
        self.logger.info(f"Firewall isolation executed for system {system.system_id}")

    async def _execute_network_segmentation(
        self, system: LegacySystem, quarantine_result: QuarantineResult
    ):
        """Execute network segmentation."""
        # In real implementation, this would create network segments
        quarantine_result.isolation_method = "network_segmentation"
        self.logger.info(f"Network segmentation executed for system {system.system_id}")

    async def _execute_enhanced_monitoring(
        self, system: LegacySystem, quarantine_result: QuarantineResult
    ):
        """Execute enhanced monitoring instead of isolation."""
        quarantine_result.isolation_method = "enhanced_monitoring"
        quarantine_result.monitoring_enabled = True
        self.logger.info(f"Enhanced monitoring enabled for system {system.system_id}")

    async def _execute_monitoring_quarantine(
        self, system: LegacySystem, quarantine_result: QuarantineResult
    ):
        """Execute monitoring-only quarantine."""
        quarantine_result.isolation_method = "monitoring_only"
        quarantine_result.monitoring_enabled = True
        self.logger.info(
            f"Monitoring quarantine executed for system {system.system_id}"
        )

    # Release methods for different isolation types
    async def _release_vlan_isolation(
        self, system_id: str, quarantine: QuarantineResult
    ):
        """Release VLAN isolation."""
        # Restore original VLAN configuration
        self.logger.info(f"Released VLAN isolation for system {system_id}")

    async def _release_firewall_isolation(
        self, system_id: str, quarantine: QuarantineResult
    ):
        """Release firewall isolation."""
        # Remove firewall rules
        self.logger.info(f"Released firewall isolation for system {system_id}")

    async def _release_network_segmentation(
        self, system_id: str, quarantine: QuarantineResult
    ):
        """Release network segmentation."""
        # Remove network segmentation
        self.logger.info(f"Released network segmentation for system {system_id}")

    # Helper methods
    async def _generate_virtual_patch_rule(
        self, protocol: ProtocolType, system: LegacySystem
    ) -> Dict[str, Any]:
        """Generate virtual patch rule for specific protocol."""

        rule = {
            "protocol": protocol.value,
            "system_id": system.system_id,
            "rule_type": "signature_based",
            "enabled": True,
        }

        # Protocol-specific rules
        if protocol == ProtocolType.HL7_MLLP:
            rule.update(
                {
                    "port": 2575,
                    "pattern": "malformed_hl7_message",
                    "action": "drop_and_alert",
                }
            )
        elif protocol == ProtocolType.MODBUS:
            rule.update(
                {
                    "port": 502,
                    "pattern": "unauthorized_function_code",
                    "action": "block_and_log",
                }
            )

        return rule

    async def _create_network_segment(self, system: LegacySystem) -> Dict[str, Any]:
        """Create isolated network segment for system."""

        segment_config = {
            "segment_id": f"quarantine_{system.system_id}_{int(time.time())}",
            "isolation_level": "partial",
            "allowed_traffic": [],
            "blocked_traffic": ["internet", "cross_segment"],
        }

        # Allow essential traffic based on system type
        if system.criticality in {
            SystemCriticality.MISSION_CRITICAL,
            SystemCriticality.BUSINESS_CRITICAL,
        }:
            segment_config["allowed_traffic"].extend(
                ["management", "monitoring", "backup"]
            )
            segment_config["isolation_level"] = "partial"
        else:
            segment_config["isolation_level"] = "strict"

        return segment_config

    # Safety and result creation methods
    async def _create_safety_violation_result(
        self, response_id: str, violations: List[str]
    ) -> Dict[str, Any]:
        """Create result for safety violation case."""
        return {
            "status": "blocked_by_safety",
            "response_id": response_id,
            "violations": violations,
            "action_taken": "forced_human_approval",
            "message": "Response blocked due to safety violations - human approval required",
        }

    async def _create_pending_approval_result(self, response_id: str) -> Dict[str, Any]:
        """Create result for pending approval case."""
        return {
            "status": "pending_approval",
            "response_id": response_id,
            "message": "Response requires human approval before execution",
            "approval_required": True,
        }

    def get_active_quarantines(self) -> Dict[str, QuarantineResult]:
        """Get all active quarantine operations."""
        return {
            qid: quarantine
            for qid, quarantine in self.quarantine_registry.items()
            if quarantine.status == "active"
        }

    def get_response_status(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a response execution."""
        return self.active_responses.get(response_id)

    async def get_response_metrics(self) -> Dict[str, Any]:
        """Get response execution metrics."""
        active_quarantines = len(self.get_active_quarantines())
        active_responses = len(
            [r for r in self.active_responses.values() if r["status"] == "executing"]
        )

        return {
            "active_quarantines": active_quarantines,
            "active_responses": active_responses,
            "total_responses_executed": len(self.active_responses),
            "quarantine_registry_size": len(self.quarantine_registry),
        }

    async def shutdown(self):
        """Shutdown the response manager."""
        self.logger.info("Shutting down Legacy-Aware Response Manager...")

        # Release all active quarantines
        for system_id in list(self.quarantine_registry.keys()):
            await self.release_quarantine(system_id)

        # Shutdown executor
        self.executor.shutdown(wait=True)

        self._initialized = False
        self.logger.info("Legacy-Aware Response Manager shutdown complete")
