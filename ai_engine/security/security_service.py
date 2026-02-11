"""
QBITEL Engine - Security Orchestrator Service

Main service orchestrating zero-touch security operations with enterprise-grade reliability.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from ..core.config import Config
from ..core.exceptions import QbitelAIException
from ..monitoring.metrics import MetricsCollector
from .models import (
    SecurityEvent,
    ThreatAnalysis,
    AutomatedResponse,
    SecurityContext,
    LegacySystem,
    QuarantineResult,
    SecurityMetrics,
    ThreatIntelligence,
    SecurityException,
    ThreatAnalysisException,
    ResponseExecutionException,
)
from .decision_engine import ZeroTouchDecisionEngine
from .legacy_response import LegacyAwareResponseManager
from .threat_analyzer import ThreatAnalyzer

from prometheus_client import Counter, Histogram, Gauge, Summary


_METRIC_CACHE = {}


def _get_metric(metric_cls, name: str, *args, **kwargs):
    """Return cached Prometheus metric or create an unregistered one."""

    if name in _METRIC_CACHE:
        return _METRIC_CACHE[name]

    kwargs = dict(kwargs)
    kwargs.setdefault("registry", None)
    metric = metric_cls(name, *args, **kwargs)
    _METRIC_CACHE[name] = metric
    return metric


# Prometheus metrics
SECURITY_EVENTS_COUNTER = _get_metric(
    Counter,
    "qbitel_security_events_total",
    "Total security events processed",
    ["event_type", "severity"],
)
RESPONSE_TIME_HISTOGRAM = _get_metric(
    Histogram,
    "qbitel_security_response_time_seconds",
    "Security response time",
    ["response_type"],
)
ACTIVE_INCIDENTS_GAUGE = _get_metric(
    Gauge, "qbitel_security_active_incidents", "Number of active security incidents"
)
AUTONOMOUS_DECISIONS_COUNTER = _get_metric(
    Counter,
    "qbitel_security_autonomous_decisions_total",
    "Autonomous decisions made",
    ["decision_outcome"],
)
HUMAN_ESCALATIONS_COUNTER = _get_metric(
    Counter,
    "qbitel_security_human_escalations_total",
    "Human escalations",
    ["escalation_reason"],
)
SYSTEM_QUARANTINES_GAUGE = _get_metric(
    Gauge,
    "qbitel_security_active_quarantines",
    "Number of systems currently quarantined",
)
MTTR_SUMMARY = _get_metric(
    Summary,
    "qbitel_security_mean_time_to_response_seconds",
    "Mean time to response for security incidents",
)

logger = logging.getLogger(__name__)


class SecurityOrchestratorService:
    """
    Main security orchestrator service providing zero-touch security operations.

    This service integrates threat analysis, decision making, and response execution
    with enterprise-grade reliability, monitoring, and legacy system awareness.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core components
        self.decision_engine: Optional[ZeroTouchDecisionEngine] = None
        self.response_manager: Optional[LegacyAwareResponseManager] = None
        self.threat_analyzer: Optional[ThreatAnalyzer] = None

        # Infrastructure
        self.metrics_collector: Optional[MetricsCollector] = None

        # State management
        self.active_incidents: Dict[str, Dict[str, Any]] = {}
        self.security_context = SecurityContext()
        self.legacy_systems: Dict[str, LegacySystem] = {}

        # Performance tracking
        self.performance_metrics = SecurityMetrics()

        # Service state
        self._initialized = False
        self._running = False
        self._background_tasks: List[asyncio.Task] = []

        # Configuration
        self.max_concurrent_incidents = getattr(config, "max_concurrent_incidents", 50)
        self.incident_retention_hours = getattr(config, "incident_retention_hours", 24)
        self.auto_cleanup_enabled = getattr(config, "auto_cleanup_enabled", True)

        self.logger.info("Security Orchestrator Service initialized")

    async def initialize(self) -> None:
        """Initialize the security orchestrator service."""
        if self._initialized:
            return

        try:
            self.logger.info("Initializing Security Orchestrator Service...")

            # Initialize core components
            self.decision_engine = ZeroTouchDecisionEngine(self.config)
            await self.decision_engine.initialize()

            self.response_manager = LegacyAwareResponseManager(self.config)
            await self.response_manager.initialize()

            self.threat_analyzer = ThreatAnalyzer(self.config)
            await self.threat_analyzer.initialize()

            # Initialize metrics collector
            if hasattr(self.config, "metrics_enabled") and self.config.metrics_enabled:
                self.metrics_collector = MetricsCollector(self.config)
                await self.metrics_collector.initialize()

            # Start background tasks
            await self._start_background_tasks()

            self._initialized = True
            self._running = True

            self.logger.info("Security Orchestrator Service initialized successfully")

        except Exception as e:
            self.logger.error(
                f"Failed to initialize Security Orchestrator Service: {e}"
            )
            raise SecurityException(f"Service initialization failed: {e}")

    async def process_security_event(
        self,
        security_event: SecurityEvent,
        legacy_systems: Optional[List[LegacySystem]] = None,
        auto_execute: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Process a security event through the complete zero-touch pipeline.

        Args:
            security_event: The security event to process
            legacy_systems: Legacy systems context
            auto_execute: Override auto-execution setting

        Returns:
            Complete processing results including analysis and response
        """

        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        incident_id = str(uuid.uuid4())

        try:
            self.logger.info(
                f"Processing security event {security_event.event_id} as incident {incident_id}"
            )

            # Update metrics
            SECURITY_EVENTS_COUNTER.labels(
                event_type=security_event.event_type.value,
                severity=security_event.threat_level.value,
            ).inc()

            # Register incident
            incident_data = {
                "incident_id": incident_id,
                "event": security_event,
                "start_time": start_time,
                "status": "processing",
                "stages": {
                    "threat_analysis": {"status": "pending"},
                    "decision_making": {"status": "pending"},
                    "response_execution": {"status": "pending"},
                },
            }

            self.active_incidents[incident_id] = incident_data
            ACTIVE_INCIDENTS_GAUGE.set(len(self.active_incidents))

            # Update security context
            await self._update_security_context(security_event)

            # Get legacy systems context
            if not legacy_systems:
                legacy_systems = await self._get_legacy_systems_context(security_event)

            # Stage 1: Threat Analysis
            self.logger.info(f"Starting threat analysis for incident {incident_id}")
            incident_data["stages"]["threat_analysis"]["status"] = "running"

            threat_analysis = await self.threat_analyzer.analyze_threat(
                security_event, self.security_context, legacy_systems
            )

            incident_data["stages"]["threat_analysis"].update(
                {
                    "status": "completed",
                    "result": threat_analysis,
                    "duration": time.time() - start_time,
                }
            )

            # Stage 2: Decision Making
            self.logger.info(f"Starting decision making for incident {incident_id}")
            decision_start = time.time()
            incident_data["stages"]["decision_making"]["status"] = "running"

            automated_response = await self.decision_engine.analyze_and_respond(
                security_event, self.security_context, legacy_systems
            )

            incident_data["stages"]["decision_making"].update(
                {
                    "status": "completed",
                    "result": automated_response,
                    "duration": time.time() - decision_start,
                }
            )

            # Record decision outcome
            if automated_response.auto_execute:
                AUTONOMOUS_DECISIONS_COUNTER.labels(
                    decision_outcome="auto_execute"
                ).inc()
            elif automated_response.requires_human_approval:
                AUTONOMOUS_DECISIONS_COUNTER.labels(
                    decision_outcome="human_approval"
                ).inc()
                HUMAN_ESCALATIONS_COUNTER.labels(
                    escalation_reason="requires_approval"
                ).inc()
            else:
                AUTONOMOUS_DECISIONS_COUNTER.labels(decision_outcome="deferred").inc()

            # Stage 3: Response Execution
            response_result = None
            if auto_execute is not None:
                automated_response.auto_execute = auto_execute

            if automated_response.auto_execute or (auto_execute is True):
                self.logger.info(
                    f"Executing automated response for incident {incident_id}"
                )
                response_start = time.time()
                incident_data["stages"]["response_execution"]["status"] = "running"

                try:
                    response_result = await self.response_manager.execute_response(
                        automated_response, legacy_systems, self.security_context
                    )

                    incident_data["stages"]["response_execution"].update(
                        {
                            "status": "completed",
                            "result": response_result,
                            "duration": time.time() - response_start,
                        }
                    )

                    # Update metrics
                    with RESPONSE_TIME_HISTOGRAM.labels(
                        response_type="automated"
                    ).time():
                        pass

                except Exception as e:
                    incident_data["stages"]["response_execution"].update(
                        {
                            "status": "failed",
                            "error": str(e),
                            "duration": time.time() - response_start,
                        }
                    )
                    response_result = {"status": "failed", "error": str(e)}
            else:
                incident_data["stages"]["response_execution"].update(
                    {
                        "status": "pending_approval",
                        "message": "Awaiting human approval for execution",
                    }
                )
                response_result = {"status": "pending_approval"}

            # Calculate total processing time
            total_time = time.time() - start_time
            MTTR_SUMMARY.observe(total_time)

            # Update incident status
            incident_data.update(
                {
                    "status": "completed",
                    "total_duration": total_time,
                    "completed_at": datetime.now(),
                }
            )

            # Create comprehensive result
            result = {
                "incident_id": incident_id,
                "event_id": security_event.event_id,
                "status": "completed",
                "processing_time_seconds": total_time,
                "threat_analysis": asdict(threat_analysis),
                "automated_response": asdict(automated_response),
                "response_execution": response_result,
                "stages": incident_data["stages"],
                "recommendations": self._generate_incident_recommendations(
                    threat_analysis, automated_response, response_result
                ),
            }

            self.logger.info(
                f"Security event processing completed for incident {incident_id} in {total_time:.2f}s: "
                f"threat_level={threat_analysis.threat_level.value}, "
                f"confidence={threat_analysis.confidence_score:.2f}, "
                f"auto_executed={automated_response.auto_execute}"
            )

            return result

        except Exception as e:
            # Update incident with error
            if incident_id in self.active_incidents:
                self.active_incidents[incident_id].update(
                    {
                        "status": "failed",
                        "error": str(e),
                        "total_duration": time.time() - start_time,
                    }
                )

            self.logger.error(
                f"Security event processing failed for incident {incident_id}: {e}"
            )

            # Record escalation due to processing failure
            HUMAN_ESCALATIONS_COUNTER.labels(
                escalation_reason="processing_failure"
            ).inc()

            raise SecurityException(f"Security event processing failed: {e}")

    async def execute_response_by_id(
        self, response_id: str, approved_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a previously created automated response.

        Args:
            response_id: ID of the response to execute
            approved_by: ID of the user who approved the response

        Returns:
            Execution results
        """

        if not self._initialized:
            await self.initialize()

        # Find the incident with this response
        incident_data = None
        for incident in self.active_incidents.values():
            response = (
                incident.get("stages", {}).get("decision_making", {}).get("result")
            )
            if response and getattr(response, "response_id", None) == response_id:
                incident_data = incident
                break

        if not incident_data:
            raise SecurityException(
                f"No active incident found with response ID: {response_id}"
            )

        automated_response = incident_data["stages"]["decision_making"]["result"]
        security_event = incident_data["event"]

        # Update approval information
        if approved_by:
            automated_response.approved_by = approved_by
            automated_response.approved_at = datetime.now()

        # Get legacy systems context
        legacy_systems = await self._get_legacy_systems_context(security_event)

        try:
            self.logger.info(f"Executing approved response {response_id}")

            # Execute the response
            response_result = await self.response_manager.execute_response(
                automated_response, legacy_systems, self.security_context
            )

            # Update incident data
            incident_data["stages"]["response_execution"].update(
                {
                    "status": "completed",
                    "result": response_result,
                    "approved_by": approved_by,
                    "executed_at": datetime.now(),
                }
            )

            self.logger.info(f"Response {response_id} executed successfully")

            return {
                "response_id": response_id,
                "status": "executed",
                "execution_result": response_result,
                "approved_by": approved_by,
            }

        except Exception as e:
            # Update incident with execution failure
            incident_data["stages"]["response_execution"].update(
                {
                    "status": "failed",
                    "error": str(e),
                    "approved_by": approved_by,
                    "failed_at": datetime.now(),
                }
            )

            self.logger.error(f"Response execution failed for {response_id}: {e}")
            raise ResponseExecutionException(f"Response execution failed: {e}")

    async def quarantine_system(
        self, system_id: str, threat_level: str = "high", duration_hours: int = 72
    ) -> QuarantineResult:
        """
        Quarantine a specific system.

        Args:
            system_id: ID of the system to quarantine
            threat_level: Severity level justifying quarantine
            duration_hours: Duration of quarantine in hours

        Returns:
            Quarantine operation result
        """

        if not self._initialized:
            await self.initialize()

        # Get system information
        legacy_system = self.legacy_systems.get(system_id)
        if not legacy_system:
            raise SecurityException(f"Unknown system ID: {system_id}")

        from .models import ThreatLevel

        threat_level_enum = ThreatLevel(threat_level.lower())

        try:
            self.logger.info(
                f"Quarantining system {system_id} at threat level {threat_level}"
            )

            quarantine_result = await self.response_manager.quarantine_legacy_system(
                legacy_system, threat_level_enum, self.security_context
            )

            # Update quarantine metrics
            if quarantine_result.success:
                SYSTEM_QUARANTINES_GAUGE.inc()

            return quarantine_result

        except Exception as e:
            self.logger.error(f"System quarantine failed for {system_id}: {e}")
            raise SecurityException(f"System quarantine failed: {e}")

    async def release_quarantine(self, system_id: str) -> bool:
        """
        Release a system from quarantine.

        Args:
            system_id: ID of the system to release

        Returns:
            True if successful, False otherwise
        """

        if not self._initialized:
            await self.initialize()

        try:
            success = await self.response_manager.release_quarantine(system_id)

            if success:
                SYSTEM_QUARANTINES_GAUGE.dec()
                self.logger.info(f"System {system_id} released from quarantine")
            else:
                self.logger.warning(
                    f"Failed to release system {system_id} from quarantine"
                )

            return success

        except Exception as e:
            self.logger.error(f"Quarantine release failed for {system_id}: {e}")
            return False

    async def get_incident_status(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific incident."""
        incident_data = self.active_incidents.get(incident_id)

        if not incident_data:
            return None

        return {
            "incident_id": incident_id,
            "status": incident_data.get("status"),
            "event_id": (
                incident_data.get("event", {}).event_id
                if hasattr(incident_data.get("event", {}), "event_id")
                else None
            ),
            "start_time": incident_data.get("start_time"),
            "duration": time.time() - incident_data.get("start_time", time.time()),
            "stages": incident_data.get("stages", {}),
            "error": incident_data.get("error"),
        }

    async def list_active_incidents(self) -> List[Dict[str, Any]]:
        """List all active incidents."""
        active_list = []

        for incident_id, incident_data in self.active_incidents.items():
            if incident_data.get("status") in ["processing", "pending_approval"]:
                active_list.append(
                    {
                        "incident_id": incident_id,
                        "status": incident_data.get("status"),
                        "event_type": (
                            getattr(incident_data.get("event"), "event_type", {}).value
                            if hasattr(incident_data.get("event", {}), "event_type")
                            else "unknown"
                        ),
                        "threat_level": (
                            getattr(
                                incident_data.get("event"), "threat_level", {}
                            ).value
                            if hasattr(incident_data.get("event", {}), "threat_level")
                            else "unknown"
                        ),
                        "start_time": incident_data.get("start_time"),
                        "duration": time.time()
                        - incident_data.get("start_time", time.time()),
                    }
                )

        return active_list

    async def get_active_quarantines(self) -> Dict[str, QuarantineResult]:
        """Get all active quarantine operations."""
        if not self.response_manager:
            return {}

        return self.response_manager.get_active_quarantines()

    async def update_legacy_system(self, system: LegacySystem) -> None:
        """Update legacy system information."""
        self.legacy_systems[system.system_id] = system
        self.logger.info(f"Updated legacy system: {system.system_name}")

    async def remove_legacy_system(self, system_id: str) -> bool:
        """Remove legacy system from tracking."""
        if system_id in self.legacy_systems:
            del self.legacy_systems[system_id]
            self.logger.info(f"Removed legacy system: {system_id}")
            return True
        return False

    async def update_threat_intelligence(
        self, intelligence: ThreatIntelligence
    ) -> None:
        """Update threat intelligence information."""
        if self.threat_analyzer:
            await self.threat_analyzer.update_threat_intelligence(intelligence)

    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""

        metrics = {
            "service_status": {
                "initialized": self._initialized,
                "running": self._running,
                "components": {
                    "decision_engine": self.decision_engine is not None,
                    "response_manager": self.response_manager is not None,
                    "threat_analyzer": self.threat_analyzer is not None,
                },
            },
            "incidents": {
                "active_count": len(self.active_incidents),
                "total_processed": len(
                    [
                        i
                        for i in self.active_incidents.values()
                        if i.get("status") == "completed"
                    ]
                ),
            },
            "systems": {
                "legacy_systems_tracked": len(self.legacy_systems),
                "active_quarantines": len(await self.get_active_quarantines()),
            },
        }

        # Get component-specific metrics
        if self.decision_engine:
            try:
                decision_metrics = await self.decision_engine.get_decision_metrics()
                metrics["decision_engine"] = decision_metrics
            except Exception as e:
                metrics["decision_engine"] = {"error": str(e)}

        if self.response_manager:
            try:
                response_metrics = await self.response_manager.get_response_metrics()
                metrics["response_manager"] = response_metrics
            except Exception as e:
                metrics["response_manager"] = {"error": str(e)}

        if self.threat_analyzer:
            try:
                analyzer_metrics = await self.threat_analyzer.get_analyzer_metrics()
                metrics["threat_analyzer"] = analyzer_metrics
            except Exception as e:
                metrics["threat_analyzer"] = {"error": str(e)}

        return metrics

    async def get_security_context(self) -> SecurityContext:
        """Get current security context."""
        return self.security_context

    async def update_security_context(self, **kwargs) -> None:
        """Update security context parameters."""
        for key, value in kwargs.items():
            if hasattr(self.security_context, key):
                setattr(self.security_context, key, value)

        self.security_context.updated_at = datetime.now()
        self.logger.info(f"Security context updated: {list(kwargs.keys())}")

    @asynccontextmanager
    async def managed_processing(self, security_event: SecurityEvent):
        """Context manager for managed security event processing."""
        incident_id = None
        try:
            result = await self.process_security_event(security_event)
            incident_id = result.get("incident_id")
            yield result
        except Exception as e:
            self.logger.error(f"Managed processing failed: {e}")
            raise
        finally:
            # Cleanup if needed
            if incident_id and self.auto_cleanup_enabled:
                await self._schedule_incident_cleanup(incident_id)

    # Private methods

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""

        if self.auto_cleanup_enabled:
            cleanup_task = asyncio.create_task(self._periodic_cleanup())
            self._background_tasks.append(cleanup_task)

        metrics_task = asyncio.create_task(self._periodic_metrics_update())
        self._background_tasks.append(metrics_task)

        context_task = asyncio.create_task(self._periodic_context_update())
        self._background_tasks.append(context_task)

        self.logger.info(f"Started {len(self._background_tasks)} background tasks")

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up old incidents."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_incidents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")

    async def _cleanup_old_incidents(self) -> None:
        """Clean up old completed incidents."""
        cutoff_time = time.time() - (self.incident_retention_hours * 3600)
        incidents_to_remove = []

        for incident_id, incident_data in self.active_incidents.items():
            if (
                incident_data.get("status") in ["completed", "failed"]
                and incident_data.get("start_time", 0) < cutoff_time
            ):
                incidents_to_remove.append(incident_id)

        for incident_id in incidents_to_remove:
            del self.active_incidents[incident_id]

        if incidents_to_remove:
            ACTIVE_INCIDENTS_GAUGE.set(len(self.active_incidents))
            self.logger.info(f"Cleaned up {len(incidents_to_remove)} old incidents")

    async def _schedule_incident_cleanup(
        self, incident_id: str, delay_hours: int = 1
    ) -> None:
        """Schedule cleanup of a specific incident."""

        async def delayed_cleanup():
            await asyncio.sleep(delay_hours * 3600)
            if incident_id in self.active_incidents:
                incident_data = self.active_incidents[incident_id]
                if incident_data.get("status") in ["completed", "failed"]:
                    del self.active_incidents[incident_id]
                    ACTIVE_INCIDENTS_GAUGE.set(len(self.active_incidents))
                    self.logger.info(f"Cleaned up incident {incident_id}")

        asyncio.create_task(delayed_cleanup())

    async def _periodic_metrics_update(self) -> None:
        """Periodically update service metrics."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._update_service_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics update task error: {e}")

    async def _update_service_metrics(self) -> None:
        """Update Prometheus metrics."""
        ACTIVE_INCIDENTS_GAUGE.set(len(self.active_incidents))

        if self.response_manager:
            active_quarantines = await self.get_active_quarantines()
            SYSTEM_QUARANTINES_GAUGE.set(len(active_quarantines))

    async def _periodic_context_update(self) -> None:
        """Periodically update security context."""
        while self._running:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                await self._update_security_context_automatically()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Context update task error: {e}")

    async def _update_security_context_automatically(self) -> None:
        """Automatically update security context based on current state."""

        # Update active incidents list
        active_incident_ids = [
            iid
            for iid, data in self.active_incidents.items()
            if data.get("status") in ["processing", "pending_approval"]
        ]
        self.security_context.active_incidents = active_incident_ids

        # Update current threat level based on recent incidents
        recent_threat_levels = []
        cutoff_time = time.time() - 3600  # Last hour

        for incident_data in self.active_incidents.values():
            if incident_data.get("start_time", 0) > cutoff_time:
                event = incident_data.get("event")
                if event and hasattr(event, "threat_level"):
                    recent_threat_levels.append(event.threat_level.value)

        if recent_threat_levels:
            # Set context threat level to highest recent level
            from .models import ThreatLevel

            threat_levels = {
                "critical": ThreatLevel.CRITICAL,
                "high": ThreatLevel.HIGH,
                "medium": ThreatLevel.MEDIUM,
                "low": ThreatLevel.LOW,
                "info": ThreatLevel.INFO,
            }

            priority_order = ["critical", "high", "medium", "low", "info"]
            for level in priority_order:
                if level in recent_threat_levels:
                    self.security_context.current_threat_level = threat_levels[level]
                    break

        self.security_context.updated_at = datetime.now()

    async def _update_security_context(self, security_event: SecurityEvent) -> None:
        """Update security context based on new security event."""

        # Add to recent attacks
        attack_info = {
            "event_id": security_event.event_id,
            "event_type": security_event.event_type.value,
            "threat_level": security_event.threat_level.value,
            "timestamp": security_event.event_timestamp.isoformat(),
            "source_ip": security_event.source_ip,
        }

        self.security_context.recent_attacks.append(attack_info)

        # Keep only recent attacks (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.security_context.recent_attacks = [
            attack
            for attack in self.security_context.recent_attacks
            if datetime.fromisoformat(attack["timestamp"]) > cutoff_time
        ]

        # Update threat indicators
        if security_event.indicators_of_compromise:
            self.security_context.threat_indicators.extend(
                security_event.indicators_of_compromise
            )
            # Keep unique indicators only
            self.security_context.threat_indicators = list(
                set(self.security_context.threat_indicators)
            )

        self.security_context.updated_at = datetime.now()

    async def _get_legacy_systems_context(
        self, security_event: SecurityEvent
    ) -> List[LegacySystem]:
        """Get legacy systems context for the security event."""

        affected_systems = []

        for system_id in security_event.affected_systems:
            if system_id in self.legacy_systems:
                affected_systems.append(self.legacy_systems[system_id])

        return affected_systems

    def _generate_incident_recommendations(
        self,
        threat_analysis: ThreatAnalysis,
        automated_response: AutomatedResponse,
        response_result: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations based on incident analysis and response."""

        recommendations = []

        # Based on threat analysis
        if threat_analysis.confidence_score < 0.7:
            recommendations.append(
                "Consider additional validation due to low confidence score"
            )

        if threat_analysis.threat_level.value in ["critical", "high"]:
            recommendations.append("Conduct immediate forensic analysis")
            recommendations.append("Review and update incident response procedures")

        # Based on response execution
        if response_result and response_result.get("status") == "failed":
            recommendations.append(
                "Review response execution failures and update procedures"
            )

        if automated_response.requires_human_approval:
            recommendations.append(
                "Consider automating similar low-risk responses in the future"
            )

        # General recommendations
        recommendations.extend(
            [
                "Update threat intelligence based on incident findings",
                "Conduct lessons learned session",
                "Review security monitoring coverage",
            ]
        )

        return recommendations[:5]  # Top 5 recommendations

    async def shutdown(self) -> None:
        """Shutdown the security orchestrator service."""
        self.logger.info("Shutting down Security Orchestrator Service...")

        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Shutdown components
        if self.decision_engine:
            await self.decision_engine.shutdown()

        if self.response_manager:
            await self.response_manager.shutdown()

        if self.threat_analyzer:
            await self.threat_analyzer.shutdown()

        # Final cleanup
        self.active_incidents.clear()
        ACTIVE_INCIDENTS_GAUGE.set(0)

        self._initialized = False

        self.logger.info("Security Orchestrator Service shutdown complete")

    def __repr__(self) -> str:
        return f"SecurityOrchestratorService(initialized={self._initialized}, running={self._running})"


# Global service instance management
_security_service: Optional[SecurityOrchestratorService] = None


def get_security_service() -> SecurityOrchestratorService:
    """Get global security service instance."""
    global _security_service
    if _security_service is None:
        from ..core.config import get_config

        _security_service = SecurityOrchestratorService(get_config())
    return _security_service


async def initialize_security_service(
    config: Optional[Config] = None,
) -> SecurityOrchestratorService:
    """Initialize global security service."""
    global _security_service
    if config:
        _security_service = SecurityOrchestratorService(config)
    else:
        _security_service = get_security_service()

    await _security_service.initialize()
    return _security_service
