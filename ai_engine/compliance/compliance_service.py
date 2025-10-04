"""
CRONOS AI - Compliance Service

Main compliance service that orchestrates all compliance operations
and integrates with existing infrastructure systems.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from ..core.config import Config
from ..core.exceptions import CronosAIException
from ..llm.unified_llm_service import UnifiedLLMService, get_llm_service
from ..monitoring.enterprise_metrics import get_enterprise_metrics
from .regulatory_kb import RegulatoryKnowledgeBase, ComplianceAssessment
from .assessment_engine import ComplianceAssessmentEngine, SystemStateAnalyzer
from .report_generator import AutomatedReportGenerator, ReportFormat, ReportType
from .audit_trail import AuditTrailManager, EventType, EventSeverity
from .prompt_templates import CompliancePromptManager

logger = logging.getLogger(__name__)


class ComplianceException(CronosAIException):
    """Compliance service specific exception."""

    pass


@dataclass
class ComplianceServiceConfig:
    """Configuration for compliance service."""

    enable_assessment: bool = True
    enable_reporting: bool = True
    enable_audit_trail: bool = True
    assessment_interval_hours: int = 24
    report_retention_days: int = 365
    cache_ttl_hours: int = 6
    max_concurrent_assessments: int = 3

    # Integration settings
    timescaledb_integration: bool = True
    redis_integration: bool = True
    security_integration: bool = True

    # Performance settings
    batch_size: int = 100
    timeout_seconds: int = 300
    retry_attempts: int = 3

    # Storage settings
    storage_backend: str = "timescale"  # timescale, postgres, mongodb
    cache_backend: str = "redis"  # redis, memory


class ComplianceService:
    """
    Main compliance service orchestrating all compliance operations.

    Features:
    - Automated compliance assessments
    - Multi-format report generation
    - Blockchain audit trails
    - TimescaleDB integration for time-series data
    - Redis caching for performance
    - Security system integration
    """

    def __init__(
        self,
        config: Config,
        service_config: Optional[ComplianceServiceConfig] = None,
        llm_service: Optional[UnifiedLLMService] = None,
    ):
        self.config = config
        self.service_config = service_config or ComplianceServiceConfig()
        self.llm_service = llm_service or get_llm_service()
        self.logger = logging.getLogger(__name__)
        self.metrics = get_enterprise_metrics()

        # Core components
        self.regulatory_kb: Optional[RegulatoryKnowledgeBase] = None
        self.assessment_engine: Optional[ComplianceAssessmentEngine] = None
        self.report_generator: Optional[AutomatedReportGenerator] = None
        self.audit_trail: Optional[AuditTrailManager] = None
        self.prompt_manager: Optional[CompliancePromptManager] = None

        # Integration components (will be initialized in start())
        self.timescale_client = None
        self.redis_client = None
        self.security_integration = None

        # Service state
        self._running = False
        self._assessment_task = None
        self._cleanup_task = None

        # Performance tracking
        self.assessment_cache: Dict[str, ComplianceAssessment] = {}
        self.active_assessments: Dict[str, asyncio.Task] = {}

    async def start(self):
        """Start compliance service and all components."""
        try:
            if self._running:
                self.logger.warning("Compliance service already running")
                return

            self.logger.info("Starting CRONOS AI Compliance Service...")

            # Initialize core components
            await self._initialize_components()

            # Initialize integrations
            await self._initialize_integrations()

            # Start background tasks
            await self._start_background_tasks()

            self._running = True
            self.logger.info("Compliance service started successfully")

            # Record service start event
            if self.audit_trail:
                await self.audit_trail.record_compliance_event(
                    EventType.SYSTEM_ACTION,
                    "system",
                    "compliance_service",
                    "start",
                    "success",
                    {"component": "compliance_service", "status": "started"},
                )

        except Exception as e:
            self.logger.error(f"Failed to start compliance service: {e}")
            raise ComplianceException(f"Service startup failed: {e}")

    async def stop(self):
        """Stop compliance service and cleanup resources."""
        try:
            if not self._running:
                return

            self.logger.info("Stopping compliance service...")

            # Cancel background tasks
            if self._assessment_task:
                self._assessment_task.cancel()
                try:
                    await self._assessment_task
                except asyncio.CancelledError:
                    pass

            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Cancel active assessments
            for assessment_id, task in self.active_assessments.items():
                self.logger.info(f"Cancelling active assessment: {assessment_id}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Stop components
            if self.audit_trail:
                await self.audit_trail.stop()

            # Close integrations
            await self._cleanup_integrations()

            self._running = False
            self.logger.info("Compliance service stopped")

        except Exception as e:
            self.logger.error(f"Error stopping compliance service: {e}")

    async def _initialize_components(self):
        """Initialize core compliance components."""
        try:
            # Initialize regulatory knowledge base
            self.regulatory_kb = RegulatoryKnowledgeBase(self.config, self.llm_service)

            # Initialize assessment engine
            self.assessment_engine = ComplianceAssessmentEngine(
                self.config, self.regulatory_kb, self.llm_service
            )

            # Initialize report generator
            self.report_generator = AutomatedReportGenerator(
                self.config, self.llm_service
            )

            # Initialize audit trail
            if self.service_config.enable_audit_trail:
                self.audit_trail = AuditTrailManager(self.config)
                await self.audit_trail.start()

            # Initialize prompt manager
            self.prompt_manager = CompliancePromptManager(self.config)

            self.logger.info("Core compliance components initialized")

        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise ComplianceException(f"Component initialization failed: {e}")

    async def _initialize_integrations(self):
        """Initialize integration components."""
        try:
            # TimescaleDB integration
            if self.service_config.timescaledb_integration:
                from .data_integrations import TimescaleComplianceIntegration

                self.timescale_client = TimescaleComplianceIntegration(self.config)
                await self.timescale_client.initialize()
                self.logger.info("TimescaleDB integration initialized")

            # Redis integration
            if self.service_config.redis_integration:
                from .data_integrations import RedisComplianceCache

                self.redis_client = RedisComplianceCache(self.config)
                await self.redis_client.initialize()
                self.logger.info("Redis integration initialized")

            # Security integration
            if self.service_config.security_integration:
                from .security_integration import ComplianceSecurityIntegration

                self.security_integration = ComplianceSecurityIntegration(self.config)
                await self.security_integration.initialize()
                self.logger.info("Security integration initialized")

        except Exception as e:
            self.logger.error(f"Integration initialization failed: {e}")
            # Don't fail service start if integrations fail
            self.logger.warning("Continuing without failed integrations")

    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # Periodic assessment task
        if self.service_config.assessment_interval_hours > 0:
            self._assessment_task = asyncio.create_task(
                self._periodic_assessment_loop()
            )

        # Cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self.logger.info("Background tasks started")

    async def _cleanup_integrations(self):
        """Cleanup integration resources."""
        if self.timescale_client:
            await self.timescale_client.close()

        if self.redis_client:
            await self.redis_client.close()

        if self.security_integration:
            await self.security_integration.close()

    async def assess_compliance(
        self,
        framework: str,
        target_requirements: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> ComplianceAssessment:
        """
        Perform compliance assessment for specified framework.

        Args:
            framework: Compliance framework to assess
            target_requirements: Specific requirements to assess
            force_refresh: Skip cache and perform fresh assessment

        Returns:
            Compliance assessment results
        """
        try:
            assessment_start = datetime.utcnow()
            assessment_id = f"{framework}_{int(assessment_start.timestamp())}"

            self.logger.info(f"Starting compliance assessment: {assessment_id}")

            # Check cache first unless force refresh
            if not force_refresh and self.redis_client:
                cached_result = await self.redis_client.get_assessment(framework)
                if cached_result:
                    self.logger.info(f"Returning cached assessment for {framework}")
                    return cached_result

            # Check concurrent assessment limit
            if (
                len(self.active_assessments)
                >= self.service_config.max_concurrent_assessments
            ):
                raise ComplianceException("Maximum concurrent assessments reached")

            # Start assessment
            assessment_task = asyncio.create_task(
                self._perform_assessment(framework, target_requirements, assessment_id)
            )
            self.active_assessments[assessment_id] = assessment_task

            try:
                assessment = await assessment_task

                # Cache result
                if self.redis_client:
                    await self.redis_client.cache_assessment(framework, assessment)

                # Store in TimescaleDB
                if self.timescale_client:
                    await self.timescale_client.store_assessment(assessment)

                # Record audit event
                if self.audit_trail:
                    await self.audit_trail.record_compliance_event(
                        EventType.ASSESSMENT_COMPLETED,
                        "system",
                        f"framework_{framework}",
                        "assess_compliance",
                        "success",
                        {
                            "framework": framework,
                            "assessment_id": assessment_id,
                            "compliance_score": assessment.overall_compliance_score,
                            "duration_seconds": (
                                datetime.utcnow() - assessment_start
                            ).total_seconds(),
                        },
                        compliance_framework=framework,
                    )

                # Record metrics
                self.metrics.record_protocol_discovery_metric(
                    "compliance_assessment_score",
                    assessment.overall_compliance_score,
                    {"framework": framework},
                )

                self.logger.info(
                    f"Assessment completed: {assessment_id} - {assessment.overall_compliance_score:.1f}%"
                )
                return assessment

            finally:
                # Clean up active assessment tracking
                self.active_assessments.pop(assessment_id, None)

        except Exception as e:
            self.logger.error(f"Compliance assessment failed: {e}")

            # Record audit event for failure
            if self.audit_trail:
                await self.audit_trail.record_compliance_event(
                    EventType.ASSESSMENT_COMPLETED,
                    "system",
                    f"framework_{framework}",
                    "assess_compliance",
                    "failure",
                    {"framework": framework, "error": str(e)},
                    EventSeverity.ERROR,
                    framework,
                )

            raise ComplianceException(f"Assessment failed: {e}")

    async def _perform_assessment(
        self,
        framework: str,
        target_requirements: Optional[List[str]],
        assessment_id: str,
    ) -> ComplianceAssessment:
        """Perform the actual compliance assessment."""
        # Record start event
        if self.audit_trail:
            await self.audit_trail.record_compliance_event(
                EventType.ASSESSMENT_STARTED,
                "system",
                f"framework_{framework}",
                "start_assessment",
                "success",
                {
                    "framework": framework,
                    "assessment_id": assessment_id,
                    "target_requirements": (
                        len(target_requirements) if target_requirements else "all"
                    ),
                },
                compliance_framework=framework,
            )

        # Perform assessment using assessment engine
        assessment = await self.assessment_engine.assess_compliance(
            framework=framework,
            target_requirements=target_requirements,
            use_cached_snapshot=True,
        )

        return assessment

    async def generate_report(
        self,
        framework: str,
        report_type: ReportType = ReportType.DETAILED_TECHNICAL,
        format: ReportFormat = ReportFormat.PDF,
        force_refresh: bool = False,
    ) -> bytes:
        """
        Generate compliance report.

        Args:
            framework: Framework to generate report for
            report_type: Type of report to generate
            format: Output format
            force_refresh: Force fresh assessment

        Returns:
            Generated report content
        """
        try:
            self.logger.info(f"Generating {report_type.value} report for {framework}")

            # Get or perform assessment
            assessment = await self.assess_compliance(
                framework, force_refresh=force_refresh
            )

            # Generate report
            report = await self.report_generator.generate_compliance_report(
                assessment=assessment, report_type=report_type, format=format
            )

            # Record audit event
            if self.audit_trail:
                await self.audit_trail.record_compliance_event(
                    EventType.REPORT_GENERATED,
                    "system",
                    f"report_{report.report_id}",
                    "generate_report",
                    "success",
                    {
                        "framework": framework,
                        "report_type": report_type.value,
                        "format": format.value,
                        "report_id": report.report_id,
                        "file_size": report.file_size,
                    },
                    compliance_framework=framework,
                )

            # Store report metadata in TimescaleDB
            if self.timescale_client:
                await self.timescale_client.store_report_metadata(report)

            return report.content

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise ComplianceException(f"Report generation failed: {e}")

    async def get_compliance_dashboard(
        self, frameworks: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive compliance dashboard data.

        Args:
            frameworks: List of frameworks to include (None for all)

        Returns:
            Dashboard data with compliance status and metrics
        """
        try:
            if not frameworks:
                frameworks = self.regulatory_kb.get_available_frameworks()

            dashboard_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "frameworks": {},
                "summary": {
                    "total_frameworks": len(frameworks),
                    "average_compliance": 0.0,
                    "critical_gaps": 0,
                    "assessments_today": 0,
                },
                "trends": {},
                "recommendations": [],
            }

            total_compliance = 0.0
            critical_gaps = 0

            # Get data for each framework
            for framework in frameworks:
                try:
                    # Try to get recent assessment
                    assessment = None
                    if self.redis_client:
                        assessment = await self.redis_client.get_assessment(framework)

                    if not assessment and self.timescale_client:
                        assessment = await self.timescale_client.get_latest_assessment(
                            framework
                        )

                    if assessment:
                        framework_data = {
                            "compliance_score": assessment.overall_compliance_score,
                            "risk_score": assessment.risk_score,
                            "last_assessment": assessment.assessment_date.isoformat(),
                            "total_requirements": (
                                assessment.compliant_requirements
                                + assessment.non_compliant_requirements
                                + assessment.partially_compliant_requirements
                            ),
                            "compliant": assessment.compliant_requirements,
                            "non_compliant": assessment.non_compliant_requirements,
                            "critical_gaps": len(
                                [
                                    g
                                    for g in assessment.gaps
                                    if g.severity.value == "critical"
                                ]
                            ),
                            "status": (
                                "compliant"
                                if assessment.overall_compliance_score >= 80
                                else "non_compliant"
                            ),
                        }

                        total_compliance += assessment.overall_compliance_score
                        critical_gaps += framework_data["critical_gaps"]
                    else:
                        # Framework not assessed recently
                        framework_data = {
                            "compliance_score": 0.0,
                            "risk_score": 100.0,
                            "last_assessment": None,
                            "status": "not_assessed",
                            "critical_gaps": 0,
                        }

                    dashboard_data["frameworks"][framework] = framework_data

                except Exception as e:
                    self.logger.error(
                        f"Error getting dashboard data for {framework}: {e}"
                    )
                    dashboard_data["frameworks"][framework] = {
                        "status": "error",
                        "error": str(e),
                    }

            # Calculate summary
            valid_frameworks = [
                f
                for f in dashboard_data["frameworks"].values()
                if f.get("status") not in ["error", "not_assessed"]
            ]

            if valid_frameworks:
                dashboard_data["summary"]["average_compliance"] = (
                    total_compliance / len(valid_frameworks)
                )
                dashboard_data["summary"]["critical_gaps"] = critical_gaps

            # Get trends from TimescaleDB if available
            if self.timescale_client:
                dashboard_data["trends"] = (
                    await self.timescale_client.get_compliance_trends(
                        frameworks, days=30
                    )
                )

            # Get top recommendations
            all_recommendations = []
            for framework_data in valid_frameworks:
                if "recommendations" in framework_data:
                    all_recommendations.extend(framework_data["recommendations"])

            # Sort by priority and take top 5
            dashboard_data["recommendations"] = all_recommendations[:5]

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Dashboard data generation failed: {e}")
            raise ComplianceException(f"Dashboard data generation failed: {e}")

    async def get_audit_trail_report(
        self,
        framework: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get comprehensive audit trail report."""
        try:
            if not self.audit_trail:
                raise ComplianceException("Audit trail not enabled")

            # Default to last 30 days if no dates provided
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()

            report = await self.audit_trail.generate_compliance_audit_report(
                framework or "all", start_date, end_date
            )

            return report

        except Exception as e:
            self.logger.error(f"Audit trail report generation failed: {e}")
            raise ComplianceException(f"Audit trail report failed: {e}")

    async def _periodic_assessment_loop(self):
        """Background task for periodic assessments."""
        while self._running:
            try:
                await asyncio.sleep(
                    self.service_config.assessment_interval_hours * 3600
                )

                if not self._running:
                    break

                self.logger.info("Starting periodic compliance assessments")

                # Get all frameworks
                frameworks = self.regulatory_kb.get_available_frameworks()

                # Assess each framework
                for framework in frameworks:
                    try:
                        if (
                            len(self.active_assessments)
                            < self.service_config.max_concurrent_assessments
                        ):
                            await self.assess_compliance(framework)
                        else:
                            self.logger.warning(
                                f"Skipping {framework} assessment - too many active assessments"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Periodic assessment failed for {framework}: {e}"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic assessment loop: {e}")

    async def _cleanup_loop(self):
        """Background task for cleanup operations."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run cleanup every hour

                if not self._running:
                    break

                # Clear expired cache entries
                if self.redis_client:
                    await self.redis_client.cleanup_expired()

                # Clean up old TimescaleDB data if configured
                if self.timescale_client:
                    retention_days = self.service_config.report_retention_days
                    await self.timescale_client.cleanup_old_data(retention_days)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    @asynccontextmanager
    async def compliance_session(self, session_id: str):
        """Context manager for compliance assessment sessions."""
        try:
            self.logger.info(f"Starting compliance session: {session_id}")

            if self.audit_trail:
                await self.audit_trail.record_compliance_event(
                    EventType.USER_ACTION,
                    "user",
                    f"session_{session_id}",
                    "start_session",
                    "success",
                    {"session_id": session_id},
                    session_id=session_id,
                )

            yield self

        except Exception as e:
            self.logger.error(f"Compliance session error: {e}")

            if self.audit_trail:
                await self.audit_trail.record_compliance_event(
                    EventType.USER_ACTION,
                    "user",
                    f"session_{session_id}",
                    "session_error",
                    "failure",
                    {"session_id": session_id, "error": str(e)},
                    EventSeverity.ERROR,
                    session_id=session_id,
                )
            raise

        finally:
            self.logger.info(f"Ending compliance session: {session_id}")

            if self.audit_trail:
                await self.audit_trail.record_compliance_event(
                    EventType.USER_ACTION,
                    "user",
                    f"session_{session_id}",
                    "end_session",
                    "success",
                    {"session_id": session_id},
                    session_id=session_id,
                )

    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            "service_running": self._running,
            "components": {
                "regulatory_kb": self.regulatory_kb is not None,
                "assessment_engine": self.assessment_engine is not None,
                "report_generator": self.report_generator is not None,
                "audit_trail": self.audit_trail is not None,
                "prompt_manager": self.prompt_manager is not None,
            },
            "integrations": {
                "timescaledb": self.timescale_client is not None,
                "redis": self.redis_client is not None,
                "security": self.security_integration is not None,
            },
            "active_assessments": len(self.active_assessments),
            "cached_assessments": len(self.assessment_cache),
            "configuration": {
                "assessment_interval_hours": self.service_config.assessment_interval_hours,
                "max_concurrent_assessments": self.service_config.max_concurrent_assessments,
                "cache_ttl_hours": self.service_config.cache_ttl_hours,
            },
        }


# Global service instance
_compliance_service: Optional[ComplianceService] = None


async def get_compliance_service(config: Optional[Config] = None) -> ComplianceService:
    """Get global compliance service instance."""
    global _compliance_service
    if _compliance_service is None:
        from ..core.config import get_config

        config = config or get_config()
        _compliance_service = ComplianceService(config)
        await _compliance_service.start()
    return _compliance_service


async def shutdown_compliance_service():
    """Shutdown global compliance service."""
    global _compliance_service
    if _compliance_service:
        await _compliance_service.stop()
        _compliance_service = None
