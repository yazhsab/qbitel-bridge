"""
CRONOS AI Engine - Legacy System Whisperer Service

Main service integration for the Legacy System Whisperer feature.
Orchestrates all components and provides unified API for legacy system management.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..core.config import Config
from ..core.exceptions import CronosAIException
from ..llm.unified_llm_service import UnifiedLLMService
from ..monitoring.metrics import AIEngineMetrics

from .models import (
    LegacySystemContext,
    SystemFailurePrediction,
    MaintenanceRecommendation,
    SystemBehaviorPattern,
    FormalizedKnowledge,
    SystemMetrics,
    TimeSeriesData
)
from .enhanced_detector import EnhancedLegacySystemDetector
from .knowledge_capture import TribalKnowledgeCapture
from .predictive_analytics import (
    FailurePredictor,
    PerformanceMonitor,
    MaintenanceScheduler,
    PredictionHorizon
)
from .decision_support import (
    RecommendationEngine,
    ImpactAssessor,
    ActionPlanner,
    DecisionContext,
    DecisionCategory
)


@dataclass
class ServiceHealthStatus:
    """Health status of the Legacy System Whisperer service."""
    overall_status: str
    components_status: Dict[str, str]
    last_health_check: datetime
    active_systems: int
    active_predictions: int
    knowledge_base_size: int
    performance_score: float
    error_count: int = 0
    warnings: List[str] = None


class LegacySystemWhispererService:
    """
    Main service for Legacy System Whisperer feature.
    
    Provides unified interface for all legacy system management capabilities
    including anomaly detection, predictive analytics, knowledge management,
    and intelligent decision support.
    """
    
    def __init__(self, config: Config):
        """Initialize Legacy System Whisperer service."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.llm_service: Optional[UnifiedLLMService] = None
        self.metrics: Optional[AIEngineMetrics] = None
        
        # Feature components
        self.enhanced_detector: Optional[EnhancedLegacySystemDetector] = None
        self.knowledge_capture: Optional[TribalKnowledgeCapture] = None
        self.failure_predictor: Optional[FailurePredictor] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.maintenance_scheduler: Optional[MaintenanceScheduler] = None
        self.recommendation_engine: Optional[RecommendationEngine] = None
        self.impact_assessor: Optional[ImpactAssessor] = None
        self.action_planner: Optional[ActionPlanner] = None
        
        # Service state
        self.is_initialized = False
        self.start_time: Optional[datetime] = None
        self.health_status: Optional[ServiceHealthStatus] = None
        
        # System registry
        self.registered_systems: Dict[str, LegacySystemContext] = {}
        self.active_monitoring: Dict[str, bool] = {}
        
        # Performance tracking
        self.service_metrics = {
            "requests_processed": 0,
            "predictions_generated": 0,
            "recommendations_created": 0,
            "knowledge_items_captured": 0,
            "maintenance_scheduled": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0
        }
        
        self.logger.info("LegacySystemWhispererService created")
    
    async def initialize(
        self,
        llm_service: UnifiedLLMService,
        metrics: Optional[AIEngineMetrics] = None
    ) -> None:
        """Initialize the Legacy System Whisperer service."""
        
        if self.is_initialized:
            self.logger.warning("Service already initialized")
            return
        
        try:
            self.logger.info("Initializing Legacy System Whisperer service...")
            self.start_time = datetime.now()
            
            # Store dependencies
            self.llm_service = llm_service
            self.metrics = metrics
            
            # Initialize core components
            await self._initialize_components()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Initial health check
            await self._update_health_status()
            
            self.is_initialized = True
            self.logger.info("Legacy System Whisperer service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize service: {e}")
            raise CronosAIException(f"Service initialization failed: {e}")
    
    async def _initialize_components(self) -> None:
        """Initialize all service components."""
        
        # Enhanced anomaly detector
        self.enhanced_detector = EnhancedLegacySystemDetector(self.config)
        await self.enhanced_detector.initialize_enhanced(self.llm_service, self.metrics)
        
        # Knowledge capture system
        self.knowledge_capture = TribalKnowledgeCapture(self.config, self.llm_service)
        
        # Predictive analytics components
        self.failure_predictor = FailurePredictor(self.config, self.llm_service)
        self.performance_monitor = PerformanceMonitor(self.config, self.metrics)
        self.maintenance_scheduler = MaintenanceScheduler(self.config, self.llm_service)
        
        # Decision support components
        self.recommendation_engine = RecommendationEngine(self.config, self.llm_service)
        self.impact_assessor = ImpactAssessor(self.config, self.llm_service)
        self.action_planner = ActionPlanner(self.config, self.llm_service)
        
        self.logger.info("All service components initialized")
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        
        # Start health monitoring
        asyncio.create_task(self._health_monitor_loop())
        
        # Start performance collection if metrics available
        if self.metrics:
            asyncio.create_task(self._metrics_collection_loop())
        
        self.logger.info("Background tasks started")
    
    async def register_legacy_system(
        self,
        system_context: LegacySystemContext,
        enable_monitoring: bool = True
    ) -> Dict[str, Any]:
        """
        Register a legacy system for monitoring and management.
        
        Args:
            system_context: System context information
            enable_monitoring: Whether to enable continuous monitoring
            
        Returns:
            Registration confirmation with system info
        """
        
        try:
            system_id = system_context.system_id
            
            # Validate system context
            if not system_id or not system_context.system_name:
                raise CronosAIException("Invalid system context: missing required fields")
            
            # Register system
            self.registered_systems[system_id] = system_context
            self.active_monitoring[system_id] = enable_monitoring
            
            # Add to enhanced detector
            if self.enhanced_detector:
                self.enhanced_detector.add_system_context(system_context)
            
            # Initialize monitoring if enabled
            if enable_monitoring:
                await self._setup_system_monitoring(system_context)
            
            registration_info = {
                "system_id": system_id,
                "system_name": system_context.system_name,
                "system_type": system_context.system_type.value,
                "monitoring_enabled": enable_monitoring,
                "registration_time": datetime.now().isoformat(),
                "capabilities_enabled": [
                    "anomaly_detection",
                    "failure_prediction",
                    "performance_monitoring",
                    "maintenance_scheduling",
                    "decision_support"
                ]
            }
            
            self.logger.info(f"Registered legacy system: {system_id}")
            return registration_info
            
        except Exception as e:
            self.logger.error(f"Failed to register system {system_context.system_id}: {e}")
            raise CronosAIException(f"System registration failed: {e}")
    
    async def _setup_system_monitoring(self, system_context: LegacySystemContext) -> None:
        """Setup monitoring for a registered system."""
        
        system_id = system_context.system_id
        
        # Initialize baseline metrics if available
        # In production, this would integrate with actual monitoring systems
        self.logger.info(f"Setup monitoring for system {system_id}")
    
    async def analyze_system_health(
        self,
        system_id: str,
        current_metrics: SystemMetrics,
        time_series_data: Optional[List[TimeSeriesData]] = None,
        prediction_horizon: PredictionHorizon = PredictionHorizon.MEDIUM_TERM
    ) -> Dict[str, Any]:
        """
        Perform comprehensive system health analysis.
        
        Args:
            system_id: System identifier
            current_metrics: Current system metrics
            time_series_data: Historical time series data
            prediction_horizon: Prediction time horizon
            
        Returns:
            Comprehensive health analysis including predictions and recommendations
        """
        
        start_time = time.time()
        
        try:
            if system_id not in self.registered_systems:
                raise CronosAIException(f"System {system_id} not registered")
            
            system_context = self.registered_systems[system_id]
            analysis_results = {}
            
            # Performance analysis
            if self.performance_monitor:
                performance_analysis = await self.performance_monitor.analyze_performance(
                    system_id, current_metrics
                )
                analysis_results["performance"] = performance_analysis
            
            # Anomaly detection
            if self.enhanced_detector:
                # Convert metrics to feature format
                system_data = {
                    "cpu_utilization": current_metrics.cpu_utilization,
                    "memory_utilization": current_metrics.memory_utilization,
                    "disk_utilization": current_metrics.disk_utilization,
                    "response_time_ms": current_metrics.response_time_ms,
                    "error_rate": current_metrics.error_rate,
                    "transaction_rate": current_metrics.transaction_rate
                }
                
                failure_prediction = await self.enhanced_detector.predict_system_failure(
                    system_data, system_context, prediction_horizon.value
                )
                analysis_results["failure_prediction"] = failure_prediction
            
            # Failure probability analysis
            if self.failure_predictor and time_series_data:
                failure_analysis = await self.failure_predictor.predict_failure_probability(
                    system_id, time_series_data, system_context, prediction_horizon
                )
                analysis_results["failure_analysis"] = failure_analysis
            
            # Generate recommendations based on analysis
            if self.recommendation_engine and any(analysis_results.values()):
                recommendations = await self._generate_health_recommendations(
                    system_context, analysis_results
                )
                analysis_results["recommendations"] = recommendations
            
            # Calculate overall health score
            health_score = self._calculate_overall_health_score(analysis_results)
            
            # Update service metrics
            processing_time = time.time() - start_time
            self._update_service_metrics("health_analysis", processing_time, True)
            
            return {
                "system_id": system_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "overall_health_score": health_score,
                "analysis_results": analysis_results,
                "processing_time_seconds": processing_time,
                "recommendations_count": len(analysis_results.get("recommendations", {}).get("recommended_actions", [])),
                "next_analysis_recommended": (datetime.now() + timedelta(hours=24)).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"System health analysis failed for {system_id}: {e}")
            self._update_service_metrics("health_analysis", time.time() - start_time, False)
            raise CronosAIException(f"Health analysis failed: {e}")
    
    async def _generate_health_recommendations(
        self,
        system_context: LegacySystemContext,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommendations based on health analysis."""
        
        # Create decision context
        decision_context = DecisionContext(
            decision_id=f"health_analysis_{int(time.time())}",
            decision_category=DecisionCategory.MAINTENANCE_PLANNING,
            system_context=system_context,
            current_situation=analysis_results,
            objectives=["maintain_system_health", "prevent_failures", "optimize_performance"],
            stakeholders=["operations_team", "management"],
            risk_tolerance="medium"
        )
        
        # Generate recommendations
        recommendations = await self.recommendation_engine.generate_recommendations(
            decision_context=decision_context,
            system_analysis=analysis_results
        )
        
        return recommendations
    
    def _calculate_overall_health_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)."""
        
        scores = []
        weights = []
        
        # Performance score
        if "performance" in analysis_results:
            perf_score = analysis_results["performance"].get("performance_score", 50)
            scores.append(perf_score)
            weights.append(0.4)
        
        # Failure prediction score (inverted - lower failure probability = higher health)
        if "failure_prediction" in analysis_results:
            failure_prob = analysis_results["failure_prediction"].get("failure_probability", 0.5)
            failure_score = (1.0 - failure_prob) * 100
            scores.append(failure_score)
            weights.append(0.4)
        
        # Anomaly score (inverted - lower anomaly = higher health)
        if "failure_analysis" in analysis_results:
            anomaly_score = analysis_results["failure_analysis"].get("anomaly_analysis", {}).get("anomaly_score", 0.5)
            health_score = (1.0 - min(anomaly_score, 1.0)) * 100
            scores.append(health_score)
            weights.append(0.2)
        
        if not scores:
            return 50.0  # Neutral score if no data
        
        # Calculate weighted average
        total_weight = sum(weights)
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        
        return weighted_sum / total_weight if total_weight > 0 else 50.0
    
    async def capture_expert_knowledge(
        self,
        expert_id: str,
        session_type: str,
        expert_input: str,
        system_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture expert knowledge through structured interview process.
        
        Args:
            expert_id: Expert identifier
            session_type: Type of knowledge session
            expert_input: Expert input/knowledge
            system_id: Target system (optional)
            context: Additional context
            
        Returns:
            Knowledge capture result
        """
        
        try:
            if not self.knowledge_capture:
                raise CronosAIException("Knowledge capture system not initialized")
            
            # Start knowledge session
            session_id = await self.knowledge_capture.start_expert_session(
                expert_id=expert_id,
                session_type=session_type,
                system_id=system_id,
                context=context
            )
            
            # Capture and process expert input
            capture_result = await self.knowledge_capture.capture_expert_input(
                session_id=session_id,
                expert_input=expert_input,
                context=context
            )
            
            # Finalize session
            session_result = await self.knowledge_capture.finalize_session(session_id)
            
            # Update service metrics
            self._update_service_metrics("knowledge_capture", 0, True)
            self.service_metrics["knowledge_items_captured"] += 1
            
            return {
                "session_id": session_id,
                "capture_result": capture_result,
                "session_result": session_result,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Knowledge capture failed: {e}")
            self._update_service_metrics("knowledge_capture", 0, False)
            raise CronosAIException(f"Knowledge capture failed: {e}")
    
    async def schedule_maintenance(
        self,
        maintenance_requests: List[MaintenanceRecommendation],
        resource_constraints: Optional[Dict[str, Any]] = None,
        business_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Schedule maintenance activities with optimization.
        
        Args:
            maintenance_requests: List of maintenance requests
            resource_constraints: Resource availability constraints
            business_constraints: Business scheduling constraints
            
        Returns:
            Optimized maintenance schedule
        """
        
        try:
            if not self.maintenance_scheduler:
                raise CronosAIException("Maintenance scheduler not initialized")
            
            # Optimize maintenance schedule
            schedule_result = await self.maintenance_scheduler.optimize_maintenance_schedule(
                maintenance_requests=maintenance_requests,
                resource_constraints=resource_constraints,
                business_constraints=business_constraints
            )
            
            # Add scheduled items to scheduler
            for scheduled_item in schedule_result.get("optimized_schedule", []):
                self.maintenance_scheduler.add_scheduled_maintenance(scheduled_item)
            
            # Update service metrics
            self.service_metrics["maintenance_scheduled"] += len(schedule_result.get("optimized_schedule", []))
            
            return schedule_result
            
        except Exception as e:
            self.logger.error(f"Maintenance scheduling failed: {e}")
            raise CronosAIException(f"Maintenance scheduling failed: {e}")
    
    async def create_decision_support(
        self,
        decision_category: DecisionCategory,
        system_id: str,
        current_situation: Dict[str, Any],
        objectives: List[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create decision support recommendations.
        
        Args:
            decision_category: Category of decision
            system_id: Target system
            current_situation: Current system situation
            objectives: Decision objectives
            constraints: Decision constraints
            
        Returns:
            Decision support recommendations
        """
        
        try:
            if not self.recommendation_engine:
                raise CronosAIException("Recommendation engine not initialized")
            
            if system_id not in self.registered_systems:
                raise CronosAIException(f"System {system_id} not registered")
            
            system_context = self.registered_systems[system_id]
            
            # Create decision context
            decision_context = DecisionContext(
                decision_id=f"decision_{int(time.time())}",
                decision_category=decision_category,
                system_context=system_context,
                current_situation=current_situation,
                constraints=constraints or {},
                objectives=objectives,
                stakeholders=["operations_team", "management"]
            )
            
            # Generate recommendations
            recommendations = await self.recommendation_engine.generate_recommendations(
                decision_context=decision_context
            )
            
            # Assess business impact
            if self.impact_assessor:
                impact_assessment = await self.impact_assessor.assess_business_impact(
                    decision_context=decision_context,
                    action_recommendations=recommendations.recommended_actions,
                    system_context=system_context
                )
                recommendations.business_impact = impact_assessment
            
            # Update service metrics
            self.service_metrics["recommendations_created"] += 1
            
            return {
                "decision_recommendations": recommendations,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Decision support creation failed: {e}")
            raise CronosAIException(f"Decision support failed: {e}")
    
    def get_system_dashboard(self, system_id: str) -> Dict[str, Any]:
        """Get comprehensive dashboard for a system."""
        
        if system_id not in self.registered_systems:
            raise CronosAIException(f"System {system_id} not registered")
        
        system_context = self.registered_systems[system_id]
        
        dashboard = {
            "system_info": {
                "system_id": system_id,
                "system_name": system_context.system_name,
                "system_type": system_context.system_type.value,
                "criticality": system_context.criticality.value,
                "monitoring_enabled": self.active_monitoring.get(system_id, False)
            },
            "current_status": "operational",  # Would be determined from latest analysis
            "health_score": 85.0,  # Would come from latest health analysis
            "last_analysis": "2024-01-15T10:30:00Z",  # Would be actual timestamp
            "active_alerts": 0,
            "scheduled_maintenance": [],
            "recent_recommendations": [],
            "knowledge_base_items": 0
        }
        
        # Add enhanced detector summary if available
        if self.enhanced_detector:
            health_summary = self.enhanced_detector.get_system_health_summary(system_id)
            dashboard.update({
                "pattern_count": health_summary.get("pattern_count", 0),
                "recent_predictions": health_summary.get("recent_predictions", 0),
                "health_score": health_summary.get("health_score", 85.0)
            })
        
        # Add performance monitor summary if available
        if self.performance_monitor:
            perf_summary = self.performance_monitor.get_system_performance_summary(system_id)
            dashboard["performance_summary"] = perf_summary
        
        # Add maintenance scheduler summary if available
        if self.maintenance_scheduler:
            maint_stats = self.maintenance_scheduler.get_scheduling_statistics()
            dashboard["maintenance_stats"] = maint_stats.get("by_system", {}).get(system_id, {})
        
        return dashboard
    
    def get_service_status(self) -> ServiceHealthStatus:
        """Get current service health status."""
        
        if not self.health_status:
            # Create basic status if not available
            self.health_status = ServiceHealthStatus(
                overall_status="initializing",
                components_status={},
                last_health_check=datetime.now(),
                active_systems=len(self.registered_systems),
                active_predictions=0,
                knowledge_base_size=0,
                performance_score=0.0
            )
        
        return self.health_status
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        
        return {
            "service_metrics": self.service_metrics.copy(),
            "system_registry": {
                "registered_systems": len(self.registered_systems),
                "monitored_systems": sum(1 for enabled in self.active_monitoring.values() if enabled),
                "system_types": list(set(sys.system_type.value for sys in self.registered_systems.values()))
            },
            "component_status": {
                "enhanced_detector": self.enhanced_detector is not None,
                "knowledge_capture": self.knowledge_capture is not None,
                "failure_predictor": self.failure_predictor is not None,
                "performance_monitor": self.performance_monitor is not None,
                "maintenance_scheduler": self.maintenance_scheduler is not None,
                "recommendation_engine": self.recommendation_engine is not None,
                "impact_assessor": self.impact_assessor is not None,
                "action_planner": self.action_planner is not None
            },
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "last_updated": datetime.now().isoformat()
        }
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._update_health_status()
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _update_health_status(self) -> None:
        """Update service health status."""
        
        try:
            # Check component health
            components_status = {
                "enhanced_detector": "healthy" if self.enhanced_detector else "not_initialized",
                "knowledge_capture": "healthy" if self.knowledge_capture else "not_initialized",
                "failure_predictor": "healthy" if self.failure_predictor else "not_initialized",
                "performance_monitor": "healthy" if self.performance_monitor else "not_initialized",
                "maintenance_scheduler": "healthy" if self.maintenance_scheduler else "not_initialized",
                "recommendation_engine": "healthy" if self.recommendation_engine else "not_initialized",
                "impact_assessor": "healthy" if self.impact_assessor else "not_initialized",
                "action_planner": "healthy" if self.action_planner else "not_initialized",
                "llm_service": "healthy" if self.llm_service else "not_available"
            }
            
            # Calculate overall status
            healthy_components = sum(1 for status in components_status.values() if status == "healthy")
            total_components = len(components_status)
            health_ratio = healthy_components / total_components
            
            if health_ratio >= 0.9:
                overall_status = "healthy"
            elif health_ratio >= 0.7:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"
            
            # Get knowledge base size
            kb_size = 0
            if self.knowledge_capture:
                kb_summary = self.knowledge_capture.get_knowledge_summary()
                kb_size = kb_summary.get("total_knowledge_items", 0)
            
            # Calculate performance score
            performance_score = health_ratio * 100
            
            self.health_status = ServiceHealthStatus(
                overall_status=overall_status,
                components_status=components_status,
                last_health_check=datetime.now(),
                active_systems=len(self.registered_systems),
                active_predictions=self.service_metrics.get("predictions_generated", 0),
                knowledge_base_size=kb_size,
                performance_score=performance_score,
                error_count=0,  # Would track actual errors
                warnings=[]  # Would include actual warnings
            )
            
        except Exception as e:
            self.logger.error(f"Health status update failed: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute
                
                if self.metrics:
                    # Record service-level metrics
                    self.metrics.custom.set_custom_gauge(
                        "legacy_whisperer_registered_systems",
                        len(self.registered_systems)
                    )
                    
                    self.metrics.custom.set_custom_gauge(
                        "legacy_whisperer_requests_processed",
                        self.service_metrics["requests_processed"]
                    )
                    
                    self.metrics.custom.set_custom_gauge(
                        "legacy_whisperer_predictions_generated",
                        self.service_metrics["predictions_generated"]
                    )
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    def _update_service_metrics(self, operation: str, processing_time: float, success: bool) -> None:
        """Update service performance metrics."""
        
        self.service_metrics["requests_processed"] += 1
        
        if operation == "health_analysis" and success:
            self.service_metrics["predictions_generated"] += 1
        
        # Update average response time
        current_avg = self.service_metrics["average_response_time"]
        total_requests = self.service_metrics["requests_processed"]
        
        if total_requests > 1:
            new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
            self.service_metrics["average_response_time"] = new_avg
        else:
            self.service_metrics["average_response_time"] = processing_time
        
        # Update error rate
        if not success:
            error_count = self.service_metrics.get("error_count", 0) + 1
            self.service_metrics["error_rate"] = error_count / total_requests
            self.service_metrics["error_count"] = error_count
    
    async def shutdown(self) -> None:
        """Shutdown the Legacy System Whisperer service."""
        
        try:
            self.logger.info("Shutting down Legacy System Whisperer service...")
            
            # Stop monitoring for all systems
            for system_id in self.registered_systems:
                self.active_monitoring[system_id] = False
            
            # Cleanup components
            if self.llm_service:
                # LLM service shutdown handled by parent system
                pass
            
            self.is_initialized = False
            self.logger.info("Legacy System Whisperer service shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during service shutdown: {e}")