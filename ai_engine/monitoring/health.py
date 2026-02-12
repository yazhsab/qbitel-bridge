"""
QBITEL Engine - Health Monitoring

This module provides comprehensive health checking and system monitoring
for all AI Engine components and dependencies.
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import logging

import torch
import requests

from ..core.config import Config
from ..core.exceptions import HealthCheckException


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    last_check_time: float = field(default_factory=time.time)
    check_duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY

    def is_degraded(self) -> bool:
        """Check if component is degraded."""
        return self.status == HealthStatus.DEGRADED

    def is_unhealthy(self) -> bool:
        """Check if component is unhealthy."""
        return self.status == HealthStatus.UNHEALTHY


@dataclass
class SystemHealth:
    """Overall system health information."""

    overall_status: HealthStatus
    component_health: Dict[str, ComponentHealth] = field(default_factory=dict)
    system_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0

    def get_healthy_components(self) -> List[ComponentHealth]:
        """Get list of healthy components."""
        return [comp for comp in self.component_health.values() if comp.is_healthy()]

    def get_unhealthy_components(self) -> List[ComponentHealth]:
        """Get list of unhealthy components."""
        return [comp for comp in self.component_health.values() if comp.is_unhealthy()]

    def get_degraded_components(self) -> List[ComponentHealth]:
        """Get list of degraded components."""
        return [comp for comp in self.component_health.values() if comp.is_degraded()]

    def calculate_overall_status(self) -> HealthStatus:
        """Calculate overall system health status."""
        if not self.component_health:
            return HealthStatus.UNKNOWN

        unhealthy_count = len(self.get_unhealthy_components())
        degraded_count = len(self.get_degraded_components())
        total_count = len(self.component_health)

        # If more than 50% are unhealthy, system is unhealthy
        if unhealthy_count > total_count * 0.5:
            return HealthStatus.UNHEALTHY

        # If any critical components are unhealthy, system is unhealthy
        critical_components = ["ai_engine", "model_registry", "database"]
        for comp_name in critical_components:
            if comp_name in self.component_health:
                if self.component_health[comp_name].is_unhealthy():
                    return HealthStatus.UNHEALTHY

        # If there are unhealthy or degraded components, system is degraded
        if unhealthy_count > 0 or degraded_count > 0:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY


class HealthChecker:
    """
    Comprehensive health checker for AI Engine components.

    This class provides health monitoring capabilities for all system
    components including AI models, databases, external services, and
    system resources.
    """

    def __init__(self, config: Config):
        """Initialize health checker."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Health check configuration
        self.check_interval = getattr(config, "health_check_interval", 30)  # seconds
        self.timeout = getattr(config, "health_check_timeout", 10)  # seconds
        self.startup_time = time.time()

        # Component health checkers
        self.health_checkers: Dict[str, Callable[[], Awaitable[ComponentHealth]]] = {}

        # Health history for trending
        self.health_history: List[SystemHealth] = []
        self.max_history_size = 100

        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = asyncio.Event()
        self._current_health: Optional[SystemHealth] = None
        self._health_lock = threading.RLock()

        # Register default health checkers
        self._register_default_checkers()

        self.logger.info("HealthChecker initialized")

    def register_health_checker(
        self,
        component_name: str,
        checker_func: Callable[[], Awaitable[ComponentHealth]],
    ) -> None:
        """Register a health checker for a component."""
        self.health_checkers[component_name] = checker_func
        self.logger.info(f"Registered health checker for: {component_name}")

    async def check_all_components(self) -> SystemHealth:
        """Check health of all registered components."""
        start_time = time.time()
        component_health = {}

        # Run all health checks concurrently
        tasks = {}
        for name, checker in self.health_checkers.items():
            tasks[name] = asyncio.create_task(self._run_health_check(name, checker))

        # Wait for all checks to complete
        for name, task in tasks.items():
            try:
                component_health[name] = await asyncio.wait_for(task, timeout=self.timeout)
            except asyncio.TimeoutError:
                component_health[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check timed out after {self.timeout}s",
                )
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
                component_health[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {type(e).__name__}",
                )

        # Collect system metrics
        system_metrics = await self._collect_system_metrics()

        # Create system health
        system_health = SystemHealth(
            overall_status=HealthStatus.UNKNOWN,  # Will be calculated
            component_health=component_health,
            system_metrics=system_metrics,
            timestamp=time.time(),
            uptime_seconds=time.time() - self.startup_time,
        )

        # Calculate overall status
        system_health.overall_status = system_health.calculate_overall_status()

        # Update current health
        with self._health_lock:
            self._current_health = system_health
            self.health_history.append(system_health)

            # Limit history size
            if len(self.health_history) > self.max_history_size:
                self.health_history = self.health_history[-self.max_history_size :]

        check_duration = (time.time() - start_time) * 1000
        self.logger.info(f"Health check completed in {check_duration:.2f}ms - Status: {system_health.overall_status}")

        return system_health

    async def check_component(self, component_name: str) -> ComponentHealth:
        """Check health of a specific component."""
        if component_name not in self.health_checkers:
            return ComponentHealth(
                name=component_name,
                status=HealthStatus.UNKNOWN,
                message="No health checker registered",
            )

        checker = self.health_checkers[component_name]
        return await self._run_health_check(component_name, checker)

    def get_current_health(self) -> Optional[SystemHealth]:
        """Get current system health status."""
        with self._health_lock:
            return self._current_health

    def get_health_history(self) -> List[SystemHealth]:
        """Get health check history."""
        with self._health_lock:
            return self.health_history.copy()

    def get_health_trends(self) -> Dict[str, Any]:
        """Get health trends and statistics."""
        with self._health_lock:
            if not self.health_history:
                return {}

            trends = {
                "total_checks": len(self.health_history),
                "time_range": {
                    "start": self.health_history[0].timestamp,
                    "end": self.health_history[-1].timestamp,
                },
                "status_distribution": {
                    "healthy": 0,
                    "degraded": 0,
                    "unhealthy": 0,
                    "unknown": 0,
                },
                "component_availability": {},
                "average_check_duration": {},
            }

            # Calculate status distribution
            for health in self.health_history:
                status_key = health.overall_status.value
                trends["status_distribution"][status_key] += 1

            # Calculate component availability
            for health in self.health_history:
                for comp_name, comp_health in health.component_health.items():
                    if comp_name not in trends["component_availability"]:
                        trends["component_availability"][comp_name] = {
                            "healthy": 0,
                            "degraded": 0,
                            "unhealthy": 0,
                            "unknown": 0,
                        }

                    status_key = comp_health.status.value
                    trends["component_availability"][comp_name][status_key] += 1

            # Calculate availability percentages
            total_checks = len(self.health_history)
            for comp_name in trends["component_availability"]:
                comp_stats = trends["component_availability"][comp_name]
                for status in comp_stats:
                    comp_stats[f"{status}_percentage"] = (comp_stats[status] / total_checks) * 100

            return trends

    async def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._monitoring_task:
            self.logger.warning("Health monitoring already started")
            return

        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started background health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        if not self._monitoring_task:
            return

        self._stop_monitoring.set()

        try:
            await asyncio.wait_for(self._monitoring_task, timeout=5.0)
        except asyncio.TimeoutError:
            self._monitoring_task.cancel()

        self._monitoring_task = None
        self._stop_monitoring.clear()
        self.logger.info("Stopped background health monitoring")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                await self.check_all_components()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _run_health_check(
        self,
        component_name: str,
        checker_func: Callable[[], Awaitable[ComponentHealth]],
    ) -> ComponentHealth:
        """Run a single health check with error handling."""
        start_time = time.time()

        try:
            health = await checker_func()
            health.check_duration_ms = (time.time() - start_time) * 1000
            health.last_check_time = time.time()
            return health
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Health check failed for {component_name}: {e}")

            return ComponentHealth(
                name=component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check exception: {type(e).__name__}",
                check_duration_ms=duration_ms,
                last_check_time=time.time(),
            )

    def _register_default_checkers(self) -> None:
        """Register default health checkers."""

        async def check_system_resources() -> ComponentHealth:
            """Check system resource health."""
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)

                # Memory usage
                memory = psutil.virtual_memory()

                # Disk usage
                disk = psutil.disk_usage("/")

                details = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": (disk.used / disk.total) * 100,
                    "disk_free_gb": disk.free / (1024**3),
                }

                # Determine status
                status = HealthStatus.HEALTHY
                messages = []

                if cpu_percent > 90:
                    status = HealthStatus.UNHEALTHY
                    messages.append(f"High CPU usage: {cpu_percent:.1f}%")
                elif cpu_percent > 80:
                    status = HealthStatus.DEGRADED
                    messages.append(f"Elevated CPU usage: {cpu_percent:.1f}%")

                if memory.percent > 90:
                    status = HealthStatus.UNHEALTHY
                    messages.append(f"High memory usage: {memory.percent:.1f}%")
                elif memory.percent > 80:
                    if status == HealthStatus.HEALTHY:
                        status = HealthStatus.DEGRADED
                    messages.append(f"Elevated memory usage: {memory.percent:.1f}%")

                disk_percent = (disk.used / disk.total) * 100
                if disk_percent > 95:
                    status = HealthStatus.UNHEALTHY
                    messages.append(f"High disk usage: {disk_percent:.1f}%")
                elif disk_percent > 90:
                    if status == HealthStatus.HEALTHY:
                        status = HealthStatus.DEGRADED
                    messages.append(f"Elevated disk usage: {disk_percent:.1f}%")

                message = "; ".join(messages) if messages else "System resources normal"

                return ComponentHealth(
                    name="system_resources",
                    status=status,
                    message=message,
                    details=details,
                )

            except Exception as e:
                self.logger.error(f"Failed to check system resources: {e}")
                return ComponentHealth(
                    name="system_resources",
                    status=HealthStatus.UNHEALTHY,
                    message=f"System resource check failed: {type(e).__name__}",
                )

        async def check_gpu_availability() -> ComponentHealth:
            """Check GPU availability and health."""
            try:
                if not torch.cuda.is_available():
                    return ComponentHealth(
                        name="gpu",
                        status=HealthStatus.HEALTHY,
                        message="GPU not available (CPU-only mode)",
                        details={"cuda_available": False, "device_count": 0},
                    )

                device_count = torch.cuda.device_count()
                details = {
                    "cuda_available": True,
                    "device_count": device_count,
                    "devices": [],
                }

                status = HealthStatus.HEALTHY
                messages = []

                for i in range(device_count):
                    try:
                        device_name = torch.cuda.get_device_name(i)
                        memory_allocated = torch.cuda.memory_allocated(i)
                        memory_reserved = torch.cuda.memory_reserved(i)

                        # Get total memory (approximate)
                        torch.cuda.empty_cache()
                        total_memory = torch.cuda.get_device_properties(i).total_memory

                        device_info = {
                            "device_id": i,
                            "name": device_name,
                            "memory_allocated_mb": memory_allocated / (1024**2),
                            "memory_reserved_mb": memory_reserved / (1024**2),
                            "memory_total_mb": total_memory / (1024**2),
                            "memory_utilization": (memory_reserved / total_memory) * 100,
                        }

                        details["devices"].append(device_info)

                        # Check memory utilization
                        memory_util = (memory_reserved / total_memory) * 100
                        if memory_util > 90:
                            status = HealthStatus.DEGRADED
                            messages.append(f"GPU {i} high memory usage: {memory_util:.1f}%")

                    except Exception as e:
                        self.logger.warning(f"Error checking GPU {i}: {e}")
                        messages.append(f"Error checking GPU {i}: {type(e).__name__}")
                        if status == HealthStatus.HEALTHY:
                            status = HealthStatus.DEGRADED

                message = "; ".join(messages) if messages else f"{device_count} GPU(s) available"

                return ComponentHealth(name="gpu", status=status, message=message, details=details)

            except Exception as e:
                self.logger.error(f"Failed to check GPU: {e}")
                return ComponentHealth(
                    name="gpu",
                    status=HealthStatus.UNHEALTHY,
                    message=f"GPU check failed: {type(e).__name__}",
                )

        async def check_ai_engine() -> ComponentHealth:
            """Check AI Engine health."""
            try:
                from ..core.engine import QbitelAIEngine
                from ..api.rest import _ai_engine

                if _ai_engine is None:
                    return ComponentHealth(
                        name="ai_engine",
                        status=HealthStatus.DEGRADED,
                        message="AI Engine not yet initialized",
                        details={"initialized": False},
                    )
                status_info = await _ai_engine.get_model_info() if hasattr(_ai_engine, "get_model_info") else {}
                return ComponentHealth(
                    name="ai_engine",
                    status=HealthStatus.HEALTHY,
                    message="AI Engine operational",
                    details={
                        "initialized": True,
                        "models_loaded": len(status_info.get("loaded_models", [])),
                    },
                )
            except ImportError:
                return ComponentHealth(
                    name="ai_engine",
                    status=HealthStatus.UNKNOWN,
                    message="AI Engine module not available",
                    details={"initialized": False},
                )
            except Exception as e:
                self.logger.warning(f"AI Engine health check failed: {e}")
                return ComponentHealth(
                    name="ai_engine",
                    status=HealthStatus.DEGRADED,
                    message=f"AI Engine check failed: {type(e).__name__}",
                    details={"initialized": False},
                )

        async def check_model_registry() -> ComponentHealth:
            """Check model registry health."""
            try:
                from ..models.model_manager import get_model_manager

                manager = get_model_manager()
                if manager is None:
                    return ComponentHealth(
                        name="model_registry",
                        status=HealthStatus.DEGRADED,
                        message="Model manager not initialized",
                        details={"models_loaded": False},
                    )
                is_ready = hasattr(manager, "is_ready") and manager.is_ready()
                return ComponentHealth(
                    name="model_registry",
                    status=HealthStatus.HEALTHY if is_ready else HealthStatus.DEGRADED,
                    message="Model registry operational" if is_ready else "Models not yet loaded",
                    details={"models_loaded": is_ready},
                )
            except ImportError:
                return ComponentHealth(
                    name="model_registry",
                    status=HealthStatus.UNKNOWN,
                    message="Model manager module not available",
                    details={"models_loaded": False},
                )
            except Exception as e:
                self.logger.warning(f"Model registry health check failed: {e}")
                return ComponentHealth(
                    name="model_registry",
                    status=HealthStatus.DEGRADED,
                    message=f"Model registry check failed: {type(e).__name__}",
                    details={"models_loaded": False},
                )

        async def check_external_dependencies() -> ComponentHealth:
            """Check external service dependencies."""
            try:
                import os

                issues = []
                details = {}

                # Check database connectivity
                db_url = os.getenv("DATABASE_URL", "")
                details["database_configured"] = bool(db_url)
                if not db_url:
                    issues.append("DATABASE_URL not configured")

                # Check Redis connectivity
                redis_url = os.getenv("REDIS_URL", os.getenv("QBITEL_REDIS_URL", ""))
                details["redis_configured"] = bool(redis_url)

                # Check LLM endpoint
                llm_endpoint = os.getenv("LLM_ENDPOINT", "")
                details["llm_configured"] = bool(llm_endpoint)

                if issues:
                    return ComponentHealth(
                        name="external_dependencies",
                        status=HealthStatus.DEGRADED,
                        message="; ".join(issues),
                        details=details,
                    )

                return ComponentHealth(
                    name="external_dependencies",
                    status=HealthStatus.HEALTHY,
                    message="External dependencies configured",
                    details=details,
                )
            except Exception as e:
                self.logger.warning(f"External dependencies check failed: {e}")
                return ComponentHealth(
                    name="external_dependencies",
                    status=HealthStatus.DEGRADED,
                    message=f"Dependency check failed: {type(e).__name__}",
                    details={},
                )

        # Register all default checkers
        self.register_health_checker("system_resources", check_system_resources)
        self.register_health_checker("gpu", check_gpu_availability)
        self.register_health_checker("ai_engine", check_ai_engine)
        self.register_health_checker("model_registry", check_model_registry)
        self.register_health_checker("external_dependencies", check_external_dependencies)

    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics."""
        try:
            metrics = {}

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            metrics["cpu_percent"] = cpu_percent
            metrics["cpu_count"] = float(cpu_count)

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics["memory_total_gb"] = memory.total / (1024**3)
            metrics["memory_available_gb"] = memory.available / (1024**3)
            metrics["memory_used_gb"] = memory.used / (1024**3)
            metrics["memory_percent"] = memory.percent

            # Disk metrics
            disk = psutil.disk_usage("/")
            metrics["disk_total_gb"] = disk.total / (1024**3)
            metrics["disk_used_gb"] = disk.used / (1024**3)
            metrics["disk_free_gb"] = disk.free / (1024**3)
            metrics["disk_percent"] = (disk.used / disk.total) * 100

            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                metrics["network_bytes_sent"] = float(network.bytes_sent)
                metrics["network_bytes_recv"] = float(network.bytes_recv)
            except Exception:
                pass

            # Process metrics
            process = psutil.Process()
            metrics["process_cpu_percent"] = process.cpu_percent()
            metrics["process_memory_mb"] = process.memory_info().rss / (1024**2)
            metrics["process_open_files"] = float(len(process.open_files()))

            # GPU metrics (if available)
            try:
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    metrics["gpu_device_count"] = float(device_count)

                    total_memory = 0
                    total_allocated = 0

                    for i in range(device_count):
                        props = torch.cuda.get_device_properties(i)
                        total_memory += props.total_memory
                        total_allocated += torch.cuda.memory_allocated(i)

                    metrics["gpu_total_memory_gb"] = total_memory / (1024**3)
                    metrics["gpu_allocated_memory_gb"] = total_allocated / (1024**3)
                    metrics["gpu_memory_utilization"] = (total_allocated / total_memory) * 100 if total_memory > 0 else 0
            except Exception:
                pass

            return metrics

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {}
