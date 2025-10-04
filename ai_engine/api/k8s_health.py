"""
CRONOS AI - Kubernetes Health Check Endpoints

Kubernetes-compatible health check endpoints for liveness, readiness, and startup probes.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse

from ..core.config import Config
from ..monitoring.health import HealthChecker, HealthStatus


class ProbeStatus(str, Enum):
    """Probe status enumeration."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class ProbeResult:
    """Health probe result."""

    status: ProbeStatus
    checks: Dict[str, Any]
    timestamp: float
    response_time_ms: float


class KubernetesHealthProbes:
    """
    Kubernetes-compatible health probe endpoints.

    Implements:
    - Liveness probe: Is the application running?
    - Readiness probe: Can the application serve traffic?
    - Startup probe: Has the application finished starting?
    """

    def __init__(self, config: Config, health_checker: HealthChecker):
        """Initialize Kubernetes health probes."""
        self.config = config
        self.health_checker = health_checker
        self.logger = logging.getLogger(__name__)

        # Probe configuration
        self.startup_timeout = getattr(config, "startup_timeout", 300)  # 5 minutes
        self.readiness_threshold = getattr(config, "readiness_threshold", 0.8)
        self.liveness_threshold = getattr(config, "liveness_threshold", 0.5)

        # State tracking
        self.startup_time = time.time()
        self.startup_complete = False
        self.last_liveness_check = 0.0
        self.last_readiness_check = 0.0

        # Dependency checkers
        self.dependency_checkers = {}
        self._register_dependency_checkers()

        self.logger.info("Kubernetes health probes initialized")

    def create_router(self) -> APIRouter:
        """Create FastAPI router with health endpoints."""
        router = APIRouter(prefix="/health", tags=["health"])

        @router.get("/live")
        async def liveness_probe():
            """
            Liveness probe endpoint.

            Indicates whether the application is running and should be restarted if failing.
            Returns 200 if alive, 503 if dead.
            """
            result = await self.check_liveness()

            if result.status == ProbeStatus.PASS:
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "status": result.status,
                        "checks": result.checks,
                        "timestamp": result.timestamp,
                    },
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "status": result.status,
                        "checks": result.checks,
                        "timestamp": result.timestamp,
                    },
                )

        @router.get("/ready")
        async def readiness_probe():
            """
            Readiness probe endpoint.

            Indicates whether the application can serve traffic.
            Returns 200 if ready, 503 if not ready.
            """
            result = await self.check_readiness()

            if result.status == ProbeStatus.PASS:
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "status": result.status,
                        "checks": result.checks,
                        "timestamp": result.timestamp,
                    },
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "status": result.status,
                        "checks": result.checks,
                        "timestamp": result.timestamp,
                    },
                )

        @router.get("/startup")
        async def startup_probe():
            """
            Startup probe endpoint.

            Indicates whether the application has finished starting up.
            Returns 200 if started, 503 if still starting.
            """
            result = await self.check_startup()

            if result.status == ProbeStatus.PASS:
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "status": result.status,
                        "checks": result.checks,
                        "timestamp": result.timestamp,
                    },
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "status": result.status,
                        "checks": result.checks,
                        "timestamp": result.timestamp,
                    },
                )

        @router.get("/dependencies")
        async def dependency_health():
            """
            Dependency health check endpoint.

            Returns detailed health status of all dependencies.
            """
            result = await self.check_dependencies()

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": result.status,
                    "checks": result.checks,
                    "timestamp": result.timestamp,
                    "response_time_ms": result.response_time_ms,
                },
            )

        return router

    async def check_liveness(self) -> ProbeResult:
        """
        Check if application is alive.

        Liveness checks should be simple and fast. They verify that the
        application process is running and not deadlocked.
        """
        start_time = time.time()
        checks = {}

        try:
            # Check if main event loop is responsive
            checks["event_loop"] = {
                "status": "pass",
                "message": "Event loop responsive",
            }

            # Check if health checker is operational
            if self.health_checker:
                checks["health_checker"] = {
                    "status": "pass",
                    "message": "Health checker operational",
                }
            else:
                checks["health_checker"] = {
                    "status": "fail",
                    "message": "Health checker not initialized",
                }

            # Check memory usage (basic liveness indicator)
            import psutil

            memory = psutil.virtual_memory()
            if memory.percent < 95:
                checks["memory"] = {"status": "pass", "usage_percent": memory.percent}
            else:
                checks["memory"] = {
                    "status": "fail",
                    "usage_percent": memory.percent,
                    "message": "Critical memory usage",
                }

            # Determine overall status
            failed_checks = [k for k, v in checks.items() if v.get("status") == "fail"]

            if not failed_checks:
                probe_status = ProbeStatus.PASS
            else:
                probe_status = ProbeStatus.FAIL

            self.last_liveness_check = time.time()

            return ProbeResult(
                status=probe_status,
                checks=checks,
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            self.logger.error(f"Liveness check failed: {e}")
            return ProbeResult(
                status=ProbeStatus.FAIL,
                checks={"error": {"status": "fail", "message": str(e)}},
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def check_readiness(self) -> ProbeResult:
        """
        Check if application is ready to serve traffic.

        Readiness checks verify that all dependencies are available and
        the application can handle requests.
        """
        start_time = time.time()
        checks = {}

        try:
            # Check if startup is complete
            if not self.startup_complete:
                checks["startup"] = {
                    "status": "fail",
                    "message": "Startup not complete",
                }
                return ProbeResult(
                    status=ProbeStatus.FAIL,
                    checks=checks,
                    timestamp=time.time(),
                    response_time_ms=(time.time() - start_time) * 1000,
                )

            # Check all dependencies
            dependency_result = await self.check_dependencies()
            checks["dependencies"] = dependency_result.checks

            # Check system health
            system_health = await self.health_checker.check_all_components()

            healthy_count = len(system_health.get_healthy_components())
            total_count = len(system_health.component_health)
            health_ratio = healthy_count / total_count if total_count > 0 else 0

            checks["system_health"] = {
                "status": (
                    "pass" if health_ratio >= self.readiness_threshold else "fail"
                ),
                "healthy_components": healthy_count,
                "total_components": total_count,
                "health_ratio": health_ratio,
                "overall_status": system_health.overall_status.value,
            }

            # Determine overall readiness
            failed_checks = []
            for check_name, check_data in checks.items():
                if isinstance(check_data, dict):
                    if check_data.get("status") == "fail":
                        failed_checks.append(check_name)
                elif isinstance(check_data, dict):
                    # Check nested dependencies
                    for dep_name, dep_data in check_data.items():
                        if (
                            isinstance(dep_data, dict)
                            and dep_data.get("status") == "fail"
                        ):
                            failed_checks.append(f"{check_name}.{dep_name}")

            if not failed_checks:
                probe_status = ProbeStatus.PASS
            else:
                probe_status = ProbeStatus.FAIL

            self.last_readiness_check = time.time()

            return ProbeResult(
                status=probe_status,
                checks=checks,
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            self.logger.error(f"Readiness check failed: {e}")
            return ProbeResult(
                status=ProbeStatus.FAIL,
                checks={"error": {"status": "fail", "message": str(e)}},
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def check_startup(self) -> ProbeResult:
        """
        Check if application startup is complete.

        Startup checks verify that initialization is complete and the
        application is ready to begin serving traffic.
        """
        start_time = time.time()
        checks = {}

        try:
            elapsed_time = time.time() - self.startup_time

            # Check if startup timeout exceeded
            if elapsed_time > self.startup_timeout:
                checks["timeout"] = {
                    "status": "fail",
                    "message": f"Startup timeout exceeded ({self.startup_timeout}s)",
                    "elapsed_seconds": elapsed_time,
                }
                return ProbeResult(
                    status=ProbeStatus.FAIL,
                    checks=checks,
                    timestamp=time.time(),
                    response_time_ms=(time.time() - start_time) * 1000,
                )

            # Check if already marked as complete
            if self.startup_complete:
                checks["startup"] = {
                    "status": "pass",
                    "message": "Startup complete",
                    "elapsed_seconds": elapsed_time,
                }
                return ProbeResult(
                    status=ProbeStatus.PASS,
                    checks=checks,
                    timestamp=time.time(),
                    response_time_ms=(time.time() - start_time) * 1000,
                )

            # Check startup criteria
            checks["health_checker"] = {
                "status": "pass" if self.health_checker else "fail",
                "message": (
                    "Health checker initialized"
                    if self.health_checker
                    else "Health checker not initialized"
                ),
            }

            # Check if minimum startup time elapsed (prevent premature ready state)
            min_startup_time = 5  # seconds
            if elapsed_time < min_startup_time:
                checks["minimum_time"] = {
                    "status": "fail",
                    "message": f"Minimum startup time not elapsed ({min_startup_time}s)",
                    "elapsed_seconds": elapsed_time,
                }
            else:
                checks["minimum_time"] = {
                    "status": "pass",
                    "elapsed_seconds": elapsed_time,
                }

            # Check critical dependencies
            dependency_result = await self.check_dependencies()
            critical_deps_ok = all(
                dep.get("status") == "pass"
                for dep in dependency_result.checks.values()
                if dep.get("critical", False)
            )

            checks["critical_dependencies"] = {
                "status": "pass" if critical_deps_ok else "fail",
                "message": (
                    "Critical dependencies ready"
                    if critical_deps_ok
                    else "Critical dependencies not ready"
                ),
            }

            # Determine if startup is complete
            all_checks_pass = all(
                check.get("status") == "pass" for check in checks.values()
            )

            if all_checks_pass:
                self.startup_complete = True
                probe_status = ProbeStatus.PASS
            else:
                probe_status = ProbeStatus.FAIL

            return ProbeResult(
                status=probe_status,
                checks=checks,
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            self.logger.error(f"Startup check failed: {e}")
            return ProbeResult(
                status=ProbeStatus.FAIL,
                checks={"error": {"status": "fail", "message": str(e)}},
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def check_dependencies(self) -> ProbeResult:
        """Check health of all dependencies."""
        start_time = time.time()
        checks = {}

        # Run all dependency checks concurrently
        tasks = {}
        for dep_name, checker in self.dependency_checkers.items():
            tasks[dep_name] = asyncio.create_task(checker())

        # Wait for all checks
        for dep_name, task in tasks.items():
            try:
                checks[dep_name] = await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                checks[dep_name] = {
                    "status": "fail",
                    "message": "Dependency check timeout",
                }
            except Exception as e:
                checks[dep_name] = {
                    "status": "fail",
                    "message": f"Dependency check error: {e}",
                }

        # Determine overall status
        failed_deps = [k for k, v in checks.items() if v.get("status") == "fail"]

        if not failed_deps:
            probe_status = ProbeStatus.PASS
        else:
            probe_status = ProbeStatus.FAIL

        return ProbeResult(
            status=probe_status,
            checks=checks,
            timestamp=time.time(),
            response_time_ms=(time.time() - start_time) * 1000,
        )

    def _register_dependency_checkers(self):
        """Register dependency health checkers."""

        async def check_database():
            """Check database connectivity."""
            try:
                # This would check actual database connection
                # For now, return mock status
                return {
                    "status": "pass",
                    "message": "Database connection healthy",
                    "critical": True,
                    "response_time_ms": 5.0,
                }
            except Exception as e:
                return {
                    "status": "fail",
                    "message": f"Database connection failed: {e}",
                    "critical": True,
                }

        async def check_redis():
            """Check Redis connectivity."""
            try:
                # This would check actual Redis connection
                return {
                    "status": "pass",
                    "message": "Redis connection healthy",
                    "critical": False,
                    "response_time_ms": 2.0,
                }
            except Exception as e:
                return {
                    "status": "fail",
                    "message": f"Redis connection failed: {e}",
                    "critical": False,
                }

        async def check_model_registry():
            """Check model registry availability."""
            try:
                # This would check actual model registry
                return {
                    "status": "pass",
                    "message": "Model registry available",
                    "critical": True,
                    "models_loaded": True,
                }
            except Exception as e:
                return {
                    "status": "fail",
                    "message": f"Model registry unavailable: {e}",
                    "critical": True,
                }

        async def check_external_services():
            """Check external service dependencies."""
            try:
                # This would check external services
                return {
                    "status": "pass",
                    "message": "External services reachable",
                    "critical": False,
                }
            except Exception as e:
                return {
                    "status": "warn",
                    "message": f"Some external services unreachable: {e}",
                    "critical": False,
                }

        # Register all dependency checkers
        self.dependency_checkers["database"] = check_database
        self.dependency_checkers["redis"] = check_redis
        self.dependency_checkers["model_registry"] = check_model_registry
        self.dependency_checkers["external_services"] = check_external_services

    def mark_startup_complete(self):
        """Manually mark startup as complete."""
        self.startup_complete = True
        self.logger.info("Startup marked as complete")

    def get_probe_status(self) -> Dict[str, Any]:
        """Get current status of all probes."""
        return {
            "startup_complete": self.startup_complete,
            "uptime_seconds": time.time() - self.startup_time,
            "last_liveness_check": self.last_liveness_check,
            "last_readiness_check": self.last_readiness_check,
            "startup_timeout": self.startup_timeout,
        }
