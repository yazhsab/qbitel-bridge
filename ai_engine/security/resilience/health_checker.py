"""
QBITEL Engine - Health Checker Implementation

Enterprise-grade health monitoring with configurable checks,
dependency tracking, and automated recovery for the Zero-Touch Security Orchestrator.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

from ..logging import get_security_logger, SecurityLogType, LogLevel


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(str, Enum):
    """Types of health checks."""

    LIVENESS = "liveness"  # Is component alive?
    READINESS = "readiness"  # Is component ready to serve?
    STARTUP = "startup"  # Is component starting up?
    DEPENDENCY = "dependency"  # Is dependency available?
    PERFORMANCE = "performance"  # Is performance acceptable?


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    status: HealthStatus
    message: str
    check_duration: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "message": self.message,
            "check_duration": self.check_duration,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details or {},
        }


@dataclass
class HealthCheckConfig:
    """Configuration for a health check."""

    name: str
    check_type: CheckType
    interval: float = 30.0  # seconds
    timeout: float = 5.0  # seconds
    retries: int = 3
    failure_threshold: int = 3
    success_threshold: int = 1
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self.logger = get_security_logger(
            f"qbitel.security.resilience.health_check.{config.name}"
        )

        # State tracking
        self._current_status = HealthStatus.UNKNOWN
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_check_time: Optional[datetime] = None
        self._last_result: Optional[HealthCheckResult] = None
        self._check_history: List[HealthCheckResult] = []

        # Metrics
        self._total_checks = 0
        self._successful_checks = 0
        self._failed_checks = 0
        self._average_duration = 0.0
        self._max_duration = 0.0

    @abstractmethod
    async def perform_check(self) -> HealthCheckResult:
        """Perform the actual health check. Must be implemented by subclasses."""
        pass

    async def check(self) -> HealthCheckResult:
        """Execute the health check with error handling and metrics."""

        if not self.config.enabled:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message="Health check disabled",
                check_duration=0.0,
            )

        start_time = time.time()
        self._total_checks += 1

        try:
            # Execute check with timeout
            result = await asyncio.wait_for(
                self.perform_check(), timeout=self.config.timeout
            )

            # Update state based on result
            await self._update_state(result)

            return result

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timeout after {self.config.timeout}s",
                check_duration=duration,
            )
            await self._update_state(result)
            return result

        except Exception as e:
            duration = time.time() - start_time
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                check_duration=duration,
                details={"exception": str(e), "exception_type": type(e).__name__},
            )
            await self._update_state(result)
            return result

    async def _update_state(self, result: HealthCheckResult):
        """Update health check state based on result."""

        self._last_check_time = result.timestamp
        self._last_result = result
        self._check_history.append(result)

        # Keep only recent history
        if len(self._check_history) > 100:
            self._check_history = self._check_history[-50:]

        # Update metrics
        self._average_duration = (
            self._average_duration * (self._total_checks - 1) + result.check_duration
        ) / self._total_checks
        self._max_duration = max(self._max_duration, result.check_duration)

        # Update state counters
        if result.status == HealthStatus.HEALTHY:
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            self._successful_checks += 1
        else:
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._failed_checks += 1

        # Determine overall status
        previous_status = self._current_status

        if self._consecutive_failures >= self.config.failure_threshold:
            self._current_status = HealthStatus.UNHEALTHY
        elif self._consecutive_successes >= self.config.success_threshold:
            self._current_status = HealthStatus.HEALTHY
        elif result.status == HealthStatus.DEGRADED:
            self._current_status = HealthStatus.DEGRADED

        # Log status changes
        if self._current_status != previous_status:
            self.logger.log_security_event(
                SecurityLogType.PERFORMANCE_METRIC,
                f"Health check '{self.config.name}' status changed: {previous_status.value} -> {self._current_status.value}",
                level=(
                    LogLevel.WARNING
                    if self._current_status == HealthStatus.UNHEALTHY
                    else LogLevel.INFO
                ),
                metadata={
                    "check_name": self.config.name,
                    "previous_status": previous_status.value,
                    "current_status": self._current_status.value,
                    "consecutive_failures": self._consecutive_failures,
                    "consecutive_successes": self._consecutive_successes,
                },
            )

    def get_status(self) -> HealthStatus:
        """Get current health status."""
        return self._current_status

    def get_metrics(self) -> Dict[str, Any]:
        """Get health check metrics."""
        return {
            "name": self.config.name,
            "type": self.config.check_type.value,
            "current_status": self._current_status.value,
            "enabled": self.config.enabled,
            "last_check": (
                self._last_check_time.isoformat() if self._last_check_time else None
            ),
            "last_result": self._last_result.to_dict() if self._last_result else None,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_successes": self._consecutive_successes,
            "total_checks": self._total_checks,
            "successful_checks": self._successful_checks,
            "failed_checks": self._failed_checks,
            "success_rate": (
                self._successful_checks / self._total_checks
                if self._total_checks > 0
                else 0
            ),
            "average_duration": self._average_duration,
            "max_duration": self._max_duration,
            "dependencies": self.config.dependencies,
        }


class BasicHealthCheck(HealthCheck):
    """Basic health check that executes a custom check function."""

    def __init__(self, config: HealthCheckConfig, check_function: Callable[[], Any]):
        super().__init__(config)
        self.check_function = check_function

    async def perform_check(self) -> HealthCheckResult:
        """Execute the provided check function."""

        start_time = time.time()

        try:
            # Execute check function
            if asyncio.iscoroutinefunction(self.check_function):
                result = await self.check_function()
            else:
                result = self.check_function()

            duration = time.time() - start_time

            # Interpret result
            if result is True or result == "healthy":
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="Check passed",
                    check_duration=duration,
                )
            elif result == "degraded":
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="Service degraded",
                    check_duration=duration,
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {result}",
                    check_duration=duration,
                )

        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Check exception: {str(e)}",
                check_duration=duration,
            )


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connections."""

    def __init__(self, config: HealthCheckConfig, connection_factory: Callable):
        super().__init__(config)
        self.connection_factory = connection_factory

    async def perform_check(self) -> HealthCheckResult:
        """Check database connectivity."""

        start_time = time.time()

        try:
            # Get connection and execute simple query
            connection = await self.connection_factory()

            # Execute a simple query (implementation depends on database type)
            # This is a generic example
            cursor = await connection.execute("SELECT 1")
            result = await cursor.fetchone()

            await connection.close()

            duration = time.time() - start_time

            if result:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    check_duration=duration,
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Database query failed",
                    check_duration=duration,
                )

        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Database health check failed: {str(e)}",
                check_duration=duration,
            )


class HTTPHealthCheck(HealthCheck):
    """Health check for HTTP endpoints."""

    def __init__(self, config: HealthCheckConfig, url: str, expected_status: int = 200):
        super().__init__(config)
        self.url = url
        self.expected_status = expected_status

    async def perform_check(self) -> HealthCheckResult:
        """Check HTTP endpoint health."""

        start_time = time.time()

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(self.url) as response:
                    duration = time.time() - start_time

                    if response.status == self.expected_status:
                        return HealthCheckResult(
                            status=HealthStatus.HEALTHY,
                            message=f"HTTP endpoint healthy (status: {response.status})",
                            check_duration=duration,
                            details={"status_code": response.status, "url": self.url},
                        )
                    elif 200 <= response.status < 400:
                        return HealthCheckResult(
                            status=HealthStatus.DEGRADED,
                            message=f"HTTP endpoint degraded (status: {response.status})",
                            check_duration=duration,
                            details={"status_code": response.status, "url": self.url},
                        )
                    else:
                        return HealthCheckResult(
                            status=HealthStatus.UNHEALTHY,
                            message=f"HTTP endpoint unhealthy (status: {response.status})",
                            check_duration=duration,
                            details={"status_code": response.status, "url": self.url},
                        )

        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"HTTP health check failed: {str(e)}",
                check_duration=duration,
            )


class HealthChecker:
    """
    Manages multiple health checks and provides aggregated health status
    with dependency tracking and automated monitoring.
    """

    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.logger = get_security_logger("qbitel.security.resilience.health_checker")

        # Background monitoring
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._running = False

        # Dependency graph
        self._dependency_graph: Dict[str, Set[str]] = {}

        # Global metrics
        self._start_time = datetime.utcnow()
        self._total_health_checks = 0

    def register_check(self, health_check: HealthCheck):
        """Register a health check."""

        name = health_check.config.name

        if name in self.health_checks:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Replacing existing health check: {name}",
                level=LogLevel.WARNING,
            )

        self.health_checks[name] = health_check
        self._total_health_checks += 1

        # Build dependency graph
        self._dependency_graph[name] = set(health_check.config.dependencies)

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Registered health check: {name}",
            level=LogLevel.INFO,
            metadata={
                "check_name": name,
                "check_type": health_check.config.check_type.value,
                "interval": health_check.config.interval,
                "dependencies": health_check.config.dependencies,
            },
        )

        # Start monitoring if system is running
        if self._running and health_check.config.enabled:
            self._start_monitoring(name)

    def unregister_check(self, name: str):
        """Unregister a health check."""

        if name in self.health_checks:
            # Stop monitoring
            self._stop_monitoring(name)

            # Remove from registry
            del self.health_checks[name]

            # Remove from dependency graph
            if name in self._dependency_graph:
                del self._dependency_graph[name]

            # Remove as dependency from other checks
            for deps in self._dependency_graph.values():
                deps.discard(name)

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Unregistered health check: {name}",
                level=LogLevel.INFO,
            )

    async def check_health(self, name: str) -> Optional[HealthCheckResult]:
        """Perform a single health check."""

        if name not in self.health_checks:
            return None

        health_check = self.health_checks[name]
        return await health_check.check()

    async def check_all_health(self) -> Dict[str, HealthCheckResult]:
        """Perform all health checks."""

        results = {}

        # Execute all checks concurrently
        tasks = {
            name: asyncio.create_task(health_check.check())
            for name, health_check in self.health_checks.items()
            if health_check.config.enabled
        }

        # Wait for all to complete
        completed_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Collect results
        for (name, task), result in zip(tasks.items(), completed_tasks):
            if isinstance(result, Exception):
                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(result)}",
                    check_duration=0.0,
                )
            else:
                results[name] = result

        return results

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""

        if not self.health_checks:
            return HealthStatus.UNKNOWN

        # Check critical dependencies first
        critical_unhealthy = any(
            check.get_status() == HealthStatus.UNHEALTHY
            for check in self.health_checks.values()
            if check.config.check_type in [CheckType.LIVENESS, CheckType.DEPENDENCY]
        )

        if critical_unhealthy:
            return HealthStatus.UNHEALTHY

        # Check if any component is degraded
        any_degraded = any(
            check.get_status() == HealthStatus.DEGRADED
            for check in self.health_checks.values()
        )

        if any_degraded:
            return HealthStatus.DEGRADED

        # Check if all enabled checks are healthy
        enabled_checks = [
            check for check in self.health_checks.values() if check.config.enabled
        ]

        if not enabled_checks:
            return HealthStatus.UNKNOWN

        all_healthy = all(
            check.get_status() == HealthStatus.HEALTHY for check in enabled_checks
        )

        return HealthStatus.HEALTHY if all_healthy else HealthStatus.DEGRADED

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""

        overall_status = self.get_overall_health()

        check_summaries = {}
        for name, check in self.health_checks.items():
            metrics = check.get_metrics()
            check_summaries[name] = {
                "status": metrics["current_status"],
                "type": metrics["type"],
                "enabled": metrics["enabled"],
                "last_check": metrics["last_check"],
                "success_rate": metrics["success_rate"],
                "dependencies": metrics["dependencies"],
            }

        # Calculate aggregate metrics
        total_checks = sum(c._total_checks for c in self.health_checks.values())
        successful_checks = sum(
            c._successful_checks for c in self.health_checks.values()
        )

        return {
            "overall_status": overall_status.value,
            "uptime": str(datetime.utcnow() - self._start_time),
            "total_registered_checks": len(self.health_checks),
            "enabled_checks": len(
                [c for c in self.health_checks.values() if c.config.enabled]
            ),
            "healthy_checks": len(
                [
                    c
                    for c in self.health_checks.values()
                    if c.get_status() == HealthStatus.HEALTHY
                ]
            ),
            "degraded_checks": len(
                [
                    c
                    for c in self.health_checks.values()
                    if c.get_status() == HealthStatus.DEGRADED
                ]
            ),
            "unhealthy_checks": len(
                [
                    c
                    for c in self.health_checks.values()
                    if c.get_status() == HealthStatus.UNHEALTHY
                ]
            ),
            "total_check_executions": total_checks,
            "successful_check_executions": successful_checks,
            "overall_success_rate": (
                successful_checks / total_checks if total_checks > 0 else 0
            ),
            "checks": check_summaries,
            "monitoring_active": self._running,
        }

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics for all health checks."""

        return {name: check.get_metrics() for name, check in self.health_checks.items()}

    async def start_monitoring(self):
        """Start background health monitoring for all checks."""

        if self._running:
            return

        self._running = True

        # Start monitoring tasks for all enabled checks
        for name, health_check in self.health_checks.items():
            if health_check.config.enabled:
                self._start_monitoring(name)

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Started health monitoring for {len(self._monitoring_tasks)} checks",
            level=LogLevel.INFO,
        )

    def _start_monitoring(self, name: str):
        """Start monitoring for a specific health check."""

        if name in self._monitoring_tasks:
            return

        health_check = self.health_checks[name]
        task = asyncio.create_task(self._monitor_health_check(name, health_check))
        self._monitoring_tasks[name] = task

    async def _monitor_health_check(self, name: str, health_check: HealthCheck):
        """Background monitoring loop for a health check."""

        try:
            while self._running and name in self.health_checks:
                try:
                    await health_check.check()
                    await asyncio.sleep(health_check.config.interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.log_security_event(
                        SecurityLogType.PERFORMANCE_METRIC,
                        f"Health monitoring error for '{name}': {str(e)}",
                        level=LogLevel.ERROR,
                        error_code="HEALTH_MONITORING_ERROR",
                    )
                    await asyncio.sleep(health_check.config.interval)
        finally:
            if name in self._monitoring_tasks:
                del self._monitoring_tasks[name]

    def _stop_monitoring(self, name: str):
        """Stop monitoring for a specific health check."""

        if name in self._monitoring_tasks:
            task = self._monitoring_tasks.pop(name)
            task.cancel()

    async def stop_monitoring(self):
        """Stop all health monitoring."""

        self._running = False

        # Cancel all monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()

        if self._monitoring_tasks:
            await asyncio.gather(
                *self._monitoring_tasks.values(), return_exceptions=True
            )

        self._monitoring_tasks.clear()

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Stopped all health monitoring",
            level=LogLevel.INFO,
        )

    def validate_dependencies(self) -> Dict[str, List[str]]:
        """Validate dependency graph and return any circular dependencies."""

        circular_deps = {}

        def has_circular_dependency(node: str, path: List[str]) -> bool:
            if node in path:
                return True

            path = path + [node]
            for dependency in self._dependency_graph.get(node, set()):
                if has_circular_dependency(dependency, path):
                    return True

            return False

        for check_name in self.health_checks:
            if has_circular_dependency(check_name, []):
                # Find the actual circular path
                visited = set()
                path = []

                def find_cycle(node: str) -> Optional[List[str]]:
                    if node in visited:
                        cycle_start = path.index(node)
                        return path[cycle_start:] + [node]

                    visited.add(node)
                    path.append(node)

                    for dependency in self._dependency_graph.get(node, set()):
                        cycle = find_cycle(dependency)
                        if cycle:
                            return cycle

                    path.pop()
                    return None

                cycle = find_cycle(check_name)
                if cycle:
                    circular_deps[check_name] = cycle

        return circular_deps


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
