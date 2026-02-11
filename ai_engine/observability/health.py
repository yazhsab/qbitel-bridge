"""
Health Checks and Readiness Probes

Kubernetes-compatible health checking for:
- Liveness probes (is the service alive?)
- Readiness probes (is the service ready to serve?)
- Startup probes (has the service started?)
- Component health aggregation

Supports:
- HTTP endpoints
- gRPC health checking protocol
- Custom health checks
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    checked_at: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "checked_at": self.checked_at.isoformat(),
            "details": self.details,
        }


@dataclass
class HealthCheck:
    """Health check definition."""

    name: str
    check_fn: Callable[[], bool]
    description: str = ""
    timeout_seconds: float = 5.0
    critical: bool = True
    tags: List[str] = field(default_factory=list)


class LivenessCheck(HealthCheck):
    """Liveness check - is the service alive?"""

    def __init__(
        self,
        name: str,
        check_fn: Callable[[], bool],
        description: str = "",
        timeout_seconds: float = 5.0,
    ):
        super().__init__(
            name=name,
            check_fn=check_fn,
            description=description or "Liveness check",
            timeout_seconds=timeout_seconds,
            critical=True,
            tags=["liveness"],
        )


class ReadinessCheck(HealthCheck):
    """Readiness check - is the service ready to serve?"""

    def __init__(
        self,
        name: str,
        check_fn: Callable[[], bool],
        description: str = "",
        timeout_seconds: float = 10.0,
        critical: bool = True,
    ):
        super().__init__(
            name=name,
            check_fn=check_fn,
            description=description or "Readiness check",
            timeout_seconds=timeout_seconds,
            critical=critical,
            tags=["readiness"],
        )


class HealthRegistry:
    """
    Central health check registry.

    Provides:
    - Health check registration
    - Aggregated health status
    - Kubernetes probe endpoints
    - Health check caching
    """

    def __init__(
        self,
        cache_ttl_seconds: float = 5.0,
    ):
        self.cache_ttl = cache_ttl_seconds
        self._checks: Dict[str, HealthCheck] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()
        self._last_check_time: Dict[str, float] = {}

        # Register built-in checks
        self._register_builtin_checks()

    def register(
        self,
        name: str,
        check_fn: Callable[[], bool],
        description: str = "",
        timeout_seconds: float = 5.0,
        critical: bool = True,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register a health check.

        Args:
            name: Check name
            check_fn: Function returning True if healthy
            description: Check description
            timeout_seconds: Check timeout
            critical: If True, unhealthy fails overall health
            tags: Tags for filtering (e.g., "liveness", "readiness")
        """
        check = HealthCheck(
            name=name,
            check_fn=check_fn,
            description=description,
            timeout_seconds=timeout_seconds,
            critical=critical,
            tags=tags or [],
        )
        with self._lock:
            self._checks[name] = check

    def register_liveness(
        self,
        name: str,
        check_fn: Callable[[], bool],
        description: str = "",
    ) -> None:
        """Register a liveness check."""
        self.register(
            name=name,
            check_fn=check_fn,
            description=description,
            critical=True,
            tags=["liveness"],
        )

    def register_readiness(
        self,
        name: str,
        check_fn: Callable[[], bool],
        description: str = "",
        critical: bool = True,
    ) -> None:
        """Register a readiness check."""
        self.register(
            name=name,
            check_fn=check_fn,
            description=description,
            critical=critical,
            tags=["readiness"],
        )

    def unregister(self, name: str) -> bool:
        """Unregister a health check."""
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                self._results.pop(name, None)
                return True
            return False

    def check(
        self,
        name: str,
        use_cache: bool = True,
    ) -> HealthCheckResult:
        """
        Run a specific health check.

        Args:
            name: Check name
            use_cache: Use cached result if available

        Returns:
            HealthCheckResult
        """
        with self._lock:
            check = self._checks.get(name)
            if not check:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message="Check not found",
                )

            # Check cache
            if use_cache:
                last_check = self._last_check_time.get(name, 0)
                if time.time() - last_check < self.cache_ttl:
                    cached = self._results.get(name)
                    if cached:
                        return cached

        # Run check
        result = self._run_check(check)

        with self._lock:
            self._results[name] = result
            self._last_check_time[name] = time.time()

        return result

    def check_all(
        self,
        tags: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks.

        Args:
            tags: Filter by tags (if provided)
            use_cache: Use cached results if available

        Returns:
            Dict of check name to result
        """
        results = {}

        with self._lock:
            checks_to_run = list(self._checks.items())

        for name, check in checks_to_run:
            # Filter by tags
            if tags and not any(t in check.tags for t in tags):
                continue

            results[name] = self.check(name, use_cache)

        return results

    def check_liveness(self, use_cache: bool = True) -> Dict[str, HealthCheckResult]:
        """Check all liveness checks."""
        return self.check_all(tags=["liveness"], use_cache=use_cache)

    def check_readiness(self, use_cache: bool = True) -> Dict[str, HealthCheckResult]:
        """Check all readiness checks."""
        return self.check_all(tags=["readiness"], use_cache=use_cache)

    def get_overall_status(
        self,
        tags: Optional[List[str]] = None,
    ) -> HealthStatus:
        """
        Get overall health status.

        Args:
            tags: Filter checks by tags

        Returns:
            Aggregated HealthStatus
        """
        results = self.check_all(tags=tags)

        if not results:
            return HealthStatus.UNKNOWN

        # Check for critical failures
        critical_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY
            for name, r in results.items()
            if self._checks.get(name, HealthCheck("", lambda: True)).critical
        )

        if critical_unhealthy:
            return HealthStatus.UNHEALTHY

        # Check for any degraded
        if any(r.status == HealthStatus.DEGRADED for r in results.values()):
            return HealthStatus.DEGRADED

        # Check for any unhealthy (non-critical)
        if any(r.status == HealthStatus.UNHEALTHY for r in results.values()):
            return HealthStatus.DEGRADED

        # All healthy
        if all(r.status == HealthStatus.HEALTHY for r in results.values()):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def get_health_response(
        self,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get health response suitable for HTTP endpoint.

        Args:
            tags: Filter checks by tags

        Returns:
            Dict suitable for JSON response
        """
        results = self.check_all(tags=tags)
        overall = self.get_overall_status(tags=tags)

        return {
            "status": overall.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                name: result.to_dict()
                for name, result in results.items()
            },
        }

    def is_live(self) -> bool:
        """Quick liveness check."""
        status = self.get_overall_status(tags=["liveness"])
        return status != HealthStatus.UNHEALTHY

    def is_ready(self) -> bool:
        """Quick readiness check."""
        status = self.get_overall_status(tags=["readiness"])
        return status == HealthStatus.HEALTHY

    def _run_check(self, check: HealthCheck) -> HealthCheckResult:
        """Run a single health check."""
        start_time = time.time()

        try:
            # Run with timeout
            # Note: For true timeout support, use threading or asyncio
            is_healthy = check.check_fn()
            latency = (time.time() - start_time) * 1000

            if is_healthy:
                return HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.HEALTHY,
                    message="OK",
                    latency_ms=latency,
                )
            else:
                return HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Check failed",
                    latency_ms=latency,
                )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check error: {str(e)}",
                latency_ms=latency,
            )

    def _register_builtin_checks(self) -> None:
        """Register built-in health checks."""
        # Basic liveness - always passes
        self.register_liveness(
            "basic_liveness",
            lambda: True,
            "Basic liveness check",
        )

        # Memory check
        def check_memory() -> bool:
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent < 90
            except ImportError:
                return True  # psutil not available

        self.register_readiness(
            "memory",
            check_memory,
            "Memory usage check",
            critical=False,
        )

        # Disk check
        def check_disk() -> bool:
            try:
                import psutil
                disk = psutil.disk_usage("/")
                return disk.percent < 90
            except ImportError:
                return True

        self.register_readiness(
            "disk",
            check_disk,
            "Disk usage check",
            critical=False,
        )


# Global health registry
_health_registry: Optional[HealthRegistry] = None
_health_lock = threading.Lock()


def get_health_registry() -> HealthRegistry:
    """Get or create the global health registry."""
    global _health_registry
    with _health_lock:
        if _health_registry is None:
            _health_registry = HealthRegistry()
        return _health_registry


# Common health check helpers
def create_http_check(
    url: str,
    timeout_seconds: float = 5.0,
    expected_status: int = 200,
) -> Callable[[], bool]:
    """Create an HTTP health check function."""
    def check() -> bool:
        try:
            import urllib.request
            import urllib.error

            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return response.status == expected_status
        except Exception:
            return False

    return check


def create_tcp_check(
    host: str,
    port: int,
    timeout_seconds: float = 5.0,
) -> Callable[[], bool]:
    """Create a TCP health check function."""
    def check() -> bool:
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout_seconds)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    return check


def create_database_check(
    connection_fn: Callable[[], Any],
    query: str = "SELECT 1",
) -> Callable[[], bool]:
    """Create a database health check function."""
    def check() -> bool:
        try:
            conn = connection_fn()
            cursor = conn.cursor()
            cursor.execute(query)
            cursor.fetchone()
            cursor.close()
            return True
        except Exception:
            return False

    return check
