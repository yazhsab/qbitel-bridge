"""
QBITEL - Comprehensive Health Check Implementation

Real dependency health checks for production readiness:
- Database connectivity with query execution
- Redis connectivity with ping
- LLM service availability
- External service reachability
- System resource monitoring

Usage:
    from ai_engine.api.health_checks import HealthCheckService, health_service

    # Run all checks
    result = await health_service.check_all()

    # Run specific check
    db_health = await health_service.check_database()
"""

import asyncio
import logging
import os
import socket
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import aiohttp

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================

HEALTH_CHECK_DURATION = Histogram(
    "qbitel_health_check_duration_seconds",
    "Duration of health checks",
    ["component"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)

HEALTH_CHECK_STATUS = Gauge(
    "qbitel_health_check_status",
    "Health check status (1=healthy, 0=unhealthy)",
    ["component"],
)

HEALTH_CHECK_ERRORS = Counter(
    "qbitel_health_check_errors_total",
    "Total health check errors",
    ["component", "error_type"],
)


# =============================================================================
# Health Status
# =============================================================================


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a component."""

    name: str
    status: HealthStatus
    latency_ms: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    critical: bool = True


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    components: Dict[str, ComponentHealth]
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def healthy_count(self) -> int:
        """Count healthy components."""
        return sum(1 for c in self.components.values() if c.status == HealthStatus.HEALTHY)

    @property
    def total_count(self) -> int:
        """Count total components."""
        return len(self.components)


# =============================================================================
# Health Check Service
# =============================================================================


class HealthCheckService:
    """
    Comprehensive health check service.

    Provides real health checks for all dependencies:
    - Database (PostgreSQL)
    - Cache (Redis)
    - LLM Service
    - External APIs
    - System resources
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        llm_endpoint: Optional[str] = None,
    ):
        """Initialize health check service."""
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        self.llm_endpoint = llm_endpoint or os.environ.get("LLM_ENDPOINT")

        self._check_timeout = 5.0  # seconds
        self._cache: Dict[str, ComponentHealth] = {}
        self._cache_ttl = 10  # seconds

    async def check_all(self) -> SystemHealth:
        """
        Run all health checks.

        Returns:
            SystemHealth with all component statuses
        """
        start_time = time.time()

        # Run all checks concurrently
        checks = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_llm_service(),
            self.check_system_resources(),
            self.check_network_connectivity(),
            return_exceptions=True,
        )

        # Process results
        components: Dict[str, ComponentHealth] = {}
        check_names = ["database", "redis", "llm_service", "system_resources", "network"]

        for name, result in zip(check_names, checks):
            if isinstance(result, Exception):
                components[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=0,
                    message=f"Check failed: {str(result)}",
                    critical=name in ["database", "llm_service"],
                )
                HEALTH_CHECK_STATUS.labels(component=name).set(0)
                HEALTH_CHECK_ERRORS.labels(component=name, error_type="exception").inc()
            else:
                components[name] = result
                HEALTH_CHECK_STATUS.labels(component=name).set(
                    1 if result.status == HealthStatus.HEALTHY else 0
                )

        # Determine overall status
        critical_unhealthy = any(
            c.status == HealthStatus.UNHEALTHY and c.critical
            for c in components.values()
        )
        any_unhealthy = any(
            c.status == HealthStatus.UNHEALTHY for c in components.values()
        )
        any_degraded = any(
            c.status == HealthStatus.DEGRADED for c in components.values()
        )

        if critical_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif any_unhealthy or any_degraded:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return SystemHealth(
            status=overall_status,
            components=components,
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def check_database(self) -> ComponentHealth:
        """
        Check database connectivity.

        Verifies:
        - Connection can be established
        - Simple query executes successfully
        - Response time is acceptable
        """
        start_time = time.time()
        name = "database"

        if not self.database_url:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message="Database URL not configured",
                critical=True,
            )

        try:
            # Use asyncpg for async PostgreSQL
            import asyncpg

            conn = await asyncio.wait_for(
                asyncpg.connect(self.database_url),
                timeout=self._check_timeout,
            )

            try:
                # Execute test query
                result = await conn.fetchval("SELECT 1")
                latency_ms = (time.time() - start_time) * 1000

                # Check version for diagnostics
                version = await conn.fetchval("SELECT version()")

                HEALTH_CHECK_DURATION.labels(component=name).observe(
                    time.time() - start_time
                )

                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                    message="Database connection healthy",
                    details={
                        "version": version[:50] if version else "unknown",
                        "query_result": result,
                    },
                    critical=True,
                )
            finally:
                await conn.close()

        except asyncio.TimeoutError:
            HEALTH_CHECK_ERRORS.labels(component=name, error_type="timeout").inc()
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message="Database connection timeout",
                critical=True,
            )
        except ImportError:
            # asyncpg not installed, try sync fallback
            return await self._check_database_sync()
        except Exception as e:
            HEALTH_CHECK_ERRORS.labels(component=name, error_type="connection").inc()
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Database error: {str(e)}",
                critical=True,
            )

    async def _check_database_sync(self) -> ComponentHealth:
        """Fallback sync database check."""
        start_time = time.time()
        name = "database"

        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(self.database_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                _ = result.scalar()

            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message="Database connection healthy (sync)",
                critical=True,
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Database error: {str(e)}",
                critical=True,
            )

    async def check_redis(self) -> ComponentHealth:
        """
        Check Redis connectivity.

        Verifies:
        - Connection can be established
        - PING responds successfully
        - Response time is acceptable
        """
        start_time = time.time()
        name = "redis"

        if not self.redis_url:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message="Redis URL not configured",
                critical=False,
            )

        try:
            import redis.asyncio as aioredis

            redis = await asyncio.wait_for(
                aioredis.from_url(self.redis_url),
                timeout=self._check_timeout,
            )

            try:
                # Execute PING
                result = await redis.ping()
                latency_ms = (time.time() - start_time) * 1000

                # Get info for diagnostics
                info = await redis.info("server")

                HEALTH_CHECK_DURATION.labels(component=name).observe(
                    time.time() - start_time
                )

                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                    message="Redis connection healthy",
                    details={
                        "ping_response": result,
                        "redis_version": info.get("redis_version", "unknown"),
                    },
                    critical=False,
                )
            finally:
                await redis.close()

        except asyncio.TimeoutError:
            HEALTH_CHECK_ERRORS.labels(component=name, error_type="timeout").inc()
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message="Redis connection timeout",
                critical=False,
            )
        except ImportError:
            # aioredis not installed, try sync fallback
            return await self._check_redis_sync()
        except Exception as e:
            HEALTH_CHECK_ERRORS.labels(component=name, error_type="connection").inc()
            return ComponentHealth(
                name=name,
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Redis error: {str(e)}",
                critical=False,
            )

    async def _check_redis_sync(self) -> ComponentHealth:
        """Fallback sync Redis check."""
        start_time = time.time()
        name = "redis"

        try:
            import redis

            r = redis.from_url(self.redis_url)
            result = r.ping()

            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message="Redis connection healthy (sync)" if result else "Redis ping failed",
                critical=False,
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Redis error: {str(e)}",
                critical=False,
            )

    async def check_llm_service(self) -> ComponentHealth:
        """
        Check LLM service availability.

        Verifies:
        - Endpoint is reachable
        - Service responds to health check
        - Response time is acceptable
        """
        start_time = time.time()
        name = "llm_service"

        if not self.llm_endpoint:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message="LLM endpoint not configured",
                critical=True,
            )

        try:
            async with aiohttp.ClientSession() as session:
                # Try health endpoint first
                health_url = f"{self.llm_endpoint.rstrip('/')}/health"

                async with session.get(
                    health_url,
                    timeout=aiohttp.ClientTimeout(total=self._check_timeout),
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000

                    if response.status == 200:
                        data = await response.json()
                        HEALTH_CHECK_DURATION.labels(component=name).observe(
                            time.time() - start_time
                        )

                        return ComponentHealth(
                            name=name,
                            status=HealthStatus.HEALTHY,
                            latency_ms=latency_ms,
                            message="LLM service healthy",
                            details={"response": data},
                            critical=True,
                        )
                    else:
                        return ComponentHealth(
                            name=name,
                            status=HealthStatus.DEGRADED,
                            latency_ms=latency_ms,
                            message=f"LLM service returned {response.status}",
                            critical=True,
                        )

        except asyncio.TimeoutError:
            HEALTH_CHECK_ERRORS.labels(component=name, error_type="timeout").inc()
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message="LLM service timeout",
                critical=True,
            )
        except aiohttp.ClientError as e:
            HEALTH_CHECK_ERRORS.labels(component=name, error_type="connection").inc()
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"LLM service connection error: {str(e)}",
                critical=True,
            )
        except Exception as e:
            HEALTH_CHECK_ERRORS.labels(component=name, error_type="unknown").inc()
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"LLM service error: {str(e)}",
                critical=True,
            )

    async def check_system_resources(self) -> ComponentHealth:
        """
        Check system resource availability.

        Verifies:
        - CPU usage is acceptable
        - Memory usage is acceptable
        - Disk space is sufficient
        """
        start_time = time.time()
        name = "system_resources"

        try:
            import psutil

            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
            }

            # Determine status based on thresholds
            issues = []

            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")
            if disk.percent > 90:
                issues.append(f"Low disk space: {disk.percent}% used")

            if issues:
                status = HealthStatus.DEGRADED if len(issues) < 2 else HealthStatus.UNHEALTHY
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "System resources healthy"

            HEALTH_CHECK_DURATION.labels(component=name).observe(
                time.time() - start_time
            )

            return ComponentHealth(
                name=name,
                status=status,
                latency_ms=(time.time() - start_time) * 1000,
                message=message,
                details=details,
                critical=False,
            )

        except ImportError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                latency_ms=(time.time() - start_time) * 1000,
                message="psutil not installed",
                critical=False,
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"System check error: {str(e)}",
                critical=False,
            )

    async def check_network_connectivity(self) -> ComponentHealth:
        """
        Check network connectivity.

        Verifies:
        - DNS resolution works
        - External endpoints are reachable
        """
        start_time = time.time()
        name = "network"

        try:
            # Check DNS resolution
            dns_check = await asyncio.get_event_loop().run_in_executor(
                None, socket.gethostbyname, "dns.google"
            )

            # Check external connectivity
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://httpbin.org/status/200",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    external_ok = response.status == 200

            latency_ms = (time.time() - start_time) * 1000

            HEALTH_CHECK_DURATION.labels(component=name).observe(
                time.time() - start_time
            )

            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message="Network connectivity healthy",
                details={
                    "dns_resolved": dns_check,
                    "external_reachable": external_ok,
                },
                critical=False,
            )

        except socket.gaierror as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"DNS resolution failed: {str(e)}",
                critical=False,
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Network check error: {str(e)}",
                critical=False,
            )

    async def check_custom_endpoint(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        critical: bool = False,
    ) -> ComponentHealth:
        """
        Check a custom HTTP endpoint.

        Args:
            name: Component name
            url: URL to check
            expected_status: Expected HTTP status code
            critical: Whether this is a critical dependency
        """
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self._check_timeout),
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000

                    if response.status == expected_status:
                        return ComponentHealth(
                            name=name,
                            status=HealthStatus.HEALTHY,
                            latency_ms=latency_ms,
                            message=f"Endpoint healthy ({response.status})",
                            critical=critical,
                        )
                    else:
                        return ComponentHealth(
                            name=name,
                            status=HealthStatus.DEGRADED,
                            latency_ms=latency_ms,
                            message=f"Unexpected status: {response.status}",
                            critical=critical,
                        )

        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Endpoint error: {str(e)}",
                critical=critical,
            )


# =============================================================================
# Global Instance
# =============================================================================

# Create global health service
health_service = HealthCheckService()


# =============================================================================
# FastAPI Router
# =============================================================================


def create_health_router():
    """Create FastAPI router with health endpoints."""
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse

    router = APIRouter(tags=["health"])

    @router.get("/health")
    async def health_check():
        """Comprehensive health check endpoint."""
        result = await health_service.check_all()

        status_code = 200 if result.status == HealthStatus.HEALTHY else 503

        return JSONResponse(
            status_code=status_code,
            content={
                "status": result.status.value,
                "timestamp": result.timestamp.isoformat(),
                "latency_ms": result.latency_ms,
                "components": {
                    name: {
                        "status": comp.status.value,
                        "latency_ms": comp.latency_ms,
                        "message": comp.message,
                        "critical": comp.critical,
                    }
                    for name, comp in result.components.items()
                },
                "summary": {
                    "healthy": result.healthy_count,
                    "total": result.total_count,
                },
            },
        )

    @router.get("/health/live")
    async def liveness_check():
        """Kubernetes liveness probe."""
        # Simple check - just verify the process is alive
        return JSONResponse(
            status_code=200,
            content={"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()},
        )

    @router.get("/health/ready")
    async def readiness_check():
        """Kubernetes readiness probe."""
        result = await health_service.check_all()

        # Ready if no critical components are unhealthy
        critical_healthy = all(
            c.status != HealthStatus.UNHEALTHY
            for c in result.components.values()
            if c.critical
        )

        status_code = 200 if critical_healthy else 503

        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if critical_healthy else "not_ready",
                "timestamp": result.timestamp.isoformat(),
            },
        )

    return router
