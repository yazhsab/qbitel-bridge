"""
CRONOS AI Engine - Graceful Shutdown Manager

Tracks in-flight requests and ensures clean shutdown without interrupting active operations.
"""

import asyncio
import logging
import time
from typing import Set, Dict, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus Metrics
ACTIVE_REQUESTS = Gauge(
    "http_requests_active",
    "Number of active HTTP requests",
    ["method", "endpoint"],
)
SHUTDOWN_DURATION = Histogram(
    "shutdown_duration_seconds",
    "Time taken to shutdown gracefully",
)
INTERRUPTED_REQUESTS = Counter(
    "shutdown_interrupted_requests_total",
    "Number of requests interrupted during shutdown",
)


@dataclass
class RequestInfo:
    """Information about an active request."""
    request_id: str
    method: str
    path: str
    client_ip: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get request duration in seconds."""
        return (datetime.utcnow() - self.started_at).total_seconds()


class GracefulShutdownManager:
    """
    Manages graceful shutdown of the application.

    Features:
    - Tracks active HTTP requests
    - Waits for in-flight requests to complete
    - Configurable shutdown timeout
    - Metrics for monitoring
    - Request interruption tracking
    """

    def __init__(
        self,
        shutdown_timeout: int = 30,
        max_request_duration_warn: int = 10,
    ):
        """
        Initialize graceful shutdown manager.

        Args:
            shutdown_timeout: Maximum time to wait for active requests (seconds)
            max_request_duration_warn: Warn if request exceeds this duration (seconds)
        """
        self.shutdown_timeout = shutdown_timeout
        self.max_request_duration_warn = max_request_duration_warn

        self._active_requests: Dict[str, RequestInfo] = {}
        self._shutdown_event = asyncio.Event()
        self._is_shutting_down = False
        self._lock = asyncio.Lock()

        logger.info(
            f"GracefulShutdownManager initialized "
            f"(timeout={shutdown_timeout}s, warn={max_request_duration_warn}s)"
        )

    @property
    def is_shutting_down(self) -> bool:
        """Check if system is shutting down."""
        return self._is_shutting_down

    @property
    def active_request_count(self) -> int:
        """Get count of active requests."""
        return len(self._active_requests)

    @asynccontextmanager
    async def track_request(
        self,
        request_id: str,
        method: str,
        path: str,
        client_ip: str = "unknown",
        **metadata
    ):
        """
        Context manager to track an active request.

        Usage:
            async with shutdown_manager.track_request(
                request_id="req-123",
                method="POST",
                path="/api/v1/discover",
                client_ip="192.168.1.100"
            ):
                # Handle request
                await process_request()

        Args:
            request_id: Unique request identifier
            method: HTTP method
            path: Request path
            client_ip: Client IP address
            **metadata: Additional metadata

        Raises:
            RuntimeError: If system is shutting down
        """
        # Check if shutting down
        if self._is_shutting_down:
            raise RuntimeError(
                "System is shutting down. New requests are not accepted."
            )

        # Register request
        request_info = RequestInfo(
            request_id=request_id,
            method=method,
            path=path,
            client_ip=client_ip,
            metadata=metadata,
        )

        async with self._lock:
            self._active_requests[request_id] = request_info

        # Update metrics
        ACTIVE_REQUESTS.labels(method=method, endpoint=path).inc()

        logger.debug(
            f"Request started: {request_id} [{method} {path}] from {client_ip}"
        )

        try:
            yield request_info

        finally:
            # Unregister request
            async with self._lock:
                self._active_requests.pop(request_id, None)

            # Update metrics
            ACTIVE_REQUESTS.labels(method=method, endpoint=path).dec()

            # Log duration
            duration = request_info.duration
            if duration > self.max_request_duration_warn:
                logger.warning(
                    f"Long request: {request_id} [{method} {path}] "
                    f"took {duration:.2f}s"
                )
            else:
                logger.debug(
                    f"Request completed: {request_id} [{method} {path}] "
                    f"in {duration:.2f}s"
                )

    async def initiate_shutdown(self) -> None:
        """
        Initiate graceful shutdown sequence.

        This will:
        1. Set shutdown flag (reject new requests)
        2. Wait for active requests to complete (up to timeout)
        3. Log any interrupted requests
        """
        if self._is_shutting_down:
            logger.warning("Shutdown already in progress")
            return

        logger.info("=" * 80)
        logger.info("Initiating graceful shutdown...")
        logger.info("=" * 80)

        shutdown_start = time.time()

        # Set shutdown flag
        self._is_shutting_down = True
        self._shutdown_event.set()

        # Get active request count
        active_count = self.active_request_count
        if active_count == 0:
            logger.info("✅ No active requests. Shutdown can proceed immediately.")
            SHUTDOWN_DURATION.observe(time.time() - shutdown_start)
            return

        logger.info(
            f"⏳ Waiting for {active_count} active request(s) to complete "
            f"(timeout: {self.shutdown_timeout}s)..."
        )

        # Wait for requests to complete
        await self._wait_for_requests_to_complete()

        shutdown_duration = time.time() - shutdown_start
        SHUTDOWN_DURATION.observe(shutdown_duration)

        # Log final status
        remaining = self.active_request_count
        if remaining == 0:
            logger.info(
                f"✅ All requests completed. "
                f"Shutdown took {shutdown_duration:.2f}s"
            )
        else:
            logger.warning(
                f"⚠️  {remaining} request(s) did not complete within timeout. "
                f"Shutdown took {shutdown_duration:.2f}s"
            )
            INTERRUPTED_REQUESTS.inc(remaining)

            # Log interrupted requests
            async with self._lock:
                for req_id, req_info in self._active_requests.items():
                    logger.warning(
                        f"Interrupted request: {req_id} "
                        f"[{req_info.method} {req_info.path}] "
                        f"from {req_info.client_ip} "
                        f"(duration: {req_info.duration:.2f}s)"
                    )

        logger.info("=" * 80)
        logger.info("Graceful shutdown complete")
        logger.info("=" * 80)

    async def _wait_for_requests_to_complete(self) -> None:
        """Wait for active requests to complete with timeout."""
        end_time = time.time() + self.shutdown_timeout

        while self.active_request_count > 0:
            # Check timeout
            remaining_time = end_time - time.time()
            if remaining_time <= 0:
                logger.warning(
                    f"Shutdown timeout reached. "
                    f"{self.active_request_count} request(s) still active."
                )
                break

            # Log progress every 5 seconds
            if int(remaining_time) % 5 == 0:
                logger.info(
                    f"⏳ Still waiting for {self.active_request_count} request(s). "
                    f"Timeout in {remaining_time:.0f}s..."
                )

            # Wait a bit before checking again
            await asyncio.sleep(0.5)

    def get_active_requests(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active requests.

        Returns:
            Dict mapping request_id to request info
        """
        return {
            req_id: {
                "request_id": req_info.request_id,
                "method": req_info.method,
                "path": req_info.path,
                "client_ip": req_info.client_ip,
                "started_at": req_info.started_at.isoformat(),
                "duration_seconds": req_info.duration,
                "metadata": req_info.metadata,
            }
            for req_id, req_info in self._active_requests.items()
        }

    def get_shutdown_status(self) -> Dict[str, Any]:
        """
        Get current shutdown status.

        Returns:
            Dict with shutdown status information
        """
        return {
            "is_shutting_down": self._is_shutting_down,
            "active_requests": self.active_request_count,
            "shutdown_timeout": self.shutdown_timeout,
            "active_request_details": self.get_active_requests(),
        }


# Global shutdown manager instance
_shutdown_manager: Optional[GracefulShutdownManager] = None


def get_shutdown_manager() -> GracefulShutdownManager:
    """
    Get global shutdown manager instance.

    Returns:
        GracefulShutdownManager: Global instance

    Raises:
        RuntimeError: If not initialized
    """
    global _shutdown_manager
    if _shutdown_manager is None:
        raise RuntimeError(
            "GracefulShutdownManager not initialized. "
            "Call initialize_shutdown_manager() during app startup."
        )
    return _shutdown_manager


def initialize_shutdown_manager(
    shutdown_timeout: int = 30,
    max_request_duration_warn: int = 10,
) -> GracefulShutdownManager:
    """
    Initialize global shutdown manager.

    Args:
        shutdown_timeout: Maximum time to wait for active requests
        max_request_duration_warn: Warn threshold for long requests

    Returns:
        GracefulShutdownManager: Initialized manager
    """
    global _shutdown_manager

    if _shutdown_manager is not None:
        logger.warning("GracefulShutdownManager already initialized")
        return _shutdown_manager

    _shutdown_manager = GracefulShutdownManager(
        shutdown_timeout=shutdown_timeout,
        max_request_duration_warn=max_request_duration_warn,
    )

    return _shutdown_manager


# FastAPI middleware for automatic request tracking
class RequestTrackingMiddleware:
    """
    FastAPI middleware to automatically track requests during shutdown.

    This middleware:
    - Tracks all incoming requests
    - Rejects new requests during shutdown
    - Provides request tracking for graceful shutdown
    """

    def __init__(self, app, shutdown_manager: GracefulShutdownManager):
        """
        Initialize middleware.

        Args:
            app: FastAPI application
            shutdown_manager: Shutdown manager instance
        """
        self.app = app
        self.shutdown_manager = shutdown_manager

    async def __call__(self, scope, receive, send):
        """Process request with tracking."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check if shutting down
        if self.shutdown_manager.is_shutting_down:
            # Return 503 Service Unavailable
            await send({
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"retry-after", b"60"],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"detail":"Service is shutting down. Please retry later."}',
            })
            return

        # Generate request ID
        request_id = scope.get("headers", {}).get("x-request-id", f"req-{id(scope)}")
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        client = scope.get("client", ("unknown", 0))
        client_ip = client[0] if client else "unknown"

        # Track request
        async with self.shutdown_manager.track_request(
            request_id=str(request_id),
            method=method,
            path=path,
            client_ip=client_ip,
        ):
            await self.app(scope, receive, send)
