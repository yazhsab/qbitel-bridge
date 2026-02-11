"""
QBITEL Engine - Base Integration Connector

Abstract base classes and common functionality for external system integrations.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..config import get_security_config
from ..logging import get_security_logger, SecurityLogType, LogLevel
from ..models import SecurityEvent, ThreatAnalysis, AutomatedResponse


class IntegrationStatus(str, Enum):
    """Integration status values."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    ERROR = "error"
    UNKNOWN = "unknown"


class IntegrationType(str, Enum):
    """Types of integrations."""

    SIEM = "siem"
    TICKETING = "ticketing"
    COMMUNICATION = "communication"
    NETWORK_SECURITY = "network_security"
    THREAT_INTELLIGENCE = "threat_intelligence"
    ORCHESTRATION = "orchestration"


@dataclass
class IntegrationConfig:
    """Configuration for an integration."""

    name: str
    integration_type: IntegrationType
    enabled: bool
    endpoint: Optional[str] = None
    credentials: Optional[Dict[str, str]] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    rate_limit_requests_per_minute: Optional[int] = None
    custom_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_config is None:
            self.custom_config = {}


@dataclass
class IntegrationResult:
    """Result of an integration operation."""

    success: bool
    message: str
    response_data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    execution_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseIntegrationConnector(ABC):
    """
    Abstract base class for external system integrations.

    Provides common functionality for authentication, error handling,
    rate limiting, and logging.
    """

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.logger = get_security_logger(f"qbitel.security.integration.{config.name}")
        self.security_config = get_security_config()

        # Connection state
        self.status = IntegrationStatus.UNKNOWN
        self.last_connection_attempt: Optional[datetime] = None
        self.connection_errors: List[str] = []

        # Rate limiting
        self._rate_limit_tokens = config.rate_limit_requests_per_minute or 1000
        self._rate_limit_window_start = datetime.now()
        self._rate_limit_lock = asyncio.Lock()

        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_response_time = 0.0

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Integration connector initialized: {config.name}",
            level=LogLevel.INFO,
            metadata={
                "integration_type": config.integration_type.value,
                "enabled": config.enabled,
                "endpoint": config.endpoint,
            },
        )

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the integration connection.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def test_connection(self) -> IntegrationResult:
        """
        Test the integration connection.

        Returns:
            Result of connection test
        """
        pass

    @abstractmethod
    async def send_security_event(
        self, security_event: SecurityEvent
    ) -> IntegrationResult:
        """
        Send a security event to the external system.

        Args:
            security_event: Security event to send

        Returns:
            Result of the operation
        """
        pass

    async def send_threat_analysis(
        self, threat_analysis: ThreatAnalysis
    ) -> IntegrationResult:
        """
        Send threat analysis results to the external system.

        Default implementation - can be overridden by specific connectors.

        Args:
            threat_analysis: Threat analysis to send

        Returns:
            Result of the operation
        """
        return IntegrationResult(
            success=False,
            message="Threat analysis sending not implemented for this integration",
            error_code="NOT_IMPLEMENTED",
        )

    async def send_response_execution(
        self, response: AutomatedResponse
    ) -> IntegrationResult:
        """
        Send automated response execution details to the external system.

        Default implementation - can be overridden by specific connectors.

        Args:
            response: Automated response to send

        Returns:
            Result of the operation
        """
        return IntegrationResult(
            success=False,
            message="Response execution sending not implemented for this integration",
            error_code="NOT_IMPLEMENTED",
        )

    async def execute_with_retry(
        self, operation_func, operation_name: str, *args, **kwargs
    ) -> IntegrationResult:
        """
        Execute an operation with retry logic and error handling.

        Args:
            operation_func: The operation function to execute
            operation_name: Name of the operation for logging
            *args: Arguments for the operation function
            **kwargs: Keyword arguments for the operation function

        Returns:
            Result of the operation
        """

        start_time = asyncio.get_event_loop().time()
        self.total_requests += 1

        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                # Rate limiting check
                if not await self._check_rate_limit():
                    return IntegrationResult(
                        success=False,
                        message="Rate limit exceeded",
                        error_code="RATE_LIMIT_EXCEEDED",
                    )

                # Execute operation with timeout
                result = await asyncio.wait_for(
                    operation_func(*args, **kwargs), timeout=self.config.timeout_seconds
                )

                # Calculate execution time
                execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

                if isinstance(result, IntegrationResult):
                    result.execution_time_ms = execution_time

                    if result.success:
                        self.successful_requests += 1
                        self.status = IntegrationStatus.CONNECTED
                        self._update_average_response_time(execution_time)

                        self.logger.log_security_event(
                            SecurityLogType.PERFORMANCE_METRIC,
                            f"{operation_name} completed successfully",
                            level=LogLevel.DEBUG,
                            component=self.config.name,
                            execution_time_ms=execution_time,
                        )
                    else:
                        self.failed_requests += 1
                        self.logger.log_security_event(
                            SecurityLogType.PERFORMANCE_METRIC,
                            f"{operation_name} failed: {result.message}",
                            level=LogLevel.WARNING,
                            component=self.config.name,
                            error_code=result.error_code,
                            execution_time_ms=execution_time,
                        )

                    return result
                else:
                    # Convert non-IntegrationResult to IntegrationResult
                    self.successful_requests += 1
                    self._update_average_response_time(execution_time)

                    return IntegrationResult(
                        success=True,
                        message=f"{operation_name} completed",
                        response_data=(
                            result if isinstance(result, dict) else {"result": result}
                        ),
                        execution_time_ms=execution_time,
                    )

            except asyncio.TimeoutError:
                error_msg = (
                    f"{operation_name} timed out after {self.config.timeout_seconds}s"
                )
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Attempt {attempt}/{self.config.retry_attempts}: {error_msg}",
                    level=LogLevel.WARNING,
                    component=self.config.name,
                    error_code="TIMEOUT",
                )

                if attempt < self.config.retry_attempts:
                    await asyncio.sleep(self.config.retry_delay_seconds * attempt)
                    continue
                else:
                    self.failed_requests += 1
                    self.status = IntegrationStatus.ERROR
                    return IntegrationResult(
                        success=False,
                        message=error_msg,
                        error_code="TIMEOUT",
                        execution_time_ms=(asyncio.get_event_loop().time() - start_time)
                        * 1000,
                    )

            except Exception as e:
                error_msg = f"{operation_name} failed: {str(e)}"
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Attempt {attempt}/{self.config.retry_attempts}: {error_msg}",
                    level=LogLevel.ERROR,
                    component=self.config.name,
                    error_code="INTEGRATION_ERROR",
                    stack_trace=str(e),
                )

                if attempt < self.config.retry_attempts:
                    await asyncio.sleep(self.config.retry_delay_seconds * attempt)
                    continue
                else:
                    self.failed_requests += 1
                    self.status = IntegrationStatus.ERROR
                    self.connection_errors.append(str(e))

                    return IntegrationResult(
                        success=False,
                        message=error_msg,
                        error_code="INTEGRATION_ERROR",
                        execution_time_ms=(asyncio.get_event_loop().time() - start_time)
                        * 1000,
                        metadata={"exception": str(e)},
                    )

        # Should not reach here
        self.failed_requests += 1
        return IntegrationResult(
            success=False,
            message=f"{operation_name} failed after all retry attempts",
            error_code="MAX_RETRIES_EXCEEDED",
        )

    async def _check_rate_limit(self) -> bool:
        """Check if request is within rate limit."""
        if not self.config.rate_limit_requests_per_minute:
            return True

        async with self._rate_limit_lock:
            now = datetime.now()

            # Reset window if a minute has passed
            if (now - self._rate_limit_window_start).total_seconds() >= 60:
                self._rate_limit_tokens = self.config.rate_limit_requests_per_minute
                self._rate_limit_window_start = now

            if self._rate_limit_tokens > 0:
                self._rate_limit_tokens -= 1
                return True
            else:
                return False

    def _update_average_response_time(self, response_time_ms: float):
        """Update average response time with new measurement."""
        if self.successful_requests == 1:
            self.average_response_time = response_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_response_time = (
                alpha * response_time_ms + (1 - alpha) * self.average_response_time
            )

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and metrics."""
        success_rate = self.successful_requests / max(self.total_requests, 1) * 100

        return {
            "name": self.config.name,
            "type": self.config.integration_type.value,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "last_connection_attempt": (
                self.last_connection_attempt.isoformat()
                if self.last_connection_attempt
                else None
            ),
            "metrics": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate_percent": success_rate,
                "average_response_time_ms": self.average_response_time,
            },
            "configuration": {
                "endpoint": self.config.endpoint,
                "timeout_seconds": self.config.timeout_seconds,
                "retry_attempts": self.config.retry_attempts,
                "rate_limit": self.config.rate_limit_requests_per_minute,
            },
            "recent_errors": (
                self.connection_errors[-5:] if self.connection_errors else []
            ),
        }

    def reset_metrics(self):
        """Reset connection metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_response_time = 0.0
        self.connection_errors.clear()

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Integration metrics reset: {self.config.name}",
            level=LogLevel.INFO,
            component=self.config.name,
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the integration.

        Returns:
            Health check result
        """
        try:
            # Test basic connectivity
            test_result = await self.test_connection()

            if test_result.success:
                health_status = "healthy"
                message = "Integration is healthy"
            else:
                health_status = (
                    "unhealthy"
                    if self.status == IntegrationStatus.ERROR
                    else "degraded"
                )
                message = test_result.message

            return {
                "status": health_status,
                "message": message,
                "details": {
                    "integration_status": self.status.value,
                    "response_time_ms": test_result.execution_time_ms,
                    "last_error": (
                        self.connection_errors[-1] if self.connection_errors else None
                    ),
                },
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "details": {"error": str(e)},
            }

    async def shutdown(self):
        """Shutdown the integration connector."""
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Integration connector shutting down: {self.config.name}",
            level=LogLevel.INFO,
            component=self.config.name,
        )

        self.status = IntegrationStatus.DISCONNECTED


class HTTPIntegrationConnector(BaseIntegrationConnector):
    """
    Base class for HTTP-based integrations.

    Provides common HTTP functionality including authentication,
    request formatting, and response handling.
    """

    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.session: Optional[Any] = None  # aiohttp session
        self.auth_headers: Dict[str, str] = {}

    async def initialize(self) -> bool:
        """Initialize HTTP session and authentication."""
        try:
            import aiohttp

            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Setup authentication
            await self._setup_authentication()

            self.last_connection_attempt = datetime.now()
            self.status = IntegrationStatus.CONNECTED

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"HTTP integration initialized: {self.config.name}",
                level=LogLevel.INFO,
                component=self.config.name,
            )

            return True

        except Exception as e:
            self.status = IntegrationStatus.ERROR
            self.connection_errors.append(str(e))

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"HTTP integration initialization failed: {self.config.name} - {str(e)}",
                level=LogLevel.ERROR,
                component=self.config.name,
                error_code="INIT_FAILED",
            )

            return False

    async def _setup_authentication(self):
        """Setup HTTP authentication headers."""
        if not self.config.credentials:
            return

        auth_type = self.config.credentials.get("type", "bearer")

        if auth_type == "bearer":
            token = self.config.credentials.get("token")
            if token:
                self.auth_headers["Authorization"] = f"Bearer {token}"

        elif auth_type == "api_key":
            api_key = self.config.credentials.get("api_key")
            key_header = self.config.credentials.get("key_header", "X-API-Key")
            if api_key:
                self.auth_headers[key_header] = api_key

        elif auth_type == "basic":
            username = self.config.credentials.get("username")
            password = self.config.credentials.get("password")
            if username and password:
                import base64

                credentials = base64.b64encode(
                    f"{username}:{password}".encode()
                ).decode()
                self.auth_headers["Authorization"] = f"Basic {credentials}"

    async def make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to the integration endpoint.

        Args:
            method: HTTP method
            endpoint: API endpoint (relative to base URL)
            data: Request data
            headers: Additional headers

        Returns:
            Response data
        """

        if not self.session:
            raise Exception("HTTP session not initialized")

        # Build full URL
        base_url = self.config.endpoint.rstrip("/")
        full_url = f"{base_url}/{endpoint.lstrip('/')}"

        # Prepare headers
        request_headers = self.auth_headers.copy()
        request_headers["Content-Type"] = "application/json"
        request_headers["User-Agent"] = "QBITEL-Security-Orchestrator/1.0"

        if headers:
            request_headers.update(headers)

        # Make request
        async with self.session.request(
            method=method.upper(), url=full_url, json=data, headers=request_headers
        ) as response:

            # Check response status
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")

            # Parse response
            if response.content_type == "application/json":
                return await response.json()
            else:
                return {"content": await response.text()}

    async def test_connection(self) -> IntegrationResult:
        """Test HTTP connection."""
        try:
            if not self.session:
                await self.initialize()

            # Try to make a simple request (usually a health check endpoint)
            health_endpoint = self.config.custom_config.get(
                "health_endpoint", "/health"
            )

            response_data = await self.make_request("GET", health_endpoint)

            return IntegrationResult(
                success=True,
                message="HTTP connection test successful",
                response_data=response_data,
            )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"HTTP connection test failed: {str(e)}",
                error_code="CONNECTION_TEST_FAILED",
            )

    async def shutdown(self):
        """Shutdown HTTP integration."""
        if self.session:
            await self.session.close()
            self.session = None

        await super().shutdown()
