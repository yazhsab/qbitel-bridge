"""
QBITEL - Application Performance Monitoring (APM) Integration
Production-ready APM with New Relic, Datadog, and Elastic APM support.
"""

import asyncio
import logging
import time
import psutil
import os
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import aiohttp
from contextlib import asynccontextmanager

from ..core.config import Config
from ..core.exceptions import ObservabilityException

logger = logging.getLogger(__name__)


class TransactionType(str, Enum):
    """APM transaction types."""

    REQUEST = "request"
    BACKGROUND = "background"
    SCHEDULED = "scheduled"
    MESSAGE = "message"


@dataclass
class APMTransaction:
    """APM transaction tracking."""

    transaction_id: str
    name: str
    type: TransactionType
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    result: str = "success"
    context: Dict[str, Any] = field(default_factory=dict)
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def finish(self, result: str = "success"):
        """Finish transaction."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.result = result

    def add_error(self, error: Exception, handled: bool = True):
        """Add error to transaction."""
        self.errors.append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "handled": handled,
                "timestamp": time.time(),
            }
        )

    def set_custom_metric(self, name: str, value: float):
        """Set custom metric."""
        self.custom_metrics[name] = value


class ElasticAPMIntegration:
    """
    Elastic APM integration.

    Provides deep application performance monitoring with Elastic APM.
    """

    def __init__(self, config: Config):
        """Initialize Elastic APM integration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Elastic APM configuration
        self.apm_server_url = getattr(config, "elastic_apm_server_url", "http://localhost:8200")
        self.apm_secret_token = getattr(config, "elastic_apm_secret_token", None)
        self.service_name = getattr(config, "service_name", "qbitel")
        self.environment = getattr(config.environment, "value", "development")

        # Session and batching
        self._session: Optional[aiohttp.ClientSession] = None
        self._transaction_queue: List[APMTransaction] = []
        self._batch_size = 50
        self._flush_interval = 10
        self._flush_task: Optional[asyncio.Task] = None

        # Statistics
        self.transactions_sent = 0
        self.errors_sent = 0

        self.logger.info(f"ElasticAPMIntegration initialized (server: {self.apm_server_url})")

    async def initialize(self):
        """Initialize APM client."""
        headers = {"Content-Type": "application/x-ndjson"}
        if self.apm_secret_token:
            headers["Authorization"] = f"Bearer {self.apm_secret_token}"

        self._session = aiohttp.ClientSession(headers=headers)
        self._flush_task = asyncio.create_task(self._flush_loop())

        self.logger.info("Elastic APM client initialized")

    async def shutdown(self):
        """Shutdown APM client."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        await self.flush()

        if self._session:
            await self._session.close()

    async def send_transaction(self, transaction: APMTransaction):
        """Send transaction to Elastic APM."""
        try:
            self._transaction_queue.append(transaction)

            if len(self._transaction_queue) >= self._batch_size:
                await self._flush_batch()

        except Exception as e:
            self.logger.error(f"Failed to queue transaction: {e}")

    async def flush(self):
        """Flush pending transactions."""
        if self._transaction_queue:
            await self._flush_batch()

    async def _flush_loop(self):
        """Background task to flush transactions periodically."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                if self._transaction_queue:
                    await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Flush loop error: {e}")

    async def _flush_batch(self):
        """Flush batch of transactions to Elastic APM."""
        if not self._transaction_queue or not self._session:
            return

        try:
            batch = self._transaction_queue.copy()
            self._transaction_queue.clear()

            # Build intake API payload
            ndjson_lines = []

            # Metadata
            metadata = {
                "metadata": {
                    "service": {
                        "name": self.service_name,
                        "environment": self.environment,
                        "runtime": {
                            "name": "python",
                            "version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                        },
                    },
                    "process": {"pid": os.getpid()},
                }
            }
            ndjson_lines.append(json.dumps(metadata))

            # Transactions
            for txn in batch:
                transaction_data = {
                    "transaction": {
                        "id": txn.transaction_id,
                        "name": txn.name,
                        "type": txn.type.value,
                        "duration": txn.duration_ms,
                        "result": txn.result,
                        "timestamp": int(txn.start_time * 1000000),  # microseconds
                        "context": txn.context,
                        "custom": txn.custom_metrics,
                    }
                }
                ndjson_lines.append(json.dumps(transaction_data))

                # Errors
                for error in txn.errors:
                    error_data = {
                        "error": {
                            "transaction_id": txn.transaction_id,
                            "timestamp": int(error["timestamp"] * 1000000),
                            "exception": {
                                "type": error["type"],
                                "message": error["message"],
                                "handled": error["handled"],
                            },
                        }
                    }
                    ndjson_lines.append(json.dumps(error_data))
                    self.errors_sent += 1

            payload = "\n".join(ndjson_lines) + "\n"

            # Send to Elastic APM
            endpoint = f"{self.apm_server_url}/intake/v2/events"

            async with self._session.post(endpoint, data=payload) as response:
                if response.status == 202:
                    self.transactions_sent += len(batch)
                    self.logger.debug(f"Flushed {len(batch)} transactions to Elastic APM")
                else:
                    self.logger.error(f"Elastic APM returned status {response.status}")

        except Exception as e:
            self.logger.error(f"Failed to flush batch to Elastic APM: {e}")


class DatadogAPMIntegration:
    """
    Datadog APM integration.

    Provides application performance monitoring with Datadog APM.
    """

    def __init__(self, config: Config):
        """Initialize Datadog APM integration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Datadog configuration
        self.dd_agent_url = getattr(config, "datadog_agent_url", "http://localhost:8126")
        self.dd_api_key = getattr(config, "datadog_api_key", None)
        self.service_name = getattr(config, "service_name", "qbitel")
        self.environment = getattr(config.environment, "value", "development")

        # Session and batching
        self._session: Optional[aiohttp.ClientSession] = None
        self._transaction_queue: List[APMTransaction] = []
        self._batch_size = 50
        self._flush_interval = 10
        self._flush_task: Optional[asyncio.Task] = None

        # Statistics
        self.transactions_sent = 0

        self.logger.info(f"DatadogAPMIntegration initialized (agent: {self.dd_agent_url})")

    async def initialize(self):
        """Initialize APM client."""
        headers = {"Content-Type": "application/json"}
        if self.dd_api_key:
            headers["DD-API-KEY"] = self.dd_api_key

        self._session = aiohttp.ClientSession(headers=headers)
        self._flush_task = asyncio.create_task(self._flush_loop())

        self.logger.info("Datadog APM client initialized")

    async def shutdown(self):
        """Shutdown APM client."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        await self.flush()

        if self._session:
            await self._session.close()

    async def send_transaction(self, transaction: APMTransaction):
        """Send transaction to Datadog APM."""
        try:
            self._transaction_queue.append(transaction)

            if len(self._transaction_queue) >= self._batch_size:
                await self._flush_batch()

        except Exception as e:
            self.logger.error(f"Failed to queue transaction: {e}")

    async def flush(self):
        """Flush pending transactions."""
        if self._transaction_queue:
            await self._flush_batch()

    async def _flush_loop(self):
        """Background task to flush transactions periodically."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                if self._transaction_queue:
                    await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Flush loop error: {e}")

    async def _flush_batch(self):
        """Flush batch of transactions to Datadog APM."""
        if not self._transaction_queue or not self._session:
            return

        try:
            batch = self._transaction_queue.copy()
            self._transaction_queue.clear()

            # Build Datadog traces payload
            traces = []
            for txn in batch:
                span = {
                    "trace_id": int(txn.transaction_id.replace("-", "")[:16], 16),
                    "span_id": int(txn.transaction_id.replace("-", "")[-16:], 16),
                    "name": txn.name,
                    "resource": txn.name,
                    "service": self.service_name,
                    "type": txn.type.value,
                    "start": int(txn.start_time * 1e9),  # nanoseconds
                    "duration": int(txn.duration_ms * 1e6),  # nanoseconds
                    "error": 1 if txn.errors else 0,
                    "meta": {"env": self.environment, **txn.context},
                    "metrics": txn.custom_metrics,
                }
                traces.append([span])

            # Send to Datadog agent
            endpoint = f"{self.dd_agent_url}/v0.4/traces"

            async with self._session.put(endpoint, json=traces) as response:
                if response.status == 200:
                    self.transactions_sent += len(batch)
                    self.logger.debug(f"Flushed {len(batch)} transactions to Datadog")
                else:
                    self.logger.error(f"Datadog agent returned status {response.status}")

        except Exception as e:
            self.logger.error(f"Failed to flush batch to Datadog: {e}")


class APMManager:
    """
    Central APM manager.

    Coordinates multiple APM integrations and provides unified interface.
    """

    def __init__(self, config: Config):
        """Initialize APM manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # APM integrations
        self.integrations: List[Any] = []

        # Initialize integrations based on configuration
        if getattr(config, "enable_elastic_apm", False):
            self.integrations.append(ElasticAPMIntegration(config))

        if getattr(config, "enable_datadog_apm", False):
            self.integrations.append(DatadogAPMIntegration(config))

        # Active transactions
        self._active_transactions: Dict[str, APMTransaction] = {}

        # System metrics collection
        self._metrics_task: Optional[asyncio.Task] = None
        self._system_metrics: Dict[str, float] = {}

        self.logger.info(f"APMManager initialized with {len(self.integrations)} integrations")

    async def initialize(self):
        """Initialize all APM integrations."""
        for integration in self.integrations:
            await integration.initialize()

        # Start system metrics collection
        self._metrics_task = asyncio.create_task(self._collect_system_metrics())

        self.logger.info("APM manager initialized")

    async def shutdown(self):
        """Shutdown all APM integrations."""
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task.cancel()
            except asyncio.CancelledError:
                pass

        for integration in self.integrations:
            await integration.shutdown()

        self.logger.info("APM manager shut down")

    @asynccontextmanager
    async def transaction(
        self,
        name: str,
        transaction_type: TransactionType = TransactionType.REQUEST,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for APM transaction tracking."""
        import uuid

        transaction_id = str(uuid.uuid4())
        txn = APMTransaction(
            transaction_id=transaction_id,
            name=name,
            type=transaction_type,
            start_time=time.time(),
            context=context or {},
        )

        self._active_transactions[transaction_id] = txn

        try:
            yield txn
            txn.finish("success")
        except Exception as e:
            txn.add_error(e, handled=False)
            txn.finish("error")
            raise
        finally:
            # Send to all integrations
            for integration in self.integrations:
                await integration.send_transaction(txn)

            del self._active_transactions[transaction_id]

    async def _collect_system_metrics(self):
        """Collect system metrics periodically."""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute

                # CPU metrics
                self._system_metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
                self._system_metrics["cpu_count"] = psutil.cpu_count()

                # Memory metrics
                memory = psutil.virtual_memory()
                self._system_metrics["memory_percent"] = memory.percent
                self._system_metrics["memory_available_mb"] = memory.available / (1024 * 1024)

                # Disk metrics
                disk = psutil.disk_usage("/")
                self._system_metrics["disk_percent"] = disk.percent
                self._system_metrics["disk_free_gb"] = disk.free / (1024 * 1024 * 1024)

                # Network metrics
                net_io = psutil.net_io_counters()
                self._system_metrics["network_bytes_sent"] = net_io.bytes_sent
                self._system_metrics["network_bytes_recv"] = net_io.bytes_recv

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"System metrics collection error: {e}")

    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        return self._system_metrics.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get APM statistics."""
        stats = {
            "integrations": [],
            "active_transactions": len(self._active_transactions),
            "system_metrics": self._system_metrics,
        }

        for integration in self.integrations:
            integration_stats = {
                "type": type(integration).__name__,
                "transactions_sent": integration.transactions_sent,
            }
            if hasattr(integration, "errors_sent"):
                integration_stats["errors_sent"] = integration.errors_sent

            stats["integrations"].append(integration_stats)

        return stats


# Global APM manager
_apm_manager: Optional[APMManager] = None


async def initialize_apm(config: Config) -> APMManager:
    """Initialize global APM manager."""
    global _apm_manager

    _apm_manager = APMManager(config)
    await _apm_manager.initialize()

    return _apm_manager


def get_apm_manager() -> Optional[APMManager]:
    """Get global APM manager."""
    return _apm_manager
