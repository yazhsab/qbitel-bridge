"""
Sandbox Manager

Manages pool of Firecracker sandboxes for efficient execution:
- Pre-warmed sandbox pool
- Automatic scaling
- Health monitoring
- Load balancing
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from ai_engine.sandbox.firecracker import (
    FirecrackerSandbox,
    SandboxConfig,
    SandboxResult,
    SandboxStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for sandbox pool."""

    # Pool sizing
    min_size: int = 2
    max_size: int = 10
    target_free: int = 3  # Target number of free sandboxes

    # Timing
    warmup_interval_seconds: int = 30
    health_check_interval_seconds: int = 60
    max_idle_seconds: int = 300
    max_age_seconds: int = 3600

    # Sandbox defaults
    sandbox_config: SandboxConfig = field(default_factory=SandboxConfig)


class SandboxPool:
    """
    Pool of pre-warmed Firecracker sandboxes.

    Provides:
    - Fast sandbox acquisition
    - Automatic warming
    - Health monitoring
    - Resource cleanup

    Example:
        pool = SandboxPool(PoolConfig(min_size=3, max_size=10))
        await pool.start()

        async with pool.acquire() as sandbox:
            result = await sandbox.execute(code="print('Hello')")

        await pool.stop()
    """

    def __init__(self, config: Optional[PoolConfig] = None):
        """Initialize pool."""
        self.config = config or PoolConfig()

        self._free_sandboxes: deque[FirecrackerSandbox] = deque()
        self._busy_sandboxes: Set[str] = set()
        self._all_sandboxes: Dict[str, FirecrackerSandbox] = {}

        self._lock = asyncio.Lock()
        self._warmup_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._running = False

        self._metrics = {
            "total_created": 0,
            "total_destroyed": 0,
            "total_executions": 0,
            "current_free": 0,
            "current_busy": 0,
        }

    @property
    def size(self) -> int:
        """Get current pool size."""
        return len(self._all_sandboxes)

    @property
    def free_count(self) -> int:
        """Get number of free sandboxes."""
        return len(self._free_sandboxes)

    @property
    def busy_count(self) -> int:
        """Get number of busy sandboxes."""
        return len(self._busy_sandboxes)

    async def start(self) -> None:
        """Start the sandbox pool."""
        if self._running:
            return

        self._running = True
        logger.info("Starting sandbox pool")

        # Initial warmup
        await self._warmup(self.config.min_size)

        # Start background tasks
        self._warmup_task = asyncio.create_task(self._warmup_loop())
        self._health_task = asyncio.create_task(self._health_check_loop())

        logger.info(f"Sandbox pool started with {self.size} sandboxes")

    async def stop(self) -> None:
        """Stop the sandbox pool."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping sandbox pool")

        # Cancel background tasks
        if self._warmup_task:
            self._warmup_task.cancel()
            try:
                await self._warmup_task
            except asyncio.CancelledError:
                pass

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Stop all sandboxes
        for sandbox in list(self._all_sandboxes.values()):
            try:
                await sandbox.stop()
            except Exception as e:
                logger.error(f"Error stopping sandbox: {e}")

        self._all_sandboxes.clear()
        self._free_sandboxes.clear()
        self._busy_sandboxes.clear()

        logger.info("Sandbox pool stopped")

    async def acquire(self, timeout: float = 30.0) -> FirecrackerSandbox:
        """
        Acquire a sandbox from the pool.

        Args:
            timeout: Maximum time to wait for a sandbox

        Returns:
            Available sandbox

        Raises:
            TimeoutError: If no sandbox available within timeout
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            async with self._lock:
                if self._free_sandboxes:
                    sandbox = self._free_sandboxes.popleft()
                    self._busy_sandboxes.add(sandbox.sandbox_id)
                    self._update_metrics()
                    logger.debug(f"Acquired sandbox: {sandbox.sandbox_id}")
                    return sandbox

                # Try to create new sandbox if under max
                if self.size < self.config.max_size:
                    sandbox = await self._create_sandbox()
                    self._busy_sandboxes.add(sandbox.sandbox_id)
                    self._update_metrics()
                    logger.debug(f"Created and acquired sandbox: {sandbox.sandbox_id}")
                    return sandbox

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                raise TimeoutError("No sandbox available")

            # Wait a bit and retry
            await asyncio.sleep(0.1)

    async def release(self, sandbox: FirecrackerSandbox) -> None:
        """
        Release a sandbox back to the pool.

        Args:
            sandbox: Sandbox to release
        """
        async with self._lock:
            if sandbox.sandbox_id in self._busy_sandboxes:
                self._busy_sandboxes.remove(sandbox.sandbox_id)

                # Check if sandbox is still healthy
                if sandbox.status == SandboxStatus.RUNNING:
                    self._free_sandboxes.append(sandbox)
                    logger.debug(f"Released sandbox: {sandbox.sandbox_id}")
                else:
                    # Destroy unhealthy sandbox
                    await self._destroy_sandbox(sandbox)
                    logger.debug(f"Destroyed unhealthy sandbox: {sandbox.sandbox_id}")

                self._update_metrics()

    async def execute(
        self,
        code: str,
        language: str = "python",
        **kwargs,
    ) -> SandboxResult:
        """
        Execute code using a pooled sandbox.

        Args:
            code: Code to execute
            language: Programming language
            **kwargs: Additional execution arguments

        Returns:
            SandboxResult
        """
        sandbox = await self.acquire()
        try:
            result = await sandbox.execute(code, language, **kwargs)
            self._metrics["total_executions"] += 1
            return result
        finally:
            await self.release(sandbox)

    async def _warmup(self, count: int) -> None:
        """Warm up sandboxes."""
        tasks = []
        for _ in range(count):
            if self.size >= self.config.max_size:
                break
            tasks.append(self._create_sandbox())

        if tasks:
            sandboxes = await asyncio.gather(*tasks, return_exceptions=True)
            for sandbox in sandboxes:
                if isinstance(sandbox, FirecrackerSandbox):
                    async with self._lock:
                        self._free_sandboxes.append(sandbox)
                else:
                    logger.error(f"Failed to create sandbox: {sandbox}")

            self._update_metrics()

    async def _warmup_loop(self) -> None:
        """Background warmup task."""
        while self._running:
            try:
                # Calculate how many sandboxes to warm up
                needed = self.config.target_free - self.free_count
                if needed > 0 and self.size < self.config.max_size:
                    await self._warmup(min(needed, self.config.max_size - self.size))

                await asyncio.sleep(self.config.warmup_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Warmup error: {e}")
                await asyncio.sleep(5)

    async def _health_check_loop(self) -> None:
        """Background health check task."""
        while self._running:
            try:
                await self._check_health()
                await asyncio.sleep(self.config.health_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)

    async def _check_health(self) -> None:
        """Check health of all sandboxes."""
        async with self._lock:
            unhealthy = []

            for sandbox_id, sandbox in list(self._all_sandboxes.items()):
                # Check status
                if sandbox.status != SandboxStatus.RUNNING:
                    unhealthy.append(sandbox)
                    continue

                # Check age
                if sandbox._start_time:
                    age = (datetime.utcnow() - sandbox._start_time).total_seconds()
                    if age > self.config.max_age_seconds:
                        if sandbox_id not in self._busy_sandboxes:
                            unhealthy.append(sandbox)

            # Destroy unhealthy sandboxes
            for sandbox in unhealthy:
                if sandbox.sandbox_id in self._busy_sandboxes:
                    continue
                await self._destroy_sandbox(sandbox)
                self._free_sandboxes = deque(s for s in self._free_sandboxes if s.sandbox_id != sandbox.sandbox_id)

            self._update_metrics()

    async def _create_sandbox(self) -> FirecrackerSandbox:
        """Create a new sandbox."""
        sandbox = FirecrackerSandbox(self.config.sandbox_config)
        await sandbox.start()

        self._all_sandboxes[sandbox.sandbox_id] = sandbox
        self._metrics["total_created"] += 1

        logger.debug(f"Created sandbox: {sandbox.sandbox_id}")
        return sandbox

    async def _destroy_sandbox(self, sandbox: FirecrackerSandbox) -> None:
        """Destroy a sandbox."""
        try:
            await sandbox.stop()
        except Exception as e:
            logger.error(f"Error stopping sandbox: {e}")

        if sandbox.sandbox_id in self._all_sandboxes:
            del self._all_sandboxes[sandbox.sandbox_id]

        self._metrics["total_destroyed"] += 1
        logger.debug(f"Destroyed sandbox: {sandbox.sandbox_id}")

    def _update_metrics(self) -> None:
        """Update pool metrics."""
        self._metrics["current_free"] = len(self._free_sandboxes)
        self._metrics["current_busy"] = len(self._busy_sandboxes)

    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics."""
        return {
            **self._metrics,
            "pool_size": self.size,
            "utilization": self.busy_count / max(self.size, 1),
        }


class SandboxManager:
    """
    High-level sandbox manager.

    Provides:
    - Multiple sandbox pools
    - Execution scheduling
    - Resource allocation
    - Monitoring

    Example:
        manager = SandboxManager()
        await manager.start()

        result = await manager.execute(
            code="print('Hello')",
            language="python",
            priority="high"
        )
    """

    def __init__(
        self,
        default_pool_config: Optional[PoolConfig] = None,
    ):
        """Initialize manager."""
        self.default_pool_config = default_pool_config or PoolConfig()

        self._pools: Dict[str, SandboxPool] = {}
        self._running = False

        # Execution queue
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []

    async def start(
        self,
        num_workers: int = 4,
    ) -> None:
        """Start the sandbox manager."""
        if self._running:
            return

        self._running = True

        # Create default pool
        await self.create_pool("default", self.default_pool_config)

        # Start workers
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._workers.append(worker)

        logger.info(f"Sandbox manager started with {num_workers} workers")

    async def stop(self) -> None:
        """Stop the sandbox manager."""
        if not self._running:
            return

        self._running = False

        # Stop workers
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        # Stop pools
        for pool in self._pools.values():
            await pool.stop()

        self._pools.clear()
        logger.info("Sandbox manager stopped")

    async def create_pool(
        self,
        name: str,
        config: PoolConfig,
    ) -> SandboxPool:
        """Create a new sandbox pool."""
        if name in self._pools:
            raise ValueError(f"Pool already exists: {name}")

        pool = SandboxPool(config)
        await pool.start()

        self._pools[name] = pool
        logger.info(f"Created pool: {name}")

        return pool

    async def execute(
        self,
        code: str,
        language: str = "python",
        pool_name: str = "default",
        priority: str = "normal",
        **kwargs,
    ) -> SandboxResult:
        """
        Execute code in a sandbox.

        Args:
            code: Code to execute
            language: Programming language
            pool_name: Pool to use
            priority: Execution priority
            **kwargs: Additional arguments

        Returns:
            SandboxResult
        """
        pool = self._pools.get(pool_name)
        if not pool:
            raise ValueError(f"Pool not found: {pool_name}")

        return await pool.execute(code, language, **kwargs)

    async def submit(
        self,
        code: str,
        language: str = "python",
        callback: Optional[Callable[[SandboxResult], None]] = None,
        **kwargs,
    ) -> str:
        """
        Submit code for async execution.

        Args:
            code: Code to execute
            language: Programming language
            callback: Callback for result
            **kwargs: Additional arguments

        Returns:
            Execution ID
        """
        exec_id = str(uuid.uuid4())

        await self._queue.put(
            {
                "id": exec_id,
                "code": code,
                "language": language,
                "callback": callback,
                **kwargs,
            }
        )

        return exec_id

    async def _worker_loop(self, worker_id: str) -> None:
        """Worker loop for processing executions."""
        while self._running:
            try:
                task = await self._queue.get()

                result = await self.execute(
                    code=task["code"],
                    language=task["language"],
                    pool_name=task.get("pool_name", "default"),
                )

                if task.get("callback"):
                    try:
                        task["callback"](result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get manager metrics."""
        return {
            "pools": {name: pool.get_metrics() for name, pool in self._pools.items()},
            "queue_size": self._queue.qsize(),
            "worker_count": len(self._workers),
        }
